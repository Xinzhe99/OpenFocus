"""
图像栈加载器模块
负责从文件夹中加载图像栈并生成缩略图
"""

import os
import math
import cv2
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

try:
    from PIL import Image as PILImage, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

try:
    import rawpy  # optional: camera RAW via LibRaw
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False

from utils.image_utils import (SourceExifTooLarge, _sanitize_exif, export_exif,
                               read_image_exif, to_display_uint8)
from utils.sorting import natural_sort_key

# LibRaw recognizes file contents, not extensions. This shared allow-list
# controls folder, drag-and-drop and CLI discovery; camera and compression
# support still depends on the installed LibRaw version.
RAW_FORMATS = {'.3fr', '.arw', '.bay', '.cap', '.cr2', '.cr3', '.crw',
               '.dc2', '.dcr', '.dng', '.erf', '.fff', '.iiq', '.k25',
               '.kc2', '.kdc', '.mdc', '.mef', '.mos', '.mrw', '.nef',
               '.nrw', '.orf', '.pef', '.raf', '.raw', '.rdc', '.rw2',
               '.rwl', '.srf', '.sr2', '.srw', '.sti', '.x3f'}

from core.frame_storage import store_frame


def _is_dng_tiff(path):
    """DNG may have a TIFF extension and a thumbnail as its first IFD."""
    if os.path.splitext(path)[1].lower() not in ('.tif', '.tiff'):
        return False
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            return bool(tif.is_dng)
    except (OSError, ValueError, tifffile.TiffFileError):
        return False

def is_raw_source(path):
    """Recognize camera RAW extensions and TIFF containers marked as DNG."""
    return (os.path.splitext(os.fspath(path))[1].lower() in RAW_FORMATS
            or _is_dng_tiff(path))

def tiff_source_pages(tif):
    """Yield physical IFD indices and full-resolution image pages."""
    return ((index, page) for index, page in enumerate(tif.pages)
            if not getattr(page, 'is_reduced', False))


def source_dimensions(path, page=None):
    """Read a still image or TIFF page's dimensions before allocating pixels."""
    extension = os.path.splitext(os.fspath(path))[1].lower()
    if is_raw_source(path):
        if not RAWPY_AVAILABLE:
            raise ValueError('Camera RAW support is unavailable (rawpy/LibRaw).')
        with rawpy.RawPy() as raw:
            raw.open_file(os.fspath(path))
            return raw.sizes.width, raw.sizes.height
    if extension in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            entry = (next(tiff_source_pages(tif))[1] if page is None else tif.pages[page])
            validate_tiff_page(path, entry)
            return entry.imagewidth, entry.imagelength
    if not PIL_AVAILABLE:
        raise ValueError('Image header support is unavailable (Pillow).')
    with PILImage.open(path) as image:
        validate_pil_header(path, image)
        return image.size

def admit_image_decode(path, *, user_limit=None):
    """Reserve encoded and decoded working space before reading image pixels."""
    from core.resource_policy import require_memory

    encoded_bytes = os.path.getsize(path) * 2
    require_memory(encoded_bytes, user_limit=user_limit)
    width, height = source_dimensions(path)
    require_memory(width * height * 32 + encoded_bytes, user_limit=user_limit)



def linear_raw_metadata_matches(first, second):
    """Whether two camera-linear sources share the same calibration."""
    return (first.camera_make == second.camera_make
            and first.camera_model == second.camera_model
            and first.profile_tags == second.profile_tags
            and np.allclose(first.color_matrix, second.color_matrix, rtol=1e-5, atol=1e-7)
            and np.allclose(first.as_shot_neutral, second.as_shot_neutral,
                            rtol=1e-5, atol=1e-7))


def _consistent_px_size_um(values):
    """One physical pixel size shared by every frame, or None.

    A missing frame or a disagreement means the stack cannot claim a single
    calibration: propagating the first frame's value would silently mislabel
    every figure made from the others.
    """
    if not values:
        return None
    first = values[0]
    if first is None or not math.isfinite(first) or first <= 0:
        return None
    for value in values[1:]:
        if (value is None or not math.isfinite(value)
                or not math.isclose(float(value), float(first),
                                    rel_tol=1e-6, abs_tol=1e-9)):
            return None
    return float(first)


class UnsupportedTransparency(ValueError):
    """The source cannot be processed without losing transparency."""


class UnsupportedDepth(ValueError):
    """A display source has samples outside the supported uint8/uint16 domain."""


class UnsupportedColorSpace(ValueError):
    """A source cannot be converted to RGB without guessing its color layout."""


def _validate_depth(path, dtype):
    dtype = np.dtype(dtype)
    if dtype.kind != 'u' or dtype.itemsize not in (1, 2):
        raise UnsupportedDepth(
            f"Unsupported input depth: {path} (dtype={dtype}). "
            "Convert this image to unsigned 8-bit or 16-bit RGB; samples are not scaled automatically.")


def _cmyk_profiles(path, profile):
    """Validate a CMYK source profile and its usable CMYK -> sRGB transform."""
    from io import BytesIO
    from PIL import ImageCms
    from utils.image_utils import srgb_icc_profile

    try:
        destination = ImageCms.ImageCmsProfile(BytesIO(srgb_icc_profile()))
        if not profile:
            return None, destination
        if (not isinstance(profile, bytes) or len(profile) < 128
                or profile[16:20] != b'CMYK' or profile[36:40] != b'acsp'
                or int.from_bytes(profile[:4], 'big') != len(profile)):
            raise ValueError('expected an embedded CMYK ICC profile')
        source = ImageCms.ImageCmsProfile(BytesIO(profile))
        transform = ImageCms.buildTransformFromOpenProfiles(source, destination, 'CMYK', 'RGB')
        return transform, destination
    except (OSError, ValueError, TypeError, ImageCms.PyCMSError) as exc:
        raise UnsupportedColorSpace(f"Invalid or incompatible CMYK ICC profile: {path}: {exc}") from exc


def cmyk_to_bgr(image, path, profile=b''):
    """Convert uint8 CMYK samples, using embedded ICC or Pillow's standard conversion.

    Without an ICC profile Pillow uses its documented CMYK -> RGB decoder
    conversion. The returned pixels are attributed to sRGB by _read_icc_profile.
    TIFF callers supply an in-memory CMYK image; Pillow never decodes TIFF pixels.
    """
    from PIL import ImageCms

    if image.mode != 'CMYK':
        raise UnsupportedColorSpace(f"Expected CMYK samples: {path} (mode={image.mode})")
    transform, _destination = _cmyk_profiles(path, profile)
    rgb = ImageCms.applyTransform(image, transform) if transform is not None else image.convert('RGB')
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def validate_pil_header(path, image):
    """Reject unsupported display sample types and CMYK profiles without decoding."""
    if image.mode == 'F':
        _validate_depth(path, np.float32)
    if image.mode == 'I':
        # Pillow exposes unsigned 16-bit PNG grayscale as mode I on some versions.
        with open(path, 'rb') as stream:
            header = stream.read(26)
        if not (header[:8] == b'\x89PNG\r\n\x1a\n' and len(header) == 26 and header[24] == 16):
            _validate_depth(path, np.int32)
    if image.mode == 'CMYK':
        _cmyk_profiles(path, image.info.get('icc_profile', b''))


def validate_tiff_page(path, page):
    """Validate display TIFF metadata only; return whether samples are CMYK.

    Call only after RAW/DNG dispatch: camera-linear metadata has its own domain.
    CMYK TIFF is limited to unsigned 8-bit contiguous four-ink samples with a
    tifffile decoder available for its lossless compression. No 16-bit CMYK
    conversion is attempted because Pillow/ImageCms would discard precision.
    """
    _validate_depth(path, page.dtype)
    if int(page.photometric) != 5:
        return False
    import tifffile

    def tag_value(name, default):
        tag = page.tags.get(name)
        return tag.value if tag is not None else default

    if (np.dtype(page.dtype).itemsize != 1 or page.bitspersample != 8
            or page.samplesperpixel != 4 or int(page.planarconfig) != 1
            or page.extrasamples or tag_value('InkSet', 1) != 1
            or tag_value('NumberOfInks', 4) != 4 or int(page.predictor) not in (1, 2)):
        raise UnsupportedColorSpace(
            f"Unsupported CMYK TIFF layout: {path} (dtype={page.dtype}, "
            f"bits={page.bitspersample}, samples={page.samplesperpixel}, "
            f"planar={page.planarconfig}). Convert to 8-bit or 16-bit RGB TIFF.")
    compression = int(page.compression)
    try:
        if compression not in (1, 5, 8, 32946, 32773):
            raise ValueError('only supported lossless CMYK codecs are accepted')
        tifffile.TIFF.DECOMPRESSORS[compression]
    except (KeyError, ImportError, ValueError) as exc:
        raise UnsupportedColorSpace(
            f"Unsupported CMYK TIFF compression: {path} ({page.compression}). "
            "Convert to uncompressed RGB TIFF or install the required TIFF codec.") from exc
    _cmyk_profiles(path, tag_value('InterColorProfile', b''))
    return True


def decode_tiff_page(path, page_index=None, *, preserve_grayscale=False):
    """Decode a validated display TIFF page, applying its orientation exactly once."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        if page_index is None:
            page_index, page = next(tiff_source_pages(tif))
        else:
            page = tif.pages[page_index]
        cmyk = validate_tiff_page(path, page)
        if cmyk:
            samples = page.asarray()
            _validate_depth(path, samples.dtype)
            if samples.dtype != np.uint8 or samples.shape != (page.imagelength, page.imagewidth, 4):
                raise UnsupportedColorSpace(f"Unexpected CMYK TIFF decoded layout: {path}: {samples.shape}/{samples.dtype}")
            profile_tag = page.tags.get('InterColorProfile')
            image = cmyk_to_bgr(PILImage.fromarray(samples, 'CMYK'), path,
                                profile_tag.value if profile_tag is not None else b'')
            orientation_tag = page.tags.get('Orientation')
            transform = _EXIF_TRANSFORMS.get(int(orientation_tag.value) if orientation_tag is not None else 1)
            return np.ascontiguousarray(transform(image)) if transform is not None else image
    data = np.memmap(path, dtype=np.uint8, mode='r')
    success, decoded = cv2.imdecodemulti(data, flags=cv2.IMREAD_UNCHANGED,
                                        range=(page_index, page_index + 1))
    if not success or len(decoded) != 1:
        raise ValueError(f"Could not decode TIFF page {page_index + 1}: {path}")
    image = decoded[0]
    _validate_depth(path, image.dtype)
    if _has_transparency(image):
        raise UnsupportedTransparency(f"Transparent TIFF pages are unsupported: {path}")
    if image.ndim == 2:
        return image if preserve_grayscale else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image

def _has_transparency(image: np.ndarray) -> bool:
    from core.source_frames import SourceFrame
    if isinstance(image, SourceFrame):
        # Metadata checks must never decode alpha pixels on the GUI thread.
        return image.ndim == 3 and image.shape[2] == 4
    if image.ndim != 3 or image.shape[2] != 4:
        return False
    return (image.dtype not in (np.uint8, np.uint16)
            or (image.size > 0 and np.min(image[:, :, 3]) != np.iinfo(image.dtype).max))

def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Reverse the channel order of decoded RGB samples into a new BGR array.

    OpenCV swaps three 8/16-bit channels several times faster than copying a
    reversed NumPy view and returns the same pixels. Any other layout keeps
    the plain copy.
    """
    if rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype in (np.uint8, np.uint16):
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _raw_half_size(raw, scale_factor: float) -> bool:
    """Whether LibRaw can combine each Bayer block itself for this load.

    At half scale or less every 2x2 sensor block ends up inside one output
    pixel, so interpolating the missing colours at full resolution first only
    costs time. Other mosaics and non-mosaic RAW keep the full decode, as does
    the camera-linear loader.
    """
    if not 0 < scale_factor <= 0.5:
        return False
    if (raw.raw_type != rawpy.RawType.Flat or raw.num_colors != 3
            or np.shape(raw.raw_pattern) != (2, 2)):
        return False
    # The block on an odd edge is incomplete and decodes with a colour
    # missing, which would tint the last row or column.
    return raw.sizes.width % 2 == 0 and raw.sizes.height % 2 == 0


def _read_raw(path: str, scale_factor: float = 1.0) -> Optional[np.ndarray]:
    """Camera RAW -> 16-bit BGR (sRGB color space, camera white balance).

    Auto-brightening is off on purpose: LibRaw's per-image auto exposure
    would brighten each frame of a focus stack differently (the defocused
    ones most), which reads as flicker and corrupts the focus measures.
    Orientation is applied by LibRaw itself.
    """
    if not RAWPY_AVAILABLE:
        return None
    from core.resource_policy import ResourceLimitError, raw_sensor_shape, raw_decode_bytes, require_memory
    try:
        # Opening a pathname lets LibRaw read directly; rawpy.imread(file)
        # first copies the entire encoded RAW into a Python bytes object.
        with rawpy.RawPy() as raw:
            raw.open_file(os.fspath(path))
            sizes = raw.sizes
            sensor_height, sensor_width = raw_sensor_shape(sizes)
            width, height = sizes.width, sizes.height
            aspect = float(sizes.pixel_aspect)
            if aspect < 1:
                height = round(height / aspect)
            elif aspect > 1:
                width = round(width * aspect)
            if sizes.flip in (4, 5, 6, 7):
                width, height = height, width
            output_scale = scale_factor if 0 < scale_factor < 1 else 1
            output = max(1, int(width * output_scale)) * max(1, int(height * output_scale)) * 6
            require_memory(raw_decode_bytes(sensor_width, sensor_height, output, os.path.getsize(path)))
            raw.unpack()
            half_size = _raw_half_size(raw, scale_factor)
            rgb = raw.postprocess(use_camera_wb=True, output_bps=16,
                                  no_auto_bright=True, gamma=(2.4, 12.92),
                                  half_size=half_size)
        # Resize while still RGB to avoid a second full-size channel-swapped
        # copy. INTER_AREA treats each channel independently, so the result
        # is identical to resizing BGR after conversion.
        if not half_size:
            rgb = _downscale_page(rgb, scale_factor)
        else:
            # The size a scaled full decode has, so that a frame keeps its
            # dimensions whichever route decoded it.
            target = max(1, int(width * scale_factor)), max(1, int(height * scale_factor))
            if (rgb.shape[1], rgb.shape[0]) != target:
                rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)
        return rgb_to_bgr(rgb)
    except ResourceLimitError:
        raise
    except Exception:
        return None


def decode_image(path: str, *, allow_transparent_png: bool = False,
                 preserve_grayscale: bool = False, scale_factor: float = 1.0) -> Optional[np.ndarray]:
    """Decode an oriented still image, rejecting unsupported transparency.

    ``scale_factor`` applies to RAW decoding only; other callers resize after
    orientation and transparency handling.
    """
    ext = os.path.splitext(path)[1].lower()
    if is_raw_source(path):
        return _read_raw(path, scale_factor)
    if ext in ('.tif', '.tiff'):
        return decode_tiff_page(path, preserve_grayscale=preserve_grayscale)
    if PIL_AVAILABLE:
        with PILImage.open(path) as header:
            validate_pil_header(path, header)
            if header.mode == 'CMYK':
                image = cmyk_to_bgr(header, path, header.info.get('icc_profile', b''))
                return apply_exif_orientation(path, image)
    # IMREAD_IGNORE_ORIENTATION is required: modern OpenCV already rotates by
    # the EXIF Orientation tag itself, so letting apply_exif_orientation run
    # afterwards applied every transform twice (portrait phone shots loaded
    # sideways). The PIL fallback below does not rotate on its own, so the
    # single explicit pass is the only orientation step for both paths.
    data = np.fromfile(path, dtype=np.uint8)
    unchanged = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if unchanged is not None:
        _validate_depth(path, unchanged.dtype)
    if unchanged is not None and _has_transparency(unchanged):
        if allow_transparent_png and ext == '.png':
            return apply_exif_orientation(path, unchanged)
        raise UnsupportedTransparency(
            f"Transparent input is unsupported here: {path}. "
            "Matching rectangular PNG transparency is supported in the desktop and CLI workflows; see the manual.")
    # Preserve unsigned sample precision; unsupported depths never reach a color conversion.
    img = None
    if (unchanged is not None and unchanged.dtype in (np.uint8, np.uint16)
            and ext in ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.bmp', '.webp', '.tif', '.tiff')):
        if unchanged.ndim == 2:
            img = unchanged if preserve_grayscale else cv2.cvtColor(unchanged, cv2.COLOR_GRAY2BGR)
        elif unchanged.ndim == 3 and unchanged.shape[2] == 3:
            img = unchanged
        elif unchanged.ndim == 3 and unchanged.shape[2] == 4:
            img = cv2.cvtColor(unchanged, cv2.COLOR_BGRA2BGR)
    if img is None:
        img = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
                           | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        # OpenCV ships no HEIC/HEIF decoder: fall back to Pillow
        img = _read_via_pil(path)
    if img is not None:
        _validate_depth(path, img.dtype)
        img = apply_exif_orientation(path, img)
    return img


# TIFF-structured RAW variants: byte-order mark + magic other than 42.
_RAW_TIFF_MAGICS = {b"II*\x00", b"MM\x00*", b"IIRO", b"IIRS", b"MMOR", b"IIU\x00"}


class _RawExifReader:
    """Seek within a RAW while bounding the total metadata read by Pillow."""

    def __init__(self, stream):
        self.stream = stream
        self.remaining = 16 * 1024 * 1024

    def read(self, size=-1):
        if size < 0 or size > self.remaining:
            raise OSError("RAW EXIF metadata exceeds the read limit")
        data = self.stream.read(size)
        self.remaining -= len(data)
        return data

    def seek(self, *args):
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()


def _read_raw_exif_bytes(path: str) -> bytes:
    """Read export-safe EXIF from a TIFF-based RAW without reading its pixels."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic not in _RAW_TIFF_MAGICS:
            return b""
        offset = fh.read(4)
        if len(offset) != 4:
            return b""
        try:
            # Olympus and Panasonic replace the TIFF magic but retain the
            # standard byte order and IFD offsets. Supplying the offset lets
            # Pillow use a standard header without copying or changing the file.
            src = PILImage.Exif()
            src.endian = "<" if magic[:2] == b"II" else ">"
            src.load_from_fp(_RawExifReader(fh), int.from_bytes(
                offset, "little" if src.endian == "<" else "big"))
            exif = _sanitize_exif(src)
            return exif.tobytes() if exif is not None else b""
        except Exception:
            return b""


def _exif_bytes_from_image(image) -> bytes:
    """Read EXIF from the active image page without loading pixels."""
    try:
        raw = image.info.get('exif', b'') or b''
        if raw and image.format != 'TIFF':
            return raw
        exif = read_image_exif(image)
        if not len(exif):
            return b''
        safe = export_exif(exif.tobytes())
        return safe.tobytes() if safe is not None else b''
    except Exception:
        return b''


def _read_exif_bytes(path: str, page: Optional[int] = None) -> bytes:
    """Read export-safe metadata from one physical page."""
    if not PIL_AVAILABLE:
        return b''
    try:
        ext = os.path.splitext(path)[1].lower()
        if is_raw_source(path):
            return _read_raw_exif_bytes(path)
        if ext not in ('.jpg', '.jpeg', '.jpe', '.jfif', '.tif', '.tiff', '.png', '.webp'):
            return b''
        with PILImage.open(path) as image:
            if page is not None and image.format == 'TIFF':
                image.seek(page)
            return _exif_bytes_from_image(image)
    except Exception:
        return b''


_EXIF_SNAPSHOT_LIMIT = 16 * 1024 * 1024


def _read_safe_exif(path: str, page: Optional[int] = None) -> bytes:
    """Read and sanitize a bounded EXIF snapshot for one source page."""
    raw = _read_exif_bytes(path) if page is None else _read_exif_bytes(path, page)
    return _safe_exif_bytes(path, raw)


def _safe_exif_bytes(path, raw):
    raw = raw or b''
    if not raw:
        return b""
    try:
        cleaned = export_exif(raw)
        data = cleaned.tobytes() if cleaned is not None else b""
    except Exception:
        # A malformed block is retained for the writer to reject explicitly;
        # do not turn it into an empty metadata result.
        data = raw
    if len(data) > _EXIF_SNAPSHOT_LIMIT:
        raise SourceExifTooLarge(path, len(data), _EXIF_SNAPSHOT_LIMIT)
    return data


def _icc_profile_from_image(path, image):
    """Validate the RGB profile on the currently selected image page."""
    from io import BytesIO
    try:
        from PIL import ImageCms
    except ImportError:
        return b''
    try:
        profile = (image.tag_v2.get(34675, b'') if image.format == 'TIFF'
                   else image.info.get('icc_profile', b''))
        if image.mode == 'CMYK':
            from utils.image_utils import srgb_icc_profile
            _cmyk_profiles(path, profile)
            return srgb_icc_profile()
        if not (isinstance(profile, bytes) and len(profile) >= 128
                and profile[16:20] == b'RGB ' and profile[36:40] == b'acsp'
                and int.from_bytes(profile[:4], 'big') == len(profile)):
            return b''
        ImageCms.ImageCmsProfile(BytesIO(profile))
        return profile
    except UnsupportedColorSpace:
        raise
    except (OSError, ValueError, TypeError, EOFError, ImageCms.PyCMSError):
        return b''


def _read_icc_profile(path: str, page: Optional[int] = None) -> bytes:
    """Read one physical page, or the common profile of a TIFF container."""
    if not PIL_AVAILABLE:
        return b''
    if is_raw_source(path):
        from utils.image_utils import srgb_icc_profile
        return srgb_icc_profile()
    try:
        with PILImage.open(path) as image:
            if image.format == 'TIFF' and page is None:
                import tifffile
                with tifffile.TiffFile(path) as tif:
                    pages = tuple(index for index, _ in tiff_source_pages(tif))
            else:
                pages = (page if page is not None else 0,)
            profiles = []
            for index in pages:
                image.seek(index)
                profiles.append(_icc_profile_from_image(path, image))
            return common_profile_values(profiles)
    except UnsupportedColorSpace:
        raise
    except (OSError, ValueError, TypeError, EOFError):
        return b''


def _frame_metadata(path, frames, should_cancel):
    """Snapshot active TIFF pages in one forward traversal."""
    if not PIL_AVAILABLE or is_raw_source(path) or os.path.splitext(path)[1].lower() not in ('.tif', '.tiff'):
        exif = _read_safe_exif(path)
        profile = _read_icc_profile(path)
        return [(exif, profile) for _ in frames]
    result = []
    try:
        with PILImage.open(path) as image:
            for frame in frames:
                if should_cancel():
                    raise _DecodeCancelled()
                try:
                    image.seek(frame.page)
                    raw = _exif_bytes_from_image(image)
                    profile = _icc_profile_from_image(path, image)
                except UnsupportedColorSpace:
                    raise
                except (OSError, ValueError, TypeError, EOFError):
                    raw, profile = b'', b''
                result.append((_safe_exif_bytes(path, raw), profile))
    except SourceExifTooLarge:
        raise
    except (OSError, EOFError):
        return [(b'', b'') for _ in frames]
    return result


def common_profile_values(profiles) -> bytes:
    """Return a profile only if every supplied frame has the same profile."""
    common = None
    for profile in profiles:
        if not profile or (common is not None and profile != common):
            return b""
        common = profile
    return common or b""


def common_icc_profile(paths) -> bytes:
    """Return the common RGB profile only when every source has the same one."""
    return common_profile_values(_read_icc_profile(path) for path in dict.fromkeys(paths))


def stack_icc_profile(window, indices=None) -> bytes:
    """Get cached provenance for the selected frames without reading files."""
    profiles = getattr(window, "image_icc_profiles", ()) or ()
    count = len(getattr(window, "raw_images", ()) or ())
    if len(profiles) != count:
        return b""
    if indices is None:
        indices = range(count)
    selected = list(indices)
    if any(i < 0 or i >= count for i in selected):
        return b""
    return common_profile_values(profiles[i] for i in selected)


# EXIF Orientation tag values that need a pixel transform (1 = already OK).
_EXIF_TRANSFORMS = {
    2: lambda a: a[:, ::-1],                       # mirror horizontal
    3: lambda a: np.rot90(a, 2),                   # rotate 180
    4: lambda a: a[::-1, :],                       # mirror vertical
    5: lambda a: np.rot90(a[:, ::-1], 1),          # transpose
    6: lambda a: np.rot90(a, -1),                  # rotate 90 CW
    7: lambda a: np.rot90(a[::-1, :], 1),          # anti-transpose
    8: lambda a: np.rot90(a, 1),                   # rotate 90 CCW
}


def apply_exif_orientation(path: str, img: np.ndarray) -> np.ndarray:
    """Rotate/flip a decoded image according to its EXIF Orientation tag.

    Cameras store sensor-native orientation and record how to display it in
    EXIF; OpenCV ignores that tag, so portrait shots load sideways without
    this correction. Returns the image unchanged when Pillow is missing or
    the tag is absent/neutral.
    """
    if not PIL_AVAILABLE:
        return img
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.jpe', '.jfif', '.tif', '.tiff', '.png', '.webp'):
            return img
        with PILImage.open(path) as pil_img:
            orientation = read_image_exif(pil_img).get(274, 1)
        transform = _EXIF_TRANSFORMS.get(int(orientation))
        if transform is not None:
            return np.ascontiguousarray(transform(img))
    except Exception as e:
        print(f"EXIF orientation check failed for {path}: {e}")
    return img


def _read_via_pil(path: str) -> Optional[np.ndarray]:
    """Decode a file OpenCV cannot (HEIC/HEIF) through Pillow.

    Returns a 3-channel BGR array like cv2.imdecode(IMREAD_COLOR), or None when
    Pillow is missing or the file is unreadable, so callers count it as a
    failed file instead of raising.
    """
    if not PIL_AVAILABLE:
        return None
    # Pillow's TIFF decoder ABORTS the process (0xC0000409) on compressed
    # 16-bit TIFFs — LZW, PackBits and colour LZW all reproduce it on Pillow
    # 11.1.0/Windows, and `cv2.imwrite` writes LZW by default. A try/except
    # cannot catch a native abort, so a stack of otherwise-readable frames
    # would take the whole app down. OpenCV decodes every one of those files
    # correctly, so TIFF simply never goes through Pillow.
    if os.path.splitext(path)[1].lower() in ('.tif', '.tiff'):
        return None
    try:
        with PILImage.open(path) as pil_img:
            validate_pil_header(path, pil_img)
            if pil_img.mode == 'CMYK':
                return cmyk_to_bgr(pil_img, path, pil_img.info.get('icc_profile', b''))
            if pil_img.mode in ('I', 'I;16', 'I;16B'):
                arr = np.array(pil_img).astype(np.uint16)
            elif 'A' in pil_img.getbands() or 'transparency' in pil_img.info:
                rgba = np.array(pil_img.convert('RGBA'))
                if np.any(rgba[:, :, 3] != 255):
                    raise UnsupportedTransparency(
                        f"Transparent input is unsupported here: {path}")
                arr = rgba[:, :, :3]
            else:
                arr = np.array(pil_img.convert('RGB'))
        if arr.ndim == 2:
            return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except (UnsupportedTransparency, UnsupportedDepth, UnsupportedColorSpace):
        raise
    except Exception as e:
        print(f"Unsupported image for PIL fallback {path}: {e}")
        return None


def _read_tiff_pages(path: str, scale_factor=1.0, should_cancel=None) -> Optional[List[np.ndarray]]:
    """Decode multi-page TIFF one page at a time into disk-backed frames."""
    if os.path.splitext(path)[1].lower() not in ('.tif', '.tiff') or _is_dng_tiff(path):
        return None
    import tifffile
    with tifffile.TiffFile(path) as tif:
        indices = [index for index, _ in tiff_source_pages(tif)]
    if len(indices) < 2:
        return None
    pages = []
    for index in indices:
        if should_cancel is not None and should_cancel():
            raise _DecodeCancelled()
        page = decode_tiff_page(path, index)
        pages.append(store_frame(_downscale_page(page, scale_factor)))
    return pages


def count_frames(path: str) -> int:
    """Frames a path contributes to a stack, without decoding pixels.

    TIFF reduced-image directories do not contribute source frames.
    """
    if os.path.splitext(path)[1].lower() in (".tif", ".tiff"):
        if _is_dng_tiff(path):
            return 1
        try:
            import tifffile
            with tifffile.TiffFile(path) as tif:
                return sum(1 for _ in tiff_source_pages(tif))
        except Exception:
            pass
    return 1


def _collapse_repeats(paths: list) -> list:
    """Drop consecutive duplicates from a source list.

    Every frame of a multi-page TIFF records the *same* container path, and
    each entry is expanded into all of its pages on load — so replaying such
    a list turned an N-page stack into N² frames (projects and crash
    recovery both do exactly that).
    """
    out = []
    for path in paths:
        if out and os.path.normcase(os.path.abspath(out[-1])) == os.path.normcase(
                os.path.abspath(path)):
            continue
        out.append(path)
    return out


# Parallel decoding is limited to plain single-frame formats whose decoder
# releases the GIL. TIFF containers, RAW and HEIF stay serial: their decoders
# own the whole container, and running them beside another large decode only
# doubles the peak without a measured gain.
_PARALLEL_DECODE_FORMATS = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp',
                            '.webp'}
_MAX_DECODE_WORKERS = 4


@dataclass
class DecodedSource:
    """One input path's contribution, in input order."""

    path: str
    pages: List[np.ndarray] = field(default_factory=list)
    error: Optional[BaseException] = None


def _source_is_parallel(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _PARALLEL_DECODE_FORMATS


def _decode_one_source(path: str, scale_factor: float,
                       should_cancel=None, allow_transparent_png=False) -> DecodedSource:
    """Decode one source exactly as the serial loaders do (pages included)."""
    try:
        if should_cancel is not None and should_cancel():
            raise _DecodeCancelled()
        pages = _read_tiff_pages(path, scale_factor, should_cancel)
        if pages is not None:
            return DecodedSource(path, pages)
        if is_raw_source(path):
            image = _read_raw(path, scale_factor)
            if image is None:
                return DecodedSource(path, error=ValueError("decode failed"))
            return DecodedSource(path, [image])
        image = (decode_image(path, allow_transparent_png=True)
                 if allow_transparent_png else decode_image(path))
        if image is not None and _has_transparency(image) and scale_factor != 1.0:
            raise UnsupportedTransparency("Transparent images cannot be downsampled during loading")
        if image is None:
            return DecodedSource(path, error=ValueError("decode failed"))
        return DecodedSource(path, [_downscale_page(image, scale_factor)])
    except _DecodeCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - reported per source
        return DecodedSource(path, error=exc)


class _DecodeCancelled(Exception):
    """Internal marker: a cancelled decode is not a failed file."""


def _downscale_page(image: np.ndarray, scale_factor: float) -> np.ndarray:
    if scale_factor != 1.0 and 0 < scale_factor < 1.0:
        width = max(1, int(image.shape[1] * scale_factor))
        height = max(1, int(image.shape[0] * scale_factor))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _peek_source_size(path: str) -> Optional[Tuple[int, int, int, int]]:
    """(width, height, channels, itemsize) from the header, or None.

    Used only to size the decode budget; a header we cannot read reliably
    means "do not parallelise this source", never a guessed allocation.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in _PARALLEL_DECODE_FORMATS or not PIL_AVAILABLE:
        return None
    try:
        with PILImage.open(path) as image:
            width, height = image.size
            bands = len(image.getbands())
            itemsize = 1 if image.mode in ("L", "RGB", "P", "CMYK") else 2
        return int(width), int(height), max(3, min(4, bands)), itemsize
    except Exception:
        return None


def iter_decoded_sources(sources, scale_factor: float = 1.0, *,
                         thread_count: int = 0,
                         resource_preference: str = "balanced",
                         should_cancel=None, allow_transparent_png=False,
                         disk_backed=False):
    """Yield DecodedSource objects in input order, with bounded parallelism.

    Plain single-frame formats decode in a window of at most the admitted
    worker count; every other source is decoded alone, after the window is
    drained. Results are consumed in input order, so names, paths and TIFF
    page identities line up with the serial loaders.

    The caller decides what a failed source means (GUI counts it, the CLI
    aborts); this iterator only reports it in place.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor, wait

    sources = list(sources)
    if not sources:
        return

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise _DecodeCancelled()

    workers = 1
    if sum(1 for p in sources if _source_is_parallel(p)) > 1:
        try:
            requested = max(0, int(thread_count or 0))
        except (TypeError, ValueError):
            requested = 0
        from core import resource_policy as policy
        from core.resource_policy import ResourceLimitError
        peeked = [s for s in (_peek_source_size(p) for p in sources)
                  if s is not None]
        if peeked:
            # Final resident pixels plus one decoded frame per worker.
            width = max(s[0] for s in peeked)
            height = max(s[1] for s in peeked)
            channels = max(s[2] for s in peeked)
            itemsize = max(s[3] for s in peeked)
            fixed = 0 if disk_backed else sum(w * h * c * i for w, h, c, i in peeked)
            per_worker = width * height * channels * itemsize * 3
            plan = policy.plan_execution(
                policy.Workload(
                    algorithm="decode", frames=len(peeked),
                    height=height, width=width, channels=channels,
                    itemsize=itemsize, fixed_bytes=fixed,
                    per_worker_bytes=per_worker,
                    outer_tasks=min(_MAX_DECODE_WORKERS, len(peeked))),
                preference=resource_preference,
                user_threads=requested)
            workers = max(1, min(_MAX_DECODE_WORKERS, plan.workers,
                                 len(peeked)))
        elif requested != 1 and sources and not any(
                _source_is_parallel(p) for p in sources):
            workers = 1

    check_cancel()
    if workers == 1:
        for path in sources:
            check_cancel()
            yield _decode_one_source(path, scale_factor, should_cancel, allow_transparent_png)
        return

    pool = ThreadPoolExecutor(max_workers=workers)
    window = deque()
    index = 0
    try:
        while index < len(sources) or window:
            check_cancel()
            while (index < len(sources) and len(window) < workers
                   and _source_is_parallel(sources[index])):
                window.append((sources[index],
                               pool.submit(_decode_one_source, sources[index],
                                           scale_factor, should_cancel, allow_transparent_png)))
                index += 1
            if not window:
                # A serial source: decode it alone, then continue.
                path = sources[index]
                yield _decode_one_source(path, scale_factor, should_cancel, allow_transparent_png)
                index += 1
                continue
            path, future = window[0]
            for pending in window:
                if pending[1].done() and pending[1].exception() is not None:
                    pending[1].result()
            if not future.done():
                wait((future,), timeout=0.05)
                continue
            window.popleft()
            result = future.result()
            del future
            check_cancel()
            yield result
            del result
    finally:
        for _path, future in window:
            future.cancel()
        window.clear()
        pool.shutdown(wait=True, cancel_futures=True)


class ImageStackLoader:
    """图像栈加载器"""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp',
                         '.tiff', '.tif', '.webp'}
    if HEIF_AVAILABLE:
        SUPPORTED_FORMATS |= {'.heic', '.heif'}
    if RAWPY_AVAILABLE:
        SUPPORTED_FORMATS |= RAW_FORMATS
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def __init__(self):
        self.image_paths = []
        self.images = []
        self.thumbnail_size = (600, 400)
        # Native µm/px auto-detected from TIFF/DNG metadata (None = unknown);
        # source_manager divides it by the load scale for working pixels.
        self.px_size_um = None
        self.source_icc_profile = b""
        self.source_icc_profiles = ()
        self.source_exif = b""
        self.source_exifs = ()
        self.should_cancel = lambda: False
        self.progress_callback = None
        self.failed_count = 0
        # `*_registered*` copies found beside their originals and skipped
        self.skipped_derived_count = 0

    def _drop_derived_aligned_copies(self, image_files):
        """Skip `*_registered*` files that duplicate an original in the folder.

        Older builds wrote aligned copies beside the sources. Those copies are
        not focus-stack frames: re-loading a folder after a batch turned 9
        frames into 36 and fused each frame three times. A file is skipped
        only when its original stem is present in the same folder, so a user
        who genuinely named a file `_registered` keeps it.
        """
        from utils.export_paths import derived_aligned_source

        stems = {os.path.splitext(name)[0].lower() for name, _ in image_files}
        kept = []
        for name, path in image_files:
            original = derived_aligned_source(name)
            if original and original.lower() in stems:
                continue
            kept.append((name, path))
        self.skipped_derived_count = len(image_files) - len(kept)
        return kept

    def load_linear_raw_stack(self, filepaths, scale_factor=1.0, *,
                              thread_count=0, resource_preference="balanced", expected_metadata=None):
        """Load a complete camera-linear stack, committing state only on success."""
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        from core.resource_policy import Workload, plan_execution
        from core.source_frames import scan_source, ensure_display_preview
        from locales import trans
        from utils.logging_setup import log_operation
        # Camera calibration is inspected while building the source manifest.
        from utils.scalebar import detect_px_size_um

        paths = list(filepaths)
        if not paths:
            return False, trans.t("linear_raw_invalid"), [], []
        images = []
        px_sizes = []
        exifs = []
        metadata = expected_metadata
        def unknown_camera(profile):
            return (not profile.camera_make or not profile.camera_model
                    or (profile.camera_make, profile.camera_model) == ("OpenFocus", "Linear RAW"))

        if expected_metadata is not None and unknown_camera(expected_metadata):
            return False, trans.t("linear_raw_unknown_camera"), [], []
        if self.should_cancel():
            return False, 'CANCELLED', [], []
        if self.progress_callback is not None:
            self.progress_callback(0, len(paths))

        # rawpy unpacks the sensor when reading camera calibration. Inspect
        # only cheap headers first so parallel admission accounts for that
        # native allocation, not just the embedded JPEG and parser buffers.
        import rawpy
        sensor_pixels = 0
        for path in paths:
            if self.should_cancel():
                return False, 'CANCELLED', [], []
            if not is_raw_source(path):
                continue
            try:
                with rawpy.RawPy() as raw:
                    raw.open_file(os.fspath(path))
                    sizes = raw.sizes
                    width = max(getattr(sizes, 'raw_width', 0), sizes.width)
                    height = max(getattr(sizes, 'raw_height', 0), sizes.height)
                    sensor_pixels = max(sensor_pixels, width * height)
            except (OSError, ValueError, rawpy.LibRawError):
                # The ordered scan reports the original file error below.
                continue
        per_worker = 64 * 1024 * 1024 + sensor_pixels * 4
        plan = plan_execution(Workload(algorithm='source_manifest', frames=len(paths),
                                       fixed_bytes=0, per_worker_bytes=per_worker,
                                       outer_tasks=len(paths)),
                              preference=resource_preference,
                              user_threads=min(4, max(1, int(thread_count or 4))))

        def inspect(path):
            if self.should_cancel():
                raise InterruptedError('Source manifest cancelled')
            if not is_raw_source(path):
                return None
            return scan_source(path, scale_factor, linear=True, header_only=True,
                               should_cancel=self.should_cancel)[0]

        with log_operation('linear_source_manifest', count=len(paths), workers=plan.workers), \
                ThreadPoolExecutor(max_workers=plan.workers) as pool:
            pending = deque()
            next_index = 0
            while next_index < min(len(paths), plan.workers):
                pending.append(pool.submit(inspect, paths[next_index]))
                next_index += 1
            while pending:
                index = len(images)
                path = paths[index]
                try:
                    image = pending.popleft().result()
                except InterruptedError:
                    return False, 'CANCELLED', [], []
                except MemoryError:
                    raise
                except Exception as exc:
                    if self.should_cancel():
                        return False, 'CANCELLED', [], []
                    return False, f"{trans.t('linear_raw_invalid')}\n{path}: {exc}", [], []
                if self.should_cancel():
                    return False, 'CANCELLED', [], []
                if image is None:
                    return False, trans.t("linear_raw_invalid"), [], []
                profile = image.linear_metadata
                if (expected_metadata is not None or len(paths) > 1) and unknown_camera(profile):
                    return False, f"{trans.t('linear_raw_unknown_camera')}\n{path}", [], []
                if metadata is not None and not linear_raw_metadata_matches(metadata, profile):
                    return False, f"{trans.t('linear_raw_mismatch')}\n{path}", [], []
                metadata = profile if metadata is None else metadata
                # Preserve native calibration; source_manager applies the load scale.
                px_sizes.append(detect_px_size_um(path))
                exifs.append(_read_safe_exif(path))
                images.append(image)
                if self.progress_callback is not None:
                    self.progress_callback(index + 1, len(paths))
                if next_index < len(paths) and not self.should_cancel():
                    pending.append(pool.submit(inspect, paths[next_index]))
                    next_index += 1
        if self.should_cancel():
            return False, 'CANCELLED', [], []
        try:
            ensure_display_preview(images[0], should_cancel=self.should_cancel)
        except InterruptedError:
            return False, 'CANCELLED', [], []
        except (OSError, ValueError) as exc:
            return False, f'Could not load source preview: {exc}', [], []
        if self.should_cancel():
            return False, 'CANCELLED', [], []
        self.images = images
        self.image_paths = paths
        self.source_exif = exifs[0] if exifs else _read_safe_exif(paths[0])
        self.source_exifs = tuple(exifs)
        self.source_icc_profile = b""
        self.source_icc_profiles = tuple(b'' for _ in images)
        self.px_size_um = _consistent_px_size_um(px_sizes)
        self.failed_count = 0
        self.linear_dng_metadata = metadata
        return True, f"Loaded {len(images)} image(s)", images, [os.path.basename(p) for p in paths]

    def load_from_folder(self, folder_path: str, scale_factor: float = 1.0, *,
                         thread_count: int = 0,
                         resource_preference: str = "balanced", exclude_paths=(),
                         ) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        self.failed_count = 0
        self.skipped_derived_count = 0
        if not os.path.isdir(folder_path):
            return False, "Selected path is not a valid directory", [], []

        from utils.path_identity import same_path
        image_files = []
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.SUPPORTED_FORMATS and not any(
                    same_path(os.path.join(folder_path, filename), excluded) for excluded in exclude_paths):
                full_path = os.path.join(folder_path, filename)
                image_files.append((filename, full_path))
        image_files = self._drop_derived_aligned_copies(image_files)

        if not image_files:
            return False, "No supported image files found in the folder", [], []

        image_files.sort(key=lambda x: natural_sort_key(x[0]))

        result = self.load_from_filepaths(
            [path for _, path in image_files], scale_factor,
            thread_count=thread_count, resource_preference=resource_preference)
        if result[0] and self.skipped_derived_count:
            return (True, result[1] + f" (skipped {self.skipped_derived_count} aligned copies)",
                    result[2], result[3])
        return result

    def load_from_video(self, video_path: str, scale_factor: float = 1.0) -> Tuple[bool, str, List[np.ndarray], List[str]]:
        if not os.path.isfile(video_path):
            return False, "Video file does not exist", [], []

        ext = os.path.splitext(video_path)[1].lower()
        if ext not in self.SUPPORTED_VIDEO_FORMATS:
            return False, f"Unsupported video format: {ext}", [], []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Failed to open video file", [], []

        loaded_images = []
        filenames = []
        frame_index = 0
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while True:
                if self.should_cancel():
                    return False, 'CANCELLED', [], []
                ret, frame = cap.read()
                if not ret:
                    break

                frame = _downscale_page(frame, scale_factor)

                loaded_images.append(store_frame(frame))
                filenames.append(f"{video_name}_frame_{frame_index:04d}.png")
                frame_index += 1
        finally:
            cap.release()

        if self.should_cancel():
            return False, 'CANCELLED', [], []

        if not loaded_images:
            return False, "No frames could be extracted from the video", [], []

        self.images = loaded_images
        self.image_paths = [video_path] * len(loaded_images)
        self.source_exif = b""
        self.source_exifs = tuple(b"" for _ in loaded_images)
        self.source_icc_profile = b""
        self.source_icc_profiles = tuple(b'' for _ in loaded_images)
        self.px_size_um = None
        self.linear_dng_metadata = None
        self.failed_count = 0

        message = f"Loaded {len(loaded_images)} frame(s) from video"
        if total_frames > 0 and len(loaded_images) < total_frames:
            message += f" (expected: {total_frames})"

        return True, message, loaded_images, filenames

    def is_video_file(self, filepath: str) -> bool:
        ext = os.path.splitext(filepath)[1].lower()
        return ext in self.SUPPORTED_VIDEO_FORMATS

    def load_from_filepaths(self, filepaths: list[str], scale_factor: float = 1.0, *,
                            thread_count: int = 0,
                            resource_preference: str = "balanced"):
        """Open headers and prepare the first display in an admitted background job."""
        from concurrent.futures import ThreadPoolExecutor
        from core.source_frames import scan_source, ensure_display_preview
        from utils.logging_setup import log_operation
        from utils.scalebar import detect_px_size_um

        sources = _collapse_repeats(list(filepaths))
        if not sources:
            return False, "No file paths provided", [], []
        images, names, paths, px_sizes, exifs, profiles = [], [], [], [], [], []
        failed = 0
        exif = b""
        workers = min(4, max(1, int(thread_count or 4)))
        def inspect(path):
            if self.should_cancel():
                raise _DecodeCancelled()
            try:
                return scan_source(path, scale_factor, header_only=True,
                                   should_cancel=self.should_cancel), None
            except SourceExifTooLarge:
                raise
            except Exception as exc:
                return [], exc
        if self.progress_callback is not None:
            self.progress_callback(0, len(sources))
        with log_operation('source_manifest', count=len(sources), preview_threads=workers):
            # Only descriptors are queued. The first preview is prepared here;
            # remaining thumbnails are generated after the workspace is usable.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                try:
                    for done, (path, result) in enumerate(zip(sources, pool.map(inspect, sources)), 1):
                        if self.should_cancel():
                            raise _DecodeCancelled()
                        frames, error = result
                        if error is not None:
                            if isinstance(error, InterruptedError):
                                return False, 'CANCELLED', [], []
                            if isinstance(error, UnsupportedTransparency):
                                from locales import trans
                                return False, trans.t('transparent_desktop_unsupported').format(path=path), [], []
                            failed += 1
                            import logging
                            logging.getLogger(__name__).warning('Source preview failed path=%s: %s', path, error)
                        else:
                            metadata = _frame_metadata(path, frames, self.should_cancel)
                            for frame_exif, frame_icc in metadata:
                                if not exif:
                                    exif = frame_exif
                                exifs.append(frame_exif)
                                profiles.append(frame_icc)
                            px_sizes.append(detect_px_size_um(path))
                            stem = os.path.splitext(os.path.basename(path))[0]
                            images.extend(frames)
                            paths.extend([path] * len(frames))
                            names.extend(f'{stem}_page_{index + 1:03d}.tif' if len(frames) > 1
                                         else os.path.basename(path) for index, frame in enumerate(frames))
                        if self.progress_callback is not None:
                            self.progress_callback(done, len(sources))
                except _DecodeCancelled:
                    return False, 'CANCELLED', [], []
        if not images:
            return False, 'Could not load any image files', [], []
        if self.should_cancel():
            return False, 'CANCELLED', [], []
        try:
            ensure_display_preview(images[0], should_cancel=self.should_cancel)
        except InterruptedError:
            return False, 'CANCELLED', [], []
        except UnsupportedTransparency:
            from locales import trans
            return False, trans.t('transparent_desktop_unsupported').format(path=images[0].path), [], []
        except (OSError, ValueError) as exc:
            return False, f'Could not load source preview: {exc}', [], []
        if self.should_cancel():
            return False, 'CANCELLED', [], []
        self.images, self.image_paths = images, paths
        self.source_exif = exif
        self.source_exifs = tuple(exifs)
        self.px_size_um = _consistent_px_size_um(px_sizes)
        self.source_icc_profiles = tuple(profiles)
        self.source_icc_profile = common_profile_values(profiles)
        self.failed_count = failed
        message = f'Loaded {len(images)} image(s)'
        if failed:
            message += f' (failed: {failed})'
        return True, message, images, names

    def create_pixmaps(self, images: List[np.ndarray], max_size: Tuple[int, int] = (800, 600)) -> list:
        pixmaps = []
        for img in images:
            pixmap = self._cv_to_pixmap(img, max_size)
            pixmaps.append(pixmap)
        return pixmaps

    def create_thumbnails(self, images: List[np.ndarray], thumb_size: int = 40) -> list:
        thumbnails = []
        for img in images:
            from core.source_frames import display_image, preview_ready
            if not preview_ready(img):
                from PyQt6.QtGui import QPixmap
                thumbnails.append(QPixmap())
                continue
            img = display_image(img)
            if img.ndim == 3 and img.shape[2] == 4:
                from utils.image_utils import composite_transparent_for_display
                img = composite_transparent_for_display(img)
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid image dimensions: {w}x{h}")
            scale = thumb_size / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pixmap = self._cv_to_pixmap(to_display_uint8(resized), (thumb_size, thumb_size))
            thumbnails.append(pixmap)
        return thumbnails

    def _cv_to_pixmap(self, cv_img: np.ndarray, max_size: Optional[Tuple[int, int]] = None):
        # Imported lazily: the headless PyPI package uses this module for
        # decoding without PyQt6 installed.
        from utils.image_utils import cv2_to_pixmap
        from core.source_frames import display_image
        cv_img = display_image(cv_img)

        if max_size is not None:
            if cv_img.ndim == 3 and cv_img.shape[2] == 4:
                from utils.image_utils import composite_transparent_for_display
                cv_img = composite_transparent_for_display(cv_img)
            h, w = cv_img.shape[:2]
            max_w, max_h = max_size

            scale = min(max_w / w, max_h / h)
            if scale < 1.0:
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return cv2_to_pixmap(cv_img)

    def get_image_info(self, index: int) -> dict:
        if not self.images or index < 0 or index >= len(self.images):
            return {}

        img = self.images[index]
        path = self.image_paths[index] if index < len(self.image_paths) else ""

        return {
            'index': index,
            'path': path,
            'filename': os.path.basename(path) if path else "",
            'shape': img.shape,
            'size': f"{img.shape[1]}x{img.shape[0]}",
            'channels': img.shape[2] if len(img.shape) > 2 else 1,
            'dtype': img.dtype
        }

    @staticmethod
    def missing_timestamps(images_with_times: list) -> int:
        """How many frames in the group have no capture time."""
        return sum(1 for item in images_with_times if item[2] is None)

    def get_image_timestamp(self, filepath: str) -> Optional[float]:
        if not PIL_AVAILABLE:
            return None

        try:
            with PILImage.open(filepath) as img:
                exif_data = read_image_exif(img)
                date_time_original = exif_data.get_ifd(0x8769).get(36867) or exif_data.get(36867)
                if isinstance(date_time_original, str) and date_time_original:
                    return datetime.strptime(date_time_original, '%Y:%m:%d %H:%M:%S').timestamp()
        except (OSError, ValueError, TypeError, AttributeError, KeyError, EOFError):
            pass

        return None

    def load_images_with_timestamps(self, folder_path: str, sort_by: str = 'filename',
                                    scale_factor: float = 1.0, *,
                                    thread_count: int = 0,
                                    resource_preference: str = "balanced", exclude_paths=()):
        """Decode through the normal loader, preserving all container pages.

        Each entry retains its real source path and a separate export/page name.
        """
        ok, message, images, filenames = self.load_from_folder(
            folder_path, scale_factor, thread_count=thread_count,
            resource_preference=resource_preference, exclude_paths=exclude_paths)
        if not ok:
            return ok, message, [], []
        timestamps = {path: self.get_image_timestamp(path) for path in dict.fromkeys(self.image_paths)}
        data = [(path, image, timestamps[path], filename)
                for path, image, filename in zip(self.image_paths, images, filenames)]
        if sort_by == 'timestamp':
            data.sort(key=lambda item: (item[2] is None, item[2] or 0, natural_sort_key(item[3])))
        else:
            data.sort(key=lambda item: natural_sort_key(item[3]))
        return True, message, data, [item[3] for item in data]

    def split_by_count(
        self,
        images_with_times: List[Tuple[str, np.ndarray, Optional[float]]],
        count_per_stack: int
    ) -> List[List[Tuple[str, np.ndarray, Optional[float]]]]:
        # The dialog's stepper returns a float even with decimals=0; range()
        # rejects it with TypeError. Normalise at this boundary rather than
        # weakening the generic numeric widget.
        try:
            count_per_stack = int(round(float(count_per_stack)))
        except (TypeError, ValueError):
            count_per_stack = 1
        if count_per_stack <= 0:
            count_per_stack = 1

        stacks = []
        for i in range(0, len(images_with_times), count_per_stack):
            stack = images_with_times[i:i + count_per_stack]
            if stack:
                stacks.append(stack)

        return stacks

    def split_by_time_threshold(
        self,
        images_with_times: list,
        threshold_seconds: float
    ) -> list:
        """Group by capture gaps; the last group keeps trailing frames.

        Frames without a timestamp cannot be placed in time, so they start a
        new group instead of being silently merged into whichever neighbour
        happened to come first. `missing_timestamps()` lets the UI say how
        many there are.
        """
        try:
            threshold_seconds = float(threshold_seconds)
        except (TypeError, ValueError):
            threshold_seconds = 1.0
        if threshold_seconds <= 0:
            threshold_seconds = 1.0

        stacks = []
        current_stack = []

        for i, item in enumerate(images_with_times):
            timestamp = item[2]
            if i == 0:
                current_stack.append(item)
                continue

            if timestamp is None:
                if current_stack:
                    stacks.append(current_stack)
                current_stack = [item]
                continue

            prev_timestamp = images_with_times[i - 1][2]
            if prev_timestamp is None:
                stacks.append(current_stack)
                current_stack = [item]
                continue

            time_diff = timestamp - prev_timestamp
            if time_diff > threshold_seconds:
                if current_stack:
                    stacks.append(current_stack)
                current_stack = [item]
            else:
                current_stack.append(item)

        if current_stack:
            stacks.append(current_stack)

        return stacks

    def get_stack_info(
        self,
        stacks: List[List[Tuple[str, np.ndarray, Optional[float]]]]
    ) -> List[Dict[str, Any]]:
        info_list = []
        for i, stack in enumerate(stacks):
            count = len(stack)
            first_img = stack[0][1]
            height, width = first_img.shape[:2]

            timestamps = [item[2] for item in stack if item[2] is not None]
            if timestamps:
                first_time = datetime.fromtimestamp(min(timestamps)).strftime('%H:%M:%S')
                last_time = datetime.fromtimestamp(max(timestamps)).strftime('%H:%M:%S')
                time_range = f"{first_time}-{last_time}"
            else:
                time_range = "Unknown"

            info_list.append({
                'stack_index': i,
                'name': f"Stack {i + 1}",
                'count': count,
                'resolution': f"{width}x{height}",
                'time_range': time_range
            })

        return info_list
