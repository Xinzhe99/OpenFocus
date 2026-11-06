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

from utils.image_utils import _sanitize_exif, export_exif, read_image_exif, to_display_uint8
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

def source_dimensions(path, page=0):
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
            entry = tif.pages[page]
            return entry.imagewidth, entry.imagelength
    if not PIL_AVAILABLE:
        raise ValueError('Image header support is unavailable (Pillow).')
    with PILImage.open(path) as image:
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


def _has_transparency(image: np.ndarray) -> bool:
    from core.source_frames import SourceFrame
    if isinstance(image, SourceFrame):
        # Metadata checks must never decode alpha pixels on the GUI thread.
        return image.ndim == 3 and image.shape[2] == 4
    if image.ndim != 3 or image.shape[2] != 4:
        return False
    return (image.dtype not in (np.uint8, np.uint16)
            or (image.size > 0 and np.min(image[:, :, 3]) != np.iinfo(image.dtype).max))

def _read_raw(path: str, scale_factor: float = 1.0) -> Optional[np.ndarray]:
    """Camera RAW -> 16-bit BGR (sRGB color space, camera white balance).

    Auto-brightening is off on purpose: LibRaw's per-image auto exposure
    would brighten each frame of a focus stack differently (the defocused
    ones most), which reads as flicker and corrupts the focus measures.
    Orientation is applied by LibRaw itself.
    """
    if not RAWPY_AVAILABLE:
        return None
    try:
        with open(path, 'rb') as stream, rawpy.imread(stream) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=16,
                                  no_auto_bright=True, gamma=(2.4, 12.92))
        # Resize while still RGB to avoid a second full-size channel-swapped
        # copy. INTER_AREA treats each channel independently, so the result
        # is identical to resizing BGR after conversion.
        rgb = _downscale_page(rgb, scale_factor)
        return np.ascontiguousarray(rgb[:, :, ::-1])
    except Exception:
        return None


def decode_image(path: str, *, allow_transparent_png: bool = False,
                 preserve_grayscale: bool = False) -> Optional[np.ndarray]:
    """Decode an oriented still image, rejecting unsupported transparency."""
    ext = os.path.splitext(path)[1].lower()
    if is_raw_source(path):
        return _read_raw(path)
    # IMREAD_IGNORE_ORIENTATION is required: modern OpenCV already rotates by
    # the EXIF Orientation tag itself, so letting apply_exif_orientation run
    # afterwards applied every transform twice (portrait phone shots loaded
    # sideways). The PIL fallback below does not rotate on its own, so the
    # single explicit pass is the only orientation step for both paths.
    data = np.fromfile(path, dtype=np.uint8)
    unchanged = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if unchanged is not None and _has_transparency(unchanged):
        if allow_transparent_png and ext == '.png':
            return apply_exif_orientation(path, unchanged)
        raise UnsupportedTransparency(
            f"Transparent input is unsupported here: {path}. "
            "Matching rectangular PNG transparency is supported in the desktop and CLI workflows; see the manual.")
    # Standard uint8/uint16 decoders already return the BGR samples used by
    # the previous color-read path. Exotic sample types retain that path.
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


def _read_exif_bytes(path: str) -> bytes:
    """Raw EXIF block of an image (b"") when absent or unreadable."""
    if not PIL_AVAILABLE:
        return b""
    try:
        ext = os.path.splitext(path)[1].lower()
        if is_raw_source(path):
            return _read_raw_exif_bytes(path)
        if ext not in ('.jpg', '.jpeg', '.jpe', '.jfif', '.tif', '.tiff', '.png', '.webp'):
            return b""
        with PILImage.open(path) as pil_img:
            raw = pil_img.info.get('exif', b"") or b""
            if raw:
                return raw
            # Pillow reports TIFF EXIF only through getexif(); info is empty
            # there, so a stacked TIFF's camera data was previously dropped and
            # a 16-bit export of those frames came out without metadata. Keep
            # the same export-safe subset the RAW path returns: IFD0 carries
            # source strip/size pointers that must not reach an export.
            exif = read_image_exif(pil_img)
            if not len(exif):
                return b""
            safe = export_exif(exif.tobytes())
            return safe.tobytes() if safe is not None else b""
    except Exception:
        return b""


def _read_icc_profile(path: str, page: Optional[int] = None) -> bytes:
    """Read one frame's RGB profile, requiring common profiles for TIFF stacks."""
    if not PIL_AVAILABLE:
        return b""
    if is_raw_source(path):
        from utils.image_utils import srgb_icc_profile
        return srgb_icc_profile()
    from io import BytesIO
    try:
        from PIL import ImageCms
    except ImportError:
        return b""
    try:
        with PILImage.open(path) as image:
            profiles = []
            pages = range(getattr(image, "n_frames", 1)) if image.format == "TIFF" else (0,)
            if page is not None:
                pages = (page,)
            for page_index in pages:
                image.seek(page_index)
                # TIFF tags are page-specific; Pillow's info dictionary may
                # otherwise retain metadata from a previous page.
                if image.format == "TIFF":
                    profile = image.tag_v2.get(34675, b"")
                else:
                    profile = image.info.get("icc_profile", b"")
                if not (isinstance(profile, bytes) and len(profile) >= 128
                        and profile[16:20] == b"RGB " and profile[36:40] == b"acsp"):
                    return b""
                if int.from_bytes(profile[:4], "big") != len(profile):
                    return b""
                ImageCms.ImageCmsProfile(BytesIO(profile))
                profiles.append(profile)
            return common_profile_values(profiles)
    except (OSError, ValueError, TypeError, EOFError, ImageCms.PyCMSError):
        return b""


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
    except UnsupportedTransparency:
        raise
    except Exception as e:
        print(f"Unsupported image for PIL fallback {path}: {e}")
        return None


def _read_tiff_pages(path: str, scale_factor=1.0, should_cancel=None) -> Optional[List[np.ndarray]]:
    """Decode multi-page TIFF one page at a time into disk-backed frames."""
    if os.path.splitext(path)[1].lower() not in ('.tif', '.tiff') or _is_dng_tiff(path):
        return None
    count = count_frames(path)
    if count < 2:
        return None
    data = np.memmap(path, dtype=np.uint8, mode='r')
    pages = []
    for index in range(count):
        if should_cancel is not None and should_cancel():
            raise _DecodeCancelled()
        success, decoded = cv2.imdecodemulti(data, flags=cv2.IMREAD_UNCHANGED,
                                            range=(index, index + 1))
        if not success or len(decoded) != 1:
            raise ValueError(f"Could not decode TIFF page {index + 1}: {path}")
        page = decoded[0]
        if _has_transparency(page):
            raise UnsupportedTransparency(f"Transparent TIFF pages are unsupported: {path}")
        if page.ndim == 2:
            page = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
        elif page.ndim == 3 and page.shape[2] == 4:
            page = cv2.cvtColor(page, cv2.COLOR_BGRA2BGR)
        pages.append(store_frame(_downscale_page(page, scale_factor)))
    return pages


def count_frames(path: str) -> int:
    """Frames a path contributes to a stack, without decoding pixels.

    A multi-page TIFF counts as all of its pages (PIL exposes the page count
    from the directory, so counting a 40-page Z-stack costs a metadata read
    instead of a full decode); anything else counts as one.
    """
    if os.path.splitext(path)[1].lower() in (".tif", ".tiff"):
        if _is_dng_tiff(path):
            return 1
        try:
            from PIL import Image
            with Image.open(path) as im:
                frames = int(getattr(im, "n_frames", 1) or 1)
            if frames > 1:
                return frames
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
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
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
        self.source_icc_profiles = {}
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
                              resource_preference="balanced", expected_metadata=None):
        """Load a complete camera-linear stack, committing state only on success."""
        from core.source_frames import scan_source, ensure_preview
        from locales import trans
        # Camera calibration is inspected while building the source manifest.
        from utils.scalebar import detect_px_size_um

        paths = list(filepaths)
        if not paths:
            return False, trans.t("linear_raw_invalid"), [], []
        images = []
        px_sizes = []
        metadata = expected_metadata
        def unknown_camera(profile):
            return (not profile.camera_make or not profile.camera_model
                    or (profile.camera_make, profile.camera_model) == ("OpenFocus", "Linear RAW"))

        if expected_metadata is not None and unknown_camera(expected_metadata):
            return False, trans.t("linear_raw_unknown_camera"), [], []
        if self.progress_callback is not None:
            self.progress_callback(0, len(paths))
        for index, path in enumerate(paths):
            if self.should_cancel():
                return False, 'CANCELLED', [], []
            if not is_raw_source(path):
                return False, trans.t("linear_raw_invalid"), [], []

            try:
                image = scan_source(path, scale_factor, linear=True, header_only=True,
                                    should_cancel=self.should_cancel)[0]
                profile = image.linear_metadata
            except InterruptedError:
                return False, 'CANCELLED', [], []
            except MemoryError:
                raise
            except Exception as exc:
                return False, f"{trans.t('linear_raw_invalid')}\n{path}: {exc}", [], []
            if self.should_cancel():
                return False, 'CANCELLED', [], []
            if (expected_metadata is not None or len(paths) > 1) and unknown_camera(profile):
                return False, f"{trans.t('linear_raw_unknown_camera')}\n{path}", [], []
            if metadata is not None and not linear_raw_metadata_matches(metadata, profile):
                return False, f"{trans.t('linear_raw_mismatch')}\n{path}", [], []
            metadata = profile if metadata is None else metadata
            # Native DNG may carry the same microscopy calibration as TIFF.
            # Keep the file's native unit here; source_manager divides by the
            # load scale when it publishes source_px_um for working pixels.
            px_sizes.append(detect_px_size_um(path))
            images.append(image)
            if self.progress_callback is not None:
                self.progress_callback(index + 1, len(paths))
        try:
            ensure_preview(images[0], should_cancel=self.should_cancel)
        except InterruptedError:
            return False, 'CANCELLED', [], []
        if self.should_cancel():
            return False, 'CANCELLED', [], []
        self.images = images
        self.image_paths = paths
        self.source_exif = _read_exif_bytes(paths[0])
        self.source_icc_profile = b""
        self.source_icc_profiles = {}
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
        self.source_exif = b""
        self.source_icc_profile = b""
        self.source_icc_profiles = {}
        self.px_size_um = None
        self.image_paths = []
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

                if scale_factor != 1.0 and 0 < scale_factor < 1.0:
                    width = int(frame.shape[1] * scale_factor)
                    height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                loaded_images.append(store_frame(frame))
                filenames.append(f"{video_name}_frame_{frame_index:04d}.png")
                frame_index += 1
        finally:
            cap.release()

        if not loaded_images:
            return False, "No frames could be extracted from the video", [], []

        self.images = loaded_images
        self.image_paths = [video_path] * len(loaded_images)

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
        """Open paths and bounded previews; original pixels remain on disk."""
        from concurrent.futures import ThreadPoolExecutor
        from core.source_frames import scan_source, ensure_preview
        from utils.logging_setup import log_operation
        from utils.scalebar import detect_px_size_um

        sources = _collapse_repeats(list(filepaths))
        if not sources:
            return False, "No file paths provided", [], []
        images, names, paths, px_sizes = [], [], [], []
        failed = 0
        exif = b""
        workers = min(4, max(1, int(thread_count or 4)))
        def inspect(path):
            if self.should_cancel():
                raise _DecodeCancelled()
            try:
                return scan_source(path, scale_factor, header_only=True,
                                   should_cancel=self.should_cancel), None
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
                            if not exif:
                                exif = _read_exif_bytes(path)
                            px_sizes.append(detect_px_size_um(path))
                            stem = os.path.splitext(os.path.basename(path))[0]
                            images.extend(frames)
                            paths.extend([path] * len(frames))
                            names.extend(f'{stem}_page_{frame.page + 1:03d}.tif' if len(frames) > 1
                                         else os.path.basename(path) for frame in frames)
                        if self.progress_callback is not None:
                            self.progress_callback(done, len(sources))
                except _DecodeCancelled:
                    return False, 'CANCELLED', [], []
        self.failed_count = failed
        if not images:
            return False, 'Could not load any image files', [], []
        if self.should_cancel():
            return False, 'CANCELLED', [], []
        try:
            ensure_preview(images[0], should_cancel=self.should_cancel)
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
        self.px_size_um = _consistent_px_size_um(px_sizes)
        self.source_icc_profiles = {path: _read_icc_profile(path) for path in dict.fromkeys(paths)}
        self.source_icc_profile = common_profile_values(self.source_icc_profiles.values())
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
            new_w = int(w * scale)
            new_h = int(h * scale)

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
                new_w = int(w * scale)
                new_h = int(h * scale)
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
                exif_data = img._getexif()
                if exif_data is None:
                    return None

                date_time_original = None
                for tag_id, value in exif_data.items():
                    tag_name = PILImage.ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag_name == 'DateTimeOriginal':
                        date_time_original = value
                        break

                if date_time_original:
                    dt = datetime.strptime(date_time_original, '%Y:%m:%d %H:%M:%S')
                    return dt.timestamp()
        except Exception as e:
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
