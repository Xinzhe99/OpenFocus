"""Camera-linear RAW decoding, display rendering and LinearRaw DNG export.

The image API uses BGR uint16, with black removed and a common sensor white
mapped to 65535.  DNG color metadata is always in camera RGB order.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
from io import BytesIO
import os
import tempfile
import struct

import cv2
import numpy as np
import rawpy
import tifffile
from core.resource_policy import require_memory
from PIL import Image


@dataclass(frozen=True)
class LinearDNGMetadata:
    color_matrix: tuple
    as_shot_neutral: tuple
    camera_make: str = "OpenFocus"
    camera_model: str = "Linear RAW"
    # Exact TIFF tag values are immutable and can be written without changing
    # the profile's rational values or floating-point lookup tables.
    profile_tags: tuple = ()


_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)


def _valid_xtrans(pattern):
    values = np.asarray(pattern, dtype=np.uint8)
    if values.shape != (6, 6):
        return False
    standard = np.array([
        [1, 2, 1, 1, 0, 1], [0, 1, 0, 2, 1, 2], [1, 2, 1, 1, 0, 1],
        [1, 0, 1, 1, 2, 1], [2, 1, 2, 0, 1, 0], [1, 0, 1, 1, 2, 1],
    ], dtype=np.uint8)
    for rotation in range(4):
        rotated = np.rot90(standard, rotation)
        for candidate in (rotated, rotated[:, ::-1]):
            for y in range(6):
                for x in range(6):
                    if np.array_equal(values, np.roll(candidate, (y, x), axis=(0, 1))):
                        return True
    return False


def _validate_metadata(metadata):
    if not isinstance(metadata, LinearDNGMetadata):
        raise ValueError("Expected LinearDNGMetadata")
    try:
        matrix = np.asarray(metadata.color_matrix, dtype=np.float64)
        neutral = np.asarray(metadata.as_shot_neutral, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid camera color metadata") from exc
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix)) < 1e-10):
        raise ValueError("Invalid camera color matrix")
    if (neutral.shape != (3,) or not np.isfinite(neutral).all()
            or np.any(neutral <= 0)):
        raise ValueError("Invalid as-shot neutral")
    camera_to_srgb = matrix @ _RGB_TO_XYZ
    if np.any(np.abs(camera_to_srgb.sum(axis=1)) < 1e-10):
        raise ValueError("Invalid camera color matrix normalization")
    for value in (metadata.camera_make, metadata.camera_model):
        if not isinstance(value, str) or not value.strip() or '\x00' in value:
            raise ValueError("Invalid camera identity")
        try:
            value.encode('ascii')
        except UnicodeEncodeError as exc:
            raise ValueError("Camera identity must be ASCII") from exc
    return matrix, neutral


def _validate_image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint16
            or image.ndim != 3 or image.shape[2] != 3
            or image.shape[0] == 0 or image.shape[1] == 0):
        raise ValueError("Expected a nonempty BGR uint16 image")


def _canon_cr3_identity(path):
    """Read Canon's bounded CMT1 TIFF from the CR3 metadata box."""
    canon_uuid = bytes.fromhex('85c0b687820f11e08111f4ce462b6a48')
    with open(path, 'rb') as stream:
        stream.seek(0, os.SEEK_END)
        file_size = stream.tell()

        def boxes(start, end):
            position = start
            for _ in range(4096):
                if position + 8 > end:
                    return
                stream.seek(position)
                header = stream.read(8)
                length = int.from_bytes(header[:4], 'big')
                kind = header[4:]
                header_size = 8
                if length == 1:
                    extra = stream.read(8)
                    if len(extra) != 8:
                        return
                    length = int.from_bytes(extra, 'big')
                    header_size = 16
                elif length == 0:
                    length = end - position
                if length < header_size or position + length > end:
                    return
                yield kind, position + header_size, position + length
                position += length

        for kind, start, end in boxes(0, file_size):
            if kind != b'moov':
                continue
            for child, child_start, child_end in boxes(start, end):
                if child != b'uuid' or child_start + 16 > child_end:
                    continue
                stream.seek(child_start)
                if stream.read(16) != canon_uuid:
                    continue
                for item, payload, item_end in boxes(child_start + 16, child_end):
                    size = item_end - payload
                    if item != b'CMT1' or not 8 <= size <= 1024 * 1024:
                        continue
                    stream.seek(payload)
                    data = stream.read(size)
                    if data[:4] not in (b'II*\x00', b'MM\x00*'):
                        return None
                    exif = Image.Exif()
                    exif.load(b'Exif\x00\x00' + data)
                    return exif.get(271), exif.get(272)
    return None


def _raf_preview_identity(path):
    """Read only a bounded embedded JPEG and inspect its EXIF metadata."""
    max_preview = 16 * 1024 * 1024
    with open(path, 'rb') as stream:
        header = stream.read(92)
        if len(header) != 92 or header[:16] != b'FUJIFILMCCD-RAW ':
            return None
        offset = int.from_bytes(header[84:88], 'big')
        length = int.from_bytes(header[88:92], 'big')
        size = os.fstat(stream.fileno()).st_size
        if offset < 92 or length < 4 or length > max_preview or offset + length > size:
            return None
        require_memory(length * 3)
        stream.seek(offset)
        jpeg = stream.read(length)
        if len(jpeg) != length or not jpeg.startswith(b'\xff\xd8'):
            return None
    with Image.open(BytesIO(jpeg)) as preview:
        if preview.format != 'JPEG':
            return None
        exif = preview.getexif()
        return exif.get(271), exif.get(272)


def _raf_header_identity(path):
    """Read the bounded model field from a Fujifilm RAF fixed header."""
    with open(path, 'rb') as stream:
        header = stream.read(92)
    if len(header) != 92 or header[:16] != b'FUJIFILMCCD-RAW ':
        return None
    field = header[28:60]
    model, separator, padding = field.partition(b'\x00')
    if separator and padding.strip(b'\x00 '):
        return None
    model = model.strip(b' ')
    if (not model or any(byte < 32 or byte > 126 for byte in model)
            or not any(48 <= byte <= 57 or 65 <= byte <= 90 or 97 <= byte <= 122
                       for byte in model)):
        return None
    return 'FUJIFILM', model.decode('ascii')


def _tiff_camera_identity(path):
    """Read Make/Model from TIFF-based RAW headers without decoding pixels."""
    from core.image_loader import _RAW_TIFF_MAGICS

    with open(path, 'rb') as stream:
        header = stream.read(8)
        if len(header) != 8 or header[:4] not in _RAW_TIFF_MAGICS:
            return None
        endian = '<' if header[:2] == b'II' else '>'
        size = os.fstat(stream.fileno()).st_size
        offset = struct.unpack(endian + 'I', header[4:])[0]
        if offset < 8 or offset + 2 > size:
            return None
        stream.seek(offset)
        count = struct.unpack(endian + 'H', stream.read(2))[0]
        if count > 4096 or offset + 2 + count * 12 > size:
            return None
        directory = stream.read(count * 12)
        values = {}
        for index in range(count):
            tag, field_type, length, data = struct.unpack_from(
                endian + 'HHI4s', directory, index * 12)
            if tag not in (271, 272) or field_type != 2 or not 0 < length <= 512:
                continue
            if length <= 4:
                value = data[:length]
            else:
                location = struct.unpack(endian + 'I', data)[0]
                if location > size or length > size - location:
                    continue
                stream.seek(location)
                value = stream.read(length)
            values[tag] = value.rstrip(b'\0').decode('ascii').strip()
        return values.get(271), values.get(272)


def _camera_identity(path, raw=None):
    # rawpy does not expose make/model; inspect only container metadata.
    try:
        identity = _tiff_camera_identity(path)
        if identity and all(isinstance(v, str) and v.strip() and v.isascii()
                            and '\x00' not in v for v in identity):
            return tuple(v.strip() for v in identity)
    except (OSError, UnicodeError, ValueError, struct.error):
        pass
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            make, model = exif.get(271), exif.get(272)
            if all(isinstance(v, str) and v.strip() and v.isascii()
                   and '\x00' not in v for v in (make, model)):
                return make.strip(), model.strip()
    except (OSError, ValueError, SyntaxError):
        pass
    try:
        identity = _canon_cr3_identity(path)
        if identity and all(isinstance(v, str) and v.strip() and v.isascii()
                            and '\x00' not in v for v in identity):
            return tuple(v.strip() for v in identity)
    except (OSError, ValueError, SyntaxError, TypeError):
        pass
    try:
        identity = _raf_preview_identity(path)
        if identity and all(isinstance(v, str) and v.strip() and v.isascii()
                            and '\x00' not in v for v in identity):
            return tuple(v.strip() for v in identity)
    except (OSError, ValueError, SyntaxError, TypeError, struct.error,
            Image.DecompressionBombError):
        pass
    try:
        identity = _raf_header_identity(path)
        if identity:
            return identity
    except OSError:
        pass
    return "OpenFocus", "Linear RAW"

_DNG_PROFILE_TYPES = {
    50707: 'B', 50708: 's', 50721: '2i', 50722: '2i', 50723: '2i', 50724: '2i',
    50725: '2i', 50726: '2i', 50727: '2I', 50728: '2I', 50730: '2i',
    50778: 'H', 50779: 'H', 50931: 's', 50932: 's', 50934: 's', 50936: 's',
    50937: 'I', 50938: 'f', 50939: 'f', 50940: 'f', 50941: 'I',
    50942: 's', 50964: '2i', 50965: '2i', 50981: 'I', 50982: 'f',
    51107: 'I', 51108: 'I',
}
_DNG_TYPE_NUMBERS = {'B': 1, 's': 2, '2i': 10, '2I': 5, 'H': 3, 'I': 4, 'f': 11}


def _linear_dng_calibration(path):
    """Read the root DNG color profile and locate its sole LinearRaw IFD."""
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) != 1:
            raise ValueError("Unsupported DNG with multiple image directories")
        page = tif.pages[0]
        tags = page.tags
        pages = [page] + (list(page.pages) if page.pages else [])
        if any(child.pages for child in pages[1:]):
            raise ValueError("Unsupported DNG with nested SubIFDs")
        if any(code in child.tags for child in pages[1:] for code in _DNG_PROFILE_TYPES):
            raise ValueError("Unsupported DNG: color profile tags in SubIFDs")
        raw_pages = [p for p in pages if (p.photometric.value == 34892 and p.samplesperpixel == 3)
                     or (p.photometric.value == 32803 and p.samplesperpixel == 1)]
        if len(raw_pages) != 1 or 50706 not in tags:
            raise ValueError("Unsupported DNG: expected one LinearRaw or CFA image")
        raw_page = raw_pages[0]
        if raw_page.photometric.value == 32803:
            raw_tags = raw_page.tags
            dimensions = raw_tags.get(33421)
            pattern = raw_tags.get(33422)
            planes = tags.get(50710) or raw_tags.get(50710)
            layout = tags.get(50711) or raw_tags.get(50711)
            dims = tuple(dimensions.value) if dimensions is not None else ()
            values = tuple(pattern.value) if pattern is not None else ()
            bayer = dims == (2, 2) and values in (
                (0, 1, 1, 2), (1, 0, 2, 1), (1, 2, 0, 1), (2, 1, 1, 0))
            xtrans = (dims == (6, 6) and len(values) == 36
                      and _valid_xtrans(np.asarray(values).reshape(6, 6)))
            if (not (bayer or xtrans)
                    or (planes is not None and tuple(planes.value) != (0, 1, 2))
                    or (layout is not None and layout.value != 1)):
                raise ValueError("Unsupported DNG CFA: expected RGB Bayer or X-Trans")
        version = tuple(tags[50706].value)
        if version > (1, 4, 0, 0):
            raise ValueError("Unsupported DNG version: profiles newer than 1.4 cannot be preserved")
        unsupported = {
            50729: 'as-shot white XY', 50831: 'as-shot ICC profile',
            50832: 'as-shot pre-profile matrix', 50833: 'current ICC profile',
            50834: 'current pre-profile matrix', 50879: 'colorimetric reference',
            50933: 'additional profile IFD', 51008: 'raw processing opcode list',
            51009: 'raw processing opcode list', 51022: 'raw processing opcode list',
            51109: 'profile baseline exposure offset', 51110: 'default black rendering',
            52525: 'profile gain table', 52529: 'third calibration illuminant',
            52530: 'third camera calibration matrix', 52531: 'third color matrix',
            52532: 'third forward matrix', 52533: 'illuminant data',
            52534: 'second illuminant data', 52537: 'third profile hue/saturation map',
            52538: 'third reduction matrix', 52543: 'profile RGB tables',
            52544: 'second profile gain table',
        }
        for directory in pages:
            for code, description in unsupported.items():
                if code in directory.tags:
                    raise ValueError(f"Unsupported DNG: {description} (tag {code})")
        if tags.get(50941) is not None and tags[50941].value not in (0, 1):
            raise ValueError("DNG profile embedding policy does not permit profile copying")
        profile = []
        for code, dtype in _DNG_PROFILE_TYPES.items():
            tag = tags.get(code)
            if tag is None:
                continue
            if int(tag.dtype) != _DNG_TYPE_NUMBERS[dtype]:
                raise ValueError(f"Unsupported DNG profile tag {code} datatype")
            value = tag.value
            if isinstance(value, np.ndarray):
                value = tuple(value.reshape(-1).tolist())
            elif isinstance(value, (list, tuple)):
                value = tuple(value)
            if dtype == 'f' and not np.isfinite(np.asarray(value)).all():
                raise ValueError(f"Invalid DNG profile tag {code}")
            profile.append((code, dtype, tag.count, value))
        for code, dtype, count, value in profile:
            if dtype in ('2i', '2I'):
                values = np.asarray(value).reshape(-1)
                if values.size != count * 2 or np.any(values[1::2] <= 0):
                    raise ValueError(f"Invalid DNG profile rational tag {code}")
            if code in (50721, 50722, 50723, 50724, 50725, 50726, 50964, 50965) and count != 9:
                raise ValueError(f"Invalid DNG profile matrix tag {code}")
            if code in (50727, 50728) and count != 3:
                raise ValueError(f"Invalid DNG profile color tag {code}")
        if (50722 in tags) != (50779 in tags):
            raise ValueError("Incomplete DNG second illuminant profile")
        for dims_code, data_codes in ((50937, (50938, 50939)), (50981, (50982,))):
            if dims_code in tags:
                dims = tuple(tags[dims_code].value)
                if len(dims) != 3 or any(d <= 0 for d in dims):
                    raise ValueError(f"Invalid DNG profile table dimensions (tag {dims_code})")
                expected = math.prod(dims) * 3
                if data_codes[0] not in tags or any(
                    code in tags and tags[code].count != expected for code in data_codes
                ):
                    raise ValueError(f"Incomplete DNG profile table (tag {dims_code})")
            elif any(code in tags for code in data_codes):
                raise ValueError(f"Missing DNG profile table dimensions (tag {dims_code})")

        def rational_values(tag_number, count, dtype):
            tag = tags.get(tag_number)
            if tag is None or int(tag.dtype) != dtype:
                raise ValueError(f"Missing or invalid DNG calibration tag {tag_number}")
            values = np.asarray(tag.value).reshape(-1)
            if values.size != count * 2:
                raise ValueError(f"Invalid DNG calibration rational count in tag {tag_number}")
            numerator = values[::2].astype(np.float64)
            denominator = values[1::2].astype(np.float64)
            if np.any(denominator <= 0):
                raise ValueError(f"Invalid DNG calibration denominator in tag {tag_number}")
            return numerator / denominator

        illuminant = tags.get(50778)
        second = tags.get(50779)
        if illuminant is not None and illuminant.value == 21:
            matrix_tag = 50721
        elif second is not None and second.value == 21:
            matrix_tag = 50722
        else:
            raise ValueError("Unsupported DNG profile: a D65 matrix is required for preview")
        matrix = rational_values(matrix_tag, 9, 10).reshape(3, 3)
        neutral = rational_values(50728, 3, 5)
        make_tag, model_tag = tags.get(271), tags.get(272)
        if make_tag is None or model_tag is None:
            raise ValueError("LinearRaw DNG is missing camera identity")
        if 50708 not in tags or not tags[50708].value:
            raise ValueError("LinearRaw DNG is missing UniqueCameraModel (tag 50708)")
        simple = (set(code for code, _, _, _ in profile) <= {50707, 50708, 50721, 50728, 50778}
                  and (50707 not in tags or tuple(tags[50707].value) == (1, 1, 0, 0))
                  and tags[50708].value == model_tag.value)
        metadata = LinearDNGMetadata(tuple(map(tuple, matrix)), tuple(neutral),
                                     make_tag.value, model_tag.value, () if simple else tuple(profile))
        _validate_metadata(metadata)
        return metadata


def load_linear_raw(path, scale_factor=1.0, *, before_decode=None):
    """Decode an oriented RAW into black-corrected, camera-linear BGR."""
    if not math.isfinite(scale_factor) or not 0 < scale_factor <= 1:
        raise ValueError("scale_factor must be in (0, 1]")
    # Inspect TIFF headers before LibRaw unpacks pixels. DNG color profiles
    # must never fall through to LibRaw's simplified camera matrix.
    try:
        with tifffile.TiffFile(path) as tif:
            is_dng = bool(tif.is_dng)
    except (OSError, ValueError, tifffile.TiffFileError):
        is_dng = False
    if os.path.splitext(os.fspath(path))[1].lower() == '.dng' and not is_dng:
        raise ValueError("Invalid DNG: DNGVersion tag is missing or unreadable")
    dng_metadata = _linear_dng_calibration(path) if is_dng else None
    with rawpy.RawPy() as raw:
        raw.open_file(os.fspath(path))
        sizes = raw.sizes
        aspect = float(sizes.pixel_aspect)
        if not math.isfinite(aspect) or aspect <= 0:
            raise ValueError("Invalid RAW pixel aspect ratio")
        width = math.ceil(max(sizes.raw_width, sizes.width) * max(1, aspect, 1 / aspect))
        height = max(sizes.raw_height, sizes.height)
        if width <= 0 or height <= 0:
            raise ValueError("Invalid RAW dimensions")
        if before_decode is not None:
            before_decode(width, height)
        # RawPy's sensor properties implicitly unpack, so admit that allocation
        # before inspecting the mosaic or choosing a postprocess route.
        require_memory(width * height * 16)
        raw.unpack()
        # Bayer half-size decoding combines each sensor block before demosaic.
        # Other mosaics and LinearRaw DNGs keep the full-resolution path.
        pattern = raw.raw_pattern if raw.raw_type == rawpy.RawType.Flat else None
        half_size = (scale_factor <= 0.5 and pattern is not None
                     and np.shape(pattern) == (2, 2))
        if half_size:
            output_width, output_height = sizes.width, sizes.height
            if aspect < 1:
                output_height = round(output_height / aspect)
            elif aspect > 1:
                output_width = round(output_width * aspect)
            if sizes.flip in (4, 5, 6, 7):
                output_width, output_height = output_height, output_width
            target_size = (max(1, int(output_width * scale_factor)),
                           max(1, int(output_height * scale_factor)))
        require_memory(width * height * (24 if half_size else 64))
        if raw.num_colors != 3:
            raise ValueError("Only three-color RAW files are supported")
        if dng_metadata is not None:
            if raw.raw_type not in (rawpy.RawType.Stack, rawpy.RawType.Flat):
                raise ValueError("Unsupported DNG: expected LinearRaw or Bayer CFA pixels")
            metadata = dng_metadata
        else:
            matrix = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
            wb = np.asarray(raw.camera_whitebalance, dtype=np.float64)[:3]
            if wb.shape != (3,) or not np.isfinite(wb).all() or np.any(wb <= 0):
                raise ValueError("RAW has no valid camera white balance")
            neutral = wb[1] / wb
            make, model = _camera_identity(path, raw)
            metadata = LinearDNGMetadata(tuple(map(tuple, matrix)), tuple(neutral), make, model)
            _validate_metadata(metadata)
        black = np.asarray(raw.black_level_per_channel, dtype=np.float64)
        if black.size < 3 or not np.isfinite(black).all() or np.any(black < 0):
            raise ValueError("Invalid RAW black level")
        white = float(raw.white_level) - float(black.min())
        if not math.isfinite(white) or not 0 < white <= 65535:
            raise ValueError("Invalid RAW white level")
        rgb = raw.postprocess(output_color=rawpy.ColorSpace.raw,
                              user_wb=[1, 1, 1, 1], no_auto_scale=True,
                              no_auto_bright=True, gamma=(1, 1), output_bps=16,
                              half_size=half_size)
    if rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("LibRaw returned an unsupported image")
    require_memory(rgb.nbytes * 2 + 65536 * 10)
    # LibRaw has subtracted the channel black levels, but no_auto_scale leaves
    # the remaining sensor range unscaled. A LUT avoids a full float image.
    if white != 65535:
        lut = np.minimum(np.rint(np.arange(65536, dtype=np.float64) *
                                 (65535.0 / white)), 65535).astype(np.uint16)
        rgb = lut[rgb]
    if scale_factor != 1:
        width, height = (target_size if half_size else
                         (max(1, int(rgb.shape[1] * scale_factor)),
                          max(1, int(rgb.shape[0] * scale_factor))))
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb[:, :, ::-1]), metadata


def develop_linear(linear_bgr, metadata):
    """Render camera-linear pixels to white-balanced, display-referred BGR."""
    _validate_image(linear_bgr)
    matrix, neutral = _validate_metadata(metadata)
    conversion = matrix @ _RGB_TO_XYZ
    conversion /= conversion.sum(axis=1)[:, None]
    conversion = np.linalg.inv(conversion)
    wb = 1.0 / neutral
    wb /= wb.min()
    require_memory(linear_bgr.nbytes + min(128, linear_bgr.shape[0]) * linear_bgr.shape[1] * 80)
    # Process bounded strips to avoid several full-frame floating point buffers.
    result = np.empty_like(linear_bgr)
    for start in range(0, linear_bgr.shape[0], 128):
        stop = min(start + 128, linear_bgr.shape[0])
        rgb = linear_bgr[start:stop, :, ::-1].astype(np.float32)
        rgb *= (wb / 65535.0).astype(np.float32)
        rgb = cv2.transform(rgb, conversion.astype(np.float32))
        np.maximum(rgb, 0, out=rgb)
        rgb = np.where(rgb <= 0.0031308, 12.92 * rgb,
                       1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
        result[start:stop] = np.rint(np.clip(rgb[:, :, ::-1], 0, 1) * 65535).astype(np.uint16)
    return result


def _rationals(values, signed=False):
    result = []
    limit = 2147483647 if signed else 4294967295
    for value in values:
        fraction = Fraction(float(value)).limit_denominator(1000000)
        if abs(fraction.numerator) > limit or fraction.denominator > limit:
            raise ValueError("Color metadata is outside DNG rational range")
        result.extend((fraction.numerator, fraction.denominator))
    return tuple(result)


def _write_linear_dng_atomic(path, linear_bgr, metadata, *, should_cancel=None,
                             px_size_um=None, display_preview=None):
    """Write the DNG payload and replace the destination atomically.

    `px_size_um` is the physical pixel size of *these* pixels (after any
    loading scale). It is stored as ResolutionUnit=cm resolution tags so a
    re-import recovers the same microscopy calibration as a TIFF.

    `display_preview`, when supplied by RAW conversion, is bounded display RGB;
    it occupies a separate reduced IFD without changing the LinearRaw payload.
    """
    _validate_image(linear_bgr)
    matrix, neutral = _validate_metadata(metadata)
    matrix_values = _rationals(matrix.flat, signed=True)
    neutral_values = _rationals(neutral)
    # Do not write values that round to zero in the DNG representation.
    if any(neutral_values[i] == 0 for i in range(0, 6, 2)):
        raise ValueError("As-shot neutral cannot be represented in DNG")
    resolution = None
    if px_size_um is not None:
        try:
            px_size_um = float(px_size_um)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid physical pixel size") from exc
        if not math.isfinite(px_size_um) or not 1e-4 <= px_size_um <= 1e3:
            raise ValueError("Physical pixel size is outside the DNG calibration range")
        # TIFF stores pixels per resolution unit. Centimetres keep the DNG
        # readable while matching scalebar's explicit-unit detection.
        pixels_per_cm = 1e4 / px_size_um
        resolution = (pixels_per_cm, pixels_per_cm)
    tags = [
        (271, 's', 0, metadata.camera_make, False),
        (272, 's', 0, metadata.camera_model, False),
        (274, 'H', 1, 1, False),
        (50706, 'B', 4, (1, 4, 0, 0), False),
        (50707, 'B', 4, (1, 1, 0, 0), False),
        (50708, 's', 0, metadata.camera_model, False),
        (50713, 'H', 2, (1, 1), False),
        (50714, '2I', 3, (0, 1, 0, 1, 0, 1), False),
        (50717, 'I', 3, (65535, 65535, 65535), False),
        (50721, '2i', 9, matrix_values, False),
        (50728, '2I', 3, neutral_values, False),
        (50778, 'H', 1, 21, False),
    ]
    if metadata.profile_tags:
        tags = [tag for tag in tags if tag[0] not in _DNG_PROFILE_TYPES]
        tags.extend((code, dtype, count, value, False)
                    for code, dtype, count, value in metadata.profile_tags)
    rows_per_strip = min(128, linear_bgr.shape[0])
    require_memory(rows_per_strip * linear_bgr.shape[1] * 12)
    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    check_cancel()

    def strips():
        for start in range(0, linear_bgr.shape[0], rows_per_strip):
            check_cancel()
            rgb = np.ascontiguousarray(linear_bgr[start:start + rows_per_strip, :, ::-1],
                                       dtype='<u2')
            yield rgb.tobytes()

    destination = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(destination)
    fd, temporary = tempfile.mkstemp(prefix='.openfocus-', suffix='.dng', dir=directory)
    os.close(fd)
    try:
        with tifffile.TiffWriter(temporary, byteorder='<') as writer:
            writer.write(strips(), shape=linear_bgr.shape, dtype='<u2',
                         rowsperstrip=rows_per_strip,
                         planarconfig='contig', photometric=34892, subfiletype=0, metadata=None,
                         resolution=resolution,
                         resolutionunit=3 if resolution is not None else None,
                         subifds=1 if display_preview is not None else None,
                         extratags=tags, software='OpenFocus', description=None)
            if display_preview is not None:
                check_cancel()
                _write_display_ifd(writer, display_preview)
        check_cancel()
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_linear_dng(path, linear_bgr, metadata, *, should_cancel=None,
                     px_size_um=None):
    """Atomically write a 16-bit, three-channel camera-linear DNG."""
    from utils.image_utils import imwrite_auto
    if os.path.splitext(os.fspath(path))[1].lower() != '.dng':
        raise ValueError("Linear DNG output requires a .dng extension")

    if not imwrite_auto(path, linear_bgr, linear_dng_metadata=metadata,
                        should_cancel=should_cancel,
                        linear_dng_px_size_um=px_size_um):
        raise OSError(f"Could not write linear DNG: {path}")


def _cfa_metadata(raw, source):
    """Describe unrotated LibRaw samples; never develop or rescale a mosaic."""
    sizes = raw.sizes
    if raw.raw_type != rawpy.RawType.Flat or raw.num_colors != 3:
        raise ValueError('Unsupported RAW sensor: expected a three-color flat CFA')
    pattern = raw.raw_pattern
    if pattern is None or pattern.shape not in ((2, 2), (6, 6)):
        raise ValueError('Unsupported RAW CFA layout (expected Bayer or X-Trans)')
    if not math.isclose(float(sizes.pixel_aspect), 1.0, abs_tol=1e-6):
        raise ValueError('Unsupported non-square or diagonal RAW sensor layout')
    repeat = pattern.shape
    # DNG CFA and black patterns start at ActiveArea; raw_color includes margins.
    indices = np.array([[raw.raw_color(y + sizes.top_margin, x + sizes.left_margin)
                         for x in range(repeat[1])]
                        for y in range(repeat[0])], dtype=np.uint8)
    description = raw.color_desc.decode('ascii').rstrip('\x00')
    try:
        colors = np.array([{'R': 0, 'G': 1, 'B': 2}[description[i]]
                           for i in indices.flat], dtype=np.uint8).reshape(repeat)
    except (KeyError, IndexError) as exc:
        raise ValueError('Unsupported RAW CFA color planes') from exc
    if (repeat == (2, 2) and tuple(colors.flat) not in (
            (0, 1, 1, 2), (1, 0, 2, 1), (1, 2, 0, 1), (2, 1, 1, 0))
            or repeat == (6, 6) and not _valid_xtrans(colors)):
        raise ValueError('Unsupported RAW CFA arrangement')
    top, left = int(sizes.top_margin), int(sizes.left_margin)
    bottom, right = top + int(sizes.height), left + int(sizes.width)
    if not (0 <= top < bottom <= sizes.raw_height and
            0 <= left < right <= sizes.raw_width):
        raise ValueError('Unsupported RAW active sensor bounds')
    if sizes.flip not in range(8):
        raise ValueError('Unsupported RAW orientation')
    matrix = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    wb = np.asarray(raw.camera_whitebalance, dtype=np.float64)[:3]
    if wb.shape != (3,) or not np.isfinite(wb).all() or np.any(wb <= 0):
        raise ValueError('RAW is missing valid as-shot white balance')
    make, model = _camera_identity(source, raw)
    if (make, model) == ('OpenFocus', 'Linear RAW'):
        raise ValueError('Cannot identify the RAW camera for DNG calibration')
    metadata = LinearDNGMetadata(tuple(map(tuple, matrix)), tuple(wb[1] / wb), make, model)
    _validate_metadata(metadata)
    black = np.asarray(raw.black_level_per_channel, dtype=np.float64)
    if (black.size <= int(indices.max()) or not np.isfinite(black).all()
            or np.any(black < 0)):
        raise ValueError('Invalid RAW per-channel black levels')
    white = int(raw.white_level)
    channel_white = raw.camera_white_level_per_channel
    if channel_white is not None:
        used = np.asarray(channel_white)[np.unique(indices)]
        used = used[used > 0]
        if used.size and np.any(used != white):
            raise ValueError('Unsupported unequal channel saturation levels; CFA samples cannot be preserved exactly')
    if not 0 < white <= 65535 or np.any(black[indices] >= white):
        raise ValueError('Invalid RAW black/white level range')
    crop_origin, crop_size = (0, 0), (sizes.width, sizes.height)
    inset_width, inset_height = getattr(sizes, 'crop_width', 0), getattr(sizes, 'crop_height', 0)
    if inset_width > 0 and inset_height > 0:
        crop_x = int(getattr(sizes, 'crop_left_margin', left)) - left
        crop_y = int(getattr(sizes, 'crop_top_margin', top)) - top
        if not (0 <= crop_x < sizes.width and 0 <= crop_y < sizes.height
                and crop_x + inset_width <= sizes.width and crop_y + inset_height <= sizes.height):
            raise ValueError('Unsupported RAW default crop bounds')
        crop_origin, crop_size = (crop_x, crop_y), (inset_width, inset_height)
    orientation = (1, 2, 4, 3, 5, 8, 6, 7)[sizes.flip]
    return [
        (271, 's', 0, make, False), (272, 's', 0, model, False),
        (274, 'H', 1, orientation, False),
        (33421, 'H', 2, repeat, False),
        (33422, 'B', colors.size, tuple(colors.flat), False),
        (50706, 'B', 4, (1, 4, 0, 0), False),
        (50707, 'B', 4, (1, 3, 0, 0) if repeat == (6, 6) else (1, 1, 0, 0), False),
        (50708, 's', 0, model, False),
        (50710, 'B', 3, (0, 1, 2), False), (50711, 'H', 1, 1, False),
        (50713, 'H', 2, repeat, False),
        (50714, '2I', colors.size, _rationals(black[indices].flat), False),
        (50717, 'I', 1, white, False),
        (50719, 'I', 2, crop_origin, False),
        (50720, 'I', 2, crop_size, False),
        (50721, '2i', 9, _rationals(matrix.flat, signed=True), False),
        (50728, '2I', 3, _rationals(metadata.as_shot_neutral), False),
        (50778, 'H', 1, 21, False),
        (50829, 'I', 4, (0, 0, sizes.height, sizes.width), False),
    ]


def _conversion_display_preview(raw, source, should_cancel):
    """Reuse the bounded display reader; never unpack a second sensor."""
    from core.source_frames import _raw_preview, _UnavailablePreview, _small, PREVIEW_EDGE
    try:
        # CFA samples retain their orientation tag. LinearRaw export is already
        # oriented by LibRaw, so its display must use the same pixel domain.
        orientation = (1 if raw.raw_type == rawpy.RawType.Flat else
                       (1, 2, 4, 3, 5, 8, 6, 7)[raw.sizes.flip])
        preview = _raw_preview(raw, orientation, source, should_cancel)
    except InterruptedError:
        raise
    except (rawpy.LibRawError, OSError, ValueError, MemoryError):
        if should_cancel is not None and should_cancel():
            raise InterruptedError('DNG conversion cancelled')
        return None
    if isinstance(preview, _UnavailablePreview):
        return None
    if should_cancel is not None and should_cancel():
        raise InterruptedError('DNG conversion cancelled')
    preview = _small(preview, edge=min(PREVIEW_EDGE, max(raw.sizes.width, raw.sizes.height)))
    if preview.dtype == np.uint16:
        preview = (preview >> 8).astype(np.uint8)
    return np.ascontiguousarray(preview[:, :, ::-1])


def _write_display_ifd(writer, rgb, *, orientation=1):
    # Uncompressed RGB needs no optional codec and remains bounded to 768px.
    writer.write(rgb, photometric='rgb', planarconfig='contig', subfiletype=1,
                 metadata=None, description=None, software='OpenFocus',
                 extratags=[(274, 'H', 1, orientation, False),
                            (50970, 'I', 1, 2, False)])  # PreviewColorSpace: sRGB


def convert_raw_to_dng(path, source, *, should_cancel=None):
    """Export CFA samples losslessly, or preserve an existing DNG verbatim.

    Non-CFA three-color sensors use the explicit camera-linear LinearRaw
    representation. Unsupported calibration/layouts fail before publication.
    The caller owns destination reservation and atomic publication.
    """
    from utils.logging_setup import log_operation, log_action
    from core.registration import RegistrationCancelled
    if os.path.exists(path) and os.path.samefile(path, source):
        raise ValueError('DNG destination must differ from its source')
    stat = os.stat(source)
    source_stamp = (stat.st_size, stat.st_mtime_ns)

    def check_source():
        stat = os.stat(source)
        if (stat.st_size, stat.st_mtime_ns) != source_stamp:
            raise OSError(f'Source changed during DNG conversion: {source}')

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise InterruptedError('DNG conversion cancelled')

    with log_operation('converter.dng', source=os.fspath(source), output=os.fspath(path)) as operation:
        try:
            check_cancel()
            try:
                with tifffile.TiffFile(source) as tif:
                    existing_dng = bool(tif.is_dng)
            except (OSError, ValueError, tifffile.TiffFileError):
                existing_dng = False
            if existing_dng:
                # Preserve profiles, opcodes, linearization and private camera
                # metadata that LibRaw's Python API cannot round-trip.
                with rawpy.RawPy() as raw:
                    raw.open_file(os.fspath(source))
                    if raw.sizes.raw_width <= 0 or raw.sizes.raw_height <= 0:
                        raise ValueError('Invalid DNG dimensions')
                with open(source, 'rb') as incoming, open(path, 'wb') as outgoing:
                    while True:
                        check_cancel()
                        chunk = incoming.read(1024 * 1024)
                        if not chunk:
                            break
                        outgoing.write(chunk)
                log_action('converter.dng.format', format='original DNG (unchanged)')
                check_cancel()
                check_source()
                return
            with rawpy.RawPy() as raw:
                raw.open_file(os.fspath(source))
                check_cancel()
                display_preview = _conversion_display_preview(raw, source, should_cancel)
                check_cancel()
                check_source()
                if raw.raw_type != rawpy.RawType.Flat:
                    linear_sensor = True
                else:
                    linear_sensor = False
                    check_cancel()
                    tags = _cfa_metadata(raw, source)
                    # Trim only masked optical borders. This keeps odd sensor
                    # margins from triggering LibRaw's Bayer phase rounding.
                    pixels = raw.raw_image_visible
                    if (pixels.dtype != np.uint16 or pixels.ndim != 2
                            or pixels.shape != (raw.sizes.height, raw.sizes.width)):
                        raise ValueError('Unsupported RAW sample representation')
                    if pixels.nbytes > 2**32 - 1024 * 1024:
                        raise ValueError('CFA DNG exceeds the standard TIFF 4 GiB limit')
                    rows = min(128, pixels.shape[0])

                    def strips():
                        for start in range(0, pixels.shape[0], rows):
                            check_cancel()
                            yield np.ascontiguousarray(pixels[start:start + rows], dtype='<u2').tobytes()

                    with tifffile.TiffWriter(path, byteorder='<', bigtiff=False) as writer:
                        writer.write(strips(), shape=pixels.shape, dtype='<u2',
                                     rowsperstrip=rows, photometric=32803,
                                     subfiletype=0, metadata=None, description=None, software='OpenFocus',
                                     subifds=1 if display_preview is not None else None, extratags=tags)
                        if display_preview is not None:
                            check_cancel()
                            _write_display_ifd(writer, display_preview,
                                               orientation=(1, 2, 4, 3, 5, 8, 6, 7)[raw.sizes.flip])
                    log_action('converter.dng.format', format='CFA', repeat=raw.raw_pattern.shape,
                               width=pixels.shape[1], height=pixels.shape[0], storage_bits=16,
                               white_level=raw.white_level)
            if linear_sensor:
                check_cancel()
                pixels, metadata = load_linear_raw(source)
                check_source()
                _write_linear_dng_atomic(path, pixels, metadata, should_cancel=should_cancel,
                                         display_preview=display_preview)
                log_action('converter.dng.format', format='LinearRaw (camera-linear RGB)')
            check_cancel()
            check_source()
        except (InterruptedError, RegistrationCancelled):
            operation.cancel()
            raise
