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


_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)


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


def _raf_preview_identity(path, raw):
    """Extract only a bounded, JPEG RAF preview from this LibRaw handle."""
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
        stream.seek(offset)
        if stream.read(2) != b'\xff\xd8':
            return None
    require_memory(length * 3)
    try:
        thumb = raw.extract_thumb()
    except rawpy.LibRawNonFatalError:
        return None
    if thumb.format != rawpy.ThumbFormat.JPEG or len(thumb.data) != length:
        return None
    with Image.open(BytesIO(thumb.data)) as preview:
        exif = preview.getexif()
        return exif.get(271), exif.get(272)


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
    if raw is not None:
        try:
            identity = _raf_preview_identity(path, raw)
            if identity and all(isinstance(v, str) and v.strip() and v.isascii()
                                and '\x00' not in v for v in identity):
                return tuple(v.strip() for v in identity)
        except (OSError, ValueError, SyntaxError, TypeError):
            pass
    return "OpenFocus", "Linear RAW"

def _linear_dng_calibration(path):
    """Read a single-IFD, single-D65-matrix LinearRaw DNG calibration."""
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) != 1:
            raise ValueError("Unsupported DNG with multiple image directories")
        page = tif.pages[0]
        tags = page.tags
        if (page.photometric.value != 34892 or page.samplesperpixel != 3
                or 50706 not in tags or 330 in tags
                or any(tag in tags for tag in (50722, 50723, 50724, 50725,
                                                50726, 50727, 50729, 50964, 50965))
                or 50778 not in tags or tags[50778].value != 21):
            raise ValueError("Unsupported LinearRaw DNG calibration")

        def rational_values(tag_number, count, dtype):
            tag = tags.get(tag_number)
            if tag is None or int(tag.dtype) != dtype:
                raise ValueError("Missing or invalid DNG calibration tag")
            values = np.asarray(tag.value).reshape(-1)
            if values.size != count * 2:
                raise ValueError("Invalid DNG calibration rational count")
            numerator = values[::2].astype(np.float64)
            denominator = values[1::2].astype(np.float64)
            if np.any(denominator <= 0):
                raise ValueError("Invalid DNG calibration denominator")
            return numerator / denominator

        matrix = rational_values(50721, 9, 10).reshape(3, 3)
        neutral = rational_values(50728, 3, 5)
        make_tag, model_tag = tags.get(271), tags.get(272)
        if make_tag is None or model_tag is None:
            raise ValueError("LinearRaw DNG is missing camera identity")
        metadata = LinearDNGMetadata(tuple(map(tuple, matrix)), tuple(neutral),
                                     make_tag.value, model_tag.value)
        _validate_metadata(metadata)
        return metadata


def load_linear_raw(path, scale_factor=1.0, *, before_decode=None):
    """Decode an oriented RAW into black-corrected, camera-linear BGR."""
    if not math.isfinite(scale_factor) or not 0 < scale_factor <= 1:
        raise ValueError("scale_factor must be in (0, 1]")
    with rawpy.imread(os.fspath(path)) as raw:
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
        # LibRaw's unpacking, demosaic and output may coexist during postprocess.
        require_memory(width * height * 64)
        if raw.num_colors != 3:
            raise ValueError("Only three-color RAW files are supported")
        if (os.path.splitext(os.fspath(path))[1].lower() == '.dng'
                and raw.raw_type == rawpy.RawType.Stack):
            metadata = _linear_dng_calibration(path)
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
                              no_auto_bright=True, gamma=(1, 1), output_bps=16)
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
        width = max(1, int(rgb.shape[1] * scale_factor))
        height = max(1, int(rgb.shape[0] * scale_factor))
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
        rgb = np.where(rgb < 0.018, 4.5 * rgb,
                       1.099 * np.power(rgb, 0.45) - 0.099)
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
                             px_size_um=None):
    """Write the DNG payload and replace the destination atomically.

    `px_size_um` is the physical pixel size of *these* pixels (after any
    loading scale). It is stored as ResolutionUnit=cm resolution tags so a
    re-import recovers the same microscopy calibration as a TIFF.
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
        tifffile.imwrite(temporary, strips(), shape=linear_bgr.shape, dtype='<u2',
                         byteorder='<', rowsperstrip=rows_per_strip,
                         planarconfig='contig', photometric=34892, metadata=None,
                         resolution=resolution,
                         resolutionunit=3 if resolution is not None else None,
                         extratags=tags, software='OpenFocus', description=None)
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
