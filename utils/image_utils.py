import numbers
import os
from functools import lru_cache

import cv2
import numpy as np
from typing import Iterable, Optional

try:  # GUI stack; the headless CLI package runs without PyQt6
    from PyQt6.QtGui import QPixmap, QImage
except ImportError:  # pragma: no cover - headless openfocus package
    QPixmap = QImage = None


@lru_cache(maxsize=1)
def srgb_icc_profile() -> bytes:
    """Return an ICC profile for pixels developed into sRGB."""
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()


def pixmap_to_cv2(pixmap) -> Optional[np.ndarray]:
    try:
        qimage = pixmap.toImage()
        qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)

        width = qimage.width()
        height = qimage.height()
        bytes_per_line = qimage.bytesPerLine()

        ptr = qimage.bits()
        ptr.setsize(bytes_per_line * height)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

        bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        return bgr_image
    except Exception:
        return None


def composite_transparent_for_display(image: np.ndarray, *, origin=(0, 0), scale=(1, 1), alpha=None) -> np.ndarray:
    """Composite straight-alpha pixels, optionally supplied as a separate plane."""
    image = to_display_uint8(image)
    if alpha is None:
        if image.ndim != 3 or image.shape[2] != 4:
            return image
    else:
        if image.ndim != 3 or image.shape[2] != 3 or np.shape(alpha) != image.shape[:2]:
            raise ValueError('Separate alpha must match the BGR image dimensions')
        alpha = to_display_uint8(alpha)
    h, w = image.shape[:2]
    rows = np.floor((origin[1] + np.arange(h) * scale[1]) / 12).astype(np.int64)
    cols = np.floor((origin[0] + np.arange(w) * scale[0]) / 12).astype(np.int64)
    checker = (((rows[:, None] + cols[None, :]) % 2) * 48 + 160).astype(np.uint16)
    alpha = (image[:, :, 3] if alpha is None else alpha).astype(np.uint16)
    out = np.empty((h, w, 3), dtype=np.uint8)
    for channel in range(3):
        out[:, :, channel] = ((image[:, :, channel].astype(np.uint16) * alpha
                               + checker * (255 - alpha) + 127) // 255).astype(np.uint8)
    return out


def cv2_to_pixmap(cv2_img: np.ndarray) -> QPixmap:
    """BGR ndarray -> QPixmap, with no channel-conversion copy.

    QImage.Format_BGR888 takes OpenCV's byte order directly, so a 12 MP
    frame costs 11 ms instead of 23 ms (cvtColor BGR->RGB + RGB888).
    """
    try:
        if cv2_img.dtype != np.uint8:
            cv2_img = to_display_uint8(cv2_img)
        if cv2_img.ndim == 3 and cv2_img.shape[2] == 4:
            cv2_img = composite_transparent_for_display(cv2_img)

        if cv2_img.ndim == 3 and cv2_img.shape[2] == 3:
            bgr = np.ascontiguousarray(cv2_img)
            height, width = bgr.shape[:2]
            qimage = QImage(bgr.data, width, height, 3 * width,
                            QImage.Format.Format_BGR888)
        else:
            gray = np.ascontiguousarray(
                cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
                if cv2_img.ndim == 3 else cv2_img)
            height, width = gray.shape[:2]
            qimage = QImage(gray.data, width, height, width,
                            QImage.Format.Format_Grayscale8)

        return QPixmap.fromImage(qimage)
    except Exception:
        return QPixmap()


def get_imwrite_params(extension: str) -> list:
    """Get OpenCV imwrite parameters for maximum quality based on file extension.

    Args:
        extension: File extension (e.g., '.jpg', '.png', '.tif', '.bmp')

    Returns:
        List of parameter tuples for cv2.imwrite, or empty list if no special params needed
    """
    ext = extension.lower()
    if ext in (".jpg", ".jpeg", ".jpe", ".jfif"):
        # JPG: 100 quality (highest, default is ~95)
        return [cv2.IMWRITE_JPEG_QUALITY, 100]
    if ext in (".webp",):
        # WebP: quality > 100 means lossless
        return [cv2.IMWRITE_WEBP_QUALITY, 101]
    elif ext in (".png",):
        # PNG: 0 compression (no compression, default is 3)
        return [cv2.IMWRITE_PNG_COMPRESSION, 0]
    elif ext in (".tif", ".tiff"):
        # TIFF: compression flag 1 = no compression
        return [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    elif ext in (".bmp",):
        # BMP: always lossless, no quality parameters
        return []
    return []


def thumbnail_bgr(img: np.ndarray, size: int) -> np.ndarray:
    """Downscale first, then convert to display BGR.

    The order matters: converting a 24 MP uint16 frame costs a full-size
    temporary, while an INTER_AREA resize to 72 px costs almost nothing and
    keeps the same visual range.  Used by the output list and the retouch
    frame strip.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return np.zeros((1, 1, 3), np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = composite_transparent_for_display(img)
    h, w = img.shape[:2]
    target = max(1, int(size))
    scale = min(target / float(max(1, w)), target / float(max(1, h)), 1.0)
    if scale < 1.0:
        img = cv2.resize(
            img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA)
    display = to_display_uint8(img)
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    return display


def to_display_uint8(img: np.ndarray) -> np.ndarray:
    """Convert any supported bit depth to uint8 for on-screen display.

    uint16 is scaled by 1/257 (exactly 65535->255); float images are
    interpreted as 0-1 and clipped. uint8 passes through untouched — the
    same array is returned, no copy: every display path converts it into a
    fresh buffer (cvtColor / QPixmap.fromImage) before anything can mutate
    it, and copying a 24 MP frame per repaint was 13 ms of pure waste.

    The uint16 path goes through convertScaleAbs rather than a float32
    divide: verified identical for all 65536 input values (1/257 rounds
    half-up, which is what .round() did) and 14x faster (232 ms -> 16 ms
    on a 12 MP frame).
    """
    if img.dtype == np.uint16:
        # convertScaleAbs turns a 1-D array into a column (N,1); reshape keeps
        # the caller's shape for every rank.
        return cv2.convertScaleAbs(img, alpha=1.0 / 257.0).reshape(img.shape)
    if img.dtype != np.uint8 and img.dtype.kind == "f":
        return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if img.dtype == np.uint8:
        return img
    return img.astype(np.uint8)


def imwrite_multi_tiff(path: str, images, *, icc_profile: bytes = b"", page_specs=None) -> bool:
    """Atomically write a lazy stack with the shared bounded TIFF page writer."""
    params = [cv2.IMWRITE_TIFF_COMPRESSION, 1 if icc_profile else 5]
    return imwrite_auto(path, pages=images, params=params, exif=b'', icc_profile=icc_profile,
                        page_specs=page_specs)


def imwrite_float_tiff(path: str, image: np.ndarray, metadata: dict) -> bool:
    """Write physical depth as float32 TIFF, preserving NaN and unit metadata."""
    import json
    from PIL import Image

    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        return False
    try:
        description = Image.Exif()
        description[270] = json.dumps(metadata, ensure_ascii=True, allow_nan=False)
        return imwrite_auto(path, values, get_imwrite_params('.tif'),
                            exif=description.tobytes(), stream_tiff=True)
    except (OSError, ValueError, TypeError):
        return False


# Raw EXIF of the loaded source stack; set by the loaders, embedded into
# lossless exports automatically. Empty bytes = no metadata to embed.
SOURCE_EXIF = b""
_PNG_EXIF_LIMIT = 16 * 1024 * 1024


class SourceExifTooLarge(OSError):
    """A source image's EXIF exceeds the loader's metadata budget.

    Pure data error (no Qt imports), so headless callers can surface the
    English message and GUI callers can localize ``filename``.
    """

    def __init__(self, filename, size: int, limit: int):
        self.filename = os.fspath(filename) if filename else ""
        self.path = self.filename
        self.size = int(size)
        self.limit = int(limit)
        self.message = (
            f"EXIF metadata is too large to load: {self.size} bytes "
            f"(limit {self.limit}); remove or reduce the EXIF metadata in "
            f"{self.filename} before loading it.")
        super().__init__(self.message)

    def __str__(self):
        # OSError's default __str__ treats ``filename`` as a system-error
        # attribute and would print "[Errno None] None: ...".
        return self.message


class ImageWriteError(OSError):
    """An image write failed for a metadata/layout reason with a stable code.

    It inherits OSError so best-effort writers that already catch OSError keep
    their contract, while callers that surface exceptions can show the actual
    remediation instead of a generic "write failed".
    """

    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = str(code)
        self.detail = str(detail)


def _validate_jpeg_metadata(meta, profile: bytes) -> None:
    """Reject JPEG metadata that cannot fit before encoding any pixels."""
    exif_bytes = meta.tobytes() if meta is not None else b""
    if len(exif_bytes) + 2 > 65535:
        raise ImageWriteError(
            'exif_too_large',
            f'EXIF is too large for JPEG ({len(exif_bytes)} bytes; max 65533). '
            'Use PNG/TIFF or remove the metadata.')
    segment_size = 65519
    count = (len(profile) + segment_size - 1) // segment_size if profile else 0
    if count > 255:
        raise ImageWriteError(
            'icc_too_large',
            f'ICC profile is too large for JPEG ({len(profile)} bytes; {count} segments; '
            'max 255). Use PNG/TIFF or remove the profile.')


def set_source_exif(exif: bytes) -> None:
    global SOURCE_EXIF
    SOURCE_EXIF = exif or b""


def read_image_exif(image):
    """Read EXIF without PNG's getexif() forcing a full pixel decode."""
    if getattr(image, "format", None) != "PNG":
        return image.getexif()
    from PIL import Image

    exif = Image.Exif()
    data = image.info.get("exif")
    if not data:
        import zlib

        stream = getattr(image, "fp", None)
        if stream is None:
            return exif
        position = stream.tell()
        try:
            stream.seek(0)
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise OSError("Invalid PNG signature")
            while True:
                header = stream.read(8)
                if len(header) != 8:
                    raise OSError("Truncated PNG metadata")
                length = int.from_bytes(header[:4], "big")
                kind = header[4:]
                if kind == b"IEND":
                    return exif
                if kind == b"eXIf":
                    if length > _PNG_EXIF_LIMIT:
                        raise SourceExifTooLarge(
                            getattr(image, "filename", ""), length, _PNG_EXIF_LIMIT)
                    data = stream.read(length)
                    crc = stream.read(4)
                    if (len(data) != length or len(crc) != 4
                            or zlib.crc32(data, zlib.crc32(kind))
                            != int.from_bytes(crc, "big")):
                        raise OSError("Invalid PNG EXIF chunk")
                    break
                stream.seek(length + 4, 1)
        finally:
            stream.seek(position)
    if len(data) > _PNG_EXIF_LIMIT:
        raise SourceExifTooLarge(
            getattr(image, "filename", ""), len(data), _PNG_EXIF_LIMIT)
    exif.load(data)
    return exif


# IFD0 tags worth carrying into an export. Everything else in a source IFD0
# describes the *source* pixels (size, strips, compression, RAW sub-IFDs)
# and would corrupt or misdescribe the fused image.
_EXPORT_IFD0_TAGS = {
    270,    # ImageDescription
    271,    # Make
    272,    # Model
    282, 283, 296,  # X/YResolution, ResolutionUnit
    305,    # Software
    306,    # DateTime
    315,    # Artist
    33432,  # Copyright
}
_EXIF_IFD = 0x8769
_GPS_IFD = 0x8825
# Exif sub-IFD tags dropped on export: vendor MakerNotes can be megabytes
# (Olympus embeds previews) and some use absolute offsets that break once
# relocated; the pixel dimensions no longer match after cropping/alignment.
_EXPORT_EXIF_DROP = {37500, 40962, 40963}
# Exif 2.32 tags missing from Pillow's type table. Without these Pillow
# infers from the Python value (int -> SHORT, bytes -> BYTE, float ->
# RATIONAL), writing e.g. a negative TG-series water depth as unsigned.
_EXIF_TAG_TYPES = {
    0x8831: ("StandardOutputSensitivity", 4),
    0x8832: ("RecommendedExposureIndex", 4),
    0x8833: ("ISOSpeed", 4),
    0x9286: ("UserComment", 7),
    0x9400: ("AmbientTemperature", 10),
    0x9401: ("Humidity", 5),
    0x9402: ("Pressure", 5),
    0x9403: ("WaterDepth", 10),
    0x9404: ("Acceleration", 5),
    0x9405: ("CameraElevationAngle", 10),
    0xA300: ("FileSource", 7),
    0xA301: ("SceneType", 7),
    0xA302: ("CFAPattern", 7),
}


def _register_exif_tag_types() -> None:
    from PIL import TiffTags
    group = TiffTags.TAGS_V2_GROUPS[_EXIF_IFD]
    for tag, (name, typ) in _EXIF_TAG_TYPES.items():
        if tag not in group:  # never override Pillow's own definitions
            group[tag] = TiffTags.TagInfo(tag, name, typ, 1 if typ != 7 else 0)


def _blank(value) -> bool:
    """Placeholder values cameras write in place of omitting a tag."""
    if isinstance(value, str):
        return not value.strip("\x00 ")
    if isinstance(value, numbers.Real):
        return value != value  # 0/0 rationals load as NaN
    return False


def _sanitize_exif(src):
    """Copy export-safe tags from an already parsed Pillow Exif object."""
    from PIL import Image
    _register_exif_tag_types()
    out = Image.Exif()
    out.endian = "<"
    for tag in _EXPORT_IFD0_TAGS:
        if tag in src and not _blank(src[tag]):
            out[tag] = src[tag]
    out[274] = 1  # Orientation
    exif_ifd = {k: v for k, v in src.get_ifd(_EXIF_IFD).items()
                if k not in _EXPORT_EXIF_DROP and k != 0xA005
                and not _blank(v)}
    if exif_ifd:
        out[_EXIF_IFD] = exif_ifd
    gps_ifd = {k: v for k, v in src.get_ifd(_GPS_IFD).items()
               if not _blank(v)}
    if 2 in gps_ifd and 4 in gps_ifd:  # cameras without a fix write stubs
        out[_GPS_IFD] = gps_ifd
    return out if len(out) > 1 else None  # nothing beyond Orientation


def export_exif(exif_bytes: bytes):
    """Parse source EXIF into a PIL Exif that is safe to embed in an export.

    Keeps camera, lens, exposure, date and GPS; drops source pixel-layout
    tags and blank placeholder strings, and resets Orientation to 1 because
    the loaders already rotated the pixels. Returns None when unparsable.
    """
    if not exif_bytes:
        return None
    try:
        from PIL import Image
        src = Image.Exif()
        src.load(exif_bytes)
        return _sanitize_exif(src)
    except Exception:
        return None


def _png_insert_exif(png: bytes, tiff_exif: bytes) -> bytes:
    """Insert an eXIf chunk right after IHDR; the pixel data is untouched."""
    import struct
    import zlib
    ihdr_end = 8 + 8 + 13 + 4  # signature + IHDR length/type + data + CRC
    if png[12:16] != b"IHDR":
        raise ValueError("not a PNG produced by cv2")
    chunk = (struct.pack(">I", len(tiff_exif)) + b"eXIf" + tiff_exif
             + struct.pack(">I", zlib.crc32(b"eXIf" + tiff_exif)))
    return png[:ihdr_end] + chunk + png[ihdr_end:]


def _png_insert_icc(png: bytes, profile: bytes) -> bytes:
    """Add a PNG iCCP chunk without changing the encoded pixels."""
    import struct
    import zlib
    ihdr_end = 8 + 8 + 13 + 4
    if png[12:16] != b"IHDR":
        raise ValueError("not a PNG produced by cv2")
    data = b"ICC\x00\x00" + zlib.compress(profile)
    chunk = (struct.pack(">I", len(data)) + b"iCCP" + data
             + struct.pack(">I", zlib.crc32(b"iCCP" + data)))
    return png[:ihdr_end] + chunk + png[ihdr_end:]


def _write_encoded_jpeg(handle, encoded, exif, profile, should_cancel):
    """Write one OpenCV buffer plus metadata segments without copying that buffer."""
    import struct
    pixels = memoryview(encoded).cast('B')
    if bytes(pixels[:2]) != b'\xff\xd8':
        raise ValueError('Invalid encoded JPEG')
    exif_bytes = exif.tobytes() if exif is not None else b''
    if len(exif_bytes) + 2 > 65535:
        raise ImageWriteError('exif_too_large',
                              f'EXIF is too large for JPEG ({len(exif_bytes)} bytes; max 65533).')
    segment_size = 65519
    count = (len(profile) + segment_size - 1) // segment_size
    if count > 255:
        raise ImageWriteError('icc_too_large',
                              f'ICC profile is too large for JPEG ({len(profile)} bytes).')
    _check_write_cancelled(should_cancel)
    handle.write(pixels[:2])
    if exif_bytes:
        handle.write(b'\xff\xe1' + struct.pack('>H', len(exif_bytes) + 2))
        handle.write(exif_bytes)
    for index in range(count):
        _check_write_cancelled(should_cancel)
        data = (b'ICC_PROFILE\0' + bytes((index + 1, count))
                + profile[index * segment_size:(index + 1) * segment_size])
        handle.write(b'\xff\xe2' + struct.pack('>H', len(data) + 2))
        handle.write(data)
    for start in range(2, len(pixels), 65536):
        _check_write_cancelled(should_cancel)
        handle.write(pixels[start:start + 65536])


def _write_opencv_jpeg(handle, image, params, exif, profile, should_cancel):
    """Admit the complete native codec working set for advanced OpenCV flags.

    OpenCV can express sampling and independent luma/chroma quality settings
    that Pillow cannot preserve. This path retains those parameters verbatim;
    it is whole-image encoding and can only cancel before/after the native call.
    """
    from core.resource_policy import require_memory
    pixels = image.shape[0] * image.shape[1]
    samples = image.shape[2] if image.ndim == 3 else 1
    exif_bytes = len(exif.tobytes()) if exif is not None else 0
    # Eight-bit JPEG has at most 64 coefficients/block, bounded Huffman/amplitude
    # codes, and byte stuffing. 32 bytes/pixel plus 1 MiB covers the coded stream.
    # Reserve three such buffers for vector capacity growth / the NumPy return,
    # alongside input conversion/contiguity, coefficients and codec bookkeeping.
    encoded_bound = pixels * 32 + 1024 ** 2
    scratch = encoded_bound * 3 + pixels * (samples * 2 + 8) + 256 * 1024
    scratch += image.nbytes + exif_bytes * 3 + len(profile) * 3
    require_memory(scratch)
    _check_write_cancelled(should_cancel)
    if image.dtype == np.uint16:
        image = to_display_uint8(image)
    ok, encoded = cv2.imencode('.jpg', image, params)
    if not ok:
        raise ValueError('JPEG encoding failed')
    _check_write_cancelled(should_cancel)
    _write_encoded_jpeg(handle, encoded, exif, profile, should_cancel)


_TIFF_STRIP_TARGET_BYTES = 4 * 1024 ** 2
_BOUNDED_WRITE_THRESHOLD = 32 * 1024 ** 2
_CLASSIC_TIFF_LIMIT = 2 ** 32


def _check_write_cancelled(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def _write_row_count(shape, dtype):
    samples = shape[2] if len(shape) == 3 else 1
    row_bytes = shape[1] * samples * dtype.itemsize
    return min(128, shape[0], max(1, _TIFF_STRIP_TARGET_BYTES // row_bytes))


def _image_write_rows(image, *, scratch_factor=3, codec_scratch=0,
                      metadata_bytes=0, strip_tables=False, allow_float_gray=False):
    """Admit conversion/codec scratch independently of the complete image size."""
    from core.resource_policy import require_memory
    if (image.ndim not in (2, 3)
            or image.ndim == 3 and image.shape[2] not in (3, 4)
            or (image.dtype not in (np.uint8, np.uint16)
                and not (allow_float_gray and image.ndim == 2 and image.dtype == np.float32))
            or min(image.shape[:2]) < 1):
        raise ValueError('Unsupported pixel layout or sample type for bounded export')
    samples = image.shape[2] if image.ndim == 3 else 1
    row_bytes = image.shape[1] * samples * image.dtype.itemsize
    rows = _write_row_count(image.shape, image.dtype)
    tables = ((image.shape[0] + rows - 1) // rows * 256) if strip_tables else 0
    require_memory(row_bytes * rows * scratch_factor + codec_scratch + metadata_bytes * 6 + tables)
    return rows


def _tiff_write_rows(image, exif, profile, options):
    compressed = options.get(cv2.IMWRITE_TIFF_COMPRESSION, 1) != 1
    metadata_bytes = len(profile) + (len(exif.tobytes()) if exif is not None else 0)
    return _image_write_rows(image, scratch_factor=8 if compressed else 3,
                             codec_scratch=1024 ** 2 if compressed else 0,
                             metadata_bytes=metadata_bytes, strip_tables=True, allow_float_gray=True)


def _exif_tiff_entries(exif):
    """Reuse Pillow's scalar encodings, relocating only sanitized EXIF/GPS IFDs.

    Pillow's BigTIFF serializer does not consistently emit eight-byte next-IFD
    fields. Decode its small classic metadata directory, then lay out either
    TIFF variant explicitly. Rational, signed and undefined values stay bytes.
    """
    import struct
    if exif is None:
        return {}
    data = exif.tobytes()[6:]
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1,
             8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8}

    def directory(offset):
        entries = {}
        count = struct.unpack_from('<H', data, offset)[0]
        for index in range(count):
            pos = offset + 2 + index * 12
            tag, typ, count = struct.unpack_from('<HHI', data, pos)
            value = data[pos + 8:pos + 12]
            if tag in (_EXIF_IFD, _GPS_IFD):
                entries[tag] = (typ, 1, directory(struct.unpack('<I', value)[0]))
            else:
                length = sizes[typ] * count
                if length > 4:
                    start = struct.unpack('<I', value)[0]
                    value = data[start:start + length]
                entries[tag] = (typ, count, value[:length])
        return entries

    return directory(struct.unpack_from('<I', data, 4)[0])


def _pack_tiff_directory(entries, offset, bigtiff):
    """Serialize one IFD tree with standard TIFF/BigTIFF entry widths."""
    import struct
    size = 8 if bigtiff else 4
    count_size, entry_size = (8, 20) if bigtiff else (2, 12)
    cursor = offset + count_size + len(entries) * entry_size + size
    records, extra = [], bytearray()
    for tag, (typ, count, value) in sorted(entries.items()):
        nested = isinstance(value, dict)
        if nested or len(value) > size:
            padding = (-cursor) % size
            extra.extend(b'\0' * padding)
            cursor += padding
            payload = (_pack_tiff_directory(value, cursor, bigtiff)[0]
                       if nested else value)
            # LONG keeps common EXIF readers interoperable. Later BigTIFF
            # pages may place their metadata above 4 GiB and require LONG8.
            if nested:
                typ = 16 if cursor >= 2 ** 32 else 4
            pointer = struct.pack('<Q' if (nested and typ == 16) or (not nested and bigtiff) else '<I', cursor)
            extra.extend(payload)
            cursor += len(payload)
            value = pointer
        records.append(struct.pack('<HHQ' if bigtiff else '<HHI', tag, typ, count)
                       + value.ljust(size, b'\0'))
    body = struct.pack('<Q' if bigtiff else '<H', len(entries))
    next_pointer = offset + len(body) + len(entries) * entry_size
    extra.extend(b'\0' * ((-cursor) % size))
    return body + b''.join(records) + b'\0' * size + extra, next_pointer


def _tiff_header(shape, dtype, exif, icc_profile, rows_per_strip, *, bigtiff=False,
                 offset=0, strip_offsets=None, strip_counts=None, compression=1,
                 predictor=1, options=None):
    """Build only layout/metadata; pixel storage is always written in bands."""
    import struct
    samples = shape[2] if len(shape) == 3 else 1
    height, width = shape[:2]
    count = (height + rows_per_strip - 1) // rows_per_strip
    entries = _exif_tiff_entries(exif)

    def integers(tag, values, typ=3):
        fmt = {3: 'H', 4: 'I', 16: 'Q'}[typ]
        entries[tag] = (typ, len(values), struct.pack('<' + fmt * len(values), *values))

    integers(256, (width,), 4)
    integers(257, (height,), 4)
    integers(258, (dtype.itemsize * 8,) * samples)
    if dtype == np.float32:
        integers(339, (3,))  # SampleFormat: IEEE floating point, including NaN
    integers(259, (compression,))
    integers(262, (2 if samples > 1 else 1,))
    integers(277, (samples,))
    integers(278, (rows_per_strip,), 4)
    integers(284, (1,))
    options = options or {}
    if cv2.IMWRITE_TIFF_RESUNIT in options:
        integers(296, (options[cv2.IMWRITE_TIFF_RESUNIT],))
    for flag, tag in ((cv2.IMWRITE_TIFF_XDPI, 282), (cv2.IMWRITE_TIFF_YDPI, 283)):
        if flag in options:
            entries[tag] = (5, 1, struct.pack('<II', options[flag], 1))
    if predictor != 1:
        integers(317, (predictor,))
    if samples == 4:
        integers(338, (2,))
    if icc_profile:
        entries[34675] = (7, len(icc_profile), icc_profile)
    typ = 16 if bigtiff else 4
    integers(273, strip_offsets if strip_offsets is not None else (0,) * count, typ)
    integers(279, strip_counts if strip_counts is not None else (0,) * count, typ)
    head = (b'II+\0\x08\0\0\0' + struct.pack('<Q', 16) if bigtiff
            else b'II*\0' + struct.pack('<I', 8)) if offset == 0 else b''
    directory, next_pointer = _pack_tiff_directory(entries, offset + len(head), bigtiff)
    return head + directory, next_pointer


def _compressed_tiff_strip(strip, params):
    """Use the installed OpenCV TIFF codec on one band, including its predictor."""
    import io
    import tifffile
    options = list(params) + [getattr(cv2, 'IMWRITE_TIFF_ROWSPERSTRIP', 278), strip.shape[0]]
    ok, encoded = cv2.imencode('.tif', strip, options)
    if not ok:
        raise ValueError('TIFF band encoding failed')
    with tifffile.TiffFile(io.BytesIO(encoded)) as source:
        page = source.pages[0]
        if len(page.dataoffsets) != 1 or page.photometric not in (1, 2):
            raise ValueError('TIFF compression does not support independent RGB strips')
        start, length = page.dataoffsets[0], page.databytecounts[0]
        return bytes(memoryview(encoded)[start:start + length]), int(page.compression), int(page.predictor)


def _write_tiff_page(handle, image, exif, profile, params, should_cancel, bigtiff, rows=None):
    from core.frame_storage import frame_region
    options = dict(zip(params[::2], params[1::2]))
    rows = _tiff_write_rows(image, exif, profile, options) if rows is None else rows
    compression = options.get(cv2.IMWRITE_TIFF_COMPRESSION, 1)
    if compression not in (1, 5, 8, 32946, 32773):
        raise ValueError('Bounded TIFF supports uncompressed, LZW, Deflate and PackBits')
    offset = handle.tell()
    _check_write_cancelled(should_cancel)
    first = None
    predictor = 1
    if compression != 1:
        with frame_region(image, np.s_[:rows]) as first_strip:
            first, actual, predictor = _compressed_tiff_strip(first_strip, params)
        if actual != compression:
            raise ValueError('TIFF codec changed the requested compression')
    header, next_pointer = _tiff_header(image.shape, image.dtype, exif, profile, rows, bigtiff=bigtiff,
                                       offset=offset, compression=compression,
                                       predictor=predictor, options=options)
    handle.write(header)
    offsets, counts = [], []
    samples = image.shape[2] if image.ndim == 3 else 1
    for start in range(0, image.shape[0], rows):
        _check_write_cancelled(should_cancel)
        with frame_region(image, np.s_[start:start + rows]) as strip:
            if compression != 1:
                if start == 0:
                    pixels, first = first, None
                else:
                    pixels, actual, strip_predictor = _compressed_tiff_strip(strip, params)
                    if actual != compression or strip_predictor != predictor:
                        raise ValueError('TIFF codec changed strip layout')
            else:
                if samples > 1:
                    code = cv2.COLOR_BGRA2RGBA if samples == 4 else cv2.COLOR_BGR2RGB
                    strip = cv2.cvtColor(strip, code)
                strip = np.ascontiguousarray(strip, dtype=strip.dtype.newbyteorder('<'))
                pixels = memoryview(strip).cast('B')
            offsets.append(handle.tell())
            counts.append(len(pixels))
            if not bigtiff and handle.tell() + len(pixels) >= _CLASSIC_TIFF_LIMIT:
                raise ImageWriteError('tiff_too_large',
                                      'Classic TIFF exceeds the 4 GiB limit; use BigTIFF.')
            handle.write(pixels)
    end = handle.tell()
    header, _ = _tiff_header(image.shape, image.dtype, exif, profile, rows, bigtiff=bigtiff, offset=offset,
                             strip_offsets=offsets, strip_counts=counts,
                             compression=compression, predictor=predictor, options=options)
    handle.seek(offset)
    handle.write(header)
    handle.seek(end)
    return next_pointer


def _tiff_uses_bigtiff(page_specs, exif, profile, params, should_cancel):
    """Bound the complete classic file using metadata only, before decoding pages."""
    from math import prod
    from core.resource_policy import require_memory
    if page_specs is None:
        return True
    options = dict(zip(params[::2], params[1::2]))
    compression = options.get(cv2.IMWRITE_TIFF_COMPRESSION, 1)
    metadata_bytes = len(profile) + (len(exif.tobytes()) if exif is not None else 0)
    largest_table = 0
    for shape, dtype in page_specs:
        _check_write_cancelled(should_cancel)
        if (len(shape) not in (2, 3) or min(shape[:2]) < 1
                or len(shape) == 3 and shape[2] not in (3, 4)
                or dtype not in (np.dtype('uint8'), np.dtype('uint16'), np.dtype('float32'))):
            raise ValueError('Unsupported TIFF page specification')
        rows = _write_row_count(shape, dtype)
        largest_table = max(largest_table, (shape[0] + rows - 1) // rows * 256)
    require_memory(largest_table + metadata_bytes * 6)
    position = 0
    for index, (shape, dtype) in enumerate(page_specs):
        _check_write_cancelled(should_cancel)
        rows = _write_row_count(shape, dtype)
        strips = (shape[0] + rows - 1) // rows
        header, _ = _tiff_header(shape, dtype, exif, profile if len(shape) == 3 else b'', rows,
                                 compression=compression,
                                 predictor=2 if compression != 1 else 1, options=options)
        # Subsequent IFDs omit the file header. Four extra bytes bound the
        # difference in auxiliary alignment at their eventual absolute offsets.
        position = (position + 1) // 2 * 2
        position += len(header) if index == 0 else len(header) - 8 + 4
        raw_bytes = prod(shape) * dtype.itemsize
        position += raw_bytes if compression == 1 else raw_bytes * 2 + strips * 1024
        if position >= _CLASSIC_TIFF_LIMIT:
            return True
    return False


def _write_streamed_tiff(path, image, should_cancel=None, exif=None, icc_profile=b'',
                         params=None, allow_transparency=False, page_specs=None):
    """Atomically write pages, using classic TIFF when a complete bound fits."""
    import struct
    import tempfile
    single = isinstance(image, np.ndarray)
    pages = (image,) if single else image
    if page_specs is None and (single or isinstance(image, (list, tuple))):
        page_specs = [(pixels.shape, pixels.dtype) for pixels in pages]
    if page_specs is not None:
        page_specs = tuple((tuple(shape), np.dtype(dtype)) for shape, dtype in page_specs)
        if not page_specs:
            return False
    params = get_imwrite_params('.tif') if params is None else params
    _check_write_cancelled(should_cancel)
    bigtiff = _tiff_uses_bigtiff(page_specs, exif, icc_profile, params, should_cancel)
    fd, temporary = tempfile.mkstemp(suffix='.tif', dir=os.path.dirname(os.path.abspath(path)))
    reading_page = False
    try:
        with os.fdopen(fd, 'w+b') as handle:
            previous = None
            iterator, page_index = iter(pages), 0
            while True:
                # Source/decode errors must remain visible to the controller;
                # they are distinct from a failed codec or destination write.
                reading_page = True
                try:
                    pixels = next(iterator)
                except StopIteration:
                    reading_page = False
                    break
                reading_page = False
                _check_write_cancelled(should_cancel)
                if page_specs is not None:
                    if page_index >= len(page_specs):
                        raise ValueError('More TIFF pages than the declared layout')
                    shape, dtype = page_specs[page_index]
                    samples = pixels.shape[2] if pixels.ndim == 3 else 1
                    max_samples = shape[2] if len(shape) == 3 else 1
                    if (pixels.shape[0] > shape[0] or pixels.shape[1] > shape[1]
                            or samples > max_samples or pixels.dtype.itemsize > dtype.itemsize):
                        raise ValueError('TIFF page exceeds its declared layout bound')
                page_index += 1
                if not allow_transparency and pixels.ndim == 3 and pixels.shape[2] == 4:
                    from core.image_loader import _has_transparency
                    if _has_transparency(pixels):
                        return False
                if previous is not None:
                    padding = (-handle.tell()) % (8 if bigtiff else 2)
                    handle.write(b'\0' * padding)
                    position = handle.tell()
                    handle.seek(previous)
                    handle.write(struct.pack('<Q' if bigtiff else '<I', position))
                    handle.seek(position)
                profile = icc_profile if pixels.ndim == 3 else b''
                previous = _write_tiff_page(handle, pixels, exif, profile, params, should_cancel, bigtiff)
                del pixels
            if previous is None:
                return False
            if page_specs is not None and page_index != len(page_specs):
                raise ValueError('Fewer TIFF pages than the declared layout')
        _check_write_cancelled(should_cancel)
        os.replace(temporary, path)
        return True
    except (OSError, ValueError, cv2.error):
        if reading_page:
            raise
        return False
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_streamed_png(handle, image, params, exif, profile, should_cancel):
    """Write PNG scanline bands with zlib; never retain the encoded image."""
    import struct
    import zlib
    from core.frame_storage import frame_region
    metadata_bytes = len(profile) + (len(exif.tobytes()) if exif is not None else 0)
    rows = _image_write_rows(image, scratch_factor=5, codec_scratch=512 * 1024,
                             metadata_bytes=metadata_bytes)
    options = dict(zip(params[::2], params[1::2]))
    supported = {cv2.IMWRITE_PNG_COMPRESSION, cv2.IMWRITE_PNG_STRATEGY,
                 cv2.IMWRITE_PNG_BILEVEL}
    if options.keys() - supported or options.get(cv2.IMWRITE_PNG_BILEVEL, 0):
        raise ValueError('Bounded PNG supports standard 8/16-bit grayscale, RGB and RGBA')
    level = options.get(cv2.IMWRITE_PNG_COMPRESSION, 3)
    strategy = options.get(cv2.IMWRITE_PNG_STRATEGY,
                           zlib.Z_DEFAULT_STRATEGY if cv2.IMWRITE_PNG_COMPRESSION in options else zlib.Z_RLE)
    compressor = zlib.compressobj(level, zlib.DEFLATED, 15, 8, strategy)

    def chunk(kind, data):
        handle.write(struct.pack('>I', len(data)))
        handle.write(kind)
        handle.write(data)
        handle.write(struct.pack('>I', zlib.crc32(data, zlib.crc32(kind))))

    samples = image.shape[2] if image.ndim == 3 else 1
    handle.write(b'\x89PNG\r\n\x1a\n')
    chunk(b'IHDR', struct.pack('>IIBBBBB', image.shape[1], image.shape[0],
                               image.dtype.itemsize * 8, {1: 0, 3: 2, 4: 6}[samples], 0, 0, 0))
    if exif is not None:
        chunk(b'eXIf', exif.tobytes()[6:])
    if profile:
        chunk(b'iCCP', b'ICC\0\0' + zlib.compress(profile))
    row_bytes = image.shape[1] * samples * image.dtype.itemsize
    for start in range(0, image.shape[0], rows):
        _check_write_cancelled(should_cancel)
        with frame_region(image, np.s_[start:start + rows]) as strip:
            if samples > 1:
                strip = cv2.cvtColor(strip, cv2.COLOR_BGRA2RGBA if samples == 4 else cv2.COLOR_BGR2RGB)
            strip = np.ascontiguousarray(strip, dtype=strip.dtype.newbyteorder('>'))
            filtered = np.empty((len(strip), row_bytes + 1), np.uint8)
            filtered[:, 0] = 0  # PNG's standard None filter
            filtered[:, 1:] = strip.view(np.uint8).reshape(len(strip), row_bytes)
            encoded = compressor.compress(memoryview(filtered).cast('B'))
            if encoded:
                chunk(b'IDAT', encoded)
    encoded = compressor.flush()
    if encoded:
        chunk(b'IDAT', encoded)
    chunk(b'IEND', b'')


def _encode_mapped_jpeg(image, handle, settings):
    """Use Pillow's registered plugin handler without Image.save's pixel copy.

    Pillow's plugin save contract accepts (image, file, filename), with options
    in encoderinfo. Keep that integration here: the ordinary Image.save wrapper
    makes readonly frombuffer images mutable by copying their complete core.
    The JPEG handler only reads the mapped L/RGBX core through ImageFile._save.
    """
    from PIL import Image, JpegImagePlugin  # registers the standard JPEG handler
    writer = Image.SAVE.get(JpegImagePlugin.JpegImageFile.format)
    if writer is None:
        raise RuntimeError('Pillow does not provide a registered JPEG encoder')
    image.encoderinfo = settings
    writer(image, handle, '')


def _write_streamed_jpeg(handle, image, params, exif, profile, should_cancel):
    """Map disk-backed RGBX into Pillow's native baseline JPEG scanline encoder."""
    import mmap
    import tempfile
    from PIL import Image
    from core.frame_storage import (require_disk_space, scratch_directory, select_scratch,
                                    frame_region)
    from core.resource_policy import require_memory
    options = dict(zip(params[::2], params[1::2]))
    supported = {cv2.IMWRITE_JPEG_QUALITY, cv2.IMWRITE_JPEG_PROGRESSIVE,
                 cv2.IMWRITE_JPEG_OPTIMIZE}
    if options.keys() - supported:
        return _write_opencv_jpeg(handle, image, params, exif, profile, should_cancel)
    rows = _image_write_rows(image)
    progressive = bool(options.get(cv2.IMWRITE_JPEG_PROGRESSIVE, 0))
    optimize = bool(options.get(cv2.IMWRITE_JPEG_OPTIMIZE, 0))
    gray = image.ndim == 2
    size = image.shape[0] * image.shape[1] * (1 if gray else 4)
    exif_bytes = exif.tobytes() if exif else b''
    scratch = max(128 * 1024, image.shape[1] * 256) + rows * image.shape[1] * 12
    scratch += len(exif_bytes) * 2 + len(profile) * 4
    if progressive or optimize:
        # libjpeg retains DCT coefficients and Pillow requests a whole-image
        # compressed-output buffer for these modes. Admit both before staging.
        scratch += image.shape[0] * image.shape[1] * 8
    require_memory(scratch)
    with scratch_directory():
        directory = select_scratch(size)
        require_disk_space(size, directory)
        # RGB frombuffer copies the full image. L and RGBX map the file instead.
        with tempfile.TemporaryFile(dir=directory) as raw:
            for start in range(0, image.shape[0], rows):
                _check_write_cancelled(should_cancel)
                with frame_region(image, np.s_[start:start + rows]) as source:
                    strip = to_display_uint8(source)
                    if not gray:
                        code = cv2.COLOR_BGRA2RGBA if strip.shape[2] == 4 else cv2.COLOR_BGR2RGBA
                        strip = cv2.cvtColor(strip, code)
                    raw.write(memoryview(np.ascontiguousarray(strip)).cast('B'))
            raw.flush()
            with mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                mode = 'L' if gray else 'RGBX'
                mapped = Image.frombuffer(mode, (image.shape[1], image.shape[0]), mapping, 'raw', mode, 0, 1)

                class CancellableOutput:
                    # No fileno: Pillow must send codec blocks through write().
                    def write(self, data):
                        _check_write_cancelled(should_cancel)
                        return handle.write(data)

                    def flush(self):
                        return handle.flush()

                try:
                    _encode_mapped_jpeg(mapped, CancellableOutput(),
                                        dict(quality=options.get(cv2.IMWRITE_JPEG_QUALITY, 95),
                                             subsampling=2, exif=exif_bytes,
                                             icc_profile=profile or None,
                                             progressive=progressive, optimize=optimize))
                finally:
                    mapped.close()


def imwrite_auto(path: str, image: Optional[np.ndarray] = None, params: Optional[list] = None,
                 exif: Optional[bytes] = None, icc_profile: bytes = b"",
                 linear_dng_metadata=None, should_cancel=None,
                 linear_dng_px_size_um=None, *, stream_tiff=False,
                 pages: Optional[Iterable[np.ndarray]] = None, page_specs=None) -> bool:
    """Atomically save pixels; large TIFF/PNG/baseline JPEG select bounded codecs.

    ``pages`` accepts a lazy TIFF page iterator instead of ``image``. Known lists
    or optional ``page_specs`` (shape/dtype pairs bounding each output page) keep
    small stacks classic; unknown streams select BigTIFF upfront. ``stream_tiff``
    forces the bounded writer for a small TIFF; metadata also uses that writer.
    """
    import tempfile

    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    _check_write_cancelled(should_cancel)
    if stream_tiff and ext not in ('.tif', '.tiff'):
        raise ValueError('Streaming export requires TIFF')
    embed = SOURCE_EXIF if exif is None else exif
    jpeg_exts = ('.jpg', '.jpeg', '.jpe', '.jfif')
    meta = export_exif(embed) if ext in ('.png', '.tif', '.tiff') + jpeg_exts else None
    if pages is not None:
        if image is not None or ext not in ('.tif', '.tiff'):
            raise ValueError('TIFF pages require a TIFF destination and no single image')
        return _write_streamed_tiff(path, pages, should_cancel, meta, icc_profile, params,
                                    page_specs=page_specs)
    if not isinstance(image, np.ndarray):
        raise ValueError('A single image must be a NumPy array')
    if ext == '.dng':
        if linear_dng_metadata is None:
            raise ValueError('Linear DNG export requires camera metadata')
        from utils.linear_dng import _write_linear_dng_atomic
        try:
            _write_linear_dng_atomic(path, image, linear_dng_metadata,
                                     should_cancel=should_cancel,
                                     px_size_um=linear_dng_px_size_um)
            return True
        except OSError:
            return False
    if ext != '.png' and not stream_tiff and image.ndim == 3 and image.shape[2] == 4:
        from core.image_loader import _has_transparency
        if _has_transparency(image):
            return False
    profile = icc_profile if (icc_profile and image.ndim == 3
                              and image.shape[2] in (3, 4)) else b''
    if ext in jpeg_exts:
        _validate_jpeg_metadata(meta, profile)
    bounded = image.nbytes >= _BOUNDED_WRITE_THRESHOLD or should_cancel is not None
    if ext in ('.tif', '.tiff') and (bounded or stream_tiff or meta is not None or profile):
        options = params
        if not options:
            # Retain OpenCV's ordinary LZW default and the existing uncompressed
            # metadata/explicit-stream default. Explicit parameters always win.
            uncompressed = meta is not None or profile or stream_tiff or image.dtype == np.float32
            options = get_imwrite_params('.tif') if uncompressed else [cv2.IMWRITE_TIFF_COMPRESSION, 5]
        return _write_streamed_tiff(path, image, should_cancel, meta, profile, options,
                                    allow_transparency=True)
    advanced_jpeg = (ext in jpeg_exts and bool(params)
                     and any(flag != cv2.IMWRITE_JPEG_QUALITY for flag in params[::2]))
    part = None
    try:
        fd, part = tempfile.mkstemp(suffix=ext, dir=os.path.dirname(os.path.abspath(path)))
        with os.fdopen(fd, 'wb') as handle:
            if bounded and ext == '.png':
                _write_streamed_png(handle, image, params or [], meta, profile, should_cancel)
            elif ext in jpeg_exts and (bounded or advanced_jpeg):
                _write_streamed_jpeg(handle, image, params or [], meta, profile, should_cancel)
            else:
                if bounded:
                    # BMP/WebP and other legacy codecs still use whole-image
                    # buffers. Admit that path explicitly rather than calling it bounded.
                    from core.resource_policy import require_memory
                    require_memory(image.nbytes * 4)
                if image.dtype == np.uint16 and ext not in ('.png', '.tif', '.tiff'):
                    image = to_display_uint8(image)
                encode_ext = '.jpg' if ext in jpeg_exts else ext
                ok, encoded = cv2.imencode(encode_ext, image, params or [])
                if not ok:
                    return False
                if meta is not None and ext == '.png':
                    encoded = _png_insert_exif(bytes(encoded), meta.tobytes()[6:])
                if profile and ext == '.png':
                    encoded = _png_insert_icc(bytes(encoded), profile)
                if ext in jpeg_exts:
                    _write_encoded_jpeg(handle, encoded, meta, profile, should_cancel)
                else:
                    handle.write(encoded)
        if image.dtype == np.uint8 and ext == '.webp' and (embed or profile):
            from PIL import Image
            with Image.open(part) as source:
                source.load()
                metadata = {'lossless': True}
                if embed:
                    metadata['exif'] = _upright_exif(embed)
                if profile:
                    metadata['icc_profile'] = profile
                source.save(part, **metadata)
        _check_write_cancelled(should_cancel)
        os.replace(part, path)
        return True
    except ImageWriteError:
        raise
    except (OSError, ValueError, cv2.error):
        return False
    finally:
        if part and os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass


def _upright_exif(exif_bytes: bytes) -> bytes:
    """Return EXIF bytes with the Orientation tag reset to 1.

    Exports store upright pixels, while the source EXIF still says "rotate
    this by 90°" — embedding it unchanged makes viewers rotate the finished
    picture again. Unparseable blobs are passed through unchanged.
    """
    try:
        from PIL import Image
        exif = Image.Exif()
        exif.load(exif_bytes)
        if exif.get(274, 1) == 1:
            return exif_bytes
        exif[274] = 1
        return exif.tobytes()
    except Exception:
        return exif_bytes


def normalize_fuse_input(images):
    """Scale uint16 stacks down to float32 in the 0-255 range the back-ends use.

    Returns (frames, is_16bit) — caller re-quantizes the fused result with
    quantize_fuse_output. The 0-255 range is what every back-end already
    assumes, so their own `/ 255.0` preprocessing stays correct; keeping the
    values float rather than rounding to uint8 is what preserves the depth.
    """
    # Per-frame, not frame-0-only: one 16-bit TIFF dropped into a folder of
    # 8-bit JPEGs used to stay in the 0-65535 domain while every back-end
    # assumed 0-255, saturating the whole result to white (and the reverse
    # order crushed the 8-bit frame to black).
    is_16 = any(f.dtype == np.uint16 for f in images)
    if not is_16:
        return images, False
    out = []
    for frame in images:
        if frame.dtype == np.uint16:
            out.append(frame.astype(np.float32) * (255.0 / 65535.0))
        elif frame.dtype == np.float32:
            out.append(frame)
        else:
            out.append(frame.astype(np.float32))
    return out, True


def prepare_fusion_input(images, should_cancel=None):
    """Capture source references and output depth; fusion owns normalization."""
    from core.registration import RegistrationCancelled

    frames = []
    is_16bit = False
    for frame in images:
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()
        frames.append(frame)
        is_16bit = is_16bit or frame.dtype == np.uint16
    return frames, is_16bit


def fuse_output_dtype(images) -> type:
    """Return dtype a back-end should produce for this stack.

    8-bit stacks keep returning uint8; float stacks (from normalize_fuse_input)
    must return float32 in 0-255 so quantize_fuse_output can restore 16 bits.
    """
    if len(images) > 0 and images[0].dtype != np.uint8:
        return np.float32
    return np.uint8


def quantize_fuse_output(result, is_16bit: bool):
    """Quantize 0-255 pixels using bounded chunks, including strided views."""
    if result is None:
        return None
    if not is_16bit and result.dtype == np.uint8:
        return result
    from core.frame_storage import create_transient_frame

    pixels = np.asarray(result)
    output = create_transient_frame(pixels.shape, np.uint16 if is_16bit else np.uint8)
    # Buffered external loops cap every arithmetic temporary independently of
    # image dimensions while retaining source/output coordinates for any stride.
    with np.nditer([pixels, output], flags=['external_loop', 'buffered', 'zerosize_ok'],
                   op_flags=[['readonly'], ['writeonly']], buffersize=128 * 1024) as chunks:
        for source, destination in chunks:
            clipped = np.clip(source, 0.0, 255.0)
            if is_16bit:
                clipped *= 65535.0 / 255.0
            if clipped.dtype.kind == 'f':
                np.rint(clipped, out=clipped)
            destination[...] = clipped
    return output[()] if pixels.ndim == 0 else output


# ---------------------------------------------------------------------------
# Exposure matching (pipeline-level, before fusion)
# ---------------------------------------------------------------------------
def _exposure_mean(frame, should_cancel=None, *, normalize_uint16=False):
    """Mean of color channels, using at most 128 rows of float32 scratch.

    Deferred tiled input uses the same per-pixel uint16 normalization as the
    whole-frame preparation pass. Accumulate in float64 so strip boundaries
    and stack depth do not amplify float32 reduction error.
    """
    total = 0.0
    count = 0
    for y in range(0, frame.shape[0], 128):
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()
        strip = frame[y:y + 128]
        if strip.ndim == 3 and strip.shape[2] > 3:
            strip = strip[:, :, :3]
        strip = np.ascontiguousarray(strip, dtype=np.float32)
        if normalize_uint16 and frame.dtype == np.uint16:
            strip *= 255.0 / 65535.0
        # Contiguous float32 input lets OpenCV accumulate into doubles without
        # another layout copy or NumPy's buffered float64 casting workspace.
        total += sum(cv2.sumElems(strip))
        count += strip.size
        del strip
    return total / count if count else float('nan')


def exposure_gains(frames):
    """Per-frame gains that put every frame at the last frame's mean level.

    Non-OpenFocus methods match each frame to the last frame before focus
    measurement. Whole-stack gains keep tiled output free of exposure seams.
    """
    n = len(frames)
    if n == 0:
        return []
    means = [_exposure_mean(f) for f in frames]
    target = means[-1]
    if target <= 1e-6:
        return [1.0] * n
    return [1.0 if m <= 1e-6 else target / m for m in means]


def apply_exposure_gain(frame, gain):
    """Apply one gain to the color channels, preserving dtype and alpha.

    Alpha is opacity, not exposure: scaling it turns transparent pixels
    opaque (or vice versa) and changes the fused result.
    """
    if gain == 1.0 or frame is None:
        return frame
    dtype = frame.dtype
    limit = 65535.0 if dtype == np.uint16 else 255.0
    out = frame.astype(np.float32)
    if out.ndim == 3 and out.shape[2] == 4:
        color = out[:, :, :3]
        color *= float(gain)
        np.clip(color, 0.0, limit, out=color)
    else:
        out *= float(gain)
        np.clip(out, 0.0, limit, out=out)
    if dtype in (np.uint8, np.uint16):
        return out.astype(dtype)
    return out
