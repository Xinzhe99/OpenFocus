import numbers
import os
from functools import lru_cache

import cv2
import numpy as np
from typing import Optional

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


def composite_transparent_for_display(image: np.ndarray, *, origin=(0, 0), scale=(1, 1)) -> np.ndarray:
    """Composite straight-alpha pixels on a checkerboard for display only."""
    image = to_display_uint8(image)
    if image.ndim != 3 or image.shape[2] != 4:
        return image
    h, w = image.shape[:2]
    rows = np.floor((origin[1] + np.arange(h) * scale[1]) / 12).astype(np.int64)
    cols = np.floor((origin[0] + np.arange(w) * scale[0]) / 12).astype(np.int64)
    checker = (((rows[:, None] + cols[None, :]) % 2) * 48 + 160).astype(np.uint16)
    alpha = image[:, :, 3].astype(np.uint16)
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


def imwrite_multi_tiff(path: str, images, *, icc_profile: bytes = b"") -> bool:
    """Write a whole stack as ONE multi-page TIFF (uint8/uint16 preserved).

    cv2.imwritemulti needs an encodable filename, so non-ASCII targets are
    written to a .part neighbour first and renamed into place.
    """
    if not images:
        return False
    from core.image_loader import _has_transparency
    if any(_has_transparency(i) for i in images):
        return False
    import tempfile
    # cv2.imwritemulti silently fails on non-encodable filenames, so always
    # write to an ASCII temp path and rename onto the (possibly unicode)
    # destination — os.replace handles unicode targets fine.
    dest_dir = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd, part = tempfile.mkstemp(suffix=".tif", dir=dest_dir)
        os.close(fd)
    except OSError:
        return False
    try:
        if icc_profile:
            import tifffile
            total = sum(np.asarray(image).nbytes for image in images)
            with tifffile.TiffWriter(part, bigtiff=total >= (2**32 - 2**25)) as writer:
                for image in images:
                    pixels = np.asarray(image)
                    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
                        raise ValueError('RGB profile requires RGB images')
                    code = (cv2.COLOR_BGRA2RGBA if pixels.shape[2] == 4
                            else cv2.COLOR_BGR2RGB)
                    writer.write(cv2.cvtColor(pixels, code), photometric='rgb',
                                 metadata=None,
                                 extratags=[(34675, 'B', len(icc_profile), icc_profile, False)])
            os.replace(part, path)
            return True
        try:
            ok = cv2.imwritemulti(part, [np.asarray(i) for i in images])
        except cv2.error:
            ok = False
        if not ok:
            # 目标目录非 ASCII 时 mkstemp 生成的随机名也会带上它——
            # 退回 imencode+tofile（对 unicode 路径免疫）
            try:
                ok_ext, buf = cv2.imencodemulti(".tif", [np.asarray(i) for i in images])
                if ok_ext:
                    buf.tofile(part)
                    ok = True
            except (cv2.error, AttributeError):
                ok = False
        if not ok:
            return False
        os.replace(part, path)
        return True
    except (OSError, ValueError, cv2.error):
        return False
    finally:
        try:
            if os.path.exists(part):
                os.remove(part)
        except OSError:
            pass


def imwrite_float_tiff(path: str, image: np.ndarray, metadata: dict) -> bool:
    """Write physical depth as float32 TIFF, preserving NaN and unit metadata."""
    import json
    import tempfile
    from PIL import Image, TiffImagePlugin

    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        return False
    part = None
    try:
        fd, part = tempfile.mkstemp(
            suffix=".tif", dir=os.path.dirname(os.path.abspath(path)))
        os.close(fd)
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        tags[270] = json.dumps(metadata, ensure_ascii=True, allow_nan=False)
        Image.fromarray(values).save(part, format="TIFF", tiffinfo=tags)
        os.replace(part, path)
        return True
    except (OSError, ValueError, TypeError):
        return False
    finally:
        if part and os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass


# Raw EXIF of the loaded source stack; set by the loaders, embedded into
# lossless exports automatically. Empty bytes = no metadata to embed.
SOURCE_EXIF = b""
_PNG_EXIF_LIMIT = 16 * 1024 * 1024


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
                        raise OSError("PNG EXIF exceeds the read limit")
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
        raise OSError("PNG EXIF metadata exceeds the read limit")
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


def _jpeg_insert_exif(jpeg: bytes, exif: bytes) -> bytes:
    """Insert a sanitized EXIF APP1 segment into an encoded JPEG."""
    import struct
    if jpeg[:2] != b"\xff\xd8" or not exif.startswith(b"Exif\x00\x00"):
        raise ValueError("invalid JPEG or EXIF")
    if len(exif) + 2 > 65535:
        raise ValueError("EXIF is too large for JPEG")
    return jpeg[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + jpeg[2:]


def _jpeg_insert_icc(jpeg: bytes, profile: bytes) -> bytes:
    """Add the standard APP2 ICC segments without recompressing JPEG pixels."""
    import struct
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG produced by cv2")
    size = 65519
    count = (len(profile) + size - 1) // size
    if count > 255:
        raise ValueError("ICC profile is too large for JPEG")
    segments = []
    for index in range(count):
        data = b"ICC_PROFILE\x00" + bytes((index + 1, count)) + profile[index * size:(index + 1) * size]
        segments.append(b"\xff\xe2" + struct.pack(">H", len(data) + 2) + data)
    return jpeg[:2] + b"".join(segments) + jpeg[2:]


def _encode_tiff_with_exif(image: np.ndarray, exif, icc_profile: bytes = b"") -> bytes:
    """Uncompressed TIFF (uint8/uint16, gray or BGR) with an EXIF sub-IFD.

    Pillow cannot hold 16-bit RGB, so the file is assembled directly: an
    IFD0 serialised by Pillow's TIFF writer (which also lays out the Exif
    and GPS sub-IFDs) followed by the raw strips.
    """
    from PIL import TiffImagePlugin, TiffTags
    if image.ndim == 3 and image.shape[2] == 3:
        pixels = np.ascontiguousarray(image[:, :, ::-1])  # BGR -> RGB
        samples, photometric = 3, 2
    elif image.ndim == 3 and image.shape[2] == 4:
        pixels = np.ascontiguousarray(image[:, :, [2, 1, 0, 3]])
        samples, photometric = 4, 2
    elif image.ndim == 2:
        pixels = np.ascontiguousarray(image)
        samples, photometric = 1, 1
    else:
        raise ValueError("unsupported channel layout for EXIF TIFF")
    pixels = pixels.astype(pixels.dtype.newbyteorder("<"), copy=False)
    height, width = pixels.shape[:2]
    bits = pixels.dtype.itemsize * 8
    row_bytes = width * samples * pixels.dtype.itemsize
    rows_per_strip = max(1, min(height, (1 << 20) // row_bytes))
    strip_offsets, strip_counts = [], []
    for row in range(0, height, rows_per_strip):
        strip_offsets.append(row * row_bytes)
        strip_counts.append(min(rows_per_strip, height - row) * row_bytes)

    head = b"II*\x00" + (8).to_bytes(4, "little")
    ifd = TiffImagePlugin.ImageFileDirectory_v2(ifh=head)
    for tag, value in (exif or {}).items():
        ifd[tag] = value  # sub-IFDs are plain dicts (see export_exif)
    if icc_profile:
        ifd[34675] = icc_profile
    ifd[256] = width
    ifd[257] = height
    ifd[258] = (bits,) * samples
    ifd[259] = 1  # no compression
    ifd[262] = photometric
    ifd[277] = samples
    ifd[278] = rows_per_strip
    if samples == 4:
        ifd[338] = (2,)  # unassociated alpha
    ifd[284] = 1  # chunky
    # Strip offsets/counts are LONG regardless of magnitude; Pillow's writer
    # rebases StripOffsets onto the end of the IFD, where the strips go.
    ifd.tagtype[273] = ifd.tagtype[279] = TiffTags.LONG
    ifd[273] = tuple(strip_offsets)
    ifd[279] = tuple(strip_counts)
    return head + ifd.tobytes(8) + pixels.tobytes()


_TIFF_STRIP_TARGET_BYTES = 4 * 1024 ** 2


def _write_streamed_tiff(path, image, should_cancel=None):
    """Atomically write BGR/BGRA pixels in bounded strips, using BigTIFF as needed."""
    import tempfile
    import tifffile

    if (image.ndim != 3 or image.shape[2] not in (3, 4)
            or image.dtype not in (np.uint8, np.uint16)):
        raise ValueError('Streaming TIFF requires an 8- or 16-bit BGR/BGRA image')
    from core.resource_policy import require_memory

    row_bytes = image.shape[1] * image.shape[2] * image.dtype.itemsize
    rows_per_strip = min(128, image.shape[0], max(1, _TIFF_STRIP_TARGET_BYTES // row_bytes))
    # The RGB conversion and yielded bytes coexist; allow for writer buffering.
    require_memory(row_bytes * rows_per_strip * 3)

    def strips():
        for start in range(0, image.shape[0], rows_per_strip):
            if should_cancel is not None and should_cancel():
                from core.registration import RegistrationCancelled
                raise RegistrationCancelled()
            strip = image[start:start + rows_per_strip]
            if image.shape[2] == 4:
                rgb = np.concatenate((strip[:, :, 2::-1], strip[:, :, 3:4]), axis=2)
            else:
                rgb = np.ascontiguousarray(strip[:, :, ::-1])
            yield rgb.tobytes()

    fd, temporary = tempfile.mkstemp(suffix='.tif', dir=os.path.dirname(os.path.abspath(path)))
    os.close(fd)
    try:
        tifffile.imwrite(temporary, strips(), shape=image.shape, dtype=image.dtype,
                         photometric='rgb', planarconfig='contig',
                         extrasamples=('unassalpha',) if image.shape[2] == 4 else None,
                         rowsperstrip=rows_per_strip, metadata=None,
                         bigtiff=image.nbytes >= (2 ** 32 - 2 ** 25))
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()
        os.replace(temporary, path)
        return True
    except OSError:
        return False
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def imwrite_auto(path: str, image: np.ndarray, params: Optional[list] = None,
                 exif: Optional[bytes] = None, icc_profile: bytes = b"",
                 linear_dng_metadata=None, should_cancel=None,
                 linear_dng_px_size_um=None, *, stream_tiff=False) -> bool:
    """Atomically write pixels, optionally embedding their RGB color profile."""
    import tempfile

    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    if stream_tiff:
        if ext not in ('.tif', '.tiff') or exif not in (None, b'') or icc_profile:
            raise ValueError('Streaming export supports TIFF without EXIF or ICC metadata')
        return _write_streamed_tiff(path, image, should_cancel)
    if ext == '.dng':
        if linear_dng_metadata is None:
            raise ValueError("Linear DNG export requires camera metadata")
        from utils.linear_dng import _write_linear_dng_atomic
        try:
            _write_linear_dng_atomic(path, image, linear_dng_metadata,
                                     should_cancel=should_cancel,
                                     px_size_um=linear_dng_px_size_um)
            return True
        except OSError:
            return False
    if ext != '.png' and image.ndim == 3 and image.shape[2] == 4:
        from core.image_loader import _has_transparency
        if _has_transparency(image):
            return False
    if image.dtype == np.uint16 and ext not in (".png", ".tif", ".tiff"):
        image = to_display_uint8(image)
    embed = SOURCE_EXIF if exif is None else exif
    # Splice sanitized metadata into encoded bytes without recompressing pixels.
    jpeg_exts = (".jpg", ".jpeg", ".jpe", ".jfif")
    meta = export_exif(embed) if ext in (".png", ".tif", ".tiff") + jpeg_exts else None
    profile = icc_profile if (icc_profile and image.ndim == 3
                              and image.shape[2] in (3, 4)) else b""
    part = None
    try:
        encoded = None
        if (meta is not None and ext in (".png", ".tif", ".tiff")) or (profile and ext in (".tif", ".tiff")):
            try:
                if ext == ".png":
                    ok, buf = cv2.imencode(ext, image, params or [])
                    if not ok:
                        return False
                    encoded = _png_insert_exif(buf.tobytes(), meta.tobytes()[6:])
                else:
                    encoded = _encode_tiff_with_exif(image, meta, profile)
            except ValueError as exc:
                print(f"EXIF embed skipped for {path}: {exc}")
                meta = None
        if encoded is None:
            ok, encoded = cv2.imencode(ext, image, params or [])
            if not ok:
                return False
        if meta is not None and ext in jpeg_exts:
            encoded = _jpeg_insert_exif(bytes(encoded), meta.tobytes())
        if profile and ext == ".png":
            encoded = _png_insert_icc(bytes(encoded), profile)
        elif profile and ext in (".jpg", ".jpeg", ".jpe", ".jfif"):
            encoded = _jpeg_insert_icc(bytes(encoded), profile)
        fd, part = tempfile.mkstemp(
            suffix=ext, dir=os.path.dirname(os.path.abspath(path)))
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
        # Pillow re-save fallback: 8-bit only, since re-opening a 16-bit file
        # with Pillow would downgrade it. Covers WebP and unparsable EXIF.
        if image.dtype == np.uint8 and (
                (embed and meta is None and ext in (".png", ".tif", ".tiff", ".webp"))
                or (profile and ext == ".webp")):
            from PIL import Image
            with Image.open(part) as source:
                source.load()
                metadata = {}
                if embed and meta is None:
                    metadata["exif"] = _upright_exif(embed)
                if profile:
                    metadata["icc_profile"] = profile
                if ext == ".webp":
                    metadata["lossless"] = True
                source.save(part, **metadata)
        os.replace(part, path)
        return True
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


def prepare_fusion_input(images, fusion, should_cancel=None):
    """Normalize one frame at a time on disk, or defer to spatial tile crops."""
    from core.algorithms import resolve_runner
    from core.frame_storage import store_frame
    from core.registration import RegistrationCancelled
    from core.resource_policy import require_memory

    frames = list(images)
    is_16bit = any(frame.dtype == np.uint16 for frame in frames)
    spec = fusion.descriptor
    if frames and fusion.tile_enabled:
        needs_tiles = (resolve_runner(spec.tile_decider)(frames, fusion.tile_threshold)
                       if spec.tile_decider else
                       max(frames[0].shape[:2]) > fusion.tile_threshold)
        if needs_tiles:
            return frames, is_16bit
    if not is_16bit:
        return frames, False
    normalized = []
    for frame in frames:
        if should_cancel is not None and should_cancel():
            raise RegistrationCancelled()
        require_memory(frame.size * 8)
        converted = normalize_fuse_input([frame])[0][0]
        if converted is frame and frame.dtype != np.float32:
            converted = frame.astype(np.float32)
        normalized.append(frame if converted is frame else store_frame(converted))
    return normalized, True


def fuse_output_dtype(images) -> type:
    """Return dtype a back-end should produce for this stack.

    8-bit stacks keep returning uint8; float stacks (from normalize_fuse_input)
    must return float32 in 0-255 so quantize_fuse_output can restore 16 bits.
    """
    if len(images) > 0 and images[0].dtype != np.uint8:
        return np.float32
    return np.uint8


def quantize_fuse_output(result, is_16bit: bool):
    """Quantize a 0-255 fusion result back to the source bit depth."""
    if result is None:
        return None
    if is_16bit:
        return (np.clip(result, 0.0, 255.0) * (65535.0 / 255.0)).round().astype(np.uint16)
    if result.dtype == np.uint8:
        return result
    clipped = np.clip(result, 0.0, 255.0)
    if clipped.dtype.kind == "f":
        np.rint(clipped, out=clipped)
    return clipped.astype(np.uint8)


# ---------------------------------------------------------------------------
# Exposure matching (pipeline-level, before fusion)
# ---------------------------------------------------------------------------
def _exposure_mean(frame):
    """Mean of the color channels (alpha is not exposure)."""
    arr = frame
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return float(np.asarray(arr, dtype=np.float32).mean())


def exposure_gains(frames):
    """Per-frame gains that put every frame at the last frame's mean level.

    The reference A/B/C pipeline matches each frame's exposure to the last
    frame (the base reference) before the focus measure. Doing it once here,
    on the whole stack and before tiling, keeps the decision independent of
    tile boundaries; a per-tile implementation would show brightness seams.
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
        color = out[:, :, :3] * float(gain)
        np.clip(color, 0.0, limit, out=color)
        out[:, :, :3] = color
    else:
        out *= float(gain)
        np.clip(out, 0.0, limit, out=out)
    if dtype in (np.uint8, np.uint16):
        return out.astype(dtype)
    return out
