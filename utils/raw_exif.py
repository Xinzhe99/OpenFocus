"""Bounded EXIF extraction for camera RAW containers without decoding pixels.

Supports:
- TIFF-structured RAWs (DNG, CR2, NEF, ARW, ORF, RW2, PEF, SRW, BigTIFF)
- Panasonic RW2 embedded JPEG APP1 fallback (tag 0x002E / JpgFromRaw)
- Canon CR3 ISOBMFF containers (moov -> Canon UUID -> CMT1/CMT2/CMT4)
- Fujifilm RAF containers (bounded embedded JPEG APP1 scan + header fallback)
- Resilient per-tag sanitization so one exotic vendor tag never drops
  standard camera, lens, exposure, date, or GPS metadata on export.
"""
from __future__ import annotations

import io
import math
import numbers
import os
import struct
from typing import Optional

from PIL import Image as PILImage

from utils.image_utils import (
    _EXIF_IFD,
    _EXPORT_EXIF_DROP,
    _EXPORT_IFD0_TAGS,
    _GPS_IFD,
    _register_exif_tag_types,
    _sanitize_exif,
)

# TIFF-structured RAW variants (classic + Olympus/Panasonic + BigTIFF).
RAW_TIFF_MAGICS = frozenset({
    b"II*\x00",
    b"MM\x00*",
    b"IIRO",
    b"IIRS",
    b"MMOR",
    b"IIU\x00",
    b"II+\x00",
    b"MM\x00+",
})

_CANON_CR3_UUID = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")
_RAF_MAGIC = b"FUJIFILMCCD-RAW "
_MAX_EXIF_BYTES = 16 * 1024 * 1024
_MAX_CR3_BOX_BYTES = 1024 * 1024
_MAX_JPEG_HEADER_SCAN = 512 * 1024
_MAX_IFD_ENTRIES = 4096
_DNG_UNIQUE_CAMERA_MODEL_TAG = 50708
_RW2_JPG_FROM_RAW_TAG = 0x002E
_TIFF_JPEG_OFFSET_TAG = 513
_TIFF_JPEG_LENGTH_TAG = 514

_TIFF_TYPE_SIZES = {
    1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1,
    8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8,
}


class _ExifReadLimit(Exception):
    """Stop Pillow's IFD parser immediately (Pillow catches OSError per tag)."""

    def __init__(self, size: int):
        self.size = int(size)
        super().__init__(f"Bounded RAW EXIF read limit exceeded ({self.size} bytes)")


class _BoundedReader:
    """Bound total bytes read while Pillow or IFD parsers walk metadata."""

    def __init__(self, stream, limit: int = _MAX_EXIF_BYTES):
        self.stream = stream
        self.limit = int(limit)
        self.remaining = int(limit)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise _ExifReadLimit(self.limit - self.remaining)
        if size > self.remaining:
            raise _ExifReadLimit(self.limit - self.remaining + size)
        data = self.stream.read(size)
        self.remaining -= len(data)
        return data

    def seek(self, *args):
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()


def _is_blank_value(value) -> bool:
    """True for empty strings, NaN rationals, or tuples containing NaN/empty values."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip("\x00 ")
    if isinstance(value, numbers.Real):
        try:
            return not math.isfinite(float(value))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return True
    if isinstance(value, (tuple, list)):
        if not value:
            return True
        return all(_is_blank_value(item) for item in value)
    return False


def _clean_tag_value(value):
    """Normalize a tag value, stripping NaN members from rational tuples when needed."""
    if isinstance(value, tuple):
        cleaned = []
        for item in value:
            if isinstance(item, numbers.Real):
                try:
                    num = float(item)
                except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                    return None
                if not math.isfinite(num):
                    return None
            cleaned.append(item)
        return tuple(cleaned)
    return value


def _try_serialize_exif(exif_obj: PILImage.Exif) -> bytes:
    """Return serialized EXIF bytes or b'' if Pillow cannot encode the structure."""
    if len(exif_obj) <= 1:
        return b""
    try:
        return exif_obj.tobytes()
    except Exception:
        return b""


def _get_sub_ifd(src: PILImage.Exif, tag: int) -> dict:
    """Read a sub-IFD dict safely for both file-backed and in-memory Exif objects."""
    raw = src.get(tag)
    if isinstance(raw, dict):
        return dict(raw)
    try:
        return dict(src.get_ifd(tag))
    except Exception:
        return {}


def sanitize_exif_resilient(src: PILImage.Exif) -> bytes:
    """Sanitize a parsed Pillow Exif object into export-safe EXIF bytes.

    Uses ``utils.image_utils._sanitize_exif`` first for complete parity with
    OpenFocus's export policy when ``src`` is file-backed. For in-memory
    assembled Exif objects (e.g. CR3 CMT1/CMT2/CMT4 or RW2 merged EXIF) or if
    a camera RAW carries an exotic vendor tag that makes Pillow's serializer
    fail, falls back to resilient sanitization so standard camera, lens,
    exposure, date, and GPS tags always survive.
    """
    _register_exif_tag_types()
    try:
        # Fill missing Model from DNG UniqueCameraModel when available.
        if 272 not in src and _DNG_UNIQUE_CAMERA_MODEL_TAG in src:
            unique_model = src[_DNG_UNIQUE_CAMERA_MODEL_TAG]
            if isinstance(unique_model, str) and unique_model.strip("\x00 "):
                src[272] = unique_model.strip("\x00 ")
    except Exception:
        pass

    # Pillow's Exif.get_ifd() expects a file-backed fp when the sub-IFD was
    # stored directly as an in-memory dict; only use the fast _sanitize_exif
    # path when no sub-IFD is stored as a raw dict.
    has_in_memory_sub_ifd = isinstance(src.get(_EXIF_IFD), dict) or isinstance(src.get(_GPS_IFD), dict)
    if not has_in_memory_sub_ifd:
        try:
            sanitized = _sanitize_exif(src)
            if sanitized is not None:
                encoded = sanitized.tobytes()
                if encoded:
                    return encoded
        except Exception:
            pass

    # Fast batch build using _get_sub_ifd (handles in-memory dicts cleanly).
    out = PILImage.Exif()
    out.endian = "<"
    out[274] = 1
    for tag in sorted(_EXPORT_IFD0_TAGS):
        if tag in src:
            val = _clean_tag_value(src[tag])
            if val is not None and not _is_blank_value(val):
                out[tag] = val
    raw_exif_ifd = _get_sub_ifd(src, _EXIF_IFD)
    batch_exif = {}
    for k, v in raw_exif_ifd.items():
        if k in _EXPORT_EXIF_DROP or k == 0xA005 or _is_blank_value(v):
            continue
        val = _clean_tag_value(v)
        if val is not None:
            batch_exif[k] = val
    if batch_exif:
        out[_EXIF_IFD] = batch_exif
    raw_gps_ifd = _get_sub_ifd(src, _GPS_IFD)
    batch_gps = {}
    for k, v in raw_gps_ifd.items():
        if _is_blank_value(v):
            continue
        val = _clean_tag_value(v)
        if val is not None:
            batch_gps[k] = val
    if 2 in batch_gps and 4 in batch_gps:
        out[_GPS_IFD] = batch_gps
    encoded = _try_serialize_exif(out)
    if encoded:
        return encoded

    # Resilient tag-by-tag fallback: keep every valid export-safe tag.
    out = PILImage.Exif()
    out.endian = "<"
    out[274] = 1

    def _test_probe(tag, value, sub_ifd=None) -> bool:
        probe = PILImage.Exif()
        probe.endian = "<"
        probe[274] = 1
        try:
            if sub_ifd is None:
                probe[tag] = value
            else:
                probe[sub_ifd] = {tag: value}
            probe.tobytes()
            return True
        except Exception:
            return False

    for tag in sorted(_EXPORT_IFD0_TAGS):
        if tag not in src:
            continue
        try:
            raw_val = src[tag]
        except Exception:
            continue
        if _is_blank_value(raw_val):
            continue
        val = _clean_tag_value(raw_val)
        if val is not None and _test_probe(tag, val):
            out[tag] = val

    kept_exif = {}
    for k, v in raw_exif_ifd.items():
        if k in _EXPORT_EXIF_DROP or k == 0xA005 or _is_blank_value(v):
            continue
        val = _clean_tag_value(v)
        if val is not None and _test_probe(k, val, sub_ifd=_EXIF_IFD):
            kept_exif[k] = val
    if kept_exif:
        out[_EXIF_IFD] = kept_exif

    kept_gps = {}
    for k, v in raw_gps_ifd.items():
        if _is_blank_value(v):
            continue
        val = _clean_tag_value(v)
        if val is not None and _test_probe(k, val, sub_ifd=_GPS_IFD):
            kept_gps[k] = val
    if 2 in kept_gps and 4 in kept_gps:
        out[_GPS_IFD] = kept_gps

    return _try_serialize_exif(out)


def extract_jpeg_app1_exif(
    stream,
    *,
    start_offset: int = 0,
    max_length: Optional[int] = None,
    max_scan_bytes: int = _MAX_JPEG_HEADER_SCAN,
) -> bytes:
    """Scan only JPEG header markers for an APP1 ``Exif\\x00\\x00`` segment.

    Stops immediately at SOS (0xFFDA), EOI (0xFFD9), or once ``max_scan_bytes``
    have been inspected, so the embedded image scan payload is never read.
    """
    try:
        stream.seek(start_offset)
        if stream.read(2) != b"\xff\xd8":
            return b""
    except OSError:
        return b""

    scanned = 2
    limit = max_scan_bytes if max_length is None else min(max_scan_bytes, max_length)
    while scanned < limit:
        byte = stream.read(1)
        if not byte:
            return b""
        scanned += 1
        if byte != b"\xff":
            return b""
        marker_byte = stream.read(1)
        while marker_byte == b"\xff":
            scanned += 1
            if scanned >= limit:
                return b""
            marker_byte = stream.read(1)
        if not marker_byte:
            return b""
        scanned += 1
        code = marker_byte[0]
        if code in (0xD9, 0xDA):  # EOI or SOS (pixel scan begins)
            return b""
        if 0xD0 <= code <= 0xD7 or code == 0x01:  # standalone markers
            continue
        len_bytes = stream.read(2)
        if len(len_bytes) != 2:
            return b""
        scanned += 2
        seg_len = int.from_bytes(len_bytes, "big")
        if seg_len < 2:
            return b""
        payload_len = seg_len - 2
        if code == 0xE1 and payload_len >= 6 and payload_len <= 65533:
            prefix = stream.read(6)
            scanned += len(prefix)
            if prefix == b"Exif\x00\x00":
                rest = stream.read(payload_len - 6)
                if len(rest) == payload_len - 6:
                    return prefix + rest
                return b""
            skip = payload_len - len(prefix)
            if skip > 0:
                stream.seek(skip, os.SEEK_CUR)
                scanned += skip
        else:
            if scanned + payload_len > limit:
                return b""
            stream.seek(payload_len, os.SEEK_CUR)
            scanned += payload_len
    return b""


def _merge_exif_into(target: PILImage.Exif, donor: PILImage.Exif) -> None:
    """Fill missing IFD0/ExifIFD/GPSIFD tags on ``target`` from ``donor``."""
    for tag in _EXPORT_IFD0_TAGS:
        if tag not in target and tag in donor and not _is_blank_value(donor[tag]):
            target[tag] = donor[tag]
    for sub_tag in (_EXIF_IFD, _GPS_IFD):
        existing = _get_sub_ifd(target, sub_tag)
        extra = _get_sub_ifd(donor, sub_tag)
        if extra:
            merged = dict(extra)
            merged.update(existing)
            if merged:
                target[sub_tag] = merged


def _scan_classic_ifd0_minimal(stream, endian: str, ifd_offset: int, file_size: int):
    """Read classic IFD0 directory entries without eagerly loading large blobs.

    Returns ``(scalar_ifd0_dict, exif_ifd_offset, gps_ifd_offset, embedded_jpeg_range)``.
    """
    if ifd_offset < 8 or ifd_offset + 2 > file_size:
        return {}, None, None, None
    stream.seek(ifd_offset)
    raw_count = stream.read(2)
    if len(raw_count) != 2:
        return {}, None, None, None
    count = struct.unpack(endian + "H", raw_count)[0]
    if count == 0 or count > _MAX_IFD_ENTRIES or ifd_offset + 2 + count * 12 > file_size:
        return {}, None, None, None
    directory = stream.read(count * 12)
    if len(directory) != count * 12:
        return {}, None, None, None

    scalars = {}
    exif_offset = None
    gps_offset = None
    rw2_jpeg = None
    jpeg_off = None
    jpeg_len = None

    wanted_scalars = _EXPORT_IFD0_TAGS | {274, _DNG_UNIQUE_CAMERA_MODEL_TAG}
    for index in range(count):
        tag, field_type, length, data = struct.unpack_from(
            endian + "HHI4s", directory, index * 12
        )
        if tag == _EXIF_IFD and length == 1:
            exif_offset = struct.unpack(endian + "I", data)[0]
            continue
        if tag == _GPS_IFD and length == 1:
            gps_offset = struct.unpack(endian + "I", data)[0]
            continue
        if tag == _RW2_JPG_FROM_RAW_TAG and field_type in (1, 7) and length > 16:
            offset_val = struct.unpack(endian + "I", data)[0]
            if 8 <= offset_val < file_size:
                rw2_jpeg = (offset_val, min(length, file_size - offset_val))
            continue
        if tag == _TIFF_JPEG_OFFSET_TAG and length == 1:
            jpeg_off = struct.unpack(endian + "I", data)[0]
            continue
        if tag == _TIFF_JPEG_LENGTH_TAG and length == 1:
            jpeg_len = struct.unpack(endian + "I", data)[0]
            continue
        if tag not in wanted_scalars or field_type not in _TIFF_TYPE_SIZES:
            continue
        byte_len = _TIFF_TYPE_SIZES[field_type] * length
        if byte_len <= 0 or byte_len > 65536:
            continue
        if byte_len <= 4:
            raw_val = data[:byte_len]
        else:
            val_offset = struct.unpack(endian + "I", data)[0]
            if val_offset < 8 or val_offset + byte_len > file_size:
                continue
            stream.seek(val_offset)
            raw_val = stream.read(byte_len)
            if len(raw_val) != byte_len:
                continue
        if field_type == 2:  # ASCII
            text = raw_val.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
            if text:
                scalars[tag] = text
        elif field_type == 3 and length == 1:  # SHORT
            scalars[tag] = struct.unpack(endian + "H", raw_val)[0]
        elif field_type == 4 and length == 1:  # LONG
            scalars[tag] = struct.unpack(endian + "I", raw_val)[0]
        elif field_type == 5 and length == 1:  # RATIONAL
            num, den = struct.unpack(endian + "II", raw_val)
            if den:
                scalars[tag] = num / den

    embedded_jpeg = rw2_jpeg
    if embedded_jpeg is None and jpeg_off is not None and jpeg_len is not None:
        if 8 <= jpeg_off < file_size and jpeg_len > 16:
            embedded_jpeg = (jpeg_off, min(jpeg_len, file_size - jpeg_off))

    return scalars, exif_offset, gps_offset, embedded_jpeg


def _stream_size(stream) -> int:
    pos = stream.tell()
    try:
        stream.seek(0, os.SEEK_END)
        return stream.tell()
    finally:
        stream.seek(pos)


def read_tiff_raw_exif_from_stream(stream, magic: bytes) -> bytes:
    """Read sanitized EXIF from an open TIFF-based RAW stream positioned at byte 4."""
    if magic not in RAW_TIFF_MAGICS:
        return b""
    endian = "<" if magic[:2] == b"II" else ">"
    is_bigtiff = magic in (b"II+\x00", b"MM\x00+")
    if is_bigtiff:
        rest = stream.read(12)
        if len(rest) != 12:
            return b""
        offset_size, zero = struct.unpack(endian + "HH", rest[:4])
        if offset_size != 8 or zero != 0:
            return b""
        ifd_offset = struct.unpack(endian + "Q", rest[4:12])[0]
        try:
            src = PILImage.Exif()
            src.endian = endian
            src.bigtiff = True
            src.load_from_fp(_BoundedReader(stream), ifd_offset)
            return sanitize_exif_resilient(src)
        except Exception:
            return b""

    offset_bytes = stream.read(4)
    if len(offset_bytes) != 4:
        return b""
    ifd_offset = int.from_bytes(offset_bytes, "little" if endian == "<" else "big")

    src = PILImage.Exif()
    src.endian = endian
    loaded_ok = False
    try:
        src.load_from_fp(_BoundedReader(stream), ifd_offset)
        loaded_ok = bool(len(src))
    except Exception:
        loaded_ok = False

    has_exif_ifd = False
    if loaded_ok:
        try:
            has_exif_ifd = bool(src.get_ifd(_EXIF_IFD))
        except Exception:
            has_exif_ifd = False

    # Panasonic RW2 (IIU\x00) and some RAW TIFFs store full camera exposure/lens
    # EXIF inside an embedded JPEG (tag 0x002E or 513/514) rather than IFD0->0x8769,
    # or carry non-standard IFD0 tags that confuse Pillow's IFD loader.
    if not loaded_ok or not has_exif_ifd or magic == b"IIU\x00":
        try:
            file_size = _stream_size(stream)
            scalars, exif_off, gps_off, embedded_jpeg = _scan_classic_ifd0_minimal(
                stream, endian, ifd_offset, file_size
            )
            if not loaded_ok:
                src = PILImage.Exif()
                src.endian = endian
                for tag, val in scalars.items():
                    if tag in _EXPORT_IFD0_TAGS or tag == _DNG_UNIQUE_CAMERA_MODEL_TAG:
                        src[tag] = val
                if exif_off and 8 <= exif_off < file_size:
                    sub = PILImage.Exif()
                    sub.endian = endian
                    try:
                        sub.load_from_fp(_BoundedReader(stream), exif_off)
                        if len(sub):
                            src[_EXIF_IFD] = dict(sub.items())
                            has_exif_ifd = True
                    except Exception:
                        pass
                if gps_off and 8 <= gps_off < file_size:
                    sub_gps = PILImage.Exif()
                    sub_gps.endian = endian
                    try:
                        sub_gps.load_from_fp(_BoundedReader(stream), gps_off)
                        if len(sub_gps):
                            src[_GPS_IFD] = dict(sub_gps.items())
                    except Exception:
                        pass
            if not has_exif_ifd and embedded_jpeg is not None:
                jpeg_offset, jpeg_len = embedded_jpeg
                app1 = extract_jpeg_app1_exif(
                    stream, start_offset=jpeg_offset, max_length=jpeg_len
                )
                if app1:
                    donor = PILImage.Exif()
                    donor.load(app1)
                    _merge_exif_into(src, donor)
        except Exception:
            pass

    if not len(src):
        return b""
    return sanitize_exif_resilient(src)


def _iter_iso_boxes(stream, start: int, end: int):
    """Yield ``(kind, payload_start, box_end)`` within an ISOBMFF container."""
    position = start
    for _ in range(_MAX_IFD_ENTRIES):
        if position + 8 > end:
            return
        stream.seek(position)
        header = stream.read(8)
        if len(header) != 8:
            return
        length = int.from_bytes(header[:4], "big")
        kind = header[4:]
        header_size = 8
        if length == 1:
            extra = stream.read(8)
            if len(extra) != 8:
                return
            length = int.from_bytes(extra, "big")
            header_size = 16
        elif length == 0:
            length = end - position
        if length < header_size or position + length > end:
            return
        yield kind, position + header_size, position + length
        position += length


def read_cr3_exif_from_stream(stream) -> bytes:
    """Extract sanitized EXIF from a Canon CR3 ISOBMFF stream (CMT1/CMT2/CMT4).

    Seeks directly across box headers and never reads ``mdat`` sensor pixels.
    """
    try:
        file_size = _stream_size(stream)
    except OSError:
        return b""
    if file_size < 16:
        return b""

    # Verify ISOBMFF ftyp box carries Canon CR3 brand ('crx ').
    is_cr3 = False
    for kind, start, end in _iter_iso_boxes(stream, 0, min(file_size, 4096)):
        if kind == b"ftyp":
            size = min(end - start, 256)
            stream.seek(start)
            brands = stream.read(size)
            if b"crx " in brands:
                is_cr3 = True
            break
    if not is_cr3:
        return b""

    cmt_blobs = {}
    for kind, start, end in _iter_iso_boxes(stream, 0, file_size):
        if kind != b"moov":
            continue
        for child, c_start, c_end in _iter_iso_boxes(stream, start, end):
            if child != b"uuid" or c_start + 16 > c_end:
                continue
            stream.seek(c_start)
            if stream.read(16) != _CANON_CR3_UUID:
                continue
            for item, payload_start, item_end in _iter_iso_boxes(stream, c_start + 16, c_end):
                if item not in (b"CMT1", b"CMT2", b"CMT4"):
                    continue
                size = item_end - payload_start
                if not 8 <= size <= _MAX_CR3_BOX_BYTES:
                    continue
                stream.seek(payload_start)
                data = stream.read(size)
                if len(data) == size and data[:4] in (b"II*\x00", b"MM\x00*"):
                    cmt_blobs[item] = data
            if cmt_blobs:
                break
        if cmt_blobs:
            break

    if not cmt_blobs:
        return b""

    combined = PILImage.Exif()
    combined.endian = "<"

    if b"CMT1" in cmt_blobs:
        try:
            ifd0 = PILImage.Exif()
            ifd0.load(b"Exif\x00\x00" + cmt_blobs[b"CMT1"])
            for tag, val in ifd0.items():
                combined[tag] = val
            # If CMT1 itself has nested ExifIFD/GPSIFD pointers, carry them too.
            for sub in (_EXIF_IFD, _GPS_IFD):
                sub_dict = dict(ifd0.get_ifd(sub))
                if sub_dict:
                    combined[sub] = sub_dict
        except Exception:
            pass

    if b"CMT2" in cmt_blobs:
        try:
            exif_box = PILImage.Exif()
            exif_box.load(b"Exif\x00\x00" + cmt_blobs[b"CMT2"])
            exif_tags = dict(exif_box.items())
            if exif_tags:
                existing = _get_sub_ifd(combined, _EXIF_IFD)
                existing.update(exif_tags)
                combined[_EXIF_IFD] = existing
        except Exception:
            pass

    if b"CMT4" in cmt_blobs:
        try:
            gps_box = PILImage.Exif()
            gps_box.load(b"Exif\x00\x00" + cmt_blobs[b"CMT4"])
            gps_tags = dict(gps_box.items())
            if gps_tags:
                existing_gps = _get_sub_ifd(combined, _GPS_IFD)
                existing_gps.update(gps_tags)
                combined[_GPS_IFD] = existing_gps
        except Exception:
            pass

    if not len(combined):
        return b""
    return sanitize_exif_resilient(combined)


def _raf_header_make_model(header: bytes) -> Optional[tuple[str, str]]:
    if len(header) < 92 or header[:16] != _RAF_MAGIC:
        return None
    field = header[28:60]
    model, separator, padding = field.partition(b"\x00")
    if separator and padding.strip(b"\x00 "):
        return None
    model = model.strip(b" ")
    if (
        not model
        or any(byte < 32 or byte > 126 for byte in model)
        or not any(
            48 <= byte <= 57 or 65 <= byte <= 90 or 97 <= byte <= 122
            for byte in model
        )
    ):
        return None
    return "FUJIFILM", model.decode("ascii")


def read_raf_exif_from_stream(stream) -> bytes:
    """Extract sanitized EXIF from a Fujifilm RAF stream without reading pixels.

    Reads the 92-byte fixed header, seeks to the embedded JPEG preview offset,
    and scans only JPEG APP1 EXIF markers (stopping at SOS). Falls back to the
    fixed-header camera model when the preview carries no EXIF.
    """
    try:
        file_size = _stream_size(stream)
        stream.seek(0)
        header = stream.read(92)
    except OSError:
        return b""
    if len(header) != 92 or header[:16] != _RAF_MAGIC:
        return b""

    jpeg_offset = int.from_bytes(header[84:88], "big")
    jpeg_length = int.from_bytes(header[88:92], "big")
    if 92 <= jpeg_offset < file_size and 4 <= jpeg_length <= file_size - jpeg_offset:
        app1 = extract_jpeg_app1_exif(
            stream, start_offset=jpeg_offset, max_length=jpeg_length
        )
        if app1:
            try:
                src = PILImage.Exif()
                src.load(app1)
                encoded = sanitize_exif_resilient(src)
                if encoded:
                    return encoded
            except Exception:
                pass

    identity = _raf_header_make_model(header)
    if identity is not None:
        make, model = identity
        fallback = PILImage.Exif()
        fallback.endian = "<"
        fallback[271] = make
        fallback[272] = model
        fallback[274] = 1
        return sanitize_exif_resilient(fallback)
    return b""


def read_container_raw_exif(path: str | os.PathLike, magic: bytes = b"") -> bytes:
    """Read EXIF from non-TIFF RAW containers (Fujifilm RAF, Canon CR3)."""
    try:
        with open(path, "rb") as stream:
            head = magic if len(magic) == 4 else stream.read(4)
            if head == b"FUJI":
                return read_raf_exif_from_stream(stream)
            stream.seek(4)
            box_type = stream.read(4)
            if box_type == b"ftyp":
                return read_cr3_exif_from_stream(stream)
    except Exception:
        return b""
    return b""


def read_raw_exif_bytes(path_or_bytes) -> bytes:
    """Read export-safe EXIF bytes from any supported RAW file path or buffer."""
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        stream = io.BytesIO(bytes(path_or_bytes))
        return _read_raw_exif_from_stream(stream)
    try:
        with open(os.fspath(path_or_bytes), "rb") as stream:
            return _read_raw_exif_from_stream(stream)
    except Exception:
        return b""


def _read_raw_exif_from_stream(stream) -> bytes:
    magic = stream.read(4)
    if len(magic) != 4:
        return b""
    if magic in RAW_TIFF_MAGICS:
        return read_tiff_raw_exif_from_stream(stream, magic)
    if magic == b"FUJI":
        return read_raf_exif_from_stream(stream)
    box_type = stream.read(4)
    if box_type == b"ftyp":
        return read_cr3_exif_from_stream(stream)
    return b""
