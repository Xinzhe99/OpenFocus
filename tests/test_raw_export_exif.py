"""Tests for RAW export EXIF preservation across RAW camera formats.

Verifies that camera, lens, exposure, date, and GPS EXIF metadata from DNG,
CR2, CR3, NEF, ARW, ORF, RW2, and RAF inputs are extracted with bounded header
reads and preserved cleanly when exporting fused JPEG and 16-bit TIFF outputs.
"""
from __future__ import annotations

import builtins
import io
import struct

import numpy as np
import pytest
import tifffile
from PIL import Image, TiffImagePlugin

from core import image_loader
from utils import image_utils


def _full_camera_exif(
    *,
    endian: str = "<",
    make: str = "Canon",
    model: str = "Canon EOS R5",
    lens: str = "RF100mm F2.8 L MACRO IS USM",
    focal: tuple[int, int] = (100, 1),
    fnumber: tuple[int, int] = (28, 10),
    exposure: tuple[int, int] = (1, 200),
    iso: int | tuple[int, ...] = 320,
    dt: str = "2026:10:20 09:15:00",
    orientation: int = 6,
) -> Image.Exif:
    ex = Image.Exif()
    ex.endian = endian
    ex[271] = make
    ex[272] = model
    ex[274] = orientation
    ex[256] = 8192  # source pixel tag: must be dropped on export
    ex[306] = dt
    ex[0x8769] = {
        0x829A: TiffImagePlugin.IFDRational(*exposure),
        0x829D: TiffImagePlugin.IFDRational(*fnumber),
        0x8827: iso,
        0x9003: dt,
        0x920A: TiffImagePlugin.IFDRational(*focal),
        0xA434: lens,
        0x927C: b"VENDOR_MAKERNOTE" + b"\x00" * 64,  # MakerNote must be dropped
    }
    ex[0x8825] = {1: "N", 2: (35.0, 40.0, 12.0), 3: "E", 4: (139.0, 45.0, 5.0)}
    return ex


def _build_cr3_bytes(
    *,
    make: str = "Canon",
    model: str = "Canon EOS R5",
    lens: str = "RF100mm F2.8 L MACRO IS USM",
    mdat_size: int = 0,
) -> bytes:
    def box(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I4s", len(payload) + 8, kind) + payload

    canon_uuid = bytes.fromhex("85c0b687820f11e08111f4ce462b6a48")

    cmt1 = Image.Exif()
    cmt1.endian = "<"
    cmt1[271] = make
    cmt1[272] = model
    cmt1[274] = 6
    cmt1[306] = "2026:10:20 09:15:00"

    # In CR3, CMT2 is a standalone TIFF whose root IFD stores ExifIFD tags directly.
    cmt2 = Image.Exif()
    cmt2.endian = "<"
    cmt2[0x829A] = TiffImagePlugin.IFDRational(1, 200)
    cmt2[0x829D] = TiffImagePlugin.IFDRational(28, 10)
    cmt2[0x8827] = 320
    cmt2[0x9003] = "2026:10:20 09:15:00"
    cmt2[0x920A] = TiffImagePlugin.IFDRational(100, 1)
    cmt2[0xA434] = lens

    cmt4 = Image.Exif()
    cmt4.endian = "<"
    cmt4[1] = "N"
    cmt4[2] = (35.0, 40.0, 12.0)
    cmt4[3] = "E"
    cmt4[4] = (139.0, 45.0, 5.0)

    uuid_payload = (
        canon_uuid
        + box(b"CMT1", cmt1.tobytes()[6:])
        + box(b"CMT2", cmt2.tobytes()[6:])
        + box(b"CMT4", cmt4.tobytes()[6:])
    )
    moov = box(b"moov", box(b"uuid", uuid_payload))
    ftyp = box(b"ftyp", b"crx \x00\x00\x00\x01crx isom")
    mdat = box(b"mdat", b"\x00" * mdat_size) if mdat_size else b""
    return ftyp + moov + mdat


def _build_raf_bytes(exif_obj: Image.Exif, *, sensor_pad: int = 0) -> bytes:
    jpg_buf = io.BytesIO()
    Image.new("RGB", (16, 16), (40, 80, 120)).save(
        jpg_buf, format="JPEG", exif=exif_obj.tobytes()
    )
    jpeg = jpg_buf.getvalue()
    header = bytearray(92)
    header[:16] = b"FUJIFILMCCD-RAW "
    header[28:60] = b"GFX100 II".ljust(32, b"\x00")
    header[84:88] = struct.pack(">I", 92)
    header[88:92] = struct.pack(">I", len(jpeg))
    return bytes(header) + jpeg + (b"\x00" * sensor_pad)


def _build_rw2_with_embedded_jpeg_exif(exif_obj: Image.Exif, *, pad_bytes: int = 0) -> bytes:
    """Build a Panasonic RW2 (IIU\\x00) where IFD0 holds Make/Model + tag 0x002E JpgFromRaw."""
    jpg_buf = io.BytesIO()
    Image.new("RGB", (16, 16), (50, 90, 130)).save(
        jpg_buf, format="JPEG", exif=exif_obj.tobytes()
    )
    jpeg = jpg_buf.getvalue()

    make_b = b"Panasonic\x00"
    model_b = b"DC-G9M2\x00"
    # Layout:
    # 0..8: header (IIU\x00 + offset 8)
    # 8..8+2+3*12+4 = 50: IFD0 with 3 entries (0x002E, 271, 272)
    ifd_offset = 8
    entry_count = 3
    data_start = ifd_offset + 2 + entry_count * 12 + 4
    make_off = data_start
    model_off = make_off + len(make_b)
    jpeg_off = model_off + len(model_b)

    header = b"IIU\x00" + struct.pack("<I", ifd_offset)
    entries = (
        struct.pack("<HHII", 0x002E, 7, len(jpeg), jpeg_off)
        + struct.pack("<HHII", 271, 2, len(make_b), make_off)
        + struct.pack("<HHII", 272, 2, len(model_b), model_off)
    )
    ifd = struct.pack("<H", entry_count) + entries + struct.pack("<I", 0)
    return header + ifd + make_b + model_b + jpeg + (b"\x00" * pad_bytes)


def _assert_exported_exif(
    tmp_path,
    raw_exif_bytes: bytes,
    *,
    expected_make: str,
    expected_model: str,
    expected_lens: str,
    expected_iso: int = 320,
    expect_gps: bool = True,
):
    assert raw_exif_bytes, "Expected non-empty sanitized RAW EXIF bytes"
    fused_u16 = np.full((24, 32, 3), 24000, dtype=np.uint16)
    fused_u8 = np.full((24, 32, 3), 120, dtype=np.uint8)

    for ext, img in ((".jpg", fused_u8), (".tif", fused_u16)):
        out = tmp_path / f"fused_{expected_model.replace(' ', '_')}{ext}"
        ok = image_utils.imwrite_auto(
            str(out), img, image_utils.get_imwrite_params(ext), exif=raw_exif_bytes
        )
        assert ok and out.is_file()
        with Image.open(out) as written:
            got = written.getexif()
            assert got.get(271) == expected_make
            assert got.get(272) == expected_model
            assert got.get(274) == 1
            sub = got.get_ifd(0x8769)
            assert sub.get(0xA434) == expected_lens
            assert float(sub.get(0x829D)) == pytest.approx(2.8, abs=0.05)
            assert float(sub.get(0x920A)) == pytest.approx(100.0, abs=0.5)
            iso_val = sub.get(0x8827)
            if isinstance(iso_val, tuple):
                iso_val = iso_val[0]
            assert int(iso_val) == expected_iso
            assert 0x927C not in sub  # MakerNote dropped
            if expect_gps:
                assert got.get_ifd(0x8825).get(3) == "E"


@pytest.mark.parametrize(
    "ext,endian,magic,make,model,lens",
    [
        (".dng", "<", b"II*\x00", "Leica", "SL2-S", "APO-Macro-Elmarit-R 100mm"),
        (".cr2", "<", b"II*\x00", "Canon", "Canon EOS 5D Mark IV", "EF100mm f/2.8L Macro IS USM"),
        (".nef", ">", b"MM\x00*", "NIKON CORPORATION", "NIKON Z 8", "NIKKOR Z MC 105mm f/2.8 VR S"),
        (".arw", "<", b"II*\x00", "SONY", "ILCE-7RM5", "FE 90mm F2.8 Macro G OSS"),
        (".orf", "<", b"IIRO", "OM Digital Solutions", "OM-1", "OM 90mm F3.5"),
    ],
)
def test_tiff_raw_formats_preserve_exif_on_jpeg_and_tiff_export(
    tmp_path, ext, endian, magic, make, model, lens
):
    ex = _full_camera_exif(
        endian=endian,
        make=make,
        model=model,
        lens=lens,
        iso=(320, 0) if ext == ".nef" else 320,
    )
    raw_bytes = bytearray(ex.tobytes()[6:])
    raw_bytes[:4] = magic
    raw_path = tmp_path / f"frame{ext}"
    raw_path.write_bytes(bytes(raw_bytes) + b"\x00" * 128)

    extracted = image_loader._read_exif_bytes(str(raw_path))
    _assert_exported_exif(
        tmp_path, extracted, expected_make=make, expected_model=model, expected_lens=lens
    )


def test_cr3_preserves_camera_lens_exposure_gps_without_reading_mdat(tmp_path):
    cr3_path = tmp_path / "macro.cr3"
    cr3_path.write_bytes(_build_cr3_bytes(mdat_size=4 * 1024 * 1024))

    bytes_read = []
    real_open = builtins.open

    class TrackedFile:
        def __init__(self, stream):
            self._s = stream

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self._s.__exit__(*a)

        def read(self, size=-1):
            data = self._s.read(size)
            bytes_read.append(len(data))
            return data

        def __getattr__(self, name):
            return getattr(self._s, name)

    import utils.raw_exif as raw_exif

    orig_open = getattr(raw_exif, "open", builtins.open)
    raw_exif.open = lambda *a, **k: TrackedFile(real_open(*a, **k))
    try:
        extracted = image_loader._read_exif_bytes(str(cr3_path))
    finally:
        if orig_open is builtins.open:
            delattr(raw_exif, "open")
        else:
            raw_exif.open = orig_open

    assert sum(bytes_read) < 256 * 1024  # Never reads the 4 MiB mdat payload
    _assert_exported_exif(
        tmp_path,
        extracted,
        expected_make="Canon",
        expected_model="Canon EOS R5",
        expected_lens="RF100mm F2.8 L MACRO IS USM",
    )


def test_raf_preserves_embedded_jpeg_exif_and_header_fallback(tmp_path):
    ex = _full_camera_exif(
        make="FUJIFILM",
        model="GFX100 II",
        lens="GF120mmF4 R LM OIS WR Macro",
    )
    raf_path = tmp_path / "stack_01.raf"
    raf_path.write_bytes(_build_raf_bytes(ex, sensor_pad=4 * 1024 * 1024))

    extracted = image_loader._read_exif_bytes(str(raf_path))
    _assert_exported_exif(
        tmp_path,
        extracted,
        expected_make="FUJIFILM",
        expected_model="GFX100 II",
        expected_lens="GF120mmF4 R LM OIS WR Macro",
    )

    # Header-only fallback when embedded JPEG has no EXIF.
    header = bytearray(92)
    header[:16] = b"FUJIFILMCCD-RAW "
    header[28:60] = b"X-T5".ljust(32, b"\x00")
    fallback_raf = tmp_path / "header_only.raf"
    fallback_raf.write_bytes(bytes(header) + b"\x00" * 64)
    fb_exif = Image.Exif()
    fb_exif.load(image_loader._read_exif_bytes(str(fallback_raf)))
    assert fb_exif[271] == "FUJIFILM"
    assert fb_exif[272] == "X-T5"


def test_rw2_preserves_embedded_jpeg_exif_from_tag_0x002e(tmp_path):
    ex = _full_camera_exif(
        make="Panasonic",
        model="DC-G9M2",
        lens="LEICA DG MACRO-ELMARIT 45/F2.8",
    )
    rw2_path = tmp_path / "macro.rw2"
    rw2_path.write_bytes(_build_rw2_with_embedded_jpeg_exif(ex, pad_bytes=2 * 1024 * 1024))

    extracted = image_loader._read_exif_bytes(str(rw2_path))
    _assert_exported_exif(
        tmp_path,
        extracted,
        expected_make="Panasonic",
        expected_model="DC-G9M2",
        expected_lens="LEICA DG MACRO-ELMARIT 45/F2.8",
    )


def test_bigtiff_dng_and_unique_camera_model_fallback(tmp_path):
    dng_path = tmp_path / "big.dng"
    tifffile.imwrite(
        dng_path,
        np.zeros((16, 16, 3), dtype=np.uint16),
        bigtiff=True,
        metadata=None,
        extratags=[
            (271, "s", 0, "Hasselblad", False),
            (50706, "B", 4, (1, 4, 0, 0), False),
            (50708, "s", 0, "X2D 100C", False),
        ],
    )
    extracted = image_loader._read_exif_bytes(str(dng_path))
    assert extracted
    ex = Image.Exif()
    ex.load(extracted)
    assert ex[271] == "Hasselblad"
    assert ex[272] == "X2D 100C"


def test_resilient_sanitization_keeps_camera_exif_when_one_tag_is_unserializable():
    from utils.raw_exif import sanitize_exif_resilient

    src = Image.Exif()
    src.endian = "<"
    src[271] = "SONY"
    src[272] = "ILCE-7RM5"
    src[0x8769] = {
        0x829D: TiffImagePlugin.IFDRational(28, 10),
        0x920A: TiffImagePlugin.IFDRational(90, 1),
        0xA432: (
            TiffImagePlugin.IFDRational(90, 1),
            TiffImagePlugin.IFDRational(90, 1),
            TiffImagePlugin.IFDRational(0, 0),
            TiffImagePlugin.IFDRational(0, 0),
        ),
        0xA434: "FE 90mm F2.8 Macro G OSS",
        0xFFFE: object(),  # unserializable exotic tag must not nuke valid tags
    }
    encoded = sanitize_exif_resilient(src)
    assert encoded
    parsed = Image.Exif()
    parsed.load(encoded)
    assert parsed[271] == "SONY"
    assert parsed[272] == "ILCE-7RM5"
    assert parsed.get_ifd(0x8769)[0xA434] == "FE 90mm F2.8 Macro G OSS"


def test_bounded_reader_aborts_pillow_ifd_walk_immediately_on_limit():
    """Pillow swallows OSError per tag; _BoundedReader must raise _ExifReadLimit."""
    from utils.raw_exif import _BoundedReader, _ExifReadLimit

    ex = Image.Exif()
    ex.endian = "<"
    ex[270] = "A" * 2048  # ImageDescription exceeds our tight 256-byte read budget
    ex[271] = "Canon"
    ex[272] = "Canon EOS R5"
    tiff_bytes = ex.tobytes()[6:]
    ifd_offset = struct.unpack("<I", tiff_bytes[4:8])[0]

    reads = []

    class TrackedBytesIO(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    stream = TrackedBytesIO(tiff_bytes)
    reader = _BoundedReader(stream, limit=256)
    target = Image.Exif()
    target.endian = "<"
    with pytest.raises(_ExifReadLimit) as caught:
        target.load_from_fp(reader, ifd_offset)

    assert caught.value.size > 256
    # Must abort immediately on the oversized tag without continuing to read subsequent tags.
    assert sum(sz for sz in reads if sz > 0) <= 256
    assert 272 not in target
