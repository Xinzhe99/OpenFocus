"""Unit tests for core.gallery_media (EXIF summary, watermark, WebP prep, WebP->JPG)."""
from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, TiffImagePlugin

from core.gallery_media import (
    DEFAULT_MAX_LONG_EDGE,
    DEFAULT_MAX_THUMB_BYTES,
    DEFAULT_THUMB_LONG_EDGE,
    PreparedGalleryUpload,
    convert_webp_to_jpeg,
    extract_exif_summary,
    prepare_gallery_upload,
    render_watermark,
)


def _sample_exif(
    *,
    make: str = "Sony",
    model: str = "ILCE-7RM5",
    lens: str = "FE 90mm F2.8 Macro G OSS",
    focal: tuple[int, int] = (90, 1),
    fnumber: tuple[int, int] = (28, 10),
    exposure: tuple[int, int] = (1, 250),
    iso: int | tuple[int, ...] = 400,
    dt: str = "2026:10:15 18:21:27",
    orientation: int = 1,
) -> Image.Exif:
    ex = Image.Exif()
    ex.endian = "<"
    ex[271] = make
    ex[272] = model
    ex[274] = orientation
    ex[306] = dt
    ex[0x8769] = {
        0x829A: TiffImagePlugin.IFDRational(*exposure),
        0x829D: TiffImagePlugin.IFDRational(*fnumber),
        0x8827: iso,
        0x9003: dt,
        0x920A: TiffImagePlugin.IFDRational(*focal),
        0xA434: lens,
    }
    return ex


def _make_sample_bgr(height: int, width: int, *, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.linspace(20, 235, height, dtype=np.float32)[:, None]
    x = np.linspace(30, 220, width, dtype=np.float32)[None, :]
    b = np.broadcast_to(y, (height, width))
    g = np.broadcast_to(x, (height, width))
    r = ((b + g) * 0.5)
    img = np.stack((b, g, r), axis=2).astype(np.uint8)
    noise = rng.integers(0, 16, size=(height, width, 3), dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# EXIF summary extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fmt,ext",
    [("JPEG", ".jpg"), ("TIFF", ".tif"), ("PNG", ".png"), ("WEBP", ".webp")],
)
def test_extract_exif_summary_across_standard_formats(tmp_path, fmt, ext):
    path = tmp_path / f"sample{ext}"
    Image.new("RGB", (32, 24), (40, 80, 120)).save(
        path, format=fmt, exif=_sample_exif().tobytes()
    )
    summary = extract_exif_summary(path)
    assert summary["make"] == "Sony"
    assert summary["model"] == "ILCE-7RM5"
    assert summary["camera"] == "Sony ILCE-7RM5"
    assert summary["lens"] == "FE 90mm F2.8 Macro G OSS"
    assert summary["focal_length"] == "90mm"
    assert summary["aperture"] == "f/2.8"
    assert summary["shutter"] == "1/250s"
    assert summary["iso"] == 400
    assert summary["date_time"] == "2026:10:15 18:21:27"


def test_extract_exif_summary_from_bytes_and_camera_dedup():
    ex = _sample_exif(
        make="NIKON CORPORATION",
        model="NIKON Z 8",
        lens="NIKKOR Z MC 105mm f/2.8 VR S",
        focal=(105, 1),
        fnumber=(80, 10),
        exposure=(2, 1),
        iso=(800, 0),
    )
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, format="JPEG", exif=ex.tobytes())
    summary = extract_exif_summary(buf.getvalue())
    assert summary["camera"] == "NIKON Z 8"
    assert summary["aperture"] == "f/8"
    assert summary["shutter"] == "2s"
    assert summary["iso"] == 800
    assert summary["focal_length"] == "105mm"


def test_extract_exif_summary_from_raw_orf_and_raf(tmp_path):
    # Olympus ORF
    orf_data = bytearray(_sample_exif(make="OM Digital Solutions", model="OM-1").tobytes()[6:])
    orf_data[2:4] = b"RO"
    orf_path = tmp_path / "macro.orf"
    orf_path.write_bytes(bytes(orf_data) + b"\x00" * 64)
    orf_summary = extract_exif_summary(orf_path)
    assert orf_summary["camera"] == "OM Digital Solutions OM-1"
    assert orf_summary["lens"] == "FE 90mm F2.8 Macro G OSS"
    assert orf_summary["iso"] == 400

    # Fujifilm RAF with embedded JPEG EXIF
    jpg_buf = io.BytesIO()
    Image.new("RGB", (16, 16), (30, 60, 90)).save(
        jpg_buf,
        format="JPEG",
        exif=_sample_exif(make="FUJIFILM", model="X-T5", lens="XF80mmF2.8 R LM OIS WR Macro").tobytes(),
    )
    jpg_bytes = jpg_buf.getvalue()
    header = bytearray(92)
    header[:16] = b"FUJIFILMCCD-RAW "
    header[28:60] = b"X-T5".ljust(32, b"\x00")
    header[84:88] = struct.pack(">I", 92)
    header[88:92] = struct.pack(">I", len(jpg_bytes))
    raf_path = tmp_path / "macro.raf"
    raf_path.write_bytes(bytes(header) + jpg_bytes)
    raf_summary = extract_exif_summary(raf_path)
    assert raf_summary["camera"] == "FUJIFILM X-T5"
    assert raf_summary["lens"] == "XF80mmF2.8 R LM OIS WR Macro"


def test_extract_exif_summary_graceful_on_missing_or_corrupt(tmp_path):
    plain = tmp_path / "plain.png"
    Image.new("RGB", (16, 16), (10, 10, 10)).save(plain)
    assert extract_exif_summary(plain) == {}
    assert extract_exif_summary(b"not-an-image-at-all") == {}
    assert extract_exif_summary(b"") == {}
    assert extract_exif_summary(tmp_path / "missing.jpg") == {}
    assert extract_exif_summary(None) == {}


# ---------------------------------------------------------------------------
# Watermark rendering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bg_val", [0, 255])
def test_render_watermark_legible_on_dark_and_bright_corners(bg_val):
    canvas = np.full((360, 640, 3), bg_val, dtype=np.uint8)
    original = canvas.copy()

    marked = render_watermark(canvas, "Alice Macro", enabled=True)

    # Input must never be mutated in place.
    np.testing.assert_array_equal(canvas, original)
    assert marked.shape == canvas.shape
    assert marked.dtype == canvas.dtype

    # Top-left region must stay untouched.
    np.testing.assert_array_equal(marked[:100, :200], original[:100, :200])

    # Bottom-right corner must show visible contrast on both pure black and pure white.
    br = marked[260:, 340:]
    diff = np.abs(br.astype(np.int16) - bg_val)
    assert int(diff.max()) >= 80
    assert int((diff > 30).sum()) >= 150


def test_render_watermark_noop_when_disabled_or_empty():
    canvas = _make_sample_bgr(120, 160)
    assert render_watermark(canvas, "Alice", enabled=False) is canvas
    assert render_watermark(canvas, "", enabled=True) is canvas
    assert render_watermark(canvas, "   ", enabled=True) is canvas
    assert render_watermark(canvas, "©", enabled=True) is canvas


def test_render_watermark_supports_uint16_grayscale_and_unicode():
    gray = np.full((240, 320), 220, dtype=np.uint8)
    marked_gray = render_watermark(gray, "© 微距摄影", enabled=True)
    assert marked_gray.shape == gray.shape
    assert marked_gray.dtype == np.uint8
    assert np.any(marked_gray[160:, 140:] != 220)

    u16 = np.full((240, 320, 3), 50000, dtype=np.uint16)
    marked_u16 = render_watermark(u16, "OpenFocus Creator", enabled=True)
    assert marked_u16.shape == u16.shape
    assert marked_u16.dtype == np.uint16
    assert np.any(marked_u16[160:, 120:] != 50000)


# ---------------------------------------------------------------------------
# Upload preparation (2K resize, watermark, WebP + bounded <= 256 KiB thumb)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "src_h,src_w,exp_h,exp_w",
    [
        (2000, 4000, 1280, 2560),  # landscape downscale to 2560
        (4000, 2000, 2560, 1280),  # portrait downscale to 2560
        (600, 800, 600, 800),      # smaller image: NO upscaling
    ],
)
def test_prepare_gallery_upload_resize_bounds_and_thumb(src_h, src_w, exp_h, exp_w):
    img = _make_sample_bgr(src_h, src_w)
    prepared = prepare_gallery_upload(
        img,
        attribution="Macro Artist",
        watermark_enabled=True,
        max_long_edge=DEFAULT_MAX_LONG_EDGE,
        thumb_long_edge=DEFAULT_THUMB_LONG_EDGE,
    )
    assert isinstance(prepared, PreparedGalleryUpload)
    assert (prepared.height, prepared.width) == (exp_h, exp_w)
    assert max(prepared.thumb_height, prepared.thumb_width) == DEFAULT_THUMB_LONG_EDGE
    assert prepared.full_bytes == len(prepared.main_webp)
    assert prepared.thumb_bytes == len(prepared.thumb_webp)
    assert prepared.thumb_bytes <= DEFAULT_MAX_THUMB_BYTES
    assert prepared.sha256 == hashlib.sha256(prepared.main_webp).hexdigest()
    assert prepared.thumb_sha256 == hashlib.sha256(prepared.thumb_webp).hexdigest()

    with Image.open(io.BytesIO(prepared.main_webp)) as main_im:
        assert main_im.format == "WEBP"
        assert main_im.size == (exp_w, exp_h)
    with Image.open(io.BytesIO(prepared.thumb_webp)) as thumb_im:
        assert thumb_im.format == "WEBP"
        assert thumb_im.size == (prepared.thumb_width, prepared.thumb_height)


def test_prepare_gallery_upload_watermark_toggle_and_file_exif(tmp_path):
    src = tmp_path / "shot.jpg"
    arr = np.full((400, 600, 3), 240, dtype=np.uint8)
    Image.fromarray(arr).save(src, format="JPEG", exif=_sample_exif().tobytes(), quality=95)

    with_wm = prepare_gallery_upload(src, attribution="Studio X", watermark_enabled=True)
    without_wm = prepare_gallery_upload(src, attribution="Studio X", watermark_enabled=False)

    assert with_wm.exif["camera"] == "Sony ILCE-7RM5"
    assert with_wm.exif["lens"] == "FE 90mm F2.8 Macro G OSS"
    assert with_wm.sha256 != without_wm.sha256

    dec_with = np.asarray(Image.open(io.BytesIO(with_wm.main_webp)).convert("RGB"))
    dec_without = np.asarray(Image.open(io.BytesIO(without_wm.main_webp)).convert("RGB"))
    br_diff = np.abs(dec_with[300:, 350:].astype(np.int16) - dec_without[300:, 350:].astype(np.int16))
    assert int(br_diff.max()) >= 50


def test_prepare_gallery_upload_thumb_respects_256kib_cap_on_high_noise():
    # High-entropy random noise is worst-case for WebP compression.
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 256, size=(1024, 1024, 3), dtype=np.uint8)

    prepared_default = prepare_gallery_upload(noisy, watermark_enabled=False, thumb_quality=95)
    assert prepared_default.thumb_bytes <= DEFAULT_MAX_THUMB_BYTES

    # Tight budget forces quality step-down cleanly.
    tight_budget = 16 * 1024
    prepared_tight = prepare_gallery_upload(
        noisy,
        watermark_enabled=False,
        thumb_quality=95,
        max_thumb_bytes=tight_budget,
    )
    assert prepared_tight.thumb_bytes <= tight_budget
    assert prepared_tight.thumb_quality < 95


# ---------------------------------------------------------------------------
# WebP -> JPEG conversion
# ---------------------------------------------------------------------------

def test_convert_webp_to_jpeg_preserves_exif_and_writes_atomically(tmp_path):
    ex = _sample_exif(make="Canon", model="Canon EOS R5", orientation=6)
    webp_path = tmp_path / "downloaded.webp"
    Image.new("RGB", (96, 64), (80, 120, 160)).save(
        webp_path, format="WEBP", quality=88, exif=ex.tobytes()
    )

    # Convert from path (preserving embedded WebP EXIF).
    jpg_from_path = convert_webp_to_jpeg(webp_path, tmp_path / "out_from_path.jpg", quality=94)
    assert isinstance(jpg_from_path, Path) and jpg_from_path.is_file()
    with Image.open(jpg_from_path) as im:
        assert im.format == "JPEG"
        assert im.size == (96, 64)
        got = im.getexif()
        assert got[271] == "Canon"
        assert got[272] == "Canon EOS R5"
        assert got[274] == 1  # Orientation neutralized on export
        assert got.get_ifd(0x8769)[0xA434] == "FE 90mm F2.8 Macro G OSS"

    # Convert from bytes with explicit exif_bytes override.
    override_ex = _sample_exif(make="Nikon", model="Z 8")
    jpg_from_bytes = convert_webp_to_jpeg(
        webp_path.read_bytes(),
        tmp_path / "nested" / "out_from_bytes.jpg",
        exif_bytes=override_ex.tobytes(),
    )
    with Image.open(jpg_from_bytes) as im:
        assert im.format == "JPEG"
        assert im.getexif()[271] == "Nikon"

    # Verify no leftover temp files.
    assert not list(tmp_path.rglob("*.tmp"))


def test_convert_webp_to_jpeg_handles_rgba_and_preserves_existing_on_error(tmp_path):
    rgba_buf = io.BytesIO()
    Image.new("RGBA", (32, 32), (200, 100, 50, 128)).save(rgba_buf, format="WEBP")
    out = convert_webp_to_jpeg(rgba_buf.getvalue(), tmp_path / "rgba.jpg")
    with Image.open(out) as im:
        assert im.format == "JPEG"
        assert im.mode == "RGB"

    existing = tmp_path / "keep.jpg"
    existing.write_bytes(b"original-bytes")
    with pytest.raises(Exception):
        convert_webp_to_jpeg(b"corrupt-webp-bytes", existing)
    assert existing.read_bytes() == b"original-bytes"
    assert not list(tmp_path.glob("*.tmp"))
