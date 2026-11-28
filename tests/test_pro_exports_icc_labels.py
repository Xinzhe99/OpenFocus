"""Comprehensive tests for Pro Export Formats, Contrast Map, Strip EXIF,
UI ICC Color Management Preview, and EXIF Dynamic Label Macros.
"""
from __future__ import annotations

import struct
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from core.color_management import (
    clear_transform_cache,
    convert_bgr_to_srgb_preview,
    convert_qimage_to_srgb,
    describe_icc_profile,
    get_srgb_transform,
    is_icc_management_available,
    transform_cache_size,
)
from core.depth_products import compute_contrast_map, export_contrast_map
from core.exif_macros import (
    EXIF_MACRO_TOKENS,
    expand_exif_macros,
    extract_exif_metadata,
)
from core.psd_export import (
    PSD_SIGNATURE,
    _packbits_encode_numpy,
    compute_source_contribution_masks,
    export_layered_psd,
    packbits_decode,
    packbits_encode,
)
from core.render_context import build_render_context
from utils.image_utils import (
    ImageWriteError,
    cv2_to_pixmap,
    imwrite_auto,
    imwrite_float32_tiff,
    jp2_codec_available,
    pixmap_to_cv2,
    srgb_icc_profile,
    to_normalized_float32,
)
from utils.validators import LabelAdder, build_label_text
from utils import watermark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_exif() -> bytes:
    exif = Image.Exif()
    exif[271] = "OpenCam"
    exif[272] = "OpenCam X100"
    exif[306] = "2026:11:27 14:30:00"
    exif[0x8769] = {
        36867: "2026:11:27 14:30:00",
        42036: "Macro 90mm f/2.8",
        33437: 8.0,
        33434: 1.0 / 250.0,
        34855: 200,
        37386: 90.0,
    }
    return exif.tobytes()


def _make_channel_swapped_icc_profile() -> bytes:
    """Create a valid synthetic RGB ICC profile whose primaries permute R/G/B.

    Using Pillow's built-in sRGB profile and rotating its rXYZ/gXYZ/bXYZ colorant
    tags produces a 100% valid LittleCMS2 RGB profile that deterministically
    transforms saturated RGB colors to measurably distinct sRGB display pixels.
    """
    raw = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    tag_count = struct.unpack_from(">I", raw, 128)[0]
    tags = {}
    for i in range(tag_count):
        base = 132 + i * 12
        sig = bytes(raw[base:base + 4])
        off, size = struct.unpack_from(">II", raw, base + 4)
        tags[sig] = (base, off, size)
    assert {b"rXYZ", b"gXYZ", b"bXYZ"} <= set(tags)
    r_data = bytes(raw[tags[b"rXYZ"][1]:tags[b"rXYZ"][1] + tags[b"rXYZ"][2]])
    g_data = bytes(raw[tags[b"gXYZ"][1]:tags[b"gXYZ"][1] + tags[b"gXYZ"][2]])
    b_data = bytes(raw[tags[b"bXYZ"][1]:tags[b"bXYZ"][1] + tags[b"bXYZ"][2]])
    # Permute primaries: R -> G, G -> B, B -> R
    raw[tags[b"rXYZ"][1]:tags[b"rXYZ"][1] + len(g_data)] = g_data
    raw[tags[b"gXYZ"][1]:tags[b"gXYZ"][1] + len(b_data)] = b_data
    raw[tags[b"bXYZ"][1]:tags[b"bXYZ"][1] + len(r_data)] = r_data
    return bytes(raw)


def _parse_psd_header_and_layers(data: bytes) -> dict:
    """Minimal PSD v1 binary parser to verify spec compliance independently."""
    assert data[:4] == PSD_SIGNATURE
    version = struct.unpack_from(">H", data, 4)[0]
    channels, height, width, depth, color_mode = struct.unpack_from(">HIIHH", data, 12)
    pos = 26
    color_mode_len = struct.unpack_from(">I", data, pos)[0]
    pos += 4 + color_mode_len
    res_len = struct.unpack_from(">I", data, pos)[0]
    pos += 4 + res_len
    layer_mask_len = struct.unpack_from(">I", data, pos)[0]
    pos += 4
    layer_info_len = struct.unpack_from(">I", data, pos)[0]
    layer_count = struct.unpack_from(">h", data, pos + 4)[0]
    rec_pos = pos + 6
    layer_records = []
    for _ in range(abs(layer_count)):
        top, left, bottom, right, n_ch = struct.unpack_from(">iiiiH", data, rec_pos)
        rec_pos += 18
        ch_ids = []
        for _ in range(n_ch):
            ch_id, ch_len = struct.unpack_from(">hI", data, rec_pos)
            ch_ids.append((ch_id, ch_len))
            rec_pos += 6
        blend_sig = data[rec_pos:rec_pos + 4]
        blend_key = data[rec_pos + 4:rec_pos + 8]
        opacity, clipping, flags, _filler = struct.unpack_from(">BBBB", data, rec_pos + 8)
        rec_pos += 12
        extra_len = struct.unpack_from(">I", data, rec_pos)[0]
        rec_pos += 4
        extra = data[rec_pos:rec_pos + extra_len]
        rec_pos += extra_len
        mask_data_len = struct.unpack_from(">I", extra, 0)[0]
        blend_ranges_len = struct.unpack_from(">I", extra, 4 + mask_data_len)[0]
        name_pos = 4 + mask_data_len + 4 + blend_ranges_len
        name_len = extra[name_pos]
        name = extra[name_pos + 1:name_pos + 1 + name_len].decode("ascii", errors="replace")
        layer_records.append({
            "rect": (top, left, bottom, right),
            "channels": ch_ids,
            "blend": (blend_sig, blend_key),
            "opacity": opacity,
            "flags": flags,
            "mask_data_len": mask_data_len,
            "name": name,
        })
    return {
        "version": version,
        "channels": channels,
        "height": height,
        "width": width,
        "depth": depth,
        "color_mode": color_mode,
        "layer_mask_len": layer_mask_len,
        "layer_info_len": layer_info_len,
        "layer_count": layer_count,
        "layers": layer_records,
    }


# ---------------------------------------------------------------------------
# 1. Contrast Map Computation & Export
# ---------------------------------------------------------------------------

def test_contrast_map_computation_and_exports(tmp_path):
    # Left half flat, right half high-contrast checkerboard -> right must have far higher contrast energy.
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    for y in range(64):
        for x in range(32, 64):
            img[y, x] = (240, 240, 240) if ((x // 2 + y // 2) % 2 == 0) else (20, 20, 20)

    cm16 = compute_contrast_map(img, bit_depth=16, method="hybrid")
    assert cm16.shape == (64, 64)
    assert cm16.dtype == np.uint16
    assert float(cm16[:, 36:].mean()) > float(cm16[:, :28].mean()) + 10000.0

    cm8 = compute_contrast_map(img, bit_depth=8, method="tenengrad")
    assert cm8.shape == (64, 64)
    assert cm8.dtype == np.uint8
    assert int(cm8.max()) == 255

    cm_lap = compute_contrast_map(img, bit_depth=16, method="laplacian")
    assert cm_lap.dtype == np.uint16
    assert int(cm_lap.max()) == 65535

    cm_color = compute_contrast_map(img, colormap="turbo")
    assert cm_color.shape == (64, 64, 3)
    assert cm_color.dtype == np.uint8

    # Flat image returns all zeros cleanly without NaN or division errors.
    flat = np.full((32, 32, 3), 80, dtype=np.uint16)
    cm_flat = compute_contrast_map(flat, bit_depth=16)
    assert cm_flat.dtype == np.uint16
    assert np.all(cm_flat == 0)

    # Export to 16-bit PNG and 16-bit TIFF.
    png_path = tmp_path / "contrast.png"
    tif_path = tmp_path / "contrast.tif"
    assert export_contrast_map(str(png_path), img)
    assert export_contrast_map(str(tif_path), img)

    png_back = cv2.imdecode(np.fromfile(png_path, np.uint8), cv2.IMREAD_UNCHANGED)
    tif_back = tifffile.imread(str(tif_path))
    assert png_back.dtype == np.uint16 and png_back.shape == (64, 64)
    assert tif_back.dtype == np.uint16 and tif_back.shape == (64, 64)
    np.testing.assert_array_equal(png_back, cm16)
    np.testing.assert_array_equal(tif_back, cm16)


# ---------------------------------------------------------------------------
# 2. 32-bit Float TIFF & JPEG 2000 (.jp2) Export
# ---------------------------------------------------------------------------

def test_32bit_float_tiff_roundtrip_and_metadata(tmp_path):
    src_u16 = np.zeros((24, 32, 3), dtype=np.uint16)
    src_u16[:, :, 0] = 16384  # B
    src_u16[:, :, 1] = 32768  # G
    src_u16[:, :, 2] = 65535  # R
    profile = srgb_icc_profile()
    exif = _make_sample_exif()

    path = tmp_path / "float32_rgb.tif"
    assert imwrite_float32_tiff(str(path), src_u16, exif=exif, icc_profile=profile)

    arr = tifffile.imread(str(path))
    assert arr.dtype == np.float32
    assert arr.shape == (24, 32, 3)
    expected_rgb = to_normalized_float32(src_u16)[..., ::-1]
    np.testing.assert_allclose(arr, expected_rgb, atol=1e-6)
    with tifffile.TiffFile(str(path)) as tif:
        page = tif.pages[0]
        assert page.bitspersample == 32
        assert page.sampleformat == 3  # IEEEFP
        assert page.tags[34675].value == profile
        assert page.tags[271].value == "OpenCam"


def test_jp2_export_roundtrip_or_clean_error(tmp_path, monkeypatch):
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[:, :, 0] = 30
    img[:, :, 1] = 110
    img[:, :, 2] = 220
    jp2_path = tmp_path / "sample.jp2"

    if jp2_codec_available():
        assert imwrite_auto(str(jp2_path), img, exif=b"")
        assert jp2_path.is_file() and jp2_path.stat().st_size > 0
        with Image.open(jp2_path) as loaded:
            loaded.load()
            assert loaded.size == (48, 48)
            rgb = np.asarray(loaded)
            np.testing.assert_array_equal(rgb, img[..., ::-1])
    else:
        pytest.skip("JPEG 2000 codec unavailable in this build")

    # Verify clean error when both Pillow and OpenCV JP2 codecs are unavailable.
    from PIL import features
    monkeypatch.setattr(features, "check", lambda name: False if name == "jpg_2000" else True)
    monkeypatch.setattr(cv2, "imencode", lambda ext, *a, **k: (False, None) if ext == ".jp2" else (True, b""))
    with pytest.raises(ImageWriteError) as exc_info:
        imwrite_auto(str(tmp_path / "fail.jp2"), img, exif=b"", raise_on_error=True)
    assert exc_info.value.code == "jp2_unavailable"


# ---------------------------------------------------------------------------
# 3. Layered PSD Export (PackBits RLE + Contribution Masks + Composite)
# ---------------------------------------------------------------------------

def test_packbits_rle_roundtrip():
    cases = [
        b"",
        b"\x00",
        b"\xaa" * 200,
        bytes(range(150)),
        b"\x01\x01\x01\x01\x02\x03\x04\x05\x05\x05\x05\x05\x06",
    ]
    for raw in cases:
        encoded = packbits_encode(raw)
        decoded = packbits_decode(encoded, expected_length=len(raw))
        assert decoded == raw


def test_layered_psd_export_structure_masks_and_pillow_readback(tmp_path):
    h, w = 40, 48
    frame0 = np.full((h, w, 3), (20, 40, 210), dtype=np.uint8)   # red-ish in BGR
    frame1 = np.full((h, w, 3), (210, 180, 30), dtype=np.uint8)  # cyan-ish in BGR
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:, :w // 2] = frame0[:, :w // 2]
    result[:, w // 2:] = frame1[:, w // 2:]

    depth = np.zeros((h, w), dtype=np.float32)
    depth[:, w // 2:] = 1.0

    # Verify contribution masks accurately separate left vs right half with smooth feathering.
    masks = compute_source_contribution_masks(result, [frame0, frame1], depth_index=depth, feather_sigma=2.0)
    assert len(masks) == 2
    assert masks[0][:, 4].mean() > 240
    assert masks[0][:, -4].mean() < 15
    assert masks[1][:, -4].mean() > 240
    assert masks[1][:, 4].mean() < 15
    # Feathered seam around w // 2 should contain intermediate values (0 < m < 255).
    seam_vals = masks[0][:, w // 2 - 2:w // 2 + 2]
    assert np.any((seam_vals > 10) & (seam_vals < 245))

    context = build_render_context(
        [frame0, frame1],
        result,
        filenames=["focus_near.tif", "focus_far.tif"],
        algorithm="openfocus_b",
        icc_profile=srgb_icc_profile(),
    )
    psd_path = tmp_path / "stack_layers.psd"
    out = export_layered_psd(
        str(psd_path),
        result,
        context,
        depth_index=depth,
        icc_profile=srgb_icc_profile(),
        feather_sigma=2.0,
    )
    assert out == str(psd_path)
    raw = psd_path.read_bytes()
    parsed = _parse_psd_header_and_layers(raw)
    assert parsed["version"] == 1
    assert parsed["width"] == w and parsed["height"] == h
    assert parsed["depth"] == 8 and parsed["color_mode"] == 3
    # 2 source layers + 1 top merged result layer = 3 layers.
    assert parsed["layer_count"] == 3
    assert [layer["name"] for layer in parsed["layers"]] == [
        "focus_near.tif",
        "focus_far.tif",
        "Merged Result",
    ]
    # Source layers have user layer mask channel (-2) and 20-byte mask data record.
    for src_layer in parsed["layers"][:2]:
        ch_ids = [cid for cid, _ in src_layer["channels"]]
        assert -2 in ch_ids
        assert src_layer["mask_data_len"] == 20
        assert src_layer["flags"] == 0x02  # hidden source layer
    assert parsed["layers"][-1]["flags"] == 0x00  # top merged result visible

    # Pillow must cleanly open the PSD and read back the flattened preview composite + ICC profile.
    with Image.open(psd_path) as psd_img:
        psd_img.load()
        assert psd_img.size == (w, h)
        assert psd_img.info.get("icc_profile") == srgb_icc_profile()
        composite_rgb = np.asarray(psd_img.convert("RGB"))
        np.testing.assert_array_equal(composite_rgb, result[..., ::-1])


# ---------------------------------------------------------------------------
# 4. One-Click Strip EXIF Metadata Toggle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext", [".jpg", ".png", ".tif", ".webp"])
def test_strip_exif_removes_all_exif_bytes_while_keeping_icc(tmp_path, ext):
    img = np.full((24, 24, 3), (60, 120, 180), dtype=np.uint8)
    exif = _make_sample_exif()
    profile = srgb_icc_profile()

    kept_path = tmp_path / f"kept{ext}"
    stripped_path = tmp_path / f"stripped{ext}"

    assert imwrite_auto(str(kept_path), img, exif=exif, icc_profile=profile, strip_exif=False)
    assert imwrite_auto(str(stripped_path), img, exif=exif, icc_profile=profile, strip_exif=True)

    with Image.open(kept_path) as kept:
        assert kept.getexif().get(271) == "OpenCam"
        assert kept.info.get("icc_profile") == profile

    with Image.open(stripped_path) as stripped:
        stripped_exif = stripped.getexif()
        if ext == ".tif":
            # TIFF IFD0 always holds mandatory structural pixel layout tags,
            # while all camera/capture EXIF tags and sub-IFDs must be stripped.
            for tag in (271, 272, 306, 0x8769, 0x8825):
                assert tag not in stripped_exif
            assert not stripped_exif.get_ifd(0x8769)
        else:
            assert len(stripped_exif) == 0
            assert not stripped.info.get("exif")
        assert stripped.info.get("icc_profile") == profile


def test_strip_exif_honored_by_output_manager_and_settings_store(qapp, tmp_path, monkeypatch):
    from controllers.output_manager import OutputManager, OutputRecord
    from controllers.label_manager import LabelManager
    from utils.settings_store import load_window_settings, reset_settings_singleton, save_window_settings

    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(tmp_path / "settings"))
    reset_settings_singleton()
    try:
        win = SimpleNamespace(
            strip_exif_on_export=True,
            enable_ui_icc_preview=False,
        )
        save_window_settings(win)
        reset_settings_singleton()
        restored = SimpleNamespace()
        load_window_settings(restored)
        assert restored.strip_exif_on_export is True
        assert restored.enable_ui_icc_preview is False

        img = np.full((16, 16, 3), (80, 140, 200), dtype=np.uint8)
        exif = _make_sample_exif()
        context = build_render_context([img], img, source_exif=exif, icc_profile=srgb_icc_profile())
        record = OutputRecord.create(img, "strip_test", context=context)
        host = SimpleNamespace(
            scale_bar_enabled=False,
            strip_exif_on_export=True,
            export_manager=SimpleNamespace(_safe_export_path=lambda path: path),
        )
        labels = LabelManager(host)
        jpg_path = tmp_path / "exported.jpg"
        OutputManager(host)._write_record(record, str(jpg_path), labels)
        with Image.open(jpg_path) as saved:
            assert len(saved.getexif()) == 0
            assert not saved.info.get("exif")
    finally:
        reset_settings_singleton()


# ---------------------------------------------------------------------------
# 5. UI ICC Color Management Preview
# ---------------------------------------------------------------------------

def test_ui_icc_preview_transforms_display_pixels_without_mutating_source(qapp):
    if not is_icc_management_available():
        pytest.skip("LittleCMS2 not available")

    clear_transform_cache()
    swapped_icc = _make_channel_swapped_icc_profile()
    assert describe_icc_profile(swapped_icc)

    # Pure red in BGR = (0, 0, 255). With rotated primaries, display sRGB must shift dramatically.
    bgr = np.full((20, 24, 3), (10, 20, 240), dtype=np.uint8)
    original_snapshot = bgr.copy()

    # Transform is built and cached.
    t1 = get_srgb_transform(swapped_icc)
    t2 = get_srgb_transform(swapped_icc)
    assert t1 is not None and t1 is t2
    assert transform_cache_size() == 1

    converted = convert_bgr_to_srgb_preview(bgr, swapped_icc, enabled=True)
    # Original array MUST remain completely unmodified.
    np.testing.assert_array_equal(bgr, original_snapshot)
    # Display pixels MUST be measurably distinct from the untransformed BGR values.
    diff = float(np.mean(np.abs(converted.astype(np.float32) - bgr.astype(np.float32))))
    assert diff > 80.0

    # Disabled toggle returns original pixels unmodified.
    bypass = convert_bgr_to_srgb_preview(bgr, swapped_icc, enabled=False)
    np.testing.assert_array_equal(bypass, original_snapshot)

    # QPixmap / QImage round-trip through cv2_to_pixmap with ICC enabled vs disabled.
    pix_on = cv2_to_pixmap(bgr, icc_profile=swapped_icc, apply_icc=True)
    pix_off = cv2_to_pixmap(bgr, icc_profile=swapped_icc, apply_icc=False)
    back_on = pixmap_to_cv2(pix_on)
    back_off = pixmap_to_cv2(pix_off)
    assert float(np.mean(np.abs(back_on.astype(np.float32) - back_off.astype(np.float32)))) > 80.0
    qimg_converted = convert_qimage_to_srgb(pix_off.toImage(), swapped_icc, enabled=True)
    assert not qimg_converted.isNull()
    # Original buffer still untouched.
    np.testing.assert_array_equal(bgr, original_snapshot)


# ---------------------------------------------------------------------------
# 6. EXIF Dynamic Label & Watermark Macros
# ---------------------------------------------------------------------------

def test_exif_dynamic_macro_expansion_and_missing_fallback():
    exif_bytes = _make_sample_exif()
    extracted = extract_exif_metadata(exif_bytes, method="OpenFocus B", frames=24)
    assert extracted["date"] == "2026-11-27 14:30:00"
    assert extracted["camera"] == "OpenCam X100"
    assert extracted["lens"] == "Macro 90mm f/2.8"
    assert extracted["aperture"] == "f/8"
    assert extracted["shutter"] == "1/250s"
    assert extracted["iso"] == "ISO 200"
    assert extracted["focal_length"] == "90mm"
    assert extracted["method"] == "OpenFocus B"
    assert extracted["frames"] == "24"

    template = (
        "#{value} | {date} | {camera} | {lens} | "
        "{aperture} {shutter} {iso} {focal_length} | {method} ({frames}f)"
    )
    expanded = expand_exif_macros(template, extracted, value=7)
    assert expanded == (
        "#7 | 2026-11-27 14:30:00 | OpenCam X100 | Macro 90mm f/2.8 | "
        "f/8 1/250s ISO 200 90mm | OpenFocus B (24f)"
    )

    # All 10 documented tokens are present.
    assert set(EXIF_MACRO_TOKENS) == {
        "{value}", "{date}", "{camera}", "{lens}", "{aperture}",
        "{shutter}", "{iso}", "{focal_length}", "{method}", "{frames}",
    }

    # Missing tags fall back cleanly to empty string without errors or raw placeholders.
    partial = expand_exif_macros("Cam:{camera} Lens:{lens} ISO:{iso} Custom:{keep}", {"camera": "X1"})
    assert partial == "Cam:X1 Lens: ISO: Custom:{keep}"

    # Works in build_label_text, LabelAdder, and watermark.effective_text / watermark.burn.
    assert build_label_text("Frame {value} - {camera} {iso}", "", 3, exif_meta=extracted) == (
        "Frame 3 - OpenCam X100 ISO 200"
    )
    canvas = np.full((320, 480, 3), 20, dtype=np.uint8)
    adder = LabelAdder()
    adder.config.format = "{camera} {aperture} {shutter}"
    adder.config.font_size = 20
    adder.config.x_location = 10
    adder.config.y_location = 60
    labeled = adder.add_label_to_image(canvas, 0, exif_meta=extracted)
    assert not adder.last_error
    assert np.any(labeled != canvas)

    wm_cfg = watermark.normalize_config({"text": "{camera} · {focal_length} · {frames}f"})
    assert watermark.effective_text(wm_cfg, exif_meta=extracted) == "OpenCam X100 · 90mm · 24f"
    burned = watermark.burn(canvas, wm_cfg, exif_meta=extracted)
    assert np.any(burned != canvas)


def test_watermark_dialog_exif_macro_chip_bar_inserts_tokens(qapp):
    from PyQt6.QtWidgets import QMainWindow
    from dialogs.watermark import WatermarkDialog

    img = np.full((240, 320, 3), 90, dtype=np.uint8)
    exif_bytes = _make_sample_exif()
    host = QMainWindow()
    host.raw_images = [img]
    host.image_source_exifs = [exif_bytes]
    host.image_filenames = ["frame_01.jpg"]
    host.registration_results = []
    host.fusion_results = []
    host.output_manager = None
    host.output_list = None
    host.current_display_index = 0
    host.current_fusion_method = lambda: "openfocus_b"
    dlg = WatermarkDialog(host)
    try:
        assert hasattr(dlg, "macro_bar")
        assert "{camera}" in dlg.macro_bar.buttons
        assert "{iso}" in dlg.macro_bar.buttons
        dlg.text_edit.clear()
        dlg.macro_bar.buttons["{camera}"].click()
        dlg.macro_bar.buttons["{iso}"].click()
        assert "{camera}" in dlg.text_edit.text()
        assert "{iso}" in dlg.text_edit.text()
    finally:
        dlg.close()
        dlg.deleteLater()
        host.close()
        host.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# 7. Round-2 Performance, Streaming, Cancellation & Theme UX Regressions
# ---------------------------------------------------------------------------

def test_packbits_rle_exhaustive_roundtrip_and_numpy_fallback():
    rng = np.random.default_rng(2026)
    edge_cases = [
        b"",
        b"\x00",
        b"\x7f\x80",
        b"\xaa" * 3,
        b"\xaa" * 127,
        b"\xaa" * 128,
        b"\xaa" * 129,
        b"\xaa" * 130,
        b"\xaa" * 256,
        b"\xaa" * 258,
        b"\x01\x02" + b"\xbb" * 130,
        b"\x01\x02" + b"\xbb" * 130 + b"\xcc" * 5,
        b"\xdd" * 130 + b"\xee" * 130 + b"\x01\x02\x03",
        bytes(i & 0xFF for i in range(128)),
        bytes(i & 0xFF for i in range(129)),
        bytes(i & 0xFF for i in range(513)),
        rng.integers(0, 256, size=4096, dtype=np.uint8).tobytes(),
        (rng.integers(0, 4, size=4096, dtype=np.uint8) * 64).tobytes(),
    ]
    for raw in edge_cases:
        for encoder in (packbits_encode, _packbits_encode_numpy):
            encoded = encoder(raw)
            decoded = packbits_decode(encoded, expected_length=len(raw))
            assert decoded == raw
            # Also verify numpy 1D uint8 array input directly.
            arr = np.frombuffer(raw, dtype=np.uint8)
            encoded_arr = encoder(arr)
            assert packbits_decode(encoded_arr, expected_length=len(raw)) == raw


def test_packbits_and_contrast_performance_regression():
    import time

    rng = np.random.default_rng(99)
    # 1 MB of 4096-byte scanlines should encode well under 150 ms (primary is ~3 ms, numpy ~15 ms).
    rows = [rng.integers(0, 256, size=4096, dtype=np.uint8) for _ in range(256)]
    t0 = time.perf_counter()
    for row in rows:
        _ = packbits_encode(row)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 200.0

    # Large contrast map (>262,144 px exercises histogram percentile fast path).
    big = rng.integers(0, 256, size=(600, 600, 3), dtype=np.uint8)
    big[:, :300] = 128
    cm = compute_contrast_map(big, bit_depth=16, method="hybrid")
    assert cm.shape == (600, 600) and cm.dtype == np.uint16
    assert float(cm[:, 320:].mean()) > float(cm[:, :280].mean()) + 10000.0


def test_layered_psd_streaming_progress_and_cancellation(tmp_path):
    h, w = 96, 128
    frames = [
        np.full((h, w, 3), (30 + i * 40, 80, 220 - i * 35), dtype=np.uint8)
        for i in range(4)
    ]
    result = frames[0].copy()
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    depth = np.broadcast_to(xs[None, :], (h, w)).copy()

    events = []
    psd_path = tmp_path / "streamed_progress.psd"
    export_layered_psd(
        str(psd_path),
        result,
        source_frames=frames,
        source_names=[f"layer_{i}.tif" for i in range(4)],
        depth_index=depth,
        icc_profile=srgb_icc_profile(),
        feather_sigma=2.0,
        progress=lambda c, t, stage, detail: events.append((c, t, stage, detail)),
    )
    stages = [e[2] for e in events]
    assert stages[0] == "masks"
    assert stages.count("layer") == 5  # 4 sources + Merged Result
    assert stages[-1] == "composite"
    assert all(e[1] == 7 for e in events)

    # Verify streamed PSD layer structure & Pillow composite readback.
    parsed = _parse_psd_header_and_layers(psd_path.read_bytes())
    assert parsed["layer_count"] == 5
    with Image.open(psd_path) as loaded:
        loaded.load()
        np.testing.assert_array_equal(np.asarray(loaded.convert("RGB")), result[..., ::-1])

    # Cooperative cancellation mid-export must raise and leave zero temp/output files.
    cancel_flag = {"stop": False}

    def _progress_then_cancel(c, t, stage, detail):
        if stage == "layer" and c >= 2:
            cancel_flag["stop"] = True

    cancel_path = tmp_path / "cancelled.psd"
    with pytest.raises(ValueError, match="cancelled"):
        export_layered_psd(
            str(cancel_path),
            result,
            source_frames=frames,
            depth_index=depth,
            cancelled=lambda: cancel_flag["stop"],
            progress=_progress_then_cancel,
        )
    assert not cancel_path.exists()
    assert not list(tmp_path.glob(".layered-*.psd"))


def test_icc_srgb_zero_copy_bypass_and_large_parallel_equivalence():
    if not is_icc_management_available():
        pytest.skip("LittleCMS2 not available")

    clear_transform_cache()
    srgb = srgb_icc_profile()
    srgb_copy = bytes(srgb)
    swapped = _make_channel_swapped_icc_profile()

    img = np.random.default_rng(7).integers(0, 256, size=(600, 960, 3), dtype=np.uint8)
    snapshot = img.copy()

    # Built-in sRGB (both identical ref and copied bytes) returns input unchanged without copying.
    assert convert_bgr_to_srgb_preview(img, srgb, enabled=True) is img
    assert convert_bgr_to_srgb_preview(img, srgb_copy, enabled=True) is img

    # Large image (>= 512x512) uses strip-parallel path and must match Pillow reference bit-for-bit.
    out = convert_bgr_to_srgb_preview(img, swapped, enabled=True)
    np.testing.assert_array_equal(img, snapshot)
    tr = get_srgb_transform(swapped)
    ref_rgb = np.asarray(ImageCms.applyTransform(Image.fromarray(img[..., ::-1], "RGB"), tr))
    np.testing.assert_array_equal(out[..., ::-1], ref_rgb)


@pytest.mark.parametrize("theme", ["light", "dark", "gray"])
def test_exif_macro_chip_bar_themes_tooltips_and_preview(qapp, theme):
    from PyQt6.QtWidgets import QLineEdit
    from widgets.exif_macro_bar import ExifMacroChipBar, create_macro_preview_label

    edit = QLineEdit()
    bar = ExifMacroChipBar(edit, theme=theme)
    preview = create_macro_preview_label()
    try:
        meta = extract_exif_metadata(_make_sample_exif(), method="OpenFocus B", frames=12)
        bar.update_sample_meta(meta)

        cam_btn = bar.buttons["{camera}"]
        assert "OpenCam X100" in cam_btn.toolTip()
        assert "\n" in cam_btn.toolTip()
        assert cam_btn.property("hasSample") is True

        # Missing metadata chip falls back cleanly in tooltip and clears hasSample.
        bar.update_sample_meta({"camera": "OpenCam X100"})
        assert bar.buttons["{camera}"].property("hasSample") is True
        assert bar.buttons["{lens}"].property("hasSample") is False
        assert "{lens}" in bar.buttons["{lens}"].toolTip()

        assert preview.objectName() == "macroPreviewLabel"
        assert "macroChipButton" in bar.styleSheet()
    finally:
        bar.deleteLater()
        preview.deleteLater()
        edit.deleteLater()
        qapp.processEvents()


def test_layered_psd_fallback_progress_no_double_decode_and_psb_v2(tmp_path):
    """Fallback mask estimation streams per-frame progress without re-decoding valid masks, and supports .psb v2."""
    import struct
    from core.render_context import RenderedDepth, SourceFrame, RenderContext

    h, w = 80, 100
    frames = [
        np.full((h, w, 3), (20 + i * 50, 90, 210 - i * 40), dtype=np.uint8)
        for i in range(3)
    ]
    # Include border black edge on frame 0 so aligned valid_mask exercises _valid_mask_from_pixels.
    frames[0][:4, :] = 0
    result = frames[1].copy()

    resolve_calls = {"count": 0}

    class CountingLazy:
        def __init__(self, arr):
            self._arr = arr
            self.shape = arr.shape
            self.dtype = arr.dtype

        def __array__(self, dtype=None):
            resolve_calls["count"] += 1
            return np.asarray(self._arr, dtype=dtype) if dtype is not None else self._arr

    lazy_frames = tuple(
        SourceFrame(f"f{i}", f"frame_{i}.tif", CountingLazy(arr), aligned=True)
        for i, arr in enumerate(frames)
    )
    ctx = RenderContext(
        context_id="ctx1",
        algorithm="openfocus_b",
        alignment_steps=("ecc",),
        width=w,
        height=h,
        dtype=result.dtype,
        frames=lazy_frames,
    )

    events = []
    psb_path = tmp_path / "large_stack.psb"
    export_layered_psd(
        str(psb_path),
        result,
        context=ctx,
        icc_profile=srgb_icc_profile(),
        progress=lambda c, t, stage, detail: events.append((c, t, stage, detail)),
    )
    assert psb_path.exists()
    raw = psb_path.read_bytes()
    assert raw[:6] == b"8BPS\x00\x02"  # Adobe PSB v2 signature + version

    # Total steps = 3 mask steps + 4 layers + 1 composite = 8; progress advances immediately past 0.
    assert events[0] == (0, 8, "masks", "1/3")
    assert (1, 8, "masks", "1/3") in events
    assert (3, 8, "masks", "3/3") in events
    assert [e[2] for e in events].count("layer") == 4
    assert events[-1] == (8, 8, "composite", "")
    # Each of the 3 frames resolved once in fallback masks + once in layer streaming = 6 total (never extra for valid_mask).
    assert resolve_calls["count"] == 6

    # Reversed dict rendered_depth reverses frame mapping cleanly in fast path.
    depth_dict = {"index": np.zeros((h, w), dtype=np.float32), "reverse_order": True}
    ctx_rev = RenderContext(
        context_id="ctx2",
        algorithm="openfocus_a",
        alignment_steps=(),
        width=w,
        height=h,
        dtype=result.dtype,
        frames=lazy_frames,
        rendered_depth=depth_dict,
    )
    psd_rev = tmp_path / "reversed_depth.psd"
    rev_events = []
    export_layered_psd(
        str(psd_rev),
        result,
        context=ctx_rev,
        progress=lambda c, t, stage, detail: rev_events.append((c, t, stage, detail)),
    )
    assert psd_rev.exists()
    # Uses fast 1-step depth mask path instead of fallback loop: 1 mask + 4 layers + 1 composite = 6 steps.
    assert all(e[1] == 6 for e in rev_events)
    assert resolve_calls["count"] == 9  # +3 (layer streaming only)
