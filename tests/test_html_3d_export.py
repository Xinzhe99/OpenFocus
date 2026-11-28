"""Tests for self-contained Interactive HTML5 / WebGL 3D Viewer export."""
from __future__ import annotations

import base64
import io
import json
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import pytest

from core.html_3d_export import (
    HTML_3D_MOTION_MODES,
    Html3DExportCancelled,
    Html3DOptions,
    build_html_3d_viewer,
    encode_heightfield_png_data_uri,
    encode_lut_hex,
    export_html_3d,
)
from core.stereo_3d_export import LightingSettings, RENDER_MODES


def _surface(h: int = 120, w: int = 160, with_nans: bool = False):
    yy, xx = np.linspace(-1.0, 1.0, h, dtype=np.float32), np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gx, gy = np.meshgrid(xx, yy)
    z = np.clip(0.5 + 0.4 * np.exp(-(gx * gx + gy * gy) * 2.2) + 0.1 * np.sin(gx * 5.0), 0.0, 1.0).astype(np.float32)
    if with_nans:
        z[:6, :8] = np.nan
    tex = np.zeros((h, w, 3), dtype=np.uint8)
    tex[..., 2] = np.clip(np.nan_to_num(z, nan=0.0) * 220 + 25, 0, 255).astype(np.uint8)
    tex[..., 1] = np.clip((gx + 1.0) * 105 + 30, 0, 255).astype(np.uint8)
    tex[..., 0] = np.clip((1.0 - gy) * 105 + 30, 0, 255).astype(np.uint8)
    return z, tex


def test_encode_heightfield_png_16bit_and_nan_mask_roundtrip():
    h, w = 80, 120
    z, _ = _surface(h, w, with_nans=True)
    uri, gw, gh, half_w, half_h = encode_heightfield_png_data_uri(
        z, max_grid_size=120, invert_depth=False
    )
    assert (gw, gh) == (120, 80)
    assert half_w == pytest.approx(1.0)
    assert half_h == pytest.approx((h - 1) / float(w - 1), rel=1e-4)
    assert uri.startswith("data:image/png;base64,")

    png_bytes = base64.b64decode(uri.split(",", 1)[1])
    with Image.open(io.BytesIO(png_bytes)) as im:
        rgba = np.asarray(im.convert("RGBA"))
    assert rgba.shape == (80, 120, 4)
    # Alpha must remain 255 everywhere so browser canvas never premultiplies RGB.
    assert int(rgba[:, :, 3].min()) == 255
    # Top-left NaN block marked invalid (B==0), interior marked valid (B==255).
    assert np.all(rgba[:6, :8, 2] == 0)
    assert np.all(rgba[10:, 12:, 2] == 255)
    q = (rgba[10:, 12:, 0].astype(np.uint16) << 8) | rgba[10:, 12:, 1].astype(np.uint16)
    decoded_z = (q.astype(np.float32) - 1.0) / 65534.0
    np.testing.assert_allclose(decoded_z, z[10:, 12:], atol=1e-4)

    # Inverted encoding flips normalized depth cleanly.
    uri_inv, *_ = encode_heightfield_png_data_uri(z, max_grid_size=120, invert_depth=True)
    with Image.open(io.BytesIO(base64.b64decode(uri_inv.split(",", 1)[1]))) as im_inv:
        rgba_inv = np.asarray(im_inv.convert("RGBA"))
    q_inv = (rgba_inv[10:, 12:, 0].astype(np.uint16) << 8) | rgba_inv[10:, 12:, 1].astype(np.uint16)
    np.testing.assert_allclose((q_inv - 1.0) / 65534.0, 1.0 - z[10:, 12:], atol=1e-4)


def test_build_html_3d_viewer_offline_xss_safe_and_valid_js():
    z, tex = _surface(128, 160)
    bad_title = '</script><img src=x onerror="alert(1)"> & Co.'
    progress = []
    opts = Html3DOptions(
        max_grid_size=128,
        max_texture_width=160,
        image_format="webp",
        quality=86,
        render_mode="textured_wireframe",
        relief=0.22,
        auto_motion="rock_25",
        title=bad_title,
        theme="light",
        show_footer=True,
        physical_z_range_um=(10.0, 134.5),
        pixel_size_um=1.25,
    )
    html_text, stats = build_html_3d_viewer(
        z, tex, options=opts, progress_callback=lambda i, n: progress.append((i, n))
    )
    assert progress[0] == (0, 4) and progress[-1] == (4, 4)
    assert stats["grid_w"] == 128 or stats["grid_w"] == 160
    assert html_text.startswith("<!DOCTYPE html>")
    # Zero external CDN or network URLs
    assert "http://" not in html_text.lower()
    assert "https://" not in html_text.lower()
    # Escaped title + neutralized script breakout
    assert "</script><img" not in html_text
    assert "&lt;/script&gt;" in html_text
    assert "<\\/script>" in html_text
    assert "data:image/png;base64," in html_text
    assert "data:image/webp;base64," in html_text

    # Validate generated inline JS syntax with Node when present in PATH.
    node_exe = shutil.which("node")
    if node_exe is not None:
        m = re.search(r"<script>\n(.*?)\n</script>", html_text, re.DOTALL)
        assert m is not None
        res = subprocess.run(
            [node_exe, "--check", "-"],
            input=m.group(1),
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert res.returncode == 0, f"Generated JS syntax error:\n{res.stderr}"


@pytest.mark.parametrize("mode", RENDER_MODES)
@pytest.mark.parametrize("motion", HTML_3D_MOTION_MODES)
def test_html_3d_supports_all_modes_motions_and_no_texture(mode, motion):
    z, _ = _surface(48, 64)
    opts = Html3DOptions(
        max_grid_size=48,
        render_mode=mode,
        auto_motion=motion,
        image_format="jpeg",
        theme="auto",
    )
    html_text, _ = build_html_3d_viewer(z, None, options=opts)
    m = re.search(r"const DATA = (\{.*?\});", html_text)
    assert m is not None
    payload = json.loads(m.group(1).replace("<\\/", "</"))
    assert payload["hasTexture"] is False
    expected_mode = "shaded_clay" if mode in ("textured", "textured_wireframe") else mode
    assert payload["mode"] == expected_mode
    assert payload["motion"] == motion
    assert len(payload["lutHex"]) == 256 * 6


def test_export_html_3d_atomic_guards_and_cancellation(tmp_path):
    z, tex16 = _surface(64, 80)
    tex16 = tex16.astype(np.uint16) * 257
    out_file = tmp_path / "model.html"

    info = export_html_3d(str(out_file), z, tex16, options=Html3DOptions(image_format="jpeg"))
    assert info["files"] == [str(out_file)]
    assert out_file.exists() and out_file.stat().st_size > 2000

    with pytest.raises(FileExistsError):
        export_html_3d(str(out_file), z, tex16, overwrite=False)

    # Cancelled export raises and leaves destination untouched / no temp files.
    cancel_target = tmp_path / "cancelled.html"
    seen = []

    def cancel_midway():
        return len(seen) >= 1

    with pytest.raises(Html3DExportCancelled):
        export_html_3d(
            str(cancel_target),
            z,
            tex16,
            progress_callback=lambda i, n: seen.append((i, n)),
            should_cancel=cancel_midway,
        )
    assert not cancel_target.exists()
    assert not list(tmp_path.glob(".openfocus-html3d-*"))


def test_html_3d_export_performance_and_compact_size():
    z, tex = _surface(512, 640)
    t0 = time.perf_counter()
    html_text, stats = build_html_3d_viewer(
        z,
        tex,
        options=Html3DOptions(max_grid_size=384, max_texture_width=1024, image_format="webp"),
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.5
    assert stats["html_bytes"] < 800_000


def test_html_3d_dialog_and_depthmap_integration(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from dialogs.html_3d_export import Html3DExportDialog
    from tests.test_depth_dialog import _click_and_finish, _wait
    from tests.test_depth_dialog import depth_dialog  # noqa: F401
    import dialogs.depthmap as dm

    dlg = Html3DExportDialog(
        source_shape=(240, 320),
        has_texture=True,
        default_render_mode="shaded_clay",
    )
    try:
        opts = dlg.build_options(
            initial_yaw=-20.0,
            initial_pitch=35.0,
            initial_zoom=1.2,
            relief=0.25,
            lighting=LightingSettings(key_azimuth=200.0),
        )
        assert opts.render_mode == "shaded_clay"
        assert opts.max_grid_size == 384
        assert opts.relief == pytest.approx(0.25)
    finally:
        dlg.deleteLater()

    # Check DepthMapDialog buttons + end-to-end worker export.
    out_html = tmp_path / "exported_3d.html"
    monkeypatch.setattr(Html3DExportDialog, "exec", lambda self: 1)
    monkeypatch.setattr(
        dm.FileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_html), "Interactive HTML 3D (*.html *.htm)")),
    )

    from PyQt6.QtWidgets import QWidget
    from PyQt6.QtCore import QSettings
    from core.algorithms import resolve_fusion_parameters, resolve_registration_parameters

    settings = QSettings(str(tmp_path / "dm.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(dm, "get_settings", lambda: settings)
    win = QWidget()
    win.experimental_features = True
    win.raw_images = [np.full((48, 64, 3), i * 80 + 40, np.uint8) for i in range(3)]
    win.image_filenames = ["a.tif", "b.tif", "c.tif"]
    win.current_fusion_method = lambda: "openfocus_b"
    win.current_alignment_method = lambda: "none"
    win.ui_theme = "dark"
    win.use_gpu = False
    win.thread_count = 1
    win.fusion_result = win.raw_images[1].copy()
    win.fusion_parameters_for = lambda m, depth=False: resolve_fusion_parameters(m, depth=depth)
    win.registration_parameters_for = resolve_registration_parameters

    dialog = dm.DepthMapDialog(win)
    try:
        dialog.show()
        assert hasattr(dialog, "btn_export_html_3d")
        assert hasattr(dialog, "btn_save_html_3d")
        assert not dialog.btn_export_html_3d.isEnabled()
        assert not dialog.btn_save_html_3d.isEnabled()
        _click_and_finish(qapp, dialog)
        dialog.tabs.setCurrentIndex(1)
        _wait(qapp, lambda: dialog._visual_ready_revision == dialog._visual_revision)
        assert dialog.btn_export_html_3d.isEnabled()
        assert dialog.btn_save_html_3d.isEnabled()
        dialog.btn_export_html_3d.click()
        _wait(qapp, lambda: dialog._export_worker is None)
        assert out_html.exists() and out_html.stat().st_size > 2000
    finally:
        dialog.reject()
        _wait(qapp, lambda: not dm._ORPHAN_WORKERS)


def test_html_3d_translations_present_in_all_12_languages():
    from locales import trans
    from locales.export_3d_i18n import EXPORT_3D_TRANSLATIONS

    keys = [k for k in EXPORT_3D_TRANSLATIONS["en"] if k.startswith(("html_3d_", "depth_3d_export_html"))]
    assert len(keys) >= 15
    assert len(trans.PACK_ORDER) == 12
    for code in trans.PACK_ORDER:
        table = trans.translations[code]
        for key in keys:
            assert table.get(key), f"{code}: missing {key}"


def test_html_3d_viewer_defaults_to_app_language_and_embeds_12_lang_switcher(qapp, monkeypatch):
    from core.html_viewer_i18n import HTML_3D_VIEWER_I18N, SUPPORTED_VIEWER_LANGS
    from dialogs.html_3d_export import Html3DExportDialog
    from locales import trans

    z, tex = _surface(64, 80)
    monkeypatch.setattr(trans, "current_lang", "zh")
    dlg = Html3DExportDialog(source_shape=(64, 80), has_texture=True)
    try:
        opts = dlg.build_options()
        assert opts.language == "zh"
    finally:
        dlg.deleteLater()

    html_text, _ = build_html_3d_viewer(z, tex, options=Html3DOptions(max_grid_size=64, language="ja_JP"))
    assert '<html lang="ja">' in html_text
    assert 'id="selLang"' in html_text
    m = re.search(r"const DATA = (\{.*?\});", html_text)
    assert m is not None
    payload = json.loads(m.group(1).replace("<\\/", "</"))
    assert payload["lang"] == "ja"
    assert set(payload["langNames"]) == set(SUPPORTED_VIEWER_LANGS)
    assert set(payload["i18n"]) == set(SUPPORTED_VIEWER_LANGS)
    assert payload["i18n"]["zh"]["surfSec"] == HTML_3D_VIEWER_I18N["zh"]["surfSec"]

    node_exe = shutil.which("node")
    if node_exe is not None:
        ms = re.search(r"<script>\n(.*?)\n</script>", html_text, re.DOTALL)
        assert ms is not None
        res = subprocess.run([node_exe, "--check", "-"], input=ms.group(1), text=True, capture_output=True, timeout=10)
        assert res.returncode == 0, res.stderr
