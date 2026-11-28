"""Tests for Interactive HTML5 Focus-Stack Export, Stereoscopic 3D / Turntable Video Export,
and the Upgraded Multi-Light / Multi-Mode 3D Depth Surface Viewer.
"""
from __future__ import annotations

import base64
import io
import re

import cv2
import numpy as np
from PIL import Image
import pytest

from core.html_animation_export import (
    HtmlAnimationExportCancelled,
    HtmlAnimationOptions,
    build_html_animation,
    export_html_animation,
)
from core.stereo_3d_export import (
    LightingSettings,
    RENDER_MODES,
    Stereo3DExportCancelled,
    TurntableOptions,
    export_turntable_video,
    render_anaglyph,
    render_heightfield,
    render_stereo_pair,
    synthesize_stereo_views,
    turntable_camera_angles,
)


def _synthetic_stack(n: int = 5, h: int = 96, w: int = 128) -> list[np.ndarray]:
    frames = []
    yy, xx = np.indices((h, w))
    for i in range(n):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = (30 + i * 40) % 255
        img[..., 1] = ((xx * 2 + i * 25) % 255).astype(np.uint8)
        img[..., 2] = ((yy * 2 + i * 35) % 255).astype(np.uint8)
        cv2.circle(img, (20 + i * 20, h // 2), 14, (240, 220, 40), -1)
        frames.append(img)
    return frames


def _synthetic_surface(h: int = 96, w: int = 128):
    yy, xx = np.linspace(-1.0, 1.0, h, dtype=np.float32), np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gx, gy = np.meshgrid(xx, yy)
    dome = np.exp(-(gx * gx + gy * gy) * 2.5).astype(np.float32)
    ridge = 0.25 * np.sin(gx * 4.0) * np.cos(gy * 3.0)
    height = np.clip(0.15 + 0.75 * dome + ridge, 0.0, 1.0).astype(np.float32)
    texture = np.zeros((h, w, 3), dtype=np.uint8)
    texture[..., 2] = np.clip(height * 230 + 20, 0, 255).astype(np.uint8)
    texture[..., 1] = np.clip((gx + 1.0) * 110 + 20, 0, 255).astype(np.uint8)
    texture[..., 0] = np.clip((1.0 - gy) * 100 + 30, 0, 255).astype(np.uint8)
    return height, texture


# ---------------------------------------------------------------------------
# 1. Interactive Single-File HTML5 Focus-Stack Animation Export
# ---------------------------------------------------------------------------

def test_html_animation_export_self_contained_structure_and_frames(tmp_path):
    frames = _synthetic_stack(n=6, h=120, w=200)
    labels = [f"Slice_{i:02d}.tif" for i in range(len(frames))]
    out_file = tmp_path / "focus_stack_viewer.html"
    opts = HtmlAnimationOptions(
        max_width=160,
        quality=85,
        image_format="webp",
        fps=12,
        ping_pong=True,
        include_fused_frame=False,
        title="Specimen Focus Stack",
        show_footer=True,
    )

    written = export_html_animation(str(out_file), frames, labels=labels, options=opts)
    assert written == str(out_file)
    assert out_file.exists() and out_file.stat().st_size > 1000

    html_text = out_file.read_text(encoding="utf-8")
    assert html_text.startswith("<!DOCTYPE html>")
    assert "<html" in html_text and "</html>" in html_text
    assert "<style>" in html_text and "<script>" in html_text
    assert "Specimen Focus Stack" in html_text
    assert "OpenFocus Interactive Focus Stack" in html_text

    # Zero external CDN or network URLs anywhere in the generated file
    assert "http://" not in html_text.lower()
    assert "https://" not in html_text.lower()

    # Embedded base64 frames count matches stack length
    data_uris = re.findall(r"data:image/(?:webp|jpeg);base64,[A-Za-z0-9+/=]+", html_text)
    assert len(data_uris) == len(frames)

    # Verify decoded dimensions respect max_width=160
    b64_payload = data_uris[0].split(",", 1)[1]
    decoded = Image.open(io.BytesIO(base64.b64decode(b64_payload)))
    assert decoded.width == 160
    assert decoded.height == 96

    # Interactive JS controls present
    for token in (
        'id="scrubber"',
        'id="btnPlay"',
        'id="btnMode"',
        'id="fpsSelect"',
        'id="frameBadge"',
        "pointerdown",
        "touchstart",
        "ArrowLeft",
        "ArrowRight",
        "Space",
    ):
        assert token in html_text, f"missing interactive control token: {token}"


def test_html_animation_export_jpeg_and_fused_comparison(tmp_path):
    frames = _synthetic_stack(n=4, h=80, w=100)
    # Include a 16-bit fused frame to verify 16-bit display conversion
    fused_16 = (frames[0].astype(np.uint16) * 257)
    opts = HtmlAnimationOptions(
        max_width=800,
        quality=90,
        image_format="jpeg",
        include_fused_frame=True,
    )
    html_text = build_html_animation(frames, fused_image=fused_16, options=opts)
    data_uris = re.findall(r"data:image/jpeg;base64,[A-Za-z0-9+/=]+", html_text)
    # 4 stack frames + 1 fused result frame
    assert len(data_uris) == 5
    assert 'id="btnFused"' in html_text
    assert "http://" not in html_text.lower()
    assert "https://" not in html_text.lower()


def test_html_animation_export_cancellation():
    frames = _synthetic_stack(n=4, h=48, w=64)
    seen = []

    def cancel_after_one():
        return len(seen) >= 1

    with pytest.raises(HtmlAnimationExportCancelled):
        build_html_animation(
            frames,
            progress_callback=lambda i, n: seen.append((i, n)),
            should_cancel=cancel_after_one,
        )


# ---------------------------------------------------------------------------
# 2. Stereoscopic 3D Synthesis: Anaglyph & Side-by-Side Stereo Pair
# ---------------------------------------------------------------------------

def _bar_scene(h: int = 80, w: int = 160):
    """Bright white vertical bar on black background at near depth (d=1.0)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 0.5, dtype=np.float32)  # zero-parallax background
    cx = w // 2
    img[:, cx - 4:cx + 5] = (255, 255, 255)
    depth[:, cx - 8:cx + 9] = 1.0  # foreground pops out
    return img, depth, cx


def _weighted_center_x(mask_2d: np.ndarray) -> float:
    weights = mask_2d.astype(np.float64)
    total = float(weights.sum())
    assert total > 0
    xs = np.arange(mask_2d.shape[1], dtype=np.float64)[None, :]
    return float((weights * xs).sum() / total)


@pytest.mark.parametrize("mode", ["red_cyan", "red_green"])
def test_anaglyph_shape_channels_and_parallax(mode):
    img, depth, cx = _bar_scene()
    anaglyph = render_anaglyph(img, depth, mode=mode, max_disparity_px=12.0, convergence=0.5)

    assert anaglyph.shape == img.shape
    assert anaglyph.dtype == np.uint8

    red_x = _weighted_center_x(anaglyph[:, :, 2])    # Left eye -> Red channel
    green_x = _weighted_center_x(anaglyph[:, :, 1])  # Right eye -> Green channel

    # Near feature (d=1.0 > 0.5) shifts right in left eye (R) and left in right eye (G)
    assert red_x > cx + 1.0
    assert green_x < cx - 1.0
    assert (red_x - green_x) > 3.5

    if mode == "red_green":
        assert int(anaglyph[:, :, 0].max()) == 0
    else:
        blue_x = _weighted_center_x(anaglyph[:, :, 0])
        assert abs(blue_x - green_x) < 0.5

    # Stronger separation increases parallax shift
    stronger = render_anaglyph(img, depth, mode=mode, max_disparity_px=24.0, convergence=0.5)
    stronger_shift = _weighted_center_x(stronger[:, :, 2]) - _weighted_center_x(stronger[:, :, 1])
    assert stronger_shift > (red_x - green_x) + 2.0


def test_stereo_pair_parallel_and_cross_eyed_geometry():
    img, depth, cx = _bar_scene(h=80, w=140)
    h, w = img.shape[:2]

    parallel = render_stereo_pair(img, depth, mode="parallel", max_disparity_px=12.0, convergence=0.5)
    cross = render_stereo_pair(img, depth, mode="cross_eyed", max_disparity_px=12.0, convergence=0.5)

    assert parallel.shape == (h, 2 * w, 3)
    assert cross.shape == (h, 2 * w, 3)
    assert parallel.dtype == np.uint8

    # Parallel = [Left | Right]; Cross-Eyed = [Right | Left]
    par_left, par_right = parallel[:, :w], parallel[:, w:]
    crs_left, crs_right = cross[:, :w], cross[:, w:]
    np.testing.assert_array_equal(crs_left, par_right)
    np.testing.assert_array_equal(crs_right, par_left)

    # Disparity direction: foreground bar shifts right in Left eye, left in Right eye
    left_cx = _weighted_center_x(par_left[:, :, 1])
    right_cx = _weighted_center_x(par_right[:, :, 1])
    assert left_cx > cx + 1.0
    assert right_cx < cx - 1.0

    # Optional center divider widens output cleanly
    with_div = render_stereo_pair(img, depth, mode="parallel", divider_width=6)
    assert with_div.shape == (h, 2 * w + 6, 3)


# ---------------------------------------------------------------------------
# 3. 3D Turntable Rotation Video (MP4 / AVI) & Animated GIF Export
# ---------------------------------------------------------------------------

def test_turntable_gif_export(tmp_path):
    height, texture = _synthetic_surface(64, 80)
    target = tmp_path / "turntable.gif"
    opts = TurntableOptions(
        output_format="gif",
        width=200,
        height=150,
        frame_count=8,
        fps=10,
        motion_mode="rock_25",
        render_mode="shaded_clay",
    )
    progress = []
    written = export_turntable_video(
        str(target),
        height,
        texture,
        options=opts,
        progress_callback=lambda i, n: progress.append((i, n)),
    )
    assert written == str(target)
    assert target.exists() and target.stat().st_size > 1000
    assert progress[-1] == (8, 8)

    with Image.open(target) as gif:
        assert getattr(gif, "is_animated", False)
        assert gif.n_frames == 8
        assert gif.size == (200, 150)


def test_turntable_mp4_export(tmp_path):
    height, texture = _synthetic_surface(64, 80)
    target = tmp_path / "turntable.mp4"
    opts = TurntableOptions(
        output_format="mp4",
        width=240,
        height=180,
        frame_count=10,
        fps=20,
        motion_mode="turntable_360",
        render_mode="textured_wireframe",
    )
    written = export_turntable_video(str(target), height, texture, options=opts)
    assert written.endswith((".mp4", ".avi"))
    cap = cv2.VideoCapture(written)
    try:
        assert cap.isOpened()
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        assert len(frames) == 10
        assert frames[0].shape == (180, 240, 3)
        # Camera actually rotated between frame 0 and frame 3
        assert np.mean(np.abs(frames[0].astype(np.float32) - frames[3].astype(np.float32))) > 1.0
    finally:
        cap.release()


def test_turntable_export_cancellation_and_angles(tmp_path):
    height, texture = _synthetic_surface(48, 64)
    opts = TurntableOptions(output_format="gif", width=160, height=120, frame_count=12, motion_mode="orbit_pitch")
    y0, p0 = turntable_camera_angles(0, opts)
    y3, p3 = turntable_camera_angles(3, opts)
    assert y3 != y0 and p3 != p0

    target = tmp_path / "cancelled.gif"
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(Stereo3DExportCancelled):
        export_turntable_video(str(target), height, texture, options=opts, should_cancel=cancel)
    assert not target.exists()


# ---------------------------------------------------------------------------
# 4. Upgraded DepthSurface Multi-Light + Render Modes in Light & Dark Themes
# ---------------------------------------------------------------------------

def test_depth_surface_render_modes_and_lighting_in_themes(qapp):
    from ui.styles import CURRENT_THEME, get_theme_style, set_current_theme
    from utils.image_utils import pixmap_to_cv2
    from widgets.depth_surface import DepthSurface

    prev_theme = CURRENT_THEME
    height, texture = _synthetic_surface(80, 100)
    surface = DepthSurface()
    surface.resize(420, 320)
    surface.set_surface(height, texture)
    surface.show()
    qapp.processEvents()
    try:
        for theme in ("light", "dark", "gray"):
            set_current_theme(theme)
            qapp.setStyleSheet(get_theme_style(theme))
            captures = {}
            for mode in RENDER_MODES:
                surface.set_render_mode(mode)
                qapp.processEvents()
                img = pixmap_to_cv2(surface.grab())
                assert img.shape[:2] == (320, 420)
                captures[mode] = img
            # Distinct visual outputs across modes
            assert np.mean(np.abs(captures["textured"].astype(float) - captures["shaded_clay"].astype(float))) > 3.0
            assert np.mean(np.abs(captures["shaded_clay"].astype(float) - captures["wireframe"].astype(float))) > 5.0
            assert np.mean(np.abs(captures["textured"].astype(float) - captures["depth_color"].astype(float))) > 2.0

        # Multi-light changes visibly alter shading in Shaded Clay mode
        surface.set_render_mode("shaded_clay")
        surface.set_lighting(LightingSettings(key_azimuth=45.0, key_elevation=25.0, key_intensity=1.2, fill_intensity=0.0, ambient=0.1))
        img_a = pixmap_to_cv2(surface.grab())
        surface.set_lighting(LightingSettings(key_azimuth=225.0, key_elevation=25.0, key_intensity=1.2, fill_intensity=0.0, ambient=0.1))
        img_b = pixmap_to_cv2(surface.grab())
        assert np.mean(np.abs(img_a.astype(float) - img_b.astype(float))) > 2.0
    finally:
        set_current_theme(prev_theme)
        surface.close()
        surface.deleteLater()
        qapp.processEvents()


def test_html_animation_dialog_and_menu_wiring(qapp):
    from dialogs.html_animation import HtmlAnimationDialog
    from tests.test_menu_translations import _menu_window
    from ui.menus import setup_menus

    dlg = HtmlAnimationDialog(has_registered=True, has_input=True, has_fused=True, default_title="Demo Stack")
    try:
        dlg.width_combo.setCurrentIndex(2)  # 1920 px
        dlg.quality_slider.setValue(90)
        dlg.fps_spin.setValue(15)
        dlg.ping_pong_chk.setChecked(False)
        opts = dlg.options()
        assert opts.max_width == 1920
        assert opts.quality == 90
        assert opts.fps == 15.0
        assert opts.ping_pong is False
        assert opts.include_fused_frame is True
        assert opts.title == "Demo Stack"
    finally:
        dlg.close()
        dlg.deleteLater()

    win = _menu_window(qapp)
    called = []
    win.export_manager.export_html_animation = lambda: called.append(True)
    setup_menus(win)
    try:
        action = win.ui_objs["action_export_html_animation"]
        assert action in win.ui_objs["menu_file"].actions()
        action.trigger()
        assert called == [True]
    finally:
        win.close()
        win.deleteLater()
        qapp.processEvents()


def test_depth_dialog_upgraded_3d_controls_and_themes(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QWidget
    import dialogs.depthmap as dm
    from core.algorithms import resolve_fusion_parameters, resolve_registration_parameters
    from ui.styles import CURRENT_THEME, get_theme_style, set_current_theme

    prev_theme = CURRENT_THEME
    settings = QSettings(str(tmp_path / "depth-3d-prefs.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(dm, "get_settings", lambda: settings)

    height, texture = _synthetic_surface(64, 80)
    win = QWidget()
    win.raw_images = [texture, texture.copy()]
    win.image_filenames = ["a.tif", "b.tif"]
    win.current_fusion_method = lambda: "dct"
    win.current_alignment_method = lambda: "none"
    win.fusion_parameters_for = lambda m, depth=False: resolve_fusion_parameters(m, depth=depth)
    win.registration_parameters_for = resolve_registration_parameters
    win.use_gpu = False
    win.thread_count = 1
    win.fusion_result = texture.copy()

    dialog = dm.DepthMapDialog(win)
    try:
        dialog.show()
        dialog.tabs.setCurrentIndex(1)
        # Before computing a map, stereo/video export buttons stay disabled
        assert not dialog.btn_export_anaglyph.isEnabled()
        assert not dialog.btn_export_stereo_pair.isEnabled()
        assert not dialog.btn_export_turntable.isEnabled()

        # Populate a surface and exercise render modes + lighting sliders across Light & Dark
        dialog.surface.set_surface(height, texture, extent=(63, 79))
        for theme in ("light", "dark"):
            set_current_theme(theme)
            qapp.setStyleSheet(get_theme_style(theme))
            for idx in range(dialog.render_mode_combo.count()):
                dialog.render_mode_combo.setCurrentIndex(idx)
                qapp.processEvents()
            dialog.render_mode_combo.setCurrentIndex(0)
            dialog.light_azimuth_slider.setValue(180)
            dialog.light_elevation_slider.setValue(35)
            dialog.light_key_slider.setValue(110)
            dialog.light_fill_slider.setValue(40)
            dialog.light_ambient_slider.setValue(25)
            qapp.processEvents()
            assert dialog.surface.lighting().key_azimuth == pytest.approx(180.0)
            assert dialog.surface.lighting().key_elevation == pytest.approx(35.0)
            dialog.btn_light_reset.click()
            qapp.processEvents()
            assert dialog.surface.lighting().key_azimuth == pytest.approx(315.0)
    finally:
        set_current_theme(prev_theme)
        dialog.reject()
        win.close()
        win.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------
# 5. Round-2 Optimization Verification: Parallel HTML5 Ordering, Cached Normals,
#    Adaptive LOD Refine, Fast Vectorized Stereo & Parallel Turntable Parity
# ---------------------------------------------------------------------------

def test_html_animation_parallel_frame_ordering_and_canvas_ux(tmp_path):
    """Parallel ThreadPoolExecutor encoding must preserve exact frame order & viewer UX."""
    import json as _json

    n_frames = 14
    h, w = 96, 128
    # Each frame has a distinct uniform RGB color so any reordering is immediately caught.
    frames = []
    labels = []
    expected_b = []
    for i in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        b_val = 20 + i * 16
        g_val = 230 - i * 14
        r_val = 40 + ((i * 37) % 190)
        img[:, :] = (b_val, g_val, r_val)  # BGR
        frames.append(img)
        labels.append(f"Focus_{i:02d}")
        expected_b.append(b_val)

    opts = HtmlAnimationOptions(
        max_width=128,
        quality=95,
        image_format="jpeg",
        fps=15,
        include_fused_frame=True,
    )
    progress_seen = []
    html_parallel = build_html_animation(
        frames,
        labels=labels,
        fused_image=frames[-1],
        options=opts,
        max_workers=6,
        progress_callback=lambda done, total: progress_seen.append((done, total)),
    )
    html_serial = build_html_animation(
        frames,
        labels=labels,
        fused_image=frames[-1],
        options=opts,
        max_workers=1,
    )
    # Monotonic progress reporting (1..total)
    assert progress_seen == [(i, n_frames + 1) for i in range(1, n_frames + 2)]
    assert html_parallel == html_serial

    # Extract embedded JSON payload and verify exact frame order & decoded color parity
    m = re.search(r"var DATA = (\{.*?\});\n", html_parallel)
    assert m is not None
    payload = _json.loads(m.group(1).replace("<\\/", "</"))
    assert len(payload["frames"]) == n_frames
    assert [f["label"] for f in payload["frames"]] == labels
    assert payload["fused"] is not None

    for idx, item in enumerate(payload["frames"]):
        raw_bytes = base64.b64decode(item["src"].split(",", 1)[1])
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        arr = np.asarray(pil_img)
        # RGB blue channel corresponds to BGR[0]
        mean_b = float(arr[:, :, 2].mean())
        assert abs(mean_b - expected_b[idx]) < 4.0, f"frame {idx} out of order: got B={mean_b}, expected {expected_b[idx]}"

    # Polished offline 60fps canvas viewer UX tokens & strict zero-network check
    for token in (
        'id="viewerCanvas"',
        'addEventListener("wheel"',
        "hud-idle",
        "touchAxis",
        "requestAnimationFrame",
    ):
        assert token in html_parallel, f"missing polished HTML viewer token: {token}"
    assert "http://" not in html_parallel.lower()
    assert "https://" not in html_parallel.lower()


def test_cached_heightfield_normals_and_render_parity():
    """Cached PreparedHeightfield normals and renders must match fresh computation exactly."""
    from core.stereo_3d_export import compute_heightfield_normals, prepare_heightfield

    height, texture = _synthetic_surface(120, 160)
    prep = prepare_heightfield(height, mesh_step=60, invert=False, extent=(119, 159))

    # Direct reference normal computation
    z_smooth = cv2.GaussianBlur(prep.z, (0, 0), 1.0)
    dz_dx = np.gradient(z_smooth, axis=1).reshape(-1)
    dz_dy = np.gradient(z_smooth, axis=0).reshape(-1)
    strength = 2.4 * 0.25 + 0.35
    ref_nx = -dz_dx * strength
    ref_ny = dz_dy * strength
    ref_nz = np.ones_like(dz_dx)
    inv_norm = 1.0 / np.sqrt(ref_nx * ref_nx + ref_ny * ref_ny + ref_nz * ref_nz)
    ref_nx, ref_ny, ref_nz = ref_nx * inv_norm, ref_ny * inv_norm, ref_nz * inv_norm

    nx, ny, nz = prep.normals(0.25)
    np.testing.assert_allclose(nx, ref_nx, atol=1e-6)
    np.testing.assert_allclose(ny, ref_ny, atol=1e-6)
    np.testing.assert_allclose(nz, ref_nz, atol=1e-6)

    # Calling normals() again for the same relief returns the exact cached arrays
    nx2, ny2, nz2 = prep.normals(0.25)
    assert nx2 is nx and ny2 is ny and nz2 is nz

    # Standalone helper matches too
    snx, sny, snz = compute_heightfield_normals(prep.z, 0.25)
    np.testing.assert_allclose(snx, ref_nx, atol=1e-6)
    np.testing.assert_allclose(sny, ref_ny, atol=1e-6)
    np.testing.assert_allclose(snz, ref_nz, atol=1e-6)

    # Render parity across all 5 render modes & multiple camera angles
    light = LightingSettings(key_azimuth=120.0, key_elevation=40.0, key_intensity=1.0, fill_intensity=0.3, ambient=0.25)
    for mode in RENDER_MODES:
        cached_rgb = render_heightfield(
            height, texture, 320, 240,
            yaw=25.0, pitch=38.0, zoom=1.1, relief=0.25, mesh_step=60,
            extent=(119, 159), render_mode=mode, lighting=light, prepared=prep,
        )
        fresh_rgb = render_heightfield(
            height, texture, 320, 240,
            yaw=25.0, pitch=38.0, zoom=1.1, relief=0.25, mesh_step=60,
            extent=(119, 159), render_mode=mode, lighting=light, prepared=None,
        )
        np.testing.assert_array_equal(cached_rgb, fresh_rgb)


def test_depth_surface_adaptive_lod_and_release_refine(qapp):
    """Interactive drag/slider lowers LOD and reuses cached normals; release sharpens to full detail."""
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from widgets.depth_surface import DepthSurface

    height, texture = _synthetic_surface(240, 320)
    surf = DepthSurface()
    surf.resize(480, 360)
    surf.set_surface(height, texture, extent=(239, 319))
    surf.set_detail(80)
    surf.show()
    qapp.processEvents()
    try:
        # Settled full-detail render populates full-step prepared cache
        img_full = surf._render_cached("#12161b", "#4c8dff", "#4c8dff")
        assert img_full is not None and not img_full.isNull()
        assert (80, False) in surf._prepared_cache
        full_prep = surf._prepared_cache[(80, False)]

        # Mouse press + move enters interactive LOD with lower mesh_step
        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(100.0, 100.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        surf.mousePressEvent(press)
        assert surf.is_interactive()
        lod_step = surf._interactive_mesh_step()
        assert lod_step < 80

        move1 = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(130.0, 110.0),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        surf.mouseMoveEvent(move1)
        img_drag1 = surf._render_cached("#12161b", "#4c8dff", "#4c8dff")
        assert img_drag1 is not None
        assert (lod_step, False) in surf._prepared_cache
        lod_prep = surf._prepared_cache[(lod_step, False)]
        lod_normals_before = lod_prep._normals

        # Second drag angle reuses the exact same cached normals without recomputation
        move2 = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(165.0, 125.0),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        surf.mouseMoveEvent(move2)
        img_drag2 = surf._render_cached("#12161b", "#4c8dff", "#4c8dff")
        assert img_drag2 is not None
        assert surf._prepared_cache[(lod_step, False)]._normals is lod_normals_before

        # Mouse release exits interactive mode and immediately refines at full detail
        release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(165.0, 125.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        surf.mouseReleaseEvent(release)
        assert not surf.is_interactive()
        assert surf._cache is None
        img_refined = surf._render_cached("#12161b", "#4c8dff", "#4c8dff")
        assert img_refined is not None
        assert surf._cache is not None
        # Original full-detail PreparedHeightfield was kept cached across the drag
        assert surf._prepared_cache[(80, False)] is full_prep

        # Slider interactive hook toggles LOD and refines cleanly on release
        surf.set_interactive(True)
        assert surf.is_interactive()
        surf.set_key_azimuth(180.0)
        surf._render_cached("#12161b", "#4c8dff", "#4c8dff")
        surf.set_interactive(False)
        assert not surf.is_interactive()
        assert surf._cache is None
    finally:
        surf.close()
        surf.deleteLater()
        qapp.processEvents()


def test_fast_vectorized_stereo_large_multistriped_parity():
    """Multi-striped parallel DIBR stereo synthesis (>=200k px) matches single-stripe occlusion."""
    from core.stereo_3d_export import _warp_eye_stripe

    h, w = 480, 560  # 268,800 px -> exercises multi-threaded striped path
    depth, img = _synthetic_surface(h, w)
    # Add sharp foreground bar to stress disocclusion & foreground-over-background occlusion
    cx = w // 2
    img[:, cx - 12:cx + 12] = (245, 250, 255)
    depth[:, cx - 16:cx + 16] = 1.0

    left_par, right_par = synthesize_stereo_views(img, depth, max_disparity_px=24.0, convergence=0.5)
    assert left_par.shape == (h, w, 3) and right_par.shape == (h, w, 3)

    # Compare against single-stripe reference on the exact same smoothed depth
    d_smooth = np.ascontiguousarray(cv2.GaussianBlur(depth, (0, 0), 0.8), dtype=np.float32)
    xs_f32 = np.arange(w, dtype=np.float32)[None, :]
    xs_u64 = np.arange(w, dtype=np.uint64)[None, :]
    xs_i32 = np.arange(w, dtype=np.int32)[None, :]
    ref_left = _warp_eye_stripe(img, d_smooth, +0.5, 24.0, 0.5, xs_f32, xs_u64, xs_i32)
    ref_right = _warp_eye_stripe(img, d_smooth, -0.5, 24.0, 0.5, xs_f32, xs_u64, xs_i32)
    np.testing.assert_array_equal(left_par, ref_left)
    np.testing.assert_array_equal(right_par, ref_right)

    # Disparity shifts foreground right in left view and left in right view
    left_bar = _weighted_center_x((left_par[:, :, 0] > 235).astype(np.uint8))
    right_bar = _weighted_center_x((right_par[:, :, 0] > 235).astype(np.uint8))
    assert left_bar > cx + 4.0
    assert right_bar < cx - 4.0


def test_parallel_turntable_matches_serial_order(tmp_path):
    """Multi-threaded turntable GIF export produces frame-identical output to serial export."""
    height, texture = _synthetic_surface(64, 80)
    opts = TurntableOptions(
        output_format="gif",
        width=180,
        height=135,
        frame_count=10,
        fps=12,
        motion_mode="orbit_pitch",
        render_mode="textured",
    )
    serial_path = tmp_path / "serial.gif"
    parallel_path = tmp_path / "parallel.gif"
    export_turntable_video(str(serial_path), height, texture, options=opts, max_workers=1)
    export_turntable_video(str(parallel_path), height, texture, options=opts, max_workers=4)

    with Image.open(serial_path) as g1, Image.open(parallel_path) as g2:
        assert g1.n_frames == g2.n_frames == 10
        for idx in range(10):
            g1.seek(idx)
            g2.seek(idx)
            np.testing.assert_array_equal(np.asarray(g1.convert("RGB")), np.asarray(g2.convert("RGB")))


def test_html_animation_includes_desktop_parity_loupe_and_valid_js(tmp_path):
    """Exported HTML5 viewer embeds a self-contained magnifier loupe with valid JS and zero external URLs."""
    import re
    import shutil
    import subprocess
    from core.html_animation_export import HtmlAnimationOptions, build_html_animation

    frames = [
        np.full((90, 120, 3), (30 + i * 50, 100, 200), dtype=np.uint8)
        for i in range(3)
    ]
    fused = np.full((90, 120, 3), (120, 160, 210), dtype=np.uint8)
    html_doc = build_html_animation(
        frames,
        fused_image=fused,
        options=HtmlAnimationOptions(max_width=120, include_fused_frame=True),
    )

    assert "http://" not in html_doc and "https://" not in html_doc
    for token in (
        'id="btnLoupe"',
        'id="btnLoupeShape"',
        "drawLoupe",
        "ensureNativeLoupe",
        "imageSmoothingEnabled = deviceScale < 1.0",
        "LOUPE_MAX_ZOOM",
        "LOUPE_MAX_SIZE",
    ):
        assert token in html_doc

    match = re.search(r"<script>\s*(.*?)\s*</script>", html_doc, re.S)
    assert match is not None
    node_exe = shutil.which("node")
    if node_exe is not None:
        js_path = tmp_path / "viewer.js"
        js_path.write_text(match.group(1), encoding="utf-8")
        proc = subprocess.run([node_exe, "--check", str(js_path)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


def test_html_animation_defaults_to_app_language_and_embeds_12_lang_switcher(qapp, tmp_path, monkeypatch):
    import json
    import re
    import shutil
    import subprocess
    from core.html_animation_export import HtmlAnimationOptions, build_html_animation
    from core.html_viewer_i18n import HTML_ANIM_VIEWER_I18N, SUPPORTED_VIEWER_LANGS
    from dialogs.html_animation import HtmlAnimationDialog
    from locales import trans

    monkeypatch.setattr(trans, "current_lang", "zh")
    dlg = HtmlAnimationDialog()
    try:
        opts = dlg.options()
        assert opts.language == "zh"
    finally:
        dlg.deleteLater()

    frames = _synthetic_stack(n=3, h=64, w=80)
    fused = frames[0].copy()
    html_doc = build_html_animation(
        frames,
        fused_image=fused,
        options=HtmlAnimationOptions(max_width=80, include_fused_frame=True, language="zh_CN"),
    )
    assert '<html lang="zh"' in html_doc
    assert 'id="selLang"' in html_doc
    m = re.search(r"var DATA = (\{.*?\});", html_doc)
    assert m is not None
    payload = json.loads(m.group(1).replace("<\\/", "</"))
    assert payload["lang"] == "zh"
    assert set(payload["langNames"]) == set(SUPPORTED_VIEWER_LANGS)
    assert set(payload["i18n"]) == set(SUPPORTED_VIEWER_LANGS)
    assert payload["i18n"]["zh"]["loupeBtn"] == HTML_ANIM_VIEWER_I18N["zh"]["loupeBtn"]

    match = re.search(r"<script>\s*(.*?)\s*</script>", html_doc, re.S)
    assert match is not None
    node_exe = shutil.which("node")
    if node_exe is not None:
        js_path = tmp_path / "viewer_i18n.js"
        js_path.write_text(match.group(1), encoding="utf-8")
        proc = subprocess.run([node_exe, "--check", str(js_path)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
