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

