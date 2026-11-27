"""Tests for OpenFocus Retouching macro precision tools (#6, #7, #8, #9)."""
from __future__ import annotations

import time
import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtTest import QTest

from controllers.output_manager import OutputRecord
from core.render_context import RenderedDepth, build_render_context
from dialogs.retouch import RetouchDialog
from locales import trans
from locales._retouch_macro import RETOUCH_MACRO_TRANSLATIONS
from ui.styles import get_theme_style
from utils.retouch import RetouchDocument


def _context(frames, result, **kwargs):
    return build_render_context(
        frames, result, filenames=[f"frame_{i + 1:02d}.png" for i in range(len(frames))],
        algorithm="openfocus_b", **kwargs)


def _wait_for_source(qapp, dialog):
    for _ in range(500):
        if dialog._source_ready and not dialog._source_loading:
            return
        QTest.qWait(10)
    raise AssertionError("Retouch source did not become ready")


def _open_dialog(qapp, frames, result, *, rendered_depth=None):
    ctx = _context(frames, result, rendered_depth=rendered_depth)
    record = OutputRecord.create(result, "macro_stack.png", context=ctx)
    dlg = RetouchDialog(None, record=record, apply_callback=lambda *_: (True, ""))
    dlg.show()
    _wait_for_source(qapp, dlg)
    return dlg


# ---------------------------------------------------------------------------
# #6 Edge-Aware Color Tolerance
# ---------------------------------------------------------------------------

def test_color_tolerance_preserves_high_contrast_macro_edge():
    # Light background (220) with a thin dark macro hair (25) across y=30..34
    base = np.full((64, 64, 3), 220, dtype=np.uint8)
    base[30:35, :] = 25
    # Source frame has clean lavender-tinted background (180, 140, 160) everywhere
    source = np.full((64, 64, 3), (180, 140, 160), dtype=np.uint8)

    # Standard brush (tolerance=0) overwrites the dark hair
    doc_off = RetouchDocument(base.copy(), _context([source], base))
    doc_off.set_brush(diameter=24, hardness=1.0, opacity=1.0)
    doc_off.set_color_tolerance(0)
    assert doc_off.begin_stroke(32, 26)
    doc_off.stroke_to(32, 32)
    assert doc_off.end_stroke() == "committed"
    assert np.all(doc_off.working[32, 32] == (180, 140, 160))

    # Color tolerance=25 anchored on background (220) replaces background
    # pixels but preserves the dark hair (25) untouched!
    doc_tol = RetouchDocument(base.copy(), _context([source], base))
    doc_tol.set_brush(diameter=24, hardness=1.0, opacity=1.0)
    assert doc_tol.set_color_tolerance(25)
    assert doc_tol.begin_stroke(32, 26)
    doc_tol.stroke_to(32, 32)
    assert doc_tol.end_stroke() == "committed"
    assert np.all(doc_tol.working[26, 32] == (180, 140, 160))
    assert np.all(doc_tol.working[32, 32] == 25)
    assert np.all(doc_tol.working[38, 32] == (180, 140, 160))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_color_tolerance_16bit_and_checkpoint_roundtrip(tmp_path, dtype):
    hi = 220 if dtype == np.uint8 else 56000
    lo = 20 if dtype == np.uint8 else 5000
    src_val = 150 if dtype == np.uint8 else 38000
    base = np.full((64, 64, 3), hi, dtype=dtype)
    base[28:36, :] = lo
    source = np.full((64, 64, 3), src_val, dtype=dtype)

    doc = RetouchDocument(base.copy(), _context([source], base))
    doc.initialize_checkpoint()
    doc.set_brush(diameter=20, hardness=1.0, opacity=1.0)
    doc.set_color_tolerance(20)
    assert doc.begin_stroke(32, 24)
    doc.stroke_to(32, 32)
    assert doc.end_stroke() == "committed"
    assert np.all(doc.working[24, 32] == src_val)
    assert np.all(doc.working[32, 32] == lo)

    path = tmp_path / "tol.ofretouch"
    doc.save_checkpoint(path)
    replayed = RetouchDocument(base.copy(), _context([source], base))
    replayed.initialize_checkpoint()
    replayed.load_checkpoint(path)
    np.testing.assert_array_equal(replayed.working, doc.working)


def test_color_tolerance_dab_is_fast_enough_for_60fps():
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    source = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    doc = RetouchDocument(base, _context([source], base))
    doc.set_brush(diameter=128, hardness=0.5, opacity=1.0)
    doc.set_color_tolerance(30)
    t0 = time.perf_counter()
    assert doc.begin_stroke(128, 128)
    for x in range(132, 172, 4):
        doc.stroke_to(x, 128)
    assert doc.end_stroke() == "committed"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # 11 large 128px dabs should complete well within interactive budget
    assert elapsed_ms < 500.0


# ---------------------------------------------------------------------------
# #7 Clone Stamp (Clone Area & Paint with Pattern)
# ---------------------------------------------------------------------------

def test_clone_area_tracks_offset_undo_redo_and_checkpoint(tmp_path):
    base = np.full((96, 96, 3), 40, dtype=np.uint8)
    # Distinctive colored marker block around (20, 20)
    base[15:26, 15:26] = (30, 180, 240)
    base[20, 20] = (255, 10, 200)
    doc = RetouchDocument(base.copy(), _context([base.copy()], base))
    doc.initialize_checkpoint()

    assert doc.set_tool("clone")
    assert doc.set_clone_mode("area")
    # Without an origin, stroke must be refused
    assert doc.source_valid_at(60, 60) is False
    assert doc.begin_stroke(60, 60) is False

    assert doc.set_clone_origin(20, 20)
    doc.set_brush(diameter=12, hardness=1.0, opacity=1.0)
    assert doc.begin_stroke(60, 60)
    doc.stroke_to(63, 60)
    assert doc.end_stroke() == "committed"

    # At (60, 60) we sampled (20, 20); at (63, 60) we sampled (23, 20)
    assert tuple(doc.working[60, 60]) == (255, 10, 200)
    assert tuple(doc.working[60, 63]) == (30, 180, 240)
    # Original source block untouched
    assert tuple(doc.working[20, 20]) == (255, 10, 200)

    painted = doc.working.copy()
    assert doc.undo()
    np.testing.assert_array_equal(doc.working, base)
    assert doc.redo()
    np.testing.assert_array_equal(doc.working, painted)

    path = tmp_path / "clone_area.ofretouch"
    doc.save_checkpoint(path)
    replayed = RetouchDocument(base.copy(), _context([base.copy()], base))
    replayed.initialize_checkpoint()
    replayed.load_checkpoint(path)
    np.testing.assert_array_equal(replayed.working, painted)


def test_clone_pattern_tiles_around_origin_without_drifting_off_subject(tmp_path):
    base = np.full((128, 128, 3), 20, dtype=np.uint8)
    # Textured 24x24 patch around (24, 24) with values in [160..220]
    yy, xx = np.mgrid[12:36, 12:36]
    base[12:36, 12:36, 0] = (170 + ((xx * 7 + yy * 5) % 40)).astype(np.uint8)
    base[12:36, 12:36, 1] = 190
    base[12:36, 12:36, 2] = 210

    doc = RetouchDocument(base.copy(), _context([base.copy()], base))
    doc.initialize_checkpoint()
    assert doc.set_tool("clone")
    assert doc.set_clone_mode("pattern")
    assert doc.set_clone_origin(24, 24)
    doc.set_brush(diameter=16, hardness=1.0, opacity=1.0)

    # Paint a long horizontal stroke far across the canvas (x=40 -> x=110)
    assert doc.begin_stroke(40, 96)
    doc.stroke_to(110, 96)
    assert doc.end_stroke() == "committed"

    # Even 70px away from stroke start, Paint with Pattern stays strictly
    # within the sampled subject texture around (24, 24) instead of drifting
    # into the dark background (20)!
    for x in (40, 60, 80, 100, 110):
        px = doc.working[96, x]
        assert px[1] == 190 and px[2] == 210
        assert 170 <= int(px[0]) <= 210

    painted = doc.working.copy()
    assert doc.undo()
    np.testing.assert_array_equal(doc.working, base)
    assert doc.redo()
    np.testing.assert_array_equal(doc.working, painted)

    path = tmp_path / "clone_pattern.ofretouch"
    doc.save_checkpoint(path)
    replayed = RetouchDocument(base.copy(), _context([base.copy()], base))
    replayed.initialize_checkpoint()
    replayed.load_checkpoint(path)
    np.testing.assert_array_equal(replayed.working, painted)


# ---------------------------------------------------------------------------
# #8 F9 Contributing Source Frame Lookup & Depth/Source Map Overlay
# ---------------------------------------------------------------------------

def test_contributing_frame_lookup_uses_rendered_depth_and_reverse_order():
    frames = [np.full((48, 64, 3), v, dtype=np.uint8) for v in (40, 120, 210)]
    result = np.full((48, 64, 3), 120, dtype=np.uint8)
    # Top third = 0.0 (frame 0), middle = 0.5 (frame 1), bottom = 1.0 (frame 2)
    index01 = np.zeros((48, 64), dtype=np.float32)
    index01[16:32, :] = 0.5
    index01[32:, :] = 1.0

    for reverse_order, expected in ((False, (0, 1, 2)), (True, (2, 1, 0))):
        rendered = {
            "index": index01.copy(),
            "valid_rect": (0, 0, 64, 48),
            "reverse_order": reverse_order,
            "parameters": {},
            "frame_count": 3,
            "method": "openfocus_b",
            "roi_rect": None,
        }
        ctx = _context(frames, result, rendered_depth=rendered)
        assert isinstance(ctx.rendered_depth, RenderedDepth)
        doc = RetouchDocument(result.copy(), ctx)
        assert doc.locate_contributing_frame(32, 8) == expected[0]
        assert doc.locate_contributing_frame(32, 24) == expected[1]
        assert doc.locate_contributing_frame(32, 40) == expected[2]
        idx_map, origin, valid_mask = doc.source_index_map()
        assert idx_map.dtype == np.uint8
        assert origin == (0, 0) and valid_mask is None
        assert int(idx_map[8, 32]) == expected[0]
        assert int(idx_map[24, 32]) == expected[1]
        assert int(idx_map[40, 32]) == expected[2]


def test_contributing_frame_fallback_estimator_without_rendered_depth():
    # Three source frames with distinct bands; fused result combines band i from frame i
    h, w = 60, 90
    frames = [
        np.full((h, w, 3), (30, 50, 200), dtype=np.uint8),
        np.full((h, w, 3), (40, 210, 60), dtype=np.uint8),
        np.full((h, w, 3), (220, 70, 40), dtype=np.uint8),
    ]
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:, :30] = frames[0][:, :30]
    result[:, 30:60] = frames[1][:, 30:60]
    result[:, 60:] = frames[2][:, 60:]

    doc = RetouchDocument(result.copy(), _context(frames, result))
    assert doc.context.rendered_depth is None
    # Single-pixel lookup before full map exists
    assert doc.locate_contributing_frame(15, 30) == 0
    assert doc.locate_contributing_frame(45, 30) == 1
    assert doc.locate_contributing_frame(75, 30) == 2

    # Full fallback source index map
    idx_map, origin, _valid = doc.source_index_map()
    assert idx_map.dtype == np.uint8
    assert origin == (0, 0)
    assert int(idx_map[30, 15]) == 0
    assert int(idx_map[30, 45]) == 1
    assert int(idx_map[30, 75]) == 2


# ---------------------------------------------------------------------------
# #9 Source Brush Brightness Compensation & Local Alignment Offset
# ---------------------------------------------------------------------------

def test_source_brush_manual_and_auto_brightness_compensation(tmp_path):
    base = np.full((64, 64, 3), 150, dtype=np.uint8)
    source = np.full((64, 64, 3), 100, dtype=np.uint8)

    # Manual +20% brightness: 100 * 1.20 = 120
    doc_manual = RetouchDocument(base.copy(), _context([source], base))
    doc_manual.set_brush(diameter=12, hardness=1.0, opacity=1.0)
    assert doc_manual.set_brush_brightness(20)
    assert doc_manual.begin_stroke(32, 32)
    assert doc_manual.end_stroke() == "committed"
    assert np.all(doc_manual.working[32, 32] == 120)

    # Auto match brightness: source=100, target=150 -> gain=1.5 -> painted=150
    # Add a slight blue tint on source so auto brightness matches luminance while keeping hue!
    tinted_src = np.full((64, 64, 3), (120, 100, 80), dtype=np.uint8)
    doc_auto = RetouchDocument(base.copy(), _context([tinted_src], base))
    doc_auto.initialize_checkpoint()
    doc_auto.set_brush(diameter=16, hardness=1.0, opacity=1.0)
    assert doc_auto.set_brush_auto_brightness(True)
    assert doc_auto.begin_stroke(32, 32)
    assert doc_auto.end_stroke() == "committed"
    painted = doc_auto.working[32, 32].astype(np.float32)
    painted_luma = painted[0] * 0.114 + painted[1] * 0.587 + painted[2] * 0.299
    assert abs(painted_luma - 150.0) <= 3.0
    # Hue relationship preserved (B > G > R)
    assert painted[0] > painted[1] > painted[2]

    # Checkpoint roundtrip preserves auto brightness gain
    path = tmp_path / "auto_br.ofretouch"
    doc_auto.save_checkpoint(path)
    replayed = RetouchDocument(base.copy(), _context([tinted_src], base))
    replayed.initialize_checkpoint()
    replayed.load_checkpoint(path)
    np.testing.assert_array_equal(replayed.working, doc_auto.working)


def test_source_local_offset_and_auto_align_patch(tmp_path):
    rng = np.random.default_rng(7)
    h, w = 128, 128
    # Create textured target and a source shifted by (-6x, +4y) relative to target
    # (i.e. target[y, x] == source[y + 4, x - 6] => shifting source by (+6, -4) aligns them)
    pattern = rng.integers(40, 220, (h, w, 3), dtype=np.uint8)
    source = pattern.copy()
    target = np.roll(np.roll(source, 6, axis=1), -4, axis=0)
    # Mark a distinct pixel in source at (58, 68) which shifts to (64, 64)
    source[68, 58] = (10, 250, 30)

    doc = RetouchDocument(target.copy(), _context([source], target))
    doc.initialize_checkpoint()
    dx, dy, resp = doc.auto_align_patch(64, 64, window_size=96)
    assert abs(dx - 6) <= 1 and abs(dy - (-4)) <= 1
    assert resp > 0.2
    assert (doc.source_offset_x, doc.source_offset_y) == (dx, dy)

    # Painting with exact offset (6, -4) copies aligned pixel without ghosting
    doc.set_source_offset(6, -4)
    doc.set_brush(diameter=8, hardness=1.0, opacity=1.0)
    assert doc.begin_stroke(64, 64)
    assert doc.end_stroke() == "committed"
    assert tuple(doc.working[64, 64]) == (10, 250, 30)

    path = tmp_path / "offset.ofretouch"
    doc.save_checkpoint(path)
    replayed = RetouchDocument(target.copy(), _context([source], target))
    replayed.initialize_checkpoint()
    replayed.load_checkpoint(path)
    np.testing.assert_array_equal(replayed.working, doc.working)


# ---------------------------------------------------------------------------
# GUI integration: Sidebar visibility, F9, Depth Map Overlay, Alt+Click, Themes
# ---------------------------------------------------------------------------

def test_gui_tool_specific_sidebar_visibility_and_shortcuts(qapp, monkeypatch):
    h, w = 80, 100
    frames = [
        np.full((h, w, 3), (40, 60, 200), dtype=np.uint8),
        np.full((h, w, 3), (50, 200, 70), dtype=np.uint8),
        np.full((h, w, 3), (210, 80, 50), dtype=np.uint8),
    ]
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:, :34] = frames[0][:, :34]
    result[:, 34:67] = frames[1][:, 34:67]
    result[:, 67:] = frames[2][:, 67:]

    dlg = _open_dialog(qapp, frames, result)
    monkeypatch.setattr(dlg, "_canvas_focused", lambda: True)
    try:
        # Source tool active by default -> only source_options_box visible
        assert dlg.source_options_box.isVisible()
        assert not dlg.clone_options_box.isVisible()
        assert not dlg.num_exposure.isVisible()

        # Switch to Clone tool via 'S' key -> only clone_options_box visible
        key_s = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_S, Qt.KeyboardModifier.NoModifier)
        assert dlg._handle_key(key_s)
        qapp.processEvents()
        assert dlg.doc.tool == "clone"
        assert dlg.clone_options_box.isVisible()
        assert not dlg.source_options_box.isVisible()
        assert not dlg.num_exposure.isVisible()

        # In Clone tool, Alt does NOT enter source preview mode (reserved for Alt+Click)
        key_alt = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Alt, Qt.KeyboardModifier.AltModifier)
        assert not dlg._handle_key(key_alt)
        assert dlg.preview_mode is None

        # Alt+Click on result canvas picks clone origin
        widget_pt = dlg.result_canvas.image_to_widget(20, 30)
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress, widget_pt, widget_pt,
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.AltModifier,
        )
        dlg.result_canvas.mousePressEvent(press)
        assert dlg.doc.clone_origin is not None
        assert abs(dlg.doc.clone_origin[0] - 20) <= 1.5
        assert abs(dlg.doc.clone_origin[1] - 30) <= 1.5
        assert dlg.live_clone_source_point() == dlg.doc.clone_origin

        # Switch back to Source tool and test Shift+Arrow local offset nudging
        dlg._set_tool("source")
        key_right = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        key_up = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.ShiftModifier)
        assert dlg._handle_key(key_right)
        assert dlg._handle_key(key_up)
        assert (dlg.doc.source_offset_x, dlg.doc.source_offset_y) == (1, -1)
        assert dlg.spin_offset_x.value() == 1 and dlg.spin_offset_y.value() == -1
        dlg.btn_reset_offset.click()
        assert (dlg.doc.source_offset_x, dlg.doc.source_offset_y) == (0, 0)

        # F9 locates contributing source frame at cursor
        dlg.set_cursor(80, 40)
        key_f9 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F9, Qt.KeyboardModifier.NoModifier)
        assert dlg._handle_key(key_f9)
        assert dlg.doc.frame_index == 2
        assert "3 / 3" in dlg._notice

        # Toggle Depth/Source Map overlay on and wait for worker completion
        before_display = dlg._result_display.copy()
        dlg.chk_depth_map.setChecked(True)
        for _ in range(300):
            if dlg._depth_overlay_bgr is not None:
                break
            QTest.qWait(10)
        assert dlg._depth_overlay_bgr is not None
        assert not np.array_equal(dlg._result_display, before_display)
        # Toggle off restores clean display
        dlg.chk_depth_map.setChecked(False)
        qapp.processEvents()
        np.testing.assert_array_equal(dlg._result_display, before_display)
    finally:
        dlg._allow_close = True
        dlg.close()


@pytest.mark.parametrize("theme", ["light", "dark", "gray"])
def test_retouch_macro_tools_render_in_all_themes(qapp, theme):
    previous_style = qapp.styleSheet()
    qapp.setStyleSheet(get_theme_style(theme))
    frames = [
        np.full((64, 80, 3), 60, dtype=np.uint8),
        np.full((64, 80, 3), 180, dtype=np.uint8),
    ]
    result = np.full((64, 80, 3), 120, dtype=np.uint8)
    index01 = np.linspace(0.0, 1.0, 80, dtype=np.float32)[None, :].repeat(64, axis=0)
    rendered = {
        "index": index01,
        "valid_rect": (0, 0, 80, 64),
        "reverse_order": False,
        "parameters": {},
        "frame_count": 2,
        "method": "openfocus_b",
        "roi_rect": None,
    }
    dlg = _open_dialog(qapp, frames, result, rendered_depth=rendered)
    try:
        dlg.chk_depth_map.setChecked(True)
        qapp.processEvents()
        assert dlg._depth_overlay_bgr is not None
        pix = dlg.grab()
        assert not pix.isNull() and pix.width() > 0 and pix.height() > 0
    finally:
        dlg._allow_close = True
        dlg.close()
        qapp.setStyleSheet(previous_style)


def test_retouch_macro_translations_exist_in_all_12_languages():
    en_keys = set(RETOUCH_MACRO_TRANSLATIONS["en"])
    assert len(en_keys) >= 30
    for code in trans.PACK_ORDER:
        table = trans.translations[code]
        missing = en_keys - set(table)
        assert not missing, f"{code}: missing retouch macro keys {sorted(missing)}"
        for key in en_keys:
            assert table[key] and table[key].strip(), f"{code}: empty translation for {key}"
