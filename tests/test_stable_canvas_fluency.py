"""Regression tests for canvas, wipe/compare, and retouch interaction fluency."""
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QApplication

from utils.image_utils import cv2_to_pixmap
from utils.retouch import RetouchDocument, brush_mask
from widgets.compare_canvas import SourceCanvas
from widgets.magnifier_label import MagnifierLabel
from widgets.wipe_compare import WipeCompareWidget


def _mouse(widget, kind, pos, button, buttons):
    pt = QPointF(*pos) if not isinstance(pos, QPointF) else pos
    event = QMouseEvent(kind, pt, pt, button, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


def _wheel(widget, pos, delta, *, control=False):
    pt = QPointF(*pos) if not isinstance(pos, QPointF) else pos
    mods = Qt.KeyboardModifier.ControlModifier if control else Qt.KeyboardModifier.NoModifier
    event = QWheelEvent(pt, pt, QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton,
                        mods, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)
    return event


# ---------------------------------------------------------------------------
# 1. MagnifierLabel fluency, zoom badge, wheel scaling, middle-click, ROI
# ---------------------------------------------------------------------------


def test_magnifier_skips_redundant_base_draw_when_source_pixmap_present(qapp, monkeypatch):
    label = MagnifierLabel()
    label.resize(300, 200)
    pix = QPixmap(300, 200)
    pix.fill(QColor(40, 80, 120))
    label.set_display_pixmap(pix)
    assert label._source_pixmap is pix

    draw_calls = []
    real_draw = QPainter.drawPixmap

    def spy_draw(self, *args, **kwargs):
        draw_calls.append(args)
        return real_draw(self, *args, **kwargs)

    monkeypatch.setattr(QPainter, "drawPixmap", spy_draw)
    label._composite_key = None
    label._update_scaled_pixmap()
    assert len(draw_calls) == 1
    label.close()


def test_magnifier_coalesces_wheel_pan_and_resize_into_paint(qapp):
    class CountingLabel(MagnifierLabel):
        rebuilds = 0

        def _update_scaled_pixmap(self, *, clamp=True, **kwargs):
            self.rebuilds += 1
            super()._update_scaled_pixmap(clamp=clamp, **kwargs)

    label = CountingLabel()
    label.resize(400, 300)
    pix = QPixmap(800, 600)
    pix.fill(QColor(50, 60, 70))
    label.set_display_pixmap(pix)
    label.show()
    qapp.processEvents()

    label.rebuilds = 0
    for _ in range(5):
        _wheel(label, (200, 150), 120)
    assert label.rebuilds == 0
    assert label._interactive_transform
    label.grab()
    assert label.rebuilds == 1

    label._refine_timer.timeout.emit()
    assert not label._interactive_transform
    label.rebuilds = 0

    _mouse(label, QEvent.Type.MouseButtonPress, (200, 150),
           Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    for step in range(1, 6):
        _mouse(label, QEvent.Type.MouseMove, (200 + step * 5, 150 + step * 3),
               Qt.MouseButton.NoButton, Qt.MouseButton.RightButton)
    assert label.rebuilds == 0
    assert label._interactive_transform
    label.grab()
    assert label.rebuilds == 1
    _mouse(label, QEvent.Type.MouseButtonRelease, (225, 165),
           Qt.MouseButton.RightButton, Qt.MouseButton.NoButton)

    label._refine_timer.timeout.emit()
    label.rebuilds = 0
    label.resize(460, 340)
    assert label.rebuilds == 0
    assert label._interactive_transform
    label.grab()
    assert label.rebuilds == 1
    label.close()


def test_magnifier_zoom_badge_reuses_single_timer_and_single_finished_connection(qapp):
    label = MagnifierLabel()
    label.resize(400, 300)
    pix = QPixmap(400, 300)
    pix.fill(QColor(20, 20, 20))
    label.set_display_pixmap(pix)
    label.show()
    qapp.processEvents()

    finished_calls = []
    real_finished = label._on_zoom_indicator_animation_finished
    label._zoom_indicator_animation.finished.disconnect(real_finished)

    def track_finished():
        finished_calls.append(1)
        real_finished()

    label._zoom_indicator_animation.finished.connect(track_finished)

    for _ in range(10):
        _wheel(label, (200, 150), 120)
    assert label._zoom_indicator_visible
    assert label._zoom_indicator_timer.isActive()
    assert label._zoom_indicator_timer.isSingleShot()

    label._start_zoom_indicator_fade()
    label._zoom_indicator_animation.finished.emit()
    assert len(finished_calls) == 1
    assert not label._zoom_indicator_visible
    label.close()


def test_magnifier_wheel_scales_by_delta_and_middle_click_resets_once(qapp):
    label = MagnifierLabel()
    label.resize(400, 300)
    pix = QPixmap(400, 300)
    pix.fill(QColor(10, 20, 30))
    label.set_display_pixmap(pix)

    _wheel(label, (200, 150), 60)
    assert label._zoom_factor == pytest.approx(1.1 ** 0.5)

    label.reset_view()
    _wheel(label, (200, 150), 1200)
    assert label._zoom_factor == pytest.approx(1.1 ** 4.0)

    resets = []
    real_reset = label.reset_view
    label.reset_view = lambda: (resets.append(1), real_reset())
    _mouse(label, QEvent.Type.MouseButtonPress, (200, 150),
           Qt.MouseButton.MiddleButton, Qt.MouseButton.MiddleButton)
    _mouse(label, QEvent.Type.MouseButtonRelease, (200, 150),
           Qt.MouseButton.MiddleButton, Qt.MouseButton.NoButton)
    assert len(resets) == 1
    assert label._zoom_factor == pytest.approx(1.0)
    label.close()


def test_magnifier_roi_space_pan_bounds_clamping_and_delete_edge_clamp(qapp):
    label = MagnifierLabel()
    label.resize(400, 300)
    pix = QPixmap(400, 300)
    pix.fill(QColor(30, 40, 50))
    label.set_display_pixmap(pix)
    label.roi_mode = True
    original = QRectF(80, 60, 120, 100)
    label.set_roi_rect(original)

    # Space + left-drag in ROI mode must pan without altering the ROI.
    label.set_space_pressed(True)
    _mouse(label, QEvent.Type.MouseButtonPress, (200, 150),
           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    assert label._is_panning
    _mouse(label, QEvent.Type.MouseMove, (240, 175),
           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    _mouse(label, QEvent.Type.MouseButtonRelease, (240, 175),
           Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    label.set_space_pressed(False)
    assert label._pan_offset == QPointF(40, 25)
    assert label.get_roi_rect() == original

    # Dragging ROI creation past image bounds clamps cleanly to [0, W] x [0, H].
    label.reset_view()
    _mouse(label, QEvent.Type.MouseButtonPress, (350, 250),
           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    _mouse(label, QEvent.Type.MouseMove, (600, 500),
           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    _mouse(label, QEvent.Type.MouseButtonRelease, (600, 500),
           Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    roi = label.get_roi_rect()
    assert roi is not None
    assert 0.0 <= roi.left() <= roi.right() <= 400.0
    assert 0.0 <= roi.top() <= roi.bottom() <= 300.0

    # Moving an ROI past the left/top edge preserves its size and clamps inside bounds.
    label.set_roi_rect(QRectF(20, 20, 100, 80))
    _mouse(label, QEvent.Type.MouseButtonPress, (60, 60),
           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    _mouse(label, QEvent.Type.MouseMove, (-100, -100),
           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    _mouse(label, QEvent.Type.MouseButtonRelease, (-100, -100),
           Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    moved = label.get_roi_rect()
    assert moved == QRectF(0, 0, 100, 80)

    # Delete button stays completely inside widget rect even at the top-right edge.
    label.set_roi_rect(QRectF(300, 0, 100, 80))
    screen_roi = label._roi_in_widget(label.get_roi_rect())
    del_rect = label._get_handle_rects(screen_roi)["del"]
    assert QRectF(label.rect()).contains(del_rect)
    label.close()


# ---------------------------------------------------------------------------
# 2. WipeCompareWidget & Compare SourceCanvas tile retention, debounce, clamp
# ---------------------------------------------------------------------------


def test_wipe_divider_drag_reuses_tiles_and_pan_debounces_clamps_and_keeps_tiles(qapp, monkeypatch):
    from core import source_previews
    from tests.test_label_native_preview import RegionWorker

    class Worker(RegionWorker):
        instances = []

        def request_cancel(self):
            self.cancelled = True

        def isRunning(self):
            return False

    monkeypatch.setattr(source_previews, "SourceRegionWorker", Worker)
    widget = WipeCompareWidget()
    widget.resize(220, 220)
    preview = cv2_to_pixmap(np.zeros((10, 10, 3), np.uint8))
    widget.set_images(preview, preview, left_native=(object(), {}), right_native=(object(), {}),
                      left_size=(1000, 1000), right_size=(1000, 1000))
    widget._zoom = 8
    widget.show()
    widget.grab()

    left_side = widget._left_regions
    worker = left_side.worker
    first_req = left_side.request
    assert first_req is not None
    tile_pixels = np.full((32, 32, 3), 180, np.uint8)
    worker.ready.emit(worker.key, first_req, tile_pixels, first_req[2])
    assert len(left_side.tiles) == 1
    assert left_side.request is None

    # Dragging the divider must reuse cached tiles with zero re-decode requests.
    _mouse(widget, QEvent.Type.MouseButtonPress, (110, 110),
           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    for x in (60, 150, 180, 90):
        _mouse(widget, QEvent.Type.MouseMove, (x, 110),
               Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
        widget.grab()
        assert left_side.request is None
        assert len(left_side.tiles) == 1
    _mouse(widget, QEvent.Type.MouseButtonRelease, (90, 110),
           Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)

    # Panning uses ClosedHandCursor, debounces decode requests, and retains prior tiles.
    _mouse(widget, QEvent.Type.MouseButtonPress, (30, 110),
           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    assert widget.cursor().shape() == Qt.CursorShape.ClosedHandCursor
    _mouse(widget, QEvent.Type.MouseMove, (50, 130),
           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    assert widget.cursor().shape() == Qt.CursorShape.ClosedHandCursor
    assert widget._detail_timer.isActive()
    widget.grab()
    assert left_side.request is None  # debounced while dragging

    # Releasing flushes immediately and retains prior tiles until the new tile arrives.
    _mouse(widget, QEvent.Type.MouseButtonRelease, (50, 130),
           Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    assert widget.cursor().shape() == Qt.CursorShape.OpenHandCursor
    widget.grab()
    second_req = left_side.request
    assert second_req is not None and second_req != first_req
    assert len(left_side._retained_tiles) == 1
    worker.ready.emit(worker.key, second_req, tile_pixels, second_req[2])
    assert len(left_side.tiles) == 1
    assert not left_side._retained_tiles

    # Extreme pan drag is clamped so at least 40 px of the image stays on screen.
    _mouse(widget, QEvent.Type.MouseButtonPress, (30, 110),
           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    _mouse(widget, QEvent.Type.MouseMove, (50000, 50000),
           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    _mouse(widget, QEvent.Type.MouseButtonRelease, (50000, 50000),
           Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    rect = widget._target_rect()
    assert rect.x() <= widget.width() - 40.0 + 1e-6
    assert rect.y() <= widget.height() - 40.0 + 1e-6
    widget.clear()
    widget.close()


def test_compare_source_canvas_debounces_pan_zoom_and_retains_prior_tiles(qapp):
    canvas = SourceCanvas("Source")
    canvas.resize(300, 240)
    pix = QPixmap(1200, 900)
    pix.fill(QColor(20, 30, 40))
    canvas.set_pixmap(pix, source_size=(1200, 900), preview_only=True)
    canvas.show()
    qapp.processEvents()

    emissions = []
    canvas.viewportChanged.connect(lambda: emissions.append(1))

    # Ctrl+wheel debounces viewportChanged instead of emitting synchronously per tick.
    for _ in range(4):
        _wheel(canvas, (150, 120), 120, control=True)
    assert emissions == []
    assert canvas._viewport_timer.isActive()
    canvas._viewport_timer.timeout.emit()
    assert len(emissions) == 1

    # Pan mouseMove debounces, mouseRelease flushes immediately.
    emissions.clear()
    _mouse(canvas, QEvent.Type.MouseButtonPress, (150, 120),
           Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    _mouse(canvas, QEvent.Type.MouseMove, (180, 140),
           Qt.MouseButton.NoButton, Qt.MouseButton.RightButton)
    assert emissions == []
    _mouse(canvas, QEvent.Type.MouseButtonRelease, (180, 140),
           Qt.MouseButton.RightButton, Qt.MouseButton.NoButton)
    assert len(emissions) == 1

    # stash_tiles retains prior tiles until clear_retained_tiles; clear_tiles wipes both.
    tile = QPixmap(64, 64)
    tile.fill(QColor(0, 200, 0))
    canvas.add_tile((0, 0, 64, 64), tile)
    assert len(canvas._tiles) == 1
    canvas.stash_tiles()
    assert canvas._tiles == []
    assert len(canvas._retained_tiles) == 1
    canvas.add_tile((10, 10, 64, 64), tile)
    canvas.clear_retained_tiles()
    assert canvas._retained_tiles == []
    assert len(canvas._tiles) == 1
    canvas.clear_tiles()
    assert canvas._tiles == [] and canvas._retained_tiles == []
    canvas.close()


# ---------------------------------------------------------------------------
# 3. Retouch brush_mask memoization, incremental undo/redo, source retention
# ---------------------------------------------------------------------------


def test_brush_mask_memoized_and_read_only():
    m1 = brush_mask(33, 0.5)
    m2 = brush_mask(33, 0.50001)
    assert m1 is m2
    assert not m1.flags.writeable
    with pytest.raises(ValueError):
        m1[0, 0] = 0.0


def test_retouch_document_undo_redo_return_dirty_rect():
    from core.render_context import build_render_context

    base = np.full((300, 300, 3), 100, np.uint8)
    src = np.full((300, 300, 3), 220, np.uint8)
    context = build_render_context([src], base, filenames=["f0.png"], algorithm="test")
    doc = RetouchDocument(base, context)
    assert doc.undo() is False
    doc.diameter = 12
    doc.begin_stroke(20, 20)
    doc.end_stroke()

    undo_rect = doc.undo()
    assert isinstance(undo_rect, tuple) and len(undo_rect) == 4
    x0, y0, x1, y1 = undo_rect
    assert (x0, y0, x1, y1) == (0, 0, 128, 128)
    assert np.array_equal(doc.working, base)

    redo_rect = doc.redo()
    assert redo_rect == (0, 0, 128, 128)
    assert doc.working[20, 20, 0] > 100


def test_retouch_dialog_incremental_undo_redo_and_non_blanking_source_and_apply(qapp, monkeypatch):
    from PyQt6.QtTest import QTest
    from controllers.output_manager import OutputRecord
    from core.render_context import build_render_context
    from dialogs.retouch import RetouchDialog

    base = np.full((300, 300, 3), 60, np.uint8)
    frames = [
        np.full((300, 300, 3), 180, np.uint8),
        np.full((300, 300, 3), 220, np.uint8),
    ]
    context = build_render_context(frames, base, filenames=["f0.png", "f1.png"], algorithm="test")
    record = OutputRecord.create(base, "result.png", context=context)
    dialog = RetouchDialog(None, record=record, apply_callback=lambda *_: (True, ""))
    dialog.show()
    for _ in range(500):
        if dialog._source_ready:
            break
        QTest.qWait(10)
    assert dialog._source_ready
    try:
        dialog.doc.diameter = 16
        assert dialog.begin_stroke(20, 20)
        dialog.end_stroke()

        # Verify _can_apply() stays True even while another source frame is loading.
        dialog._source_loading = True
        assert dialog._can_apply()
        dialog._source_loading = False

        # Verify switching to an uncached frame keeps prior source preview visible.
        prior_img = dialog._source_qimage
        assert prior_img is not None
        dialog.doc.source_navigation_previews[1] = None
        dialog._select_frame(1)
        assert dialog._source_loading
        assert dialog._source_qimage is prior_img
        assert dialog._can_apply()

        # Verify undo/redo patch incrementally instead of full-image refresh.
        patches = []
        real_patch = dialog._patch_result
        monkeypatch.setattr(dialog, "_patch_result", lambda r: (patches.append(r), real_patch(r)))
        monkeypatch.setattr(
            dialog,
            "_refresh_result_display",
            lambda: (_ for _ in ()).throw(AssertionError("unexpected full redraw on stroke undo/redo")),
        )

        dialog._undo()
        assert patches == [(0, 0, 128, 128)]
        assert np.array_equal(dialog.doc.working, base)

        dialog._redo()
        assert patches == [(0, 0, 128, 128), (0, 0, 128, 128)]
        assert dialog.doc.working[20, 20, 0] > 60
    finally:
        dialog._allow_close = True
        dialog.close()
