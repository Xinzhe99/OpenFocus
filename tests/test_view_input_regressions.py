"""Regression coverage for image-view input and geometry."""
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QApplication

from widgets.depth_surface import DepthSurface
from widgets.magnifier_label import MagnifierLabel
from widgets.wipe_compare import WipeCompareWidget


def mouse(widget, kind, point, button, buttons):
    point = QPointF(*point)
    QApplication.sendEvent(widget, QMouseEvent(kind, point, point, button, buttons,
                                               Qt.KeyboardModifier.NoModifier))


def wheel(widget, point, delta, *, angle_x=0, pixel=QPoint(), phase=Qt.ScrollPhase.NoScrollPhase):
    pos = QPointF(*point)
    event = QWheelEvent(pos, pos, pixel, QPoint(angle_x, delta), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, phase, False)
    QApplication.sendEvent(widget, event)
    return event


@pytest.mark.parametrize('dpr', [1, 1.25, 2])
def test_roi_right_drag_preserves_source_and_clears_state(qapp, dpr):
    label = MagnifierLabel()
    label.resize(600, 400)
    pixmap = QPixmap(600, 400)
    pixmap.setDevicePixelRatio(dpr)
    label.set_display_pixmap(pixmap, source_size=(600, 400))
    label.roi_mode = True
    label.set_roi_rect(QRectF(100, 100, 300, 200))
    changed = []
    label.roiChanged.connect(lambda rect: changed.append(rect))
    mouse(label, QEvent.Type.MouseButtonPress, (200, 200), Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    mouse(label, QEvent.Type.MouseMove, (230, 220), Qt.MouseButton.NoButton, Qt.MouseButton.RightButton)
    mouse(label, QEvent.Type.MouseButtonRelease, (230, 220), Qt.MouseButton.RightButton, Qt.MouseButton.NoButton)
    assert label._pan_offset == QPointF(30, 20)
    assert label.get_roi_rect() == QRectF(100, 100, 300, 200)
    assert not label._is_panning and not changed
    wheel(label, (300, 200), 120)
    assert label._zoom_factor > 1
    label.close()


@pytest.mark.parametrize('cancel', ['mode', 'clear', 'empty', 'replace', 'outside'])
def test_roi_pan_cancellation(qapp, cancel):
    label = MagnifierLabel()
    label.resize(600, 400)
    pixmap = QPixmap(600, 400)
    label.set_display_pixmap(pixmap)
    label.roi_mode = True
    label.set_roi_rect(QRectF(100, 100, 300, 200))
    changed, exited = [], []
    label.roiChanged.connect(lambda rect: changed.append(rect))
    label.roiModeExitRequested.connect(lambda: exited.append(True))
    point = (20, 20) if cancel == 'outside' else (200, 200)
    mouse(label, QEvent.Type.MouseButtonPress, point, Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    if cancel == 'mode':
        label.roi_mode = False
    elif cancel == 'clear':
        label.clear()
    elif cancel == 'empty':
        label.set_display_pixmap(None)
    elif cancel == 'replace':
        label.set_display_pixmap(QPixmap(600, 400))
    assert not label._is_panning and label._roi_state == 'none'
    assert not changed
    assert bool(exited) == (cancel == 'outside')
    label.close()


def test_roi_left_create_move_and_resize(qapp):
    label = MagnifierLabel()
    label.resize(600, 400)
    label.set_display_pixmap(QPixmap(600, 400))
    label.roi_mode = True
    changes = []
    label.roiChanged.connect(lambda rect: changes.append(QRectF(rect)))

    def drag(start, end):
        start = label._map_from_base(QPointF(*start))
        end = label._map_from_base(QPointF(*end))
        mouse(label, QEvent.Type.MouseButtonPress, (start.x(), start.y()), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
        mouse(label, QEvent.Type.MouseMove, (end.x(), end.y()), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
        mouse(label, QEvent.Type.MouseButtonRelease, (end.x(), end.y()), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)

    drag((100, 100), (300, 250))
    assert label.get_roi_rect() == QRectF(100, 100, 200, 150)
    drag((200, 175), (220, 185))
    assert label.get_roi_rect() == QRectF(120, 110, 200, 150)
    drag((320, 260), (340, 280))
    assert label.get_roi_rect() == QRectF(120, 110, 220, 170)
    assert len(changes) == 3
    label.close()


def test_wipe_missing_images_and_drag_reset(qapp):
    widget = WipeCompareWidget()
    widget.resize(900, 600)
    pixmap = QPixmap(600, 400)
    widget.set_images(None, pixmap, independent=True)
    wheel(widget, (100, 300), 120)
    assert widget._pan == QPointF()
    widget.set_comparison_enabled(False)
    rect = widget._image_rect(*widget._regions()[0])
    point = rect.center()
    wheel(widget, (point.x(), point.y()), 120)
    assert widget._image_rect(*widget._regions()[0]).center() == point
    widget._drag_mode = 'pan'
    widget.set_comparison_enabled(True)
    assert not widget._drag_mode
    widget._drag_mode = 'divider'
    widget.clear()
    assert not widget._drag_mode
    wheel(widget, (100, 300), 120)
    widget.close()

@pytest.mark.parametrize('divider', [0.2, 0.5, 0.8])
@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('dpr', [1, 1.25, 2])
def test_independent_wipe_anchor(qapp, divider, side, dpr):
    widget = WipeCompareWidget()
    widget.resize(900, 600)
    left, right = QPixmap(600, 400), QPixmap(400, 600)
    left.setDevicePixelRatio(dpr)
    right.setDevicePixelRatio(dpr)
    widget.set_images(left, right, independent=True)
    widget._divider_ratio = divider
    widget._pan = QPointF(7, -9)
    region = widget._regions()[side]
    rect = widget._image_rect(*region)
    clip = QRectF(region[3])
    visible = rect.intersected(clip)
    point = visible.center()
    fraction = lambda r: ((point.x() - r.x()) / r.width(), (point.y() - r.y()) / r.height())
    before = fraction(rect)
    for delta in (120, 120, -120, -120):
        wheel(widget, (point.x(), point.y()), delta)
        assert fraction(widget._image_rect(*region)) == pytest.approx(before, abs=1e-6)
    widget.close()


@pytest.mark.parametrize('angle_x,pixel,phase', [(0, QPoint(), Qt.ScrollPhase.ScrollBegin),
    (0, QPoint(), Qt.ScrollPhase.ScrollEnd), (120, QPoint(), Qt.ScrollPhase.NoScrollPhase),
    (0, QPoint(0, 25), Qt.ScrollPhase.ScrollUpdate)])
def test_depth_ignores_zero_vertical_angle(qapp, angle_x, pixel, phase):
    widget = DepthSurface()
    cache = object()
    widget._cache = cache
    event = wheel(widget, (20, 20), 0, angle_x=angle_x, pixel=pixel, phase=phase)
    assert widget._zoom == 1 and widget._cache is cache
    assert not event.isAccepted()
    widget.close()


def test_depth_wheel_limits_and_cache(qapp):
    widget = DepthSurface()
    wheel(widget, (20, 20), 120)
    assert widget._zoom == pytest.approx(1.15)
    wheel(widget, (20, 20), -120)
    assert widget._zoom == pytest.approx(1.0)
    widget._zoom = 6.0
    cache = object()
    widget._cache = cache
    wheel(widget, (20, 20), 120)
    assert widget._zoom == 6.0 and widget._cache is cache
    widget._zoom = 0.2
    wheel(widget, (20, 20), -120)
    assert widget._zoom == 0.2 and widget._cache is cache
    widget.close()
