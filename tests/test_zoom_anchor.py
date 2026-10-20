"""Zoom must keep the image point under the cursor fixed.

Regression: after a right-drag pan the wheel zoom anchored somewhere else.
Scaling the pan offset alone is not enough — the drawn image is also centred
by (widget - scaled)/2, which changes with the zoom, and the pan clamp could
silently move the anchor. Measured drift before the fix: 378 px on a
4000x3000 image in a 400x300 view.
"""
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap, QWheelEvent


@pytest.fixture()
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _send_mouse(widget, kind, pos, button, buttons):
    from PyQt6.QtWidgets import QApplication
    event = QMouseEvent(kind, QPointF(pos), QPointF(pos), button, buttons,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)
    QApplication.processEvents()


def _send_wheel(widget, pos, delta):
    from PyQt6.QtWidgets import QApplication
    event = QWheelEvent(QPointF(pos), QPointF(pos), QPoint(0, 0),
                        QPoint(0, delta), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)
    QApplication.processEvents()


def _pan(widget, dx, dy, view=(400, 300)):
    _send_mouse(widget, QEvent.Type.MouseButtonPress,
                QPointF(view[0] / 2, view[1] / 2),
                Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    for i in range(1, 13):
        _send_mouse(widget, QEvent.Type.MouseMove,
                    QPointF(view[0] / 2 + dx * i / 12,
                            view[1] / 2 + dy * i / 12),
                    Qt.MouseButton.NoButton, Qt.MouseButton.RightButton)
    _send_mouse(widget, QEvent.Type.MouseButtonRelease,
                QPointF(view[0] / 2 + dx, view[1] / 2 + dy),
                Qt.MouseButton.RightButton, Qt.MouseButton.NoButton)


class TestMagnifierZoomAnchor:
    def _label(self, qapp, image=(4000, 3000), view=(400, 300)):
        from widgets.magnifier_label import MagnifierLabel
        label = MagnifierLabel()
        label.resize(*view)
        pixmap = QPixmap(*image)
        pixmap.fill(0)
        label.set_display_pixmap(pixmap)
        label.show()
        qapp.processEvents()
        return label

    @pytest.mark.parametrize("pan", [(0, 0), (60, 40), (200, 150),
                                     (-200, -150), (400, 300)])
    @pytest.mark.parametrize("cursor", [(200, 150), (380, 280)])
    @pytest.mark.parametrize("zoom_in", [True, False])
    def test_cursor_point_stays_fixed(self, qapp, pan, cursor, zoom_in):
        label = self._label(qapp)
        try:
            _pan(label, *pan)
            point = QPointF(*cursor)
            before = label._map_to_base(point)
            pix = label._base_pixmap
            if not (0 <= before.x() <= pix.width()
                    and 0 <= before.y() <= pix.height()):
                pytest.skip("cursor is over the canvas, not the image")
            _send_wheel(label, point, 120 if zoom_in else -120)
            scaled = label._cached_scaled_pixmap
            if (scaled is not None and scaled.width() <= label.width()
                    and scaled.height() <= label.height()):
                pytest.skip("zoomed out to fit the window (re-centres)")
            after = label._map_to_base(point)
            drift = ((after.x() - before.x()) ** 2
                     + (after.y() - before.y()) ** 2) ** 0.5
            assert drift < 1.0, (
                f"anchor drifted {drift:.1f} image px (pan={pan}, "
                f"cursor={cursor}, zoom_in={zoom_in})")
        finally:
            label.close()

    def test_repeated_zoom_steps_stay_anchored(self, qapp):
        label = self._label(qapp)
        try:
            _pan(label, 200, 150)
            point = QPointF(300, 220)
            before = label._map_to_base(point)
            for _ in range(5):
                _send_wheel(label, point, 120)
            after = label._map_to_base(point)
            drift = ((after.x() - before.x()) ** 2
                     + (after.y() - before.y()) ** 2) ** 0.5
            assert drift < 2.0, drift
        finally:
            label.close()

    def test_cursor_over_canvas_anchors_on_the_image(self, qapp):
        """With the cursor over empty canvas there is no image point to keep
        fixed; the zoom must fall back to the image centre instead of
        dragging the canvas (the 378 px case)."""
        label = self._label(qapp)
        try:
            _pan(label, 360, 260)
            point = QPointF(20, 20)
            before = label._map_to_base(point)
            pix = label._base_pixmap
            assert not (0 <= before.x() <= pix.width()), (
                "test setup: cursor should be outside the image")
            _send_wheel(label, point, 120)
            # the image must still cover the viewport centre
            centre_base = label._map_to_base(
                QPointF(label.width() / 2, label.height() / 2))
            assert 0 <= centre_base.x() <= pix.width()
            assert 0 <= centre_base.y() <= pix.height()
        finally:
            label.close()


class TestWipeZoomAnchor:
    def test_cursor_point_stays_fixed(self, qapp):
        from widgets.wipe_compare import WipeCompareWidget
        widget = WipeCompareWidget()
        widget.resize(900, 600)
        pixmap = QPixmap(1600, 1200)
        pixmap.fill(0)
        widget.set_images(pixmap, pixmap)
        widget.show()
        qapp.processEvents()

        def fraction(point):
            rect = widget._target_rect()
            return QPointF((point.x() - rect.x()) / rect.width(),
                           (point.y() - rect.y()) / rect.height())

        try:
            for pan in ((0, 0), (100, 70), (-150, -120)):
                for cursor in ((30, 30), (450, 300), (870, 570)):
                    widget.reset_view()
                    widget._pan = QPointF(*pan)
                    widget._zoom = 1.0
                    widget.update()
                    qapp.processEvents()
                    point = QPointF(*cursor)
                    start = fraction(point)
                    if not (0 <= start.x() <= 1 and 0 <= start.y() <= 1):
                        continue
                    for delta in (120, -120):
                        before = fraction(point)
                        _send_wheel(widget, point, delta)
                        after = fraction(point)
                        drift = ((after.x() - before.x()) ** 2
                                 + (after.y() - before.y()) ** 2) ** 0.5
                        assert drift < 0.005, (
                            f"wipe anchor drifted {drift:.3%} of the image")
        finally:
            widget.close()
