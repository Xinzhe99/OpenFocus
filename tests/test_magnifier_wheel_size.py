"""Press-and-hold magnifier size and zoom gestures."""
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QPixmap, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from widgets.magnifier_label import MAGNIFIER_MAX_SIZE, MAGNIFIER_MIN_SIZE, MagnifierLabel


@pytest.fixture
def labels(qapp):
    result = []
    for _ in range(2):
        label = MagnifierLabel()
        label.resize(600, 600)
        pixmap = QPixmap(600, 600)
        pixmap.fill(Qt.GlobalColor.blue)
        label.set_display_pixmap(pixmap)
        label.show()
        result.append(label)
    qapp.processEvents()
    yield result
    for label in result:
        label.close()


def wheel(label, delta, control=False, held=True):
    pos = QPointF(300, 300)
    event = QWheelEvent(pos, pos, QPoint(), QPoint(0, delta),
                        Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.ControlModifier if control else Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(label, event)
    return event


def press(label):
    QTest.mousePress(label, Qt.MouseButton.LeftButton, pos=QPoint(300, 300))
    assert label._magnifier_active


def release(label):
    QTest.mouseRelease(label, Qt.MouseButton.LeftButton, pos=QPoint(300, 300))
    assert not label._magnifier_active


def test_resize_preserves_zoom_and_each_label_remembers_its_size(labels):
    source, output = labels
    press(source)
    original_size = source._magnifier_size
    original_zoom = source._magnifier_zoom
    assert wheel(source, -60, control=True).isAccepted()
    assert 0 < source._magnifier_size < original_size
    assert source._magnifier_zoom == original_zoom
    assert source._magnifier_pos == QPointF(300, 300)
    smaller_size = source._magnifier_size
    wheel(source, 60, control=True)
    assert source._magnifier_size == pytest.approx(original_size)
    wheel(source, -120, control=True)
    resized = source._magnifier_size
    assert resized < smaller_size
    wheel(source, 120)
    assert source._magnifier_size == resized
    assert source._magnifier_zoom == pytest.approx(original_zoom * 1.2)
    release(source)

    press(output)
    assert output._magnifier_size == original_size
    assert output._magnifier_zoom == original_zoom
    wheel(output, 120, control=True)
    assert output._magnifier_size > original_size
    release(output)
    press(source)
    assert source._magnifier_size == resized
    assert source._magnifier_zoom == pytest.approx(original_zoom * 1.2)
    release(source)


def test_resize_has_finite_bounds_and_inactive_control_wheel_zooms_view(labels):
    label = labels[0]
    press(label)
    zoom = label._magnifier_zoom
    for _ in range(100):
        wheel(label, -120, control=True)
    assert label._magnifier_size == MAGNIFIER_MIN_SIZE
    wheel(label, 120, control=True)
    assert label._magnifier_size > MAGNIFIER_MIN_SIZE
    for _ in range(100):
        wheel(label, 120, control=True)
    assert label._magnifier_size == MAGNIFIER_MAX_SIZE
    assert label._magnifier_zoom == zoom
    release(label)
    size = label._magnifier_size
    view_zoom = label._zoom_factor
    wheel(label, 120, control=True, held=False)
    assert label._zoom_factor > view_zoom
    assert label._magnifier_size == size
    assert label._magnifier_zoom == zoom
