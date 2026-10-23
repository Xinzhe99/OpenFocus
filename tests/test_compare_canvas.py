"""Behavioral coverage for persistent source selection and exact detail crops."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QApplication, QWidget

from widgets.compare_canvas import DEFAULT_DETAIL_SIZE, DetailCanvas, SourceCanvas


class _PaintedSource(SourceCanvas):
    def __init__(self, *args, **kwargs):
        self.paint_count = 0
        super().__init__(*args, **kwargs)

    def paintEvent(self, event):
        self.paint_count += 1
        super().paintEvent(event)


@pytest.fixture()
def host(qapp):
    widget = QWidget()
    widget.resize(900, 700)
    yield widget
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def _pixmap(width, height, color=Qt.GlobalColor.black):
    result = QPixmap(width, height)
    result.fill(color)
    return result


def _source(host, qapp, image=(600, 500), view=(608, 508)):
    canvas = _PaintedSource("Source", host)
    canvas.resize(*view)
    canvas.set_pixmap(_pixmap(*image))
    host.show()
    canvas.show()
    qapp.processEvents()
    return canvas


def _mouse(canvas, kind, position, button, buttons):
    event = QMouseEvent(kind, position, position, button, buttons,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(canvas, event)
    QApplication.processEvents()


def _wheel(canvas, position, delta, control=False):
    modifiers = (Qt.KeyboardModifier.ControlModifier if control
                 else Qt.KeyboardModifier.NoModifier)
    event = QWheelEvent(position, position, QPoint(), QPoint(0, delta),
                        Qt.MouseButton.NoButton, modifiers,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(canvas, event)
    QApplication.processEvents()


def _assert_point(actual, expected):
    assert actual.x() == pytest.approx(expected.x(), abs=1e-8)
    assert actual.y() == pytest.approx(expected.y(), abs=1e-8)


def _marker_point(canvas):
    x, y, width, _ = canvas.region()
    return canvas.image_to_widget(QPointF(x + width / 2, y))


def _assert_marker_visible(canvas):
    point = _marker_point(canvas)
    image = canvas.grab().toImage()
    x, y = round(point.x()), round(point.y())
    assert any(image.pixelColor(px, py) == QColor(Qt.GlobalColor.white)
               for px in range(x - 1, x + 2)
               for py in range(y - 1, y + 2))


def test_drag_repaints_marker_and_release_leave_retain_region(host, qapp):
    canvas = _source(host, qapp)
    canvas.set_magnification(4)
    changed = []
    canvas.regionChanged.connect(changed.append)
    _mouse(canvas, QEvent.Type.MouseButtonPress,
           canvas.image_to_widget(QPointF(150, 250)),
           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    old_marker = _marker_point(canvas)
    _assert_marker_visible(canvas)
    paints = canvas.paint_count

    _mouse(canvas, QEvent.Type.MouseMove,
           canvas.image_to_widget(QPointF(350, 250)),
           Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    assert canvas.paint_count > paints
    assert changed[-1] == canvas.region()
    assert isinstance(changed[-1], tuple)
    _assert_marker_visible(canvas)
    screenshot = canvas.grab().toImage()
    assert screenshot.pixelColor(round(old_marker.x()), round(old_marker.y())) == QColor(
        Qt.GlobalColor.black)

    # Release differs from the last move: its coordinate must be committed.
    _mouse(canvas, QEvent.Type.MouseButtonRelease,
           canvas.image_to_widget(QPointF(450, 350)),
           Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    expected = (round(450 - 85 / 2), round(350 - 85 / 2), 85, 85)
    assert canvas.region() == expected
    QApplication.sendEvent(canvas, QEvent(QEvent.Type.Leave))
    qapp.processEvents()
    assert canvas.region() == expected
    _assert_marker_visible(canvas)


@pytest.mark.parametrize("button", [Qt.MouseButton.RightButton,
                                    Qt.MouseButton.MiddleButton])
def test_pan_and_anchored_zoom_share_paint_mapping(host, qapp, button):
    canvas = _source(host, qapp)
    canvas.set_magnification(4)
    canvas.set_region_centre(QPointF(250, 250))
    selected = canvas.region()
    changes = []
    canvas.regionChanged.connect(changes.append)
    image_point = QPointF(123.25, 201.75)
    original_widget_point = canvas.image_to_widget(image_point)
    start, finish = QPointF(200, 200), QPointF(260, 240)
    _mouse(canvas, QEvent.Type.MouseButtonPress, start, button, button)
    _mouse(canvas, QEvent.Type.MouseMove, finish,
           Qt.MouseButton.NoButton, button)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, finish,
           button, Qt.MouseButton.NoButton)
    _assert_point(canvas.image_to_widget(image_point),
                  original_widget_point + finish - start)
    _assert_point(canvas.widget_to_image(canvas.image_to_widget(image_point)),
                  image_point)
    _assert_marker_visible(canvas)

    cursor = QPointF(360, 300)
    anchor = canvas.widget_to_image(cursor)
    for delta in (120, 120, -120):
        _wheel(canvas, cursor, delta, control=True)
        _assert_point(canvas.widget_to_image(cursor), anchor)
        _assert_point(canvas.widget_to_image(canvas.image_to_widget(image_point)),
                      image_point)
        _assert_marker_visible(canvas)
    assert canvas.region() == selected
    assert changes == []


@pytest.mark.parametrize("dimensions", [(1, 1), (11, 19), (80, 1200),
                                       (1200, 80), (800, 900)])
@pytest.mark.parametrize("magnification", [0.25, 1.0, 3.0, 8.0])
def test_square_region_stays_inside_every_image_edge(host, qapp, dimensions,
                                                     magnification):
    canvas = _source(host, qapp, image=dimensions)
    canvas.set_magnification(magnification)
    width, height = dimensions
    side = min(width, height, max(1, round(DEFAULT_DETAIL_SIZE / magnification)))
    for point in (QPointF(-1000, -1000), QPointF(width + 1000, height + 1000),
                  QPointF(0, height), QPointF(width, 0),
                  QPointF(width / 2, height / 2)):
        canvas.set_region_centre(point)
        x, y, rw, rh = canvas.region()
        assert all(isinstance(value, int) for value in canvas.region())
        assert rw == rh == side
        assert 0 <= x <= width - side
        assert 0 <= y <= height - side


def test_plain_wheel_changes_region_magnification_only(host, qapp):
    canvas = _source(host, qapp)
    image_point = QPointF(150, 200)
    original = canvas.image_to_widget(image_point)
    original_region = canvas.region()
    _wheel(canvas, QPointF(200, 250), 120)
    assert canvas.magnification() == pytest.approx(1.2)
    assert canvas.region() != original_region
    assert canvas.region()[2] == round(DEFAULT_DETAIL_SIZE / 1.2)
    _assert_point(canvas.image_to_widget(image_point), original)
    canvas.set_magnification(100)
    assert canvas.magnification() == 8.0
    _wheel(canvas, QPointF(200, 250), 120)
    assert canvas.magnification() == 8.0
    canvas.set_magnification(0.01)
    _wheel(canvas, QPointF(200, 250), -120)
    assert canvas.magnification() == 0.25


def test_preview_size_preserves_zoom_and_tracks_image_bounds(host, qapp):
    canvas = _source(host, qapp, image=(1200, 1000))
    canvas.set_magnification(2)
    canvas.set_region_centre(QPointF(600.5, 500.5))
    anchor = canvas.image_to_widget(QPointF(600.5, 500.5))
    for size in range(321, 641):
        canvas.set_detail_size(size)
        x, y, width, height = canvas.region()
        assert abs(x + width / 2 - 600.5) <= 0.5
        assert abs(y + height / 2 - 500.5) <= 0.5
    assert canvas.region() == (440, 340, 320, 320)
    assert canvas.magnification() == 2
    _assert_point(canvas.image_to_widget(QPointF(600.5, 500.5)), anchor)
    _assert_marker_visible(canvas)

    _wheel(canvas, anchor, 120)
    assert canvas.region()[2] == round(640 / 2.4)
    canvas.set_region_centre(QPointF(1200, 1000))
    canvas.set_detail_size(960)
    assert canvas.region() == (800, 600, 400, 400)
    canvas.set_magnification(0.25)
    assert canvas.region()[2:] == (1000, 1000)
    assert canvas.region()[0] + 1000 <= 1200
    assert canvas.region()[1] == 0


def test_no_image_wheel_is_safe(host, qapp):
    canvas = SourceCanvas("Source", host)
    _wheel(canvas, QPointF(10, 10), 120)
    _wheel(canvas, QPointF(10, 10), -120, control=True)
    assert canvas.region() is None
    assert canvas.magnification() == 1.0


def test_first_show_fits_final_geometry_and_resize_preserves_region(host, qapp):
    canvas = SourceCanvas("Source", host)
    canvas.set_pixmap(_pixmap(2400, 1800))
    canvas.resize(608, 508)
    host.show()
    canvas.show()
    qapp.processEvents()
    first = canvas.image_to_widget(QPointF())
    last = canvas.image_to_widget(QPointF(2400, 1800))
    assert first.x() == pytest.approx(4)
    assert first.y() >= 4
    assert last.x() == pytest.approx(canvas.width() - 4)
    assert last.y() <= canvas.height() - 4
    canvas.set_magnification(3)
    canvas.set_region_centre(QPointF(700, 600))
    region = canvas.region()
    changes = []
    canvas.regionChanged.connect(changes.append)
    canvas.resize(700, 550)
    qapp.processEvents()
    assert canvas.region() == region
    assert changes == []
    # Resizing must not silently refit the camera either.
    scaled_width = (canvas.image_to_widget(QPointF(2400, 0))
                    - canvas.image_to_widget(QPointF())).x()
    assert scaled_width == pytest.approx(600)
    canvas.set_pixmap(_pixmap(2400, 1800, Qt.GlobalColor.red))
    assert canvas.region() == region
    assert changes == []
    canvas.set_pixmap(_pixmap(2400, 1800), reset=True)
    assert canvas.region() != region
    canvas.set_pixmap(_pixmap(17, 12))
    assert canvas.region() == (2, 0, 12, 12)


def test_detail_crop_translates_nonzero_image_origin(host, qapp):
    canvas = DetailCanvas("Method", host)
    pixmap = _pixmap(100, 100, Qt.GlobalColor.red)
    painter = QPainter(pixmap)
    painter.fillRect(50, 0, 50, 50, QColor(Qt.GlobalColor.green))
    painter.fillRect(0, 50, 50, 50, QColor(Qt.GlobalColor.blue))
    painter.fillRect(50, 50, 50, 50, QColor(Qt.GlobalColor.yellow))
    painter.end()
    canvas.set_result(pixmap, (1000, 2000, 100, 100))
    canvas.set_region((1050, 2000, 50, 50))
    host.show()
    canvas.show()
    qapp.processEvents()
    assert canvas.has_region()
    image = canvas.grab().toImage()
    for point in (QPoint(20, 100), QPoint(170, 170), QPoint(320, 320)):
        assert image.pixelColor(point) == QColor(Qt.GlobalColor.green)
    canvas.set_region((1000, 2050, 50, 50))
    assert canvas.has_region()
    assert canvas.grab().toImage().pixelColor(170, 170) == QColor(Qt.GlobalColor.blue)
    canvas.setFixedSize(600, 600)
    qapp.processEvents()
    image = canvas.grab().toImage()
    for point in (QPoint(30, 100), QPoint(300, 300), QPoint(570, 570)):
        assert image.pixelColor(point) == QColor(Qt.GlobalColor.blue)


@pytest.mark.parametrize("region", [(990, 2000, 50, 50), (1080, 2000, 50, 50),
                                    (1000, 1980, 50, 50), (1000, 2080, 50, 50)])
def test_detail_missing_pixels_never_shift_the_requested_crop(host, qapp, region):
    canvas = DetailCanvas("Method", host)
    canvas.set_result(_pixmap(100, 100, Qt.GlobalColor.green),
                      (1000, 2000, 100, 100))
    canvas.set_region((1000, 2000, 50, 50))
    host.show()
    canvas.show()
    qapp.processEvents()
    assert canvas.has_region()
    canvas.set_status("Waiting for this region")
    canvas.set_region(region)
    assert not canvas.has_region()
    image = canvas.grab().toImage()
    assert image.pixelColor(50, 100) != QColor(Qt.GlobalColor.green)
    canvas.set_region((1000, 2000, 50, 50))
    assert canvas.has_region()
    canvas.set_result(None, None)
    assert not canvas.has_region()
    assert canvas.grab().toImage().pixelColor(50, 100) != QColor(Qt.GlobalColor.green)
