"""Coordinate/pixel regressions use tiny synthetic frames, never camera RAW files."""
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QApplication

from widgets.magnifier_label import MagnifierLabel


def coordinate_grid(width=317, height=193):
    image = QImage(width, height, QImage.Format.Format_RGB32)
    for y in range(height):
        for x in range(width):
            # Coordinates stay distinguishable after smooth resampling.
            image.setPixelColor(x, y, QColor(int(230 * x / width),
                                             int(230 * y / height),
                                             35 if (x // 19 + y // 19) % 2 else 170))
    return QPixmap.fromImage(image)


@pytest.fixture
def view(qapp):
    label = MagnifierLabel()
    label.resize(503, 347)
    label.show()
    yield label
    label.close()


def mouse(label, kind, pos, button=Qt.MouseButton.NoButton):
    buttons = Qt.MouseButton.LeftButton if kind != QMouseEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton
    event = QMouseEvent(kind, pos, label.mapToGlobal(pos), button, buttons,
                        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(label, event)


def wheel(label, pos, control=False, held=False):
    event = QWheelEvent(pos, label.mapToGlobal(pos), QPoint(), QPoint(0, 120),
                        Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.ControlModifier if control else Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(label, event)


def pixel(pixmap, point):
    dpr = pixmap.devicePixelRatioF()
    return pixmap.toImage().pixelColor(round(point.x() * dpr), round(point.y() * dpr))


def assert_color(actual, expected, tolerance=5):
    for channel in ("red", "green", "blue"):
        assert abs(getattr(actual, channel)() - getattr(expected, channel)()) <= tolerance


def install_loupe(view, pixmap, key, request=None):
    request = request or view._region_requests['loupe']
    return view.set_source_region(pixmap, (0, 0, view._source_size.width(), view._source_size.height()),
                                  source_key=key, request=request)


@pytest.mark.parametrize("dpr", [1.0, 1.25, 2.0])
@pytest.mark.parametrize("zoom,pan", [(0.2, (0, 0)), (1.0, (13.25, -9.5)), (3.7, (-389.5, 186.25))])
def test_loupe_centre_matches_drawn_pixel_at_edges(view, qapp, dpr, zoom, pan):
    original = coordinate_grid()
    original.setDevicePixelRatio(dpr)
    view.set_display_pixmap(original)
    view._zoom_factor = zoom
    view._pan_offset = QPointF(*pan)
    view.devicePixelRatioF = lambda: dpr
    view.refresh_display()
    view.set_magnifier_settings(size=96, zoom=3)
    # A corner and off-centre points reproduce the former source-window clamp.
    for fraction in ((0.025, 0.04), (0.65, 0.7), (0.975, 0.95)):
        source_point = QPointF(original.width() * fraction[0], original.height() * fraction[1])
        pos = view._map_from_base(source_point)
        # Panning can put some points outside the viewport. Geometry must still
        # hold there; rendering comparisons apply to visible points only.
        view._magnifier_pos = pos
        sample, target = view._magnifier_rects()
        assert target.center().x() == pytest.approx(pos.x())
        assert target.center().y() == pytest.approx(pos.y())
        assert sample.center().x() == pytest.approx(source_point.x())
        assert sample.center().y() == pytest.approx(source_point.y())
        mapped = view._map_from_base(sample.center())
        assert mapped.x() == pytest.approx(pos.x())
        assert mapped.y() == pytest.approx(pos.y())
        if 2 <= pos.x() < view.width() - 2 and 2 <= pos.y() < view.height() - 2:
            view._magnifier_active = False
            before = view.grab()
            view._magnifier_active = True
            after = view.grab()
            # Red/green encode position. The blue checkerboard is averaged
            # by the reduced base image and resolved differently by the loupe.
            actual, expected = pixel(after, pos), pixel(before, pos)
            for channel in ("red", "green"):
                assert abs(getattr(actual, channel)() - getattr(expected, channel)()) <= 12
    view._magnifier_active = False


def test_wheel_rebuilds_composite_after_anchor_pan(view, qapp):
    view.set_display_pixmap(coordinate_grid())
    view._zoom_factor = 2.5
    view._pan_offset = QPointF(35.25, -19.5)
    view.refresh_display()
    pos = QPointF(211, 179)
    source_point = view._map_to_base(pos)
    before = pixel(view.pixmap(), pos)
    wheel(view, pos)
    assert view._map_to_base(pos).x() == pytest.approx(source_point.x())
    assert view._map_to_base(pos).y() == pytest.approx(source_point.y())
    # Test the actual cached pixels immediately, not only invertible formulas.
    assert_color(pixel(view.pixmap(), pos), before)


def test_move_and_wheel_keep_cursor_as_loupe_centre(view, qapp):
    view.set_display_pixmap(coordinate_grid())
    start = QPointF(190, 140)
    finish = QPointF(340, 225)
    mouse(view, QMouseEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    mouse(view, QMouseEvent.Type.MouseMove, finish)
    size, zoom = view._magnifier_size, view._magnifier_zoom
    wheel(view, finish, control=True, held=True)
    assert view._magnifier_size == pytest.approx(size * 1.1)
    assert view._magnifier_zoom == zoom
    wheel(view, finish, held=True)
    assert view._magnifier_zoom == pytest.approx(zoom * 1.2)
    sample, target = view._magnifier_rects()
    assert target.center() == finish
    assert view._map_from_base(sample.center()).x() == pytest.approx(finish.x())
    assert view._map_from_base(sample.center()).y() == pytest.approx(finish.y())
    mouse(view, QMouseEvent.Type.MouseButtonRelease, finish, Qt.MouseButton.LeftButton)


def test_lazy_original_pixels_and_stale_completion(view, qapp):
    preview = QPixmap(106, 64)
    preview.fill(QColor("magenta"))
    requests = []
    view.sourceRegionRequested.connect(lambda key, request: requests.append((key, request)))
    # Simulated 100MP domain, with a rounded preview aspect and no large allocation.
    view.set_display_pixmap(preview, source_size=(12500, 7500), source_key=(1, 0))
    pos = QPointF(325, 217)
    mouse(view, QMouseEvent.Type.MouseButtonPress, pos, Qt.MouseButton.LeftButton)
    assert requests[-1][0] == (1, 0)
    assert requests[-1][1][0] == 'loupe'
    before = (view._zoom_factor, QPointF(view._pan_offset), view._map_to_base(pos))
    # A tiny surrogate full-pixel buffer has a different color than the preview.
    original = QPixmap(500, 300)
    original.fill(QColor("lime"))
    original_request = view._region_requests['loupe']
    assert install_loupe(view, original, (1, 0))
    assert (view._zoom_factor, view._pan_offset, view._map_to_base(pos)) == before
    assert_color(pixel(view.grab(), pos), QColor("lime"))
    # A portrait frame arrives while the button remains held; old worker results
    # must never appear inside its loupe, including when returning to frame 0.
    portrait = QPixmap(64, 106)
    portrait.fill(QColor("cyan"))
    view.set_display_pixmap(portrait, source_size=(7500, 12500), source_key=(1, 1))
    assert requests[-1][0] == (1, 1)
    assert not install_loupe(view, original, (1, 0), original_request)
    assert view._source_pixmap is None
    sample, target = view._magnifier_rects()
    assert target.center() == pos
    assert view._map_from_base(sample.center()).x() == pytest.approx(pos.x())
    assert view._map_from_base(sample.center()).y() == pytest.approx(pos.y())
    portrait_full = QPixmap(300, 500)
    portrait_full.fill(QColor("yellow"))
    assert install_loupe(view, portrait_full, (1, 1))
    assert_color(pixel(view.grab(), pos), QColor("yellow"))
    view.set_display_pixmap(preview, source_size=(12500, 7500), source_key=(2, 0))
    latest_request = view._region_requests['loupe']
    assert not install_loupe(view, original, (1, 0), original_request)
    view.clear()
    assert not install_loupe(view, original, (2, 0), latest_request)


def test_styled_label_uses_widget_coordinates(view, qapp):
    view.setStyleSheet("border: 3px solid black; padding: 7px")
    view.set_display_pixmap(coordinate_grid())
    pos = QPointF(291, 209)
    expected = pixel(view.pixmap(), pos)
    assert_color(pixel(view.grab(), pos), expected)
    mouse(view, QMouseEvent.Type.MouseButtonPress, pos, Qt.MouseButton.LeftButton)
    assert_color(pixel(view.grab(), pos), expected, tolerance=12)

def test_original_sampling_uses_metadata_scale_in_both_axes(view, qapp):
    original = coordinate_grid(500, 300)
    preview = QPixmap(106, 64)
    preview.fill(QColor("magenta"))
    view.set_display_pixmap(preview, source_size=(12500, 7500), source_key="frame",
                            source_pixmap=original)
    scale = min(view.width() / 12500, view.height() / 7500)
    # Compute the displayed point independently of the widget's mapping helpers.
    fx, fy = .62, .73
    pos = QPointF((view.width() - 12500 * scale) / 2 + fx * 12500 * scale,
                  (view.height() - 7500 * scale) / 2 + fy * 7500 * scale)
    mouse(view, QMouseEvent.Type.MouseButtonPress, pos, Qt.MouseButton.LeftButton)
    expected = original.toImage().pixelColor(round(fx * 500), round(fy * 300))
    assert_color(pixel(view.grab(), pos), expected)
    sample, _ = view._magnifier_rects()
    assert sample.center().x() == pytest.approx(fx * 12500)
    assert sample.center().y() == pytest.approx(fy * 7500)



@pytest.mark.parametrize('dpr', [1.0, 1.25, 2.0])
@pytest.mark.parametrize('source_size', [(10000, 10000), (48000, 32000), (8000, 24000)])
def test_large_source_reaches_native_pixels_with_anchored_pan_and_roi(view, qapp, source_size, dpr):
    view.devicePixelRatioF = lambda: dpr
    preview = QPixmap(96, 64)
    preview.fill(QColor('gray'))
    view.set_display_pixmap(preview, source_size=source_size, source_key=source_size)
    view._zoom_factor = 2
    view.refresh_display()
    pos = QPointF(291, 209)
    native = view._map_to_base(pos)
    view.set_roi_rect(QRectF(native.x() - 25, native.y() - 20, 50, 40))
    roi = view.get_roi_rect()
    for _ in range(100):
        wheel(view, pos)
    scale, _, _ = view._get_layout_params()
    assert scale * view.devicePixelRatioF() == pytest.approx(8)
    assert view._zoom_factor > 10
    assert view._map_to_base(pos).x() == pytest.approx(native.x())
    assert view._map_to_base(pos).y() == pytest.approx(native.y())
    assert view.get_roi_rect() == roi
    assert view._map_to_base(view._map_from_base(roi.topLeft())).x() == pytest.approx(roi.x())
    assert view.pixmap().width() <= round(view.width() * view.devicePixelRatioF())
    assert view.pixmap().height() <= round(view.height() * view.devicePixelRatioF())


@pytest.mark.parametrize('dpr', [1.0, 1.25, 2.0])
def test_large_loupe_starts_at_native_and_stale_regions_cannot_reappear(view, qapp, dpr):
    view.devicePixelRatioF = lambda: dpr
    preview = QPixmap(100, 100)
    preview.fill(QColor('gray'))
    view.set_display_pixmap(preview, source_size=(10000, 10000), source_key='large')
    pos = QPointF(250, 170)
    mouse(view, QMouseEvent.Type.MouseButtonPress, pos, Qt.MouseButton.LeftButton)
    source, target = view._magnifier_rects()
    assert target.width() / source.width() * view.devicePixelRatioF() >= 1 - 1e-12
    assert view._magnifier_zoom > 10
    first = view._region_requests['loupe']
    tile = QPixmap(32, 32)
    tile.fill(QColor('green'))
    mouse(view, QMouseEvent.Type.MouseMove, QPointF(270, 190))
    assert not view.set_source_region(tile, first[2], source_key='large', request=first)
    for _ in range(60):
        wheel(view, QPointF(270, 190), held=True)
    source, target = view._magnifier_rects()
    assert target.width() / source.width() * view.devicePixelRatioF() == pytest.approx(16)
    zoom = view._magnifier_zoom
    wheel(view, QPointF(270, 190), control=True, held=True)
    assert view._magnifier_zoom == zoom
    source, target = view._magnifier_rects()
    assert target.width() / source.width() * view.devicePixelRatioF() == pytest.approx(16)
