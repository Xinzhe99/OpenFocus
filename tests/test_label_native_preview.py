"""Native and bounded-fit contracts for the label preview."""
import math
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent

from core.source_frames import PREVIEW_EDGE, SourceFrame
from widgets.label_preview import LabelPreview


class RegionWorker(QObject):
    ready = pyqtSignal(object, object, object, object)
    failed = pyqtSignal(object, str)
    finished = pyqtSignal()
    instances = []

    def __init__(self, frame, key, **kwargs):
        super().__init__()
        self.frame, self.key, self.options = frame, key, kwargs
        self.requests = []
        self.cancelled = False
        self.instances.append(self)

    def start(self):
        pass

    def submit(self, request):
        self.requests.append(request)

    def request_cancel(self):
        if not self.cancelled:
            self.cancelled = True
            self.finished.emit()


@pytest.mark.parametrize('dpr', [1.0, 1.25, 2.0])
def test_native_uses_source_domain_and_rejects_stale_tiles(qapp, monkeypatch, tmp_path, dpr):
    from core import source_previews
    monkeypatch.setattr(source_previews, "SourceRegionWorker", RegionWorker)
    monkeypatch.setattr(SourceFrame, "full_resolution", lambda *a, **k: pytest.fail("decoded on GUI"))
    preview_pixels = np.full((96, 144, 3), 12, np.uint8)
    source = SourceFrame(str(tmp_path / "source.png"), 0, (8000, 12000, 3),
                         np.dtype("uint8"), preview_pixels, (0, 0))
    class Preview(LabelPreview):
        def devicePixelRatioF(self):
            return dpr
        def setPixmap(self, pixmap):
            self._supplied_dpr = pixmap.devicePixelRatioF()
            super().setPixmap(pixmap)
    widget = Preview()
    widget.resize(320, 220)
    widget.set_preview(source, {"enabled": False}, 0)
    assert widget._frame is preview_pixels
    widget.set_zoom_mode("actual")
    first = widget._region_worker
    assert widget.native_status()
    widget._pan = (12000, 8000)
    widget._show_rendered()
    worker = widget._region_worker
    assert first.cancelled
    request = worker.requests[0]
    x, y, width, height = request[2]
    assert (x + width, y + height) == (12000, 8000)
    assert (width, height) == (math.ceil((widget.width() - 8) * dpr),
                               math.ceil((widget.height() - 8) * dpr))
    assert widget._supplied_dpr == dpr
    before = widget.pixmap().toImage().pixelColor(0, 0)
    stale = first.requests[0]
    first.ready.emit(first.key, stale, np.full((stale[2][3], stale[2][2], 3), 255, np.uint8), stale[2])
    assert widget.pixmap().toImage().pixelColor(0, 0) == before
    worker.ready.emit(worker.key, request, np.full((height, width, 3), 45, np.uint8), request[2])
    assert not widget.native_status()
    assert widget.pixmap().toImage().pixelColor(0, 0).red() == 45
    widget.close_preview()


@pytest.mark.parametrize('dpr', [1.0, 1.25, 2.0])
def test_large_viewport_requests_tiles_sequentially(qapp, monkeypatch, dpr):
    from core import source_previews
    monkeypatch.setattr(source_previews, "SourceRegionWorker", RegionWorker)
    monkeypatch.setattr(source_previews, "DISPLAY_REGION_MAX_SIDE", 64)
    class Preview(LabelPreview):
        def devicePixelRatioF(self):
            return dpr
        def setPixmap(self, pixmap):
            self._supplied_dpr = pixmap.devicePixelRatioF()
            super().setPixmap(pixmap)
    widget = Preview()
    widget.resize(280, 190)
    widget.set_preview(np.zeros((400, 500, 3), np.uint8), {"enabled": False}, 0)
    widget.set_zoom_mode("actual")
    worker = widget._region_worker
    rects = []
    while widget._native_request is not None:
        assert len(worker.requests) == len(rects) + 1
        request = worker.requests[-1]
        rect = request[2]
        rects.append(rect)
        assert max(rect[2:]) <= 62
        tile = np.full((rect[3], rect[2], 3), 80, np.uint8)
        worker.ready.emit(worker.key, request, tile, rect)
    view_w = min(500, math.ceil((widget.width() - 8) * dpr))
    view_h = min(400, math.ceil((widget.height() - 8) * dpr))
    assert sum(w * h for _, _, w, h in rects) == view_w * view_h
    assert widget._supplied_dpr == dpr
    assert all(x >= 0 and y >= 0 and x + w <= view_w and y + h <= view_h
               for x, y, w, h in rects)
    for i, (x, y, w, h) in enumerate(rects):
        assert all(x + w <= xx or xx + ww <= x or y + h <= yy or yy + hh <= y
                   for xx, yy, ww, hh in rects[i + 1:])
    image = widget.pixmap().toImage()
    assert image.pixelColor(image.width() - 1, image.height() - 1).red() == 80
    widget.close_preview()


@pytest.mark.parametrize("dpr", [1.0, 1.25, 2.0])
def test_native_drag_uses_device_pixels(qapp, monkeypatch, dpr):
    class Preview(LabelPreview):
        def devicePixelRatioF(self):
            return dpr

    widget = Preview()
    widget._zoom_mode = "actual"
    widget._pan = (100.0, 100.0)
    widget._drag_start = QPointF(20, 20)
    monkeypatch.setattr(widget, "_show_rendered", lambda: None)
    event = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(16, 14),
                        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
    widget.mouseMoveEvent(event)
    assert widget._pan == (100 + 4 * dpr, 100 + 6 * dpr)
    widget.close_preview()


@pytest.mark.parametrize("dtype,channels", [(np.uint16, 3), (np.float32, 1),
                                             (np.uint8, 4)])
def test_fit_resizes_once_and_preserves_borrowed_pixels(qapp, monkeypatch, dtype, channels):
    from utils import frame_pixels
    from core import source_frames
    calls = []
    original = source_frames._small
    monkeypatch.setattr(source_frames, "_small", lambda image: (calls.append(image), original(image))[1])
    monkeypatch.setattr(frame_pixels, "run_pixel_worker", lambda parent, title, operation: operation())
    shape = (1600, 2400) if channels == 1 else (1600, 2400, channels)
    image = np.full(shape, 32, dtype=dtype)
    widget = LabelPreview()
    widget.set_preview(image, {"enabled": False}, 0)
    assert max(widget._frame.shape[:2]) <= PREVIEW_EDGE
    widget.set_preview(image, {"enabled": False, "text": "edited"}, 0)
    assert calls == [image]
    assert np.all(image == 32)
    widget.close_preview()


def test_native_failure_recovers_to_fit(qapp, monkeypatch):
    from core import source_previews
    monkeypatch.setattr(source_previews, "SourceRegionWorker", RegionWorker)
    widget = LabelPreview()
    widget.set_preview(np.zeros((300, 400, 3), np.uint8), {"enabled": False}, 0)
    widget.set_zoom_mode("actual")
    worker = widget._region_worker
    worker.failed.emit(worker.key, "source unavailable")
    assert widget.native_status() == "source unavailable"
    assert not widget.pixmap().isNull()
    widget.set_zoom_mode("fit")
    assert not widget.native_status()
    widget.set_zoom_mode("actual")
    assert widget._region_worker is not worker
    widget.close_preview()


def test_native_tile_matches_export_crop(qapp):
    from core.source_previews import SourceRegionWorker, render_source_region
    from utils.validators import LabelAdder
    from widgets.label_preview import _DraftLabels

    frame = np.full((120, 200, 3), 80, np.uint8)
    adder = LabelAdder()
    adder.config.update_config({"enabled": True, "format": "Sample",
                                "font_size": 24, "x_location": 45,
                                "y_location": 70})
    exported = adder.add_label_to_image(frame, 0)
    assert not adder.last_error
    rect = (35, 30, 80, 60)
    tile, sampled = render_source_region(frame, rect, rect[2:])
    worker = SourceRegionWorker(frame, 1, labels=_DraftLabels(adder.config))
    decorated = worker._decorate(tile, sampled, frame)
    np.testing.assert_array_equal(decorated, exported[30:90, 35:115])


def test_fit_failure_discards_previous_frame(qapp, monkeypatch):
    from utils import frame_pixels
    widget = LabelPreview()
    widget.set_preview(np.zeros((50, 50, 3), np.uint8), {"enabled": False}, 0)
    def fail(*args, **kwargs):
        raise MemoryError("fit allocation denied")
    monkeypatch.setattr(frame_pixels, "run_pixel_worker", fail)
    widget.set_preview(np.zeros((60, 60, 3), np.uint8), {"enabled": False}, 0)
    assert widget._rendered.isNull()
    assert widget._frame is None
    assert widget.message() == "error"
    widget.close_preview()


def test_native_renders_one_source_pixel_per_screen_pixel(qapp, monkeypatch):
    from core import source_previews
    monkeypatch.setattr(source_previews, 'SourceRegionWorker', RegionWorker)
    widget = LabelPreview()
    widget.resize(320, 220)
    widget.show()
    qapp.processEvents()
    widget.set_preview(np.zeros((2000, 2000, 3), np.uint8), {'enabled': False}, 0)
    widget.set_zoom_mode('actual')
    worker = widget._region_worker
    request = worker.requests[0]
    x, y, width, height = request[2]
    tile = np.zeros((height, width, 3), np.uint8)
    tile[:, :, 2] = ((np.arange(width) + x) % 2 * 255).astype(np.uint8)
    worker.ready.emit(worker.key, request, tile, request[2])
    qapp.processEvents()
    captured = widget.grab().toImage()
    mid_x, mid_y = captured.width() // 2, captured.height() // 2
    samples = [captured.pixelColor(mid_x + offset, mid_y).red()
               for offset in range(-20, 20)]
    assert set(samples) == {0, 255}
    assert all(a != b for a, b in zip(samples, samples[1:]))
    widget.close_preview()
    widget.close()


def test_screen_change_refreshes_native_viewport(qapp, monkeypatch):
    from PyQt6.QtCore import QEvent
    event_type = (getattr(QEvent.Type, 'DevicePixelRatioChange', None)
                  or getattr(QEvent.Type, 'ScreenChangeInternal', None))
    if event_type is None:
        pytest.skip('Qt does not expose a screen change event')
    widget = LabelPreview()
    widget._zoom_mode = 'actual'
    refreshes = []
    monkeypatch.setattr(widget, '_show_rendered', lambda: refreshes.append(True))
    widget.event(QEvent(event_type))
    assert refreshes
    widget.close_preview()
