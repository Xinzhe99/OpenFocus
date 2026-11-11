"""Live preview of a label configuration.

The add-label dialog used to be a blind form: you picked a font size, an
x/y offset and a format string, confirmed, and only then saw whether the
label landed inside the frame, how big it was relative to a real frame, or
whether it showed anything at all. Several combinations drew nothing (see
utils/validators.build_label_text), and nothing in the UI said so.

This widget renders one sample frame with the current settings applied, so
the answer is visible before the dialog closes.
"""
from copy import copy, deepcopy
import math

import numpy as np

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

from utils.validators import LabelAdder, build_label_text
from utils.frame_pixels import preview_pixels as display_image, preview_label_config

# One display-depth copy of the current frame is enough for form edits. The
# source stack already lives in the window; this cache must never grow with it.
MAX_DEVELOPED_CACHE_BYTES = 512 * 1024 * 1024

# Retain cancelled threads until Qt reports their completion.
_LIVE_REGION_WORKERS = set()


class _DraftLabels:
    """Independent single-label fallback for standalone consumers."""

    def __init__(self, config):
        self.config = deepcopy(config)

    def _draw_for(self, image, target, index, source_rect=None):
        config = copy(self.config)
        x, y, width, height = source_rect
        config.x_location -= x
        config.y_location -= y
        adder = LabelAdder()
        adder.config = preview_label_config(config, (height, width), image.shape)
        return adder.add_label_to_image(image, index), adder.last_error


class LabelPreview(QLabel):
    """A scaled preview of a source frame carrying one label."""
    status_changed = pyqtSignal()


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 190)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Ignored)
        self.setObjectName("labelPreview")
        self._adder = LabelAdder()
        self._frame = None
        self._linear_dng_metadata = None
        self._developed_source = None
        self._developed_image = None
        self._developed_metadata = None
        self._developed_shape = None
        self._developed_dtype = None
        self._status = ""
        # Family actually used when the request could not be honoured.
        self.substituted_font = ""
        # Which frame the preview pretends to be, so {value} resolves the way
        # the first real frame will.
        self._index = 0
        self._zoom_mode = "fit"
        self._rendered = QPixmap()
        self._pan = (0, 0)
        self._drag_start = None
        self._source = None
        self._source_shape = None
        self._source_key = None
        self._generation = 0
        self._labels = None
        self._target = "input"
        self._region_worker = None
        self._native_batch = 0
        self._native_serial = 0
        self._native_pending = []
        self._native_request = None
        self._native_pixmap = QPixmap()
        self._native_error = ""
        self._native_loading = False
        self._closed = False

    def set_preview(self, frame, config, index, *, linear_dng_metadata=None,
                    labels=None, target="input"):
        """Retain the source and build one bounded fit image per frame."""
        from core.source_frames import ProcessedFrame, SourceFrame, _small
        from utils.frame_pixels import run_pixel_worker

        self._closed = False
        self._generation += 1
        generation = self._generation
        self._cancel_native()
        shape = None if frame is None else tuple(frame.shape)
        dtype = None if frame is None else frame.dtype
        same = (frame is self._source and self._source_key is not None
                and self._source_key[:2] == (shape, dtype)
                and self._same_metadata(self._source_key[2], linear_dng_metadata))
        self._source = frame
        self._source_shape = shape
        self._labels = labels
        self._target = target
        self._adder.config.update_config(config)
        self._index = max(0, int(index))
        if not same:
            self._frame = None
            self._source_key = None
            self._rendered = QPixmap()
            self.release_developed_cache()
            try:
                if frame is None:
                    source = None
                elif isinstance(frame, (SourceFrame, ProcessedFrame)):
                    source = display_image(frame, self)
                else:
                    source = run_pixel_worker(self, "OpenFocus", lambda: _small(frame))
            except Exception as exc:
                if generation == self._generation and not self._closed:
                    self._status = "error"
                    self._native_error = str(exc)
                    self.clear()
                return
            if generation != self._generation or self._closed:
                return
            self._frame = source
            self._source_key = (shape, dtype, linear_dng_metadata)
        self._linear_dng_metadata = (None if getattr(frame, "preview_is_display", False)
                                     else linear_dng_metadata)
        self.refresh()

    def release_developed_cache(self):
        """Drop the one display-depth frame cache immediately.

        Large camera frames are held only while this dialog shows them; the
        raw stack remains owned by the main window.
        """
        self._developed_source = None
        self._developed_image = None
        self._developed_metadata = None
        self._developed_shape = None
        self._developed_dtype = None

    @staticmethod
    def _same_metadata(first, second):
        if first is second:
            return True
        try:
            return bool(first == second)
        except Exception:
            return False

    def _developed_frame(self, metadata):
        """Display-depth pixels for the current frame, reused across edits."""
        frame = self._frame
        if (self._developed_image is not None
                and self._developed_source is frame
                and self._developed_shape == frame.shape
                and self._developed_dtype == frame.dtype
                and self._same_metadata(self._developed_metadata, metadata)):
            return self._developed_image
        # Release the previous frame before allocating its replacement so two
        # full-resolution display buffers never coexist in this widget.
        self.release_developed_cache()
        from utils.linear_dng import develop_linear
        developed = develop_linear(frame, metadata)
        if developed.nbytes <= MAX_DEVELOPED_CACHE_BYTES:
            self._developed_source = frame
            self._developed_image = developed
            self._developed_metadata = metadata
            self._developed_shape = frame.shape
            self._developed_dtype = frame.dtype
        return developed

    def set_zoom_mode(self, mode):
        """At 100%, one source pixel occupies one device pixel."""
        self._zoom_mode = mode
        self._pan = (0, 0)
        self._show_rendered()

    def _cancel_native(self):
        self._native_batch += 1
        self._native_loading = False
        self._native_pending = []
        self._native_request = None
        self._native_pixmap = QPixmap()
        worker, self._region_worker = self._region_worker, None
        if worker is not None:
            worker.request_cancel()

    def close_preview(self):
        """Cancel outstanding work without waiting for the decoder on Qt."""
        self._closed = True
        self._generation += 1
        self._cancel_native()
        self._source = self._frame = None
        self._source_key = self._source_shape = None
        self._rendered = QPixmap()
        self.release_developed_cache()

    def _show_rendered(self):
        self._cancel_native()
        if self._rendered.isNull():
            self.clear()
            return
        width, height = max(1, self.width() - 8), max(1, self.height() - 8)
        if self._zoom_mode != "actual" or self._source is None:
            self._native_error = ""
            self.status_changed.emit()
            self.setPixmap(self._rendered.scaled(width, height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            return
        from core.resource_policy import require_memory
        from core.source_previews import DISPLAY_REGION_MAX_SIDE, SourceRegionWorker

        dpr = self.devicePixelRatioF()
        source_h, source_w = self._source_shape[:2]
        view_w = min(source_w, max(1, math.ceil(width * dpr)))
        view_h = min(source_h, max(1, math.ceil(height * dpr)))
        x = max(0, min(self._pan[0], max(0, source_w - view_w)))
        y = max(0, min(self._pan[1], max(0, source_h - view_h)))
        self._pan = (x, y)
        try:
            require_memory(view_w * view_h * 12 + 1024 * 1024)
            pixmap = QPixmap(view_w, view_h)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.drawPixmap(QRectF(0, 0, view_w, view_h), self._rendered,
                QRectF(x * self._rendered.width() / source_w,
                       y * self._rendered.height() / source_h,
                       view_w * self._rendered.width() / source_w,
                       view_h * self._rendered.height() / source_h))
            painter.end()
            self._native_pixmap = pixmap
            self._native_error = ""
            self._native_origin = (x, y)
            self._native_dpr = dpr
            edge = DISPLAY_REGION_MAX_SIDE - 2
            self._native_pending = [
                (x + col, y + row, min(edge, view_w - col), min(edge, view_h - row))
                for row in range(0, view_h, edge) for col in range(0, view_w, edge)]
            self._native_loading = True
            key = (self._generation, self._native_batch)
            labels = self._labels if self._labels is not None else _DraftLabels(self._adder.config)
            worker = SourceRegionWorker(self._source, key, labels=labels,
                                        target=self._target, index=self._index,
                                        metadata=self._linear_dng_metadata)
            self._region_worker = worker
            _LIVE_REGION_WORKERS.add(worker)
            worker.finished.connect(lambda worker=worker: _LIVE_REGION_WORKERS.discard(worker))
            worker.finished.connect(worker.deleteLater)
            worker.ready.connect(self._native_ready)
            worker.failed.connect(self._native_failed)
            self.destroyed.connect(worker.request_cancel)
            self._publish_native()
            worker.start()
            self._submit_native()
            self.status_changed.emit()
        except Exception as exc:
            self._native_failed((self._generation, self._native_batch), str(exc))

    def _publish_native(self):
        pixmap = self._native_pixmap.copy()
        pixmap.setDevicePixelRatio(self._native_dpr)
        self.setPixmap(pixmap)

    def _submit_native(self):
        if not self._native_pending or self._region_worker is None:
            self._native_request = None
            if self._native_loading:
                self._native_loading = False
                self.status_changed.emit()
            return
        rect = self._native_pending.pop(0)
        self._native_serial += 1
        request = ("viewport", self._native_serial, rect, rect[2:])
        self._native_request = request
        self._region_worker.submit(request)

    def _native_ready(self, key, request, tile, sampled):
        if (self._closed or key != (self._generation, self._native_batch)
                or request != self._native_request):
            return
        if tile is None:
            self._native_failed(key, "Native preview returned no pixels")
            return
        from core.resource_policy import require_memory
        try:
            require_memory(tile.shape[0] * tile.shape[1] * 16
                           + self._native_pixmap.width() * self._native_pixmap.height() * 8
                           + 1024 * 1024)
        except Exception as exc:
            self._native_failed(key, str(exc))
            return
        from utils.image_utils import to_display_uint8
        pixels = to_display_uint8(tile)
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[:, :, None], 3, axis=2)
        rgb = np.ascontiguousarray(pixels[:, :, :3][:, :, ::-1])
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        painter = QPainter(self._native_pixmap)
        rect = request[2]
        painter.setClipRect(QRectF(rect[0] - self._native_origin[0],
                                   rect[1] - self._native_origin[1], rect[2], rect[3]))
        painter.drawImage(QPointF(sampled[0] - self._native_origin[0],
                                  sampled[1] - self._native_origin[1]), image)
        painter.end()
        self._publish_native()
        self._submit_native()

    def _native_failed(self, key, error):
        if self._closed or key != (self._generation, self._native_batch):
            return
        self._cancel_native()
        self._native_error = error or "Native preview failed"
        width, height = max(1, self.width() - 8), max(1, self.height() - 8)
        self.setPixmap(self._rendered.scaled(width, height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self.status_changed.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._zoom_mode == "actual":
            self._drag_start = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            delta = event.position() - self._drag_start
            dpr = self.devicePixelRatioF()
            self._pan = (self._pan[0] - delta.x() * dpr,
                         self._pan[1] - delta.y() * dpr)
            self._drag_start = event.position()
            self._show_rendered()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------ data
    def set_frame(self, frame) -> None:
        """Use a real frame as the preview canvas (falls back to a placeholder)."""
        self.set_preview(frame, vars(self._adder.config), self._index)

    def set_index(self, index: int) -> None:
        self._index = max(0, int(index))
        self.refresh()

    def set_config(self, config: dict) -> None:
        self._adder.config.update_config(config)
        self.refresh()

    def message(self) -> str:
        """Why the preview shows no label, for the dialog's status line."""
        return self._status

    def draw_error(self) -> str:
        """Non-empty when the last draw attempt failed outright."""
        return self._adder.last_error

    def native_status(self):
        """Describe asynchronous native viewport work for the dialog."""
        if self._native_loading:
            return "Loading 100% preview…"
        return self._native_error

    # ------------------------------------------------------------------ draw
    def _preview_value(self) -> int:
        """The number this preview's frame would carry.

        The steppers hand back floats, so both terms are rounded: without it
        "{value}" rendered as "1.0" in the preview while the real export drew
        "1" (str() of the substituted value).
        """
        start = int(round(float(self._adder.config.starting_value)))
        step = int(round(float(self._adder.config.interval)))
        return start + self._index * step

    def _canvas(self):
        if self._frame is not None and self._frame.size:
            frame = self._frame
            # Fit rendering uses the bounded image; native uses the source.
            return frame
        # Placeholder: mid grey with a faint grid, so white labels are legible.
        canvas = np.full((360, 540, 3), 70, np.uint8)
        for x in range(0, 540, 60):
            canvas[:, x] = 88
        for y in range(0, 360, 60):
            canvas[y, :] = 88
        return canvas

    def refresh(self) -> None:
        from utils.image_utils import to_display_uint8

        self._native_error = ""
        canvas = self._canvas()
        metadata = self._linear_dng_metadata
        if (metadata is not None and self._frame is not None
                and canvas.dtype == np.uint16 and canvas.ndim == 3
                and canvas.shape[2] == 3):
            # Draw in display space, never by scaling the linear pixels through
            # to_display_uint8 (which turns native RAW midtones nearly black).
            canvas = self._developed_frame(metadata)
        else:
            self.release_developed_cache()
        if canvas.ndim == 3 and canvas.shape[2] == 4:
            from utils.image_utils import composite_transparent_for_display
            canvas = composite_transparent_for_display(canvas)
        config = self._adder.config
        text = build_label_text(config.format, config.text,
                                self._preview_value())
        # Draw first: the status must describe *this* attempt, not the
        # previous one (out-of-range and draw failures are per-frame).
        try:
            original = self._adder.config
            if self._frame is not None:
                self._adder.config = preview_label_config(original, self._source_shape, canvas.shape)
            try:
                drawn = self._adder.add_label_to_image(canvas, self._index)
            finally:
                self._adder.config = original
            self.substituted_font = self._adder.last_substituted_font
            error = self._adder.last_error
            out_of_range = self._adder.last_out_of_range
        except Exception:
            drawn = canvas
            self.substituted_font = ""
            error = "draw_failed"
            out_of_range = False

        if error:
            self._status = "error"
        elif not text:
            self._status = "empty"
        elif out_of_range:
            self._status = "out_of_range"
        else:
            self._status = ""

        drawn = to_display_uint8(drawn)
        if drawn.ndim == 2:
            drawn = np.repeat(drawn[:, :, None], 3, axis=2)
        h, w = drawn.shape[:2]
        rgb = np.ascontiguousarray(drawn[:, :, ::-1])
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._rendered = QPixmap.fromImage(image)
        bounds = self._adder.last_bounds
        if bounds is not None:
            painter = QPainter(self._rendered)
            pen = QPen(QColor("#40c8d5"))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(max(1, min(w, h) // 300))
            painter.setPen(pen)
            x, y, text_width, text_height = bounds
            painter.drawRect(x, y, text_width, text_height)
            painter.drawEllipse(x - 3, y + text_height - 3, 6, 6)
            painter.end()
        self._show_rendered()

    def event(self, event):
        result = super().event(event)
        # A window can move to a screen with a different pixel ratio without
        # changing its logical size. Rebuild the native viewport for the new
        # device grid and invalidate any tiles from the previous screen.
        ratio_events = (getattr(QEvent.Type, 'DevicePixelRatioChange', None),
                        getattr(QEvent.Type, 'ScreenChangeInternal', None))
        if (hasattr(self, '_zoom_mode') and self._zoom_mode == 'actual'
                and event.type() in ratio_events and not self._closed):
            self._show_rendered()
        return result

    def resizeEvent(self, event):  # noqa: N802 (Qt casing)
        super().resizeEvent(event)
        self._show_rendered()
