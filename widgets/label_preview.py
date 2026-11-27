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
from PyQt6.QtWidgets import QLabel, QSizePolicy, QStyle

from locales import trans
from utils.validators import LabelAdder, build_label_text
from utils.frame_pixels import BoundedPreviewWorker, preview_label_config

# One display-depth copy of the current frame is enough for form edits. The
# source stack already lives in the window; this cache must never grow with it.
MAX_DEVELOPED_CACHE_BYTES = 512 * 1024 * 1024

# Retain cancelled threads until Qt reports their completion.
_LIVE_PREVIEW_WORKERS = set()


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
    fit_changed = pyqtSignal()
    position_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
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
        self._placement_drag = None
        self._source = None
        self._source_shape = None
        self._source_key = None
        self._generation = 0
        self._fit_worker = None
        self._fit_loading = False
        self._fit_error = ""
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
        from core.source_frames import PREVIEW_EDGE, SourceFrame

        self._closed = False
        self._generation += 1
        self._cancel_native()
        shape = None if frame is None else tuple(frame.shape)
        dtype = None if frame is None else frame.dtype
        same = (frame is self._source and self._source_key is not None
                and self._source_key[:2] == (shape, dtype)
                and self._same_metadata(self._source_key[2], linear_dng_metadata))
        if not same or self._index != index or self._target != target:
            self.cancel_drag()
        self._source = frame
        self._source_shape = shape
        self._labels = labels
        self._target = target
        self._adder.config.update_config(config)
        self._index = max(0, int(index))
        self._linear_dng_metadata = linear_dng_metadata
        if not same:
            self._frame = None
            self._source_key = (shape, dtype, linear_dng_metadata)
            self._fit_error = ""
            self._rendered = QPixmap()
            self.release_developed_cache()
            if self._fit_worker is not None:
                self._fit_worker.requestInterruption()
            # Small in-memory images have no pixel I/O or unbounded resize.
            pixels = frame
            if isinstance(frame, SourceFrame):
                pixels = frame.preview if frame.preview_is_available else None
            if isinstance(pixels, np.ndarray) and pixels.size and max(pixels.shape[:2]) <= PREVIEW_EDGE:
                self._frame = pixels
            self._fit_loading = frame is not None and self._frame is None
            if self._fit_loading:
                self._start_fit()
        self.refresh()

    def _start_fit(self):
        from core.source_frames import PREVIEW_EDGE

        if self._fit_worker is not None:
            return
        worker = BoundedPreviewWorker(self._source, self._linear_dng_metadata,
                                      self._source_key, edge=PREVIEW_EDGE)
        self._fit_worker = worker
        _LIVE_PREVIEW_WORKERS.add(worker)
        worker.finished.connect(lambda worker=worker: _LIVE_PREVIEW_WORKERS.discard(worker))
        worker.finished.connect(self._fit_finished)
        worker.finished.connect(worker.deleteLater)
        self.destroyed.connect(worker.request_cancel)
        worker.start()

    def _fit_finished(self):
        worker, self._fit_worker = self._fit_worker, None
        if self._closed or not self._fit_loading:
            return
        if worker.key is not self._source_key:
            self._start_fit()
            return
        self._fit_loading = False
        self._fit_error = worker.error or ""
        self._frame = worker.pixels
        self.refresh()
        self.fit_changed.emit()

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
        self.cancel_drag()
        self._zoom_mode = mode
        self._pan = (0, 0)
        self._show_rendered()
        self._update_cursor()

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
        self.cancel_drag()
        self._generation += 1
        self._cancel_native()
        if self._fit_worker is not None:
            self._fit_worker.requestInterruption()
        self._fit_loading = False
        self._fit_error = ""
        self._source = self._frame = None
        self._source_key = self._source_shape = None
        self._rendered = QPixmap()
        self.release_developed_cache()

    def _show_rendered(self):
        self._cancel_native()
        if self._rendered.isNull():
            if self._fit_loading or self._fit_error:
                self.setText(self._fit_error or trans.t('native_detail_loading'))
            else:
                self.clear()
            self.status_changed.emit()
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
            # The fit preview can already be developed. The native worker
            # still uses the original frame's camera calibration.
            worker = SourceRegionWorker(self._source, key, labels=labels,
                                        target=self._target, index=self._index,
                                        metadata=self._linear_dng_metadata)
            self._region_worker = worker
            _LIVE_PREVIEW_WORKERS.add(worker)
            worker.finished.connect(lambda worker=worker: _LIVE_PREVIEW_WORKERS.discard(worker))
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

    def placement_available(self):
        return (self._zoom_mode == 'fit' and self._frame is not None
                and not self._status and self._adder.last_bounds is not None)

    def _fit_geometry(self, position):
        """Find the displayed pixmap and the label hit in logical pixels."""
        if not self.placement_available():
            return None
        pixmap = self.pixmap()
        if pixmap.isNull():
            return None
        image_rect = QRectF(QStyle.alignedRect(
            self.layoutDirection(), self.alignment(),
            pixmap.deviceIndependentSize().toSize(), self.contentsRect()))
        if not image_rect.contains(position):
            return None

        sx = image_rect.width() / self._rendered.width()
        sy = image_rect.height() / self._rendered.height()
        x, y, width, height = self._adder.last_bounds
        hit = QRectF(image_rect.x() + x * sx, image_rect.y() + y * sy,
                     width * sx, height * sy).adjusted(-5, -5, 5, 5)
        return image_rect if hit.contains(position) else None

    def _update_cursor(self, position=None):
        if self._placement_drag is not None or (position is not None
                and self._fit_geometry(position) is not None):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif self._zoom_mode == 'actual' and self._frame is not None:
            self.setCursor(Qt.CursorShape.ClosedHandCursor if self._drag_start is not None
                           else Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    def cancel_drag(self):
        self._drag_start = self._placement_drag = None
        self.unsetCursor()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            position = event.position()
            if self._zoom_mode == 'actual' and self._frame is not None:
                self._drag_start = position
            elif (image_rect := self._fit_geometry(position)) is not None:
                config = self._adder.config
                self._placement_drag = (position, image_rect, config.x_location, config.y_location)
            if self._drag_start is not None or self._placement_drag is not None:
                self._update_cursor(position)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        position = event.position()
        if self._placement_drag is not None:
            start, image_rect, x, y = self._placement_drag
            source_h, source_w = self._source_shape[:2]
            # Keep the point grabbed on the image, including when the pointer
            # leaves the canvas. Preserve the offset from the label anchor.
            px = max(image_rect.left(), min(position.x(), image_rect.right() - 1))
            py = max(image_rect.top(), min(position.y(), image_rect.bottom() - 1))
            x = round(x + (px - start.x()) * source_w / image_rect.width())
            y = round(y + (py - start.y()) * source_h / image_rect.height())
            config = self._adder.config
            if (x, y) != (config.x_location, config.y_location):
                self.position_changed.emit(x, y)
            event.accept()
            return
        if self._drag_start is not None:
            delta = position - self._drag_start
            dpr = self.devicePixelRatioF()
            self._pan = (self._pan[0] - delta.x() * dpr,
                         self._pan[1] - delta.y() * dpr)
            self._drag_start = position
            self._show_rendered()
            event.accept()
            return
        self._update_cursor(position)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.cancel_drag()
            self._update_cursor(event.position())
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
        return self._adder.last_error if not (self._fit_loading or self._fit_error) else ""

    def native_status(self):
        """Describe asynchronous native viewport work for the dialog."""
        if self._native_loading:
            return trans.t('label_preview_native_loading')
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

        if self._fit_loading or self._fit_error:
            self._status = "error" if self._fit_error else ""
            self.substituted_font = ""
            self._show_rendered()
            return

        self._native_error = ""
        canvas = self._canvas()
        metadata = self._linear_dng_metadata
        if getattr(self._source, 'preview_is_display', False):
            metadata = None
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
        exif_meta = None
        if self._labels is not None and hasattr(self._labels, "_exif_meta_for"):
            try:
                exif_meta = self._labels._exif_meta_for(
                    self._target, self._index,
                    getattr(self._labels, "_snapshot_output_id", None),
                )
            except Exception:
                exif_meta = None
        text = build_label_text(config.format, config.text,
                                self._preview_value(), exif_meta=exif_meta)
        # Draw first: the status must describe *this* attempt, not the
        # previous one (out-of-range and draw failures are per-frame).
        try:
            original = self._adder.config
            if self._frame is not None:
                self._adder.config = preview_label_config(original, self._source_shape, canvas.shape)
            try:
                self._adder.exif_meta = exif_meta
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
