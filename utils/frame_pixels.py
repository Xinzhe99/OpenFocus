"""Bounded preview geometry and background pixel work for lazy consumers."""
from copy import copy

import numpy as np

from PyQt6.QtCore import QEventLoop, QThread, QTimer, Qt, pyqtSlot
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout, QWidget

from utils.platform_utils import configure_native_worker_stack


def preview_label_config(config, source_shape, preview_shape):
    """Map absolute working-image label coordinates onto a preview."""
    sx = preview_shape[1] / source_shape[1]
    sy = preview_shape[0] / source_shape[0]
    result = copy(config)
    for name, factor in (("x_location", sx), ("y_location", sy),
                         ("font_size", min(sx, sy))):
        if isinstance(result, dict):
            result[name] = float(result.get(name, 0)) * factor
        else:
            setattr(result, name, float(getattr(result, name)) * factor)
    return result


def preview_scale_config(config, source_shape, preview_shape):
    if not config or source_shape[:2] == preview_shape[:2]:
        return config
    result = dict(config)
    scale = preview_shape[1] / source_shape[1]
    result["px_um"] /= scale
    result["font_size"] = result.get("font_size", 0) * scale
    return result


PREVIEW_MAX_SIDE = 1200


class BoundedPreviewWorker(QThread):
    """Develop available fit pixels in the bounded preview lane."""

    def __init__(self, frame, metadata, key, *, preserve_alpha=False, edge=PREVIEW_MAX_SIDE):
        super().__init__()
        configure_native_worker_stack(self)
        self.frame, self.metadata, self.key = frame, metadata, key
        self.preserve_alpha = preserve_alpha
        self.edge = edge
        self.pixels = None
        self.error = None

    @pyqtSlot()
    def request_cancel(self):
        self.requestInterruption()

    def _available(self, cancelled):
        from core.aligned_previews import registered_preview
        from core.resource_policy import require_memory
        from core.source_frames import (ProcessedFrame, RetainedFrame, SourceFrame,
                                        ensure_preview, resolve_frame, validate_frame_source)
        from core.source_previews import _PREVIEW_WORKING_BYTES

        frame = self.frame
        if isinstance(frame, np.ndarray):
            return frame, False
        if isinstance(frame, SourceFrame) and (frame.preview is None or not frame.preview_is_available):
            pixels = frame.cached_pixels()
            return (pixels, False) if pixels is not None else None
        require_memory(_PREVIEW_WORKING_BYTES)
        if isinstance(frame, SourceFrame):
            return ensure_preview(frame, should_cancel=cancelled), frame.preview_is_display
        if isinstance(frame, ProcessedFrame):
            validate_frame_source(frame)
            registered = registered_preview(frame, cancelled, captured_only=True, edge=self.edge)
            if registered is not None:
                return registered
        if isinstance(frame, RetainedFrame):
            return resolve_frame(frame, should_cancel=cancelled), False
        return None

    def _render(self, pixels, is_display):
        from core.source_previews import render_source_region

        if self.isInterruptionRequested():
            return
        height, width = pixels.shape[:2]
        scale = min(1.0, self.edge / max(width, height))
        metadata = (self.metadata if not is_display
                    and pixels.dtype == np.uint16 and pixels.ndim == 3
                    and pixels.shape[2] == 3 else None)
        self.pixels, _ = render_source_region(
            pixels, (0, 0, width, height),
            (max(1, round(width * scale) - 2), max(1, round(height * scale) - 2)),
            metadata, preserve_alpha=self.preserve_alpha)

    def run(self):
        from core.aligned_previews import registered_preview
        from core.resource_policy import ResourceLimitError, heavy_job
        from core.source_frames import ProcessedFrame, ensure_display_preview
        from core.source_previews import _acquire_preview_turn, _release_preview_turn

        cancelled = self.isInterruptionRequested
        try:
            if not _acquire_preview_turn(cancelled, True):
                return
            try:
                available = self._available(cancelled)
                if available is not None:
                    self._render(*available)
                    return
            except ResourceLimitError:
                # Try again after the current heavy job frees its memory.
                available = None
            finally:
                _release_preview_turn()

            with heavy_job(cancelled):
                if not _acquire_preview_turn(cancelled, True):
                    return
                try:
                    registered = (registered_preview(self.frame, cancelled, edge=self.edge)
                                  if isinstance(self.frame, ProcessedFrame) else None)
                    if registered is None:
                        pixels = ensure_display_preview(self.frame, should_cancel=cancelled)
                        registered = pixels, getattr(self.frame, 'preview_is_display', False)
                    self._render(*registered)
                finally:
                    _release_preview_turn()
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.error = str(exc)
        finally:
            self.frame = self.metadata = None


class _PixelWorker(QThread):
    def __init__(self, operation):
        super().__init__()
        configure_native_worker_stack(self)
        self.operation = operation
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.operation()
        except BaseException as exc:
            self.error = exc
        finally:
            self.operation = None


class _PixelProgress(QDialog):
    def reject(self):
        # The caller owns its transaction until the worker completes.
        pass

    def closeEvent(self, event):
        event.ignore()


def run_pixel_worker(parent, title, operation, *, show_progress=True):
    """Keep Qt responsive while a synchronous caller awaits pixel I/O.

    Operations never access Qt widgets. The default modal dialog keeps source
    state stable. Windowless callers must guard reentry and reject stale results
    because their own widgets remain interactive during the nested event loop.
    """
    application = QApplication.instance()
    if application is None or QThread.currentThread() is not application.thread():
        return operation()
    dialog = None
    if show_progress:
        dialog = _PixelProgress(parent if isinstance(parent, QWidget) else None)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(title))
        waiter = dialog
    else:
        waiter = QEventLoop()
    worker = _PixelWorker(operation)
    worker.finished.connect(dialog.accept if dialog is not None else waiter.quit)
    QTimer.singleShot(0, worker.start)
    waiter.exec()
    worker.wait()
    result, error = worker.result, worker.error
    worker.deleteLater()
    waiter.deleteLater()
    if error is not None:
        raise error
    return result


def preview_pixels(frame, parent=None):
    """Return bounded display pixels, generating missing previews off Qt."""
    from core.source_frames import SourceFrame, ProcessedFrame, display_image
    if isinstance(frame, ProcessedFrame) or (isinstance(frame, SourceFrame) and frame.preview is None):
        return run_pixel_worker(parent, "OpenFocus", lambda: display_image(frame))
    return display_image(frame)
