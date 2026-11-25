"""Bounded preview geometry and background pixel work for lazy consumers."""
import threading
from copy import copy

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import QEvent, QEventLoop, QObject, QThread, QTimer, Qt, pyqtSlot
from PyQt6.QtGui import QShortcut
from PyQt6.QtWidgets import QApplication, QProgressBar, QProgressDialog, QPushButton, QWidget

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


PIXEL_PROGRESS_DELAY_MS = 250
_INPUT_EVENTS = frozenset((
    QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
    QEvent.Type.MouseButtonDblClick, QEvent.Type.Wheel,
    QEvent.Type.KeyPress, QEvent.Type.KeyRelease, QEvent.Type.Shortcut,
    QEvent.Type.ShortcutOverride, QEvent.Type.ContextMenu, QEvent.Type.Drop,
))


class PixelProgress:
    """Thread-safe progress handle shared by a pixel operation and its dialog.

    ``total`` of zero keeps the dialog indeterminate. Operations may poll
    ``cancelled`` between frames when the caller offered a Cancel button.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._done = 0
        self._total = 0
        self._cancel = threading.Event()

    def update(self, done, total=None):
        with self._lock:
            self._done = int(done)
            if total is not None:
                self._total = int(total)

    def snapshot(self):
        with self._lock:
            return self._done, self._total

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self):
        return self._cancel.is_set()


class _InputBlocker(QObject):
    """Swallow user input during the short window before the dialog appears."""

    def eventFilter(self, watched, event):
        return event.type() in _INPUT_EVENTS


class _PixelProgress(QProgressDialog):
    """Styled, indeterminate-by-default progress dialog for pixel workers."""

    def __init__(self, parent, title, *, cancellable=False):
        super().__init__(parent)
        from locales import trans
        from ui import styles as ui_styles
        self._base_title = title
        self._cancellable = bool(cancellable)
        self._cancelling = False
        self._finished = False
        self._cancel_btn = None
        self.setStyleSheet(ui_styles.PROGRESS_DIALOG_STYLE)
        self.setWindowTitle(title)
        self.setLabelText(title)
        self._bar = QProgressBar(self)
        self.setBar(self._bar)
        self.setRange(0, 0)
        self.setMinimumWidth(320)
        self.setMinimumDuration(2 ** 31 - 1)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self.setCancelButton(None)
        if self._cancellable:
            self._cancel_btn = QPushButton(trans.t("btn_cancel"))
            self.setCancelButton(self._cancel_btn)
            # Qt's default canceled() slot hides and resets the dialog even
            # with auto-close disabled. Keep ownership until the worker exits.
            self.canceled.disconnect()
            self.canceled.connect(self.mark_cancelling)
        self._cancel_shortcut = self.findChild(QShortcut)

    def mark_cancelling(self):
        if self._cancelling:
            return
        self._cancelling = True
        from locales import trans
        self.setLabelText(trans.t("depth_cancelling"))
        if self._cancel_btn is not None:
            self._cancel_btn.setEnabled(False)
        if self._cancel_shortcut is not None:
            self._cancel_shortcut.setEnabled(False)

    def finish(self):
        self._finished = True
        self.hide()

    def reject(self):
        # The caller owns its transaction until the worker completes.
        if self._cancellable and not self._cancelling and not self._finished:
            self.canceled.emit()

    def closeEvent(self, event):
        event.ignore() if not self._finished else event.accept()

    def set_progress(self, done, total, message=None):
        if total > 0:
            total = max(done, total)
            if self.maximum() != total:
                self.setRange(0, total)
            self._bar.setValue(max(0, done))
        elif self.maximum() != 0:
            self.setRange(0, 0)
        if not self._cancelling:
            if message is None:
                message = f"{self._base_title} ({done}/{total})" if total > 0 else self._base_title
            self.setLabelText(message)


def run_pixel_worker(parent, title, operation, *, show_progress=True,
                     progress=None, cancellable=False,
                     delay_ms=PIXEL_PROGRESS_DELAY_MS):
    """Keep Qt responsive while a synchronous caller awaits pixel I/O.

    Operations never access Qt widgets. The progress dialog keeps source state
    stable but appears only after ``delay_ms`` so quick saves never flash; user
    input is withheld until then. Pass a ``PixelProgress`` to show determinate
    progress, and ``cancellable=True`` to offer Cancel (the operation must poll
    ``progress.cancelled``). Windowless callers must guard reentry and reject
    stale results because their own widgets remain interactive during the
    nested event loop.
    """
    application = QApplication.instance()
    if application is None or QThread.currentThread() is not application.thread():
        return operation()
    if cancellable and progress is None:
        progress = PixelProgress()
    dialog = None
    blocker = None
    reveal_timer = None
    poll = None
    completed = False
    waiter = QEventLoop()
    worker = _PixelWorker(operation)
    if show_progress:
        dialog = _PixelProgress(parent if isinstance(parent, QWidget) else None,
                                title, cancellable=cancellable)
        if cancellable:
            dialog.canceled.connect(progress.cancel)
        blocker = _InputBlocker()
        application.installEventFilter(blocker)

        def reveal():
            if completed:
                return
            if not worker.isFinished() and not sip.isdeleted(dialog):
                dialog.show()
            application.removeEventFilter(blocker)

        reveal_timer = QTimer()
        reveal_timer.setSingleShot(True)
        reveal_timer.setInterval(max(0, int(delay_ms)))
        reveal_timer.timeout.connect(reveal)
        reveal_timer.start()
        if progress is not None:
            poll = QTimer()
            poll.setInterval(100)
            def update_dialog():
                if not sip.isdeleted(dialog):
                    dialog.set_progress(*progress.snapshot())

            poll.timeout.connect(update_dialog)
            poll.start()
    worker.finished.connect(waiter.quit)
    worker.start()
    waiter.exec()
    completed = True
    worker.wait()
    if reveal_timer is not None:
        reveal_timer.stop()
        reveal_timer.deleteLater()
    if poll is not None:
        poll.stop()
        poll.deleteLater()
    if blocker is not None:
        application.removeEventFilter(blocker)
    if dialog is not None and not sip.isdeleted(dialog):
        dialog.finish()
        dialog.deleteLater()
    result, error = worker.result, worker.error
    worker.deleteLater()
    waiter.deleteLater()
    if blocker is not None:
        blocker.deleteLater()
    if error is not None:
        raise error
    return result


def invoke_pixel_worker(worker_fn, parent, title, operation, *, progress=None,
                        cancellable=False, show_progress=True):
    """Call ``worker_fn`` passing progress kwargs only when supported."""
    if worker_fn is run_pixel_worker:
        return worker_fn(parent, title, operation, show_progress=show_progress,
                         progress=progress, cancellable=cancellable)
    return worker_fn(parent, title, operation)

def preview_pixels(frame, parent=None):
    """Return bounded display pixels, generating missing previews off Qt."""
    from core.source_frames import SourceFrame, ProcessedFrame, display_image
    if isinstance(frame, ProcessedFrame) or (isinstance(frame, SourceFrame) and frame.preview is None):
        return run_pixel_worker(parent, "OpenFocus", lambda: display_image(frame))
    return display_image(frame)
