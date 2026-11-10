"""Bounded preview geometry and background pixel work for lazy consumers."""
from copy import copy

from PyQt6.QtCore import QEventLoop, QThread, QTimer, Qt
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
