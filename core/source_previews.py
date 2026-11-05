"""Incremental bounded previews; never decode original pixels on the GUI thread."""
import threading

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from core.source_frames import ensure_preview
from utils.image_utils import to_display_uint8
from utils.logging_setup import log_operation

_PREVIEW_DECODE = threading.Semaphore(1)


class SourcePreviewWorker(QThread):
    ready = pyqtSignal(object, int, object, object)  # request, index, identity, thumbnail
    failed = pyqtSignal(object, str)

    def __init__(self, frames, request, indices=None):
        super().__init__()
        self.frames = tuple(frames)
        self.request = request
        self.indices = tuple(range(len(self.frames))) if indices is None else tuple(indices)
        self.cancelled = False

    def request_cancel(self):
        self.cancelled = True
        self.requestInterruption()

    cancel = request_cancel

    def run(self):
        with log_operation('source_previews', frames=len(self.indices)) as operation:
            for index in self.indices:
                if self.cancelled:
                    operation.cancel()
                    return
                while not _PREVIEW_DECODE.acquire(timeout=.05):
                    if self.cancelled:
                        operation.cancel()
                        return
                try:
                    if self.cancelled:
                        operation.cancel()
                        return
                    frame = self.frames[index]
                    image = ensure_preview(frame)
                    h, w = image.shape[:2]
                    factor = min(1, 40 / max(h, w))
                    thumb = cv2.resize(image, (max(1, round(w * factor)), max(1, round(h * factor))),
                                       interpolation=cv2.INTER_AREA)
                    if getattr(frame, 'linear_metadata', None) is not None and not frame.preview_is_display:
                        from utils.linear_dng import develop_linear
                        thumb = develop_linear(thumb, frame.linear_metadata)
                    thumb = to_display_uint8(thumb)
                    if not self.cancelled:
                        self.ready.emit(self.request, index, id(frame), thumb)
                except Exception as exc:
                    operation.fail(exc)
                    self.failed.emit(self.request, str(exc))
                finally:
                    _PREVIEW_DECODE.release()
