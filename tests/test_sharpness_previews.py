"""Regression coverage for sharpness after asynchronous source previews."""
import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap

from controllers.transform_manager import TransformManager
from core.source_frames import SourceFrame
from tests.test_stack_review_gui import _window
from widgets.sharpness_curve import SharpnessCurveWidget


def test_curve_tracks_completed_previews_and_ignores_replaced_stack(qapp, monkeypatch):
    import core.source_previews
    from main import OpenFocus

    workers = []

    class Worker(QObject):
        ready = pyqtSignal(object, int, object, object)
        failed = pyqtSignal(object, str)
        finished = pyqtSignal()

        def __init__(self, frames, request, indices, **kwargs):
            super().__init__()
            self.frames = tuple(frames)
            self.request = request
            self.indices = tuple(indices)
            self.cancelled = False
            workers.append(self)

        def start(self):
            pass

        def request_cancel(self):
            self.cancelled = True

    monkeypatch.setattr(core.source_previews, "SourcePreviewWorker", Worker)
    window, manager, panel = _window()
    window.sharpness_curve = SharpnessCurveWidget()
    window.current_display_index = -1
    window.update_source_view = lambda index: None
    window._start_source_previews = lambda selected_index=None: OpenFocus._start_source_previews(
        window, selected_index)

    def install_stack(revision):
        window.source_revision = revision
        window.raw_images = [SourceFrame(str(i), 0, (16, 16, 3), np.dtype('uint8'), None, ())
                             for i in range(6)]
        window.stack_images = [None] * 6
        manager.update_file_list([str(i) for i in range(6)], [QPixmap() for _ in range(6)])
        TransformManager._update_sharpness_curve(window)
        assert window.sharpness_curve.isHidden()
        window._start_source_previews()

    def complete(worker, count):
        for index in range(count):
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[::2, ::2] = 20 * (index + 1)
            object.__setattr__(worker.frames[index], 'preview', image)
            worker.ready.emit(worker.request, index, id(worker.frames[index]), image)

    try:
        install_stack(1)
        old = workers[-1]
        complete(old, 3)
        assert window.sharpness_curve.isHidden()

        install_stack(2)
        assert old.cancelled
        complete(old, 6)
        old.finished.emit()
        assert window.sharpness_curve.isHidden()
        current = workers[-1]
        assert current is not old
        complete(current, 5)
        current.finished.emit()
        assert window.sharpness_curve.isHidden()

        # Retry the missing frame, as happens after an interrupted preview batch.
        window.file_list.item(5).setIcon(QIcon())
        window._start_source_previews()
        retry = workers[-1]
        image = np.full((16, 16, 3), 50, dtype=np.uint8)
        object.__setattr__(retry.frames[5], 'preview', image)
        retry.ready.emit(retry.request, 5, id(retry.frames[5]), image)
        window.current_display_index = 4
        retry.finished.emit()
        assert not window.sharpness_curve.isHidden()
        assert len(window.sharpness_curve._values) == 6
        assert window.sharpness_curve._values[0] > 0
        assert window.sharpness_curve._current == 4

        window.raw_images = []
        TransformManager._update_sharpness_curve(window)
        assert window.sharpness_curve.isHidden()
    finally:
        window.sharpness_curve.close()
        panel.widget.close()
