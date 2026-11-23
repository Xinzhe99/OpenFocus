"""Regression coverage for the sharpness curve and source previews."""
import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import (QMimeData, QModelIndex, QObject, QPoint, QPointF,
                          QSignalBlocker, Qt, pyqtSignal)
from PyQt6.QtGui import QDropEvent, QIcon, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QListWidget

from controllers.transform_manager import TransformManager
from core.frame_storage import scratch_directory
from core.source_frames import SourceFrame, store_preview
from tests.test_control_channel import control_window as control_window
from tests.test_stack_review_gui import _window
from widgets.sharpness_curve import SharpnessCurveWidget


@pytest.fixture
def preview_workers(monkeypatch):
    import core.source_previews

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

        def isRunning(self):
            return False

    monkeypatch.setattr(core.source_previews, "SourcePreviewWorker", Worker)
    return workers


def test_curve_tracks_completed_previews_and_ignores_replaced_stack(qapp, preview_workers):
    from main import OpenFocus

    workers = preview_workers
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


@pytest.mark.parametrize("selected,completed_before_change", [
    (None, False), (0, False), (None, True),
])
def test_inclusion_change_keeps_pending_sharpness_previews(
        qapp, preview_workers, selected, completed_before_change):
    from main import OpenFocus

    workers = preview_workers
    window, manager, panel = _window()
    window.sharpness_curve = SharpnessCurveWidget()
    window.source_revision = 1
    window.current_display_index = -1
    window.update_source_view = lambda index: None
    window._start_source_previews = lambda selected_index=None: OpenFocus._start_source_previews(
        window, selected_index)
    window.raw_images = [SourceFrame(str(i), 0, (16, 16, 3), np.dtype('uint8'), None, ())
                         for i in range(3)]
    window.stack_images = [None] * 3
    manager.update_file_list([str(i) for i in range(3)], [QPixmap() for _ in range(3)])

    def complete(worker):
        for index in worker.indices:
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[::2, ::2] = 20 * (index + 1)
            object.__setattr__(worker.frames[index], 'preview', image)
            worker.ready.emit(worker.request, index, id(worker.frames[index]), image)

    try:
        TransformManager._update_sharpness_curve(window)
        window._start_source_previews(selected)
        old = workers[-1]
        if completed_before_change:
            complete(old)
        item = window.file_list.item(2)
        item.setCheckState(Qt.CheckState.Unchecked)
        item.setCheckState(Qt.CheckState.Checked)
        assert old.cancelled
        assert window.sharpness_curve.isHidden()
        old.finished.emit()

        if completed_before_change:
            assert len(workers) == 1
        else:
            current = workers[-1]
            assert current is not old
            assert current.request[:2] == (window.source_revision, id(window.raw_images))
            complete(current)
            current.finished.emit()
        assert not window.sharpness_curve.isHidden()
        assert len(window.sharpness_curve._values) == 3
        assert all(value > 0 for value in window.sharpness_curve._values)
    finally:
        window.sharpness_curve.close()
        panel.widget.close()


def test_source_drop_remaps_navigation_and_in_flight_previews(
        control_window, monkeypatch, preview_workers, tmp_path):
    window = control_window
    view = window.file_list
    curve = window.sharpness_curve
    names = ['one.nef', 'two.nef', 'three.nef', 'four.nef']

    def pattern(value):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[::2, ::2] = value
        return image

    frames = [SourceFrame(name, 0, (16, 16, 3), np.dtype('uint8'), preview, ())
              for name, preview in zip(names, [pattern(10), pattern(20), pattern(30), None])]
    window.raw_images = list(frames)
    window.image_filenames = list(names)
    window.image_source_exifs = [b'a', b'b', b'c', b'd']
    window.excluded_frames = {2}
    assert window.transform_manager.reload_image_stack(initial_index=1)
    old = preview_workers[-1]
    assert old.indices == (3,)
    view.item(3).setSelected(True)

    class InternalDrop(QDropEvent):
        def source(self):
            return view

    def native_drop(widget, event):
        # Exercise the QListWidget model's native rowsMoved path. The OS
        # drag session that stamps QDropEvent.source() is outside this test.
        model = widget.model()
        root = QModelIndex()
        assert model.moveRows(root, 1, 1, root, 4)
        assert model.moveRows(root, 2, 1, root, 0)
        event.accept()

    monkeypatch.setattr(QListWidget, 'dropEvent', native_drop)
    mime = QMimeData()
    event = InternalDrop(QPointF(1, 1), Qt.DropAction.MoveAction, mime,
                         Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    view.dropEvent(event)
    assert old.cancelled
    assert window.image_filenames == [names[3], names[0], names[2], names[1]]
    assert window.image_source_exifs == [b'd', b'a', b'c', b'b']
    assert window.excluded_frames == {2}
    assert {view.row(item) for item in view.selectedItems()} == {0, 3}
    assert window.stack_slider.value() == view.currentRow() == window.current_display_index == 3
    assert window.lbl_source_img._source_key[1] == id(frames[1])
    assert curve.isHidden()

    window.stack_slider.setValue(0)
    assert view.currentRow() == window.current_display_index == curve._current == 0
    assert window.lbl_stack_info.text() == '1 / 4'
    assert window.lbl_source_img._base_pixmap is None

    # The obsolete index is still in range and now belongs to a different frame.
    previous_icon = view.item(3).icon().cacheKey()
    old.ready.emit(old.request, 3, id(frames[3]), pattern(99))
    assert view.item(3).icon().cacheKey() == previous_icon
    old.finished.emit()
    current = preview_workers[-1]
    assert current is not old and current.indices == (0,)
    pixels = pattern(40)
    with scratch_directory(explicit_directory=tmp_path):
        object.__setattr__(frames[3], 'preview', store_preview(pixels))
    current.ready.emit(current.request, 0, id(frames[3]), pixels)
    current.finished.emit()
    assert window.lbl_source_img._base_pixmap.toImage().pixelColor(0, 0).red() == 40
    assert not curve.isHidden()
    assert sorted(range(4), key=curve._values.__getitem__) == [1, 3, 2, 0]

    x = round(1 + 2 * (curve.width() - 2) / 3)
    QTest.mouseClick(curve, Qt.MouseButton.LeftButton, pos=QPoint(x, 10))
    assert window.stack_slider.value() == view.currentRow() == curve._current == 2
    assert view.currentItem().text() == names[2]
    assert window.lbl_source_img._source_key[1] == id(frames[2])


def test_stack_refresh_reuses_scores_until_preview_pixels_change(
        control_window, monkeypatch, tmp_path):
    window = control_window
    view = window.file_list
    curve = window.sharpness_curve
    manager = window.source_manager
    names = ['one.jpg', 'two.jpg', 'three.jpg']

    def pattern(value):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[::2, ::2] = value
        return image

    with scratch_directory(explicit_directory=tmp_path):
        frames = [SourceFrame(name, 0, (16, 16, 3), np.dtype('uint8'),
                              store_preview(pattern(value)), ())
                  for name, value in zip(names, (10, 20, 30))]
        replacement = store_preview(pattern(40))
    window.raw_images = list(frames)
    window.image_filenames = names.copy()
    window.excluded_frames = {1}
    manager._review_frames = tuple(frames)
    manager._review_results = [{'index': 2, 'reference_index': 0, 'difference': 0.0}]
    manager._review_status_key = 'stack_review_found'

    calls = []
    laplacian = cv2.Laplacian

    def count_score(gray, *args, **kwargs):
        calls.append(int(gray[0, 0]))
        return laplacian(gray, *args, **kwargs)

    monkeypatch.setattr(cv2, 'Laplacian', count_score)
    assert window.transform_manager.reload_image_stack(initial_index=1)
    original = list(curve._values)
    assert calls == [10, 20, 30]
    assert view.item(2).toolTip()
    assert view.item(1).checkState() == Qt.CheckState.Unchecked

    view.item(2).setSelected(True)
    with QSignalBlocker(view):
        assert view.model().moveRows(QModelIndex(), 1, 1, QModelIndex(), 3)
    manager.apply_list_order()
    assert [view.item(i).text() for i in range(3)] == [names[0], names[2], names[1]]
    assert not any(view.item(i).toolTip() for i in range(3))
    assert view.item(2).checkState() == Qt.CheckState.Unchecked
    assert {view.row(item) for item in view.selectedItems()} == {1, 2}
    assert window.stack_slider.value() == view.currentRow() == curve._current == 2
    assert curve._values == [original[0], original[2], original[1]]

    manager.delete_source_image(view.item(0))
    assert curve._values == [original[2], original[1]]
    assert window.stack_slider.value() == view.currentRow() == curve._current == 0
    assert view.item(1).checkState() == Qt.CheckState.Unchecked
    assert calls == [10, 20, 30]

    # Selected RAW decoding can replace a preview on the same descriptor.
    object.__setattr__(frames[1], 'preview', replacement)
    assert window.transform_manager.reload_image_stack(initial_index=1)
    assert calls == [10, 20, 30, 40]
    assert curve._values[1] > original[2]


def test_stack_refresh_recomputes_scores_for_mutable_pixels(control_window):
    window = control_window
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    window.raw_images = [image]
    window.image_filenames = ['image.tif']
    assert window.transform_manager.reload_image_stack()
    assert window.sharpness_curve._values == [0.0]

    image[::2, ::2] = 90
    assert window.transform_manager.reload_image_stack()
    assert window.sharpness_curve._values[0] > 0


def test_curve_click_uses_the_plotted_x_coordinates(qapp):
    curve = SharpnessCurveWidget()
    curve.resize(200, 40)
    curve.set_data([1.0] * 211)
    clicked = []
    curve.frame_clicked.connect(clicked.append)
    QTest.mouseClick(curve, Qt.MouseButton.LeftButton, pos=QPoint(197, 10))
    assert clicked == [208]
    curve.close()
