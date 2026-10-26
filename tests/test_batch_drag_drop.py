"""Dragging folders onto the batch dialog queues them as stacks.

The dialog's folder list is the batch queue. Dragging a folder used to do
nothing there (only the main window handled drops), so the documented
"drag several stack folders" workflow silently required the picker instead.
"""
import os

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEventLoop, QMimeData, QPointF, Qt, QTimer, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent


def _make_stacks(root, count=3, frames=3):
    folders = []
    for s in range(count):
        folder = os.path.join(str(root), f"stack{s}")
        os.makedirs(folder)
        folders.append(folder)
        for i in range(frames):
            assert cv2.imwrite(
                os.path.join(folder, f"img{i}.png"),
                np.full((48, 64, 3), (s * 40 + i * 10) % 255, np.uint8))
    return folders


def _wait_import(dialog, qapp, timeout_ms=20000):
    loop = QEventLoop()

    def poll():
        if dialog._import_busy:
            QTimer.singleShot(20, poll)
        else:
            loop.quit()

    QTimer.singleShot(20, poll)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    qapp.processEvents()


def _drag(folders):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(path) for path in folders])
    return mime


def _drop_on(target, folders, app):
    """Deliver the real drag sequence: enter first, then drop.

    Qt drops a Drop event that was not preceded by an accepted DragEnter, so a
    bare dropEvent() call does not reproduce what a user does.
    """
    mime = _drag(folders)
    enter = QDragEnterEvent(
        QPointF(5, 5).toPoint(), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    app.sendEvent(target, enter)
    drop = QDropEvent(
        QPointF(5, 5), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    app.sendEvent(target, drop)
    return enter, drop


def test_drop_queues_every_folder_in_order(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog

    folders = _make_stacks(tmp_path)
    dialog = BatchProcessingDialog(None)
    try:
        enter, _drop = _drop_on(dialog, folders, qapp)
        assert enter.isAccepted(), "a folder drag must be accepted"

        assert [os.path.basename(p) for p in dialog.folder_paths] == [
            "stack0", "stack1", "stack2"]
        assert dialog.folder_list.count() == 3
        _wait_import(dialog, qapp)
        assert all(not dialog.folder_list.item(i).icon().isNull()
                   for i in range(dialog.folder_list.count()))
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_drop_on_the_list_uses_the_same_queue(qapp, tmp_path):
    """The list does not consume drops; the dialog queues them instead."""
    from dialogs.batch import BatchProcessingDialog

    folders = _make_stacks(tmp_path, count=2)
    dialog = BatchProcessingDialog(None)
    try:
        _enter, drop = _drop_on(dialog.folder_list, folders, qapp)
        assert dialog.folder_paths == [os.path.normpath(p) for p in folders]
        assert drop.isAccepted()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_drop_switches_single_folder_mode_to_multiple(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog

    folders = _make_stacks(tmp_path, count=2)
    dialog = BatchProcessingDialog(None)
    try:
        dialog.rb_single_folder.setChecked(True)
        _drop_on(dialog, folders, qapp)
        assert dialog.rb_multiple_folders.isChecked(), (
            "dropped stack folders must be processed in multi-folder mode")
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_dropped_files_are_not_treated_as_folders(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog

    _make_stacks(tmp_path, count=1)
    loose = os.path.join(str(tmp_path), "loose.png")
    assert cv2.imwrite(loose, np.zeros((16, 16, 3), np.uint8))
    dialog = BatchProcessingDialog(None)
    try:
        mime = _drag([loose])
        enter = QDragEnterEvent(
            QPointF(5, 5).toPoint(), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        qapp.sendEvent(dialog, enter)
        assert not enter.isAccepted(), (
            "a loose image is not a stack folder; the queue must not gain it")
        assert dialog.folder_paths == []
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_dropping_the_same_folder_twice_keeps_one_row(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog

    folders = _make_stacks(tmp_path, count=1)
    dialog = BatchProcessingDialog(None)
    try:
        for _ in range(2):
            _drop_on(dialog, folders, qapp)
        assert dialog.folder_paths == [os.path.normpath(folders[0])]
        assert dialog.folder_list.count() == 1
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
