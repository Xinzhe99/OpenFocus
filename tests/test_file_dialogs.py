"""Behavioral coverage for the shared address bar and picker result semantics."""
import os
import sys

import pytest
from PyQt6.QtCore import QItemSelectionModel, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLineEdit, QTreeView

from utils.file_dialogs import FileDialog, expand_dialog_path


def same_path(actual, expected):
    return os.path.normcase(os.path.normpath(actual)) == os.path.normcase(str(expected))


def navigate(dialog, text):
    edit = dialog.findChild(QLineEdit, "folderPathEdit")
    edit.setFocus()
    edit.setText(text)
    QTest.keyClick(edit, Qt.Key.Key_Return)
    assert dialog.isVisible(), "Address Enter must never accept a picker"
    return edit


def answer_picker(qapp, callback, invoke):
    errors = []
    watchdog = QTimer()
    watchdog.setSingleShot(True)

    def timed_out():
        errors.append(AssertionError("Picker did not finish its scheduled interaction"))
        active = qapp.activeModalWidget()
        if active is not None:
            active.reject()
        for widget in qapp.topLevelWidgets():
            if isinstance(widget, FileDialog) and widget.isVisible():
                widget.reject()

    watchdog.timeout.connect(timed_out)

    def answer():
        dialog = qapp.activeModalWidget()
        try:
            assert isinstance(dialog, FileDialog)
            callback(dialog)
        except BaseException as exc:
            errors.append(exc)
            if dialog is not None:
                dialog.reject()

    QTimer.singleShot(0, answer)
    watchdog.start(3000)
    try:
        result = invoke()
    finally:
        watchdog.stop()
    if errors:
        raise errors[0]
    return result


@pytest.mark.parametrize("picker", ["getOpenFileName", "getOpenFileNames", "getSaveFileName", "getExistingDirectory"])
def test_every_picker_navigates_pasted_unicode_address_and_cancel_is_empty(qapp, tmp_path, monkeypatch, picker):
    target = tmp_path / "目录 with spaces"
    target.mkdir()
    monkeypatch.setenv("OPENFOCUS_PICK_TEST", str(target))

    def answer(dialog):
        assert dialog.testOption(FileDialog.Option.DontUseNativeDialog)
        edit = navigate(dialog, '"%OPENFOCUS_PICK_TEST%"')
        assert same_path(dialog.directory().absolutePath(), target)
        invalid = str(target / "missing")
        assert navigate(dialog, invalid).text() == invalid
        assert same_path(dialog.directory().absolutePath(), target)
        edit.clear()
        qapp.clipboard().setText('"' + str(target) + '"')
        edit.paste()
        QTest.keyClick(edit, Qt.Key.Key_Return)
        assert same_path(dialog.directory().absolutePath(), target)
        dialog.reject()

    result = answer_picker(qapp, answer, lambda: getattr(FileDialog, picker)(None, "Pick", str(tmp_path)))
    assert result == ("" if picker == "getExistingDirectory" else
                      ([], "") if picker == "getOpenFileNames" else ("", ""))
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("picker", ["getOpenFileName", "getOpenFileNames", "getSaveFileName"])
def test_file_address_only_selects_until_explicit_accept(qapp, tmp_path, picker):
    folder = tmp_path / "目录"
    folder.mkdir()
    source = folder / "frame space.tif"
    source.write_bytes(b"keep original")

    def answer(dialog):
        navigate(dialog, '"' + str(source) + '"')
        assert same_path(dialog.directory().absolutePath(), folder)
        assert same_path(dialog.selectedFiles()[0], source)
        assert source.read_bytes() == b"keep original"
        # Save must stay open without even raising an overwrite confirmation.
        if picker == "getSaveFileName":
            dialog.reject()
        else:
            dialog.accept()

    result = answer_picker(qapp, answer, lambda: getattr(FileDialog, picker)(None, "Pick", str(tmp_path), "TIFF (*.tif)"))
    if picker == "getSaveFileName":
        assert result == ("", "")
    elif picker == "getOpenFileNames":
        assert len(result[0]) == 1 and same_path(result[0][0], source)
    else:
        assert same_path(result[0], source)
    assert source.read_bytes() == b"keep original"


def test_new_save_filename_navigates_without_creating_file(qapp, tmp_path):
    target = tmp_path / "new 文件.tif"

    def answer(dialog):
        navigate(dialog, str(target))
        assert same_path(dialog.selectedFiles()[0], target)
        assert not target.exists()
        dialog.accept()

    result = answer_picker(qapp, answer, lambda: FileDialog.getSaveFileName(None, "Save", str(tmp_path), "TIFF (*.tif)"))
    assert same_path(result[0], target)
    assert result[1] == "TIFF (*.tif)"
    assert not target.exists()


def test_open_multiple_files_retains_multiple_selection(qapp, tmp_path):
    paths = [tmp_path / "first frame.tif", tmp_path / "第二帧.tif"]
    for path in paths:
        path.write_bytes(b"picker only")

    def answer(dialog):
        navigate(dialog, str(tmp_path))
        filename = dialog.findChild(QLineEdit, "fileNameEdit")
        filename.setText(' '.join('"' + path.name + '"' for path in paths))
        dialog.accept()

    result = answer_picker(qapp, answer, lambda: FileDialog.getOpenFileNames(None, "Open", str(tmp_path), "TIFF (*.tif)"))
    assert {os.path.normcase(os.path.normpath(p)) for p in result[0]} == {
        os.path.normcase(str(p)) for p in paths}


def test_ctrl_l_focuses_address_without_touching_filename(qapp, tmp_path):
    dialog = FileDialog(None, "Pick", str(tmp_path))
    try:
        dialog.show()
        dialog.activateWindow()
        QTest.qWait(30)
        filename = dialog.findChild(QLineEdit, "fileNameEdit")
        filename.setText("result.tif")
        filename.setFocus()
        QTest.keyClick(dialog, Qt.Key.Key_L, Qt.KeyboardModifier.ControlModifier)
        edit = dialog.findChild(QLineEdit, "folderPathEdit")
        assert edit.hasFocus()
        assert edit.selectedText() == edit.text()
        assert filename.text() == "result.tif"
    finally:
        dialog.reject()
        dialog.deleteLater()


def wait_for_row(qapp, view, path):
    for _ in range(150):
        qapp.processEvents()
        index = view.model().index(str(path))
        if index.isValid() and view.visualRect(index).isValid():
            return index
        QTest.qWait(20)
    pytest.fail("Picker filesystem model did not load the requested row")


def test_folder_double_click_navigates_then_accepts_child_not_parent(qapp, tmp_path):
    child = tmp_path / "child 目录"
    child.mkdir()
    dialog = FileDialog.for_directories(None, "Folders", str(tmp_path))
    try:
        dialog.setViewMode(FileDialog.ViewMode.Detail)
        dialog.show()
        view = dialog.findChild(QTreeView, "treeView")
        index = wait_for_row(qapp, view, child)
        view.setCurrentIndex(index)
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                        pos=view.visualRect(index).center())
        QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton,
                         pos=view.visualRect(index).center())
        qapp.processEvents()
        assert dialog.isVisible()
        assert same_path(dialog.directory().absolutePath(), child)
        dialog.accept()
        assert same_path(dialog.selectedFiles()[0], child)
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_multiple_folder_picker_selects_both_directories(qapp, tmp_path):
    folders = [tmp_path / "first", tmp_path / "第二"]
    for folder in folders:
        folder.mkdir()
    dialog = FileDialog.for_directories(None, "Folders", str(tmp_path), multiple=True)
    try:
        dialog.setViewMode(FileDialog.ViewMode.Detail)
        dialog.show()
        view = dialog.findChild(QTreeView, "treeView")
        for folder in folders:
            index = wait_for_row(qapp, view, folder)
            view.selectionModel().select(index, QItemSelectionModel.SelectionFlag.Select |
                                         QItemSelectionModel.SelectionFlag.Rows)
        dialog.accept()
        assert dialog.result() == FileDialog.DialogCode.Accepted
        assert {os.path.normcase(os.path.normpath(p)) for p in dialog.selectedFiles()} == {
            os.path.normcase(str(p)) for p in folders}
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_relative_home_environment_and_inner_quotes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENFOCUS_PICK_TEST", str(tmp_path))
    assert same_path(expand_dialog_path('"$OPENFOCUS_PICK_TEST/目录 with spaces"', str(tmp_path)), tmp_path / "目录 with spaces")
    assert same_path(expand_dialog_path("'${OPENFOCUS_PICK_TEST}/目录'", str(tmp_path)), tmp_path / "目录")
    assert same_path(expand_dialog_path("~/Pictures", str(tmp_path)), os.path.join(os.path.expanduser("~"), "Pictures"))
    assert same_path(expand_dialog_path("../sibling", str(tmp_path / "child")), tmp_path / "sibling")
    assert same_path(expand_dialog_path('"a\'b.tif"', str(tmp_path)), tmp_path / "a'b.tif")
    assert expand_dialog_path("   ", str(tmp_path)) == ""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows drive and UNC syntax")
@pytest.mark.parametrize("address", [r'C:\目录 with spaces', r'\\server\共享\目录'])
def test_windows_drive_and_unc_addresses_preserved(address, tmp_path):
    assert expand_dialog_path('"' + address + '"', str(tmp_path)) == address
