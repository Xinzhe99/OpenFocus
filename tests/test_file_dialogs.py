"""Behavioral coverage for the shared address bar and picker result semantics."""
import os
import sys

import pytest
from PyQt6.QtCore import QItemSelectionModel, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLineEdit, QTreeView

from utils import file_dialogs
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
def test_every_picker_navigates_quoted_unicode_address_and_cancel_is_empty(qapp, tmp_path, monkeypatch, picker):
    monkeypatch.setattr(file_dialogs, "_use_native_folder_dialog", lambda: False)
    monkeypatch.setattr(file_dialogs, "_use_native_file_dialog", lambda: False)
    def forbidden_clipboard():
        raise AssertionError("Picker navigation must not access the clipboard")
    monkeypatch.setattr(QApplication, "clipboard", forbidden_clipboard)
    target = tmp_path / "目录 with spaces"
    target.mkdir()
    monkeypatch.setenv("OPENFOCUS_PICK_TEST", str(target))

    def answer(dialog):
        assert dialog.testOption(FileDialog.Option.DontUseNativeDialog)
        navigate(dialog, '"%OPENFOCUS_PICK_TEST%"')
        assert same_path(dialog.directory().absolutePath(), target)
        invalid = str(target / "missing")
        assert navigate(dialog, invalid).text() == invalid
        assert same_path(dialog.directory().absolutePath(), target)
        navigate(dialog, '"' + str(target) + '"')
        assert same_path(dialog.directory().absolutePath(), target)
        dialog.reject()

    result = answer_picker(qapp, answer, lambda: getattr(FileDialog, picker)(None, "Pick", str(tmp_path)))
    assert result == ("" if picker == "getExistingDirectory" else
                      ([], "") if picker == "getOpenFileNames" else ("", ""))
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("picker", ["getOpenFileName", "getOpenFileNames", "getSaveFileName"])
def test_file_address_only_selects_until_explicit_accept(qapp, tmp_path, monkeypatch, picker):
    monkeypatch.setattr(file_dialogs, "_use_native_file_dialog", lambda: False)
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


def test_new_save_filename_navigates_without_creating_file(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(file_dialogs, "_use_native_file_dialog", lambda: False)
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


def test_bare_save_name_gets_the_selected_filters_suffix_before_overwrite_check(
        qapp, tmp_path, monkeypatch):
    """Qt confirms replacing the file it returns, so the suffix the export
    adds for a bare name must already be in that path."""
    from PyQt6.QtWidgets import QComboBox
    monkeypatch.setattr(file_dialogs, "_use_native_file_dialog", lambda: False)
    filters = "TIFF image (*.tif *.tiff);;PNG image (*.png);;All files (*)"

    def answer(dialog):
        name = dialog.findChild(QLineEdit, "fileNameEdit")
        name.setText("out")
        assert dialog.defaultSuffix() == "png"
        assert same_path(dialog.selectedFiles()[0], tmp_path / "out.png")
        combo = dialog.findChild(QComboBox, "fileTypeCombo")
        combo.setFocus()
        # Real key presses: Qt only reports a filter change made by the user.
        for key, suffix in ((Qt.Key.Key_Down, ""), (Qt.Key.Key_Up, "png"), (Qt.Key.Key_Up, "tif")):
            QTest.keyClick(combo, key)
            assert dialog.defaultSuffix() == suffix
        name.setText("typed.png")
        assert same_path(dialog.selectedFiles()[0], tmp_path / "typed.png")
        name.setText("out")
        dialog.accept()

    result = answer_picker(qapp, answer, lambda: FileDialog.getSaveFileName(
        None, "Save", str(tmp_path), filters, "PNG image (*.png)"))
    assert same_path(result[0], tmp_path / "out.tif")
    assert result[1] == "TIFF image (*.tif *.tiff)"

def test_open_multiple_files_retains_multiple_selection(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(file_dialogs, "_use_native_file_dialog", lambda: False)
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


def test_folder_double_click_navigates_then_accepts_child_not_parent(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(file_dialogs, "_use_native_folder_dialog", lambda: False)
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


def test_multiple_folder_picker_selects_both_directories(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(file_dialogs, "_use_native_folder_dialog", lambda: False)
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


def test_picker_sidebar_places_and_visited_folders(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(file_dialogs, "_use_native_folder_dialog", lambda: False)
    from PyQt6.QtCore import QDir, QSettings, QStandardPaths
    from PyQt6.QtWidgets import QListView
    from utils import settings_store

    settings = QSettings(str(tmp_path / "picker-settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, "get_settings", lambda: settings)
    visited = tmp_path / "visited 文件夹"
    visited.mkdir()
    first = FileDialog(None, "Pick", str(tmp_path))
    try:
        first.show()
        navigate(first, str(visited))
    finally:
        first.close()
        first.deleteLater()

    dialog = FileDialog.for_directories(None, "Pick", str(tmp_path), multiple=True)
    try:
        urls = dialog.sidebarUrls()
        assert any(url.scheme() == "file" and not url.toLocalFile() for url in urls)
        paths = [url.toLocalFile() for url in urls if url.toLocalFile()]
        expected = [visited, *(drive.absoluteFilePath() for drive in QDir.drives())]
        for location in (QStandardPaths.StandardLocation.DesktopLocation,
                         QStandardPaths.StandardLocation.DocumentsLocation,
                         QStandardPaths.StandardLocation.DownloadLocation,
                         QStandardPaths.StandardLocation.HomeLocation):
            path = QStandardPaths.writableLocation(location)
            if os.path.isdir(path):
                expected.append(path)
        assert all(any(same_path(path, item) for path in paths) for item in expected)
        assert len({os.path.normcase(os.path.normpath(path)) for path in paths}) == len(paths)

        dialog.show()
        qapp.processEvents()
        sidebar = dialog.findChild(QListView, "sidebar")
        model = sidebar.model()
        row = next(i for i, path in enumerate(paths) if same_path(path, visited)) + 1
        index = model.index(row, 0)
        sidebar.scrollTo(index)
        QTest.mouseClick(sidebar.viewport(), Qt.MouseButton.LeftButton,
                         pos=sidebar.visualRect(index).center())
        qapp.processEvents()
        assert same_path(dialog.directory().absolutePath(), visited)
        assert same_path(dialog.findChild(QLineEdit, "folderPathEdit").text(), visited)
    finally:
        dialog.close()
        dialog.deleteLater()


def test_recent_picker_folders_are_bounded_deduplicated_and_existing(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from utils import settings_store

    settings = QSettings(str(tmp_path / "picker-settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, "get_settings", lambda: settings)
    folders = [tmp_path / str(i) for i in range(settings_store.MAX_RECENT_DIALOG_DIRS + 1)]
    for folder in folders:
        folder.mkdir()
        settings_store.add_recent_dialog_dir(str(folder))
    settings_store.add_recent_dialog_dir(str(folders[1]))
    recent = settings_store.get_recent_dialog_dirs()
    assert recent == [str(folders[1]), *map(str, reversed(folders[2:]))]
    folders[1].rmdir()
    assert str(folders[1]) not in settings_store.get_recent_dialog_dirs()


def test_windows_folder_picker_routes_results_and_cancellation(monkeypatch, tmp_path):
    monkeypatch.setattr(file_dialogs, "_use_native_folder_dialog", lambda: True)
    calls = []
    remembered = []
    results = [[str(tmp_path / "one"), str(tmp_path / "二")], [], [str(tmp_path / "one")]]
    monkeypatch.setattr(file_dialogs, "_native_folder_paths",
                        lambda *args: calls.append(args) or results.pop(0))
    monkeypatch.setattr(file_dialogs, "_remember_dialog_directory", remembered.append)

    parent = object()
    dialog = FileDialog.for_directories(parent, "Folders", str(tmp_path), multiple=True)
    assert dialog.exec() == FileDialog.DialogCode.Accepted
    assert dialog.selectedFiles() == [str(tmp_path / "one"), str(tmp_path / "二")]
    assert dialog.exec() == FileDialog.DialogCode.Rejected
    assert dialog.selectedFiles() == []
    assert FileDialog.getExistingDirectory(parent, "Single", str(tmp_path)) == str(tmp_path / "one")
    assert calls == [(parent, "Folders", str(tmp_path), True),
                     (parent, "Folders", str(tmp_path), True),
                     (parent, "Single", str(tmp_path), False)]
    assert remembered == [str(tmp_path / "二"), str(tmp_path / "one")]


def test_native_folder_picker_displays_backend_errors(monkeypatch):
    monkeypatch.setattr(file_dialogs, "_use_native_folder_dialog", lambda: True)

    def fail(*args):
        raise OSError("COM failed")

    errors = []
    monkeypatch.setattr(file_dialogs, "_native_folder_paths", fail)
    monkeypatch.setattr(file_dialogs.QMessageBox, "critical",
                        lambda parent, title, message: errors.append((title, message)))
    assert FileDialog.getExistingDirectory(None, "Pick") == ""
    assert errors == [("Pick", "COM failed")]


@pytest.mark.parametrize("platform,backend", [
    ("win32", "_windows_folder_paths"),
    ("darwin", "_macos_folder_paths"),
    ("linux", "_portal_folder_paths"),
])
def test_native_folder_backend_dispatch(monkeypatch, platform, backend):
    calls = []
    monkeypatch.setattr(file_dialogs.sys, "platform", platform)
    monkeypatch.setattr(file_dialogs, backend, lambda *args: calls.append(args) or ["folder"])
    assert file_dialogs._native_folder_paths(None, "Pick", "/start", True) == ["folder"]
    assert calls == [(None, "Pick", "/start", True)]


@pytest.mark.parametrize("response,expected", [
    ((0, {"uris": ["file:///tmp/first%20folder", "file:///tmp/second%0Aline"]}),
     ["/tmp/first folder", "/tmp/second\nline"]),
    ((1, {}), []),
    ((2, {}), None),
])
def test_portal_folder_selection_and_cancel(qapp, monkeypatch, tmp_path, response, expected):
    import types

    class Variant:
        def __init__(self, value):
            self.value = value

        def variant(self):
            return self.value

    class Message:
        class MessageType:
            ErrorMessage = 1

        def __init__(self, method="", args=None):
            self.method, self.args = method, args or []

        @classmethod
        def createMethodCall(cls, service, path, interface, method):
            return cls(method)

        def setArguments(self, args):
            self.args = args

        def arguments(self):
            return self.args

        def type(self):
            return 0

    class Bus:
        def isConnected(self):
            return True

        def baseService(self):
            return ":1.42"

        def connect(self, service, path, interface, signal, callback):
            self.path, self.callback = path, callback
            return True

        def disconnect(self, *args):
            self.disconnected = True

        def call(self, message, mode):
            if message.method == "Get":
                return Message(args=[Variant(4)])
            self.options = message.args[2]
            self.callback(*response)
            return Message(args=[types.SimpleNamespace(path=lambda: self.path)])

    bus = Bus()
    dbus = types.ModuleType("PyQt6.QtDBus")
    dbus.QDBus = types.SimpleNamespace(CallMode=types.SimpleNamespace(BlockWithGui=1))
    dbus.QDBusConnection = types.SimpleNamespace(sessionBus=lambda: bus)
    dbus.QDBusMessage = Message
    dbus.QDBusVariant = Variant
    monkeypatch.setitem(sys.modules, "PyQt6.QtDBus", dbus)
    if expected is None:
        with pytest.raises(OSError, match="response 2"):
            file_dialogs._portal_folder_paths(None, "Pick", str(tmp_path), True)
    else:
        assert file_dialogs._portal_folder_paths(None, "Pick", str(tmp_path), True) == expected
    assert bus.options["directory"].variant() is True
    assert bus.options["multiple"].variant() is True
    assert bytes(bus.options["current_folder"].variant()).endswith(b"\0")
    assert bus.disconnected


@pytest.mark.parametrize("version,connected,error", [
    (4, False, "D-Bus session"),
    (2, True, "version 3"),
])
def test_portal_reports_unavailable_backend(qapp, monkeypatch, version, connected, error):
    import types

    class Variant:
        def variant(self):
            return version

    class Message:
        MessageType = types.SimpleNamespace(ErrorMessage=1)

        @classmethod
        def createMethodCall(cls, *args):
            return cls()

        def setArguments(self, args):
            pass

        def type(self):
            return 0

        def arguments(self):
            return [Variant()]

    class Bus:
        def isConnected(self):
            return connected

        def call(self, *args):
            return Message()

    dbus = types.ModuleType("PyQt6.QtDBus")
    dbus.QDBus = types.SimpleNamespace(CallMode=types.SimpleNamespace(BlockWithGui=1))
    dbus.QDBusConnection = types.SimpleNamespace(sessionBus=Bus)
    dbus.QDBusMessage = Message
    dbus.QDBusVariant = Variant
    monkeypatch.setitem(sys.modules, "PyQt6.QtDBus", dbus)
    with pytest.raises(OSError, match=error):
        file_dialogs._portal_folder_paths(None, "Pick", "", True)


def test_unsupported_platform_reports_no_native_picker(monkeypatch):
    monkeypatch.setattr(file_dialogs.sys, "platform", "unknown")
    with pytest.raises(OSError, match="No native folder picker"):
        file_dialogs._native_folder_paths(None, "", "", False)


@pytest.mark.parametrize("picker,mode,accept", [
    ("getOpenFileName", FileDialog.FileMode.ExistingFile, FileDialog.AcceptMode.AcceptOpen),
    ("getOpenFileNames", FileDialog.FileMode.ExistingFiles, FileDialog.AcceptMode.AcceptOpen),
    ("getSaveFileName", FileDialog.FileMode.AnyFile, FileDialog.AcceptMode.AcceptSave),
])
def test_native_file_picker_forwards_settings_and_selection(monkeypatch, tmp_path, picker, mode, accept):
    calls = []
    selected = [str(tmp_path / "结果.tif")]
    monkeypatch.setattr(file_dialogs, "_use_native_file_dialog", lambda: True)
    monkeypatch.setattr(file_dialogs, "_pick_native_file",
                        lambda *args: (calls.append(args) or selected, "TIFF (*.tif)"))
    options = FileDialog.Option.DontConfirmOverwrite
    result = getattr(FileDialog, picker)(None, "Pick", tmp_path, "PNG (*.png);;TIFF (*.tif)",
                                         "TIFF (*.tif)", options)
    assert result == (selected if picker == "getOpenFileNames" else selected[0], "TIFF (*.tif)")
    assert calls == [(None, "Pick", tmp_path, "PNG (*.png);;TIFF (*.tif)",
                      "TIFF (*.tif)", options, mode, accept)]


@pytest.mark.parametrize("accepted", [True, False])
def test_native_file_dialog_returns_results_and_preserves_options(qapp, monkeypatch, tmp_path, accepted):
    actual = file_dialogs.QFileDialog
    remembered = []
    dialogs = []

    class Signal:
        def connect(self, callback):
            self.callback = callback

    class NativeDialog:
        Option = actual.Option
        DialogCode = actual.DialogCode

        def __init__(self, parent, caption, directory, name_filter):
            self.arguments = (parent, caption, directory, name_filter)
            self.directoryEntered, self.finished = Signal(), Signal()
            self.filterSelected, self.suffix = Signal(), None
            dialogs.append(self)

        def setOptions(self, value): self.options = value
        def setFileMode(self, value): self.mode = value
        def setAcceptMode(self, value): self.accept = value
        def selectNameFilter(self, value): self.name_filter = value
        def setDefaultSuffix(self, value): self.suffix = value
        def layout(self): return None
        def open(self): self.finished.callback(actual.DialogCode.Accepted if accepted else actual.DialogCode.Rejected)
        def selectedFiles(self): return [str(tmp_path / "文件.tif")]
        def selectedNameFilter(self): return self.name_filter
        def deleteLater(self): self.deleted = True

    monkeypatch.setattr(file_dialogs, "QFileDialog", NativeDialog)
    monkeypatch.setattr(file_dialogs, "_remember_dialog_directory", remembered.append)
    options = actual.Option.DontConfirmOverwrite | actual.Option.DontUseNativeDialog
    result = file_dialogs._pick_native_file(None, "Save", tmp_path / "文件.tif", "TIFF (*.tif)",
                                           "TIFF (*.tif)", options, actual.FileMode.AnyFile,
                                           actual.AcceptMode.AcceptSave)
    dialog = dialogs[0]
    assert dialog.arguments == (None, "Save", str(tmp_path / "文件.tif"), "TIFF (*.tif)")
    assert dialog.options == actual.Option.DontConfirmOverwrite
    assert dialog.mode == actual.FileMode.AnyFile and dialog.accept == actual.AcceptMode.AcceptSave
    # A bare name is completed inside the platform dialog too, and follows the filter.
    assert dialog.suffix == "tif"
    dialog.filterSelected.callback("PNG (*.png)")
    assert dialog.suffix == "png"
    assert result == (([str(tmp_path / "文件.tif")], "TIFF (*.tif)") if accepted else ([], ""))
    assert remembered == ([str(tmp_path)] if accepted else [])
    assert dialog.deleted


def test_native_file_dialog_reports_unavailable_helper(qapp, monkeypatch):
    actual = file_dialogs.QFileDialog
    errors = []

    class NoNativeDialog:
        Option = actual.Option

        def __init__(self, *args): self.directoryEntered = type("Signal", (), {"connect": lambda *args: None})()
        def setOptions(self, value): pass
        def setFileMode(self, value): pass
        def setAcceptMode(self, value): pass
        def layout(self): return object()
        def deleteLater(self): pass

    monkeypatch.setattr(file_dialogs, "QFileDialog", NoNativeDialog)
    monkeypatch.setattr(file_dialogs.QMessageBox, "critical", lambda *args: errors.append(args))
    assert file_dialogs._pick_native_file(None, "Open", "", "", "", actual.Option(0),
                                          actual.FileMode.ExistingFile, actual.AcceptMode.AcceptOpen) == ([], "")
    assert errors and "unavailable" in errors[0][2]
