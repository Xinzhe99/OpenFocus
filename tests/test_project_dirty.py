"""Project files must behave like any editor's document.

Regression: the window title never showed whether the current state was
saved, and closing (or opening another project) discarded the stack without
asking.
"""
import pytest

pytest.importorskip("PyQt6")


class _Host:
    """Minimal window carrying only what the title/dirty helpers touch."""

    def __init__(self, project_path=None):
        self.project_path = project_path
        self._project_dirty = False
        self.titles = []

    def setWindowTitle(self, text):  # noqa: N802 - Qt naming
        self.titles.append(text)


def _bind(host):
    import main
    for name in ("_update_project_title", "_mark_project_dirty",
                 "_clear_project_dirty"):
        setattr(host, name, main.OpenFocus.__dict__[name].__get__(host))
    return host


def test_title_marks_unsaved_changes():
    host = _bind(_Host())
    host._update_project_title()
    assert host.titles[-1] == "OpenFocus"

    host._mark_project_dirty()
    assert host.titles[-1] == "OpenFocus*"

    host.project_path = "C:/work/stack.ofproj"
    host._update_project_title()
    assert host.titles[-1] == "OpenFocus — stack.ofproj*"

    host._clear_project_dirty()
    assert host.titles[-1] == "OpenFocus — stack.ofproj"


def test_marking_twice_only_retitles_once():
    host = _bind(_Host())
    host._mark_project_dirty()
    count = len(host.titles)
    host._mark_project_dirty()
    assert len(host.titles) == count, "an already-dirty project must not retitle"


@pytest.mark.parametrize("action", ["open", "close"])
@pytest.mark.parametrize("choice", ["cancel", "discard", "save_ok", "save_cancel", "save_failed", "escape", "default"])
def test_dirty_project_confirmation(qapp, tmp_path, monkeypatch, action, choice):
    import numpy as np
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QMessageBox, QListWidgetItem
    from tests.test_control_channel import _destroy_main_window, _new_main_window
    from utils.file_dialogs import FileDialog
    from utils import project_file
    import main

    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        window.raw_images = [np.zeros((2, 2, 3), dtype=np.uint8)]
        window.image_filenames = ["original.png"]
        item = QListWidgetItem("original.png")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        window.file_list.addItem(item)
        window._clear_project_dirty()
        item.setCheckState(Qt.CheckState.Unchecked)
        assert window._project_dirty and window.windowTitle().endswith("*")
        assert 0 in window.excluded_frames

        opened = []
        new_path = str(tmp_path / "new.ofproj")
        monkeypatch.setattr(FileDialog, "getOpenFileName", lambda *a, **k: (opened.append(True) or new_path, ""))
        monkeypatch.setattr(project_file, "validate_project_for_open", lambda *a: (True, None, {}))
        def apply_project(host, state, on_loaded):
            host.image_filenames = ["new.png"]
            on_loaded()
        monkeypatch.setattr(project_file, "apply_project", apply_project)
        monkeypatch.setattr(FileDialog, "getSaveFileName", lambda *a, **k: ("" if choice == "save_cancel" else str(tmp_path / "saved.ofproj"), ""))
        monkeypatch.setattr(project_file, "save_project", lambda *a: (choice == "save_ok", "failed"))
        monkeypatch.setattr(main, "show_warning_box", lambda *a, **k: None)

        def respond():
            box = qapp.activeModalWidget()
            assert isinstance(box, QMessageBox)
            assert box.buttonRole(box.defaultButton()) == QMessageBox.ButtonRole.RejectRole
            if choice in ("escape", "default"):
                QTest.keyClick(box, Qt.Key.Key_Escape if choice == "escape" else Qt.Key.Key_Return)
            else:
                role = (QMessageBox.ButtonRole.RejectRole if choice == "cancel" else
                        QMessageBox.ButtonRole.DestructiveRole if choice == "discard" else
                        QMessageBox.ButtonRole.AcceptRole)
                next(button for button in box.buttons() if box.buttonRole(button) == role).click()

        QTimer.singleShot(0, respond)
        if action == "open":
            window.open_project_dialog()
            continued = bool(opened)
        else:
            continued = window.close()
        assert continued == (choice in ("discard", "save_ok"))
        if action == "open" and continued:
            assert window.image_filenames == ["new.png"]
            assert window.project_path == new_path
            assert not window._project_dirty
        else:
            assert 0 in window.excluded_frames
            assert window.image_filenames == ["original.png"]
            assert window._project_dirty == (choice != "save_ok")
    finally:
        _destroy_main_window(qapp, window)
