"""Application file pickers with an always reachable, editable address bar."""
from __future__ import annotations

import os
import re

from PyQt6.QtWidgets import QFileDialog, QLineEdit, QListView, QTreeView, QAbstractItemView

from utils.ui_utils import make_folder_path_editable


def expand_dialog_path(text: str, directory: str) -> str:
    """Expand a pasted address relative to the picker, preserving inner quotes."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    if not text:
        return ""

    def variable(match):
        name = next(group for group in match.groups() if group is not None)
        return os.environ.get(name, os.path.expanduser("~") if name == "HOME" else match[0])

    text = re.sub(r"%([^%]+)%|\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)", variable, text)
    text = os.path.expanduser(text)
    return os.path.normpath(text if os.path.isabs(text) else os.path.join(directory, text))


class FileDialog(QFileDialog):
    """Shared Qt picker. Static helpers remain injectable at the application API."""

    def __init__(self, parent=None, caption="", directory="", filter=""):
        super().__init__(parent, caption, directory, filter)
        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        make_folder_path_editable(self)

    def setDirectory(self, directory):
        super().setDirectory(directory)
        edit = getattr(self, "_folder_path_edit", None)
        if edit is not None:
            edit.setText(self.directory().absolutePath())
            edit.setToolTip("")
        # A directory selected before navigation must not survive as the next
        # pick (in particular after double clicking into a child directory).
        if self.fileMode() == self.FileMode.Directory:
            filename = self.findChild(QLineEdit, "fileNameEdit")
            if filename is not None:
                filename.clear()

    @classmethod
    def for_directories(cls, parent=None, caption="", directory="", *, multiple=False):
        dialog = cls(parent, caption, directory)
        dialog.setFileMode(cls.FileMode.Directory)
        dialog.setOption(cls.Option.ShowDirsOnly, True)
        if multiple:
            for view_class, name in ((QListView, "listView"), (QTreeView, "treeView")):
                view = dialog.findChild(view_class, name)
                if view is not None:
                    view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        return dialog

    @classmethod
    def _pick(cls, parent, caption, directory, filter, selectedFilter, options, mode, accept):
        dialog = cls(parent, caption, directory, filter)
        dialog.setOptions(options | cls.Option.DontUseNativeDialog)
        dialog.setFileMode(mode)
        dialog.setAcceptMode(accept)
        if selectedFilter:
            dialog.selectNameFilter(selectedFilter)
        try:
            if dialog.exec() == cls.DialogCode.Accepted:
                return dialog.selectedFiles(), dialog.selectedNameFilter()
            return [], ""
        finally:
            dialog.deleteLater()

    @classmethod
    def getOpenFileName(cls, parent=None, caption="", directory="", filter="",
                        initialFilter="", options=QFileDialog.Option(0)):
        files, selected = cls._pick(parent, caption, directory, filter, initialFilter,
                                    options, cls.FileMode.ExistingFile, cls.AcceptMode.AcceptOpen)
        return (files[0] if files else ""), selected

    @classmethod
    def getOpenFileNames(cls, parent=None, caption="", directory="", filter="",
                         initialFilter="", options=QFileDialog.Option(0)):
        return cls._pick(parent, caption, directory, filter, initialFilter,
                         options, cls.FileMode.ExistingFiles, cls.AcceptMode.AcceptOpen)

    @classmethod
    def getSaveFileName(cls, parent=None, caption="", directory="", filter="",
                        initialFilter="", options=QFileDialog.Option(0)):
        files, selected = cls._pick(parent, caption, directory, filter, initialFilter,
                                    options, cls.FileMode.AnyFile, cls.AcceptMode.AcceptSave)
        return (files[0] if files else ""), selected

    @classmethod
    def getExistingDirectory(cls, parent=None, caption="", directory="",
                             options=QFileDialog.Option.ShowDirsOnly):
        files, _ = cls._pick(parent, caption, directory, "", "", options,
                             cls.FileMode.Directory, cls.AcceptMode.AcceptOpen)
        return files[0] if files else ""
