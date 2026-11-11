"""Reveal a file in the operating system's file manager.

Paths are never interpolated into a shell command.  Windows uses the shell
item-identifier API so Explorer selects the exact file even when its path
contains spaces or non-ASCII characters; macOS passes ``open -R`` an argument
vector, and Linux tries org.freedesktop.FileManager1 before opening the parent
directory.
"""
from __future__ import annotations

import os
import shutil
import sys

from PyQt6.QtCore import QObject, QProcess, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices


class _WindowsShellApi:
    """Minimal shell32 wrapper: ILCreateFromPathW + SHOpenFolderAndSelectItems."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        self._ole32 = ctypes.WinDLL("ole32", use_last_error=True)
        self._shell32.ILCreateFromPathW.argtypes = [wintypes.LPCWSTR]
        self._shell32.ILCreateFromPathW.restype = ctypes.c_void_p
        self._shell32.SHOpenFolderAndSelectItems.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, wintypes.DWORD]
        self._shell32.SHOpenFolderAndSelectItems.restype = ctypes.c_long
        self._shell32.ILFree.argtypes = [ctypes.c_void_p]
        self._shell32.ILFree.restype = None
        self._release_com = False
        # The GUI thread normally already is an STA; S_FALSE still counts as
        # a successful initialization and must be balanced by CoUninitialize.
        result = self._ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
        self._release_com = result in (0, 1)

    def il_create(self, path: str):
        return self._shell32.ILCreateFromPathW(path)

    def open_and_select(self, pidl) -> int:
        return int(self._shell32.SHOpenFolderAndSelectItems(pidl, 0, None, 0))

    def il_free(self, pidl) -> None:
        self._shell32.ILFree(pidl)

    def release(self) -> None:
        if self._release_com:
            self._release_com = False
            self._ole32.CoUninitialize()


def _select_windows(path: str, *, api=None) -> bool:
    """Open Explorer with ``path`` selected using native shell APIs."""
    api = api or _WindowsShellApi()
    try:
        pidl = api.il_create(path)
        if not pidl:
            return False
        try:
            result = api.open_and_select(pidl)
        finally:
            api.il_free(pidl)
        return result == 0
    finally:
        api.release()


def selection_argv(path: str, *, platform: str | None = None) -> list[str]:
    """File-manager command that selects ``path``, where an argv is used."""
    path = os.path.abspath(os.fspath(path))
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        return ["open", "-R", path]
    return []


def _start_detached(program: str, arguments: list[str], starter=None) -> bool:
    starter = starter or (lambda program, arguments: QProcess.startDetached(
        program, arguments))
    try:
        result = starter(program, arguments)
    except OSError:
        return False
    if isinstance(result, tuple):
        result = result[0] if result else False
    return bool(result)


def _open_parent(path: str, opener=None) -> bool:
    opener = opener or (lambda directory: QDesktopServices.openUrl(
        QUrl.fromLocalFile(directory)))
    try:
        return bool(opener(os.path.dirname(os.path.abspath(path))))
    except Exception:
        return False


class _LinuxReveal(QObject):
    """Own a single D-Bus request until it finishes or its window closes."""

    def __init__(self, path, parent, on_finished, opener):
        super().__init__(parent)
        self.path = path
        self.on_finished = on_finished
        self.opener = opener
        self.done = False
        self.process = QProcess(self)
        self.timed_out = False
        self.process.setStandardOutputFile(QProcess.nullDevice())
        self.process.setStandardErrorFile(QProcess.nullDevice())
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._timeout)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        if parent is not None:
            parent.destroyed.connect(self._parent_destroyed)

    def start(self):
        program = shutil.which("dbus-send")
        if not program:
            self._complete(False)
            return
        uri = QUrl.fromLocalFile(self.path).toString().replace(",", "%2C")
        self.process.start(program, ["--session", "--print-reply",
                           "--dest=org.freedesktop.FileManager1",
                           "/org/freedesktop/FileManager1",
                           "org.freedesktop.FileManager1.ShowItems",
                           f"array:string:{uri}", "string:"])
        if not self.done:
            self.timer.start(2000)

    def _finished(self, code, status):
        self._complete(not self.timed_out and code == 0 and status == QProcess.ExitStatus.NormalExit)

    def _error(self, _error):
        self._complete(False)

    def _timeout(self):
        self.timed_out = True
        self.process.kill()

    def _parent_destroyed(self):
        self.done = True
        self.timer.stop()
        self.process.kill()

    def _complete(self, selected):
        if self.done:
            return
        self.done = True
        self.timer.stop()
        success = selected or _open_parent(self.path, self.opener)
        try:
            self.on_finished(bool(success))
        finally:
            self.deleteLater()


def select_in_file_manager(path: str, *, parent, on_finished,
                           platform: str | None = None, starter=None,
                           opener=None, windows_select=None) -> None:
    """Reveal a fixed path and report whether the request was accepted."""
    path = os.path.abspath(os.fspath(path))
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        on_finished(bool((windows_select or _select_windows)(path)))
    elif platform.startswith("linux") or "bsd" in platform:
        request = _LinuxReveal(path, parent, on_finished, opener)
        request.start()
    else:
        argv = selection_argv(path, platform=platform)
        on_finished(_start_detached(argv[0], argv[1:], starter) if argv
                    else _open_parent(path, opener))
