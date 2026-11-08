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
import subprocess
import sys

from PyQt6.QtCore import QProcess, QUrl
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


def _select_linux(path: str, *, runner=None, opener=None) -> bool:
    """Ask the desktop file manager to select the file, else open its folder."""
    path = os.path.abspath(os.fspath(path))
    if shutil.which("dbus-send"):
        # dbus-send parses its argv itself; quotes would become part of the
        # URI. A literal comma separates array elements, so the path's commas
        # are percent-encoded and decoded again by the file manager.
        uri = QUrl.fromLocalFile(path).toString().replace(",", "%2C")
        arguments = [
            "dbus-send", "--session", "--print-reply",
            "--dest=org.freedesktop.FileManager1",
            "/org/freedesktop/FileManager1",
            "org.freedesktop.FileManager1.ShowItems",
            f"array:string:{uri}", "string:",
        ]
        runner = runner or subprocess.run
        try:
            completed = runner(
                arguments, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=2, check=False)
        except (OSError, subprocess.TimeoutExpired):
            completed = None
        if completed is not None and completed.returncode == 0:
            return True
    return _open_parent(path, opener)


def select_in_file_manager(path: str, *, platform: str | None = None,
                           starter=None, runner=None, opener=None,
                           windows_select=None) -> bool:
    """Open the system file manager with ``path`` selected when supported."""
    path = os.path.abspath(os.fspath(path))
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        return (windows_select or _select_windows)(path)
    if platform.startswith("linux") or "bsd" in platform:
        return _select_linux(path, runner=runner, opener=opener)
    argv = selection_argv(path, platform=platform)
    if argv:
        return _start_detached(argv[0], argv[1:], starter)
    return _open_parent(path, opener)
