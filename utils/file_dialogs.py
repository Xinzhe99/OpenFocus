"""Application file pickers with an always reachable, editable address bar."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
import re
import sys
import uuid

from PyQt6.QtCore import QDir, QStandardPaths, QUrl
from PyQt6.QtWidgets import (QFileDialog, QLineEdit, QListView, QTreeView,
                             QAbstractItemView, QMessageBox, QComboBox, QSizePolicy)

from utils.ui_utils import make_folder_path_editable

_control_picker = ContextVar("control_picker", default=False)


@contextmanager
def control_picker_context():
    """Use addressable Qt pickers during a control-initiated interaction."""
    token = _control_picker.set(True)
    try:
        yield
    finally:
        _control_picker.reset(token)


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


def _sidebar_urls(dialog):
    """Use Qt's filesystem places and real folders visited in this app."""
    from utils.settings_store import get_recent_dialog_dirs

    urls = []
    seen = set()

    def add_path(path):
        if not path or not os.path.isdir(path):
            return
        path = os.path.abspath(path)
        key = os.path.normcase(os.path.normpath(path))
        if key not in seen:
            seen.add(key)
            urls.append(QUrl.fromLocalFile(path))

    # The empty file URL is Qt's computer/drives entry, not a filesystem path.
    urls.append(QUrl("file:"))
    for location in (QStandardPaths.StandardLocation.DesktopLocation,
                     QStandardPaths.StandardLocation.DocumentsLocation,
                     QStandardPaths.StandardLocation.DownloadLocation,
                     QStandardPaths.StandardLocation.HomeLocation):
        add_path(QStandardPaths.writableLocation(location))
    for drive in QDir.drives():
        add_path(drive.absoluteFilePath())
    for url in dialog.sidebarUrls():
        if url.isLocalFile():
            add_path(url.toLocalFile())
    for path in get_recent_dialog_dirs():
        add_path(path)
    return urls


def _remember_dialog_directory(path):
    from utils.settings_store import add_recent_dialog_dir
    add_recent_dialog_dir(path)

def _use_native_folder_dialog():
    return not _control_picker.get()

def _windows_folder_paths(parent, caption, directory, multiple):
    """Show the Windows shell folder picker and return filesystem paths."""
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]

        def __init__(self, value):
            super().__init__((ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le))

    ptr = ctypes.c_void_p
    hresult = ctypes.c_int32
    ole = ctypes.WinDLL("ole32")
    shell = ctypes.WinDLL("shell32")
    ole.CoInitializeEx.argtypes = [ptr, wintypes.DWORD]
    ole.CoInitializeEx.restype = hresult
    ole.CoUninitialize.argtypes = []
    ole.CoTaskMemFree.argtypes = [ptr]
    ole.CoCreateInstance.argtypes = [ctypes.POINTER(GUID), ptr, wintypes.DWORD,
                                     ctypes.POINTER(GUID), ctypes.POINTER(ptr)]
    ole.CoCreateInstance.restype = hresult
    shell.SHCreateItemFromParsingName.argtypes = [wintypes.LPCWSTR, ptr,
                                                   ctypes.POINTER(GUID), ctypes.POINTER(ptr)]
    shell.SHCreateItemFromParsingName.restype = hresult

    def check(result, operation):
        if result < 0:
            raise OSError(f"{operation} failed (HRESULT 0x{result & 0xffffffff:08X})")

    def call(obj, index, result_type, argtypes, *args):
        table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ptr))).contents
        method = ctypes.WINFUNCTYPE(result_type, ptr, *argtypes)(table[index])
        return method(obj, *args)

    def release(obj):
        if obj.value:
            call(obj, 2, wintypes.ULONG, ())

    initialized = ole.CoInitializeEx(None, 2)  # Shell dialogs require an STA.
    check(initialized, "CoInitializeEx")
    dialog = ptr()
    try:
        clsid = GUID("dc1c5a9c-e88a-4dde-a5a1-60f82a20aef7")
        iid_dialog = GUID("d57c7288-d4ad-4768-be02-9d969532d960")
        iid_item = GUID("43826d1e-e718-42ee-bc55-a1e261c37bfe")
        check(ole.CoCreateInstance(ctypes.byref(clsid), None, 1,
                                   ctypes.byref(iid_dialog), ctypes.byref(dialog)),
              "CoCreateInstance(IFileOpenDialog)")
        flags = wintypes.DWORD()
        check(call(dialog, 10, hresult, (ctypes.POINTER(wintypes.DWORD),),
                   ctypes.byref(flags)), "GetOptions")
        # FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST.
        flags.value |= 0x20 | 0x40 | 0x800
        if multiple:
            flags.value |= 0x200  # FOS_ALLOWMULTISELECT
        check(call(dialog, 9, hresult, (wintypes.DWORD,), flags), "SetOptions")
        if caption:
            check(call(dialog, 17, hresult, (wintypes.LPCWSTR,), caption), "SetTitle")
        if directory and os.path.isdir(directory):
            folder = ptr()
            check(shell.SHCreateItemFromParsingName(os.path.abspath(directory), None,
                                                     ctypes.byref(iid_item), ctypes.byref(folder)),
                  "SHCreateItemFromParsingName")
            try:
                check(call(dialog, 12, hresult, (ptr,), folder), "SetFolder")
            finally:
                release(folder)
        owner = int(parent.winId()) if parent is not None else 0
        result = call(dialog, 3, hresult, (wintypes.HWND,), owner)
        if result & 0xffffffff == 0x800704C7:  # HRESULT_FROM_WIN32(ERROR_CANCELLED)
            return []
        check(result, "IFileOpenDialog.Show")
        items = ptr()
        check(call(dialog, 27, hresult, (ctypes.POINTER(ptr),), ctypes.byref(items)),
              "GetResults")
        try:
            count = wintypes.DWORD()
            check(call(items, 7, hresult, (ctypes.POINTER(wintypes.DWORD),),
                       ctypes.byref(count)), "GetCount")
            paths = []
            for index in range(count.value):
                item = ptr()
                check(call(items, 8, hresult, (wintypes.DWORD, ctypes.POINTER(ptr)),
                           index, ctypes.byref(item)), "GetItemAt")
                try:
                    name = ptr()
                    check(call(item, 5, hresult, (ctypes.c_uint32, ctypes.POINTER(ptr)),
                               0x80058000, ctypes.byref(name)), "GetDisplayName")
                    try:
                        paths.append(ctypes.wstring_at(name))
                    finally:
                        ole.CoTaskMemFree(name)
                finally:
                    release(item)
            return paths
        finally:
            release(items)
    finally:
        release(dialog)
        ole.CoUninitialize()

def _macos_folder_paths(parent, caption, directory, multiple):
    """Use NSOpenPanel so multiple folders remain a native Finder selection."""
    import ctypes

    try:
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        ctypes.CDLL("/System/Library/Frameworks/AppKit.framework/AppKit")
    except OSError as exc:
        raise OSError("The macOS folder picker is unavailable") from exc
    ptr = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ptr
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ptr
    address = ctypes.cast(objc.objc_msgSend, ptr).value

    def send(obj, selector, result=ptr, types=(), *args):
        fn = ctypes.CFUNCTYPE(result, ptr, ptr, *types)(address)
        return fn(obj, objc.sel_registerName(selector.encode()), *args)

    def nsstring(text):
        return send(objc.objc_getClass(b"NSString"), "stringWithUTF8String:", ptr,
                    (ctypes.c_char_p,), os.fsencode(text))

    pool = send(send(objc.objc_getClass(b"NSAutoreleasePool"), "alloc"), "init")
    try:
        panel = send(objc.objc_getClass(b"NSOpenPanel"), "openPanel")
        if not panel:
            raise OSError("The macOS folder picker is unavailable")
        for selector, value in (("setCanChooseDirectories:", True),
                                ("setCanChooseFiles:", False),
                                ("setAllowsMultipleSelection:", multiple)):
            send(panel, selector, None, (ctypes.c_bool,), value)
        if caption:
            send(panel, "setTitle:", None, (ptr,), nsstring(caption))
        if directory and os.path.isdir(directory):
            url = send(objc.objc_getClass(b"NSURL"), "fileURLWithPath:", ptr,
                       (ptr,), nsstring(os.path.abspath(directory)))
            send(panel, "setDirectoryURL:", None, (ptr,), url)
        if send(panel, "runModal", ctypes.c_long) != 1:
            return []
        urls = send(panel, "URLs")
        count = send(urls, "count", ctypes.c_ulong)
        paths = []
        for index in range(count):
            url = send(urls, "objectAtIndex:", ptr, (ctypes.c_ulong,), index)
            path = send(url, "path")
            encoded = send(path, "UTF8String", ctypes.c_char_p)
            if not encoded:
                raise OSError("The macOS folder picker returned an invalid path")
            paths.append(os.fsdecode(encoded))
        return paths
    finally:
        send(pool, "drain", None)


def _portal_folder_paths(parent, caption, directory, multiple):
    """Select folders through the desktop's xdg-desktop-portal backend."""
    try:
        from PyQt6.QtCore import QByteArray, QEventLoop, QObject, pyqtSlot
        from PyQt6.QtDBus import QDBus, QDBusConnection, QDBusMessage, QDBusVariant
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        raise OSError("The desktop folder picker requires Qt D-Bus") from exc

    bus = QDBusConnection.sessionBus()
    if not bus.isConnected():
        raise OSError("The desktop folder picker requires a D-Bus session")
    service = "org.freedesktop.portal.Desktop"
    portal_path = "/org/freedesktop/portal/desktop"
    chooser = "org.freedesktop.portal.FileChooser"
    request_interface = "org.freedesktop.portal.Request"
    properties = QDBusMessage.createMethodCall(service, portal_path,
                                                "org.freedesktop.DBus.Properties", "Get")
    properties.setArguments([chooser, "version"])
    reply = bus.call(properties, QDBus.CallMode.BlockWithGui)
    if reply.type() == QDBusMessage.MessageType.ErrorMessage or not reply.arguments():
        raise OSError("The desktop folder picker portal is unavailable: " + reply.errorMessage())
    version = reply.arguments()[0]
    if isinstance(version, QDBusVariant):
        version = version.variant()
    if int(version) < 3:
        raise OSError("The desktop folder picker requires FileChooser portal version 3 or later")

    token = "openfocus_" + uuid.uuid4().hex
    sender = bus.baseService().lstrip(":").replace(".", "_")
    request_path = f"/org/freedesktop/portal/desktop/request/{sender}/{token}"
    loop = QEventLoop()

    class Response(QObject):
        def __init__(self):
            super().__init__()
            self.value = None

        @pyqtSlot("uint", "QVariantMap")
        def receive(self, code, results):
            self.value = (code, results)
            loop.quit()

    response = Response()
    if not bus.connect(service, request_path, request_interface, "Response", response.receive):
        raise OSError("Cannot subscribe to the desktop folder picker response")
    parent_window = parent.window() if parent is not None else None
    was_enabled = parent_window.isEnabled() if parent_window is not None else False
    try:
        options = {"handle_token": QDBusVariant(token), "directory": QDBusVariant(True),
                   "multiple": QDBusVariant(multiple), "modal": QDBusVariant(True)}
        if directory and os.path.isdir(directory):
            options["current_folder"] = QDBusVariant(QByteArray(os.fsencode(os.path.abspath(directory)) + b"\0"))
        window_id = ""
        if parent_window is not None and QApplication.platformName().lower() == "xcb":
            window_id = f"x11:{int(parent_window.winId()):x}"
        message = QDBusMessage.createMethodCall(service, portal_path, chooser, "OpenFile")
        message.setArguments([window_id, caption, options])
        if was_enabled:
            parent_window.setEnabled(False)
        reply = bus.call(message, QDBus.CallMode.BlockWithGui)
        if reply.type() == QDBusMessage.MessageType.ErrorMessage:
            raise OSError("The desktop folder picker failed: " + reply.errorMessage())
        handles = reply.arguments()
        if not handles or handles[0].path() != request_path:
            raise OSError("The desktop folder picker returned an unexpected request handle")
        if response.value is None:
            loop.exec()
        code, results = response.value
        if code == 1:
            return []
        if code != 0:
            raise OSError(f"The desktop folder picker failed with response {code}")
        uris = results.get("uris", [])
        if isinstance(uris, QDBusVariant):
            uris = uris.variant()
        paths = []
        for uri in uris:
            url = QUrl(uri)
            if not url.isLocalFile() or not url.toLocalFile():
                raise OSError("The desktop folder picker returned a non-local folder")
            paths.append(url.toLocalFile())
        if not paths or (not multiple and len(paths) != 1):
            raise OSError("The desktop folder picker returned an invalid selection")
        return paths
    finally:
        if was_enabled:
            parent_window.setEnabled(True)
        bus.disconnect(service, request_path, request_interface, "Response", response.receive)


def _native_folder_paths(parent, caption, directory, multiple):
    if sys.platform == "win32":
        return _windows_folder_paths(parent, caption, directory, multiple)
    if sys.platform == "darwin":
        return _macos_folder_paths(parent, caption, directory, multiple)
    if sys.platform.startswith("linux"):
        return _portal_folder_paths(parent, caption, directory, multiple)
    raise OSError(f"No native folder picker is available on {sys.platform}")


class _NativeFolderDialog:
    """Minimal exec/selectedFiles interface shared by folder consumers."""

    def __init__(self, parent, caption, directory, multiple):
        self._parent = parent
        self._caption = caption
        self._directory = directory
        self._multiple = multiple
        self._files = []

    def exec(self):
        self._files = []
        try:
            self._files = _native_folder_paths(self._parent, self._caption,
                                               self._directory, self._multiple)
        except OSError as exc:
            QMessageBox.critical(self._parent, self._caption or "Select Folder", str(exc))
            return QFileDialog.DialogCode.Rejected
        if self._files:
            _remember_dialog_directory(self._files[-1])
            return QFileDialog.DialogCode.Accepted
        return QFileDialog.DialogCode.Rejected

    def selectedFiles(self):
        return list(self._files)



def _use_native_file_dialog():
    return not _control_picker.get()


def _pick_native_file(parent, caption, directory, file_filter, selected_filter,
                      options, mode, accept):
    """Use Qt's platform dialog, reporting an unavailable native helper."""
    from PyQt6.QtCore import QEventLoop

    directory = os.fspath(directory) if directory is not None else ""
    dialog = QFileDialog(parent, caption, directory, file_filter)
    try:
        dialog.setOptions(options & ~QFileDialog.Option.DontUseNativeDialog)
        dialog.setFileMode(mode)
        dialog.setAcceptMode(accept)
        if selected_filter:
            dialog.selectNameFilter(selected_filter)
        dialog.directoryEntered.connect(_remember_dialog_directory)

        # Qt creates its widget layout only when no platform helper exists.
        # Check again after showing: a helper can decline to open at runtime.
        if dialog.layout() is not None:
            QMessageBox.critical(parent, caption or "Select File",
                                 "The system file picker is unavailable. "
                                 "Install or enable the desktop's Qt platform integration.")
            return [], ""
        loop = QEventLoop()
        result = []

        def finished(code):
            result.append(code)
            loop.quit()

        dialog.finished.connect(finished)
        dialog.open()
        if dialog.layout() is not None:
            dialog.reject()
            QMessageBox.critical(parent, caption or "Select File",
                                 "The system file picker could not be opened.")
            return [], ""
        if not result:
            loop.exec()
        if result[0] == QFileDialog.DialogCode.Accepted:
            files = dialog.selectedFiles()
            if files:
                _remember_dialog_directory(os.path.dirname(files[0]))
            return files, dialog.selectedNameFilter()
        return [], ""
    finally:
        dialog.deleteLater()


class FileDialog(QFileDialog):
    """Shared Qt picker. Static helpers remain injectable at the application API."""

    def __init__(self, parent=None, caption="", directory="", filter=""):
        super().__init__(parent, caption, directory, filter)
        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        self.setSidebarUrls(_sidebar_urls(self))
        self.directoryEntered.connect(_remember_dialog_directory)
        self._remember_directory_changes = True
        make_folder_path_editable(self)
        file_types = self.findChild(QComboBox, "fileTypeCombo")
        if file_types is not None:
            file_types.setMinimumContentsLength(18)
            file_types.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            file_types.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def setDirectory(self, directory):
        super().setDirectory(directory)
        if getattr(self, "_remember_directory_changes", False):
            _remember_dialog_directory(self.directory().absolutePath())
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
        if _use_native_folder_dialog():
            return _NativeFolderDialog(parent, caption, directory, multiple)
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
        if mode != cls.FileMode.Directory and _use_native_file_dialog():
            return _pick_native_file(parent, caption, directory, filter, selectedFilter,
                                     options, mode, accept)
        dialog = cls(parent, caption, directory, filter)
        dialog.setOptions(options | cls.Option.DontUseNativeDialog)
        dialog.setFileMode(mode)
        dialog.setAcceptMode(accept)
        if selectedFilter:
            dialog.selectNameFilter(selectedFilter)
        screen = parent.screen() if parent is not None else dialog.screen()
        if screen is not None:
            available_width = screen.availableGeometry().width()
            if dialog.width() > available_width * 0.9:
                # Qt may restore a width saved when filter text stretched the
                # dialog across the screen. Keep the dialog resizable.
                dialog.resize(min(900, int(available_width * 0.8)), dialog.height())
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
        if not _use_native_folder_dialog():
            files, _ = cls._pick(parent, caption, directory, "", "", options,
                                 cls.FileMode.Directory, cls.AcceptMode.AcceptOpen)
            return files[0] if files else ""
        dialog = cls.for_directories(parent, caption, directory)
        return dialog.selectedFiles()[0] if dialog.exec() == cls.DialogCode.Accepted else ""
