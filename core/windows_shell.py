"""Windows Explorer's out-of-process IDropTarget verb.

The shell supplies one IDataObject for the entire selection.  A temporary
request file carries the selection to the GUI without the CreateProcess command
line length limit; the GUI removes it after reading it.
"""

import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid


CLSID = "{08A1D5C2-DB59-4A62-AC2B-ED9E960A986E}"


def is_request_file(path):
    """Only consume request files created in this user's temporary directory."""
    if not isinstance(path, str):
        return False
    path = os.path.abspath(path)
    name = os.path.basename(path)
    return (os.path.normcase(os.path.dirname(path)) == os.path.normcase(os.path.abspath(tempfile.gettempdir()))
            and name.startswith("openfocus-shell-") and name.endswith(".json")
            and os.path.isfile(path) and not os.path.islink(path))


def run_drop_target():
    """Serve one Explorer activation, then exit."""
    ole = ctypes.OleDLL("ole32")
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    user = ctypes.WinDLL("user32", use_last_error=True)
    callback = ctypes.WINFUNCTYPE
    hresult = ctypes.c_long
    ptr = ctypes.c_void_p
    ulong = wintypes.ULONG

    class GUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]

        @classmethod
        def parse(cls, text):
            return cls((ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(text).bytes_le))

    class POINTL(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

    class FORMATETC(ctypes.Structure):
        _fields_ = [("cfFormat", wintypes.WORD), ("ptd", ptr),
                    ("dwAspect", wintypes.DWORD), ("lindex", wintypes.LONG),
                    ("tymed", wintypes.DWORD)]

    class STGMEDIUM(ctypes.Structure):
        _fields_ = [("tymed", wintypes.DWORD), ("handle", ptr), ("release", ptr)]

    ole.CoInitializeEx.argtypes = [ptr, wintypes.DWORD]
    ole.CoUninitialize.restype = None
    ole.CoRegisterClassObject.argtypes = [ctypes.POINTER(GUID), ptr, wintypes.DWORD,
                                          wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    ole.CoRevokeClassObject.argtypes = [wintypes.DWORD]
    ole.ReleaseStgMedium.argtypes = [ctypes.POINTER(STGMEDIUM)]
    ole.ReleaseStgMedium.restype = None
    shell.DragQueryFileW.argtypes = [ptr, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    shell.DragQueryFileW.restype = wintypes.UINT
    user.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), ptr, wintypes.UINT,
                                  wintypes.UINT, wintypes.UINT]
    user.MessageBoxW.argtypes = [ptr, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]

    iid_unknown = uuid.UUID("00000000-0000-0000-C000-000000000046").bytes_le
    iid_factory = uuid.UUID("00000001-0000-0000-C000-000000000046").bytes_le
    iid_drop = uuid.UUID("00000122-0000-0000-C000-000000000046").bytes_le
    e_nointerface = ctypes.c_long(0x80004002).value
    e_fail = ctypes.c_long(0x80004005).value
    class_noaggregation = ctypes.c_long(0x80040110).value
    finished = [False]
    references = [1, 1]

    def query(this, iid, result, allowed, obj, index):
        result[0] = None
        if bytes(iid.contents.data) not in allowed:
            return e_nointerface
        result[0] = ctypes.addressof(obj)
        references[index] += 1
        return 0

    Query = callback(hresult, ptr, ctypes.POINTER(GUID), ctypes.POINTER(ptr))
    Ref = callback(ulong, ptr)
    Create = callback(hresult, ptr, ptr, ctypes.POINTER(GUID), ctypes.POINTER(ptr))
    Lock = callback(hresult, ptr, wintypes.BOOL)
    DragEnter = callback(hresult, ptr, ptr, wintypes.DWORD, POINTL, ctypes.POINTER(wintypes.DWORD))
    DragOver = callback(hresult, ptr, wintypes.DWORD, POINTL, ctypes.POINTER(wintypes.DWORD))
    DragLeave = callback(hresult, ptr)
    Drop = callback(hresult, ptr, ptr, wintypes.DWORD, POINTL, ctypes.POINTER(wintypes.DWORD))

    class Object(ctypes.Structure):
        _fields_ = [("vtable", ctypes.POINTER(ptr))]

    def ref(index, delta):
        references[index] += delta
        return references[index]

    def enter(this, data, keys, point, effect):
        effect[0] &= 1  # DROPEFFECT_COPY
        return 0

    def over(this, keys, point, effect):
        effect[0] &= 1
        return 0

    def drop(this, data, keys, point, effect):
        request = None
        try:
            # IDataObject::GetData, CF_HDROP, DVASPECT_CONTENT, TYMED_HGLOBAL.
            vtable = ctypes.cast(data, ctypes.POINTER(ctypes.POINTER(ptr))).contents
            get_data = callback(hresult, ptr, ctypes.POINTER(FORMATETC),
                                ctypes.POINTER(STGMEDIUM))(vtable[3])
            fmt = FORMATETC(15, None, 1, -1, 1)
            medium = STGMEDIUM()
            if get_data(data, ctypes.byref(fmt), ctypes.byref(medium)) < 0:
                raise OSError("Explorer did not provide filesystem paths")
            try:
                count = shell.DragQueryFileW(medium.handle, 0xffffffff, None, 0)
                paths = []
                for index in range(count):
                    length = shell.DragQueryFileW(medium.handle, index, None, 0)
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    if not shell.DragQueryFileW(medium.handle, index, buffer, length + 1):
                        raise OSError("Could not read an Explorer selection")
                    paths.append(buffer.value)
            finally:
                ole.ReleaseStgMedium(ctypes.byref(medium))
            if not paths:
                raise OSError("Explorer supplied an empty selection")
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json",
                                             prefix="openfocus-shell-", delete=False) as stream:
                request = stream.name
                json.dump(paths, stream, ensure_ascii=False)
            executable = ([sys.executable] if getattr(sys, "frozen", False)
                          else [sys.executable, os.path.abspath(sys.argv[0])])
            environment = os.environ.copy()
            environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            subprocess.Popen(executable + ["--shell-import", request], close_fds=True,
                             env=environment)
            effect[0] &= 1
            return 0
        except Exception as exc:
            if request:
                try:
                    os.unlink(request)
                except OSError:
                    pass
            effect[0] = 0
            user.MessageBoxW(None, str(exc), "OpenFocus", 0x10)
            return e_fail
        finally:
            finished[0] = True

    drop_callbacks = [
        Query(lambda this, iid, result: query(this, iid, result, (iid_unknown, iid_drop), target, 1)),
        Ref(lambda this: ref(1, 1)), Ref(lambda this: ref(1, -1)),
        DragEnter(enter), DragOver(over), DragLeave(lambda this: 0), Drop(drop),
    ]
    drop_table = (ptr * len(drop_callbacks))(*(ctypes.cast(fn, ptr).value for fn in drop_callbacks))
    target = Object(drop_table)

    def create(this, outer, iid, result):
        if outer:
            result[0] = None
            return class_noaggregation
        return query(this, iid, result, (iid_unknown, iid_drop), target, 1)

    factory_callbacks = [
        Query(lambda this, iid, result: query(this, iid, result, (iid_unknown, iid_factory), factory, 0)),
        Ref(lambda this: ref(0, 1)), Ref(lambda this: ref(0, -1)),
        Create(create), Lock(lambda this, lock: 0),
    ]
    factory_table = (ptr * len(factory_callbacks))(*(ctypes.cast(fn, ptr).value for fn in factory_callbacks))
    factory = Object(factory_table)
    cookie = wintypes.DWORD()
    ole.CoInitializeEx(None, 2)  # Apartment-threaded: callbacks run on this thread.
    try:
        ole.CoRegisterClassObject(ctypes.byref(GUID.parse(CLSID)), ctypes.byref(factory),
                                  4, 0, ctypes.byref(cookie))  # LOCAL_SERVER, SINGLEUSE
        deadline = time.monotonic() + 60
        message = wintypes.MSG()
        while not finished[0] and time.monotonic() < deadline:
            while user.PeekMessageW(ctypes.byref(message), None, 0, 0, 1):
                user.TranslateMessage(ctypes.byref(message))
                user.DispatchMessageW(ctypes.byref(message))
            time.sleep(0.01)
    finally:
        if cookie.value:
            ole.CoRevokeClassObject(cookie)
        ole.CoUninitialize()
