"""Wire format, discovery file and errors for the GUI control channel.

Standard library only: the headless CLI imports this module without PyQt.
"""
import json
import os
import secrets
from pathlib import Path
import time

SCHEMA = "openfocus/control/1"
SOCKET_PREFIX = "OpenFocus-control"
DISCOVERY_FILE = "control.json"
MAX_LINE = 1024 * 1024

COMMANDS = (
    "ping", "app.info", "app.state", "app.quit",
    "stack.load", "stack.frames", "stack.clear", "stack.include",
    "method.set", "method.parameters", "align.set", "option.set",
    "render.start", "render.cancel", "render.state", "render.result",
    "output.list", "output.select", "output.display", "output.export",
    "output.delete",
    "project.save", "project.open",
    "dialog.open", "dialog.list", "dialog.close", "dialog.accept",
    "dialog.reject", "dialog.file", "dialog.color",
    "batch.state", "batch.cancel",
    "ui.actions", "ui.trigger", "ui.tree", "ui.find", "ui.click", "ui.set",
    "ui.grab", "ui.context", "ui.focus", "ui.mouse", "ui.key", "ui.type",
    "ui.menu",
)

ERROR_CODES = (
    "bad_request", "unauthorized", "not_found", "invalid_argument", "busy",
    "confirm_required", "insufficient_storage", "unsupported", "internal", "not_running",
)


class ControlError(Exception):
    def __init__(self, code, message, hint=None, details=None):
        super().__init__(message)
        if code not in ERROR_CODES:
            code = "internal"
        self.code = code
        self.message = str(message)
        self.hint = hint
        self.details = details

    def as_dict(self):
        payload = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload


def discovery_path() -> Path:
    """Per-user file naming the live socket and its token.

    ``OPENFOCUS_CONTROL_FILE`` isolates tests and parallel sessions.
    """
    override = os.environ.get("OPENFOCUS_CONTROL_FILE")
    if override:
        return Path(override).expanduser()
    from utils.app_paths import user_data_dir
    return Path(user_data_dir()) / DISCOVERY_FILE


def _private_file(path):
    """Create a new file with access limited to the current user."""
    if os.name != "nt":
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel.LocalFree.restype = wintypes.HLOCAL
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                       ctypes.POINTER(wintypes.HANDLE)]
    advapi.OpenProcessToken.restype = wintypes.BOOL
    advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                           wintypes.LPVOID, wintypes.DWORD,
                                           ctypes.POINTER(wintypes.DWORD)]
    advapi.GetTokenInformation.restype = wintypes.BOOL
    advapi.ConvertSidToStringSidW.argtypes = [wintypes.LPVOID,
                                              ctypes.POINTER(wintypes.LPWSTR)]
    advapi.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.ULONG)]
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("descriptor", wintypes.LPVOID),
                    ("inherit", wintypes.BOOL)]

    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   ctypes.POINTER(SecurityAttributes), wintypes.DWORD,
                                   wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    token = wintypes.HANDLE()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())
    sid_string = wintypes.LPWSTR()
    descriptor = wintypes.LPVOID()
    try:
        size = wintypes.DWORD()
        advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        if not size.value:
            raise ctypes.WinError(ctypes.get_last_error())
        info = ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(token, 1, info, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        # TOKEN_USER starts with SID_AND_ATTRIBUTES, whose first member is PSID.
        sid = ctypes.cast(info, ctypes.POINTER(wintypes.LPVOID)).contents
        if not advapi.ConvertSidToStringSidW(sid, ctypes.byref(sid_string)):
            raise ctypes.WinError(ctypes.get_last_error())
        sddl = f"D:P(A;;FA;;;{sid_string.value})"
        if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl, 1, ctypes.byref(descriptor), None):
            raise ctypes.WinError(ctypes.get_last_error())
        attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
        handle = kernel.CreateFileW(str(path), 0x40000000, 0, ctypes.byref(attributes),
                                    1, 0x80, None)  # GENERIC_WRITE, CREATE_NEW
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
        except BaseException:
            kernel.CloseHandle(handle)
            raise
    finally:
        if descriptor:
            kernel.LocalFree(descriptor)
        if sid_string:
            kernel.LocalFree(ctypes.cast(sid_string, wintypes.HLOCAL))
        kernel.CloseHandle(token)


def write_discovery(socket_name, token, pid, version):
    path = discovery_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "version": version,
        "pid": int(pid),
        "socket": socket_name,
        "token": token,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    descriptor = _private_file(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def read_discovery(path=None):
    """Parsed discovery file, or ControlError('not_running')."""
    path = Path(path) if path is not None else discovery_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ControlError(
            "not_running",
            "OpenFocus is not running or its control channel is unavailable.",
            "Launch it with `openfocus gui` or add --launch.")
    except (OSError, ValueError) as exc:
        raise ControlError("not_running", f"Unreadable control file: {exc}",
                           "Launch the GUI to recreate it.")
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ControlError("not_running", "Control file has an unknown schema.",
                           "Start the GUI from this version.")
    for key in ("socket", "token", "pid"):
        if key not in payload:
            raise ControlError("not_running", f"Control file lacks {key}.")
    return payload


def clear_discovery(pid=None):
    path = discovery_path()
    try:
        if pid is not None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("pid") != int(pid):
                return
        path.unlink()
    except (OSError, ValueError):
        pass


def request_payload(cmd, args=None, token=None, request_id="1"):
    payload = {"id": str(request_id), "cmd": cmd, "args": args or {}}
    if token is not None:
        payload["token"] = token
    return payload


def encode(payload):
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(line.encode("utf-8")) > MAX_LINE:
        raise ControlError("bad_request", "Request is too large")
    return (line + "\n").encode("utf-8")


def decode(line):
    try:
        payload = json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ControlError("bad_request", f"Malformed JSON: {exc}")
    if not isinstance(payload, dict):
        raise ControlError("bad_request", "Request must be a JSON object")
    return payload


def ok_response(request_id, result, elapsed_ms=0.0):
    return {"id": str(request_id), "ok": True, "result": result,
            "elapsed_ms": round(float(elapsed_ms), 3)}


def error_response(request_id, error):
    if isinstance(error, ControlError):
        payload = error.as_dict()
    else:
        payload = {"code": "internal", "message": str(error)}
    return {"id": str(request_id), "ok": False, "error": payload}
