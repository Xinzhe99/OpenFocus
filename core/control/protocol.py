"""Wire format, discovery file and errors for the GUI control channel.

Standard library only: the headless CLI imports this module without PyQt.
"""
import json
import os
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
    "dialog.reject",
    "batch.state", "batch.cancel",
    "ui.actions", "ui.trigger", "ui.tree", "ui.find", "ui.click", "ui.set",
    "ui.grab",
)

ERROR_CODES = (
    "bad_request", "unauthorized", "not_found", "invalid_argument", "busy",
    "unsupported", "internal", "not_running",
)


class ControlError(Exception):
    def __init__(self, code, message, hint=None):
        super().__init__(message)
        if code not in ERROR_CODES:
            code = "internal"
        self.code = code
        self.message = str(message)
        self.hint = hint

    def as_dict(self):
        payload = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
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
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, path)
    return path


def read_discovery():
    """Parsed discovery file, or ControlError('not_running')."""
    path = discovery_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ControlError(
            "not_running",
            "OpenFocus is not running with the control channel enabled.",
            "Launch it with `openfocus gui --control` or add --launch.")
    except (OSError, ValueError) as exc:
        raise ControlError("not_running", f"Unreadable control file: {exc}",
                           "Launch the GUI with --control to recreate it.")
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ControlError("not_running", "Control file has an unknown schema.",
                           "Start the GUI from this version with --control.")
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
