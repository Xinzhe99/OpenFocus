"""Client side of the GUI control channel.

PyQt is imported lazily: the headless CLI must still import without Qt, and
only the ``ctl``/``gui`` commands need this module.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from core.control.protocol import (
    ControlError, MAX_LINE, decode, discovery_path, encode,
    read_discovery, request_payload,
)


_QT_APP = None

# Arguments that name a filesystem target. They are resolved on the caller's
# side: the GUI's working directory is unrelated to the shell that sent the
# request.
_PATH_ARG_KEYS = ("path", "folder", "video", "file", "output", "output_dir")
_PATH_LIST_KEYS = ("files", "folders")


def _resolve_paths(args, cmd=None):
    if not isinstance(args, dict):
        return args

    def absolute(value):
        if not isinstance(value, str) or not value:
            return value
        if "://" in value:
            return value
        return os.path.abspath(os.path.expanduser(value))

    resolved = dict(args)
    widget_path = isinstance(cmd, str) and cmd.startswith("ui.")
    for key in _PATH_ARG_KEYS:
        if key in resolved:
            if key == "path" and widget_path:
                # ui.* commands address a widget ("main/btn_render"), not a
                # file; making it absolute destroyed the path.
                continue
            resolved[key] = absolute(resolved[key])
    for key in _PATH_LIST_KEYS:
        value = resolved.get(key)
        if isinstance(value, list):
            resolved[key] = [absolute(item) for item in value]
    return resolved


def _qt_core():
    """Keep a created QCoreApplication alive for the process lifetime."""
    global _QT_APP
    from PyQt6.QtCore import QCoreApplication
    instance = QCoreApplication.instance()
    if instance is None:
        _QT_APP = QCoreApplication([])
        instance = _QT_APP
    return instance


def request(cmd, args=None, *, timeout=60.0, discovery=None, token=None):
    """Send one command and return its parsed response."""
    try:
        from PyQt6.QtNetwork import QLocalSocket
    except ImportError as exc:  # headless wheel without the Qt runtime
        raise _no_pyqt_error(exc) from None

    discovery = discovery or read_discovery()
    _qt_core()
    socket = QLocalSocket()
    socket.setReadBufferSize(MAX_LINE + 1)
    socket.connectToServer(discovery["socket"])
    if not socket.waitForConnected(int(timeout * 1000)):
        raise ControlError(
            "not_running",
            f"Could not connect to {discovery['socket']}: {socket.errorString()}",
            "The GUI may have exited; launch it with `openfocus gui`.")
    request_id = f"{os.getpid()}-{time.monotonic_ns()}"
    payload = request_payload(cmd, _resolve_paths(args, cmd),
                              token=token or discovery.get("token"),
                              request_id=request_id)
    try:
        if socket.write(encode(payload)) < 0 or (
                socket.bytesToWrite() and not socket.waitForBytesWritten(int(timeout * 1000))):
            raise ControlError("internal", "Could not send the request")
        buffer = b""
        deadline = time.monotonic() + timeout
        while True:
            if b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if len(line) > MAX_LINE:
                    raise ControlError("internal", "Response is too large")
                if not line.strip():
                    continue
                response = decode(line)
                # A late reply to an earlier, abandoned request must not be
                # mistaken for this command's result.
                if str(response.get("id")) != request_id:
                    continue
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ControlError(
                    "internal", f"Timed out waiting for {cmd}",
                    "Increase --timeout; a modal dialog may be waiting for input.")
            if (not socket.bytesAvailable()
                    and not socket.waitForReadyRead(int(remaining * 1000) + 1)
                    and not socket.bytesAvailable()):
                if socket.state() == QLocalSocket.LocalSocketState.UnconnectedState:
                    raise ControlError("internal", "The GUI closed the connection")
                continue
            buffer += bytes(socket.read(MAX_LINE + 1 - len(buffer)))
            if len(buffer) > MAX_LINE and b"\n" not in buffer:
                raise ControlError("internal", "Response is too large")
    finally:
        socket.abort()


def require_ok(response):
    if not response.get("ok"):
        error = response.get("error") or {}
        raise ControlError(error.get("code", "internal"),
                           error.get("message", "Command failed"),
                           error.get("hint"), error.get("details"))
    return response.get("result")


def wait_for(cmd, predicate, *, timeout=1800.0, interval=0.25,
             discovery=None, should_stop=None):
    """Poll one command until predicate(result) is true."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            break
        try:
            last = require_ok(request(cmd, {}, timeout=30, discovery=discovery))
        except ControlError as error:
            if error.code == "not_running":
                raise
            last = {"error": error.as_dict()}
        if predicate(last):
            return last
        time.sleep(interval)
    raise ControlError("internal", f"Timed out waiting for {cmd}",
                       f"Last state: {json.dumps(last, ensure_ascii=False)[:500]}")


def _no_pyqt_error(exc):
    return ControlError(
        "unsupported",
        f"The control client needs PyQt6, which is unavailable ({exc})",
        "pip install PyQt6, or use the desktop build where the client "
        "is bundled.")


def _launch_command(socket_name=None, paths=()):
    if getattr(sys, "frozen", False):
        command = [sys.executable]
    else:
        entry = Path(__file__).resolve().parents[2] / "main.py"
        if not entry.is_file():
            raise ControlError(
                "unsupported",
                "This install does not ship the desktop app (no main.py)",
                "The pip wheel is headless-only: install the desktop build "
                "or run from a source checkout.")
        command = [sys.executable, str(entry)]
    # Control starts automatically with the desktop application.
    if socket_name:
        command.extend(["--control-socket", socket_name])
    command.extend(os.path.abspath(os.path.expanduser(str(path))) for path in paths)
    return command


def launch_gui(socket_name=None, timeout=45.0, paths=(), *, env=None):
    """Start a detached GUI and wait for its control discovery file."""
    before = None
    child_env = os.environ.copy() if env is None else env.copy()
    override = child_env.get("OPENFOCUS_CONTROL_FILE")
    path = Path(override).expanduser().absolute() if override else discovery_path()
    try:
        before = path.stat().st_mtime_ns
    except OSError:
        before = None
    creation = 0
    if sys.platform == "win32":
        creation = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_GROUP
    if override:
        child_env["OPENFOCUS_CONTROL_FILE"] = str(path)
    process = subprocess.Popen(
        _launch_command(socket_name, paths), cwd=str(Path(__file__).resolve().parents[2]), env=child_env,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=creation,
        start_new_session=sys.platform != "win32",
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None and process.returncode not in (0, None):
            raise ControlError(
                "internal", f"OpenFocus exited during startup ({process.returncode})",
                "Start it manually with `openfocus gui` to see the error.")
        try:
            payload = read_discovery(path)
            matching = socket_name is None or payload["socket"] == socket_name
            fresh = before is None or path.stat().st_mtime_ns != before
            if matching and fresh:
                return payload
            # A second GUI process forwards paths to the existing instance and
            # exits without rewriting its discovery file.
            if matching and process.poll() == 0:
                try:
                    if require_ok(request("ping", discovery=payload, timeout=2)).get("pong"):
                        return payload
                except ControlError:
                    pass
        except (ControlError, OSError):
            pass
        time.sleep(0.25)
    raise ControlError(
        "internal", "OpenFocus did not open the control channel in time",
        "Start it with `openfocus gui` and retry.")

class Session:
    """One persistent local connection for a sequence of commands.

    A new QLocalSocket per request works for one-shot CLI calls, but a long
    agent session creates hundreds of short-lived connections. Reusing one
    connection avoids Windows named-pipe accept churn and keeps responses
    ordered while dialogs are open.
    """


    def __init__(self, *, discovery=None, timeout=60.0):
        try:
            from PyQt6.QtNetwork import QLocalSocket
        except ImportError as exc:  # headless wheel without the Qt runtime
            raise _no_pyqt_error(exc) from None

        if not isinstance(discovery, dict):
            discovery = read_discovery()
        self.discovery = discovery
        self.timeout = timeout
        self._sequence = 0
        _qt_core()
        self.socket = QLocalSocket()
        self.socket.setReadBufferSize(MAX_LINE + 1)
        self.socket.connectToServer(self.discovery["socket"])
        if not self.socket.waitForConnected(int(timeout * 1000)):
            raise ControlError(
                "not_running",
                f"Could not connect to {self.discovery['socket']}: "
                f"{self.socket.errorString()}",
                "The GUI may have exited; launch it with `openfocus gui`.")
        self.buffer = b""

    def _ensure_connected(self, timeout):
        from PyQt6.QtNetwork import QLocalSocket

        if self.socket.state() == QLocalSocket.LocalSocketState.ConnectedState:
            return
        # A timed-out request may still be queued on the pipe; start clean so
        # the next command cannot read the previous reply.
        self.socket.abort()
        self.buffer = b""
        self.socket.connectToServer(self.discovery["socket"])
        if not self.socket.waitForConnected(int(timeout * 1000)):
            raise ControlError(
                "not_running",
                f"Could not reconnect to {self.discovery['socket']}: "
                f"{self.socket.errorString()}",
                "The GUI may have exited; launch it with `openfocus gui`.")

    def _reset(self):
        self.buffer = b""
        self.socket.abort()

    def request(self, cmd, args=None, *, timeout=None):
        from PyQt6.QtNetwork import QLocalSocket

        timeout = self.timeout if timeout is None else timeout
        self._ensure_connected(timeout)
        self._sequence += 1
        request_id = f"{os.getpid()}-{self._sequence}"
        payload = request_payload(
            cmd, _resolve_paths(args, cmd), token=self.discovery.get("token"),
            request_id=request_id)
        if self.socket.write(encode(payload)) < 0 or (
                self.socket.bytesToWrite() and not self.socket.waitForBytesWritten(int(timeout * 1000))):
            self._reset()
            raise ControlError("internal", f"Could not send {cmd}")
        deadline = time.monotonic() + timeout
        while True:
            if b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if len(line) > MAX_LINE:
                    self._reset()
                    raise ControlError("internal", "Response is too large")
                if not line.strip():
                    continue
                response = decode(line)
                if str(response.get("id")) != request_id:
                    continue  # reply to an abandoned request
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._reset()
                raise ControlError(
                    "internal", f"Timed out waiting for {cmd}",
                    "Check ui.context; the GUI may be in a modal dialog. "
                    "The socket was reset and the next request reconnects.")
            if (not self.socket.bytesAvailable()
                    and not self.socket.waitForReadyRead(int(remaining * 1000) + 1)
                    and not self.socket.bytesAvailable()):
                if self.socket.state() == QLocalSocket.LocalSocketState.UnconnectedState:
                    self.buffer = b""
                    raise ControlError("internal", "The GUI closed the connection")
                continue
            self.buffer += bytes(self.socket.read(MAX_LINE + 1 - len(self.buffer)))
            if len(self.buffer) > MAX_LINE and b"\n" not in self.buffer:
                self._reset()
                raise ControlError("internal", "Response is too large")

    def call(self, cmd, args=None, *, timeout=None):
        return require_ok(self.request(cmd, args, timeout=timeout))

    def close(self):
        self.socket.abort()
        self.socket.deleteLater()


def tool_session(timeout=60.0, launch=False, allow_existing=False,
                 launch_timeout=60.0):
    """Session for tool runs (smoke, sweep).

    These tools clear stacks, answer prompts and trigger renders, so they
    must never land on a user's live session by accident: attaching
    requires ``allow_existing``, and ``launch`` starts a private instance
    with its own discovery file and socket that cannot see the user's.
    """
    import tempfile
    if launch:
        from PyQt6.QtCore import QSettings

        root = Path(tempfile.mkdtemp(prefix="openfocus-tool-"))
        settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)
        settings.setValue("ui/first_run_done", "1")
        settings.setValue("updates/last_check_epoch", 9999999999)
        settings.setValue("ui/ask_load_scale", False)
        settings.setValue("rendering/use_gpu", False)
        settings.setValue("rendering/ai_cpu_warning_dismissed", True)
        settings.sync()
        if settings.status() != QSettings.Status.NoError:
            raise ControlError("internal", "Could not initialize private settings")
        env = os.environ.copy()
        env["OPENFOCUS_DATA_DIR"] = str(root)
        env["OPENFOCUS_CONTROL_FILE"] = str(root / "control.json")
        env["OPENFOCUS_INSTANCE_NAME"] = f"OpenFocus-tool-instance-{root.name}"
        env.pop("OPENFOCUS_LOG_DIR", None)
        discovery = launch_gui(f"OpenFocus-tool-control-{root.name}",
                               timeout=launch_timeout, env=env)
        return Session(discovery=discovery, timeout=timeout)
    if not allow_existing and discovery_path().exists():
        raise ControlError(
            "confirm_required", "A GUI control session is already running",
            "This tool is destructive; pass allow_existing=True to use the "
            "live session anyway, or launch=True for a private instance.")
    return Session(timeout=timeout)
