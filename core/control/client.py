"""Client side of the GUI control channel.

PyQt is imported lazily: the headless CLI must still import without Qt, and
only the ``ctl``/``gui`` commands need this module.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

from core.control.protocol import (
    ControlError, MAX_LINE, decode, discovery_path, encode,
    read_discovery, request_payload,
)


_QT_APP = None


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
    from PyQt6.QtNetwork import QLocalSocket

    discovery = discovery or read_discovery()
    _qt_core()
    socket = QLocalSocket()
    socket.connectToServer(discovery["socket"])
    if not socket.waitForConnected(int(timeout * 1000)):
        raise ControlError(
            "not_running",
            f"Could not connect to {discovery['socket']}: {socket.errorString()}",
            "The GUI may have exited; launch it with `openfocus gui --control`.")
    payload = request_payload(cmd, args, token=token or discovery.get("token"),
                              request_id=str(int(time.time() * 1000)))
    try:
        socket.write(encode(payload))
        if not socket.waitForBytesWritten(int(timeout * 1000)):
            raise ControlError("internal", "Could not send the request")
        buffer = b""
        deadline = time.monotonic() + timeout
        while b"\n" not in buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ControlError(
                    "internal", f"Timed out waiting for {cmd}",
                    "Increase --timeout; a modal dialog may be waiting for input.")
            if not socket.waitForReadyRead(int(remaining * 1000) + 1):
                continue
            buffer += bytes(socket.readAll())
            if len(buffer) > MAX_LINE * 4:
                raise ControlError("internal", "Response is too large")
    finally:
        socket.disconnectFromServer()
    line = buffer.split(b"\n", 1)[0]
    if not line:
        raise ControlError("internal", "The GUI closed the connection")
    return decode(line)


def require_ok(response):
    if not response.get("ok"):
        error = response.get("error") or {}
        raise ControlError(error.get("code", "internal"),
                           error.get("message", "Command failed"),
                           error.get("hint"))
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


def _launch_command(socket_name=None, paths=()):
    if getattr(sys, "frozen", False):
        command = [sys.executable]
    else:
        entry = Path(__file__).resolve().parents[2] / "main.py"
        command = [sys.executable, str(entry)]
    command.append("--control")
    if socket_name:
        command.extend(["--control-socket", socket_name])
    command.extend(str(path) for path in paths)
    return command


def launch_gui(socket_name=None, timeout=45.0, paths=()):
    """Start a detached GUI with control enabled and wait for discovery."""
    before = None
    path = discovery_path()
    try:
        before = path.stat().st_mtime_ns
    except OSError:
        before = None
    creation = 0
    if sys.platform == "win32":
        creation = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_GROUP
    process = subprocess.Popen(
        _launch_command(socket_name, paths), cwd=str(Path(__file__).resolve().parents[2]),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=creation,
        start_new_session=sys.platform != "win32",
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None and process.returncode not in (0, None):
            raise ControlError(
                "internal", f"OpenFocus exited during startup ({process.returncode})",
                "Start it manually with `openfocus gui --control` to see the error.")
        try:
            payload = read_discovery()
            fresh = before is None or path.stat().st_mtime_ns != before
            if fresh and (socket_name is None or payload["socket"] == socket_name):
                return payload
        except ControlError:
            pass
        time.sleep(0.25)
    raise ControlError(
        "internal", "OpenFocus did not open the control channel in time",
        "Start it with `openfocus gui --control` and retry.")
