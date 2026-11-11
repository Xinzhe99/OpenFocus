"""Local command server that drives the live OpenFocus window.

Every command runs on the GUI thread, so it uses the same controllers and
dialogs the UI uses. Long operations return immediately; clients poll state.
"""
import faulthandler
import json
import os
import secrets
import socket
import tempfile
import threading
import time

from PyQt6 import sip
from PyQt6.QtCore import QDir, QObject, Qt, QTimer
from PyQt6.QtNetwork import QLocalServer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QApplication, QColorDialog, QFileDialog,
                             QLineEdit, QMenu)

from constants import app_display_version
from core.control.color_dialogs import (install_non_native_color_dialogs,
                                        uninstall_non_native_color_dialogs)
from core.control.protocol import (
    COMMANDS as _COMMANDS, ControlError, MAX_LINE, SCHEMA, SOCKET_PREFIX,
    clear_discovery, decode, encode, error_response, ok_response,
    write_discovery,
)
from core.algorithms import (
    fusion_algorithms, get_fusion, get_registration, registration_algorithms,
    resolve_fusion_parameters, resolve_registration_parameters,
)
from locales import trans
from utils.logging_setup import get_logger

_log = get_logger(__name__)
MAX_MOUSE_STEPS = 1024

class _DeferredResponse:
    def __init__(self, start):
        self.start = start
        self.request_id = '0'
        self.started = 0.0




def _string(args, key, required=True, default=None, choices=None):
    value = args.get(key, default)
    if value is None:
        if required:
            raise ControlError("invalid_argument", f"{key} is required")
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise ControlError("invalid_argument", f"{key} must be a non-empty string")
    if choices is not None and value not in choices:
        raise ControlError("invalid_argument",
                           f"{key} must be one of {list(choices)}")
    return value


def _integer(args, key, required=False, default=None, minimum=None):
    value = args.get(key, default)
    if value is None:
        if required:
            raise ControlError("invalid_argument", f"{key} is required")
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or isinstance(value, float) and not value.is_integer()):
        raise ControlError("invalid_argument", f"{key} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ControlError("invalid_argument", f"{key} must be >= {minimum}")
    return value


def _number(args, key, required=False, default=None):
    value = args.get(key, default)
    if value is None:
        if required:
            raise ControlError("invalid_argument", f"{key} is required")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlError("invalid_argument", f"{key} must be a number")
    return float(value)


def _boolean(args, key, required=False, default=None):
    value = args.get(key, default)
    if value is None:
        if required:
            raise ControlError("invalid_argument", f"{key} is required")
        return None
    if not isinstance(value, bool):
        raise ControlError("invalid_argument", f"{key} must be a boolean")
    return value


def _dialog_candidate(widget):
    """A picker is answerable when it is visible or exec'ing modally.

    Some platforms (offscreen Linux in CI, native wrappers) report
    ``isVisible() == False`` for a dialog that is inside ``exec()``; the
    modal flag is the reliable signal there, and ``dialog.list`` already
    reports such windows.
    """
    return bool(widget.isVisible() or widget.isModal())


def _invoke(widget, method, *values):
    """Call a dialog slot without touching a SIP-restricted Python member.

    Dialogs constructed by Qt itself (the static ``QFileDialog`` helpers on
    some platforms) come with wrappers whose Python members raise
    "no access to protected functions or signals for objects not created
    from Python"; ``QMetaObject.invokeMethod`` still works on them.
    """
    from PyQt6.QtCore import Q_ARG, QMetaObject, Qt
    try:
        getattr(widget, method)(*values)
        return True
    except RuntimeError:
        pass
    if values:
        return QMetaObject.invokeMethod(
            widget, method, Qt.ConnectionType.DirectConnection,
            Q_ARG(str, values[0]))
    return QMetaObject.invokeMethod(
        widget, method, Qt.ConnectionType.DirectConnection)


def _spec_dict(spec):
    return {
        "key": spec.key,
        "label": spec.label_key,
        "default": spec.default,
        "minimum": spec.minimum,
        "maximum": spec.maximum,
        "step": spec.step,
        "decimals": spec.decimals,
        "choices": list(spec.choices),
    }


def _stack_info(window):
    images = list(getattr(window, "raw_images", None) or [])
    first = images[0] if images else None
    excluded = set(getattr(window, "excluded_frames", set()) or set())
    return {
        "count": len(images),
        "names": list(getattr(window, "image_filenames", None) or []),
        "included": [index for index in range(len(images)) if index not in excluded],
        "folder": getattr(window, "current_folder_path", "") or "",
        "scale": float(getattr(window, "current_scale_factor", 1.0) or 1.0),
        "shape": list(first.shape) if first is not None else None,
        "dtype": str(first.dtype) if first is not None else None,
    }


def _render_state(window):
    manager = getattr(window, "render_manager", None)
    worker = getattr(manager, "worker", None)
    busy = bool((worker is not None and worker.isRunning())
                or getattr(manager, '_pending_storage', None) is not None)
    error = getattr(manager, "last_error", None)
    last_state = getattr(manager, "last_state", None)
    if busy or last_state == "running":
        state = "rendering"
    elif last_state in ("done", "cancelled", "discarded", "error"):
        # A cancelled or stack-changed render is terminal but not a success;
        # a newer fusion_result must not turn it into "done".
        state = last_state
    elif error:
        state = "error"
    elif getattr(manager, "last_result", None) is not None:
        state = "done"
    else:
        state = "idle"
    progress = getattr(manager, "_control_progress", None)
    result = getattr(manager, "last_result", None)
    return {
        "state": state,
        "busy": busy,
        "progress": ({"done": int(progress[0]), "total": int(progress[1])}
                     if progress else None),
        "method": (getattr(manager, "_render_context", None) or {}).get("algorithm")
        or window.current_fusion_method(),
        "preview": bool((getattr(manager, "_render_context", None) or {}).get("preview")),
        "result": result,
        "error": error,
    }


def _view_state(window):
    def zoom(widget):
        return float(getattr(widget, "_zoom_factor", 1.0) or 1.0)

    source = getattr(window, "lbl_source_img", None)
    result = getattr(window, "lbl_result_img", None)
    slider = getattr(window, "stack_slider", None)
    result_slider = getattr(window, "result_slider", None)
    view_stack = getattr(window, "view_stack", None)

    def roi(widget):
        getter = getattr(widget, "get_roi_rect", None)
        if not callable(getter):
            return None
        rect = getter()
        return list(rect.getRect()) if rect is not None else None

    return {
        "source_index": int(slider.value()) if slider is not None else None,
        "result_index": (int(result_slider.value())
                         if result_slider is not None else None),
        "source_zoom": zoom(source),
        "result_zoom": zoom(result),
        "result_page": (int(view_stack.currentIndex())
                        if view_stack is not None else None),
        "wipe_active": bool(getattr(window, "wipe_active", False)),
        "roi_source": bool(getattr(source, "roi_mode", False)),
        "roi_result": bool(getattr(result, "roi_mode", False)),
        "roi_source_rect": roi(source),
        "roi_result_rect": roi(result),
    }


def _root_context(window):
    from ui.control_surface import root_context
    return root_context(window)


class ControlServer(QObject):
    def __init__(self, window, socket_name=None, token=None, parent=None):
        super().__init__(parent)
        self.window = window
        self.token = token or secrets.token_hex(16)
        self.socket_name = socket_name or f"{SOCKET_PREFIX}-{os.getpid()}"
        if os.name == 'posix' and not socket_name:
            temp_dir = QDir.tempPath()
            # macOS limits sockaddr_un paths to 103 bytes, including the
            # private staging directory used to publish the socket safely.
            stage_path = os.path.join(temp_dir, '.of-12345678', 's')
            endpoint = os.path.join(temp_dir, self.socket_name)
            if max(len(os.fsencode(stage_path)), len(os.fsencode(endpoint))) >= 104:
                self.socket_name = os.path.join('/tmp', f'ofc-{os.getpid()}')
        self._buffers = {}
        self._pending_replies = {}
        self._dialogs = {}
        # Generation counters let --wait bind to one request instead of any
        # stack that happens to have frames.
        self._load_id = 0
        self._load_baseline = 0
        self._load_error = None
        self._load_finished = True
        self._render_id = 0
        self._tracked_load_bound = None
        self._tracked_render_bound = None
        self._dialogs_patched = False
        self._closed = False
        self._watchdog_stop = None
        self._unix_stage = None
        self._unix_endpoint = None
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        if os.name == "posix":
            self._listen_unix()
        elif not self._server.listen(self.socket_name):
            QLocalServer.removeServer(self.socket_name)
            if not self._server.listen(self.socket_name):
                error = self._server.errorString()
                self._server.close()
                raise RuntimeError(f"Could not listen on {self.socket_name}: {error}")
        self._server.newConnection.connect(self._accept)
        try:
            self.discovery = write_discovery(
                self.socket_name, self.token, os.getpid(), app_display_version())
            self._install_tracking()
            self._dialogs_patched = install_non_native_color_dialogs()
            if os.environ.get("OPENFOCUS_CONTROL_WATCHDOG"):
                self._start_watchdog()
        except Exception:
            self.close()
            raise
        _log.info("control channel listening on %s", self.socket_name)

    def _listen_unix(self):
        # Qt's UserAccessOption can publish with rename(), overwriting an
        # existing path. A hard link publishes the private socket exclusively.
        path = self.socket_name
        if not path.startswith('/'):
            path = os.path.join(QDir.tempPath(), path)
        listener = None
        try:
            self._unix_stage = tempfile.TemporaryDirectory(
                prefix='.of-', dir=os.path.dirname(path))
            staged = os.path.join(self._unix_stage.name, 's')
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(staged)
            os.chmod(staged, 0o700)
            listener.listen(50)
            info = os.lstat(staged)
            os.link(staged, path)
            self._unix_endpoint = (path, info.st_dev, info.st_ino)
            if not self._server.listen(listener.fileno()):
                raise RuntimeError(self._server.errorString())
            listener.detach()  # QLocalServer owns the descriptor after adoption.
        except (OSError, ValueError, RuntimeError) as error:
            if listener is not None:
                listener.close()
            self._cleanup_unix_endpoint()
            if self._unix_stage is not None:
                self._unix_stage.cleanup()
                self._unix_stage = None
            raise RuntimeError(f"Could not listen on {self.socket_name}: {error}") from error

    def _cleanup_unix_endpoint(self):
        if self._unix_endpoint is None:
            return
        path, device, inode = self._unix_endpoint
        self._unix_endpoint = None
        try:
            info = os.lstat(path)
            if (info.st_dev, info.st_ino) == (device, inode):
                os.unlink(path)
        except FileNotFoundError:
            pass

    def _start_watchdog(self):
        """Opt-in: dump the GUI thread if it stops answering for 10 s."""
        self._last_tick = time.monotonic()
        self._watchdog_stop = threading.Event()
        timer = QTimer(self)
        timer.setInterval(500)
        timer.timeout.connect(
            lambda: setattr(self, "_last_tick", time.monotonic()))
        timer.start()
        self._watchdog_timer = timer
        path = os.path.join(os.environ.get("TEMP", "."), "openfocus_watchdog.txt")

        def watch():
            while not self._watchdog_stop.wait(2.0):
                lag = time.monotonic() - self._last_tick
                if lag < 10:
                    continue
                with open(path, "a", encoding="utf-8") as stream:
                    stream.write(f"\n=== GUI stalled {lag:.1f}s ===\n")
                    faulthandler.dump_traceback(file=stream, all_threads=True)
                self._last_tick = time.monotonic()

        self._watchdog_thread = threading.Thread(
            target=watch, name="control-watchdog", daemon=True)
        self._watchdog_thread.start()

    # ------------------------------------------------------ generation tracking
    def _install_tracking(self):
        """Expose loads and renders started outside the control commands.

        `--wait` must bind to one request, but a folder opened from the menu
        and a render started by the button used to leave the counters stale,
        so a wait could accept an earlier load or a cancelled render. Wrap the
        manager entry points once and advance the same generations for every
        caller.
        """
        manager = getattr(self.window, "source_manager", None)
        starter = getattr(manager, "_start_load_worker", None)
        if starter is not None:
            self._original_start_load = starter
            self._tracked_load_bound = self._tracked_start_load
            manager._start_load_worker = self._tracked_load_bound
        renderer = getattr(self.window, "render_manager", None)
        start_render = getattr(renderer, "start_render", None)
        if start_render is not None:
            self._original_start_render = start_render
            self._tracked_render_bound = self._tracked_start_render
            renderer.start_render = self._tracked_render_bound

    def _remove_tracking(self):
        try:
            manager = getattr(self.window, "source_manager", None)
        except RuntimeError:  # the window was already destroyed
            manager = None
        if (self._tracked_load_bound is not None
                and getattr(manager, "_start_load_worker", None)
                is self._tracked_load_bound):
            manager._start_load_worker = self._original_start_load
        try:
            renderer = getattr(self.window, "render_manager", None)
        except RuntimeError:
            renderer = None
        if (self._tracked_render_bound is not None
                and getattr(renderer, "start_render", None)
                is self._tracked_render_bound):
            renderer.start_render = self._original_start_render

    def _tracked_start_load(self, *args, **kwargs):
        load_id = self._begin_load()
        success = kwargs.get("on_success")
        failure = kwargs.get("on_error")

        def on_success():
            try:
                if success is not None:
                    success()
            finally:
                self._finish_load(load_id)

        def on_error(message):
            try:
                if failure is not None:
                    failure(message)
            finally:
                self._finish_load(load_id, message)

        kwargs["on_success"] = on_success
        kwargs["on_error"] = on_error
        try:
            self._original_start_load(*args, **kwargs)
        except Exception:
            self._finish_load(load_id, "Load failed to start")
            raise
        self._watch_load(load_id)
        return load_id

    def _watch_load(self, load_id):
        """Finish the generation even when no callback fires (superseded load)."""
        worker = getattr(self.window.source_manager, "_load_worker", None)
        if worker is None:
            self._finish_load(load_id, self._load_error or "Load did not start")
            return
        worker.finished_load.connect(
            lambda ok, message, images, names, lid=load_id:
            self._finish_load(lid, None if ok else (message or "Load failed")))
        worker.finished.connect(lambda lid=load_id: self._finish_load(lid))

    def _tracked_start_render(self, *args, **kwargs):
        status = self._original_start_render(*args, **kwargs)
        if status == "started":
            self._render_id += 1
        return status

    def _begin_load(self):
        """Open a new load generation; --wait must match this id."""
        self._load_id += 1
        self._load_finished = False
        self._load_error = None
        self._load_baseline = len(getattr(self.window, "raw_images", None) or [])
        return self._load_id

    def _finish_load(self, load_id, error=None) -> None:
        if load_id != self._load_id:
            return  # a newer request owns the state
        self._load_finished = True
        text = str(error or "").strip()
        if text:
            self._load_error = ("cancelled"
                                if text.upper().startswith("CANCELLED")
                                else text)

    def _load_info(self) -> dict:
        manager = self.window.source_manager
        active = (getattr(manager, "_load_worker", None) is not None
                  and not self._load_finished)
        return {
            "id": self._load_id,
            "active": active,
            "finished": bool(self._load_finished),
            "error": self._load_error if self._load_finished else None,
            "baseline_count": self._load_baseline,
        }

    # ------------------------------------------------------------- sockets
    def _accept(self):
        while self._server.hasPendingConnections():
            connection = self._server.nextPendingConnection()
            if not self._client_is_local(connection):
                connection.abort()
                connection.deleteLater()
                continue
            self._buffers[connection] = b""
            connection.setReadBufferSize(MAX_LINE + 1)
            connection.readyRead.connect(lambda c=connection: self._read(c))
            connection.disconnected.connect(lambda c=connection: self._drop(c))
            # Data can arrive before newConnection reaches this slot. Qt will
            # not emit readyRead again for bytes already buffered.
            if connection.bytesAvailable():
                self._read(connection)

    @staticmethod
    def _client_is_local(connection) -> bool:
        """Reject Windows pipe clients whose machine cannot be verified."""
        if os.name != "nt":
            return True
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_client = kernel32.GetNamedPipeClientComputerNameW
            get_client.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.ULONG]
            get_client.restype = wintypes.BOOL
            get_local = kernel32.GetComputerNameW
            get_local.argtypes = [wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
            get_local.restype = wintypes.BOOL
            client_name = ctypes.create_unicode_buffer(256)
            local_name = ctypes.create_unicode_buffer(256)
            local_size = wintypes.DWORD(len(local_name))
            handle = int(connection.socketDescriptor())
            if handle == -1:
                return False
            ctypes.set_last_error(0)
            if not get_client(wintypes.HANDLE(handle), client_name,
                              len(client_name)):
                # A local client is reported as ERROR_PIPE_LOCAL rather than
                # a computer name; that answer is itself proof of locality.
                if ctypes.get_last_error() == 229:  # ERROR_PIPE_LOCAL
                    return True
                _log.warning("control: cannot verify pipe client locality (error %s)",
                             ctypes.get_last_error())
                return False
            if not get_local(local_name, ctypes.byref(local_size)):
                _log.warning("control: cannot read the local computer name (error %s)",
                             ctypes.get_last_error())
                return False
            return client_name.value.lstrip("\\").casefold() == local_name.value.casefold()
        except Exception:  # noqa: BLE001 - fail closed when locality is unknown
            _log.exception("control: client locality check failed")
            return False

    def _drop(self, connection):
        worker = self._pending_replies.pop(connection, None)
        self._buffers.pop(connection, None)
        if worker is not None:
            worker.cancel()
        connection.deleteLater()

    def _read(self, connection):
        buffer = self._buffers.get(connection)
        if buffer is None:
            return
        if connection in self._pending_replies:
            return
        while b'\n' in buffer or connection.bytesAvailable():
            if b'\n' not in buffer:
                # Never copy an unbounded Qt read buffer into Python.
                chunk = bytes(connection.read(min(65536, MAX_LINE + 1 - len(buffer))))
                if not chunk:
                    break
                buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if len(line) > MAX_LINE:
                    self._reject_request(connection, "Request line is too large")
                    return
                if not line.strip():
                    continue
                self._buffers[connection] = buffer
                response = self.handle_line(line, token=self.token)
                if connection not in self._buffers:
                    return
                if isinstance(response, _DeferredResponse):
                    def complete(success, message, c=connection, deferred=response):
                        self._pending_replies.pop(c, None)
                        if c not in self._buffers:
                            return
                        if success:
                            result = message if isinstance(message, dict) else deferred.result
                            reply = ok_response(deferred.request_id, result,
                                                (time.perf_counter() - deferred.started) * 1000)
                        else:
                            error = message if isinstance(message, ControlError) else ControlError(
                                'internal', message or 'Export cancelled')
                            reply = error_response(deferred.request_id, error)
                        if self._send_response(c, reply):
                            QTimer.singleShot(0, lambda: self._read(c))
                    try:
                        self._pending_replies[connection] = response.start(complete)
                    except Exception as error:
                        reply = error_response(response.request_id, ControlError(
                            'invalid_argument' if isinstance(error, (ValueError, TypeError)) else 'internal',
                            str(error) or type(error).__name__))
                        if not self._send_response(connection, reply):
                            return
                    else:
                        return
                elif not self._send_response(connection, response):
                    return
                buffer = self._buffers.get(connection)
                if buffer is None:
                    return
            if len(buffer) > MAX_LINE:
                self._reject_request(connection, "Request line is too large")
                return
            self._buffers[connection] = buffer

    def _send_response(self, connection, response):
        payload = self._encode_response(response)
        if payload is None or connection.bytesToWrite() + len(payload) > MAX_LINE * 2:
            connection.abort()
            return False
        if connection.write(payload) != len(payload):
            connection.abort()
            return False
        connection.flush()
        return True

    def _encode_response(self, response):
        """Encode one reply; a too-large payload becomes a small error.

        Without this the encode() failure escaped the Qt slot and the client
        waited for a line that would never arrive.
        """
        try:
            return encode(response)
        except Exception:  # noqa: BLE001 - report serialization and size errors
            try:
                return encode(error_response(
                    str(response.get("id", "0")),
                    ControlError("internal", "Response is too large or cannot be encoded")))
            except Exception:  # noqa: BLE001 - even the request id may be too large
                return None

    def _reject_request(self, connection, message):
        self._buffers.pop(connection, None)
        if connection.bytesToWrite() > MAX_LINE:
            connection.abort()
            return
        try:
            connection.write(encode(error_response(
                "0", ControlError("bad_request", message))))
            connection.flush()
        except Exception:  # noqa: BLE001 - closing anyway
            connection.abort()
            return
        connection.disconnectFromServer()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._watchdog_stop is not None:
            self._watchdog_stop.set()
        self._remove_tracking()
        for dialog in list(self._dialogs.values()):
            try:
                dialog.close()
            except RuntimeError:
                pass
        self._dialogs.clear()
        for connection in list(self._buffers):
            connection.abort()
            if connection in self._buffers:
                self._drop(connection)
        self._server.close()
        self._cleanup_unix_endpoint()
        if self._unix_stage is not None:
            self._unix_stage.cleanup()
            self._unix_stage = None
        clear_discovery(os.getpid())
        if self._dialogs_patched:
            uninstall_non_native_color_dialogs()
            self._dialogs_patched = False

    # ------------------------------------------------------------ dispatch
    def handle_line(self, line, token=None):
        request_id = "0"
        started = time.perf_counter()
        try:
            payload = decode(line)
            request_id = str(payload.get("id", "0"))
            if token is not None and payload.get("token") != token:
                raise ControlError("unauthorized", "Invalid control token")
            command = payload.get("cmd")
            if not isinstance(command, str) or not command:
                raise ControlError("bad_request", "cmd is required")
            args = payload.get("args") or {}
            if not isinstance(args, dict):
                raise ControlError("bad_request", "args must be an object")
            handler = getattr(self, "_cmd_" + command.replace(".", "_"), None)
            if handler is None:
                raise ControlError("not_found", f"Unknown command: {command}",
                                   f"Known commands: {', '.join(_COMMANDS)}")
            from utils.file_dialogs import control_picker_context
            with control_picker_context():
                result = handler(args)
            if isinstance(result, _DeferredResponse):
                result.request_id = request_id
                result.started = started
                return result
            return ok_response(request_id, result,
                               (time.perf_counter() - started) * 1000)
        except ControlError as error:
            return error_response(request_id, error)
        except KeyError as error:
            return error_response(request_id, ControlError(
                "not_found", str(error).strip("'") or "Not found"))
        except (ValueError, TypeError) as error:
            return error_response(request_id, ControlError(
                "invalid_argument", str(error).strip("'") or type(error).__name__))
        except Exception as error:  # noqa: BLE001 - reported to the client
            _log.exception("control command failed")
            return error_response(request_id, ControlError(
                "internal", f"{type(error).__name__}: {error}"))

    # ----------------------------------------------------------- app/stack
    def _cmd_ping(self, args):
        return {"pong": True, "version": app_display_version(), "schema": SCHEMA}

    def _cmd_app_info(self, args):
        import platform
        import sys
        from importlib.util import find_spec

        methods = []
        for spec in fusion_algorithms():
            methods.append({
                "id": spec.id,
                "label": trans.t(spec.label_key),
                "category": spec.category,
                "neural": spec.neural,
                "order_sensitive": spec.order_sensitive,
                "depth_map": spec.depth is not None,
                "weights": spec.weights,
                "parameters": [_spec_dict(parameter) for parameter in spec.parameters],
            })
        capabilities = {
            "torch": find_spec("torch") is not None,
            "heic": find_spec("pillow_heif") is not None,
            "raw": find_spec("rawpy") is not None,
            "dtcwt": find_spec("dtcwt") is not None,
            "pyqt": True,
        }
        try:
            from utils.platform_utils import cached_compute_gpu
            capabilities["gpu"] = cached_compute_gpu()
        except Exception:
            capabilities["gpu"] = None
        return {
            "version": app_display_version(),
            "pid": os.getpid(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "methods": methods,
            "registrations": [spec.id for spec in registration_algorithms()],
            "capabilities": capabilities,
            "control": {"schema": SCHEMA, "socket": self.socket_name,
                        "token_required": True, "commands": list(_COMMANDS)},
        }

    def _cmd_app_state(self, args):
        window = self.window
        method = window.current_fusion_method()
        alignment = window.current_alignment_method()
        return {
            "stack": _stack_info(window),
            "method": {"id": method,
                       "label": trans.t(get_fusion(method).label_key)
                       if method != "none" else "none",
                       "parameters": window.fusion_parameters_for(method)},
            "alignment": {"id": alignment,
                          "parameters": window.registration_parameters_for(alignment)},
            "options": {
                "base_correct": bool(window.chk_base_correct.isChecked()),
                "exposure_match": bool(window.chk_exposure_match.isChecked()),
                "reverse_order": bool(getattr(window, "stack_reverse_order", False)),
                "preview": bool(window.chk_quick_preview.isChecked()),
                "thread_count": int(getattr(window, "thread_count", 0)),
                "use_gpu": bool(getattr(window, "use_gpu", False)),
                "tile_enabled": bool(getattr(window, "tile_enabled", True)),
                "tile_block_size": int(getattr(window, "tile_block_size", 1024)),
                "tile_overlap": int(getattr(window, "tile_overlap", 256)),
                "tile_threshold": int(getattr(window, "tile_threshold", 2048)),
                "ai_batch_size": int(getattr(window, "ai_batch_size", 2)),
                "resource_preference": getattr(window, "resource_preference", "balanced"),
                "reg_downscale_width": int(getattr(window, "reg_downscale_width", 1024)),
                "storage_directory": getattr(window, "storage_directory", "") or "",
                "profile_performance": bool(getattr(window, "profile_performance", False)),
                "debug_logging": bool(getattr(window, "debug_logging", False)),
            },
            "render": _render_state(window),
            "outputs": {"count": window.output_list.count(),
                        "selected": window.output_list.currentRow()},
            "dialogs": self._dialog_list(),
            "view": _view_state(window),
            "context": _root_context(window),
        }

    def _cmd_app_quit(self, args):
        force = _boolean(args, "force", default=False)
        if force:
            # Automation teardown: skip unexported/unsaved/in-flight prompts.
            # The agent explicitly asked to discard them.
            self.window._force_quit = True
        QTimer.singleShot(100, self.window.close)
        return {"quitting": True, "force": force}

    def _cmd_stack_load(self, args):
        folder = _string(args, "folder", required=False)
        video = _string(args, "video", required=False)
        files = args.get("files")
        if not folder and not video and not files:
            raise ControlError("invalid_argument",
                               "Provide folder, video or files")
        if files is not None and (not isinstance(files, list)
                                 or not all(isinstance(item, str) and item for item in files)):
            raise ControlError("invalid_argument", "files must be a list of paths")
        scale = float(args.get("scale", 1.0) or 1.0)
        if scale <= 0:
            raise ControlError("invalid_argument", "scale must be positive")
        append = _boolean(args, "append", default=False)
        linear_raw = _boolean(args, "linear_raw", default=False)
        if linear_raw and video:
            raise ControlError("invalid_argument", "linear_raw requires a folder or files")
        manager = self.window.source_manager
        if getattr(manager, "_load_worker", None) is not None:
            raise ControlError("busy", "A stack is already loading",
                               "Poll stack.frames, then retry.")
        load_id = None
        if folder:
            load_id = manager._start_load_worker(
                folder=folder, scale=scale, append=append, linear_raw=linear_raw)
            source = folder
        elif video:
            load_id = manager._start_load_worker(
                video=video, scale=scale, append=append)
            source = video
        else:
            load_id = manager._start_load_worker(
                filepaths=list(files), scale=scale, append=append, linear_raw=linear_raw)
            source = list(files)
        return {"loading": True, "source": source, "load_id": load_id,
                "baseline_count": self._load_baseline}

    def _cmd_stack_frames(self, args):
        info = _stack_info(self.window)
        info["load"] = self._load_info()
        return info

    def _cmd_stack_clear(self, args):
        self.window.source_manager.clear_image_stack()
        return {"cleared": True}

    def _cmd_stack_include(self, args):
        indices = args.get("indices")
        if (not isinstance(indices, list)
                or not all(isinstance(index, int) and not isinstance(index, bool)
                           for index in indices)):
            raise ControlError("invalid_argument", "indices must be integers")
        include = _boolean(args, "include", default=True)
        window = self.window
        manager = window.source_manager
        items = []
        for index in indices:
            if not 0 <= index < len(window.raw_images):
                raise ControlError("invalid_argument", f"Frame index out of range: {index}")
            item = window.file_list.item(index)
            if item is None:
                raise ControlError("not_found", f"No frame row at index {index}")
            items.append(item)
        for item in items:
            # Drive the same handler the checkbox uses: it pushes undo, marks
            # the project dirty, bumps source_revision and drops the cached
            # alignment/ROI. Block the list signal so the state change is
            # handled exactly once, whichever path called us.
            blocked = window.file_list.blockSignals(True)
            try:
                item.setCheckState(Qt.CheckState.Checked if include
                                   else Qt.CheckState.Unchecked)
            finally:
                window.file_list.blockSignals(blocked)
            manager.on_item_changed(item)
        return _stack_info(window)

    # -------------------------------------------------- method/align/options
    def _cmd_method_set(self, args):
        method = _string(args, "id")
        self._validate_method_selection(method, args, fusion=True)
        self.window.set_fusion_method(method)
        if "parameters" in args:
            self.window.set_fusion_parameters(method, args["parameters"])
        self.window.persist_settings()
        return {"id": method,
                "parameters": self.window.fusion_parameters_for(method)}

    def _cmd_method_parameters(self, args):
        method = _string(args, "id", required=False) or self.window.current_fusion_method()
        spec = get_fusion(method)
        return {
            "id": method,
            "parameters": self.window.fusion_parameters_for(method),
            "specs": [_spec_dict(parameter) for parameter in spec.parameters],
            "depth": spec.depth is not None,
        }

    def _cmd_align_set(self, args):
        method = _string(args, "id")
        self._validate_method_selection(method, args, fusion=False)
        self.window.set_alignment_method(method)
        if "parameters" in args:
            self.window.set_registration_parameters(method, args["parameters"])
        self.window.persist_settings()
        return {"id": method,
                "parameters": self.window.registration_parameters_for(method)}

    def _validate_method_selection(self, method, args, *, fusion):
        from core.experimental import (alignment_is_experimental, experimental_enabled,
                                       fusion_is_experimental)
        window = self.window
        experimental = fusion_is_experimental if fusion else alignment_is_experimental
        if (hasattr(window, 'experimental_features') and not experimental_enabled(window)
                and experimental(method) and not getattr(window, '_restore_depth', 0)):
            raise ValueError(trans.t('experimental_locked_hint'))
        if method != "none":
            (get_fusion if fusion else get_registration)(method)
        combo = window.combo_fusion if fusion else window.combo_align
        if combo.findData(method) < 0:
            kind = "Fusion" if fusion else "Registration"
            raise ValueError(f"{kind} method is not in the selector: {method}")
        if "parameters" not in args:
            return
        values = args["parameters"]
        if not isinstance(values, dict):
            raise ControlError("invalid_argument", "parameters must be an object")
        if method == "none":
            if values:
                raise ValueError("Alignment-only output has no fusion parameters" if fusion
                                 else "Disabled registration has no parameters")
            return
        current = (window.fusion_parameters_for(method) if fusion
                   else window.registration_parameters_for(method))
        resolver = resolve_fusion_parameters if fusion else resolve_registration_parameters
        resolver(method, {**current, **values})

    def _cmd_option_set(self, args):
        name = _string(args, "name")
        window = self.window
        boolean_options = {"base_correct", "exposure_match", "reverse_order", "use_gpu",
                           "tile_enabled", "preview", "profile_performance", "debug_logging"}
        integer_options = {"thread_count": 0, "tile_block_size": 64, "tile_overlap": 0,
                           "tile_threshold": 1, "ai_batch_size": 1, "reg_downscale_width": 64}
        if "value" not in args:
            raise ControlError("invalid_argument", "value is required")
        if name in boolean_options:
            value = _boolean(args, "value", required=True)
        elif name in integer_options:
            value = _integer(args, "value", required=True, minimum=integer_options[name])
        elif name == "resource_preference":
            value = _string(args, "value", choices=("balanced", "responsive", "maximum"))
        elif name == "storage_directory":
            value = "" if args["value"] is None else _string(args, "value", required=False)
            value = os.path.abspath(value) if value else ""
        else:
            raise ControlError("invalid_argument", f"Unknown option: {name}")

        if name in ("tile_block_size", "tile_overlap"):
            block = value if name == "tile_block_size" else window.tile_block_size
            overlap = value if name == "tile_overlap" else window.tile_overlap
            if overlap >= block:
                raise ControlError("invalid_argument", "tile_overlap must be less than tile_block_size")

        if name in ("base_correct", "exposure_match"):
            window.set_brightness_option(name, window.current_fusion_method(), value)
            window._sync_brightness_checkboxes()
            value = getattr(window, "chk_" + name).isChecked()
        elif name == "reverse_order":
            window.set_reverse_order(value)
            checkbox = getattr(window, "chk_reverse_order", None)
            if checkbox is not None:
                blocked = checkbox.blockSignals(True)
                checkbox.setChecked(value)
                checkbox.blockSignals(blocked)
            value = window.stack_reverse_order
        elif name == "use_gpu":
            window.set_use_gpu(value)
            value = window.use_gpu
        elif name == "preview":
            window.chk_quick_preview.setChecked(value)
            value = window.chk_quick_preview.isChecked()
        elif name == "storage_directory":
            from core.frame_storage import configure_storage_directory
            try:
                configure_storage_directory(value)
            except OSError as error:
                raise ControlError("invalid_argument",
                                   f"Temporary directory is not writable: {error}") from error
            window.storage_directory = value
            value = window.storage_directory
        elif name == "debug_logging":
            from utils.logging_setup import set_debug_logging
            window.debug_logging = value
            set_debug_logging(value)
            value = window.debug_logging
        else:
            setattr(window, name, value)
            value = getattr(window, name)
        window.persist_settings()
        return {"name": name, "value": value}

    # ------------------------------------------------------------- render
    def _cmd_render_start(self, args):
        manager = self.window.render_manager
        worker = getattr(manager, "worker", None)
        if ((worker is not None and worker.isRunning())
                or getattr(manager, '_pending_storage', None) is not None):
            raise ControlError("busy", "A render is already running",
                               "Wait for render.state to leave 'rendering' or call render.cancel.")
        preview = _boolean(args, "preview", default=False)
        # Apply both ways: leaving an earlier preview request checked would
        # silently turn every later "full" render into a downscaled draft.
        self.window.chk_quick_preview.setChecked(preview)
        method = self.window.current_fusion_method()
        alignment = self.window.current_alignment_method()
        if method == "none" and alignment == "none":
            raise ControlError(
                "invalid_argument",
                "Select a fusion method or an alignment method first",
                "method.set id=none renders the aligned stack only; with "
                "alignment 'none' there is nothing to do.")
        confirm = _boolean(args, "confirm", default=False)
        roi_mode = _string(args, "roi_mode", required=False)
        roi_choice = None
        if roi_mode is not None:
            if roi_mode not in ("crop", "paste"):
                raise ControlError("invalid_argument",
                                   "roi_mode must be 'crop' or 'paste'")
            roi_choice = {"mode": roi_mode,
                          "base_index": _integer(args, "roi_base_index",
                                                 default=0, minimum=0)}
        # interactive=False: a decision that would open a modal box comes back
        # as a status, so the socket is answered before anything blocks.
        before = self._render_id
        try:
            status = manager.start_render(interactive=False, confirmed=confirm,
                                          roi_choice=roi_choice)
        except ValueError as error:
            raise ControlError("invalid_argument", str(error)) from error
        if status == 'preflight':
            def start(complete):
                def finished(final_status):
                    try:
                        result = self._render_start_response(final_status, before, method, alignment, preview)
                    except ControlError as error:
                        complete(False, error)
                    else:
                        complete(True, result)
                return manager.set_storage_completion(finished)
            return _DeferredResponse(start)
        return self._render_start_response(status, before, method, alignment, preview)

    def _render_start_response(self, status, before, method, alignment, preview):
        manager = self.window.render_manager
        storage = getattr(manager, "last_storage_assessment", None)
        if status == "started":
            if self._render_id == before:
                # A monkeypatched/test renderer bypassed the generation
                # wrapper; keep the id accurate for --wait all the same.
                self._render_id += 1
            reply = {"started": True, "render_id": self._render_id,
                     "method": method, "alignment": alignment,
                     "preview": preview}
            if storage is not None:
                # Also after confirm=true: a pre-confirming client still sees it.
                reply["storage"] = storage
            return reply
        if status == "confirm_storage":
            raise ControlError(
                "confirm_required",
                "Temporary storage is low for this render",
                "Choose another directory with option.set name=storage_directory, "
                "or retry with confirm=true to continue.",
                details={"storage": storage})
        if status == "insufficient_storage":
            raise ControlError(
                "insufficient_storage",
                "The temporary directory cannot hold this render",
                "Choose another directory with option.set name=storage_directory "
                "and retry; confirm=true does not override this.",
                details={"storage": storage})
        if status == "busy":
            raise ControlError("busy", "A render is already running",
                               "Wait for render.state to leave 'rendering' or "
                               "call render.cancel.")
        if status == "confirm_ai":
            raise ControlError(
                "confirm_required",
                "The selected AI model runs on the CPU and is much slower",
                "Retry with confirm=true to accept the warning.")
        if status == "confirm_roi":
            raise ControlError(
                "confirm_required",
                "An ROI render needs an output decision",
                "Pass roi_mode ('crop' or 'paste') and roi_base_index, or "
                "leave ROI mode.")
        if status == "no_frames":
            raise ControlError("invalid_argument",
                               "At least two usable frames are required",
                               "Load a stack or include more frames first.")
        if status == "unavailable":
            raise ControlError("unsupported",
                               "The selected AI model is not available",
                               "Install torch or pick another method.")
        if status == "stack_changed":
            raise ControlError("invalid_argument",
                               "The stack changed while the ROI was active",
                               "Reset the ROI selection and retry.")
        if status in ("sensor_history_unsupported", "sensor_roi_unsupported"):
            raise ControlError("invalid_argument", manager.last_error)
        if status == "cancelled":
            raise ControlError("internal",
                               "The render was cancelled before it started",
                               "Retry render.start.")
        raise ControlError("internal", f"Render did not start ({status})")

    def _cmd_render_cancel(self, args):
        cancel = getattr(self.window.render_manager, 'cancel_storage_preflight', None)
        if callable(cancel) and cancel():
            return {"cancelling": True, "render_id": self._render_id}
        worker = getattr(self.window.render_manager, "worker", None)
        if worker is None:
            return {"cancelling": False}
        worker.is_cancelled = True
        return {"cancelling": True, "render_id": self._render_id}

    def _cmd_render_state(self, args):
        state = _render_state(self.window)
        state["render_id"] = self._render_id
        return state

    def _cmd_render_result(self, args):
        state = _render_state(self.window)
        result = dict(state.get("result") or {})
        image = getattr(self.window, "fusion_result", None)
        if image is not None:
            result.setdefault("shape", list(image.shape))
            result.setdefault("dtype", str(image.dtype))
        outputs = self._cmd_output_list({})
        row = self.window.output_list.currentRow()
        result["output"] = next((record for record in outputs["records"]
                                 if record["index"] == row), None)
        # The last profiled render's report, also for a failed or cancelled one.
        report = getattr(self.window.render_manager, "performance_report", None)
        if report:
            result["performance"] = report
        return result

    # ------------------------------------------------------------- outputs
    def _cmd_output_list(self, args):
        window = self.window
        manager = window.output_manager
        records = []
        for row in range(window.output_list.count()):
            item = window.output_list.item(row)
            record = manager.record_for_item(item)
            image = record.image if record is not None else None
            records.append({
                "index": row,
                "label": item.text() if item is not None else "",
                "kind": getattr(record, "kind", None),
                "exported": bool(getattr(record, "exported", False)),
                "retouchable": bool(getattr(record, "retouchable", False)),
                "export_path": getattr(record, "export_path", None),
                "shape": list(image.shape) if image is not None else None,
                "dtype": str(image.dtype) if image is not None else None,
            })
        return {"count": len(records), "selected": window.output_list.currentRow(),
                "records": records}

    def _cmd_output_select(self, args):
        index = _integer(args, "index", required=True, minimum=0)
        item = self.window.output_list.item(index)
        if item is None:
            raise ControlError("not_found", f"No output at index {index}")
        self.window.output_list.setCurrentItem(item)
        return {"selected": index, "label": item.text()}

    def _cmd_output_display(self, args):
        index = _integer(args, "index", required=True, minimum=0)
        item = self.window.output_list.item(index)
        if item is None:
            raise ControlError("not_found", f"No output at index {index}")
        self.window.output_list.setCurrentItem(item)
        self.window.output_manager.display_current_output_row(index)
        return {"displayed": True}

    def _cmd_output_export(self, args):
        from copy import deepcopy
        from dataclasses import replace
        path = _string(args, "path")
        index = _integer(args, "index", required=False,
                         default=self.window.output_list.currentRow(), minimum=0)
        window = self.window
        item = window.output_list.item(index)
        if item is None:
            raise ControlError("not_found", f"No output at index {index}")
        manager = window.output_manager
        original = manager.record_for_item(item)
        if original is None:
            raise ControlError("not_found", "Output record is unavailable")
        record = replace(original, scale_bar_config=deepcopy(original.scale_bar_config))
        target = window.export_manager.normalize_export_path(path)
        labels = window.label_manager.snapshot()
        if getattr(original, "kind", None) == "aligned":
            # An aligned-stack output is a folder of frames, not one image;
            # handing it to the image writer produced a single broken file.
            destination = os.path.dirname(target) or None
            written = manager._export_aligned_record(
                record, destination=destination, labels_snapshot=labels)
            if not written:
                raise ControlError("internal", "The aligned stack was not written")
            manager._mark_record_exported(
                original, record.image, written,
                manager._annotation_state(record, labels))
            return {"path": written, "written": True, "index": index,
                    "kind": "aligned"}
        if os.path.splitext(target)[1].lower() == '.dng':
            deferred = _DeferredResponse(lambda complete: manager.start_linear_dng_export(
                record, original, target, labels, completion=complete, show_progress=False))
            deferred.result = {"path": target, "written": True, "index": index}
            return deferred
        manager._write_record(record, target, labels)
        manager._mark_record_exported(
            original, record.image, target,
            manager._annotation_state(record, labels))
        return {"path": target, "written": True, "index": index}

    def _cmd_output_delete(self, args):
        index = _integer(args, "index", required=True, minimum=0)
        force = _boolean(args, "force", default=False)
        window = self.window
        item = window.output_list.item(index)
        if item is None:
            raise ControlError("not_found", f"No output at index {index}")
        manager = window.output_manager
        record = manager.record_for_item(item)
        if record is not None and record.protected and not force:
            raise ControlError(
                "confirm_required",
                "The output is an unexported retouch/compare result",
                "Export it first, or retry with force=true to discard it.")
        manager._remove_row(index)
        # Mirror delete_selected_output_images: the canvas and
        # window.fusion_result must follow the list instead of keeping the
        # deleted pixels. Keep the row the user had selected (shifted by the
        # removal) instead of always jumping to the deleted index.
        if window.output_list.count():
            current = window.output_list.currentRow()
            new_index = current - 1 if current > index else current
            new_index = max(0, min(new_index, window.output_list.count() - 1))
            window.output_list.setCurrentRow(new_index)
            manager.display_output_image_in_result_view(
                window.output_list.item(new_index))
        else:
            window.fusion_result = None
            window.lbl_result_img.clear()
            window.lbl_result_img.setText(trans.t("result_hint"))
            window.result_control_bar.setVisible(False)
        return {"deleted": True, "index": index}

    # ------------------------------------------------------------- project
    def _cmd_project_save(self, args):
        from utils.project_file import save_project
        window = self.window
        path = _string(args, "path") if "path" in args else getattr(window, "project_path", None)
        if not path:
            raise ControlError(
                "invalid_argument", "No path given and no project open",
                "Pass path=... or open a project first.")
        ok, error = save_project(window, path)
        if not ok:
            raise ControlError("invalid_argument", error or "Project was not saved")
        # Mirror save_project_dialog: track the file and clear the dirty
        # flag, otherwise a later GUI save-overwrite hits the wrong target.
        window.project_path = path
        window._clear_project_dirty()
        return {"saved": True, "path": path}

    def _cmd_project_open(self, args):
        from utils.project_file import apply_project, validate_project
        path = _string(args, "path")
        force = _boolean(args, "force", default=False)
        window = self.window
        if getattr(window, "_project_dirty", False) and not force:
            # The GUI path asks before discarding; an agent must opt in.
            raise ControlError(
                "confirm_required", "The open project has unsaved changes",
                "Save it with project.save, or retry with force=true to "
                "discard the changes.")
        ok, error, state = validate_project(path)
        if not ok:
            raise ControlError("invalid_argument", error or "Project is invalid")
        if getattr(window.source_manager, "_load_worker", None) is not None:
            raise ControlError("busy", "A stack is already loading",
                               "Poll stack.frames, then retry.")

        def loaded():
            window.project_path = path
            window._clear_project_dirty()
            window.add_recent_project(path)

        try:
            apply_project(window, state, on_loaded=loaded)
        except Exception as error:  # noqa: BLE001 - reported to the client
            raise ControlError("invalid_argument", str(error)) from error
        return {"loading": True, "path": path, "load_id": self._load_id,
                "discarded_unsaved": bool(force)}

    # ------------------------------------------------------------- dialogs
    @staticmethod
    def _dialog_busy(dialog):
        if getattr(dialog, "_import_busy", False):
            return True
        sensor = getattr(dialog, "sensor_cleanup", None)
        if getattr(sensor, "_pending", False):
            return True
        for attribute in ("_worker", "_export_worker"):
            worker = getattr(dialog, attribute, None)
            if worker is not None and hasattr(worker, "isRunning") \
                    and worker.isRunning():
                return True
        return False

    def _window_entries(self):
        from ui.control_surface import window_roots
        roots = window_roots(self.window)
        return [(name, widget) for name, widget in roots.items()
                if name != "main"]

    def _dialog_list(self):
        managed = {id(dialog) for dialog in self._dialogs.values()}
        active_modal = QApplication.activeModalWidget()
        active_window = QApplication.activeWindow()
        return [{"index": index,
                 "root": name,
                 "title": (dialog.windowTitle()
                           or dialog.objectName()
                           or type(dialog).__name__),
                 "class": ("QFileDialog" if isinstance(dialog, QFileDialog)
                           else type(dialog).__name__),
                 "visible": True,
                 "modal": bool(dialog.isModal()),
                 "popup": name.startswith("popup["),
                 "managed": id(dialog) in managed,
                 "active": dialog is active_modal or dialog is active_window,
                 "busy": self._dialog_busy(dialog)}
                for index, (name, dialog) in enumerate(self._window_entries())]

    def _open_dialog(self, name, dialog):
        existing = self._dialogs.get(name)
        if existing is not None and not sip.isdeleted(existing):
            if existing.isVisible():
                existing.raise_()
                existing.activateWindow()
                return {"opened": True, "title": existing.windowTitle(),
                        "existing": True}
            self._dialogs.pop(name, None)
            existing.close()
        dialog.setModal(False)
        def forget(*_args):
            if self._dialogs.get(name) is dialog:
                self._dialogs.pop(name, None)
        dialog.finished.connect(forget)
        dialog.destroyed.connect(forget)
        if name == "batch":
            dialog.accepted.connect(lambda d=dialog: self._start_batch(d))
        self._dialogs[name] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return {"opened": True, "title": dialog.windowTitle()}

    def _start_batch(self, dialog):
        try:
            self.window._start_batch_from_dialog(dialog)
        except Exception:
            _log.exception("control: batch start failed")

    def _cmd_dialog_open(self, args):
        name = _string(args, "name")
        window = self.window
        # Reuse a managed instance before building a new dialog: constructing
        # first leaked images and signal handlers on every repeated open.
        existing = self._dialogs.get(name)
        if existing is not None and not sip.isdeleted(existing):
            if existing.isVisible():
                existing.raise_()
                existing.activateWindow()
                return {"opened": True, "title": existing.windowTitle(),
                        "existing": True}
            self._dialogs.pop(name, None)
            existing.close()
        if name == "settings":
            from dialogs.advanced_settings import AdvancedSettingsDialog
            dialog = AdvancedSettingsDialog(window)
        elif name == "batch":
            from dialogs.batch import BatchProcessingDialog
            dialog = BatchProcessingDialog(window)
            folders = args.get("folders")
            if folders is not None:
                if (not isinstance(folders, list)
                        or not all(isinstance(folder, str) and folder
                                   for folder in folders)):
                    raise ControlError("invalid_argument",
                                       "folders must be a list of paths")
                dialog.add_folders_to_list(folders)
            output_dir = args.get("output_dir")
            if output_dir is not None:
                if not isinstance(output_dir, str) or not output_dir:
                    raise ControlError("invalid_argument",
                                       "output_dir must be a path")
                dialog.rb_custom_folder.setChecked(True)
                dialog.custom_folder_path.setText(output_dir)
        elif name == "compare":
            if len(getattr(window, "raw_images", []) or []) < 2:
                raise ControlError("invalid_argument",
                                   "Comparison needs at least two loaded frames")
            from dialogs.compare import CompareDialog
            dialog = CompareDialog(window)
        elif name == "depth":
            from dialogs.depthmap import DepthMapDialog
            dialog = DepthMapDialog(window)
        elif name == "feedback":
            from dialogs.feedback import FeedbackDialog
            dialog = FeedbackDialog(window)
        elif name == "scalebar":
            from dialogs.scalebar import ScaleBarDialog
            dialog = ScaleBarDialog(window)
        elif name == "dust":
            from dialogs.dustmap import DustMapDialog
            dialog = DustMapDialog(window)
        elif name == "retouch":
            dialog = self._retouch_dialog()
        elif name == "welcome":
            from dialogs.welcome import WelcomeDialog
            dialog = WelcomeDialog(window, main_window=window)
        elif name == "method_help":
            from dialogs.help import RenderMethodHelpDialog
            dialog = RenderMethodHelpDialog(window)
        elif name == "registration_help":
            from dialogs.help import RegistrationHelpDialog
            dialog = RegistrationHelpDialog(window)
        elif name == "environment":
            from dialogs.about import EnvironmentInfoDialog
            dialog = EnvironmentInfoDialog(window)
        elif name == "contact":
            from dialogs.about import ContactInfoDialog
            dialog = ContactInfoDialog(window)
        else:
            raise ControlError("invalid_argument", f"Unknown dialog: {name}")
        return self._open_dialog(name, dialog)

    def _retouch_dialog(self):
        window = self.window
        item = window.output_list.currentItem()
        if item is None and window.output_list.count() == 1:
            item = window.output_list.item(0)
        if item is None:
            raise ControlError("invalid_argument", "Select an output first")
        record = window.output_manager.record_for_item(item)
        if record is None or not record.retouchable:
            raise ControlError("invalid_argument",
                               "The selected output has no retouchable source")
        from dialogs.retouch import RetouchDialog
        return RetouchDialog(window, record=record,
                             apply_callback=window._apply_retouch_output)

    def _cmd_dialog_list(self, args):
        return {"dialogs": self._dialog_list()}

    def _find_dialog(self, args):
        index = _integer(args, "index", required=False, default=None, minimum=0)
        title = _string(args, "title", required=False)
        root_name = _string(args, "root", required=False)
        entries = self._window_entries()
        if root_name is not None:
            for name, dialog in entries:
                if name == root_name:
                    return dialog
            raise ControlError("not_found", f"No window root: {root_name}")
        if index is not None:
            if index >= len(entries):
                raise ControlError("not_found", f"No window at index {index}")
            return entries[index][1]
        if title is None:
            if len(entries) == 1:
                return entries[0][1]
            if not entries:
                raise ControlError("not_found", "No dialog or popup is open")
            raise ControlError("invalid_argument",
                               "Several windows are open; provide index or title")
        matches = [dialog for _name, dialog in entries
                   if title.casefold() in (dialog.windowTitle() or "").casefold()]
        if not matches:
            raise ControlError("not_found", f"No dialog titled {title!r}")
        if len(matches) > 1:
            raise ControlError("invalid_argument",
                               f"Several dialogs match {title!r}; provide index")
        return matches[0]

    def _cmd_dialog_close(self, args):
        dialog = self._find_dialog(args)
        dialog.close()
        return {"closed": True}

    def _cmd_dialog_accept(self, args):
        from PyQt6.QtWidgets import QDialogButtonBox, QPushButton
        dialog = self._find_dialog(args)
        title = dialog.windowTitle()
        if self._dialog_busy(dialog):
            raise ControlError("busy", "The dialog is still working",
                               "Poll dialog.list until busy is false, then retry.")
        if not hasattr(dialog, "accept"):
            raise ControlError("unsupported",
                               f"{type(dialog).__name__} has no accept()")
        # Raw accept() skips per-dialog commit handlers (retouch would drop
        # unapplied strokes, settings would drop pending edits), so route
        # through the dialog's own commit action whenever one exists.
        apply_button = getattr(dialog, "btn_apply", None)
        if isinstance(apply_button, QPushButton) and apply_button.isEnabled():
            apply_button.click()
        # Batch's Start action validates before accepting; raw accept() would
        # launch an unvalidated job.
        ok_button = getattr(dialog, "btn_start", None)
        if not isinstance(ok_button, QPushButton):
            ok_button = getattr(dialog, "btn_ok", None)
        if not isinstance(ok_button, QPushButton):
            for box in dialog.findChildren(QDialogButtonBox):
                ok_button = box.button(QDialogButtonBox.StandardButton.Ok)
                if ok_button is not None:
                    break
        if isinstance(ok_button, QPushButton):
            if not ok_button.isEnabled():
                raise ControlError(
                    "invalid_argument",
                    "The dialog's OK action is disabled",
                    "Complete the dialog's inputs (see ui.tree), then retry.")
            ok_button.click()
        else:
            dialog.accept()
        try:
            still_open = dialog.isVisible()
        except RuntimeError:  # a delete-on-close dialog was destroyed
            still_open = False
        if still_open:
            if self._dialog_busy(dialog):
                raise ControlError("busy", "The dialog is still working",
                                   "Poll dialog.list until busy is false, then retry.")
            raise ControlError("invalid_argument", "The dialog did not accept its inputs",
                               "Check the dialog for invalid inputs or required confirmation.")
        return {"accepted": True, "title": title}

    def _cmd_dialog_reject(self, args):
        dialog = self._find_dialog(args)
        if not hasattr(dialog, "reject"):
            raise ControlError("unsupported",
                               f"{type(dialog).__name__} has no reject()")
        dialog.reject()
        return {"rejected": True, "title": dialog.windowTitle()}

    def _cmd_dialog_color(self, args):
        value = _string(args, "color")
        color = QColor(value)
        if not color.isValid():
            raise ControlError("invalid_argument", f"Not a color: {value}")
        index = _integer(args, "index", required=False, default=None, minimum=0)
        title = _string(args, "title", required=False)
        if index is not None or title is not None:
            dialog = self._find_dialog({"index": index, "title": title})
            if not isinstance(dialog, QColorDialog):
                raise ControlError("invalid_argument",
                                   "The selected window is not a color dialog")
        else:
            candidates = [widget for widget in QApplication.topLevelWidgets()
                          if isinstance(widget, QColorDialog)
                          and _dialog_candidate(widget)]
            if not candidates:
                raise ControlError("not_found", "No color dialog is open")
            dialog = candidates[-1]
        dialog.setCurrentColor(color)
        dialog.accept()
        return {"accepted": True, "color": color.name()}

    def _cmd_dialog_file(self, args):
        """Answer a file/directory picker opened by a menu action or dialog."""
        from utils.file_dialogs import native_picker_active
        if native_picker_active():
            raise ControlError("busy", "A native file picker is open")
        path = _string(args, "path", required=False)
        cancel = _boolean(args, "cancel", default=False)
        index = _integer(args, "index", required=False, default=None, minimum=0)
        title = _string(args, "title", required=False)
        candidates = [widget for widget in QApplication.topLevelWidgets()
                      if isinstance(widget, QFileDialog)
                      and _dialog_candidate(widget)]
        if not candidates:
            active = QApplication.activeModalWidget()
            if isinstance(active, QFileDialog):
                candidates = [active]
        if not candidates:
            raise ControlError("not_found", "No file dialog is open",
                               "Trigger the action, then poll dialog.list for a "
                               "QFileDialog entry.")
        if index is not None or title is not None:
            dialog = self._find_dialog({"index": index, "title": title})
            if not isinstance(dialog, QFileDialog):
                raise ControlError("invalid_argument",
                                   "The selected window is not a file dialog")
        else:
            active = QApplication.activeModalWidget()
            dialog = active if active in candidates else candidates[-1]
        if cancel:
            _invoke(dialog, "reject")
            return {"cancelled": True}
        if not path:
            raise ControlError("invalid_argument", "path is required unless cancel")
        path = os.path.abspath(os.path.expanduser(path))
        try:
            mode = dialog.fileMode()
        except RuntimeError:
            # C++-owned wrappers (native/offscreen platforms) can refuse
            # member access; the path itself tells us which picker this is.
            mode = (QFileDialog.FileMode.Directory if os.path.isdir(path)
                    else QFileDialog.FileMode.ExistingFile)
        if mode == QFileDialog.FileMode.Directory:
            if not os.path.isdir(path):
                raise ControlError("invalid_argument",
                                   f"Not a directory: {path}")
            _invoke(dialog, "setDirectory", path)
        else:
            # selectFile() can be overwritten by the dialog's own default
            # filename when accept() runs in the same event-loop turn; set the
            # visible edit the user would type into.
            try:
                edit = dialog.findChild(QLineEdit, "fileNameEdit")
            except RuntimeError:
                edit = None
            if edit is not None:
                try:
                    edit.setText(path)
                except RuntimeError:
                    _invoke(dialog, "selectFile", path)
            else:
                _invoke(dialog, "selectFile", path)
        _invoke(dialog, "accept")
        try:
            selected = list(dialog.selectedFiles())
        except RuntimeError:
            selected = [path]
        return {"accepted": True, "path": path,
                "mode": str(mode.name),
                "selected": selected}

    # --------------------------------------------------------------- batch
    def _cmd_batch_state(self, args):
        manager = getattr(self.window, "batch_manager", None)
        busy = bool(manager is not None and getattr(manager, "_worker", None) is not None
                    and getattr(manager, "_thread", None) is not None
                    and manager._thread.isRunning())
        return {"busy": busy,
                "results": getattr(manager, "last_results", None),
                "error": getattr(manager, "last_error", None)}

    def _cmd_batch_cancel(self, args):
        manager = getattr(self.window, "batch_manager", None)
        if manager is not None and getattr(manager, "_worker", None) is not None:
            manager._worker.cancel()
            return {"cancelling": True}
        return {"cancelling": False}

    # ------------------------------------------------------------ ui layer
    def _roots(self):
        from ui.control_surface import window_roots
        return window_roots(self.window)

    def _cmd_ui_actions(self, args):
        from ui.control_surface import action_catalog
        return {"actions": action_catalog(self.window)}

    def _cmd_ui_trigger(self, args):
        from ui.control_surface import find_action
        key = _string(args, "key")
        from utils.file_dialogs import native_picker_active
        if native_picker_active() or isinstance(QApplication.activeModalWidget(), QFileDialog):
            raise ControlError("busy", "A file picker is open")
        action = find_action(self.window, key)
        if action is None or not hasattr(action, "trigger"):
            raise ControlError("not_found", f"Unknown action: {key}")
        if not action.isEnabled():
            raise ControlError("busy", f"Action is disabled: {key}")
        # Let the response reach the socket before a modal exec() starts; a
        # 0 ms timer runs before the buffered write is flushed.
        def fire():
            if native_picker_active() or isinstance(QApplication.activeModalWidget(), QFileDialog):
                QTimer.singleShot(100, fire)
                return
            from utils.file_dialogs import control_picker_context
            with control_picker_context():
                action.trigger()

        QTimer.singleShot(100, fire)
        return {"scheduled": True, "key": key}

    def _cmd_ui_tree(self, args):
        from ui.control_surface import widget_tree
        roots = self._roots()
        window = _string(args, "window", required=False)
        depth = _integer(args, "depth", required=False, default=8)
        if window:
            if window not in roots:
                raise ControlError("not_found", f"Unknown window: {window}")
            return widget_tree(roots[window], depth, window)
        return [widget_tree(root, depth, name) for name, root in roots.items()]

    def _cmd_ui_find(self, args):
        from ui.control_surface import find_widgets
        object_name = _string(args, "object_name", required=False)
        control = _string(args, "control_id", required=False)
        if control:
            object_name = control
        class_name = _string(args, "class", required=False)
        text = _string(args, "text", required=False)
        window = _string(args, "window", required=False)
        roots = self._roots()
        if window:
            if window not in roots:
                raise ControlError("not_found", f"Unknown window: {window}")
            roots = {window: roots[window]}
        return {"matches": find_widgets(roots, object_name, class_name, text)}

    def _cmd_ui_click(self, args):
        from ui.control_surface import click_widget, resolve_path
        path = _string(args, "path")
        widget = resolve_path(self._roots(), path)
        return click_widget(widget,
                            asynchronous=_boolean(args, "async", default=False))

    def _cmd_ui_set(self, args):
        from ui.control_surface import resolve_path, set_widget_value
        path = _string(args, "path")
        if "value" not in args:
            raise ControlError("invalid_argument", "value is required")
        widget = resolve_path(self._roots(), path)
        return {"path": path, "value": set_widget_value(widget, args["value"])}

    def _cmd_ui_grab(self, args):
        from ui.control_surface import grab_widget, resolve_path
        output = _string(args, "output")
        path = _string(args, "path", required=False)
        roots = self._roots()
        widget = roots["main"] if not path else resolve_path(roots, path)
        return grab_widget(widget, output)

    def _cmd_ui_context(self, args):
        from ui.control_surface import root_context
        context = root_context(self.window)
        context["view"] = _view_state(self.window)
        context["stack"] = _stack_info(self.window)
        context["render"] = _render_state(self.window)
        context["dialogs"] = self._dialog_list()
        return context

    def _cmd_ui_focus(self, args):
        from ui.control_surface import focus_widget, resolve_path
        path = _string(args, "path")
        widget = resolve_path(self._roots(), path)
        return {"path": path, **focus_widget(widget)}

    def _cmd_ui_menu(self, args):
        """List or trigger actions in the open context menu."""
        menu = QApplication.activePopupWidget()
        if not isinstance(menu, QMenu):
            candidates = [widget for widget in QApplication.topLevelWidgets()
                          if isinstance(widget, QMenu) and widget.isVisible()]
            menu = candidates[-1] if candidates else None
        if menu is None:
            raise ControlError("not_found", "No context menu is open")
        actions = []
        for index, action in enumerate(menu.actions()):
            actions.append({
                "index": index,
                "text": action.text(),
                "enabled": bool(action.isEnabled()),
                "separator": bool(action.isSeparator()),
                "checkable": bool(action.isCheckable()),
                "checked": bool(action.isChecked()),
                "shortcut": action.shortcut().toString(),
            })
        if "action" not in args and "index" not in args:
            return {"title": menu.title(), "actions": actions}
        action = args.get("action")
        if action is not None:
            candidates = [item for item in menu.actions()
                          if item.text().casefold() == str(action).casefold()]
            if not candidates:
                raise ControlError("not_found", f"No menu action: {action}")
            selected = candidates[0]
        else:
            index = _integer(args, "index", required=True, minimum=0)
            if index >= len(menu.actions()):
                raise ControlError("not_found", f"No menu action at {index}")
            selected = menu.actions()[index]
        if not selected.isEnabled() or selected.isSeparator():
            raise ControlError("busy", "Menu action is not usable")
        from PyQt6 import sip

        def fire(action=selected):
            if not sip.isdeleted(action):
                from utils.file_dialogs import control_picker_context
                with control_picker_context():
                    action.trigger()

        # Delete / Save As open modal dialogs inside trigger(); the reply must
        # reach the client before that nested event loop starts.
        QTimer.singleShot(100, fire)
        return {"scheduled": True, "triggered": selected.text(),
                "index": menu.actions().index(selected)}

    def _cmd_ui_mouse(self, args):
        from ui.control_surface import resolve_path, send_mouse
        steps = _integer(args, "steps", required=True, default=12, minimum=1)
        if steps > MAX_MOUSE_STEPS:
            raise ControlError("invalid_argument", f"steps must be between 1 and {MAX_MOUSE_STEPS}")
        path = _string(args, "path")
        action = _string(args, "action", choices=(
            "click", "double_click", "right_click", "middle_click",
            "press", "release", "move", "drag", "wheel"))
        widget = resolve_path(self._roots(), path)
        return send_mouse(
            widget, action,
            x=_number(args, "x", default=0),
            y=_number(args, "y", default=0),
            to_x=_number(args, "to_x"),
            to_y=_number(args, "to_y"),
            steps=steps,
            button=args.get("button", "left"),
            modifiers=args.get("modifiers"),
            delta=_number(args, "delta", default=-120),
        )

    def _cmd_ui_key(self, args):
        from ui.control_surface import resolve_path, send_key
        path = _string(args, "path", required=False)
        widget = (resolve_path(self._roots(), path) if path
                  else QApplication.focusWidget())
        if widget is None:
            raise ControlError("not_found", "No widget has focus")
        action = _string(args, "action", required=False, default="click",
                         choices=("click", "press", "release"))
        key = args.get("key")
        text = args.get("text")
        if key is None and text is None:
            raise ControlError("invalid_argument", "Provide key or text")
        return send_key(widget, key=key, text=text, action=action,
                        modifiers=args.get("modifiers"))

    def _cmd_ui_type(self, args):
        from ui.control_surface import resolve_path, send_key
        text = _string(args, "text", required=False)
        if text is None or not isinstance(text, str):
            raise ControlError("invalid_argument", "text is required")
        path = _string(args, "path", required=False)
        widget = (resolve_path(self._roots(), path) if path
                  else QApplication.focusWidget())
        if widget is None:
            raise ControlError("not_found", "No widget has focus")
        result = send_key(widget, text=text)
        if _boolean(args, "submit", default=False):
            send_key(widget, key="Enter")
            result["submitted"] = True
        return result
