"""Local command server that drives the live OpenFocus window.

Every command runs on the GUI thread, so it uses the same controllers and
dialogs the UI uses. Long operations return immediately; clients poll state.
"""
import json
import os
import secrets
import time

from PyQt6.QtCore import QObject, Qt, QTimer
from PyQt6.QtNetwork import QLocalServer

from constants import APP_VERSION
from core.control.protocol import (
    COMMANDS as _COMMANDS, ControlError, MAX_LINE, SCHEMA, SOCKET_PREFIX,
    clear_discovery, decode, encode, error_response, ok_response,
    write_discovery,
)
from core.algorithms import (
    fusion_algorithms, get_fusion, get_registration, registration_algorithms,
)
from locales import trans
from utils.logging_setup import get_logger

_log = get_logger(__name__)


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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControlError("invalid_argument", f"{key} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ControlError("invalid_argument", f"{key} must be >= {minimum}")
    return value


def _boolean(args, key, required=False, default=None):
    value = args.get(key, default)
    if value is None:
        if required:
            raise ControlError("invalid_argument", f"{key} is required")
        return None
    if not isinstance(value, bool):
        raise ControlError("invalid_argument", f"{key} must be a boolean")
    return value


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
    busy = bool(worker is not None and worker.isRunning())
    error = getattr(manager, "last_error", None)
    if busy:
        state = "rendering"
    elif error:
        state = "error"
    elif getattr(window, "fusion_result", None) is not None:
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


class ControlServer(QObject):
    def __init__(self, window, socket_name=None, token=None, parent=None):
        super().__init__(parent)
        self.window = window
        self.token = token or secrets.token_hex(16)
        self.socket_name = socket_name or f"{SOCKET_PREFIX}-{os.getpid()}"
        self._buffers = {}
        self._dialogs = {}
        self._server = QLocalServer(self)
        QLocalServer.removeServer(self.socket_name)
        if not self._server.listen(self.socket_name):
            raise RuntimeError(
                f"Could not listen on {self.socket_name}: "
                f"{self._server.errorString()}")
        self._server.newConnection.connect(self._accept)
        self.discovery = write_discovery(
            self.socket_name, self.token, os.getpid(), APP_VERSION)
        _log.info("control channel listening on %s", self.socket_name)

    # ------------------------------------------------------------- sockets
    def _accept(self):
        while self._server.hasPendingConnections():
            connection = self._server.nextPendingConnection()
            self._buffers[connection] = b""
            connection.readyRead.connect(lambda c=connection: self._read(c))
            connection.disconnected.connect(lambda c=connection: self._drop(c))

    def _drop(self, connection):
        self._buffers.pop(connection, None)
        connection.deleteLater()

    def _read(self, connection):
        buffer = self._buffers.get(connection, b"") + bytes(connection.readAll())
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if not line.strip():
                continue
            response = self.handle_line(line, token=self.token)
            connection.write(encode(response))
            connection.flush()
        if len(buffer) > MAX_LINE:
            connection.write(encode(error_response(
                "0", ControlError("bad_request", "Request line is too large"))))
            connection.flush()
            connection.disconnectFromServer()
            return
        self._buffers[connection] = buffer

    def close(self):
        for dialog in list(self._dialogs.values()):
            try:
                dialog.close()
            except RuntimeError:
                pass
        self._dialogs.clear()
        self._server.close()
        clear_discovery(os.getpid())

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
            result = handler(args)
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
        return {"pong": True, "version": APP_VERSION, "schema": SCHEMA}

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
            from utils.platform_utils import has_compute_gpu
            capabilities["gpu"] = bool(has_compute_gpu())
        except Exception:
            capabilities["gpu"] = None
        return {
            "version": APP_VERSION,
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
            },
            "render": _render_state(window),
            "outputs": {"count": window.output_list.count(),
                        "selected": window.output_list.currentRow()},
            "dialogs": self._dialog_list(),
        }

    def _cmd_app_quit(self, args):
        QTimer.singleShot(0, self.window.close)
        return {"quitting": True}

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
        manager = self.window.source_manager
        if getattr(manager, "_load_worker", None) is not None:
            raise ControlError("busy", "A stack is already loading",
                               "Poll stack.frames, then retry.")
        if folder:
            manager._start_load_worker(folder=folder, scale=scale, append=append)
            source = folder
        elif video:
            manager._start_load_worker(video=video, scale=scale, append=append)
            source = video
        else:
            manager._start_load_worker(filepaths=list(files), scale=scale,
                                       append=append)
            source = list(files)
        return {"loading": True, "source": source}

    def _cmd_stack_frames(self, args):
        return _stack_info(self.window)

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
        excluded = set(getattr(self.window, "excluded_frames", set()) or set())
        for index in indices:
            if not 0 <= index < len(self.window.raw_images):
                raise ControlError("invalid_argument", f"Frame index out of range: {index}")
            if include:
                excluded.discard(index)
            else:
                excluded.add(index)
        self.window.excluded_frames = excluded
        self.window.source_manager.decorate_source_items()
        return _stack_info(self.window)

    # -------------------------------------------------- method/align/options
    def _cmd_method_set(self, args):
        method = _string(args, "id")
        get_fusion(method)
        self.window.set_fusion_method(method)
        parameters = args.get("parameters")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ControlError("invalid_argument", "parameters must be an object")
            self.window.set_fusion_parameters(method, parameters)
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
        get_registration(method)
        self.window.set_alignment_method(method)
        parameters = args.get("parameters")
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise ControlError("invalid_argument", "parameters must be an object")
            self.window.set_registration_parameters(method, parameters)
        self.window.persist_settings()
        return {"id": method,
                "parameters": self.window.registration_parameters_for(method)}

    def _cmd_option_set(self, args):
        name = _string(args, "name")
        value = args.get("value")
        window = self.window
        if name == "base_correct":
            window.set_base_correct(bool(value))
        elif name == "exposure_match":
            window.set_exposure_match(bool(value))
        elif name == "reverse_order":
            window.set_reverse_order(bool(value))
        elif name == "thread_count":
            window.thread_count = max(0, int(value))
        elif name == "use_gpu":
            window.set_use_gpu(bool(value))
        elif name == "tile_enabled":
            window.tile_enabled = bool(value)
        elif name == "tile_block_size":
            window.tile_block_size = max(64, int(value))
        elif name == "tile_overlap":
            window.tile_overlap = max(0, int(value))
        elif name == "tile_threshold":
            window.tile_threshold = max(1, int(value))
        elif name == "preview":
            window.chk_quick_preview.setChecked(bool(value))
        elif name == "ai_batch_size":
            window.ai_batch_size = max(1, int(value))
        elif name == "resource_preference":
            choice = _string(args, "value", choices=("balanced", "responsive", "maximum"))
            window.resource_preference = choice
        elif name == "reg_downscale_width":
            window.reg_downscale_width = max(64, int(value))
        else:
            raise ControlError("invalid_argument", f"Unknown option: {name}")
        window.persist_settings()
        return {"name": name, "value": value}

    # ------------------------------------------------------------- render
    def _cmd_render_start(self, args):
        manager = self.window.render_manager
        worker = getattr(manager, "worker", None)
        if worker is not None and worker.isRunning():
            raise ControlError("busy", "A render is already running",
                               "Wait for render.state to leave 'rendering' or call render.cancel.")
        preview = _boolean(args, "preview", default=False)
        if preview:
            self.window.chk_quick_preview.setChecked(True)
        method = self.window.current_fusion_method()
        if method == "none":
            raise ControlError("invalid_argument", "Select a fusion method first")
        manager.start_render()
        return {"started": True, "method": method, "preview": preview}

    def _cmd_render_cancel(self, args):
        worker = getattr(self.window.render_manager, "worker", None)
        if worker is None:
            return {"cancelling": False}
        worker.is_cancelled = True
        return {"cancelling": True}

    def _cmd_render_state(self, args):
        return _render_state(self.window)

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
        manager._write_record(record, target, labels)
        manager._mark_record_exported(
            original, record.image, target,
            manager._annotation_state(record, labels))
        return {"path": target, "written": True, "index": index}

    def _cmd_output_delete(self, args):
        index = _integer(args, "index", required=True, minimum=0)
        window = self.window
        if window.output_list.item(index) is None:
            raise ControlError("not_found", f"No output at index {index}")
        window.output_manager._remove_row(index)
        return {"deleted": True, "index": index}

    # ------------------------------------------------------------- project
    def _cmd_project_save(self, args):
        from utils.project_file import save_project
        path = _string(args, "path")
        ok, error = save_project(self.window, path)
        if not ok:
            raise ControlError("invalid_argument", error or "Project was not saved")
        return {"saved": True, "path": path}

    def _cmd_project_open(self, args):
        from utils.project_file import apply_project, validate_project
        path = _string(args, "path")
        ok, error, state = validate_project(path)
        if not ok:
            raise ControlError("invalid_argument", error or "Project is invalid")
        apply_project(self.window, state)
        return {"loading": True, "path": path}

    # ------------------------------------------------------------- dialogs
    def _dialog_list(self):
        return [{"index": index,
                 "title": dialog.windowTitle(),
                 "class": type(dialog).__name__,
                 "visible": bool(dialog.isVisible()),
                 "modal": bool(dialog.isModal())}
                for index, (_name, dialog) in enumerate(self._dialogs.items())]

    def _open_dialog(self, name, dialog):
        existing = self._dialogs.get(name)
        if existing is not None:
            if existing.isVisible():
                existing.raise_()
                existing.activateWindow()
                return {"opened": True, "title": existing.windowTitle(),
                        "existing": True}
            self._dialogs.pop(name, None)
            existing.close()
        dialog.setModal(False)
        dialog.finished.connect(lambda _result, key=name: self._dialogs.pop(key, None))
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
        if name == "settings":
            from dialogs.advanced_settings import AdvancedSettingsDialog
            dialog = AdvancedSettingsDialog(window)
        elif name == "batch":
            from dialogs.batch import BatchProcessingDialog
            dialog = BatchProcessingDialog(window)
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
        entries = list(self._dialogs.items())
        if index is not None:
            if index >= len(entries):
                raise ControlError("not_found", f"No dialog at index {index}")
            return entries[index][1]
        if title is None:
            if len(entries) == 1:
                return entries[0][1]
            if not entries:
                raise ControlError("not_found", "No dialog is open")
            raise ControlError("invalid_argument",
                               "Several dialogs are open; provide index or title")
        dialog = next((value for _name, value in entries
                       if title.casefold() in value.windowTitle().casefold()), None)
        if dialog is None:
            raise ControlError("not_found", f"No dialog titled {title!r}")
        return dialog

    def _cmd_dialog_close(self, args):
        dialog = self._find_dialog(args)
        dialog.close()
        return {"closed": True}

    def _cmd_dialog_accept(self, args):
        dialog = self._find_dialog(args)
        dialog.accept()
        return {"accepted": True, "title": dialog.windowTitle()}

    def _cmd_dialog_reject(self, args):
        dialog = self._find_dialog(args)
        dialog.reject()
        return {"rejected": True, "title": dialog.windowTitle()}

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
        key = _string(args, "key")
        action = (getattr(self.window, "ui_objs", None) or {}).get(key)
        if action is None or not hasattr(action, "trigger"):
            raise ControlError("not_found", f"Unknown action: {key}")
        if not action.isEnabled():
            raise ControlError("busy", f"Action is disabled: {key}")
        QTimer.singleShot(0, action.trigger)
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
        return click_widget(widget)

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
