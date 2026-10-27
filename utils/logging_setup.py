"""Process-isolated rotating logs and task diagnostics shared by GUI and CLI.

No Qt or algorithm imports: diagnostic setup must also work in headless builds.
"""
import datetime
import json
import logging
import logging.handlers
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
import zipfile

from constants import APP_VERSION
from utils.app_paths import user_data_dir

LOG_DIR_NAME = "logs"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3
RETENTION_DAYS = 14

_logger = logging.getLogger("openfocus")
_logger.addHandler(logging.NullHandler())
_lock = threading.RLock()
_handler = None
_directory = None
_session_marker = None
_session_started = False
_context = ContextVar("openfocus_operation", default=None)
_SESSION_FILE = re.compile(
    r"openfocus-\d{8}T\d{6}_\d{6}-(\d+)\.(?:log(?:\.\d+)?|crash\.txt)$")


def session_id() -> str:
    """Stable identity for this process, also used in diagnostic filenames."""
    global _session_marker
    with _lock:
        if _session_marker is None:
            stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S_%f")
            _session_marker = f"{stamp}-{os.getpid()}"
        return _session_marker


def logs_dir() -> Path:
    """Actual configured directory, or the portable/profile default."""
    if _directory is not None:
        return _directory
    override = os.environ.get("OPENFOCUS_LOG_DIR")
    return Path(override).expanduser() if override else user_data_dir() / LOG_DIR_NAME


def session_log_path() -> Path:
    return logs_dir() / f"openfocus-{session_id()}.log"


class _ContextFilter(logging.Filter):
    def filter(self, record):
        context = _context.get()
        record.session_id = session_id()
        if not hasattr(record, "operation_id"):
            record.operation_id = context.operation_id if context else "-"
        return True


def _process_alive(pid):
    """Conservatively keep logs from running processes, including on Windows."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() != 87  # INVALID_PARAMETER: PID absent
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _prune_old_logs(directory):
    cutoff = time.time() - RETENTION_DAYS * 86400
    active = {os.getpid(): True}
    try:
        for path in directory.iterdir():
            match = _SESSION_FILE.fullmatch(path.name)
            if not match or path.is_symlink():
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                pid = int(match.group(1))
                if pid not in active:
                    active[pid] = _process_alive(pid)
                if not active[pid]:
                    path.unlink()
            except OSError:
                continue  # Retention cannot prevent the application starting.
    except OSError:
        pass


def setup_logging(log_dir=None, level=None) -> Path:
    """Install one owned handler; repeated calls never duplicate log records.

    OPENFOCUS_LOG_DIR and OPENFOCUS_LOG_LEVEL override the defaults for support
    and development. An unwritable directory must not prevent image processing.
    """
    global _handler, _directory, _session_started
    with _lock:
        directory = Path(log_dir) if log_dir is not None else logs_dir()
        selected_level = level if level is not None else os.environ.get("OPENFOCUS_LOG_LEVEL", "INFO")
        if isinstance(selected_level, str):
            selected_level = logging.getLevelName(selected_level.upper())
        if not isinstance(selected_level, int):
            selected_level = logging.INFO
        if _handler is not None and directory == _directory:
            _logger.setLevel(selected_level)
            return directory
        shutdown_logging()
        _directory = directory
        try:
            directory.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                session_log_path(), maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
                encoding="utf-8")
        except OSError:
            return directory
        handler.addFilter(_ContextFilter())
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s pid=%(process)d thread=%(threadName)s "
            "session=%(session_id)s operation=%(operation_id)s %(name)s: %(message)s"))
        _logger.addHandler(handler)
        _handler = handler
        _logger.setLevel(selected_level)
        _logger.propagate = False
        _session_started = False
        _prune_old_logs(directory)
        return directory


def shutdown_logging() -> None:
    """Close only OpenFocus's handler; leave embedding/test handlers intact."""
    global _handler, _directory, _session_started
    with _lock:
        if _handler is not None:
            _logger.removeHandler(_handler)
            _handler.close()
            _handler = None
        _directory = None
        _session_started = False


def get_logger(name: str = "openfocus") -> logging.Logger:
    """Use module names without escaping the application's logger namespace."""
    if name != "openfocus" and not name.startswith("openfocus."):
        name = f"openfocus.{name}"
    return logging.getLogger(name)


def _event(event, details, *, level=logging.INFO, exc_info=None):
    context = _context.get()
    try:
        payload = json.dumps(details, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = '{"details_unavailable": true}'
    _logger.log(level, "event=%s %s", event, payload, exc_info=exc_info,
                extra={"event": event, "session_id": session_id(),
                       "operation_id": context.operation_id if context else "-"})


def log_action(event: str, **details) -> None:
    """Record a user action as unambiguous, escaped JSON fields."""
    _event(event, details)


def log_session_start() -> None:
    """Emit one banner even when multiple windows/entry points initialize."""
    global _session_started
    with _lock:
        if _session_started or _handler is None:
            return
        _session_started = True
        log_action("session.started", version=APP_VERSION,
                   python=sys.version.split()[0], platform=sys.platform,
                   frozen=bool(getattr(sys, "frozen", False)),
                   log_level=logging.getLevelName(_logger.level))


def frame_details(images) -> dict:
    """Describe a stack without copying, inspecting pixels or logging arrays."""
    try:
        count = len(images) if images is not None else 0
    except TypeError:
        return {"frame_count": None}
    details = {"frame_count": count}
    if count:
        first = images[0]
        shape = getattr(first, "shape", None)
        dtype = getattr(first, "dtype", None)
        details.update(first_shape=tuple(shape) if shape is not None else None,
                       first_dtype=str(dtype) if dtype is not None else None)
    return details


class _Operation:
    def __init__(self, name):
        from uuid import uuid4
        self.name = name
        self.operation_id = uuid4().hex[:12]
        self.status = "completed"
        self.exc_info = None

    def cancel(self):
        if self.status != "failed":
            self.status = "cancelled"

    def fail(self, error=None):
        self.status = "failed"
        info = ((type(error), error, error.__traceback__) if error is not None
                else sys.exc_info())
        if info[0] is not None:
            self.exc_info = info


@contextmanager
def log_operation(name: str, **details):
    """Correlate nested task logs; record one terminal outcome and elapsed time.

    Catching an error inside the scope requires operation.fail() in the except
    block. Cooperative cancellation requires operation.cancel(). Neither changes
    processing, signals, or exception propagation.
    """
    operation = _Operation(name)
    parent = _context.get()
    token = _context.set(operation)
    started = time.perf_counter()
    try:
        if parent is not None:
            details = {**details, "parent_operation_id": parent.operation_id}
        _event(f"{name}.started", details)
        try:
            yield operation
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                operation.cancel()
            elif isinstance(exc, SystemExit):
                operation.status = "completed" if exc.code in (None, 0) else "failed"
            elif operation.status != "cancelled":
                operation.fail()
            raise
        finally:
            _event(f"{name}.{operation.status}",
                   {"elapsed_ms": round((time.perf_counter() - started) * 1000, 3)},
                   level=logging.ERROR if operation.status == "failed" else logging.INFO,
                   exc_info=operation.exc_info)
    finally:
        operation.exc_info = None  # Do not retain image stacks via traceback frames.
        _context.reset(token)


def _flush():
    with _lock:
        if _handler is not None:
            _handler.flush()


def all_log_files():
    """Current/retained session logs, native crash reports and legacy logs."""
    directory = logs_dir()
    try:
        files = []
        for path in directory.iterdir():
            if (_SESSION_FILE.fullmatch(path.name) or re.fullmatch(r"openfocus\.log(?:\.\d+)?", path.name)) and path.is_file():
                try:
                    files.append((path.stat().st_mtime, path))
                except OSError:
                    continue
        return [path for _, path in sorted(files, key=lambda pair: pair[0], reverse=True)]
    except OSError:
        return []


def session_log_excerpt(max_bytes: int = 60_000) -> str:
    """Read a bounded tail across this process's rotations, newest lines last."""
    if max_bytes <= 0:
        return ""
    _flush()
    current = session_log_path()
    parts = []
    remaining = max_bytes
    trimmed = False
    for path in (current, *(Path(f"{current}.{i}") for i in range(1, BACKUP_COUNT + 1))):
        try:
            with path.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                count = min(size, remaining)
                if count:
                    stream.seek(-count, os.SEEK_END)
                    parts.append(stream.read(count))
                    remaining -= count
                trimmed |= size > count
        except OSError:
            continue
    if not parts:
        return "(log unavailable)"
    prefix = "...(older lines trimmed; export full logs for the complete record)...\n" if trimmed else ""
    return prefix + b"".join(reversed(parts)).decode("utf-8", errors="replace")


def export_feedback_package(dest_path, description: str = "",
                            include_all_logs: bool = True) -> Path:
    """Atomically export current diagnostics, tolerating vanished old logs."""
    from utils.diagnostics import crash_log_path, system_facts

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _flush()
    facts = system_facts()
    fd, temporary = tempfile.mkstemp(prefix=".feedback-", suffix=".zip", dir=dest.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("description.txt", description or "")
            archive.writestr("system.txt", "\n".join(f"{key}: {value}" for key, value in facts.items()) + "\n")
            archive.writestr("session_log.txt", session_log_excerpt(200_000))
            paths = all_log_files() if include_all_logs else []
            crash = crash_log_path()
            if crash is not None and crash not in paths:
                paths.append(crash)
            for path in paths:
                try:
                    data = path.read_bytes()
                except OSError:
                    continue  # Another process may have rotated/removed an old log.
                archive.writestr(f"logs/{path.name}", data)
        os.replace(temporary, dest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return dest
