"""Exception/crash capture for executable entry points, independent of Qt.

Importing this module never replaces process hooks. Embedders retain control;
GUI/CLI launchers explicitly install diagnostics and can restore previous hooks.
"""
import atexit
import datetime
import faulthandler
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import sys
import stat
import tempfile
import threading
import time
import traceback

from constants import app_display_version
from utils.logging_setup import get_logger, session_id, session_log_path
from utils import logging_setup

_logger = get_logger(__name__)
_lock = threading.RLock()
_hooks = None
_crash_stream = None
_crash_path = None
_owns_faulthandler = False
# True when we redirected a faulthandler that PYTHONFAULTHANDLER / -X
# faulthandler had already enabled on stderr; shutdown hands it back.
_restore_stderr_faulthandler = False
_atexit_registered = False


def crash_log_path():
    """The actual crash report, including a temporary-directory fallback."""
    return _crash_path


def fallback_crash_dir():
    """Return this profile's private temporary crash directory, or None."""
    try:
        profile = os.path.normcase(os.path.realpath(os.fspath(logging_setup.logs_dir())))
        key = hashlib.sha256(os.fsencode(profile)).hexdigest()
        root = Path(tempfile.gettempdir()) / 'openfocus-crashes'
        directory = root / key
        for path in (root, directory):
            try:
                path.mkdir(mode=0o700)
            except FileExistsError:
                pass
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode):
                return None
            if os.name != 'nt' and (info.st_uid != os.getuid() or info.st_mode & 0o077):
                return None
        _prune_fallback_crashes(directory)
        return directory
    except OSError:
        return None


def _prune_fallback_crashes(directory):
    cutoff = time.time() - logging_setup.RETENTION_DAYS * 86400
    active = {os.getpid(): True}
    try:
        for path in directory.iterdir():
            match = logging_setup._SESSION_FILE.fullmatch(path.name)
            if not match or not path.name.endswith('.crash.txt') or path.is_symlink():
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                pid = int(match.group(1))
                if pid not in active:
                    active[pid] = logging_setup._process_alive(pid)
                if not active[pid]:
                    path.unlink()
            except OSError:
                continue
    except OSError:
        pass


def system_facts() -> dict:
    """Collect versions without importing optional GUI/ML packages."""
    facts = {
        "OpenFocus": app_display_version(),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "session": session_id(),
        "pid": os.getpid(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "frozen": bool(getattr(sys, "frozen", False)),
        "cpu_count": os.cpu_count(),
        "log": str(session_log_path()),
    }
    from utils import memory
    from utils.platform_utils import cached_compute_gpu
    facts["ram"] = (
        f"{memory.total_memory_bytes() / 1024 ** 3:.1f} GB total, "
        f"{memory.available_memory_bytes() / 1024 ** 3:.1f} GB free")
    gpu = cached_compute_gpu()
    facts["gpu (neural acceleration)"] = gpu if gpu is not None else "not checked"
    distributions = (
        ("numpy", "numpy", "__version__"),
        ("opencv-python", "cv2", "__version__"),
        ("opencv-python-headless", "cv2", "__version__"),
        ("scipy", "scipy", "__version__"),
        ("torch", "torch", "__version__"),
        ("PyQt6", "PyQt6.QtCore", "PYQT_VERSION_STR"),
        ("dtcwt", "dtcwt", "__version__"),
        ("Pillow", "PIL", "__version__"),
    )
    for distribution, module, attribute in distributions:
        try:
            facts[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            # Frozen builds may omit dist-info while bundling the library.
            loaded_version = getattr(sys.modules.get(module), attribute, None)
            facts[distribution] = (str(loaded_version) if loaded_version is not None
                                   else "version unavailable")
    return facts


def _report(kind, exc_info):
    from utils.logging_setup import mark_session_error
    mark_session_error()
    _logger.critical("%s", kind, exc_info=exc_info)
    with _lock:
        if _crash_stream is not None:
            try:
                _crash_stream.write(
                    f"\n{datetime.datetime.now().isoformat()} session={session_id()} {kind}\n")
                traceback.print_exception(*exc_info, file=_crash_stream)
                _crash_stream.flush()
            except (OSError, ValueError):
                pass  # File logging remains the other diagnostic sink.


class _ExceptionHooks:
    """One installation owns its predecessor chain, even after deactivation."""

    def __init__(self):
        self.targets = ((sys, "excepthook"), (threading, "excepthook"),
                        (sys, "unraisablehook"))
        self.previous = tuple(getattr(module, name) for module, name in self.targets)
        self.callbacks = (self.main_exception, self.thread_exception,
                          self.unraisable_exception)
        self.active = True

    def install(self):
        for (module, name), callback in zip(self.targets, self.callbacks):
            setattr(module, name, callback)

    def restore(self):
        self.active = False
        for (module, name), callback, previous in zip(
                self.targets, self.callbacks, self.previous):
            if getattr(module, name) is callback:
                setattr(module, name, previous)

    def main_exception(self, exc_type, value, tb):
        if self.active and not issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            _report("unhandled exception", (exc_type, value, tb))
        self.previous[0](exc_type, value, tb)

    def thread_exception(self, args):
        if self.active and args.exc_type is not SystemExit:
            name = args.thread.name if args.thread is not None else "unknown"
            _report(f"unhandled thread exception: {name}",
                    (args.exc_type, args.exc_value, args.exc_traceback))
        self.previous[1](args)

    def unraisable_exception(self, args):
        if self.active:
            _report(f"unraisable exception: {args.err_msg or 'object finalizer'}",
                    (args.exc_type, args.exc_value, args.exc_traceback))
        self.previous[2](args)


def install_exception_hooks(claim_inherited_faulthandler: bool = False):
    """Capture full tracebacks and native faults once; preserve console hooks.

    A pre-existing faulthandler belongs to the host and is left intact. Only
    an executable entry point that owns the process passes
    ``claim_inherited_faulthandler=True``: a handler enabled by
    PYTHONFAULTHANDLER / ``-X faulthandler`` (often inherited from a parent
    launcher) only points at stderr, which may be discarded, so it is
    redirected to the crash file and handed back to stderr on shutdown.
    If the log folder is unwritable, use a profile-specific temporary directory.
    """
    global _hooks, _crash_stream, _crash_path
    global _owns_faulthandler, _restore_stderr_faulthandler, _atexit_registered
    with _lock:
        if _hooks is not None:
            return _crash_path
        _crash_path = None
        target = session_log_path().with_suffix(".crash.txt")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _crash_stream = target.open("a", encoding="utf-8")
            _crash_path = target
        except OSError:
            directory = fallback_crash_dir()
            if directory is not None:
                try:
                    _crash_path = directory / target.name
                    _crash_stream = _crash_path.open("a", encoding="utf-8")
                except OSError:
                    _crash_path = None
        was_enabled = faulthandler.is_enabled()
        inherited = was_enabled and claim_inherited_faulthandler and bool(
            os.environ.get("PYTHONFAULTHANDLER")
            or "faulthandler" in getattr(sys, "_xoptions", {}))
        if _crash_stream is not None and (inherited or not was_enabled):
            try:
                faulthandler.enable(_crash_stream, all_threads=True)
                _owns_faulthandler = True
                _restore_stderr_faulthandler = inherited
            except (OSError, RuntimeError):
                _logger.warning("native crash capture unavailable", exc_info=True)
        _hooks = _ExceptionHooks()
        _hooks.install()
        if not _atexit_registered:
            atexit.register(shutdown_exception_hooks)
            _atexit_registered = True
        _logger.info("exception diagnostics active: %s", _crash_path)
        return _crash_path


def shutdown_exception_hooks() -> None:
    """Restore only hooks still owned by us and close the native fault file."""
    global _hooks, _crash_stream, _owns_faulthandler, _restore_stderr_faulthandler
    with _lock:
        if _hooks is not None:
            _hooks.restore()
            _hooks = None
        if _owns_faulthandler:
            faulthandler.disable()
            if _restore_stderr_faulthandler:
                # Return the pre-install state the env flag configured.
                try:
                    if sys.__stderr__ is not None:
                        faulthandler.enable(sys.__stderr__, all_threads=True)
                except (OSError, RuntimeError, ValueError):
                    pass
            _owns_faulthandler = False
            _restore_stderr_faulthandler = False
        if _crash_stream is not None:
            _crash_stream.close()
            _crash_stream = None
