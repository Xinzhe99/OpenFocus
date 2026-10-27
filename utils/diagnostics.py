"""Exception/crash capture for executable entry points, independent of Qt.

Importing this module never replaces process hooks. Embedders retain control;
GUI/CLI launchers explicitly install diagnostics and can restore previous hooks.
"""
import atexit
import datetime
import faulthandler
from importlib import metadata
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import traceback

from constants import APP_VERSION
from utils.logging_setup import get_logger, session_id, session_log_path

_logger = get_logger(__name__)
_lock = threading.RLock()
_hooks = None
_crash_stream = None
_crash_path = None
_owns_faulthandler = False
_atexit_registered = False


def crash_log_path():
    """The actual crash report, including a temporary-directory fallback."""
    return _crash_path


def system_facts() -> dict:
    """Collect versions without importing optional GUI/ML packages."""
    facts = {
        "OpenFocus": APP_VERSION,
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
    from utils.platform_utils import has_compute_gpu
    facts["ram"] = (
        f"{memory.total_memory_bytes() / 1024 ** 3:.1f} GB total, "
        f"{memory.available_memory_bytes() / 1024 ** 3:.1f} GB free")
    facts["gpu (cuda/mps)"] = has_compute_gpu()
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


def install_exception_hooks():
    """Capture full tracebacks and native faults once; preserve console hooks.

    A pre-existing faulthandler belongs to the host and is left intact. If the
    log folder is unwritable, the crash report uses a unique temp filename.
    """
    global _hooks, _crash_stream, _crash_path
    global _owns_faulthandler, _atexit_registered
    with _lock:
        if _hooks is not None:
            return _crash_path
        _crash_path = None
        target = session_log_path().with_suffix(".crash.txt")
        for path in (target, Path(tempfile.gettempdir()) / target.name):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                _crash_stream = path.open("a", encoding="utf-8")
                _crash_path = path
                break
            except OSError:
                continue
        if _crash_stream is not None and not faulthandler.is_enabled():
            try:
                faulthandler.enable(_crash_stream, all_threads=True)
                _owns_faulthandler = True
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
    global _hooks, _crash_stream, _owns_faulthandler
    with _lock:
        if _hooks is not None:
            _hooks.restore()
            _hooks = None
        if _owns_faulthandler:
            faulthandler.disable()
            _owns_faulthandler = False
        if _crash_stream is not None:
            _crash_stream.close()
            _crash_stream = None
