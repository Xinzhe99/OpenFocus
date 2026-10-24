"""Application file logging: rotating files under the user's app-data dir.

The packaged (windowed) build has no console, so this is the only place
diagnostics survive a crash. Attach the newest openfocus.log when reporting
issues.
"""
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from constants import APP_VERSION

LOG_DIR_NAME = "logs"
LOG_FILE_NAME = "openfocus.log"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3

_logger: logging.Logger = logging.getLogger("openfocus")


def logs_dir() -> Path:
    """Logs next to the exe in portable mode; user profile dir otherwise."""
    from utils.settings_store import is_portable_mode, portable_data_dir
    if is_portable_mode():
        return Path(portable_data_dir()) / LOG_DIR_NAME
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "OpenFocus" / LOG_DIR_NAME


def setup_logging() -> Path:
    """Install the rotating file handler; safe to call multiple times."""
    log_dir = logs_dir()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return log_dir

    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in _logger.handlers):
        handler = logging.handlers.RotatingFileHandler(
            log_dir / LOG_FILE_NAME,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
        _logger.info("OpenFocus %s starting (python %s)",
                     APP_VERSION, sys.version.split()[0])
    return log_dir


def get_logger(name: str = "openfocus") -> logging.Logger:
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Session logs, user-action logging and feedback export
# ---------------------------------------------------------------------------
import datetime
import zipfile

_session_marker = None


def _session_stamp() -> str:
    global _session_marker
    if _session_marker is None:
        _session_marker = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _session_marker


def session_log_path() -> Path:
    """The file holding THIS session's lines (the rotating main file)."""
    return logs_dir() / LOG_FILE_NAME


def log_session_start() -> None:
    """Write a session banner with the facts needed to reproduce a report."""
    _logger.info("================ session %s ================",
                 _session_stamp())
    _logger.info("version=%s python=%s platform=%s",
                 APP_VERSION, sys.version.split()[0], sys.platform)
    try:
        from utils import memory
        _logger.info("ram total=%.1f GB available=%.1f GB",
                     memory.total_memory_bytes() / 1024 ** 3,
                     memory.available_memory_bytes() / 1024 ** 3)
        import shutil
        _logger.info("gpu_hint=%s", bool(
            shutil.which("nvidia-smi")) and "nvidia-smi present" or "none")
    except Exception:
        pass
    try:
        import multiprocessing
        _logger.info("cores=%s", multiprocessing.cpu_count())
    except Exception:
        pass


def log_action(event: str, **details) -> None:
    """Record one user action with its parameters.

    The goal is reproducibility: a feedback report should show exactly what
    the user did (loaded what, rendered with which method and settings,
    exported where) without guessing.
    """
    try:
        if details:
            pretty = " ".join(f"{k}={v}" for k, v in details.items())
            _logger.info("action %s %s", event, pretty)
        else:
            _logger.info("action %s", event)
    except Exception:
        pass


def all_log_files():
    """Every log file (current + rotated backups), newest first."""
    directory = logs_dir()
    if not directory.is_dir():
        return []
    files = sorted(directory.glob("openfocus.log*"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    return files


def session_log_excerpt(max_bytes: int = 60_000) -> str:
    """The tail of the current session's log, for embedding in a report."""
    path = session_log_path()
    try:
        data = path.read_bytes()
    except OSError:
        return "(log unavailable)"
    marker = f"================ session {_session_stamp()}".encode()
    start = data.rfind(marker)
    if start > 0:
        data = data[start:]
    if len(data) > max_bytes:
        head = ("...(older lines trimmed — export the full log for the "
                "complete record)...\n").encode()
        data = head + data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


def export_feedback_package(dest_path, description: str = "",
                            include_all_logs: bool = True) -> Path:
    """Zip the session log (+ full logs) and system facts for a report."""
    import platform

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    system_facts = [
        f"OpenFocus {APP_VERSION}",
        f"date: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"platform: {platform.platform()}",
        f"python: {sys.version.split()[0]}",
    ]
    try:
        from utils import memory
        system_facts.append(
            f"ram: {memory.total_memory_bytes() / 1024 ** 3:.1f} GB total, "
            f"{memory.available_memory_bytes() / 1024 ** 3:.1f} GB free")
    except Exception:
        pass
    try:
        from utils.platform_utils import has_compute_gpu
        system_facts.append(f"gpu (cuda/mps): {has_compute_gpu()}")
    except Exception:
        pass

    import tempfile
    fd, temporary = tempfile.mkstemp(prefix=".feedback-", suffix=".zip", dir=dest.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("description.txt", description or "")
            zf.writestr("system.txt", "\n".join(system_facts) + "\n")
            zf.writestr("session_log.txt", session_log_excerpt(200_000))
            if include_all_logs:
                for log_file in all_log_files():
                    if log_file.is_file():
                        arcname = f"logs/{log_file.name}"
                        if arcname not in zf.namelist():
                            zf.write(log_file, arcname)
        os.replace(temporary, dest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return dest
