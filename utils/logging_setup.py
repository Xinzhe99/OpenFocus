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
    """%APPDATA%/OpenFocus/logs on Windows, ~/Library/Application Support/... on macOS."""
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
