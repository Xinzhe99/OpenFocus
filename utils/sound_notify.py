"""Cross-platform non-blocking completion sound notifications for OpenFocus."""
from __future__ import annotations

import os
import sys


def is_headless_environment() -> bool:
    """True when running under offscreen/headless Qt or automated tests."""
    qpa = (os.environ.get("QT_QPA_PLATFORM") or "").strip().lower()
    if qpa in ("offscreen", "minimal", "minimalegl", "vnc"):
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    silent = (os.environ.get("OPENFOCUS_SILENT") or "").strip().lower()
    if silent in ("1", "true", "yes", "on"):
        return True
    return False


def _qt_beep() -> bool:
    try:
        from PyQt6.QtWidgets import QApplication
        if QApplication.instance() is not None:
            QApplication.beep()
            return True
    except Exception:
        pass
    return False


def play_completion_sound(enabled: bool = True, *, force: bool = False) -> bool:
    """Play a brief, non-blocking completion chime when enabled.

    Returns ``True`` when a sound backend was triggered and ``False`` when
    disabled, suppressed in headless/test mode, or unavailable.
    """
    if not enabled:
        return False
    if not force and is_headless_environment():
        return False

    if sys.platform == "win32":
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
            return True
        except Exception:
            return _qt_beep()

    return _qt_beep()
