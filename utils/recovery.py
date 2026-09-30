"""Crash auto-recovery: session snapshots plus a dirty-exit lock.

A project snapshot (.ofproj) is rewritten after every meaningful state
change (stack loaded, render finished) into a fixed recovery slot next to
the user's settings, and a lock file marks the session as live. A lock
left behind at startup means the previous session died without a clean
closeEvent (crash, kill, power loss) — the user is then offered a
one-click restore of that snapshot.
"""
import os
from typing import Optional

RECOVERY_DIR_NAME = "recovery"
SESSION_FILE = "session.ofproj"
LOCK_FILE = "session.lock"


def _base_dir() -> str:
    """Directory holding settings — portable-aware, so recovery data
    travels with a portable install exactly like the settings do."""
    from utils.settings_store import get_settings, is_portable_mode, portable_data_dir
    if is_portable_mode():
        return portable_data_dir()
    ini = get_settings().fileName()
    return os.path.dirname(str(ini))


def recovery_dir() -> str:
    return os.path.join(_base_dir(), RECOVERY_DIR_NAME)


def session_path() -> str:
    return os.path.join(recovery_dir(), SESSION_FILE)


def lock_path() -> str:
    return os.path.join(recovery_dir(), LOCK_FILE)


def init_lock() -> None:
    """Create the recovery dir and mark the session as live."""
    try:
        os.makedirs(recovery_dir(), exist_ok=True)
        with open(lock_path(), "w", encoding="utf-8") as f:
            f.write("live")
    except OSError:
        pass


def mark_clean_exit() -> None:
    """Remove the lock on a normal shutdown (snapshot is kept)."""
    try:
        if os.path.exists(lock_path()):
            os.remove(lock_path())
    except OSError:
        pass


def write_snapshot(window) -> bool:
    """Persist the current session to the recovery slot. Best-effort:
    never raises into callers (render/load handlers)."""
    if not getattr(window, "raw_images", None):
        return False
    try:
        from utils.project_file import save_project
        os.makedirs(recovery_dir(), exist_ok=True)
        ok, _err = save_project(window, session_path())
        return bool(ok)
    except Exception:
        return False


def pending_recovery() -> Optional[dict]:
    """Return a validated project state when the last session crashed.

    None when the last exit was clean, no snapshot exists, or its source
    files have since moved.
    """
    try:
        if not os.path.exists(lock_path()):
            return None
        if not os.path.isfile(session_path()):
            return None
        from utils.project_file import validate_project
        ok, _err, state = validate_project(session_path())
        if not ok:
            return None
        return state
    except Exception:
        return None


def snapshot_frame_count(state: Optional[dict]) -> int:
    if not state:
        return 0
    return len(state.get("resolved_sources") or state.get("sources") or [])
