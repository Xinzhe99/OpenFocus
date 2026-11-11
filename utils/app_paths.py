"""Shared application-data locations without GUI or third-party imports."""
import os
import sys
from pathlib import Path

_PORTABLE_MARKER = "OpenFocus.portable"


def is_portable_mode() -> bool:
    """True when a writable portable install has a marker beside the executable.

    Source checkouts look for the marker in the project root. A marker inside
    a read-only installation falls back to the user profile.
    """
    try:
        if getattr(sys, "frozen", False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        marker = os.path.join(base, _PORTABLE_MARKER)
        if os.path.isfile(marker):
            data_dir = os.path.join(base, "OpenFocusData")
            os.makedirs(data_dir, exist_ok=True)
            # Probe an actual write: the marker may be shipped in a read-only
            # installation, where settings must fall back to the user profile.
            probe = os.path.join(data_dir, ".write_test")
            try:
                with open(probe, "w") as f:
                    f.write("ok")
                os.remove(probe)
            except OSError:
                return False
            return True
    except Exception:
        pass
    return False


def portable_data_dir() -> str:
    """Directory holding portable user data (created on demand)."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(base, "OpenFocusData")
    os.makedirs(d, exist_ok=True)
    return d


def user_data_dir() -> Path:
    """Application data root, with an explicit override for isolated processes.

    Without the override, settings retain their normal QSettings location and
    recovery stays beside settings.
    """
    override = os.environ.get("OPENFOCUS_DATA_DIR")
    if override:
        return Path(override).expanduser().absolute()
    if is_portable_mode():
        return Path(portable_data_dir())
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "OpenFocus"
