"""Shared application-data locations without GUI or third-party imports."""
import os
import sys
from pathlib import Path
from typing import Iterator

_PORTABLE_MARKER = "OpenFocus.portable"
_WINDOWS_PREVIEW_DIR = "OpenFocus Preview"


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


def _windows_roaming_data_dir() -> Path:
    return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "OpenFocus"


def shared_account_device_dir(profile: Path) -> Path | None:
    """Return the Windows per-user device root for normal installs and previews."""
    if sys.platform != "win32":
        return None
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    if os.environ.get("OPENFOCUS_DATA_DIR"):
        # Preview launchers only redirect their profile data. Other overrides
        # select isolated credentials and an isolated device identity.
        preview_root = local / _WINDOWS_PREVIEW_DIR
        try:
            parts = Path(os.path.abspath(profile)).relative_to(os.path.abspath(preview_root)).parts
        except ValueError:
            return None
        if len(parts) != 2 or parts[1].casefold() != "profile":
            return None
    return local / "OpenFocus"


def legacy_windows_account_dirs(profile: Path, shared_dir: Path) -> Iterator[Path]:
    """Find the active, normal and preview profiles that may hold former IDs."""
    yield profile
    roaming = _windows_roaming_data_dir()
    if roaming != profile:
        yield roaming
    # A release can be the first upgraded app to start after a preview was used.
    try:
        previews = sorted((shared_dir.parent / _WINDOWS_PREVIEW_DIR).iterdir(),
                          key=lambda path: path.name.casefold())
    except (FileNotFoundError, NotADirectoryError):
        return
    for preview in previews:
        candidate = preview / "profile"
        if candidate != profile and candidate.is_dir():
            yield candidate


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
        return _windows_roaming_data_dir()
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "OpenFocus"
