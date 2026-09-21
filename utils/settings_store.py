"""QSettings-backed persistence for user preferences and recent files.

All user-tunable state on the main window (thread count, tile parameters,
registration downscale, StackMFF-V4 batch size, GPU toggle, UI language,
recently opened stacks) survives restarts through this module.
"""
import os
from typing import Any

from PyQt6.QtCore import QSettings

from locales import trans

ORGANIZATION = "OpenFocus"
APPLICATION = "OpenFocus"
MAX_RECENT_FILES = 8

RECENT_FILES_KEY = "recent/files"
LANGUAGE_KEY = "ui/language"

# window attribute -> (settings key, default, cast)
_PERSISTED_FIELDS = {
    "thread_count": ("rendering/thread_count", 4, int),
    "tile_enabled": ("rendering/tile_enabled", True, "bool"),
    "tile_block_size": ("rendering/tile_block_size", 1024, int),
    "tile_overlap": ("rendering/tile_overlap", 256, int),
    "tile_threshold": ("rendering/tile_threshold", 2048, int),
    "reg_downscale_width": ("registration/downscale_width", 1024, int),
    "stackmffv4_batch_size": ("rendering/stackmffv4_batch_size", 2, int),
    "use_gpu": ("rendering/use_gpu", True, "bool"),
}


def get_settings() -> QSettings:
    return QSettings(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        ORGANIZATION,
        APPLICATION,
    )


def _cast(raw: Any, kind: Any, default: Any) -> Any:
    try:
        if kind == "bool":
            if isinstance(raw, bool):
                return raw
            return str(raw).strip().lower() in ("true", "1", "yes", "on")
        return kind(raw)
    except (TypeError, ValueError):
        return default


def load_window_settings(window) -> None:
    """Restore persisted preferences onto window attributes (defaults kept)."""
    settings = get_settings()
    for attr, (key, default, kind) in _PERSISTED_FIELDS.items():
        if settings.contains(key):
            setattr(window, attr, _cast(settings.value(key), kind, default))
        else:
            setattr(window, attr, getattr(window, attr, default))

    raw_recent = settings.value(RECENT_FILES_KEY, []) or []
    if not isinstance(raw_recent, list):
        raw_recent = [raw_recent]
    window.recent_files = [str(p) for p in raw_recent][:MAX_RECENT_FILES]


def save_window_settings(window) -> None:
    """Write current window preferences (including language) to QSettings."""
    settings = get_settings()
    for attr, (key, default, _kind) in _PERSISTED_FIELDS.items():
        settings.setValue(key, getattr(window, attr, default))
    settings.setValue(LANGUAGE_KEY, trans.current_lang)
    settings.setValue(RECENT_FILES_KEY, list(getattr(window, "recent_files", [])))


def get_saved_language():
    """Return the persisted language code ('en'/'zh') or None for auto-detect."""
    settings = get_settings()
    if not settings.contains(LANGUAGE_KEY):
        return None
    lang = str(settings.value(LANGUAGE_KEY, "")).lower()
    return lang if lang in ("en", "zh") else None


def add_recent_file(window, path: str) -> None:
    """Track a successfully opened stack; dedupe, cap and persist immediately."""
    path = os.path.abspath(path)
    recent = list(getattr(window, "recent_files", []))
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    window.recent_files = recent[:MAX_RECENT_FILES]
    get_settings().setValue(RECENT_FILES_KEY, list(window.recent_files))
    if hasattr(window, "rebuild_recent_menu"):
        window.rebuild_recent_menu()
