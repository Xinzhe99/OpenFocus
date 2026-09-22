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
WINDOW_GEOMETRY_KEY = "ui/window_geometry"
MAIN_SPLITTER_KEY = "ui/main_splitter"
RIGHT_SPLITTER_KEY = "ui/right_splitter"
DRAG_FORMAT_KEY = "export/drag_format"
LAST_UPDATE_CHECK_KEY = "updates/last_check_epoch"
LAST_STACK_FOLDER_KEY = "ui/last_stack_folder"
LAST_DIALOG_DIR_KEY = "ui/last_dialog_dir"

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
    "align_cache_enabled": ("rendering/align_cache_enabled", True, "bool"),
    "restore_last_stack": ("ui/restore_last_stack", False, "bool"),
}


_settings_singleton = None


def get_settings() -> QSettings:
    """Process-wide QSettings singleton.

    A fresh instance per call risks losing writes when the temporary Python
    wrapper is garbage-collected before Qt syncs the backing store.
    """
    global _settings_singleton
    if _settings_singleton is None:
        _settings_singleton = QSettings(
            QSettings.Format.IniFormat,
            QSettings.Scope.UserScope,
            ORGANIZATION,
            APPLICATION,
        )
    return _settings_singleton


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


def save_window_layout(window) -> None:
    """Persist window geometry and splitter positions (call on close)."""
    settings = get_settings()
    try:
        settings.setValue(WINDOW_GEOMETRY_KEY, window.saveGeometry())
        if hasattr(window, "main_splitter"):
            settings.setValue(MAIN_SPLITTER_KEY, window.main_splitter.saveState())
        if hasattr(window, "right_splitter") and window.right_splitter is not None:
            settings.setValue(RIGHT_SPLITTER_KEY, window.right_splitter.saveState())
    except Exception:
        pass


def restore_window_layout(window) -> None:
    """Restore window geometry and splitter positions saved on last close."""
    settings = get_settings()
    try:
        geo = settings.value(WINDOW_GEOMETRY_KEY)
        if geo is not None:
            window.restoreGeometry(geo)
        main_state = settings.value(MAIN_SPLITTER_KEY)
        if main_state is not None and hasattr(window, "main_splitter"):
            window.main_splitter.restoreState(main_state)
        right_state = settings.value(RIGHT_SPLITTER_KEY)
        if right_state is not None and getattr(window, "right_splitter", None) is not None:
            window.right_splitter.restoreState(right_state)
    except Exception:
        pass


def get_drag_export_format(window) -> str:
    """Format extension used when dragging results out ('.jpg' default)."""
    value = getattr(window, "drag_export_format", None)
    if value in (".jpg", ".png", ".webp", ".tif", ".tiff"):
        return ".tif" if value == ".tiff" else value
    return ".jpg"


def set_drag_export_format(window, fmt: str) -> None:
    fmt = fmt.lower()
    if fmt == ".tiff":
        fmt = ".tif"
    if fmt not in (".jpg", ".png", ".tif"):
        return
    window.drag_export_format = fmt
    get_settings().setValue(DRAG_FORMAT_KEY, fmt)


def get_last_dialog_dir() -> str:
    value = str(get_settings().value(LAST_DIALOG_DIR_KEY, "") or "")
    return value if value and os.path.isdir(value) else ""


def set_last_dialog_dir(path: str) -> None:
    """Remember the folder used for the last file dialog (stores the dir
    component when given a file path)."""
    folder = os.path.dirname(path) if os.path.isfile(path) else path
    if folder and os.path.isdir(folder):
        get_settings().setValue(LAST_DIALOG_DIR_KEY, folder)


def load_drag_export_format(window) -> None:
    settings = get_settings()
    fmt = str(settings.value(DRAG_FORMAT_KEY, ".jpg") or ".jpg").lower()
    window.drag_export_format = fmt if fmt in (".jpg", ".png", ".webp", ".tif") else ".jpg"


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
