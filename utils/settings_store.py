"""QSettings-backed persistence for user preferences and recent files.

All user-tunable state on the main window (thread count, tile parameters,
registration downscale, AI batch size and output mode, GPU toggle, UI language,
recently opened stacks) survives restarts through this module.
"""
import json
import math
import os
from typing import Any, Optional

from PyQt6.QtCore import QSettings

from core.algorithms import (fusion_algorithms, get_fusion, get_registration,
                             resolve_fusion_parameters, resolve_registration_parameters)
from locales import trans
from utils import app_paths
from utils.validators import normalize_kernel_size

ORGANIZATION = "OpenFocus"
APPLICATION = "OpenFocus"
MAX_RECENT_FILES = 8
MAX_RECENT_PROJECTS = 8

# First-run theme. Light reads better on a fresh install for most users; a
# saved preference always wins, so an existing dark-theme user stays dark.
DEFAULT_THEME = "light"

RECENT_FILES_KEY = "recent/files"
RECENT_PROJECTS_KEY = "recent/projects"
LANGUAGE_KEY = "ui/language"
WINDOW_GEOMETRY_KEY = "ui/window_geometry"
MAIN_SPLITTER_KEY = "ui/main_splitter"
RIGHT_SPLITTER_KEY = "ui/right_splitter"
DRAG_FORMAT_KEY = "export/drag_format"
LAST_UPDATE_CHECK_KEY = "updates/last_check_epoch"
# Release tag whose startup banner the user dismissed: the same
# version must not nag again, a newer one re-shows the banner.
UPDATE_BANNER_DISMISSED_KEY = "updates/banner_dismissed_tag"
LAST_STACK_FOLDER_KEY = "ui/last_stack_folder"
LAST_DIALOG_DIR_KEY = "ui/last_dialog_dir"
# The "AI runs on the CPU" warning is informational and repeat-per-launch
# annoying; once the user says "don't show again" it must stay gone.
AI_CPU_WARNING_DISMISSED_KEY = "rendering/ai_cpu_warning_dismissed"
# The per-method parameter memories are the authority for what a method
# renders with; the global exposure_match / base_frame_correct fields only
# mirror the method that was last touched. Each memory is stored as JSON so
# a restart no longer reverts the user's per-algorithm choices.
BRIGHTNESS_BY_METHOD_KEY = "fusion/brightness_by_method"
METHOD_PARAMETERS_KEY = "fusion/method_parameters"
REGISTRATION_PARAMETERS_KEY = "registration/parameters"
# Historical keys are read only at this boundary.
SMOOTHING_BY_METHOD_KEY = "fusion/smoothing_by_method"
KERNEL_BY_MODE_KEY = "fusion/kernel_by_mode"
BRIGHTNESS_OPTIONS = ("exposure_match", "base_correct")


# window attribute -> (settings key, default, cast)
_PERSISTED_FIELDS = {
    "thread_count": ("rendering/thread_count", 0, int),
    "resource_preference": ("rendering/resource_preference", "balanced", "str"),
    "tile_enabled": ("rendering/tile_enabled", True, "bool"),
    "tile_block_size": ("rendering/tile_block_size", 1024, int),
    "tile_overlap": ("rendering/tile_overlap", 256, int),
    "tile_threshold": ("rendering/tile_threshold", 2048, int),
    "reg_downscale_width": ("registration/downscale_width", 1024, int),
    "ai_batch_size": ("rendering/ai_batch_size", 2, int),
    "use_gpu": ("rendering/use_gpu", True, "bool"),
    "align_cache_enabled": ("rendering/align_cache_enabled", True, "bool"),
    "restore_last_stack": ("ui/restore_last_stack", False, "bool"),
    "ui_theme": ("ui/theme", DEFAULT_THEME, "str"),
    "scale_bar_enabled": ("export/scale_bar_enabled", False, "bool"),
    "scale_um_per_px_manual": ("export/scale_um_per_px_manual", 0.0, "float"),
    "scale_bar_position": ("export/scale_bar_position", "bottom-right", "str"),
    "scale_bar_target": ("export/scale_bar_target", "all", "str"),
    "scale_bar_color": ("export/scale_bar_color", "auto", "str"),
    "scale_bar_font_family": ("export/scale_bar_font_family", "", "str"),
    "scale_bar_font_size": ("export/scale_bar_font_size", 0.0, "float"),
    "scale_bar_font_bold": ("export/scale_bar_font_bold", False, "bool"),
    "scale_bar_text_align": ("export/scale_bar_text_align", "auto", "str"),
    "scale_bar_custom_x": ("export/scale_bar_custom_x", 0.95, "float"),
    "scale_bar_custom_y": ("export/scale_bar_custom_y", 0.95, "float"),
    "base_frame_correct": ("fusion/base_frame_correct", False, "bool"),
    "exposure_match": ("fusion/exposure_match", True, "bool"),
    "stack_reverse_order": ("fusion/stack_reverse_order", False, "bool"),
    "ask_load_scale": ("ui/ask_load_scale", True, "bool"),
    "load_scale_default": ("ui/load_scale_default", 1.0, "float"),
}


_settings_singleton = None


def get_settings() -> QSettings:
    """Process-wide QSettings singleton.

    A fresh instance per call risks losing writes when the temporary Python
    wrapper is garbage-collected before Qt syncs the backing store.
    Portable mode (marker file next to the executable) stores the INI next
    to the executable so settings travel with it.
    """
    global _settings_singleton
    if _settings_singleton is None:
        if app_paths.is_portable_mode():
            _settings_singleton = QSettings(
                os.path.join(app_paths.portable_data_dir(), "settings.ini"),
                QSettings.Format.IniFormat,
            )
        else:
            _settings_singleton = QSettings(
                QSettings.Format.IniFormat,
                QSettings.Scope.UserScope,
                ORGANIZATION,
                APPLICATION,
            )
    return _settings_singleton


def reset_settings_singleton() -> None:
    """Drop the cached QSettings so the next get_settings() re-reads disk.

    The singleton caches values in memory, so wiping the settings directory
    (or calling QSettings.setPath) after it exists has no effect on it — the
    test suite wiped its sandbox but kept reading the previous values, and
    then wrote them back. Tests must call this around such a wipe.
    """
    global _settings_singleton
    if _settings_singleton is not None:
        try:
            _settings_singleton.sync()
        except Exception:
            pass
    _settings_singleton = None


_STR_KINDS = {"str": str, "float": float, "int": int}


def _cast(raw: Any, kind: Any, default: Any) -> Any:
    try:
        if raw is None:
            return default
        if kind == "bool":
            if isinstance(raw, bool):
                return raw
            return str(raw).strip().lower() in ("true", "1", "yes", "on")
        # 字段表里的 kind 可能是类型也可能是其名字字符串（"str"/"float"）；
        # 直接 "str"(raw) 会 TypeError 被吞成 default——ui_theme、比例尺
        # 标定等所有字符串/浮点设置曾因此从未恢复过
        if isinstance(kind, str):
            kind = _STR_KINDS[kind]
        return kind(raw)
    except (TypeError, ValueError, KeyError):
        return default


# --- Per-method parameter memories -------------------------------------------
#
# The three helpers below are the single validation point for those memories:
# QSettings reads them at startup, project files reuse the same rules, and a
# hand-edited value must never reach a spin box or a render as-is. Anything
# malformed is dropped locally; one bad entry must not discard the rest.

def _loads_object(raw: Any) -> dict:
    """JSON object from *raw*, or {} for absent/malformed/non-object input."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _finite_number(value: Any) -> Optional[float]:
    """A finite, non-boolean number, else None (bools are ints in Python)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _brightness_methods():
    return {"none", *(spec.id for spec in fusion_algorithms())}


def brightness_memory_from_json(raw: Any) -> dict:
    """{(option, method): bool} from the stored JSON; malformed parts dropped."""
    nested = _loads_object(raw)
    memory = {}
    for option in BRIGHTNESS_OPTIONS:
        per_method = nested.get(option)
        if not isinstance(per_method, dict):
            continue
        for method, enabled in per_method.items():
            # bool(enabled) would turn the string "false" into True, which is
            # exactly the silent revert this persistence exists to prevent.
            if isinstance(method, str) and method in _brightness_methods() \
                    and isinstance(enabled, bool):
                memory[(option, method)] = enabled
    return memory


def brightness_memory_to_json(memory: Any) -> str:
    """Stable JSON text {option: {method: bool}} for the brightness memory."""
    nested = {option: {} for option in BRIGHTNESS_OPTIONS}
    for (option, method), enabled in (memory or {}).items():
        if option in BRIGHTNESS_OPTIONS and method in _brightness_methods() \
                and isinstance(enabled, bool):
            nested[option][method] = enabled
    return json.dumps(nested, sort_keys=True)


def parameter_memory_from_json(raw: Any, *, registration=False) -> dict:
    """Restore each valid declared field without losing adjacent good values."""
    get = get_registration if registration else get_fusion
    resolve = resolve_registration_parameters if registration else resolve_fusion_parameters
    memory = {}
    for method, values in _loads_object(raw).items():
        try:
            get(method)
        except (TypeError, ValueError):
            continue
        if not isinstance(values, dict):
            continue
        valid = {}
        for key, value in values.items():
            try:
                valid[key] = resolve(method, {key: value})[key]
            except (TypeError, ValueError):
                continue
        memory[method] = valid
    return memory


def parameter_memory_to_json(memory: Any, *, registration=False) -> str:
    """Serialize canonical dictionaries, refusing invalid new state."""
    get = get_registration if registration else get_fusion
    resolve = resolve_registration_parameters if registration else resolve_fusion_parameters
    values = {}
    for method, parameters in (memory or {}).items():
        get(method)
        if not isinstance(parameters, dict):
            raise ValueError(f"Invalid parameters for {method}")
        resolved = resolve(method, parameters)
        values[method] = {key: resolved[key] for key in parameters}
    return json.dumps(values, sort_keys=True, allow_nan=False)


_LEGACY_KERNEL_METHODS = {"guided": ("guided_filter",), "dct": ("dct",),
                          "gfg": ("gfgfgf",), "ofab": ("openfocus_a", "openfocus_b")}


def _legacy_kernel_memory(raw):
    memory = {}
    for mode, value in _loads_object(raw).items():
        if mode in _LEGACY_KERNEL_METHODS and isinstance(value, int) \
                and not isinstance(value, bool) and value > 0:
            method = _LEGACY_KERNEL_METHODS[mode][0]
            spec = next(p for p in get_fusion(method).parameters if p.key == "kernel_size")
            memory[mode] = normalize_kernel_size(min(value, spec.maximum))
    return memory


def _legacy_parameters(settings):
    """Migrate the shared OF kernel identically into every declaring method."""
    memory = {}
    kernels = _legacy_kernel_memory(settings.value(KERNEL_BY_MODE_KEY))
    for mode, methods in _LEGACY_KERNEL_METHODS.items():
        if mode in kernels:
            for method in methods:
                memory.setdefault(method, {})["kernel_size"] = kernels[mode]
    smoothing = _loads_object(settings.value(SMOOTHING_BY_METHOD_KEY))
    if not settings.contains(SMOOTHING_BY_METHOD_KEY):
        for old, methods in (("fusion/ofb_depth_smoothing", ("openfocus_b",)),
                             ("fusion/of_blend_smoothing", ("openfocus_a", "openfocus_c"))):
            raw = settings.value(old)
            if settings.contains(old) and not isinstance(raw, bool):
                value = _cast(raw, "float", None)
                smoothing.update({method: value for method in methods})
    for method, key in (("openfocus_a", "blend_smoothing"),
                        ("openfocus_b", "depth_smoothing"),
                        ("openfocus_c", "blend_smoothing")):
        value = _finite_number(smoothing.get(method))
        if value is not None:
            spec = next(p for p in get_fusion(method).parameters if p.key == key)
            memory.setdefault(method, {})[key] = min(max(value, spec.minimum), spec.maximum)
    if settings.contains("fusion/ai_mode"):
        memory.setdefault("stackmffv3", {})["mode"] = settings.value("fusion/ai_mode")
    raw_levels = settings.value("fusion/dtcwt_levels")
    if settings.contains("fusion/dtcwt_levels") and not isinstance(raw_levels, bool):
        memory.setdefault("dtcwt", {})["N"] = _cast(raw_levels, int, None)
    return parameter_memory_from_json(memory)


def load_parameter_memories(window) -> None:
    """New empty objects are authoritative; legacy scalars never revive them."""
    settings = get_settings()
    if settings.contains(BRIGHTNESS_BY_METHOD_KEY):
        window._brightness_by_method = brightness_memory_from_json(
            settings.value(BRIGHTNESS_BY_METHOD_KEY))
    window._fusion_parameters_by_method = (
        parameter_memory_from_json(settings.value(METHOD_PARAMETERS_KEY))
        if settings.contains(METHOD_PARAMETERS_KEY) else _legacy_parameters(settings))
    window._registration_parameters_by_method = parameter_memory_from_json(
        settings.value(REGISTRATION_PARAMETERS_KEY), registration=True)


def save_parameter_memories(window) -> None:
    """Write only registry parameters and separate common brightness state."""
    settings = get_settings()
    settings.setValue(BRIGHTNESS_BY_METHOD_KEY, brightness_memory_to_json(
        getattr(window, "_brightness_by_method", {})))
    settings.setValue(METHOD_PARAMETERS_KEY, parameter_memory_to_json(
        getattr(window, "_fusion_parameters_by_method", {})))
    settings.setValue(REGISTRATION_PARAMETERS_KEY, parameter_memory_to_json(
        getattr(window, "_registration_parameters_by_method", {}), registration=True))
    for key in (SMOOTHING_BY_METHOD_KEY, KERNEL_BY_MODE_KEY, "fusion/ai_mode",
                "fusion/dtcwt_levels", "fusion/ofb_depth_smoothing", "fusion/of_blend_smoothing"):
        settings.remove(key)



def load_window_settings(window) -> None:
    """Restore persisted preferences onto window attributes (defaults kept)."""
    settings = get_settings()
    if getattr(window, "ui_theme", None) not in ("dark", "light"):
        window.ui_theme = DEFAULT_THEME
    for attr, (key, default, kind) in _PERSISTED_FIELDS.items():
        if settings.contains(key):
            setattr(window, attr, _cast(settings.value(key), kind, default))
        else:
            setattr(window, attr, getattr(window, attr, default))

    # Must run after the scalar fields: the smoothing migration reads the
    # legacy ofb_depth_smoothing value restored just above.
    load_parameter_memories(window)

    raw_recent = settings.value(RECENT_FILES_KEY, []) or []
    if not isinstance(raw_recent, list):
        raw_recent = [raw_recent]
    window.recent_files = [str(p) for p in raw_recent][:MAX_RECENT_FILES]

    raw_projects = settings.value(RECENT_PROJECTS_KEY, []) or []
    if not isinstance(raw_projects, list):
        raw_projects = [raw_projects]
    window.recent_projects = [str(p) for p in raw_projects][:MAX_RECENT_PROJECTS]


def save_window_settings(window) -> None:
    """Write current window preferences (including language) to QSettings."""
    settings = get_settings()
    for attr, (key, default, _kind) in _PERSISTED_FIELDS.items():
        settings.setValue(key, getattr(window, attr, default))
    settings.setValue(LANGUAGE_KEY, trans.current_lang)
    settings.setValue(RECENT_FILES_KEY, list(getattr(window, "recent_files", [])))
    settings.setValue(RECENT_PROJECTS_KEY, list(getattr(window, "recent_projects", [])))
    save_parameter_memories(window)
    settings.sync()
    if settings.status() != QSettings.Status.NoError:
        raise OSError("Could not write OpenFocus settings")


def _supported_languages():
    """Codes the running build can actually render.

    Read from the translation manager rather than a literal tuple: the old
    hardcoded list silently dropped any newly added pack, so picking a new
    language was saved but ignored on the next launch (that is exactly what
    happened to ja/es in v1.50).
    """
    try:
        from locales import trans
        return tuple(trans.translations)
    except Exception:  # pragma: no cover - settings must never import-fail
        return ("en", "zh", "ja", "es")


def get_saved_language():
    """Return the persisted language code or None for auto-detect."""
    settings = get_settings()
    if not settings.contains(LANGUAGE_KEY):
        return None
    lang = str(settings.value(LANGUAGE_KEY, "")).lower()
    return lang if lang in _supported_languages() else None


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


# A window smaller than this is unusable (no menu bar, no canvas).
_MIN_RESTORABLE_WIDTH = 640
_MIN_RESTORABLE_HEIGHT = 480


def restore_window_layout(window) -> None:
    """Restore window geometry and splitter positions saved on last close.

    A saved geometry is rejected when it is unusable or off-screen. Qt stores
    whatever size the window had when it closed, and a window created at a
    tiny size (by a script, a screenshot harness, or a broken layout) would
    otherwise be restored as a sliver the user cannot grab — the app looked
    like it failed to start.
    """
    settings = get_settings()
    try:
        geo = settings.value(WINDOW_GEOMETRY_KEY)
        if geo is not None:
            window.restoreGeometry(geo)
            size = window.size()
            screens = window.screen() or (
                window.windowHandle().screen() if window.windowHandle() else None)
            if (size.width() < _MIN_RESTORABLE_WIDTH
                    or size.height() < _MIN_RESTORABLE_HEIGHT):
                window.resize(max(size.width(), 1280),
                              max(size.height(), 800))
            if screens is not None:
                available = screens.availableGeometry()
                frame = window.frameGeometry()
                if not available.intersects(frame):
                    window.move(available.center() - window.rect().center())
                    window.resize(min(1280, available.width()),
                                  min(800, available.height()))
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
    # The context menu offers WebP; rejecting it here made that entry a
    # silent no-op (the drag stayed JPG while the menu looked selectable).
    if fmt not in (".jpg", ".png", ".webp", ".tif"):
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


def add_recent_project(window, path: str) -> None:
    """Track a successfully opened project; dedupe, cap and persist."""
    path = os.path.abspath(path)
    recent = list(getattr(window, "recent_projects", []))
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    window.recent_projects = recent[:MAX_RECENT_PROJECTS]
    get_settings().setValue(RECENT_PROJECTS_KEY, list(window.recent_projects))
    if hasattr(window, "rebuild_recent_projects_menu"):
        window.rebuild_recent_projects_menu()
