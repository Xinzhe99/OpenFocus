"""Dynamic output filename template formatting for OpenFocus exports.

Supports safe token substitution, OS path sanitization, extension stripping,
graceful handling of unknown or empty tokens, and full backward compatibility
with ``output_name_prefix``.
"""
from __future__ import annotations

import datetime
import os
import re
from typing import Any, Mapping, Optional, Sequence

from utils.export_paths import prefix_output_name

TEMPLATE_TOKENS: tuple[str, ...] = (
    "[SOURCE_FILE_NAME]",
    "[FIRST_FILE]",
    "[LAST_FILE]",
    "[FOLDER]",
    "[METHOD]",
    "[ALIGN]",
    "[FRAMES]",
    "[DATE]",
    "[TIME]",
    "[WIDTH]",
    "[HEIGHT]",
    "[RADIUS]",
    "[SMOOTHING]",
)

DEFAULT_FILENAME_TEMPLATE = "OpenFocus_[DATE]_[TIME]_[METHOD]_[ALIGN]"

_TOKEN_PATTERN = re.compile(r"\[([A-Za-z0-9_]+)\]")
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_MULTI_UNDERSCORE = re.compile(r"_{2,}")
_MULTI_HYPHEN = re.compile(r"-{2,}")
_MULTI_SPACE = re.compile(r"\s{2,}")
_MAX_STEM_LENGTH = 180

_WINDOWS_RESERVED = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})


def _strip_file_stem(value: Any) -> str:
    """Return a clean filename stem without directory path or extension."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    base = os.path.basename(text.replace("\\", "/").rstrip("/"))
    stem, _ext = os.path.splitext(base)
    return stem or base


def _folder_basename(value: Any) -> str:
    """Return a clean directory basename."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/").rstrip("/")
    return os.path.basename(normalized)


def _format_number(value: Any) -> str:
    """Format numeric metadata concisely (ints without trailing .0)."""
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value).strip()


def sanitize_filename_component(text: str, *, max_length: int = _MAX_STEM_LENGTH) -> str:
    """Sanitize a filename stem so it is portable across Windows, macOS, and Linux."""
    if not text:
        return ""
    cleaned = str(text)
    cleaned = cleaned.replace("..", "_")
    cleaned = _ILLEGAL_CHARS.sub("_", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    cleaned = re.sub(r"\s*_\s*", "_", cleaned)
    cleaned = _MULTI_UNDERSCORE.sub("_", cleaned)
    cleaned = _MULTI_HYPHEN.sub("-", cleaned)
    cleaned = re.sub(r"[-_]{2,}", lambda m: m.group(0)[0], cleaned)
    cleaned = cleaned.strip(" ._-")
    if not cleaned:
        return ""
    if max_length > 0 and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" ._-")
    if cleaned.upper() in _WINDOWS_RESERVED:
        cleaned = f"{cleaned}_file"
    return cleaned


def _lookup_ci(mapping: Mapping[str, Any], *keys: str) -> Any:
    """Case-insensitive lookup across candidate keys."""
    if not mapping:
        return None
    lower_map = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        val = lower_map.get(key.lower())
        if val is not None and val != "":
            return val
    return None


def _resolve_token_values(context: Mapping[str, Any] | None) -> dict[str, str]:
    """Normalize a raw context mapping into canonical uppercase token values."""
    ctx = context or {}
    now = _lookup_ci(ctx, "now", "timestamp")
    if not isinstance(now, datetime.datetime):
        now = datetime.datetime.now()

    source_files = _lookup_ci(ctx, "source_files", "filenames", "source_paths")
    first_from_list = ""
    last_from_list = ""
    if isinstance(source_files, (list, tuple)) and source_files:
        first_from_list = _strip_file_stem(source_files[0])
        last_from_list = _strip_file_stem(source_files[-1])

    first_file = _strip_file_stem(_lookup_ci(ctx, "FIRST_FILE", "first_file")) or first_from_list
    last_file = _strip_file_stem(_lookup_ci(ctx, "LAST_FILE", "last_file")) or last_from_list or first_file
    source_file = _strip_file_stem(
        _lookup_ci(ctx, "SOURCE_FILE_NAME", "source_file_name", "source_file")
    ) or first_file

    folder = _folder_basename(_lookup_ci(ctx, "FOLDER", "folder", "folder_name", "folder_path"))
    method = str(_lookup_ci(ctx, "METHOD", "method", "fusion_method") or "").strip()
    align = str(_lookup_ci(ctx, "ALIGN", "align", "alignment", "align_method") or "").strip()

    frames_val = _lookup_ci(ctx, "FRAMES", "frames", "frame_count")
    if frames_val is None and isinstance(source_files, (list, tuple)):
        frames_val = len(source_files)
    frames = _format_number(frames_val)

    date_val = _lookup_ci(ctx, "DATE", "date")
    date_str = str(date_val).strip() if date_val is not None else now.strftime("%Y%m%d")

    time_val = _lookup_ci(ctx, "TIME", "time")
    time_str = str(time_val).strip() if time_val is not None else now.strftime("%H%M%S")

    width = _format_number(_lookup_ci(ctx, "WIDTH", "width"))
    height = _format_number(_lookup_ci(ctx, "HEIGHT", "height"))

    fusion_params = _lookup_ci(ctx, "fusion_params", "fusion_parameters")
    params = fusion_params if isinstance(fusion_params, Mapping) else {}

    radius_val = _lookup_ci(ctx, "RADIUS", "radius")
    if radius_val is None:
        radius_val = _lookup_ci(params, "radius", "kernel_size")
    radius = _format_number(radius_val)

    smoothing_val = _lookup_ci(ctx, "SMOOTHING", "smoothing")
    if smoothing_val is None:
        smoothing_val = _lookup_ci(params, "smoothing", "depth_smoothing", "blend_smoothing")
    smoothing = _format_number(smoothing_val)

    return {
        "SOURCE_FILE_NAME": source_file,
        "FIRST_FILE": first_file,
        "LAST_FILE": last_file,
        "FOLDER": folder,
        "METHOD": method,
        "ALIGN": align,
        "FRAMES": frames,
        "DATE": date_str,
        "TIME": time_str,
        "WIDTH": width,
        "HEIGHT": height,
        "RADIUS": radius,
        "SMOOTHING": smoothing,
    }


def format_output_filename(
    template: str,
    context: Mapping[str, Any] | None = None,
    *,
    fallback: str = "",
) -> str:
    """Render *template* with *context* into a safe, portable filename stem.

    * Supported tokens are substituted case-insensitively.
    * Unknown or empty tokens degrade gracefully without leaving dangling
      brackets or duplicate separators.
    * File paths and extensions in source-file tokens are stripped cleanly.
    * Illegal OS filename characters and traversal segments are removed.
    * If the rendered result is empty, returns sanitized *fallback* or a
      timestamped ``OpenFocus_<DATE>_<TIME>`` name.
    """
    values = _resolve_token_values(context)
    raw_template = (template or "").strip() if isinstance(template, str) else ""

    if raw_template:
        def _replace(match: re.Match[str]) -> str:
            token = match.group(1).upper()
            return values.get(token, "")

        rendered = _TOKEN_PATTERN.sub(_replace, raw_template)
        # Strip any accidental trailing image extension typed into the template.
        stem, ext = os.path.splitext(rendered)
        if ext.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".dng"):
            rendered = stem
        sanitized = sanitize_filename_component(rendered)
        if sanitized:
            return sanitized

    if fallback:
        sanitized_fallback = sanitize_filename_component(fallback)
        if sanitized_fallback:
            return sanitized_fallback

    return sanitize_filename_component(
        f"OpenFocus_{values['DATE']}_{values['TIME']}"
    ) or "OpenFocus_result"


def apply_output_naming(
    default_name: str,
    *,
    template: str = "",
    prefix: str = "",
    context: Mapping[str, Any] | None = None,
) -> str:
    """Apply optional dynamic template and/or legacy prefix cleanly.

    When *template* is empty, preserves exact legacy ``output_name_prefix``
    behavior on *default_name*.
    """
    tpl = (template or "").strip() if isinstance(template, str) else ""
    if tpl:
        base = format_output_filename(tpl, context, fallback=default_name)
        return prefix_output_name(prefix, base)
    return prefix_output_name(prefix, default_name)


def build_template_context(
    *,
    filenames: Sequence[str] | None = None,
    folder_path: str = "",
    method: str = "",
    align: str = "",
    frame_count: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fusion_params: Mapping[str, Any] | None = None,
    now: Optional[datetime.datetime] = None,
) -> dict[str, Any]:
    """Assemble a canonical context mapping for :func:`format_output_filename`."""
    files = [str(f) for f in (filenames or ()) if f]
    ts = now if isinstance(now, datetime.datetime) else datetime.datetime.now()
    ctx: dict[str, Any] = {
        "source_files": files,
        "folder": folder_path,
        "method": method,
        "align": align,
        "frames": frame_count if frame_count is not None else (len(files) if files else ""),
        "date": ts.strftime("%Y%m%d"),
        "time": ts.strftime("%H%M%S"),
        "width": width if width is not None else "",
        "height": height if height is not None else "",
        "fusion_params": dict(fusion_params) if fusion_params else {},
        "now": ts,
    }
    return ctx


def example_template_context() -> dict[str, Any]:
    """Deterministic sample context used for live settings previews."""
    return {
        "SOURCE_FILE_NAME": "IMG_0001.JPG",
        "FIRST_FILE": "IMG_0001.JPG",
        "LAST_FILE": "IMG_0015.JPG",
        "FOLDER": "Specimen_01",
        "METHOD": "OpenFocusB",
        "ALIGN": "ECC",
        "FRAMES": 15,
        "DATE": "20261015",
        "TIME": "120000",
        "WIDTH": 4000,
        "HEIGHT": 3000,
        "RADIUS": 8,
        "SMOOTHING": 4,
    }
