"""OpenFocus project files (.ofproj): save/restore the complete work state.

A project stores the exact source file list (in stack order), the load-time
downsample scale, all rendering settings, label configurations and the
current frame index — everything needed to rebuild the working session.
Source images themselves are referenced, not copied.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

PROJECT_SUFFIX = ".ofproj"
FORMAT_VERSION = 1


def _collect_state(window) -> Dict[str, Any]:
    """Snapshot everything restorable from the main window."""
    from constants import APP_VERSION
    settings = {
        "fusion_method": (
            "guided_filter" if window.rb_a.isChecked()
            else "dct" if window.rb_b.isChecked()
            else "dtcwt" if window.rb_c.isChecked()
            else "gfgfgf" if window.rb_gfg.isChecked()
            else "stackmffv4" if window.rb_d.isChecked()
            else "guided_filter"
        ),
        "align_homography": window.cb_align_homography.isChecked(),
        "align_ecc": window.cb_align_ecc.isChecked(),
        "kernel_size": window.slider_smooth.value(),
        "thread_count": getattr(window, "thread_count", 4),
        "tile_enabled": getattr(window, "tile_enabled", True),
        "tile_block_size": getattr(window, "tile_block_size", 1024),
        "tile_overlap": getattr(window, "tile_overlap", 256),
        "tile_threshold": getattr(window, "tile_threshold", 2048),
        "reg_downscale_width": getattr(window, "reg_downscale_width", None),
        "stackmffv4_batch_size": getattr(window, "stackmffv4_batch_size", 2),
        "quick_preview": getattr(window, "chk_quick_preview", None) is not None
        and window.chk_quick_preview.isChecked(),
        "current_index": getattr(window, "current_display_index", -1),
    }
    labels = {}
    lm = getattr(window, "label_manager", None)
    if lm is not None and hasattr(lm, "export_state"):
        labels = lm.export_state()
    # Absolute paths: image_filenames are base names relative to the source
    # folder, and appended frames may come from several folders. The loader
    # keeps the real path of every frame; fall back to joining the current
    # folder (and finally to the bare name) when that record is unavailable.
    folder = getattr(window, "current_folder_path", "") or ""
    recorded = list(getattr(window, "image_source_paths", None) or [])
    names = list(getattr(window, "image_filenames", []))
    abs_paths = []
    for i, name in enumerate(names):
        if i < len(recorded) and recorded[i] and os.path.isfile(recorded[i]):
            abs_paths.append(recorded[i])
            continue
        cand = os.path.join(folder, name)
        abs_paths.append(cand if os.path.isfile(cand) else name)
    return {
        "openfocus_project": FORMAT_VERSION,
        "app_version": APP_VERSION,
        "sources": abs_paths,
        # Folder each frame was read from, so the stack still round-trips after
        # the frames are reached through another directory (moved, network
        # share, appended from a second folder).
        "source_folders": [os.path.dirname(p) if os.path.dirname(p) else ""
                           for p in abs_paths],
        "scale_factor": getattr(window, "current_scale_factor", 1.0),
        "settings": settings,
        "labels": labels,
    }


def save_project(window, path: str) -> Tuple[bool, str]:
    """Write the current session state to an .ofproj file."""
    if not getattr(window, "raw_images", []):
        return False, "no images loaded"
    try:
        state = _collect_state(window)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        return True, path
    except OSError as exc:
        return False, str(exc)


def _resolve_sources(state: Dict[str, Any], project_dir: str) -> Optional[List[str]]:
    """Locate every recorded frame. Returns the aligned path list, or None.

    Order of attempts: the recorded absolute path, the recorded per-frame
    folder, then a basename search next to the project file (the folder it was
    saved in) — which is also what old .ofproj files, that only stored names,
    can be matched with.
    """
    recorded_folders = state.get("source_folders") or []
    resolved: List[str] = []
    for i, recorded in enumerate(state.get("sources", [])):
        name = os.path.basename(recorded)
        candidates = [recorded]
        if i < len(recorded_folders) and recorded_folders[i]:
            candidates.append(os.path.join(str(recorded_folders[i]), name))
        candidates.append(os.path.join(project_dir, name))
        hit = next((c for c in candidates if c and os.path.isfile(c)), None)
        if hit is None:
            return None
        resolved.append(os.path.abspath(hit))
    return resolved


def validate_project(path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Read and validate a project file. Returns (ok, error, state).

    The resolved paths are stored back into the state as `resolved_sources`,
    so apply_project() opens exactly the frames that were found here."""
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cannot read project: {exc}", {}
    if not isinstance(state, dict) or "openfocus_project" not in state:
        return False, "not an OpenFocus project file", {}
    sources = state.get("sources", [])
    if not sources:
        return False, "project contains no sources", {}
    resolved = _resolve_sources(state, os.path.dirname(os.path.abspath(path)))
    if resolved is None:
        missing = [p for p in sources if not os.path.isfile(p)]
        return False, f"{len(missing)} source file(s) missing (first: {missing[0]})", {}
    state["resolved_sources"] = resolved

    # Numeric settings reach int()/float() casts inside Qt slots; a
    # hand-edited .ofproj must not crash the app there. Coerce here so
    # apply_project only ever sees clean types.
    def _num(key, cast, default):
        try:
            settings[key] = cast(settings.get(key, default))
        except (TypeError, ValueError):
            settings[key] = default

    settings = state.get("settings", {})
    _num("kernel_size", int, 31)
    _num("thread_count", int, 4)
    _num("tile_block_size", int, 1024)
    _num("tile_overlap", int, 256)
    _num("tile_threshold", int, 2048)
    _num("reg_downscale_width", int, 1024)
    _num("stackmffv4_batch_size", int, 2)
    _num("current_index", int, -1)
    try:
        state["scale_factor"] = float(state.get("scale_factor", 1.0))
    except (TypeError, ValueError):
        state["scale_factor"] = 1.0
    return True, "", state


def apply_project(window, state: Dict[str, Any]) -> None:
    """Load the project's sources and restore settings/labels/index."""
    sources: List[str] = state.get("resolved_sources") or state["sources"]
    settings = state.get("settings", {})

    # The stack may span folders; the registration cache and the "save project"
    # suggestion both key off the folder of the first frame.
    window.current_folder_path = os.path.dirname(sources[0])

    # Load frames from the exact recorded file list (no dialog)
    window.source_manager._start_load_worker(
        filepaths=list(sources),
        scale=float(state.get("scale_factor", 1.0) or 1.0),
        append=False,
        on_success=lambda: _apply_after_load(window, state),
    )


def _apply_after_load(window, state: Dict[str, Any]) -> None:
    settings = state.get("settings", {})

    method = settings.get("fusion_method")
    for code, rb in (("guided_filter", window.rb_a), ("dct", window.rb_b),
                     ("dtcwt", window.rb_c), ("gfgfgf", window.rb_gfg),
                     ("stackmffv4", window.rb_d)):
        rb.setChecked(code == method)

    window.cb_align_homography.setChecked(bool(settings.get("align_homography", False)))
    window.cb_align_ecc.setChecked(bool(settings.get("align_ecc", False)))

    kernel = int(settings.get("kernel_size", 31) or 31)
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    window.slider_smooth.setValue(kernel)

    if settings.get("thread_count"):
        window.thread_count = int(settings["thread_count"])
    window.tile_enabled = bool(settings.get("tile_enabled", True))
    window.tile_block_size = int(settings.get("tile_block_size", 1024))
    window.tile_overlap = int(settings.get("tile_overlap", 256))
    window.tile_threshold = int(settings.get("tile_threshold", 2048))
    if settings.get("reg_downscale_width"):
        window.reg_downscale_width = int(settings["reg_downscale_width"])
    if settings.get("stackmffv4_batch_size"):
        window.stackmffv4_batch_size = int(settings["stackmffv4_batch_size"])
    qp = window.chk_quick_preview if hasattr(window, "chk_quick_preview") else None
    if qp is not None:
        qp.setChecked(bool(settings.get("quick_preview", False)))

    labels = state.get("labels", {})
    lm = getattr(window, "label_manager", None)
    if lm is not None and labels and hasattr(lm, "import_state"):
        lm.import_state(labels)

    idx = int(settings.get("current_index", -1) or -1)
    if idx >= 0 and idx < len(window.raw_images):
        window.update_source_view(idx)


def suggest_project_path(window) -> Optional[str]:
    """Default save location: next to the source stack."""
    folder = getattr(window, "current_folder_path", "") or ""
    name = os.path.basename(folder.rstrip("/\\")) or "OpenFocus"
    return os.path.join(folder, name + PROJECT_SUFFIX) if folder else None
