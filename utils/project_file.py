"""OpenFocus project files (.ofproj): save/restore the complete work state.

A project stores the exact source file list (in stack order), the load-time
downsample scale, all rendering settings, label configurations and the
current frame index — everything needed to rebuild the working session.
Source images themselves are referenced, not copied.
"""
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

PROJECT_SUFFIX = ".ofproj"
FORMAT_VERSION = 1


def _collect_state(window) -> Dict[str, Any]:
    """Snapshot everything restorable from the main window."""
    from constants import APP_VERSION
    settings = {
        "fusion_method": window.current_fusion_method(),
        "alignment_method": window.current_alignment_method(),
        "kernel_size": window.slider_smooth.value(),
        "stack_reverse_order": bool(getattr(window, "stack_reverse_order", False)),
        "base_frame_correct": bool(getattr(window, "base_frame_correct", False)),
        "exposure_match": bool(getattr(window, "exposure_match", True)),
        "thread_count": getattr(window, "thread_count", 4),
        "tile_enabled": getattr(window, "tile_enabled", True),
        "tile_block_size": getattr(window, "tile_block_size", 1024),
        "tile_overlap": getattr(window, "tile_overlap", 256),
        "tile_threshold": getattr(window, "tile_threshold", 2048),
        "reg_downscale_width": getattr(window, "reg_downscale_width", None),
        "ai_batch_size": getattr(window, "ai_batch_size", 2),
        "ai_mode": getattr(window, "ai_mode", "soft"),
        "quick_preview": getattr(window, "chk_quick_preview", None) is not None
        and window.chk_quick_preview.isChecked(),
        "current_index": getattr(window, "current_display_index", -1),
        # Scale-bar calibration & style (manual value only; metadata is
        # re-detected per stack on load)
        "scale_bar_enabled": bool(getattr(window, "scale_bar_enabled", False)),
        "scale_um_per_px_manual": float(getattr(window, "scale_um_per_px_manual", 0.0) or 0.0),
        "scale_bar_position": getattr(window, "scale_bar_position", "bottom-right"),
        "scale_bar_color": getattr(window, "scale_bar_color", "auto"),
        # Full style and scope: reopening a project must reproduce the bar
        # the user committed, not fall back to defaults for the rest.
        "scale_bar_target": getattr(window, "scale_bar_target", "all"),
        "scale_bar_font_family": getattr(window, "scale_bar_font_family", ""),
        "scale_bar_font_size": float(getattr(window, "scale_bar_font_size", 0.0) or 0.0),
        "scale_bar_font_bold": bool(getattr(window, "scale_bar_font_bold", False)),
        "scale_bar_text_align": getattr(window, "scale_bar_text_align", "auto"),
        "scale_bar_custom_x": float(getattr(window, "scale_bar_custom_x", 0.95) or 0.95),
        "scale_bar_custom_y": float(getattr(window, "scale_bar_custom_y", 0.95) or 0.95),
        "scale_bar_calibration_mode": (
            "manual" if float(getattr(window, "scale_um_per_px_manual", 0.0) or 0.0) > 0
            else "metadata"),
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
        if os.path.isfile(cand):
            abs_paths.append(cand)
            continue
        # 多页 TIFF 的每帧是合成名（X_page_NNN.tif），磁盘上只有容器
        # X.tif——不映射回容器的话工程永远校验失败，崩溃恢复也被连带
        # 静默丢弃（pending_recovery 会过滤校验失败的快照）
        mapped = _multipage_container_for(name, folder)
        abs_paths.append(mapped if mapped else name)
    # 多页 TIFF 的每一帧都映射到同一个容器：原样保存会让重载时每个容器
    # 再展开一遍全部页（N 页 -> N² 帧，工程/崩溃恢复都会 OOM）。折叠连续
    # 重复项，容器只保留一次。
    collapsed: list = []
    for path in abs_paths:
        if collapsed and os.path.normcase(collapsed[-1]) == os.path.normcase(path):
            continue
        collapsed.append(path)
    abs_paths = collapsed
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


def _multipage_container_for(name: str, folder: str) -> str:
    """Map a synthetic multi-page frame name (X_page_NNN.tif, emitted by
    core.image_loader for each TIFF page) back to its container file."""
    m = re.match(r"^(.*)_page_\d{3,}\.(tif|tiff)$", name, re.IGNORECASE)
    if not m:
        return ""
    for ext in (m.group(2), m.group(2).lower(), m.group(2).upper(), "tif", "tiff"):
        cand = os.path.join(folder, f"{m.group(1)}.{ext}")
        if os.path.isfile(cand):
            return cand
    return ""


def save_project(window, path: str) -> Tuple[bool, str]:
    """Write the current session state to an .ofproj file."""
    if not getattr(window, "raw_images", []):
        return False, "no images loaded"
    try:
        state = _collect_state(window)
        # 原子写：快照写入中途断电/被杀留下的截断文件会让
        # validate_project 拒绝它——恰好废掉崩溃恢复存在的意义
        fd, tmp = tempfile.mkstemp(
            suffix=".ofproj.tmp", dir=os.path.dirname(os.path.abspath(path)) or ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
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
    _num("ai_batch_size", int, 2)
    if settings.get("ai_mode") not in ("soft", "hard"):
        settings["ai_mode"] = "soft"
    _num("current_index", int, -1)
    try:
        state["scale_factor"] = float(state.get("scale_factor", 1.0))
    except (TypeError, ValueError):
        state["scale_factor"] = 1.0
    return True, "", state


def apply_project(window, state: Dict[str, Any]) -> None:
    """Load the project's sources and restore settings/labels/index."""
    sources: List[str] = state.get("resolved_sources") or state["sources"]

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
    if method:
        window.set_fusion_method(method)

    # 对齐方法：新格式是 alignment_method 单选代码；旧格式（v1.39 及以前）
    # 是 align_homography/align_ecc 两个布尔，按 ECC > Homography 优先迁移
    alignment = settings.get("alignment_method")
    if alignment not in ("none", "homography", "ecc", "openfocus"):
        alignment = (
            "ecc" if settings.get("align_ecc")
            else "homography" if settings.get("align_homography")
            else "none")
    window.set_alignment_method(alignment)

    kernel = int(settings.get("kernel_size", 31) or 31)
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    window.slider_smooth.setValue(kernel)

    # 反转帧序开关（v1.42）：属性 + 复选框同步
    reverse = bool(settings.get("stack_reverse_order", False))
    window.stack_reverse_order = reverse
    chk = getattr(window, "chk_reverse_order", None)
    if chk is not None:
        chk.setChecked(reverse)

    base_correct = bool(settings.get("base_frame_correct", False))
    window.base_frame_correct = base_correct
    base_chk = getattr(window, "chk_base_correct", None)
    if base_chk is not None:
        base_chk.setChecked(base_correct)

    exposure_match = bool(settings.get("exposure_match", True))
    window.exposure_match = exposure_match
    exp_chk = getattr(window, "chk_exposure_match", None)
    if exp_chk is not None:
        exp_chk.setChecked(exposure_match)

    if settings.get("thread_count"):
        window.thread_count = int(settings["thread_count"])
    window.tile_enabled = bool(settings.get("tile_enabled", True))
    window.tile_block_size = int(settings.get("tile_block_size", 1024))
    window.tile_overlap = int(settings.get("tile_overlap", 256))
    window.tile_threshold = int(settings.get("tile_threshold", 2048))
    if settings.get("reg_downscale_width"):
        window.reg_downscale_width = int(settings["reg_downscale_width"])
    if settings.get("ai_batch_size"):
        window.ai_batch_size = int(settings["ai_batch_size"])
    if settings.get("ai_mode") in ("soft", "hard"):
        window.ai_mode = settings["ai_mode"]
        radios = (getattr(window, "radio_ai_soft", None),
                  getattr(window, "radio_ai_hard", None))
        target = radios[0] if window.ai_mode == "soft" else radios[1]
        if target is not None:
            target.setChecked(True)
    qp = window.chk_quick_preview if hasattr(window, "chk_quick_preview") else None
    if qp is not None:
        qp.setChecked(bool(settings.get("quick_preview", False)))

    window.scale_bar_enabled = bool(settings.get("scale_bar_enabled", False))
    window.scale_um_per_px_manual = float(settings.get("scale_um_per_px_manual", 0.0) or 0.0)
    window.scale_bar_position = settings.get("scale_bar_position", "bottom-right")
    window.scale_bar_color = settings.get("scale_bar_color", "auto")
    window.scale_bar_target = settings.get("scale_bar_target", "all")
    window.scale_bar_font_family = settings.get("scale_bar_font_family", "")
    window.scale_bar_font_size = float(settings.get("scale_bar_font_size", 0.0) or 0.0)
    window.scale_bar_font_bold = bool(settings.get("scale_bar_font_bold", False))
    window.scale_bar_text_align = settings.get("scale_bar_text_align", "auto")
    window.scale_bar_custom_x = float(settings.get("scale_bar_custom_x", 0.95) or 0.95)
    window.scale_bar_custom_y = float(settings.get("scale_bar_custom_y", 0.95) or 0.95)

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
