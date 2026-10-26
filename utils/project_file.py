"""Project files reference source frames and reproducible processing settings.

Output pixels, retouch history and applied white-field correction are not stored;
those images require a separate export. Source order includes TIFF page identity.
"""
import json
import os
import math
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from core.algorithms import (fusion_algorithms, get_fusion, get_registration,
                             resolve_fusion_parameters, resolve_registration_parameters)

PROJECT_SUFFIX = ".ofproj"
FORMAT_VERSION = 3


def _collect_state(window) -> Dict[str, Any]:
    """Snapshot everything restorable from the main window."""
    from constants import APP_VERSION
    method = window.current_fusion_method()
    brightness = getattr(window, "brightness_option_for", None)
    alignment = window.current_alignment_method()
    if method != "none":
        get_fusion(method)
    get_registration(alignment)
    settings = {
        "fusion_method": method,
        "alignment_method": alignment,
        "stack_reverse_order": bool(getattr(window, "stack_reverse_order", False)),
        # The top-level flags describe *this* method; the per-method memory
        # below is what a reopen actually restores.
        "base_frame_correct": bool(brightness("base_correct", method))
        if callable(brightness) else bool(getattr(window, "base_frame_correct", False)),
        "exposure_match": bool(brightness("exposure_match", method))
        if callable(brightness) else bool(getattr(window, "exposure_match", True)),
        "thread_count": getattr(window, "thread_count", 4),
        "tile_enabled": getattr(window, "tile_enabled", True),
        "tile_block_size": getattr(window, "tile_block_size", 1024),
        "tile_overlap": getattr(window, "tile_overlap", 256),
        "tile_threshold": getattr(window, "tile_threshold", 2048),
        "reg_downscale_width": getattr(window, "reg_downscale_width", None),
        "ai_batch_size": getattr(window, "ai_batch_size", 2),
        "quick_preview": getattr(window, "chk_quick_preview", None) is not None
        and window.chk_quick_preview.isChecked(),
        "current_index": getattr(window, "current_display_index", -1),
        # Frames the user excluded from processing: without this a reopened
        # project silently renders a different set than the one on screen.
        "excluded_frames": sorted(
            int(i) for i in (getattr(window, "excluded_frames", set()) or set())),
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
        "scale_bar_custom_x": float(getattr(window, "scale_bar_custom_x", 0.95)),
        "scale_bar_custom_y": float(getattr(window, "scale_bar_custom_y", 0.95)),
        "scale_bar_calibration_mode": (
            "manual" if float(getattr(window, "scale_um_per_px_manual", 0.0) or 0.0) > 0
            else "metadata"),
    }
    settings["method_parameters"] = {
        spec.id: resolve_fusion_parameters(spec.id, window.fusion_parameters_for(spec.id))
        for spec in fusion_algorithms()}
    settings["registration_parameters"] = {
        name: resolve_registration_parameters(name, values)
        for name, values in getattr(window, "_registration_parameters_by_method", {}).items()}
    from utils.settings_store import brightness_memory_to_json
    settings["brightness_by_method"] = json.loads(brightness_memory_to_json(
        getattr(window, "_brightness_by_method", {})))
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
    # Decode each container once; a separate ordered frame list preserves
    # deleted, repeated and reordered TIFF pages without expanding N pages N times.
    source_indices = {}
    sources = []
    frame_order = []
    for path, name in zip(abs_paths, names):
        key = os.path.normcase(os.path.abspath(path))
        if key not in source_indices:
            source_indices[key] = len(sources)
            sources.append(path)
        frame_order.append({"source": source_indices[key], "name": name})
    abs_paths = sources
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
        "frame_order": frame_order,
        "source_transforms": list(getattr(window, "source_transforms", [])),
        "unrecoverable_source_changes": list(getattr(window, "unrecoverable_source_changes", [])),
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
    limitations = getattr(window, "unrecoverable_source_changes", [])
    if limitations:
        return False, "Export the processed source stack before saving a project: " + "; ".join(limitations)
    try:
        state = _collect_state(window)
        missing = [source for source in state["sources"] if not os.path.isfile(source)]
        if missing:
            return False, "Source file is unavailable: " + missing[0]
        from core.image_loader import ImageStackLoader
        videos = [source for source in state["sources"]
                  if os.path.splitext(source)[1].lower() in ImageStackLoader.SUPPORTED_VIDEO_FORMATS]
        if videos and len(state["sources"]) != 1:
            return False, "Export the mixed video/image source stack before saving a project"
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
    except (OSError, TypeError, ValueError) as exc:
        return False, str(exc)


def _resolve_sources(state: Dict[str, Any], project_dir: str) -> Optional[List[str]]:
    """Locate every recorded frame. Returns the aligned path list, or None.
    Absolute references must still exist. Legacy bare names may resolve next
    to the project or in their recorded folder; a missing absolute reference
    never silently binds to an unrelated same-named file.
    """
    recorded_folders = state.get("source_folders") or []
    resolved: List[str] = []
    for i, recorded in enumerate(state.get("sources", [])):
        name = os.path.basename(recorded)
        candidates = [recorded] if os.path.isabs(recorded) else []
        if not os.path.isabs(recorded):
            if i < len(recorded_folders) and recorded_folders[i]:
                candidates.append(os.path.join(str(recorded_folders[i]), name))
            candidates.append(os.path.join(project_dir, recorded))
        hit = next((c for c in candidates if c and os.path.isfile(c)), None)
        if hit is None:
            return None
        resolved.append(os.path.abspath(hit))
    return resolved


def validate_project(path: str, source_root: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
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
    version = state.get("openfocus_project")
    if not isinstance(version, int) or not 1 <= version <= FORMAT_VERSION:
        return False, "unsupported project format version", {}
    sources = state.get("sources", [])
    if not isinstance(sources, list) or not sources or not all(
            isinstance(p, str) and p for p in sources):
        return False, "project contains no valid sources", {}
    if not isinstance(state.get("source_folders", []), list):
        return False, "invalid source folders", {}
    if not isinstance(state.get("settings", {}), dict) or not isinstance(state.get("labels", {}), dict):
        return False, "invalid project settings or labels", {}
    frames = state.get("frame_order")
    if frames is not None and (not isinstance(frames, list) or not frames or any(
            not isinstance(frame, dict)
            or not isinstance(frame.get("source"), int)
            or not 0 <= frame["source"] < len(sources)
            or not isinstance(frame.get("name"), str) for frame in frames)):
        return False, "invalid project frame order", {}
    transforms = state.get("source_transforms", [])
    if not isinstance(transforms, list) or any(
            not isinstance(item, dict) or item.get("operation") not in ("rotate", "flip")
            or item.get("code") not in ((0, 1, 2) if item.get("operation") == "rotate" else (-1, 0, 1))
            for item in transforms):
        return False, "invalid source transformations", {}
    project_dir = os.path.dirname(os.path.abspath(path))
    resolved = (_relocated_sources(state, source_root, project_dir) if source_root is not None
                else _resolve_sources(state, project_dir))
    if resolved is None:
        missing = [p for p in sources if not os.path.isfile(p)] or sources
        return False, f"{len(missing)} source file(s) missing:\n" + "\n".join(missing), {}
    state["resolved_sources"] = resolved

    # Numeric settings reach int()/float() casts inside Qt slots; a
    # hand-edited .ofproj must not crash the app there. Coerce here so
    # apply_project only ever sees clean types.
    def _num(key, cast, default):
        try:
            settings[key] = cast(settings.get(key, default))
        except (TypeError, ValueError, OverflowError):
            settings[key] = default

    settings = state.setdefault("settings", {})
    try:
        _normalize_algorithm_settings(settings, legacy=version < 3)
    except (TypeError, ValueError) as exc:
        return False, str(exc), {}
    _num("thread_count", int, 4)
    _num("tile_block_size", int, 1024)
    _num("tile_overlap", int, 256)
    _num("tile_threshold", int, 2048)
    _num("reg_downscale_width", int, 1024)
    _num("ai_batch_size", int, 2)
    for key, default in (("scale_um_per_px_manual", 0.0), ("scale_bar_font_size", 0.0),
                         ("scale_bar_custom_x", 0.95), ("scale_bar_custom_y", 0.95)):
        _num(key, float, default)
        if not math.isfinite(settings[key]):
            settings[key] = default
    if not isinstance(settings.get("excluded_frames", []), list):
        return False, "invalid excluded frame list", {}
    _num("current_index", int, -1)
    try:
        state["scale_factor"] = float(state.get("scale_factor", 1.0))
    except (TypeError, ValueError):
        state["scale_factor"] = 1.0
    if not math.isfinite(state["scale_factor"]) or not 0 < state["scale_factor"] <= 1:
        return False, "invalid working scale", {}
    return True, "", state


def _normalize_algorithm_settings(settings, *, legacy=True):
    """Strict canonical validation with legacy fields translated only on read."""
    method = settings.get("fusion_method", "none")
    if method != "none":
        get_fusion(method)
    alignment = settings.get("alignment_method")
    if alignment is None:
        alignment = ("both" if settings.get("align_homography") and settings.get("align_ecc")
                     else "ecc" if settings.get("align_ecc")
                     else "homography" if settings.get("align_homography") else "none")
    get_registration(alignment)
    settings["alignment_method"] = alignment
    parameters = settings.get("method_parameters", {})
    registrations = settings.get("registration_parameters", {})
    brightness = settings.get("brightness_by_method", {})
    if not all(isinstance(value, dict) for value in (parameters, registrations, brightness)):
        raise ValueError("invalid method, registration or brightness parameters")
    canonical = {}
    brightness = {option: dict(values) if isinstance(values, dict) else values
                  for option, values in brightness.items()}
    for name, values in parameters.items():
        get_fusion(name)
        if not isinstance(values, dict):
            raise ValueError(f"invalid method parameters: {name}")
        values = dict(values)
        if legacy:
            if "kernel_size" in values:
                kernel = values["kernel_size"]
                if isinstance(kernel, bool) or not isinstance(kernel, int) or kernel < 1:
                    raise ValueError("invalid legacy method kernel")
            if "smoothing" in values:
                smoothing = values["smoothing"]
                if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)) \
                        or not math.isfinite(smoothing) or not 0 <= smoothing <= 8:
                    raise ValueError("invalid legacy method smoothing")
            declared = {parameter.key for parameter in get_fusion(name).parameters}
            # Historical snapshots emitted unused kernel/smoothing copies.
            if "kernel_size" not in declared:
                values.pop("kernel_size", None)
            if name not in ("openfocus_a", "openfocus_b", "openfocus_c"):
                values.pop("smoothing", None)
            for option in ("exposure_match", "base_correct"):
                if option in values:
                    brightness.setdefault(option, {})[name] = values.pop(option)
            if "smoothing" in values:
                smoothing = values.pop("smoothing")
                key = "depth_smoothing" if name == "openfocus_b" else "blend_smoothing"
                values.setdefault(key, smoothing)
        canonical[name] = values
    if legacy:
        for old, owner, key in (("ai_mode", "stackmffv3", "mode"),
                                ("dtcwt_levels", "dtcwt", "N")):
            if old in settings:
                canonical.setdefault(owner, {}).setdefault(key, settings[old])
    # The selected legacy scalar describes only the selected method.
    if method != "none":
        spec = get_fusion(method)
        values = canonical.setdefault(method, {})
        declared = {parameter.key for parameter in spec.parameters}
        if legacy and "kernel_size" in settings and "kernel_size" in declared:
            kernel = settings["kernel_size"]
            if isinstance(kernel, bool) or not isinstance(kernel, int) or kernel < 1:
                raise ValueError("invalid legacy method kernel")
            values.setdefault("kernel_size", kernel - 1 if kernel % 2 == 0 else kernel)
        for old, key in (("ofb_depth_smoothing", "depth_smoothing"),
                         ("of_blend_smoothing", "blend_smoothing")):
            if legacy and old in settings and key in declared:
                values.setdefault(key, settings[old])
        for old, option in (("base_frame_correct", "base_correct"),
                            ("exposure_match", "exposure_match")):
            if old in settings:
                brightness.setdefault(option, {}).setdefault(method, settings[old])
    for option, values in brightness.items():
        if option not in ("exposure_match", "base_correct") or not isinstance(values, dict):
            raise ValueError("invalid method brightness")
        for name, value in values.items():
            if name != "none":
                get_fusion(name)
            if not isinstance(value, bool):
                raise ValueError("invalid method brightness")
    settings["method_parameters"] = {}
    for name, values in canonical.items():
        resolved = resolve_fusion_parameters(name, values)
        settings["method_parameters"][name] = {key: resolved[key] for key in values}
    for name, values in registrations.items():
        if not isinstance(values, dict):
            raise ValueError(f"invalid registration parameters: {name}")
    settings["registration_parameters"] = {}
    for name, values in registrations.items():
        resolved = resolve_registration_parameters(name, values)
        settings["registration_parameters"][name] = {key: resolved[key] for key in values}
    settings["brightness_by_method"] = brightness


def _relocated_sources(state, root, project_dir):
    """Map one selected replacement root without guessing among basenames."""
    sources = state.get("sources", [])
    if not sources:
        return None
    absolute = [os.path.abspath(path if os.path.isabs(path)
                                else os.path.join(project_dir, path)) for path in sources]
    try:
        old_root = os.path.commonpath([os.path.dirname(path) for path in absolute])
    except ValueError:
        return None
    mapped = [os.path.abspath(os.path.join(root, os.path.relpath(path, old_root)))
              for path in absolute]
    return mapped if all(os.path.isfile(path) for path in mapped) else None


def validate_project_for_open(window, path):
    """Offer a relocation map for explicit confirmation before loading pixels."""
    ok, error, state = validate_project(path)
    if ok or "source file(s) missing:" not in error:
        return ok, error, state
    from PyQt6.QtWidgets import QFileDialog, QMessageBox
    from locales import trans
    from utils.ui_utils import style_message_box
    root = QFileDialog.getExistingDirectory(window, trans.t("project_relocate_title"),
                                             os.path.dirname(os.path.abspath(path)))
    if not root:
        return False, "", {}
    ok, error, state = validate_project(path, source_root=root)
    if not ok:
        return ok, error, state
    mapping = list(zip(state["sources"], state["resolved_sources"]))
    box = QMessageBox(window)
    style_message_box(box)
    box.setWindowTitle(trans.t("project_relocate_title"))
    box.setText(trans.t("project_relocate_confirm").format(count=len(mapping)))
    descriptions = [f"{before}\n→ {after}" for before, after in mapping]
    box.setInformativeText("\n\n".join(descriptions[:5]))
    box.setDetailedText("\n\n".join(descriptions))
    accept = box.addButton(trans.t("btn_continue"), QMessageBox.ButtonRole.AcceptRole)
    cancel = box.addButton(trans.t("btn_cancel_generic"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(cancel)
    box.exec()
    if box.clickedButton() is not accept:
        return False, "", {}
    replacements = {os.path.normcase(os.path.abspath(before)): os.path.normcase(os.path.abspath(after))
                    for before, after in mapping}
    for label in state.get("labels", {}).get("labels_v2", []):
        if not isinstance(label, dict) or not isinstance(label.get("frame_ids"), list):
            continue
        identities = []
        for identity in label["frame_ids"]:
            origin, separator, name = identity.rpartition("::")
            identities.append(replacements.get(origin, origin) + separator + name)
        label["frame_ids"] = identities
    state["sources_relocated"] = True
    return True, "", state


def apply_project(window, state: Dict[str, Any],
                  on_loaded=None, on_error=None) -> None:
    """Load the project's sources and restore settings/labels/index.

    `on_loaded` runs only after the frames were actually decoded and the
    settings applied — an update-restore must not report success when the
    asynchronous load then fails.
    """
    sources: List[str] = state.get("resolved_sources") or state["sources"]

    from core.image_loader import ImageStackLoader
    videos = [p for p in sources if os.path.splitext(p)[1].lower() in ImageStackLoader.SUPPORTED_VIDEO_FORMATS]
    if videos and len(sources) != 1:
        raise ValueError("Mixed video/image projects require exporting the source stack first")
    window.source_manager._start_load_worker(
        **({"video": sources[0]} if videos else {"filepaths": list(sources)}),
        scale=float(state.get("scale_factor", 1.0) or 1.0),
        append=False,
        prepare_options=lambda options: _prepare_project_load(window, state, options),
        on_error=on_error,
        on_success=lambda: (_apply_after_load(window, state),
                            on_loaded() if on_loaded is not None else None),
    )


def _prepare_project_load(window, state, options):
    """Resolve every saved frame before replacing the currently usable stack."""
    import cv2
    sources = state.get("resolved_sources") or state["sources"]
    paths = list(getattr(window.image_loader, "image_paths", []))
    frames = state.get("frame_order")
    decoded = {os.path.normcase(os.path.abspath(path)) for path in paths}
    for source in sources:
        if os.path.normcase(os.path.abspath(source)) not in decoded:
            raise ValueError("Project source could not be decoded: " + source)
    if frames is not None:
        lookup = {(os.path.normcase(os.path.abspath(path)), name): i
                  for i, (path, name) in enumerate(zip(paths, options.filenames))}
        order = []
        for frame in frames:
            key = (os.path.normcase(os.path.abspath(sources[frame["source"]])), frame["name"])
            if key not in lookup:
                raise ValueError("Project frame could not be decoded: " + frame["name"])
            order.append(lookup[key])
        options.filenames = [options.filenames[i] for i in order]
        options.base_images = [options.base_images[i] for i in order]
        paths = [paths[i] for i in order]
    for item in state.get("source_transforms", []):
        transform = cv2.rotate if item["operation"] == "rotate" else cv2.flip
        options.base_images = [transform(image, item["code"]) for image in options.base_images]
    options.working_images = list(options.base_images)
    window.image_loader.image_paths = paths
    return options


def _apply_after_load(window, state: Dict[str, Any]) -> None:
    """Apply saved settings; persist once, after the whole restore."""
    scope = getattr(window, "_restore_scope", None)
    if callable(scope):
        with scope():
            _apply_settings(window, state)
    else:
        _apply_settings(window, state)
    persist = getattr(window, "persist_settings", None)
    if callable(persist):
        persist()


def _apply_settings(window, state: Dict[str, Any]) -> None:
    settings = state.get("settings", {})
    window.source_transforms = list(state.get("source_transforms", []))
    window.unrecoverable_source_changes = list(state.get("unrecoverable_source_changes", []))
    _normalize_algorithm_settings(settings, legacy=state.get("openfocus_project", 2) < 3)
    method = settings.get("fusion_method")
    for name, values in settings["method_parameters"].items():
        window.set_fusion_parameters(name, {**window.fusion_parameters_for(name), **values})
    for name, values in settings["registration_parameters"].items():
        window.set_registration_parameters(name, {**window.registration_parameters_for(name), **values})
    memory = dict(getattr(window, "_brightness_by_method", {}) or {})
    for option, values in settings["brightness_by_method"].items():
        memory.update({(option, name): enabled for name, enabled in values.items()})
    window._brightness_by_method = memory
    if method:
        window.set_fusion_method(method)
    window.set_alignment_method(settings["alignment_method"])
    for hook in ("refresh_algorithm_parameters", "refresh_registration_parameters"):
        refresh = getattr(window, hook, None)
        if callable(refresh):
            refresh()


    # 反转帧序开关（v1.42）：属性 + 复选框同步
    reverse = bool(settings.get("stack_reverse_order", False))
    window.stack_reverse_order = reverse
    chk = getattr(window, "chk_reverse_order", None)
    if chk is not None:
        chk.setChecked(reverse)

    # Old projects carry only these two scalars, and they describe the method
    # the project was saved on. They must not be broadcast to other methods,
    # and a structured value for the current method wins.
    base_correct = window._brightness_by_method.get(
        ("base_correct", method), settings.get("base_frame_correct", False))
    if method:
        window._brightness_by_method[("base_correct", method)] = base_correct
    window.base_frame_correct = base_correct
    base_chk = getattr(window, "chk_base_correct", None)
    if base_chk is not None:
        base_chk.setChecked(base_correct)

    exposure_match = window._brightness_by_method.get(
        ("exposure_match", method), settings.get("exposure_match", True))
    if method:
        window._brightness_by_method[("exposure_match", method)] = exposure_match
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
    window.scale_bar_custom_x = float(settings.get("scale_bar_custom_x", 0.95))
    window.scale_bar_custom_y = float(settings.get("scale_bar_custom_y", 0.95))

    labels = state.get("labels", {})
    lm = getattr(window, "label_manager", None)
    if lm is not None and hasattr(lm, "import_state"):
        lm.import_state(labels)

    excluded = settings.get("excluded_frames")
    if excluded is not None:
        frame_count = len(getattr(window, "raw_images", None) or [])
        window.excluded_frames = {
            int(i) for i in excluded
            if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < frame_count}
        # Refresh the checkbox column so the list matches the restored set.
        if hasattr(window, "source_manager") and hasattr(
                window.source_manager, "decorate_source_items"):
            window.source_manager.decorate_source_items()

    if hasattr(window, "_sync_brightness_checkboxes"):
        window._sync_brightness_checkboxes()
    idx = int(settings.get("current_index", -1))
    if 0 <= idx < len(window.raw_images) and hasattr(window, "update_source_view"):
        window.update_source_view(idx)


def suggest_project_path(window) -> Optional[str]:
    """Default save location: next to the source stack."""
    folder = getattr(window, "current_folder_path", "") or ""
    name = os.path.basename(folder.rstrip("/\\")) or "OpenFocus"
    return os.path.join(folder, name + PROJECT_SUFFIX) if folder else None
