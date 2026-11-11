"""Project files reference source frames and reproducible processing settings.

Output pixels and retouch history are not stored. File-backed flat-field
corrections are restored only while their reference remains unchanged.

Lazy frames record container path, zero-based page, working scale and ordered
transforms. Legacy pathless arrays are persisted as TIFF files in the adjacent
``<project>.sources`` directory and reopened lazily without a second resize.
"""
import json
import os
import math
import re
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Tuple

from core.algorithms import (fusion_algorithms, registration_algorithms, get_fusion, get_registration,
                             migrate_openfocus_parameters, resolve_fusion_parameters,
                             resolve_registration_parameters)

PROJECT_SUFFIX = ".ofproj"
FORMAT_VERSION = 4

def _sensor_mask_digest(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

TEMPLATE_SUFFIX = ".oftemplate"
TEMPLATE_VERSION = 2
TEMPLATE_SETTINGS = (
    "fusion_method", "alignment_method", "stack_reverse_order",
    "base_frame_correct", "exposure_match", "thread_count", "tile_enabled",
    "tile_block_size", "tile_overlap", "tile_threshold", "reg_downscale_width",
    "ai_batch_size", "method_parameters", "registration_parameters",
    "brightness_by_method", "sharpen_output", "sensor_denoise",
)


def validate_template_settings(settings, *, old_parameters=False):
    """Validate processing settings without retaining source-specific state."""
    if not isinstance(settings, dict) or set(settings) - set(TEMPLATE_SETTINGS):
        raise ValueError("invalid template settings")
    settings = json.loads(json.dumps(settings, allow_nan=False))
    for key in ("stack_reverse_order", "base_frame_correct", "exposure_match", "tile_enabled", "sharpen_output", "sensor_denoise"):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("thread_count", "tile_block_size", "tile_overlap", "tile_threshold",
                "reg_downscale_width", "ai_batch_size"):
        value = settings.get(key)
        if value is None and key == "reg_downscale_width":
            continue
        if key in settings and (type(value) is not int or value < (0 if key in ("thread_count", "tile_overlap") else 1)):
            raise ValueError(f"invalid {key}")
    if settings.get("tile_overlap", 0) >= settings.get("tile_block_size", 1024):
        raise ValueError("tile overlap must be smaller than the tile size")
    _normalize_algorithm_settings(settings, legacy=False, old_parameters=old_parameters)
    return settings


def template_from_state(state):
    """Extract settings from a validated project or a window snapshot."""
    settings = state.get("settings", {})
    return validate_template_settings({key: settings[key] for key in TEMPLATE_SETTINGS if key in settings})


def atomic_json_write(path, data):
    """Write JSON beside its destination and replace it only after success."""
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def save_template(window, path):
    settings = template_from_state(_collect_state(window))
    atomic_json_write(path, {"openfocus_template": TEMPLATE_VERSION, "settings": settings})


def load_template(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate field: {key}")
            result[key] = value
        return result
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream, object_pairs_hook=unique_pairs)
    if isinstance(data, dict) and "openfocus_project" in data:
        ok, error, state = validate_project(path, require_sources=False)
        if not ok:
            raise ValueError(error)
        return template_from_state(state)
    if isinstance(data, dict) and "version" in data:
        from core.recipes import load_recipe
        recipe = load_recipe(path)
        if "project_settings" not in recipe:
            raise ValueError("recipe has no project settings")
        return validate_template_settings(recipe["project_settings"])
    if (not isinstance(data, dict) or set(data) != {"openfocus_template", "settings"}
            or type(data["openfocus_template"]) is not int
            or data["openfocus_template"] not in (1, TEMPLATE_VERSION)):
        raise ValueError("unsupported or invalid template")
    return validate_template_settings(data["settings"], old_parameters=data["openfocus_template"] == 1)


def apply_template(window, settings):
    """Apply processing settings to the currently loaded sources, untouched."""
    settings = validate_template_settings(settings)
    registrations = settings.setdefault("registration_parameters", {})
    alignment = settings.get("alignment_method", "none")
    registrations.setdefault(alignment, {})
    settings["registration_parameters"] = {
        name: resolve_registration_parameters(name, values)
        for name, values in registrations.items()}
    before = template_from_state(_collect_state(window))
    state = _collect_state(window)
    state["settings"].update(settings)
    if settings.get("thread_count") is not None:
        # Applying a template is an explicit choice of processing settings.
        window.thread_count = int(settings["thread_count"])
    _apply_after_load(window, state)
    after = template_from_state(_collect_state(window))
    mark = getattr(window, "_mark_project_dirty", None)
    if after != before and callable(mark):
        mark()


def _finite_scalar(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {field}")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"invalid {field}") from None
    if not math.isfinite(number):
        raise ValueError(f"invalid {field}")
    return number


def _validate_source_transforms(transforms, scale, *, require_sources=True):
    from utils.dust_map import load_flat_field_reference
    if not isinstance(transforms, list):
        raise ValueError("invalid source transformations")
    for item in transforms:
        if not isinstance(item, dict):
            raise ValueError("invalid source transformations")
        operation = item.get("operation")
        if operation in ("rotate", "flip"):
            if item.get("code") not in ((0, 1, 2) if operation == "rotate" else (-1, 0, 1)):
                raise ValueError("invalid source transformations")
        elif operation == "flat_field":
            path, digest = item.get("path"), item.get("sha256")
            sigma, reference_scale = item.get("sigma"), item.get("scale")
            sigma = _finite_scalar(sigma, "source_transforms sigma")
            reference_scale = _finite_scalar(reference_scale, "source_transforms scale")
            if (not isinstance(path, str) or not os.path.isabs(path)
                    or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or not 1 <= sigma <= 40 or abs(reference_scale - scale) > 1e-9):
                raise ValueError("Invalid flat-field reference or working scale; export the source stack first")
            if require_sources:
                load_flat_field_reference(path, expected_hash=digest)
        else:
            raise ValueError("invalid source transformations")


def _collect_state(window) -> Dict[str, Any]:
    """Snapshot everything restorable from the main window."""
    from constants import app_display_version
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
        "sharpen_output": bool(getattr(window, "sharpen_output", False)),
        "defect_mask_path": os.path.abspath(window.defect_mask_path) if getattr(window, "defect_mask_path", "") else "",
        "sensor_denoise": bool(getattr(window, "sensor_denoise", False)),
        # The top-level flags describe *this* method; the per-method memory
        # below is what a reopen actually restores.
        "base_frame_correct": bool(brightness("base_correct", method))
        if callable(brightness) else bool(getattr(window, "base_frame_correct", False)),
        "exposure_match": bool(brightness("exposure_match", method))
        if callable(brightness) else bool(getattr(window, "exposure_match", False)),
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
    from utils import watermark
    settings["watermark"] = watermark.settings_from_window(window)
    settings["watermark"]["outputs"] = watermark.record_configs(window)
    if settings["defect_mask_path"]:
        settings["defect_mask_sha256"] = _sensor_mask_digest(settings["defect_mask_path"])
    settings["method_parameters"] = {
        spec.id: resolve_fusion_parameters(spec.id, window.fusion_parameters_for(spec.id))
        for spec in fusion_algorithms()}
    registration_for = getattr(window, "registration_parameters_for", None)
    memory = getattr(window, "_registration_parameters_by_method", {})
    settings["registration_parameters"] = {
        spec.id: resolve_registration_parameters(
            spec.id, registration_for(spec.id) if callable(registration_for) else memory.get(spec.id, {}))
        for spec in registration_algorithms()}
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
    images = list(getattr(window, "raw_images", ()) or ())
    if not names:
        names = [f"frame_{index + 1:04d}.tif" for index in range(len(images))]
    abs_paths = []
    for i, name in enumerate(names):
        image = images[i] if i < len(images) else None
        descriptor_path = getattr(image, "path", None)
        if descriptor_path:
            abs_paths.append(os.path.abspath(descriptor_path))
            continue
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
    baseline = getattr(getattr(window, "source_manager", None), "_stack_baseline", None)
    snapshot_scales = {}
    if baseline:
        for image in baseline["images"]:
            if getattr(image, "path", None) and image.scale != baseline["scale"]:
                key = (os.path.normcase(os.path.abspath(image.path)), image.page)
                snapshot_scales[key] = baseline["scale"] / image.scale
    for index, (path, name) in enumerate(zip(abs_paths, names)):
        key = os.path.normcase(os.path.abspath(path))
        if key not in source_indices:
            source_indices[key] = len(sources)
            sources.append(path)
        frame = {"source": source_indices[key], "name": name}
        image = images[index] if index < len(images) else None
        if getattr(image, "path", None):
            frame["page"] = getattr(image, "page", None)
            frame["scale"] = float(image.scale)
            frame["transforms"] = list(image.transforms)
            snapshot_scale = snapshot_scales.get((key, image.page))
            if snapshot_scale is not None:
                frame.update(snapshot=True, snapshot_scale=snapshot_scale)
        frame_order.append(frame)
    abs_paths = sources
    return {
        "openfocus_project": FORMAT_VERSION,
        "app_version": app_display_version(),
        "sources": abs_paths,
        # Folder each frame was read from, so the stack still round-trips after
        # the frames are reached through another directory (moved, network
        # share, appended from a second folder).
        "source_folders": [os.path.dirname(p) if os.path.dirname(p) else ""
                           for p in abs_paths],
        "scale_factor": getattr(window, "current_scale_factor", 1.0),
        "source_px_um": getattr(window, "source_px_um", None),
        "linear_raw": getattr(window, "linear_dng_metadata", None) is not None,
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


def _owned_sidecars(project_path):
    """Read only explicitly recorded, safe sidecars belonging to this project."""
    root = os.path.abspath(str(project_path) + ".sources")
    try:
        with open(project_path, encoding="utf-8") as stream:
            manifest = json.load(stream).get("owned_sidecars", [])
    except (OSError, ValueError, AttributeError):
        return set()
    if not isinstance(manifest, list):
        return set()
    owned = set()
    for relative in manifest:
        if not isinstance(relative, str) or os.path.isabs(relative):
            continue
        parts = relative.replace("\\", "/").split("/")
        if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
            continue
        directory = os.path.join(root, parts[0])
        filename = os.path.join(directory, parts[1])
        if os.path.islink(root) or os.path.islink(directory) or os.path.islink(filename):
            continue
        if os.path.commonpath((root, os.path.abspath(filename))) == root and os.path.isfile(filename):
            owned.add(os.path.abspath(filename))
    return owned


def _foreign_sidecar(source, project_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(source)))
    if not root.endswith(".sources"):
        return False
    owner = root[:-len(".sources")]
    return (os.path.normcase(os.path.abspath(owner)) != os.path.normcase(os.path.abspath(project_path))
            and os.path.abspath(source) in _owned_sidecars(owner))


def _remove_owned_files(files):
    for path in files:
        directory = os.path.dirname(path)
        root = os.path.dirname(directory)
        if os.path.islink(root) or os.path.islink(directory) or os.path.islink(path):
            continue
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(directory)
        except OSError:
            pass

def _retire_owned_files(files):
    from core.source_frames import retire_owned_files
    for path in files:
        retire_owned_files((path,), lambda path=path: _remove_owned_files((path,)))


def remove_project_snapshot(path):
    """Remove a recovery snapshot, then retire only its recorded sidecars."""
    owned = _owned_sidecars(path)
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    _retire_owned_files(owned)


def _persist_pathless_sources(window, state, project_path, created, directories):
    """Create a private generation for in-memory and foreign-owned sources."""
    import numpy as np
    from utils.frame_pixels import run_pixel_worker
    from utils.image_utils import imwrite_auto
    pending = []
    recorded = list(getattr(window, "image_source_paths", ()) or ())
    for index, frame in enumerate(state["frame_order"]):
        source = state["sources"][frame["source"]]
        image = window.raw_images[index]
        pathless = (not os.path.isfile(source) and isinstance(image, np.ndarray)
                    and not (index < len(recorded) and recorded[index]))
        if pathless or _foreign_sidecar(source, project_path):
            pending.append((index, image, source, pathless))
    if not pending:
        return
    root = os.path.abspath(str(project_path) + ".sources")
    def persist():
        os.makedirs(root, exist_ok=True)
        if os.path.islink(root):
            raise OSError("Project sidecar directory is a symbolic link")
        directory = tempfile.mkdtemp(prefix="generation-", dir=root)
        directories.add(directory)
        paths = {}
        for index, image, source, pathless in pending:
            fd, destination = tempfile.mkstemp(suffix=".tif", dir=directory)
            created.add(destination)
            os.close(fd)
            if pathless:
                if not imwrite_auto(destination, image, exif=b""):
                    raise OSError("Cannot preserve project source: " + destination)
            else:
                shutil.copyfile(source, destination)
            paths[index] = (destination, pathless)
        return paths
    paths = run_pixel_worker(window, "OpenFocus project", persist)
    sources, folders, order = [], [], []
    for index, frame in enumerate(state["frame_order"]):
        source, pathless = paths.get(index, (state["sources"][frame["source"]], False))
        updated = dict(frame, source=len(sources))
        if pathless:
            updated.update(page=0, snapshot=True, snapshot_scale=state["scale_factor"])
        sources.append(source)
        folders.append(os.path.dirname(source))
        order.append(updated)
    state.update(sources=sources, source_folders=folders, frame_order=order)


def save_project(window, path: str, *, recovery_document=None) -> Tuple[bool, str]:
    """Write the current session state to an .ofproj file."""
    if not getattr(window, "raw_images", []):
        return False, "no images loaded"
    limitations = getattr(window, "unrecoverable_source_changes", [])
    if limitations:
        return False, "Export the processed source stack before saving a project: " + "; ".join(limitations)
    created, directories = set(), set()
    committed = False
    old_owned = _owned_sidecars(path)
    try:
        state = _collect_state(window)
        if recovery_document is not None:
            _validate_recovery_document(recovery_document)
            state["recovery_document"] = dict(recovery_document)
        _persist_pathless_sources(window, state, path, created, directories)
        _validate_source_transforms(state.get("source_transforms", []), state["scale_factor"])
        missing = [source for source in state["sources"] if not os.path.isfile(source)]
        if missing:
            return False, "Source file is unavailable: " + missing[0]
        from core.image_loader import ImageStackLoader
        videos = [source for source in state["sources"]
                  if os.path.splitext(source)[1].lower() in ImageStackLoader.SUPPORTED_VIDEO_FORMATS]
        if videos and len(state["sources"]) != 1:
            return False, "Export the mixed video/image source stack before saving a project"
        referenced = {os.path.abspath(source) for source in state["sources"]}
        retained = (old_owned & referenced) | created
        root = os.path.abspath(str(path) + ".sources")
        state["owned_sidecars"] = sorted(os.path.relpath(source, root) for source in old_owned | created)
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
        committed = True
        try:
            _retire_owned_files(old_owned - retained)
        except OSError:
            import logging
            logging.getLogger(__name__).warning("Could not retire old project sidecars", exc_info=True)
        return True, path
    except (OSError, TypeError, ValueError) as exc:
        return False, str(exc)
    finally:
        if not committed:
            _remove_owned_files(created)
            for directory in directories:
                try:
                    os.rmdir(directory)
                except OSError:
                    pass


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


def _validate_recovery_document(document):
    if (not isinstance(document, dict) or set(document) != {"version", "project_path", "dirty"}
            or type(document["version"]) is not int or document["version"] != 1
            or not isinstance(document["project_path"], str) or type(document["dirty"]) is not bool):
        raise ValueError("invalid recovery_document")


def validate_project(path: str, source_root: Optional[str] = None, *, require_sources=True) -> Tuple[bool, str, Dict[str, Any]]:
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
    try:
        if "recovery_document" in state:
            _validate_recovery_document(state["recovery_document"])
    except ValueError as exc:
        return False, str(exc), {}
    sources = state.get("sources", [])
    if not isinstance(sources, list) or not sources or not all(
            isinstance(p, str) and p for p in sources):
        return False, "project contains no valid sources", {}
    if not isinstance(state.get("source_folders", []), list):
        return False, "invalid source folders", {}
    if not isinstance(state.get("settings", {}), dict) or not isinstance(state.get("labels", {}), dict):
        return False, "invalid project settings or labels", {}
    if type(state.get("linear_raw", False)) is not bool:
        return False, "invalid linear RAW setting", {}
    source_px_um = state.get("source_px_um")
    try:
        if source_px_um is not None and _finite_scalar(source_px_um, "source_px_um") <= 0:
            raise ValueError("invalid source_px_um")
    except ValueError as exc:
        return False, str(exc), {}
    if "sharpen_output" in state.get("settings", {}) and not isinstance(
            state["settings"]["sharpen_output"], bool):
        return False, "invalid output sharpening setting", {}
    sensor = state.get("settings", {})
    mask_path = sensor.get("defect_mask_path", "")
    digest = sensor.get("defect_mask_sha256", "")
    if (not isinstance(mask_path, str) or type(sensor.get("sensor_denoise", False)) is not bool
            or (mask_path and (not os.path.isabs(mask_path) or not isinstance(digest, str)
                               or not re.fullmatch(r"[0-9a-f]{64}", digest)))
            or (not mask_path and digest)):
        return False, "invalid sensor cleanup settings", {}
    if mask_path and require_sources:
        try:
            actual = _sensor_mask_digest(mask_path)
            if actual != digest:
                return False, "sensor defect mask has changed: " + mask_path, {}
            from utils.sensor_cleanup import load_sensor_defect_mask
            load_sensor_defect_mask(mask_path)
        except (OSError, ValueError) as exc:
            return False, str(exc), {}
    frames = state.get("frame_order")
    if frames is not None and (not isinstance(frames, list) or not frames or any(
            not isinstance(frame, dict)
            or not isinstance(frame.get("source"), int)
            or not 0 <= frame["source"] < len(sources)
            or not isinstance(frame.get("name"), str)
            or (frame.get("page") is not None and
                (type(frame["page"]) is not int or frame["page"] < 0)) for frame in frames)):
        return False, "invalid project frame order", {}
    try:
        for frame in frames or ():
            if type(frame.get("snapshot", False)) is not bool:
                raise ValueError("invalid source snapshot")
            for key in ("scale", "snapshot_scale"):
                if key in frame:
                    if _finite_scalar(frame[key], f"frame {key}") <= 0:
                        raise ValueError(f"invalid frame {key}")
            if "transforms" in frame:
                if not isinstance(frame["transforms"], list):
                    raise ValueError("invalid frame transformations")
                for operation in frame["transforms"]:
                    if not isinstance(operation, dict):
                        raise ValueError("invalid frame transformations")
                    if operation.get("operation") == "resize":
                        if any(type(operation.get(key)) is not int or operation[key] <= 0 for key in ("width", "height")):
                            raise ValueError("invalid frame dimensions")
                    elif operation.get("operation") in ("rotate", "flip"):
                        _validate_source_transforms([operation], frame.get("scale", 1.0), require_sources=False)
                    else:
                        raise ValueError("invalid frame transformations")
    except (ValueError, TypeError) as exc:
        return False, str(exc), {}
    project_dir = os.path.dirname(os.path.abspath(path))
    if require_sources:
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
        _normalize_algorithm_settings(settings, legacy=version < 3, old_parameters=version < 4)
        registrations = settings["registration_parameters"]
        settings["registration_parameters"] = {
            spec.id: resolve_registration_parameters(spec.id, registrations.get(spec.id, {}))
            for spec in registration_algorithms()}
        state['openfocus_project'] = FORMAT_VERSION
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
    except OverflowError:
        return False, "invalid scale_factor", {}
    except (TypeError, ValueError):
        state["scale_factor"] = 1.0
    if not math.isfinite(state["scale_factor"]) or not 0 < state["scale_factor"] <= 1:
        return False, "invalid scale_factor", {}
    try:
        _validate_source_transforms(state.get("source_transforms", []), state["scale_factor"], require_sources=require_sources)
    except (OSError, ValueError) as exc:
        return False, str(exc), {}
    return True, "", state


def _normalize_algorithm_settings(settings, *, legacy=True, old_parameters=False):
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
            if name in ('openfocus_a', 'openfocus_b'):
                declared.add('kernel_size')
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
        if old_parameters and not legacy and name in ('openfocus_a', 'openfocus_b', 'openfocus_c'):
            unknown = set(values) & {'radius', 'smoothing', 'feather'}
            if unknown:
                raise ValueError(f"invalid legacy method parameters: {sorted(unknown)}")
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
        if legacy and method in ('openfocus_a', 'openfocus_b', 'openfocus_c'):
            declared.add('depth_smoothing' if method == 'openfocus_b' else 'blend_smoothing')
            if method != 'openfocus_c':
                declared.add('kernel_size')
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
        if old_parameters:
            values = migrate_openfocus_parameters(name, values)
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
    from PyQt6.QtWidgets import QMessageBox
    from utils.file_dialogs import FileDialog
    from locales import trans
    from utils.ui_utils import style_message_box
    root = FileDialog.getExistingDirectory(window, trans.t("project_relocate_title"),
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

    `on_loaded` runs only after lazy frames and their settings are installed;
    asynchronous failures reach `on_error` before replacing the current stack.
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
        linear_raw=state.get("linear_raw", False),
        prepare_options=lambda options: _prepare_project_load(window, state, options),
        on_error=on_error,
        on_success=lambda: (_apply_after_load(window, state),
                            on_loaded() if on_loaded is not None else None),
    )


def _prepare_project_load(window, state, options):
    from utils.frame_pixels import run_pixel_worker
    return run_pixel_worker(window, "OpenFocus project",
                            lambda: _prepare_project_frames(window, state, options))


def _prepare_project_frames(window, state, options):
    """Order lazy frames and replay descriptors before publishing the stack."""
    from core.source_frames import transform_frame, materialize_frames, scan_source
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
        page_lookup = {(os.path.normcase(os.path.abspath(path)), getattr(image, "page", None)): i
                       for i, (path, image) in enumerate(zip(paths, options.base_images))}
        for frame in frames:
            source = os.path.normcase(os.path.abspath(sources[frame["source"]]))
            key = (source, frame["name"])
            match = (page_lookup.get((source, frame.get("page")))
                     if "page" in frame else lookup.get(key))
            if match is None:
                raise ValueError("Project frame could not be decoded: " + frame["name"])
            order.append(match)
        profiles = tuple(getattr(window.image_loader, "source_icc_profiles", ()) or ())
        options.source_icc_profiles = tuple(
            profiles[i] if i < len(profiles) else b"" for i in order)
        exifs = tuple(getattr(options, "source_exifs", ()) or ())
        if exifs:
            options.source_exifs = tuple(exifs[i] if i < len(exifs) else b"" for i in order)
        options.filenames = [frame["name"] for frame in frames]
        options.base_images = [options.base_images[i] for i in order]
        paths = [paths[i] for i in order]
        if any(frame.get("snapshot") or ("scale" in frame and frame["scale"] != getattr(image, "scale", None))
               for frame, image in zip(frames, options.base_images)):
            from utils.frame_pixels import run_pixel_worker
            def restore_frame_scales():
                restored = []
                for index, (frame, image) in enumerate(zip(frames, options.base_images)):
                    scale = frame.get("scale", 1.0 if frame.get("snapshot") else getattr(image, "scale", 1.0))
                    if frame.get("snapshot") or scale != getattr(image, "scale", None):
                        image = scan_source(paths[index], scale=scale, linear=state.get("linear_raw", False),
                                            header_only=True, page_index=frame.get("page", 0))[0]
                    restored.append(image)
                return restored
            options.base_images = run_pixel_worker(window, "OpenFocus project", restore_frame_scales)
    # A descriptor's scale precedes its replayed resize operations. Snapshot
    # files already contain working pixels, even though their file scale is 1.
    working_scale = getattr(options, "scale_factor", state.get("scale_factor", 1.0))
    reset_scales = [(frames[index].get("snapshot_scale", working_scale) * image.scale)
                    if frames and frames[index].get("snapshot") else
                    getattr(image, "scale", working_scale)
                    for index, image in enumerate(options.base_images)]
    reset_scale = (reset_scales[0] if reset_scales and
                   all(scale == reset_scales[0] for scale in reset_scales) else working_scale)
    options.reset_scale_factor = reset_scale
    if "source_px_um" in state:
        options.source_px_um = state["source_px_um"]
    # Mixed origins must still reset to one common pixel scale. Read headers
    # only; reset retains lazy, untransformed sources rather than their pixels.
    options.reset_images = [
        image if scale == reset_scale else scan_source(
            paths[index], scale=image.scale * reset_scale / scale, linear=state.get("linear_raw", False),
            header_only=True, page_index=getattr(image, "page", 0))[0]
        for index, (image, scale) in enumerate(zip(options.base_images, reset_scales))]
    if frames:
        for index, frame in enumerate(frames):
            for operation in frame.get("transforms", []):
                options.base_images[index] = transform_frame(
                    options.base_images[index], operation["operation"],
                    **{key: value for key, value in operation.items() if key != "operation"})
    for item in state.get("source_transforms", []):
        if item["operation"] == "flat_field":
            from utils.frame_pixels import run_pixel_worker
            def apply_reference():
                from utils.dust_map import load_flat_field_reference, compute_flat_field, apply_flat_field
                white, _ = load_flat_field_reference(item["path"], expected_hash=item["sha256"])
                gain = compute_flat_field(white, sigma=item["sigma"])
                originals = materialize_frames(options.base_images)
                return materialize_frames(
                    image if frames and frames[index].get("snapshot") else apply_flat_field(image, gain)
                    for index, image in enumerate(originals))
            options.base_images = run_pixel_worker(window, "OpenFocus project", apply_reference)
        else:
            options.base_images = [image if frames and (frames[index].get("snapshot") or "transforms" in frames[index]) else
                                   transform_frame(image, item["operation"], code=item["code"])
                                   for index, image in enumerate(options.base_images)]
    options.working_images = list(options.base_images)
    options.reset_paths = list(paths)
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
    window.defect_mask_path = settings.get("defect_mask_path", "")
    window.sensor_denoise = settings.get("sensor_denoise", False)
    version = state.get("openfocus_project", 2)
    _normalize_algorithm_settings(settings, legacy=version < 3, old_parameters=version < 4)
    state["openfocus_project"] = FORMAT_VERSION
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
    window.sharpen_output = bool(settings.get("sharpen_output", False))
    sharpen_chk = getattr(window, "chk_sharpen_output", None)
    if sharpen_chk is not None:
        sharpen_chk.setChecked(window.sharpen_output)

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

    from core.algorithms import get_fusion
    default_exposure = (get_fusion(method).default_exposure_match
                        if method and method != "none" else False)
    exposure_match = window._brightness_by_method.get(
        ("exposure_match", method), settings.get("exposure_match", default_exposure))
    if method:
        window._brightness_by_method[("exposure_match", method)] = exposure_match
    window.exposure_match = exposure_match
    exp_chk = getattr(window, "chk_exposure_match", None)
    if exp_chk is not None:
        exp_chk.setChecked(exposure_match)

    # thread_count is still saved for provenance, but it describes the computer
    # that saved the file. Applying it on open, crash recovery or the
    # after-update restore overrode this machine's Advanced Settings value, and
    # persist_settings() then wrote it back. Templates apply it explicitly.
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

    from utils import watermark
    watermark.apply_settings(window, settings.get("watermark") or {})

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
