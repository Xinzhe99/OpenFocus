"""Headless command-line interface for OpenFocus fusion.

Lets batch users and CI pipelines run registration + fusion without the GUI:

    openfocus --input ./stack_folder --output ./out.png --method stackmffv3
    openfocus --input ./stackA ./stackB --output-dir ./results   (batch)

Exit codes: 0 success, 1 processing error, 2 usage error.
"""
from contextlib import contextmanager
import argparse
import json
import math
import os
import subprocess
import sys
import time

# Must precede the numpy/torch import below: two OpenMP runtimes (MKL through
# numpy, torch's libiomp5md.dll) otherwise abort the process with
# "OMP: Error #15 ... already initialized" as soon as ECC and the AI model run
# in the same process. Same guard as main.py — the `openfocus` console script
# enters here and never imports main.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from typing import List, Optional, Tuple

from utils import performance_profile
from utils.sorting import natural_sort_key
from utils.path_identity import file_identity, path_key
from core.resource_policy import heavy_job, require_memory
from core.frame_storage import scratch_directory

from core.algorithms import (fusion_algorithms, registration_algorithms, get_fusion,
                             get_registration, resolve_fusion_parameters,
                             resolve_registration_parameters, alignment_steps)
from constants import app_display_version
from utils.logging_setup import (setup_logging, log_session_start, log_operation,
                                 log_action, get_logger, frame_details)

_log = get_logger(__name__)

VALID_METHODS = [spec.id for spec in fusion_algorithms()]
VALID_ALIGN = [spec.id for spec in registration_algorithms()]
VALID_FORMATS = [".png", ".jpg", ".webp", ".bmp", ".tif", ".tiff"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2

SUBCOMMANDS = {"ctl", "gui", "describe", "version", "recipe", "retouch"}

# With --json-progress, stdout belongs to NDJSON: engine and progress prose is
# redirected to stderr, while events are written to this saved stream.
_machine_stream = None


class _StderrProxy:
    encoding = "utf-8"

    def write(self, text):
        return sys.stderr.write(text)

    def flush(self):
        sys.stderr.flush()

    def isatty(self):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="OpenFocus",
        description="Multi-focus image fusion (headless mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  openfocus --input ./stack --output ./result/fused.png\n"
            "  openfocus --input ./stack --output ./result/fused.png --method dtcwt --align ecc --threads 8\n"
            "  openfocus --input ./stackA ./stackB --output-dir ./results -m stackmffv3\n"
            "  openfocus --input img1.jpg img2.jpg img3.jpg --output fused.png --method gfgfgf\n"
            "  openfocus --input ./stack --output fused.tif --align ecc --auto-background\n"
            "\nTransparent input: PNG frames must share one rectangular, constant-opacity\n"
            "region and bit depth. Output must be PNG; alignment, dust correction,\n"
            "sensor cleanup, depth maps and scale bars are unsupported. See the user manual.\n"
        ),
    )
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="Image folder(s), video file(s), or a list of image files")
    parser.add_argument("--suggest-duplicates", action="store_true",
                        help="Report JSON suggestions for adjacent files in input order; "
                             "reconsider suggestions if their reference is excluded")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file for a single stack (required without --output-dir)")
    parser.add_argument("--output-dir", "-d", default=None,
                        help="Batch mode: fuse every input folder, writing <folder>.png here")
    parser.add_argument("--native-raw", action="store_true",
                        help="Fuse compatible RAW files in camera-linear space; export DNG, 16-bit TIFF or PNG")
    parser.add_argument("--native-batch-format", choices=["dng", "tiff", "png"], default=None,
                        help="Native RAW batch output format (default: dng; requires --native-raw and --output-dir)")
    parser.add_argument("--group-size", type=int, default=0, metavar="FRAMES",
                        help="Fuse a folder in overlapping groups of at most FRAMES (0 disables; minimum 2). "
                             "Approximate; each round uses group-local registration and exposure references, "
                             "crops conservatively, and uses temporary disk space. Matching single-frame "
                             "Pillow-readable inputs only; no RAW, alpha, multipage or depth maps.")
    parser.add_argument("--group-overlap", type=int, default=0, metavar="FRAMES",
                        help="Frames shared by consecutive groups (default 0; at most group size minus 2).")
    parser.add_argument("--method", "-m", default="stackmffv3",
                        choices=[spec.id for spec in fusion_algorithms()],
                        help="Fusion method (default: stackmffv3); use `openfocus describe` "
                             "for available methods and their parameters")
    parser.add_argument("--align", "-a", default="none",
                        choices=[spec.id for spec in registration_algorithms()],
                        help="Registration applied before fusion (default: none); openfocus = "
                             "fast bounded similarity search with auto-crop")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=JSON_VALUE",
                        help="Fusion parameter; repeat for multiple keys. Overrides legacy flags.")
    parser.add_argument("--align-param", action="append", default=[], metavar="KEY=JSON_VALUE",
                        help="Registration parameter; repeat for multiple keys.")
    parser.add_argument("--align-shift-limit", type=float, default=None, metavar="RATIO",
                        help="openfocus alignment only: translation search domain as a fraction "
                             "of the frame size (default 0.03)")
    parser.add_argument("--base-frame", default="none", choices=["none", "last", "first"],
                        help="Optional post-fusion base correction: match the output's low "
                             "frequencies to the last/first frame (default: none).")
    parser.add_argument("--exposure-match", dest="exposure_match",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Match exposure before fusion: OpenFocus A/B/C use the first "
                             "processed frame, other methods use the last (default: on; "
                             "--no-exposure-match disables it). Applied before tiling.")
    parser.add_argument("--sharpen-output", action="store_true",
                        help="Apply mild unsharp masking to the fused output (off by default).")
    parser.add_argument("--sort", default="asc", choices=["asc", "desc", "off"],
                        help="Frame order for openfocus_a/b/c: 'asc' keeps file order "
                             "(default), 'desc' reverses it, 'off' leaves it unchanged")
    parser.add_argument("--dust-map", default=None, metavar="WHITE",
                        help="Flat-field (dust map) correction: path to a defocused shot of a blank/white "
                             "area; removes static dust specks and vignetting from every frame before "
                             "registration")
    parser.add_argument("--defect-mask-path", default=None, metavar="MASK",
                        help="Sensor-coordinate defect mask; nonzero pixels are repaired before registration")
    parser.add_argument("--sensor-denoise", action="store_true",
                        help="Apply conservative sensor denoising before registration")
    parser.add_argument("--kernel", "-k", type=int, default=None,
                        help="Odd kernel size for methods with a kernel_size parameter; "
                             "use --param radius=8 for OpenFocus A/B")
    parser.add_argument("--threads", "-t", type=int, default=0,
                        help="Worker thread upper limit (0 = automatic; default: 0)")
    parser.add_argument("--resource-preference", default="balanced",
                        choices=["balanced", "responsive", "maximum"],
                        help="Resource scheduling preference (default: balanced)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="StackMFF-V3 tile batch size (default: 2)")
    parser.add_argument("--ai-mode", default=None, choices=["soft", "hard"],
                        metavar="{soft,hard}",
                        help="StackMFF-V3 output: 'soft' blends the frames by "
                             "their predicted probabilities (default); 'hard' "
                             "takes the most probable frame per pixel")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU for OpenFocus B and AI fusion")
    parser.add_argument("--downscale", type=int, default=None,
                        help="Registration feature-detection downscale width")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Tile block size for large images (default: auto)")
    parser.add_argument("--tile-enabled", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable tiled fusion when needed (default: on)")
    parser.add_argument("--tile-overlap", type=int, default=None,
                        help="Tile overlap in pixels (default: automatic)")
    parser.add_argument("--tile-threshold", type=int, default=None,
                        help="Minimum image size for tiling (default: automatic)")
    parser.add_argument("--scale-bar", action="store_true",
                        help="Draw a scale bar on the output (calibration auto-detected from TIFF/DNG microscopy metadata, or --scale-um)")
    parser.add_argument("--scale-um", type=float, default=None,
                        help="Pixel size in um/px, overriding metadata (implies --scale-bar)")
    parser.add_argument("--depth-map", "-D", dest="depth_map", nargs="?", const=True, default=None,
                        help="Also save a colorized focus-position (pseudo-depth) map for a supported method "
                             "(optional value: output path; default <output>_depth.png; batch requires default path)")
    parser.add_argument("--depth-colormap", default="turbo",
                        help="Colormap for --depth-map: turbo, viridis, jet, gray, ... (default turbo)")
    parser.add_argument("--frame-order", default="auto", choices=["auto", "name"],
                        help="Frames of an input folder: auto (default) orders them near to far by EXIF "
                             "focus steps when every frame has them (OM SYSTEM focus bracketing starts at "
                             "mid-depth), otherwise by name; name = always by filename. Files listed "
                             "explicitly keep their order")
    parser.add_argument("--auto-background", nargs="?", const="auto", default=None,
                        choices=["auto", "soft", "sharp"], metavar="PRESET",
                        help="Replace the stacked background with an aligned frame's, as Retouch > "
                             "Auto background does: auto (default) = the blurriest background unless the "
                             "stack recorded much of the background in focus, soft = blurriest, "
                             "sharp = the far-focus frame. Needs the aligned frames, so not with --group-size")
    parser.add_argument("--background-mode", default="clean", choices=["clean", "faithful"],
                        help="Auto background: clean (default) also removes the stack's outline-following "
                             "edge flaws, within what the frames recorded; faithful keeps them")
    parser.add_argument("--keep-soft-parts", action="store_true",
                        help="Auto background: keep soft objects attached to the subject that some frame "
                             "focused on (can keep nearby background objects in their stacked rendering too)")
    parser.add_argument("--save-stacked", nargs="?", const=True, default=None, metavar="PATH",
                        help="With --auto-background, also save the result before the background was "
                             "replaced (default <output>_stacked<ext>; batch requires the default path)")
    parser.add_argument("--missing-data-map", nargs="?", const=True, default=None, metavar="PATH",
                        help="With --auto-background, also save where no frame recorded the scene: a tinted "
                             "PNG plus one 8-bit mask PNG per class (default <output>_missing.png; batch "
                             "requires the default path)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print the plan without fusing")
    parser.add_argument("--resume", action="store_true",
                        help="Batch mode: skip folders only when every requested result is a nonempty file; content and settings are not checked")
    parser.add_argument("--json", default=None, metavar="FILE",
                        help="Write a machine-readable result summary to FILE when done")
    parser.add_argument("--alignment-report", default=None, metavar="FILE",
                        help="Write registration diagnostics as JSON (requires alignment; single stack only; no grouped fusion or dry run)")
    parser.add_argument("--json-progress", action="store_true",
                        help="Print one JSON progress object per step to stdout (NDJSON)")
    parser.add_argument("--temp-dir", default=None, metavar="DIR",
                        help="Directory for temporary data (must exist and be writable). "
                             "Default: the system temporary directory or the local volume "
                             "with the most space; the GUI's Temporary directory setting "
                             "is not used by the CLI")
    parser.add_argument("--profile-performance", nargs="?", const=True, default=None, metavar="FILE",
                        help="Time supported render stages; print a summary to stderr, embed the "
                             "report in --json, and optionally write it as JSON to FILE "
                             "(single, non-grouped run only)")
    parser.add_argument("--log-level", default=None, type=str.upper, metavar="LEVEL",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        help="Detail of this run's log file: debug, info (default), warning "
                             "or error. Overrides OPENFOCUS_LOG_LEVEL; the log path is "
                             "printed to stderr")
    parser.add_argument("--config", default=None, metavar="FILE",
                        help="Batch mode: JSON file with per-folder overrides, "
                             "e.g. {\"stackA\": {\"method\": \"dct\", \"kernel\": 9}}")
    return parser


def _parameter_assignments(assignments):
    values = {}
    for assignment in assignments or ():
        key, separator, encoded = assignment.partition("=")
        if not separator or not key:
            raise ValueError(f"Expected KEY=JSON_VALUE, got {assignment!r}")
        try:
            values[key] = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid JSON for parameter {key}: {exc}") from exc
    return values


def _validate_execution_args(args):
    """Validate execution options shared by CLI, batch overrides and recipes."""
    import math

    for name, minimum, optional in (
            ('threads', 0, False), ('batch_size', 1, False),
            ('downscale', 1, True), ('tile_size', 64, True),
            ('kernel', None, True), ('tile_overlap', 0, True),
            ('tile_threshold', 1, True), ('group_size', 0, False),
            ('group_overlap', 0, False)):
        value = getattr(args, name)
        if value is None and optional:
            continue
        if type(value) is not int or (minimum is not None and value < minimum):
            raise ValueError(f"invalid {name}: {value!r}")
    for name in ('exposure_match', 'sharpen_output', 'cpu', 'keep_soft_parts'):
        if type(getattr(args, name)) is not bool:
            raise ValueError(f"{name} must be boolean")
    actions = {action.dest: action for action in build_parser()._actions}
    for name in ('method', 'align', 'base_frame', 'sort', 'ai_mode',
                 'resource_preference', 'auto_background', 'background_mode'):
        value = getattr(args, name)
        if value is None and name in ('ai_mode', 'auto_background'):
            continue
        if not isinstance(value, str) or value not in actions[name].choices:
            raise ValueError(f"invalid {name}: {value!r}")
    for name in ('fusion_parameters', 'registration_parameters'):
        if not isinstance(getattr(args, name, {}), dict):
            raise ValueError(f"{name} must be an object")
    value = args.align_shift_limit
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value)):
        raise ValueError('align_shift_limit must be a finite number')
    if args.group_size == 1 or (args.group_size and args.group_overlap > args.group_size - 2):
        raise ValueError('invalid group_size or group_overlap')
    if args.group_size and args.depth_map:
        raise ValueError('depth_map is unavailable with grouped fusion')
    if args.tile_overlap is not None and args.tile_size is not None and args.tile_overlap >= args.tile_size:
        raise ValueError('tile_overlap must be smaller than tile_size')
    if args.native_batch_format and (not args.native_raw or not args.output_dir):
        raise ValueError('native_batch_format requires native_raw and output_dir')
    if args.native_raw and (args.suggest_duplicates or args.group_size or args.dust_map or
                            args.defect_mask_path or args.sensor_denoise or args.depth_map):
        raise ValueError('native_raw cannot be combined with grouped fusion, references or depth maps')
    if args.native_raw and (args.scale_bar or args.scale_um is not None) and (
            (args.output_dir and (args.native_batch_format or 'dng') == 'dng') or
            (args.output and os.path.splitext(args.output)[1].lower() == '.dng')):
        raise ValueError('scale bars are not supported in native DNG output')
    extras = args.background_mode != 'clean' or args.keep_soft_parts or args.save_stacked or args.missing_data_map
    if extras and not args.auto_background:
        raise ValueError('background options require auto_background')
    if args.auto_background and (args.group_size or args.native_raw):
        raise ValueError('auto_background is unavailable with grouped fusion or native_raw')
    if args.output_dir:
        for name in ('depth_map', 'save_stacked', 'missing_data_map'):
            if isinstance(getattr(args, name), str):
                option = '--' + name.replace('_', '-')
                raise ValueError(f'{option} with an explicit path cannot be used with --output-dir')
    if isinstance(args.missing_data_map, str) and not args.missing_data_map.lower().endswith('.png'):
        raise ValueError('missing_data_map must be a .png path')
    if isinstance(args.save_stacked, str) and os.path.splitext(args.save_stacked)[1].lower() not in VALID_FORMATS:
        raise ValueError('save_stacked has an unsupported extension')
    return args


def _resolve_algorithm_arguments(args):
    """Translate old flags once; generic dictionaries/assignments win per field."""
    spec = get_fusion(args.method)
    get_registration(args.align)
    declared = {parameter.key for parameter in spec.parameters}
    values = {}
    if getattr(args, "kernel", None) is not None:
        if "kernel_size" not in declared:
            raise ValueError("--kernel is not available for this method; use --param radius=... for OpenFocus A/B")
        default = next(parameter.default for parameter in spec.parameters if parameter.key == "kernel_size")
        values["kernel_size"] = _normalize_kernel(args.kernel, default)
    if getattr(args, "ai_mode", None) is not None:
        if "mode" not in declared:
            raise ValueError("--ai-mode is not available for this method")
        values["mode"] = args.ai_mode
    provided = getattr(args, "fusion_parameters", {})
    registration = getattr(args, "registration_parameters", {})
    if not isinstance(provided, dict) or not isinstance(registration, dict):
        raise ValueError("fusion_parameters and registration_parameters must be objects")
    values.update(provided)
    values.update(_parameter_assignments(getattr(args, "param", ())))
    alignment = {}
    if getattr(args, "align_shift_limit", None) is not None:
        if args.align != "openfocus":
            raise ValueError("--align-shift-limit requires --align openfocus")
        alignment["max_shift_ratio"] = args.align_shift_limit
    alignment.update(registration)
    alignment.update(_parameter_assignments(getattr(args, "align_param", ())))
    args._fusion_parameters = resolve_fusion_parameters(args.method, values)
    args._registration_parameters = resolve_registration_parameters(args.align, alignment)
    if getattr(args, "depth_map", None) and spec.depth is None:
        raise ValueError(f"--depth-map is not available for method '{args.method}'")
    return args


def _list_folder_images(folder: str, loader) -> List[str]:
    return sorted(
        (os.path.join(folder, f) for f in os.listdir(folder)
         if f.lower().endswith(tuple(loader.SUPPORTED_FORMATS))),
        key=natural_sort_key,
    )


def _folder_stack_paths(folder: str, loader, args) -> List[str]:
    """A folder's frames: by name, or near to far by EXIF focus steps.

    Calibration files selected from the same folder (dust map, defect mask)
    are not stack frames.  Remove them before reading EXIF, otherwise their
    missing focus metadata silently disables the ordering for every frame.
    """
    paths = _list_folder_images(folder, loader)
    for reference in (getattr(args, "dust_map", None),
                      getattr(args, "defect_mask_path", None)):
        paths = _filter_stack_paths(paths, reference)
    if getattr(args, "frame_order", "name") == "auto":
        from core.focus_order import order_by_focus
        paths, how = order_by_focus(paths)
        if how:
            _say(f"Frame order: near to far by {how}")
    return paths


def _filter_stack_paths(paths, reference):
    from utils.path_identity import same_path
    return [path for path in paths if not same_path(path, reference)]


def folder_args_mask_conflict(args, path):
    from utils.path_identity import same_path
    return bool(getattr(args, "defect_mask_path", None) and path
                and same_path(args.defect_mask_path, path))


def _sensor_mask(args):
    path = getattr(args, "defect_mask_path", None)
    if not path:
        return None
    from utils.sensor_cleanup import load_sensor_defect_mask
    return load_sensor_defect_mask(path, cancel_callback=getattr(args, "_should_cancel", None))


def _validate_sensor_sources(paths, args):
    if getattr(args, "defect_mask_path", None):
        from utils.sensor_cleanup import validate_sensor_source_orientation
        validate_sensor_source_orientation(paths, cancel_callback=getattr(args, "_should_cancel", None))


@contextmanager
def _video_directory(args, directory):
    import tempfile
    from core import scratch_registry

    recorded = []
    try:
        with tempfile.TemporaryDirectory(prefix='openfocus_video_', dir=directory) as path:
            args._video_scratch = recorded
            recorded.append(path)
            try:
                scratch_registry.register(path)
            except Exception as exc:
                _log.warning('could not register video scratch directory: %s', exc)
            yield path
    finally:
        args._video_scratch = None
        try:
            scratch_registry.forget([path for path in recorded if not os.path.exists(path)])
        except Exception as exc:
            _log.warning('could not forget video scratch: %s', exc)


def _resolve_stack(args, video_temp_root: str) -> Tuple[str, List[str]]:
    """Return (source_label, image file paths) from folders, videos, or files.

    Video frames are materialized as PNGs under video_temp_root (a
    TemporaryDirectory managed by the caller) so downstream code can treat
    every input as a plain file path.
    """
    from core.image_loader import ImageStackLoader

    loader = ImageStackLoader()
    inputs = [os.path.abspath(p) for p in args.input]
    paths: List[str] = []

    for entry in inputs:
        if os.path.isdir(entry):
            paths.extend(_folder_stack_paths(entry, loader, args))
        elif loader.is_video_file(entry):
            if getattr(args, "native_raw", False):
                raise ValueError("Native RAW fusion does not accept video input")
            if getattr(args, "defect_mask_path", None):
                raise ValueError("Cannot use a sensor defect mask with video frames: "
                                 "their sensor orientation is unknown. Use sensor-oriented "
                                 "still images or disable the defect mask.")
            ok, message, frames, _names = loader.load_from_video(entry)
            if not ok:
                raise RuntimeError(f"Failed to read video {entry}: {message}")
            for i, frame in enumerate(frames):
                frame_path = os.path.join(video_temp_root, f"video_{len(paths):05d}.png")
                recorded = getattr(args, '_video_scratch', None)
                if recorded is not None:
                    recorded.append(frame_path)
                    try:
                        from core import scratch_registry
                        scratch_registry.register(frame_path)
                    except Exception as exc:
                        _log.warning('could not register video scratch frame: %s', exc)
                from utils.image_utils import imwrite_auto
                if not imwrite_auto(frame_path, frame):
                    raise RuntimeError(f"Failed to write frame {i + 1} from video {entry}")
                paths.append(frame_path)
        elif os.path.isfile(entry):
            paths.append(entry)
        else:
            raise FileNotFoundError(f"Input not found: {entry}")
    paths = _filter_stack_paths(paths, args.dust_map)
    paths = _filter_stack_paths(paths, getattr(args, "defect_mask_path", None))
    _validate_sensor_sources(paths, args)

    # A single multi-page TIFF is a whole stack: count its pages, not files
    from core.image_loader import count_frames
    total_frames = len(paths) if getattr(args, "native_raw", False) else sum(count_frames(p) for p in paths)
    if total_frames < 2:
        raise RuntimeError("Need at least 2 images in the stack for fusion")
    return ", ".join(args.input), paths


def _load_images(paths: List[str], thread_count: int = 0,
                 resource_preference: str = "balanced",
                 should_cancel=None, registration_steps=0, exposure_match=False, lazy=False):
    with scratch_directory():
        return _load_images_admitted(paths, thread_count, resource_preference,
                                     should_cancel, registration_steps, exposure_match, lazy)


def _load_images_admitted(paths, thread_count, resource_preference,
                          should_cancel, registration_steps, exposure_match, lazy=False):
    from core.image_loader import iter_decoded_sources, _DecodeCancelled
    from core.frame_storage import is_mapped_frame, store_frame
    from core.frame_storage import select_scratch, estimate_registration_space, estimate_fusion_space
    from core.source_frames import scan_source
    # Admit known headers without consuming the decoder's streaming iterator.
    # Unknown formats remain subject to per-frame checks as they are decoded.
    manifest = []
    for path in paths:
        if should_cancel is not None and should_cancel():
            raise RuntimeError('Loading cancelled')
        try:
            manifest.extend(scan_source(path, header_only=True, should_cancel=should_cancel))
        except InterruptedError as exc:
            raise RuntimeError('Loading cancelled') from exc
        except (OSError, ValueError, ImportError):
            if lazy:
                raise
            continue
    if lazy:
        return manifest
    size = sum(frame.nbytes for frame in manifest)
    staging = estimate_registration_space(manifest, registration_steps) if registration_steps else 0
    staging += estimate_fusion_space(manifest, exposure_match)
    select_scratch(size, paths, preferred_size=size + staging)
    images = []
    try:
        decoded = iter_decoded_sources(
            paths, 1.0, thread_count=thread_count,
            resource_preference=resource_preference,
            should_cancel=should_cancel, allow_transparent_png=True,
            disk_backed=True)
        try:
            for item in decoded:
                if item.error is not None:
                    # The CLI is strict: one unreadable frame is an error,
                    # never a silently shorter stack.
                    raise RuntimeError(f"Failed to decode image: {item.path}: {item.error}")
                images.extend(page if is_mapped_frame(page) else store_frame(page)
                              for page in item.pages)
                del item
        finally:
            decoded.close()
    except _DecodeCancelled:
        raise RuntimeError("Loading cancelled")
    return images

def _validate_native_paths(paths, destinations=()):
    from core.image_loader import is_raw_source
    from utils.path_identity import same_path

    if len(paths) < 2:
        raise ValueError("Native RAW fusion requires at least two RAW files")
    for path in paths:
        if not is_raw_source(path):
            raise ValueError(f"Native RAW fusion requires RAW or DNG inputs: {path}")
    for destination in destinations:
        if destination and any(same_path(destination, path) for path in paths):
            raise ValueError(f"Output would overwrite an input image: {destination}")


def _load_native_stack(paths, args):
    from core.image_loader import ImageStackLoader

    loader = ImageStackLoader()
    if getattr(args, "_should_cancel", None) is not None:
        loader.should_cancel = args._should_cancel
    ok, message, images, _names = loader.load_linear_raw_stack(paths)
    if not ok:
        raise ValueError(message)
    metadata = loader.linear_dng_metadata
    if metadata is None:
        raise ValueError("Native RAW loader did not provide camera calibration")
    if getattr(metadata, "profile_tags", None):
        print("warning: this RAW stack contains a complex DNG color profile; "
              "the editable DNG retains it, but display rendering uses an "
              "approximate D65 matrix and white balance", file=sys.stderr)
    return images, metadata


def _save_native(path, fused, metadata, scale_cfg=None):
    from utils.linear_dng import develop_linear, write_linear_dng

    if os.path.splitext(path)[1].lower() == ".dng":
        if scale_cfg:
            raise ValueError("Scale bars are not supported in native DNG output")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        write_linear_dng(path, fused, metadata)
    else:
        if getattr(metadata, "profile_tags", None):
            print("warning: developed TIFF/PNG color is approximate; profile "
                  "transforms and baseline exposure are not applied", file=sys.stderr)
        from utils.image_utils import srgb_icc_profile
        _save(path, develop_linear(fused, metadata), scale_cfg,
              icc_profile=srgb_icc_profile())


def _suggest_duplicates(args):
    """Review ordered, single-frame files without changing the source stack."""
    from core.image_loader import ImageStackLoader, iter_decoded_sources
    from core.stack_review import analyze_frames
    from utils.path_identity import same_path

    if args.output or args.output_dir or args.json_progress or args.dry_run:
        print("error: --suggest-duplicates cannot be combined with --output, "
              "--output-dir, --json-progress or --dry-run", file=sys.stderr)
        return EXIT_USAGE
    if args.threads < 0:
        print("error: --threads must be >= 0 (0 = automatic)", file=sys.stderr)
        return EXIT_USAGE

    try:
        loader = ImageStackLoader()
        paths = []
        for source in args.input:
            source = os.path.abspath(source)
            if os.path.isdir(source):
                paths.extend(_list_folder_images(source, loader))
            elif os.path.isfile(source):
                if not source.lower().endswith(tuple(loader.SUPPORTED_FORMATS)):
                    raise ValueError(f"Expected an image file: {source}")
                paths.append(source)
            else:
                raise FileNotFoundError(f"Input not found: {source}")
        if len(paths) < 2:
            raise ValueError("Need at least 2 image files for review")
        if args.json and any(same_path(args.json, path) for path in paths):
            raise ValueError("--json must not overwrite an input image")

        def identity(path):
            stat = os.stat(path)
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

        snapshots = [identity(path) for path in paths]

        def frames():
            decoded = iter_decoded_sources(
                paths, 1.0, thread_count=args.threads,
                resource_preference=args.resource_preference,
                allow_transparent_png=True)
            try:
                for item in decoded:
                    if item.error is not None:
                        raise RuntimeError(f"Failed to decode image: {item.path}: {item.error}")
                    if len(item.pages) != 1:
                        raise ValueError(f"Review requires single-frame image files: {item.path}")
                    yield item.pages[0]
            finally:
                decoded.close()

        source_frames = frames()
        try:
            suggestions = analyze_frames(source_frames)
        finally:
            source_frames.close()
        if any(identity(path) != snapshot for path, snapshot in zip(paths, snapshots)):
            raise RuntimeError("Input files changed during review; run it again")
        payload = {"mode": "suggest-duplicates", "files": paths,
                   "suggestions": suggestions}
        result = json.dumps(payload, ensure_ascii=False)
        if args.json:
            import tempfile
            destination = os.path.abspath(args.json)
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False,
                                                 dir=os.path.dirname(destination),
                                                 prefix=".openfocus-review-", suffix=".tmp") as output:
                    temp_path = output.name
                    output.write(result + "\n")
                if any(same_path(destination, path) for path in paths):
                    raise ValueError("--json must not overwrite an input image")
                os.replace(temp_path, destination)
                temp_path = None
            finally:
                if temp_path is not None:
                    os.unlink(temp_path)
        print(result, file=_machine_stream or sys.stdout)
        return EXIT_OK
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _normalize_kernel(kernel: Optional[int], default: int) -> int:
    size = int(kernel) if kernel else default
    if size % 2 == 0:
        size = max(1, size - 1)
    return max(1, size)


def _transparent_region(images, args):
    """Return a common constant-alpha rectangle, excluding hidden pixels."""
    if not any(image.ndim == 3 and image.shape[2] == 4 for image in images):
        return images, None
    if (args.align != "none" or getattr(args, "dust_map", None)
            or getattr(args, "defect_mask_path", None) or getattr(args, "sensor_denoise", False)
            or getattr(args, "depth_map", None) or getattr(args, "scale_bar", False)
            or getattr(args, "scale_um", None) is not None):
        raise ValueError("Transparent PNG stacks require no alignment, dust map, sensor cleanup, depth map or scale bar")
    from core.source_frames import crop_transparent_frames
    return crop_transparent_frames(images, getattr(args, '_should_cancel', None))


def _fuse_stack(images, args):
    with heavy_job(getattr(args, "_should_cancel", None)), scratch_directory():
        return _fuse_stack_admitted(images, args)


def _fuse_stack_admitted(images, args):
    from core.algorithms import alignment_steps, get_fusion
    from core.frame_resolution_cache import (mandatory_staging_bytes,
                                             render_scope)
    frames = list(images) if isinstance(images, (list, tuple)) else []
    mandatory = 0
    if frames:
        steps = (len(alignment_steps(args.align))
                 if getattr(args, "align", "none") != "none" else 0)
        mandatory = mandatory_staging_bytes(
            frames, alignment_steps=steps,
            exposure_match=bool(getattr(args, "exposure_match", True)),
            deferred=bool(get_fusion(args.method).streaming_sources))
    from core.resource_policy import log_render_resources
    log_render_resources('start', getattr(args, "resource_preference", "balanced"))
    try:
        with render_scope(frames, mandatory_bytes=mandatory,
                          should_cancel=getattr(args, '_should_cancel', None)), \
                performance_profile.current().stage('render', resources=True):
            return _fuse_stack_admitted_body(images, args)
    finally:
        log_render_resources('end')


def _fuse_stack_admitted_body(images, args):
    """Register and fuse one stack; returns the fused image."""
    import cv2
    from core.registration import ImageRegistration
    from core.multi_focus_fusion import MultiFocusFusion
    from utils.image_utils import prepare_fusion_input, quantize_fuse_output
    from core.frame_storage import store_frame
    from core.source_frames import materialize_frames, process_frame

    _resolve_algorithm_arguments(args)
    streaming_sources = get_fusion(args.method).streaming_sources
    should_cancel = getattr(args, '_should_cancel', None)
    if not streaming_sources:
        images = materialize_frames(images, should_cancel)
    images, transparent = _transparent_region(images, args)
    if getattr(args, "defect_mask_path", None) or getattr(args, "sensor_denoise", False):
        from utils.sensor_cleanup import cleanup_sensor_image, require_sensor_cleanup_stack
        require_sensor_cleanup_stack(images, getattr(args, "defect_mask_path", None),
                                     getattr(args, "sensor_denoise", False), disk_backed=True)
        mask = _sensor_mask(args)
        def clean(image):
            return cleanup_sensor_image(image, mask, getattr(args, 'sensor_denoise', False),
                                        cancel_callback=should_cancel)
        images = ([process_frame(image, clean, should_cancel=should_cancel) for image in images]
                  if streaming_sources else [store_frame(clean(image)) for image in images])
    # Flat-field correction also runs in the raw pixel domain before registration.
    dust_white = getattr(args, "dust_map", None)
    if dust_white:
        from utils.dust_map import compute_flat_field, apply_flat_field, load_flat_field_reference
        try:
            white, _ = load_flat_field_reference(dust_white)
        except ValueError as exc:
            raise RuntimeError(f"Failed to decode white-field image: {dust_white}") from exc
        gain = compute_flat_field(white)
        images = ([process_frame(image, lambda pixels: apply_flat_field(pixels, gain),
                                 should_cancel=should_cancel) for image in images]
                  if streaming_sources else [store_frame(apply_flat_field(image, gain)) for image in images])
        _say(f"Flat-field (dust map) correction applied from {dust_white}")
    preordered = (args.align == 'openfocus' and get_fusion(args.method).sort_frames
                  and getattr(args, 'sort', 'asc') == 'desc')
    if preordered:
        images = list(reversed(images))
    args._alignment_reversed = preordered
    fusion_order = 'asc' if preordered else getattr(args, 'sort', 'asc')


    if args.align != "none":
        defer_warps = streaming_sources
        if not defer_warps:
            images = materialize_frames(images, should_cancel)
        t0 = time.time()
        reg = ImageRegistration(
            method=args.align, downscale_width=args.downscale,
            parameters=args._registration_parameters,
            brightness_reference=("last" if getattr(args, "exposure_match", True)
                                  else "first"),
            use_gpu=False, defer_warps=defer_warps)
        with performance_profile.current().stage('registration', resources=True):
            images = reg.process(images, output_path=None, thread_count=args.threads,
                                 resource_preference=getattr(args, "resource_preference", "balanced"),
                                 should_cancel=should_cancel)
        _say(f"Registration ({args.align}) done in {time.time() - t0:.1f}s")

    spec = get_fusion(args.method)
    fusion_kwargs = dict(algorithm=args.method, use_gpu=not args.cpu,
                         fusion_parameters=args._fusion_parameters,
                         resource_preference=getattr(args, "resource_preference", "balanced"),
                         base_frame=getattr(args, "base_frame", "none"),
                         sort_mode=fusion_order,
                         exposure_match=getattr(args, "exposure_match", True),
                         ai_batch_size=args.batch_size)
    if args.tile_size:
        fusion_kwargs["tile_block_size"] = args.tile_size
    if not getattr(args, "tile_enabled", True):
        fusion_kwargs["tile_enabled"] = False
    if getattr(args, "tile_overlap", None) is not None:
        fusion_kwargs["tile_overlap"] = args.tile_overlap
    if getattr(args, "tile_threshold", None) is not None:
        fusion_kwargs["tile_threshold"] = args.tile_threshold
    fusion = MultiFocusFusion(**fusion_kwargs)
    from core.fusion_diagnostics import log_effective_fusion
    log_effective_fusion(fusion, images, origin='cli', thread_count=args.threads,
        registration=alignment_steps(args.align), registration_parameters=args._registration_parameters,
        registration_downscale_width=args.downscale,
        registration_brightness_reference='last' if getattr(args, 'exposure_match', True) else 'first',
        reverse_registration=preordered, sharpen_output=bool(getattr(args, 'sharpen_output', False)),
        sensor_denoise=bool(getattr(args, 'sensor_denoise', False)),
        defect_mask=bool(getattr(args, 'defect_mask_path', None)),
        flat_field=bool(getattr(args, 'dust_map', None)), grouped=bool(getattr(args, '_grouped', False)))
    profile = performance_profile.current()
    with profile.stage('prepare_input'):
        images, is_16bit = prepare_fusion_input(images, getattr(args, "_should_cancel", None))
    t0 = time.time()
    fuse_kwargs = dict(input_source=images, img_resize=None, thread_count=args.threads,
                       should_cancel=should_cancel)
    model_weights = _method_weights(spec)
    if model_weights:
        fuse_kwargs["model_path"] = model_weights
    with profile.stage('fusion', resources=True):
        fused = fusion.fuse(**fuse_kwargs)
    if profile.enabled and fused is not None:
        profile.set('render', algorithm=args.method,
                    mode=args._fusion_parameters.get('mode'), frames=len(images),
                    source_dtype=str(images[0].dtype) if len(images) else None,
                    output_shape=list(fused.shape), device=fusion.get_info()['device'],
                    resource_preference=getattr(args, "resource_preference", "balanced"),
                    use_gpu_requested=not args.cpu, tile_block_size=fusion.tile_block_size,
                    tile_overlap=fusion.tile_overlap, tile_enabled=fusion.tile_enabled)
        profile.set('threads', requested=args.threads)
    info = fusion.get_info()
    if fused is None:
        raise RuntimeError("Fusion returned no result")
    if getattr(args, "sharpen_output", False):
        from utils.output_sharpen import sharpen_output
        fused = sharpen_output(fused)
    fused = quantize_fuse_output(fused, is_16bit)
    from core.transparency import restore_transparent_canvas
    fused = restore_transparent_canvas(fused, transparent)
    # 保留配准后的源帧供 --depth-map 使用；深度计算负责位深规范化。帧序必须与
    # 融合内部一致：--sort desc 会在 fuse() 内反转帧序，深度图若仍按原序
    # 计算，0/N-1 标签就指向相反的一端。
    if not getattr(args, "_grouped", False):
        depth_frames = list(images)
        if spec.sort_frames:
            from fusion_methods.openfocus_ab import sort_frames_for_openfocus
            depth_frames, _flipped = sort_frames_for_openfocus(
                depth_frames, fusion_order)
        args._depth_frames = depth_frames
    _say(f"Fusion ({args.method}, device={info['device']}) done in {time.time() - t0:.1f}s")
    return fused


def _inspect_grouped_cli(paths, args):
    from core.grouped_fusion import inspect_grouped_inputs
    _resolve_algorithm_arguments(args)
    if (args.align == 'openfocus'
            and args._registration_parameters.get('reference_frame', 1) != 1):
        raise ValueError('Grouped fusion cannot use a stack-wide alignment reference; use frame 1 or disable grouping')
    inspect_grouped_inputs(paths, should_cancel=getattr(args, "_should_cancel", None))


def _fuse_grouped(paths, args):
    """Fuse an ordered folder in bounded groups, then combine the results."""
    from core.grouped_fusion import grouped_fuse
    _inspect_grouped_cli(paths, args)

    group_args = argparse.Namespace(**vars(args))
    group_args._grouped = True
    group_args.sharpen_output = False
    group_args.dust_map = None
    group_args.defect_mask_path = None
    group_args.sensor_denoise = False
    gain = None
    if args.dust_map:
        from utils.dust_map import grouped_flat_field_gain
        gain = grouped_flat_field_gain(args.dust_map)
    preprocess = None
    mask = None
    if gain is not None or getattr(args, "defect_mask_path", None) or getattr(args, "sensor_denoise", False):
        def preprocess(image):
            nonlocal mask
            if getattr(args, "defect_mask_path", None) or getattr(args, "sensor_denoise", False):
                from utils.sensor_cleanup import cleanup_sensor_image, require_sensor_cleanup_stack
                mask_path = getattr(args, "defect_mask_path", None)
                require_sensor_cleanup_stack([image], mask if mask is not None else mask_path,
                                             getattr(args, "sensor_denoise", False))
                if mask is None and mask_path:
                    mask = _sensor_mask(args)
                image = cleanup_sensor_image(image, mask,
                                             getattr(args, "sensor_denoise", False),
                                             cancel_callback=getattr(args, "_should_cancel", None))
            if gain is not None:
                from utils.dust_map import apply_flat_field
                image = apply_flat_field(image, gain)
            return image
    # Reverse the whole source once. Reversing each group independently would
    # interleave the original order and reverse it again at every later stage.
    if get_fusion(args.method).sort_frames and args.sort == "desc":
        paths = list(reversed(paths))
        group_args.sort = "asc"

    _say("Grouped fusion is approximate and uses temporary disk space; "
         "registration and exposure matching use each group's reference.")

    def progress(stage, completed, total):
        _say(f"Group stage {stage}: {completed}/{total}")
        emit = getattr(args, "_progress", None)
        if emit is not None:
            emit(event="group", stage=stage, done=completed, total=total)

    result = grouped_fuse(paths, args.group_size, args.group_overlap,
                          lambda images: _fuse_stack_admitted(images, group_args),
                          thread_count=args.threads,
                          resource_preference=args.resource_preference,
                          should_cancel=getattr(args, "_should_cancel", None),
                          progress=progress, preprocess=preprocess)
    if args.sharpen_output:
        from utils.image_utils import normalize_fuse_input, quantize_fuse_output
        from utils.output_sharpen import sharpen_output
        frames, is_16bit = normalize_fuse_input([result])
        result = quantize_fuse_output(sharpen_output(frames[0]), is_16bit)
    return result


def _method_weights(spec) -> str | None:
    """Resolve the descriptor's checkpoint; back-ends own their default too."""
    if not spec.weights:
        return None
    from utils.weights import weights_path
    return weights_path(os.path.basename(spec.weights))


def _depth_out_path(args, fused_out: str) -> str:
    dm = args.depth_map
    if isinstance(dm, str) and dm:
        return dm
    stem, _ = os.path.splitext(fused_out)
    return stem + "_depth.png"


def _validate_alignment_report_path(args, paths=()):
    from utils.path_identity import same_path

    protected = [*args.input, *paths, args.output, args.json, args.dust_map,
                 args.defect_mask_path, args.config]
    if args.depth_map and args.output:
        protected.append(_depth_out_path(args, args.output))
    if any(same_path(args.alignment_report, path) for path in protected):
        raise ValueError("--alignment-report must differ from input and output paths")


def _replace_text_file(path, text, prefix, before_replace=None):
    """Create the folder, then replace ``path`` only once ``text`` is fully written."""
    import tempfile

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=prefix, suffix=".tmp",
                                         delete=False) as stream:
            temporary = stream.name
            stream.write(text)
        if before_replace is not None:
            before_replace()
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


def _write_alignment_report(path, payload):
    _replace_text_file(path, json.dumps(payload, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n", ".openfocus_alignment_")


def _save_depth(args, fused_out: str) -> None:
    frames = getattr(args, "_depth_frames", None)
    if not frames:
        raise RuntimeError("--depth-map: no fused frames available")
    from core.depth_map import compute_focus_index
    from utils.colormap import colorize
    from utils.image_utils import imwrite_auto

    path = _depth_out_path(args, fused_out)
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    spec = get_fusion(args.method)
    if spec.depth is None:
        raise RuntimeError(f"--depth-map is not available for method '{args.method}'")
    depth_keys = {parameter.key for parameter in spec.depth.parameters}
    parameters = {key: value for key, value in args._fusion_parameters.items() if key in depth_keys}
    if "kernel_size" in depth_keys and getattr(args, "kernel", None) is not None:
        default = next(parameter.default for parameter in spec.depth.parameters if parameter.key == "kernel_size")
        parameters.setdefault("kernel_size", _normalize_kernel(args.kernel, default))
    idx = compute_focus_index(
        frames, args.method, parameters=parameters,
        model_path=_method_weights(spec),
        thread_count=getattr(args, "threads", 0),
        resource_preference=getattr(args, "resource_preference", "balanced"),
        use_gpu=not args.cpu)
    if not imwrite_auto(path, colorize(idx, args.depth_colormap)):
        raise RuntimeError(f"Failed to write depth map: {path}")
    _say(f"Depth map ({args.depth_colormap}) saved: {path} "
          f"[{time.time() - t0:.1f}s]")


def _extra_out_path(value, fused_out: str, suffix: str, ext: Optional[str] = None) -> str:
    """An explicit PATH, or <fused stem><suffix><ext or the fused extension>."""
    if isinstance(value, str) and value:
        return value
    stem, fused_ext = os.path.splitext(fused_out)
    return stem + suffix + (fused_ext if ext is None else ext)


def _background_outputs(args, fused_out: str):
    """(flag, path) for every extra file --auto-background writes for `fused_out`."""
    outputs = []
    if getattr(args, "save_stacked", None):
        outputs.append(("--save-stacked",
                        _extra_out_path(args.save_stacked, fused_out, "_stacked")))
    if getattr(args, "missing_data_map", None):
        overlay = _extra_out_path(args.missing_data_map, fused_out, "_missing", ".png")
        import dataclasses
        from core.background_fill import MissingData
        stem = overlay[:-4]
        outputs.append(("--missing-data-map", overlay))
        outputs += [(f"--missing-data-map ({field.name} mask)", f"{stem}-{field.name}.png")
                    for field in dataclasses.fields(MissingData)]
    return outputs


def _background_out_paths(args, fused_out: str) -> List[str]:
    """Every extra file --auto-background will write for `fused_out`."""
    return [path for _flag, path in _background_outputs(args, fused_out)]


def _result_outputs(args, fused_out: str):
    """(flag, path) for every file one result run writes for `fused_out`."""
    outputs = [("--output", fused_out)]
    if getattr(args, "depth_map", None):
        outputs.append(("--depth-map", _depth_out_path(args, fused_out)))
    if getattr(args, "auto_background", None):
        outputs += _background_outputs(args, fused_out)
    return outputs


def _profile_destination(args):
    """The --profile-performance FILE, or None when no file was requested."""
    path = getattr(args, "profile_performance", None)
    return path if isinstance(path, str) else None


def _run_outputs(args, fused_out: str):
    """`_result_outputs` plus the run-wide JSON summary, alignment report and profile."""
    outputs = _result_outputs(args, fused_out)
    for flag, path in (("--json", getattr(args, "json", None)),
                       ("--alignment-report", getattr(args, "alignment_report", None)),
                       ("--profile-performance", _profile_destination(args))):
        if path:
            outputs.append((flag, path))
    return outputs


def _first_duplicate(outputs):
    """The first (path, flag, earlier flag) whose destination conflicts.

    Two names for one file (a hard link, a symlinked directory, Windows
    case/8.3 aliases) count as one destination. A file also cannot serve as
    another output's parent directory.
    """
    seen_paths, seen_files, seen_parents = {}, {}, {}
    for flag, path in outputs:
        if not path:
            continue
        key = path_key(path)
        if key in seen_paths:
            return path, flag, seen_paths[key][0]
        if key in seen_parents:
            return path, flag, seen_parents[key]
        parent = os.path.dirname(key)
        while parent != key:
            if parent in seen_paths:
                return path, flag, seen_paths[parent][0]
            seen_parents.setdefault(parent, flag)
            key_parent = os.path.dirname(parent)
            if key_parent == parent:
                break
            parent = key_parent
        seen_paths[key] = (flag, path)
        identity = file_identity(path)
        if identity is None:
            continue
        if identity in seen_files:
            return path, flag, seen_files[identity][0]
        seen_files[identity] = (flag, path)
    return None


def _input_key_sets(candidates):
    """(path keys, file identities) for inputs no output may overwrite."""
    keys, identities = set(), set()
    for candidate in candidates:
        if not candidate:
            continue
        keys.add(path_key(candidate))
        identity = file_identity(candidate)
        if identity is not None:
            identities.add(identity)
    return keys, identities


def _admitted_input_keys(candidates):
    """Collect identities only when every selected source can be stat'ed."""
    keys, identities = set(), set()
    for candidate in candidates:
        if not candidate:
            continue
        info = os.stat(candidate)
        keys.add(path_key(candidate))
        if info.st_ino:
            identities.add((info.st_dev, info.st_ino))
    return keys, identities


def _single_inputs(args, loader):
    import stat

    candidates = [getattr(args, "dust_map", None),
                  getattr(args, "defect_mask_path", None),
                  getattr(args, "config", None)]
    for entry in args.input:
        info = os.stat(entry)
        if stat.S_ISDIR(info.st_mode):
            candidates.extend(_list_folder_images(entry, loader))
        else:
            candidates.append(entry)
    return candidates, _admitted_input_keys(candidates)


def _first_input_overlap(outputs, keys, identities):
    """The first (flag, path) planned output that would overwrite an input."""
    for flag, path in outputs:
        if not path:
            continue
        if path_key(path) in keys or file_identity(path) in identities:
            return flag, path
    return None


def _output_over_inputs(args, fused_out: str, paths):
    """The first (flag, path) planned output that would overwrite an input."""
    entries = [entry for entry in (getattr(args, "input", None) or ())
               if isinstance(entry, str) and os.path.isfile(entry)]
    keys, identities = _input_key_sets([
        *paths,
        getattr(args, "dust_map", None),
        getattr(args, "defect_mask_path", None),
        getattr(args, "config", None),
        *entries,
    ])
    return _first_input_overlap(_run_outputs(args, fused_out), keys, identities)


def _batch_inputs(args, folders, loader):
    """Collect all batch sources without decoding or inspecting frame counts."""
    import stat

    candidates = [getattr(args, "config", None)]
    for entry in args.input:
        info = os.stat(entry)
        if not stat.S_ISDIR(info.st_mode):
            candidates.append(entry)
    for folder in folders:
        name = os.path.basename(folder.rstrip("/\\")) or "stack"
        folder_args, _override = _folder_run_args(args, folder, name)
        candidates.extend(_list_folder_images(folder, loader))
        candidates.append(getattr(folder_args, "dust_map", None))
        candidates.append(getattr(folder_args, "defect_mask_path", None))
        candidates.append(getattr(folder_args, "config", None))
    return candidates, _admitted_input_keys(candidates)


def _folder_run_args(args, folder: str, name: str):
    """The namespace one batch folder runs with (per-folder --config merge)."""
    overrides = getattr(args, "_folder_overrides", None) or {}
    override = overrides.get(name) or overrides.get(folder)
    if not override:
        return args, None
    merged = argparse.Namespace(**{**vars(args), **override})
    _validate_execution_args(merged)
    _resolve_algorithm_arguments(merged)
    return merged, override


def _auto_background(args, fused):
    """Replace the background of a fused result (Retouch > Auto background).

    Uses the registered frames the fusion consumed, cropped to the result
    like the GUI's render context.  Returns the `BackgroundFillResult`."""
    from core.background_fill import background_working_set_bytes, replace_background
    from core.render_context import _alignment_valid_mask

    frames = getattr(args, "_depth_frames", None)
    if not frames:
        raise RuntimeError("--auto-background: no aligned frames available")
    if fused.ndim == 3 and fused.shape[2] == 4:
        raise RuntimeError("--auto-background does not support transparent results")
    h, w = fused.shape[:2]
    if any(f.shape[0] < h or f.shape[1] < w for f in frames):
        raise RuntimeError("--auto-background: the aligned frames are smaller than the result")
    frames = [f[:h, :w] for f in frames]
    # Admit the analysis before building anything: one mask byte per pixel
    # per aligned frame plus the mask construction's scratch, as in Retouch.
    aligned = args.align != "none"
    mask_bytes = h * w * len(frames) if aligned else 0
    mask_scratch = h * w * 16 if aligned else 0
    require_memory(background_working_set_bytes(h, w, len(frames)) + mask_bytes + mask_scratch)
    # Registration fills uncovered borders with black; never sample those.
    valid = [_alignment_valid_mask(f) for f in frames] if aligned else None
    progress = None
    if getattr(args, "json_progress", False):
        def progress(fraction, message):
            args._progress(event="background", stage=message, fraction=round(fraction, 3))
    t0 = time.time()
    try:
        result = replace_background(
            frames, fused, preset=args.auto_background,
            clean=args.background_mode == "clean",
            keep_soft_parts=bool(getattr(args, "keep_soft_parts", False)),
            valid_masks=valid, progress=progress,
            cancelled=getattr(args, "_should_cancel", None))
    finally:
        del frames, valid
    if result.changed:
        _say(f"Auto background ({args.auto_background} -> {result.preset_used}, "
             f"{args.background_mode}): frame {result.background_frame + 1} of "
             f"{len(args._depth_frames)}, subject {result.subject_fraction:.0%} of the image "
             f"[{time.time() - t0:.1f}s]")
    else:
        _say(f"Auto background: the subject covers {result.subject_fraction:.0%} of the image "
             f"(a ground-level scene), so the background was left as stacked "
             f"[{time.time() - t0:.1f}s]")
    return result


def _save_background_extras(args, fused_out, stacked, result, scale_cfg, icc_profile) -> List[str]:
    """Write --save-stacked / --missing-data-map files; returns their paths."""
    written = []
    if getattr(args, "save_stacked", None):
        path = _extra_out_path(args.save_stacked, fused_out, "_stacked")
        _save(path, stacked, scale_cfg, icc_profile)
        _say(f"Saved stacked result: {path}")
        written.append(path)
    if getattr(args, "missing_data_map", None):
        from core.background_fill import missing_data_overlay
        from utils.image_utils import imwrite_auto
        overlay = _extra_out_path(args.missing_data_map, fused_out, "_missing", ".png")
        os.makedirs(os.path.dirname(os.path.abspath(overlay)), exist_ok=True)
        stem = overlay[:-4]
        outputs = [(overlay, missing_data_overlay(result.image, result.missing))]
        outputs += [(f"{stem}-{name}.png", layer) for name, layer, _c in result.missing.layers()]
        for path, image in outputs:
            if not imwrite_auto(path, image):
                raise RuntimeError(f"Failed to write missing-data map: {path}")
            written.append(path)
        _say(f"Missing-data map saved: {overlay} (+{len(outputs) - 1} masks)")
    return written


def _background_summary(result) -> dict:
    return {"preset_used": result.preset_used, "background_frame": result.background_frame,
            "changed": bool(result.changed),
            "subject_fraction": round(float(result.subject_fraction), 4)}


def _resolve_scale_cfg(args, paths):
    """Scale-bar config from --scale-bar/--scale-um + TIFF/DNG metadata.

    Returns None when the bar was not requested or no calibration is
    available (raising in the latter case only when the user asked for it).
    """
    if not args.scale_bar and args.scale_um is None:
        return None
    px_um = args.scale_um
    if px_um is None:
        from utils.scalebar import detect_px_size_um
        for p in paths:
            px_um = detect_px_size_um(p)
            if px_um:
                break
    if not px_um or px_um <= 0:
        raise RuntimeError(
            "--scale-bar: no pixel size available — pass --scale-um <um/px>")
    return {"px_um": px_um, "position": "bottom-right", "color": "auto"}


def _save(path: str, fused, scale_cfg=None, icc_profile=b"") -> None:
    from utils.image_utils import imwrite_auto
    if fused.ndim == 3 and fused.shape[2] == 4:
        if os.path.splitext(path)[1].lower() != '.png':
            raise ValueError("Transparent output requires a PNG file")
        if scale_cfg:
            raise ValueError("Scale bars are not supported on transparent output")
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    if scale_cfg:
        from utils import scalebar
        fused = scalebar.burn(fused, scale_cfg)
    if not imwrite_auto(path, fused, icc_profile=icc_profile):
        raise RuntimeError(f"Failed to write output: {path}")


def _run_batch(args, _emit_json=None, _allow_summary=None) -> int:
    """Batch mode: fuse every input folder into --output-dir."""
    from core.image_loader import ImageStackLoader

    loader = ImageStackLoader()
    folders = [os.path.abspath(p) for p in args.input if os.path.isdir(p)]
    machine = getattr(args, '_machine_state', None)
    if machine is not None:
        machine['total'] = len(folders)
    skipped = [p for p in args.input if not os.path.isdir(os.path.abspath(p))]
    for p in skipped:
        print(f"warning: skipping non-folder input in batch mode: {p}", file=sys.stderr)
    if not folders:
        print("error: --output-dir requires image folder(s) as --input", file=sys.stderr)
        return EXIT_USAGE
    if args.output:
        print(f"warning: --output is ignored in batch mode (using --output-dir)", file=sys.stderr)

    # Every file the batch will write -- each folder's result, depth map and
    # background extras (one mask per missing-data class included), plus the
    # run-wide JSON summary -- planned before the first write.  Folder names
    # only decide the main path, so one folder's sidecar can land on another
    # folder's result, and a config can change which sidecars a folder writes.
    planned = []
    folder_plans = []
    for folder in folders:
        name = os.path.basename(folder.rstrip("/\\")) or "stack"
        folder_args, override = _folder_run_args(args, folder, name)
        out_path = os.path.join(args.output_dir, f"{name}{args._batch_ext}")
        results = _result_outputs(folder_args, out_path)
        folder_plans.append((folder, name, folder_args, override, out_path, results))
        planned.extend(results)
    for flag, path in (("--json", args.json), ("--alignment-report", args.alignment_report)):
        if path:
            planned.append((flag, path))
    duplicate = _first_duplicate(planned)
    if duplicate is not None:
        path, flag, earlier = duplicate
        if flag == earlier:
            print(f"error: two folders would write to {path}; "
                  f"rename one of them or choose another --output-dir", file=sys.stderr)
        else:
            print(f"error: {path} would be written twice ({earlier} and {flag}); "
                  f"rename the folder or choose another --output-dir", file=sys.stderr)
        return EXIT_USAGE
    # An output may not replace a frame, the dust or defect reference, or the
    # config in any folder.  Nothing is written (not even the output dir)
    # until this holds; if the inputs cannot be listed, refuse the run rather
    # than write without the guarantee.
    try:
        candidates, (input_keys, input_files) = _batch_inputs(args, folders, loader)
    except Exception as exc:  # noqa: BLE001 - refuse rather than write blind
        print(f"error: cannot list the batch inputs: {exc}", file=sys.stderr)
        return EXIT_USAGE
    overlap = _first_input_overlap(planned, input_keys, input_files)
    if overlap is not None:
        flag, path = overlap
        print(f"error: {flag} would overwrite an input image: {path}", file=sys.stderr)
        return EXIT_USAGE
    if _allow_summary is not None:
        _allow_summary(planned, candidates, input_keys, input_files)
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    ok_count, failures = 0, []
    skipped_existing = []
    if machine is not None:
        machine['failed'] = failures
        machine['skipped_existing'] = skipped_existing
    started = time.time()
    for i, (folder, name, folder_args, override, out_path, results) in enumerate(folder_plans):
        if (folder_args_mask_conflict(folder_args, out_path) or
                (folder_args.depth_map and folder_args_mask_conflict(
                    folder_args, _depth_out_path(folder_args, out_path)))
                or (folder_args.auto_background and any(
                    folder_args_mask_conflict(folder_args, path)
                    for path in _background_out_paths(folder_args, out_path)))):
            print(f"error: output would overwrite defect mask: {out_path}", file=sys.stderr)
            failures.append(name)
            continue
        _say(f"\n=== {name} ===")

        if args.resume:
            import stat
            complete = True
            try:
                for _flag, result_path in results:
                    try:
                        info = os.stat(result_path)
                    except FileNotFoundError:
                        complete = False
                        continue
                    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
                        complete = False
            except OSError as exc:
                print(f"error: {name}: cannot inspect {result_path}: {exc}", file=sys.stderr)
                failures.append(name)
                continue
            if complete:
                _say(f"skip (all requested outputs exist): {out_path}")
                skipped_existing.append(name)
                ok_count += 1
                if machine is not None:
                    machine['succeeded'] = ok_count
                continue

        if override:
            applied = ", ".join(f"{k}={v}" for k, v in override.items())
            _say(f"config override: {applied}")

        if args.dry_run:
            try:
                paths = _filter_stack_paths(_folder_stack_paths(folder, loader, folder_args), folder_args.dust_map)
                paths = _filter_stack_paths(paths, getattr(folder_args, "defect_mask_path", None))
                _validate_sensor_sources(paths, folder_args)
                from core.image_loader import count_frames
                frames = len(paths) if args.native_raw else sum(count_frames(p) for p in paths)
                if frames < 2:
                    raise ValueError(f"need at least 2 frames, found {frames}")
                if folder_args.group_size:
                    _inspect_grouped_cli(paths, folder_args)
                _say(f"[dry-run] frames={frames} method={folder_args.method} "
                     f"align={folder_args.align} output={out_path}")
                ok_count += 1
                if machine is not None:
                    machine['succeeded'] = ok_count
            except Exception as exc:
                print(f"error: {name}: {exc}", file=sys.stderr)
                failures.append(name)
            continue

        progress = getattr(args, "_progress", None)
        if progress is not None:
            progress(event="folder", name=name, index=i + 1,
                     total=len(folders))
        scratch_scope = scratch_directory(separate=True)
        scratch_scope.__enter__()
        try:
            paths = _filter_stack_paths(_folder_stack_paths(folder, loader, folder_args), folder_args.dust_map)
            paths = _filter_stack_paths(paths, getattr(folder_args, "defect_mask_path", None))
            _validate_sensor_sources(paths, folder_args)
            if not folder_args.group_size:
                _warn_low_storage(folder_args, paths, folder=folder)
            # Count frames, not files: a folder holding one multi-page TIFF
            # Z-stack is a complete stack (same rule as single-stack mode).
            from core.image_loader import count_frames
            frame_count = len(paths) if args.native_raw else sum(count_frames(p) for p in paths)
            if frame_count < 2:
                raise RuntimeError(
                    f"need at least 2 frames, found {frame_count}")
            if args.native_raw:
                images, metadata = _load_native_stack(paths, folder_args)
                fused = _fuse_stack(images, folder_args)
            elif folder_args.group_size:
                fused = _fuse_grouped(paths, folder_args)
            else:
                images = _load_images(
                    paths, thread_count=folder_args.threads,
                    resource_preference=folder_args.resource_preference,
                    registration_steps=int(folder_args.align != 'none'),
                    exposure_match=getattr(folder_args, 'exposure_match', False),
                    lazy=get_fusion(folder_args.method).streaming_sources)
                fused = _fuse_stack(images, folder_args)
                images = None
            background = stacked = None
            if args.native_raw:
                _save_native(out_path, fused, metadata, _resolve_scale_cfg(folder_args, paths))
            else:
                from core.image_loader import common_icc_profile
                scale_cfg = _resolve_scale_cfg(folder_args, paths)
                icc_profile = common_icc_profile(paths)
                if getattr(folder_args, "auto_background", None):
                    stacked = fused
                    background = _auto_background(folder_args, fused)
                    fused = background.image
                _save(out_path, fused, scale_cfg, icc_profile)
            _say(f"Saved: {out_path}")
            if background is not None:
                _save_background_extras(folder_args, out_path, stacked, background,
                                        scale_cfg, icc_profile)
                background = stacked = None
            if folder_args.depth_map:
                _save_depth(folder_args, out_path)
            ok_count += 1
            if machine is not None:
                machine['succeeded'] = ok_count
        except Exception as exc:
            _log.exception("batch folder failed: %s", folder)
            print(f"error: {name}: {exc}", file=sys.stderr)
            failures.append(name)
        finally:
            # Each folder's aligned frames were kept for its depth map and
            # background; release them before the next folder loads.
            for holder in {id(args): args, id(folder_args): folder_args}.values():
                holder._depth_frames = None
            scratch_scope.__exit__(None, None, None)

    summary = {
        "mode": "batch", "dry_run": bool(args.dry_run),
        "method": args.method, "align": args.align,
        "total": len(folders),
        "succeeded": ok_count,
        "skipped_existing": skipped_existing,
        "failed": failures,
        "duration_s": round(time.time() - started, 1),
    }
    if machine is not None:
        machine['summary'] = summary
    _say(f"\nBatch {'validation' if args.dry_run else 'processing'} finished: "
         f"{ok_count}/{len(folders)} stacks {'validated' if args.dry_run else 'succeeded'}")
    if skipped_existing:
        _say(f"skipped (already done): {', '.join(skipped_existing)}")
    if failures:
        print("failed: " + ", ".join(failures), file=sys.stderr)
    if _emit_json is not None:
        _emit_json(summary)
    if failures:
        return EXIT_ERROR
    return EXIT_OK


def _say(*args, **kwargs):
    """Human-readable progress.

    With --json-progress stdout is already redirected to stderr, so the
    machine stream keeps stdout to itself; forcing stderr here broke plain
    stdout pipelines (dry-run reports).
    """
    print(*args, **kwargs)


def _ensure_utf8_stdio() -> None:
    """Keep non-ASCII stack paths from aborting progress output.

    Windows keeps the locale code page (cp1252 on some hosts) when stdout or
    stderr is redirected to a file or pipe. A Chinese folder name then raised
    UnicodeEncodeError inside ``print`` and killed a fusion that had already
    been requested. Consoles already use UTF-8; only redirected streams are
    reconfigured, and replacement keeps a broken pipe from hiding the result.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        encoding = (getattr(stream, "encoding", "") or "").replace("-", "").lower()
        if encoding == "utf8":
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            pass


def run_cli(argv: List[str]) -> int:
    _ensure_utf8_stdio()
    if argv:
        head = argv[0]
        if head in ("version", "--version", "-V"):
            print(app_display_version())
            return EXIT_OK
        if head in ("describe", "--describe"):
            print(json.dumps(describe_payload(), ensure_ascii=False))
            return EXIT_OK
        if head == "gui":
            return _run_gui(argv[1:])
        if head == "ctl":
            return _run_ctl(argv[1:])
        if head == "recipe":
            return _run_recipe_command(argv[1:])
        if head == "retouch":
            return _run_retouch_command(argv[1:])
    global _machine_stream
    redirected = "--json-progress" in argv or "--suggest-duplicates" in argv
    if redirected:
        _machine_stream = sys.stdout
        sys.stdout = _StderrProxy()
    setup_logging()
    log_session_start()
    try:
        return _run_with_logging(argv)
    finally:
        if redirected:
            sys.stdout = _machine_stream
            _machine_stream = None


def _run_retouch_command(argv: List[str]) -> int:
    """Replay a checkpoint against its original base and processed frames."""
    from core.image_loader import _read_icc_profile
    from core.render_context import RenderContext, SourceFrame
    from utils.retouch import RetouchDocument
    from utils.retouch_checkpoint import read_checkpoint

    parser = argparse.ArgumentParser(prog="openfocus retouch")
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("replay", help="replay a retouch checkpoint")
    replay.add_argument("checkpoint", help="retouch checkpoint JSON or exact archive")
    replay.add_argument("--base", required=True, help="original fused base image")
    replay.add_argument("--sources", nargs="+", required=True,
                        help="processed source images in checkpoint order")
    replay.add_argument("--output", "-o", required=True,
                        help="output PNG or TIFF")
    args = parser.parse_args(argv)

    try:
        if os.path.splitext(args.output)[1].lower() not in (".png", ".tif", ".tiff"):
            raise ValueError("replay output must be PNG or TIFF to preserve bit depth")
        output_path = os.path.normcase(os.path.realpath(os.path.abspath(args.output)))
        for path in [args.checkpoint, args.base, *args.sources]:
            input_path = os.path.normcase(os.path.realpath(os.path.abspath(path)))
            if output_path == input_path or (os.path.exists(args.output)
                    and os.path.exists(path) and os.path.samefile(args.output, path)):
                raise ValueError("output must not overwrite the checkpoint or an input image")

        checkpoint = read_checkpoint(args.checkpoint)
        if (not isinstance(checkpoint, dict) or type(checkpoint.get("version")) is not int
                or checkpoint["version"] not in (1, 2)):
            raise ValueError("unsupported retouch checkpoint version")
        records = checkpoint.get("frames") if isinstance(checkpoint, dict) else None
        if not isinstance(records, list) or len(records) != len(args.sources):
            raise ValueError("checkpoint frame count does not match --sources")

        def _recorded_ndim(fingerprint):
            shape = fingerprint.get("shape") if isinstance(fingerprint, dict) else None
            return len(shape) if isinstance(shape, list) else None

        recorded = [checkpoint.get("base"),
                    *(r.get("image") if isinstance(r, dict) else None for r in records)]
        images = []
        for path, fingerprint in zip([args.base, *args.sources], recorded):
            if _recorded_ndim(fingerprint) == 2:
                # Standalone retouch keeps grayscale as one channel; decode
                # the same way so shapes and hashes match the checkpoint.
                from utils.standalone_retouch import _read_single_image
                images.append(_read_single_image(path))
                continue
            decoded = _load_images([path])
            if len(decoded) != 1:
                raise ValueError(f"expected one image in {path}")
            images.append(decoded[0])
        base, sources = images[0], images[1:]
        frames = []
        for record, image, path in zip(records, sources, args.sources):
            if not isinstance(record, dict):
                raise ValueError("invalid checkpoint frame")
            frame_id = record.get("id")
            ox, oy, aligned = record.get("offset_x"), record.get("offset_y"), record.get("aligned")
            if (not isinstance(frame_id, str) or not frame_id
                    or type(ox) is not int or type(oy) is not int
                    or type(aligned) is not bool
                    or ox < 0 or oy < 0
                    or ox + image.shape[1] > base.shape[1]
                    or oy + image.shape[0] > base.shape[0]):
                raise ValueError("invalid checkpoint frame geometry or identity")
            frames.append(SourceFrame(frame_id, os.path.basename(path), image,
                                      offset_x=ox, offset_y=oy, aligned=aligned))
        del checkpoint, records, record
        context = RenderContext("replay", "replay", (), base.shape[1], base.shape[0],
                                base.dtype, tuple(frames))
        from core.resource_policy import require_memory
        require_memory(2 * base.nbytes)
        document = RetouchDocument(base, context)
        document.initialize_checkpoint()
        document.load_checkpoint(args.checkpoint)
        _save(args.output, document.result(), icc_profile=_read_icc_profile(args.base))
    except (ValueError, OSError, RuntimeError, TypeError, MemoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


def _run_recipe_command(argv: List[str]) -> int:
    """Capture or execute a portable processing recipe."""
    from core.recipes import capture_project_recipe, capture_recipe, run_recipe
    parser = argparse.ArgumentParser(prog="openfocus recipe")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="capture processing options")
    create.add_argument("file", help="recipe JSON file")
    create.add_argument("--project", help="capture settings from a project")
    run = commands.add_parser("run", help="run a recipe")
    run.add_argument("file", help="recipe JSON file")
    run.add_argument("--dry-run", action="store_true")
    if argv and argv[0] == "create":
        args, processing = parser.parse_known_args(argv)
        try:
            if args.project:
                capture_project_recipe(args.file, args.project, processing)
            else:
                capture_recipe(args.file, processing)
            return EXIT_OK
        except (ValueError, OSError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
    args = parser.parse_args(argv)
    try:
        return run_recipe(args.file, dry_run=args.dry_run)
    except (ValueError, OSError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


def _machine_event(state, **fields):
    if state['broken']:
        return
    if fields.get('event') == 'start':
        state['start_emitted'] = True
        state['started_at'] = time.time()
    elif fields.get('event') == 'done':
        if not state['start_emitted'] or state['terminal_emitted']:
            return
        state['terminal_emitted'] = True
    try:
        print(json.dumps(fields, ensure_ascii=False), file=_machine_stream or sys.stdout, flush=True)
    except BrokenPipeError:
        state['broken'] = True


def _run_with_logging(argv: List[str]) -> int:
    with log_operation("cli", args=argv) as operation:
        state = None
        result = EXIT_ERROR
        try:
            args = build_parser().parse_args(argv)
            state = args._machine_state = {
                'start_emitted': False, 'terminal_emitted': False, 'broken': False,
                'mode': 'batch' if args.output_dir is not None else 'single',
                'started_at': time.time(), 'total': 0, 'succeeded': 0,
                'failed': [], 'skipped_existing': [], 'cancelled': False,
            }
            directory = None
            if args.temp_dir is not None:
                from core.frame_storage import validate_storage_directory
                try:
                    directory = validate_storage_directory(args.temp_dir)
                except OSError as exc:
                    print(f"error: --temp-dir is not a writable directory: {exc}", file=sys.stderr)
                    return EXIT_USAGE
            try:
                from core import scratch_registry
                scratch_registry.sweep_in_background([directory] if directory else [])
            except Exception as exc:
                _log.warning('could not start scratch sweep: %s', exc)
            with heavy_job(), scratch_directory(separate=True, explicit_directory=directory):
                result = _run_cli_admitted(args)
            if result == EXIT_ERROR:
                operation.fail()
            log_action("cli.exit", exit_code=result)
            return result
        except KeyboardInterrupt:
            if state is not None:
                state['cancelled'] = True
            operation.cancel()
            print("interrupted", file=sys.stderr)
            return EXIT_ERROR
        finally:
            if state is not None and state['start_emitted']:
                done = {'event': 'done', 'ok': result == EXIT_OK and not state['cancelled'],
                        'mode': state['mode'], 'exit_code': result,
                        'duration_s': round(time.time() - state['started_at'], 1)}
                if state['mode'] == 'batch':
                    done.update({key: state[key] for key in
                                 ('total', 'succeeded', 'failed', 'skipped_existing')})
                elif 'outputs' in state:
                    done['outputs'] = state['outputs']
                if state['cancelled']:
                    done.update(cancelled=True, code='cancelled', exit_code=EXIT_ERROR)
                elif result == EXIT_USAGE:
                    done['code'] = 'usage_error'
                elif result == EXIT_ERROR:
                    done['code'] = 'processing_error'
                _machine_event(state, **done)


def _warn_low_storage(args, paths, *, folder=None):
    """Print a stderr warning when temporary storage looks low; never blocks."""
    try:
        from core.frame_storage import assess_scratch, readable_bytes
        from core.source_frames import scan_source
        from core.storage_preflight import render_storage_estimate
        frames = []
        for path in paths:
            # Every page: a multipage TIFF is rendered as a whole stack.
            frames.extend(scan_source(path, header_only=True))
        minimum, expected, optional, sources = render_storage_estimate(
            frames, algorithm=args.method, alignment=args.align,
            exposure_match=bool(getattr(args, "exposure_match", False)),
            cleanup=bool(args.defect_mask_path or args.sensor_denoise))
        assessment = assess_scratch(minimum, expected, optional, sources)
    except Exception:
        _log.info("storage preflight unavailable", exc_info=True)
        return None
    if assessment["status"] in ("advisory", "critical"):
        location = f" for {folder}" if folder is not None else ""
        directory = assessment['directory']
        where = f" in {directory}" if directory is not None else " across available volumes"
        bound = '' if directory is not None else 'at most '
        print(f"warning: low temporary storage{where}{location}: "
              f"{bound}{readable_bytes(assessment['available_bytes'])} available after the safety "
              f"reserve, an estimated {readable_bytes(assessment['required_bytes'])} needed "
              f"(an estimate; caches may be disabled or the render may fail). "
              f"Use --temp-dir DIR to choose another directory.", file=sys.stderr)
    return assessment


def _run_cli_admitted(args) -> int:
    if args.log_level is not None:
        from utils.logging_setup import session_log_path, set_log_level
        set_log_level(args.log_level)
        print(f"Log file: {session_log_path()}", file=sys.stderr)
    if args.profile_performance is None:
        return _run_cli_parsed(args)
    return _run_cli_profiled(args)


def _cli_performance_report(args, profile, outcome):
    """Finalize the CLI report once; --json and FILE share the same object."""
    report = getattr(args, '_performance_report', None)
    if report is None:
        report = profile.report(outcome)
        report['finished_at'] = performance_profile.finished_now()
        report['runtime'] = performance_profile.runtime_metadata()
        from core.resource_policy import cpu_facts
        report['runtime'].update(cpu_facts(getattr(args, 'resource_preference', 'balanced')))
        args._performance_report = report
    return report


def _run_cli_profiled(args) -> int:
    """Run with an active profile; a profiling failure never changes the exit code."""
    profile = performance_profile.PerformanceProfile()
    args._performance_profile = profile
    profile.unmeasured('source decoding runs lazily inside registration/fusion for streaming '
                       'methods; see concurrent.source_decode (summed across threads)')
    profile.unmeasured('peak live scratch bytes')
    outcome = 'failed'
    try:
        with performance_profile.active(profile):
            result = _run_cli_parsed(args)
        outcome = 'completed' if result == EXIT_OK else 'failed'
    except KeyboardInterrupt:
        outcome = 'cancelled'
        raise
    finally:
        if getattr(args, '_performance_started', False):
            try:
                report = _cli_performance_report(args, profile, outcome)
                if outcome == 'cancelled':
                    report['outcome'] = outcome
                log_action('render.performance', **performance_profile.compact(report))
                print(performance_profile.summary(report), file=sys.stderr)
                if isinstance(args.profile_performance, str):
                    # After a long render: a failed write must not cost an earlier report.
                    _replace_text_file(args.profile_performance,
                                       performance_profile.dumps(report) + '\n', ".openfocus_profile_")
            except Exception as exc:
                print(f"warning: --profile-performance: {exc}", file=sys.stderr)
    return result


def _run_cli_parsed(args) -> int:
    if args.native_batch_format and (not args.native_raw or not args.output_dir):
        print("error: --native-batch-format requires --native-raw and --output-dir", file=sys.stderr)
        return EXIT_USAGE
    if args.native_raw:
        if (args.suggest_duplicates or args.group_size or args.dust_map or
                args.defect_mask_path or args.sensor_denoise or args.depth_map):
            print("error: --native-raw cannot be combined with duplicate review, grouped fusion, dust correction, sensor cleanup or depth maps", file=sys.stderr)
            return EXIT_USAGE
        if (args.scale_bar or args.scale_um is not None) and (
                (args.output_dir and (args.native_batch_format or "dng") == "dng") or
                (args.output and os.path.splitext(args.output)[1].lower() == ".dng")):
            print("error: scale bars are not supported in native DNG output", file=sys.stderr)
            return EXIT_USAGE
    if args.profile_performance is not None:
        if args.suggest_duplicates or args.output_dir or args.group_size or args.dry_run:
            print("error: --profile-performance requires a single, non-grouped fusion run", file=sys.stderr)
            return EXIT_USAGE
    if args.alignment_report:
        if args.suggest_duplicates or args.output_dir or args.group_size or args.dry_run:
            print("error: --alignment-report requires a single, non-grouped fusion run", file=sys.stderr)
            return EXIT_USAGE
        if args.align == "none":
            print("error: --alignment-report requires an alignment method", file=sys.stderr)
            return EXIT_USAGE
        try:
            _validate_alignment_report_path(args)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
    if args.suggest_duplicates:
        if args.defect_mask_path or args.sensor_denoise:
            print("error: sensor cleanup is unavailable with --suggest-duplicates", file=sys.stderr)
            return EXIT_USAGE
        return _suggest_duplicates(args)

    if args.threads < 0:
        print("error: --threads must be >= 0 (0 = automatic)", file=sys.stderr)
        return EXIT_USAGE
    if args.downscale is not None and args.downscale < 1:
        print("error: --downscale must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    if args.batch_size < 1:
        print("error: --batch-size must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    if args.tile_size is not None and args.tile_size < 64:
        print("error: --tile-size must be >= 64", file=sys.stderr)
        return EXIT_USAGE
    if args.group_size < 0 or args.group_size == 1:
        print("error: --group-size must be 0 or at least 2", file=sys.stderr)
        return EXIT_USAGE
    if args.group_overlap < 0 or (args.group_size and args.group_overlap > args.group_size - 2):
        print("error: --group-overlap must be between 0 and group size minus 2", file=sys.stderr)
        return EXIT_USAGE
    if args.group_size:
        if args.depth_map:
            print("error: --depth-map is unavailable with grouped fusion: group positions are not original frame indices", file=sys.stderr)
            return EXIT_USAGE
        if any(not os.path.isdir(path) for path in args.input):
            print("error: grouped fusion accepts folders only", file=sys.stderr)
            return EXIT_USAGE
    if args.tile_overlap is not None and (args.tile_overlap < 0 or
            args.tile_size is not None and args.tile_overlap >= args.tile_size):
        print("error: --tile-overlap must be nonnegative and smaller than --tile-size", file=sys.stderr)
        return EXIT_USAGE
    if args.tile_threshold is not None and args.tile_threshold < 1:
        print("error: --tile-threshold must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    if args.depth_map:
        # 提前校验：否则拼写错误要到融合全部跑完后才在 colorize 里炸掉
        try:
            from utils.colormap import get_lut
            get_lut(args.depth_colormap)
        except ValueError as exc:
            print(f"error: --depth-colormap: {exc}", file=sys.stderr)
            return EXIT_USAGE
    background_extras = [flag for flag, used in (
        ("--background-mode", args.background_mode != "clean"),
        ("--keep-soft-parts", args.keep_soft_parts),
        ("--save-stacked", args.save_stacked),
        ("--missing-data-map", args.missing_data_map)) if used]
    if background_extras and not args.auto_background:
        print(f"error: {', '.join(background_extras)} requires --auto-background", file=sys.stderr)
        return EXIT_USAGE
    if args.auto_background:
        if args.group_size:
            print("error: --auto-background is unavailable with grouped fusion: it needs the "
                  "aligned frames of one stack", file=sys.stderr)
            return EXIT_USAGE
        if args.native_raw:
            print("error: --auto-background is unavailable with --native-raw: it works on "
                  "rendered colour, not camera-linear samples", file=sys.stderr)
            return EXIT_USAGE
        if args.output_dir and any(isinstance(value, str) for value in
                                   (args.save_stacked, args.missing_data_map)):
            print("error: --save-stacked / --missing-data-map with an explicit path cannot be "
                  "used with --output-dir (every stack would overwrite the same file); use them "
                  "without a value to derive one per stack", file=sys.stderr)
            return EXIT_USAGE
        if isinstance(args.missing_data_map, str) and \
                not args.missing_data_map.lower().endswith(".png"):
            print("error: --missing-data-map must be a .png path", file=sys.stderr)
            return EXIT_USAGE
        if isinstance(args.save_stacked, str) and \
                os.path.splitext(args.save_stacked)[1].lower() not in VALID_FORMATS:
            print(f"error: --save-stacked: unsupported extension; expected one of "
                  f"{', '.join(VALID_FORMATS)}", file=sys.stderr)
            return EXIT_USAGE

    if args.output and not args.output_dir:
        # One single-stack run writes the result, its depth map, the background
        # extras (one mask per missing-data class included), the JSON summary
        # and the alignment report.  Two of those may not share a path: the
        # later write would silently replace the earlier one.
        duplicate = _first_duplicate(_run_outputs(args, args.output))
        if duplicate is not None:
            path, flag, earlier = duplicate
            print(f"error: {path} would be written twice ({earlier} and {flag}); "
                  f"choose different paths", file=sys.stderr)
            return EXIT_USAGE

    # Per-folder overrides (--config) are batch-only; load once here so a bad
    # file is a usage error instead of a mid-run failure.
    args._folder_overrides = {}
    if args.config:
        if not args.output_dir:
            print("error: --config requires batch mode (--output-dir)", file=sys.stderr)
            return EXIT_USAGE
        try:
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError(f"duplicate field: {key}")
                    result[key] = value
                return result

            def invalid_number(value):
                raise ValueError(f"invalid number: {value}")

            with open(args.config, encoding="utf-8") as f:
                args._folder_overrides = json.load(f, object_pairs_hook=pairs,
                                                   parse_constant=invalid_number)
            if not isinstance(args._folder_overrides, dict):
                raise ValueError("top level must be a JSON object")
        except Exception as exc:
            print(f"error: --config: {exc}", file=sys.stderr)
            return EXIT_USAGE

    try:
        _validate_execution_args(args)
        _resolve_algorithm_arguments(args)
        for folder, override in args._folder_overrides.items():
            if not isinstance(override, dict):
                raise ValueError(f"Overrides for {folder} must be an object")
            unknown = set(override) - {
                "method", "align", "kernel",
                "ai_mode", "base_frame", "sort", "exposure_match", "sharpen_output", "cpu", "threads",
                "batch_size", "downscale", "tile_size", "align_shift_limit",
                "resource_preference", "fusion_parameters", "registration_parameters",
                "auto_background", "background_mode", "keep_soft_parts"}
            if unknown:
                raise ValueError(f"Unknown override keys for {folder}: {sorted(unknown)}")
            try:
                merged = argparse.Namespace(**{**vars(args), **override})
                _validate_execution_args(merged)
                _resolve_algorithm_arguments(merged)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"config {folder}: {exc}") from exc
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    summary_write_allowed = False
    summary_plan = None

    def _validate_summary_plan(planned, candidates, keys, identities):
        # Unlike file_identity, stat must not hide access errors here.
        current_keys, current_identities = _admitted_input_keys(candidates)
        for _flag, path in planned:
            if path:
                try:
                    os.stat(path)
                except FileNotFoundError:
                    pass
        duplicate = _first_duplicate(planned)
        if duplicate:
            path, flag, earlier = duplicate
            raise ValueError(f"{path} would be written twice ({earlier} and {flag})")
        overlap = _first_input_overlap(planned, keys | current_keys,
                                       identities | current_identities)
        if overlap:
            flag, path = overlap
            raise ValueError(f"{flag} would overwrite an input image: {path}")

    def _allow_summary(planned, candidates, keys, identities):
        nonlocal summary_write_allowed, summary_plan
        if not args.json:
            return
        _validate_summary_plan(planned, candidates, keys, identities)
        summary_plan = (planned, candidates, keys, identities)
        summary_write_allowed = True

    def _emit_json(payload: dict) -> None:
        nonlocal summary_write_allowed
        if not args.json or not summary_write_allowed:
            return
        import json
        profile = getattr(args, '_performance_profile', None)
        if profile is not None and getattr(args, '_performance_started', False):
            try:
                payload = {**payload, "performance": _cli_performance_report(
                    args, profile, 'failed' if 'error' in payload else 'completed')}
            except Exception as exc:
                print(f"warning: --profile-performance: {exc}", file=sys.stderr)
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2)

            def before_replace():
                nonlocal summary_write_allowed
                try:
                    cancelled = getattr(args, '_should_cancel', None)
                    if cancelled is not None and cancelled():
                        raise RuntimeError("summary write cancelled")
                    _validate_summary_plan(*summary_plan)
                except BaseException:
                    summary_write_allowed = False
                    raise

            _replace_text_file(args.json, text, ".openfocus_summary_", before_replace)
        except Exception as exc:
            print(f"warning: --json: {exc}", file=sys.stderr)

    def _progress(**fields) -> None:
        if args.json_progress:
            _machine_event(args._machine_state, **fields)

    args._progress = _progress

    batch_mode = args.output_dir is not None
    if batch_mode and isinstance(args.depth_map, str) and args.depth_map:
        print("error: --depth-map with an explicit path cannot be used with "
              "--output-dir (every stack would overwrite the same file); "
              "use --depth-map without a value to derive one per stack",
              file=sys.stderr)
        return EXIT_USAGE
    if args.json_progress:
        _progress(event="start", version=app_display_version(),
                  mode="batch" if batch_mode else "single",
                  method=args.method, align=args.align)
    if batch_mode:
        args._batch_ext = ("." + (args.native_batch_format or "dng")) if args.native_raw else ".png"
    else:
        if args.resume:
            print("error: --resume requires batch mode (--output-dir)", file=sys.stderr)
            return EXIT_USAGE
        if not args.output:
            print("error: one of --output / --output-dir is required", file=sys.stderr)
            return EXIT_USAGE
        # A single output means "one stack". Several folders used to be
        # silently concatenated into one unrelated stack; batch users must
        # opt into --output-dir.
        dir_inputs = [p for p in args.input if os.path.isdir(os.path.abspath(p))]
        if len(dir_inputs) > 1:
            print("error: multiple input folders require --output-dir "
                  "(--output fuses every input into a single stack)",
                  file=sys.stderr)
            return EXIT_USAGE
        ext = os.path.splitext(args.output)[1].lower()
        formats = [".dng", ".tif", ".tiff", ".png"] if args.native_raw else VALID_FORMATS
        if not ext or ext not in formats:
            print(f"error: unsupported output extension '{ext or '(none)'}'; "
                  f"expected one of {', '.join(formats)}", file=sys.stderr)
            return EXIT_USAGE

    if args.native_raw:
        # Check all batch destinations against all inputs before writing any result.
        from core.image_loader import ImageStackLoader
        from utils.path_identity import same_path
        loader = ImageStackLoader()
        try:
            all_paths = []
            for entry in args.input:
                if os.path.isdir(entry):
                    paths = _list_folder_images(entry, loader)
                elif os.path.isfile(entry):
                    paths = [entry]
                else:
                    raise FileNotFoundError(f"Input not found: {entry}")
                all_paths.extend(paths)
                if batch_mode:
                    _validate_native_paths(paths)
            _validate_native_paths(all_paths)
            destinations = [args.json, args.alignment_report, _profile_destination(args)]
            if batch_mode:
                destinations.extend(os.path.join(args.output_dir,
                    (os.path.basename(os.path.abspath(folder)) or "stack") + args._batch_ext)
                    for folder in args.input if os.path.isdir(folder))
            else:
                destinations.append(args.output)
            for destination in destinations:
                if destination and any(same_path(destination, path) for path in all_paths):
                    raise ValueError(f"Output would overwrite an input image: {destination}")
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE

    if args.defect_mask_path:
        destinations = [args.output, args.json]
        if args.output and args.depth_map:
            destinations.append(_depth_out_path(args, args.output))
        if args.output and args.auto_background:
            destinations += _background_out_paths(args, args.output)
        if any(folder_args_mask_conflict(args, path) for path in destinations if path):
            print("error: an output would overwrite the defect mask", file=sys.stderr)
            return EXIT_USAGE
    try:
        import cv2

        if batch_mode:
            return _run_batch(args, _emit_json, _allow_summary)

        if args.json:
            from core.image_loader import ImageStackLoader
            candidates, (input_keys, input_files) = _single_inputs(args, ImageStackLoader())
            planned = _run_outputs(args, args.output)
            try:
                _allow_summary(planned, candidates, input_keys, input_files)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return EXIT_USAGE

        from core.multi_focus_fusion import MultiFocusFusion
        from core.frame_storage import storage_directory

        # Video inputs are unpacked to a temp dir that is always cleaned up
        with _video_directory(args, storage_directory()) as video_temp:
            _src, paths = _resolve_stack(args, video_temp)
            # No output may replace one of the frames it is made from (or the
            # dust/defect inputs); checked before load, fusion and any write.
            overlap = _output_over_inputs(args, args.output, paths)
            if overlap is not None:
                flag, path = overlap
                print(f"error: {flag} would overwrite an input image: {path}",
                      file=sys.stderr)
                return EXIT_USAGE
            if args.alignment_report:
                _validate_alignment_report_path(args, paths)
            _say(f"Loaded {len(paths)} image(s) from {_src}")

            if args.group_size and args.dry_run:
                _inspect_grouped_cli(paths, args)
            if args.dry_run:
                _say(f"[dry-run] method={args.method} align={args.align} "
                      f"frames={len(paths)} output={args.output}")
                for p in paths:
                    _say(f"[dry-run]   {p}")
                _emit_json({"mode": "single", "dry_run": True,
                            "method": args.method, "align": args.align,
                            "frames": len(paths), "input": args.input,
                            "output": args.output})
                return EXIT_OK

            if not args.native_raw and not args.group_size:
                _warn_low_storage(args, paths)
            _progress(event="load", done=1, total=1, frames=len(paths))
            args._performance_started = True
            if args.native_raw:
                images, native_metadata = _load_native_stack(paths, args)
            elif not args.group_size:
                with performance_profile.current().stage('load', resources=True):
                    images = _load_images(
                        paths, thread_count=getattr(args, "threads", 0) or 0,
                        resource_preference=getattr(args, "resource_preference", "balanced"),
                        registration_steps=int(args.align != 'none'),
                        exposure_match=getattr(args, 'exposure_match', False),
                        lazy=get_fusion(args.method).streaming_sources)
            if paths:
                from utils.image_utils import set_source_exif
                from core.image_loader import _read_exif_bytes
                set_source_exif(_read_exif_bytes(paths[0]))

            if args.alignment_report:
                from core.registration import alignment_report
                entries = []
                token = alignment_report.set(entries)
                try:
                    fused = _fuse_stack(images, args)
                finally:
                    alignment_report.reset(token)
            else:
                fused = _fuse_grouped(paths, args) if args.group_size else _fuse_stack(images, args)
            if fused is None:
                raise RuntimeError("Fusion returned no result")
            if args.alignment_report and not entries:
                raise RuntimeError("The selected alignment method did not provide diagnostics")
            scale_cfg = _resolve_scale_cfg(args, paths)
            source_icc_profile = b""
            if not args.native_raw:
                from core.image_loader import common_icc_profile
                source_icc_profile = common_icc_profile(paths)
            frame_count = len(images) if not args.group_size else len(paths)
            background = stacked = None
            if args.auto_background:
                # The decoded originals are no longer needed: the aligned
                # frames are what the background is rebuilt from.
                images = None
                stacked = fused
                background = _auto_background(args, fused)
                fused = background.image

        with performance_profile.current().stage('save'):
            if args.native_raw:
                _save_native(args.output, fused, native_metadata, scale_cfg)
            else:
                _save(args.output, fused, scale_cfg, source_icc_profile)
        _say(f"Saved: {args.output}")
        _progress(event="save", output=args.output)
        outputs = [args.output]
        if background is not None:
            outputs += _save_background_extras(args, args.output, stacked, background,
                                               scale_cfg, source_icc_profile)
            stacked = None
        if args.depth_map:
            _save_depth(args, args.output)
            outputs.append(_depth_out_path(args, args.output))
        if args.alignment_report:
            if getattr(args, '_alignment_reversed', False):
                for entry in entries:
                    for key in ('frame_index', 'reference_index'):
                        if entry.get(key) is not None:
                            entry[key] = frame_count - 1 - entry[key]
            _write_alignment_report(args.alignment_report, {
                "method": args.align,
                "input": args.input,
                "frame_count": frame_count,
                "entries": entries,
            })
        summary = {"mode": "single", "method": args.method, "align": args.align,
                   "frames": len(paths), "input": args.input,
                   "output": args.output, "outputs": outputs}
        if background is not None:
            summary["auto_background"] = {"preset": args.auto_background,
                                          "mode": args.background_mode,
                                          **_background_summary(background)}
        _emit_json(summary)
        args._machine_state['outputs'] = outputs
        return EXIT_OK
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        _log.exception("CLI processing failed")
        _emit_json({"mode": "batch" if batch_mode else "single", "error": str(exc)})
        if args.json_progress:
            _progress(event="error", code="processing_error", message=str(exc))
        return EXIT_ERROR



# ---------------------------------------------------------------------------
# Agent-facing subcommands: version / describe / gui / ctl
# ---------------------------------------------------------------------------
def describe_payload() -> dict:
    """Machine-readable capabilities for agents, with no optional imports."""
    from importlib.util import find_spec
    from core.control.protocol import COMMANDS, SCHEMA as CONTROL_SCHEMA

    methods = []
    for spec in fusion_algorithms():
        methods.append({
            "id": spec.id,
            "label": spec.display_name,
            "category": spec.category,
            "neural": spec.neural,
            "order_sensitive": spec.order_sensitive,
            "depth_map": spec.depth is not None,
            "weights": spec.weights,
            "parameters": [
                {"key": parameter.key, "default": parameter.default,
                 "minimum": parameter.minimum, "maximum": parameter.maximum,
                 "step": parameter.step, "decimals": parameter.decimals,
                 "choices": list(parameter.choices)}
                for parameter in spec.parameters
            ],
        })
    capabilities = {
        "torch": find_spec("torch") is not None,
        "heic": find_spec("pillow_heif") is not None,
        "raw": find_spec("rawpy") is not None,
        "dtcwt": find_spec("dtcwt") is not None,
        "pyqt": find_spec("PyQt6") is not None,
    }
    try:
        from utils.platform_utils import has_compute_gpu
        capabilities["gpu"] = bool(has_compute_gpu())
    except Exception:
        capabilities["gpu"] = None
    return {
        "schema": "openfocus/describe/1",
        "version": app_display_version(),
        "capabilities": capabilities,
        "methods": methods,
        "registrations": [spec.id for spec in registration_algorithms()],
        "control": {"schema": CONTROL_SCHEMA, "commands": list(COMMANDS)},
        "cli": {
            "batch": ["--output-dir", "--resume", "--config"],
            "machine": ["--json", "--json-progress", "--alignment-report", "--dry-run"],
            "review": ["--suggest-duplicates"],
            "grouped": ["--group-size", "--group-overlap"],
            "cleanup": ["--dust-map", "--defect-mask-path", "--sensor-denoise"],
            "export": ["--scale-bar", "--scale-um", "--depth-map", "--depth-colormap"],
            "background": ["--auto-background", "--background-mode", "--keep-soft-parts",
                           "--save-stacked", "--missing-data-map"],
            "order": ["--sort"],
            "subcommands": sorted(SUBCOMMANDS),
            "options": sorted(option for action in build_parser()._actions
                              for option in action.option_strings if option.startswith("--")),
            "control": ["gui", "ctl <command>", "describe", "version"],
        },
    }


def _spawn_detached(command: list, cwd: str) -> subprocess.Popen:
    import subprocess as _subprocess
    creation = 0
    if sys.platform == "win32":
        creation = 0x00000008 | 0x00000200
    return _subprocess.Popen(
        command, cwd=cwd, stdin=_subprocess.DEVNULL,
        stdout=_subprocess.DEVNULL, stderr=_subprocess.DEVNULL,
        creationflags=creation, start_new_session=sys.platform != "win32")


def _run_gui(argv: List[str]) -> int:
    """Launch the desktop app; its control channel starts automatically."""
    control = "--control" in argv
    socket_name = None
    paths = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--control":
            control = True
            index += 1
        elif token in ("--socket", "--control-socket"):
            if index + 1 >= len(argv):
                print("error: --socket requires a name", file=sys.stderr)
                return EXIT_USAGE
            socket_name = argv[index + 1]
            index += 2
        elif token.startswith("-"):
            print(f"error: unknown gui option: {token}", file=sys.stderr)
            return EXIT_USAGE
        else:
            paths.append(token)
            index += 1
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if control or socket_name:
        from core.control.client import launch_gui
        from core.control.protocol import discovery_path
        try:
            payload = launch_gui(socket_name, paths=paths)
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        print(json.dumps({
            "ready": True, "socket": payload["socket"], "pid": payload["pid"],
            "version": payload["version"], "discovery": str(discovery_path()),
        }, ensure_ascii=False))
        return EXIT_OK
    if getattr(sys, "frozen", False):
        command = [sys.executable, *paths]
    else:
        entry = os.path.join(root, "main.py")
        if not os.path.isfile(entry):
            print("error: this install does not ship the desktop app "
                  "(no main.py); use the desktop build or a source checkout",
                  file=sys.stderr)
            return EXIT_ERROR
        command = [sys.executable, entry, *paths]
    process = _spawn_detached(command, root)
    print(json.dumps({"launched": True, "pid": process.pid}, ensure_ascii=False))
    return EXIT_OK


_CTL_BOOL = {"append", "preview", "include", "cancel", "submit",
             "async", "force", "confirm"}
_CTL_INT = {"index", "depth", "steps", "x", "y", "to_x", "to_y", "delta",
            "roi_base_index"}
_CTL_FLOAT = {"scale"}
_CTL_JSON = {"value"}


def _coerce_json(raw: str):
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _extract_ctl_flags(argv: List[str]):
    flags = {"pretty": False, "launch": False, "wait": False, "timeout": 60.0}
    tokens = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--pretty":
            flags["pretty"] = True
        elif token == "--launch":
            flags["launch"] = True
        elif token == "--wait":
            flags["wait"] = True
        elif token == "--json":
            pass
        elif token == "--timeout":
            if index + 1 >= len(argv):
                raise ValueError("--timeout requires seconds")
            flags["timeout"] = float(argv[index + 1])
            index += 1
        else:
            tokens.append(token)
        index += 1
    return flags, tokens


def _parse_ctl_args(tokens: List[str]) -> dict:
    args = {}
    files = []
    parameters = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--args-json":
            if index + 1 >= len(tokens):
                raise ValueError("--args-json requires a JSON object")
            payload = json.loads(tokens[index + 1])
            if not isinstance(payload, dict):
                raise ValueError("--args-json must be a JSON object")
            args.update(payload)
            index += 2
            continue
        if token == "--param":
            if index + 1 >= len(tokens) or "=" not in tokens[index + 1]:
                raise ValueError("--param expects KEY=VALUE")
            key, _, raw = tokens[index + 1].partition("=")
            parameters[key] = _coerce_json(raw)
            index += 2
            continue
        if not token.startswith("--"):
            raise ValueError(f"Unexpected argument: {token}")
        name = token[2:].replace("-", "_")
        if name in _CTL_BOOL:
            args[name] = True
            index += 1
            continue
        if index + 1 >= len(tokens):
            raise ValueError(f"{token} requires a value")
        raw = tokens[index + 1]
        if name in _CTL_INT:
            args[name] = int(raw)
        elif name in _CTL_FLOAT:
            args[name] = float(raw)
        elif name in _CTL_JSON:
            args[name] = _coerce_json(raw)
        elif name == "file":
            files.append(raw)
        else:
            args[name] = raw
        index += 2
    if files:
        args["files"] = files
    if parameters:
        args["parameters"] = parameters
    return args


def _run_ctl(argv: List[str]) -> int:
    """Send one command to the running GUI and print its JSON response."""
    from core.control.client import launch_gui, request, wait_for
    from core.control.protocol import ControlError, error_response

    try:
        flags, tokens = _extract_ctl_flags(argv)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not tokens:
        print("usage: openfocus ctl <command> [--key value ...] [--pretty] "
              "[--launch] [--wait] [--timeout N]", file=sys.stderr)
        return EXIT_USAGE
    command, rest = tokens[0], tokens[1:]
    try:
        if command == "raw":
            if len(rest) != 1:
                raise ValueError("raw expects one JSON request")
            payload = json.loads(rest[0])
            if not isinstance(payload, dict) or not isinstance(payload.get("cmd"), str):
                raise ValueError("raw JSON needs a cmd string")
            command = payload["cmd"]
            args = payload.get("args") or {}
            if not isinstance(args, dict):
                raise ValueError("raw args must be an object")
        else:
            args = _parse_ctl_args(rest)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    def send():
        return request(command, args, timeout=flags["timeout"])

    try:
        response = send()
    except ControlError as error:
        if flags["launch"] and error.code == "not_running":
            try:
                launch_gui(timeout=45)
                response = send()
            except ControlError as launch_error:
                response = error_response("0", launch_error)
        else:
            response = error_response("0", error)

    if flags["wait"] and response.get("ok"):
        result = response.get("result") or {}
        try:
            if command == "render.start":
                render_id = result.get("render_id")
                if not result.get("started") or render_id is None:
                    raise ControlError("internal", "Render did not start with a render_id")

                def finished(value):
                    if value.get("render_id") != render_id:
                        # Superseded by a newer render (possibly from the
                        # GUI): this request will never publish again.
                        return True
                    if value.get("busy"):
                        return False
                    return value.get("state") in (
                        "done", "error", "cancelled", "discarded", "idle")

                state = wait_for("render.state", finished,
                                 timeout=flags["timeout"])
                superseded = state.get("render_id") != render_id
                ok = not superseded and state.get("state") == "done"
                response = {"id": response.get("id", "0"), "ok": ok,
                            "result": state,
                            "elapsed_ms": response.get("elapsed_ms", 0)}
                if not ok:
                    response["error"] = {
                        "code": "internal",
                        "message": ("Render was superseded by a newer render"
                                    if superseded else state.get("error")
                                    or "The render did not finish (cancelled?)")}
            elif command in ("stack.load", "project.open"):
                load_id = result.get("load_id")
                if load_id is None:
                    raise ControlError("internal", "Load did not return a load_id")

                def loaded(value):
                    load = value.get("load") or {}
                    if load.get("id") != load_id:
                        return False  # frames from an earlier stack
                    return bool(load.get("finished"))

                frames = wait_for("stack.frames", loaded,
                                  timeout=flags["timeout"])
                error = (frames.get("load") or {}).get("error")
                ok = error is None and frames.get("count", 0) > 0
                response = {"id": response.get("id", "0"), "ok": ok,
                            "result": frames,
                            "elapsed_ms": response.get("elapsed_ms", 0)}
                if not ok:
                    response["error"] = {
                        "code": "internal",
                        "message": error or "The stack did not load"}
        except ControlError as error:
            response = error_response("0", error)

    print(json.dumps(response, ensure_ascii=False,
                     indent=2 if flags["pretty"] else None))
    return EXIT_OK if response.get("ok") else EXIT_ERROR



def main() -> None:
    """Console-script entry point (pip-installed ``openfocus`` command)."""
    from utils.diagnostics import install_exception_hooks
    setup_logging()
    log_session_start()
    install_exception_hooks(claim_inherited_faulthandler=True)
    sys.exit(run_cli(sys.argv[1:]))
