"""Headless command-line interface for OpenFocus fusion.

Lets batch users and CI pipelines run registration + fusion without the GUI:

    python main.py --input ./stack_folder --output ./out --method stackmffv3
    python main.py --input ./stackA ./stackB --output-dir ./results   (batch)

Exit codes: 0 success, 1 processing error, 2 usage error.
"""
import argparse
import json
import math
import os
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

from utils.sorting import natural_sort_key
from core.resource_policy import heavy_job, require_memory

from core.algorithms import (fusion_algorithms, registration_algorithms, get_fusion,
                             get_registration, resolve_fusion_parameters,
                             resolve_registration_parameters)
from utils.logging_setup import (setup_logging, log_session_start, log_operation,
                                 log_action, get_logger, frame_details)

_log = get_logger(__name__)

VALID_METHODS = [spec.id for spec in fusion_algorithms()]
VALID_ALIGN = [spec.id for spec in registration_algorithms()]
VALID_FORMATS = [".png", ".jpg", ".webp", ".bmp", ".tif", ".tiff"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="OpenFocus",
        description="Multi-focus image fusion (headless mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py --input ./stack --output ./result/fused.png\n"
            "  python main.py --input ./stack --output ./result/fused.png --method dtcwt --align ecc --threads 8\n"
            "  python main.py --input ./stackA ./stackB --output-dir ./results -m stackmffv3\n"
            "  python main.py --input img1.jpg img2.jpg img3.jpg --output fused.png --method gfgfgf\n"
        ),
    )
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="Image folder(s), video file(s), or a list of image files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file for a single stack (default: fused.png)")
    parser.add_argument("--output-dir", "-d", default=None,
                        help="Batch mode: fuse every input folder, writing <folder>.<ext> here")
    parser.add_argument("--method", "-m", default="stackmffv3",
                        choices=[spec.id for spec in fusion_algorithms()],
                        help="Fusion method (default: stackmffv3); openfocus_a/b/c select the "
                             "OpenFocus family (A = variance-weighted average, B = variance depth-map pick, "
                             "C = Laplacian pyramid)")
    parser.add_argument("--align", "-a", default="none",
                        choices=[spec.id for spec in registration_algorithms()],
                        help="Registration applied before fusion (default: none); openfocus = fast bounded "
                             "similarity alignment (rotation + scale + NCC translation) with brightness "
                             "equalization")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=JSON_VALUE",
                        help="Fusion parameter; repeat for multiple keys. Overrides legacy flags.")
    parser.add_argument("--align-param", action="append", default=[], metavar="KEY=JSON_VALUE",
                        help="Registration parameter; repeat for multiple keys.")
    parser.add_argument("--align-shift-limit", type=float, default=None, metavar="RATIO",
                        help="openfocus alignment only: maximum accepted frame shift as a fraction of the "
                             "frame size (default 0.08; frames above it are "
                             "left unaligned rather than half-corrected)")
    parser.add_argument("--base-frame", default="none", choices=["none", "last", "first"],
                        help="Optional post-fusion base correction: match the output's low "
                             "frequencies to the last/first frame (default: none).")
    parser.add_argument("--exposure-match", dest="exposure_match",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Match every frame's exposure to the last frame before fusion "
                             "(default: on; --no-exposure-match disables it). Computed on the "
                             "whole stack before tiling, for every fusion method.")
    parser.add_argument("--sort", default="asc", choices=["asc", "desc", "off"],
                        help="Stack order for the openfocus_a/b/c methods: 'asc' keeps the file order "
                             "(default), 'desc' reverses a far-to-near sweep so the "
                             "base/last-frame semantics anchor to the intended end, 'off' is an explicit no-op")
    parser.add_argument("--dust-map", default=None, metavar="WHITE",
                        help="Flat-field (dust map) correction: path to a defocused shot of a blank/white "
                             "area; removes static dust specks and vignetting from every frame before "
                             "registration")
    parser.add_argument("--kernel", "-k", type=int, default=None,
                        help="Kernel size (forced odd); for the openfocus family it is the "
                             "source of the variance-window radius")
    parser.add_argument("--depth-smoothing", type=float, default=None, metavar="SIGMA",
                        help="openfocus_b only: depth-map transition smoothing (Gaussian sigma, "
                             "default 1.2; 0 = sharpest, larger = calmer on noisy stacks)")
    parser.add_argument("--blend-smoothing", type=float, default=None, metavar="SIGMA",
                        help="openfocus_a / openfocus_c only: blend smoothing (Gaussian sigma "
                             "of the per-frame weight map; defaults 2.0 for A and 3.0 for C; "
                             "0 = crispest per-pixel choice, larger = softer transitions)")
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
                        help="Force CPU even if CUDA/MPS is available")
    parser.add_argument("--downscale", type=int, default=None,
                        help="Registration feature-detection downscale width")
    parser.add_argument("--tile-size", type=int, default=None,
                        help="Tile block size for large images (default: auto)")
    parser.add_argument("--scale-bar", action="store_true",
                        help="Draw a scale bar on the output (calibration auto-detected from TIFF metadata, or --scale-um)")
    parser.add_argument("--scale-um", type=float, default=None,
                        help="Pixel size in um/px, overriding metadata (implies --scale-bar)")
    parser.add_argument("--depth-map", "-D", dest="depth_map", nargs="?", const=True, default=None,
                        help="Also save a colorized focus-position (pseudo-depth) map computed with the same method "
                             "(optional value: output path; default <output>_depth.png)")
    parser.add_argument("--depth-colormap", default="turbo",
                        help="Colormap for --depth-map: turbo, viridis, jet, gray, ... (default turbo)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print the plan without fusing")
    parser.add_argument("--resume", action="store_true",
                        help="Batch mode: skip folders whose output file already exists")
    parser.add_argument("--json", default=None, metavar="FILE",
                        help="Write a machine-readable result summary to FILE when done")
    parser.add_argument("--json-progress", action="store_true",
                        help="Print one JSON progress object per step to stdout (NDJSON)")
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


def _resolve_algorithm_arguments(args):
    """Translate old flags once; generic dictionaries/assignments win per field."""
    spec = get_fusion(args.method)
    get_registration(args.align)
    declared = {parameter.key for parameter in spec.parameters}
    values = {}
    if getattr(args, "kernel", None) is not None and "kernel_size" in declared:
        default = next(parameter.default for parameter in spec.parameters if parameter.key == "kernel_size")
        values["kernel_size"] = _normalize_kernel(args.kernel, default)
    for old, key in (("depth_smoothing", "depth_smoothing"),
                     ("blend_smoothing", "blend_smoothing"), ("ai_mode", "mode")):
        value = getattr(args, old, None)
        if value is not None and key in declared:
            if key.endswith("smoothing") and not math.isfinite(value):
                raise ValueError(f"{old} must be finite")
            values[key] = max(0.0, value) if key.endswith("smoothing") else value
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
            paths.extend(_list_folder_images(entry, loader))
        elif loader.is_video_file(entry):
            ok, message, frames, _names = loader.load_from_video(entry)
            if not ok:
                raise RuntimeError(f"Failed to read video {entry}: {message}")
            import cv2
            for i, frame in enumerate(frames):
                frame_path = os.path.join(video_temp_root, f"video_{len(paths) + i:05d}.png")
                from utils.image_utils import imwrite_auto
                imwrite_auto(frame_path, frame)
                paths.append(frame_path)
        elif os.path.isfile(entry):
            paths.append(entry)
        else:
            raise FileNotFoundError(f"Input not found: {entry}")

    # A single multi-page TIFF is a whole stack: count its pages, not files
    from core.image_loader import count_frames
    total_frames = sum(count_frames(p) for p in paths)
    if total_frames < 2:
        raise RuntimeError("Need at least 2 images in the stack for fusion")
    return ", ".join(args.input), paths


def _load_images(paths: List[str], thread_count: int = 0,
                 resource_preference: str = "balanced",
                 should_cancel=None):
    from core.image_loader import iter_decoded_sources, _DecodeCancelled
    images = []
    try:
        decoded = iter_decoded_sources(
            paths, 1.0, thread_count=thread_count,
            resource_preference=resource_preference,
            should_cancel=should_cancel)
        try:
            for item in decoded:
                if item.error is not None:
                    # The CLI is strict: one unreadable frame is an error,
                    # never a silently shorter stack.
                    raise RuntimeError(f"Failed to decode image: {item.path}")
                images.extend(item.pages)
                del item
        finally:
            decoded.close()
    except _DecodeCancelled:
        raise RuntimeError("Loading cancelled")
    return images


def _normalize_kernel(kernel: Optional[int], default: int) -> int:
    size = int(kernel) if kernel else default
    if size % 2 == 0:
        size = max(1, size - 1)
    return max(1, size)


def _fuse_stack(images, args):
    with heavy_job():
        return _fuse_stack_admitted(images, args)


def _fuse_stack_admitted(images, args):
    """Register and fuse one stack; returns the fused image."""
    import cv2
    from core.registration import ImageRegistration
    from core.multi_focus_fusion import MultiFocusFusion
    from utils.image_utils import normalize_fuse_input, quantize_fuse_output

    _resolve_algorithm_arguments(args)
    # Flat-field (dust map) correction runs FIRST and in the raw pixel
    # domain: dust lives in sensor coordinates, so it must be divided out
    # before registration warps the content.
    dust_white = getattr(args, "dust_map", None)
    if dust_white:
        from utils.dust_map import compute_flat_field, apply_flat_field_stack
        data = np.fromfile(dust_white, dtype=np.uint8)
        white = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if white is None:
            raise RuntimeError(f"Failed to decode white-field image: {dust_white}")
        gain = compute_flat_field(white)
        images = apply_flat_field_stack(images, gain)
        print(f"Flat-field (dust map) correction applied from {dust_white}")

    # Normalize uint16 to the float32 0–255 working domain before registration.
    if any(image.dtype == np.uint16 for image in images):
        require_memory(sum(image.size * 4 for image in images if image.dtype != np.float32)
                       + max(image.size * 4 for image in images if image.dtype == np.uint16))
    images, is_16bit = normalize_fuse_input(images)

    if args.align != "none":
        t0 = time.time()
        reg = ImageRegistration(
            method=args.align, downscale_width=args.downscale,
            parameters=args._registration_parameters,
            brightness_reference=("last" if getattr(args, "exposure_match", True)
                                  else "first"),
            use_gpu=False)
        images = reg.process(images, output_path=None, thread_count=args.threads,
                             resource_preference=getattr(args, "resource_preference", "balanced"))
        print(f"Registration ({args.align}) done in {time.time() - t0:.1f}s")

    spec = get_fusion(args.method)
    fusion_kwargs = dict(algorithm=args.method, use_gpu=not args.cpu,
                         fusion_parameters=args._fusion_parameters,
                         resource_preference=getattr(args, "resource_preference", "balanced"),
                         base_frame=getattr(args, "base_frame", "none"),
                         sort_mode=getattr(args, "sort", "asc"),
                         exposure_match=getattr(args, "exposure_match", True),
                         ai_batch_size=args.batch_size)
    log_action("cli.fusion", method=args.method, alignment=args.align,
               fusion_parameters=args._fusion_parameters,
               registration_parameters=args._registration_parameters,
               threads=args.threads, use_gpu=not args.cpu,
               tile_size=args.tile_size, base_frame=fusion_kwargs['base_frame'],
               exposure_match=fusion_kwargs['exposure_match'], **frame_details(images))
    if args.tile_size:
        fusion_kwargs["tile_block_size"] = args.tile_size
    fusion = MultiFocusFusion(**fusion_kwargs)
    info = fusion.get_info()
    t0 = time.time()
    fuse_kwargs = dict(input_source=images, img_resize=None, thread_count=args.threads)
    model_weights = _method_weights(spec)
    if model_weights:
        fuse_kwargs["model_path"] = model_weights
    fused = fusion.fuse(**fuse_kwargs)
    if fused is None:
        raise RuntimeError("Fusion returned no result")
    fused = quantize_fuse_output(fused, is_16bit)
    # 注册/归一化后的帧留给 --depth-map 用（与融合同一输入）。帧序必须与
    # 融合内部一致：--sort desc 会在 fuse() 内反转帧序，深度图若仍按原序
    # 计算，0/N-1 标签就指向相反的一端。
    depth_frames = list(images)
    if spec.sort_frames:
        from fusion_methods.openfocus_ab import sort_frames_for_openfocus
        depth_frames, _flipped = sort_frames_for_openfocus(
            depth_frames, getattr(args, "sort", "asc"))
    args._depth_frames = depth_frames
    print(f"Fusion ({args.method}, device={info['device']}) done in {time.time() - t0:.1f}s")
    return fused


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
    print(f"Depth map ({args.depth_colormap}) saved: {path} "
          f"[{time.time() - t0:.1f}s]")


def _resolve_scale_cfg(args, paths):
    """Scale-bar config from --scale-bar/--scale-um + TIFF metadata.

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


def _save(path: str, fused, scale_cfg=None) -> None:
    from utils.image_utils import imwrite_auto
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    if scale_cfg:
        from utils import scalebar
        fused = scalebar.burn(fused, scale_cfg)
    if not imwrite_auto(path, fused):
        raise RuntimeError(f"Failed to write output: {path}")


def _run_batch(args, _emit_json=None) -> int:
    """Batch mode: fuse every input folder into --output-dir."""
    import argparse
    from core.image_loader import ImageStackLoader

    loader = ImageStackLoader()
    folders = [os.path.abspath(p) for p in args.input if os.path.isdir(p)]
    skipped = [p for p in args.input if not os.path.isdir(os.path.abspath(p))]
    for p in skipped:
        print(f"warning: skipping non-folder input in batch mode: {p}", file=sys.stderr)
    if not folders:
        print("error: --output-dir requires image folder(s) as --input", file=sys.stderr)
        return EXIT_USAGE
    if args.output:
        print(f"warning: --output is ignored in batch mode (using --output-dir)", file=sys.stderr)

    # Duplicate basenames would silently overwrite each other
    seen, duplicates = set(), False
    for folder in folders:
        name = os.path.basename(folder.rstrip("/\\")) or "stack"
        if name in seen:
            duplicates = True
            print(f"error: duplicate folder name '{name}' would overwrite results; "
                  f"rename the folders or fuse them individually", file=sys.stderr)
        seen.add(name)
    if duplicates:
        return EXIT_USAGE
    os.makedirs(args.output_dir, exist_ok=True)

    ok_count, failures = 0, []
    skipped_existing = []
    started = time.time()
    for i, folder in enumerate(folders):
        name = os.path.basename(folder.rstrip("/\\")) or "stack"
        out_path = os.path.join(args.output_dir, f"{name}{args._batch_ext}")
        print(f"\n=== {name} ===")

        # --resume: a finished folder (output exists) is skipped, so rerunning
        # a failed batch only processes what is still missing.
        if args.resume and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            print(f"skip (output exists): {out_path}")
            skipped_existing.append(name)
            ok_count += 1
            continue

        # Per-folder overrides from --config, keyed by folder name or path.
        folder_args = args
        override = (args._folder_overrides or {}).get(name) \
            or (args._folder_overrides or {}).get(folder)
        if override:
            folder_args = argparse.Namespace(**{**vars(args), **override})
            applied = ", ".join(f"{k}={v}" for k, v in override.items())
            print(f"config override: {applied}")

        if args.dry_run:
            try:
                paths = _list_folder_images(folder, loader)
                from core.image_loader import count_frames
                frames = sum(count_frames(p) for p in paths)
                print(f"[dry-run] frames={frames} method={folder_args.method} "
                      f"align={folder_args.align} output={out_path}")
            except Exception as exc:
                print(f"[dry-run] {name}: {exc}")
            continue

        if args.json_progress:
            print('{"event": "folder", "name": "%s", "index": %d, "total": %d}'
                  % (name, i + 1, len(folders)), flush=True)
        try:
            paths = _list_folder_images(folder, loader)
            # Count frames, not files: a folder holding one multi-page TIFF
            # Z-stack is a complete stack (same rule as single-stack mode).
            from core.image_loader import count_frames
            frame_count = sum(count_frames(p) for p in paths)
            if frame_count < 2:
                raise RuntimeError(
                    f"need at least 2 frames, found {frame_count}")
            images = _load_images(
                paths, thread_count=getattr(args, "threads", 0) or 0,
                resource_preference=getattr(args, "resource_preference",
                                            "balanced"))
            fused = _fuse_stack(images, folder_args)
            _save(out_path, fused, _resolve_scale_cfg(folder_args, paths))
            print(f"Saved: {out_path}")
            if args.depth_map:
                _save_depth(folder_args, out_path)
            ok_count += 1
        except Exception as exc:
            _log.exception("batch folder failed: %s", folder)
            print(f"error: {name}: {exc}", file=sys.stderr)
            failures.append(name)

    summary = {
        "mode": "batch", "dry_run": bool(args.dry_run),
        "method": args.method, "align": args.align,
        "total": len(folders),
        "succeeded": ok_count,
        "skipped_existing": skipped_existing,
        "failed": failures,
        "duration_s": round(time.time() - started, 1),
    }
    print(f"\nBatch finished: {ok_count}/{len(folders)} stacks succeeded")
    if skipped_existing:
        print(f"skipped (already done): {', '.join(skipped_existing)}")
    if failures:
        print("failed: " + ", ".join(failures), file=sys.stderr)
    if _emit_json is not None:
        _emit_json(summary)
    if failures:
        return EXIT_ERROR
    return EXIT_OK


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
    setup_logging()
    log_session_start()
    with log_operation("cli", args=argv) as operation:
        try:
            with heavy_job():
                result = _run_cli_admitted(argv)
            if result == EXIT_ERROR:
                operation.fail()
            log_action("cli.exit", exit_code=result)
            return result
        except KeyboardInterrupt:
            operation.cancel()
            print("interrupted", file=sys.stderr)
            return EXIT_ERROR


def _run_cli_admitted(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)

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
    if args.depth_map:
        # 提前校验：否则拼写错误要到融合全部跑完后才在 colorize 里炸掉
        try:
            from utils.colormap import get_lut
            get_lut(args.depth_colormap)
        except ValueError as exc:
            print(f"error: --depth-colormap: {exc}", file=sys.stderr)
            return EXIT_USAGE

    # Per-folder overrides (--config) are batch-only; load once here so a bad
    # file is a usage error instead of a mid-run failure.
    args._folder_overrides = {}
    if args.config:
        if not args.output_dir:
            print("error: --config requires batch mode (--output-dir)", file=sys.stderr)
            return EXIT_USAGE
        try:
            import json
            with open(args.config, encoding="utf-8") as f:
                args._folder_overrides = json.load(f)
            if not isinstance(args._folder_overrides, dict):
                raise ValueError("top level must be a JSON object")
        except Exception as exc:
            print(f"error: --config: {exc}", file=sys.stderr)
            return EXIT_USAGE

    try:
        _resolve_algorithm_arguments(args)
        for folder, override in args._folder_overrides.items():
            if not isinstance(override, dict):
                raise ValueError(f"Overrides for {folder} must be an object")
            unknown = set(override) - {
                "method", "align", "kernel", "depth_smoothing", "blend_smoothing",
                "ai_mode", "base_frame", "sort", "exposure_match", "cpu", "threads",
                "batch_size", "downscale", "tile_size", "align_shift_limit",
                "resource_preference", "fusion_parameters", "registration_parameters"}
            if unknown:
                raise ValueError(f"Unknown override keys for {folder}: {sorted(unknown)}")
            _resolve_algorithm_arguments(argparse.Namespace(**{**vars(args), **override}))
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    def _emit_json(payload: dict) -> None:
        if not args.json:
            return
        import json
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"warning: --json: {exc}", file=sys.stderr)

    def _progress(**fields) -> None:
        if args.json_progress:
            import json
            print(json.dumps(fields, ensure_ascii=False), flush=True)

    args._progress = _progress

    batch_mode = args.output_dir is not None
    if batch_mode and isinstance(args.depth_map, str) and args.depth_map:
        print("error: --depth-map with an explicit path cannot be used with "
              "--output-dir (every stack would overwrite the same file); "
              "use --depth-map without a value to derive one per stack",
              file=sys.stderr)
        return EXIT_USAGE
    if batch_mode:
        args._batch_ext = ".png"
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
        if not ext or ext not in VALID_FORMATS:
            print(f"error: unsupported output extension '{ext or '(none)'}'; "
                  f"expected one of {', '.join(VALID_FORMATS)}", file=sys.stderr)
            return EXIT_USAGE

    try:
        import tempfile
        import cv2

        if batch_mode:
            return _run_batch(args, _emit_json)

        from core.multi_focus_fusion import MultiFocusFusion

        # Video inputs are unpacked to a temp dir that is always cleaned up
        with tempfile.TemporaryDirectory(prefix="openfocus_video_") as video_temp:
            _src, paths = _resolve_stack(args, video_temp)
            print(f"Loaded {len(paths)} image(s) from {_src}")

            if args.dry_run:
                print(f"[dry-run] method={args.method} align={args.align} "
                      f"frames={len(paths)} output={args.output}")
                for p in paths:
                    print(f"[dry-run]   {p}")
                _emit_json({"mode": "single", "dry_run": True,
                            "method": args.method, "align": args.align,
                            "frames": len(paths), "input": args.input,
                            "output": args.output})
                return EXIT_OK

            _progress(event="load", done=1, total=1, frames=len(paths))
            images = _load_images(
                paths, thread_count=getattr(args, "threads", 0) or 0,
                resource_preference=getattr(args, "resource_preference",
                                            "balanced"))
            if paths:
                from utils.image_utils import set_source_exif
                from core.image_loader import _read_exif_bytes
                set_source_exif(_read_exif_bytes(paths[0]))

            fused = _fuse_stack(images, args)
            if fused is None:
                raise RuntimeError("Fusion returned no result")
            scale_cfg = _resolve_scale_cfg(args, paths)

        _save(args.output, fused, scale_cfg)
        print(f"Saved: {args.output}")
        _progress(event="save", output=args.output)
        outputs = [args.output]
        if args.depth_map:
            _save_depth(args, args.output)
            outputs.append(_depth_out_path(args, args.output))
        _emit_json({"mode": "single", "method": args.method, "align": args.align,
                    "frames": len(paths), "input": args.input,
                    "output": args.output, "outputs": outputs})
        return EXIT_OK
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        _log.exception("CLI processing failed")
        _emit_json({"mode": "single", "error": str(exc)})
        return EXIT_ERROR


def main() -> None:
    """Console-script entry point (pip-installed ``openfocus`` command)."""
    from utils.diagnostics import install_exception_hooks
    setup_logging()
    log_session_start()
    install_exception_hooks()
    sys.exit(run_cli(sys.argv[1:]))
