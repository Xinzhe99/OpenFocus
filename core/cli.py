"""Headless command-line interface for OpenFocus fusion.

Lets batch users and CI pipelines run registration + fusion without the GUI:

    python main.py --input ./stack_folder --output ./out --method guided_filter
    python main.py --input ./stackA ./stackB --output-dir ./results   (batch)

Exit codes: 0 success, 1 processing error, 2 usage error.
"""
import argparse
import os
import sys
import time

import numpy as np
from typing import List, Optional, Tuple

VALID_METHODS = ["guided_filter", "dct", "dtcwt", "gfgfgf", "stackmffv4"]
VALID_ALIGN = ["none", "homography", "ecc", "both"]
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
            "  python main.py --input ./stackA ./stackB --output-dir ./results -m guided_filter\n"
            "  python main.py --input img1.jpg img2.jpg img3.jpg --output fused.png --method gfgfgf\n"
        ),
    )
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="Image folder(s), video file(s), or a list of image files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file for a single stack (default: fused.png)")
    parser.add_argument("--output-dir", "-d", default=None,
                        help="Batch mode: fuse every input folder, writing <folder>.<ext> here")
    parser.add_argument("--method", "-m", default="guided_filter", choices=VALID_METHODS,
                        help="Fusion method (default: guided_filter)")
    parser.add_argument("--align", "-a", default="none", choices=VALID_ALIGN,
                        help="Registration applied before fusion (default: none)")
    parser.add_argument("--kernel", "-k", type=int, default=None,
                        help="Kernel size for guided_filter/dct/gfgfgf (forced odd)")
    parser.add_argument("--threads", "-t", type=int, default=4,
                        help="Worker thread count (default: 4)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="StackMFF-V4 tile batch size (default: 2)")
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
    return parser


def _list_folder_images(folder: str, loader) -> List[str]:
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(tuple(loader.SUPPORTED_FORMATS))
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


def _load_images(paths: List[str]):
    import cv2
    from core.image_loader import _read_tiff_pages
    images = []
    for p in paths:
        # A multi-page TIFF is a whole stack in one file
        pages = _read_tiff_pages(p)
        if pages is not None:
            images.extend(pages)
            continue
        data = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to decode image: {p}")
        images.append(img)
    return images


def _normalize_kernel(kernel: Optional[int], default: int) -> int:
    size = int(kernel) if kernel else default
    if size % 2 == 0:
        size = max(1, size - 1)
    return max(1, size)


def _fuse_stack(images, args):
    """Register and fuse one stack; returns the fused image."""
    import cv2
    from core.registration import ImageRegistration
    from core.multi_focus_fusion import MultiFocusFusion
    from utils.image_utils import normalize_fuse_input, quantize_fuse_output

    # 16-bit sources are normalized to 0-1 float for the back-ends and
    # quantized back to uint16 afterwards, preserving the extra depth.
    images, is_16bit = normalize_fuse_input(images)

    if args.align != "none":
        t0 = time.time()
        reg = ImageRegistration(method=args.align, downscale_width=args.downscale)
        images = reg.process(images, output_path=None, thread_count=args.threads)
        print(f"Registration ({args.align}) done in {time.time() - t0:.1f}s")

    fusion_kwargs = dict(algorithm=args.method, use_gpu=not args.cpu)
    if args.tile_size:
        fusion_kwargs["tile_block_size"] = args.tile_size
    if args.method == "stackmffv4":
        fusion_kwargs["stackmffv4_batch_size"] = args.batch_size
    fusion = MultiFocusFusion(**fusion_kwargs)
    info = fusion.get_info()

    t0 = time.time()
    fuse_kwargs = dict(input_source=images, img_resize=None, thread_count=args.threads)
    if args.method == "guided_filter":
        fuse_kwargs["kernel_size"] = _normalize_kernel(args.kernel, 31)
    elif args.method in ("dct", "gfgfgf"):
        fuse_kwargs["kernel_size"] = _normalize_kernel(args.kernel, 7)
    elif args.method == "stackmffv4":
        fuse_kwargs["model_path"] = _find_model_path()

    fused = fusion.fuse(**fuse_kwargs)
    if fused is None:
        raise RuntimeError("Fusion returned no result")
    fused = quantize_fuse_output(fused, is_16bit)
    # 注册/归一化后的帧留给 --depth-map 用（与融合同一输入）
    args._depth_frames = list(images)
    print(f"Fusion ({args.method}, device={info['device']}) done in {time.time() - t0:.1f}s")
    return fused


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
    kernel = None
    if args.method in ("dct", "gfgfgf"):
        kernel = _normalize_kernel(args.kernel, 7)
    idx = compute_focus_index(
        frames, args.method, kernel_size=kernel,
        model_path=_find_model_path() if args.method == "stackmffv4" else None,
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


def _run_batch(args) -> int:
    """Batch mode: fuse every input folder into --output-dir."""
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
    for folder in folders:
        name = os.path.basename(folder.rstrip("/\\")) or "stack"
        out_path = os.path.join(args.output_dir, f"{name}{args._batch_ext}")
        print(f"\n=== {name} ===")
        try:
            paths = _list_folder_images(folder, loader)
            # Count frames, not files: a folder holding one multi-page TIFF
            # Z-stack is a complete stack (same rule as single-stack mode).
            from core.image_loader import count_frames
            frame_count = sum(count_frames(p) for p in paths)
            if frame_count < 2:
                raise RuntimeError(
                    f"need at least 2 frames, found {frame_count}")
            images = _load_images(paths)
            fused = _fuse_stack(images, args)
            _save(out_path, fused, _resolve_scale_cfg(args, paths))
            print(f"Saved: {out_path}")
            if args.depth_map:
                _save_depth(args, out_path)
            ok_count += 1
        except Exception as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            failures.append(name)

    print(f"\nBatch finished: {ok_count}/{len(folders)} stacks succeeded")
    if failures:
        print("failed: " + ", ".join(failures), file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


def run_cli(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)

    if args.threads < 1:
        print("error: --threads must be >= 1", file=sys.stderr)
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
        if not args.output:
            print("error: one of --output / --output-dir is required", file=sys.stderr)
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
            return _run_batch(args)

        from core.multi_focus_fusion import MultiFocusFusion

        # Video inputs are unpacked to a temp dir that is always cleaned up
        with tempfile.TemporaryDirectory(prefix="openfocus_video_") as video_temp:
            _src, paths = _resolve_stack(args, video_temp)
            print(f"Loaded {len(paths)} images from {args.input[0]}")
            images = _load_images(paths)
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
        if args.depth_map:
            _save_depth(args, args.output)
        return EXIT_OK
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _find_model_path() -> str:
    """Locate stackmffv4.pth: repo checkout first, then the installed
    weights package (pip-installed openfocus ships it as package data)."""
    repo = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "weights", "stackmffv4.pth")
    if os.path.isfile(repo):
        return repo
    try:
        import weights
        shipped = os.path.join(os.path.dirname(weights.__file__), "stackmffv4.pth")
        if os.path.isfile(shipped):
            return shipped
    except ImportError:
        pass
    return repo


def main() -> None:
    """Console-script entry point (pip-installed ``openfocus`` command)."""
    sys.exit(run_cli(sys.argv[1:]))
