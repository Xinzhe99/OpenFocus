"""Headless command-line interface for OpenFocus fusion.

Lets batch users and CI pipelines run registration + fusion without the GUI:

    python main.py --input ./stack_folder --output ./out --method guided_filter

Exit codes: 0 success, 1 processing error, 2 usage error.
"""
import argparse
import os
import sys
import time
from typing import List, Optional

VALID_METHODS = ["guided_filter", "dct", "dtcwt", "gfgfgf", "stackmffv4"]
VALID_ALIGN = ["none", "homography", "ecc", "both"]
VALID_FORMATS = [".png", ".jpg", ".bmp", ".tif", ".tiff"]

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
            "  python main.py --input ./stack --output ./result\n"
            "  python main.py --input ./stack --output ./result --method dtcwt --align ecc --threads 8\n"
            "  python main.py --input img1.jpg img2.jpg img3.jpg --output fused.png --method gfgfgf\n"
        ),
    )
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="Image folder, video file, or a list of image files")
    parser.add_argument("--output", "-o", required=True,
                        help="Output file (single stack) or folder (default OpenFocus_result.png)")
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
    return parser


def _resolve_stack(args, video_temp_root: str) -> List[str]:
    """Return image file paths from a folder, video, or explicit file list.

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
            paths.extend(sorted(
                os.path.join(entry, f) for f in os.listdir(entry)
                if f.lower().endswith(tuple(loader.SUPPORTED_FORMATS))
            ))
        elif loader.is_video_file(entry):
            ok, message, frames, _names = loader.load_from_video(entry)
            if not ok:
                raise RuntimeError(f"Failed to read video {entry}: {message}")
            import cv2
            for i, frame in enumerate(frames):
                frame_path = os.path.join(video_temp_root, f"frame_{len(paths) + i:05d}.png")
                cv2.imwrite(frame_path, frame)
                paths.append(frame_path)
        elif os.path.isfile(entry):
            paths.append(entry)
        else:
            raise FileNotFoundError(f"Input not found: {entry}")

    if len(paths) < 2:
        raise RuntimeError("Need at least 2 images in the stack for fusion")
    return paths


def _normalize_kernel(kernel: Optional[int], default: int) -> int:
    size = int(kernel) if kernel else default
    if size % 2 == 0:
        size = max(1, size - 1)
    return max(1, size)


def run_cli(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)

    ext = os.path.splitext(args.output)[1].lower()
    if not ext or ext not in VALID_FORMATS:
        print(f"error: unsupported output extension '{ext or '(none)'}'; "
              f"expected one of {', '.join(VALID_FORMATS)}", file=sys.stderr)
        return EXIT_USAGE
    if args.threads < 1:
        print("error: --threads must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    if args.downscale is not None and args.downscale < 1:
        print("error: --downscale must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    if args.batch_size < 1:
        print("error: --batch-size must be >= 1", file=sys.stderr)
        return EXIT_USAGE

    try:
        import tempfile
        from core.registration import ImageRegistration
        from core.multi_focus_fusion import MultiFocusFusion

        # Video inputs are unpacked to a temp dir that is always cleaned up
        with tempfile.TemporaryDirectory(prefix="openfocus_video_") as video_temp:
            paths = _resolve_stack(args, video_temp)
            print(f"Loaded {len(paths)} images from {args.input[0]}")
            import cv2
            images = []
            for p in paths:
                img = cv2.imread(p)
                if img is None:
                    raise RuntimeError(f"Failed to decode image: {p}")
                images.append(img)

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
            fuse_kwargs = dict(input_source=images, img_resize=None,
                               thread_count=args.threads)
            if args.method == "guided_filter":
                fuse_kwargs["kernel_size"] = _normalize_kernel(args.kernel, 31)
            elif args.method in ("dct", "gfgfgf"):
                fuse_kwargs["kernel_size"] = _normalize_kernel(args.kernel, 7)
            elif args.method == "stackmffv4":
                fuse_kwargs["model_path"] = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "weights", "stackmffv4.pth")

            fused = fusion.fuse(**fuse_kwargs)
            if fused is None:
                raise RuntimeError("Fusion returned no result")

        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        if not cv2.imwrite(args.output, fused):
            raise RuntimeError(f"Failed to write output: {args.output}")

        print(f"Fusion ({args.method}, device={info['device']}) done in {time.time() - t0:.1f}s")
        print(f"Saved: {args.output}")
        return EXIT_OK
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
