"""Time every fusion method and hash its output.

Use the same stack for before/after:

    python tools/algorithm_bench.py --make-stack
    python tools/algorithm_bench.py baseline
    ... change code ...
    python tools/algorithm_bench.py optimised
    python tools/algorithm_bench.py --compare baseline optimised

A failed method is recorded as SKIP with its exception and makes the run fail.
Comparison requires complete SHA-256 digests, matching shape and dtype, and
successful results for the same nonempty set of methods. Historical reports
with 16-character digests can be displayed but cannot prove identity.
Needs a Python with every backend installed (3.10-3.12 for dtcwt).
"""
import argparse
import ast
import hashlib
import math
import os
import re
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.algorithms import fusion_algorithms

STACK_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "_perf", "stack")


def load_report(path):
    """Read and validate a benchmark report, including legacy short digests."""
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.endswith("\n"):
                raise ValueError(f"{path}:{line_number}: truncated row")
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected five columns")
            method, elapsed, digest, shape, dtype = fields
            if not method.strip() or method != method.strip() or method in rows:
                raise ValueError(f"{path}:{line_number}: empty or duplicate method {method!r}")
            if digest == "SKIP":
                if elapsed != "None" or dtype:
                    raise ValueError(f"{path}:{line_number}: invalid SKIP row")
                rows[method] = (None, digest, shape, dtype)
                continue
            try:
                seconds = float(elapsed)
                dimensions = ast.literal_eval(shape)
                np.dtype(dtype)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid time or metadata") from exc
            if (not math.isfinite(seconds) or seconds <= 0
                    or not isinstance(dimensions, tuple) or not dimensions
                    or any(type(d) is not int or d < 0 for d in dimensions)
                    or not dtype.strip()):
                raise ValueError(f"{path}:{line_number}: invalid time or metadata")
            if not re.fullmatch(r"(?:[0-9a-f]{16}|[0-9a-f]{64})", digest):
                raise ValueError(f"{path}:{line_number}: invalid SHA-256 digest")
            rows[method] = (seconds, digest, dimensions, dtype)
    if not rows:
        raise ValueError(f"{path}: empty report")
    return rows


def compare_reports(before_path, after_path):
    """Print a comparison and return its process exit status."""
    try:
        before, after = load_report(before_path), load_report(after_path)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Cannot compare reports: {exc}", file=sys.stderr)
        return 1
    print(f"{'method':15} {'before':>9} {'after':>9} {'speedup':>8}  output")
    same = True
    for method in sorted(before.keys() | after.keys()):
        left, right = before.get(method), after.get(method)
        if left is None or right is None:
            print(f"{method:15} missing from {'before' if left is None else 'after'}")
            same = False
            continue
        tb, db, sb, yb = left
        ta, da, sa, ya = right
        ratio = f"{tb / ta:.2f}x" if tb is not None and ta is not None else "-"
        if db == "SKIP" or da == "SKIP":
            details = "; ".join(f"{side}: {row[2]}" for side, row in
                                (("before", left), ("after", right))
                                if row[1] == "SKIP" and row[2])
            status = "unavailable" + (f" ({details})" if details else "")
            same = False
        elif len(db) != 64 or len(da) != 64:
            status = "unavailable (legacy short digest)"
            same = False
        elif (sb, yb, db) != (sa, ya, da):
            status = "CHANGED" + (" (metadata)" if (sb, yb) != (sa, ya) else "")
            same = False
        else:
            status = "identical"
        print(f"{method:15} {str(tb) if tb is not None else '-':>9} "
              f"{str(ta) if ta is not None else '-':>9} {ratio:>8}  {status}")
    print("\nall outputs bit-identical" if same else
          "\nUnable to prove output identity")
    return 0 if same else 1


def make_stack():
    os.makedirs(STACK_DIR, exist_ok=True)
    rng = np.random.default_rng(12345)
    H, W = 720, 960
    base_img = cv2.GaussianBlur(
        rng.integers(0, 256, (H, W, 3), dtype=np.uint8), (0, 0), 1.2)
    for i in range(5):
        blur = cv2.GaussianBlur(base_img, (0, 0), 3.0 + i)
        y0 = i * H // 5
        frame = blur.copy()
        frame[y0:y0 + H // 5] = base_img[y0:y0 + H // 5]
        cv2.imwrite(os.path.join(STACK_DIR, f"f{i}.png"), frame)
    print(f"wrote 5 frames to {STACK_DIR}")


def run_benchmark(label):
    frames = [cv2.imread(os.path.join(STACK_DIR, f"f{i}.png")) for i in range(5)]
    if any(f is None for f in frames):
        raise SystemExit("run --make-stack first")
    print(f"stack: {len(frames)} x {frames[0].shape[1]}x{frames[0].shape[0]}", flush=True)

    from core.multi_focus_fusion import MultiFocusFusion

    rows = []
    succeeded = failed = 0
    for method in (spec.id for spec in fusion_algorithms()):
        try:
            fuser = MultiFocusFusion(algorithm=method, use_gpu=False,
                                     tile_enabled=False)
            # Three timed runs, report the minimum to reduce desktop scheduling noise.
            runs = []
            out = None
            for _ in range(3):
                t0 = time.perf_counter()
                out = fuser.fuse(frames)
                runs.append(time.perf_counter() - t0)
            dt = min(runs)
            arr = np.ascontiguousarray(out)
            digest = hashlib.sha256(arr.tobytes()).hexdigest()
            rows.append((method, dt, digest, arr.shape, str(arr.dtype)))
            succeeded += 1
            print(f"{method:14} {dt:7.3f}s  {digest}  {arr.shape} {arr.dtype}", flush=True)
        except Exception as exc:  # noqa: BLE001
            diagnostic = f"{type(exc).__name__}: {exc}".replace("\t", " ").replace("\r", " ").replace("\n", " ")
            print(f"{method:14} SKIP  {diagnostic}", flush=True)
            rows.append((method, None, "SKIP", diagnostic, ""))
            failed += 1

    perf_dir = os.path.dirname(STACK_DIR)
    destination = os.path.join(perf_dir, f"{label}.txt")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=perf_dir, delete=False) as fh:
            temp_path = fh.name
            for row in rows:
                fh.write("\t".join(str(value) for value in row) + "\n")
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            os.unlink(temp_path)
    print(f"\nwritten {destination} ({succeeded} succeeded, {failed} failed)")
    return 1 if failed or not succeeded else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label", nargs="?", default="run", help="report label")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--make-stack", action="store_true", help="generate the input stack")
    group.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                       help="compare reports; SKIP and legacy short digests cannot pass")
    args = parser.parse_args(argv)
    if args.make_stack:
        make_stack()
        return 0
    if args.compare:
        perf_dir = os.path.dirname(STACK_DIR)
        return compare_reports(*(os.path.join(perf_dir, label + ".txt")
                                 for label in args.compare))
    return run_benchmark(args.label)


if __name__ == "__main__":
    raise SystemExit(main())
