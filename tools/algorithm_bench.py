"""Time every fusion method and hash its output.

Hashes make a performance change auditable: an optimisation that alters even
one pixel shows up as a different digest. Use the same stack for before/after:

    python tools/algorithm_bench.py --make-stack
    python tools/algorithm_bench.py baseline
    ... change code ...
    python tools/algorithm_bench.py optimised
    python tools/algorithm_bench.py --compare baseline optimised

Needs a Python with every backend installed (3.10-3.12 for dtcwt).
"""
import hashlib
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.algorithms import fusion_algorithms

STACK_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "_perf", "stack")

if "--make-stack" in sys.argv:
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
    raise SystemExit(0)

if "--compare" in sys.argv:
    a, b = sys.argv[sys.argv.index("--compare") + 1:][:2]
    def load(path):
        out = {}
        with open(os.path.join(os.path.dirname(STACK_DIR), path + ".txt"),
                  encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    out[parts[0]] = (parts[1], parts[2])
        return out
    before, after = load(a), load(b)
    print(f"{'method':15} {'before':>9} {'after':>9} {'speedup':>8}  output")
    same = True
    for m in before:
        tb = float(before[m][0]) if before[m][0] != "None" else None
        ta = float(after[m][0]) if m in after and after[m][0] != "None" else None
        ident = m in after and before[m][1] == after[m][1]
        same = same and ident
        ratio = f"{tb/ta:.2f}x" if tb and ta else "-"
        print(f"{m:15} {tb if tb else '-':>8}s {ta if ta else '-':>8}s "
              f"{ratio:>8}  {'identical' if ident else 'CHANGED'}")
    print("\nall outputs bit-identical" if same else "\nOUTPUT CHANGED")
    raise SystemExit(0 if same else 1)

label = sys.argv[1] if len(sys.argv) > 1 else "run"
frames = [cv2.imread(os.path.join(STACK_DIR, f"f{i}.png")) for i in range(5)]
if any(f is None for f in frames):
    raise SystemExit("run --make-stack first")
print(f"stack: {len(frames)} x {frames[0].shape[1]}x{frames[0].shape[0]}", flush=True)

from core.multi_focus_fusion import MultiFocusFusion

rows = []
for method in (spec.id for spec in fusion_algorithms()):
    try:
        fuser = MultiFocusFusion(algorithm=method, use_gpu=False,
                                 tile_enabled=False)
        # Three timed runs, report the minimum. A single run on a desktop OS
        # swings by 40% (other processes, cache state); comparing medians of
        # n=1 is how OpenFocus B "got 30% slower" from a change that only
        # touched DTCWT.
        runs = []
        out = None
        for _ in range(3):
            t0 = time.perf_counter()
            out = fuser.fuse(frames)
            runs.append(time.perf_counter() - t0)
        dt = min(runs)
        arr = np.ascontiguousarray(out)
        digest = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
        rows.append((method, dt, digest, arr.shape, str(arr.dtype)))
        print(f"{method:14} {dt:7.3f}s  {digest}  {arr.shape} {arr.dtype}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{method:14} SKIP  {type(exc).__name__}: {str(exc)[:70]}", flush=True)
        rows.append((method, None, "SKIP", "", ""))

perf_dir = os.path.dirname(STACK_DIR)
with open(os.path.join(perf_dir, f"{label}.txt"), "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write("\t".join(str(x) for x in r) + "\n")
print(f"\nwritten _perf/{label}.txt")
