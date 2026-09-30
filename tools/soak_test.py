"""OpenFocus large-stack soak test.

Measures wall time and peak working-set memory for every fusion algorithm
across realistic stack sizes (with registration and tiling variants), and
verifies output sanity (correct shape, non-degenerate pixels).

Run (project env):
    python tools/soak_test.py [--quick] [--json out.json]

--quick skips the 120-frame synthetic case and 16-bit variants.
Results table is printed and (with --json) dumped for the docs.
"""
import argparse
import ctypes
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _PeakMemoryWindows:
    """Peak working-set tracker via GetProcessMemoryInfo (KB)."""

    def __init__(self):
        self.proc = ctypes.windll.kernel32.GetCurrentProcess()
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint), ("PageFaultCount", ctypes.c_uint),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        self.pmc = PMC()
        self.pmc.cb = ctypes.sizeof(PMC)

    def peak_mb(self) -> float:
        ctypes.windll.psapi.GetProcessMemoryInfo(self.proc, ctypes.byref(self.pmc), self.pmc.cb)
        return self.pmc.PeakWorkingSetSize / (1024 * 1024)


def _mem_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return _PeakMemoryWindows().peak_mb()


def load_stack(folder):
    files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                   if os.path.splitext(f)[1].lower() in
                   (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"))
    imgs = []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
    return imgs


def synth_big_stack(base_stack, target_n):
    """Cycle+shift a real stack up to target_n frames (memory-realistic)."""
    rng = np.random.default_rng(7)
    out = []
    for i in range(target_n):
        img = base_stack[i % len(base_stack)]
        dx, dy = int(rng.integers(-9, 10)), int(rng.integers(-9, 10))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                  borderMode=cv2.BORDER_REFLECT))
    return out


def sanity(name, out, ref_shape, failures):
    """ref_shape must be the shape actually fused (post-registration)."""
    if out is None:
        failures.append(f"{name}: returned None")
        return
    if out.shape[:2] != ref_shape[:2]:
        failures.append(f"{name}: shape {out.shape[:2]} != {ref_shape[:2]}")
    if float(np.asarray(out).mean()) < 1.0:
        failures.append(f"{name}: output near-black (mean<1)")
    if float(np.asarray(out).std()) < 0.5:
        failures.append(f"{name}: output flat (std<0.5)")


def fuse(imgs, algo, threads=4):
    from core.multi_focus_fusion import MultiFocusFusion
    from utils.image_utils import normalize_fuse_input, quantize_fuse_output
    norm, is16 = normalize_fuse_input(list(imgs))
    f = MultiFocusFusion(algorithm=algo, use_gpu=False)
    kw = dict(input_source=norm, img_resize=None, thread_count=threads)
    if algo == "guided_filter":
        kw["kernel_size"] = 31
    elif algo in ("dct", "gfgfgf"):
        kw["kernel_size"] = 7
    elif algo == "stackmffv4":
        from core.cli import _find_model_path
        kw["model_path"] = _find_model_path()
    out = f.fuse(**kw)
    return quantize_fuse_output(out, is16)


def run_case(label, imgs, algo, failures, results, reg=None):
    from core.registration import ImageRegistration
    gc_hint = _mem_mb()
    t0 = time.time()
    frames = imgs
    reg_s = 0.0
    if reg:
        tR = time.time()
        frames = ImageRegistration(method=reg).process(list(imgs), output_path=None, thread_count=4)
        reg_s = time.time() - tR
    out = fuse(frames, algo)
    dt = time.time() - t0
    rss = _mem_mb()
    # ECC crops to the common valid region by design; compare against the
    # frames that were actually fused, and report absolute RSS (freed
    # memory makes deltas meaningless).
    sanity(label, out, frames[0].shape, failures)
    results.append({"case": label, "frames": len(imgs), "algo": algo,
                    "reg": reg or "-", "fuse_s": round(dt - reg_s, 1),
                    "reg_s": round(reg_s, 1), "total_s": round(dt, 1),
                    "rss_mb": round(rss, 0)})
    print(f"  {label:<38} {len(imgs):>4}f  fuse={dt - reg_s:7.1f}s  reg={reg_s:6.1f}s  rss={rss:7.0f}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hetao = load_stack(r"C:\Users\dell\Pictures\Helicon Focus\hetao_aligned_stabled")
    samples = load_stack(r"C:\Users\dell\Pictures\Helicon Focus\Samples1")
    print(f"hetao  : {len(hetao)} frames {hetao[0].shape}")
    print(f"samples: {len(samples)} frames {samples.shape if hasattr(samples,'shape') else samples[0].shape}")

    results, failures = [], []
    algos = ["guided_filter", "dct", "dtcwt", "gfgfgf", "stackmffv4"]

    print("\n== A. real 40-frame stack, each algorithm (no registration) ==")
    for a in algos:
        run_case(f"hetao40/{a}", hetao, a, failures, results)

    print("\n== B. real 40-frame stack + ECC registration ==")
    for a in ("guided_filter", "dtcwt", "stackmffv4"):
        run_case(f"hetao40/ecc+{a}", hetao, a, failures, results, reg="ecc")

    print("\n== C. high-resolution stack (3636x4756 x24) ==")
    for a in ("guided_filter", "dtcwt", "stackmffv4"):
        run_case(f"samples24/{a}", samples, a, failures, results)

    print("\n== D. synthetic 120-frame stack (real frames cycled) ==")
    if not args.quick:
        big = synth_big_stack(hetao, 120)
        for a in ("guided_filter", "dtcwt", "stackmffv4"):
            run_case(f"synth120/{a}", big, a, failures, results)
        del big

    print("\n== E. 16-bit variants ==")
    if not args.quick:
        u16 = [im.astype(np.uint16) * 257 for im in hetao]
        for a in ("guided_filter", "dtcwt"):
            run_case(f"u16-hetao40/{a}", u16, a, failures, results)
        del u16

    print("\n== F. scale bar + multipage round-trip on the big output ==")
    try:
        from utils.scalebar import draw_scale_bar
        from utils.image_utils import imwrite_multi_tiff
        import tempfile
        out = fuse(hetao, "guided_filter")
        bar = draw_scale_bar(out, 0.35, "bottom-right", "auto")
        assert bar.shape == out.shape
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, " soak.tif".strip())
            assert imwrite_multi_tiff(p, hetao)
            ok, pages = cv2.imdecodemulti(np.fromfile(p, dtype=np.uint8),
                                           flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            assert ok and len(pages) == len(hetao)
        print("  scalebar + multipage round-trip: OK")
        results.append({"case": "io-roundtrip", "frames": len(hetao), "algo": "-",
                        "reg": "-", "fuse_s": 0, "reg_s": 0, "total_s": 0, "peak_mb": 0})
    except Exception as exc:
        failures.append(f"io-roundtrip: {exc}")

    print("\n===== SUMMARY =====")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
    else:
        print("All soak cases passed (output sanity verified).")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"results": results, "failures": failures}, fh, indent=1)
        print(f"json -> {args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
