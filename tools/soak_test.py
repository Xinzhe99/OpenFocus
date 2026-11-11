"""OpenFocus large-stack soak test.

Measures wall time and peak working-set memory for every fusion algorithm
across realistic stack sizes (with registration and tiling variants), and
verifies output sanity (correct shape, non-degenerate pixels).

Run (project env):
    python tools/soak_test.py [--quick] [--json out.json]

--quick skips the 120-frame synthetic case and 16-bit variants.
Results table is printed and (with --json) dumped for the docs.
Synthetic and IO cases require the hetao photo fixture. Skipped and unfinished
cases are reported; zero executed cases are unsuccessful.
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

from core.algorithms import fusion_algorithms, get_fusion


class _PeakMemoryWindows:
    """Working-set tracker via GetProcessMemoryInfo.

    argtypes/restype must be declared: without them the Win64 call can
    silently fail and leave the struct zeroed (all readings 0MB).
    """

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

    def __init__(self):
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        self._fn = psapi.GetProcessMemoryInfo
        self.proc = ctypes.windll.kernel32.GetCurrentProcess()
        self.pmc = self.PMC()
        self.pmc.cb = ctypes.sizeof(self.PMC)

    def _query(self):
        ok = self._fn(self.proc, ctypes.byref(self.pmc), self.pmc.cb)
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return self.pmc

    def now_mb(self) -> float:
        """Current working set (snapshot at end of a case)."""
        return self._query().WorkingSetSize / (1024 * 1024)

    def peak_mb(self) -> float:
        """Lifetime peak working set (process-wide, monotonic)."""
        return self._query().PeakWorkingSetSize / (1024 * 1024)


_TRACKER = None


def _mem_mb():
    global _TRACKER
    if os.name == "nt":
        if _TRACKER is None:
            _TRACKER = _PeakMemoryWindows()
        return _TRACKER.now_mb()
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


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
    weights_name = get_fusion(algo).weights
    if weights_name:
        from utils.weights import weights_path
        kw["model_path"] = weights_path(os.path.basename(weights_name))
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
    ap.add_argument("--only", default="",
                    help="comma-separated substrings; run only matching case labels "
                         "(e.g. 'synth120/stackmffv3,u16,io-roundtrip')")
    args = ap.parse_args()
    wanted = [s.strip().lower() for s in args.only.split(",") if s.strip()]

    def want(label):
        return not wanted or any(s in label.lower() for s in wanted)

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Sections A-C need the author's real stacks. Point OPENFOCUS_SOAK_DIR at
    # your own photo folder, or the sections are skipped with a notice —
    # indexing an empty list used to crash the harness before any case ran.
    photo_root = os.environ.get(
        "OPENFOCUS_SOAK_DIR", os.path.expanduser("~"))
    fixture_error = None
    try:
        hetao = load_stack(os.path.join(photo_root, "hetao_aligned_stabled"))
        samples = load_stack(os.path.join(photo_root, "Samples1"))
    except (Exception, KeyboardInterrupt) as exc:
        hetao, samples = [], []
        fixture_error = f'fixture loading: {type(exc).__name__}: {exc}'
    if hetao:
        print(f"hetao  : {len(hetao)} frames {hetao[0].shape}")
    else:
        print(f"hetao  : not found under {photo_root} — sections A/B skipped "
              f"(set OPENFOCUS_SOAK_DIR to a folder with real stacks)")
    if samples:
        shape = samples.shape if hasattr(samples, "shape") else samples[0].shape
        print(f"samples: {len(samples)} frames {shape}")
    else:
        print(f"samples: not found under {photo_root} — section C skipped")

    results, failures = [], []
    if fixture_error:
        failures.append(fixture_error)
    algos = [spec.id for spec in fusion_algorithms()]
    skipped = []
    candidates = [(f'hetao40/{a}', bool(hetao), 'hetao fixture missing') for a in algos]
    candidates += [(f'hetao40/ecc+{a}', bool(hetao), 'hetao fixture missing') for a in algos]
    candidates += [(f'samples24/{a}', bool(samples), 'samples fixture missing') for a in algos]
    candidates += [(f'synth120/{a}', bool(hetao) and not args.quick, 'hetao fixture missing or --quick') for a in algos]
    candidates += [(f'u16-hetao40/{a}', bool(hetao) and not args.quick, 'hetao fixture missing or --quick') for a in algos]
    candidates += [('io-roundtrip', bool(hetao), 'hetao fixture missing')]
    selected = [label for label, _, _ in candidates if want(label)]
    skipped = [{'case': label, 'reason': reason} for label, available, reason in candidates if want(label) and not available]

    current = None
    interrupted = fixture_error is not None
    big = u16 = None
    try:
        for label, available, _reason in candidates:
            if not want(label) or not available:
                continue
            current = label
            if label == 'io-roundtrip':
                from utils.scalebar import draw_scale_bar
                from utils.image_utils import imwrite_multi_tiff
                import tempfile
                started = time.time()
                out = fuse(hetao, 'guided_filter')
                bar = draw_scale_bar(out, 0.35, 'bottom-right', 'auto')
                assert bar.shape == out.shape
                with tempfile.TemporaryDirectory() as td:
                    p = os.path.join(td, 'soak.tif')
                    assert imwrite_multi_tiff(p, hetao)
                    ok, pages = cv2.imdecodemulti(np.fromfile(p, dtype=np.uint8),
                                                   flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                    assert ok and len(pages) == len(hetao)
                results.append({'case': label, 'frames': len(hetao), 'algo': '-',
                                'reg': '-', 'total_s': round(time.time() - started, 1),
                                'rss_mb': round(_mem_mb(), 0)})
            else:
                section, algo = label.split('/', 1)
                registration = None
                if algo.startswith('ecc+'):
                    registration, algo = 'ecc', algo[4:]
                if section == 'samples24':
                    frames = samples
                elif section == 'synth120':
                    if big is None:
                        big = synth_big_stack(hetao, 120)
                    frames = big
                elif section == 'u16-hetao40':
                    if u16 is None:
                        u16 = [im.astype(np.uint16) * 257 for im in hetao]
                    frames = u16
                else:
                    frames = hetao
                run_case(label, frames, algo, failures, results, reg=registration)
            current = None
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = True
        failures.append(f'{current or "setup"}: {type(exc).__name__}: {exc}')
    completed = {result['case'] for result in results}
    skipped_names = {item['case'] for item in skipped}
    unfinished = [label for label in selected if label not in completed and label not in skipped_names]

    print("\n===== SUMMARY =====")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
    elif results:
        print(f'Passed {len(results)} executed soak cases (output sanity verified).')
    else:
        print('No soak cases executed' + (f' (filter: {args.only})' if args.only else ''))
    for item in skipped:
        print(f"  skipped {item['case']}: {item['reason']}")
    if _TRACKER is not None:
        print(f"process peak working set: {_TRACKER.peak_mb():.0f}MB")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({'results': results, 'failures': failures, 'selected': selected,
                       'skipped': skipped, 'unfinished': unfinished, 'interrupted': interrupted,
                       'executed_count': len(results), 'passed': bool(results) and not failures and not unfinished}, fh, indent=1)
        print(f"json -> {args.json}")
    return 0 if results and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
