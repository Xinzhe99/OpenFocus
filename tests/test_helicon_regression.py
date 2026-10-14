"""Helicon-benchmark regression tests for the reverse-engineered ports.

Locks ``openfocus_a/b/c`` to the fidelity measured against the local Helicon
Focus 8.1 CLI outputs collected during the reverse-engineering project
(docs/HELICON_REPRODUCTION_REPORT.md, tools/helicon_repro/audit_port_v139.py).
If someone later changes the variance window, the depth smoothing, the
pyramid decision rule or the base-correction sigma, these tests catch the
fidelity regression before it ships.

Skips automatically when the benchmarks are not on this machine, so CI stays
green; run locally next to the reference library to guard Helicon parity:

  refs    (default) F:\\Working\\helicon-rea\\tests\\out      -> $OPENFOCUS_HELICON_REFS
  stacks  (default) E:\\ImagefusionDatasets\\StackMFF_jpg\\Mobile Depth\\image stack
                                                  -> $OPENFOCUS_STACKMFF_DATA

Thresholds carry ~25-30% headroom over the measured registered MAE (2026-10-14,
cv2 4.11): B real 1.18/1.77/3.49 %, C real 1.37/2.00/3.51 %, synthetic A
1.76/4.90, B 4.88/3.04, C 0.50/4.99/3.39 %. The synthetic stackB cases were
6.5 % under the old center-crop comparison — Helicon's AutoCrop is not
centered, so _metrics() sweeps integer offsets (report §7.4) instead.
"""
import glob
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
PILImage = pytest.importorskip("PIL.Image",
                               reason="benchmark decoding uses PIL")

REFS = os.environ.get("OPENFOCUS_HELICON_REFS",
                      r"F:\Working\helicon-rea\tests\out")
DATA = os.environ.get("OPENFOCUS_STACKMFF_DATA",
                      r"E:\ImagefusionDatasets\StackMFF_jpg\Mobile Depth\image stack")
SYNTH = os.path.dirname(REFS)

if not os.path.isdir(REFS):
    pytest.skip("Helicon reference library not present (%s)" % REFS,
                allow_module_level=True)


# ------------------------------------------------------------------ helpers
def _load_rgb(path):
    return np.asarray(PILImage.open(path).convert("RGB"), dtype=np.float64)


def _metrics(mine, ref, m=4, R=5):
    """MAE % of ref range, minimized over integer crop offsets.

    Helicon's AutoCrop is *not* centered: the kept window sits off by up to a
    couple of pixels (docs/HELICON_REPRODUCTION_REPORT.md §7.4 measured
    (0,+1) and (2,2)). A naive center-crop then compares two grids shifted by
    1-2 px, which costs ~25 levels of pure misalignment and inflates the
    reported error up to 9x — a normal alignment-semantics difference would
    ship as an "algorithm regression". Sweeping small integer offsets and
    taking the minimum measures the fusion, not the crop convention.
    """
    best = None
    # Same-size overlap first (centered), then sweep integer offsets — the
    # identical recipe to tools/helicon_repro/registered_table.py, with the
    # overlap length computed per offset so *every* shift in [-R, R] is
    # evaluated (a fixed length silently drops |offset| > m).
    hh = min(mine.shape[0], ref.shape[0])
    ww = min(mine.shape[1], ref.shape[1])
    a0 = mine[(mine.shape[0] - hh) // 2:(mine.shape[0] - hh) // 2 + hh,
              (mine.shape[1] - ww) // 2:(mine.shape[1] - ww) // 2 + ww]
    b0 = ref[(ref.shape[0] - hh) // 2:(ref.shape[0] - hh) // 2 + hh,
             (ref.shape[1] - ww) // 2:(ref.shape[1] - ww) // 2 + ww]
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            ya, yb = m + max(dy, 0), m + max(-dy, 0)
            xa, xb = m + max(dx, 0), m + max(-dx, 0)
            h, w = min(hh - ya, hh - yb) - m, min(ww - xa, ww - xb) - m
            if h < 8 or w < 8:
                continue
            a, b = a0[ya:ya + h, xa:xa + w], b0[yb:yb + h, xb:xb + w]
            e = float(np.abs(a - b).mean()) / max(float(b.max()), 1.0) * 100.0
            if best is None or e < best:
                best = e
    if best is None:  # images too small to sweep — fall back to center crop
        a, b = a0[m:-m, m:-m], b0[m:-m, m:-m]
        best = float(np.abs(a - b).mean()) / max(float(b.max()), 1.0) * 100.0
    return best


def _box_blur(a, r):
    k = 2 * r + 1
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    c = np.cumsum(p, 0)
    a = (c[k - 1:] - np.concatenate([np.zeros((1,) + a.shape[1:]), c[:-k]])) / k
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    c = np.cumsum(p, 1)
    return (c[:, k - 1:] - np.concatenate(
        [np.zeros((a.shape[0], 1)), c[:, :-k]], 1)) / k


def _gauss1d(a, sigma):
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad0 = ((r, r), (0, 0)) + ((0, 0),) * (a.ndim - 2)
    a = sum(k[i] * np.pad(a, pad0, mode="edge")[i:i + a.shape[0]]
            for i in range(2 * r + 1))
    pad1 = ((0, 0), (r, r)) + ((0, 0),) * (a.ndim - 2)
    return sum(k[i] * np.pad(a, pad1, mode="edge")[:, i:i + a.shape[1]]
               for i in range(2 * r + 1))


def _hanning_corr(ref, mov):
    """Integer FFT cross-correlation (session-validated aligner)."""
    h, w = ref.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    a = ref * win
    b = mov * win
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    corr = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=ref.shape)
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = py, px
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return dy, dx


def _align_real(frames):
    """Sub-pixel align to frame 0 + margin-1 crop (real-data pipeline)."""
    gray = frames.mean(axis=3)
    ref = gray[0]
    out = [frames[0]]
    shifts = [(0.0, 0.0)]

    def subpix(a, dy, dx):
        y0, x0 = int(np.floor(dy)), int(np.floor(dx))
        fy, fx = dy - y0, dx - x0
        t = lambda yy, xx: np.roll(np.roll(a, yy, 0), xx, 1)
        return ((1 - fy) * (1 - fx) * t(y0, x0) + fy * (1 - fx) * t(y0 + 1, x0)
                + (1 - fy) * fx * t(y0, x0 + 1) + fy * fx * t(y0 + 1, x0 + 1))

    for k in range(1, len(frames)):
        dy, dx = _hanning_corr(ref, gray[k])
        if abs(dy) < 0.05 and abs(dx) < 0.05:
            out.append(frames[k])
            shifts.append((0.0, 0.0))
        else:
            out.append(np.stack([subpix(frames[k, ..., c], dy, dx)
                                 for c in range(3)], axis=2))
            shifts.append((dy, dx))
    ys = int(max(abs(s[0]) for s in shifts)) + 1
    xs = int(max(abs(s[1]) for s in shifts)) + 1
    h, w = gray.shape[1:]
    sl = (slice(ys, h - ys), slice(xs, w - xs))
    return [f[sl] for f in out]


def _base_correct(out, last, sigma=32.0):
    a = np.ascontiguousarray(out, np.float32)
    b = np.ascontiguousarray(last, np.float32)
    return a + cv2.GaussianBlur(b, (0, 0), sigma) - cv2.GaussianBlur(a, (0, 0), sigma)


def _load_real(stack):
    files = sorted(glob.glob(os.path.join(DATA, stack, "*.jpg")))
    assert len(files) >= 2, "no frames for %s" % stack
    return np.stack([_load_rgb(f) for f in files])


def _run(method, frames, base_last=True):
    from fusion_methods import openfocus_ab as ofab
    frames_f = [f.astype(np.float32) for f in frames]
    fn = {"a": ofab.openfocus_a_impl, "b": ofab.openfocus_b_impl,
          "c": ofab.openfocus_c_impl}[method]
    out = fn(list(frames_f), None, 7)
    out = np.asarray(out, np.float64)
    if base_last and method != "c":  # C carries base-band-last intrinsically
        out = _base_correct(out, frames_f[-1])
    return out


def _load_synth(name, tag):
    files = sorted(glob.glob(os.path.join(SYNTH, name, "*.tif")))
    raw = [np.asarray(PILImage.open(p), dtype=np.float64) for p in files]
    # 合成基准是 16-bit 灰度——保持原样，不做 RGB 转换（与验证管线一致）
    ref_raw = np.asarray(PILImage.open(os.path.join(REFS, tag)), dtype=np.float64)
    scale = 255.0 / max(max(f.max() for f in raw), ref_raw.max())
    al = [raw[0]]
    for f in raw[1:]:
        dy, dx = _hanning_corr(raw[0], f)
        moved = np.roll(np.roll(f, dy, 0), dx, 1)
        al.append(moved)
    ys = xs = 1
    h, w = raw[0].shape
    return [ (f * scale)[ys:h - ys, xs:w - xs] for f in al ], ref_raw * scale


# ------------------------------------------------------------------ tests
# Budgets = registered MAE measured 2026-10-14 (offset sweep, cv2 4.11),
# ~25-30% headroom. Real: B 1.18-3.49 / C 1.37-3.51 %; synth stackB 4.88-4.99
# (naive center-crop was 6.45-6.56 — the asymmetry of Helicon's AutoCrop,
# see _metrics).
REAL_B_BUDGET = {"keyboard": 1.6, "balls": 2.3, "kitchen": 4.2}
REAL_C_BUDGET = {"keyboard": 1.8, "balls": 2.6, "kitchen": 4.5}
SYNTH_BUDGET = {
    ("stackA", "a", "A_m0.tif"): 2.3,
    ("stackB", "a", "B_m0_r3_s3.tif"): 6.1,
    ("stackB", "b", "B_m1_r1_s0.tif"): 6.1,
    ("stackC", "b", "refB_stackC.tif"): 3.5,
    ("stackA", "c", "refC_stackA.tif"): 1.0,
    ("stackB", "c", "refC_r1_s0.tif"): 6.2,
    ("stackC", "c", "refC_stackC.tif"): 4.0,
}


class TestHeliconParity:
    @pytest.mark.parametrize("stack,budget", sorted(REAL_B_BUDGET.items()))
    def test_method_b_real_stack(self, stack, budget):
        ref = _load_rgb(os.path.join(REFS, "real_%s_r3_s3.tif" % stack))
        frames = _align_real(_load_real(stack))
        assert _metrics(_run("b", frames), ref) < budget

    @pytest.mark.parametrize("stack,budget", sorted(REAL_C_BUDGET.items()))
    def test_method_c_real_stack(self, stack, budget):
        ref_path = os.path.join(REFS, "realC_%s_r3_s3.tif" % stack)
        if not os.path.exists(ref_path):
            pytest.skip("no Helicon C reference for %s" % stack)
        ref = _load_rgb(ref_path)
        frames = _align_real(_load_real(stack))
        assert _metrics(_run("c", frames), ref) < budget

    @pytest.mark.parametrize("case,budget",
                             sorted(SYNTH_BUDGET.items(), key=str))
    def test_synthetic_stack(self, case, budget):
        stack, method, tag = case
        if not os.path.exists(os.path.join(SYNTH, stack)):
            pytest.skip("synthetic stack %s not present" % stack)
        frames, ref = _load_synth(stack, tag)
        assert _metrics(_run(method, frames), ref) < budget
