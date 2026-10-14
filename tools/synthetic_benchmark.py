"""Synthetic ground-truth benchmark for every fusion method.

A sharp texture image is defocused along a known depth map (smooth ramps,
a bump and a hard occlusion step) into an N-frame focus stack. Because the
all-in-focus answer is known, each method can be scored objectively:

    PSNR / SSIM vs. ground truth, plus the fraction of "ghosted" pixels
    (|error| > 12 in 0-255) and wall time.

Usage:
    python tools/synthetic_benchmark.py                 # all methods, default scene
    python tools/synthetic_benchmark.py -m gfgfgf openfocus_b --frames 24
    python tools/synthetic_benchmark.py --noise 2.0     # sensor noise (sigma, 0-255)
    python tools/synthetic_benchmark.py --scene small   # small near objects

The scene is deterministic (fixed seed), so numbers are comparable between
runs and between code revisions.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ALL_METHODS = ["guided_filter", "dct", "dtcwt", "gfgfgf", "stackmffv4",
               "openfocus_a", "openfocus_b", "openfocus_c"]


def make_texture(h, w, seed=0):
    """Deterministic, detail-rich texture: multi-scale noise, edges, text."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for scale, amp in ((64, 70), (16, 45), (4, 30), (1, 18)):
        n = rng.standard_normal((h // scale + 2, w // scale + 2, 3)).astype(np.float32)
        img += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    img += 128
    for _ in range(40):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, c, int(rng.integers(1, 3)), cv2.LINE_AA)
    for i in range(12):
        cv2.putText(img, "OpenFocus %d" % i,
                    (int(rng.integers(0, w - 200)), int(rng.integers(20, h))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.float32)


def make_depth(h, w):
    """Depth in [0, 1]: diagonal ramp + gaussian bump + an occluding disc."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = 0.6 * (xx / w) + 0.2 * (yy / h)
    d += 0.25 * np.exp(-(((xx - 0.3 * w) ** 2 + (yy - 0.6 * h) ** 2)
                         / (2 * (0.12 * w) ** 2)))
    disc = ((xx - 0.72 * w) ** 2 + (yy - 0.3 * h) ** 2) < (0.14 * w) ** 2
    d[disc] = 0.05  # near object in front of a far background: hard step
    return np.clip(d, 0, 1)


def make_depth_small(h, w):
    """Mostly-far scene with two small near objects: punishes any method that
    judges sharpness per *frame* instead of per pixel (a frame whose only
    sharp content is a small object must still win there)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.full((h, w), 0.9, np.float32)
    d[((xx - 0.25 * w) ** 2 + (yy - 0.3 * h) ** 2) < (0.07 * w) ** 2] = 0.0
    d[((xx - 0.75 * w) ** 2 + (yy - 0.65 * h) ** 2) < (0.05 * w) ** 2] = 0.45
    return d


SCENES = {"default": make_depth, "small": make_depth_small}


def make_stack(gt, depth, n_frames, max_sigma=6.0, noise=0.0, seed=1):
    """Frame i focuses at z_i; blur sigma grows linearly with |depth - z_i|.

    Per-pixel blur is approximated by a bank of discrete blur levels and a
    per-pixel linear interpolation between neighbouring levels.
    """
    rng = np.random.default_rng(seed)
    levels = np.linspace(0, max_sigma, 25)
    bank = [gt if s < 1e-3 else cv2.GaussianBlur(gt, (0, 0), s) for s in levels]
    bank = np.stack(bank)  # (L, H, W, 3)
    frames = []
    zs = np.linspace(0, 1, n_frames)
    for z in zs:
        sigma = np.clip(np.abs(depth - z) * max_sigma * 2.0, 0, max_sigma)
        pos = sigma / max_sigma * (len(levels) - 1)
        lo = np.floor(pos).astype(np.int32)
        hi = np.minimum(lo + 1, len(levels) - 1)
        t = (pos - lo)[..., None]
        yy, xx = np.indices(depth.shape)
        f = bank[lo, yy, xx] * (1 - t) + bank[hi, yy, xx] * t
        if noise > 0:
            f = f + rng.normal(0, noise, f.shape).astype(np.float32)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    s_aa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    s_bb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    s_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return float(m.mean())


def score(fused, gt, border=16):
    h, w = gt.shape[:2]
    if fused.shape[:2] != gt.shape[:2]:  # dct crops to its block grid
        fh, fw = fused.shape[:2]
        gt = gt[:fh, :fw]
        h, w = fh, fw
    f = fused[border:h - border, border:w - border].astype(np.float32)
    g = gt[border:h - border, border:w - border].astype(np.float32)
    err = np.abs(f - g).max(axis=2)
    return psnr(f, g), ssim(f, g), float((err > 12).mean() * 100)


def run(methods, n_frames=16, size=(768, 1024), noise=0.0, kernel=None,
        scene="default"):
    from core.multi_focus_fusion import MultiFocusFusion
    h, w = size
    gt = make_texture(h, w)
    depth = SCENES[scene](h, w)
    frames = make_stack(gt, depth, n_frames, noise=noise)
    gt8 = np.clip(gt, 0, 255).astype(np.uint8)
    # Oracle: the best any per-pixel frame pick can do (true depth known).
    zs = np.linspace(0, 1, n_frames)
    pick = np.abs(depth[None] - zs[:, None, None]).argmin(0)
    yy, xx = np.indices(depth.shape)
    oracle = np.stack(frames)[pick, yy, xx]
    rows = [("(oracle pick)",) + score(oracle, gt8) + (0.0, "")]
    for m in methods:
        try:
            fuser = MultiFocusFusion(algorithm=m, use_gpu=False)
        except Exception as exc:  # missing optional dependency
            rows.append((m, None, None, None, None, str(exc)[:60]))
            continue
        kw = {}
        if kernel is not None:
            kw["kernel_size"] = kernel
        t0 = time.time()
        try:
            out = fuser.fuse([f.copy() for f in frames], **kw)
        except Exception as exc:
            rows.append((m, None, None, None, None, str(exc)[:60]))
            continue
        dt = time.time() - t0
        p, s, g = score(out, gt8)
        rows.append((m, p, s, g, dt, ""))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-m", "--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--kernel", type=int, default=None)
    ap.add_argument("--scene", choices=sorted(SCENES), default="default")
    args = ap.parse_args(argv)
    rows = run(args.methods, args.frames, noise=args.noise, kernel=args.kernel,
               scene=args.scene)
    print("%-14s %8s %7s %8s %7s" % ("method", "PSNR", "SSIM", "ghost%", "time"))
    for m, p, s, g, dt, err in rows:
        if p is None:
            print("%-14s  skipped: %s" % (m, err))
        else:
            print("%-14s %8.2f %7.4f %8.2f %6.1fs" % (m, p, s, g, dt))


if __name__ == "__main__":
    main()
