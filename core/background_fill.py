"""Automatic background replacement for a fused focus stack.

A focus stack sharpens the subject and the background alike: the stacked
background is a patchwork of the frames' partial blurs, often with halos
along the subject's edges.  This module swaps that background for one real
frame (the blurriest background for the "soft" preset, the far-focus frame
for "sharp"), keeps everything the stack shows of the subject, and records
where no frame saw the scene cleanly.

Principles:

* Never add what was not there.  The new background is a frame the camera
  recorded; next to the subject, where every frame is darkened by partial
  occlusion or carries the subject's blurred light, the stacked result's
  own brightness is kept and only its grain comes from the frame.
* Preserve the subject.  Anything a frame shows in focus and the stack
  kept is subject, even where the focus measures miss it (a smooth cap).
* Clean (default) removes only edge flaws that follow the outline and that
  a halo can physically cause (darkening, or a shift toward the subject's
  own colour), and never leaves the range the frames recorded at that pixel.
  Faithful (``clean=False``) keeps the stack's own edge look.
* The missing-data maps say where the result is not backed by any frame:
  a retouching aid, and the only place a future generative fill may go.

The module is UI-free.  `replace_background` takes the render's aligned
frames and its fused result (any project dtype) and returns a
`BackgroundFillResult`; it reports progress and honours cancellation
through optional callbacks.  Peak memory is about 3.5 GB for 15 frames of
20 MP, independent of how the frames are stored.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

PRESETS = ("auto", "soft", "sharp")
MAX_SUBJECT_DEFAULT = 0.90
# "auto" keeps the soft background unless more than this share of the
# background is detail some frame recorded in focus (scene, not bokeh).
# Measured: 0-7% where soft looked best (Panaeolus, Dendrocollybia,
# Marthamyces), 24% on a pine-needle floor where soft smeared a log the
# stack had recorded sharply; ground-level scenes are 70%+.
AUTO_SHARP_SHARE = 0.15


class BackgroundFillCancelled(Exception):
    """Raised when the caller's `cancelled()` callback returns True."""


class BackgroundFillError(ValueError):
    """The inputs cannot be used (too few frames, mismatched sizes...)."""


@dataclass
class MissingData:
    """Where the result is not backed by a clean observation, uint8 0-255
    at full result resolution."""

    background: np.ndarray          # background no frame saw cleanly
    hidden: np.ndarray              # subject hidden from the camera
    out_of_range: np.ndarray        # subject beyond the stack's focus range
    featureless: np.ndarray         # too smooth/dark to judge sharpness

    def layers(self):
        """(name, map, BGR colour) in drawing order."""
        return (("background", self.background, (255, 0, 255)),
                ("hidden", self.hidden, (255, 255, 0)),
                ("out_of_range", self.out_of_range, (0, 220, 255)),
                ("featureless", self.featureless, (200, 140, 90)))


@dataclass
class BackgroundFillResult:
    image: np.ndarray               # same shape and dtype as the fused input
    changed: bool                   # False: left as stacked (ground-level scene)
    background_frame: int
    far_end: int
    subject_fraction: float
    missing: MissingData
    preset_used: str = "soft"       # "soft", "sharp" or "forced" (what "auto" resolved to)
    recorded_share: float = 0.0     # share of the background recorded in focus by some frame
    stats: dict = field(default_factory=dict)


def _to16f(img: np.ndarray) -> np.ndarray:
    """Any project image -> float32 BGR on a 0-65535 scale.

    8-bit is x257 (as uint16 would store it); float images in this project
    are 0-255 scaled."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype == np.uint16:
        return img.astype(np.float32)
    if img.dtype == np.uint8:
        return img.astype(np.float32) * 257.0
    return np.asarray(img, np.float32) * 257.0


def _from16f(img16: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Back to the fused input's dtype (and channel count)."""
    if like.dtype == np.uint16:
        out = np.clip(img16, 0, 65535).astype(np.uint16)
    elif like.dtype == np.uint8:
        out = np.clip(np.round(img16 / 257.0), 0, 255).astype(np.uint8)
    else:
        out = (img16 / 257.0).astype(like.dtype)
    if like.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    elif like.shape[2] == 4:
        out = np.dstack([out, like[..., 3]])
    return out


# ---------------------------------------------------------------------- helpers
def focus_measures(G, sigma=6.0):
    """Per-pixel focus measures from smoothed Laplacian energy E across the stack.

    ratio = log10(max E / median E): how much the best frame stands out.
            Clean for thick subjects; fails where a thin subject is fairly
            sharp in most frames (median is high too).
    peak  = log10(max E): is there real structure in *any* frame, compared
            with the empty-background noise floor.
    """
    E = np.empty_like(G)
    for i in range(len(G)):
        L = cv2.Laplacian(G[i], cv2.CV_32F, ksize=3)
        E[i] = cv2.GaussianBlur(L * L, (0, 0), sigma)
    eps = 1e-10
    peak = E.max(0)
    ratio = np.log10((peak + eps) / (np.median(E, 0) + eps))
    # uint8 is enough for ordinary stacks, while uint16 preserves frame 256+
    # instead of wrapping it back to zero.
    index_dtype = np.uint8 if len(G) <= 256 else np.uint16
    return {"ratio": ratio, "peak": np.log10(peak + eps)}, E.argmax(0).astype(index_dtype)


def otsu_threshold(m):
    lo, hi = np.percentile(m, [0.5, 99.5])
    u8 = (np.clip((m - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    t8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return lo + t8 / 255.0 * (hi - lo)


def subject_mask(measures, use, thresh=None, min_island=2e-5):
    """Union of the chosen measures' Otsu masks, cleaned.

    min_island is a fraction of the image; the default only drops noise-sized
    specks so small real objects (spores, debris) stay with the subject."""
    h, w = next(iter(measures.values())).shape
    sub = np.zeros((h, w), np.uint8)
    used = {}
    for name in use:
        t = thresh if (thresh is not None and len(use) == 1) else otsu_threshold(measures[name])
        sub |= (measures[name] > t).astype(np.uint8)
        used[name] = t
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, k)
    sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_island * h * w
    sub = keep[lab].astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - sub, 4)
    for i in range(1, n):                      # enclosed holes belong to the subject
        x, y, ww, hh, _ = st[i]
        if not (x == 0 or y == 0 or x + ww == w or y + hh == h):
            sub[lab == i] = 1
    return sub.astype(bool), used


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter_color(I, p, r, eps):
    """He et al. guided filter with a 3-channel guide I (float32 0-1)."""
    mI = [_box(I[..., c], r) for c in range(3)]
    mp = _box(p, r)
    cov_Ip = [_box(I[..., c] * p, r) - mI[c] * mp for c in range(3)]
    var = np.empty(I.shape[:2] + (3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            v = _box(I[..., i] * I[..., j], r) - mI[i] * mI[j]
            if i == j:
                v = v + eps
            var[..., i, j] = var[..., j, i] = v
    a = np.linalg.solve(var, np.stack(cov_Ip, -1)[..., None])[..., 0]   # H,W,3
    b = mp - (a * np.stack(mI, -1)).sum(-1)
    ma = np.stack([_box(a[..., c], r) for c in range(3)], -1)
    return (ma * I).sum(-1) + _box(b, r)


def fine_outline(sub_half, G, color=None, band=20, sigma=1.5, sliver=3, bg_ring=(3, 15), step=2.5 / 255, wall_k=5):
    """Re-decide the outer `band` half-res px of the subject at fine scale.

    The coarse measures are blurred (sigma 6) to beat noise, which pushes the
    outline ~25-30 full-res px out into the background; that strip then keeps
    the fused pixels and reads as a halo.  Peak Laplacian energy at sigma 1.5
    follows the real edge.  Deep inside the coarse mask stays subject; gaps
    the fine mask leaves on smooth surfaces are filled when enclosed."""
    peak = None
    for g in G:
        L = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        e = cv2.GaussianBlur(L * L, (0, 0), sigma)
        peak = e if peak is None else np.maximum(peak, e)
    peak = np.log10(peak + 1e-10)
    fine = (peak > otsu_threshold(peak)).astype(np.uint8)
    fine = cv2.morphologyEx(fine, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    d_in = cv2.distanceTransform(sub_half.astype(np.uint8), cv2.DIST_L2, 5)
    # Candidate pixels to peel off: inside the coarse mask, near its outline,
    # without fine texture.  Only peel the overshoot band around the outside:
    # thin slivers (a peg's smooth core, the seam where two pegs cross) and
    # pieces not connected to the real background stay subject; peeling them
    # cut dark slits into the pegs.
    peel = (sub_half & (fine == 0) & (d_in <= band)).astype(np.uint8)
    if color is not None:
        # Smooth subject surface near the edge (a cap's untextured rim zone)
        # also lacks fine texture; only peel what looks like the background
        # just outside rather than the subject just inside.
        d_out = cv2.distanceTransform((~sub_half).astype(np.uint8), cv2.DIST_L2, 5)
        def local_mean(region, sigma=12):
            reg = region.astype(np.float32)
            est = np.zeros_like(color)
            done = np.zeros(region.shape, bool)
            for sg in (sigma, 3 * sigma, 9 * sigma):
                num = cv2.GaussianBlur(color * reg[..., None], (0, 0), sg)
                den = cv2.GaussianBlur(reg, (0, 0), sg)
                ok = ~done & (den > 0.02)
                est[ok] = num[ok] / den[ok][..., None]
                done |= ok
            return est
        bg_col = local_mean((~sub_half) & (d_out > bg_ring[0]) & (d_out < bg_ring[1]))
        sub_col = local_mean(sub_half & (d_in > band))
        c = cv2.GaussianBlur(color, (0, 0), 1.0)
        bg_like = (np.linalg.norm(c - bg_col, axis=2) < np.linalg.norm(c - sub_col, axis=2))
        # Right beside an edge the background is darkened by partial
        # occlusion, which can make it closer in RGB to a dark subject than
        # to the lighter background further out (the bulge beside the
        # Panaeolus left stem, y 3590-3750).  Darkening scales brightness and
        # keeps the hue, so also count a pixel as background when its
        # chromaticity matches the background's.
        chrom = lambda x: x / np.maximum(x.sum(axis=2, keepdims=True), 1e-4)
        cc, cb, cs_ = chrom(c), chrom(bg_col), chrom(sub_col)
        bg_like |= np.linalg.norm(cc - cb, axis=2) < np.linalg.norm(cc - cs_, axis=2)
        # Light from a blurred dark stem mixes into the background beside it,
        # so that strip sits between the two colours and no colour test can
        # place it (the notch beside the Panaeolus left stem, y 3630-3720).
        # Its shape can: brightness ramps smoothly from the background into
        # it and jumps only at the textured edge, while a real smooth subject
        # surface (a cap rim) ends in a silhouette step.  Also peel what the
        # outside reaches without crossing a step.
        # Largest step in any channel: the brown cap and the green blade
        # behind it are equally bright, so a brightness step test bit the cap.
        gm = np.max(np.hypot(cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3),
                             cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)), axis=2) / 8.0
        # Thicken the steps so a short weak stretch of a silhouette (where
        # the cap happens to match the background) cannot let the flood in.
        wall = cv2.dilate((gm >= step).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wall_k, wall_k))) > 0
        passable = (peel > 0) & ~wall
        outside0 = cv2.dilate((~sub_half).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        n_, lab_, _, _ = cv2.connectedComponentsWithStats(passable.astype(np.uint8), 8)
        hit = np.zeros(n_, bool)
        hit[np.unique(lab_[outside0 & passable])] = True
        hit[0] = False
        bg_like |= hit[lab_]
        peel &= bg_like.astype(np.uint8)
    peel = cv2.morphologyEx(peel, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sliver + 1, 2 * sliver + 1)))
    outside = cv2.dilate((~sub_half).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(peel, 8)
    touches = np.zeros(n, bool)
    touches[np.unique(lab[outside & (peel > 0)])] = True
    touches[0] = False
    sub = (sub_half & ~touches[lab]).astype(np.uint8)
    h, w = sub.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - sub, 4)
    for i in range(1, n):                     # enclosed gaps belong to the subject
        x, y, ww, hh, _ = st[i]
        if not (x == 0 or y == 0 or x + ww == w or y + hh == h):
            sub[lab == i] = 1
    return sub.astype(bool)


def protect_missed_subject(sub_half, G, k_far, min_area, near=6, sigma=4.0):
    """Add back subject the focus measures missed.

    Smooth, untextured parts (bare bristle segments, a stalk that swayed
    during the burst) are never *sharp* in any frame, so the focus measures
    call them background and the far frame would overwrite them.  But some
    frame still shows their edges far more clearly than the far frame does,
    whereas real background always has its most detail in the far frame.
    So R = log10(max_k E_k / E_far) is ~0 on background and high on anything
    in front of it.  Frames only, so fusion seams in the result cannot
    trigger it.  Pieces the focus mask does not already cover (within
    `near` half-res px) are added back."""
    E = []
    for g in G:
        L = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F, ksize=3)
        E.append(cv2.GaussianBlur(L * L, (0, 0), sigma))
    E = np.stack(E)
    R = np.log10((E.max(0) + 1e-9) / (E[k_far] + 1e-9))
    d = cv2.distanceTransform((~sub_half).astype(np.uint8), cv2.DIST_L2, 5)
    bg_noise = np.percentile(R[d > 30], 99.9) if (d > 30).any() else 0.8
    t = min(otsu_threshold(R), max(0.8, bg_noise + 0.05))
    covered = cv2.dilate(sub_half.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * near + 1, 2 * near + 1)))
    missed = ((R > t) & (covered == 0)).astype(np.uint8)
    missed = cv2.morphologyEx(missed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(missed, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    missed = keep[lab].astype(np.uint8)
    missed = cv2.dilate(missed, np.ones((5, 5), np.uint8))   # join to the subject it touches
    return sub_half | (missed > 0), t, int(keep[1:].sum()), float(missed.mean() * 100)


def refine_subject(sub_half, fused, band=24, r=12, eps=2e-4):
    """Snap the coarse focus mask to the fused image's colour edges.

    Works at half resolution: only a +-band strip around the coarse boundary
    is allowed to change.  Returns a full-res float subject matte 0-1."""
    H, W = fused.shape[:2]
    h, w = sub_half.shape
    guide = cv2.resize(fused, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 65535.0
    p = sub_half.astype(np.float32)
    q = np.clip(guided_filter_color(guide, p, r, eps), 0, 1)
    d_in = cv2.distanceTransform(sub_half.astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~sub_half).astype(np.uint8), cv2.DIST_L2, 5)
    unknown = (d_in < band) & (d_out < band)
    matte = np.where(unknown, q, p).astype(np.float32)
    return cv2.resize(matte, (W, H), interpolation=cv2.INTER_LINEAR)


def edge_frame_map(sub_half, idx_half, inner=(3, 20)):
    """For every pixel, the in-focus frame of the nearest subject edge.

    Reads the argmax index just inside the subject (between inner[0] and
    inner[1] half-res px from its outline, where focus is reliable) after a
    median filter, then propagates it outward to the nearest such pixel."""
    d_in = cv2.distanceTransform(sub_half.astype(np.uint8), cv2.DIST_L2, 5)
    ring = sub_half & (d_in >= inner[0]) & (d_in <= inner[1])
    if not ring.any():
        ring = sub_half
    if not ring.any():
        raise BackgroundFillError("no subject found to replace the background around")
    idx = cv2.medianBlur(idx_half, 5)
    src = np.where(ring, 0, 1).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    vals = idx[ring]                       # raster order == label order
    return vals[labels - 1]


def fill_halo_zone(bg, dist, sub, width, scale=4):
    """Rebuild the background within `width` px of the subject.

    Every single frame darkens (or tints) the out-of-focus background right
    next to a sharp edge: the foreground object blocks part of the lens's view
    of what is behind it.  No frame has a clean background there, so instead
    extend the smooth background from beyond the zone inward (normalized
    Gaussian convolution, widening until every pixel is reached), then add back
    the frame's own fine grain, clamped to its noise level so edge structure
    cannot ride along."""
    H, W = dist.shape
    h, w = H // scale, W // scale
    bs = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
    ds = cv2.resize(dist, (w, h), interpolation=cv2.INTER_NEAREST)
    width = np.asarray(width, np.float32)
    ws = cv2.resize(width, (w, h), interpolation=cv2.INTER_LINEAR) if width.ndim else width
    lum_s = bs.mean(axis=2)
    usable = lum_s > 0.35 * np.median(lum_s[ds > ws]) if (ds > ws).any() else np.ones_like(lum_s, bool)
    # Not the field stop, nor its blurred rim: the rim is above the darkness
    # cutoff but still far darker/bluer than the glass, and filled in from
    # there it drew light lines across the Marthamyces corner.
    if (~usable).any():
        rim = cv2.distanceTransform(usable.astype(np.uint8), cv2.DIST_L2, 5) < max(float(np.median(ws)), 8.0)
        usable &= ~rim
    known = ((ds > ws) & usable).astype(np.float32)   # background beyond the zone
    r = float(np.median(width)) / scale if width.ndim else float(width) / scale
    est = normconv(bs, known, (r / 2, r, 2 * r, 4 * r, 8 * r, 16 * r))
    est = cv2.resize(est, (W, H), interpolation=cv2.INTER_LINEAR)

    grain = bg - cv2.GaussianBlur(bg, (0, 0), 1.5)
    far = dist > 2 * width
    if not far.any():
        far = dist > np.percentile(dist, 75)
    lim = 3 * 1.4826 * np.median(np.abs(grain[far]), axis=0)        # per channel
    grain = np.clip(grain, -lim, lim)

    t = np.clip((dist - 0.6 * width) / (0.4 * width), 0, 1).astype(np.float32)
    keep = (t * t * (3 - 2 * t))[..., None]                           # 0 near edge -> 1 at width
    return bg * keep + (est + grain) * (1 - keep), est


def edge_width_map(fz, est, sub, dist, lo=6.0, hi=48.0):
    """Per-pixel blend width matching how soft the nearest subject edge is.

    A blurred edge ramps from subject to background over a width of about
    contrast / steepest slope.  Soft pegs fade over 10-15 px; cutting the
    blend at a fixed 6 px ended that fade abruptly against the rebuilt
    background and drew a light outline.  Sharp edges keep the minimum."""
    lum = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) / 257.0
    lf, le = lum(fz), lum(est)
    g = cv2.GaussianBlur(lf, (0, 0), 1.0)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    d_in = cv2.distanceTransform(sub.astype(np.uint8), cv2.DIST_L2, 5)
    inner = (sub & (d_in > 3) & (d_in < 15)).astype(np.float32)
    sub_lvl = cv2.GaussianBlur(lf * inner, (0, 0), 8) / np.maximum(cv2.GaussianBlur(inner, (0, 0), 8), 1e-3)
    band = (dist < 12) & ((~sub) | (d_in < 6))                  # straddles the outline
    slope = cv2.dilate(np.where(band, grad, 0).astype(np.float32), np.ones((13, 13), np.uint8))
    contrast = np.abs(sub_lvl - le)
    width = np.clip(contrast / np.maximum(slope, 0.05), lo, hi)
    ring = sub & (d_in <= 2)                                    # read the width at the outline
    src = np.where(ring, 0, 1).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    vals = width[ring]
    wmap = vals[labels - 1].astype(np.float32)
    return cv2.GaussianBlur(wmap, (0, 0), 6)


def occlusion_width(G, sub_half, k_edge):
    """How far (full-res px) the background stays darkened/tinted beside an
    edge in the frame where that edge is sharp: the edge blocks part of the
    lens's view of the out-of-focus background.  Weighted median over edge
    depths; also returns the per-depth table."""
    d = cv2.distanceTransform((~sub_half).astype(np.uint8), cv2.DIST_L2, 5) * 2
    edges = [0, 8, 16, 24, 32, 48, 64, 96, 128, 160, 320]
    widths, weights, table = [], [], {}
    for k in np.unique(k_edge[~sub_half]):
        sel = (~sub_half) & (k_edge == k)
        if sel.sum() < 3000:
            continue
        lp = cv2.GaussianBlur(G[k], (0, 0), 2)
        prof = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = sel & (d > lo) & (d <= hi)
            prof.append(float(lp[m].mean()) if m.sum() > 200 else np.nan)
        prof = np.array(prof)
        base = prof[-1]
        if not (np.isfinite(base) and np.isfinite(prof[0])):
            continue
        dev = np.abs(prof - base)
        if dev[0] < 0.03 * base:                 # no measurable effect
            continue
        w = edges[1]
        for hi, v in zip(edges[1:-1], dev[:-1]):
            if not np.isfinite(v) or v <= 0.25 * dev[0]:
                break
            w = hi
        table[int(k)] = w
        widths.append(w)
        weights.append(int(sel.sum()))
    if not widths:
        return 32.0, table
    o = np.argsort(widths)
    cw = np.cumsum(np.array(weights)[o])
    return float(np.array(widths)[o][np.searchsorted(cw, cw[-1] / 2)]), table


def missing_data_map(sub, dist, k_edge_full, peak_half, peak_t, N, k_far, c_raw, w_occ, k_occ, size_cap=None):
    """Where no frame shows the scene cleanly (0 = observed, 1 = missing).

    Background: an edge sharp in frame k_e blurs over c*|k - k_e| px in frame
    k (its glow), and in frame k the background's own blur is cut off by the
    edge over c*|k - k_B|/2 px (the darkening), with k_B the background's
    focus position, fitted so the darkening measured in the edge's sharp frame
    (w_occ at k_occ) matches.  A background pixel d px from the edge is clean
    in frame k only beyond both; the best frame gives the clean distance W.
    Subject: pixels never sharp in any frame (peak focus below threshold)."""
    side = 1 if k_far >= N // 2 else -1
    k_B = k_occ + side * 2.0 * w_occ / max(c_raw, 1e-3)
    ks = np.arange(N, dtype=np.float32)
    W_by_ke = {}
    for ke in range(N):
        w = np.maximum(c_raw * np.abs(ks - ke), c_raw * np.abs(ks - k_B) / 2.0)
        W_by_ke[ke] = float(w.min())
    width_lookup = np.fromiter((W_by_ke[ke] for ke in range(N)), np.float32, count=N)
    W = width_lookup[k_edge_full]
    if size_cap is not None:
        W = np.minimum(W, size_cap)
    within = np.clip(1.0 - dist / np.maximum(W, 1.0), 0, 1)
    # How enclosed the background is: the share of the surrounding disc that
    # is subject.  ~0.5 beside a straight edge, more in a concave corner
    # (cap meets stem), most in a narrow gap; that is where the background
    # is hidden from the most directions of the lens aperture.
    sg = max(float(np.median(W[~sub])) if (~sub).any() else 32.0, 8.0) / 2
    scale = 4
    sm = cv2.resize(sub.astype(np.float32), (sub.shape[1] // scale, sub.shape[0] // scale), interpolation=cv2.INTER_AREA)
    blocked = cv2.resize(blur(sm, sg / scale), (sub.shape[1], sub.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    enclosed = np.clip((blocked - 0.2) / 0.5, 0, 1)
    bg_missing = np.clip(within * (0.35 + 0.65 * enclosed) + 0.5 * enclosed * (dist < W), 0, 1) * (~sub)
    H, Wd = sub.shape
    soft = np.clip((peak_t - peak_half) / 0.5, 0, 1).astype(np.float32)
    soft = cv2.resize(soft, (Wd, H), interpolation=cv2.INTER_LINEAR) * sub
    return bg_missing.astype(np.float32), soft.astype(np.float32), W_by_ke, k_B, W


def natural_near_background(bgf, fz, sub, dist, reach, sigma=6.0, scale=4):
    """Near the subject keep the fused result's own brightness and colour
    (what the frames actually recorded there, including the natural
    darkening beside sharp edges) and take only fine grain from the
    background frame; hand over to the background frame gradually across
    `reach` px so no ring forms.  Replacing that zone with a smooth
    extrapolation looked cleaner on paper but read as a glow (a bright green
    band along the Panaeolus cap) and lost real detail (the dark gap between
    the caps, which every sharp frame shows)."""
    H, W = dist.shape
    h, w = H // scale, W // scale
    bgmask = ((~sub) & (dist > 2)).astype(np.float32)
    ms = cv2.resize(bgmask, (w, h), interpolation=cv2.INTER_AREA)
    sg = sigma / scale
    def lp_masked(img):
        im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        out = normconv(im, ms, (sg, 3 * sg, 9 * sg, 27 * sg))
        return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
    shift = lp_masked(fz) - lp_masked(bgf)
    t = np.clip(dist / np.maximum(reach, 1.0), 0, 1).astype(np.float32)
    wnear = 1.0 - t * t * (3 - 2 * t)
    return bgf + shift * wnear[..., None]


def blur(x, sigma):
    """Gaussian blur that stays cheap and smooth at large sigma.

    cv2.GaussianBlur with sigma in the thousands builds a kernel and a padded
    copy of the image to match (over 20 GB; systemd-oomd killed the
    terminal).  Shrinking hard and enlarging bilinearly instead left planar
    facets with creases, which showed as hard-edged polygons once two such
    blurs were divided.  So go through a Gaussian pyramid: pyrDown
    until the remaining sigma is small, blur there, pyrUp back (each pyrUp
    is a smooth Gaussian interpolation), then match the exact size."""
    h, w = x.shape[:2]
    sigma = float(min(sigma, 2 * max(h, w)))
    if sigma <= 12:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    levels = 0
    cur = x
    shapes = []
    # Each pyrDown applies ~sigma 1 (at its own scale) and halves the size.
    while sigma / (2 ** levels) > 6 and min(cur.shape[:2]) > 16:
        shapes.append(cur.shape[:2])
        cur = cv2.pyrDown(cur)
        levels += 1
    rem2 = (sigma / (2 ** levels)) ** 2 - sum(1.0 / (4 ** i) for i in range(1, levels + 1))
    cur = cv2.GaussianBlur(cur, (0, 0), float(np.sqrt(max(rem2, 0.25))))
    for hh, ww in reversed(shapes):
        cur = cv2.pyrUp(cur, dstsize=(ww, hh))
    return cur


def normconv(img, mask, sigmas, eps=0.03):
    """Seamless multi-scale normalized convolution.

    Every scale contributes, weighted eps**i, so the finest scale dominates
    where it has data and coarser ones take over smoothly where it has none.
    The earlier "first scale that reaches a pixel wins" fill left hard seams
    between scales: discs and wedges across the Marthamyces glass."""
    m = mask.astype(np.float32)
    num = np.zeros_like(img, dtype=np.float32)
    den = np.zeros(m.shape, np.float32)
    for i, s in enumerate(sigmas):
        wgt = eps ** i
        num += wgt * blur(img * m[..., None], s)
        den += wgt * blur(m, s)
    return num / np.maximum(den, 1e-12)[..., None]


def protect_visible_detail(sub_half, G, k_bg, fused_half=None, factor=4.0, ratio=3.0, spread=6,
                           keep_share=0.25, frame_medians=None):
    """Never replace what the stacked image clearly shows.

    Fine detail well above the noise floor in the stacked image that the
    background frame lacks is real, in-focus content.  True background
    (bokeh) has almost none, so it stays replaceable; a smooth cap or broad
    straws missed by the focus measures in a ground-level scene (Psilocybe)
    are kept instead of being swapped for a blurred version."""
    def energy(g):
        L = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.8), cv2.CV_32F, ksize=3)
        return cv2.GaussianBlur(L * L, (0, 0), 3)
    # Measured on the frames, not the stacked result: fusion seams (OpenFocus
    # A's patchwork) are detail that exists in no frame, and counting them
    # protected 84% of the Dendrocollybia A image.
    medians = frame_medians
    if medians is None:
        medians = [np.median(g) for g in G]
    eb = energy(G[k_bg])
    ef = None
    for k in range(len(G)):
        if k == k_bg:
            continue
        g = G[k] * (medians[k_bg] / max(float(medians[k]), 1e-6))
        e = energy(g)
        ef = e if ef is None else np.maximum(ef, e)
    if ef is None:
        return sub_half, 0.0
    noise = np.percentile(ef, 1)
    keep = (ef > factor * noise) & (ef > ratio * eb)
    if fused_half is not None:
        # ...and the stacked image must have kept a fair share of it.  Dust
        # sharp in a single frame that the stack did not keep was protected as blobs of
        # stacked pixels on the empty Marthamyces glass (grey blotches, and
        # light lines along the midlines between them).  A share, not a
        # noise-floor test: a smooth cap's stacked surface is quieter than
        # any frame (the Psilocybe left cap was replaced again).  Seams in
        # the stack still do not count, since the frames must agree.
        e_f = energy(fused_half)
        keep &= e_f > keep_share * ef
    keep = keep.astype(np.uint8)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    # The energy is blurred (sigma 3) and closed, so around every sharp
    # subject edge it spreads ~6 half-res px into the background.  Unioned
    # as is, that ring undid the fine outline: a 10-12 px strip of the
    # stacked image's darker near-edge background stayed "subject" and read
    # as a dark halo beside the Panaeolus stem.  Only add pieces
    # that reach beyond that ring; within it, add just what connects such a
    # piece to the subject.
    ring = spread + 2
    d_out = cv2.distanceTransform((~sub_half).astype(np.uint8), cv2.DIST_L2, 5)
    far = (keep & (d_out > ring)).astype(np.uint8)
    far = cv2.morphologyEx(far, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    bridge = cv2.dilate(far, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * ring + 1, 4 * ring + 1))) > 0
    keep = keep & ((far > 0) | bridge)
    sub = (sub_half | keep).astype(np.uint8)
    h, w = sub.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - sub, 4)
    for i in range(1, n):                     # enclosed gaps belong to the subject
        x, y, ww, hh, _ = st[i]
        if not (x == 0 or y == 0 or x + ww == w or y + hh == h):
            sub[lab == i] = 1
    return sub.astype(bool), float(keep.mean())


def protect_focused_objects(sub_half, G, k_bg, fused_half, ratio=6.0, factor=4.0, keep_share=0.25,
                            ring=22, min_area=2e-4, frame_medians=None):
    """Keep soft objects that some frame focused on and the stack kept.

    A peg's knob, a leaf or a twig just outside the depth of field is never
    sharp enough for the fine-detail test, yet the stack shows it far more
    clearly than the (blurriest) background frame, which turns it into a
    smear: half a knob replaced, the end of a peg blurred.  Mid-scale
    structure (band-pass sigma 1.5-4 half-res px) in its best frame many
    times above the background frame's, and kept by the stack, marks such an
    object; true bokeh and the subject's own glow stay within ~1.6x.

    Only pieces that reach more than `ring` px beyond the subject and are
    attached to it count, plus the strip joining them to it: the defocused
    edges of every peg also carry mid-scale structure, and pieces floating
    free are background, which the soft background is meant to soften.
    Whether an attached piece is subject or background nearly in focus is a
    matter of taste (kept moss behind debris looked worse on one stack,
    blurred peg knobs looked wrong on another), so callers make it an option."""
    medians = frame_medians
    if medians is None:
        medians = [np.median(g) for g in G]

    def focused(s1, s2, smooth):
        def band(g):
            d = cv2.GaussianBlur(g, (0, 0), s1) - cv2.GaussianBlur(g, (0, 0), s2)
            return cv2.GaussianBlur(d * d, (0, 0), smooth)
        ref = float(medians[k_bg])
        mb = band(G[k_bg])
        mx = None
        for k in range(len(G)):
            if k == k_bg:
                continue
            m = band(G[k] * (ref / max(float(medians[k]), 1e-6)))
            mx = m if mx is None else np.maximum(mx, m)
        if mx is None:
            return np.zeros(sub_half.shape, bool)
        return (mx > ratio * mb) & (mx > factor * np.percentile(mx, 1)) & (band(fused_half) > keep_share * mx)

    keep = focused(1.5, 4.0, 4)
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    d_out = cv2.distanceTransform((~sub_half).astype(np.uint8), cv2.DIST_L2, 5)
    far = (keep & (d_out > ring)).astype(np.uint8)
    far = cv2.morphologyEx(far, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(far, 8)
    ok = np.zeros(n, bool)
    ok[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area * far.size
    near_subject = cv2.dilate(sub_half.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * ring + 1, 4 * ring + 1))) > 0
    attached = np.zeros(n, bool)
    attached[np.unique(lab[near_subject & (far > 0)])] = True
    far = (ok & attached)[lab].astype(np.uint8)
    bridge = cv2.dilate(far, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * ring + 1, 4 * ring + 1))) > 0
    sub = (sub_half | (keep & ((far > 0) | bridge))).astype(np.uint8)
    h, w = sub.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - sub, 4)
    for i in range(1, n):                     # enclosed gaps belong to the subject
        x, y, ww, hh, _ = st[i]
        if not (x == 0 or y == 0 or x + ww == w or y + hh == h):
            sub[lab == i] = 1
    return sub.astype(bool)


def object_size_cap(sub_half, shape, factor=6.0, floor=32.0):
    """Per-pixel cap (full-res px) on how far an object's glow/occlusion can
    matter: ~factor x the half-thickness of the nearest object.  A debris
    speck's defocus glow spreads wide but is negligibly faint; sizing its
    zone by the large-object glow radius left almost no usable background on
    the Marthamyces slide."""
    H, W = shape
    sub8 = sub_half.astype(np.uint8)
    d_in = cv2.distanceTransform(sub8, cv2.DIST_L2, 5) * 2          # full-res px
    n, lab = cv2.connectedComponents(sub8, connectivity=8)
    rad = np.zeros(n, np.float32)
    np.maximum.at(rad, lab.ravel(), d_in.ravel())
    _, near = cv2.distanceTransformWithLabels(1 - sub8, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    comp_of_subpix = lab[sub8 > 0]                                   # raster order == label order
    cap_half = np.maximum(factor * rad[comp_of_subpix[near - 1]], floor).astype(np.float32)
    # "Nearest object" switches along straight lines between objects; a small
    # fragment beside a big stem then cut hard-edged polygons into every
    # zone sized by this cap.  Smooth it, but only ever downward (local
    # minimum, then blur): letting large objects' caps spread onto nearby
    # specks brought back the Marthamyces blotches.
    cap_half = cv2.erode(cap_half, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49)))
    cap_half = blur(cap_half, 12.0)
    return cv2.resize(cap_half, (W, H), interpolation=cv2.INTER_LINEAR)


def lp_background(img, bgmask, sigma, scale=4):
    """Quarter-res low-pass of `img` over background pixels only (normalized
    convolution, widening until every pixel is reached)."""
    H, W = bgmask.shape
    h, w = H // scale, W // scale
    ms = cv2.resize(bgmask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    sg = sigma / scale
    out = normconv(im, ms, (sg, 3 * sg, 9 * sg, 27 * sg))
    return out


def recorded_range(count, load_matched, shape, sigma=6.0, scale=4,
                   valid_masks=None):
    """Per-pixel min and max (quarter res, low-passed, exposure-matched) over
    every frame: the span of what the camera actually recorded at each spot.
    Near an edge the frame where it is sharp records the darkened view and
    the defocused frames record the glow; the clean background lies between."""
    H, W = shape
    lo = hi = None
    seen = None
    for k in range(count):
        f = load_matched(k)[0]
        mask = None if valid_masks is None else valid_masks[k]
        if mask is None:
            q = cv2.GaussianBlur(
                cv2.resize(f, (W // scale, H // scale), interpolation=cv2.INTER_AREA),
                (0, 0), sigma / scale)
            usable = np.ones(q.shape[:2], bool)
        else:
            m = cv2.resize(mask.astype(np.float32), (W // scale, H // scale),
                           interpolation=cv2.INTER_AREA)
            numerator = cv2.GaussianBlur(
                cv2.resize(f * mask[..., None], (W // scale, H // scale),
                           interpolation=cv2.INTER_AREA),
                (0, 0), sigma / scale)
            denominator = cv2.GaussianBlur(m, (0, 0), sigma / scale)
            usable = denominator > 1e-3
            q = numerator / np.maximum(denominator, 1e-12)[..., None]
        if lo is None:
            lo = np.full_like(q, np.inf)
            hi = np.full_like(q, -np.inf)
            seen = np.zeros(q.shape[:2], bool)
        lo = np.where(usable[..., None], np.minimum(lo, q), lo)
        hi = np.where(usable[..., None], np.maximum(hi, q), hi)
        seen |= usable
        del f
    lo[~seen] = -np.inf
    hi[~seen] = np.inf
    return lo, hi


def contour_profile(D, sub_q, zone_q, seg=24, dbin=2):
    """The part of a deviation field that follows the subject outline.

    A stacking halo runs along an edge with the same profile for a long way;
    real background (a grass blade, the gap between two caps) crosses the
    edge instead.  For every background pixel in `zone_q`, find the nearest
    outline point; group by (stretch of outline, distance band) and take the
    median deviation; smooth along the outline.  Returns that profile, which
    is what gets removed.  Quarter-res arrays; seg and dbin in quarter px."""
    h, w = sub_q.shape
    cs, _ = cv2.findContours(sub_q.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    seg_of = np.full((h, w), -1, np.int64)
    next_seg = 0
    neighbours = {}
    for c in cs:
        pts = c[:, 0]
        if len(pts) < seg:
            continue
        ids = next_seg + np.arange(len(pts)) // seg
        n = int(ids.max() - next_seg + 1)
        for j in range(n):
            neighbours[next_seg + j] = [next_seg + (j + o) % n for o in (-1, 1)]
        seg_of[pts[:, 1], pts[:, 0]] = ids
        next_seg += n
    if next_seg == 0:
        return np.zeros_like(D)
    src = np.where(seg_of >= 0, 0, 1).astype(np.uint8)
    dq, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    seg_vals = seg_of[seg_of >= 0]                          # raster order == label order
    pix_seg = seg_vals[labels - 1]
    sel = zone_q & (~sub_q)
    nb = int(np.ceil(dq[sel].max() / dbin)) + 1 if sel.any() else 1
    key = pix_seg[sel] * nb + (dq[sel] / dbin).astype(np.int64)
    vals = D[sel]
    uk, inv = np.unique(key, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uk) + 1))
    med = {}
    for j, k_ in enumerate(uk):
        a_, b_ = bounds[j], bounds[j + 1]
        if b_ - a_ >= 4:
            med[int(k_)] = np.median(vals[order[a_:b_]], axis=0)
    # Profile relative to each stretch's outermost level: a halo is the part
    # of the deviation that differs from the background just beyond it.
    # (Fitting a trend line there instead invented halos on honestly curved
    # gradients; a monotone fit corrected nothing, since halos are shoulders.)
    # Distances: bins are dbin quarter px = 4*dbin full px.
    by_seg = {}
    for k_ in med:
        sg_, b = divmod(k_, nb)
        by_seg.setdefault(sg_, []).append(b)
    for sg_, bs in by_seg.items():
        bs = sorted(bs)
        vals_b = np.stack([med[sg_ * nb + b] for b in bs])            # (bins, channels)
        trend = np.broadcast_to(vals_b[-2:].mean(axis=0), vals_b.shape)
        resid = vals_b - trend
        for j, b in enumerate(bs):
            med[sg_ * nb + b] = resid[j]
    smooth = np.zeros((len(uk), D.shape[2]), np.float32)
    for j, k_ in enumerate(uk):               # average with the neighbouring stretches of outline
        sg_, b = divmod(int(k_), nb)
        acc, wsum = 0.0, 0.0
        for s2, wt in ((sg_, 2.0), *[(x, 1.0) for x in neighbours.get(sg_, [])]):
            v = med.get(s2 * nb + b)
            if v is not None:
                acc = acc + wt * v
                wsum += wt
        if wsum:
            smooth[j] = acc / wsum
    # Interpolate between distance bands (nearest-band lookup left a faint
    # line parallel to the edge at band boundaries).
    size = int(uk.max()) + 2 if len(uk) else 1
    table = np.zeros((size, D.shape[2]), np.float32)
    have = np.zeros(size, bool)
    table[uk] = smooth
    have[uk] = True
    pos = dq[sel] / dbin - 0.5
    b0 = np.clip(np.floor(pos).astype(np.int64), 0, nb - 1)
    frac = np.clip(pos - b0, 0, 1)[:, None]
    k0 = pix_seg[sel] * nb + b0
    k1 = pix_seg[sel] * nb + np.minimum(b0 + 1, nb - 1)
    k0 = np.clip(k0, 0, size - 1)
    k1 = np.clip(k1, 0, size - 1)
    v0 = table[k0]
    v1 = np.where(have[k1][:, None], table[k1], v0)
    prof = np.zeros_like(D)
    prof[sel] = v0 * (1 - frac) + v1 * frac
    # Smooth across changes of nearest outline stretch: in a concave corner
    # (stem meets cap) the nearest stretch flips along the corner's diagonal,
    # which drew a straight wedge.  Normalized over background pixels only.
    msk = sel.astype(np.float32)
    num = cv2.GaussianBlur(prof * msk[..., None], (0, 0), 3.0)
    den = cv2.GaussianBlur(msk, (0, 0), 3.0)
    return np.where(sel[..., None], num / np.maximum(den, 1e-3)[..., None], 0).astype(np.float32)


def halo_directions(prof, bg, img_q, sub_q, ring=6):
    """Keep only the part of an edge deviation a halo can physically cause.

    Beside an edge the background can be darkened (partial occlusion: every
    channel scaled down, direction -bg) or mixed with the subject's own
    blurred light (direction subject colour - bg).  Anything else is real
    background that happens to run along the outline, such as the green
    blade behind the left Panaeolus cap, which brightens toward the cap and
    was flattened into a bright line along the rim.  Per pixel,
    decompose the deviation onto those two directions with non-negative
    weights and return just that part.  Quarter-res arrays."""
    d_in = cv2.distanceTransform(sub_q.astype(np.uint8), cv2.DIST_L2, 5)
    inner = sub_q & (d_in <= ring)
    if not inner.any():
        return prof
    subc = normconv(img_q.astype(np.float32), inner, (3, 9, 27, 81))
    u1 = subc - bg                                   # toward the subject colour
    u2 = -bg                                         # darkening
    a11 = (u1 * u1).sum(2); a22 = (u2 * u2).sum(2); a12 = (u1 * u2).sum(2)
    b1 = (u1 * prof).sum(2); b2 = (u2 * prof).sum(2)
    det = a11 * a22 - a12 * a12
    ok = np.abs(det) > 1e-6 * np.maximum(a11 * a22, 1e-12)
    c1 = np.where(ok, (b1 * a22 - b2 * a12) / np.where(ok, det, 1), 0)
    c2 = np.where(ok, (b2 * a11 - b1 * a12) / np.where(ok, det, 1), 0)
    both = (c1 >= 0) & (c2 >= 0)
    # Otherwise the best single direction with a non-negative weight.
    s1 = np.clip(b1 / np.maximum(a11, 1e-12), 0, None)
    s2 = np.clip(b2 / np.maximum(a22, 1e-12), 0, None)
    r1 = ((prof - s1[..., None] * u1) ** 2).sum(2)
    r2 = ((prof - s2[..., None] * u2) ** 2).sum(2)
    use1 = r1 <= r2
    c1 = np.where(both, c1, np.where(use1, s1, 0))
    c2 = np.where(both, c2, np.where(use1, 0, s2))
    return (c1[..., None] * u1 + c2[..., None] * u2).astype(np.float32)


def clean_edges(fz, bgf, est, sub, dist, Wmap, lo_q, hi_q, strength, scale=4):
    """'Clean' mode: remove the outline-following part of the fused image's
    near-edge deviation from the clean background estimate, never going
    outside what some frame recorded there.  Background only, fading out by
    the clean distance W.  Returns the corrected background and the applied
    shift (full res)."""
    H, W = dist.shape
    bgmask = (~sub) & (dist > 2)
    lpf = lp_background(fz, bgmask, 6.0, scale)
    T = cv2.resize(est, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
    D = lpf - T
    dq = cv2.resize(dist, (W // scale, H // scale), interpolation=cv2.INTER_NEAREST)
    Wq = cv2.resize(Wmap, (W // scale, H // scale), interpolation=cv2.INTER_LINEAR)
    sub_q = cv2.resize(sub.astype(np.uint8), (W // scale, H // scale), interpolation=cv2.INTER_NEAREST) > 0
    # Analyse a little beyond the clean distance, so each stretch's
    # outermost bins are background the halo no longer reaches.
    zone = (dq > 0) & (dq < 1.2 * Wq)
    prof = contour_profile(D, sub_q, zone)
    prof = halo_directions(prof, lpf, cv2.resize(fz, (W // scale, H // scale), interpolation=cv2.INTER_AREA), sub_q)
    corrected = np.clip(lpf - prof, lo_q, hi_q)
    cap = 0.12 * np.abs(lpf)                       # gentle means gentle, even if something is off
    corrected = np.clip(corrected, lpf - cap, lpf + cap)
    t = np.clip(1.0 - dq / np.maximum(1.2 * Wq, 1.0), 0, 1)
    weight = (t * t * (3 - 2 * t)) * strength
    # Leave alone whatever borders very dark regions (a microscope's field
    # stop, a black vignette): they are not part of the specimen, and the
    # correction there produced dark blotches on the Marthamyces slide.
    lum = cv2.cvtColor(cv2.resize(fz, (W // scale, H // scale), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    ref = np.median(lum[~sub_q]) if (~sub_q).any() else np.median(lum)
    # Dark background only: the dark brown of a stem or the lower Panaeolus
    # cap held the correction to 20-60% right beside the stem.
    dark = ((lum < 0.35 * ref) & ~sub_q).astype(np.uint8)
    if dark.any():
        near_dark = cv2.distanceTransform(1 - dark, cv2.DIST_L2, 5) < np.maximum(1.5 * Wq / scale, 8)
        weight = weight * (~near_dark)
        weight = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), 4)
    shift_q = (corrected - lpf) * weight[..., None] * (~sub_q)[..., None]
    # Extend the correction a little into the subject before enlarging, then
    # cut it at the full-res outline: cutting at quarter res left a
    # stair-stepped edge that showed in the blend strip (Panaeolus stem).
    if (~sub_q).any():
        shift_q = normconv(shift_q, ~sub_q, (1.0, 2.0, 4.0))
    shift = cv2.resize(shift_q, (W, H), interpolation=cv2.INTER_LINEAR) * (~sub)[..., None]
    return bgf + shift, shift


def classify_never_sharp(sub, peak_half, peak_t, idx_half, N, fused, dist):
    """Split subject pixels no frame shows sharply into three causes:
    hidden (the camera never had a clear view: truly missing), out of range
    (sharpness still rising at the first or last frame: the stack did not
    reach that depth), featureless (too smooth/dark to measure sharpness at
    all).  Full-res float maps 0..1."""
    H, W = sub.shape
    h, w = peak_half.shape
    never = np.clip((peak_t - peak_half) / 0.5, 0, 1).astype(np.float32)
    ends = cv2.medianBlur(idx_half, 5)
    at_end = ((ends == 0) | (ends == N - 1)).astype(np.float32)
    g = cv2.resize(cv2.cvtColor(fused.astype(np.float32) / 257.0, cv2.COLOR_BGR2GRAY), (w, h),
                   interpolation=cv2.INTER_AREA)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    noise = 1.4826 * float(np.median(np.abs(hp))) + 1e-3
    local = np.sqrt(np.maximum(cv2.GaussianBlur(g * g, (0, 0), 3) - cv2.GaussianBlur(g, (0, 0), 3) ** 2, 0))
    flat = np.clip((4 * noise - local) / (2 * noise), 0, 1).astype(np.float32)
    # Hidden data sits where something blocks the view: next to the outline
    # or next to a depth step inside the subject (one cap in front of the
    # other).  Smooth, dark shadow there is hidden, not merely featureless.
    depth = cv2.medianBlur(idx_half, 5).astype(np.float32)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8
    sub_h = cv2.resize(sub.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    step = ((np.hypot(gx, gy) > 0.6) & (sub_h > 0)).astype(np.uint8)
    outline = cv2.morphologyEx(sub_h, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    blockers = np.maximum(step, outline)
    near_block = (cv2.distanceTransform((blockers == 0).astype(np.uint8), cv2.DIST_L2, 5) < 30).astype(np.float32)
    up = lambda x: cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR) * sub
    out_of_range = never * at_end
    rest = never * (1 - at_end)
    hidden = rest * np.maximum(near_block, 1 - flat)
    featureless = rest * (1 - near_block) * flat
    return up(hidden), up(out_of_range), up(featureless)


def match_noise(bgf, fz, dist):
    """Scale the background frame's fine grain to the fused image's.

    A single frame is noisier than the fused result (StackMFF's soft blend
    averages frames), so a pasted-in background looked grainier than the
    subject beside it.  Measured on background well away from the subject."""
    far = dist > 150
    if far.sum() < 10000:
        return bgf, 1.0
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 1.5)
    hb = hp(bgf)
    n_b = np.median(np.abs(hb[far]))
    n_f = np.median(np.abs(hp(fz)[far]))
    ratio = float(np.clip(n_f / max(n_b, 1e-6), 0.3, 1.0))
    return bgf - hb * (1 - ratio), ratio


def soft_object_mask(fz, est, dist, sub, width):
    """Objects the fused result clearly shows inside the rebuilt zone.

    A peg or stalk that is soft in every frame is missed by the focus
    measures; if it lies within `width` px of masked subject, rebuilding that
    zone from the clean background beyond wipes it out (dark slits where pegs
    cross, holes beside the stalk).  The edge darkening the rebuild exists to
    remove stays under ~30% of the background level, and patchy fusions
    (OpenFocus A) leave patches up to ~70% off it, so anything differing by
    more than the level itself (and at least 10 of 255) is an object: keep
    the fused pixels there.  Only ever adds protection."""
    lum = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) / 257.0
    lf, le = lum(fz), lum(est)
    obj = (dist > 0) & (dist < 1.5 * width) & (np.abs(lf - le) > np.maximum(1.0 * le, 10.0))
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    near = cv2.dilate(sub.astype(np.uint8), np.ones((5, 5), np.uint8))
    n, lab, _, _ = cv2.connectedComponentsWithStats(obj, 8)
    touch = np.zeros(n, bool)
    touch[np.unique(lab[(near > 0) & (obj > 0)])] = True
    touch[0] = False
    obj = cv2.dilate(touch[lab].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    return cv2.GaussianBlur(obj, (0, 0), 1.5)


def _spill_radius_results(G, sub_half, k_edge, k_fars, floor=16.0, map_targets=None):
    """Measure spill radii for one or more background frames in one pass.

    The subject geometry, edge selections and low-pass image for each source
    frame are independent of the requested far frame.  Sharing them avoids
    repeating that work when replacement and missing-data mapping use
    different frames.  Only maps in ``map_targets`` are materialised; every
    requested frame still receives its per-step table.
    """
    targets = tuple(dict.fromkeys(int(k) for k in k_fars))
    wanted_maps = set(targets if map_targets is None else map(int, map_targets))
    background = ~sub_half
    d = cv2.distanceTransform(background.astype(np.uint8), cv2.DIST_L2, 5) * 2
    farzone = background & (d > 256)
    has_farzone = bool(farzone.any())

    def normalised(k):
        g = G[k]
        if not has_farzone:
            return g
        return g / max(float(np.median(g[farzone])), 1e-6)

    def lowpass(k):
        return cv2.GaussianBlur(normalised(k), (0, 0), 3)

    far_images = {k: lowpass(k) for k in targets}
    edges = [0, 16, 32, 48, 64, 96, 128, 192, 256, 384]
    ratios = {k: [] for k in targets}
    weights = {k: [] for k in targets}
    for frame in np.unique(k_edge[background]):
        frame = int(frame)
        sel = background & (k_edge == frame)
        weight = int(sel.sum())
        active = [k for k in targets if abs(k - frame) >= 2 and weight >= 2000]
        if not active:
            continue
        near = far_images.get(frame)
        if near is None:
            near = lowpass(frame)
        tail = sel & (d > 256)
        tail_count = int(tail.sum())
        for k_far in active:
            dk = abs(k_far - frame)
            diff = np.abs(far_images[k_far] - near)
            prof = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = sel & (d > lo) & (d <= hi)
                prof.append(float(diff[m].mean()) if m.sum() > 200 else np.nan)
            prof = np.array(prof)
            if not np.isfinite(prof[0]):
                continue
            base = min(float(np.median(diff[tail])) if tail_count > 500 else np.inf,
                       np.nanmin(prof))
            far_mean = np.nanmean(prof[6:]) if np.isfinite(prof[6:]).any() else base
            if prof[0] < 1.5 * far_mean:      # no decline: scene detail, not glow
                continue
            limit = base + 0.15 * (prof[0] - base)
            radius = edges[1]
            for hi, value in zip(edges[1:], prof):
                if not np.isfinite(value) or value <= limit:
                    break
                radius = hi
            ratios[k_far].append(radius / dk)
            weights[k_far].append(weight)

    results = {}
    for k_far in targets:
        target_ratios = ratios[k_far]
        if target_ratios:
            order = np.argsort(target_ratios)
            cw = np.cumsum(np.array(weights[k_far])[order])
            per_step = 1.5 * float(
                np.array(target_ratios)[order][np.searchsorted(cw, cw[-1] / 2)])
        else:
            per_step = 48.0 / max(len(G) // 2, 1)
        table = {dk: float(np.clip(per_step * dk, floor, 384)) for dk in range(len(G))}
        table["per_step_raw"] = per_step / 1.5
        rmap = None
        if k_far in wanted_maps:
            radius_lookup = np.fromiter(
                (table[abs(k_far - k)] for k in range(len(G))), np.float32, count=len(G))
            rmap = cv2.GaussianBlur(radius_lookup[k_edge], (0, 0), 8)
        results[k_far] = rmap, table
    return results


def spill_radius_map(G, sub_half, k_edge, k_far, floor=16.0):
    """Per-pixel distance (full-res px) over which the far frame still shows
    the subject's own defocus glow.

    Measured on the stack, not assumed: for the background beside edges that
    are sharp in frame k, compare the far frame with frame k (low-passed,
    exposure-normalised) by distance from the subject, and find where the
    difference falls back to its far-away baseline.  More defocus steps means
    a wider glow, so the radius is made non-decreasing in |k_far - k|.
    Returns the half-res radius map and the per-step table."""
    return _spill_radius_results(G, sub_half, k_edge, (k_far,), floor)[int(k_far)]


def background_curve(G, bgmask):
    """Coarse-scale background detail per frame (rises toward the far end)."""
    h, w = G.shape[1:]
    m = cv2.resize(bgmask.astype(np.uint8), (w // 4, h // 4),
                   interpolation=cv2.INTER_NEAREST) > 0
    out = []
    for g in G:
        s = cv2.resize(g, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        lap = cv2.Laplacian(s, cv2.CV_32F, ksize=3)
        out.append(float((lap[m] ** 2).mean()))
    return np.array(out)


def _u16(img: np.ndarray) -> np.ndarray:
    """Any project image -> uint16 BGR (8-bit x257, float 0-255 scaled)."""
    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype == np.uint16:
            # Callers treat this as read-only.  Preserve the old contiguous
            # result for cropped/non-contiguous render-context views.
            return np.ascontiguousarray(img)
        if img.dtype == np.uint8:
            return np.ascontiguousarray(img, dtype=np.uint16) * np.uint16(257)
    return np.clip(_to16f(img), 0, 65535).astype(np.uint16)


def replace_background(frames: Sequence[np.ndarray], fused: np.ndarray, *,
                       preset: str = "auto", clean: bool = True,
                       background_frame: Optional[int] = None,
                       max_subject: float = MAX_SUBJECT_DEFAULT,
                       keep_soft_parts: bool = False,
                       valid_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
                       progress: Optional[Callable[[float, str], None]] = None,
                       cancelled: Optional[Callable[[], bool]] = None) -> BackgroundFillResult:
    """Replace the background of `fused` with one of `frames`.

    `frames` are the aligned frames that produced `fused`, in stack order
    (either focus direction), each the same size as `fused`.  `preset`
    "soft" takes the frame with the blurriest background, "sharp" the
    far-focus frame, "auto" soft unless much of the background is detail
    the stack recorded in focus; `background_frame` forces a frame index.  `clean`
    removes the stack's outline-following edge flaws within what the frames
    recorded; False keeps them (Faithful).  `keep_soft_parts` also keeps
    soft objects attached to the subject that some frame focused on (peg
    knobs just past the depth of field), which can keep background objects
    near the subject in the stack's own rendering too.  When the subject covers more
    than `max_subject` of the frame (a ground-level scene) the image is left
    as stacked and only the missing-data maps are computed.  `valid_masks`
    optionally marks registration-valid pixels for each frame; invalid warp
    fill is excluded from matching and never composited into the result.
    """
    if preset not in PRESETS:
        raise BackgroundFillError(f"unknown preset {preset!r}")
    N = len(frames)
    if N < 2:
        raise BackgroundFillError("background replacement needs at least two frames")
    H, W = fused.shape[:2]
    if H < 64 or W < 64:
        raise BackgroundFillError("image too small for background replacement")
    for f in frames:
        if f.shape[:2] != (H, W):
            raise BackgroundFillError("every frame must match the result's size")
    if valid_masks is None:
        valid_masks = (None,) * N
    else:
        if len(valid_masks) != N:
            raise BackgroundFillError("valid_masks must match the frame count")
        normalized_masks = []
        for mask in valid_masks:
            if mask is None:
                normalized_masks.append(None)
                continue
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (H, W):
                raise BackgroundFillError("every validity mask must match the result's size")
            normalized_masks.append(mask)
        valid_masks = tuple(normalized_masks)

    steps = [0]
    total = 14

    def step(msg: str) -> None:
        if cancelled is not None and cancelled():
            raise BackgroundFillCancelled()
        if progress is not None:
            progress(min(steps[0] / total, 1.0), msg)
        steps[0] += 1

    margin, feather, halo_fill = 2.0, 6.0, 32.0
    min_island = 2e-5

    step("analysing focus")
    grays = []
    for f in frames:
        g = cv2.cvtColor(_u16(f).astype(np.float32) / 65535.0, cv2.COLOR_BGR2GRAY)
        grays.append(cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA))
        if cancelled is not None and cancelled():
            raise BackgroundFillCancelled()
    G = np.stack(grays)
    del grays
    frame_medians = [np.median(g) for g in G]
    measures, idx = focus_measures(G)
    sub_half, thresh = subject_mask(measures, ["ratio", "peak"], None, min_island)
    fused16 = _u16(fused)
    color_half = cv2.resize(fused16, (G.shape[2], G.shape[1]), interpolation=cv2.INTER_AREA).astype(np.float32) / 65535.0

    step("finding the subject's outline")
    sub_half = fine_outline(sub_half, G, color_half)
    # Skip the strip next to the subject: blurred subject edges there would
    # count as background detail and skew which end looks like the background.
    away = cv2.distanceTransform((~sub_half).astype(np.uint8), cv2.DIST_L2, 5) > 20
    if away.sum() < 64:
        raise BackgroundFillError("no background found: the whole frame looks like subject")
    curve = background_curve(G, away)
    far_end = N - 1 if curve[-1] >= curve[0] else 0
    # Blurriest background: least coarse-scale detail away from the
    # subject.  Frames within 3% of that look equally blurry; of those take
    # the one closest in focus to the subject, which carries the least
    # subject glow.
    tied = np.flatnonzero(curve <= curve.min() * 1.03)
    k_subject = float(np.median(idx[sub_half])) if sub_half.any() else N / 2
    k_soft = int(tied[np.argmin(np.abs(tied - k_subject))])
    if background_frame is not None:
        if not 0 <= background_frame < N:
            raise BackgroundFillError("background frame index out of range")
        k_bg, used = int(background_frame), "forced"
    elif preset == "sharp":
        k_bg, used = far_end, "sharp"
    else:
        k_bg, used = k_soft, "soft"

    step("protecting smooth subject")
    sub_half, _t, _n, _pct = protect_missed_subject(sub_half, G, far_end, min_island * sub_half.size)
    gray_fused = cv2.cvtColor(color_half, cv2.COLOR_BGR2GRAY)
    gray_fused_median = np.median(gray_fused)

    def protect(k, gray, gray_median):
        fused_half = gray * (frame_medians[k] / max(float(gray_median), 1e-6))
        return protect_visible_detail(
            sub_half, G, k, fused_half, frame_medians=frame_medians)[0]

    protected = protect(k_bg, gray_fused, gray_fused_median)
    # Share of the background that is detail some frame recorded in focus
    # and the stack kept: scene rather than bokeh.  Blurring that away with
    # the soft frame looks worse than the stack; the far frame keeps it.
    free = max(1.0 - float(sub_half.mean()), 1e-6)
    recorded_share = float(protected.mean() - sub_half.mean()) / free
    if preset == "auto" and background_frame is None and recorded_share > AUTO_SHARP_SHARE \
            and far_end != k_soft:
        k_bg, used = far_end, "sharp"
        protected = protect(k_bg, gray_fused, gray_fused_median)
    fused_bg = gray_fused * (frame_medians[k_bg] / max(float(gray_fused_median), 1e-6))
    if keep_soft_parts:
        protected = protect_focused_objects(
            protected, G, k_bg, fused_bg, frame_medians=frame_medians)
    sub_half = protected
    del protected, gray_fused, fused_bg
    log.info("background frame %d (preset %s -> %s, far end %d, recorded detail %.1f%% of background)",
             k_bg, preset, used, far_end, recorded_share * 100)

    step("refining the outline")
    matte = refine_subject(sub_half, fused16, band=6, r=4)
    sub = matte > 0.5
    # Precise distances: the fast approximation has polygonal iso-lines,
    # and every band defined by distance inherited those straight edges.
    dist = cv2.distanceTransform((~sub).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = np.clip((dist - margin) / feather, 0, 1).astype(np.float32)
    alpha = alpha * alpha * (3 - 2 * alpha)
    # Never let the background cover soft subject detail the matte still sees.
    alpha = np.minimum(alpha, 1 - np.clip(matte, 0, 1))
    del matte

    fz = fused16.astype(np.float32)
    # Colour-match on background well clear of the subject: right next to it
    # every frame is darkened by partial occlusion, which biased the gain.
    band = (dist > max(64, 2 * halo_fill)) & (dist < 600)
    lum_f = fz.mean(axis=2)
    if (~sub).any():
        band &= lum_f > 0.35 * float(np.median(lum_f[~sub]))   # not a microscope's field stop
    del lum_f
    if band.sum() < 10000:
        band = dist > margin + feather
    match_target = [np.median(fz[..., c][band]) for c in range(3)]

    def load_matched(k):
        f = _u16(frames[k]).astype(np.float32)
        sample = band if valid_masks[k] is None else band & valid_masks[k]
        if sample.sum() < 64 and valid_masks[k] is not None:
            sample = (~sub) & valid_masks[k]
        if not sample.any():
            raise BackgroundFillError(f"frame {k + 1} has no valid background pixels")
        target = match_target if valid_masks[k] is None else [
            np.median(fz[..., c][sample]) for c in range(3)]
        g = np.array([target[c] / max(np.median(f[..., c][sample]), 1e-3)
                      for c in range(3)], np.float32)
        return f * g, g

    step("matching the background frame")
    bgf, gains = load_matched(k_bg)
    k_edge = edge_frame_map(sub_half, idx)
    # Next to the subject no single frame has a clean background: the frame
    # where an edge is sharp is darkened by partial occlusion, the background
    # frame carries the subject's glow out to the measured radius.  Rebuild
    # from the background frame beyond max(halo_fill, 1.25 x glow radius).
    spill_results = _spill_radius_results(
        G, sub_half, k_edge, (k_bg, far_end), map_targets=(k_bg,))
    rmap, _table = spill_results[k_bg]
    _, table_far = spill_results[far_end]
    del spill_results
    size_cap = object_size_cap(sub_half, (H, W))
    fill_width = np.maximum(np.float32(halo_fill),
                            np.minimum(1.25 * cv2.resize(rmap, (W, H), interpolation=cv2.INTER_LINEAR), size_cap))
    # The width follows the nearest edge's focus position, which switches
    # along straight lines (a concave corner's diagonal); smooth it so the
    # zone boundary cannot jump there and draw a wedge.
    fill_width = blur(fill_width.astype(np.float32), 48.0)
    del rmap
    bgf, _ratio = match_noise(bgf, fz, dist)

    step("rebuilding the background beside the subject")
    bgf, est = fill_halo_zone(bgf, dist, sub, fill_width)
    wmap = edge_width_map(fz, est, sub, dist, lo=feather)
    soft_a = np.clip((dist - margin) / wmap, 0, 1).astype(np.float32)
    soft_a = soft_a * soft_a * (3 - 2 * soft_a)
    alpha = np.minimum(alpha, soft_a)
    del wmap, soft_a

    step("matching the stack's shading")
    # Very-low-frequency shading (illumination, vignetting, exposure drift)
    # from the stacked image everywhere, so the background frame contributes
    # only its blur and grain; otherwise each object's hand-over zone shows
    # as a patch.
    big = 0.08 * float(np.hypot(H, W))
    bgm = (~sub) & (dist > 2)
    vlf = lp_background(fz, bgm, big) - lp_background(bgf, bgm, big)
    bgf = bgf + cv2.resize(blur(vlf, big / 16), (W, H), interpolation=cv2.INTER_LINEAR)
    del vlf, bgm
    reach = np.clip(2.5 * np.asarray(fill_width, np.float32), 48, 400)
    bgf = natural_near_background(bgf, fz, sub, dist, reach)
    del reach
    keep_obj = soft_object_mask(fz, est, dist, sub, fill_width)
    alpha = np.minimum(alpha, 1 - keep_obj)
    del keep_obj

    step("mapping missing data")
    c_raw = table_far["per_step_raw"]
    w_occ, _occ = occlusion_width(G, sub_half, k_edge)
    ks_sub = idx[sub_half]
    k_occ = int(np.median(ks_sub)) if ks_sub.size else N // 2
    k_edge_full = cv2.resize(k_edge, (W, H), interpolation=cv2.INTER_NEAREST)
    peak_t = thresh.get("peak", otsu_threshold(measures["peak"]))
    bg_miss, _soft, _w_by_ke, _k_B, Wmap = missing_data_map(sub, dist, k_edge_full, measures["peak"], peak_t,
                                                         N, far_end, c_raw, w_occ, k_occ,
                                                         size_cap)
    del k_edge_full, size_cap
    step("classifying unseen subject")
    hidden, out_range, featureless = classify_never_sharp(sub, measures["peak"], peak_t, idx, N, fused16, dist)
    to8 = lambda m: (np.clip(m, 0, 1) * 255).astype(np.uint8)
    missing = MissingData(to8(bg_miss), to8(hidden), to8(out_range), to8(featureless))
    del bg_miss, hidden, out_range, featureless

    shift = None
    if clean:
        step("cleaning edge flaws")
        lo_q, hi_q = recorded_range(N, load_matched, (H, W), valid_masks=valid_masks)
        if cancelled is not None and cancelled():
            raise BackgroundFillCancelled()
        bgf, shift = clean_edges(fz, bgf, est, sub, dist, Wmap, lo_q, hi_q, 1.0)
        del lo_q, hi_q
    del est, Wmap

    step("compositing")
    valid_bg = valid_masks[k_bg]
    if valid_bg is not None:
        alpha = np.where(valid_bg, alpha, 0.0).astype(np.float32)
    subject_fraction = float(sub.mean())
    changed = subject_fraction <= max_subject
    if changed:
        comp = fz * (1 - alpha[..., None]) + bgf * alpha[..., None]
        if shift is not None:
            # The blend strip at the edge keeps stacked pixels; correct their
            # background side too, or a thin line is left along the edge.
            shift_region = (~sub) if valid_bg is None else ((~sub) & valid_bg)
            comp += shift * ((1 - alpha) * shift_region)[..., None]
        out16 = np.clip(comp, 0, 65535).astype(np.uint16)
        del comp
    else:
        # A ground-level scene: nearly everything is scene the stack
        # recorded, and the few percent left are far-range objects the
        # background frame would only blur.
        log.info("subject covers %.1f%% of the frame: background left as stacked", subject_fraction * 100)
        # ``_u16`` may return an already-uint16 input without copying.  Keep
        # result ownership independent from the caller as it was before the
        # integer fast path.
        out16 = fused16.copy()
    image = _from16f(out16.astype(np.float32), fused) if fused.dtype != np.uint16 or fused.ndim != 3 \
        or fused.shape[2] != 3 else out16
    step("done")
    return BackgroundFillResult(
        image=image, changed=changed, background_frame=k_bg, far_end=far_end,
        subject_fraction=subject_fraction, missing=missing, preset_used=used,
        recorded_share=recorded_share,
        stats={"gains": [float(g) for g in gains], "thresholds": {k: float(v) for k, v in thresh.items()}})


def missing_data_overlay(image: np.ndarray, missing: MissingData, dim: float = 0.55,
                         strength: float = 0.75) -> np.ndarray:
    """8-bit BGR preview: the image dimmed, each missing-data class tinted."""
    from_any = _to16f(image) / 257.0
    ov = from_any * dim
    for _name, m, colour in missing.layers():
        amt = (m.astype(np.float32) / 255.0 * strength)[..., None]
        ov = ov * (1 - amt) + np.array(colour, np.float32) * amt
    return np.clip(ov, 0, 255).astype(np.uint8)
