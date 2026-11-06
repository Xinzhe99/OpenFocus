"""Bounded similarity registration using a sampled difference detector."""
from collections import OrderedDict
import hashlib
import itertools
from threading import Lock

import cv2
import numpy as np

# Results contain only five scalars, never source pixels or source lifetimes.
_TRANSFORMS = OrderedDict()
_TRANSFORM_LOCK = Lock()
_TRANSFORM_LIMIT = 128


def _identity(image):
    image = np.ascontiguousarray(image)
    return (image.shape, image.dtype.str,
            hashlib.sha256(memoryview(image).cast('B')).digest())


def _samples(image, x, y):
    h, w = image.shape
    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x = np.clip(x, 0, max(0, w - 1.001)).astype(np.float32)
    y = np.clip(y, 0, max(0, h - 1.001)).astype(np.float32)
    ix, iy = x.astype(np.int32), y.astype(np.int32)
    fx = ((x - ix) * 256).astype(np.uint32)
    fy = ((y - iy) * 256).astype(np.uint32)
    jx, jy = np.minimum(ix + 1, w - 1), np.minimum(iy + 1, h - 1)
    # The convex fixed-point sum is at most 65535 * 65536, so uint32
    # preserves every detector bit while halving interpolation scratch.
    a = image[iy, ix].astype(np.uint32)
    b = image[iy, jx].astype(np.uint32)
    c = image[jy, ix].astype(np.uint32)
    d = image[jy, jx].astype(np.uint32)
    value = ((a * (256 - fx) + b * fx) * (256 - fy)
             + (c * (256 - fx) + d * fx) * fy) >> 16
    return value, valid


def _native_proposal(a, b, check_cancel):
    """Propose a similarity with native ECC; the detector decides acceptance."""
    check_cancel()
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        # ECC is exposure invariant. Only the small estimation images enter
        # its affine fit; shear is removed before any detector evaluation.
        cv2.findTransformECC(a.astype(np.float32) / 65535.0,
                             b.astype(np.float32) / 65535.0, warp,
                             cv2.MOTION_AFFINE,
                             (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                              40, 1e-4), None, 5)
        correction = cv2.invertAffineTransform(warp).astype(np.float64)
    except cv2.error:
        return None
    check_cancel()
    aa = (correction[0, 0] + correction[1, 1]) / 2
    bb = (correction[0, 1] - correction[1, 0]) / 2
    center = np.array([a.shape[1] / 2, a.shape[0] / 2])
    shift = correction[:, :2] @ center + correction[:, 2] - center
    return np.array([*shift, np.hypot(aa, bb) - 1,
                     np.degrees(np.arctan2(bb, aa))])


def estimate(ref, cur, shift_ratio, rot_bound, scale_range, check_cancel):
    """Return quality, rotation, scale and residual shift in thumbnail pixels."""
    check_cancel()
    key = (_identity(ref), _identity(cur), float(shift_ratio),
           float(rot_bound), tuple(scale_range))
    check_cancel()
    with _TRANSFORM_LOCK:
        cached = _TRANSFORMS.get(key)
        if cached is not None:
            _TRANSFORMS.move_to_end(key)
    if cached is not None:
        check_cancel()
        return cached
    result = _estimate(ref, cur, shift_ratio, rot_bound, scale_range, check_cancel)
    check_cancel()
    with _TRANSFORM_LOCK:
        _TRANSFORMS[key] = result
        _TRANSFORMS.move_to_end(key)
        while len(_TRANSFORMS) > _TRANSFORM_LIMIT:
            _TRANSFORMS.popitem(last=False)
    return result


def _estimate(ref, cur, shift_ratio, rot_bound, scale_range, check_cancel):
    h, w = ref.shape
    if h < 2 or w < 2:
        return -1.0, 0.0, 1.0, 0.0, 0.0
    # A shared scale preserves relative exposure and the detector's integer
    # differences for both 8-bit/float 0..255 and true 16-bit inputs.
    peak = max(float(np.max(ref)), float(np.max(cur)))
    factor = 257.0 if peak <= 255.0 else 1.0
    a = np.clip(ref * factor, 0, 65535).astype(np.uint16)
    b = np.clip(cur * factor, 0, 65535).astype(np.uint16)
    # Detector scale compensates exposure; its offset is removed by variance.
    mean_a, mean_b = float(a.mean()), float(b.mean())
    gain_q10 = int(1024 * mean_b / mean_a) if mean_a > 0 else 1024
    gain_q10 = min(max(gain_q10, 0), 32768)
    if float(a.std()) < 1.0 or float(b.std()) < 1.0:
        return -1.0, 0.0, 1.0, 0.0, 0.0

    # A one-dimensional gradient cannot constrain motion along its level
    # curves. Reject it instead of inventing a shift from equivalent minima.
    for image in (a, b):
        check_cancel()
        dx = np.diff(image.astype(np.float32), axis=1)[:-1, :]
        dy = np.diff(image.astype(np.float32), axis=0)[:, :-1]
        xx = float(np.mean(dx * dx))
        yy = float(np.mean(dy * dy))
        xy = float(np.mean(dx * dy))
        trace = xx + yy
        if trace == 0 or xx * yy - xy * xy <= 1e-6 * trace * trace:
            return -1.0, 0.0, 1.0, 0.0, 0.0

    # Search on a small pyramid. Translation is expressed in original
    # thumbnail coordinates so asymmetric dimensions keep independent bounds.
    # The floor must stay below the grid's capture radius: with only the
    # full-resolution image a 2-3 px translation sits between 6-7 px samples
    # and is never found.
    pyramid = [(a, b)]
    while min(pyramid[-1][0].shape) >= 64:
        aa, bb = pyramid[-1]
        size = ((aa.shape[1] + 1) // 2, (aa.shape[0] + 1) // 2)
        pyramid.append((cv2.resize(aa, size, interpolation=cv2.INTER_AREA),
                        cv2.resize(bb, size, interpolation=cv2.INTER_AREA)))
    bounds = np.array([w * shift_ratio, h * shift_ratio,
                       max(0.0, 1.0 - scale_range[0]), rot_bound], dtype=np.float64)
    lower = np.array([-bounds[0], -bounds[1], scale_range[0] - 1.0, -rot_bound])
    upper = np.array([bounds[0], bounds[1], scale_range[1] - 1.0, rot_bound])
    best = np.zeros(4, dtype=np.float64)  # forward dx, dy, scale-1, degrees
    radius = bounds.copy()
    quality = -1.0
    proposal = _native_proposal(a, b, check_cancel)
    # A native proposal is only a seed. Keep the exhaustive pyramid whenever
    # its full-resolution sampled difference cannot establish a strong match.
    fast = False
    if (proposal is not None and np.all(np.isfinite(proposal))
            and np.all(proposal >= lower) and np.all(proposal <= upper)):
        step = max(1, int(np.sqrt(h * w / 1024.0)))
        yy, xx = np.mgrid[0:h:step, 0:w:step].astype(np.float32)
        angle = np.deg2rad(proposal[3])
        cos, sin = np.cos(angle) / (1 + proposal[2]), np.sin(angle) / (1 + proposal[2])
        x, y = xx - w / 2 - proposal[0], yy - h / 2 - proposal[1]
        sampled, valid = _samples(b, cos * x - sin * y + w / 2,
                                 sin * x + cos * y + h / 2)
        reference = (a[::step, ::step].astype(np.int64) * gain_q10) >> 10
        delta = reference[valid] - sampled[valid].astype(np.int64)
        fast = (valid.sum() >= reference.size * 0.75 and delta.size > 0
                and float(delta.std()) <= 0.1 * max(float(reference.std()), 1.0))
        if fast:
            best = proposal
            pyramid = pyramid[:1]
            radius = np.minimum(bounds, [2.0, 2.0, 0.01, 0.5])
    for level, (aa, bb) in enumerate(reversed(pyramid)):
        hh, ww = aa.shape
        sx, sy = ww / w, hh / h
        # Detector sampling is deterministic and excludes out-of-bounds pixels.
        step = max(1, int(np.sqrt(hh * ww / 1024.0)))
        yy, xx = np.mgrid[0:hh:step, 0:ww:step].astype(np.float32)
        xx, yy = xx.reshape(1, -1), yy.reshape(1, -1)
        reference = aa[yy.astype(np.int32), xx.astype(np.int32)].astype(np.int64)
        reference = (reference * gain_q10) >> 10
        ref_std = max(float(reference.std()), 1.0)
        if level:
            radius = np.minimum(bounds, np.maximum(radius, [2 / sx, 2 / sy, 0.01, 0.5]))
        for iteration in range(5):
            check_cancel()
            axes = []
            for k in range(4):
                lo = max(lower[k], best[k] - radius[k])
                hi = min(upper[k], best[k] + radius[k])
                # A sparse first grid can lock onto a repeated texture before
                # refinement reaches the correct translation basin. Sample
                # translation more densely at the coarsest level; later grids
                # still refine all four parameters within the same bounds.
                points = 17 if not fast and level == 0 and iteration == 0 and k < 2 else 5
                axis = np.linspace(lo, hi, points) if hi > lo + 1e-12 else np.array([lo])
                # Near a bound the grid is asymmetric. Retain the validated
                # seed so refinement can never replace it with a worse fit.
                axes.append(np.unique(np.append(axis, best[k])) if fast else axis)
            # Prefer the smallest normalized motion when costs are equal.
            candidates = np.array([tuple(reversed(v)) for v in itertools.product(*axes[::-1])])
            best_cost, best_motion, winner = float('inf'), float('inf'), None
            for start in range(0, len(candidates), 32):
                check_cancel()
                chunk = candidates[start:start + 32]
                dx, dy = chunk[:, 0:1] * sx, chunk[:, 1:2] * sy
                scale = (1.0 + chunk[:, 2:3]).astype(np.float32)
                angle = np.deg2rad(chunk[:, 3:4]).astype(np.float32)
                cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
                x, y = xx - ww / 2.0 - dx, yy - hh / 2.0 - dy
                u = cos * x - sin * y + ww / 2.0
                v = sin * x + cos * y + hh / 2.0
                sampled, valid = _samples(bb, u, v)
                delta = np.where(valid, reference - sampled, 0)
                count = valid.sum(axis=1)
                denom = np.maximum(count, 1)
                mean = delta.sum(axis=1) / denom
                variance = (delta * delta).sum(axis=1) / denom - mean * mean
                cost = np.sqrt(np.maximum(variance, 0))
                cost[count < reference.size * 0.75] = np.inf
                motion = np.sum((chunk / np.where(bounds > 0, bounds, 1)) ** 2, axis=1)
                idx = int(np.lexsort((motion, cost))[0])
                if np.isfinite(cost[idx]) and (cost[idx] < best_cost or
                        (cost[idx] == best_cost and motion[idx] < best_motion)):
                    best_cost, best_motion = float(cost[idx]), float(motion[idx])
                    winner = chunk[idx].copy()
            if winner is None:
                break
            best = winner
            quality = 1.0 - best_cost / ref_std
            radius *= 0.5
    # Existing composition subtracts the returned residual shift.
    return quality, float(best[3]), float(1 + best[2]), -float(best[0]), -float(best[1])
