"""Global low-resolution content seams; caller owns warping and pixel storage.

LowSample colors are normalized BGR, alpha includes source masks. Labels contain
source index + 1, zero is uncovered. No per-source canvas weights are retained.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


def _cancel(callback):
    if callback is not None and callback():
        raise InterruptedError('Mosaic seam planning cancelled')


@dataclass(frozen=True)
class SeamSpec:
    mode: str = 'dp'
    precision: int = 8
    band: int = 64
    cost: str = 'color_gradient'
    loop_refine: bool = False

    def __post_init__(self):
        if self.mode not in ('dp', 'graphcut', 'feather'):
            raise ValueError('Unknown seam mode')
        if self.precision not in (1, 2, 4, 8) or isinstance(self.precision, bool):
            raise ValueError('Seam precision must be 1, 2, 4 or 8')
        if not isinstance(self.band, (int, np.integer)) or not 0 <= self.band <= 2048:
            raise ValueError('Seam band must be an integer between 0 and 2048')
        if self.cost not in ('color', 'color_gradient'):
            raise ValueError('Unknown seam cost')


@dataclass(frozen=True)
class LowSample:
    index: int
    colors: np.ndarray
    alpha: np.ndarray
    origin: tuple

    @property
    def box(self):
        x, y = self.origin
        h, w = self.alpha.shape
        return x, y, x + w, y + h


def _intersection(first, second):
    a, b = first.box, second.box
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    sa = np.s_[y0 - a[1]:y1 - a[1], x0 - a[0]:x1 - a[0]]
    sb = np.s_[y0 - b[1]:y1 - b[1], x0 - b[0]:x1 - b[0]]
    return (x0, y0, x1, y1), sa, sb


def _path(cost, cancel):
    """Three-neighbor DP with rolling costs and one-byte traceback storage."""
    h, w = cost.shape
    previous = cost[0].copy()
    back = np.zeros((h, w), np.int8)
    columns = np.arange(w)
    for y in range(1, h):
        _cancel(cancel)
        padded = np.pad(previous, 1, constant_values=np.inf)
        candidates = np.stack((padded[:w], padded[1:w + 1], padded[2:w + 2]))
        choice = np.argmin(candidates, axis=0)
        current = cost[y] + candidates[choice, columns]
        if not np.isfinite(current).any():
            # A masked row separates paths; restart on the next valid segment.
            current = cost[y].copy()
            choice.fill(1)
        back[y] = choice.astype(np.int8) - 1
        previous = current
    path = np.empty(h, np.int32)
    path[-1] = int(np.argmin(previous))
    for y in range(h - 1, 0, -1):
        path[y - 1] = np.clip(path[y] + back[y, path[y]], 0, w - 1)
    return path


def _child_side(first, second, sa, sb, common, spec, cancel):
    a, b = first.colors[sa], second.colors[sb]
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    cost = np.mean(np.abs(a - b), axis=2)
    if spec.cost == 'color_gradient':
        gradient = np.abs(cv2.Sobel(ga, cv2.CV_32F, 1, 0)
                          - cv2.Sobel(gb, cv2.CV_32F, 1, 0))
        gradient += np.abs(cv2.Sobel(ga, cv2.CV_32F, 0, 1)
                           - cv2.Sobel(gb, cv2.CV_32F, 0, 1))
        cost += gradient * 0.125
    distance = cv2.distanceTransform(common.astype(np.uint8), cv2.DIST_L2, 5)
    cost += 0.5 / (distance + 1.0)
    cost[~common] = np.inf
    h, w = common.shape
    ca = ((first.box[0] + first.box[2]) / 2, (first.box[1] + first.box[3]) / 2)
    cb = ((second.box[0] + second.box[2]) / 2, (second.box[1] + second.box[3]) / 2)
    # Camera/source centers determine orientation, rather than tile aspect alone.
    horizontal = abs(cb[0] - ca[0]) >= abs(cb[1] - ca[1])
    if horizontal:
        path = _path(cost, cancel)
        side = np.arange(w)[None, :] >= path[:, None]
        return side if cb[0] >= ca[0] else ~side
    path = _path(cost.T, cancel)
    side = np.arange(h)[:, None] >= path[None, :]
    return side if cb[1] >= ca[1] else ~side


def build_labels(samples, shape, spec=SeamSpec(), *, allocate=None,
                 should_cancel=None):
    """Solve one global label canvas from already corrected, cropped low warps.

    ``shape`` is (low_height, low_width). ``allocate(shape,dtype)`` may create a
    disk mapping; scratch is bounded by the largest low-resolution overlap.
    Explicit graphcut requests fail visibly if that OpenCV binding is absent.
    """
    _cancel(should_cancel)
    samples = tuple(samples)
    if len(samples) > 65535 or len({s.index for s in samples}) != len(samples):
        raise ValueError('Seam samples need unique indices below 65535')
    labels = (np.empty(shape, np.uint16) if allocate is None
              else allocate(shape, np.uint16))
    labels.fill(0)
    for sample in samples:
        _cancel(should_cancel)
        x0, y0, x1, y1 = sample.box
        if (not 0 <= sample.index < 65535 or x0 < 0 or y0 < 0
                or x1 > shape[1] or y1 > shape[0]
                or sample.colors.shape != (*sample.alpha.shape, 3)):
            raise ValueError('Invalid low-resolution seam sample')
        region = labels[y0:y1, x0:x1]
        region[(sample.alpha > 0) & (region == 0)] = sample.index + 1
    active = [s for s in samples if s.alpha.size and np.any(s.alpha > 0)]
    if spec.mode == 'feather' or len(active) <= 1:
        return labels
    if spec.mode == 'graphcut':
        detail = getattr(cv2, 'detail', None)
        factory = getattr(detail, 'GraphCutSeamFinder', None)
        if factory is None:
            raise ValueError('This OpenCV build has no graph-cut seam finder')
        finder = factory('COST_COLOR' if spec.cost == 'color' else 'COST_COLOR_GRAD')
        masks = finder.find([s.colors for s in active], [s.origin for s in active],
                            [(s.alpha > 0).astype(np.uint8) * 255 for s in active])
        _cancel(should_cancel)
        labels.fill(0)
        for sample, mask in zip(active, masks):
            mask = mask.get() if hasattr(mask, 'get') else np.asarray(mask)
            x0, y0, x1, y1 = sample.box
            labels[y0:y1, x0:x1][(mask > 0) & (sample.alpha > 0)] = sample.index + 1
        return labels
    edges = []
    for i, first in enumerate(active):
        for j in range(i + 1, len(active)):
            _cancel(should_cancel)
            found = _intersection(first, active[j])
            if found is None:
                continue
            _, sa, sb = found
            area = int(np.count_nonzero((first.alpha[sa] > 0) & (active[j].alpha[sb] > 0)))
            if area:
                edges.append((area, i, j))
    edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    parent = list(range(len(active)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    neighbors = [[] for _ in active]
    loops = []
    for _, i, j in edges:
        ri, rj = root(i), root(j)
        if ri == rj:
            loops.append((i, j))
        else:
            parent[rj] = ri
            neighbors[i].append(j)
            neighbors[j].append(i)
    order, seen = [], set()
    for first in range(len(active)):
        if first in seen:
            continue
        seen.add(first)
        queue = [first]
        for i in queue:
            for j in sorted(neighbors[i]):
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
                    order.append((i, j))
    if spec.loop_refine:
        order.extend(loops)
    for i, j in order:
        _cancel(should_cancel)
        first, second = active[i], active[j]
        box, sa, sb = _intersection(first, second)
        common = (first.alpha[sa] > 0) & (second.alpha[sb] > 0)
        if not common.any():
            continue
        child = _child_side(first, second, sa, sb, common, spec, should_cancel)
        x0, y0, x1, y1 = box
        region = labels[y0:y1, x0:x1]
        update = common & ((region == first.index + 1) | (region == second.index + 1)
                           | (region == 0))
        region[update] = np.where(child[update], second.index + 1, first.index + 1)
    _cancel(should_cancel)
    return labels


def label_weight(labels, index, *, origin, scale, x, y, width, height, band,
                 pixel_step=1.0):
    """Sample one source's seam weight using absolute full-resolution centers.

    Distance transforms include the transition neighborhood, independently of
    render tile boundaries. Resampling uses global coordinates, never resize.
    """
    if not math.isfinite(pixel_step) or pixel_step <= 0:
        raise ValueError('Seam sampling step must be positive')
    px = x + np.arange(width, dtype=np.float64) * pixel_step
    py = y + np.arange(height, dtype=np.float64) * pixel_step
    if band == 0:
        xs = np.floor((px - origin[0]) / scale).astype(np.int64)
        ys = np.floor((py - origin[1]) / scale).astype(np.int64)
        valid_x = (xs >= 0) & (xs < labels.shape[1])
        valid_y = (ys >= 0) & (ys < labels.shape[0])
        selected = labels[np.clip(ys, 0, labels.shape[0] - 1)[:, None],
                          np.clip(xs, 0, labels.shape[1] - 1)[None, :]] == index + 1
        return (selected & valid_y[:, None] & valid_x[None, :]).astype(np.float32)
    radius = int(math.ceil(band / scale)) + 3
    lx0 = max(0, int(math.floor((x - origin[0]) / scale)) - radius)
    ly0 = max(0, int(math.floor((y - origin[1]) / scale)) - radius)
    lx1 = min(labels.shape[1], int(math.ceil((px[-1] + pixel_step - origin[0]) / scale)) + radius)
    ly1 = min(labels.shape[0], int(math.ceil((py[-1] + pixel_step - origin[1]) / scale)) + radius)
    if lx1 <= lx0 or ly1 <= ly0:
        return np.zeros((height, width), np.float32)
    mask = (labels[ly0:ly1, lx0:lx1] == index + 1).astype(np.uint8)
    if not mask.any():
        return np.zeros((height, width), np.float32)
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    weight = np.clip(0.5 + (inside - outside) * (scale / (2.0 * band)), 0, 1)
    mx = (px - origin[0] + 0.5) / scale - 0.5 - lx0
    my = (py - origin[1] + 0.5) / scale - 0.5 - ly0
    return cv2.remap(weight, np.broadcast_to(mx[None, :], (height, width)).astype(np.float32),
                     np.broadcast_to(my[:, None], (height, width)).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
