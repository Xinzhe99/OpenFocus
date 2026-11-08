"""Independent planar control-point fitting; no image, canvas, or UI ownership.

Coordinates are source pixel centers and transforms are 3x3, orientation-preserving
similarities mapping those centers into canvas pixels. ``dimensions``, when given,
is one (width, height) integer pair per frame. Overlap uses transformed pixel
footprints [-.5, width-.5] x [-.5, height-.5], divided by the smaller polygon area.
Only pairs represented in the supplied control points appear in ``qa.pairs``.

Four enabled points with two-dimensional support are required for an edge. The
requested anchor is exact; every other component fixes its lowest-index frame to
its exact starting placement. Insufficient/degenerate edges preserve placements
unless another supported edge connects the frame. Connectivity describes this
support graph, independently of whether a starting placement already aligns.

The fit is a joint sparse LSMR similarity solve with Cauchy IRLS and multiplicative
per-point weights. A deterministic, bounded two-point hypothesis search initializes
edges before joint fitting. This robust estimator needs a dominant consistent
population; it cannot identify an intended solution from arbitrary contradictory
or majority-outlier correspondences. Inliers are enabled CPs with final canvas
residual <= 2 pixels. A pair is weak for insufficient/degenerate support, fewer than
four inliers, or p95 > 2 pixels. Disabled points retain measured residuals but do not
contribute to fitting, statistics, inliers, or connectivity. No hole/coverage-raster
statistics are synthesized. Cancellation raises InterruptedError, including from
LSMR's matrix-vector products. Inputs are never mutated; published arrays own
immutable byte-backed storage rather than borrowing caller arrays.
"""

from dataclasses import dataclass
from itertools import combinations
import math
import operator
from typing import Callable, Sequence

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, lsmr


__all__ = [
    "MosaicControlPoint", "MosaicPairQa", "MosaicQa", "ControlPointFit",
    "optimize_control_points", "measure_control_points",
]

_INLIER_PIXELS = 2.0


def _index(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer index")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer index") from error


def _point(value):
    result = tuple(float(x) for x in value)
    if len(result) != 2 or not all(math.isfinite(x) for x in result):
        raise ValueError("Control-point coordinates must contain two finite numbers")
    return result


@dataclass(frozen=True, slots=True)
class MosaicControlPoint:
    frame_a: int
    frame_b: int
    point_a: tuple[float, float]
    point_b: tuple[float, float]
    source: str = "manual"
    enabled: bool = True
    weight: float = 2.0

    def __post_init__(self):
        a, b = _index(self.frame_a, "frame_a"), _index(self.frame_b, "frame_b")
        if a < 0 or b < 0 or a == b:
            raise ValueError("Control points need two distinct nonnegative frame indices")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Control-point source must be a nonempty string")
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise ValueError("Control-point enabled must be boolean")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Control-point weight must be finite and positive")
        object.__setattr__(self, "frame_a", a)
        object.__setattr__(self, "frame_b", b)
        object.__setattr__(self, "point_a", _point(self.point_a))
        object.__setattr__(self, "point_b", _point(self.point_b))
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class MosaicPairQa:
    frame_a: int
    frame_b: int
    count: int                       # Enabled CP count, including outliers.
    median: float                    # Enabled CP residuals; NaN when count == 0.
    p95: float
    inlier_indices: tuple[int, ...]   # Indices in the original supplied CP order.
    weak: bool
    reasons: tuple[str, ...]
    overlap_fraction: float | None


@dataclass(frozen=True, slots=True)
class MosaicQa:
    residuals: tuple[float, ...]      # All supplied CPs, including disabled CPs.
    pairs: tuple[MosaicPairQa, ...]   # Sorted by canonical (frame_a, frame_b).
    connected: tuple[bool, ...]      # Reachable from the requested anchor.
    disconnected_indices: tuple[int, ...]
    component_anchors: tuple[int, ...]
    inlier_indices: tuple[int, ...]
    cp_count: int                     # Total enabled CPs.
    residual_median: float
    residual_p95: float


@dataclass(frozen=True, slots=True)
class ControlPointFit:
    transforms: tuple[np.ndarray, ...]
    qa: MosaicQa


@dataclass(frozen=True, slots=True)
class _Pair:
    a: int
    b: int
    indices: tuple[int, ...]
    enabled: tuple[int, ...]
    points_a: np.ndarray
    points_b: np.ndarray
    weights: np.ndarray
    supported: bool
    reason: str | None


def _cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise InterruptedError("Control-point operation cancelled")


def _immutable_matrix(matrix):
    return np.frombuffer(np.asarray(matrix, dtype=np.float64).tobytes(),
                         dtype=np.float64).reshape(3, 3)


def _prepare(transforms, control_points, dimensions, anchor, should_cancel):
    _cancel(should_cancel)
    matrices = []
    for value in transforms:
        _cancel(should_cancel)
        matrix = np.array(value, dtype=np.float64, copy=True)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Each placement must be a finite 3x3 similarity")
        if not np.array_equal(matrix[2], (0.0, 0.0, 1.0)):
            raise ValueError("Placements must have homogeneous row (0, 0, 1)")
        a, b = float(matrix[0, 0]), float(matrix[1, 0])
        if (not np.allclose(matrix[:2, :2], ((a, -b), (b, a)), rtol=1e-12, atol=1e-12)
                or not math.isfinite(math.hypot(a, b)) or math.hypot(a, b) <= 1e-12):
            raise ValueError("Placements must be nonsingular orientation-preserving similarities")
        matrices.append(matrix)
    anchor = _index(anchor, "anchor")
    if (matrices and not 0 <= anchor < len(matrices)) or (not matrices and anchor != 0):
        raise ValueError("Anchor must reference a supplied frame (or be 0 for no frames)")
    points = tuple(control_points)
    grouped = {}
    for index, cp in enumerate(points):
        _cancel(should_cancel)
        if not isinstance(cp, MosaicControlPoint):
            raise TypeError("control_points must contain MosaicControlPoint instances")
        if max(cp.frame_a, cp.frame_b) >= len(matrices):
            raise ValueError("Control point references an absent frame")
        grouped.setdefault(tuple(sorted((cp.frame_a, cp.frame_b))), []).append(index)
    sizes = None
    if dimensions is not None:
        sizes = []
        for size in dimensions:
            values = tuple(size)
            if len(values) != 2:
                raise ValueError("Dimensions must be (width, height) pairs")
            width, height = (_index(v, "dimension") for v in values)
            if width <= 0 or height <= 0:
                raise ValueError("Dimensions must be positive")
            sizes.append((width, height))
        if len(sizes) != len(matrices):
            raise ValueError("Each placement needs one (width, height) dimension pair")
        sizes = tuple(sizes)
    pairs = []
    for (a, b), indices in sorted(grouped.items()):
        _cancel(should_cancel)
        enabled = tuple(i for i in indices if points[i].enabled)
        pa, pb, weights = [], [], []
        for i in enabled:
            cp = points[i]
            pa.append(cp.point_a if cp.frame_a == a else cp.point_b)
            pb.append(cp.point_b if cp.frame_b == b else cp.point_a)
            weights.append(cp.weight)
        pa = np.asarray(pa, dtype=np.float64).reshape(-1, 2)
        pb = np.asarray(pb, dtype=np.float64).reshape(-1, 2)
        reason = ("insufficient_points" if len(enabled) < 4 else
                  "degenerate_support" if not (_has_area(pa) and _has_area(pb)) else None)
        pairs.append(_Pair(a, b, tuple(indices), enabled, pa, pb,
                           np.asarray(weights, dtype=np.float64), reason is None, reason))
    return matrices, points, tuple(pairs), sizes, anchor


def _has_area(points):
    """Reject collapsed, collinear, and numerically near-collinear support."""
    if len(points) < 4:
        return False
    centered = points - np.mean(points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    # The absolute floor is relative to source-coordinate precision, not image size.
    floor = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(points)))) * 64
    return bool(singular[0] > floor and singular[1] > max(floor, singular[0] * 1e-4))


def _components(count, pairs, anchor, should_cancel):
    adjacency = [[] for _ in range(count)]
    for pair in pairs:
        if pair.supported:
            adjacency[pair.a].append(pair.b)
            adjacency[pair.b].append(pair.a)
    components = []
    seen = set()
    order = ([anchor] + [i for i in range(count) if i != anchor]) if count else []
    for root in order:
        _cancel(should_cancel)
        if root in seen:
            continue
        seen.add(root)
        nodes, pending = [], [root]
        while pending:
            _cancel(should_cancel)
            node = pending.pop()
            nodes.append(node)
            for other in adjacency[node]:
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        components.append((root, tuple(sorted(nodes))))
    return tuple(components)


def _project(matrix, points):
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def _seed_pair(pair, should_cancel):
    """Deterministic least-median initialization, mapping frame b into frame a."""
    pa, pb = pair.points_a, pair.points_b
    za, zb = pa[:, 0] + 1j * pa[:, 1], pb[:, 0] + 1j * pb[:, 1]
    count = len(pa)
    if count <= 32:
        candidates = combinations(range(count), 2)
    else:
        # Local generator leaves NumPy/OpenCV global random state untouched.
        candidates = np.random.default_rng(0).integers(0, count, size=(512, 2))
    best = None
    floor = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(zb)))) * 64
    for i, j in candidates:
        _cancel(should_cancel)
        delta = zb[j] - zb[i]
        if abs(delta) <= floor:
            continue
        coefficient = (za[j] - za[i]) / delta
        if not np.isfinite(coefficient) or abs(coefficient) <= 1e-12:
            continue
        offset = za[i] - coefficient * zb[i]
        squared = np.abs(coefficient * zb + offset - za) ** 2
        median = float(np.median(squared))
        score = (median, float(np.mean(np.minimum(squared, max(4.0, median * 4)))))
        if best is None or score < best[0]:
            best = (score, coefficient, offset, squared)
    if best is None:
        raise ValueError("Supported control points do not yield a finite similarity seed")
    _, coefficient, offset, squared = best
    # Refining the consensus improves graph propagation before the global solve.
    keep = squared <= max(4.0, float(np.median(squared)) * 6.25)
    w = pair.weights[keep]
    w = w / np.max(w)
    aa, bb = za[keep], zb[keep]
    am, bm = np.sum(w * aa) / np.sum(w), np.sum(w * bb) / np.sum(w)
    denominator = float(np.sum(w * np.abs(bb - bm) ** 2))
    if denominator > floor * floor:
        refined = np.sum(w * np.conjugate(bb - bm) * (aa - am)) / denominator
        if np.isfinite(refined) and abs(refined) > 1e-12:
            coefficient, offset = refined, am - refined * bm
    a, b = float(coefficient.real), float(coefficient.imag)
    return np.array(((a, -b, offset.real), (b, a, offset.imag), (0.0, 0.0, 1.0)))


def _fit_component(result, root, nodes, pairs, should_cancel):
    if len(nodes) == 1:
        return
    members = set(nodes)
    edges = [p for p in pairs if p.supported and p.a in members]
    edges.sort(key=lambda p: (-len(p.enabled), p.a, p.b))
    seeds = [_seed_pair(p, should_cancel) for p in edges]
    placed = {root}
    while len(placed) < len(nodes):
        _cancel(should_cancel)
        for pair, seed in zip(edges, seeds):
            if pair.a in placed and pair.b not in placed:
                result[pair.b] = result[pair.a] @ seed
                placed.add(pair.b)
            elif pair.b in placed and pair.a not in placed:
                result[pair.a] = result[pair.b] @ np.linalg.inv(seed)
                placed.add(pair.a)
    observations = {node: [] for node in nodes}
    for pair in edges:
        observations[pair.a].append(pair.points_a)
        observations[pair.b].append(pair.points_b)
    centers = {node: np.mean(np.concatenate(observations[node]), axis=0) for node in nodes}
    unit = max(1.0, max(float(np.max(np.ptp(np.concatenate(observations[node]), axis=0)))
                        for node in nodes))
    origin = _project(result[root], centers[root])
    slots = {node: 4 * i for i, node in enumerate(n for n in nodes if n != root)}
    rows, columns, values, targets, base_weights = [], [], [], [], []
    row = 0
    for pair in edges:
        _cancel(should_cancel)
        n = len(pair.enabled)
        fixed = np.zeros((n, 2), dtype=np.float64)
        for node, points, sign in ((pair.a, pair.points_a, 1.0),
                                   (pair.b, pair.points_b, -1.0)):
            centered = points - centers[node]
            if node == root:
                fixed += sign * (centered @ result[root][:2, :2].T) / unit
                continue
            normalized = centered / unit
            offset = slots[node]
            for axis in (0, 1):
                coefficients = ((normalized[:, 0], -normalized[:, 1], np.ones(n))
                                if axis == 0 else
                                (normalized[:, 1], normalized[:, 0], np.ones(n)))
                for column, coefficient in zip((offset, offset + 1, offset + 2 + axis),
                                                coefficients):
                    rows.append(row + np.arange(n, dtype=np.int64) * 2 + axis)
                    columns.append(np.full(n, column, dtype=np.int64))
                    values.append(sign * coefficient)
        targets.append((-fixed).ravel())
        base_weights.append(pair.weights)
        row += n * 2
    design = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                        shape=(row, 4 * len(slots))).tocsr()
    target = np.concatenate(targets)
    weights = np.concatenate(base_weights)
    weights /= np.max(weights)
    current = np.concatenate([np.array((result[node][0, 0], result[node][1, 0],
                                        *((_project(result[node], centers[node]) - origin) / unit)))
                              for node in slots])
    for _ in range(60):
        _cancel(should_cancel)
        error = target - design @ current
        distance = np.linalg.norm(error.reshape(-1, 2), axis=1) * unit
        loss_scale = max(0.25, min(2.0, 2.5 * float(np.median(distance))))
        row_weights = np.repeat(np.sqrt(weights / (1.0 + (distance / loss_scale) ** 2)), 2)

        def matvec(vector):
            _cancel(should_cancel)
            return row_weights * (design @ vector)

        def rmatvec(vector):
            _cancel(should_cancel)
            return design.T @ (row_weights * vector)

        weighted = LinearOperator(design.shape, matvec=matvec, rmatvec=rmatvec, dtype=np.float64)
        solution = lsmr(weighted, error * row_weights, atol=1e-12, btol=1e-12,
                        conlim=1e12, maxiter=max(200, 20 * len(current)))
        _cancel(should_cancel)
        if solution[1] not in (0, 1, 2, 4, 5):
            raise ValueError(f"Control-point LSMR did not converge (stop code {solution[1]})")
        delta = solution[0]
        current += delta
        if not np.isfinite(current).all():
            raise ValueError("Control-point fit produced nonfinite similarity parameters")
        if np.max(np.abs(delta)) <= 1e-11:
            break
    else:
        raise ValueError("Robust control-point fit did not converge in 60 iterations")
    fitted = {}
    for node, offset in slots.items():
        a, b, x, y = current[offset:offset + 4]
        if math.hypot(a, b) <= 1e-12:
            raise ValueError("Control-point fit collapsed a frame to a singular similarity")
        linear = np.array(((a, -b), (b, a)))
        translation = origin + unit * np.array((x, y)) - linear @ centers[node]
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :2], matrix[:2, 2] = linear, translation
        fitted[node] = matrix
    result.update(fitted)


def _overlap(first, second, size_a, size_b):
    def polygon(matrix, size):
        width, height = size
        corners = np.array(((-0.5, -0.5), (width - 0.5, -0.5),
                            (width - 0.5, height - 0.5), (-0.5, height - 0.5)))
        return _project(matrix, corners)

    a, b = polygon(first, size_a), polygon(second, size_b)
    # Removing the common canvas translation protects float32 OpenCV geometry.
    origin = a[0].copy()
    a, b = np.asarray(a - origin, np.float32), np.asarray(b - origin, np.float32)
    denominator = min(abs(cv2.contourArea(a)), abs(cv2.contourArea(b)))
    if denominator <= 0:
        raise ValueError("Frame footprint is numerically degenerate for OpenCV overlap")
    area, _ = cv2.intersectConvexConvex(a, b, handleNested=True)
    return float(np.clip(area / denominator, 0.0, 1.0))


def _qa(matrices, points, pairs, sizes, components, anchor, should_cancel):
    residuals = []
    for cp in points:
        _cancel(should_cancel)
        residuals.append(float(np.linalg.norm(_project(matrices[cp.frame_a], np.asarray(cp.point_a))
                                              - _project(matrices[cp.frame_b], np.asarray(cp.point_b)))))
    records = []
    all_inliers = []
    for pair in pairs:
        _cancel(should_cancel)
        distances = [residuals[i] for i in pair.enabled]
        median = float(np.median(distances)) if distances else math.nan
        p95 = float(np.percentile(distances, 95)) if distances else math.nan
        inliers = tuple(i for i in pair.enabled if residuals[i] <= _INLIER_PIXELS)
        all_inliers.extend(inliers)
        reasons = [pair.reason] if pair.reason is not None else []
        if len(inliers) < 4:
            reasons.append("insufficient_inliers")
        else:
            # Reversed CP records must use the same canonical source frames.
            keep = np.asarray([i in inliers for i in pair.enabled], dtype=bool)
            if not (_has_area(pair.points_a[keep]) and _has_area(pair.points_b[keep])):
                reasons.append("degenerate_inliers")
        if math.isfinite(p95) and p95 > _INLIER_PIXELS:
            reasons.append("high_residual")
        overlap = None if sizes is None else _overlap(matrices[pair.a], matrices[pair.b],
                                                     sizes[pair.a], sizes[pair.b])
        records.append(MosaicPairQa(pair.a, pair.b, len(pair.enabled), median, p95,
                                    inliers, bool(reasons), tuple(reasons), overlap))
    connected = [False] * len(matrices)
    for root, nodes in components:
        if root == anchor:
            for node in nodes:
                connected[node] = True
    enabled_distances = [residuals[i] for i, cp in enumerate(points) if cp.enabled]
    _cancel(should_cancel)
    return MosaicQa(tuple(residuals), tuple(records), tuple(connected),
                    tuple(i for i, value in enumerate(connected) if not value),
                    tuple(root for root, _ in components), tuple(sorted(all_inliers)),
                    len(enabled_distances),
                    float(np.median(enabled_distances)) if enabled_distances else math.nan,
                    float(np.percentile(enabled_distances, 95)) if enabled_distances else math.nan)


def optimize_control_points(
    transforms: Sequence[np.ndarray], control_points: Sequence[MosaicControlPoint], *,
    dimensions: Sequence[tuple[int, int]] | None = None, anchor: int = 0,
    should_cancel: Callable[[], bool] | None = None,
) -> ControlPointFit:
    """Jointly fit supported CP components; return owned immutable placements/QA.

    Invalid input raises ValueError (wrong CP object types raise TypeError).
    Numerical nonconvergence or a singular fitted placement raises ValueError;
    no failed fit is silently published as successful. No input is mutated.
    """
    matrices, points, pairs, sizes, anchor = _prepare(
        transforms, control_points, dimensions, anchor, should_cancel)
    components = _components(len(matrices), pairs, anchor, should_cancel)
    result = dict(enumerate(matrices))
    for root, nodes in components:
        _fit_component(result, root, nodes, pairs, should_cancel)
    output = tuple(_immutable_matrix(result[i]) for i in range(len(matrices)))
    qa = _qa(output, points, pairs, sizes, components, anchor, should_cancel)
    return ControlPointFit(output, qa)


def measure_control_points(
    transforms: Sequence[np.ndarray], control_points: Sequence[MosaicControlPoint], *,
    dimensions: Sequence[tuple[int, int]] | None = None, anchor: int = 0,
    should_cancel: Callable[[], bool] | None = None,
) -> MosaicQa:
    """Measure original-order canvas residuals and support connectivity, without fitting."""
    matrices, points, pairs, sizes, anchor = _prepare(
        transforms, control_points, dimensions, anchor, should_cancel)
    components = _components(len(matrices), pairs, anchor, should_cancel)
    return _qa(matrices, points, pairs, sizes, components, anchor, should_cancel)
