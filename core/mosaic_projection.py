"""Point/tile panorama geometry; no image allocation, sampling, or UI dependencies.

Public angles are degrees. Coordinates address pixel centers, x right/y down;
world rays use the same axes with +z forward. R = Ry(yaw) Rx(pitch) Rz(roll)
maps camera rays into world rays (positive pitch therefore looks upward).
All point functions broadcast x/y, return float64 coordinates and boolean masks,
and put NaN in invalid coordinates. Callers choose their own tile sizes.

The radial model is PTGui's r_source = a*r**4+b*r**3+c*r**2+d*r,
d=1-a-b-c, normalized by half the shorter source dimension. Ideal pixel
coordinates use the unshifted image center; distortion then adds the source
pixel shifts. Only the strictly increasing branch attached to r=0 is usable.
Camera construction rejects models whose branch cannot cover the source frame;
points beyond that branch are explicitly invalid, even outside the frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from numbers import Integral
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

Cancel = Callable[[], bool] | None
_EPS = 1e-12


def _poll(cancel: Cancel) -> None:
    if cancel is not None and cancel():
        raise InterruptedError("Panorama geometry cancelled")


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class LensSpec:
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    shift_x: float = 0.0
    shift_y: float = 0.0

    def __post_init__(self) -> None:
        if not _finite(self.a, self.b, self.c, self.shift_x, self.shift_y):
            raise ValueError("Lens parameters must be finite")
        for name in ("a", "b", "c", "shift_x", "shift_y"):
            object.__setattr__(self, name, float(getattr(self, name)))
        derivative = (4 * self.a, 3 * self.b, 2 * self.c,
                      1.0 - self.a - self.b - self.c)
        if not _finite(*derivative):
            raise ValueError("Lens derivative coefficients must be finite")
        if 1.0 - self.a - self.b - self.c <= 0.0:
            raise ValueError("Lens derivative at the optical center must be positive")


def _radial(r: np.ndarray | float, lens: LensSpec):
    return ((lens.a * r + lens.b) * r + lens.c) * r * r + (
        1.0 - lens.a - lens.b - lens.c
    ) * r


def _branch(lens: LensSpec) -> tuple[float, float]:
    coefficients = np.trim_zeros(
        np.array([4 * lens.a, 3 * lens.b, 2 * lens.c,
                  1.0 - lens.a - lens.b - lens.c], dtype=np.float64), "f"
    )
    try:
        roots = np.roots(coefficients / np.max(np.abs(coefficients))) if coefficients.size > 1 else ()
    except np.linalg.LinAlgError as exc:
        raise ValueError("Lens radial branch is numerically indeterminate") from exc
    positive = [float(root.real) for root in roots
                if abs(root.imag) <= 1e-9 * max(1.0, abs(root.real))
                and root.real > 0.0]
    if not positive:
        return math.inf, math.inf
    limit = min(positive)
    source_limit = float(_radial(limit, lens))
    if not _finite(limit, source_limit) or source_limit <= 0:
        raise ValueError("Lens has no usable increasing radial branch")
    return limit, source_limit


@dataclass(frozen=True)
class CameraSpec:
    width: int
    height: int
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    hfov: float = 60.0
    lens: LensSpec = field(default_factory=LensSpec)
    _radius_limit: float = field(init=False, repr=False, compare=False)
    _source_limit: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (not isinstance(self.width, Integral) or isinstance(self.width, bool)
                or not isinstance(self.height, Integral) or isinstance(self.height, bool)
                or self.width < 2 or self.height < 2):
            raise ValueError("Camera dimensions must be integers >= 2")
        if not _finite(self.yaw, self.pitch, self.roll, self.hfov):
            raise ValueError("Camera angles must be finite")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        for name in ("yaw", "pitch", "roll", "hfov"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not 0.0 < self.hfov < 180.0:
            raise ValueError("Perspective camera hfov must lie strictly between 0 and 180")
        if not isinstance(self.lens, LensSpec):
            raise TypeError("lens must be a LensSpec")
        limit, source_limit = _branch(self.lens)
        cx, cy = self.center
        far_x = max(abs(cx + self.lens.shift_x),
                    abs(self.width - 1 - cx - self.lens.shift_x))
        far_y = max(abs(cy + self.lens.shift_y),
                    abs(self.height - 1 - cy - self.lens.shift_y))
        required = math.hypot(far_x, far_y) / self.radius_scale
        if required >= source_limit * (1.0 - 1e-12):
            raise ValueError("Noninvertible lens branch does not cover the source frame")
        if not math.isfinite(self.focal) or self.focal <= 0:
            raise ValueError("Camera focal length must be finite and positive")
        object.__setattr__(self, "_radius_limit", limit)
        object.__setattr__(self, "_source_limit", source_limit)

    @property
    def center(self) -> tuple[float, float]:
        return (self.width - 1) / 2.0, (self.height - 1) / 2.0

    @property
    def focal(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov) / 2.0)

    @property
    def radius_scale(self) -> float:
        return min(self.width, self.height) / 2.0


@dataclass(frozen=True)
class ProjectionSpec:
    kind: str = "rectilinear"
    focal: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("rectilinear", "cylindrical", "equirectangular"):
            raise ValueError("Unsupported projection: " + str(self.kind))
        if not _finite(self.focal, self.center_x, self.center_y) or self.focal <= 0:
            raise ValueError("Projection focal must be positive; center must be finite")
        for name in ("focal", "center_x", "center_y"):
            object.__setattr__(self, name, float(getattr(self, name)))


def rotation_matrix(camera: CameraSpec) -> np.ndarray:
    """Return a fresh camera-to-world 3x3 rotation, Ry @ Rx @ Rz."""
    yaw, pitch, roll = np.deg2rad([camera.yaw, camera.pitch, camera.roll])
    sy, cy = math.sin(yaw), math.cos(yaw)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sr, cr = math.sin(roll), math.cos(roll)
    return np.array([[cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp],
                     [cp * sr, cp * cr, -sp],
                     [-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp]])


def _angles(rotation: np.ndarray) -> tuple[float, float, float]:
    cos_pitch = math.hypot(rotation[1, 0], rotation[1, 1])
    pitch = math.atan2(-rotation[1, 2], cos_pitch)
    if cos_pitch > 1e-10:
        yaw = math.atan2(rotation[0, 2], rotation[2, 2])
        roll = math.atan2(rotation[1, 0], rotation[1, 1])
    else:
        yaw = math.atan2(-rotation[2, 0], rotation[0, 0])
        roll = 0.0
    return tuple(float(v) for v in np.rad2deg([yaw, pitch, roll]))


def intrinsic_matrix(camera: CameraSpec) -> np.ndarray:
    """Ideal pinhole intrinsics; homographies must use undistorted coordinates."""
    cx, cy = camera.center
    return np.array([[camera.focal, 0.0, cx], [0.0, camera.focal, cy], [0.0, 0.0, 1.0]])


def _xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    return np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                               np.asarray(y, dtype=np.float64))


def _masked_xy(x, y, valid):
    valid = valid & np.isfinite(x) & np.isfinite(y)
    return np.where(valid, x, np.nan), np.where(valid, y, np.nan), valid


def distort_points(x, y, camera: CameraSpec, *, cancel: Cancel = None):
    """Ideal pixels -> distorted source pixels; mask covers finite branch, not bounds."""
    _poll(cancel)
    x, y = _xy(x, y)
    cx, cy = camera.center
    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy) / camera.radius_scale
    valid = np.isfinite(r) & (r < camera._radius_limit)
    safe = np.where(valid, r, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        scale = ((camera.lens.a * safe + camera.lens.b) * safe
                 + camera.lens.c) * safe + 1 - camera.lens.a - camera.lens.b - camera.lens.c
        sx = cx + camera.lens.shift_x + dx * scale
        sy = cy + camera.lens.shift_y + dy * scale
    _poll(cancel)
    return _masked_xy(sx, sy, valid)


def undistort_points(x, y, camera: CameraSpec, *, cancel: Cancel = None):
    """Source pixels -> ideal pixels by bracketed inversion of the central branch."""
    _poll(cancel)
    x, y = _xy(x, y)
    cx, cy = camera.center
    dx = x - cx - camera.lens.shift_x
    dy = y - cy - camera.lens.shift_y
    source_r = np.hypot(dx, dy) / camera.radius_scale
    valid = np.isfinite(source_r) & (source_r < camera._source_limit)
    target = np.where(valid, source_r, 0.0)
    if camera.lens.a == camera.lens.b == camera.lens.c == 0.0:
        return _masked_xy(x - camera.lens.shift_x, y - camera.lens.shift_y, valid)
    if math.isfinite(camera._radius_limit):
        upper = camera._radius_limit
    else:
        upper = max(1.0, float(np.max(target, initial=0.0)))
        for _ in range(1024):
            _poll(cancel)
            with np.errstate(over="ignore", invalid="ignore"):
                extent = _radial(upper, camera.lens)
            if extent >= float(np.max(target, initial=0.0)):
                break
            upper *= 2.0
        else:
            raise ValueError("Unable to bracket radial inverse")
    lo = np.zeros_like(target)
    hi = np.full_like(target, upper)
    for _ in range(64):
        _poll(cancel)
        mid = lo + (hi - lo) / 2.0
        with np.errstate(over="ignore", invalid="ignore"):
            above = _radial(mid, camera.lens) >= target
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    radius = (lo + hi) / 2.0
    scale = np.ones_like(target)
    np.divide(radius, target, out=scale, where=target > 0)
    ux, uy = cx + dx * scale, cy + dy * scale
    # Do not claim success when a numerically pathological polynomial fails inversion.
    error = np.abs(_radial(radius, camera.lens) - target)
    valid &= error <= 1e-10 * np.maximum(1.0, target)
    return _masked_xy(ux, uy, valid)


def _unit(rays) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(rays, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("Rays must have shape (..., 3)")
    largest = np.max(np.abs(values), axis=-1)
    valid = np.isfinite(values).all(axis=-1) & (largest > 0)
    scaled = values / np.where(valid, largest, 1.0)[..., None]
    length = np.sqrt(np.sum(scaled * scaled, axis=-1))
    unit = scaled / np.where(valid, length, 1.0)[..., None]
    return np.where(valid[..., None], unit, np.nan), valid


def _in_frame(x, y, camera: CameraSpec):
    tolerance = 1e-8
    return ((x >= -tolerance) & (x <= camera.width - 1 + tolerance)
            & (y >= -tolerance) & (y <= camera.height - 1 + tolerance))


def camera_rays(x, y, camera: CameraSpec, *, cancel: Cancel = None):
    """Distorted source pixels -> unit world rays, with source-frame visibility mask."""
    x, y = _xy(x, y)
    ux, uy, valid = undistort_points(x, y, camera, cancel=cancel)
    cx, cy = camera.center
    rays, finite = _unit(np.stack([(ux - cx) / camera.focal,
                                   (uy - cy) / camera.focal, np.ones_like(ux)], axis=-1))
    valid &= finite & _in_frame(x, y, camera)
    rays = rays @ rotation_matrix(camera).T
    _poll(cancel)
    return np.where(valid[..., None], rays, np.nan), valid


def project(rays, projection: ProjectionSpec, *, cancel: Cancel = None):
    """World rays -> canvas pixels. Cylindrical/equirectangular seam is +/-pi."""
    _poll(cancel)
    unit, valid = _unit(rays)
    rx, ry, rz = unit[..., 0], unit[..., 1], unit[..., 2]
    horizontal = np.hypot(rx, rz)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if projection.kind == "rectilinear":
            valid &= rz > _EPS
            px, py = rx / rz, ry / rz
        else:
            valid &= horizontal > _EPS
            px = np.arctan2(rx, rz)
            py = (ry / horizontal if projection.kind == "cylindrical"
                  else np.arctan2(ry, horizontal))
        x = projection.center_x + projection.focal * px
        y = projection.center_y + projection.focal * py
    _poll(cancel)
    return _masked_xy(x, y, valid)


def unproject(x, y, projection: ProjectionSpec, *, cancel: Cancel = None):
    """Canvas pixels -> unit world rays; panorama x is restricted to [-pi*f, pi*f]."""
    _poll(cancel)
    x, y = _xy(x, y)
    with np.errstate(over="ignore", invalid="ignore"):
        px = (x - projection.center_x) / projection.focal
        py = (y - projection.center_y) / projection.focal
    valid = np.isfinite(px) & np.isfinite(py)
    if projection.kind == "rectilinear":
        rays = np.stack([px, py, np.ones_like(px)], axis=-1)
    else:
        valid &= np.abs(px) <= math.pi
        safe_x = np.where(valid, px, 0.0)
        if projection.kind == "cylindrical":
            rays = np.stack([np.sin(safe_x), py, np.cos(safe_x)], axis=-1)
        else:
            valid &= np.abs(py) < math.pi / 2
            safe_y = np.where(valid, py, 0.0)
            rays = np.stack([np.cos(safe_y) * np.sin(safe_x), np.sin(safe_y),
                             np.cos(safe_y) * np.cos(safe_x)], axis=-1)
    rays, finite = _unit(rays)
    valid &= finite
    _poll(cancel)
    return np.where(valid[..., None], rays, np.nan), valid


def source_coordinates(x, y, camera: CameraSpec, projection: ProjectionSpec,
                       *, cancel: Cancel = None):
    """Inverse map caller-supplied canvas points/tile to source (mx, my, valid)."""
    rays, valid = unproject(x, y, projection, cancel=cancel)
    local = rays @ rotation_matrix(camera)
    valid &= local[..., 2] > _EPS
    cx, cy = camera.center
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ux = cx + camera.focal * local[..., 0] / local[..., 2]
        uy = cy + camera.focal * local[..., 1] / local[..., 2]
    mx, my, lens_valid = distort_points(ux, uy, camera, cancel=cancel)
    valid &= lens_valid & _in_frame(mx, my, camera)
    _poll(cancel)
    return _masked_xy(mx, my, valid)


def _index(index, count: int) -> int:
    if not isinstance(index, Integral) or isinstance(index, bool) or not 0 <= index < count:
        raise ValueError("Camera index out of range")
    return int(index)


def _connected(count: int, edges, anchor: int) -> None:
    neighbors = [set() for _ in range(count)]
    for i, j in edges:
        neighbors[i].add(j)
        neighbors[j].add(i)
    seen, queue = {anchor}, [anchor]
    for i in queue:
        for j in sorted(neighbors[i]):
            if j not in seen:
                seen.add(j)
                queue.append(j)
    if len(seen) != count:
        raise ValueError("Camera graph must be connected to the exact anchor")


def seed_cameras(cameras: Sequence[CameraSpec],
                 homographies: Iterable[tuple[int, int, np.ndarray]], *, anchor: int = 0,
                 cancel: Cancel = None) -> tuple[CameraSpec, ...]:
    """Seed rotations from (i,j,H_ji), H_ji maps ideal image j pixels into i.

    Normalize K_i^-1 H_ji K_j, remove homogeneous scale/sign, take its
    closest proper rotation, then R_j=R_i R_ji. A sorted breadth-first tree
    makes graph/order handling deterministic. Translation/parallax is not
    estimated: a nonrotational homography only supplies a rotation initial guess.
    The anchor object, including its angles/FoV/lens, is returned unchanged.
    """
    _poll(cancel)
    cameras = tuple(cameras)
    anchor = _index(anchor, len(cameras))
    edges = []
    for i, j, matrix in homographies:
        _poll(cancel)
        i, j = _index(i, len(cameras)), _index(j, len(cameras))
        if i == j:
            raise ValueError("Homography edge must connect different cameras")
        h = np.array(matrix, dtype=np.float64, copy=True)
        if h.shape != (3, 3) or not np.isfinite(h).all():
            raise ValueError("Homography must be a finite 3x3 matrix")
        magnitude = float(np.max(np.abs(h)))
        if magnitude == 0:
            raise ValueError("Homography is singular")
        h /= magnitude
        a = np.linalg.solve(intrinsic_matrix(cameras[i]), h @ intrinsic_matrix(cameras[j]))
        determinant = float(np.linalg.det(a))
        if not math.isfinite(determinant) or determinant == 0:
            raise ValueError("Homography is singular")
        a *= math.copysign(1.0, determinant)
        u, singular, vt = np.linalg.svd(a)
        if singular[-1] <= singular[0] * 1e-12:
            raise ValueError("Homography is numerically singular")
        correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
        relative = u @ correction @ vt
        edges.append((i, j, relative))
    _connected(len(cameras), [(i, j) for i, j, _ in edges], anchor)
    edges.sort(key=lambda e: (e[0], e[1], tuple(e[2].ravel())))
    graph = [[] for _ in cameras]
    for i, j, relative in edges:
        graph[i].append((j, relative))
        graph[j].append((i, relative.T))
    rotations = {anchor: rotation_matrix(cameras[anchor])}
    queue = [anchor]
    for i in queue:
        _poll(cancel)
        for j, relative in sorted(graph[i], key=lambda e: (e[0], tuple(e[1].ravel()))):
            if j not in rotations:
                rotations[j] = rotations[i] @ relative
                queue.append(j)
    result = list(cameras)
    for i in range(len(cameras)):
        if i != anchor:
            yaw, pitch, roll = _angles(rotations[i])
            result[i] = replace(cameras[i], yaw=yaw, pitch=pitch, roll=roll)
    _poll(cancel)
    return tuple(result)


@dataclass(frozen=True, eq=False)
class PairObservation:
    """Matched distorted source coordinates, copied to immutable Nx2 snapshots."""
    i: int
    j: int
    points_i: np.ndarray
    points_j: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("points_i", "points_j"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True, order="C")
            if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
                raise ValueError("Observation points must be finite Nx2 arrays")
            # Immutable bytes backing prevents callers from re-enabling writeability.
            snapshot = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
            object.__setattr__(self, name, snapshot)
        if self.points_i.shape != self.points_j.shape or len(self.points_i) < 3:
            raise ValueError("Each pair needs at least three matched points")
        weights = (np.ones(len(self.points_i), np.float64) if self.weights is None
                   else np.asarray(self.weights, np.float64))
        if (weights.shape != (len(self.points_i),) or not np.isfinite(weights).all()
                or np.any(weights <= 0)):
            raise ValueError('Camera observation weights must be finite and positive')
        object.__setattr__(self, 'weights',
                           np.frombuffer(weights.tobytes(), np.float64))


@dataclass(frozen=True)
class CameraFitResult:
    cameras: tuple[CameraSpec, ...]
    initial_rms: float
    final_rms: float
    nfev: int
    cost: float
    optimality: float
    message: str


def fit_cameras(cameras: Sequence[CameraSpec],
                observations: Iterable[PairObservation | tuple], *, anchor: int = 0,
                optimize_fov: bool = False,
                rotation_bounds=((-180.0, -89.0, -180.0), (180.0, 89.0, 180.0)),
                fov_bounds=(5.0, 175.0), loss: str = "soft_l1", f_scale: float = 2.0,
                max_nfev: int = 300, cancel: Cancel = None) -> CameraFitResult:
    """Robust bounded least-squares fit of matched world-ray chord residuals.

    Rotations optimize degree offsets from the supplied seeds within the explicit
    rotation_bounds (three lower/upper offsets). Optional FoVs use absolute degree
    fov_bounds. The anchor's entire spec, including FoV, is fixed exactly, removing
    the rotational gauge and anchoring focal calibration. Lens/dimensions are fixed.
    Each ray difference is scaled by the geometric mean of the pair's *initial*
    focal lengths: RMS reports sqrt(mean squared chord length), in approximate
    pixel units, across matches. No CP detection or feature allocation is performed.
    Disconnected/empty graphs, unobservable fits and nonconvergence raise errors.
    """
    _poll(cancel)
    cameras = tuple(cameras)
    anchor = _index(anchor, len(cameras))
    if len(cameras) < 2:
        raise ValueError("Camera fitting requires at least two cameras")
    if loss not in ("soft_l1", "huber", "cauchy", "arctan"):
        raise ValueError("Select a supported robust least-squares loss")
    if not _finite(f_scale) or f_scale <= 0:
        raise ValueError("f_scale must be positive and finite")
    if not isinstance(max_nfev, Integral) or max_nfev <= 0:
        raise ValueError("max_nfev must be a positive integer")
    bounds = np.array(rotation_bounds, dtype=np.float64, copy=True)
    fov = np.array(fov_bounds, dtype=np.float64, copy=True)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or not np.all(bounds[0] < 0) or not np.all(bounds[1] > 0)):
        raise ValueError("Rotation offset bounds must strictly contain zero")
    if (fov.shape != (2,) or not np.isfinite(fov).all()
            or not 0 < fov[0] < fov[1] < 180):
        raise ValueError("FoV bounds must satisfy 0 < lower < upper < 180")
    free = [i for i in range(len(cameras)) if i != anchor]
    stride = 4 if optimize_fov else 3
    slots = {i: k * stride for k, i in enumerate(free)}
    lower, upper, initial = [], [], []
    for i in free:
        lower.extend(bounds[0])
        upper.extend(bounds[1])
        initial.extend([0.0, 0.0, 0.0])
        if optimize_fov:
            if not fov[0] < cameras[i].hfov < fov[1]:
                raise ValueError("Initial free-camera FoVs must lie strictly inside bounds")
            lower.append(fov[0])
            upper.append(fov[1])
            initial.append(cameras[i].hfov)
    pairs, prepared, row_count = [], [], 0
    for observation in observations:
        _poll(cancel)
        obs = observation if isinstance(observation, PairObservation) else PairObservation(*observation)
        i, j = _index(obs.i, len(cameras)), _index(obs.j, len(cameras))
        if i == j:
            raise ValueError("Observation must connect different cameras")
        ideal = []
        for index, points in ((i, obs.points_i), (j, obs.points_j)):
            cam = cameras[index]
            ux, uy, valid = undistort_points(points[:, 0], points[:, 1], cam, cancel=cancel)
            if not np.all(valid & _in_frame(points[:, 0], points[:, 1], cam)):
                raise ValueError("Observation lies outside source frame or valid lens branch")
            ideal.append(np.column_stack([ux - cam.center[0], uy - cam.center[1]]))
        count = len(obs.points_i)
        scale = math.sqrt(cameras[i].focal) * math.sqrt(cameras[j].focal)
        prepared.append((i, j, ideal[0], ideal[1], scale, row_count, count,
                         np.sqrt(obs.weights)))
        pairs.append((i, j))
        row_count += 3 * count
    _connected(len(cameras), pairs, anchor)
    if not prepared or row_count < len(initial):
        raise ValueError("Insufficient control point observations")
    sparsity = lil_matrix((row_count, len(initial)), dtype=np.int8)
    for i, j, _, _, _, start, count, _weights in prepared:
        for index in (i, j):
            if index in slots:
                k = slots[index]
                sparsity[start:start + 3 * count, k:k + stride] = 1

    def decode(parameters):
        result = list(cameras)
        for index in free:
            k = slots[index]
            result[index] = replace(
                cameras[index], yaw=cameras[index].yaw + parameters[k],
                pitch=cameras[index].pitch + parameters[k + 1],
                roll=cameras[index].roll + parameters[k + 2],
                hfov=parameters[k + 3] if optimize_fov else cameras[index].hfov)
        return tuple(result)

    def residual(parameters):
        _poll(cancel)
        current = decode(parameters)
        rotations = [rotation_matrix(cam) for cam in current]
        output = np.empty(row_count, dtype=np.float64)
        for i, j, pi, pj, scale, start, count, weights in prepared:
            _poll(cancel)
            ri, _ = _unit(np.column_stack([pi / current[i].focal, np.ones(count)]))
            rj, _ = _unit(np.column_stack([pj / current[j].focal, np.ones(count)]))
            output[start:start + 3 * count] = (
                (ri @ rotations[i].T - rj @ rotations[j].T)
                * (scale * weights[:, None])).ravel()
        return output

    initial = np.array(initial, dtype=np.float64)
    before = residual(initial)
    solved = least_squares(residual, initial, bounds=(lower, upper),
                           jac_sparsity=sparsity.tocsr(), loss=loss, f_scale=f_scale,
                           max_nfev=int(max_nfev), xtol=1e-10, ftol=1e-10, gtol=1e-10)
    _poll(cancel)
    if not solved.success or not np.isfinite(solved.x).all():
        raise RuntimeError("Camera fit failed: " + str(solved.message))
    # Small parameter-space Gram matrix avoids materializing a dense CP Jacobian.
    gram = (solved.jac.T @ solved.jac).toarray()
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues[-1] <= 0 or eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        raise ValueError("Camera fit is unobservable or numerically rank deficient")
    after = residual(solved.x)
    _poll(cancel)
    return CameraFitResult(
        decode(solved.x), math.sqrt(float(before @ before) / (row_count / 3)),
        math.sqrt(float(after @ after) / (row_count / 3)), int(solved.nfev),
        float(solved.cost), float(solved.optimality), str(solved.message))
