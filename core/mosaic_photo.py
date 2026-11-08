"""Bounded photometric fitting and tile ICC conversion, independent of Qt.

Working colors are normalized BGR/BGRA. ``apply_photometric_tile`` returns
linear colors when spec.linear is true: blend in that domain, then call
``linear_to_srgb`` once. Alpha is always copied unchanged. No image loading,
warping, blending, quantization, or export is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import math
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from utils import image_utils


Cancel = Callable[[], bool] | None


def _cancel(should_cancel: Cancel) -> None:
    if should_cancel is not None and should_cancel():
        raise InterruptedError("Photometric operation cancelled")


def _immutable(array: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Own immutable storage; callers cannot re-enable writeability."""
    result = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


@dataclass(frozen=True)
class PhotometricSpec:
    """All gains, including manual correction, are clipped to [0.5, 2].

    Modes: off disables fitted gains; exposure fits one shared BGR gain;
    color fits independent constant channel gains; gain_field fits six terms
    per channel [1,u,v,u*u,u*v,v*v]. Each connected component is anchored to
    reference_index if present, otherwise its smallest frame index. Isolated
    frames are their own anchors. Regularization penalizes nonconstant terms.

    vignette_coefficients contains one (k1,k2,k3) per frame, or is empty.
    Manual correction is 1/(1+k1*r²+k2*r⁴+k3*r⁶), r²=u²+v², in the selected
    working domain. It operates even in off mode; empty means no correction.
    Anchoring applies to fitted coefficients, not explicitly requested manual
    corrections. Coefficients must describe positive attenuation on [0,2].
    """
    mode: str = "off"
    reference_index: int = 0
    linear: bool = False
    regularization: float = 1e-4
    huber_delta: float = 0.08
    robust_iterations: int = 2
    max_samples_per_observation: int = 2048
    max_total_samples: int = 65536
    min_signal: float = 1e-4
    solver_tolerance: float = 1e-8
    solver_iterations: int = 1000
    vignette_coefficients: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self):
        if self.mode not in {"off", "exposure", "color", "gain_field"}:
            raise ValueError("Unknown photometric mode")
        object.__setattr__(self, "reference_index", _integer(self.reference_index, "reference_index", 0))
        if not isinstance(self.linear, (bool, np.bool_)):
            raise ValueError("linear must be boolean")
        for name, minimum in (("robust_iterations", 0), ("max_samples_per_observation", 1),
                              ("max_total_samples", 1), ("solver_iterations", 1)):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        for name in ("regularization", "huber_delta", "min_signal", "solver_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0 or (name != "regularization" and value == 0):
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, value)
        if self.min_signal >= 1 or self.solver_tolerance >= 1:
            raise ValueError("min_signal and solver_tolerance must be below one")
        coefficients = tuple(tuple(float(k) for k in row) for row in self.vignette_coefficients)
        for row in coefficients:
            if len(row) != 3 or not all(math.isfinite(k) for k in row):
                raise ValueError("Vignetting requires finite (k1,k2,k3) triples")
            k1, k2, k3 = row
            critical = [0.0, 2.0]
            derivative = np.trim_zeros(np.array([3 * k3, 2 * k2, k1]), "f")
            if len(derivative) > 1:
                critical.extend(float(root.real) for root in np.roots(derivative)
                                if abs(root.imag) < 1e-12 and 0 < root.real < 2)
            if any(1 + t * (k1 + t * (k2 + t * k3)) <= 0 for t in critical):
                raise ValueError("Vignetting attenuation must remain positive on the source")
        object.__setattr__(self, "vignette_coefficients", coefficients)


@dataclass(frozen=True, eq=False)
class PhotoObservation:
    """Owned immutable matched samples. Weights are optional nonnegative N-vectors.

    Samples must already exclude masks and geometric mismatches. Zero weights
    remove samples. Fitting subsamples deterministically before constructing
    any sparse system; it never accepts source images.
    """
    frame_a: int
    frame_b: int
    uv_a: np.ndarray
    uv_b: np.ndarray
    colors_a: np.ndarray
    colors_b: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "frame_a", _integer(self.frame_a, "frame_a", 0))
        object.__setattr__(self, "frame_b", _integer(self.frame_b, "frame_b", 0))
        if self.frame_a == self.frame_b:
            raise ValueError("An overlap must join distinct frames")
        n = None
        for name, channels, low in (("uv_a", 2, -1), ("uv_b", 2, -1),
                                    ("colors_a", 3, 0), ("colors_b", 3, 0)):
            array = np.asarray(getattr(self, name))
            if array.ndim != 2 or array.shape[1] != channels:
                raise ValueError(f"{name} must be Nx{channels}")
            if n is None:
                n = len(array)
            if len(array) != n or not np.all(np.isfinite(array)) or np.any(array < low) or np.any(array > 1):
                raise ValueError(f"Invalid {name} sample values or count")
            object.__setattr__(self, name, _immutable(array))
        if self.weights is not None:
            weights = np.asarray(self.weights)
            if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("weights must be a finite nonnegative N-vector")
            object.__setattr__(self, "weights", _immutable(weights))


def _transfer(values, inverse: bool, preserve_alpha: bool) -> np.ndarray:
    source = np.asarray(values)
    if not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source):
        raise ValueError("Transfer input must be real numeric values")
    result = np.array(source, dtype=np.float64 if source.dtype == np.float64 else np.float32, copy=True)
    color = result[..., :3] if preserve_alpha and result.ndim and result.shape[-1] == 4 else result
    if not np.all(np.isfinite(color)):
        raise ValueError("Transfer colors must be finite")
    # Extend the linear branch below zero; do not clip HDR blend results.
    if inverse:
        mask = color <= 0.0031308
        color[mask] *= 12.92
        color[~mask] = 1.055 * np.power(color[~mask], 1 / 2.4) - 0.055
    else:
        mask = color <= 0.04045
        color[mask] /= 12.92
        color[~mask] = np.power((color[~mask] + 0.055) / 1.055, 2.4)
    return result


def srgb_to_linear(values, *, preserve_alpha: bool = True) -> np.ndarray:
    """Decode normalized sRGB; last-channel alpha in BGRA is unchanged.

    Set preserve_alpha=False for arbitrary sample vectors or transfer LUTs.
    Integer source pixels must first be normalized, or indexed into the LUT.
    """
    return _transfer(values, False, preserve_alpha)


def linear_to_srgb(values, *, preserve_alpha: bool = True) -> np.ndarray:
    """Encode after linear blending; preserve BGRA alpha and do not quantize."""
    return _transfer(values, True, preserve_alpha)


@lru_cache(maxsize=2)
def srgb_eotf_lut(dtype) -> np.ndarray:
    """Immutable float32 decoding table for uint8 or uint16 pixel values."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("sRGB LUT supports uint8 and uint16")
    maximum = np.iinfo(dtype).max
    values = np.arange(maximum + 1, dtype=np.float64) / maximum
    return _immutable(srgb_to_linear(values, preserve_alpha=False), np.float32)


def _basis(uv: np.ndarray) -> np.ndarray:
    u, v = uv[:, 0], uv[:, 1]
    return np.column_stack((np.ones(len(uv)), u, v, u * u, u * v, v * v))


def _manual_gain(spec: PhotometricSpec, index: int, x, y):
    if not spec.vignette_coefficients:
        return 1.0
    k1, k2, k3 = spec.vignette_coefficients[index]
    radius2 = x * x + y * y
    return 1.0 / (1.0 + radius2 * (k1 + radius2 * (k2 + radius2 * k3)))


@dataclass(frozen=True, eq=False)
class PhotometricModel:
    spec: PhotometricSpec
    coefficients: np.ndarray
    anchors: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.spec, PhotometricSpec):
            raise TypeError("spec must be PhotometricSpec")
        coefficients = np.asarray(self.coefficients)
        if coefficients.ndim != 3 or coefficients.shape[1:] != (3, 6) or not len(coefficients):
            raise ValueError("coefficients must be frame_count x 3 x 6")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("Nonfinite photometric model")
        _validate_spec_frames(self.spec, len(coefficients))
        anchors = tuple(_integer(i, "anchor", 0) for i in self.anchors)
        if not anchors or len(set(anchors)) != len(anchors) or any(i >= len(coefficients) for i in anchors):
            raise ValueError("Invalid model anchors")
        if np.any(coefficients[list(anchors)] != 0):
            raise ValueError("Anchor coefficients must be exactly zero")
        if self.spec.mode == "off" and np.any(coefficients != 0):
            raise ValueError("Off model cannot have fitted gains")
        if self.spec.mode in {"exposure", "color"} and np.any(coefficients[:, :, 1:] != 0):
            raise ValueError("Constant gain mode cannot have field coefficients")
        if self.spec.mode == "exposure" and not np.all(coefficients == coefficients[:, :1, :]):
            raise ValueError("Exposure gains must be channel independent")
        object.__setattr__(self, "coefficients", _immutable(coefficients))
        object.__setattr__(self, "anchors", anchors)

    @property
    def frame_count(self) -> int:
        return len(self.coefficients)


def _validate_spec_frames(spec: PhotometricSpec, count: int):
    if spec.reference_index >= count:
        raise ValueError("reference_index is outside frame_count")
    if spec.vignette_coefficients and len(spec.vignette_coefficients) != count:
        raise ValueError("Provide one vignetting triple per frame")


def solve_photometric(observations: Iterable[PhotoObservation], frame_count: int,
                      spec: PhotometricSpec, should_cancel: Cancel = None) -> PhotometricModel:
    """Fit bounded log equations using sparse LSQR and Huber IRLS.

    Caps apply to retained sample pairs, not whole images. Per-overlap selection
    is evenly spaced, and the total budget is consumed in supplied order.
    Black samples (any channel <= min_signal in the working domain) and zero
    weights are excluded. No saturation or scene-content detector is implied.
    Cancellation is checked between overlaps, IRLS solves, and LSQR chunks.
    Insufficient rank uses the regularized minimum-norm solution; disconnected
    and unobserved frames retain exact component anchors.
    """
    _cancel(should_cancel)
    count = _integer(frame_count, "frame_count", 1)
    if not isinstance(spec, PhotometricSpec):
        raise TypeError("spec must be PhotometricSpec")
    _validate_spec_frames(spec, count)
    coefficients = np.zeros((count, 3, 6), dtype=np.float64)
    if spec.mode == "off":
        return PhotometricModel(spec, coefficients, tuple(range(count)))
    parent = list(range(count))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    retained = []
    used = 0
    for observation in observations:
        _cancel(should_cancel)
        if not isinstance(observation, PhotoObservation):
            raise TypeError("Expected PhotoObservation")
        a, b = observation.frame_a, observation.frame_b
        if max(a, b) >= count:
            raise ValueError("Observation frame index is outside frame_count")
        remaining = spec.max_total_samples - used
        if remaining <= 0:
            break
        n = min(len(observation.colors_a), spec.max_samples_per_observation, remaining)
        if not n:
            continue
        indices = np.linspace(0, len(observation.colors_a) - 1, n, dtype=np.int64)
        ca, cb = observation.colors_a[indices], observation.colors_b[indices]
        ua, ub = observation.uv_a[indices], observation.uv_b[indices]
        if spec.linear:
            ca, cb = srgb_to_linear(ca), srgb_to_linear(cb)
        weight = np.ones(n) if observation.weights is None else observation.weights[indices]
        valid = (weight > 0) & np.all(ca > spec.min_signal, axis=1) & np.all(cb > spec.min_signal, axis=1)
        if not np.any(valid):
            continue
        ca, cb, ua, ub, weight = ca[valid], cb[valid], ua[valid], ub[valid], weight[valid]
        ga = _manual_gain(spec, a, ua[:, 0], ua[:, 1])
        gb = _manual_gain(spec, b, ub[:, 0], ub[:, 1])
        ca = ca * np.asarray(ga)[..., None]
        cb = cb * np.asarray(gb)[..., None]
        if spec.mode == "exposure":
            # Rec.709 luminance, with the project BGR channel order.
            luminance = np.array([0.0722, 0.7152, 0.2126])
            target = (np.log(cb @ luminance) - np.log(ca @ luminance))[:, None]
        else:
            target = np.log(cb) - np.log(ca)
        retained.append((a, b, _basis(ua), _basis(ub), target, weight))
        used += len(weight)
        parent[root(b)] = root(a)
    groups = {}
    for i in range(count):
        groups.setdefault(root(i), []).append(i)
    anchors = tuple(sorted(spec.reference_index if spec.reference_index in group else min(group)
                           for group in groups.values()))
    if not retained:
        return PhotometricModel(spec, coefficients, anchors)
    terms = 6 if spec.mode == "gain_field" else 1
    channels = 1 if spec.mode == "exposure" else 3
    free = {i: k for k, i in enumerate(i for i in range(count) if i not in anchors)}
    rows, columns, data, targets, weights = [], [], [], [], []
    offset = 0
    for a, b, ba, bb, target, weight in retained:
        _cancel(should_cancel)
        n = len(weight)
        sample_rows = np.arange(offset, offset + n)
        for index, basis, sign in ((a, ba, 1.0), (b, bb, -1.0)):
            if index in free:
                rows.append(np.repeat(sample_rows, terms))
                columns.append(np.tile(np.arange(free[index] * terms, (free[index] + 1) * terms), n))
                data.append((sign * basis[:, :terms]).ravel())
        targets.append(target)
        weights.append(weight)
        offset += n
    matrix = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
                               shape=(offset, len(free) * terms)).tocsr()
    target = np.concatenate(targets)
    base_weights = np.concatenate(weights)
    regularizer = None
    if terms > 1 and spec.regularization:
        penalty = np.tile(np.array([0, 1, 1, 1, 1, 1], dtype=np.float64), len(free))
        regularizer = sparse.diags(np.sqrt(spec.regularization) * penalty, format="csr")
    for channel in range(channels):
        robust = np.ones(offset)
        solution = np.zeros(matrix.shape[1])
        for iteration in range(spec.robust_iterations + 1):
            _cancel(should_cancel)
            row_scale = np.sqrt(base_weights * robust)
            system = matrix.multiply(row_scale[:, None]).tocsr()
            rhs = target[:, channel] * row_scale
            if regularizer is not None:
                system = sparse.vstack((system, regularizer), format="csr")
                rhs = np.concatenate((rhs, np.zeros(matrix.shape[1])))
            # Chunked residual correction bounds cancellation latency without
            # using SciPy private APIs. Each step solves the same weighted system.
            solution = np.zeros(matrix.shape[1])
            remaining_iterations = spec.solver_iterations
            while remaining_iterations:
                _cancel(should_cancel)
                chunk = min(32, remaining_iterations)
                result = lsqr(system, rhs - system @ solution, atol=spec.solver_tolerance,
                              btol=spec.solver_tolerance, iter_lim=chunk)
                solution += result[0]
                remaining_iterations -= chunk
                if result[1] in (0, 1, 2, 4, 5):
                    break
                if result[1] in (3, 6):
                    raise RuntimeError("Photometric solve is too ill-conditioned")
            else:
                raise RuntimeError("Photometric solve did not converge within solver_iterations")
            if iteration < spec.robust_iterations:
                residual = np.abs(matrix @ solution - target[:, channel])
                robust = np.minimum(1.0, spec.huber_delta / np.maximum(residual, np.finfo(float).tiny))
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("Photometric solve produced nonfinite coefficients")
        for index, column in free.items():
            if channels == 1:
                coefficients[index, :, :terms] = solution[column * terms:(column + 1) * terms]
            else:
                coefficients[index, channel, :terms] = solution[column * terms:(column + 1) * terms]
    _cancel(should_cancel)
    return PhotometricModel(spec, coefficients, anchors)


def apply_photometric_tile(colors, index: int, uv_x, uv_y, model: PhotometricModel) -> np.ndarray:
    """Return corrected tile; input is normalized floating BGR or BGRA.

    Coordinates broadcast to colors.shape[:-1] and must describe source u,v.
    Actual combined manual/fitted gain is clipped, not individual coefficients.
    Color may exceed one after gain, retaining headroom until output encoding.
    """
    index = _integer(index, "index", 0)
    if index >= model.frame_count:
        raise ValueError("Frame index outside model")
    source = np.asarray(colors)
    if source.ndim < 2 or source.shape[-1] not in (3, 4) or not np.issubdtype(source.dtype, np.floating):
        raise ValueError("Expected normalized floating BGR/BGRA tile")
    if not np.all(np.isfinite(source)) or np.any(source < 0) or np.any(source > 1):
        raise ValueError("Tile values must be finite and normalized")
    x = np.broadcast_to(np.asarray(uv_x, dtype=np.float64), source.shape[:-1])
    y = np.broadcast_to(np.asarray(uv_y, dtype=np.float64), source.shape[:-1])
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(np.abs(x) > 1) or np.any(np.abs(y) > 1):
        raise ValueError("Source coordinates must lie in [-1,1]")
    result = srgb_to_linear(source) if model.spec.linear else np.array(source, copy=True)
    log_gain = np.zeros(source.shape[:-1] + (3,), dtype=np.float64)
    c = model.coefficients[index]
    for k, basis in enumerate((1.0, x, y, x * x, x * y, y * y)):
        log_gain += np.asarray(basis)[..., None] * c[:, k]
    manual = _manual_gain(model.spec, index, x, y)
    # Clip before exp to avoid overflow for underconstrained field extrapolation.
    log_gain += np.log(np.asarray(manual))[..., None]
    gain = np.exp(np.clip(log_gain, math.log(0.5), math.log(2.0)))
    result[..., :3] *= gain
    return result


class UnsupportedICCConversionError(ValueError):
    """Requested profile conversion cannot preserve the supplied tile layout."""




@dataclass(frozen=True)
class ICCPolicy:
    """Resolved output tag and per-source conversion decision.

    Empty source profiles in an explicitly mixed/sRGB conversion are assumed
    sRGB. keep_first/drop intentionally do no pixel conversion: the caller has
    explicitly chosen the metadata consequence. Common tags bypass parsing.
    """
    source_profiles: tuple[bytes, ...]
    output_profile: bytes
    convert_frames: tuple[bool, ...]

    def __post_init__(self):
        profiles = tuple(bytes(profile) for profile in self.source_profiles)
        conversions = tuple(bool(value) for value in self.convert_frames)
        if not profiles or len(profiles) != len(conversions):
            raise ValueError("ICC policy requires matching source and conversion counts")
        output = bytes(self.output_profile)
        if any(conversions) and not output:
            raise ValueError("ICC conversion requires an output profile")
        object.__setattr__(self, "source_profiles", profiles)
        object.__setattr__(self, "output_profile", output)
        object.__setattr__(self, "convert_frames", conversions)


def resolve_icc_policy(profiles: Sequence[bytes | None], policy: str = "auto") -> ICCPolicy:
    """auto: common bytes unchanged, empty stays empty, mixed converts to sRGB.

    srgb explicitly tags/converts to sRGB, including all-empty inputs.
    keep_first copies the first tag; drop emits no tag; both preserve pixels.
    Profile validation happens only when a tile genuinely needs conversion.
    """
    source = tuple(bytes(profile) if profile is not None else b"" for profile in profiles)
    if not source:
        raise ValueError("At least one source profile is required")
    if policy not in {"auto", "srgb", "keep_first", "drop"}:
        raise ValueError("Unknown ICC policy")
    if policy == "drop":
        return ICCPolicy(source, b"", (False,) * len(source))
    if policy == "keep_first" or (policy == "auto" and all(p == source[0] for p in source)):
        return ICCPolicy(source, source[0], (False,) * len(source))
    output = image_utils.srgb_icc_profile()
    return ICCPolicy(source, output, tuple(bool(p) and p != output for p in source))


@lru_cache(maxsize=16)
def _icc_transform(source: bytes, output: bytes):
    from PIL import ImageCms
    try:
        return ImageCms.buildTransformFromOpenProfiles(
            ImageCms.ImageCmsProfile(BytesIO(source)), ImageCms.ImageCmsProfile(BytesIO(output)),
            "RGB", "RGB", renderingIntent=0)
    except Exception as error:
        raise UnsupportedICCConversionError(f"Cannot build RGB to RGB ICC transform: {error}") from error


def convert_icc_tile(tile, index: int, policy: ICCPolicy, *, layout: str = "BGR",
                     should_cancel: Cancel = None) -> np.ndarray:
    """Convert one uint8 tile via actual LittleCMS, preserving alpha exactly.

    Supported conversions: BGR/BGRA or RGB/RGBA with RGB-space profiles.
    No-conversion policy returns a value-identical owned tile in its original
    layout and dtype. Uint16 and float ICC conversion, gray, CMYK and Lab
    layouts are explicitly rejected, never routed via uint8 or retagged.
    """
    _cancel(should_cancel)
    index = _integer(index, "index", 0)
    if index >= len(policy.source_profiles):
        raise ValueError("ICC frame index outside policy")
    source = np.asarray(tile)
    channels = {"BGR": 3, "BGRA": 4, "RGB": 3, "RGBA": 4}
    if layout not in channels:
        raise UnsupportedICCConversionError("Unsupported ICC tile layout")
    if source.ndim != 3 or source.shape[-1] != channels[layout]:
        raise ValueError("Tile shape does not match ICC layout")
    if not policy.convert_frames[index]:
        return np.array(source, copy=True)
    if source.dtype != np.uint8:
        raise UnsupportedICCConversionError(
            "Pillow ICC conversion here supports uint8 tiles only; select keep_first or drop for uint16/float")
    from PIL import Image, ImageCms
    alpha = source[..., -1] if layout in {"BGRA", "RGBA"} else None
    pixels = source[..., :3][..., ::-1] if layout in {"BGR", "BGRA"} else source[..., :3]
    transform = _icc_transform(policy.source_profiles[index] or image_utils.srgb_icc_profile(), policy.output_profile)
    _cancel(should_cancel)
    try:
        image = Image.fromarray(np.ascontiguousarray(pixels))
        converted = np.asarray(ImageCms.applyTransform(image, transform))
    except Exception as error:
        raise UnsupportedICCConversionError(f"ICC tile conversion failed: {error}") from error
    _cancel(should_cancel)
    if layout in {"BGR", "BGRA"}:
        converted = converted[..., ::-1]
    if alpha is not None:
        return np.concatenate((converted, alpha[..., None]), axis=-1)
    return np.array(converted, copy=True)
