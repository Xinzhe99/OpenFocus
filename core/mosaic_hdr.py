"""Bounded, tile-only HDR preparation; no I/O or production integration.

The sample project motivates exposure_target=.5 and exposure_sigma=.2;
these are not claims about another application's defaults or pixel equivalence.
Inputs are aligned encoded BGR. Fusion operates in encoded space; Debevec estimates log radiance
and applies luminance Reinhard followed by the sRGB encoding transfer.

The caller supplies tile halos and a common pyramid phase (tile origins divisible
by 2**(levels-1)), then crops the halo. A conservative halo is 8*2**(levels-1)
pixels. Image borders use OpenCV REFLECT_101. There is no image loading, geometric
alignment, deghosting, response fitting from full images, or automatic exposure
normalization. Calibration and rendering must use the same camera encoding.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.special import logsumexp


Cancel = Optional[Callable[[], bool]]
_KNOTS = np.linspace(0.0, 1.0, 256)
_LUMA = np.array([0.0722, 0.7152, 0.2126], dtype=np.float64)


def _cancel(should_cancel: Cancel) -> None:
    if should_cancel is not None and should_cancel():
        raise InterruptedError("HDR operation cancelled")


def _times(values: Sequence[float]) -> tuple:
    result = tuple(float(value) for value in values)
    if any(not np.isfinite(value) or value <= 0 for value in result):
        raise ValueError("exposure_times must contain positive finite seconds")
    return result


@dataclass(frozen=True, slots=True)
class HDRSpec:
    """Mode parameters; fusion controls affect fusion, tone controls affect Debevec.

    Off returns the opacity-weighted encoded average. Output opacity in every
    mode is the coverage union 1-product(1-alpha), with straight BGR output.
    levels includes the original resolution; tiny tiles stop at a 1x1 level.
    Debevec requires one time per image and an explicit ResponseModel.
    """

    mode: str = "off"
    exposure_times: tuple = ()
    contrast_weight: float = 1.0
    saturation_weight: float = 1.0
    exposure_weight: float = 1.0
    exposure_sigma: float = 0.2
    exposure_target: float = 0.5
    levels: int = 5
    tone_key: float = 0.18
    tone_white: float = 4.0

    def __post_init__(self) -> None:
        if self.mode not in ("off", "exposure_fusion", "debevec"):
            raise ValueError("unsupported HDR mode")
        object.__setattr__(self, "exposure_times", _times(self.exposure_times))
        for name in ("contrast_weight", "saturation_weight", "exposure_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        for name in ("exposure_sigma", "tone_key", "tone_white"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        target = float(self.exposure_target)
        if not np.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("exposure_target must be in [0, 1]")
        object.__setattr__(self, "exposure_target", target)
        if isinstance(self.levels, (bool, np.bool_)) or not isinstance(self.levels, (int, np.integer)) or self.levels < 1:
            raise ValueError("levels must be a positive integer")
        object.__setattr__(self, "levels", int(self.levels))


@dataclass(frozen=True, slots=True, eq=False)
class ResponseModel:
    """256 log-exposure knots per BGR channel, at z=k/255.

    log_response has shape (256, 3). It is copied onto immutable bytes storage:
    neither the constructor input nor setflags(write=True) can mutate a model.
    fit_response anchors knot 128 to zero. A caller may supply independently
    calibrated knots with its own explicit radiance scale.
    """

    log_response: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.log_response, dtype=np.float64)
        if values.shape != (256, 3) or not np.isfinite(values).all():
            raise ValueError("log_response must be finite with shape (256, 3)")
        owned = np.frombuffer(values.tobytes(order="C"), dtype=np.float64).reshape(256, 3)
        object.__setattr__(self, "log_response", owned)


def _fit_channel(values: np.ndarray, times: np.ndarray, smoothness: float, should_cancel: Cancel) -> np.ndarray:
    weights = 1.0 - np.abs(2.0 * values - 1.0)
    informative = weights > 0
    keep = informative.any(axis=0)
    values = values[:, keep]
    weights = weights[:, keep]
    informative = informative[:, keep]
    if not values.shape[1]:
        raise ValueError("response calibration has no unclipped observations")
    low = np.min(np.where(informative, values, np.inf), axis=0)
    high = np.max(np.where(informative, values, -np.inf), axis=0)
    if not np.any(high - low > 1e-6):
        raise ValueError("response calibration is rank deficient: no matched exposure variation")

    # Unknowns are 256 response knots followed by one log irradiance per sample.
    rows, cols, data, rhs = [], [], [], []
    row = 0
    for exposure in range(values.shape[0]):
        _cancel(should_cancel)
        for pixel in range(values.shape[1]):
            weight = weights[exposure, pixel]
            if weight <= 0:
                continue
            position = values[exposure, pixel] * 255.0
            knot = min(int(position), 254)
            fraction = position - knot
            rows.extend((row, row, row))
            cols.extend((knot, knot + 1, 256 + pixel))
            data.extend((weight * (1.0 - fraction), weight * fraction, -weight))
            rhs.append(weight * np.log(times[exposure]))
            row += 1
    for knot in range(1, 255):
        if knot % 32 == 0:
            _cancel(should_cancel)
        weight = smoothness * (1.0 - abs(2.0 * knot / 255.0 - 1.0))
        rows.extend((row, row, row))
        cols.extend((knot - 1, knot, knot + 1))
        data.extend((weight, -2.0 * weight, weight))
        rhs.append(0.0)
        row += 1
    rows.append(row)
    cols.append(128)
    data.append(1.0)
    rhs.append(0.0)
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(row + 1, 256 + values.shape[1])).tocsr()

    # lsmr has no callback; checking both operator products also cancels inside
    # the sparse iterative solve rather than only between color channels.
    def multiply(vector):
        _cancel(should_cancel)
        return matrix @ vector

    def transpose_multiply(vector):
        _cancel(should_cancel)
        return matrix.T @ vector

    operator = LinearOperator(matrix.shape, matvec=multiply, rmatvec=transpose_multiply, dtype=np.float64)
    result = lsmr(operator, np.asarray(rhs), atol=1e-10, btol=1e-10,
                  conlim=1e12, maxiter=32 * matrix.shape[1])
    if result[1] not in (0, 1, 2, 4, 5) or not np.isfinite(result[0]).all():
        raise ValueError("response calibration sparse least squares did not converge")
    response = result[0][:256]
    return response - response[128]


def fit_response(samples, exposure_times, *, smoothness: float = 20.0,
                 should_cancel: Cancel = None) -> ResponseModel:
    """Fit Debevec's weighted sparse log-response equations independently in BGR.

    samples is normalized encoded [exposures, matched_pixels, 3]. At most 256
    evenly spaced matched pixel indices, including both ends, enter the fit.
    Input validation is chunked; fitting retains only those bounded observations.
    Endpoint observations have triangular weight zero. Each channel must have
    at least one matched pixel with distinct, unclipped exposure measurements;
    otherwise the response slope is unidentifiable and ValueError is raised.
    """
    _cancel(should_cancel)
    values = np.asarray(samples)
    if values.ndim != 3 or values.shape[2] != 3 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("samples must have shape (N>=2, M>=1, 3)")
    if values.dtype.kind not in "fiu":
        raise ValueError("samples must be numeric normalized encoded values")
    times = np.asarray(_times(exposure_times), dtype=np.float64)
    if len(times) != values.shape[0] or np.unique(times).size < 2:
        raise ValueError("calibration needs one time per exposure and distinct exposure times")
    smoothness = float(smoothness)
    if not np.isfinite(smoothness) or smoothness <= 0:
        raise ValueError("smoothness must be positive and finite")
    for exposure in range(values.shape[0]):
        for start in range(0, values.shape[1], 256):
            _cancel(should_cancel)
            chunk = values[exposure, start:start + 256]
            if not np.isfinite(chunk).all() or np.any(chunk < 0) or np.any(chunk > 1):
                raise ValueError("samples must be finite normalized encoded values in [0, 1]")
    indices = np.linspace(0, values.shape[1] - 1, min(256, values.shape[1]), dtype=np.int64)
    selected = np.array(values[:, indices, :], dtype=np.float64, copy=True)
    response = np.empty((256, 3), dtype=np.float64)
    for channel in range(3):
        _cancel(should_cancel)
        response[:, channel] = _fit_channel(selected[..., channel], times, smoothness, should_cancel)
    _cancel(should_cancel)
    return ResponseModel(response)


def _inputs(images, alpha, should_cancel: Cancel):
    source = tuple(images)
    if not source:
        raise ValueError("at least one image tile is required")
    if alpha is None:
        masks = (None,) * len(source)
    else:
        masks = tuple(alpha)
        if len(masks) != len(source):
            raise ValueError("alpha must contain one opacity tile per image")
    normalized, opacities = [], []
    shape = None
    for image, opacity in zip(source, masks):
        _cancel(should_cancel)
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
            raise ValueError("image tiles must have nonempty shape (H, W, 3)")
        if shape is None:
            shape = value.shape
        elif value.shape != shape:
            raise ValueError("image tiles must have the same shape")
        if value.dtype == np.uint8:
            value = value.astype(np.float32) / np.float32(255.0)
        elif value.dtype == np.uint16:
            value = value.astype(np.float32) / np.float32(65535.0)
        elif value.dtype.kind == "f":
            if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
                raise ValueError("float images must be finite normalized values in [0, 1]")
            value = np.array(value, dtype=np.float32, copy=True)
        else:
            raise ValueError("image dtype must be uint8, uint16, or normalized float")
        if opacity is None:
            coverage = np.ones(value.shape[:2], dtype=np.float32)
        else:
            coverage = np.asarray(opacity)
            if coverage.shape != value.shape[:2] or coverage.dtype.kind not in "fiub":
                raise ValueError("opacity must be numeric with shape (H, W)")
            if not np.isfinite(coverage).all() or np.any(coverage < 0) or np.any(coverage > 1):
                raise ValueError("opacity must be finite in [0, 1]")
            coverage = np.array(coverage, dtype=np.float32, copy=True)
        value[coverage <= 0] = 0.0
        normalized.append(value)
        opacities.append(coverage)
    return normalized, opacities


def _down(value):
    return cv2.pyrDown(value, borderType=cv2.BORDER_REFLECT_101)


def _up(value, shape):
    return cv2.pyrUp(value, dstsize=(shape[1], shape[0]), borderType=cv2.BORDER_DEFAULT)


def _fusion(images, alpha, spec: HDRSpec, should_cancel: Cancel):
    shape = images[0].shape[:2]
    shapes = [shape]
    while len(shapes) < spec.levels and shapes[-1] != (1, 1):
        shapes.append(tuple((size + 1) // 2 for size in shapes[-1]))
    sums = [np.zeros((*size, 3), dtype=np.float32) for size in shapes]
    totals = [np.zeros(size, dtype=np.float32) for size in shapes]
    cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    base_weights = []
    base_total = np.zeros(shape, dtype=np.float64)
    for image, coverage in zip(images, alpha):
        _cancel(should_cancel)
        gray = image @ _LUMA.astype(np.float32)
        # Opacity-normalized neighbor averages exclude missing colors without
        # changing contrast merely because a whole tile is partly opaque.
        neighbor_sum = cv2.filter2D(gray * coverage, -1, cross, borderType=cv2.BORDER_REFLECT_101)
        neighbor_coverage = cv2.filter2D(coverage, -1, cross, borderType=cv2.BORDER_REFLECT_101)
        neighbor_mean = np.divide(neighbor_sum, neighbor_coverage, out=gray.copy(), where=neighbor_coverage > 0)
        contrast = np.abs(neighbor_mean - gray)
        saturation = np.std(image, axis=2, dtype=np.float64)
        log_quality = np.zeros(shape, dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore"):
            if spec.contrast_weight:
                log_quality += spec.contrast_weight * np.log(np.maximum(contrast, 1e-30))
            if spec.saturation_weight:
                log_quality += spec.saturation_weight * np.log(np.maximum(saturation, 1e-30))
            if spec.exposure_weight:
                distance = (image.astype(np.float64) - spec.exposure_target) / spec.exposure_sigma
                log_quality -= spec.exposure_weight * np.sum(distance * distance, axis=2) / 2.0
        # Mertens' small positive floor also defines constant/achromatic regions.
        weights = np.exp(np.logaddexp(log_quality, np.log(1e-12))) * coverage
        base_weights.append(weights)
        base_total += weights
    for image, coverage, raw_weight in zip(images, alpha, base_weights):
        _cancel(should_cancel)
        weights = np.divide(raw_weight, base_total, out=np.zeros(shape, dtype=np.float32), where=base_total > 0)
        colors = [image]
        masks = [coverage]
        weight_levels = [weights]
        for level in range(1, len(shapes)):
            _cancel(should_cancel)
            mask = _down(masks[-1])
            premultiplied = _down(colors[-1] * masks[-1][..., None])
            color = np.divide(premultiplied, mask[..., None], out=np.zeros_like(premultiplied), where=mask[..., None] > 0)
            colors.append(color)
            masks.append(mask)
            weight_levels.append(_down(weight_levels[-1]))
        for level in range(len(shapes)):
            _cancel(should_cancel)
            laplacian = colors[level]
            if level + 1 < len(shapes):
                laplacian = laplacian - _up(colors[level + 1], shapes[level])
            sums[level] += laplacian * weight_levels[level][..., None]
            totals[level] += weight_levels[level]
    blended = []
    for level in range(len(shapes)):
        _cancel(should_cancel)
        blended.append(np.divide(sums[level], totals[level][..., None], out=np.zeros_like(sums[level]), where=totals[level][..., None] > 0))
    result = blended[-1]
    for level in range(len(shapes) - 2, -1, -1):
        _cancel(should_cancel)
        result = _up(result, shapes[level]) + blended[level]
    return np.clip(result, 0.0, 1.0)


def _debevec(images, alpha, spec: HDRSpec, response: ResponseModel, should_cancel: Cancel):
    shape = images[0].shape
    weighted = np.zeros(shape, dtype=np.float64)
    weights = np.zeros(shape, dtype=np.float64)
    endpoint_sum = np.zeros(shape, dtype=np.float64)
    endpoint_weight = np.zeros(shape[:2], dtype=np.float64)
    for image, coverage, time in zip(images, alpha, spec.exposure_times):
        _cancel(should_cancel)
        weight = (1.0 - np.abs(2.0 * image.astype(np.float64) - 1.0)) * coverage[..., None]
        log_exposure = np.empty(shape, dtype=np.float64)
        for channel in range(3):
            _cancel(should_cancel)
            log_exposure[..., channel] = np.interp(image[..., channel], _KNOTS, response.log_response[:, channel]) - np.log(time)
        weighted += weight * log_exposure
        weights += weight
        endpoint_sum += coverage[..., None] * log_exposure
        endpoint_weight += coverage
    # Explicit endpoint policy: where all covered measurements are clipped,
    # use the opacity-weighted mean endpoint log-exposure/time estimate. This is
    # deterministic and bounded by those estimates, not recovered highlight data.
    log_radiance = np.divide(endpoint_sum, endpoint_weight[..., None], out=np.zeros_like(endpoint_sum), where=endpoint_weight[..., None] > 0)
    np.divide(weighted, weights, out=log_radiance, where=weights > 0)
    _cancel(should_cancel)
    log_luminance = logsumexp(log_radiance + np.log(_LUMA), axis=2)
    log_key = np.log(spec.tone_key)
    scaled_luminance = log_luminance + log_key
    # Ld = (key*L) * (1 + key*L/white**2) / (1 + key*L).
    # Working in logs avoids overflow without any tile-dependent normalization.
    log_scale = log_key + np.logaddexp(0.0, scaled_luminance - 2.0 * np.log(spec.tone_white)) - np.logaddexp(0.0, scaled_luminance)
    linear = np.exp(np.minimum(log_radiance + log_scale[..., None], 0.0))
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)
    return encoded.astype(np.float32)


def merge_hdr_tile(images, spec: HDRSpec, *, alpha=None, response=None,
                   should_cancel: Cancel = None):
    """Return (straight normalized encoded BGR float32, union opacity float32).

    images is a nonempty sequence of same-size uint8/uint16/normalized-float
    tiles. alpha, when present, is a sequence of fractional HxW opacities.
    Excluded pixels are removed before all filtering and radiance estimates.
    No-valid-coverage pixels return exactly zero color and opacity. Input arrays
    (including strided/read-only arrays) are never modified. Cancellation raises
    InterruptedError, including during response solves and pyramid loops.
    """
    _cancel(should_cancel)
    if not isinstance(spec, HDRSpec):
        raise TypeError("spec must be HDRSpec")
    images, coverage = _inputs(images, alpha, should_cancel)
    if spec.exposure_times and len(spec.exposure_times) != len(images):
        raise ValueError("exposure_times must match the image count")
    if spec.mode == "debevec":
        if not spec.exposure_times or response is None:
            raise ValueError("Debevec requires exposure_times and a ResponseModel")
        if not isinstance(response, ResponseModel):
            raise TypeError("response must be ResponseModel")
    opacity = np.zeros(images[0].shape[:2], dtype=np.float32)
    for mask in coverage:
        _cancel(should_cancel)
        opacity += (1.0 - opacity) * mask
    if spec.mode == "off":
        total = np.zeros_like(opacity)
        result = np.zeros_like(images[0])
        for image, mask in zip(images, coverage):
            _cancel(should_cancel)
            result += image * mask[..., None]
            total += mask
        np.divide(result, total[..., None], out=result, where=total[..., None] > 0)
    elif spec.mode == "exposure_fusion":
        result = _fusion(images, coverage, spec, should_cancel)
    else:
        result = _debevec(images, coverage, spec, response, should_cancel)
    _cancel(should_cancel)
    result[opacity <= 0] = 0.0
    return np.asarray(result, dtype=np.float32), opacity
