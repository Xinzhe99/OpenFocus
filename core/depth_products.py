"""Products derived from estimated normalized focus positions.

Indices identify positions in the exact ordered stack used for the estimate;
they do not establish which pixels contributed to a fused image.
"""
import numpy as np

from core.depth_map import DepthMapCancelled


_ROWS = 128


def _cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise DepthMapCancelled()


def estimate_indices(index, count, valid_mask=None, should_cancel=None):
    """Quantize normalized positions to 1-based uint16 indices; 0 is invalid."""
    if not 2 <= count <= 65535:
        raise ValueError("Source-index export requires 2 to 65535 frames")
    index = np.asarray(index)
    if index.ndim != 2:
        raise ValueError("Focus positions must be a two-dimensional map")
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != index.shape:
            raise ValueError("Validity mask does not match focus positions")
    result = np.zeros(index.shape, dtype=np.uint16)
    for y in range(0, index.shape[0], _ROWS):
        _cancel(should_cancel)
        positions = index[y:y + _ROWS]
        valid = np.isfinite(positions)
        if valid_mask is not None:
            valid &= valid_mask[y:y + _ROWS]
        out = result[y:y + _ROWS]
        # Use float64 for the quantization so frame boundaries are not shifted
        # by float32 multiplication on deep stacks.
        values = np.clip(positions[valid].astype(np.float64), 0, 1)
        out[valid] = np.floor(values * (count - 1) + 0.5).astype(np.uint16) + 1
    _cancel(should_cancel)
    return result


def render_indices(indices, frames, should_cancel=None):
    """Copy source pixels; promote 8-bit values when the stack contains 16-bit frames."""
    indices = np.asarray(indices)
    if indices.ndim != 2 or indices.dtype != np.uint16:
        raise ValueError("Source indices must be a two-dimensional uint16 map")
    if not 2 <= len(frames) <= 65535:
        raise ValueError("Source-index render requires 2 to 65535 frames")
    first = np.asarray(frames[0])
    if first.dtype not in (np.uint8, np.uint16) or first.shape[:2] != indices.shape:
        raise ValueError("Source frames must match the map and use 8- or 16-bit pixels")
    if first.ndim not in (2, 3):
        raise ValueError("Unsupported source frame shape")
    output_dtype = np.uint8
    for frame in frames:
        if frame.shape != first.shape or frame.dtype not in (np.uint8, np.uint16):
            raise ValueError("Source frames must have matching shapes and use 8- or 16-bit pixels")
        if frame.dtype == np.uint16:
            output_dtype = np.uint16
    result = np.zeros(first.shape, dtype=output_dtype)
    for y in range(0, indices.shape[0], _ROWS):
        _cancel(should_cancel)
        selection = indices[y:y + _ROWS]
        if np.any(selection > len(frames)):
            raise ValueError("Source index exceeds the number of frames")
        destination = result[y:y + _ROWS]
        for number in np.unique(selection):
            if number == 0:
                continue
            _cancel(should_cancel)
            mask = selection == number
            frame = frames[int(number) - 1]
            values = frame[y:y + _ROWS][mask]
            if output_dtype == np.uint16 and frame.dtype == np.uint8:
                values = values.astype(np.uint16) * 257
            destination[mask] = values
    _cancel(should_cancel)
    return result


def _fast_percentile_positive(energy: np.ndarray, peak: float, pct: float) -> float:
    """Compute percentile on a non-negative float32 array in [0, peak] with minimal RAM."""
    import cv2

    if energy.size <= 262144:
        return float(np.percentile(energy, pct))

    # 8192-bin histogram over [0, peak]: O(N) C++ pass, 32 KB RAM, ~0.012% bin width.
    n_bins = 8192
    upper = float(peak) * (1.0 + 1e-6)
    hist = cv2.calcHist([energy], [0], None, [n_bins], [0.0, upper]).ravel()
    target = (pct / 100.0) * float(energy.size)
    cdf = np.cumsum(hist, dtype=np.float64)
    idx = int(np.searchsorted(cdf, target, side="left"))
    if idx >= n_bins:
        return peak
    bin_width = upper / n_bins
    prev_c = float(cdf[idx - 1]) if idx > 0 else 0.0
    cur_c = float(cdf[idx])
    frac = (target - prev_c) / max(cur_c - prev_c, 1.0)
    return min(peak, (idx + max(0.0, min(1.0, frac))) * bin_width)


def compute_contrast_map(
    image: np.ndarray,
    *,
    bit_depth: int = 16,
    method: str = "hybrid",
    colormap: str | None = None,
    percentile: float = 99.5,
) -> np.ndarray:
    """Compute a normalized local contrast energy map from a single image.

    Combines Tenengrad Sobel gradient magnitude and absolute Laplacian energy
    (or selects either individually via ``method`` = ``"tenengrad"`` /
    ``"laplacian"`` / ``"hybrid"``), normalized cleanly to 16-bit grayscale,
    8-bit grayscale, or a BGR pseudocolor visualization when ``colormap`` is set.
    """
    import cv2

    arr = np.asarray(image)
    if arr.size == 0 or arr.ndim not in (2, 3):
        raise ValueError("Contrast map requires a non-empty 2D or 3D image")
    mode = str(method or "hybrid").strip().lower()
    if mode not in ("hybrid", "tenengrad", "laplacian"):
        raise ValueError(f"Unsupported contrast map method: {method!r}")
    if not colormap and int(bit_depth) not in (8, 16):
        raise ValueError(f"Unsupported contrast map bit_depth: {bit_depth!r}")

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]

    # Convert to single-channel luminance BEFORE float32 promotion so Sobel/Laplacian
    # run directly on uint8/uint16 via OpenCV SIMD without 3-channel float32 copies.
    # Constant input scale factors cancel out when normalizing by t_max / l_max.
    if arr.dtype in (np.uint8, np.uint16):
        if arr.ndim == 3 and arr.shape[2] == 3:
            gray = cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 3:
            gray = np.mean(arr, axis=2, dtype=np.float32)
        else:
            gray = np.ascontiguousarray(arr)
    elif np.issubdtype(arr.dtype, np.floating):
        if arr.ndim == 3 and arr.shape[2] == 3:
            f32 = np.ascontiguousarray(arr, dtype=np.float32)
            gray = cv2.cvtColor(f32, cv2.COLOR_BGR2GRAY)
            del f32
        elif arr.ndim == 3:
            gray = np.mean(arr, axis=2, dtype=np.float32)
        else:
            gray = np.array(arr, dtype=np.float32, copy=True)
        np.nan_to_num(gray, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        hi = float(cv2.minMaxLoc(gray)[1]) if gray.size else 1.0
        if hi > 1.0:
            gray *= np.float32(1.0 / hi)
        np.clip(gray, 0.0, 1.0, out=gray)
    else:
        if arr.ndim == 3 and arr.shape[2] == 3:
            f32 = np.ascontiguousarray(arr, dtype=np.float32)
            gray = cv2.cvtColor(f32, cv2.COLOR_BGR2GRAY)
            del f32
        elif arr.ndim == 3:
            gray = np.mean(arr, axis=2, dtype=np.float32)
        else:
            gray = np.ascontiguousarray(arr, dtype=np.float32)

    # Reuse at most two float32 scratch buffers (buf1, buf2) in-place across all stages.
    energy: np.ndarray
    if mode in ("hybrid", "tenengrad"):
        buf1 = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        buf2 = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        cv2.magnitude(buf1, buf2, magnitude=buf1)
        t_max = float(cv2.minMaxLoc(buf1)[1])
        if t_max > 0.0:
            weight = 0.65 if mode == "hybrid" else 1.0
            buf1 *= np.float32(weight / t_max)
        if mode == "hybrid":
            cv2.Laplacian(gray, cv2.CV_32F, dst=buf2, ksize=3)
            np.abs(buf2, out=buf2)
            l_max = float(cv2.minMaxLoc(buf2)[1])
            if l_max > 0.0:
                cv2.addWeighted(buf1, 1.0, buf2, 0.35 / l_max, 0.0, dst=buf1)
        del buf2
        energy = buf1
    else:
        buf1 = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        np.abs(buf1, out=buf1)
        l_max = float(cv2.minMaxLoc(buf1)[1])
        if l_max > 0.0:
            buf1 *= np.float32(1.0 / l_max)
        energy = buf1
    del gray

    peak = float(cv2.minMaxLoc(energy)[1]) if energy.size else 0.0
    if peak <= 0.0:
        if colormap:
            from utils import colormap as cmap
            return cmap.colorize(energy, str(colormap))
        out_dtype = np.uint8 if int(bit_depth) == 8 else np.uint16
        return np.zeros(energy.shape, dtype=out_dtype)

    pct = float(percentile)
    scale = _fast_percentile_positive(energy, peak, pct) if 0.0 < pct < 100.0 else peak
    if scale <= 1e-8:
        scale = peak

    if colormap:
        energy *= np.float32(1.0 / scale)
        np.clip(energy, 0.0, 1.0, out=energy)
        from utils import colormap as cmap
        return cmap.colorize(energy, str(colormap))

    if int(bit_depth) == 8:
        return cv2.addWeighted(energy, 255.0 / scale, energy, 0.0, 0.0, dtype=cv2.CV_8U)
    return cv2.addWeighted(energy, 65535.0 / scale, energy, 0.0, 0.0, dtype=cv2.CV_16U)


def export_contrast_map(
    path: str,
    image: np.ndarray,
    *,
    bit_depth: int | None = None,
    method: str = "hybrid",
    colormap: str | None = None,
    should_cancel=None,
) -> bool:
    """Compute and atomically export a contrast map to PNG, TIFF, or 8-bit formats."""
    import os
    from utils.image_utils import get_imwrite_params, imwrite_auto

    ext = os.path.splitext(os.fspath(path))[1].lower()
    if bit_depth is None:
        if colormap:
            bit_depth = 8
        elif ext in (".png", ".tif", ".tiff"):
            bit_depth = 16
        else:
            bit_depth = 8
    contrast = compute_contrast_map(
        image, bit_depth=bit_depth, method=method, colormap=colormap
    )
    return imwrite_auto(
        path,
        contrast,
        get_imwrite_params(ext),
        exif=b"",
        icc_profile=b"",
        should_cancel=should_cancel,
    )
