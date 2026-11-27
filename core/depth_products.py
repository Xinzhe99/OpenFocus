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
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]

    if arr.dtype == np.uint16:
        scaled = arr.astype(np.float32) * (1.0 / 65535.0)
    elif arr.dtype == np.uint8:
        scaled = arr.astype(np.float32) * (1.0 / 255.0)
    elif np.issubdtype(arr.dtype, np.floating):
        scaled = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        hi = float(np.max(scaled)) if scaled.size else 1.0
        if hi > 1.0:
            scaled = scaled / hi
        scaled = np.clip(scaled, 0.0, 1.0)
    else:
        scaled = arr.astype(np.float32)
        hi = float(np.max(scaled)) if scaled.size else 1.0
        if hi > 0:
            scaled = scaled / hi

    if scaled.ndim == 3 and scaled.shape[2] == 3:
        # OpenCV BGR luminance weights
        gray = (
            0.114 * scaled[..., 0]
            + 0.587 * scaled[..., 1]
            + 0.299 * scaled[..., 2]
        ).astype(np.float32)
    elif scaled.ndim == 3:
        gray = np.mean(scaled, axis=2, dtype=np.float32)
    else:
        gray = np.ascontiguousarray(scaled, dtype=np.float32)

    mode = str(method or "hybrid").strip().lower()
    if mode not in ("hybrid", "tenengrad", "laplacian"):
        raise ValueError(f"Unsupported contrast map method: {method!r}")

    energy = np.zeros_like(gray, dtype=np.float32)
    if mode in ("hybrid", "tenengrad"):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad = cv2.magnitude(gx, gy)
        t_max = float(np.max(tenengrad))
        if t_max > 0.0:
            tenengrad = tenengrad / t_max
        energy = tenengrad if mode == "tenengrad" else (energy + 0.65 * tenengrad)

    if mode in ("hybrid", "laplacian"):
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        l_max = float(np.max(lap))
        if l_max > 0.0:
            lap = lap / l_max
        energy = lap if mode == "laplacian" else (energy + 0.35 * lap)

    peak = float(np.max(energy)) if energy.size else 0.0
    if peak > 0.0:
        pct = float(percentile)
        scale = float(np.percentile(energy, pct)) if 0.0 < pct < 100.0 else peak
        if scale <= 1e-8:
            scale = peak
        norm = np.clip(energy / scale, 0.0, 1.0)
    else:
        norm = np.zeros_like(gray, dtype=np.float32)

    if colormap:
        from utils import colormap as cmap
        return cmap.colorize(norm, str(colormap))

    if int(bit_depth) == 8:
        return np.clip(np.rint(norm * 255.0), 0, 255).astype(np.uint8)
    if int(bit_depth) == 16:
        return np.clip(np.rint(norm * 65535.0), 0, 65535).astype(np.uint16)
    raise ValueError(f"Unsupported contrast map bit_depth: {bit_depth!r}")


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
