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
