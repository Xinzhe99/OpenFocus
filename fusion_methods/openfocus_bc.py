"""Depth-map and multilevel fusion for OpenFocus B and C.

B merges neighboring frames at the estimated depth. C merges their multiscale
residuals using a separately reduced depth map at each scale.
"""
import numpy as np

from fusion_methods.openfocus_depth import depth_map
from core.frame_storage import create_frame, store_frame


def _weight(depth, index):
    # Adjacent-frame linear weights, quantized as Q10. The two weights add to
    # exactly 1024, including the endpoints of stacks with more than two frames.
    lower = np.floor(depth).astype(np.int32)
    fraction = np.clip(np.rint((depth - lower) * 1024), 0, 1024).astype(np.int32)
    return (np.where(lower == index, 1024 - fraction, 0) +
            np.where(lower + 1 == index, fraction, 0)).astype(np.float32) / 1024.0


def _reduce(image):
    """Truncated 2x2 mean, extending incomplete edges."""
    h, w = image.shape[:2]
    if h % 2 or w % 2:
        padding = ((0, h % 2), (0, w % 2)) + ((0, 0),) * (image.ndim - 2)
        image = np.pad(image, padding, mode="edge")
    return ((image[0::2, 0::2] + image[0::2, 1::2]
             + image[1::2, 0::2] + image[1::2, 1::2]) >> 2)


def _expand_axis(image, axis):
    """Four-tap phase-mirrored expansion, with edge replication."""
    n = image.shape[axis]
    shape = list(image.shape)
    shape[axis] = 2 * n
    out = create_frame(tuple(shape), np.int32)
    positions = np.arange(n)
    for phase, offsets, weights in (
            (0, (-2, -1, 0, 1), (2, 37, 64, 12)),
            (1, (-1, 0, 1, 2), (12, 64, 37, 2))):
        accum = np.zeros_like(image, dtype=np.int32)
        for offset, weight in zip(offsets, weights):
            accum += np.take(image, np.clip(positions + offset, 0, n - 1),
                             axis=axis) * weight
        target = [slice(None)] * image.ndim
        target[axis] = slice(phase, None, 2)
        out[tuple(target)] = (accum + 57) // 115
    return out


def _expand(image, shape):
    return _expand_axis(_expand_axis(image, 0), 1)[:shape[0], :shape[1]]


def _inputs(input_source, img_resize, should_cancel):
    from fusion_methods.openfocus_ab import _collect_images, _prepare_frame
    frames = _collect_images(input_source, normalize=False, should_cancel=should_cancel)
    first = _prepare_frame(frames[0], img_resize, should_cancel)
    shape = first.shape
    return frames, shape


def _output(image, frames):
    dtype = np.uint8 if all(frame.dtype == np.uint8 for frame in frames) else np.float32
    output = create_frame(image.shape, dtype)
    for start in range(0, image.shape[0], 128):
        rows = slice(start, min(image.shape[0], start + 128))
        output[rows] = np.clip(image[rows], 0, 255)
    return output

def _depth(frames, shape, img_resize, kernel_size, smoothing, should_cancel,
           depth_override):
    if depth_override is None:
        return depth_map(frames, img_resize, kernel_size, smoothing, should_cancel)
    depth = np.asarray(depth_override, dtype=np.float32)
    if depth.shape != shape[:2]:
        raise ValueError("depth_override must match the source height and width")
    return depth


def openfocus_b_impl(input_source, img_resize=None, kernel_size: int = 11,
                     thread_count: int = None, depth_smoothing: float = 1.2,
                     should_cancel=None, depth_override=None):
    """Fuse source pixels using a continuous, smoothed depth map."""
    from fusion_methods.openfocus_ab import _check_cancel, _prepare_frame
    frames, shape = _inputs(input_source, img_resize, should_cancel)
    smoothing = 1.2 if depth_smoothing is None else float(depth_smoothing)
    depth = _depth(frames, shape, img_resize, kernel_size,
                   smoothing * (5.0 / 1.2), should_cancel, depth_override)
    result = create_frame(shape, np.float32)
    result[...] = 0
    for index, source in enumerate(frames):
        _check_cancel(should_cancel)
        weight = _weight(depth, index)
        frame = _prepare_frame(source, img_resize, should_cancel)
        result += frame * (weight[:, :, None] if frame.ndim == 3 else weight)
    _check_cancel(should_cancel)
    return _output(result, frames)


def openfocus_c_impl(input_source, img_resize=None, kernel_size: int = 7,
                     thread_count: int = None, levels: int = 3,
                     blend_smoothing: float = 3.0, should_cancel=None,
                     depth_override=None):
    """Fuse depth-guided pyramids with Q10 merge and integer collapse."""
    from fusion_methods.openfocus_ab import _check_cancel, _prepare_frame
    frames, shape = _inputs(input_source, img_resize, should_cancel)
    smoothing = 3.0 if blend_smoothing is None else float(blend_smoothing)
    # Method C exposes smoothing only; use the default radius internally.
    depth = _depth(frames, shape, img_resize, 11,
                   smoothing * (5.0 / 3.0), should_cancel, depth_override)
    depth_levels = [store_frame(np.rint(depth * 1024).astype(np.int32))]
    for _ in range(max(1, int(levels)) - 1):
        _check_cancel(should_cancel)
        if max(depth_levels[-1].shape) <= 1:
            break
        depth_levels.append(store_frame(_reduce(depth_levels[-1])))
    blends = [create_frame(shape, np.int32)]
    for code in depth_levels[1:]:
        blends.append(create_frame(code.shape + shape[2:], np.int32))
    for blend in blends:
        blend[...] = 0
    is_8bit = all(source.dtype == np.uint8 for source in frames)
    scale = 256.0 if is_8bit else 65535.0 / 255.0
    for index, source in enumerate(frames):
        _check_cancel(should_cancel)
        frame = _prepare_frame(source, img_resize, should_cancel)
        if frame.shape != shape:
            raise ValueError("OpenFocus C requires frames with matching shapes")
        current = np.rint(np.clip(frame, 0, 255) * scale).astype(np.int32)
        for level, code in enumerate(depth_levels):
            _check_cancel(should_cancel)
            if level + 1 < len(depth_levels):
                reduced = _reduce(current)
                band = current - _expand(reduced, current.shape)
            else:
                band = current
            weight = np.rint(_weight(code.astype(np.float32) / 1024, index)
                             * 1024).astype(np.int32)
            if current.ndim == 3:
                weight = weight[:, :, None]
            blends[level] += band * weight
            if level + 1 < len(depth_levels):
                current = reduced
    result = store_frame(np.clip((blends[-1] + 512) >> 10, 0, 65535))
    for level in range(len(blends) - 2, -1, -1):
        _check_cancel(should_cancel)
        detail = (blends[level] + 512) >> 10
        result = store_frame(np.clip(_expand(result, blends[level].shape) + detail, 0, 65535))
    _check_cancel(should_cancel)
    output = create_frame(result.shape, np.uint8 if is_8bit else np.float32)
    for start in range(0, result.shape[0], 128):
        _check_cancel(should_cancel)
        rows = slice(start, min(result.shape[0], start + 128))
        output[rows] = ((result[rows] >> 8) if is_8bit else
                        result[rows].astype(np.float32) * np.float32(255.0 / 65535.0))
    return output
