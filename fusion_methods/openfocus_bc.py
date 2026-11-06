"""Depth-map and multilevel fusion for OpenFocus B and C.

B merges neighboring frames at the estimated depth. C merges their multiscale
residuals using a separately reduced depth map at each scale.
Independent per-frame depth activities may run concurrently in an admitted
window. Global winner selection, depth smoothing and pyramid collapse remain
ordered stages separated by a producer barrier.
"""
import numpy as np

from fusion_methods.openfocus_depth import depth_map
from core.frame_storage import create_frame


DEFAULT_PYRAMID_LEVELS = 3


def _weight(depth, index):
    # Adjacent-frame linear weights, quantized as Q10. The two weights add to
    # exactly 1024, including the endpoints of stacks with more than two frames.
    lower = np.floor(depth).astype(np.int32)
    fraction = np.clip(np.rint((depth - lower) * 1024), 0, 1024).astype(np.int32)
    return (np.where(lower == index, 1024 - fraction, 0) +
            np.where(lower + 1 == index, fraction, 0)).astype(np.float32) / 1024.0


def _reduce(image, should_cancel=None):
    """Truncated 2x2 mean with replicated edges and bounded row scratch."""
    from fusion_methods.openfocus_ab import _check_cancel
    h, w = image.shape[:2]
    shape = ((h + 1) // 2, (w + 1) // 2) + image.shape[2:]
    out = create_frame(shape, image.dtype)
    left = np.arange(shape[1]) * 2
    right = np.minimum(left + 1, w - 1)
    for start in range(0, shape[0], 128):
        _check_cancel(should_cancel)
        rows = slice(start, min(shape[0], start + 128))
        top = (np.arange(rows.start, rows.stop) * 2)[:, None]
        bottom = np.minimum(top + 1, h - 1)
        out[rows] = ((image[top, left] + image[top, right]
                      + image[bottom, left] + image[bottom, right]) >> 2)
    return out


def _expand_axis(image, axis, should_cancel=None):
    """Four-tap phase-mirrored expansion with bounded row scratch."""
    from fusion_methods.openfocus_ab import _check_cancel
    n = image.shape[axis]
    shape = list(image.shape)
    shape[axis] = 2 * n
    out = create_frame(tuple(shape), np.int32)
    for start in range(0, image.shape[0], 128):
        _check_cancel(should_cancel)
        stop = min(image.shape[0], start + 128)
        source = image[start:stop] if axis else image
        positions = np.arange(n) if axis else np.arange(start, stop)
        for phase, offsets, weights in (
                (0, (-2, -1, 0, 1), (2, 37, 64, 12)),
                (1, (-1, 0, 1, 2), (12, 64, 37, 2))):
            accum = np.zeros((stop - start,) + image.shape[1:], np.int32)
            for offset, weight in zip(offsets, weights):
                accum += np.take(source, np.clip(positions + offset, 0, n - 1),
                                 axis=axis) * weight
            if axis:
                out[start:stop, phase::2] = (accum + 57) // 115
            else:
                out[2 * start + phase:2 * stop:2] = (accum + 57) // 115
    return out


def _expand(image, shape, should_cancel=None):
    return _expand_axis(_expand_axis(image, 0, should_cancel), 1,
                        should_cancel)[:shape[0], :shape[1]]


def _inputs(input_source, img_resize, should_cancel):
    from fusion_methods.openfocus_ab import _collect_images
    frames = _collect_images(input_source, normalize=False, should_cancel=should_cancel)
    shape = frames[0].shape
    if img_resize is not None:
        shape = (int(img_resize[1]), int(img_resize[0])) + shape[2:]
    return frames, shape


def _output(image, frames, should_cancel):
    from fusion_methods.openfocus_ab import _check_cancel
    dtype = np.uint8 if all(frame.dtype == np.uint8 for frame in frames) else np.float32
    output = create_frame(image.shape, dtype)
    for start in range(0, image.shape[0], 128):
        _check_cancel(should_cancel)
        rows = slice(start, min(image.shape[0], start + 128))
        output[rows] = np.clip(image[rows], 0, 255)
    return output

def _depth(frames, shape, img_resize, kernel_size, smoothing, should_cancel,
           depth_override, thread_count, progress_callback=None):
    if depth_override is None:
        return depth_map(frames, img_resize, kernel_size, smoothing, should_cancel,
                         progress_callback=progress_callback, thread_count=thread_count)
    depth = np.asarray(depth_override, dtype=np.float32)
    if depth.shape != shape[:2]:
        raise ValueError("depth_override must match the source height and width")
    return depth


def openfocus_b_impl(input_source, img_resize=None, kernel_size: int = 11,
                     thread_count: int = None, depth_smoothing: float = 1.2,
                     should_cancel=None, depth_override=None, progress_callback=None):
    """Fuse source pixels using a continuous, smoothed depth map."""
    from fusion_methods.openfocus_ab import _check_cancel, _prepare_frame
    frames, shape = _inputs(input_source, img_resize, should_cancel)
    smoothing = 1.2 if depth_smoothing is None else float(depth_smoothing)
    count = len(frames)
    activity_steps = count if depth_override is None else 0
    total = activity_steps + count + 1
    if progress_callback is not None:
        progress_callback(0, total)
    _check_cancel(should_cancel)
    def depth_progress(index, _total):
        if progress_callback is not None:
            progress_callback(index + 1, total)
    depth = _depth(frames, shape, img_resize, kernel_size,
                   smoothing * (5.0 / 1.2), should_cancel, depth_override,
                   thread_count, depth_progress)
    result = create_frame(shape, np.float32)
    result[...] = 0
    for index, source in enumerate(frames):
        _check_cancel(should_cancel)
        frame = _prepare_frame(source, img_resize, should_cancel)
        if frame.shape != shape:
            raise ValueError("OpenFocus B requires frames with matching shapes")
        for start in range(0, shape[0], 128):
            _check_cancel(should_cancel)
            rows = slice(start, min(shape[0], start + 128))
            weight = _weight(depth[rows], index)
            result[rows] += frame[rows] * (weight[:, :, None] if frame.ndim == 3 else weight)
        del frame, weight
        if progress_callback is not None:
            progress_callback(activity_steps + index + 1, total)
    _check_cancel(should_cancel)
    output = _output(result, frames, should_cancel)
    if progress_callback is not None:
        progress_callback(total, total)
    _check_cancel(should_cancel)
    return output


def openfocus_c_impl(input_source, img_resize=None, kernel_size: int = 7,
                     thread_count: int = None, levels: int = DEFAULT_PYRAMID_LEVELS,
                     blend_smoothing: float = 3.0, should_cancel=None,
                     depth_override=None, progress_callback=None):
    """Fuse depth-guided pyramids with Q10 merge and integer collapse."""
    from fusion_methods.openfocus_ab import _check_cancel, _prepare_frame
    frames, shape = _inputs(input_source, img_resize, should_cancel)
    smoothing = 3.0 if blend_smoothing is None else float(blend_smoothing)
    # Method C exposes smoothing only; use the default radius internally.
    count = len(frames)
    activity_steps = count if depth_override is None else 0
    total = activity_steps + count + 1
    if progress_callback is not None:
        progress_callback(0, total)
    _check_cancel(should_cancel)
    def depth_progress(index, _total):
        if progress_callback is not None:
            progress_callback(index + 1, total)
    depth = _depth(frames, shape, img_resize, 11,
                   smoothing * (5.0 / 3.0), should_cancel, depth_override,
                   thread_count, depth_progress)
    depth_code = create_frame(depth.shape, np.int32)
    for start in range(0, depth.shape[0], 128):
        _check_cancel(should_cancel)
        rows = slice(start, min(depth.shape[0], start + 128))
        depth_code[rows] = np.rint(depth[rows] * 1024).astype(np.int32)
    del depth
    depth_levels = [depth_code]
    for _ in range(max(1, int(levels)) - 1):
        _check_cancel(should_cancel)
        if max(depth_levels[-1].shape) <= 1:
            break
        depth_levels.append(_reduce(depth_levels[-1], should_cancel))
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
        current = create_frame(shape, np.int32)
        for start in range(0, shape[0], 128):
            _check_cancel(should_cancel)
            rows = slice(start, min(shape[0], start + 128))
            current[rows] = np.rint(np.clip(frame[rows], 0, 255) * scale).astype(np.int32)
        del frame
        for level, code in enumerate(depth_levels):
            _check_cancel(should_cancel)
            if level + 1 < len(depth_levels):
                reduced = _reduce(current, should_cancel)
                expanded = _expand(reduced, current.shape, should_cancel)
            for start in range(0, code.shape[0], 128):
                _check_cancel(should_cancel)
                rows = slice(start, min(code.shape[0], start + 128))
                band = (current[rows] - expanded[rows]
                        if level + 1 < len(depth_levels) else current[rows])
                weight = np.rint(_weight(code[rows].astype(np.float32) / 1024, index)
                                 * 1024).astype(np.int32)
                if current.ndim == 3:
                    weight = weight[:, :, None]
                blends[level][rows] += band * weight
            del weight, band
            if level + 1 < len(depth_levels):
                current = reduced
                del reduced, expanded
        del current
        if progress_callback is not None:
            progress_callback(activity_steps + index + 1, total)
    result = create_frame(blends[-1].shape, np.int32)
    coarse = blends.pop()
    for start in range(0, result.shape[0], 128):
        _check_cancel(should_cancel)
        rows = slice(start, min(result.shape[0], start + 128))
        result[rows] = np.clip((coarse[rows] + 512) >> 10, 0, 65535)
    del coarse
    while blends:
        _check_cancel(should_cancel)
        detail = blends.pop()
        expanded = _expand(result, detail.shape, should_cancel)
        result = create_frame(detail.shape, np.int32)
        for start in range(0, result.shape[0], 128):
            _check_cancel(should_cancel)
            rows = slice(start, min(result.shape[0], start + 128))
            result[rows] = np.clip(expanded[rows] + ((detail[rows] + 512) >> 10), 0, 65535)
        del detail, expanded
    _check_cancel(should_cancel)
    output = create_frame(result.shape, np.uint8 if is_8bit else np.float32)
    for start in range(0, result.shape[0], 128):
        _check_cancel(should_cancel)
        rows = slice(start, min(result.shape[0], start + 128))
        output[rows] = ((result[rows] >> 8) if is_8bit else
                        result[rows].astype(np.float32) * np.float32(255.0 / 65535.0))
    if progress_callback is not None:
        progress_callback(total, total)
    _check_cancel(should_cancel)
    return output
