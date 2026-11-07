"""Ordered HF9 code-band fusion with bounded source preparation.

B uses a processed depth map and reads only active source bands. C updates
shared running depth and all pyramid accumulators as each source arrives;
its second stage reconstructs the accumulated levels without rereading frames.
"""
import numpy as np

from fusion_methods.openfocus_depth import depth_map
from core.frame_storage import create_transient_frame as create_frame


def automatic_pyramid_levels(height, width):
    """Stop before a reduction whose shortest side would be below 128."""
    shortest = min(int(height), int(width))
    levels = 1
    while (shortest + 1) // 2 >= 128:
        shortest = (shortest + 1) // 2
        levels += 1
    return levels




def _reduce(image, should_cancel=None, *, source_scale=1):
    """Truncated code-domain 2x2 mean, including native u8 mean-before-shift."""
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._hf9_kernels import down2_rows
    h, w = image.shape[:2]
    shape = ((h + 1) // 2, (w + 1) // 2) + image.shape[2:]
    dtype = np.uint16 if image.dtype in (np.uint8, np.uint16) else image.dtype
    out = create_frame(shape, dtype)
    for start in range(0, shape[0], 128):
        _check_cancel(should_cancel)
        down2_rows(image, out, start, min(shape[0], start + 128), source_scale)
    return out


def _expand_axis(image, axis, should_cancel=None):
    """Four-tap phase-mirrored expansion without gathered row copies."""
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._hf9_kernels import expand_rows
    shape = list(image.shape)
    shape[axis] *= 2
    dtype = np.uint16 if image.dtype == np.uint16 else np.int32
    out = create_frame(tuple(shape), dtype)
    for start in range(0, image.shape[0], 128):
        _check_cancel(should_cancel)
        expand_rows(image, out, axis, start, min(image.shape[0], start + 128))
    return out


def _expand(image, shape, should_cancel=None):
    return _expand_axis(_expand_axis(image, 0, should_cancel), 1,
                        should_cancel)[:shape[0], :shape[1]]


def _inputs(input_source, img_resize, should_cancel):
    from fusion_methods.openfocus_ab import _collect_images
    frames = _collect_images(input_source, normalize=False, should_cancel=should_cancel)
    if len(frames) > 65536:
        raise ValueError('OpenFocus depth requires between 2 and 65536 source frames')
    shape = frames[0].shape
    if img_resize is not None:
        shape = (int(img_resize[1]), int(img_resize[0])) + shape[2:]
    return frames, shape



def _depth(frames, shape, img_resize, kernel_size, smoothing, should_cancel,
           depth_override, thread_count, progress_callback=None, *, return_codes=False):
    if depth_override is None:
        return depth_map(frames, img_resize, kernel_size, smoothing, should_cancel,
                         progress_callback=progress_callback, thread_count=thread_count,
                         return_codes=return_codes)
    depth = np.asarray(depth_override, dtype=np.float32)
    if depth.shape != shape[:2]:
        raise ValueError("depth_override must match the source height and width")
    if return_codes:
        step = 65535 // max(1, len(frames) - 1)
        return np.rint(depth.astype(np.float64) * step).astype(np.int64)
    return depth


def _source_codes(source, img_resize, should_cancel):
    """Borrow native samples; allocate code planes only at the legacy boundary."""
    from core.source_frames import resolve_frame
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._hf9_pixels import frame_codes
    _check_cancel(should_cancel)
    frame = resolve_frame(source)
    _check_cancel(should_cancel)
    if img_resize is None and frame.dtype in (np.uint8, np.uint16):
        return frame, 256 if frame.dtype == np.uint8 else 1
    return frame_codes(frame, img_resize, should_cancel), 1


def _merge_codes(accumulator, current, depth, index, count, should_cancel,
                 *, source_scale=1):
    """Ordered MergeWeighted bands, skipping inactive samples before color reads."""
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._hf9_kernels import merge_rows
    step = 65535 // max(1, count - 1)
    for start in range(0, current.shape[0], 128):
        _check_cancel(should_cancel)
        merge_rows(accumulator, current, depth, index, step, start,
                   min(current.shape[0], start + 128), source_scale)

def _collapse_blend(blend, lowpass, should_cancel):
    """Collapse the owned blend canvas in place after its final merge."""
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._hf9_kernels import collapse_rows
    for start in range(0, blend.shape[0], 128):
        _check_cancel(should_cancel)
        collapse_rows(blend, lowpass, blend, start, min(blend.shape[0], start + 128))
    return blend


def _output_codes(codes, frames, should_cancel, *, source_scale=1):
    from fusion_methods.openfocus_ab import _check_cancel
    is_8bit = all(frame.dtype == np.uint8 for frame in frames)
    output = create_frame(codes.shape, np.uint8 if is_8bit else np.float32)
    for start in range(0, codes.shape[0], 128):
        _check_cancel(should_cancel)
        rows = slice(start, min(codes.shape[0], start + 128))
        if is_8bit:
            output[rows] = codes[rows] if source_scale == 256 else codes[rows] >> 8
        else:
            block = codes[rows].astype(np.float32)
            if source_scale != 1:
                block *= np.float32(source_scale)
            output[rows] = block * np.float32(255.0 / 65535.0)
    return output


def _active_sources(depth, count, should_cancel):
    """Reject whole sources before decoding when no code band uses them."""
    from fusion_methods.openfocus_ab import _check_cancel
    active = np.zeros(count, dtype=bool)
    active[0] = True  # The first merge initializes every output sample.
    step = 65535 // (count - 1)
    for start in range(0, depth.shape[0], 128):
        _check_cancel(should_cancel)
        codes = depth[start:start + 128].astype(np.int64)
        lower = np.clip(codes // step, 0, count - 1)
        active[np.unique(lower)] = True
        fraction = codes % step != 0
        active[np.unique(np.minimum(lower[fraction] + 1, count - 1))] = True
    return active


def openfocus_b_impl(input_source, img_resize=None, kernel_size: int = 17,
                     thread_count: int = None, depth_smoothing: float = 1.44,
                     should_cancel=None, depth_override=None, progress_callback=None):
    """Fuse integer source codes with the HF9 ordered MergeWeighted bands."""
    from core import source_frames
    from fusion_methods.openfocus_ab import _check_cancel
    frames, shape = _inputs(input_source, img_resize, should_cancel)
    smoothing = 1.44 if depth_smoothing is None else float(depth_smoothing)
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
                   thread_count, depth_progress, return_codes=True)
    active = _active_sources(depth, count, should_cancel)
    result = create_frame(shape, np.uint16)
    for index, source in enumerate(frames):
        _check_cancel(should_cancel)
        if img_resize is None and source.shape != shape:
            raise ValueError('OpenFocus B requires frames with matching shapes')
        if not active[index]:
            if isinstance(source, source_frames.ProcessedFrame) or (
                    source.ndim == 3 and source.shape[2] == 4):
                # Lazy processors may enforce alpha/calibration invariants.
                source_frames.resolve_frame(source)
            elif isinstance(source, source_frames.SourceFrame):
                source.validate_source()
            if progress_callback is not None:
                progress_callback(activity_steps + index + 1, total)
            continue
        frame, source_scale = _source_codes(source, img_resize, should_cancel)
        if frame.shape != shape:
            raise ValueError("OpenFocus B requires frames with matching shapes")
        _merge_codes(result, frame, depth, index, count, should_cancel,
                     source_scale=source_scale)
        del frame
        if progress_callback is not None:
            progress_callback(activity_steps + index + 1, total)
    del depth
    source_frames.SOURCE_CACHE.clear()
    _check_cancel(should_cancel)
    output = _output_codes(result, frames, should_cancel)
    if progress_callback is not None:
        progress_callback(total, total)
    _check_cancel(should_cancel)
    return output


def openfocus_c_impl(input_source, img_resize=None, kernel_size: int = 17,
                     thread_count: int = None, levels: int = None,
                     blend_smoothing: float = 3.6, should_cancel=None,
                     depth_override=None, progress_callback=None):
    """Merge each source once using running shared depth, then collapse levels.

    Smoothing belongs to the separately computed depth-map sidecar. Explicit
    depth overrides retain their supplied code-band decisions.
    """
    from core import source_frames
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods.openfocus_depth import _activity_frames, _rows
    from fusion_methods._hf9_depth_rows import select_activity_rows
    from fusion_methods._hf9_kernels import equal_code_rows

    frames, shape = _inputs(input_source, img_resize, should_cancel)
    count = len(frames)
    level_count = (automatic_pyramid_levels(*shape[:2]) if levels is None
                   else max(1, int(levels)))
    shapes = [shape[:2]]
    while len(shapes) < level_count and max(shapes[-1]) > 1:
        h, w = shapes[-1]
        shapes.append(((h + 1) // 2, (w + 1) // 2))
    blends = [create_frame(size + shape[2:], np.uint16) for size in shapes]
    total = count + 1
    if progress_callback is not None:
        progress_callback(0, total)

    def cancel():
        _check_cancel(should_cancel)

    best = running_depth = None
    supplied_levels = None
    if depth_override is None:
        produced = _activity_frames(frames, img_resize, kernel_size, thread_count,
                                    should_cancel, keep_pixels=True)
    else:
        supplied_levels = [_depth(frames, shape, img_resize, kernel_size, 0,
                                  should_cancel, depth_override, thread_count,
                                  return_codes=True)]
        for _ in shapes[1:]:
            supplied_levels.append(_reduce(supplied_levels[-1], should_cancel))
        produced = ((index, (source_frames.resolve_frame(source), None))
                    for index, source in enumerate(frames))
    step = 65535 // (count - 1)
    identical = True
    try:
        for index, (pixels, activity) in produced:
            cancel()
            current, source_scale = _source_codes(pixels, img_resize, should_cancel)
            if current.shape != shape:
                raise ValueError('OpenFocus C requires frames with matching shapes')
            if supplied_levels is None:
                if activity.shape != shape[:2]:
                    raise ValueError('OpenFocus C requires matching activity shapes')
                if best is None:
                    best = create_frame(activity.shape, np.uint16)
                    running_depth = create_frame(activity.shape, np.uint16)
                    for rows in _rows(activity, cancel):
                        best[rows] = activity[rows]
                        running_depth[rows] = 0
                else:
                    for rows in _rows(activity, cancel):
                        select_activity_rows(activity, best, running_depth,
                                             index * step, rows.start, rows.stop)
                code = running_depth
            else:
                code = supplied_levels[0]
            if index and identical:
                # Until a mismatch, level zero still equals source zero. Its
                # owned code plane avoids retaining an extra native frame.
                for rows in _rows(current, cancel):
                    if not equal_code_rows(current, blends[0], rows.start, rows.stop,
                                           source_scale, 1):
                        identical = False
                        break
            for level, blend in enumerate(blends):
                if supplied_levels is not None:
                    code = supplied_levels[level]
                _merge_codes(blend, current, code, index, count, should_cancel,
                             source_scale=source_scale)
                if level + 1 < len(blends):
                    current = _reduce(current, should_cancel, source_scale=source_scale)
                    source_scale = 1
                    if supplied_levels is None:
                        code = _reduce(code, should_cancel)
            del pixels, activity, current, code, blend
            if progress_callback is not None:
                progress_callback(index + 1, total)
    finally:
        produced.close()
        source_frames.SOURCE_CACHE.clear()
    del best, running_depth, supplied_levels
    if identical:
        result = blends.pop(0)
        blends.clear()
    else:
        coarse = blends.pop()
        lowpass = _expand(_reduce(coarse, should_cancel), coarse.shape, should_cancel)
        result = _collapse_blend(coarse, lowpass, should_cancel)
        del coarse, lowpass
        while blends:
            cancel()
            blend = blends.pop()
            lowpass = _expand(result, blend.shape, should_cancel)
            result = _collapse_blend(blend, lowpass, should_cancel)
            del blend, lowpass
    output = _output_codes(result, frames, should_cancel)
    if progress_callback is not None:
        progress_callback(total, total)
    cancel()
    return output
