"""Streaming depth estimation returning a continuous frame index."""
import ctypes
from functools import lru_cache
import numpy as np
from core.frame_storage import create_transient_frame as create_frame


@lru_cache(maxsize=1)
def _powf_function():
    for name in ('ucrtbase.dll', 'msvcrt.dll', 'libm.so.6', 'libSystem.B.dylib'):
        try:
            function = ctypes.CDLL(name).powf
            function.restype = ctypes.c_float
            function.argtypes = (ctypes.c_float, ctypes.c_float)
            return function
        except (OSError, AttributeError):
            continue
    return None


def _powf32(value, exponent):
    function = _powf_function()
    if function is None:
        return np.float32(np.power(np.float32(value), np.float32(exponent)))
    return np.float32(function(np.float32(value), np.float32(exponent)))


@lru_cache(maxsize=128)
def _weight(radius, allow_zero=False):
    r = np.float32(max(float(radius), 1e-6))
    half = np.float32(r / np.float32(3.0))
    exponent = np.float32(np.float32(1.0) / np.float32(half + half))
    value = np.float32(_powf32(np.float32(0.045999825), exponent) * np.float32(4096))
    weight = max(1, int(value)) & ~1
    return weight if allow_zero else max(weight, 2)


@lru_cache(maxsize=1)
def _snap_lut():
    exponent = np.float32(np.float32(13) / np.float32(3))
    return np.array([int(np.float32(_powf32(np.float32(k) / np.float32(1024),
                                          exponent) * np.float32(1024)))
                     for k in range(1025)], dtype=np.int64)


def _axis(src, guide, main, low, count, axis, cancel, allocate):
    from fusion_methods._depth_kernel import axis_block

    src = np.asarray(src)
    guide = None if guide is None else np.asarray(guide)
    if axis:
        src = src.T
        if guide is not None:
            guide = guide.T
    h, w = src.shape
    count = max(1, min(int(count), h))
    # Guided forward rounding can reach 65536; only that history needs u32.
    first = allocate((h, min(w, 64)),
                     np.uint16 if guide is None else np.uint32)
    out = allocate((h, w), np.uint16)
    first_pixels, out_pixels = np.asarray(first), np.asarray(out)
    q = t = tm = tl = 0
    if guide is None:
        q = _weight(main)
        t = (4096 - q) >> 1
    else:
        tm = (4096 - _weight(main, True)) >> 1
        tl = (4096 - _weight(low, True)) >> 1
    # Columns have independent recurrences. Complete both directions before
    # reusing history for the next strip; rows always retain source order.
    for left in range(0, w, 64):
        right = min(w, left + 64)
        initial = ((src[:count, left:right].sum(axis=0, dtype=np.uint64) << 4)
                   // count)
        if guide is None:
            initial *= t + q
        acc = (initial & 0xffffffff).astype(np.uint32)
        for start in range(0, h, 64):
            cancel()
            axis_block(src, guide, first_pixels, out_pixels, acc, q, t, tm, tl,
                       start, min(h, start + 64), left, right, False)
        acc = first_pixels[-1, :right - left].astype(np.uint32) << 4
        if guide is None:
            acc *= t + q
        for stop in range(h, 0, -64):
            cancel()
            axis_block(src, guide, first_pixels, out_pixels, acc, q, t, tm, tl,
                       max(0, stop - 64), stop, left, right, True)
    return out.T if axis else out


def _rows(image, cancel):
    for start in range(0, image.shape[0], 128):
        cancel()
        yield slice(start, min(image.shape[0], start + 128))


def _mapped_cast(image, dtype, cancel, allocate):
    if image.dtype == dtype:
        return image
    result = allocate(image.shape, dtype)
    for rows in _rows(image, cancel):
        result[rows] = image[rows]
    return result


def _filter(src, radius, cancel, guide=None, low=None, *, allocate=None):
    low = radius if low is None else low
    allocate = create_frame if allocate is None else allocate
    # Code planes contain only final 16-bit values. Q12 state and guided
    # forward history remain uint32 inside _axis/axis_block.
    result = _mapped_cast(src, np.uint16, cancel, allocate)
    guide = (None if guide is None
             else _mapped_cast(guide, np.uint16, cancel, allocate))
    for axis in (0, 1):
        result = _axis(result, guide, radius, low, radius, axis, cancel, allocate)
    return result


def _activity_frames(frames, img_resize, kernel_size, thread_count, should_cancel,
                     *, keep_pixels=False):
    """Return ordered (pixels, activity) payloads from the admitted window."""
    from core.source_frames import resolve_frame
    from fusion_methods.openfocus_ab import (
        _check_cancel, _frame_pass_workers, _ordered_frame_results)
    from fusion_methods._depth_kernel import prepare_depth_kernel
    from fusion_methods._hf9_pixels import frame_gray_codes
    from fusion_methods._hf9_depth_rows import activity_rows

    if not 2 <= len(frames) <= 65536:
        raise ValueError('OpenFocus depth requires between 2 and 65536 source frames')
    prepare_depth_kernel(guided=False, should_cancel=should_cancel)
    workers = _frame_pass_workers(frames, img_resize, 'openfocus_b', thread_count,
                                  activity_only=not keep_pixels)
    radius = max(1, int(kernel_size or 17) // 2)
    parameter = float(np.float64(radius) * np.float64(np.float32(0.075)))
    fine_parameter = np.float32(parameter * 0.5)
    coarse_parameter = np.float32(parameter * 2.0)

    def produce(index, cancelled):
        def cancel():
            _check_cancel(cancelled)

        pixels = resolve_frame(frames[index])
        gray = frame_gray_codes(pixels, img_resize, cancelled)
        retained = pixels if keep_pixels else None
        del pixels
        extent = np.float32(max(gray.shape))
        fine_radius = np.float32(extent * fine_parameter / np.float32(100))
        coarse_radius = np.float32(extent * coarse_parameter / np.float32(100))
        fine = _filter(gray, fine_radius, cancel)
        diff = create_frame(gray.shape, np.uint16)
        for rows in _rows(gray, cancel):
            activity_rows(gray, fine, diff, rows.start, rows.stop)
        del gray, fine
        return retained, _filter(diff, coarse_radius, cancel)

    return _ordered_frame_results(len(frames), produce, workers, should_cancel)


def depth_map(frames, img_resize, kernel_size, smoothing, should_cancel,
              progress_callback=None, thread_count=1, *, return_codes=False):
    """HF9 first pass, percentile guide, two guided passes and feather snap."""
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._depth_kernel import prepare_depth_kernel
    from fusion_methods._hf9_depth_rows import (
        select_activity_rows, guide_rows, winner_depth_rows, snap_rows)

    if not 2 <= len(frames) <= 65536:
        raise ValueError('OpenFocus depth requires between 2 and 65536 source frames')
    prepare_depth_kernel(guided=True, should_cancel=should_cancel)

    def cancel():
        _check_cancel(should_cancel)


    best = winner = None
    produced = _activity_frames(frames, img_resize, kernel_size, thread_count,
                                should_cancel)
    try:
        for index, (_pixels, activity) in produced:
            cancel()
            if best is None:
                best = activity
                winner = create_frame(activity.shape, np.uint16)
                winner[...] = 0
            else:
                if activity.shape != best.shape:
                    raise ValueError('OpenFocus depth requires frames with matching shapes')
                for rows in _rows(activity, cancel):
                    select_activity_rows(activity, best, winner, index,
                                         rows.start, rows.stop)
            if progress_callback is not None:
                progress_callback(index, len(frames))
            del activity
    finally:
        produced.close()
    cancel()
    hist = np.zeros(65536, np.int64)
    for rows in _rows(best, cancel):
        hist += np.bincount(best[rows].ravel(), minlength=65536)
    threshold = int(np.float32(np.float32(best.size) * np.float32(0.009999999776482582)))
    lo = int(np.searchsorted(np.cumsum(hist), threshold)) if threshold else 0
    hi = (65535 - int(np.searchsorted(np.cumsum(hist[::-1]), threshold))
          if threshold else 65535)
    gain = np.float32(np.float32(65535) / np.float32(max(hi - lo, 1)))
    positions = np.arange(65536, dtype=np.float64) / 65535.0
    guide_lut = np.float32(np.clip(positions * gain, 0, 1) * 65536.0)
    np.minimum(guide_lut, np.float32(65535), out=guide_lut)
    guide = create_frame(best.shape, np.uint16)
    for rows in _rows(best, cancel):
        guide_rows(best, guide_lut, guide, rows.start, rows.stop)
    del best
    step = 65535 // (len(frames) - 1)
    depth = create_frame(winner.shape, np.uint16)
    for rows in _rows(winner, cancel):
        winner_depth_rows(winner, depth, step, rows.start, rows.stop)
    del winner
    smooth = np.float32(np.float32(smoothing) * np.float32(0.75) * np.float32(0.03125))
    extent = np.float32(max(depth.shape))
    scale = np.float32(extent * smooth / np.float32(100))
    low = np.float32(extent * np.float32(smooth * np.float32(64)) / np.float32(100))
    for _ in range(2):
        depth = _filter(depth, scale, cancel, guide, low)
    del guide
    half = step >> 1
    if half:
        lut = _snap_lut()
        for rows in _rows(depth, cancel):
            snap_rows(depth, lut, step, rows.start, rows.stop)
    if return_codes:
        return depth
    result = create_frame(depth.shape, np.float32)
    for rows in _rows(depth, cancel):
        result[rows] = depth[rows].astype(np.float32) / np.float32(step)
    return result
