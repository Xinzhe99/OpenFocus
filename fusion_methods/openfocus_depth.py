"""Streaming depth estimation returning a continuous frame index."""
import ctypes
from functools import lru_cache
import numpy as np
from core.frame_storage import create_transient_frame as create_frame
from fusion_methods._openfocus_execution import (
    execution_job, configure_rows, run_bands, cancellation_check,
)


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
def _weight(radius):
    r = np.float32(max(float(radius), 1e-6))
    half = np.float32(r / np.float32(3.0))
    exponent = np.float32(np.float32(1.0) / np.float32(half + half))
    value = np.float32(_powf32(np.float32(0.045999825), exponent) * np.float32(4096))
    weight = max(1, int(value)) & ~1
    return max(weight, 2)


def _validate_feather(feather):
    if isinstance(feather, (bool, np.bool_)) or not isinstance(feather, (int, np.integer)) or not 0 <= feather <= 10:
        raise ValueError('feather must be an integer between 0 and 10')


@lru_cache(maxsize=128)
def _snap_lut(feather=0):
    exponent = np.float32((np.float32(13) - np.float32(feather)) / np.float32(3))
    return np.array([int(np.float32(_powf32(np.float32(k) / np.float32(1024),
                                          exponent) * np.float32(1024)))
                     for k in range(1025)], dtype=np.int64)


def _axis(src, guide, main, low, count, axis, cancel, allocate):
    from fusion_methods._depth_kernel import axis_block
    cancel = cancellation_check(cancel)

    src = np.asarray(src)
    guide = None if guide is None else np.asarray(guide)
    if axis:
        src = src.T
        if guide is not None:
            guide = guide.T
    h, w = src.shape
    count = max(1, min(int(count), h))
    out = allocate((h, w), np.uint16)
    out_pixels = np.asarray(out)
    q = t = tm = tl = 0
    if guide is None:
        q = _weight(main)
        t = (4096 - q) >> 1
    else:
        tm = (4096 - _weight(main)) >> 1
        tl = (4096 - _weight(low)) >> 1
    # Both recurrence directions stay ordered inside each independent strip.
    # Each worker reuses its history plane for all its assigned columns.
    def columns(left, right, first):
        first_pixels = np.asarray(first)
        initial = ((src[:count, left:right].sum(axis=0, dtype=np.uint64) << 4)
                   // count)
        if guide is None:
            initial *= t + q
        acc = (initial & 0xffffffff).astype(np.uint32)
        for start in range(0, h, 128):
            cancel()
            axis_block(src, guide, first_pixels, out_pixels, acc, q, t, tm, tl,
                       start, min(h, start + 128), left, right, False)
        acc = first_pixels[-1, :right - left].astype(np.uint32) << 4
        if guide is None:
            acc *= t + q
        for stop in range(h, 0, -128):
            cancel()
            axis_block(src, guide, first_pixels, out_pixels, acc, q, t, tm, tl,
                       max(0, stop - 128), stop, left, right, True)
    run_bands(w, 64, columns,
              scratch_bytes=h * min(w, 64) * (2 if guide is None else 4)
              + count * min(w, 64) * 8 + min(w, 64) * 32,
              workspace=lambda: allocate((h, min(w, 64)),
                                         np.uint16 if guide is None else np.uint32))
    return out.T if axis else out


def _rows(image, cancel):
    for start in range(0, image.shape[0], 128):
        cancel()
        yield slice(start, min(image.shape[0], start + 128))


def _mapped_cast(image, dtype, cancel, allocate):
    if image.dtype == dtype:
        return image
    result = allocate(image.shape, dtype)
    def cast(start, stop):
        cancel()
        result[start:stop] = image[start:stop]
    run_bands(image.shape[0], 128, cast)
    return result


def _filter(src, radius, cancel, guide=None, low=None, *, allocate=None):
    low = radius if low is None else low
    allocate = create_frame if allocate is None else allocate
    # Code planes contain only final 16-bit values. Q12 state and guided
    # forward history remain uint32 inside _axis/axis_block.
    result = _mapped_cast(src, np.uint16, cancel, allocate)
    if radius == 0:
        return result
    guide = (None if guide is None
             else _mapped_cast(guide, np.uint16, cancel, allocate))
    for axis in (1, 0):
        result = _axis(result, guide, radius, low, radius, axis, cancel, allocate)
    return result


def _activity_frames(frames, img_resize, kernel_size, thread_count, should_cancel,
                     *, half_resolution=False, preserve_detail=False, gpu=None):
    """Return ordered activity planes from the admitted source window."""
    from core.source_frames import resolve_frame
    from fusion_methods.openfocus_ab import (
        _check_cancel, _frame_pass_workers, _ordered_frame_results)
    from fusion_methods._depth_kernel import prepare_depth_kernel
    from fusion_methods._openfocus_pixels import frame_gray_codes
    from fusion_methods._openfocus_depth_rows import activity_rows, detail_rows, smooth_detail_rows

    if not 2 <= len(frames) <= 65536:
        raise ValueError('OpenFocus depth requires between 2 and 65536 source frames')
    prepare_depth_kernel(guided=False, should_cancel=should_cancel)
    workers = _frame_pass_workers(frames, img_resize, 'openfocus_b', thread_count,
                                  activity_only=True)
    if gpu is not None and not gpu.failed:
        workers = 1
    shape = (frames[0].shape[:2] if img_resize is None
             else (int(img_resize[1]), int(img_resize[0])))
    axes = None
    if half_resolution:
        from fusion_methods._openfocus_resample import half_gray_axes
        axes = half_gray_axes(*shape)
    radius = max(1, int(kernel_size or 17) // 2)
    parameter = float(np.float64(radius) * np.float64(np.float32(0.075)))
    fine_parameter = np.float32(parameter * 0.5)
    coarse_parameter = np.float32(parameter * 2.0)

    def produce(index, cancelled):
        def cancel():
            _check_cancel(cancelled)

        pixels = resolve_frame(frames[index])
        gray = frame_gray_codes(pixels, img_resize, cancelled, depth_color=True)
        del pixels
        if gray.shape != shape:
            raise ValueError('OpenFocus depth requires frames with matching shapes')
        detail = None
        if preserve_detail:
            detail_shape = (max(1, shape[0] // 2), max(1, shape[1] // 2))
            pooled = create_frame(detail_shape, np.uint16)
            detail = create_frame(detail_shape, np.uint16)
            def measure_detail(start, stop):
                cancel()
                detail_rows(gray, pooled, start, stop)
            run_bands(detail_shape[0], 128, measure_detail)
            def smooth_detail(start, stop):
                cancel()
                smooth_detail_rows(pooled, detail, start, stop)
            run_bands(detail_shape[0], 128, smooth_detail)
            del pooled
        if half_resolution:
            from fusion_methods._openfocus_resample import half_gray
            gray = half_gray(gray, cancelled, axes=axes)
        extent = np.float32(max(gray.shape))
        fine_radius = np.float32(extent * fine_parameter / np.float32(100))
        coarse_radius = np.float32(extent * coarse_parameter / np.float32(100))
        fine = _filter(gray, fine_radius, cancel)
        diff = create_frame(gray.shape, np.uint16)
        if gpu is not None and not gpu.failed:
            from fusion_methods._openfocus_gpu import GpuError, log_fallback
            for start in range(0, gray.shape[0], gpu.rows):
                cancel()
                stop = min(start + gpu.rows, gray.shape[0])
                if not gpu.failed:
                    gpu.acquire(cancelled)
                    try:
                        try:
                            gpu.activity(gray[start:stop], fine[start:stop], diff[start:stop])
                            gpu.used = True
                            continue
                        except GpuError as exc:
                            gpu.failed = True
                            log_fallback(exc)
                    finally:
                        gpu.release()
                activity_rows(gray, fine, diff, start, stop)
        else:
            def activity(start, stop):
                cancel()
                activity_rows(gray, fine, diff, start, stop)
            run_bands(gray.shape[0], 128, activity)
        del gray, fine
        activity = _filter(diff, coarse_radius, cancel)
        return (activity, detail) if preserve_detail else activity

    return _ordered_frame_results(len(frames), produce, workers, should_cancel)


@execution_job
def depth_map(frames, img_resize, kernel_size, smoothing, should_cancel,
              progress_callback=None, thread_count=1, *, return_codes=False,
              half_resolution=False, preserve_detail=False, feather: int = 0,
              use_gpu=False, device_callback=None, _gpu=None):
    """Estimate depth with the specified integer feather, defaulting to F0."""
    _validate_feather(feather)
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._depth_kernel import prepare_depth_kernel
    from fusion_methods._openfocus_depth_rows import (
        initialize_activity_rows, select_activity_rows, confidence_rows,
        guide_rows, snap_rows, initialize_detail_rows, select_detail_rows,
        detail_confidence_rows, apply_detail_rows)

    if not 2 <= len(frames) <= 65536:
        raise ValueError('OpenFocus depth requires between 2 and 65536 source frames')
    if preserve_detail and not half_resolution:
        raise ValueError('Detail preservation requires half-resolution depth')
    algorithm = 'openfocus_c' if half_resolution else 'openfocus_b'
    configure_rows(frames, img_resize, algorithm, thread_count, should_cancel)
    prepare_depth_kernel(guided=True, should_cancel=should_cancel)

    def cancel():
        _check_cancel(should_cancel)
    gpu = _gpu
    if gpu is None and use_gpu:
        from fusion_methods._openfocus_gpu import Merge, GpuError, log_fallback
        width = frames[0].shape[1] if img_resize is None else int(img_resize[0])
        try:
            gpu = Merge(width, 1)
        except GpuError as exc:
            log_fallback(exc)
    if progress_callback is not None:
        callback = progress_callback
        reported = -1
        def progress_callback(index, total):
            nonlocal reported
            if index > reported:
                callback(index, total)
                reported = index

    if gpu is not None and half_resolution and preserve_detail:
        from fusion_methods._openfocus_gpu import GpuError, log_fallback, device_status
        shape = (frames[0].shape[:2] if img_resize is None
                 else (int(img_resize[1]), int(img_resize[0])))
        try:
            accelerated = gpu.full_depth(frames, shape, img_resize, kernel_size,
                                         smoothing, feather, should_cancel,
                                         progress_callback)
        except GpuError as exc:
            gpu.failed = True
            log_fallback(exc)
            accelerated = None
        if accelerated is not None:
            cancel()
            if device_callback is not None:
                device_callback(device_status(gpu))
            if return_codes:
                return accelerated
            step = 65535 // (len(frames) - 1)
            result = create_frame(accelerated.shape, np.float32)
            def convert_gpu(start, stop):
                cancel()
                result[start:stop] = accelerated[start:stop].astype(np.float32) / np.float32(step)
            run_bands(accelerated.shape[0], 128, convert_gpu,
                      scratch_bytes=min(128, accelerated.shape[0]) * accelerated.shape[1] * 8)
            return result


    step = 65535 // (len(frames) - 1)
    best = depth = votes = None
    detail_best = detail_second = detail_winner = None
    produced = _activity_frames(frames, img_resize, kernel_size, thread_count,
                                should_cancel, half_resolution=half_resolution,
                                preserve_detail=preserve_detail, gpu=gpu)
    try:
        for index, activity in produced:
            cancel()
            if preserve_detail:
                activity, detail = activity
                if detail_best is None:
                    detail_best = detail
                    detail_second = create_frame(detail.shape, np.uint16)
                    detail_winner = create_frame(detail.shape, np.uint16)
                    def initialize_detail(start, stop):
                        cancel()
                        initialize_detail_rows(detail_second, detail_winner, start, stop)
                    run_bands(detail.shape[0], 128, initialize_detail)
                else:
                    def select_detail(start, stop):
                        cancel()
                        select_detail_rows(detail, detail_best, detail_second,
                                           detail_winner, index * step, start, stop)
                    run_bands(detail.shape[0], 128, select_detail)
                del detail
            if best is None:
                best = activity
                depth = create_frame(activity.shape, np.uint16)
                votes = create_frame(activity.shape, np.uint16)
                def initialize(start, stop):
                    cancel()
                    initialize_activity_rows(activity, depth, votes, len(frames),
                                             start, stop)
                run_bands(activity.shape[0], 128, initialize)
            else:
                if activity.shape != best.shape:
                    raise ValueError('OpenFocus depth requires frames with matching shapes')
                code = index * step
                def select(start, stop):
                    cancel()
                    select_activity_rows(activity, best, depth, votes, code,
                                         len(frames), start, stop)
                run_bands(activity.shape[0], 128, select)
            if progress_callback is not None:
                progress_callback(index, len(frames))
            del activity
    finally:
        produced.close()
    cancel()
    if preserve_detail:
        # Only replace low-frequency choices when the full-size signal has a
        # clear, strong winner; smoothing each score rejects isolated noise.
        def detail_confidence(start, stop):
            cancel()
            detail_confidence_rows(best, detail_best, detail_second, start, stop)
        run_bands(best.shape[0], 128, detail_confidence)
        del detail_best
    hist = np.zeros(65536, np.int64)
    for rows in _rows(best, cancel):
        confidence_rows(best, votes, rows.start, rows.stop)
        hist += np.bincount(best[rows].ravel(), minlength=65536)
    del votes
    threshold = int(np.float32(np.float32(best.size) * np.float32(0.009999999776482582)))
    lo = int(np.searchsorted(np.cumsum(hist), threshold)) if threshold else 0
    hi = (65535 - int(np.searchsorted(np.cumsum(hist[::-1]), threshold))
          if threshold else 65535)
    positions = np.arange(4096, dtype=np.float64) * 65535.0 / 4095.0
    span = max(hi - lo, 1)
    guide_lut = (65535.0 * np.sqrt(np.clip((positions - lo) / span, 0, 1))).astype(np.float32)
    guide = create_frame(best.shape, np.uint16)
    def make_guide(start, stop):
        cancel()
        guide_rows(best, guide_lut, guide, start, stop)
    run_bands(best.shape[0], 128, make_guide)
    del best
    smooth = np.float32(np.float32(smoothing) * np.float32(0.75) * np.float32(0.03125))
    extent = np.float32(max(depth.shape))
    scale = np.float32(extent * smooth / np.float32(100))
    low = np.float32(extent * np.float32(smooth * np.float32(64)) / np.float32(100))
    for _ in range(2):
        depth = _filter(depth, scale, cancel, guide, low)
    del guide
    half = step >> 1
    if half:
        lut = _snap_lut(feather)
        def snap(start, stop):
            cancel()
            snap_rows(depth, lut, step, start, stop)
        run_bands(depth.shape[0], 128, snap)
    if preserve_detail:
        def apply_detail(start, stop):
            cancel()
            apply_detail_rows(depth, detail_second, detail_winner, start, stop)
        run_bands(depth.shape[0], 128, apply_detail)
        del detail_second, detail_winner
    if half_resolution:
        from fusion_methods._openfocus_resample import expand_depth
        shape = (frames[0].shape[:2] if img_resize is None
                 else (int(img_resize[1]), int(img_resize[0])))
        depth = expand_depth(depth, shape, should_cancel)
    if device_callback is not None:
        cancel()
        from fusion_methods._openfocus_gpu import device_status
        device_callback(device_status(gpu))
    if return_codes:
        return depth
    result = create_frame(depth.shape, np.float32)
    def convert(start, stop):
        cancel()
        result[start:stop] = depth[start:stop].astype(np.float32) / np.float32(step)
    run_bands(depth.shape[0], 128, convert,
              scratch_bytes=min(128, depth.shape[0]) * depth.shape[1] * 8)
    return result
