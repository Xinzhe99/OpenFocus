"""Streaming depth estimation returning a continuous frame index."""
import ctypes
from functools import lru_cache
import numpy as np
from core.frame_storage import create_transient_frame as create_frame
from core.frame_storage import frame_region
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


# Rows advanced per native call in one 64-column strip. Each call takes the
# GIL for dispatch and a cancel check; 128-row calls (about 8k samples) made
# that the bottleneck with several workers. A 2048-row call still lasts only
# a fraction of a millisecond, so cancellation stays prompt.
_AXIS_BLOCK_ROWS = 2048


def _axis(src, guide, main, low, count, axis, cancel, allocate, *, output=None):
    from core._native import load_native
    from fusion_methods import _depth_kernel
    from fusion_methods._depth_kernel import axis_block, _pack_rows_T, _unpack_rows_T
    from fusion_methods._openfocus_execution import _current
    cancel = cancellation_check(cancel)

    src_arr = np.asarray(src)
    if type(src_arr) is not np.ndarray:
        src_arr = src_arr.view(np.ndarray)
    guide_arr = None if guide is None else np.asarray(guide)
    if guide_arr is not None and type(guide_arr) is not np.ndarray:
        guide_arr = guide_arr.view(np.ndarray)
    native = (load_native()
              if axis_block is getattr(_depth_kernel, '_DEFAULT_AXIS_BLOCK', axis_block)
              else None)
    if (native is not None and hasattr(native, 'of_depth_axis_iir')
            and src_arr.ndim == 2 and src_arr.dtype == np.uint16 and src_arr.flags.c_contiguous
            and (guide_arr is None or (guide_arr.ndim == 2 and guide_arr.dtype == np.uint16
                                       and guide_arr.flags.c_contiguous))):
        h, w = src_arr.shape
        out_arr = allocate((h, w), np.uint16) if output is None else output
        out_pixels = np.asarray(out_arr)
        if type(out_pixels) is not np.ndarray:
            out_pixels = out_pixels.view(np.ndarray)
        if out_pixels.ndim == 2 and out_pixels.dtype == np.uint16 and out_pixels.flags.c_contiguous:
            q = t = tm = tl = 0
            if guide_arr is None:
                q = _weight(main)
                t = (4096 - q) >> 1
            else:
                tm = (4096 - _weight(main)) >> 1
                tl = (4096 - _weight(low)) >> 1
            job = _current.get()
            n_threads = job.workers if (job is not None and job.row_mode) else 1
            cancel()
            native.of_depth_axis_iir(src_arr, guide_arr, out_pixels, q, t, tm, tl,
                                     int(count), 1 if axis else 0, max(1, int(n_threads)))
            cancel()
            return out_arr

    src = src_arr
    guide = guide_arr
    if axis:
        orig_src = src
        orig_guide = guide
        src = src.T
        if guide is not None:
            guide = guide.T
    else:
        orig_src = None
        orig_guide = None
    h, w = src.shape
    count = max(1, min(int(count), h))
    # Pack horizontal strips into contiguous scratch so both recurrence sweeps
    # stay vectorizable and avoid cache-set thrashing on wide power-of-two rows.
    pack = bool(axis)
    if output is None:
        out = allocate((h, w), np.uint16)
    else:
        out = output.T if axis else output
    out_pixels = np.asarray(out)
    orig_out = out_pixels.T if axis else None
    fast_pack_src = bool(
        pack and orig_src is not None and orig_out is not None
        and orig_src.dtype == np.uint16 and orig_out.dtype == np.uint16
        and orig_src.flags.c_contiguous and orig_out.flags.c_contiguous
    )
    fast_pack_guide = bool(
        pack and orig_guide is not None
        and orig_guide.dtype == np.uint16 and orig_guide.flags.c_contiguous
    )
    q = t = tm = tl = 0
    if guide is None:
        q = _weight(main)
        t = (4096 - q) >> 1
    else:
        tm = (4096 - _weight(main)) >> 1
        tl = (4096 - _weight(low)) >> 1
    # Both recurrence directions stay ordered inside each independent strip.
    # Each worker reuses its history plane for all its assigned columns.
    def columns(left, right, workspace):
        first, packed_src, packed_guide, packed_out = workspace
        first = np.asarray(first)
        strip_cols = right - left
        if pack:
            source = np.asarray(packed_src)
            if fast_pack_src:
                _pack_rows_T(orig_src, source, left, right)
            else:
                source[:, :strip_cols] = src[:, left:right]
            if guide is not None:
                strip_guide = np.asarray(packed_guide)
                if fast_pack_guide:
                    _pack_rows_T(orig_guide, strip_guide, left, right)
                else:
                    strip_guide[:, :strip_cols] = guide[:, left:right]
                destination = np.asarray(packed_out)
            else:
                strip_guide = None
                destination = source
            begin, end = 0, strip_cols
        else:
            source, strip_guide, destination = src, guide, out_pixels
            begin, end = left, right
        initial = ((source[:count, begin:end].sum(axis=0, dtype=np.uint64) << 4)
                   // count)
        if strip_guide is None:
            initial *= t + q
        acc = (initial & 0xffffffff).astype(np.uint32)
        for start in range(0, h, _AXIS_BLOCK_ROWS):
            cancel()
            axis_block(source, strip_guide, first, destination, acc, q, t, tm, tl,
                       start, min(h, start + _AXIS_BLOCK_ROWS), begin, end, False)
        acc = first[-1, :end - begin].astype(np.uint32) << 4
        if strip_guide is None:
            acc *= t + q
        for stop in range(h, 0, -_AXIS_BLOCK_ROWS):
            cancel()
            axis_block(source, strip_guide, first, destination, acc, q, t, tm, tl,
                       max(0, stop - _AXIS_BLOCK_ROWS), stop, begin, end, True)
        if pack:
            if fast_pack_src:
                _unpack_rows_T(destination, orig_out, left, right)
            else:
                out_pixels[:, left:right] = destination[:, :strip_cols]

    strip_shape = h, min(w, 64)
    history_dtype = np.uint16 if guide is None else np.uint32
    packed_planes = (3 if guide is not None else 1) if pack else 0
    strip_bytes = h * strip_shape[1] * ((2 if guide is None else 4) + 2 * packed_planes)
    # The row planner admits this bounded workspace independently of the
    # transient-frame cap, which may already be full of long-lived planes.
    scratch = np.empty if allocate is create_frame else allocate
    run_bands(w, 64, columns,
              scratch_bytes=strip_bytes + count * strip_shape[1] * 8 + strip_shape[1] * 32,
              workspace=lambda: (scratch(strip_shape, history_dtype),
                                 scratch(strip_shape, np.uint16) if pack else None,
                                 scratch(strip_shape, np.uint16) if (pack and guide is not None) else None,
                                 scratch(strip_shape, np.uint16) if (pack and guide is not None) else None))
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


def _filter(src, radius, cancel, guide=None, low=None, *, allocate=None, overwrite=False):
    low = radius if low is None else low
    allocate = create_frame if allocate is None else allocate
    # Code planes contain only final 16-bit values. Q12 state and guided
    # forward history remain uint32 inside _axis/axis_block.
    result = _mapped_cast(src, np.uint16, cancel, allocate)
    if radius == 0:
        return result
    if guide is None:
        output = result if overwrite else allocate(src.shape, np.uint16)
        _axis(result, None, radius, low, radius, 1, cancel, allocate, output=output)
        return _axis(output, None, radius, low, radius, 0, cancel, allocate, output=output)
    guide = _mapped_cast(guide, np.uint16, cancel, allocate)
    for axis in (1, 0):
        result = _axis(result, guide, radius, low, radius, axis, cancel, allocate)
    return result


def _activity_frames(frames, img_resize, kernel_size, thread_count, should_cancel,
                     *, half_resolution=False, preserve_detail=False, gpu=None):
    """Return ordered activity planes from the admitted source window."""
    from core.registration import _deferred_ecc_gpu
    from core.source_frames import resolve_frame, without_source_cache_growth
    from fusion_methods.openfocus_ab import (
        _check_cancel, _frame_pass_workers, _ordered_frame_results)
    from fusion_methods._openfocus_pixels import frame_gray_codes
    from fusion_methods._openfocus_depth_rows import activity_rows, detail_rows, smooth_detail_rows

    if not 2 <= len(frames) <= 65536:
        raise ValueError('OpenFocus depth requires between 2 and 65536 source frames')
    workers = _frame_pass_workers(frames, img_resize, 'openfocus_b', thread_count,
                                  activity_only=True)
    if gpu is not None and not gpu.failed:
        workers = 1
    warp = _deferred_ecc_gpu.get()
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

        # Activity workers inherit scratch alone. Lend the accelerator without
        # lending the owning job's native threads or its row executor.
        with without_source_cache_growth():
            token = _deferred_ecc_gpu.set(warp)
            try:
                pixels = resolve_frame(frames[index], should_cancel=cancelled)
            finally:
                _deferred_ecc_gpu.reset(token)
        with frame_region(pixels) as source:
            gray = frame_gray_codes(source, img_resize, cancelled, depth_color=True)
        del pixels
        if gray.shape != shape:
            raise ValueError('OpenFocus depth requires frames with matching shapes')
        detail = None
        if preserve_detail:
            height, width = max(1, shape[0] // 2), max(1, shape[1] // 2)
            detail = create_frame((height, width), np.uint16)
            band = 64
            scratch_rows = min(height, band + 4)

            def measure_detail(start, stop, workspace):
                cancel()
                pooled, columns = workspace
                # Reflect only at image boundaries. Interior bands include
                # the two pooled rows needed on each side by the 5x5 mean.
                first = max(0, start - 2)
                last = min(height, stop + 2)
                pooled = pooled[:last - first]
                detail_rows(gray, pooled, first)
                cancel()
                smooth_detail_rows(pooled, detail[first:stop], start - first,
                                   stop - first, columns)

            run_bands(height, band, measure_detail,
                      scratch_bytes=width * (2 * scratch_rows + 4),
                      workspace=lambda: (create_frame((scratch_rows, width), np.uint16),
                                         np.empty(width, np.int32)))
        if half_resolution:
            from fusion_methods._openfocus_resample import half_gray
            gray = half_gray(gray, cancelled, axes=axes)
        extent = np.float32(max(gray.shape))
        fine_radius = np.float32(extent * fine_parameter / np.float32(100))
        coarse_radius = np.float32(extent * coarse_parameter / np.float32(100))
        fine = _filter(gray, fine_radius, cancel)
        # The pointwise difference can replace the fine filter's result.
        diff = fine
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
        activity = _filter(diff, coarse_radius, cancel, overwrite=True)
        return activity, detail

    return _ordered_frame_results(len(frames), produce, workers, should_cancel)


@execution_job
def depth_map(frames, img_resize, kernel_size, smoothing, should_cancel,
              progress_callback=None, thread_count=1, *, return_codes=False,
              half_resolution=False, preserve_detail=False, feather: int = 0,
              use_gpu=False, device_callback=None, _gpu=None, _live_preview=None):
    """Estimate depth with the specified integer feather, defaulting to F0."""
    _validate_feather(feather)
    from fusion_methods.openfocus_ab import _check_cancel

    if not 2 <= len(frames) <= 65536:
        raise ValueError('OpenFocus depth requires between 2 and 65536 source frames')
    if preserve_detail and not half_resolution:
        raise ValueError('Detail preservation requires half-resolution depth')
    algorithm = 'openfocus_c' if half_resolution else 'openfocus_b'
    configure_rows(frames, img_resize, algorithm, thread_count, should_cancel)

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
                                         progress_callback, live_preview=_live_preview)
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


    from fusion_methods._depth_kernel import prepare_depth_kernel
    from fusion_methods._openfocus_depth_rows import (
        apply_detail_rows, confidence_hist_rows, detail_confidence_rows,
        guide_rows, initialize_activity_rows, initialize_detail_rows, reciprocal32,
        select_activity_rows, select_detail_rows, snap_rows,
    )
    smooth = np.float32(np.float32(smoothing) * np.float32(0.75) * np.float32(0.03125))
    prepare_depth_kernel(guided=smooth != 0, should_cancel=should_cancel)

    step = 65535 // (len(frames) - 1)
    reciprocal = reciprocal32(len(frames))
    best = depth = votes = None
    detail_best = detail_second = detail_winner = None
    produced = _activity_frames(frames, img_resize, kernel_size, thread_count,
                                should_cancel, half_resolution=half_resolution,
                                preserve_detail=preserve_detail, gpu=gpu)
    try:
        for index, (activity, detail) in produced:
            cancel()
            if preserve_detail:
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
                extent = np.float32(max(depth.shape))
                # Activity votes are used only to build the smoothing guide.
                scale = np.float32(extent * smooth / np.float32(100))
                if scale != 0:
                    votes = create_frame(activity.shape, np.uint16)
                def initialize(start, stop):
                    cancel()
                    initialize_activity_rows(activity, depth, votes, reciprocal,
                                             start, stop)
                run_bands(activity.shape[0], 128, initialize)
            else:
                if activity.shape != best.shape:
                    raise ValueError('OpenFocus depth requires frames with matching shapes')
                code = index * step
                def select(start, stop):
                    cancel()
                    select_activity_rows(activity, best, depth, votes, code,
                                         reciprocal, start, stop)
                run_bands(activity.shape[0], 128, select)
            if _live_preview is not None:
                _live_preview.show_positions(depth, index, len(frames), step * (len(frames) - 1))
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
    if votes is not None:
        hist = np.zeros(65536, np.int64)
        for rows in _rows(best, cancel):
            confidence_hist_rows(best, votes, hist, rows.start, rows.stop)
        del votes
        threshold = int(np.float32(np.float32(best.size) * np.float32(0.009999999776482582)))
        lo = int(np.searchsorted(np.cumsum(hist), threshold)) if threshold else 0
        hi = (65535 - int(np.searchsorted(np.cumsum(hist[::-1]), threshold))
              if threshold else 65535)
        positions = np.arange(4096, dtype=np.float64) * 65535.0 / 4095.0
        span = max(hi - lo, 1)
        guide_lut = (65535.0 * np.sqrt(np.clip((positions - lo) / span, 0, 1))).astype(np.float32)
        # Reuse the confidence plane for the pointwise guide transform.
        def make_guide(start, stop):
            cancel()
            guide_rows(best, guide_lut, start, stop)
        run_bands(best.shape[0], 128, make_guide)
        low = np.float32(extent * np.float32(smooth * np.float32(64)) / np.float32(100))
        for _ in range(2):
            depth = _filter(depth, scale, cancel, best, low)
    del best
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
