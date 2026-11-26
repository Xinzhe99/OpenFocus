"""OpenFocus A: Burt response with low-detail background blending."""

import os

import cv2
import numpy as np
from core.frame_storage import (
    create_transient_frame as create_frame, create_frame as create_disk_frame,
    flush_mapped_frame, frame_region, is_mapped_frame,
)
from fusion_methods._openfocus_pixels import frame_codes, frame_gray_codes, gray_codes, output_codes
from fusion_methods._openfocus_execution import execution_job, configure_rows, run_bands

_DEFAULT_BACKGROUND_RAM_LIMIT = 32 * 1024 * 1024
_BACKGROUND_RAM_LIMIT_BYTES = _DEFAULT_BACKGROUND_RAM_LIMIT


def _down2(gray, should_cancel, check_cancel):
    """Truncated 2x2 mean with replicated odd edges."""
    height, width = gray.shape
    result = create_frame(((height + 1) // 2, (width + 1) // 2), np.int32)
    left = 2 * np.arange(result.shape[1])
    right = np.minimum(left + 1, width - 1)
    def reduce_rows(start, stop):
        check_cancel(should_cancel)
        top = (2 * np.arange(start, stop))[:, None]
        bottom = np.minimum(top + 1, height - 1)
        result[start:stop] = ((gray[top, left] + gray[top, right]
                               + gray[bottom, left] + gray[bottom, right]) >> 2)
    run_bands(result.shape[0], 128, reduce_rows,
              scratch_bytes=min(128, result.shape[0]) * result.shape[1] * 24)
    return result


def _expand_burt(gray, shape, should_cancel, check_cancel):
    """Bilinear phase (dst+0.5)/2-0.5, with edge clamping."""
    result = create_frame(shape, np.float64)
    sx = (np.arange(shape[1], dtype=np.float64) + 0.5) / 2.0 - 0.5
    x0 = np.floor(sx).astype(np.int32)
    fx = sx - x0
    x1 = np.clip(x0 + 1, 0, gray.shape[1] - 1)
    x0 = np.clip(x0, 0, gray.shape[1] - 1)
    def expand_rows(start, stop):
        check_cancel(should_cancel)
        sy = (np.arange(start, stop, dtype=np.float64) + 0.5) / 2.0 - 0.5
        y0 = np.floor(sy).astype(np.int32)
        fy = (sy - y0)[:, None]
        y1 = np.clip(y0 + 1, 0, gray.shape[0] - 1)[:, None]
        y0 = np.clip(y0, 0, gray.shape[0] - 1)[:, None]
        top = gray[y0, x0] * (1.0 - fx) + gray[y0, x1] * fx
        bottom = gray[y1, x0] * (1.0 - fx) + gray[y1, x1] * fx
        result[start:stop] = top * (1.0 - fy) + bottom * fy
    run_bands(shape[0], 128, expand_rows,
              scratch_bytes=min(128, shape[0]) * shape[1] * 64)
    return result


def _response(code, radius, should_cancel, check_cancel, *, u8=False):
    """Sum nearest-expanded |L_k| in the verified integer code domain."""
    gray = gray_codes(code, should_cancel, u8=u8)
    height, width = gray.shape
    levels = int(np.floor(np.log2(max(1, radius)))) + 3
    response = create_frame((height, width), np.float64)
    response[...] = 0
    for level in range(levels):
        check_cancel(should_cancel)
        reduced = _down2(gray, should_cancel, check_cancel)
        laplacian = _expand_burt(reduced, gray.shape, should_cancel, check_cancel)
        def difference(start, stop):
            check_cancel(should_cancel)
            rows = slice(start, stop)
            np.subtract(gray[rows], laplacian[rows], out=laplacian[rows])
            np.abs(laplacian[rows], out=laplacian[rows])
        run_bands(gray.shape[0], 128, difference)
        factor = 1 << level
        columns = np.arange(width) // factor
        def accumulate(start, stop):
            check_cancel(should_cancel)
            rows = (np.arange(start, stop) // factor)[:, None]
            response[start:stop] += laplacian[rows, columns]
        run_bands(height, 128, accumulate,
                  scratch_bytes=min(128, height) * width * 8)
        gray = reduced
        del laplacian
    def clamp(start, stop):
        check_cancel(should_cancel)
        np.minimum(response[start:stop], 65535, out=response[start:stop])
    run_bands(height, 128, clamp)
    return response


def _hard_pyramid_shapes(shape, radius):
    shapes = [shape]
    for _ in range(int(np.floor(np.log2(max(1, radius)))) + 3):
        height, width = shapes[-1]
        reduced_shape = ((height + 1) // 2, (width + 1) // 2)
        shapes.append(reduced_shape)
        if reduced_shape == (1, 1):
            break  # Further Laplacians of a single pixel are zero.
    return shapes


def _hard_pyramid(gray, radius, should_cancel, check_cancel):
    """Keep only the half-size gray plane and packed coarse Laplacians."""
    from fusion_methods._openfocus_kernels import down2_rows, burt_laplacian_rows

    shapes = _hard_pyramid_shapes(gray.shape, radius)

    lap_size = sum(h * w for h, w in shapes[1:-1])
    laplacians = (create_frame((lap_size,), np.int32) if lap_size
                  else np.empty(0, np.int32))
    offset = 0
    current = gray
    for level, shape in enumerate(shapes[1:]):
        check_cancel(should_cancel)
        reduced = create_frame(shape, np.int32)
        def reduce_rows(start, stop):
            check_cancel(should_cancel)
            down2_rows(current, reduced, start, stop)
        run_bands(shape[0], 128, reduce_rows, width=shape[1])
        if level == 0:
            half_gray = reduced
        else:
            end = offset + current.size
            laplacian = laplacians[offset:end].reshape(current.shape)
            def laplacian_rows(start, stop):
                check_cancel(should_cancel)
                burt_laplacian_rows(current, reduced, laplacian, start, stop)
            run_bands(current.shape[0], 128, laplacian_rows, width=current.shape[1])
            offset = end
        current = reduced
    return half_gray, laplacians, len(shapes) - 1


def _soft_log_cache(response, frames, img_resize, workers):
    """Reserve a RAM-only cache after allowing for the remaining soft passes."""
    from core import resource_policy as policy
    from core.multi_focus_fusion import memory_budget_bytes
    from core.source_frames import frame_resolution_bytes

    cache_bytes = len(frames) * response.nbytes
    budget = min(policy.ram_budget(policy.probe_memory()), memory_budget_bytes())
    if cache_bytes > budget // 4:
        return None

    height, width = response.shape
    pixels = height * width
    band_pixels = min(128, height) * width
    channels = max(frame.shape[2] if len(frame.shape) == 3 else 1 for frame in frames)
    frame_bytes = pixels * channels * 4
    decode_bytes = max(frame_resolution_bytes(frame, metadata_only=True) for frame in frames)
    # Codes, gray pyramid, and response coexist with the active Burt band.
    response_bytes = frame_bytes + pixels * 16 + response.nbytes + band_pixels * 64 + width * 64
    prepare_bytes = frame_bytes
    if img_resize is not None:
        source_bytes = max(frame.shape[0] * frame.shape[1] *
                           (frame.shape[2] if len(frame.shape) == 3 else 1) * 4
                           for frame in frames)
        response_bytes = max(response_bytes, 2 * (source_bytes + frame_bytes))
        prepare_bytes = source_bytes + 2 * frame_bytes
    # The log maximum already exists. The last pass adds a denominator,
    # the output, and band work while prepared frames are in flight.
    remaining_bytes = max(
        workers * (decode_bytes + response_bytes),
        workers * (decode_bytes + prepare_bytes) + response.nbytes + frame_bytes
        + band_pixels * max(64, 24 + channels * 8),
    )
    if cache_bytes + remaining_bytes > budget:
        return None

    try:
        return np.empty((len(frames), height, width), dtype=response.dtype)
    except MemoryError:
        return None


def _background_plane(shape, nbytes, frames, workers):
    """Allocate the low-response accumulators in RAM whenever they fit.

    Small planes share the transient frame cap. A larger plane is admitted
    against half the comfortable RAM budget after reserving every in-flight
    decode, because a disk-backed plane is read and rewritten for every
    source. Otherwise it is mapped to disk, as before.
    """
    from core import resource_policy as policy
    from core.frame_storage import ram_snapshot_bytes

    limit = _BACKGROUND_RAM_LIMIT_BYTES
    if limit != _DEFAULT_BACKGROUND_RAM_LIMIT:
        allocate = create_frame if nbytes <= limit else create_disk_frame
        return allocate(shape, np.float32)
    snapshot = policy.probe_memory()
    frame_budget = policy.frame_ram_budget(snapshot)
    reserve = 0
    if nbytes > _DEFAULT_BACKGROUND_RAM_LIMIT and len(frames) > 1:
        from core.source_frames import frame_resolution_bytes

        decode_peak = max(frame_resolution_bytes(frames[i]) for i in range(1, len(frames)))
        reserve = workers * decode_peak
        frame_budget = min(frame_budget, policy.ram_ceiling(snapshot) - reserve)
    if nbytes <= frame_budget - ram_snapshot_bytes():
        return create_frame(shape, np.float32)
    if nbytes + reserve <= policy.ram_budget(snapshot) // 2:
        try:
            policy.require_memory(nbytes + reserve, snapshot)
            return np.empty(shape, np.float32)
        except MemoryError:
            pass
    return create_disk_frame(shape, np.float32)


def _writeback_selection(*planes):
    if os.name != 'nt' or not any(is_mapped_frame(plane) for plane in planes):
        return
    from core import resource_policy as policy

    # A child allocation cap cannot tell us whether Windows is short on
    # reclaimable pages. Use the machine's reserve-protected headroom.
    if policy._detected_ram_ceiling(policy.probe_memory()) < 256 * 1024 * 1024:
        for plane in planes:
            flush_mapped_frame(plane)


@execution_job
def openfocus_a_impl(input_source, img_resize=None, kernel_size: int = 17,
                     thread_count: int = None, blend_smoothing: float = 2.0,
                     blend_mode: str = 'hard', soft_gamma: float = 1.0,
                     feather_theta: float = 0.9,
                     should_cancel=None, progress_callback=None,
                     use_gpu: bool = True, device_callback=None, *, live_preview=None):
    """Fuse with the Burt-pyramid focus response.

    ``kernel_size=2*R+1`` controls ``floor(log2(R))+3`` levels. Smoothing is
    intentionally inert. Clear focus keeps the highest-response source;
    low-response areas blend the stack unless all responses tie. Eight-bit
    output applies E-37; sixteen-bit codes retain their input precision.

    ``soft`` and ``feather`` are experimental, unregistered merge modes.
    Their historical blending stays separate from the default path.
    """
    from fusion_methods.openfocus_ab import (
        _check_cancel, _collect_images, _prepare_frame,
        _frame_pass_workers, _ordered_frame_results)

    frames = _collect_images(input_source, normalize=False, should_cancel=should_cancel)
    configure_rows(frames, img_resize, 'openfocus_a', thread_count, should_cancel)
    radius = max(1, int(kernel_size) // 2)
    hard = blend_mode not in ('soft', 'feather')
    if progress_callback is not None:
        callback = progress_callback
        reported = -1
        def progress_callback(index, total):
            nonlocal reported
            if index > reported:
                callback(index, total)
                reported = index
    if hard and use_gpu:
        from core.source_frames import SOURCE_CACHE
        from fusion_methods._openfocus_gpu import GpuError, log_fallback
        from fusion_methods._openfocus_gpu_a import GpuA
        gpu = None
        try:
            gpu = GpuA()
            args = (frames, img_resize, radius, should_cancel, progress_callback)
            result = (gpu.fuse(*args) if live_preview is None else
                      gpu.fuse(*args, live_preview=live_preview))
            if device_callback is not None:
                device_callback(gpu.device_name)
            return result
        except GpuError as exc:
            log_fallback(exc)
        finally:
            gpu = None
            SOURCE_CACHE.clear()
    native_hard = hard and img_resize is None
    workers = _frame_pass_workers(frames, img_resize, 'openfocus_a', thread_count)
    if progress_callback is not None:
        progress_callback(0, len(frames))
    _check_cancel(should_cancel)

    from core.source_frames import resolve_frame, without_source_cache_growth

    def produce_response(index, cancelled):
        source = frames[index]
        with without_source_cache_growth():
            code = frame_codes(source, img_resize, cancelled)
        return _response(code, radius, cancelled, _check_cancel,
                         u8=source.dtype == np.uint8)

    def produce(index, cancelled):
        source = frames[index]
        if native_hard and source.dtype in (np.uint8, np.uint16):
            _check_cancel(cancelled)
            with without_source_cache_growth():
                frame = resolve_frame(source, should_cancel=cancelled)
            _check_cancel(cancelled)
            gray = frame_gray_codes(frame, None, cancelled)
            return frame, (gray, *_hard_pyramid(gray, radius, cancelled, _check_cancel))
        with without_source_cache_growth():
            code = frame_codes(source, img_resize, cancelled)
        if hard:
            gray = gray_codes(code, cancelled, u8=source.dtype == np.uint8)
            return code, (gray, *_hard_pyramid(gray, radius, cancelled, _check_cancel))
        response = _response(code, radius, cancelled, _check_cancel,
                             u8=source.dtype == np.uint8)
        with without_source_cache_growth():
            return _prepare_frame(source, img_resize, cancelled), response

    def prepare(index, cancelled):
        with without_source_cache_growth():
            return _prepare_frame(frames[index], img_resize, cancelled)

    soft = blend_mode == 'soft'
    feather = blend_mode == 'feather'
    _soft_gamma = float(soft_gamma)
    total_steps = len(frames) * (3 if soft else 2 if feather else 1)
    step = 0

    output = best = None
    first_pass_fn = produce if hard else produce_response
    produced = _ordered_frame_results(len(frames), first_pass_fn, workers, should_cancel)
    try:
        if feather:
            # pass 1: streaming top-2 (value + index) per pixel
            i1 = i2 = None
            v1 = v2 = None
            for index, response in produced:
                _check_cancel(should_cancel)
                if i1 is None:
                    i1 = np.full(response.shape, index, np.int32)
                    v1 = response.copy()
                    i2 = i1.copy()
                    v2 = np.full_like(response, -np.inf)
                else:
                    def select_top_two(start, stop):
                        _check_cancel(should_cancel)
                        rows = slice(start, stop)
                        a, b = v1[rows], v2[rows]
                        ia, ib = i1[rows], i2[rows]
                        r = response[rows]
                        # new top-1: old top-1 becomes top-2
                        take1 = r > a
                        np.copyto(v2[rows], a, where=take1)
                        np.copyto(i2[rows], ia, where=take1)
                        np.copyto(v1[rows], r, where=take1)
                        np.copyto(i1[rows], np.full(r.shape, index, np.int32), where=take1)
                        # new top-2 only: between old top-2 and response
                        take2 = (~take1) & (r > b)
                        np.copyto(v2[rows], r, where=take2)
                        np.copyto(i2[rows], np.full(r.shape, index, np.int32), where=take2)
                    run_bands(response.shape[0], 128, select_top_two,
                              scratch_bytes=min(128, response.shape[0]) * response.shape[1] * 8)
                if live_preview is not None:
                    live_preview.show_positions(i1, index, len(frames), len(frames) - 1)
                del response
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            _check_cancel(should_cancel)
            produced.close()

            w2 = np.where(v1 > 0, v2 / np.where(v1 > 0, v1, 1.0), 0.0)
            blend2 = w2 >= float(feather_theta)

            # pass 2: weighted accumulation over top-2 picks
            accumulated = None
            produced = _ordered_frame_results(len(frames), prepare, workers, should_cancel)
            for index, frame in produced:
                _check_cancel(should_cancel)
                if accumulated is None:
                    accumulated = np.zeros(frame.shape, np.float32)
                    height, width = accumulated.shape[:2]
                def accumulate_top_two(start, stop):
                    _check_cancel(should_cancel)
                    rows = slice(start, stop)
                    is1 = i1[rows] == index
                    is2 = blend2[rows] & (i2[rows] == index)
                    if frame.ndim == 3:
                        accumulated[rows] += np.where(is1[:, :, None], 1.0 - 0.5 * blend2[rows][:, :, None], 0.0) * frame[rows]
                        accumulated[rows] += np.where(is2[:, :, None], 0.5, 0.0) * frame[rows]
                    else:
                        accumulated[rows] += np.where(is1, 1.0 - 0.5 * blend2[rows], 0.0) * frame[rows]
                        accumulated[rows] += np.where(is2, 0.5, 0.0) * frame[rows]
                run_bands(height, 128, accumulate_top_two,
                          scratch_bytes=min(128, height) * width * 64)
                del frame
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            output = accumulated
        elif soft:
            def _log_rows(response, out=None):
                if out is None:
                    out = np.empty_like(response)
                def log_rows(start, stop):
                    _check_cancel(should_cancel)
                    rows = slice(start, stop)
                    np.log(np.maximum(response[rows], 1e-12, dtype=np.float32),
                           out=out[rows])
                run_bands(response.shape[0], 128, log_rows,
                          scratch_bytes=min(128, response.shape[0]) * response.shape[1] * 4)
                return out

            # Pass 1: keep the per-pixel log maximum, and retain logs when admitted.
            log_max = None
            cached_logs = None
            for index, response in produced:
                _check_cancel(should_cancel)
                if log_max is None:
                    log_max = _log_rows(response)
                    cached_logs = _soft_log_cache(response, frames, img_resize, workers)
                    if cached_logs is not None:
                        np.copyto(cached_logs[0], log_max)
                else:
                    lr = _log_rows(response, None if cached_logs is None else cached_logs[index])
                    np.maximum(log_max, lr, out=log_max)
                    del lr
                del response
                if live_preview is not None:
                    live_preview.update_best(log_max, index, len(frames))
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            _check_cancel(should_cancel)
            produced.close()

            # pass 2: softmax denominator with the stabilized exponent
            denom = np.zeros_like(log_max)
            if cached_logs is None:
                produced = _ordered_frame_results(len(frames), produce_response, workers, should_cancel)
                responses = produced
            else:
                responses = enumerate(cached_logs)
            for _, response in responses:
                _check_cancel(should_cancel)
                lr = _log_rows(response) if cached_logs is None else response
                def denominator(start, stop):
                    _check_cancel(should_cancel)
                    rows = slice(start, stop)
                    denom[rows] += np.exp(_soft_gamma * (lr[rows] - log_max[rows]))
                run_bands(denom.shape[0], 128, denominator,
                          scratch_bytes=min(128, denom.shape[0]) * denom.shape[1] * 24)
                del response, lr
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            _check_cancel(should_cancel)
            produced.close()
            del responses

            # Pass 3: accumulate source pixels with the normalized weights.
            accumulated = None
            producer = produce if cached_logs is None else prepare
            produced = _ordered_frame_results(len(frames), producer, workers, should_cancel)
            for index, result in produced:
                _check_cancel(should_cancel)
                if cached_logs is None:
                    frame, response = result
                    lr = _log_rows(response)
                    del response
                else:
                    frame, lr = result, cached_logs[index]
                del result
                if accumulated is None:
                    accumulated = np.zeros(frame.shape, np.float32)
                    height, width = accumulated.shape[:2]
                def accumulate_soft(start, stop):
                    _check_cancel(should_cancel)
                    rows = slice(start, stop)
                    weight = np.exp(_soft_gamma * (lr[rows] - log_max[rows])) / \
                        np.where(denom[rows] > 0, denom[rows], 1.0)
                    if frame.ndim == 3:
                        accumulated[rows] += weight[:, :, None] * frame[rows]
                    else:
                        accumulated[rows] += weight * frame[rows]
                run_bands(height, 128, accumulate_soft,
                          scratch_bytes=min(128, height) * width * 64)
                del frame, lr
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            del cached_logs, log_max, denom
            output = accumulated
        else:
            from fusion_methods._openfocus_kernels import (
                A_BACKGROUND, A_SATURATED, burt_cache_bytes, select_rows,
            )

            band = 128
            for index, (frame, pyramid) in produced:
                first = index == 0
                if first:
                    height, width = frame.shape[:2]
                    output = create_frame(frame.shape, np.uint16)
                    best = create_frame((height, width), np.int32)
                    channels = 1 if frame.ndim == 2 else frame.shape[2]
                    # One region lease covers the interleaved sums, weight,
                    # and packed response statistics for each live band.
                    background_shape = (height, width, channels + 2)
                    background_bytes = height * width * (channels + 2) * 4
                    background = _background_plane(background_shape, background_bytes,
                                                   frames, workers)
                    no_background = background[:0]
                    band_state = [A_BACKGROUND] * ((height + band - 1) // band)
                    mapped_band_bytes = min(band, height) * width * (channels + 2) * 4
                    row_scratch = width * 4 + mapped_band_bytes
                if frame.shape != output.shape:
                    raise ValueError("OpenFocus A requires frames with matching shapes")
                scale = 256 if frame.dtype == np.uint8 else 1
                cache_bytes = burt_cache_bytes(pyramid[0], pyramid[-1])
                def select(start, stop, response):
                    _check_cancel(should_cancel)
                    slot = start // band
                    state = band_state[slot]
                    if state == A_SATURATED:
                        return
                    args = (frame, *pyramid, response, best, output, start, stop, scale, first)
                    if state == A_BACKGROUND:
                        with frame_region(background, slice(start, stop), write=True,
                                          writeback=True) as region:
                            band_state[slot] = select_rows(*args, region)
                    else:
                        band_state[slot] = select_rows(*args, no_background)
                run_bands(height, band, select, scratch_bytes=row_scratch + cache_bytes,
                          workspace=lambda: np.empty(width, np.int32))
                if live_preview is not None:
                    _check_cancel(should_cancel)
                    live_preview.update_best(best, index, len(frames))
                del frame, pyramid
                _writeback_selection(best, output)
                if progress_callback is not None:
                    progress_callback(index + 1, len(frames))
                _check_cancel(should_cancel)
    finally:
        produced.close()

    _check_cancel(should_cancel)
    if hard:
        from core.source_frames import SOURCE_CACHE
        from fusion_methods._openfocus_kernels import blend_low_response_rows
        def blend_background(start, stop):
            _check_cancel(should_cancel)
            if band_state[start // band] == A_BACKGROUND:
                with frame_region(background, slice(start, stop)) as region:
                    blend_low_response_rows(output, best, region, start, stop)
        if A_BACKGROUND in band_state:
            run_bands(height, band, blend_background, scratch_bytes=mapped_band_bytes)
            _writeback_selection(output)
        del best, background, no_background
        SOURCE_CACHE.clear()
        result = output_codes(output, frames, should_cancel, e37=True)
        if device_callback is not None:
            device_callback('CPU')
        return result
    np.clip(output, 0, 255, out=output)
    if all(frame.dtype == np.uint8 for frame in frames):
        result = create_frame(output.shape, np.uint8)
        def convert(start, stop):
            _check_cancel(should_cancel)
            rows = slice(start, stop)
            result[rows] = output[rows]
        run_bands(output.shape[0], 128, convert)
        if device_callback is not None:
            device_callback('CPU')
        return result
    if device_callback is not None:
        device_callback('CPU')
    return output
