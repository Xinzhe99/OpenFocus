"""HF9 method A: Burt pyramid response and ordered hard source selection."""

import cv2
import numpy as np
from core.frame_storage import create_transient_frame as create_frame
from fusion_methods._hf9_pixels import frame_codes, frame_gray_codes, gray_codes, output_codes
from fusion_methods._hf9_execution import execution_job, configure_rows, run_bands


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
    """Prototype bilinear phase (dst+0.5)/2-0.5, with edge clamping."""
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


def _hard_response(gray, radius, should_cancel, check_cancel):
    """Exact Burt response in Q4; dyadic interpolation needs no float plane."""
    from fusion_methods._hf9_kernels import (
        down2_rows, burt_laplacian_rows, burt_accumulate_rows,
    )

    height, width = gray.shape
    levels = int(np.floor(np.log2(max(1, radius)))) + 3
    response = create_frame((height, width), np.int32)
    response[...] = 0
    for level in range(levels):
        check_cancel(should_cancel)
        reduced = create_frame(((gray.shape[0] + 1) // 2,
                                (gray.shape[1] + 1) // 2), np.int32)
        def reduce_rows(start, stop):
            check_cancel(should_cancel)
            down2_rows(gray, reduced, start, stop)
        run_bands(reduced.shape[0], 128, reduce_rows)
        laplacian = create_frame(gray.shape, np.int32)
        def laplacian_rows(start, stop):
            check_cancel(should_cancel)
            burt_laplacian_rows(gray, reduced, laplacian, start, stop)
        run_bands(gray.shape[0], 128, laplacian_rows)
        def accumulate_rows(start, stop):
            check_cancel(should_cancel)
            burt_accumulate_rows(laplacian, response, level, start, stop)
        run_bands(height, 128, accumulate_rows)
        gray = reduced
        del laplacian
    return response


@execution_job
def openfocus_a_impl(input_source, img_resize=None, kernel_size: int = 17,
                     thread_count: int = None, blend_smoothing: float = 2.0,
                     blend_mode: str = 'hard', soft_gamma: float = 1.0,
                     feather_theta: float = 0.9,
                     should_cancel=None, progress_callback=None):
    """Select source pixels with the HF9 Burt-pyramid response.

    ``kernel_size=2*R+1`` controls ``floor(log2(R))+3`` levels. Smoothing is
    intentionally inert, as in HF9 method A. Strict source-order argmax keeps
    the first frame on ties. Eight-bit output applies E-37; sixteen-bit codes
    retain all bits through the application's float32 output boundary.

    ``soft`` and ``feather`` are experimental, unregistered merge modes.
    Their historical blending stays separate from the default hard path.
    """
    from fusion_methods.openfocus_ab import (
        _check_cancel, _collect_images, _prepare_frame,
        _frame_pass_workers, _ordered_frame_results)

    frames = _collect_images(input_source, normalize=False, should_cancel=should_cancel)
    configure_rows(frames, img_resize, 'openfocus_a', thread_count, should_cancel)
    radius = max(1, int(kernel_size) // 2)
    hard = blend_mode not in ('soft', 'feather')
    fast_hard = hard and img_resize is None and all(
        source.dtype in (np.uint8, np.uint16) for source in frames)
    workers = _frame_pass_workers(frames, img_resize, 'openfocus_a', thread_count)
    if progress_callback is not None:
        progress_callback(0, len(frames))
    _check_cancel(should_cancel)

    def produce(index, cancelled):
        source = frames[index]
        if fast_hard:
            from core.source_frames import resolve_frame

            _check_cancel(cancelled)
            frame = resolve_frame(source)
            _check_cancel(cancelled)
            gray = frame_gray_codes(frame, None, cancelled)
            response = _hard_response(gray, radius, cancelled, _check_cancel)
            return frame, response
        code = frame_codes(source, img_resize, cancelled)
        if hard:
            gray = gray_codes(code, cancelled, u8=source.dtype == np.uint8)
            response = _hard_response(gray, radius, cancelled, _check_cancel)
        else:
            response = _response(code, radius, cancelled, _check_cancel,
                                 u8=source.dtype == np.uint8)
        frame = code if hard else _prepare_frame(source, img_resize, cancelled)
        return frame, response

    soft = blend_mode == 'soft'
    feather = blend_mode == 'feather'
    _soft_gamma = float(soft_gamma)
    total_steps = len(frames) * (3 if soft else 2 if feather else 1)
    step = 0

    output = best = None
    produced = _ordered_frame_results(len(frames), produce, workers, should_cancel)
    try:
        if feather:
            # pass 1: streaming top-2 (value + index) per pixel
            i1 = i2 = None
            v1 = v2 = None
            for index, (frame, response) in produced:
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
                del frame, response
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            _check_cancel(should_cancel)
            produced.close()

            w2 = np.where(v1 > 0, v2 / np.where(v1 > 0, v1, 1.0), 0.0)
            blend2 = w2 >= float(feather_theta)

            # pass 2: weighted accumulation over top-2 picks
            accumulated = None
            produced = _ordered_frame_results(len(frames), produce, workers, should_cancel)
            for index, (frame, response) in produced:
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
                del frame, response
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            output = accumulated
        elif soft:
            def _log_rows(response):
                """In-place log of a copy, rows blocked."""
                out = np.empty_like(response)
                def log_rows(start, stop):
                    _check_cancel(should_cancel)
                    rows = slice(start, stop)
                    np.log(np.maximum(response[rows], 1e-12, dtype=np.float32),
                           out=out[rows])
                run_bands(response.shape[0], 128, log_rows,
                          scratch_bytes=min(128, response.shape[0]) * response.shape[1] * 4)
                return out

            # pass 1: per-pixel max of log response
            log_max = None
            for index, (frame, response) in produced:
                _check_cancel(should_cancel)
                lr = _log_rows(response)
                m = lr.max(axis=0)
                log_max = m if log_max is None else np.maximum(log_max, m)
                del frame, response, lr, m
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            _check_cancel(should_cancel)
            produced.close()

            # pass 2: softmax denominator with the stabilized exponent
            denom = None
            produced = _ordered_frame_results(len(frames), produce, workers, should_cancel)
            for index, (frame, response) in produced:
                _check_cancel(should_cancel)
                lr = _log_rows(response)
                if denom is None:
                    denom = np.zeros_like(lr)
                def denominator(start, stop):
                    _check_cancel(should_cancel)
                    rows = slice(start, stop)
                    denom[rows] += np.exp(_soft_gamma * (lr[rows] - log_max[rows, None]))
                run_bands(denom.shape[0], 128, denominator,
                          scratch_bytes=min(128, denom.shape[0]) * denom.shape[1] * 24)
                del frame, response, lr
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            _check_cancel(should_cancel)
            produced.close()

            # pass 3: weighted accumulation
            accumulated = None
            produced = _ordered_frame_results(len(frames), produce, workers, should_cancel)
            for index, (frame, response) in produced:
                _check_cancel(should_cancel)
                if accumulated is None:
                    accumulated = np.zeros(frame.shape, np.float32)
                    height, width = accumulated.shape[:2]
                lr = _log_rows(response)
                def accumulate_soft(start, stop):
                    _check_cancel(should_cancel)
                    rows = slice(start, stop)
                    weight = np.exp(_soft_gamma * (lr[rows] - log_max[rows, None])) / \
                        np.where(denom[rows] > 0, denom[rows], 1.0)
                    if frame.ndim == 3:
                        accumulated[rows] += weight[:, :, None] * frame[rows]
                    else:
                        accumulated[rows] += weight * frame[rows]
                run_bands(height, 128, accumulate_soft,
                          scratch_bytes=min(128, height) * width * 64)
                del frame, response, lr
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            output = accumulated
        else:
            for index, (frame, response) in produced:
                if index == 0:
                    best = response
                    if fast_hard:
                        output = create_frame(frame.shape, np.uint16)
                    else:
                        output = frame
                        del frame, response
                        if progress_callback is not None:
                            progress_callback(index + 1, len(frames))
                        _check_cancel(should_cancel)
                        continue
                if frame.shape != output.shape:
                    raise ValueError("OpenFocus A requires frames with matching shapes")
                def select(start, stop):
                    _check_cancel(should_cancel)
                    if fast_hard:
                        from fusion_methods._hf9_kernels import select_rows

                        select_rows(frame, response, best, output, start, stop,
                                    256 if frame.dtype == np.uint8 else 1, index == 0)
                    else:
                        rows = slice(start, stop)
                        better = response[rows] > best[rows]
                        np.copyto(output[rows], frame[rows],
                                  where=better[:, :, None] if frame.ndim == 3 else better)
                        np.copyto(best[rows], response[rows], where=better)
                        del better
                run_bands(output.shape[0], 128, select,
                          scratch_bytes=min(128, output.shape[0]) * output.shape[1])
                del frame, response
                if progress_callback is not None:
                    progress_callback(index + 1, len(frames))
                _check_cancel(should_cancel)
    finally:
        produced.close()

    _check_cancel(should_cancel)
    if hard:
        from core.source_frames import SOURCE_CACHE
        del best
        SOURCE_CACHE.clear()
        return output_codes(output, frames, should_cancel, e37=True)
    np.clip(output, 0, 255, out=output)
    if all(frame.dtype == np.uint8 for frame in frames):
        result = create_frame(output.shape, np.uint8)
        def convert(start, stop):
            _check_cancel(should_cancel)
            rows = slice(start, stop)
            result[rows] = output[rows]
        run_bands(output.shape[0], 128, convert)
        return result
    return output
