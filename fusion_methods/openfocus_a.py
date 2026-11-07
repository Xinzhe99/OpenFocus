"""High-frequency energy selection for OpenFocus A."""

import cv2
import numpy as np
from core.frame_storage import create_frame, store_frame


def _response(frame, fine_sigma, spread_sigma, should_cancel, check_cancel):
    """Measure high-frequency energy with globally aligned, mapped row passes."""
    height, width = frame.shape[:2]
    gray = create_frame((height, width), np.float32)
    for start in range(0, height, 128):
        check_cancel(should_cancel)
        rows = slice(start, min(height, start + 128))
        pixels = frame[rows]
        gray[rows] = (pixels if frame.ndim == 2 else pixels[:, :, 0]
                      if frame.shape[2] == 1 else
                      np.mean(pixels[:, :, :3], axis=2, dtype=np.float32))
    energy = create_frame((height, width), np.float32)
    halo = int(np.ceil(4 * fine_sigma)) + 2
    for start in range(0, height, 128):
        check_cancel(should_cancel)
        stop = min(height, start + 128)
        top, bottom = max(0, start - halo), min(height, stop + halo)
        blurred = cv2.GaussianBlur(gray[top:bottom], (0, 0), sigmaX=fine_sigma,
                                    borderType=cv2.BORDER_REPLICATE)
        rows = slice(start, stop)
        np.subtract(gray[rows], blurred[start - top:stop - top], out=energy[rows])
        np.square(energy[rows], out=energy[rows])
    del gray, pixels, blurred
    response = create_frame((height, width), np.float32)
    halo = int(np.ceil(4 * spread_sigma)) + 2
    for start in range(0, height, 128):
        check_cancel(should_cancel)
        stop = min(height, start + 128)
        top, bottom = max(0, start - halo), min(height, stop + halo)
        blurred = cv2.GaussianBlur(energy[top:bottom], (0, 0), sigmaX=spread_sigma,
                                    borderType=cv2.BORDER_REPLICATE)
        response[start:stop] = blurred[start - top:stop - top]
    return response


def openfocus_a_impl(input_source, img_resize=None, kernel_size: int = 17,
                     thread_count: int = None, blend_smoothing: float = 2.0,
                     blend_mode: str = 'hard', soft_gamma: float = 1.0,
                     feather_theta: float = 0.9,
                     should_cancel=None, progress_callback=None):
    """Select original pixels using local high-frequency energy.

    ``kernel_size`` controls the fine Gaussian scale (kernel/5.5, minimum 1).
    ``blend_smoothing`` controls the energy spread (smoothing/2, minimum 1).
    ``blend_mode`` selects the fusion rule: ``hard`` (default) keeps the
    argmax frame per pixel; ``soft`` blends all frames with log-domain
    power-softmax weights ``exp(gamma*(log r - max log r))`` in a
    deterministic three-pass streaming pipeline with per-pixel (H×W)
    auxiliary memory. ``soft`` triples response computation and is an
    experimental, unregistered path (Issue #21: content-dependent quality —
    not shipped); ``soft_gamma`` controls hardness (higher = closer to hard).
    ``feather`` keeps hard selection everywhere except pixels where the
    second-best response is within ``feather_theta`` of the best (r2/r1 >=
    theta), which are blended 50/50 between the top-2 frames — the depth
    transitions where halos form. Two-pass streaming, H×W auxiliary memory.
    ``thread_count`` bounds independent response producers. Selection remains
    strict and in source order, so ties keep the first frame. RAM admission
    bounds both live producers and completed results; low RAM stays serial.

    Frames use the application's float32 0-255 domain. Float results retain
    the precision needed by 16-bit export.
    """
    from fusion_methods.openfocus_ab import (
        _check_cancel, _collect_images, _prepare_frame,
        _frame_pass_workers, _ordered_frame_results)

    frames = _collect_images(input_source, normalize=False, should_cancel=should_cancel)
    fine_sigma = max(1.0, float(kernel_size) / 5.5)
    spread_sigma = max(1.0, float(
        2.0 if blend_smoothing is None else blend_smoothing) / 2.0)
    workers = _frame_pass_workers(frames, img_resize, 'openfocus_a', thread_count)
    if progress_callback is not None:
        progress_callback(0, len(frames))
    _check_cancel(should_cancel)

    def produce(index, cancelled):
        frame = _prepare_frame(frames[index], img_resize, cancelled)
        response = _response(frame, fine_sigma, spread_sigma,
                             cancelled, _check_cancel)
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
                    for start in range(0, response.shape[0], 128):
                        _check_cancel(should_cancel)
                        rows = slice(start, min(response.shape[0], start + 128))
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
                for start in range(0, height, 128):
                    _check_cancel(should_cancel)
                    rows = slice(start, min(height, start + 128))
                    is1 = i1[rows] == index
                    is2 = blend2[rows] & (i2[rows] == index)
                    if frame.ndim == 3:
                        accumulated[rows] += np.where(is1[:, :, None], 1.0 - 0.5 * blend2[rows][:, :, None], 0.0) * frame[rows]
                        accumulated[rows] += np.where(is2[:, :, None], 0.5, 0.0) * frame[rows]
                    else:
                        accumulated[rows] += np.where(is1, 1.0 - 0.5 * blend2[rows], 0.0) * frame[rows]
                        accumulated[rows] += np.where(is2, 0.5, 0.0) * frame[rows]
                del frame, response
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            output = accumulated
        elif soft:
            def _log_rows(response):
                """In-place log of a copy, rows blocked."""
                out = np.empty_like(response)
                for start in range(0, response.shape[0], 128):
                    _check_cancel(should_cancel)
                    rows = slice(start, min(response.shape[0], start + 128))
                    np.log(np.maximum(response[rows], 1e-12, dtype=np.float32),
                           out=out[rows])
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
                for start in range(0, denom.shape[0], 128):
                    _check_cancel(should_cancel)
                    rows = slice(start, min(denom.shape[0], start + 128))
                    denom[rows] += np.exp(_soft_gamma * (lr[rows] - log_max[rows, None]))
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
                for start in range(0, height, 128):
                    _check_cancel(should_cancel)
                    rows = slice(start, min(height, start + 128))
                    weight = np.exp(_soft_gamma * (lr[rows] - log_max[rows, None])) / \
                        np.where(denom[rows] > 0, denom[rows], 1.0)
                    if frame.ndim == 3:
                        accumulated[rows] += weight[:, :, None] * frame[rows]
                    else:
                        accumulated[rows] += weight * frame[rows]
                del frame, response, lr, weight
                step += 1
                if progress_callback is not None:
                    progress_callback(step, total_steps)
            output = accumulated
        else:
            for index, (frame, response) in produced:
                if index == 0:
                    output = store_frame(frame)
                    best = response
                    del frame, response
                    if progress_callback is not None:
                        progress_callback(index + 1, len(frames))
                    _check_cancel(should_cancel)
                    continue
                if frame.shape != output.shape:
                    raise ValueError("OpenFocus A requires frames with matching shapes")
                for start in range(0, output.shape[0], 128):
                    _check_cancel(should_cancel)
                    rows = slice(start, min(output.shape[0], start + 128))
                    better = response[rows] > best[rows]
                    np.copyto(output[rows], frame[rows],
                              where=better[:, :, None] if frame.ndim == 3 else better)
                    np.copyto(best[rows], response[rows], where=better)
                del frame, response, better
                if progress_callback is not None:
                    progress_callback(index + 1, len(frames))
                _check_cancel(should_cancel)
    finally:
        produced.close()

    _check_cancel(should_cancel)
    np.clip(output, 0, 255, out=output)
    if all(frame.dtype == np.uint8 for frame in frames):
        result = create_frame(output.shape, np.uint8)
        for start in range(0, output.shape[0], 128):
            _check_cancel(should_cancel)
            rows = slice(start, min(output.shape[0], start + 128))
            result[rows] = output[rows]
        return result
    return output
