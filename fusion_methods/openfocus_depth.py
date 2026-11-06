"""Streaming depth estimation returning a continuous frame index."""
import numpy as np
from core.frame_storage import create_frame


def _weight(radius, allow_zero=False):
    r = max(float(radius), 1e-6)
    w = int(np.float32(np.power(np.float32(0.045999825),
                                np.float32(1.5 / r))) * 4096) & ~1
    return max(0 if allow_zero else 2, min(w, 4096))


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
    first = allocate((h, w), np.uint32)
    out = allocate((h, w), np.uint32)
    first_pixels, out_pixels = np.asarray(first), np.asarray(out)
    q = t = tm = tl = 0
    if guide is None:
        q = _weight(main)
        t = (4096 - q) >> 1
        acc = ((((src[:count].sum(axis=0, dtype=np.uint64) << 4) // count)
                * (t + q)) & 0xffffffff).astype(np.uint32)
    else:
        tm = (4096 - _weight(main, True)) >> 1
        tl = (4096 - _weight(low, True)) >> 1
        acc = (((src[:count].sum(axis=0, dtype=np.uint64) << 4) // count)
               & 0xffffffff).astype(np.uint32)
    for start in range(0, h, 64):
        cancel()
        axis_block(src, guide, first_pixels, out_pixels, acc, q, t, tm, tl,
                   start, min(h, start + 64), False)
    acc = first_pixels[-1] << 4
    if guide is None:
        acc *= t + q
    for stop in range(h, 0, -64):
        cancel()
        axis_block(src, guide, first_pixels, out_pixels, acc, q, t, tm, tl,
                   max(0, stop - 64), stop, True)
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
    # Arithmetic in each recurrence wraps at 32 bits. Narrowing the work
    # arrays preserves that wrap while halving their memory bandwidth.
    result = _mapped_cast(src, np.uint32, cancel, allocate)
    guide = None if guide is None else _mapped_cast(guide, np.uint32, cancel, allocate)
    for axis in (0, 1):
        result = _axis(result, guide, radius, low, radius, axis, cancel, allocate)
    return _mapped_cast(result, np.int64, cancel, allocate)


def depth_map(frames, img_resize, kernel_size, smoothing, should_cancel,
              progress_callback=None, thread_count=1):
    """Estimate independent activities in a bounded window; reduce globally."""
    from fusion_methods.openfocus_ab import (
        _check_cancel, _prepare_frame, _frame_pass_workers, _ordered_frame_results)
    from fusion_methods._depth_kernel import prepare_depth_kernel

    prepare_depth_kernel(guided=float(smoothing) > 0,
                         should_cancel=should_cancel)

    def cancel():
        _check_cancel(should_cancel)

    # Mixed stacks must retain the same fractional channel means and scaling
    # as already-normalized/tiled inputs, independently of producer ordering.
    normalize_16bit = any(source.dtype == np.uint16 for source in frames)
    workers = _frame_pass_workers(frames, img_resize, 'openfocus_b', thread_count,
                                  activity_only=True)

    def produce(index, cancelled):
        def cancel():
            _check_cancel(cancelled)

        # Multi-producer admission includes complete resident envelopes. Keep
        # their private planes in RAM; serial low-RAM work keeps the mappings.
        allocate = np.empty if workers > 1 else create_frame

        source = frames[index]
        frame = _prepare_frame(source, img_resize, cancelled)
        gray = allocate(frame.shape[:2], np.int64)
        for rows in _rows(gray, cancel):
            pixels = frame[rows]
            value = (pixels[:, :, :min(pixels.shape[2], 3)].mean(axis=2)
                     if pixels.ndim == 3 else pixels)
            if source.dtype == np.uint8 and not normalize_16bit:
                value = np.floor(value) * 256.0
            else:
                value = value * (65535.0 / 255.0)
            gray[rows] = np.clip(value, 0, 65535).astype(np.int64)
        del frame, pixels, value
        radius = max(1, int(kernel_size or 11) // 2)
        extent = max(gray.shape)
        fine = _filter(gray, extent * radius * 0.0375 / 100.0, cancel,
                       allocate=allocate)
        diff = allocate(gray.shape, np.int64)
        for rows in _rows(gray, cancel):
            g, f = gray[rows], fine[rows]
            diff[rows] = np.minimum(65535, 110 * np.abs(g - f) //
                                   (np.maximum(g, f) * 100 // 65535 + 10))
        del gray, fine, g, f
        return _filter(diff, extent * radius * 0.15 / 100.0, cancel,
                       allocate=allocate)

    best = winner = None
    produced = _ordered_frame_results(len(frames), produce, workers, should_cancel)
    try:
        for index, activity in produced:
            if progress_callback is not None:
                cancel()
                try:
                    progress_callback(index, len(frames))
                except Exception:
                    pass
            if best is None:
                best = activity
                winner = create_frame(activity.shape, np.int32)
                winner[...] = 0
            else:
                for rows in _rows(activity, cancel):
                    selected = activity[rows] > best[rows]
                    np.copyto(best[rows], activity[rows], where=selected)
                    np.copyto(winner[rows], index, where=selected)
                del selected
            del activity
    finally:
        produced.close()
    cancel()
    sigma = max(0.0, float(smoothing))
    if sigma == 0:
        result = create_frame(winner.shape, np.float32)
        for rows in _rows(winner, cancel):
            result[rows] = winner[rows]
        return result
    if len(frames) > 65536:
        raise ValueError("Smoothed depth maps support at most 65536 source frames")
    # Match the percentile guide's LUT interpolation and 16-bit truncation.
    hist = np.zeros(65536, np.int64)
    for rows in _rows(best, cancel):
        hist += np.bincount(best[rows].ravel(), minlength=65536)
    cumulative = np.cumsum(hist)
    threshold = int(np.float32(np.float32(best.size) * np.float32(0.01)))
    lo = int(np.searchsorted(cumulative, threshold)) if threshold else 0
    hi = (int(np.searchsorted(cumulative, best.size - threshold, side="right"))
          if threshold else 65535)
    gain = np.float32(65535.0) / np.float32(max(hi - lo, 1))
    lut_values = np.arange(65536, dtype=np.float64) / 65535.0
    guide_lut = np.float32(np.clip((lut_values - lo / 65535.0) * gain, 0, 1) * 65536.0)
    np.minimum(guide_lut, np.float32(65535.0), out=guide_lut)
    guide = create_frame(best.shape, np.int64)
    for rows in _rows(best, cancel):
        lut_pos = best[rows].astype(np.float32) * (np.float32(65534.0) / np.float32(65535.0))
        lut_index = np.clip(lut_pos.astype(np.int32), 0, 65533)
        fraction = lut_pos - lut_index
        guide[rows] = ((guide_lut[lut_index] * (np.float32(1) - fraction)
                        + guide_lut[lut_index + 1] * fraction).astype(np.int64) & 65535)
    del lut_pos, lut_index, fraction, best
    # Spread all indices across the full 16-bit range, including the last one.
    count = len(frames) - 1
    depth = create_frame(winner.shape, np.int64)
    for rows in _rows(winner, cancel):
        depth[rows] = (winner[rows].astype(np.int64) * 65535 + count // 2) // count
    del winner
    scale = max(depth.shape) * sigma * 0.0234375 / 100.0
    for _ in range(2):
        depth = _filter(depth, scale, cancel, guide, scale * 64)
    # Snap near each representable frame index without assuming an integral step.
    half = 65535 // (2 * count)
    if half:
        values = np.arange(1025, dtype=np.float32) / np.float32(1024)
        lut = (np.power(values, np.float32(13.0 / 3.0)) * 1024).astype(np.int64)
        for rows in _rows(depth, cancel):
            block = depth[rows]
            nearest = np.clip((block * count + 32767) // 65535, 0, count)
            snapped = (nearest * 65535 + count // 2) // count
            distance = block - snapped
            indices = np.minimum((np.abs(distance) << 10) // half, 1024)
            correction = half * lut[indices] * np.where(distance > 0, 1, -1)
            depth[rows] = np.clip(snapped + ((correction + (correction & 1023)) >> 10), 0, 65535)
    result = create_frame(depth.shape, np.float32)
    for start in range(0, depth.shape[0], 128):
        cancel()
        rows = slice(start, min(depth.shape[0], start + 128))
        result[rows] = depth[rows].astype(np.float32) * (count / 65535.0)
    return result
