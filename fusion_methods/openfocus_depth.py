"""Streaming depth estimation returning a continuous frame index."""
import numpy as np


def _weight(radius, allow_zero=False):
    r = max(float(radius), 1e-6)
    w = int(np.float32(np.power(np.float32(0.045999825),
                                np.float32(1.5 / r))) * 4096) & ~1
    return max(0 if allow_zero else 2, min(w, 4096))


def _axis(src, guide, main, low, count, axis, cancel):
    if axis:
        src = src.T
        if guide is not None:
            guide = guide.T
    h, w = src.shape
    count = max(1, min(int(count), h))
    first = np.empty((h, w), np.uint32)
    out = np.empty_like(first)
    if guide is None:
        q = _weight(main)
        t = (4096 - q) >> 1
        acc = ((((src[:count].sum(axis=0, dtype=np.uint64) << 4) // count)
                * (t + q)) & 0xffffffff).astype(np.uint32)
        for y in range(h):
            if y % 64 == 0:
                cancel()
            innovation = (src[y] << 4) * t
            a = (acc + 2048 + innovation) >> 12
            first[y] = ((a + 4) >> 4) & 65535
            acc = a * q + innovation
        acc = (first[-1] << 4) * (t + q)
        for y in range(h - 1, -1, -1):
            if y % 64 == 0:
                cancel()
            innovation = (src[y] << 4) * t
            a = (acc + 2048 + innovation) >> 12
            out[y] = ((((a + 3) >> 4) + 1 + first[y]) >> 1) & 65535
            acc = a * q + innovation
    else:
        tm = (4096 - _weight(main, True)) >> 1
        tl = (4096 - _weight(low, True)) >> 1
        acc = (((src[:count].sum(axis=0, dtype=np.uint64) << 4) // count)
               & 0xffffffff).astype(np.uint32)
        for y in range(h):
            if y % 64 == 0:
                cancel()
            t = ((65536 - guide[y]) * tl + guide[y] * tm) >> 16
            prev = src[max(0, y - 1)]
            acc = ((4096 - 2 * t) * acc + 2048 +
                   (src[y] + prev) * (t << 4)) >> 12
            first[y] = (acc + 4) >> 4
        acc = first[-1] << 4
        prev = first[-1]
        for y in range(h - 1, -1, -1):
            if y % 64 == 0:
                cancel()
            t = ((65536 - guide[y]) * tl + guide[y] * tm) >> 16
            acc = ((4096 - 2 * t) * acc + 2048 +
                   (src[y] + prev) * (t << 4)) >> 12
            out[y] = ((((acc + 3) >> 4) + 1 + first[y]) >> 1) & 65535
            prev = src[y]
    return out.T if axis else out


def _filter(src, radius, cancel, guide=None, low=None):
    low = radius if low is None else low
    # Arithmetic in each recurrence wraps at 32 bits. Narrowing the work
    # arrays preserves that wrap while halving their memory bandwidth.
    result = np.asarray(src, dtype=np.uint32)
    guide = None if guide is None else np.asarray(guide, dtype=np.uint32)
    for axis in (0, 1):
        result = _axis(result, guide, radius, low, radius, axis, cancel)
    return result.astype(np.int64)


def depth_map(frames, img_resize, kernel_size, smoothing, should_cancel,
              progress_callback=None):
    """Return a continuous index map, using only one frame's activity at a time."""
    from fusion_methods.openfocus_ab import _check_cancel, _prepare_frame

    def cancel():
        _check_cancel(should_cancel)

    best = winner = None
    for index, source in enumerate(frames):
        if progress_callback is not None:
            cancel()
            try:
                progress_callback(index, len(frames))
            except Exception:
                pass
        frame = _prepare_frame(source, img_resize, should_cancel)
        if frame.ndim == 3:
            gray = frame[:, :, :min(frame.shape[2], 3)].mean(axis=2)
        else:
            gray = frame
        if source.dtype == np.uint8:
            gray = np.floor(gray) * 256.0
        else:
            gray = gray * (65535.0 / 255.0)
        gray = np.clip(gray, 0, 65535).astype(np.int64)
        # The kernel is a diameter. Scale the two filter radii by 0.0375R
        # and 0.15R percent of the longer image dimension.
        radius = max(1, int(kernel_size or 11) // 2)
        extent = max(gray.shape)
        fine = _filter(gray, extent * radius * 0.0375 / 100.0, cancel)
        diff = np.minimum(65535, 110 * np.abs(gray - fine) //
                          (np.maximum(gray, fine) * 100 // 65535 + 10))
        activity = _filter(diff, extent * radius * 0.15 / 100.0, cancel)
        if best is None:
            best = activity
            winner = np.zeros(gray.shape, np.int32)
        else:
            selected = activity > best
            best[selected] = activity[selected]
            winner[selected] = index
    cancel()
    sigma = max(0.0, float(smoothing))
    if sigma == 0:
        return winner.astype(np.float32)
    if len(frames) > 65536:
        raise ValueError("Smoothed depth maps support at most 65536 source frames")
    # Match the percentile guide's LUT interpolation and 16-bit truncation.
    hist = np.bincount(best.ravel(), minlength=65536)
    cumulative = np.cumsum(hist)
    threshold = int(np.float32(np.float32(best.size) * np.float32(0.01)))
    lo = int(np.searchsorted(cumulative, threshold)) if threshold else 0
    hi = (int(np.searchsorted(cumulative, best.size - threshold, side="right"))
          if threshold else 65535)
    gain = np.float32(65535.0) / np.float32(max(hi - lo, 1))
    lut_values = np.arange(65536, dtype=np.float64) / 65535.0
    guide_lut = np.float32(np.clip((lut_values - lo / 65535.0) * gain, 0, 1) * 65536.0)
    np.minimum(guide_lut, np.float32(65535.0), out=guide_lut)
    lut_pos = best.astype(np.float32) * (np.float32(65534.0) / np.float32(65535.0))
    lut_index = np.clip(lut_pos.astype(np.int32), 0, 65533)
    fraction = lut_pos - lut_index
    guide = (guide_lut[lut_index] * (np.float32(1) - fraction)
             + guide_lut[lut_index + 1] * fraction).astype(np.int64) & 65535
    # Spread all indices across the full 16-bit range, including the last one.
    count = len(frames) - 1
    depth = (winner.astype(np.int64) * 65535 + count // 2) // count
    scale = max(best.shape) * sigma * 0.0234375 / 100.0
    for _ in range(2):
        depth = _filter(depth, scale, cancel, guide, scale * 64)
    # Snap near each representable frame index without assuming an integral step.
    half = 65535 // (2 * count)
    if half:
        nearest = np.clip((depth * count + 32767) // 65535, 0, count)
        snapped = (nearest * 65535 + count // 2) // count
        distance = depth - snapped
        indices = np.minimum((np.abs(distance) << 10) // half, 1024)
        values = np.arange(1025, dtype=np.float32) / np.float32(1024)
        lut = (np.power(values, np.float32(13.0 / 3.0)) * 1024).astype(np.int64)
        correction = half * lut[indices] * np.where(distance > 0, 1, -1)
        depth = np.clip(snapped + ((correction + (correction & 1023)) >> 10), 0, 65535)
    return depth.astype(np.float32) * (count / 65535.0)
