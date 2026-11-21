"""CPU Numba kernels for StackMFF RGB TIFF packing, luma, resizing, and color reconstruction."""

import numpy as np

from fusion_methods._depth_kernel import njit


@njit(nogil=True)
def pack_rgb_u16_to_bgr_f32(src_rgb, y0, x0, h, w, dst_bgr, gain):
    """Copy a positive-stride uint16 RGB window into contiguous float32 BGR (0-255)."""
    scale = np.float32(255.0 / 65535.0)
    g32 = np.float32(gain)
    has_gain = gain != 1.0
    zero = np.float32(0.0)
    top = np.float32(255.0)
    for dy in range(h):
        sy = y0 + dy
        for dx in range(w):
            sx = x0 + dx
            b = np.float32(src_rgb[sy, sx, 2]) * scale
            g = np.float32(src_rgb[sy, sx, 1]) * scale
            r = np.float32(src_rgb[sy, sx, 0]) * scale
            if has_gain:
                b = min(top, max(zero, np.float32(b * g32)))
                g = min(top, max(zero, np.float32(g * g32)))
                r = min(top, max(zero, np.float32(r * g32)))
            dst_bgr[dy, dx, 0] = b
            dst_bgr[dy, dx, 1] = g
            dst_bgr[dy, dx, 2] = r


@njit(nogil=True)
def pack_rgb_u16_to_bgr_f32_indexed(src_rgb, rows, cols, dst_bgr, gain):
    """Gather an indexed uint16 RGB padded window into contiguous float32 BGR (0-255)."""
    scale = np.float32(255.0 / 65535.0)
    g32 = np.float32(gain)
    has_gain = gain != 1.0
    zero = np.float32(0.0)
    top = np.float32(255.0)
    h = rows.shape[0]
    w = cols.shape[0]
    for dy in range(h):
        sy = rows[dy]
        for dx in range(w):
            sx = cols[dx]
            b = np.float32(src_rgb[sy, sx, 2]) * scale
            g = np.float32(src_rgb[sy, sx, 1]) * scale
            r = np.float32(src_rgb[sy, sx, 0]) * scale
            if has_gain:
                b = min(top, max(zero, np.float32(b * g32)))
                g = min(top, max(zero, np.float32(g * g32)))
                r = min(top, max(zero, np.float32(r * g32)))
            dst_bgr[dy, dx, 0] = b
            dst_bgr[dy, dx, 1] = g
            dst_bgr[dy, dx, 2] = r


@njit(nogil=True)
def pack_rgb_u8_to_bgr_f32(src_rgb, y0, x0, h, w, dst_bgr, gain):
    """Copy a uint8 RGB window into contiguous float32 BGR (0-255)."""
    g32 = np.float32(gain)
    has_gain = gain != 1.0
    zero = np.float32(0.0)
    top = np.float32(255.0)
    for dy in range(h):
        sy = y0 + dy
        for dx in range(w):
            sx = x0 + dx
            b = np.float32(src_rgb[sy, sx, 2])
            g = np.float32(src_rgb[sy, sx, 1])
            r = np.float32(src_rgb[sy, sx, 0])
            if has_gain:
                b = min(top, max(zero, np.float32(b * g32)))
                g = min(top, max(zero, np.float32(g * g32)))
                r = min(top, max(zero, np.float32(r * g32)))
            dst_bgr[dy, dx, 0] = b
            dst_bgr[dy, dx, 1] = g
            dst_bgr[dy, dx, 2] = r


@njit(nogil=True)
def pack_rgb_u8_to_bgr_f32_indexed(src_rgb, rows, cols, dst_bgr, gain):
    """Gather an indexed uint8 RGB padded window into contiguous float32 BGR (0-255)."""
    g32 = np.float32(gain)
    has_gain = gain != 1.0
    zero = np.float32(0.0)
    top = np.float32(255.0)
    h = rows.shape[0]
    w = cols.shape[0]
    for dy in range(h):
        sy = rows[dy]
        for dx in range(w):
            sx = cols[dx]
            b = np.float32(src_rgb[sy, sx, 2])
            g = np.float32(src_rgb[sy, sx, 1])
            r = np.float32(src_rgb[sy, sx, 0])
            if has_gain:
                b = min(top, max(zero, np.float32(b * g32)))
                g = min(top, max(zero, np.float32(g * g32)))
                r = min(top, max(zero, np.float32(r * g32)))
            dst_bgr[dy, dx, 0] = b
            dst_bgr[dy, dx, 1] = g
            dst_bgr[dy, dx, 2] = r


@njit(nogil=True)
def pack_rgb_u8_to_bgr_u8(src_rgb, y0, x0, h, w, dst_bgr):
    """Copy a uint8 RGB window into contiguous uint8 BGR."""
    for dy in range(h):
        sy = y0 + dy
        for dx in range(w):
            sx = x0 + dx
            dst_bgr[dy, dx, 0] = src_rgb[sy, sx, 2]
            dst_bgr[dy, dx, 1] = src_rgb[sy, sx, 1]
            dst_bgr[dy, dx, 2] = src_rgb[sy, sx, 0]


@njit(nogil=True)
def pack_rgb_u8_to_bgr_u8_indexed(src_rgb, rows, cols, dst_bgr):
    """Gather an indexed uint8 RGB padded window into contiguous uint8 BGR."""
    h = rows.shape[0]
    w = cols.shape[0]
    for dy in range(h):
        sy = rows[dy]
        for dx in range(w):
            sx = cols[dx]
            dst_bgr[dy, dx, 0] = src_rgb[sy, sx, 2]
            dst_bgr[dy, dx, 1] = src_rgb[sy, sx, 1]
            dst_bgr[dy, dx, 2] = src_rgb[sy, sx, 0]


@njit(nogil=True)
def rgb_u8_to_luma(src_rgb, y0, x0, h, w, dst_luma):
    """Write BT.601 normalized [0-1] luma directly from a uint8 RGB window."""
    inv255 = np.float32(255.0)
    for dy in range(h):
        sy = y0 + dy
        for dx in range(w):
            sx = x0 + dx
            gray = (int(src_rgb[sy, sx, 0]) * 9798
                    + int(src_rgb[sy, sx, 1]) * 19235
                    + int(src_rgb[sy, sx, 2]) * 3735 + 16384) >> 15
            dst_luma[dy, dx] = np.float32(gray) / inv255


@njit(nogil=True)
def rgb_u8_to_luma_indexed(src_rgb, rows, cols, dst_luma):
    """Write BT.601 normalized [0-1] luma from an indexed uint8 RGB padded window."""
    inv255 = np.float32(255.0)
    h = rows.shape[0]
    w = cols.shape[0]
    for dy in range(h):
        sy = rows[dy]
        for dx in range(w):
            sx = cols[dx]
            gray = (int(src_rgb[sy, sx, 0]) * 9798
                    + int(src_rgb[sy, sx, 1]) * 19235
                    + int(src_rgb[sy, sx, 2]) * 3735 + 16384) >> 15
            dst_luma[dy, dx] = np.float32(gray) / inv255


def precompute_antialias_taps(old, scale):
    """Precompute separable triangle filter indices and normalized weights."""
    new = int(old * scale)
    if new < 1:
        raise ValueError('Input is too small for the inference scale')
    radius = max(1.0, 1.0 / scale)
    max_taps = int(np.ceil(2.0 * radius)) + 3
    first_arr = np.empty(new, dtype=np.int32)
    cnt_arr = np.empty(new, dtype=np.int32)
    weights_arr = np.zeros((new, max_taps), dtype=np.float32)
    for i in range(new):
        center = (i + 0.5) / scale - 0.5
        first = max(0, int(np.floor(center - radius)) + 1)
        last = min(old, int(np.ceil(center + radius)))
        cnt = last - first
        first_arr[i] = first
        cnt_arr[i] = cnt
        w = np.maximum(0.0, 1.0 - np.abs(np.arange(first, last) - center) / radius).astype(np.float32)
        w /= w.sum()
        weights_arr[i, :cnt] = w
    return first_arr, cnt_arr, weights_arr


@njit(nogil=True)
def resize_antialias_range(planes, b, n_start, n_stop,
                           vy_first, vy_cnt, vy_w,
                           hx_first, hx_cnt, hx_w, out):
    """L1-resident separable triangle downsample for layer range [n_start, n_stop)."""
    _, _, _, width = planes.shape
    nh = vy_first.shape[0]
    nw = hx_first.shape[0]
    tmp = np.empty((nh, width), dtype=np.float32)
    zero = np.float32(0.0)
    for n in range(n_start, n_stop):
        plane = planes[b, n]
        for i in range(nh):
            f = vy_first[i]
            c = vy_cnt[i]
            if c == 3:
                w0 = vy_w[i, 0]
                w1 = vy_w[i, 1]
                w2 = vy_w[i, 2]
                for x in range(width):
                    s = np.float32(plane[f, x] * w0)
                    s = np.float32(s + np.float32(plane[f + 1, x] * w1))
                    tmp[i, x] = np.float32(s + np.float32(plane[f + 2, x] * w2))
            elif c == 2:
                w0 = vy_w[i, 0]
                w1 = vy_w[i, 1]
                for x in range(width):
                    tmp[i, x] = np.float32(np.float32(plane[f, x] * w0)
                                           + np.float32(plane[f + 1, x] * w1))
            else:
                for x in range(width):
                    acc = zero
                    for k in range(c):
                        acc = np.float32(acc + np.float32(plane[f + k, x] * vy_w[i, k]))
                    tmp[i, x] = acc
        dst = out[b, n]
        for y in range(nh):
            row = tmp[y]
            for j in range(nw):
                f = hx_first[j]
                c = hx_cnt[j]
                if c == 3:
                    s = np.float32(row[f] * hx_w[j, 0])
                    s = np.float32(s + np.float32(row[f + 1] * hx_w[j, 1]))
                    dst[y, j] = np.float32(s + np.float32(row[f + 2] * hx_w[j, 2]))
                elif c == 2:
                    dst[y, j] = np.float32(np.float32(row[f] * hx_w[j, 0])
                                           + np.float32(row[f + 1] * hx_w[j, 1]))
                else:
                    acc = zero
                    for k in range(c):
                        acc = np.float32(acc + np.float32(row[f + k] * hx_w[j, k]))
                    dst[y, j] = acc


@njit(nogil=True)
def v1_update_nearest_f32(src_bgr, target, best, cb, cr, is_first):
    """Update V1 nearest-luma Cb/Cr planes from float32 BGR(A) frame."""
    h, w = target.shape
    cR = np.float32(0.299)
    cG = np.float32(0.587)
    cB = np.float32(0.114)
    for y in range(h):
        for x in range(w):
            b = src_bgr[y, x, 0]
            g = src_bgr[y, x, 1]
            r = src_bgr[y, x, 2]
            luma = np.float32(np.float32(np.float32(r * cR) + np.float32(g * cG))
                              + np.float32(b * cB))
            diff = abs(np.float32(luma - target[y, x]))
            prev = best[y, x]
            better = is_first or (diff < prev) or (np.isnan(diff) and not np.isnan(prev))
            if better:
                best[y, x] = diff
                cb[y, x] = np.float32(b - luma)
                cr[y, x] = np.float32(r - luma)


@njit(nogil=True)
def v1_update_nearest_u16(src_bgr, target, best, cb, cr, is_first):
    """Update V1 nearest-luma Cb/Cr planes from uint16 BGR(A) normalized to 0-255."""
    h, w = target.shape
    scale = np.float32(255.0 / 65535.0)
    cR = np.float32(0.299)
    cG = np.float32(0.587)
    cB = np.float32(0.114)
    for y in range(h):
        for x in range(w):
            b = np.float32(src_bgr[y, x, 0]) * scale
            g = np.float32(src_bgr[y, x, 1]) * scale
            r = np.float32(src_bgr[y, x, 2]) * scale
            luma = np.float32(np.float32(np.float32(r * cR) + np.float32(g * cG))
                              + np.float32(b * cB))
            diff = abs(np.float32(luma - target[y, x]))
            prev = best[y, x]
            better = is_first or (diff < prev) or (np.isnan(diff) and not np.isnan(prev))
            if better:
                best[y, x] = diff
                cb[y, x] = np.float32(b - luma)
                cr[y, x] = np.float32(r - luma)


@njit(nogil=True)
def v1_update_nearest_u8(src_bgr, target, best, cb, cr, is_first):
    """Update V1 nearest-luma Cb/Cr planes from uint8 BGR(A)."""
    h, w = target.shape
    cR = np.float32(0.299)
    cG = np.float32(0.587)
    cB = np.float32(0.114)
    for y in range(h):
        for x in range(w):
            b = np.float32(src_bgr[y, x, 0])
            g = np.float32(src_bgr[y, x, 1])
            r = np.float32(src_bgr[y, x, 2])
            luma = np.float32(np.float32(np.float32(r * cR) + np.float32(g * cG))
                              + np.float32(b * cB))
            diff = abs(np.float32(luma - target[y, x]))
            prev = best[y, x]
            better = is_first or (diff < prev) or (np.isnan(diff) and not np.isnan(prev))
            if better:
                best[y, x] = diff
                cb[y, x] = np.float32(b - luma)
                cr[y, x] = np.float32(r - luma)


@njit(nogil=True)
def v2_weighted_sum_layers(probs_flat, p_start, p_stop, out_flat):
    """Compute sum_{n=0..N-1} probs_flat[n, i] * (n + 1) across [p_start, p_stop)."""
    n_layers = probs_flat.shape[0]
    for i in range(p_start, p_stop):
        acc = np.float32(0.0)
        for n in range(n_layers):
            acc = np.float32(acc + np.float32(probs_flat[n, i] * np.float32(n + 1)))
        out_flat[i] = acc


@njit(nogil=True)
def v2_scatter_strip_rgb_u16_to_f32(src_rgb, idx_strip, layer, out_strip, gain, clamp_max):
    """Scatter selected pixels from a uint16 RGB strip directly into float32 BGR out_strip."""
    h, w = idx_strip.shape
    scale = np.float32(255.0 / 65535.0)
    g32 = np.float32(gain)
    has_gain = gain != 1.0
    zero = np.float32(0.0)
    top = np.float32(clamp_max)
    for y in range(h):
        for x in range(w):
            if idx_strip[y, x] == layer:
                b = np.float32(src_rgb[y, x, 2]) * scale
                g = np.float32(src_rgb[y, x, 1]) * scale
                r = np.float32(src_rgb[y, x, 0]) * scale
                if has_gain:
                    b = min(top, max(zero, np.float32(b * g32)))
                    g = min(top, max(zero, np.float32(g * g32)))
                    r = min(top, max(zero, np.float32(r * g32)))
                out_strip[y, x, 0] = b
                out_strip[y, x, 1] = g
                out_strip[y, x, 2] = r


@njit(nogil=True)
def v2_scatter_strip_rgb_u8(src_rgb, idx_strip, layer, out_strip):
    """Scatter selected pixels from a uint8 RGB strip directly into uint8 BGR out_strip."""
    h, w = idx_strip.shape
    for y in range(h):
        for x in range(w):
            if idx_strip[y, x] == layer:
                out_strip[y, x, 0] = src_rgb[y, x, 2]
                out_strip[y, x, 1] = src_rgb[y, x, 1]
                out_strip[y, x, 2] = src_rgb[y, x, 0]


@njit(nogil=True)
def v3_accumulate_soft_f32(src, prob, out, expected, index_f, has_expected):
    """Accumulate probability-weighted float32 BGR/gray row band into out."""
    h, w, ch = out.shape
    idx32 = np.float32(index_f)
    for y in range(h):
        for x in range(w):
            p = prob[y, x]
            for c in range(ch):
                out[y, x, c] = np.float32(out[y, x, c] + np.float32(p * src[y, x, c]))
            if has_expected:
                expected[y, x] = np.float32(expected[y, x] + np.float32(p * idx32))


@njit(nogil=True)
def v3_accumulate_soft_u16(src, prob, out, expected, index_f, has_expected):
    """Accumulate probability-weighted uint16 (normalized to 0-255) BGR/gray row band."""
    h, w, ch = out.shape
    scale = np.float32(255.0 / 65535.0)
    idx32 = np.float32(index_f)
    for y in range(h):
        for x in range(w):
            p = prob[y, x]
            for c in range(ch):
                s = np.float32(src[y, x, c]) * scale
                out[y, x, c] = np.float32(out[y, x, c] + np.float32(p * s))
            if has_expected:
                expected[y, x] = np.float32(expected[y, x] + np.float32(p * idx32))


@njit(nogil=True)
def v3_accumulate_soft_u8(src, prob, out, expected, index_f, has_expected):
    """Accumulate probability-weighted uint8 BGR/gray row band into float32 out."""
    h, w, ch = out.shape
    idx32 = np.float32(index_f)
    for y in range(h):
        for x in range(w):
            p = prob[y, x]
            for c in range(ch):
                out[y, x, c] = np.float32(out[y, x, c] + np.float32(p * np.float32(src[y, x, c])))
            if has_expected:
                expected[y, x] = np.float32(expected[y, x] + np.float32(p * idx32))


@njit(nogil=True)
def v3_scatter_hard_f32(src, selected, layer, out):
    """Copy float32 pixels where selected[y, x] == layer into out."""
    h, w, ch = out.shape
    for y in range(h):
        for x in range(w):
            if selected[y, x] == layer:
                for c in range(ch):
                    out[y, x, c] = src[y, x, c]


@njit(nogil=True)
def v3_scatter_hard_u16(src, selected, layer, out):
    """Copy uint16 pixels (normalized to 0-255) where selected[y, x] == layer into out."""
    h, w, ch = out.shape
    scale = np.float32(255.0 / 65535.0)
    for y in range(h):
        for x in range(w):
            if selected[y, x] == layer:
                for c in range(ch):
                    out[y, x, c] = np.float32(src[y, x, c]) * scale


@njit(nogil=True)
def v3_scatter_hard_u8(src, selected, layer, out):
    """Copy uint8 pixels as float32 where selected[y, x] == layer into out."""
    h, w, ch = out.shape
    for y in range(h):
        for x in range(w):
            if selected[y, x] == layer:
                for c in range(ch):
                    out[y, x, c] = np.float32(src[y, x, c])
