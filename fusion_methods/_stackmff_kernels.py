"""CPU Numba kernels for StackMFF RGB TIFF packing, luma, resizing, and color reconstruction."""

import numpy as np
from numba.core.compiler_lock import global_compiler_lock

from core.algorithm_adapters import DEPTH_KERNEL_STARTUP_BYTES
from core.resource_policy import probe_memory, require_memory
from fusion_methods._depth_kernel import njit
from fusion_methods.neural_common import (
    StackMFFKernelUnavailable, disable_stackmff_kernels, load_stackmff_kernels,
)


def _optional_kernel(function):
    dispatcher = njit(nogil=True)(function)
    compile_signature = dispatcher.compile

    def compile_with_admission(signature):
        # Lazy overload misses include disk cache loads. Serialize the memory
        # check with compilation so workers can reuse an overload just built.
        with global_compiler_lock:
            if signature in dispatcher.overloads:
                return compile_signature(signature)
            if load_stackmff_kernels() is None:
                raise StackMFFKernelUnavailable()
            try:
                require_memory(DEPTH_KERNEL_STARTUP_BYTES, probe_memory())
                return compile_signature(signature)
            except MemoryError as exc:
                disable_stackmff_kernels()
                raise StackMFFKernelUnavailable() from exc

    dispatcher.compile = compile_with_admission
    return dispatcher


@_optional_kernel
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


@_optional_kernel
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


@_optional_kernel
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


@_optional_kernel
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


@_optional_kernel
def pack_rgb_u8_to_bgr_u8(src_rgb, y0, x0, h, w, dst_bgr):
    """Copy a uint8 RGB window into contiguous uint8 BGR."""
    for dy in range(h):
        sy = y0 + dy
        for dx in range(w):
            sx = x0 + dx
            dst_bgr[dy, dx, 0] = src_rgb[sy, sx, 2]
            dst_bgr[dy, dx, 1] = src_rgb[sy, sx, 1]
            dst_bgr[dy, dx, 2] = src_rgb[sy, sx, 0]


@_optional_kernel
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


@_optional_kernel
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


@_optional_kernel
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


@_optional_kernel
def resize_antialias_range(planes, b, n_start, n_stop,
                           vy_first, vy_cnt, vy_w,
                           hx_first, hx_cnt, hx_w, out, row_start, row_stop, row):
    """Resize a band of output rows for each layer using one vertical scratch row."""
    width = planes.shape[-1]
    nw = hx_first.shape[0]
    zero = np.float32(0.0)
    for n in range(n_start, n_stop):
        plane = planes[b, n]
        dst = out[b, n]
        for y in range(row_start, row_stop):
            f = vy_first[y]
            c = vy_cnt[y]
            if c == 3:
                w0 = vy_w[y, 0]
                w1 = vy_w[y, 1]
                w2 = vy_w[y, 2]
                for x in range(width):
                    s = np.float32(plane[f, x] * w0)
                    s = np.float32(s + np.float32(plane[f + 1, x] * w1))
                    row[x] = np.float32(s + np.float32(plane[f + 2, x] * w2))
            elif c == 2:
                w0 = vy_w[y, 0]
                w1 = vy_w[y, 1]
                for x in range(width):
                    row[x] = np.float32(np.float32(plane[f, x] * w0)
                                        + np.float32(plane[f + 1, x] * w1))
            else:
                for x in range(width):
                    acc = zero
                    for k in range(c):
                        acc = np.float32(acc + np.float32(plane[f + k, x] * vy_w[y, k]))
                    row[x] = acc
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


@_optional_kernel
def v1_update_nearest_f32_rows(src_bgr, target, best, cb, cr, is_first, y_start, y_stop):
    """Update V1 nearest-luma Cb/Cr planes for [y_start, y_stop) from float32 BGR(A)."""
    w = target.shape[1]
    cR = np.float32(0.299)
    cG = np.float32(0.587)
    cB = np.float32(0.114)
    for y in range(y_start, y_stop):
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


@_optional_kernel
def v1_update_nearest_u16_rows(src_bgr, target, best, cb, cr, is_first, y_start, y_stop):
    """Update V1 nearest-luma Cb/Cr planes for [y_start, y_stop) from uint16 BGR(A)."""
    w = target.shape[1]
    scale = np.float32(255.0 / 65535.0)
    cR = np.float32(0.299)
    cG = np.float32(0.587)
    cB = np.float32(0.114)
    for y in range(y_start, y_stop):
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


@_optional_kernel
def v1_update_nearest_u8_rows(src_bgr, target, best, cb, cr, is_first, y_start, y_stop):
    """Update V1 nearest-luma Cb/Cr planes for [y_start, y_stop) from uint8 BGR(A)."""
    w = target.shape[1]
    cR = np.float32(0.299)
    cG = np.float32(0.587)
    cB = np.float32(0.114)
    for y in range(y_start, y_stop):
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


@_optional_kernel
def v1_update_nearest_rgb_u16_rows(src_rgb, target, best, cb, cr, is_first, y_start, y_stop):
    """Update V1 nearest-luma Cb/Cr planes for [y_start, y_stop) from uint16 RGB."""
    w = target.shape[1]
    scale = np.float32(255.0 / 65535.0)
    cR = np.float32(0.299)
    cG = np.float32(0.587)
    cB = np.float32(0.114)
    for y in range(y_start, y_stop):
        for x in range(w):
            b = np.float32(src_rgb[y, x, 2]) * scale
            g = np.float32(src_rgb[y, x, 1]) * scale
            r = np.float32(src_rgb[y, x, 0]) * scale
            luma = np.float32(np.float32(np.float32(r * cR) + np.float32(g * cG))
                              + np.float32(b * cB))
            diff = abs(np.float32(luma - target[y, x]))
            prev = best[y, x]
            better = is_first or (diff < prev) or (np.isnan(diff) and not np.isnan(prev))
            if better:
                best[y, x] = diff
                cb[y, x] = np.float32(b - luma)
                cr[y, x] = np.float32(r - luma)


@_optional_kernel
def v1_update_nearest_rgb_u8_rows(src_rgb, target, best, cb, cr, is_first, y_start, y_stop):
    """Update V1 nearest-luma Cb/Cr planes for [y_start, y_stop) from uint8 RGB."""
    w = target.shape[1]
    cR = np.float32(0.299)
    cG = np.float32(0.587)
    cB = np.float32(0.114)
    for y in range(y_start, y_stop):
        for x in range(w):
            b = np.float32(src_rgb[y, x, 2])
            g = np.float32(src_rgb[y, x, 1])
            r = np.float32(src_rgb[y, x, 0])
            luma = np.float32(np.float32(np.float32(r * cR) + np.float32(g * cG))
                              + np.float32(b * cB))
            diff = abs(np.float32(luma - target[y, x]))
            prev = best[y, x]
            better = is_first or (diff < prev) or (np.isnan(diff) and not np.isnan(prev))
            if better:
                best[y, x] = diff
                cb[y, x] = np.float32(b - luma)
                cr[y, x] = np.float32(r - luma)


def v1_update_nearest_f32(src_bgr, target, best, cb, cr, is_first):
    """Update V1 nearest-luma Cb/Cr planes from float32 BGR(A) frame."""
    v1_update_nearest_f32_rows(src_bgr, target, best, cb, cr, is_first, 0, target.shape[0])


def v1_update_nearest_u16(src_bgr, target, best, cb, cr, is_first):
    """Update V1 nearest-luma Cb/Cr planes from uint16 BGR(A) normalized to 0-255."""
    v1_update_nearest_u16_rows(src_bgr, target, best, cb, cr, is_first, 0, target.shape[0])


def v1_update_nearest_u8(src_bgr, target, best, cb, cr, is_first):
    """Update V1 nearest-luma Cb/Cr planes from uint8 BGR(A)."""
    v1_update_nearest_u8_rows(src_bgr, target, best, cb, cr, is_first, 0, target.shape[0])


@_optional_kernel
def v1_weighted_luma_rows(probs, planes, out_luma, y_start, y_stop):
    """Fused weighted luma reduction for StackMFF V1 across [y_start, y_stop)."""
    n_layers = probs.shape[1]
    w = out_luma.shape[1]
    zero = np.float32(0.0)
    one = np.float32(1.0)
    top = np.float32(255.0)
    for y in range(y_start, y_stop):
        if n_layers > 0:
            for x in range(w):
                out_luma[y, x] = np.float32(probs[0, 0, y, x] * planes[0, 0, y, x])
            for n in range(1, n_layers):
                for x in range(w):
                    out_luma[y, x] = np.float32(
                        out_luma[y, x] + np.float32(probs[0, n, y, x] * planes[0, n, y, x])
                    )
            for x in range(w):
                val = out_luma[y, x]
                if val == val:
                    val = min(one, max(zero, val))
                out_luma[y, x] = np.float32(val * top)
        else:
            for x in range(w):
                out_luma[y, x] = zero


@_optional_kernel
def v1_ycbcr_to_bgr_rows(luma, cb, cr, out_bgr, y_start, y_stop):
    """Convert V1 fused luma and selected chroma deltas into clipped float32 BGR."""
    w = luma.shape[1]
    c_cb_scale = np.float32(0.564)
    c_cr_scale = np.float32(0.713)
    c_128 = np.float32(128.0)
    c_b = np.float32(1.773)
    c_g_cr = np.float32(0.714)
    c_g_cb = np.float32(0.344)
    c_r = np.float32(1.403)
    zero = np.float32(0.0)
    top = np.float32(255.0)
    for y in range(y_start, y_stop):
        for x in range(w):
            y_val = luma[y, x]
            cb_v = np.float32(np.float32(np.float32(cb[y, x] * c_cb_scale) + c_128) - c_128)
            cr_v = np.float32(np.float32(np.float32(cr[y, x] * c_cr_scale) + c_128) - c_128)
            bv = np.float32(y_val + np.float32(c_b * cb_v))
            gv = np.float32(np.float32(y_val - np.float32(c_g_cr * cr_v)) - np.float32(c_g_cb * cb_v))
            rv = np.float32(y_val + np.float32(c_r * cr_v))
            out_bgr[y, x, 0] = min(top, max(zero, bv)) if bv == bv else bv
            out_bgr[y, x, 1] = min(top, max(zero, gv)) if gv == gv else gv
            out_bgr[y, x, 2] = min(top, max(zero, rv)) if rv == rv else rv


@_optional_kernel
def normalize_probs_rows(probs, y_start, y_stop):
    """Normalize (B, N, H, W) probabilities over layers for rows [y_start, y_stop)."""
    batches = probs.shape[0]
    n_layers = probs.shape[1]
    w = probs.shape[3]
    row_sum = np.empty(w, dtype=np.float32)
    for b in range(batches):
        for y in range(y_start, y_stop):
            for x in range(w):
                row_sum[x] = probs[b, 0, y, x]
            for n in range(1, n_layers):
                for x in range(w):
                    row_sum[x] = np.float32(row_sum[x] + probs[b, n, y, x])
            for n in range(n_layers):
                for x in range(w):
                    probs[b, n, y, x] = np.float32(probs[b, n, y, x] / row_sum[x])


@_optional_kernel
def bgr_u8_to_luma_rows(src_bgr, dst_luma, y_start, y_stop):
    """Write BT.601 normalized [0-1] luma for rows [y_start, y_stop) from uint8 BGR."""
    w = src_bgr.shape[1]
    inv255 = np.float32(255.0)
    for y in range(y_start, y_stop):
        for x in range(w):
            gray = (int(src_bgr[y, x, 2]) * 9798
                    + int(src_bgr[y, x, 1]) * 19235
                    + int(src_bgr[y, x, 0]) * 3735 + 16384) >> 15
            dst_luma[y, x] = np.float32(gray) / inv255


@_optional_kernel
def rgb_u8_to_luma_rows(src_rgb, dst_luma, y_start, y_stop):
    """Write BT.601 normalized [0-1] luma for rows [y_start, y_stop) from uint8 RGB."""
    w = src_rgb.shape[1]
    inv255 = np.float32(255.0)
    for y in range(y_start, y_stop):
        for x in range(w):
            gray = (int(src_rgb[y, x, 0]) * 9798
                    + int(src_rgb[y, x, 1]) * 19235
                    + int(src_rgb[y, x, 2]) * 3735 + 16384) >> 15
            dst_luma[y, x] = np.float32(gray) / inv255


@_optional_kernel
def pack_bgr_u16_to_bgr_f32_rows(src_bgr, y_start, y_stop, dst_bgr):
    """Normalize uint16 BGR rows [y_start, y_stop) into float32 BGR [0-255]."""
    w = src_bgr.shape[1]
    scale = np.float32(255.0 / 65535.0)
    for y in range(y_start, y_stop):
        dy = y - y_start
        for x in range(w):
            dst_bgr[dy, x, 0] = np.float32(src_bgr[y, x, 0]) * scale
            dst_bgr[dy, x, 1] = np.float32(src_bgr[y, x, 1]) * scale
            dst_bgr[dy, x, 2] = np.float32(src_bgr[y, x, 2]) * scale


@_optional_kernel
def pack_bgr_u8_to_bgr_f32_rows(src_bgr, y_start, y_stop, dst_bgr):
    """Convert uint8 BGR rows [y_start, y_stop) into float32 BGR [0-255]."""
    w = src_bgr.shape[1]
    for y in range(y_start, y_stop):
        dy = y - y_start
        for x in range(w):
            dst_bgr[dy, x, 0] = np.float32(src_bgr[y, x, 0])
            dst_bgr[dy, x, 1] = np.float32(src_bgr[y, x, 1])
            dst_bgr[dy, x, 2] = np.float32(src_bgr[y, x, 2])


@_optional_kernel
def pack_rgb_u16_to_bgr_f32_rows(src_rgb, y_start, y_stop, dst_bgr):
    """Normalize uint16 RGB rows [y_start, y_stop) into float32 BGR [0-255]."""
    w = src_rgb.shape[1]
    scale = np.float32(255.0 / 65535.0)
    for y in range(y_start, y_stop):
        dy = y - y_start
        for x in range(w):
            dst_bgr[dy, x, 0] = np.float32(src_rgb[y, x, 2]) * scale
            dst_bgr[dy, x, 1] = np.float32(src_rgb[y, x, 1]) * scale
            dst_bgr[dy, x, 2] = np.float32(src_rgb[y, x, 0]) * scale


@_optional_kernel
def pack_rgb_u8_to_bgr_f32_rows(src_rgb, y_start, y_stop, dst_bgr):
    """Convert uint8 RGB rows [y_start, y_stop) into float32 BGR [0-255]."""
    w = src_rgb.shape[1]
    for y in range(y_start, y_stop):
        dy = y - y_start
        for x in range(w):
            dst_bgr[dy, x, 0] = np.float32(src_rgb[y, x, 2])
            dst_bgr[dy, x, 1] = np.float32(src_rgb[y, x, 1])
            dst_bgr[dy, x, 2] = np.float32(src_rgb[y, x, 0])


@_optional_kernel
def v2_weighted_sum_layers(probs_flat, p_start, p_stop, out_flat):
    """Compute sum_{n=0..N-1} probs_flat[n, i] * (n + 1) across [p_start, p_stop)."""
    n_layers = probs_flat.shape[0]
    block = 1024
    for b0 in range(p_start, p_stop, block):
        b1 = min(p_stop, b0 + block)
        if n_layers == 0:
            for i in range(b0, b1):
                out_flat[i] = np.float32(0.0)
            continue
        w0 = np.float32(1.0)
        for i in range(b0, b1):
            out_flat[i] = np.float32(probs_flat[0, i] * w0)
        for n in range(1, n_layers):
            wn = np.float32(n + 1)
            for i in range(b0, b1):
                out_flat[i] = np.float32(out_flat[i] + np.float32(probs_flat[n, i] * wn))


@_optional_kernel
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


@_optional_kernel
def v2_scatter_strip_rgb_u8_to_f32(src_rgb, idx_strip, layer, out_strip, gain, clamp_max):
    """Scatter selected pixels from a uint8 RGB strip directly into float32 BGR out_strip."""
    h, w = idx_strip.shape
    g32 = np.float32(gain)
    has_gain = gain != 1.0
    zero = np.float32(0.0)
    top = np.float32(clamp_max)
    for y in range(h):
        for x in range(w):
            if idx_strip[y, x] == layer:
                b = np.float32(src_rgb[y, x, 2])
                g = np.float32(src_rgb[y, x, 1])
                r = np.float32(src_rgb[y, x, 0])
                if has_gain:
                    b = min(top, max(zero, np.float32(b * g32)))
                    g = min(top, max(zero, np.float32(g * g32)))
                    r = min(top, max(zero, np.float32(r * g32)))
                out_strip[y, x, 0] = b
                out_strip[y, x, 1] = g
                out_strip[y, x, 2] = r


@_optional_kernel
def v2_scatter_strip_rgb_u8(src_rgb, idx_strip, layer, out_strip):
    """Scatter selected pixels from a uint8 RGB strip directly into uint8 BGR out_strip."""
    h, w = idx_strip.shape
    for y in range(h):
        for x in range(w):
            if idx_strip[y, x] == layer:
                out_strip[y, x, 0] = src_rgb[y, x, 2]
                out_strip[y, x, 1] = src_rgb[y, x, 1]
                out_strip[y, x, 2] = src_rgb[y, x, 0]


@_optional_kernel
def v2_scatter_strip_bgr_u16_to_f32(src_bgr, idx_strip, layer, out_strip, gain, clamp_max):
    """Scatter selected pixels from a uint16 BGR strip into float32 BGR out_strip."""
    h, w = idx_strip.shape
    scale = np.float32(255.0 / 65535.0)
    g32 = np.float32(gain)
    has_gain = gain != 1.0
    zero = np.float32(0.0)
    top = np.float32(clamp_max)
    for y in range(h):
        for x in range(w):
            if idx_strip[y, x] == layer:
                b = np.float32(src_bgr[y, x, 0]) * scale
                g = np.float32(src_bgr[y, x, 1]) * scale
                r = np.float32(src_bgr[y, x, 2]) * scale
                if has_gain:
                    b = min(top, max(zero, np.float32(b * g32)))
                    g = min(top, max(zero, np.float32(g * g32)))
                    r = min(top, max(zero, np.float32(r * g32)))
                out_strip[y, x, 0] = b
                out_strip[y, x, 1] = g
                out_strip[y, x, 2] = r


@_optional_kernel
def v2_scatter_strip_bgr_u8_to_f32(src_bgr, idx_strip, layer, out_strip, gain, clamp_max):
    """Scatter selected pixels from a uint8 BGR strip into float32 BGR out_strip."""
    h, w = idx_strip.shape
    g32 = np.float32(gain)
    has_gain = gain != 1.0
    zero = np.float32(0.0)
    top = np.float32(clamp_max)
    for y in range(h):
        for x in range(w):
            if idx_strip[y, x] == layer:
                b = np.float32(src_bgr[y, x, 0])
                g = np.float32(src_bgr[y, x, 1])
                r = np.float32(src_bgr[y, x, 2])
                if has_gain:
                    b = min(top, max(zero, np.float32(b * g32)))
                    g = min(top, max(zero, np.float32(g * g32)))
                    r = min(top, max(zero, np.float32(r * g32)))
                out_strip[y, x, 0] = b
                out_strip[y, x, 1] = g
                out_strip[y, x, 2] = r


@_optional_kernel
def v2_scatter_strip_bgr_u8(src_bgr, idx_strip, layer, out_strip):
    """Scatter selected pixels from a uint8 BGR strip into uint8 BGR out_strip."""
    h, w = idx_strip.shape
    for y in range(h):
        for x in range(w):
            if idx_strip[y, x] == layer:
                out_strip[y, x, 0] = src_bgr[y, x, 0]
                out_strip[y, x, 1] = src_bgr[y, x, 1]
                out_strip[y, x, 2] = src_bgr[y, x, 2]


@_optional_kernel
def v2_scatter_strip_bgr_f32(src_bgr, idx_strip, layer, out_strip, gain, clamp_max):
    """Scatter selected pixels from a float32 BGR strip into float32 BGR out_strip."""
    h, w = idx_strip.shape
    g32 = np.float32(gain)
    has_gain = gain != 1.0
    zero = np.float32(0.0)
    top = np.float32(clamp_max)
    for y in range(h):
        for x in range(w):
            if idx_strip[y, x] == layer:
                b = src_bgr[y, x, 0]
                g = src_bgr[y, x, 1]
                r = src_bgr[y, x, 2]
                if has_gain:
                    b = min(top, max(zero, np.float32(b * g32))) if b == b else b
                    g = min(top, max(zero, np.float32(g * g32))) if g == g else g
                    r = min(top, max(zero, np.float32(r * g32))) if r == r else r
                out_strip[y, x, 0] = b
                out_strip[y, x, 1] = g
                out_strip[y, x, 2] = r


@_optional_kernel
def v3_argmax_rows(probs, selected, counts, y_start, y_stop):
    """Compute argmax and layer histogram over [y_start, y_stop) with SIMD row access."""
    n_layers = probs.shape[0]
    w = probs.shape[2]
    best_val = np.empty(w, dtype=np.float32)
    for y in range(y_start, y_stop):
        for x in range(w):
            best_val[x] = probs[0, y, x]
            selected[y, x] = 0
        for n in range(1, n_layers):
            for x in range(w):
                bv = best_val[x]
                if bv == bv:
                    v = probs[n, y, x]
                    if not (v <= bv):
                        best_val[x] = v
                        selected[y, x] = n
        for x in range(w):
            counts[selected[y, x]] += 1


@_optional_kernel
def v3_accumulate_soft_f32(src, prob, out, expected, index_f, has_expected):
    """Accumulate probability-weighted float32 BGR/gray row band into out."""
    h, w, ch = out.shape
    idx32 = np.float32(index_f)
    if ch == 3:
        for y in range(h):
            for x in range(w):
                p = prob[y, x]
                out[y, x, 0] = np.float32(out[y, x, 0] + np.float32(p * src[y, x, 0]))
                out[y, x, 1] = np.float32(out[y, x, 1] + np.float32(p * src[y, x, 1]))
                out[y, x, 2] = np.float32(out[y, x, 2] + np.float32(p * src[y, x, 2]))
                if has_expected:
                    expected[y, x] = np.float32(expected[y, x] + np.float32(p * idx32))
    else:
        for y in range(h):
            for x in range(w):
                p = prob[y, x]
                for c in range(ch):
                    out[y, x, c] = np.float32(out[y, x, c] + np.float32(p * src[y, x, c]))
                if has_expected:
                    expected[y, x] = np.float32(expected[y, x] + np.float32(p * idx32))


@_optional_kernel
def v3_accumulate_soft_u16(src, prob, out, expected, index_f, has_expected):
    """Accumulate probability-weighted uint16 (normalized to 0-255) BGR/gray row band."""
    h, w, ch = out.shape
    scale = np.float32(255.0 / 65535.0)
    idx32 = np.float32(index_f)
    if ch == 3:
        for y in range(h):
            for x in range(w):
                p = prob[y, x]
                out[y, x, 0] = np.float32(out[y, x, 0] + np.float32(p * np.float32(np.float32(src[y, x, 0]) * scale)))
                out[y, x, 1] = np.float32(out[y, x, 1] + np.float32(p * np.float32(np.float32(src[y, x, 1]) * scale)))
                out[y, x, 2] = np.float32(out[y, x, 2] + np.float32(p * np.float32(np.float32(src[y, x, 2]) * scale)))
                if has_expected:
                    expected[y, x] = np.float32(expected[y, x] + np.float32(p * idx32))
    else:
        for y in range(h):
            for x in range(w):
                p = prob[y, x]
                for c in range(ch):
                    s = np.float32(src[y, x, c]) * scale
                    out[y, x, c] = np.float32(out[y, x, c] + np.float32(p * s))
                if has_expected:
                    expected[y, x] = np.float32(expected[y, x] + np.float32(p * idx32))


@_optional_kernel
def v3_accumulate_soft_u8(src, prob, out, expected, index_f, has_expected):
    """Accumulate probability-weighted uint8 BGR/gray row band into float32 out."""
    h, w, ch = out.shape
    idx32 = np.float32(index_f)
    if ch == 3:
        for y in range(h):
            for x in range(w):
                p = prob[y, x]
                out[y, x, 0] = np.float32(out[y, x, 0] + np.float32(p * np.float32(src[y, x, 0])))
                out[y, x, 1] = np.float32(out[y, x, 1] + np.float32(p * np.float32(src[y, x, 1])))
                out[y, x, 2] = np.float32(out[y, x, 2] + np.float32(p * np.float32(src[y, x, 2])))
                if has_expected:
                    expected[y, x] = np.float32(expected[y, x] + np.float32(p * idx32))
    else:
        for y in range(h):
            for x in range(w):
                p = prob[y, x]
                for c in range(ch):
                    out[y, x, c] = np.float32(out[y, x, c] + np.float32(p * np.float32(src[y, x, c])))
                if has_expected:
                    expected[y, x] = np.float32(expected[y, x] + np.float32(p * idx32))


@_optional_kernel
def v3_accumulate_soft_rgb_u16(src_rgb, prob, out, expected, index_f, has_expected):
    """Accumulate probability-weighted uint16 RGB row band (0-255) into float32 BGR out."""
    h, w = prob.shape
    scale = np.float32(255.0 / 65535.0)
    idx32 = np.float32(index_f)
    for y in range(h):
        for x in range(w):
            p = prob[y, x]
            out[y, x, 0] = np.float32(out[y, x, 0] + np.float32(p * np.float32(np.float32(src_rgb[y, x, 2]) * scale)))
            out[y, x, 1] = np.float32(out[y, x, 1] + np.float32(p * np.float32(np.float32(src_rgb[y, x, 1]) * scale)))
            out[y, x, 2] = np.float32(out[y, x, 2] + np.float32(p * np.float32(np.float32(src_rgb[y, x, 0]) * scale)))
            if has_expected:
                expected[y, x] = np.float32(expected[y, x] + np.float32(p * idx32))


@_optional_kernel
def v3_accumulate_soft_rgb_u8(src_rgb, prob, out, expected, index_f, has_expected):
    """Accumulate probability-weighted uint8 RGB row band into float32 BGR out."""
    h, w = prob.shape
    idx32 = np.float32(index_f)
    for y in range(h):
        for x in range(w):
            p = prob[y, x]
            out[y, x, 0] = np.float32(out[y, x, 0] + np.float32(p * np.float32(src_rgb[y, x, 2])))
            out[y, x, 1] = np.float32(out[y, x, 1] + np.float32(p * np.float32(src_rgb[y, x, 1])))
            out[y, x, 2] = np.float32(out[y, x, 2] + np.float32(p * np.float32(src_rgb[y, x, 0])))
            if has_expected:
                expected[y, x] = np.float32(expected[y, x] + np.float32(p * idx32))


@_optional_kernel
def v3_scatter_hard_f32(src, selected, layer, out):
    """Copy float32 pixels where selected[y, x] == layer into out."""
    h, w, ch = out.shape
    if ch == 3:
        for y in range(h):
            for x in range(w):
                if selected[y, x] == layer:
                    out[y, x, 0] = src[y, x, 0]
                    out[y, x, 1] = src[y, x, 1]
                    out[y, x, 2] = src[y, x, 2]
    else:
        for y in range(h):
            for x in range(w):
                if selected[y, x] == layer:
                    for c in range(ch):
                        out[y, x, c] = src[y, x, c]


@_optional_kernel
def v3_scatter_hard_u16(src, selected, layer, out):
    """Copy uint16 pixels (normalized to 0-255) where selected[y, x] == layer into out."""
    h, w, ch = out.shape
    scale = np.float32(255.0 / 65535.0)
    if ch == 3:
        for y in range(h):
            for x in range(w):
                if selected[y, x] == layer:
                    out[y, x, 0] = np.float32(src[y, x, 0]) * scale
                    out[y, x, 1] = np.float32(src[y, x, 1]) * scale
                    out[y, x, 2] = np.float32(src[y, x, 2]) * scale
    else:
        for y in range(h):
            for x in range(w):
                if selected[y, x] == layer:
                    for c in range(ch):
                        out[y, x, c] = np.float32(src[y, x, c]) * scale


@_optional_kernel
def v3_scatter_hard_u8(src, selected, layer, out):
    """Copy uint8 pixels as float32 where selected[y, x] == layer into out."""
    h, w, ch = out.shape
    if ch == 3:
        for y in range(h):
            for x in range(w):
                if selected[y, x] == layer:
                    out[y, x, 0] = np.float32(src[y, x, 0])
                    out[y, x, 1] = np.float32(src[y, x, 1])
                    out[y, x, 2] = np.float32(src[y, x, 2])
    else:
        for y in range(h):
            for x in range(w):
                if selected[y, x] == layer:
                    for c in range(ch):
                        out[y, x, c] = np.float32(src[y, x, c])


@_optional_kernel
def v3_scatter_hard_rgb_u16(src_rgb, selected, layer, out):
    """Copy uint16 RGB pixels (normalized to 0-255) where selected[y, x] == layer into BGR out."""
    h, w = selected.shape
    scale = np.float32(255.0 / 65535.0)
    for y in range(h):
        for x in range(w):
            if selected[y, x] == layer:
                out[y, x, 0] = np.float32(src_rgb[y, x, 2]) * scale
                out[y, x, 1] = np.float32(src_rgb[y, x, 1]) * scale
                out[y, x, 2] = np.float32(src_rgb[y, x, 0]) * scale


@_optional_kernel
def v3_scatter_hard_rgb_u8(src_rgb, selected, layer, out):
    """Copy uint8 RGB pixels as float32 where selected[y, x] == layer into BGR out."""
    h, w = selected.shape
    for y in range(h):
        for x in range(w):
            if selected[y, x] == layer:
                out[y, x, 0] = np.float32(src_rgb[y, x, 2])
                out[y, x, 1] = np.float32(src_rgb[y, x, 1])
                out[y, x, 2] = np.float32(src_rgb[y, x, 0])
