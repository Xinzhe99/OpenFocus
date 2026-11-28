"""Module-level compiled CPU DTCWT kernels with Numba disk caching enabled.

Imported lazily inside ``get_native_dtcwt()`` after compiler memory admission.
Defining every ``@njit`` kernel at module top level allows ``cache=True`` in
``fusion_methods._depth_kernel.njit`` to persist compiled machine code across
runs rather than recompiling in memory for each process.
"""
from functools import lru_cache
from threading import Lock

import numpy as np

from fusion_methods._depth_kernel import njit
from fusion_methods._dtcwt_kernels import _magnitude_scratch, make_reflect_table

_SQRT_HALF_32 = np.float32(np.sqrt(0.5))
_W_R = np.float32(0.299)
_W_G = np.float32(0.587)
_W_B = np.float32(0.114)
_INV_255 = np.float32(255.0)
_U16_SCALE = np.float32(255.0 / 65535.0)

# Bit-exact 256-entry float32 lookup tables eliminate per-pixel FP32 divisions.
_LUT_U8 = (np.arange(256, dtype=np.float32) / _INV_255).astype(np.float32)
_LUT_R = (_W_R * _LUT_U8).astype(np.float32)
_LUT_G = (_W_G * _LUT_U8).astype(np.float32)
_LUT_B = (_W_B * _LUT_U8).astype(np.float32)
_LUT_GRAY = ((_LUT_R + _LUT_G) + _LUT_B).astype(np.float32)


@njit(nogil=True, fastmath=False)
def _ref_idx(i, n):
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i - 1
        if i >= n:
            i = 2 * n - 1 - i
    return i


@njit(nogil=True, fastmath=False)
def _extract_luma_u8_3_rows(src, dst, h, w, lut_r, lut_g, lut_b, y_start, y_stop):
    c0 = dst.shape[1]
    for y in range(y_start, y_stop):
        sy = y if y < h else h - 1
        s_row = src[sy]
        d_row = dst[y]
        for x in range(w):
            d_row[x] = np.float32(lut_r[s_row[x, 2]] + lut_g[s_row[x, 1]]) + lut_b[s_row[x, 0]]
        if c0 > w:
            d_row[w] = d_row[w - 1]


@njit(nogil=True, fastmath=False)
def _extract_luma_u8_3(src, dst, h, w, lut_r, lut_g, lut_b):
    _extract_luma_u8_3_rows(src, dst, h, w, lut_r, lut_g, lut_b, 0, dst.shape[0])


@njit(nogil=True, fastmath=False)
def _extract_luma_u8_2_rows(src, dst, h, w, lut_gray, y_start, y_stop):
    c0 = dst.shape[1]
    for y in range(y_start, y_stop):
        sy = y if y < h else h - 1
        s_row = src[sy]
        d_row = dst[y]
        for x in range(w):
            d_row[x] = lut_gray[s_row[x]]
        if c0 > w:
            d_row[w] = d_row[w - 1]


@njit(nogil=True, fastmath=False)
def _extract_luma_u8_2(src, dst, h, w, lut_gray):
    _extract_luma_u8_2_rows(src, dst, h, w, lut_gray, 0, dst.shape[0])


@njit(nogil=True, fastmath=False)
def _extract_luma_u16_3(src, dst, h, w):
    r0, c0 = dst.shape
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            b = np.float32(np.float32(src[sy, sx, 0]) * _U16_SCALE) / _INV_255
            g = np.float32(np.float32(src[sy, sx, 1]) * _U16_SCALE) / _INV_255
            r = np.float32(np.float32(src[sy, sx, 2]) * _U16_SCALE) / _INV_255
            dst[y, x] = np.float32(
                np.float32(_W_R * r) + np.float32(_W_G * g)
            ) + np.float32(_W_B * b)


@njit(nogil=True, fastmath=False)
def _extract_luma_u16_2(src, dst, h, w):
    r0, c0 = dst.shape
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            v = np.float32(np.float32(src[sy, sx]) * _U16_SCALE) / _INV_255
            dst[y, x] = np.float32(
                np.float32(_W_R * v) + np.float32(_W_G * v)
            ) + np.float32(_W_B * v)


@njit(nogil=True, fastmath=False)
def _extract_luma_f32_3(src, dst, h, w):
    r0, c0 = dst.shape
    nonfinite = False
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            v0 = np.float32(src[sy, sx, 0])
            v1 = np.float32(src[sy, sx, 1])
            v2 = np.float32(src[sy, sx, 2])
            if (v0 - v0 != np.float32(0.0)) or (v1 - v1 != np.float32(0.0)) or (v2 - v2 != np.float32(0.0)):
                nonfinite = True
            b = v0 / _INV_255
            g = v1 / _INV_255
            r = v2 / _INV_255
            dst[y, x] = np.float32(
                np.float32(_W_R * r) + np.float32(_W_G * g)
            ) + np.float32(_W_B * b)
    return nonfinite


@njit(nogil=True, fastmath=False)
def _extract_luma_f32_2(src, dst, h, w):
    r0, c0 = dst.shape
    nonfinite = False
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            raw = np.float32(src[sy, sx])
            if raw - raw != np.float32(0.0):
                nonfinite = True
            v = raw / _INV_255
            dst[y, x] = np.float32(
                np.float32(_W_R * v) + np.float32(_W_G * v)
            ) + np.float32(_W_B * v)
    return nonfinite


@njit(nogil=True, fastmath=False)
def _extract_chan_u8_3_rows(src, dst, h, w, ch, lut_u8, y_start, y_stop):
    c0 = dst.shape[1]
    for y in range(y_start, y_stop):
        sy = y if y < h else h - 1
        s_row = src[sy]
        d_row = dst[y]
        for x in range(w):
            d_row[x] = lut_u8[s_row[x, ch]]
        if c0 > w:
            d_row[w] = d_row[w - 1]


@njit(nogil=True, fastmath=False)
def _extract_chan_u8_3(src, dst, h, w, ch, lut_u8):
    _extract_chan_u8_3_rows(src, dst, h, w, ch, lut_u8, 0, dst.shape[0])


@njit(nogil=True, fastmath=False)
def _extract_chan_u8_2_rows(src, dst, h, w, lut_u8, y_start, y_stop):
    c0 = dst.shape[1]
    for y in range(y_start, y_stop):
        sy = y if y < h else h - 1
        s_row = src[sy]
        d_row = dst[y]
        for x in range(w):
            d_row[x] = lut_u8[s_row[x]]
        if c0 > w:
            d_row[w] = d_row[w - 1]


@njit(nogil=True, fastmath=False)
def _extract_chan_u8_2(src, dst, h, w, lut_u8):
    _extract_chan_u8_2_rows(src, dst, h, w, lut_u8, 0, dst.shape[0])


@njit(nogil=True, fastmath=False)
def _extract_chan_u16_3(src, dst, h, w, ch):
    r0, c0 = dst.shape
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            dst[y, x] = np.float32(np.float32(src[sy, sx, ch]) * _U16_SCALE) / _INV_255


@njit(nogil=True, fastmath=False)
def _extract_chan_u16_2(src, dst, h, w):
    r0, c0 = dst.shape
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            dst[y, x] = np.float32(np.float32(src[sy, sx]) * _U16_SCALE) / _INV_255


@njit(nogil=True, fastmath=False)
def _extract_chan_f32_3(src, dst, h, w, ch):
    r0, c0 = dst.shape
    nonfinite = False
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            raw = np.float32(src[sy, sx, ch])
            if raw - raw != np.float32(0.0):
                nonfinite = True
            dst[y, x] = raw / _INV_255
    return nonfinite


@njit(nogil=True, fastmath=False)
def _extract_chan_f32_2(src, dst, h, w):
    r0, c0 = dst.shape
    nonfinite = False
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            raw = np.float32(src[sy, sx])
            if raw - raw != np.float32(0.0):
                nonfinite = True
            dst[y, x] = raw / _INV_255
    return nonfinite


@njit(nogil=True, fastmath=False)
def _colfilter_v5_rows(src, dst, h5, r, c, y_start, y_stop):
    w0, w1, w2, w3, w4 = h5[0], h5[1], h5[2], h5[3], h5[4]
    z = np.float32(0.0)
    head_stop = min(y_stop, 2)
    int_start = max(y_start, head_stop)
    int_stop = max(int_start, min(y_stop, r - 2))
    for y in range(y_start, min(y_stop, head_stop)):
        rp2 = src[_ref_idx(y + 2, r)]
        rp1 = src[_ref_idx(y + 1, r)]
        r0 = src[y]
        rm1 = src[_ref_idx(y - 1, r)]
        rm2 = src[_ref_idx(y - 2, r)]
        out = dst[y]
        for x in range(c):
            acc = z
            acc += rp2[x] * w0
            acc += rp1[x] * w1
            acc += r0[x] * w2
            acc += rm1[x] * w3
            acc += rm2[x] * w4
            out[x] = acc
    for y in range(int_start, int_stop):
        rp2 = src[y + 2]
        rp1 = src[y + 1]
        r0 = src[y]
        rm1 = src[y - 1]
        rm2 = src[y - 2]
        out = dst[y]
        for x in range(c):
            acc = z
            acc += rp2[x] * w0
            acc += rp1[x] * w1
            acc += r0[x] * w2
            acc += rm1[x] * w3
            acc += rm2[x] * w4
            out[x] = acc
    for y in range(int_stop, y_stop):
        rp2 = src[_ref_idx(y + 2, r)]
        rp1 = src[_ref_idx(y + 1, r)]
        r0 = src[y]
        rm1 = src[_ref_idx(y - 1, r)]
        rm2 = src[_ref_idx(y - 2, r)]
        out = dst[y]
        for x in range(c):
            acc = z
            acc += rp2[x] * w0
            acc += rp1[x] * w1
            acc += r0[x] * w2
            acc += rm1[x] * w3
            acc += rm2[x] * w4
            out[x] = acc


@njit(nogil=True, fastmath=False)
def _colfilter_v5(src, dst, h5, r, c):
    _colfilter_v5_rows(src, dst, h5, r, c, 0, r)


@njit(nogil=True, fastmath=False)
def _colfilter_v5_v7_strip(src, lo_dst, hi_dst, h5, h7, r, c, y_start, y_stop):
    """Fused single-pass vertical 5-tap (lo) + 7-tap (hi) over a row strip."""
    a0, a1, a2, a3, a4 = h5[0], h5[1], h5[2], h5[3], h5[4]
    w0, w1, w2, w3, w4, w5, w6 = (
        h7[0], h7[1], h7[2], h7[3], h7[4], h7[5], h7[6]
    )
    z = np.float32(0.0)
    head_stop = min(y_stop, 3)
    int_start = max(y_start, head_stop)
    int_stop = max(int_start, min(y_stop, r - 3))
    for y in range(y_start, min(y_stop, head_stop)):
        rp3 = src[_ref_idx(y + 3, r)]
        rp2 = src[_ref_idx(y + 2, r)]
        rp1 = src[_ref_idx(y + 1, r)]
        r0 = src[y]
        rm1 = src[_ref_idx(y - 1, r)]
        rm2 = src[_ref_idx(y - 2, r)]
        rm3 = src[_ref_idx(y - 3, r)]
        dy = y - y_start
        lo_row = lo_dst[dy]
        hi_row = hi_dst[dy]
        for x in range(c):
            s6 = rp3[x]
            s5 = rp2[x]
            s4 = rp1[x]
            s3 = r0[x]
            s2 = rm1[x]
            s1 = rm2[x]
            s0 = rm3[x]
            al = z
            al += s5 * a0
            al += s4 * a1
            al += s3 * a2
            al += s2 * a3
            al += s1 * a4
            lo_row[x] = al
            ah = z
            ah += s6 * w0
            ah += s5 * w1
            ah += s4 * w2
            ah += s3 * w3
            ah += s2 * w4
            ah += s1 * w5
            ah += s0 * w6
            hi_row[x] = ah
    for y in range(int_start, int_stop):
        rp3 = src[y + 3]
        rp2 = src[y + 2]
        rp1 = src[y + 1]
        r0 = src[y]
        rm1 = src[y - 1]
        rm2 = src[y - 2]
        rm3 = src[y - 3]
        dy = y - y_start
        lo_row = lo_dst[dy]
        hi_row = hi_dst[dy]
        for x in range(c):
            s6 = rp3[x]
            s5 = rp2[x]
            s4 = rp1[x]
            s3 = r0[x]
            s2 = rm1[x]
            s1 = rm2[x]
            s0 = rm3[x]
            al = z
            al += s5 * a0
            al += s4 * a1
            al += s3 * a2
            al += s2 * a3
            al += s1 * a4
            lo_row[x] = al
            ah = z
            ah += s6 * w0
            ah += s5 * w1
            ah += s4 * w2
            ah += s3 * w3
            ah += s2 * w4
            ah += s1 * w5
            ah += s0 * w6
            hi_row[x] = ah
    for y in range(int_stop, y_stop):
        rp3 = src[_ref_idx(y + 3, r)]
        rp2 = src[_ref_idx(y + 2, r)]
        rp1 = src[_ref_idx(y + 1, r)]
        r0 = src[y]
        rm1 = src[_ref_idx(y - 1, r)]
        rm2 = src[_ref_idx(y - 2, r)]
        rm3 = src[_ref_idx(y - 3, r)]
        dy = y - y_start
        lo_row = lo_dst[dy]
        hi_row = hi_dst[dy]
        for x in range(c):
            s6 = rp3[x]
            s5 = rp2[x]
            s4 = rp1[x]
            s3 = r0[x]
            s2 = rm1[x]
            s1 = rm2[x]
            s0 = rm3[x]
            al = z
            al += s5 * a0
            al += s4 * a1
            al += s3 * a2
            al += s2 * a3
            al += s1 * a4
            lo_row[x] = al
            ah = z
            ah += s6 * w0
            ah += s5 * w1
            ah += s4 * w2
            ah += s3 * w3
            ah += s2 * w4
            ah += s1 * w5
            ah += s0 * w6
            hi_row[x] = ah


@njit(nogil=True, fastmath=False)
def _colfilter_v7_rows_span(src, dst, h7, r, y_start, y_stop, x_lo, x_hi):
    w0, w1, w2, w3, w4, w5, w6 = (
        h7[0], h7[1], h7[2], h7[3], h7[4], h7[5], h7[6]
    )
    z = np.float32(0.0)
    head_stop = min(y_stop, 3)
    int_start = max(y_start, head_stop)
    int_stop = max(int_start, min(y_stop, r - 3))
    for y in range(y_start, min(y_stop, head_stop)):
        rp3 = src[_ref_idx(y + 3, r)]
        rp2 = src[_ref_idx(y + 2, r)]
        rp1 = src[_ref_idx(y + 1, r)]
        r0 = src[y]
        rm1 = src[_ref_idx(y - 1, r)]
        rm2 = src[_ref_idx(y - 2, r)]
        rm3 = src[_ref_idx(y - 3, r)]
        out = dst[y - y_start]
        for x in range(x_lo, x_hi):
            acc = z
            acc += rp3[x] * w0
            acc += rp2[x] * w1
            acc += rp1[x] * w2
            acc += r0[x] * w3
            acc += rm1[x] * w4
            acc += rm2[x] * w5
            acc += rm3[x] * w6
            out[x] = acc
    for y in range(int_start, int_stop):
        rp3 = src[y + 3]
        rp2 = src[y + 2]
        rp1 = src[y + 1]
        r0 = src[y]
        rm1 = src[y - 1]
        rm2 = src[y - 2]
        rm3 = src[y - 3]
        out = dst[y - y_start]
        for x in range(x_lo, x_hi):
            acc = z
            acc += rp3[x] * w0
            acc += rp2[x] * w1
            acc += rp1[x] * w2
            acc += r0[x] * w3
            acc += rm1[x] * w4
            acc += rm2[x] * w5
            acc += rm3[x] * w6
            out[x] = acc
    for y in range(int_stop, y_stop):
        rp3 = src[_ref_idx(y + 3, r)]
        rp2 = src[_ref_idx(y + 2, r)]
        rp1 = src[_ref_idx(y + 1, r)]
        r0 = src[y]
        rm1 = src[_ref_idx(y - 1, r)]
        rm2 = src[_ref_idx(y - 2, r)]
        rm3 = src[_ref_idx(y - 3, r)]
        out = dst[y - y_start]
        for x in range(x_lo, x_hi):
            acc = z
            acc += rp3[x] * w0
            acc += rp2[x] * w1
            acc += rp1[x] * w2
            acc += r0[x] * w3
            acc += rm1[x] * w4
            acc += rm2[x] * w5
            acc += rm3[x] * w6
            out[x] = acc


@njit(nogil=True, fastmath=False)
def _colfilter_h5_span(src, dst, h5, c, y_start, y_stop):
    w0, w1, w2, w3, w4 = h5[0], h5[1], h5[2], h5[3], h5[4]
    z = np.float32(0.0)
    x_lo = 2 if c >= 4 else c
    x_hi = max(x_lo, c - 2)
    for y in range(y_start, y_stop):
        row = src[y]
        out = dst[y]
        for x in range(x_lo):
            acc = z
            acc += row[_ref_idx(x + 2, c)] * w0
            acc += row[_ref_idx(x + 1, c)] * w1
            acc += row[x] * w2
            acc += row[_ref_idx(x - 1, c)] * w3
            acc += row[_ref_idx(x - 2, c)] * w4
            out[x] = acc
        for x in range(x_lo, x_hi):
            acc = z
            acc += row[x + 2] * w0
            acc += row[x + 1] * w1
            acc += row[x] * w2
            acc += row[x - 1] * w3
            acc += row[x - 2] * w4
            out[x] = acc
        for x in range(x_hi, c):
            acc = z
            acc += row[_ref_idx(x + 2, c)] * w0
            acc += row[_ref_idx(x + 1, c)] * w1
            acc += row[x] * w2
            acc += row[_ref_idx(x - 1, c)] * w3
            acc += row[_ref_idx(x - 2, c)] * w4
            out[x] = acc


@njit(nogil=True, fastmath=False)
def _colfilter_h5(src, dst, h5, r, c):
    _colfilter_h5_span(src, dst, h5, c, 0, r)


@njit(nogil=True, fastmath=False)
def _colfilter_h5_rows(src_strip, dst_full, h5, dst_y0, n_rows, c):
    """Horizontal 5-tap filter from L2-hot strip into dst_full[dst_y0 : dst_y0 + n_rows]."""
    w0, w1, w2, w3, w4 = h5[0], h5[1], h5[2], h5[3], h5[4]
    z = np.float32(0.0)
    x_lo = 2 if c >= 4 else c
    x_hi = max(x_lo, c - 2)
    for ry in range(n_rows):
        row = src_strip[ry]
        out = dst_full[dst_y0 + ry]
        for x in range(x_lo):
            acc = z
            acc += row[_ref_idx(x + 2, c)] * w0
            acc += row[_ref_idx(x + 1, c)] * w1
            acc += row[x] * w2
            acc += row[_ref_idx(x - 1, c)] * w3
            acc += row[_ref_idx(x - 2, c)] * w4
            out[x] = acc
        for x in range(x_lo, x_hi):
            acc = z
            acc += row[x + 2] * w0
            acc += row[x + 1] * w1
            acc += row[x] * w2
            acc += row[x - 1] * w3
            acc += row[x - 2] * w4
            out[x] = acc
        for x in range(x_hi, c):
            acc = z
            acc += row[_ref_idx(x + 2, c)] * w0
            acc += row[_ref_idx(x + 1, c)] * w1
            acc += row[x] * w2
            acc += row[_ref_idx(x - 1, c)] * w3
            acc += row[_ref_idx(x - 2, c)] * w4
            out[x] = acc


@njit(nogil=True, fastmath=False)
def _conv5_x(row, x, c, h5):
    z = np.float32(0.0)
    if 2 <= x < c - 2:
        acc = z
        acc += row[x + 2] * h5[0]
        acc += row[x + 1] * h5[1]
        acc += row[x] * h5[2]
        acc += row[x - 1] * h5[3]
        acc += row[x - 2] * h5[4]
        return acc
    acc = z
    acc += row[_ref_idx(x + 2, c)] * h5[0]
    acc += row[_ref_idx(x + 1, c)] * h5[1]
    acc += row[x] * h5[2]
    acc += row[_ref_idx(x - 1, c)] * h5[3]
    acc += row[_ref_idx(x - 2, c)] * h5[4]
    return acc


@njit(nogil=True, fastmath=False)
def _conv7_x(row, x, c, h7):
    z = np.float32(0.0)
    if 3 <= x < c - 3:
        acc = z
        acc += row[x + 3] * h7[0]
        acc += row[x + 2] * h7[1]
        acc += row[x + 1] * h7[2]
        acc += row[x] * h7[3]
        acc += row[x - 1] * h7[4]
        acc += row[x - 2] * h7[5]
        acc += row[x - 3] * h7[6]
        return acc
    acc = z
    acc += row[_ref_idx(x + 3, c)] * h7[0]
    acc += row[_ref_idx(x + 2, c)] * h7[1]
    acc += row[_ref_idx(x + 1, c)] * h7[2]
    acc += row[x] * h7[3]
    acc += row[_ref_idx(x - 1, c)] * h7[4]
    acc += row[_ref_idx(x - 2, c)] * h7[5]
    acc += row[_ref_idx(x - 3, c)] * h7[6]
    return acc


@njit(nogil=True, fastmath=False)
def _store_q2c(dst_hp, y, qx, a, b, cc, d, c0, c1):
    s = _SQRT_HALF_32
    a_s = np.float32(a * s)
    b_s = np.float32(b * s)
    c_s = np.float32(cc * s)
    d_s = np.float32(d * s)
    dst_hp[y, qx, c0] = np.complex64(np.float32(a_s - d_s) + 1j * np.float32(b_s + c_s))
    dst_hp[y, qx, c1] = np.complex64(np.float32(a_s + d_s) + 1j * np.float32(b_s - c_s))


@njit(nogil=True, fastmath=False)
def _colfilter_h_q2c_strip_local(lo_strip, hi_strip, hp_strip, h0, h1, n_pairs, c):
    hp_w = c >> 1
    a0, a1, a2, a3, a4 = h0[0], h0[1], h0[2], h0[3], h0[4]
    b0, b1, b2, b3, b4, b5, b6 = (
        h1[0], h1[1], h1[2], h1[3], h1[4], h1[5], h1[6]
    )
    z = np.float32(0.0)
    s = _SQRT_HALF_32
    qx_lo = 2 if hp_w >= 4 else hp_w
    qx_hi = max(qx_lo, hp_w - 2)

    for yp in range(n_pairs):
        lo_top = lo_strip[2 * yp]
        lo_bot = lo_strip[2 * yp + 1]
        hi_top = hi_strip[2 * yp]
        hi_bot = hi_strip[2 * yp + 1]

        for qx in range(qx_lo):
            cx = 2 * qx
            _store_q2c(
                hp_strip, yp, qx,
                _conv5_x(hi_top, cx, c, h0),
                _conv5_x(hi_top, cx + 1, c, h0),
                _conv5_x(hi_bot, cx, c, h0),
                _conv5_x(hi_bot, cx + 1, c, h0),
                0, 5,
            )
            _store_q2c(
                hp_strip, yp, qx,
                _conv7_x(lo_top, cx, c, h1),
                _conv7_x(lo_top, cx + 1, c, h1),
                _conv7_x(lo_bot, cx, c, h1),
                _conv7_x(lo_bot, cx + 1, c, h1),
                2, 3,
            )
            _store_q2c(
                hp_strip, yp, qx,
                _conv7_x(hi_top, cx, c, h1),
                _conv7_x(hi_top, cx + 1, c, h1),
                _conv7_x(hi_bot, cx, c, h1),
                _conv7_x(hi_bot, cx + 1, c, h1),
                1, 4,
            )

        for qx in range(qx_lo, qx_hi):
            cx = 2 * qx
            aa = (((((z + hi_top[cx + 2] * a0) + hi_top[cx + 1] * a1) + hi_top[cx] * a2) + hi_top[cx - 1] * a3) + hi_top[cx - 2] * a4) * s
            ab = (((((z + hi_top[cx + 3] * a0) + hi_top[cx + 2] * a1) + hi_top[cx + 1] * a2) + hi_top[cx] * a3) + hi_top[cx - 1] * a4) * s
            ac = (((((z + hi_bot[cx + 2] * a0) + hi_bot[cx + 1] * a1) + hi_bot[cx] * a2) + hi_bot[cx - 1] * a3) + hi_bot[cx - 2] * a4) * s
            ad = (((((z + hi_bot[cx + 3] * a0) + hi_bot[cx + 2] * a1) + hi_bot[cx + 1] * a2) + hi_bot[cx] * a3) + hi_bot[cx - 1] * a4) * s
            hp_strip[yp, qx, 0] = np.float32(aa - ad) + 1j * np.float32(ab + ac)
            hp_strip[yp, qx, 5] = np.float32(aa + ad) + 1j * np.float32(ab - ac)

            la = (((((((z + lo_top[cx + 3] * b0) + lo_top[cx + 2] * b1) + lo_top[cx + 1] * b2) + lo_top[cx] * b3) + lo_top[cx - 1] * b4) + lo_top[cx - 2] * b5) + lo_top[cx - 3] * b6) * s
            lb = (((((((z + lo_top[cx + 4] * b0) + lo_top[cx + 3] * b1) + lo_top[cx + 2] * b2) + lo_top[cx + 1] * b3) + lo_top[cx] * b4) + lo_top[cx - 1] * b5) + lo_top[cx - 2] * b6) * s
            lc = (((((((z + lo_bot[cx + 3] * b0) + lo_bot[cx + 2] * b1) + lo_bot[cx + 1] * b2) + lo_bot[cx] * b3) + lo_bot[cx - 1] * b4) + lo_bot[cx - 2] * b5) + lo_bot[cx - 3] * b6) * s
            ld = (((((((z + lo_bot[cx + 4] * b0) + lo_bot[cx + 3] * b1) + lo_bot[cx + 2] * b2) + lo_bot[cx + 1] * b3) + lo_bot[cx] * b4) + lo_bot[cx - 1] * b5) + lo_bot[cx - 2] * b6) * s
            hp_strip[yp, qx, 2] = np.float32(la - ld) + 1j * np.float32(lb + lc)
            hp_strip[yp, qx, 3] = np.float32(la + ld) + 1j * np.float32(lb - lc)

            ha = (((((((z + hi_top[cx + 3] * b0) + hi_top[cx + 2] * b1) + hi_top[cx + 1] * b2) + hi_top[cx] * b3) + hi_top[cx - 1] * b4) + hi_top[cx - 2] * b5) + hi_top[cx - 3] * b6) * s
            hb = (((((((z + hi_top[cx + 4] * b0) + hi_top[cx + 3] * b1) + hi_top[cx + 2] * b2) + hi_top[cx + 1] * b3) + hi_top[cx] * b4) + hi_top[cx - 1] * b5) + hi_top[cx - 2] * b6) * s
            hc = (((((((z + hi_bot[cx + 3] * b0) + hi_bot[cx + 2] * b1) + hi_bot[cx + 1] * b2) + hi_bot[cx] * b3) + hi_bot[cx - 1] * b4) + hi_bot[cx - 2] * b5) + hi_bot[cx - 3] * b6) * s
            hd = (((((((z + hi_bot[cx + 4] * b0) + hi_bot[cx + 3] * b1) + hi_bot[cx + 2] * b2) + hi_bot[cx + 1] * b3) + hi_bot[cx] * b4) + hi_bot[cx - 1] * b5) + hi_bot[cx - 2] * b6) * s
            hp_strip[yp, qx, 1] = np.float32(ha - hd) + 1j * np.float32(hb + hc)
            hp_strip[yp, qx, 4] = np.float32(ha + hd) + 1j * np.float32(hb - hc)

        for qx in range(qx_hi, hp_w):
            cx = 2 * qx
            _store_q2c(
                hp_strip, yp, qx,
                _conv5_x(hi_top, cx, c, h0),
                _conv5_x(hi_top, cx + 1, c, h0),
                _conv5_x(hi_bot, cx, c, h0),
                _conv5_x(hi_bot, cx + 1, c, h0),
                0, 5,
            )
            _store_q2c(
                hp_strip, yp, qx,
                _conv7_x(lo_top, cx, c, h1),
                _conv7_x(lo_top, cx + 1, c, h1),
                _conv7_x(lo_bot, cx, c, h1),
                _conv7_x(lo_bot, cx + 1, c, h1),
                2, 3,
            )
            _store_q2c(
                hp_strip, yp, qx,
                _conv7_x(hi_top, cx, c, h1),
                _conv7_x(hi_top, cx + 1, c, h1),
                _conv7_x(hi_bot, cx, c, h1),
                _conv7_x(hi_bot, cx + 1, c, h1),
                1, 4,
            )


@njit(nogil=True, fastmath=False)
def _level0_select_sparse_rows(src_x, lo_full, has_lo_full, decisions, frame_idx,
                               out_hp, is_packed, h0, h1, r, c, lo_2row, hi_2row,
                               qy_start, qy_stop):
    hp_w = decisions.shape[1]
    pos = 0
    w0, w1, w2, w3, w4 = h0[0], h0[1], h0[2], h0[3], h0[4]
    z = np.float32(0.0)

    for qy in range(qy_start, qy_stop):
        dec_row = decisions[qy]
        first_qx = -1
        last_qx = -1
        for qx in range(hp_w):
            if dec_row[qx] == frame_idx:
                if first_qx < 0:
                    first_qx = qx
                last_qx = qx
        if first_qx < 0:
            continue

        # Compute vertical convolution only over the column span needed by winning qx.
        if c < 8:
            x_lo = 0
            x_hi = c
        else:
            x_lo = 0 if first_qx < 2 else (2 * first_qx - 3)
            x_hi = c if last_qx >= hp_w - 2 else (2 * last_qx + 5)
            if first_qx < 2 and x_hi < 4:
                x_hi = 4
            if last_qx >= hp_w - 2 and x_lo > c - 4:
                x_lo = c - 4
            if x_lo < 0:
                x_lo = 0
            if x_hi > c:
                x_hi = c

        y0 = 2 * qy
        _colfilter_v7_rows_span(src_x, hi_2row, h1, r, y0, y0 + 2, x_lo, x_hi)
        hi_top = hi_2row[0]
        hi_bot = hi_2row[1]
        if has_lo_full:
            lo_top = lo_full[y0]
            lo_bot = lo_full[y0 + 1]
        else:
            if 2 <= y0 and y0 + 1 < r - 2:
                for row_off in range(2):
                    ry = y0 + row_off
                    rp2 = src_x[ry + 2]
                    rp1 = src_x[ry + 1]
                    r0_s = src_x[ry]
                    rm1 = src_x[ry - 1]
                    rm2 = src_x[ry - 2]
                    out_l = lo_2row[row_off]
                    for cx_i in range(x_lo, x_hi):
                        acc = z
                        acc += rp2[cx_i] * w0
                        acc += rp1[cx_i] * w1
                        acc += r0_s[cx_i] * w2
                        acc += rm1[cx_i] * w3
                        acc += rm2[cx_i] * w4
                        out_l[cx_i] = acc
            else:
                for row_off in range(2):
                    ry = y0 + row_off
                    rp2 = src_x[_ref_idx(ry + 2, r)]
                    rp1 = src_x[_ref_idx(ry + 1, r)]
                    r0_s = src_x[ry]
                    rm1 = src_x[_ref_idx(ry - 1, r)]
                    rm2 = src_x[_ref_idx(ry - 2, r)]
                    out_l = lo_2row[row_off]
                    for cx_i in range(x_lo, x_hi):
                        acc = z
                        acc += rp2[cx_i] * w0
                        acc += rp1[cx_i] * w1
                        acc += r0_s[cx_i] * w2
                        acc += rm1[cx_i] * w3
                        acc += rm2[cx_i] * w4
                        out_l[cx_i] = acc
            lo_top = lo_2row[0]
            lo_bot = lo_2row[1]

        for qx in range(first_qx, last_qx + 1):
            if dec_row[qx] != frame_idx:
                continue
            cx = 2 * qx
            out_y = pos if is_packed else qy
            out_x = 0 if is_packed else qx
            _store_q2c(
                out_hp, out_y, out_x,
                _conv5_x(hi_top, cx, c, h0),
                _conv5_x(hi_top, cx + 1, c, h0),
                _conv5_x(hi_bot, cx, c, h0),
                _conv5_x(hi_bot, cx + 1, c, h0),
                0, 5,
            )
            _store_q2c(
                out_hp, out_y, out_x,
                _conv7_x(lo_top, cx, c, h1),
                _conv7_x(lo_top, cx + 1, c, h1),
                _conv7_x(lo_bot, cx, c, h1),
                _conv7_x(lo_bot, cx + 1, c, h1),
                2, 3,
            )
            _store_q2c(
                out_hp, out_y, out_x,
                _conv7_x(hi_top, cx, c, h1),
                _conv7_x(hi_top, cx + 1, c, h1),
                _conv7_x(hi_bot, cx, c, h1),
                _conv7_x(hi_bot, cx + 1, c, h1),
                1, 4,
            )
            if is_packed:
                pos += 1


@njit(nogil=True, fastmath=False)
def _level0_select_sparse(src_x, lo_full, has_lo_full, decisions, frame_idx,
                          out_hp, is_packed, h0, h1, r, c, lo_2row, hi_2row):
    _level0_select_sparse_rows(
        src_x, lo_full, has_lo_full, decisions, frame_idx,
        out_hp, is_packed, h0, h1, r, c, lo_2row, hi_2row, 0, decisions.shape[0],
    )


@njit(nogil=True, fastmath=False)
def _coldfilt_v_rows_span(src, dst, ha, hb, pos_sign, xe_r, j_start, j_stop, x_lo, x_hi):
    ha0, ha2, ha4, ha6, ha8 = ha[0], ha[2], ha[4], ha[6], ha[8]
    ha1, ha3, ha5, ha7, ha9 = ha[1], ha[3], ha[5], ha[7], ha[9]
    hb0, hb2, hb4, hb6, hb8 = hb[0], hb[2], hb[4], hb[6], hb[8]
    hb1, hb3, hb5, hb7, hb9 = hb[1], hb[3], hb[5], hb[7], hb[9]
    z = np.float32(0.0)
    off1 = 0 if pos_sign else 1
    off2 = 1 if pos_sign else 0

    for j in range(j_start, j_stop):
        p = 10 + 4 * j
        se0 = src[xe_r[p + 10]]
        se1 = src[xe_r[p + 8]]
        se2 = src[xe_r[p + 6]]
        se3 = src[xe_r[p + 4]]
        se4 = src[xe_r[p + 2]]
        se5 = src[xe_r[p]]
        se6 = src[xe_r[p - 2]]
        se7 = src[xe_r[p - 4]]
        se8 = src[xe_r[p - 6]]
        se9 = src[xe_r[p - 8]]

        so0 = src[xe_r[p + 11]]
        so1 = src[xe_r[p + 9]]
        so2 = src[xe_r[p + 7]]
        so3 = src[xe_r[p + 5]]
        so4 = src[xe_r[p + 3]]
        so5 = src[xe_r[p + 1]]
        so6 = src[xe_r[p - 1]]
        so7 = src[xe_r[p - 3]]
        so8 = src[xe_r[p - 5]]
        so9 = src[xe_r[p - 7]]

        out1 = dst[2 * (j - j_start) + off1]
        out2 = dst[2 * (j - j_start) + off2]
        for x in range(x_lo, x_hi):
            a_even = ((((z + se0[x] * ha0) + se2[x] * ha2) + se4[x] * ha4) + se6[x] * ha6) + se8[x] * ha8
            a_odd = ((((z + se1[x] * ha1) + se3[x] * ha3) + se5[x] * ha5) + se7[x] * ha7) + se9[x] * ha9
            out1[x] = a_even + a_odd

            b_even = ((((z + so0[x] * hb0) + so2[x] * hb2) + so4[x] * hb4) + so6[x] * hb6) + so8[x] * hb8
            b_odd = ((((z + so1[x] * hb1) + so3[x] * hb3) + so5[x] * hb5) + so7[x] * hb7) + so9[x] * hb9
            out2[x] = b_even + b_odd


@njit(nogil=True, fastmath=False)
def _coldfilt_v_rows(src, dst, ha, hb, pos_sign, xe_r, j_start, j_stop, c):
    _coldfilt_v_rows_span(src, dst, ha, hb, pos_sign, xe_r, j_start, j_stop, 0, c)


@njit(nogil=True, fastmath=False)
def _coldfilt_h_eval(row, ha, hb, pos_sign, xe_c, j):
    z = np.float32(0.0)
    p = 10 + 4 * j
    a_even = ((((z + row[xe_c[p + 10]] * ha[0]) + row[xe_c[p + 6]] * ha[2]) + row[xe_c[p + 2]] * ha[4]) + row[xe_c[p - 2]] * ha[6]) + row[xe_c[p - 6]] * ha[8]
    a_odd = ((((z + row[xe_c[p + 8]] * ha[1]) + row[xe_c[p + 4]] * ha[3]) + row[xe_c[p]] * ha[5]) + row[xe_c[p - 4]] * ha[7]) + row[xe_c[p - 8]] * ha[9]
    v1 = a_even + a_odd

    b_even = ((((z + row[xe_c[p + 11]] * hb[0]) + row[xe_c[p + 7]] * hb[2]) + row[xe_c[p + 3]] * hb[4]) + row[xe_c[p - 1]] * hb[6]) + row[xe_c[p - 5]] * hb[8]
    b_odd = ((((z + row[xe_c[p + 9]] * hb[1]) + row[xe_c[p + 5]] * hb[3]) + row[xe_c[p + 1]] * hb[5]) + row[xe_c[p - 3]] * hb[7]) + row[xe_c[p - 7]] * hb[9]
    v2 = b_even + b_odd
    if pos_sign:
        return v1, v2
    return v2, v1


@njit(nogil=True, fastmath=False)
def _coldfilt_h_rows(src, dst, ha, hb, pos_sign, xe_c, y_start, y_stop, groups_c):
    ha0, ha2, ha4, ha6, ha8 = ha[0], ha[2], ha[4], ha[6], ha[8]
    ha1, ha3, ha5, ha7, ha9 = ha[1], ha[3], ha[5], ha[7], ha[9]
    hb0, hb2, hb4, hb6, hb8 = hb[0], hb[2], hb[4], hb[6], hb[8]
    hb1, hb3, hb5, hb7, hb9 = hb[1], hb[3], hb[5], hb[7], hb[9]
    z = np.float32(0.0)
    off1 = 0 if pos_sign else 1
    off2 = 1 if pos_sign else 0

    for y in range(y_start, y_stop):
        row = src[y]
        out = dst[y]
        for j in range(groups_c):
            p = 10 + 4 * j
            a_even = ((((z + row[xe_c[p + 10]] * ha0) + row[xe_c[p + 6]] * ha2) + row[xe_c[p + 2]] * ha4) + row[xe_c[p - 2]] * ha6) + row[xe_c[p - 6]] * ha8
            a_odd = ((((z + row[xe_c[p + 8]] * ha1) + row[xe_c[p + 4]] * ha3) + row[xe_c[p]] * ha5) + row[xe_c[p - 4]] * ha7) + row[xe_c[p - 8]] * ha9
            b_even = ((((z + row[xe_c[p + 11]] * hb0) + row[xe_c[p + 7]] * hb2) + row[xe_c[p + 3]] * hb4) + row[xe_c[p - 1]] * hb6) + row[xe_c[p - 5]] * hb8
            b_odd = ((((z + row[xe_c[p + 9]] * hb1) + row[xe_c[p + 5]] * hb3) + row[xe_c[p + 1]] * hb5) + row[xe_c[p - 3]] * hb7) + row[xe_c[p - 7]] * hb9
            out[2 * j + off1] = a_even + a_odd
            out[2 * j + off2] = b_even + b_odd


@njit(nogil=True, fastmath=False)
def _coldfilt_h(src, dst, ha, hb, pos_sign, xe_c, r_out, groups_c):
    _coldfilt_h_rows(src, dst, ha, hb, pos_sign, xe_c, 0, r_out, groups_c)


@njit(nogil=True, fastmath=False)
def _coldfilt_h_q2c_strip(lo, hi_strip, hp_strip, fa0b, fa0a, fa1b, fa1a,
                          xe_c, y_pair_start, n_pairs, hp_w):
    """Evaluate all 6 orientations at level >= 1, reusing hi_strip sample loads."""
    h0b0, h0b2, h0b4, h0b6, h0b8 = fa0b[0], fa0b[2], fa0b[4], fa0b[6], fa0b[8]
    h0b1, h0b3, h0b5, h0b7, h0b9 = fa0b[1], fa0b[3], fa0b[5], fa0b[7], fa0b[9]
    h0a0, h0a2, h0a4, h0a6, h0a8 = fa0a[0], fa0a[2], fa0a[4], fa0a[6], fa0a[8]
    h0a1, h0a3, h0a5, h0a7, h0a9 = fa0a[1], fa0a[3], fa0a[5], fa0a[7], fa0a[9]

    h1b0, h1b2, h1b4, h1b6, h1b8 = fa1b[0], fa1b[2], fa1b[4], fa1b[6], fa1b[8]
    h1b1, h1b3, h1b5, h1b7, h1b9 = fa1b[1], fa1b[3], fa1b[5], fa1b[7], fa1b[9]
    h1a0, h1a2, h1a4, h1a6, h1a8 = fa1a[0], fa1a[2], fa1a[4], fa1a[6], fa1a[8]
    h1a1, h1a3, h1a5, h1a7, h1a9 = fa1a[1], fa1a[3], fa1a[5], fa1a[7], fa1a[9]

    z = np.float32(0.0)
    s = _SQRT_HALF_32

    for yp in range(n_pairs):
        lo_top = lo[2 * (y_pair_start + yp)]
        lo_bot = lo[2 * (y_pair_start + yp) + 1]
        hi_top = hi_strip[2 * yp]
        hi_bot = hi_strip[2 * yp + 1]
        for qx in range(hp_w):
            p = 10 + 4 * qx
            i0 = xe_c[p + 10]
            i1 = xe_c[p + 8]
            i2 = xe_c[p + 6]
            i3 = xe_c[p + 4]
            i4 = xe_c[p + 2]
            i5 = xe_c[p]
            i6 = xe_c[p - 2]
            i7 = xe_c[p - 4]
            i8 = xe_c[p - 6]
            i9 = xe_c[p - 8]

            j0 = xe_c[p + 11]
            j1 = xe_c[p + 9]
            j2 = xe_c[p + 7]
            j3 = xe_c[p + 5]
            j4 = xe_c[p + 3]
            j5 = xe_c[p + 1]
            j6 = xe_c[p - 1]
            j7 = xe_c[p - 3]
            j8 = xe_c[p - 5]
            j9 = xe_c[p - 7]

            # Load hi_top once and compute both (0, 5) and (1, 4) top components
            e0 = hi_top[i0]; e1 = hi_top[i1]; e2 = hi_top[i2]; e3 = hi_top[i3]; e4 = hi_top[i4]
            e5 = hi_top[i5]; e6 = hi_top[i6]; e7 = hi_top[i7]; e8 = hi_top[i8]; e9 = hi_top[i9]
            o0 = hi_top[j0]; o1 = hi_top[j1]; o2 = hi_top[j2]; o3 = hi_top[j3]; o4 = hi_top[j4]
            o5 = hi_top[j5]; o6 = hi_top[j6]; o7 = hi_top[j7]; o8 = hi_top[j8]; o9 = hi_top[j9]

            ht_a0 = (((((z + e0 * h0b0) + e2 * h0b2) + e4 * h0b4) + e6 * h0b6) + e8 * h0b8) + (((((z + e1 * h0b1) + e3 * h0b3) + e5 * h0b5) + e7 * h0b7) + e9 * h0b9)
            ht_b0 = (((((z + o0 * h0a0) + o2 * h0a2) + o4 * h0a4) + o6 * h0a6) + o8 * h0a8) + (((((z + o1 * h0a1) + o3 * h0a3) + o5 * h0a5) + o7 * h0a7) + o9 * h0a9)
            ht_b1 = (((((z + e0 * h1b0) + e2 * h1b2) + e4 * h1b4) + e6 * h1b6) + e8 * h1b8) + (((((z + e1 * h1b1) + e3 * h1b3) + e5 * h1b5) + e7 * h1b7) + e9 * h1b9)
            ht_a1 = (((((z + o0 * h1a0) + o2 * h1a2) + o4 * h1a4) + o6 * h1a6) + o8 * h1a8) + (((((z + o1 * h1a1) + o3 * h1a3) + o5 * h1a5) + o7 * h1a7) + o9 * h1a9)

            # Load hi_bot once and compute both (0, 5) and (1, 4) bottom components
            e0 = hi_bot[i0]; e1 = hi_bot[i1]; e2 = hi_bot[i2]; e3 = hi_bot[i3]; e4 = hi_bot[i4]
            e5 = hi_bot[i5]; e6 = hi_bot[i6]; e7 = hi_bot[i7]; e8 = hi_bot[i8]; e9 = hi_bot[i9]
            o0 = hi_bot[j0]; o1 = hi_bot[j1]; o2 = hi_bot[j2]; o3 = hi_bot[j3]; o4 = hi_bot[j4]
            o5 = hi_bot[j5]; o6 = hi_bot[j6]; o7 = hi_bot[j7]; o8 = hi_bot[j8]; o9 = hi_bot[j9]

            hb_c0 = (((((z + e0 * h0b0) + e2 * h0b2) + e4 * h0b4) + e6 * h0b6) + e8 * h0b8) + (((((z + e1 * h0b1) + e3 * h0b3) + e5 * h0b5) + e7 * h0b7) + e9 * h0b9)
            hb_d0 = (((((z + o0 * h0a0) + o2 * h0a2) + o4 * h0a4) + o6 * h0a6) + o8 * h0a8) + (((((z + o1 * h0a1) + o3 * h0a3) + o5 * h0a5) + o7 * h0a7) + o9 * h0a9)
            hb_d1 = (((((z + e0 * h1b0) + e2 * h1b2) + e4 * h1b4) + e6 * h1b6) + e8 * h1b8) + (((((z + e1 * h1b1) + e3 * h1b3) + e5 * h1b5) + e7 * h1b7) + e9 * h1b9)
            hb_c1 = (((((z + o0 * h1a0) + o2 * h1a2) + o4 * h1a4) + o6 * h1a6) + o8 * h1a8) + (((((z + o1 * h1a1) + o3 * h1a3) + o5 * h1a5) + o7 * h1a7) + o9 * h1a9)

            a_s = np.float32(ht_a0 * s); b_s = np.float32(ht_b0 * s)
            c_s = np.float32(hb_c0 * s); d_s = np.float32(hb_d0 * s)
            hp_strip[yp, qx, 0] = np.complex64(np.float32(a_s - d_s) + 1j * np.float32(b_s + c_s))
            hp_strip[yp, qx, 5] = np.complex64(np.float32(a_s + d_s) + 1j * np.float32(b_s - c_s))

            a_s = np.float32(ht_a1 * s); b_s = np.float32(ht_b1 * s)
            c_s = np.float32(hb_c1 * s); d_s = np.float32(hb_d1 * s)
            hp_strip[yp, qx, 1] = np.complex64(np.float32(a_s - d_s) + 1j * np.float32(b_s + c_s))
            hp_strip[yp, qx, 4] = np.complex64(np.float32(a_s + d_s) + 1j * np.float32(b_s - c_s))

            # Orientations (2, 3): h1(lo)
            lt_b = (((((z + lo_top[i0] * h1b0) + lo_top[i2] * h1b2) + lo_top[i4] * h1b4) + lo_top[i6] * h1b6) + lo_top[i8] * h1b8) + (((((z + lo_top[i1] * h1b1) + lo_top[i3] * h1b3) + lo_top[i5] * h1b5) + lo_top[i7] * h1b7) + lo_top[i9] * h1b9)
            lt_a = (((((z + lo_top[j0] * h1a0) + lo_top[j2] * h1a2) + lo_top[j4] * h1a4) + lo_top[j6] * h1a6) + lo_top[j8] * h1a8) + (((((z + lo_top[j1] * h1a1) + lo_top[j3] * h1a3) + lo_top[j5] * h1a5) + lo_top[j7] * h1a7) + lo_top[j9] * h1a9)
            lb_d = (((((z + lo_bot[i0] * h1b0) + lo_bot[i2] * h1b2) + lo_bot[i4] * h1b4) + lo_bot[i6] * h1b6) + lo_bot[i8] * h1b8) + (((((z + lo_bot[i1] * h1b1) + lo_bot[i3] * h1b3) + lo_bot[i5] * h1b5) + lo_bot[i7] * h1b7) + lo_bot[i9] * h1b9)
            lb_c = (((((z + lo_bot[j0] * h1a0) + lo_bot[j2] * h1a2) + lo_bot[j4] * h1a4) + lo_bot[j6] * h1a6) + lo_bot[j8] * h1a8) + (((((z + lo_bot[j1] * h1a1) + lo_bot[j3] * h1a3) + lo_bot[j5] * h1a5) + lo_bot[j7] * h1a7) + lo_bot[j9] * h1a9)

            a_s = np.float32(lt_a * s); b_s = np.float32(lt_b * s)
            c_s = np.float32(lb_c * s); d_s = np.float32(lb_d * s)
            hp_strip[yp, qx, 2] = np.complex64(np.float32(a_s - d_s) + 1j * np.float32(b_s + c_s))
            hp_strip[yp, qx, 3] = np.complex64(np.float32(a_s + d_s) + 1j * np.float32(b_s - c_s))


@njit(nogil=True, fastmath=False)
def _levelk_select_sparse_rows(lolo, lo_full, has_lo_full, decisions, frame_idx,
                               out_hp, is_packed, fa0b, fa0a, fa1b, fa1a,
                               xe_r, xe_c, c_raw, lo_2row, hi_2row, qy_start, qy_stop):
    hp_w = decisions.shape[1]
    pos = 0
    for qy in range(qy_start, qy_stop):
        dec_row = decisions[qy]
        first_qx = -1
        last_qx = -1
        for qx in range(hp_w):
            if dec_row[qx] == frame_idx:
                if first_qx < 0:
                    first_qx = qx
                last_qx = qx
        if first_qx < 0:
            continue

        # Exact active column span in [0, c_raw) required by xe_c for first_qx..last_qx.
        if first_qx >= 3 and last_qx < hp_w - 3:
            p_lo = 10 + 4 * first_qx
            p_hi = 10 + 4 * last_qx
            x_lo = c_raw
            x_hi = 0
            for off in range(-8, 12):
                v = xe_c[p_lo + off]
                if v < x_lo:
                    x_lo = v
                if v + 1 > x_hi:
                    x_hi = v + 1
                v2 = xe_c[p_hi + off]
                if v2 < x_lo:
                    x_lo = v2
                if v2 + 1 > x_hi:
                    x_hi = v2 + 1
        else:
            x_lo = 0
            x_hi = c_raw

        _coldfilt_v_rows_span(lolo, hi_2row, fa1b, fa1a, False, xe_r, qy, qy + 1, x_lo, x_hi)
        hi_top = hi_2row[0]
        hi_bot = hi_2row[1]
        if has_lo_full:
            lo_top = lo_full[2 * qy]
            lo_bot = lo_full[2 * qy + 1]
        else:
            _coldfilt_v_rows_span(lolo, lo_2row, fa0b, fa0a, True, xe_r, qy, qy + 1, x_lo, x_hi)
            lo_top = lo_2row[0]
            lo_bot = lo_2row[1]
        for qx in range(first_qx, last_qx + 1):
            if dec_row[qx] != frame_idx:
                continue
            out_y = pos if is_packed else qy
            out_x = 0 if is_packed else qx
            a, b = _coldfilt_h_eval(hi_top, fa0b, fa0a, True, xe_c, qx)
            cc, d = _coldfilt_h_eval(hi_bot, fa0b, fa0a, True, xe_c, qx)
            _store_q2c(out_hp, out_y, out_x, a, b, cc, d, 0, 5)

            a, b = _coldfilt_h_eval(lo_top, fa1b, fa1a, False, xe_c, qx)
            cc, d = _coldfilt_h_eval(lo_bot, fa1b, fa1a, False, xe_c, qx)
            _store_q2c(out_hp, out_y, out_x, a, b, cc, d, 2, 3)

            a, b = _coldfilt_h_eval(hi_top, fa1b, fa1a, False, xe_c, qx)
            cc, d = _coldfilt_h_eval(hi_bot, fa1b, fa1a, False, xe_c, qx)
            _store_q2c(out_hp, out_y, out_x, a, b, cc, d, 1, 4)
            if is_packed:
                pos += 1


@njit(nogil=True, fastmath=False)
def _levelk_select_sparse(lolo, lo_full, has_lo_full, decisions, frame_idx,
                          out_hp, is_packed, fa0b, fa0a, fa1b, fa1a,
                          xe_r, xe_c, c_raw, lo_2row, hi_2row):
    _levelk_select_sparse_rows(
        lolo, lo_full, has_lo_full, decisions, frame_idx,
        out_hp, is_packed, fa0b, fa0a, fa1b, fa1a,
        xe_r, xe_c, c_raw, lo_2row, hi_2row, 0, decisions.shape[0],
    )


@njit(nogil=True, fastmath=False)
def _sum6_strip(abs_strip, mag, y_start):
    n_rows, w, _ = abs_strip.shape
    for ry in range(n_rows):
        dy = y_start + ry
        m_row = mag[dy]
        a_row = abs_strip[ry]
        for x in range(w):
            acc = np.float32(a_row[x, 0] + a_row[x, 1])
            acc = np.float32(acc + a_row[x, 2])
            acc = np.float32(acc + a_row[x, 3])
            acc = np.float32(acc + a_row[x, 4])
            m_row[x] = np.float32(acc + a_row[x, 5])


def _write_activity_strip(hp_strip, scratch, mag, y_start):
    absolute = scratch[:hp_strip.size].reshape(hp_strip.shape)
    np.abs(hp_strip, out=absolute)
    _sum6_strip(absolute, mag, y_start)


@njit(nogil=True, fastmath=False)
def _max3x3_update_rows(mag, best_act, best_idx, frame_idx, is_first, y_start, y_stop):
    h, w = mag.shape
    for y in range(y_start, y_stop):
        ym = 0 if y == 0 else y - 1
        yp = h - 1 if y == h - 1 else y + 1
        rm = mag[ym]
        r_cur = mag[y]
        rp = mag[yp]
        b_row = best_act[y]
        i_row = best_idx[y]
        for x in range(w):
            xm = 0 if x == 0 else x - 1
            xp = w - 1 if x == w - 1 else x + 1
            m = rm[xm]
            v = rm[x]
            if v > m:
                m = v
            v = rm[xp]
            if v > m:
                m = v
            v = r_cur[xm]
            if v > m:
                m = v
            v = r_cur[x]
            if v > m:
                m = v
            v = r_cur[xp]
            if v > m:
                m = v
            v = rp[xm]
            if v > m:
                m = v
            v = rp[x]
            if v > m:
                m = v
            v = rp[xp]
            if v > m:
                m = v
            if is_first:
                b_row[x] = m
                i_row[x] = 0
            elif m > b_row[x]:
                b_row[x] = m
                i_row[x] = frame_idx


@njit(nogil=True, fastmath=False)
def _max3x3_update(mag, best_act, best_idx, frame_idx, is_first):
    _max3x3_update_rows(mag, best_act, best_idx, frame_idx, is_first, 0, mag.shape[0])


@njit(nogil=True, fastmath=False)
def _scan_count_decisions(decisions, counts_row):
    h, w = decisions.shape
    n = counts_row.shape[0]
    for y in range(h):
        row = decisions[y]
        for x in range(w):
            idx = row[x]
            if 0 <= idx < n:
                counts_row[idx] += 1


@njit(nogil=True, fastmath=False)
def _scatter_hp(decisions, frame_idx, packed, fused_hp):
    hp_h, hp_w = decisions.shape
    pos = 0
    for qy in range(hp_h):
        row = decisions[qy]
        for qx in range(hp_w):
            if row[qx] == frame_idx:
                for o in range(6):
                    fused_hp[qy, qx, o] = packed[pos, 0, o]
                pos += 1


@njit(nogil=True, fastmath=False)
def _acc_weighted_lp(lp, weight, src):
    h, w = lp.shape
    for y in range(h):
        l_row = lp[y]
        w_row = weight[y]
        s_row = src[y]
        for x in range(w):
            l_row[x] = l_row[x] + w_row[x] * s_row[x]


@njit(nogil=True, fastmath=False)
def _c2q_pair_rows(hp, quad, c0, c1, scale, y_start, y_stop):
    w2 = hp.shape[1]
    for y in range(y_start, y_stop):
        qy0 = 2 * y
        qy1 = qy0 + 1
        q0 = quad[qy0]
        q1 = quad[qy1]
        hp_row = hp[y]
        for x in range(w2):
            z0 = hp_row[x, c0]
            z1 = hp_row[x, c1]
            w0r = z0.real * scale
            w0i = z0.imag * scale
            w1r = z1.real * scale
            w1i = z1.imag * scale
            qx0 = 2 * x
            qx1 = qx0 + 1
            q0[qx0] = np.float32(w0r + w1r)
            q0[qx1] = np.float32(w0i + w1i)
            q1[qx0] = np.float32(w0i - w1i)
            q1[qx1] = np.float32(-(w0r - w1r))


@njit(nogil=True, fastmath=False)
def _c2q_pair(hp, quad, c0, c1, scale):
    _c2q_pair_rows(hp, quad, c0, c1, scale, 0, hp.shape[0])


@njit(nogil=True, fastmath=False)
def _any_nonzero_2d(arr, r, c):
    z = np.float32(0.0)
    for y in range(r):
        row = arr[y]
        for x in range(c):
            if row[x] != z:
                return True
    return False


@njit(nogil=True, fastmath=False)
def _colifilt_v_add(src0, src1, dst, f0b, f0a, f1b, f1a, xe_r, r, c):
    z0 = not _any_nonzero_2d(src0, r, c)
    z1 = not _any_nonzero_2d(src1, r, c)
    if z0 and z1:
        dst[:2 * r, :c] = np.float32(0.0)
        return
    groups = r >> 1
    zero32 = np.float32(0.0)
    for j in range(groups):
        a0 = xe_r[2 * j + 9]
        a1 = xe_r[2 * j + 7]
        a2 = xe_r[2 * j + 5]
        a3 = xe_r[2 * j + 3]
        a4 = xe_r[2 * j + 1]
        b0 = xe_r[2 * j + 10]
        b1 = xe_r[2 * j + 8]
        b2 = xe_r[2 * j + 6]
        b3 = xe_r[2 * j + 4]
        b4 = xe_r[2 * j + 2]

        y0 = 4 * j
        for x in range(c):
            if z0:
                vy0 = vy1 = vy2 = vy3 = zero32
            else:
                s0_0 = src0[a0, x]
                s0_1 = src0[a1, x]
                s0_2 = src0[a2, x]
                s0_3 = src0[a3, x]
                s0_4 = src0[a4, x]
                t0_0 = src0[b0, x]
                t0_1 = src0[b1, x]
                t0_2 = src0[b2, x]
                t0_3 = src0[b3, x]
                t0_4 = src0[b4, x]
                vy0 = ((((zero32 + s0_0 * f0b[0]) + s0_1 * f0b[2]) + s0_2 * f0b[4]) + s0_3 * f0b[6]) + s0_4 * f0b[8]
                vy1 = ((((zero32 + t0_0 * f0a[0]) + t0_1 * f0a[2]) + t0_2 * f0a[4]) + t0_3 * f0a[6]) + t0_4 * f0a[8]
                vy2 = ((((zero32 + s0_0 * f0b[1]) + s0_1 * f0b[3]) + s0_2 * f0b[5]) + s0_3 * f0b[7]) + s0_4 * f0b[9]
                vy3 = ((((zero32 + t0_0 * f0a[1]) + t0_1 * f0a[3]) + t0_2 * f0a[5]) + t0_3 * f0a[7]) + t0_4 * f0a[9]

            if z1:
                hy0 = hy1 = hy2 = hy3 = zero32
            else:
                s1_0 = src1[b0, x]
                s1_1 = src1[b1, x]
                s1_2 = src1[b2, x]
                s1_3 = src1[b3, x]
                s1_4 = src1[b4, x]
                t1_0 = src1[a0, x]
                t1_1 = src1[a1, x]
                t1_2 = src1[a2, x]
                t1_3 = src1[a3, x]
                t1_4 = src1[a4, x]
                hy0 = ((((zero32 + s1_0 * f1b[0]) + s1_1 * f1b[2]) + s1_2 * f1b[4]) + s1_3 * f1b[6]) + s1_4 * f1b[8]
                hy1 = ((((zero32 + t1_0 * f1a[0]) + t1_1 * f1a[2]) + t1_2 * f1a[4]) + t1_3 * f1a[6]) + t1_4 * f1a[8]
                hy2 = ((((zero32 + s1_0 * f1b[1]) + s1_1 * f1b[3]) + s1_2 * f1b[5]) + s1_3 * f1b[7]) + s1_4 * f1b[9]
                hy3 = ((((zero32 + t1_0 * f1a[1]) + t1_1 * f1a[3]) + t1_2 * f1a[5]) + t1_3 * f1a[7]) + t1_4 * f1a[9]

            dst[y0, x] = vy0 + hy0
            dst[y0 + 1, x] = vy1 + hy1
            dst[y0 + 2, x] = vy2 + hy2
            dst[y0 + 3, x] = vy3 + hy3


@njit(nogil=True, fastmath=False)
def _colifilt_h_add_trim(src0, src1, dst, f0b, f0a, f1b, f1a, xe_c, r, c,
                         off_y, off_x, exp_r, exp_c):
    z0 = not _any_nonzero_2d(src0, r, c)
    z1 = not _any_nonzero_2d(src1, r, c)
    if z0 and z1:
        dst[:exp_r, :exp_c] = np.float32(0.0)
        return
    groups = c >> 1
    zero32 = np.float32(0.0)
    for y in range(off_y, off_y + exp_r):
        r0 = src0[y]
        r1 = src1[y]
        out = dst[y - off_y]
        for j in range(groups):
            a0 = xe_c[2 * j + 9]
            a1 = xe_c[2 * j + 7]
            a2 = xe_c[2 * j + 5]
            a3 = xe_c[2 * j + 3]
            a4 = xe_c[2 * j + 1]
            b0 = xe_c[2 * j + 10]
            b1 = xe_c[2 * j + 8]
            b2 = xe_c[2 * j + 6]
            b3 = xe_c[2 * j + 4]
            b4 = xe_c[2 * j + 2]

            if z0:
                vx0 = vx1 = vx2 = vx3 = zero32
            else:
                vx0 = ((((zero32 + r0[a0] * f0b[0]) + r0[a1] * f0b[2]) + r0[a2] * f0b[4]) + r0[a3] * f0b[6]) + r0[a4] * f0b[8]
                vx1 = ((((zero32 + r0[b0] * f0a[0]) + r0[b1] * f0a[2]) + r0[b2] * f0a[4]) + r0[b3] * f0a[6]) + r0[b4] * f0a[8]
                vx2 = ((((zero32 + r0[a0] * f0b[1]) + r0[a1] * f0b[3]) + r0[a2] * f0b[5]) + r0[a3] * f0b[7]) + r0[a4] * f0b[9]
                vx3 = ((((zero32 + r0[b0] * f0a[1]) + r0[b1] * f0a[3]) + r0[b2] * f0a[5]) + r0[b3] * f0a[7]) + r0[b4] * f0a[9]

            if z1:
                hx0 = hx1 = hx2 = hx3 = zero32
            else:
                hx0 = ((((zero32 + r1[b0] * f1b[0]) + r1[b1] * f1b[2]) + r1[b2] * f1b[4]) + r1[b3] * f1b[6]) + r1[b4] * f1b[8]
                hx1 = ((((zero32 + r1[a0] * f1a[0]) + r1[a1] * f1a[2]) + r1[a2] * f1a[4]) + r1[a3] * f1a[6]) + r1[a4] * f1a[8]
                hx2 = ((((zero32 + r1[b0] * f1b[1]) + r1[b1] * f1b[3]) + r1[b2] * f1b[5]) + r1[b3] * f1b[7]) + r1[b4] * f1b[9]
                hx3 = ((((zero32 + r1[a0] * f1a[1]) + r1[a1] * f1a[3]) + r1[a2] * f1a[5]) + r1[a3] * f1a[7]) + r1[a4] * f1a[9]

            x0 = 4 * j
            if off_x <= x0 < off_x + exp_c:
                out[x0 - off_x] = vx0 + hx0
            if off_x <= x0 + 1 < off_x + exp_c:
                out[x0 + 1 - off_x] = vx1 + hx1
            if off_x <= x0 + 2 < off_x + exp_c:
                out[x0 + 2 - off_x] = vx2 + hx2
            if off_x <= x0 + 3 < off_x + exp_c:
                out[x0 + 3 - off_x] = vx3 + hx3


@njit(nogil=True, fastmath=False)
def _colfilter_v_add0_rows(src7, src5, dst, g0, g1, r, c, y_start, y_stop):
    k0, k1, k2, k3, k4, k5, k6 = (
        g0[0], g0[1], g0[2], g0[3], g0[4], g0[5], g0[6]
    )
    m0, m1, m2, m3, m4 = g1[0], g1[1], g1[2], g1[3], g1[4]
    z = np.float32(0.0)
    head_stop = min(y_stop, 3) if r >= 6 else y_stop
    int_start = max(y_start, head_stop)
    int_stop = max(int_start, min(y_stop, r - 3)) if r >= 6 else int_start

    for y in range(y_start, min(y_stop, head_stop)):
        yp3 = _ref_idx(y + 3, r)
        yp2 = _ref_idx(y + 2, r)
        yp1 = _ref_idx(y + 1, r)
        ym1 = _ref_idx(y - 1, r)
        ym2 = _ref_idx(y - 2, r)
        ym3 = _ref_idx(y - 3, r)
        s7p3 = src7[yp3]; s7p2 = src7[yp2]; s7p1 = src7[yp1]; s70 = src7[y]
        s7m1 = src7[ym1]; s7m2 = src7[ym2]; s7m3 = src7[ym3]
        s5p2 = src5[yp2]; s5p1 = src5[yp1]; s50 = src5[y]; s5m1 = src5[ym1]; s5m2 = src5[ym2]
        out = dst[y]
        for x in range(c):
            v7 = (((((((z + s7p3[x] * k0) + s7p2[x] * k1) + s7p1[x] * k2) + s70[x] * k3) + s7m1[x] * k4) + s7m2[x] * k5) + s7m3[x] * k6)
            v5 = (((((z + s5p2[x] * m0) + s5p1[x] * m1) + s50[x] * m2) + s5m1[x] * m3) + s5m2[x] * m4)
            out[x] = v7 + v5

    for y in range(int_start, int_stop):
        s7p3 = src7[y + 3]; s7p2 = src7[y + 2]; s7p1 = src7[y + 1]; s70 = src7[y]
        s7m1 = src7[y - 1]; s7m2 = src7[y - 2]; s7m3 = src7[y - 3]
        s5p2 = src5[y + 2]; s5p1 = src5[y + 1]; s50 = src5[y]; s5m1 = src5[y - 1]; s5m2 = src5[y - 2]
        out = dst[y]
        for x in range(c):
            v7 = (((((((z + s7p3[x] * k0) + s7p2[x] * k1) + s7p1[x] * k2) + s70[x] * k3) + s7m1[x] * k4) + s7m2[x] * k5) + s7m3[x] * k6)
            v5 = (((((z + s5p2[x] * m0) + s5p1[x] * m1) + s50[x] * m2) + s5m1[x] * m3) + s5m2[x] * m4)
            out[x] = v7 + v5

    for y in range(int_stop, y_stop):
        yp3 = _ref_idx(y + 3, r)
        yp2 = _ref_idx(y + 2, r)
        yp1 = _ref_idx(y + 1, r)
        ym1 = _ref_idx(y - 1, r)
        ym2 = _ref_idx(y - 2, r)
        ym3 = _ref_idx(y - 3, r)
        s7p3 = src7[yp3]; s7p2 = src7[yp2]; s7p1 = src7[yp1]; s70 = src7[y]
        s7m1 = src7[ym1]; s7m2 = src7[ym2]; s7m3 = src7[ym3]
        s5p2 = src5[yp2]; s5p1 = src5[yp1]; s50 = src5[y]; s5m1 = src5[ym1]; s5m2 = src5[ym2]
        out = dst[y]
        for x in range(c):
            v7 = (((((((z + s7p3[x] * k0) + s7p2[x] * k1) + s7p1[x] * k2) + s70[x] * k3) + s7m1[x] * k4) + s7m2[x] * k5) + s7m3[x] * k6)
            v5 = (((((z + s5p2[x] * m0) + s5p1[x] * m1) + s50[x] * m2) + s5m1[x] * m3) + s5m2[x] * m4)
            out[x] = v7 + v5


@njit(nogil=True, fastmath=False)
def _colfilter_v_add0(src7, src5, dst, g0, g1, r, c):
    _colfilter_v_add0_rows(src7, src5, dst, g0, g1, r, c, 0, r)


@njit(nogil=True, fastmath=False)
def _colfilter_h_add0_rows(src7, src5, dst, g0, g1, c, y_start, y_stop):
    k0, k1, k2, k3, k4, k5, k6 = (
        g0[0], g0[1], g0[2], g0[3], g0[4], g0[5], g0[6]
    )
    m0, m1, m2, m3, m4 = g1[0], g1[1], g1[2], g1[3], g1[4]
    z = np.float32(0.0)
    x_lo = 3 if c >= 6 else c
    x_hi = max(x_lo, c - 3)
    for y in range(y_start, y_stop):
        r7 = src7[y]
        r5 = src5[y]
        out = dst[y]
        for x in range(x_lo):
            v7 = (((((((z + r7[_ref_idx(x + 3, c)] * k0) + r7[_ref_idx(x + 2, c)] * k1) + r7[_ref_idx(x + 1, c)] * k2) + r7[x] * k3) + r7[_ref_idx(x - 1, c)] * k4) + r7[_ref_idx(x - 2, c)] * k5) + r7[_ref_idx(x - 3, c)] * k6)
            v5 = (((((z + r5[_ref_idx(x + 2, c)] * m0) + r5[_ref_idx(x + 1, c)] * m1) + r5[x] * m2) + r5[_ref_idx(x - 1, c)] * m3) + r5[_ref_idx(x - 2, c)] * m4)
            out[x] = v7 + v5
        for x in range(x_lo, x_hi):
            v7 = (((((((z + r7[x + 3] * k0) + r7[x + 2] * k1) + r7[x + 1] * k2) + r7[x] * k3) + r7[x - 1] * k4) + r7[x - 2] * k5) + r7[x - 3] * k6)
            v5 = (((((z + r5[x + 2] * m0) + r5[x + 1] * m1) + r5[x] * m2) + r5[x - 1] * m3) + r5[x - 2] * m4)
            out[x] = v7 + v5
        for x in range(x_hi, c):
            v7 = (((((((z + r7[_ref_idx(x + 3, c)] * k0) + r7[_ref_idx(x + 2, c)] * k1) + r7[_ref_idx(x + 1, c)] * k2) + r7[x] * k3) + r7[_ref_idx(x - 1, c)] * k4) + r7[_ref_idx(x - 2, c)] * k5) + r7[_ref_idx(x - 3, c)] * k6)
            v5 = (((((z + r5[_ref_idx(x + 2, c)] * m0) + r5[_ref_idx(x + 1, c)] * m1) + r5[x] * m2) + r5[_ref_idx(x - 1, c)] * m3) + r5[_ref_idx(x - 2, c)] * m4)
            out[x] = v7 + v5


@njit(nogil=True, fastmath=False)
def _colfilter_h_add0(src7, src5, dst, g0, g1, r, c):
    _colfilter_h_add0_rows(src7, src5, dst, g0, g1, c, 0, r)


def _run_bands_pool(pool, n_workers: int, total: int, fn):
    """Dispatch ``fn(band_idx, start, stop)`` across disjoint row bands."""
    if total <= 0:
        return
    if pool is None or n_workers <= 1 or total < 2:
        fn(0, 0, total)
        return
    n_bands = min(n_workers, total)
    chunk = (total + n_bands - 1) // n_bands
    futs = []
    b = 0
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        futs.append(pool.submit(fn, b, start, stop))
        b += 1
    for f in futs:
        f.result()


class _LevelPlan:
    __slots__ = (
        "in_r", "in_c", "pad4", "pad_r", "pad_c",
        "lolo_r", "lolo_c", "hp_h", "hp_w",
        "xe_r_fwd", "xe_c_fwd", "xe_r_inv", "xe_c_inv",
    )

    def __init__(self, in_r, in_c, pad4, pad_r, pad_c, lolo_r, lolo_c,
                 xe_r_fwd, xe_c_fwd, xe_r_inv, xe_c_inv):
        self.in_r = in_r
        self.in_c = in_c
        self.pad4 = pad4
        self.pad_r = pad_r
        self.pad_c = pad_c
        self.lolo_r = lolo_r
        self.lolo_c = lolo_c
        self.hp_h = lolo_r >> 1
        self.hp_w = lolo_c >> 1
        self.xe_r_fwd = xe_r_fwd
        self.xe_c_fwd = xe_c_fwd
        self.xe_r_inv = xe_r_inv
        self.xe_c_inv = xe_c_inv


class _NativeWorkspace:
    """Reusable bounded float32/complex64 scratch owned by a single thread."""

    def __init__(self, engine, orig_h: int, orig_w: int, nlevels: int,
                 strip_pairs: int = 32, for_pass2: bool = False):
        self.plans = engine.build_level_plans(orig_h, orig_w, nlevels)
        sp0 = self.plans[0]
        r0, c0 = sp0.in_r, sp0.in_c
        self.buf_a = np.empty((r0, c0), dtype=np.float32)
        self.buf_b = np.empty((r0, c0), dtype=np.float32)
        if for_pass2:
            self.strip_pairs = 0
            self.lo_strip = None
            self.hi_strip = None
            self.hp_strip = None
            self.abs_strip = None
            self.lo_2row = np.empty((2, c0), dtype=np.float32)
            self.hi_2row = np.empty((2, c0), dtype=np.float32)
        else:
            self.strip_pairs = max(1, min(strip_pairs, max(1, sp0.hp_h)))
            self.lo_strip = np.empty((2 * self.strip_pairs, c0), dtype=np.float32)
            self.hi_strip = np.empty((2 * self.strip_pairs, c0), dtype=np.float32)
            self.hp_strip = np.empty((self.strip_pairs, sp0.hp_w, 6), dtype=np.complex64)
            self.abs_strip = _magnitude_scratch(self.hp_strip.size, np.float32)
            self.lo_2row = None
            self.hi_2row = None

    def ensure_pass2(self):
        if self.lo_2row is None:
            c0 = self.plans[0].in_c
            self.lo_2row = np.empty((2, c0), dtype=np.float32)
            self.hi_2row = np.empty((2, c0), dtype=np.float32)


class _NativeDtcwtEngine:
    """Disk-cached native CPU DTCWT engine with reusable level plans."""

    max3x3_update = staticmethod(_max3x3_update)
    scan_count_decisions = staticmethod(_scan_count_decisions)
    scatter_hp = staticmethod(_scatter_hp)
    acc_weighted_lp = staticmethod(_acc_weighted_lp)

    def __init__(self, h0o, g0o, h1o, g1o, h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b):
        self.h0o = h0o
        self.g0o = g0o
        self.h1o = h1o
        self.g1o = g1o
        self.h0a = h0a
        self.h0b = h0b
        self.g0a = g0a
        self.g0b = g0b
        self.h1a = h1a
        self.h1b = h1b
        self.g1a = g1a
        self.g1b = g1b
        scale = np.sqrt(0.5)
        product_dtype = np.result_type(np.empty(0, dtype=np.complex64), scale)
        self._inverse_scale = product_dtype.type(scale).real
        self._warmup_lock = Lock()
        self._warmed_keys = set()

    def build_level_plans(self, orig_h: int, orig_w: int, nlevels: int):
        r0 = orig_h + (orig_h & 1)
        c0 = orig_w + (orig_w & 1)
        plans = [
            _LevelPlan(
                in_r=r0, in_c=c0, pad4=False, pad_r=r0, pad_c=c0,
                lolo_r=r0, lolo_c=c0,
                xe_r_fwd=None, xe_c_fwd=None, xe_r_inv=None, xe_c_inv=None,
            )
        ]
        prev_r, prev_c = r0, c0
        for _ in range(1, nlevels):
            pad_r = prev_r + (2 if prev_r % 4 != 0 else 0)
            pad_c = prev_c + (2 if prev_c % 4 != 0 else 0)
            lolo_r = pad_r >> 1
            lolo_c = pad_c >> 1
            plans.append(
                _LevelPlan(
                    in_r=prev_r, in_c=prev_c, pad4=True,
                    pad_r=pad_r, pad_c=pad_c,
                    lolo_r=lolo_r, lolo_c=lolo_c,
                    xe_r_fwd=make_reflect_table(prev_r, 10, pad4=True),
                    xe_c_fwd=make_reflect_table(prev_c, 10, pad4=True),
                    xe_r_inv=make_reflect_table(lolo_r, 5, pad4=False),
                    xe_c_inv=make_reflect_table(lolo_c, 5, pad4=False),
                )
            )
            prev_r, prev_c = lolo_r, lolo_c
        return plans

    def extract_luma(self, image: np.ndarray, dst: np.ndarray) -> bool:
        """Extract BGR/grayscale image directly into padded float32 luma buffer."""
        h, w = image.shape[:2]
        dtype = image.dtype
        if image.ndim == 2:
            if dtype == np.uint8:
                _extract_luma_u8_2(image, dst, h, w, _LUT_GRAY)
                return False
            if dtype == np.uint16:
                _extract_luma_u16_2(image, dst, h, w)
                return False
            return bool(_extract_luma_f32_2(np.asarray(image, dtype=np.float32), dst, h, w))
        ch = image.shape[2]
        if ch == 1:
            plane = image[:, :, 0]
            if dtype == np.uint8:
                _extract_luma_u8_2(plane, dst, h, w, _LUT_GRAY)
                return False
            if dtype == np.uint16:
                _extract_luma_u16_2(plane, dst, h, w)
                return False
            return bool(_extract_luma_f32_2(np.asarray(plane, dtype=np.float32), dst, h, w))
        if ch not in (3, 4):
            raise ValueError(f"Unsupported image shape: {image.shape}")
        if dtype == np.uint8:
            _extract_luma_u8_3(image, dst, h, w, _LUT_R, _LUT_G, _LUT_B)
            return False
        if dtype == np.uint16:
            _extract_luma_u16_3(image, dst, h, w)
            return False
        return bool(_extract_luma_f32_3(np.asarray(image, dtype=np.float32), dst, h, w))

    def extract_channel(self, image: np.ndarray, channel: int, dst: np.ndarray) -> bool:
        """Extract one RGB-ordered channel (``0=R, 1=G, 2=B``) directly into ``dst``."""
        h, w = image.shape[:2]
        dtype = image.dtype
        if image.ndim == 2:
            if dtype == np.uint8:
                _extract_chan_u8_2(image, dst, h, w, _LUT_U8)
                return False
            if dtype == np.uint16:
                _extract_chan_u16_2(image, dst, h, w)
                return False
            return bool(_extract_chan_f32_2(np.asarray(image, dtype=np.float32), dst, h, w))
        if image.ndim == 3 and image.shape[2] == 1:
            plane = image[:, :, 0]
            if dtype == np.uint8:
                _extract_chan_u8_2(plane, dst, h, w, _LUT_U8)
                return False
            if dtype == np.uint16:
                _extract_chan_u16_2(plane, dst, h, w)
                return False
            return bool(_extract_chan_f32_2(np.asarray(plane, dtype=np.float32), dst, h, w))
        if image.ndim == 3 and image.shape[2] in (3, 4):
            src_ch = 2 - channel
            if dtype == np.uint8:
                _extract_chan_u8_3(image, dst, h, w, src_ch, _LUT_U8)
                return False
            if dtype == np.uint16:
                _extract_chan_u16_3(image, dst, h, w, src_ch)
                return False
            return bool(_extract_chan_f32_3(np.asarray(image, dtype=np.float32), dst, h, w, src_ch))
        raise ValueError(f"Unsupported image shape: {image.shape}")

    def forward_activity_into(self, ws: _NativeWorkspace, nlevels: int):
        """Run level-0..nlevels-1 activity magnitude extraction from ``ws.buf_a``."""
        plans = ws.plans
        mags = []
        sp0 = plans[0]
        r, c = sp0.in_r, sp0.in_c
        hp0_h, hp0_w = sp0.hp_h, sp0.hp_w
        mag0 = np.empty((hp0_h, hp0_w), dtype=np.float32)
        strip_pairs = ws.strip_pairs
        src0 = ws.buf_a[:r, :c]
        lolo_out = ws.buf_b[:r, :c] if nlevels > 1 else None

        # Fused level-0 strip pass keeps lo_strip and hi_strip hot in L2 cache.
        for yp in range(0, hp0_h, strip_pairs):
            n_pairs = min(strip_pairs, hp0_h - yp)
            lo_strip = ws.lo_strip[:2 * n_pairs, :c]
            hi_strip = ws.hi_strip[:2 * n_pairs, :c]
            _colfilter_v5_v7_strip(
                src0, lo_strip, hi_strip, self.h0o, self.h1o, r, c,
                2 * yp, 2 * (yp + n_pairs),
            )
            hp_strip = ws.hp_strip[:n_pairs, :hp0_w]
            _colfilter_h_q2c_strip_local(
                lo_strip, hi_strip, hp_strip, self.h0o, self.h1o, n_pairs, c,
            )
            _write_activity_strip(hp_strip, ws.abs_strip, mag0, yp)
            if lolo_out is not None:
                _colfilter_h5_rows(lo_strip, lolo_out, self.h0o, 2 * yp, 2 * n_pairs, c)
        mags.append(mag0)

        if nlevels == 1:
            return mags

        lolo = lolo_out
        for lvl in range(1, nlevels):
            sp = plans[lvl]
            lo_r = sp.lolo_r
            lo = ws.buf_a[:lo_r, :sp.in_c]
            _coldfilt_v_rows(lolo, lo, self.h0b, self.h0a, True, sp.xe_r_fwd, 0, lo_r >> 1, sp.in_c)

            mag_lvl = np.empty((sp.hp_h, sp.hp_w), dtype=np.float32)
            for yp in range(0, sp.hp_h, strip_pairs):
                n_pairs = min(strip_pairs, sp.hp_h - yp)
                hi_strip = ws.hi_strip[:2 * n_pairs, :sp.in_c]
                _coldfilt_v_rows(lolo, hi_strip, self.h1b, self.h1a, False, sp.xe_r_fwd, yp, yp + n_pairs, sp.in_c)
                hp_strip = ws.hp_strip[:n_pairs, :sp.hp_w]
                _coldfilt_h_q2c_strip(
                    lo, hi_strip, hp_strip, self.h0b, self.h0a, self.h1b, self.h1a,
                    sp.xe_c_fwd, yp, n_pairs, sp.hp_w,
                )
                _write_activity_strip(hp_strip, ws.abs_strip, mag_lvl, yp)
            mags.append(mag_lvl)

            if lvl + 1 < nlevels:
                next_lolo = ws.buf_b[:lo_r, :sp.lolo_c]
                _coldfilt_h(lo, next_lolo, self.h0b, self.h0a, True, sp.xe_c_fwd, lo_r, sp.lolo_c >> 1)
                lolo = next_lolo

        return mags

    def forward_select_into(self, ws: _NativeWorkspace, nlevels: int, max_level: int,
                            need_lp: bool, active_row, decisions, frame_idx: int,
                            targets: list, is_packed: bool = False):
        """Extract winning highpass coefficients & optional final lowpass from ``ws.buf_a``."""
        ws.ensure_pass2()
        plans = ws.plans
        sp0 = plans[0]
        r, c = sp0.in_r, sp0.in_c
        descend_0 = need_lp or max_level > 1
        lo0 = ws.buf_b[:r, :c]
        if descend_0:
            _colfilter_v5(ws.buf_a[:r, :c], lo0, self.h0o, r, c)

        if active_row[0]:
            _level0_select_sparse(
                ws.buf_a[:r, :c], lo0, bool(descend_0), decisions[0], frame_idx,
                targets[0], is_packed, self.h0o, self.h1o, r, c,
                ws.lo_2row[:2, :c], ws.hi_2row[:2, :c],
            )

        if not descend_0:
            return None

        lolo = ws.buf_a[:r, :c]
        _colfilter_h5(lo0, lolo, self.h0o, r, c)

        for lvl in range(1, nlevels):
            descend_k = need_lp or (lvl + 1 < max_level)
            if not active_row[lvl] and not descend_k:
                break
            sp = plans[lvl]
            lo_r = sp.lolo_r
            lo = ws.buf_b[:lo_r, :sp.in_c]
            if descend_k:
                _coldfilt_v_rows(
                    lolo, lo, self.h0b, self.h0a, True, sp.xe_r_fwd, 0, lo_r >> 1, sp.in_c,
                )

            if active_row[lvl]:
                _levelk_select_sparse(
                    lolo, lo, bool(descend_k), decisions[lvl], frame_idx,
                    targets[lvl], is_packed, self.h0b, self.h0a, self.h1b, self.h1a,
                    sp.xe_r_fwd, sp.xe_c_fwd, sp.in_c,
                    ws.lo_2row[:2, :sp.in_c], ws.hi_2row[:2, :sp.in_c],
                )

            if not descend_k:
                return None

            next_lolo = ws.buf_a[:lo_r, :sp.lolo_c]
            _coldfilt_h(lo, next_lolo, self.h0b, self.h0a, True, sp.xe_c_fwd, lo_r, sp.lolo_c >> 1)
            lolo = next_lolo

        return lolo.copy() if need_lp else None

    def inverse(self, lp: np.ndarray, fused_high, plans) -> np.ndarray:
        """Exact native inverse DTCWT from ``lp`` and ``fused_high``."""
        nlevels = len(fused_high)
        if nlevels == 0:
            return lp.astype(np.float32, copy=False)

        sp0 = plans[0]
        buf_r = max(sp0.in_r, max((2 * p.lolo_r for p in plans[1:]), default=0))
        buf_c = max(sp0.in_c, max((2 * p.lolo_c for p in plans[1:]), default=0))
        z = np.empty((buf_r, buf_c), dtype=np.float32)
        cur_h, cur_w = lp.shape
        z[:cur_h, :cur_w] = lp

        y1 = np.empty((buf_r, buf_c), dtype=np.float32)
        y2 = np.empty((buf_r, buf_c), dtype=np.float32)

        for lvl in range(nlevels - 1, 0, -1):
            sp = plans[lvl]
            hp = fused_high[lvl]
            in_r, in_c = sp.lolo_r, sp.lolo_c
            out_r = 2 * in_r
            prev_hp = fused_high[lvl - 1]
            exp_r, exp_c = 2 * prev_hp.shape[0], 2 * prev_hp.shape[1]
            off_y = 1 if out_r != exp_r else 0
            off_x = 1 if (2 * in_c) != exp_c else 0

            quad = y2[:in_r, :in_c]
            _c2q_pair(hp, quad, 0, 5, self._inverse_scale)
            _colifilt_v_add(z[:in_r, :in_c], quad, y1[:out_r, :in_c],
                            self.g0b, self.g0a, self.g1b, self.g1a, sp.xe_r_inv, in_r, in_c)

            q_hl = np.empty((in_r, in_c), dtype=np.float32)
            q_hh = z[:in_r, :in_c]
            _c2q_pair(hp, q_hl, 2, 3, self._inverse_scale)
            _c2q_pair(hp, q_hh, 1, 4, self._inverse_scale)
            _colifilt_v_add(q_hl, q_hh, y2[:out_r, :in_c],
                            self.g0b, self.g0a, self.g1b, self.g1a, sp.xe_r_inv, in_r, in_c)
            del q_hl

            _colifilt_h_add_trim(
                y1[:out_r, :in_c], y2[:out_r, :in_c], z[:exp_r, :exp_c],
                self.g0b, self.g0a, self.g1b, self.g1a, sp.xe_c_inv,
                out_r, in_c, off_y, off_x, exp_r, exp_c,
            )

        r, c = sp0.in_r, sp0.in_c
        hp0 = fused_high[0]
        quad = y2[:r, :c]
        _c2q_pair(hp0, quad, 0, 5, self._inverse_scale)
        _colfilter_v_add0(z[:r, :c], quad, y1[:r, :c], self.g0o, self.g1o, r, c)

        q_hl = np.empty((r, c), dtype=np.float32)
        q_hh = z[:r, :c]
        _c2q_pair(hp0, q_hl, 2, 3, self._inverse_scale)
        _c2q_pair(hp0, q_hh, 1, 4, self._inverse_scale)
        _colfilter_v_add0(q_hl, q_hh, y2[:r, :c], self.g0o, self.g1o, r, c)
        del q_hl

        _colfilter_h_add0(y1[:r, :c], y2[:r, :c], z[:r, :c], self.g0o, self.g1o, r, c)
        del y1, y2
        return z[:r, :c]

    def make_workspace(self, orig_h: int, orig_w: int, nlevels: int,
                       strip_pairs: int = 32, for_pass2: bool = False):
        return _NativeWorkspace(
            self, orig_h, orig_w, nlevels, strip_pairs=strip_pairs, for_pass2=for_pass2,
        )

    def make_pass1_scratches(self, orig_h: int, orig_w: int, nlevels: int,
                             n_workers: int, strip_pairs: int = 16):
        sp0 = self.build_level_plans(orig_h, orig_w, nlevels)[0]
        sp = max(1, min(strip_pairs, max(1, sp0.hp_h)))
        c0 = sp0.in_c
        hp0_w = sp0.hp_w
        scratches = []
        for _ in range(max(1, n_workers)):
            lo_s = np.empty((2 * sp, c0), dtype=np.float32)
            hi_s = np.empty((2 * sp, c0), dtype=np.float32)
            hp_s = np.empty((sp, hp0_w, 6), dtype=np.complex64)
            abs_s = np.empty(hp_s.size, dtype=np.float32)
            scratches.append((lo_s, hi_s, hp_s, abs_s))
        return scratches

    def make_pass2_scratches(self, orig_h: int, orig_w: int, nlevels: int,
                             n_workers: int):
        sp0 = self.build_level_plans(orig_h, orig_w, nlevels)[0]
        c0 = sp0.in_c
        return [
            (np.empty((2, c0), dtype=np.float32), np.empty((2, c0), dtype=np.float32))
            for _ in range(max(1, n_workers))
        ]

    def extract_luma_parallel(self, image: np.ndarray, dst: np.ndarray,
                              pool, row_workers: int) -> bool:
        if pool is None or row_workers <= 1 or dst.shape[0] < 64 or image.dtype != np.uint8:
            return self.extract_luma(image, dst)
        h, w = image.shape[:2]
        r0 = dst.shape[0]
        if image.ndim == 2:
            _run_bands_pool(
                pool, row_workers, r0,
                lambda _b, y0, y1: _extract_luma_u8_2_rows(image, dst, h, w, _LUT_GRAY, y0, y1),
            )
            return False
        ch = image.shape[2]
        if ch == 1:
            plane = image[:, :, 0]
            _run_bands_pool(
                pool, row_workers, r0,
                lambda _b, y0, y1: _extract_luma_u8_2_rows(plane, dst, h, w, _LUT_GRAY, y0, y1),
            )
            return False
        if ch in (3, 4):
            _run_bands_pool(
                pool, row_workers, r0,
                lambda _b, y0, y1: _extract_luma_u8_3_rows(
                    image, dst, h, w, _LUT_R, _LUT_G, _LUT_B, y0, y1,
                ),
            )
            return False
        return self.extract_luma(image, dst)

    def extract_channel_parallel(self, image: np.ndarray, channel: int,
                                 dst: np.ndarray, pool, row_workers: int) -> bool:
        if pool is None or row_workers <= 1 or dst.shape[0] < 64 or image.dtype != np.uint8:
            return self.extract_channel(image, channel, dst)
        h, w = image.shape[:2]
        r0 = dst.shape[0]
        if image.ndim == 2:
            _run_bands_pool(
                pool, row_workers, r0,
                lambda _b, y0, y1: _extract_chan_u8_2_rows(image, dst, h, w, _LUT_U8, y0, y1),
            )
            return False
        if image.ndim == 3 and image.shape[2] == 1:
            plane = image[:, :, 0]
            _run_bands_pool(
                pool, row_workers, r0,
                lambda _b, y0, y1: _extract_chan_u8_2_rows(plane, dst, h, w, _LUT_U8, y0, y1),
            )
            return False
        if image.ndim == 3 and image.shape[2] in (3, 4):
            src_ch = 2 - channel
            _run_bands_pool(
                pool, row_workers, r0,
                lambda _b, y0, y1: _extract_chan_u8_3_rows(
                    image, dst, h, w, src_ch, _LUT_U8, y0, y1,
                ),
            )
            return False
        return self.extract_channel(image, channel, dst)

    def forward_activity_parallel(self, ws: _NativeWorkspace, nlevels: int,
                                  pool, row_workers: int, p1_scratches):
        """Row-band parallel Pass 1 activity extraction sharing one workspace."""
        if pool is None or row_workers <= 1 or not p1_scratches:
            return self.forward_activity_into(ws, nlevels)
        plans = ws.plans
        mags = []
        sp0 = plans[0]
        r, c = sp0.in_r, sp0.in_c
        hp0_h, hp0_w = sp0.hp_h, sp0.hp_w
        mag0 = np.empty((hp0_h, hp0_w), dtype=np.float32)
        strip_pairs = p1_scratches[0][2].shape[0]
        src0 = ws.buf_a[:r, :c]
        lolo_out = ws.buf_b[:r, :c] if nlevels > 1 else None

        def _lv0_band(b, yp_start, yp_stop):
            lo_buf, hi_buf, hp_buf, abs_buf = p1_scratches[b]
            for yp in range(yp_start, yp_stop, strip_pairs):
                n_pairs = min(strip_pairs, yp_stop - yp)
                lo_s = lo_buf[:2 * n_pairs, :c]
                hi_s = hi_buf[:2 * n_pairs, :c]
                _colfilter_v5_v7_strip(
                    src0, lo_s, hi_s, self.h0o, self.h1o, r, c,
                    2 * yp, 2 * (yp + n_pairs),
                )
                hp_s = hp_buf[:n_pairs, :hp0_w]
                _colfilter_h_q2c_strip_local(
                    lo_s, hi_s, hp_s, self.h0o, self.h1o, n_pairs, c,
                )
                _write_activity_strip(hp_s, abs_buf, mag0, yp)
                if lolo_out is not None:
                    _colfilter_h5_rows(lo_s, lolo_out, self.h0o, 2 * yp, 2 * n_pairs, c)

        if hp0_h >= 64:
            _run_bands_pool(pool, row_workers, hp0_h, _lv0_band)
        else:
            _lv0_band(0, 0, hp0_h)
        mags.append(mag0)

        if nlevels == 1:
            return mags

        lolo = lolo_out
        for lvl in range(1, nlevels):
            sp = plans[lvl]
            lo_r = sp.lolo_r
            lo = ws.buf_a[:lo_r, :sp.in_c]
            j_total = lo_r >> 1
            if j_total >= 64:
                _run_bands_pool(
                    pool, row_workers, j_total,
                    lambda _b, j0, j1, _lolo=lolo, _lo=lo, _sp=sp: _coldfilt_v_rows(
                        _lolo, _lo[2 * j0:2 * j1], self.h0b, self.h0a, True,
                        _sp.xe_r_fwd, j0, j1, _sp.in_c,
                    ),
                )
            else:
                _coldfilt_v_rows(lolo, lo, self.h0b, self.h0a, True, sp.xe_r_fwd, 0, j_total, sp.in_c)

            mag_lvl = np.empty((sp.hp_h, sp.hp_w), dtype=np.float32)

            def _lvk_band(b, yp_start, yp_stop, _lolo=lolo, _lo=lo, _sp=sp, _mag=mag_lvl):
                _, hi_buf, hp_buf, abs_buf = p1_scratches[b]
                for yp in range(yp_start, yp_stop, strip_pairs):
                    n_pairs = min(strip_pairs, yp_stop - yp)
                    hi_s = hi_buf[:2 * n_pairs, :_sp.in_c]
                    _coldfilt_v_rows(
                        _lolo, hi_s, self.h1b, self.h1a, False,
                        _sp.xe_r_fwd, yp, yp + n_pairs, _sp.in_c,
                    )
                    hp_s = hp_buf[:n_pairs, :_sp.hp_w]
                    _coldfilt_h_q2c_strip(
                        _lo, hi_s, hp_s, self.h0b, self.h0a, self.h1b, self.h1a,
                        _sp.xe_c_fwd, yp, n_pairs, _sp.hp_w,
                    )
                    _write_activity_strip(hp_s, abs_buf, _mag, yp)

            if sp.hp_h >= 64:
                _run_bands_pool(pool, row_workers, sp.hp_h, _lvk_band)
            else:
                _lvk_band(0, 0, sp.hp_h)
            mags.append(mag_lvl)

            if lvl + 1 < nlevels:
                next_lolo = ws.buf_b[:lo_r, :sp.lolo_c]
                if lo_r >= 64:
                    _run_bands_pool(
                        pool, row_workers, lo_r,
                        lambda _b, y0, y1, _lo=lo, _nl=next_lolo, _sp=sp: _coldfilt_h_rows(
                            _lo, _nl, self.h0b, self.h0a, True, _sp.xe_c_fwd,
                            y0, y1, _sp.lolo_c >> 1,
                        ),
                    )
                else:
                    _coldfilt_h(lo, next_lolo, self.h0b, self.h0a, True, sp.xe_c_fwd, lo_r, sp.lolo_c >> 1)
                lolo = next_lolo

        return mags

    def max3x3_update_parallel(self, mag, best_act, best_idx, frame_idx: int,
                               is_first: bool, pool, row_workers: int):
        h = mag.shape[0]
        if pool is not None and row_workers > 1 and h >= 128:
            _run_bands_pool(
                pool, row_workers, h,
                lambda _b, y0, y1: _max3x3_update_rows(
                    mag, best_act, best_idx, frame_idx, is_first, y0, y1,
                ),
            )
        else:
            _max3x3_update(mag, best_act, best_idx, frame_idx, is_first)

    def forward_select_parallel(self, ws: _NativeWorkspace, nlevels: int,
                                max_level: int, need_lp: bool, active_row,
                                decisions, frame_idx: int, targets: list,
                                pool, row_workers: int, p2_scratches):
        """Row-band parallel Pass 2 sparse coefficient extraction directly into targets."""
        if pool is None or row_workers <= 1 or not p2_scratches:
            return self.forward_select_into(
                ws, nlevels, max_level, need_lp, active_row, decisions,
                frame_idx, targets, is_packed=False,
            )
        plans = ws.plans
        sp0 = plans[0]
        r, c = sp0.in_r, sp0.in_c
        descend_0 = need_lp or max_level > 1
        src0 = ws.buf_a[:r, :c]
        lo0 = ws.buf_b[:r, :c]
        if descend_0:
            if r >= 64:
                _run_bands_pool(
                    pool, row_workers, r,
                    lambda _b, y0, y1: _colfilter_v5_rows(src0, lo0, self.h0o, r, c, y0, y1),
                )
            else:
                _colfilter_v5(src0, lo0, self.h0o, r, c)

        if active_row[0]:
            dec0 = decisions[0]
            tgt0 = targets[0]
            if dec0.shape[0] >= 64:
                _run_bands_pool(
                    pool, row_workers, dec0.shape[0],
                    lambda b, q0, q1: _level0_select_sparse_rows(
                        src0, lo0, bool(descend_0), dec0, frame_idx,
                        tgt0, False, self.h0o, self.h1o, r, c,
                        p2_scratches[b][0][:2, :c], p2_scratches[b][1][:2, :c], q0, q1,
                    ),
                )
            else:
                _level0_select_sparse_rows(
                    src0, lo0, bool(descend_0), dec0, frame_idx,
                    tgt0, False, self.h0o, self.h1o, r, c,
                    p2_scratches[0][0][:2, :c], p2_scratches[0][1][:2, :c], 0, dec0.shape[0],
                )

        if not descend_0:
            return None

        lolo = ws.buf_a[:r, :c]
        if r >= 64:
            _run_bands_pool(
                pool, row_workers, r,
                lambda _b, y0, y1: _colfilter_h5_span(lo0, lolo, self.h0o, c, y0, y1),
            )
        else:
            _colfilter_h5(lo0, lolo, self.h0o, r, c)

        for lvl in range(1, nlevels):
            descend_k = need_lp or (lvl + 1 < max_level)
            if not active_row[lvl] and not descend_k:
                break
            sp = plans[lvl]
            lo_r = sp.lolo_r
            lo = ws.buf_b[:lo_r, :sp.in_c]
            if descend_k:
                j_total = lo_r >> 1
                if j_total >= 64:
                    _run_bands_pool(
                        pool, row_workers, j_total,
                        lambda _b, j0, j1, _lolo=lolo, _lo=lo, _sp=sp: _coldfilt_v_rows(
                            _lolo, _lo[2 * j0:2 * j1], self.h0b, self.h0a, True,
                            _sp.xe_r_fwd, j0, j1, _sp.in_c,
                        ),
                    )
                else:
                    _coldfilt_v_rows(lolo, lo, self.h0b, self.h0a, True, sp.xe_r_fwd, 0, j_total, sp.in_c)

            if active_row[lvl]:
                dec_k = decisions[lvl]
                tgt_k = targets[lvl]
                if dec_k.shape[0] >= 64:
                    _run_bands_pool(
                        pool, row_workers, dec_k.shape[0],
                        lambda b, q0, q1, _lolo=lolo, _lo=lo, _dk=bool(descend_k),
                               _dec=dec_k, _tgt=tgt_k, _sp=sp: _levelk_select_sparse_rows(
                            _lolo, _lo, _dk, _dec, frame_idx,
                            _tgt, False, self.h0b, self.h0a, self.h1b, self.h1a,
                            _sp.xe_r_fwd, _sp.xe_c_fwd, _sp.in_c,
                            p2_scratches[b][0][:2, :_sp.in_c],
                            p2_scratches[b][1][:2, :_sp.in_c], q0, q1,
                        ),
                    )
                else:
                    _levelk_select_sparse_rows(
                        lolo, lo, bool(descend_k), dec_k, frame_idx,
                        tgt_k, False, self.h0b, self.h0a, self.h1b, self.h1a,
                        sp.xe_r_fwd, sp.xe_c_fwd, sp.in_c,
                        p2_scratches[0][0][:2, :sp.in_c],
                        p2_scratches[0][1][:2, :sp.in_c], 0, dec_k.shape[0],
                    )

            if not descend_k:
                return None

            next_lolo = ws.buf_a[:lo_r, :sp.lolo_c]
            if lo_r >= 64:
                _run_bands_pool(
                    pool, row_workers, lo_r,
                    lambda _b, y0, y1, _lo=lo, _nl=next_lolo, _sp=sp: _coldfilt_h_rows(
                        _lo, _nl, self.h0b, self.h0a, True, _sp.xe_c_fwd,
                        y0, y1, _sp.lolo_c >> 1,
                    ),
                )
            else:
                _coldfilt_h(lo, next_lolo, self.h0b, self.h0a, True, sp.xe_c_fwd, lo_r, sp.lolo_c >> 1)
            lolo = next_lolo

        return lolo.copy() if need_lp else None

    def inverse_parallel(self, lp: np.ndarray, fused_high, plans,
                         pool, row_workers: int) -> np.ndarray:
        """Exact native inverse DTCWT with row-band parallel level-0 synthesis."""
        if pool is None or row_workers <= 1 or len(fused_high) == 0 or plans[0].in_r < 128:
            return self.inverse(lp, fused_high, plans)
        nlevels = len(fused_high)
        sp0 = plans[0]
        buf_r = max(sp0.in_r, max((2 * p.lolo_r for p in plans[1:]), default=0))
        buf_c = max(sp0.in_c, max((2 * p.lolo_c for p in plans[1:]), default=0))
        z = np.empty((buf_r, buf_c), dtype=np.float32)
        cur_h, cur_w = lp.shape
        z[:cur_h, :cur_w] = lp

        y1 = np.empty((buf_r, buf_c), dtype=np.float32)
        y2 = np.empty((buf_r, buf_c), dtype=np.float32)

        for lvl in range(nlevels - 1, 0, -1):
            sp = plans[lvl]
            hp = fused_high[lvl]
            in_r, in_c = sp.lolo_r, sp.lolo_c
            out_r = 2 * in_r
            prev_hp = fused_high[lvl - 1]
            exp_r, exp_c = 2 * prev_hp.shape[0], 2 * prev_hp.shape[1]
            off_y = 1 if out_r != exp_r else 0
            off_x = 1 if (2 * in_c) != exp_c else 0

            quad = y2[:in_r, :in_c]
            _c2q_pair(hp, quad, 0, 5, self._inverse_scale)
            _colifilt_v_add(z[:in_r, :in_c], quad, y1[:out_r, :in_c],
                            self.g0b, self.g0a, self.g1b, self.g1a, sp.xe_r_inv, in_r, in_c)

            q_hl = np.empty((in_r, in_c), dtype=np.float32)
            q_hh = z[:in_r, :in_c]
            _c2q_pair(hp, q_hl, 2, 3, self._inverse_scale)
            _c2q_pair(hp, q_hh, 1, 4, self._inverse_scale)
            _colifilt_v_add(q_hl, q_hh, y2[:out_r, :in_c],
                            self.g0b, self.g0a, self.g1b, self.g1a, sp.xe_r_inv, in_r, in_c)
            del q_hl

            _colifilt_h_add_trim(
                y1[:out_r, :in_c], y2[:out_r, :in_c], z[:exp_r, :exp_c],
                self.g0b, self.g0a, self.g1b, self.g1a, sp.xe_c_inv,
                out_r, in_c, off_y, off_x, exp_r, exp_c,
            )

        r, c = sp0.in_r, sp0.in_c
        hp0 = fused_high[0]
        hp0_h = hp0.shape[0]
        z0 = z[:r, :c]
        y1_0 = y1[:r, :c]
        y2_0 = y2[:r, :c]
        scale = self._inverse_scale

        _run_bands_pool(
            pool, row_workers, hp0_h,
            lambda _b, y0, y1_b: _c2q_pair_rows(hp0, y2_0, 0, 5, scale, y0, y1_b),
        )
        _run_bands_pool(
            pool, row_workers, r,
            lambda _b, y0, y1_b: _colfilter_v_add0_rows(
                z0, y2_0, y1_0, self.g0o, self.g1o, r, c, y0, y1_b,
            ),
        )

        q_hl = np.empty((r, c), dtype=np.float32)
        _run_bands_pool(
            pool, row_workers, hp0_h,
            lambda _b, y0, y1_b: (
                _c2q_pair_rows(hp0, q_hl, 2, 3, scale, y0, y1_b),
                _c2q_pair_rows(hp0, z0, 1, 4, scale, y0, y1_b),
            ),
        )
        _run_bands_pool(
            pool, row_workers, r,
            lambda _b, y0, y1_b: _colfilter_v_add0_rows(
                q_hl, z0, y2_0, self.g0o, self.g1o, r, c, y0, y1_b,
            ),
        )
        del q_hl

        _run_bands_pool(
            pool, row_workers, r,
            lambda _b, y0, y1_b: _colfilter_h_add0_rows(
                y1_0, y2_0, z0, self.g0o, self.g1o, c, y0, y1_b,
            ),
        )
        del y1, y2
        return z0

    def warmup(self, dtype, ndim: int, is_bgra: bool = False):
        """Load/compile every needed njit signature on the caller thread before pooling."""
        key = (np.dtype(dtype).name, 2 if ndim == 2 else (4 if is_bgra else 3))
        with self._warmup_lock:
            if key in self._warmed_keys:
                return
            ws1 = self.make_workspace(10, 10, 2, strip_pairs=4, for_pass2=False)
            ws2 = self.make_workspace(10, 10, 2, strip_pairs=4, for_pass2=True)
            shape = (10, 10) if key[1] == 2 else (10, 10, key[1])
            dummy = np.zeros(shape, dtype=np.dtype(key[0]))
            self.extract_luma(dummy, ws1.buf_a)
            mags = self.forward_activity_into(ws1, 2)
            best_act = np.zeros_like(mags[0])
            best_idx = np.zeros(mags[0].shape, dtype=np.int32)
            _max3x3_update(mags[0], best_act, best_idx, 0, True)
            _max3x3_update(mags[0], best_act, best_idx, 1, False)
            dec = [np.zeros((sp.hp_h, sp.hp_w), dtype=np.int32) for sp in ws2.plans]
            counts_row = np.zeros(2, dtype=np.int32)
            _scan_count_decisions(dec[0], counts_row)
            active_all = np.array([dec[0].size, dec[1].size], dtype=np.int32)
            self.extract_channel(dummy, 0, ws2.buf_a)
            fused_hi = [
                np.zeros((sp.hp_h, sp.hp_w, 6), dtype=np.complex64) for sp in ws2.plans
            ]
            lp = self.forward_select_into(
                ws2, 2, 2, True, active_all, dec, 0, fused_hi, is_packed=False,
            )
            self.forward_select_into(
                ws2, 2, 1, False, active_all, dec, 0, fused_hi, is_packed=False,
            )
            packed = [
                np.empty((sp.hp_h * sp.hp_w, 1, 6), dtype=np.complex64)
                for sp in ws2.plans
            ]
            self.forward_select_into(
                ws2, 2, 2, True, active_all, dec, 0, packed, is_packed=True,
            )
            _scatter_hp(dec[0], 0, packed[0], fused_hi[0])
            w_lp = np.ones_like(lp)
            _acc_weighted_lp(w_lp, w_lp, lp)
            self.inverse(lp, fused_hi, ws2.plans)
            self._warmed_keys.add(key)


@lru_cache(maxsize=1)
def get_engine() -> _NativeDtcwtEngine:
    import dtcwt.coeffs as _coeffs

    h0o, g0o, h1o, g1o = (
        np.ascontiguousarray(np.asarray(c, dtype=np.float32).ravel())
        for c in _coeffs.biort("near_sym_a")
    )
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = (
        np.ascontiguousarray(np.asarray(c, dtype=np.float32).ravel())
        for c in _coeffs.qshift("qshift_a")
    )
    return _NativeDtcwtEngine(
        h0o, g0o, h1o, g1o, h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b,
    )
