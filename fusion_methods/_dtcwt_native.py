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
from fusion_methods._dtcwt_kernels import make_reflect_table

_SQRT_HALF_32 = np.float32(np.sqrt(0.5))
_SQRT_HALF_64 = np.float64(np.sqrt(0.5))
_W_R = np.float32(0.299)
_W_G = np.float32(0.587)
_W_B = np.float32(0.114)
_INV_255 = np.float32(255.0)
_U16_SCALE = np.float32(255.0 / 65535.0)


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
def _extract_luma_u8_3(src, dst, h, w):
    r0, c0 = dst.shape
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            b = np.float32(src[sy, sx, 0]) / _INV_255
            g = np.float32(src[sy, sx, 1]) / _INV_255
            r = np.float32(src[sy, sx, 2]) / _INV_255
            dst[y, x] = np.float32(
                np.float32(_W_R * r) + np.float32(_W_G * g)
            ) + np.float32(_W_B * b)


@njit(nogil=True, fastmath=False)
def _extract_luma_u8_2(src, dst, h, w):
    r0, c0 = dst.shape
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            v = np.float32(src[sy, sx]) / _INV_255
            dst[y, x] = np.float32(
                np.float32(_W_R * v) + np.float32(_W_G * v)
            ) + np.float32(_W_B * v)


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
def _extract_chan_u8_3(src, dst, h, w, ch):
    r0, c0 = dst.shape
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            dst[y, x] = np.float32(src[sy, sx, ch]) / _INV_255


@njit(nogil=True, fastmath=False)
def _extract_chan_u8_2(src, dst, h, w):
    r0, c0 = dst.shape
    for y in range(r0):
        sy = y if y < h else h - 1
        for x in range(c0):
            sx = x if x < w else w - 1
            dst[y, x] = np.float32(src[sy, sx]) / _INV_255


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
def _colfilter_v5(src, dst, h5, r, c):
    w0, w1, w2, w3, w4 = h5[0], h5[1], h5[2], h5[3], h5[4]
    z = np.float32(0.0)
    y_lo = 2 if r >= 4 else r
    for y in range(y_lo):
        yp2 = _ref_idx(y + 2, r)
        yp1 = _ref_idx(y + 1, r)
        ym1 = _ref_idx(y - 1, r)
        ym2 = _ref_idx(y - 2, r)
        for x in range(c):
            acc = z
            acc += src[yp2, x] * w0
            acc += src[yp1, x] * w1
            acc += src[y, x] * w2
            acc += src[ym1, x] * w3
            acc += src[ym2, x] * w4
            dst[y, x] = acc
    y_hi = max(y_lo, r - 2)
    for y in range(y_lo, y_hi):
        for x in range(c):
            acc = z
            acc += src[y + 2, x] * w0
            acc += src[y + 1, x] * w1
            acc += src[y, x] * w2
            acc += src[y - 1, x] * w3
            acc += src[y - 2, x] * w4
            dst[y, x] = acc
    for y in range(y_hi, r):
        yp2 = _ref_idx(y + 2, r)
        yp1 = _ref_idx(y + 1, r)
        ym1 = _ref_idx(y - 1, r)
        ym2 = _ref_idx(y - 2, r)
        for x in range(c):
            acc = z
            acc += src[yp2, x] * w0
            acc += src[yp1, x] * w1
            acc += src[y, x] * w2
            acc += src[ym1, x] * w3
            acc += src[ym2, x] * w4
            dst[y, x] = acc


@njit(nogil=True, fastmath=False)
def _colfilter_v7_rows(src, dst, h7, r, c, y_start, y_stop):
    w0, w1, w2, w3, w4, w5, w6 = (
        h7[0], h7[1], h7[2], h7[3], h7[4], h7[5], h7[6]
    )
    z = np.float32(0.0)
    head_stop = min(y_stop, 3)
    int_start = max(y_start, head_stop)
    int_stop = max(int_start, min(y_stop, r - 3))
    for y in range(y_start, min(y_stop, head_stop)):
        yp3 = _ref_idx(y + 3, r)
        yp2 = _ref_idx(y + 2, r)
        yp1 = _ref_idx(y + 1, r)
        ym1 = _ref_idx(y - 1, r)
        ym2 = _ref_idx(y - 2, r)
        ym3 = _ref_idx(y - 3, r)
        dy = y - y_start
        for x in range(c):
            acc = z
            acc += src[yp3, x] * w0
            acc += src[yp2, x] * w1
            acc += src[yp1, x] * w2
            acc += src[y, x] * w3
            acc += src[ym1, x] * w4
            acc += src[ym2, x] * w5
            acc += src[ym3, x] * w6
            dst[dy, x] = acc
    for y in range(int_start, int_stop):
        dy = y - y_start
        for x in range(c):
            acc = z
            acc += src[y + 3, x] * w0
            acc += src[y + 2, x] * w1
            acc += src[y + 1, x] * w2
            acc += src[y, x] * w3
            acc += src[y - 1, x] * w4
            acc += src[y - 2, x] * w5
            acc += src[y - 3, x] * w6
            dst[dy, x] = acc
    for y in range(int_stop, y_stop):
        yp3 = _ref_idx(y + 3, r)
        yp2 = _ref_idx(y + 2, r)
        yp1 = _ref_idx(y + 1, r)
        ym1 = _ref_idx(y - 1, r)
        ym2 = _ref_idx(y - 2, r)
        ym3 = _ref_idx(y - 3, r)
        dy = y - y_start
        for x in range(c):
            acc = z
            acc += src[yp3, x] * w0
            acc += src[yp2, x] * w1
            acc += src[yp1, x] * w2
            acc += src[y, x] * w3
            acc += src[ym1, x] * w4
            acc += src[ym2, x] * w5
            acc += src[ym3, x] * w6
            dst[dy, x] = acc


@njit(nogil=True, fastmath=False)
def _colfilter_h5(src, dst, h5, r, c):
    w0, w1, w2, w3, w4 = h5[0], h5[1], h5[2], h5[3], h5[4]
    z = np.float32(0.0)
    x_lo = 2 if c >= 4 else c
    x_hi = max(x_lo, c - 2)
    for y in range(r):
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
def _colfilter_h_q2c_strip(lo, hi_strip, hp_strip, h0, h1, y_pair_start, n_pairs, c):
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
        lo_top = lo[2 * (y_pair_start + yp)]
        lo_bot = lo[2 * (y_pair_start + yp) + 1]
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
def _level0_select_sparse(src_x, lo_full, has_lo_full, decisions, frame_idx,
                          out_hp, is_packed, h0, h1, r, c, lo_2row, hi_2row):
    hp_h, hp_w = decisions.shape
    pos = 0
    for qy in range(hp_h):
        dec_row = decisions[qy]
        any_win = False
        for qx in range(hp_w):
            if dec_row[qx] == frame_idx:
                any_win = True
                break
        if not any_win:
            continue
        y0 = 2 * qy
        _colfilter_v7_rows(src_x, hi_2row, h1, r, c, y0, y0 + 2)
        hi_top = hi_2row[0]
        hi_bot = hi_2row[1]
        if has_lo_full:
            lo_top = lo_full[y0]
            lo_bot = lo_full[y0 + 1]
        else:
            if 2 <= y0 and y0 + 1 < r - 2:
                w0, w1, w2, w3, w4 = h0[0], h0[1], h0[2], h0[3], h0[4]
                z = np.float32(0.0)
                for row_off in range(2):
                    ry = y0 + row_off
                    for cx_i in range(c):
                        acc = z
                        acc += src_x[ry + 2, cx_i] * w0
                        acc += src_x[ry + 1, cx_i] * w1
                        acc += src_x[ry, cx_i] * w2
                        acc += src_x[ry - 1, cx_i] * w3
                        acc += src_x[ry - 2, cx_i] * w4
                        lo_2row[row_off, cx_i] = acc
            else:
                w0, w1, w2, w3, w4 = h0[0], h0[1], h0[2], h0[3], h0[4]
                z = np.float32(0.0)
                for row_off in range(2):
                    ry = y0 + row_off
                    yp2 = _ref_idx(ry + 2, r)
                    yp1 = _ref_idx(ry + 1, r)
                    ym1 = _ref_idx(ry - 1, r)
                    ym2 = _ref_idx(ry - 2, r)
                    for cx_i in range(c):
                        acc = z
                        acc += src_x[yp2, cx_i] * w0
                        acc += src_x[yp1, cx_i] * w1
                        acc += src_x[ry, cx_i] * w2
                        acc += src_x[ym1, cx_i] * w3
                        acc += src_x[ym2, cx_i] * w4
                        lo_2row[row_off, cx_i] = acc
            lo_top = lo_2row[0]
            lo_bot = lo_2row[1]
        for qx in range(hp_w):
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
def _coldfilt_v_rows(src, dst, ha, hb, pos_sign, xe_r, j_start, j_stop, c):
    ha0, ha2, ha4, ha6, ha8 = ha[0], ha[2], ha[4], ha[6], ha[8]
    ha1, ha3, ha5, ha7, ha9 = ha[1], ha[3], ha[5], ha[7], ha[9]
    hb0, hb2, hb4, hb6, hb8 = hb[0], hb[2], hb[4], hb[6], hb[8]
    hb1, hb3, hb5, hb7, hb9 = hb[1], hb[3], hb[5], hb[7], hb[9]
    z = np.float32(0.0)
    off1 = 0 if pos_sign else 1
    off2 = 1 if pos_sign else 0

    for j in range(j_start, j_stop):
        p = 10 + 4 * j
        re0 = xe_r[p + 10]
        re1 = xe_r[p + 8]
        re2 = xe_r[p + 6]
        re3 = xe_r[p + 4]
        re4 = xe_r[p + 2]
        re5 = xe_r[p]
        re6 = xe_r[p - 2]
        re7 = xe_r[p - 4]
        re8 = xe_r[p - 6]
        re9 = xe_r[p - 8]

        ro0 = xe_r[p + 11]
        ro1 = xe_r[p + 9]
        ro2 = xe_r[p + 7]
        ro3 = xe_r[p + 5]
        ro4 = xe_r[p + 3]
        ro5 = xe_r[p + 1]
        ro6 = xe_r[p - 1]
        ro7 = xe_r[p - 3]
        ro8 = xe_r[p - 5]
        ro9 = xe_r[p - 7]

        out1 = dst[2 * (j - j_start) + off1]
        out2 = dst[2 * (j - j_start) + off2]
        for x in range(c):
            a_even = ((((z + src[re0, x] * ha0) + src[re2, x] * ha2) + src[re4, x] * ha4) + src[re6, x] * ha6) + src[re8, x] * ha8
            a_odd = ((((z + src[re1, x] * ha1) + src[re3, x] * ha3) + src[re5, x] * ha5) + src[re7, x] * ha7) + src[re9, x] * ha9
            out1[x] = a_even + a_odd

            b_even = ((((z + src[ro0, x] * hb0) + src[ro2, x] * hb2) + src[ro4, x] * hb4) + src[ro6, x] * hb6) + src[ro8, x] * hb8
            b_odd = ((((z + src[ro1, x] * hb1) + src[ro3, x] * hb3) + src[ro5, x] * hb5) + src[ro7, x] * hb7) + src[ro9, x] * hb9
            out2[x] = b_even + b_odd


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
def _coldfilt_h(src, dst, ha, hb, pos_sign, xe_c, r_out, groups_c):
    ha0, ha2, ha4, ha6, ha8 = ha[0], ha[2], ha[4], ha[6], ha[8]
    ha1, ha3, ha5, ha7, ha9 = ha[1], ha[3], ha[5], ha[7], ha[9]
    hb0, hb2, hb4, hb6, hb8 = hb[0], hb[2], hb[4], hb[6], hb[8]
    hb1, hb3, hb5, hb7, hb9 = hb[1], hb[3], hb[5], hb[7], hb[9]
    z = np.float32(0.0)
    off1 = 0 if pos_sign else 1
    off2 = 1 if pos_sign else 0

    for y in range(r_out):
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
def _coldfilt_h_q2c_strip(lo, hi_strip, hp_strip, fa0b, fa0a, fa1b, fa1a,
                          xe_c, y_pair_start, n_pairs, hp_w):
    for yp in range(n_pairs):
        lo_top = lo[2 * (y_pair_start + yp)]
        lo_bot = lo[2 * (y_pair_start + yp) + 1]
        hi_top = hi_strip[2 * yp]
        hi_bot = hi_strip[2 * yp + 1]
        for qx in range(hp_w):
            a, b = _coldfilt_h_eval(hi_top, fa0b, fa0a, True, xe_c, qx)
            cc, d = _coldfilt_h_eval(hi_bot, fa0b, fa0a, True, xe_c, qx)
            _store_q2c(hp_strip, yp, qx, a, b, cc, d, 0, 5)

            a, b = _coldfilt_h_eval(lo_top, fa1b, fa1a, False, xe_c, qx)
            cc, d = _coldfilt_h_eval(lo_bot, fa1b, fa1a, False, xe_c, qx)
            _store_q2c(hp_strip, yp, qx, a, b, cc, d, 2, 3)

            a, b = _coldfilt_h_eval(hi_top, fa1b, fa1a, False, xe_c, qx)
            cc, d = _coldfilt_h_eval(hi_bot, fa1b, fa1a, False, xe_c, qx)
            _store_q2c(hp_strip, yp, qx, a, b, cc, d, 1, 4)


@njit(nogil=True, fastmath=False)
def _levelk_select_sparse(lolo, lo_full, has_lo_full, decisions, frame_idx,
                          out_hp, is_packed, fa0b, fa0a, fa1b, fa1a,
                          xe_r, xe_c, c_raw, lo_2row, hi_2row):
    hp_h, hp_w = decisions.shape
    pos = 0
    for qy in range(hp_h):
        dec_row = decisions[qy]
        any_win = False
        for qx in range(hp_w):
            if dec_row[qx] == frame_idx:
                any_win = True
                break
        if not any_win:
            continue
        _coldfilt_v_rows(lolo, hi_2row, fa1b, fa1a, False, xe_r, qy, qy + 1, c_raw)
        hi_top = hi_2row[0]
        hi_bot = hi_2row[1]
        if has_lo_full:
            lo_top = lo_full[2 * qy]
            lo_bot = lo_full[2 * qy + 1]
        else:
            _coldfilt_v_rows(lolo, lo_2row, fa0b, fa0a, True, xe_r, qy, qy + 1, c_raw)
            lo_top = lo_2row[0]
            lo_bot = lo_2row[1]
        for qx in range(hp_w):
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
def _sum6_strip(abs_strip, mag, y_start):
    n_rows, w, _ = abs_strip.shape
    for ry in range(n_rows):
        dy = y_start + ry
        for x in range(w):
            acc = np.float32(abs_strip[ry, x, 0] + abs_strip[ry, x, 1])
            acc = np.float32(acc + abs_strip[ry, x, 2])
            acc = np.float32(acc + abs_strip[ry, x, 3])
            acc = np.float32(acc + abs_strip[ry, x, 4])
            mag[dy, x] = np.float32(acc + abs_strip[ry, x, 5])


@njit(nogil=True, fastmath=False)
def _max3x3_update(mag, best_act, best_idx, frame_idx, is_first):
    h, w = mag.shape
    for y in range(h):
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
def _c2q_pair(hp, quad, c0, c1):
    h2, w2, _ = hp.shape
    sc = _SQRT_HALF_64
    for y in range(h2):
        qy0 = 2 * y
        qy1 = qy0 + 1
        for x in range(w2):
            z0 = hp[y, x, c0]
            z1 = hp[y, x, c1]
            z0r = np.float64(z0.real)
            z0i = np.float64(z0.imag)
            z1r = np.float64(z1.real)
            z1i = np.float64(z1.imag)
            w0r = z0r * sc
            w0i = z0i * sc
            w1r = z1r * sc
            w1i = z1i * sc
            qx0 = 2 * x
            qx1 = qx0 + 1
            quad[qy0, qx0] = np.float32(w0r + w1r)
            quad[qy0, qx1] = np.float32(w0i + w1i)
            quad[qy1, qx0] = np.float32(w0i - w1i)
            quad[qy1, qx1] = np.float32(-(w0r - w1r))


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
def _colfilter_v_add0(src7, src5, dst, g0, g1, r, c):
    k0, k1, k2, k3, k4, k5, k6 = (
        g0[0], g0[1], g0[2], g0[3], g0[4], g0[5], g0[6]
    )
    m0, m1, m2, m3, m4 = g1[0], g1[1], g1[2], g1[3], g1[4]
    z = np.float32(0.0)
    y_lo = 3 if r >= 6 else r
    y_hi = max(y_lo, r - 3)

    for y in range(y_lo):
        yp3 = _ref_idx(y + 3, r)
        yp2 = _ref_idx(y + 2, r)
        yp1 = _ref_idx(y + 1, r)
        ym1 = _ref_idx(y - 1, r)
        ym2 = _ref_idx(y - 2, r)
        ym3 = _ref_idx(y - 3, r)
        for x in range(c):
            v7 = (((((((z + src7[yp3, x] * k0) + src7[yp2, x] * k1) + src7[yp1, x] * k2) + src7[y, x] * k3) + src7[ym1, x] * k4) + src7[ym2, x] * k5) + src7[ym3, x] * k6)
            v5 = (((((z + src5[yp2, x] * m0) + src5[yp1, x] * m1) + src5[y, x] * m2) + src5[ym1, x] * m3) + src5[ym2, x] * m4)
            dst[y, x] = v7 + v5

    for y in range(y_lo, y_hi):
        for x in range(c):
            v7 = (((((((z + src7[y + 3, x] * k0) + src7[y + 2, x] * k1) + src7[y + 1, x] * k2) + src7[y, x] * k3) + src7[y - 1, x] * k4) + src7[y - 2, x] * k5) + src7[y - 3, x] * k6)
            v5 = (((((z + src5[y + 2, x] * m0) + src5[y + 1, x] * m1) + src5[y, x] * m2) + src5[y - 1, x] * m3) + src5[y - 2, x] * m4)
            dst[y, x] = v7 + v5

    for y in range(y_hi, r):
        yp3 = _ref_idx(y + 3, r)
        yp2 = _ref_idx(y + 2, r)
        yp1 = _ref_idx(y + 1, r)
        ym1 = _ref_idx(y - 1, r)
        ym2 = _ref_idx(y - 2, r)
        ym3 = _ref_idx(y - 3, r)
        for x in range(c):
            v7 = (((((((z + src7[yp3, x] * k0) + src7[yp2, x] * k1) + src7[yp1, x] * k2) + src7[y, x] * k3) + src7[ym1, x] * k4) + src7[ym2, x] * k5) + src7[ym3, x] * k6)
            v5 = (((((z + src5[yp2, x] * m0) + src5[yp1, x] * m1) + src5[y, x] * m2) + src5[ym1, x] * m3) + src5[ym2, x] * m4)
            dst[y, x] = v7 + v5


@njit(nogil=True, fastmath=False)
def _colfilter_h_add0(src7, src5, dst, g0, g1, r, c):
    k0, k1, k2, k3, k4, k5, k6 = (
        g0[0], g0[1], g0[2], g0[3], g0[4], g0[5], g0[6]
    )
    m0, m1, m2, m3, m4 = g1[0], g1[1], g1[2], g1[3], g1[4]
    z = np.float32(0.0)
    x_lo = 3 if c >= 6 else c
    x_hi = max(x_lo, c - 3)
    for y in range(r):
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
            self.hi_strip = None
            self.hp_strip = None
            self.lo_2row = np.empty((2, c0), dtype=np.float32)
            self.hi_2row = np.empty((2, c0), dtype=np.float32)
        else:
            self.strip_pairs = max(1, min(strip_pairs, max(1, sp0.hp_h)))
            self.hi_strip = np.empty((2 * self.strip_pairs, c0), dtype=np.float32)
            self.hp_strip = np.empty((self.strip_pairs, sp0.hp_w, 6), dtype=np.complex64)
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
                _extract_luma_u8_2(image, dst, h, w)
                return False
            if dtype == np.uint16:
                _extract_luma_u16_2(image, dst, h, w)
                return False
            return bool(_extract_luma_f32_2(np.asarray(image, dtype=np.float32), dst, h, w))
        ch = image.shape[2]
        if ch == 1:
            plane = image[:, :, 0]
            if dtype == np.uint8:
                _extract_luma_u8_2(plane, dst, h, w)
                return False
            if dtype == np.uint16:
                _extract_luma_u16_2(plane, dst, h, w)
                return False
            return bool(_extract_luma_f32_2(np.asarray(plane, dtype=np.float32), dst, h, w))
        if ch not in (3, 4):
            raise ValueError(f"Unsupported image shape: {image.shape}")
        if dtype == np.uint8:
            _extract_luma_u8_3(image, dst, h, w)
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
                _extract_chan_u8_2(image, dst, h, w)
                return False
            if dtype == np.uint16:
                _extract_chan_u16_2(image, dst, h, w)
                return False
            return bool(_extract_chan_f32_2(np.asarray(image, dtype=np.float32), dst, h, w))
        if image.ndim == 3 and image.shape[2] == 1:
            plane = image[:, :, 0]
            if dtype == np.uint8:
                _extract_chan_u8_2(plane, dst, h, w)
                return False
            if dtype == np.uint16:
                _extract_chan_u16_2(plane, dst, h, w)
                return False
            return bool(_extract_chan_f32_2(np.asarray(plane, dtype=np.float32), dst, h, w))
        if image.ndim == 3 and image.shape[2] in (3, 4):
            src_ch = 2 - channel
            if dtype == np.uint8:
                _extract_chan_u8_3(image, dst, h, w, src_ch)
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
        lo = ws.buf_b[:r, :c]
        _colfilter_v5(ws.buf_a[:r, :c], lo, self.h0o, r, c)

        hp0_h, hp0_w = sp0.hp_h, sp0.hp_w
        mag0 = np.empty((hp0_h, hp0_w), dtype=np.float32)
        strip_pairs = ws.strip_pairs
        for yp in range(0, hp0_h, strip_pairs):
            n_pairs = min(strip_pairs, hp0_h - yp)
            hi_strip = ws.hi_strip[:2 * n_pairs, :c]
            _colfilter_v7_rows(ws.buf_a[:r, :c], hi_strip, self.h1o, r, c, 2 * yp, 2 * (yp + n_pairs))
            hp_strip = ws.hp_strip[:n_pairs, :hp0_w]
            _colfilter_h_q2c_strip(lo, hi_strip, hp_strip, self.h0o, self.h1o, yp, n_pairs, c)
            _sum6_strip(np.abs(hp_strip), mag0, yp)
        mags.append(mag0)

        if nlevels == 1:
            return mags

        lolo = ws.buf_a[:r, :c]
        _colfilter_h5(lo, lolo, self.h0o, r, c)

        for lvl in range(1, nlevels):
            sp = plans[lvl]
            lo_r = sp.lolo_r
            lo = ws.buf_b[:lo_r, :sp.in_c]
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
                _sum6_strip(np.abs(hp_strip), mag_lvl, yp)
            mags.append(mag_lvl)

            if lvl + 1 < nlevels:
                next_lolo = ws.buf_a[:lo_r, :sp.lolo_c]
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
            _c2q_pair(hp, quad, 0, 5)
            _colifilt_v_add(z[:in_r, :in_c], quad, y1[:out_r, :in_c],
                            self.g0b, self.g0a, self.g1b, self.g1a, sp.xe_r_inv, in_r, in_c)

            q_hl = np.empty((in_r, in_c), dtype=np.float32)
            q_hh = z[:in_r, :in_c]
            _c2q_pair(hp, q_hl, 2, 3)
            _c2q_pair(hp, q_hh, 1, 4)
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
        _c2q_pair(hp0, quad, 0, 5)
        _colfilter_v_add0(z[:r, :c], quad, y1[:r, :c], self.g0o, self.g1o, r, c)

        q_hl = np.empty((r, c), dtype=np.float32)
        q_hh = z[:r, :c]
        _c2q_pair(hp0, q_hl, 2, 3)
        _c2q_pair(hp0, q_hh, 1, 4)
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
