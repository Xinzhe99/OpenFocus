"""Disk-cached native CPU homography ECC solver for OpenFocus registration.

Implements a centered coarse-to-fine Inverse-Compositional (IC) pyramid
solver compatible with ``cv2.findTransformECC(..., cv2.MOTION_HOMOGRAPHY)``
semantics: the returned ``3x3`` matrix ``W`` maps template coordinates to
input coordinates so ``cv2.warpPerspective(src, W, dsize, WARP_INVERSE_MAP)``
aligns ``src`` onto ``tpl``.

All ``@njit`` kernels live at module top level and share the guarded startup
compiler wrapper in ``fusion_methods._depth_kernel`` so LLVM bitcode caches on
disk when writable and degrades safely when Numba is unavailable or memory is
constrained.
"""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np

from fusion_methods._depth_kernel import njit

_HANN_CACHE: dict[tuple[int, int], np.ndarray] = {}
_HANN_LOCK = threading.Lock()


def _hann_window(height: int, width: int) -> np.ndarray:
    key = (height, width)
    with _HANN_LOCK:
        win = _HANN_CACHE.get(key)
        if win is None:
            wy = np.hanning(height + 2)[1:-1].astype(np.float32) if height > 2 else np.ones(height, dtype=np.float32)
            wx = np.hanning(width + 2)[1:-1].astype(np.float32) if width > 2 else np.ones(width, dtype=np.float32)
            win = np.outer(wy, wx).astype(np.float32)
            win.setflags(write=False)
            if len(_HANN_CACHE) >= 32:
                _HANN_CACHE.clear()
            _HANN_CACHE[key] = win
        return win


def _phase_correlate_np(tpl: np.ndarray, src: np.ndarray) -> tuple[float, float, float]:
    """Thread-safe, non-mutating phase correlation estimate on float32 2D planes."""
    h, w = tpl.shape
    win = _hann_window(h, w)
    a0 = (tpl - float(tpl.mean())) * win
    b0 = (src - float(src.mean())) * win
    fa = np.fft.rfft2(a0)
    fb = np.fft.rfft2(b0)
    cross = np.conj(fa) * fb
    mag = np.abs(cross)
    np.maximum(mag, np.float32(1e-12), out=mag)
    cross /= mag
    corr = np.fft.irfft2(cross, s=(h, w))
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    ym, yp = (py - 1) % h, (py + 1) % h
    xm, xp = (px - 1) % w, (px + 1) % w
    denom_x = 2.0 * corr[py, px] - corr[py, xm] - corr[py, xp]
    denom_y = 2.0 * corr[py, px] - corr[ym, px] - corr[yp, px]
    dx_s = 0.5 * (corr[py, xp] - corr[py, xm]) / denom_x if denom_x > 1e-12 else 0.0
    dy_s = 0.5 * (corr[yp, px] - corr[ym, px]) / denom_y if denom_y > 1e-12 else 0.0
    if not (-1.0 <= dx_s <= 1.0):
        dx_s = 0.0
    if not (-1.0 <= dy_s <= 1.0):
        dy_s = 0.0
    sx = (px + dx_s) if px <= w // 2 else (px - w + dx_s)
    sy = (py + dy_s) if py <= h // 2 else (py - h + dy_s)
    return float(sx), float(sy), float(corr[py, px])


@njit(nogil=True, fastmath=False)
def _check_identical_and_stats(tpl, src):
    """Verify finiteness, non-flat range, exact equality, and identity NCC in one pass."""
    h, w = tpl.shape
    t_min = np.float64(tpl[0, 0])
    t_max = t_min
    s_min = np.float64(src[0, 0])
    s_max = s_min
    if not (math.isfinite(t_min) and math.isfinite(s_min)):
        return False, False, False, -2.0
    max_diff = 0.0
    st = 0.0
    st2 = 0.0
    sw = 0.0
    sw2 = 0.0
    stw = 0.0
    for y in range(h):
        rst = 0.0
        rst2 = 0.0
        rsw = 0.0
        rsw2 = 0.0
        rstw = 0.0
        for x in range(w):
            tv = np.float64(tpl[y, x])
            sv = np.float64(src[y, x])
            if not (math.isfinite(tv) and math.isfinite(sv)):
                return False, False, False, -2.0
            if tv < t_min:
                t_min = tv
            elif tv > t_max:
                t_max = tv
            if sv < s_min:
                s_min = sv
            elif sv > s_max:
                s_max = sv
            diff = abs(tv - sv)
            if diff > max_diff:
                max_diff = diff
            rst += tv
            rst2 += tv * tv
            rsw += sv
            rsw2 += sv * sv
            rstw += tv * sv
        st += rst
        st2 += rst2
        sw += rsw
        sw2 += rsw2
        stw += rstw
    is_nonflat = (t_max - t_min) > 1e-6 and (s_max - s_min) > 1e-6
    if not is_nonflat:
        return True, False, False, -2.0
    is_identical = max_diff <= 1e-7
    if is_identical:
        return True, True, True, 1.0
    inv_n = 1.0 / np.float64(h * w)
    var_t = st2 - st * st * inv_n
    var_w = sw2 - sw * sw * inv_n
    if var_t <= 1e-12 or var_w <= 1e-12:
        rho_ident = -2.0
    else:
        rho_ident = (stw - st * sw * inv_n) / math.sqrt(var_t * var_w)
    return True, True, False, rho_ident


@njit(nogil=True, fastmath=False)
def _eval_rho_warped(tpl, src, W, y0, y1, x0, x1, all_in):
    """Evaluate zero-mean NCC between tpl[y0:y1, x0:x1] and warped src."""
    h, w = src.shape
    w00 = np.float64(W[0, 0])
    w01 = np.float64(W[0, 1])
    w02 = np.float64(W[0, 2])
    w10 = np.float64(W[1, 0])
    w11 = np.float64(W[1, 1])
    w12 = np.float64(W[1, 2])
    w20 = np.float64(W[2, 0])
    w21 = np.float64(W[2, 1])
    w22 = np.float64(W[2, 2])

    w_max = np.float64(w - 1)
    h_max = np.float64(h - 1)
    is_affine = abs(w20) < 1e-13 and abs(w21) < 1e-13 and abs(w22 - 1.0) < 1e-12

    st = 0.0
    st2 = 0.0
    sw = 0.0
    sw2 = 0.0
    stw = 0.0
    cnt = 0

    for y in range(y0, y1):
        yf = np.float64(y)
        if is_affine:
            sx = w00 * np.float64(x0) + w01 * yf + w02
            sy = w10 * np.float64(x0) + w11 * yf + w12
            if all_in:
                for x in range(x0, x1):
                    ix = int(sx)
                    iy = int(sy)
                    if ix >= w - 1:
                        ix = w - 2
                    elif ix < 0:
                        ix = 0
                    if iy >= h - 1:
                        iy = h - 2
                    elif iy < 0:
                        iy = 0
                    fx = sx - np.float64(ix)
                    fy = sy - np.float64(iy)
                    v00 = np.float64(src[iy, ix])
                    v01 = np.float64(src[iy, ix + 1])
                    v10 = np.float64(src[iy + 1, ix])
                    v11 = np.float64(src[iy + 1, ix + 1])
                    iw = (v00 + fx * (v01 - v00)) * (1.0 - fy) + (v10 + fx * (v11 - v10)) * fy
                    tv = np.float64(tpl[y, x])
                    st += tv
                    st2 += tv * tv
                    sw += iw
                    sw2 += iw * iw
                    stw += tv * iw
                    cnt += 1
                    sx += w00
                    sy += w10
            else:
                for x in range(x0, x1):
                    if 0.0 <= sx <= w_max and 0.0 <= sy <= h_max:
                        ix = int(sx)
                        iy = int(sy)
                        if ix >= w - 1:
                            ix = w - 2
                        if iy >= h - 1:
                            iy = h - 2
                        fx = sx - np.float64(ix)
                        fy = sy - np.float64(iy)
                        v00 = np.float64(src[iy, ix])
                        v01 = np.float64(src[iy, ix + 1])
                        v10 = np.float64(src[iy + 1, ix])
                        v11 = np.float64(src[iy + 1, ix + 1])
                        iw = (v00 + fx * (v01 - v00)) * (1.0 - fy) + (v10 + fx * (v11 - v10)) * fy
                        tv = np.float64(tpl[y, x])
                        st += tv
                        st2 += tv * tv
                        sw += iw
                        sw2 += iw * iw
                        stw += tv * iw
                        cnt += 1
                    sx += w00
                    sy += w10
        else:
            nx = w00 * np.float64(x0) + w01 * yf + w02
            ny = w10 * np.float64(x0) + w11 * yf + w12
            nd = w20 * np.float64(x0) + w21 * yf + w22
            for x in range(x0, x1):
                if nd > 1e-7:
                    inv_d = 1.0 / nd
                    sx = nx * inv_d
                    sy = ny * inv_d
                    if all_in or (0.0 <= sx <= w_max and 0.0 <= sy <= h_max):
                        ix = int(sx)
                        iy = int(sy)
                        if ix >= w - 1:
                            ix = w - 2
                        elif ix < 0:
                            ix = 0
                        if iy >= h - 1:
                            iy = h - 2
                        elif iy < 0:
                            iy = 0
                        fx = sx - np.float64(ix)
                        fy = sy - np.float64(iy)
                        v00 = np.float64(src[iy, ix])
                        v01 = np.float64(src[iy, ix + 1])
                        v10 = np.float64(src[iy + 1, ix])
                        v11 = np.float64(src[iy + 1, ix + 1])
                        iw = (v00 + fx * (v01 - v00)) * (1.0 - fy) + (v10 + fx * (v11 - v10)) * fy
                        tv = np.float64(tpl[y, x])
                        st += tv
                        st2 += tv * tv
                        sw += iw
                        sw2 += iw * iw
                        stw += tv * iw
                        cnt += 1
                nx += w00
                ny += w10
                nd += w20

    total = (y1 - y0) * (x1 - x0)
    if cnt == 0 or cnt * 4 < total:
        return -2.0, 0
    inv_n = 1.0 / np.float64(cnt)
    var_t = st2 - st * st * inv_n
    var_w = sw2 - sw * sw * inv_n
    if var_t <= 1e-12 or var_w <= 1e-12:
        return -2.0, cnt
    cov = stw - st * sw * inv_n
    return cov / math.sqrt(var_t * var_w), cnt


@njit(nogil=True, fastmath=False)
def _prepare_level_affine(tpl, y0, y1, x0, x1, cx, cy, s, inv_s):
    """Precompute zero-mean T0, scaled Sobel gradients, t_vec, j_sum, and 6x6 Hessian."""
    rh = y1 - y0
    rw = x1 - x0
    inv_n = 1.0 / np.float64(rh * rw)

    sum_t = 0.0
    for y in range(y0, y1):
        row_s = 0.0
        for x in range(x0, x1):
            row_s += np.float64(tpl[y, x])
        sum_t += row_s
    mean_t = sum_t * inv_n

    T0 = np.empty((rh, rw), dtype=np.float32)
    gx_arr = np.empty((rh, rw), dtype=np.float32)
    gy_arr = np.empty((rh, rw), dtype=np.float32)

    g_scale = np.float64(s) * 0.125
    stride2 = rh >= 64 and rw >= 64

    norm_T2 = 0.0
    js0 = js1 = js2 = js3 = js4 = js5 = 0.0
    tv0 = tv1 = tv2 = tv3 = tv4 = tv5 = 0.0

    h00 = h01 = h02 = h03 = h04 = h05 = 0.0
    h11 = h12 = h13 = h14 = h15 = 0.0
    h22 = h23 = h24 = h25 = 0.0
    h33 = h34 = h35 = 0.0
    h44 = h45 = 0.0
    h55 = 0.0

    for ry in range(rh):
        y = y0 + ry
        v = (np.float64(y) - cy) * inv_s
        even_y = (ry & 1) == 0
        sample_h_row = even_y or (not stride2)
        for rx in range(rw):
            x = x0 + rx
            u = (np.float64(x) - cx) * inv_s
            t0 = np.float64(tpl[y, x]) - mean_t

            tl = np.float64(tpl[y - 1, x - 1])
            tc = np.float64(tpl[y - 1, x])
            tr = np.float64(tpl[y - 1, x + 1])
            ml = np.float64(tpl[y, x - 1])
            mr = np.float64(tpl[y, x + 1])
            bl = np.float64(tpl[y + 1, x - 1])
            bc = np.float64(tpl[y + 1, x])
            br = np.float64(tpl[y + 1, x + 1])

            gx = ((tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)) * g_scale
            gy = ((bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr)) * g_scale

            T0[ry, rx] = np.float32(t0)
            gx_arr[ry, rx] = np.float32(gx)
            gy_arr[ry, rx] = np.float32(gy)

            norm_T2 += t0 * t0

            j0 = gx * u
            j1 = gy * u
            j2 = gx * v
            j3 = gy * v
            j4 = gx
            j5 = gy

            js0 += j0; js1 += j1; js2 += j2; js3 += j3; js4 += j4; js5 += j5
            tv0 += j0 * t0; tv1 += j1 * t0; tv2 += j2 * t0
            tv3 += j3 * t0; tv4 += j4 * t0; tv5 += j5 * t0

            if sample_h_row and ((rx & 1) == 0 or not stride2):
                h00 += j0 * j0; h01 += j0 * j1; h02 += j0 * j2; h03 += j0 * j3; h04 += j0 * j4; h05 += j0 * j5
                h11 += j1 * j1; h12 += j1 * j2; h13 += j1 * j3; h14 += j1 * j4; h15 += j1 * j5
                h22 += j2 * j2; h23 += j2 * j3; h24 += j2 * j4; h25 += j2 * j5
                h33 += j3 * j3; h34 += j3 * j4; h35 += j3 * j5
                h44 += j4 * j4; h45 += j4 * j5
                h55 += j5 * j5

    j_sum = np.array([js0, js1, js2, js3, js4, js5], dtype=np.float64)
    t_vec = np.array([tv0, tv1, tv2, tv3, tv4, tv5], dtype=np.float64)
    h_scale = 4.0 if stride2 else 1.0
    H = np.empty((6, 6), dtype=np.float64)
    H[0, 0] = h00; H[0, 1] = h01; H[0, 2] = h02; H[0, 3] = h03; H[0, 4] = h04; H[0, 5] = h05
    H[1, 0] = h01; H[1, 1] = h11; H[1, 2] = h12; H[1, 3] = h13; H[1, 4] = h14; H[1, 5] = h15
    H[2, 0] = h02; H[2, 1] = h12; H[2, 2] = h22; H[2, 3] = h23; H[2, 4] = h24; H[2, 5] = h25
    H[3, 0] = h03; H[3, 1] = h13; H[3, 2] = h23; H[3, 3] = h33; H[3, 4] = h34; H[3, 5] = h35
    H[4, 0] = h04; H[4, 1] = h14; H[4, 2] = h24; H[4, 3] = h34; H[4, 4] = h44; H[4, 5] = h45
    H[5, 0] = h05; H[5, 1] = h15; H[5, 2] = h25; H[5, 3] = h35; H[5, 4] = h45; H[5, 5] = h55
    for r in range(6):
        for c in range(6):
            H[r, c] = H[r, c] * h_scale - j_sum[r] * j_sum[c] * inv_n

    return T0, gx_arr, gy_arr, norm_T2, j_sum, t_vec, H


@njit(nogil=True, fastmath=False)
def _prepare_level_homo(tpl, y0, y1, x0, x1, cx, cy, s, inv_s):
    """Precompute zero-mean T0, scaled Sobel gradients, t_vec, j_sum, and 8x8 Hessian."""
    rh = y1 - y0
    rw = x1 - x0
    inv_n = 1.0 / np.float64(rh * rw)

    sum_t = 0.0
    for y in range(y0, y1):
        row_s = 0.0
        for x in range(x0, x1):
            row_s += np.float64(tpl[y, x])
        sum_t += row_s
    mean_t = sum_t * inv_n

    T0 = np.empty((rh, rw), dtype=np.float32)
    gx_arr = np.empty((rh, rw), dtype=np.float32)
    gy_arr = np.empty((rh, rw), dtype=np.float32)

    g_scale = np.float64(s) * 0.125
    stride2 = rh >= 64 and rw >= 64

    norm_T2 = 0.0
    js0 = js1 = js2 = js3 = js4 = js5 = js6 = js7 = 0.0
    tv0 = tv1 = tv2 = tv3 = tv4 = tv5 = tv6 = tv7 = 0.0

    h00 = h01 = h02 = h03 = h04 = h05 = h06 = h07 = 0.0
    h11 = h12 = h13 = h14 = h15 = h16 = h17 = 0.0
    h22 = h23 = h24 = h25 = h26 = h27 = 0.0
    h33 = h34 = h35 = h36 = h37 = 0.0
    h44 = h45 = h46 = h47 = 0.0
    h55 = h56 = h57 = 0.0
    h66 = h67 = 0.0
    h77 = 0.0

    for ry in range(rh):
        y = y0 + ry
        v = (np.float64(y) - cy) * inv_s
        even_y = (ry & 1) == 0
        sample_h_row = even_y or (not stride2)
        for rx in range(rw):
            x = x0 + rx
            u = (np.float64(x) - cx) * inv_s
            t0 = np.float64(tpl[y, x]) - mean_t

            tl = np.float64(tpl[y - 1, x - 1])
            tc = np.float64(tpl[y - 1, x])
            tr = np.float64(tpl[y - 1, x + 1])
            ml = np.float64(tpl[y, x - 1])
            mr = np.float64(tpl[y, x + 1])
            bl = np.float64(tpl[y + 1, x - 1])
            bc = np.float64(tpl[y + 1, x])
            br = np.float64(tpl[y + 1, x + 1])

            gx = ((tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)) * g_scale
            gy = ((bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr)) * g_scale

            T0[ry, rx] = np.float32(t0)
            gx_arr[ry, rx] = np.float32(gx)
            gy_arr[ry, rx] = np.float32(gy)

            norm_T2 += t0 * t0

            guv = gx * u + gy * v
            j0 = gx * u
            j1 = gy * u
            j2 = -u * guv
            j3 = gx * v
            j4 = gy * v
            j5 = -v * guv
            j6 = gx
            j7 = gy

            js0 += j0; js1 += j1; js2 += j2; js3 += j3
            js4 += j4; js5 += j5; js6 += j6; js7 += j7

            tv0 += j0 * t0; tv1 += j1 * t0; tv2 += j2 * t0; tv3 += j3 * t0
            tv4 += j4 * t0; tv5 += j5 * t0; tv6 += j6 * t0; tv7 += j7 * t0

            if sample_h_row and ((rx & 1) == 0 or not stride2):
                h00 += j0 * j0; h01 += j0 * j1; h02 += j0 * j2; h03 += j0 * j3; h04 += j0 * j4; h05 += j0 * j5; h06 += j0 * j6; h07 += j0 * j7
                h11 += j1 * j1; h12 += j1 * j2; h13 += j1 * j3; h14 += j1 * j4; h15 += j1 * j5; h16 += j1 * j6; h17 += j1 * j7
                h22 += j2 * j2; h23 += j2 * j3; h24 += j2 * j4; h25 += j2 * j5; h26 += j2 * j6; h27 += j2 * j7
                h33 += j3 * j3; h34 += j3 * j4; h35 += j3 * j5; h36 += j3 * j6; h37 += j3 * j7
                h44 += j4 * j4; h45 += j4 * j5; h46 += j4 * j6; h47 += j4 * j7
                h55 += j5 * j5; h56 += j5 * j6; h57 += j5 * j7
                h66 += j6 * j6; h67 += j6 * j7
                h77 += j7 * j7

    j_sum = np.array([js0, js1, js2, js3, js4, js5, js6, js7], dtype=np.float64)
    t_vec = np.array([tv0, tv1, tv2, tv3, tv4, tv5, tv6, tv7], dtype=np.float64)
    h_scale = 4.0 if stride2 else 1.0
    H = np.empty((8, 8), dtype=np.float64)
    H[0, 0] = h00; H[0, 1] = h01; H[0, 2] = h02; H[0, 3] = h03; H[0, 4] = h04; H[0, 5] = h05; H[0, 6] = h06; H[0, 7] = h07
    H[1, 0] = h01; H[1, 1] = h11; H[1, 2] = h12; H[1, 3] = h13; H[1, 4] = h14; H[1, 5] = h15; H[1, 6] = h16; H[1, 7] = h17
    H[2, 0] = h02; H[2, 1] = h12; H[2, 2] = h22; H[2, 3] = h23; H[2, 4] = h24; H[2, 5] = h25; H[2, 6] = h26; H[2, 7] = h27
    H[3, 0] = h03; H[3, 1] = h13; H[3, 2] = h23; H[3, 3] = h33; H[3, 4] = h34; H[3, 5] = h35; H[3, 6] = h36; H[3, 7] = h37
    H[4, 0] = h04; H[4, 1] = h14; H[4, 2] = h24; H[4, 3] = h34; H[4, 4] = h44; H[4, 5] = h45; H[4, 6] = h46; H[4, 7] = h47
    H[5, 0] = h05; H[5, 1] = h15; H[5, 2] = h25; H[5, 3] = h35; H[5, 4] = h45; H[5, 5] = h55; H[5, 6] = h56; H[5, 7] = h57
    H[6, 0] = h06; H[6, 1] = h16; H[6, 2] = h26; H[6, 3] = h36; H[6, 4] = h46; H[6, 5] = h56; H[6, 6] = h66; H[6, 7] = h67
    H[7, 0] = h07; H[7, 1] = h17; H[7, 2] = h27; H[7, 3] = h37; H[7, 4] = h47; H[7, 5] = h57; H[7, 6] = h67; H[7, 7] = h77
    for r in range(8):
        for c in range(8):
            H[r, c] = H[r, c] * h_scale - j_sum[r] * j_sum[c] * inv_n

    return T0, gx_arr, gy_arr, norm_T2, j_sum, t_vec, H


@njit(nogil=True, fastmath=False)
def _ic_pass_affine(T0, gx_arr, gy_arr, src, M, y0, y1, x0, x1, cx, cy, s, inv_s, all_in):
    """Fused zero-allocation affine warp + gradient projection pass with OOB tracking."""
    h, w = src.shape
    rh, rw = T0.shape
    m00 = np.float64(M[0, 0])
    m01 = np.float64(M[0, 1])
    m02 = np.float64(M[0, 2])
    m10 = np.float64(M[1, 0])
    m11 = np.float64(M[1, 1])
    m12 = np.float64(M[1, 2])

    u0 = (np.float64(x0) - cx) * inv_s
    dsx = m00
    dsy = m10
    w_max = np.float64(w - 1)
    h_max = np.float64(h - 1)

    cnt = 0
    sum_iw = 0.0
    sum_iw2 = 0.0
    sum_tiw = 0.0
    oob_t = 0.0
    oob_t2 = 0.0
    rg0 = rg1 = rg2 = rg3 = rg4 = rg5 = 0.0
    oj0 = oj1 = oj2 = oj3 = oj4 = oj5 = 0.0
    ojt0 = ojt1 = ojt2 = ojt3 = ojt4 = ojt5 = 0.0

    for ry in range(rh):
        v = (np.float64(y0 + ry) - cy) * inv_s
        sx = (m00 * u0 + m01 * v + m02) * s + cx
        sy = (m10 * u0 + m11 * v + m12) * s + cy
        u = u0
        rs_iw = 0.0
        rs_iw2 = 0.0
        rs_tiw = 0.0
        if all_in:
            for rx in range(rw):
                ix = int(sx)
                iy = int(sy)
                if ix >= w - 1:
                    ix = w - 2
                elif ix < 0:
                    ix = 0
                if iy >= h - 1:
                    iy = h - 2
                elif iy < 0:
                    iy = 0
                fx = sx - np.float64(ix)
                fy = sy - np.float64(iy)
                v00 = np.float64(src[iy, ix])
                v01 = np.float64(src[iy, ix + 1])
                v10 = np.float64(src[iy + 1, ix])
                v11 = np.float64(src[iy + 1, ix + 1])
                iw = (v00 + fx * (v01 - v00)) * (1.0 - fy) + (v10 + fx * (v11 - v10)) * fy

                t0 = np.float64(T0[ry, rx])
                gx = np.float64(gx_arr[ry, rx])
                gy = np.float64(gy_arr[ry, rx])
                gxw = gx * iw
                gyw = gy * iw

                rs_iw += iw
                rs_iw2 += iw * iw
                rs_tiw += t0 * iw
                rg0 += gxw * u
                rg1 += gyw * u
                rg2 += gxw * v
                rg3 += gyw * v
                rg4 += gxw
                rg5 += gyw

                sx += dsx
                sy += dsy
                u += inv_s
            cnt += rw
        else:
            for rx in range(rw):
                t0 = np.float64(T0[ry, rx])
                gx = np.float64(gx_arr[ry, rx])
                gy = np.float64(gy_arr[ry, rx])
                if 0.0 <= sx <= w_max and 0.0 <= sy <= h_max:
                    ix = int(sx)
                    iy = int(sy)
                    if ix >= w - 1:
                        ix = w - 2
                    if iy >= h - 1:
                        iy = h - 2
                    fx = sx - np.float64(ix)
                    fy = sy - np.float64(iy)
                    v00 = np.float64(src[iy, ix])
                    v01 = np.float64(src[iy, ix + 1])
                    v10 = np.float64(src[iy + 1, ix])
                    v11 = np.float64(src[iy + 1, ix + 1])
                    iw = (v00 + fx * (v01 - v00)) * (1.0 - fy) + (v10 + fx * (v11 - v10)) * fy

                    gxw = gx * iw
                    gyw = gy * iw
                    rs_iw += iw
                    rs_iw2 += iw * iw
                    rs_tiw += t0 * iw
                    rg0 += gxw * u
                    rg1 += gyw * u
                    rg2 += gxw * v
                    rg3 += gyw * v
                    rg4 += gxw
                    rg5 += gyw
                    cnt += 1
                else:
                    j0 = gx * u
                    j1 = gy * u
                    j2 = gx * v
                    j3 = gy * v
                    j4 = gx
                    j5 = gy
                    oob_t += t0
                    oob_t2 += t0 * t0
                    oj0 += j0; oj1 += j1; oj2 += j2; oj3 += j3; oj4 += j4; oj5 += j5
                    ojt0 += j0 * t0; ojt1 += j1 * t0; ojt2 += j2 * t0
                    ojt3 += j3 * t0; ojt4 += j4 * t0; ojt5 += j5 * t0
                sx += dsx
                sy += dsy
                u += inv_s
        sum_iw += rs_iw
        sum_iw2 += rs_iw2
        sum_tiw += rs_tiw

    raw_g = np.array([rg0, rg1, rg2, rg3, rg4, rg5], dtype=np.float64)
    oob_j = np.array([oj0, oj1, oj2, oj3, oj4, oj5], dtype=np.float64)
    oob_jt = np.array([ojt0, ojt1, ojt2, ojt3, ojt4, ojt5], dtype=np.float64)
    return cnt, sum_iw, sum_iw2, sum_tiw, oob_t, oob_t2, raw_g, oob_j, oob_jt


@njit(nogil=True, fastmath=False)
def _ic_pass_homo(T0, gx_arr, gy_arr, src, M, y0, y1, x0, x1, cx, cy, s, inv_s, all_in):
    """Fused zero-allocation homography warp + gradient projection pass with OOB tracking."""
    h, w = src.shape
    rh, rw = T0.shape
    m00 = np.float64(M[0, 0])
    m01 = np.float64(M[0, 1])
    m02 = np.float64(M[0, 2])
    m10 = np.float64(M[1, 0])
    m11 = np.float64(M[1, 1])
    m12 = np.float64(M[1, 2])
    m20 = np.float64(M[2, 0])
    m21 = np.float64(M[2, 1])
    m22 = np.float64(M[2, 2])

    u0 = (np.float64(x0) - cx) * inv_s
    d_den = m20 * inv_s
    d_nsx = m00 + cx * d_den
    d_nsy = m10 + cy * d_den

    w_max = np.float64(w - 1)
    h_max = np.float64(h - 1)

    cnt = 0
    sum_iw = 0.0
    sum_iw2 = 0.0
    sum_tiw = 0.0
    oob_t = 0.0
    oob_t2 = 0.0
    rg0 = rg1 = rg2 = rg3 = rg4 = rg5 = rg6 = rg7 = 0.0
    oj0 = oj1 = oj2 = oj3 = oj4 = oj5 = oj6 = oj7 = 0.0
    ojt0 = ojt1 = ojt2 = ojt3 = ojt4 = ojt5 = ojt6 = ojt7 = 0.0

    for ry in range(rh):
        v = (np.float64(y0 + ry) - cy) * inv_s
        nu0 = m00 * u0 + m01 * v + m02
        nv0 = m10 * u0 + m11 * v + m12
        den = m20 * u0 + m21 * v + m22
        nsx = nu0 * s + cx * den
        nsy = nv0 * s + cy * den
        u = u0
        rs_iw = 0.0
        rs_iw2 = 0.0
        rs_tiw = 0.0
        if all_in:
            for rx in range(rw):
                inv_d = 1.0 / den
                sx = nsx * inv_d
                sy = nsy * inv_d
                ix = int(sx)
                iy = int(sy)
                if ix >= w - 1:
                    ix = w - 2
                elif ix < 0:
                    ix = 0
                if iy >= h - 1:
                    iy = h - 2
                elif iy < 0:
                    iy = 0
                fx = sx - np.float64(ix)
                fy = sy - np.float64(iy)
                v00 = np.float64(src[iy, ix])
                v01 = np.float64(src[iy, ix + 1])
                v10 = np.float64(src[iy + 1, ix])
                v11 = np.float64(src[iy + 1, ix + 1])
                iw = (v00 + fx * (v01 - v00)) * (1.0 - fy) + (v10 + fx * (v11 - v10)) * fy

                t0 = np.float64(T0[ry, rx])
                gx = np.float64(gx_arr[ry, rx])
                gy = np.float64(gy_arr[ry, rx])
                gxw = gx * iw
                gyw = gy * iw
                guvw = (gxw * u + gyw * v)

                rs_iw += iw
                rs_iw2 += iw * iw
                rs_tiw += t0 * iw
                rg0 += gxw * u
                rg1 += gyw * u
                rg2 -= u * guvw
                rg3 += gxw * v
                rg4 += gyw * v
                rg5 -= v * guvw
                rg6 += gxw
                rg7 += gyw

                nsx += d_nsx
                nsy += d_nsy
                den += d_den
                u += inv_s
            cnt += rw
        else:
            for rx in range(rw):
                t0 = np.float64(T0[ry, rx])
                gx = np.float64(gx_arr[ry, rx])
                gy = np.float64(gy_arr[ry, rx])
                valid = False
                sx = 0.0
                sy = 0.0
                if den > 1e-7:
                    inv_d = 1.0 / den
                    sx = nsx * inv_d
                    sy = nsy * inv_d
                    if 0.0 <= sx <= w_max and 0.0 <= sy <= h_max:
                        valid = True
                if valid:
                    ix = int(sx)
                    iy = int(sy)
                    if ix >= w - 1:
                        ix = w - 2
                    if iy >= h - 1:
                        iy = h - 2
                    fx = sx - np.float64(ix)
                    fy = sy - np.float64(iy)
                    v00 = np.float64(src[iy, ix])
                    v01 = np.float64(src[iy, ix + 1])
                    v10 = np.float64(src[iy + 1, ix])
                    v11 = np.float64(src[iy + 1, ix + 1])
                    iw = (v00 + fx * (v01 - v00)) * (1.0 - fy) + (v10 + fx * (v11 - v10)) * fy

                    gxw = gx * iw
                    gyw = gy * iw
                    guvw = (gxw * u + gyw * v)

                    rs_iw += iw
                    rs_iw2 += iw * iw
                    rs_tiw += t0 * iw
                    rg0 += gxw * u
                    rg1 += gyw * u
                    rg2 -= u * guvw
                    rg3 += gxw * v
                    rg4 += gyw * v
                    rg5 -= v * guvw
                    rg6 += gxw
                    rg7 += gyw
                    cnt += 1
                else:
                    guv = gx * u + gy * v
                    j0 = gx * u
                    j1 = gy * u
                    j2 = -u * guv
                    j3 = gx * v
                    j4 = gy * v
                    j5 = -v * guv
                    j6 = gx
                    j7 = gy
                    oob_t += t0
                    oob_t2 += t0 * t0
                    oj0 += j0; oj1 += j1; oj2 += j2; oj3 += j3
                    oj4 += j4; oj5 += j5; oj6 += j6; oj7 += j7
                    ojt0 += j0 * t0; ojt1 += j1 * t0; ojt2 += j2 * t0; ojt3 += j3 * t0
                    ojt4 += j4 * t0; ojt5 += j5 * t0; ojt6 += j6 * t0; ojt7 += j7 * t0
                nsx += d_nsx
                nsy += d_nsy
                den += d_den
                u += inv_s
        sum_iw += rs_iw
        sum_iw2 += rs_iw2
        sum_tiw += rs_tiw

    raw_g = np.array([rg0, rg1, rg2, rg3, rg4, rg5, rg6, rg7], dtype=np.float64)
    oob_j = np.array([oj0, oj1, oj2, oj3, oj4, oj5, oj6, oj7], dtype=np.float64)
    oob_jt = np.array([ojt0, ojt1, ojt2, ojt3, ojt4, ojt5, ojt6, ojt7], dtype=np.float64)
    return cnt, sum_iw, sum_iw2, sum_tiw, oob_t, oob_t2, raw_g, oob_j, oob_jt


def _centered_mats(w: int, h: int):
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    s = max(w, h) * 0.5
    inv_s = 1.0 / s
    T = np.array([
        [inv_s, 0.0, -cx * inv_s],
        [0.0, inv_s, -cy * inv_s],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    T_inv = np.array([
        [s, 0.0, cx],
        [0.0, s, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return cx, cy, s, inv_s, T, T_inv


def _roi_all_in(W: np.ndarray, w: int, h: int, y0: int, y1: int, x0: int, x1: int) -> bool:
    corners = np.array([
        [float(x0), float(y0), 1.0],
        [float(x1 - 1), float(y0), 1.0],
        [float(x1 - 1), float(y1 - 1), 1.0],
        [float(x0), float(y1 - 1), 1.0],
    ], dtype=np.float64)
    pts = corners @ W.T
    dens = pts[:, 2]
    if np.any(dens <= 1e-4):
        return False
    xs = pts[:, 0] / dens
    ys = pts[:, 1] / dens
    return bool(
        np.all(xs >= 0.001) and np.all(xs <= w - 1.002)
        and np.all(ys >= 0.001) and np.all(ys <= h - 1.002)
    )


def _inscribed_roi(W: np.ndarray, w: int, h: int) -> tuple[int, int, int, int, bool]:
    full = (1, h - 1, 1, w - 1)
    if _roi_all_in(W, w, h, *full):
        return 1, h - 1, 1, w - 1, True
    if min(w, h) < 64:
        return 1, h - 1, 1, w - 1, False
    try:
        W_inv = np.linalg.inv(W)
    except np.linalg.LinAlgError:
        return 1, h - 1, 1, w - 1, False
    pad = 1.0
    src_corners = np.array([
        [pad, pad, 1.0],
        [w - 1.0 - pad, pad, 1.0],
        [w - 1.0 - pad, h - 1.0 - pad, 1.0],
        [pad, h - 1.0 - pad, 1.0],
    ], dtype=np.float64)
    mapped = src_corners @ W_inv.T
    if np.any(mapped[:, 2] <= 1e-4):
        return 1, h - 1, 1, w - 1, False
    tx = mapped[:, 0] / mapped[:, 2]
    ty = mapped[:, 1] / mapped[:, 2]
    x0 = max(1, int(math.ceil(max(tx[0], tx[3]))))
    x1 = min(w - 1, int(math.floor(min(tx[1], tx[2]))) + 1)
    y0 = max(1, int(math.ceil(max(ty[0], ty[1]))))
    y1 = min(h - 1, int(math.floor(min(ty[2], ty[3]))) + 1)
    area = max(0, x1 - x0) * max(0, y1 - y0)
    if x1 - x0 < 12 or y1 - y0 < 12 or area * 4 < w * h:
        return 1, h - 1, 1, w - 1, False
    return y0, y1, x0, x1, _roi_all_in(W, w, h, y0, y1, x0, x1)


def _max_corner_step_px(M_new: np.ndarray, M_old: np.ndarray, cx: float, cy: float, s: float,
                        w: int, h: int) -> float:
    u_min = (0.0 - cx) / s
    u_max = (w - 1.0 - cx) / s
    v_min = (0.0 - cy) / s
    v_max = (h - 1.0 - cy) / s
    corners = np.array([
        [u_min, v_min, 1.0],
        [u_max, v_min, 1.0],
        [u_max, v_max, 1.0],
        [u_min, v_max, 1.0],
    ], dtype=np.float64)
    p1 = corners @ M_new.T
    p0 = corners @ M_old.T
    if np.any(np.abs(p1[:, 2]) < 1e-6) or np.any(np.abs(p0[:, 2]) < 1e-6):
        return float("inf")
    d = np.empty((4, 2), dtype=np.float64)
    d[:, 0] = (p1[:, 0] / p1[:, 2] - p0[:, 0] / p0[:, 2]) * s
    d[:, 1] = (p1[:, 1] / p1[:, 2] - p0[:, 1] / p0[:, 2]) * s
    return float(np.sqrt(np.max(d[:, 0] ** 2 + d[:, 1] ** 2)))


def _solve_level(tpl: np.ndarray, src: np.ndarray, W_init: np.ndarray,
                 *, is_affine: bool, max_iter: int, eps_px: float) -> tuple[np.ndarray, float]:
    h, w = tpl.shape
    cx, cy, s, inv_s, T, T_inv = _centered_mats(w, h)
    y0, y1, x0, x1, all_in = _inscribed_roi(W_init, w, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return W_init.copy(), -2.0

    if is_affine:
        T0, gx_arr, gy_arr, norm_T2, j_sum, t_vec, H = _prepare_level_affine(
            tpl, y0, y1, x0, x1, cx, cy, s, inv_s)
    else:
        T0, gx_arr, gy_arr, norm_T2, j_sum, t_vec, H = _prepare_level_homo(
            tpl, y0, y1, x0, x1, cx, cy, s, inv_s)

    if norm_T2 <= 1e-12:
        return W_init.copy(), -2.0

    diag = np.diag(H)
    max_diag = float(np.max(diag)) if diag.size else 0.0
    if not np.isfinite(max_diag) or max_diag <= 1e-15:
        return W_init.copy(), -2.0
    H_reg = H.copy()
    reg = max(1e-12, 1e-8 * max_diag)
    H_reg[np.diag_indices_from(H_reg)] += reg
    try:
        hp_base = np.linalg.solve(H_reg, t_vec)
    except np.linalg.LinAlgError:
        return W_init.copy(), -2.0

    A_base = norm_T2 - float(np.dot(t_vec, hp_base))
    if not np.isfinite(A_base) or A_base <= 1e-12:
        return W_init.copy(), -2.0

    M = T @ W_init @ T_inv
    if abs(M[2, 2]) <= 1e-9:
        return W_init.copy(), -2.0
    M /= M[2, 2]
    if is_affine:
        M[2, 0] = 0.0
        M[2, 1] = 0.0
        M[2, 2] = 1.0

    n_pix = (y1 - y0) * (x1 - x0)
    best_rho = -2.0
    best_M = M.copy()
    prev_rho = -2.0
    roi_all_in = all_in
    last_eval_M = None
    converged = False

    for it in range(max_iter):
        if it == 0:
            roi_all_in = all_in
        else:
            W_tmp = T_inv @ M @ T
            roi_all_in = _roi_all_in(W_tmp, w, h, y0, y1, x0, x1)

        if is_affine:
            cnt, sum_iw, sum_iw2, sum_tiw, oob_t, oob_t2, raw_g, oob_j, oob_jt = _ic_pass_affine(
                T0, gx_arr, gy_arr, src, M, y0, y1, x0, x1, cx, cy, s, inv_s, roi_all_in)
        else:
            cnt, sum_iw, sum_iw2, sum_tiw, oob_t, oob_t2, raw_g, oob_j, oob_jt = _ic_pass_homo(
                T0, gx_arr, gy_arr, src, M, y0, y1, x0, x1, cx, cy, s, inv_s, roi_all_in)

        if cnt * 4 < n_pix:
            break
        n_f = float(cnt)
        mean_iw = sum_iw / n_f
        var_iw = sum_iw2 - sum_iw * mean_iw
        if not np.isfinite(var_iw) or var_iw <= 1e-12:
            break

        if cnt == n_pix:
            norm_T2_v = norm_T2
            cov = sum_tiw
            j_sum_v = j_sum
            hp = hp_base
            A = A_base
        else:
            sum_t_v = -oob_t
            mean_t_v = sum_t_v / n_f
            norm_T2_v = (norm_T2 - oob_t2) - sum_t_v * mean_t_v
            if not np.isfinite(norm_T2_v) or norm_T2_v <= 1e-12:
                break
            cov = sum_tiw - mean_t_v * sum_iw
            j_sum_v = j_sum - oob_j
            t_vec_v = (t_vec - oob_jt) - mean_t_v * j_sum_v
            try:
                hp = np.linalg.solve(H_reg, t_vec_v)
            except np.linalg.LinAlgError:
                break
            A = norm_T2_v - float(np.dot(t_vec_v, hp))
            if not np.isfinite(A) or A <= 1e-12:
                break

        rho = cov / math.sqrt(norm_T2_v * var_iw)
        if not np.isfinite(rho):
            break
        if it == 0:
            init_rho = rho
        last_eval_M = M
        if rho > best_rho:
            best_rho = rho
            best_M = M.copy()
        if it >= 3 and rho < prev_rho - 2e-3:
            break
        prev_rho = rho

        g = raw_g - mean_iw * j_sum_v
        try:
            hw = np.linalg.solve(H_reg, g)
        except np.linalg.LinAlgError:
            break
        lam_d = cov - float(np.dot(g, hp))
        if not np.isfinite(lam_d) or lam_d <= 1e-12 * A:
            break
        lam = A / lam_d
        dp = lam * hw - hp

        if is_affine:
            M_inc = np.array([
                [1.0 + dp[0], dp[2], dp[4]],
                [dp[1], 1.0 + dp[3], dp[5]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        else:
            M_inc = np.array([
                [1.0 + dp[0], dp[3], dp[6]],
                [dp[1], 1.0 + dp[4], dp[7]],
                [dp[2], dp[5], 1.0],
            ], dtype=np.float64)

        if abs(np.linalg.det(M_inc)) <= 1e-6:
            break
        try:
            M_cand = M @ np.linalg.inv(M_inc)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(M_cand)) or abs(M_cand[2, 2]) <= 1e-7:
            break
        M_cand /= M_cand[2, 2]
        if is_affine:
            M_cand[2, 0] = 0.0
            M_cand[2, 1] = 0.0
            M_cand[2, 2] = 1.0

        step_px = _max_corner_step_px(M_cand, M, cx, cy, s, w, h)
        if not np.isfinite(step_px) or step_px > 0.5 * max(w, h):
            break
        M = M_cand
        if step_px < eps_px:
            converged = True
            break

    if last_eval_M is not M:
        W_last = T_inv @ M @ T
        if abs(W_last[2, 2]) > 1e-7:
            W_last /= W_last[2, 2]
            rho_last, valid_last = _eval_rho_warped(
                tpl, src, W_last, y0, y1, x0, x1, _roi_all_in(W_last, w, h, y0, y1, x0, x1))
            if valid_last > 0 and (rho_last > best_rho or (converged and rho_last >= best_rho - 5e-3)):
                best_rho = rho_last
                best_M = M

    W_best = T_inv @ best_M @ T
    if not np.all(np.isfinite(W_best)) or abs(W_best[2, 2]) <= 1e-7:
        return W_init.copy(), -2.0
    W_best /= W_best[2, 2]
    return W_best, float(best_rho)


def _scale_homography(W: np.ndarray, sx: float, sy: float) -> np.ndarray:
    out = W.copy()
    out[0, 1] *= sx / sy
    out[0, 2] *= sx
    out[1, 0] *= sy / sx
    out[1, 2] *= sy
    out[2, 0] /= sx
    out[2, 1] /= sy
    out /= out[2, 2]
    return out


def _homography_plausible(W: np.ndarray, w: int, h: int) -> bool:
    if not np.all(np.isfinite(W)) or abs(W[2, 2]) <= 1e-6:
        return False
    W = W / W[2, 2]
    corners = np.array([
        [0.0, 0.0, 1.0],
        [w - 1.0, 0.0, 1.0],
        [w - 1.0, h - 1.0, 1.0],
        [0.0, h - 1.0, 1.0],
    ], dtype=np.float64)
    pts = corners @ W.T
    dens = pts[:, 2]
    if np.any(dens <= 0.2) or np.any(dens >= 5.0):
        return False
    px = pts[:, 0] / dens
    py = pts[:, 1] / dens
    area2 = 0.0
    for i in range(4):
        j = (i + 1) & 3
        area2 += px[i] * py[j] - px[j] * py[i]
    img_area = float(w * h)
    if area2 <= 0.1 * img_area or area2 >= 10.0 * img_area:
        return False
    det_ul = W[0, 0] * W[1, 1] - W[0, 1] * W[1, 0]
    if det_ul <= 0.1 or det_ul >= 10.0:
        return False
    return True


def build_ecc_pyramid(img: np.ndarray) -> list[np.ndarray]:
    """Build contiguous float32 coarsening pyramid (level 0 is full resolution)."""
    cur = np.ascontiguousarray(img, dtype=np.float32)
    pyr: list[np.ndarray] = [cur]
    for _ in range(3):
        ch, cw = cur.shape
        if max(ch, cw) <= 192 or min(ch, cw) < 48:
            break
        nh = (ch + 1) // 2
        nw = (cw + 1) // 2
        if min(nh, nw) < 24:
            break
        cur = np.ascontiguousarray(cv2.resize(cur, (nw, nh), interpolation=cv2.INTER_AREA))
        pyr.append(cur)
    return pyr


def _coarsest_initial_warp(
    tc: np.ndarray,
    sc: np.ndarray,
    native_mod,
) -> np.ndarray:
    """Compute initial translation-aware 3x3 homography on the coarsest pyramid level."""
    hc, wc = tc.shape
    W = np.eye(3, 3, dtype=np.float64)
    if native_mod is not None:
        rho_ident_c, _ = native_mod.ecc_eval_rho_warped(tc, sc, W, 1, hc - 1, 1, wc - 1, True)
    else:
        rho_ident_c, _ = _eval_rho_warped(tc, sc, W, 1, hc - 1, 1, wc - 1, True)
    best_rho_c = rho_ident_c
    if rho_ident_c < 0.995:
        dx, dy, response = _phase_correlate_np(tc, sc)
        max_shift = 0.38 * max(wc, hc)
        if np.isfinite(dx) and np.isfinite(dy) and max(abs(dx), abs(dy)) <= max_shift and response > 0.02:
            for sgn in (1.0, -1.0):
                W_try = np.array([
                    [1.0, 0.0, sgn * dx],
                    [0.0, 1.0, sgn * dy],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                y0, y1, x0, x1, all_in = _inscribed_roi(W_try, wc, hc)
                if native_mod is not None:
                    rho_try, valid_try = native_mod.ecc_eval_rho_warped(tc, sc, W_try, y0, y1, x0, x1, all_in)
                else:
                    rho_try, valid_try = _eval_rho_warped(tc, sc, W_try, y0, y1, x0, x1, all_in)
                if valid_try > 0 and rho_try > best_rho_c + 0.005:
                    best_rho_c = rho_try
                    W = W_try
    return W


def find_transform_ecc_homography_pyramids(
    tpl_pyr: list[np.ndarray],
    src_pyr: list[np.ndarray],
    *,
    max_iter: int = 50,
    eps: float = 1e-4,
    num_threads: int = 1,
) -> tuple[float | None, np.ndarray | None]:
    """Estimate 3x3 ECC homography from prebuilt float32 pyramids."""
    if not tpl_pyr or len(tpl_pyr) != len(src_pyr):
        return None, None
    tpl_f = tpl_pyr[0]
    src_f = src_pyr[0]
    if (not isinstance(tpl_f, np.ndarray) or not isinstance(src_f, np.ndarray)
            or tpl_f.ndim != 2 or src_f.ndim != 2 or tpl_f.shape != src_f.shape):
        return None, None
    h, w = tpl_f.shape
    if h < 16 or w < 16:
        return None, None

    from core._native import load_native
    native_mod = load_native()
    if native_mod is not None and hasattr(native_mod, 'ecc_refine_pyramids'):
        is_finite, is_nonflat, is_identical, rho_ident_full = native_mod.ecc_check_identical_and_stats(
            tpl_f, src_f, int(num_threads)
        )
    else:
        native_mod = None
        is_finite, is_nonflat, is_identical, rho_ident_full = _check_identical_and_stats(tpl_f, src_f)

    if not is_finite or not is_nonflat:
        return None, None
    if is_identical:
        return 1.0, np.eye(3, 3, dtype=np.float32)

    tc, sc = tpl_pyr[-1], src_pyr[-1]
    W = _coarsest_initial_warp(tc, sc, native_mod)
    eps = float(eps)

    if native_mod is not None:
        ok, final_cc, W_f32 = native_mod.ecc_refine_pyramids(
            tpl_pyr, src_pyr, W, float(rho_ident_full), int(max_iter), eps, int(num_threads)
        )
        if not ok:
            return None, None
        return float(final_cc), np.array(W_f32, copy=True)

    # Coarse-to-fine IC refinement (Numba fallback)
    n_levels = len(tpl_pyr)
    for lvl in range(n_levels - 1, -1, -1):
        tl, sl = tpl_pyr[lvl], src_pyr[lvl]
        hl, wl = tl.shape
        if lvl > 0:
            lvl_iter = min(max_iter, 16 if lvl == n_levels - 1 else 12)
            lvl_eps = max(eps * 2.0, 1e-3)
            is_affine = True
        else:
            lvl_iter = max_iter
            lvl_eps = min(max(eps * 2.0, 2e-4), 3e-4)
            is_affine = False

        W_next, _ = _solve_level(
            tl, sl, W,
            is_affine=is_affine,
            max_iter=lvl_iter,
            eps_px=lvl_eps,
        )
        if _homography_plausible(W_next, wl, hl):
            W = W_next
        if lvl > 0:
            tf = tpl_pyr[lvl - 1]
            hf, wf = tf.shape
            W = _scale_homography(W, wf / float(wl), hf / float(hl))

    if not _homography_plausible(W, w, h):
        return None, None

    final_all_in = _roi_all_in(W, w, h, 0, h, 0, w)
    final_cc, valid_final = _eval_rho_warped(tpl_f, src_f, W, 0, h, 0, w, final_all_in)
    if valid_final == 0 or not np.isfinite(final_cc):
        return None, None

    if final_cc < 0.25 or final_cc < rho_ident_full - 1e-3:
        if rho_ident_full > final_cc and rho_ident_full > 0.9999:
            return float(rho_ident_full), np.eye(3, 3, dtype=np.float32)
        return None, None

    return float(final_cc), W.astype(np.float32)


def find_transform_ecc_homography(
    tpl: np.ndarray,
    src: np.ndarray,
    *,
    max_iter: int = 50,
    eps: float = 1e-4,
    num_threads: int = 1,
) -> tuple[float | None, np.ndarray | None]:
    """Estimate a 3x3 homography aligning ``src`` onto ``tpl`` via ECC.

    Returns ``(cc, warp_matrix)`` or ``(None, None)`` when the caller should
    fall back to ``cv2.findTransformECC`` (e.g., flat/degenerate/tiny frames or
    unusual divergence).
    """
    if (not isinstance(tpl, np.ndarray) or not isinstance(src, np.ndarray)
            or tpl.ndim != 2 or src.ndim != 2 or tpl.shape != src.shape):
        return None, None
    h, w = tpl.shape
    if h < 16 or w < 16:
        return None, None

    tpl_f = np.ascontiguousarray(tpl, dtype=np.float32)
    src_f = np.ascontiguousarray(src, dtype=np.float32)

    from core._native import load_native
    native_mod = load_native()
    if native_mod is not None and hasattr(native_mod, 'ecc_check_identical_and_stats'):
        is_finite, is_nonflat, is_identical, _ = native_mod.ecc_check_identical_and_stats(
            tpl_f, src_f, int(num_threads)
        )
    else:
        is_finite, is_nonflat, is_identical, _ = _check_identical_and_stats(tpl_f, src_f)
    if not is_finite or not is_nonflat:
        return None, None
    if is_identical:
        return 1.0, np.eye(3, 3, dtype=np.float32)

    tpl_pyr = build_ecc_pyramid(tpl_f)
    src_pyr = build_ecc_pyramid(src_f)
    return find_transform_ecc_homography_pyramids(
        tpl_pyr, src_pyr, max_iter=max_iter, eps=eps, num_threads=num_threads
    )
