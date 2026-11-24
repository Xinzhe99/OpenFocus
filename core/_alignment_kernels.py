"""Native CPU kernels for OpenFocus translation and similarity alignment.

These mirror the row-wise definitions in core.openfocus_alignment exactly:
uint32 accumulation, truncating integer division, edge replication and the
float32 sampler clamp order. Numba is optional; callers fall back to NumPy.
"""
import numpy as np

from core._native import load_native

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - optional dependency
    njit = None
    prange = range
    _HAVE_NUMBA = False

_WEIGHTS = (12, 64, 37, 2)


def _native():
    mod = load_native()
    if mod is not None and hasattr(mod, "of_align_gray16_rows"):
        return mod
    return None


def _as_ndarray(arr):
    return arr if type(arr) is np.ndarray else arr.view(np.ndarray)


def available():
    return (_native() is not None) or _HAVE_NUMBA


if _HAVE_NUMBA:
    @njit(cache=True, nogil=True)
    def _nb_swap_rgb_to_bgr_rows_u8(src, dst, y_start, y_stop):
        w = src.shape[1]
        for y in range(y_start, y_stop):
            for x in range(w):
                dst[y, x, 0] = src[y, x, 2]
                dst[y, x, 1] = src[y, x, 1]
                dst[y, x, 2] = src[y, x, 0]

    @njit(cache=True, nogil=True)
    def _nb_swap_rgb_to_bgr_rows_u16(src, dst, y_start, y_stop):
        w = src.shape[1]
        for y in range(y_start, y_stop):
            for x in range(w):
                dst[y, x, 0] = src[y, x, 2]
                dst[y, x, 1] = src[y, x, 1]
                dst[y, x, 2] = src[y, x, 0]

    @njit(cache=True, nogil=True)
    def _nb_swap_bgr_inplace_rows_u8(arr, y_start, y_stop):
        w = arr.shape[1]
        for y in range(y_start, y_stop):
            for x in range(w):
                v0 = arr[y, x, 0]
                arr[y, x, 0] = arr[y, x, 2]
                arr[y, x, 2] = v0

    @njit(cache=True, nogil=True)
    def _nb_swap_bgr_inplace_rows_u16(arr, y_start, y_stop):
        w = arr.shape[1]
        for y in range(y_start, y_stop):
            for x in range(w):
                v0 = arr[y, x, 0]
                arr[y, x, 0] = arr[y, x, 2]
                arr[y, x, 2] = v0

    @njit(cache=True, nogil=True)
    def _nb_gray16_u8_rows(image, out, y_start, y_stop):
        w = image.shape[1]
        if image.ndim == 2:
            for y in range(y_start, y_stop):
                for x in range(w):
                    out[y, x] = np.uint16(np.uint32(image[y, x]) << np.uint32(8))
        elif image.shape[2] >= 3:
            for y in range(y_start, y_stop):
                for x in range(w):
                    value = (
                        np.uint32(image[y, x, 0])
                        + np.uint32(image[y, x, 1])
                        + np.uint32(image[y, x, 2])
                    ) // np.uint32(3)
                    out[y, x] = np.uint16(value << np.uint32(8))
        else:
            take = image.shape[2]
            for y in range(y_start, y_stop):
                for x in range(w):
                    value = np.uint32(0)
                    for c in range(take):
                        value += np.uint32(image[y, x, c])
                    out[y, x] = np.uint16((value // np.uint32(take)) << np.uint32(8))
        return out

    @njit(cache=True, nogil=True)
    def _nb_gray16_u8(image, out):
        return _nb_gray16_u8_rows(image, out, 0, image.shape[0])

    @njit(cache=True, nogil=True)
    def _nb_down_half_rows(src, out, oy_start, oy_stop):
        h, w = src.shape
        ow = out.shape[1]
        scratch = np.empty(w, np.uint32)
        for oy in range(oy_start, oy_stop):
            y = 2 * oy
            for x in range(w):
                v = np.uint32(0)
                for i in range(4):
                    sy = y - 1 + i
                    if sy < 0:
                        sy = 0
                    elif sy >= h:
                        sy = h - 1
                    v += np.uint32(src[sy, x]) * np.uint32(_WEIGHTS[i])
                scratch[x] = (v + np.uint32(57)) // np.uint32(115)
            for ox in range(ow):
                v = np.uint32(0)
                for i in range(4):
                    sx = 2 * ox - 1 + i
                    if sx < 0:
                        sx = 0
                    elif sx >= w:
                        sx = w - 1
                    v += scratch[sx] * np.uint32(_WEIGHTS[i])
                out[oy, ox] = np.uint16((v + np.uint32(57)) // np.uint32(115))
        return out

    @njit(cache=True, nogil=True)
    def _nb_down_half(src):
        h, w = src.shape
        oh, ow = (h + 1) // 2, (w + 1) // 2
        out = np.empty((oh, ow), np.uint16)
        return _nb_down_half_rows(src, out, 0, oh)

if _HAVE_NUMBA:
    @njit(cache=True, nogil=True)
    def _nb_gray16_u16_rows(image, out, y_start, y_stop):
        w = image.shape[1]
        if image.ndim == 2:
            for y in range(y_start, y_stop):
                for x in range(w):
                    out[y, x] = np.uint16(image[y, x])
        elif image.shape[2] >= 3:
            for y in range(y_start, y_stop):
                for x in range(w):
                    value = (
                        np.uint32(image[y, x, 0])
                        + np.uint32(image[y, x, 1])
                        + np.uint32(image[y, x, 2])
                    ) // np.uint32(3)
                    out[y, x] = np.uint16(value)
        else:
            take = image.shape[2]
            for y in range(y_start, y_stop):
                for x in range(w):
                    value = np.uint32(0)
                    for c in range(take):
                        value += np.uint32(image[y, x, c])
                    out[y, x] = np.uint16(value // np.uint32(take))
        return out

    @njit(cache=True, nogil=True)
    def _nb_gray16_u16(image, out):
        return _nb_gray16_u16_rows(image, out, 0, image.shape[0])

if _HAVE_NUMBA:
    @njit(cache=True, nogil=True)
    def _axis(coord, length):
        extent = length
        invalid = coord >= np.float32(extent) or coord < np.float32(0)
        if coord >= np.float32(extent - 1):
            coord = np.float32(extent) - np.float32(1.0010000467300415)
        if coord < np.float32(0):
            coord = np.float32(0)
        index = np.int32(coord)
        frac = np.uint32((coord - np.float32(index)) * np.float32(256))
        lo = index
        if lo < 0:
            lo = 0
        elif lo > length - 1:
            lo = length - 1
        hi = index + 1
        if hi < 0:
            hi = 0
        elif hi > length - 1:
            hi = length - 1
        return lo, hi, frac, invalid

    @njit(cache=True, nogil=True)
    def _nb_warp_u16(image, dx, dy):
        out = np.empty_like(image)
        _nb_warp_int_rows(image, out, dx, dy, 0, image.shape[0])
        return out

    @njit(cache=True, nogil=True)
    def _map_uv_row(y, cy, p1c, p0c):
        dy_ = np.float32(np.float32(y) - cy)
        return np.float32(dy_ * p1c), np.float32(dy_ * p0c)

    @njit(cache=True, nogil=True)
    def _map_uv_col(x, cx, p08, p18, tx, ty, v_dy, u_dy):
        dx_ = np.float32(np.float32(x) - cx)
        v = np.float32(np.float32(v_dy - np.float32(dx_ * p08)) + ty)
        u = np.float32(np.float32(np.float32(dx_ * p18) + u_dy) + tx)
        return u, v

    @njit(cache=True, nogil=True)
    def _bilinear_u32_2d(image, iy, jy, ix, jx, gx, fx, gy, fy):
        top = np.uint32(image[iy, ix]) * gx + np.uint32(image[iy, jx]) * fx
        bottom = np.uint32(image[jy, ix]) * gx + np.uint32(image[jy, jx]) * fx
        return (top * gy + bottom * fy) >> np.uint32(16)

    @njit(cache=True, nogil=True)
    def _nb_warp_int_rows(image, out, dx, dy, y_start, y_stop):
        """Translate uint8/uint16 2D or 3D image rows using exact Q8.8 weights."""
        h, w = image.shape[:2]
        dx32 = np.float32(dx)
        dy32 = np.float32(dy)
        ix = np.empty(w, np.int32)
        jx = np.empty(w, np.int32)
        fx = np.empty(w, np.uint32)
        inv_x = np.empty(w, np.bool_)
        for x in range(w):
            ix_v, jx_v, fx_v, inv_v = _axis(np.float32(x) - dx32, w)
            ix[x] = ix_v
            jx[x] = jx_v
            fx[x] = fx_v
            inv_x[x] = inv_v
        if image.ndim == 2:
            for y in range(y_start, y_stop):
                iy, jy, fy, inv_y = _axis(np.float32(y) - dy32, h)
                if inv_y:
                    for x in range(w):
                        out[y, x] = np.uint32(0)
                    continue
                gy = np.uint32(256) - fy
                for x in range(w):
                    if inv_x[x]:
                        out[y, x] = np.uint32(0)
                    else:
                        fxx = fx[x]
                        out[y, x] = _bilinear_u32_2d(
                            image, iy, jy, ix[x], jx[x], np.uint32(256) - fxx, fxx, gy, fy
                        )
        elif image.shape[2] == 3:
            for y in range(y_start, y_stop):
                iy, jy, fy, inv_y = _axis(np.float32(y) - dy32, h)
                if inv_y:
                    for x in range(w):
                        out[y, x, 0] = 0
                        out[y, x, 1] = 0
                        out[y, x, 2] = 0
                    continue
                gy = np.uint32(256) - fy
                for x in range(w):
                    if inv_x[x]:
                        out[y, x, 0] = 0
                        out[y, x, 1] = 0
                        out[y, x, 2] = 0
                    else:
                        fxx = fx[x]
                        gxx = np.uint32(256) - fxx
                        ixx = ix[x]
                        jxx = jx[x]
                        t0 = np.uint32(image[iy, ixx, 0]) * gxx + np.uint32(image[iy, jxx, 0]) * fxx
                        b0 = np.uint32(image[jy, ixx, 0]) * gxx + np.uint32(image[jy, jxx, 0]) * fxx
                        out[y, x, 0] = (t0 * gy + b0 * fy) >> np.uint32(16)
                        t1 = np.uint32(image[iy, ixx, 1]) * gxx + np.uint32(image[iy, jxx, 1]) * fxx
                        b1 = np.uint32(image[jy, ixx, 1]) * gxx + np.uint32(image[jy, jxx, 1]) * fxx
                        out[y, x, 1] = (t1 * gy + b1 * fy) >> np.uint32(16)
                        t2 = np.uint32(image[iy, ixx, 2]) * gxx + np.uint32(image[iy, jxx, 2]) * fxx
                        b2 = np.uint32(image[jy, ixx, 2]) * gxx + np.uint32(image[jy, jxx, 2]) * fxx
                        out[y, x, 2] = (t2 * gy + b2 * fy) >> np.uint32(16)
        else:
            channels = image.shape[2]
            for y in range(y_start, y_stop):
                iy, jy, fy, inv_y = _axis(np.float32(y) - dy32, h)
                if inv_y:
                    for x in range(w):
                        for c in range(channels):
                            out[y, x, c] = np.uint32(0)
                    continue
                gy = np.uint32(256) - fy
                for x in range(w):
                    if inv_x[x]:
                        for c in range(channels):
                            out[y, x, c] = np.uint32(0)
                    else:
                        fxx = fx[x]
                        gxx = np.uint32(256) - fxx
                        ixx = ix[x]
                        jxx = jx[x]
                        for c in range(channels):
                            top = np.uint32(image[iy, ixx, c]) * gxx + np.uint32(image[iy, jxx, c]) * fxx
                            bottom = np.uint32(image[jy, ixx, c]) * gxx + np.uint32(image[jy, jxx, c]) * fxx
                            out[y, x, c] = (top * gy + bottom * fy) >> np.uint32(16)

    @njit(cache=True, nogil=True)
    def _nb_warp_int_rows_bgr_rev(src_rgb, out, dx, dy, y_start, y_stop):
        """Translate C-contiguous RGB view into C-contiguous BGR output with exact Q8.8 weights."""
        h, w = src_rgb.shape[:2]
        dx32 = np.float32(dx)
        dy32 = np.float32(dy)
        ix = np.empty(w, np.int32)
        jx = np.empty(w, np.int32)
        fx = np.empty(w, np.uint32)
        inv_x = np.empty(w, np.bool_)
        for x in range(w):
            ix_v, jx_v, fx_v, inv_v = _axis(np.float32(x) - dx32, w)
            ix[x] = ix_v
            jx[x] = jx_v
            fx[x] = fx_v
            inv_x[x] = inv_v
        for y in range(y_start, y_stop):
            iy, jy, fy, inv_y = _axis(np.float32(y) - dy32, h)
            if inv_y:
                for x in range(w):
                    out[y, x, 0] = 0
                    out[y, x, 1] = 0
                    out[y, x, 2] = 0
                continue
            gy = np.uint32(256) - fy
            for x in range(w):
                if inv_x[x]:
                    out[y, x, 0] = 0
                    out[y, x, 1] = 0
                    out[y, x, 2] = 0
                else:
                    fxx = fx[x]
                    gxx = np.uint32(256) - fxx
                    ixx = ix[x]
                    jxx = jx[x]
                    t0 = np.uint32(src_rgb[iy, ixx, 2]) * gxx + np.uint32(src_rgb[iy, jxx, 2]) * fxx
                    b0 = np.uint32(src_rgb[jy, ixx, 2]) * gxx + np.uint32(src_rgb[jy, jxx, 2]) * fxx
                    out[y, x, 0] = (t0 * gy + b0 * fy) >> np.uint32(16)
                    t1 = np.uint32(src_rgb[iy, ixx, 1]) * gxx + np.uint32(src_rgb[iy, jxx, 1]) * fxx
                    b1 = np.uint32(src_rgb[jy, ixx, 1]) * gxx + np.uint32(src_rgb[jy, jxx, 1]) * fxx
                    out[y, x, 1] = (t1 * gy + b1 * fy) >> np.uint32(16)
                    t2 = np.uint32(src_rgb[iy, ixx, 0]) * gxx + np.uint32(src_rgb[iy, jxx, 0]) * fxx
                    b2 = np.uint32(src_rgb[jy, ixx, 0]) * gxx + np.uint32(src_rgb[jy, jxx, 0]) * fxx
                    out[y, x, 2] = (t2 * gy + b2 * fy) >> np.uint32(16)

    @njit(cache=True, nogil=True)
    def _nb_fast_downsize_u16_rows(img, out, oy_start, oy_stop):
        """2x2 mean (sum+2)>>2 matching fast_downsize with odd edge replication."""
        h, w = img.shape
        ow = out.shape[1]
        for oy in range(oy_start, oy_stop):
            y0 = 2 * oy
            y1 = y0 + 1 if y0 + 1 < h else h - 1
            for ox in range(ow):
                x0 = 2 * ox
                x1 = x0 + 1 if x0 + 1 < w else w - 1
                total = (
                    np.uint32(img[y0, x0])
                    + np.uint32(img[y0, x1])
                    + np.uint32(img[y1, x0])
                    + np.uint32(img[y1, x1])
                    + np.uint32(2)
                ) >> np.uint32(2)
                out[oy, ox] = np.uint16(total)

    @njit(cache=True, nogil=True)
    def _nb_score_chunk_u16_slice(ref, img_b, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy,
                                  idx, k_start, k_stop, ssum, ssq, cnt):
        """Score a slice [k_start, k_stop) of alive candidates over a uint16 level."""
        h, w = img_b.shape
        h_f = np.float32(h)
        w_f = np.float32(w)
        nyy = ys.shape[0]
        nxx = xs.shape[0]
        for k in range(k_start, k_stop):
            c_idx = idx[k]
            p08_k = p08[c_idx]
            p0c_k = p0c[c_idx]
            p18_k = p18[c_idx]
            p1c_k = p1c[c_idx]
            tx_k = tx[c_idx]
            ty_k = ty[c_idx]
            cx_k = cx[c_idx]
            cy_k = cy[c_idx]
            acc_sum = np.int64(0)
            acc_sq = np.int64(0)
            acc_cnt = np.int64(0)
            for iy_i in range(nyy):
                v_dy, u_dy = _map_uv_row(ys[iy_i], cy_k, p1c_k, p0c_k)
                for ix_i in range(nxx):
                    u, v = _map_uv_col(xs[ix_i], cx_k, p08_k, p18_k, tx_k, ty_k, v_dy, u_dy)
                    if not (u >= np.float32(0.0) and u < w_f and v >= np.float32(0.0) and v < h_f):
                        continue
                    ix, jx, fx, _ = _axis(u, w)
                    iy, jy, fy, _ = _axis(v, h)
                    gx = np.uint32(256) - fx
                    gy = np.uint32(256) - fy
                    val = _bilinear_u32_2d(img_b, iy, jy, ix, jx, gx, fx, gy, fy)
                    diff = np.int64(ref[iy_i, ix_i]) - np.int64(val)
                    acc_sum += diff
                    acc_sq += diff * diff
                    acc_cnt += 1
            ssum[c_idx] = acc_sum
            ssq[c_idx] = acc_sq
            cnt[c_idx] = acc_cnt

    @njit(cache=True, nogil=True)
    def _nb_score_chunk_u16(ref, img_b, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy,
                            idx, ssum, ssq, cnt):
        """Score alive similarity candidates over a uint16 level with zero temporaries."""
        _nb_score_chunk_u16_slice(
            ref, img_b, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy,
            idx, 0, idx.shape[0], ssum, ssq, cnt,
        )

    @njit(cache=True, nogil=True)
    def _nb_warp_alignment_int_rows(image, out, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop):
        """Similarity warp for uint8/uint16 2D or 3D images matching _warp_alignment."""
        h, w = image.shape[:2]
        if image.ndim == 2:
            for y in range(y_start, y_stop):
                v_dy, u_dy = _map_uv_row(np.float32(y), cy, p1c, p0c)
                for x in range(w):
                    u, v = _map_uv_col(np.float32(x), cx, p08, p18, tx, ty, v_dy, u_dy)
                    ix, jx, fx, inv_x = _axis(u, w)
                    iy, jy, fy, inv_y = _axis(v, h)
                    if inv_x or inv_y:
                        out[y, x] = np.uint32(0)
                    else:
                        out[y, x] = _bilinear_u32_2d(
                            image, iy, jy, ix, jx,
                            np.uint32(256) - fx, fx, np.uint32(256) - fy, fy,
                        )
        elif image.shape[2] == 3:
            for y in range(y_start, y_stop):
                v_dy, u_dy = _map_uv_row(np.float32(y), cy, p1c, p0c)
                for x in range(w):
                    u, v = _map_uv_col(np.float32(x), cx, p08, p18, tx, ty, v_dy, u_dy)
                    ix, jx, fx, inv_x = _axis(u, w)
                    iy, jy, fy, inv_y = _axis(v, h)
                    if inv_x or inv_y:
                        out[y, x, 0] = 0
                        out[y, x, 1] = 0
                        out[y, x, 2] = 0
                    else:
                        gx = np.uint32(256) - fx
                        gy = np.uint32(256) - fy
                        t0 = np.uint32(image[iy, ix, 0]) * gx + np.uint32(image[iy, jx, 0]) * fx
                        b0 = np.uint32(image[jy, ix, 0]) * gx + np.uint32(image[jy, jx, 0]) * fx
                        out[y, x, 0] = (t0 * gy + b0 * fy) >> np.uint32(16)
                        t1 = np.uint32(image[iy, ix, 1]) * gx + np.uint32(image[iy, jx, 1]) * fx
                        b1 = np.uint32(image[jy, ix, 1]) * gx + np.uint32(image[jy, jx, 1]) * fx
                        out[y, x, 1] = (t1 * gy + b1 * fy) >> np.uint32(16)
                        t2 = np.uint32(image[iy, ix, 2]) * gx + np.uint32(image[iy, jx, 2]) * fx
                        b2 = np.uint32(image[jy, ix, 2]) * gx + np.uint32(image[jy, jx, 2]) * fx
                        out[y, x, 2] = (t2 * gy + b2 * fy) >> np.uint32(16)
        else:
            channels = image.shape[2]
            for y in range(y_start, y_stop):
                v_dy, u_dy = _map_uv_row(np.float32(y), cy, p1c, p0c)
                for x in range(w):
                    u, v = _map_uv_col(np.float32(x), cx, p08, p18, tx, ty, v_dy, u_dy)
                    ix, jx, fx, inv_x = _axis(u, w)
                    iy, jy, fy, inv_y = _axis(v, h)
                    if inv_x or inv_y:
                        for c in range(channels):
                            out[y, x, c] = np.uint32(0)
                    else:
                        gx = np.uint32(256) - fx
                        gy = np.uint32(256) - fy
                        for c in range(channels):
                            top = np.uint32(image[iy, ix, c]) * gx + np.uint32(image[iy, jx, c]) * fx
                            bottom = np.uint32(image[jy, ix, c]) * gx + np.uint32(image[jy, jx, c]) * fx
                            out[y, x, c] = (top * gy + bottom * fy) >> np.uint32(16)

    @njit(cache=True, nogil=True)
    def _nb_warp_alignment_int_rows_bgr_rev(src_rgb, out, tx, ty, cx, cy, p08, p0c, p18, p1c,
                                            y_start, y_stop):
        """Similarity warp from C-contiguous RGB view into C-contiguous BGR output."""
        h, w = src_rgb.shape[:2]
        for y in range(y_start, y_stop):
            v_dy, u_dy = _map_uv_row(np.float32(y), cy, p1c, p0c)
            for x in range(w):
                u, v = _map_uv_col(np.float32(x), cx, p08, p18, tx, ty, v_dy, u_dy)
                ix, jx, fx, inv_x = _axis(u, w)
                iy, jy, fy, inv_y = _axis(v, h)
                if inv_x or inv_y:
                    out[y, x, 0] = 0
                    out[y, x, 1] = 0
                    out[y, x, 2] = 0
                else:
                    gx = np.uint32(256) - fx
                    gy = np.uint32(256) - fy
                    t0 = np.uint32(src_rgb[iy, ix, 2]) * gx + np.uint32(src_rgb[iy, jx, 2]) * fx
                    b0 = np.uint32(src_rgb[jy, ix, 2]) * gx + np.uint32(src_rgb[jy, jx, 2]) * fx
                    out[y, x, 0] = (t0 * gy + b0 * fy) >> np.uint32(16)
                    t1 = np.uint32(src_rgb[iy, ix, 1]) * gx + np.uint32(src_rgb[iy, jx, 1]) * fx
                    b1 = np.uint32(src_rgb[jy, ix, 1]) * gx + np.uint32(src_rgb[jy, jx, 1]) * fx
                    out[y, x, 1] = (t1 * gy + b1 * fy) >> np.uint32(16)
                    t2 = np.uint32(src_rgb[iy, ix, 0]) * gx + np.uint32(src_rgb[iy, jx, 0]) * fx
                    b2 = np.uint32(src_rgb[jy, ix, 0]) * gx + np.uint32(src_rgb[jy, jx, 0]) * fx
                    out[y, x, 2] = (t2 * gy + b2 * fy) >> np.uint32(16)

    @njit(cache=True, nogil=True)
    def _nb_warp_alignment_float_rows(image, out, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop):
        """Similarity warp for internal float32/float64 2D or 3D images."""
        h, w = image.shape[:2]
        inv_scale = np.float64(65536.0)
        if image.ndim == 2:
            for y in range(y_start, y_stop):
                v_dy, u_dy = _map_uv_row(np.float32(y), cy, p1c, p0c)
                for x in range(w):
                    u, v = _map_uv_col(np.float32(x), cx, p08, p18, tx, ty, v_dy, u_dy)
                    ix, jx, fx, inv_x = _axis(u, w)
                    iy, jy, fy, inv_y = _axis(v, h)
                    if inv_x or inv_y:
                        out[y, x] = np.float64(0.0)
                    else:
                        fgx = np.float64(np.uint32(256) - fx)
                        ffx = np.float64(fx)
                        fgy = np.float64(np.uint32(256) - fy)
                        ffy = np.float64(fy)
                        top = np.float64(image[iy, ix]) * fgx + np.float64(image[iy, jx]) * ffx
                        bottom = np.float64(image[jy, ix]) * fgx + np.float64(image[jy, jx]) * ffx
                        out[y, x] = (top * fgy + bottom * ffy) / inv_scale
        else:
            channels = image.shape[2]
            for y in range(y_start, y_stop):
                v_dy, u_dy = _map_uv_row(np.float32(y), cy, p1c, p0c)
                for x in range(w):
                    u, v = _map_uv_col(np.float32(x), cx, p08, p18, tx, ty, v_dy, u_dy)
                    ix, jx, fx, inv_x = _axis(u, w)
                    iy, jy, fy, inv_y = _axis(v, h)
                    if inv_x or inv_y:
                        for c in range(channels):
                            out[y, x, c] = np.float64(0.0)
                    else:
                        fgx = np.float64(np.uint32(256) - fx)
                        ffx = np.float64(fx)
                        fgy = np.float64(np.uint32(256) - fy)
                        ffy = np.float64(fy)
                        for c in range(channels):
                            top = np.float64(image[iy, ix, c]) * fgx + np.float64(image[iy, jx, c]) * ffx
                            bottom = np.float64(image[jy, ix, c]) * fgx + np.float64(image[jy, jx, c]) * ffx
                            out[y, x, c] = (top * fgy + bottom * ffy) / inv_scale

    @njit(cache=True, nogil=True)
    def _nb_valid_mask_rows(mask, h, w, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop):
        """Geometric validity mask matching alignment_valid_mask float32 boundaries."""
        h_f = np.float32(h)
        w_f = np.float32(w)
        for y in range(y_start, y_stop):
            v_dy, u_dy = _map_uv_row(np.float32(y), cy, p1c, p0c)
            for x in range(w):
                u, v = _map_uv_col(np.float32(x), cx, p08, p18, tx, ty, v_dy, u_dy)
                mask[y, x] = (
                    u >= np.float32(0.0) and u < w_f
                    and v >= np.float32(0.0) and v < h_f
                )


if _HAVE_NUMBA:
    @njit(cache=True, nogil=True)
    def _axis_pad(coord, length, pad):
        extent = length + 2 * pad
        invalid = coord >= np.float32(extent) or coord < np.float32(0)
        if coord >= np.float32(extent - 1):
            coord = np.float32(extent) - np.float32(1.0010000467300415)
        if coord < np.float32(0):
            coord = np.float32(0)
        index = np.int32(coord)
        frac = np.uint32((coord - np.float32(index)) * np.float32(256))
        lo = index - pad
        if lo < 0:
            lo = 0
        elif lo > length - 1:
            lo = length - 1
        hi = index + 1 - pad
        if hi < 0:
            hi = 0
        elif hi > length - 1:
            hi = length - 1
        return lo, hi, frac, invalid
if _HAVE_NUMBA:
    @njit(cache=True, nogil=True)
    def _nb_refine_level(a, b, c_dx, c_dy, half, step, crop):
        h, w = a.shape
        pad = np.int32(np.ceil(max(abs(c_dx), abs(c_dy), np.float32(1.0)) + half)) + 1
        n = min(crop, h, w)
        cy, cx = (h - n) // 2, (w - n) // 2
        count = np.int32(round(2.0 * half / step)) | 1
        centre = (count - 1) // 2
        best_score = np.float32(np.finfo(np.float32).max)
        best_dx = np.float32(0.0)
        best_dy = np.float32(0.0)
        found = False
        for oy in range(count):
            ody = np.float32((oy - centre) * step)
            dy = np.float32(c_dy + ody)
            for ox in range(count):
                odx = np.float32((ox - centre) * step)
                dx = np.float32(c_dx + odx)
                ssum = np.float64(0.0)
                ssq = np.float64(0.0)
                for yy in range(cy, cy + n):
                    ycoord = np.float32(np.float32(yy) + np.float32(pad) - dy)
                    iy, jy, fy, inv_y = _axis_pad(ycoord, h, pad)
                    gy = np.uint32(256) - fy
                    for xx in range(cx, cx + n):
                        xcoord = np.float32(np.float32(xx) + np.float32(pad) - dx)
                        ix, jx, fx, inv_x = _axis_pad(xcoord, w, pad)
                        gx = np.uint32(256) - fx
                        if inv_y or inv_x:
                            sample = 0
                        else:
                            top = np.uint32(b[iy, ix]) * gx + np.uint32(b[iy, jx]) * fx
                            bottom = np.uint32(b[jy, ix]) * gx + np.uint32(b[jy, jx]) * fx
                            sample = np.uint32((top * gy + bottom * fy) >> np.uint32(16))
                        diff = np.float64(a[yy, xx]) - np.float64(sample)
                        ssum += diff
                        ssq += diff * diff
                total = np.float64(n * n)
                mean = ssum / total
                score = np.float64(np.sqrt(max(ssq / total - mean * mean, 0.0)))
                if score < np.float64(best_score):
                    best_score = np.float32(score)
                    best_dx = dx
                    best_dy = dy
                    found = True
        return best_dx, best_dy, found

    @njit(cache=True, nogil=True, fastmath=False)
    def _nb_score_grid_4d_slice(ref, img_b, xs, ys, tx0, ty0, cx_tab, cy_tab, p08_tab, p18_tab,
                                n0, n1, n2, n3, k_start, k_stop, out_best_k, out_best_sc, slot):
        """Fused candidate generation, bbox culling, bilinear scoring, and min reduction."""
        h, w = img_b.shape
        h_f = np.float32(h)
        w_f = np.float32(w)
        zero_f = np.float32(0.0)
        nyy = ys.shape[0]
        nxx = xs.shape[0]
        x_lo = xs[0]
        x_hi = xs[nxx - 1]
        y_lo = ys[0]
        y_hi = ys[nyy - 1]

        best_sc = np.float32(np.inf)
        best_k = np.int64(-1)

        for k in range(k_start, k_stop):
            rem = k
            t0 = rem % n0
            rem //= n0
            t1 = rem % n1
            rem //= n1
            t2 = rem % n2
            t3 = rem // n2

            cx_k = cx_tab[t0]
            cy_k = cy_tab[t1]
            p08_k = p08_tab[t3, t2]
            p18_k = p18_tab[t3, t2]
            p0c_k = p08_k
            p1c_k = p18_k

            c_u = np.float32(np.float32(tx0 - np.float32(cx_k * p18_k)) - np.float32(cy_k * p0c_k))
            c_v = np.float32(np.float32(ty0 - np.float32(cy_k * p1c_k)) - np.float32(cx_k * p08_k))

            aux0 = np.float32(p18_k * x_lo)
            aux1 = np.float32(p18_k * x_hi)
            buy0 = np.float32(p0c_k * y_lo)
            buy1 = np.float32(p0c_k * y_hi)
            u_lo = np.float32(np.float32((aux0 if aux0 < aux1 else aux1) + (buy0 if buy0 < buy1 else buy1)) + c_u)
            u_hi = np.float32(np.float32((aux0 if aux0 > aux1 else aux1) + (buy0 if buy0 > buy1 else buy1)) + c_u)

            avx0 = np.float32(p1c_k * x_lo)
            avx1 = np.float32(p1c_k * x_hi)
            bvy0 = np.float32(p08_k * y_lo)
            bvy1 = np.float32(p08_k * y_hi)
            v_lo = np.float32(np.float32((avx0 if avx0 < avx1 else avx1) + (bvy0 if bvy0 < bvy1 else bvy1)) + c_v)
            v_hi = np.float32(np.float32((avx0 if avx0 > avx1 else avx1) + (bvy0 if bvy0 > bvy1 else bvy1)) + c_v)

            if not (u_hi >= zero_f and u_lo < w_f and v_hi >= zero_f and v_lo < h_f):
                continue

            acc_sum = np.int64(0)
            acc_sq = np.int64(0)
            acc_cnt = np.int64(0)
            for iy_i in range(nyy):
                v_dy, u_dy = _map_uv_row(ys[iy_i], cy_k, p1c_k, p0c_k)
                for ix_i in range(nxx):
                    u, v = _map_uv_col(xs[ix_i], cx_k, p08_k, p18_k, tx0, ty0, v_dy, u_dy)
                    if not (u >= zero_f and u < w_f and v >= zero_f and v < h_f):
                        continue
                    ix, jx, fx, _ = _axis(u, w)
                    iy, jy, fy, _ = _axis(v, h)
                    gx = np.uint32(256) - fx
                    gy = np.uint32(256) - fy
                    val = _bilinear_u32_2d(img_b, iy, jy, ix, jx, gx, fx, gy, fy)
                    diff = np.int64(ref[iy_i, ix_i]) - np.int64(val)
                    acc_sum += diff
                    acc_sq += diff * diff
                    acc_cnt += 1

            if acc_cnt >= 2:
                denom = np.float64(np.float32(acc_cnt))
                msq_f = np.float32(np.float64(acc_sq) / denom)
                msu_f = np.float32(np.float64(acc_sum) / denom)
                var_f = np.float32(msq_f - np.float32(msu_f * msu_f))
                if var_f >= zero_f:
                    sc = np.sqrt(var_f)
                    if sc < best_sc:
                        best_sc = sc
                        best_k = k

        out_best_k[slot] = best_k
        out_best_sc[slot] = best_sc

    @njit(cache=True, nogil=True)
    def _nb_perspective_has_rounding_edge(inverse, x0, x1, y0, y1, bins, tolerance):
        for y in range(y0, y1):
            nx0 = inverse[0, 1] * y + inverse[0, 2]
            ny0 = inverse[1, 1] * y + inverse[1, 2]
            denom0 = inverse[2, 1] * y + inverse[2, 2]
            for x in range(x0, x1):
                scale = bins / (denom0 + inverse[2, 0] * x)
                nx = (nx0 + inverse[0, 0] * x) * scale
                ny = (ny0 + inverse[1, 0] * x) * scale
                if (abs(nx - np.floor(nx) - 0.5) <= tolerance
                        or abs(ny - np.floor(ny) - 0.5) <= tolerance):
                    return True
        return False


    @njit(cache=True, nogil=True)
    def _nb_warp_alignment_region_int_rows(image, out, src_h, src_w, src_y0, x_off, y_off,
                                           tx, ty, cx, cy, p08, p0c, p18, p1c, oy_start, oy_stop):
        """Region similarity warp for uint8/uint16 2D or 3D image/band buffers."""
        rw = out.shape[1]
        if out.ndim == 2:
            for oy in range(oy_start, oy_stop):
                v_dy, u_dy = _map_uv_row(np.float32(y_off + oy), cy, p1c, p0c)
                for ox in range(rw):
                    u, v = _map_uv_col(np.float32(x_off + ox), cx, p08, p18, tx, ty, v_dy, u_dy)
                    ix, jx, fx, inv_x = _axis(u, src_w)
                    iy, jy, fy, inv_y = _axis(v, src_h)
                    if inv_x or inv_y:
                        out[oy, ox] = 0
                    else:
                        out[oy, ox] = _bilinear_u32_2d(
                            image, iy - src_y0, jy - src_y0, ix, jx,
                            np.uint32(256) - fx, fx, np.uint32(256) - fy, fy,
                        )
        elif out.shape[2] == 3:
            for oy in range(oy_start, oy_stop):
                v_dy, u_dy = _map_uv_row(np.float32(y_off + oy), cy, p1c, p0c)
                for ox in range(rw):
                    u, v = _map_uv_col(np.float32(x_off + ox), cx, p08, p18, tx, ty, v_dy, u_dy)
                    ix, jx, fx, inv_x = _axis(u, src_w)
                    iy, jy, fy, inv_y = _axis(v, src_h)
                    if inv_x or inv_y:
                        out[oy, ox, 0] = 0
                        out[oy, ox, 1] = 0
                        out[oy, ox, 2] = 0
                    else:
                        gx = np.uint32(256) - fx
                        gy = np.uint32(256) - fy
                        iy_b = iy - src_y0
                        jy_b = jy - src_y0
                        t0 = np.uint32(image[iy_b, ix, 0]) * gx + np.uint32(image[iy_b, jx, 0]) * fx
                        b0 = np.uint32(image[jy_b, ix, 0]) * gx + np.uint32(image[jy_b, jx, 0]) * fx
                        out[oy, ox, 0] = (t0 * gy + b0 * fy) >> np.uint32(16)
                        t1 = np.uint32(image[iy_b, ix, 1]) * gx + np.uint32(image[iy_b, jx, 1]) * fx
                        b1 = np.uint32(image[jy_b, ix, 1]) * gx + np.uint32(image[jy_b, jx, 1]) * fx
                        out[oy, ox, 1] = (t1 * gy + b1 * fy) >> np.uint32(16)
                        t2 = np.uint32(image[iy_b, ix, 2]) * gx + np.uint32(image[iy_b, jx, 2]) * fx
                        b2 = np.uint32(image[jy_b, ix, 2]) * gx + np.uint32(image[jy_b, jx, 2]) * fx
                        out[oy, ox, 2] = (t2 * gy + b2 * fy) >> np.uint32(16)
        else:
            channels = out.shape[2]
            for oy in range(oy_start, oy_stop):
                v_dy, u_dy = _map_uv_row(np.float32(y_off + oy), cy, p1c, p0c)
                for ox in range(rw):
                    u, v = _map_uv_col(np.float32(x_off + ox), cx, p08, p18, tx, ty, v_dy, u_dy)
                    ix, jx, fx, inv_x = _axis(u, src_w)
                    iy, jy, fy, inv_y = _axis(v, src_h)
                    if inv_x or inv_y:
                        for c in range(channels):
                            out[oy, ox, c] = 0
                    else:
                        gx = np.uint32(256) - fx
                        gy = np.uint32(256) - fy
                        iy_b = iy - src_y0
                        jy_b = jy - src_y0
                        for c in range(channels):
                            top = np.uint32(image[iy_b, ix, c]) * gx + np.uint32(image[iy_b, jx, c]) * fx
                            bottom = np.uint32(image[jy_b, ix, c]) * gx + np.uint32(image[jy_b, jx, c]) * fx
                            out[oy, ox, c] = (top * gy + bottom * fy) >> np.uint32(16)

    @njit(cache=True, nogil=True)
    def _nb_warp_alignment_region_int_rows_bgr_rev(src_rgb, out, src_h, src_w, src_y0, x_off, y_off,
                                                   tx, ty, cx, cy, p08, p0c, p18, p1c, oy_start, oy_stop):
        """Region similarity warp from C-contiguous RGB band/view into C-contiguous BGR output."""
        rw = out.shape[1]
        for oy in range(oy_start, oy_stop):
            v_dy, u_dy = _map_uv_row(np.float32(y_off + oy), cy, p1c, p0c)
            for ox in range(rw):
                u, v = _map_uv_col(np.float32(x_off + ox), cx, p08, p18, tx, ty, v_dy, u_dy)
                ix, jx, fx, inv_x = _axis(u, src_w)
                iy, jy, fy, inv_y = _axis(v, src_h)
                if inv_x or inv_y:
                    out[oy, ox, 0] = 0
                    out[oy, ox, 1] = 0
                    out[oy, ox, 2] = 0
                else:
                    gx = np.uint32(256) - fx
                    gy = np.uint32(256) - fy
                    iy_b = iy - src_y0
                    jy_b = jy - src_y0
                    t0 = np.uint32(src_rgb[iy_b, ix, 2]) * gx + np.uint32(src_rgb[iy_b, jx, 2]) * fx
                    b0 = np.uint32(src_rgb[jy_b, ix, 2]) * gx + np.uint32(src_rgb[jy_b, jx, 2]) * fx
                    out[oy, ox, 0] = (t0 * gy + b0 * fy) >> np.uint32(16)
                    t1 = np.uint32(src_rgb[iy_b, ix, 1]) * gx + np.uint32(src_rgb[iy_b, jx, 1]) * fx
                    b1 = np.uint32(src_rgb[jy_b, ix, 1]) * gx + np.uint32(src_rgb[jy_b, jx, 1]) * fx
                    out[oy, ox, 1] = (t1 * gy + b1 * fy) >> np.uint32(16)
                    t2 = np.uint32(src_rgb[iy_b, ix, 0]) * gx + np.uint32(src_rgb[iy_b, jx, 0]) * fx
                    b2 = np.uint32(src_rgb[jy_b, ix, 0]) * gx + np.uint32(src_rgb[jy_b, jx, 0]) * fx
                    out[oy, ox, 2] = (t2 * gy + b2 * fy) >> np.uint32(16)

    @njit(cache=True, nogil=True)
    def _nb_warp_alignment_region_float_rows(image, out, src_h, src_w, src_y0, x_off, y_off,
                                             tx, ty, cx, cy, p08, p0c, p18, p1c, oy_start, oy_stop):
        """Region similarity warp for internal float32/float64 2D or 3D images."""
        rw = out.shape[1]
        inv_scale = np.float64(65536.0)
        if out.ndim == 2:
            for oy in range(oy_start, oy_stop):
                v_dy, u_dy = _map_uv_row(np.float32(y_off + oy), cy, p1c, p0c)
                for ox in range(rw):
                    u, v = _map_uv_col(np.float32(x_off + ox), cx, p08, p18, tx, ty, v_dy, u_dy)
                    ix, jx, fx, inv_x = _axis(u, src_w)
                    iy, jy, fy, inv_y = _axis(v, src_h)
                    if inv_x or inv_y:
                        out[oy, ox] = np.float64(0.0)
                    else:
                        fgx = np.float64(np.uint32(256) - fx)
                        ffx = np.float64(fx)
                        fgy = np.float64(np.uint32(256) - fy)
                        ffy = np.float64(fy)
                        iy_b = iy - src_y0
                        jy_b = jy - src_y0
                        top = np.float64(image[iy_b, ix]) * fgx + np.float64(image[iy_b, jx]) * ffx
                        bottom = np.float64(image[jy_b, ix]) * fgx + np.float64(image[jy_b, jx]) * ffx
                        out[oy, ox] = (top * fgy + bottom * ffy) / inv_scale
        else:
            channels = out.shape[2]
            for oy in range(oy_start, oy_stop):
                v_dy, u_dy = _map_uv_row(np.float32(y_off + oy), cy, p1c, p0c)
                for ox in range(rw):
                    u, v = _map_uv_col(np.float32(x_off + ox), cx, p08, p18, tx, ty, v_dy, u_dy)
                    ix, jx, fx, inv_x = _axis(u, src_w)
                    iy, jy, fy, inv_y = _axis(v, src_h)
                    if inv_x or inv_y:
                        for c in range(channels):
                            out[oy, ox, c] = np.float64(0.0)
                    else:
                        fgx = np.float64(np.uint32(256) - fx)
                        ffx = np.float64(fx)
                        fgy = np.float64(np.uint32(256) - fy)
                        ffy = np.float64(fy)
                        iy_b = iy - src_y0
                        jy_b = jy - src_y0
                        for c in range(channels):
                            top = np.float64(image[iy_b, ix, c]) * fgx + np.float64(image[iy_b, jx, c]) * ffx
                            bottom = np.float64(image[jy_b, ix, c]) * fgx + np.float64(image[jy_b, jx, c]) * ffx
                            out[oy, ox, c] = (top * fgy + bottom * ffy) / inv_scale

    @njit(cache=True, nogil=True)
    def _nb_warp_translation_region_int_rows(image, out, src_h, src_w, src_y0, x_off, y_off,
                                             dx, dy, oy_start, oy_stop):
        """Region translation warp for uint8/uint16 2D or 3D image/band buffers."""
        rw = out.shape[1]
        dx32 = np.float32(dx)
        dy32 = np.float32(dy)
        ix = np.empty(rw, np.int32)
        jx = np.empty(rw, np.int32)
        fx = np.empty(rw, np.uint32)
        inv_x = np.empty(rw, np.bool_)
        for ox in range(rw):
            ix_v, jx_v, fx_v, inv_v = _axis(np.float32(x_off + ox) - dx32, src_w)
            ix[ox] = ix_v
            jx[ox] = jx_v
            fx[ox] = fx_v
            inv_x[ox] = inv_v
        if out.ndim == 2:
            for oy in range(oy_start, oy_stop):
                iy, jy, fy, inv_y = _axis(np.float32(y_off + oy) - dy32, src_h)
                if inv_y:
                    for ox in range(rw):
                        out[oy, ox] = 0
                    continue
                gy = np.uint32(256) - fy
                iy_b = iy - src_y0
                jy_b = jy - src_y0
                for ox in range(rw):
                    if inv_x[ox]:
                        out[oy, ox] = 0
                    else:
                        fxx = fx[ox]
                        out[oy, ox] = _bilinear_u32_2d(
                            image, iy_b, jy_b, ix[ox], jx[ox], np.uint32(256) - fxx, fxx, gy, fy
                        )
        elif out.shape[2] == 3:
            for oy in range(oy_start, oy_stop):
                iy, jy, fy, inv_y = _axis(np.float32(y_off + oy) - dy32, src_h)
                if inv_y:
                    for ox in range(rw):
                        out[oy, ox, 0] = 0
                        out[oy, ox, 1] = 0
                        out[oy, ox, 2] = 0
                    continue
                gy = np.uint32(256) - fy
                iy_b = iy - src_y0
                jy_b = jy - src_y0
                for ox in range(rw):
                    if inv_x[ox]:
                        out[oy, ox, 0] = 0
                        out[oy, ox, 1] = 0
                        out[oy, ox, 2] = 0
                    else:
                        fxx = fx[ox]
                        gxx = np.uint32(256) - fxx
                        ixx = ix[ox]
                        jxx = jx[ox]
                        t0 = np.uint32(image[iy_b, ixx, 0]) * gxx + np.uint32(image[iy_b, jxx, 0]) * fxx
                        b0 = np.uint32(image[jy_b, ixx, 0]) * gxx + np.uint32(image[jy_b, jxx, 0]) * fxx
                        out[oy, ox, 0] = (t0 * gy + b0 * fy) >> np.uint32(16)
                        t1 = np.uint32(image[iy_b, ixx, 1]) * gxx + np.uint32(image[iy_b, jxx, 1]) * fxx
                        b1 = np.uint32(image[jy_b, ixx, 1]) * gxx + np.uint32(image[jy_b, jxx, 1]) * fxx
                        out[oy, ox, 1] = (t1 * gy + b1 * fy) >> np.uint32(16)
                        t2 = np.uint32(image[iy_b, ixx, 2]) * gxx + np.uint32(image[iy_b, jxx, 2]) * fxx
                        b2 = np.uint32(image[jy_b, ixx, 2]) * gxx + np.uint32(image[jy_b, jxx, 2]) * fxx
                        out[oy, ox, 2] = (t2 * gy + b2 * fy) >> np.uint32(16)
        else:
            channels = out.shape[2]
            for oy in range(oy_start, oy_stop):
                iy, jy, fy, inv_y = _axis(np.float32(y_off + oy) - dy32, src_h)
                if inv_y:
                    for ox in range(rw):
                        for c in range(channels):
                            out[oy, ox, c] = 0
                    continue
                gy = np.uint32(256) - fy
                iy_b = iy - src_y0
                jy_b = jy - src_y0
                for ox in range(rw):
                    if inv_x[ox]:
                        for c in range(channels):
                            out[oy, ox, c] = 0
                    else:
                        fxx = fx[ox]
                        gxx = np.uint32(256) - fxx
                        ixx = ix[ox]
                        jxx = jx[ox]
                        for c in range(channels):
                            top = np.uint32(image[iy_b, ixx, c]) * gxx + np.uint32(image[iy_b, jxx, c]) * fxx
                            bottom = np.uint32(image[jy_b, ixx, c]) * gxx + np.uint32(image[jy_b, jxx, c]) * fxx
                            out[oy, ox, c] = (top * gy + bottom * fy) >> np.uint32(16)

    @njit(cache=True, nogil=True)
    def _nb_warp_translation_region_int_rows_bgr_rev(src_rgb, out, src_h, src_w, src_y0, x_off, y_off,
                                                     dx, dy, oy_start, oy_stop):
        """Region translation warp from C-contiguous RGB band/view into C-contiguous BGR output."""
        rw = out.shape[1]
        dx32 = np.float32(dx)
        dy32 = np.float32(dy)
        ix = np.empty(rw, np.int32)
        jx = np.empty(rw, np.int32)
        fx = np.empty(rw, np.uint32)
        inv_x = np.empty(rw, np.bool_)
        for ox in range(rw):
            ix_v, jx_v, fx_v, inv_v = _axis(np.float32(x_off + ox) - dx32, src_w)
            ix[ox] = ix_v
            jx[ox] = jx_v
            fx[ox] = fx_v
            inv_x[ox] = inv_v
        for oy in range(oy_start, oy_stop):
            iy, jy, fy, inv_y = _axis(np.float32(y_off + oy) - dy32, src_h)
            if inv_y:
                for ox in range(rw):
                    out[oy, ox, 0] = 0
                    out[oy, ox, 1] = 0
                    out[oy, ox, 2] = 0
                continue
            gy = np.uint32(256) - fy
            iy_b = iy - src_y0
            jy_b = jy - src_y0
            for ox in range(rw):
                if inv_x[ox]:
                    out[oy, ox, 0] = 0
                    out[oy, ox, 1] = 0
                    out[oy, ox, 2] = 0
                else:
                    fxx = fx[ox]
                    gxx = np.uint32(256) - fxx
                    ixx = ix[ox]
                    jxx = jx[ox]
                    t0 = np.uint32(src_rgb[iy_b, ixx, 2]) * gxx + np.uint32(src_rgb[iy_b, jxx, 2]) * fxx
                    b0 = np.uint32(src_rgb[jy_b, ixx, 2]) * gxx + np.uint32(src_rgb[jy_b, jxx, 2]) * fxx
                    out[oy, ox, 0] = (t0 * gy + b0 * fy) >> np.uint32(16)
                    t1 = np.uint32(src_rgb[iy_b, ixx, 1]) * gxx + np.uint32(src_rgb[iy_b, jxx, 1]) * fxx
                    b1 = np.uint32(src_rgb[jy_b, ixx, 1]) * gxx + np.uint32(src_rgb[jy_b, jxx, 1]) * fxx
                    out[oy, ox, 1] = (t1 * gy + b1 * fy) >> np.uint32(16)
                    t2 = np.uint32(src_rgb[iy_b, ixx, 0]) * gxx + np.uint32(src_rgb[iy_b, jxx, 0]) * fxx
                    b2 = np.uint32(src_rgb[jy_b, ixx, 0]) * gxx + np.uint32(src_rgb[jy_b, jxx, 0]) * fxx
                    out[oy, ox, 2] = (t2 * gy + b2 * fy) >> np.uint32(16)


def _check_numba():
    if not _HAVE_NUMBA:
        raise RuntimeError("Neither native C++ alignment kernels nor Numba are available")


def swap_rgb_to_bgr_rows_u8(src, dst, y_start, y_stop):
    mod = _native()
    s = _as_ndarray(src)
    d = _as_ndarray(dst)
    if (mod is not None and s.dtype == np.uint8 and d.dtype == np.uint8
            and s.ndim == 3 and d.ndim == 3 and s.shape[2] >= 3 and d.shape[2] >= 3):
        mod.of_align_swap_rgb_to_bgr_rows(s, d, int(y_start), int(y_stop))
        return
    _check_numba()
    _nb_swap_rgb_to_bgr_rows_u8(s, d, int(y_start), int(y_stop))


def swap_rgb_to_bgr_rows_u16(src, dst, y_start, y_stop):
    mod = _native()
    s = _as_ndarray(src)
    d = _as_ndarray(dst)
    if (mod is not None and s.dtype == np.uint16 and d.dtype == np.uint16
            and s.ndim == 3 and d.ndim == 3 and s.shape[2] >= 3 and d.shape[2] >= 3):
        mod.of_align_swap_rgb_to_bgr_rows(s, d, int(y_start), int(y_stop))
        return
    _check_numba()
    _nb_swap_rgb_to_bgr_rows_u16(s, d, int(y_start), int(y_stop))


def swap_bgr_inplace_rows_u8(arr, y_start, y_stop):
    mod = _native()
    a = _as_ndarray(arr)
    if (mod is not None and a.dtype == np.uint8 and a.ndim == 3
            and a.shape[2] >= 3 and a.flags.writeable):
        mod.of_align_swap_bgr_inplace_rows(a, int(y_start), int(y_stop))
        return
    _check_numba()
    _nb_swap_bgr_inplace_rows_u8(a, int(y_start), int(y_stop))


def swap_bgr_inplace_rows_u16(arr, y_start, y_stop):
    mod = _native()
    a = _as_ndarray(arr)
    if (mod is not None and a.dtype == np.uint16 and a.ndim == 3
            and a.shape[2] >= 3 and a.flags.writeable):
        mod.of_align_swap_bgr_inplace_rows(a, int(y_start), int(y_stop))
        return
    _check_numba()
    _nb_swap_bgr_inplace_rows_u16(a, int(y_start), int(y_stop))


def gray16_u8_rows(image, out, y_start, y_stop):
    mod = _native()
    img = _as_ndarray(image)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype == np.uint8 and dst.dtype == np.uint16
            and img.ndim in (2, 3) and dst.ndim == 2):
        mod.of_align_gray16_rows(img, dst, int(y_start), int(y_stop))
        return out
    _check_numba()
    return _nb_gray16_u8_rows(img, dst, int(y_start), int(y_stop))


def gray16_u8(image, out=None):
    if out is None:
        out = np.empty(image.shape[:2], dtype=np.uint16)
    return gray16_u8_rows(image, out, 0, image.shape[0])


def gray16_u16_rows(image, out, y_start, y_stop):
    mod = _native()
    img = _as_ndarray(image)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype == np.uint16 and dst.dtype == np.uint16
            and img.ndim in (2, 3) and dst.ndim == 2):
        mod.of_align_gray16_rows(img, dst, int(y_start), int(y_stop))
        return out
    _check_numba()
    return _nb_gray16_u16_rows(img, dst, int(y_start), int(y_stop))


def gray16_u16(image, out=None):
    if out is None:
        out = np.empty(image.shape[:2], dtype=np.uint16)
    return gray16_u16_rows(image, out, 0, image.shape[0])


def down_half_rows(src, out, oy_start, oy_stop):
    mod = _native()
    s = _as_ndarray(src)
    d = _as_ndarray(out)
    if (mod is not None and s.dtype == np.uint16 and d.dtype == np.uint16
            and s.ndim == 2 and d.ndim == 2):
        mod.of_align_down_half_rows(s, d, int(oy_start), int(oy_stop))
        return out
    _check_numba()
    return _nb_down_half_rows(s, d, int(oy_start), int(oy_stop))


def down_half(src):
    h, w = src.shape
    oh, ow = (h + 1) // 2, (w + 1) // 2
    out = np.empty((oh, ow), np.uint16)
    return down_half_rows(src, out, 0, oh)


def fast_downsize_u16_rows(img, out, oy_start, oy_stop):
    mod = _native()
    s = _as_ndarray(img)
    d = _as_ndarray(out)
    if (mod is not None and s.dtype == np.uint16 and d.dtype == np.uint16
            and s.ndim == 2 and d.ndim == 2):
        mod.of_align_fast_downsize_u16_rows(s, d, int(oy_start), int(oy_stop))
        return
    _check_numba()
    _nb_fast_downsize_u16_rows(s, d, int(oy_start), int(oy_stop))


def warp_int_rows(image, out, dx, dy, y_start, y_stop):
    mod = _native()
    img = _as_ndarray(image)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype in (np.uint8, np.uint16) and dst.dtype in (np.uint8, np.uint16, np.uint32)
            and img.ndim in (2, 3) and dst.ndim == img.ndim):
        mod.of_align_warp_int_rows(
            img, dst, float(np.float32(dx)), float(np.float32(dy)),
            int(y_start), int(y_stop), False,
        )
        return
    _check_numba()
    _nb_warp_int_rows(img, dst, dx, dy, int(y_start), int(y_stop))


def warp_int_rows_bgr_rev(src_rgb, out, dx, dy, y_start, y_stop):
    mod = _native()
    img = _as_ndarray(src_rgb)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype in (np.uint8, np.uint16) and dst.dtype in (np.uint8, np.uint16, np.uint32)
            and img.ndim == 3 and dst.ndim == 3 and img.shape[2] >= 3 and dst.shape[2] >= 3):
        mod.of_align_warp_int_rows(
            img, dst, float(np.float32(dx)), float(np.float32(dy)),
            int(y_start), int(y_stop), True,
        )
        return
    _check_numba()
    _nb_warp_int_rows_bgr_rev(img, dst, dx, dy, int(y_start), int(y_stop))


def warp_u16(image, dx, dy):
    out = np.empty_like(_as_ndarray(image))
    warp_int_rows(image, out, dx, dy, 0, image.shape[0])
    return out


def warp_alignment_int_rows(image, out, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop):
    mod = _native()
    img = _as_ndarray(image)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype in (np.uint8, np.uint16) and dst.dtype in (np.uint8, np.uint16, np.uint32)
            and img.ndim in (2, 3) and dst.ndim == img.ndim):
        mod.of_align_warp_alignment_int_rows(
            img, dst,
            float(np.float32(tx)), float(np.float32(ty)),
            float(np.float32(cx)), float(np.float32(cy)),
            float(np.float32(p08)), float(np.float32(p0c)),
            float(np.float32(p18)), float(np.float32(p1c)),
            int(y_start), int(y_stop), False,
        )
        return
    _check_numba()
    _nb_warp_alignment_int_rows(
        img, dst, tx, ty, cx, cy, p08, p0c, p18, p1c, int(y_start), int(y_stop),
    )


def warp_alignment_int_rows_bgr_rev(src_rgb, out, tx, ty, cx, cy, p08, p0c, p18, p1c,
                                    y_start, y_stop):
    mod = _native()
    img = _as_ndarray(src_rgb)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype in (np.uint8, np.uint16) and dst.dtype in (np.uint8, np.uint16, np.uint32)
            and img.ndim == 3 and dst.ndim == 3 and img.shape[2] >= 3 and dst.shape[2] >= 3):
        mod.of_align_warp_alignment_int_rows(
            img, dst,
            float(np.float32(tx)), float(np.float32(ty)),
            float(np.float32(cx)), float(np.float32(cy)),
            float(np.float32(p08)), float(np.float32(p0c)),
            float(np.float32(p18)), float(np.float32(p1c)),
            int(y_start), int(y_stop), True,
        )
        return
    _check_numba()
    _nb_warp_alignment_int_rows_bgr_rev(
        img, dst, tx, ty, cx, cy, p08, p0c, p18, p1c, int(y_start), int(y_stop),
    )


def warp_alignment_float_rows(image, out, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop):
    mod = _native()
    img = _as_ndarray(image)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype in (np.float32, np.float64)
            and dst.dtype in (np.float32, np.float64)
            and img.ndim in (2, 3) and dst.ndim == img.ndim):
        mod.of_align_warp_alignment_float_rows(
            img, dst,
            float(np.float32(tx)), float(np.float32(ty)),
            float(np.float32(cx)), float(np.float32(cy)),
            float(np.float32(p08)), float(np.float32(p0c)),
            float(np.float32(p18)), float(np.float32(p1c)),
            int(y_start), int(y_stop),
        )
        return
    _check_numba()
    _nb_warp_alignment_float_rows(
        img, dst, tx, ty, cx, cy, p08, p0c, p18, p1c, int(y_start), int(y_stop),
    )


def valid_mask_rows(mask, h, w, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop):
    mod = _native()
    m = _as_ndarray(mask)
    if mod is not None and m.dtype == np.bool_ and m.ndim == 2 and m.flags.writeable:
        mod.of_align_valid_mask_rows(
            m.view(np.uint8), int(h), int(w),
            float(np.float32(tx)), float(np.float32(ty)),
            float(np.float32(cx)), float(np.float32(cy)),
            float(np.float32(p08)), float(np.float32(p0c)),
            float(np.float32(p18)), float(np.float32(p1c)),
            int(y_start), int(y_stop),
        )
        return
    _check_numba()
    _nb_valid_mask_rows(
        m, int(h), int(w), tx, ty, cx, cy, p08, p0c, p18, p1c, int(y_start), int(y_stop),
    )


def refine_level(a, b, c_dx, c_dy, half, step, crop):
    mod = _native()
    aa = _as_ndarray(a)
    bb = _as_ndarray(b)
    if (mod is not None and aa.dtype == np.uint16 and bb.dtype == np.uint16
            and aa.ndim == 2 and bb.ndim == 2):
        dx, dy, found = mod.of_align_refine_level(
            aa, bb, float(np.float32(c_dx)), float(np.float32(c_dy)),
            float(half), float(step), int(crop),
        )
        return np.float32(dx), np.float32(dy), bool(found)
    _check_numba()
    return _nb_refine_level(aa, bb, c_dx, c_dy, half, step, int(crop))


def score_chunk_u16_slice(ref, img_b, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy,
                          idx, k_start, k_stop, ssum, ssq, cnt):
    mod = _native()
    r = _as_ndarray(ref)
    b = _as_ndarray(img_b)
    o_sum = _as_ndarray(ssum)
    o_sq = _as_ndarray(ssq)
    o_cnt = _as_ndarray(cnt)
    if (mod is not None and r.dtype == np.uint16 and b.dtype == np.uint16
            and r.ndim == 2 and b.ndim == 2
            and o_sum.dtype == np.int64 and o_sq.dtype == np.int64 and o_cnt.dtype == np.int64
            and o_sum.ndim == 1 and o_sq.ndim == 1 and o_cnt.ndim == 1
            and o_sum.flags.c_contiguous and o_sq.flags.c_contiguous and o_cnt.flags.c_contiguous):
        mod.of_align_score_chunk_u16_slice(
            r, b,
            np.ascontiguousarray(_as_ndarray(xs), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(ys), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(p08), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(p0c), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(p18), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(p1c), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(tx), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(ty), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(cx), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(cy), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(idx), dtype=np.int64),
            int(k_start), int(k_stop),
            o_sum, o_sq, o_cnt,
        )
        return
    _check_numba()
    _nb_score_chunk_u16_slice(
        r, b, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy,
        idx, int(k_start), int(k_stop), o_sum, o_sq, o_cnt,
    )


def score_chunk_u16(ref, img_b, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy,
                    idx, ssum, ssq, cnt):
    score_chunk_u16_slice(
        ref, img_b, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy,
        idx, 0, idx.shape[0], ssum, ssq, cnt,
    )


def score_grid_4d_slice(ref, img_b, xs, ys, tx0, ty0, cx_tab, cy_tab, p08_tab, p18_tab,
                        n0, n1, n2, n3, k_start, k_stop, out_best_k, out_best_sc, slot):
    mod = _native()
    r = _as_ndarray(ref)
    b = _as_ndarray(img_b)
    p08t = _as_ndarray(p08_tab)
    p18t = _as_ndarray(p18_tab)
    ok = _as_ndarray(out_best_k)
    osc = _as_ndarray(out_best_sc)
    if (mod is not None and r.dtype == np.uint16 and b.dtype == np.uint16
            and r.ndim == 2 and b.ndim == 2
            and p08t.dtype == np.float32 and p18t.dtype == np.float32
            and ok.dtype == np.int64 and osc.dtype == np.float32
            and ok.ndim == 1 and osc.ndim == 1
            and ok.flags.c_contiguous and osc.flags.c_contiguous):
        mod.of_align_score_grid_4d_slice(
            r, b,
            np.ascontiguousarray(_as_ndarray(xs), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(ys), dtype=np.float32),
            float(np.float32(tx0)), float(np.float32(ty0)),
            np.ascontiguousarray(_as_ndarray(cx_tab), dtype=np.float32),
            np.ascontiguousarray(_as_ndarray(cy_tab), dtype=np.float32),
            p08t, p18t,
            int(n0), int(n1), int(n2), int(n3),
            int(k_start), int(k_stop),
            ok, osc, int(slot),
        )
        return
    _check_numba()
    _nb_score_grid_4d_slice(
        r, b, xs, ys, tx0, ty0, cx_tab, cy_tab, p08_tab, p18_tab,
        int(n0), int(n1), int(n2), int(n3),
        int(k_start), int(k_stop), ok, osc, int(slot),
    )


def perspective_has_rounding_edge(inverse, x0, x1, y0, y1, bins, tolerance):
    mod = _native()
    inv = np.ascontiguousarray(_as_ndarray(inverse), dtype=np.float64)
    if mod is not None and inv.ndim == 2 and inv.shape[0] >= 3 and inv.shape[1] >= 3:
        return bool(mod.of_align_perspective_has_rounding_edge(
            inv, int(x0), int(x1), int(y0), int(y1), float(bins), float(tolerance),
        ))
    _check_numba()
    return _nb_perspective_has_rounding_edge(
        inv, int(x0), int(x1), int(y0), int(y1), bins, tolerance,
    )


def warp_alignment_region_int_rows(image, out, src_h, src_w, src_y0, x_off, y_off,
                                   tx, ty, cx, cy, p08, p0c, p18, p1c, oy_start, oy_stop):
    mod = _native()
    img = _as_ndarray(image)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype in (np.uint8, np.uint16) and dst.dtype in (np.uint8, np.uint16, np.uint32)
            and img.ndim in (2, 3) and dst.ndim == img.ndim):
        mod.of_align_warp_alignment_region_int_rows(
            img, dst, int(src_h), int(src_w), int(src_y0), int(x_off), int(y_off),
            float(np.float32(tx)), float(np.float32(ty)),
            float(np.float32(cx)), float(np.float32(cy)),
            float(np.float32(p08)), float(np.float32(p0c)),
            float(np.float32(p18)), float(np.float32(p1c)),
            int(oy_start), int(oy_stop), False,
        )
        return
    _check_numba()
    _nb_warp_alignment_region_int_rows(
        img, dst, int(src_h), int(src_w), int(src_y0), int(x_off), int(y_off),
        tx, ty, cx, cy, p08, p0c, p18, p1c, int(oy_start), int(oy_stop),
    )


def warp_alignment_region_int_rows_bgr_rev(src_rgb, out, src_h, src_w, src_y0, x_off, y_off,
                                           tx, ty, cx, cy, p08, p0c, p18, p1c, oy_start, oy_stop):
    mod = _native()
    img = _as_ndarray(src_rgb)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype in (np.uint8, np.uint16) and dst.dtype in (np.uint8, np.uint16, np.uint32)
            and img.ndim == 3 and dst.ndim == 3 and img.shape[2] >= 3 and dst.shape[2] >= 3):
        mod.of_align_warp_alignment_region_int_rows(
            img, dst, int(src_h), int(src_w), int(src_y0), int(x_off), int(y_off),
            float(np.float32(tx)), float(np.float32(ty)),
            float(np.float32(cx)), float(np.float32(cy)),
            float(np.float32(p08)), float(np.float32(p0c)),
            float(np.float32(p18)), float(np.float32(p1c)),
            int(oy_start), int(oy_stop), True,
        )
        return
    _check_numba()
    _nb_warp_alignment_region_int_rows_bgr_rev(
        img, dst, int(src_h), int(src_w), int(src_y0), int(x_off), int(y_off),
        tx, ty, cx, cy, p08, p0c, p18, p1c, int(oy_start), int(oy_stop),
    )


def warp_alignment_region_float_rows(image, out, src_h, src_w, src_y0, x_off, y_off,
                                     tx, ty, cx, cy, p08, p0c, p18, p1c, oy_start, oy_stop):
    mod = _native()
    img = _as_ndarray(image)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype in (np.float32, np.float64)
            and dst.dtype in (np.float32, np.float64)
            and img.ndim in (2, 3) and dst.ndim == img.ndim):
        mod.of_align_warp_alignment_region_float_rows(
            img, dst, int(src_h), int(src_w), int(src_y0), int(x_off), int(y_off),
            float(np.float32(tx)), float(np.float32(ty)),
            float(np.float32(cx)), float(np.float32(cy)),
            float(np.float32(p08)), float(np.float32(p0c)),
            float(np.float32(p18)), float(np.float32(p1c)),
            int(oy_start), int(oy_stop),
        )
        return
    _check_numba()
    _nb_warp_alignment_region_float_rows(
        img, dst, int(src_h), int(src_w), int(src_y0), int(x_off), int(y_off),
        tx, ty, cx, cy, p08, p0c, p18, p1c, int(oy_start), int(oy_stop),
    )


def warp_translation_region_int_rows(image, out, src_h, src_w, src_y0, x_off, y_off,
                                     dx, dy, oy_start, oy_stop):
    mod = _native()
    img = _as_ndarray(image)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype in (np.uint8, np.uint16) and dst.dtype in (np.uint8, np.uint16, np.uint32)
            and img.ndim in (2, 3) and dst.ndim == img.ndim):
        mod.of_align_warp_translation_region_int_rows(
            img, dst, int(src_h), int(src_w), int(src_y0), int(x_off), int(y_off),
            float(np.float32(dx)), float(np.float32(dy)),
            int(oy_start), int(oy_stop), False,
        )
        return
    _check_numba()
    _nb_warp_translation_region_int_rows(
        img, dst, int(src_h), int(src_w), int(src_y0), int(x_off), int(y_off),
        dx, dy, int(oy_start), int(oy_stop),
    )


def warp_translation_region_int_rows_bgr_rev(src_rgb, out, src_h, src_w, src_y0, x_off, y_off,
                                             dx, dy, oy_start, oy_stop):
    mod = _native()
    img = _as_ndarray(src_rgb)
    dst = _as_ndarray(out)
    if (mod is not None and img.dtype in (np.uint8, np.uint16) and dst.dtype in (np.uint8, np.uint16, np.uint32)
            and img.ndim == 3 and dst.ndim == 3 and img.shape[2] >= 3 and dst.shape[2] >= 3):
        mod.of_align_warp_translation_region_int_rows(
            img, dst, int(src_h), int(src_w), int(src_y0), int(x_off), int(y_off),
            float(np.float32(dx)), float(np.float32(dy)),
            int(oy_start), int(oy_stop), True,
        )
        return
    _check_numba()
    _nb_warp_translation_region_int_rows_bgr_rev(
        img, dst, int(src_h), int(src_w), int(src_y0), int(x_off), int(y_off),
        dx, dy, int(oy_start), int(oy_stop),
    )
