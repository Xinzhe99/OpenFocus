"""Native CPU kernels for OpenFocus translation and similarity alignment.

These mirror the row-wise definitions in core.openfocus_alignment exactly:
uint32 accumulation, truncating integer division, edge replication and the
float32 sampler clamp order. Numba is optional; callers fall back to NumPy.
"""
import numpy as np

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - optional dependency
    njit = None
    prange = range
    _HAVE_NUMBA = False

_WEIGHTS = (12, 64, 37, 2)


def available():
    return _HAVE_NUMBA


if _HAVE_NUMBA:
    @njit(cache=True, nogil=True)
    def swap_rgb_to_bgr_rows_u8(src, dst, y_start, y_stop):
        w = src.shape[1]
        for y in range(y_start, y_stop):
            for x in range(w):
                dst[y, x, 0] = src[y, x, 2]
                dst[y, x, 1] = src[y, x, 1]
                dst[y, x, 2] = src[y, x, 0]

    @njit(cache=True, nogil=True)
    def swap_rgb_to_bgr_rows_u16(src, dst, y_start, y_stop):
        w = src.shape[1]
        for y in range(y_start, y_stop):
            for x in range(w):
                dst[y, x, 0] = src[y, x, 2]
                dst[y, x, 1] = src[y, x, 1]
                dst[y, x, 2] = src[y, x, 0]

    @njit(cache=True, nogil=True)
    def swap_bgr_inplace_rows_u8(arr, y_start, y_stop):
        w = arr.shape[1]
        for y in range(y_start, y_stop):
            for x in range(w):
                v0 = arr[y, x, 0]
                arr[y, x, 0] = arr[y, x, 2]
                arr[y, x, 2] = v0

    @njit(cache=True, nogil=True)
    def swap_bgr_inplace_rows_u16(arr, y_start, y_stop):
        w = arr.shape[1]
        for y in range(y_start, y_stop):
            for x in range(w):
                v0 = arr[y, x, 0]
                arr[y, x, 0] = arr[y, x, 2]
                arr[y, x, 2] = v0

    @njit(cache=True, nogil=True)
    def gray16_u8_rows(image, out, y_start, y_stop):
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
    def gray16_u8(image, out):
        return gray16_u8_rows(image, out, 0, image.shape[0])

    @njit(cache=True, nogil=True)
    def down_half_rows(src, out, oy_start, oy_stop):
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
    def down_half(src):
        h, w = src.shape
        oh, ow = (h + 1) // 2, (w + 1) // 2
        out = np.empty((oh, ow), np.uint16)
        return down_half_rows(src, out, 0, oh)

if _HAVE_NUMBA:
    @njit(cache=True, nogil=True)
    def gray16_u16_rows(image, out, y_start, y_stop):
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
    def gray16_u16(image, out):
        return gray16_u16_rows(image, out, 0, image.shape[0])

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
    def warp_u16(image, dx, dy):
        out = np.empty_like(image)
        warp_int_rows(image, out, dx, dy, 0, image.shape[0])
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
    def warp_int_rows(image, out, dx, dy, y_start, y_stop):
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
    def warp_int_rows_bgr_rev(src_rgb, out, dx, dy, y_start, y_stop):
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
    def fast_downsize_u16_rows(img, out, oy_start, oy_stop):
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
    def score_chunk_u16_slice(ref, img_b, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy,
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
    def score_chunk_u16(ref, img_b, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy,
                        idx, ssum, ssq, cnt):
        """Score alive similarity candidates over a uint16 level with zero temporaries."""
        score_chunk_u16_slice(
            ref, img_b, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy,
            idx, 0, idx.shape[0], ssum, ssq, cnt,
        )

    @njit(cache=True, nogil=True)
    def warp_alignment_int_rows(image, out, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop):
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
    def warp_alignment_int_rows_bgr_rev(src_rgb, out, tx, ty, cx, cy, p08, p0c, p18, p1c,
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
    def warp_alignment_float_rows(image, out, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop):
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
    def valid_mask_rows(mask, h, w, tx, ty, cx, cy, p08, p0c, p18, p1c, y_start, y_stop):
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
    def refine_level(a, b, c_dx, c_dy, half, step, crop):
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
