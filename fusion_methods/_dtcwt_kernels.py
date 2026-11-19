"""Compiled Numba CPU kernels and fast Transform2d for DTCWT fusion."""
import math
import numpy as np

from fusion_methods._depth_kernel import njit

# Exact near_sym_a filters from dtcwt 0.14.0 (float32 matching X.dtype=float32 in dtcwt)
_H0O = np.array([-0.05, 0.25, 0.6, 0.25, -0.05], dtype=np.float32)
_G0O = np.array([
    -0.010714285714285713, -0.05357142857142857, 0.26071428571428573,
    0.6071428571428571, 0.26071428571428573, -0.05357142857142857,
    -0.010714285714285713,
], dtype=np.float32)
_H1O = np.array([
    0.010714285714285713, -0.05357142857142857, -0.26071428571428573,
    0.6071428571428571, -0.26071428571428573, -0.05357142857142857,
    0.010714285714285713,
], dtype=np.float32)
_G1O = np.array([-0.05, -0.25, 0.6, -0.25, -0.05], dtype=np.float32)

# Exact qshift_a filters from dtcwt 0.14.0
_H0A = np.array([
    0.051130405283831656, -0.013975370246888838, -0.10983605166597087,
    0.26383956105893763, 0.7666284677930372, 0.5636557101270515,
    0.0008736226952170968, -0.1002312195074762, -0.0016896812725281543,
    -0.006181881892116438,
], dtype=np.float32)
_H0B = _H0A[::-1].copy()
_G0A = _H0B.copy()
_G0B = _H0A.copy()
_H1A = np.array([
    -0.006181881892116438, 0.0016896812725281543, -0.1002312195074762,
    -0.0008736226952170968, 0.5636557101270515, -0.7666284677930372,
    0.26383956105893763, 0.10983605166597087, -0.013975370246888838,
    -0.051130405283831656,
], dtype=np.float32)
_H1B = np.array([
    -0.051130405283831656, -0.013975370246888838, 0.10983605166597087,
    0.26383956105893763, -0.7666284677930372, 0.5636557101270515,
    -0.0008736226952170968, -0.1002312195074762, 0.0016896812725281543,
    -0.006181881892116438,
], dtype=np.float32)
_G1A = np.array([
    -0.051130405283831656, -0.013975370246888838, 0.10983605166597087,
    0.26383956105893763, -0.7666284677930372, 0.5636557101270515,
    -0.0008736226952170968, -0.1002312195074762, 0.0016896812725281543,
    -0.006181881892116438,
], dtype=np.float32)
_G1B = np.array([
    -0.006181881892116438, 0.0016896812725281543, -0.1002312195074762,
    -0.0008736226952170968, 0.5636557101270515, -0.7666284677930372,
    0.26383956105893763, 0.10983605166597087, -0.013975370246888838,
    -0.051130405283831656,
], dtype=np.float32)


@njit(nogil=True)
def _reflect_idx(i, n):
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i - 1
        if i >= n:
            i = 2 * n - 1 - i
    return i


@njit(nogil=True)
def _colfilter_v(src, h, dst):
    """Symmetric odd-length 1D filter along columns (axis 0), matching dtcwt order."""
    r, c = src.shape
    m = h.shape[0]
    m2 = m // 2
    zero = np.float32(0.0)
    for y in range(r):
        if m == 5:
            y0 = y - 2 if 2 <= y < r - 2 else _reflect_idx(y - 2, r)
            y1 = y - 1 if 2 <= y < r - 2 else _reflect_idx(y - 1, r)
            y2 = y
            y3 = y + 1 if 2 <= y < r - 2 else _reflect_idx(y + 1, r)
            y4 = y + 2 if 2 <= y < r - 2 else _reflect_idx(y + 2, r)
            w0, w1, w2, w3, w4 = h[0], h[1], h[2], h[3], h[4]
            for x in range(c):
                acc = zero + src[y4, x] * w0
                acc += src[y3, x] * w1
                acc += src[y2, x] * w2
                acc += src[y1, x] * w3
                acc += src[y0, x] * w4
                dst[y, x] = acc
        else:
            interior = m2 <= y < r - m2
            if interior:
                y0, y1, y2, y3, y4, y5, y6 = y - 3, y - 2, y - 1, y, y + 1, y + 2, y + 3
            else:
                y0 = _reflect_idx(y - 3, r)
                y1 = _reflect_idx(y - 2, r)
                y2 = _reflect_idx(y - 1, r)
                y3 = y
                y4 = _reflect_idx(y + 1, r)
                y5 = _reflect_idx(y + 2, r)
                y6 = _reflect_idx(y + 3, r)
            w0, w1, w2, w3, w4, w5, w6 = h[0], h[1], h[2], h[3], h[4], h[5], h[6]
            for x in range(c):
                acc = zero + src[y6, x] * w0
                acc += src[y5, x] * w1
                acc += src[y4, x] * w2
                acc += src[y3, x] * w3
                acc += src[y2, x] * w4
                acc += src[y1, x] * w5
                acc += src[y0, x] * w6
                dst[y, x] = acc


@njit(nogil=True)
def _colfilter_h(src, h, dst):
    """Symmetric odd-length 1D filter along rows (axis 1), matching dtcwt order."""
    r, c = src.shape
    m = h.shape[0]
    m2 = m // 2
    zero = np.float32(0.0)
    if m == 5:
        w0, w1, w2, w3, w4 = h[0], h[1], h[2], h[3], h[4]
        x_mid_end = max(m2, c - m2)
        for y in range(r):
            for x in range(min(m2, c)):
                acc = zero + src[y, _reflect_idx(x + 2, c)] * w0
                acc += src[y, _reflect_idx(x + 1, c)] * w1
                acc += src[y, x] * w2
                acc += src[y, _reflect_idx(x - 1, c)] * w3
                acc += src[y, _reflect_idx(x - 2, c)] * w4
                dst[y, x] = acc
            for x in range(m2, x_mid_end):
                acc = zero + src[y, x + 2] * w0
                acc += src[y, x + 1] * w1
                acc += src[y, x] * w2
                acc += src[y, x - 1] * w3
                acc += src[y, x - 2] * w4
                dst[y, x] = acc
            for x in range(x_mid_end, c):
                acc = zero + src[y, _reflect_idx(x + 2, c)] * w0
                acc += src[y, _reflect_idx(x + 1, c)] * w1
                acc += src[y, x] * w2
                acc += src[y, _reflect_idx(x - 1, c)] * w3
                acc += src[y, _reflect_idx(x - 2, c)] * w4
                dst[y, x] = acc
    else:
        w0, w1, w2, w3, w4, w5, w6 = h[0], h[1], h[2], h[3], h[4], h[5], h[6]
        x_mid_end = max(m2, c - m2)
        for y in range(r):
            for x in range(min(m2, c)):
                acc = zero + src[y, _reflect_idx(x + 3, c)] * w0
                acc += src[y, _reflect_idx(x + 2, c)] * w1
                acc += src[y, _reflect_idx(x + 1, c)] * w2
                acc += src[y, x] * w3
                acc += src[y, _reflect_idx(x - 1, c)] * w4
                acc += src[y, _reflect_idx(x - 2, c)] * w5
                acc += src[y, _reflect_idx(x - 3, c)] * w6
                dst[y, x] = acc
            for x in range(m2, x_mid_end):
                acc = zero + src[y, x + 3] * w0
                acc += src[y, x + 2] * w1
                acc += src[y, x + 1] * w2
                acc += src[y, x] * w3
                acc += src[y, x - 1] * w4
                acc += src[y, x - 2] * w5
                acc += src[y, x - 3] * w6
                dst[y, x] = acc
            for x in range(x_mid_end, c):
                acc = zero + src[y, _reflect_idx(x + 3, c)] * w0
                acc += src[y, _reflect_idx(x + 2, c)] * w1
                acc += src[y, _reflect_idx(x + 1, c)] * w2
                acc += src[y, x] * w3
                acc += src[y, _reflect_idx(x - 1, c)] * w4
                acc += src[y, _reflect_idx(x - 2, c)] * w5
                acc += src[y, _reflect_idx(x - 3, c)] * w6
                dst[y, x] = acc


@njit(nogil=True)
def _coldfilt_v(src, ha, hb, pos_sign, dst):
    """Decimating 10-tap qshift analysis filter along columns (axis 0)."""
    r, c = src.shape
    groups = r // 4
    zero = np.float32(0.0)
    ha0, ha1, ha2, ha3, ha4, ha5, ha6, ha7, ha8, ha9 = (
        ha[0], ha[1], ha[2], ha[3], ha[4], ha[5], ha[6], ha[7], ha[8], ha[9]
    )
    hb0, hb1, hb2, hb3, hb4, hb5, hb6, hb7, hb8, hb9 = (
        hb[0], hb[1], hb[2], hb[3], hb[4], hb[5], hb[6], hb[7], hb[8], hb[9]
    )
    for j in range(groups):
        r1 = 2 * j if pos_sign else 2 * j + 1
        r2 = 2 * j + 1 if pos_sign else 2 * j
        base = 4 * j
        i10 = _reflect_idx(base + 10, r)
        i8 = _reflect_idx(base + 8, r)
        i6 = _reflect_idx(base + 6, r)
        i4 = _reflect_idx(base + 4, r)
        i2 = _reflect_idx(base + 2, r)
        i0 = _reflect_idx(base, r)
        im2 = _reflect_idx(base - 2, r)
        im4 = _reflect_idx(base - 4, r)
        im6 = _reflect_idx(base - 6, r)
        im8 = _reflect_idx(base - 8, r)

        i11 = _reflect_idx(base + 11, r)
        i9 = _reflect_idx(base + 9, r)
        i7 = _reflect_idx(base + 7, r)
        i5 = _reflect_idx(base + 5, r)
        i3 = _reflect_idx(base + 3, r)
        i1 = _reflect_idx(base + 1, r)
        im1 = _reflect_idx(base - 1, r)
        im3 = _reflect_idx(base - 3, r)
        im5 = _reflect_idx(base - 5, r)
        im7 = _reflect_idx(base - 7, r)
        for x in range(c):
            ao = zero + src[i10, x] * ha0
            ao += src[i6, x] * ha2
            ao += src[i2, x] * ha4
            ao += src[im2, x] * ha6
            ao += src[im6, x] * ha8
            ae = zero + src[i8, x] * ha1
            ae += src[i4, x] * ha3
            ae += src[i0, x] * ha5
            ae += src[im4, x] * ha7
            ae += src[im8, x] * ha9
            dst[r1, x] = ao + ae

            bo = zero + src[i11, x] * hb0
            bo += src[i7, x] * hb2
            bo += src[i3, x] * hb4
            bo += src[im1, x] * hb6
            bo += src[im5, x] * hb8
            be = zero + src[i9, x] * hb1
            be += src[i5, x] * hb3
            be += src[i1, x] * hb5
            be += src[im3, x] * hb7
            be += src[im7, x] * hb9
            dst[r2, x] = bo + be


@njit(nogil=True)
def _coldfilt_h(src, ha, hb, pos_sign, dst):
    """Decimating 10-tap qshift analysis filter along rows (axis 1)."""
    r, c = src.shape
    groups = c // 4
    zero = np.float32(0.0)
    ha0, ha1, ha2, ha3, ha4, ha5, ha6, ha7, ha8, ha9 = (
        ha[0], ha[1], ha[2], ha[3], ha[4], ha[5], ha[6], ha[7], ha[8], ha[9]
    )
    hb0, hb1, hb2, hb3, hb4, hb5, hb6, hb7, hb8, hb9 = (
        hb[0], hb[1], hb[2], hb[3], hb[4], hb[5], hb[6], hb[7], hb[8], hb[9]
    )
    for y in range(r):
        for j in range(groups):
            c1 = 2 * j if pos_sign else 2 * j + 1
            c2 = 2 * j + 1 if pos_sign else 2 * j
            base = 4 * j
            ao = zero + src[y, _reflect_idx(base + 10, c)] * ha0
            ao += src[y, _reflect_idx(base + 6, c)] * ha2
            ao += src[y, _reflect_idx(base + 2, c)] * ha4
            ao += src[y, _reflect_idx(base - 2, c)] * ha6
            ao += src[y, _reflect_idx(base - 6, c)] * ha8
            ae = zero + src[y, _reflect_idx(base + 8, c)] * ha1
            ae += src[y, _reflect_idx(base + 4, c)] * ha3
            ae += src[y, _reflect_idx(base, c)] * ha5
            ae += src[y, _reflect_idx(base - 4, c)] * ha7
            ae += src[y, _reflect_idx(base - 8, c)] * ha9
            dst[y, c1] = ao + ae

            bo = zero + src[y, _reflect_idx(base + 11, c)] * hb0
            bo += src[y, _reflect_idx(base + 7, c)] * hb2
            bo += src[y, _reflect_idx(base + 3, c)] * hb4
            bo += src[y, _reflect_idx(base - 1, c)] * hb6
            bo += src[y, _reflect_idx(base - 5, c)] * hb8
            be = zero + src[y, _reflect_idx(base + 9, c)] * hb1
            be += src[y, _reflect_idx(base + 5, c)] * hb3
            be += src[y, _reflect_idx(base + 1, c)] * hb5
            be += src[y, _reflect_idx(base - 3, c)] * hb7
            be += src[y, _reflect_idx(base - 7, c)] * hb9
            dst[y, c2] = bo + be


@njit(nogil=True)
def _colifilt_v(src, ha, hb, pos_sign, dst):
    """Interpolating 10-tap qshift synthesis filter along columns (axis 0, m2=5 odd)."""
    r, c = src.shape
    groups = r // 2
    zero = np.float32(0.0)
    off_a = 0 if pos_sign else -1
    off_b = -1 if pos_sign else 0
    ha0, ha1, ha2, ha3, ha4, ha5, ha6, ha7, ha8, ha9 = (
        ha[0], ha[1], ha[2], ha[3], ha[4], ha[5], ha[6], ha[7], ha[8], ha[9]
    )
    hb0, hb1, hb2, hb3, hb4, hb5, hb6, hb7, hb8, hb9 = (
        hb[0], hb[1], hb[2], hb[3], hb[4], hb[5], hb[6], hb[7], hb[8], hb[9]
    )
    for j in range(groups):
        b0 = _reflect_idx(2 * j + 5 + off_b, r)
        b1 = _reflect_idx(2 * j + 3 + off_b, r)
        b2 = _reflect_idx(2 * j + 1 + off_b, r)
        b3 = _reflect_idx(2 * j - 1 + off_b, r)
        b4 = _reflect_idx(2 * j - 3 + off_b, r)

        a0 = _reflect_idx(2 * j + 5 + off_a, r)
        a1 = _reflect_idx(2 * j + 3 + off_a, r)
        a2 = _reflect_idx(2 * j + 1 + off_a, r)
        a3 = _reflect_idx(2 * j - 1 + off_a, r)
        a4 = _reflect_idx(2 * j - 3 + off_a, r)

        y0 = 4 * j
        for x in range(c):
            d0 = zero + src[b0, x] * ha0
            d0 += src[b1, x] * ha2
            d0 += src[b2, x] * ha4
            d0 += src[b3, x] * ha6
            d0 += src[b4, x] * ha8
            dst[y0, x] = d0

            d1 = zero + src[a0, x] * hb0
            d1 += src[a1, x] * hb2
            d1 += src[a2, x] * hb4
            d1 += src[a3, x] * hb6
            d1 += src[a4, x] * hb8
            dst[y0 + 1, x] = d1

            d2 = zero + src[b0, x] * ha1
            d2 += src[b1, x] * ha3
            d2 += src[b2, x] * ha5
            d2 += src[b3, x] * ha7
            d2 += src[b4, x] * ha9
            dst[y0 + 2, x] = d2

            d3 = zero + src[a0, x] * hb1
            d3 += src[a1, x] * hb3
            d3 += src[a2, x] * hb5
            d3 += src[a3, x] * hb7
            d3 += src[a4, x] * hb9
            dst[y0 + 3, x] = d3


@njit(nogil=True)
def _colifilt_v_add(src_z, ga0, gb0, pos0, src_h, ga1, gb1, pos1, dst):
    """Fused two-branch colifilt along columns storing sum into dst."""
    tmp = np.empty_like(dst)
    _colifilt_v(src_z, ga0, gb0, pos0, dst)
    _colifilt_v(src_h, ga1, gb1, pos1, tmp)
    r, c = dst.shape
    for y in range(r):
        for x in range(c):
            dst[y, x] = dst[y, x] + tmp[y, x]


@njit(nogil=True)
def _colifilt_h(src, ha, hb, pos_sign, dst):
    """Interpolating 10-tap qshift synthesis filter along rows (axis 1, m2=5 odd)."""
    r, c = src.shape
    groups = c // 2
    zero = np.float32(0.0)
    off_a = 0 if pos_sign else -1
    off_b = -1 if pos_sign else 0
    ha0, ha1, ha2, ha3, ha4, ha5, ha6, ha7, ha8, ha9 = (
        ha[0], ha[1], ha[2], ha[3], ha[4], ha[5], ha[6], ha[7], ha[8], ha[9]
    )
    hb0, hb1, hb2, hb3, hb4, hb5, hb6, hb7, hb8, hb9 = (
        hb[0], hb[1], hb[2], hb[3], hb[4], hb[5], hb[6], hb[7], hb[8], hb[9]
    )
    for y in range(r):
        for j in range(groups):
            b0 = _reflect_idx(2 * j + 5 + off_b, c)
            b1 = _reflect_idx(2 * j + 3 + off_b, c)
            b2 = _reflect_idx(2 * j + 1 + off_b, c)
            b3 = _reflect_idx(2 * j - 1 + off_b, c)
            b4 = _reflect_idx(2 * j - 3 + off_b, c)

            a0 = _reflect_idx(2 * j + 5 + off_a, c)
            a1 = _reflect_idx(2 * j + 3 + off_a, c)
            a2 = _reflect_idx(2 * j + 1 + off_a, c)
            a3 = _reflect_idx(2 * j - 1 + off_a, c)
            a4 = _reflect_idx(2 * j - 3 + off_a, c)

            x0 = 4 * j
            d0 = zero + src[y, b0] * ha0
            d0 += src[y, b1] * ha2
            d0 += src[y, b2] * ha4
            d0 += src[y, b3] * ha6
            d0 += src[y, b4] * ha8
            dst[y, x0] = d0

            d1 = zero + src[y, a0] * hb0
            d1 += src[y, a1] * hb2
            d1 += src[y, a2] * hb4
            d1 += src[y, a3] * hb6
            d1 += src[y, a4] * hb8
            dst[y, x0 + 1] = d1

            d2 = zero + src[y, b0] * ha1
            d2 += src[y, b1] * ha3
            d2 += src[y, b2] * ha5
            d2 += src[y, b3] * ha7
            d2 += src[y, b4] * ha9
            dst[y, x0 + 2] = d2

            d3 = zero + src[y, a0] * hb1
            d3 += src[y, a1] * hb3
            d3 += src[y, a2] * hb5
            d3 += src[y, a3] * hb7
            d3 += src[y, a4] * hb9
            dst[y, x0 + 3] = d3


@njit(nogil=True)
def _q2c_pair(quad, hp, c0, c1):
    """Convert real quad array into complex pair slots c0 and c1 of hp."""
    h2, w2 = quad.shape[0] // 2, quad.shape[1] // 2
    inv_sqrt2 = np.float32(math.sqrt(0.5))
    for y in range(h2):
        y2 = 2 * y
        for x in range(w2):
            x2 = 2 * x
            a = quad[y2, x2] * inv_sqrt2
            b = quad[y2, x2 + 1] * inv_sqrt2
            c = -(quad[y2 + 1, x2] * inv_sqrt2)
            d = quad[y2 + 1, x2 + 1] * inv_sqrt2
            re0 = a - d
            im0 = b - c
            re1 = a + d
            im1 = b + c
            hp[y, x, c0] = np.complex64(re0 + 1j * im0)
            hp[y, x, c1] = np.complex64(re1 + 1j * im1)


@njit(nogil=True)
def _c2q_pair(hp, c0, c1, quad):
    """Convert complex pair slots c0 and c1 back to real quad array."""
    h2, w2 = hp.shape[0], hp.shape[1]
    sc = math.sqrt(0.5)
    for y in range(h2):
        y2 = 2 * y
        for x in range(w2):
            x2 = 2 * x
            z0 = hp[y, x, c0]
            z1 = hp[y, x, c1]
            re0 = np.float64(z0.real) * sc
            im0 = np.float64(z0.imag) * sc
            re1 = np.float64(z1.real) * sc
            im1 = np.float64(z1.imag) * sc
            quad[y2, x2] = np.float32(re0 + re1)
            quad[y2, x2 + 1] = np.float32(im0 + im1)
            quad[y2 + 1, x2] = np.float32(im0 - im1)
            quad[y2 + 1, x2 + 1] = np.float32(-(re0 - re1))


def _extend_even(plane):
    """Duplicate last row/column if input dimension is odd (matches dtcwt)."""
    h, w = plane.shape
    eh, ew = h + (h % 2), w + (w % 2)
    if eh == h and ew == w:
        return np.ascontiguousarray(plane, dtype=np.float32)
    out = np.empty((eh, ew), dtype=np.float32)
    out[:h, :w] = plane
    if eh != h:
        out[h, :w] = out[h - 1, :w]
    if ew != w:
        out[:, w] = out[:, w - 1]
    return out


def _pad4(arr):
    """Prepend first edge and append last edge when size is not divisible by 4."""
    r, c = arr.shape
    pr = 1 if (r % 4 != 0) else 0
    pc = 1 if (c % 4 != 0) else 0
    if not pr and not pc:
        return arr
    out = np.empty((r + 2 * pr, c + 2 * pc), dtype=np.float32)
    out[pr:pr + r, pc:pc + c] = arr
    if pr:
        out[0, pc:pc + c] = arr[0, :]
        out[r + 1, pc:pc + c] = arr[r - 1, :]
    if pc:
        out[:, 0] = out[:, 1]
        out[:, c + 1] = out[:, c]
    return out


class FastTransform2d:
    """Drop-in Numba-accelerated Transform2d for near_sym_a + qshift_a."""

    def forward(self, plane, nlevels=4):
        import dtcwt
        x = _extend_even(plane)
        if nlevels == 0:
            return dtcwt.Pyramid(x, ())

        r, c = x.shape
        lo = np.empty((r, c), dtype=np.float32)
        hi = np.empty((r, c), dtype=np.float32)
        _colfilter_v(x, _H0O, lo)
        _colfilter_v(x, _H1O, hi)

        lolo = np.empty((r, c), dtype=np.float32)
        _colfilter_h(lo, _H0O, lolo)
        qtmp = np.empty((r, c), dtype=np.float32)
        yh = [None] * nlevels

        hp0 = np.empty((r >> 1, c >> 1, 6), dtype=np.complex64)
        _colfilter_h(hi, _H0O, qtmp)
        _q2c_pair(qtmp, hp0, 0, 5)
        _colfilter_h(lo, _H1O, qtmp)
        _q2c_pair(qtmp, hp0, 2, 3)
        _colfilter_h(hi, _H1O, qtmp)
        _q2c_pair(qtmp, hp0, 1, 4)
        yh[0] = hp0
        del lo, hi, qtmp

        for level in range(1, nlevels):
            lolo = _pad4(lolo)
            rr, cc = lolo.shape
            r2, c2 = rr >> 1, cc >> 1
            lo2 = np.empty((r2, cc), dtype=np.float32)
            hi2 = np.empty((r2, cc), dtype=np.float32)
            _coldfilt_v(lolo, _H0B, _H0A, True, lo2)
            _coldfilt_v(lolo, _H1B, _H1A, False, hi2)

            lolo_next = np.empty((r2, c2), dtype=np.float32)
            _coldfilt_h(lo2, _H0B, _H0A, True, lolo_next)
            qtmp = np.empty((r2, c2), dtype=np.float32)
            hpk = np.empty((r2 >> 1, c2 >> 1, 6), dtype=np.complex64)
            _coldfilt_h(hi2, _H0B, _H0A, True, qtmp)
            _q2c_pair(qtmp, hpk, 0, 5)
            _coldfilt_h(lo2, _H1B, _H1A, False, qtmp)
            _q2c_pair(qtmp, hpk, 2, 3)
            _coldfilt_h(hi2, _H1B, _H1A, False, qtmp)
            _q2c_pair(qtmp, hpk, 1, 4)
            yh[level] = hpk
            lolo = lolo_next
            del lo2, hi2, qtmp

        return dtcwt.Pyramid(lolo, tuple(yh))

    def inverse(self, pyramid):
        yl = np.ascontiguousarray(pyramid.lowpass, dtype=np.float32)
        yh = pyramid.highpasses
        nlevels = len(yh)
        if nlevels == 0:
            return yl

        z = yl
        for lvl in range(nlevels, 1, -1):
            hp = yh[lvl - 1]
            rq, cq = hp.shape[0] << 1, hp.shape[1] << 1
            q_lh = np.empty((rq, cq), dtype=np.float32)
            q_hl = np.empty((rq, cq), dtype=np.float32)
            q_hh = np.empty((rq, cq), dtype=np.float32)
            _c2q_pair(hp, 0, 5, q_lh)
            _c2q_pair(hp, 2, 3, q_hl)
            _c2q_pair(hp, 1, 4, q_hh)

            r2, c2 = rq << 1, cq << 1
            y1 = np.empty((r2, cq), dtype=np.float32)
            y2 = np.empty((r2, cq), dtype=np.float32)
            _colifilt_v_add(z, _G0B, _G0A, True, q_lh, _G1B, _G1A, False, y1)
            _colifilt_v_add(q_hl, _G0B, _G0A, True, q_hh, _G1B, _G1A, False, y2)

            z1 = np.empty((r2, c2), dtype=np.float32)
            z2 = np.empty((r2, c2), dtype=np.float32)
            _colifilt_h(y1, _G0B, _G0A, True, z1)
            _colifilt_h(y2, _G1B, _G1A, False, z2)
            z = z1 + z2

            exp_r = 2 * yh[lvl - 2].shape[0]
            exp_c = 2 * yh[lvl - 2].shape[1]
            if r2 != exp_r:
                z = z[1:-1, :]
            if c2 != exp_c:
                z = z[:, 1:-1]
            if r2 != exp_r or c2 != exp_c:
                z = np.ascontiguousarray(z)

        hp0 = yh[0]
        rq, cq = hp0.shape[0] << 1, hp0.shape[1] << 1
        q_lh = np.empty((rq, cq), dtype=np.float32)
        q_hl = np.empty((rq, cq), dtype=np.float32)
        q_hh = np.empty((rq, cq), dtype=np.float32)
        _c2q_pair(hp0, 0, 5, q_lh)
        _c2q_pair(hp0, 2, 3, q_hl)
        _c2q_pair(hp0, 1, 4, q_hh)

        y1 = np.empty((rq, cq), dtype=np.float32)
        y1_b = np.empty((rq, cq), dtype=np.float32)
        y2 = np.empty((rq, cq), dtype=np.float32)
        y2_b = np.empty((rq, cq), dtype=np.float32)
        _colfilter_v(z, _G0O, y1)
        _colfilter_v(q_lh, _G1O, y1_b)
        y1 += y1_b
        _colfilter_v(q_hl, _G0O, y2)
        _colfilter_v(q_hh, _G1O, y2_b)
        y2 += y2_b

        z1 = np.empty((rq, cq), dtype=np.float32)
        z2 = np.empty((rq, cq), dtype=np.float32)
        _colfilter_h(y1, _G0O, z1)
        _colfilter_h(y2, _G1O, z2)
        z1 += z2
        return z1


@njit(nogil=True)
def luma_from_bgr_u8(src, dst):
    """Extract normalized float32 luminance directly from uint8 3/4-ch BGR(A)."""
    h, w = src.shape[0], src.shape[1]
    inv = np.float32(1.0 / 255.0)
    kb, kg, kr = 0.114, 0.587, 0.299
    for y in range(h):
        for x in range(w):
            r = np.float32(src[y, x, 2]) * inv
            g = np.float32(src[y, x, 1]) * inv
            b = np.float32(src[y, x, 0]) * inv
            dst[y, x] = np.float32(kr * r + kg * g + kb * b)


@njit(nogil=True)
def luma_from_bgr_u16(src, dst):
    """Extract normalized float32 luminance directly from uint16 3/4-ch BGR(A)."""
    h, w = src.shape[0], src.shape[1]
    to_f32 = np.float32(255.0 / 65535.0)
    inv = np.float32(1.0 / 255.0)
    kb, kg, kr = 0.114, 0.587, 0.299
    for y in range(h):
        for x in range(w):
            r = (np.float32(src[y, x, 2]) * to_f32) * inv
            g = (np.float32(src[y, x, 1]) * to_f32) * inv
            b = (np.float32(src[y, x, 0]) * to_f32) * inv
            dst[y, x] = np.float32(kr * r + kg * g + kb * b)


@njit(nogil=True)
def luma_from_bgr_f32(src, dst):
    """Extract normalized float32 luminance directly from float 0..255 3/4-ch BGR(A)."""
    h, w = src.shape[0], src.shape[1]
    inv = np.float32(1.0 / 255.0)
    kb, kg, kr = 0.114, 0.587, 0.299
    for y in range(h):
        for x in range(w):
            r = np.float32(src[y, x, 2]) * inv
            g = np.float32(src[y, x, 1]) * inv
            b = np.float32(src[y, x, 0]) * inv
            dst[y, x] = np.float32(kr * r + kg * g + kb * b)


@njit(nogil=True)
def luma_from_gray_u8(src, dst):
    """Extract normalized float32 luminance from 2D uint8 grayscale."""
    h, w = src.shape[0], src.shape[1]
    inv = np.float32(1.0 / 255.0)
    kb, kg, kr = 0.114, 0.587, 0.299
    for y in range(h):
        for x in range(w):
            v = np.float32(src[y, x]) * inv
            dst[y, x] = np.float32(kr * v + kg * v + kb * v)


@njit(nogil=True)
def luma_from_gray_u16(src, dst):
    """Extract normalized float32 luminance from 2D uint16 grayscale."""
    h, w = src.shape[0], src.shape[1]
    to_f32 = np.float32(255.0 / 65535.0)
    inv = np.float32(1.0 / 255.0)
    kb, kg, kr = 0.114, 0.587, 0.299
    for y in range(h):
        for x in range(w):
            v = (np.float32(src[y, x]) * to_f32) * inv
            dst[y, x] = np.float32(kr * v + kg * v + kb * v)


@njit(nogil=True)
def luma_from_gray_f32(src, dst):
    """Extract normalized float32 luminance from 2D float 0..255 grayscale."""
    h, w = src.shape[0], src.shape[1]
    inv = np.float32(1.0 / 255.0)
    kb, kg, kr = 0.114, 0.587, 0.299
    for y in range(h):
        for x in range(w):
            v = np.float32(src[y, x]) * inv
            dst[y, x] = np.float32(kr * v + kg * v + kb * v)


def extract_luma(image, dst):
    """Dispatch luminance extraction across 2D/(H,W,1)/(H,W,3)/(H,W,4) and dtypes."""
    if image.ndim == 2:
        plane = image
        is_gray = True
    elif image.ndim == 3 and image.shape[2] == 1:
        plane = image[:, :, 0]
        is_gray = True
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        plane = image
        is_gray = False
    else:
        raise ValueError(f'Unsupported image shape: {image.shape}')

    if is_gray:
        if plane.dtype == np.uint8:
            luma_from_gray_u8(plane, dst)
        elif plane.dtype == np.uint16:
            luma_from_gray_u16(plane, dst)
        else:
            luma_from_gray_f32(np.asarray(plane, dtype=np.float32), dst)
    else:
        if plane.dtype == np.uint8:
            luma_from_bgr_u8(plane, dst)
        elif plane.dtype == np.uint16:
            luma_from_bgr_u16(plane, dst)
        else:
            luma_from_bgr_f32(np.asarray(plane, dtype=np.float32), dst)


@njit(nogil=True)
def activity_max3x3_update(hp, best_act, best_idx, frame_idx, first_frame):
    """Compute sum(|hp|, axis=2), 3x3 reflect max, and streaming argmax in-place."""
    h, w = hp.shape[0], hp.shape[1]
    mag = np.empty((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            s = np.float32(0.0)
            for o in range(6):
                z = hp[y, x, o]
                s += math.sqrt(z.real * z.real + z.imag * z.imag)
            mag[y, x] = s

    for y in range(h):
        ym = 0 if h == 1 else (1 if y == 0 else y - 1)
        yp = h - 1 if h == 1 else (h - 2 if y == h - 1 else y + 1)
        for x in range(w):
            xm = 0 if w == 1 else (1 if x == 0 else x - 1)
            xp = w - 1 if w == 1 else (w - 2 if x == w - 1 else x + 1)
            m = mag[ym, xm]
            v = mag[ym, x]
            if v > m:
                m = v
            v = mag[ym, xp]
            if v > m:
                m = v
            v = mag[y, xm]
            if v > m:
                m = v
            v = mag[y, x]
            if v > m:
                m = v
            v = mag[y, xp]
            if v > m:
                m = v
            v = mag[yp, xm]
            if v > m:
                m = v
            v = mag[yp, x]
            if v > m:
                m = v
            v = mag[yp, xp]
            if v > m:
                m = v
            if first_frame or m > best_act[y, x]:
                best_act[y, x] = m
                best_idx[y, x] = frame_idx


@njit(nogil=True)
def consistent_argmax_3x3(idx, out):
    """Exact 3x3 nearest-border majority mode with smallest-label tie-break."""
    h, w = idx.shape
    for y in range(h):
        ym = 0 if y == 0 else y - 1
        yp = h - 1 if y == h - 1 else y + 1
        for x in range(w):
            xm = 0 if x == 0 else x - 1
            xp = w - 1 if x == w - 1 else x + 1
            v0 = idx[ym, xm]
            v1 = idx[ym, x]
            v2 = idx[ym, xp]
            v3 = idx[y, xm]
            v4 = idx[y, x]
            v5 = idx[y, xp]
            v6 = idx[yp, xm]
            v7 = idx[yp, x]
            v8 = idx[yp, xp]
            if v0 == v1 == v2 == v3 == v4 == v5 == v6 == v7 == v8:
                out[y, x] = v4
                continue
            vals = (v0, v1, v2, v3, v4, v5, v6, v7, v8)
            best_c = 0
            best_l = v0
            for a in range(9):
                la = vals[a]
                c = 1
                for b in range(a + 1, 9):
                    if vals[b] == la:
                        c += 1
                if c > best_c or (c == best_c and la < best_l):
                    best_c = c
                    best_l = la
            out[y, x] = best_l


@njit(nogil=True)
def select_highpass_level(hp_src, decision, hp_dst, frame_idx):
    """Copy complex coefficients at pixels assigned to frame_idx."""
    h, w = decision.shape
    for y in range(h):
        for x in range(w):
            if decision[y, x] == frame_idx:
                for o in range(6):
                    hp_dst[y, x, o] = hp_src[y, x, o]


@njit(nogil=True)
def accum_lowpass(lp_src, weight, lp_dst):
    """Accumulate weighted low-pass plane in-place."""
    h, w = lp_dst.shape
    for y in range(h):
        for x in range(w):
            lp_dst[y, x] += weight[y, x] * lp_src[y, x]


@njit(nogil=True)
def pack_channel_u8(rec, fused_img, c_idx, h, w):
    """Clip rec * 255 to [0, 255] and write uint8 channel using truncation."""
    scale = np.float32(255.0)
    zero = np.float32(0.0)
    for y in range(h):
        for x in range(w):
            v = rec[y, x] * scale
            if v < zero:
                v = zero
            elif v > scale:
                v = scale
            fused_img[y, x, c_idx] = np.uint8(v)


@njit(nogil=True)
def pack_channel_f32(rec, fused_img, c_idx, h, w):
    """Clip rec * 255 to [0, 255] and write float32 channel."""
    scale = np.float32(255.0)
    zero = np.float32(0.0)
    for y in range(h):
        for x in range(w):
            v = rec[y, x] * scale
            if v < zero:
                v = zero
            elif v > scale:
                v = scale
            fused_img[y, x, c_idx] = v
