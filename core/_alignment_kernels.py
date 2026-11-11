"""Native CPU kernels for the fast translation estimator.

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
    def gray16_u8(image, out):
        h, w = image.shape[:2]
        take = min(image.shape[2], 3) if image.ndim == 3 else 1
        for y in range(h):
            for x in range(w):
                if image.ndim == 2:
                    value = np.uint32(image[y, x])
                else:
                    value = np.uint32(0)
                    for c in range(take):
                        value += np.uint32(image[y, x, c])
                    value //= np.uint32(take)
                out[y, x] = np.uint16(value << np.uint32(8))
        return out

    @njit(cache=True, nogil=True)
    def down_half(src):
        h, w = src.shape
        oh, ow = (h + 1) // 2, (w + 1) // 2
        out = np.empty((oh, ow), np.uint16)
        for oy in range(oh):
            scratch = np.empty(w, np.uint32)
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

if _HAVE_NUMBA:
    @njit(cache=True, nogil=True)
    def gray16_u16(image, out):
        h, w = image.shape[:2]
        take = min(image.shape[2], 3) if image.ndim == 3 else 1
        for y in range(h):
            for x in range(w):
                if image.ndim == 2:
                    out[y, x] = np.uint16(image[y, x])
                else:
                    value = np.uint32(0)
                    for c in range(take):
                        value += np.uint32(image[y, x, c])
                    out[y, x] = np.uint16(value // np.uint32(take))
        return out

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
        h, w = image.shape[:2]
        out = np.empty_like(image)
        channels = 1 if image.ndim == 2 else image.shape[2]
        for y in range(h):
            iy, jy, fy, inv_y = _axis(np.float32(y) - np.float32(dy), h)
            gy = np.uint32(256) - fy
            for x in range(w):
                ix, jx, fx, inv_x = _axis(np.float32(x) - np.float32(dx), w)
                gx = np.uint32(256) - fx
                invalid = inv_y or inv_x
                for c in range(channels):
                    if image.ndim == 2:
                        top = np.uint32(image[iy, ix]) * gx + np.uint32(image[iy, jx]) * fx
                        bottom = np.uint32(image[jy, ix]) * gx + np.uint32(image[jy, jx]) * fx
                        value = (top * gy + bottom * fy) >> np.uint32(16)
                        out[y, x] = np.uint16(0 if invalid else value)
                    else:
                        top = np.uint32(image[iy, ix, c]) * gx + np.uint32(image[iy, jx, c]) * fx
                        bottom = np.uint32(image[jy, ix, c]) * gx + np.uint32(image[jy, jx, c]) * fx
                        value = (top * gy + bottom * fy) >> np.uint32(16)
                        out[y, x, c] = np.uint16(0 if invalid else value)
        return out


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
