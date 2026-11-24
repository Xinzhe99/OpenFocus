"""Streaming weighted-pyramid fusion used by OpenFocus C.

The pyramid stores normalized detail and reconstructs it with a synthesis filter.
"""
import numpy as np
from fusion_methods._depth_kernel import njit

from core.frame_storage import create_transient_frame as create_frame
from fusion_methods._openfocus_execution import run_bands

_EVEN = np.array([0.019027505, 0.318410337, 0.559359729, 0.103202470, 0], dtype=np.float32)
_ODD = _EVEN[::-1].copy()
_DOWN = np.array([0.125, 0.375, 0.375, 0.125], dtype=np.float32)
_PYRAMID_BAND = 64

_ZERO = np.float32(0)
_ONE = np.float32(1)
_D0 = np.float32(0.125)
_D1 = np.float32(0.375)
_E0 = np.float32(0.019027505)
_E1 = np.float32(0.318410337)
_E2 = np.float32(0.559359729)
_E3 = np.float32(0.103202470)
_POINT_ONE = np.float32(0.1)
_THREE = np.float32(3)
_ONE_HALF = np.float32(1.5)
_W_SCALE = np.float32(21845)
_F65535 = np.float32(65535)
_INV257 = np.float32(1.0 / 257.0)


@njit(nogil=True)
def _normalize_rows(source, output, factor, start, stop):
    _, width, channels = output.shape
    if channels == 3:
        for y in range(start, stop):
            for x in range(width):
                output[y, x, 0] = np.float32(source[y, x, 0]) * factor
                output[y, x, 1] = np.float32(source[y, x, 1]) * factor
                output[y, x, 2] = np.float32(source[y, x, 2]) * factor
    elif channels == 1:
        for y in range(start, stop):
            for x in range(width):
                output[y, x, 0] = np.float32(source[y, x, 0]) * factor
    else:
        for y in range(start, stop):
            for x in range(width):
                for c in range(channels):
                    output[y, x, c] = np.float32(source[y, x, c]) * factor


@njit(nogil=True)
def _normalize_flat_rows(source, output, factor, row_size, start, stop):
    for i in range(start * row_size, stop * row_size):
        output[i] = np.float32(source[i]) * factor


@njit(nogil=True)
def _down_horizontal(source, output, origin):
    _, width, channels = source.shape
    out_w = output.shape[1]
    ix_stop = min(out_w, (width - 1) // 2) if width >= 3 else 0
    for row in range(output.shape[0]):
        y = row + origin
        if channels == 3 and ix_stop > 1:
            sx0 = 0
            sx1 = min(width - 1, 0)
            sx2 = min(width - 1, 1)
            sx3 = min(width - 1, 2)
            for c in range(3):
                v = np.float32(_ZERO + np.float32(source[y, sx0, c] * _D0))
                v = np.float32(v + np.float32(source[y, sx1, c] * _D1))
                v = np.float32(v + np.float32(source[y, sx2, c] * _D1))
                output[row, 0, c] = np.float32(v + np.float32(source[y, sx3, c] * _D0))
            for x in range(1, ix_stop):
                sx = 2 * x - 1
                v0 = np.float32(_ZERO + np.float32(source[y, sx, 0] * _D0))
                v1 = np.float32(_ZERO + np.float32(source[y, sx, 1] * _D0))
                v2 = np.float32(_ZERO + np.float32(source[y, sx, 2] * _D0))
                v0 = np.float32(v0 + np.float32(source[y, sx + 1, 0] * _D1))
                v1 = np.float32(v1 + np.float32(source[y, sx + 1, 1] * _D1))
                v2 = np.float32(v2 + np.float32(source[y, sx + 1, 2] * _D1))
                v0 = np.float32(v0 + np.float32(source[y, sx + 2, 0] * _D1))
                v1 = np.float32(v1 + np.float32(source[y, sx + 2, 1] * _D1))
                v2 = np.float32(v2 + np.float32(source[y, sx + 2, 2] * _D1))
                output[row, x, 0] = np.float32(v0 + np.float32(source[y, sx + 3, 0] * _D0))
                output[row, x, 1] = np.float32(v1 + np.float32(source[y, sx + 3, 1] * _D0))
                output[row, x, 2] = np.float32(v2 + np.float32(source[y, sx + 3, 2] * _D0))
            for x in range(ix_stop, out_w):
                b = 2 * x - 1
                sx0 = min(width - 1, max(0, b))
                sx1 = min(width - 1, max(0, b + 1))
                sx2 = min(width - 1, max(0, b + 2))
                sx3 = min(width - 1, max(0, b + 3))
                for c in range(3):
                    v = np.float32(_ZERO + np.float32(source[y, sx0, c] * _D0))
                    v = np.float32(v + np.float32(source[y, sx1, c] * _D1))
                    v = np.float32(v + np.float32(source[y, sx2, c] * _D1))
                    output[row, x, c] = np.float32(v + np.float32(source[y, sx3, c] * _D0))
        elif channels == 1 and ix_stop > 1:
            sx2 = min(width - 1, 1)
            sx3 = min(width - 1, 2)
            v = np.float32(_ZERO + np.float32(source[y, 0, 0] * _D0))
            v = np.float32(v + np.float32(source[y, 0, 0] * _D1))
            v = np.float32(v + np.float32(source[y, sx2, 0] * _D1))
            output[row, 0, 0] = np.float32(v + np.float32(source[y, sx3, 0] * _D0))
            for x in range(1, ix_stop):
                sx = 2 * x - 1
                v = np.float32(_ZERO + np.float32(source[y, sx, 0] * _D0))
                v = np.float32(v + np.float32(source[y, sx + 1, 0] * _D1))
                v = np.float32(v + np.float32(source[y, sx + 2, 0] * _D1))
                output[row, x, 0] = np.float32(v + np.float32(source[y, sx + 3, 0] * _D0))
            for x in range(ix_stop, out_w):
                b = 2 * x - 1
                sx0 = min(width - 1, max(0, b))
                sx1 = min(width - 1, max(0, b + 1))
                sx2 = min(width - 1, max(0, b + 2))
                sx3 = min(width - 1, max(0, b + 3))
                v = np.float32(_ZERO + np.float32(source[y, sx0, 0] * _D0))
                v = np.float32(v + np.float32(source[y, sx1, 0] * _D1))
                v = np.float32(v + np.float32(source[y, sx2, 0] * _D1))
                output[row, x, 0] = np.float32(v + np.float32(source[y, sx3, 0] * _D0))
        else:
            for x in range(out_w):
                b = 2 * x - 1
                sx0 = min(width - 1, max(0, b))
                sx1 = min(width - 1, max(0, b + 1))
                sx2 = min(width - 1, max(0, b + 2))
                sx3 = min(width - 1, max(0, b + 3))
                for c in range(channels):
                    v = np.float32(_ZERO + np.float32(source[y, sx0, c] * _D0))
                    v = np.float32(v + np.float32(source[y, sx1, c] * _D1))
                    v = np.float32(v + np.float32(source[y, sx2, c] * _D1))
                    output[row, x, c] = np.float32(v + np.float32(source[y, sx3, c] * _D0))


@njit(nogil=True)
def _down_vertical(source, output, height, origin, start, stop):
    _, width, channels = source.shape
    for y in range(start, stop):
        r0 = min(height - 1, max(0, 2 * y - 1)) - origin
        r1 = min(height - 1, max(0, 2 * y)) - origin
        r2 = min(height - 1, max(0, 2 * y + 1)) - origin
        r3 = min(height - 1, max(0, 2 * y + 2)) - origin
        for x in range(width):
            for c in range(channels):
                v = np.float32(_ZERO + np.float32(source[r0, x, c] * _D0))
                v = np.float32(v + np.float32(source[r1, x, c] * _D1))
                v = np.float32(v + np.float32(source[r2, x, c] * _D1))
                output[y, x, c] = np.float32(v + np.float32(source[r3, x, c] * _D0))


@njit(nogil=True)
def _down_vertical_flat(source, output, height, origin, start, stop):
    elems = source.shape[1]
    for y in range(start, stop):
        r0 = min(height - 1, max(0, 2 * y - 1)) - origin
        r1 = min(height - 1, max(0, 2 * y)) - origin
        r2 = min(height - 1, max(0, 2 * y + 1)) - origin
        r3 = min(height - 1, max(0, 2 * y + 2)) - origin
        for i in range(elems):
            v = np.float32(_ZERO + np.float32(source[r0, i] * _D0))
            v = np.float32(v + np.float32(source[r1, i] * _D1))
            v = np.float32(v + np.float32(source[r2, i] * _D1))
            output[y, i] = np.float32(v + np.float32(source[r3, i] * _D0))


@njit(nogil=True)
def _up_horizontal(source, output, origin):
    _, width, channels = source.shape
    out_w = output.shape[1]
    ix_start = min(out_w, 4)
    raw_stop = min(out_w, 2 * (width - 2)) if width >= 3 else ix_start
    pair_stop = ix_start + (max(0, raw_stop - ix_start) & ~1)
    for row in range(output.shape[0]):
        y = row + origin
        for x in range(ix_start):
            weights = _ODD if x % 2 else _EVEN
            left = x // 2 - 2
            first = max(0, -left)
            last = min(5, width - left)
            total = np.float32(0)
            for k in range(first, last):
                total = np.float32(total + weights[k])
            for c in range(channels):
                value = np.float32(0)
                for k in range(first, last):
                    value = np.float32(value + np.float32(source[y, left + k, c] * weights[k]))
                output[row, x, c] = np.float32(value / total)
        if channels == 3 and pair_stop > ix_start:
            for x in range(ix_start, pair_stop, 2):
                left = x // 2 - 2
                s0_0 = source[y, left, 0]; s0_1 = source[y, left, 1]; s0_2 = source[y, left, 2]
                s1_0 = source[y, left + 1, 0]; s1_1 = source[y, left + 1, 1]; s1_2 = source[y, left + 1, 2]
                s2_0 = source[y, left + 2, 0]; s2_1 = source[y, left + 2, 1]; s2_2 = source[y, left + 2, 2]
                s3_0 = source[y, left + 3, 0]; s3_1 = source[y, left + 3, 1]; s3_2 = source[y, left + 3, 2]
                s4_0 = source[y, left + 4, 0]; s4_1 = source[y, left + 4, 1]; s4_2 = source[y, left + 4, 2]

                ve0 = np.float32(np.float32(_ZERO + np.float32(s0_0 * _E0)) + np.float32(s1_0 * _E1))
                ve1 = np.float32(np.float32(_ZERO + np.float32(s0_1 * _E0)) + np.float32(s1_1 * _E1))
                ve2 = np.float32(np.float32(_ZERO + np.float32(s0_2 * _E0)) + np.float32(s1_2 * _E1))
                ve0 = np.float32(ve0 + np.float32(s2_0 * _E2))
                ve1 = np.float32(ve1 + np.float32(s2_1 * _E2))
                ve2 = np.float32(ve2 + np.float32(s2_2 * _E2))
                ve0 = np.float32(np.float32(ve0 + np.float32(s3_0 * _E3)) + np.float32(s4_0 * _ZERO))
                ve1 = np.float32(np.float32(ve1 + np.float32(s3_1 * _E3)) + np.float32(s4_1 * _ZERO))
                ve2 = np.float32(np.float32(ve2 + np.float32(s3_2 * _E3)) + np.float32(s4_2 * _ZERO))
                output[row, x, 0] = ve0
                output[row, x, 1] = ve1
                output[row, x, 2] = ve2

                vo0 = np.float32(np.float32(_ZERO + np.float32(s0_0 * _ZERO)) + np.float32(s1_0 * _E3))
                vo1 = np.float32(np.float32(_ZERO + np.float32(s0_1 * _ZERO)) + np.float32(s1_1 * _E3))
                vo2 = np.float32(np.float32(_ZERO + np.float32(s0_2 * _ZERO)) + np.float32(s1_2 * _E3))
                vo0 = np.float32(vo0 + np.float32(s2_0 * _E2))
                vo1 = np.float32(vo1 + np.float32(s2_1 * _E2))
                vo2 = np.float32(vo2 + np.float32(s2_2 * _E2))
                vo0 = np.float32(np.float32(vo0 + np.float32(s3_0 * _E1)) + np.float32(s4_0 * _E0))
                vo1 = np.float32(np.float32(vo1 + np.float32(s3_1 * _E1)) + np.float32(s4_1 * _E0))
                vo2 = np.float32(np.float32(vo2 + np.float32(s3_2 * _E1)) + np.float32(s4_2 * _E0))
                output[row, x + 1, 0] = vo0
                output[row, x + 1, 1] = vo1
                output[row, x + 1, 2] = vo2
        elif channels == 1 and pair_stop > ix_start:
            for x in range(ix_start, pair_stop, 2):
                left = x // 2 - 2
                s0 = source[y, left, 0]
                s1 = source[y, left + 1, 0]
                s2 = source[y, left + 2, 0]
                s3 = source[y, left + 3, 0]
                s4 = source[y, left + 4, 0]
                ve = np.float32(np.float32(_ZERO + np.float32(s0 * _E0)) + np.float32(s1 * _E1))
                ve = np.float32(ve + np.float32(s2 * _E2))
                output[row, x, 0] = np.float32(np.float32(ve + np.float32(s3 * _E3)) + np.float32(s4 * _ZERO))
                vo = np.float32(np.float32(_ZERO + np.float32(s0 * _ZERO)) + np.float32(s1 * _E3))
                vo = np.float32(vo + np.float32(s2 * _E2))
                output[row, x + 1, 0] = np.float32(np.float32(vo + np.float32(s3 * _E1)) + np.float32(s4 * _E0))
        else:
            for x in range(ix_start, pair_stop):
                weights = _ODD if x % 2 else _EVEN
                left = x // 2 - 2
                for c in range(channels):
                    value = np.float32(0)
                    for k in range(5):
                        value = np.float32(value + np.float32(source[y, left + k, c] * weights[k]))
                    output[row, x, c] = value
        for x in range(pair_stop, out_w):
            weights = _ODD if x % 2 else _EVEN
            left = x // 2 - 2
            first = max(0, -left)
            last = min(5, width - left)
            total = np.float32(0)
            for k in range(first, last):
                total = np.float32(total + weights[k])
            for c in range(channels):
                value = np.float32(0)
                for k in range(first, last):
                    value = np.float32(value + np.float32(source[y, left + k, c] * weights[k]))
                output[row, x, c] = np.float32(value / total)


@njit(nogil=True)
def _synthesis_rows(horizontal, target, weight, coarse_height, origin, start, stop):
    height, width, channels = target.shape
    for y in range(start, stop):
        top = y // 2 - 2
        weights = _ODD if y % 2 else _EVEN
        total = np.float32(0)
        first = max(0, -top)
        last = min(5, coarse_height - top)
        for k in range(first, last):
            total = np.float32(total + weights[k])
        for x in range(width):
            strength = np.float32(0)
            for c in range(channels):
                lower = np.float32(0)
                for k in range(first, last):
                    sample = horizontal[top + k - origin, x, c]
                    lower = np.float32(lower + np.float32(sample * weights[k]))
                lower = np.float32(lower / total)
                denom = np.float32(lower + np.float32(0.1))
                if weight is None:
                    value = np.float32(np.float32(denom * target[y, x, c]) + lower)
                    target[y, x, c] = min(np.float32(1), max(np.float32(0), value))
                else:
                    value = np.float32(np.float32(target[y, x, c] - lower) / denom)
                    target[y, x, c] = value
                    if c < 3:
                        strength = np.float32(strength + abs(value))
            if weight is not None:
                if channels == 1:
                    strength = np.float32(strength * np.float32(3))
                elif channels == 2:
                    strength = np.float32(strength * np.float32(1.5))
                if ((height > 2 and (y == 0 or y == height - 1))
                        or (width > 2 and (x == 0 or x == width - 1))):
                    weight[y, x] = 0
                else:
                    weight[y, x] = min(65535, int(np.float32(strength * np.float32(21845))))


@njit(nogil=True)
def _reconstruct_rows_flat(horizontal, target, coarse_height, origin, start, stop):
    elems = target.shape[1]
    for y in range(start, stop):
        top = y // 2 - 2
        odd = y & 1
        weights = _ODD if odd else _EVEN
        first = max(0, -top)
        last = min(5, coarse_height - top)
        if first == 0 and last == 5:
            base = top - origin
            if odd:
                r0 = base
                r1 = base + 1
                r2 = base + 2
                r3 = base + 3
                r4 = base + 4
                for i in range(elems):
                    lo = np.float32(np.float32(_ZERO + np.float32(horizontal[r0, i] * _ZERO))
                                    + np.float32(horizontal[r1, i] * _E3))
                    lo = np.float32(lo + np.float32(horizontal[r2, i] * _E2))
                    lo = np.float32(np.float32(lo + np.float32(horizontal[r3, i] * _E1))
                                    + np.float32(horizontal[r4, i] * _E0))
                    denom = np.float32(lo + _POINT_ONE)
                    val = np.float32(np.float32(denom * target[y, i]) + lo)
                    target[y, i] = min(_ONE, max(_ZERO, val))
            else:
                r0 = base
                r1 = base + 1
                r2 = base + 2
                r3 = base + 3
                r4 = base + 4
                for i in range(elems):
                    lo = np.float32(np.float32(_ZERO + np.float32(horizontal[r0, i] * _E0))
                                    + np.float32(horizontal[r1, i] * _E1))
                    lo = np.float32(lo + np.float32(horizontal[r2, i] * _E2))
                    lo = np.float32(np.float32(lo + np.float32(horizontal[r3, i] * _E3))
                                    + np.float32(horizontal[r4, i] * _ZERO))
                    denom = np.float32(lo + _POINT_ONE)
                    val = np.float32(np.float32(denom * target[y, i]) + lo)
                    target[y, i] = min(_ONE, max(_ZERO, val))
        else:
            total = np.float32(0)
            for k in range(first, last):
                total = np.float32(total + weights[k])
            for i in range(elems):
                lo = np.float32(0)
                for k in range(first, last):
                    lo = np.float32(lo + np.float32(horizontal[top + k - origin, i] * weights[k]))
                lo = np.float32(lo / total)
                denom = np.float32(lo + _POINT_ONE)
                val = np.float32(np.float32(denom * target[y, i]) + lo)
                target[y, i] = min(_ONE, max(_ZERO, val))


@njit(nogil=True)
def _detail_weight_rows(horizontal, target, weight, coarse_height, origin, start, stop):
    height, width, channels = target.shape
    for y in range(start, stop):
        top = y // 2 - 2
        odd = y & 1
        weights = _ODD if odd else _EVEN
        first = max(0, -top)
        last = min(5, coarse_height - top)
        is_y_border = height > 2 and (y == 0 or y == height - 1)
        if first == 0 and last == 5 and channels == 3 and width > 2:
            r0 = top - origin
            r1 = r0 + 1
            r2 = r0 + 2
            r3 = r0 + 3
            r4 = r0 + 4
            w0 = _ZERO if odd else _E0
            w1 = _E3 if odd else _E1
            w2 = _E2
            w3 = _E1 if odd else _E3
            w4 = _E0 if odd else _ZERO
            if is_y_border:
                for x in range(width):
                    s = _ZERO
                    for c in range(3):
                        lo = np.float32(np.float32(_ZERO + np.float32(horizontal[r0, x, c] * w0))
                                        + np.float32(horizontal[r1, x, c] * w1))
                        lo = np.float32(lo + np.float32(horizontal[r2, x, c] * w2))
                        lo = np.float32(np.float32(lo + np.float32(horizontal[r3, x, c] * w3))
                                        + np.float32(horizontal[r4, x, c] * w4))
                        denom = np.float32(lo + _POINT_ONE)
                        val = np.float32(np.float32(target[y, x, c] - lo) / denom)
                        target[y, x, c] = val
                        s = np.float32(s + abs(val))
                    weight[y, x] = 0
            else:
                # x = 0 border
                for c in range(3):
                    lo = np.float32(np.float32(_ZERO + np.float32(horizontal[r0, 0, c] * w0))
                                    + np.float32(horizontal[r1, 0, c] * w1))
                    lo = np.float32(lo + np.float32(horizontal[r2, 0, c] * w2))
                    lo = np.float32(np.float32(lo + np.float32(horizontal[r3, 0, c] * w3))
                                    + np.float32(horizontal[r4, 0, c] * w4))
                    denom = np.float32(lo + _POINT_ONE)
                    target[y, 0, c] = np.float32(np.float32(target[y, 0, c] - lo) / denom)
                weight[y, 0] = 0
                # Interior 1 <= x < width - 1: completely branchless 3-channel synthesis
                for x in range(1, width - 1):
                    lo0 = np.float32(np.float32(_ZERO + np.float32(horizontal[r0, x, 0] * w0))
                                     + np.float32(horizontal[r1, x, 0] * w1))
                    lo1 = np.float32(np.float32(_ZERO + np.float32(horizontal[r0, x, 1] * w0))
                                     + np.float32(horizontal[r1, x, 1] * w1))
                    lo2 = np.float32(np.float32(_ZERO + np.float32(horizontal[r0, x, 2] * w0))
                                     + np.float32(horizontal[r1, x, 2] * w1))

                    lo0 = np.float32(lo0 + np.float32(horizontal[r2, x, 0] * w2))
                    lo1 = np.float32(lo1 + np.float32(horizontal[r2, x, 1] * w2))
                    lo2 = np.float32(lo2 + np.float32(horizontal[r2, x, 2] * w2))

                    lo0 = np.float32(np.float32(lo0 + np.float32(horizontal[r3, x, 0] * w3))
                                     + np.float32(horizontal[r4, x, 0] * w4))
                    lo1 = np.float32(np.float32(lo1 + np.float32(horizontal[r3, x, 1] * w3))
                                     + np.float32(horizontal[r4, x, 1] * w4))
                    lo2 = np.float32(np.float32(lo2 + np.float32(horizontal[r3, x, 2] * w3))
                                     + np.float32(horizontal[r4, x, 2] * w4))

                    v0 = np.float32(np.float32(target[y, x, 0] - lo0) / np.float32(lo0 + _POINT_ONE))
                    v1 = np.float32(np.float32(target[y, x, 1] - lo1) / np.float32(lo1 + _POINT_ONE))
                    v2 = np.float32(np.float32(target[y, x, 2] - lo2) / np.float32(lo2 + _POINT_ONE))
                    target[y, x, 0] = v0
                    target[y, x, 1] = v1
                    target[y, x, 2] = v2
                    strength = np.float32(np.float32(np.float32(_ZERO + abs(v0)) + abs(v1)) + abs(v2))
                    weight[y, x] = min(65535, int(np.float32(strength * _W_SCALE)))
                # x = width - 1 border
                xm = width - 1
                for c in range(3):
                    lo = np.float32(np.float32(_ZERO + np.float32(horizontal[r0, xm, c] * w0))
                                    + np.float32(horizontal[r1, xm, c] * w1))
                    lo = np.float32(lo + np.float32(horizontal[r2, xm, c] * w2))
                    lo = np.float32(np.float32(lo + np.float32(horizontal[r3, xm, c] * w3))
                                    + np.float32(horizontal[r4, xm, c] * w4))
                    denom = np.float32(lo + _POINT_ONE)
                    target[y, xm, c] = np.float32(np.float32(target[y, xm, c] - lo) / denom)
                weight[y, xm] = 0
        elif first == 0 and last == 5 and channels == 1 and width >= 2 and not is_y_border:
            r0 = top - origin
            r1 = r0 + 1
            r2 = r0 + 2
            r3 = r0 + 3
            r4 = r0 + 4
            w0 = _ZERO if odd else _E0
            w1 = _E3 if odd else _E1
            w2 = _E2
            w3 = _E1 if odd else _E3
            w4 = _E0 if odd else _ZERO
            for x in range(width):
                lo = np.float32(np.float32(_ZERO + np.float32(horizontal[r0, x, 0] * w0))
                                + np.float32(horizontal[r1, x, 0] * w1))
                lo = np.float32(lo + np.float32(horizontal[r2, x, 0] * w2))
                lo = np.float32(np.float32(lo + np.float32(horizontal[r3, x, 0] * w3))
                                + np.float32(horizontal[r4, x, 0] * w4))
                v = np.float32(np.float32(target[y, x, 0] - lo) / np.float32(lo + _POINT_ONE))
                target[y, x, 0] = v
                if width > 2 and (x == 0 or x == width - 1):
                    weight[y, x] = 0
                else:
                    weight[y, x] = min(65535, int(np.float32(np.float32(abs(v) * _THREE) * _W_SCALE)))
        else:
            total = np.float32(0)
            for k in range(first, last):
                total = np.float32(total + weights[k])
            for x in range(width):
                strength = np.float32(0)
                for c in range(channels):
                    lower = np.float32(0)
                    for k in range(first, last):
                        sample = horizontal[top + k - origin, x, c]
                        lower = np.float32(lower + np.float32(sample * weights[k]))
                    lower = np.float32(lower / total)
                    denom = np.float32(lower + _POINT_ONE)
                    value = np.float32(np.float32(target[y, x, c] - lower) / denom)
                    target[y, x, c] = value
                    if c < 3:
                        strength = np.float32(strength + abs(value))
                if channels == 1:
                    strength = np.float32(strength * _THREE)
                elif channels == 2:
                    strength = np.float32(strength * _ONE_HALF)
                if is_y_border or (width > 2 and (x == 0 or x == width - 1)):
                    weight[y, x] = 0
                else:
                    weight[y, x] = min(65535, int(np.float32(strength * _W_SCALE)))


@njit(nogil=True)
def _merge_rows(detail, weight, accumulator, best, first, start, stop,
                winner=None, index=0):
    width = weight.shape[1]
    channels = detail.shape[2]
    if first:
        if channels == 3:
            for y in range(start, stop):
                for x in range(width):
                    best[y, x] = weight[y, x]
                    if winner is not None:
                        winner[y, x] = index
                    accumulator[y, x, 0] = detail[y, x, 0]
                    accumulator[y, x, 1] = detail[y, x, 1]
                    accumulator[y, x, 2] = detail[y, x, 2]
        elif channels == 1:
            for y in range(start, stop):
                for x in range(width):
                    best[y, x] = weight[y, x]
                    if winner is not None:
                        winner[y, x] = index
                    accumulator[y, x, 0] = detail[y, x, 0]
        else:
            for y in range(start, stop):
                for x in range(width):
                    best[y, x] = weight[y, x]
                    if winner is not None:
                        winner[y, x] = index
                    for c in range(channels):
                        accumulator[y, x, c] = detail[y, x, c]
        return

    if winner is None:
        if channels == 3:
            for y in range(start, stop):
                for x in range(width):
                    w = weight[y, x]
                    if w > best[y, x]:
                        best[y, x] = w
                        accumulator[y, x, 0] = detail[y, x, 0]
                        accumulator[y, x, 1] = detail[y, x, 1]
                        accumulator[y, x, 2] = detail[y, x, 2]
        elif channels == 1:
            for y in range(start, stop):
                for x in range(width):
                    w = weight[y, x]
                    if w > best[y, x]:
                        best[y, x] = w
                        accumulator[y, x, 0] = detail[y, x, 0]
        else:
            for y in range(start, stop):
                for x in range(width):
                    w = weight[y, x]
                    if w > best[y, x]:
                        best[y, x] = w
                        for c in range(channels):
                            accumulator[y, x, c] = detail[y, x, c]
    else:
        if channels == 3:
            for y in range(start, stop):
                for x in range(width):
                    w = weight[y, x]
                    if w > best[y, x]:
                        best[y, x] = w
                        winner[y, x] = index
                        accumulator[y, x, 0] = detail[y, x, 0]
                        accumulator[y, x, 1] = detail[y, x, 1]
                        accumulator[y, x, 2] = detail[y, x, 2]
        elif channels == 1:
            for y in range(start, stop):
                for x in range(width):
                    w = weight[y, x]
                    if w > best[y, x]:
                        best[y, x] = w
                        winner[y, x] = index
                        accumulator[y, x, 0] = detail[y, x, 0]
        else:
            for y in range(start, stop):
                for x in range(width):
                    w = weight[y, x]
                    if w > best[y, x]:
                        best[y, x] = w
                        winner[y, x] = index
                        for c in range(channels):
                            accumulator[y, x, c] = detail[y, x, c]


@njit(nogil=True)
def _output_rows(source, output, is_u8, start, stop):
    for y in range(start, stop):
        for x in range(source.shape[1]):
            for c in range(source.shape[2]):
                value = min(_ONE, max(_ZERO, source[y, x, c]))
                code = np.float32(value * _F65535)
                if is_u8:
                    output[y, x, c] = int(code) >> 8
                else:
                    output[y, x, c] = np.float32(np.rint(code)) * _INV257


@njit(nogil=True)
def _output_rows_flat(source, output, is_u8, row_size, start, stop):
    if is_u8:
        for i in range(start * row_size, stop * row_size):
            value = min(_ONE, max(_ZERO, source[i]))
            output[i] = int(np.float32(value * _F65535)) >> 8
    else:
        for i in range(start * row_size, stop * row_size):
            value = min(_ONE, max(_ZERO, source[i]))
            code = np.float32(value * _F65535)
            output[i] = np.float32(np.rint(code)) * _INV257


def _as_ndarray(arr):
    return None if arr is None else (arr if type(arr) is np.ndarray else arr.view(np.ndarray))


def _rows(length, function, cancel):
    def run(start, stop):
        cancel()
        function(start, stop)
    run_bands(length, 128, run)


def _normalize(source, output, factor, cancel):
    # Flatten only packed sources. Crops and flipped arrays retain the
    # strided path without allocating a contiguous copy.
    height, width, channels = source.shape
    if source.flags.c_contiguous:
        pixels = source.reshape(-1)
        normalized = output.reshape(-1)
        _rows(height, lambda a, b: _normalize_flat_rows(
            pixels, normalized, factor, width * channels, a, b), cancel)
    else:
        _rows(height, lambda a, b: _normalize_rows(source, output, factor, a, b), cancel)


def _down(source, output, cancel):
    from core._native import load_native

    height, _, channels = source.shape
    out_height, out_width, _ = output.shape
    rows = min(height, 2 * min(out_height, _PYRAMID_BAND) + 2)
    scratch_shape = rows, out_width, channels
    native = load_native() if (_down_horizontal is _DEFAULT_DOWN_HORIZONTAL
                               and _down_vertical is _DEFAULT_DOWN_VERTICAL) else None
    if native is not None and hasattr(native, 'of_pyr_down_f32'):
        src_nd = _as_ndarray(source)
        out_nd = _as_ndarray(output)

        def down_native(start, stop, scratch):
            cancel()
            native.of_pyr_down_f32(src_nd, out_nd, _as_ndarray(scratch), int(start), int(stop))

        run_bands(out_height, _PYRAMID_BAND, down_native,
                  scratch_bytes=int(np.prod(scratch_shape)) * 4,
                  workspace=lambda: np.empty(scratch_shape, np.float32))
        return

    out_flat = output.reshape(out_height, out_width * channels) if output.flags.c_contiguous else None

    def down(start, stop, scratch):
        cancel()
        origin = max(0, 2 * start - 1)
        end = min(height, 2 * stop + 1)
        horizontal = scratch[:end - origin]
        _down_horizontal(source, horizontal, origin)
        if out_flat is not None:
            _down_vertical_flat(horizontal.reshape(end - origin, out_width * channels),
                                out_flat, height, origin, start, stop)
        else:
            _down_vertical(horizontal, output, height, origin, start, stop)

    run_bands(out_height, _PYRAMID_BAND, down,
              scratch_bytes=int(np.prod(scratch_shape)) * 4,
              workspace=lambda: np.empty(scratch_shape, np.float32))


def _synthesize(coarse, target, weight, cancel):
    from core._native import load_native

    coarse_height = coarse.shape[0]
    height, width, channels = target.shape
    rows = min(coarse_height, (_PYRAMID_BAND + 1) // 2 + 4)
    scratch_shape = rows, width, channels
    native = load_native() if (_up_horizontal is _DEFAULT_UP_HORIZONTAL
                               and _reconstruct_rows_flat is _DEFAULT_RECONSTRUCT_ROWS_FLAT
                               and _detail_weight_rows is _DEFAULT_DETAIL_WEIGHT_ROWS
                               and _synthesis_rows is _DEFAULT_SYNTHESIS_ROWS) else None
    if (native is not None and hasattr(native, 'of_pyr_detail_weight_f32')
            and (weight is None or (weight.dtype == np.uint16 and weight.strides[1] == 2))):
        coarse_nd = _as_ndarray(coarse)
        target_nd = _as_ndarray(target)
        weight_nd = _as_ndarray(weight)

        def synth_native(start, stop, scratch):
            cancel()
            if weight_nd is not None:
                native.of_pyr_detail_weight_f32(coarse_nd, target_nd, weight_nd,
                                                _as_ndarray(scratch), int(start), int(stop))
            else:
                native.of_pyr_reconstruct_f32(coarse_nd, target_nd,
                                              _as_ndarray(scratch), int(start), int(stop))

        run_bands(height, _PYRAMID_BAND, synth_native,
                  scratch_bytes=int(np.prod(scratch_shape)) * 4,
                  workspace=lambda: np.empty(scratch_shape, np.float32))
        return

    target_flat = (target.reshape(height, width * channels)
                   if weight is None and target.flags.c_contiguous else None)

    def synthesize(start, stop, scratch):
        cancel()
        origin = max(0, start // 2 - 2)
        end = min(coarse_height, (stop - 1) // 2 + 3)
        horizontal = scratch[:end - origin]
        _up_horizontal(coarse, horizontal, origin)
        if target_flat is not None:
            _reconstruct_rows_flat(horizontal.reshape(end - origin, width * channels),
                                   target_flat, coarse_height, origin, start, stop)
        elif weight is not None:
            _detail_weight_rows(horizontal, target, weight, coarse_height, origin, start, stop)
        else:
            _synthesis_rows(horizontal, target, weight, coarse_height, origin, start, stop)

    run_bands(height, _PYRAMID_BAND, synthesize,
              scratch_bytes=int(np.prod(scratch_shape)) * 4,
              workspace=lambda: np.empty(scratch_shape, np.float32))


_DEFAULT_NORMALIZE = _normalize
_DEFAULT_DOWN = _down
_DEFAULT_DOWN_HORIZONTAL = _down_horizontal
_DEFAULT_DOWN_VERTICAL = _down_vertical
_DEFAULT_SYNTHESIZE = _synthesize
_DEFAULT_UP_HORIZONTAL = _up_horizontal
_DEFAULT_RECONSTRUCT_ROWS_FLAT = _reconstruct_rows_flat
_DEFAULT_DETAIL_WEIGHT_ROWS = _detail_weight_rows
_DEFAULT_SYNTHESIS_ROWS = _synthesis_rows
_DEFAULT_MERGE_ROWS = _merge_rows
_DEFAULT_OUTPUT_ROWS = _output_rows
_DEFAULT_OUTPUT_ROWS_FLAT = _output_rows_flat


def _down_from_raw(native, source, output, factor, cancel):
    height, _, channels = source.shape
    out_height, out_width, _ = output.shape
    rows = min(height, 2 * min(out_height, _PYRAMID_BAND) + 2)
    scratch_shape = rows, out_width, channels
    src_nd = _as_ndarray(source)
    out_nd = _as_ndarray(output)
    factor_f = float(factor)

    def band(start, stop, scratch):
        cancel()
        native.of_pyr_down_from_raw(src_nd, out_nd, _as_ndarray(scratch),
                                    factor_f, int(start), int(stop))

    run_bands(out_height, _PYRAMID_BAND, band,
              scratch_bytes=int(np.prod(scratch_shape)) * 4,
              workspace=lambda: np.empty(scratch_shape, np.float32))


def _level0_weights_from_raw(native, coarse, source, weight, factor, cancel):
    coarse_height = coarse.shape[0]
    height, width, channels = source.shape
    rows = min(coarse_height, (_PYRAMID_BAND + 1) // 2 + 4)
    scratch_shape = rows, width, channels
    coarse_nd = _as_ndarray(coarse)
    src_nd = _as_ndarray(source)
    weight_nd = _as_ndarray(weight)
    factor_f = float(factor)

    def band(start, stop, scratch):
        cancel()
        native.of_pyr_level0_weights_from_raw(coarse_nd, src_nd, weight_nd,
                                              _as_ndarray(scratch), factor_f,
                                              int(start), int(stop))

    run_bands(height, _PYRAMID_BAND, band,
              scratch_bytes=int(np.prod(scratch_shape)) * 4,
              workspace=lambda: np.empty(scratch_shape, np.float32))


def _level0_merge_from_raw(native, coarse, source, weight, accumulator, best,
                           winner, factor, first, index, cancel):
    coarse_height = coarse.shape[0]
    height, width, channels = source.shape
    rows = min(coarse_height, (_PYRAMID_BAND + 1) // 2 + 4)
    scratch_shape = rows, width, channels
    coarse_nd = _as_ndarray(coarse)
    src_nd = _as_ndarray(source)
    weight_nd = _as_ndarray(weight)
    acc_nd = _as_ndarray(accumulator)
    best_nd = _as_ndarray(best)
    win_nd = _as_ndarray(winner)
    factor_f = float(factor)
    first_b = bool(first)
    idx_u = int(index)

    def band(start, stop, scratch):
        cancel()
        native.of_pyr_level0_merge_from_raw(
            coarse_nd, src_nd, weight_nd, acc_nd, best_nd, win_nd,
            _as_ndarray(scratch), factor_f, first_b, idx_u, int(start), int(stop))

    run_bands(height, _PYRAMID_BAND, band,
              scratch_bytes=int(np.prod(scratch_shape)) * 4,
              workspace=lambda: np.empty(scratch_shape, np.float32))


def _merge_level(native, detail, weights, accumulator, best, winner, first, index, cancel):
    height = weights.shape[0]
    if native is not None and weights.strides[1] == 2 and best.strides[1] == 2:
        d_nd = _as_ndarray(detail)
        w_nd = _as_ndarray(weights)
        a_nd = _as_ndarray(accumulator)
        b_nd = _as_ndarray(best)
        win_nd = _as_ndarray(winner)
        first_b = bool(first)
        idx_u = int(index)
        _rows(height, lambda a, b: native.of_pyr_merge_f32(
            d_nd, w_nd, a_nd, b_nd, win_nd, first_b, idx_u, int(a), int(b)), cancel)
    else:
        _rows(height, lambda a, b: _merge_rows(
            detail, weights, accumulator, best, first, a, b, winner, index), cancel)


def fuse_pyramid(frames, shape, img_resize, smoothing, should_cancel, progress_callback=None,
                 focus_callback=None, *, live_preview=None):
    """Fuse ordered sources in mapped, cancellable pyramid passes."""
    from core._native import load_native
    from core.source_frames import without_source_cache_growth
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods.openfocus_bc import _pyramid_levels, _source_codes
    from fusion_methods.openfocus_depth import _filter

    def cancel():
        _check_cancel(should_cancel)

    is_u8 = all(frame.dtype == np.uint8 for frame in frames)
    is_gray = len(shape) == 2
    channels = 1 if is_gray else shape[2]
    levels = _pyramid_levels(*shape[:2])
    detail_levels = list(zip(levels, levels[1:]))
    accumulators = [create_frame((h, w, channels), np.float32) for h, w in levels[:-1]]
    maxima = [create_frame((h, w), np.uint16) for h, w in levels[:-1]]
    coarsest = create_frame((*levels[-1], channels), np.float32)
    winner = create_frame(shape[:2], np.uint16) if focus_callback is not None else None

    native = load_native()
    use_native_pyr = (
        native is not None
        and hasattr(native, 'of_pyr_down_from_raw')
        and _normalize is _DEFAULT_NORMALIZE
        and _down is _DEFAULT_DOWN
        and _synthesize is _DEFAULT_SYNTHESIZE
        and _merge_rows is _DEFAULT_MERGE_ROWS
    )
    if detail_levels:
        if use_native_pyr:
            rolling_even = (create_frame((levels[2][0] * levels[2][1] * channels,), np.float32)
                            if len(levels) >= 3 else None)
            rolling_odd = create_frame((levels[1][0] * levels[1][1] * channels,), np.float32)
            rolling = [rolling_even, rolling_odd]
            full = None
        else:
            rolling = [create_frame((h * w * channels,), np.float32) for h, w in levels[:2]]
            full = rolling[0].reshape((*shape[:2], channels))
        weight_plane = create_frame((shape[0] * shape[1],), np.uint16)
    else:
        rolling = None
        full = coarsest
    total = len(frames) + 1
    if progress_callback is not None:
        progress_callback(0, total)
    parameter = np.float32(max(0.0, (float(smoothing) - 1.0) / 5.0))

    for index, frame in enumerate(frames):
        cancel()
        with without_source_cache_growth():
            pixels, scale = _source_codes(frame, img_resize, should_cancel)
        if pixels.shape != shape:
            raise ValueError('OpenFocus C requires frames with matching shapes')
        if detail_levels or index == 0:
            source = pixels[:, :, None] if is_gray else pixels
            factor = np.float32(scale / (65280.0 if is_u8 else 65535.0))
            if (use_native_pyr and detail_levels
                    and source.dtype in (np.uint8, np.uint16)):
                (height0, width0), (ch0, cw0) = detail_levels[0]
                coarse0 = (coarsest if (index == 0 and len(detail_levels) == 1)
                           else rolling[1][:ch0 * cw0 * channels].reshape(ch0, cw0, channels))
                _down_from_raw(native, source, coarse0, factor, cancel)
                weights0 = weight_plane[:height0 * width0].reshape(height0, width0)
                _level0_weights_from_raw(native, coarse0, source, weights0, factor, cancel)
                if parameter > 0:
                    radius = np.float32(np.float32(max(height0, width0)) * parameter / np.float32(100))
                    _filter(weights0, radius, cancel, overwrite=True)
                level_winner = winner
                _level0_merge_from_raw(
                    native, coarse0, source, weights0, accumulators[0],
                    maxima[0], level_winner, factor, index == 0, index, cancel)
                del source, pixels
                fine = coarse0
                start_level = 1
            else:
                if full is None:
                    full = create_frame((*shape[:2], channels), np.float32)
                _normalize(source, full, factor, cancel)
                del source, pixels
                fine = full
                start_level = 0
        else:
            del pixels
            fine = full
            start_level = 0

        for level in range(start_level, len(detail_levels)):
            (height, width), (ch, cw) = detail_levels[level]
            cancel()
            # Coarsest weights depend only on geometry. Keep the first plane.
            coarse = (coarsest if index == 0 and level == len(detail_levels) - 1
                      else rolling[(level + 1) % 2][:ch * cw * channels].reshape(ch, cw, channels))
            _down(fine, coarse, cancel)
            weights = weight_plane[:height * width].reshape(height, width)
            _synthesize(coarse, fine, weights, cancel)
            if parameter > 0:
                radius = np.float32(np.float32(max(height, width)) * parameter / np.float32(100))
                _filter(weights, radius, cancel, overwrite=True)
            level_winner = winner if level == 0 else None
            _merge_level(native if use_native_pyr else None, fine, weights,
                         accumulators[level], maxima[level], level_winner,
                         index == 0, index, cancel)
            fine = coarse
        if live_preview is not None:
            if detail_levels:
                live_preview.update_best(maxima[0], index, len(frames))
            else:
                live_preview.show_positions(np.zeros((1, 1), np.uint8), index, len(frames), 1)
        if progress_callback is not None:
            progress_callback(index + 1, total)
    del maxima, full, fine
    if detail_levels:
        del rolling, weight_plane
    elif winner is not None:
        winner.fill(0)
    result = coarsest
    del coarsest
    while accumulators:
        cancel()
        detail = accumulators.pop()
        _synthesize(result, detail, None, cancel)
        result = detail
    output_shape = (shape[0], shape[1], channels)
    output = create_frame(output_shape, np.uint8 if is_u8 else np.float32)
    if (native is not None and hasattr(native, 'of_pyr_output_f32')
            and _output_rows is _DEFAULT_OUTPUT_ROWS
            and _output_rows_flat is _DEFAULT_OUTPUT_ROWS_FLAT):
        res_nd = _as_ndarray(result)
        out_nd = _as_ndarray(output)
        _rows(shape[0], lambda a, b: native.of_pyr_output_f32(
            res_nd, out_nd, bool(is_u8), int(a), int(b)), cancel)
    elif result.flags.c_contiguous and output.flags.c_contiguous:
        res_flat = result.reshape(-1)
        out_flat = output.reshape(-1)
        _rows(shape[0], lambda a, b: _output_rows_flat(
            res_flat, out_flat, is_u8, shape[1] * channels, a, b), cancel)
    else:
        _rows(shape[0], lambda a, b: _output_rows(result, output, is_u8, a, b), cancel)
    if progress_callback is not None:
        progress_callback(total, total)
    cancel()
    if focus_callback is not None:
        # Each pyramid level makes its own selection. Export the actual
        # finest-level detail winner, not a claim about the collapsed color.
        focus_callback(winner, len(frames) - 1)
    return output[:, :, 0] if is_gray else output
