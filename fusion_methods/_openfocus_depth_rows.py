"""Exact depth elementwise rows; callers allocate and bound cancellation work.

Use signed int64 scalars for differences, products and snap corrections.
Stored code planes contain final 16-bit values; _depth_kernel retains uint32
Q12 state and guided forward history independently of these planes.
"""
import numpy as np

from fusion_methods._depth_kernel import njit


def reciprocal32(divisor):
    """Round 2**32 / divisor up for bounded, exact unsigned division."""
    return ((1 << 32) + divisor - 1) // divisor


_ACTIVITY_RECIPROCALS = np.array([reciprocal32(d) for d in range(10, 111)],
                                 dtype=np.int64)


@njit(nogil=True, inline='always')
def _normalized_difference(value, mean):
    maximum = max(value, mean)
    denominator = (maximum * 100 * np.int64(0x80008001)) >> 47
    numerator = 110 * abs(value - mean)
    # The upward error is below 110*65535 / 2**32, smaller than 1/110.
    # Thus it cannot cross the next integer for any of these denominators.
    return min(65535, (numerator * _ACTIVITY_RECIPROCALS[denominator]) >> 32)


@njit(nogil=True)
def activity_rows(gray, fine, output, start, stop):
    for y in range(start, stop):
        for x in range(gray.shape[1]):
            output[y, x] = _normalized_difference(np.int64(gray[y, x]),
                                                  np.int64(fine[y, x]))


@njit(nogil=True)
def _reflect(position, length):
    if length == 1:
        return 0
    period = 2 * (length - 1)
    position %= period
    return period - position if position >= length else position


@njit(nogil=True)
def detail_rows(gray, output, row_offset):
    """Pool normalized full-resolution high-frequency energy onto the depth grid."""
    height, width = gray.shape
    for y in range(output.shape[0]):
        y0 = min(2 * (y + row_offset), height - 1)
        y1 = min(y0 + 1, height - 1)
        for x in range(output.shape[1]):
            x0 = min(2 * x, width - 1)
            x1 = min(2 * x + 1, width - 1)
            a = np.int64(gray[y0, x0])
            b = np.int64(gray[y0, x1])
            c = np.int64(gray[y1, x0])
            d = np.int64(gray[y1, x1])
            mean = (a + b + c + d) // 4
            energy = 0
            for value in (a, b, c, d):
                energy += _normalized_difference(value, mean)
            output[y, x] = energy // 4


@njit(nogil=True)
def smooth_detail_rows(detail, output, start, stop, columns=None):
    """Slide an integer sum vertically and horizontally over reflected edges."""
    if start >= stop:
        return
    height, width = detail.shape
    if columns is None:
        columns = np.empty(width, np.int32)
    r0 = _reflect(start - 2, height)
    r1 = _reflect(start - 1, height)
    r3 = _reflect(start + 1, height)
    r4 = _reflect(start + 2, height)
    for x in range(width):
        columns[x] = (np.int32(detail[r0, x]) + np.int32(detail[r1, x])
                      + np.int32(detail[start, x]) + np.int32(detail[r3, x])
                      + np.int32(detail[r4, x]))
    for y in range(start, stop):
        total = 0
        for dx in range(-2, 3):
            total += columns[_reflect(dx, width)]
        for x in range(width):
            output[y, x] = total // 25
            if x + 1 < width:
                entering = x + 3 if x + 3 < width else _reflect(x + 3, width)
                leaving = x - 2 if x >= 2 else _reflect(x - 2, width)
                total += columns[entering] - columns[leaving]
        if y + 1 < stop:
            entering = _reflect(y + 3, height)
            leaving = _reflect(y - 2, height)
            for x in range(width):
                columns[x] += (np.int32(detail[entering, x])
                               - np.int32(detail[leaving, x]))


@njit(nogil=True)
def initialize_detail_rows(second, winner, start, stop):
    for y in range(start, stop):
        for x in range(second.shape[1]):
            second[y, x] = 0
            winner[y, x] = 0


@njit(nogil=True)
def select_detail_rows(activity, best, second, winner, code, start, stop):
    for y in range(start, stop):
        for x in range(activity.shape[1]):
            value = activity[y, x]
            if value > best[y, x]:
                second[y, x] = best[y, x]
                best[y, x] = value
                winner[y, x] = code
            elif value > second[y, x]:
                second[y, x] = value


@njit(nogil=True)
def detail_confidence_rows(coarse, best, second, start, stop):
    for y in range(start, stop):
        for x in range(coarse.shape[1]):
            top = np.int64(best[y, x])
            gap = top - np.int64(second[y, x])
            second[y, x] = (top >= 128 and top > 4 * (np.int64(coarse[y, x]) + 1)
                            and 2 * gap > top)


@njit(nogil=True)
def apply_detail_rows(depth, mask, winner, start, stop):
    for y in range(start, stop):
        for x in range(depth.shape[1]):
            if mask[y, x]:
                depth[y, x] = winner[y, x]


@njit(nogil=True)
def initialize_activity_rows(activity, depth, votes, reciprocal, start, stop):
    for y in range(start, stop):
        for x in range(activity.shape[1]):
            depth[y, x] = 0
            if votes is not None:
                # 65535 / 2**32 < 1 / count for every supported stack size.
                votes[y, x] = (np.int64(activity[y, x]) * reciprocal) >> 32


@njit(nogil=True)
def select_activity_rows(activity, best, depth, votes, code, reciprocal, start, stop):
    for y in range(start, stop):
        for x in range(activity.shape[1]):
            value = activity[y, x]
            if votes is not None:
                votes[y, x] += (np.int64(value) * reciprocal) >> 32
            if value > best[y, x]:
                best[y, x] = value
                depth[y, x] = code


@njit(nogil=True)
def confidence_hist_rows(best, votes, histogram, start, stop):
    for y in range(start, stop):
        for x in range(best.shape[1]):
            confidence = abs(np.int64(best[y, x]) - np.int64(votes[y, x]))
            best[y, x] = confidence
            histogram[confidence] += 1


@njit(nogil=True)
def guide_rows(best, lut, start, stop):
    ratio = np.float32(4095.0 / 65535.0)
    for y in range(start, stop):
        for x in range(best.shape[1]):
            position = np.float32(np.float32(best[y, x]) * ratio)
            index = min(4094, max(0, np.int32(position)))
            fraction = np.float32(position - np.float32(index))
            left = np.float32(lut[index] * np.float32(np.float32(1) - fraction))
            right = np.float32(lut[index + 1] * fraction)
            value = np.float32(left + right)
            best[y, x] = min(65535, max(0, np.int64(value)))




@njit(nogil=True)
def snap_rows(depth, lut, step, start, stop):
    half = step >> 1
    for y in range(start, stop):
        for x in range(depth.shape[1]):
            block = np.int64(depth[y, x])
            total = half + block
            snapped = total - total % step
            distance = block - snapped
            signed = ((abs(distance) << 10) // half) * (-1 if distance < 0 else 1)
            index = min(1024, abs(signed))
            correction = half * lut[index] * (1 if signed > 0 else -1)
            correction = ((correction >> 10) if correction >= 0
                          else -((-correction) >> 10))
            depth[y, x] = (snapped + correction) & 65535
