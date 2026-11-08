"""Exact depth elementwise rows; callers allocate and bound cancellation work.

Use signed int64 scalars for differences, products and snap corrections.
Stored code planes contain final 16-bit values; _depth_kernel retains uint32
Q12 state and guided forward history independently of these planes.
"""
import numpy as np

from fusion_methods._depth_kernel import njit


@njit(nogil=True)
def activity_rows(gray, fine, output, start, stop):
    for y in range(start, stop):
        for x in range(gray.shape[1]):
            g = np.int64(gray[y, x])
            f = np.int64(fine[y, x])
            maximum = max(g, f)
            denominator = ((maximum * 100 * np.int64(0x80008001)) >> 32) >> 15
            output[y, x] = min(65535, 110 * abs(g - f) // (denominator + 10))

@njit(nogil=True)
def _reflect(position, length):
    if length == 1:
        return 0
    period = 2 * (length - 1)
    position %= period
    return period - position if position >= length else position


@njit(nogil=True)
def detail_rows(gray, output, start, stop):
    """Pool normalized full-resolution high-frequency energy onto the depth grid."""
    height, width = gray.shape
    for y in range(start, stop):
        for x in range(output.shape[1]):
            y0 = min(2 * y, height - 1)
            y1 = min(2 * y + 1, height - 1)
            x0 = min(2 * x, width - 1)
            x1 = min(2 * x + 1, width - 1)
            a = np.int64(gray[y0, x0])
            b = np.int64(gray[y0, x1])
            c = np.int64(gray[y1, x0])
            d = np.int64(gray[y1, x1])
            mean = (a + b + c + d) // 4
            energy = 0
            for value in (a, b, c, d):
                maximum = max(value, mean)
                denominator = ((maximum * 100 * np.int64(0x80008001)) >> 32) >> 15
                energy += min(65535, 110 * abs(value - mean) // (denominator + 10))
            output[y, x] = energy // 4


@njit(nogil=True)
def smooth_detail_rows(detail, output, start, stop):
    height, width = detail.shape
    for y in range(start, stop):
        total = 0
        for dy in range(-2, 3):
            sy = _reflect(y + dy, height)
            for dx in range(-2, 3):
                total += np.int64(detail[sy, _reflect(dx, width)])
        for x in range(width):
            output[y, x] = total // 25
            if x + 1 < width:
                leaving = _reflect(x - 2, width)
                entering = _reflect(x + 3, width)
                for dy in range(-2, 3):
                    sy = _reflect(y + dy, height)
                    total += np.int64(detail[sy, entering]) - np.int64(detail[sy, leaving])


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
def initialize_activity_rows(activity, depth, votes, count, start, stop):
    for y in range(start, stop):
        for x in range(activity.shape[1]):
            depth[y, x] = 0
            votes[y, x] = np.int64(activity[y, x]) // count


@njit(nogil=True)
def select_activity_rows(activity, best, depth, votes, code, count, start, stop):
    for y in range(start, stop):
        for x in range(activity.shape[1]):
            value = activity[y, x]
            votes[y, x] = (np.int64(votes[y, x]) + np.int64(value) // count) & 65535
            if value > best[y, x]:
                best[y, x] = value
                depth[y, x] = code


@njit(nogil=True)
def confidence_rows(best, votes, start, stop):
    for y in range(start, stop):
        for x in range(best.shape[1]):
            best[y, x] = abs(np.int64(best[y, x]) - np.int64(votes[y, x]))


@njit(nogil=True)
def guide_rows(best, lut, output, start, stop):
    ratio = np.float32(4095.0 / 65535.0)
    for y in range(start, stop):
        for x in range(best.shape[1]):
            position = np.float32(np.float32(best[y, x]) * ratio)
            index = min(4094, max(0, np.int32(position)))
            fraction = np.float32(position - np.float32(index))
            left = np.float32(lut[index] * np.float32(np.float32(1) - fraction))
            right = np.float32(lut[index + 1] * fraction)
            value = np.float32(left + right)
            output[y, x] = min(65535, max(0, np.int64(value)))




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
