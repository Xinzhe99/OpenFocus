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
def select_activity_rows(activity, best, winner, index, start, stop):
    for y in range(start, stop):
        for x in range(activity.shape[1]):
            value = activity[y, x]
            if value > best[y, x]:
                best[y, x] = value
                winner[y, x] = index


@njit(nogil=True)
def guide_rows(best, lut, output, start, stop):
    ratio = np.float32(np.float32(65534) / np.float32(65535))
    for y in range(start, stop):
        for x in range(best.shape[1]):
            position = np.float32(np.float32(best[y, x]) * ratio)
            index = min(65533, max(0, np.int32(position)))
            fraction = np.float32(position - np.float32(index))
            left = np.float32(lut[index] * np.float32(np.float32(1) - fraction))
            right = np.float32(lut[index + 1] * fraction)
            value = np.float32(left + right)
            output[y, x] = np.int64(value) & 65535


@njit(nogil=True)
def winner_depth_rows(winner, output, step, start, stop):
    for y in range(start, stop):
        for x in range(winner.shape[1]):
            output[y, x] = np.int64(winner[y, x]) * step


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
            depth[y, x] = (snapped + ((correction + (correction & 1023)) >> 10)) & 65535
