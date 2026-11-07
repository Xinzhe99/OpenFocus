"""Exact integer rows for HF9 native source preparation and pyramid merges.

The caller owns allocation and checks cancellation between bounded row calls.
Use the existing admitted compiler import; frozen installs need no disk cache.
"""
import numpy as np
from fusion_methods._depth_kernel import njit


@njit(nogil=True)
def gray_rows(image, output, start, stop, scale):
    for y in range(start, stop):
        for x in range(image.shape[1]):
            if image.ndim == 2:
                value = np.int64(image[y, x])
            elif image.shape[2] >= 3:
                value = (np.int64(image[y, x, 0]) + np.int64(image[y, x, 1])
                         + np.int64(image[y, x, 2])) // 3
            elif image.shape[2] == 1:
                value = np.int64(image[y, x, 0])
            else:
                value = (np.int64(image[y, x, 0])
                         + np.int64(image[y, x, 1])) // 3
            output[y, x] = value * scale


@njit(nogil=True)
def down2_rows(image, output, start, stop, scale=1):
    height, width = image.shape[:2]
    for y in range(start, stop):
        top = 2 * y
        bottom = min(top + 1, height - 1)
        for x in range(output.shape[1]):
            left = 2 * x
            right = min(left + 1, width - 1)
            if image.ndim == 2:
                total = (np.int64(image[top, left]) + np.int64(image[top, right])
                         + np.int64(image[bottom, left]) + np.int64(image[bottom, right]))
                output[y, x] = (total * scale) >> 2
            else:
                for channel in range(image.shape[2]):
                    total = (np.int64(image[top, left, channel])
                             + np.int64(image[top, right, channel])
                             + np.int64(image[bottom, left, channel])
                             + np.int64(image[bottom, right, channel]))
                    output[y, x, channel] = (total * scale) >> 2


@njit(nogil=True)
def burt_laplacian_rows(gray, reduced, output, start, stop):
    """Absolute Burt difference in Q4; quarter-phase weights are exact."""
    height, width = reduced.shape
    for y in range(start, stop):
        y0 = y // 2 - (1 if y % 2 == 0 else 0)
        y1 = min(y0 + 1, height - 1)
        y0 = max(y0, 0)
        wy0 = 1 if y % 2 == 0 else 3
        wy1 = 4 - wy0
        for x in range(gray.shape[1]):
            x0 = x // 2 - (1 if x % 2 == 0 else 0)
            x1 = min(x0 + 1, width - 1)
            x0 = max(x0, 0)
            wx0 = 1 if x % 2 == 0 else 3
            wx1 = 4 - wx0
            top = np.int64(reduced[y0, x0]) * wx0 + np.int64(reduced[y0, x1]) * wx1
            bottom = np.int64(reduced[y1, x0]) * wx0 + np.int64(reduced[y1, x1]) * wx1
            value = np.int64(gray[y, x]) * 16 - top * wy0 - bottom * wy1
            output[y, x] = abs(value)


@njit(nogil=True)
def burt_accumulate_rows(laplacian, response, level, start, stop):
    for y in range(start, stop):
        row = y >> level
        for x in range(response.shape[1]):
            value = np.int64(response[y, x]) + np.int64(laplacian[row, x >> level])
            response[y, x] = min(value, 65535 * 16)


@njit(nogil=True)
def select_rows(image, response, best, output, start, stop, scale, first):
    for y in range(start, stop):
        for x in range(response.shape[1]):
            score = response[y, x]
            if first or score > best[y, x]:
                best[y, x] = score
                if image.ndim == 2:
                    output[y, x] = np.int64(image[y, x]) * scale
                else:
                    for channel in range(image.shape[2]):
                        output[y, x, channel] = np.int64(image[y, x, channel]) * scale


@njit(nogil=True)
def merge_rows(accumulator, image, depth, index, step, start, stop, scale=1):
    for y in range(start, stop):
        for x in range(depth.shape[1]):
            weight = 1024
            if index != 0:
                delta = np.int64(depth[y, x]) - step * index
                if delta >= 0:
                    if delta > step:
                        continue
                else:
                    if delta <= -step:
                        continue
                    weight = 1024 - ((-delta << 10) // step)
                    if weight <= 0:
                        continue
            if image.ndim == 2:
                value = np.int64(image[y, x]) * scale
                if weight == 1024:
                    accumulator[y, x] = value
                else:
                    accumulator[y, x] = ((np.int64(accumulator[y, x]) * (1024 - weight)
                                          + value * weight) >> 10)
            else:
                for channel in range(image.shape[2]):
                    value = np.int64(image[y, x, channel]) * scale
                    if weight == 1024:
                        accumulator[y, x, channel] = value
                    else:
                        accumulator[y, x, channel] = (
                            (np.int64(accumulator[y, x, channel]) * (1024 - weight)
                             + value * weight) >> 10)


@njit(nogil=True)
def equal_code_rows(first, second, start, stop, first_scale=1, second_scale=1):
    for y in range(start, stop):
        for x in range(first.shape[1]):
            if first.ndim == 2:
                if np.int64(first[y, x]) * first_scale != np.int64(second[y, x]) * second_scale:
                    return False
            else:
                for channel in range(first.shape[2]):
                    if (np.int64(first[y, x, channel]) * first_scale
                            != np.int64(second[y, x, channel]) * second_scale):
                        return False
    return True


@njit(nogil=True)
def expand_rows(image, output, axis, start, stop):
    height, width = image.shape[:2]
    for y in range(start, stop):
        for phase in range(2):
            shift = -2 if phase == 0 else -1
            w0 = 2 if phase == 0 else 12
            w1 = 37 if phase == 0 else 64
            w2 = 64 if phase == 0 else 37
            w3 = 12 if phase == 0 else 2
            for x in range(width):
                if axis == 0:
                    a0, a1 = max(y + shift, 0), max(y + shift + 1, 0)
                    a2, a3 = min(y + shift + 2, height - 1), min(y + shift + 3, height - 1)
                    a0, a1 = min(a0, height - 1), min(a1, height - 1)
                    a2, a3 = max(a2, 0), max(a3, 0)
                    oy, ox = 2 * y + phase, x
                    if image.ndim == 2:
                        total = (np.int64(image[a0, x]) * w0 + np.int64(image[a1, x]) * w1
                                 + np.int64(image[a2, x]) * w2 + np.int64(image[a3, x]) * w3)
                        output[oy, ox] = (total + 57) // 115
                    else:
                        for channel in range(image.shape[2]):
                            total = (np.int64(image[a0, x, channel]) * w0
                                     + np.int64(image[a1, x, channel]) * w1
                                     + np.int64(image[a2, x, channel]) * w2
                                     + np.int64(image[a3, x, channel]) * w3)
                            output[oy, ox, channel] = (total + 57) // 115
                else:
                    a0, a1 = max(x + shift, 0), max(x + shift + 1, 0)
                    a2, a3 = min(x + shift + 2, width - 1), min(x + shift + 3, width - 1)
                    a0, a1 = min(a0, width - 1), min(a1, width - 1)
                    a2, a3 = max(a2, 0), max(a3, 0)
                    oy, ox = y, 2 * x + phase
                    if image.ndim == 2:
                        total = (np.int64(image[y, a0]) * w0 + np.int64(image[y, a1]) * w1
                                 + np.int64(image[y, a2]) * w2 + np.int64(image[y, a3]) * w3)
                        output[oy, ox] = (total + 57) // 115
                    else:
                        for channel in range(image.shape[2]):
                            total = (np.int64(image[y, a0, channel]) * w0
                                     + np.int64(image[y, a1, channel]) * w1
                                     + np.int64(image[y, a2, channel]) * w2
                                     + np.int64(image[y, a3, channel]) * w3)
                            output[oy, ox, channel] = (total + 57) // 115


@njit(nogil=True)
def collapse_rows(blend, lowpass, output, start, stop):
    for y in range(start, stop):
        for x in range(blend.shape[1]):
            if blend.ndim == 2:
                difference = np.int64(blend[y, x]) - np.int64(lowpass[y, x])
                canvas = min(65535, max(0, 32768 + ((difference + 1) >> 1)))
                output[y, x] = min(65535, max(0, 2 * canvas + np.int64(lowpass[y, x]) - 65536))
            else:
                for channel in range(blend.shape[2]):
                    difference = (np.int64(blend[y, x, channel])
                                  - np.int64(lowpass[y, x, channel]))
                    canvas = min(65535, max(0, 32768 + ((difference + 1) >> 1)))
                    output[y, x, channel] = min(65535, max(
                        0, 2 * canvas + np.int64(lowpass[y, x, channel]) - 65536))
