"""Exact integer rows for HF9 source preparation and weighted merges.

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
def merge_rows(accumulator, image, depth, index, step, start, stop,
               scale, first, reverse):
    for y in range(start, stop):
        for x in range(depth.shape[1]):
            weight = 1024
            if not first:
                delta = np.int64(depth[y, x]) - step * index
                if reverse:
                    delta = -delta
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


