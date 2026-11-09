"""Exact integer rows for OpenFocus source preparation and weighted merges.

The caller owns allocation and checks cancellation between bounded row calls.
Use the existing admitted compiler import; frozen installs need no disk cache.
"""

import numpy as np
from fusion_methods._depth_kernel import njit




@njit(nogil=True)
def gray_rows(image, output, start, stop, scale, depth_color=False):
    for y in range(start, stop):
        for x in range(image.shape[1]):
            if image.ndim == 2:
                value = np.int64(image[y, x])
            elif image.shape[2] >= 3:
                total = (np.int64(image[y, x, 0]) + np.int64(image[y, x, 1])
                         + np.int64(image[y, x, 2]))
                # Match the depth color-code range without an intermediate image.
                if depth_color:
                    output[y, x] = (total * scale * 85) >> 8
                    continue
                value = total // 3
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
    """Accumulate half-pixel bilinear response without a full-size buffer."""
    if level == 0:
        for y in range(start, stop):
            for x in range(response.shape[1]):
                value = np.int64(response[y, x]) + np.int64(laplacian[y, x])
                response[y, x] = min(value, 65535 * 16)
        return
    shift = level + 1
    factor = 1 << level
    den = 1 << shift
    width, height = laplacian.shape[1], laplacian.shape[0]
    rounding = 1 << (2 * shift - 1)
    for y in range(start, stop):
        py = 2 * y + 1 - factor
        y0 = py >> shift
        fy = py - (y0 << shift)
        ya = max(0, min(y0, height - 1))
        yb = max(0, min(y0 + 1, height - 1))
        px = 1 - factor
        x0 = px >> shift
        fx = px - (x0 << shift)
        xa = max(0, min(x0, width - 1))
        xb = max(0, min(x0 + 1, width - 1))
        left = np.int64(laplacian[ya, xa]) * (den - fy) + np.int64(laplacian[yb, xa]) * fy
        right = np.int64(laplacian[ya, xb]) * (den - fy) + np.int64(laplacian[yb, xb]) * fy
        for x in range(response.shape[1]):
            value = ((left << shift) + (right - left) * fx + rounding) >> (2 * shift)
            response[y, x] = min(np.int64(response[y, x]) + value, 65535 * 16)
            fx += 2
            if fx >= den and x + 1 < response.shape[1]:
                fx -= den
                x0 += 1
                left = right
                xb = max(0, min(x0 + 1, width - 1))
                right = np.int64(laplacian[ya, xb]) * (den - fy) + np.int64(laplacian[yb, xb]) * fy


@njit(nogil=True)
def select_rows(image, response, best, output, start, stop, scale, first,
                accumulated, weights, statistics):
    for y in range(start, stop):
        local_y = y - start
        for x in range(response.shape[1]):
            score = np.int64(response[y, x])
            previous = np.int64(best[y, x])
            take = first or score > previous
            collect = first or previous < 65536
            if collect:
                # Responses are saturated at 65535*16. Store the minimum and
                # runner-up in code units to keep both in one uint16 pair.
                current_code = score >> 4
                if first:
                    statistics[local_y, x, 0] = current_code
                    statistics[local_y, x, 1] = 0
                else:
                    minimum = min(np.int64(statistics[local_y, x, 0]), current_code)
                    second = np.int64(statistics[local_y, x, 1])
                    if score > previous:
                        second = previous >> 4
                    else:
                        second = max(second, current_code)
                    statistics[local_y, x, 0] = minimum
                    statistics[local_y, x, 1] = second
                factor = np.float32(1.0)
                if first:
                    weight = np.float32(1.0 if score else 0.0)
                    weights[local_y, x] = weight
                elif score > previous:
                    ratio = np.float32(previous) / np.float32(score)
                    factor = ratio * ratio
                    factor *= factor
                    weight = np.float32(1.0)
                    weights[local_y, x] = weights[local_y, x] * factor + weight
                else:
                    ratio = (np.float32(score) / np.float32(previous)
                             if previous else np.float32(0.0))
                    weight = ratio * ratio
                    weight *= weight
                    weights[local_y, x] += weight
            if image.ndim == 2:
                value = np.int64(image[y, x]) * scale
                if collect:
                    if first:
                        accumulated[local_y, x] = np.float32(value) * weight
                    else:
                        accumulated[local_y, x] = (accumulated[local_y, x] * factor
                                                    + np.float32(value) * weight)
                if take:
                    output[y, x] = value
            else:
                for channel in range(image.shape[2]):
                    value = np.int64(image[y, x, channel]) * scale
                    if collect:
                        if first:
                            accumulated[local_y, x, channel] = np.float32(value) * weight
                        else:
                            accumulated[local_y, x, channel] = (
                                accumulated[local_y, x, channel] * factor
                                + np.float32(value) * weight)
                    if take:
                        output[y, x, channel] = value
            if take:
                best[y, x] = score


@njit(nogil=True)
def blend_low_response_rows(output, best, accumulated, weights, statistics, start, stop):
    for y in range(start, stop):
        local_y = y - start
        for x in range(best.shape[1]):
            score = np.int64(best[y, x])
            if score >= 65536 or (score >> 4) == statistics[local_y, x, 0]:
                continue
            second = np.int64(statistics[local_y, x, 1]) << 4
            gap = max(0, score - second)
            if gap * 2 >= score:
                continue
            # Blend only when the two leading focus responses are close.
            confidence = min(score, 5 * score - 10 * gap)
            strength = (np.float32(min(32768, 65536 - score)) / np.float32(32768)
                        * np.float32(confidence) / np.float32(score))
            denominator = weights[local_y, x]
            if denominator <= 0:
                continue
            if output.ndim == 2:
                average = accumulated[local_y, x] / denominator
                value = np.float32(output[y, x]) * (np.float32(1.0) - strength) + average * strength
                output[y, x] = min(65535, max(0, int(value + np.float32(0.5))))
            else:
                for channel in range(output.shape[2]):
                    average = accumulated[local_y, x, channel] / denominator
                    value = (np.float32(output[y, x, channel]) * (np.float32(1.0) - strength)
                             + average * strength)
                    output[y, x, channel] = min(65535, max(0, int(value + np.float32(0.5))))


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


