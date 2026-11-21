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
def table_rows(image, table, output, start, stop):
    """Map every sample through its channel's column of an integer table.

    A two-dimensional image is one plane of samples that all use column 0.
    """
    if image.ndim == 2:
        for y in range(start, stop):
            for x in range(image.shape[1]):
                output[y, x] = table[image[y, x], 0]
    else:
        for y in range(start, stop):
            for x in range(image.shape[1]):
                for channel in range(image.shape[2]):
                    output[y, x, channel] = table[image[y, x, channel], channel]


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
def burt_laplacian_rows(gray, reduced, output, start, stop, clamp=False):
    """Absolute Burt difference in Q4; quarter-phase weights are exact."""
    height, width = reduced.shape
    fine_width = gray.shape[1]
    if fine_width == 0:
        return
    for y in range(start, stop):
        y0 = y // 2 - (1 if y % 2 == 0 else 0)
        y1 = min(y0 + 1, height - 1)
        y0 = max(y0, 0)
        wy0 = 1 if y % 2 == 0 else 3
        wy1 = 4 - wy0

        # Interpolate vertically first. Every pair within the coarse grid then
        # uses the same two columns with complementary quarter-phase weights.
        edge = np.int64(reduced[y0, 0]) * wy0 + np.int64(reduced[y1, 0]) * wy1
        value = abs(np.int64(gray[y, 0]) * 16 - edge * 4)
        output[y, 0] = min(value, 65535 * 16) if clamp else value
        for x in range(1, width):
            left = np.int64(reduced[y0, x - 1]) * wy0 + np.int64(reduced[y1, x - 1]) * wy1
            right = np.int64(reduced[y0, x]) * wy0 + np.int64(reduced[y1, x]) * wy1
            odd = 2 * x - 1
            value = abs(np.int64(gray[y, odd]) * 16 - (3 * left + right))
            output[y, odd] = min(value, 65535 * 16) if clamp else value
            value = abs(np.int64(gray[y, odd + 1]) * 16 - (left + 3 * right))
            output[y, odd + 1] = min(value, 65535 * 16) if clamp else value
        if fine_width % 2 == 0:
            edge = (np.int64(reduced[y0, width - 1]) * wy0
                    + np.int64(reduced[y1, width - 1]) * wy1)
            value = abs(np.int64(gray[y, fine_width - 1]) * 16 - edge * 4)
            output[y, fine_width - 1] = min(value, 65535 * 16) if clamp else value


@njit(nogil=True)
def burt_accumulate_rows(laplacian, response, level, start, stop):
    """Accumulate the rounded, half-pixel bilinear expansion in Q4."""
    height, width = laplacian.shape
    fine_width = response.shape[1]
    if fine_width == 0:
        return
    if level == 1:
        # The two-pixel spans are the hottest case. Adjacent output pixels
        # share four coarse codes and use complementary horizontal weights.
        for y in range(start, stop):
            y0 = y // 2 - (1 if y % 2 == 0 else 0)
            y1 = min(y0 + 1, height - 1)
            y0 = max(y0, 0)
            wy0 = 1 if y % 2 == 0 else 3
            wy1 = 4 - wy0
            edge = np.int64(laplacian[y0, 0]) * wy0 + np.int64(laplacian[y1, 0]) * wy1
            value = (edge + 2) >> 2
            response[y, 0] = min(np.int64(response[y, 0]) + value, 65535 * 16)
            for x in range(1, width):
                left = (np.int64(laplacian[y0, x - 1]) * wy0
                        + np.int64(laplacian[y1, x - 1]) * wy1)
                right = np.int64(laplacian[y0, x]) * wy0 + np.int64(laplacian[y1, x]) * wy1
                odd = 2 * x - 1
                value = (3 * left + right + 8) >> 4
                response[y, odd] = min(np.int64(response[y, odd]) + value, 65535 * 16)
                value = (left + 3 * right + 8) >> 4
                response[y, odd + 1] = min(np.int64(response[y, odd + 1]) + value, 65535 * 16)
            if fine_width % 2 == 0:
                edge = (np.int64(laplacian[y0, width - 1]) * wy0
                        + np.int64(laplacian[y1, width - 1]) * wy1)
                value = (edge + 2) >> 2
                response[y, fine_width - 1] = min(
                    np.int64(response[y, fine_width - 1]) + value, 65535 * 16)
        return

    shift = level + 1
    factor = 1 << level
    den = 1 << shift
    phase = factor >> 1
    rounding = 1 << (2 * shift - 1)
    for y in range(start, stop):
        py = 2 * y + 1 - factor
        y0 = py >> shift
        fy = py - (y0 << shift)
        ya = max(0, min(y0, height - 1))
        yb = max(0, min(y0 + 1, height - 1))
        left = np.int64(laplacian[ya, 0]) * (den - fy) + np.int64(laplacian[yb, 0]) * fy
        end = min(phase, fine_width)
        value = (left + factor) >> shift
        for x in range(end):
            response[y, x] = min(np.int64(response[y, x]) + value, 65535 * 16)

        # Each span starts at horizontal weight 1/den, then advances by 2/den.
        # Hoist clamping and both vertical interpolations out of the pixel loop.
        for column in range(1, width):
            right = (np.int64(laplacian[ya, column]) * (den - fy)
                     + np.int64(laplacian[yb, column]) * fy)
            delta = right - left
            initial = (left << shift) + delta + rounding
            span = min(factor, fine_width - end)
            for dx in range(span):
                value = (initial + 2 * delta * dx) >> (2 * shift)
                x = end + dx
                response[y, x] = min(np.int64(response[y, x]) + value, 65535 * 16)
            end += span
            left = right

        value = (left + factor) >> shift
        for x in range(end, fine_width):
            response[y, x] = min(np.int64(response[y, x]) + value, 65535 * 16)


@njit(nogil=True)
def select_rows(image, response, best, output, start, stop, scale, first,
                accumulated, weights, statistics):
    for y in range(start, stop):
        local_y = y - start
        for x in range(response.shape[1]):
            score = np.int64(response[y, x])
            previous = 0 if first else np.int64(best[y, x])
            take = first or score > previous
            # A pixel that has reached the sharp threshold can never blend.
            collect = previous < 65536 and score < 65536
            if not (collect or take):
                continue
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


@njit(nogil=True)
def active_source_rows(depth, step, active, start, stop):
    """Mark every source that a depth code in these rows gives merge weight.

    A code selects source code // step (clipped to the stack), and also the
    next source when it falls between two sources.
    """
    last = active.shape[0] - 1
    for y in range(start, stop):
        for x in range(depth.shape[1]):
            code = np.int64(depth[y, x])
            lower = min(max(code // step, 0), last)
            active[lower] = True
            if code % step != 0:
                active[min(lower + 1, last)] = True
