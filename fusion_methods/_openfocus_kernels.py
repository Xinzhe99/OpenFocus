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


A_MAX_RESPONSE = 65535 * 16
A_BLEND_LIMIT = 65536
A_SATURATED, A_SHARP, A_BACKGROUND = range(3)


@njit(nogil=True)
def _burt_laplacian_row(gray, reduced, output, y):
    """Absolute Burt difference in Q4; quarter-phase weights are exact."""
    height, width = reduced.shape
    fine_width = gray.shape[1]
    if fine_width == 0:
        return
    y0 = y // 2 - (1 if y % 2 == 0 else 0)
    y1 = min(y0 + 1, height - 1)
    y0 = max(y0, 0)
    wy0 = 1 if y % 2 == 0 else 3
    wy1 = 4 - wy0

    # Interpolate vertically first; each interior pair shares columns.
    left = np.int64(reduced[y0, 0]) * wy0 + np.int64(reduced[y1, 0]) * wy1
    output[0] = abs(np.int64(gray[y, 0]) * 16 - left * 4)
    for x in range(1, width):
        right = np.int64(reduced[y0, x]) * wy0 + np.int64(reduced[y1, x]) * wy1
        odd = 2 * x - 1
        output[odd] = abs(np.int64(gray[y, odd]) * 16 - (3 * left + right))
        output[odd + 1] = abs(np.int64(gray[y, odd + 1]) * 16 - (left + 3 * right))
        left = right
    if fine_width % 2 == 0:
        output[fine_width - 1] = abs(np.int64(gray[y, fine_width - 1]) * 16 - left * 4)


@njit(nogil=True)
def burt_laplacian_rows(gray, reduced, output, start, stop):
    for y in range(start, stop):
        _burt_laplacian_row(gray, reduced, output[y], y)


@njit(nogil=True)
def _burt_accumulate_row(laplacians, offset, height, width, response, level, y):
    """Add one rounded, half-pixel bilinear expansion in Q4."""
    fine_width = response.size
    if fine_width == 0:
        return
    shift = level + 1
    factor = 1 << level
    den = 1 << shift
    py = 2 * y + 1 - factor
    y0 = py >> shift
    fy = py - (y0 << shift)
    row_a = offset + max(0, min(y0, height - 1)) * width
    row_b = offset + max(0, min(y0 + 1, height - 1)) * width
    left = np.int64(laplacians[row_a]) * (den - fy) + np.int64(laplacians[row_b]) * fy
    if level == 1:
        # The two-pixel spans use complementary horizontal weights.
        response[0] += (left + 2) >> 2
        for x in range(1, width):
            right = (np.int64(laplacians[row_a + x]) * (den - fy)
                     + np.int64(laplacians[row_b + x]) * fy)
            odd = 2 * x - 1
            response[odd] += (3 * left + right + 8) >> 4
            response[odd + 1] += (left + 3 * right + 8) >> 4
            left = right
        if fine_width % 2 == 0:
            response[fine_width - 1] += (left + 2) >> 2
        return

    end = min(factor >> 1, fine_width)
    value = (left + factor) >> shift
    for x in range(end):
        response[x] += value

    # Each span starts at horizontal weight 1/den, then advances by 2/den.
    # Hoist clamping and both vertical interpolations out of the pixel loop.
    rounding = 1 << (2 * shift - 1)
    for column in range(1, width):
        right = (np.int64(laplacians[row_a + column]) * (den - fy)
                 + np.int64(laplacians[row_b + column]) * fy)
        delta = right - left
        initial = (left << shift) + delta + rounding
        span = min(factor, fine_width - end)
        for dx in range(span):
            response[end + dx] += (initial + 2 * delta * dx) >> (2 * shift)
        end += span
        left = right

    value = (left + factor) >> shift
    for x in range(end, fine_width):
        response[x] += value


@njit(nogil=True)
def select_rows(image, gray, half_gray, laplacians, levels, response, best, output,
                start, stop, scale, first, background):
    """Finish each response row in scratch before consuming its winning codes."""
    state = A_SATURATED
    channels = 1 if image.ndim == 2 else image.shape[2]
    weight_column, stat_column = channels, channels + 1
    statistics = background.view(np.uint32)
    for y in range(start, stop):
        local_y = y - start
        _burt_laplacian_row(gray, half_gray, response, y)
        height, width = half_gray.shape
        offset = 0
        for level in range(1, levels):
            _burt_accumulate_row(laplacians, offset, height, width, response, level, y)
            offset += height * width
            height, width = (height + 1) // 2, (width + 1) // 2

        for x in range(response.size):
            previous = 0 if first else np.int64(best[y, x])
            if previous == A_MAX_RESPONSE:
                continue
            # Every contribution is nonnegative, so clamp only the final sum.
            score = min(np.int64(response[x]), A_MAX_RESPONSE)
            take = first or score > previous
            collect = previous < A_BLEND_LIMIT and score < A_BLEND_LIMIT
            if collect:
                state = A_BACKGROUND
            elif state == A_SATURATED and score != A_MAX_RESPONSE:
                state = A_SHARP
            if not (collect or take):
                continue
            factor = np.float32(1.0)
            weight = np.float32(0.0)
            if collect:
                current_code = score >> 4
                if first:
                    statistics[local_y, x, stat_column] = current_code
                    weight = np.float32(1.0 if score else 0.0)
                    background[local_y, x, weight_column] = weight
                else:
                    # Low responses fit 12 bits in code units. Keep the
                    # minimum and runner-up in one slot beside the sums.
                    packed = np.int64(statistics[local_y, x, stat_column])
                    minimum = packed & 4095
                    second = packed >> 12
                    if score == 0:
                        if minimum:
                            statistics[local_y, x, stat_column] = second << 12
                        continue  # Zero cannot win and adds no weight.
                    if take:
                        second = previous >> 4
                        ratio = np.float32(previous) / np.float32(score)
                        factor = ratio * ratio
                        factor *= factor
                        weight = np.float32(1.0)
                        background[local_y, x, weight_column] = (
                            background[local_y, x, weight_column] * factor + weight)
                    else:
                        second = max(second, current_code)
                        ratio = np.float32(score) / np.float32(previous)
                        weight = ratio * ratio
                        weight *= weight
                        background[local_y, x, weight_column] += weight
                    statistics[local_y, x, stat_column] = (second << 12) | min(minimum, current_code)
            for channel in range(channels):
                if image.ndim == 2:
                    value = np.int64(image[y, x]) * scale
                else:
                    value = np.int64(image[y, x, channel]) * scale
                if collect:
                    weighted = np.float32(value) * weight
                    if not first:
                        weighted = background[local_y, x, channel] * factor + weighted
                    background[local_y, x, channel] = weighted
                if take:
                    if image.ndim == 2:
                        output[y, x] = value
                    else:
                        output[y, x, channel] = value
            if take:
                best[y, x] = score
    return state


@njit(nogil=True)
def blend_low_response_rows(output, best, background, start, stop):
    channels = 1 if output.ndim == 2 else output.shape[2]
    statistics = background.view(np.uint32)
    for y in range(start, stop):
        local_y = y - start
        for x in range(best.shape[1]):
            score = np.int64(best[y, x])
            if score >= A_BLEND_LIMIT:
                continue
            packed = np.int64(statistics[local_y, x, channels + 1])
            if (score >> 4) == (packed & 4095):
                continue
            second = (packed >> 12) << 4
            gap = max(0, score - second)
            if gap * 2 >= score:
                continue
            # Blend only when the two leading focus responses are close.
            confidence = min(score, 5 * score - 10 * gap)
            strength = (np.float32(min(32768, A_BLEND_LIMIT - score)) / np.float32(32768)
                        * np.float32(confidence) / np.float32(score))
            denominator = background[local_y, x, channels]
            if denominator <= 0:
                continue
            if output.ndim == 2:
                average = background[local_y, x, 0] / denominator
                value = np.float32(output[y, x]) * (np.float32(1.0) - strength) + average * strength
                output[y, x] = min(65535, max(0, int(value + np.float32(0.5))))
            else:
                for channel in range(channels):
                    average = background[local_y, x, channel] / denominator
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
