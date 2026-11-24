"""Exact integer rows for OpenFocus source preparation and weighted merges.

The caller owns allocation and checks cancellation between bounded row calls.
Use the existing admitted compiler import; frozen installs need no disk cache.
"""

import numpy as np
from fusion_methods._depth_kernel import njit


A_MAX_RESPONSE = 65535 * 16
A_BLEND_LIMIT = 65536
A_SATURATED, A_SHARP, A_BACKGROUND = range(3)

_BURT_MAX_HALF_WIDTH = 65536
# Small renders seldom recover the extra time to compile the cached-row kernel.
_BURT_CACHE_MIN_PIXELS = 512 * 512


@njit(nogil=True)
def _nb_gray_rows(image, output, start, stop, scale, depth_color=False):
    if start >= stop or image.size == 0:
        return
    width = image.shape[1]
    is_u8_u16 = isinstance(image.flat[0], (np.uint8, np.uint16))
    if image.ndim == 2:
        for y in range(start, stop):
            row = image[y]
            out = output[y]
            for x in range(width):
                out[x] = np.int64(row[x]) * scale
        return
    channels = image.shape[2]
    if channels >= 3:
        if depth_color:
            if is_u8_u16 and 0 <= scale <= 256:
                s32 = np.uint32(scale * 85)
                for y in range(start, stop):
                    row = image[y]
                    out = output[y]
                    for x in range(width):
                        total = (np.uint32(row[x, 0]) + np.uint32(row[x, 1])
                                 + np.uint32(row[x, 2]))
                        out[x] = (total * s32) >> np.uint32(8)
            else:
                for y in range(start, stop):
                    row = image[y]
                    out = output[y]
                    for x in range(width):
                        total = (np.int64(row[x, 0]) + np.int64(row[x, 1])
                                 + np.int64(row[x, 2]))
                        out[x] = (total * scale * 85) >> 8
        elif is_u8_u16:
            if scale == 1:
                for y in range(start, stop):
                    row = image[y]
                    out = output[y]
                    for x in range(width):
                        total = (np.uint32(row[x, 0]) + np.uint32(row[x, 1])
                                 + np.uint32(row[x, 2]))
                        out[x] = total // np.uint32(3)
            elif scale == 256:
                for y in range(start, stop):
                    row = image[y]
                    out = output[y]
                    for x in range(width):
                        total = (np.uint32(row[x, 0]) + np.uint32(row[x, 1])
                                 + np.uint32(row[x, 2]))
                        out[x] = (total // np.uint32(3)) << np.uint32(8)
            else:
                for y in range(start, stop):
                    row = image[y]
                    out = output[y]
                    for x in range(width):
                        total = (np.uint32(row[x, 0]) + np.uint32(row[x, 1])
                                 + np.uint32(row[x, 2]))
                        out[x] = np.int64(total // np.uint32(3)) * scale
        else:
            for y in range(start, stop):
                row = image[y]
                out = output[y]
                for x in range(width):
                    total = (np.int64(row[x, 0]) + np.int64(row[x, 1])
                             + np.int64(row[x, 2]))
                    out[x] = (total // 3) * scale
    elif channels == 1:
        for y in range(start, stop):
            row = image[y]
            out = output[y]
            for x in range(width):
                out[x] = np.int64(row[x, 0]) * scale
    else:
        for y in range(start, stop):
            row = image[y]
            out = output[y]
            for x in range(width):
                value = (np.int64(row[x, 0]) + np.int64(row[x, 1])) // 3
                out[x] = value * scale


@njit(nogil=True)
def table_rows(image, table, output, start, stop):
    """Map every sample through its channel's column of an integer table.

    A two-dimensional image is one plane of samples that all use column 0.
    """
    if start >= stop or image.size == 0:
        return
    width = image.shape[1]
    if image.ndim == 2:
        t0 = table[:, 0]
        for y in range(start, stop):
            row = image[y]
            out = output[y]
            for x in range(width):
                out[x] = t0[row[x]]
    elif image.shape[2] == 3:
        for y in range(start, stop):
            row = image[y]
            out = output[y]
            for x in range(width):
                out[x, 0] = table[row[x, 0], 0]
                out[x, 1] = table[row[x, 1], 1]
                out[x, 2] = table[row[x, 2], 2]
    else:
        channels = image.shape[2]
        for y in range(start, stop):
            row = image[y]
            out = output[y]
            for x in range(width):
                for channel in range(channels):
                    out[x, channel] = table[row[x, channel], channel]


@njit(nogil=True)
def _nb_down2_rows(image, output, start, stop, scale=1):
    if start >= stop or image.size == 0 or output.size == 0:
        return
    height, width = image.shape[:2]
    out_w = output.shape[1]
    full_w = width // 2
    if full_w > out_w:
        full_w = out_w
    is_u8_u16 = isinstance(image.flat[0], (np.uint8, np.uint16))
    if image.ndim == 2:
        if scale == 1 and is_u8_u16:
            for y in range(start, stop):
                top = 2 * y
                bottom = top + 1 if top + 1 < height else height - 1
                r0 = image[top]
                r1 = image[bottom]
                out = output[y]
                for x in range(full_w):
                    left = 2 * x
                    out[x] = (np.uint32(r0[left]) + np.uint32(r0[left + 1])
                              + np.uint32(r1[left]) + np.uint32(r1[left + 1])) >> np.uint32(2)
                for x in range(full_w, out_w):
                    left = 2 * x
                    right = left + 1 if left + 1 < width else width - 1
                    out[x] = (np.uint32(r0[left]) + np.uint32(r0[right])
                              + np.uint32(r1[left]) + np.uint32(r1[right])) >> np.uint32(2)
        elif scale == 1:
            for y in range(start, stop):
                top = 2 * y
                bottom = top + 1 if top + 1 < height else height - 1
                r0 = image[top]
                r1 = image[bottom]
                out = output[y]
                for x in range(full_w):
                    left = 2 * x
                    out[x] = (np.int64(r0[left]) + np.int64(r0[left + 1])
                              + np.int64(r1[left]) + np.int64(r1[left + 1])) >> 2
                for x in range(full_w, out_w):
                    left = 2 * x
                    right = left + 1 if left + 1 < width else width - 1
                    out[x] = (np.int64(r0[left]) + np.int64(r0[right])
                              + np.int64(r1[left]) + np.int64(r1[right])) >> 2
        else:
            for y in range(start, stop):
                top = 2 * y
                bottom = top + 1 if top + 1 < height else height - 1
                r0 = image[top]
                r1 = image[bottom]
                out = output[y]
                for x in range(full_w):
                    left = 2 * x
                    total = (np.int64(r0[left]) + np.int64(r0[left + 1])
                             + np.int64(r1[left]) + np.int64(r1[left + 1]))
                    out[x] = (total * scale) >> 2
                for x in range(full_w, out_w):
                    left = 2 * x
                    right = left + 1 if left + 1 < width else width - 1
                    total = (np.int64(r0[left]) + np.int64(r0[right])
                             + np.int64(r1[left]) + np.int64(r1[right]))
                    out[x] = (total * scale) >> 2
    else:
        channels = image.shape[2]
        for y in range(start, stop):
            top = 2 * y
            bottom = top + 1 if top + 1 < height else height - 1
            r0 = image[top]
            r1 = image[bottom]
            out = output[y]
            for x in range(full_w):
                left = 2 * x
                for channel in range(channels):
                    total = (np.int64(r0[left, channel])
                             + np.int64(r0[left + 1, channel])
                             + np.int64(r1[left, channel])
                             + np.int64(r1[left + 1, channel]))
                    out[x, channel] = (total * scale) >> 2
            for x in range(full_w, out_w):
                left = 2 * x
                right = left + 1 if left + 1 < width else width - 1
                for channel in range(channels):
                    total = (np.int64(r0[left, channel])
                             + np.int64(r0[right, channel])
                             + np.int64(r1[left, channel])
                             + np.int64(r1[right, channel]))
                    out[x, channel] = (total * scale) >> 2


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
    g_row = gray[y]
    r0 = reduced[y0]
    r1 = reduced[y1]
    left = np.int64(r0[0]) * wy0 + np.int64(r1[0]) * wy1
    output[0] = abs(np.int64(g_row[0]) * 16 - left * 4)
    for x in range(1, width):
        right = np.int64(r0[x]) * wy0 + np.int64(r1[x]) * wy1
        odd = 2 * x - 1
        output[odd] = abs(np.int64(g_row[odd]) * 16 - (3 * left + right))
        output[odd + 1] = abs(np.int64(g_row[odd + 1]) * 16 - (left + 3 * right))
        left = right
    if fine_width % 2 == 0:
        output[fine_width - 1] = abs(np.int64(g_row[fine_width - 1]) * 16 - left * 4)


@njit(nogil=True)
def _nb_burt_laplacian_rows(gray, reduced, output, start, stop):
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
def _double_single_row_u32(raw_row, w, level, out, tmp0, tmp1, n_out):
    """Exact midpoint doubling of one coarse row into a half-grid padded buffer."""
    steps = level - 1
    src = tmp0
    dst = tmp1
    for x in range(w):
        src[x] = np.uint32(raw_row[x])
    cur_len = w
    max_pairs = n_out // 2 + 1
    cap = n_out + 2
    for _ in range(steps):
        limit = cur_len - 1
        if limit > max_pairs:
            limit = max_pairs
        for x in range(limit):
            a = src[x]
            b = src[x + 1]
            dst[2 * x] = np.uint32(a + a)
            dst[2 * x + 1] = np.uint32(a + b)
        last_idx = 2 * limit
        if last_idx < cap:
            a = src[limit]
            dst[last_idx] = np.uint32(a + a)
            cur_len = last_idx + 1
        else:
            cur_len = cap
        src, dst = dst, src
    pad = 1 << (level - 2)
    v_first = src[0]
    v_last = src[cur_len - 1]
    left_end = pad if pad < n_out else n_out
    for j in range(left_end):
        out[j] = v_first
    copy_end = pad + cur_len
    if copy_end > n_out:
        copy_end = n_out
    for j in range(left_end, copy_end):
        out[j] = src[j - pad]
    for j in range(copy_end, n_out):
        out[j] = v_last


@njit(nogil=True)
def _update_level_pair_u32(laps, off, width, level, ya, yb, cur_ya, cur_yb,
                           da, db, tmp0, tmp1, n_out):
    if ya == cur_ya and yb == cur_yb:
        return cur_ya, cur_yb
    if ya == cur_yb:
        for j in range(n_out):
            da[j] = db[j]
    elif ya != cur_ya:
        _double_single_row_u32(laps[off + ya * width:off + ya * width + width],
                               width, level, da, tmp0, tmp1, n_out)
    if yb == ya:
        for j in range(n_out):
            db[j] = da[j]
    elif yb != cur_yb:
        _double_single_row_u32(laps[off + yb * width:off + yb * width + width],
                               width, level, db, tmp0, tmp1, n_out)
    return ya, yb


@njit(nogil=True)
def _burt_response_row_generic(gray, half_gray, laplacians, levels, response, y):
    _burt_laplacian_row(gray, half_gray, response, y)
    height, width = half_gray.shape
    offset = 0
    for level in range(1, levels):
        _burt_accumulate_row(laplacians, offset, height, width, response, level, y)
        offset += height * width
        height, width = (height + 1) // 2, (width + 1) // 2


@njit(nogil=True)
def _burt_combine_l0_l5_u32(gray, half_gray, laps, response, y,
                            m0, m1, m2, m3, m4, m5,
                            da2, db2, da3, db3, da4, db4, da5, db5,
                            tmp0, tmp1, n_out, full_pairs, odd_width,
                            h0, w0, off1, h2, w2, off2, h3, w3, off3,
                            h4, w4, off4, h5, w5, off5,
                            cya2, cyb2, cya3, cyb3, cya4, cyb4, cya5, cyb5):
    y0 = y // 2 - (1 if (y & 1) == 0 else 0)
    y1 = y0 + 1 if y0 + 1 < h0 else h0 - 1
    if y0 < 0:
        y0 = 0
    wy0 = np.uint32(1 if (y & 1) == 0 else 3)
    wy1 = np.uint32(4) - wy0
    r0 = half_gray[y0]
    r1 = half_gray[y1]
    for x in range(w0):
        m0[x + 1] = np.uint32(np.uint32(r0[x]) * wy0 + np.uint32(r1[x]) * wy1)
    m0[0] = m0[1]
    m0[w0 + 1] = m0[w0]

    py = 2 * y - 1
    cy0 = py >> 2
    fy1 = np.uint32(py - (cy0 << 2))
    wy_1 = np.uint32(4) - fy1
    ya = 0 if cy0 < 0 else (h0 - 1 if cy0 >= h0 else cy0)
    yb = 0 if cy0 + 1 < 0 else (h0 - 1 if cy0 + 1 >= h0 else cy0 + 1)
    ra = laps[off1 + ya * w0:off1 + ya * w0 + w0]
    rb = laps[off1 + yb * w0:off1 + yb * w0 + w0]
    for x in range(w0):
        m1[x + 1] = np.uint32(np.uint32(ra[x]) * wy_1 + np.uint32(rb[x]) * fy1)
    m1[0] = m1[1]
    m1[w0 + 1] = m1[w0]

    py = 2 * y - 3
    cy0 = py >> 3
    fy2 = np.uint32(py - (cy0 << 3))
    wy2 = np.uint32(8) - fy2
    ya = 0 if cy0 < 0 else (h2 - 1 if cy0 >= h2 else cy0)
    yb = 0 if cy0 + 1 < 0 else (h2 - 1 if cy0 + 1 >= h2 else cy0 + 1)
    cya2, cyb2 = _update_level_pair_u32(
        laps, off2, w2, 2, ya, yb, cya2, cyb2, da2, db2, tmp0, tmp1, n_out)

    py = 2 * y - 7
    cy0 = py >> 4
    fy3 = np.uint32(py - (cy0 << 4))
    wy3 = np.uint32(16) - fy3
    ya = 0 if cy0 < 0 else (h3 - 1 if cy0 >= h3 else cy0)
    yb = 0 if cy0 + 1 < 0 else (h3 - 1 if cy0 + 1 >= h3 else cy0 + 1)
    cya3, cyb3 = _update_level_pair_u32(
        laps, off3, w3, 3, ya, yb, cya3, cyb3, da3, db3, tmp0, tmp1, n_out)

    py = 2 * y - 15
    cy0 = py >> 5
    fy4 = np.uint32(py - (cy0 << 5))
    wy4 = np.uint32(32) - fy4
    ya = 0 if cy0 < 0 else (h4 - 1 if cy0 >= h4 else cy0)
    yb = 0 if cy0 + 1 < 0 else (h4 - 1 if cy0 + 1 >= h4 else cy0 + 1)
    cya4, cyb4 = _update_level_pair_u32(
        laps, off4, w4, 4, ya, yb, cya4, cyb4, da4, db4, tmp0, tmp1, n_out)

    py = 2 * y - 31
    cy0 = py >> 6
    fy5 = np.uint32(py - (cy0 << 6))
    wy5 = np.uint32(64) - fy5
    ya = 0 if cy0 < 0 else (h5 - 1 if cy0 >= h5 else cy0)
    yb = 0 if cy0 + 1 < 0 else (h5 - 1 if cy0 + 1 >= h5 else cy0 + 1)
    cya5, cyb5 = _update_level_pair_u32(
        laps, off5, w5, 5, ya, yb, cya5, cyb5, da5, db5, tmp0, tmp1, n_out)

    for j in range(n_out):
        m2[j] = np.uint32(np.uint32(da2[j] * wy2) + np.uint32(db2[j] * fy2))
        m3[j] = np.uint32(np.uint32(da3[j] * wy3) + np.uint32(db3[j] * fy3))
        m4[j] = np.uint32(np.uint32(da4[j] * wy4) + np.uint32(db4[j] * fy4))
        m5[j] = np.uint32(np.uint32(da5[j] * wy5) + np.uint32(db5[j] * fy5))

    g_row = gray[y]
    for j in range(full_pairs):
        ev = 2 * j
        a0_l = m0[j]
        a0_c = m0[j + 1]
        a0_r = m0[j + 2]
        a1_l = m1[j]
        a1_c = m1[j + 1]
        a1_r = m1[j + 2]
        d0 = np.int32(np.int32(np.uint32(g_row[ev]) << np.uint32(4))
                      - np.int32(a0_l + np.uint32(a0_c * np.uint32(3))))
        d1 = np.int32(np.int32(np.uint32(g_row[ev + 1]) << np.uint32(4))
                      - np.int32(np.uint32(a0_c * np.uint32(3)) + a0_r))
        s0 = np.int32(-d0) if d0 < np.int32(0) else d0
        s1 = np.int32(-d1) if d1 < np.int32(0) else d1
        s0 = np.int32(s0 + np.int32(
            np.uint32(a1_l + np.uint32(a1_c * np.uint32(3)) + np.uint32(8)) >> np.uint32(4)))
        s1 = np.int32(s1 + np.int32(
            np.uint32(np.uint32(a1_c * np.uint32(3)) + a1_r + np.uint32(8)) >> np.uint32(4)))
        l2 = m2[j]
        r2 = m2[j + 1]
        l3 = m3[j]
        r3 = m3[j + 1]
        l4 = m4[j]
        r4 = m4[j + 1]
        l5 = m5[j]
        r5 = m5[j + 1]
        s0 = np.int32(s0 + np.int32(
            np.uint32(np.uint32(l2 * np.uint32(3)) + r2 + np.uint32(32)) >> np.uint32(6)))
        s1 = np.int32(s1 + np.int32(
            np.uint32(l2 + np.uint32(r2 * np.uint32(3)) + np.uint32(32)) >> np.uint32(6)))
        s0 = np.int32(s0 + np.int32(
            np.uint32(np.uint32(l3 * np.uint32(3)) + r3 + np.uint32(128)) >> np.uint32(8)))
        s1 = np.int32(s1 + np.int32(
            np.uint32(l3 + np.uint32(r3 * np.uint32(3)) + np.uint32(128)) >> np.uint32(8)))
        s0 = np.int32(s0 + np.int32(
            np.uint32(np.uint32(l4 * np.uint32(3)) + r4 + np.uint32(512)) >> np.uint32(10)))
        s1 = np.int32(s1 + np.int32(
            np.uint32(l4 + np.uint32(r4 * np.uint32(3)) + np.uint32(512)) >> np.uint32(10)))
        s0 = np.int32(s0 + np.int32(
            np.uint32(np.uint32(l5 * np.uint32(3)) + r5 + np.uint32(2048)) >> np.uint32(12)))
        s1 = np.int32(s1 + np.int32(
            np.uint32(l5 + np.uint32(r5 * np.uint32(3)) + np.uint32(2048)) >> np.uint32(12)))
        response[ev] = s0
        response[ev + 1] = s1

    if odd_width:
        j = full_pairs
        ev = 2 * j
        d0 = np.int32(np.int32(np.uint32(g_row[ev]) << np.uint32(4))
                      - np.int32(m0[j] + np.uint32(m0[j + 1] * np.uint32(3))))
        s0 = np.int32(-d0) if d0 < np.int32(0) else d0
        s0 = np.int32(s0 + np.int32(
            np.uint32(m1[j] + np.uint32(m1[j + 1] * np.uint32(3)) + np.uint32(8)) >> np.uint32(4)))
        s0 = np.int32(s0 + np.int32(
            np.uint32(np.uint32(m2[j] * np.uint32(3)) + m2[j + 1] + np.uint32(32)) >> np.uint32(6)))
        s0 = np.int32(s0 + np.int32(
            np.uint32(np.uint32(m3[j] * np.uint32(3)) + m3[j + 1] + np.uint32(128)) >> np.uint32(8)))
        s0 = np.int32(s0 + np.int32(
            np.uint32(np.uint32(m4[j] * np.uint32(3)) + m4[j + 1] + np.uint32(512)) >> np.uint32(10)))
        s0 = np.int32(s0 + np.int32(
            np.uint32(np.uint32(m5[j] * np.uint32(3)) + m5[j + 1] + np.uint32(2048)) >> np.uint32(12)))
        response[ev] = s0

    return cya2, cyb2, cya3, cyb3, cya4, cyb4, cya5, cyb5


@njit(nogil=True)
def _select_band_u32_3c_s1(image, gray, half_gray, laps, levels, response,
                           best, output, start, stop, first, background):
    state = A_SATURATED
    width = response.size
    h0, w0 = half_gray.shape
    n_out = (width + 1) // 2 + 1
    m0 = np.empty(w0 + 2, np.uint32)
    m1 = np.empty(w0 + 2, np.uint32)
    m2 = np.empty(n_out, np.uint32)
    m3 = np.empty(n_out, np.uint32)
    m4 = np.empty(n_out, np.uint32)
    m5 = np.empty(n_out, np.uint32)
    da2 = np.empty(n_out, np.uint32)
    db2 = np.empty(n_out, np.uint32)
    da3 = np.empty(n_out, np.uint32)
    db3 = np.empty(n_out, np.uint32)
    da4 = np.empty(n_out, np.uint32)
    db4 = np.empty(n_out, np.uint32)
    da5 = np.empty(n_out, np.uint32)
    db5 = np.empty(n_out, np.uint32)
    tmp0 = np.empty(n_out + 4, np.uint32)
    tmp1 = np.empty(n_out + 4, np.uint32)

    off1 = 0
    h2, w2 = (h0 + 1) // 2, (w0 + 1) // 2
    off2 = off1 + h0 * w0
    h3, w3 = (h2 + 1) // 2, (w2 + 1) // 2
    off3 = off2 + h2 * w2
    h4, w4 = (h3 + 1) // 2, (w3 + 1) // 2
    off4 = off3 + h3 * w3
    h5, w5 = (h4 + 1) // 2, (w4 + 1) // 2
    off5 = off4 + h4 * w4
    h6, w6 = (h5 + 1) // 2, (w5 + 1) // 2
    off6 = off5 + h5 * w5
    cya2 = cyb2 = cya3 = cyb3 = cya4 = cyb4 = cya5 = cyb5 = -1
    full_pairs = width // 2
    odd_width = (width & 1) == 1
    no_bg = (not first) and background.shape[0] == 0
    has_bg = background.shape[0] > 0
    statistics = background.view(np.uint32)

    for y in range(start, stop):
        cya2, cyb2, cya3, cyb3, cya4, cyb4, cya5, cyb5 = _burt_combine_l0_l5_u32(
            gray, half_gray, laps, response, y,
            m0, m1, m2, m3, m4, m5,
            da2, db2, da3, db3, da4, db4, da5, db5,
            tmp0, tmp1, n_out, full_pairs, odd_width,
            h0, w0, off1, h2, w2, off2, h3, w3, off3,
            h4, w4, off4, h5, w5, off5,
            cya2, cyb2, cya3, cyb3, cya4, cyb4, cya5, cyb5,
        )
        if levels > 6:
            hh, ww = h6, w6
            off = off6
            for lv in range(6, levels):
                _burt_accumulate_row(laps, off, hh, ww, response, lv, y)
                off += hh * ww
                hh, ww = (hh + 1) // 2, (ww + 1) // 2

        b_row = best[y]
        o_row = output[y]
        i_row = image[y]
        if no_bg:
            for x in range(width):
                prev = b_row[x]
                if prev == A_MAX_RESPONSE:
                    continue
                sc = response[x]
                if sc >= A_MAX_RESPONSE:
                    sc = A_MAX_RESPONSE
                elif state == A_SATURATED:
                    state = A_SHARP
                if sc > prev:
                    b_row[x] = sc
                    o_row[x, 0] = i_row[x, 0]
                    o_row[x, 1] = i_row[x, 1]
                    o_row[x, 2] = i_row[x, 2]
        elif first and has_bg:
            local_y = y - start
            bg_row = background[local_y]
            st_row = statistics[local_y]
            for x in range(width):
                sc = response[x]
                if sc > A_MAX_RESPONSE:
                    sc = A_MAX_RESPONSE
                v0 = i_row[x, 0]
                v1 = i_row[x, 1]
                v2 = i_row[x, 2]
                o_row[x, 0] = v0
                o_row[x, 1] = v1
                o_row[x, 2] = v2
                b_row[x] = sc
                if sc < A_BLEND_LIMIT:
                    state = A_BACKGROUND
                    st_row[x, 4] = np.uint32(sc >> 4)
                    if sc:
                        bg_row[x, 3] = np.float32(1.0)
                        bg_row[x, 0] = np.float32(v0)
                        bg_row[x, 1] = np.float32(v1)
                        bg_row[x, 2] = np.float32(v2)
                    else:
                        bg_row[x, 3] = np.float32(0.0)
                        bg_row[x, 0] = np.float32(0.0)
                        bg_row[x, 1] = np.float32(0.0)
                        bg_row[x, 2] = np.float32(0.0)
                elif state == A_SATURATED and sc != A_MAX_RESPONSE:
                    state = A_SHARP
        else:
            local_y = y - start
            bg_row = background[local_y]
            st_row = statistics[local_y]
            for x in range(width):
                prev = b_row[x]
                if prev == A_MAX_RESPONSE:
                    continue
                sc = response[x]
                if sc > A_MAX_RESPONSE:
                    sc = A_MAX_RESPONSE
                if prev >= A_BLEND_LIMIT:
                    if state == A_SATURATED and sc != A_MAX_RESPONSE:
                        state = A_SHARP
                    if sc > prev:
                        b_row[x] = sc
                        o_row[x, 0] = i_row[x, 0]
                        o_row[x, 1] = i_row[x, 1]
                        o_row[x, 2] = i_row[x, 2]
                elif sc >= A_BLEND_LIMIT:
                    if state == A_SATURATED and sc != A_MAX_RESPONSE:
                        state = A_SHARP
                    b_row[x] = sc
                    o_row[x, 0] = i_row[x, 0]
                    o_row[x, 1] = i_row[x, 1]
                    o_row[x, 2] = i_row[x, 2]
                else:
                    state = A_BACKGROUND
                    packed = st_row[x, 4]
                    minimum = packed & np.uint32(4095)
                    second = packed >> np.uint32(12)
                    if sc == 0:
                        if minimum:
                            st_row[x, 4] = second << np.uint32(12)
                        continue
                    current_code = np.uint32(sc >> 4)
                    if sc > prev:
                        second = np.uint32(prev >> 4)
                        ratio = np.float32(prev) / np.float32(sc)
                        factor = ratio * ratio
                        factor *= factor
                        bg_row[x, 3] = bg_row[x, 3] * factor + np.float32(1.0)
                        v0 = i_row[x, 0]
                        v1 = i_row[x, 1]
                        v2 = i_row[x, 2]
                        bg_row[x, 0] = bg_row[x, 0] * factor + np.float32(v0)
                        bg_row[x, 1] = bg_row[x, 1] * factor + np.float32(v1)
                        bg_row[x, 2] = bg_row[x, 2] * factor + np.float32(v2)
                        o_row[x, 0] = v0
                        o_row[x, 1] = v1
                        o_row[x, 2] = v2
                        b_row[x] = sc
                    else:
                        if current_code > second:
                            second = current_code
                        ratio = np.float32(sc) / np.float32(prev)
                        weight = ratio * ratio
                        weight *= weight
                        bg_row[x, 3] += weight
                        bg_row[x, 0] += np.float32(i_row[x, 0]) * weight
                        bg_row[x, 1] += np.float32(i_row[x, 1]) * weight
                        bg_row[x, 2] += np.float32(i_row[x, 2]) * weight
                    if current_code < minimum:
                        minimum = current_code
                    st_row[x, 4] = (second << np.uint32(12)) | minimum
    return state


@njit(nogil=True)
def _select_band_u32_generic(image, gray, half_gray, laps, levels, response,
                             best, output, start, stop, scale, first, background):
    state = A_SATURATED
    width = response.size
    h0, w0 = half_gray.shape
    n_out = (width + 1) // 2 + 1
    m0 = np.empty(w0 + 2, np.uint32)
    m1 = np.empty(w0 + 2, np.uint32)
    m2 = np.empty(n_out, np.uint32)
    m3 = np.empty(n_out, np.uint32)
    m4 = np.empty(n_out, np.uint32)
    m5 = np.empty(n_out, np.uint32)
    da2 = np.empty(n_out, np.uint32)
    db2 = np.empty(n_out, np.uint32)
    da3 = np.empty(n_out, np.uint32)
    db3 = np.empty(n_out, np.uint32)
    da4 = np.empty(n_out, np.uint32)
    db4 = np.empty(n_out, np.uint32)
    da5 = np.empty(n_out, np.uint32)
    db5 = np.empty(n_out, np.uint32)
    tmp0 = np.empty(n_out + 4, np.uint32)
    tmp1 = np.empty(n_out + 4, np.uint32)

    off1 = 0
    h2, w2 = (h0 + 1) // 2, (w0 + 1) // 2
    off2 = off1 + h0 * w0
    h3, w3 = (h2 + 1) // 2, (w2 + 1) // 2
    off3 = off2 + h2 * w2
    h4, w4 = (h3 + 1) // 2, (w3 + 1) // 2
    off4 = off3 + h3 * w3
    h5, w5 = (h4 + 1) // 2, (w4 + 1) // 2
    off5 = off4 + h4 * w4
    h6, w6 = (h5 + 1) // 2, (w5 + 1) // 2
    off6 = off5 + h5 * w5
    cya2 = cyb2 = cya3 = cyb3 = cya4 = cyb4 = cya5 = cyb5 = -1
    full_pairs = width // 2
    odd_width = (width & 1) == 1
    channels = 1 if image.ndim == 2 else image.shape[2]
    weight_column, stat_column = channels, channels + 1
    statistics = background.view(np.uint32)

    for y in range(start, stop):
        local_y = y - start
        cya2, cyb2, cya3, cyb3, cya4, cyb4, cya5, cyb5 = _burt_combine_l0_l5_u32(
            gray, half_gray, laps, response, y,
            m0, m1, m2, m3, m4, m5,
            da2, db2, da3, db3, da4, db4, da5, db5,
            tmp0, tmp1, n_out, full_pairs, odd_width,
            h0, w0, off1, h2, w2, off2, h3, w3, off3,
            h4, w4, off4, h5, w5, off5,
            cya2, cyb2, cya3, cyb3, cya4, cyb4, cya5, cyb5,
        )
        if levels > 6:
            hh, ww = h6, w6
            off = off6
            for lv in range(6, levels):
                _burt_accumulate_row(laps, off, hh, ww, response, lv, y)
                off += hh * ww
                hh, ww = (hh + 1) // 2, (ww + 1) // 2

        for x in range(width):
            previous = 0 if first else np.int64(best[y, x])
            if previous == A_MAX_RESPONSE:
                continue
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
                    packed = np.int64(statistics[local_y, x, stat_column])
                    minimum = packed & 4095
                    second = packed >> 12
                    if score == 0:
                        if minimum:
                            statistics[local_y, x, stat_column] = second << 12
                        continue
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


def burt_cache_bytes(gray, levels):
    """Size the expanded-row cache for the native selector."""
    half_width = (gray.shape[1] + 1) // 2 + 1
    if (gray.dtype != np.uint16 or levels < 6 or half_width > _BURT_MAX_HALF_WIDTH
            or gray.size < _BURT_CACHE_MIN_PIXELS):
        return 0
    # Six mixed rows, eight cached coarse rows, and two doubling buffers.
    return 4 * (16 * half_width + 10)


def _as_ndarray(arr):
    return arr if type(arr) is np.ndarray else arr.view(np.ndarray)


def _native_ok_arr(arr):
    return isinstance(arr, np.ndarray) and arr.dtype in (np.uint8, np.uint16, np.int32, np.uint32, np.float32)


def gray_rows(image, output, start, stop, scale, depth_color=False):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_gray_rows')
            and _native_ok_arr(image) and _native_ok_arr(output)):
        native.of_gray_rows(_as_ndarray(image), _as_ndarray(output),
                            int(start), int(stop), int(scale), bool(depth_color))
        return
    _nb_gray_rows(image, output, start, stop, scale, depth_color)


def down2_rows(image, output, start, stop, scale=1):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_down2_rows')
            and _native_ok_arr(image) and _native_ok_arr(output)):
        native.of_down2_rows(_as_ndarray(image), _as_ndarray(output),
                             int(start), int(stop), int(scale))
        return
    _nb_down2_rows(image, output, start, stop, scale)


def burt_laplacian_rows(gray, reduced, output, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_burt_laplacian_rows')
            and gray.dtype in (np.uint16, np.int32)
            and reduced.dtype in (np.uint16, np.int32)
            and output.dtype in (np.uint16, np.int32)):
        native.of_burt_laplacian_rows(_as_ndarray(gray), _as_ndarray(reduced),
                                      _as_ndarray(output), int(start), int(stop))
        return
    _nb_burt_laplacian_rows(gray, reduced, output, start, stop)


def select_rows(image, gray, half_gray, laplacians, levels, response, best, output,
                start, stop, scale, first, background):
    """Choose the native kernel before compiling a signature."""
    if start >= stop or response.size == 0 or gray.size == 0:
        return A_SATURATED
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_select_rows_a')
            and image.dtype in (np.uint8, np.uint16)
            and gray.dtype in (np.uint16, np.int32)
            and half_gray.dtype == np.int32
            and (laplacians.size == 0 or laplacians.dtype == np.int32)
            and best.dtype == np.int32
            and output.dtype == np.uint16
            and background.dtype == np.float32
            and gray.strides[-1] == gray.itemsize
            and half_gray.strides[-1] == half_gray.itemsize):
        return int(native.of_select_rows_a(
            _as_ndarray(image), _as_ndarray(gray), _as_ndarray(half_gray),
            _as_ndarray(laplacians), int(levels), _as_ndarray(best),
            _as_ndarray(output), int(start), int(stop), int(scale),
            bool(first), _as_ndarray(background),
        ))
    if burt_cache_bytes(gray, levels):
        if (image.ndim == 3 and image.shape[2] == 3 and scale == 1
                and output.ndim == 3 and (background.shape[0] > 0 or not first)):
            return _select_band_u32_3c_s1(
                image, gray, half_gray, laplacians, levels, response,
                best, output, start, stop, first, background,
            )
        return _select_band_u32_generic(
            image, gray, half_gray, laplacians, levels, response,
            best, output, start, stop, scale, first, background,
        )
    return _select_rows_scalar(image, gray, half_gray, laplacians, levels, response,
                               best, output, start, stop, scale, first, background)


@njit(nogil=True)
def _select_rows_scalar(image, gray, half_gray, laplacians, levels, response, best, output,
                        start, stop, scale, first, background):
    """Finish one response row at a time without caching expanded rows."""
    width = response.size

    state = A_SATURATED
    channels = 1 if image.ndim == 2 else image.shape[2]
    weight_column, stat_column = channels, channels + 1
    statistics = background.view(np.uint32)
    for y in range(start, stop):
        local_y = y - start
        _burt_response_row_generic(gray, half_gray, laplacians, levels, response, y)

        for x in range(width):
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
def _nb_blend_low_response_rows(output, best, background, start, stop):
    width = best.shape[1]
    statistics = background.view(np.uint32)
    if output.ndim == 2:
        for y in range(start, stop):
            local_y = y - start
            b_row = best[y]
            o_row = output[y]
            bg_row = background[local_y]
            st_row = statistics[local_y]
            for x in range(width):
                score = np.int64(b_row[x])
                if score >= A_BLEND_LIMIT:
                    continue
                packed = np.int64(st_row[x, 2])
                if (score >> 4) == (packed & 4095):
                    continue
                second = (packed >> 12) << 4
                gap = max(0, score - second)
                if gap * 2 >= score:
                    continue
                confidence = min(score, 5 * score - 10 * gap)
                strength = (np.float32(min(32768, A_BLEND_LIMIT - score)) / np.float32(32768)
                            * np.float32(confidence) / np.float32(score))
                denominator = bg_row[x, 1]
                if denominator <= 0:
                    continue
                average = bg_row[x, 0] / denominator
                value = np.float32(o_row[x]) * (np.float32(1.0) - strength) + average * strength
                o_row[x] = min(65535, max(0, int(value + np.float32(0.5))))
    else:
        channels = output.shape[2]
        for y in range(start, stop):
            local_y = y - start
            b_row = best[y]
            o_row = output[y]
            bg_row = background[local_y]
            st_row = statistics[local_y]
            for x in range(width):
                score = np.int64(b_row[x])
                if score >= A_BLEND_LIMIT:
                    continue
                packed = np.int64(st_row[x, channels + 1])
                if (score >> 4) == (packed & 4095):
                    continue
                second = (packed >> 12) << 4
                gap = max(0, score - second)
                if gap * 2 >= score:
                    continue
                confidence = min(score, 5 * score - 10 * gap)
                strength = (np.float32(min(32768, A_BLEND_LIMIT - score)) / np.float32(32768)
                            * np.float32(confidence) / np.float32(score))
                denominator = bg_row[x, channels]
                if denominator <= 0:
                    continue
                inv_strength = np.float32(1.0) - strength
                for channel in range(channels):
                    average = bg_row[x, channel] / denominator
                    value = np.float32(o_row[x, channel]) * inv_strength + average * strength
                    o_row[x, channel] = min(65535, max(0, int(value + np.float32(0.5))))


@njit(nogil=True)
def _nb_merge_rows(accumulator, image, depth, index, step, start, stop,
                   scale, first, reverse):
    width = depth.shape[1]
    if first:
        if image.ndim == 2:
            for y in range(start, stop):
                a_row = accumulator[y]
                i_row = image[y]
                for x in range(width):
                    a_row[x] = np.int64(i_row[x]) * scale
        elif image.shape[2] == 3 and scale == 1:
            for y in range(start, stop):
                a_row = accumulator[y]
                i_row = image[y]
                for x in range(width):
                    a_row[x, 0] = i_row[x, 0]
                    a_row[x, 1] = i_row[x, 1]
                    a_row[x, 2] = i_row[x, 2]
        elif image.shape[2] == 3:
            for y in range(start, stop):
                a_row = accumulator[y]
                i_row = image[y]
                for x in range(width):
                    a_row[x, 0] = np.int64(i_row[x, 0]) * scale
                    a_row[x, 1] = np.int64(i_row[x, 1]) * scale
                    a_row[x, 2] = np.int64(i_row[x, 2]) * scale
        else:
            channels = image.shape[2]
            for y in range(start, stop):
                a_row = accumulator[y]
                i_row = image[y]
                for x in range(width):
                    for channel in range(channels):
                        a_row[x, channel] = np.int64(i_row[x, channel]) * scale
        return

    if start >= stop or depth.size == 0:
        return
    base = np.int64(step) * np.int64(index)
    if image.ndim == 2:
        for y in range(start, stop):
            d_row = depth[y]
            a_row = accumulator[y]
            i_row = image[y]
            for x in range(width):
                weight = 1024
                delta = np.int64(d_row[x]) - base
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
                value = np.int64(i_row[x]) * scale
                if weight == 1024:
                    a_row[x] = value
                else:
                    a_row[x] = ((np.int64(a_row[x]) * (1024 - weight)
                                 + value * weight) >> 10)
        return

    use_recip = (accumulator.ndim == 3 and (not reverse) and 2 <= step <= 65535
                 and image.shape[2] == 3 and 1 <= scale <= 256
                 and isinstance(image.flat[0], (np.uint8, np.uint16)))
    if use_recip:
        recip48 = np.uint64(((1 << 48) + step - 1) // step)
        base32 = np.int32(base)
        step32 = np.int32(step)
        neg_step = -step32
        if scale == 1:
            for y in range(start, stop):
                d_row = depth[y]
                a_row = accumulator[y]
                i_row = image[y]
                for x in range(width):
                    delta = np.int32(d_row[x]) - base32
                    if delta >= 0:
                        if delta <= step32:
                            a_row[x, 0] = i_row[x, 0]
                            a_row[x, 1] = i_row[x, 1]
                            a_row[x, 2] = i_row[x, 2]
                    elif delta > neg_step:
                        w = np.int32(1024) - np.int32(
                            (np.uint64((-delta) << 10) * recip48) >> np.uint64(48))
                        if w > 0:
                            iw = np.int32(1024) - w
                            a_row[x, 0] = (np.int32(a_row[x, 0]) * iw + np.int32(i_row[x, 0]) * w) >> np.int32(10)
                            a_row[x, 1] = (np.int32(a_row[x, 1]) * iw + np.int32(i_row[x, 1]) * w) >> np.int32(10)
                            a_row[x, 2] = (np.int32(a_row[x, 2]) * iw + np.int32(i_row[x, 2]) * w) >> np.int32(10)
        else:
            s32 = np.int32(scale)
            for y in range(start, stop):
                d_row = depth[y]
                a_row = accumulator[y]
                i_row = image[y]
                for x in range(width):
                    delta = np.int32(d_row[x]) - base32
                    if delta >= 0:
                        if delta <= step32:
                            a_row[x, 0] = np.int32(i_row[x, 0]) * s32
                            a_row[x, 1] = np.int32(i_row[x, 1]) * s32
                            a_row[x, 2] = np.int32(i_row[x, 2]) * s32
                    elif delta > neg_step:
                        w = np.int32(1024) - np.int32(
                            (np.uint64((-delta) << 10) * recip48) >> np.uint64(48))
                        if w > 0:
                            iw = np.int32(1024) - w
                            v0 = np.int32(i_row[x, 0]) * s32
                            v1 = np.int32(i_row[x, 1]) * s32
                            v2 = np.int32(i_row[x, 2]) * s32
                            a_row[x, 0] = (np.int32(a_row[x, 0]) * iw + v0 * w) >> np.int32(10)
                            a_row[x, 1] = (np.int32(a_row[x, 1]) * iw + v1 * w) >> np.int32(10)
                            a_row[x, 2] = (np.int32(a_row[x, 2]) * iw + v2 * w) >> np.int32(10)
        return

    channels = image.shape[2]
    for y in range(start, stop):
        d_row = depth[y]
        a_row = accumulator[y]
        i_row = image[y]
        for x in range(width):
            weight = 1024
            delta = np.int64(d_row[x]) - base
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
            inv_w = 1024 - weight
            for channel in range(channels):
                value = np.int64(i_row[x, channel]) * scale
                if weight == 1024:
                    a_row[x, channel] = value
                else:
                    a_row[x, channel] = (
                        (np.int64(a_row[x, channel]) * inv_w
                         + value * weight) >> 10)


@njit(nogil=True)
def _nb_active_source_rows(depth, step, active, start, stop):
    """Mark every source that a depth code in these rows gives merge weight.

    A code selects source code // step (clipped to the stack), and also the
    next source when it falls between two sources.
    """
    if start >= stop or depth.size == 0:
        return
    last = active.shape[0] - 1
    width = depth.shape[1]
    if 1 <= step <= 65535 and isinstance(depth.flat[0], np.uint16):
        recip48 = np.uint64(((1 << 48) + step - 1) // step)
        step64 = np.uint64(step)
        for y in range(start, stop):
            d_row = depth[y]
            for x in range(width):
                code = np.uint64(d_row[x])
                raw_lower = np.int64((code * recip48) >> np.uint64(48))
                lower = raw_lower if raw_lower <= last else last
                active[lower] = True
                if code != np.uint64(raw_lower) * step64:
                    nxt = lower + 1
                    active[nxt if nxt <= last else last] = True
        return

    for y in range(start, stop):
        d_row = depth[y]
        for x in range(width):
            code = np.int64(d_row[x])
            lower = min(max(code // step, 0), last)
            active[lower] = True
            if code % step != 0:
                active[min(lower + 1, last)] = True


@njit(nogil=True)
def _nb_output_u16_to_f32_rows(code, output, start, stop, factor):
    """Convert uint16 code bands into float32 output without extra band copies."""
    width = code.shape[1]
    f32 = np.float32(factor)
    if code.ndim == 2:
        for y in range(start, stop):
            c_row = code[y]
            o_row = output[y]
            for x in range(width):
                o_row[x] = np.float32(c_row[x]) * f32
    elif code.shape[2] == 3:
        for y in range(start, stop):
            c_row = code[y]
            o_row = output[y]
            for x in range(width):
                o_row[x, 0] = np.float32(c_row[x, 0]) * f32
                o_row[x, 1] = np.float32(c_row[x, 1]) * f32
                o_row[x, 2] = np.float32(c_row[x, 2]) * f32
    else:
        channels = code.shape[2]
        for y in range(start, stop):
            c_row = code[y]
            o_row = output[y]
            for x in range(width):
                for ch in range(channels):
                    o_row[x, ch] = np.float32(c_row[x, ch]) * f32


@njit(nogil=True)
def _nb_output_u16_to_u8_rows(code, output, start, stop, e37=False):
    """Quantize uint16 code bands into uint8 output with optional E-37 offset."""
    width = code.shape[1]
    if code.ndim == 2:
        if e37:
            for y in range(start, stop):
                c_row = code[y]
                o_row = output[y]
                for x in range(width):
                    v = c_row[x] >> np.uint16(8)
                    o_row[x] = v - 1 if v > 0 else 0
        else:
            for y in range(start, stop):
                c_row = code[y]
                o_row = output[y]
                for x in range(width):
                    o_row[x] = c_row[x] >> np.uint16(8)
    elif code.shape[2] == 4:
        if e37:
            for y in range(start, stop):
                c_row = code[y]
                o_row = output[y]
                for x in range(width):
                    v0 = c_row[x, 0] >> np.uint16(8)
                    v1 = c_row[x, 1] >> np.uint16(8)
                    v2 = c_row[x, 2] >> np.uint16(8)
                    o_row[x, 0] = v0 - 1 if v0 > 0 else 0
                    o_row[x, 1] = v1 - 1 if v1 > 0 else 0
                    o_row[x, 2] = v2 - 1 if v2 > 0 else 0
                    o_row[x, 3] = c_row[x, 3] >> np.uint16(8)
        else:
            for y in range(start, stop):
                c_row = code[y]
                o_row = output[y]
                for x in range(width):
                    o_row[x, 0] = c_row[x, 0] >> np.uint16(8)
                    o_row[x, 1] = c_row[x, 1] >> np.uint16(8)
                    o_row[x, 2] = c_row[x, 2] >> np.uint16(8)
                    o_row[x, 3] = c_row[x, 3] >> np.uint16(8)
    else:
        channels = code.shape[2]
        if e37:
            for y in range(start, stop):
                c_row = code[y]
                o_row = output[y]
                for x in range(width):
                    for ch in range(channels):
                        v = c_row[x, ch] >> np.uint16(8)
                        o_row[x, ch] = v - 1 if v > 0 else 0
        else:
            for y in range(start, stop):
                c_row = code[y]
                o_row = output[y]
                for x in range(width):
                    for ch in range(channels):
                        o_row[x, ch] = c_row[x, ch] >> np.uint16(8)


def blend_low_response_rows(output, best, background, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_blend_low_response_rows')
            and output.dtype == np.uint16 and best.dtype == np.int32
            and background.dtype == np.float32):
        native.of_blend_low_response_rows(_as_ndarray(output), _as_ndarray(best),
                                          _as_ndarray(background), int(start), int(stop))
        return
    _nb_blend_low_response_rows(output, best, background, start, stop)


def merge_rows(accumulator, image, depth, index, step, start, stop,
               scale, first, reverse):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_merge_rows')
            and accumulator.dtype == np.uint16
            and image.dtype in (np.uint8, np.uint16)
            and depth.dtype == np.uint16 and depth.strides[1] == 2):
        native.of_merge_rows(_as_ndarray(accumulator), _as_ndarray(image),
                             _as_ndarray(depth), int(index), int(step),
                             int(start), int(stop), int(scale), bool(first), bool(reverse))
        return
    _nb_merge_rows(accumulator, image, depth, index, step, start, stop,
                   scale, first, reverse)


def active_source_rows(depth, step, active, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_active_source_rows')
            and depth.dtype == np.uint16 and depth.strides[1] == 2
            and active.dtype == np.bool_ and active.flags.c_contiguous):
        native.of_active_source_rows(_as_ndarray(depth), int(step),
                                     _as_ndarray(active.view(np.uint8)),
                                     int(start), int(stop))
        return
    _nb_active_source_rows(depth, step, active, start, stop)


def output_u16_to_f32_rows(code, output, start, stop, factor):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_output_u16_to_f32_rows')
            and code.dtype == np.uint16 and output.dtype == np.float32
            and (code.ndim != 2 or (code.strides[1] == 2 and output.strides[1] == 4))):
        native.of_output_u16_to_f32_rows(_as_ndarray(code), _as_ndarray(output),
                                         int(start), int(stop), float(factor))
        return
    _nb_output_u16_to_f32_rows(code, output, start, stop, factor)


def output_u16_to_u8_rows(code, output, start, stop, e37=False):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_output_u16_to_u8_rows')
            and code.dtype == np.uint16 and output.dtype == np.uint8
            and (code.ndim != 2 or (code.strides[1] == 2 and output.strides[1] == 1))):
        native.of_output_u16_to_u8_rows(_as_ndarray(code), _as_ndarray(output),
                                        int(start), int(stop), bool(e37))
        return
    _nb_output_u16_to_u8_rows(code, output, start, stop, e37)
