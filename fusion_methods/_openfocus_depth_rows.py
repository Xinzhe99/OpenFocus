"""Exact depth elementwise rows; callers allocate and bound cancellation work.

Use signed int64 scalars for differences, products and snap corrections.
Stored code planes contain final 16-bit values; _depth_kernel retains uint32
Q12 state and guided forward history independently of these planes.
"""
import numpy as np

from fusion_methods._depth_kernel import njit
from fusion_methods._openfocus_kernels import merge_rows, _nb_merge_rows


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
def _nb_activity_rows(gray, fine, output, start, stop):
    if start >= stop or gray.size == 0:
        return
    width = gray.shape[1]
    for y in range(start, stop):
        g_row = gray[y]
        f_row = fine[y]
        o_row = output[y]
        for x in range(width):
            o_row[x] = _normalized_difference(np.int64(g_row[x]),
                                              np.int64(f_row[x]))


@njit(nogil=True)
def _reflect(position, length):
    if length == 1:
        return 0
    period = 2 * (length - 1)
    position %= period
    return period - position if position >= length else position


@njit(nogil=True)
def _nb_detail_rows(gray, output, row_offset):
    """Pool normalized full-resolution high-frequency energy onto the depth grid."""
    if output.size == 0 or gray.size == 0:
        return
    height, width = gray.shape
    out_h, out_w = output.shape
    full_w = width // 2
    if full_w > out_w:
        full_w = out_w
    for y in range(out_h):
        y0 = 2 * (y + row_offset)
        if y0 >= height:
            y0 = height - 1
        y1 = y0 + 1 if y0 + 1 < height else height - 1
        r0 = gray[y0]
        r1 = gray[y1]
        out = output[y]
        for x in range(full_w):
            x0 = 2 * x
            a = np.int64(r0[x0])
            b = np.int64(r0[x0 + 1])
            c = np.int64(r1[x0])
            d = np.int64(r1[x0 + 1])
            mean = (a + b + c + d) >> 2
            energy = (_normalized_difference(a, mean)
                      + _normalized_difference(b, mean)
                      + _normalized_difference(c, mean)
                      + _normalized_difference(d, mean))
            out[x] = energy >> 2
        for x in range(full_w, out_w):
            x0 = 2 * x
            if x0 >= width:
                x0 = width - 1
            x1 = x0 + 1 if x0 + 1 < width else width - 1
            a = np.int64(r0[x0])
            b = np.int64(r0[x1])
            c = np.int64(r1[x0])
            d = np.int64(r1[x1])
            mean = (a + b + c + d) >> 2
            energy = (_normalized_difference(a, mean)
                      + _normalized_difference(b, mean)
                      + _normalized_difference(c, mean)
                      + _normalized_difference(d, mean))
            out[x] = energy >> 2


@njit(nogil=True)
def _nb_smooth_detail_rows(detail, output, start, stop, columns=None):
    """Slide an integer sum vertically and horizontally over reflected edges."""
    if start >= stop or detail.size == 0:
        return
    height, width = detail.shape
    if columns is None:
        columns = np.empty(width, np.int32)
    r0 = detail[_reflect(start - 2, height)]
    r1 = detail[_reflect(start - 1, height)]
    r2 = detail[start]
    r3 = detail[_reflect(start + 1, height)]
    r4 = detail[_reflect(start + 2, height)]
    for x in range(width):
        columns[x] = (np.int32(r0[x]) + np.int32(r1[x])
                      + np.int32(r2[x]) + np.int32(r3[x])
                      + np.int32(r4[x]))
    use_u64 = isinstance(detail.flat[0], (np.uint8, np.uint16))
    recip25 = np.uint64(0x51EB851F)
    for y in range(start, stop):
        out = output[y]
        if width >= 5 and use_u64:
            total = columns[2] + columns[1] + columns[0] + columns[1] + columns[2]
            out[0] = (np.uint64(total) * recip25) >> np.uint64(35)
            total += columns[3] - columns[2]
            out[1] = (np.uint64(total) * recip25) >> np.uint64(35)
            total += columns[4] - columns[1]
            for x in range(2, width - 3):
                out[x] = (np.uint64(total) * recip25) >> np.uint64(35)
                total += columns[x + 3] - columns[x - 2]
            for x in range(width - 3, width):
                out[x] = (np.uint64(total) * recip25) >> np.uint64(35)
                if x + 1 < width:
                    entering = x + 3 if x + 3 < width else _reflect(x + 3, width)
                    leaving = x - 2 if x >= 2 else _reflect(x - 2, width)
                    total += columns[entering] - columns[leaving]
        else:
            total = 0
            for dx in range(-2, 3):
                total += columns[_reflect(dx, width)]
            for x in range(width):
                out[x] = total // 25
                if x + 1 < width:
                    entering = x + 3 if x + 3 < width else _reflect(x + 3, width)
                    leaving = x - 2 if x >= 2 else _reflect(x - 2, width)
                    total += columns[entering] - columns[leaving]
        if y + 1 < stop:
            re = detail[_reflect(y + 3, height)]
            rl = detail[_reflect(y - 2, height)]
            for x in range(width):
                columns[x] += np.int32(re[x]) - np.int32(rl[x])


@njit(nogil=True)
def _nb_initialize_detail_rows(second, winner, start, stop):
    width = second.shape[1]
    for y in range(start, stop):
        s_row = second[y]
        w_row = winner[y]
        for x in range(width):
            s_row[x] = 0
            w_row[x] = 0


@njit(nogil=True)
def _nb_select_detail_rows(activity, best, second, winner, code, start, stop):
    width = activity.shape[1]
    for y in range(start, stop):
        a_row = activity[y]
        b_row = best[y]
        s_row = second[y]
        w_row = winner[y]
        for x in range(width):
            value = a_row[x]
            if value > b_row[x]:
                s_row[x] = b_row[x]
                b_row[x] = value
                w_row[x] = code
            elif value > s_row[x]:
                s_row[x] = value


@njit(nogil=True)
def _nb_detail_confidence_rows(coarse, best, second, start, stop):
    width = coarse.shape[1]
    for y in range(start, stop):
        c_row = coarse[y]
        b_row = best[y]
        s_row = second[y]
        for x in range(width):
            top = np.int64(b_row[x])
            gap = top - np.int64(s_row[x])
            s_row[x] = (top >= 128 and top > 4 * (np.int64(c_row[x]) + 1)
                        and 2 * gap > top)


@njit(nogil=True)
def _nb_apply_detail_rows(depth, mask, winner, start, stop):
    width = depth.shape[1]
    for y in range(start, stop):
        d_row = depth[y]
        m_row = mask[y]
        w_row = winner[y]
        for x in range(width):
            if m_row[x]:
                d_row[x] = w_row[x]


@njit(nogil=True)
def _nb_initialize_activity_rows(activity, depth, votes, reciprocal, start, stop):
    width = activity.shape[1]
    if votes is not None:
        r64 = np.int64(reciprocal)
        for y in range(start, stop):
            a_row = activity[y]
            d_row = depth[y]
            v_row = votes[y]
            for x in range(width):
                d_row[x] = 0
                # 65535 / 2**32 < 1 / count for every supported stack size.
                v_row[x] = (np.int64(a_row[x]) * r64) >> 32
    else:
        for y in range(start, stop):
            d_row = depth[y]
            for x in range(width):
                d_row[x] = 0


@njit(nogil=True)
def _nb_select_activity_rows(activity, best, depth, votes, code, reciprocal, start, stop):
    width = activity.shape[1]
    if votes is not None:
        r64 = np.int64(reciprocal)
        for y in range(start, stop):
            a_row = activity[y]
            b_row = best[y]
            d_row = depth[y]
            v_row = votes[y]
            for x in range(width):
                value = a_row[x]
                v_row[x] += (np.int64(value) * r64) >> 32
                if value > b_row[x]:
                    b_row[x] = value
                    d_row[x] = code
    else:
        for y in range(start, stop):
            a_row = activity[y]
            b_row = best[y]
            d_row = depth[y]
            for x in range(width):
                value = a_row[x]
                if value > b_row[x]:
                    b_row[x] = value
                    d_row[x] = code


@njit(nogil=True)
def _nb_confidence_hist_rows(best, votes, histogram, start, stop):
    width = best.shape[1]
    for y in range(start, stop):
        b_row = best[y]
        v_row = votes[y]
        for x in range(width):
            confidence = np.int64(b_row[x]) - np.int64(v_row[x])
            if confidence < 0:
                confidence = -confidence
            b_row[x] = confidence
            histogram[confidence] += 1


@njit(nogil=True)
def _nb_guide_rows(best, lut, start, stop):
    ratio = np.float32(4095.0 / 65535.0)
    one = np.float32(1)
    width = best.shape[1]
    for y in range(start, stop):
        b_row = best[y]
        for x in range(width):
            position = np.float32(np.float32(b_row[x]) * ratio)
            index = min(4094, max(0, np.int32(position)))
            fraction = np.float32(position - np.float32(index))
            left = np.float32(lut[index] * np.float32(one - fraction))
            right = np.float32(lut[index + 1] * fraction)
            value = np.float32(left + right)
            b_row[x] = min(65535, max(0, np.int64(value)))


@njit(nogil=True)
def _nb_snap_rows(depth, lut, step, start, stop):
    if start >= stop or depth.size == 0:
        return
    half = step >> 1
    width = depth.shape[1]
    if 2 <= step <= 65535 and half >= 1 and isinstance(depth.flat[0], np.uint16):
        step_recip = np.uint64(((1 << 48) + step - 1) // step)
        half_recip = np.uint64(((1 << 48) + half - 1) // half)
        step64 = np.int64(step)
        half64 = np.int64(half)
        for y in range(start, stop):
            d_row = depth[y]
            for x in range(width):
                block = np.int64(d_row[x])
                total = half64 + block
                snapped = np.int64((np.uint64(total) * step_recip) >> np.uint64(48)) * step64
                distance = block - snapped
                if distance >= 0:
                    idx = min(1024, np.int64((np.uint64(distance << 10) * half_recip) >> np.uint64(48)))
                    corr = (half64 * lut[idx]) >> 10
                    d_row[x] = (snapped + corr) & 65535
                else:
                    idx = min(1024, np.int64((np.uint64((-distance) << 10) * half_recip) >> np.uint64(48)))
                    corr = (half64 * lut[idx]) >> 10
                    d_row[x] = (snapped - corr) & 65535
        return

    for y in range(start, stop):
        d_row = depth[y]
        for x in range(width):
            block = np.int64(d_row[x])
            total = half + block
            snapped = total - total % step
            distance = block - snapped
            signed = ((abs(distance) << 10) // half) * (-1 if distance < 0 else 1)
            index = min(1024, abs(signed))
            correction = half * lut[index] * (1 if signed > 0 else -1)
            correction = ((correction >> 10) if correction >= 0
                          else -((-correction) >> 10))
            d_row[x] = (snapped + correction) & 65535


@njit(nogil=True)
def _nb_merge_active_rows(accumulator, image, depth, index, step, scale, active, tile_width):
    """Merge each contiguous run of active tiles in a row band."""
    col = 0
    while col < active.size:
        if not active[col]:
            col += 1
            continue
        left = col * tile_width
        while col < active.size and active[col]:
            col += 1
        right = min(col * tile_width, depth.shape[1])
        _nb_merge_rows(accumulator[:, left:right], image[:, left:right], depth[:, left:right],
                       index, step, 0, depth.shape[0], scale, False, False)


def merge_active_rows(accumulator, image, depth, index, step, scale, active, tile_width):
    """Merge each contiguous run of active tiles in a row band."""
    from core._native import load_native
    native = load_native()
    if native is not None and hasattr(native, 'of_merge_rows'):
        col = 0
        active_arr = np.asarray(active)
        total_active = active_arr.size
        width = depth.shape[1]
        height = depth.shape[0]
        while col < total_active:
            if not active_arr[col]:
                col += 1
                continue
            left = col * tile_width
            while col < total_active and active_arr[col]:
                col += 1
            right = min(col * tile_width, width)
            merge_rows(accumulator[:, left:right], image[:, left:right], depth[:, left:right],
                       index, step, 0, height, scale, False, False)
        return
    _nb_merge_active_rows(accumulator, image, depth, index, step, scale, active, tile_width)


def _as_ndarray(arr):
    return None if arr is None else (arr if type(arr) is np.ndarray else arr.view(np.ndarray))


def _u16_rows_2d(*arrays):
    return all(arr.dtype == np.uint16 and arr.ndim == 2 and arr.strides[1] == 2 for arr in arrays)


def activity_rows(gray, fine, output, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_activity_rows')
            and _u16_rows_2d(gray, fine, output)):
        native.of_activity_rows(_as_ndarray(gray), _as_ndarray(fine),
                                _as_ndarray(output), int(start), int(stop))
        return
    _nb_activity_rows(gray, fine, output, start, stop)


def detail_rows(gray, output, row_offset):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_detail_rows')
            and _u16_rows_2d(gray, output)):
        native.of_detail_rows(_as_ndarray(gray), _as_ndarray(output), int(row_offset))
        return
    _nb_detail_rows(gray, output, row_offset)


def smooth_detail_rows(detail, output, start, stop, columns=None):
    # The native kernel keeps its own column sums; ``columns`` is Numba scratch.
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_smooth_detail_rows')
            and detail.dtype in (np.uint16, np.int32)
            and output.dtype in (np.uint16, np.int32)
            and detail.ndim == 2 and output.ndim == 2
            and detail.strides[1] == detail.itemsize
            and output.strides[1] == output.itemsize):
        native.of_smooth_detail_rows(_as_ndarray(detail), _as_ndarray(output),
                                     int(start), int(stop))
        return
    _nb_smooth_detail_rows(detail, output, start, stop, columns)


def initialize_detail_rows(second, winner, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_initialize_detail_rows')
            and _u16_rows_2d(second, winner)):
        native.of_initialize_detail_rows(_as_ndarray(second), _as_ndarray(winner),
                                         int(start), int(stop))
        return
    _nb_initialize_detail_rows(second, winner, start, stop)


def select_detail_rows(activity, best, second, winner, code, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_select_detail_rows')
            and _u16_rows_2d(activity, best, second, winner)):
        native.of_select_detail_rows(_as_ndarray(activity), _as_ndarray(best),
                                     _as_ndarray(second), _as_ndarray(winner),
                                     int(code), int(start), int(stop))
        return
    _nb_select_detail_rows(activity, best, second, winner, code, start, stop)


def detail_confidence_rows(coarse, best, second, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_detail_confidence_rows')
            and _u16_rows_2d(coarse, best, second)):
        native.of_detail_confidence_rows(_as_ndarray(coarse), _as_ndarray(best),
                                         _as_ndarray(second), int(start), int(stop))
        return
    _nb_detail_confidence_rows(coarse, best, second, start, stop)


def apply_detail_rows(depth, mask, winner, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_apply_detail_rows')
            and _u16_rows_2d(depth, mask, winner)):
        native.of_apply_detail_rows(_as_ndarray(depth), _as_ndarray(mask),
                                    _as_ndarray(winner), int(start), int(stop))
        return
    _nb_apply_detail_rows(depth, mask, winner, start, stop)


def initialize_activity_rows(activity, depth, votes, reciprocal, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_initialize_activity_rows')
            and _u16_rows_2d(activity, depth)
            and (votes is None or _u16_rows_2d(votes))):
        native.of_initialize_activity_rows(_as_ndarray(activity), _as_ndarray(depth),
                                           _as_ndarray(votes), int(reciprocal),
                                           int(start), int(stop))
        return
    _nb_initialize_activity_rows(activity, depth, votes, reciprocal, start, stop)


def select_activity_rows(activity, best, depth, votes, code, reciprocal, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_select_activity_rows')
            and _u16_rows_2d(activity, best, depth)
            and (votes is None or _u16_rows_2d(votes))):
        native.of_select_activity_rows(_as_ndarray(activity), _as_ndarray(best),
                                       _as_ndarray(depth), _as_ndarray(votes),
                                       int(code), int(reciprocal), int(start), int(stop))
        return
    _nb_select_activity_rows(activity, best, depth, votes, code, reciprocal, start, stop)


def confidence_hist_rows(best, votes, histogram, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_confidence_hist_rows')
            and _u16_rows_2d(best, votes)
            and histogram.dtype == np.int64 and histogram.ndim == 1
            and histogram.strides[0] == 8):
        native.of_confidence_hist_rows(_as_ndarray(best), _as_ndarray(votes),
                                       _as_ndarray(histogram), int(start), int(stop))
        return
    _nb_confidence_hist_rows(best, votes, histogram, start, stop)


def guide_rows(best, lut, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_guide_rows')
            and _u16_rows_2d(best)):
        lut_f32 = lut if (lut.dtype == np.float32 and lut.strides[0] == 4) else np.ascontiguousarray(lut, dtype=np.float32)
        native.of_guide_rows(_as_ndarray(best), _as_ndarray(lut_f32), int(start), int(stop))
        return
    _nb_guide_rows(best, lut, start, stop)


def snap_rows(depth, lut, step, start, stop):
    from core._native import load_native
    native = load_native()
    if (native is not None and hasattr(native, 'of_snap_rows')
            and _u16_rows_2d(depth)):
        lut_i32 = lut if (lut.dtype == np.int32 and lut.strides[0] == 4) else np.ascontiguousarray(lut, dtype=np.int32)
        native.of_snap_rows(_as_ndarray(depth), _as_ndarray(lut_i32), int(step), int(start), int(stop))
        return
    _nb_snap_rows(depth, lut, step, start, stop)
