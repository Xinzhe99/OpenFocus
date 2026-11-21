"""Streaming weighted-pyramid fusion used by OpenFocus C.

The pyramid stores normalized detail and reconstructs it with a synthesis filter.
"""
from math import prod

import numpy as np
from fusion_methods._depth_kernel import njit

from core.frame_storage import create_transient_frame as create_frame
from fusion_methods._openfocus_execution import run_bands

_EVEN = np.array([0.019027505, 0.318410337, 0.559359729, 0.103202470, 0], dtype=np.float32)
_ODD = _EVEN[::-1].copy()
_DOWN = np.array([0.125, 0.375, 0.375, 0.125], dtype=np.float32)
_DOWN_BAND = 32
_UP_BAND = 64


@njit(nogil=True)
def _normalize_rows(source, output, factor, start, stop):
    for y in range(start, stop):
        for x in range(output.shape[1]):
            for c in range(output.shape[2]):
                output[y, x, c] = np.float32(source[y, x, c]) * factor


@njit(inline='always')
def _down_horizontal(source, output, start, stop):
    _, width, channels = source.shape
    for y in range(start, stop):
        for x in range(output.shape[1]):
            for c in range(channels):
                value = np.float32(0)
                for k in range(4):
                    sx = min(width - 1, max(0, 2 * x + k - 1))
                    value = np.float32(value + np.float32(source[y, sx, c] * _DOWN[k]))
                output[y - start, x, c] = value


@njit(nogil=True)
def _down_rows(source, output, horizontal, start, stop):
    height = source.shape[0]
    _, width, channels = output.shape
    first = max(0, 2 * start - 1)
    _down_horizontal(source, horizontal, first, min(height, 2 * stop + 1))
    for y in range(start, stop):
        for x in range(width):
            for c in range(channels):
                value = np.float32(0)
                for k in range(4):
                    sy = min(height - 1, max(0, 2 * y + k - 1))
                    value = np.float32(value + np.float32(horizontal[sy - first, x, c] * _DOWN[k]))
                output[y, x, c] = value


@njit(inline='always')
def _up_horizontal(source, output, start, stop):
    _, width, channels = source.shape
    # Interleave horizontal phases so vertical synthesis writes in pixel order.
    for y in range(start, stop):
        for x in range(output.shape[1]):
            weights = _EVEN if x % 2 == 0 else _ODD
            sx = x // 2 - 2
            first, last = max(0, -sx), min(5, width - sx)
            total = np.float32(0)
            for k in range(first, last):
                total = np.float32(total + weights[k])
            for c in range(channels):
                value = np.float32(0)
                for k in range(first, last):
                    value = np.float32(value + np.float32(source[y, sx + k, c] * weights[k]))
                output[y - start, x, c] = np.float32(value / total)


@njit(inline='always')
def _select_pixel(detail, accumulator, best, winner, index, y, x, weight):
    if index == 0 or weight > best[y, x]:
        best[y, x] = weight
        if winner is not None:
            winner[y, x] = index
        for c in range(detail.shape[2]):
            accumulator[y, x, c] = detail[y, x, c]


@njit(nogil=True)
def _synthesize_rows(source, fine, horizontal, start, stop, weights, accumulator, best,
                     winner, index):
    height, width, channels = fine.shape
    source_height = source.shape[0]
    origin = max(0, start // 2 - 2)
    _up_horizontal(source, horizontal, origin, min(source_height, (stop - 1) // 2 + 3))
    # Collapse overwrites selected detail. Selection overwrites the fine Gaussian
    # and either keeps weights for smoothing or consumes them in this band.
    reconstruct = weights is None and best is None
    for y in range(start, stop):
        sy = y // 2 - 2
        phase = _EVEN if y % 2 == 0 else _ODD
        first, last = max(0, -sy), min(5, source_height - sy)
        total = np.float32(0)
        for k in range(first, last):
            total = np.float32(total + phase[k])
        for x in range(width):
            strength = np.float32(0)
            for c in range(channels):
                value = np.float32(0)
                for k in range(first, last):
                    value = np.float32(value + np.float32(horizontal[sy + k - origin, x, c] * phase[k]))
                lower = np.float32(value / total)
                if reconstruct:
                    value = np.float32(np.float32(np.float32(lower + np.float32(0.1))
                                                   * fine[y, x, c]) + lower)
                    fine[y, x, c] = min(np.float32(1), max(np.float32(0), value))
                else:
                    value = np.float32(np.float32(fine[y, x, c] - lower)
                                       / np.float32(lower + np.float32(0.1)))
                    fine[y, x, c] = value
                    if c < 3:
                        strength = np.float32(strength + abs(value))
            if not reconstruct:
                weight = 0
                if not ((height > 2 and (y == 0 or y == height - 1))
                        or (width > 2 and (x == 0 or x == width - 1))):
                    if channels == 1:
                        strength = np.float32(strength * np.float32(3))
                    elif channels == 2:
                        strength = np.float32(strength * np.float32(1.5))
                    weight = min(65535, int(np.float32(strength * np.float32(21845))))
                if weights is not None:
                    weights[y, x] = weight
                elif best is not None:
                    _select_pixel(fine, accumulator, best, winner, index, y, x, np.uint16(weight))


@njit(nogil=True)
def _merge_rows(detail, weights, accumulator, best, winner, index, start, stop):
    for y in range(start, stop):
        for x in range(weights.shape[1]):
            _select_pixel(detail, accumulator, best, winner, index, y, x, weights[y, x])


@njit(nogil=True)
def _output_rows(source, output, is_u8, start, stop):
    for y in range(start, stop):
        for x in range(source.shape[1]):
            for c in range(source.shape[2]):
                value = min(np.float32(1), max(np.float32(0), source[y, x, c]))
                code = np.float32(value * np.float32(65535))
                if is_u8:
                    output[y, x, c] = int(code) >> 8
                else:
                    output[y, x, c] = np.float32(np.rint(code)) * np.float32(1.0 / 257.0)


def _rows(length, function, cancel):
    def run(start, stop):
        cancel()
        function(start, stop)
    run_bands(length, 128, run)


def _down(source, cancel):
    height, width, channels = source.shape
    output = create_frame(((height + 1) // 2, (width + 1) // 2, channels), np.float32)
    scratch_shape = (min(height, 2 * _DOWN_BAND + 2), output.shape[1], channels)

    def run(start, stop, horizontal):
        cancel()
        _down_rows(source, output, horizontal, start, stop)

    run_bands(output.shape[0], _DOWN_BAND, run, scratch_bytes=4 * prod(scratch_shape),
              workspace=lambda: create_frame(scratch_shape, np.float32))
    return output


def _synthesize(coarse, fine, cancel, *, weights=None, accumulator=None, best=None,
                winner=None, index=0):
    # Each fine band needs half as many coarse rows plus the five-tap halo.
    scratch_shape = (min(coarse.shape[0], _UP_BAND // 2 + 4), *fine.shape[1:])

    def run(start, stop, horizontal):
        cancel()
        _synthesize_rows(coarse, fine, horizontal, start, stop, weights, accumulator, best,
                         winner, index)

    run_bands(fine.shape[0], _UP_BAND, run, scratch_bytes=4 * prod(scratch_shape),
              workspace=lambda: create_frame(scratch_shape, np.float32))


def fuse_pyramid(frames, shape, img_resize, smoothing, should_cancel, progress_callback=None,
                 focus_callback=None):
    """Fuse ordered sources in mapped, cancellable pyramid passes."""
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods.openfocus_bc import _source_codes
    from fusion_methods.openfocus_depth import _filter

    def cancel():
        _check_cancel(should_cancel)

    is_u8 = all(frame.dtype == np.uint8 for frame in frames)
    is_gray = len(shape) == 2
    channels = 1 if is_gray else shape[2]
    levels = [shape[:2]]
    while True:
        next_level = ((levels[-1][0] + 1) // 2, (levels[-1][1] + 1) // 2)
        # Even small or narrow images need a detail level for focus selection.
        if min(next_level) < 16 and (len(levels) > 1 or next_level == levels[-1]):
            break
        levels.append(next_level)
    accumulators = [create_frame((h, w, channels), np.float32) for h, w in levels[:-1]]
    maxima = [create_frame((h, w), np.uint16) for h, w in levels[:-1]]
    winner = create_frame(shape[:2], np.uint16) if focus_callback is not None else None
    if winner is not None and not accumulators:
        winner.fill(0)
    total = len(frames) + 1
    if progress_callback is not None:
        progress_callback(0, total)
    parameter = np.float32(max(0.0, (float(smoothing) - 1.0) / 5.0))

    for index, frame in enumerate(frames):
        cancel()
        pixels, scale = _source_codes(frame, img_resize, should_cancel)
        if pixels.shape != shape:
            raise ValueError('OpenFocus C requires frames with matching shapes')
        source = pixels[:, :, None] if is_gray else pixels
        factor = np.float32(scale / (65280.0 if is_u8 else 65535.0))
        fine = create_frame(source.shape, np.float32)
        _rows(shape[0], lambda a, b: _normalize_rows(source, fine, factor, a, b), cancel)
        del source, pixels
        for level in range(len(accumulators)):
            cancel()
            coarse = _down(fine, cancel)
            height, width = fine.shape[:2]
            level_winner = winner if level == 0 else None
            if parameter > 0:
                weights = create_frame((height, width), np.uint16)
                _synthesize(coarse, fine, cancel, weights=weights)
                radius = np.float32(np.float32(max(height, width)) * parameter / np.float32(100))
                weights = _filter(weights, radius, cancel)
                _rows(height, lambda a, b: _merge_rows(fine, weights, accumulators[level],
                                                       maxima[level], level_winner, index, a, b), cancel)
                del weights
            else:
                _synthesize(coarse, fine, cancel, accumulator=accumulators[level],
                            best=maxima[level], winner=level_winner, index=index)
            fine = coarse
            del coarse
        # The coarsest weights depend only on geometry, so every source ties.
        if index == 0:
            result = fine
        del fine
        if progress_callback is not None:
            progress_callback(index + 1, total)
    del maxima
    while accumulators:
        cancel()
        fine = accumulators.pop()
        _synthesize(result, fine, cancel)
        result = fine
        del fine
    output_shape = (shape[0], shape[1], channels)
    output = create_frame(output_shape, np.uint8 if is_u8 else np.float32)
    _rows(shape[0], lambda a, b: _output_rows(result, output, is_u8, a, b), cancel)
    if progress_callback is not None:
        progress_callback(total, total)
    cancel()
    if focus_callback is not None:
        # Each pyramid level makes its own selection. Export the actual
        # finest-level detail winner, not a claim about the collapsed color.
        focus_callback(winner, len(frames) - 1)
    return output[:, :, 0] if is_gray else output
