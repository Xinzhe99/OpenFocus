"""Streaming weighted-pyramid fusion used by OpenFocus C.

The pyramid stores normalized detail and reconstructs it with a synthesis filter.
"""
import numpy as np
from fusion_methods._depth_kernel import njit

from core.frame_storage import create_transient_frame as create_frame
from fusion_methods._openfocus_execution import run_bands

_EVEN = np.array([0.019027505, 0.318410337, 0.559359729, 0.103202470, 0], dtype=np.float32)
_ODD = _EVEN[::-1].copy()
_DOWN = np.array([0.125, 0.375, 0.375, 0.125], dtype=np.float32)


@njit(nogil=True)
def _normalize_rows(source, output, factor, start, stop):
    for y in range(start, stop):
        for x in range(output.shape[1]):
            for c in range(output.shape[2]):
                output[y, x, c] = np.float32(source[y, x, c]) * factor


@njit(nogil=True)
def _down_horizontal(source, output, start, stop):
    height, width, channels = source.shape
    for y in range(start, stop):
        for x in range(output.shape[1]):
            for c in range(channels):
                value = np.float32(0)
                for k in range(4):
                    sx = min(width - 1, max(0, 2 * x + k - 1))
                    value = np.float32(value + np.float32(source[y, sx, c] * _DOWN[k]))
                output[y, x, c] = value


@njit(nogil=True)
def _down_vertical(source, output, start, stop):
    height, width, channels = source.shape
    for y in range(start, stop):
        for x in range(width):
            for c in range(channels):
                value = np.float32(0)
                for k in range(4):
                    sy = min(height - 1, max(0, 2 * y + k - 1))
                    value = np.float32(value + np.float32(source[sy, x, c] * _DOWN[k]))
                output[y, x, c] = value


@njit(nogil=True)
def _up_horizontal(source, output, weights, start, stop):
    height, width, channels = source.shape
    for y in range(start, stop):
        for x in range(width):
            total = np.float32(0)
            for k in range(5):
                sx = x + k - 2
                if 0 <= sx < width:
                    total = np.float32(total + weights[k])
            for c in range(channels):
                value = np.float32(0)
                for k in range(5):
                    sx = x + k - 2
                    if 0 <= sx < width:
                        value = np.float32(value + np.float32(source[y, sx, c] * weights[k]))
                output[y, x, c] = np.float32(value / total)


@njit(nogil=True)
def _up_vertical(source, output, parity, start, stop):
    height, width, channels = source.shape
    for y in range(start, stop):
        sy0 = y // 2
        weights = _EVEN if y % 2 == 0 else _ODD
        total = np.float32(0)
        for k in range(5):
            sy = sy0 + k - 2
            if 0 <= sy < height:
                total = np.float32(total + weights[k])
        for x in range(width):
            ox = 2 * x + parity
            if ox >= output.shape[1]:
                continue
            for c in range(channels):
                value = np.float32(0)
                for k in range(5):
                    sy = sy0 + k - 2
                    if 0 <= sy < height:
                        value = np.float32(value + np.float32(source[sy, x, c] * weights[k]))
                output[y, ox, c] = np.float32(value / total)


@njit(nogil=True)
def _detail_and_weight(fine, coarse, detail, weight, start, stop):
    height, width, channels = fine.shape
    for y in range(start, stop):
        for x in range(width):
            strength = np.float32(0)
            for c in range(channels):
                lower = coarse[y, x, c]
                value = np.float32(np.float32(fine[y, x, c] - lower)
                                   / np.float32(lower + np.float32(0.1)))
                detail[y, x, c] = value
                if c < 3:
                    strength = np.float32(strength + abs(value))
            if channels == 1:
                strength = np.float32(strength * np.float32(3))
            elif channels == 2:
                strength = np.float32(strength * np.float32(1.5))
            if ((height > 2 and (y == 0 or y == height - 1))
                    or (width > 2 and (x == 0 or x == width - 1))):
                weight[y, x] = 0
            else:
                weight[y, x] = min(65535, int(np.float32(strength * np.float32(21845))))


@njit(nogil=True)
def _last_weight(weight, start, stop):
    height, width = weight.shape
    for y in range(start, stop):
        for x in range(width):
            weight[y, x] = 0 if y == 0 or x == 0 or y == height - 1 or x == width - 1 else 65535


@njit(nogil=True)
def _merge_rows(detail, weight, accumulator, best, first, start, stop):
    for y in range(start, stop):
        for x in range(weight.shape[1]):
            w = weight[y, x]
            if first or w > best[y, x]:
                best[y, x] = w
                for c in range(detail.shape[2]):
                    accumulator[y, x, c] = detail[y, x, c]


@njit(nogil=True)
def _collapse_rows(coarse, detail, output, start, stop):
    for y in range(start, stop):
        for x in range(output.shape[1]):
            for c in range(output.shape[2]):
                v = coarse[y, x, c]
                value = np.float32(np.float32(np.float32(v + np.float32(0.1))
                                               * detail[y, x, c]) + v)
                output[y, x, c] = min(np.float32(1), max(np.float32(0), value))


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
    horizontal = create_frame((height, (width + 1) // 2, channels), np.float32)
    _rows(height, lambda a, b: _down_horizontal(source, horizontal, a, b), cancel)
    output = create_frame(((height + 1) // 2, horizontal.shape[1], channels), np.float32)
    _rows(output.shape[0], lambda a, b: _down_vertical(horizontal, output, a, b), cancel)
    return output


def _up(source, shape, cancel):
    output = create_frame((shape[0], shape[1], source.shape[2]), np.float32)
    for parity, weights in ((0, _EVEN), (1, _ODD)):
        horizontal = create_frame(source.shape, np.float32)
        _rows(source.shape[0], lambda a, b: _up_horizontal(source, horizontal, weights, a, b), cancel)
        _rows(output.shape[0], lambda a, b: _up_vertical(horizontal, output, parity, a, b), cancel)
    return output


def fuse_pyramid(frames, shape, img_resize, smoothing, should_cancel, progress_callback=None):
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
    accumulators = [create_frame((h, w, channels), np.float32) for h, w in levels]
    maxima = [create_frame((h, w), np.uint16) for h, w in levels]
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
        base = create_frame(source.shape, np.float32)
        _rows(shape[0], lambda a, b: _normalize_rows(source, base, factor, a, b), cancel)
        del source, pixels
        gaussian = [base]
        del base
        for _ in range(1, len(levels)):
            gaussian.append(_down(gaussian[-1], cancel))
        for level, (height, width) in enumerate(levels):
            cancel()
            weights = create_frame((height, width), np.uint16)
            if level + 1 == len(levels):
                detail = gaussian[level]
                _rows(height, lambda a, b: _last_weight(weights, a, b), cancel)
            else:
                up = _up(gaussian[level + 1], (height, width), cancel)
                detail = create_frame((height, width, channels), np.float32)
                _rows(height, lambda a, b: _detail_and_weight(gaussian[level], up, detail, weights, a, b), cancel)
                del up
            if parameter > 0:
                radius = np.float32(np.float32(max(height, width)) * parameter / np.float32(100))
                weights = _filter(weights, radius, cancel)
            _rows(height, lambda a, b: _merge_rows(detail, weights, accumulators[level], maxima[level], index == 0, a, b), cancel)
            gaussian[level] = None
            del detail, weights
        if progress_callback is not None:
            progress_callback(index + 1, total)
    del maxima
    result = accumulators[-1]
    for level in range(len(levels) - 2, -1, -1):
        cancel()
        coarse = _up(result, levels[level], cancel)
        result = create_frame(coarse.shape, np.float32)
        _rows(result.shape[0], lambda a, b: _collapse_rows(coarse, accumulators[level], result, a, b), cancel)
        accumulators[level + 1] = None
        del coarse
    output_shape = (shape[0], shape[1], channels)
    output = create_frame(output_shape, np.uint8 if is_u8 else np.float32)
    _rows(shape[0], lambda a, b: _output_rows(result, output, is_u8, a, b), cancel)
    if progress_callback is not None:
        progress_callback(total, total)
    cancel()
    return output[:, :, 0] if is_gray else output
