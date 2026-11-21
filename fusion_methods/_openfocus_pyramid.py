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
_PYRAMID_BAND = 64


@njit(nogil=True)
def _normalize_rows(source, output, factor, start, stop):
    for y in range(start, stop):
        for x in range(output.shape[1]):
            for c in range(output.shape[2]):
                output[y, x, c] = np.float32(source[y, x, c]) * factor


@njit(nogil=True)
def _normalize_flat_rows(source, output, factor, row_size, start, stop):
    for i in range(start * row_size, stop * row_size):
        output[i] = np.float32(source[i]) * factor


@njit(nogil=True)
def _down_horizontal(source, output, origin):
    _, width, channels = source.shape
    for row in range(output.shape[0]):
        y = row + origin
        for x in range(output.shape[1]):
            for c in range(channels):
                value = np.float32(0)
                for k in range(4):
                    sx = min(width - 1, max(0, 2 * x + k - 1))
                    value = np.float32(value + np.float32(source[y, sx, c] * _DOWN[k]))
                output[row, x, c] = value


@njit(nogil=True)
def _down_vertical(source, output, height, origin, start, stop):
    _, width, channels = source.shape
    for y in range(start, stop):
        for x in range(width):
            for c in range(channels):
                value = np.float32(0)
                for k in range(4):
                    sy = min(height - 1, max(0, 2 * y + k - 1))
                    value = np.float32(value + np.float32(source[sy - origin, x, c] * _DOWN[k]))
                output[y, x, c] = value


@njit(nogil=True)
def _up_horizontal(source, output, origin):
    _, width, channels = source.shape
    for row in range(output.shape[0]):
        y = row + origin
        for x in range(output.shape[1]):
            weights = _ODD if x % 2 else _EVEN
            left = x // 2 - 2
            first = max(0, -left)
            last = min(5, width - left)
            total = np.float32(0)
            for k in range(first, last):
                total = np.float32(total + weights[k])
            for c in range(channels):
                value = np.float32(0)
                for k in range(first, last):
                    value = np.float32(value + np.float32(source[y, left + k, c] * weights[k]))
                output[row, x, c] = np.float32(value / total)


@njit(nogil=True)
def _synthesis_rows(horizontal, target, weight, coarse_height, origin, start, stop):
    height, width, channels = target.shape
    for y in range(start, stop):
        top = y // 2 - 2
        weights = _ODD if y % 2 else _EVEN
        total = np.float32(0)
        first = max(0, -top)
        last = min(5, coarse_height - top)
        for k in range(first, last):
            total = np.float32(total + weights[k])
        for x in range(width):
            strength = np.float32(0)
            for c in range(channels):
                lower = np.float32(0)
                for k in range(first, last):
                    sample = horizontal[top + k - origin, x, c]
                    lower = np.float32(lower + np.float32(sample * weights[k]))
                lower = np.float32(lower / total)
                denom = np.float32(lower + np.float32(0.1))
                if weight is None:
                    value = np.float32(np.float32(denom * target[y, x, c]) + lower)
                    target[y, x, c] = min(np.float32(1), max(np.float32(0), value))
                else:
                    value = np.float32(np.float32(target[y, x, c] - lower) / denom)
                    target[y, x, c] = value
                    if c < 3:
                        strength = np.float32(strength + abs(value))
            if weight is not None:
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
def _merge_rows(detail, weight, accumulator, best, first, start, stop,
                winner=None, index=0):
    for y in range(start, stop):
        for x in range(weight.shape[1]):
            w = weight[y, x]
            if first or w > best[y, x]:
                best[y, x] = w
                if winner is not None:
                    winner[y, x] = index
                for c in range(detail.shape[2]):
                    accumulator[y, x, c] = detail[y, x, c]


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


def _normalize(source, output, factor, cancel):
    # Flatten only packed sources. Crops and flipped arrays retain the
    # strided path without allocating a contiguous copy.
    height, width, channels = source.shape
    if source.flags.c_contiguous:
        pixels = source.reshape(-1)
        normalized = output.reshape(-1)
        _rows(height, lambda a, b: _normalize_flat_rows(
            pixels, normalized, factor, width * channels, a, b), cancel)
    else:
        _rows(height, lambda a, b: _normalize_rows(source, output, factor, a, b), cancel)


def _down(source, output, cancel):
    height, _, channels = source.shape
    rows = min(height, 2 * min(output.shape[0], _PYRAMID_BAND) + 2)
    scratch_shape = rows, output.shape[1], channels

    def down(start, stop, scratch):
        cancel()
        origin = max(0, 2 * start - 1)
        end = min(height, 2 * stop + 1)
        horizontal = scratch[:end - origin]
        _down_horizontal(source, horizontal, origin)
        _down_vertical(horizontal, output, height, origin, start, stop)

    run_bands(output.shape[0], _PYRAMID_BAND, down,
              scratch_bytes=int(np.prod(scratch_shape)) * 4,
              workspace=lambda: np.empty(scratch_shape, np.float32))


def _synthesize(coarse, target, weight, cancel):
    coarse_height = coarse.shape[0]
    height, width, channels = target.shape
    rows = min(coarse_height, (_PYRAMID_BAND + 1) // 2 + 4)
    scratch_shape = rows, width, channels

    def synthesize(start, stop, scratch):
        cancel()
        origin = max(0, start // 2 - 2)
        end = min(coarse_height, (stop - 1) // 2 + 3)
        horizontal = scratch[:end - origin]
        _up_horizontal(coarse, horizontal, origin)
        _synthesis_rows(horizontal, target, weight, coarse_height, origin, start, stop)

    run_bands(height, _PYRAMID_BAND, synthesize,
              scratch_bytes=int(np.prod(scratch_shape)) * 4,
              workspace=lambda: np.empty(scratch_shape, np.float32))


def fuse_pyramid(frames, shape, img_resize, smoothing, should_cancel, progress_callback=None,
                 focus_callback=None):
    """Fuse ordered sources in mapped, cancellable pyramid passes."""
    from core.source_frames import without_source_cache_growth
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods.openfocus_bc import _pyramid_levels, _source_codes
    from fusion_methods.openfocus_depth import _filter

    def cancel():
        _check_cancel(should_cancel)

    is_u8 = all(frame.dtype == np.uint8 for frame in frames)
    is_gray = len(shape) == 2
    channels = 1 if is_gray else shape[2]
    levels = _pyramid_levels(*shape[:2])
    detail_levels = list(zip(levels, levels[1:]))
    accumulators = [create_frame((h, w, channels), np.float32) for h, w in levels[:-1]]
    maxima = [create_frame((h, w), np.uint16) for h, w in levels[:-1]]
    coarsest = create_frame((*levels[-1], channels), np.float32)
    winner = create_frame(shape[:2], np.uint16) if focus_callback is not None else None
    if detail_levels:
        rolling = [create_frame((h * w * channels,), np.float32) for h, w in levels[:2]]
        weight_plane = create_frame((shape[0] * shape[1],), np.uint16)
        full = rolling[0].reshape((*shape[:2], channels))
    else:
        full = coarsest
    total = len(frames) + 1
    if progress_callback is not None:
        progress_callback(0, total)
    parameter = np.float32(max(0.0, (float(smoothing) - 1.0) / 5.0))

    for index, frame in enumerate(frames):
        cancel()
        with without_source_cache_growth():
            pixels, scale = _source_codes(frame, img_resize, should_cancel)
        if pixels.shape != shape:
            raise ValueError('OpenFocus C requires frames with matching shapes')
        if detail_levels or index == 0:
            source = pixels[:, :, None] if is_gray else pixels
            factor = np.float32(scale / (65280.0 if is_u8 else 65535.0))
            _normalize(source, full, factor, cancel)
            del source
        del pixels
        fine = full
        for level, ((height, width), (ch, cw)) in enumerate(detail_levels):
            cancel()
            # Coarsest weights depend only on geometry. Keep the first plane.
            coarse = (coarsest if index == 0 and level == len(detail_levels) - 1
                      else rolling[(level + 1) % 2][:ch * cw * channels].reshape(ch, cw, channels))
            _down(fine, coarse, cancel)
            weights = weight_plane[:height * width].reshape(height, width)
            _synthesize(coarse, fine, weights, cancel)
            if parameter > 0:
                radius = np.float32(np.float32(max(height, width)) * parameter / np.float32(100))
                _filter(weights, radius, cancel, overwrite=True)
            level_winner = winner if level == 0 else None
            _rows(height, lambda a, b: _merge_rows(fine, weights, accumulators[level],
                  maxima[level], index == 0, a, b, level_winner, index), cancel)
            fine = coarse
        if progress_callback is not None:
            progress_callback(index + 1, total)
    del maxima, full, fine
    if detail_levels:
        del rolling, coarse, weights, weight_plane
    elif winner is not None:
        winner.fill(0)
    result = coarsest
    del coarsest
    while accumulators:
        cancel()
        detail = accumulators.pop()
        _synthesize(result, detail, None, cancel)
        result = detail
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
