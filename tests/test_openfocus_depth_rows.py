"""Depth row kernels equal their definitions on every edge configuration."""
import numpy as np
import pytest

from fusion_methods._openfocus_depth_rows import _reflect, smooth_detail_rows


def _reflected_box_mean(detail):
    """Integer mean of the 5x5 neighbourhood with mirrored (non-repeated) edges."""
    height, width = detail.shape
    output = np.empty_like(detail)
    for y in range(height):
        for x in range(width):
            total = 0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    total += int(detail[_reflect(y + dy, height), _reflect(x + dx, width)])
            output[y, x] = total // 25
    return output


@pytest.mark.parametrize('shape', [(1, 1), (1, 6), (6, 1), (2, 2), (2, 9), (3, 7),
                                   (7, 3), (5, 5), (6, 6), (19, 23)])
def test_smooth_detail_rows_match_the_reflected_box_mean(shape):
    rng = np.random.default_rng(sum(shape) * 7919)
    detail = rng.integers(0, 65536, shape, dtype=np.uint16)
    detail.flat[::5] = 65535
    output = np.full(shape, 7, np.uint16)
    smooth_detail_rows(detail, output, 0, shape[0])
    np.testing.assert_array_equal(output, _reflected_box_mean(detail))


def test_smooth_detail_rows_bands_are_independent():
    rng = np.random.default_rng(31)
    detail = rng.integers(0, 65536, (37, 41), dtype=np.uint16)
    expected = _reflected_box_mean(detail)
    output = np.zeros_like(detail)
    for start, stop in ((0, 3), (3, 4), (4, 20), (20, 37)):
        smooth_detail_rows(detail, output, start, stop)
    np.testing.assert_array_equal(output, expected)


def _whole_axis(source, radius):
    from fusion_methods._depth_kernel import axis_block
    from fusion_methods.openfocus_depth import _weight

    height, width = source.shape
    q = _weight(radius)
    t = (4096 - q) >> 1
    count = max(1, min(int(radius), height))
    initial = ((source[:count].sum(axis=0, dtype=np.uint64) << 4) // count) * (t + q)
    acc = initial.astype(np.uint32)
    history = np.empty(source.shape, np.uint16)
    output = np.empty_like(history)
    axis_block(source, None, history, output, acc, q, t, 0, 0, 0, height, 0, width, False)
    acc = (history[-1].astype(np.uint32) << 4) * (t + q)
    axis_block(source, None, history, output, acc, q, t, 0, 0, 0, height, 0, width, True)
    return output


@pytest.mark.parametrize('shape', [(1, 67), (129, 1), (133, 137)])
@pytest.mark.parametrize('radius', [0.25, 17.5])
def test_unguided_filters_match_whole_axes_on_strided_pixels(shape, radius):
    from fusion_methods.openfocus_depth import _filter

    rng = np.random.default_rng(4709)
    original = rng.integers(0, 65536, (shape[0], shape[1] * 2), dtype=np.uint16)
    source = original[:, ::-2]
    before = source.copy()
    other_columns = original[:, ::2].copy()
    expected = _whole_axis(_whole_axis(source.T, radius).T, radius)
    out = _filter(source, radius, lambda: None)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(source, before)

    assert _filter(source, radius, lambda: None, overwrite=True) is source
    np.testing.assert_array_equal(source, expected)
    np.testing.assert_array_equal(original[:, ::2], other_columns)
