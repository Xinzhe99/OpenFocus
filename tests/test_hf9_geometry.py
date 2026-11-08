"""HF9 A pixels preserve independent Burt interpolation at odd edges."""

import numpy as np
import pytest


def _reference_burt_response(gray, radius, should_cancel, check_cancel):
    """Separable floating interpolation is exact on the dyadic Q4 grid."""
    height, width = gray.shape
    result = np.zeros(gray.shape, np.int64)
    levels = int(np.floor(np.log2(max(1, radius)))) + 3
    for level in range(levels):
        h, w = gray.shape
        padded = np.pad(gray.astype(np.int64), ((0, h % 2), (0, w % 2)), mode='edge')
        reduced = (padded[0::2, 0::2] + padded[0::2, 1::2]
                   + padded[1::2, 0::2] + padded[1::2, 1::2]) // 4
        x = (np.arange(w) + 0.5) / 2 - 0.5
        y = (np.arange(h) + 0.5) / 2 - 0.5
        horizontal = np.stack([np.interp(x, np.arange(reduced.shape[1]), row)
                               for row in reduced])
        expanded = np.stack([np.interp(y, np.arange(reduced.shape[0]), column)
                             for column in horizontal.T], axis=1)
        laplacian = np.abs(gray.astype(np.int64) * 16 - expanded * 16).astype(np.int64)
        full = np.repeat(np.repeat(laplacian, 1 << level, axis=0), 1 << level, axis=1)
        result += full[:height, :width]
        gray = reduced
    return np.minimum(result, 65535 * 16).astype(np.int32)


@pytest.mark.parametrize('shape', [(1, 1), (1, 9), (9, 1), (5, 7), (65, 67)])
def test_a_consumer_pixels_match_independent_burt_interpolation(monkeypatch, shape):
    from fusion_methods import openfocus_a as method

    rng = np.random.default_rng(64219)
    frames = [rng.integers(0, 65536, shape + (3,), np.uint16) for _ in range(3)]
    with monkeypatch.context() as reference:
        reference.setattr(method, '_hard_response', _reference_burt_response)
        expected = method.openfocus_a_impl(frames, thread_count=1)
    actual = method.openfocus_a_impl(frames, thread_count=1)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('shape', [(1, 1), (1, 9), (9, 1), (5, 7), (65, 67)])
def test_a_float_expansion_matches_independent_bilinear_interpolation(shape):
    from fusion_methods.openfocus_a import _expand_burt
    from fusion_methods.openfocus_ab import _check_cancel

    rng = np.random.default_rng(98741)
    reduced = rng.integers(0, 65536, ((shape[0] + 1) // 2,
                                     (shape[1] + 1) // 2), np.int32)
    x = (np.arange(shape[1]) + 0.5) / 2 - 0.5
    y = (np.arange(shape[0]) + 0.5) / 2 - 0.5
    horizontal = np.stack([np.interp(x, np.arange(reduced.shape[1]), row)
                           for row in reduced])
    expected = np.stack([np.interp(y, np.arange(reduced.shape[0]), column)
                         for column in horizontal.T], axis=1)
    np.testing.assert_array_equal(_expand_burt(reduced, shape, None, _check_cancel),
                                  expected)
