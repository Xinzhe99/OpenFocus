"""OpenFocus A preserves Burt interpolation at odd edges."""

import numpy as np
import pytest


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
