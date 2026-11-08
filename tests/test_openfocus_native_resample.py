"""Exact OpenFocus Method C resampling and depth regressions."""
from pathlib import Path

import numpy as np
import pytest

from fusion_methods._openfocus_resample import half_gray


_FIXTURE = Path(__file__).parent / 'fixtures' / 'openfocus_native_gray_half_strips.npz'


@pytest.mark.parametrize('call', [1, 3])
@pytest.mark.parametrize('edge, result_rows', [('top', slice(0, 6)),
                                              ('bottom', slice(2, 8))])
def test_half_gray_matches_native_capture_strips(call, edge, result_rows):
    with np.load(_FIXTURE, allow_pickle=False) as captured:
        source = captured[f'call{call}_src_{edge}']
        expected = captured[f'call{call}_dst_{edge}']

    assert source.shape == (16, 512)
    assert source.dtype == np.uint16
    assert expected.shape == (6, 256)
    assert expected.dtype == np.uint16

    result = half_gray(source)

    assert result.shape == (8, 256)
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result[result_rows, :], expected)


def test_method_c_depth_matches_official_s1_zones_sidecar():
    from fusion_methods.openfocus_depth import depth_map

    fixture = Path(__file__).parent / 'fixtures' / 'openfocus_native_s1_zones_depth.npz'
    with np.load(fixture, allow_pickle=False) as captured:
        frames = captured['frames']
        expected = captured['depth']

    assert frames.shape == (4, 512, 512)
    assert frames.dtype == np.uint8
    assert expected.shape == (512, 512)
    assert expected.dtype == np.uint16

    result = depth_map(list(frames), None, 45, 5, None, thread_count=1,
                       return_codes=True, half_resolution=True)

    assert result.shape == expected.shape
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, expected)


def test_method_c_fusion_matches_official_s1_zones_result():
    from fusion_methods.openfocus_bc import openfocus_c_impl

    fixture = Path(__file__).parent / 'fixtures' / 'openfocus_native_s1_zones_depth.npz'
    with np.load(fixture, allow_pickle=False) as captured:
        frames = captured['frames']
        expected = captured['result']

    assert frames.shape == (4, 512, 512)
    assert frames.dtype == np.uint8
    assert expected.shape == (512, 512)
    assert expected.dtype == np.uint8

    result = openfocus_c_impl(list(frames), kernel_size=45,
                             blend_smoothing=3, thread_count=1)

    assert result.shape == expected.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, expected)


def test_depth_expansion_matches_native_odd_canvas():
    from fusion_methods._openfocus_resample import expand_depth

    fixture = Path(__file__).parent / 'fixtures' / 'openfocus_native_odd_depth_expand.npz'
    with np.load(fixture, allow_pickle=False) as captured:
        source = captured['source']
        expected = captured['result']

    assert source.shape == (64, 65)
    assert source.dtype == np.uint16
    assert expected.shape == (129, 131)
    assert expected.dtype == np.uint16

    result = expand_depth(source, expected.shape)

    assert result.shape == expected.shape
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, expected)
