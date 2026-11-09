"""Regression coverage for resampling, depth estimation, and fusion."""
from pathlib import Path

import numpy as np
import pytest

from fusion_methods._openfocus_resample import half_gray


def test_b_feather_preserves_more_depth_at_focus_transitions():
    from fusion_methods.openfocus_depth import depth_map

    fixture = Path(__file__).parent / 'fixtures' / 'openfocus_native_s1_zones_depth.npz'
    with np.load(fixture, allow_pickle=False) as data:
        frames = list(data['frames'])
    kwargs = dict(img_resize=None, kernel_size=45, smoothing=5.0,
                  should_cancel=None, thread_count=1, return_codes=True)
    f0 = depth_map(frames, feather=0, **kwargs).astype(np.int64)
    f3 = depth_map(frames, feather=3, **kwargs).astype(np.int64)
    step = 65535 // (len(frames) - 1)
    nearest0 = ((f0 + (step >> 1)) // step) * step
    nearest3 = ((f3 + (step >> 1)) // step) * step
    distance0 = np.abs(f0 - nearest0)
    distance3 = np.abs(f3 - nearest3)
    assert np.any(distance3 > distance0)
    assert np.all(distance3 >= distance0)


def test_b_feather_keeps_identical_frames_at_constant_depth():
    from fusion_methods.openfocus_depth import depth_map

    frame = np.full((24, 24), 127, dtype=np.uint8)
    kwargs = dict(img_resize=None, kernel_size=17, smoothing=5.0,
                  should_cancel=None, thread_count=1, return_codes=True)
    f0 = depth_map([frame, frame], feather=0, **kwargs)
    f3 = depth_map([frame, frame], feather=3, **kwargs)
    np.testing.assert_array_equal(f0, f3)
    assert np.all(f0 == f0[0, 0])


@pytest.mark.parametrize('feather', [-1, 11, True, np.bool_(False), 3.0, 3.5, '3'])
def test_b_feather_rejects_invalid_values_before_processing(feather):
    from fusion_methods.openfocus_bc import openfocus_b_impl
    from fusion_methods.openfocus_depth import depth_map

    with pytest.raises(ValueError, match='feather'):
        depth_map([], None, 17, 5.0, None, feather=feather)
    with pytest.raises(ValueError, match='feather'):
        openfocus_b_impl([], depth_override=np.zeros((1, 1)), feather=feather)


_FIXTURE = Path(__file__).parent / 'fixtures' / 'openfocus_native_gray_half_strips.npz'


@pytest.mark.parametrize('call', [1, 3])
@pytest.mark.parametrize('edge, result_rows', [('top', slice(0, 6)),
                                              ('bottom', slice(2, 8))])
def test_half_gray_matches_expected_strips(call, edge, result_rows):
    with np.load(_FIXTURE, allow_pickle=False) as fixture:
        source = fixture[f'call{call}_src_{edge}']
        expected = fixture[f'call{call}_dst_{edge}']

    assert source.shape == (16, 512)
    assert source.dtype == np.uint16
    assert expected.shape == (6, 256)
    assert expected.dtype == np.uint16

    result = half_gray(source)

    assert result.shape == (8, 256)
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result[result_rows, :], expected)


def test_method_b_depth_matches_expected_codes():
    from core.algorithm_adapters import depth_openfocus_b
    from core.algorithms import resolve_fusion_parameters

    fixture = Path(__file__).parent / 'fixtures' / 'openfocus_native_s1_zones_depth.npz'
    with np.load(fixture, allow_pickle=False) as data:
        frames = data['frames']
        expected = data['depth']
    parameters = resolve_fusion_parameters('openfocus_b',
                                           {'radius': 22, 'smoothing': 5, 'feather': 0})
    depth = depth_openfocus_b(list(frames), **parameters, thread_count=1)
    step = 65535 // (len(frames) - 1)
    np.testing.assert_array_equal(np.rint(depth * step).astype(np.uint16), expected)


def test_method_b_fusion_matches_expected_output():
    from fusion_methods.openfocus_bc import openfocus_b_impl

    fixtures = Path(__file__).parent / 'fixtures'
    with np.load(fixtures / 'openfocus_native_s1_zones_depth.npz', allow_pickle=False) as data:
        frames = data['frames']
    with np.load(fixtures / 'openfocus_native_s1_method_b.npz', allow_pickle=False) as data:
        expected = data['result']
    result = openfocus_b_impl(list(frames), kernel_size=45, depth_smoothing=1.2,
                              thread_count=1)
    difference = np.abs(result.astype(np.int16) - expected.astype(np.int16))
    # Allow small pixel differences in the merged result.
    assert float(difference.mean()) < 0.39
    assert int(difference.max()) <= 16



def test_method_c_depth_matches_expected_codes():
    from fusion_methods.openfocus_depth import depth_map

    fixture = Path(__file__).parent / 'fixtures' / 'openfocus_native_s1_zones_depth.npz'
    with np.load(fixture, allow_pickle=False) as data:
        frames = data['frames']
        expected = data['depth']

    assert frames.shape == (4, 512, 512)
    assert frames.dtype == np.uint8
    assert expected.shape == (512, 512)
    assert expected.dtype == np.uint16

    result = depth_map(list(frames), None, 45, 5, None, thread_count=1,
                       return_codes=True, half_resolution=True)

    assert result.shape == expected.shape
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(('dtype', 'key'), [(np.uint8, 'rgb8'), (np.uint16, 'rgb16')])
def test_color_depth_matches_expected_codes(dtype, key):
    from fusion_methods.openfocus_depth import depth_map

    rng = np.random.default_rng(9001)
    frames = [rng.integers(0, np.iinfo(dtype).max + 1,
                           size=(512, 512, 3), dtype=dtype) for _ in range(4)]
    fixture = Path(__file__).parent / 'fixtures' / 'openfocus_native_color_depth.npz'
    with np.load(fixture, allow_pickle=False) as data:
        expected = data[key]
    result = depth_map(frames, None, 17, 6, None, thread_count=1,
                       return_codes=True, half_resolution=True)
    np.testing.assert_array_equal(result, expected)




def test_depth_expansion_matches_expected_odd_canvas():
    from fusion_methods._openfocus_resample import expand_depth

    fixture = Path(__file__).parent / 'fixtures' / 'openfocus_native_odd_depth_expand.npz'
    with np.load(fixture, allow_pickle=False) as data:
        source = data['source']
        expected = data['result']

    assert source.shape == (64, 65)
    assert source.dtype == np.uint16
    assert expected.shape == (129, 131)
    assert expected.dtype == np.uint16

    result = expand_depth(source, expected.shape)

    assert result.shape == expected.shape
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, expected)
