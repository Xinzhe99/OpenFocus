"""Integer-domain, source selection and identity boundaries."""
import numpy as np
import pytest

from fusion_methods.openfocus_a import openfocus_a_impl
from utils.image_utils import quantize_fuse_output


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('shape', [(1, 1), (1, 9), (9, 1), (5, 4), (4, 5), (5, 5), (65, 67)])
def test_a_identity_uses_the_source_depth_exit(dtype, shape):
    values = np.arange(np.prod(shape) * 3, dtype=np.uint32).reshape(shape + (3,))
    frame = (values * 13 % (256 if dtype == np.uint8 else 65536)).astype(dtype)
    result = openfocus_a_impl([frame, frame.copy()], thread_count=1)
    expected = (np.maximum(frame.astype(np.int16) - 1, 0).astype(np.uint8)
                if dtype == np.uint8 else frame)
    np.testing.assert_array_equal(quantize_fuse_output(result, dtype == np.uint16), expected)


def test_a_smoothing_is_inert_and_sharp_frame_pixels_are_selected():
    pattern = np.where(np.indices((65, 67)).sum(axis=0) % 2, 200, 20).astype(np.uint8)
    frames = [np.full(pattern.shape, 80, np.uint8), pattern]
    for smoothing in (0, 1, 5, 20):
        result = openfocus_a_impl(frames, kernel_size=17,
                                 blend_smoothing=smoothing, thread_count=1)
        np.testing.assert_array_equal(result, pattern - 1)




@pytest.mark.parametrize('channels', [1, 3])
def test_a_float_sources_match_equivalent_16bit_sources(channels):
    rng = np.random.default_rng(978)
    shape = (33, 35) if channels == 1 else (33, 35, channels)
    floats = [rng.uniform(0, 255, shape).astype(np.float32) for _ in range(3)]
    integers = [np.rint(frame * np.float32(257)).astype(np.uint16) for frame in floats]
    expected = openfocus_a_impl(integers, thread_count=1)
    actual = openfocus_a_impl(floats, thread_count=1)
    np.testing.assert_array_equal(actual, expected)


def test_a_mixed_stack_uses_256_for_eight_bit_pixels():
    pattern = np.where(np.indices((33, 35)).sum(axis=0) % 2, 200, 20).astype(np.uint8)
    result = openfocus_a_impl([np.full(pattern.shape, 10000, np.uint16), pattern],
                             thread_count=1)
    np.testing.assert_array_equal(quantize_fuse_output(result, True),
                                  pattern.astype(np.uint16) << 8)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_a_reduces_flat_background_noise_and_preserves_focused_texture(dtype):
    rng = np.random.default_rng(421)
    shape = (192, 192)
    scale = 1 if dtype == np.uint8 else 256
    base = 120 * scale
    frames = [np.clip(base + rng.integers(-3, 4, shape) * scale,
                      0, np.iinfo(dtype).max).astype(dtype) for _ in range(8)]
    pattern = np.where(np.indices((48, 48)).sum(axis=0) % 2,
                       210 * scale, 30 * scale).astype(dtype)
    for frame in frames:
        frame[72:120, 72:120] = base
    frames[3][72:120, 72:120] = pattern

    result = quantize_fuse_output(openfocus_a_impl(frames, thread_count=1),
                                  dtype == np.uint16)
    expected_background = base - (1 if dtype == np.uint8 else 0)
    original = frames[0][:32, :32].astype(np.int64) - expected_background
    output = result[:32, :32].astype(np.int64) - expected_background
    assert np.mean(output * output) < np.mean(original * original) * 0.6
    expected_pattern = pattern - 1 if dtype == np.uint8 else pattern
    np.testing.assert_array_equal(result[80:112, 80:112], expected_pattern[8:40, 8:40])


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('focused_frames', [1, 2])
def test_a_keeps_low_contrast_detail_with_one_or_more_focused_frames(dtype, focused_frames):
    shape = (129, 131)
    scale = 1 if dtype == np.uint8 else 256
    pattern = np.where(np.indices(shape).sum(axis=0) % 2,
                       124 * scale, 116 * scale).astype(dtype)
    frames = [np.full(shape, 120 * scale, dtype) for _ in range(8 - focused_frames)]
    frames += [pattern.copy() for _ in range(focused_frames)]
    result = quantize_fuse_output(openfocus_a_impl(frames, thread_count=1),
                                  dtype == np.uint16)
    expected = pattern - 1 if dtype == np.uint8 else pattern
    np.testing.assert_array_equal(result, expected)


def test_a_mapped_background_accumulators_match_in_memory(monkeypatch):
    from fusion_methods import openfocus_a

    rng = np.random.default_rng(422)
    frames = [np.clip(30000 + rng.integers(-600, 601, (97, 103, 3)),
                      0, 65535).astype(np.uint16) for _ in range(4)]
    expected = openfocus_a.openfocus_a_impl(frames, thread_count=1)
    monkeypatch.setattr(openfocus_a, '_BACKGROUND_RAM_LIMIT_BYTES', 0)
    actual = openfocus_a.openfocus_a_impl(frames, thread_count=1)
    np.testing.assert_array_equal(actual, expected)


def test_burt_odd_edges_do_not_amplify_repeated_samples():
    from fusion_methods.openfocus_a import _down2
    from fusion_methods.openfocus_ab import _check_cancel

    for shape in ((5, 4), (4, 5), (5, 5)):
        values = np.arange(np.prod(shape), dtype=np.int32).reshape(shape) * 937
        padded = np.pad(values, ((0, shape[0] % 2), (0, shape[1] % 2)), mode='edge')
        expected = (padded[0::2, 0::2] + padded[0::2, 1::2]
                    + padded[1::2, 0::2] + padded[1::2, 1::2]) >> 2
        np.testing.assert_array_equal(_down2(values, None, _check_cancel), expected)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('shape', [(1, 1), (1, 128), (2, 3), (2, 128), (20, 128),
                                   (128, 20), (64, 67), (128, 131), (256, 259), (512, 515)])
def test_c_identity_preserves_every_source_bit(dtype, shape):
    from fusion_methods.openfocus_bc import openfocus_c_impl

    rng = np.random.default_rng(2729)
    frame = rng.integers(0, np.iinfo(dtype).max + 1, shape + (3,), dtype=dtype)
    result = openfocus_c_impl([frame, frame.copy(), frame.copy()], thread_count=1)
    np.testing.assert_array_equal(quantize_fuse_output(result, dtype == np.uint16), frame)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('smoothing', [0, None])
@pytest.mark.parametrize('shape', [(1, 128), (2, 128), (20, 128), (128, 20)])
def test_c_small_images_select_focused_detail_from_each_source(shape, dtype, smoothing):
    from fusion_methods.openfocus_bc import openfocus_c_impl

    rows, cols = np.indices(shape)
    horizontal = shape[1] >= shape[0]
    axis = cols if horizontal else rows
    midpoint = max(shape) // 2
    scale = 1 if dtype == np.uint8 else 257
    detail = np.where(axis % 2, 220 * scale, 40 * scale).astype(dtype)
    first = np.where(axis < midpoint, detail, 128 * scale).astype(dtype)
    second = np.where(axis >= midpoint, detail, 128 * scale).astype(dtype)
    first = np.repeat(first[:, :, None], 3, axis=2)
    second = np.repeat(second[:, :, None], 3, axis=2)

    fused = openfocus_c_impl([first, second], blend_smoothing=smoothing, thread_count=1)
    pixels = quantize_fuse_output(fused, dtype == np.uint16).astype(np.float32) / scale
    left = pixels[:, 8:midpoint - 8] if horizontal else pixels[8:midpoint - 8, :]
    right = pixels[:, midpoint + 8:-8] if horizontal else pixels[midpoint + 8:-8, :]
    assert np.std(left) > 25
    assert np.std(right) > 25


@pytest.mark.parametrize('smoothing', [0, None])
def test_c_single_pixel_keeps_first_source_and_focus_winner(smoothing):
    from fusion_methods.openfocus_bc import openfocus_c_impl

    frames = [np.array([[value]], dtype=np.uint16) for value in (12345, 49153, 65535)]
    maps = []
    result = openfocus_c_impl(frames, blend_smoothing=smoothing, thread_count=1,
                             focus_callback=lambda indices, scale: maps.append((indices, scale)))

    np.testing.assert_array_equal(quantize_fuse_output(result, True), frames[0])
    (indices, scale), = maps
    np.testing.assert_array_equal(indices, np.zeros((1, 1), np.uint16))
    assert scale == len(frames) - 1


def _pyramid_axis(source, count, axis, expand):
    source = np.moveaxis(source, axis, 0)
    result = np.empty((count, *source.shape[1:]), np.float32)
    even = np.array([0.019027505, 0.318410337, 0.559359729, 0.103202470, 0], np.float32)
    down = np.array([0.125, 0.375, 0.375, 0.125], np.float32)
    for i, row in enumerate(result):
        row.fill(0)
        if expand:
            total = np.float32(0)
            for k, weight in enumerate(even[::-1] if i % 2 else even):
                index = i // 2 + k - 2
                if 0 <= index < source.shape[0]:
                    row += source[index] * weight
                    total += weight
            row /= total
        else:
            for k, weight in enumerate(down):
                index = min(source.shape[0] - 1, max(0, 2 * i + k - 1))
                row += source[index] * weight
    return np.moveaxis(result, 0, axis)


@pytest.mark.parametrize('shape', [(1, 131, 1), (133, 1, 2), (263, 135, 3)])
def test_c_banded_pyramid_matches_full_separable_passes(shape):
    from fusion_methods._openfocus_pyramid import _down, _synthesize

    height, width, channels = shape
    fine = np.random.default_rng(481).random(shape, dtype=np.float32)
    coarse = np.empty(((height + 1) // 2, (width + 1) // 2, channels), np.float32)
    expected_coarse = _pyramid_axis(
        _pyramid_axis(fine, coarse.shape[1], 1, False), coarse.shape[0], 0, False)
    _down(fine, coarse, lambda: None)
    np.testing.assert_array_equal(coarse, expected_coarse)

    expanded = _pyramid_axis(_pyramid_axis(coarse, width, 1, True), height, 0, True)
    denom = expanded + np.float32(0.1)
    expected_detail = (fine - expanded) / denom
    strength = np.zeros((height, width), np.float32)
    for channel in range(channels):
        strength += np.abs(expected_detail[:, :, channel])
    strength *= np.float32(3 / channels)
    expected_weight = np.minimum(np.float32(65535), strength * np.float32(21845)).astype(np.uint16)
    if height > 2:
        expected_weight[[0, -1], :] = 0
    if width > 2:
        expected_weight[:, [0, -1]] = 0

    detail = fine.copy()
    weights = np.empty(fine.shape[:2], np.uint16)
    _synthesize(coarse, detail, weights, lambda: None)
    np.testing.assert_array_equal(detail, expected_detail)
    np.testing.assert_array_equal(weights, expected_weight)
    _synthesize(coarse, detail, None, lambda: None)
    np.testing.assert_array_equal(detail, np.clip(denom * expected_detail + expanded, 0, 1))
