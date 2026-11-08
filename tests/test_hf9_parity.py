"""Consumer-visible HF9 integer-domain, source selection and identity boundaries."""
import numpy as np
import pytest

from fusion_methods.openfocus_a import openfocus_a_impl
from utils.image_utils import quantize_fuse_output


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('shape', [(5, 4), (4, 5), (5, 5), (65, 67)])
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


def test_a_mixed_stack_uses_256_for_eight_bit_pixels():
    pattern = np.where(np.indices((33, 35)).sum(axis=0) % 2, 200, 20).astype(np.uint8)
    result = openfocus_a_impl([np.full(pattern.shape, 10000, np.uint16), pattern],
                             thread_count=1)
    np.testing.assert_array_equal(quantize_fuse_output(result, True),
                                  pattern.astype(np.uint16) << 8)


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
@pytest.mark.parametrize('shape', [(64, 67), (128, 131), (256, 259), (512, 515)])
def test_c_identity_preserves_every_source_bit(dtype, shape):
    from fusion_methods.openfocus_bc import openfocus_c_impl

    rng = np.random.default_rng(2729)
    frame = rng.integers(0, np.iinfo(dtype).max + 1, shape + (3,), dtype=dtype)
    result = openfocus_c_impl([frame, frame.copy(), frame.copy()], thread_count=1)
    np.testing.assert_array_equal(quantize_fuse_output(result, dtype == np.uint16), frame)
