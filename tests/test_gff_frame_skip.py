"""Guided-filter fusion preserves output when a frame has no winning pixels."""
import cv2
import numpy as np
import pytest

from fusion_methods.gff import gff_impl


def _stack(dtype):
    rng = np.random.default_rng(0)
    scale = 65535 if dtype == np.uint16 else 255
    sharp = (rng.random((96, 128, 3)) * scale).astype(dtype)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4)
    winners = []
    for band in range(2):
        frame = blurred.copy()
        section = slice(band * 64, (band + 1) * 64)
        frame[:, section] = sharp[:, section]
        winners.append(frame)
    losers = [cv2.GaussianBlur(sharp, (0, 0), sigma) for sigma in (6, 8, 10)]
    return winners, losers


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_nonwinning_frames_leave_output_bit_identical(dtype):
    winners, losers = _stack(dtype)
    expected = gff_impl(list(winners), None)
    mixed = [losers[0], winners[0], losers[1], winners[1], losers[2]]
    actual = gff_impl(mixed, None)
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def test_each_winning_frame_still_contributes():
    winners, _ = _stack(np.uint8)
    both = gff_impl(list(winners), None).astype(np.int32)
    for band, frame in enumerate(winners):
        half = slice(band * 64 + 16, (band + 1) * 64 - 16)
        diff = np.abs(both[:, half] - frame[:, half].astype(np.int32))
        assert diff.mean() < 2.0


def test_nonwinning_nan_frame_still_affects_float_output():
    finite = np.zeros((12, 12, 3), dtype=np.float32)
    nonfinite = np.full_like(finite, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = gff_impl([finite, nonfinite], None)
    assert np.isnan(result).all()
