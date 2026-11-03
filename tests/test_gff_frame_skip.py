"""Guided-filter fusion: frames that are never the sharpest change nothing.

A frame that wins no pixel of the initial decision map has an all-zero mask,
so its refined weights are exactly zero. Adding such frames to a stack must
leave the fused pixels bit-identical, while frames that do win still count.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _stack(dtype):
    rng = np.random.default_rng(0)
    scale = 65535 if dtype == np.uint16 else 255
    sharp = (rng.random((96, 128, 3)) * scale).astype(dtype)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4)
    winners = []
    for band in range(2):
        frame = blurred.copy()
        frame[:, band * 64:(band + 1) * 64] = sharp[:, band * 64:(band + 1) * 64]
        winners.append(frame)
    # Blurrier than every winner everywhere: never the sharpest frame.
    losers = [cv2.GaussianBlur(sharp, (0, 0), sigma) for sigma in (6, 8, 10)]
    return winners, losers


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_never_sharpest_frames_leave_output_bit_identical(dtype):
    from fusion_methods.gff import gff_impl

    winners, losers = _stack(dtype)
    expected = gff_impl(list(winners), None)

    # Interleave the losers before, between and after the winning frames.
    mixed = [losers[0], winners[0], losers[1], winners[1], losers[2]]
    actual = gff_impl(mixed, None)

    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def test_every_winning_frame_still_contributes():
    from fusion_methods.gff import gff_impl

    winners, _losers = _stack(np.uint8)
    both = gff_impl(list(winners), None).astype(np.int32)
    # Each half of the result comes from the frame that is sharp there.
    for band, frame in enumerate(winners):
        half = slice(band * 64 + 16, (band + 1) * 64 - 16)
        diff = np.abs(both[:, half] - frame[:, half].astype(np.int32))
        assert diff.mean() < 2.0
