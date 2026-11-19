"""Exact integer exposure matching and masked range cleaning."""

import cv2
import numpy as np
import pytest

from core.background_fill import _apply_match_gains, _match_medians, _u16, recorded_range


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("layout", ["gray", "bgr", "bgra"])
def test_integer_matching_reproduces_float32_reference(dtype, layout):
    rng = np.random.default_rng(846)
    shape = (271, 301) if layout == "gray" else (271, 301, 3 if layout == "bgr" else 4)
    frame = rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)[:, ::-1]
    sample = np.ones(frame.shape[:2], bool)
    sample[::3, ::4] = False
    reference = _u16(frame).astype(np.float32)
    # The large sample exercises the full 65536-bin histogram for 16-bit
    # input. A pair of pixels also exercises exact even-sample averaging.
    for selection in (sample, np.indices(sample.shape).sum(axis=0) == 1):
        expected = [np.median(reference[..., c][selection]) for c in range(3)]
        np.testing.assert_array_equal(_match_medians(frame, selection, np.count_nonzero(selection)), expected)
    gains = np.array([0.8927963, 1.0018321, 1.2472357], np.float32)
    np.testing.assert_array_equal(_apply_match_gains(frame, gains), reference * gains)


def test_range_reports_each_completed_frame_and_preserves_warp_fill():
    rng = np.random.default_rng(519)
    frames = [rng.random((77, 105, 3), dtype=np.float32) for _ in range(3)]
    shape = frames[0].shape[:2]
    mask = np.ones(shape, bool)
    mask[:, :60] = False
    # Absent data in every frame must leave the recorded interval open.
    recorded = []
    for frame in frames:
        small = cv2.resize(frame * mask[..., None], (26, 19), interpolation=cv2.INTER_AREA)
        numerator = cv2.GaussianBlur(small, (0, 0), 1.5)
        denominator = cv2.GaussianBlur(cv2.resize(mask.astype(np.float32), (26, 19),
                                                   interpolation=cv2.INTER_AREA), (0, 0), 1.5)
        recorded.append(numerator / np.maximum(denominator, 1e-12)[..., None])
    expected_lo = np.min(recorded, axis=0)
    expected_hi = np.max(recorded, axis=0)
    expected_lo[denominator <= 1e-3] = -np.inf
    expected_hi[denominator <= 1e-3] = np.inf
    fractions = []
    lo, hi = recorded_range(3, lambda k: (frames[k], None), shape, sigma=6,
                            valid_masks=[mask] * 3, progress=fractions.append)
    np.testing.assert_array_equal(lo, expected_lo)
    np.testing.assert_array_equal(hi, expected_hi)
    assert fractions == [1 / 3, 2 / 3, 1.0]
