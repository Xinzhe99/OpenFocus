import numpy as np
import pytest

from core.stack_review import analyze_frames


@pytest.mark.parametrize("dtype,scale", [(np.uint8, 1), (np.uint16, 257)])
def test_neighbours_and_unique_focused_patch(dtype, scale):
    rng = np.random.default_rng(31)
    base = rng.integers(20, 180, (40, 50, 3), dtype=np.uint8).astype(dtype) * scale
    near = base + scale
    unique = near.copy()
    unique[10:14, 15:19] = 230 * scale
    result = analyze_frames(iter([base, base.copy(), near, unique]))
    assert [(item["index"], item["reference_index"]) for item in result] == [(1, 0), (2, 1)]
    assert result[0]["difference"] == 0
    assert result[1]["difference"] > 0


def test_tiny_mismatched_and_noncontiguous_frames():
    tiny = np.array([[12]], dtype=np.uint8)
    assert analyze_frames([tiny, tiny.copy()])[0]["index"] == 1
    assert analyze_frames([tiny, np.zeros((2, 1), np.uint8)]) == []
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)[:, ::2]
    assert len(analyze_frames([image, image.copy()])) == 1


def test_cancel_stops_during_comparison():
    calls = iter([False, False, True])
    with pytest.raises(InterruptedError):
        analyze_frames([np.zeros((2, 2), np.uint8)] * 2, lambda: next(calls))
