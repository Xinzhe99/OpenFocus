"""Golden-image regression: fuse the committed demo stack and compare the
result against a committed baseline within a tolerance.

Tolerant on purpose — float ordering differs slightly across CPUs — but any
structural regression (black seams, wrong frame picked, bit-depth collapse)
moves the metrics by orders of magnitude and fails.
"""
import os

import numpy as np
import cv2
import pytest

cv2 = pytest.importorskip("cv2")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(PROJECT_ROOT, "assets", "demo_stack")
BASELINE = os.path.join(PROJECT_ROOT, "tests", "golden", "demo_guided_filter.png")

TOLERANCE_MEAN = 2.0   # mean |diff| in 0-255 units
TOLERANCE_P99 = 32.0   # 99th percentile |diff|


def _load_frames():
    names = sorted(os.listdir(DEMO))
    frames = [cv2.imdecode(np.fromfile(os.path.join(DEMO, n), dtype=np.uint8),
                           cv2.IMREAD_COLOR) for n in names
              if n.lower().endswith((".jpg", ".png"))]
    assert len(frames) >= 2
    return frames


def _metrics(a, b):
    a = a.astype(np.int16)
    b = b.astype(np.int16)
    diff = np.abs(a - b)
    return float(diff.mean()), float(np.percentile(diff, 99))


def test_demo_stack_guided_filter_matches_golden():
    from core.multi_focus_fusion import MultiFocusFusion

    assert os.path.isdir(DEMO) and os.path.isfile(BASELINE), \
        "demo stack or golden baseline is missing from the checkout"

    fusion = MultiFocusFusion(algorithm="guided_filter", use_gpu=False)
    fused = fusion.fuse(input_source=_load_frames(), img_resize=None, kernel_size=31)
    assert fused is not None and fused.ndim == 3

    golden = cv2.imread(BASELINE, cv2.IMREAD_COLOR)
    assert golden is not None and golden.shape == fused.shape, \
        "golden baseline shape mismatch — regenerate tests/golden/demo_guided_filter.png"

    mean_diff, p99_diff = _metrics(fused, golden)
    assert mean_diff <= TOLERANCE_MEAN, f"mean diff {mean_diff:.2f}"
    assert p99_diff <= TOLERANCE_P99, f"p99 diff {p99_diff:.2f}"


def test_fusion_is_deterministic():
    """Same input, same output: the tiled/classical paths accumulate in fixed
    frame order (a thread-completion-order bug shipped once)."""
    from core.multi_focus_fusion import MultiFocusFusion

    frames = _load_frames()
    fusion = MultiFocusFusion(algorithm="guided_filter", use_gpu=False)
    a = fusion.fuse(input_source=[f.copy() for f in frames], img_resize=None, kernel_size=31)
    b = fusion.fuse(input_source=[f.copy() for f in frames], img_resize=None, kernel_size=31)
    assert np.array_equal(a, b)
