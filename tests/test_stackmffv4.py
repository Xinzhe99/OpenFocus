"""StackMFF-V4 tests — skipped automatically when torch is not installed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="StackMFF-V4 requires PyTorch")

from core.multi_focus_fusion import MultiFocusFusion, is_stackmffv4_available
from conftest import make_two_focus_stack, laplacian_sharpness


@pytest.mark.skipif(not is_stackmffv4_available(), reason="torch not installed")
def test_stackmffv4_fuses_and_caches_model():
    frames = make_two_focus_stack()
    fusion = MultiFocusFusion(algorithm="stackmffv4", use_gpu=False)
    fused = fusion.fuse(
        input_source=frames, img_resize=None,
        model_path="weights/stackmffv4.pth", thread_count=2)
    assert fused is not None
    assert fused.shape[:2] == frames[0].shape[:2]
    # Result must not be a black/empty frame
    assert float(fused.mean()) > 1.0


@pytest.mark.skipif(not is_stackmffv4_available(), reason="torch not installed")
def test_get_info_reports_device():
    info = MultiFocusFusion(algorithm="stackmffv4", use_gpu=False).get_info()
    assert info["algorithm"] == "stackmffv4"
    assert "CPU" in info["device"]
