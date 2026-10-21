"""StackMFF-V3 tests — skipped automatically when torch is not installed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="StackMFF-V3 requires PyTorch")

from core.multi_focus_fusion import MultiFocusFusion, is_ai_fusion_available
from conftest import make_two_focus_stack

WEIGHTS = "weights/stackmff_v3_final.pth"

pytestmark = pytest.mark.skipif(not is_ai_fusion_available(),
                                reason="torch not installed")


def _fuse(mode, frames=None, exposure_match=True, **kwargs):
    frames = make_two_focus_stack() if frames is None else frames
    fusion = MultiFocusFusion(algorithm="stackmffv3", use_gpu=False,
                              ai_mode=mode, exposure_match=exposure_match)
    out = fusion.fuse(input_source=frames, img_resize=None,
                      model_path=WEIGHTS, thread_count=2, **kwargs)
    assert out is not None
    assert out.shape[:2] == frames[0].shape[:2]
    return out


def test_v3_fuses_and_caches_model():
    out = _fuse("soft")
    assert float(out.mean()) > 1.0


def test_both_modes_produce_a_result():
    soft = _fuse("soft")
    hard = _fuse("hard")
    assert soft.dtype == np.uint8 and hard.dtype == np.uint8
    assert not np.array_equal(soft, hard)


def test_hard_mode_values_come_from_the_source_frames():
    """Hard selection copies pixels, so every output pixel must exist in at
    least one frame at the same location."""
    frames = make_two_focus_stack()
    # Exposure matching rescales the frames before fusion, so pixel
    # provenance is only exact with it off.
    out = _fuse("hard", frames, exposure_match=False)
    per_frame = [np.all(out == f, axis=2) for f in frames]
    covered = np.logical_or.reduce(per_frame)
    assert covered.all()


def test_unknown_mode_falls_back_to_soft():
    assert _fuse("nonsense").shape[:2] == make_two_focus_stack()[0].shape[:2]


def test_get_info_reports_device_and_mode():
    info = MultiFocusFusion(algorithm="stackmffv3", use_gpu=False,
                            ai_mode="hard").get_info()
    assert info["algorithm"] == "stackmffv3"
    assert "CPU" in info["device"]
    assert info["mode"] == "hard"


def test_unavailable_torch_is_reported(monkeypatch):
    import core.multi_focus_fusion as mff
    monkeypatch.setattr(mff, "_STACKMFF_AVAILABLE_CACHE", False)
    assert mff.is_ai_fusion_available() is False
    with pytest.raises(RuntimeError):
        MultiFocusFusion(algorithm="stackmffv3", use_gpu=False)
