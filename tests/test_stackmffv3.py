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


def test_tiled_path_honours_the_output_mode():
    """A stack large enough to tile must still fuse in the chosen mode.

    The mode default was set AFTER the tiled dispatch returned, so every
    tiled run used 'soft' regardless of ai_mode. A 999x750 demo stack hit
    that branch and produced byte-identical soft and hard output.
    """
    import cv2
    frames = [cv2.resize(f, (700, 520), interpolation=cv2.INTER_CUBIC)
              for f in make_two_focus_stack()]
    outs = {}
    for mode in ("soft", "hard"):
        fusion = MultiFocusFusion(
            algorithm="stackmffv3", use_gpu=False, ai_mode=mode,
            exposure_match=False, tile_enabled=True,
            tile_block_size=512, tile_overlap=128)
        outs[mode] = fusion.fuse(input_source=list(frames), img_resize=None,
                                 model_path=WEIGHTS)
        assert outs[mode].shape[:2] == frames[0].shape[:2]
    assert not np.array_equal(outs["soft"], outs["hard"]), (
        "the tiled path ignored ai_mode: soft and hard produced the same "
        "image")
