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

def test_cpu_tiles_are_inferred_one_at_a_time(monkeypatch):
    """Real batching is GPU-only; the CPU path must keep B=1 per call.

    A batched CPU forward measured slower (19.3 s vs 16.9 s for two
    1024x1021 tiles), so the batch size may only change how many tiles are
    prepared, never how many are inferred together.
    """
    from fusion_methods import stackmffv3 as ai

    shapes = []
    original = ai._stackmffv3_batch_impl

    def spy(tiles_list, *args, **kwargs):
        shapes.append(len(tiles_list))
        return original(tiles_list, *args, **kwargs)

    monkeypatch.setattr(ai, "_stackmffv3_batch_impl", spy)
    fusion = MultiFocusFusion(
        algorithm="stackmffv3", use_gpu=False, ai_mode="soft",
        exposure_match=False, tile_enabled=True,
        tile_block_size=256, tile_overlap=64, ai_batch_size=2)
    fusion.fuse(input_source=make_two_focus_stack(), img_resize=None,
                model_path=WEIGHTS, thread_count=1)
    assert shapes, "the neural tiled path made no model call"
    assert set(shapes) == {1}, f"CPU tiles were batched: {shapes}"


def test_ai_tile_progress_never_skips_ahead(monkeypatch):
    """Progress may only report tiles whose inference already happened."""
    from fusion_methods import stackmffv3 as ai

    finished = []
    original = ai._stackmffv3_batch_impl

    def spy(tiles_list, *args, **kwargs):
        out = original(tiles_list, *args, **kwargs)
        finished.append(len(out))
        return out

    monkeypatch.setattr(ai, "_stackmffv3_batch_impl", spy)
    seen = []
    fusion = MultiFocusFusion(
        algorithm="stackmffv3", use_gpu=False, ai_mode="soft",
        exposure_match=False, tile_enabled=True,
        tile_block_size=256, tile_overlap=64)
    fusion.fuse(input_source=make_two_focus_stack(), img_resize=None,
                model_path=WEIGHTS,
                progress_callback=lambda done, total: seen.append((done, total)))
    assert seen
    totals = {total for _done, total in seen}
    assert len(totals) == 1
    assert [done for done, _t in seen] == list(range(0, totals.pop() + 1))
