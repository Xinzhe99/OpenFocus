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
    """The default pipeline keeps B=1; CUDA batching still needs GPU gates."""
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


@pytest.mark.parametrize('shape', [(1, 1), (63, 65), (64, 64), (65, 67)])
@pytest.mark.parametrize('grayscale', [False, True])
def test_gray_staging_matches_single_replicate_padding(shape, grayscale):
    from fusion_methods import stackmffv3 as ai
    ai._ensure_torch()
    rng = np.random.default_rng(315)
    h, w = shape
    source_shape = (h, w * 2) if grayscale else (h, w * 2, 3)
    frames = [rng.integers(0, 65536, source_shape, np.uint16)[:, ::2]
              for _ in range(3)]
    expected = torch.from_numpy(np.stack([ai._gray01(f) for f in frames]))[None]
    expected = torch.nn.functional.pad(expected, (0, (-w) % 64, 0, (-h) % 64),
                                       mode='replicate')
    actual = ai._stage_gray(frames, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_true_uint16_grayscale_noncontiguous_fusion(mode):
    from fusion_methods import stackmffv3 as ai
    from utils.image_utils import normalize_fuse_input, quantize_fuse_output
    rng = np.random.default_rng(614)
    frames = [rng.integers(1, 65535, (17, 38), np.uint16)[:, ::2]
              for _ in range(3)]
    normalized, _ = normalize_fuse_input(frames)
    expected = ai._stackmffv3_batch_impl([normalized], WEIGHTS, False, mode,
                                       thread_count=1)[0]
    actual = ai._stackmffv3_batch_impl([frames], WEIGHTS, False, mode,
                                     thread_count=1)[0]
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    restored = quantize_fuse_output(actual, True)
    assert np.any(restored % 257), 'true 16-bit low bits were lost'
    if mode == 'hard':
        assert np.logical_or.reduce([restored == frame for frame in frames]).all()


def test_tile_batch_retains_order_and_each_tile_dtype():
    from fusion_methods import stackmffv3 as ai
    tiles = [[np.full((17, 19, 3), value, dtype) for _ in range(2)]
             for value, dtype in [(31, np.uint8), (32001, np.uint16),
                                  (190, np.uint8)]]
    results = ai._stackmffv3_batch_impl(tiles, WEIGHTS, False, 'hard', thread_count=1)
    for result, tile in zip(results, tiles):
        if tile[0].dtype == np.uint16:
            assert result.dtype == np.float32
            np.testing.assert_allclose(result, tile[0].astype(np.float32) / 257,
                                       rtol=0, atol=1e-5)
        else:
            np.testing.assert_array_equal(result, tile[0])
            assert result.dtype == np.uint8


def test_checkpoint_replacement_with_preserved_timestamp_invalidates_cache(tmp_path):
    import os
    import shutil
    from core.resource_policy import heavy_job
    from core.models.stackmff_v3.model_io import _torch_load
    from fusion_methods import stackmffv3 as ai
    checkpoint = tmp_path / 'model.pth'
    shutil.copyfile(WEIGHTS, checkpoint)
    stat = checkpoint.stat()
    with heavy_job():
        original, _ = ai._get_model_and_device(checkpoint, False)
        saved = _torch_load(checkpoint)
        for tensor in saved['model'].values():
            tensor.zero_()
        torch.save(saved, checkpoint)
        os.utime(checkpoint, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        replacement, _ = ai._get_model_and_device(checkpoint, False)
        assert replacement is not original
        assert all(torch.count_nonzero(p).item() == 0 for p in replacement.parameters())
        with torch.no_grad():
            probabilities = replacement.forward_probs(torch.rand(1, 3, 64, 64))
        torch.testing.assert_close(probabilities, torch.full_like(probabilities, 1 / 3))


def test_mutating_cached_model_dtype_reloads_fp32_model():
    from core.resource_policy import heavy_job
    from fusion_methods import stackmffv3 as ai
    with heavy_job():
        original, _ = ai._get_model_and_device(WEIGHTS, False)
        original.to(dtype=torch.float64)
        replacement, _ = ai._get_model_and_device(WEIGHTS, False)
        assert replacement is not original
        assert all(p.dtype == torch.float32 for p in replacement.parameters())


def test_cancel_between_native_inference_scales(monkeypatch):
    from core.resource_policy import heavy_job
    from core.registration import RegistrationCancelled
    from fusion_methods import stackmffv3 as ai
    cancelled = False
    completed = []
    with heavy_job():
        model, _ = ai._get_model_and_device(WEIGHTS, False)
        original = model.forward_logits

        def forward(x):
            nonlocal cancelled
            output = original(x)
            completed.append(tuple(x.shape))
            cancelled = True
            return output

        monkeypatch.setattr(model, 'forward_logits', forward)
        with pytest.raises(RegistrationCancelled), torch.no_grad():
            model.forward_probs(torch.rand(1, 2, 64, 64),
                                should_cancel=lambda: cancelled)
    assert len(completed) == 1


def test_cancel_during_color_rows_does_not_return_partial_output():
    from core.registration import RegistrationCancelled
    from fusion_methods import stackmffv3 as ai
    frames = np.ones((2, 513, 8, 3), np.float32)
    probabilities = np.full((2, 513, 8), 0.5, np.float32)
    polls = 0

    def cancel():
        nonlocal polls
        polls += 1
        return polls >= 3

    with pytest.raises(RegistrationCancelled):
        ai._blend_tile(probabilities, frames, 'soft', cancel)
    assert np.all(frames == 1)


def test_cancel_from_final_progress_prevents_fusion_return():
    from core.registration import RegistrationCancelled
    from fusion_methods import stackmffv3 as ai
    cancelled = False

    def progress(done, total):
        nonlocal cancelled
        cancelled = done == total

    frames = [np.full((17, 19, 3), value, np.uint8) for value in (30, 70)]
    with pytest.raises(RegistrationCancelled):
        ai._stackmffv3_impl(frames, None, WEIGHTS, False,
                           progress_callback=progress, should_cancel=lambda: cancelled,
                           thread_count=1)


def test_direct_probability_inference_cannot_bypass_memory_budget(monkeypatch):
    from core import resource_policy
    from fusion_methods import stackmffv3 as ai
    frames = [np.full((17, 19, 3), value, np.uint16) for value in (9001, 31003)]
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *args, **kwargs:
                        resource_policy.ResourceSnapshot(available_ram=1024,
                                                         total_ram=8 * 1024 ** 3))
    with pytest.raises(resource_policy.ResourceLimitError):
        ai.model_probabilities(frames, WEIGHTS, False)


@pytest.mark.parametrize('mode', ['soft', 'hard'])
def test_adapter_matches_baseline_bgr_numeric_pipeline(mode):
    import cv2
    from core.resource_policy import heavy_job
    from fusion_methods import stackmffv3 as ai
    rng = np.random.default_rng(905)
    frames = [rng.uniform(0, 255, (65, 134, 3)).astype(np.float32)[:, ::2]
              for _ in range(3)]
    with heavy_job():
        torch.set_num_threads(1)
        model, device = ai._get_model_and_device(WEIGHTS, False)
        luma = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) / 255.0 for f in frames])
        batch = torch.nn.functional.pad(torch.from_numpy(luma)[None],
                                        (0, 61, 0, 63), mode='replicate')
        with torch.no_grad():
            probs = model.forward_probs(batch.to(device),
                                        sigma=0.5 if mode == 'soft' else 0.0)
        probs = probs[0].cpu().numpy()
        colors = np.pad(np.stack(frames), ((0, 0), (0, 63), (0, 61), (0, 0)),
                        mode='edge')
        if mode == 'soft':
            expected = (probs[..., None] * colors).sum(0)[:65, :67]
        else:
            expected = colors[probs.argmax(0), np.arange(128)[:, None],
                              np.arange(128)[None, :]][:65, :67]
        expected = np.clip(expected, 0, 255)
    actual = ai._stackmffv3_batch_impl([frames], WEIGHTS, False, mode,
                                     thread_count=1)[0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def test_direct_cpu_inference_applies_thread_ceiling():
    from fusion_methods import stackmffv3 as ai
    frames = [np.full((17, 19, 3), value, np.uint8) for value in (30, 70)]
    ai.model_probabilities(frames, WEIGHTS, False, thread_count=1)
    assert torch.get_num_threads() == 1


@pytest.mark.parametrize('tiled', [False, True])
def test_real_neural_render_worker_preserves_uint16_low_bits(qapp, tiled):
    from core.workers import RenderWorker
    values = (np.arange(65 * 67, dtype=np.uint16).reshape(65, 67) * 11 + 1001)
    frame = np.repeat(values[..., None], 3, axis=2)
    worker = RenderWorker(
        raw_images=[frame, frame.copy()], aligned_images=[], is_images_aligned=False,
        last_alignment_options=(False, False), need_align_homography=False,
        need_align_ecc=False, need_fusion=True, rb_a_checked=False,
        rb_b_checked=False, rb_c_checked=False, rb_gfg_checked=False,
        rb_d_checked=True, kernel_slider_value=7, algorithm_override='stackmffv3',
        use_gpu=False, thread_count=1, ai_mode='hard', exposure_match=False,
        tile_enabled=tiled, tile_block_size=64, tile_overlap=16, tile_threshold=32)
    emitted = {}
    worker.finished_signal.connect(lambda *args: emitted.setdefault('result', args[1]))
    worker.error_signal.connect(lambda error: emitted.setdefault('error', error))
    worker.run()
    assert 'error' not in emitted, emitted.get('error')
    result = emitted['result']
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, frame)
    assert np.unique(result).size > 256
    assert np.any(result % 257)
