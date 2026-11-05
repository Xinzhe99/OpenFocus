"""StackMFF V3 ONNX inference and output contracts."""
import numpy as np
import pytest


from core.multi_focus_fusion import MultiFocusFusion, is_ai_fusion_available
from conftest import make_two_focus_stack

WEIGHTS = "weights/stackmff_v3_final.onnx"

pytestmark = pytest.mark.skipif(not is_ai_fusion_available(),
                                reason="ONNX Runtime not installed")


def _fuse(mode, frames=None, exposure_match=True, **kwargs):
    frames = make_two_focus_stack() if frames is None else frames
    fusion = MultiFocusFusion(algorithm="stackmffv3", use_gpu=False,
                              fusion_parameters={'mode': mode}, exposure_match=exposure_match)
    out = fusion.fuse(input_source=frames, img_resize=None,
                      model_path=WEIGHTS, thread_count=2, **kwargs)
    assert out is not None
    assert out.shape[:2] == frames[0].shape[:2]
    return out


def test_v3_fuses_on_cpu():
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


def test_tiled_hard_selection_does_not_blend_conflicting_tiles(monkeypatch):
    """Overlapping tiles may choose different source frames at one pixel."""
    import core.multi_focus_fusion as mff

    frames = [np.full((16, 24, 3), value, np.uint8) for value in (16, 240)]
    fusion = MultiFocusFusion('stackmffv3', use_gpu=False, exposure_match=False)

    def render(mode):
        choices = iter([0, 1] * 20)

        def predict(_fusion, crops, _resize, **_kwargs):
            return crops[next(choices)]

        monkeypatch.setattr(mff, 'resolve_runner', lambda _name: predict)
        return fusion._fuse_tiled_neural(
            frames, None, [], 16, 24, 3, 12, 4,
            prepare=lambda *args, **kwargs: (None, 'CPU', None),
            label='test', default_weights=WEIGHTS,
            mode_normalizer=lambda value: value, mode=mode)

    soft = render('soft')
    hard = render('hard')
    assert np.any((soft > 16) & (soft < 240))
    assert np.isin(hard, (16, 240)).all()



def test_get_info_reports_device_and_mode():
    info = MultiFocusFusion(algorithm="stackmffv3", use_gpu=False,
                            fusion_parameters={'mode': 'hard'}).get_info()
    assert info["algorithm"] == "stackmffv3"
    assert "CPU" in info["device"]
    assert info["mode"] == "hard"


def test_unavailable_onnx_runtime_is_reported(monkeypatch):
    import core.multi_focus_fusion as mff
    monkeypatch.setattr(mff, "_STACKMFF_AVAILABLE_CACHE", False)
    assert mff.is_ai_fusion_available() is False
    with pytest.raises(RuntimeError):
        MultiFocusFusion(algorithm="stackmffv3", use_gpu=False)


def test_tiled_path_honours_the_output_mode():
    frames = make_two_focus_stack()
    outs = {}
    for mode in ("soft", "hard"):
        fusion = MultiFocusFusion(
            algorithm='stackmffv3', use_gpu=False, fusion_parameters={'mode': mode},
            exposure_match=False, tile_enabled=True, tile_threshold=32,
            tile_block_size=64, tile_overlap=16)
        outs[mode] = fusion.fuse(input_source=list(frames), model_path=WEIGHTS,
                                 thread_count=1)
        assert outs[mode].shape[:2] == frames[0].shape[:2]
    assert not np.array_equal(outs["soft"], outs["hard"])

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
        algorithm='stackmffv3', use_gpu=False, fusion_parameters={'mode': 'soft'},
        exposure_match=False, tile_enabled=True,
        tile_block_size=64, tile_overlap=16, tile_threshold=32, ai_batch_size=2)
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
        algorithm='stackmffv3', use_gpu=False, fusion_parameters={'mode': 'soft'},
        exposure_match=False, tile_enabled=True,
        tile_block_size=64, tile_overlap=16, tile_threshold=32)
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
    from utils.image_utils import normalize_fuse_input

    rng = np.random.default_rng(315)
    h, w = shape
    source_shape = (h, w * 2) if grayscale else (h, w * 2, 3)
    frames = [rng.integers(0, 65536, source_shape, np.uint16)[:, ::2]
              for _ in range(3)]
    normalized, _ = normalize_fuse_input(frames)
    expected = np.stack([ai._gray01(f) for f in normalized])[None]
    expected = np.pad(expected, ((0, 0), (0, 0), (0, (-h) % 64),
                                 (0, (-w) % 64)), mode='edge')
    actual = ai._stage_gray(frames, None)
    np.testing.assert_array_equal(actual, expected)


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


def test_cancel_between_native_inference_scales(monkeypatch):
    from core.registration import RegistrationCancelled
    from fusion_methods import neural_runtime
    from fusion_methods import stackmffv3 as ai
    original = neural_runtime.logits
    completed = []

    def logits(*args, **kwargs):
        result = original(*args, **kwargs)
        completed.append(True)
        return result

    monkeypatch.setattr(neural_runtime, 'logits', logits)
    frames = [np.full((17, 19, 3), value, np.uint8) for value in (30, 70)]
    with pytest.raises(RegistrationCancelled):
        ai.model_probabilities(frames, WEIGHTS, False, thread_count=1,
                               should_cancel=lambda: bool(completed))
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
def test_adapter_matches_probability_weighted_source_pixels(mode):
    from fusion_methods import stackmffv3 as ai
    rng = np.random.default_rng(905)
    frames = [rng.uniform(0, 255, (17, 38, 3)).astype(np.float32)[:, ::2]
              for _ in range(3)]
    probs = ai.model_probabilities(frames, WEIGHTS, False,
                                   sigma=0.5 if mode == 'soft' else 0.0,
                                   thread_count=1)
    colors = np.stack(frames)
    if mode == 'soft':
        expected = (probs[..., None] * colors).sum(axis=0)
    else:
        expected = colors[probs.argmax(0), np.arange(17)[:, None],
                          np.arange(19)[None, :]]
    actual = ai._stackmffv3_batch_impl([frames], WEIGHTS, False, mode,
                                       thread_count=1)[0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize('tiled', [False, True])
def test_real_neural_render_worker_preserves_uint16_low_bits(qapp, tiled):
    from core.workers import RenderWorker
    values = (np.arange(65 * 67, dtype=np.uint16).reshape(65, 67) * 11 + 1001)
    frame = np.repeat(values[..., None], 3, axis=2)
    worker = RenderWorker(
        raw_images=[frame, frame.copy()], aligned_images=[], is_images_aligned=False,
        algorithm='stackmffv3', fusion_parameters={'mode': 'hard'},
        use_gpu=False, thread_count=1, exposure_match=False,
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
