"""Regression coverage for bounded neural source reconstruction."""
import numpy as np
import pytest

from fusion_methods import neural_common, stackmffv3
from utils.image_utils import normalize_fuse_input


def _mixed_frames():
    rng = np.random.default_rng(42)
    return [rng.integers(0, 256, (13, 17, 3), dtype=np.uint8),
            rng.integers(0, 65536, (13, 17, 3), dtype=np.uint16),
            rng.integers(0, 256, (13, 17, 3), dtype=np.uint8)]


@pytest.mark.parametrize('mode', ['hard', 'soft'])
def test_streamed_v3_color_matches_full_stack_reference(mode):
    frames = _mixed_frames()
    rng = np.random.default_rng(7)
    probs = rng.random((len(frames), 13, 17), dtype=np.float32)
    probs /= probs.sum(axis=0, keepdims=True)
    normalized, _ = normalize_fuse_input(frames)
    stacked = np.stack(normalized).astype(np.float32)
    if mode == 'hard':
        index = probs.argmax(axis=0)
        expected = stacked[index, np.arange(13)[:, None],
                           np.arange(17)[None, :]]
    else:
        expected = (probs[..., None] * stacked).sum(axis=0)
    np.testing.assert_allclose(stackmffv3._blend_tile(probs, frames, mode),
                               expected, rtol=1e-6, atol=1e-5)


def test_streamed_v1_color_keeps_nearest_frame_chroma():
    frames = _mixed_frames()
    normalized, _ = normalize_fuse_input(frames)
    colors = [neural_common._ycbcr(frame) for frame in normalized]
    target = np.random.default_rng(4).uniform(0, 255, (13, 17))
    target = target.astype(np.float32)
    nearest = np.abs(np.stack([item[0] for item in colors]) - target)
    nearest = nearest.argmin(axis=0)
    rows, cols = np.arange(13)[:, None], np.arange(17)[None, :]
    cb = np.stack([item[1] for item in colors])[nearest, rows, cols]
    cr = np.stack([item[2] for item in colors])[nearest, rows, cols]
    expected = np.clip(neural_common._ycbcr_to_bgr(target, cb, cr), 0, 255)
    np.testing.assert_array_equal(neural_common.tensor_to_bgr(target, frames),
                                  expected)


def test_v2_single_padded_tile_matches_whole_frame_depth():
    pytest.importorskip('onnxruntime')
    from fusion_methods.stackmff_v2 import infer
    from fusion_methods.stackmff_v2_tiled import _Source, tiled_depth
    from utils.weights import weights_path

    frames = _mixed_frames()
    model = weights_path('stackmff_v2_final.onnx')
    _image, expected_depth, expected_index = infer(
        frames, model, False, thread_count=1)
    depth, index = tiled_depth(_Source(frames, None, None, None), model, False,
                               tile_size=32, thread_count=1)
    np.testing.assert_allclose(depth, expected_depth, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(index, expected_index)


def test_two_hundred_large_frames_get_a_bounded_full_stack_plan():
    from core.resource_policy import ResourceSnapshot, plan_neural_tiles
    from fusion_methods.neural_runtime import estimate_inference_bytes

    gib = 1024 ** 3
    snapshot = ResourceSnapshot(available_ram=12 * gib, total_ram=16 * gib,
                                physical_cores=4, logical_cpus=8)
    block, plan = plan_neural_tiles(
        200, 5520, 8288, 3, 1024, 'CPU', estimate_inference_bytes,
        snapshot=snapshot)
    assert 64 <= block <= 1024
    assert plan.workers == plan.max_in_flight == 1
    assert plan.ram_budget <= snapshot.available_ram


@pytest.mark.parametrize('entry', ['fusion', 'stackmff', 'stackmffv2', 'stackmffv3'])
def test_directory_rejects_oversized_frame_before_decode(tmp_path, monkeypatch, entry):
    import cv2
    from core import image_loader, resource_policy
    from core.multi_focus_fusion import MultiFocusFusion
    from fusion_methods.stackmff_v2_tiled import _Source

    frame = np.zeros((8, 8, 3), np.uint8)
    for name in ('first.png', 'second.png'):
        assert cv2.imwrite(str(tmp_path / name), frame)
    snapshot = resource_policy.ResourceSnapshot(
        available_ram=2 * 1024 ** 3, total_ram=4 * 1024 ** 3,
        physical_cores=2, logical_cpus=2)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *a, **k: snapshot)
    monkeypatch.setattr(image_loader, 'source_dimensions',
                        lambda path: (8, 8) if path.endswith('first.png') else (500000, 500000))
    decoded = []
    def decode(path, **kwargs):
        decoded.append(path)
        return frame.copy()
    monkeypatch.setattr(image_loader, 'decode_image', decode)

    with pytest.raises(resource_policy.ResourceLimitError):
        if entry == 'fusion':
            MultiFocusFusion(algorithm='guided_filter', use_gpu=False).fuse(str(tmp_path))
        elif entry == 'stackmff':
            neural_common.read_stack(str(tmp_path), None)
        elif entry == 'stackmffv2':
            source = _Source(str(tmp_path), None, None, None)
            source.get(1)
        else:
            stackmffv3._read_stack(str(tmp_path), None)
    assert decoded == [str(tmp_path / 'first.png')]


@pytest.mark.parametrize('method', ['stackmff', 'stackmffv2', 'stackmffv3'])
@pytest.mark.parametrize('exposure_match', [False, True])
def test_direct_neural_mixed_depth_matches_normalized_consumer(method, exposure_match):
    pytest.importorskip('onnxruntime')
    from core.multi_focus_fusion import MultiFocusFusion
    from utils.image_utils import quantize_fuse_output

    frames = _mixed_frames()
    originals = [frame.copy() for frame in frames]
    normalized, is_16bit = normalize_fuse_input(frames)
    outputs = []
    for inputs in (normalized, frames):
        fusion = MultiFocusFusion(method, use_gpu=False, tile_enabled=False,
                                  exposure_match=exposure_match)
        outputs.append(quantize_fuse_output(fusion.fuse(inputs, thread_count=1), is_16bit))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    for frame, original in zip(frames, originals):
        np.testing.assert_array_equal(frame, original)
