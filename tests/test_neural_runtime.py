"""Consumer and session isolation contracts for the shipped ONNX models."""
from collections import OrderedDict
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from fusion_methods import neural_runtime
from utils.weights import weights_path


def _focus_stack():
    texture = np.random.default_rng(37).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(texture, (9, 9), 2)
    top = (np.indices((64, 80))[0] < 32)[..., None]
    return [np.where(top, texture, blurred), np.where(top, blurred, texture)]


@pytest.mark.parametrize('tiled', [False, True])
def test_public_consumers_keep_v1_v3_models_and_reconstruction_distinct(tiled):
    pytest.importorskip('onnxruntime')
    from core.multi_focus_fusion import MultiFocusFusion
    from fusion_methods import stackmff_v1, stackmffv3

    frames = _focus_stack()
    actual = {}
    for method in ('stackmff', 'stackmffv3', 'stackmff'):
        fusion = MultiFocusFusion(method, use_gpu=False, exposure_match=False,
            tile_enabled=tiled, tile_block_size=64, tile_overlap=16, tile_threshold=32)
        result = fusion.fuse(frames, thread_count=1)
        if method in actual:
            np.testing.assert_array_equal(result, actual[method])
        actual[method] = result
    assert actual['stackmff'].shape == frames[0].shape
    assert actual['stackmff'].dtype == np.uint8
    # This focus-and-colour fixture exercises V1's closest-source chroma as
    # well as independent learned scores. Identical source frames need not
    # produce distinct outputs, so no universal inequality is claimed.
    assert np.max(np.abs(actual['stackmff'].astype(float) - actual['stackmffv3'])) > 1
    if not tiled:
        np.testing.assert_array_equal(actual['stackmff'], stackmff_v1.stackmff_impl(
            frames, None, weights_path('stackmff_final.onnx'), False, thread_count=1))
        np.testing.assert_array_equal(actual['stackmffv3'], stackmffv3._stackmffv3_impl(
            frames, None, weights_path('stackmff_v3_final.onnx'), False, thread_count=1))


def test_sessions_isolate_checkpoint_shape_and_directml_adapter(monkeypatch, tmp_path):
    calls = []

    class Options:
        def add_free_dimension_override_by_name(self, *args):
            pass

    class Session:
        def __init__(self, path, sess_options, providers):
            calls.append((path, providers))

        def disable_fallback(self):
            pass

        def get_providers(self):
            return ['DmlExecutionProvider', 'CPUExecutionProvider']

    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        SessionOptions=Options, InferenceSession=Session,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0)))
    monkeypatch.setattr(neural_runtime, '_SESSIONS', OrderedDict())
    first = tmp_path / 'v1.onnx'
    second = tmp_path / 'v3.onnx'
    first.write_bytes(b'v1')
    second.write_bytes(b'v3')
    a = neural_runtime._session(str(first), (2, 64, 64), 'CPU', 1)
    b = neural_runtime._session(str(second), (2, 64, 64), 'CPU', 1)
    assert a is not b
    assert neural_runtime._session(str(first), (2, 64, 64), 'CPU', 1) is a
    assert neural_runtime._session(str(first), (3, 64, 64), 'CPU', 1) is not a
    monkeypatch.setattr(neural_runtime, '_DIRECTML_ADAPTER_INDEX', 0)
    gpu = neural_runtime._session(str(first), (2, 64, 64), 'DIRECTML', 1)
    monkeypatch.setattr(neural_runtime, '_DIRECTML_ADAPTER_INDEX', 2)
    assert neural_runtime._session(str(first), (2, 64, 64), 'DIRECTML', 1) is not gpu
    assert calls[-1][1][0] == ('DmlExecutionProvider', {'device_id': 2})


def test_unknown_vram_does_not_erase_verified_provider_capability(monkeypatch, tmp_path):
    from core import resource_policy
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps([{'args': {'provider': 'DmlExecutionProvider'}}]))
    hardware = [(2, 'Discrete adapter', 0x10de, 8 * 1024 ** 3)]

    class Options:
        def add_free_dimension_override_by_name(self, *args):
            pass

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def disable_fallback(self):
            pass

        def run(self, *args):
            return [np.zeros((1, 2, 64, 64), np.float32)]

        def end_profiling(self):
            return str(profile)

    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        get_available_providers=lambda: ['DmlExecutionProvider', 'CPUExecutionProvider'],
        SessionOptions=Options, InferenceSession=Session,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0)))
    monkeypatch.setattr(neural_runtime, '_DEVICE', None)
    monkeypatch.setattr(neural_runtime, '_DIRECTML_ADAPTER_INDEX', 0)
    monkeypatch.setattr(neural_runtime, '_DIRECTML_ADAPTER_NAME', '')
    for name in ('_AVAILABLE_PROVIDERS', '_HARDWARE_ADAPTERS', '_PROBE_ERRORS'):
        monkeypatch.setattr(neural_runtime, name, ())
    monkeypatch.setattr(resource_policy, 'directml_adapters', lambda: hardware)
    monkeypatch.setattr(resource_policy, 'set_directml_adapter', lambda index: None)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *args: resource_policy.ResourceSnapshot())
    monkeypatch.setattr(neural_runtime, '_ACTUAL_DEVICE', None)
    monkeypatch.setattr(neural_runtime, '_ACTUAL_CHECKPOINT', None)
    assert neural_runtime.resolve_device(True) == 'DIRECTML'
    diagnostics = neural_runtime.device_diagnostics()
    assert diagnostics['hardware'] == tuple(hardware)
    assert diagnostics['selected'] == 'DIRECTML'
    assert diagnostics['actual'] is None
    assert 'DmlExecutionProvider' in diagnostics['providers']


def test_fusion_info_does_not_borrow_another_jobs_actual_device(monkeypatch):
    from core.multi_focus_fusion import MultiFocusFusion
    monkeypatch.setattr(neural_runtime, 'resolve_device', lambda use_gpu: 'DIRECTML')
    monkeypatch.setattr(neural_runtime, '_DEVICE', 'DIRECTML')
    monkeypatch.setattr(neural_runtime, '_DIRECTML_ADAPTER_NAME', '')
    first = MultiFocusFusion('stackmff', use_gpu=True)
    first._actual_device = 'CPU'
    second = MultiFocusFusion('stackmffv3', use_gpu=True)
    monkeypatch.setattr(neural_runtime, '_ACTUAL_DEVICE', 'DIRECTML')
    assert first.get_info()['device'] == 'CPU'
    assert second.get_info()['device'] == 'DirectML'
    classical = MultiFocusFusion('guided_filter', use_gpu=False)
    assert classical.get_info()['device'] == 'CPU'
    assert neural_runtime.cached_device() == 'DIRECTML'


def test_frozen_defaults_resolve_each_bundled_model(monkeypatch, tmp_path):
    from core import multi_focus_fusion
    from fusion_methods import stackmff_v1
    seen = []
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
    monkeypatch.setattr(stackmff_v1, 'stackmff_impl', lambda frames, resize, path, *a, **k: seen.append(path))
    monkeypatch.setattr(multi_focus_fusion, '_stackmffv3_impl',
                        lambda frames, resize, path, *a, **k: seen.append(path))
    multi_focus_fusion.MultiFocusFusion('stackmff', use_gpu=False)._fuse_stackmff([])
    multi_focus_fusion.MultiFocusFusion('stackmffv3', use_gpu=False)._fuse_stackmffv3([])
    from pathlib import Path
    assert [Path(path).name for path in seen] == ['stackmff_final.onnx', 'stackmff_v3_final.onnx']
    assert all(Path(path).parent == tmp_path / 'weights' for path in seen)


def test_v1_preserves_fused_luma_and_closest_source_chroma(monkeypatch):
    from fusion_methods import stackmff_v1
    frames = [np.full((8, 8, 3), color, np.uint8)
              for color in ((70, 80, 140), (140, 70, 60))]

    def probabilities(planes, path, device, threads, sigma, cancel):
        assert sigma == 0.5
        assert planes.shape == (1, 2, 8, 8)
        return np.broadcast_to(np.array([0.25, 0.75], np.float32)[None, :, None, None],
                               planes.shape)

    monkeypatch.setattr(neural_runtime, 'probabilities', probabilities)
    result = stackmff_v1._fuse_tile(frames, 'v1.onnx', False, thread_count=1)
    source = np.array((140, 70, 60), np.float32)
    source_luma = 0.114 * source[0] + 0.587 * source[1] + 0.299 * source[2]
    fused_luma = 0.25 * 97 + 0.75 * 75  # cv2's uint8 BT.601 luma.
    expected = np.broadcast_to(source + fused_luma - source_luma, result.shape)
    np.testing.assert_allclose(result, expected, atol=1)
    blended_color = 0.25 * frames[0] + 0.75 * frames[1]
    assert np.max(np.abs(result.astype(float) - blended_color)) > 10


def test_probabilities_use_both_released_scales_and_smooth_after_averaging(monkeypatch):
    calls = []

    def logits(planes, *args):
        calls.append(planes.shape)
        values = np.zeros_like(planes)
        values[:, 0, planes.shape[2] // 2, planes.shape[3] // 2] = 4
        return values

    monkeypatch.setattr(neural_runtime, 'logits', logits)
    planes = np.zeros((1, 2, 64, 80), np.float32)
    smooth = neural_runtime.probabilities(planes, 'v1.onnx', 'CPU', 1, 0.5)
    assert calls == [(1, 2, 64, 80), (1, 2, 48, 60)]
    raw = neural_runtime.probabilities(planes, 'v1.onnx', 'CPU', 1, 0)
    np.testing.assert_allclose(smooth.sum(axis=1), 1, atol=1e-6)
    assert smooth.max() < raw.max()
