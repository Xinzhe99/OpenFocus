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


@pytest.mark.parametrize('method', ['stackmff', 'stackmffv2', 'stackmffv3'])
@pytest.mark.parametrize('shape,block', [((65, 129), 64), ((190, 259), 128)])
def test_neural_tiles_cover_short_edge_without_black_seams(method, shape, block):
    pytest.importorskip('onnxruntime')
    from core.multi_focus_fusion import MultiFocusFusion

    frames = [np.full((*shape, 3), 100, np.uint8) for _ in range(2)]
    result = MultiFocusFusion(method, use_gpu=False, exposure_match=False,
                              tile_threshold=32, tile_block_size=block,
                              tile_overlap=block // 4).fuse(frames, thread_count=1)
    assert result.shape == frames[0].shape
    assert result.min() >= 95 and result.max() <= 105


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


def test_accelerator_sessions_keep_two_shapes_and_evict_oldest(monkeypatch, tmp_path):
    created = []

    class Options:
        def add_free_dimension_override_by_name(self, *args):
            pass

    class Session:
        def __init__(self, path, sess_options, providers):
            created.append(self)

        def disable_fallback(self):
            pass

        def get_providers(self):
            return ['DmlExecutionProvider', 'CPUExecutionProvider']

    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        SessionOptions=Options, InferenceSession=Session,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0)))
    monkeypatch.setattr(neural_runtime, '_SESSIONS', OrderedDict())
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'model')
    full = neural_runtime._session(str(path), (3, 256, 256), 'DIRECTML', 1)
    small = neural_runtime._session(str(path), (3, 192, 192), 'DIRECTML', 1)
    assert neural_runtime._session(str(path), (3, 256, 256), 'DIRECTML', 1) is full
    assert neural_runtime._session(str(path), (3, 192, 192), 'DIRECTML', 1) is small
    assert len(created) == 2
    neural_runtime._session(str(path), (3, 128, 128), 'DIRECTML', 1)
    assert neural_runtime._session(str(path), (3, 256, 256), 'DIRECTML', 1) is not full


def test_cached_gpu_workspace_is_not_charged_again(monkeypatch, tmp_path):
    from core import resource_policy

    path = tmp_path / 'model.onnx'
    path.write_bytes(b'model')
    planes = np.zeros((1, 2, 64, 64), np.float32)
    class Session:
        actual_device = 'DIRECTML'

        def run(self, outputs, inputs):
            return [inputs['input']]

    session = Session()
    key = neural_runtime._session_key(str(path), planes.shape[1:], 'DIRECTML', 1)
    monkeypatch.setattr(neural_runtime, '_SESSIONS', OrderedDict([(key, session)]))
    monkeypatch.setattr(neural_runtime, '_session', lambda *args: session)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *_: pytest.fail('cached workspace was charged again'))
    neural_runtime.start_inference_job()
    try:
        np.testing.assert_array_equal(neural_runtime.logits(planes, str(path), 'DIRECTML', 1), planes)
        assert neural_runtime.actual_device() == 'DIRECTML'
    finally:
        neural_runtime.start_inference_job()


def test_gpu_readmission_releases_old_session_before_fallback(monkeypatch, tmp_path):
    from core import resource_policy

    path = tmp_path / 'model.onnx'
    path.write_bytes(b'model')
    planes = np.zeros((1, 2, 64, 64), np.float32)
    class Session:
        actual_device = 'DIRECTML'

        def run(self, outputs, inputs):
            return [inputs['input']]

    old_key = neural_runtime._session_key(str(path), (2, 128, 128), 'DIRECTML', 1)
    monkeypatch.setattr(neural_runtime, '_SESSIONS', OrderedDict([(old_key, Session())]))
    snapshots = iter([
        resource_policy.ResourceSnapshot(free_vram=1, total_vram=1),
        resource_policy.ResourceSnapshot(free_vram=8 * 1024 ** 3, total_vram=8 * 1024 ** 3),
    ])
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *_: next(snapshots))
    requests = []
    def session(path, shape, device, threads):
        requests.append(device)
        return Session()

    monkeypatch.setattr(neural_runtime, '_session', session)
    neural_runtime.start_inference_job()
    try:
        neural_runtime.logits(planes, str(path), 'DIRECTML', 1)
        assert requests == ['DIRECTML']
        assert not neural_runtime.device_diagnostics()['fallback_cpu']
    finally:
        neural_runtime.start_inference_job()


def test_traced_session_released_before_replacement(monkeypatch, tmp_path):
    from utils import performance_profile

    events = []

    class Options:
        def add_free_dimension_override_by_name(self, name, value):
            if name == 'height':
                self.dimensions = value

    class Session:
        def __init__(self, path, sess_options, providers):
            self.shape = sess_options.dimensions
            events.append(('construct', self.shape))

        def disable_fallback(self):
            pass

        def end_profiling(self):
            events.append(('end_trace', self.shape))
            return ''

        def __del__(self):
            events.append(('destroy', self.shape))

    def make_session(path, sess_options, providers):
        sess_options.shape = sess_options.dimensions
        session = Session(path, sess_options, providers)
        session.shape = sess_options.shape
        return session

    def remember_dimension(self, name, value):
        if name == 'height':
            self.dimensions = value

    monkeypatch.setattr(Options, 'add_free_dimension_override_by_name', remember_dimension)
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        SessionOptions=Options, InferenceSession=make_session))
    monkeypatch.setattr(neural_runtime, '_SESSIONS', OrderedDict())
    monkeypatch.setattr(neural_runtime, '_TRACED', {})
    monkeypatch.setattr(neural_runtime, 'operator_trace_directory', lambda: str(tmp_path))
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'model')
    profile = performance_profile.PerformanceProfile()
    with performance_profile.active(profile):
        for height in (64, 128, 192):
            neural_runtime._session(str(model), (2, height, 64), 'CPU', 1)
    assert events.index(('end_trace', 64)) < events.index(('destroy', 64))
    assert events.index(('destroy', 64)) < events.index(('construct', 192))
    assert profile.report('completed')['sessions']['evictions'] == 1
    neural_runtime._clear_sessions()


@pytest.mark.parametrize('hardware', [
    [(2, 'Discrete adapter', 0x10de, 8 * 1024 ** 3)],
    [(0, 'Intel UHD integrated adapter', 0x8086, 0)],
])
def test_unknown_vram_does_not_erase_verified_provider_capability(monkeypatch, tmp_path, hardware):
    from core import resource_policy
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps([{'args': {'provider': 'DmlExecutionProvider'}}]))

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
    assert neural_runtime._DIRECTML_ADAPTER_INDEX == hardware[0][0]


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


def test_explicit_cpu_mode_does_not_probe_or_erase_accelerator_capability(monkeypatch):
    def unexpected_probe():
        raise AssertionError('explicit CPU mode must not probe a provider')

    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        get_available_providers=unexpected_probe))
    monkeypatch.setattr(neural_runtime, '_DEVICE', 'DIRECTML')
    assert neural_runtime.resolve_device(False) == 'CPU'
    assert neural_runtime.cached_device() == 'DIRECTML'


def test_registered_provider_and_discrete_adapter_failure_do_not_hide_usable_igpu(monkeypatch, tmp_path):
    from core import resource_policy
    tried = []
    selected = []

    class Options:
        def add_free_dimension_override_by_name(self, *args):
            pass

    class Session:
        def __init__(self, path, sess_options, providers):
            self.provider, self.options = providers[0]
            self.profile = tmp_path / f'probe-{len(tried)}.json'
            tried.append((self.provider, self.options['device_id']))

        def disable_fallback(self):
            pass

        def run(self, outputs, inputs):
            if self.provider == 'DmlExecutionProvider' and self.options['device_id'] == 0:
                raise RuntimeError('discrete driver cannot execute graph')
            return [np.zeros_like(inputs['input'])]

        def end_profiling(self):
            # CUDA registered successfully but placed all nodes on CPU.
            provider = ('CPUExecutionProvider' if self.provider == 'CUDAExecutionProvider'
                        else self.provider)
            self.profile.write_text(json.dumps([{'args': {'provider': provider}}]))
            return str(self.profile)

    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        get_available_providers=lambda: ['CUDAExecutionProvider', 'DmlExecutionProvider',
                                         'CPUExecutionProvider'],
        SessionOptions=Options, InferenceSession=Session,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0)))
    monkeypatch.setattr(neural_runtime, '_DEVICE', None)
    monkeypatch.setattr(neural_runtime, '_DIRECTML_ADAPTER_INDEX', 0)
    monkeypatch.setattr(neural_runtime, '_DIRECTML_ADAPTER_NAME', '')
    for name in ('_AVAILABLE_PROVIDERS', '_HARDWARE_ADAPTERS', '_PROBE_ERRORS'):
        monkeypatch.setattr(neural_runtime, name, ())
    monkeypatch.setattr(resource_policy, 'directml_adapters', lambda: [
        (0, 'Discrete adapter', 0x10de, 8 * 1024 ** 3),
        (1, 'Intel UHD integrated adapter', 0x8086, 0)])
    monkeypatch.setattr(resource_policy, 'set_directml_adapter', selected.append)
    assert neural_runtime.resolve_device(True) == 'DIRECTML'
    assert tried == [('CUDAExecutionProvider', 0), ('DmlExecutionProvider', 0),
                     ('DmlExecutionProvider', 1)]
    assert selected == [1]
    assert 'Intel UHD' in neural_runtime.display_name('DIRECTML')
    assert len(neural_runtime.device_diagnostics()['probe_errors']) == 2
    assert not list(tmp_path.glob('probe-*.json'))


@pytest.mark.parametrize('placement,expected,requests', [
    ('DmlExecutionProvider', 'DIRECTML', ['DIRECTML', 'DIRECTML']),
    ('CPUExecutionProvider', 'CPU', ['DIRECTML', 'CPU']),
    ('failure', 'CPU', ['DIRECTML', 'CPU', 'CPU']),
    ('unverified', 'CPU', ['DIRECTML', 'CPU', 'CPU']),
])
def test_real_session_placement_is_profiled_once_and_cpu_placement_sticks(
        monkeypatch, tmp_path, placement, expected, requests):
    from core import resource_policy
    requested = []
    profile = tmp_path / 'inference.json'
    ended = []

    class Session:
        def __init__(self, accelerated=False):
            self.accelerated = accelerated

        def run(self, outputs, inputs):
            if self.accelerated and placement == 'failure':
                raise RuntimeError('driver failed during inference')
            return [inputs['input'] + np.float32(1)]

        def end_profiling(self):
            ended.append(True)
            events = [] if placement == 'unverified' else [{'args': {'provider': placement}}]
            profile.write_text(json.dumps(events))
            return str(profile)

    profiled = neural_runtime._ProfiledSession(Session(accelerated=True), 'DIRECTML')
    def session(path, shape, device, threads):
        requested.append(device)
        return profiled if device == 'DIRECTML' else Session()

    monkeypatch.setattr(neural_runtime, '_session', session)
    monkeypatch.setattr(neural_runtime, '_DEVICE', 'DIRECTML')
    monkeypatch.setattr(neural_runtime, '_SESSIONS', OrderedDict())
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *_: resource_policy.ResourceSnapshot(
        available_ram=8 * 1024 ** 3, total_ram=16 * 1024 ** 3,
        free_vram=8 * 1024 ** 3, total_vram=8 * 1024 ** 3))
    monkeypatch.setattr(resource_policy, 'require_memory', lambda *_: None)
    neural_runtime.start_inference_job()
    try:
        planes = np.zeros((1, 2, 64, 64), np.float32)
        first = neural_runtime.logits(planes, 'model.onnx', 'DIRECTML', 1)
        second = neural_runtime.logits(planes, 'model.onnx', 'DIRECTML', 1)
        np.testing.assert_array_equal(first, planes + 1)
        np.testing.assert_array_equal(second, first)
        assert requested == requests
        assert neural_runtime.actual_device() == expected
        assert neural_runtime.cached_device() == 'DIRECTML'
        assert neural_runtime.device_diagnostics()['fallback_cpu'] == (expected == 'CPU')
        assert ended == [True]
        assert not profile.exists()
    finally:
        neural_runtime.start_inference_job()


def test_failed_inference_removes_its_profile(monkeypatch, tmp_path):
    profile = tmp_path / 'failed.json'

    class Session:
        def run(self, outputs, inputs):
            raise RuntimeError('driver failed during inference')

        def end_profiling(self):
            profile.write_text('[]')
            return str(profile)

    session = neural_runtime._ProfiledSession(Session(), 'DIRECTML')
    with pytest.raises(RuntimeError, match='driver failed'):
        session.run(['logits'], {'input': np.zeros((1, 2, 64, 64), np.float32)})
    assert session.actual_device is None
    assert not profile.exists()


@pytest.mark.parametrize('neural,use_gpu,checkpoint,capability,actual,actual_model,expected', [
    (False, True, 'v3.onnx', 'DIRECTML', 'DIRECTML', 'v3.onnx', ('CPU', 'selected')),
    (True, False, 'v3.onnx', 'DIRECTML', 'DIRECTML', 'v3.onnx', ('CPU', 'selected')),
    (True, True, 'v3.onnx', None, None, None, (None, 'pending')),
    (True, True, 'v3.onnx', 'DIRECTML', None, None, ('DIRECTML', 'ready')),
    (True, True, 'v3.onnx', 'N/A', None, None, ('CPU', 'selected')),
    (True, True, 'v3.onnx', 'DIRECTML', 'CPU', 'v3.onnx', ('CPU', 'actual')),
    (True, True, 'v1.onnx', 'DIRECTML', 'CPU', 'v3.onnx', ('DIRECTML', 'ready')),
    (True, True, 'v3.pth', 'DIRECTML', 'DIRECTML', 'v3.onnx', ('DIRECTML', 'actual')),
    (True, True, 'v3.onnx', 'CUDA', 'DIRECTML', 'v3.onnx', ('CUDA', 'ready')),
])
def test_execution_status_separates_capability_preference_and_actual_model(
        monkeypatch, neural, use_gpu, checkpoint, capability, actual, actual_model, expected):
    monkeypatch.setattr(neural_runtime, '_ACTUAL_DEVICE', actual)
    monkeypatch.setattr(neural_runtime, '_ACTUAL_CHECKPOINT', actual_model)
    assert neural_runtime.execution_status(neural, use_gpu, checkpoint, capability) == expected
