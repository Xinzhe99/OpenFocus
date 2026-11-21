"""Memory admission and device fallback for neural inference."""
import numpy as np
import pytest

from core import resource_policy
from fusion_methods import neural_runtime
from fusion_methods import stackmff_v1 as v1, stackmff_v2 as v2, stackmffv3 as v3

GiB = 1024 ** 3


def _snapshot(kind='CPU', free_vram=0, total_vram=0, available_ram=8 * GiB,
              commit_headroom=None):
    return resource_policy.ResourceSnapshot(
        available_ram=available_ram, total_ram=16 * GiB,
        physical_cores=4, logical_cpus=4, device_kind=kind,
        free_vram=free_vram, total_vram=total_vram, commit_headroom=commit_headroom)


def _frames():
    return [np.full((8, 8, 3), value, np.uint8) for value in (10, 90, 170)]


@pytest.mark.parametrize('adapter', [v1, v2, v3])
@pytest.mark.parametrize('device', ['CPU', 'DIRECTML', 'COREML', 'TORCH_MPS'])
def test_all_adapters_admit_host_memory_before_session(monkeypatch, adapter, device):
    monkeypatch.setattr(neural_runtime, 'resolve_device', lambda _: device)
    monkeypatch.setattr(neural_runtime, '_DEVICE', 'CPU')
    monkeypatch.setattr(neural_runtime, '_AVAILABLE_PROVIDERS', ())
    if adapter is v3:
        monkeypatch.setattr(v3, '_resolve_device', lambda _: device)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *a, **k: _snapshot())
    _, actual_device, plan = adapter._prepare_inference(
        _frames(), 'model.onnx', device != 'CPU', thread_count=2)
    assert actual_device == device
    assert plan.native_threads <= 2
    assert plan.ram_budget > 0

    monkeypatch.setattr(resource_policy, 'probe_resources',
                        lambda *a, **k: _snapshot(available_ram=1024))
    with pytest.raises(resource_policy.ResourceLimitError):
        adapter._prepare_inference(_frames(), 'model.onnx', device != 'CPU')

    monkeypatch.setattr(resource_policy, 'probe_resources',
                        lambda *a, **k: _snapshot(commit_headroom=1024))
    with pytest.raises(resource_policy.ResourceLimitError):
        adapter._prepare_inference(_frames(), 'model.onnx', device != 'CPU')


@pytest.mark.parametrize('kind', ['DIRECTML', 'CUDA', 'ROCM', 'XPU'])
@pytest.mark.parametrize('free,total', [(0, 0), (GiB // 2, 8 * GiB)])
def test_insufficient_or_unknown_device_memory_falls_back_before_inference(
        monkeypatch, kind, free, total):
    if kind not in neural_runtime._PROVIDER:
        pytest.skip(f'{kind} ONNX provider is not supported by this runtime')
    devices = []

    class Session:
        def run(self, _outputs, inputs):
            return [np.zeros_like(inputs['input'])]

    monkeypatch.setattr(neural_runtime, '_session',
                        lambda path, shape, device, threads: devices.append(device) or Session())
    monkeypatch.setattr(neural_runtime, '_DEVICE', kind)
    monkeypatch.setattr(neural_runtime, '_AVAILABLE_PROVIDERS', (neural_runtime._PROVIDER[kind],))
    monkeypatch.setattr(resource_policy, 'probe_resources',
                        lambda *a, **k: _snapshot(kind, free, total))
    neural_runtime.start_inference_job()
    planes = np.zeros((1, 3, 64, 64), np.float32)
    result = neural_runtime.logits(planes, 'model.onnx', kind, 1)
    assert result.shape == planes.shape
    assert devices == ['CPU']
    assert neural_runtime.actual_device() == 'CPU'
    assert neural_runtime.cached_device() == kind
    neural_runtime.start_inference_job()


def test_directml_with_sufficient_budget_uses_device(monkeypatch):
    devices = []

    class Session:
        actual_device = 'DIRECTML'
        def run(self, _outputs, inputs):
            return [np.zeros_like(inputs['input'])]

    monkeypatch.setattr(neural_runtime, '_session',
                        lambda path, shape, device, threads: devices.append(device) or Session())
    monkeypatch.setattr(resource_policy, 'probe_resources',
                        lambda *a, **k: _snapshot('DIRECTML', 8 * GiB, 8 * GiB))
    neural_runtime.start_inference_job()
    neural_runtime.logits(np.zeros((1, 3, 64, 64), np.float32), 'model.onnx', 'DIRECTML', 1)
    assert devices == ['DIRECTML']
    assert neural_runtime.actual_device() == 'DIRECTML'
    neural_runtime.start_inference_job()


def test_coreml_shared_memory_requires_host_admission(monkeypatch):
    monkeypatch.setattr(neural_runtime, 'resolve_device', lambda _: 'COREML')
    monkeypatch.setattr(resource_policy, 'probe_resources',
                        lambda *a, **k: _snapshot('COREML', available_ram=1024))
    with pytest.raises(resource_policy.ResourceLimitError):
        v2._prepare_inference(_frames(), 'model.onnx', True)


@pytest.mark.parametrize('fallback', ['COREML', 'CPU'])
def test_mps_host_charge_in_planners_and_preflight(monkeypatch, fallback):
    from core.render_preflight import estimate_render_memory
    from fusion_methods.neural_common import attention_inference_bytes

    device = 'TORCH_MPS'
    model = 'model.onnx'
    frames = [np.zeros((64, 64, 3), np.uint8)] * 3
    shape = (3, 64, 64, 3)
    estimate_inference = neural_runtime.estimate_inference_bytes
    startup = neural_runtime.native_startup_bytes(device)
    whole, vram = attention_inference_bytes(*shape, device)
    coreml, _ = attention_inference_bytes(*shape, 'COREML')
    workspace_delta = estimate_inference(3, 64, 64, device) - estimate_inference(3, 64, 64, 'COREML')
    assert whole == coreml + workspace_delta + startup and vram == 0
    tile_args = (*shape, 64)
    tile, _ = resource_policy.neural_tile_bytes(*tile_args, device, estimate_inference)
    coreml_tile, _ = resource_policy.neural_tile_bytes(*tile_args, 'COREML', estimate_inference)
    assert tile == coreml_tile + workspace_delta + startup

    monkeypatch.setattr(neural_runtime, 'resolve_device', lambda _: device)
    monkeypatch.setattr(neural_runtime, 'cached_device', lambda: device)
    monkeypatch.setattr(neural_runtime, '_DEVICE', fallback)
    monkeypatch.setattr(neural_runtime, '_AVAILABLE_PROVIDERS',
                        ('CoreMLExecutionProvider',) if fallback == 'COREML' else ())
    monkeypatch.setattr(neural_runtime, '_vram_snapshot', lambda _: pytest.fail('MPS queried VRAM'))
    headroom = whole
    monkeypatch.setattr(resource_policy, 'probe_resources',
                        lambda kind='CPU': _snapshot(kind, commit_headroom=headroom))
    with neural_runtime.inference_job():
        _, selected, plan = v1._prepare_inference(frames, model, True)
        assert selected == device and plan.vram_budget == 0
        preflight = estimate_render_memory(frames, 'stackmff', {})
        assert preflight.device == device and preflight.vram_required == 0

        headroom = tile
        block, plan = resource_policy.plan_neural_tiles(*tile_args, device, estimate_inference)
        assert block == 64 and plan.device_kind == device
        assert plan.required_ram == tile and plan.required_vram == 0
        headroom -= 1
        _, plan = resource_policy.plan_neural_tiles(*tile_args, device, estimate_inference)
        assert plan.device_kind == fallback

        headroom = whole - 1
        _, selected, _ = v1._prepare_inference(frames, model, True)
        assert selected == fallback
        preflight = estimate_render_memory(frames, 'stackmff', {})
        assert preflight.device == fallback and preflight.vram_required == 0
        assert preflight.accelerator == device and not preflight.ram_short
    neural_runtime.start_inference_job()


def test_mps_live_and_frozen_ram_limits_count_worker_once(monkeypatch):
    device = 'TORCH_MPS'
    shape = (3, 64, 64)
    inference = neural_runtime.estimate_inference_bytes(*shape, device) + int(np.prod(shape)) * 4
    startup = neural_runtime.native_startup_bytes(device)
    headroom = inference + startup
    monkeypatch.setattr(resource_policy, 'probe_memory', lambda: _snapshot(commit_headroom=headroom))
    monkeypatch.setattr(neural_runtime, '_vram_snapshot', lambda _: pytest.fail('MPS queried VRAM'))
    with neural_runtime.inference_job():
        plan = resource_policy.ExecutionPlan(ram_budget=headroom, device_kind=device)
        neural_runtime.apply_inference_budget(plan)
        assert neural_runtime._memory_refusal('model', shape, device, 1) is None
        headroom = inference
        assert 'Host memory budget' in neural_runtime._memory_refusal('model', shape, device, 1)
        with monkeypatch.context() as m:
            m.setattr(neural_runtime, '_WORKER', object())
            assert neural_runtime._memory_refusal('model', shape, device, 1) is None
            neural_runtime.apply_inference_budget(resource_policy.ExecutionPlan(
                ram_budget=inference + startup - 1, device_kind=device))
            assert 'Host memory plan budget' in neural_runtime._memory_refusal('model', shape, device, 1)
    neural_runtime.start_inference_job()


@pytest.mark.parametrize('device', ['TORCH_CUDA', 'TORCH_MPS'])
def test_deep_tile_planner_admits_native_attention_with_its_bounded_workspace(monkeypatch, device):
    from core.models.attention_workspace import released_attention_workspace_bytes

    estimate = neural_runtime.estimate_inference_bytes
    tile_args = (500, 4320, 7680, 3, 128)
    host, vram = resource_policy.neural_tile_bytes(*tile_args, device, estimate)
    assert released_attention_workspace_bytes(500, 128, 128) == 64 * 1024**2
    assert vram < 4 * GiB and estimate(500, 128, 128, 'CUDA') > vram
    monkeypatch.setattr(resource_policy, 'vram_budget', lambda _: vram)
    snapshot = _snapshot(device, free_vram=4 * GiB, total_vram=4 * GiB, commit_headroom=host)
    block, plan = resource_policy.plan_neural_tiles(*tile_args, device, estimate, snapshot=snapshot)
    assert block == 128 and plan.device_kind == device
    assert plan.required_ram == host
    assert plan.required_vram == (vram if device == 'TORCH_CUDA' else 0)


def test_deep_tile_plan_prefers_larger_native_cuda_fit(monkeypatch):
    args = (500, 4320, 7680, 3, 128)
    estimate = neural_runtime.estimate_inference_bytes
    monkeypatch.setattr(neural_runtime, '_JOB_PACK', (('TENSORRT', 'TORCH_CUDA'), (), 'NVIDIA'))
    monkeypatch.setattr(neural_runtime, '_DEVICE', 'CPU')
    monkeypatch.setattr(neural_runtime, '_AVAILABLE_PROVIDERS', ())
    vram = resource_policy.neural_tile_bytes(*args, 'TORCH_CUDA', estimate)[1]
    device = 'TENSORRT'
    snapshot = _snapshot(device, free_vram=8 * GiB, total_vram=8 * GiB,
                         available_ram=16 * GiB)
    monkeypatch.setattr(resource_policy, 'vram_budget', lambda _: vram)
    block, plan = resource_policy.plan_neural_tiles(*args, device, estimate, snapshot=snapshot)
    assert (block, plan.device_kind) == (128, 'TORCH_CUDA')

    vram = resource_policy.neural_tile_bytes(*args, 'TENSORRT', estimate)[1]
    block, plan = resource_policy.plan_neural_tiles(*args, device, estimate, snapshot=snapshot)
    assert (block, plan.device_kind) == (128, 'TENSORRT')
