"""Memory admission and device fallback for ONNX neural inference."""
import numpy as np
import pytest

from core import resource_policy
from fusion_methods import neural_runtime
from fusion_methods import stackmff_v1 as v1, stackmff_v2 as v2, stackmffv3 as v3

GiB = 1024 ** 3


def _snapshot(kind='CPU', free_vram=0, total_vram=0, available_ram=8 * GiB):
    return resource_policy.ResourceSnapshot(
        available_ram=available_ram, total_ram=16 * GiB,
        physical_cores=4, logical_cpus=4, device_kind=kind,
        free_vram=free_vram, total_vram=total_vram)


def _frames():
    return [np.full((8, 8, 3), value, np.uint8) for value in (10, 90, 170)]


@pytest.mark.parametrize('adapter', [v1, v2, v3])
@pytest.mark.parametrize('device', ['CPU', 'DIRECTML', 'COREML'])
def test_all_adapters_admit_host_memory_before_session(monkeypatch, adapter, device):
    monkeypatch.setattr(neural_runtime, 'resolve_device', lambda _: device)
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
