"""Exercise the real worker transport without requiring a GPU or pack install."""
import gc
import os
import subprocess
import sys
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from core import resource_policy
from core.registration import RegistrationCancelled
from fusion_methods import inference_client, neural_runtime as runtime


@pytest.fixture
def worker_command(tmp_path):
    script = tmp_path / 'protocol.py'
    script.write_text('''import array, json, math, mmap, os, sys, time
print(json.dumps({'ready':True,'backends':['TORCH_CUDA','TORCH_MPS'],'device':'test device'}), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    assert all(os.path.isabs(r[k]) for k in ('model', 'cache'))
    if 'exit.onnx' in r['model']:
        sys.exit(3)
    if 'delay.onnx' in r['model']:
        time.sleep(60)
    if os.name == 'nt':
        m = mmap.mmap(-1, math.prod(r['shape']) * 4, tagname=r['mapping'])
    else:
        with open(r['tensor'], 'r+b') as f:
            m = mmap.mmap(f.fileno(), 0)
    with m:
        data = array.array('f', m[:])
        m[:] = array.array('f', (v + 2 for v in data)).tobytes()
    print(json.dumps({'ok':True,'backend':r['backend']}), flush=True)
''', encoding='utf-8')
    return [sys.executable, '-I', '-u', str(script)]


@pytest.mark.parametrize('backend', ['TORCH_CUDA', 'TORCH_MPS'])
def test_worker_maps_independent_input_and_releases_output(worker_command, tmp_path, backend):
    worker = inference_client.Worker(worker_command, None)
    planes = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
    try:
        assert backend in worker.backends
        result = worker.run(backend, str(tmp_path / 'original.onnx'), planes, tmp_path, None)
        path = Path(result.filename) if isinstance(result, np.memmap) else None
        retained = result[:, :, :1]
        np.testing.assert_array_equal(result, planes + 2)
        assert planes[0, 0, 0, 0] == 0
        del result
        np.testing.assert_array_equal(retained, planes[:, :, :1] + 2)
        del retained
        gc.collect()
        if path:
            assert not path.exists()
    finally:
        worker.close()
    assert worker.process.poll() == 0


@pytest.mark.parametrize('mode', ['exit', 'timeout', 'cancel'])
def test_worker_failure_reaps_child(worker_command, tmp_path, mode):
    worker = inference_client.Worker(worker_command, None)
    if mode == 'timeout':
        worker.run_timeout = 0.05
    checks = iter([False, False, True])
    cancel = (lambda: next(checks, True)) if mode == 'cancel' else None
    model = tmp_path / ('exit.onnx' if mode == 'exit' else 'delay.onnx')
    try:
        with pytest.raises(RegistrationCancelled if mode == 'cancel' else inference_client.WorkerLost):
            worker.run('TORCH_MPS', str(model), np.zeros((1, 2, 4, 4), np.float32), tmp_path, cancel)
        assert worker.process.poll() is not None
    finally:
        worker.close()


def test_slow_termination_keeps_completed_logits(worker_command, tmp_path, monkeypatch):
    worker = inference_client.Worker(worker_command, None)
    planes = np.zeros((1, 2, 4, 4), np.float32)
    waited = Event()
    wait = worker.process.wait

    def wait_after_timeout(timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired(worker.process.args, timeout)
        try:
            return wait()
        finally:
            waited.set()

    try:
        result = worker.run('TORCH_CUDA', str(tmp_path / 'model.onnx'), planes, tmp_path, None)
        monkeypatch.setattr(worker.process, 'wait', wait_after_timeout)
    finally:
        worker.close()
    assert waited.wait(5) and worker.process.poll() is not None
    np.testing.assert_array_equal(result, planes + 2)


def test_native_failure_only_memoizes_current_shape(monkeypatch, tmp_path):
    model = tmp_path / 'selected.onnx'
    model.write_bytes(b'graph')
    monkeypatch.setattr(runtime, '_PACK', (('TENSORRT', 'TORCH_CUDA'), ('installed',), 'test GPU'))
    monkeypatch.setattr(runtime, '_DEVICE', 'CUDA')
    monkeypatch.setattr(runtime, '_AVAILABLE_PROVIDERS', ('CUDAExecutionProvider', 'DmlExecutionProvider'))
    monkeypatch.setattr(runtime, '_memory_refusal', lambda *_: None)
    paths = []
    def native(planes, checkpoint, device, *_):
        paths.append((device, planes.shape[2], checkpoint))
        if planes.shape[2] == 64:
            raise inference_client.WorkerError('unsupported graph shape')
        return planes, device
    monkeypatch.setattr(runtime, '_run_native', native)
    def ort(planes, path, device, *_):
        paths.append((device, planes.shape[2], path))
        if device == 'CUDA':
            raise RuntimeError('provider unavailable')
        return planes, device
    monkeypatch.setattr(runtime, '_run_ort', ort)
    with runtime.inference_job():
        for height in (64, 64, 128):
            runtime.logits(np.zeros((1, 3, height, 64), np.float32), str(model), 'TENSORRT', 1)
        assert [x[0] for x in paths] == ['TENSORRT', 'TORCH_CUDA', 'CUDA', 'DIRECTML', 'DIRECTML', 'TENSORRT']
        assert all(x[2] == str(model) for x in paths)
        assert runtime.device_diagnostics()['actual_devices'] == ('DIRECTML', 'TENSORRT')
        assert any('unsupported graph shape' in s for s in runtime.device_diagnostics()['fallback_reasons'])
    with runtime.inference_job():
        runtime.logits(np.zeros((1, 3, 64, 64), np.float32), str(model), 'TENSORRT', 1)
    assert [x[0] for x in paths[-4:]] == ['TENSORRT', 'TORCH_CUDA', 'CUDA', 'DIRECTML']
    runtime.start_inference_job()


def test_cpu_preference_overrides_external_gpu_plan(monkeypatch):
    from fusion_methods import stackmff_v1, stackmff_v2, stackmffv3
    frames = [np.zeros((8, 8), np.uint8)] * 2
    monkeypatch.setattr(runtime, '_PACK', None)
    monkeypatch.setattr(runtime, 'refresh_accelerator_pack', lambda: pytest.fail('CPU requested a pack'))
    monkeypatch.setattr(runtime, 'resolve_ort_device', lambda: pytest.fail('CPU requested a GPU'))
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *_: resource_policy.ResourceSnapshot(
        available_ram=8 * 1024**3, total_ram=16 * 1024**3))
    plan = resource_policy.ExecutionPlan(device_kind='TENSORRT', ram_budget=1024**3)
    with runtime.inference_job():
        for adapter in (stackmff_v1, stackmff_v2, stackmffv3):
            assert adapter._prepare_inference(frames, 'chosen.onnx', False, execution_plan=plan)[1] == 'CPU'


def test_lost_worker_retries_ort_without_restarting_until_next_job(monkeypatch, tmp_path):
    from core import inference_pack
    model = tmp_path / 'selected.onnx'
    model.write_bytes(b'graph')
    starts = []
    class BrokenWorker:
        def __init__(self, *_):
            starts.append(1)
            raise inference_client.WorkerLost('worker startup timed out')

    monkeypatch.setattr(inference_client, 'Worker', BrokenWorker)
    monkeypatch.setattr(inference_pack, 'pack_root', lambda: tmp_path)
    monkeypatch.setattr(runtime, '_PACK', (('TENSORRT', 'TORCH_CUDA'), ('isolated',), 'test GPU'))
    monkeypatch.setattr(runtime, '_DEVICE', 'DIRECTML')
    monkeypatch.setattr(runtime, '_AVAILABLE_PROVIDERS', ('DmlExecutionProvider',))
    monkeypatch.setattr(runtime, '_memory_refusal', lambda *a, **k: None)
    monkeypatch.setattr(resource_policy, 'require_memory', lambda *_: None)
    monkeypatch.setattr(runtime, '_run_ort', lambda planes, _p, device, *_: (planes, device))
    for job in (1, 2):
        with runtime.inference_job():
            for height in (64, 128):
                inputs = np.zeros((1, 2, height, 64), np.float32)
                runtime.logits(inputs, model, 'TENSORRT', 1)
            assert runtime.actual_device() == 'DIRECTML'
            assert len(starts) == job
            assert any('startup timed out' in reason for reason in runtime.device_diagnostics()['fallback_reasons'])
    with runtime.inference_job(), pytest.raises(FileNotFoundError):
        runtime.logits(inputs, tmp_path / 'missing.onnx', 'TENSORRT', 1)
    assert len(starts) == 2
    runtime.start_inference_job()


def test_native_admission_charges_tensorrt_workspace_and_plan_limit(monkeypatch):
    gib = 1024 ** 3
    monkeypatch.setattr(runtime, '_vram_snapshot', lambda _: resource_policy.ResourceSnapshot(
        free_vram=8 * gib, total_vram=8 * gib))
    with runtime.inference_job():
        shape = (29, 512, 512)
        # Leave only 256 MiB of budget for the builder above the dense estimate.
        base = runtime.estimate_inference_bytes(*shape, 'CUDA') + 6 * 4 * np.prod(shape)
        plan = resource_policy.ExecutionPlan(ram_budget=8 * gib, vram_budget=base + 256 * 1024**2)
        runtime.apply_inference_budget(plan)
        assert runtime._memory_refusal('model', shape, 'TENSORRT', 1)
        assert runtime._memory_refusal('model', shape, 'TORCH_CUDA', 1) is None
    runtime.start_inference_job()


def test_cuda_reuses_admitted_worker_for_full_and_reduced_tiles(monkeypatch, tmp_path):
    from core import inference_pack

    model = str(tmp_path / 'verified.onnx')
    Path(model).touch()
    alternate = str(tmp_path / 'alternate.onnx')
    Path(alternate).touch()
    device = 'TORCH_CUDA'
    probes, worker_events = [], []
    available = 4 * 1024**3

    def snapshot(kind):
        probes.append(kind)
        return resource_policy.ResourceSnapshot(free_vram=available, total_vram=4 * 1024**3)

    class Worker:
        backends = (device,)

        def __init__(self, *_):
            worker_events.append('start')

        def run(self, _device, _path, planes, *_):
            worker_events.append(planes.shape[1:])
            return planes.copy()

        def close(self):
            worker_events.append('close')

    monkeypatch.setattr(runtime, '_PACK', ((device,), ('installed',), 'test GPU'))
    monkeypatch.setattr(runtime, '_vram_snapshot', snapshot)
    monkeypatch.setattr(inference_client, 'Worker', Worker)
    monkeypatch.setattr(inference_pack, 'pack_root', lambda: tmp_path)
    monkeypatch.setattr(resource_policy, 'require_memory', lambda *_: None)

    full = np.zeros((1, 4, 128, 128), np.float32)
    reduced = np.zeros((1, 4, 64, 64), np.float32)
    with runtime.inference_job():
        runtime.logits(full, model, device, 1)
        available = 1024  # Worker-held space is unavailable to a fresh process.
        runtime.logits(reduced, model, device, 1)
        runtime.logits(full, model, device, 1)
        assert probes == [device, device]  # Cold and post-startup admission.
        assert worker_events == ['start', full.shape[1:], reduced.shape[1:], full.shape[1:]]
        for path, shape in [(alternate, full.shape[1:]), (model, (4, 192, 128))]:
            assert runtime._memory_refusal(path, shape, device, 1, reclaim=False)
        runtime.apply_inference_budget(resource_policy.ExecutionPlan(vram_budget=1, device_kind=device))
        assert 'GPU memory plan budget' in runtime._memory_refusal(model, full.shape[1:], device, 1)
        runtime.apply_inference_budget(resource_policy.ExecutionPlan(device_kind=device))
    assert worker_events[-1] == 'close'
    assert 'GPU memory budget' in runtime._memory_refusal(model, full.shape[1:], device, 1)
    runtime.start_inference_job()


def test_mps_job_reports_execution_and_tries_ort_before_cpu(worker_command, monkeypatch, tmp_path):
    from core import inference_pack

    model = tmp_path / 'model.onnx'
    model.touch()
    monkeypatch.setattr(runtime, '_PACK', (('TORCH_MPS',), tuple(worker_command), 'Apple GPU'))
    monkeypatch.setattr(runtime, '_DEVICE', 'COREML')
    monkeypatch.setattr(runtime, '_AVAILABLE_PROVIDERS', ('CoreMLExecutionProvider',))
    monkeypatch.setattr(runtime, '_vram_snapshot', lambda _: pytest.fail('MPS queried VRAM'))
    monkeypatch.setattr(inference_pack, 'pack_root', lambda: tmp_path)
    monkeypatch.setattr(resource_policy, 'probe_memory', lambda: resource_policy.ResourceSnapshot(
        available_ram=8 * 1024**3, total_ram=16 * 1024**3))
    planes = np.zeros((1, 3, 64, 64), np.float32)
    with runtime.inference_job():
        result = runtime.logits(planes, model, runtime.resolve_device(True), 1)
        np.testing.assert_array_equal(result, planes + 2)
        assert runtime.actual_device() == 'TORCH_MPS'
        assert runtime.actual_device_report() == 'PyTorch MPS (Apple GPU)'

    calls = []

    def reject_model(*_):
        calls.append('TORCH_MPS')
        raise inference_client.WorkerError('unsupported MPS operator')

    def run_ort(planes, _path, device, *_):
        calls.append(device)
        if device == 'COREML':
            raise RuntimeError('graph is unsupported by CoreML')
        return planes, device

    monkeypatch.setattr(runtime, '_run_native', reject_model)
    monkeypatch.setattr(runtime, '_run_ort', run_ort)
    monkeypatch.setattr(resource_policy, 'require_memory', lambda *_: None)
    with runtime.inference_job():
        runtime.logits(planes, model, 'TORCH_MPS', 1)
        assert calls == ['TORCH_MPS', 'COREML', 'CPU']
        assert runtime.actual_device() == 'CPU'
        assert any('unsupported MPS operator' in s for s in runtime.device_diagnostics()['fallback_reasons'])
    runtime.start_inference_job()


@pytest.mark.parametrize('graph', ['stackmff_final', 'stackmff_v2_final', 'stackmff_v3_final'])
def test_native_worker_runs_deep_released_checkpoints(tmp_path, graph):
    from core.inference_worker import ROOT

    python = os.environ.get('OPENFOCUS_TEST_NATIVE_WORKER_PYTHON')
    if not python:
        pytest.skip('Set OPENFOCUS_TEST_NATIVE_WORKER_PYTHON to the GPU pack interpreter')
    worker = inference_client.Worker([python, '-I', '-u', str(ROOT / 'core' / 'inference_worker.py')], None)
    try:
        backend, = (kind for kind in worker.backends if kind in ('TORCH_CUDA', 'TORCH_MPS'))
        assert worker.device
        model = str(ROOT / 'weights' / f'{graph}.onnx')
        planes = np.random.default_rng(217).random((1, 500, 128, 128), dtype=np.float32)
        result = worker.run(backend, model, planes, tmp_path, None)
        assert result.shape == planes.shape and result.dtype == np.float32
        assert np.isfinite(result).all()
    finally:
        worker.close()
