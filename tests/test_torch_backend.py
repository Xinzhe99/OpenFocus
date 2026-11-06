# -*- coding: utf-8 -*-
"""Backend decisions without a GPU or a real torch installation.

Every hardware-dependent test supplies an explicit fake module (or isolates
``loaded_torch``), so the suite gives the same result on a CPU-only machine
and on a machine with a working GPU.
"""
import builtins
import subprocess
import sys
import threading
import types

import pytest

from core import torch_backend as backend


class FakeDevice:
    def __init__(self, name="cpu"):
        self.type = name.split(":")[0]

    def __str__(self):
        return self.type


class FakeCuda:
    def __init__(self, available=True, memory=(4 * 1024 ** 3, 8 * 1024 ** 3),
                 error=False):
        self.available = available
        self.memory = memory
        self.error = error

    def is_available(self):
        return self.available

    def mem_get_info(self):
        if self.error:
            raise RuntimeError("driver query failed")
        return self.memory


class BrokenCuda:
    def is_available(self):
        raise RuntimeError("CUDA driver crashed")


class FakeXpu(FakeCuda):
    def __init__(self, available=True, memory=(2 * 1024 ** 3, 16 * 1024 ** 3)):
        super().__init__(available=available, memory=memory)


class FakeMps:
    def __init__(self, memory=(2 * 1024 ** 3, 6 * 1024 ** 3), error=False):
        self.memory = memory
        self.error = error

    def is_available(self):
        return True

    def recommended_max_memory(self):
        if self.error:
            raise RuntimeError("no MPS memory query")
        return self.memory[1]

    def driver_allocated_memory(self):
        if self.error:
            raise RuntimeError("no MPS memory query")
        return self.memory[1] - self.memory[0]


class FakeBackends:
    def __init__(self, mps=None):
        self.mps = mps


class FakeTorch:
    def __init__(self, cuda=None, xpu=None, mps=None, hip=None):
        self.cuda = cuda
        self.xpu = xpu
        self.mps = mps
        self.backends = FakeBackends(mps)
        self.version = types.SimpleNamespace(hip=hip, cuda=None)
        self.Tensor = object

    def device(self, name):
        return FakeDevice(name)


def _torch(cuda=None, xpu=None, mps=None, hip=None):
    return FakeTorch(cuda=cuda, xpu=xpu, mps=mps, hip=hip)


@pytest.fixture
def xpu_probe_usable(monkeypatch):
    backend.reset_cache()
    monkeypatch.setattr(backend, "_xpu_probe_command",
                        lambda: [sys.executable, "-c", "pass"])
    yield
    backend.reset_cache()


@pytest.fixture
def rocm_probe_usable(monkeypatch):
    backend.reset_cache()
    monkeypatch.setattr(backend, "_rocm_probe_command",
                        lambda: [sys.executable, "-c", "pass"])
    yield
    backend.reset_cache()


def test_cuda_and_rocm_are_distinguished(rocm_probe_usable):
    assert backend.detect_kind(_torch(cuda=FakeCuda())) == backend.KIND_CUDA
    assert backend.detect_kind(_torch(cuda=FakeCuda(), hip="6.2")) == (
        backend.KIND_ROCM)


@pytest.mark.parametrize("exit_code,expected", [
    (0, backend.KIND_ROCM),
    (7, backend.KIND_CPU),
])
def test_rocm_probe_survives_child_exit_and_is_cached(monkeypatch, exit_code, expected):
    class UnsafeCuda(FakeCuda):
        def is_available(self):
            pytest.fail("ROCm availability must not run in the desktop process")

    module = _torch(cuda=UnsafeCuda(), hip="6.2")
    monkeypatch.setattr(backend, "_rocm_probe_command", lambda: [
        sys.executable, "-c", f"import os; os._exit({exit_code})",
    ])
    backend.reset_cache()
    try:
        assert backend.detect_kind(module) == expected
        monkeypatch.setattr(backend, "_rocm_probe_command",
                            lambda: pytest.fail("probe should be cached"))
        assert backend.resolve_device(True, module).type == (
            "cuda" if expected == backend.KIND_ROCM else "cpu")
        assert backend.device_memory(backend.KIND_ROCM, module) == (
            module.cuda.memory if expected == backend.KIND_ROCM else None)
    finally:
        backend.reset_cache()


def test_hanging_rocm_probe_falls_back_to_cpu(monkeypatch, caplog):
    class UnsafeCuda(FakeCuda):
        def is_available(self):
            pytest.fail("ROCm availability must not run in the desktop process")

    module = _torch(cuda=UnsafeCuda(), hip="6.2")
    monkeypatch.setattr(backend, "_ROCM_PROBE_TIMEOUT", 0.5)
    monkeypatch.setattr(backend, "_rocm_probe_command", lambda: [
        sys.executable, "-c", "import time; time.sleep(30)",
    ])
    backend.reset_cache()
    try:
        assert backend.detect_kind(module) == backend.KIND_CPU
        assert backend.resolve_device(True, module).type == "cpu"
        assert "Isolated ROCm probe failed" in caplog.text
    finally:
        backend.reset_cache()


def test_failed_rocm_probe_preserves_xpu_fallback(monkeypatch, xpu_probe_usable):
    monkeypatch.setattr(backend, "_rocm_probe_command", lambda: [
        sys.executable, "-c", "import os; os._exit(7)",
    ])
    module = _torch(cuda=BrokenCuda(), hip="6.2", xpu=FakeXpu())
    assert backend.detect_kind(module) == backend.KIND_XPU


def test_nvidia_cuda_does_not_spawn_rocm_probe(monkeypatch):
    monkeypatch.setattr(backend, "_rocm_probe_command",
                        lambda: pytest.fail("NVIDIA must not spawn a ROCm probe"))
    assert backend.detect_kind(_torch(cuda=FakeCuda())) == backend.KIND_CUDA
    assert backend.detect_kind(_torch(cuda=FakeCuda(available=False))) == backend.KIND_CPU


def test_xpu_is_selected_when_cuda_is_unavailable(xpu_probe_usable):
    module = _torch(cuda=FakeCuda(available=False), xpu=FakeXpu())
    assert backend.detect_kind(module) == backend.KIND_XPU
    assert backend.resolve_device(True, module).type == "xpu"


@pytest.mark.parametrize("xpu_version", ["test", None])
@pytest.mark.parametrize("exit_code,expected", [
    (0, backend.KIND_XPU),
    (7, backend.KIND_CPU),
])
def test_windows_xpu_probe_survives_native_process_failure(monkeypatch, exit_code, expected, xpu_version):
    class UnsafeXpu(FakeXpu):
        def is_available(self):
            pytest.fail("XPU availability must not run in the desktop process")

    module = _torch(cuda=FakeCuda(available=False), xpu=UnsafeXpu())
    module.version.xpu = xpu_version
    monkeypatch.setattr(backend.sys, "platform", "win32")
    monkeypatch.setattr(backend, "_xpu_probe_command", lambda: [
        sys.executable, "-c", f"import os; os._exit({exit_code})",
    ])
    backend.reset_cache()
    try:
        assert backend.detect_kind(module) == expected
        assert backend.resolve_device(True, module).type == expected.lower()
        memory = backend.device_memory(backend.KIND_XPU, module)
        assert memory == ((2 * 1024 ** 3, 16 * 1024 ** 3)
                          if expected == backend.KIND_XPU else None)
    finally:
        backend.reset_cache()


def test_hanging_xpu_probe_falls_back_to_cpu(monkeypatch, caplog):
    module = _torch(cuda=FakeCuda(available=False), xpu=FakeXpu())
    module.version.xpu = "test"
    monkeypatch.setattr(backend, "_XPU_PROBE_TIMEOUT", 0.5)
    monkeypatch.setattr(backend, "_xpu_probe_command", lambda: [
        sys.executable, "-c", "import time; time.sleep(30)",
    ])
    backend.reset_cache()
    try:
        assert backend.detect_kind(module) == backend.KIND_CPU
        assert backend.resolve_device(True, module).type == "cpu"
        assert "Isolated XPU probe failed" in caplog.text
    finally:
        backend.reset_cache()


def test_source_xpu_probe_uses_parent_import_paths(tmp_path, monkeypatch):
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "torch.py").write_text(
        "class xpu:\n    @staticmethod\n    def is_available(): return False\n",
        encoding="utf-8",
    )
    (tmp_path / "torch.py").write_text(
        "raise RuntimeError('wrong working-directory torch')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(installed))
    result = subprocess.run(backend._xpu_probe_command(), cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stderr


@pytest.mark.parametrize("fallback,expected", [
    ("xpu", backend.KIND_XPU),
    ("mps", backend.KIND_MPS),
])
def test_a_broken_cuda_driver_does_not_mask_another_backend(fallback, expected, xpu_probe_usable):
    kwargs = {"cuda": BrokenCuda()}
    if fallback == "xpu":
        kwargs["xpu"] = FakeXpu()
    else:
        kwargs["mps"] = FakeMps()
    assert backend.detect_kind(_torch(**kwargs)) == expected


def test_explicit_cpu_request_wins_over_an_accelerator():
    module = _torch(cuda=FakeCuda(), xpu=FakeXpu())
    assert backend.resolve_device(False, module).type == "cpu"


def test_rocm_and_xpu_memory_readings(xpu_probe_usable, rocm_probe_usable):
    rocm = _torch(cuda=FakeCuda(memory=(3 * 1024 ** 3, 24 * 1024 ** 3)),
                  hip="6.2")
    assert backend.device_memory(backend.KIND_ROCM, rocm) == (
        3 * 1024 ** 3, 24 * 1024 ** 3)
    xpu = _torch(xpu=FakeXpu(memory=(2 * 1024 ** 3, 16 * 1024 ** 3)))
    assert backend.device_memory(backend.KIND_XPU, xpu) == (
        2 * 1024 ** 3, 16 * 1024 ** 3)


def test_unknown_memory_readings_stay_unknown(monkeypatch, xpu_probe_usable):
    assert backend.device_memory(
        backend.KIND_CUDA, _torch(cuda=FakeCuda(error=True))) is None
    assert backend.device_memory(
        backend.KIND_CUDA,
        _torch(cuda=FakeCuda(memory=(9 * 1024 ** 3, 8 * 1024 ** 3)))) is None

    class XpuWithoutMemoryQuery:
        def is_available(self):
            return True

    assert backend.device_memory(
        backend.KIND_XPU, _torch(xpu=XpuWithoutMemoryQuery())) is None

    # Explicitly isolate the "torch is not loaded" case: passing None must not
    # read whatever torch happens to be in sys.modules on this machine.
    monkeypatch.setattr(backend, "loaded_torch", lambda: None)
    assert backend.device_memory(backend.KIND_CUDA, None) is None


def test_mps_memory_reading_and_unknown_query():
    module = _torch(mps=FakeMps(memory=(2 * 1024 ** 3, 6 * 1024 ** 3)))
    assert backend.mps_memory(module) == (2 * 1024 ** 3, 6 * 1024 ** 3)
    assert backend.mps_memory(_torch(mps=FakeMps(error=True))) is None


def test_kind_of_device_distinguishes_rocm_from_cuda():
    device = FakeDevice("cuda")
    assert backend.kind_of_device(device, _torch(hip=None)) == backend.KIND_CUDA
    assert backend.kind_of_device(device, _torch(hip="6.2")) == backend.KIND_ROCM
    assert backend.kind_of_device(FakeDevice("xpu"), _torch()) == backend.KIND_XPU


def test_ensure_torch_never_probes_a_half_initialized_module(monkeypatch, xpu_probe_usable):
    partial = types.SimpleNamespace(cuda=None, xpu=None, backends=None,
                                    version=None)
    full = _torch(cuda=BrokenCuda(), xpu=FakeXpu())
    monkeypatch.setitem(sys.modules, "torch", partial)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return full
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    backend.reset_cache()
    try:
        assert backend.ensure_torch() is full
        assert backend.probe_kind() == backend.KIND_XPU
    finally:
        backend.reset_cache()


@pytest.mark.parametrize("kind,use_gpu,torch_installed,enabled,checked", [
    ("XPU", False, False, True, False),
    (None, True, True, True, True),
    ("N/A", True, True, False, False),
    ("N/A", False, False, False, False),
    ("ROCM", True, False, True, True),
    (None, True, False, False, False),
])
def test_gpu_toggle_state_preserves_the_users_choice(
        kind, use_gpu, torch_installed, enabled, checked):
    assert backend.gpu_toggle_state(kind, use_gpu, torch_installed) == (enabled, checked)


@pytest.mark.parametrize('kind,expected', [(None, None), ('DIRECTML', True), ('N/A', False)])
def test_cached_ai_capability_never_probes_hardware_or_runtime(monkeypatch, kind, expected):
    from core import multi_focus_fusion
    from utils import platform_utils

    monkeypatch.setattr(multi_focus_fusion, 'neural_device_kind', lambda: kind)
    def unexpected_probe():
        raise AssertionError('cached capability must not start inference')
    monkeypatch.setattr(multi_focus_fusion, 'ensure_neural_device_kind', unexpected_probe)
    assert platform_utils.cached_compute_gpu() is expected


@pytest.mark.parametrize('kind,expected', [('DIRECTML', True), ('N/A', False)])
def test_synchronous_capability_uses_runtime_probe_when_unknown(monkeypatch, kind, expected):
    from core import multi_focus_fusion
    from utils import platform_utils

    calls = []
    monkeypatch.setattr(multi_focus_fusion, 'neural_device_kind', lambda: None)
    monkeypatch.setattr(multi_focus_fusion, 'ensure_neural_device_kind',
                        lambda: calls.append('inference') or kind)
    assert platform_utils.has_compute_gpu() is expected
    assert calls == ['inference']


@pytest.mark.parametrize("kind,expected", [
    ("DIRECTML", True), ("CUDA", True), ("ROCM", True),
    ("OPENVINO", True), ("COREML", True), ("CPU", False),
])
def test_completed_neural_probe_is_authoritative(monkeypatch, kind, expected):
    from fusion_methods import neural_runtime
    from utils import platform_utils

    monkeypatch.setattr(neural_runtime, "_DEVICE", kind)
    assert platform_utils.has_compute_gpu() is expected


def test_cached_capability_does_not_wait_for_an_active_device_probe(monkeypatch):
    from core import multi_focus_fusion
    from fusion_methods import neural_runtime
    from utils import platform_utils

    entered = threading.Event()
    release = threading.Event()
    result = []
    status = []
    status_ready = threading.Event()

    def blocked_options():
        entered.set()
        if not release.wait(5):
            raise RuntimeError("probe was not released")
        raise RuntimeError("provider unavailable")

    ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider"],
        SessionOptions=blocked_options,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setattr(neural_runtime, "_DEVICE", None)
    monkeypatch.setattr(multi_focus_fusion, "_NEURAL_DEVICE_KIND", None)
    thread = threading.Thread(target=lambda: result.append(neural_runtime.resolve_device(True)))
    query = threading.Thread(target=lambda: (status.append(platform_utils.cached_compute_gpu()),
                                             status_ready.set()))
    thread.start()
    try:
        assert entered.wait(5), "device probe did not start"
        query.start()
        assert status_ready.wait(2), "GPU status waited for the device probe"
        assert status == [None]
    finally:
        release.set()
        thread.join(timeout=5)
        if query.ident is not None:
            query.join(timeout=5)
    assert not thread.is_alive()
    assert result == ["CPU"]
    assert platform_utils.cached_compute_gpu() is False


@pytest.mark.parametrize("kind,label", [
    ("DIRECTML", "DirectML"), ("CUDA", "CUDA"), ("ROCM", "ROCm"),
    ("OPENVINO", "OpenVINO GPU"), ("COREML", "CoreML"), ("CPU", "CPU"),
])
def test_fusion_info_reports_the_backend_label(monkeypatch, kind, label):
    import core.multi_focus_fusion as mff
    from fusion_methods import neural_runtime

    monkeypatch.setattr(mff, "is_ai_fusion_available", lambda: True)
    monkeypatch.setattr(neural_runtime, "_DIRECTML_ADAPTER_NAME", "")
    monkeypatch.setattr(neural_runtime, '_DEVICE', kind)
    monkeypatch.setattr(neural_runtime, "resolve_device", lambda use_gpu: kind)
    fusion = mff.MultiFocusFusion("stackmffv3", use_gpu=True)
    assert fusion.get_info()["device"] == label
