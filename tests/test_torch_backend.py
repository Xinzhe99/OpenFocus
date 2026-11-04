# -*- coding: utf-8 -*-
"""Backend decisions without a GPU or a real torch installation.

Every hardware-dependent test supplies an explicit fake module (or isolates
``loaded_torch``), so the suite gives the same result on a CPU-only machine
and on a machine with a working GPU.
"""
import builtins
import sys
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


def test_cuda_and_rocm_are_distinguished():
    assert backend.detect_kind(_torch(cuda=FakeCuda())) == backend.KIND_CUDA
    assert backend.detect_kind(_torch(cuda=FakeCuda(), hip="6.2")) == (
        backend.KIND_ROCM)


def test_xpu_is_selected_when_cuda_is_unavailable():
    module = _torch(cuda=FakeCuda(available=False), xpu=FakeXpu())
    assert backend.detect_kind(module) == backend.KIND_XPU
    assert backend.resolve_device(True, module).type == "xpu"


@pytest.mark.parametrize("exit_code,expected", [
    (0, backend.KIND_XPU),
    (7, backend.KIND_CPU),
])
def test_windows_xpu_probe_survives_native_process_failure(monkeypatch, exit_code, expected):
    class UnsafeXpu(FakeXpu):
        def is_available(self):
            pytest.fail("XPU availability must not run in the desktop process")

    module = _torch(cuda=FakeCuda(available=False), xpu=UnsafeXpu())
    module.version.xpu = "test"
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


@pytest.mark.parametrize("fallback,expected", [
    ("xpu", backend.KIND_XPU),
    ("mps", backend.KIND_MPS),
])
def test_a_broken_cuda_driver_does_not_mask_another_backend(fallback, expected):
    kwargs = {"cuda": BrokenCuda()}
    if fallback == "xpu":
        kwargs["xpu"] = FakeXpu()
    else:
        kwargs["mps"] = FakeMps()
    assert backend.detect_kind(_torch(**kwargs)) == expected


def test_explicit_cpu_request_wins_over_an_accelerator():
    module = _torch(cuda=FakeCuda(), xpu=FakeXpu())
    assert backend.resolve_device(False, module).type == "cpu"


def test_rocm_and_xpu_memory_readings():
    rocm = _torch(cuda=FakeCuda(memory=(3 * 1024 ** 3, 24 * 1024 ** 3)),
                  hip="6.2")
    assert backend.device_memory(backend.KIND_ROCM, rocm) == (
        3 * 1024 ** 3, 24 * 1024 ** 3)
    xpu = _torch(xpu=FakeXpu(memory=(2 * 1024 ** 3, 16 * 1024 ** 3)))
    assert backend.device_memory(backend.KIND_XPU, xpu) == (
        2 * 1024 ** 3, 16 * 1024 ** 3)


def test_unknown_memory_readings_stay_unknown(monkeypatch):
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


def test_ensure_torch_never_probes_a_half_initialized_module(monkeypatch):
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


@pytest.mark.parametrize("hardware,kind,use_gpu,torch_installed,enabled,checked", [
    (False, "XPU", False, False, True, False),    # found XPU, user turned off
    (False, None, True, True, True, False),       # unresolved, probe pending
    (False, "N/A", True, True, False, False),     # resolved: no accelerator
    (True, "N/A", False, False, True, False),     # hardware present, user off
    (True, "ROCM", True, False, True, True),      # hardware + backend, on
    (False, None, True, False, False, False),     # no hardware, no torch
])
def test_gpu_toggle_state_preserves_the_users_choice(
        hardware, kind, use_gpu, torch_installed, enabled, checked):
    assert backend.gpu_toggle_state(
        hardware, kind, use_gpu, torch_installed) == (enabled, checked)


@pytest.mark.parametrize("vendors,tools,expected", [
    ({"0x1002"}, False, True),      # AMD ROCm capable stack
    ({"0x10de"}, False, True),      # NVIDIA on Linux
    ({"0x8086"}, False, True),      # Intel XPU without xpu-smi/sycl-ls
    (set(), True, True),            # ROCm tooling without a visible DRM card
    (set(), False, False),
])
def test_linux_gpu_status_covers_amd_nvidia_and_intel(
        monkeypatch, vendors, tools, expected):
    from utils import platform_utils
    monkeypatch.setattr(platform_utils, "is_macos", lambda: False)
    monkeypatch.setattr(platform_utils, "is_windows", lambda: False)
    monkeypatch.setattr(platform_utils, "is_linux", lambda: True)
    monkeypatch.setattr(platform_utils, "_torch_accelerator_present",
                        lambda: False)
    monkeypatch.setattr(platform_utils, "_nvidia_smi_reports_gpu",
                        lambda: False)
    monkeypatch.setattr(platform_utils.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(platform_utils, "_linux_gpu_vendors",
                        lambda: frozenset(vendors))
    monkeypatch.setattr(platform_utils, "_vendor_tool_present",
                        lambda *_names: tools)
    platform_utils.reset_gpu_probe_cache()
    try:
        assert platform_utils.has_compute_gpu() is expected
    finally:
        platform_utils.reset_gpu_probe_cache()


def test_loaded_xpu_backend_makes_the_gpu_probe_true(monkeypatch):
    from utils import platform_utils
    monkeypatch.setitem(sys.modules, "torch",
                        _torch(cuda=FakeCuda(available=False), xpu=FakeXpu()))
    backend.reset_cache()
    platform_utils.reset_gpu_probe_cache()
    try:
        assert platform_utils.has_compute_gpu() is True
    finally:
        platform_utils.reset_gpu_probe_cache()
        backend.reset_cache()


@pytest.mark.parametrize("kind,label", [
    ("ROCM", "ROCm"), ("XPU", "XPU"), ("MPS", "MPS"), ("CUDA", "CUDA"),
])
def test_fusion_info_reports_the_backend_label(monkeypatch, kind, label):
    import core.multi_focus_fusion as mff
    monkeypatch.setattr(mff, "is_ai_fusion_available", lambda: True)
    monkeypatch.setattr(mff.torch_backend, "probe_kind",
                        lambda import_if_needed=False: kind)
    fusion = mff.MultiFocusFusion("stackmffv3", use_gpu=True)
    assert fusion.get_info()["device"] == label
