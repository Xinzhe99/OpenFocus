# -*- coding: utf-8 -*-
"""Admission for the neural adapters on CUDA/ROCm/XPU/MPS/CPU.

Uses a fake torch module so the tests pin the decision and the budget gate
without requiring a GPU, or even torch, to be installed.
"""
import types

import numpy as np
import pytest

from core import resource_policy, torch_backend
from fusion_methods import neural_common
from fusion_methods import stackmffv3 as v3

GiB = 1024 ** 3


class FakeDevice:
    def __init__(self, name):
        self.type = name.split(":")[0]

    def __str__(self):
        return self.type


class FakeCuda:
    def __init__(self, available=True):
        self.available = available

    def is_available(self):
        return self.available


class FakeXpu(FakeCuda):
    pass


class FakeMps:
    def is_available(self):
        return True

    def empty_cache(self):
        pass


class FakeBackends:
    def __init__(self, mps=None):
        self.mps = mps


class FakeTorch:
    def __init__(self, kind="CPU", hip=None):
        self.version = types.SimpleNamespace(hip=hip, cuda=None)
        self.cuda = FakeCuda(available=kind in ("CUDA", "ROCM"))
        self.xpu = FakeXpu(available=kind == "XPU")
        self.mps = FakeMps() if kind == "MPS" else None
        self.backends = FakeBackends(FakeMps() if kind == "MPS" else None)

    def device(self, name):
        return FakeDevice(name)

    def set_num_threads(self, count):
        pass


def _frames(size=(8, 8)):
    return [np.zeros((*size, 3), np.uint8), np.ones((*size, 3), np.uint8)]


def _model_file(tmp_path):
    path = tmp_path / "model.pth"
    path.write_bytes(b"weights" * 256)
    return str(path)


def _snapshot(kind, free_vram=0, total_vram=0):
    return resource_policy.ResourceSnapshot(
        available_ram=8 * GiB, total_ram=16 * GiB,
        physical_cores=8, logical_cpus=8,
        device_kind=kind, free_vram=free_vram, total_vram=total_vram)


@pytest.mark.parametrize("algorithm,kind", [
    ("stackmff", "ROCM"),
    ("stackmffv2", "XPU"),
])
def test_shared_plan_accounts_for_rocm_and_xpu_device_memory(
        monkeypatch, tmp_path, algorithm, kind):
    fake = FakeTorch(kind=kind, hip="6.2" if kind == "ROCM" else None)
    monkeypatch.setattr(torch_backend, "_XPU_AVAILABLE", kind == "XPU")
    monkeypatch.setattr(neural_common, "torch", fake)
    monkeypatch.setattr(resource_policy, "probe_resources",
                        lambda _kind: _snapshot(kind, 6 * GiB, 8 * GiB))
    device, plan = neural_common.plan_inference(
        algorithm, _frames(), _model_file(tmp_path), True, 2, "balanced",
        None, workspace_bytes=GiB, host_bytes=1024)
    assert device.type == ("cuda" if kind == "ROCM" else "xpu")
    assert plan.device_kind == kind
    assert plan.vram_budget > 0
    assert plan.native_threads == 2


@pytest.mark.parametrize("algorithm", ["stackmff", "stackmffv2"])
def test_unknown_device_memory_refuses_instead_of_running_unbounded(
        monkeypatch, tmp_path, algorithm):
    fake = FakeTorch(kind="XPU")
    monkeypatch.setattr(torch_backend, "_XPU_AVAILABLE", True)
    monkeypatch.setattr(neural_common, "torch", fake)
    monkeypatch.setattr(resource_policy, "probe_resources",
                        lambda _kind: _snapshot("XPU"))
    with pytest.raises(resource_policy.ResourceLimitError,
                       match="Insufficient or unknown"):
        neural_common.plan_inference(
            algorithm, _frames(), _model_file(tmp_path), True, 2, "balanced",
            None, workspace_bytes=GiB, host_bytes=1024)


def test_mps_budget_uses_shared_memory_and_refuses_when_unknown(
        monkeypatch, tmp_path):
    fake = FakeTorch(kind="MPS")
    monkeypatch.setattr(torch_backend, "_XPU_AVAILABLE", False)
    monkeypatch.setattr(neural_common, "torch", fake)
    monkeypatch.setattr(resource_policy, "probe_resources",
                        lambda _kind: _snapshot("MPS"))
    monkeypatch.setattr(torch_backend, "mps_memory", lambda _module: None)
    with pytest.raises(resource_policy.ResourceLimitError,
                       match="Cannot determine the MPS"):
        neural_common.plan_inference(
            "stackmffv2", _frames(), _model_file(tmp_path), True, 25,
            "balanced", None, workspace_bytes=GiB, host_bytes=1024)


class FakeModel:
    C1 = 48
    C2 = 96
    single_scale = True
    window = 8

    def modules(self):
        return iter(())


def test_v3_counts_inference_workspace_as_xpu_memory(monkeypatch, tmp_path):
    fake = FakeTorch(kind="XPU")
    device = FakeDevice("xpu")
    monkeypatch.setattr(v3, "torch", fake)
    monkeypatch.setattr(v3, "_resolve_device", lambda _use_gpu: device)
    monkeypatch.setattr(v3, "_get_model_and_device",
                        lambda *_args: (FakeModel(), device))
    monkeypatch.setattr(resource_policy, "probe_resources",
                        lambda _kind: _snapshot("XPU", 6 * GiB, 8 * GiB))
    model, got_device, plan = v3._prepare_inference(
        _frames((32, 32)), _model_file(tmp_path), True, thread_count=2)
    assert got_device.type == "xpu"
    assert plan.device_kind == "XPU"
    assert plan.vram_budget > 0
    assert plan.native_threads == 2


def test_v3_refuses_an_unreadable_mps_budget_before_loading(monkeypatch, tmp_path):
    fake = FakeTorch(kind="MPS")
    device = FakeDevice("mps")
    monkeypatch.setattr(v3, "torch", fake)
    monkeypatch.setattr(v3, "_resolve_device", lambda _use_gpu: device)
    monkeypatch.setattr(resource_policy, "probe_resources",
                        lambda _kind: _snapshot("MPS"))
    monkeypatch.setattr(torch_backend, "mps_memory", lambda _module: None)
    monkeypatch.setattr(v3, "_get_model_and_device",
                        lambda *_args: pytest.fail("model loaded without a budget"))
    with pytest.raises(resource_policy.ResourceLimitError,
                       match="Cannot determine the MPS"):
        v3._prepare_inference(_frames((32, 32)), _model_file(tmp_path), True,
                              thread_count=2)
