import ctypes
from dataclasses import replace
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import core._ecc_area_gpu as ecc_gpu
import core._perspective_gpu as persp_gpu
import core.resource_policy as resource_policy
import fusion_methods._openfocus_gpu as ogpu
import fusion_methods._openfocus_gpu_a as ogpu_a
import fusion_methods._openfocus_gpu_c as ogpu_c
import fusion_methods._openfocus_gpu_dct as ogpu_dct
import fusion_methods._openfocus_gpu_dtcwt as ogpu_dtcwt
import fusion_methods._openfocus_gpu_gfg as ogpu_gfg
import fusion_methods._openfocus_gpu_gff as ogpu_gff
from utils import memory as memory_utils
from utils import platform_utils


def _ample_snapshot():
    real = resource_policy.probe_memory()
    return replace(
        real,
        total_ram=max(16 * 1024**3, real.total_ram),
        available_ram=8 * 1024**3,
        commit_headroom=8 * 1024**3,
        process_commit_headroom=None,
    )


def test_macos_memory_uses_c_size_t_page_size_and_stat64_fallback(monkeypatch):
    class FakeFn:
        def __init__(self, fn):
            self._fn = fn
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self._fn(*args)

    total_bytes = 16 * 1024**3
    page_bytes = 16384
    state = {"use_stat64": True, "free_pages": 2000, "inactive_pages": 3000}

    def fake_sysctl(name, oldp, oldlenp, _newp, _newlen):
        if name == b"hw.memsize" and oldp is not None:
            ctypes.memmove(oldp, ctypes.byref(ctypes.c_uint64(total_bytes)), 8)
            return 0
        return -1

    def fake_host_page_size(_host, out_ptr):
        ctypes.memmove(out_ptr, ctypes.byref(ctypes.c_size_t(page_bytes)), ctypes.sizeof(ctypes.c_size_t))
        return 0

    def fake_stat64(_host, flavor, buf, count_ptr):
        assert flavor == 4
        if not state["use_stat64"]:
            return -1
        arr = (ctypes.c_uint32 * 38)()
        arr[0] = state["free_pages"]
        arr[2] = state["inactive_pages"]
        ctypes.memmove(buf, arr, ctypes.sizeof(arr))
        ctypes.memmove(count_ptr, ctypes.byref(ctypes.c_uint(38)), 4)
        return 0

    def fake_stat32(_host, flavor, buf, count_ptr):
        assert flavor == 2
        arr = (ctypes.c_uint * 15)()
        arr[0] = 100
        arr[2] = 200
        ctypes.memmove(buf, arr, ctypes.sizeof(arr))
        ctypes.memmove(count_ptr, ctypes.byref(ctypes.c_uint(15)), 4)
        return 0

    libc = SimpleNamespace(
        sysctlbyname=FakeFn(fake_sysctl),
        mach_host_self=FakeFn(lambda: 77),
        host_page_size=FakeFn(fake_host_page_size),
        host_statistics64=FakeFn(fake_stat64),
        host_statistics=FakeFn(fake_stat32),
        mach_port_deallocate=FakeFn(lambda _t, _h: 0),
    )

    monkeypatch.setattr(ctypes, "CDLL", lambda _name: libc)
    monkeypatch.setattr(
        ctypes.c_uint,
        "in_dll",
        classmethod(lambda cls, _lib, _sym: cls(88)),
    )

    avail, total = memory_utils._macos_memory()
    assert total == total_bytes
    assert avail == (2000 + 3000) * page_bytes
    assert libc.host_page_size.argtypes[1] is ctypes.POINTER(ctypes.c_size_t)
    assert libc.host_statistics64.argtypes is not None

    # Clamps oversized page counts cleanly to total physical memory.
    state["free_pages"] = 10**9
    avail_clamped, _ = memory_utils._macos_memory()
    assert avail_clamped == total_bytes

    # Falls back cleanly to HOST_VM_INFO (host_statistics) if host_statistics64 fails.
    state["use_stat64"] = False
    avail_fb, total_fb = memory_utils._macos_memory()
    assert (avail_fb, total_fb) == (300 * page_bytes, total_bytes)


def test_macos_physical_cores_declares_argtypes(monkeypatch):
    class FakeFn:
        def __init__(self, fn):
            self._fn = fn
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self._fn(*args)

    def fake_sysctl(name, oldp, oldlenp, _newp, _newlen):
        assert name == b"hw.physicalcpu"
        if oldp is None:
            ctypes.memmove(oldlenp, ctypes.byref(ctypes.c_size_t(4)), ctypes.sizeof(ctypes.c_size_t))
            return 0
        ctypes.memmove(oldp, ctypes.byref(ctypes.c_uint32(10)), 4)
        return 0

    libc = SimpleNamespace(sysctlbyname=FakeFn(fake_sysctl))
    monkeypatch.setattr(ctypes, "CDLL", lambda _name: libc)
    assert platform_utils._macos_physical_cores() == 10
    assert libc.sysctlbyname.argtypes is not None
    assert libc.sysctlbyname.restype is ctypes.c_int


def test_fp32_build_options_and_all_six_gpu_modules_respect_optional_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENFOCUS_OPENCL_CACHE_DIR", str(tmp_path))
    recorded_options = []

    class FakeKernel:
        def __init__(self, _prog, _name):
            pass

        def get_work_group_info(self, _info, _dev):
            return 256

    class FakeProgram:
        def __init__(self, _ctx, _src):
            pass

        def build(self, options=None):
            recorded_options.append(list(options or []))
            return self

    class FakeContext:
        def __init__(self, _devs):
            pass

    cl = SimpleNamespace(
        Context=FakeContext,
        CommandQueue=lambda ctx, device=None: SimpleNamespace(ctx=ctx, device=device),
        Program=FakeProgram,
        Kernel=FakeKernel,
        device_fp_config=SimpleNamespace(CORRECTLY_ROUNDED_DIVIDE_SQRT=8),
        kernel_work_group_info=SimpleNamespace(WORK_GROUP_SIZE=1),
    )

    unsupported_dev = SimpleNamespace(
        int_ptr=0x1111,
        name="IntegratedGPU",
        vendor="Test",
        driver_version="1.0",
        version="OpenCL 1.2",
        platform=SimpleNamespace(name="P", vendor="V", version="1.2"),
        global_mem_size=4 * 1024**3,
        max_work_group_size=256,
        max_work_item_sizes=(256, 256, 64),
        local_mem_size=32768,
        single_fp_config=0,
    )
    supported_dev = SimpleNamespace(**{**vars(unsupported_dev), "int_ptr": 0x2222, "single_fp_config": 8})

    assert ogpu._fp32_build_options(cl, unsupported_dev) == []
    assert ogpu._fp32_build_options(cl, supported_dev) == ["-cl-fp32-correctly-rounded-divide-sqrt"]

    for mod in (ogpu, ogpu_a, ogpu_c, ogpu_dct, ogpu_dtcwt, ogpu_gff, ogpu_gfg):
        mod._invalidate_program_cache()
    try:
        for mod in (ogpu_a, ogpu_c, ogpu_dct, ogpu_dtcwt, ogpu_gff, ogpu_gfg):
            mod._get_or_build_program(cl, unsupported_dev)
        assert len(recorded_options) == 6
        assert all(opt == [] for opt in recorded_options)
    finally:
        for mod in (ogpu, ogpu_a, ogpu_c, ogpu_dct, ogpu_dtcwt, ogpu_gff, ogpu_gfg):
            mod._invalidate_program_cache()


def test_linux_drm_free_vram_and_free_vram_routing(monkeypatch, tmp_path):
    drm = tmp_path / "drm"
    card0 = drm / "card0" / "device"
    card1 = drm / "card1" / "device"
    card0.mkdir(parents=True)
    card1.mkdir(parents=True)
    (drm / "card0-DP-1").mkdir()

    (card0 / "vendor").write_text("0x1002\n", encoding="ascii")
    (card0 / "mem_info_vram_total").write_text(str(8 * 1024**3), encoding="ascii")
    (card0 / "mem_info_vram_used").write_text(str(2 * 1024**3), encoding="ascii")

    (card1 / "vendor").write_text("0x8086\n", encoding="ascii")
    (card1 / "mem_info_vram_total").write_text(str(16 * 1024**3), encoding="ascii")
    (card1 / "mem_info_vram_used").write_text(str(4 * 1024**3), encoding="ascii")

    amd_dev = SimpleNamespace(vendor_id=0x1002, global_mem_size=8 * 1024**3, name="Radeon", extensions="")
    intel_dev = SimpleNamespace(vendor_id=0x8086, global_mem_size=16 * 1024**3, name="Arc", extensions="")
    other_dev = SimpleNamespace(vendor_id=0x10de, global_mem_size=8 * 1024**3, name="RTX", extensions="")

    assert ogpu._linux_drm_free_vram(amd_dev, drm_root=str(drm)) == (6 * 1024**3, 8 * 1024**3)
    assert ogpu._linux_drm_free_vram(intel_dev, drm_root=str(drm)) == (12 * 1024**3, 16 * 1024**3)
    assert ogpu._linux_drm_free_vram(other_dev, drm_root=str(drm)) is None

    # Dual identical GPUs under separate PCI devices must be treated as ambiguous -> None.
    card2 = drm / "card2" / "device"
    card2.mkdir(parents=True)
    (card2 / "vendor").write_text("0x1002\n", encoding="ascii")
    (card2 / "mem_info_vram_total").write_text(str(8 * 1024**3), encoding="ascii")
    (card2 / "mem_info_vram_used").write_text(str(1 * 1024**3), encoding="ascii")
    assert ogpu._linux_drm_free_vram(amd_dev, drm_root=str(drm)) is None

    # Route check on win32 DXGI and linux DRM without calling external nvidia-smi.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ogpu, "_dxgi_free_vram", lambda _d: (5 * 1024**3, 8 * 1024**3))
    assert ogpu._free_vram(intel_dev) == (5 * 1024**3, 8 * 1024**3)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ogpu, "_linux_drm_free_vram", lambda _d: (11 * 1024**3, 16 * 1024**3))
    assert ogpu._free_vram(intel_dev) == (11 * 1024**3, 16 * 1024**3)


@pytest.mark.skipif(sys.platform != "win32", reason="DXGI probe is Windows-only")
def test_dxgi_live_windows_discrete_gpu():
    import pyopencl as cl

    dev = ogpu._device(cl)
    if dev is None or getattr(dev, "host_unified_memory", False):
        pytest.skip("No discrete Windows OpenCL GPU available")
    reading = ogpu._dxgi_free_vram(dev)
    assert reading is not None
    free_b, total_b = reading
    assert 0 <= free_b <= total_b and total_b > 0


def test_perspective_and_ecc_area_support_unified_memory(monkeypatch):
    snap = _ample_snapshot()
    monkeypatch.setattr(resource_policy, "probe_memory", lambda: snap)
    monkeypatch.setattr(resource_policy, "probe_resources", lambda: snap)

    def fail_discrete(*_args, **_kwargs):
        raise AssertionError("unified-memory path must not require discrete _free_vram")

    monkeypatch.setattr(persp_gpu, "_free_vram", fail_discrete)
    monkeypatch.setattr(ogpu, "_free_vram", fail_discrete)

    unified_dev = SimpleNamespace(
        host_unified_memory=True,
        global_mem_size=8 * 1024**3,
        max_mem_alloc_size=2 * 1024**3,
    )

    p = persp_gpu.ExactPerspectiveGpu()
    p._device = unified_dev
    inv = np.eye(3, dtype=np.float64)
    plan = p._plan_bands(inv, (0, 0, 512, 512), (512, 512), 3, 1, 1024)
    assert plan is not None

    class DummyBuf:
        def release(self):
            pass

    a = ecc_gpu.ExactAreaGpu()
    a._device = unified_dev
    a._context = object()
    a._queue = object()
    a._kernels = {}
    a._cl = SimpleNamespace(
        mem_flags=SimpleNamespace(READ_ONLY=1, WRITE_ONLY=2, COPY_HOST_PTR=4),
        Buffer=lambda *_args, **_kwargs: DummyBuf(),
    )
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    assert a._workspace(img, 256, 256, 3, (2, 2)) is True
    a.close()
