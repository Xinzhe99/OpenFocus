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
    state = {'count64': 40, 'memsize_count': 8, 'host': 77, 'free_pages': 2000}
    released = []

    def fake_sysctl(name, oldp, oldlenp, _newp, _newlen):
        if name == b"hw.memsize" and oldp is not None:
            ctypes.memmove(oldp, ctypes.byref(ctypes.c_uint64(total_bytes)), 8)
            ctypes.cast(oldlenp, ctypes.POINTER(ctypes.c_size_t))[0] = state['memsize_count']
            return 0
        return -1

    def fake_host_page_size(_host, out_ptr):
        assert ctypes.sizeof(out_ptr._obj) == ctypes.sizeof(ctypes.c_size_t)
        ctypes.memmove(out_ptr, ctypes.byref(ctypes.c_size_t(page_bytes)), ctypes.sizeof(ctypes.c_size_t))
        return 0

    def fake_stat64(_host, flavor, buf, count_ptr):
        assert flavor == 4
        count = ctypes.cast(count_ptr, ctypes.POINTER(ctypes.c_uint))
        assert count[0] == 40
        if state['count64'] == 0:
            return -1
        stats = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint32))
        stats[0] = state['free_pages']
        stats[2] = 3000
        count[0] = state['count64']
        return 0

    def fake_stat32(_host, flavor, buf, count_ptr):
        assert flavor == 2
        count = ctypes.cast(count_ptr, ctypes.POINTER(ctypes.c_uint))
        assert count[0] == 15
        stats = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint32))
        stats[0] = 100
        stats[2] = 200
        count[0] = 12
        return 0

    libc = SimpleNamespace(
        sysctlbyname=FakeFn(fake_sysctl),
        mach_host_self=FakeFn(lambda: state['host']),
        host_page_size=FakeFn(fake_host_page_size),
        host_statistics64=FakeFn(fake_stat64),
        host_statistics=FakeFn(fake_stat32),
        mach_port_deallocate=FakeFn(lambda task, host: released.append((task, host)) or 0),
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
    assert released == [(88, 77)]

    state['free_pages'] = 10**9
    avail_clamped, _ = memory_utils._macos_memory()
    assert avail_clamped == total_bytes

    # A short, failed, or oversized reply cannot supply the page counters.
    for count in (2, 0, 41):
        state['count64'] = count
        assert memory_utils._macos_memory() == (300 * page_bytes, total_bytes)
    assert released == [(88, 77)] * 5

    state['host'] = 0
    assert memory_utils._macos_memory() == (0, total_bytes)
    state['memsize_count'] = 4
    assert memory_utils._macos_memory() == (0, 0)
    assert released == [(88, 77)] * 5


def test_macos_physical_cores_declares_argtypes(monkeypatch):
    class FakeFn:
        def __init__(self, fn):
            self._fn = fn
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self._fn(*args)
    size_written = 4

    def fake_sysctl(name, oldp, oldlenp, _newp, _newlen):
        assert name == b"hw.physicalcpu"
        assert oldp is not None
        count = ctypes.cast(oldlenp, ctypes.POINTER(ctypes.c_size_t))
        assert count[0] == ctypes.sizeof(ctypes.c_int)
        ctypes.cast(oldp, ctypes.POINTER(ctypes.c_int))[0] = 10
        count[0] = size_written
        return 0

    libc = SimpleNamespace(sysctlbyname=FakeFn(fake_sysctl))
    monkeypatch.setattr(ctypes, "CDLL", lambda _name: libc)
    assert platform_utils._macos_physical_cores() == 10
    assert libc.sysctlbyname.argtypes is not None
    assert libc.sysctlbyname.restype is ctypes.c_int
    size_written = 2
    assert platform_utils._macos_physical_cores() == 0


def test_linux_cpu_limits_use_complete_sysfs_topology(monkeypatch, tmp_path):
    for cpu, files in (
        (0, {"core_cpus_list": "0-1\n\0"}),
        (1, {"physical_package_id": "0\n", "core_id": "0\n"}),
        (3, {"physical_package_id": "0\n", "core_id": "1\n"}),
        (7, {"core_cpus_list": "7\n"}),
    ):
        topology = tmp_path / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        for name, value in files.items():
            (topology / name).write_text(value, encoding="ascii")

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/cpuinfo":
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    sysfs_cores = platform_utils._linux_sysfs_physical_cores
    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(platform_utils, "_linux_sysfs_physical_cores",
                        lambda affinity: sysfs_cores(affinity, sys_cpu_root=str(tmp_path)))
    monkeypatch.setattr(platform_utils, "is_windows", lambda: False)
    monkeypatch.setattr(platform_utils, "is_linux", lambda: True)
    monkeypatch.setattr(platform_utils.os, "sched_getaffinity", lambda pid: {0, 1, 3}, raising=False)
    monkeypatch.setattr(platform_utils.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(platform_utils, "_linux_cpu_quota", lambda: 0)

    assert platform_utils.cpu_limits() == (2, 3, 0, True)
    (tmp_path / "cpu3" / "topology" / "physical_package_id").write_text("-1\n", encoding="ascii")
    assert platform_utils.cpu_limits() == (1, 3, 0, False)


@pytest.mark.parametrize('fp_flags', [None, 0, 8])
def test_opencl_fusion_build_options_match_device_capabilities(monkeypatch, tmp_path, fp_flags):
    monkeypatch.setenv("OPENFOCUS_OPENCL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ogpu, '_SHARED_CONTEXT', None)
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
        def __init__(self, devices):
            pass

    cl = SimpleNamespace(
        Context=FakeContext,
        CommandQueue=lambda ctx, device=None: SimpleNamespace(ctx=ctx, device=device),
        Program=FakeProgram,
        Kernel=FakeKernel,
        device_fp_config=SimpleNamespace(CORRECTLY_ROUNDED_DIVIDE_SQRT=8),
        kernel_work_group_info=SimpleNamespace(WORK_GROUP_SIZE=1),
    )

    device = SimpleNamespace(
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
    )
    if fp_flags is not None:
        device.single_fp_config = fp_flags
    expected = ['-cl-fp32-correctly-rounded-divide-sqrt'] if fp_flags == 8 else []

    for mod in (ogpu, ogpu_a, ogpu_c, ogpu_dct, ogpu_dtcwt, ogpu_gff, ogpu_gfg):
        mod._invalidate_program_cache()
    try:
        for mod in (ogpu_a, ogpu_c, ogpu_dct, ogpu_dtcwt, ogpu_gff, ogpu_gfg):
            mod._get_or_build_program(cl, device)
        assert recorded_options == [expected] * 6
    finally:
        for mod in (ogpu, ogpu_a, ogpu_c, ogpu_dct, ogpu_dtcwt, ogpu_gff, ogpu_gfg):
            mod._invalidate_program_cache()


def test_linux_drm_free_vram_and_free_vram_routing(monkeypatch, tmp_path):
    import subprocess

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

    amd_dev = SimpleNamespace(vendor_id=0x1002, global_mem_size=8 * 1024**3, name="Radeon", extensions="")
    intel_dev = SimpleNamespace(vendor_id=0x8086, global_mem_size=16 * 1024**3, name="Arc", extensions="")
    other_dev = SimpleNamespace(vendor_id=0x10de, global_mem_size=8 * 1024**3, name="RTX", extensions="")

    assert ogpu._linux_drm_free_vram(amd_dev, drm_root=str(drm)) == (6 * 1024**3, 8 * 1024**3)
    assert ogpu._linux_drm_free_vram(intel_dev, drm_root=str(drm)) is None
    assert ogpu._linux_drm_free_vram(other_dev, drm_root=str(drm)) is None

    # Dual identical GPUs under separate PCI devices must be treated as ambiguous -> None.
    card2 = drm / "card2" / "device"
    card2.mkdir(parents=True)
    (card2 / "vendor").write_text("0x1002\n", encoding="ascii")
    (card2 / "mem_info_vram_total").write_text(str(8 * 1024**3), encoding="ascii")
    (card2 / "mem_info_vram_used").write_text(str(1 * 1024**3), encoding="ascii")
    assert ogpu._linux_drm_free_vram(amd_dev, drm_root=str(drm)) is None

    realpath = os.path.realpath
    pci_paths = {str(card0): '/sys/devices/pci0000:00/0000:03:00.0',
                 str(card2): '/sys/devices/pci0001:00/0001:03:00.0'}

    def resolve_device(path):
        return pci_paths[str(path)] if str(path) in pci_paths else realpath(path)

    monkeypatch.setattr(os.path, 'realpath', resolve_device)
    monkeypatch.setattr(ogpu, '_opencl_pci_address', lambda device: (1, 3, 0, 0))
    assert ogpu._linux_drm_free_vram(amd_dev, drm_root=str(drm)) == (7 * 1024**3, 8 * 1024**3)
    (card2 / 'mem_info_vram_used').unlink()
    assert ogpu._linux_drm_free_vram(amd_dev, drm_root=str(drm)) is None
    monkeypatch.setattr(ogpu, '_opencl_pci_address', lambda device: (None, 3, 0, 0))
    assert ogpu._linux_drm_free_vram(amd_dev, drm_root=str(drm)) is None

    # Route check on win32 DXGI and linux DRM without calling external nvidia-smi.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ogpu, "_dxgi_free_vram", lambda _d: (5 * 1024**3, 8 * 1024**3))
    assert ogpu._free_vram(intel_dev) == (5 * 1024**3, 8 * 1024**3)
    monkeypatch.setattr(ogpu, '_nvml_free_vram', lambda device: None)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: pytest.fail('DXGI already supplied a budget'))
    assert ogpu._free_vram(other_dev) == (5 * 1024**3, 8 * 1024**3)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ogpu, '_linux_drm_free_vram', lambda device: (5 * 1024**3, 8 * 1024**3))
    assert ogpu._free_vram(amd_dev) == (5 * 1024**3, 8 * 1024**3)


@pytest.mark.parametrize('second', ['same-adapter', 'equal-budget', 'unknown-budget'])
def test_dxgi_rejects_distinct_matching_adapters(monkeypatch, second):
    total = 8 * 1024**3
    reading = total // 2, total
    device = SimpleNamespace(vendor_id=0x8086, global_mem_size=total, name='Intel Arc')
    first = (111, device.name, total, reading)
    other = first if second == 'same-adapter' else (
        222, device.name, total, None if second == 'unknown-budget' else reading,
    )
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(resource_policy, '_dxgi_adapter_budgets', lambda vendor: (first, other))
    expected = reading if second == 'same-adapter' else None
    assert ogpu._dxgi_free_vram(device) == expected


@pytest.mark.parametrize('luid, node_mask, expected_free', [
    ((222, 0), 1, 2), ((333, 0), 1, None), ((222, 0), 2, None),
])
def test_dxgi_uses_the_opencl_adapter_luid(monkeypatch, luid, node_mask, expected_free):
    gib = 1024**3
    device = SimpleNamespace(vendor_id=0x8086, global_mem_size=8 * gib, name='Intel Arc')
    adapters = [((111, 0), device.name, 8 * gib, (6 * gib, 8 * gib)),
                ((222, 0), device.name, 8 * gib, (2 * gib, 8 * gib))]
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(resource_policy, '_dxgi_adapter_budgets', lambda vendor: adapters)
    monkeypatch.setattr(ogpu, '_opencl_dxgi_identity', lambda device: (luid, node_mask))
    reading = ogpu._dxgi_free_vram(device)
    assert reading == (None if expected_free is None else (expected_free * gib, 8 * gib))


@pytest.mark.skipif(sys.platform != "win32", reason="DXGI probe is Windows-only")
def test_dxgi_live_windows_discrete_gpu():
    cl = pytest.importorskip('pyopencl')
    try:
        dev = ogpu._device(cl)
    except cl.Error:
        pytest.skip('OpenCL platform unavailable')
    if dev is None or getattr(dev, "host_unified_memory", False):
        pytest.skip("No discrete Windows OpenCL GPU available")
    reading = ogpu._dxgi_free_vram(dev)
    if reading is None:
        pytest.skip('Selected OpenCL device has no uniquely verified DXGI reading')
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

    # Output, driver staging, and device buffers share one process envelope.
    limit = 2 * 1024**2
    with resource_policy.process_budget(resource_policy.ProcessBudget(limit, limit, limit)):
        smaller = p._plan_bands(inv, (0, 0, 512, 512), (512, 512), 3, 1, 1024)
        assert smaller is not None
        bands, source_bytes, dest_bytes, staged = smaller
        assert bands[0][1] < plan[0][0][1]
        assert img.nbytes + 2 * (source_bytes + dest_bytes) + staged + 1024 <= limit
        assert not a._workspace(img, 128, 128, 3, (4, 4))
