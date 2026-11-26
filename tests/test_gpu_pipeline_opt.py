"""Regression and adaptive resource-scheduling tests for the OpenFocus default pipeline."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import hashlib
import os
import threading

import cv2
import numpy as np
import pytest

from core import openfocus_alignment, registration, resource_policy
from core.multi_focus_fusion import MultiFocusFusion
from core.render_preflight import estimate_render_memory
from core.source_frames import SourceFrame, process_frame, scan_source, without_source_cache_growth
from fusion_methods import _openfocus_gpu
from fusion_methods._openfocus_execution import serial_execution_scope
from fusion_methods.openfocus_a import openfocus_a_impl
from fusion_methods.openfocus_bc import openfocus_b_impl, openfocus_c_impl


def _require_opencl_gpu():
    device_name = _openfocus_gpu.probe_device_name()
    if not device_name:
        pytest.skip('OpenCL GPU unavailable')
    return device_name


def test_prefetch_window_and_multi_worker_order(monkeypatch):
    monkeypatch.setattr(os, 'cpu_count', lambda: 8)
    assert _openfocus_gpu._prefetch_window(1000, 950, 100, 10) == 0
    assert _openfocus_gpu._prefetch_window(1000, 500, 100, 10, thread_count=1) == 0
    assert _openfocus_gpu._prefetch_window(1000, 500, 100, 10, thread_count=2) == 1
    assert _openfocus_gpu._prefetch_window(1000, 500, 100, 3) == 2
    assert _openfocus_gpu._prefetch_window(2000, 100, 100, 12, max_slots=3) <= 3

    active = 0
    peak_active = 0
    lock = threading.Lock()
    barrier = threading.Event()

    def prepare(index, item, cancelled):
        nonlocal active, peak_active
        assert not cancelled()
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            if peak_active >= 2:
                barrier.set()
        barrier.wait(timeout=1.0)
        with lock:
            active -= 1
        return index, item * 10

    items = list(range(6))
    seen = list(_openfocus_gpu._iter_prefetched(
        items, prepare, enabled=True, max_in_flight=3,
    ))
    assert seen == [(i, (i, i * 10)) for i in items]
    assert peak_active >= 2

    with serial_execution_scope(None):
        single = list(_openfocus_gpu._iter_prefetched(
            [1, 2, 3], lambda idx, val, _c: val + 1, enabled=True, max_in_flight=3,
        ))
    assert single == [(0, 2), (1, 3), (2, 4)]


def test_weak_integrated_gpu_filter(monkeypatch):
    weak = SimpleNamespace(host_unified_memory=True, max_compute_units=8, available=True,
                           compiler_available=True, global_mem_size=4 << 30, name='Weak iGPU')
    strong_igpu = SimpleNamespace(host_unified_memory=True, max_compute_units=24, available=True,
                                  compiler_available=True, global_mem_size=4 << 30, name='Strong iGPU')
    discrete = SimpleNamespace(host_unified_memory=False, max_compute_units=4, available=True,
                               compiler_available=True, global_mem_size=8 << 30, name='Discrete GPU')
    mock_default = SimpleNamespace(host_unified_memory=True, max_compute_units=0)

    monkeypatch.setattr(os, 'cpu_count', lambda: 16)
    monkeypatch.setattr('core._native.native_available', lambda: True)
    assert _openfocus_gpu._is_weak_integrated_gpu(weak) is True
    assert _openfocus_gpu._is_weak_integrated_gpu(strong_igpu) is False
    assert _openfocus_gpu._is_weak_integrated_gpu(discrete) is False
    assert _openfocus_gpu._is_weak_integrated_gpu(mock_default) is False

    monkeypatch.setattr(os, 'cpu_count', lambda: 4)
    assert _openfocus_gpu._is_weak_integrated_gpu(weak) is False


def test_opencl_binary_disk_cache_hit_miss_and_corrupt_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENFOCUS_OPENCL_CACHE_DIR', str(tmp_path))
    monkeypatch.delenv('PYOPENCL_BUILD_OPTIONS', raising=False)
    builds = []

    class FakeProgram:
        def __init__(self, ctx, *args):
            self.args = args

        def build(self, options=None):
            if len(self.args) == 1:
                builds.append(('source', self.args[0]))
            else:
                builds.append(('binary', self.args[1]))
            return self

        def get_info(self, info_key):
            assert info_key == 'BINARIES'
            return [b'compiled-opencl-binary']

    fake_cl = SimpleNamespace(
        Program=FakeProgram,
        program_info=SimpleNamespace(BINARIES='BINARIES'),
    )
    device = SimpleNamespace(name='TestGPU', vendor='TestVendor', driver_version='1.0', version='OpenCL 3.0')

    prog1 = _openfocus_gpu._build_cached_program(fake_cl, device, 'ctx', '__kernel void k() {}')
    assert builds == [('source', '__kernel void k() {}')]
    cached_files = list(tmp_path.glob('*.clbin'))
    assert len(cached_files) == 1
    assert prog1.args == ('__kernel void k() {}',)

    prog2 = _openfocus_gpu._build_cached_program(fake_cl, device, 'ctx', '__kernel void k() {}')
    assert builds == [
        ('source', '__kernel void k() {}'),
        ('binary', [b'compiled-opencl-binary']),
    ]
    assert prog2.args == ([device], [b'compiled-opencl-binary'])

    class CorruptFallbackProgram(FakeProgram):
        def build(self, options=None):
            if len(self.args) == 2:
                raise RuntimeError('corrupt binary')
            return super().build(options=options)

    cached_files[0].write_bytes(b'corrupt')
    fake_cl.Program = CorruptFallbackProgram
    _openfocus_gpu._build_cached_program(fake_cl, device, 'ctx', '__kernel void k() {}')
    assert builds[-1] == ('source', '__kernel void k() {}')
    assert cached_files[0].read_bytes() == b'compiled-opencl-binary'

    fake_cl.Program = FakeProgram
    monkeypatch.setenv('PYOPENCL_BUILD_OPTIONS', '-D CACHE_VARIANT=1')
    _openfocus_gpu._build_cached_program(fake_cl, device, 'ctx', '__kernel void k() {}')
    assert builds[-1] == ('source', '__kernel void k() {}')
    assert len(list(tmp_path.glob('*.clbin'))) == 2

    device.platform = SimpleNamespace(name='Second OpenCL platform', vendor='TestVendor', version='OpenCL 3.0')
    _openfocus_gpu._build_cached_program(fake_cl, device, 'ctx', '__kernel void k() {}')
    assert builds[-1] == ('source', '__kernel void k() {}')
    assert len(list(tmp_path.glob('*.clbin'))) == 3


def test_file_backed_alignment_rejects_changed_pixels_with_same_stamp(tmp_path):
    path = tmp_path / 'source.bmp'
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), pixels)
    source = scan_source(str(path), header_only=True)[0]
    stat = path.stat()
    with without_source_cache_growth():
        plan = openfocus_alignment.estimate_alignment_plan([source])
        transform = plan.transforms[0]
        assert transform.source.digest_kind == 'file'
        assert transform.source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        np.testing.assert_array_equal(
            openfocus_alignment.apply_alignment_region(source, transform, 128, 128, 16, 16),
            pixels[128:144, 128:144],
        )
        # This BMP row lies outside the samples used by the old file digest.
        pixels[128, 128] ^= 255
        assert cv2.imwrite(str(path), pixels)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert (path.stat().st_size, path.stat().st_mtime_ns) == source.fingerprint
        with pytest.raises(ValueError, match='Alignment source pixels changed'):
            openfocus_alignment.apply_alignment_region(source, transform, 128, 128, 16, 16)


def test_perspective_gpu_vram_ttl_refreshes_before_fallback(monkeypatch):
    from core import _perspective_gpu

    clock = [100.0]
    available = [1024]
    reads = []

    def query(_device):
        reads.append(available[0])
        return available[0], 8 << 30

    monkeypatch.setattr(_perspective_gpu.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(_perspective_gpu, '_free_vram', query)
    monkeypatch.setattr(resource_policy, 'vram_budget', lambda snapshot: snapshot.free_vram)

    gpu = _perspective_gpu.ExactPerspectiveGpu()
    gpu._device = SimpleNamespace(max_mem_alloc_size=1 << 30)
    inverse = np.eye(3, dtype=np.float64)

    def plan(side):
        return gpu._plan_bands(inverse, (0, 0, side, side), (512, 512), 3, 1, 0)

    assert plan(8) is not None
    clock[0] = 101.0
    assert plan(8) is not None
    assert reads == [1024]

    available[0] = 2 * 1024 ** 2
    bands, *_ = plan(512)
    assert sum(band[1] for band in bands) == 512
    assert reads == [1024, available[0]]

    clock[0] = 103.5
    assert plan(512) is not None
    assert reads == [1024, available[0], available[0]]
    gpu.close()
    assert gpu._vram_cache is None


def test_openfocus_b_processed_disk_stack_stages_once(tmp_path):
    rng = np.random.default_rng(11)
    paths = []
    for i in range(4):
        p = tmp_path / f'stack_{i}.png'
        img = rng.integers(0, 255, (1800, 1800, 3), dtype=np.uint8)
        cv2.imwrite(str(p), img)
        paths.append(p)

    resolve_counts = [0] * len(paths)

    def make_op(idx):
        def op(arr):
            resolve_counts[idx] += 1
            return arr
        return op

    sources = [process_frame(scan_source(str(p), header_only=True)[0], make_op(i))
               for i, p in enumerate(paths)]
    merger = MultiFocusFusion(
        algorithm='openfocus_b', use_gpu=False, tile_enabled=False, exposure_match=False,
    )
    merger.align_images = lambda *args, **kwargs: None
    result = merger.fuse(sources)
    assert result.shape == (1800, 1800, 3)
    assert all(c == 1 for c in resolve_counts)


def test_openfocus_b_skips_optional_staging_when_scratch_tight_or_huge(tmp_path, monkeypatch):
    from core import frame_storage, source_frames

    p0 = tmp_path / 's0.png'
    p1 = tmp_path / 's1.png'
    arr = np.zeros((1800, 1800, 3), dtype=np.uint8)
    cv2.imwrite(str(p0), arr)
    cv2.imwrite(str(p1), arr)
    s0 = scan_source(str(p0), header_only=True)[0]
    s1 = scan_source(str(p1), header_only=True)[0]
    sources = [process_frame(s0, lambda x: x), process_frame(s1, lambda x: x),
               process_frame(s0, lambda x: x), process_frame(s1, lambda x: x)]

    staged = []
    monkeypatch.setattr(source_frames, 'materialize_for_repeated_access',
                        lambda srcs, *args, **kwargs: staged.append(len(srcs)) or srcs)
    monkeypatch.setattr(frame_storage, 'assess_scratch', lambda *args, **kwargs: {'status': 'advisory'})
    merger = MultiFocusFusion(algorithm='openfocus_b', use_gpu=False, tile_enabled=False)
    merger.align_images = lambda *args, **kwargs: None
    merger.fuse(sources)
    assert staged == []


def test_preflight_8k_300_frames_no_false_ram_warning(monkeypatch):
    GiB = 1024 ** 3
    snapshot = resource_policy.ResourceSnapshot(
        available_ram=6 * GiB, total_ram=16 * GiB, physical_cores=8, logical_cpus=16,
    )
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda kind='CPU': snapshot)
    frame_8k = SourceFrame(
        'frame_000.png', 0, (4320, 7680, 3), np.dtype('uint8'), None, (12 * 1024 ** 2, 0),
    )

    est_10 = estimate_render_memory([frame_8k] * 10, 'openfocus_b', {}, register=True, alignment='ecc')
    est_300 = estimate_render_memory([frame_8k] * 300, 'openfocus_b', {}, register=True, alignment='ecc')
    assert not est_300.ram_short
    assert est_300.fusion_ram == est_10.fusion_ram
    assert est_300.registration_ram < GiB

    for algo in ('openfocus_a', 'openfocus_c'):
        est_algo = estimate_render_memory([frame_8k] * 300, algo, {}, register=True, alignment='openfocus')
        assert not est_algo.ram_short


@pytest.mark.parametrize('alignment', ['ecc', 'openfocus'])
def test_preflight_uses_imported_jpeg_metadata(monkeypatch, tmp_path, alignment):
    from PIL import Image

    path = str(tmp_path / 'imported.jpg')
    frame = SourceFrame(path, 0, (4320, 7680, 3), np.dtype('uint8'), None,
                        (12 * 1024 ** 2, 0), _jpeg_components=3, _jpeg_single_scan=True)
    original_getsize = os.path.getsize

    def getsize(candidate):
        if candidate == path:
            pytest.fail('Preflight queried the JPEG on disk')
        return original_getsize(candidate)

    def open_image(*args, **kwargs):
        pytest.fail('Preflight opened the JPEG on disk')

    monkeypatch.setattr(os.path, 'getsize', getsize)
    monkeypatch.setattr(Image, 'open', open_image)
    estimate = estimate_render_memory([frame] * 300, 'openfocus_b', {},
                                      register=True, alignment=alignment)
    assert 0 < estimate.registration_ram < 1024 ** 3


def test_openfocus_low_ram_retry_keeps_rejection_for_resized_render(monkeypatch):
    from core import multi_focus_fusion as mff
    from core.algorithm_adapters import openfocus_b_resources

    frames = [np.full((64, 64, 3), 20 + i * 30, np.uint8) for i in range(4)]
    # In-memory frames retry with the fixed buffers and one 128-row band. The
    # budget holds that floor but not the RAM a 64 -> 128 resize needs.
    fixed, per_worker = openfocus_b_resources(128, 128, {}, len(frames))
    floor = fixed + min(per_worker, 128 * 128 * 64)
    resize_bytes = (64 * 64 + 128 * 128) * 3 * 8
    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: floor + resize_bytes // 2)
    events = []
    monkeypatch.setattr(mff, 'log_action', lambda event, **kw: events.append(event))
    original_plan = resource_policy.plan_execution
    rejection = resource_policy.ResourceLimitError('first plan rejected')
    calls = []

    def first_plan_rejects(workload, **kwargs):
        if workload.algorithm == 'openfocus_b':
            calls.append(kwargs.get('user_threads'))
            if len(calls) == 1:
                raise rejection
        return original_plan(workload, **kwargs)

    monkeypatch.setattr(resource_policy, 'plan_execution', first_plan_rejects)
    fusion = MultiFocusFusion('openfocus_b', use_gpu=False, tile_enabled=False, exposure_match=False)
    with pytest.raises(resource_policy.ResourceLimitError) as rejected:
        fusion.fuse(frames, img_resize=(128, 128))
    assert rejected.value is rejection
    assert len(calls) == 1
    assert 'openfocus_low_ram_degrade' not in events

    # The same budget still lets an unresized render fall back to one worker.
    calls.clear()
    result = fusion.fuse(frames)
    assert result.shape == (64, 64, 3)
    assert calls[1] == 1
    assert 'openfocus_low_ram_degrade' in events


def test_openfocus_b_low_ram_auto_degrades_to_serial(monkeypatch):
    from core import multi_focus_fusion as mff

    frames = [np.full((64, 64, 3), 20 + i * 30, np.uint8) for i in range(4)]
    original_plan = resource_policy.plan_execution
    calls = []

    def tight_plan(workload, **kwargs):
        calls.append((workload.outer_tasks, kwargs.get('user_threads'), workload.per_worker_bytes))
        if len(calls) == 1:
            raise resource_policy.ResourceLimitError('tight memory simulation')
        return original_plan(workload, **kwargs)

    monkeypatch.setattr(resource_policy, 'plan_execution', tight_plan)
    fusion = MultiFocusFusion('openfocus_b', use_gpu=False, tile_enabled=False, exposure_match=False)
    result = fusion.fuse(frames, thread_count=4)
    assert result.shape == (64, 64, 3)
    assert len(calls) >= 2 and calls[1][1] == 1

    monkeypatch.setattr(mff, 'memory_budget_bytes', lambda: 0)
    with pytest.raises(resource_policy.ResourceLimitError):
        fusion.fuse(frames, thread_count=4)


def test_openfocus_low_ram_retry_releases_source_cache_before_rejecting(monkeypatch):
    from core import multi_focus_fusion as mff, source_frames

    GiB = 1024 ** 3
    rng = np.random.default_rng(43)
    frames = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
    roomy = resource_policy.ResourceSnapshot(
        available_ram=6 * GiB, total_ram=8 * GiB, physical_cores=4, logical_cpus=8,
    )
    reserve = max(
        resource_policy.RESERVE_BYTES,
        min(int(resource_policy.RESERVE_RATIO * roomy.total_ram), resource_policy.RESERVE_CAP_BYTES),
    )
    # Known RAM with nothing above the desktop reserve until cached frames are released.
    squeezed = replace(roomy, available_ram=reserve)
    assert squeezed.ram_known and resource_policy.ram_ceiling(squeezed) == 0
    pressure, released = [], []
    clear = source_frames.SOURCE_CACHE.clear

    def release():
        if pressure:
            released.append(True)
        clear()

    monkeypatch.setattr(source_frames.SOURCE_CACHE, 'clear', release)
    probe = lambda *args, **kwargs: squeezed if pressure and not released else roomy
    monkeypatch.setattr(resource_policy, 'probe_resources', probe)
    monkeypatch.setattr(resource_policy, 'probe_memory', probe)
    fusion = MultiFocusFusion('openfocus_b', use_gpu=False, tile_enabled=False, exposure_match=False)
    expected = fusion.fuse(frames, thread_count=4)

    original_plan = resource_policy.plan_execution
    threads = []

    def tight_plan(workload, **kwargs):
        if workload.algorithm == 'openfocus_b':
            threads.append(kwargs.get('user_threads'))
            if len(threads) == 1:
                # Free RAM falls to the desktop reserve at admission.
                pressure.append(True)
                raise resource_policy.ResourceLimitError('tight memory simulation')
        return original_plan(workload, **kwargs)

    monkeypatch.setattr(resource_policy, 'plan_execution', tight_plan)
    events = []
    monkeypatch.setattr(mff, 'log_action', lambda event, **kw: events.append(event))
    result = fusion.fuse(frames, thread_count=4)
    assert released
    assert len(threads) >= 2 and threads[1] == 1
    assert 'openfocus_low_ram_degrade' in events
    np.testing.assert_array_equal(result, expected)

    # Unknown RAM still rejects at once and keeps the cached frames.
    pressure.clear()
    released.clear()
    threads.clear()
    squeezed = resource_policy.ResourceSnapshot()
    with pytest.raises(resource_policy.ResourceLimitError):
        fusion.fuse(frames, thread_count=4)
    assert not released


def test_ecc_plan_falls_back_to_cpu_resize_when_gpu_stripe_exceeds_budget(monkeypatch):
    frames = [np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (4320, 7680, 3))] * 4
    native_snap = resource_policy.ResourceSnapshot(
        available_ram=4 * 1024 ** 3, total_ram=16 * 1024 ** 3, physical_cores=8, logical_cpus=16,
    )
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *args, **kwargs: native_snap)
    gpu = registration._plan_deferred_ecc(frames, 1024, True, 'balanced', None, None)
    cpu_only = registration._plan_deferred_ecc(frames, 1024, False, 'balanced', None, None)
    assert cpu_only.accelerate_ecc is False
    min_cpu = 2 * 6 * (576 * 1024) + max(cpu_only.prepare_bytes, cpu_only.solve_bytes)
    assert gpu.prepare_bytes > cpu_only.prepare_bytes
    reserve = max(
        resource_policy.RESERVE_BYTES,
        min(int(resource_policy.RESERVE_RATIO * native_snap.total_ram), resource_policy.RESERVE_CAP_BYTES),
    )
    tight_avail = reserve + min_cpu + 4 * 1024 ** 2
    tight_snap = replace(native_snap, available_ram=tight_avail)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *args, **kwargs: tight_snap)
    degraded = registration._plan_deferred_ecc(frames, 1024, True, 'balanced', None, None)
    assert degraded.accelerate_ecc is False


def test_ecc_plan_still_admits_opencv_fallback_with_native_solver(monkeypatch):
    frame = np.zeros((576, 1024, 3), np.uint8)
    snapshot = resource_policy.ResourceSnapshot(
        available_ram=1024 ** 3 + 48 * 1024 ** 2, total_ram=4 * 1024 ** 3,
        physical_cores=8, logical_cpus=16,
    )
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda: snapshot)
    monkeypatch.setattr(registration, '_load_ecc_native', lambda: object())
    with pytest.raises(resource_policy.ResourceLimitError):
        registration._plan_deferred_ecc([frame, frame], 1024, False, 'balanced', None, None)


@pytest.mark.parametrize('algo', ['a', 'b', 'c'])
def test_openfocus_abc_gpu_ping_pong_matches_cpu(algo):
    device_name = _require_opencl_gpu()
    rng = np.random.default_rng(101)
    base = rng.integers(20, 235, (128, 128, 3), dtype=np.uint8)
    frames = []
    for k in range(6):
        shifted = base.copy()
        y0, x0 = 16 + k * 8, 16 + k * 8
        shifted[y0:y0 + 32, x0:x0 + 32] = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        frames.append(shifted)

    fuse = {'a': openfocus_a_impl, 'b': openfocus_b_impl, 'c': openfocus_c_impl}[algo]
    cpu = fuse(frames, use_gpu=False)
    devices = []
    gpu = fuse(frames, use_gpu=True, device_callback=devices.append)
    assert devices == [device_name]

    diff = np.abs(cpu.astype(np.int32) - gpu.astype(np.int32))
    assert int(diff.max()) <= 1


def test_openfocus_b_gpu_direct_u8_readback_and_ping_pong_parity(monkeypatch):
    _require_opencl_gpu()
    rng = np.random.default_rng(202)
    frames = [rng.integers(0, 255, (96, 128, 3), dtype=np.uint8) for _ in range(8)]
    readback_calls = []
    orig_u8 = _openfocus_gpu.Merge.finish_merge_u8

    def tracked_u8(self, output):
        readback_calls.append(output.dtype)
        return orig_u8(self, output)

    monkeypatch.setattr(_openfocus_gpu.Merge, 'finish_merge_u8', tracked_u8)
    cpu = openfocus_b_impl(frames, use_gpu=False)
    gpu = openfocus_b_impl(frames, use_gpu=True)
    assert readback_calls == [np.dtype(np.uint8)]
    assert int(np.abs(cpu.astype(np.int32) - gpu.astype(np.int32)).max()) <= 1


def test_openfocus_b_cancelled_active_scan_releases_resident_merge(monkeypatch):
    _require_opencl_gpu()
    from fusion_methods import openfocus_bc

    stop = threading.Event()
    active_sources = openfocus_bc._active_sources
    begin_merge = _openfocus_gpu.Merge.begin_merge
    active_merges = []

    def begin(merge, *args):
        admitted = begin_merge(merge, *args)
        if admitted:
            active_merges.append(merge)
        return admitted

    def cancel_scan(*args, **kwargs):
        assert active_merges, 'Resident OpenCL merge was not admitted'
        stop.set()
        return active_sources(*args, **kwargs)

    monkeypatch.setattr(_openfocus_gpu.Merge, 'begin_merge', begin)
    monkeypatch.setattr(openfocus_bc, '_active_sources', cancel_scan)
    depth = np.zeros((96, 128), dtype=np.float32)
    frames = [np.full((96, 128, 3), level, np.uint8) for level in (32, 128)]
    lock = _openfocus_gpu._device_lock
    try:
        with pytest.raises(registration.RegistrationCancelled):
            openfocus_b_impl(frames, depth_override=depth, use_gpu=True,
                             should_cancel=stop.is_set)
        assert lock.acquire(blocking=False), 'An OpenCL job remains locked after cancellation'
        lock.release()
    finally:
        if lock.locked() and active_merges:
            active_merges[-1].end_merge()


def test_active_sources_parallel_matches_tiled():
    from fusion_methods.openfocus_bc import _MERGE_TILE, _active_sources

    rng = np.random.default_rng(303)
    count = 16
    depth = (rng.integers(0, count, (300, 300), dtype=np.uint16) * (65535 // (count - 1))).astype(np.uint16)
    bands = (depth.shape[0] + _MERGE_TILE - 1) // _MERGE_TILE
    cols = (depth.shape[1] + _MERGE_TILE - 1) // _MERGE_TILE
    tiles = np.zeros((bands, cols, count), dtype=bool)
    tiled = _active_sources(depth, count, None, tiles=tiles)
    parallel = _active_sources(depth, count, None, tiles=None)
    np.testing.assert_array_equal(parallel, tiled)


def test_adaptive_frame_ram_budget_scales_by_machine_tier(monkeypatch):
    GiB = 1024 ** 3
    MiB = 1024 ** 2

    snap_8g_tight = resource_policy.ResourceSnapshot(
        available_ram=1600 * MiB, total_ram=8 * GiB, physical_cores=4, logical_cpus=8,
    )
    snap_16g = resource_policy.ResourceSnapshot(
        available_ram=10 * GiB, total_ram=16 * GiB, physical_cores=8, logical_cpus=16,
    )
    snap_32g = resource_policy.ResourceSnapshot(
        available_ram=20 * GiB, total_ram=32 * GiB, physical_cores=12, logical_cpus=24,
    )
    snap_64g = resource_policy.ResourceSnapshot(
        available_ram=48 * GiB, total_ram=64 * GiB, physical_cores=16, logical_cpus=32,
    )
    snap_128g = resource_policy.ResourceSnapshot(
        available_ram=96 * GiB, total_ram=128 * GiB, physical_cores=32, logical_cpus=64,
    )

    assert resource_policy.frame_ram_budget(snap_8g_tight) < 384 * MiB
    assert resource_policy.frame_ram_budget(snap_16g) == 384 * MiB
    assert resource_policy.frame_ram_budget(snap_32g) == 1024 * MiB
    assert resource_policy.frame_ram_budget(snap_64g) == 2048 * MiB
    assert resource_policy.frame_ram_budget(snap_128g) == 2048 * MiB

    monkeypatch.setattr(resource_policy, 'probe_memory', lambda: snap_64g)
    proc_64g = resource_policy.make_process_budget(snapshot=snap_64g)
    assert proc_64g.frame_ram == 2048 * MiB

    monkeypatch.setattr(resource_policy, 'FRAME_RAM_BUDGET_BYTES', 256 * MiB)
    assert resource_policy.frame_ram_budget(snap_64g) == 256 * MiB


def test_adaptive_source_cache_trims_on_low_ram_and_grows_on_high_ram():
    from core.source_frames import FrameCache

    GiB = 1024 ** 3
    MiB = 1024 ** 2
    cache = FrameCache()

    for idx in range(2):
        cache.put(('frame', idx), np.empty((40, 1024, 1024), dtype=np.uint8))
    assert cache.bytes == 80 * MiB

    low_snap = resource_policy.ResourceSnapshot(
        available_ram=1500 * MiB, total_ram=8 * GiB, physical_cores=4, logical_cpus=8,
    )
    adapted_low = cache.adapt_to_memory(low_snap)
    assert adapted_low == 64 * MiB
    assert cache.max_bytes == 64 * MiB
    assert cache.bytes <= 64 * MiB
    depleted = replace(low_snap, available_ram=MiB)
    assert cache.adapt_to_memory(depleted) == 0
    assert cache.bytes == 0

    high_snap = resource_policy.ResourceSnapshot(
        available_ram=48 * GiB, total_ram=64 * GiB, physical_cores=16, logical_cpus=32,
    )
    adapted_high = cache.adapt_to_memory(high_snap)
    assert adapted_high == GiB
    assert cache.max_bytes == GiB

    pinned = FrameCache(0)
    assert pinned.adapt_to_memory(high_snap) == 0

    # Explicit max_bytes override pins capacity and is not overwritten.
    cache.max_bytes = 128 * MiB
    assert cache.adapt_to_memory(high_snap) == 128 * MiB


def test_openfocus_alignment_low_ram_spill_degrade_and_zero_budget_reject(monkeypatch, tmp_path):
    from core import frame_storage

    GiB = 1024 ** 3
    MiB = 1024 ** 2

    first = np.random.default_rng(37).integers(0, 256, size=(1500, 1500), dtype=np.uint8)
    sources = [first, np.roll(first, 2, axis=1)]
    parameters = {'max_shift_ratio': 0.01, 'magnification': 0.0}
    expected = openfocus_alignment.estimate_alignment_plan(sources, **parameters)
    monkeypatch.setattr(resource_policy, 'FRAME_RAM_BUDGET_BYTES', 1 * MiB)
    reserve_16g = max(
        resource_policy.RESERVE_BYTES,
        min(int(resource_policy.RESERVE_RATIO * (16 * GiB)), resource_policy.RESERVE_CAP_BYTES),
    )
    tight_avail = reserve_16g + 40 * MiB
    tight_snap = resource_policy.ResourceSnapshot(
        available_ram=tight_avail, total_ram=16 * GiB, physical_cores=8, logical_cpus=16,
    )
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda kind='CPU': tight_snap)
    monkeypatch.setattr(resource_policy, 'probe_memory', lambda: tight_snap)

    mapped_shapes = set()
    create_transient = frame_storage.create_transient_frame

    def capture_mapping(shape, dtype):
        plane = create_transient(shape, dtype)
        if frame_storage.is_mapped_frame(plane):
            mapped_shapes.add(shape)
        return plane

    monkeypatch.setattr(frame_storage, 'create_transient_frame', capture_mapping)
    with frame_storage.scratch_directory(explicit_directory=tmp_path):
        plan = openfocus_alignment.estimate_alignment_plan(sources, **parameters)
    assert plan == expected
    assert {(1500, 1500), (750, 750)} <= mapped_shapes

    zero_snap = resource_policy.ResourceSnapshot(
        available_ram=0, total_ram=16 * GiB, physical_cores=8, logical_cpus=16,
        commit_headroom=0,
    )
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda kind='CPU': zero_snap)
    monkeypatch.setattr(resource_policy, 'probe_memory', lambda: zero_snap)
    with pytest.raises(resource_policy.ResourceLimitError):
        openfocus_alignment.estimate_alignment_plan(sources)

    # Preflight on 8K and 12K with 6 GiB free on 16 GiB laptop reports no false RAM warning.
    snap_ok = resource_policy.ResourceSnapshot(
        available_ram=6 * GiB, total_ram=16 * GiB, physical_cores=8, logical_cpus=16,
    )
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda kind='CPU': snap_ok)
    monkeypatch.setattr(resource_policy, 'probe_memory', lambda: snap_ok)
    frame_12k = SourceFrame(
        'frame_12k.png', 0, (6480, 11520, 3), np.dtype('uint8'), None, (18 * MiB, 0),
    )
    est_12k = estimate_render_memory([frame_12k] * 100, 'openfocus_b', {}, register=True, alignment='openfocus')
    assert not est_12k.ram_short


def test_openfocus_alignment_releases_source_cache_before_rejecting_low_ram(monkeypatch, tmp_path):
    from core import frame_storage, source_frames
    from utils import logging_setup

    GiB = 1024 ** 3
    first = np.random.default_rng(41).integers(0, 256, size=(384, 384), dtype=np.uint8)
    sources = [first, np.roll(first, 2, axis=1)]
    parameters = {'max_shift_ratio': 0.02, 'magnification': 0.0}
    expected = openfocus_alignment.estimate_alignment_plan(sources, **parameters)

    roomy = resource_policy.ResourceSnapshot(
        available_ram=6 * GiB, total_ram=16 * GiB, physical_cores=8, logical_cpus=16,
    )
    reserve = max(
        resource_policy.RESERVE_BYTES,
        min(int(resource_policy.RESERVE_RATIO * roomy.total_ram), resource_policy.RESERVE_CAP_BYTES),
    )
    # Known RAM with nothing above the desktop reserve until cached frames are released.
    squeezed = replace(roomy, available_ram=reserve)
    assert squeezed.ram_known and resource_policy.ram_ceiling(squeezed) == 0
    released = []
    clear = source_frames.SOURCE_CACHE.clear

    def release():
        released.append(True)
        clear()

    monkeypatch.setattr(source_frames.SOURCE_CACHE, 'clear', release)
    probe = lambda *args, **kwargs: roomy if released else squeezed
    monkeypatch.setattr(resource_policy, 'probe_resources', probe)
    monkeypatch.setattr(resource_policy, 'probe_memory', probe)
    events = []
    monkeypatch.setattr(logging_setup, 'log_action', lambda event, **kw: events.append(event))
    with frame_storage.scratch_directory(explicit_directory=tmp_path):
        plan = openfocus_alignment.estimate_alignment_plan(sources, **parameters)
    assert released
    assert 'openfocus_alignment_low_ram_degrade' in events
    assert plan == expected

    # Unknown RAM still rejects at once and keeps the cached frames.
    released.clear()
    squeezed = resource_policy.ResourceSnapshot()
    with pytest.raises(resource_policy.ResourceLimitError):
        openfocus_alignment.estimate_alignment_plan(sources, **parameters)
    assert not released


def test_openfocus_b_unmapped_source_admits_first_pass_without_degrade(tmp_path, monkeypatch):
    from core import multi_focus_fusion as mff, source_frames
    from core.algorithm_adapters import openfocus_b_resources

    MiB = 1024 ** 2
    GiB = 1024 ** 3
    p0 = tmp_path / 'u0.png'
    p1 = tmp_path / 'u1.png'
    arr = np.zeros((2000, 2000, 3), dtype=np.uint8)
    cv2.imwrite(str(p0), arr)
    cv2.imwrite(str(p1), arr)
    sources = [scan_source(str(p0), header_only=True)[0], scan_source(str(p1), header_only=True)[0]]

    # Admit the spillable streaming pass while the unclamped resident estimate
    # would force a full disk staging pass.
    cap = 16 * MiB
    monkeypatch.setattr(resource_policy, 'FRAME_RAM_BUDGET_BYTES', cap)
    fixed, _ = openfocus_b_resources(2000, 2000, {}, len(sources))
    decode = max(source_frames.frame_resolution_bytes(source) for source in sources)
    snap = resource_policy.ResourceSnapshot(
        available_ram=6 * GiB, total_ram=8 * GiB, physical_cores=4, logical_cpus=8,
        commit_headroom=fixed + cap + decode + MiB,
    )
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda kind='CPU': snap)
    monkeypatch.setattr(resource_policy, 'probe_memory', lambda: snap)

    def unexpected_staging(*args, **kwargs):
        pytest.fail('OpenFocus B can stream these sources with the available memory')

    monkeypatch.setattr(source_frames, 'materialize_for_repeated_access', unexpected_staging)

    events = []
    monkeypatch.setattr(mff, 'log_action', lambda event, **kw: events.append(event))
    calls = []
    orig_plan = resource_policy.plan_execution

    def spy_plan(workload, **kwargs):
        if workload.algorithm == 'openfocus_b':
            calls.append(workload.per_worker_bytes)
        return orig_plan(workload, **kwargs)

    monkeypatch.setattr(resource_policy, 'plan_execution', spy_plan)
    fusion = MultiFocusFusion(algorithm='openfocus_b', use_gpu=False, tile_enabled=False, exposure_match=False)
    fusion.align_images = lambda *args, **kwargs: None
    result = fusion.fuse(sources, thread_count=1)
    assert result.shape == (2000, 2000, 3)
    assert calls == [cap + decode]
    assert 'openfocus_low_ram_degrade' not in events


def test_openfocus_a_spills_background_to_admit_next_decode(monkeypatch, tmp_path):
    from core import frame_storage, source_frames
    from fusion_methods import openfocus_a

    first = np.full((128, 128, 3), 88, np.uint8)
    second = first.copy()
    expected = openfocus_a_impl([first, second], use_gpu=False, thread_count=1).copy()
    ceiling = 64 * 1024 ** 2
    decode_peak = ceiling - first.shape[0] * first.shape[1] * 20
    baseline_ram = frame_storage.ram_snapshot_bytes()

    def probe(kind='CPU'):
        return resource_policy.ResourceSnapshot(
            available_ram=6 * 1024 ** 3, total_ram=8 * 1024 ** 3,
            commit_headroom=ceiling - (frame_storage.ram_snapshot_bytes() - baseline_ram),
        )

    def decode(frame, *, should_cancel=None):
        if frame is second:
            resource_policy.require_memory(decode_peak, probe())
        return frame

    monkeypatch.setattr(openfocus_a, '_DEFAULT_BACKGROUND_RAM_LIMIT', 0)
    monkeypatch.setattr(openfocus_a, '_BACKGROUND_RAM_LIMIT_BYTES', 0)
    monkeypatch.setattr(resource_policy, 'probe_memory', probe)
    monkeypatch.setattr(resource_policy, 'probe_resources', probe)
    monkeypatch.setattr(source_frames, 'frame_resolution_bytes', lambda frame: decode_peak)
    monkeypatch.setattr(source_frames, 'resolve_frame', decode)
    with frame_storage.scratch_directory(explicit_directory=tmp_path):
        actual = openfocus_a_impl([first, second], use_gpu=False, thread_count=1)
    np.testing.assert_array_equal(actual, expected)
