"""Tests for GPU-first defaults, discrete OpenCL device ranking, tiled GPU depth, and buffer release."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np


def test_opencl_device_prefers_discrete_over_integrated():
    from fusion_methods._openfocus_gpu import _device

    integrated = SimpleNamespace(
        type=4, available=True, compiler_available=True,
        host_unified_memory=True, global_mem_size=8 * 1024**3, max_compute_units=32,
        name='Integrated GPU',
    )
    discrete_small = SimpleNamespace(
        type=4, available=True, compiler_available=True,
        host_unified_memory=False, global_mem_size=4 * 1024**3, max_compute_units=16,
        name='Discrete Small',
    )
    discrete_large = SimpleNamespace(
        type=4, available=True, compiler_available=True,
        host_unified_memory=False, global_mem_size=8 * 1024**3, max_compute_units=24,
        name='Discrete Large',
    )
    platforms = [
        SimpleNamespace(get_devices=lambda device_type: [integrated]),
        SimpleNamespace(get_devices=lambda device_type: [discrete_small, discrete_large]),
    ]
    cl = SimpleNamespace(
        get_platforms=lambda: platforms,
        device_type=SimpleNamespace(GPU=4),
        Error=RuntimeError,
    )
    assert _device(cl) is discrete_large


def test_opencl_device_handles_minimal_mock():
    from fusion_methods._openfocus_gpu import _device

    minimal = SimpleNamespace(type=4, available=True, compiler_available=True, name='Minimal')
    cl = SimpleNamespace(
        get_platforms=lambda: [SimpleNamespace(get_devices=lambda device_type: [minimal])],
        device_type=SimpleNamespace(GPU=4),
        Error=RuntimeError,
    )
    assert _device(cl) is minimal


def _fake_opencl(*devices):
    return SimpleNamespace(
        get_platforms=lambda: [SimpleNamespace(get_devices=lambda device_type: list(devices))],
        device_type=SimpleNamespace(GPU=4),
        Error=RuntimeError,
    )


def _fake_gpu(name, gib, *, unified=False):
    return SimpleNamespace(
        type=4, available=True, compiler_available=True, host_unified_memory=unified,
        global_mem_size=gib * 1024**3, max_compute_units=32, name=name,
    )


def _record_free_vram(monkeypatch, readings):
    from fusion_methods import _openfocus_gpu

    queried = []

    def free_vram(device):
        queried.append(device.name)
        reading = readings[device.name]
        if isinstance(reading, Exception):
            raise reading
        return reading

    monkeypatch.setattr(_openfocus_gpu, '_free_vram', free_vram)
    return queried


def test_opencl_device_prefers_discrete_gpu_with_free_vram_reading(monkeypatch):
    from fusion_methods._openfocus_gpu import _device

    unread = _fake_gpu('Discrete 16 GB', 16)
    readable = _fake_gpu('Discrete 12 GB', 12)
    integrated = _fake_gpu('Integrated', 32, unified=True)
    queried = _record_free_vram(monkeypatch, {
        unread.name: None, readable.name: (6 * 1024**3, 12 * 1024**3)})
    assert _device(_fake_opencl(integrated, unread, readable)) is readable
    assert queried == [unread.name, readable.name]


def test_opencl_device_keeps_memory_order_among_readable_discrete_gpus(monkeypatch):
    from fusion_methods._openfocus_gpu import _device

    small = _fake_gpu('Discrete 8 GB', 8)
    large = _fake_gpu('Discrete 12 GB', 12)
    queried = _record_free_vram(monkeypatch, {
        small.name: (4 * 1024**3, 8 * 1024**3), large.name: (1 * 1024**3, 12 * 1024**3)})
    assert _device(_fake_opencl(small, large)) is large
    assert len(queried) <= 1


def test_opencl_device_keeps_memory_order_without_any_discrete_reading(monkeypatch):
    from fusion_methods._openfocus_gpu import _device

    small = _fake_gpu('Discrete 8 GB', 8)
    large = _fake_gpu('Discrete 16 GB', 16)
    queried = _record_free_vram(monkeypatch, {
        small.name: None, large.name: RuntimeError('driver query failed')})
    assert _device(_fake_opencl(small, large)) is large
    assert len(queried) == len(set(queried))


def test_opencl_device_reads_no_vram_without_a_second_discrete_gpu(monkeypatch):
    from fusion_methods._openfocus_gpu import _device

    discrete = _fake_gpu('Discrete 16 GB', 16)
    integrated = _fake_gpu('Integrated', 32, unified=True)
    queried = _record_free_vram(monkeypatch, {discrete.name: None, integrated.name: None})
    assert _device(_fake_opencl(discrete)) is discrete
    assert _device(_fake_opencl(integrated, discrete)) is discrete
    assert queried == []


def test_gpu_first_api_defaults():
    from core.background_fill import _replace_background, focus_measures
    from core.depth_map import compute_focus_index
    from core.multi_focus_fusion import MultiFocusFusion, fuse_images
    from core.registration import ImageRegistration, _align_ecc_impl, _align_homography_impl
    from fusion_methods.openfocus_a import openfocus_a_impl
    from fusion_methods.openfocus_bc import openfocus_b_impl, openfocus_c_impl
    from fusion_methods.openfocus_depth import depth_map
    from utils.retouch import RetouchDocument

    assert MultiFocusFusion().use_gpu is True
    reg = ImageRegistration('ecc')
    assert reg.accelerate_ecc is True
    assert reg.use_gpu is False

    for fn, param in (
        (fuse_images, 'use_gpu'),
        (openfocus_a_impl, 'use_gpu'),
        (openfocus_b_impl, 'use_gpu'),
        (openfocus_c_impl, 'use_gpu'),
        (depth_map, 'use_gpu'),
        (compute_focus_index, 'use_gpu'),
        (_align_ecc_impl, 'accelerate_ecc'),
        (_align_homography_impl, 'accelerate_warp'),
        (focus_measures, 'use_gpu'),
        (_replace_background, 'use_gpu'),
        (RetouchDocument.auto_background, 'use_gpu'),
    ):
        assert inspect.signature(fn).parameters[param].default is True, f'{fn.__name__}.{param}'


def test_release_buffers_handles_nested_duplicates_and_errors():
    from fusion_methods._openfocus_gpu import _PreviewPlaneReader, _release_buffers

    calls = []

    class DummyBuffer:
        def __init__(self, fail=False):
            self.fail = fail

        def release(self):
            calls.append(id(self))
            if self.fail:
                raise RuntimeError('already released')

    shared = DummyBuffer()
    broken = DummyBuffer(fail=True)
    _release_buffers([shared, (shared, broken), {'a': shared, 'b': None}])
    assert len(calls) == 2

    reader = _PreviewPlaneReader(None, None, None, None, (8, 8), np.uint16)
    reader._buffer = DummyBuffer()
    reader.close()
    assert reader._buffer is None
    assert len(calls) == 3


def test_tiled_openfocus_b_forwards_use_gpu_to_global_depth(monkeypatch):
    from core import multi_focus_fusion as mff
    from fusion_methods import openfocus_depth

    seen = {}

    def fake_depth_map(images, *args, **kwargs):
        seen['use_gpu'] = kwargs.get('use_gpu')
        seen['device_callback'] = kwargs.get('device_callback')
        h, w = images[0].shape[:2]
        return np.zeros((h, w), dtype=np.uint16)

    monkeypatch.setattr(openfocus_depth, 'depth_map', fake_depth_map)
    frames = [np.full((48, 48, 3), 100 + i * 10, dtype=np.uint8) for i in range(3)]
    fusion = mff.MultiFocusFusion(
        algorithm='openfocus_b',
        use_gpu=True,
        exposure_match=False,
        tile_enabled=True,
        tile_block_size=32,
        tile_overlap=8,
        tile_threshold=16,
    )
    out = fusion.fuse(frames, thread_count=1)
    assert out.shape == (48, 48, 3)
    assert seen['use_gpu'] is True
    assert callable(seen['device_callback'])


def test_tiled_gpu_caps_worker_pool_to_two(monkeypatch):
    from core import multi_focus_fusion as mff
    from fusion_methods import openfocus_a

    worker_counts = []
    monkeypatch.setattr(mff, 'ensure_openfocus_gpu_available', lambda: True)
    monkeypatch.setattr(openfocus_a, 'openfocus_a_impl',
                        lambda crops, *a, **kw: np.zeros(crops[0].shape, dtype=np.uint8))

    orig_pool = __import__('concurrent.futures').futures.ThreadPoolExecutor

    class TrackingPool(orig_pool):
        def __init__(self, max_workers=None, *args, **kwargs):
            worker_counts.append(max_workers)
            super().__init__(max_workers=max_workers, *args, **kwargs)

    monkeypatch.setattr(__import__('concurrent.futures').futures, 'ThreadPoolExecutor', TrackingPool)
    frames = [np.full((64, 64, 3), 80 + i * 5, dtype=np.uint8) for i in range(3)]
    fusion = mff.MultiFocusFusion(
        algorithm='openfocus_a',
        use_gpu=True,
        exposure_match=False,
        tile_enabled=True,
        tile_block_size=24,
        tile_overlap=4,
        tile_threshold=16,
    )
    fusion._calculate_optimal_thread_count = lambda *a, **kw: 8
    # Trigger tiled path using monkeypatched resource_policy worker count if present
    out = fusion.fuse(frames, thread_count=8)
    assert out.shape == (64, 64, 3)
    assert worker_counts and max(worker_counts) <= 2
