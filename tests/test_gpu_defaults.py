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
