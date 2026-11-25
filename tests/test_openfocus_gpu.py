"""OpenFocus A/B/C device selection and deterministic CPU/OpenCL parity."""
import numpy as np
import pytest

from core.multi_focus_fusion import MultiFocusFusion
from core.source_frames import process_frame


def _stack(dtype, channels):
    rng = np.random.default_rng(4019)
    shape = (65, 67) if channels == 1 else (65, 67, channels)
    maximum = np.iinfo(dtype).max
    return [rng.integers(0, maximum + 1, shape, dtype=dtype) for _ in range(3)]


def test_opencl_status_uses_the_same_selected_adapter_as_render_jobs(monkeypatch):
    import sys
    from types import SimpleNamespace
    from core import multi_focus_fusion as mff
    from fusion_methods import _openfocus_gpu

    devices = [SimpleNamespace(type=4, available=True, compiler_available=False,
                               name='GPU without a compiler'),
               SimpleNamespace(type=4, available=True, compiler_available=True,
                               name=' Intel Arc A380 ')]
    platform = SimpleNamespace(get_devices=lambda device_type: devices)
    cl = SimpleNamespace(get_platforms=lambda: [platform],
                         device_type=SimpleNamespace(GPU=4), Error=RuntimeError)
    monkeypatch.setitem(sys.modules, 'pyopencl', cl)
    monkeypatch.setattr(mff, '_OPENFOCUS_GPU_NAME', None)
    assert mff.openfocus_gpu_available() is None
    assert mff.ensure_openfocus_gpu_available() is True
    name = _openfocus_gpu.opencl_device_name(_openfocus_gpu._device(cl))
    gpu = SimpleNamespace(used=True, failed=True, device_name=name)
    assert mff.openfocus_gpu_name() == gpu.device_name == 'OpenCL: Intel Arc A380'
    assert mff.short_render_device_name(_openfocus_gpu.device_status(gpu)) == 'CPU + Intel Arc A380'
    assert _openfocus_gpu.opencl_device_name(SimpleNamespace()) == 'OpenCL'
    assert mff.short_render_device_name('OpenCL') == 'OpenCL'


@pytest.mark.parametrize('dtype,channels,resize,low_detail', [
    (np.uint8, 1, None, False),
    (np.uint8, 3, (51, 49), False),
    (np.uint8, 3, None, True),
    (np.uint16, 1, None, False),
    (np.uint16, 3, None, False),
    (np.uint16, 3, None, True),
    (np.float32, 3, None, False),
])
def test_openfocus_a_gpu_matches_cpu_and_reports_device(
        dtype, channels, resize, low_detail, monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),
                                        int(device.global_mem_size)))
    rng = np.random.default_rng(4019)
    shape = (65, 67) if channels == 1 else (65, 67, channels)
    if low_detail:
        delta = 3 if dtype == np.uint8 else 40
        frames = [rng.integers(100, 100 + delta, shape, dtype=dtype)
                  for _ in range(5)]
    elif dtype == np.float32:
        frames = [rng.uniform(0, 255, shape).astype(dtype) for _ in range(3)]
    else:
        frames = _stack(dtype, channels)
    lazy = [process_frame(frame, lambda image: image.copy()) for frame in frames]
    options = dict(algorithm='openfocus_a', exposure_match=False,
                   base_frame='none', sort_mode='off')
    cpu = MultiFocusFusion(use_gpu=False, **options)
    gpu = MultiFocusFusion(use_gpu=True, **options)
    expected = cpu.fuse(lazy, img_resize=resize, thread_count=1, radius=8)
    actual = gpu.fuse(lazy, img_resize=resize, thread_count=1, radius=8)
    assert cpu.get_info()['device'] == 'CPU'
    assert gpu.get_info()['device'].startswith('OpenCL: ')
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('shape', [(1, 2, 4), (1, 130, 4), (130, 1, 4)])
def test_openfocus_a_gpu_mixed_sources_match_cpu_on_thin_pyramids(shape, monkeypatch):
    from fusion_methods import _openfocus_gpu
    from fusion_methods.openfocus_a import openfocus_a_impl

    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),) * 2)
    rng = np.random.default_rng(4020)
    frame8 = rng.integers(119, 124, shape, dtype=np.uint8)
    frame16 = rng.integers(30705, 31324, shape, dtype=np.uint16)
    frame_float = rng.uniform(119, 124, shape).astype(np.float32)
    frames = [frame8, frame16, frame_float, frame8.copy()]
    devices = []
    expected = openfocus_a_impl(frames, thread_count=1, kernel_size=129)
    actual = openfocus_a_impl(frames, thread_count=1, kernel_size=129,
                              use_gpu=True, device_callback=devices.append)
    assert devices[0].startswith('OpenCL: ')
    np.testing.assert_array_equal(actual, expected)


def test_openfocus_a_missing_opencl_falls_back(monkeypatch):
    import builtins
    frames = _stack(np.uint8, 3)
    options = dict(algorithm='openfocus_a', exposure_match=False,
                   base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    original_import = builtins.__import__

    def without_opencl(name, *args, **kwargs):
        if name == 'pyopencl':
            raise ModuleNotFoundError('pyopencl unavailable')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_opencl)
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(frames, thread_count=1)
    assert fusion.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


def test_openfocus_a_device_failure_falls_back_with_monotonic_progress(monkeypatch):
    from types import SimpleNamespace
    from fusion_methods._openfocus_gpu_a import GpuA

    class DeviceError(Exception):
        pass

    def initialize(self):
        self.cl = SimpleNamespace(Error=DeviceError)

    def fail(self, frames, resize, radius, cancel, progress):
        progress(1, len(frames))
        raise DeviceError('device lost')

    frames = _stack(np.uint8, 3)
    options = dict(algorithm='openfocus_a', exposure_match=False,
                   base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    monkeypatch.setattr(GpuA, '__init__', initialize)
    monkeypatch.setattr(GpuA, '_fuse_locked', fail)
    progress = []
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(frames, thread_count=1,
                         progress_callback=lambda current, total: progress.append(current))
    assert progress == sorted(progress)
    assert progress[-1] == len(frames)
    assert fusion.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('maximum_mib,unified,reason', [
    (640, False, None),
    (500, False, 'host memory budget'),
    (640, True, 'device budget'),
])
def test_openfocus_a_admits_one_scaled_ecc_source_under_job_cap(
        tmp_path, monkeypatch, maximum_mib, unified, reason):
    from types import SimpleNamespace
    from core import resource_policy, source_frames
    from fusion_methods import _openfocus_gpu
    from fusion_methods._openfocus_gpu_a import GpuA

    # Test admission alone, with the reported stack geometry and a cold source
    # cache. No pixels or OpenCL allocations are needed to reach the decision.
    path = tmp_path / 'stack.tif'
    path.touch()
    stamp = source_frames._fingerprint(path)
    shape = (1366, 2048, 3)
    frames = [process_frame(source_frames.SourceFrame(
        str(path), page, shape, np.dtype(np.uint8), None, stamp, scale=.25,
        native_shape=(5464, 8192, 3)),
        lambda _: pytest.fail('Admission must not resolve source pixels')) for page in range(92)]
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache())
    mib = 1024 ** 2
    monkeypatch.setattr(resource_policy, 'probe_memory', lambda: resource_policy.ResourceSnapshot(
        available_ram=8 * 1024 * mib, total_ram=16 * 1024 * mib))
    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda device: (8 * 1024 * mib,) * 2)

    class AllocationReached(Exception):
        pass

    def allocate(*args):
        raise AllocationReached

    gpu = GpuA.__new__(GpuA)
    gpu.cl = SimpleNamespace(Buffer=allocate)
    gpu.ctx = None
    gpu.mf = SimpleNamespace(READ_WRITE=1)
    gpu.device = SimpleNamespace(global_mem_size=8 * 1024 * mib,
                                 max_mem_alloc_size=2 * 1024 * mib,
                                 host_unified_memory=unified)
    envelope = resource_policy.ProcessBudget(384 * mib, maximum_mib * mib, 192 * mib)
    with resource_policy.process_budget(envelope):
        if reason is None:
            with pytest.raises(AllocationReached):
                gpu._fuse_locked(frames, None, 8, None, None)
        else:
            with pytest.raises(_openfocus_gpu.GpuError, match=reason):
                gpu._fuse_locked(frames, None, 8, None, None)


@pytest.mark.parametrize('disk_reuse', [False, True])
def test_openfocus_a_gpu_lazy_ecc_matches_cpu_without_source_cache_growth(
        tmp_path, monkeypatch, disk_reuse):
    from contextlib import nullcontext
    import cv2
    from core import frame_resolution_cache, frame_storage, registration, resource_policy, source_frames
    from fusion_methods import _openfocus_gpu

    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),) * 2)
    cache = source_frames.FrameCache()
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', cache)
    eager, lazy = [], []
    for index, pixels in enumerate(_stack(np.uint8, 3)):
        path = tmp_path / f'ecc-{index}.png'
        ok, encoded = cv2.imencode('.png', pixels)
        assert ok
        encoded.tofile(str(path))
        source = source_frames.scan_source(str(path), header_only=True)[0]
        identity = registration._perspective_source(source, pixels, None)
        matrix = np.float32([[1, 0, index * .6], [0, 1, -.35 * index], [0, 0, 1]])
        warp = registration.PerspectiveWarp(identity, tuple(matrix.ravel()), matrix.dtype.str,
                                            (pixels.shape[1], pixels.shape[0]))
        eager.append(warp(pixels))
        lazy.append(process_frame(source, warp))

    options = dict(algorithm='openfocus_a', exposure_match=True, base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(eager, thread_count=1)
    disk = (frame_resolution_cache.ResolutionCache(3 * sum(frame.nbytes for frame in lazy))
            if disk_reuse else None)
    use_cache = frame_resolution_cache.active(disk) if disk is not None else nullcontext()
    gpu = MultiFocusFusion(use_gpu=True, **options)
    ram_usage = []
    mib = 1024 ** 2
    envelope = resource_policy.ProcessBudget(64 * mib, 96 * mib, 2 * mib)
    try:
        with frame_storage.scratch_directory(explicit_directory=str(tmp_path)), use_cache, \
                resource_policy.process_budget(envelope):
            actual = gpu.fuse(lazy, thread_count=1,
                              progress_callback=lambda *_: ram_usage.append(cache.bytes))
            if disk is not None:
                assert all(disk.get(frame) is not None for frame in lazy)
        assert gpu.get_info()['device'].startswith('OpenCL: ')
        # With disk disabled only the exposure reference was decoded before
        # Method A began its streaming upload.
        assert ram_usage and set(ram_usage) == {0 if disk_reuse else lazy[0].source.nbytes}
        np.testing.assert_array_equal(actual, expected)
    finally:
        if disk is not None:
            disk.close()


def test_openfocus_a_unknown_gpu_memory_falls_back(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    device = _openfocus_gpu._device(__import__('pyopencl'))
    if device.host_unified_memory:
        pytest.skip('Discrete GPU required')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda device: None)
    frames = _stack(np.uint8, 3)
    options = dict(algorithm='openfocus_a', exposure_match=False,
                   base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(frames, thread_count=1)
    assert fusion.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('channels', [1, 3])
@pytest.mark.parametrize('resize', [None, (51, 49)])
def test_opencl_matches_cpu_for_lazy_sources(dtype, channels, resize):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    frames = _stack(dtype, channels)
    lazy = [process_frame(frame, lambda image: image.copy()) for frame in frames]
    cpu = MultiFocusFusion(algorithm='openfocus_b', use_gpu=False, exposure_match=False)
    gpu = MultiFocusFusion(algorithm='openfocus_b', use_gpu=True, exposure_match=False)
    expected = cpu.fuse(lazy, img_resize=resize, thread_count=1)
    actual = gpu.fuse(lazy, img_resize=resize, thread_count=1)
    assert cpu.get_info()['device'] == 'CPU'
    assert gpu.get_info()['device'] != 'CPU', 'GPU job silently fell back to CPU'
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('dtype,channels,resize,smoothing,software_divide', [
    (np.uint8, 1, None, 0., False),
    (np.uint8, 3, (51, 49), 4., False),
    (np.uint16, 1, None, 4., False),
    (np.uint16, 3, None, 0., False),
    (np.float32, 3, None, 4., False),
    pytest.param(np.uint16, 3, None, 0., True, id='uint16-software-division'),
])
def test_openfocus_c_gpu_matches_cpu_and_reports_device(
        dtype, channels, resize, smoothing, software_divide, monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    if software_divide:
        from fusion_methods import _openfocus_gpu_c

        monkeypatch.setattr(_openfocus_gpu_c, '_PROGRAM_CACHE', None)
        monkeypatch.setattr(_openfocus_gpu, '_fp32_build_options', lambda cl, device: [])
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),
                                        int(device.global_mem_size)))
    if dtype == np.float32:
        rng = np.random.default_rng(4019)
        frames = [rng.uniform(0, 255, (65, 67, channels)).astype(dtype)
                  for _ in range(3)]
    else:
        frames = _stack(dtype, channels)
    lazy = [process_frame(frame, lambda image: image.copy()) for frame in frames]
    options = dict(algorithm='openfocus_c', exposure_match=False,
                   base_frame='none', sort_mode='off')
    cpu = MultiFocusFusion(use_gpu=False, **options)
    gpu = MultiFocusFusion(use_gpu=True, **options)
    expected = cpu.fuse(lazy, img_resize=resize, thread_count=1, smoothing=smoothing)
    actual = gpu.fuse(lazy, img_resize=resize, thread_count=1, smoothing=smoothing)
    assert cpu.get_info()['device'] == 'CPU'
    assert gpu.get_info()['device'].startswith('OpenCL: ')
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('maximum_mib,unified,reason', [
    (640, False, None),
    (500, False, 'host memory budget'),
    (640, True, 'device budget'),
])
def test_openfocus_c_admits_one_scaled_source_under_job_cap(
        tmp_path, monkeypatch, maximum_mib, unified, reason):
    from types import SimpleNamespace
    from core import resource_policy, source_frames
    from fusion_methods import _openfocus_gpu
    from fusion_methods._openfocus_gpu_c import GpuC

    path = tmp_path / 'stack.tif'
    path.touch()
    stamp = source_frames._fingerprint(path)
    shape = (1366, 2048, 3)
    frames = [process_frame(source_frames.SourceFrame(
        str(path), page, shape, np.dtype(np.uint8), None, stamp, scale=.25,
        native_shape=(5464, 8192, 3)),
        lambda _: pytest.fail('Admission must not resolve source pixels')) for page in range(92)]
    mib = 1024 ** 2
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda: resource_policy.ResourceSnapshot(
        available_ram=8 * 1024 * mib, total_ram=16 * 1024 * mib))
    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda device: (8 * 1024 * mib,) * 2)

    class AllocationReached(Exception):
        pass

    def allocate(*args):
        raise AllocationReached

    gpu = GpuC.__new__(GpuC)
    gpu.cl = SimpleNamespace(Buffer=allocate)
    gpu.ctx = None
    gpu.mf = SimpleNamespace(READ_WRITE=1)
    gpu.device = SimpleNamespace(global_mem_size=8 * 1024 * mib,
                                 max_mem_alloc_size=2 * 1024 * mib,
                                 host_unified_memory=unified)
    envelope = resource_policy.ProcessBudget(384 * mib, maximum_mib * mib, 192 * mib)
    with resource_policy.process_budget(envelope):
        if reason is None:
            with pytest.raises(AllocationReached):
                gpu._fuse_locked(frames, shape, None, 4., None, None)
        else:
            with pytest.raises(_openfocus_gpu.GpuError, match=reason):
                gpu._fuse_locked(frames, shape, None, 4., None, None)


def test_openfocus_c_gpu_single_pixel_keeps_first_frame_and_focus_index(monkeypatch):
    from fusion_methods import _openfocus_gpu
    from fusion_methods.openfocus_bc import openfocus_c_impl

    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),
                                        int(device.global_mem_size)))
    frames = [np.array([[value]], dtype=np.uint8) for value in (201, 17, 63)]
    devices, focus = [], []
    actual = openfocus_c_impl(frames, use_gpu=True, thread_count=1,
                             device_callback=devices.append,
                             focus_callback=lambda positions, scale: focus.append((positions, scale)))

    assert devices[0].startswith('OpenCL: ')
    np.testing.assert_array_equal(actual, frames[0])
    [(positions, scale)] = focus
    assert positions.shape == (1, 1)
    np.testing.assert_array_equal(positions, 0)
    assert scale == len(frames) - 1


def test_openfocus_c_missing_opencl_falls_back(monkeypatch):
    import builtins
    frames = _stack(np.uint8, 3)
    options = dict(algorithm='openfocus_c', exposure_match=False,
                   base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    original_import = builtins.__import__

    def without_opencl(name, *args, **kwargs):
        if name == 'pyopencl':
            raise ModuleNotFoundError('pyopencl unavailable')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_opencl)
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(frames, thread_count=1)
    assert fusion.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


def test_openfocus_c_device_failure_falls_back_with_monotonic_progress(monkeypatch):
    from types import SimpleNamespace
    from fusion_methods._openfocus_gpu_c import GpuC

    class DeviceError(Exception):
        pass

    def initialize(self):
        self.cl = SimpleNamespace(Error=DeviceError)

    def fail(self, frames, shape, resize, smoothing, cancel, progress):
        progress(1, len(frames) + 1)
        raise DeviceError('device lost')

    frames = _stack(np.uint8, 3)
    options = dict(algorithm='openfocus_c', exposure_match=False,
                   base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    monkeypatch.setattr(GpuC, '__init__', initialize)
    monkeypatch.setattr(GpuC, '_fuse_locked', fail)
    progress = []
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(frames, thread_count=1,
                         progress_callback=lambda current, total: progress.append(current))
    assert progress == sorted(progress)
    assert progress[-1] == len(frames) + 1
    assert fusion.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


def test_openfocus_c_unknown_gpu_memory_falls_back(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    device = _openfocus_gpu._device(__import__('pyopencl'))
    if device.host_unified_memory:
        pytest.skip('Discrete GPU required')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda device: None)
    frames = _stack(np.uint8, 3)
    options = dict(algorithm='openfocus_c', exposure_match=False,
                   base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(frames, thread_count=1)
    assert fusion.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


def test_cpu_toggle_never_constructs_opencl_backend(monkeypatch):
    from fusion_methods import _openfocus_gpu
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU backend used while disabled')
    monkeypatch.setattr(_openfocus_gpu, 'Merge', forbidden)
    fusion = MultiFocusFusion(algorithm='openfocus_b', use_gpu=False, exposure_match=False)
    fusion.fuse(_stack(np.uint8, 3), thread_count=1)
    assert fusion.get_info()['device'] == 'CPU'


def test_opencl_failure_falls_back_to_cpu(monkeypatch):
    from fusion_methods import _openfocus_gpu
    def unavailable(*args, **kwargs):
        raise _openfocus_gpu.GpuError('unavailable')
    monkeypatch.setattr(_openfocus_gpu, 'Merge', unavailable)
    frames = _stack(np.uint8, 3)
    expected = MultiFocusFusion(algorithm='openfocus_b', use_gpu=False, exposure_match=False).fuse(frames, thread_count=1)
    fusion = MultiFocusFusion(algorithm='openfocus_b', use_gpu=True, exposure_match=False)
    actual = fusion.fuse(frames, thread_count=1)
    np.testing.assert_array_equal(actual, expected)
    assert fusion.get_info()['device'] == 'CPU'



def test_mid_job_device_failure_preserves_result_and_reports_mixed(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (device.global_mem_size, device.global_mem_size))
    original = _openfocus_gpu.Merge.resident_frame
    calls = 0

    def fail_after_first(self, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _openfocus_gpu.GpuError('device lost')
        return original(self, *args)

    monkeypatch.setattr(_openfocus_gpu.Merge, 'resident_frame', fail_after_first)
    rng = np.random.default_rng(118)
    frames = [rng.integers(0, 256, (193, 67, 3), dtype=np.uint8) for _ in range(3)]
    expected = MultiFocusFusion(algorithm='openfocus_b', use_gpu=False, exposure_match=False).fuse(frames, thread_count=1)
    fusion = MultiFocusFusion(algorithm='openfocus_b', use_gpu=True, exposure_match=False)
    actual = fusion.fuse(frames, thread_count=1)
    assert calls == 2
    assert 'CPU' in fusion.get_info()['device'] and 'OpenCL' in fusion.get_info()['device']
    np.testing.assert_array_equal(actual, expected)

def test_progress_callback_errors_are_not_swallowed():
    class ProgressError(Exception):
        pass
    def progress(current, total):
        if current > 0:
            raise ProgressError('progress failed')
    fusion = MultiFocusFusion(algorithm='openfocus_b', use_gpu=True, exposure_match=False)
    with pytest.raises(ProgressError, match='progress failed'):
        fusion.fuse(_stack(np.uint8, 3), thread_count=1, progress_callback=progress)


def test_cancellation_stops_gpu_requested_job():
    from core.registration import RegistrationCancelled
    cancelled = False

    def progress(current, total):
        nonlocal cancelled
        if current > 0:
            cancelled = True

    fusion = MultiFocusFusion(algorithm='openfocus_b', use_gpu=True, exposure_match=False)
    with pytest.raises(RegistrationCancelled):
        fusion.fuse(_stack(np.uint8, 3), thread_count=1,
                    progress_callback=progress, should_cancel=lambda: cancelled)


def test_opencl_cpu_device_is_not_a_gpu():
    from types import SimpleNamespace
    from fusion_methods import _openfocus_gpu

    cpu = SimpleNamespace(type=2, available=True, compiler_available=True)
    platform = SimpleNamespace(get_devices=lambda **kwargs: [cpu])
    cl = SimpleNamespace(get_platforms=lambda: [platform],
                         device_type=SimpleNamespace(GPU=4), Error=RuntimeError)
    assert _openfocus_gpu._device(cl) is None


@pytest.mark.parametrize('method', ['openfocus_b', 'openfocus_c'])
@pytest.mark.parametrize('smoothing', [0.0, 4.0])
def test_standalone_opencl_depth_matches_cpu(method, smoothing, monkeypatch):
    from core.depth_map import compute_focus_index
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    frames = [process_frame(frame, lambda image: image.copy())
              for frame in _stack(np.uint16, 3)]
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size), int(device.global_mem_size)))
    full_depth = _openfocus_gpu.Merge.full_depth
    completed = []

    def track_full_depth(self, *args, preserve_detail=True, **kwargs):
        result = full_depth(self, *args, preserve_detail=preserve_detail, **kwargs)
        if result is not None:
            completed.append(preserve_detail)
        return result

    monkeypatch.setattr(_openfocus_gpu.Merge, 'full_depth', track_full_depth)
    expected = compute_focus_index(frames, method, use_gpu=False, thread_count=1,
                                   smoothing=smoothing)
    actual = compute_focus_index(frames, method, use_gpu=True, thread_count=1,
                                 smoothing=smoothing)
    assert completed == [method == 'openfocus_b']
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('shape', [(1, 65), (65, 1), (2, 3)])
def test_narrow_images_match_cpu(shape):
    from fusion_methods.openfocus_bc import openfocus_b_impl
    rng = np.random.default_rng(781)
    frames = [rng.integers(0, 65536, shape, dtype=np.uint16) for _ in range(3)]
    cpu = openfocus_b_impl(frames, use_gpu=False, thread_count=1)
    gpu = openfocus_b_impl(frames, use_gpu=True, thread_count=1)
    np.testing.assert_array_equal(gpu, cpu)


def test_depth_device_failure_keeps_progress_monotonic(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')

    def fail(self, frames, shape, img_resize, kernel_size, smoothing,
             feather, should_cancel, progress_callback, *, live_preview=None):
        progress_callback(1, len(frames))
        raise _openfocus_gpu.GpuError('device lost')

    monkeypatch.setattr(_openfocus_gpu.Merge, 'full_depth', fail)
    frames = _stack(np.uint8, 3)
    expected = MultiFocusFusion(algorithm='openfocus_b', use_gpu=False, exposure_match=False).fuse(frames, thread_count=1)
    progress = []
    fusion = MultiFocusFusion(algorithm='openfocus_b', use_gpu=True, exposure_match=False)
    actual = fusion.fuse(frames, thread_count=1,
                         progress_callback=lambda current, total: progress.append(current))
    assert progress == sorted(progress)
    np.testing.assert_array_equal(actual, expected)


def test_mixed_precision_sources_match_cpu():
    from fusion_methods.openfocus_bc import openfocus_b_impl
    rng = np.random.default_rng(77)
    frames = [rng.integers(0, 256, (65, 67, 3), dtype=np.uint8),
              rng.integers(0, 65536, (65, 67, 3), dtype=np.uint16),
              rng.uniform(0, 255, (65, 67, 3)).astype(np.float32)]
    cpu = openfocus_b_impl(frames, use_gpu=False, thread_count=1)
    gpu = openfocus_b_impl(frames, use_gpu=True, thread_count=1)
    np.testing.assert_array_equal(gpu, cpu)


def test_full_depth_declines_insufficient_gpu_memory(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda device: (1, 2))
    engine = _openfocus_gpu.Merge(67, 3)
    if engine.device.host_unified_memory:
        pytest.skip('Unified GPUs use the host memory budget')
    frames = _stack(np.uint8, 3)
    result = engine.full_depth(frames, frames[0].shape, None, 17, 1.44,
                               0, None, None)
    assert result is None


@pytest.mark.parametrize('use_gpu', [False, True])
def test_mismatched_frame_shapes_are_rejected(use_gpu):
    from fusion_methods.openfocus_bc import openfocus_b_impl
    frames = [np.zeros((65, 67, 3), np.uint8), np.zeros((66, 67, 3), np.uint8)]
    with pytest.raises(ValueError, match='matching shapes'):
        openfocus_b_impl(frames, use_gpu=use_gpu, thread_count=1)


def test_openfocus_b_program_cache_reuse_and_invalidation():
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    _openfocus_gpu._invalidate_program_cache()
    try:
        m1 = _openfocus_gpu.Merge(67, 3)
        m2 = _openfocus_gpu.Merge(128, 3)
        assert m1.context is m2.context
        assert m1.queue is m2.queue
        assert m1.program is m2.program

        d1 = _openfocus_gpu._get_or_build_depth_program(m1.cl, m1.device, m1.context)
        d2 = _openfocus_gpu._get_or_build_depth_program(m2.cl, m2.device, m2.context)
        assert d1[0] is d2[0]
        assert d1[1] is d2[1]
    finally:
        _openfocus_gpu._invalidate_program_cache()
    assert _openfocus_gpu._PROGRAM_CACHE is None
    assert _openfocus_gpu._DEPTH_PROGRAM_CACHE is None


@pytest.mark.parametrize('module_name,cls_name', [
    ('fusion_methods._openfocus_gpu_a', 'GpuA'),
    ('fusion_methods._openfocus_gpu_c', 'GpuC'),
])
def test_openfocus_ac_program_cache_reuse_and_invalidation(module_name, cls_name):
    import importlib
    from fusion_methods import _openfocus_gpu

    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    mod = importlib.import_module(module_name)
    mod._invalidate_program_cache()
    cl = __import__('pyopencl')
    dev = _openfocus_gpu._device(cl)
    try:
        ctx1, q1, k1, *_ = mod._get_or_build_program(cl, dev)
        ctx2, q2, k2, *_ = mod._get_or_build_program(cl, dev)
        assert ctx1 is ctx2 and q1 is q2 and k1 is k2
    finally:
        mod._invalidate_program_cache()
    assert mod._PROGRAM_CACHE is None


@pytest.mark.parametrize('algo,module_name,cls_name', [
    ('openfocus_a', 'fusion_methods._openfocus_gpu_a', 'GpuA'),
    ('openfocus_c', 'fusion_methods._openfocus_gpu_c', 'GpuC'),
])
def test_openfocus_ac_drain_queue_on_error_and_cancel(
        monkeypatch, algo, module_name, cls_name):
    import importlib
    from types import SimpleNamespace
    from core.registration import RegistrationCancelled
    from fusion_methods import _openfocus_gpu

    mod = importlib.import_module(module_name)
    cls = getattr(mod, cls_name)

    class DeviceError(Exception):
        pass

    drained = []

    def initialize(self):
        self.cl = SimpleNamespace(Error=DeviceError)
        self.device_name = 'FakeCL'

    def make_fail(exc):
        def fail(self, *args, **kwargs):
            self.q = SimpleNamespace(
                finish=lambda: drained.append(_openfocus_gpu._device_lock.locked()),
            )
            raise exc
        return fail

    frames = _stack(np.uint8, 3)
    options = dict(algorithm=algo, exposure_match=False, base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    monkeypatch.setattr(cls, '__init__', initialize)

    # 1) cl.Error falls back cleanly, drains queue under lock, and invalidates cache.
    monkeypatch.setattr(cls, '_fuse_locked', make_fail(DeviceError('lost')))
    monkeypatch.setattr(mod, '_PROGRAM_CACHE', ('stale',))
    actual = MultiFocusFusion(use_gpu=True, **options).fuse(frames, thread_count=1)
    assert drained == [True]
    assert mod._PROGRAM_CACHE is None
    np.testing.assert_array_equal(actual, expected)

    # 2) Cancellation drains queue under lock before releasing device lock.
    drained.clear()
    monkeypatch.setattr(cls, '_fuse_locked', make_fail(RegistrationCancelled('stop')))
    with pytest.raises(RegistrationCancelled):
        MultiFocusFusion(use_gpu=True, **options).fuse(frames, thread_count=1)
    assert drained == [True]


def test_openfocus_b_full_depth_drains_queue_before_clearing_buffers_on_cancel():
    from types import SimpleNamespace
    from core.registration import RegistrationCancelled
    from fusion_methods import _openfocus_gpu

    events = []

    class TrackingDict(dict):
        def clear(self):
            events.append(('clear', _openfocus_gpu._device_lock.locked()))
            super().clear()

    class DummyEvent:
        def __call__(self, *args, **kwargs):
            return self

        def wait(self):
            return None

    gpu = _openfocus_gpu.Merge.__new__(_openfocus_gpu.Merge)
    gpu.cl = SimpleNamespace(
        Error=RuntimeError,
        mem_flags=SimpleNamespace(READ_WRITE=1, READ_ONLY=2, COPY_HOST_PTR=4),
        Buffer=lambda *a, **kw: object(),
        enqueue_copy=lambda *a, **kw: None,
        Kernel=lambda prg, name: DummyEvent(),
    )
    gpu.device = SimpleNamespace(max_mem_alloc_size=10**9, host_unified_memory=True)
    gpu.context = object()
    gpu.queue = SimpleNamespace(
        finish=lambda: events.append(('finish', _openfocus_gpu._device_lock.locked())),
    )
    gpu.depth_program = object()
    gpu.used = False
    gpu.failed = False

    depth_k = {n: DummyEvent() for n in _openfocus_gpu._DEPTH_KERNEL_NAMES}
    filt_k = {n: DummyEvent() for n in _openfocus_gpu._FILTER_KERNEL_NAMES}
    resamp_k = {n: DummyEvent() for n in _openfocus_gpu._RESAMPLE_KERNEL_NAMES}

    engine = _openfocus_gpu._GpuB(
        gpu, None,
        depth_kernels=depth_k, filter_kernels=filt_k, resample_kernels=resamp_k,
    )
    engine._cbuf = TrackingDict()

    gpu.acquire(lambda: False)
    try:
        with pytest.raises(RegistrationCancelled):
            engine.depth(
                _stack(np.uint8, 3), 17, 1.44, 0, (65, 67, 3), 3, lambda: True, None,
            )
    finally:
        gpu.release()

    assert events[0] == ('finish', True)
    assert ('clear', True) in events
    assert events.index(('finish', True)) < events.index(('clear', True))


def test_openfocus_b_resident_frame_reuses_upload_buffer():
    from types import SimpleNamespace
    from fusion_methods import _openfocus_gpu

    allocs = []
    copies = []

    class FakeEvent:
        def wait(self):
            return None

    def fake_buffer(ctx, flags, size=0, hostbuf=None):
        allocs.append((flags, size))
        return object()

    def fake_copy(q, dst, src, is_blocking=True):
        copies.append((dst, src.dtype, is_blocking))

    gpu = _openfocus_gpu.Merge.__new__(_openfocus_gpu.Merge)
    gpu.cl = SimpleNamespace(
        Error=RuntimeError,
        mem_flags=SimpleNamespace(READ_ONLY=1),
        Buffer=fake_buffer,
        enqueue_copy=fake_copy,
    )
    gpu.context = object()
    gpu.queue = SimpleNamespace(finish=lambda: None)
    gpu._merge_acc = object()
    gpu._merge_depth = object()
    gpu._merge_src = None
    gpu._merge_src_bytes = 0
    gpu._merge_shape = (4, 5, 3)
    gpu._merge_u16_depth = False
    gpu._merge_needs_drain = False
    gpu.used = False
    gpu.merge_u8_i64_kernel = lambda *args: FakeEvent()
    gpu.kernel = lambda *args: FakeEvent()

    f_u8 = np.zeros((4, 5, 3), np.uint8)
    f_u16 = np.zeros((4, 5, 3), np.uint16)

    gpu.acquire(None)
    try:
        gpu.resident_frame(f_u8, 0, 3, 256)
        first_buf = gpu._merge_src
        gpu.resident_frame(f_u8, 1, 3, 256)
        assert gpu._merge_src is first_buf
        assert len(allocs) == 1
        assert copies[0][2] is False and copies[1][2] is False

        gpu.resident_frame(f_u16, 2, 3, 1)
        assert len(allocs) == 2
        assert gpu._merge_src is not first_buf
        assert not gpu._merge_needs_drain
    finally:
        gpu.end_merge()

    assert gpu._merge_src is None
    assert gpu._merge_src_bytes == 0


def test_nvml_free_vram_reuses_session_and_recovers_on_error(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from fusion_methods import _openfocus_gpu

    state_calls = {'init': 0, 'shutdown': 0, 'mem': 0, 'fail_once': False, 'devices': 1}

    class FakeNvml:
        def nvmlInit_v2(self):
            state_calls['init'] += 1
            return 0

        def nvmlShutdown(self):
            state_calls['shutdown'] += 1
            return 0

        class _Fn:
            def __init__(self, fn):
                self._fn = fn
                self.argtypes = None

            def __call__(self, *args):
                return self._fn(*args)

        def __init__(self):
            self.nvmlDeviceGetCount_v2 = self._Fn(self._get_count)
            self.nvmlDeviceGetHandleByIndex_v2 = self._Fn(self._get_handle)
            self.nvmlDeviceGetName = self._Fn(self._get_name)
            self.nvmlDeviceGetMemoryInfo = self._Fn(self._get_mem)

        @staticmethod
        def _get_count(count_ref):
            count_ref._obj.value = state_calls['devices']
            return 0

        @staticmethod
        def _get_handle(idx, handle_ref):
            handle_ref._obj.value = 0x1234 + idx
            return 0

        @staticmethod
        def _get_name(handle, buf, size):
            buf.value = b'Fake RTX GPU'
            return 0

        @staticmethod
        def _get_mem(handle, info_ref):
            state_calls['mem'] += 1
            if state_calls['fail_once']:
                state_calls['fail_once'] = False
                return 1
            info_ref._obj.total = 8 * 1024**3
            info_ref._obj.free = 6 * 1024**3
            info_ref._obj.used = 2 * 1024**3
            return 0

    fake_lib = FakeNvml()
    dev = SimpleNamespace(name='Fake RTX GPU', global_mem_size=8 * 1024**3)
    prev_state = _openfocus_gpu._nvml_state
    prev_atexit = _openfocus_gpu._nvml_atexit_registered
    monkeypatch.setattr(ctypes, 'CDLL', lambda name: fake_lib)
    try:
        _openfocus_gpu._nvml_state = None
        _openfocus_gpu._nvml_atexit_registered = True

        assert _openfocus_gpu._nvml_free_vram(dev) == (6 * 1024**3, 8 * 1024**3)
        assert _openfocus_gpu._nvml_free_vram(dev) == (6 * 1024**3, 8 * 1024**3)
        assert state_calls['init'] == 1
        assert state_calls['shutdown'] == 0
        assert state_calls['mem'] == 2

        # Transient error tears down session, re-initializes cleanly, and succeeds.
        state_calls['fail_once'] = True
        assert _openfocus_gpu._nvml_free_vram(dev) == (6 * 1024**3, 8 * 1024**3)
        assert state_calls['shutdown'] == 1
        assert state_calls['init'] == 2

        # A partial enumeration cannot disambiguate matching adapters.
        state_calls['devices'] = 2
        state_calls['fail_once'] = True
        assert _openfocus_gpu._nvml_free_vram(dev) is None
        assert state_calls['shutdown'] == 2
        assert state_calls['init'] == 3
    finally:
        _openfocus_gpu._nvml_state = prev_state
        _openfocus_gpu._nvml_atexit_registered = prev_atexit
