"""DCT CPU Numba optimization, OpenCL GPU parity, and CPU fallback tests."""
from contextlib import nullcontext
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.ndimage import median_filter

from core.depth_map import compute_focus_index
from core.multi_focus_fusion import MultiFocusFusion
from core.registration import RegistrationCancelled
from core.source_frames import process_frame
from fusion_methods import dct


def _stack(dtype, channels, shape=(65, 67), count=4, seed=2029):
    rng = np.random.default_rng(seed)
    full_shape = shape if channels == 1 else shape + (channels,)
    if dtype == np.float32:
        frames = [rng.uniform(0.0, 255.0, full_shape).astype(np.float32)
                  for _ in range(count)]
        if channels == 4:
            for frame in frames:
                frame[:, :, 3] = 255.0
        return frames
    maximum = np.iinfo(dtype).max
    frames = [rng.integers(0, maximum + 1, full_shape, dtype=dtype)
              for _ in range(count)]
    if channels == 4:
        for frame in frames:
            frame[:, :, 3] = maximum
    return frames


def _color(frame):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame[:, :, :3]


def _dct_reference(frames, block_size=8, kernel_size=7):
    if kernel_size % 2 == 0:
        kernel_size += 1
    h, w = frames[0].shape[:2]
    ph = ((h + block_size - 1) // block_size) * block_size
    pw = ((w + block_size - 1) // block_size) * block_size
    padded = []
    for f in frames:
        img = _color(np.ascontiguousarray(f))
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        if img.dtype == np.uint16:
            img = img.astype(np.float32) * (255.0 / 65535.0)
        if ph != h or pw != w:
            img = cv2.copyMakeBorder(img, 0, ph - h, 0, pw - w, cv2.BORDER_REPLICATE)
        padded.append(img)
    variances = []
    for frame in padded:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean_sq = cv2.resize(gray ** 2, (pw // block_size, ph // block_size),
                             interpolation=cv2.INTER_AREA)
        mean = cv2.resize(gray, (pw // block_size, ph // block_size),
                          interpolation=cv2.INTER_AREA)
        variances.append(mean_sq - mean ** 2)
    index = np.argmax(np.stack(variances), axis=0).astype(np.int32)
    index = median_filter(median_filter(index, size=kernel_size, mode="nearest"),
                          size=kernel_size, mode="nearest")
    out_dtype = np.uint8 if all(f.dtype == np.uint8 for f in frames) else np.float32
    result = np.zeros((ph, pw, 3), out_dtype)
    for i in np.unique(index):
        mask = cv2.resize((index == i).astype(np.uint8), (pw, ph),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        result[mask] = padded[i][mask]
    return result[:h, :w].copy()


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize('channels', [1, 3, 4])
@pytest.mark.parametrize('shape,block_size,kernel_size', [
    ((17, 19), 8, 7),
    ((32, 48), 4, 3),
    ((65, 67), 8, 1),
    ((33, 41), 16, 5),
])
def test_dct_cpu_matches_reference_exact(dtype, channels, shape, block_size, kernel_size):
    frames = _stack(dtype, channels, shape=shape)
    expected = _dct_reference(frames, block_size=block_size, kernel_size=kernel_size)
    actual_t1 = dct.dct_focus_stack_fusion(
        frames, block_size=block_size, kernel_size=kernel_size, thread_count=1)
    actual_t4 = dct.dct_focus_stack_fusion(
        frames, block_size=block_size, kernel_size=kernel_size, thread_count=4)
    if dtype == np.uint8:
        np.testing.assert_array_equal(actual_t1, expected)
        np.testing.assert_array_equal(actual_t4, expected)
    else:
        np.testing.assert_allclose(actual_t1, expected, atol=1e-4)
        np.testing.assert_array_equal(actual_t4, actual_t1)


def test_dct_cpu_numba_unavailable_fallback_bit_exact(monkeypatch):
    frames = _stack(np.uint8, 3, shape=(33, 39))
    expected = _dct_reference(frames, block_size=8, kernel_size=7)
    monkeypatch.setattr(dct, '_load_kernels', lambda: None)
    actual = dct.dct_focus_stack_fusion(frames, block_size=8, kernel_size=7, thread_count=2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('dtype,channels,block_size,kernel_size', [
    (np.uint8, 1, 8, 7),
    (np.uint8, 3, 8, 7),
    (np.uint8, 4, 8, 1),
    (np.uint8, 3, 16, 5),
    (np.uint16, 1, 8, 7),
    (np.uint16, 3, 8, 3),
    (np.float32, 3, 8, 7),
])
def test_dct_gpu_matches_cpu_and_reports_device(
        dtype, channels, block_size, kernel_size, monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),
                                        int(device.global_mem_size)))
    frames = _stack(dtype, channels, shape=(65, 67))
    lazy = [process_frame(frame, lambda image: image.copy()) for frame in frames]
    options = dict(algorithm='dct', exposure_match=False,
                   base_frame='none', sort_mode='off', tile_enabled=False)
    cpu = MultiFocusFusion(use_gpu=False, **options)
    gpu = MultiFocusFusion(use_gpu=True, **options)
    expected = cpu.fuse(lazy, thread_count=1, block_size=block_size, kernel_size=kernel_size)
    actual = gpu.fuse(lazy, thread_count=1, block_size=block_size, kernel_size=kernel_size)
    assert cpu.get_info()['device'] == 'CPU'
    assert gpu.get_info()['device'].startswith('OpenCL: ')
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if dtype == np.uint8:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, atol=0.05)


def test_dct_gpu_flat_deep_tiny_mismatched_and_tiled(monkeypatch, tmp_path):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),
                                        int(device.global_mem_size)))
    options = dict(algorithm='dct', exposure_match=False,
                   base_frame='none', sort_mode='off', tile_enabled=False)

    # Flat tie frames pick frame 0
    flat = [np.full((33, 39, 3), v, np.uint8) for v in (32, 96, 160)]
    actual_flat = MultiFocusFusion(use_gpu=True, **options).fuse(flat, thread_count=1)
    np.testing.assert_array_equal(actual_flat, flat[0])

    # Deep stack (>255 frames) selects sharp frame 259
    deep_flat = np.full((33, 35, 3), 64, np.uint8)
    tex = np.random.default_rng(259).integers(32, 225, deep_flat.shape[:2], dtype=np.uint8)
    sharp = np.repeat(tex[:, :, None], 3, axis=2)
    deep_frames = [deep_flat] * 259 + [sharp]
    actual_deep = MultiFocusFusion(use_gpu=True, **options).fuse(deep_frames, kernel_size=1, thread_count=1)
    np.testing.assert_array_equal(actual_deep, sharp)

    # Sub-8px frames padded cleanly via MultiFocusFusion
    tiny = _stack(np.uint8, 3, shape=(5, 6), count=3)
    expected_tiny = MultiFocusFusion(use_gpu=False, **options).fuse(tiny, thread_count=1)
    gpu_tiny = MultiFocusFusion(use_gpu=True, **options)
    actual_tiny = gpu_tiny.fuse(tiny, thread_count=1)
    assert gpu_tiny.get_info()['device'].startswith('OpenCL: ')
    np.testing.assert_array_equal(actual_tiny, expected_tiny)

    # Mismatched frame shape + direct output_path write on GPU
    base_frames = _stack(np.uint8, 3, shape=(48, 64), count=3)
    mismatched = [base_frames[0], cv2.resize(base_frames[1], (32, 24)), base_frames[2]]
    out_file = tmp_path / 'dct_gpu.png'
    devs = []
    expected_mm = dct.dct_focus_stack_fusion(mismatched, use_gpu=False, thread_count=1)
    actual_mm = dct.dct_focus_stack_fusion(
        mismatched, output_path=str(out_file), use_gpu=True,
        thread_count=1, device_callback=devs.append)
    assert devs and devs[-1].startswith('OpenCL: ')
    assert out_file.is_file()
    np.testing.assert_array_equal(actual_mm, expected_mm)

    # Tiled GPU execution
    tiled_frames = _stack(np.uint8, 3, shape=(80, 96), count=3)
    tiled_opts = dict(algorithm='dct', exposure_match=False,
                      base_frame='none', sort_mode='off',
                      tile_enabled=True, tile_threshold=32,
                      tile_block_size=48, tile_overlap=16)
    expected_tiled = MultiFocusFusion(use_gpu=False, **tiled_opts).fuse(tiled_frames, thread_count=1)
    gpu_tiled = MultiFocusFusion(use_gpu=True, **tiled_opts)
    actual_tiled = gpu_tiled.fuse(tiled_frames, thread_count=1)
    assert gpu_tiled.get_info()['device'].startswith('OpenCL: ')
    np.testing.assert_array_equal(actual_tiled, expected_tiled)


def test_dct_gpu_resizes_native_16bit_samples_before_normalizing(monkeypatch):
    from fusion_methods import _openfocus_gpu

    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),) * 2)
    rng = np.random.default_rng(131)
    frames = [np.full((65, 67, 3), 32768, np.uint16),
              rng.integers(0, 65536, (32, 35, 3), dtype=np.uint16)]
    expected = dct.dct_focus_stack_fusion(frames, kernel_size=1, thread_count=1)
    used = []
    actual = dct.dct_focus_stack_fusion(frames, kernel_size=1, use_gpu=True,
                                       device_callback=used.append)
    assert used[0].startswith('OpenCL: ')
    np.testing.assert_array_equal(actual, expected)


def test_dct_cpu_toggle_never_constructs_gpu_backend(monkeypatch):
    from fusion_methods import _openfocus_gpu_dct

    def forbidden(*args, **kwargs):
        raise AssertionError('GPU backend constructed while use_gpu=False')

    monkeypatch.setattr(_openfocus_gpu_dct, 'GpuDct', forbidden)
    fusion = MultiFocusFusion(algorithm='dct', use_gpu=False, exposure_match=False)
    fusion.fuse(_stack(np.uint8, 3, shape=(33, 39)), thread_count=1)
    assert fusion.get_info()['device'] == 'CPU'


def test_dct_missing_opencl_falls_back(monkeypatch):
    import builtins

    frames = _stack(np.uint8, 3, shape=(33, 39))
    options = dict(algorithm='dct', exposure_match=False,
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


def test_dct_device_failure_falls_back_with_monotonic_progress(monkeypatch):
    from fusion_methods import _openfocus_gpu_dct

    class DeviceError(Exception):
        pass

    def initialize(self):
        self.cl = SimpleNamespace(Error=DeviceError)

    def fail(self, frames, block_size, kernel_size, cancel, progress):
        progress(1, len(frames) * 2)
        raise DeviceError('device lost')

    frames = _stack(np.uint8, 3, shape=(33, 39))
    options = dict(algorithm='dct', exposure_match=False,
                   base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    monkeypatch.setattr(_openfocus_gpu_dct.GpuDct, '__init__', initialize)
    monkeypatch.setattr(_openfocus_gpu_dct.GpuDct, '_fuse_locked', fail)
    progress = []
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(frames, thread_count=1,
                         progress_callback=lambda current, total: progress.append(current))
    assert progress == sorted(progress)
    assert progress[-1] == len(frames) * 2
    assert fusion.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


def test_dct_program_cache_reuse_and_invalidation(monkeypatch):
    from fusion_methods import _openfocus_gpu, _openfocus_gpu_dct
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    _openfocus_gpu_dct._invalidate_program_cache()
    cl = __import__('pyopencl')
    dev = _openfocus_gpu._device(cl)
    ctx1, q1, _ = _openfocus_gpu_dct._get_or_build_program(cl, dev)
    ctx2, q2, _ = _openfocus_gpu_dct._get_or_build_program(cl, dev)
    assert ctx1 is ctx2 and q1 is q2
    _openfocus_gpu_dct._invalidate_program_cache()
    assert _openfocus_gpu_dct._PROGRAM_CACHE is None


def test_dct_unknown_gpu_memory_falls_back(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    device = _openfocus_gpu._device(__import__('pyopencl'))
    if device.host_unified_memory:
        pytest.skip('Discrete GPU required')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda device: None)
    frames = _stack(np.uint8, 3, shape=(33, 39))
    options = dict(algorithm='dct', exposure_match=False,
                   base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(frames, thread_count=1)
    assert fusion.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('maximum_mib,unified,reason', [
    (640, False, None),
    (500, False, 'host memory budget'),
    (580, True, 'device budget'),
])
def test_dct_gpu_admits_one_scaled_ecc_source_under_job_cap(
        tmp_path, monkeypatch, maximum_mib, unified, reason):
    from core import resource_policy, source_frames
    from fusion_methods import _openfocus_gpu
    from fusion_methods._openfocus_gpu_dct import GpuDct

    path = tmp_path / 'stack.tif'
    path.touch()
    stamp = source_frames._fingerprint(path)
    shape = (1366, 2048, 3)
    frames = [process_frame(source_frames.SourceFrame(
        str(path), page, shape, np.dtype(np.uint8), None, stamp, scale=.25,
        native_shape=(5464, 8192, 3)),
        lambda _: pytest.fail('Admission must not resolve source pixels')) for page in range(20)]
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache())
    mib = 1024 ** 2
    monkeypatch.setattr(resource_policy, 'probe_memory', lambda: resource_policy.ResourceSnapshot(
        available_ram=8 * 1024 * mib, total_ram=16 * 1024 * mib))
    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda device: (8 * 1024 * mib,) * 2)

    class AllocationReached(Exception):
        pass

    def allocate(*args):
        raise AllocationReached

    gpu = GpuDct.__new__(GpuDct)
    gpu.cl = SimpleNamespace(Buffer=allocate)
    gpu.ctx = None
    gpu.mf = SimpleNamespace(READ_ONLY=1, WRITE_ONLY=2, READ_WRITE=3)
    gpu.device = SimpleNamespace(global_mem_size=8 * 1024 * mib,
                                 max_mem_alloc_size=2 * 1024 * mib,
                                 host_unified_memory=unified)
    envelope = resource_policy.ProcessBudget(384 * mib, maximum_mib * mib, 192 * mib)
    with resource_policy.process_budget(envelope):
        if reason is None:
            with pytest.raises(AllocationReached):
                gpu._fuse_locked(frames, 8, 7, None, None)
        else:
            with pytest.raises(_openfocus_gpu.GpuError, match=reason):
                gpu._fuse_locked(frames, 8, 7, None, None)


@pytest.mark.parametrize('disk_reuse,exposure_match', [
    (False, False),
    (False, True),
    (True, True),
])
def test_dct_gpu_lazy_ecc_matches_cpu_without_source_cache_growth(
        tmp_path, monkeypatch, disk_reuse, exposure_match):
    from core import frame_resolution_cache, frame_storage, registration, resource_policy, source_frames
    from fusion_methods import _openfocus_gpu

    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),) * 2)
    cache = source_frames.FrameCache()
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', cache)
    eager, lazy = [], []
    for index, pixels in enumerate(_stack(np.uint8, 3, shape=(65, 67), count=3)):
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

    options = dict(algorithm='dct', exposure_match=exposure_match,
                   base_frame='none', sort_mode='off', tile_enabled=False)
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
        assert len(set(ram_usage)) == 1
        if disk_reuse or not exposure_match:
            assert ram_usage[0] == 0
        else:
            assert ram_usage[0] <= sum(frame.source.nbytes for frame in lazy)
        np.testing.assert_array_equal(actual, expected)
    finally:
        if disk is not None:
            disk.close()


def test_dct_gpu_cancellation_and_callback_errors(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),
                                        int(device.global_mem_size)))
    frames = _stack(np.uint8, 3, shape=(33, 39), count=3)
    options = dict(algorithm='dct', exposure_match=False,
                   base_frame='none', sort_mode='off', tile_enabled=False)

    cancelled = False

    def progress_cancel(current, total):
        nonlocal cancelled
        if current > 0:
            cancelled = True

    with pytest.raises(RegistrationCancelled):
        MultiFocusFusion(use_gpu=True, **options).fuse(
            frames, thread_count=1, progress_callback=progress_cancel,
            should_cancel=lambda: cancelled)

    class ProgressError(Exception):
        pass

    def progress_fail(current, total):
        if current > 0:
            raise ProgressError('progress failed')

    with pytest.raises(ProgressError, match='progress failed'):
        MultiFocusFusion(use_gpu=True, **options).fuse(
            frames, thread_count=1, progress_callback=progress_fail)

    def cancel_fail():
        raise RuntimeError('cancel failed')

    with pytest.raises(RuntimeError, match='cancel failed'):
        MultiFocusFusion(use_gpu=True, **options).fuse(
            frames, thread_count=1, should_cancel=cancel_fail)


@pytest.mark.parametrize('bad_val', [np.nan, 500.0])
def test_dct_non_finite_or_unbounded_float_falls_back_to_cpu(monkeypatch, bad_val):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),
                                        int(device.global_mem_size)))
    frames = _stack(np.float32, 3, shape=(33, 39), count=3)
    frames[1][5, 5, 0] = bad_val
    options = dict(algorithm='dct', exposure_match=False,
                   base_frame='none', sort_mode='off', tile_enabled=False)
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    gpu = MultiFocusFusion(use_gpu=True, **options)
    actual = gpu.fuse(frames, thread_count=1)
    assert gpu.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


def test_dct_depth_map_matches_reference(monkeypatch):
    frames = _stack(np.uint8, 3, shape=(33, 39), count=4)
    idx_fast = compute_focus_index(frames, 'dct', {'kernel_size': 7})
    monkeypatch.setattr(dct, '_load_kernels', lambda: None)
    idx_ref = compute_focus_index(frames, 'dct', {'kernel_size': 7})
    np.testing.assert_array_equal(idx_fast, idx_ref)
