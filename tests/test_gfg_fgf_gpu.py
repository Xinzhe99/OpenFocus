"""GFG-FGF CPU acceleration, OpenCL GPU parity, and CPU fallback tests."""
from contextlib import nullcontext
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.depth_map import compute_focus_index
from core.multi_focus_fusion import MultiFocusFusion
from core.registration import RegistrationCancelled
from core.source_frames import process_frame
from fusion_methods import gfg_fgf


def _stack(dtype, channels, shape=(65, 67), count=4, seed=4029):
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


def _gfg_reference(frames, kernel_size=7, img_resize=None, min_frame_focus=0.0):
    images = []
    resize = (img_resize is not None
              and (frames[0].shape[1], frames[0].shape[0]) != img_resize)
    for f in frames:
        img = _color(np.ascontiguousarray(f))
        if resize:
            img = cv2.resize(img, img_resize)
        if img.dtype == np.uint16:
            img = img.astype(np.float32) * (255.0 / 65535.0)
        images.append(img.astype(np.float32) / 255.0)
    channel = int(np.argmax(images[0].sum(axis=(0, 1))))
    guides = [img[:, :, channel] for img in images]

    kx = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)
    ky = kx.T.copy()
    scores = []
    for g in guides:
        gx = cv2.filter2D(g, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT)
        gy = cv2.filter2D(g, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT)
        scores.append(np.mean(gx[1:-1, 1:-1] ** 2 + gy[1:-1, 1:-1] ** 2))
    max_focus = max(scores) or 1.0

    tpl = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    shifts = []
    for r, c in ((1, 2), (1, 0), (2, 1), (0, 1)):
        k = np.zeros((3, 3), np.float32)
        k[r, c] = 1.0
        shifts.append(k)

    def focus_map(g):
        blur = cv2.blur(g, (kernel_size, kernel_size))
        ia = cv2.absdiff(g, blur)
        p = cv2.filter2D(g, cv2.CV_32F, tpl, borderType=cv2.BORDER_REFLECT)
        grad = np.zeros_like(p)
        for sk in shifts:
            grad += np.abs(cv2.filter2D(g, cv2.CV_32F, sk, borderType=cv2.BORDER_REFLECT) - p)
        grad *= 0.5
        grad[grad < 0.005] = 0.0
        edge = grad > 0
        ia[edge] = grad[edge]
        return gfg_fgf._run_guided_filter(g, ia, 5, 0.3)

    best_val = best_sob = idx = None
    for i, (g, score) in enumerate(zip(guides, scores)):
        if min_frame_focus > 0 and score < min_frame_focus * max_focus:
            continue
        afm = focus_map(g)
        sob = np.abs(cv2.Sobel(afm, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(afm, cv2.CV_32F, 0, 1, ksize=3))
        if best_val is None:
            best_val = afm.copy()
            best_sob = sob.copy()
            idx = np.full(afm.shape, i, np.int32)
        else:
            diff = afm - best_val
            win = (diff > 1e-6) | ((np.abs(diff) <= 1e-6) & (sob > best_sob))
            best_val[win] = afm[win]
            best_sob[win] = sob[win]
            idx[win] = i
    if idx is None:
        idx = np.zeros(guides[0].shape, np.int32)

    total = np.zeros(idx.shape, np.float32)
    num = np.zeros_like(images[0])
    for i, (img, g) in enumerate(zip(images, guides)):
        mask = idx == i
        if not mask.any():
            continue
        w = gfg_fgf._run_guided_filter(g, mask.astype(np.float32), 5, 0.3)
        total += w
        num += img * w[:, :, None]

    nz = total > 1e-6
    out = np.zeros_like(images[0])
    for c in range(3):
        ch = num[:, :, c].copy()
        np.divide(ch, total, out=ch, where=nz)
        if not nz.all():
            for i in np.unique(idx[~nz]):
                m = (~nz) & (idx == i)
                ch[m] = images[i][:, :, c][m]
        np.clip(ch * 255.0, 0, 255, out=ch)
        out[:, :, c] = ch
    out_dtype = np.uint8 if frames[0].dtype == np.uint8 else np.float32
    return out.astype(out_dtype)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize('channels', [1, 3, 4])
@pytest.mark.parametrize('shape,resize,kernel_size,min_focus', [
    ((17, 19), None, 7, 0.0),
    ((32, 48), None, 15, 0.0),
    ((65, 67), (51, 49), 1, 0.0),
    ((23, 27), (19, 17), 7, 0.25),
])
def test_gfg_cpu_matches_reference_bit_exact(dtype, channels, shape, resize, kernel_size, min_focus):
    frames = _stack(dtype, channels, shape=shape)
    expected = _gfg_reference(frames, kernel_size=kernel_size, img_resize=resize,
                              min_frame_focus=min_focus)
    actual_t1 = gfg_fgf.gfgfgf_impl(frames, resize, kernel_size=kernel_size,
                                    thread_count=1, min_frame_focus=min_focus)
    actual_t4 = gfg_fgf.gfgfgf_impl(frames, resize, kernel_size=kernel_size,
                                    thread_count=4, min_frame_focus=min_focus)
    np.testing.assert_array_equal(actual_t1, expected)
    np.testing.assert_array_equal(actual_t4, expected)


def test_gfg_cpu_numba_unavailable_fallback_bit_exact(monkeypatch):
    frames = _stack(np.uint8, 3, shape=(33, 39))
    expected = _gfg_reference(frames, kernel_size=7)
    monkeypatch.setattr(gfg_fgf, '_load_kernels', lambda: None)
    actual = gfg_fgf.gfgfgf_impl(frames, None, kernel_size=7, thread_count=2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('dtype,channels,resize,kernel_size', [
    (np.uint8, 1, None, 7),
    (np.uint8, 3, None, 7),
    (np.uint8, 4, None, 15),
    (np.uint8, 3, (51, 49), 1),
    (np.uint16, 1, None, 7),
    (np.uint16, 3, None, 15),
    (np.float32, 3, None, 7),
])
def test_gfg_gpu_matches_cpu_and_reports_device(
        dtype, channels, resize, kernel_size, monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),
                                        int(device.global_mem_size)))
    frames = _stack(dtype, channels, shape=(65, 67))
    lazy = [process_frame(frame, lambda image: image.copy()) for frame in frames]
    options = dict(algorithm='gfgfgf', exposure_match=False,
                   base_frame='none', sort_mode='off', tile_enabled=False)
    cpu = MultiFocusFusion(use_gpu=False, **options)
    gpu = MultiFocusFusion(use_gpu=True, **options)
    expected = cpu.fuse(lazy, img_resize=resize, thread_count=1, kernel_size=kernel_size)
    actual = gpu.fuse(lazy, img_resize=resize, thread_count=1, kernel_size=kernel_size)
    assert cpu.get_info()['device'] == 'CPU'
    assert gpu.get_info()['device'].startswith('OpenCL: ')
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    if dtype == np.uint8:
        assert diff.max() <= 1
        assert np.mean(diff == 0) >= 0.995
    else:
        assert diff.max() <= 0.05


def test_gfg_gpu_tiny_fallback_flat_deep_min_focus_and_tiled(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),
                                        int(device.global_mem_size)))
    options = dict(algorithm='gfgfgf', exposure_match=False,
                   base_frame='none', sort_mode='off', tile_enabled=False)

    # Sub-4px images fall back cleanly to CPU
    tiny_sub4 = _stack(np.uint8, 3, shape=(3, 3), count=3)
    gpu_sub4 = MultiFocusFusion(use_gpu=True, **options)
    actual_sub4 = gpu_sub4.fuse(tiny_sub4, thread_count=1)
    assert gpu_sub4.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual_sub4, MultiFocusFusion(use_gpu=False, **options).fuse(tiny_sub4, thread_count=1))

    # Tiny odd dimensions on GPU
    tiny = _stack(np.uint8, 3, shape=(17, 19), count=3)
    expected_tiny = MultiFocusFusion(use_gpu=False, **options).fuse(tiny, thread_count=1)
    gpu_tiny = MultiFocusFusion(use_gpu=True, **options)
    actual_tiny = gpu_tiny.fuse(tiny, thread_count=1)
    assert gpu_tiny.get_info()['device'].startswith('OpenCL: ')
    assert np.abs(actual_tiny.astype(int) - expected_tiny.astype(int)).max() <= 1

    # Flat tie frames keep first frame
    flat = [np.full((33, 39, 3), v, np.uint8) for v in (32, 96, 160)]
    actual_flat = MultiFocusFusion(use_gpu=True, **options).fuse(flat, thread_count=1)
    np.testing.assert_array_equal(actual_flat, flat[0])

    # Deep stack + min_frame_focus threshold
    deep_flat = np.full((33, 35, 3), 64, np.uint8)
    tex = np.random.default_rng(259).integers(32, 225, deep_flat.shape[:2], dtype=np.uint8)
    sharp = np.repeat(tex[:, :, None], 3, axis=2)
    deep_frames = [deep_flat] * 19 + [sharp]
    actual_deep = MultiFocusFusion(use_gpu=True, **options).fuse(deep_frames, kernel_size=7, thread_count=1)
    assert np.abs(actual_deep.astype(int) - sharp.astype(int)).max() <= 1

    mff_devs = []
    actual_mff = gfg_fgf.gfgfgf_impl(deep_frames, kernel_size=7, min_frame_focus=0.5,
                                     use_gpu=True, device_callback=mff_devs.append)
    assert mff_devs and mff_devs[-1].startswith('OpenCL: ')
    assert np.abs(actual_mff.astype(int) - sharp.astype(int)).max() <= 1

    # Tiled GPU execution
    tiled_frames = _stack(np.uint8, 3, shape=(80, 96), count=3)
    tiled_opts = dict(algorithm='gfgfgf', exposure_match=False,
                      base_frame='none', sort_mode='off',
                      tile_enabled=True, tile_threshold=32,
                      tile_block_size=48, tile_overlap=16)
    expected_tiled = MultiFocusFusion(use_gpu=False, **tiled_opts).fuse(tiled_frames, thread_count=1)
    gpu_tiled = MultiFocusFusion(use_gpu=True, **tiled_opts)
    actual_tiled = gpu_tiled.fuse(tiled_frames, thread_count=1)
    assert gpu_tiled.get_info()['device'].startswith('OpenCL: ')
    assert np.abs(actual_tiled.astype(int) - expected_tiled.astype(int)).max() <= 1


def test_gfg_cpu_toggle_never_constructs_gpu_backend(monkeypatch):
    from fusion_methods import _openfocus_gpu_gfg

    def forbidden(*args, **kwargs):
        raise AssertionError('GPU backend constructed while use_gpu=False')

    monkeypatch.setattr(_openfocus_gpu_gfg, 'GpuGfg', forbidden)
    fusion = MultiFocusFusion(algorithm='gfgfgf', use_gpu=False, exposure_match=False)
    fusion.fuse(_stack(np.uint8, 3, shape=(33, 39)), thread_count=1)
    assert fusion.get_info()['device'] == 'CPU'


def test_gfg_missing_opencl_falls_back(monkeypatch):
    import builtins

    frames = _stack(np.uint8, 3, shape=(33, 39))
    options = dict(algorithm='gfgfgf', exposure_match=False,
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


def test_gfg_device_failure_falls_back_with_monotonic_progress(monkeypatch):
    from fusion_methods._openfocus_gpu_gfg import GpuGfg

    class DeviceError(Exception):
        pass

    def initialize(self):
        self.cl = SimpleNamespace(Error=DeviceError)

    def fail(self, frames, resize, kernel_size, min_frame_focus, cancel, progress):
        progress(1, len(frames) * 2)
        raise DeviceError('device lost')

    frames = _stack(np.uint8, 3, shape=(33, 39))
    options = dict(algorithm='gfgfgf', exposure_match=False,
                   base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    monkeypatch.setattr(GpuGfg, '__init__', initialize)
    monkeypatch.setattr(GpuGfg, '_fuse_locked', fail)
    progress = []
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(frames, thread_count=1,
                         progress_callback=lambda current, total: progress.append(current))
    assert progress == sorted(progress)
    assert progress[-1] == len(frames) * 2
    assert fusion.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


def test_gfg_unknown_gpu_memory_falls_back(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    device = _openfocus_gpu._device(__import__('pyopencl'))
    if device.host_unified_memory:
        pytest.skip('Discrete GPU required')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda device: None)
    frames = _stack(np.uint8, 3, shape=(33, 39))
    options = dict(algorithm='gfgfgf', exposure_match=False,
                   base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(frames, thread_count=1)
    assert fusion.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('maximum_mib,unified,reason', [
    (640, False, None),
    (500, False, 'host memory budget'),
    (640, True, 'device budget'),
])
def test_gfg_gpu_admits_one_scaled_ecc_source_under_job_cap(
        tmp_path, monkeypatch, maximum_mib, unified, reason):
    from core import resource_policy, source_frames
    from fusion_methods import _openfocus_gpu
    from fusion_methods._openfocus_gpu_gfg import GpuGfg

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

    gpu = GpuGfg.__new__(GpuGfg)
    gpu.cl = SimpleNamespace(Buffer=allocate)
    gpu.ctx = None
    gpu.mf = SimpleNamespace(READ_ONLY=1, READ_WRITE=2)
    gpu.device = SimpleNamespace(global_mem_size=8 * 1024 * mib,
                                 max_mem_alloc_size=2 * 1024 * mib,
                                 host_unified_memory=unified)
    envelope = resource_policy.ProcessBudget(384 * mib, maximum_mib * mib, 192 * mib)
    with resource_policy.process_budget(envelope):
        if reason is None:
            with pytest.raises(AllocationReached):
                gpu._fuse_locked(frames, None, 7, 0.0, None, None)
        else:
            with pytest.raises(_openfocus_gpu.GpuError, match=reason):
                gpu._fuse_locked(frames, None, 7, 0.0, None, None)


@pytest.mark.parametrize('disk_reuse,exposure_match', [
    (False, False),
    (False, True),
    (True, True),
])
def test_gfg_gpu_lazy_ecc_matches_cpu_without_source_cache_growth(
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

    options = dict(algorithm='gfgfgf', exposure_match=exposure_match,
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
        assert np.abs(actual.astype(int) - expected.astype(int)).max() <= 1
    finally:
        if disk is not None:
            disk.close()


def test_gfg_gpu_cancellation_and_callback_errors(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),
                                        int(device.global_mem_size)))
    frames = _stack(np.uint8, 3, shape=(33, 39), count=3)
    options = dict(algorithm='gfgfgf', exposure_match=False,
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


def test_gfg_non_finite_float_falls_back_to_cpu(monkeypatch):
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    monkeypatch.setattr(_openfocus_gpu, '_free_vram',
                        lambda device: (int(device.global_mem_size),
                                        int(device.global_mem_size)))
    frames = _stack(np.float32, 3, shape=(33, 39), count=3)
    frames[1][5, 5, 0] = np.nan
    options = dict(algorithm='gfgfgf', exposure_match=False,
                   base_frame='none', sort_mode='off', tile_enabled=False)
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    gpu = MultiFocusFusion(use_gpu=True, **options)
    actual = gpu.fuse(frames, thread_count=1)
    assert gpu.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


def test_gfg_depth_map_matches_reference(monkeypatch):
    frames = _stack(np.uint8, 3, shape=(33, 39), count=4)
    idx_fast = compute_focus_index(frames, 'gfgfgf', {'kernel_size': 7})
    monkeypatch.setattr(gfg_fgf, '_load_kernels', lambda: None)
    idx_ref = compute_focus_index(frames, 'gfgfgf', {'kernel_size': 7})
    np.testing.assert_array_equal(idx_fast, idx_ref)
