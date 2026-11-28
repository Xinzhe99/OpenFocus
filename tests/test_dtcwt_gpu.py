"""CPU optimization and OpenCL GPU acceleration tests for DTCWT fusion."""
import builtins
import warnings
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.multi_focus_fusion import MultiFocusFusion
from core.registration import RegistrationCancelled
from core.source_frames import SOURCE_CACHE, process_frame
from fusion_methods import _openfocus_gpu, _openfocus_gpu_dtcwt, dtcwt as dtcwt_mod


pytestmark = pytest.mark.skipif(
    dtcwt_mod.dtcwt is None, reason='dtcwt backend not installed'
)


def _stack(shape=(65, 67), dtype=np.uint8, channels=3, count=4, low_detail=False, seed=4019):
    rng = np.random.default_rng(seed)
    h, w = shape
    if low_detail:
        base = rng.integers(100, 104, shape if channels == 1 else (h, w, channels), dtype=dtype)
        return [base.copy() for _ in range(count)]
    tex = rng.integers(0, 256, (h, w), dtype=np.uint8)
    blur = cv2.GaussianBlur(tex, (21, 21), 0)
    frames = []
    for k in range(count):
        band = np.zeros((h, w), dtype=bool)
        band[k * h // count:(k + 1) * h // count] = True
        g = np.where(band, tex, blur)
        if channels == 1:
            arr = g.copy()
        elif channels == 3:
            arr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            arr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGRA)
        else:
            raise ValueError(channels)
        if dtype == np.uint16:
            arr = arr.astype(np.uint16) * 257
        elif dtype == np.float32:
            arr = arr.astype(np.float32) + rng.uniform(-0.25, 0.25, arr.shape).astype(np.float32)
            np.clip(arr, 0.0, 255.0, out=arr)
        frames.append(arr)
    return frames


def _require_gpu(monkeypatch=None):
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    if monkeypatch is not None:
        monkeypatch.setattr(
            _openfocus_gpu,
            '_free_vram',
            lambda device: (int(device.global_mem_size), int(device.global_mem_size)),
        )


@pytest.mark.parametrize('dtype,channels,resize,low_detail', [
    (np.uint8, 1, None, False),
    (np.uint8, 3, None, False),
    (np.uint8, 4, None, False),
    (np.uint8, 3, (51, 49), False),
    (np.uint8, 3, None, True),
    (np.uint16, 1, None, False),
    (np.uint16, 3, None, False),
    (np.float32, 3, None, False),
])
def test_dtcwt_gpu_matches_cpu_and_reports_device(
        dtype, channels, resize, low_detail, monkeypatch):
    _require_gpu(monkeypatch)
    frames = _stack(shape=(65, 67), dtype=dtype, channels=channels, low_detail=low_detail)
    lazy = [process_frame(f, lambda image: image.copy()) for f in frames]
    options = dict(
        algorithm='dtcwt', exposure_match=False, base_frame='none', sort_mode='off'
    )
    cpu = MultiFocusFusion(use_gpu=False, **options)
    gpu = MultiFocusFusion(use_gpu=True, **options)

    expected = cpu.fuse(lazy, img_resize=resize, thread_count=1, N=4)
    actual = gpu.fuse(lazy, img_resize=resize, thread_count=1, N=4)

    assert cpu.get_info()['device'] == 'CPU'
    assert gpu.get_info()['device'].startswith('OpenCL: ')
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    if dtype == np.uint8:
        assert diff.max() <= 2
    else:
        assert diff.max() <= 0.1


@pytest.mark.parametrize('nlevels', [1, 2, 3, 4, 5])
def test_dtcwt_gpu_matches_cpu_across_levels(nlevels, monkeypatch):
    _require_gpu(monkeypatch)
    frames = _stack(shape=(65, 67), dtype=np.uint8, channels=3, count=4)
    cpu_out = dtcwt_mod._dtcwt_impl(frames, None, nlevels, False, thread_count=1)
    seen = []
    gpu_out = dtcwt_mod._dtcwt_impl(
        frames, None, nlevels, True, thread_count=1, device_callback=seen.append)
    assert len(seen) == 1 and seen[0].startswith('OpenCL: ')
    assert gpu_out.shape == cpu_out.shape and gpu_out.dtype == cpu_out.dtype
    diff = np.abs(gpu_out.astype(np.int32) - cpu_out.astype(np.int32))
    assert diff.max() <= 2


def test_dtcwt_gpu_tiled_matches_cpu(monkeypatch):
    _require_gpu(monkeypatch)
    frames = _stack(shape=(96, 112), dtype=np.uint8, channels=3, count=4)
    fixed, worker = dtcwt_mod.estimate_working_set(48, 48, 3, len(frames))
    monkeypatch.setattr('core.multi_focus_fusion.memory_budget_bytes', lambda: fixed + worker)
    options = dict(
        algorithm='dtcwt',
        exposure_match=False,
        base_frame='none',
        sort_mode='off',
        tile_enabled=True,
        tile_block_size=48,
        tile_overlap=16,
        tile_threshold=1,
    )
    cpu = MultiFocusFusion(use_gpu=False, **options)
    gpu = MultiFocusFusion(use_gpu=True, **options)
    expected = cpu.fuse(frames, thread_count=1, N=3)
    actual = gpu.fuse(frames, thread_count=1, N=3)
    assert gpu.get_info()['device'].startswith('OpenCL: ')
    diff = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
    assert diff.max() <= 2


def test_dtcwt_gpu_1ch_streaming_mode_matches_3ch_mode(monkeypatch):
    _require_gpu(monkeypatch)
    from core import resource_policy
    from fusion_methods._openfocus_gpu_dtcwt import GpuDtcwt, _plan_levels

    frames = _stack(shape=(65, 67), dtype=np.uint8, channels=3, count=3)
    full = GpuDtcwt().fuse(frames, 4)

    # Set VRAM limit tightly between 1-channel and 3-channel workspace sizes
    specs = _plan_levels(65, 67, 4)
    _, per_ch, need_1ch, need_3ch, _ = _openfocus_gpu_dtcwt._estimate_vram_bytes(
        specs, len(frames), frames[0].nbytes, frames[0].nbytes,
    )
    active = len(frames) * len(specs) * 4
    mid_vram = need_1ch + per_ch
    assert need_1ch <= mid_vram < need_3ch

    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda device: (mid_vram, mid_vram * 4))
    monkeypatch.setattr(resource_policy, 'vram_budget', lambda snap: mid_vram)
    from fusion_methods._openfocus_gpu_dtcwt import _source_staging_bytes

    staging = max(_source_staging_bytes(frames[0]), 2 * frames[0].nbytes) + active + 16 * 1024 * 1024
    monkeypatch.setattr(resource_policy, 'ram_ceiling', lambda snap: mid_vram + staging)
    uploads = []
    import pyopencl as cl
    enqueue_copy = cl.enqueue_copy

    def observe_upload(queue, dest, src, *args, **kwargs):
        if isinstance(src, np.ndarray):
            uploads.append(src.nbytes)
        return enqueue_copy(queue, dest, src, *args, **kwargs)

    monkeypatch.setattr(cl, 'enqueue_copy', observe_upload)
    streamed = GpuDtcwt().fuse(frames, 4)
    np.testing.assert_array_equal(streamed, full)
    # The three-channel path uploads each frame at most twice. Ensure the
    # constrained run actually repeated uploads for separate channels.
    assert len(uploads) > 2 * len(frames)


def test_dtcwt_missing_opencl_falls_back(monkeypatch):
    frames = _stack(shape=(48, 52), dtype=np.uint8, channels=3, count=3)
    options = dict(algorithm='dtcwt', exposure_match=False, base_frame='none', sort_mode='off')
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


def test_dtcwt_unknown_vram_falls_back_to_cpu(monkeypatch):
    _require_gpu()
    frames = _stack(shape=(48, 52), dtype=np.uint8, channels=3, count=3)
    options = dict(algorithm='dtcwt', exposure_match=False, base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)

    monkeypatch.setattr(_openfocus_gpu, '_free_vram', lambda device: None)
    # Trigger discrete GPU path cleanly via a device with host_unified_memory=False
    orig_device = _openfocus_gpu._device

    def discrete_dev(cl):
        d = orig_device(cl)
        if d is not None and getattr(d, 'host_unified_memory', False):
            pytest.skip('Host uses unified memory OpenCL device')
        return d

    monkeypatch.setattr(_openfocus_gpu, '_device', discrete_dev)
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(frames, thread_count=1)
    assert fusion.get_info()['device'] == 'CPU'
    np.testing.assert_array_equal(actual, expected)


def test_dtcwt_device_failure_falls_back_with_monotonic_progress(monkeypatch):
    from fusion_methods._openfocus_gpu_dtcwt import GpuDtcwt

    class DeviceError(Exception):
        pass

    drained = []

    def initialize(self):
        self.cl = SimpleNamespace(Error=DeviceError)
        self.device_name = 'FakeCL'

    def fail(self, images, nlevels, cancel, progress):
        self.q = SimpleNamespace(finish=lambda: drained.append(_openfocus_gpu._device_lock.locked()))
        progress(1, 2 * len(images) + 1)
        raise DeviceError('device lost')

    frames = _stack(shape=(48, 52), dtype=np.uint8, channels=3, count=4)
    options = dict(algorithm='dtcwt', exposure_match=False, base_frame='none', sort_mode='off')
    expected = MultiFocusFusion(use_gpu=False, **options).fuse(frames, thread_count=1)
    monkeypatch.setattr(GpuDtcwt, '__init__', initialize)
    monkeypatch.setattr(GpuDtcwt, '_fuse_locked', fail)
    progress = []
    fusion = MultiFocusFusion(use_gpu=True, **options)
    actual = fusion.fuse(
        frames,
        thread_count=1,
        progress_callback=lambda current, total: progress.append((current, total)),
    )
    assert progress
    fracs = [c / max(1, t) for c, t in progress]
    assert fracs == sorted(fracs)
    currents = [c for c, _ in progress]
    assert currents == sorted(currents)
    assert progress[-1][0] == progress[-1][1]
    assert fusion.get_info()['device'] == 'CPU'
    assert drained == [True]
    np.testing.assert_array_equal(actual, expected)


def test_dtcwt_cancellation_drains_queue_without_cpu_retry(monkeypatch):
    from fusion_methods._openfocus_gpu_dtcwt import GpuDtcwt

    class DeviceError(Exception):
        pass

    drained = []

    def initialize(self):
        self.cl = SimpleNamespace(Error=DeviceError)

    def cancel(self, *_):
        self.q = SimpleNamespace(finish=lambda: drained.append(_openfocus_gpu._device_lock.locked()))
        raise RegistrationCancelled()

    monkeypatch.setattr(GpuDtcwt, '__init__', initialize)
    monkeypatch.setattr(GpuDtcwt, '_fuse_locked', cancel)
    devices = []
    frames = _stack(count=2)
    with pytest.raises(RegistrationCancelled):
        dtcwt_mod._dtcwt_impl(frames, None, 3, True, device_callback=devices.append)
    assert drained == [True]
    assert devices == []
    assert not _openfocus_gpu._device_lock.locked()


def test_dtcwt_tiny_budget_falls_back_to_cpu(monkeypatch):
    _require_gpu(monkeypatch)
    from core import resource_policy

    frames = _stack(shape=(48, 52), dtype=np.uint8, channels=3, count=3)
    expected = dtcwt_mod._dtcwt_impl(frames, None, 3, False, thread_count=1)
    monkeypatch.setattr(resource_policy, 'ram_ceiling', lambda snap: 1024)
    seen = []
    actual = dtcwt_mod._dtcwt_impl(
        frames, None, 3, True, thread_count=1, device_callback=seen.append
    )
    assert seen == ['CPU']
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, 300.0, -300.0])
def test_dtcwt_unbounded_or_nonfinite_float_falls_back(bad_value, monkeypatch):
    _require_gpu(monkeypatch)
    frames = _stack(shape=(36, 40), dtype=np.float32, channels=3, count=3)
    frames[1][5, 5, 0] = bad_value
    expected = dtcwt_mod._dtcwt_impl(frames, None, 3, False, thread_count=1)
    seen = []
    actual = dtcwt_mod._dtcwt_impl(
        frames, None, 3, True, thread_count=1, device_callback=seen.append
    )
    assert seen == ['CPU']
    np.testing.assert_allclose(actual, expected, atol=1e-5, equal_nan=True)


def test_dtcwt_gpu_cancellation_raises_before_and_during_execution(monkeypatch):
    _require_gpu(monkeypatch)
    frames = _stack(shape=(64, 64), dtype=np.uint8, channels=3, count=4)
    with pytest.raises(RegistrationCancelled):
        dtcwt_mod._dtcwt_impl(frames, None, 3, True, should_cancel=lambda: True)

    progress = []
    devices = []
    with pytest.raises(RegistrationCancelled):
        dtcwt_mod._dtcwt_impl(
            frames, None, 3, True, should_cancel=lambda: bool(progress and progress[-1] >= 1),
            progress_callback=lambda done, total: progress.append(done),
            device_callback=devices.append)
    assert progress == [0, 1]
    assert devices == []
    dtcwt_mod._dtcwt_impl(frames, None, 3, True, device_callback=devices.append)
    assert len(devices) == 1 and devices[0].startswith('OpenCL: ')


def test_dtcwt_gpu_clears_source_cache_and_emits_no_kernel_warnings(monkeypatch):
    _require_gpu(monkeypatch)
    SOURCE_CACHE.clear()
    raw = _stack(shape=(64, 64), dtype=np.uint8, channels=3, count=3)
    lazy = [process_frame(f, lambda img: img.copy()) for f in raw]
    seen = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for i in range(2):
            SOURCE_CACHE.put(('dtcwt-probe', i), raw[0])
            dtcwt_mod._dtcwt_impl(lazy, None, 3, True, device_callback=seen.append)
            assert SOURCE_CACHE.bytes == 0
    assert len(seen) == 2 and all(device.startswith('OpenCL: ') for device in seen)
    repeated = [w for w in caught if 'RepeatedKernelRetrieval' in str(w.category.__name__)]
    assert repeated == []


@pytest.mark.parametrize('shape', [(1, 1), (1, 9), (11, 1), (131, 53)])
def test_dtcwt_compiled_vote_matches_scipy_and_python(shape):
    from scipy.ndimage import convolve
    from fusion_methods._dtcwt_kernels import _vote, consistent_vote

    rng = np.random.default_rng(99)
    for n_labels in (3, 17, 300):
        idx = rng.integers(0, n_labels, shape, dtype=np.int32)
        out = np.empty_like(idx)
        vote = consistent_vote()
        for start in range(0, shape[0], 128):
            vote(idx, out, start, min(start + 128, shape[0]))
        python_out = np.empty_like(idx)
        _vote(idx, python_out, 0, shape[0])
        np.testing.assert_array_equal(out, python_out)
        # Verify exact match with convolve argmax
        kernel = np.ones((3, 3), dtype=np.float32)
        best_c = None
        best_l = None
        for label in range(n_labels):
            votes = convolve((idx == label).astype(np.float32), kernel, mode='nearest')
            if best_c is None:
                best_c = votes
                best_l = np.full_like(idx, label)
            else:
                win = votes > best_c
                best_c[win] = votes[win]
                best_l[win] = label
        np.testing.assert_array_equal(out, best_l)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize('nlevels', [1, 2, 4])
def test_dtcwt_gpu_grayscale_fast_path_and_multi_level_parity(monkeypatch, dtype, nlevels):
    _require_gpu(monkeypatch)
    frames = _stack(shape=(67, 73), dtype=dtype, channels=1, count=3)
    expected = dtcwt_mod._dtcwt_impl(frames, None, nlevels, False, thread_count=1)
    seen = []
    actual = dtcwt_mod._dtcwt_impl(
        frames, None, nlevels, True, thread_count=1, device_callback=seen.append,
    )
    assert seen and seen[0].startswith('OpenCL: ')
    if dtype == np.uint8:
        assert np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16))) <= 1
    else:
        assert np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))) <= 1.0
