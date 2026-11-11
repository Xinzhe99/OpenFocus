"""OpenFocus B device selection and deterministic CPU/OpenCL parity."""
import numpy as np
import pytest

from core.multi_focus_fusion import MultiFocusFusion
from core.source_frames import process_frame


def _stack(dtype, channels):
    rng = np.random.default_rng(4019)
    shape = (65, 67) if channels == 1 else (65, 67, channels)
    maximum = np.iinfo(dtype).max
    return [rng.integers(0, maximum + 1, shape, dtype=dtype) for _ in range(3)]


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
    original = _openfocus_gpu.Merge.band
    calls = 0

    def fail_after_first(self, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _openfocus_gpu.GpuError('device lost')
        return original(self, *args)

    monkeypatch.setattr(_openfocus_gpu.Merge, 'band', fail_after_first)
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


def test_standalone_opencl_depth_matches_cpu():
    from core.depth_map import compute_focus_index
    from fusion_methods import _openfocus_gpu
    if not _openfocus_gpu.gpu_available():
        pytest.skip('No OpenCL GPU is available')
    frames = [process_frame(frame, lambda image: image.copy())
              for frame in _stack(np.uint16, 3)]
    expected = compute_focus_index(frames, 'openfocus_b', use_gpu=False, thread_count=1)
    actual = compute_focus_index(frames, 'openfocus_b', use_gpu=True, thread_count=1)
    np.testing.assert_array_equal(actual, expected)
