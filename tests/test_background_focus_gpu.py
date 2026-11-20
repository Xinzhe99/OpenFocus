"""Exact focus reduction, device fallback and CPU-only routing."""
import logging

import cv2
import numpy as np
import pytest

from core import _background_focus_gpu as gpu
from core import background_fill as bf


def _reference(gray):
    energies = []
    for plane in gray:
        lap = cv2.Laplacian(plane, cv2.CV_32F, ksize=3)
        energies.append(cv2.GaussianBlur(lap * lap, (0, 0), 0.5))
    energy = np.stack(energies)
    peak = energy.max(axis=0) + 1e-10
    median = np.median(energy, axis=0) + 1e-10
    index_type = np.uint8 if len(energy) <= 256 else np.uint16
    return np.log10(peak / median), np.log10(peak), energy.argmax(axis=0).astype(index_type)


def _compare(result, expected):
    measures, index = result
    for actual, reference in zip((measures['ratio'], measures['peak'], index), expected):
        assert actual.dtype == reference.dtype
        assert actual.tobytes() == reference.tobytes()


@pytest.mark.parametrize('dtype,frames', [(np.uint8, 29), (np.uint16, 30), (np.float32, 258)])
def test_camera_focus_matches_numpy_on_multiple_device_bands(dtype, frames, monkeypatch, caplog):
    from fusion_methods._openfocus_gpu import gpu_available

    if not gpu_available():
        pytest.skip('No OpenCL GPU available')
    monkeypatch.setattr(bf, '_GPU_FOCUS_MIN_BYTES', 0)
    # Cross a transfer boundary within an image row, including a short tail.
    monkeypatch.setattr(gpu, '_band_pixels', lambda *args: 157)
    caplog.set_level(logging.INFO, logger=gpu.__name__)
    rng = np.random.default_rng(44)
    gray = []
    for _ in range(frames):
        if dtype == np.float32:
            frame = rng.uniform(0, 255, (47, 49, 3)).astype(dtype)
        else:
            frame = rng.integers(0, np.iinfo(dtype).max, (47, 49, 3), dtype=dtype)
        plane = cv2.cvtColor(bf._u16(frame).astype(np.float32) / 65535.0, cv2.COLOR_BGR2GRAY)
        gray.append(cv2.resize(plane, (24, 23), interpolation=cv2.INTER_AREA))
    gray = np.stack(gray)
    gray[:, :5] = 0  # Regions with identical energies must retain the first frame.
    gray[-1, 12, 12] += 1
    expected = _reference(gray)
    progress = []
    _compare(bf.focus_measures(gray, 0.5, use_gpu=True, progress=progress.append), expected)
    assert 'Background focus reduction completed on OpenCL' in caplog.text
    assert len(progress) >= frames + 1 and progress == sorted(progress) and progress[-1] == 1.0
    assert expected[-1][12, 12] == frames - 1


def test_forced_cpu_skips_probe_and_gpu_failure_reuses_original_energies(monkeypatch):
    gray = np.random.default_rng(52).random((8, 31, 33), dtype=np.float32)
    expected = _reference(gray)
    monkeypatch.setattr(bf, '_GPU_FOCUS_MIN_BYTES', 0)
    calls = []

    def unavailable(*args):
        calls.append(1)
        raise gpu.GpuError('device unavailable')

    monkeypatch.setattr(gpu, 'reduce_focus', unavailable)
    _compare(bf.focus_measures(gray, 0.5, use_gpu=False), expected)
    assert not calls
    _compare(bf.focus_measures(gray, 0.5, use_gpu=True), expected)
    assert calls == [1]


def test_mid_run_device_failure_discards_partial_results(monkeypatch, caplog):
    from fusion_methods._openfocus_gpu import gpu_available

    if not gpu_available():
        pytest.skip('No OpenCL GPU available')
    import pyopencl as cl

    gray = np.random.default_rng(63).random((7, 25, 23), dtype=np.float32)
    expected = _reference(gray)
    monkeypatch.setattr(bf, '_GPU_FOCUS_MIN_BYTES', 0)
    monkeypatch.setattr(gpu, '_band_pixels', lambda *args: 50)
    copy = cl.enqueue_copy

    def fail_second_upload(queue, destination, source, **kwargs):
        if kwargs.get('host_origin') == (50 * 4, 0):
            raise cl.RuntimeError('second band failed')
        return copy(queue, destination, source, **kwargs)

    monkeypatch.setattr(cl, 'enqueue_copy', fail_second_upload)
    _compare(bf.focus_measures(gray, 0.5, use_gpu=True), expected)
    assert 'second band failed' in caplog.text


def test_reduce_focus_reuses_cached_program_and_drains_queue_on_cancel(monkeypatch):
    from fusion_methods._openfocus_gpu import gpu_available

    if not gpu_available():
        pytest.skip('No OpenCL GPU available')
    import pyopencl as cl

    gpu._PROGRAM_CACHE.clear()
    energy = np.random.default_rng(77).random((5, 19, 17), dtype=np.float32)
    builds = []
    finishes = []
    real_program = cl.Program

    class ObservedProgram(real_program):
        def build(self, *args, **kwargs):
            builds.append(1)
            return super().build(*args, **kwargs)

    real_queue_cls = cl.CommandQueue

    class ObservedQueue(real_queue_cls):
        def finish(self):
            finishes.append(1)
            return super().finish()

    monkeypatch.setattr(cl, 'Program', ObservedProgram)
    monkeypatch.setattr(cl, 'CommandQueue', ObservedQueue)

    out1 = gpu.reduce_focus(energy, lambda: None)
    out2 = gpu.reduce_focus(energy, lambda: None)
    assert len(builds) == 1
    assert len(finishes) == 2
    for a, b in zip(out1, out2):
        assert a.tobytes() == b.tobytes()

    # Cancellation during band execution must still drain the command queue.
    monkeypatch.setattr(gpu, '_band_pixels', lambda *args: 50)
    calls = 0

    def cancel_on_band():
        nonlocal calls
        calls += 1
        if calls >= 4:
            raise bf.BackgroundFillCancelled()

    finishes.clear()
    with pytest.raises(bf.BackgroundFillCancelled):
        gpu.reduce_focus(energy, cancel_on_band)
    assert len(finishes) == 1
    gpu._PROGRAM_CACHE.clear()
