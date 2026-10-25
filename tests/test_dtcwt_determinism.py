"""DTCWT Q0, cancellation and resource bounds using real wavelet transforms."""
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref

import cv2
import numpy as np
import pytest

from core.registration import RegistrationCancelled
from fusion_methods import dtcwt as module
from utils.image_utils import normalize_fuse_input, quantize_fuse_output

pytestmark = pytest.mark.skipif(module.dtcwt is None, reason="dtcwt backend not installed")


def _stack(kind="bgr", count=4):
    rng = np.random.default_rng(4242)
    dtype = np.uint16 if kind == "uint16" else np.uint8
    frames = [rng.integers(0, 65536 if dtype == np.uint16 else 256,
                           (66, 78, 3), dtype=dtype)[::2, ::-2] for _ in range(count)]
    if kind == "gray":
        frames = [f[:, :, 0] for f in frames]
    elif kind == "alpha":
        frames = [np.concatenate((f, np.full((*f.shape[:2], 1), 219, np.uint8)), axis=2)
                  for f in frames]
    return frames


def _reference(frames, levels):
    """Baseline 4659b7bc: stacked activity/votes, original lowpass summation."""
    images = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for f in frames]
    h, w = images[0].shape[:2]
    transform = module.dtcwt.Transform2d()
    activities = [[] for _ in range(levels)]
    for image in images:
        luma = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        pyramid = transform.forward(luma, nlevels=levels)
        for level in range(levels):
            activity = np.abs(pyramid.highpasses[level]).sum(axis=2).astype(np.float32)
            activities[level].append(module.maximum_filter(activity, size=3, mode="reflect"))
    decisions = []
    for activity in activities:
        index = np.argmax(np.stack(activity), axis=0)
        votes = np.stack([module.convolve((index == i).astype(np.float32),
                                         np.ones((3, 3), np.float32), mode="nearest")
                          for i in range(len(frames))])
        decisions.append(np.argmax(votes, axis=0))
    weights = None
    channels = []
    for channel in range(3):
        high = low = None
        for i, image in enumerate(images):
            pyramid = transform.forward(image[:, :, channel], nlevels=levels)
            if high is None:
                high = [np.zeros_like(band) for band in pyramid.highpasses]
                low = np.zeros_like(pyramid.lowpass)
                if weights is None:
                    size = (low.shape[1], low.shape[0])
                    masks = [cv2.resize((decisions[-1] == k).astype(np.float32), size,
                                        interpolation=cv2.INTER_LINEAR) for k in range(len(frames))]
                    ws = np.stack([cv2.GaussianBlur(mask, (0, 0), 1.5) for mask in masks])
                    weights = ws / np.maximum(ws.sum(axis=0, keepdims=True), 1e-6)
            for level in range(levels):
                selected = decisions[level] == i
                high[level][selected] = pyramid.highpasses[level][selected]
            low += weights[i] * pyramid.lowpass
        channels.append(transform.inverse(module.dtcwt.Pyramid(low, tuple(high)))[:h, :w])
    dtype = np.uint8 if frames[0].dtype == np.uint8 else np.float32
    result = np.clip(np.stack(channels, axis=-1) * 255.0, 0, 255).astype(dtype)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


@pytest.mark.parametrize("kind", ["bgr", "uint16", "gray", "alpha"])
@pytest.mark.parametrize("levels", [2, 4])
def test_baseline_parity_odd_strided_frames_and_depth(kind, levels):
    raw = _stack(kind)
    originals = [f.copy() for f in raw]
    frames, is16 = normalize_fuse_input(raw)
    colors = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f[:, :, :3]
              for f in frames]
    expected = _reference(colors, levels)
    for workers in (1, 2, 4):
        result = module._dtcwt_impl(frames, None, levels, False, thread_count=workers)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(quantize_fuse_output(result, is16),
                                      quantize_fuse_output(expected, is16))
    for frame, original in zip(raw, originals):
        np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize("cap", [1, 2, 4])
def test_real_transforms_obey_thread_cap(monkeypatch, cap):
    original = module.dtcwt.Transform2d.forward
    lock = threading.Lock()
    threads = set()
    active = peak = 0
    caller = threading.get_ident()

    def observed(self, *args, **kwargs):
        nonlocal active, peak
        with lock:
            threads.add(threading.get_ident())
            active += 1
            peak = max(peak, active)
        try:
            return original(self, *args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(module.dtcwt.Transform2d, "forward", observed)
    module._dtcwt_impl(_stack(), None, 3, False, thread_count=cap)
    assert peak <= min(cap, module._DTCWT_ACTIVITY_WORKERS, module._DTCWT_MAX_WORKERS)
    if cap == 1:
        assert threads == {caller}
    assert active == 0


def test_module_cap_one_keeps_default_activity_pass_serial(monkeypatch):
    monkeypatch.setattr(module, "_DTCWT_MAX_WORKERS", 1)
    original = module.dtcwt.Transform2d.forward
    caller = threading.get_ident()

    def observed(self, *args, **kwargs):
        assert threading.get_ident() == caller
        return original(self, *args, **kwargs)

    monkeypatch.setattr(module.dtcwt.Transform2d, "forward", observed)
    module._dtcwt_impl(_stack(), None, 3, False)


def test_transform_window_drops_consumed_future_before_refill(monkeypatch):
    import concurrent.futures

    refs = []
    peak = 0

    class ObservedPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            nonlocal peak
            future = super().submit(*args, **kwargs)
            refs.append(weakref.ref(future))
            peak = max(peak, sum(ref() is not None for ref in refs))
            return future

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", ObservedPool)
    transform = module.dtcwt.Transform2d()
    planes = [f[:, :, 0].astype(np.float32) for f in _stack(count=7)]
    for plane, pyramid in zip(planes, module._iter_forward(
            transform, planes, 3, workers=2, transform_factory=module.dtcwt.Transform2d)):
        expected = transform.forward(plane, nlevels=3)
        np.testing.assert_array_equal(pyramid.lowpass, expected.lowpass)
        for actual, reference in zip(pyramid.highpasses, expected.highpasses):
            np.testing.assert_array_equal(actual, reference)
        del pyramid
    assert peak <= 2
    assert all(ref() is None for ref in refs)


def test_cancel_before_start_never_runs_transform(monkeypatch):
    original = module.dtcwt.Transform2d.forward
    calls = []

    def observed(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(module.dtcwt.Transform2d, "forward", observed)
    with pytest.raises(RegistrationCancelled):
        module._dtcwt_impl(_stack(), None, 3, False, should_cancel=lambda: True)
    assert calls == []


def test_cancel_after_real_transform_does_not_start_next_frame(monkeypatch):
    original = module.dtcwt.Transform2d.forward
    cancelled = threading.Event()
    calls = []

    def observed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(1)
        cancelled.set()
        return result

    monkeypatch.setattr(module.dtcwt.Transform2d, "forward", observed)
    with pytest.raises(RegistrationCancelled):
        module._dtcwt_impl(_stack(), None, 3, False, thread_count=1,
                           should_cancel=cancelled.is_set)
    assert len(calls) == 1


def test_flat_ties_preserve_baseline_lowpass():
    frames = [np.full((32, 40, 3), value, np.uint8) for value in (32, 96, 160)]
    result = module._dtcwt_impl(frames, None, 3, False, thread_count=1)
    np.testing.assert_array_equal(result, _reference(frames, 3))


def test_deep_stack_retains_high_label_decisions():
    flat = np.full((8, 8, 3), 64, np.uint8)
    sharp = np.random.default_rng(99).integers(0, 256, flat.shape, dtype=np.uint8)
    frames = [flat] * 259 + [sharp]
    expected = _reference(frames, 2)
    result = module._dtcwt_impl(frames, None, 2, False, thread_count=1)
    np.testing.assert_array_equal(result, expected)
    assert np.abs(result.astype(int) - sharp.astype(int)).mean() < 2


def test_raw_uint16_matches_normalized_work_domain():
    raw = _stack("uint16")
    frames, _ = normalize_fuse_input(raw)
    expected = module._dtcwt_impl(frames, None, 3, False, thread_count=1)
    result = module._dtcwt_impl(raw, None, 3, False, thread_count=1)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float32
