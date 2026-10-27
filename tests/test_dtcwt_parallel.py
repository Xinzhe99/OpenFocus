"""DTCWT colour reconstruction runs in a bounded, ordered pool.

The serial version transformed each channel's frames one at a time. Parallel
forwards must not change a single output pixel, must stay inside the admitted
plan, and must never open a second parallel layer under the tiled path.
"""
import threading

import cv2
import numpy as np
import pytest

from core.registration import RegistrationCancelled
from fusion_methods import dtcwt as module

pytestmark = pytest.mark.skipif(
    module.dtcwt is None, reason="dtcwt backend not installed")


def _stack(count=4, size=(66, 78)):
    """Real depth variation: each frame is sharp on a different band."""
    h, w = size
    rng = np.random.default_rng(7)
    texture = rng.integers(0, 256, (h, w), dtype=np.uint8)
    blurred = cv2.GaussianBlur(texture, (31, 31), 0)
    frames = []
    for i in range(count):
        band = np.zeros((h, w), bool)
        band[i * h // count:(i + 1) * h // count] = True
        gray = np.where(band, texture, blurred)
        frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    return frames


def _serial_reference(frames, levels=3):
    """The pre-parallel implementation: one forward per channel and frame."""
    images = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
              for f in frames]
    h, w = images[0].shape[:2]
    n = len(images)
    transform = module.dtcwt.Transform2d()
    activities = [[] for _ in range(levels)]
    for image in images:
        luma = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        pyr = transform.forward(luma, nlevels=levels)
        for level in range(levels):
            activities[level].append(module.maximum_filter(
                np.abs(pyr.highpasses[level]).sum(axis=2).astype(np.float32),
                size=3, mode="reflect"))
    decisions = []
    for level in range(levels):
        idx = np.argmax(np.stack(activities[level]), axis=0)
        votes = np.stack([
            module.convolve((idx == i).astype(np.float32),
                            np.ones((3, 3), np.float32), mode="nearest")
            for i in range(n)])
        decisions.append(np.argmax(votes, axis=0))
    weights = None
    channels = []
    for c in range(3):
        high = low = None
        for i, image in enumerate(images):
            pyr = transform.forward(image[:, :, c], nlevels=levels)
            if high is None:
                high = [np.zeros_like(hp) for hp in pyr.highpasses]
                low = np.zeros_like(pyr.lowpass)
                if weights is None:
                    shape = (low.shape[1], low.shape[0])
                    masks = [cv2.resize((decisions[-1] == k).astype(np.float32),
                                        shape, interpolation=cv2.INTER_LINEAR)
                             for k in range(n)]
                    ws = np.stack([cv2.GaussianBlur(m, (0, 0), 1.5)
                                   for m in masks])
                    weights = ws / np.maximum(ws.sum(axis=0, keepdims=True), 1e-6)
            for level in range(levels):
                selected = decisions[level] == i
                high[level][selected] = pyr.highpasses[level][selected]
            low += weights[i] * pyr.lowpass
        channels.append(transform.inverse(
            module.dtcwt.Pyramid(low, tuple(high)))[:h, :w])
    result = np.clip(np.stack(channels, axis=-1) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_colour_matches_serial_implementation(workers):
    frames = _stack()
    expected = _serial_reference(frames)
    result = module._dtcwt_impl(frames, None, 3, False, thread_count=workers)
    np.testing.assert_array_equal(result, expected)


def test_colour_phase_actually_runs_in_parallel(monkeypatch):
    """The activity pass caps itself at 2; the colour phase must exceed that.

    A two-core host admits one worker in balanced mode, so the property is not
    observable there; the stage correctly runs serially instead of oversubscribing.
    """
    from core.resource_policy import (PREFERENCE_BALANCED, cpu_capacity,
                                      probe_resources)
    if cpu_capacity(probe_resources(), PREFERENCE_BALANCED) < 3:
        pytest.skip("host admits fewer than three concurrent frame transforms")
    module._DTCWT_MAX_WORKERS = 4
    original = module.dtcwt.Transform2d.forward
    lock = threading.Lock()
    active = peak = 0

    def observed(self, *args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return original(self, *args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(module.dtcwt.Transform2d, "forward", observed)
    module._dtcwt_impl(_stack(count=6), None, 3, False, thread_count=4)
    assert peak > module._DTCWT_ACTIVITY_WORKERS
    assert peak <= 4
    assert active == 0


def test_plan_cap_is_an_upper_bound(monkeypatch):
    from core.resource_policy import ExecutionPlan

    original = module.dtcwt.Transform2d.forward
    lock = threading.Lock()
    peak = 0
    active = 0

    def observed(self, *args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return original(self, *args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(module.dtcwt.Transform2d, "forward", observed)
    plan = ExecutionPlan(workers=2, native_threads=1, max_in_flight=2,
                         device_kind="CPU")
    module._dtcwt_impl(_stack(count=6), None, 3, False, thread_count=4,
                       execution_plan=plan)
    assert 1 <= peak <= 2


def test_tiled_child_plan_is_serial(monkeypatch):
    """A tile already runs in the parent pool: no nested forward pool."""
    from core.multi_focus_fusion import MultiFocusFusion
    from core.resource_policy import ExecutionPlan

    seen = {}
    real = module._dtcwt_impl

    def observed(input_source, img_resize, N, use_gpu, thread_count=None,
                 should_cancel=None, execution_plan=None):
        seen["plan"] = execution_plan
        return real(input_source, img_resize, N, use_gpu,
                    thread_count=thread_count, should_cancel=should_cancel,
                    execution_plan=execution_plan)

    monkeypatch.setattr(module, "_dtcwt_impl", observed)
    monkeypatch.setattr(
        "core.multi_focus_fusion._dtcwt_impl", observed, raising=False)
    # DTCWT only tiles when its whole-image activity set is too large; make
    # that condition true for this small stack instead of allocating a huge one.
    monkeypatch.setattr("core.multi_focus_fusion.DTCWT_ACTIVITY_BUDGET", 1)
    fusion = MultiFocusFusion(algorithm="dtcwt", use_gpu=False,
                              tile_enabled=True, tile_block_size=32,
                              tile_overlap=8, tile_threshold=1,
                              exposure_match=False)
    fusion.fuse(_stack(count=3, size=(48, 48)), thread_count=2)
    assert seen["plan"] is not None
    assert seen["plan"].workers == 1
    assert seen["plan"].max_in_flight == 1


def test_cancel_during_colour_phase_stops_without_publishing(monkeypatch):
    cancelled = threading.Event()
    original = module.dtcwt.Transform2d.forward
    calls = []

    def observed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(1)
        if len(calls) >= 3:
            cancelled.set()
        return result

    monkeypatch.setattr(module.dtcwt.Transform2d, "forward", observed)
    with pytest.raises(RegistrationCancelled):
        module._dtcwt_impl(_stack(count=8), None, 3, False, thread_count=1,
                           should_cancel=cancelled.is_set)
    assert len(calls) < 8 * 3


def test_forward_failure_in_a_later_frame_propagates():
    calls = []

    class Boom(RuntimeError):
        pass

    original = module.dtcwt.Transform2d.forward

    def observed(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 5:
            raise Boom("forward failed")
        return original(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module.dtcwt.Transform2d, "forward", observed)
        with pytest.raises(Boom):
            module._dtcwt_impl(_stack(count=6), None, 3, False, thread_count=2)


def test_pyramid_envelope_covers_installed_backend():
    transform = module.dtcwt.Transform2d()
    for size in ((64, 64), (65, 79), (100, 150), (33, 41)):
        for levels in (1, 2, 4, 6):
            pyramid = transform.forward(
                np.zeros(size, np.float32), nlevels=levels)
            shapes, (low_h, low_w) = module.pyramid_shape_envelope(
                size[0], size[1], levels)
            for (rh, rw), band in zip(shapes, pyramid.highpasses):
                assert band.shape[0] <= rh and band.shape[1] <= rw
            assert pyramid.lowpass.shape[0] <= low_h
            assert pyramid.lowpass.shape[1] <= low_w


def test_estimator_scales_with_frames_and_workers():
    fixed, per_worker = module.estimate_working_set(512, 512, 4, 8)
    assert fixed > 0 and per_worker > 0
    fixed_more, _ = module.estimate_working_set(512, 512, 4, 16)
    assert fixed_more > fixed
    fixed_new, _ = module.estimate_working_set(
        512, 512, 4, 8, new_source_bytes=1024 * 1024)
    assert fixed_new == fixed + 1024 * 1024
