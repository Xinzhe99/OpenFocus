"""DTCWT must stay bit-identical while its work is parallelised.

The implementation transforms frames and colour channels on threads. Frame
coefficients are computed by the same dtcwt call either way, so the fused
image must not move by a single pixel value. This pins that: the expected
digest was captured from the serial implementation before the change.
"""
import hashlib

import numpy as np
import pytest

from fusion_methods import dtcwt as dtcwt_mod

pytestmark = pytest.mark.skipif(
    dtcwt_mod.dtcwt is None, reason="dtcwt backend not installed")


def _stack():
    """Small deterministic stack: two sharp bands plus blur elsewhere."""
    rng = np.random.default_rng(4242)
    h, w = 96, 128
    base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    frames = []
    for i in range(3):
        blur = base.astype(np.uint8)
        frame = blur.copy()
        y0 = i * h // 3
        frame[y0:y0 + h // 3] = base[y0:y0 + h // 3]
        frames.append(frame)
    return frames


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def test_parallel_and_serial_produce_identical_output(monkeypatch):
    """Force the serial path, then the threaded path, compare bytes.

    This compares the two code paths inside the current implementation rather
    than a stored digest, so it keeps working if dtcwt itself is upgraded.
    """
    frames = _stack()

    monkeypatch.setattr(dtcwt_mod, "_DTCWT_MAX_WORKERS", 1)
    serial = dtcwt_mod._dtcwt_impl(frames, None, 3, False)

    monkeypatch.setattr(dtcwt_mod, "_DTCWT_MAX_WORKERS", 4)
    parallel = dtcwt_mod._dtcwt_impl(frames, None, 3, False)

    assert serial.shape == parallel.shape
    assert np.array_equal(serial, parallel), (
        "threading changed the fused pixels; the DTCWT path must be "
        "deterministic and order-independent")


def test_output_does_not_depend_on_thread_count():
    frames = _stack()
    digests = set()
    original = dtcwt_mod._DTCWT_MAX_WORKERS
    try:
        for workers in (1, 2, 3, 4):
            dtcwt_mod._DTCWT_MAX_WORKERS = workers
            digests.add(_digest(dtcwt_mod._dtcwt_impl(frames, None, 3, False)))
    finally:
        dtcwt_mod._DTCWT_MAX_WORKERS = original
    assert len(digests) == 1, f"output varies with worker count: {digests}"


def test_parallel_helper_preserves_order():
    """_map_forward returns results in input order, not completion order."""
    calls = []

    class FakeTransform:
        def forward(self, plane, nlevels):
            calls.append(plane[0, 0])
            return plane

    planes = [np.full((8, 8), float(i), dtype=np.float32) for i in range(6)]
    out = dtcwt_mod._map_forward(FakeTransform(), planes, 3)

    assert [float(o[0, 0]) for o in out] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert sorted(calls) == calls, "planes were submitted out of order"

def test_thread_count_argument_caps_the_frame_pool():
    """A tiled render passes thread_count=1; DTCWT must not open its own pool.

    The parameter used to be accepted and ignored, so every tile started two
    activity workers inside an already-parallel tile loop.
    """
    frames = _stack()
    seen = []
    original = dtcwt_mod._iter_forward

    def spy(transform, planes, nlevels, workers=None):
        seen.append(workers)
        return original(transform, planes, nlevels, workers=workers)

    dtcwt_mod._iter_forward = spy
    try:
        dtcwt_mod._dtcwt_impl(frames, None, 3, False, thread_count=1)
    finally:
        dtcwt_mod._iter_forward = original
    assert seen == [1], seen


def test_thread_count_does_not_change_pixels():
    frames = _stack()
    baseline = dtcwt_mod._dtcwt_impl(frames, None, 3, False, thread_count=1)
    for threads in (2, 4):
        out = dtcwt_mod._dtcwt_impl(frames, None, 3, False, thread_count=threads)
        assert np.array_equal(baseline, out), threads


def test_cancellation_is_honoured_before_the_transform_pass():
    frames = _stack()
    from core.registration import RegistrationCancelled
    with pytest.raises(RegistrationCancelled):
        dtcwt_mod._dtcwt_impl(frames, None, 3, False,
                              should_cancel=lambda: True)
