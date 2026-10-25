"""Behavioral bounds for registration scheduling, cancellation and admission."""
from dataclasses import replace
from threading import Event, Lock, Thread

import cv2
import numpy as np
import pytest

import core.registration as registration
from core import resource_policy


def _stack(count=4, dtype=np.uint8):
    rng = np.random.default_rng(830)
    gray = cv2.resize(rng.integers(20, 230, (12, 16), dtype=np.uint8),
                      (128, 96), interpolation=cv2.INTER_CUBIC)
    base = np.repeat(gray[..., None], 3, axis=2)
    frames = [cv2.warpAffine(base, np.float32([[1, 0, i], [0, 1, i / 2]]),
                            (128, 96), borderMode=cv2.BORDER_REFLECT)
              for i in range(count)]
    if dtype == np.uint16:
        frames = [image.astype(np.uint16) * 257 + 11 for image in frames]
    return frames


def test_ordered_scheduler_never_prefetches_the_whole_source():
    consumed = []

    def source():
        for index in range(30):
            consumed.append(index)
            yield np.full((8, 8), index, np.float32)

    with registration._bounded_registration_map(np.square, source(), 3) as results:
        first = next(results)
        assert len(consumed) == 3
        np.testing.assert_array_equal(first, 0)
        for index, result in enumerate(results, 1):
            assert len(consumed) <= index + 3
            np.testing.assert_array_equal(result, index * index)
    assert len(consumed) == 30


def test_cancelled_scheduler_stops_replenishing_and_discards_native_result():
    cancelled = Event()
    started = []

    def square(value):
        started.append(value)
        result = np.square(np.full((32, 32), value, np.float32))
        cancelled.set()
        return result

    with pytest.raises(registration.RegistrationCancelled):
        with registration._bounded_registration_map(square, range(20), 1,
                                                      cancelled.is_set) as results:
            list(results)
    assert started == [0]


@pytest.mark.parametrize('method', ['homography', 'ecc', 'openfocus'])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_worker_count_preserves_pixels_and_frame_order(method, dtype):
    frames = _stack(dtype=dtype)
    serial = registration.ImageRegistration(method).process(frames, thread_count=1)
    parallel = registration.ImageRegistration(method).process(frames, thread_count=3)
    assert len(parallel) == len(frames)
    for expected, actual in zip(serial, parallel):
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, expected)


def test_homography_failed_frame_keeps_last_accepted_reference():
    base, shifted = _stack(count=2)
    blank = np.zeros_like(base)
    result = registration.ImageRegistration('homography').process(
        [base, blank, shifted], thread_count=2)
    assert not np.any(result[1])
    raw_error = np.abs(base[10:-10, 10:-10].astype(float)
                       - shifted[10:-10, 10:-10]).mean()
    aligned_error = np.abs(result[0][10:-10, 10:-10].astype(float)
                           - result[2][10:-10, 10:-10]).mean()
    assert aligned_error < raw_error


@pytest.mark.parametrize('method', ['homography', 'ecc', 'openfocus'])
def test_cancel_after_final_native_warp_never_publishes(method, monkeypatch, tmp_path):
    cancelled = Event()
    function = 'warpAffine' if method == 'openfocus' else 'warpPerspective'
    original = getattr(cv2, function)

    def cancel_after_warp(*args, **kwargs):
        result = original(*args, **kwargs)
        # OpenFocus also uses affine warps for the estimation grid.
        if kwargs.get('flags', 0) & cv2.INTER_LANCZOS4:
            cancelled.set()
        return result

    frames = _stack(2)
    monkeypatch.setattr(cv2, function, cancel_after_warp)
    destination = tmp_path / 'aligned'
    with pytest.raises(registration.RegistrationCancelled):
        registration.ImageRegistration(method).process(
            frames, output_path=str(destination), thread_count=1,
            should_cancel=cancelled.is_set)
    assert not destination.exists()


def test_direct_registration_holds_lease_until_native_work_finishes(monkeypatch):
    frames = _stack(2)
    entered, release, cancelled, second_started = Event(), Event(), Event(), Event()
    second_acquired = Event()
    original = cv2.warpPerspective
    results, errors = [], []
    lock = Lock()
    first_call = [True]

    def held_warp(*args, **kwargs):
        with lock:
            hold = first_call[0]
            first_call[0] = False
        if hold:
            entered.set()
            assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, 'warpPerspective', held_warp)

    def run_first():
        try:
            results.append(registration.ImageRegistration('homography').process(
                frames, thread_count=1, should_cancel=cancelled.is_set))
        except Exception as error:
            errors.append(error)

    def run_second():
        second_started.set()
        try:
            # Nested same-thread acquisition in process must be reentrant.
            with resource_policy.heavy_job():
                second_acquired.set()
                results.append(registration.ImageRegistration('homography').process(
                    frames, thread_count=1))
        except Exception as error:
            errors.append(error)

    first = Thread(target=run_first)
    second = Thread(target=run_second)
    first.start()
    try:
        assert entered.wait(10)
        cancelled.set()
        second.start()
        assert second_started.wait(10)
        assert not second_acquired.wait(0.1)
        assert not results
    finally:
        release.set()
        first.join(10)
        if second.ident is not None:
            second.join(10)
    assert not first.is_alive() and not second.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], registration.RegistrationCancelled)
    assert len(results) == 1 and len(results[0]) == 2


def test_cpu_budget_caps_actual_registration_pool(monkeypatch):
    snapshot = resource_policy.probe_resources()
    monkeypatch.setattr(resource_policy, 'probe_resources',
                        lambda *args, **kwargs: replace(snapshot, physical_cores=2))
    original = registration.concurrent.futures.ThreadPoolExecutor
    pool_sizes = []

    class ObservedPool(original):
        def __init__(self, max_workers, *args, **kwargs):
            pool_sizes.append(max_workers)
            super().__init__(max_workers, *args, **kwargs)

    monkeypatch.setattr(registration.concurrent.futures, 'ThreadPoolExecutor', ObservedPool)
    result = registration.ImageRegistration('ecc').process(
        _stack(), thread_count=32, resource_preference='responsive')
    assert len(result) == 4
    assert pool_sizes and max(pool_sizes) == 1


def test_openfocus_cancels_between_search_candidates(monkeypatch):
    cancelled = Event()
    original = registration._of_ncc_score
    candidates = []

    def score_and_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        candidates.append(result)
        cancelled.set()
        return result

    monkeypatch.setattr(registration, '_of_ncc_score', score_and_cancel)
    gray = cv2.cvtColor(_stack(1)[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    with pytest.raises(registration.RegistrationCancelled):
        registration._of_estimate_similarity(gray, gray,
                                              should_cancel=cancelled.is_set)
    assert len(candidates) == 1


@pytest.mark.parametrize('method', ['homography', 'ecc', 'openfocus'])
def test_native_failure_does_not_publish_frames(method, monkeypatch, tmp_path):
    frames = _stack(2)
    destination = tmp_path / 'aligned'
    destination.mkdir()
    previous = destination / 'frame_0000.png'
    previous.write_bytes(b'previous output')
    function = 'warpAffine' if method == 'openfocus' else 'warpPerspective'
    original = getattr(cv2, function)

    def fail_after_warp(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get('flags', 0) & cv2.INTER_LANCZOS4:
            raise RuntimeError('native warp failed')
        return result

    monkeypatch.setattr(cv2, function, fail_after_warp)
    with pytest.raises(RuntimeError, match='native warp failed'):
        registration.ImageRegistration(method).process(
            frames, output_path=str(destination), thread_count=1)
    assert previous.read_bytes() == b'previous output'
    assert len(list(destination.iterdir())) == 1


def test_memory_rejection_precedes_any_native_warp(monkeypatch):
    snapshot = resource_policy.probe_resources()
    monkeypatch.setattr(resource_policy, 'probe_resources',
                        lambda *args, **kwargs: replace(snapshot, available_ram=1,
                                                       total_ram=16 * 1024 ** 3))
    with pytest.raises(resource_policy.ResourceLimitError):
        registration.ImageRegistration('ecc').process(_stack(), thread_count=4)
