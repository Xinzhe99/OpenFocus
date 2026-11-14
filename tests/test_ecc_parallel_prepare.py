"""ECC prepares frames in parallel within the stage's thread budget."""
import threading
import time
import weakref

import numpy as np
import pytest

from core import registration, source_frames
from core.registration import RegistrationCancelled, _prepare_and_estimate_pairs


class _Recorder:
    def __init__(self, prepare_delay=0.02, estimate_delay=0.005):
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0
        self.prepares_running = 0
        self.peak_prepares = 0
        self.retained = set()
        self.peak_retained = 0
        self.prepare_delay = prepare_delay
        self.estimate_delay = estimate_delay
        self.pairs = []

    def _enter(self, prepare):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
            if prepare:
                self.prepares_running += 1
                self.peak_prepares = max(self.peak_prepares, self.prepares_running)

    def _leave(self, prepare):
        with self.lock:
            self.running -= 1
            if prepare:
                self.prepares_running -= 1

    def prepare(self, value):
        self._enter(True)
        try:
            time.sleep(self.prepare_delay * (1 + value % 3))
            with self.lock:
                self.retained.add(value)
                self.peak_retained = max(self.peak_retained, len(self.retained))
            return (value * 10, f'id-{value}')
        finally:
            self._leave(True)

    def estimate(self, index, previous, current):
        self._enter(False)
        try:
            time.sleep(self.estimate_delay)
            with self.lock:
                self.retained.discard(index - 1)
                self.pairs.append(index)
            return (index, previous[0], current[0])
        finally:
            self._leave(False)


@pytest.mark.parametrize('workers', [2, 3, 6])
def test_results_keep_pair_order_and_identities_keep_frame_order(workers):
    recorder = _Recorder()
    values = list(range(11))
    results, identities = _prepare_and_estimate_pairs(
        values, recorder.prepare, recorder.estimate, workers)
    assert results == [(index, (index - 1) * 10, index * 10) for index in range(1, 11)]
    assert identities == [f'id-{value}' for value in values]
    assert sorted(recorder.pairs) == list(range(1, 11))


@pytest.mark.parametrize('workers', [2, 4])
def test_never_more_tasks_than_the_thread_budget_and_preparation_overlaps(workers):
    recorder = _Recorder()
    _prepare_and_estimate_pairs(list(range(12)), recorder.prepare, recorder.estimate, workers)
    assert recorder.peak <= workers
    # The point of the change: several frames are prepared (decoded) at once.
    assert recorder.peak_prepares >= 2


def test_prepared_frames_ahead_of_the_estimates_stay_within_the_window():
    recorder = _Recorder(prepare_delay=0.002, estimate_delay=0.02)
    _prepare_and_estimate_pairs(list(range(16)), recorder.prepare, recorder.estimate, 3,
                                window=4)
    assert recorder.peak_retained <= 4


def test_prepared_pixel_owners_are_released_within_the_window():
    class Pixels:
        pass

    live = weakref.WeakSet()
    lock = threading.Lock()
    peak = 0

    def prepare(value):
        nonlocal peak
        pixels = Pixels()
        with lock:
            live.add(pixels)
            peak = max(peak, len(live))
        return pixels, value

    def estimate(index, previous, current):
        time.sleep(0.002)
        return index

    _prepare_and_estimate_pairs(list(range(40)), prepare, estimate, 3, window=4)
    assert peak <= 4
    assert not live


def test_out_of_order_estimates_keep_their_later_frame_counted():
    # Pair 1 stalls while later pairs finish. Its later frame left the
    # preparation map when pair 2 started, but pair 1 still holds it.
    class Pixels:
        pass

    live = weakref.WeakSet()
    lock = threading.Lock()
    peak = 0
    release = threading.Event()
    timer = threading.Timer(0.3, release.set)

    def prepare(value):
        nonlocal peak
        pixels = Pixels()
        with lock:
            live.add(pixels)
            peak = max(peak, len(live))
        return pixels, value

    def estimate(index, previous, current):
        if index == 1:
            release.wait(5)
        return index

    timer.start()
    try:
        results, _ = _prepare_and_estimate_pairs(list(range(12)), prepare, estimate, 3,
                                                 window=4)
    finally:
        timer.cancel()
        release.set()
    assert results == list(range(1, 12))
    assert peak <= 4


def test_on_prepared_sees_each_frame_once():
    seen = []
    lock = threading.Lock()

    def record(index, prepared):
        with lock:
            seen.append((index, prepared[0]))

    recorder = _Recorder(prepare_delay=0.001, estimate_delay=0.001)
    _prepare_and_estimate_pairs(list(range(6)), recorder.prepare, recorder.estimate, 2,
                                on_prepared=record)
    assert sorted(seen) == [(index, index * 10) for index in range(6)]


def test_a_failed_preparation_stops_the_run_and_running_calls_finish_first():
    finished = []

    def prepare(value):
        if value == 3:
            raise ValueError('decode failed')
        time.sleep(0.03)
        finished.append(value)
        return (value, value)

    def estimate(index, previous, current):
        return index

    started = time.perf_counter()
    with pytest.raises(ValueError, match='decode failed'):
        _prepare_and_estimate_pairs(list(range(40)), prepare, estimate, 3)
    assert time.perf_counter() - started < 5
    snapshot = list(finished)
    time.sleep(0.1)
    # Nothing keeps running after the call returned.
    assert finished == snapshot
    assert max(snapshot) < 10


@pytest.mark.parametrize('error_type', [ValueError, RegistrationCancelled])
def test_late_preparation_failure_preserves_the_original_error(monkeypatch, error_type):
    """A frame can fail between the scheduler's scan and pair submission."""
    import concurrent.futures

    release = threading.Event()
    futures = []
    real_executor = concurrent.futures.ThreadPoolExecutor
    real_exception = concurrent.futures.Future.exception

    class TrackingExecutor(real_executor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            if len(futures) < 2:
                futures.append(future)
            return future

    def exception_after_failure(future, timeout=None):
        if len(futures) == 2 and future is futures[1] and not release.is_set():
            release.set()
            with pytest.raises(error_type, match='late preparation failure'):
                futures[0].result(timeout=5)
        return real_exception(future, timeout)

    def prepare(value):
        if value == 0:
            assert release.wait(5)
            raise error_type('late preparation failure')
        return value, value

    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', TrackingExecutor)
    monkeypatch.setattr(concurrent.futures.Future, 'exception', exception_after_failure)
    try:
        with pytest.raises(error_type, match='late preparation failure'):
            _prepare_and_estimate_pairs([0, 1], prepare, lambda i, p, c: i, 2)
    finally:
        release.set()


def test_cancellation_is_raised_and_leaves_no_running_task():
    cancel = threading.Event()
    running = []

    def prepare(value):
        running.append(value)
        time.sleep(0.02)
        if value == 4:
            cancel.set()
        running.remove(value)
        return (value, value)

    with pytest.raises(RegistrationCancelled):
        _prepare_and_estimate_pairs(list(range(50)), prepare, lambda i, p, c: i, 3,
                                    should_cancel=cancel.is_set)
    assert running == []


def test_fewer_than_two_values_are_refused():
    with pytest.raises(ValueError):
        _prepare_and_estimate_pairs([1], lambda v: (v, v), lambda i, p, c: i, 2)


@pytest.mark.parametrize('outcome', ['success', 'prepare_error', 'estimate_error', 'cancel'])
def test_worker_openmp_pools_are_serial_and_owner_limit_is_preserved(outcome):
    # OpenMP limits can be thread-local or process-wide, depending on the runtime.
    from threadpoolctl import threadpool_info, threadpool_limits

    def limits():
        return {pool['filepath']: pool['num_threads'] for pool in threadpool_info()
                if pool['user_api'] == 'openmp'}

    if not limits():
        pytest.skip('No OpenMP runtime installed')
    cancelled = threading.Event()
    with threadpool_limits(limits=3, user_api='openmp'):
        before = limits()

        def prepare(value):
            assert limits()
            assert set(limits().values()) == {1}
            if value == 1 and outcome == 'prepare_error':
                raise ValueError('decode failed')
            if value == 1 and outcome == 'cancel':
                cancelled.set()
            return value, value

        def estimate(index, previous, current):
            if outcome == 'estimate_error':
                raise RuntimeError('estimation failed')
            return index

        if outcome == 'success':
            _prepare_and_estimate_pairs(list(range(8)), prepare, estimate, 3,
                                        should_cancel=cancelled.is_set)
        else:
            error = {'prepare_error': ValueError, 'estimate_error': RuntimeError,
                     'cancel': RegistrationCancelled}[outcome]
            with pytest.raises(error):
                _prepare_and_estimate_pairs(list(range(8)), prepare, estimate, 3,
                                            should_cancel=cancelled.is_set)
        assert limits() == before


def test_ecc_openmp_scope_preserves_concurrent_blas_change():
    """A concurrent BLAS limit is not reset when ECC's OpenMP scope exits."""
    import subprocess
    import sys

    script = '''
import sys
from threadpoolctl import ThreadpoolController
from core.registration import _prepare_and_estimate_pairs

controller = ThreadpoolController()
libraries = [item for item in controller.select(user_api="blas").info()
             if item.get("internal_api") == "openblas"
             and item.get("threading_layer") in ("pthreads", "windows")]
if not libraries:
    sys.exit(77)
target = controller.select(filepath=libraries[0]["filepath"])
before = target.info()[0]["num_threads"]
if before is None:
    sys.exit(77)
desired = 1 if before != 1 else 2
observed = []

def prepare(value):
    if value == 0:
        target.limit(limits=desired)
        observed.append(target.info()[0]["num_threads"])
    return value, value

_prepare_and_estimate_pairs([0, 1], prepare, lambda i, previous, current: i, 2)
if observed != [desired]:
    sys.exit(77)
assert target.info()[0]["num_threads"] == desired
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                            text=True, timeout=30)
    if result.returncode == 77:
        pytest.skip('No independently adjustable process-wide OpenBLAS runtime')
    assert result.returncode == 0, result.stdout + result.stderr


def test_failed_estimate_drains_preparation_before_returning():
    started = threading.Event()
    finished = threading.Event()

    def prepare(value):
        if value == 2:
            started.set()
            time.sleep(0.05)
            finished.set()
        return value, value

    def estimate(index, previous, current):
        assert started.wait(2)
        raise ValueError('estimation failed')

    with pytest.raises(ValueError, match='estimation failed'):
        _prepare_and_estimate_pairs(list(range(10)), prepare, estimate, 3)
    assert finished.is_set()


def _stack(tmp_path, count=5):
    import cv2
    rng = np.random.default_rng(5)
    base = cv2.GaussianBlur(rng.integers(0, 65535, (220, 260)).astype(np.float32), (0, 0), 4)
    base = cv2.normalize(base, None, 2000, 60000, cv2.NORM_MINMAX)
    sources = []
    for index in range(count):
        shift = np.float32([[1.0 + 0.002 * index, 0, 1.5 * index],
                            [0, 1.0 + 0.002 * index, -1.0 * index]])
        frame = cv2.warpAffine(base, shift, (260, 220), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        frame = np.dstack([frame, frame * 0.9, frame * 0.8]).astype(np.uint16)
        path = tmp_path / f'frame-{index}.png'
        assert cv2.imwrite(str(path), frame)
        sources.append(source_frames.scan_source(str(path), header_only=True)[0])
    return sources


@pytest.mark.parametrize('defer', [False, True])
def test_ecc_gives_identical_frames_at_every_thread_count(tmp_path, defer):
    sources = _stack(tmp_path)
    outputs = {}
    for threads in (1, 2, 4):
        source_frames.SOURCE_CACHE.clear()
        engine = registration.ImageRegistration('ecc', downscale_width=120, defer_warps=defer)
        frames = engine.process(sources, thread_count=threads)
        outputs[threads] = [np.array(source_frames.resolve_frame(frame)) for frame in frames]
    for threads in (2, 4):
        assert len(outputs[threads]) == len(outputs[1])
        for expected, actual in zip(outputs[1], outputs[threads]):
            np.testing.assert_array_equal(actual, expected)


def test_ecc_prepares_on_pool_threads_when_given_several(tmp_path, monkeypatch):
    sources = _stack(tmp_path, count=6)
    from core import resource_policy

    snapshot = resource_policy.ResourceSnapshot(
        available_ram=4 * 1024**3, total_ram=8 * 1024**3,
        physical_cores=4, logical_cpus=4, physical_cores_known=True)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *args, **kwargs: snapshot)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    state = {'running': 0, 'peak': 0, 'threads': set(), 'calls': 0}
    original = source_frames.resolve_frame

    def slow_resolve(frame, *args, **kwargs):
        with lock:
            state['running'] += 1
            state['peak'] = max(state['peak'], state['running'])
            state['threads'].add(threading.get_ident())
            state['calls'] += 1
            first_pair = state['calls'] <= 2
        try:
            if first_pair:
                barrier.wait(timeout=5)
            return original(frame, *args, **kwargs)
        finally:
            with lock:
                state['running'] -= 1

    monkeypatch.setattr(source_frames, 'resolve_frame', slow_resolve)
    source_frames.SOURCE_CACHE.clear()
    engine = registration.ImageRegistration('ecc', downscale_width=120, defer_warps=True)
    engine.process(sources, thread_count=3)
    assert state['peak'] >= 2
    assert threading.get_ident() not in state['threads']
