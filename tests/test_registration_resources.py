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
    if method == 'openfocus':
        from core import openfocus_alignment
        target, function = openfocus_alignment, 'apply_alignment'
    else:
        target, function = cv2, 'warpPerspective'
    original = getattr(target, function)

    def cancel_after_warp(image, transform, *args, **kwargs):
        result = original(image, transform, *args, **kwargs)
        if (method == 'openfocus' and transform.source_index == len(frames) - 1
                or method != 'openfocus' and kwargs.get('flags', 0) & cv2.INTER_LANCZOS4):
            cancelled.set()
        return result

    frames = _stack(2)
    monkeypatch.setattr(target, function, cancel_after_warp)
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


def test_openfocus_cancels_between_search_candidates():
    calls = 0

    def should_cancel():
        nonlocal calls
        calls += 1
        return calls >= 3

    from core.openfocus_alignment import align_pair_wide
    gray = cv2.cvtColor(_stack(1)[0], cv2.COLOR_BGR2GRAY).astype(np.uint16)
    with pytest.raises(registration.RegistrationCancelled):
        align_pair_wide(gray, gray, should_cancel=should_cancel)
    assert calls == 3


@pytest.mark.parametrize('method', ['homography', 'ecc'])
def test_native_failure_does_not_publish_frames(method, monkeypatch, tmp_path):
    frames = _stack(2)
    destination = tmp_path / 'aligned'
    destination.mkdir()
    previous = destination / 'frame_0000.png'
    previous.write_bytes(b'previous output')
    function = 'warpPerspective'
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


def test_directory_registration_checks_memory_before_decode(tmp_path, monkeypatch):
    from core import image_loader

    assert cv2.imwrite(str(tmp_path / 'frame.png'), np.zeros((8, 8, 3), np.uint8))
    snapshot = resource_policy.probe_resources()
    monkeypatch.setattr(resource_policy, 'probe_resources',
                        lambda *args, **kwargs: replace(snapshot, available_ram=1,
                                                       total_ram=16 * 1024 ** 3))
    decoded = []
    monkeypatch.setattr(image_loader, 'decode_image', lambda path: decoded.append(path))
    with pytest.raises(resource_policy.ResourceLimitError):
        registration.ImageRegistration('ecc').process(str(tmp_path))
    assert not decoded


@pytest.mark.parametrize('frame_count', [2, 100])
@pytest.mark.parametrize('headroom_mib,admitted', [(300, True), (200, False)])
def test_combined_alignment_admits_disk_backed_stack_by_active_working_set(
        monkeypatch, frame_count, headroom_mib, admitted):
    from core import source_frames

    # Isolate admission from disk materialization and native alignment. The
    # broadcast frame describes real geometry without allocating the stack.
    frame = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (2048, 2048, 3))
    frames = [frame] * frame_count
    snapshot = resource_policy.ResourceSnapshot(
        available_ram=1024 ** 3 + headroom_mib * 1024 ** 2,
        total_ram=4 * 1024 ** 3, physical_cores=8, logical_cpus=8)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(source_frames, 'materialize_frames',
                        lambda images, should_cancel, **kwargs: images)
    plans = []
    plan_execution = resource_policy.plan_execution

    def plan(workload, **kwargs):
        execution = plan_execution(workload, **kwargs)
        plans.append((workload, execution))
        return execution

    monkeypatch.setattr(resource_policy, 'plan_execution', plan)
    dispatched = []

    def align(self, images, **kwargs):
        dispatched.append(kwargs['thread_count'])
        return images

    monkeypatch.setattr(registration.ImageRegistration, '_process_homography', align)
    monkeypatch.setattr(registration.ImageRegistration, '_process_ecc', align)
    engine = registration.ImageRegistration('both')
    if not admitted:
        with pytest.raises(resource_policy.ResourceLimitError):
            engine.process(frames, thread_count=8, resource_preference='maximum')
        assert dispatched == []
        return

    result = engine.process(frames, thread_count=8, resource_preference='maximum')
    assert len(result) == frame_count
    assert len(plans) == 1
    workload, execution = plans[0]
    fixed, per_worker = resource_policy.estimate_working_set(workload)
    assert fixed + per_worker * execution.workers <= execution.ram_budget
    assert fixed + per_worker * (execution.workers + 1) > execution.ram_budget
    assert dispatched == [1, 1]


@pytest.mark.parametrize('outcome', ['aligned', 'native', 'fallback', 'reject', 'fail', 'cancel'])
def test_deferred_ecc_keeps_solvers_parallel_with_one_decode_slot(tmp_path, monkeypatch, outcome):
    from threading import Barrier
    from core import source_frames

    image = np.broadcast_to(np.arange(3073, dtype=np.uint16)[None, :, None], (321, 3073, 3))
    path = tmp_path / 'frame.png'
    assert cv2.imwrite(str(path), image)
    source = source_frames.scan_source(str(path), header_only=True)[0]
    mib = 1024 ** 2
    # Models a source with expensive decode or upstream processing. At 72 MiB
    # there is room for one preparation or three solvers, but not for both
    # kinds of work at once. At 64 MiB even one preparation cannot fit.
    scratch = 66 * mib - source_frames.frame_resolution_bytes(source)
    assert scratch > 0
    # Distinct prepared planes keep the exact-pair cache from skipping a batch.
    frames = [source_frames.process_frame(source, lambda pixels, offset=i: pixels + offset,
                                         scratch_bytes=scratch) for i in range(7)]
    snapshot = resource_policy.ResourceSnapshot(
        available_ram=1024 ** 3 + (64 if outcome == 'reject' else 72) * mib,
        total_ram=4 * 1024 ** 3, physical_cores=3, logical_cpus=3)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *args, **kwargs: snapshot)

    cancelled, together, lock = Event(), Barrier(3), Lock()
    preparing = solving = 0
    prepared = estimated = 0
    original = source_frames.ProcessedFrame.full_resolution

    def resolve(frame, **kwargs):
        nonlocal preparing, prepared
        with lock:
            assert preparing == solving == 0
            preparing += 1
        try:
            return original(frame, **kwargs)
        finally:
            with lock:
                preparing -= 1
                prepared += 1

    def estimate(previous, current, matrix, *_args):
        nonlocal solving, estimated
        assert previous.shape == current.shape == (107, 1024)
        assert previous.dtype == current.dtype == np.float32
        with lock:
            assert preparing == 0
            solving += 1
            estimated += 1
        try:
            together.wait(timeout=10)
            if outcome in ('fail', 'fallback'):
                raise cv2.error('injected failure to converge')
            if outcome == 'cancel':
                cancelled.set()
            return 1.0, matrix
        finally:
            with lock:
                solving -= 1

    monkeypatch.setattr(source_frames.ProcessedFrame, 'full_resolution', resolve)
    if outcome == 'native':
        def solve_native(previous, current, *, max_iter, eps):
            return estimate(previous, current, np.eye(3, dtype=np.float32))

        monkeypatch.setattr(registration, '_load_ecc_native', lambda: solve_native)
    else:
        monkeypatch.setattr(cv2, 'findTransformECC', estimate)
    engine = registration.ImageRegistration('ecc', defer_warps=True, downscale_width=200,
        parameters={'failure_policy': 'continue' if outcome == 'fallback' else 'fail'})
    destination = tmp_path / 'aligned'

    def process(output_path=None):
        return engine.process(frames, output_path=output_path, thread_count=3,
                              resource_preference='maximum', should_cancel=cancelled.is_set)

    if outcome in ('aligned', 'native', 'fallback'):
        report = []
        token = registration.alignment_report.set(report)
        try:
            output = process()
        finally:
            registration.alignment_report.reset(token)
        status = 'fallback' if outcome == 'fallback' else 'aligned'
        assert [item['status'] for item in report] == ['reference'] + [status] * 6
        assert estimated == 6
        assert prepared == len(output) == len(frames)
        assert engine.alignment_plan.stages[0].bounds == (2, 319, 2, 3071)
        monkeypatch.setattr(source_frames.ProcessedFrame, 'full_resolution', original)
        # Rendering the deferred result has its own full-resolution budget.
        render_snapshot = replace(snapshot, available_ram=1024 ** 3 + 128 * mib)
        monkeypatch.setattr(resource_policy, 'probe_resources', lambda *args, **kwargs: render_snapshot)
        pixels = source_frames.resolve_frame(output[-1])
        assert pixels.dtype == np.uint16
        np.testing.assert_array_equal(pixels, image[2:-2, 2:-2] + len(frames) - 1)
    else:
        errors = {'reject': resource_policy.ResourceLimitError,
                  'fail': registration.RejectedAlignmentFrame,
                  'cancel': registration.RegistrationCancelled}
        with pytest.raises(errors[outcome]):
            process(str(destination))
        assert prepared == (0 if outcome == 'reject' else 4)
        assert engine.alignment_plan is None
        assert not destination.exists()
    assert preparing == solving == 0


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('shape', [(1, 9, 3), (37, 41, 3), (259, 311, 3)])
def test_native_gray_keeps_equal_mean_and_uint16_codes(dtype, shape):
    from core.openfocus_alignment import _gray16
    rng = np.random.default_rng(177)
    image = rng.integers(0, 256, shape, dtype=np.uint16).astype(dtype)
    if dtype == np.uint16:
        image *= 257
    expected = image.sum(axis=2, dtype=np.uint32) // 3
    if dtype == np.uint8:
        expected <<= 8
    np.testing.assert_array_equal(_gray16(image, None), expected)


def test_grayscale_preparation_cancels_between_rows():
    from core.openfocus_alignment import _gray16
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls == 3

    with pytest.raises(registration.RegistrationCancelled):
        _gray16(np.zeros((513, 97, 3), np.uint16), cancelled)
    assert calls == 3


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_openfocus_fixed_brightness_preserves_identity_codes(dtype):
    frame = np.full((32, 40, 3), 80, dtype)
    output = registration._align_openfocus_impl([frame, frame.copy()])
    for image in output:
        np.testing.assert_array_equal(image, frame)


@pytest.mark.parametrize('dtype,width', [(np.uint8, 64), (np.uint16, 128), (np.float32, 128)])
def test_ecc_releases_deferred_color_and_preserves_estimator_inputs(monkeypatch, dtype, width):
    from weakref import ref
    from core.source_frames import process_frame

    rng = np.random.default_rng(182)
    base = rng.integers(0, 65536, (96, 128, 3), dtype=np.uint16).astype(dtype)
    if dtype == np.float32:
        base /= 257.0
    images = [base, np.roll(base, 1, axis=1)]
    expected = []
    for image in images:
        if width < image.shape[1]:
            scale = width / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        expected.append(gray.astype(np.float32) / (65535.0 if dtype == np.uint16 else 255.0))

    decoded = []
    def produce(image):
        pixels = image.copy()
        decoded.append(ref(pixels))
        return pixels

    blur = cv2.GaussianBlur
    def check_lifetime(*args):
        assert decoded[-1]() is None  # only the gray plane is needed at this point
        return blur(*args)

    estimate = cv2.findTransformECC
    def check_pair(previous, current, *args):
        for actual, reference in zip((previous, current), expected):
            np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
        return estimate(previous, current, *args)

    monkeypatch.setattr(cv2, 'GaussianBlur', check_lifetime)
    monkeypatch.setattr(cv2, 'findTransformECC', check_pair)
    sources = [process_frame(image, produce) for image in images]
    registration._align_ecc_impl(sources, thread_count=1, downscale_width=width, defer_warps=True)
    assert len(decoded) == 2


@pytest.mark.parametrize('workers', [1, 2, 3])
def test_ecc_preparation_and_estimation_share_budget(monkeypatch, workers):
    from time import sleep

    lock = Lock()
    active = 0
    peak = 0
    estimate_started = Event()
    prepare_overlap = Event()
    original_blur = cv2.GaussianBlur

    def enter():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)

    def leave():
        nonlocal active
        with lock:
            active -= 1

    def blur(*args, **kwargs):
        enter()
        try:
            # Frame 2 can overlap estimation of the already prepared pair 0–1.
            if workers > 1 and args[0][0, 0] == 2:
                assert estimate_started.wait(10)
                prepare_overlap.set()
            sleep(0.005)
            return original_blur(*args, **kwargs)
        finally:
            leave()

    def estimate(_prev, _curr, matrix, *_args):
        enter()
        try:
            estimate_started.set()
            if workers > 1:
                assert prepare_overlap.wait(10)
            sleep(0.01)
            return 1.0, matrix
        finally:
            leave()

    frames = _stack(8)
    for index, frame in enumerate(frames):
        frame[0, 0] = index
    monkeypatch.setattr(cv2, 'GaussianBlur', blur)
    monkeypatch.setattr(cv2, 'findTransformECC', estimate)
    result = registration._align_ecc_impl(frames, thread_count=workers, defer_warps=True)
    assert len(result) == len(frames)
    assert peak <= workers
    if workers > 1:
        assert peak >= 2


def test_registration_forwards_preparation_resources(monkeypatch):
    from core import source_frames
    seen = []
    original = source_frames.materialize_frames

    def materialize(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(source_frames, 'materialize_frames', materialize)
    registration.ImageRegistration('none').process(
        _stack(2), thread_count=1, resource_preference='responsive')
    assert seen[0]['thread_count'] == 1
    assert seen[0]['resource_preference'] == 'responsive'


def test_deferred_warp_budget_requires_known_active_backend(monkeypatch):
    monkeypatch.setattr(cv2, 'currentParallelFramework', lambda: 'GCD', raising=False)
    with registration.deferred_warp_threads(3):
        assert registration._warp_thread_budget() == 1
        monkeypatch.setattr(cv2, 'currentParallelFramework', lambda: 'TBB')
        assert registration._warp_thread_budget() == 3
        with registration.deferred_warp_threads(1):
            assert registration._warp_thread_budget() == 1
        assert registration._warp_thread_budget() == 3
    assert registration._warp_thread_budget() == 1


def _without_framework_query(monkeypatch, framework):
    monkeypatch.delattr(cv2, 'currentParallelFramework', raising=False)
    monkeypatch.delenv('OPENCV_PARALLEL_BACKEND', raising=False)
    monkeypatch.delenv('OPENCV_PARALLEL_PRIORITY_LIST', raising=False)
    monkeypatch.setattr(cv2, 'getBuildInformation', lambda: (
        'General configuration for OpenCV 5.0.0 =====\n'
        '  Parallel framework:            %s\n'
        '  Trace:                         YES (with Intel ITT)\n' % framework))


@pytest.mark.parametrize('framework,lent', [
    ('pthreads', 3), ('TBB (ver 2022.0 interface 12140)', 3), ('OpenMP', 3),
    ('Concurrency', 3), ('GCD', 1), ('HPX', 1), ('NO', 1), ('', 1)])
def test_deferred_warp_budget_reads_the_build_when_bindings_lack_the_query(
        monkeypatch, framework, lent):
    # Published Python packages do not bind currentParallelFramework().
    _without_framework_query(monkeypatch, framework)
    with registration.deferred_warp_threads(3):
        assert registration._warp_thread_budget() == lent
    assert registration._warp_thread_budget() == 1


def test_deferred_warp_budget_stays_serial_for_unknown_environment_plugins(monkeypatch):
    _without_framework_query(monkeypatch, 'pthreads')
    with registration.deferred_warp_threads(3):
        monkeypatch.setenv('OPENCV_PARALLEL_BACKEND', 'ONETBB')
        assert registration._warp_thread_budget() == 3
        monkeypatch.setenv('OPENCV_PARALLEL_PRIORITY_LIST', 'tbb,custom_pool')
        assert registration._warp_thread_budget() == 1


def test_installed_opencv_lends_warp_threads_on_a_bounded_backend(monkeypatch):
    # The installed bindings, unpatched: a bounded build must lend threads.
    monkeypatch.delenv('OPENCV_PARALLEL_BACKEND', raising=False)
    monkeypatch.delenv('OPENCV_PARALLEL_PRIORITY_LIST', raising=False)
    framework = next((line.partition(':')[2] for line in cv2.getBuildInformation().splitlines()
                      if line.partition(':')[0].strip() == 'Parallel framework'), '')
    bounded = registration._bounded_backend_name(
        cv2.currentParallelFramework() if hasattr(cv2, 'currentParallelFramework') else framework)
    with registration.deferred_warp_threads(3):
        assert registration._warp_thread_budget() == (3 if bounded else 1)
