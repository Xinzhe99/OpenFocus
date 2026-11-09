"""Spatial execution, resource boundaries and worker lifetimes."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
from dataclasses import replace

import cv2
import numpy as np
import pytest

from core import frame_storage, resource_policy, source_frames
from core.registration import RegistrationCancelled
from fusion_methods import _openfocus_execution as execution
from fusion_methods import openfocus_a, openfocus_bc, openfocus_depth


@pytest.fixture
def resources(monkeypatch):
    from core import multi_focus_fusion
    snapshot = resource_policy.ResourceSnapshot(
        available_ram=4 * 1024 ** 3, total_ram=8 * 1024 ** 3,
        physical_cores=4, logical_cpus=8, cpu_limit=4,
        physical_cores_known=True)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *a, **k: snapshot)
    monkeypatch.setattr(multi_focus_fusion, 'memory_budget_bytes', lambda: 512 * 1024 ** 2)
    monkeypatch.setattr(source_frames, 'SOURCE_CACHE', source_frames.FrameCache(0))
    return snapshot


def _stack(dtype, height=513, width=515):
    rng = np.random.default_rng(5306)
    high = 256 if dtype == np.uint8 else 65536
    first = rng.integers(0, high, (height, width, 3), dtype=dtype)
    last = rng.integers(0, high, first.shape, dtype=dtype)
    # Identical sources exercise strict tie handling alongside real changes.
    return [first, first.copy(), last]


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('method', ['a', 'b', 'depth'])
def test_native_parallel_result_matches_serial_with_odd_edges(resources, monkeypatch, dtype, method):
    frames = _stack(dtype)
    before = [frame.copy() for frame in frames]
    def fuse(threads):
        if method == 'depth':
            return openfocus_depth.depth_map(frames, None, 17, 6, None,
                                             thread_count=threads, return_codes=True)
        runner = {'a': openfocus_a.openfocus_a_impl,
                  'b': openfocus_bc.openfocus_b_impl}[method]
        return runner(frames, thread_count=threads)
    serial = fuse(1)
    from fusion_methods import _openfocus_kernels, _openfocus_depth_rows
    module, name = ((_openfocus_depth_rows, 'select_activity_rows') if method == 'depth'
                    else (_openfocus_kernels, 'select_rows' if method == 'a' else 'merge_rows'))
    original = getattr(module, name)
    lock = threading.Lock()
    barrier = threading.Barrier(4)
    arrivals = 0
    identities = set()
    def native(*args):
        nonlocal arrivals
        with lock:
            slot = arrivals
            arrivals += 1
            identities.add(threading.get_ident())
        if slot < 4:
            barrier.wait(timeout=5)
        return original(*args)
    monkeypatch.setattr(module, name, native)
    parallel = fuse(4)
    np.testing.assert_array_equal(parallel, serial)
    assert len(identities) == 4
    for actual, expected in zip(frames, before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_c_parallel_matches_serial_on_odd_dimensions(resources, dtype):
    frames = _stack(dtype)
    originals = [frame.copy() for frame in frames]
    serial = openfocus_bc.openfocus_c_impl(frames, thread_count=1)
    parallel = openfocus_bc.openfocus_c_impl(frames, thread_count=4)
    np.testing.assert_array_equal(parallel, serial)
    for actual, expected in zip(frames, originals):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('thread_count,expected', [(0, 2), (1, 1), (2, 2), (9, 2)])
def test_passes_obey_affinity_and_user_bounds_and_keep_scratch(
        resources, monkeypatch, thread_count, expected):
    snapshot = replace(resources, cpu_limit=2.75)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda *a, **k: snapshot)
    frame = np.zeros((512, 512), np.uint8)
    output = np.zeros((512, 2), np.int32)
    observed = []
    previous_threads = cv2.getNumThreads()

    @execution.execution_job
    def compute():
        execution.configure_rows([frame, frame], None, 'openfocus_b', thread_count, None)
        scratch = frame_storage.capture_scratch()
        for phase in range(2):
            barrier = threading.Barrier(expected)
            identities = set()
            lock = threading.Lock()
            def workspace():
                assert frame_storage.capture_scratch() is scratch
                with lock:
                    identities.add(threading.get_ident())
                barrier.wait(timeout=5)
                return np.empty(1024, np.uint8)
            def write(start, stop, local):
                local.fill(phase + 1)
                output[start:stop, phase] = int(local[0])
            execution.run_bands(512, 64, write, scratch_bytes=1024, workspace=workspace)
            observed.append(identities)
    compute()
    np.testing.assert_array_equal(output, np.tile([1, 2], (512, 1)))
    assert [len(ids) for ids in observed] == [expected, expected]
    assert observed[0] == observed[1]
    assert cv2.getNumThreads() == previous_threads


@pytest.mark.parametrize('budget,expected', [(2 * 1024 ** 2 - 1, 1), (2 * 1024 ** 2, 2)])
def test_fresh_pass_ram_admission_at_worker_boundary(resources, monkeypatch, budget, expected):
    from core import multi_focus_fusion
    frame = np.zeros((512, 512), np.uint8)
    output = np.zeros(512, np.int32)
    identities = set()
    lock = threading.Lock()
    barrier = threading.Barrier(expected)

    @execution.execution_job
    def compute():
        execution.configure_rows([frame, frame], None, 'openfocus_b', 4, None)
        # Available job RAM changes after the full-image planes are resident.
        monkeypatch.setattr(multi_focus_fusion, 'memory_budget_bytes', lambda: budget)
        def workspace():
            with lock:
                identities.add(threading.get_ident())
            barrier.wait(timeout=5)
            return np.empty(1024 ** 2, np.uint8)
        def write(start, stop, local):
            local.fill(7)
            output[start:stop] = local[0]
        execution.run_bands(512, 64, write, scratch_bytes=1024 ** 2, workspace=workspace)
    compute()
    assert len(identities) == expected
    np.testing.assert_array_equal(output, np.full(512, 7))


@pytest.mark.parametrize('phase', ['merge', 'iir'])
@pytest.mark.parametrize('failure', ['cancel', 'error'])
def test_native_workers_join_before_cancel_or_error_escapes(resources, monkeypatch, phase, failure):
    from fusion_methods import _openfocus_kernels, _depth_kernel
    frames = _stack(np.uint16)
    original_frames = [frame.copy() for frame in frames]
    barrier = threading.Barrier(2)
    stop = threading.Event()
    lock = threading.Lock()
    active = 0
    entered = 0
    finished = 0
    scratch_states = []
    module, name = ((_openfocus_kernels, 'merge_rows') if phase == 'merge'
                    else (_depth_kernel, 'axis_block'))
    original = getattr(module, name)

    def native(*args, **kwargs):
        nonlocal active, entered, finished
        with lock:
            slot = entered
            entered += 1
            active += 1
            scratch_states.append(frame_storage.capture_scratch())
        try:
            if slot < 2:
                barrier.wait(timeout=5)
                if slot == 0:
                    stop.set()
                    if failure == 'error':
                        raise RuntimeError('injected native failure')
                else:
                    assert stop.wait(timeout=5)
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1
                finished += 1
    if phase == 'iir':
        # Preparation queries these dispatcher attributes before row execution.
        native.signatures = original.signatures
        native.compile = original.compile
    monkeypatch.setattr(module, name, native)
    cancelled = stop.is_set if failure == 'cancel' else lambda: False
    expected = RegistrationCancelled if failure == 'cancel' else RuntimeError
    with pytest.raises(expected):
        if phase == 'merge':
            depth = np.zeros(frames[0].shape[:2], np.float32)
            openfocus_bc.openfocus_b_impl(frames, thread_count=2,
                depth_override=depth, should_cancel=cancelled)
        else:
            openfocus_depth.depth_map(frames, None, 17, 6, cancelled, thread_count=2)
    assert entered >= 2
    assert active == 0 and finished == entered
    assert scratch_states and scratch_states[0] is not None
    assert all(state is scratch_states[0] for state in scratch_states)
    for actual, original_frame in zip(frames, original_frames):
        np.testing.assert_array_equal(actual, original_frame)


def test_small_fusion_uses_calling_thread(resources, monkeypatch):
    from fusion_methods import _openfocus_kernels
    frames = _stack(np.uint8, 65, 67)
    original = _openfocus_kernels.select_rows
    identities = set()
    def select(*args):
        identities.add(threading.get_ident())
        return original(*args)
    monkeypatch.setattr(_openfocus_kernels, 'select_rows', select)
    result = openfocus_a.openfocus_a_impl(frames, thread_count=4)
    assert result.shape == frames[0].shape
    assert identities == {threading.get_ident()}


def test_medium_images_use_bounded_frame_parallelism_without_row_pool(resources, monkeypatch):
    from fusion_methods import _openfocus_kernels
    frames = _stack(np.uint16, 257, 259)
    expected = openfocus_a.openfocus_a_impl(frames, thread_count=1)
    original_reduce = _openfocus_kernels.down2_rows
    original_select = _openfocus_kernels.select_rows
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    arrivals = 0
    producers = set()
    consumers = set()
    def reduce_rows(*args):
        nonlocal arrivals
        with lock:
            slot = arrivals
            arrivals += 1
            producers.add(threading.get_ident())
        if slot < 2:
            barrier.wait(timeout=5)
        return original_reduce(*args)
    def select_rows(*args):
        consumers.add(threading.get_ident())
        return original_select(*args)
    monkeypatch.setattr(_openfocus_kernels, 'down2_rows', reduce_rows)
    monkeypatch.setattr(_openfocus_kernels, 'select_rows', select_rows)
    actual = openfocus_a.openfocus_a_impl(frames, thread_count=4)
    np.testing.assert_array_equal(actual, expected)
    assert len(producers) == 2
    assert consumers == {threading.get_ident()}
    assert producers.isdisjoint(consumers)


@pytest.mark.parametrize('method', ['a', 'b', 'c'])
def test_parallel_flat_activity_ties_keep_first_source_pixels(resources, method):
    frames = [np.full((513, 515, 3), value, np.uint8) for value in (17, 201, 88)]
    runner = {'a': openfocus_a.openfocus_a_impl,
              'b': openfocus_bc.openfocus_b_impl,
              'c': openfocus_bc.openfocus_c_impl}[method]
    forward = runner(frames, thread_count=4)
    reverse = runner(frames[::-1], thread_count=4)
    expected_forward, expected_reverse = frames[0], frames[-1]
    if method == 'a':
        expected_forward = np.maximum(expected_forward.astype(np.int16) - 1, 0).astype(np.uint8)
        expected_reverse = np.maximum(expected_reverse.astype(np.int16) - 1, 0).astype(np.uint8)
    np.testing.assert_array_equal(forward, expected_forward)
    np.testing.assert_array_equal(reverse, expected_reverse)


def _run_isolated_native(code, *args):
    environment = dict(os.environ)
    # A non-serial native default makes leaked limits observable on cold load.
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        environment[variable] = '3'
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(code), *args],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b'])
def test_real_explicit_tiles_finish_with_serial_and_parallel_workers(method):
    # The subprocess timeout converts a lease deadlock into a bounded failure.
    _run_isolated_native('''
        import sys
        import threading
        import numpy as np
        from core import resource_policy as policy
        from core.multi_focus_fusion import MultiFocusFusion

        snapshot = policy.ResourceSnapshot(available_ram=4 * 1024**3,
            total_ram=8 * 1024**3, physical_cores=2, logical_cpus=2,
            cpu_limit=2, physical_cores_known=True)
        policy.probe_resources = lambda *a, **k: snapshot
        method = sys.argv[1]
        frame = np.full((64, 80, 3), 17, np.uint8)
        frames = [frame, np.full_like(frame, 201)]
        def render(threads):
            fusion = MultiFocusFusion(method, use_gpu=False, exposure_match=False,
                resource_preference='maximum',
                tile_enabled=True, tile_threshold=32, tile_block_size=48,
                tile_overlap=16)
            return fusion.fuse(frames, thread_count=threads)
        serial = render(1)
        from fusion_methods import _openfocus_kernels as kernels
        name = 'select_rows' if method == 'openfocus_a' else 'merge_rows'
        original = getattr(kernels, name)
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        entered = 0
        identities = set()
        def native(*args):
            global entered
            with lock:
                slot = entered
                entered += 1
                identities.add(threading.get_ident())
            if slot < 2:
                barrier.wait(timeout=10)
            return original(*args)
        setattr(kernels, name, native)
        parallel = render(2)
        np.testing.assert_array_equal(parallel, serial)
        assert parallel.shape == frame.shape and parallel.dtype == np.uint8
        assert parallel.mean() > 0 and len(identities) == 2
    ''', method)


@pytest.mark.parametrize('cancel', ['normal', 'cancel'])
def test_direct_cold_import_restores_existing_and_new_native_pools(cancel):
    _run_isolated_native('''
        import sys
        import cv2
        import numpy as np
        from threadpoolctl import threadpool_info
        from core import resource_policy as policy
        from fusion_methods.openfocus_a import openfocus_a_impl

        assert 'core.multi_focus_fusion' not in sys.modules
        snapshot = policy.ResourceSnapshot(available_ram=4 * 1024**3,
            total_ram=8 * 1024**3, physical_cores=2, logical_cpus=2,
            cpu_limit=2, physical_cores_known=True)
        policy.probe_resources = lambda *a, **k: snapshot
        cv2.setNumThreads(3)
        before = {pool['filepath']: pool['num_threads'] for pool in threadpool_info()}
        previous_cv = cv2.getNumThreads()
        frame = np.full((513, 515, 3), 17, np.uint8)
        stopped = [False]
        def progress(done, total):
            if done and sys.argv[1] == 'cancel':
                stopped[0] = True
        caught = False
        try:
            output = openfocus_a_impl([frame, frame.copy()], thread_count=2,
                should_cancel=lambda: stopped[0], progress_callback=progress)
        except Exception as error:
            from core.registration import RegistrationCancelled
            assert isinstance(error, RegistrationCancelled), repr(error)
            caught = True
        assert caught == (sys.argv[1] == 'cancel')
        assert 'scipy.linalg' in sys.modules
        after = threadpool_info()
        for pool in after:
            if pool['filepath'] in before:
                assert pool['num_threads'] == before[pool['filepath']], pool
            elif pool['user_api'] == 'blas':
                assert pool['num_threads'] == 3, pool
        assert cv2.getNumThreads() == previous_cv
    ''', cancel)


def test_late_initializer_joins_before_native_limits_are_restored(resources, monkeypatch):
    from contextlib import contextmanager
    late_entered = threading.Event()
    close_started = threading.Event()
    finished = threading.Event()
    restore_started = threading.Event()
    scope_finished = threading.Event()
    lock = threading.Lock()
    arrivals = 0
    original_initialize = execution._worker_native_limit
    original_close = execution._Execution.close
    original_scope = execution.native_execution_scope

    def initialize():
        nonlocal arrivals
        with lock:
            slot = arrivals
            arrivals += 1
        if slot == 1:
            late_entered.set()
            assert close_started.wait(timeout=10)
            assert not restore_started.is_set()
        original_initialize()
        if slot == 1:
            finished.set()

    def close(job):
        close_started.set()
        original_close(job)

    @contextmanager
    def observe_scope():
        with original_scope():
            try:
                yield
            finally:
                assert finished.is_set(), 'native limits restored before initializer joined'
                restore_started.set()
        scope_finished.set()

    monkeypatch.setattr(execution, '_worker_native_limit', initialize)
    monkeypatch.setattr(execution._Execution, 'close', close)
    monkeypatch.setattr(execution, 'native_execution_scope', observe_scope)
    frame = np.zeros((512, 512), np.uint8)
    output = np.zeros(512, np.uint8)

    @execution.execution_job
    def compute():
        execution.configure_rows([frame, frame], None, 'openfocus_b', 2, None)
        def write(start, stop):
            assert late_entered.wait(timeout=10)
            output[start:stop] = 7
        execution.run_bands(512, 64, write)

    compute()
    assert finished.is_set() and scope_finished.is_set()
    np.testing.assert_array_equal(output, np.full(512, 7, np.uint8))
