"""One bounded execution layer for an HF9 job and its ordered row passes."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature
from threading import Event

from core.frame_storage import capture_scratch, run_with_scratch, scratch_directory

_current = ContextVar('hf9_execution', default=None)
# Below this size scheduling competes with the native row kernels themselves.
_ROW_PARALLEL_PIXELS = 512 * 512


def execution_job(function):
    """Share one executor through nested depth/merge calls, joining on all exits."""
    call_signature = signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _current.get() is not None:
            return function(*args, **kwargs)
        from core.resource_policy import heavy_job

        job = _Execution()
        job.should_cancel = call_signature.bind_partial(*args, **kwargs).arguments.get('should_cancel')
        with heavy_job(job.should_cancel), scratch_directory(), native_execution_scope():
            token = _current.set(job)
            try:
                return function(*args, **kwargs)
            finally:
                # Join initializers and workers before restoring native pools.
                job.close()
                _current.reset(token)
    return wrapped


@contextmanager
def native_execution_scope():
    """Limit native pools on their owner thread, restoring after workers join."""
    import cv2
    previous = cv2.getNumThreads()
    # Planning imports MFF -> DCT -> SciPy. Load those native libraries before
    # threadpoolctl snapshots them, including direct backend calls on cold start.
    from core import multi_focus_fusion  # noqa: F401
    from threadpoolctl import threadpool_limits

    try:
        # Zero disables parallel regions on every backend, including macOS GCD.
        cv2.setNumThreads(0)
        with threadpool_limits(limits=1):
            yield
    finally:
        cv2.setNumThreads(0 if previous == 1 else previous)


@contextmanager
def serial_execution_scope(should_cancel):
    """Run one tile under the parent's lease with a private, serial child job."""
    job = _Execution()
    job.configured = True
    job.should_cancel = should_cancel
    token = _current.set(job)
    try:
        yield
    finally:
        job.close()
        _current.reset(token)


def _worker_native_limit():
    # OpenMP's thread limit belongs to the calling thread. BLAS limits are
    # process-wide and are changed/restored only by native_execution_scope.
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api='openmp')


def configure_rows(frames, img_resize, algorithm, thread_count, should_cancel):
    """Choose spatial work for large images; producers then remain serial."""
    job = _current.get()
    if job is None or job.configured:
        return
    job.configured = True
    job.should_cancel = should_cancel
    h, w = frames[0].shape[:2]
    if img_resize is not None:
        w, h = map(int, img_resize)
    job.height, job.width = h, w
    job.algorithm, job.thread_count = algorithm, thread_count
    if h * w < _ROW_PARALLEL_PIXELS or thread_count == 1:
        return
    from core import resource_policy as policy
    from core.source_frames import frame_resolution_bytes, source_cache_growth_bytes
    from constants import FRAME_RAM_BUDGET_BYTES
    # Accumulators, one producer, cache growth and decoder are job-wide, never
    # multiplied by spatial workers. Mapped planes remain the serial fallback.
    channels = frames[0].shape[2] if len(frames[0].shape) == 3 else 1
    plane_bytes = min(h * w * (48 if algorithm == 'openfocus_c' else 24),
                      FRAME_RAM_BUDGET_BYTES)
    fixed = (4 * 1024 * 1024 + plane_bytes
             + source_cache_growth_bytes(frames)
             + max((frame_resolution_bytes(frame) for frame in frames), default=0))
    if img_resize is not None:
        fixed += max((frame.shape[0] * frame.shape[1] * channels * 8
                      for frame in frames), default=0) + h * w * channels * 8
    per_worker = max(max(h, w) * 64 * 12,
                     min(h, 128) * w * (64 if algorithm == 'openfocus_a'
                                        else max(16, channels * 8)))
    try:
        plan = job.plan(max(1, (max(h, w) + 63) // 64), per_worker, fixed)
    except policy.ResourceLimitError:
        return
    job.workers = plan.workers
    job.row_mode = job.workers > 1


def uses_rows():
    job = _current.get()
    return job is not None and job.row_mode


def frame_parallelism_allowed():
    job = _current.get()
    return job is None or job.height * job.width >= 256 * 256


def cancellation_check(cancel):
    """Capture the owning job's stop flag for native loops in worker threads."""
    job = _current.get()
    if job is None:
        return cancel
    def check():
        job.check()
        cancel()
    return check


def run_bands(length, band, function, *, scratch_bytes=1, workspace=None):
    """Barrier after disjoint bands; optionally allocate scratch once per worker.

    The callback takes (start, stop), or (start, stop, workspace) when a factory
    is supplied. Every submitted worker finishes before this call returns or
    raises, so callers may immediately release source planes after the barrier.
    """
    job = _current.get()
    if job is None:
        local = workspace() if workspace is not None else None
        for start in range(0, length, band):
            if workspace is None:
                function(start, min(length, start + band))
            else:
                function(start, min(length, start + band), local)
        return
    job.run(length, band, function, scratch_bytes, workspace)


class _Execution:
    def __init__(self):
        self.configured = self.row_mode = False
        self.workers = 1
        self.pool = None
        self.stop = Event()
        self.should_cancel = None
        self.height = self.width = 0
        self.algorithm = ''
        self.thread_count = 1

    def check(self):
        from fusion_methods.openfocus_ab import _check_cancel
        _check_cancel(lambda: self.stop.is_set() or (
            self.should_cancel is not None and self.should_cancel()))

    def plan(self, tasks, scratch_bytes, fixed=0):
        from core import resource_policy as policy
        from core.multi_focus_fusion import memory_budget_bytes
        return policy.plan_execution(policy.Workload(
            algorithm=self.algorithm, height=self.height, width=self.width,
            fixed_bytes=fixed, per_worker_bytes=max(1, int(scratch_bytes)),
            outer_tasks=tasks),
            preference='maximum' if self.thread_count is not None else 'balanced',
            user_threads=self.thread_count,
            ram_budget_override=memory_budget_bytes())

    def run(self, length, band, function, scratch_bytes, workspace):
        self.check()
        tasks = (length + band - 1) // band
        workers = 1
        if self.row_mode and tasks > 1 and length * self.width >= 256 * 256:
            # Current free RAM, including all already resident planes, governs
            # each pass. Admission never happens inside frame release locks.
            workers = min(self.workers, self.plan(tasks, scratch_bytes).workers)
        scratch = capture_scratch()

        def work(slot):
            self.check()
            local = workspace() if workspace is not None else None
            for start in range(slot * band, length, workers * band):
                self.check()
                if workspace is None:
                    function(start, min(length, start + band))
                else:
                    function(start, min(length, start + band), local)
            self.check()

        if workers <= 1:
            work(0)
            return
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=self.workers,
                                           initializer=_worker_native_limit)
        futures = []
        try:
            for slot in range(workers):
                futures.append(self.pool.submit(run_with_scratch, scratch, work, slot))
            pending = set(futures)
            while pending:
                self.check()
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
            self.check()
        except BaseException:
            self.stop.set()
            for future in futures:
                future.cancel()
            # Never unwind source/workspace owners while native work runs.
            wait(futures)
            raise

    def close(self):
        self.stop.set()
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
