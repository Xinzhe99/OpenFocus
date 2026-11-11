"""Job-scoped disk reuse for decoded and processed frames.

A render may resolve the same lazy frame several times (registration, depth pass,
blend pass).  This cache stores the already-resolved pixels once, on disk, and
returns a read-only view for every later consumer.  It is strictly best-effort:

* pixel identity is preserved (the stored copy is the exact `full_resolution()`
  output, never re-encoded);
* writes and lookups never raise into the render; failures degrade to the
  existing repeated-decode behaviour;
* the cache is only created once the job's scratch directory is bound, and only
  when that volume can hold the whole distinct frame working set **and** the
  caller's mandatory staging estimate; otherwise it stays disabled;
* planning happens lazily at the first lookup, so installing the scope never
  changes where the job's own scratch selection would place files.

The active scope is process-scoped (not context-local) so producer threads see
the same store without every call site copying a context.  Renders are
serialized per process (GUI child process / CLI / one batch task).
"""
from __future__ import annotations

import contextlib
import threading
import time

import numpy as np

from utils.logging_setup import get_logger
from utils.performance_profile import current as _profile

logger = get_logger(__name__)

_active = []
_active_lock = threading.RLock()


def current():
    """Return the innermost active scope object, or None outside a render."""
    with _active_lock:
        return _active[-1] if _active else None


@contextlib.contextmanager
def active(scope):
    with _active_lock:
        _active.append(scope)
    try:
        yield scope
    finally:
        with _active_lock:
            if _active and _active[-1] is scope:
                _active.pop()
            else:
                try:
                    _active.remove(scope)
                except ValueError:
                    pass


def pixel_digest(image, compute):
    """Return ``compute()``, the digest of ``image``, hashing a cached copy once.

    Pixels the active scope stored and still retains are immutable for the
    render, so their digest is remembered. Any other array is hashed each time.
    """
    scope = current()
    remembered = getattr(scope, 'pixel_digest', None)
    return compute() if remembered is None else remembered(image, compute)


def _buffer_identity(image):
    """The memory and layout an array reads, or None for a non-array or empty one."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None
    return (image.__array_interface__['data'][0], image.shape, image.strides,
            image.dtype.str)


def _source_paths(frames):
    paths = []
    seen = set()
    stack = list(frames)
    while stack:
        frame = stack.pop()
        if frame is None or id(frame) in seen:
            continue
        seen.add(id(frame))
        source = getattr(frame, 'source', None)
        if source is not None and not isinstance(source, np.ndarray):
            stack.append(source)
            continue
        path = getattr(frame, 'path', None)
        if isinstance(path, str) and path:
            paths.append(path)
    return paths


def _frame_need(frames):
    """Sum the bytes of every distinct lazy frame and its lazy sources."""
    total = 0
    seen = set()
    stack = list(frames)
    while stack:
        frame = stack.pop()
        if frame is None or isinstance(frame, np.ndarray):
            continue
        identity = id(frame)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(getattr(frame, 'nbytes', 0) or 0)
        source = getattr(frame, 'source', None)
        if source is not None:
            stack.append(source)
    return total


def mandatory_staging_bytes(frames, *, alignment_steps=0, exposure_match=False, deferred=False):
    """Conservative disk reserve the pipeline itself may need while caching.

    Uses the pipeline's own estimators plus four per-frame resolution envelopes
    (the bound ``select_frame_scratch`` already asks for). Deferred warps do not
    allocate registration staging up front, so their estimator is skipped; the
    per-frame envelope still covers later depth/merge mappings.
    """
    from core.frame_storage import estimate_fusion_space, estimate_registration_space
    from core.source_frames import frame_resolution_bytes

    frames = list(frames)
    if not frames:
        return 0
    peak = 0
    for frame in frames:
        peak = max(peak, int(frame_resolution_bytes(frame)))
    fusion = int(estimate_fusion_space(frames, exposure_match))
    registration = (0 if deferred
                    else int(estimate_registration_space(frames, int(alignment_steps or 0))))
    # Phases can overlap (aligned frames stay live while fusion stages), so the
    # reserve sums the phase envelopes instead of taking a maximum.
    return registration + fusion + max(4 * peak, 64 * 1024 ** 2)


def _largest_frame_bytes(frames):
    biggest = 0
    stack = list(frames)
    seen = set()
    while stack:
        frame = stack.pop()
        if frame is None or isinstance(frame, np.ndarray):
            continue
        identity = id(frame)
        if identity in seen:
            continue
        seen.add(identity)
        biggest = max(biggest, int(getattr(frame, 'nbytes', 0) or 0))
        source = getattr(frame, 'source', None)
        if source is not None:
            stack.append(source)
    return biggest


def _validate_source(frame):
    """Re-check the source fingerprint before serving a cached copy."""
    from core.source_frames import validate_frame_source
    validate_frame_source(frame)


def planned_cache_bytes(frames):
    """What a render's cache asks for when planned: ``(bytes, source paths)``.

    The working set of every lazy frame plus one frame of staging floor, as
    ``ScopedCache`` admits it. Zero for in-memory arrays, which are never
    cached. For estimates made before a job binds its scratch directory.
    """
    frames = list(frames)
    need = _frame_need(frames)
    return (need + _staging_floor(frames) if need else 0), _source_paths(frames)


def _staging_floor(frames):
    return max(64 * 1024 * 1024, _largest_frame_bytes(frames))


def entry_kind(frame):
    """Classify a cache key for performance counters (never for admission)."""
    from functools import partial
    from core.source_frames import ProcessedFrame, SourceFrame, _crop_pixels
    if isinstance(frame, SourceFrame):
        return 'source'
    if isinstance(frame, ProcessedFrame):
        processor = frame.processor
        if isinstance(processor, partial) and processor.func is _crop_pixels:
            return 'crop'
        return 'processed'
    return 'other'


class ResolutionCache:
    """Disk-backed resolution results for one job; never raises into callers."""

    def __init__(self, budget_bytes, *, max_bytes=None, mandatory_bytes=0,
                 floor_bytes=0, directory=None, should_cancel=None):
        self.budget = max(0, int(budget_bytes))
        self.max_bytes = max(self.budget, int(max_bytes if max_bytes is not None
                                             else budget_bytes))
        self.mandatory = max(0, int(mandatory_bytes or 0))
        self.floor = max(0, int(floor_bytes or 0))
        self.directory = directory
        self.should_cancel = should_cancel
        self._entries = {}
        # Buffer identity of every stored entry -> its digest once computed.
        self._digests = {}
        self._used = 0
        self._degraded = False
        self._closed = False
        self._lock = threading.RLock()

    @property
    def closed(self):
        return self._closed

    @property
    def degraded(self):
        return self._degraded

    def pixel_digest(self, image, compute):
        """Digest of ``image``; computed once for a whole stored entry.

        Only an array reading exactly one retained entry's memory, in that
        entry's layout, is recognised: callers receive read-only views of it,
        and the mapping stays owned until the scope closes. A copy, a slice,
        a reordered view or a fresh decode is hashed by ``compute`` as before.
        """
        identity = _buffer_identity(image)
        with self._lock:
            stored = identity in self._digests
            digest = self._digests[identity] if stored else None
        if digest is not None:
            return digest
        digest = compute()
        if stored:
            with self._lock:
                if identity in self._digests:      # not closed in the meantime
                    self._digests[identity] = digest
        return digest

    def _check_cancel(self, should_cancel=None):
        if ((self.should_cancel is not None and self.should_cancel()) or
                (should_cancel is not None and should_cancel())):
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    def get(self, frame, *, should_cancel=None):
        """Return a read-only view of the cached pixels, or None on a miss."""
        self._check_cancel(should_cancel)
        if self._closed or (self._degraded and not self._entries):
            return None
        with self._lock:
            self._check_cancel(should_cancel)
            image = self._entries.get(frame)
        profile = _profile()
        if profile.enabled:
            profile.cache_event(entry_kind(frame), 'miss' if image is None else 'hit')
        if image is None:
            return None
        _validate_source(frame)
        self._check_cancel(should_cancel)
        view = image.view()
        view.setflags(write=False)
        self._check_cancel(should_cancel)
        return view

    def _admit(self, size):
        if self._used + size > self.max_bytes:
            return False
        if self.directory:
            import os
            from core.frame_storage import require_disk_space, storage_directory
            if (os.path.normcase(os.path.abspath(storage_directory()))
                    != os.path.normcase(os.path.abspath(self.directory))):
                # A nested scope rebound the job to another volume; writing a
                # mapped file there would contradict the admitted budget.
                return False
            require_disk_space(size + self.mandatory + self.floor, self.directory)
        return True

    def put(self, frame, image, *, should_cancel=None):
        """Store one resolved frame; returns a read-only view or None on failure."""
        self._check_cancel(should_cancel)
        if self._closed or self._degraded or image is None:
            return None
        size = int(image.nbytes)
        profile = _profile()
        started = time.perf_counter_ns() if profile.enabled else 0
        with self._lock:
            self._check_cancel(should_cancel)
            if frame in self._entries:
                stored = self._entries[frame]
            else:
                try:
                    admitted = self._admit(size)
                except OSError as exc:
                    # Disk space can shrink between planning and the actual
                    # write. Admission failure is a cache failure, never a
                    # render failure.
                    self._degraded = True
                    profile.cache_state(degraded=True)
                    logger.info('frame resolution cache degraded: %s', exc)
                    return None
                if not admitted:
                    # Derived frames may appear after planning and grow the
                    # working set; the ceiling and the staging reserve decide
                    # whether another mapping is safe. Never exceed them.
                    self._degraded = True
                    profile.cache_state(degraded=True)
                    logger.info('frame resolution cache degraded: used=%d need=%d max=%d',
                                self._used, size, self.max_bytes)
                    return None
                try:
                    from core import frame_storage
                    stored = frame_storage.store_frame(image, prefer_ram=False)
                except OSError as exc:  # ScratchSpaceError and disk/permission failures
                    self._degraded = True
                    profile.cache_state(degraded=True)
                    logger.info('frame resolution cache degraded: %s', exc)
                    return None
                try:
                    self._check_cancel(should_cancel)
                except BaseException:
                    # The new mapping has not been published; dropping its
                    # owner lets frame_storage's finalizer reclaim it now.
                    del stored
                    raise
                self._entries[frame] = stored
                identity = _buffer_identity(stored)
                if identity is not None:
                    self._digests[identity] = None
                self._used += size
                if profile.enabled:
                    profile.cache_event(entry_kind(frame), 'put', size,
                                        time.perf_counter_ns() - started)
                    profile.cache_state(retained_bytes=self._used)
                if self._used > self.budget:
                    self.budget = self._used
                    logger.info('frame resolution cache grew: used=%d max=%d',
                                self._used, self.max_bytes)
        self._check_cancel(should_cancel)
        view = stored.view()
        view.setflags(write=False)
        self._check_cancel(should_cancel)
        return view

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            logger.debug('frame resolution cache closed entries=%d used=%d max=%d degraded=%s',
                         len(self._entries), self._used, self.max_bytes, self._degraded)
            self._entries.clear()
            self._digests.clear()
            self._used = 0


class ScopedCache:
    """Lazily planned cache owned by one render scope.

    ``get``/``put`` are no-ops until the job's scratch directory is bound; the
    first lookup plans the cache against that directory.  Planning failures
    leave the scope permanently disabled.
    """

    def __init__(self, frames, *, mandatory_bytes=0, should_cancel=None):
        self._frames = list(frames)
        self._mandatory = max(0, int(mandatory_bytes or 0))
        self._should_cancel = should_cancel
        self._cache = None
        self._attempted = False
        self._closed = False
        self._lock = threading.RLock()

    @property
    def enabled(self):
        return self._cache is not None

    @property
    def closed(self):
        return self._closed

    @property
    def degraded(self):
        cache = self._cache
        return bool(cache is not None and cache.degraded)

    def _plan(self):
        if self._attempted:
            return self._cache
        from core.frame_storage import capture_scratch, require_disk_space
        state = capture_scratch()
        if state is None or state.directory is None:
            # The job has not selected its scratch volume yet. Never guess a
            # volume: stay dormant and retry after the pipeline binds one, so
            # the cache cannot pre-empt or contradict the original selection.
            return None
        directory = state.directory
        with self._lock:
            if self._attempted:
                return self._cache
            self._attempted = True
            if self._closed:
                return None
            need = _frame_need(self._frames)
            if need <= 0:
                return None
            try:
                import shutil
                floor = _staging_floor(self._frames)
                # Admission reserves the whole initial set, the caller's staging
                # estimate and one frame of staging floor. Derived frames may
                # extend the cache later, but only inside the plan-time ceiling.
                require_disk_space(need + self._mandatory + floor, directory)
                usage = shutil.disk_usage(directory)
                reserve = max(1024 ** 3, min(usage.total // 20, 4 * 1024 ** 3))
                available = max(0, usage.free - reserve)
                max_budget = max(need, available - self._mandatory - floor)
                self._cache = ResolutionCache(
                    need, max_bytes=max_budget, mandatory_bytes=self._mandatory,
                    floor_bytes=floor, directory=directory,
                    should_cancel=self._should_cancel)
                logger.info('frame resolution cache enabled bytes=%d max=%d mandatory=%d dir=%s',
                            need, max_budget, self._mandatory, directory)
            except OSError as exc:
                logger.info('frame resolution cache disabled: need=%d mandatory=%d reason=%s',
                            need, self._mandatory, exc)
                self._cache = None
            return self._cache

    def _check_cancel(self, should_cancel=None):
        if ((self._should_cancel is not None and self._should_cancel()) or
                (should_cancel is not None and should_cancel())):
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    def get(self, frame, *, should_cancel=None):
        self._check_cancel(should_cancel)
        cache = self._plan()
        self._check_cancel(should_cancel)
        return None if cache is None else cache.get(frame, should_cancel=should_cancel)

    def put(self, frame, image, *, should_cancel=None):
        self._check_cancel(should_cancel)
        cache = self._plan()
        self._check_cancel(should_cancel)
        return None if cache is None else cache.put(frame, image, should_cancel=should_cancel)

    def pixel_digest(self, image, compute):
        # Never plans: hashing must not choose the job's scratch volume.
        cache = self._cache
        return compute() if cache is None else cache.pixel_digest(image, compute)

    def close(self):
        with self._lock:
            self._closed = True
            cache = self._cache
        if cache is not None:
            cache.close()


@contextlib.contextmanager
def render_scope(frames, *, mandatory_bytes=0, should_cancel=None):
    """Bind one lazy resolution scope around a render.

    The scope never selects a scratch directory or changes one the job already
    chose; planning is deferred to the first resolution, after the pipeline has
    bound its own scratch directory.
    """
    from core.frame_storage import scratch_directory

    scope = ScopedCache(frames, mandatory_bytes=mandatory_bytes,
                        should_cancel=should_cancel)
    try:
        frames = list(frames)
    except TypeError:
        frames = []
    scope._frames = frames
    with scratch_directory():
        with active(scope):
            try:
                yield scope
            finally:
                scope.close()
