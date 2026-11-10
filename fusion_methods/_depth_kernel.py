"""CPU Q12 recurrence, advanced in cancellable blocks.

Import only from the processing path. Compiled kernels use Numba's cache so
isolated render processes can reuse them; an unavailable cache keeps the
same in-memory compilation path.
"""
from contextlib import contextmanager
import logging
import os
import pickle
from threading import Lock

from core.algorithm_adapters import DEPTH_KERNEL_STARTUP_BYTES
from core.resource_policy import require_memory


require_memory(DEPTH_KERNEL_STARTUP_BYTES)
from numba import njit as _numba_njit  # noqa: E402 - admit compiler memory first


@contextmanager
def _cache_access(directory):
    """Serialize a Numba index/data transaction without waiting for other jobs."""
    descriptor = None
    locked = False
    try:
        descriptor = os.open(os.path.join(directory, '.openfocus-numba.lock'),
                             os.O_CREAT | os.O_RDWR, 0o600)
        if os.name == 'nt':
            import msvcrt
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b'\0')
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
    except OSError:
        pass
    try:
        yield locked
    finally:
        if descriptor is not None:
            try:
                if locked and os.name == 'nt':
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                elif locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


class _GuardedCache:
    """Make Numba's cache optional when storage or its index is unavailable."""
    def __init__(self, cache):
        self._cache = cache
        self.cache_path = cache.cache_path

    def _call(self, method, *args):
        with _cache_access(self.cache_path) as locked:
            if not locked:
                return None
            try:
                return getattr(self._cache, method)(*args)
            except (OSError, EOFError, pickle.UnpicklingError) as exc:
                logging.getLogger(__name__).debug('Numba cache unavailable: %s', exc)
                return None

    def load_overload(self, *args):
        return self._call('load_overload', *args)

    def save_overload(self, *args):
        return self._call('save_overload', *args)

    def flush(self):
        return self._call('flush')


def njit(*args, **kwargs):
    """Cache kernels when possible, retaining in-memory compilation otherwise."""
    def decorate(function):
        try:
            dispatcher = _numba_njit(*args, cache=True, **kwargs)(function)
        except RuntimeError as exc:
            if 'cannot cache function' not in str(exc):
                raise
            return _numba_njit(*args, **kwargs)(function)
        # Numba's public dispatcher does not provide a cache error policy;
        # wrap its cache so index and data stay paired across render processes.
        cache = getattr(dispatcher, '_cache', None)
        if (not isinstance(getattr(cache, 'cache_path', None), (str, os.PathLike))
                or any(not callable(getattr(cache, name, None))
                       for name in ('load_overload', 'save_overload', 'flush'))):
            return _numba_njit(*args, **kwargs)(function)
        dispatcher._cache = _GuardedCache(cache)
        return dispatcher
    return decorate


_prepare_lock = Lock()


@njit(nogil=True)
def axis_block(src, guide, first, output, acc, q, t, tm, tl,
               start, stop, left, right, backward):
    """Advance one column strip, wrapping state before every right shift."""
    height = src.shape[0]
    mask = 0xffffffff
    for step in range(stop - start):
        y = stop - 1 - step if backward else start + step
        for x in range(left, right):
            column = x - left
            value = int(src[y, x])
            if guide is None:
                innovation = (value * 16 * t) & mask
                state = ((int(acc[column]) + 2048 + innovation) & mask) >> 12
                if backward:
                    output[y, x] = (((state + 3) >> 4) + 1
                                    + int(first[y, column])) >> 1 & 65535
                else:
                    first[y, column] = ((state + 4) >> 4) & 65535
                acc[column] = (state * q + innovation) & mask
            else:
                strength = int(guide[y, x])
                amount = (((65536 - strength) * tl
                           + strength * tm) & mask) >> 16
                if backward:
                    previous = (int(first[height - 1, column])
                                if y == height - 1 else int(src[y + 1, x]))
                else:
                    previous = int(src[max(0, y - 1), x])
                state = (((4096 - 2 * amount) * int(acc[column]) + 2048
                          + (value + previous) * (amount << 4))
                         & mask) >> 12
                acc[column] = state
                if backward:
                    output[y, x] = (((state + 3) >> 4) + 1
                                    + int(first[y, column])) >> 1 & 65535
                else:
                    first[y, column] = (state + 4) >> 4


def prepare_depth_kernel(guided=True, should_cancel=None):
    """Compile needed layouts before frame pools and workspace admission."""
    from numba import types
    from fusion_methods.openfocus_ab import _check_cancel

    pixels = types.Array(types.uint16, 2, 'C')
    forward = types.Array(types.uint32, 2, 'C')
    accumulator = types.Array(types.uint32, 1, 'C')
    with _prepare_lock:
        for layout in ('C', 'F'):
            source = types.Array(types.uint16, 2, layout)
            guides = ((types.none, pixels, types.Array(types.uint16, 2, 'F'))
                      if guided else (types.none,))
            for guide in guides:
                _check_cancel(should_cancel)
                first = pixels if guide is types.none else forward
                signature = (source, guide, first, pixels, accumulator,
                             *(types.int64,) * 8, types.boolean)
                if signature not in axis_block.signatures:
                    require_memory(DEPTH_KERNEL_STARTUP_BYTES)
                    axis_block.compile(signature)
                _check_cancel(should_cancel)
