"""CPU Q12 recurrence, advanced in cancellable blocks.

Import only from the processing path. Compiled kernels use Numba's cache so
isolated render processes can reuse them; an unavailable cache keeps the
same in-memory compilation path.
"""
from contextlib import contextmanager
import errno
import logging
import os
import pickle
import time
from threading import Lock

from core.algorithm_adapters import DEPTH_KERNEL_STARTUP_BYTES
from core.resource_policy import require_memory


require_memory(DEPTH_KERNEL_STARTUP_BYTES)
from numba import njit as _numba_njit  # noqa: E402 - admit compiler memory first
import numpy as np  # noqa: E402 - admit compiler memory first


@contextmanager
def _cache_access(directory):
    """Serialize cache transactions, falling back to in-memory compilation."""
    descriptor = None
    held = False
    locked = False
    try:
        for attempt in range(51):
            try:
                descriptor = os.open(os.path.join(directory, '.openfocus-numba.lock'),
                                     os.O_CREAT | os.O_RDWR, 0o600)
                break
            except PermissionError:
                if os.name != 'nt' or attempt == 50:
                    raise
                time.sleep(0.01)
        if os.name == 'nt':
            import msvcrt
            # An initialized lock can be held by another long-running process.
            # Retry only during creation so each compiler cache miss can proceed.
            initializing = os.fstat(descriptor).st_size == 0
            attempts = 51 if initializing else 1
            for attempt in range(attempts):
                try:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    held = True
                    break
                except OSError as exc:
                    if (attempt + 1 == attempts
                            or exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK)):
                        raise
                    time.sleep(0.01)
            if initializing and os.fstat(descriptor).st_size == 0:
                os.lseek(descriptor, 0, os.SEEK_SET)
                os.write(descriptor, b'\0')
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = True
        locked = True
    except OSError:
        pass
    try:
        yield locked
    finally:
        if descriptor is not None:
            try:
                if held and os.name == 'nt':
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                elif held:
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
    """Advance one column strip, wrapping state in uint32 across row slices."""
    height = src.shape[0]
    cols = right - left
    c2048 = np.uint32(2048)
    u1 = np.uint32(1)
    u3 = np.uint32(3)
    u4 = np.uint32(4)
    u12 = np.uint32(12)

    if guide is None:
        q32 = np.uint32(q)
        t16 = np.uint32(t) * np.uint32(16)
        if not backward:
            for y in range(start, stop):
                s_row = src[y, left:right]
                f_row = first[y]
                for col in range(cols):
                    inn = np.uint32(np.uint32(s_row[col]) * t16)
                    st = np.uint32(acc[col] + c2048 + inn) >> u12
                    f_row[col] = np.uint16((st + u4) >> u4)
                    acc[col] = np.uint32(st * q32 + inn)
        else:
            for y in range(stop - 1, start - 1, -1):
                s_row = src[y, left:right]
                f_row = first[y]
                o_row = output[y, left:right]
                for col in range(cols):
                    inn = np.uint32(np.uint32(s_row[col]) * t16)
                    st = np.uint32(acc[col] + c2048 + inn) >> u12
                    o_row[col] = np.uint16((((st + u3) >> u4) + u1 + np.uint32(f_row[col])) >> u1)
                    acc[col] = np.uint32(st * q32 + inn)
        return

    tm32 = np.uint32(tm)
    tl32 = np.uint32(tl)
    c4096 = np.uint32(4096)
    c65536 = np.uint32(65536)
    u2 = np.uint32(2)
    u16 = np.uint32(16)
    if not backward:
        if start < stop and start == 0:
            s_row = src[0, left:right]
            g_row = guide[0, left:right]
            f_row = first[0]
            for col in range(cols):
                v = np.uint32(s_row[col])
                s = np.uint32(g_row[col])
                amt = np.uint32((c65536 - s) * tl32 + s * tm32) >> u16
                st = np.uint32((c4096 - u2 * amt) * acc[col] + c2048 + (v + v) * (amt << u4)) >> u12
                acc[col] = st
                f_row[col] = (st + u4) >> u4
        y_begin = 1 if start == 0 else start
        for y in range(y_begin, stop):
            s_row = src[y, left:right]
            p_row = src[y - 1, left:right]
            g_row = guide[y, left:right]
            f_row = first[y]
            for col in range(cols):
                v = np.uint32(s_row[col])
                p = np.uint32(p_row[col])
                s = np.uint32(g_row[col])
                amt = np.uint32((c65536 - s) * tl32 + s * tm32) >> u16
                st = np.uint32((c4096 - u2 * amt) * acc[col] + c2048 + (v + p) * (amt << u4)) >> u12
                acc[col] = st
                f_row[col] = (st + u4) >> u4
    else:
        y_top = stop - 1
        if y_top >= start and y_top == height - 1:
            s_row = src[y_top, left:right]
            g_row = guide[y_top, left:right]
            f_row = first[y_top]
            o_row = output[y_top, left:right]
            for col in range(cols):
                v = np.uint32(s_row[col])
                p = f_row[col]
                s = np.uint32(g_row[col])
                amt = np.uint32((c65536 - s) * tl32 + s * tm32) >> u16
                st = np.uint32((c4096 - u2 * amt) * acc[col] + c2048 + (v + p) * (amt << u4)) >> u12
                acc[col] = st
                o_row[col] = np.uint16((((st + u3) >> u4) + u1 + np.uint32(f_row[col])) >> u1)
            y_top -= 1
        for y in range(y_top, start - 1, -1):
            s_row = src[y, left:right]
            p_row = src[y + 1, left:right]
            g_row = guide[y, left:right]
            f_row = first[y]
            o_row = output[y, left:right]
            for col in range(cols):
                v = np.uint32(s_row[col])
                p = np.uint32(p_row[col])
                s = np.uint32(g_row[col])
                amt = np.uint32((c65536 - s) * tl32 + s * tm32) >> u16
                st = np.uint32((c4096 - u2 * amt) * acc[col] + c2048 + (v + p) * (amt << u4)) >> u12
                acc[col] = st
                o_row[col] = np.uint16((((st + u3) >> u4) + u1 + np.uint32(f_row[col])) >> u1)


@njit(nogil=True)
def _pack_rows_T(orig, dst, left, right):
    """Cache-blocked 16x16 transpose from orig[left:right, :] to dst[:, :right-left]."""
    rows = right - left
    width = orig.shape[1]
    for x0 in range(0, width, 16):
        x1 = min(width, x0 + 16)
        for col in range(rows):
            y = left + col
            for x in range(x0, x1):
                dst[x, col] = orig[y, x]


@njit(nogil=True)
def _unpack_rows_T(src_buf, orig_out, left, right):
    """Cache-blocked 16x16 transpose from src_buf[:, :right-left] to orig_out[left:right, :]."""
    rows = right - left
    width = orig_out.shape[1]
    for y0 in range(0, rows, 16):
        y1 = min(rows, y0 + 16)
        for x0 in range(0, width, 16):
            x1 = min(width, x0 + 16)
            for col in range(y0, y1):
                y = left + col
                for x in range(x0, x1):
                    orig_out[y, x] = src_buf[x, col]


def prepare_depth_kernel(guided=True, should_cancel=None):
    """Compile needed layouts before frame pools and workspace admission."""
    from numba import types
    from fusion_methods.openfocus_ab import _check_cancel

    pixels = types.Array(types.uint16, 2, 'C')
    forward = types.Array(types.uint32, 2, 'C')
    accumulator = types.Array(types.uint32, 1, 'C')
    transpose_signature = (pixels, pixels, types.int64, types.int64)
    with _prepare_lock:
        for kernel in (_pack_rows_T, _unpack_rows_T):
            _check_cancel(should_cancel)
            if transpose_signature not in kernel.signatures:
                require_memory(DEPTH_KERNEL_STARTUP_BYTES)
                kernel.compile(transpose_signature)
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
