"""Independent snapshots and bounded RAM scratch whose lifetime follows views."""
import errno
import logging
import os
import math
import operator
import shutil
import tempfile
import weakref
import contextvars
import contextlib
import subprocess
import sys
import threading


import numpy as np

_storage_directory = None
_storage_notice_lock = threading.Lock()
_storage_notice_path = None
_storage_fallback_notifier = None
_scratch = contextvars.ContextVar('openfocus_scratch', default=None)

# Shared across jobs and copied producer contexts. Views keep their allocating
# ndarray alive, so its finalizer releases the reservation after the last view.
_ram_lock = threading.Lock()
_ram_live_bytes = 0


def _release_ram(size):
    global _ram_live_bytes
    with _ram_lock:
        _ram_live_bytes -= size


def _try_ram_frame(shape, dtype):
    """Reserve and allocate RAM, returning None when snapshots must spill."""
    from core import resource_policy as policy
    global _ram_live_bytes
    dtype = np.dtype(dtype)
    size = math.prod(shape) * dtype.itemsize
    if size < 0 or dtype.hasobject:
        raise ValueError('Transient frames require nonnegative numeric storage')
    # Probing and allocating can collect arrays whose finalizers take this lock.
    snapshot = policy.probe_memory()
    cap = policy.frame_ram_budget(snapshot)
    with _ram_lock:
        if size > cap - _ram_live_bytes:
            return None
        _ram_live_bytes += size
    try:
        policy.require_memory(size, snapshot)
        image = np.empty(shape, dtype=dtype)
        weakref.finalize(image, _release_ram, size)
    except BaseException:
        _release_ram(size)
        raise
    return image


def create_transient_frame(shape, dtype):
    """Allocate independent scratch pixels within a shared RAM cap.

    Live snapshots and scratch share a fixed RAM ceiling and at most half the
    comfortable resource budget. Unknown or insufficient RAM uses an admitted
    disk mapping. NumPy views retain their allocation's reservation.
    """
    image = _try_ram_frame(shape, dtype)
    return create_frame(shape, dtype) if image is None else image


def _local_mounts():
    """Discover fixed local volumes conservatively; never infer network media."""
    if sys.platform == 'win32':
        import ctypes
        kernel = ctypes.windll.kernel32
        mask = kernel.GetLogicalDrives()
        return [f'{chr(65 + i)}:\\' for i in range(26)
                if mask & (1 << i) and kernel.GetDriveTypeW(f'{chr(65 + i)}:\\') == 3]
    if sys.platform.startswith('linux'):
        mounts = []
        try:
            with open('/proc/self/mountinfo', encoding='utf-8') as stream:
                for line in stream:
                    before, after = line.split(' - ', 1)
                    fields, fs = before.split(), after.split()
                    if fs[0] in ('tmpfs', 'overlay', 'ramfs') or not fs[1].startswith('/dev/'):
                        continue
                    device = os.path.realpath('/sys/dev/block/' + fields[2])
                    if not os.path.exists(device):
                        continue
                    # Partitions inherit the removable flag from their parent.
                    block = device
                    while block.startswith('/sys/') and not os.path.exists(os.path.join(block, 'removable')):
                        parent = os.path.dirname(block)
                        if parent == block:
                            break
                        block = parent
                    try:
                        with open(os.path.join(block, 'removable'), encoding='ascii') as flag:
                            if flag.read().strip() != '0':
                                continue
                    except OSError:
                        continue
                    mount = fields[4]
                    for encoded, char in (('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')):
                        mount = mount.replace(encoded, char)
                    mounts.append(mount)
        except (OSError, ValueError, IndexError):
            pass
        return mounts
    if sys.platform == 'darwin':
        import plistlib
        mounts = []
        try:
            listing = plistlib.loads(subprocess.check_output(['/usr/sbin/diskutil', 'list', '-plist'], timeout=10))
            for disk in listing.get('AllDisks', []):
                try:
                    info = plistlib.loads(subprocess.check_output(
                        ['/usr/sbin/diskutil', 'info', '-plist', disk], timeout=3))
                except (OSError, subprocess.SubprocessError, ValueError):
                    continue
                if info.get('Internal') is True and not info.get('RemovableMedia', True) and info.get('MountPoint'):
                    mounts.append(info['MountPoint'])
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return mounts
    return []


def _default_is_local(path):
    """Reject system temp overrides on network or removable volumes."""
    if sys.platform == 'win32':
        import ctypes
        kernel = ctypes.windll.kernel32
        volume = ctypes.create_unicode_buffer(32768)
        return bool(kernel.GetVolumePathNameW(os.path.abspath(path), volume, len(volume))
                    and kernel.GetDriveTypeW(volume.value) == 3)
    if sys.platform.startswith('linux'):
        path = os.path.realpath(path)
        best = None
        try:
            with open('/proc/self/mountinfo', encoding='utf-8') as stream:
                for line in stream:
                    before, after = line.split(' - ', 1)
                    mount = before.split()[4]
                    for encoded, char in (('\\040', ' '), ('\\011', '\t'), ('\\012', '\n'), ('\\134', '\\')):
                        mount = mount.replace(encoded, char)
                    if os.path.commonpath((path, mount)) == mount and (best is None or len(mount) > len(best[0])):
                        best = (mount, after.split()[0])
            if best is None:
                return False
            if best[1] in ('tmpfs', 'ramfs', 'overlay'):
                return True
            return any(os.path.realpath(mount) == os.path.realpath(best[0]) for mount in _local_mounts())
        except (OSError, ValueError, IndexError):
            return False
    if sys.platform == 'darwin':
        import plistlib
        try:
            # diskutil accepts a device or mount point, not an arbitrary
            # temporary subdirectory (including APFS Data firmlink paths).
            usage = subprocess.check_output(['/bin/df', '-P', path], timeout=3, text=True)
            device = usage.splitlines()[-1].split()[0]
            info = plistlib.loads(subprocess.check_output(
                ['/usr/sbin/diskutil', 'info', '-plist', device], timeout=10))
            return info.get('Internal') is True and info.get('RemovableMedia') is False
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return False
    return False


def _space(directory):
    usage = shutil.disk_usage(directory)
    reserve = max(1024 ** 3, min(usage.total // 20, 4 * 1024 ** 3))
    return usage, reserve, max(0, usage.free - reserve)


class ScratchSpaceError(OSError):
    """A localized, actionable scratch-space admission failure."""

    def __init__(self, directory, required, available, detail=''):
        self.directory, self.required, self.available = directory, required, available
        self.detail = detail
        super().__init__(directory)

    def __str__(self):
        try:
            from locales import trans
            template = trans.t('scratch_space_error')
        except ImportError:
            template = ('Temporary storage cannot be used or lacks space: {path}\n'
                        'Estimated additional space needed: {required}; available after reserve: {available}. '
                        'Later processing may need more. Free space or choose a writable Temporary directory '
                        'in Advanced settings.')
        available = '—' if self.available is None else readable_bytes(self.available)
        return template.format(path=self.directory, required=readable_bytes(self.required),
                               available=available)


def readable_bytes(value):
    """Binary-unit size as shown in scratch-space messages."""
    if value < 1024:
        return f'{value} B'
    for unit, factor in (('GiB', 1024 ** 3), ('MiB', 1024 ** 2), ('KiB', 1024)):
        if value >= factor:
            return f'{value / factor:.2f} {unit}'


def estimate_registration_space(images, steps):
    if len(images) < 2 or not steps:
        return 0
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    pixel_bytes = max(image.dtype.itemsize * (image.shape[2] if image.ndim == 3 else 1)
                      for image in images)
    return height * width * pixel_bytes * len(images) * min(2, steps)


def estimate_fusion_space(images, exposure_match=False):
    """Upper estimate for optional staging; tiling or unity gains may use less."""
    if any(image.dtype == np.uint16 for image in images):
        return sum(image.size * 4 for image in images if image.dtype != np.float32)
    return sum(image.nbytes for image in images) if exposure_match else 0


def set_storage_fallback_notifier(notifier):
    """Register a thread-safe callback for a configured directory fallback."""
    global _storage_fallback_notifier, _storage_notice_path
    with _storage_notice_lock:
        _storage_fallback_notifier = notifier
        _storage_notice_path = None


def _usable_storage_directory():
    """Use the saved directory if writable, otherwise the system temp folder."""
    global _storage_notice_path
    directory = _storage_directory
    if directory is None:
        return None
    try:
        fd, probe = tempfile.mkstemp(prefix='openfocus-probe-', dir=directory)
        try:
            os.close(fd)
        finally:
            os.unlink(probe)
    except OSError as exc:
        # A full selected disk stays binding: silently switching to another
        # volume would defeat the user's explicit space and privacy choice.
        if exc.errno in (errno.ENOSPC, getattr(errno, 'EDQUOT', -1)):
            return directory
        fallback = tempfile.gettempdir()
        with _storage_notice_lock:
            first_notice = _storage_notice_path != directory
            notify = _storage_fallback_notifier if first_notice else None
            _storage_notice_path = directory
        if first_notice:
            logging.getLogger(__name__).warning(
                'Temporary directory %s is unavailable; using %s: %s',
                directory, fallback, exc)
        if notify is not None:
            try:
                notify(directory, str(exc), fallback)
            except RuntimeError:
                logging.getLogger(__name__).exception('Could not display temporary-directory warning')
        return fallback
    with _storage_notice_lock:
        _storage_notice_path = None
    return directory


class _Scratch:
    def __init__(self):
        self.explicit_directory = _usable_storage_directory()
        self.directory = None
        self.owned = False
        self.closed = False
        self.live = 0
        self.lock = threading.Lock()

    def retain(self):
        with self.lock:
            self.live += 1

    def release(self):
        with self.lock:
            self.live -= 1
            self._cleanup()

    def close(self):
        with self.lock:
            self.closed = True
            self._cleanup()

    def _cleanup(self):
        if self.closed and not self.live and self.owned:
            try:
                os.rmdir(self.directory)
            except OSError:
                pass


@contextlib.contextmanager
def scratch_directory(*, separate=False):
    """Keep one scratch decision local to this job and its copied contexts."""
    existing = _scratch.get()
    if existing is not None and not separate:
        yield existing
        return
    state = _Scratch()
    token = _scratch.set(state)
    try:
        yield state
    finally:
        _scratch.reset(token)
        state.close()


def capture_scratch():
    return _scratch.get()


def run_with_scratch(state, function, *args, **kwargs):
    token = _scratch.set(state)
    try:
        return function(*args, **kwargs)
    finally:
        _scratch.reset(token)


def _scratch_candidates(explicit, source_paths):
    """Directories select_scratch considers, in preference order."""
    default = explicit or tempfile.gettempdir()
    if explicit is not None or _default_is_local(default):
        yield default
    if explicit is not None:
        return
    mounts = _local_mounts()
    # Prefer a source's fixed local volume, not an arbitrary source folder.
    for source in source_paths:
        source = os.path.normcase(os.path.abspath(source))
        for mount in sorted(mounts, key=len, reverse=True):
            root = os.path.normcase(os.path.abspath(mount))
            try:
                if (os.path.commonpath((source, root)) == root
                        and os.stat(source).st_dev == os.stat(mount).st_dev):
                    yield os.path.dirname(source)
                    yield mount
                    break
            except (OSError, ValueError):
                pass
    yield from mounts


def assess_scratch(minimum, expected, optional=0, source_paths=(), *, directory=None):
    """Classify the prospective scratch directory before a job starts.

    A point-in-time estimate: it binds nothing, creates nothing and reserves
    nothing; every runtime check still applies. ``minimum`` is what the job
    cannot run without, ``expected`` adds the pipeline's own staging reserve,
    and ``optional`` is cache space that degrades gracefully when missing.
    The directory is ``directory`` when given, else the configured one, else
    the one select_scratch would choose for ``expected + optional``.

    Status: ``critical`` when ``minimum`` exceeds the space left after the
    safety reserve (the job would be rejected), ``advisory`` when ``expected
    + optional`` does not fit (caches will be disabled or staging may run out),
    ``ok`` otherwise, and ``unknown`` when the volume cannot be queried.
    """
    minimum, expected, optional = (max(0, int(value)) for value in (minimum, expected, optional))
    expected = max(expected, minimum)
    explicit = directory if directory is not None else _usable_storage_directory()
    chosen = None
    if explicit is None:
        best = None
        for candidate in _scratch_candidates(None, source_paths):
            try:
                available = _space(candidate)[2]
            except OSError:
                continue
            if best is None or available > best[1]:
                best = (candidate, available)
            if available >= expected + optional:
                chosen = candidate
                break
        chosen = chosen or (best[0] if best else tempfile.gettempdir())
    else:
        chosen = explicit
    result = {'directory': os.path.abspath(chosen), 'explicit': explicit is not None,
              'minimum_bytes': minimum, 'expected_bytes': expected, 'optional_bytes': optional,
              'required_bytes': expected + optional}
    try:
        usage, reserve, available = _space(chosen)
    except OSError as exc:
        return {**result, 'status': 'unknown', 'reason': str(exc)}
    result.update(free_bytes=int(usage.free), reserve_bytes=int(reserve),
                  available_bytes=int(available))
    if minimum > available:
        result['status'] = 'critical'
    elif expected + optional > available:
        result['status'] = 'advisory'
    else:
        result['status'] = 'ok'
    return result


def validate_storage_directory(path):
    """Raise OSError unless ``path`` is an existing writable directory."""
    directory = os.path.abspath(os.fsdecode(path))
    fd, probe = tempfile.mkstemp(prefix='openfocus-probe-', dir=directory)
    try:
        os.close(fd)
    finally:
        os.unlink(probe)
    return directory


def select_scratch(size, source_paths=(), *, preferred_size=None, working_size=None):
    """Admit an estimate and bind a writable directory to the current job.

    The estimate is a snapshot, not a reservation. Allocation checks continue
    for every file and later stages may require additional storage.
    """
    state = _scratch.get()
    if state is None:
        raise RuntimeError('Scratch selection requires a job scope')
    if state.directory is not None:
        require_disk_space(size, state.directory)
        return state.directory
    default = state.explicit_directory or tempfile.gettempdir()

    def candidates():
        return _scratch_candidates(state.explicit_directory, source_paths)
    best = (default, None)
    # Optional staging and bounded-workspace estimates guide placement without
    # rejecting RAM-only work or jobs that fit the known required allocation.
    goals = [size]
    if state.explicit_directory is None:
        goals = sorted({size, *(max(size, hint) for hint in
                        (preferred_size, working_size) if hint is not None)}, reverse=True)
    for goal in goals:
        seen = set()
        for candidate in candidates():
            identity = os.path.normcase(os.path.abspath(candidate))
            if identity in seen:
                continue
            seen.add(identity)
            try:
                _, _, available = _space(candidate)
                if best[1] is None or available > best[1]:
                    best = (candidate, available)
                if goal > available:
                    continue
                if candidate == default:
                    fd, probe = tempfile.mkstemp(prefix='openfocus-probe-', dir=candidate)
                    os.close(fd)
                    os.unlink(probe)
                    state.directory = candidate
                else:
                    state.directory = tempfile.mkdtemp(prefix='openfocus-', dir=candidate)
                    state.owned = True
                return state.directory
            except OSError:
                continue
    raise ScratchSpaceError(best[0], size, best[1])


def storage_directory():
    """Return the directory currently used for new temporary data."""
    state = _scratch.get()
    if state is not None:
        return state.directory or state.explicit_directory or tempfile.gettempdir()
    return _usable_storage_directory() or tempfile.gettempdir()


def configure_storage_directory(path):
    """Select a writable temporary directory, or reset to the system default."""
    global _storage_directory
    if path is None or os.fspath(path) in ('', b''):
        _storage_directory = None
        return
    _storage_directory = validate_storage_directory(path)


def restore_storage_directory(path):
    """Restore a saved choice without replacing it if it is unavailable."""
    global _storage_directory
    _storage_directory = (None if path is None or os.fspath(path) in ('', b'')
                          else os.path.abspath(os.fsdecode(path)))


# Temporary frame files may be reopened by independent mappings in this or
# another process, so deletion follows a path-keyed owner count instead of each
# finalizer unlinking independently. A disowned path keeps its pixels after the
# last mapping closes while still closing maps and releasing scratch references.
_owner_lock = threading.RLock()
_frame_owners = {}
_observed_files = set()
_disowned_files = set()
_cleanup_directories = {}
_temp_observers = []



def _file_key(path):
    """Stable comparison key for a temporary path, including cross-thread."""
    return os.path.normcase(os.path.abspath(os.fsdecode(os.fspath(path))))




def _remove(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _remove_dir(path):
    try:
        os.rmdir(path)
    except OSError:
        pass


def _observe_file(path):
    """Report a fresh file before any caller fills it."""
    key = _file_key(path)
    with _owner_lock:
        _observed_files.add(key)
        callbacks = tuple(_temp_observers)
    try:
        for callback in callbacks:
            callback(path)
    except BaseException:
        with _owner_lock:
            _observed_files.discard(key)
        raise


def _own_file(path, cleanup_directory=None):
    """Record one owner and return the path key used by its finalizer."""
    key = _file_key(path)
    directory = None if cleanup_directory is None else _file_key(cleanup_directory)
    with _owner_lock:
        _observed_files.add(key)
        _frame_owners[key] = _frame_owners.get(key, 0) + 1
        if directory is not None:
            _cleanup_directories[key] = directory
    return key


def _release_file_owner(key):
    """Drop one owner; the last owner removes the file unless it was disowned."""
    with _owner_lock:
        count = _frame_owners.get(key)
        if count is not None:
            count -= 1
            if count > 0:
                _frame_owners[key] = count
                return
            _frame_owners.pop(key, None)
        elif key not in _observed_files:
            return
        _observed_files.discard(key)
        directory = _cleanup_directories.pop(key, None)
        suppress = key in _disowned_files
        if suppress:
            _disowned_files.discard(key)
    if suppress:
        return
    _remove(key)
    if directory is not None:
        # Only an empty directory is removed; never recursive deletion.
        _remove_dir(directory)


def _release_file(key, state):
    try:
        _release_file_owner(key)
    finally:
        if state is not None:
            state.release()


def track_temp_file(owner, path, *, cleanup_directory=None):
    """Keep a temporary file and its scratch directory until ``owner`` dies.

    The path shares one count with mappings of the same file. ``cleanup_directory``
    is the auto-created scratch directory that may be removed once its last owned
    file is gone; explicit/user directories are not removed.
    """
    state = _scratch.get()
    state = state if state is not None and state.directory == os.path.dirname(path) else None
    cleanup_directory = _check_cleanup_directory(cleanup_directory, path)
    if state is not None:
        state.retain()
    key = _own_file(path, cleanup_directory)
    weakref.finalize(owner, _release_file, key, state)


def owns_temp_file(path):
    """Whether this process created or explicitly adopted the temporary file.

    A filename alone never establishes ownership of borrowed user pixels.
    """
    try:
        key = _file_key(path)
    except (TypeError, ValueError):
        return False
    with _owner_lock:
        return key in _frame_owners or key in _observed_files


def disown_temp_files(paths):
    """Never delete these temporary files; closing maps still releases scratch."""
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = (paths,)
    keys = {_file_key(path) for path in paths}
    with _owner_lock:
        for key in keys:
            if key in _frame_owners or key in _observed_files:
                _disowned_files.add(key)


@contextlib.contextmanager
def observe_temp_files(callback):
    """Observe every temporary frame file created during this context.

    The callback runs in the allocating producer thread immediately after the
    file is created and before any pixel is written, so a supervising process
    can remove partial results if the producer crashes.
    """
    if not callable(callback):
        raise TypeError('Temporary file observer must be callable')
    with _owner_lock:
        _temp_observers.append(callback)
    try:
        yield callback
    finally:
        with _owner_lock:
            try:
                _temp_observers.remove(callback)
            except ValueError:
                pass


def ram_snapshot_bytes():
    """Bytes currently reserved by transient in-RAM frames in this process."""
    with _ram_lock:
        return _ram_live_bytes


def _release(mapping, key, state):
    try:
        mapping.close()
    finally:
        try:
            _release_file_owner(key)
        finally:
            if state is not None:
                state.release()

def require_disk_space(size, directory=None):
    """Check a point-in-time estimate, retaining space for the system.

    Other processes can consume space after this check; it is not a reservation.
    """
    directory = storage_directory() if directory is None else directory
    try:
        _, _, available = _space(directory)
    except OSError as exc:
        raise ScratchSpaceError(directory, size, None, str(exc)) from exc
    if size > available:
        raise ScratchSpaceError(directory, size, available)

def _storage_error(directory, size, exc):
    try:
        available = _space(directory)[2]
    except OSError:
        available = None
    return ScratchSpaceError(directory, size, available, str(exc))


def _frame_file(directory, size, *, prefix='openfocus-frame-'):
    try:
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=".bin", dir=directory)
    except OSError as exc:
        raise _storage_error(directory, size, exc) from exc
    try:
        _observe_file(path)
    except BaseException:
        os.close(fd)
        _remove(path)
        raise
    return fd, path



def create_frame(shape, dtype, *, directory=None):
    """Create an empty temporary mapping owned by its NumPy views."""
    directory = storage_directory() if directory is None else directory
    size = math.prod(shape) * np.dtype(dtype).itemsize
    require_disk_space(size, directory)
    fd, path = _frame_file(directory, size)
    key = _own_file(path)
    try:
        allocated = False
        if size and hasattr(os, 'posix_fallocate'):
            try:
                os.posix_fallocate(fd, 0, size)
                allocated = True
            except OSError as exc:
                if exc.errno not in (errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL):
                    raise
        os.close(fd)
        fd = None
        mapped = np.memmap(path, dtype=dtype, mode="r+" if allocated else "w+", shape=shape)
        state = _scratch.get()
        state = state if state is not None and state.directory == directory else None
        if state is not None:
            state.retain()
        weakref.finalize(mapped, _release, mapped._mmap, key, state)
        return mapped
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        _release_file_owner(key)
        raise _storage_error(directory, size, exc) from exc
    except BaseException:
        if fd is not None:
            os.close(fd)
        _release_file_owner(key)
        raise


def store_frame(image, *, prefer_ram=True):
    """Copy a frame independently, preserving dtype and shape.

    Snapshots use the shared RAM cap before spilling to an admitted mapping.
    prefer_ram=False explicitly requests disk storage. Both paths copy, so
    modifying a snapshot never mutates borrowed source pixels.
    """
    if prefer_ram:
        owned = _try_ram_frame(image.shape, image.dtype)
        if owned is not None:
            np.copyto(owned, image)
            return owned
    directory = storage_directory()
    require_disk_space(image.nbytes, directory)
    fd, path = _frame_file(directory, image.nbytes)
    key = _own_file(path)
    try:
        # Write through the file descriptor before mapping it. Filling a writable
        # mapping faults every source-stack page into this process even though
        # the caller only needs one frame/strip at a time afterward.
        with os.fdopen(fd, "wb") as stream:
            fd = None
            np.asarray(image).tofile(stream)
        mapped = np.memmap(path, dtype=image.dtype, mode="r+", shape=image.shape)
        state = _scratch.get()
        state = state if state is not None and state.directory == directory else None
        if state is not None:
            state.retain()
        weakref.finalize(mapped, _release, mapped._mmap, key, state)
        return mapped
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        _release_file_owner(key)
        raise _storage_error(directory, image.nbytes, exc) from exc
    except BaseException:
        if fd is not None:
            os.close(fd)
        _release_file_owner(key)
        raise


@contextlib.contextmanager
def frame_region(image, key=Ellipsis, *, write=False):
    """Borrow one region without retaining its mapped pages after the context.

    The original array and its file owner remain valid. Consumers must copy
    arrays retained after this context; disk views close with the lease.
    ``None`` represents an absent optional source mask.
    """
    if image is None:
        yield None
        return
    region = image[key]
    if not isinstance(region, np.ndarray) or not region.size:
        yield region
        return
    root = _mapped_base(region)
    if root is None:
        yield region
        return
    offset = root.offset + region.ctypes.data - root.ctypes.data
    extent = [(size - 1) * stride for size, stride in zip(region.shape, region.strides)]
    start = offset + sum(min(0, value) for value in extent)
    stop = offset + sum(max(0, value) for value in extent) + region.dtype.itemsize
    mapping = np.memmap(root.filename, dtype=np.uint8,
                        mode='r+' if write else 'r', offset=start, shape=(stop - start,))
    try:
        borrowed = np.ndarray(region.shape, dtype=region.dtype, buffer=mapping,
                              offset=offset - start, strides=region.strides)
        if not write:
            borrowed.flags.writeable = False
        yield borrowed
    finally:
        mapping._mmap.close()


def _mapped_base(image):
    """The deepest real mapping in an ndarray's base chain, or None.

    Stride tricks insert a small Python wrapper into the chain, so follow any
    object that exposes ``base`` rather than only ndarray links.
    """
    mapped = None
    current = image
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (isinstance(current, np.ndarray)
                and getattr(current, '_mmap', None) is not None
                and getattr(current, 'filename', None) is not None):
            mapped = current
        current = getattr(current, 'base', None)
    return mapped


def is_mapped_frame(image):
    """Whether an array borrows mapped pixels, including reopened strided views.

    Storage location does not imply ownership of the backing file.
    """
    return isinstance(image, np.ndarray) and _mapped_base(image) is not None


def describe_frame(image):
    """Describe a mapped ndarray for transfer without copying its pixels.

    The base chain is followed to the mapping that owns the file so a view's
    absolute byte offset and strides survive, including reversed and
    non-contiguous views. Heap-backed arrays describe as None.
    """
    if not isinstance(image, np.ndarray):
        return None
    if image.dtype.hasobject or image.dtype.itemsize <= 0:
        return None
    mapped = _mapped_base(image)
    if mapped is None:
        return None
    try:
        path = os.path.abspath(os.fsdecode(os.fspath(mapped.filename)))
    except (TypeError, ValueError):
        return None
    if not os.path.isfile(path):
        return None
    try:
        base_pointer = int(mapped.__array_interface__['data'][0])
        pointer = int(image.__array_interface__['data'][0])
        base_offset = int(getattr(mapped, 'offset', 0) or 0)
    except (KeyError, TypeError, ValueError):
        return None
    offset = base_offset + pointer - base_pointer
    if offset < 0:
        return None
    dtype = image.dtype.str if image.dtype.fields is None else image.dtype.descr
    return {'path': path,
            'shape': tuple(int(value) for value in image.shape),
            'dtype': dtype,
            'offset': offset,
            'strides': tuple(int(value) for value in image.strides)}


def _close_mapping(mapping):
    mapping.close()


def _release_mapping(mapping, key):
    try:
        mapping.close()
    finally:
        _release_file_owner(key)


def _cleanup_directory(description):
    directory = description.get('cleanup_directory')
    if directory is None:
        return None
    try:
        value = os.fspath(directory)
    except TypeError as exc:
        raise ValueError('Frame cleanup_directory is not a path') from exc
    return None if value in ('', b'') else os.fsdecode(value)


def _check_cleanup_directory(directory, path):
    """Only the file's own directory may be cleaned up later."""
    if directory is None:
        return None
    if _file_key(directory) != _file_key(os.path.dirname(path)):
        raise ValueError('Frame cleanup_directory must contain the frame file')
    return directory


def open_frame(description, *, owned=False):
    """Reopen a mapped frame read without copying pixels.

    The result is read-only and keeps its mapping alive until every view is
    gone. ``owned`` explicitly adopts deletion after an ownership transfer;
    borrowed user mappings must be opened with its default False value. A
    ``cleanup_directory`` in the description is removed (empty only) after the
    last owned file there is released.
    """
    path = os.fsdecode(os.fspath(description['path']))
    dtype = np.dtype(description['dtype'])
    if dtype.hasobject or dtype.itemsize <= 0:
        raise ValueError('Frame description requires nonobjective numeric storage')
    try:
        shape = tuple(operator.index(value) for value in description['shape'])
        strides = tuple(operator.index(value) for value in description['strides'])
        offset = operator.index(description['offset'])
    except (TypeError, ValueError) as exc:
        raise ValueError('Frame description has nonnumeric layout') from exc
    if len(shape) != len(strides):
        raise ValueError('Frame description shape and strides disagree')
    if any(value < 0 for value in shape) or offset < 0:
        raise ValueError('Frame description has a negative extent')
    cleanup = _check_cleanup_directory(
        _cleanup_directory(description), path)
    adopt = bool(owned)
    file_size = os.path.getsize(path)
    itemsize = dtype.itemsize
    empty = math.prod(shape) == 0
    if empty:
        if offset > file_size:
            raise ValueError('Frame description exceeds its file')
        image = np.empty(shape, dtype=dtype)
        image.flags.writeable = False
        if adopt:
            key = _own_file(path, cleanup)
            weakref.finalize(image, _release_file, key, None)
        return image
    first = offset
    end = offset + itemsize
    for length, stride in zip(shape, strides):
        extent = (length - 1) * stride
        first += min(0, extent)
        end += max(0, extent)
    if first < 0 or end > file_size:
        raise ValueError('Frame description exceeds its file')
    span = end - first
    # Map only the addressed bytes; NumPy handles page alignment for us.
    mapping = np.memmap(path, dtype=np.uint8, mode='r', offset=first, shape=(span,))
    relative = offset - first
    # Anchor at the first element in the target dtype; as_strided keeps that
    # dtype and accepts arbitrary byte strides, including negative ones.
    anchor = np.ndarray(shape=(1,), dtype=dtype, buffer=mapping, offset=relative)
    image = np.lib.stride_tricks.as_strided(
        anchor, shape=shape, strides=strides, writeable=False)
    # The memmap is the base of every view, so its finalizer runs only after
    # the last slice of the returned array is released.
    if adopt:
        key = _own_file(path, cleanup)
        weakref.finalize(mapping, _release_mapping, mapping._mmap, key)
    else:
        weakref.finalize(mapping, _close_mapping, mapping._mmap)
    return image
