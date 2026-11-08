"""Independent snapshots and bounded RAM scratch whose lifetime follows views."""
import errno
import os
import math
import shutil
import tempfile
import weakref
import contextvars
import contextlib
import subprocess
import sys
import threading


import numpy as np
from constants import FRAME_RAM_BUDGET_BYTES

_storage_directory = None
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
    cap = (min(FRAME_RAM_BUDGET_BYTES, policy.ram_budget(snapshot) // 2)
           if snapshot.ram_known else 0)
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
        def readable(value):
            if value < 1024:
                return f'{value} B'
            for unit, factor in (('GiB', 1024 ** 3), ('MiB', 1024 ** 2), ('KiB', 1024)):
                if value >= factor:
                    return f'{value / factor:.2f} {unit}'
        available = '—' if self.available is None else readable(self.available)
        return template.format(path=self.directory, required=readable(self.required), available=available)


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


class _Scratch:
    def __init__(self):
        self.explicit_directory = _storage_directory
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
        if state.explicit_directory is not None or _default_is_local(default):
            yield default
        if state.explicit_directory is not None:
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
    return _storage_directory if _storage_directory is not None else tempfile.gettempdir()


def configure_storage_directory(path):
    """Select a writable temporary directory, or reset to the system default."""
    global _storage_directory
    if path is None or os.fspath(path) in ('', b''):
        _storage_directory = None
        return
    directory = os.path.abspath(os.fsdecode(path))
    fd, probe = tempfile.mkstemp(prefix='openfocus-probe-', dir=directory)
    try:
        os.close(fd)
    finally:
        os.unlink(probe)
    _storage_directory = directory


def restore_storage_directory(path):
    """Restore a saved choice without replacing it if it is unavailable."""
    global _storage_directory
    _storage_directory = (None if path is None or os.fspath(path) in ('', b'')
                          else os.path.abspath(os.fsdecode(path)))


def _release_file(path, state):
    try:
        _remove(path)
    finally:
        if state is not None:
            state.release()


def track_temp_file(owner, path):
    """Keep an owned scratch directory until this temporary file is released."""
    state = _scratch.get()
    state = state if state is not None and state.directory == os.path.dirname(path) else None
    if state is not None:
        state.retain()
    weakref.finalize(owner, _release_file, path, state)



def _remove(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass

def _release(mapping, path, state):
    try:
        mapping.close()
        _remove(path)
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


def _frame_file(directory, size):
    try:
        return tempfile.mkstemp(prefix="openfocus-frame-", suffix=".bin", dir=directory)
    except OSError as exc:
        raise _storage_error(directory, size, exc) from exc



def create_frame(shape, dtype, *, directory=None):
    """Create an empty temporary mapping owned by its NumPy views."""
    directory = storage_directory() if directory is None else directory
    size = math.prod(shape) * np.dtype(dtype).itemsize
    require_disk_space(size, directory)
    fd, path = _frame_file(directory, size)
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
        weakref.finalize(mapped, _release, mapped._mmap, path, state)
        return mapped
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        _remove(path)
        raise _storage_error(directory, size, exc) from exc
    except BaseException:
        if fd is not None:
            os.close(fd)
        _remove(path)
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
    try:
        # Write through the file descriptor before mapping it. Filling a writable
        # mapping faults every source-stack page into this process even though
        # the caller only needs one frame/strip at a time afterward.
        with os.fdopen(fd, "wb") as stream:
            np.asarray(image).tofile(stream)
        mapped = np.memmap(path, dtype=image.dtype, mode="r+", shape=image.shape)
        state = _scratch.get()
        state = state if state is not None and state.directory == directory else None
        if state is not None:
            state.retain()
        weakref.finalize(mapped, _release, mapped._mmap, path, state)
        return mapped
    except OSError as exc:
        _remove(path)
        raise _storage_error(directory, image.nbytes, exc) from exc
    except BaseException:
        _remove(path)
        raise
