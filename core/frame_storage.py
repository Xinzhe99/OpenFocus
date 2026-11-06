"""Disk-backed frame storage whose lifetime follows numpy views."""
import os
import math
import shutil
import tempfile
import weakref

import numpy as np

_storage_directory = None


def storage_directory():
    """Return the directory currently used for new temporary data."""
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



def _remove(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass

def _release(mapping, path):
    mapping.close()
    _remove(path)

def require_disk_space(size, directory=None):
    """Check a point-in-time estimate, retaining space for the system.

    Other processes can consume space after this check; it is not a reservation.
    """
    directory = storage_directory() if directory is None else directory
    usage = shutil.disk_usage(directory)
    reserve = max(1024 ** 3, min(usage.total // 20, 4 * 1024 ** 3))
    if size > max(0, usage.free - reserve):
        raise OSError(f"Insufficient temporary disk space in {directory}: need {size} bytes, "
                      f"with {reserve} bytes reserved (free: {usage.free} bytes)")



def create_frame(shape, dtype, *, directory=None):
    """Create an empty temporary mapping owned by its NumPy views."""
    directory = storage_directory() if directory is None else directory
    require_disk_space(math.prod(shape) * np.dtype(dtype).itemsize, directory)
    fd, path = tempfile.mkstemp(prefix="openfocus-frame-", suffix=".bin", dir=directory)
    os.close(fd)
    try:
        mapped = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
        weakref.finalize(mapped, _release, mapped._mmap, path)
        return mapped
    except BaseException:
        _remove(path)
        raise


def store_frame(image):
    """Copy one frame to a temporary mapping, preserving dtype and shape.

    The mapping, including any derived views, owns the file. It is removed
    after the last view is released, including on cancellation or exceptions.
    """
    directory = storage_directory()
    require_disk_space(image.nbytes, directory)
    fd, path = tempfile.mkstemp(prefix="openfocus-frame-", suffix=".bin", dir=directory)
    try:
        # Write through the file descriptor before mapping it. Filling a writable
        # mapping faults every source-stack page into this process even though
        # the caller only needs one frame/strip at a time afterward.
        with os.fdopen(fd, "wb") as stream:
            np.asarray(image).tofile(stream)
        mapped = np.memmap(path, dtype=image.dtype, mode="r+", shape=image.shape)
        weakref.finalize(mapped, _release, mapped._mmap, path)
        return mapped
    except BaseException:
        _remove(path)
        raise
