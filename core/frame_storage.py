"""Disk-backed frame storage whose lifetime follows numpy views."""
import os
import math
import shutil
import tempfile
import weakref

import numpy as np


def _remove(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass

def _release(mapping, path):
    mapping.close()
    _remove(path)

def require_disk_space(size):
    """Check a point-in-time estimate, retaining space for the system.

    Other processes can consume space after this check; it is not a reservation.
    """
    usage = shutil.disk_usage(tempfile.gettempdir())
    reserve = max(1024 ** 3, min(usage.total // 20, 4 * 1024 ** 3))
    if size > max(0, usage.free - reserve):
        raise OSError(f"Insufficient temporary disk space: need {size} bytes, "
                      f"with {reserve} bytes reserved (free: {usage.free} bytes)")



def create_frame(shape, dtype):
    """Create an empty temporary mapping owned by its NumPy views."""
    require_disk_space(math.prod(shape) * np.dtype(dtype).itemsize)
    fd, path = tempfile.mkstemp(prefix="openfocus-frame-", suffix=".bin")
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
    mapped = create_frame(image.shape, image.dtype)
    mapped[...] = image
    mapped.flush()
    return mapped
