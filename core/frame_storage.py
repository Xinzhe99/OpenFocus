"""Disk-backed frame storage whose lifetime follows numpy views."""
import os
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


def create_frame(shape, dtype):
    """Create an empty temporary mapping owned by its NumPy views."""
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
