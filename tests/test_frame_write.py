"""Large-chunk frame writes keep the bytes of a buffered ``tofile``."""
import errno
import os

import numpy as np
import pytest

from core import frame_storage


def _read(path):
    with open(path, 'rb') as stream:
        return stream.read()


@pytest.mark.parametrize('shape,dtype,transform', [
    ((7, 11), np.uint8, 'plain'),
    ((5, 6, 3), np.uint16, 'plain'),
    ((5, 6, 3), np.uint16, 'reversed'),
    ((5, 6, 3), '>u2', 'reversed'),
    ((4, 5), 'datetime64[ns]', 'reversed'),
    ((8, 30, 3), np.uint16, 'strided'),
    ((9, 4), np.float32, 'fortran'),
    ((13, 5, 1), np.uint16, 'drop'),
    ((0, 4, 3), np.uint8, 'plain'),
])
def test_write_frame_writes_what_tofile_writes(tmp_path, monkeypatch, shape, dtype, transform):
    rng = np.random.default_rng(5)
    if dtype == np.float32:
        image = rng.random(size=shape).astype(np.float32)
    else:
        image = rng.integers(0, 65536, size=shape, dtype=np.uint16).astype(dtype)
    if transform == 'reversed':
        image = image[..., ::-1]
    elif transform == 'strided':
        image = np.repeat(np.repeat(image, 3, axis=1), 2, axis=0)[:, ::3, :]
    elif transform == 'fortran':
        image = np.asfortranarray(image)
    elif transform == 'drop':
        image = image[..., 0]
    monkeypatch.setattr(frame_storage, '_WRITE_CHUNK_BYTES', 13)
    expected = tmp_path / 'expected.bin'
    image.tofile(str(expected))
    target = tmp_path / 'actual.bin'
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_BINARY', 0))
    try:
        frame_storage.write_frame(descriptor, image)
    finally:
        os.close(descriptor)
    assert image.dtype == dtype
    assert _read(target) == _read(expected)


def test_write_frame_continues_after_short_writes(tmp_path, monkeypatch):
    image = np.arange(3000, dtype=np.uint16).reshape(10, 300)  # 6000 bytes
    real_write = os.write
    sizes = []

    def short_write(descriptor, data):
        sizes.append(len(data))
        return real_write(descriptor, bytes(data[:777]))

    monkeypatch.setattr(frame_storage.os, 'write', short_write)
    target = tmp_path / 'short.bin'
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_BINARY', 0))
    try:
        frame_storage.write_frame(descriptor, image)
    finally:
        os.close(descriptor)
    assert sizes == [6000, 6000 - 777, 6000 - 2 * 777, 6000 - 3 * 777, 6000 - 4 * 777,
                     6000 - 5 * 777, 6000 - 6 * 777, 6000 - 7 * 777]
    with open(target, 'rb') as stream:
        assert np.frombuffer(stream.read(), dtype=np.uint16).tolist() == image.ravel().tolist()


def test_write_frame_reports_a_stalled_descriptor(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_storage.os, 'write', lambda descriptor, data: 0)
    descriptor = os.open(tmp_path / 'stalled.bin', os.O_WRONLY | os.O_CREAT | getattr(os, 'O_BINARY', 0))
    try:
        with pytest.raises(OSError, match='no progress'):
            frame_storage.write_frame(descriptor, np.ones((2, 2), np.uint8))
    finally:
        os.close(descriptor)


def test_stored_mapped_frame_round_trips_a_reversed_view(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    source = np.arange(3 * 4 * 3, dtype=np.uint16).reshape(3, 4, 3)
    mapped = frame_storage.store_frame(source[..., ::-1], prefer_ram=False)
    assert isinstance(mapped, np.memmap)
    np.testing.assert_array_equal(np.asarray(mapped), source[..., ::-1])


def test_write_frame_rejects_object_arrays(tmp_path):
    path = tmp_path / 'objects.bin'
    with path.open('wb', buffering=0) as stream:
        with pytest.raises(OSError, match='object arrays'):
            frame_storage.write_frame(stream.fileno(), np.array([None], dtype=object))
    assert path.stat().st_size == 0


@pytest.mark.parametrize('preview', [False, True])
def test_partial_scratch_write_discards_file(tmp_path, monkeypatch, preview):
    from core.source_frames import store_preview

    def disk_full(fd, image):
        os.write(fd, b'\x01')
        raise OSError(errno.ENOSPC, 'disk full during frame write')

    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))
    monkeypatch.setattr(frame_storage, 'write_frame', disk_full)
    image = np.ones((5, 6, 3), np.uint16)
    observed = []
    with frame_storage.observe_temp_files(observed.append):
        with pytest.raises(frame_storage.ScratchSpaceError) as failure:
            if preview:
                store_preview(image)
            else:
                frame_storage.store_frame(image, prefer_ram=False)
    assert failure.value.required == image.nbytes
    assert failure.value.__cause__.errno == errno.ENOSPC
    path, = observed
    assert not os.path.exists(path)
    assert not frame_storage.owns_temp_file(path)
