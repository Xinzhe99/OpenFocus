"""Non-blocking device uploads never read from a writable file mapping."""
import ast
import gc
import os
from pathlib import Path

import numpy as np
import pytest

from core import frame_storage
from core.frame_storage import _mapped_base, async_upload_view

ROOT = Path(__file__).resolve().parents[1]

# Device-to-host copies whose destination is a fresh RAM array, by (file, destination).
RAM_DOWNLOADS = {('fusion_methods/_openfocus_gpu_gfg.py', 'host_acc')}


@pytest.fixture(autouse=True)
def isolated_storage_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    monkeypatch.setattr(frame_storage, '_storage_notice_path', None)
    monkeypatch.setattr(frame_storage, '_storage_fallback_notifier', None)
    frame_storage.configure_storage_directory(tmp_path)


def _writable_frames():
    created = frame_storage.create_frame((9, 11, 3), np.uint16)
    created[:] = np.arange(created.size, dtype=np.uint16).reshape(created.shape)
    stored = frame_storage.store_frame(np.asarray(created) + 1, prefer_ram=False)
    return created, stored


@pytest.mark.skipif(os.name == 'nt', reason='Windows keeps the writable mapping')
@pytest.mark.parametrize('key', [
    np.s_[...], np.s_[3:7], np.s_[:, 2:5], np.s_[..., ::-1], np.s_[4], np.s_[1:8:3, ::2, 1],
], ids=['whole', 'band', 'columns', 'reversed-channels', 'row', 'strided-plane'])
def test_writable_mapping_is_reopened_read_only_over_the_same_bytes(key):
    for frame in _writable_frames():
        assert frame.mode in ('r+', 'w+')  # w+ where the file could not be preallocated
        source = frame[key]
        view = async_upload_view(source)
        root = _mapped_base(view)
        assert view is not source
        assert root.mode == 'r' and root.filename == frame.filename
        assert not view.flags.writeable
        assert (view.shape, view.strides, view.dtype) == (source.shape, source.strides, source.dtype)
        assert np.array_equal(view, source)
        # The same file pages, not a copy: later writes through the frame show through.
        frame[...] = 7
        assert np.all(view == 7)
        del view, root, source


@pytest.mark.skipif(os.name == 'nt', reason='Windows keeps the writable mapping')
def test_reopened_view_keeps_its_mapping_after_the_frame_is_released():
    frame = frame_storage.create_frame((6, 5), np.float32)
    frame[:] = 3.5
    view = async_upload_view(frame)
    del frame
    gc.collect()
    assert np.all(view == 3.5)


def test_ram_read_only_and_private_sources_are_returned_unchanged(tmp_path):
    ram = np.arange(12, dtype=np.uint16).reshape(3, 4)
    path = tmp_path / 'pixels.bin'
    ram.tofile(path)
    read_only = np.memmap(path, dtype=np.uint16, mode='r', shape=ram.shape)
    private = np.memmap(path, dtype=np.uint16, mode='c', shape=ram.shape)
    frame = frame_storage.create_frame((4, 4), np.uint8)
    with frame_storage.frame_region(frame, np.s_[1:3]) as borrowed:
        for source in (ram, ram[:, ::2], read_only, read_only[1:], private, borrowed,
                       np.empty((0, 3), np.uint8), frame[:0]):
            assert async_upload_view(source) is source
    del read_only, private


def test_windows_uploads_keep_the_writable_mapping(monkeypatch):
    monkeypatch.setattr(frame_storage, '_READ_ONLY_UPLOADS', False)
    frame = frame_storage.create_frame((4, 4), np.uint8)
    assert async_upload_view(frame) is frame


def _non_blocking_copies():
    for directory in ('core', 'fusion_methods'):
        for path in sorted((ROOT / directory).glob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'enqueue_copy'
                        and any(keyword.arg == 'is_blocking' and isinstance(keyword.value, ast.Constant)
                                and keyword.value.value is False for keyword in node.keywords)):
                    yield path.relative_to(ROOT).as_posix(), node


def test_every_non_blocking_upload_reads_through_the_read_only_view():
    copies = list(_non_blocking_copies())
    assert len(copies) >= 17
    unguarded = []
    for name, node in copies:
        destination, source = node.args[1], node.args[2]
        if isinstance(destination, ast.Name) and (name, destination.id) in RAM_DOWNLOADS:
            continue
        if not (isinstance(source, ast.Call) and isinstance(source.func, ast.Name)
                and source.func.id == 'async_upload_view'):
            unguarded.append(f'{name}:{node.lineno}')
    assert not unguarded
