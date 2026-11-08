"""Regression coverage for mapped frame descriptions, transfer and ownership."""
import gc
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core import frame_storage
from core import resource_policy as policy


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch):
    monkeypatch.setattr(frame_storage, '_storage_directory', None)


def test_heap_arrays_have_no_mapping(tmp_path):
    assert frame_storage.describe_frame(np.zeros((4, 5, 3), np.uint16)) is None
    assert not frame_storage.owns_temp_file(tmp_path / 'ordinary-user-file.bin')


@pytest.mark.parametrize('layout', ['crop', 'transpose', 'reverse', 'strided'])
def test_uint16_mapped_views_roundtrip_without_copy(tmp_path, layout):
    frame_storage.configure_storage_directory(tmp_path)
    source = np.arange(12 * 10 * 3, dtype=np.uint16).reshape(12, 10, 3)
    mapped = frame_storage.store_frame(source, prefer_ram=False)
    path = mapped.filename
    views = {'crop': mapped[2:9, 3:8],
             'transpose': mapped.transpose(1, 0, 2),
             'reverse': mapped[::-1, ::-1],
             'strided': mapped[::2, ::3, ::2]}
    view = views[layout]
    description = frame_storage.describe_frame(view)
    assert description is not None
    assert description['path'] == path
    assert description['shape'] == view.shape
    assert description['strides'] == view.strides
    assert description['dtype'] == np.dtype(np.uint16).str
    pointer = view.__array_interface__['data'][0]
    base = mapped.__array_interface__['data'][0]
    assert description['offset'] == pointer - base + mapped.offset
    reopened = frame_storage.open_frame(description)
    assert reopened.dtype == np.uint16 and reopened.shape == view.shape
    assert not reopened.flags.writeable
    np.testing.assert_array_equal(reopened, view)
    if layout == 'reverse':
        assert any(stride < 0 for stride in description['strides'])
    with pytest.raises(ValueError):
        reopened.flat[0] = 0
    del reopened, view, views, description
    gc.collect()
    assert os.path.exists(path)
    del mapped
    gc.collect()
    assert not os.path.exists(path)


def test_reopened_views_keep_absolute_offsets(tmp_path):
    frame_storage.configure_storage_directory(tmp_path)
    source = np.arange(7 * 9 * 2, dtype=np.uint16).reshape(7, 9, 2)
    mapped = frame_storage.store_frame(source, prefer_ram=False)
    path = mapped.filename
    outer = frame_storage.open_frame(frame_storage.describe_frame(mapped))
    inner = outer[1:-1, ::2]
    description = frame_storage.describe_frame(inner)
    assert description is not None
    assert description['path'] == path
    reopened = frame_storage.open_frame(description)
    np.testing.assert_array_equal(reopened, inner)
    del reopened, inner, outer, description
    gc.collect()
    assert os.path.exists(path)
    del mapped
    gc.collect()
    assert not os.path.exists(path)


@pytest.mark.parametrize('filename', ['user-frame.bin', 'openfocus-frame-user-owned.bin'])
def test_borrowed_user_mapping_is_never_unlinked(tmp_path, filename):
    path = tmp_path / filename
    user = np.memmap(path, dtype=np.uint16, mode='w+', shape=(6, 4))
    user[:] = np.arange(24, dtype=np.uint16).reshape(6, 4)
    user.flush()
    description = frame_storage.describe_frame(user)
    assert description is not None
    expected = np.array(user)
    del user
    gc.collect()
    reopened = frame_storage.open_frame(description)
    np.testing.assert_array_equal(reopened, expected)
    del reopened
    gc.collect()
    assert path.exists()
    assert not frame_storage.owns_temp_file(path)


def test_owned_reopen_deletes_after_the_last_owner(tmp_path):
    frame_storage.configure_storage_directory(tmp_path)
    mapped = frame_storage.create_frame((8,), np.uint16)
    path = mapped.filename
    reopened = frame_storage.open_frame(
        frame_storage.describe_frame(mapped), owned=True)
    del mapped
    gc.collect()
    assert os.path.exists(path)
    del reopened
    gc.collect()
    assert not os.path.exists(path)


def test_duplicate_owners_and_disown_keep_the_file_until_manual_cleanup(tmp_path):
    frame_storage.configure_storage_directory(tmp_path)
    mapped = frame_storage.create_frame((8,), np.uint16)
    mapped[:] = np.arange(8, dtype=np.uint16)
    path = mapped.filename
    first = frame_storage.open_frame(
        frame_storage.describe_frame(mapped), owned=True)
    second = frame_storage.open_frame(
        frame_storage.describe_frame(mapped[::-1]), owned=True)
    frame_storage.disown_temp_files([path])
    del mapped
    gc.collect()
    assert os.path.exists(path)
    del first
    gc.collect()
    assert os.path.exists(path)
    del second
    gc.collect()
    assert os.path.exists(path)
    os.unlink(path)


def test_disowning_still_releases_scratch_reference(tmp_path):
    frame_storage.configure_storage_directory(tmp_path)
    directory = tmp_path / 'owned-scratch'
    directory.mkdir()
    state = frame_storage._Scratch()
    state.directory = str(directory)
    state.owned = True
    mapped = frame_storage.run_with_scratch(
        state, frame_storage.create_frame, (4,), np.uint16)
    path = mapped.filename
    assert state.live == 1
    frame_storage.disown_temp_files([path])
    del mapped
    gc.collect()
    assert state.live == 0
    assert os.path.exists(path)
    state.close()
    assert directory.exists()
    os.unlink(path)
    os.rmdir(directory)


def test_open_frame_removes_empty_cleanup_directory_after_last_owner(tmp_path):
    frame_storage.configure_storage_directory(tmp_path)
    directory = tmp_path / 'adopted-scratch'
    directory.mkdir()
    mapped = frame_storage.create_frame((2, 3), np.uint16, directory=str(directory))
    path = mapped.filename
    description = frame_storage.describe_frame(mapped)
    description['cleanup_directory'] = str(directory)
    reopened = frame_storage.open_frame(description, owned=True)
    del mapped
    gc.collect()
    assert os.path.exists(path)
    del reopened
    gc.collect()
    assert not os.path.exists(path)
    assert not directory.exists()


def test_cleanup_directory_with_a_foreign_file_is_never_removed(tmp_path):
    frame_storage.configure_storage_directory(tmp_path)
    directory = tmp_path / 'shared-scratch'
    directory.mkdir()
    sentinel = directory / 'keep.txt'
    sentinel.write_text('keep')
    mapped = frame_storage.create_frame((2, 3), np.uint16, directory=str(directory))
    path = mapped.filename
    description = frame_storage.describe_frame(mapped)
    description['cleanup_directory'] = str(directory)
    reopened = frame_storage.open_frame(description, owned=True)
    del mapped
    gc.collect()
    del reopened
    gc.collect()
    assert not os.path.exists(path)
    assert sentinel.exists()
    assert directory.exists()


def test_observer_reports_creation_before_bytes_are_written_across_threads(tmp_path):
    frame_storage.configure_storage_directory(tmp_path)
    seen = []
    produced = []

    def observer(path):
        seen.append((path, os.path.getsize(path)))

    with frame_storage.observe_temp_files(observer):
        thread = threading.Thread(
            target=lambda: produced.append(frame_storage.create_frame((1024,), np.uint16)))
        thread.start()
        thread.join()
    mapped = produced[0]
    path = mapped.filename
    assert seen == [(path, 0)]
    assert frame_storage.owns_temp_file(path)
    del mapped, produced
    gc.collect()
    assert not os.path.exists(path)


def test_failed_observer_removes_the_partial_file(tmp_path):
    frame_storage.configure_storage_directory(tmp_path)

    def reject(path):
        raise RuntimeError(f'cannot observe {path}')

    with pytest.raises(RuntimeError):
        with frame_storage.observe_temp_files(reject):
            frame_storage.create_frame((4,), np.uint8)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change', ['negative-shape', 'text-shape', 'past-end'])
def test_open_frame_validates_numeric_layout_and_file_span(tmp_path, change):
    frame_storage.configure_storage_directory(tmp_path)
    mapped = frame_storage.create_frame((4, 5), np.uint16)
    path = mapped.filename
    description = frame_storage.describe_frame(mapped)
    if change == 'negative-shape':
        description['shape'] = (-1, 5)
    elif change == 'text-shape':
        description['shape'] = ('4', 5)
    else:
        description['offset'] = os.path.getsize(path)
    with pytest.raises(ValueError):
        frame_storage.open_frame(description)
    del mapped
    gc.collect()
    assert not os.path.exists(path)


def test_tracked_preview_shares_owners_and_the_cleanup_directory(tmp_path):
    frame_storage.configure_storage_directory(tmp_path)
    directory = tmp_path / 'preview-scratch'
    directory.mkdir()
    fd, path = frame_storage._frame_file(
        str(directory), 4, prefix='openfocus-preview-')
    with os.fdopen(fd, 'wb') as stream:
        stream.write(b'abcd')

    class Holder:
        pass

    first, second = Holder(), Holder()
    frame_storage.track_temp_file(first, path, cleanup_directory=str(directory))
    frame_storage.track_temp_file(second, path)
    assert frame_storage.owns_temp_file(path)
    del first
    gc.collect()
    assert os.path.exists(path)
    del second
    gc.collect()
    assert not os.path.exists(path)
    assert not directory.exists()


def test_ram_snapshot_reports_live_transient_reservations(monkeypatch):
    monkeypatch.setattr(policy, 'frame_ram_budget', lambda snapshot: 1 << 20)
    monkeypatch.setattr(policy, 'probe_memory', lambda: SimpleNamespace(ram_known=True))
    monkeypatch.setattr(policy, 'require_memory', lambda size, snapshot=None: size)
    before = frame_storage.ram_snapshot_bytes()
    frame = frame_storage._try_ram_frame((512,), np.uint16)
    assert frame is not None and frame.flags.owndata
    assert frame_storage.ram_snapshot_bytes() == before + 1024
    del frame
    gc.collect()
    assert frame_storage.ram_snapshot_bytes() == before


@pytest.mark.parametrize('source_kind', ['sequence', 'iterator'])
def test_materializing_reopened_strided_frames_reuses_owned_storage(tmp_path, source_kind):
    from core.source_frames import materialize_frames

    frame_storage.configure_storage_directory(tmp_path)
    pixels = np.arange(12 * 10 * 3, dtype=np.uint16).reshape(12, 10, 3)
    mapped = frame_storage.store_frame(pixels, prefer_ram=False)
    path = mapped.filename
    reopened = frame_storage.open_frame(
        frame_storage.describe_frame(mapped[::-1, ::2]), owned=True)
    expected = pixels[::-1, ::2].copy()
    argument = [reopened] if source_kind == 'sequence' else iter((reopened,))
    result = materialize_frames(argument, prefer_ram=False)
    assert set(tmp_path.iterdir()) == {tmp_path / os.path.basename(path)}
    assert frame_storage.describe_frame(result[0])['path'] == path
    assert result[0].strides == reopened.strides
    del argument, reopened, mapped
    gc.collect()
    assert os.path.exists(path)
    np.testing.assert_array_equal(result[0], expected)
    del result
    gc.collect()
    assert not os.path.exists(path)
