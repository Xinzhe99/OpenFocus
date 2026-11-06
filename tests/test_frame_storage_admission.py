"""Regression coverage for temporary frame admission and rejected imports."""
import gc
import os
import weakref
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest

from controllers import source_manager
from core import frame_storage, workers


@pytest.fixture(autouse=True)
def isolated_storage_directory(monkeypatch):
    monkeypatch.setattr(frame_storage, '_storage_directory', None)


def test_switching_directories_preserves_existing_mapping(tmp_path):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    frame_storage.configure_storage_directory(first)
    old = frame_storage.create_frame((2, 2), np.uint8)
    old_path = first / os.path.basename(old.filename)
    view = old[:]
    old[:] = 37
    frame_storage.configure_storage_directory(second)
    new = frame_storage.create_frame((2, 2), np.uint8)
    new_path = second / os.path.basename(new.filename)
    assert old_path.exists() and new_path.exists()
    assert np.all(old == 37)
    del old
    gc.collect()
    assert old_path.exists()
    del view
    gc.collect()
    assert not old_path.exists()
    del new
    gc.collect()
    assert not new_path.exists()
    frame_storage.configure_storage_directory('')
    assert frame_storage.storage_directory() == frame_storage.tempfile.gettempdir()


@pytest.mark.parametrize('layout', ['crop', 'transpose', 'reverse'])
def test_stored_frame_preserves_strides_pixels_and_live_views(tmp_path, layout):
    frame_storage.configure_storage_directory(tmp_path)
    source = np.arange(21 * 27 * 3, dtype=np.uint16).reshape(21, 27, 3)
    pixels = {'crop': source[2:-2, 3:-3],
              'transpose': source.transpose(1, 0, 2),
              'reverse': source[::-1, ::-1]}[layout]
    mapped = frame_storage.store_frame(pixels)
    path = mapped.filename
    np.testing.assert_array_equal(mapped, pixels)
    assert mapped.dtype == pixels.dtype
    view = mapped[1:-1, 1:-1]
    expected = pixels[1:-1, 1:-1].copy()
    del mapped
    gc.collect()
    assert os.path.exists(path)
    np.testing.assert_array_equal(view, expected)
    del view
    gc.collect()
    assert not os.path.exists(path)


def test_invalid_directory_keeps_previous_configuration(monkeypatch, tmp_path):
    frame_storage.configure_storage_directory(tmp_path)
    with pytest.raises(OSError):
        frame_storage.configure_storage_directory(tmp_path / 'missing')
    assert frame_storage.storage_directory() == str(tmp_path)
    def denied(*args, **kwargs):
        raise PermissionError('denied')
    monkeypatch.setattr(frame_storage.tempfile, 'mkstemp', denied)
    with pytest.raises(OSError):
        frame_storage.configure_storage_directory(tmp_path)
    assert frame_storage.storage_directory() == str(tmp_path)


def test_mapping_checks_and_creates_in_same_captured_directory(monkeypatch, tmp_path):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    frame_storage.configure_storage_directory(first)
    usage = namedtuple('usage', 'total used free')
    checked = []
    def disk_usage(path):
        checked.append(path)
        frame_storage.configure_storage_directory(second)
        return usage(20 * 1024**3, 0, 2 * 1024**3)
    monkeypatch.setattr(frame_storage.shutil, 'disk_usage', disk_usage)
    image = frame_storage.create_frame((2, 2), np.uint8)
    assert checked == [str(first)]
    assert os.path.dirname(image.filename) == str(first)
    del image
    gc.collect()


def test_disk_error_reports_directory_and_space(monkeypatch, tmp_path):
    frame_storage.configure_storage_directory(tmp_path)
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(frame_storage.shutil, 'disk_usage',
                        lambda path: usage(20 * 1024**3, 0, 100))
    with pytest.raises(OSError) as error:
        frame_storage.require_disk_space(200)
    message = str(error.value)
    assert str(tmp_path) in message
    assert '200 bytes' in message and f'{1024**3} bytes reserved' in message
    assert 'free: 100 bytes' in message


def _limit_temp_space(monkeypatch, tmp_path, available):
    usage = namedtuple('usage', 'total used free')
    reserve = 1024 ** 3
    monkeypatch.setattr(frame_storage.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(frame_storage.shutil, 'disk_usage',
                        lambda path: usage(20 * 1024**3, 0, reserve + available))


def test_mapping_checks_disk_before_creating_file(monkeypatch, tmp_path):
    _limit_temp_space(monkeypatch, tmp_path, 0)
    with pytest.raises(OSError, match='Insufficient temporary disk space'):
        frame_storage.create_frame((2, 2, 3), np.uint16)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('multipage', [False, True])
def test_known_stack_open_does_not_admit_full_stack(qapp, monkeypatch, tmp_path, multipage):
    import cv2
    import tifffile
    paths = []
    if multipage:
        path = tmp_path / 'stack.tif'
        tifffile.imwrite(path, np.zeros((2, 8, 10), np.uint16), photometric='minisblack')
        paths.append(str(path))
        available = 10 * 8 * 3 * 2
    else:
        for index in range(2):
            path = tmp_path / f'{index}.png'
            assert cv2.imwrite(str(path), np.zeros((8, 10, 3), np.uint8))
            paths.append(str(path))
        available = 10 * 8 * 3
    _limit_temp_space(monkeypatch, tmp_path, available)
    loaded = [np.zeros((2, 3, 3), np.uint8)]
    loader = SimpleNamespace(images=loaded, should_cancel=None, progress_callback=None)
    loader.load_from_filepaths = lambda *args, **kwargs: (True, '', loaded, ['source'])
    worker = workers.StackLoadWorker(loader, filepaths=paths)
    results = []
    worker.finished_load.connect(lambda *args: results.append(args))
    worker.run()
    assert results[0] == (True, '', loaded, ['source'])
    assert not list(tmp_path.glob('openfocus-frame-*.bin'))


@pytest.mark.parametrize('registration_steps', [0, 1, 2])
def test_processing_stack_rejected_before_allocations(monkeypatch, tmp_path, registration_steps):
    frames = [np.zeros((8, 10, 3), np.uint16) for _ in range(2)]
    _limit_temp_space(monkeypatch, tmp_path, frames[0].nbytes)
    with pytest.raises(OSError, match='Insufficient temporary disk space'):
        workers._check_processing_space(frames, registration_steps)
    assert list(tmp_path.iterdir()) == []


def test_large_disk_reserve_is_capped(monkeypatch, tmp_path):
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(frame_storage.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(frame_storage.shutil, 'disk_usage',
                        lambda path: usage(10 * 1024**4, 0, 4 * 1024**3 + 100))
    frame_storage.require_disk_space(100)


@pytest.mark.parametrize('extension', ['.tiff', '.nef'])
@pytest.mark.parametrize('linear', [False, True])
@pytest.mark.parametrize('folder', [False, True])
def test_image_open_reaches_loader_without_full_stack_disk_admission(
        qapp, monkeypatch, tmp_path, extension, linear, folder):
    from core import image_loader
    paths = []
    for index in range(2):
        path = tmp_path / f'{index}{extension}'
        path.write_bytes(b'header read belongs to the lazy loader')
        paths.append(str(path))
    _limit_temp_space(monkeypatch, tmp_path, 0)
    monkeypatch.setattr(image_loader, 'is_raw_source', lambda path: True)
    frames = [np.zeros((2, 3, 3), np.uint16)]
    calls = []
    def load(*args, **kwargs):
        calls.append(args)
        return True, '', frames, ['source']
    loader = SimpleNamespace(images=frames, should_cancel=None, progress_callback=None,
                             load_from_filepaths=load, load_from_folder=load,
                             load_linear_raw_stack=load, linear_dng_metadata=None)
    worker = workers.StackLoadWorker(loader, folder=str(tmp_path) if folder else None,
                                    filepaths=None if folder else paths, linear_raw=linear)
    results = []
    worker.finished_load.connect(lambda *args: results.append(args))
    worker.run()
    assert len(calls) == 1
    assert results[0] == (True, '', frames, ['source'])
    assert not list(tmp_path.glob('openfocus-frame-*.bin'))


def test_cancelled_open_does_not_enter_loader(qapp, monkeypatch):
    loader = SimpleNamespace(images=[], should_cancel=None, progress_callback=None,
        load_from_filepaths=lambda *args, **kwargs: pytest.fail('cancelled open entered loader'))
    worker = workers.StackLoadWorker(loader, filepaths=['source.png'])
    worker.request_cancel()
    results = []
    worker.finished_load.connect(lambda *args: results.append(args))
    worker.run()
    assert results == [(False, 'CANCELLED', [], [])]


@pytest.mark.parametrize('outcome', ['mixed', 'append', 'superseded', 'prepare_error', 'shutdown', 'apply_error'])
def test_rejected_load_restores_loader_and_releases_frames(qapp, monkeypatch, outcome):
    monkeypatch.setattr(frame_storage, 'require_disk_space', lambda *args, **kwargs: None)
    previous = [np.zeros((2, 2, 3), np.uint8)]
    loader = SimpleNamespace(images=previous, image_paths=['old'], px_size_um=2,
                             source_exif=b'old', source_icc_profiles={'old': b'icc'},
                             should_cancel=None, progress_callback=None)
    released = []

    def load(paths, **kwargs):
        image = frame_storage.store_frame(np.ones((3, 3, 3), np.uint8))
        released.append(weakref.ref(image))
        loader.images = [image]
        loader.image_paths = ['new']
        loader.px_size_um = 7
        loader.source_exif = b'new'
        loader.source_icc_profiles = {'new': b'icc'}
        return True, '', loader.images, ['new']

    loader.load_from_filepaths = load
    window = SimpleNamespace(raw_images=previous, image_loader=loader,
                             current_scale_factor=1.0, current_folder_path='old',
                             source_px_um=2)
    manager = source_manager.SourceManager(window)
    monkeypatch.setattr(manager, '_show_load_status', lambda: None)
    monkeypatch.setattr(manager, '_finish_load_worker', lambda worker: None)

    def mixed(images):
        if outcome == 'superseded':
            manager._load_generation += 1
        if outcome == 'shutdown':
            window._shutting_down = True
        return outcome != 'mixed'

    monkeypatch.setattr(manager, '_confirm_mixed_sizes', mixed)
    monkeypatch.setattr(manager, '_confirm_append_dimensions', lambda images: outcome != 'append')
    def apply(options, **kwargs):
        window.raw_images = options.working_images
        window.source_px_um = 99
        raise ValueError('apply failed')

    if outcome == 'apply_error':
        monkeypatch.setattr(manager, '_apply_load_options', apply)
    errors = []
    def prepare(options):
        if outcome == 'prepare_error':
            raise ValueError('prepare failed')
        return options

    class SynchronousWorker(workers.StackLoadWorker):
        def start(self):
            self.run()

    monkeypatch.setattr(source_manager, 'StackLoadWorker', SynchronousWorker)
    manager._start_load_worker(filepaths=['new'], append=True,
                               prepare_options=prepare, on_error=errors.append)
    gc.collect()
    assert loader.images is previous
    assert loader.image_paths == ['old']
    assert loader.px_size_um == 2 and loader.source_exif == b'old'
    assert loader.source_icc_profiles == {'old': b'icc'}
    assert window.raw_images is previous and window.source_px_um == 2
    assert released[0]() is None


def test_failed_load_restores_loader_metadata(qapp, monkeypatch):
    previous = [np.zeros((1, 1), np.uint8)]
    loader = SimpleNamespace(images=previous, image_paths=['old'], px_size_um=4,
                             should_cancel=None, progress_callback=None)
    def fail(paths, **kwargs):
        loader.image_paths = ['new']
        loader.px_size_um = None
        raise ValueError('decode failed')
    loader.load_from_filepaths = fail
    worker = workers.StackLoadWorker(loader, filepaths=['new'])
    results = []
    worker.finished_load.connect(lambda *args: results.append(args))
    worker.run()
    assert not results[0][0]
    assert loader.images is previous and loader.image_paths == ['old']
    assert loader.px_size_um == 4


@pytest.mark.parametrize('cancelled', [False, True])
def test_failed_load_does_not_restore_over_newer_stack(qapp, monkeypatch, cancelled):
    previous = [np.zeros((1, 1), np.uint8)]
    replacement = [np.ones((1, 1), np.uint8)]
    loader = SimpleNamespace(images=previous, image_paths=['old'], px_size_um=4,
                             should_cancel=None, progress_callback=None)
    def fail(paths, **kwargs):
        loader.images = replacement
        loader.image_paths = ['replacement']
        loader.px_size_um = 9
        if cancelled:
            worker.request_cancel()
        raise ValueError('decode failed')
    loader.load_from_filepaths = fail
    worker = workers.StackLoadWorker(loader, filepaths=['new'])
    worker.run()
    assert loader.images is replacement
    assert loader.image_paths == ['replacement'] and loader.px_size_um == 9
