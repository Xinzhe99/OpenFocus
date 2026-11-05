"""Regression coverage for temporary frame admission and rejected imports."""
import gc
import weakref
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest

from controllers import source_manager
from core import frame_storage, workers


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
def test_known_stack_rejected_before_loading(qapp, monkeypatch, tmp_path, multipage):
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
    loader = SimpleNamespace(images=[], should_cancel=None, progress_callback=None)
    loader.load_from_filepaths = lambda *args, **kwargs: pytest.fail('decode started')
    worker = workers.StackLoadWorker(loader, filepaths=paths)
    results = []
    worker.finished_load.connect(lambda *args: results.append(args))
    worker.run()
    assert not results[0][0] and 'Insufficient temporary disk space' in results[0][1]
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
def test_raw_headers_admit_full_stack_before_decode(qapp, monkeypatch, tmp_path,
                                                    extension, linear, folder):
    import tifffile
    from core import image_loader
    paths = []
    for index in range(2):
        path = tmp_path / f'{index}{extension}'
        if extension == '.tiff':
            tifffile.imwrite(path, np.zeros((2, 3, 3), np.uint8), photometric='rgb')
        else:
            path.write_bytes(b'raw header')
        paths.append(str(path))
    _limit_temp_space(monkeypatch, tmp_path, 100 * 80 * 3 * 2)
    monkeypatch.setattr(image_loader, 'is_raw_source', lambda path: str(path).endswith(extension))
    monkeypatch.setattr(image_loader, 'RAWPY_AVAILABLE', True)
    dimensions = SimpleNamespace(width=100, height=80, raw_width=100, raw_height=80, pixel_aspect=1)
    class Header:
        sizes = dimensions
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(image_loader, 'rawpy', SimpleNamespace(imread=lambda stream: Header()), raising=False)
    loader = image_loader.ImageStackLoader()
    monkeypatch.setattr(loader, 'load_from_filepaths', lambda *args, **kwargs: pytest.fail('decode started'))
    monkeypatch.setattr(loader, 'load_from_folder', lambda *args, **kwargs: pytest.fail('decode started'))
    monkeypatch.setattr(loader, 'load_linear_raw_stack', lambda *args, **kwargs: pytest.fail('decode started'))
    if extension == '.nef':
        monkeypatch.setattr(loader, 'SUPPORTED_FORMATS', loader.SUPPORTED_FORMATS | {'.nef'})
    worker = workers.StackLoadWorker(loader, folder=str(tmp_path) if folder else None,
                                     filepaths=None if folder else paths, linear_raw=linear)
    results = []
    worker.finished_load.connect(lambda *args: results.append(args))
    worker.run()
    assert not results[0][0] and 'Insufficient temporary disk space' in results[0][1]
    assert not list(tmp_path.glob('openfocus-frame-*.bin'))


def test_stack_preflight_observes_cancellation(monkeypatch, tmp_path):
    from core import image_loader
    path = tmp_path / 'frame.nef'
    path.write_bytes(b'raw')
    monkeypatch.setattr(image_loader, 'is_raw_source', lambda path: True)
    monkeypatch.setattr(image_loader, 'RAWPY_AVAILABLE', True)
    cancelled = [False]
    class Header:
        sizes = SimpleNamespace(width=10, height=10, raw_width=10, raw_height=10, pixel_aspect=1)
        def __enter__(self):
            cancelled[0] = True
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(image_loader, 'rawpy', SimpleNamespace(imread=lambda stream: Header()), raising=False)
    with pytest.raises(workers.RegistrationCancelled):
        workers._check_known_stack_space([str(path)], 1, lambda: cancelled[0])


@pytest.mark.parametrize('outcome', ['mixed', 'append', 'superseded', 'prepare_error', 'shutdown', 'apply_error'])
def test_rejected_load_restores_loader_and_releases_frames(qapp, monkeypatch, outcome):
    monkeypatch.setattr(frame_storage, 'require_disk_space', lambda size: None)
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
    monkeypatch.setattr(workers, '_check_known_stack_space', lambda *args: None)

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
    monkeypatch.setattr(workers, '_check_known_stack_space', lambda *args: None)
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
    monkeypatch.setattr(workers, '_check_known_stack_space', lambda *args: None)
    worker = workers.StackLoadWorker(loader, filepaths=['new'])
    worker.run()
    assert loader.images is replacement
    assert loader.image_paths == ['replacement'] and loader.px_size_um == 9
