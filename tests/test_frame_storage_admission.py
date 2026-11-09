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
    monkeypatch.setattr(frame_storage, '_storage_notice_path', None)
    monkeypatch.setattr(frame_storage, '_storage_fallback_notifier', None)


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


@pytest.mark.parametrize('prefer_ram', [False, True])
@pytest.mark.parametrize('layout', ['crop', 'transpose', 'reverse'])
def test_stored_frame_preserves_strides_pixels_and_live_views(tmp_path, monkeypatch, layout, prefer_ram):
    from core import resource_policy
    frame_storage.configure_storage_directory(tmp_path)
    monkeypatch.setattr(resource_policy, 'FRAME_RAM_BUDGET_BYTES', 0)
    source = np.arange(21 * 27 * 3, dtype=np.uint16).reshape(21, 27, 3)
    pixels = {'crop': source[2:-2, 3:-3],
              'transpose': source.transpose(1, 0, 2),
              'reverse': source[::-1, ::-1]}[layout]
    mapped = frame_storage.store_frame(pixels, prefer_ram=prefer_ram)
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
    assert frame_storage._storage_directory == str(tmp_path)


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
    assert '200 B' in message and '0 B' in message
    assert error.value.required == 200 and error.value.available == 0


def _limit_temp_space(monkeypatch, tmp_path, available):
    usage = namedtuple('usage', 'total used free')
    reserve = 1024 ** 3
    monkeypatch.setattr(frame_storage.tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.setattr(frame_storage.shutil, 'disk_usage',
                        lambda path: usage(20 * 1024**3, 0, reserve + available))


def test_mapping_checks_disk_before_creating_file(monkeypatch, tmp_path):
    _limit_temp_space(monkeypatch, tmp_path, 0)
    with pytest.raises(frame_storage.ScratchSpaceError):
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
    with pytest.raises(frame_storage.ScratchSpaceError) as error:
        workers._check_processing_space(frames, registration_steps)
    assert error.value.directory == str(tmp_path)
    assert error.value.required > error.value.available == frames[0].nbytes
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
                             failed_count=3, should_cancel=None, progress_callback=None)
    def fail(paths, **kwargs):
        loader.image_paths = ['new']
        loader.px_size_um = None
        loader.failed_count = 1
        raise ValueError('decode failed')
    loader.load_from_filepaths = fail
    worker = workers.StackLoadWorker(loader, filepaths=['new'])
    results = []
    worker.finished_load.connect(lambda *args: results.append(args))
    worker.run()
    assert not results[0][0]
    assert loader.images is previous and loader.image_paths == ['old']
    assert loader.px_size_um == 4 and loader.failed_count == 3


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


def _fake_volumes(monkeypatch, tmp_path, default_available, alternative_available):
    default, alternative = tmp_path / 'default', tmp_path / 'alternative'
    default.mkdir()
    alternative.mkdir()
    monkeypatch.setattr(frame_storage.tempfile, 'gettempdir', lambda: str(default))
    monkeypatch.setattr(frame_storage, '_default_is_local', lambda path: True)
    monkeypatch.setattr(frame_storage, '_local_mounts', lambda: [str(alternative)])
    def space(path):
        available = (default_available if os.path.commonpath((str(default), str(path))) == str(default)
                     else alternative_available)
        return None, 0, available
    monkeypatch.setattr(frame_storage, '_space', space)
    return default, alternative


def test_default_uses_alternative_and_releases_views(monkeypatch, tmp_path):
    from core.source_frames import materialize_frames
    default, alternative = _fake_volumes(monkeypatch, tmp_path, 3, 10000)
    pixels = [np.arange(48, dtype=np.uint16).reshape(4, 4, 3)] * 2
    with frame_storage.scratch_directory():
        mapped = materialize_frames(pixels, staging_bytes=96, prefer_ram=False)
        directory = os.path.dirname(mapped[0].filename)
        assert os.path.dirname(directory) == str(alternative)
        assert all(os.path.dirname(frame.filename) == directory for frame in mapped)
        np.testing.assert_array_equal(mapped[0], pixels[0])
        view = mapped[0][:]
    assert not list(default.iterdir())
    del mapped
    gc.collect()
    assert os.path.isdir(directory)
    del view
    gc.collect()
    assert not os.path.exists(directory)


def test_explicit_directory_never_falls_back(monkeypatch, tmp_path):
    from core.source_frames import materialize_frames
    default, alternative = _fake_volumes(monkeypatch, tmp_path, 3, 10000)
    frame_storage.configure_storage_directory(default)
    with pytest.raises(frame_storage.ScratchSpaceError) as error:
        materialize_frames([np.zeros((4, 4), np.uint8)])
    assert error.value.directory == str(default)
    assert error.value.required == 16 and error.value.available == 3
    assert not list(default.iterdir()) and not list(alternative.iterdir())


def test_unwritable_alternative_is_skipped(monkeypatch, tmp_path):
    from core.source_frames import materialize_frames
    default, alternative = _fake_volumes(monkeypatch, tmp_path, 0, 10000)
    original = frame_storage.tempfile.mkdtemp
    def denied(*args, **kwargs):
        if kwargs.get('dir') == str(alternative):
            raise PermissionError('denied')
        return original(*args, **kwargs)
    monkeypatch.setattr(frame_storage.tempfile, 'mkdtemp', denied)
    with pytest.raises(frame_storage.ScratchSpaceError):
        materialize_frames([np.zeros((4, 4), np.uint8)])
    assert not list(default.iterdir()) and not list(alternative.iterdir())


@pytest.mark.parametrize('operation', ['fusion', 'registration', 'retained'])
def test_deferred_processing_uses_available_local_volume(monkeypatch, tmp_path, operation):
    from core.multi_focus_fusion import MultiFocusFusion
    from core.registration import ImageRegistration
    from core.source_frames import process_frame, resolve_frame, scan_source
    from utils.image_utils import imwrite_auto

    pixels = np.random.default_rng(998).integers(0, 65536, (48, 64, 3), np.uint16)
    paths = [tmp_path / 'first.png', tmp_path / 'second.png']
    for path in paths:
        assert imwrite_auto(path, pixels, exif=b'')
    frames = [scan_source(str(path), header_only=True)[0] for path in paths]

    def fuse():
        return MultiFocusFusion('guided_filter', use_gpu=False, tile_enabled=False,
                                exposure_match=False).fuse(frames, thread_count=1)

    def register():
        return ImageRegistration('ecc', defer_warps=True).process(frames, thread_count=1)

    def flipped_mapping(image):
        result = frame_storage.create_frame(image.shape, image.dtype)
        result[:] = image[:, ::-1]
        return result

    retained = process_frame(frames[0], flipped_mapping)
    if operation == 'fusion':
        expected = fuse().copy()
    elif operation == 'registration':
        expected = resolve_frame(register()[1]).copy()
    else:
        expected = pixels[:, ::-1]
    default, alternative = _fake_volumes(monkeypatch, tmp_path, 0, 32 * 1024 ** 2)
    if operation == 'retained':
        actual = resolve_frame(retained)
        np.testing.assert_array_equal(actual, expected)
    else:
        with frame_storage.scratch_directory():
            actual = fuse() if operation == 'fusion' else resolve_frame(register()[1])
            np.testing.assert_array_equal(actual, expected)
            assert os.path.commonpath((str(alternative), frame_storage.storage_directory())) == str(alternative)
    assert not list(default.iterdir())


def test_bounded_workspace_hint_precedes_ram_only_fallback(monkeypatch, tmp_path):
    default, alternative = _fake_volumes(monkeypatch, tmp_path, 3, 1000)
    with frame_storage.scratch_directory():
        directory = frame_storage.select_scratch(0, preferred_size=10000, working_size=500)
        assert os.path.dirname(directory) == str(alternative)
        result = frame_storage.create_frame((2, 2), np.uint16)
        result[:] = [[7, 257], [1001, 65535]]
        np.testing.assert_array_equal(result, [[7, 257], [1001, 65535]])
    assert not list(default.iterdir())


def test_optional_estimate_prefers_larger_volume_without_rejecting_explicit(monkeypatch, tmp_path):
    default, alternative = _fake_volumes(monkeypatch, tmp_path, 100, 1000)
    with frame_storage.scratch_directory():
        chosen = frame_storage.select_scratch(50, preferred_size=500)
        assert os.path.dirname(chosen) == str(alternative)
    frame_storage.configure_storage_directory(default)
    with frame_storage.scratch_directory():
        assert frame_storage.select_scratch(50, preferred_size=500) == str(default)


def test_independent_scopes_keep_their_own_directory(monkeypatch, tmp_path):
    import concurrent.futures
    import threading
    default, alternative = _fake_volumes(monkeypatch, tmp_path, 100, 1000)
    # A timeout turns an exception in one job into BrokenBarrierError in the
    # other, instead of a hang that the executor would wait on forever.
    barrier = threading.Barrier(2, timeout=30)
    def job(size):
        with frame_storage.scratch_directory():
            selected = frame_storage.select_scratch(size)
            barrier.wait()
            mapped = frame_storage.store_frame(np.zeros(2, np.uint8), prefer_ram=False)
            assert os.path.dirname(mapped.filename) == selected
            return selected
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(job, (50, 200)))
    assert first == str(default) and os.path.dirname(second) == str(alternative)


def test_unavailable_saved_directory_falls_back_and_recovers(monkeypatch, tmp_path):
    from core import resource_policy
    from core.source_frames import materialize_frames

    configured = tmp_path / 'missing'
    fallback = tmp_path / 'system-temp'
    fallback.mkdir()
    monkeypatch.setattr(frame_storage.tempfile, 'gettempdir', lambda: str(fallback))
    monkeypatch.setattr(resource_policy, 'frame_ram_budget', lambda snapshot: 0)
    notices = []
    frame_storage.set_storage_fallback_notifier(lambda *args: notices.append(args))
    frame_storage.restore_storage_directory(configured)
    assert frame_storage.storage_directory() == str(fallback)

    with frame_storage.scratch_directory():
        frame_storage.select_scratch(4)
        frames = materialize_frames([np.arange(4, dtype=np.uint8).reshape(2, 2)], prefer_ram=False)
        assert os.path.dirname(frames[0].filename) == str(fallback)
        np.testing.assert_array_equal(frames[0], [[0, 1], [2, 3]])
    assert frame_storage._storage_directory == str(configured)
    assert len(notices) == 1 and notices[0][0] == str(configured)
    del frames
    gc.collect()

    configured.mkdir()
    with frame_storage.scratch_directory():
        assert frame_storage.select_scratch(4) == str(configured)
    assert frame_storage._storage_directory == str(configured)



def test_full_explicit_directory_does_not_switch_volumes(monkeypatch, tmp_path):
    import errno

    configured = tmp_path / 'configured'
    fallback = tmp_path / 'fallback'
    configured.mkdir()
    fallback.mkdir()
    frame_storage.restore_storage_directory(configured)
    monkeypatch.setattr(frame_storage.tempfile, 'gettempdir', lambda: str(fallback))
    original = frame_storage.tempfile.mkstemp
    def full(*args, **kwargs):
        if kwargs.get('dir') == str(configured):
            raise OSError(errno.ENOSPC, 'disk full')
        return original(*args, **kwargs)
    monkeypatch.setattr(frame_storage.tempfile, 'mkstemp', full)
    with frame_storage.scratch_directory():
        with pytest.raises(frame_storage.ScratchSpaceError) as error:
            frame_storage.select_scratch(4)
    assert error.value.directory == str(configured)
    assert not list(fallback.iterdir())


def test_preview_keeps_alternative_directory_alive(monkeypatch, tmp_path):
    from core.source_frames import store_preview
    _, alternative = _fake_volumes(monkeypatch, tmp_path, 0, 10000)
    with frame_storage.scratch_directory():
        directory = frame_storage.select_scratch(4)
        preview = store_preview(np.zeros((2, 2), np.uint8))
    assert os.path.dirname(directory) == str(alternative)
    assert os.path.exists(preview.path)
    del preview
    gc.collect()
    assert not os.path.exists(directory)


@pytest.mark.parametrize('source', ['files', 'folder'])
@pytest.mark.parametrize('explicit', [False, True])
def test_import_preview_respects_available_temporary_storage(monkeypatch, tmp_path, source, explicit):
    from core.image_loader import ImageStackLoader
    from core.source_frames import display_image
    from utils.image_utils import imwrite_auto

    images_dir = tmp_path / 'images'
    images_dir.mkdir()
    path = images_dir / 'small.png'
    pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    assert imwrite_auto(str(path), pixels, exif=b'')
    default, alternative = _fake_volumes(monkeypatch, tmp_path, 0, 10000)
    if explicit:
        frame_storage.configure_storage_directory(default)
    loader = ImageStackLoader()
    if source == 'folder':
        ok, message, frames, _ = loader.load_from_folder(str(images_dir))
    else:
        ok, message, frames, _ = loader.load_from_filepaths([str(path)])
    if explicit:
        assert not ok and str(default) in message
        assert not list(alternative.iterdir())
        return
    assert ok, message
    preview_path = frames[0].preview.path
    directory = os.path.dirname(preview_path)
    assert os.path.dirname(directory) == str(alternative)
    np.testing.assert_array_equal(display_image(frames[0]), pixels)
    assert not list(default.iterdir())
    del frames, loader
    gc.collect()
    assert not os.path.exists(preview_path)
    assert not os.path.exists(directory)


def test_headless_error_keeps_space_information(monkeypatch, tmp_path):
    import builtins
    original = builtins.__import__
    def without_qt(name, *args, **kwargs):
        if name == 'locales':
            raise ImportError('Qt unavailable')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_qt)
    message = str(frame_storage.ScratchSpaceError(str(tmp_path), 2048, 1024))
    assert str(tmp_path) in message and '2.00 KiB' in message and '1.00 KiB' in message
    assert 'Advanced settings' in message


def test_unsafe_default_is_not_selected_automatically(monkeypatch, tmp_path):
    default, alternative = _fake_volumes(monkeypatch, tmp_path, 10000, 10000)
    monkeypatch.setattr(frame_storage, '_default_is_local', lambda path: False)
    with frame_storage.scratch_directory():
        chosen = frame_storage.select_scratch(16)
        assert os.path.dirname(chosen) == str(alternative)
    assert not list(default.iterdir())


def test_failed_preallocation_removes_partial_file(monkeypatch, tmp_path):
    import errno
    frame_storage.configure_storage_directory(tmp_path)
    def full(*args):
        raise OSError(errno.ENOSPC, 'disk full')
    monkeypatch.setattr(frame_storage.os, 'posix_fallocate', full, raising=False)
    with pytest.raises(frame_storage.ScratchSpaceError):
        frame_storage.create_frame((4, 4), np.uint8)
    assert not list(tmp_path.iterdir())


def test_cancelled_materialization_cleans_alternative(monkeypatch, tmp_path):
    from core import source_frames
    from core.registration import RegistrationCancelled
    _, alternative = _fake_volumes(monkeypatch, tmp_path, 0, 10000)
    original = source_frames.store_frame
    cancelled = False
    def store(image, **kwargs):
        nonlocal cancelled
        result = original(image, **kwargs)
        cancelled = True
        return result
    monkeypatch.setattr(source_frames, 'store_frame', store)
    with frame_storage.scratch_directory():
        directory = frame_storage.select_scratch(100)
        try:
            source_frames.materialize_frames([np.zeros(10, np.uint8)] * 2,
                                             should_cancel=lambda: cancelled, prefer_ram=False)
        except RegistrationCancelled:
            pass
        else:
            raise AssertionError('cancellation did not stop materialization')
    gc.collect()
    assert os.path.dirname(directory) == str(alternative)
    assert not os.path.exists(directory)


def test_linux_recognizes_fixed_mounts_and_rejects_network_or_removable(monkeypatch):
    import builtins
    import io
    import posixpath

    mounts = (
        '1 0 8:1 / / rw - ext4 /dev/sda1 rw\n'
        '2 0 8:2 / /media/Drive\\040One rw - ext4 /dev/nvme0n1p1 rw\n'
        '3 0 8:3 / /media/USB rw - vfat /dev/sdb1 rw\n'
        '4 0 0:40 / /mnt/network rw - nfs server:/share rw\n'
        '5 0 0:41 / /run/temp rw - tmpfs tmpfs rw\n'
    )
    devices = {'/sys/dev/block/8:1': '/sys/devices/sda/sda1',
               '/sys/dev/block/8:2': '/sys/devices/nvme0n1/nvme0n1p1',
               '/sys/dev/block/8:3': '/sys/devices/sdb/sdb1'}
    flags = {'/sys/devices/sda/removable': '0',
             '/sys/devices/nvme0n1/removable': '0',
             '/sys/devices/sdb/removable': '1'}
    def realpath(path):
        return devices.get(path, posixpath.normpath(path))
    def exists(path):
        return path in devices.values() or path in flags
    path_api = SimpleNamespace(realpath=realpath, exists=exists, dirname=posixpath.dirname,
                               commonpath=posixpath.commonpath, join=posixpath.join)
    monkeypatch.setattr(frame_storage, 'os', SimpleNamespace(path=path_api))
    monkeypatch.setattr(frame_storage, 'sys', SimpleNamespace(platform='linux'))
    original_open = builtins.open
    def open_mount(path, *args, **kwargs):
        if path == '/proc/self/mountinfo':
            return io.StringIO(mounts)
        if path in flags:
            return io.StringIO(flags[path])
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', open_mount)

    assert frame_storage._local_mounts() == ['/', '/media/Drive One']
    assert frame_storage._default_is_local('/media/Drive One/tmp')
    assert frame_storage._default_is_local('/run/temp')
    assert not frame_storage._default_is_local('/media/USB/tmp')
    assert not frame_storage._default_is_local('/mnt/network/tmp')


def test_macos_recognizes_internal_fixed_volumes_only(monkeypatch):
    import plistlib
    import subprocess

    disks = {'disk0s1': {'Internal': True, 'RemovableMedia': False,
                         'MountPoint': '/System/Volumes/Data'},
             'disk2s1': {'Internal': False, 'RemovableMedia': True,
                         'MountPoint': '/Volumes/USB'},
             'disk3s1': {'Internal': False, 'RemovableMedia': False,
                         'MountPoint': '/Volumes/Network'}}
    paths = {'/System/Volumes/Data/tmp': '/dev/disk0s1',
             '/private/var/folders/app/tmp': '/dev/disk0s1',
             '/Volumes/USB/tmp': '/dev/disk2s1',
             '/Volumes/Network/tmp': '/dev/disk3s1'}
    def diskutil(command, timeout, text=False):
        if command[0] == '/bin/df':
            return 'Filesystem 512-blocks Used Available Capacity Mounted on\n' + paths[command[-1]] + ' 100 1 99 1% /\n'
        if command[1] == 'list':
            return plistlib.dumps({'AllDisks': list(disks)})
        value = command[-1].removeprefix('/dev/')
        if value not in disks:
            raise subprocess.CalledProcessError(1, command)
        return plistlib.dumps(disks[value])
    monkeypatch.setattr(frame_storage, 'sys', SimpleNamespace(platform='darwin'))
    monkeypatch.setattr(frame_storage, 'subprocess', SimpleNamespace(
        check_output=diskutil, SubprocessError=subprocess.SubprocessError))

    assert frame_storage._local_mounts() == ['/System/Volumes/Data']
    assert frame_storage._default_is_local('/System/Volumes/Data/tmp')
    assert frame_storage._default_is_local('/private/var/folders/app/tmp')
    assert not frame_storage._default_is_local('/Volumes/USB/tmp')
    assert not frame_storage._default_is_local('/Volumes/Network/tmp')


def test_collecting_snapshot_during_admission_reclaims_budget_without_deadlock(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    from textwrap import dedent

    script = dedent('''
        import gc
        import sys
        import weakref
        import numpy as np
        from core import frame_storage as storage, resource_policy as policy

        snapshot = policy.ResourceSnapshot(available_ram=8 * 1024**3,
                                           total_ram=16 * 1024**3)
        policy.probe_memory = lambda: snapshot
        policy.FRAME_RAM_BUDGET_BYTES = 64
        storage.configure_storage_directory(sys.argv[1])
        gc.disable()
        first = storage.create_transient_frame((8, 8), np.uint8)
        assert not isinstance(first, np.memmap)
        reference = weakref.ref(first)
        cycle = [first]
        cycle.append(cycle)
        del first, cycle

        def collect_snapshot():
            gc.collect()
            assert reference() is None
            return snapshot

        policy.probe_memory = collect_snapshot
        saved = storage.store_frame(np.full((8, 8), 37, np.uint8))
        assert not isinstance(saved, np.memmap)
        view = saved[1:3, 1:3]
        del saved
        spilled = storage.create_transient_frame((1,), np.uint8)
        assert isinstance(spilled, np.memmap)
        np.testing.assert_array_equal(view, np.full((2, 2), 37, np.uint8))
        del view, spilled
        gc.collect()
        reclaimed = storage.create_transient_frame((8, 8), np.uint8)
        assert not isinstance(reclaimed, np.memmap)
    ''')
    completed = subprocess.run(
        [sys.executable, '-c', script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
