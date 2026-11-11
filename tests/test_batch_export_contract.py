"""Behavioral boundaries for directory artifacts and stable image exports."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QLabel, QSlider, QWidget

from controllers.export_manager import ExportManager, _read_image_any
from controllers.output_manager import OutputManager, OutputRecord, publish_aligned_stack
from core.image_loader import ImageStackLoader
from core.workers import BatchWorker, GifSaverWorker
from utils.image_utils import imwrite_auto, imwrite_multi_tiff
from widgets.output_list import OutputListWidget

from widgets.magnifier_label import MagnifierLabel

@pytest.fixture(autouse=True)
def fail_on_export_error(monkeypatch):
    def unexpected_error(_parent, *details):
        pytest.fail(" | ".join(map(str, details)))
    monkeypatch.setattr("controllers.output_manager.show_error_box", unexpected_error)


class Labels:
    def snapshot(self):
        return self

    def prepare_bgr_image(self, target, image, index, **kwargs):
        return image


class Status:
    def __init__(self):
        self.messages = []

    def showMessage(self, message, *args):
        self.messages.append(message)


def host(qapp):
    window = QWidget()
    window.output_list = OutputListWidget(window)
    window.output_list.set_window(window)
    window.output_label = QLabel(window)
    window.lbl_result_info = QLabel(window)
    window.result_slider = QSlider(window)
    window.result_control_bar = QWidget(window)
    window.fusion_results = []
    window.fusion_result = None
    window.registration_results = []
    window.image_filenames = []
    window.label_manager = Labels()
    window.current_folder_path = ''
    window.scale_bar_enabled = False
    window.status = Status()
    window.statusBar = lambda: window.status
    window.output_manager = OutputManager(window)
    window.export_manager = ExportManager(window)
    return window


def make_stack(path, names=('img1.png', 'img2.png'), dtype=np.uint16):
    path.mkdir(parents=True)
    for index, name in enumerate(names):
        pixels = np.full((12, 16, 3), 1000 + index * 1000 if dtype == np.uint16 else 50 + index, dtype)
        assert imwrite_auto(str(path / name), pixels, exif=b'')
    return path


def run_batch(folders, destination, **overrides):
    settings = {'format': 'png', 'fusion_method': 'none', 'reg_methods': [], 'save_aligned': True}
    settings.update(overrides)
    worker = BatchWorker([str(path) for path in folders], 'custom', str(destination), settings, use_gpu=False)
    result = {}
    worker.finished.connect(result.update)
    worker.error.connect(lambda message: result.update(error=message))
    worker.run()
    assert 'error' not in result, result
    return worker, result


def test_align_only_is_one_directory_per_stack_with_original_names(qapp, tmp_path):
    first = make_stack(tmp_path / 'a' / 'stack')
    second = make_stack(tmp_path / 'b' / 'stack')
    originals = {path: path.read_bytes() for folder in (first, second) for path in folder.iterdir()}
    worker, result = run_batch([first, second], tmp_path / 'out')
    assert result['counts']['success'] == 2
    assert result['outputs'] == []
    assert [Path(item['path']).name for item in result['aligned_stacks']] == ['stack_aligned', 'stack_aligned_2']
    for artifact in result['aligned_stacks']:
        directory = Path(artifact['path'])
        assert sorted(path.name for path in directory.iterdir()) == ['img1.png', 'img2.png']
        assert _read_image_any(str(directory / 'img1.png')).dtype == np.uint16
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert not list(tmp_path.rglob('*_registered*'))
    assert not list(tmp_path.rglob('.openfocus_cache'))
    assert worker.written_paths == []


def test_leftover_aligned_copies_are_not_reloaded_as_frames(qapp, tmp_path):
    """A folder that already received `*_registered*` copies must keep its
    original frame count: older builds wrote them beside the sources, and
    reloading the folder then fused every frame three times over."""
    source = make_stack(tmp_path / 'stack')
    for original in sorted(source.iterdir()):
        stem = original.stem
        (source / f'{stem}_registered.png').write_bytes(original.read_bytes())
        (source / f'{stem}_registered_2.png').write_bytes(original.read_bytes())
        (source / f'{stem}_registered_registered.png').write_bytes(original.read_bytes())

    loader = ImageStackLoader()
    ok, message, images, names = loader.load_from_folder(str(source))
    assert ok and len(images) == 2, names
    assert names == ['img1.png', 'img2.png']
    assert loader.skipped_derived_count == 6
    assert 'skipped' in message


def test_derived_copy_names_keep_real_user_files():
    from utils.export_paths import derived_aligned_source
    assert derived_aligned_source('01_registered.jpg') == '01'
    assert derived_aligned_source('01_registered_2.jpg') == '01'
    assert derived_aligned_source('01_registered_registered.jpg') == '01'
    assert derived_aligned_source('Railroad_model01.JPG') == ''
    assert derived_aligned_source('registered.jpg') == ''


def test_batch_output_panel_lists_folders_only_without_fusion(qapp, tmp_path):
    """Fusion runs show their fused result alone; align-only runs show the
    stack folders, labelled with the directory that exists on disk."""
    from controllers.batch_manager import BatchManager

    window = host(qapp)
    window.slider_smooth = window.combo_fusion = window.combo_align = window.btn_reset = None
    manager = BatchManager(window)
    shown = []
    manager._show_message = lambda **kwargs: shown.append(kwargs)
    # The fixture window has no result canvas; only the list contract matters.
    window.output_manager.display_specific_fusion_result = lambda image: None
    folders = [{'path': str(tmp_path / 'stack_aligned'), 'name': 'stack', 'files': []}]
    (tmp_path / 'stack_aligned').mkdir()

    manager._handle_finished({'success': 1, 'total': 1, 'outputs': [], 'aligned_stacks': folders})
    assert window.output_list.count() == 1
    assert window.output_list.item(0).text() == 'stack_aligned'

    window.output_list.clear()
    window.fusion_results.clear()
    fused = tmp_path / 'stack.png'
    assert imwrite_auto(str(fused), np.full((8, 8, 3), 60, np.uint8))
    manager._handle_finished({'success': 1, 'total': 1, 'outputs': [str(fused)],
                              'aligned_stacks': folders})
    assert window.output_list.count() == 1
    assert window.output_list.item(0).text() == 'stack.png'


def test_explicit_encoding_and_duplicate_stems_do_not_overwrite(qapp, tmp_path):
    source = make_stack(tmp_path / 'stack', ('frame.png', 'frame.tif'))
    _, result = run_batch([source], tmp_path / 'out', format='jpg')
    files = result['aligned_stacks'][0]['files']
    assert [Path(path).name for path in files] == ['frame.jpg', 'frame_2.jpg']
    assert all(_read_image_any(path).dtype == np.uint8 for path in files)


def test_split_loader_keeps_tiff_pages_and_export_names(qapp, tmp_path):
    source = tmp_path / 'stack'
    source.mkdir()
    pages = [np.full((10, 12, 3), value, np.uint16) for value in (1000, 2000, 3000)]
    assert imwrite_multi_tiff(str(source / 'volume.tif'), pages)
    loader = ImageStackLoader()
    ok, _, entries, names = loader.load_images_with_timestamps(str(source))
    assert ok and len(entries) == 3
    assert names == ['volume_page_001.tif', 'volume_page_002.tif', 'volume_page_003.tif']
    assert all(entry[0] == str(source / 'volume.tif') for entry in entries)
    result = {}
    worker = BatchWorker([], 'custom', str(tmp_path / 'out'),
        {'format': 'tiff', 'fusion_method': 'none', 'frozen_stacks': loader.split_by_count(entries, 2)},
        import_mode='single_folder', single_folder_images_with_times=entries)
    worker.finished.connect(result.update)
    worker.run()
    assert result['counts']['success'] == 2
    assert [Path(path).name for stack in result['aligned_stacks'] for path in stack['files']] == [
        'volume_page_001.tiff', 'volume_page_002.tiff', 'volume_page_003.tiff']


def test_partial_writes_and_cancel_preserve_only_committed_files(qapp, tmp_path, monkeypatch):
    import utils.image_utils as image_utils
    source = make_stack(tmp_path / 'stack')
    original_write = image_utils.imwrite_auto
    monkeypatch.setattr(image_utils, 'imwrite_auto', lambda path, *args, **kwargs:
                        False if Path(path).name == 'img2.png' else original_write(path, *args, **kwargs))
    _, result = run_batch([source], tmp_path / 'partial')
    assert result['counts']['partial'] == 1 and result['success'] == 0
    assert len(result['tasks'][0]['artifacts']) == 1
    assert result['tasks'][0]['failures'][0]['path'].endswith('img2.png')
    monkeypatch.setattr(image_utils, 'imwrite_auto', original_write)
    worker = BatchWorker([str(source), str(source)], 'custom', str(tmp_path / 'cancel'),
                         {'format': 'png', 'fusion_method': 'none'})
    save = worker._save_image

    def cancel_after_first(*args, **kwargs):
        ok = save(*args, **kwargs)
        worker.cancel()
        return ok

    monkeypatch.setattr(worker, '_save_image', cancel_after_first)
    result = {}
    worker.finished.connect(result.update)
    worker.run()
    assert result['counts']['cancelled'] == 1
    assert result['counts']['not_started'] == 1
    assert result['counts']['failed'] == 0
    assert len(result['tasks'][0]['artifacts']) == 1
    assert len(list((tmp_path / 'cancel').rglob('*.png'))) == 1


def test_both_batch_entries_export_same_scaled_16bit_fusion(qapp, tmp_path):
    source = make_stack(tmp_path / 'stack')
    settings = {'format': 'png', 'fusion_method': 'guided_filter', 'reg_methods': [],
                'load_scale_factor': .5, 'fusion_params': {'kernel_size': 3},
                'runtime_settings': {'base_frame': 'last', 'sort_mode': 'desc', 'exposure_match': False}}
    normal = BatchWorker([str(source)], 'custom', str(tmp_path / 'normal'), settings,
                         exposure_match=True, use_gpu=False, thread_count=1)
    normal_result = {}
    normal.finished.connect(normal_result.update)
    normal.run()
    loader = ImageStackLoader()
    ok, _, entries, _ = loader.load_images_with_timestamps(str(source), scale_factor=.5)
    assert ok
    split = BatchWorker([], 'custom', str(tmp_path / 'split'), settings,
                        import_mode='single_folder', split_method='count', split_param=2,
                        single_folder_images_with_times=entries, exposure_match=True,
                        use_gpu=False, thread_count=1)
    split_result = {}
    split.finished.connect(split_result.update)
    split.run()
    assert normal_result['success'] == split_result['success'] == 1
    normal_pixels = _read_image_any(normal_result['outputs'][0])
    split_pixels = _read_image_any(split_result['outputs'][0])
    assert normal_pixels.dtype == split_pixels.dtype == np.uint16
    assert normal_pixels.shape == split_pixels.shape == (6, 8, 3)
    np.testing.assert_array_equal(normal_pixels, split_pixels)


def test_aligned_publication_is_memory_only_and_exports_once(qapp, tmp_path):
    window = host(qapp)
    window.current_folder_path = str(tmp_path)
    frames = [np.full((10, 12, 3), 2000, np.uint16)] * 2
    publish_aligned_stack(window, frames, filenames=['original.tif', 'original.png'])
    assert not list(tmp_path.iterdir())
    assert window.output_list.count() == 1 and window.fusion_results == [None]
    record = window.output_manager.record_for_row(0)
    assert record.image is None and record.kind == 'aligned'
    directory = window.export_manager.save_aligned_frames_as(frames, record.aligned_names,
        'stack', destination=str(tmp_path), extension='tiff', record=record)
    assert sorted(path.name for path in Path(directory).iterdir()) == ['original.tiff', 'original_2.tiff']
    assert _read_image_any(str(Path(directory) / 'original.tiff')).dtype == np.uint16


def test_save_as_freezes_output_identity_before_dialog(qapp, tmp_path, monkeypatch):
    import controllers.output_manager as module
    window = host(qapp)
    a = OutputRecord.create(np.full((8, 8, 3), 1200, np.uint16), 'A')
    b = OutputRecord.create(np.full((8, 8, 3), 3400, np.uint16), 'B')
    item_a = window.output_manager._insert_output(a.image, a.label, a)
    window.output_manager._insert_output(b.image, b.label, b)

    def pick(*args, **kwargs):
        window.output_list.setCurrentRow(0)
        return str(tmp_path / 'chosen'), 'TIFF Files (*.tif *.tiff)'

    monkeypatch.setattr(module.FileDialog, 'getSaveFileName', pick)
    window.output_manager.save_output_image_as(item_a)
    image = _read_image_any(str(tmp_path / 'chosen.tif'))
    assert image.dtype == np.uint16 and int(image[0, 0, 0]) == 1200
    assert a.exported and not b.exported


def test_export_all_keeps_folder_artifact_and_baked_pixels(qapp, tmp_path, monkeypatch):
    import controllers.output_manager as module
    window = host(qapp)
    image = np.full((8, 8, 3), 80, np.uint8)
    record = OutputRecord.create(image, 'historical name', kind='batch', exported=True)
    record.annotations_baked = True
    window.output_manager._insert_output(image, record.label, record)
    source = make_stack(tmp_path / 'source', dtype=np.uint8)
    files = [str(path) for path in sorted(source.iterdir())]
    window.output_manager.add_batch_aligned_stacks([{'path': str(source), 'files': files, 'name': 'source'}])
    destination = tmp_path / 'export'
    destination.mkdir()
    (destination / 'historical_name.png').write_bytes(b'keep')
    monkeypatch.setattr(module.FileDialog, 'getExistingDirectory', lambda *args, **kwargs: str(destination))
    window.label_manager.prepare_bgr_image = lambda *args, **kwargs: pytest.fail('Baked output annotated twice')
    window.output_manager.export_all_results()
    assert (destination / 'historical_name.png').read_bytes() == b'keep'
    assert _read_image_any(str(destination / 'historical_name_2.png'))[0, 0, 0] == 80
    assert (destination / 'source_aligned' / 'img1.png').read_bytes() == (source / 'img1.png').read_bytes()
    assert window.output_list.count() == 2


def test_gif_cancel_does_not_replace_previous_file(qapp, tmp_path, monkeypatch):
    from PIL import GifImagePlugin
    destination = tmp_path / 'existing.gif'
    destination.write_bytes(b'previous')
    worker = GifSaverWorker([np.zeros((8, 8, 3), np.uint8)], str(destination), .1, Labels(), 'registered')

    encode = GifImagePlugin.getdata
    def cancel_after_encoding(*args, **kwargs):
        data = encode(*args, **kwargs)
        worker.cancel()
        return data

    monkeypatch.setattr(GifImagePlugin, 'getdata', cancel_after_encoding)
    outcomes = []
    worker.finished_signal.connect(lambda ok, message: outcomes.append((ok, message)))
    worker.run()
    assert destination.read_bytes() == b'previous'
    assert outcomes == [(False, 'CANCELLED')]
    assert list(tmp_path.iterdir()) == [destination]


def test_batch_cancellation_keeps_controls_until_worker_finishes(qapp):
    from controllers.batch_manager import BatchManager
    window = host(qapp)
    manager = BatchManager(window)
    calls = []
    manager._worker = SimpleNamespace(cancel=lambda: calls.append('cancel'))
    manager._restore_controls = lambda: calls.append('restore')
    manager._cancel_running_batch()
    assert calls == ['cancel']


def test_unknown_export_extension_is_not_silently_reencoded():
    with pytest.raises(ValueError):
        ExportManager(None).normalize_export_path('image.jp2')


@pytest.mark.parametrize('suffix', ['.jpeg', '.jpe', '.jfif', '.JPEG', '.PNG'])
def test_export_preserves_confirmed_filename_and_writes_that_file(tmp_path, suffix):
    destination = tmp_path / ('image' + suffix)
    destination.write_bytes(b'old contents')
    other = tmp_path / ('image.jpg' if suffix.lower() != '.png' else 'other.png')
    other.write_bytes(b'untouched')

    path = ExportManager(None).normalize_export_path(str(destination))
    assert path == str(destination)
    assert imwrite_auto(path, np.zeros((8, 8, 3), np.uint8))
    assert cv2.imdecode(np.frombuffer(destination.read_bytes(), np.uint8), cv2.IMREAD_COLOR) is not None
    assert other.read_bytes() == b'untouched'


def test_source_paths_cannot_be_export_targets(qapp, tmp_path):
    window = host(qapp)
    source = make_stack(tmp_path / 'source')
    path = source / 'img1.png'
    original = path.read_bytes()
    window.image_source_paths = [str(path)]
    with pytest.raises(ValueError):
        window.export_manager.normalize_export_path(str(path))
    assert path.read_bytes() == original


def test_failed_cache_write_keeps_previous_complete_cache(qapp, tmp_path, monkeypatch):
    from utils import align_cache
    source = make_stack(tmp_path / 'source')
    frames = [np.full((8, 8, 3), 45, np.uint8)] * 2
    names = ['img1.png', 'img2.png']
    align_cache.save_aligned(str(source), names, frames, ("homography",), 1024)
    monkeypatch.setattr(align_cache, 'imwrite_auto', lambda *args, **kwargs: False)
    align_cache.save_aligned(str(source), names, frames, ("ecc",), 1024)
    assert align_cache.load_aligned(str(source), names, ("homography",), 1024) is not None
    assert align_cache.load_aligned(str(source), names, ("ecc",), 1024) is None
    assert not list((source / '.openfocus_cache').glob('.pending-*'))


def test_batch_scale_scope_and_downsample_calibration(qapp):
    worker = BatchWorker([], 'custom', '', {
        'load_scale_factor': .5,
        'scale_bar': {'manual_px_um': .4, 'target': 'registered',
                      'font_size': 19, 'custom_x': .3, 'custom_y': .8}})
    assert worker._stack_scale_cfg('output') is None
    cfg = worker._stack_scale_cfg('registered')
    assert cfg['px_um'] == .8 and cfg['font_size'] == 19
    assert (cfg['custom_x'], cfg['custom_y']) == (.3, .8)


def test_cancelled_loader_never_publishes_decoded_frames(qapp, tmp_path):
    from core.workers import StackLoadWorker
    source = make_stack(tmp_path / 'source')
    worker = StackLoadWorker(ImageStackLoader(), folder=str(source))
    results = []
    worker.finished_load.connect(lambda *args: results.append(args))
    worker.request_cancel()
    worker.run()
    assert results == [(False, 'CANCELLED', [], [])]


def test_export_annotation_edits_and_undo_restore_saved_state(qapp, tmp_path, monkeypatch):
    from controllers.label_manager import LabelManager
    from utils.label_model import Label
    import controllers.output_manager as module

    window = host(qapp)
    window.label_manager = LabelManager(window)
    record = OutputRecord.create(np.full((64, 96, 3), 90, np.uint8), 'result', kind='compare')
    item = window.output_manager._insert_output(record.image, record.label, record)
    label = Label.from_dict({'target': 'output', 'output_id': record.output_id,
                             'format': 'A', 'font_size': 12, 'x_location': 5,
                             'y_location': 24, 'enabled': True})
    window.label_manager.labels.add(label)
    monkeypatch.setattr(module.FileDialog, 'getSaveFileName',
                        lambda *args, **kwargs: (str(tmp_path / 'result.png'), 'PNG Files (*.png)'))
    window.output_manager.save_output_image_as(item)
    assert (tmp_path / 'result.png').exists()
    saved_bytes = (tmp_path / 'result.png').read_bytes()
    assert record.exported and not record.changed_since_export and not record.protected

    label.config.format = 'B'
    window.output_manager.refresh_annotation_export_state()
    assert record.changed_since_export and record.protected
    label.config.format = 'A'
    window.output_manager.refresh_annotation_export_state()
    assert not record.changed_since_export and not record.protected

    window.label_manager.labels.add(Label.from_dict({
        'target': 'output', 'output_id': 'unrelated-output', 'format': 'Other'}))
    window.output_manager.refresh_annotation_export_state()
    assert not record.changed_since_export

    record.scale_bar_config = {'px_um': .5, 'position': 'bottom-right', 'color': 'white'}
    window.output_manager.refresh_annotation_export_state()
    assert record.changed_since_export and record.protected
    record.scale_bar_config = None
    window.output_manager.refresh_annotation_export_state()
    assert not record.changed_since_export and not record.protected
    assert (tmp_path / 'result.png').read_bytes() == saved_bytes


def test_baked_output_stays_saved_when_workspace_annotations_change(qapp, tmp_path, monkeypatch):
    from controllers.label_manager import LabelManager
    from utils.label_model import Label
    import controllers.output_manager as module
    window = host(qapp)
    window.label_manager = LabelManager(window)
    record = OutputRecord.create(np.full((16, 16, 3), 80, np.uint8), 'batch', kind='batch', exported=True)
    record.annotations_baked = True
    item = window.output_manager._insert_output(record.image, record.label, record)
    monkeypatch.setattr(module.FileDialog, 'getSaveFileName',
                        lambda *args, **kwargs: (str(tmp_path / 'batch.png'), 'PNG Files (*.png)'))
    window.output_manager.save_output_image_as(item)
    window.label_manager.labels.add(Label.from_dict({'target': 'output', 'format': 'New'}))
    record.scale_bar_config = {'px_um': 2}
    window.output_manager.refresh_annotation_export_state()
    assert record.exported and not record.changed_since_export
    np.testing.assert_array_equal(_read_image_any(str(tmp_path / 'batch.png')), record.image)


@pytest.mark.parametrize('reverse', [False, True])
def test_grouped_batch_registers_each_round_without_loading_stack(qapp, tmp_path, monkeypatch, reverse):
    import core.grouped_fusion as grouped
    import core.workers as workers

    source = make_stack(tmp_path / 'stack', ('img10.png', 'img2.png', 'img1.png'), np.uint8)
    registered = []
    stages = []

    class Registration:
        def __init__(self, **kwargs):
            pass

        def process(self, images, **kwargs):
            assert not kwargs['should_cancel']()
            registered.append(len(images))
            return images

    def grouped_fuse(paths, size, overlap, fuse_group, **kwargs):
        expected = ['img1.png', 'img2.png', 'img10.png']
        assert [Path(path).name for path in paths] == (expected[::-1] if reverse else expected)
        assert (size, overlap) == (3, 1)
        image = np.full((12, 16, 3), 50, np.uint8)
        first = fuse_group([image, image])
        result = fuse_group([first, first])
        kwargs['progress'](2, 1, 1)
        return result

    monkeypatch.setattr(workers, 'ImageRegistration', Registration)
    monkeypatch.setattr(grouped, 'grouped_fuse', grouped_fuse)
    monkeypatch.setattr(ImageStackLoader, 'load_from_folder', lambda *a, **kw: pytest.fail('preloaded stack'))
    def fuse_whole(self, fusion, method, images, params):
        assert not fusion.tile_enabled
        return images[0]
    monkeypatch.setattr(BatchWorker, '_fuse_one_stack', fuse_whole)
    settings = {'format': 'png', 'fusion_method': 'openfocus_a', 'alignment': 'openfocus',
                'save_aligned': False, 'group_size': 3, 'group_overlap': 1,
                'runtime_settings': {'sort_mode': 'desc' if reverse else 'asc',
                                     'tile_enabled': True, 'tile_threshold': 8}}
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out'), settings, use_gpu=False)
    result = {}
    worker.progress_updated.connect(lambda done, total, text: stages.append(text))
    worker.finished.connect(result.update)
    worker.run()
    assert result['counts']['success'] == 1
    assert registered == [2, 2]
    assert result['aligned_stacks'] == []
    assert len(result['outputs']) == 1
    assert any('2' in text and '1 / 1' in text for text in stages)


def test_grouped_batch_real_registration_keeps_singleton_coordinates(qapp, tmp_path):
    source = tmp_path / 'registered'
    source.mkdir()
    y, x = np.indices((64, 68))
    frame = np.stack([x, y, x + y], axis=-1).astype(np.uint16) * 257
    for index in range(5):
        assert cv2.imwrite(str(source / f'{index}.png'), frame)
    settings = {'format': 'png', 'fusion_method': 'openfocus_a', 'alignment': 'openfocus',
                'group_size': 2, 'group_overlap': 0}
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out'), settings, use_gpu=False)
    result = {}
    worker.finished.connect(result.update)
    worker.run()
    assert result['counts']['success'] == 1
    saved = cv2.imread(result['outputs'][0], cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(saved, frame)


def test_grouped_batch_rejects_split_and_aligned_export(qapp, tmp_path):
    settings = {'fusion_method': 'openfocus_a', 'group_size': 2, 'group_overlap': 0,
                'save_aligned': True}
    worker = BatchWorker([], 'custom', str(tmp_path), settings, import_mode='single_folder')
    errors = []
    worker.error.connect(errors.append)
    worker.run()
    assert errors and 'split mode' in errors[0]

    source = make_stack(tmp_path / 'stack', dtype=np.uint8)
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out'), settings)
    result = {}
    worker.finished.connect(result.update)
    worker.run()
    assert result['counts']['failed'] == 1
    assert 'aligned' in result['failed'][0]


def test_batch_thumbnail_decodes_only_first_readable_image(qapp, tmp_path, monkeypatch):
    import core.image_loader as loader
    from core.workers import BatchImportWorker

    source = make_stack(tmp_path / 'stack', ('img1.png', 'img2.png', 'img3.png'), np.uint8)
    decoded = []
    original = loader.decode_image
    def decode(path):
        decoded.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(loader, 'decode_image', decode)
    monkeypatch.setattr(ImageStackLoader, 'load_from_folder', lambda *a, **kw: pytest.fail('preloaded stack'))
    worker = BatchImportWorker('thumbnails', folder_paths=[str(source)])
    results = []
    worker.folders_loaded.connect(results.extend)
    worker.run()
    assert decoded == ['img1.png']
    assert results[0][0] == str(source) and results[0][1] is not None


def test_grouped_batch_cancellation_is_reported(qapp, tmp_path, monkeypatch):
    import core.grouped_fusion as grouped
    import core.workers as workers
    from core.registration import RegistrationCancelled

    source = make_stack(tmp_path / 'stack', dtype=np.uint8)
    class Registration:
        def __init__(self, **kwargs):
            pass
        def process(self, images, **kwargs):
            worker.cancel()
            assert kwargs['should_cancel']()
            raise RegistrationCancelled()
    def cancel_during_group(paths, size, overlap, fuse_group, **kwargs):
        assert not kwargs['should_cancel']()
        image = np.full((12, 16, 3), 50, np.uint8)
        return fuse_group([image, image])
    monkeypatch.setattr(workers, 'ImageRegistration', Registration)
    monkeypatch.setattr(grouped, 'grouped_fuse', cancel_during_group)
    settings = {'fusion_method': 'openfocus_a', 'alignment': 'openfocus',
                'group_size': 2, 'group_overlap': 0}
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out'), settings)
    result = {}
    worker.finished.connect(result.update)
    worker.run()
    assert result['cancelled'] and result['counts']['cancelled'] == 1
    assert not result['outputs']


@pytest.mark.parametrize('keep_history', [False, True])
def test_stack_change_discards_draft_and_preserves_only_history(qapp, monkeypatch, keep_history):
    from PyQt6.QtGui import QAction
    from controllers.transform_manager import TransformManager
    import controllers.export_manager as exporter

    window = host(qapp)
    window.lbl_result_img = QLabel(window)
    window.lbl_result_img.set_roi_rect = lambda _rect: None
    window.add_label_action = QAction(window)
    window.resize_action = QAction(window)
    full = np.full((12, 16, 3), 40000, np.uint16)
    if keep_history:
        record = OutputRecord.create(full, 'full')
        window.output_manager._insert_output(full, record.label, record)
    draft = np.full((6, 8, 3), 1000, np.uint16)
    window.fusion_result = draft
    window._wipe_preview_record = OutputRecord.create(draft, 'draft')

    TransformManager(window).invalidate_processing_results(clear_output_view=True, preserve_outputs=True)

    assert window._wipe_preview_record is None
    assert window.fusion_result is (full if keep_history else None)
    assert window.output_list.count() == int(keep_history)
    if not keep_history:
        warnings = []
        monkeypatch.setattr(exporter, 'show_warning_box', lambda *args: warnings.append(args))
        monkeypatch.setattr(exporter.FileDialog, 'getSaveFileName',
                            lambda *args: pytest.fail('offered to export a discarded preview'))
        window.export_manager.save_result()
        assert len(warnings) == 1


def test_aligned_history_remains_navigable_after_stack_change(qapp):
    from PyQt6.QtGui import QAction
    from controllers.label_manager import LabelManager
    from controllers.transform_manager import TransformManager
    from utils.image_utils import to_display_uint8

    window = host(qapp)
    window.lbl_result_img = MagnifierLabel(parent=window)
    window.lbl_result_img.resize(16, 12)
    window.lbl_result_img.set_roi_rect = lambda _rect: None
    window.add_label_action = QAction(window)
    window.resize_action = QAction(window)
    window.label_manager = LabelManager(window)
    frames = tuple(np.full((12, 16, 3), value, np.uint16) for value in (1000, 60000))
    record = OutputRecord.create(None, 'aligned', kind='aligned')
    record.aligned_frames = frames
    record.aligned_names = ('first.tif', 'second.tif')
    item = window.output_manager._insert_output(None, record.label, record)
    window.result_slider.valueChanged.connect(window.output_manager.show_registration_result)

    TransformManager(window).invalidate_processing_results(clear_output_view=True, preserve_outputs=True)
    assert not window.result_slider.isEnabled()
    window.output_manager.display_output_image_in_result_view(item)

    assert window.result_slider.isEnabled()
    window.result_slider.setValue(1)
    from PyQt6.QtTest import QTest
    import time
    deadline = time.monotonic() + 10
    while window.output_manager._registration_preview_worker is not None and time.monotonic() < deadline:
        QTest.qWait(10)
    assert window.current_result_index == 1
    assert window.lbl_result_info.text() == '2 / 2'
    color = window.lbl_result_img.pixmap().toImage().pixelColor(0, 0)
    expected = int(to_display_uint8(frames[1])[0, 0, 0])
    assert color.red() == color.green() == color.blue() == expected


def test_aligned_slider_resolves_lazy_frames_without_popup_and_discards_old_frame(qapp):
    import threading
    import time
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from controllers.label_manager import LabelManager
    from core.source_frames import ProcessedFrame

    window = host(qapp)
    window.lbl_result_img = MagnifierLabel(parent=window)
    window.lbl_result_img.resize(16, 12)
    window.label_manager = LabelManager(window)
    entered, release = threading.Event(), threading.Event()
    pixels = np.full((12, 16, 3), 30, np.uint8)

    def slow(image):
        entered.set()
        release.wait(5)
        return image

    first = ProcessedFrame(pixels, slow, pixels.shape, pixels.dtype)
    second_pixels = np.full_like(pixels, 210)
    second = ProcessedFrame(second_pixels, lambda image: image, pixels.shape, pixels.dtype)
    window.registration_results = [first, second]
    window.result_slider.setRange(0, 1)
    window.result_slider.valueChanged.connect(window.output_manager.show_registration_result)
    before = set(QApplication.topLevelWidgets())
    try:
        window.output_manager.show_registration_result(0)
        assert entered.wait(5)
        window.result_slider.setValue(1)
        qapp.processEvents()
        assert window.lbl_result_info.text() == '2 / 2'
        assert not any(widget.isVisible() for widget in set(QApplication.topLevelWidgets()) - before)
        release.set()
        deadline = time.monotonic() + 10
        while window.output_manager._registration_preview_worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        assert window.output_manager._registration_preview_worker is None
        assert window.lbl_result_img.pixmap().toImage().pixelColor(0, 0).red() == 210
    finally:
        release.set()
        worker = window.output_manager._registration_preview_worker
        if worker is not None:
            worker.wait(10000)
        window.output_manager.cancel_registration_preview()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize('invalidate', ['source', 'shutdown'])
def test_aligned_preview_cannot_publish_after_invalidation(qapp, invalidate):
    import threading
    import time
    from PyQt6.QtTest import QTest
    from controllers.label_manager import LabelManager
    from core.source_frames import ProcessedFrame

    window = host(qapp)
    window.lbl_result_img = MagnifierLabel(parent=window)
    window.lbl_result_img.resize(16, 12)
    window.label_manager = LabelManager(window)
    entered, release = threading.Event(), threading.Event()
    pixels = np.full((12, 16, 3), 30, np.uint8)

    def slow(image):
        entered.set()
        release.wait(5)
        return image

    window.registration_results = [ProcessedFrame(pixels, slow, pixels.shape, pixels.dtype)]
    try:
        window.output_manager.show_registration_result(0)
        assert entered.wait(5)
        if invalidate == 'source':
            window.registration_results = [np.full_like(pixels, 200)]
        else:
            window._shutting_down = True
            assert window.output_manager.cancel_registration_preview()
        release.set()
        deadline = time.monotonic() + 10
        while window.output_manager._registration_preview_worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        assert window.output_manager._registration_preview_worker is None
        assert window.lbl_result_img.pixmap() is None or window.lbl_result_img.pixmap().isNull()
    finally:
        release.set()
        worker = window.output_manager._registration_preview_worker
        if worker is not None:
            worker.wait(10000)
        window.output_manager.cancel_registration_preview()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize('running', [False, True])
def test_stack_reset_releases_completed_sources_and_keeps_inflight_sources(qapp, running):
    import gc
    import weakref
    from PyQt6.QtGui import QAction
    from controllers.transform_manager import TransformManager
    from core.render_context import build_render_context

    window = host(qapp)
    window.lbl_result_img = QLabel(window)
    window.lbl_result_img.set_roi_rect = lambda _rect: None
    window.add_label_action = QAction(window)
    window.resize_action = QAction(window)

    def completed_source():
        source = np.full((12, 16, 3), 40000, np.uint16)
        result = np.full_like(source, 1000)
        window.last_render_context = build_render_context([source], result)
        window.render_manager = SimpleNamespace(
            worker=object() if running else None,
            _render_context={'source': [source]})
        return weakref.ref(source)

    captured = completed_source()
    TransformManager(window).invalidate_processing_results(clear_output_view=True)
    gc.collect()

    assert (captured() is not None) is running


def test_retouch_child_keeps_original_sources_after_replacement_and_parent_removal(qapp):
    from PyQt6.QtGui import QAction
    from controllers.label_manager import LabelManager
    from controllers.transform_manager import TransformManager
    from core.render_context import build_render_context
    from utils.retouch import RetouchDocument

    window = host(qapp)
    window.lbl_result_img = QLabel(window)
    window.lbl_result_img.set_display_pixmap = window.lbl_result_img.setPixmap
    window.lbl_result_img.set_roi_rect = lambda _rect: None
    window.add_label_action = QAction(window)
    window.resize_action = QAction(window)
    window.label_manager = LabelManager(window)
    source = np.full((12, 16, 3), 40000, np.uint16)
    fused = np.full_like(source, 1000)
    context = build_render_context([source], fused, filenames=['original.tif'])
    base = OutputRecord.create(fused, 'stack', context=context)
    parent = window.output_manager._insert_output(fused, base.label, base)
    child = window.output_manager.add_retouched_output(base, np.full_like(source, 2000))
    window.output_manager._remove_row(window.output_list.row(parent))
    window.raw_images = [np.full_like(source, 60000)]

    TransformManager(window).invalidate_processing_results(clear_output_view=True, preserve_outputs=True)

    assert window.output_manager.stacked_source(child) is None
    assert child.retouchable
    document = RetouchDocument(child.image, child.context)
    document.diameter = 4
    document.hardness = 1.0
    assert document.begin_stroke(8, 6)
    assert document.end_stroke() == 'committed'
    np.testing.assert_array_equal(document.working[6, 8], source[6, 8])
    assert int(child.image[6, 8, 0]) == 2000


def test_registered_preview_applies_warp_and_crop_without_resolving_original(monkeypatch):
    from core.aligned_previews import registered_preview
    from core.registration import PerspectiveWarp
    from core.source_frames import ProcessedFrame, crop_frame, process_frame

    source = np.zeros((64, 64, 3), np.uint8)
    source[18:28, 22:32] = (40, 120, 220)
    matrix = np.array([[1., 0., 7.], [0., 1., -3.], [0., 0., 1.]])
    warped = process_frame(source, PerspectiveWarp(None, tuple(matrix.flat), 'float64', (64, 64)))
    cropped = crop_frame(warped, 8, 4, 48, 52)
    monkeypatch.setattr(ProcessedFrame, 'full_resolution',
                        lambda *args, **kwargs: pytest.fail('resolved original pixels'))
    actual, is_display = registered_preview(cropped)
    expected = cv2.warpPerspective(source, matrix, (64, 64), flags=cv2.INTER_LINEAR)[4:56, 8:56]
    assert not is_display
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, source[4:56, 8:56])


def test_aligned_previews_remain_responsive_during_heavy_render_and_bounded(qapp, monkeypatch):
    import threading
    import time
    from PyQt6.QtTest import QTest
    from controllers.label_manager import LabelManager
    from core.registration import PerspectiveWarp
    from core.resource_policy import heavy_job
    from core.source_frames import process_frame
    monkeypatch.setattr('controllers.output_manager._REGISTRATION_PREVIEW_BYTES', 1200)

    window = host(qapp)
    window.lbl_result_img = MagnifierLabel(parent=window)
    window.lbl_result_img.resize(16, 12)
    window.label_manager = LabelManager(window)
    matrix = tuple(np.eye(3).flat)
    frames = tuple(process_frame(np.full((12, 16, 3), value, np.uint8),
                                 PerspectiveWarp(None, matrix, 'float64', (16, 12)))
                   for value in (30, 100, 210))
    record = OutputRecord.create(None, 'aligned', kind='aligned')
    record.aligned_frames, record.aligned_names = frames, ('a', 'b', 'c')
    item = window.output_manager._insert_output(None, record.label, record)
    entered, release = threading.Event(), threading.Event()

    def hold_render():
        with heavy_job():
            entered.set()
            release.wait(10)

    thread = threading.Thread(target=hold_render)
    thread.start()
    try:
        assert entered.wait(5)
        window.output_manager.display_output_image_in_result_view(item)
        deadline = time.monotonic() + 5
        while window.output_manager._registration_preview_worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        assert window.output_manager._registration_preview_worker is None
        assert len(window.output_manager._registration_previews) == 2
        assert window.output_manager._registration_preview_bytes <= 1200
        window.output_manager.show_registration_result(2)
        assert window.lbl_result_info.text() == '3 / 3'
        assert window.lbl_result_img._base_pixmap.toImage().pixelColor(0, 0).red() == 210
        assert window.output_manager._registration_preview_worker is None
        window.output_manager._remove_row(window.output_list.row(item))
        assert not window.output_manager._registration_previews
        assert window.output_manager._registration_preview_bytes == 0
    finally:
        release.set()
        thread.join(10)
        window.output_manager.clear_registration_previews()
        window.deleteLater()
        qapp.processEvents()


def test_aligned_cached_preview_rejects_changed_source(qapp, tmp_path, monkeypatch):
    import time
    from PyQt6.QtTest import QTest
    from controllers.label_manager import LabelManager
    from core.source_frames import scan_source

    path = tmp_path / 'frame.png'
    assert cv2.imwrite(str(path), np.full((12, 16, 3), 80, np.uint8))
    frame = scan_source(str(path))[0]
    window = host(qapp)
    window.lbl_result_img = MagnifierLabel(parent=window)
    window.lbl_result_img.resize(16, 12)
    window.label_manager = LabelManager(window)
    record = OutputRecord.create(None, 'aligned', kind='aligned')
    record.aligned_frames, record.aligned_names = (frame,), ('frame.png',)
    item = window.output_manager._insert_output(None, record.label, record)
    errors = []
    monkeypatch.setattr(window.output_manager, '_registration_display_error', errors.append)
    window.output_manager.display_output_image_in_result_view(item)
    deadline = time.monotonic() + 5
    while window.output_manager._registration_preview_worker is not None and time.monotonic() < deadline:
        QTest.qWait(10)
    assert window.lbl_result_img._base_pixmap is not None
    with path.open('ab') as stream:
        stream.write(b'changed')
    window.output_manager.show_registration_result(0)
    assert errors and isinstance(errors[0], OSError)
    assert window.lbl_result_img._base_pixmap is None
    assert not window.output_manager._registration_previews
    window.output_manager.clear_registration_previews()
    window.deleteLater()
    qapp.processEvents()



def test_aligned_output_switch_rejects_late_other_record(qapp):
    import threading
    import time
    from PyQt6.QtTest import QTest
    from controllers.label_manager import LabelManager
    from core.source_frames import ProcessedFrame

    window = host(qapp)
    window.lbl_result_img = MagnifierLabel(parent=window)
    window.lbl_result_img.resize(16, 12)
    window.label_manager = LabelManager(window)
    entered, release = threading.Event(), threading.Event()
    old_pixels = np.full((12, 16, 3), 30, np.uint8)

    def slow(image):
        entered.set()
        release.wait(5)
        return image

    first = OutputRecord.create(None, 'first', kind='aligned')
    first.aligned_frames = (ProcessedFrame(old_pixels, slow, old_pixels.shape, old_pixels.dtype),)
    first.aligned_names, first.aligned_frame_ids = ('old',), ('old-id',)
    old_item = window.output_manager._insert_output(None, first.label, first)
    window.output_manager.display_output_image_in_result_view(old_item)
    try:
        assert entered.wait(5)
        second = OutputRecord.create(None, 'second', kind='aligned')
        second.aligned_frames = (np.full_like(old_pixels, 220),)
        second.aligned_names, second.aligned_frame_ids = ('new',), ('new-id',)
        new_item = window.output_manager._insert_output(None, second.label, second)
        window.output_manager.display_output_image_in_result_view(new_item)
        release.set()
        deadline = time.monotonic() + 10
        while window.output_manager._registration_preview_worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        assert window.registration_frame_ids == ['new-id']
        assert window.lbl_result_img._base_pixmap.toImage().pixelColor(0, 0).red() == 220
        window.output_list.setCurrentItem(old_item)
        window.output_manager.display_output_image_in_result_view(old_item)
        deadline = time.monotonic() + 10
        while window.output_manager._registration_preview_worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        assert window.registration_frame_ids == ['old-id']
        assert window.lbl_result_img._base_pixmap.toImage().pixelColor(0, 0).red() == 30
    finally:
        release.set()
        window.output_manager.clear_registration_previews()
        worker = window.output_manager._registration_preview_worker
        if worker is not None:
            worker.wait(10000)
        window.deleteLater()
        qapp.processEvents()


def test_aligned_tooltip_preserves_result_provenance_after_source_changes(qapp, monkeypatch):
    from locales import trans
    from core.algorithms import get_registration

    window = host(qapp)
    window.registration_frame_ids = ['old-a', 'old-b']
    window.registered_source_indices = [1, 3]
    frames = [np.zeros((24, 36, 3), np.uint16), np.zeros((24, 36, 3), np.uint16)]
    monkeypatch.setattr(window.output_manager, '_queue_stack_thumbnail', lambda *args: None)
    publish_aligned_stack(window, frames, filenames=('a.tif', 'b.tif'), icc_profile=b'',
                          source_frames=frames, source_indices=(1, 3), source_revision=4,
                          alignment_options=('ecc',))
    item = window.output_list.item(0)
    record = window.output_manager.record_for_item(item)
    window.registration_frame_ids = ['new']
    window.registered_source_indices = [0]
    window.image_filenames = ['new.tif']
    window.output_manager._refresh_output_tooltip(0)
    tooltip = item.toolTip()
    assert trans.t('output_aligned_stack').format(n=2) in tooltip
    assert trans.t('status_res_fmt').format(36, 24) in tooltip
    assert trans.t(get_registration('ecc').label_key) in tooltip
    assert record.aligned_frame_ids == ('old-a', 'old-b')
    assert record.aligned_source_indices == (1, 3)


def test_fused_tooltip_keeps_render_details_after_export(qapp):
    from locales import trans

    window = host(qapp)
    pixels = np.zeros((21, 34, 3), np.uint8)
    record = OutputRecord.create(pixels, 'result')
    record.tooltip_details = ('method: example', 'time: 1.2s')
    item = window.output_manager._insert_output(pixels, record.label, record)
    window.output_manager.mark_exported(window.output_list.row(item), 'saved.tif')
    assert trans.t('status_res_fmt').format(34, 21) in item.toolTip()
    assert 'method: example' in item.toolTip()
    assert 'time: 1.2s' in item.toolTip()
    assert 'saved.tif' in item.toolTip()


def test_aligned_zoom_preserves_original_geometry_and_refines_selected_frame(qapp):
    import time
    from PyQt6.QtTest import QTest
    from controllers.label_manager import LabelManager

    window = host(qapp)
    view = MagnifierLabel(parent=window)
    view.resize(160, 120)
    window.lbl_result_img = view
    window.label_manager = LabelManager(window)
    frames = tuple(np.full((1200, 1600, 3), value, np.uint8) for value in (50, 220))
    record = OutputRecord.create(None, 'aligned', kind='aligned')
    record.aligned_frames, record.aligned_names = frames, ('a', 'b')
    item = window.output_manager._insert_output(None, record.label, record)
    try:
        window.output_manager.display_output_image_in_result_view(item)
        deadline = time.monotonic() + 10
        while window.output_manager._registration_preview_worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        assert (view._source_size.width(), view._source_size.height()) == (1600, 1200)
        view._zoom_factor = 8.0
        view.refresh_display()
        window.output_manager.show_registration_result(1)
        assert view._zoom_factor == 8.0
        deadline = time.monotonic() + 10
        while ('view' not in view._source_regions or window.current_result_index != 1) and time.monotonic() < deadline:
            QTest.qWait(10)
        tile, _ = view._source_regions['view']
        assert tile.toImage().pixelColor(tile.width() // 2, tile.height() // 2).red() == 220
        assert (view._source_size.width(), view._source_size.height()) == (1600, 1200)
    finally:
        window.output_manager.clear_registration_previews()
        worker = window.output_manager._registration_region_worker
        if worker is not None:
            worker.wait(10000)
        window.deleteLater()
        qapp.processEvents()
