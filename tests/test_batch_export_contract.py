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
    monkeypatch.setattr(BatchWorker, '_fuse_one_stack', lambda self, fusion, method, images, params: images[0])
    settings = {'format': 'png', 'fusion_method': 'openfocus_a', 'alignment': 'openfocus',
                'save_aligned': False, 'group_size': 3, 'group_overlap': 1,
                'runtime_settings': {'sort_mode': 'desc' if reverse else 'asc'}}
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


def test_grouped_batch_real_registration_keeps_singleton_coordinates(qapp, tmp_path, monkeypatch):
    import core.registration as registration
    source = tmp_path / 'registered'
    source.mkdir()
    y, x = np.indices((64, 68))
    frame = np.stack([x, y, x + y], axis=-1).astype(np.uint16) * 257
    for index in range(5):
        assert cv2.imwrite(str(source / f'{index}.png'), frame)
    monkeypatch.setattr(registration, '_of_estimate_similarity',
                        lambda *args, **kwargs: (1.0, 0.0, 1.0, 0.0, 0.0))
    settings = {'format': 'png', 'fusion_method': 'openfocus_a', 'alignment': 'openfocus',
                'group_size': 2, 'group_overlap': 0}
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out'), settings, use_gpu=False)
    result = {}
    worker.finished.connect(result.update)
    worker.run()
    assert result['counts']['success'] == 1
    saved = cv2.imread(result['outputs'][0], cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(saved, frame[12:52, 12:56])


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
    window.lbl_result_img = QLabel(window)
    window.lbl_result_img.set_display_pixmap = window.lbl_result_img.setPixmap
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
    assert window.current_result_index == 1
    assert window.lbl_result_info.text() == '2 / 2'
    color = window.lbl_result_img.pixmap().toImage().pixelColor(0, 0)
    expected = int(to_display_uint8(frames[1])[0, 0, 0])
    assert color.red() == color.green() == color.blue() == expected


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
