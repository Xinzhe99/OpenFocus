"""Behavioral boundaries for directory artifacts and stable image exports."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QLabel, QListWidget, QSlider, QWidget

from controllers.export_manager import ExportManager, _read_image_any
from controllers.output_manager import OutputManager, OutputRecord, publish_aligned_stack
from core.image_loader import ImageStackLoader
from core.workers import BatchWorker, GifSaverWorker
from utils.image_utils import imwrite_auto, imwrite_multi_tiff


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
    window.output_list = QListWidget(window)
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
                'load_scale_factor': .5, 'fusion_params': {'kernel_size': 3,
                    'base_frame': 'last', 'sort_mode': 'desc', 'exposure_match': False}}
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

    monkeypatch.setattr(module.QFileDialog, 'getSaveFileName', pick)
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
    monkeypatch.setattr(module.QFileDialog, 'getExistingDirectory', lambda *args, **kwargs: str(destination))
    window.label_manager.prepare_bgr_image = lambda *args, **kwargs: pytest.fail('Baked output annotated twice')
    window.output_manager.export_all_results()
    assert (destination / 'historical_name.png').read_bytes() == b'keep'
    assert _read_image_any(str(destination / 'historical_name_2.png'))[0, 0, 0] == 80
    assert (destination / 'source_aligned' / 'img1.png').read_bytes() == (source / 'img1.png').read_bytes()
    assert window.output_list.count() == 2


def test_gif_cancel_does_not_replace_previous_file(qapp, tmp_path, monkeypatch):
    import core.workers as workers
    destination = tmp_path / 'existing.gif'
    destination.write_bytes(b'previous')
    worker = GifSaverWorker([np.zeros((8, 8, 3), np.uint8)], str(destination), .1, Labels(), 'registered')

    def interrupted_save(path, *args, **kwargs):
        Path(path).write_bytes(b'incomplete')
        worker.cancel()

    monkeypatch.setattr(workers.imageio, 'mimsave', interrupted_save)
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
    align_cache.save_aligned(str(source), names, frames, (True, False), 1024)
    monkeypatch.setattr(align_cache, 'imwrite_auto', lambda *args, **kwargs: False)
    align_cache.save_aligned(str(source), names, frames, (False, True), 1024)
    assert align_cache.load_aligned(str(source), names, (True, False), 1024) is not None
    assert align_cache.load_aligned(str(source), names, (False, True), 1024) is None
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
    monkeypatch.setattr(module.QFileDialog, 'getSaveFileName',
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
    monkeypatch.setattr(module.QFileDialog, 'getSaveFileName',
                        lambda *args, **kwargs: (str(tmp_path / 'batch.png'), 'PNG Files (*.png)'))
    window.output_manager.save_output_image_as(item)
    window.label_manager.labels.add(Label.from_dict({'target': 'output', 'format': 'New'}))
    record.scale_bar_config = {'px_um': 2}
    window.output_manager.refresh_annotation_export_state()
    assert record.exported and not record.changed_since_export
    np.testing.assert_array_equal(_read_image_any(str(tmp_path / 'batch.png')), record.image)
