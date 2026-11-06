import cv2
import numpy as np
import pytest


def _inputs(tmp_path):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5, 5] = 255
    path = tmp_path / 'mask.png'
    assert cv2.imwrite(str(path), mask)
    image = np.full((16, 16, 3), 10001, dtype=np.uint16)
    image[5, 5] = 60000
    return path, image


def test_render_cleans_before_registration_and_rejects_scaled_mask(qapp, tmp_path, monkeypatch):
    from core.workers import RenderWorker
    path, image = _inputs(tmp_path)
    source = tmp_path / 'frame.png'
    assert cv2.imwrite(str(source), image)
    seen = []
    def register(images, should_cancel=None):
        seen.append(images[0].copy())
        return images, 0.0
    worker = RenderWorker([image, image.copy()], algorithm=None, need_fusion=False,
                          alignment='ecc', defect_mask_path=str(path), source_paths=[str(source)] * 2)
    monkeypatch.setattr(worker, '_run_registration', register)
    worker._run_admitted()
    assert seen[0][5, 5, 0] == 10001
    assert image[5, 5, 0] == 60000
    scaled = RenderWorker([image, image.copy()], algorithm=None, need_fusion=False,
                          alignment='ecc', defect_mask_path=str(path), load_scale_factor=0.5)
    with pytest.raises(ValueError, match='100%'):
        scaled._run_admitted()


@pytest.mark.parametrize('grouped', [False, True])
def test_batch_excludes_and_applies_mask_before_processing(qapp, tmp_path, monkeypatch, grouped):
    from core.workers import BatchWorker
    source = tmp_path / 'source'
    source.mkdir()
    path, image = _inputs(source)
    for i in range(2):
        assert cv2.imwrite(str(source / f'{i}.png'), image)
    settings = {'format': 'png', 'fusion_method': 'dct', 'alignment': 'none',
                'defect_mask_path': str(path), 'group_size': 2 if grouped else 0}
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out'), settings, use_gpu=False)
    seen = []
    def fuse(fusion, method, images, params):
        seen.append([frame.copy() for frame in images])
        return images[0]
    monkeypatch.setattr(worker, '_fuse_one_stack', fuse)
    outcome = {}
    worker.finished.connect(outcome.update)
    worker.run()
    assert outcome['success'] == 1
    assert len(seen[0]) == 2
    for frame in seen[0]:
        assert frame.dtype == np.float32
        np.testing.assert_allclose(frame, 10001.0 / 257.0, rtol=1e-6)


def test_render_admits_full_cleanup_stack_before_mask_or_processing(qapp, tmp_path, monkeypatch):
    from core.workers import RenderWorker
    import utils.sensor_cleanup as sensor
    path, image = _inputs(tmp_path)
    source = tmp_path / 'frame.png'
    assert cv2.imwrite(str(source), image)
    frames = [image, image.copy()]
    seen = []
    def reject(images, mask_path, denoise, *, disk_backed=False):
        seen.append((list(images), mask_path, denoise, disk_backed))
        raise MemoryError('cleanup budget')
    def unexpected(*args, **kwargs):
        pytest.fail('cleanup allocated before admission')
    monkeypatch.setattr(sensor, 'require_sensor_cleanup_stack', reject)
    monkeypatch.setattr(sensor, 'load_sensor_defect_mask', unexpected)
    monkeypatch.setattr(sensor, 'cleanup_sensor_image', unexpected)
    worker = RenderWorker(frames, algorithm=None, need_fusion=False,
                          defect_mask_path=str(path), sensor_denoise=True, source_paths=[str(source)] * 2)
    with pytest.raises(MemoryError, match='cleanup budget'):
        worker._run_admitted()
    assert len(seen) == 1
    assert len(seen[0][0]) == len(frames)
    for admitted, original in zip(seen[0][0], frames):
        assert admitted.shape == original.shape and admitted.dtype == original.dtype
        np.testing.assert_array_equal(admitted, original)
    assert seen[0][1:] == (str(path), True, True)


def test_batch_admits_cleanup_before_mask_decode(qapp, tmp_path, monkeypatch):
    from core.workers import BatchWorker
    import utils.sensor_cleanup as sensor
    source = tmp_path / 'source'
    source.mkdir()
    path, image = _inputs(source)
    for index in range(2):
        assert cv2.imwrite(str(source / f'{index}.png'), image)
    admitted = []
    def reject(images, mask_path, denoise, *, disk_backed=False):
        admitted.append((len(images), mask_path, denoise, disk_backed))
        raise MemoryError('cleanup budget')
    def unexpected(*args, **kwargs):
        pytest.fail('mask decoded before admission')
    monkeypatch.setattr(sensor, 'require_sensor_cleanup_stack', reject)
    monkeypatch.setattr(sensor, 'load_sensor_defect_mask', unexpected)
    worker = BatchWorker([str(source)], 'custom', str(tmp_path / 'out'),
        {'format': 'png', 'fusion_method': 'dct', 'alignment': 'none',
         'defect_mask_path': str(path), 'sensor_denoise': True}, use_gpu=False)
    outcome = {}
    worker.finished.connect(outcome.update)
    worker.run()
    assert admitted == [(2, str(path), True, True)]
    assert outcome['success'] == 0


def _oriented_jpegs(folder, orientations=(1, 6), values=None):
    from PIL import Image
    folder.mkdir()
    paths = []
    for index, orientation in enumerate(orientations):
        path = folder / f'{index}.jpg'
        value = 80 + index if values is None else values[index]
        image = Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8))
        exif = image.getexif()
        exif[274] = orientation
        image.save(path, exif=exif)
        paths.append(str(path))
    return paths


def test_gui_render_passes_included_sources_and_reports_orientation(qapp, tmp_path, monkeypatch):
    from core.image_loader import decode_image
    from core.workers import RenderWorker
    from locales import trans
    from tests.test_gui_units import _render_window, _patched_render_manager

    paths = _oriented_jpegs(tmp_path / 'source')
    mask, _ = _inputs(tmp_path)
    frames = [decode_image(path) for path in paths]
    window = _render_window(frames)
    window.defect_mask_path = str(mask)
    window.image_source_paths = paths
    manager, _ = _patched_render_manager(monkeypatch, window)
    manager.start_render()
    assert manager.worker.kwargs['source_paths'] == paths
    worker = RenderWorker(*manager.worker.args, **manager.worker.kwargs)
    errors = []
    worker.error_signal.connect(errors.append)
    worker.run()
    assert errors == [trans.t('sensor_source_orientation').format(path=paths[1])]


def test_excluded_rotated_source_does_not_block_masked_gui_render(qapp, tmp_path, monkeypatch):
    from core.image_loader import decode_image
    from core.workers import RenderWorker
    from tests.test_gui_units import _render_window, _patched_render_manager

    paths = _oriented_jpegs(tmp_path / 'source', (1, 6, 1), (80, 240, 80))
    mask, _ = _inputs(tmp_path)
    frames = [decode_image(path) for path in paths]
    window = _render_window(frames)
    window.defect_mask_path = str(mask)
    window.image_source_paths = paths
    window.excluded_frames = {1}
    window.effective_stack = lambda: ([frames[0], frames[2]],
                                      [window.image_filenames[0], window.image_filenames[2]], [0, 2])
    manager, _ = _patched_render_manager(monkeypatch, window)
    manager.start_render(force_algorithm='dct')
    worker = RenderWorker(*manager.worker.args, **manager.worker.kwargs)
    errors, results = [], []
    worker.error_signal.connect(errors.append)
    worker.finished_signal.connect(lambda _frames, result, *_: results.append(result))
    worker.run()
    assert errors == []
    np.testing.assert_array_equal(results[0], np.full((16, 16, 3), 80, np.uint8))


@pytest.mark.parametrize('source_paths', [[], ['missing.jpg', 'missing.jpg'], ['video.mp4', 'video.mp4']])
def test_render_rejects_unknown_mask_provenance(qapp, tmp_path, source_paths):
    from core.workers import RenderWorker
    mask, image = _inputs(tmp_path)
    if source_paths and source_paths[0] == 'video.mp4':
        video = tmp_path / 'video.mp4'
        video.write_bytes(b'video')
        source_paths = [str(video)] * 2
    worker = RenderWorker([image, image.copy()], algorithm=None, need_fusion=False,
                          defect_mask_path=str(mask), source_paths=source_paths)
    errors = []
    worker.error_signal.connect(errors.append)
    worker.run()
    assert errors


@pytest.mark.parametrize('missing', [False, True])
def test_render_reports_localized_sensor_source_problem(qapp, tmp_path, monkeypatch, missing):
    from core.image_loader import decode_image
    from core.workers import RenderWorker
    from locales import trans
    paths = _oriented_jpegs(tmp_path / 'source')
    mask, _ = _inputs(tmp_path)
    frames = [decode_image(path) for path in paths]
    monkeypatch.setattr(trans, 'current_lang', 'fr')
    worker = RenderWorker(frames, algorithm=None, need_fusion=False,
                          defect_mask_path=str(mask), source_paths=[] if missing else paths)
    errors = []
    worker.error_signal.connect(errors.append)
    worker.run()
    expected = (trans.t('sensor_source_missing') if missing else
                trans.t('sensor_source_orientation').format(path=paths[1]))
    assert errors == [expected]


def test_sensor_source_messages_exist_in_all_locales(qapp):
    from locales import trans
    for language in trans.PACK_ORDER:
        pack = trans.translations[language]
        assert pack['sensor_source_missing']
        assert '{path}' in pack['sensor_source_orientation']


@pytest.mark.parametrize('mode', ['folder', 'split', 'grouped'])
def test_batch_rejects_square_oriented_jpeg(qapp, tmp_path, mode):
    from core.image_loader import decode_image
    from core.workers import BatchWorker
    from locales import trans

    paths = _oriented_jpegs(tmp_path / 'source')
    mask, _ = _inputs(tmp_path)
    settings = {'format': 'png', 'fusion_method': 'dct', 'alignment': 'none',
                'defect_mask_path': str(mask), 'group_size': 2 if mode == 'grouped' else 0}
    entries = [(path, decode_image(path), None, f'page_{index}.jpg')
               for index, path in enumerate(paths)]
    worker = BatchWorker([str(tmp_path / 'source')] if mode != 'split' else [],
                         'custom', str(tmp_path / 'out'), settings, use_gpu=False,
                         import_mode='single_folder' if mode == 'split' else 'multiple_folders',
                         single_folder_images_with_times=entries if mode == 'split' else None)
    outcome = {}
    worker.finished.connect(outcome.update)
    worker.run()
    assert outcome['success'] == 0
    assert trans.t('sensor_source_orientation').format(path=paths[1]) in outcome['failed'][0]


@pytest.mark.parametrize('denoise', [False, True])
def test_render_without_mask_does_not_validate_orientation(qapp, tmp_path, monkeypatch, denoise):
    import utils.sensor_cleanup as sensor
    from core.workers import RenderWorker
    paths = _oriented_jpegs(tmp_path / 'source')
    from core.image_loader import decode_image
    frames = [decode_image(path) for path in paths]
    monkeypatch.setattr(sensor, 'validate_sensor_source_orientation',
                        lambda *args, **kwargs: pytest.fail('orientation checked without mask'))
    worker = RenderWorker(frames, algorithm=None, need_fusion=False,
                          sensor_denoise=denoise, source_paths=paths)
    errors = []
    worker.error_signal.connect(errors.append)
    worker.run()
    assert not errors


@pytest.mark.parametrize('mode', ['folder', 'split', 'grouped'])
def test_batch_denoise_does_not_validate_orientation(qapp, tmp_path, monkeypatch, mode):
    import utils.sensor_cleanup as sensor
    from core.image_loader import decode_image
    from core.workers import BatchWorker

    paths = _oriented_jpegs(tmp_path / 'source')
    settings = {'format': 'png', 'fusion_method': 'dct', 'alignment': 'none',
                'sensor_denoise': True, 'group_size': 2 if mode == 'grouped' else 0}
    entries = [(path, decode_image(path), None, f'page_{index}.jpg')
               for index, path in enumerate(paths)]
    monkeypatch.setattr(sensor, 'validate_sensor_source_orientation',
                        lambda *args, **kwargs: pytest.fail('orientation checked without mask'))
    worker = BatchWorker([str(tmp_path / 'source')] if mode != 'split' else [],
                         'custom', str(tmp_path / 'out'), settings, use_gpu=False,
                         import_mode='single_folder' if mode == 'split' else 'multiple_folders',
                         single_folder_images_with_times=entries if mode == 'split' else None)
    monkeypatch.setattr(worker, '_fuse_one_stack', lambda fusion, method, images, params: images[0])
    outcome = {}
    worker.finished.connect(outcome.update)
    worker.run()
    assert outcome['success'] == 1
