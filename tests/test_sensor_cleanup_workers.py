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
    seen = []
    def register(images, should_cancel=None):
        seen.append(images[0].copy())
        return images, 0.0
    worker = RenderWorker([image, image.copy()], algorithm=None, need_fusion=False,
                          alignment='ecc', defect_mask_path=str(path))
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


def test_render_admits_full_cleanup_stack_before_mask_or_copy(qapp, tmp_path, monkeypatch):
    from core.workers import RenderWorker
    import utils.sensor_cleanup as sensor
    path, image = _inputs(tmp_path)
    frames = [image, image.copy()]
    seen = []
    def reject(images, mask_path, denoise):
        seen.append((list(images), mask_path, denoise))
        raise MemoryError('cleanup budget')
    def unexpected(*args, **kwargs):
        pytest.fail('cleanup allocated before admission')
    monkeypatch.setattr(sensor, 'require_sensor_cleanup_stack', reject)
    monkeypatch.setattr(sensor, 'load_sensor_defect_mask', unexpected)
    monkeypatch.setattr(sensor, 'cleanup_sensor_image', unexpected)
    worker = RenderWorker(frames, algorithm=None, need_fusion=False,
                          defect_mask_path=str(path), sensor_denoise=True)
    with pytest.raises(MemoryError, match='cleanup budget'):
        worker._run_admitted()
    assert len(seen) == 1
    assert seen[0][0][0] is frames[0] and seen[0][0][1] is frames[1]
    assert seen[0][1:] == (str(path), True)


def test_batch_admits_cleanup_before_mask_decode(qapp, tmp_path, monkeypatch):
    from core.workers import BatchWorker
    import utils.sensor_cleanup as sensor
    source = tmp_path / 'source'
    source.mkdir()
    path, image = _inputs(source)
    for index in range(2):
        assert cv2.imwrite(str(source / f'{index}.png'), image)
    admitted = []
    def reject(images, mask_path, denoise):
        admitted.append((len(images), mask_path, denoise))
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
    assert admitted == [(2, str(path), True)]
    assert outcome['success'] == 0
