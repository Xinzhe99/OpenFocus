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
    assert all(frame[5, 5, 0] < 0.5 for frame in seen[0])
