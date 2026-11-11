import numpy as np
import pytest

from tools._proto_feather_visual import select_crop
from tools import _proto_feather_visual as visual


def test_select_crop_prefers_largest_score():
    plane = np.zeros((80, 80), dtype=np.float32)
    plane[48:72, 48:72] = 5
    score, y, x = select_crop(plane)
    assert score > 0
    assert y > 0 and x > 0


def test_select_crop_rejects_no_candidates():
    with pytest.raises(ValueError, match='larger than 48'):
        select_crop(np.zeros((48, 80)))



def test_real_stack_uses_natural_order(tmp_path, monkeypatch):
    for name in ('frame10.png', 'frame2.jpg', 'frame1.tif'):
        (tmp_path / name).touch()
    monkeypatch.setattr(visual.cv2, 'imread', lambda path, flags: path)
    frames = visual.load_stack(str(tmp_path))
    assert [__import__('os').path.basename(path) for path in frames] == [
        'frame1.tif', 'frame2.jpg', 'frame10.png']