"""Folder input ordered near to far by EXIF focus steps."""
import cv2
import numpy as np

from core.focus_order import FocusPosition, _order, order_by_focus


def test_focus_order_follows_focus_steps_only_when_direction_is_known():
    # OM SYSTEM bracketing: shot 1 at the start (mid-depth), then near -> far;
    # larger step counts are nearer (near_step > infinity_step).
    steps = {"s01": 17121, "s02": 18216, "s03": 17997, "s04": 17778, "s05": 16902}
    known = [FocusPosition(path, step, near_step=20000, infinity_step=0)
             for path, step in steps.items()]
    assert _order(known)[0] == ["s02", "s03", "s04", "s01", "s05"]
    unknown = [FocusPosition(path, step) for path, step in steps.items()]
    assert _order(unknown)[0] is None
    partial = known[:-1] + [FocusPosition("s05")]
    assert _order(partial)[0] is None


def test_focus_order_leaves_frames_without_metadata_alone(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"frame_{i}.png"
        assert cv2.imwrite(str(path), np.full((16, 16, 3), 40 * i, np.uint8))
        paths.append(str(path))
    assert order_by_focus(paths) == (paths, None)
