"""Folder input ordered near to far by EXIF focus steps."""
import os
from types import SimpleNamespace

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


def test_folder_frame_order_skips_calibration_references(tmp_path):
    """Read real maker notes while leaving calibration files out of ordering."""
    import struct
    import core.cli as cli
    from core.image_loader import ImageStackLoader
    from utils.image_utils import imwrite_auto

    def focus_jpeg(name, step):
        # Standard Exif IFD -> MakerNote -> OM FocusInfo IFD. All offsets
        # in the maker note are relative to its own header.
        note = (b'OM SYSTEM\0\0\0II\x04\0' + struct.pack('<H', 1)
                + struct.pack('<HHII', 0x2050, 4, 1, 34)
                + struct.pack('<I', 0) + struct.pack('<H', 3))
        for tag, value in ((0x0301, step), (0x0303, 0), (0x0304, 20000)):
            note += struct.pack('<HHIHH', tag, 3, 1, value, 0)
        note += struct.pack('<I', 0)
        tiff = (b'II*\0' + struct.pack('<I', 8) + struct.pack('<H', 1)
                + struct.pack('<HHII', 0x8769, 4, 1, 26)
                + struct.pack('<I', 0) + struct.pack('<H', 1)
                + struct.pack('<HHII', 0x927C, 7, len(note), 44)
                + struct.pack('<I', 0) + note)
        exif = b'Exif\0\0' + tiff
        ok, encoded = cv2.imencode('.jpg', np.full((8, 8, 3), 96, np.uint8))
        assert ok
        raw = encoded.tobytes()
        app1 = b'\xff\xe1' + struct.pack('>H', len(exif) + 2) + exif
        (tmp_path / name).write_bytes(raw[:2] + app1 + raw[2:])

    focus_jpeg('frame_1.jpg', 17121)
    focus_jpeg('frame_2.jpg', 18216)
    for name in ('flat.jpg', 'mask.png'):
        assert imwrite_auto(str(tmp_path / name), np.zeros((8, 8, 3), np.uint8))
    args = SimpleNamespace(frame_order='auto', dust_map=str(tmp_path / 'flat.jpg'),
                           defect_mask_path=str(tmp_path / 'mask.png'))
    paths = cli._folder_stack_paths(str(tmp_path), ImageStackLoader(), args)
    assert [os.path.basename(path) for path in paths] == ['frame_2.jpg', 'frame_1.jpg']
    args.frame_order = 'name'
    paths = cli._folder_stack_paths(str(tmp_path), ImageStackLoader(), args)
    assert [os.path.basename(path) for path in paths] == ['frame_1.jpg', 'frame_2.jpg']
