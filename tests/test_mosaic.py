"""Microscope mosaic placement, rendering and export contracts."""

from collections import namedtuple
from pathlib import Path
import cv2
import numpy as np
import pytest

from core import frame_storage
from core.mosaic import auto_align, make_frame, render_mosaic
from core.registration import RegistrationCancelled
from utils.image_utils import imwrite_auto


def test_mosaic_uses_captured_storage_directory(monkeypatch, tmp_path):
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', None)
    frame_storage.configure_storage_directory(first)
    checked = []
    usage = namedtuple('usage', 'total used free')
    def disk_usage(path):
        checked.append(path)
        frame_storage.configure_storage_directory(second)
        return usage(20 * 1024**3, 0, 2 * 1024**3)
    monkeypatch.setattr(frame_storage.shutil, 'disk_usage', disk_usage)
    frame = make_frame('one', np.zeros((4, 4, 3), np.uint8))
    output, _ = render_mosaic([frame], [np.eye(3)])
    assert checked == [str(first), str(first)]
    assert Path(output.filename).parent == first


def test_overlapping_microscope_tiles_align_and_reconstruct():
    rng = np.random.default_rng(17)
    scene = rng.integers(0, 65536, (256, 420, 3), dtype=np.uint16)
    frames = [make_frame('left', scene[:, :280]),
              make_frame('right', scene[:, 140:])]
    alignment = auto_align(frames)
    assert alignment.connected == (True, True)
    assert alignment.transforms[1][0, 2] == pytest.approx(140, abs=1)
    assert alignment.transforms[1][1, 2] == pytest.approx(0, abs=1)
    placements = [np.eye(3), np.array([[1, 0, 140], [0, 1, 0], [0, 0, 1]])]
    output, origin = render_mosaic(frames, placements, tile_size=64)
    assert output.dtype == np.uint16
    assert output.shape == scene.shape and origin == (0, 0)
    np.testing.assert_allclose(output, scene, rtol=0, atol=1)


def test_unmatched_tiles_remain_available_for_manual_placement():
    frames = [make_frame('a', np.full((32, 48, 3), 100, np.uint8)),
              make_frame('b', np.full((32, 48, 3), 200, np.uint8))]
    alignment = auto_align(frames)
    assert alignment.connected == (True, False)
    placements = [np.eye(3), np.array([[1, 0, 48], [0, 1, 0], [0, 0, 1]])]
    output, _ = render_mosaic(frames, placements, tile_size=32)
    assert output.shape == (32, 96, 3)
    assert np.all(output[:, :48] == 100)
    assert np.all(output[:, 48:] == 200)


def test_mixed_depth_alpha_and_streamed_tiff(tmp_path):
    first = np.full((20, 24, 4), (10, 20, 30, 255), np.uint8)
    second = np.full((20, 24, 4), (1000, 2000, 3000, 65535), np.uint16)
    frames = [make_frame('first', first), make_frame('second', second)]
    transforms = [np.eye(3), np.array([[1, 0, 24], [0, 1, 0], [0, 0, 1]])]
    output, _ = render_mosaic(frames, transforms)
    assert output.dtype == np.uint16 and output.shape == (20, 48, 4)
    np.testing.assert_array_equal(output[5, 5], np.array([2570, 5140, 7710, 65535]))
    np.testing.assert_array_equal(output[5, 30], second[5, 6])
    path = tmp_path / '拼接.tif'
    assert imwrite_auto(path, output, exif=b'', stream_tiff=True)
    actual = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, output)


def test_fractional_edges_and_feather_are_tile_independent():
    first = np.full((80, 96, 3), 10000, np.uint16)
    second = np.full((80, 96, 3), 40000, np.uint16)
    frames = [make_frame('first', first), make_frame('second', second)]
    transforms = [np.array([[1, 0, -0.75], [0, 1, -0.75], [0, 0, 1]]),
                  np.array([[1, 0, 47.25], [0, 1, -0.75], [0, 0, 1]])]
    small, origin = render_mosaic(frames, transforms, tile_size=32)
    large, other_origin = render_mosaic(frames, transforms, tile_size=128)
    assert origin == other_origin == (-1, -1)
    np.testing.assert_array_equal(small, large)
    assert small.dtype == np.uint16
    # The final row and column sample within the fractional source footprint.
    assert small[-1, -1, 0] == 40000
    assert small[-1, 10, 0] == 10000
    # Feathering must transition through intermediate values in the overlap.
    overlap = small[40, 50:95, 0]
    assert np.all(np.diff(overlap.astype(np.int32)) >= 0)
    assert np.any((overlap > 10000) & (overlap < 40000))


def test_render_cancels_between_tiles():
    frame = make_frame('frame', np.full((96, 96, 3), 125, np.uint8))
    steps = []
    with pytest.raises(RegistrationCancelled):
        render_mosaic([frame], [np.eye(3)], tile_size=32,
                      should_cancel=lambda: len(steps) > 1,
                      progress=lambda done, total: steps.append((done, total)))
    assert steps == [(0, 9), (1, 9)]


def test_streamed_tiff_limits_strip_memory(tmp_path, monkeypatch):
    import tifffile
    from core import resource_policy
    from utils import image_utils

    image = np.arange(9 * 20 * 4, dtype=np.uint16).reshape(9, 20, 4)
    requested = []
    monkeypatch.setattr(image_utils, '_TIFF_STRIP_TARGET_BYTES', 256)
    monkeypatch.setattr(resource_policy, 'require_memory', requested.append)
    path = tmp_path / 'strips.tif'
    assert imwrite_auto(path, image, exif=b'', stream_tiff=True)
    assert requested == [20 * 4 * 2 * 3]
    with tifffile.TiffFile(path) as tif:
        assert tif.pages[0].rowsperstrip == 1
    actual = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, image)


def test_cancelled_export_preserves_existing_file(tmp_path):
    path = tmp_path / 'existing.tif'
    path.write_bytes(b'previous')
    image = np.zeros((200, 20, 3), dtype=np.uint16)
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(RegistrationCancelled):
        imwrite_auto(path, image, exif=b'', stream_tiff=True, should_cancel=cancel)
    assert path.read_bytes() == b'previous'
    assert sorted(item.name for item in tmp_path.iterdir()) == ['existing.tif']


def test_workspace_placement_and_preview_opacity(qapp):
    from PyQt6.QtWidgets import QWidget
    from dialogs.mosaic import MosaicDialog

    parent = QWidget()
    dialog = MosaicDialog(parent)
    frames = [make_frame('a', np.full((32, 40, 3), 50, np.uint8)),
              make_frame('b', np.full((32, 40, 3), 200, np.uint8))]
    dialog._append_frames(frames)
    dialog.list.setCurrentRow(1)
    dialog.x_spin.setValue(55)
    dialog.y_spin.setValue(-7)
    dialog.opacity.setValue(40)
    matrix = dialog._current_transforms()[1]
    assert matrix[0, 2] == 55 and matrix[1, 2] == -7
    assert dialog.items[1].opacity() == pytest.approx(0.4)
    assert dialog.save.isEnabled() and dialog.auto.isEnabled()
    dialog.close()
    parent.close()


def test_workspace_imports_all_tiff_pages(qapp, tmp_path, monkeypatch):
    import time
    import tifffile
    from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget
    from dialogs.mosaic import MosaicDialog

    source = tmp_path / '多页.tif'
    with tifffile.TiffWriter(source) as writer:
        writer.write(np.full((12, 16, 3), 20, np.uint8), photometric='rgb')
        writer.write(np.full((12, 16, 3), 180, np.uint8), photometric='rgb')
    monkeypatch.setattr(QFileDialog, 'getOpenFileNames',
                        lambda *args, **kwargs: ([str(source)], ''))
    errors = []
    monkeypatch.setattr(QMessageBox, 'warning',
                        lambda *args, **kwargs: errors.append(args))
    parent = QWidget()
    dialog = MosaicDialog(parent)
    dialog._add_files()
    deadline = time.monotonic() + 10
    while dialog.worker is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert dialog.worker is None
    assert not errors
    assert len(dialog.frames) == 2
    assert all(isinstance(frame.image, np.memmap) for frame in dialog.frames)
    assert [int(frame.image[0, 0, 0]) for frame in dialog.frames] == [20, 180]
    dialog.close()
    parent.close()
