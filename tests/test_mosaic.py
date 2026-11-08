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
    usage = namedtuple('usage', 'total used free')
    def disk_usage(path):
        frame_storage.configure_storage_directory(second)
        return usage(20 * 1024**3, 0, 2 * 1024**3)
    monkeypatch.setattr(frame_storage.shutil, 'disk_usage', disk_usage)
    frame = make_frame('one', np.zeros((4, 4, 3), np.uint8))
    output, _ = render_mosaic([frame], [np.eye(3)])
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
    # Include strip tables as well as the minimum one-row conversion buffer.
    assert requested and max(requested) < image.nbytes * 4
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
    from PyQt6.QtWidgets import QMessageBox, QWidget
    from utils.file_dialogs import FileDialog
    from dialogs.mosaic import MosaicDialog

    source = tmp_path / '多页.tif'
    with tifffile.TiffWriter(source) as writer:
        writer.write(np.full((12, 16, 3), 20, np.uint8), photometric='rgb')
        writer.write(np.full((12, 16, 3), 180, np.uint8), photometric='rgb')
    monkeypatch.setattr(FileDialog, 'getOpenFileNames',
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
    for frame, value in zip(dialog.frames, (20, 180)):
        np.testing.assert_array_equal(frame.image, np.full((12, 16, 3), value, np.uint8))
    dialog.close()
    parent.close()


@pytest.mark.parametrize('order,direction,corner,cells', [
    ('byrow', 'unidirectional', 'topleft', [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
    ('byrow', 'zigzag', 'topright', [(0, 2), (0, 1), (0, 0), (1, 0), (1, 1), (1, 2)]),
    ('byrow', 'zigzag', 'bottomleft', [(1, 0), (1, 1), (1, 2), (0, 2), (0, 1), (0, 0)]),
    ('byrow', 'unidirectional', 'bottomright', [(1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)]),
    ('bycolumn', 'unidirectional', 'topleft', [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]),
    ('bycolumn', 'zigzag', 'topright', [(0, 2), (1, 2), (1, 1), (0, 1), (0, 0), (1, 0)]),
    ('bycolumn', 'unidirectional', 'bottomleft', [(1, 0), (0, 0), (1, 1), (0, 1), (1, 2), (0, 2)]),
    ('bycolumn', 'zigzag', 'bottomright', [(1, 2), (0, 2), (0, 1), (1, 1), (1, 0), (0, 0)]),
])
def test_capture_grid_renders_in_declared_order(order, direction, corner, cells):
    from core.mosaic import grid_transforms

    frames = [make_frame(str(index), np.full((8, 10, 3), index + 1, np.uint8))
              for index in range(6)]
    transforms = grid_transforms(frames, 2, 3, order=order, direction=direction,
                                 start_corner=corner, overlap_pct=0)
    output, origin = render_mosaic(frames, transforms, feather=0)
    assert output.shape == (16, 30, 3) and origin == (0, 0)
    for index, (row, col) in enumerate(cells):
        assert np.all(output[row * 8:(row + 1) * 8, col * 10:(col + 1) * 10] == index + 1)


def test_capture_grid_uses_actual_mixed_dimensions():
    from core.mosaic import canvas_bounds, grid_transforms

    frames = [make_frame(str(index), np.zeros((height, width, 3), np.uint8))
              for index, (height, width) in enumerate([(20, 40), (30, 60), (50, 80), (40, 30)])]
    transforms = grid_transforms(frames, 2, 2, overlap_pct=25)
    assert [(matrix[0, 2], matrix[1, 2]) for matrix in transforms] == [
        (0, 0), (60, 0), (0, 22.5), (60, 22.5)]
    assert canvas_bounds(frames, transforms) == (0, 0, 120, 73)
    with pytest.raises(ValueError):
        grid_transforms(frames, 2, 3)


def test_workspace_grid_requires_apply_and_preserves_preview_edits(qapp):
    from PyQt6.QtWidgets import QWidget
    from dialogs.mosaic import MosaicDialog

    parent = QWidget()
    dialog = MosaicDialog(parent)
    frames = [make_frame(str(index), np.full((20, 40, 3), index * 40, np.uint8))
              for index in range(4)]
    try:
        dialog._append_frames(frames)
        dialog.list.setCurrentRow(1)
        dialog.x_spin.setValue(123)
        dialog.opacity.setValue(40)
        dialog.grid_rows.setValue(2)
        dialog.grid_order.setCurrentIndex(dialog.grid_order.findData('bycolumn'))
        dialog.grid_direction.setCurrentIndex(dialog.grid_direction.findData('zigzag'))
        dialog.grid_corner.setCurrentIndex(dialog.grid_corner.findData('bottomright'))
        dialog.grid_overlap.setValue(25)
        assert dialog._current_transforms()[1][0, 2] == 123
        dialog.grid_apply.click()
        assert [(matrix[0, 2], matrix[1, 2]) for matrix in dialog._current_transforms()] == [
            (30, 15), (30, 0), (0, 0), (0, 15)]
        assert dialog.items[1].opacity() == pytest.approx(0.4)
        dialog.list.setCurrentRow(1)
        dialog.x_spin.setValue(91)
        dialog._append_frames([make_frame('extra', np.zeros((20, 40, 3), np.uint8))])
        assert not dialog.grid_apply.isEnabled()
        assert dialog._current_transforms()[1][0, 2] == 91
        assert dialog.items[1].opacity() == pytest.approx(0.4)
        dialog.list.setCurrentRow(4)
        dialog._remove_selected()
        assert dialog.grid_apply.isEnabled()
        assert dialog._current_transforms()[1][0, 2] == 91
        assert dialog.items[1].opacity() == pytest.approx(0.4)
    finally:
        dialog.close()
        parent.close()


def test_canvas_deselection_clears_remove_target_and_delete_removes_selected(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QWidget
    from dialogs.mosaic import MosaicDialog

    parent = QWidget()
    dialog = MosaicDialog(parent)
    try:
        dialog._append_frames([make_frame('one', np.zeros((20, 40, 3), np.uint8)),
                               make_frame('two', np.zeros((20, 40, 3), np.uint8))])
        dialog.scene.clearSelection()
        assert dialog.list.currentRow() == -1
        assert not dialog.remove.isEnabled() and not dialog.x_spin.isEnabled()
        QTest.keyClick(dialog.canvas.viewport(), Qt.Key.Key_Delete)
        assert len(dialog.frames) == 2
        dialog.items[0].setSelected(True)
        assert dialog.list.currentRow() == 0
        QTest.keyClick(dialog.canvas.viewport(), Qt.Key.Key_Delete)
        assert [frame.name for frame in dialog.frames] == ['two']
        QTest.keyClick(dialog.list, Qt.Key.Key_Delete)
        assert not dialog.frames and not dialog.save.isEnabled()
    finally:
        dialog.close()
        parent.close()


def _workspace_drop(qapp, target, urls):
    from PyQt6.QtCore import QMimeData, QPoint, QPointF, Qt
    from PyQt6.QtGui import QDragEnterEvent, QDropEvent

    mime = QMimeData()
    mime.setUrls(urls)
    enter = QDragEnterEvent(QPoint(5, 5), Qt.DropAction.CopyAction, mime,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    qapp.sendEvent(target, enter)
    drop = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    qapp.sendEvent(target, drop)
    return enter, drop


def _wait_workspace(qapp, dialog):
    import time

    deadline = time.monotonic() + 10
    while dialog.worker is not None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert dialog.worker is None


@pytest.mark.parametrize('target', ['list', 'canvas'])
def test_workspace_viewport_import_is_async_sorted_and_deduplicated(qapp, tmp_path, monkeypatch, target):
    import threading
    from PyQt6.QtCore import QUrl
    from PyQt6.QtWidgets import QMessageBox, QWidget
    from dialogs import mosaic

    folder = tmp_path / 'tiles'
    folder.mkdir()
    for name, value in [('img10.png', 100), ('img2.png', 20)]:
        assert imwrite_auto(str(folder / name), np.full((12, 16, 3), value, np.uint8))
    (folder / 'notes.txt').write_text('skip this')
    gui_thread = threading.get_ident()
    threads = []
    original = mosaic.make_frame

    def prepare(*args, **kwargs):
        threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(mosaic, 'make_frame', prepare)
    errors = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: errors.append(args))
    parent = QWidget()
    dialog = mosaic.MosaicDialog(parent)
    try:
        urls = [QUrl.fromLocalFile(str(folder)), QUrl.fromLocalFile(str(folder / 'img2.png'))]
        enter, drop = _workspace_drop(qapp, getattr(dialog, target).viewport(), urls)
        assert enter.isAccepted() and drop.isAccepted()
        _wait_workspace(qapp, dialog)
        assert not errors
        assert [frame.name for frame in dialog.frames] == ['img2.png', 'img10.png']
        assert threads and all(thread != gui_thread for thread in threads)
        _workspace_drop(qapp, getattr(dialog, target).viewport(), urls)
        _wait_workspace(qapp, dialog)
        assert len(dialog.frames) == 2
        enter, drop = _workspace_drop(qapp, getattr(dialog, target).viewport(),
                                      [QUrl('https://example.com/image.png')])
        assert not enter.isAccepted() and not drop.isAccepted()
        assert dialog.worker is None
    finally:
        dialog.close()
        parent.close()


def test_busy_workspace_freezes_inputs_and_cancel_preserves_positions(qapp, tmp_path):
    import threading
    from PyQt6.QtCore import QUrl
    from PyQt6.QtWidgets import QWidget
    from dialogs.mosaic import MosaicDialog

    parent = QWidget()
    dialog = MosaicDialog(parent)
    release = threading.Event()
    try:
        dialog._append_frames([make_frame('one', np.zeros((20, 40, 3), np.uint8)),
                               make_frame('two', np.zeros((20, 40, 3), np.uint8))])
        dialog.list.setCurrentRow(1)
        dialog.x_spin.setValue(71)
        before = dialog._current_transforms()

        def pending(cancel, progress):
            release.wait(5)
            return [make_frame('late', np.zeros((20, 40, 3), np.uint8))]

        dialog._run(pending, '', dialog._append_frames)
        worker = dialog.worker
        dialog._remove_selected()
        dialog._apply_grid()
        dialog._import_paths([str(tmp_path)])
        dialog.x_spin.setValue(100)
        enter, drop = _workspace_drop(qapp, dialog.canvas.viewport(), [QUrl.fromLocalFile(str(tmp_path))])
        assert not enter.isAccepted() and not drop.isAccepted()
        assert dialog.worker is worker and len(dialog.frames) == 2
        assert not dialog.grid_apply.isEnabled() and not dialog.save.isEnabled()
        dialog._cancel()
        release.set()
        _wait_workspace(qapp, dialog)
        assert len(dialog.frames) == 2
        for current, previous in zip(dialog._current_transforms(), before):
            np.testing.assert_array_equal(current, previous)
        assert dialog.x_spin.value() == 71
    finally:
        release.set()
        if dialog.worker is not None:
            dialog._cancel()
            _wait_workspace(qapp, dialog)
        dialog.close()
        parent.close()


def test_low_contrast_microscope_texture_recovers_magnification_and_rotation():
    rng = np.random.default_rng(771)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (700, 1200)).astype(np.float32), (0, 0), 1.2)
    scene = np.uint8(np.clip(110 + texture * 35, 0, 255))
    a, b = 1.035 * np.cos(0.025), 1.035 * np.sin(0.025)
    expected = np.array([[a, -b, 380], [b, a, -12], [0, 0, 1.]])
    second = cv2.warpAffine(scene, np.linalg.inv(expected)[:2], (760, 640))
    frames = [make_frame('anchor', scene[:, :800]), make_frame('magnified', second)]
    alignment = auto_align(frames)
    assert alignment.connected == (True, True)
    corners = np.array([[0, 0, 1], [759, 0, 1], [0, 639, 1], [759, 639, 1]]).T
    np.testing.assert_allclose(alignment.transforms[1] @ corners, expected @ corners, atol=1.5)
    assert np.hypot(*alignment.transforms[1][:2, 0]) == pytest.approx(1.035, abs=0.002)


def test_global_registration_uses_loop_controls_and_rejects_control_outliers(monkeypatch):
    from core import mosaic

    rng = np.random.default_rng(521)
    frames = [make_frame(str(i), np.zeros((400, 500, 3), np.uint8)) for i in range(4)]
    expected = [np.array([[1., 0, 120 * i], [0, 1., 30 * i], [0, 0, 1.]]) for i in range(3)]
    edges = {}
    for i, j in ((0, 1), (1, 2), (0, 2)):
        source = rng.uniform((40, 40), (350, 300), (100, 2))
        relative = np.linalg.inv(expected[i]) @ expected[j]
        target = mosaic._project(relative, source) + rng.normal(0, 0.1, source.shape)
        target[:5] += (25, -30)
        edges[i, j] = mosaic._ControlPoints(source, target)
    monkeypatch.setattr(mosaic, '_features', lambda *_: None)
    monkeypatch.setattr(mosaic, '_pair_match', lambda first, second, *_: edges.get((int(first.name), int(second.name))))
    placements = mosaic.initial_transforms(frames)
    placements[0][:2, 2] = (23, -17)
    placements[3][:2, 2] = (-100, 72)
    alignment = auto_align(frames, placements)
    assert alignment.connected == (True, True, True, False)
    np.testing.assert_array_equal(alignment.transforms[0], placements[0])
    np.testing.assert_array_equal(alignment.transforms[3], placements[3])
    for node in (1, 2):
        np.testing.assert_allclose(alignment.transforms[node], placements[0] @ expected[node], atol=0.2)
    assert max(match[3] for match in alignment.matches) < 0.3


def test_global_fit_can_be_cancelled_before_changing_placements():
    from core.mosaic_control import MosaicControlPoint

    frame = make_frame('one', np.zeros((80, 96, 3), np.uint8))
    original = [np.eye(3), np.array([[1., 0, 50], [0, 1., 0], [0, 0, 1.]])]
    before = [matrix.copy() for matrix in original]
    points = tuple(MosaicControlPoint(0, 1, (x + 50, y), (x, y))
                   for x, y in ((10, 10), (30, 10), (10, 50), (30, 50)))
    with pytest.raises(RegistrationCancelled):
        auto_align([frame, frame], original, control_points=points,
                   detect_features=False, should_cancel=lambda: True)
    for expected, actual in zip(before, original):
        np.testing.assert_array_equal(actual, expected)


def test_workspace_manual_scale_and_rotation_keep_dragged_position(qapp):
    from PyQt6.QtCore import QPointF
    from PyQt6.QtWidgets import QWidget
    from dialogs.mosaic import MosaicDialog

    parent = QWidget()
    dialog = MosaicDialog(parent)
    try:
        dialog._append_frames([make_frame('one', np.full((80, 96, 3), 100, np.uint16))])
        dialog.list.setCurrentRow(0)
        dialog.items[0].setPos(QPointF(31, -14))
        dialog.scale_spin.setValue(103.5)
        dialog.rotation_spin.setValue(1.25)
        matrix = dialog._current_transforms()[0]
        np.testing.assert_allclose(matrix[:2, 2], (31, -14))
        assert np.hypot(*matrix[:2, 0]) == pytest.approx(1.035)
        assert np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])) == pytest.approx(1.25)
        output, _ = render_mosaic(dialog.frames, dialog._current_transforms())
        assert output.dtype == np.uint16
        assert output.shape[1] > 96
        assert np.max(output) == 100
    finally:
        dialog.close()
        parent.close()


def test_large_manual_canvas_coordinates_preserve_native_pixels():
    image = np.arange(48 * 70 * 3, dtype=np.uint16).reshape(48, 70, 3)
    frame = make_frame('far from origin', image)
    matrix = np.array([[1., 0, 1000000000], [0, 1., -1000000000], [0, 0, 1.]])
    output, origin = render_mosaic([frame], [matrix], tile_size=32)
    assert origin == (1000000000, -1000000000)
    np.testing.assert_array_equal(output, image)
