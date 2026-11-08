"""Real Qt point/mask interactions, to run after production core integration."""

import numpy as np
import pytest
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QStyle, QStyleOptionViewItem

from core.mosaic import MosaicFrame
from core.mosaic_control import MosaicControlPoint
from dialogs.mosaic_control_points import MosaicControlPointsDialog
from dialogs.mosaic_mask import MosaicMaskDialog


@pytest.fixture(scope='module')
def qt_app():
    application = QApplication.instance() or QApplication([])
    yield application


def _frames(count=3):
    return tuple(MosaicFrame(str(index), np.zeros((160, 320, 3), np.uint8),
                             np.zeros((20, 40, 3), np.uint8)) for index in range(count))


def _click_native(view, point):
    viewport_point = view.mapFromScene(QPointF(*point))
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=viewport_point)
    # Screen pixel rounding bounds the expected source-coordinate tolerance.
    return 1.1 / abs(view.transform().m11())


def test_scaled_and_zoomed_source_clicks_add_native_manual_pair(qt_app):
    frames = _frames()
    dialog = MosaicControlPointsDialog(frames, (np.eye(3),) * len(frames), ())
    dialog.show()
    qt_app.processEvents()
    dialog.preview_a.scale(1.4, 1.4)
    dialog.preview_b.scale(1.8, 1.8)
    a, b = (110.0, 70.0), (125.0, 85.0)
    tolerance_a = _click_native(dialog.preview_a, a)
    tolerance_b = _click_native(dialog.preview_b, b)
    assert dialog.add_pair.isEnabled()
    QTest.mouseClick(dialog.add_pair, Qt.MouseButton.LeftButton)
    assert len(dialog.points) == 1
    cp = dialog.points[0]
    assert cp.point_a == pytest.approx(a, abs=tolerance_a)
    assert cp.point_b == pytest.approx(b, abs=tolerance_b)
    assert (cp.frame_a, cp.frame_b, cp.source, cp.weight) == (0, 1, 'manual', 2.0)
    assert any(marker[1] for marker in dialog.preview_a._markers)
    dialog.close()


def test_checkbox_coordinate_and_delete_controls_replace_immutable_records(qt_app):
    points = tuple(MosaicControlPoint(0, 1, (20.0 + i, 30), (30.0 + i, 30)) for i in range(10))
    points += (MosaicControlPoint(0, 2, (100, 80), (150, 80)),)
    dialog = MosaicControlPointsDialog(_frames(), (np.eye(3),) * 3, points)
    dialog.show()
    qt_app.processEvents()
    emitted = []
    dialog.pointsChanged.connect(emitted.append)
    item = dialog.table.item(0, 0)
    option = QStyleOptionViewItem()
    option.initFrom(dialog.table)
    option.rect = dialog.table.visualItemRect(item)
    option.features = QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
    option.checkState = Qt.CheckState.Checked
    checkbox = dialog.table.style().subElementRect(QStyle.SubElement.SE_ItemViewItemCheckIndicator, option, dialog.table)
    QTest.mouseClick(dialog.table.viewport(), Qt.MouseButton.LeftButton, pos=checkbox.center())
    assert not dialog.points[0].enabled
    assert points[0].enabled
    dialog.table.item(0, 3).setText('22.5, 33.25')
    assert dialog.points[0].point_a == (22.5, 33.25)
    assert points[0].point_a == (20.0, 30.0)
    dialog.table.item(0, 2).setText('2')
    assert dialog.points[0].frame_b == 2
    QTest.mouseClick(dialog.delete_nonadjacent, Qt.MouseButton.LeftButton)
    assert len(dialog.points) == 9
    assert all(abs(cp.frame_a - cp.frame_b) == 1 for cp in dialog.points)
    QTest.mouseClick(dialog.delete_worst, Qt.MouseButton.LeftButton)
    assert len(dialog.points) == 8
    dialog.table.selectRow(0)
    QTest.mouseClick(dialog.delete_selected, Qt.MouseButton.LeftButton)
    assert len(dialog.points) == 7
    assert emitted[-1] == dialog.points
    dialog.close()


def test_busy_close_requests_cancel_and_waits_for_parent(qt_app):
    cp = MosaicControlPoint(0, 1, (20, 30), (20, 30))
    dialog = MosaicControlPointsDialog(_frames(), (np.eye(3),) * 3, (cp,))
    cancelled = []
    dialog.cancelRequested.connect(lambda: cancelled.append(True))
    dialog.show()
    qt_app.processEvents()
    dialog.set_busy(True)
    assert not dialog.table.isEnabled()
    assert not dialog.optimize.isEnabled()
    dialog.close()
    qt_app.processEvents()
    assert cancelled == [True]
    assert dialog.isVisible()
    dialog.set_busy(False)
    assert not dialog.isVisible()


def test_mask_brush_keeps_native_radius_and_thumbnail_memory_then_cancel(qt_app):
    preview = np.zeros((100, 150, 3), np.uint8)
    before = preview.copy()
    dialog = MosaicMaskDialog(preview, (200000, 300000))
    dialog.show()
    qt_app.processEvents()
    dialog.canvas.scale(1.3, 1.3)
    dialog.radius.setValue(1200)
    point = (135000.0, 95000.0)
    tolerance = _click_native(dialog.canvas, point)
    assert dialog.strokes
    x, y, radius, include = dialog.strokes[0]
    assert (x, y) == pytest.approx(point, abs=tolerance)
    assert radius == 1200 and include is False
    assert (dialog._overlay.height(), dialog._overlay.width()) == preview.shape[:2]
    assert dialog._overlay.sizeInBytes() == preview.shape[0] * preview.shape[1] * 4
    QTest.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel), Qt.MouseButton.LeftButton)
    assert dialog.result() == QDialog.DialogCode.Rejected
    np.testing.assert_array_equal(preview, before)


def test_mask_clear_include_and_apply_return_pending_edit(qt_app):
    preview = np.zeros((80, 120, 3), np.uint8)
    base = np.zeros((80, 120), np.uint8)
    before = base.copy()
    dialog = MosaicMaskDialog(preview, (800, 1200), base)
    dialog.show()
    qt_app.processEvents()
    QTest.mouseClick(dialog.clear_button, Qt.MouseButton.LeftButton)
    assert dialog.clear_base and dialog.strokes == []
    QTest.mouseClick(dialog.include, Qt.MouseButton.LeftButton)
    dialog.radius.setValue(30)
    _click_native(dialog.canvas, (550.0, 350.0))
    assert dialog.strokes[0][2:] == (30.0, True)
    QTest.mouseClick(dialog.buttons.button(QDialogButtonBox.StandardButton.Apply), Qt.MouseButton.LeftButton)
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.clear_base
    np.testing.assert_array_equal(base, before)


def test_adding_multiple_control_pairs_keeps_uncaptured_editor_interactive(qapp):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QWidget
    from core.mosaic import make_frame
    from dialogs.mosaic import MosaicDialog

    parent = QWidget()
    dialog = MosaicDialog(parent)
    dialog._append_frames([make_frame(str(index), np.zeros((80, 96, 3), np.uint8))
                           for index in range(2)])
    dialog.show()
    failures = []

    def add_pairs():
        editor = dialog._cp_dialog
        try:
            for point in ((15, 15), (65, 55)):
                _click_native(editor.preview_a, point)
                _click_native(editor.preview_b, point)
                QTest.mouseClick(editor.add_pair, Qt.MouseButton.LeftButton)
                assert editor.preview_a.isEnabled() and editor.preview_b.isEnabled()
            assert len(editor.points) == 2
        except Exception as error:
            failures.append(error)
        finally:
            editor.reject()

    try:
        QTimer.singleShot(0, add_pairs)
        QTest.mouseClick(dialog.cp_button, Qt.MouseButton.LeftButton)
        assert not failures, str(failures[0]) if failures else ''
        assert len(dialog.control_points) == 2
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.close()
        parent.deleteLater()
        qapp.processEvents()


def test_preview_fit_ignores_hidden_placements_and_refits_when_restored(qapp):
    from PyQt6.QtWidgets import QWidget
    from core.mosaic import make_frame
    from dialogs.mosaic import MosaicDialog
    from utils.image_utils import cv2_to_pixmap

    parent = QWidget()
    dialog = MosaicDialog(parent)
    try:
        dialog._append_frames([make_frame('source', np.full((80, 96, 3), 100, np.uint8))])
        dialog.items[0].setPos(5000, 3000)
        dialog.show()
        qapp.processEvents()
        dialog.items[0].setVisible(False)
        dialog._preview_item = dialog.scene.addPixmap(
            cv2_to_pixmap(np.full((40, 60, 3), 220, np.uint8)))
        dialog._preview_item.setPos(-250, -100)
        dialog._fit()
        visible_center = dialog.canvas.mapFromScene(dialog._preview_item.sceneBoundingRect().center())
        center = dialog.canvas.viewport().rect().center()
        assert abs(visible_center.x() - center.x()) <= 2
        assert abs(visible_center.y() - center.y()) <= 2
        dialog._show_placement()
        source_center = dialog.canvas.mapFromScene(dialog.items[0].sceneBoundingRect().center())
        assert dialog.items[0].isVisible()
        assert abs(source_center.x() - center.x()) <= 2
        assert abs(source_center.y() - center.y()) <= 2
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.close()
        parent.deleteLater()
        qapp.processEvents()


def test_imported_mask_file_is_preserved_when_selected_as_export(qapp, tmp_path, monkeypatch):
    import time
    from PyQt6.QtWidgets import QMessageBox, QWidget
    from core.mosaic import make_frame
    from dialogs.mosaic import MosaicDialog
    from utils.image_utils import imwrite_auto

    parent = QWidget()
    dialog = MosaicDialog(parent)
    frame = make_frame('source', np.full((48, 64, 3), 160, np.uint8))
    dialog._append_frames([frame])
    editor = MosaicMaskDialog(frame.preview, frame.image.shape[:2], parent=dialog)
    path = tmp_path / 'mask.png'
    assert imwrite_auto(str(path), np.full((48, 64), 255, np.uint8))
    original = path.read_bytes()
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    monkeypatch.setattr('dialogs.mosaic.FileDialog.getSaveFileName',
                        lambda *args, **kwargs: (str(path), ''))

    def settled():
        deadline = time.monotonic() + 10
        while dialog.worker is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        assert dialog.worker is None

    try:
        dialog._import_mask(str(path), frame, editor)
        settled()
        np.testing.assert_array_equal(dialog._mask_imported, 255)
        dialog._save()
        settled()
        assert path.read_bytes() == original
        assert len(warnings) == 1
    finally:
        if dialog.worker is not None:
            dialog._cancel()
            settled()
        editor.close()
        editor.deleteLater()
        dialog.close()
        dialog.deleteLater()
        parent.close()
        parent.deleteLater()
        qapp.processEvents()


def test_invalid_camera_reports_error_and_control_points_can_reopen(qapp, monkeypatch):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QMessageBox, QWidget
    from core.mosaic import make_frame
    from dialogs.mosaic import MosaicDialog

    parent = QWidget()
    dialog = MosaicDialog(parent)
    warnings, opened = [], []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    try:
        dialog._append_frames([make_frame(str(i), np.zeros((48, 64, 3), np.uint8))
                               for i in range(2)])
        dialog.settings.projection.setCurrentIndex(1)
        dialog.settings._camera['a'].setValue(1)
        dialog._edit_control_points()
        assert len(warnings) == 1 and dialog._cp_dialog is None
        assert dialog.settings._camera['a'].value() == 1
        dialog.settings._camera['a'].setValue(0)

        def close_editor():
            opened.append(isinstance(dialog._cp_dialog, MosaicControlPointsDialog))
            dialog._cp_dialog.reject()

        QTimer.singleShot(0, close_editor)
        dialog._edit_control_points()
        assert opened == [True] and len(warnings) == 1
        assert len(dialog.frames) == 2
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.close()
        parent.deleteLater()
        qapp.processEvents()


def test_control_point_rejects_smaller_frame_and_reports_coordinate_bounds(qapp, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    frames = _frames(2) + (MosaicFrame('small', np.zeros((60, 80, 3), np.uint8),
                                      np.zeros((15, 20, 3), np.uint8)),)
    point = MosaicControlPoint(0, 1, (250, 100), (250, 100))
    dialog = MosaicControlPointsDialog(frames, (np.eye(3),) * 3, (point,))
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    try:
        dialog.table.item(0, 2).setText('2')
        assert dialog.points == (point,)
        assert all(value in warnings[-1] for value in ('250', '100', '80', '60'))
        dialog.table.item(0, 2).setText('99')
        assert dialog.points == (point,)
        assert all(value in warnings[-1] for value in ('99', '0', '2'))
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
