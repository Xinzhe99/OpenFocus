"""Dragging folders onto the batch dialog queues them as stacks.

The dialog's folder list is the batch queue. Dragging a folder used to do
nothing there (only the main window handled drops), so the documented
"drag several stack folders" workflow silently required the picker instead.
"""
import os

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEventLoop, QMimeData, QPointF, Qt, QTimer, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent


def _make_stacks(root, count=3, frames=3):
    folders = []
    for s in range(count):
        folder = os.path.join(str(root), f"stack{s}")
        os.makedirs(folder)
        folders.append(folder)
        for i in range(frames):
            assert cv2.imwrite(
                os.path.join(folder, f"img{i}.png"),
                np.full((48, 64, 3), (s * 40 + i * 10) % 255, np.uint8))
    return folders


def _wait_import(dialog, qapp, timeout_ms=20000):
    loop = QEventLoop()

    def poll():
        if dialog._import_busy:
            QTimer.singleShot(20, poll)
        else:
            loop.quit()

    QTimer.singleShot(20, poll)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    qapp.processEvents()


def _drag(folders):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(path) for path in folders])
    return mime


def _drop_on(target, folders, app):
    """Deliver the real drag sequence: enter first, then drop.

    Qt drops a Drop event that was not preceded by an accepted DragEnter, so a
    bare dropEvent() call does not reproduce what a user does.
    """
    mime = _drag(folders)
    enter = QDragEnterEvent(
        QPointF(5, 5).toPoint(), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    app.sendEvent(target, enter)
    drop = QDropEvent(
        QPointF(5, 5), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    app.sendEvent(target, drop)
    return enter, drop


def _select_fusion(dialog, method='guided_filter'):
    """Pick a real fusion method; grouping is unavailable without one."""
    index = dialog.fusion_combo.findData(method)
    assert index >= 0, method
    dialog.fusion_combo.setCurrentIndex(index)
    assert dialog.fusion_combo.currentData() == method


def test_drop_queues_every_folder_in_order(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog

    folders = _make_stacks(tmp_path)
    dialog = BatchProcessingDialog(None)
    try:
        enter, _drop = _drop_on(dialog, folders, qapp)
        assert enter.isAccepted(), "a folder drag must be accepted"

        assert [os.path.basename(p) for p in dialog.folder_paths] == [
            "stack0", "stack1", "stack2"]
        assert dialog.folder_list.count() == 3
        _wait_import(dialog, qapp)
        assert all(not dialog.folder_list.item(i).icon().isNull()
                   for i in range(dialog.folder_list.count()))
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_drop_on_the_list_uses_the_same_queue(qapp, tmp_path):
    """The list does not consume drops; the dialog queues them instead."""
    from dialogs.batch import BatchProcessingDialog

    folders = _make_stacks(tmp_path, count=2)
    dialog = BatchProcessingDialog(None)
    try:
        _enter, drop = _drop_on(dialog.folder_list, folders, qapp)
        assert dialog.folder_paths == [os.path.normpath(p) for p in folders]
        assert drop.isAccepted()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_drop_switches_single_folder_mode_to_multiple(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog

    folders = _make_stacks(tmp_path, count=2)
    dialog = BatchProcessingDialog(None)
    try:
        dialog.rb_single_folder.setChecked(True)
        _drop_on(dialog, folders, qapp)
        assert dialog.rb_multiple_folders.isChecked(), (
            "dropped stack folders must be processed in multi-folder mode")
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_dropped_files_are_not_treated_as_folders(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog

    _make_stacks(tmp_path, count=1)
    loose = os.path.join(str(tmp_path), "loose.png")
    assert cv2.imwrite(loose, np.zeros((16, 16, 3), np.uint8))
    dialog = BatchProcessingDialog(None)
    try:
        mime = _drag([loose])
        enter = QDragEnterEvent(
            QPointF(5, 5).toPoint(), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        qapp.sendEvent(dialog, enter)
        assert not enter.isAccepted(), (
            "a loose image is not a stack folder; the queue must not gain it")
        assert dialog.folder_paths == []
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_dropping_the_same_folder_twice_keeps_one_row(qapp, tmp_path):
    from dialogs.batch import BatchProcessingDialog

    folders = _make_stacks(tmp_path, count=1)
    dialog = BatchProcessingDialog(None)
    try:
        for _ in range(2):
            _drop_on(dialog, folders, qapp)
        assert dialog.folder_paths == [os.path.normpath(folders[0])]
        assert dialog.folder_list.count() == 1
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize('enable_first', [False, True])
def test_grouped_folder_selection_keeps_bounded_previews(qapp, tmp_path, monkeypatch, enable_first):
    from dialogs.batch import BatchProcessingDialog
    import core.image_loader as loader

    folders = _make_stacks(tmp_path, count=1, frames=4)
    decoded = []
    original = loader.decode_image
    def decode(path):
        decoded.append(path)
        return original(path)
    monkeypatch.setattr(loader, 'decode_image', decode)
    dialog = BatchProcessingDialog(None)
    try:
        _select_fusion(dialog)
        if enable_first:
            dialog.chk_grouping.setChecked(True)
            dialog.group_size_spin.setValue(4)
        dialog.add_folders_to_list(folders)
        _wait_import(dialog, qapp)
        if not enable_first:
            dialog.chk_grouping.setChecked(True)
            dialog.group_size_spin.setValue(4)
        assert len(decoded) == 1
        assert not dialog.folder_list.item(0).icon().isNull()
        assert dialog.get_processing_settings()['group_size'] == 4
        assert dialog.group_overlap_spin.value() == 1
        dialog.group_size_spin.setValue(2)
        assert dialog.group_overlap_spin.value() == 0
        dialog.group_size_spin.setValue(4)
        assert dialog.group_overlap_spin.value() == 1
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_grouped_split_import_is_rejected_before_decoding(qapp, monkeypatch):
    from dialogs.batch import BatchProcessingDialog
    from PyQt6.QtWidgets import QMessageBox

    messages = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: messages.append(args))
    dialog = BatchProcessingDialog(None)
    try:
        _select_fusion(dialog)
        dialog.chk_grouping.setChecked(True)
        dialog.group_size_spin.setValue(4)
        assert not dialog._start_import('stack', folder_path='unused')
        assert messages and not dialog._import_busy
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_grouping_toggle_preserves_values_and_output_modes(qapp):
    from dialogs.batch import BatchProcessingDialog

    dialog = BatchProcessingDialog(None)
    try:
        _select_fusion(dialog)
        assert dialog.group_controls.isHidden()
        assert dialog.get_processing_settings()['group_size'] == 0
        dialog.chk_grouping.setChecked(True)
        assert not dialog.group_controls.isHidden()
        dialog.group_size_spin.setValue(7)
        dialog.group_overlap_spin.setValue(3)
        dialog.chk_grouping.setChecked(False)
        assert dialog.group_controls.isHidden()
        assert dialog.get_processing_settings()['group_size'] == 0
        dialog.chk_grouping.setChecked(True)
        assert dialog.get_processing_settings()['group_size'] == 7
        assert dialog.group_overlap_spin.value() == 3
        dialog.rb_custom_folder.setChecked(True)
        assert dialog.subfolder_controls.isHidden()
        assert not dialog.custom_folder_controls.isHidden()
        dialog.rb_same_folder.setChecked(True)
        assert dialog.custom_folder_controls.isHidden()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_single_folder_switch_preserves_imported_split_stack(qapp):
    from dialogs.batch import BatchProcessingDialog
    from locales import trans

    dialog = BatchProcessingDialog(None)
    try:
        _select_fusion(dialog)
        images = [('frame.tif', 123)]
        stacks = [images]
        dialog.single_folder_images_with_times = images
        dialog.single_folder_stacks = stacks
        dialog.single_folder_path_label.setText('selected stack')
        dialog.preview_label.setText('split preview')
        dialog.rb_single_folder.setChecked(True)
        assert not dialog.chk_grouping.isEnabled()
        assert not dialog.chk_grouping.isChecked()
        assert dialog.group_note.text() == trans.t('batch_group_split_error')
        assert dialog.single_folder_images_with_times is images
        assert dialog.single_folder_stacks is stacks
        assert dialog.single_folder_path_label.text() == 'selected stack'
        assert dialog.preview_label.text() == 'split preview'
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_batch_picker_text_uses_current_language(qapp, monkeypatch):
    from utils.file_dialogs import FileDialog
    from dialogs.batch import BatchProcessingDialog
    from locales import trans

    previous = trans.current_lang
    trans.current_lang = 'zh'
    dialog = BatchProcessingDialog(None)
    titles = []
    try:
        monkeypatch.setattr(FileDialog, 'getExistingDirectory',
                            lambda parent, title, path: titles.append(title) or '')
        dialog.add_single_folder()
        dialog.browse_output_folder()
        assert titles == [trans.t('batch_btn_select_split'), trans.t('batch_out_custom')]
        monkeypatch.setattr(dialog, 'update_single_folder_preview', lambda: None)
        folder = os.path.join('root', 'stack')
        dialog._apply_single_folder_import(folder, [('frame.tif', 123)], '')
        assert dialog.single_folder_path_label.text() == trans.t('import_folder_display').format('stack')
    finally:
        trans.current_lang = previous
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_open_stack_folder_accepts_a_typed_directory(qapp, tmp_path, monkeypatch):
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLineEdit, QWidget
    from controllers.source_manager import SourceManager
    from utils import settings_store

    target = tmp_path / "stack 文件夹"
    target.mkdir()
    saved, loaded, seen, invalid, errors = [], [], [], [], []
    monkeypatch.setattr(settings_store, "get_last_dialog_dir", lambda: str(tmp_path))
    monkeypatch.setattr(settings_store, "set_last_dialog_dir", saved.append)
    window = QWidget()
    manager = SourceManager(window)
    monkeypatch.setattr(manager, "load_image_stack", loaded.append)

    def choose():
        dialog = qapp.activeModalWidget()
        try:
            line = dialog.findChild(QLineEdit, "folderPathEdit")
            line.setText(str(tmp_path / "missing"))
            QTest.keyClick(line, Qt.Key.Key_Return)
            invalid.append((dialog.isVisible(), line.text()))
            line.setText(f'"{target}"')
            QTest.keyClick(line, Qt.Key.Key_Return)
            seen.append((dialog.isVisible(), dialog.directory().absolutePath()))
            dialog.accept()
        except Exception as exc:
            errors.append(exc)
            if dialog is not None:
                dialog.reject()

    try:
        QTimer.singleShot(0, choose)
        manager.prompt_and_load_stack()
        assert not errors
        assert invalid[0][0]
        assert invalid[0][1] == str(tmp_path / "missing")
        assert seen[0][0], "Enter should navigate without closing the picker"
        assert os.path.normcase(os.path.normpath(seen[0][1])) == os.path.normcase(str(target))
        assert [os.path.normcase(os.path.normpath(path)) for path in saved] == [os.path.normcase(str(target))]
        assert loaded == saved
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_batch_folder_picker_navigates_typed_path_on_enter(qapp, tmp_path):
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLineEdit
    from utils.file_dialogs import FileDialog
    from dialogs.batch import BatchProcessingDialog

    target = tmp_path / "stack 文件夹"
    target.mkdir()
    batch = BatchProcessingDialog(None)
    seen, errors = [], []

    def navigate():
        dialog = qapp.activeModalWidget()
        try:
            line = dialog.findChild(QLineEdit, "folderPathEdit")
            line.setText(str(target))
            QTest.keyClick(line, Qt.Key.Key_Return)
            seen.append((dialog.isVisible(), dialog.directory().absolutePath()))
        except Exception as exc:
            errors.append(exc)
        finally:
            if isinstance(dialog, FileDialog):
                dialog.reject()

    try:
        QTimer.singleShot(0, navigate)
        batch.add_folders()
        assert not errors
        assert seen[0][0], "Enter should navigate without closing the picker"
        assert os.path.normcase(os.path.normpath(seen[0][1])) == os.path.normcase(str(target))
    finally:
        batch.close()
        batch.deleteLater()
        qapp.processEvents()


def test_batch_fusion_noop_entry_is_no_fusion_not_no_alignment(qapp):
    """The fusion dropdown's no-op entry must not read like registration's."""
    from dialogs.batch import BatchProcessingDialog
    from locales import trans

    dialog = BatchProcessingDialog(None)
    try:
        fusion_none = dialog.fusion_combo.findData('none')
        assert fusion_none >= 0
        assert dialog.fusion_combo.itemText(fusion_none) == trans.t('radio_no_fusion')
        assert dialog.fusion_combo.itemData(
            fusion_none, Qt.ItemDataRole.ToolTipRole) == trans.t('radio_no_fusion_hint')
        align_none = dialog.align_combo.findData('none')
        assert align_none >= 0
        assert dialog.align_combo.itemText(align_none) == trans.t('check_align_none')
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_batch_fusion_only_options_follow_the_selected_method(qapp):
    """Alignment-only disables fusion-only inputs but keeps their drafts."""
    from dialogs.batch import BatchProcessingDialog
    from locales import trans

    dialog = BatchProcessingDialog(None)
    try:
        assert dialog.fusion_combo.currentData() == 'none'
        for box in (dialog.chk_base_correct, dialog.chk_sharpen_output,
                    dialog.chk_reverse_order):
            assert not box.isEnabled()
            assert box.toolTip() == trans.t('radio_no_fusion_hint')
        dialog.align_combo.setCurrentIndex(dialog.align_combo.findData('ecc'))
        assert dialog.chk_exposure_match.isEnabled(), (
            "registration still uses exposure matching as its brightness reference")

        _select_fusion(dialog)
        for box in (dialog.chk_base_correct, dialog.chk_sharpen_output,
                    dialog.chk_reverse_order):
            assert box.isEnabled()

        dialog.chk_reverse_order.setChecked(True)
        index = dialog.fusion_combo.findData('stackmffv2')
        dialog.fusion_combo.setCurrentIndex(index)
        assert not dialog.chk_reverse_order.isEnabled()
        assert dialog.chk_reverse_order.toolTip() == trans.t(
            'chk_reverse_order_locked_hint')
        assert dialog.chk_reverse_order.isChecked(), (
            "locking must not overwrite the stored preference")

        _select_fusion(dialog)
        assert dialog.chk_reverse_order.isEnabled()
        assert dialog.chk_reverse_order.isChecked()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_batch_grouping_conflicts_are_visible_before_start(qapp):
    from dialogs.batch import BatchProcessingDialog
    from locales import trans

    dialog = BatchProcessingDialog(None)
    try:
        assert not dialog.chk_grouping.isEnabled()
        assert dialog.group_note.text() == trans.t('batch_group_output_error')

        _select_fusion(dialog)
        assert dialog.chk_grouping.isEnabled()
        assert dialog.group_note.isHidden()

        dialog.chk_grouping.setChecked(True)
        assert dialog.chk_grouping.isChecked()
        assert not dialog.save_aligned_cb.isEnabled()
        assert not dialog.save_aligned_cb.isChecked()
        assert dialog.save_aligned_note.text() == trans.t(
            'batch_group_output_error')
        assert not dialog.save_aligned_note.isHidden()

        dialog.chk_grouping.setChecked(False)
        assert dialog.save_aligned_cb.isEnabled()
        dialog.save_aligned_cb.setChecked(True)
        assert not dialog.chk_grouping.isEnabled()
        assert dialog.group_note.text() == trans.t('batch_group_output_error')

        dialog.save_aligned_cb.setChecked(False)
        assert dialog.chk_grouping.isEnabled()

        dialog.chk_grouping.setChecked(True)
        dialog.rb_single_folder.setChecked(True)
        assert not dialog.chk_grouping.isChecked()
        assert not dialog.chk_grouping.isEnabled()
        assert dialog.group_note.text() == trans.t('batch_group_split_error')
        dialog.rb_multiple_folders.setChecked(True)
        assert dialog.chk_grouping.isEnabled()
        assert dialog.chk_grouping.isChecked(), (
            "switching modes must not discard the grouping preference")
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()




def test_batch_disables_unavailable_methods_without_dropping_selection(
        qapp, monkeypatch):
    from PyQt6.QtWidgets import QWidget
    import core.multi_focus_fusion as mff
    import ui.algorithm_combo as algorithm_combo
    from dialogs.batch import BatchProcessingDialog
    from locales import trans

    monkeypatch.setattr(mff, 'is_ai_fusion_available', lambda: False)
    monkeypatch.setattr(algorithm_combo, '_package_available', lambda _name: False)
    parent = QWidget()
    parent.current_fusion_method = lambda: 'stackmffv3'
    parent.current_alignment_method = lambda: 'none'
    parent.stack_reverse_order = False
    parent.sharpen_output = False
    dialog = BatchProcessingDialog(parent)
    try:
        assert dialog.fusion_combo.currentData() == 'stackmffv3'
        for method in ('stackmff', 'stackmffv2', 'stackmffv3'):
            index = dialog.fusion_combo.findData(method)
            assert index >= 0
            assert not dialog.fusion_combo.model().item(index).isEnabled()
            assert dialog.fusion_combo.itemData(
                index, Qt.ItemDataRole.ToolTipRole) == trans.t(
                    'msg_stackmff_unavailable_text')
        dtcwt = dialog.fusion_combo.findData('dtcwt')
        assert not dialog.fusion_combo.model().item(dtcwt).isEnabled()
        assert dialog.fusion_combo.itemData(
            dtcwt, Qt.ItemDataRole.ToolTipRole) == trans.t('env_dtcwt_unavailable')
        guided = dialog.fusion_combo.findData('guided_filter')
        assert dialog.fusion_combo.model().item(guided).isEnabled()
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.deleteLater()
        qapp.processEvents()
