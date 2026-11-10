"""Behavioral coverage for output sharing: selection, payloads, files, reveal."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QMimeData, QPoint
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QAbstractItemView, QLabel, QSlider, QWidget

from controllers.export_manager import ExportManager, _read_image_any
from controllers.output_manager import OutputManager, OutputRecord
import controllers.output_manager as output_module
from locales import trans
from utils.file_manager import select_in_file_manager
from utils.image_utils import imwrite_auto
from utils.output_share import (clipboard_mime_matches, encode_jpeg_preview,
                                file_urls_mime_data, jpeg_mime_data,
                                set_clipboard_payload)
from widgets.output_list import OutputListWidget


@pytest.fixture(autouse=True)
def nonblocking_share_errors(isolated_clipboard, monkeypatch):

    def unexpected_error(*args, **kwargs):
        pytest.fail(f'Unexpected output-sharing error: {args[1:]}')

    monkeypatch.setattr(output_module, 'show_error_box', unexpected_error)


def test_file_urls_round_trip_on_qt_clipboard(tmp_path):
    """Use a separate offscreen Qt process rather than the desktop clipboard."""
    paths = [tmp_path / '结果 一.png', tmp_path / 'second image.png']
    code = '''
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from utils.output_share import file_urls_mime_data, set_clipboard_payload
app = QApplication([])
paths = [Path(value) for value in sys.argv[1:]]
assert set_clipboard_payload(file_urls_mime_data(paths))
mime = app.clipboard().mimeData()
assert [Path(url.toLocalFile()) for url in mime.urls()] == paths
assert mime.text().replace('\\r\\n', '\\n').splitlines() == [str(path) for path in paths]
'''
    result = subprocess.run([sys.executable, '-c', code, *(str(path) for path in paths)],
                            env=dict(os.environ, QT_QPA_PLATFORM='offscreen'),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


class RecordingLabels:
    """Snapshot-compatible label renderer that leaves a visible green mark."""

    def __init__(self):
        self.labels = []
        self.calls = []

    def snapshot(self):
        return self

    def prepare_bgr_image(self, target, image, index, output_id=None):
        self.calls.append((target, index, output_id))
        out = image.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        out[0:6, 0:6] = (0, 255, 0)
        return out


class Status:
    def __init__(self):
        self.messages = []

    def showMessage(self, message, *args):
        self.messages.append(message)


def host(qapp):
    window = QWidget()
    window.output_list = OutputListWidget(window)
    window.output_list.setGeometry(0, 0, 320, 240)
    window.output_list.setSelectionMode(
        QAbstractItemView.SelectionMode.ExtendedSelection)
    window.output_list.set_window(window)
    window.output_label = QLabel(window)
    window.lbl_result_info = QLabel(window)
    window.result_slider = QSlider(window)
    window.result_control_bar = QWidget(window)
    window.fusion_results = []
    window.fusion_result = None
    window.registration_results = []
    window.image_filenames = []
    window.image_source_paths = []
    window.label_manager = RecordingLabels()
    window.current_folder_path = ""
    window.scale_bar_enabled = False
    window.status = Status()
    window.statusBar = lambda: window.status
    window.output_manager = OutputManager(window)
    window.export_manager = ExportManager(window)
    return window


def add_record(window, image, label, *, kind="fusion", **attrs):
    record = OutputRecord.create(image, label, kind=kind)
    for name, value in attrs.items():
        setattr(record, name, value)
    if record.exported and record.exported_image is None and image is not None:
        # Mirror _mark_record_exported/add_batch_outputs for test fixtures.
        record.exported_image = image
    item = window.output_manager._insert_output(image, label, record)
    return item, record


def menu_action(menu, text):
    return next(action for action in menu.actions() if action.text() == text)


def share_actions(window, target):
    menu = window.output_manager._build_output_context_menu(target)
    share = menu_action(menu, trans.t("menu_share")).menu()
    return share.actions()


def item_position(qapp, window, item):
    window.show()
    qapp.processEvents()
    return window.output_list.visualItemRect(item).center()


def clipboard_mime(qapp):
    qapp.processEvents()
    return qapp.clipboard().mimeData()


def clipboard_text(qapp):
    mime = clipboard_mime(qapp)
    return mime.text() if mime is not None else ""


# ----------------------------------------------------------------------
# Menu and selection
# ----------------------------------------------------------------------
def test_right_click_keeps_multiselect_and_unselected_target_replaces_it(qapp):
    window = host(qapp)
    first, _ = add_record(window, np.zeros((8, 8, 3), np.uint8), "first")
    second, _ = add_record(window, np.zeros((8, 8, 3), np.uint8), "second")
    window.output_list.clearSelection()
    first.setSelected(True)
    second.setSelected(True)
    qapp.processEvents()

    target = window.output_manager._prepare_output_context_selection(
        item_position(qapp, window, first))
    assert target is first
    assert window.output_list.currentItem() is first
    assert {item.text() for item in window.output_list.selectedItems()} == {
        "first", "second"}

    window.output_list.clearSelection()
    first.setSelected(True)
    target = window.output_manager._prepare_output_context_selection(
        item_position(qapp, window, second))
    assert target is second
    assert [item.text() for item in window.output_list.selectedItems()] == [
        "second"]


def test_right_click_blank_does_not_share_the_current_row(qapp):
    window = host(qapp)
    add_record(window, np.zeros((8, 8, 3), np.uint8), "current")
    position = QPoint(315, 235)
    target = window.output_manager._prepare_output_context_selection(position)
    assert target is None
    actions = share_actions(window, target)
    assert [action.text() for action in actions] == [
        trans.t("share_copy_preview"), trans.t("share_copy_paths"),
        trans.t("share_reveal"), trans.t("share_private_link")]
    assert not any(action.isEnabled() for action in actions[:-1])
    assert not actions[-1].isEnabled()
    qapp.clipboard().setText("sentinel")
    window.output_manager.share_copy_paths(None)
    assert qapp.clipboard().text() == "sentinel"


def test_share_menu_state_follows_selection_and_record_kind(qapp):
    window = host(qapp)
    first, _ = add_record(window, np.zeros((8, 8, 3), np.uint8), "stack_aligned")
    second, _ = add_record(window, np.zeros((8, 8, 3), np.uint8), "second")
    aligned, _ = add_record(window, None, "folder", kind="aligned",
                            exported=True, export_path="")

    window.output_list.clearSelection()
    first.setSelected(True)
    second.setSelected(True)
    actions = share_actions(window, first)
    assert [action.isEnabled() for action in actions] == [
        True, True, True, False]

    window.output_list.clearSelection()
    first.setSelected(True)
    aligned.setSelected(True)
    actions = share_actions(window, first)
    assert [action.isEnabled() for action in actions] == [
        True, False, True, False]

    window.output_list.clearSelection()
    aligned.setSelected(True)
    actions = share_actions(window, aligned)
    assert [action.isEnabled() for action in actions] == [
        False, False, False, False]


def test_save_as_and_retouch_follow_the_right_click_target(qapp):
    window = host(qapp)
    first, first_record = add_record(
        window, np.zeros((8, 8, 3), np.uint8), "first",
        context=SimpleNamespace(source_count=1))
    second, _ = add_record(window, np.zeros((8, 8, 3), np.uint8), "second")
    window.output_list.clearSelection()
    first.setSelected(True)
    second.setSelected(True)
    saved, retouched = [], []
    window.output_manager.save_output_image_as = saved.append
    window.show_retouch_dialog = retouched.append

    actions = window.output_manager._build_output_context_menu(first).actions()
    next(action for action in actions
         if action.text() == trans.t("action_save_as")).trigger()
    next(action for action in actions
         if action.text() == trans.t("action_retouch")).trigger()
    assert saved == [first] and retouched == [first]
    assert first_record.retouchable
    assert {item.text() for item in window.output_list.selectedItems()} == {
        "first", "second"}


def test_batch_image_with_an_aligned_like_name_is_still_shareable(qapp):
    window = host(qapp)
    item, _ = add_record(window, np.zeros((8, 8, 3), np.uint8), "stack_aligned",
                         kind="batch")
    actions = share_actions(window, item)
    assert all(action.isEnabled() for action in actions[:3])


# ----------------------------------------------------------------------
# JPEG clipboard
# ----------------------------------------------------------------------
def test_copy_preview_places_real_jpeg_bytes_and_a_qimage(qapp):
    window = host(qapp)
    image = np.full((48, 64, 3), (200, 40, 20), np.uint8)
    image[8:24, 8:24] = (0, 0, 255)
    item, record = add_record(window, image, "preview")

    window.output_manager.share_copy_preview(item)
    mime = clipboard_mime(qapp)
    assert mime.hasFormat("image/jpeg")
    payload = bytes(mime.data("image/jpeg"))
    assert payload[:2] == b"\xff\xd8" and payload[-2:] == b"\xff\xd9"
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == image.shape
    assert int(decoded[16, 16, 2]) > int(decoded[16, 16, 0])
    assert int(decoded[40, 50, 0]) > int(decoded[40, 50, 2])

    assert mime.hasImage()
    qimage = mime.imageData()
    assert isinstance(qimage, QImage) and not qimage.isNull()
    assert (qimage.width(), qimage.height()) == (64, 48)
    assert window.status.messages == [trans.t("msg_share_preview_copied")]
    assert not record.exported and record.export_path == ""


def test_copy_preview_applies_annotations_once_and_respects_baked_output(qapp):
    window = host(qapp)
    image = np.full((40, 40, 3), 90, np.uint8)
    item, record = add_record(window, image, "annotated")
    window.output_manager.share_copy_preview(item)
    payload = bytes(clipboard_mime(qapp).data("image/jpeg"))
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert int(decoded[2, 2, 1]) > 140
    assert window.label_manager.calls == [("output", 0, record.output_id)]

    window.label_manager.calls.clear()
    record.annotations_baked = True
    window.output_manager.share_copy_preview(item)
    payload = bytes(clipboard_mime(qapp).data("image/jpeg"))
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert int(decoded[2, 2, 1]) < 120
    assert window.label_manager.calls == []


def test_copy_preview_freezes_state_and_ignores_reentry(qapp, monkeypatch):
    window = host(qapp)
    image = np.full((40, 40, 3), 90, np.uint8)
    item, record = add_record(window, image, "annotated")
    calls = []

    def fake_worker(parent, title, operation, *, show_progress=True):
        calls.append(show_progress)
        # A state change while the pixels encode must not alter this payload.
        record.annotations_baked = True
        # A second request must not race the first clipboard write.
        window.output_manager.share_copy_preview(item)
        return operation()

    monkeypatch.setattr(output_module, "run_pixel_worker", fake_worker)
    window.output_manager.share_copy_preview(item)

    assert calls == [True]
    assert record.annotations_baked
    assert window.label_manager.calls == [("output", 0, record.output_id)]
    payload = bytes(clipboard_mime(qapp).data("image/jpeg"))
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert int(decoded[2, 2, 1]) > 140
    assert window.status.messages == [trans.t("msg_share_preview_copied")]


def test_copy_preview_rejected_by_clipboard_reports_failure(qapp, monkeypatch):
    window = host(qapp)
    item, _ = add_record(window, np.zeros((40, 40, 3), np.uint8), "preview")
    errors = []
    monkeypatch.setattr(output_module, "show_error_box",
                        lambda *args: errors.append(args))
    monkeypatch.setattr(output_module, "set_clipboard_payload",
                        lambda mime: False)
    qapp.clipboard().setText("sentinel")
    window.output_manager.share_copy_preview(item)
    assert errors and errors[0][1] == trans.t("msg_share_failed_title")
    assert errors[0][2] == trans.t("msg_share_clipboard_failed")
    assert qapp.clipboard().text() == "sentinel"
    assert not window.status.messages
    assert not window.output_manager._share_preview_active


def test_copy_preview_handles_grayscale_and_transparency(qapp):
    window = host(qapp)
    gray = np.tile(np.arange(32, dtype=np.uint8), (24, 1))
    gray_item, _ = add_record(window, gray, "gray")
    window.output_manager.share_copy_preview(gray_item)
    payload = bytes(clipboard_mime(qapp).data("image/jpeg"))
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (24, 32, 3)

    alpha = np.zeros((24, 32, 4), np.uint8)
    alpha[:, :, :3] = 200
    alpha[:, :, 3] = 0
    alpha_item, _ = add_record(window, alpha, "alpha")
    window.output_manager.share_copy_preview(alpha_item)
    payload = bytes(clipboard_mime(qapp).data("image/jpeg"))
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (24, 32, 3)
    assert int(decoded.mean()) > 80


def test_copy_preview_failure_does_not_claim_success(qapp, monkeypatch):
    window = host(qapp)
    item, _ = add_record(window, np.zeros((0, 0, 3), np.uint8), "empty")
    errors = []
    monkeypatch.setattr(output_module, "show_error_box",
                        lambda *args: errors.append(args))
    qapp.clipboard().setText("sentinel")
    window.output_manager.share_copy_preview(item)
    assert errors and errors[0][1] == trans.t("msg_share_failed_title")
    assert qapp.clipboard().text() == "sentinel"
    assert not window.status.messages
    assert not window.output_manager._share_preview_active


# ----------------------------------------------------------------------
# File URLs and persistent files
# ----------------------------------------------------------------------
def test_copy_paths_sets_multiple_file_urls_including_unicode(qapp, tmp_path):
    window = host(qapp)
    first_path = tmp_path / "结果 一.png"
    second_path = tmp_path / "second image.png"
    assert imwrite_auto(str(first_path), np.zeros((4, 4, 3), np.uint8), [])
    assert imwrite_auto(str(second_path), np.zeros((4, 4, 3), np.uint8), [])
    first, _ = add_record(window, np.zeros((4, 4, 3), np.uint8), "one",
                          exported=True, export_path=str(first_path),
                          annotations_baked=True)
    second, _ = add_record(window, np.zeros((4, 4, 3), np.uint8), "two",
                           exported=True, export_path=str(second_path),
                           annotations_baked=True)
    window.output_list.clearSelection()
    first.setSelected(True)
    second.setSelected(True)

    window.output_manager.share_copy_paths(first)
    mime = clipboard_mime(qapp)
    assert mime.hasUrls()
    urls = mime.urls()
    assert {Path(url.toLocalFile()) for url in urls} == {
        first_path.resolve(), second_path.resolve()}
    assert set(clipboard_text(qapp).splitlines()) == {str(first_path.resolve()), str(second_path.resolve())}
    assert all(Path(url.toLocalFile()).is_file() for url in urls)
    assert window.status.messages[-1] == trans.t("msg_share_files_copied").format(n=2)


def test_clipboard_verification_detects_empty_or_different_payloads(tmp_path):
    path = tmp_path / "file.png"
    path.write_bytes(b"x")
    expected_urls = file_urls_mime_data([path])
    assert not clipboard_mime_matches(expected_urls, QMimeData())
    assert not clipboard_mime_matches(
        expected_urls, file_urls_mime_data([tmp_path / "other.png"]))
    expected_jpeg = jpeg_mime_data(encode_jpeg_preview(
        np.full((4, 4, 3), 30, np.uint8)))
    assert not clipboard_mime_matches(expected_jpeg, QMimeData())

    class RejectingClipboard:
        def setMimeData(self, mime):
            pass

        def mimeData(self):
            return QMimeData()

    assert not set_clipboard_payload(expected_urls, RejectingClipboard())


def test_copy_paths_rejected_by_clipboard_reports_failure(
        qapp, tmp_path, monkeypatch):
    window = host(qapp)
    saved = tmp_path / "saved.png"
    assert imwrite_auto(str(saved), np.zeros((4, 4, 3), np.uint8), [])
    item, _ = add_record(window, np.zeros((4, 4, 3), np.uint8), "saved",
                         exported=True, export_path=str(saved),
                         annotations_baked=True)
    errors = []
    monkeypatch.setattr(output_module, "show_error_box",
                        lambda *args: errors.append(args))
    monkeypatch.setattr(output_module, "set_clipboard_payload",
                        lambda mime: False)
    qapp.clipboard().setText("sentinel")
    window.output_manager.share_copy_paths(item)
    assert errors and errors[0][1] == trans.t("msg_share_failed_title")
    assert errors[0][2] == trans.t("msg_share_clipboard_failed")
    assert qapp.clipboard().text() == "sentinel"
    assert not window.status.messages


def test_copy_paths_single_selection_ignores_other_selected_history(qapp, tmp_path):
    window = host(qapp)
    only = tmp_path / "only.png"
    assert imwrite_auto(str(only), np.zeros((4, 4, 3), np.uint8), [])
    first, _ = add_record(window, np.zeros((4, 4, 3), np.uint8), "one",
                          exported=True, export_path=str(only),
                          annotations_baked=True)
    add_record(window, np.zeros((4, 4, 3), np.uint8), "two",
               exported=True, export_path=str(only), annotations_baked=True)
    window.output_list.clearSelection()
    first.setSelected(True)
    window.output_manager.share_copy_paths(first)
    assert [Path(url.toLocalFile()) for url in clipboard_mime(qapp).urls()] == [
        only.resolve()]


def test_unsaved_output_copy_path_does_not_open_picker_or_create_file(qapp, tmp_path, monkeypatch):
    window = host(qapp)
    item, record = add_record(window, np.full((12, 16, 3), 1200, np.uint16), 'unsaved')
    monkeypatch.setattr(output_module.FileDialog, 'getExistingDirectory',
                        lambda *args, **kwargs: pytest.fail('copying a path opened a picker'))
    qapp.clipboard().setText('sentinel')
    window.output_manager.share_copy_paths(item)
    assert clipboard_text(qapp) == 'sentinel'
    assert window.status.messages[-1] == trans.t('msg_share_save_first')
    assert not record.exported and not record.export_path
    assert not list(tmp_path.iterdir())


def test_mixed_saved_and_unsaved_selection_preserves_clipboard(qapp, tmp_path, monkeypatch):
    window = host(qapp)
    saved = tmp_path / 'saved.png'
    assert imwrite_auto(str(saved), np.zeros((4, 4, 3), np.uint8), [])
    first, _ = add_record(window, np.zeros((4, 4, 3), np.uint8), 'saved',
                          exported=True, export_path=str(saved))
    second, _ = add_record(window, np.zeros((4, 4, 3), np.uint8), 'unsaved')
    window.output_list.clearSelection()
    first.setSelected(True)
    second.setSelected(True)
    monkeypatch.setattr(output_module.FileDialog, 'getExistingDirectory',
                        lambda *args, **kwargs: pytest.fail('copying a path opened a picker'))
    qapp.clipboard().setText('sentinel')
    window.output_manager.share_copy_paths(first)
    assert clipboard_text(qapp) == 'sentinel'
    assert window.status.messages[-1] == trans.t('msg_share_save_first')


def test_save_as_copies_actual_written_unicode_path(qapp, tmp_path, monkeypatch):
    window = host(qapp)
    item, record = add_record(window, np.full((12, 16, 3), 1200, np.uint16), 'result')
    target = tmp_path / '融合 结果'
    monkeypatch.setattr(output_module.FileDialog, 'getSaveFileName',
                        lambda *args, **kwargs: (str(target), 'TIFF (*.tif *.tiff)'))
    window.output_manager.save_output_image_as(item)
    written = target.with_suffix('.tif')
    assert written.is_file() and record.export_path == str(written)
    assert _read_image_any(str(written)).dtype == np.uint16
    assert clipboard_text(qapp) == str(written.resolve())
    assert [Path(url.toLocalFile()) for url in clipboard_mime(qapp).urls()] == [written.resolve()]
    assert window.status.messages[-1] == trans.t('msg_saved_path_copied').format(path=str(written))


@pytest.mark.parametrize('cancel', [True, False])
def test_save_as_cancel_or_failure_does_not_change_clipboard(qapp, tmp_path, monkeypatch, cancel):
    window = host(qapp)
    item, record = add_record(window, np.zeros((8, 8, 3), np.uint8), 'result')
    monkeypatch.setattr(output_module.FileDialog, 'getSaveFileName',
                        lambda *args, **kwargs: ('' if cancel else str(tmp_path / 'fail.png'), ''))
    monkeypatch.setattr(output_module, 'show_error_box', lambda *args: None)
    if not cancel:
        monkeypatch.setattr(window.output_manager, '_write_record',
                            lambda *args: (_ for _ in ()).throw(OSError('write failed')))
    qapp.clipboard().setText('sentinel')
    window.output_manager.save_output_image_as(item)
    assert clipboard_text(qapp) == 'sentinel'
    assert not record.exported


def test_saved_file_is_retained_when_clipboard_is_unavailable(qapp, tmp_path, monkeypatch):
    from utils import output_share
    window = host(qapp)
    item, record = add_record(window, np.zeros((8, 8, 3), np.uint8), 'result')
    path = tmp_path / 'saved.png'
    monkeypatch.setattr(output_module.FileDialog, 'getSaveFileName',
                        lambda *args, **kwargs: (str(path), ''))
    monkeypatch.setattr(output_share, 'set_clipboard_payload', lambda mime: False)
    window.output_manager.save_output_image_as(item)
    assert path.is_file() and record.exported and record.export_path == str(path)
    assert trans.t('msg_share_clipboard_failed') in window.status.messages[-1]

def test_valid_exported_jpeg_is_reused_without_rewriting(
        qapp, tmp_path, monkeypatch):
    window = host(qapp)
    saved = tmp_path / "already.jpg"
    assert imwrite_auto(str(saved), np.full((8, 8, 3), 70, np.uint8),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
    before = saved.read_bytes()
    item, record = add_record(window, np.full((8, 8, 3), 70, np.uint8), "saved",
                              exported=True, export_path=str(saved))
    record.exported_image = record.image
    record.exported_annotations = window.output_manager._annotation_state(record)
    monkeypatch.setattr(
        output_module.FileDialog, "getExistingDirectory",
        lambda *args, **kwargs: pytest.fail("valid file must be reused"))

    window.output_manager.share_copy_paths(item)
    assert saved.read_bytes() == before
    assert [Path(url.toLocalFile()) for url in clipboard_mime(qapp).urls()] == [
        saved.resolve()]


def test_copy_path_uses_last_saved_file_and_reports_missing_file(qapp, tmp_path, monkeypatch):
    window = host(qapp)
    saved = tmp_path / 'saved.png'
    assert imwrite_auto(str(saved), np.zeros((8, 8, 3), np.uint8), [])
    original = saved.read_bytes()
    item, record = add_record(window, np.full((8, 8, 3), 10, np.uint8), 'changed',
                              exported=True, export_path=str(saved))
    record.annotations_changed = True
    monkeypatch.setattr(output_module.FileDialog, 'getExistingDirectory',
                        lambda *args, **kwargs: pytest.fail('copying a path opened a picker'))
    window.output_manager.share_copy_paths(item)
    assert clipboard_text(qapp) == str(saved.resolve())
    assert saved.read_bytes() == original
    saved.unlink()
    qapp.clipboard().setText('sentinel')
    window.output_manager.share_copy_paths(item)
    assert clipboard_text(qapp) == 'sentinel'
    assert window.status.messages[-1] == trans.t('msg_share_save_first')


def test_aligned_folder_is_never_copied_as_a_fusion_file(qapp, tmp_path):
    window = host(qapp)
    folder = tmp_path / "stack_aligned"
    folder.mkdir()
    item, _ = add_record(window, None, "stack_aligned", kind="aligned",
                         exported=True, export_path=str(folder),
                         annotations_baked=True)
    qapp.clipboard().setText("sentinel")
    window.output_manager.share_copy_paths(item)
    assert qapp.clipboard().text() == "sentinel"
    window.output_manager.share_copy_preview(item)
    assert qapp.clipboard().text() == "sentinel"


# ----------------------------------------------------------------------
# Reveal
# ----------------------------------------------------------------------
def test_disappearing_exported_file_is_not_reused(qapp, tmp_path, monkeypatch):
    window = host(qapp)
    saved = tmp_path / "vanishing.png"
    assert imwrite_auto(str(saved), np.zeros((4, 4, 3), np.uint8), [])
    item, record = add_record(
        window, np.zeros((4, 4, 3), np.uint8), "saved", exported=True,
        export_path=str(saved), annotations_baked=True)
    getsize = output_module.os.path.getsize
    monkeypatch.setattr(output_module.os.path, "getsize", lambda path:
                        (_ for _ in ()).throw(OSError("file disappeared"))
                        if str(path) == str(saved) else getsize(path))
    assert window.output_manager._share_persistent_path(record) == ""
    errors = []
    monkeypatch.setattr(output_module, "show_error_box",
                        lambda *args: errors.append(args))
    window.output_manager._share_temp_dir = str(tmp_path)
    opened = []
    monkeypatch.setattr(output_module, 'select_in_file_manager', lambda path: opened.append(path) or True)
    window.output_manager.share_reveal(item)
    assert not errors
    assert record.export_path == str(saved)
    assert len(opened) == 1 and Path(opened[0]) != saved.resolve()


def test_reveal_surfaces_a_file_manager_exception(qapp, tmp_path, monkeypatch):
    window = host(qapp)
    saved = tmp_path / "saved.png"
    assert imwrite_auto(str(saved), np.zeros((4, 4, 3), np.uint8), [])
    item, _ = add_record(window, np.zeros((4, 4, 3), np.uint8), "saved",
                         exported=True, export_path=str(saved),
                         annotations_baked=True)
    errors = []
    monkeypatch.setattr(output_module, "show_error_box",
                        lambda *args: errors.append(args))
    monkeypatch.setattr(
        output_module, "select_in_file_manager",
        lambda path: (_ for _ in ()).throw(RuntimeError("no file manager")))
    window.output_manager.share_reveal(item)
    assert errors and errors[0][1] == trans.t("msg_share_failed_title")
    assert errors[0][2] == trans.t("msg_share_reveal_failed")
    assert "no file manager" in errors[0][3]


def test_linux_selection_falls_back_only_when_selection_fails(
        tmp_path, monkeypatch):
    path = tmp_path / "结果 一.png"
    path.write_bytes(b"x")
    monkeypatch.setattr(
        "utils.file_manager.shutil.which",
        lambda name: "/usr/bin/dbus-send" if name == "dbus-send" else None)

    fallback = []
    opener = lambda directory: fallback.append(directory) or True
    assert select_in_file_manager(
        str(path), platform="linux",
        runner=lambda arguments, **kwargs: SimpleNamespace(returncode=0),
        opener=opener)
    assert fallback == []

    assert select_in_file_manager(
        str(path), platform="linux",
        runner=lambda arguments, **kwargs: SimpleNamespace(returncode=1),
        opener=opener)
    assert fallback == [str(tmp_path)]


def test_reveal_opens_the_persistent_file_for_the_target(qapp, tmp_path, monkeypatch):
    window = host(qapp)
    saved = tmp_path / "saved.png"
    assert imwrite_auto(str(saved), np.zeros((4, 4, 3), np.uint8), [])
    item, _ = add_record(window, np.zeros((4, 4, 3), np.uint8), "saved",
                         exported=True, export_path=str(saved),
                         annotations_baked=True)
    called = []
    monkeypatch.setattr(output_module, "select_in_file_manager",
                        lambda path: called.append(path) or True)
    window.output_manager.share_reveal(item)
    assert called == [str(saved)]


def test_reveal_opens_temp_for_unsaved_result_without_marking_it_saved(
        qapp, tmp_path, monkeypatch):
    window = host(qapp)
    window.output_manager._share_temp_dir = str(tmp_path)
    item, record = add_record(window, np.zeros((6, 8, 3), np.uint8), "reveal")
    monkeypatch.setattr(output_module.FileDialog, "getExistingDirectory",
                        lambda *args, **kwargs: pytest.fail("no folder picker"))
    called = []
    monkeypatch.setattr(output_module, "select_in_file_manager",
                        lambda path: called.append(path) or True)
    window.output_manager.share_reveal(item)
    expected = str(tmp_path / "reveal.png")
    assert called == [expected] and Path(expected).is_file()
    assert not record.exported and record.export_path == ""


def test_reveal_failure_is_visible(qapp, tmp_path, monkeypatch):
    window = host(qapp)
    saved = tmp_path / "saved.png"
    assert imwrite_auto(str(saved), np.zeros((4, 4, 3), np.uint8), [])
    item, _ = add_record(window, np.zeros((4, 4, 3), np.uint8), "saved",
                         exported=True, export_path=str(saved), annotations_baked=True)
    errors = []
    monkeypatch.setattr(output_module, "show_error_box", lambda *args: errors.append(args))
    monkeypatch.setattr(output_module, "select_in_file_manager", lambda path: False)
    window.output_manager.share_reveal(item)
    assert errors and errors[0][1] == trans.t("msg_share_failed_title")
    assert errors[0][2] == trans.t("msg_share_reveal_failed")


def test_encode_jpeg_preview_rejects_non_image_values():
    with pytest.raises(ValueError):
        encode_jpeg_preview(np.zeros((0, 0, 3), np.uint8))
    with pytest.raises(ValueError):
        encode_jpeg_preview(np.zeros(8, np.uint8))


def test_jpeg_preview_freezes_watermark_before_background_work(qapp, monkeypatch):
    from utils import watermark
    from utils.image_utils import to_display_uint8

    window = host(qapp)
    image = np.full((320, 480, 3), 12345, np.uint16)
    before = image.copy()
    config = watermark.WatermarkConfig(text='Frozen creator', size_percent=6.0)
    item, record = add_record(window, image, 'watermarked',
                              watermark_config=config.to_dict(), watermark_initialized=True)
    labelled = window.label_manager.prepare_bgr_image('output', image, 0, output_id=record.output_id)
    expected = encode_jpeg_preview(to_display_uint8(watermark.burn(labelled, config)))
    original_worker = output_module.run_pixel_worker

    def edit_before_work(parent, title, operation, **kwargs):
        record.watermark_config['text'] = 'Later edit'
        return original_worker(parent, title, operation, **kwargs)

    monkeypatch.setattr(output_module, 'run_pixel_worker', edit_before_work)
    window.output_manager.share_copy_preview(item)
    actual = cv2.imdecode(np.frombuffer(bytes(clipboard_mime(qapp).data('image/jpeg')), np.uint8), cv2.IMREAD_UNCHANGED)
    reference = cv2.imdecode(np.frombuffer(expected, np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(record.image, before)
    assert not record.exported


def test_transparent_jpeg_preview_burns_watermark_before_flattening(qapp):
    from utils import watermark
    from utils.image_utils import composite_transparent_for_display

    window = host(qapp)
    image = np.full((320, 480, 4), 12345, np.uint16)
    image[..., 3] = 65535
    image[:, :160, 3] = 0
    image[:, 160:320, 3] = 32767
    before = image.copy()
    config = watermark.WatermarkConfig(text='Creator', size_percent=6.0)
    item, record = add_record(window, image, 'transparent',
                              watermark_config=config.to_dict(), watermark_initialized=True)
    decorated = image.copy()
    decorated[..., :3] = window.label_manager.prepare_bgr_image(
        'output', image[..., :3], 0, output_id=record.output_id)
    expected = encode_jpeg_preview(composite_transparent_for_display(
        watermark.burn(decorated, config)))
    window.output_manager.share_copy_preview(item)
    actual = cv2.imdecode(np.frombuffer(bytes(clipboard_mime(qapp).data('image/jpeg')), np.uint8), cv2.IMREAD_UNCHANGED)
    reference = cv2.imdecode(np.frombuffer(expected, np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(record.image, before)
