"""Focused UX tests for pixel progress modal, output drag-out, and list batching."""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QWidget,
)

from controllers.output_manager import ALIGNED_DIR_ROLE, OUTPUT_RECORD_ROLE, OutputManager, OutputRecord
from locales import trans
from ui import styles as ui_styles
from utils.frame_pixels import PixelProgress, _PixelProgress, run_pixel_worker
from widgets.output_list import OutputListWidget


def test_pixel_progress_is_styled_indeterminate_and_delayed(qapp, monkeypatch):
    parent = QWidget()
    shown_events = []
    orig_show = _PixelProgress.show

    def track_show(self):
        shown_events.append((self.styleSheet(), self.maximum(), self.findChild(QProgressBar)))
        orig_show(self)

    monkeypatch.setattr(_PixelProgress, "show", track_show)
    try:
        # Fast operation finishes well under 250ms: modal must never flash.
        res = run_pixel_worker(parent, "Quick Save", lambda: 42, delay_ms=250)
        assert res == 42
        assert shown_events == []

        # Slow operation past delay_ms shows styled dialog with indeterminate QProgressBar.
        progress = PixelProgress()
        def slow_op():
            time.sleep(0.08)
            progress.update(1, 2)
            time.sleep(0.08)
            progress.update(2, 2)
            return "done"

        res2 = run_pixel_worker(parent, "Slow Save", slow_op, progress=progress, delay_ms=20)
        assert res2 == "done"
        assert shown_events
        stylesheet, _, bar = shown_events[0]
        assert stylesheet == ui_styles.PROGRESS_DIALOG_STYLE
        assert bar is not None
    finally:
        parent.close()
        parent.deleteLater()
        qapp.processEvents()


def test_pixel_progress_cancel_sets_flag(qapp):
    parent = QWidget()
    progress = PixelProgress()
    dialog = _PixelProgress(parent, "Export", cancellable=True)
    try:
        dialog.canceled.connect(progress.cancel)
        assert dialog.minimum() == 0 and dialog.maximum() == 0
        dialog.reject()
        assert progress.cancelled
    finally:
        dialog.finish()
        dialog.deleteLater()
        parent.close()
        parent.deleteLater()
        qapp.processEvents()


def test_add_batch_outputs_and_export_records_collapse_into_single_worker_run(qapp, tmp_path, monkeypatch):
    import controllers.output_manager as output_mod
    from utils.file_dialogs import FileDialog
    from utils import scalebar

    window = QMainWindow()
    window.output_list = OutputListWidget(window)
    window.output_list.set_window(window)
    window.fusion_results = []
    window.fusion_result = None
    window.result_control_bar = QWidget(window)
    window.lbl_result_img = SimpleNamespace(clear=lambda: None, setText=lambda _t: None,
                                            set_display_pixmap=lambda _p: None)
    window.lbl_result_info = SimpleNamespace(setText=lambda _t: None)
    window.output_label = SimpleNamespace(setText=lambda _t: None)
    window.label_manager = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            labels=(),
            prepare_bgr_image=lambda _target, img, _idx, **kw: img,
        ),
        display_bgr_image=lambda _target, img, _idx, **kw: img,
    )
    window.export_manager = SimpleNamespace(
        filename_from_label=lambda label: label,
        _safe_export_path=lambda path, extra=(): path,
    )
    manager = OutputManager(window)
    window.output_manager = manager
    monkeypatch.setattr(scalebar, "export_cfg", lambda *a, **kw: None)
    monkeypatch.setattr(scalebar, "calibration_px_um", lambda *a, **kw: None)
    monkeypatch.setattr(scalebar, "burn", lambda img, cfg: img)

    paths = []
    for i in range(4):
        p = tmp_path / f"batch_{i}.png"
        assert cv2.imwrite(str(p), np.full((16, 16, 3), 30 + i * 20, np.uint8))
        paths.append(str(p))

    worker_calls = []
    orig_worker = output_mod.run_pixel_worker

    def spy_worker(parent, title, op, **kwargs):
        worker_calls.append((title, kwargs.get("progress") is not None))
        return orig_worker(parent, title, op, **kwargs)

    monkeypatch.setattr(output_mod, "run_pixel_worker", spy_worker)
    try:
        added = manager.add_batch_outputs(paths)
        assert added == 4
        # Collapsed into ONE worker call with a loading title (not 'Save').
        assert len(worker_calls) == 1
        assert worker_calls[0][0] == trans.t("msg_loading_stack")

        worker_calls.clear()
        out_dir = tmp_path / "exported"
        out_dir.mkdir()
        monkeypatch.setattr(FileDialog, "getExistingDirectory", lambda *a, **kw: str(out_dir))
        records = [manager.record_for_row(r) for r in range(window.output_list.count())]
        assert manager.export_records(records)
        assert len(worker_calls) == 1
        assert worker_calls[0][0] == trans.t("action_save")
        assert len(list(out_dir.glob("*.png"))) == 4
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_aligned_stack_click_uses_loading_title_not_retouch(qapp, tmp_path, monkeypatch):
    import controllers.output_manager as output_mod

    aligned_dir = tmp_path / "stack_aligned"
    aligned_dir.mkdir()
    assert cv2.imwrite(str(aligned_dir / "f0.png"), np.full((16, 16, 3), 80, np.uint8))

    window = QMainWindow()
    window.output_list = OutputListWidget(window)
    window.fusion_results = []
    window.fusion_result = None
    window.registration_results = []
    from PyQt6.QtWidgets import QSlider
    window.result_slider = QSlider(window)
    window.result_control_bar = QWidget(window)
    manager = OutputManager(window)
    window.output_manager = manager
    monkeypatch.setattr(manager, "show_registration_result", lambda _idx: None)

    record = OutputRecord.create(None, "stack_aligned", kind="aligned", exported=True)
    record.export_path = str(aligned_dir)
    record.aligned_names = ("f0.png",)
    item = QListWidgetItem("stack_aligned")
    item.setData(OUTPUT_RECORD_ROLE, record)
    window.output_list.addItem(item)
    window.output_list.setCurrentItem(item)

    titles = []
    monkeypatch.setattr(
        output_mod, "run_pixel_worker",
        lambda parent, title, op, **kw: titles.append(title) or op(),
    )
    try:
        manager.display_output_image_in_result_view(item)
        assert titles == [trans.t("msg_loading_stack")]
        assert titles[0] != trans.t("retouch_title")
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_drag_reuses_persistent_export_and_resolves_source_frames_off_ui_thread(qapp, tmp_path, monkeypatch):
    import utils.frame_pixels as frame_pixels
    import widgets.output_list as output_list_mod
    from core.source_frames import PREVIEW_CACHE, ensure_preview, scan_source
    from utils import scalebar

    src_png = tmp_path / "batch_out.png"
    assert cv2.imwrite(str(src_png), np.full((800, 1000, 3), 140, np.uint8))
    frame = scan_source(str(src_png))[0]
    preview = ensure_preview(frame)
    assert preview.shape[:2] != frame.shape[:2]

    # Case 1: up-to-date PNG matches drag_export_format='.png' -> reused directly without temp file.
    record_reuse = OutputRecord.create(frame, "batch_out.png", kind="batch", exported=True)
    record_reuse.annotations_baked = True
    record_reuse.watermark_baked = True
    record_reuse.watermark_initialized = True
    record_reuse.export_path = str(src_png)
    record_reuse.exported_image = frame

    # Case 2: SourceFrame batch output dragged as '.jpg' -> resolved via resolve_frame off UI thread.
    record_encode = OutputRecord.create(frame, "batch_encode", kind="batch", exported=True)
    record_encode.annotations_baked = True
    record_encode.watermark_baked = True
    record_encode.watermark_initialized = True
    record_encode.export_path = str(src_png)
    record_encode.exported_image = frame

    labels = SimpleNamespace(prepare_bgr_image=lambda _t, img, _i, **kw: img)
    monkeypatch.setattr(scalebar, "export_cfg", lambda *a, **kw: None)
    monkeypatch.setattr(scalebar, "burn", lambda img, cfg: img)

    dragged = []

    class FakeDrag:
        def __init__(self, _w):
            pass

        def setMimeData(self, mime):
            dragged.extend(url.toLocalFile() for url in mime.urls())

        def exec(self, actions):
            assert actions == Qt.DropAction.CopyAction

    monkeypatch.setattr(output_list_mod, "QDrag", FakeDrag)

    # First verify reuse when format matches '.png'
    widget = OutputListWidget()
    widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    widget.addItem("batch_out.png")
    window = SimpleNamespace(
        fusion_results=[frame], fusion_result=frame, drag_export_format=".png",
        label_manager=SimpleNamespace(snapshot=lambda: labels),
        output_manager=SimpleNamespace(
            record_for_item=lambda _it: record_reuse,
            record_for_row=lambda _r: record_reuse,
        ),
    )
    manager = OutputManager(window)
    window.output_manager._share_persistent_path = manager._share_persistent_path
    widget.set_window(window)
    widget.item(0).setSelected(True)
    widget.startDrag(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction)
    assert [os.path.normpath(p) for p in dragged] == [os.path.normpath(str(src_png))]
    assert widget._temp_files == []

    # Second verify SourceFrame encodes off the GUI thread when format differs ('.jpg')
    dragged.clear()
    threads = []
    orig_worker = frame_pixels.run_pixel_worker

    def spy_worker(parent, title, op, **kwargs):
        def wrapped():
            threads.append(QThread.currentThread() is not qapp.thread())
            return op()
        return orig_worker(parent, title, wrapped, **kwargs)

    monkeypatch.setattr(frame_pixels, "run_pixel_worker", spy_worker)
    window.drag_export_format = ".jpg"
    window.output_manager = SimpleNamespace(
        record_for_item=lambda _it: record_encode,
        record_for_row=lambda _r: record_encode,
        _share_persistent_path=manager._share_persistent_path,
    )
    window.fusion_results = [preview]
    PREVIEW_CACHE.clear()
    try:
        widget.startDrag(Qt.DropAction.CopyAction)
        assert len(dragged) == 1
        assert dragged[0].lower().endswith(".jpg")
        assert os.path.isfile(dragged[0])
        saved = cv2.imdecode(np.fromfile(dragged[0], dtype=np.uint8), cv2.IMREAD_COLOR)
        assert saved.shape == frame.shape
        np.testing.assert_array_equal(saved, np.full(frame.shape, 140, np.uint8))
        assert threads == [True]
        assert [os.path.normpath(p) for p in widget._temp_files] == [os.path.normpath(dragged[0])]
    finally:
        widget.cleanup_temp_files()
        assert not os.path.isfile(dragged[0])
        assert os.path.isfile(src_png)
    src_png.write_bytes(b'externally replaced')
    assert manager._share_persistent_path(record_reuse) == ''


@pytest.mark.parametrize('button_released', [False, True])
def test_drag_cleanup_waits_for_native_handoff(qapp, tmp_path, monkeypatch, button_released):
    from PyQt6.QtCore import QUrl
    import widgets.output_list as output_list_mod

    path = tmp_path / 'drag.png'
    path.touch()
    widget = OutputListWidget()
    widget.set_window(SimpleNamespace())
    widget.addItem('drag')
    widget.item(0).setSelected(True)
    prepared = []

    def prepare(_items):
        prepared.append(True)
        assert len(prepared) == 1
        widget._temp_files.append(str(path))
        return [QUrl.fromLocalFile(str(path))]

    class Drag:
        def __init__(self, _widget):
            assert not button_released

        def setMimeData(self, mime):
            assert os.path.samefile(mime.urls()[0].toLocalFile(), path)

        def exec(self, actions):
            widget.startDrag(actions)
            widget.cleanup_temp_files()
            assert path.is_file()

    monkeypatch.setattr(widget, '_prepare_drag_urls', prepare)
    monkeypatch.setattr(output_list_mod, 'QDrag', Drag)
    if button_released:
        buttons = iter((Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton))
        monkeypatch.setattr(output_list_mod, 'QApplication', SimpleNamespace(mouseButtons=lambda: next(buttons)))
    widget.startDrag(Qt.DropAction.CopyAction)
    assert not path.exists()


def test_drag_unexported_aligned_stack_stages_to_temp_without_modal_picker(qapp, tmp_path):
    calls = []

    class FakeExporter:
        def save_aligned_frames_as(self, frames, names, label, *, destination=None, extension=None,
                                   record=None, progress=None, show_status=True):
            assert not show_status
            calls.append((destination, extension))
            assert destination is not None, "unexported aligned drag must stage to temp dir, not prompt"
            out = os.path.join(destination, f"{label}_aligned")
            os.makedirs(out, exist_ok=True)
            assert cv2.imwrite(os.path.join(out, "f0.png"), frames[0])
            return out

    widget = OutputListWidget()
    widget.set_window(SimpleNamespace(export_manager=FakeExporter()))
    record = OutputRecord.create(None, "aligned_mem", kind="aligned", exported=False)
    record.aligned_frames = (np.full((12, 12, 3), 90, np.uint8),)
    record.aligned_names = ("f0.png",)

    folder = widget._aligned_drag_folder(None, record)
    try:
        assert folder and os.path.isdir(folder)
        assert len(widget._temp_dirs) == 1
        assert calls == [(widget._temp_dirs[0], "png")]
    finally:
        stage = widget._temp_dirs[0]
        widget.cleanup_temp_files()
        assert not os.path.exists(stage)

    stale_export = tmp_path / 'old_export'
    current_export = tmp_path / 'current_export'
    stale_export.mkdir()
    current_export.mkdir()
    record.exported = True
    record.export_path = str(current_export)
    assert widget._aligned_drag_folder(str(stale_export), record) == str(current_export)
    record.annotations_changed = True
    try:
        assert widget._aligned_drag_folder(str(stale_export), record) != str(current_export)
        assert len(widget._temp_dirs) == 1
    finally:
        widget.cleanup_temp_files()


def test_file_list_uniform_item_sizes_and_updates_batching(qapp):
    from controllers.source_manager import SourceManager
    from PyQt6.QtGui import QPixmap

    class TrackingList(QListWidget):
        def __init__(self):
            super().__init__()
            self.update_states = []

        def setUpdatesEnabled(self, enabled):
            self.update_states.append(bool(enabled))
            super().setUpdatesEnabled(enabled)

    file_list = TrackingList()
    window = SimpleNamespace(
        file_list=file_list,
        raw_images=[],
        excluded_frames=set(),
        source_images_label=SimpleNamespace(setText=lambda _t: None),
    )
    manager = SourceManager(window)
    thumb = QPixmap(40, 40)
    manager.update_file_list(["a.png", "b.png"], [thumb, thumb])
    assert file_list.count() == 2
    assert file_list.update_states == [False, True]


def test_pixel_progress_cancel_disables_button_and_blocks_repeated_reject(qapp):
    from PyQt6 import sip
    from locales import trans
    from utils.frame_pixels import _PixelProgress

    dialog = _PixelProgress(None, "Exporting", cancellable=True)
    canceled_count = []

    def _on_canceled():
        canceled_count.append(True)
        dialog.mark_cancelling()

    dialog.canceled.connect(_on_canceled)
    try:
        dialog.show()
        btn = dialog._cancel_btn
        assert btn is not None and btn.isEnabled()
        btn.click()
        qapp.processEvents()
        # Cancel button stays alive (no mid-signal deletion) and is disabled.
        assert dialog.isVisible()
        assert not sip.isdeleted(btn)
        assert not btn.isEnabled()
        assert dialog.labelText() == trans.t("depth_cancelling")
        assert len(canceled_count) == 1

        # Repeated reject() calls do not re-emit canceled, and progress updates preserve the label.
        dialog.reject()
        dialog.set_progress(2, 5)
        assert len(canceled_count) == 1
        assert dialog.labelText() == trans.t("depth_cancelling")
        dialog.set_progress(6, 5)
        assert dialog.maximum() == dialog.value() == 6
    finally:
        dialog.finish()
        dialog.deleteLater()
        qapp.processEvents()
