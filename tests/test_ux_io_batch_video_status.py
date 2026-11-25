"""Focused UX tests for batch thumbnails, video progress, status bar reveal, and load warnings."""
from __future__ import annotations

import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QEventLoop, QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QMainWindow, QProgressBar, QLabel, QPushButton, QWidget

from controllers.output_manager import OutputManager, OutputRecord
from controllers.source_manager import SourceManager
from locales import trans
from ui.status_bar import WorkspaceStatusBar
from utils.output_share import notify_saved_path
from widgets.output_list import OutputListWidget


def test_batch_worker_emits_folder_loaded_progressively_with_bounded_previews(qapp, tmp_path, monkeypatch):
    import core.image_loader as loader_mod
    from core.workers import BatchImportWorker

    folders = []
    for i in range(3):
        d = tmp_path / f"stack_{i}"
        d.mkdir()
        assert cv2.imwrite(str(d / "a.png"), np.full((120, 160, 3), 40 + i * 40, np.uint8))
        assert cv2.imwrite(str(d / "b.png"), np.full((120, 160, 3), 99, np.uint8))
        folders.append(str(d))

    full_decodes = []
    orig_decode = loader_mod.decode_image
    monkeypatch.setattr(loader_mod, "decode_image", lambda p: full_decodes.append(p) or orig_decode(p))

    worker = BatchImportWorker("thumbnails", folder_paths=folders)
    per_folder = []
    final_batch = []
    worker.folder_loaded.connect(lambda folder, img: per_folder.append((folder, img)))
    worker.folders_loaded.connect(final_batch.extend)
    worker.run()

    assert [f for f, _ in per_folder] == folders
    assert all(img is not None and max(img.shape[:2]) <= 60 for _, img in per_folder)
    assert len(final_batch) == 3
    assert full_decodes == []


def test_batch_dialog_keeps_start_and_add_enabled_for_thumbnails_and_shows_busy_label(qapp, monkeypatch):
    from core import workers
    from dialogs.batch import BatchProcessingDialog

    cancelled = []

    class FakeWorker(QObject):
        folder_loaded = pyqtSignal(str, object)
        folders_loaded = pyqtSignal(object)
        stack_loaded = pyqtSignal(str, object, str)
        finished = pyqtSignal()

        def __init__(self, mode, **kwargs):
            super().__init__()
            self.mode = mode

        def start(self):
            pass

        def cancel(self):
            cancelled.append(self.mode)

    monkeypatch.setattr(workers, "BatchImportWorker", FakeWorker)
    dialog = BatchProcessingDialog(None)
    try:
        dialog.add_folders_to_list(["f1", "f2"])
        worker = dialog._import_worker
        assert dialog._import_busy
        assert dialog.btn_start.isEnabled()
        assert dialog.btn_add_folder.isEnabled()
        assert not dialog._thumbnail_status_label.isHidden()
        assert "0/2" in dialog._thumbnail_status_label.text()

        # Progressive arrival updates icon & counter before worker finishes.
        worker.folder_loaded.emit("f1", np.full((40, 40, 3), 120, np.uint8))
        assert not dialog.folder_list.item(0).icon().isNull()
        assert dialog.folder_list.item(1).icon().isNull()
        assert "1/2" in dialog._thumbnail_status_label.text()

        # Accepting cancels thumbnail worker cleanly instead of blocking.
        dialog.accept()
        assert cancelled == ["thumbnails"]
        assert not dialog._import_busy
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_batch_dialog_disables_start_and_shows_status_for_single_folder_stack_scan(qapp, monkeypatch):
    from core import workers
    from dialogs.batch import BatchProcessingDialog

    class FakeWorker(QObject):
        folder_loaded = pyqtSignal(str, object)
        folders_loaded = pyqtSignal(object)
        stack_loaded = pyqtSignal(str, object, str)
        finished = pyqtSignal()

        def __init__(self, mode, **kwargs):
            super().__init__()
            self.mode = mode

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(workers, "BatchImportWorker", FakeWorker)
    dialog = BatchProcessingDialog(None)
    try:
        assert dialog._start_import("stack", folder_path="single")
        assert not dialog.btn_start.isEnabled()
        assert not dialog.btn_add_single_folder.isEnabled()
        assert not dialog._stack_import_status_label.isHidden()
        assert dialog._stack_import_status_label.text() == trans.t("msg_loading_stack")

        dialog._import_worker.finished.emit()
        assert dialog.btn_start.isEnabled()
        assert dialog._stack_import_status_label.isHidden()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_video_import_reports_determinate_progress_and_source_manager_displays_it(qapp, tmp_path):
    from core.video_import import load_video

    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    assert writer.isOpened()
    for i in range(6):
        writer.write(np.full((24, 32, 3), 20 + i * 30, np.uint8))
    writer.release()

    events = []
    ok, msg, frames = load_video(
        str(video_path), 1.0, lambda: False,
        progress_callback=lambda done, total: events.append((done, total)),
    )
    assert ok, msg
    assert len(frames) == 6
    assert events
    assert events[-1][0] == 6 and events[-1][1] == 6

    # SourceManager._on_load_progress must NOT suppress video workers anymore.
    class FakeWorker(QObject):
        progress = pyqtSignal(int, int)

        def __init__(self):
            super().__init__()
            self.is_cancelled = False
            self.video = str(video_path)

    window = QWidget()
    manager = SourceManager(window)
    manager._load_progress_bar = QProgressBar(window)
    manager._load_status_label = QLabel(window)
    worker = FakeWorker()
    manager._load_worker = worker
    worker.progress.connect(manager._on_load_progress)
    try:
        worker.progress.emit(3, 6)
        assert manager._load_progress_bar.maximum() == 6
        assert manager._load_progress_bar.value() == 3
        assert "(3/6)" in manager._load_status_label.text()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_workspace_status_bar_inline_show_in_folder_tracks_message_lifetime(qapp, tmp_path, monkeypatch):
    import utils.file_manager as file_manager

    saved = tmp_path / "result.png"
    saved.write_bytes(b"png")

    bar = WorkspaceStatusBar()
    revealed = []
    monkeypatch.setattr(
        file_manager, "select_in_file_manager",
        lambda path, *, parent, on_finished, **kw: revealed.append(path) or on_finished(True),
    )
    try:
        assert bar._reveal_button.isHidden()
        bar.show_path_message("Saved file", str(saved), 8000)
        assert not bar._reveal_button.isHidden()
        assert bar._reveal_button.text() == trans.t("share_reveal")

        bar._reveal_button.click()
        assert revealed == [str(saved)]

        # A plain message without a path hides the button.
        bar.showMessage("Other message", 4000)
        assert bar._reveal_button.isHidden()

        # Re-showing via notify_saved_path re-enables it; clearMessage hides it.
        host = SimpleNamespace(statusBar=lambda: bar)
        notify_saved_path(host, str(saved))
        assert not bar._reveal_button.isHidden()
        bar.clearMessage()
        assert bar._reveal_button.isHidden()
    finally:
        bar.close()
        bar.deleteLater()
        qapp.processEvents()


def test_share_reveal_enabled_for_exported_aligned_stack_folder(qapp, tmp_path, monkeypatch):
    import controllers.output_manager as output_mod

    aligned_dir = tmp_path / "my_stack_aligned"
    aligned_dir.mkdir()

    window = QMainWindow()
    window.output_list = OutputListWidget(window)
    window.output_label = SimpleNamespace(setText=lambda _t: None)
    window.fusion_results = []
    manager = OutputManager(window)
    window.output_manager = manager

    record = OutputRecord.create(None, "my_stack_aligned", kind="aligned", exported=True)
    record.export_path = str(aligned_dir)
    item = window.output_manager._insert_output(None, "my_stack_aligned", record)

    revealed = []
    monkeypatch.setattr(
        output_mod, "select_in_file_manager",
        lambda path, *, parent, on_finished, **kw: revealed.append(path) or on_finished(True),
    )
    try:
        menu = manager._build_output_context_menu(item)
        share_menu = next(a.menu() for a in menu.actions() if a.text() == trans.t("menu_share"))
        actions = {a.text(): a for a in share_menu.actions()}
        assert not actions[trans.t("share_copy_preview")].isEnabled()
        assert not actions[trans.t("share_copy_paths")].isEnabled()
        assert actions[trans.t("share_reveal")].isEnabled()

        manager.share_reveal(item)
        assert revealed == [str(aligned_dir)]
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_source_manager_surfaces_warning_toast_for_failed_and_skipped_frames(qapp):
    bar = WorkspaceStatusBar()
    loader = SimpleNamespace(failed_count=2, skipped_derived_count=0)
    window = SimpleNamespace(image_loader=loader, statusBar=lambda: bar)
    manager = SourceManager(window)
    worker = SimpleNamespace(video=None)
    try:
        manager._show_load_issues_notice(worker, loaded_count=10)
        assert bar.currentMessage() == trans.t("msg_load_frames_failed").format(loaded=10, failed=2)

        loader.failed_count = 0
        loader.skipped_derived_count = 3
        manager._show_load_issues_notice(worker, loaded_count=10)
        assert bar.currentMessage() == trans.t("msg_load_frames_skipped_derived").format(loaded=10, skipped=3)

        loader.failed_count = 1
        loader.skipped_derived_count = 4
        manager._show_load_issues_notice(worker, loaded_count=10)
        assert bar.currentMessage() == trans.t("msg_load_frames_partial").format(loaded=10, failed=1, skipped=4)
    finally:
        bar.close()
        bar.deleteLater()
        qapp.processEvents()
