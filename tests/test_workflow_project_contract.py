"""Behavioral regressions for source identity, ROI and project restoration."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
cv2 = pytest.importorskip("cv2")

from controllers.source_manager import LoadOptions, SourceManager
from controllers.render_manager import RenderManager
from utils import project_file, recovery
from tests.test_project_file import _window, _make_stack


def test_excluding_last_frame_is_allowed_and_invalidates_alignment(qapp):
    from PyQt6.QtCore import Qt
    from tests.test_gui_units import _source_window
    window = _source_window(["a.png", "b.png"])
    window.raw_images = [np.zeros((4, 4, 3), np.uint8) for _ in range(2)]
    window.excluded_frames = set()
    window.aligned_images = list(window.raw_images)
    window.is_images_aligned = True
    manager = SourceManager(window)
    manager.decorate_source_items()
    for index in range(2):
        item = window.file_list.item(index)
        item.setCheckState(Qt.CheckState.Unchecked)
        manager.on_item_changed(item)
    assert window.excluded_frames == {0, 1}
    assert window.aligned_images == []
    assert window.is_images_aligned is False
    assert window.source_revision == 2


def test_delete_remaps_excluded_frames_without_reincluding_them():
    window = SimpleNamespace(excluded_frames={0, 2, 4})
    SourceManager(window)._remap_exclusions_after_delete([0, 3])
    assert window.excluded_frames == {1, 2}


def test_append_preserves_labels_exclusions_and_base_resolution(qapp):
    from tests.test_gui_units import _source_window
    window = _source_window(["a.png", "b.png"])
    window.current_scale_factor = 0.5
    window.base_scale_factor = 1.0
    window.raw_images = [np.zeros((4, 4, 3), np.uint8) for _ in range(2)]
    window.base_images = [np.zeros((8, 8, 3), np.uint8) for _ in range(2)]
    window.excluded_frames = {1}
    new = np.ones((4, 4, 3), np.uint8)
    SourceManager(window)._apply_load_options(LoadOptions(0.5, ["c.png"], [new], [new]), append=True)
    assert window.excluded_frames == {1}
    assert window.label_manager.resets == 0
    assert window.base_scale_factor == 0.5
    assert [frame.shape[:2] for frame in window.base_images] == [(4, 4)] * 3


def test_alignment_identity_changes_with_equal_count_selection_and_parameters():
    images = [np.full((4, 4, 3), i, np.uint8) for i in range(3)]
    window = SimpleNamespace(raw_images=images, source_revision=4,
                             current_fusion_method=lambda: "guided_filter",
                             current_alignment_method=lambda: "ecc",
                             reg_downscale_width=1024, exposure_match=True)
    manager = RenderManager(window)
    initial = manager._alignment_key([0, 1])
    assert manager._alignment_key([0, 2]) != initial
    window.reg_downscale_width = 512
    assert manager._alignment_key([0, 1]) != initial
    window.reg_downscale_width = 1024
    window.raw_images[0] = images[0].copy()
    assert manager._alignment_key([0, 1]) != initial


def test_roi_base_preview_uses_included_snapshot_and_actual_dimensions(qapp):
    from dialogs.roi import ROIRenderOptionsDialog
    frames = [np.zeros((12, 18, 3), np.uint8), np.full((24, 32, 3), 255, np.uint8)]
    dialog = ROIRenderOptionsDialog(2, frame_names=["first.png", "fourth.png"],
                                    frame_indices=[0, 3], frames=frames, roi_size=(5, 6))
    dialog.rb_paste.setChecked(True)
    dialog.combo_frames.setCurrentIndex(1)
    assert "fourth.png" in dialog.combo_frames.currentText()
    assert "4" in dialog.combo_frames.currentText()
    assert dialog.lbl_preview.pixmap().toImage().pixelColor(0, 0).red() == 255
    assert "32" in dialog.lbl_output.text() and "24" in dialog.lbl_output.text()
    dialog.accept()
    assert dialog.base_frame_index == 1
    assert dialog.mode == "paste"


def test_project_preserves_reordered_and_removed_tiff_pages(tmp_path):
    container = tmp_path / "stack.tif"
    container.write_bytes(b"reference")
    window = _window([str(container)] * 2, str(tmp_path))
    window.image_filenames = ["stack_page_003.tif", "stack_page_001.tif"]
    window.excluded_frames = {1}
    state = project_file._collect_state(window)
    assert state["sources"] == [str(container)]
    pages = [np.full((4, 6, 3), i, np.uint16) for i in range(3)]
    names = [f"stack_page_{i + 1:03d}.tif" for i in range(3)]
    restored = _window([str(container)] * 3, str(tmp_path))
    restored.image_loader = SimpleNamespace(image_paths=[str(container)] * 3)
    options = project_file._prepare_project_load(
        restored, state, LoadOptions(1.0, names, list(pages), list(pages)))
    assert options.filenames == window.image_filenames
    assert [int(frame[0, 0, 0]) for frame in options.working_images] == [2, 0]
    assert len(restored.image_loader.image_paths) == 2


def test_project_missing_decoded_page_keeps_current_stack(tmp_path):
    source = tmp_path / "stack.tif"
    source.write_bytes(b"reference")
    window = _window([str(source)], str(tmp_path))
    current = window.raw_images
    window.image_loader = SimpleNamespace(image_paths=[str(source)])
    state = {"sources": [str(source)], "frame_order": [{"source": 0, "name": "stack_page_002.tif"}]}
    options = LoadOptions(1.0, ["stack_page_001.tif"], current, current)
    with pytest.raises(ValueError, match="could not be decoded"):
        project_file._prepare_project_load(window, state, options)
    assert window.raw_images is current


def test_project_replays_rotate_and_flip_and_preserves_zero_coordinates(tmp_path):
    paths = _make_stack(str(tmp_path / "source"), 1)
    window = _window(paths, str(tmp_path))
    window.source_transforms = [{"operation": "rotate", "code": cv2.ROTATE_90_CLOCKWISE},
                                {"operation": "flip", "code": 1}]
    window.scale_bar_custom_x = window.scale_bar_custom_y = 0.0
    state = project_file._collect_state(window)
    source = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    window.image_loader = SimpleNamespace(image_paths=paths)
    options = project_file._prepare_project_load(window, state,
        LoadOptions(1.0, window.image_filenames, [source], [source]))
    np.testing.assert_array_equal(options.working_images[0], cv2.flip(cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE), 1))
    seen = []
    window.update_source_view = seen.append
    project_file._apply_after_load(window, state)
    assert window.scale_bar_custom_x == window.scale_bar_custom_y == 0.0
    assert seen == [0]


def test_project_roundtrips_per_method_parameters(tmp_path):
    paths = _make_stack(str(tmp_path / "source"), 1)
    window = _window(paths, str(tmp_path))
    window.brightness_option_for = lambda option, method: option == "base_correct" and method == "openfocus_b"
    window.smoothing_for_method = lambda method: 3.5 if method == "openfocus_c" else 1.25
    window.kernel_for_method = lambda method: 11 if method == "dct" else 31
    state = project_file._collect_state(window)
    restored = _window(paths, str(tmp_path))
    restored._KERNEL_MODES = {"dct": ("dct", 99, 7), "guided_filter": ("guided", 99, 31)}
    project_file._apply_after_load(restored, state)
    assert restored._brightness_by_method[("base_correct", "openfocus_b")] is True
    assert restored._brightness_by_method[("exposure_match", "openfocus_b")] is False
    assert restored._smoothing_by_method["openfocus_c"] == 3.5
    assert restored._kernel_by_mode == {"dct": 11, "guided": 31}


def test_project_does_not_rebind_missing_absolute_source_to_same_basename(tmp_path):
    paths = _make_stack(str(tmp_path), 1)
    state = {"sources": [str(tmp_path / "missing" / "f0.png")]}
    assert project_file._resolve_sources(state, str(tmp_path)) is None
    assert project_file._resolve_sources({"sources": ["f0.png"]}, str(tmp_path)) == paths


def test_unreconstructable_source_changes_do_not_claim_project_saved(tmp_path):
    paths = _make_stack(str(tmp_path / "source"), 1)
    window = _window(paths, str(tmp_path))
    window.unrecoverable_source_changes = ["White-field corrected pixels"]
    target = tmp_path / "session.ofproj"
    ok, error = project_file.save_project(window, str(target))
    assert not ok and "Export" in error
    assert not target.exists()


def test_update_restore_targets_current_snapshot_not_saved_project(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "recovery_dir", lambda: str(tmp_path))
    monkeypatch.setattr(recovery, "write_snapshot", lambda window: True)
    assert recovery.mark_pending_restore(SimpleNamespace(project_path="older.ofproj"))
    assert recovery.pending_restore() == ""


def test_update_restore_snapshot_failure_does_not_publish_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "recovery_dir", lambda: str(tmp_path))
    monkeypatch.setattr(recovery, "write_snapshot", lambda window: False)
    assert recovery.mark_pending_restore(SimpleNamespace()) is False
    assert recovery.pending_restore() is None


def test_render_with_exclusions_publishes_matching_names(qapp, monkeypatch):
    from tests.test_gui_units import _render_window, _patched_render_manager, _finish_alignment
    window = _render_window([np.full((4, 4, 3), i, np.uint8) for i in range(3)])
    window.excluded_frames = {1}
    window.effective_stack = lambda: ([window.raw_images[i] for i in (0, 2)],
                                      [window.image_filenames[i] for i in (0, 2)], [0, 2])
    manager, saved = _patched_render_manager(monkeypatch, window)
    published = []
    manager._publish_aligned_stack = lambda _window, frames: published.append(frames)
    manager.start_render()
    processed = _finish_alignment(manager, window, manager.worker.args[0])
    assert published == [processed]
    assert window.registration_filenames == ["f0.png", "f2.png"]
    assert window.registered_source_indices == [0, 2]
    assert saved == []


def test_preview_reuses_only_matching_alignment_pixels(qapp, monkeypatch):
    from tests.test_gui_units import _render_window, _patched_render_manager
    window = _render_window([np.zeros((20, 30, 3), np.uint8) for _ in range(2)])
    window.set_alignment_method("ecc")
    window.chk_quick_preview.setChecked(True)
    manager, _ = _patched_render_manager(monkeypatch, window)
    window.is_images_aligned = True
    window.aligned_images = [np.full((20, 30, 3), 77, np.uint8) for _ in range(2)]
    window.alignment_request_key = manager._alignment_key([0, 1])
    manager.start_render()
    assert int(manager.worker.args[0][0][0, 0, 0]) == 77
    assert manager.worker.args[4:6] == (False, False)


def test_preview_completion_is_discarded_after_source_revision_change(qapp, monkeypatch):
    from tests.test_gui_units import _render_window, _patched_render_manager
    window = _render_window([np.zeros((20, 30, 3), np.uint8) for _ in range(2)])
    window.chk_quick_preview.setChecked(True)
    manager, _ = _patched_render_manager(monkeypatch, window)
    manager.start_render()
    handed = manager.worker.args[0]
    window.source_revision = 1
    manager.on_render_finished(handed, np.ones((20, 30, 3), np.uint8), False, 0, 1, "CPU")
    assert window.fusion_result is None
    assert window.output_list.count() == 0


def test_cancelled_render_does_not_publish_late_success(qapp, monkeypatch):
    from tests.test_gui_units import _render_window, _patched_render_manager
    window = _render_window([np.zeros((20, 30, 3), np.uint8) for _ in range(2)])
    manager, _ = _patched_render_manager(monkeypatch, window)
    manager.start_render()
    manager.worker.is_cancelled = True
    manager.on_render_finished(window.raw_images, np.ones((20, 30, 3), np.uint8), False, 0, 1, "CPU")
    assert window.fusion_result is None
    assert window.output_list.count() == 0


def test_recovery_marker_retains_only_nonempty_document_association(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "recovery_dir", lambda: str(tmp_path))
    monkeypatch.setattr(recovery, "write_snapshot", lambda window: True)
    assert recovery.mark_pending_restore(SimpleNamespace(project_path="saved.ofproj", raw_images=[object()]))
    assert recovery.pending_restore() == "saved.ofproj"
    assert recovery.mark_pending_restore(SimpleNamespace(project_path="saved.ofproj", raw_images=[]))
    assert recovery.pending_restore() == ""


def test_relocation_preserves_relative_folders_and_requires_confirmation(qapp, tmp_path, monkeypatch):
    import json
    import os
    from PyQt6.QtWidgets import QFileDialog, QMessageBox
    old = tmp_path / "old"
    new = tmp_path / "new"
    paths = [str(old / "a" / "same.png"), str(old / "b" / "same.png")]
    for folder in (new / "a", new / "b"):
        folder.mkdir(parents=True)
        (folder / "same.png").write_bytes(b"source")
    identity = os.path.normcase(os.path.abspath(paths[1])) + "::same.png"
    data = {"openfocus_project": 2, "sources": paths, "settings": {},
            "labels": {"version": 2, "labels_v2": [{"target": "input", "frame_ids": [identity]}]}}
    project = tmp_path / "moved.ofproj"
    project.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(new))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: next(
        button for button in self.buttons() if self.buttonRole(button) == QMessageBox.ButtonRole.RejectRole))
    assert project_file.validate_project_for_open(None, str(project)) == (False, "", {})
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: next(
        button for button in self.buttons() if self.buttonRole(button) == QMessageBox.ButtonRole.AcceptRole))
    ok, error, state = project_file.validate_project_for_open(None, str(project))
    assert ok, error
    assert state["resolved_sources"] == [str(new / "a" / "same.png"), str(new / "b" / "same.png")]
    assert state["labels"]["labels_v2"][0]["frame_ids"] == [
        os.path.normcase(os.path.abspath(new / "b" / "same.png")) + "::same.png"]
    assert state["sources_relocated"] is True
    assert json.loads(project.read_text(encoding="utf-8"))["sources"] == paths


def test_failed_load_preserves_stack_and_reports_async_error(qapp, monkeypatch):
    from controllers import source_manager as module
    from tests.test_gui_units import _StubSignal, _StubStatusBar
    class Worker:
        def __init__(self, *args, **kwargs):
            self.finished_load = _StubSignal()
            self.finished = _StubSignal()
            self.disposed = False
        def start(self):
            pass
        def deleteLater(self):
            self.disposed = True
    monkeypatch.setattr(module, "StackLoadWorker", Worker)
    status = _StubStatusBar()
    original = [np.zeros((4, 4, 3), np.uint8)]
    window = SimpleNamespace(raw_images=original, current_folder_path="old",
                             image_loader=object(), statusBar=lambda: status)
    manager = SourceManager(window)
    manager._show_load_status = lambda: None
    errors = []
    manager._start_load_worker(folder="new", on_error=errors.append)
    worker = manager._load_worker
    worker.finished_load.slot(False, "decode failed", [], [])
    assert window.raw_images is original
    assert window.current_folder_path == "old"
    assert errors == ["decode failed"]
    assert manager._load_worker is worker and not worker.disposed
    worker.finished.slot()
    assert manager._load_worker is None and worker.disposed


def test_resize_scales_metadata_and_labels_using_actual_pixels(qapp, monkeypatch):
    from controllers import transform_manager as module
    from tests.test_gui_units import _resize_window, _StubResizeDialog
    window = _resize_window((10, 14, 3), 1.0, 0.5)
    window.source_px_um = 2.0
    ratios = []
    window.label_manager = SimpleNamespace(scale_stack_labels=lambda x, y: ratios.append((x, y)))
    monkeypatch.setattr(module, "DownsampleDialog", _StubResizeDialog(1.0))
    monkeypatch.setattr(module, "show_success_box", lambda *args, **kwargs: None)
    window.transform_manager.resize_all_images()
    assert ratios == [(2.0, 2.0)]
    assert window.source_px_um == 1.0


def test_worker_reference_survives_result_until_thread_finishes(qapp):
    from tests.test_gui_units import _StubSignal
    class Worker:
        def __init__(self):
            self.running = True
            self.finished = _StubSignal()
            self.disposals = 0
        def isRunning(self):
            return self.running
        def deleteLater(self):
            self.disposals += 1
    manager = RenderManager(SimpleNamespace())
    worker = manager.worker = Worker()
    manager._dispose_worker()
    assert worker in manager._retiring_workers
    assert worker.disposals == 0
    worker.running = False
    worker.finished.slot()
    assert worker.disposals == 1
    assert not manager._retiring_workers
