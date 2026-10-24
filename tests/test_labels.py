"""Behavioral regressions for label drafts, identity and native rendering."""
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QDialog, QWidget

from controllers.label_manager import LabelManager
from utils.label_model import Label, LabelSet
from utils.validators import LabelAdder, LabelConfig, parse_label_range


@pytest.fixture
def manager(qapp, monkeypatch):
    from utils import recovery
    monkeypatch.setattr(recovery, "write_snapshot", lambda window: None)
    window = QWidget()
    window.raw_images = [np.zeros((180, 240, 3), np.uint16) for _ in range(3)]
    window.stack_images = []
    window.registration_results = []
    window.image_filenames = ["a.tif", "b.tif", "c.tif"]
    window.image_source_paths = ["/stack/" + name for name in window.image_filenames]
    window.current_display_index = -1
    window.fusion_result = np.ones((180, 240, 3), np.uint16)
    window.undo = []
    window.dirty = 0
    window.refreshes = 0
    window.push_label_undo = window.undo.append
    window._mark_project_dirty = lambda: setattr(window, "dirty", window.dirty + 1)
    window.refresh_annotation_views = lambda: setattr(window, "refreshes", window.refreshes + 1)
    window.statusBar = lambda: SimpleNamespace(showMessage=lambda *args: None)
    return LabelManager(window)


def caption(target="input", text="A"):
    config = LabelConfig()
    config.target = target
    config.format = text
    config.font_size = 24
    config.x_location = 8
    config.y_location = 40
    return Label(config)


def edit_dialog(monkeypatch, manager, callback):
    def execute(dialog):
        callback(dialog)
        return dialog.result()
    monkeypatch.setattr(QDialog, "exec", execute)
    manager.show_add_label_dialog()


def test_apply_commits_all_edits_once_before_closing(manager, monkeypatch):
    first = caption(text="first")
    manager.labels.add(first)

    def edit(dialog):
        manager.format_edit.setText("edited first")
        manager._dialog_add_label()
        manager.format_edit.setText("second")
        manager._dialog_list.setCurrentRow(0)
        assert manager.format_edit.text() == "edited first"
        assert manager.labels.get(first.label_id).config.format == "first"
        manager._ok_button.click()
        assert dialog.result() == QDialog.DialogCode.Accepted

    edit_dialog(monkeypatch, manager, edit)
    assert [label.config.format for label in manager.labels] == ["edited first", "second"]
    assert manager.labels.get(first.label_id) is not None
    assert len(manager.window.undo) == 1
    assert manager.window.undo[0]["labels_v2"][0]["format"] == "first"
    assert manager.window.dirty == 1
    assert manager.window.refreshes == 1


def test_cancel_does_not_change_model_undo_dirty_or_revision(manager, monkeypatch):
    manager.labels.add(caption())
    before = manager.export_state()
    revision = manager.revision

    def edit(dialog):
        manager.format_edit.setText("discard")
        manager._dialog_add_label()
        manager._dialog_delete_label()
        dialog.reject()

    edit_dialog(monkeypatch, manager, edit)
    assert manager.export_state() == before
    assert manager.revision == revision
    assert manager.window.undo == []
    assert manager.window.dirty == 0


def test_deleting_last_label_commits_empty_set(manager, monkeypatch):
    manager.labels.add(caption())

    def edit(dialog):
        manager._dialog_delete_label()
        assert len(manager._draft) == 0
        manager._ok_button.click()
        assert dialog.result() == QDialog.DialogCode.Accepted

    edit_dialog(monkeypatch, manager, edit)
    assert len(manager.labels) == 0
    assert len(manager.window.undo) == 1


@pytest.mark.parametrize("invalid", ["1-2,x", "0", "4", "2-8"])
def test_invalid_range_keeps_dialog_open(manager, monkeypatch, invalid):
    manager.labels.add(caption())

    def edit(dialog):
        manager._range_mode.setCurrentIndex(2)
        manager.range_edit.setText(invalid)
        assert not manager._ok_button.isEnabled()
        assert not manager._commit_dialog()
        assert dialog.result() != QDialog.DialogCode.Accepted
        manager.range_edit.setText("3-2")
        assert manager._ok_button.isEnabled()
        manager._ok_button.click()

    edit_dialog(monkeypatch, manager, edit)
    assert len(manager.window.undo) == 1
    assert manager.labels.all()[0].config.frame_ids == manager.frame_ids("input")[1:]


def test_unselected_invalid_draft_is_still_validated(manager, monkeypatch):
    manager.labels.add(caption())

    def edit(dialog):
        manager._range_mode.setCurrentIndex(2)
        manager.range_edit.setText("x")
        manager._dialog_add_label()
        assert not manager._commit_dialog()
        assert manager.range_edit.text() == "x"
        dialog.reject()

    edit_dialog(monkeypatch, manager, edit)
    assert not manager.window.undo


def test_output_identity_and_numbering_ignore_history_row(manager, monkeypatch):
    label = caption("output", "Fig {value}")
    label.config.output_id = "chosen"
    label.config.starting_value = 10
    label.config.interval = 2
    manager.labels.add(label)
    calls = []

    def draw(self, image, index):
        calls.append((self.config.output_id, index))
        return image
    monkeypatch.setattr(LabelAdder, "add_label_to_image", draw)
    image = manager.window.raw_images[0]
    manager.prepare_bgr_image("output", image, 7, output_id="other")
    assert calls == []
    manager.prepare_bgr_image("output", image, 7, output_id="chosen")
    manager.prepare_bgr_image("output", image, 0, output_id="chosen")
    assert calls == [("chosen", 0), ("chosen", 0)]


def test_bound_range_survives_reorder_and_snapshot_freezes_membership(manager, monkeypatch):
    label = caption(text="Frame {value}")
    label.config.range = "2"
    label.config.frame_ids = [manager.frame_ids("input")[1]]
    manager.labels.add(label)
    frozen = manager.snapshot()
    calls = []
    monkeypatch.setattr(LabelAdder, "add_label_to_image",
                        lambda self, image, index: calls.append(index) or image)
    manager.window.image_filenames.reverse()
    manager.window.image_source_paths.reverse()
    manager.window.image_filenames = manager.window.image_filenames[1:] + manager.window.image_filenames[:1]
    manager.window.image_source_paths = manager.window.image_source_paths[1:] + manager.window.image_source_paths[:1]
    image = manager.window.raw_images[0]
    manager.prepare_bgr_image("input", image, 0)
    manager.prepare_bgr_image("input", image, 1)
    frozen.prepare_bgr_image("input", image, 1)
    assert calls == [0, 1]


def test_failed_export_is_not_silently_successful_and_error_clears(manager, monkeypatch):
    manager.labels.add(caption())

    def fail(self, image, index):
        self.last_error = "font_unavailable"
        return image
    monkeypatch.setattr(LabelAdder, "add_label_to_image", fail)
    with pytest.raises(ValueError, match="font_unavailable"):
        manager.prepare_bgr_image("input", manager.window.raw_images[0], 0)
    assert manager.draw_error_for("input")
    monkeypatch.setattr(LabelAdder, "add_label_to_image", lambda self, image, index: image)
    manager.prepare_bgr_image("input", manager.window.raw_images[0], 0)
    assert not manager.draw_error_for("input")


@pytest.mark.parametrize("gray", [False, True])
def test_native_uint16_label_preserves_uncovered_pixels_and_color(gray):
    image = np.full((120, 180) if gray else (120, 180, 3), 12345, np.uint16)
    adder = LabelAdder()
    adder.config.update_config(dict(format="A", font_size=28, x_location=10,
                                   y_location=50, font_color=(0, 0, 255)))
    out = adder.add_label_to_image(image, 0)
    assert not adder.last_error
    assert out.dtype == np.uint16 and out.shape == image.shape
    assert np.array_equal(out[70:], image[70:])
    assert np.any(out != image)
    if not gray:
        changed = np.any(out != image, axis=2)
        assert np.all(out[changed, 2] > out[changed, 0])
    assert np.all(image == 12345)


def test_restoring_labels_is_not_a_new_edit(manager):
    legacy = {"registered": {"enabled": True, "config": {"format": "", "text": "legacy"}}}
    manager.import_state(legacy)
    assert [label.target for label in manager.labels] == ["registered", "output"]
    assert len({label.label_id for label in manager.labels}) == 2
    assert all(label.config.format == "legacy" for label in manager.labels)
    assert manager.window.dirty == 0 and not manager.window.undo


def test_enabling_target_restores_disabled_labels():
    label = caption()
    label.enabled = False
    labels = LabelSet([label])
    labels.set_enabled("input", True)
    assert label.enabled


def test_range_bounds_rejected_before_expansion():
    assert not parse_label_range("1-999999999999", 12).valid
    assert parse_label_range("5-3", 12).frames == (3, 4, 5)


def test_preview_uses_selected_output_and_never_newest_fallback(manager):
    older = SimpleNamespace(output_id="older", label="older", image=np.full((20, 30, 3), 7, np.uint8))
    newest = SimpleNamespace(output_id="newest", label="newest", image=np.full((20, 30, 3), 9, np.uint8))
    selected = object()
    manager.window.output_list = SimpleNamespace(currentItem=lambda: selected, count=lambda: 2)
    manager.window.output_manager = SimpleNamespace(
        record_for_item=lambda item: older if item is selected else None,
        record_for_row=lambda row: [newest, older][row])
    assert manager._collect_frames("output")["images"][0] is older.image
    assert manager._collect_frames("output", "missing")["images"] == []
    manager.window.output_list.currentItem = lambda: None
    assert manager._collect_frames("output")["images"] == []


def test_preview_resize_and_zoom_do_not_rerender_or_mutate_frame(qapp, monkeypatch):
    from widgets.label_preview import LabelPreview
    from PyQt6.QtCore import QSize
    from PyQt6.QtGui import QResizeEvent

    preview = LabelPreview()
    frame = np.full((180, 240, 3), 12345, np.uint16)
    preview.set_preview(frame, vars(caption().config), 0)
    monkeypatch.setattr(preview._adder, "add_label_to_image",
                        lambda *args: pytest.fail("view-only changes must reuse rendered pixels"))
    preview.set_zoom_mode("actual")
    preview.resizeEvent(QResizeEvent(QSize(400, 300), QSize(280, 190)))
    preview.set_zoom_mode("fit")
    assert np.all(frame == 12345)
    assert not preview.pixmap().isNull()
    preview.deleteLater()


def test_draw_error_resets_after_position_is_corrected():
    adder = LabelAdder()
    adder.config.update_config(dict(format="A", font_size=24, x_location=9999, y_location=40))
    frame = np.zeros((120, 160, 3), np.uint8)
    assert adder.add_label_to_image(frame, 0) is frame
    assert adder.last_error == "label_outside_image"
    adder.config.x_location = 10
    assert np.any(adder.add_label_to_image(frame, 0))
    assert not adder.last_error
    assert adder.last_bounds[0] == 10


def test_disabled_label_remains_disabled_after_edit(manager, monkeypatch):
    label = caption()
    label.enabled = False
    manager.labels.add(label)

    def edit(dialog):
        manager.format_edit.setText("disabled draft")
        manager._ok_button.click()

    edit_dialog(monkeypatch, manager, edit)
    assert not manager.labels.get(label.label_id).enabled
    assert manager.labels.get(label.label_id).config.format == "disabled draft"


def test_unbounded_range_matching_does_not_expand_all_frame_numbers():
    parsed = parse_label_range("1-999999999999")
    assert parsed.matches(999999999998)
    assert not parsed.matches(999999999999)
    assert not parsed.matches(-1)


def test_stack_resize_scales_labels_without_touching_historic_output(manager):
    source, registered, output = [caption(target) for target in ("input", "registered", "output")]
    manager._labels = LabelSet([source, registered, output])
    manager.scale_stack_labels(0.5, 0.5)
    assert (source.config.x_location, source.config.y_location, source.config.font_size) == (4, 20, 12)
    assert registered.config.font_size == 12
    assert (output.config.x_location, output.config.y_location, output.config.font_size) == (8, 40, 24)
    assert not manager.window.undo


@pytest.mark.parametrize("target, method, action_name", [
    ("input", "delete_input_labels", "del_input_label_action"),
    ("registered", "delete_registered_labels", "del_reg_label_action"),
    ("output", "delete_output_labels", "del_output_label_action"),
])
def test_quick_clear_removes_only_its_target_and_restores_with_undo(
        manager, target, method, action_name):
    from PyQt6.QtGui import QAction

    actions = {
        "input": "del_input_label_action",
        "registered": "del_reg_label_action",
        "output": "del_output_label_action",
    }
    for own_target, name in actions.items():
        label = caption(own_target)
        label.enabled = False
        manager.labels.add(label)
        setattr(manager.window, name, QAction(manager.window))
    manager.window.del_all_labels_action = QAction(manager.window)
    manager._sync_delete_actions()
    before = manager.export_state()
    action = getattr(manager.window, action_name)
    action.triggered.connect(getattr(manager, method))
    assert action.isEnabled()
    action.trigger()
    assert {label.target for label in manager.labels} == set(actions) - {target}
    assert not action.isEnabled()
    assert all(getattr(manager.window, name).isEnabled()
               for own_target, name in actions.items() if own_target != target)
    assert manager.window.del_all_labels_action.isEnabled()
    assert manager.window.undo == [before]
    manager.import_state(manager.window.undo[0])
    assert manager.export_state() == before
    assert all(getattr(manager.window, name).isEnabled() for name in actions.values())
