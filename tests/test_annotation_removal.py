"""Label and scale-bar removal must reach the menu state and both canvases."""
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFontComboBox, QLineEdit, QSpinBox, QWidget,
)


class _Action:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, enabled):  # noqa: N802
        self.enabled = bool(enabled)


def _annotated_manager(qapp, target_index):
    """A LabelManager whose add-label form widgets hold default values."""
    from controllers.label_manager import LabelManager

    window = QWidget()
    window.del_input_label_action = _Action()
    window.del_reg_label_action = _Action()
    window.del_all_labels_action = _Action()
    window.fusion_result = None
    window.registration_results = []
    window.current_result_index = -1
    window.stack_images = []
    window.current_display_index = -1
    window.messages = []
    window.statusBar = lambda: SimpleNamespace(
        showMessage=lambda text, timeout=0: window.messages.append(text))

    manager = LabelManager(window)
    manager.target_stack_combo = QComboBox()
    manager.target_stack_combo.addItems(["input", "registered"])
    manager.target_stack_combo.setCurrentIndex(target_index)
    manager.format_edit = QLineEdit("{value}")
    manager.text_edit = QLineEdit("")
    manager.range_edit = QLineEdit("All")
    manager.starting_value_spin = QSpinBox()
    manager.starting_value_spin.setValue(1)
    manager.interval_spin = QSpinBox()
    manager.interval_spin.setValue(1)
    manager.x_location_spin = QSpinBox()
    manager.y_location_spin = QSpinBox()
    manager.font_size_spin = QSpinBox()
    manager.font_size_spin.setValue(24)
    manager.font_family_combo = QFontComboBox()
    manager.transparent_bg_checkbox = QCheckBox()
    return window, manager


@pytest.mark.parametrize("target_index, own, other", [
    (0, "_input_enabled", "_registered_enabled"),
    (1, "_registered_enabled", "_input_enabled"),
])
def test_adding_a_label_enables_remove_all(qapp, monkeypatch, target_index,
                                           own, other):
    import controllers.label_manager as module

    monkeypatch.setattr(module, "show_warning_box", lambda *a, **k: None)
    monkeypatch.setattr(module, "show_success_box", lambda *a, **k: None)
    window, manager = _annotated_manager(qapp, target_index)
    manager._apply_label_configuration()

    assert getattr(manager, own) is True
    assert getattr(manager, other) is False
    assert window.del_all_labels_action.enabled
    assert window.del_reg_label_action.enabled == (target_index == 1)
    assert window.del_input_label_action.enabled == (target_index == 0)

    manager.delete_all_labels()
    assert not window.del_all_labels_action.enabled
    assert not window.del_reg_label_action.enabled
    assert not window.del_input_label_action.enabled
    assert manager._input_enabled is False
    assert manager._registered_enabled is False


def test_delete_all_keeps_the_other_targets_entry_when_one_remains(qapp,
                                                                  monkeypatch):
    import controllers.label_manager as module

    monkeypatch.setattr(module, "show_warning_box", lambda *a, **k: None)
    monkeypatch.setattr(module, "show_success_box", lambda *a, **k: None)
    from utils.label_model import Label
    from utils.validators import LabelConfig

    window, manager = _annotated_manager(qapp, 1)
    for target in ("input", "registered"):
        label = Label(LabelConfig())
        label.config.target = target
        label.config.format = "SAMPLE"
        manager.labels.add(label)
    manager._sync_delete_actions()

    manager.delete_registered_labels()
    assert window.del_reg_label_action.enabled is False
    assert window.del_input_label_action.enabled is True
    assert window.del_all_labels_action.enabled is True

    manager.delete_input_labels()
    assert window.del_all_labels_action.enabled is False


class _ScaleWindow:
    def __init__(self):
        self.scale_bar_enabled = True
        self.scale_bar_target = "all"
        self.persisted = 0
        self.refreshed = 0
        self.messages = []

    def persist_settings(self):
        self.persisted += 1

    def refresh_annotation_views(self):
        self.refreshed += 1

    def statusBar(self):  # noqa: N802
        return SimpleNamespace(
            showMessage=lambda text, timeout=0: self.messages.append(text))


def _remove(window, where):
    from main import OpenFocus
    OpenFocus.remove_scale_bar(window, where)


def test_remove_scale_bar_from_everywhere_disables_it():
    from locales import trans

    window = _ScaleWindow()
    _remove(window, "all")
    assert window.scale_bar_enabled is False
    assert window.scale_bar_target == "all"
    assert window.persisted == 1 and window.refreshed == 1
    assert window.messages[-1] == trans.t("scalebar_removed_all")


def test_remove_scale_bar_from_one_canvas_narrows_the_scope():
    window = _ScaleWindow()
    _remove(window, "source")
    assert window.scale_bar_enabled is True
    assert window.scale_bar_target == "output"

    window.scale_bar_target = "all"
    _remove(window, "output")
    assert window.scale_bar_target == "source"


def test_removing_the_last_canvas_turns_the_bar_off():
    window = _ScaleWindow()
    window.scale_bar_target = "output"
    _remove(window, "output")
    assert window.scale_bar_enabled is False


def test_removing_from_a_canvas_without_the_bar_changes_nothing():
    window = _ScaleWindow()
    window.scale_bar_target = "output"
    _remove(window, "source")
    assert window.scale_bar_enabled is True
    assert window.scale_bar_target == "output"


def test_remove_scale_bar_without_a_bar_reports_it():
    from locales import trans

    window = _ScaleWindow()
    window.scale_bar_enabled = False
    _remove(window, "all")
    assert window.scale_bar_enabled is False
    assert window.persisted == 0 and window.refreshed == 0
    assert window.messages[-1] == trans.t("scalebar_none_msg")
