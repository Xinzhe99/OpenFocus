"""R1/R2: render-setting lock scope and quick-preview draft export."""
import numpy as np
import pytest


class _Widget:
    def __init__(self):
        self.enabled = True
        self.text = ""
        self.tooltip = ""

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setText(self, text):
        self.text = text

    def setToolTip(self, text):
        self.tooltip = text


class _List:
    def __init__(self, item=None):
        self._item = item
        self.cleared = False
        self.selected_row = 0

    def currentItem(self):
        return self._item

    def clearSelection(self):
        self.cleared = True

    def setCurrentRow(self, row):
        self.selected_row = row


class _Items:
    def __init__(self):
        self.saved = []

    def save_output_image_as(self, item):
        self.saved.append(item)


class _Window:
    def __init__(self, item=None):
        self.chk_quick_preview = _Widget()
        self.chk_exposure_match = _Widget()
        self.chk_base_correct = _Widget()
        self.chk_reverse_order = _Widget()
        self.chk_sharpen_output = _Widget()
        self.btn_render_settings = _Widget()
        self.btn_render = _Widget()
        self.btn_reset = _Widget()
        self.combo_align = _Widget()
        self.output_list = _List(item)
        self.output_manager = _Items()
        self.fusion_result = np.zeros((2, 2), np.uint8)
        self.preview_result = None


def _render_manager(window):
    from controllers.render_manager import RenderManager
    return RenderManager(window)


def test_render_setting_widgets_lock_and_restore():
    window = _Window()
    manager = _render_manager(window)
    manager._set_render_settings_enabled(False)
    assert all(not widget.enabled for widget in manager._render_setting_widgets())
    manager._restore_ui_controls()
    assert all(widget.enabled for widget in manager._render_setting_widgets())


def test_compare_mode_keeps_render_settings_locked():
    window = _Window()
    manager = _render_manager(window)
    manager._compare_mode = True
    manager._compare_total = 2
    manager._compare_queue = ["dct"]
    manager._restore_ui_controls()
    assert all(not widget.enabled for widget in manager._render_setting_widgets())


@pytest.mark.parametrize("kind", ["fusion", "registration"])
def test_preview_draft_without_selection_is_not_exported(monkeypatch, kind):
    import controllers.export_manager as export_manager
    window = _Window()
    window.fusion_result = np.zeros((2, 2), np.uint8)
    window.aligned_stack_dir = "old-aligned-stack"
    window.preview_result = {"kind": kind, "image": window.fusion_result}
    warnings = []
    dialogs = []
    monkeypatch.setattr(export_manager, "show_warning_box",
                        lambda *args, **kwargs: warnings.append(args))
    monkeypatch.setattr(export_manager.FileDialog, "getSaveFileName",
                        lambda *args, **kwargs: dialogs.append(args) or ("", ""))
    export_manager.ExportManager(window).save_result()
    assert warnings
    assert not dialogs


@pytest.mark.parametrize("kind", ["fusion", "registration"])
def test_selected_historical_result_survives_preview_draft(kind):
    item = object()
    window = _Window(item=item)
    window.preview_result = {"kind": kind, "image": window.fusion_result}
    from controllers.export_manager import ExportManager
    ExportManager(window).save_result()
    assert window.output_manager.saved == [item]
