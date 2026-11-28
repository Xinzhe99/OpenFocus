"""Regression tests for compact parameters, single-row controls, and context help."""
from PyQt6.QtCore import QPoint, QTimer, Qt, QUrl
from PyQt6.QtGui import QImage, QKeyEvent, QMouseEvent, QTextDocument
from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton, QVBoxLayout, QWidget
import pytest

from core.algorithms import get_fusion, get_registration
from dialogs.help import HelpDialog
from locales import trans
from ui.algorithm_parameters import AlgorithmParameterPanel
from ui.context_help import (
    ContextHelpBeacon,
    ContextHelpManager,
    IllustratedHelpDialog,
    register_help_target,
    resolve_help_target,
)
from ui.context_help_diagrams import render_help_diagram
from ui.right_panel import create_right_panel


def test_context_help_beacon_click_pick_and_escape_restores_cursor(qapp):
    owner = QWidget()
    layout = QVBoxLayout(owner)
    beacon = ContextHelpBeacon(owner)
    target = QPushButton("Target", owner)
    register_help_target(target, "fusion")
    layout.addWidget(beacon)
    layout.addWidget(target)
    manager = ContextHelpManager(owner)
    manager.attach_beacon(beacon)
    owner.resize(320, 180)
    owner.show()
    qapp.processEvents()
    try:
        assert QApplication.overrideCursor() is None
        manager.toggle_pick_mode(beacon)
        assert manager.mode() == "pick"
        assert beacon.isChecked()
        assert QApplication.overrideCursor() is not None

        esc = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(owner, esc)
        qapp.processEvents()
        assert manager.mode() == "idle"
        assert not beacon.isChecked()
        assert QApplication.overrideCursor() is None
    finally:
        manager.uninstall()
        owner.close()
        owner.deleteLater()


def test_dragging_beacon_onto_target_opens_illustrated_help(qapp, monkeypatch):
    monkeypatch.setattr(trans, "current_lang", "zh")
    owner = QWidget()
    layout = QVBoxLayout(owner)
    beacon = ContextHelpBeacon(owner)
    target = QPushButton("Fusion Area", owner)
    register_help_target(target, "fusion")
    layout.addWidget(beacon)
    layout.addWidget(target)
    manager = ContextHelpManager(owner)
    manager.attach_beacon(beacon)
    owner.resize(360, 220)
    owner.show()
    qapp.processEvents()
    try:
        shown = []

        def capture():
            dlg = owner.findChild(HelpDialog)
            if dlg is None:
                return
            doc = dlg.text_browser.document()
            img = doc.resource(QTextDocument.ResourceType.ImageResource.value, QUrl("parameter-help.png"))
            shown.append((dlg.windowTitle(), doc.toPlainText(), img))
            dlg.accept()

        manager.begin_drag_help(beacon)
        assert manager.mode() == "drag"
        drop_pos = target.mapToGlobal(target.rect().center())
        manager.update_pointer(drop_pos)

        QTimer.singleShot(0, capture)
        manager.finish_drag_help(drop_pos)
        assert len(shown) == 1
        title, text, img = shown[0]
        assert title == trans.t("ctx_help_fusion_title")
        assert trans.t("ctx_help_fusion_body") in text
        assert isinstance(img, QImage) and not img.isNull()
        assert QApplication.overrideCursor() is None
    finally:
        manager.uninstall()
        owner.close()
        owner.deleteLater()


def test_hold_h_intercepts_click_in_dialog_and_skips_text_inputs(qapp, monkeypatch):
    monkeypatch.setattr(trans, "current_lang", "en")
    owner = QWidget()
    lay = QVBoxLayout(owner)
    edit = QLineEdit(owner)
    btn_clicks = {"n": 0}
    action_btn = QPushButton("Trigger", owner)
    action_btn.clicked.connect(lambda: btn_clicks.__setitem__("n", btn_clicks["n"] + 1))
    register_help_target(action_btn, "render_action")
    lay.addWidget(edit)
    lay.addWidget(action_btn)
    manager = ContextHelpManager(owner)
    owner.resize(320, 180)
    owner.show()
    qapp.processEvents()
    try:
        # While typing in a QLineEdit, bare H must NOT enter hold_h help mode.
        edit.setFocus()
        qapp.processEvents()
        h_press = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_H, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(edit, h_press)
        assert manager.mode() == "idle"
        assert QApplication.overrideCursor() is None

        # When focus is on a non-text widget, holding H enters hold_h and intercepts clicks.
        action_btn.setFocus()
        qapp.processEvents()
        h_press = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_H, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(action_btn, h_press)
        assert manager.mode() == "hold_h"
        assert QApplication.overrideCursor() is not None

        shown = []

        def capture():
            dlg = owner.findChild(IllustratedHelpDialog)
            if dlg is None:
                return
            shown.append(dlg.windowTitle())
            dlg.accept()

        QTimer.singleShot(0, capture)
        local = action_btn.rect().center()
        gp = action_btn.mapToGlobal(local)
        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            local.toPointF(),
            gp.toPointF(),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(action_btn, press)
        qapp.processEvents()
        assert btn_clicks["n"] == 0
        assert shown == [trans.t("ctx_help_render_action_title")]
        assert QApplication.overrideCursor() is None
    finally:
        manager.uninstall()
        owner.close()
        owner.deleteLater()


@pytest.mark.parametrize("theme", ["light", "dark", "gray"])
@pytest.mark.parametrize(
    "kind",
    [
        "overview", "registration", "fusion", "align_area", "align_preview",
        "source_canvas", "sharpness_curve", "result_canvas", "wipe_compare",
        "render_action", "live_preview", "render_settings", "reset_defaults",
        "render_queue", "source_list", "output_list", "status_panel",
        "radius", "blend", "kernel", "levels", "ai_mode", "reference",
        "tiles", "registration_width", "align_limits", "align_auto_crop",
        "align_interpolation", "live_preview_histogram", "live_preview_split",
        "live_preview_size", "live_preview_stack",
    ],
)
def test_all_help_diagrams_render_cleanly(qapp, theme, kind):
    img = render_help_diagram(kind, theme)
    assert isinstance(img, QImage) and not img.isNull()
    assert img.pixelColor(0, 0) != img.pixelColor(60, 37)


def test_compact_parameter_panel_keeps_label_inline_and_bounds_spinbox_width(qapp, monkeypatch):
    monkeypatch.setattr(trans, "current_lang", "zh")
    panel = AlgorithmParameterPanel()
    panel.set_parameters(get_fusion("openfocus_b").parameters)
    panel.resize(340, 120)
    panel.show()
    qapp.processEvents()
    try:
        radius_lbl = panel._labels["radius"]
        radius_spin = panel.editors["radius"]
        smoothing_lbl = panel._labels["smoothing"]
        smoothing_spin = panel.editors["smoothing"]
        assert abs(radius_lbl.y() - radius_spin.y()) <= 6
        assert abs(radius_spin.y() - smoothing_spin.y()) <= 6
        assert abs(smoothing_lbl.y() - smoothing_spin.y()) <= 6
        assert radius_spin.width() <= 96
        assert smoothing_spin.width() <= 96
        reg_panel = AlgorithmParameterPanel()
        reg_panel.set_parameters(get_registration("ecc").parameters)
        reg_panel.resize(340, 220)
        reg_panel.show()
        qapp.processEvents()
        try:
            for key in ("max_shift_x_ratio", "max_rotation_deg"):
                assert abs(reg_panel._labels[key].y() - reg_panel.editors[key].y()) <= 6
                assert reg_panel.editors[key].width() <= 96
                assert resolve_help_target(reg_panel.editors[key])[1] == "align_limits"
        finally:
            reg_panel.close()
            reg_panel.deleteLater()
    finally:
        panel.close()
        panel.deleteLater()


def test_render_more_menu_mirrors_queue_and_compare_buttons(qapp):
    parts = create_right_panel()
    parts.widget.resize(380, 680)
    parts.widget.show()
    qapp.processEvents()
    try:
        more = parts.btn_render_more
        assert more is not None and more.width() == 34 and more.height() == 34
        menu = more._popup_menu
        menu.aboutToShow.emit()
        texts = [a.text() for a in menu.actions() if not a.isSeparator()]
        assert parts.btn_enqueue.full_text() in texts
        assert parts.btn_render_queue.full_text() in texts
        assert parts.btn_compare.full_text() in texts
        more.set_badge_count(3)
        assert more._badge_count == 3
    finally:
        parts.widget.close()
        parts.widget.deleteLater()


def test_hold_h_works_inside_sub_dialogs(qapp, tmp_path, monkeypatch):
    from dialogs.render_settings import RenderSettingsDialog
    from tests.test_control_channel import _destroy_main_window, _new_main_window

    monkeypatch.setattr(trans, "current_lang", "zh")
    window = _new_main_window(qapp, tmp_path, monkeypatch, use_gpu=False)
    dlg = RenderSettingsDialog(window)
    try:
        dlg.show()
        qapp.processEvents()
        before = window.chk_quick_preview.isChecked()
        dlg.activateWindow()
        window.chk_quick_preview.setFocus()
        qapp.processEvents()

        h_press = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_H, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(dlg, h_press)
        assert window._context_help_manager.mode() == "hold_h"

        shown = []

        def capture():
            help_dlg = dlg.findChild(IllustratedHelpDialog) or window.findChild(IllustratedHelpDialog)
            if help_dlg is None:
                return
            shown.append(help_dlg.windowTitle())
            help_dlg.accept()

        QTimer.singleShot(0, capture)
        local = window.chk_quick_preview.rect().center()
        gp = window.chk_quick_preview.mapToGlobal(local)
        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            local.toPointF(),
            gp.toPointF(),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(window.chk_quick_preview, press)
        qapp.processEvents()
        assert shown == [trans.t("ctx_help_opt_quick_preview_title")]
        assert window.chk_quick_preview.isChecked() == before
        assert QApplication.overrideCursor() is None
    finally:
        dlg.close()
        dlg.deleteLater()
        _destroy_main_window(qapp, window)
