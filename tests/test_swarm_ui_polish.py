"""Focused Qt regression tests for the swarm UI polish fixes."""
import os
import pytest

pytest.importorskip("PyQt6")


def test_groupbox_title_clears_border(qapp):
    from PyQt6.QtWidgets import QGroupBox, QStyle, QStyleOptionGroupBox, QVBoxLayout, QWidget
    from ui.styles import build_stylesheet

    for theme in ("dark", "light", "gray"):
        qss = build_stylesheet(theme)
        assert "margin-top: 18px;" in qss
        assert "top: 0px;" in qss

        host = QWidget()
        host.setStyleSheet(qss)
        layout = QVBoxLayout(host)
        box = QGroupBox("Registration", host)
        layout.addWidget(box)
        host.resize(320, 140)
        host.show()
        qapp.processEvents()
        try:
            opt = QStyleOptionGroupBox()
            box.initStyleOption(opt)
            label_rect = box.style().subControlRect(
                QStyle.ComplexControl.CC_GroupBox, opt,
                QStyle.SubControl.SC_GroupBoxLabel, box)
            frame_rect = box.style().subControlRect(
                QStyle.ComplexControl.CC_GroupBox, opt,
                QStyle.SubControl.SC_GroupBoxFrame, box)
            assert label_rect.bottom() <= frame_rect.top() + 2, (
                f"{theme}: label {label_rect} overlaps frame {frame_rect}")
        finally:
            host.close()
            host.deleteLater()


def test_panel_headers_info_labels_and_wipe_bar_geometry(qapp):
    from PyQt6.QtWidgets import QApplication
    from ui.image_panels import apply_panel_theme, create_result_panel, create_source_panel
    from ui.styles import get_theme_style

    prev_font = qapp.font()
    prev_qss = qapp.styleSheet()
    try:
        qapp.setStyleSheet(get_theme_style("light", font_size=13))
        src = create_source_panel()
        res = create_result_panel()
        apply_panel_theme(src.widget, "light")
        apply_panel_theme(res.widget, "light")

        assert res.title.objectName() == "panelTitleText"
        assert src.image_label.font().family() == QApplication.font().family()

        src.control_bar.show()
        src.info_label.setText("100 / 100")
        src.widget.resize(480, 360)
        src.widget.show()

        res.view_stack.setCurrentWidget(res.wipe_widget.parentWidget())
        res.wipe_bar.show()
        res.widget.resize(480, 360)
        res.widget.show()
        qapp.processEvents()

        assert src.info_label.width() >= src.info_label.fontMetrics().horizontalAdvance("100 / 100")
        assert res.chk_wipe_follow_left.y() < res.chk_wipe_follow_right.y()
        assert res.chk_wipe_follow_left.width() >= res.chk_wipe_follow_left.sizeHint().width()
        assert res.chk_wipe_follow_right.width() >= res.chk_wipe_follow_right.sizeHint().width()

        qapp.setStyleSheet(get_theme_style("light", font_size=19))
        follow = res.chk_wipe_follow_right
        follow.setText("Neuestes Ergebnis automatisch auswählen")
        badge = res.lbl_wipe_right_index
        full_title = "Ausrichtungsergebnis aus gespeichertem Stapel"
        badge.setText(full_title)
        res.widget.resize(245, 360)
        qapp.processEvents()
        assert res.widget.width() == 245
        assert res.slider_wipe_right.width() >= 60
        assert follow.width() < follow.sizeHint().width()
        assert follow.toolTip() == follow.text()
        assert badge.text().endswith("…") and badge.toolTip() == full_title
    finally:
        src.widget.close()
        res.widget.close()
        src.widget.deleteLater()
        res.widget.deleteLater()
        qapp.setFont(prev_font)
        qapp.setStyleSheet(prev_qss)


def test_wipe_labels_retranslate_on_language_change(qapp):
    from locales import trans
    from ui.image_panels import create_result_panel

    orig_lang = trans.current_lang
    res = create_result_panel()
    called = []
    res.widget.wipe_active = True
    res.widget._update_wipe_images = lambda: called.append(True)
    res.widget.show()
    try:
        trans.set_language("zh")
        qapp.processEvents()
        assert called
        labels = [lbl.text() for lbl in res.wipe_bar.findChildren(type(res.title))]
        assert trans.t("wipe_side_a") in labels
        assert trans.t("wipe_side_b") in labels
    finally:
        res.widget.close()
        res.widget.deleteLater()
        trans.set_language(orig_lang)
        qapp.processEvents()


def test_right_panel_status_labels_elide_with_tooltip(qapp):
    from ui.right_panel import create_right_panel

    comp = create_right_panel()
    try:
        long_text = "RAM used: 12.4 / 16.0 GB (very long memory string)"
        comp.widget.resize(280, 500)
        comp.widget.show()
        comp.lbl_status_memory.setText(long_text)
        qapp.processEvents()
        assert comp.lbl_status_memory.text().endswith("…")
        assert comp.lbl_status_memory.toolTip() == long_text

        needed = comp.lbl_status_memory.fontMetrics().horizontalAdvance(long_text)
        comp.widget.resize(max(900, int(needed * 2.2) + 180), 500)
        qapp.processEvents()
        assert comp.lbl_status_memory.text() == long_text
        assert comp.lbl_status_memory.toolTip() == ""

        loaded_text = "Loaded: 2 (0.2 MB/img)"
        comp.lbl_status_memory.setText("RAM: 8 / 16 GB")
        comp.lbl_status_loaded.setText(loaded_text)
        comp.lbl_status_gpu.setText("GPU: CPU")
        fm = comp.lbl_status_loaded.fontMetrics()
        min_fit_w = max(
            420,
            int(fm.horizontalAdvance(loaded_text) * 9 / 4) + 90,
            int(comp.lbl_status_memory.fontMetrics().horizontalAdvance("RAM: 8 / 16 GB") * 9 / 5) + 90,
        )
        comp.widget.resize(min_fit_w, 500)
        qapp.processEvents()
        assert comp.lbl_status_loaded.text() == loaded_text
    finally:
        comp.widget.close()
        comp.widget.deleteLater()


def test_set_panel_label_guards_deleted_checkbox(qapp):
    from ui.right_panel import ElidingCheckBox, set_panel_label

    chk = ElidingCheckBox("Test option", "Hint")
    chk.deleteLater()
    qapp.processEvents()
    set_panel_label(chk, "Updated", "Updated hint")


def test_font_menu_actions_have_trans_key(qapp):
    from tests.test_menu_translations import _menu_window
    from ui.menus import setup_menus

    win = _menu_window(qapp)
    setup_menus(win)
    try:
        for mode in ("large", "medium", "small", "custom"):
            key = f"ui_font_{mode}"
            assert win.ui_objs[key].property("trans_key") == key
    finally:
        win.close()
        win.deleteLater()


def test_algorithm_parameter_spinboxes_bounded_width(qapp):
    from core.algorithms import get_fusion
    from ui.algorithm_parameters import AlgorithmParameterPanel

    panel = AlgorithmParameterPanel()
    panel.set_parameters(get_fusion("openfocus_b").parameters)
    panel.resize(500, 200)
    panel.show()
    qapp.processEvents()
    try:
        for editor in panel.editors.values():
            assert editor.minimumWidth() >= 76
            assert editor.maximumWidth() <= 128
            assert editor.width() <= 128
            assert editor.width() >= editor.minimumSizeHint().width()
        # Multiple numeric parameters should be packed into a 2-column grid to avoid wasted height.
        r_geo = panel.editors["radius"].geometry()
        s_geo = panel.editors["smoothing"].geometry()
        f_geo = panel.editors["feather"].geometry()
        assert abs(r_geo.y() - s_geo.y()) <= 4
        assert s_geo.x() > r_geo.x()
        assert f_geo.y() > r_geo.y()
    finally:
        panel.close()
        panel.deleteLater()


def test_crash_report_dialog_is_styled(qapp, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QMessageBox, QWidget
    from ui import crash_report

    monkeypatch.setattr(crash_report, "previous_crash", lambda _ack: crash_report.PreviousCrash(
        key="k1", crash_text="boom", log_text="log"))
    monkeypatch.setattr(crash_report, "get_settings", lambda: SimpleNamespace(
        value=lambda *_a, **_k: [], setValue=lambda *_a: None, sync=lambda: None))
    styled = []
    monkeypatch.setattr(crash_report, "style_message_box", lambda box: styled.append(box))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)

    parent = QWidget()
    try:
        crash_report.offer_previous_crash(parent)
        assert len(styled) == 1
    finally:
        parent.close()
        parent.deleteLater()


def test_locale_ampersands_escaped_and_batch_copy_consistent():
    from locales import trans

    assert trans.translations["en"]["advanced_tab_other"] == "Startup && updates"
    assert trans.translations["en"]["retouch_apply_close"] == "Apply && Close"
    assert trans.translations["de"]["retouch_apply_close"] == "Anwenden && schließen"
    assert trans.translations["en"]["batch_btn_add"] in trans.translations["en"]["batch_no_folders"]


def test_retouch_dialog_shows_header_and_provenance(qapp):
    import numpy as np
    from controllers.output_manager import OutputRecord
    from core.render_context import build_render_context
    from dialogs.retouch import RetouchDialog

    frames = [np.full((64, 80, 3), v, np.uint8) for v in (40, 120)]
    result = np.full((64, 80, 3), 90, np.uint8)
    ctx = build_render_context(frames, result, filenames=["f0.png", "f1.png"], algorithm="openfocus_a")
    rec = OutputRecord.create(result, "OpenFocus A", context=ctx)
    dlg = RetouchDialog(None, record=rec, apply_callback=lambda *_: (True, ""))
    dlg.resize(1280, 720)
    dlg.show()
    qapp.processEvents()
    try:
        assert dlg.lbl_title.isVisible()
        assert dlg.lbl_source_summary.isVisible()
        assert "80×64" in dlg.lbl_title.text()
        assert "openfocus_a" in dlg.lbl_source_summary.text()
        from PyQt6.QtCore import QPoint
        restore_bottom = dlg.btn_restore_all.mapTo(
            dlg.side_scroll.viewport(),
            QPoint(0, dlg.btn_restore_all.height()),
        ).y()
        assert restore_bottom <= dlg.side_scroll.viewport().height()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_advanced_settings_dialog_ok_primary_and_reg_spin_unclipped(qapp):
    from PyQt6.QtWidgets import QDialogButtonBox, QWidget
    from dialogs.advanced_settings import AdvancedSettingsDialog

    host = QWidget()
    dlg = AdvancedSettingsDialog(host)
    dlg.show()
    qapp.processEvents()
    try:
        box = dlg.findChild(QDialogButtonBox)
        ok_btn = box.button(QDialogButtonBox.StandardButton.Ok)
        assert ok_btn.objectName() == "primaryButton"
        assert ok_btn.isDefault()
        dlg.tabs.setCurrentIndex(2)
        qapp.processEvents()
        assert dlg.spin_reg_width.width() >= dlg.spin_reg_width.minimumSizeHint().width()
    finally:
        dlg.close()
        dlg.deleteLater()
        host.close()
        host.deleteLater()


def test_depthmap_falls_back_to_current_stack_when_selected_output_has_no_context(qapp):
    import numpy as np
    from PyQt6.QtWidgets import QListWidget, QWidget
    from controllers.output_manager import OutputRecord
    from dialogs.depthmap import DepthMapDialog

    frames = [np.full((48, 64, 3), v, np.uint8) for v in (20, 200)]
    rec = OutputRecord.create(frames[0], "Legacy")
    host = QWidget()
    host.raw_images = frames
    host.image_filenames = ["a.png", "b.png"]
    host.output_list = QListWidget()
    host.output_list.addItem("Legacy")
    host.output_list.setCurrentRow(0)
    host.output_list.item(0).setSelected(True)
    host.output_manager = type("O", (), {"record_for_item": staticmethod(lambda _i: rec)})()

    dlg = DepthMapDialog(host)
    dlg.show()
    qapp.processEvents()
    try:
        assert dlg.source_combo.currentData() == "current"
        assert dlg.compute_btn.isEnabled()
    finally:
        dlg.close()
        dlg.deleteLater()
        host.close()
        host.deleteLater()


def test_scalebar_and_watermark_button_bars_and_combos(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDialogButtonBox, QWidget
    from dialogs.scalebar import ScaleBarDialog
    from dialogs.watermark import WatermarkDialog

    host = QWidget()
    sb = ScaleBarDialog(host)
    wm = WatermarkDialog(host)
    sb.show()
    wm.show()
    qapp.processEvents()
    try:
        for dlg in (sb, wm):
            bar = dlg.findChild(QDialogButtonBox, "dialogButtonBar")
            assert bar is not None
            assert bar.testAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        assert sb.output_combo.width() >= sb.output_combo.minimumSizeHint().width()
        assert sb.font_combo.width() >= sb.font_combo.minimumSizeHint().width()
    finally:
        sb.close()
        wm.close()
        sb.deleteLater()
        wm.deleteLater()
        host.close()
        host.deleteLater()


def test_batch_converter_idle_progress_hidden_and_primary_start(qapp):
    from dialogs.batch_converter import BatchConverterDialog

    dlg = BatchConverterDialog()
    dlg.show()
    qapp.processEvents()
    try:
        assert not dlg.progress.isVisible()
        assert dlg.start.objectName() == "primaryButton"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_environment_info_dialog_uses_monospace_and_fits_screen(qapp):
    from PyQt6.QtGui import QFontInfo
    from dialogs.about import EnvironmentInfoDialog
    from ui.styles import build_stylesheet

    original_style = qapp.styleSheet()
    dlg = EnvironmentInfoDialog()
    dlg.show()
    qapp.processEvents()
    try:
        qapp.setStyleSheet(build_stylesheet("dark", font_size=18))
        qapp.processEvents()
        report_font = QFontInfo(dlg.text_edit.document().defaultFont())
        assert report_font.fixedPitch()
        screen = dlg.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            assert dlg.width() <= avail.width()
            assert dlg.height() <= avail.height()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.setStyleSheet(original_style)


def test_custom_scheme_dialog_sets_contrasting_stop_foreground(qapp):
    from dialogs.depthmap import _CustomSchemeDialog

    dlg = _CustomSchemeDialog(
        "ContrastCheck",
        [[0.0, "#000000"], [1.0, "#FFFFFF"]],
    )
    try:
        dark_item = dlg.listw.item(0)
        light_item = dlg.listw.item(1)
        assert dark_item.foreground().color().name().lower() == "#ffffff"
        assert light_item.foreground().color().name().lower() == "#111111"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_advanced_settings_gallery_cache_scan_is_lazy_and_cached(qapp):
    import time
    from PyQt6.QtWidgets import QWidget
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from widgets.danmaku_overlay import DanmakuSettings

    class FakeGalleryController:
        def __init__(self):
            self.usage_calls = 0
            self.clear_calls = 0

        def danmaku_settings(self):
            return DanmakuSettings()

        def cache_max_mb(self):
            return 256

        def cache_usage_bytes(self):
            self.usage_calls += 1
            return 12 * 1024 * 1024

        def clear_disk_cache(self):
            self.clear_calls += 1
            return 3, 12 * 1024 * 1024

    class FakeWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.gallery_controller = FakeGalleryController()

    win = FakeWindow()
    dlg = AdvancedSettingsDialog(win)
    try:
        # __init__ must NOT scan disk synchronously
        assert win.gallery_controller.usage_calls == 0
        # Changing cache limit spinbox must NOT rescan disk
        dlg.spin_gallery_cache_limit.setValue(512)
        assert win.gallery_controller.usage_calls == 0

        # Switching to Gallery tab triggers async scan once
        dlg.show()
        dlg.tabs.setCurrentIndex(dlg._gallery_tab_index)
        deadline = time.monotonic() + 2.0
        while not dlg._gallery_cache_scanned and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert dlg._gallery_cache_scanned is True
        assert win.gallery_controller.usage_calls == 1
        assert "12.0" in dlg.lbl_gallery_cache_usage.text()

        # Spinbox step reuses cached usage bytes without rescanning
        dlg.spin_gallery_cache_limit.setValue(1024)
        assert win.gallery_controller.usage_calls == 1
        assert "1024" in dlg.lbl_gallery_cache_usage.text()

        # Clear cache runs asynchronously and resets cached usage to 0.0 MB
        dlg.btn_clear_gallery_cache.click()
        deadline = time.monotonic() + 2.0
        while not dlg.btn_clear_gallery_cache.isEnabled() and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert win.gallery_controller.clear_calls == 1
        assert "0.0" in dlg.lbl_gallery_cache_usage.text()
    finally:
        dlg.close()
        dlg.deleteLater()
        win.deleteLater()
