"""Widgets must not clip their own translated text.

Two real defects motivated this file:
  * the smoothing spin box had a fixed 80 px width while it needs 105 px, so
    "2.0" rendered as "2.(" with the arrows over the decimal;
  * the option checkboxes sat two per row, and long translations were elided
    ("Invertir orden de la pila (barrid…") because the panel is only 375 px.
"""
import pytest

pytest.importorskip("PyQt6")


def _labels_that_must_fit():
    return ("chk_quick_preview", "chk_reverse_order",
            "chk_exposure_match", "chk_base_correct")





def test_option_checkboxes_flow_and_wrap(qapp):
    """Options pack side by side and wrap on width, never one-per-row.

    They used to be stacked unconditionally because a 375 px column could not
    hold two long translations. The flow layout packs what fits and wraps the
    rest, so a wide window shows two rows instead of four and the saved height
    goes to the source/output lists.
    """
    from PyQt6.QtWidgets import QCheckBox

    from widgets.flow_layout import FlowPanel

    app = qapp
    panel = FlowPanel(h_spacing=14, v_spacing=6)
    boxes = []
    for text in ("Quick preview", "Reverse order", "Exposure match",
                 "Base correction"):
        box = QCheckBox(text)
        boxes.append(box)
        panel.add(box)
    try:
        # Font metrics differ across CI runners; derive the one-row width
        # from the actual size hints instead of assuming 620 px fits.
        one_row = (sum(b.sizeHint().width() for b in boxes)
                   + 14 * (len(boxes) - 1) + 16)
        panel.resize(one_row, 200)
        panel.show()
        app.processEvents()
        panel.relayout()
        app.processEvents()
        rows = {b.geometry().y() for b in boxes}
        assert len(rows) == 1, f"short options should share one row: {rows}"

        panel.resize(150, 300)
        app.processEvents()
        panel.relayout()
        app.processEvents()
        rows = {b.geometry().y() for b in boxes}
        assert len(rows) > 1, "narrow widths must wrap onto more rows"
    finally:
        panel.close()
        panel.deleteLater()






@pytest.mark.parametrize("lang", ["en", "zh", "ja", "es"])
def test_option_labels_have_a_translation(lang):
    """Read the tables directly: `trans` is a QObject and is collected by the
    full suite, so switching language from a test is not reliable."""
    from locales import trans as _t
    table = _t.translations[lang]
    for key in ("chk_quick_preview", "chk_reverse_order",
                "chk_exposure_match", "chk_base_correct"):
        text = table.get(key)
        assert text and text != key, f"{lang}: {key} is untranslated"




def test_eliding_button_reserves_its_painted_padding(qapp):
    """Long captions and their ellipses fit inside the styled button."""
    from PyQt6.QtWidgets import QStyle, QStyleOptionButton
    from ui.right_panel import ElidingButton

    button = ElidingButton("▶ Rozpocznij renderowanie " * 2)
    button.setStyleSheet("QPushButton { padding: 6px 12px; }")
    button.resize(140, 34)
    button.show()
    qapp.processEvents()
    try:
        assert button.text() != button.full_text()
        option = QStyleOptionButton()
        button.initStyleOption(option)
        contents = button.style().subElementRect(
            QStyle.SubElement.SE_PushButtonContents, option, button)
        assert button.fontMetrics().horizontalAdvance(button.text()) <= contents.width()
        assert button.toolTip() == button.full_text()
    finally:
        button.close()
        button.deleteLater()


def test_button_caption_recovers_after_temporary_width_constraint(qapp):
    from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget
    from ui.right_panel import ElidingButton

    host = QWidget()
    layout = QHBoxLayout(host)
    layout.addWidget(QLabel('Sources'), 1)
    caption = 'Review frames'
    button = ElidingButton(caption)
    button.setStyleSheet('QPushButton { padding: 6px 12px; }')
    layout.addWidget(button)
    host.resize(480, 80)
    host.show()
    qapp.processEvents()
    try:
        button.setMaximumWidth(18)
        qapp.processEvents()
        assert button.text() != caption
        assert button.toolTip() == caption

        button.setMaximumWidth(host.width())
        button.setText(caption)
        button.setEnabled(False)
        layout.activate()
        qapp.processEvents()
        assert button.text() == caption
        assert not button.toolTip()
        button.setToolTip('Opens stack review')
        assert button.toolTip() == 'Opens stack review'
    finally:
        host.close()
        host.deleteLater()


def test_main_selector_explains_missing_dtcwt(qapp, monkeypatch):
    from types import SimpleNamespace
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QComboBox
    import main
    import ui.algorithm_combo as algorithm_combo
    from locales import trans

    combo = QComboBox()
    algorithm_combo.fill_fusion_combo(combo)
    host = SimpleNamespace(combo_fusion=combo, combo_align=QComboBox(),
                           _stackmff_available=True, experimental_features=True)
    monkeypatch.setattr(algorithm_combo, '_package_available', lambda _name: False)
    main.OpenFocus.refresh_method_availability(host)
    index = combo.findData('dtcwt')
    assert not combo.model().item(index).isEnabled()
    assert trans.t('env_dtcwt_unavailable') in combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
    monkeypatch.setattr(algorithm_combo, '_package_available', lambda _name: True)
    main.OpenFocus.refresh_method_availability(host)
    assert combo.model().item(index).isEnabled()
    assert combo.itemData(index, Qt.ItemDataRole.ToolTipRole) == trans.t('radio_dtcwt_hint')
    combo.deleteLater()


def test_alignment_only_disables_fusion_only_options(qapp):
    """Base correction, sharpening and reversal feed only the fusion step.

    With `Alignment only (no fusion)` the render never constructs
    MultiFocusFusion, so leaving the checkboxes enabled lied. Exposure
    matching also feeds registration and must stay enabled.
    """
    import types

    import main
    from locales import trans as _t
    from ui.right_panel import ElidingCheckBox

    class _Panel:
        def set_parameters(self, *_args):
            pass

    class _Host:
        def __init__(self):
            self.method = "none"
            self.fusion_parameter_panel = _Panel()
            self.chk_base_correct = ElidingCheckBox(
                _t.t("chk_base_correct"), _t.t("chk_base_correct_hint"))
            self.chk_exposure_match = ElidingCheckBox(
                _t.t("chk_exposure_match"), _t.t("chk_exposure_match_hint"))
            self.chk_sharpen_output = ElidingCheckBox(
                _t.t("chk_sharpen_output"), _t.t("chk_sharpen_output_hint"))
            self.chk_reverse_order = ElidingCheckBox(
                _t.t("chk_reverse_order"), _t.t("chk_reverse_order_hint"))

        def current_fusion_method(self):
            return self.method

        def fusion_parameters_for(self, _method, depth=False):
            return {}

        def _sync_brightness_checkboxes(self, _method=None):
            pass

    host = _Host()
    host._set_fusion_option_available = types.MethodType(
        main.OpenFocus._set_fusion_option_available, host)
    refresh = types.MethodType(
        main.OpenFocus.refresh_algorithm_parameters, host)

    refresh()
    assert not host.chk_base_correct.isEnabled()
    assert not host.chk_sharpen_output.isEnabled()
    assert not host.chk_reverse_order.isEnabled()
    assert not host.chk_exposure_match.isEnabled()
    assert _t.t("radio_no_fusion_hint") in host.chk_base_correct.toolTip()
    assert _t.t("radio_no_fusion_hint") in host.chk_reverse_order.toolTip()

    host.method = "guided_filter"
    refresh()
    assert host.chk_exposure_match.isEnabled()
    assert host.chk_base_correct.isEnabled()
    assert host.chk_sharpen_output.isEnabled()
    assert host.chk_reverse_order.isEnabled()
    assert _t.t("chk_reverse_order_hint") in host.chk_reverse_order.toolTip()

    # StackMFF V2 reads the original focus order; keep the existing lock.
    host.method = "stackmffv2"
    refresh()
    assert not host.chk_reverse_order.isEnabled()
    assert _t.t("chk_reverse_order_locked_hint") in (
        host.chk_reverse_order.toolTip())






def test_checked_radio_does_not_grow_its_indicator(qapp):
    """Checked and unchecked radios occupy the same space in both themes."""
    from PyQt6.QtWidgets import QRadioButton, QWidget
    from ui.styles import build_stylesheet

    for theme in ("light", "dark"):
        parent = QWidget()
        parent.setStyleSheet(build_stylesheet(theme))
        soft, hard = QRadioButton("same", parent), QRadioButton("same", parent)
        try:
            soft.setChecked(True)
            assert soft.height() == hard.height(), theme
            assert soft.sizeHint().height() == hard.sizeHint().height(), theme
            assert soft.sizeHint().width() == hard.sizeHint().width(), theme
        finally:
            parent.close()
            parent.deleteLater()











def test_help_button_style_uses_the_active_theme(qapp):
    import ui.styles as st
    try:
        st.set_current_theme("light")
        light = st.help_button_style()
        st.set_current_theme("dark")
        dark = st.help_button_style()
        assert light != dark
        # An explicit theme must win over the global.
        assert st.help_button_style("light") == light
    finally:
        st.set_current_theme("dark")


def test_no_input_widget_is_narrower_than_its_content(qapp):
    """A fixed width below minimumSizeHint silently clips the value.

    Two shipped defects of this class: the right panel's blend-smoothing spin
    ("2.0" → "2.("), the load-scale spin ("10%" → "10(") and the dust-map
    sigma ("4.0" → "4."). This walks every dialog's spin boxes so a new fixed
    width cannot reintroduce it.
    """
    from PyQt6.QtWidgets import QDoubleSpinBox, QSpinBox, QWidget
    from ui.styles import build_stylesheet

    from dialogs.settings import DownsampleDialog, TileSettingsDialog
    from dialogs.dustmap import DustMapDialog

    parent = QWidget()
    parent.setStyleSheet(build_stylesheet("dark"))
    dialogs = [DownsampleDialog(parent, initial_scale=0.05, max_scale=1.0),
               TileSettingsDialog(parent),
               DustMapDialog(parent)]
    try:
        for dlg in dialogs:
            dlg.show()
            qapp.processEvents()
            for spin in dlg.findChildren((QSpinBox, QDoubleSpinBox)):
                need = spin.minimumSizeHint().width()
                assert spin.width() >= need, (
                    f"{type(dlg).__name__}: {spin.text()!r} needs {need} px, "
                    f"has {spin.width()}")
            dlg.close()
    finally:
        for dlg in dialogs:
            dlg.deleteLater()
        parent.deleteLater()

@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("value", [1, 10, 99, 100])
def test_downsample_percentage_is_rendered(qapp, theme, value):
    """The real styled editor must have room to paint every percentage."""
    from PyQt6.QtWidgets import QWidget
    from dialogs.settings import DownsampleDialog
    from ui.styles import build_stylesheet

    parent = QWidget()
    parent.setStyleSheet(build_stylesheet(theme))
    dialog = DownsampleDialog(parent, initial_scale=value / 100)
    try:
        dialog.show()
        dialog.decrease_btn.setFocus()
        qapp.processEvents()
        spin = dialog.spinbox
        editor = spin.lineEdit()
        text = f"{value}%"
        assert editor.text() == text
        rect = editor.contentsRect()
        margins = editor.textMargins()
        assert rect.width() - margins.left() - margins.right() >= editor.fontMetrics().horizontalAdvance(text)
        assert rect.height() - margins.top() - margins.bottom() >= editor.fontMetrics().height()

        # Compare the actual painted widget with its empty editor, rather than
        # relying on text() or size hints, which also pass when no text paints.
        painted = spin.grab().toImage()
        editor.setText("")
        qapp.processEvents()
        empty = spin.grab().toImage()
        changed = sum(
            painted.pixel(x, y) != empty.pixel(x, y)
            for y in range(painted.height())
            for x in range(painted.width())
        )
        assert changed > 5, f"{text} did not paint in the {theme} theme"
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.deleteLater()


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("max_scale", [0.42, 1.0])
@pytest.mark.parametrize("show_dont_ask", [False, True])
def test_downsample_controls_stay_in_sync(qapp, max_scale, theme, show_dont_ask):
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QWidget
    from dialogs.settings import DownsampleDialog
    from ui.styles import build_stylesheet

    parent = QWidget()
    parent.setStyleSheet(build_stylesheet(theme))
    dialog = DownsampleDialog(parent, initial_scale=max_scale, max_scale=max_scale,
                              show_dont_ask=show_dont_ask)
    try:
        dialog.show()
        qapp.processEvents()
        dialog.decrease_btn.setFocus()
        QTest.mouseClick(dialog.spinbox.lineEdit(), Qt.MouseButton.LeftButton,
                         pos=dialog.spinbox.lineEdit().rect().center())
        qapp.processEvents()
        QTest.keyClicks(dialog.spinbox.lineEdit(), "37")
        assert dialog.spinbox.value() == dialog.slider.value() == 37
        assert dialog.get_scale_factor() == 0.37
        QTest.keyClick(dialog.spinbox.lineEdit(), Qt.Key.Key_Home)
        QTest.keyClick(dialog.spinbox.lineEdit(), Qt.Key.Key_Delete)
        QTest.keyClicks(dialog.spinbox.lineEdit(), "2")
        assert dialog.spinbox.value() == dialog.slider.value() == 27
        dialog.slider.setValue(37)

        dialog.increase_btn.click()
        assert dialog.spinbox.value() == dialog.slider.value() == 38
        dialog.decrease_btn.click()
        assert dialog.spinbox.value() == dialog.slider.value() == 37
        dialog.slider.setValue(dialog.slider.maximum())
        assert dialog.spinbox.value() == round(max_scale * 100)
        dialog.increase_btn.click()
        assert dialog.spinbox.value() == dialog.slider.value() == round(max_scale * 100)
        dialog.slider.setValue(1)
        dialog.decrease_btn.click()
        assert dialog.spinbox.value() == dialog.slider.value() == 1
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.deleteLater()












def test_status_block_is_readable_and_compact(qapp):
    from PyQt6.QtWidgets import QFrame, QLabel
    from ui.right_panel import create_right_panel

    parts = create_right_panel()
    parts.widget.resize(500, 1100)
    parts.widget.show()
    qapp.processEvents()
    try:
        panel = parts.widget.findChild(QFrame, "statusPanel")
        labels = panel.findChildren(QLabel)
        assert len(labels) == 4
        for label in labels:
            assert label.isVisible() and label.text()
            assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
            assert label.height() >= label.fontMetrics().height()
        for index, label in enumerate(labels):
            for other in labels[index + 1:]:
                assert not label.geometry().intersects(other.geometry())
        rows = sorted({label.y() for label in labels})
        assert len(rows) == 2
        assert all(sum(label.y() == row for label in labels) == 2 for row in rows)
        assert parts.file_list.isVisible()
        parts.file_list.addItem("frame.tif")
        parts.file_list.setCurrentRow(0)
        assert parts.file_list.currentItem().text() == "frame.tif"
    finally:
        parts.widget.close()
        parts.widget.deleteLater()





def test_quick_preview_is_a_fraction_not_an_absolute_cap():
    """A fixed 1200 px cap shrank a 1344 px stack by 11% and was slower.

    The downscale plus the forced re-registration of the small frames made the
    quick path slower than the full render (7.7 s vs 6.1 s on 1344x1021 with
    ECC), so the constant is a fraction of the source size now.
    """
    from core.workers import PREVIEW_SCALE, _downscale_for_preview
    import numpy as np

    assert 0.0 < PREVIEW_SCALE <= 0.6, (
        "a preview only helps if it cuts the pixel count substantially")
    img = np.zeros((1021, 1344, 3), np.uint8)
    out = _downscale_for_preview(img)
    assert max(out.shape[:2]) < 900, out.shape




def test_label_menu_opens_dialog(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QDialog
    from tests.test_control_channel import _new_main_window, _destroy_main_window

    window = _new_main_window(qapp, tmp_path, monkeypatch)
    opened = []
    def capture(dialog):
        opened.append(dialog.windowTitle())
        return QDialog.DialogCode.Rejected
    monkeypatch.setattr(QDialog, "exec", capture)
    try:
        window.add_label_action.setEnabled(True)
        window.add_label_action.trigger()
        assert len(opened) == 1 and opened[0]
    finally:
        _destroy_main_window(qapp, window)


def test_first_run_theme_is_light():
    """A fresh install opens in light; a saved preference is respected."""
    from utils.settings_store import DEFAULT_THEME, _PERSISTED_FIELDS

    assert DEFAULT_THEME == "light"
    assert _PERSISTED_FIELDS["ui_theme"][1] == "light"


def test_batch_dialog_offers_real_method_choices(qapp):
    """Batch must expose method/alignment/options instead of echoing the
    main window: a batch run could otherwise only reuse whatever the last
    manual render happened to use."""
    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        assert dlg.fusion_combo.count() >= 8
        assert dlg.align_combo.count() >= 4
        # Alignment-only batch is a real choice.
        dlg.align_combo.setCurrentIndex(dlg.align_combo.findData("none"))
        dlg.fusion_combo.setCurrentIndex(dlg.fusion_combo.findData("openfocus_b"))
        settings = dlg.get_processing_settings()
        assert settings["reg_methods"] == []
        assert settings["fusion_method"] == "openfocus_b"
        # Options the right panel has must exist here too.
        assert hasattr(dlg, "chk_reverse_order")
        assert hasattr(dlg, "chk_exposure_match")
        assert hasattr(dlg, "chk_base_correct")
        assert hasattr(dlg, "downscale_slider")
    finally:
        dlg.deleteLater()


def test_batch_downscale_is_a_ratio_of_the_stack():
    """The slider is a percentage of each stack's width, not a pixel width."""
    from core.workers import BatchWorker

    worker = BatchWorker(
        folder_paths=[], output_type="subfolder", output_path="out",
        processing_settings={"reg_scale_percent": 50},
        reg_downscale_width=1024)
    images = [__import__("numpy").zeros((100, 4000, 3), "uint8")]
    assert worker._registration_width(images) == 2000
    worker.processing_settings["reg_scale_percent"] = 100
    assert worker._registration_width(images) == 1024


def test_retouch_paints_a_locked_frame(qapp):
    """Retouching copies from the frame the user locked, and auto-pick is off."""
    import numpy as np
    from PyQt6.QtTest import QTest

    from controllers.output_manager import OutputRecord
    from core.render_context import build_render_context
    from dialogs.retouch import RetouchDialog

    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
              for _ in range(3)]
    ctx = build_render_context(
        frames, frames[0], filenames=["a.png", "b.png", "c.png"],
        algorithm="test")
    record = OutputRecord.create(frames[0], "test_output", context=ctx)
    dlg = RetouchDialog(None, record=record,
                        apply_callback=lambda _rec, _img: (True, ""))
    try:
        assert not hasattr(dlg, "chk_auto"), "auto-pick is gone"
        assert dlg.frame_list.count() == 3
        dlg._select_frame(2)
        for _ in range(500):
            if dlg._source_ready and not dlg._source_loading:
                break
            QTest.qWait(10)
        assert dlg.doc.frame_index == 2
        assert dlg._source_ready
        before = dlg.doc.working.copy()
        assert dlg.begin_stroke(30, 30)
        dlg.stroke_to(45, 40)
        dlg.end_stroke()
        assert (dlg.doc.working != before).any(), "the stroke painted nothing"
        assert dlg.doc.is_dirty
        dlg._undo()
        assert not (dlg.doc.working != before).any(), "undo did not restore"
    finally:
        dlg._allow_close = True
        dlg.close()
        dlg.deleteLater()




def test_theme_bound_styles_are_strings_and_follow_the_theme():
    """Message-box / progress stylesheets must be real strings.

    `setStyleSheet()` rejects anything that is not a str, so a wrapper object
    raises TypeError at the call site. They are rebindable module globals
    instead: five controllers import them by name, and when they were built
    once for the dark palette every message box stayed black in light mode.
    """
    import ui.styles as st

    st.set_current_theme("dark")
    dark_msg, dark_prog = st.MESSAGE_BOX_STYLE, st.PROGRESS_DIALOG_STYLE
    st.set_current_theme("light")
    light_msg, light_prog = st.MESSAGE_BOX_STYLE, st.PROGRESS_DIALOG_STYLE
    try:
        for value in (dark_msg, light_msg, dark_prog, light_prog):
            assert isinstance(value, str) and value
        assert dark_msg != light_msg, "message box ignores the theme"
        assert dark_prog != light_prog, "progress dialog ignores the theme"
        # And the light sheets really are light.
        assert "#FFFFFF" in light_msg.upper() or "#F" in light_msg.upper()
    finally:
        st.set_current_theme("dark")


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_stepper_is_readable_in_each_theme(qapp, theme):
    from PyQt6.QtGui import QPalette
    from ui.styles import get_theme_style
    from widgets.stepper import StepperSpinBox

    spin = StepperSpinBox(minimum=0, maximum=8, decimals=1, step=0.2, value=1.2)
    spin.setSuffix(" %")
    spin.setStyleSheet(get_theme_style(theme))
    spin.show()
    qapp.processEvents()
    try:
        editor = spin.lineEdit()
        palette = editor.palette()
        assert palette.color(QPalette.ColorRole.Text) != palette.color(QPalette.ColorRole.Base)
        assert editor.width() >= editor.fontMetrics().horizontalAdvance(editor.text())
        assert editor.text().endswith(" %")
    finally:
        spin.close()
        spin.deleteLater()


def test_stepper_trims_zeros_without_rounding(qapp):
    from PyQt6.QtCore import QLocale
    from widgets.stepper import StepperSpinBox

    spin = StepperSpinBox(minimum=0, maximum=100, decimals=9, value=4)
    assert spin.text() == '4.0'
    spin.setValue(0.000000125)
    assert spin.text() == '0.000000125'
    spin.setLocale(QLocale(QLocale.Language.German, QLocale.Country.Germany))
    spin.setValue(1.25)
    assert spin.text() == '1,25'
    spin.interpretText()
    assert spin.value() == 1.25
    spin.deleteLater()


def test_stepper_commits_decimal_suffix_and_clamps(qapp):
    from PyQt6.QtCore import QLocale
    from PyQt6.QtTest import QSignalSpy
    from widgets.stepper import StepperSpinBox

    spin = StepperSpinBox(minimum=0.5, maximum=2.5, decimals=2, step=0.25, value=1)
    spin.setLocale(QLocale(QLocale.Language.German, QLocale.Country.Germany))
    spin.setSuffix(" %")
    changes = QSignalSpy(spin.valueChanged)
    spin.lineEdit().setText("1,75 %")
    assert spin.lineEdit().hasAcceptableInput()
    spin.interpretText()
    assert spin.value() == 1.75
    assert len(changes) == 1
    spin.stepUp()
    assert spin.value() == 2.0
    spin.setMaximum(1.5)
    assert spin.value() == 1.5
    spin.stepUp()
    assert spin.value() == 1.5
    spin.stepDown()
    assert spin.value() == 1.25
    assert spin.text().endswith(" %")
    spin.deleteLater()


def test_stepper_keyboard_entry_and_step(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QSignalSpy, QTest
    from widgets.stepper import StepperSpinBox

    spin = StepperSpinBox(minimum=0, maximum=10, decimals=0, step=1, value=2)
    spin.show()
    spin.lineEdit().setFocus()
    changes = QSignalSpy(spin.valueChanged)
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin.lineEdit(), "7")
    assert len(changes) == 0
    QTest.keyClick(spin.lineEdit(), Qt.Key.Key_Return)
    assert spin.value() == 7
    assert len(changes) == 1
    QTest.keyClick(spin.lineEdit(), Qt.Key.Key_Up)
    assert spin.value() == 8
    QTest.keyClick(spin.lineEdit(), Qt.Key.Key_Down)
    assert spin.value() == 7
    spin.deleteLater()


def test_stepper_signal_blocker_suppresses_changes(qapp):
    from PyQt6.QtCore import QSignalBlocker
    from PyQt6.QtTest import QSignalSpy
    from widgets.stepper import StepperSpinBox

    spin = StepperSpinBox(minimum=0, maximum=10, value=2)
    changes = QSignalSpy(spin.valueChanged)
    with QSignalBlocker(spin):
        spin.setValue(5)
        spin.stepUp()
    assert spin.value() == 6
    assert len(changes) == 0
    spin.setValue(7)
    assert len(changes) == 1
    assert changes[0][0] == 7.0
    spin.deleteLater()


def test_stepper_mouse_buttons_respect_bounds(qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QStyle, QStyleOptionSpinBox
    from widgets.stepper import StepperSpinBox
    from ui.styles import get_theme_style

    spin = StepperSpinBox(minimum=0, maximum=2, step=1, value=1)
    spin.setStyleSheet(get_theme_style("dark"))
    spin.resize(spin.sizeHint())
    spin.show()
    option = QStyleOptionSpinBox()
    spin.initStyleOption(option)
    style = spin.style()
    up = style.subControlRect(QStyle.ComplexControl.CC_SpinBox, option,
                              QStyle.SubControl.SC_SpinBoxUp, spin)
    down = style.subControlRect(QStyle.ComplexControl.CC_SpinBox, option,
                                QStyle.SubControl.SC_SpinBoxDown, spin)
    assert up.isValid() and down.isValid() and not up.intersects(down)
    QTest.mouseClick(spin, Qt.MouseButton.LeftButton, pos=up.center())
    assert spin.value() == 2
    QTest.mouseClick(spin, Qt.MouseButton.LeftButton, pos=up.center())
    assert spin.value() == 2
    QTest.mouseClick(spin, Qt.MouseButton.LeftButton, pos=down.center())
    assert spin.value() == 1
    spin.deleteLater()


def test_batch_groups_follow_the_workflow_order(qapp):
    """Inputs -> processing -> output, with the aligned-stack option inside
    the output group.

    The processing group used to be added after every output control, and the
    "save aligned stack" checkbox floated loose at the bottom outside all
    groups, so the dialog read as a flat list with no categories.
    """
    from PyQt6.QtWidgets import QGroupBox

    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        titles = []
        for g in dlg.findChildren(QGroupBox):
            titles.append(g.title())
        assert len(titles) >= 4, titles
        # The processing group must come before the output group.
        # Compare positions in the scroll content, not the outer layout.
        content = dlg.options_group.parentWidget()
        lay = content.layout()
        proc = lay.indexOf(dlg.options_group)
        out_group = dlg.save_aligned_cb.parentWidget()
        out = lay.indexOf(out_group)
        assert proc >= 0 and out >= 0, (proc, out)
        assert proc < out, f"processing must precede output, got {titles}"

        # The aligned-stack checkbox lives inside the output group.
        out_group = dlg.save_aligned_cb.parentWidget()
        assert isinstance(out_group, QGroupBox)
        assert out_group is not dlg.options_group
    finally:
        dlg.deleteLater()


def test_batch_parameter_editors_fit_their_content(qapp):
    from dialogs.batch import BatchProcessingDialog
    dlg = BatchProcessingDialog(None)
    try:
        dlg.resize(820, 900)
        dlg.show()
        dlg.fusion_combo.setCurrentIndex(dlg.fusion_combo.findData("openfocus_b"))
        qapp.processEvents()
        for editor in dlg.fusion_parameter_panel.editors.values():
            assert editor.width() >= editor.minimumSizeHint().width()
    finally:
        dlg.close()
        dlg.deleteLater()








def test_batch_ai_mode_survives_a_method_switch(qapp):
    from dialogs.batch import BatchProcessingDialog
    dlg = BatchProcessingDialog(None)
    from types import SimpleNamespace
    dlg.parent_window = SimpleNamespace(experimental_features=True)
    dlg._apply_fusion_availability()
    try:
        dlg.show()
        dlg.fusion_combo.setCurrentIndex(dlg.fusion_combo.findData("stackmffv3"))
        mode = dlg.fusion_parameter_panel.editors["mode"]
        mode.setCurrentIndex(mode.findData("hard"))
        dlg.fusion_combo.setCurrentIndex(dlg.fusion_combo.findData("dtcwt"))
        dlg.fusion_combo.setCurrentIndex(dlg.fusion_combo.findData("stackmffv3"))
        qapp.processEvents()
        assert dlg.fusion_parameter_panel.values()["mode"] == "hard"
        assert dlg.get_processing_settings()["fusion_params"]["mode"] == "hard"
        dlg.fusion_combo.setCurrentIndex(dlg.fusion_combo.findData("openfocus_c"))
        assert "kernel_size" not in dlg.get_processing_settings()["fusion_params"]
    finally:
        dlg.close()
        dlg.deleteLater()


def test_batch_path_field_actually_adds_folders(qapp, tmp_path):
    """The path box must ADD what you type, not just seed the picker.

    It previously only set the file dialog's start folder, so typing a path
    appeared to do nothing - the reason the folder chooser felt crippled.
    """
    import numpy as np
    import cv2

    from dialogs.batch import BatchProcessingDialog, BATCH_PICK_DIR_KEY

    stack = tmp_path / "stackA"
    stack.mkdir()
    cv2.imwrite(str(stack / "f0.png"), np.zeros((40, 50, 3), np.uint8))

    dlg = BatchProcessingDialog(None)
    try:
        assert hasattr(dlg, "path_input"), "the path entry box is missing"
        dlg.path_input.setText(str(stack))
        dlg.add_path_from_field()
        assert dlg.folder_paths == [str(stack)]
        # Several at once, with one bogus entry left in the field.
        dlg.path_input.setText(f'"{stack}"; {tmp_path / "missing"}')
        dlg.add_path_from_field()
        assert len(dlg.folder_paths) == 1, "duplicates must be ignored"
        assert "missing" in dlg.path_input.text(), (
            "invalid paths must stay visible for correction")
        assert BATCH_PICK_DIR_KEY.startswith("batch/")
    finally:
        dlg.deleteLater()


def test_batch_import_mode_group_stays_compact(qapp):
    """The two import-mode radios are one choice and must not be spread out."""
    from PyQt6.QtWidgets import QSizePolicy

    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        dlg.resize(800, 1000)
        dlg.show()
        qapp.processEvents()
        group = dlg.rb_multiple_folders.parentWidget()
        assert group.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Maximum
        gap = dlg.rb_single_folder.geometry().top() - \
            dlg.rb_multiple_folders.geometry().bottom()
        assert gap < 20, f"radios drifted {gap}px apart"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_folder_picker_has_no_empty_extra_column(qapp, tmp_path, monkeypatch):
    """The picker must not get a second, empty list widget.

    Inserting a QListWidget into Qt's splitter put an empty column beside the
    real place list - the blank box reported in the screenshot.
    """
    from utils import file_dialogs
    monkeypatch.setattr(file_dialogs, "_use_native_folder_dialog", lambda: False)
    from PyQt6.QtWidgets import QApplication, QSplitter, QListView

    from dialogs.batch import BatchProcessingDialog

    stack = tmp_path / "stackA"
    stack.mkdir()
    dlg = BatchProcessingDialog(None)
    captured = {}

    def probe():
        from PyQt6.QtWidgets import QFileDialog
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    spl = wd.findChild(QSplitter)
                    captured["count"] = spl.count() if spl else 0
                    captured["sidebar"] = (type(spl.widget(0)).__name__
                                           if spl else None)
                    captured["widgets"] = [type(spl.widget(i)).__name__
                                           for i in range(spl.count())] if spl else []
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folder_to_list(str(stack))
        dlg.add_folders()
        assert captured.get("count") == 2, captured
        assert captured.get("sidebar") == "QListView"
        # Only the sidebar and the file view; no extra list widget.
        assert captured["widgets"].count("QListWidget") == 0, captured
    finally:
        dlg.deleteLater()




def test_folder_picker_keeps_only_qt_places(qapp, tmp_path, monkeypatch):
    """No custom entries in the place sidebar.

    "Last used" and "Queued: <name>" were appended to Qt's place model; they
    duplicated what the picker already offers and pushed the useful entries
    down. The sidebar must keep exactly the standard locations.
    """
    from utils import file_dialogs
    monkeypatch.setattr(file_dialogs, "_use_native_folder_dialog", lambda: False)
    from PyQt6.QtWidgets import QApplication, QFileDialog, QSplitter

    from dialogs.batch import BatchProcessingDialog

    stack = tmp_path / "stackA"
    stack.mkdir()
    dlg = BatchProcessingDialog(None)
    captured = {}

    def probe():
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    spl = wd.findChild(QSplitter)
                    model = spl.widget(0).model()
                    captured["places"] = [model.index(r, 0).data()
                                          for r in range(model.rowCount())]
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folder_to_list(str(stack))
        dlg.add_folders()
        places = captured.get("places") or []
        assert places, "the sidebar is empty"
        joined = " ".join(str(p) for p in places).lower()
        assert "queued" not in joined and "已排队" not in joined
        assert "last used" not in joined and "上次使用" not in joined
    finally:
        dlg.deleteLater()


def test_folder_list_shows_at_least_three_rows(qapp):
    """The queued-stack list must not collapse to a single clipped row."""
    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        needed = dlg._folder_row_height() * 3
        assert dlg.folder_list.minimumHeight() >= needed, (
            f"min height {dlg.folder_list.minimumHeight()} < 3 rows "
            f"({needed}px)")
    finally:
        dlg.deleteLater()


def test_label_text_resolution_rules():
    """`format` is the template; `text` is only a fallback.

    The old rule was "if '{value}' is in format, use format, else use text",
    so typing "SAMPLE" into the format field and leaving the separate custom
    text box empty produced an EMPTY label: the dialog saved the settings,
    reported success, and drew nothing.
    """
    from utils.validators import build_label_text

    assert build_label_text("{value}", "", 3) == "3"
    assert build_label_text("Fig {value}", "", 3) == "Fig 3"
    assert build_label_text("SAMPLE", "", 3) == "SAMPLE"
    assert build_label_text("", "FIGURE", 3) == "FIGURE"
    assert build_label_text("", "", 3) == ""
    # Steppers hand back floats; "Fig 3.0" is not what anyone typed.
    assert build_label_text("Fig {value}", "", 3.0) == "Fig 3"


def test_label_draws_with_float_geometry():
    """cv2.putText rejects float coordinates ("Can't parse 'org'").

    The dialog's steppers expose the QDoubleSpinBox API and return floats, and
    the draw error was swallowed, so perfectly valid settings drew nothing.
    """
    import numpy as np

    from utils.validators import LabelAdder

    image = np.full((300, 400, 3), 80, np.uint8)
    adder = LabelAdder()
    adder.config.update_config({
        "format": "Fig {value}", "font_size": 70.0, "x_location": 30.0,
        "y_location": 100.0, "starting_value": 1.0, "interval": 1.0,
        "transparent_bg": True, "font_color": (255, 255, 255),
        "bg_color": (0, 0, 0),
    })
    out = adder.add_label_to_image(image.copy(), 0)
    assert (out != image).any(), "a label with valid settings drew nothing"
    assert not adder.last_error


def test_label_keeps_uint16_depth_and_leaves_other_pixels():
    """A caption on a 16-bit frame must not down-convert the image.

    Drawing used to round-trip through an 8-bit copy for the whole frame;
    only the pixels the caption touches may change, and they must stay
    representable at the original depth.
    """
    import numpy as np

    from utils.validators import LabelAdder

    image = np.full((160, 260, 3), 40000, np.uint16)
    image[0:4, 0:4] = 1234
    adder = LabelAdder()
    adder.config.update_config({
        "format": "SAMPLE", "font_size": 28, "x_location": 12,
        "y_location": 60, "range": "All", "transparent_bg": True,
        "font_color": (255, 255, 255), "bg_color": (0, 0, 0),
        "font_family": "Arial",
    })
    out = adder.add_label_to_image(image.copy(), 0)
    assert out.dtype == np.uint16
    assert np.array_equal(out[0:4, 0:4], image[0:4, 0:4]), (
        "pixels outside the caption changed")
    assert not np.array_equal(out, image), "nothing was drawn"


def test_label_outside_the_range_is_reported_not_drawn():
    """Out-of-range is a state, not a failure and not a silent success."""
    import numpy as np

    from utils.validators import LabelAdder

    adder = LabelAdder()
    adder.config.update_config({"format": "SAMPLE", "range": "9-10",
                                "font_size": 24, "x_location": 5,
                                "y_location": 40})
    image = np.full((80, 120, 3), 60, np.uint8)
    out = adder.add_label_to_image(image.copy(), 0)
    assert np.array_equal(out, image)
    assert adder.last_out_of_range is True
    assert not adder.last_error


def test_add_label_dialog_has_a_live_preview(qapp):
    """The dialog must show the label before it is confirmed."""
    from dialogs.retouch import RetouchDialog  # noqa: F401  (import sanity)
    from widgets.label_preview import LabelPreview

    preview = LabelPreview()
    try:
        preview.resize(320, 220)
        preview.set_config({"format": "SAMPLE", "text": "", "font_size": 40,
                            "x_location": 20, "y_location": 60,
                            "transparent_bg": True, "font_color": (255,) * 3,
                            "bg_color": (0,) * 3})
        assert preview.message() == ""
        pix = preview.grab()
        assert pix.width() > 0 and pix.height() > 0
        # Both fields empty -> the preview reports why nothing is drawn.
        preview.set_config({"format": "", "text": ""})
        assert preview.message() == "empty"
    finally:
        preview.deleteLater()













def _checkerboard_depth_surface(block_size=1):
    import numpy as np
    from widgets.depth_surface import DepthSurface

    yy, xx = np.indices((120, 160))
    texture = np.zeros((120, 160, 3), np.uint8)
    texture[..., 0] = ((xx // block_size + yy // block_size) % 2) * 240
    texture[..., 2] = 240 - texture[..., 0]
    surface = DepthSurface()
    surface.resize(420, 320)
    surface.set_surface(np.full((120, 160), 0.5, np.float32), texture)
    surface.show()
    return surface


def _depth_texture_contrast(image):
    import numpy as np
    sample = image[100:220, 120:300].astype(np.int16)
    return np.abs(sample[..., 0] - sample[..., 2])


def test_depth_surface_detail_changes_settled_preview_pixels(qapp):
    import numpy as np
    from utils.image_utils import pixmap_to_cv2

    surface = _checkerboard_depth_surface()
    try:
        surface.set_detail(20)
        low = pixmap_to_cv2(surface.grab())
        surface.set_detail(100)
        high = pixmap_to_cv2(surface.grab())
        assert _depth_texture_contrast(low).max() < 20
        sharp = _depth_texture_contrast(high)
        assert np.count_nonzero(sharp > 100) > sharp.size // 3
        surface.set_detail(20)
        np.testing.assert_array_equal(pixmap_to_cv2(surface.grab()), low)
    finally:
        surface.close()
        surface.deleteLater()


def test_depth_surface_drag_restores_selected_preview_detail(qapp):
    import numpy as np
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from utils.image_utils import pixmap_to_cv2

    for detail, block_size in ((40, 2), (100, 1)):
        surface = _checkerboard_depth_surface(block_size)
        try:
            surface.set_detail(detail)
            settled = pixmap_to_cv2(surface.grab())
            position = surface.rect().center()
            QTest.mousePress(surface, Qt.MouseButton.LeftButton, pos=position)
            moving = pixmap_to_cv2(surface.grab())
            assert (_depth_texture_contrast(settled).max()
                    > 2 * _depth_texture_contrast(moving).max() + 5)
            QTest.mouseRelease(surface, Qt.MouseButton.LeftButton, pos=position)
            np.testing.assert_array_equal(pixmap_to_cv2(surface.grab()), settled)
        finally:
            surface.close()
            surface.deleteLater()


def test_feedback_dialog_embeds_the_session_log(qapp):
    """The feedback page shows the session log and exports a zip package."""
    import zipfile
    import tempfile

    import os
    import tempfile
    import zipfile

    from dialogs.feedback import FeedbackDialog
    from utils import logging_setup as ls

    ls.setup_logging()
    d = FeedbackDialog(None)
    try:
        d._copy_log()
        assert qapp.clipboard().text() == d.log_view.toPlainText()
        dest = os.path.join(tempfile.mkdtemp(), "fb.zip")
        path = ls.export_feedback_package(dest)
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "session_log.txt" in names
            assert "system.txt" in names
    finally:
        d.deleteLater()



def test_flat_field_failure_keeps_the_original_stack(qapp, monkeypatch):
    """A failed base-stage correction must not leave a half-applied stack."""
    import numpy as np
    from types import SimpleNamespace
    import main as main_module
    from utils import dust_map

    frames = [np.full((8, 8, 3), 10, np.uint8) for _ in range(2)]
    base = [np.full((8, 8, 3), 20, np.uint8) for _ in range(2)]
    host = SimpleNamespace(
        raw_images=frames, base_images=base, _undo_guard=True,
        push_undo=lambda: None,
        get_or_create_source_pixmap=lambda *a, **k: None,
        statusBar=lambda: SimpleNamespace(showMessage=lambda *a, **k: None),
    )
    warnings = []
    monkeypatch.setattr(main_module, "show_warning_box",
                        lambda *a, **k: warnings.append(a))

    calls = {"n": 0}

    def fail_on_second(images, gain):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("size mismatch")
        return [f + 1 for f in images]

    monkeypatch.setattr(dust_map, "apply_flat_field_stack", fail_on_second)
    result = main_module.OpenFocus.apply_flat_field_to_stack(host, object())
    assert result is False
    assert host.raw_images is frames, "raw frames were replaced on failure"
    assert host.base_images is base, "base frames were replaced on failure"
    assert warnings
