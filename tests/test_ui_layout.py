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


def test_spin_box_is_not_narrower_than_it_needs():
    """A fixed width below minimumSizeHint clips the value."""
    from pathlib import Path
    src = Path("ui/right_panel.py").read_text(encoding="utf-8")
    block = src[src.index("spin_depth_smooth = QDoubleSpinBox()"):]
    block = block[:block.index("ds_layout.addWidget")]
    assert "setFixedWidth" not in block, (
        "a fixed width clips the spin box value; use setMinimumWidth")
    assert "setMinimumWidth" in block


def test_each_option_checkbox_gets_its_own_row():
    """Two per row elides long translations; one per row always fits."""
    from pathlib import Path
    src = Path("ui/right_panel.py").read_text(encoding="utf-8")
    block = src[src.index("def _option(key, checked):"):]
    block = block[:block.index("chk_base_correct = _option")]
    assert "config_layout.addWidget(box)" in block, (
        "options must be stacked in the column, not paired in an HBox")
    assert "QHBoxLayout" not in block


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


@pytest.mark.parametrize("lang", ["en", "zh", "ja", "es"])
def test_help_bodies_are_translated(lang):
    """The registration help used to show English text in every language."""
    from locales import trans
    for key in ("help_render_body", "help_registration_body",
                "help_tile_body"):
        body = trans.translations[lang].get(key, "")
        assert body, f"{lang}: {key} missing"
        assert "<h3>" in body
    if lang != "en":
        assert "Registration Methods" not in trans.translations[lang][
            "help_registration_body"], f"{lang}: help body is still English"


def test_checked_radio_does_not_grow_its_indicator():
    """A 5 px border on :checked grew the box from 18 px to 26 px, so the
    indicator visibly jumped when the AI mode was toggled."""
    from PyQt6.QtWidgets import QApplication, QRadioButton
    from ui.styles import build_stylesheet
    app = QApplication.instance() or QApplication([])
    previous = app.styleSheet()
    try:
        for theme in ("light", "dark"):
            app.setStyleSheet(build_stylesheet(theme))
            # Same label: only the checked state may differ.
            soft, hard = QRadioButton("same"), QRadioButton("same")
            soft.setChecked(True)
            assert soft.height() == hard.height(), theme
            assert soft.sizeHint().height() == hard.sizeHint().height(), theme
            assert soft.sizeHint().width() == hard.sizeHint().width(), theme
    finally:
        app.setStyleSheet(previous)


def test_unchecked_radio_indicator_is_a_ring():
    """Any background colour on the unchecked radio is painted as a filled
    disc, which read as a solid dot on the dark theme."""
    from pathlib import Path
    src = Path("ui/styles.py").read_text(encoding="utf-8")
    block = src[src.index("QRadioButton::indicator {{"):
                src.index("QCheckBox:disabled")]
    assert "background: transparent" in block
    assert "border: 5px" not in block


def test_ai_mode_radios_exist_and_default_to_soft():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from ui.right_panel import create_right_panel
    panel = create_right_panel()
    try:
        assert panel.radio_ai_soft.isChecked()
        assert not panel.radio_ai_hard.isChecked()
        assert not panel.ai_mode_widget.isVisible()
    finally:
        panel.widget.deleteLater()


def test_help_chips_follow_the_theme():
    """The "?" chips are styled inline at creation. Switching to light used to
    leave them dark: apply_theme refreshed them BEFORE recording the new
    CURRENT_THEME, so help_button_style() re-read the old one."""
    from pathlib import Path
    src = Path("main.py").read_text(encoding="utf-8")
    body = src[src.index("def apply_theme(self, theme: str) -> None:"):]
    body = body[:body.index("def _enable_dark_title_bar")]
    assert body.index("set_current_theme(theme)") < body.index("help_button_style(theme)"), (
        "the theme must be recorded before inline styles are recomputed")


def test_help_button_style_uses_the_active_theme():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
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


def test_no_input_widget_is_narrower_than_its_content():
    """A fixed width below minimumSizeHint silently clips the value.

    Two shipped defects of this class: the right panel's blend-smoothing spin
    ("2.0" → "2.("), the load-scale spin ("10%" → "10(") and the dust-map
    sigma ("4.0" → "4."). This walks every dialog's spin boxes so a new fixed
    width cannot reintroduce it.
    """
    from PyQt6.QtWidgets import (QApplication, QDoubleSpinBox, QSpinBox,
                                 QWidget)
    app = QApplication.instance() or QApplication([])
    from ui.styles import build_stylesheet
    app.setStyleSheet(build_stylesheet("dark"))

    from dialogs.settings import DownsampleDialog, TileSettingsDialog
    from dialogs.dustmap import DustMapDialog

    parent = QWidget()
    dialogs = [DownsampleDialog(parent, initial_scale=0.05, max_scale=1.0),
               TileSettingsDialog(parent),
               DustMapDialog(parent)]
    try:
        for dlg in dialogs:
            dlg.show()
            app.processEvents()
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


def test_fusion_dropdown_order_and_default():
    """Alignment-only stays first; AI is second but is the DEFAULT, and its
    label names the algorithm so the method is identifiable outside the app."""
    from ui.right_panel import FUSION_ITEMS, FUSION_DEFAULT
    codes = [c for c, _k, _h in FUSION_ITEMS]
    assert codes[0] == "none"
    assert codes[1] == "stackmffv3"
    assert codes[2:5] == ["openfocus_a", "openfocus_b", "openfocus_c"]
    assert codes[5:] == ["guided_filter", "dct", "dtcwt", "gfgfgf"]
    assert FUSION_DEFAULT == "stackmffv3"


def test_ai_label_names_the_algorithm_in_every_language(qapp):
    """'AI' alone tells the user nothing; the paper is StackMFF V3."""
    from locales import TranslationManager
    tables = TranslationManager().translations
    for lang in ("en", "zh", "ja", "es"):
        assert "StackMFF-V3" in tables[lang]["radio_stackmff"], lang


def test_menus_have_an_opaque_background():
    """A menu must not inherit `QWidget { background: transparent; }`.

    The base rule made the popup paint its text over whatever was behind it:
    on the dark theme a right-click menu showed the desktop through it, which
    read as a translucent panel with unreadable labels. Both the frame and
    every item need an explicit fill.
    """
    from ui.styles import build_stylesheet

    for theme in ("dark", "light"):
        qss = build_stylesheet(theme)
        menu = qss[qss.index("QMenu {"):qss.index("QMenu::item {")]
        item = qss[qss.index("QMenu::item {"):qss.index("QMenu::item:selected")]
        assert "background-color:" in menu, theme
        assert "background-color:" in item, theme
        assert "background: transparent" not in item, theme
