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
