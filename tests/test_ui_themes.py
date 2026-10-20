"""Dialogs must follow the window theme.

Every dialog used to hard-code the dark palette, so light mode showed dark
slabs; the contact dialog's GitHub link was white on white; the "?" buttons
kept a dark chip; the dust-map preview cards were black blocks.
"""
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")


def _source(name):
    return Path(name).read_text(encoding="utf-8")


def test_dialogs_do_not_carry_their_own_palette():
    """Dialogs must be styled by the generated application sheet only.

    They used to ship their own dark QSS (later gated by apply_dialog_theme);
    both are gone now, so a dialog that reintroduces a hard-coded palette
    would silently opt out of the design system.
    """
    for name in ("dialogs/settings.py", "dialogs/batch.py", "dialogs/roi.py",
                 "dialogs/dustmap.py", "dialogs/depthmap.py", "dialogs/help.py"):
        src = _source(name)
        assert "apply_dialog_theme" not in src, (
            f"{name} still applies a per-dialog palette")
        assert "background-color: #2b2b2b" not in src, (
            f"{name} hard-codes the old dark background")


def test_contact_link_colour_follows_the_theme():
    src = _source("dialogs/about.py")
    assert '#0969da' in src and '#6cb6ff' in src, (
        "the contact link must not be a fixed colour (it was white on white)")
    assert "a {{ color: #ffffff; }}" not in src


def test_help_button_style_is_theme_aware():
    from ui.styles import help_button_style
    from utils.design_tokens import palette
    light = help_button_style("light")
    dark = help_button_style("dark")
    assert light != dark
    # both pull their colours from the palette, not from literals
    assert palette("light")["surfaceHover"] in light
    assert palette("dark")["surfaceHover"] in dark
    assert "QPushButton#helpButton" in light
    assert "QPushButton#helpButton" in dark


def test_dustmap_preview_follows_the_theme():
    src = _source("dialogs/dustmap.py")
    assert "palette(CURRENT_THEME)" in src, (
        "the preview card must take its colours from the palette")
    assert "surfaceSunken" in src


def test_spin_boxes_reserve_room_for_their_buttons():
    """The up/down column is 20 px wide; the text side must not run under it."""
    from ui.styles import get_app_light_style
    qss = get_app_light_style()
    spin = qss[qss.index("QSpinBox, QDoubleSpinBox {"):]
    spin = spin[:spin.index("}")]
    assert "22px" in spin, "spin boxes must reserve the arrow column"


def test_batch_dialog_has_an_empty_state():
    src = _source("dialogs/batch.py")
    assert "_update_folder_placeholder" in src
    assert "batch_no_folders" in src
