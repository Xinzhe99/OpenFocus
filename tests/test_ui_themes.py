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


def test_dialogs_do_not_hardcode_the_dark_palette_unconditionally():
    """Each dark QSS must sit behind apply_dialog_theme (or an explicit
    theme check), never applied unconditionally."""
    for name in ("dialogs/settings.py", "dialogs/batch.py", "dialogs/roi.py"):
        src = _source(name)
        assert "apply_dialog_theme" in src, f"{name} ignores the window theme"


def test_contact_link_colour_follows_the_theme():
    src = _source("dialogs/about.py")
    assert '#0969da' in src and '#6cb6ff' in src, (
        "the contact link must not be a fixed colour (it was white on white)")
    assert "a {{ color: #ffffff; }}" not in src


def test_help_button_style_is_theme_aware():
    from ui.styles import help_button_style
    light = help_button_style("light")
    dark = help_button_style("dark")
    assert light != dark
    assert "#eaeef2" in light and "#555" in dark
    # the selector must out-specify a plain `QPushButton` dialog rule
    assert "QPushButton#helpButton" in light
    assert "QPushButton#helpButton" in dark


def test_dustmap_preview_follows_the_theme():
    src = _source("dialogs/dustmap.py")
    assert "#f6f8fa" in src, "light theme needs a light preview card"
    assert "ui_theme" in src


def test_spin_boxes_reserve_room_for_their_buttons():
    """The up/down column is 20 px wide; without right padding the text runs
    underneath it (the depth-map dialog showed '1.00' under the arrows)."""
    from ui.styles import get_app_light_style
    assert "padding-right: 22px" in get_app_light_style()


def test_batch_dialog_has_an_empty_state():
    src = _source("dialogs/batch.py")
    assert "_update_folder_placeholder" in src
    assert "batch_no_folders" in src
