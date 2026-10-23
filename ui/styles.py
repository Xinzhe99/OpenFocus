"""Theme stylesheets, generated from the design tokens.

Before this module the stylesheet was 648 lines of hand-written QSS with the
colours pasted in, plus a `_to_light()` regex that rewrote the dark sheet by
string substitution. Both are gone: `build_stylesheet(theme)` renders every
rule from `utils.design_tokens`, so light and dark are the same design with
different values, and a new widget cannot drift off-palette.

Public API (kept compatible with the old module so call sites do not churn):
    PRIMARY_BLUE, CURRENT_THEME, CURRENT_MESSAGE_BOX_STYLE
    get_theme_style(theme), get_app_light_style(), set_current_theme(theme)
    MESSAGE_BOX_STYLE, PROGRESS_DIALOG_STYLE, WHITE_COMBOBOX_STYLE
    SOURCE_LIST_STYLE, OUTPUT_LIST_STYLE, HELP_BUTTON_STYLE(+_DARK/_LIGHT)
    HOVER_HIGHLIGHT_BUTTON_STYLE, DIALOG_CONTROL_DARK_QSS, arrow_qss()
    help_button_style(theme)
"""
from __future__ import annotations

from typing import Dict

from utils.design_tokens import FONT, RADIUS, SPACE, WEIGHT, palette


def _asset_url(name: str) -> str:
    """QSS url() needs forward slashes and an absolute path."""
    try:
        from utils import resource_path
        return resource_path("assets", name).replace("\\", "/")
    except Exception:
        return ""


_CHECK_SVG = _asset_url("check_white.svg")
_RADIO_DOT_LIGHT = _asset_url("radio_dot_blue.svg")
_RADIO_DOT_DARK = _asset_url("radio_dot_dark.svg")
_ARROW_DOWN_LIGHT = _asset_url("arrow_down_light.svg")
_ARROW_UP_LIGHT = _asset_url("arrow_up_light.svg")
_ARROW_DOWN_DARK = _asset_url("arrow_down_dark.svg")
_ARROW_UP_DARK = _asset_url("arrow_up_dark.svg")
# Right-pointing glyph for QMenu submenu indicators. Qt draws its own
# triangle when the stylesheet leaves QMenu::right-arrow alone; that triangle
# sits flush against the popup edge and ignores the theme, which is what made
# the submenu column look wrong next to the padded items.
_ARROW_RIGHT_LIGHT = _asset_url("arrow_right_light.svg")
_ARROW_RIGHT_DARK = _asset_url("arrow_right_dark.svg")
# Spin-box steppers draw a - and a + instead of up/down chevrons, so the
# control reads as "increase this number" rather than "sort".
_MINUS_LIGHT = _asset_url("minus_light.svg")
_MINUS_DARK = _asset_url("minus_dark.svg")
_PLUS_LIGHT = _asset_url("plus_light.svg")
_PLUS_DARK = _asset_url("plus_dark.svg")

_UI_FONT = '"Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif'
_MONO_FONT = 'Consolas, "Segoe UI", monospace'


def arrow_qss(theme: str = "dark", button_bg: str = None,
              button_hover: str = None, border: str = None) -> str:
    """Arrows for QComboBox / QSpinBox.

    `theme` selects the glyph colour: 'dark' (dark widgets, light arrows) or
    'light'. Colours default to the current palette.
    """
    t = palette(theme)
    down = _ARROW_DOWN_LIGHT if theme == "dark" else _ARROW_DOWN_DARK
    up = _ARROW_UP_LIGHT if theme == "dark" else _ARROW_UP_DARK
    minus = _MINUS_LIGHT if theme == "dark" else _MINUS_DARK
    plus = _PLUS_LIGHT if theme == "dark" else _PLUS_DARK
    bg = button_bg or t["surfaceHover"]
    hover = button_hover or t["surfaceActive"]
    r = RADIUS
    s = SPACE
    f = FONT
    bd = border or t["border"]
    return f"""
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 22px; border: none; background: transparent;
}}
QComboBox::down-arrow {{ image: url("{down}"); width: 10px; height: 10px; }}
/* Spin box steppers keep Qt's native geometry (two sub-controls on the
   trailing edge; Qt 6 ignores subcontrol-position for them, so they cannot be
   moved to opposite sides from a stylesheet). The glyphs are still swapped for
   a + and a -, which is what makes the control read as "adjust this number"
   rather than "sort".

   Controls that need the + and - on OPPOSITE sides use widgets/stepper.py
   instead — it builds the buttons explicitly. */
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{plus}"); width: 9px; height: 9px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{minus}"); width: 9px; height: 9px;
}}

/* StepperSpinBox (widgets/stepper.py): - on the left, + on the right.
   QToolButton needs a base rule: a bare id selector alone did not beat the
   widget's native metrics here, and the buttons kept Qt's 20 px height while
   the adjacent line edit was 34 px. */
QToolButton {{
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-radius: {r['xs']}px;
    padding: 2px 4px;
}}
QToolButton:hover {{ background-color: {t['surfaceHover']}; }}
QToolButton:pressed {{ background-color: {t['surfaceActive']}; }}
QToolButton#stepperMinusButton, QToolButton#stepperPlusButton {{
    background-color: {t['surfaceHover']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    min-width: 22px; max-width: 22px;
    /* Height has to reach the 34 px of the adjacent QLineEdit; making it
       implicit left the buttons 14 px short and broke the stepper's outline. */
    min-height: 32px; max-height: 32px;
    font-size: {f['body'] + 2}pt;
    font-weight: 600;
    padding: 0;
}}
QToolButton#stepperMinusButton {{
    border-top-left-radius: {r['sm']}px;
    border-bottom-left-radius: {r['sm']}px;
    border-right: none;
}}
QToolButton#stepperPlusButton {{
    border-top-right-radius: {r['sm']}px;
    border-bottom-right-radius: {r['sm']}px;
    border-left: none;
}}
QToolButton#stepperMinusButton:hover, QToolButton#stepperPlusButton:hover {{
    background-color: {hover};
}}
QToolButton#stepperMinusButton:pressed, QToolButton#stepperPlusButton:pressed {{
    background-color: {t['surfaceActive']};
}}
QToolButton#stepperMinusButton:disabled, QToolButton#stepperPlusButton:disabled {{
    background-color: {t['controlDisabled']};
    color: {t['textFaint']};
}}
#stepperSpinBoxEdit {{
    background-color: {t['controlBg']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    /* Matches QLineEdit / QComboBox, so a stepper lines up with the plain
       inputs in the same dialog. */
    padding: {s['xs'] + 2}px {s['sm']}px;
    min-height: 20px;
    selection-background-color: {t['accent']};
    selection-color: {t['textOnAccent']};
}}
#stepperSpinBoxEdit:focus {{ border-color: {t['borderFocus']}; }}
"""


def _shadow_radius() -> Dict[str, int]:
    return RADIUS


def build_stylesheet(theme: str = "dark") -> str:
    """Render the full application stylesheet for `theme`."""
    t = palette(theme)
    r = _shadow_radius()
    s = SPACE
    f = FONT
    check = _CHECK_SVG
    dot = _RADIO_DOT_DARK if theme == "dark" else _RADIO_DOT_LIGHT
    arrow_right = _ARROW_RIGHT_LIGHT if theme == "dark" else _ARROW_RIGHT_DARK


    return f"""
/* ===================== base ===================== */
QWidget {{
    color: {t['text']};
    font-family: {_UI_FONT};
    font-size: {f['body']}pt;
    background: transparent;
}}
QMainWindow, QDialog {{ background-color: {t['bg']}; }}
QMainWindow > QWidget {{ background-color: {t['bg']}; }}

QToolTip {{
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['border']};
    border-radius: {r['sm']}px;
    padding: {s['xs'] + 2}px {s['sm']}px;
}}

QSplitter::handle {{ background-color: {t['border']}; }}
QSplitter::handle:hover {{ background-color: {t['borderStrong']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

/* ===================== menu bar / menus ===================== */
QMenuBar {{
    background-color: {t['surface']};
    color: {t['text']};
    border-bottom: 1px solid {t['border']};
    padding: 2px {s['sm']}px;
}}
QMenuBar::item {{
    background: transparent;
    padding: {s['xs'] + 2}px {s['md']}px;
    border-radius: {r['sm']}px;
}}
QMenuBar::item:selected {{ background-color: {t['surfaceHover']}; }}
QMenuBar::item:pressed {{ background-color: {t['surfaceActive']}; }}

/* Menus need an explicit opaque fill on BOTH the frame and every item.
   The base rule `QWidget {{ background: transparent; }}` applies to QMenu
   too, and QMenu::item inherits it: the popup then painted its text over
   whatever was behind it, which on the dark theme looked like a white
   translucent panel with unreadable labels. */
/* No border-radius: Qt makes a rounded QMenu a translucent window, and
   without a compositor the corner pixels show the desktop as a pale fringe
   around the popup. A square menu paints cleanly on every platform. */
QMenu {{
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['border']};
    padding: {s['xs']}px;
}}
QMenu::item {{
    padding: {s['sm'] - 2}px {s['xl']}px {s['sm'] - 2}px {s['md']}px;
    background-color: {t['surface']};
    /* No border-radius here. An unselected item's background is the same
       colour as the menu, so a rounded corner has nothing to round against:
       Qt still antialiases the corner and the row edge shows a column of
       little arcs along the popup's right side. The radius belongs to the
       highlight only. */
    padding-right: {s['2xl']}px;
}}
QMenu::item:selected {{
    background-color: {t['accentSoft']};
    color: {t['accentSoftText']};
    border-radius: {r['sm']}px;
}}
QMenu::item:disabled {{ color: {t['textFaint']}; }}
QMenu::separator {{ height: 1px; background-color: {t['border']}; margin: {s['xs']}px {s['sm']}px; }}
QMenu::icon {{ padding-left: {s['sm']}px; }}
/* The indicator gets its own column, vertically centred in the row, instead
   of the default flush-right triangle. The image is swapped per theme so the
   glyph keeps contrast on both backgrounds. */
QMenu::right-arrow {{
    image: url("{arrow_right}");
    width: 12px; height: 12px;
    subcontrol-origin: padding;
    subcontrol-position: center right;
    right: {s['sm']}px;
}}

/* ===================== buttons ===================== */
QPushButton {{
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-radius: {r['sm']}px;
    padding: {s['xs'] + 2}px {s['md']}px;
    min-height: 20px;
}}
QPushButton:hover {{ background-color: {t['surfaceHover']}; }}
QPushButton:pressed {{ background-color: {t['surfaceActive']}; }}
QPushButton:disabled {{
    background-color: {t['controlDisabled']};
    color: {t['textFaint']};
    border-color: {t['border']};
}}
QPushButton:focus {{ border-color: {t['borderFocus']}; outline: none; }}

QPushButton#primaryButton {{
    background-color: {t['accent']};
    color: {t['textOnAccent']};
    border: 1px solid transparent;
    font-weight: {WEIGHT['semibold']};
}}
QPushButton#primaryButton:hover {{ background-color: {t['accentHover']}; }}
QPushButton#primaryButton:pressed {{ background-color: {t['accentPressed']}; }}
QPushButton#primaryButton:disabled {{
    background-color: {t['controlDisabled']};
    color: {t['textFaint']};
}}

QPushButton#helpButton {{
    background-color: {t['surfaceHover']};
    color: {t['textMuted']};
    border: 1px solid {t['border']};
    border-radius: {r['pill']}px;
    padding: 0;
    /* 26 px is the smallest square that still centres the glyph on the same
       baseline as a 34 px control, so the "?" chip reads as a companion to
       the dropdown instead of a smaller unrelated button. */
    min-width: 26px; max-width: 26px;
    min-height: 26px; max-height: 26px;
    font-weight: {WEIGHT['semibold']};
    font-size: {f['caption']}pt;
}}
QPushButton#helpButton:hover {{
    background-color: {t['accentSoft']};
    color: {t['accentSoftText']};
    border-color: {t['accentSoftText']};
}}

/* Stepper +- buttons: an explicit surface so the glyph and the glyph's box
   read the same way on both themes. Inheriting the base button left them as
   white chips with a visible frame in light mode and as bare glyphs in dark
   mode, which made the same dialog look like two different designs. */
QPushButton#stepButton {{
    padding: 0;
    min-width: 26px; max-width: 26px;
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-radius: {r['xs']}px;
}}
QPushButton#stepButton:hover {{
    background-color: {t['surfaceHover']};
    border-color: {t['accent']};
}}
QPushButton#stepButton:pressed {{ background-color: {t['surfaceActive']}; }}

/* ===================== text inputs ===================== */
QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser {{
    background-color: {t['controlBg']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-radius: {r['sm']}px;
    /* Matches the combo box exactly, so an input and a dropdown in the same
       row are the same height and their text starts at the same offset. */
    padding: {s['xs'] + 2}px {s['sm']}px;
    /* Same floor as QComboBox. Without it a line edit computed 30 px while a
       dropdown next to it computed 34 px, so form rows looked ragged. */
    min-height: 20px;
    selection-background-color: {t['accent']};
    selection-color: {t['textOnAccent']};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QTextBrowser:focus {{
    border-color: {t['borderFocus']};
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {{
    background-color: {t['controlDisabled']};
    color: {t['textFaint']};
}}
QLineEdit[readOnly="true"] {{ background-color: {t['surfaceSunken']}; }}

/* ===================== combo box ===================== */
/* Inputs share the button's vertical padding so a dropdown and a button in
   the same row come out exactly the same height. */
QComboBox {{
    background-color: {t['controlBg']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-radius: {r['sm']}px;
    padding: {s['xs'] + 2}px {s['sm']}px;
    padding-right: {s['xl']}px;
    min-height: 20px;
}}
QComboBox:hover {{ border-color: {t['borderStrong']}; }}
QComboBox:focus, QComboBox:on {{ border-color: {t['borderFocus']}; }}
QComboBox:disabled {{ background-color: {t['controlDisabled']}; color: {t['textFaint']}; }}
QComboBox QAbstractItemView {{
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['border']};
    border-radius: {r['sm']}px;
    padding: {s['xs']}px;
    selection-background-color: {t['accentSoft']};
    selection-color: {t['accentSoftText']};
    outline: none;
}}

/* ===================== spin boxes ===================== */
QSpinBox, QDoubleSpinBox {{
    background-color: {t['controlBg']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-radius: {r['sm']}px;
    /* The stepper column sits on the trailing edge and is 22 px wide;
       padding must reserve exactly that or the digits run under it. */
    padding: {s['xs'] + 2}px 22px {s['xs'] + 2}px {s['sm']}px;
    min-height: 20px;
    selection-background-color: {t['accent']};
    selection-color: {t['textOnAccent']};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {t['borderFocus']}; }}
QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {t['controlDisabled']}; color: {t['textFaint']};
}}

/* ===================== item views ===================== */
QListView, QTreeView, QTableView, QListWidget, QTreeWidget, QTableWidget {{
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['border']};
    border-radius: {r['sm']}px;
    outline: none;
    alternate-background-color: {t['surfaceSunken']};
    selection-background-color: {t['accentSoft']};
    selection-color: {t['accentSoftText']};
}}
QListView::item, QTreeView::item, QListWidget::item, QTreeWidget::item {{
    padding: {s['xs'] + 2}px {s['sm']}px;
    border: none;
    margin: 0px;
}}
QListView::item:hover, QTreeView::item:hover,
QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {t['surfaceHover']};
}}
QListView::item:selected, QTreeView::item:selected,
QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {t['accentSoft']};
    color: {t['accentSoftText']};
}}
QHeaderView::section {{
    background-color: {t['surfaceSunken']};
    color: {t['textMuted']};
    border: none;
    border-bottom: 1px solid {t['border']};
    padding: {s['xs'] + 2}px {s['sm']}px;
    font-weight: {WEIGHT['medium']};
    font-size: {f['caption']}pt;
}}
QTableCornerButton::section {{ background-color: {t['surfaceSunken']}; border: none; }}

/* ===================== group boxes (cards) ===================== */
QGroupBox {{
    background-color: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: {r['md']}px;
    margin-top: {s['md']}px;
    padding: {s['sm']}px {s['sm']}px {s['sm']}px {s['sm']}px;
    font-weight: {WEIGHT['semibold']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {s['sm']}px;
    padding: 0 {s['xs']}px;
    color: {t['textMuted']};
    background: transparent;
    font-size: {f['caption']}pt;
    font-weight: {WEIGHT['semibold']};
}}

/* ===================== check / radio ===================== */
QCheckBox, QRadioButton {{ spacing: {s['sm']}px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {t['controlBorder']};
}}
/* The unchecked radio must be a hollow ring. Any background colour here is
   painted as a filled disc by Qt 6, which looked like a solid dot on the dark
   theme (controlBg is only a few steps above the panel). The checked state
   fills the same 18 px box, so toggling never resizes the control. */
QCheckBox::indicator {{ background-color: {t['controlBg']}; }}
QRadioButton::indicator {{
    background: transparent;
    border: 1px solid {t['controlBorder']};
    border-radius: 8px;
}}
QCheckBox::indicator {{ border-radius: {r['xs']}px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {t['accent']};
}}
QCheckBox::indicator:checked {{
    background-color: {t['accent']};
    border-color: {t['accent']};
    image: url("{check}");
}}
QRadioButton::indicator:checked {{
    background-color: {t['accent']};
    border-color: {t['accent']};
    border-radius: 8px;
    image: url("{dot}");
}}
QCheckBox:disabled, QRadioButton:disabled {{ color: {t['textFaint']}; }}

/* List rows that carry a check state (the source-frame list, where a row can
   be excluded from the render) need the SAME indicator as QCheckBox.
   Without these rules Qt has no QListWidget::indicator styling to apply, so
   the check box fell back to the QRadioButton rules declared above — which
   have `border-radius: 8px` — and every row showed a filled blue disc instead
   of a ticked square. */
QListWidget::indicator, QListView::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {t['controlBorder']};
    border-radius: {r['xs']}px;
    background-color: {t['controlBg']};
    margin-left: {s['xs']}px;
}}
QListWidget::indicator:hover, QListView::indicator:hover {{
    border-color: {t['accent']};
}}
QListWidget::indicator:checked, QListView::indicator:checked {{
    background-color: {t['accent']};
    border-color: {t['accent']};
    image: url("{check}");
}}

/* ===================== sliders ===================== */
QSlider::groove:horizontal {{
    height: 4px;
    background: {t['surfaceActive']};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {t['accent']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {t['surface']};
    border: 1px solid {t['borderStrong']};
    width: 14px; height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ border-color: {t['accent']}; }}
QSlider::handle:horizontal:pressed {{ background: {t['accentSoft']}; }}
QSlider::groove:horizontal:disabled {{ background: {t['controlDisabled']}; }}
QSlider::sub-page:horizontal:disabled {{ background: {t['borderStrong']}; }}
QSlider::handle:horizontal:disabled {{
    background: {t['controlDisabled']}; border-color: {t['border']};
}}

/* ===================== progress ===================== */
QProgressBar {{
    background-color: {t['surfaceActive']};
    border: none;
    border-radius: {r['xs']}px;
    text-align: center;
    color: {t['text']};
    font-size: {f['caption']}pt;
    min-height: 6px;
}}
QProgressBar::chunk {{ background-color: {t['accent']}; border-radius: {r['xs']}px; }}

/* ===================== tabs ===================== */
QTabWidget::pane {{
    border: 1px solid {t['border']};
    border-radius: {r['md']}px;
    background: {t['surface']};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {t['textMuted']};
    padding: {s['sm'] - 2}px {s['md']}px;
    border: none;
    border-bottom: 2px solid transparent;
    margin-right: 2px;
}}
QTabBar::tab:hover {{ color: {t['text']}; }}
QTabBar::tab:selected {{
    color: {t['text']};
    border-bottom: 2px solid {t['accent']};
    font-weight: {WEIGHT['medium']};
}}

/* ===================== scroll bars ===================== */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {t['borderStrong']};
    border-radius: 5px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:hover {{ background: {t['textFaint']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; border: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ===================== status bar ===================== */
QStatusBar {{
    background-color: {t['surface']};
    color: {t['textMuted']};
    border-top: 1px solid {t['border']};
    font-size: {f['caption']}pt;
}}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {t['textMuted']}; font-size: {f['caption']}pt; }}

/* ===================== misc ===================== */
QLabel {{ background: transparent; }}
QLabel:disabled {{ color: {t['textFaint']}; }}
QWidget:disabled {{ color: {t['textFaint']}; }}

QLabel#sectionLabel {{
    color: {t['textFaint']};
    font-size: {f['caption']}pt;
    font-weight: {WEIGHT['semibold']};
    letter-spacing: 0.6px;
}}
QLabel#metricValue {{
    color: {t['text']};
    font-size: {f['display']}pt;
    font-weight: {WEIGHT['bold']};
}}
QLabel#mutedLabel {{ color: {t['textMuted']}; font-size: {f['caption']}pt; }}

QFrame#card {{
    background-color: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: {r['md']}px;
}}
QFrame#rightPanel {{
    background-color: {t['surface']};
    border-left: 1px solid {t['border']};
}}
QWidget#dialogButtonBar {{
    background-color: {t['surface']};
    border-top: 1px solid {t['border']};
}}
QFrame#statusPanel {{
    background-color: {t['surfaceSunken']};
    border-top: 1px solid {t['border']};
}}
QFrame#statusPanel QLabel {{
    color: {t['textMuted']};
    font-size: {f['body']}pt;
    font-family: {_UI_FONT};
}}
/* Same padding as the base QPushButton on purpose: a different value here made
   secondary buttons a couple of pixels taller than the primary next to them,
   which reads as a broken row rather than a hierarchy. The hierarchy comes
   from the colours, not the height. */
QPushButton#secondaryButton {{
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-radius: {r['sm']}px;
    padding: {s['xs'] + 2}px {s['md']}px;
}}
QPushButton#secondaryButton:hover {{ background-color: {t['surfaceHover']}; }}
QPushButton#secondaryButton:pressed {{ background-color: {t['surfaceActive']}; }}
QPushButton#secondaryButton:disabled {{
    background-color: {t['controlDisabled']}; color: {t['textFaint']};
}}
/* Lists sit inside a bordered card: Qt draws a QListWidget's own border
   beneath its viewport, so a QSS border on the list itself never shows. The
   card carries the border/radius, the list stays transparent. */
QFrame#listCard {{
    background-color: {t['surface']};
    border: 1px solid {t['borderStrong']};
    border-radius: {r['sm']}px;
}}
QFrame#listCard QListWidget {{
    background: transparent;
    border: none;
    padding: {s['xs']}px;
}}
QFrame#listCard QListWidget::item {{
    padding: {s['xs'] + 2}px {s['sm']}px;
    border: none;
    border-radius: {r['xs']}px;
    color: {t['text']};
}}
QFrame#listCard QListWidget::item:selected {{
    background-color: {t['accentSoft']};
    color: {t['accentSoftText']};
}}
QFrame#listCard QListWidget::item:hover {{
    background-color: {t['surfaceHover']};
    color: {t['accentSoftText']};
}}
QFrame#divider {{ background-color: {t['border']}; max-height: 1px; border: none; }}

QWidget#badge {{
    border-radius: {r['pill']}px;
    padding: 1px {s['sm']}px;
}}
""" + arrow_qss(theme)


# --------------------------------------------------------------------------
# Derived styles
# --------------------------------------------------------------------------
GLOBAL_DARK_STYLE = build_stylesheet("dark")
LIGHT_STYLE = build_stylesheet("light")

# The light theme used to be a separate hand-written sheet; both now come from
# the same generator, so the app-level sheet is the same as the window sheet.
LIGHT_APP_STYLE = LIGHT_STYLE


def get_theme_style(theme: str = "dark") -> str:
    """Full stylesheet for 'dark' (default) or 'light'."""
    return build_stylesheet("light" if theme == "light" else "dark")


def get_app_light_style() -> str:
    return LIGHT_APP_STYLE


def _message_box_qss(theme: str) -> str:
    t = palette(theme)
    r = RADIUS
    s = SPACE
    f = FONT
    return f"""
QMessageBox {{
    background-color: {t['surface']};
    color: {t['text']};
    font-family: {_UI_FONT};
    font-size: {f['body']}pt;
}}
QMessageBox QLabel {{ color: {t['text']}; background: transparent; }}
QMessageBox QPushButton {{
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-radius: {r['sm']}px;
    padding: {s['xs'] + 2}px {s['lg']}px;
    min-width: 72px;
}}
QMessageBox QPushButton:hover {{ background-color: {t['surfaceHover']}; }}
QMessageBox QPushButton:default {{
    background-color: {t['accent']};
    color: {t['textOnAccent']};
    border-color: transparent;
}}
QMessageBox QPushButton:default:hover {{ background-color: {t['accentHover']}; }}
"""


def _add_label_dialog_qss(theme: str) -> str:
    """Style for the "add label" dialog (controllers/label_manager.py).

    The dialog is built inline rather than as a dialogs/ module, and it was the
    one place still importing a constant that the v1.62 visual-layer rewrite
    deleted. The import failed inside the action handler, so the menu entry did
    nothing at all instead of opening the panel.
    """
    t = palette(theme)
    r = RADIUS
    s = SPACE
    f = FONT
    return f"""
QDialog {{
    background-color: {t['surface']};
    color: {t['text']};
    font-family: {_UI_FONT};
    font-size: {f['body']}pt;
}}
QDialog QLabel {{ color: {t['text']}; }}
QDialog QLineEdit, QDialog QComboBox {{
    background-color: {t['controlBg']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    padding: {s['xs']}px {s['sm']}px;
    border-radius: {r['sm']}px;
    min-height: 20px;
}}
QDialog QLineEdit:focus, QDialog QComboBox:focus {{
    border-color: {t['borderFocus']};
}}
QDialog QComboBox QAbstractItemView {{
    background-color: {t['surface']};
    color: {t['text']};
    selection-background-color: {t['accent']};
    selection-color: {t['textOnAccent']};
}}
QDialog QPushButton {{
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-radius: {r['sm']}px;
    padding: {s['xs'] + 2}px {s['lg']}px;
    min-height: 26px;
}}
QDialog QPushButton:hover {{ background-color: {t['surfaceHover']}; }}
/* Compact navigation buttons: the generic padding above is wider than a
   34 px button, which clipped the glyph and left an empty box. */
QDialog QPushButton#previewNavButton {{
    padding: 0px;
    min-width: 30px;
    font-weight: bold;
}}
QDialog QCheckBox {{ color: {t['text']}; spacing: {s['sm']}px; }}
"""


ADD_LABEL_DIALOG_STYLE = _add_label_dialog_qss("dark")
ADD_LABEL_DIALOG_STYLE_LIGHT = _add_label_dialog_qss("light")


def get_add_label_dialog_style(theme: str) -> str:
    return (ADD_LABEL_DIALOG_STYLE_LIGHT if theme == "light"
            else ADD_LABEL_DIALOG_STYLE)


MESSAGE_BOX_STYLE_DARK = _message_box_qss("dark")
MESSAGE_BOX_STYLE_LIGHT = _message_box_qss("light")


def _progress_qss(theme: str) -> str:
    t = palette(theme)
    r = RADIUS
    s = SPACE
    return f"""
QProgressDialog {{
    background-color: {t['surface']};
    color: {t['text']};
    font-family: {_UI_FONT};
}}
QProgressDialog QLabel {{ color: {t['text']}; }}
QProgressDialog QPushButton {{
    background-color: {t['surface']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-radius: {r['sm']}px;
    padding: {s['xs'] + 2}px {s['lg']}px;
}}
QProgressDialog QPushButton:hover {{ background-color: {t['surfaceHover']}; }}
QProgressBar {{
    background-color: {t['surfaceActive']};
    border: none; border-radius: {r['xs']}px;
    text-align: center; color: {t['text']};
    min-height: 8px;
}}
QProgressBar::chunk {{ background-color: {t['accent']}; border-radius: {r['xs']}px; }}
"""


PROGRESS_DIALOG_STYLE_DARK = _progress_qss("dark")
PROGRESS_DIALOG_STYLE_LIGHT = _progress_qss("light")


def _combo_qss(theme: str) -> str:
    t = palette(theme)
    r = RADIUS
    s = SPACE
    return f"""
QComboBox {{
    background-color: {t['controlBg']};
    color: {t['text']};
    border: 1px solid {t['controlBorder']};
    border-radius: {r['sm']}px;
    /* Must stay identical to the main QComboBox rule: this second rule wins
       when arrow_qss() is appended, and a stale padding here silently made
       every dropdown 2 px shorter than the buttons beside them. */
    padding: {s['xs'] + 2}px {s['sm']}px;
    padding-right: {s['xl']}px;
}}
QComboBox QAbstractItemView {{
    background-color: {t['surface']};
    color: {t['text']};
    selection-background-color: {t['accentSoft']};
    selection-color: {t['accentSoftText']};
}}
""" + arrow_qss(theme)


WHITE_COMBOBOX_STYLE = _combo_qss("light")


def _list_qss(theme: str) -> str:
    t = palette(theme)
    r = RADIUS
    s = SPACE
    return f"""
QListWidget {{
    background-color: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: {r['sm']}px;
    padding: {s['xs']}px;
}}
QListWidget::item {{
    padding: {s['xs'] + 2}px {s['sm']}px;
    border-radius: {r['xs']}px;
    color: {t['text']};
}}
QListWidget::item:selected {{
    background-color: {t['accentSoft']};
    color: {t['accentSoftText']};
}}
QListWidget::item:hover {{ background-color: {t['surfaceHover']}; }}
"""


SOURCE_LIST_STYLE = _list_qss("dark")
OUTPUT_LIST_STYLE = SOURCE_LIST_STYLE


HELP_BUTTON_STYLE_DARK = f"""
QPushButton#helpButton {{
    background-color: {palette('dark')['surfaceHover']};
    color: {palette('dark')['textMuted']};
    /* controlBorder, not border: the inline chip is not covered by the
       generated sheet, so it kept the 1.2:1 dark 'border' token and read as
       a black square instead of a round "?" chip. */
    border: 1px solid {palette('dark')['controlBorder']};
    border-radius: {RADIUS['pill']}px;
    font-size: {FONT['caption']}pt;
    font-weight: {WEIGHT['semibold']};
    padding: 0;
    /* An inline stylesheet replaces the sheet's #helpButton rule wholesale,
       so the square geometry has to be restated here or the chip collapses
       to the text width and renders as a rounded rectangle. */
    min-width: 26px; max-width: 26px;
    min-height: 26px; max-height: 26px;
}}
QPushButton#helpButton:hover {{
    background-color: {palette('dark')['accentSoft']};
    color: {palette('dark')['accentSoftText']};
}}
"""

HELP_BUTTON_STYLE_LIGHT = f"""
QPushButton#helpButton {{
    background-color: {palette('light')['surfaceHover']};
    color: {palette('light')['textMuted']};
    border: 1px solid {palette('light')['controlBorder']};
    border-radius: {RADIUS['pill']}px;
    font-size: {FONT['caption']}pt;
    font-weight: {WEIGHT['semibold']};
    padding: 0;
    /* An inline stylesheet replaces the sheet's #helpButton rule wholesale,
       so the square geometry has to be restated here or the chip collapses
       to the text width and renders as a rounded rectangle. */
    min-width: 26px; max-width: 26px;
    min-height: 26px; max-height: 26px;
}}
QPushButton#helpButton:hover {{
    background-color: {palette('light')['accentSoft']};
    color: {palette('light')['accentSoftText']};
}}
"""

HELP_BUTTON_STYLE = HELP_BUTTON_STYLE_DARK


def help_button_style(theme: str = None) -> str:
    """Inline style for the small round "?" button, for the active theme."""
    if theme is None:
        theme = globals().get("CURRENT_THEME", "dark")
    return HELP_BUTTON_STYLE_LIGHT if theme == "light" else HELP_BUTTON_STYLE_DARK


HOVER_HIGHLIGHT_BUTTON_STYLE = f"""
QPushButton {{
    background-color: {palette('dark')['surface']};
    color: {palette('dark')['text']};
    border: 1px solid {palette('dark')['controlBorder']};
    border-radius: {RADIUS['sm']}px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background-color: {palette('dark')['accent']};
    color: {palette('dark')['textOnAccent']};
    border-color: transparent;
}}
"""

DIALOG_CONTROL_DARK_QSS = f"""
QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QLineEdit, QTextEdit {{
    background-color: {palette('dark')['controlBg']};
    color: {palette('dark')['text']};
    border: 1px solid {palette('dark')['controlBorder']};
    border-radius: {RADIUS['sm']}px;
    padding: 4px;
    selection-background-color: {palette('dark')['accent']};
}}
QPushButton {{
    background-color: {palette('dark')['surface']};
    color: {palette('dark')['text']};
    border: 1px solid {palette('dark')['controlBorder']};
    border-radius: {RADIUS['sm']}px;
    padding: 6px 16px;
}}
QPushButton:hover {{ background-color: {palette('dark')['surfaceHover']}; }}
QCheckBox {{ color: {palette('dark')['text']}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
""" + arrow_qss("dark")

PRIMARY_BLUE = palette("light")["accent"]

# --------------------------------------------------------------------------
# Current theme state (read by widgets that paint by hand)
# --------------------------------------------------------------------------
CURRENT_THEME = "dark"
CURRENT_MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE_DARK


def get_message_box_style(theme: str) -> str:
    return (MESSAGE_BOX_STYLE_LIGHT if theme == "light"
            else MESSAGE_BOX_STYLE_DARK)


def get_progress_dialog_style(theme: str) -> str:
    return (PROGRESS_DIALOG_STYLE_LIGHT if theme == "light"
            else PROGRESS_DIALOG_STYLE_DARK)


def _theme_bound(getter):
    """A module-level stylesheet constant that follows the active theme.

    `MESSAGE_BOX_STYLE` and `PROGRESS_DIALOG_STYLE` used to be plain strings
    built once for the dark palette. Five call sites import them by name, so
    every message box and progress dialog stayed black in light mode - seen as
    a dark "batch finished" panel over a light app.

    A Qt stylesheet must be a real ``str`` (``setStyleSheet`` rejects anything
    else), so the constants are rebindable module globals refreshed by
    ``set_current_theme``. Same names, same type, always current.
    """
    return getter


MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE_DARK
PROGRESS_DIALOG_STYLE = PROGRESS_DIALOG_STYLE_DARK
WARNING_MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE_DARK
ERROR_MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE_DARK
SUCCESS_MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE_DARK


def set_current_theme(theme: str) -> None:
    global CURRENT_THEME, CURRENT_MESSAGE_BOX_STYLE
    global MESSAGE_BOX_STYLE, PROGRESS_DIALOG_STYLE
    global WARNING_MESSAGE_BOX_STYLE, ERROR_MESSAGE_BOX_STYLE
    global SUCCESS_MESSAGE_BOX_STYLE
    CURRENT_THEME = "light" if theme == "light" else "dark"
    CURRENT_MESSAGE_BOX_STYLE = get_message_box_style(CURRENT_THEME)
    MESSAGE_BOX_STYLE = CURRENT_MESSAGE_BOX_STYLE
    PROGRESS_DIALOG_STYLE = get_progress_dialog_style(CURRENT_THEME)
    WARNING_MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE
    ERROR_MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE
    SUCCESS_MESSAGE_BOX_STYLE = MESSAGE_BOX_STYLE


def tokens(theme: str = None) -> Dict[str, str]:
    """Colour tokens for the active theme (convenience for widgets)."""
    return palette(theme or CURRENT_THEME)
