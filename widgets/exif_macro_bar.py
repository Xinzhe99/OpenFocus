"""Clickable EXIF dynamic macro chip bar for text label and watermark editors."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QLineEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from core.exif_macros import EXIF_MACRO_TOKENS
from locales import trans
from utils.design_tokens import FONT, RADIUS, SPACE, WEIGHT, palette
from widgets.flow_layout import FlowPanel

_TOKEN_TIP_KEYS = {
    "{value}": "exif_macro_tip_value",
    "{date}": "exif_macro_tip_date",
    "{camera}": "exif_macro_tip_camera",
    "{lens}": "exif_macro_tip_lens",
    "{aperture}": "exif_macro_tip_aperture",
    "{shutter}": "exif_macro_tip_shutter",
    "{iso}": "exif_macro_tip_iso",
    "{focal_length}": "exif_macro_tip_focal_length",
    "{method}": "exif_macro_tip_method",
    "{frames}": "exif_macro_tip_frames",
}


def macro_chip_bar_stylesheet(theme: str = "dark") -> str:
    """Return theme-aware QSS for EXIF macro chips and live preview badge."""
    t = palette(theme)
    return f"""
QPushButton#macroChipButton {{
    background-color: {t['surfaceSunken']};
    color: {t['textMuted']};
    border: 1px solid {t['border']};
    border-radius: 11px;
    padding: 1px 8px;
    min-height: 20px;
    max-height: 22px;
    font-size: {FONT['caption']}pt;
    font-weight: {WEIGHT['medium']};
}}
QPushButton#macroChipButton[hasSample="true"] {{
    background-color: {t['accentSoft']};
    color: {t['accentSoftText']};
    border: 1px solid {t['controlBorder']};
}}
QPushButton#macroChipButton:hover {{
    background-color: {t['accentSoft']};
    color: {t['accentSoftText']};
    border-color: {t['accent']};
}}
QPushButton#macroChipButton:pressed {{
    background-color: {t['surfaceActive']};
    color: {t['text']};
    border-color: {t['accentPressed']};
}}
QPushButton#macroChipButton:focus {{
    border-color: {t['borderFocus']};
    outline: none;
}}
QLabel#macroPreviewLabel {{
    background-color: {t['surfaceSunken']};
    color: {t['text']};
    border: 1px solid {t['border']};
    border-left: 3px solid {t['accent']};
    border-radius: {RADIUS['xs']}px;
    padding: {SPACE['xs']}px {SPACE['sm']}px;
    font-size: {FONT['caption']}pt;
}}
"""


def create_macro_preview_label(parent: Optional[QWidget] = None) -> QLabel:
    """Create a themed live preview badge for resolved EXIF macro templates."""
    label = QLabel("", parent)
    label.setObjectName("macroPreviewLabel")
    label.setProperty("controlId", "exif_macro_preview_label")
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    label.hide()
    return label


class ExifMacroChipBar(QWidget):
    """Compact flow bar of clickable EXIF macro chips that insert into a QLineEdit."""

    def __init__(
        self,
        target_edit: QLineEdit,
        *,
        tokens: Sequence[str] = EXIF_MACRO_TOKENS,
        sample_meta: Optional[Mapping[str, str]] = None,
        theme: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._target_edit = target_edit
        self._buttons: dict[str, QPushButton] = {}
        self._theme: Optional[str] = theme

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        self.hint_label = QLabel(trans.t("exif_macros_hint"))
        self.hint_label.setObjectName("mutedLabel")
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        self._flow = FlowPanel(h_spacing=4, v_spacing=4)
        self.setProperty("controlId", "exif_macro_chip_bar")
        for token in tokens:
            btn = QPushButton(token)
            btn.setObjectName("macroChipButton")
            btn.setProperty("controlId", f"macro_chip_{token[1:-1]}")
            btn.setProperty("hasSample", False)
            btn.setAutoDefault(False)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda _checked=False, tok=token: self._insert_token(tok))
            self._flow.add(btn)
            self._buttons[token] = btn
        layout.addWidget(self._flow)
        self.apply_theme(theme)
        self.update_sample_meta(sample_meta or {})

    @property
    def buttons(self) -> dict[str, QPushButton]:
        return self._buttons

    def apply_theme(self, theme: Optional[str] = None) -> None:
        """Apply Light/Dark/Gray theme tokens to chip buttons."""
        if theme is None:
            from ui import styles as ui_styles
            theme = getattr(ui_styles, "CURRENT_THEME", "dark")
        self._theme = str(theme)
        self.setStyleSheet(macro_chip_bar_stylesheet(self._theme))

    def update_sample_meta(self, sample_meta: Mapping[str, str]) -> None:
        for token, btn in self._buttons.items():
            key = _TOKEN_TIP_KEYS.get(token, "")
            desc = trans.t(key) if key else token
            name = token[1:-1]
            sample = str(sample_meta.get(name, "") or "").strip()
            has_sample = bool(sample) or token == "{value}"
            if sample:
                detail = trans.t("exif_macro_tip_current").format(value=sample)
            elif token == "{value}":
                detail = trans.t("exif_macro_tip_current").format(value="1")
            else:
                detail = trans.t("exif_macro_tip_unavailable")
            tip = f"{token} — {desc}\n{detail}"
            btn.setToolTip(tip)
            btn.setAccessibleName(f"{token} {desc}")
            if bool(btn.property("hasSample")) != has_sample:
                btn.setProperty("hasSample", has_sample)
                style = btn.style()
                if style is not None:
                    style.unpolish(btn)
                    style.polish(btn)

    def _insert_token(self, token: str) -> None:
        edit = self._target_edit
        if edit is None:
            return
        current = edit.text()
        pos = edit.cursorPosition()
        prefix = " " if (pos > 0 and current[pos - 1] not in (" ", "\t", "-", "|", "/", "(", "[")) else ""
        edit.insert(prefix + token)
        edit.setFocus()
