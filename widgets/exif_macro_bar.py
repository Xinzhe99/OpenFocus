"""Clickable EXIF dynamic macro chip bar for text label and watermark editors."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from PyQt6.QtWidgets import QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from core.exif_macros import EXIF_MACRO_TOKENS
from locales import trans
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


class ExifMacroChipBar(QWidget):
    """Compact flow bar of clickable EXIF macro chips that insert into a QLineEdit."""

    def __init__(
        self,
        target_edit: QLineEdit,
        *,
        tokens: Sequence[str] = EXIF_MACRO_TOKENS,
        sample_meta: Optional[Mapping[str, str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._target_edit = target_edit
        self._buttons: dict[str, QPushButton] = {}

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
            btn.setAutoDefault(False)
            btn.setFixedHeight(24)
            btn.setStyleSheet(
                "QPushButton#macroChipButton {"
                "  padding: 1px 7px;"
                "  border-radius: 10px;"
                "  font-size: 8.5pt;"
                "}"
            )
            btn.clicked.connect(lambda _checked=False, tok=token: self._insert_token(tok))
            self._flow.add(btn)
            self._buttons[token] = btn
        layout.addWidget(self._flow)
        self.update_sample_meta(sample_meta or {})

    @property
    def buttons(self) -> dict[str, QPushButton]:
        return self._buttons

    def update_sample_meta(self, sample_meta: Mapping[str, str]) -> None:
        for token, btn in self._buttons.items():
            key = _TOKEN_TIP_KEYS.get(token, "")
            tip = trans.t(key) if key else token
            name = token[1:-1]
            sample = str(sample_meta.get(name, "") or "").strip()
            if sample:
                tip = f"{tip} ({sample})"
            btn.setToolTip(tip)
            btn.setAccessibleName(f"{token} {tip}")

    def _insert_token(self, token: str) -> None:
        edit = self._target_edit
        if edit is None:
            return
        current = edit.text()
        pos = edit.cursorPosition()
        prefix = " " if (pos > 0 and current[pos - 1] not in (" ", "\t", "-", "|", "/", "(", "[")) else ""
        edit.insert(prefix + token)
        edit.setFocus()
