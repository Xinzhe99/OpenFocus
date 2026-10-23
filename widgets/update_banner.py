"""In-window update banner.

The startup update check is silent by design (status bar + Help menu), but a
user who never looks at either never learns a new version exists. This banner
sits above the image views after a check finds a release, and stays until the
user acts on it or dismisses it for that version.

Dismissal is per tag: a newer release re-shows the banner, the dismissed one
does not come back (Help -> Check for Updates re-arms it for a manual check).
"""
from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from locales import trans


class UpdateBanner(QWidget):
    """One-line notification: version found, primary action, dismiss."""

    update_requested = pyqtSignal()
    whats_new_requested = pyqtSignal()
    dismissed = pyqtSignal(str)

    def __init__(self, parent: Any | None = None) -> None:
        super().__init__(parent)
        self._tag = ""
        self._ready = False
        self._theme = "dark"
        self.setObjectName("updateBanner")
        # A bare QWidget subclass ignores its own background QSS rule unless
        # this attribute is set: the strip painted as an unstyled rectangle
        # and the text sat on the window colour.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        self.lbl_text = QLabel("", self)
        self.lbl_text.setObjectName("updateBannerText")
        self.lbl_hint = QLabel("", self)
        self.lbl_hint.setObjectName("updateBannerHint")

        self.btn_update = QPushButton("", self)
        self.btn_details = QPushButton("", self)
        # Plain text button: the default QPushButton chrome drew a filled
        # rounded box around the glyph (an empty-looking square in the light
        # theme, where the glyph colour did not carry over).
        self.btn_dismiss = QPushButton("\u2715", self)
        self.btn_dismiss.setFixedSize(22, 22)
        self.btn_dismiss.setToolTip(trans.t("update_hide_version"))
        self.btn_dismiss.setFlat(True)
        self.btn_dismiss.setObjectName("updateBannerClose")
        self.btn_dismiss.setCursor(Qt.CursorShape.PointingHandCursor)
        for btn in (self.btn_update, self.btn_details, self.btn_dismiss):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_update.setDefault(True)

        layout.addWidget(self.lbl_text)
        layout.addWidget(self.lbl_hint)
        layout.addStretch(1)
        layout.addWidget(self.btn_update)
        layout.addWidget(self.btn_details)
        layout.addWidget(self.btn_dismiss)

        self.btn_update.clicked.connect(self.update_requested.emit)
        self.btn_details.clicked.connect(self.whats_new_requested.emit)
        self.btn_dismiss.clicked.connect(self._on_dismiss)

        self.hide()
        self.apply_theme(self._theme)
        self.retranslate()

    def apply_theme(self, theme: str) -> None:
        """Colour the strip for the active theme.

        The banner is created during init_ui, so the light theme's
        "strip every inline dark stylesheet" pass would otherwise blank it
        out (a white strip with no accent). main.apply_theme calls this after
        its own pass so the correct palette is always the last word.
        """
        from utils.design_tokens import palette
        self._theme = "light" if theme == "light" else "dark"
        p = palette(self._theme)
        self.setStyleSheet(
            f"QWidget#updateBanner {{ background-color: {p['accentSoft']};"
            f" border-bottom: 1px solid {p['accentSoftText']}; }}"
            f" QLabel#updateBannerText {{ color: {p['text']};"
            " background: transparent; }"
            f" QLabel#updateBannerHint {{ color: {p['textMuted']};"
            " background: transparent; }"
            f" QPushButton#updateBannerClose {{ background: transparent;"
            f" border: none; font-size: 15px; padding: 0; color: {p['text']}; }}"
            f" QPushButton#updateBannerClose:hover {{"
            f" background: {p['surfaceActive']}; border-radius: 11px; }}"
        )

    # ------------------------------------------------------------------ state

    @property
    def tag(self) -> str:
        """Release tag currently shown ("" when the banner is hidden)."""
        return self._tag

    def show_for(self, tag: str, current_version: str, ready: bool = False) -> None:
        """Display the banner for `tag`.

        `ready` switches the text and button to "restart to install" — the
        build is already downloaded, so the primary action restarts instead
        of starting a download.
        """
        self._tag = tag or ""
        self.set_ready(ready)
        self.lbl_hint.setText(
            trans.t("update_banner_hint").format(current=current_version))
        self.show()

    def set_ready(self, ready: bool) -> None:
        self._ready = bool(ready)
        if self._ready:
            self.lbl_text.setText(
                trans.t("update_banner_ready").format(version=self._tag))
            self.btn_update.setText(trans.t("btn_restart_now"))
        else:
            self.lbl_text.setText(
                trans.t("update_banner_new_version").format(version=self._tag))
            self.btn_update.setText(trans.t("btn_update_now"))
        self.btn_details.setText(trans.t("btn_see_whats_new"))
        self.btn_details.setVisible(True)

    def hide_banner(self) -> None:
        self._tag = ""
        self.hide()

    def retranslate(self) -> None:
        """Re-apply every string after a language switch."""
        self.btn_dismiss.setToolTip(trans.t("update_hide_version"))
        if self._tag:
            self.set_ready(self._ready)

    def _on_dismiss(self) -> None:
        tag = self._tag
        self.hide_banner()
        self.dismissed.emit(tag)
