"""First-run welcome dialog: quick start guide + theme choice."""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from locales import trans


class WelcomeDialog(QDialog):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self._main_window = main_window
        self.setWindowTitle(trans.t('welcome_title'))
        self.setMinimumWidth(480)
        self._apply_dialog_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        body = QLabel(trans.t('welcome_body'))
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(body)

        # Theme choice (applies immediately through the main window and persists)
        theme_row = QHBoxLayout()
        theme_label = QLabel(trans.t('welcome_choose_theme'))
        theme_row.addWidget(theme_label)
        self._theme_buttons = {}
        for code in ("dark", "light"):
            btn = QPushButton(trans.t(f'theme_{code}'))
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, c=code: self._choose_theme(c))
            self._theme_buttons[code] = btn
            theme_row.addWidget(btn)
        layout.addLayout(theme_row)

        # Demo stack: zero-friction first success
        demo_row = QHBoxLayout()
        demo_btn = QPushButton(trans.t('btn_load_demo'))
        # Keep the button at its natural width and let the wrapping hint
        # absorb the remaining space. An Ignored policy made the button
        # collapse to 0 px next to the hint label.
        demo_btn.setSizePolicy(QSizePolicy.Policy.Fixed,
                               QSizePolicy.Policy.Fixed)
        demo_btn.setToolTip(trans.t('btn_load_demo'))
        demo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        demo_btn.clicked.connect(self._load_demo)
        demo_row.addWidget(demo_btn)
        demo_hint = QLabel(trans.t('btn_load_demo_hint'))
        # Long translations run to ~850 px on one line in a 480 px dialog.
        demo_hint.setWordWrap(True)
        # Theme-aware muted colour: the old hardcoded #57606a was ~2:1 on the
        # dark dialog background, effectively invisible.
        demo_hint.setObjectName("welcome_hint")
        self._demo_hint = demo_hint
        self._refresh_hint_color()
        demo_row.addWidget(demo_hint)
        demo_row.addStretch()
        layout.addLayout(demo_row)

        row = QHBoxLayout()
        row.addStretch()
        ok = QPushButton(trans.t('btn_get_started'))
        ok.setObjectName("primaryButton")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        row.addWidget(ok)
        layout.addLayout(row)

        self._sync_theme_buttons()

    def _load_demo(self) -> None:
        if self._main_window is not None:
            self._main_window.load_demo_stack()
            self.accept()

    def _is_dark(self) -> bool:
        from utils.settings_store import DEFAULT_THEME
        theme = (getattr(self._main_window, "ui_theme", DEFAULT_THEME)
                 if self._main_window else DEFAULT_THEME)
        return theme != "light"

    def _apply_dialog_style(self):
        """Styling comes from the generated application stylesheet."""
        self.setStyleSheet("")

    def _choose_theme(self, code: str) -> None:
        if self._main_window is not None:
            self._main_window.apply_theme(code)
        self._apply_dialog_style()
        self._sync_theme_buttons()
        self._refresh_hint_color()

    def _refresh_hint_color(self):
        from utils.design_tokens import palette
        theme = "dark" if self._is_dark() else "light"
        self._demo_hint.setStyleSheet(
            f"color: {palette(theme)['textMuted']}; background: transparent;")

    def _sync_theme_buttons(self) -> None:
        from utils.settings_store import DEFAULT_THEME
        current = (getattr(self._main_window, "ui_theme", DEFAULT_THEME)
                   if self._main_window else DEFAULT_THEME)
        for code, btn in self._theme_buttons.items():
            btn.setChecked(code == current)
