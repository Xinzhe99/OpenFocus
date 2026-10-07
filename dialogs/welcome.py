"""First-run welcome dialog: quick start guide + theme choice."""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from locales import trans
from ui.styles import CURRENT_MESSAGE_BOX_STYLE


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
        demo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        demo_btn.clicked.connect(self._load_demo)
        demo_row.addWidget(demo_btn)
        demo_hint = QLabel(trans.t('btn_load_demo_hint'))
        # Theme-aware muted colour: the old hardcoded #57606a was ~2:1 on the
        # dark dialog background, effectively invisible.
        demo_hint.setObjectName("welcome_hint")
        demo_hint.setStyleSheet(
            "background: transparent;" if self._is_dark() else
            "color: #57606a; background: transparent;")
        demo_row.addWidget(demo_hint)
        demo_row.addStretch()
        layout.addLayout(demo_row)

        row = QHBoxLayout()
        row.addStretch()
        ok = QPushButton(trans.t('btn_get_started'))
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
        theme = getattr(self._main_window, "ui_theme", "dark") if self._main_window else "dark"
        return theme != "light"

    def _apply_dialog_style(self):
        """Keep the dialog readable in whichever theme is active."""
        theme = getattr(self._main_window, "ui_theme", "dark") if self._main_window else "dark"
        if theme == "light":
            self.setStyleSheet("QDialog { background-color: #ffffff; } QLabel { color: #1f2328; }")
        else:
            self.setStyleSheet(
                "QDialog { background-color: #2b2b2b; border: 1px solid #444; }"
                "QLabel { color: #d0d0d0; }"
            )

    def _choose_theme(self, code: str) -> None:
        if self._main_window is not None:
            self._main_window.apply_theme(code)
        self._apply_dialog_style()
        self._sync_theme_buttons()

    def _sync_theme_buttons(self) -> None:
        current = getattr(self._main_window, "ui_theme", "dark") if self._main_window else "dark"
        for code, btn in self._theme_buttons.items():
            btn.setChecked(code == current)
