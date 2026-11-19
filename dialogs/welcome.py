"""First-run entry points for opening a workspace or trying the demo stack."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from locales import trans
from utils.design_tokens import FONT, SPACE
from utils.settings_store import DEFAULT_THEME


class WelcomeDialog(QDialog):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self._main_window = main_window
        self.setWindowTitle(trans.t('welcome_title'))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*(SPACE['xl'],) * 4)
        layout.setSpacing(SPACE['md'])

        heading = QLabel(trans.t('welcome_heading'))
        heading.setWordWrap(True)
        font = heading.font()
        font.setPointSize(FONT['title'])
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)

        body = QLabel(trans.t('welcome_body'))
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(body)

        self._settings_button = QToolButton()
        self._settings_button.setText(trans.t('menu_settings'))
        self._settings_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._settings_button.setCheckable(True)
        self._settings_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self._settings_button, alignment=Qt.AlignmentFlag.AlignLeft)

        self._preferences = QWidget()
        preferences = QVBoxLayout(self._preferences)
        preferences.setContentsMargins(0, 0, 0, 0)
        preferences.setSpacing(SPACE['sm'])
        theme_row = QHBoxLayout()
        theme_label = QLabel(trans.t('welcome_choose_theme'))
        theme = QComboBox()
        for code in ('light', 'dark'):
            theme.addItem(trans.t(f'theme_{code}'), code)
        current = getattr(main_window, 'ui_theme', DEFAULT_THEME)
        theme.setCurrentIndex(theme.findData(current))
        theme.setEnabled(main_window is not None)
        if main_window is not None:
            theme.currentIndexChanged.connect(lambda index: main_window.apply_theme(theme.itemData(index)))
        theme_label.setBuddy(theme)
        theme_row.addWidget(theme_label)
        theme_row.addWidget(theme)
        theme_row.addStretch()
        preferences.addLayout(theme_row)

        manager = getattr(self._main_window, 'update_manager', None)
        self.chk_include_beta = QCheckBox(trans.t('action_include_beta'))
        self.chk_include_beta.setProperty('controlId', 'chk_include_beta')
        self.chk_include_beta.setChecked(bool(getattr(manager, 'include_beta', False)))
        self.chk_include_beta.setToolTip(trans.t('action_include_beta_hint'))
        self.chk_include_beta.setEnabled(manager is not None)
        if manager is not None:
            self.chk_include_beta.toggled.connect(manager.set_include_beta)
        preferences.addWidget(self.chk_include_beta)
        hint = QLabel(trans.t('welcome_experimental_hint'))
        hint.setObjectName('mutedLabel')
        hint.setWordWrap(True)
        preferences.addWidget(hint)
        layout.addWidget(self._preferences)

        row = QHBoxLayout()
        row.addStretch()
        demo_btn = QPushButton(trans.t('btn_load_demo'))
        demo_btn.setEnabled(main_window is not None)
        demo_btn.setToolTip(trans.t('btn_load_demo_hint'))
        demo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        demo_btn.clicked.connect(self._load_demo)
        row.addWidget(demo_btn)
        ok = QPushButton(trans.t('btn_get_started'))
        ok.setObjectName("primaryButton")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        row.addWidget(ok)
        layout.addLayout(row)

        self._settings_button.toggled.connect(self._show_preferences)
        self._preferences.hide()
        self._settings_button.setArrowType(Qt.ArrowType.RightArrow)
        width = min(440, self.screen().availableGeometry().width())
        self.resize(width, layout.totalHeightForWidth(width))
        ok.setFocus()

    def _load_demo(self) -> None:
        if self._main_window is not None:
            self._main_window.load_demo_stack()
            self.accept()

    def _show_preferences(self, visible):
        self._settings_button.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)
        self._preferences.setVisible(visible)
        self.layout().activate()
        self.resize(self.width(), self.layout().totalHeightForWidth(self.width()))
