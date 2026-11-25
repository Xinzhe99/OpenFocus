"""First-run welcome entry points with step overview and interactive tour launch."""
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from locales import trans
from utils.design_tokens import FONT, PALETTES, RADIUS, SPACE, WEIGHT, palette
from utils.settings_store import DEFAULT_THEME


class WelcomeDialog(QDialog):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self._main_window = main_window
        self._step_frames: list[QFrame] = []
        self._step_badges: list[QLabel] = []
        self._step_titles: list[QLabel] = []
        self._step_descs: list[QLabel] = []
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

        # 3-step visual workflow preview cards
        steps_row = QHBoxLayout()
        steps_row.setSpacing(SPACE['sm'])
        for idx in range(1, 4):
            card = QFrame()
            card.setObjectName("welcomeStepCard")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(SPACE['md'], SPACE['sm'] + 2, SPACE['md'], SPACE['sm'] + 2)
            cl.setSpacing(SPACE['xs'])

            badge = QLabel(str(idx))
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setFixedSize(22, 22)
            cl.addWidget(badge, alignment=Qt.AlignmentFlag.AlignLeft)

            stitle = QLabel(trans.t(f'welcome_step{idx}_title'))
            stitle.setWordWrap(True)
            cl.addWidget(stitle)

            sdesc = QLabel(trans.t(f'welcome_step{idx}_desc'))
            sdesc.setWordWrap(True)
            cl.addWidget(sdesc)
            cl.addStretch()

            self._step_frames.append(card)
            self._step_badges.append(badge)
            self._step_titles.append(stitle)
            self._step_descs.append(sdesc)
            steps_row.addWidget(card, 1)
        layout.addLayout(steps_row)

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
        for code in (*PALETTES, 'system'):
            theme.addItem(trans.t(f'theme_{code}'), code)
        current = getattr(main_window, 'theme_mode', DEFAULT_THEME)
        theme.setCurrentIndex(theme.findData(current))
        theme.setEnabled(main_window is not None)
        if main_window is not None:
            theme.currentIndexChanged.connect(self._on_theme_selected)
        self._theme_combo = theme
        theme_label.setBuddy(theme)
        theme_row.addWidget(theme_label)
        theme_row.addWidget(theme)
        theme_row.addStretch()
        preferences.addLayout(theme_row)

        manager = getattr(self._main_window, 'update_manager', None)
        self.chk_include_beta = QCheckBox(trans.t('action_include_beta'))
        self.chk_include_beta.setProperty('controlId', 'chk_include_beta')
        self.chk_include_beta.setChecked(bool(getattr(manager, 'include_beta', False)))
        self.chk_include_beta.setToolTip(trans.t('action_include_beta_with_hint'))
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
        row.setSpacing(SPACE['sm'])
        self.btn_load_demo = QPushButton(trans.t('btn_load_demo'))
        self.btn_load_demo.setObjectName("secondaryButton")
        self.btn_load_demo.setEnabled(main_window is not None)
        self.btn_load_demo.setToolTip(trans.t('btn_load_demo_hint'))
        self.btn_load_demo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_demo.clicked.connect(self._load_demo)
        row.addWidget(self.btn_load_demo)

        row.addStretch()

        self.btn_get_started = QPushButton(trans.t('btn_get_started'))
        self.btn_get_started.setObjectName("secondaryButton")
        self.btn_get_started.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_get_started.clicked.connect(self.accept)
        row.addWidget(self.btn_get_started)

        self.btn_start_tour = QPushButton(trans.t('btn_start_guided_tour'))
        self.btn_start_tour.setObjectName("primaryButton")
        self.btn_start_tour.setDefault(True)
        self.btn_start_tour.setEnabled(main_window is not None)
        self.btn_start_tour.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_tour.clicked.connect(self._start_guided_tour)
        row.addWidget(self.btn_start_tour)
        layout.addLayout(row)

        self._apply_card_theme()
        self._settings_button.toggled.connect(self._show_preferences)
        self._preferences.hide()
        self._settings_button.setArrowType(Qt.ArrowType.RightArrow)
        width = min(560, self.screen().availableGeometry().width())
        self.resize(width, layout.totalHeightForWidth(width))
        if self.btn_start_tour.isEnabled():
            self.btn_start_tour.setFocus()
        else:
            self.btn_get_started.setFocus()

    def _on_theme_selected(self, index: int) -> None:
        if self._main_window is not None:
            self._main_window.apply_theme(self._theme_combo.itemData(index))
            self._apply_card_theme()

    def _apply_card_theme(self) -> None:
        theme = getattr(self._main_window, 'ui_theme', DEFAULT_THEME)
        t = palette(theme)
        for card in self._step_frames:
            card.setStyleSheet(
                f"QFrame#welcomeStepCard {{ background-color: {t['surfaceSunken']}; "
                f"border: 1px solid {t['border']}; border-radius: {RADIUS['md']}px; }}"
            )
        for badge in self._step_badges:
            badge.setStyleSheet(
                f"background-color: {t['accentSoft']}; color: {t['accentSoftText']}; "
                f"border-radius: 11px; font-weight: {WEIGHT['bold']}; font-size: 9pt;"
            )
        for stitle in self._step_titles:
            stitle.setStyleSheet(
                f"color: {t['text']}; font-weight: {WEIGHT['semibold']}; font-size: 9.5pt; background: transparent;"
            )
        for sdesc in self._step_descs:
            sdesc.setStyleSheet(f"color: {t['textMuted']}; font-size: 8.5pt; background: transparent;")

    def _load_demo(self) -> None:
        if self._main_window is not None:
            self._main_window.load_demo_stack()
            self.accept()

    def _start_guided_tour(self) -> None:
        main_window = self._main_window
        self.accept()
        if main_window is not None and hasattr(main_window, "start_guided_tour"):
            QTimer.singleShot(0, lambda: main_window.start_guided_tour(load_demo_if_empty=True))

    def _show_preferences(self, visible):
        self._settings_button.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)
        self._preferences.setVisible(visible)
        self.layout().activate()
        self.resize(self.width(), self.layout().totalHeightForWidth(self.width()))
