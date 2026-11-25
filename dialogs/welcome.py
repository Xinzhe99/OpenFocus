"""First-run welcome entry points with step overview and interactive tour launch."""
from PyQt6 import sip
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from locales import trans
from ui import styles
from utils.design_tokens import PALETTES, RADIUS, SPACE, WEIGHT, palette, ui_font_sizes
from utils.settings_store import DEFAULT_THEME


class WelcomeDialog(QDialog):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self._main_window = main_window
        self._step_frames: list[QFrame] = []
        self._step_titles: list[QLabel] = []
        self._step_descs: list[QLabel] = []
        self.setWindowTitle(trans.t('welcome_title'))

        root = QVBoxLayout(self)
        root.setContentsMargins(*(SPACE['xl'],) * 4)
        root.setSpacing(SPACE['md'])
        self._scroll = QScrollArea()
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE['md'])
        self._scroll.setWidget(content)
        root.addWidget(self._scroll, 1)

        heading = QLabel(trans.t('welcome_heading'))
        heading.setWordWrap(True)
        heading.setTextFormat(Qt.TextFormat.PlainText)
        title_size = ui_font_sizes(styles.CURRENT_FONT_SIZE)['title']
        heading.setStyleSheet(f"font-size: {title_size}pt; font-weight: {WEIGHT['bold']};")
        layout.addWidget(heading)

        body = QLabel(trans.t('welcome_body'))
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(body)

        # 3-step visual workflow preview cards
        steps_row = QHBoxLayout()
        steps_row.setSpacing(SPACE['sm'])
        self._steps_row = steps_row
        for idx in range(1, 4):
            card = QFrame()
            card.setObjectName("welcomeStepCard")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(SPACE['md'], SPACE['sm'] + 2, SPACE['md'], SPACE['sm'] + 2)
            cl.setSpacing(SPACE['xs'])

            stitle = QLabel(trans.t(f'welcome_step{idx}_title'))
            stitle.setWordWrap(True)
            stitle.setTextFormat(Qt.TextFormat.PlainText)
            cl.addWidget(stitle)

            sdesc = QLabel(trans.t(f'welcome_step{idx}_desc'))
            sdesc.setWordWrap(True)
            sdesc.setTextFormat(Qt.TextFormat.PlainText)
            cl.addWidget(sdesc)
            cl.addStretch()

            self._step_frames.append(card)
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
        theme_row = QFormLayout()
        theme_row.setContentsMargins(0, 0, 0, 0)
        theme_row.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        theme_row.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
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
        theme_row.addRow(theme_label, theme)
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
        self._action_row = row
        self.btn_load_demo = QPushButton(trans.t('btn_load_demo'))
        self.btn_load_demo.setObjectName("secondaryButton")
        self.btn_load_demo.setEnabled(callable(getattr(main_window, 'load_demo_stack', None)))
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
        self.btn_start_tour.setEnabled(callable(getattr(main_window, 'start_guided_tour', None)))
        self.btn_start_tour.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_tour.clicked.connect(self._start_guided_tour)
        row.addWidget(self.btn_start_tour)
        root.addLayout(row)

        self._apply_card_theme()
        self._settings_button.toggled.connect(self._show_preferences)
        self._preferences.hide()
        self._settings_button.setArrowType(Qt.ArrowType.RightArrow)
        self._fit(560)
        default_button = self.btn_start_tour if self.btn_start_tour.isEnabled() else self.btn_get_started
        default_button.setDefault(True)
        default_button.setFocus()

    def _on_theme_selected(self, index: int) -> None:
        if self._main_window is not None:
            self._main_window.apply_theme(self._theme_combo.itemData(index))
            self._apply_card_theme()
            self._fit(self.width())

    def _apply_card_theme(self) -> None:
        theme = getattr(self._main_window, 'ui_theme', styles.CURRENT_THEME)
        t = palette(theme)
        for card in self._step_frames:
            card.setStyleSheet(
                f"QFrame#welcomeStepCard {{ background-color: {t['surfaceSunken']}; "
                f"border: 1px solid {t['border']}; border-radius: {RADIUS['md']}px; }}"
            )
        for stitle in self._step_titles:
            stitle.setStyleSheet(
                f"color: {t['text']}; font-weight: {WEIGHT['semibold']}; background: transparent;"
            )
        for sdesc in self._step_descs:
            sdesc.setStyleSheet(f"color: {t['textMuted']}; background: transparent;")

    def _load_demo(self) -> None:
        self._finish_and_run(getattr(self._main_window, 'load_demo_stack', None))

    def _start_guided_tour(self) -> None:
        self._finish_and_run(getattr(self._main_window, 'start_guided_tour', None), load_demo_if_empty=True)

    def _finish_and_run(self, action, **kwargs) -> None:
        window = self._main_window
        self.accept()
        if not callable(action):
            return

        def run():
            if (not sip.isdeleted(window) and not getattr(window, '_shutting_down', False)
                    and not getattr(window, '_close_confirmation_active', False)):
                action(**kwargs)

        # Return from the modal event loop before acting on the workspace.
        QTimer.singleShot(0, run)

    def _show_preferences(self, visible):
        self._settings_button.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)
        self._preferences.setVisible(visible)
        self._fit(self.width())
        if visible:
            self._scroll.ensureWidgetVisible(self._theme_combo)

    def _fit(self, width: int) -> None:
        self.ensurePolished()
        parent = self.parentWidget()
        screen = parent.screen() if parent is not None and not self.isVisible() else self.screen()
        available = screen.availableGeometry()
        style = self.style()
        frame = style.pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth, None, self)
        title = style.pixelMetric(QStyle.PixelMetric.PM_TitleBarHeight, None, self)
        width = min(width, max(1, available.width() - 2 * frame))
        max_height = max(1, available.height() - title - 2 * frame)
        root = self.layout()
        margins = root.contentsMargins()
        content_width = max(1, width - margins.left() - margins.right())

        # Measure the translated labels after the application font is applied.
        buttons = (self.btn_load_demo, self.btn_get_started, self.btn_start_tour)
        actions_width = sum(button.sizeHint().width() for button in buttons) + 2 * SPACE['sm']
        card_width = max(144, *(card.minimumSizeHint().width() for card in self._step_frames))
        steps_width = len(self._step_frames) * card_width + 2 * SPACE['sm']
        direction = QBoxLayout.Direction
        self._action_row.setDirection(
            direction.LeftToRight if actions_width <= content_width else direction.TopToBottom)
        self._steps_row.setDirection(
            direction.LeftToRight if steps_width <= content_width else direction.TopToBottom)

        content = self._scroll.widget()
        content_height = content.layout().totalHeightForWidth(content_width)
        fixed_height = (margins.top() + margins.bottom() + root.spacing()
                        + self._action_row.sizeHint().height())
        self.resize(width, min(max_height, content_height + fixed_height))
        root.activate()
