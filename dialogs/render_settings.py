"""Immediate render options, using the main window's existing controls."""
from PyQt6.QtCore import QEvent, QSignalBlocker, QTimer, Qt
from PyQt6.QtWidgets import (
    QDialog, QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from locales import trans
from utils.design_tokens import SPACE
from utils.output_sharpen import DEFAULT_SHARPEN_STRENGTH, MAX_SHARPEN_STRENGTH
from utils.ui_utils import size_dialog_to_content


class _OptionLabel(QLabel):
    def __init__(self, option):
        super().__init__()
        self._option = option
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setBuddy(option)
        self.setEnabled(option.isEnabled())
        option.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802 (Qt casing)
        if watched is self._option and event.type() == QEvent.Type.EnabledChange:
            self.setEnabled(watched.isEnabled())
        return super().eventFilter(watched, event)

    def mouseReleaseEvent(self, event):  # noqa: N802 (Qt casing)
        if event.button() == Qt.MouseButton.LeftButton and self._option.isEnabled():
            self._option.click()
        super().mouseReleaseEvent(event)


class RenderSettingsDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.setProperty("controlId", "render_settings_dialog")
        self._window = window
        self._sharpen = window.chk_sharpen_output
        sections = (
            ("render_section_workflow", window.chk_quick_preview, window.chk_reverse_order),
            ("render_section_brightness", window.chk_exposure_match, window.chk_base_correct),
            ("render_section_output", self._sharpen),
        )
        self._sections = []
        self._options = tuple(option for _, *options in sections for option in options)
        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        options_layout = QVBoxLayout(page)
        options_layout.setContentsMargins(0, 0, SPACE['sm'], 0)
        options_layout.setSpacing(SPACE['md'])
        options_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for title_key, *options in sections:
            group = QGroupBox()
            self._sections.append((group, title_key))
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(SPACE['sm'])
            options_layout.addWidget(group)
            for option in options:
                row = QHBoxLayout()
                row.setSpacing(SPACE['sm'])
                label = _OptionLabel(option)
                option._dialog_label = label
                option.refit()
                row.addWidget(option, 0, Qt.AlignmentFlag.AlignTop)
                row.addWidget(label, 1)
                group_layout.addLayout(row)
                option.show()
            if options[0] is self._sharpen:
                group_layout.addWidget(self._make_sharpen_controls())
        scroll.setWidget(page)
        self._scroll = scroll
        root.addWidget(scroll, 1)
        self._close_button = QPushButton()
        self._close_button.setObjectName("secondaryButton")
        self._close_button.clicked.connect(self.accept)
        root.addWidget(self._close_button, 0, Qt.AlignmentFlag.AlignRight)
        self._close_button.setDefault(True)
        self.retranslate()
        self._fit()

    def _make_sharpen_controls(self):
        self._sharpen_controls = QWidget()
        layout = QVBoxLayout(self._sharpen_controls)
        layout.setContentsMargins(SPACE['xl'], 0, 0, 0)
        layout.setSpacing(SPACE['xs'])
        self._strength_label = QLabel()
        self._strength_label.setWordWrap(True)
        layout.addWidget(self._strength_label)
        row = QHBoxLayout()
        layout.addLayout(row)
        self._strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._strength_slider.setRange(0, MAX_SHARPEN_STRENGTH)
        self._strength_slider.setSingleStep(1)
        self._strength_slider.setPageStep(10)
        self._strength_slider.setTracking(False)
        row.addWidget(self._strength_slider, 1)
        self._strength_spin = QSpinBox()
        self._strength_spin.setRange(0, MAX_SHARPEN_STRENGTH)
        self._strength_spin.setSuffix("%")
        self._strength_spin.setKeyboardTracking(False)
        self._strength_label.setBuddy(self._strength_spin)
        row.addWidget(self._strength_spin)
        self._strength_reset = QPushButton()
        self._strength_reset.setObjectName("secondaryButton")
        self._strength_reset.setAutoDefault(False)
        self._strength_reset.clicked.connect(
            lambda: self._strength_slider.setValue(DEFAULT_SHARPEN_STRENGTH))
        layout.addWidget(self._strength_reset, 0, Qt.AlignmentFlag.AlignLeft)
        self._sharpen_risk = QLabel()
        self._sharpen_risk.setObjectName('mutedLabel')
        self._sharpen_risk.setWordWrap(True)
        layout.addWidget(self._sharpen_risk)

        self._strength_slider.sliderMoved.connect(self._display_strength)
        self._strength_slider.valueChanged.connect(self._display_strength)
        self._strength_slider.valueChanged.connect(self._window.set_sharpen_strength)
        self._strength_spin.valueChanged.connect(self._strength_slider.setValue)
        self._sharpen.installEventFilter(self)
        self._sharpen.toggled.connect(self._show_sharpen_controls)
        self._sharpen_controls.setEnabled(self._sharpen.isEnabled())
        self._sync_strength()
        self._sharpen_controls.setVisible(self._sharpen.isChecked())
        return self._sharpen_controls

    def _sync_strength(self):
        strength = self._window.sharpen_strength
        with QSignalBlocker(self._strength_slider):
            self._strength_slider.setValue(strength)
        self._display_strength(strength)

    def _display_strength(self, strength):
        with QSignalBlocker(self._strength_spin):
            self._strength_spin.setValue(strength)

    def showEvent(self, event):  # noqa: N802 (Qt casing)
        self._sync_strength()
        super().showEvent(event)

    def _fit(self):
        root = self.layout()
        scroll = self._scroll
        options_layout = scroll.widget().layout()
        size_dialog_to_content(self, scroll, preferred_width=560)
        # Wrapped labels have an inflated unconstrained sizeHint; use their
        # height at the chosen viewport width instead of a nearly full screen.
        margins = root.contentsMargins()
        content_width = max(1, self.width() - margins.left() - margins.right()
                            - 2 * scroll.frameWidth())
        content_height = options_layout.totalHeightForWidth(content_width)
        fixed_height = (margins.top() + margins.bottom() + root.spacing()
                        + self._close_button.sizeHint().height())
        self.resize(self.width(), min(self.height(), content_height + fixed_height))


    def eventFilter(self, watched, event):  # noqa: N802 (Qt casing)
        if watched is self._sharpen and event.type() == QEvent.Type.EnabledChange:
            self._sharpen_controls.setEnabled(self._sharpen.isEnabled())
        return super().eventFilter(watched, event)

    def _show_sharpen_controls(self, enabled):
        self._sync_strength()
        self._sharpen_controls.setVisible(enabled)
        if enabled and self.isVisible():
            QTimer.singleShot(0, self._reveal_sharpen_controls)

    def _reveal_sharpen_controls(self):
        if self._sharpen_controls.isVisible():
            self._scroll.ensureWidgetVisible(self._sharpen_controls)

    def retranslate(self):
        for group, title_key in self._sections:
            group.setTitle(trans.t(title_key))
        for option in self._options:
            option.refit()
        strength = trans.t('render_sharpen_strength')
        hint = trans.t('render_sharpen_strength_hint')
        self._strength_label.setText(strength)
        self._strength_label.setToolTip(hint)
        self._strength_reset.setText(trans.t('btn_reset'))
        for control in (self._strength_slider, self._strength_spin):
            control.setAccessibleName(strength)
            control.setAccessibleDescription(hint)
            control.setToolTip(hint)
        self._sharpen_risk.setText(trans.t('render_sharpen_risk'))
        self.setWindowTitle(trans.t("render_settings_title"))
        self._close_button.setText(trans.t("dialog_close"))
