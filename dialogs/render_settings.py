"""Immediate render options, using the main window's existing controls."""
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from locales import trans
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
        self._options = (
            window.chk_quick_preview, window.chk_reverse_order,
            window.chk_exposure_match, window.chk_base_correct,
            window.chk_sharpen_output,
        )
        self._hint_keys = (
            'chk_quick_preview_hint', 'chk_reverse_order_hint',
            'chk_exposure_match_hint', 'chk_base_correct_hint',
            'chk_sharpen_output_hint',
        )
        self._descriptions = []
        root = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        options_layout = QVBoxLayout(page)
        options_layout.setSpacing(12)
        options_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for option in self._options:
            row = QHBoxLayout()
            row.setAlignment(Qt.AlignmentFlag.AlignTop)
            row.setSpacing(8)
            label = _OptionLabel(option)
            option._dialog_label = label
            option.refit()
            description = QLabel()
            description.setObjectName('mutedLabel')
            description.setWordWrap(True)
            description.setSizePolicy(QSizePolicy.Policy.Ignored,
                                      QSizePolicy.Policy.Preferred)
            option.installEventFilter(self)
            self._descriptions.append(description)
            text_column = QVBoxLayout()
            text_column.setContentsMargins(0, 0, 0, 0)
            text_column.setSpacing(3)
            text_column.addWidget(label)
            text_column.addWidget(description)
            row.addWidget(option, 0, Qt.AlignmentFlag.AlignTop)
            row.addLayout(text_column, 1)
            options_layout.addLayout(row)
            option.show()
        scroll.setWidget(page)
        self._scroll = scroll
        root.addWidget(scroll, 1)
        self._close_button = QPushButton()
        self._close_button.setObjectName("secondaryButton")
        self._close_button.clicked.connect(self.accept)
        root.addWidget(self._close_button, 0, Qt.AlignmentFlag.AlignRight)
        self.retranslate()
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

    def eventFilter(self, watched, event):  # noqa: N802
        if event.type() == QEvent.Type.EnabledChange and watched in self._options:
            index = self._options.index(watched)
            self._descriptions[index].setEnabled(watched.isEnabled())
        return super().eventFilter(watched, event)

    def retranslate(self):
        for option in self._options:
            option.refit()
        for key, description, option in zip(self._hint_keys, self._descriptions,
                                            self._options):
            description.setText(trans.t(key))
            description.setToolTip(option.toolTip())
            description.setEnabled(option.isEnabled())
        self.setWindowTitle(trans.t("render_settings_title"))
        self._close_button.setText(trans.t("dialog_close"))

    def resizeEvent(self, event):  # noqa: N802 (Qt casing)
        super().resizeEvent(event)
        width = max(1, self._scroll.viewport().width() - 50)
        for option in self._options:
            option.refit(width)
