"""Compare a proposed interface text size with the standard size."""
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QVBoxLayout,
)

from locales import trans
from ui.context_help import register_help_target
from utils.design_tokens import FONT, SPACE, UI_FONT_RANGE, WEIGHT, ui_font_sizes
from widgets.stepper import StepperSpinBox


class FontSizeDialog(QDialog):
    def __init__(self, points, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('menu_ui_font'))
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE['lg'], SPACE['lg'], SPACE['lg'], SPACE['md'])
        root.setSpacing(SPACE['md'])

        form = QFormLayout()
        self.size_spin = StepperSpinBox(minimum=UI_FONT_RANGE[0], maximum=UI_FONT_RANGE[1],
                                       decimals=0, step=1, value=points)
        self.size_spin.setSuffix(' pt')
        self.size_spin.setKeyboardTracking(True)
        label = QLabel(trans.t('menu_ui_font'))
        label.setBuddy(self.size_spin)
        form.addRow(label, self.size_spin)
        root.addLayout(form)

        previews = QHBoxLayout()
        root.addLayout(previews)
        self._reference, reference = self._preview('ui_font_medium')
        previews.addWidget(reference, 1)
        self._candidate, candidate = self._preview('label_preview')
        previews.addWidget(candidate, 1)
        self._update_samples(self._reference, FONT['body'])
        self._update_samples(self._candidate, points)
        self.size_spin.valueChanged.connect(lambda value: self._update_samples(self._candidate, int(value)))

        buttons = QDialogButtonBox()
        apply = buttons.addButton(trans.t('btn_ok'), QDialogButtonBox.ButtonRole.AcceptRole)
        apply.setObjectName('primaryButton')
        apply.setDefault(True)
        buttons.addButton(trans.t('btn_cancel'), QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        register_help_target(self, "font_size")
        self.size_spin.setFocus()
        screen = self.screen().availableGeometry()
        self.resize(min(580, screen.width() - 32), self.sizeHint().height())

    @staticmethod
    def _preview(title):
        group = QGroupBox(trans.t(title))
        layout = QVBoxLayout(group)
        layout.setSpacing(SPACE['sm'])
        heading = QLabel(trans.t('group_fusion'))
        body = QLabel(trans.t('depth_source_current'))
        caption = QLabel(trans.t('depth_tab_map'))
        caption.setObjectName('mutedLabel')
        size = QLabel()
        size.setObjectName('mutedLabel')
        for label in (size, heading, body, caption):
            label.setWordWrap(True)
            layout.addWidget(label)
        layout.addStretch()
        return (heading, body, caption, size), group

    @staticmethod
    def _update_samples(samples, points):
        heading, body, caption, size = samples
        sizes = ui_font_sizes(points)
        heading.setStyleSheet(f"font-size: {sizes['title']}pt; font-weight: {WEIGHT['semibold']};")
        body.setStyleSheet(f"font-size: {sizes['body']}pt;")
        caption.setStyleSheet(f"font-size: {sizes['caption']}pt;")
        size.setText(f'{points} pt')
