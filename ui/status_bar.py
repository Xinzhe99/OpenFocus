"""Left-aligned task indicators and notices, with a separate sponsor entry."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QSizePolicy, QStatusBar, QStyle, QStyleOption,
    QStylePainter, QWidget,
)


class SponsorLink(QLabel):
    def keyPressEvent(self, event):
        # The label has one link. Qt can focus it without selecting its anchor.
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            self.linkActivated.emit('contact')
            return
        super().keyPressEvent(event)


class WorkspaceStatusBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizeGripEnabled(False)
        content = QWidget(self)
        self._row = QHBoxLayout(content)
        self._row.setContentsMargins(4, 0, 0, 0)
        self._row.setSpacing(6)
        self._message_label = QLabel(content)
        self._message_label.setTextFormat(Qt.TextFormat.PlainText)
        self._message_label.setSizePolicy(QSizePolicy.Policy.Ignored,
                                          QSizePolicy.Policy.Preferred)
        self._row.addWidget(self._message_label, 1)
        self.addPermanentWidget(content, 1)
        self._review_text = ''
        self.messageChanged.connect(self._update_message_label)

    def add_activity(self, widget):
        self._row.insertWidget(self._row.count() - 1, widget)

    def set_review_status(self, text):
        self._review_text = text
        if not self.currentMessage():
            self._update_message_label('')

    def _update_message_label(self, message):
        text = message or self._review_text
        self._message_label.setText(text)
        self._message_label.setToolTip(text)
        self._message_label.setAccessibleName(text)

    def paintEvent(self, event):
        # Qt's timed messages use our label. Only paint the background here
        # so the native message cannot overlap progress or the cancel button.
        painter = QStylePainter(self)
        option = QStyleOption()
        option.initFrom(self)
        painter.drawPrimitive(QStyle.PrimitiveElement.PE_PanelStatusBar, option)
