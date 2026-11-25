"""Left-aligned task indicators and a separate sponsor entry."""
from html import escape
import json
import os
from unicodedata import category

from PyQt6.QtCore import QEvent, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStatusBar, QStyle,
    QStyleOption, QStylePainter, QWidget,
)

from locales import trans
from utils.account_client import BASE_URL
from utils.design_tokens import palette


class SponsorLink(QLabel):
    """Show bundled copy immediately and refresh the published placement while open."""

    _endpoint = f'{BASE_URL}/v1/sponsor-placement'
    _max_bytes = 4096

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('statusSponsor')
        self.setContentsMargins(2, 0, 6, 0)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse
                                     | Qt.TextInteractionFlag.LinksAccessibleByKeyboard)
        self.linkActivated.connect(self._open)
        self._language = None
        self._placement = None
        self._reply = None
        self._credit = self._text = ''
        self._url = QUrl()
        self._network = QNetworkAccessManager(self)
        self._timer = QTimer(self, interval=5 * 60 * 1000)
        self._timer.timeout.connect(self._fetch)

    def refresh(self, theme):
        self._colors = palette(theme)
        language = trans.current_lang
        if self._language != language:
            self._language = language
            self._placement = None
            if self.isVisible():
                self._fetch()
        self._update_content()

    def _update_content(self):
        locale = 'zh' if self._language == 'zh' else 'en'
        self._credit = trans.t('status_credit')
        self._text, self._url = self._placement or (
            trans.t('status_sponsor'), QUrl(f'{BASE_URL}/sponsor?lang={locale}'))
        self.setAccessibleName(self._text)
        self.setAccessibleDescription(self._credit)
        self.setToolTip(escape(f'{self._credit}\n{self._text}'))
        self._render()
        self.updateGeometry()

    def sizeHint(self):
        metrics = self.fontMetrics()
        width = sum(metrics.horizontalAdvance(part) for part in (self._credit, '   ·   ', self._text))
        return QSize(width + 16, super().sizeHint().height())

    def minimumSizeHint(self):
        return QSize(0, super().sizeHint().height())

    def _render(self):
        if not self._text:
            return
        metrics = self.fontMetrics()
        width = max(0, self.contentsRect().width() - 4)
        sponsor = metrics.elidedText(self._text, Qt.TextElideMode.ElideRight, width)
        prefix_width = width - metrics.horizontalAdvance(sponsor + '   ·   ')
        credit = metrics.elidedText(self._credit, Qt.TextElideMode.ElideRight, prefix_width)
        prefix = f'{escape(credit)}&nbsp; · &nbsp;' if credit and credit != '…' else ''
        self.setText(f'{prefix}<a href="sponsor" style="color:{self._colors["accentSoftText"]}">'
                     f'{escape(sponsor)}</a>')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._render()

    def showEvent(self, event):
        super().showEvent(event)
        self._timer.start()
        self._fetch()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _fetch(self):
        if self._reply is not None or self._language is None:
            return
        self._request_language = self._language
        request = QNetworkRequest(QUrl(f'{self._endpoint}?lang={self._language}'))
        attribute = QNetworkRequest.Attribute
        request.setAttribute(attribute.CookieLoadControlAttribute, QNetworkRequest.LoadControl.Manual)
        request.setAttribute(attribute.CookieSaveControlAttribute, QNetworkRequest.LoadControl.Manual)
        request.setAttribute(attribute.RedirectPolicyAttribute, QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
        request.setRawHeader(b'Accept', b'application/json')
        request.setTransferTimeout(5000)
        self._reply = self._network.get(request)
        self._reply.setReadBufferSize(self._max_bytes + 1)
        self._reply.readyRead.connect(self._check_size)
        self._reply.finished.connect(self._received)

    def _check_size(self):
        if self._reply.bytesAvailable() > self._max_bytes:
            self._reply.abort()

    def _received(self):
        reply, self._reply = self._reply, None
        reply.deleteLater()
        if self._request_language != self._language:
            if self.isVisible():
                self._fetch()
            return
        if (reply.error() != QNetworkReply.NetworkError.NoError
                or reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute) != 200
                or reply.bytesAvailable() > self._max_bytes):
            return
        try:
            data = json.loads(bytes(reply.readAll()))
        except (ValueError, RecursionError):
            return
        if not isinstance(data, dict):
            return
        text, target = data.get('text'), data.get('url')
        if (not isinstance(text, str) or not 1 <= len(text) <= 80 or not text.strip()
                or any(category(ch) in ('Cc', 'Cs', 'Zl', 'Zp') for ch in text)
                or not isinstance(target, str) or len(target) > 2048):
            return
        url = QUrl(target, QUrl.ParsingMode.StrictMode)
        if not url.isValid() or url.scheme() != 'https' or not url.host() or '@' in url.authority():
            return
        if self._placement != (text, url):
            self._placement = text, url
            self._update_content()

    def _open(self):
        QDesktopServices.openUrl(self._url)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            self._open()
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
        self._reveal_path = ''
        self._pending_reveal_path = ''
        self._reveal_button = QPushButton(trans.t('share_reveal'), content)
        self._reveal_button.setObjectName('statusRevealButton')
        self._reveal_button.setFlat(True)
        self._reveal_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reveal_button.setSizePolicy(QSizePolicy.Policy.Maximum,
                                          QSizePolicy.Policy.Fixed)
        self._reveal_button.clicked.connect(self._reveal_active_path)
        self._reveal_button.hide()
        self._row.addWidget(self._reveal_button, 0)
        self.addPermanentWidget(content, 1)
        self._review_text = ''
        self.messageChanged.connect(self._update_message_label)
        self._sponsor = None

    def add_sponsor(self, label):
        self._sponsor = label
        self._size_sponsor(self.width())
        self.addPermanentWidget(label)

    def _size_sponsor(self, width):
        if self._sponsor is not None:
            # Preserve space for task progress, Cancel and transient messages.
            self._sponsor.setMaximumWidth(width * 3 // 5)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._size_sponsor(event.size().width())

    def add_activity(self, widget):
        idx = self._row.indexOf(self._message_label)
        self._row.insertWidget(idx if idx >= 0 else 0, widget)

    def set_review_status(self, text):
        self._review_text = text
        if not self.currentMessage():
            self._update_message_label('')

    def show_path_message(self, message: str, path: str, timeout: int = 8000) -> None:
        """Show a transient message paired with an inline Show in Folder button."""
        reveal = os.fspath(path) if path else ''
        self._pending_reveal_path = reveal
        self.showMessage(message, timeout)
        if self._pending_reveal_path and self.currentMessage() == message:
            self._bind_reveal_path(self._pending_reveal_path)
            self._pending_reveal_path = ''

    def _bind_reveal_path(self, path: str) -> None:
        self._reveal_path = path
        if path:
            self._reveal_button.setText(trans.t('share_reveal'))
            self._reveal_button.setToolTip(path)
            self._reveal_button.show()
        else:
            self._reveal_button.hide()

    def _reveal_active_path(self) -> None:
        path = self._reveal_path
        if not path:
            return
        from utils.file_manager import select_in_file_manager
        from utils.ui_utils import show_error_box

        def finished(opened, error=None):
            if not opened:
                show_error_box(self.window(), trans.t('msg_share_failed_title'),
                               trans.t('msg_share_reveal_failed'), error or path)

        try:
            select_in_file_manager(path, parent=self.window(), on_finished=finished)
        except Exception as exc:
            finished(False, str(exc))

    def _update_message_label(self, message):
        text = message or self._review_text
        self._message_label.setText(text)
        self._message_label.setToolTip(text)
        self._message_label.setAccessibleName(text)
        if message and self._pending_reveal_path:
            self._bind_reveal_path(self._pending_reveal_path)
            self._pending_reveal_path = ''
        else:
            self._pending_reveal_path = ''
            self._bind_reveal_path('')

    def paintEvent(self, event):
        # Qt's timed messages use our label. Only paint the background here
        # so the native message cannot overlap progress or the cancel button.
        painter = QStylePainter(self)
        option = QStyleOption()
        option.initFrom(self)
        painter.drawPrimitive(QStyle.PrimitiveElement.PE_PanelStatusBar, option)
