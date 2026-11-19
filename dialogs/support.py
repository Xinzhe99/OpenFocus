"""Payment codes and the published OpenFocus sponsor roster."""
import json

from PyQt6.QtCore import QSize, QTimer, Qt, QUrl
from PyQt6.QtGui import QPainter, QPixmap
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QListView, QListWidget, QWidget

from dialogs._community_layout import community_body, community_card, community_label
from locales import trans
from utils import resource_path
from utils.account_client import BASE_URL


class _PaymentCode(QWidget):
    """Draw the actual square at the same size with a four-module quiet zone."""

    def __init__(self, title, filename, bounds):
        super().__init__()
        self._code = QPixmap(resource_path('assets', 'readme', filename)).copy(*bounds)
        self.setFixedSize(228, 228)
        self.setAccessibleName(f'{title} · QR')

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.white)
        painter.drawPixmap(self.rect().adjusted(24, 24, -24, -24), self._code)


class SupportUsDialog(QDialog):
    """Keep payment available independently of the anonymous roster request."""

    _SPONSORS_URL = f'{BASE_URL}/v1/sponsors'
    _MAX_ROSTER_BYTES = 135 * 1024

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('support_title'))
        self.setMinimumWidth(570)
        body = community_body(self, trans.t('support_heading'), trans.t('support_message'),
                              width=610, height=552)
        codes = QHBoxLayout()
        codes.setSpacing(12)
        # Source pixels of the two module grids, excluding the different margins
        # baked into the images. Paint a fresh white scan margin around each.
        for title, filename, bounds in (
            (trans.t('support_alipay'), 'alipay-coffee.jpg', (30, 33, 534, 534)),
            ('PayPal', 'paypal-coffee.png', (64, 64, 528, 528)),
        ):
            card, column = community_card()
            column.addWidget(community_label(title, 'section'), alignment=Qt.AlignmentFlag.AlignHCenter)
            column.addWidget(_PaymentCode(title, filename, bounds), alignment=Qt.AlignmentFlag.AlignHCenter)
            codes.addWidget(card, 1)
        body.addLayout(codes)

        card, section = community_card()
        heading = trans.t('support_acknowledgments')
        section.addWidget(community_label(heading, 'section'))
        self._status = community_label(trans.t('support_roster_loading'), 'muted')
        section.addWidget(self._status)
        self._names = QListWidget()
        self._names.setObjectName('communityNames')
        self._names.setAccessibleName(heading)
        self._names.setFlow(QListView.Flow.LeftToRight)
        self._names.setWrapping(True)
        self._names.setResizeMode(QListView.ResizeMode.Adjust)
        self._names.setGridSize(QSize(160, 36))
        self._names.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._names.hide()
        section.addWidget(self._names)
        body.addWidget(card)
        body.addStretch()

        # Use a separate network manager with cookies and redirects disabled.
        # Neither the GUI account session nor the device identifier is needed.
        self._network = QNetworkAccessManager(self)
        request = QNetworkRequest(QUrl(self._SPONSORS_URL))
        attribute = QNetworkRequest.Attribute
        request.setAttribute(attribute.CookieLoadControlAttribute, QNetworkRequest.LoadControl.Manual)
        request.setAttribute(attribute.CookieSaveControlAttribute, QNetworkRequest.LoadControl.Manual)
        request.setAttribute(attribute.RedirectPolicyAttribute, QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
        request.setRawHeader(b'Accept', b'application/json')
        reply = self._reply = self._network.get(request)
        reply.setReadBufferSize(self._MAX_ROSTER_BYTES + 1)
        reply.readyRead.connect(self._check_roster_size)
        reply.finished.connect(self._roster_finished)
        self._deadline = QTimer(self)
        self._deadline.setSingleShot(True)
        self._deadline.timeout.connect(reply.abort)
        self._deadline.start(5000)

    def _check_roster_size(self):
        if self._reply is not None and self._reply.bytesAvailable() > self._MAX_ROSTER_BYTES:
            self._reply.abort()

    def _roster_finished(self):
        reply, self._reply = self._reply, None
        self._deadline.stop()
        if reply is None:
            return
        names = None
        if (reply.error() == QNetworkReply.NetworkError.NoError
                and reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute) == 200
                and reply.bytesAvailable() <= self._MAX_ROSTER_BYTES):
            try:
                payload = json.loads(bytes(reply.readAll()))
                roster = payload.get('sponsors') if isinstance(payload, dict) else None
                if (isinstance(roster, list) and len(roster) <= 250
                        and all(isinstance(name, str) and 1 <= len(name) <= 64 for name in roster)):
                    names = roster
            except (ValueError, RecursionError):
                pass
        reply.deleteLater()
        if not names:
            self._status.setText(trans.t('support_roster_offline' if names is None else 'support_roster_empty'))
            return
        self._status.hide()
        self._names.addItems(names)
        for row, name in enumerate(names):
            self._names.item(row).setToolTip(name)
        self._names.setFixedHeight(36 if len(names) <= 3 else 108)
        self._names.show()

    def done(self, result):
        self._deadline.stop()
        reply, self._reply = self._reply, None
        if reply is not None:
            reply.abort()
            reply.deleteLater()
        super().done(result)
