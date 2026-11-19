"""Payment codes and the published OpenFocus sponsor roster."""
import json

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QListView, QListWidget

from dialogs._community_download import CommunityDownloads
from dialogs._community_layout import add_published_qr, community_body, community_card, community_label
from locales import trans
from utils.account_client import BASE_URL


class SupportUsDialog(QDialog):
    """Display current payment codes and the roster using independent requests."""

    _SPONSORS_URL = f'{BASE_URL}/v1/sponsors'
    _MAX_ROSTER_BYTES = 135 * 1024

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('support_title'))
        requests = CommunityDownloads(self)
        body = community_body(self, trans.t('support_heading'), trans.t('support_message'),
                              width=610, height=552)
        self._codes = QHBoxLayout()
        self._codes.setSpacing(body.spacing())
        for title, slot in ((trans.t('support_alipay'), 'alipay'), ('PayPal', 'paypal')):
            card, column = community_card()
            add_published_qr(self, column, requests, title, slot, 228)
            self._codes.addWidget(card, 1)
        body.addLayout(self._codes)

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

        requests.get(self._SPONSORS_URL, 'application/json', self._MAX_ROSTER_BYTES, self._roster_finished)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        direction = (QHBoxLayout.Direction.LeftToRight if event.size().width() >= 590
                     else QHBoxLayout.Direction.TopToBottom)
        self._codes.setDirection(direction)

    def _roster_finished(self, _status, data):
        names = None
        if data is not None:
            try:
                payload = json.loads(data)
                roster = payload.get('sponsors') if isinstance(payload, dict) else None
                if (isinstance(roster, list) and len(roster) <= 250
                        and all(isinstance(name, str) and 1 <= len(name) <= 64 for name in roster)):
                    names = roster
            except (ValueError, RecursionError):
                pass
        if not names:
            self._status.setText(trans.t('support_roster_offline' if names is None else 'support_roster_empty'))
            return
        self._status.hide()
        self._names.addItems(names)
        for row, name in enumerate(names):
            self._names.item(row).setToolTip(name)
        self._names.setFixedHeight(36 if len(names) <= 3 else 108)
        self._names.show()

