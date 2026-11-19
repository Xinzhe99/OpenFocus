"""Shared layout for the project’s contact and support dialogs."""
from html import escape

from PyQt6.QtCore import QByteArray, QBuffer, QIODevice, Qt
from PyQt6.QtGui import QImageReader, QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from dialogs import _community_download as downloads
from locales import trans
from ui import styles
from utils.design_tokens import FONT, RADIUS, SPACE, WEIGHT, palette


def _colors(dialog):
    return palette(getattr(dialog.parentWidget(), 'ui_theme', styles.CURRENT_THEME))


def apply_community_theme(dialog):
    colors = _colors(dialog)
    dialog.setStyleSheet(f"""
        QDialog#communityDialog, QWidget#communityContent, QScrollArea#communityScroll {{
            background: {colors['bg']};
        }}
        QFrame#communityCard {{
            background: {colors['surface']};
            border: 1px solid {colors['border']};
            border-radius: {RADIUS['lg']}px;
        }}
        QLabel#communityTitle {{
            font-size: {FONT['display']}pt; font-weight: {WEIGHT['bold']};
            color: {colors['text']};
        }}
        QLabel#communitySection {{
            font-size: {FONT['subtitle']}pt; font-weight: {WEIGHT['semibold']};
            color: {colors['text']};
        }}
        QLabel#communityMuted {{ color: {colors['textMuted']}; }}
        QLabel#communityAccent {{ color: {colors['accentSoftText']}; }}
        QLabel#communityLink {{ border: 1px solid transparent; }}
        QLabel#communityLink:focus {{
            border-color: {colors['borderFocus']}; border-radius: {RADIUS['xs']}px;
        }}
        QListWidget#communityNames {{ background: transparent; border: none; }}
        QListWidget#communityNames::item {{
            padding: {SPACE['sm']}px; border-radius: {RADIUS['sm']}px;
            background: {colors['surfaceSunken']}; color: {colors['text']};
        }}
        QListWidget#communityNames::item:selected {{
            background: {colors['accentSoft']}; color: {colors['accentSoftText']};
        }}
    """)
    for label in dialog.findChildren(QLabel, 'communityLink'):
        _color_link(label, colors['accentSoftText'])


def community_body(dialog, title, intro, *, width, height):
    dialog.resize(width, height)
    dialog.setObjectName('communityDialog')
    apply_community_theme(dialog)

    root = QVBoxLayout(dialog)
    root.setContentsMargins(SPACE['xl'], SPACE['lg'], SPACE['xl'], SPACE['lg'])
    root.setSpacing(SPACE['md'])
    scroll = QScrollArea()
    scroll.setObjectName('communityScroll')
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    content = QWidget()
    content.setObjectName('communityContent')
    body = QVBoxLayout(content)
    body.setContentsMargins(0, 0, 0, 0)
    body.setSpacing(SPACE['md'])
    body.addWidget(community_label(title, 'title'))
    body.addWidget(community_label(intro, 'muted'))
    scroll.setWidget(content)
    root.addWidget(scroll, 1)

    footer = QHBoxLayout()
    footer.addStretch()
    close = QPushButton(trans.t('btn_close'))
    close.setMinimumWidth(96)
    close.setAutoDefault(False)
    close.clicked.connect(dialog.accept)
    footer.addWidget(close)
    root.addLayout(footer)
    return body


def community_card():
    card = QFrame()
    card.setObjectName('communityCard')
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*([SPACE['lg']] * 4))
    layout.setSpacing(SPACE['sm'])
    return card, layout


def community_label(text, role=''):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    if role:
        label.setObjectName(f'community{role.title()}')
    return label


def community_link(dialog, text, url):
    label = QLabel()
    label.setProperty('communityUrl', url)
    label.setObjectName('communityLink')
    label.setAccessibleName(text)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    label.setOpenExternalLinks(True)
    label.setWordWrap(True)
    _color_link(label, _colors(dialog)['accentSoftText'])
    return label


def _color_link(label, color):
    url = escape(label.property('communityUrl'), quote=True)
    title = escape(label.accessibleName())
    label.setText(f'<a style="color: {color};" href="{url}">{title}</a>')


def show_community_dialog(parent, dialog_type):
    dialog = next((item for item in parent.findChildren(dialog_type) if item.isVisible()), None)
    if dialog is None:
        dialog = dialog_type(parent)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def add_published_qr(dialog, column, requests, title, slot, size):
    """Display the currently published PNG with its original quiet zone."""
    heading = community_label(title, 'section')
    heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
    column.addWidget(heading)

    qr = QLabel()
    qr.setObjectName(slot)
    qr.setAccessibleName(title)
    qr.setAlignment(Qt.AlignmentFlag.AlignCenter)
    qr.setStyleSheet('background: white;')
    qr.setFixedSize(size, size)
    qr.hide()
    column.addWidget(qr, alignment=Qt.AlignmentFlag.AlignHCenter)

    status = community_label(trans.t('community_qr_loading'), 'muted')
    status.setAlignment(Qt.AlignmentFlag.AlignCenter)
    column.addWidget(status)
    url = f'{downloads.QR_BASE_URL}/{slot}'
    link = community_link(dialog, trans.t('community_qr_link'), url)
    link.setAlignment(Qt.AlignmentFlag.AlignCenter)
    column.addWidget(link)

    def display(code, data):
        if data:
            encoded = QByteArray(data)
            buffer = QBuffer(encoded)
            buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            reader = QImageReader(buffer, b'png')
            dimensions = reader.size()
            if (1 <= dimensions.width() <= downloads.QR_MAX_DIMENSION
                    and 1 <= dimensions.height() <= downloads.QR_MAX_DIMENSION):
                image = reader.read()
                if not image.isNull():
                    ratio = qr.devicePixelRatioF()
                    pixels = round(size * ratio)
                    pixmap = QPixmap.fromImage(image.scaled(pixels, pixels, Qt.AspectRatioMode.KeepAspectRatio))
                    pixmap.setDevicePixelRatio(ratio)
                    qr.setPixmap(pixmap)
                    qr.show()
                    status.hide()
                    return
        status.setText(trans.t('community_qr_unpublished' if code == 404 else 'community_qr_unavailable'))

    requests.get(url, 'image/png', downloads.QR_MAX_BYTES, display)
