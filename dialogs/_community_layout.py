"""Shared layout for the project’s contact and support dialogs."""
from html import escape

import cv2
import numpy as np
from PyQt6.QtCore import QByteArray, QBuffer, QIODevice, QRect, Qt
from PyQt6.QtGui import QImage, QImageReader, QPainter, QPixmap
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
    return scroll, body


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


class _QrLabel(QLabel):
    def __init__(self, title, slot, size):
        super().__init__()
        self._code = None
        self._ratio = None
        self.setObjectName(slot)
        self.setAccessibleName(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet('background: white;')
        self.setFixedSize(size, size)
        self.hide()

    def set_code(self, image, *, source=None, overlay=None):
        self._code = image, source if source is not None else image.rect(), overlay
        self._render_code()
        self.show()

    def clear(self):
        self._code = None
        super().clear()

    def paintEvent(self, event):
        if self._code is not None and self.devicePixelRatioF() != self._ratio:
            self._render_code()
        super().paintEvent(event)

    def _render_code(self):
        image, source, overlay = self._code
        self._ratio = self.devicePixelRatioF()
        side = round(self.width() * self._ratio)
        # Reserve at least four modules of white space even for a 21-module QR.
        code_side = side * 7 // 10
        inset = (side - code_side) // 2
        target = QRect(inset, inset, code_side, code_side)
        pixmap = QPixmap(side, side)
        pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(pixmap)
        painter.drawImage(target, image, source)
        if overlay is not None:
            overlay(painter, target)
        painter.end()
        pixmap.setDevicePixelRatio(self._ratio)
        self.setPixmap(pixmap)


def community_qr(column, title, slot, size):
    qr = _QrLabel(title, slot, size)
    column.addWidget(qr, alignment=Qt.AlignmentFlag.AlignHCenter)
    return qr


def _qr_region(image):
    # Published artwork is upright with light or transparent padding. Measure
    # the dark pattern on white, but draw the original pixels to keep its logo.
    if image.hasAlphaChannel():
        opaque = QImage(image.size(), QImage.Format.Format_RGB32)
        opaque.fill(Qt.GlobalColor.white)
        painter = QPainter(opaque)
        painter.drawImage(0, 0, image)
        painter.end()
        image = opaque
    gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
    bits = gray.constBits()
    bits.setsize(gray.sizeInBytes())
    rows = np.frombuffer(bits, dtype=np.uint8).reshape(gray.height(), gray.bytesPerLine())
    mask = cv2.inRange(rows[:, :gray.width()], 0, 191)
    return QRect(*cv2.boundingRect(mask))


def add_published_qr(dialog, column, requests, title, slot, size):
    """Display the currently published PNG in the shared QR geometry."""
    heading = community_label(title, 'section')
    heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
    column.addWidget(heading)

    qr = community_qr(column, title, slot, size)

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
                    region = _qr_region(image)
                    if not region.isEmpty():
                        qr.set_code(image, source=region)
                        status.hide()
                        return
        status.setText(trans.t('community_qr_unpublished' if code == 404 else 'community_qr_unavailable'))

    requests.get(url, 'image/png', downloads.QR_MAX_BYTES, display)
