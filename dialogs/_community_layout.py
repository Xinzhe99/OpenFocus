"""Shared layout for the project’s contact and support dialogs."""
from html import escape

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

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
