"""Illustrated help opened from the setting it describes."""
from html import escape

from PyQt6.QtCore import QRect, Qt, QUrl
from PyQt6.QtGui import QColor, QImage, QPainter, QTextDocument
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton

from locales import trans
from ui.styles import help_button_style
from utils.design_tokens import palette


def _illustration(kind, theme):
    colors = palette(theme)
    image = QImage(260, 108, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor(colors['surfaceSunken']))
    painter = QPainter(image)
    accent = QColor(colors['accent'])
    muted = QColor(colors['textMuted'])
    border = QColor(colors['controlBorder'])

    def box(x, y, width, height, color):
        painter.fillRect(x, y, width, height, color)

    def outline(x, y, width, height, color=border):
        painter.setPen(color)
        painter.drawRect(x, y, width, height)

    def text(x, y, width, label):
        painter.setPen(QColor(colors['text']))
        painter.drawText(QRect(x, y, width, 20), Qt.AlignmentFlag.AlignCenter, label)

    if kind == 'tiles':
        tile = QColor(colors['accentSoft'])
        box(67, 22, 72, 72, tile)
        box(121, 22, 72, 72, tile)
        box(121, 22, 18, 72, accent)
        outline(67, 22, 72, 72)
        outline(121, 22, 72, 72)
        text(67, 0, 72, '1024 px')
        text(92, 88, 76, '256 px')
    elif kind == 'registration_width':
        for x, y, width, height, number in ((25, 16, 100, 64, '4096 px'),
                                            (177, 26, 56, 36, '1024 px')):
            outline(x, y, width, height)
            box(x + width // 4, y + height // 4, width // 2, height // 2, accent)
            text(x - 8, 83, width + 16, number)
        text(132, 32, 36, '→')
    elif kind == 'reference':
        for index, (x, offset) in enumerate(((28, -7), (103, 0), (178, 7)), 1):
            outline(x, 18, 54, 58, accent if index == 2 else border)
            box(x + 19 + offset, 38, 16, 16, accent if index == 2 else muted)
            text(x, 82, 54, str(index))
    elif kind in ('kernel', 'radius'):
        for start, window in ((39, 1), (151, 2)):
            for row in range(7):
                for column in range(7):
                    distance = max(abs(row - 3), abs(column - 3))
                    color = accent if distance == 0 else (
                        QColor(colors['accentSoftText']) if distance <= window else border)
                    box(start + column * 10, 12 + row * 10, 8, 8, color)
            text(start - 5, 84, 80, ('3 × 3' if window == 1 else '5 × 5')
                 if kind == 'kernel' else f'r = {window}')
    elif kind == 'blend':
        for left, count in ((27, 1), (142, 5)):
            for column in range(9):
                mix = min(1., max(0., (column - 4) / count + .5))
                color = QColor.fromRgbF(*(a * mix + b * (1 - mix)
                                         for a, b in zip(accent.getRgbF()[:3], muted.getRgbF()[:3])))
                box(left + column * 10, 31, 10, 46, color)
        text(117, 41, 26, '→')
    elif kind == 'ai_mode':
        dark = QColor('#202020')
        light = QColor('#E0E0E0')
        box(37, 10, 36, 32, light)
        box(95, 10, 36, 32, dark)
        text(143, 15, 88, '60% / 40%')
        box(62, 57, 36, 32, QColor('#939393'))
        box(162, 57, 36, 32, light)
        text(42, 88, 76, '①')
        text(142, 88, 76, '②')
    elif kind == 'levels':
        for x, sides in ((28, (72, 48, 30)), (158, (72, 48, 30, 18, 10))):
            for side in sides:
                outline(x + (72 - side) // 2, 11 + (72 - side) // 2, side, side, accent)
            text(x, 84, 72, str(len(sides)))
    painter.end()
    return image


def _show_help(button, label, kind, description_keys):
    from dialogs.help import HelpDialog

    owner = button.window()
    while owner.parentWidget() is not None:
        owner = owner.parentWidget().window()
    theme = getattr(owner, 'ui_theme', 'dark')
    description = ''.join(f'<p>{escape(trans.t(key))}</p>' for key in description_keys)
    dialog = HelpDialog(label.text(), '', button.window())
    browser = dialog.text_browser
    image_url = QUrl('parameter-help.png')
    browser.document().addResource(QTextDocument.ResourceType.ImageResource.value, image_url,
                                   _illustration(kind, theme))
    browser.setHtml(f'<p align="center"><img src="{image_url.toString()}" alt="{escape(label.text())}"></p>'
                    f'{description}')
    dialog.exec()
    dialog.deleteLater()


def set_help_caption(button, label):
    name = f"{trans.t('menu_help')} · {label}"
    button.setAccessibleName(name)
    button.setToolTip(name)


def create_help_button(parent, label, kind, *description_keys, size=26):
    """Create a keyboard-accessible help button bound to *label* and *kind*."""
    button = QPushButton('?', parent)
    button.setObjectName('helpButton')
    button.setProperty('parameterHelp', kind)
    button.setFixedSize(size, size)
    button.setStyleSheet(help_button_style())
    caption = label.text() if hasattr(label, 'text') else str(label)
    set_help_caption(button, caption)
    button.clicked.connect(lambda: _show_help(button, label, kind, description_keys))
    return button


def add_help_row(form, label, editor, kind, *description_keys):
    """Put a keyboard-accessible help button beside an existing form field."""
    if isinstance(label, str):
        label = QLabel(label)
        label.setBuddy(editor)
    button = create_help_button(form.parentWidget(), label, kind, *description_keys)
    field = QHBoxLayout()
    field.setSpacing(6)
    field.addWidget(editor, 1)
    field.addWidget(button)
    form.addRow(label, field)
    return field, button
