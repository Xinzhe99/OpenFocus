"""Illustrated help opened from the setting it describes."""
from html import escape

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QTextDocument
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton

from locales import trans
from ui.context_help import register_help_target
from ui.context_help_diagrams import render_help_diagram
from ui.styles import help_button_style


def _illustration(kind, theme):
    return render_help_diagram(kind, theme)


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
    register_help_target(button, kind)
    if hasattr(label, 'setProperty'):
        register_help_target(label, kind)
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
    register_help_target(label, kind)
    register_help_target(editor, kind)
    button = create_help_button(form.parentWidget(), label, kind, *description_keys)
    field = QHBoxLayout()
    field.setSpacing(6)
    field.addWidget(editor, 1)
    field.addWidget(button)
    form.addRow(label, field)
    return field, button
