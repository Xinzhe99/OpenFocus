import os
import sys
from typing import Optional

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtGui import QPixmap, QImage

# Import MESSAGE_BOX_STYLE from sibling module to avoid circular imports
import importlib.util
_styles_spec = importlib.util.spec_from_file_location("styles_module", os.path.join(os.path.dirname(__file__), "..", "ui", "styles.py"))
_styles_module = importlib.util.module_from_spec(_styles_spec)
_styles_spec.loader.exec_module(_styles_module)
MESSAGE_BOX_STYLE = _styles_module.MESSAGE_BOX_STYLE


def resource_path(*relative_parts: str) -> str:
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    while os.path.basename(base_dir) != 'OpenFocus' and os.path.dirname(base_dir) != base_dir:
        base_dir = os.path.dirname(base_dir)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base_dir, "..", *relative_parts))


def show_message_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
    icon: QMessageBox.Icon = QMessageBox.Icon.Information,
) -> None:
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    if informative_text:
        msg_box.setInformativeText(informative_text)
    msg_box.setIcon(icon)
    msg_box.setStyleSheet(MESSAGE_BOX_STYLE)
    msg_box.exec()


def show_warning_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
) -> None:
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Warning)


def show_error_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
) -> None:
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Critical)


def show_success_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
) -> None:
    show_message_box(parent, title, text, informative_text, QMessageBox.Icon.Information)


def show_custom_message_box(
    parent: Optional[QMessageBox],
    title: str,
    text: str,
    informative_text: str = "",
    icon: QMessageBox.Icon = QMessageBox.Icon.Information,
    style_sheet: str = MESSAGE_BOX_STYLE,
) -> None:
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    if informative_text:
        msg_box.setInformativeText(informative_text)
    msg_box.setIcon(icon)
    msg_box.setStyleSheet(style_sheet)
    msg_box.exec()
