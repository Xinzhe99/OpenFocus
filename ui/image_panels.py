from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from widgets.magnifier_label import MagnifierLabel


@dataclass
class SourcePanel:
    widget: QWidget
    image_label: MagnifierLabel
    control_bar: QWidget
    slider: QSlider
    info_label: QLabel


@dataclass
class ResultPanel:
    widget: QWidget
    image_label: MagnifierLabel
    control_bar: QWidget
    slider: QSlider
    info_label: QLabel


def create_source_panel() -> SourcePanel:
    container = QWidget()
    container.setMinimumWidth(200)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    title = QLabel(" Source Stack")
    title.setFixedHeight(25)
    title.setStyleSheet("background-color: #333; color: #aaa; border-bottom: 1px solid #444;")
    title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    image_label = MagnifierLabel("Drag images here to add source files\nor use the image menu")
    image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    image_label.setFont(QFont("Microsoft YaHei", 16))
    image_label.setStyleSheet("background-color: #222; color: #666;")
    image_label.setAcceptDrops(True)
    image_label.setScaledContents(False)
    image_label.setMinimumSize(100, 100)
    image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    control_bar = QWidget()
    control_bar.setFixedHeight(40)
    control_bar.setStyleSheet("background-color: #2a2a2a; border-top: 1px solid #444;")

    control_layout = QHBoxLayout(control_bar)
    control_layout.setContentsMargins(10, 0, 10, 0)
    control_layout.setSpacing(5)

    info_label = QLabel("-- / --")
    info_label.setFixedWidth(60)
    info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 0)
    slider.setEnabled(False)

    control_layout.addWidget(info_label)
    control_layout.addWidget(slider)

    control_bar.setVisible(False)

    layout.addWidget(title)
    layout.addWidget(image_label, 1)
    layout.addWidget(control_bar)

    return SourcePanel(
        widget=container,
        image_label=image_label,
        control_bar=control_bar,
        slider=slider,
        info_label=info_label,
    )


def create_result_panel() -> ResultPanel:
    container = QWidget()
    container.setMinimumWidth(200)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    title = QLabel(" Output")
    title.setFixedHeight(25)
    title.setStyleSheet("background-color: #333; color: #aaa; border-bottom: 1px solid #444;")
    title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    image_label = MagnifierLabel()
    image_label.setStyleSheet("background-color: #222;")
    image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    control_bar = QWidget()
    control_bar.setFixedHeight(40)
    control_bar.setStyleSheet("background-color: #2a2a2a; border-top: 1px solid #444;")

    control_layout = QHBoxLayout(control_bar)
    control_layout.setContentsMargins(10, 0, 10, 0)
    control_layout.setSpacing(5)

    info_label = QLabel("-- / --")
    info_label.setFixedWidth(60)
    info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 0)
    slider.setEnabled(False)

    control_layout.addWidget(info_label)
    control_layout.addWidget(slider)

    control_bar.setVisible(False)

    layout.addWidget(title)
    layout.addWidget(image_label, 1)
    layout.addWidget(control_bar)

    return ResultPanel(
        widget=container,
        image_label=image_label,
        control_bar=control_bar,
        slider=slider,
        info_label=info_label,
    )
