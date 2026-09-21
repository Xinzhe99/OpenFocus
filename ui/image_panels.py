from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QPushButton,
)

from widgets.magnifier_label import MagnifierLabel
from widgets.wipe_compare import WipeCompareWidget
from locales import trans

WIPE_COMBO_STYLE = (
    "QComboBox { background-color: #333; color: #aaa; border: 1px solid #444;"
    " border-radius: 2px; padding: 0 4px; }"
    " QComboBox QAbstractItemView { background-color: #2a2a2a; color: #ddd;"
    " selection-background-color: #0078d7; }"
)


@dataclass
class SourcePanel:
    widget: QWidget
    image_label: MagnifierLabel
    control_bar: QWidget
    slider: QSlider
    info_label: QLabel
    roi_btn: QPushButton


@dataclass
class ResultPanel:
    widget: QWidget
    image_label: MagnifierLabel
    control_bar: QWidget
    slider: QSlider
    info_label: QLabel
    view_stack: QStackedWidget
    wipe_widget: WipeCompareWidget
    wipe_bar: QWidget
    btn_wipe: QPushButton
    combo_wipe_left: QComboBox
    combo_wipe_right: QComboBox


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

    roi_btn = QPushButton(trans.t('btn_roi'))
    roi_btn.setCheckable(True)
    # roi_btn.setFixedWidth(40) # Allow auto-width for longer text
    roi_btn.setToolTip("Select Region of Interest to preview")
    roi_btn.setStyleSheet("QPushButton { background-color: #333; color: #aaa; border: 1px solid #444; border-radius: 2px; padding: 0 5px; } QPushButton:checked { background-color: #0078d7; color: white; border-color: #005a9e; }")

    control_layout.addWidget(info_label)
    control_layout.addWidget(slider)
    control_layout.addWidget(roi_btn)

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
        roi_btn=roi_btn,
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

    wipe_btn = QPushButton(trans.t('btn_wipe'))
    wipe_btn.setCheckable(True)
    wipe_btn.setToolTip(trans.t('wipe_hint'))
    wipe_btn.setStyleSheet("QPushButton { background-color: #333; color: #aaa; border: 1px solid #444; border-radius: 2px; padding: 0 5px; } QPushButton:checked { background-color: #0078d7; color: white; border-color: #005a9e; }")
    control_layout.addWidget(wipe_btn)

    control_bar.setVisible(False)

    # Wipe compare page: the compare widget plus its A/B source selectors
    wipe_widget = WipeCompareWidget()
    wipe_widget.setStyleSheet("background-color: #181818;")

    combo_wipe_left = QComboBox()
    combo_wipe_right = QComboBox()
    for combo in (combo_wipe_left, combo_wipe_right):
        combo.setStyleSheet(WIPE_COMBO_STYLE)
        combo.setMinimumWidth(140)

    wipe_bar = QWidget()
    wipe_bar.setFixedHeight(36)
    wipe_bar.setStyleSheet("background-color: #2a2a2a; border-top: 1px solid #444;")
    wipe_bar_layout = QHBoxLayout(wipe_bar)
    wipe_bar_layout.setContentsMargins(10, 2, 10, 2)
    wipe_bar_layout.setSpacing(6)
    label_a = QLabel(trans.t('wipe_side_a'))
    label_b = QLabel(trans.t('wipe_side_b'))
    for label in (label_a, label_b):
        label.setStyleSheet("color: #aaa; background: transparent;")
    wipe_bar_layout.addWidget(label_a)
    wipe_bar_layout.addWidget(combo_wipe_left)
    wipe_bar_layout.addStretch()
    wipe_bar_layout.addWidget(label_b)
    wipe_bar_layout.addWidget(combo_wipe_right)
    wipe_bar.setVisible(False)

    wipe_page = QWidget()
    wipe_page_layout = QVBoxLayout(wipe_page)
    wipe_page_layout.setContentsMargins(0, 0, 0, 0)
    wipe_page_layout.setSpacing(0)
    wipe_page_layout.addWidget(wipe_widget, 1)
    wipe_page_layout.addWidget(wipe_bar)

    view_stack = QStackedWidget()
    view_stack.addWidget(image_label)  # index 0: normal result view
    view_stack.addWidget(wipe_page)    # index 1: wipe compare view

    layout.addWidget(title)
    layout.addWidget(view_stack, 1)
    layout.addWidget(control_bar)

    return ResultPanel(
        widget=container,
        image_label=image_label,
        control_bar=control_bar,
        slider=slider,
        info_label=info_label,
        view_stack=view_stack,
        wipe_widget=wipe_widget,
        wipe_bar=wipe_bar,
        btn_wipe=wipe_btn,
        combo_wipe_left=combo_wipe_left,
        combo_wipe_right=combo_wipe_right,
    )
