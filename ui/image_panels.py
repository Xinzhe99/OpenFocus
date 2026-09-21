from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QFrame,
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
    chk_wipe_follow_left: QCheckBox
    slider_wipe_left: QSlider
    lbl_wipe_left_index: QLabel
    chk_wipe_follow_right: QCheckBox
    slider_wipe_right: QSlider
    lbl_wipe_right_index: QLabel


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

    # Title bar carries the always-visible Wipe toggle: the bottom control
    # bar is hidden most of the time, which used to hide the button with it
    title_bar = QWidget()
    title_bar.setFixedHeight(25)
    title_bar.setStyleSheet("background-color: #333; border-bottom: 1px solid #444;")
    title_layout = QHBoxLayout(title_bar)
    title_layout.setContentsMargins(6, 0, 4, 0)
    title_layout.setSpacing(4)

    title = QLabel(" Output")
    title.setStyleSheet("color: #aaa; background: transparent;")
    title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    wipe_btn = QPushButton(trans.t('btn_wipe'))
    wipe_btn.setCheckable(True)
    wipe_btn.setToolTip(trans.t('wipe_hint'))
    wipe_btn.setStyleSheet("QPushButton { background-color: #333; color: #aaa; border: 1px solid #444; border-radius: 2px; padding: 0 6px; } QPushButton:checked { background-color: #0078d7; color: white; border-color: #005a9e; }")
    title_layout.addWidget(title)
    title_layout.addStretch()
    title_layout.addWidget(wipe_btn)

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

    # Wipe compare page: the compare widget plus its A/B source selectors
    wipe_widget = WipeCompareWidget()
    wipe_widget.setStyleSheet("background-color: #181818;")

    def _make_wipe_group(label_text, follow_text):
        """One wipe side: follow checkbox + frame slider + index label."""
        side_label = QLabel(label_text)
        side_label.setStyleSheet("color: #aaa; background: transparent;")
        follow_chk = QCheckBox(follow_text)
        follow_chk.setChecked(True)
        follow_chk.setStyleSheet("color: #aaa; background: transparent;")
        frame_slider = QSlider(Qt.Orientation.Horizontal)
        frame_slider.setRange(0, 0)
        frame_slider.setEnabled(False)
        frame_slider.setMinimumWidth(60)
        index_label = QLabel("-")
        index_label.setMinimumWidth(52)
        index_label.setStyleSheet("color: #aaa; background: transparent;")
        return follow_chk, frame_slider, index_label, side_label

    chk_follow_left, slider_wipe_left, lbl_left_idx, label_a = _make_wipe_group(
        trans.t('wipe_side_a'), trans.t('wipe_follow_source'))
    chk_follow_right, slider_wipe_right, lbl_right_idx, label_b = _make_wipe_group(
        trans.t('wipe_side_b'), trans.t('wipe_follow_latest'))

    wipe_bar = QWidget()
    wipe_bar.setFixedHeight(40)
    wipe_bar.setStyleSheet("background-color: #2a2a2a; border-top: 1px solid #444;")
    wipe_bar_layout = QHBoxLayout(wipe_bar)
    wipe_bar_layout.setContentsMargins(8, 2, 8, 2)
    wipe_bar_layout.setSpacing(6)

    def _add_side(layout, side_label, follow_chk, frame_slider, index_label):
        layout.addWidget(side_label)
        layout.addWidget(follow_chk)
        layout.addWidget(frame_slider, 1)
        layout.addWidget(index_label)

    _add_side(wipe_bar_layout, label_a, chk_follow_left, slider_wipe_left, lbl_left_idx)

    divider = QFrame()
    divider.setFrameShape(QFrame.Shape.VLine)
    divider.setStyleSheet("color: #444; background: transparent;")
    wipe_bar_layout.addWidget(divider)

    _add_side(wipe_bar_layout, label_b, chk_follow_right, slider_wipe_right, lbl_right_idx)
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

    layout.addWidget(title_bar)
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
        chk_wipe_follow_left=chk_follow_left,
        slider_wipe_left=slider_wipe_left,
        lbl_wipe_left_index=lbl_left_idx,
        chk_wipe_follow_right=chk_follow_right,
        slider_wipe_right=slider_wipe_right,
        lbl_wipe_right_index=lbl_right_idx,
    )
