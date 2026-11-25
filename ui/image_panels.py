from dataclasses import dataclass

from PyQt6.QtCore import QRect, QSize, QTimer, Qt
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QWidget,
    QPushButton,
)

from widgets.gallery_showcase import GalleryShowcaseWidget
from widgets.magnifier_label import MagnifierLabel
from widgets.sharpness_curve import SharpnessCurveWidget
from widgets.wipe_compare import WipeCompareWidget
from locales import trans
from ui.right_panel import ElidingLabel
from utils.design_tokens import RADIUS, SPACE, WEIGHT, palette


def _panel_qss(theme: str) -> str:
    """Title bar, canvas and control bar for one image panel."""
    t = palette(theme)
    return f"""
QWidget#panelRoot {{ background-color: {t['bg']}; }}
QLabel#panelTitle {{
    background-color: {t['surface']};
    color: {t['textMuted']};
    border-bottom: 1px solid {t['border']};
    font-size: 9pt;
    font-weight: {WEIGHT['semibold']};
    padding-left: {SPACE['md']}px;
}}
QLabel#panelTitleText {{
    background: transparent;
    color: {t['textMuted']};
    border: none;
    font-size: 9pt;
    font-weight: {WEIGHT['semibold']};
}}
QLabel#panelCanvas, QLabel#panelCanvasHint, QWidget#livePreviewPage {{
    background-color: {t['canvas']};
    color: {t['canvasText']};
}}
QWidget#panelTitleBar {{
    background-color: {t['surface']};
    border-bottom: 1px solid {t['border']};
}}
QWidget#panelControlBar {{
    background-color: {t['surface']};
    border-top: 1px solid {t['border']};
}}
QLabel#livePreviewStatus {{
    background-color: {t['surface']};
    color: {t['textMuted']};
    border-top: 1px solid {t['border']};
    padding: 4px {SPACE['md']}px;
}}
QPushButton#panelToggle {{
    background-color: transparent;
    color: {t['textMuted']};
    border: 1px solid transparent;
    border-radius: {RADIUS['sm']}px;
    padding: 2px {SPACE['sm']}px;
}}
QPushButton#panelToggle:hover {{ background-color: {t['surfaceHover']}; }}
QPushButton#panelToggle:checked {{
    background-color: {t['accentSoft']};
    color: {t['accentSoftText']};
    border-color: {t['accentSoftText']};
}}
"""


def apply_panel_theme(widget, theme: str) -> None:
    """Re-apply the panel stylesheet after a theme switch."""
    widget.setStyleSheet(_panel_qss(theme))


@dataclass
class SourcePanel:
    widget: QWidget
    title: QLabel
    image_label: MagnifierLabel
    control_bar: QWidget
    slider: QSlider
    info_label: QLabel
    roi_btn: QPushButton
    sharpness_curve: "SharpnessCurveWidget"


@dataclass
class ResultPanel:
    widget: QWidget
    title: QLabel
    image_label: MagnifierLabel
    control_bar: QWidget
    slider: QSlider
    info_label: QLabel
    view_stack: QStackedWidget
    live_preview_page: "_LiveFusionPreviewPage"
    wipe_widget: WipeCompareWidget
    wipe_bar: QWidget
    btn_wipe: QPushButton
    chk_wipe_follow_left: QCheckBox
    slider_wipe_left: QSlider
    lbl_wipe_left_index: QLabel
    chk_wipe_follow_right: QCheckBox
    slider_wipe_right: QSlider
    lbl_wipe_right_index: QLabel
    gallery_showcase_page: GalleryShowcaseWidget = None


class _LiveFusionPreviewPage(QWidget):
    """Fit the current focus or reconstruction map in the output panel."""

    def __init__(self):
        super().__init__()
        self.setObjectName('livePreviewPage')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self._image = QImage()
        self._gray = None
        self._progress = None
        self.status = QLabel()
        self.status.setObjectName('livePreviewStatus')
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addStretch(1)
        layout.addWidget(self.status)
        trans.languageChanged.connect(self._update_status)

    def show_frame(self, gray, width, height, done, total, blending):
        self._image = QImage(gray, width, height, width, QImage.Format.Format_Grayscale8)
        self._gray = gray  # The QImage borrows this snapshot for painting.
        self._progress = done, total, blending
        self._update_status()
        self.update()

    def clear(self):
        self._image = QImage()
        self._gray = None
        self._progress = None
        self.status.clear()
        self.update()

    def _update_status(self):
        if self._progress is not None:
            done, total, blending = self._progress
            key = 'render_live_blend' if blending else 'render_live_preview'
            self.status.setText(trans.t(key).format(done=done, total=total))

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        if self._image.isNull():
            return
        bounds = self.rect().adjusted(0, 0, 0, -self.status.height())
        target = QRect(bounds)
        target.setSize(self._image.size().scaled(bounds.size(), Qt.AspectRatioMode.KeepAspectRatio))
        target.moveCenter(bounds.center())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self._image)


class _ResultZoomBar(QWidget):
    """Zoom controls whose signal connections follow the widget lifetime."""

    def __init__(self, image_label):
        super().__init__()
        self._image_label = image_label
        self.setObjectName("panelControlBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.addStretch()
        self._zoom_out = QPushButton("−")
        self._zoom_in = QPushButton("+")
        self._zoom_level = QLabel()
        self._zoom_level.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zoom_level.setMinimumWidth(55)
        layout.addWidget(self._zoom_out)
        layout.addWidget(self._zoom_level)
        layout.addWidget(self._zoom_in)
        self._zoom_out.clicked.connect(lambda: image_label.zoom_view(False))
        self._zoom_in.clicked.connect(lambda: image_label.zoom_view(True))
        image_label.viewStateChanged.connect(self._update_controls)
        trans.languageChanged.connect(self._translate_controls)
        self._translate_controls()

    def _update_controls(self):
        zoom, minimum, maximum, available = self._image_label.view_zoom_state()
        self._zoom_out.setEnabled(available and zoom > minimum)
        self._zoom_in.setEnabled(available and zoom < maximum)
        self._zoom_level.setText(f"{zoom:.2f}×" if available else "—")
        self._zoom_level.setAccessibleName(
            trans.t('result_zoom_level').format(zoom=f"{zoom:.2f}×")
            if available else trans.t('result_zoom_empty'))

    def _translate_controls(self):
        for button, key in ((self._zoom_out, 'result_zoom_out'),
                            (self._zoom_in, 'result_zoom_in')):
            button.setToolTip(trans.t(key))
            button.setAccessibleName(trans.t(key))
        self._update_controls()


class _WipeFollowCheckBox(QCheckBox):
    def __init__(self, text):
        super().__init__()
        self.setText(text)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def minimumSizeHint(self):
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.text = ""
        return self.style().sizeFromContents(
            QStyle.ContentsType.CT_CheckBox, option, QSize(0, self.fontMetrics().height()), self)

    def paintEvent(self, event):
        option = QStyleOptionButton()
        self.initStyleOption(option)
        contents = self.style().subElementRect(QStyle.SubElement.SE_CheckBoxContents, option, self)
        option.text = self.fontMetrics().elidedText(
            option.text, Qt.TextElideMode.ElideRight, max(0, contents.width()),
            Qt.TextFlag.TextShowMnemonic.value)
        with QStylePainter(self) as painter:
            painter.drawControl(QStyle.ControlElement.CE_CheckBox, option)


class _WipeBar(QWidget):
    """Two-row A/B wipe controls whose signal lifetime follows the widget."""

    def __init__(self, label_a, label_b, container):
        super().__init__()
        self._label_a = label_a
        self._label_b = label_b
        self._container = container
        self.setObjectName("panelControlBar")
        trans.languageChanged.connect(self.retranslate)

    def retranslate(self):
        self._label_a.setText(trans.t('wipe_side_a'))
        self._label_b.setText(trans.t('wipe_side_b'))
        win = self._container.window()
        if getattr(win, 'wipe_active', False) and hasattr(win, '_update_wipe_images'):
            QTimer.singleShot(0, win._update_wipe_images)


def create_source_panel() -> SourcePanel:
    container = QWidget()
    container.setMinimumWidth(200)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    container.setObjectName("panelRoot")
    title = QLabel(' ' + trans.t('panel_source_stack'))
    title.setObjectName("panelTitle")
    title.setFixedHeight(26)
    title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    image_label = MagnifierLabel("Drag images here to add source files\nor use the image menu")
    image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    # The hint is a manually broken two-liner, but a translation can make
    # either line much longer than the panel (German needs 1014 px for 566 px
    # of canvas). With wrap on, Qt re-breaks the line instead of clipping it.
    image_label.setWordWrap(True)
    image_label.setObjectName("panelCanvasHint")
    image_label.setAcceptDrops(True)
    image_label.setScaledContents(False)
    image_label.setMinimumSize(100, 100)
    image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    control_bar = QWidget()
    control_bar.setObjectName("panelControlBar")
    # 42 px squeezed the navigation slider down to a 2 px sliver: the groove
    # plus the handle's negative margin need ~26 px of its own once the 14 px
    # handle hangs over a 4 px groove. 52 px leaves that space and still fits
    # the 26 px ROI chip.
    control_bar.setFixedHeight(52)

    control_layout = QHBoxLayout(control_bar)
    control_layout.setContentsMargins(12, 0, 12, 0)
    control_layout.setSpacing(8)

    info_label = QLabel("-- / --")
    info_label.setObjectName("mutedLabel")
    info_label.setMinimumWidth(64)
    info_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
    info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 0)
    slider.setEnabled(False)

    roi_btn = QPushButton(trans.t('btn_roi'))
    roi_btn.setObjectName("panelToggle")
    roi_btn.setCheckable(True)
    roi_btn.setToolTip(trans.t('btn_roi_hint'))

    control_layout.addWidget(info_label)
    control_layout.addWidget(slider)
    control_layout.addWidget(roi_btn)

    control_bar.setVisible(False)

    # Per-frame sharpness curve above the navigation slider
    sharpness_curve = SharpnessCurveWidget()
    sharpness_curve.setVisible(False)

    layout.addWidget(title)
    layout.addWidget(image_label, 1)
    layout.addWidget(sharpness_curve)
    layout.addWidget(control_bar)

    return SourcePanel(
        widget=container,
        title=title,
        image_label=image_label,
        control_bar=control_bar,
        slider=slider,
        info_label=info_label,
        roi_btn=roi_btn,
        sharpness_curve=sharpness_curve,
    )


def create_result_panel() -> ResultPanel:
    container = QWidget()
    container.setMinimumWidth(200)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    # Title bar carries the always-visible Wipe toggle: the bottom control
    # bar is hidden most of the time, which used to hide the button with it
    container.setObjectName("panelRoot")
    title_bar = QWidget()
    title_bar.setObjectName("panelTitleBar")
    title_bar.setMinimumHeight(26)
    title_layout = QHBoxLayout(title_bar)
    title_layout.setContentsMargins(SPACE['md'], 0, 6, 0)
    title_layout.setSpacing(6)

    title = QLabel(trans.t('panel_output'))
    title.setObjectName("panelTitleText")
    title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    wipe_btn = QPushButton(trans.t('btn_wipe'))
    wipe_btn.setObjectName("panelToggle")
    wipe_btn.setCheckable(True)
    wipe_btn.setToolTip(trans.t('wipe_hint'))
    title_layout.addWidget(title)
    title_layout.addStretch()
    title_layout.addWidget(wipe_btn)

    image_label = MagnifierLabel()
    image_label.setObjectName("panelCanvas")
    image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    # Same reason as the source panel: "Press Start Render …" in a long
    # translation must re-break rather than be clipped.
    image_label.setWordWrap(True)
    image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    control_bar = QWidget()
    control_bar.setObjectName("panelControlBar")
    # 42 px squeezed the navigation slider down to a 2 px sliver: the groove
    # plus the handle's negative margin need ~26 px of its own once the 14 px
    # handle hangs over a 4 px groove. 52 px leaves that space and still fits
    # the 26 px ROI chip.
    control_bar.setFixedHeight(52)

    control_layout = QHBoxLayout(control_bar)
    control_layout.setContentsMargins(12, 0, 12, 0)
    control_layout.setSpacing(8)

    info_label = QLabel("-- / --")
    info_label.setObjectName("mutedLabel")
    info_label.setMinimumWidth(64)
    info_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
    info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 0)
    slider.setEnabled(False)

    control_layout.addWidget(info_label)
    control_layout.addWidget(slider)

    control_bar.setVisible(False)

    # Wipe compare page: the compare widget plus its A/B source selectors
    wipe_widget = WipeCompareWidget()

    def _make_wipe_group(label_text, follow_text):
        """One wipe side: follow checkbox + frame slider + index label."""
        side_label = QLabel(label_text)
        side_label.setObjectName("mutedLabel")
        follow_chk = _WipeFollowCheckBox(follow_text)
        follow_chk.setChecked(True)
        frame_slider = QSlider(Qt.Orientation.Horizontal)
        frame_slider.setRange(0, 0)
        frame_slider.setEnabled(False)
        frame_slider.setMinimumWidth(60)
        index_label = ElidingLabel("-")
        index_label.setObjectName("mutedLabel")
        index_label.setMinimumWidth(36)
        index_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return follow_chk, frame_slider, index_label, side_label

    chk_follow_left, slider_wipe_left, lbl_left_idx, label_a = _make_wipe_group(
        trans.t('wipe_side_a'), trans.t('wipe_follow_source'))
    chk_follow_right, slider_wipe_right, lbl_right_idx, label_b = _make_wipe_group(
        trans.t('wipe_side_b'), trans.t('wipe_follow_latest'))

    wipe_bar = _WipeBar(label_a, label_b, container)
    wipe_bar_layout = QVBoxLayout(wipe_bar)
    wipe_bar_layout.setContentsMargins(8, 4, 8, 4)
    wipe_bar_layout.setSpacing(2)

    def _add_side(side_label, follow_chk, frame_slider, index_label):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(side_label)
        row.addWidget(follow_chk)
        row.addWidget(frame_slider, 1)
        row.addWidget(index_label)
        wipe_bar_layout.addLayout(row)

    _add_side(label_a, chk_follow_left, slider_wipe_left, lbl_left_idx)
    _add_side(label_b, chk_follow_right, slider_wipe_right, lbl_right_idx)
    wipe_bar.setVisible(False)

    wipe_page = QWidget()
    wipe_page_layout = QVBoxLayout(wipe_page)
    wipe_page_layout.setContentsMargins(0, 0, 0, 0)
    wipe_page_layout.setSpacing(0)
    wipe_page_layout.addWidget(wipe_widget, 1)
    wipe_page_layout.addWidget(wipe_bar)

    zoom_bar = _ResultZoomBar(image_label)

    result_page = QWidget()
    result_page_layout = QVBoxLayout(result_page)
    result_page_layout.setContentsMargins(0, 0, 0, 0)
    result_page_layout.setSpacing(0)
    result_page_layout.addWidget(image_label, 1)
    result_page_layout.addWidget(zoom_bar)

    view_stack = QStackedWidget()
    view_stack.addWidget(result_page)  # index 0: normal result view
    view_stack.addWidget(wipe_page)    # index 1: wipe compare view
    live_preview_page = _LiveFusionPreviewPage()
    view_stack.addWidget(live_preview_page)  # index 2: legacy live preview page
    gallery_showcase_page = GalleryShowcaseWidget()
    view_stack.addWidget(gallery_showcase_page)  # index 3: OpenFocus Gallery showcase

    layout.addWidget(title_bar)
    layout.addWidget(view_stack, 1)
    layout.addWidget(control_bar)

    return ResultPanel(
        widget=container,
        title=title,
        image_label=image_label,
        control_bar=control_bar,
        slider=slider,
        info_label=info_label,
        view_stack=view_stack,
        live_preview_page=live_preview_page,
        wipe_widget=wipe_widget,
        wipe_bar=wipe_bar,
        btn_wipe=wipe_btn,
        chk_wipe_follow_left=chk_follow_left,
        slider_wipe_left=slider_wipe_left,
        lbl_wipe_left_index=lbl_left_idx,
        chk_wipe_follow_right=chk_follow_right,
        slider_wipe_right=slider_wipe_right,
        lbl_wipe_right_index=lbl_right_idx,
        gallery_showcase_page=gallery_showcase_page,
    )
