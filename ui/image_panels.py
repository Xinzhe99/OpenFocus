from dataclasses import dataclass
import math

from PyQt6.QtCore import QPointF, QRect, QSignalBlocker, QSize, QTimer, Qt
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF
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
QPushButton#zoomBarButton {{
    background-color: transparent;
    color: {t['textMuted']};
    border: 1px solid {t['border']};
    border-radius: 12px;
    min-width: 26px;
    height: 24px;
    padding: 0 7px;
    font-size: 9pt;
    font-weight: {WEIGHT['semibold']};
}}
QPushButton#zoomBarButton:hover {{
    background-color: {t['surfaceHover']};
    color: {t['text']};
    border-color: {t['borderStrong']};
}}
QPushButton#zoomBarButton:checked {{
    background-color: {t['accentSoft']};
    color: {t['accentSoftText']};
    border-color: {t['accentSoftText']};
}}
QPushButton#zoomBarButton:disabled {{
    color: {t['textFaint']};
    border-color: {t['border']};
    background-color: transparent;
}}
QPushButton#zoomReadoutButton {{
    background-color: transparent;
    color: {t['textMuted']};
    border: 1px solid transparent;
    border-radius: {RADIUS['sm']}px;
    padding: 2px 6px;
    min-width: 48px;
    font-size: 9pt;
    font-weight: {WEIGHT['medium']};
}}
QPushButton#zoomReadoutButton:hover:enabled {{
    background-color: {t['surfaceHover']};
    color: {t['text']};
}}
QPushButton#zoomReadoutButton:disabled {{
    color: {t['textFaint']};
}}
QSlider#resultZoomSlider {{
    min-height: 20px;
}}
QSlider#resultZoomSlider::groove:horizontal {{
    height: 4px;
    background-color: {t['border']};
    border-radius: 2px;
}}
QSlider#resultZoomSlider::sub-page:horizontal {{
    background-color: {t['accent']};
    border-radius: 2px;
}}
QSlider#resultZoomSlider::handle:horizontal {{
    background-color: {t['surface']};
    border: 1.5px solid {t['borderStrong']};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider#resultZoomSlider::handle:horizontal:hover {{
    border-color: {t['accent']};
}}
QSlider#resultZoomSlider::handle:horizontal:disabled {{
    background-color: {t['surface']};
    border-color: {t['border']};
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
QPushButton#panelToggle:disabled {{
    color: {t['textFaint']};
}}
"""


def apply_panel_theme(widget, theme: str) -> None:
    """Re-apply the panel stylesheet after a theme switch."""
    widget.setStyleSheet(_panel_qss(theme))
    if isinstance(widget, _ResultZoomBar):
        widget.apply_theme(theme)
    else:
        for bar in widget.findChildren(_ResultZoomBar):
            bar.apply_theme(theme)


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
    show_aligned_btn: QPushButton
    align_area_btn: QPushButton
    clear_align_area_btn: QPushButton


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
    gallery_showcase_page: object = None


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


def _draw_fit_window_pixmap(color_hex: str) -> QPixmap:
    """Render a crisp monochrome four-way fit-to-window icon for HiDPI displays."""
    pm = QPixmap(28, 28)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    col = QColor(color_hex)
    painter.setPen(QPen(col, 1.35, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.setBrush(col)
    # Center cross hairlines + four outward cardinal arrowheads on a 14x14 canvas
    c = 7.0
    painter.drawLine(QPointF(c, 3.3), QPointF(c, 10.7))
    painter.drawLine(QPointF(3.3, c), QPointF(10.7, c))
    arrows = (
        QPolygonF([QPointF(c, 1.2), QPointF(c - 2.1, 3.8), QPointF(c + 2.1, 3.8)]),
        QPolygonF([QPointF(c, 12.8), QPointF(c - 2.1, 10.2), QPointF(c + 2.1, 10.2)]),
        QPolygonF([QPointF(1.2, c), QPointF(3.8, c - 2.1), QPointF(3.8, c + 2.1)]),
        QPolygonF([QPointF(12.8, c), QPointF(10.2, c - 2.1), QPointF(10.2, c + 2.1)]),
    )
    painter.setPen(Qt.PenStyle.NoPen)
    for poly in arrows:
        painter.drawPolygon(poly)
    painter.end()
    return pm


def _make_fit_window_icon(theme: str) -> QIcon:
    """Build a multi-state monochrome icon matching Normal/Checked/Disabled panel states."""
    t = palette(theme)
    icon = QIcon()
    normal_pm = _draw_fit_window_pixmap(t["textMuted"])
    active_pm = _draw_fit_window_pixmap(t["accentSoftText"])
    disabled_pm = _draw_fit_window_pixmap(t["textFaint"])
    icon.addPixmap(normal_pm, QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(active_pm, QIcon.Mode.Normal, QIcon.State.On)
    icon.addPixmap(disabled_pm, QIcon.Mode.Disabled, QIcon.State.Off)
    icon.addPixmap(disabled_pm, QIcon.Mode.Disabled, QIcon.State.On)
    return icon


class _ResultZoomBar(QWidget):
    """Zoom controls whose signal connections follow the widget lifetime."""

    _SLIDER_MAX = 1000

    def __init__(self, image_label):
        super().__init__()
        self._image_label = image_label
        self._theme = "light"
        self.setObjectName("panelControlBar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)
        layout.addStretch()

        self._zoom_100 = QPushButton("1:1")
        self._zoom_100.setObjectName("zoomBarButton")
        self._zoom_100.setCheckable(True)
        self._zoom_100.setProperty("controlId", "result_zoom_100")

        self._zoom_fit = QPushButton()
        self._zoom_fit.setObjectName("zoomBarButton")
        self._zoom_fit.setCheckable(True)
        self._zoom_fit.setIconSize(QSize(14, 14))
        self._zoom_fit.setProperty("controlId", "result_zoom_fit")

        self._zoom_out = QPushButton("−")
        self._zoom_out.setObjectName("zoomBarButton")
        self._zoom_out.setProperty("controlId", "result_zoom_out")

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setObjectName("resultZoomSlider")
        self._zoom_slider.setRange(0, self._SLIDER_MAX)
        self._zoom_slider.setFixedWidth(136)
        self._zoom_slider.setProperty("controlId", "result_zoom_slider")

        self._zoom_in = QPushButton("+")
        self._zoom_in.setObjectName("zoomBarButton")
        self._zoom_in.setProperty("controlId", "result_zoom_in")

        self._zoom_level = QPushButton()
        self._zoom_level.setObjectName("zoomReadoutButton")
        self._zoom_level.setFlat(True)
        self._zoom_level.setProperty("controlId", "result_zoom_readout")

        layout.addWidget(self._zoom_100)
        layout.addWidget(self._zoom_fit)
        layout.addWidget(self._zoom_out)
        layout.addWidget(self._zoom_slider)
        layout.addWidget(self._zoom_in)
        layout.addWidget(self._zoom_level)
        layout.addStretch()

        self._zoom_100.clicked.connect(self._on_zoom_100)
        self._zoom_fit.clicked.connect(self._on_zoom_fit)
        self._zoom_out.clicked.connect(lambda: image_label.zoom_view(False))
        self._zoom_in.clicked.connect(lambda: image_label.zoom_view(True))
        self._zoom_slider.sliderMoved.connect(lambda v: self._apply_slider_value(v, interactive=True))
        self._zoom_slider.sliderReleased.connect(
            lambda: self._apply_slider_value(self._zoom_slider.value(), interactive=False)
        )
        self._zoom_slider.valueChanged.connect(self._on_slider_value_changed)
        self._zoom_level.clicked.connect(self._on_zoom_readout_clicked)

        image_label.viewStateChanged.connect(self._update_controls)
        trans.languageChanged.connect(self._translate_controls)
        self.apply_theme(self._theme)
        self._translate_controls()

    def apply_theme(self, theme: str) -> None:
        self._theme = theme or "light"
        self._zoom_fit.setIcon(_make_fit_window_icon(self._theme))

    @classmethod
    def _slider_to_zoom(cls, value: int, minimum: float, maximum: float) -> float:
        if minimum <= 0.0 or maximum <= minimum:
            return minimum
        ratio = max(0.0, min(1.0, value / float(cls._SLIDER_MAX)))
        return minimum * ((maximum / minimum) ** ratio)

    @classmethod
    def _zoom_to_slider(cls, zoom: float, minimum: float, maximum: float) -> int:
        if minimum <= 0.0 or maximum <= minimum or zoom <= minimum:
            return 0
        if zoom >= maximum:
            return cls._SLIDER_MAX
        ratio = math.log(zoom / minimum) / math.log(maximum / minimum)
        return max(0, min(cls._SLIDER_MAX, int(round(ratio * cls._SLIDER_MAX))))

    def _on_zoom_100(self) -> None:
        if hasattr(self._image_label, "zoom_to_actual_pixels"):
            self._image_label.zoom_to_actual_pixels()

    def _on_zoom_fit(self) -> None:
        if hasattr(self._image_label, "zoom_to_fit"):
            self._image_label.zoom_to_fit()
        elif hasattr(self._image_label, "reset_view"):
            self._image_label.reset_view()

    def _on_zoom_readout_clicked(self) -> None:
        zoom, _, _, available = self._image_label.view_zoom_state()
        if not available:
            return
        actual_pct = (
            self._image_label.actual_zoom_percent()
            if hasattr(self._image_label, "actual_zoom_percent")
            else zoom * 100.0
        )
        if abs(actual_pct - 100.0) <= 1.5:
            self._on_zoom_fit()
        else:
            self._on_zoom_100()

    def _apply_slider_value(self, value: int, *, interactive: bool) -> None:
        _, minimum, maximum, available = self._image_label.view_zoom_state()
        if not available or not hasattr(self._image_label, "set_view_zoom"):
            return
        target = self._slider_to_zoom(value, minimum, maximum)
        self._image_label.set_view_zoom(target, interactive=interactive, show_indicator=True)

    def _on_slider_value_changed(self, value: int) -> None:
        if not self._zoom_slider.isSliderDown():
            self._apply_slider_value(value, interactive=False)

    def _update_controls(self):
        zoom, minimum, maximum, available = self._image_label.view_zoom_state()
        actual_pct = (
            self._image_label.actual_zoom_percent()
            if hasattr(self._image_label, "actual_zoom_percent")
            else zoom * 100.0
        )
        self._zoom_100.setEnabled(available)
        self._zoom_fit.setEnabled(available)
        self._zoom_out.setEnabled(available and zoom > minimum)
        self._zoom_in.setEnabled(available and zoom < maximum)
        self._zoom_slider.setEnabled(available and maximum > minimum)
        self._zoom_level.setEnabled(available)

        is_100 = bool(available and abs(actual_pct - 100.0) <= 1.5)
        is_fit = bool(available and abs(zoom - 1.0) <= 0.015 and not is_100)
        with QSignalBlocker(self._zoom_100):
            self._zoom_100.setChecked(is_100)
        with QSignalBlocker(self._zoom_fit):
            self._zoom_fit.setChecked(is_fit)
        with QSignalBlocker(self._zoom_slider):
            self._zoom_slider.setValue(
                self._zoom_to_slider(zoom, minimum, maximum) if available else 0
            )

        if available:
            pct_str = f"{actual_pct:.0f}%"
            self._zoom_level.setText(pct_str)
            self._zoom_level.setToolTip(
                trans.t('result_zoom_level').format(zoom=f"{zoom:.2f}× ({pct_str})")
            )
            self._zoom_level.setAccessibleName(
                trans.t('result_zoom_level').format(zoom=f"{zoom:.2f}×")
            )
        else:
            self._zoom_level.setText("—")
            self._zoom_level.setToolTip(trans.t('result_zoom_empty'))
            self._zoom_level.setAccessibleName(trans.t('result_zoom_empty'))

    def _translate_controls(self):
        for button, key in (
            (self._zoom_100, 'result_zoom_100'),
            (self._zoom_fit, 'result_zoom_fit'),
            (self._zoom_out, 'result_zoom_out'),
            (self._zoom_slider, 'result_zoom_slider'),
            (self._zoom_in, 'result_zoom_in'),
        ):
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
    title_bar = QWidget()
    title_bar.setObjectName("panelTitleBar")
    title_bar.setMinimumHeight(26)
    title_layout = QHBoxLayout(title_bar)
    title_layout.setContentsMargins(SPACE['md'], 0, 6, 0)
    title_layout.setSpacing(6)

    title = QLabel(trans.t('panel_source_stack'))
    title.setObjectName("panelTitleText")
    title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    align_area_btn = QPushButton(trans.t('btn_align_area'))
    align_area_btn.setObjectName("panelToggle")
    align_area_btn.setCheckable(True)
    align_area_btn.setEnabled(False)
    align_area_btn.setToolTip(trans.t('btn_align_area_hint'))
    align_area_btn.setProperty("controlId", "btn_align_area")

    clear_align_area_btn = QPushButton("×")
    clear_align_area_btn.setObjectName("panelToggle")
    clear_align_area_btn.setFixedWidth(26)
    clear_align_area_btn.setStyleSheet("padding: 2px 4px;")
    clear_align_area_btn.setVisible(False)
    clear_align_area_btn.setToolTip(trans.t('btn_clear_align_area_hint'))
    clear_align_area_btn.setProperty("controlId", "btn_clear_align_area")

    show_aligned_btn = QPushButton(trans.t('btn_show_aligned'))
    show_aligned_btn.setObjectName("panelToggle")
    show_aligned_btn.setCheckable(True)
    show_aligned_btn.setEnabled(False)
    show_aligned_btn.setToolTip(trans.t('btn_show_aligned_hint'))
    show_aligned_btn.setProperty("controlId", "btn_show_aligned")

    title_layout.addWidget(title)
    title_layout.addStretch()
    title_layout.addWidget(align_area_btn)
    title_layout.addWidget(clear_align_area_btn)
    title_layout.addWidget(show_aligned_btn)

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

    layout.addWidget(title_bar)
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
        show_aligned_btn=show_aligned_btn,
        align_area_btn=align_area_btn,
        clear_align_area_btn=clear_align_area_btn,
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
    )
