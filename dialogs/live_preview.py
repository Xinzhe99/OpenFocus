"""Standalone Interactive Live Color Preview window for OpenFocus A/B/C.

Provides real-time parameter tuning (Radius, Smoothing, Feather), instant
aligned/unaligned source stack scrubbing, live Depth Map Histogram with
dual-handle Substack Selection, and non-blocking latest-wins background updates.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from PyQt6.QtCore import (
    QEvent,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSignalBlocker,
    QSize,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.algorithms import get_fusion, resolve_fusion_parameters
from locales import trans
from ui.styles import CURRENT_THEME, build_stylesheet
from utils.design_tokens import FONT, RADIUS, SPACE, WEIGHT, palette
from utils.image_utils import cv2_to_pixmap
from utils.platform_utils import configure_native_worker_stack
from widgets.stepper import StepperSpinBox

# Supported live-preview methods and their display translation keys.
LIVE_PREVIEW_METHODS: tuple[tuple[str, str, str], ...] = (
    ("openfocus_a", "live_preview_method_a", "OpenFocus - A"),
    ("openfocus_b", "live_preview_method_b", "OpenFocus - B"),
    ("openfocus_c", "live_preview_method_c", "OpenFocus - C"),
)

# Preview size presets matching core.live_preview proxy sizes.
PREVIEW_SIZE_PRESETS: tuple[tuple[str, str], ...] = (
    ("small", "live_preview_size_small"),
    ("medium", "live_preview_size_medium"),
    ("large", "live_preview_size_large"),
)

_METHOD_SHORT_NAMES: dict[str, str] = {
    "openfocus_a": "OpenFocus - A",
    "openfocus_b": "OpenFocus - B",
    "openfocus_c": "OpenFocus - C",
}

# Keep any worker thread alive at module scope if native code takes longer than
# our shutdown timeout, preventing Qt from aborting on QThread destruction.
_ORPHAN_PREVIEW_WORKERS: set[QThread] = set()


def _qcolor_with_alpha(hex_or_css: str, alpha: int) -> QColor:
    c = QColor(hex_or_css)
    c.setAlpha(max(0, min(255, int(alpha))))
    return c


class LivePreviewCanvas(QWidget):
    """High-DPI interactive image canvas with cursor-anchored zoom, pan, split-curtain compare, and zero flicker."""

    zoomChanged = pyqtSignal(float, bool)  # (zoom_ratio, is_fit)
    splitPositionChanged = pyqtSignal(float)  # normalized [0.02, 0.98]

    _SPLIT_HIT_PX = 12.0

    def __init__(self, parent: Optional[QWidget] = None, *, theme: str = "light"):
        super().__init__(parent)
        self._theme = theme
        self._pixmap: Optional[QPixmap] = None
        self._source_pixmap: Optional[QPixmap] = None
        self._image_size: tuple[int, int] = (0, 0)  # Logical proxy size (w, h)
        self._zoom: float = 1.0
        self._is_fit: bool = True
        self._pan: QPointF = QPointF(0.0, 0.0)
        self._drag_origin: Optional[QPointF] = None
        self._drag_start_pan: QPointF = QPointF(0.0, 0.0)
        self._split_compare: bool = False
        self._split_pos: float = 0.5
        self._dragging_split: bool = False
        self._split_left_label: str = ""
        self._split_right_label: str = ""
        self._overlay_badge: str = ""
        self._placeholder_text: str = ""
        self.setMinimumSize(360, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder_text = text or ""
        if self._pixmap is None:
            self.update()

    def set_overlay_badge(self, text: str) -> None:
        text = text or ""
        if text != self._overlay_badge:
            self._overlay_badge = text
            self.update()

    def set_split_compare(self, enabled: bool) -> None:
        flag = bool(enabled)
        if flag != self._split_compare:
            self._split_compare = flag
            self._dragging_split = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.update()

    def is_split_compare(self) -> bool:
        return self._split_compare

    def split_position(self) -> float:
        return self._split_pos

    def set_split_position(self, pos: float) -> None:
        clamped = max(0.02, min(0.98, float(pos)))
        if abs(clamped - self._split_pos) > 1e-6:
            self._split_pos = clamped
            self.splitPositionChanged.emit(self._split_pos)
            self.update()

    def set_source_pixmap(
        self,
        pixmap: Optional[QPixmap],
        *,
        left_label: str = "",
        right_label: str = "",
    ) -> None:
        self._source_pixmap = pixmap
        self._split_left_label = left_label or ""
        self._split_right_label = right_label or ""
        if self._split_compare:
            self.update()

    def set_pixmap(
        self,
        pixmap: Optional[QPixmap],
        *,
        reset_view: bool = False,
        logical_size: Optional[tuple[int, int]] = None,
    ) -> None:
        prev_size = self._image_size
        self._pixmap = pixmap
        if pixmap is not None and not pixmap.isNull():
            if logical_size is not None and logical_size[0] > 0 and logical_size[1] > 0:
                self._image_size = (int(logical_size[0]), int(logical_size[1]))
            else:
                self._image_size = (pixmap.width(), pixmap.height())
            if reset_view or prev_size == (0, 0):
                self._is_fit = True
                self._pan = QPointF(0.0, 0.0)
                self._recompute_fit_zoom()
            elif prev_size != self._image_size:
                # Preserve visual framing across Small / Medium / Large size switches.
                if self._is_fit:
                    self._recompute_fit_zoom()
                else:
                    scale_ratio = prev_size[0] / max(1, self._image_size[0])
                    self._zoom = max(0.1, min(12.0, self._zoom * scale_ratio))
        else:
            self._image_size = (0, 0)
        self.update()

    def has_image(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull() and min(self._image_size) > 0

    def zoom_factor(self) -> float:
        return self._zoom

    def is_fit_mode(self) -> bool:
        return self._is_fit

    def _fit_zoom(self) -> float:
        if not self.has_image():
            return 1.0
        w, h = self._image_size
        avail_w = max(1, self.width() - 24)
        avail_h = max(1, self.height() - 24)
        return min(avail_w / max(1, w), avail_h / max(1, h))

    def _recompute_fit_zoom(self) -> None:
        if self.has_image():
            self._zoom = self._fit_zoom()
            self.zoomChanged.emit(self._zoom, self._is_fit)

    def fit_to_window(self) -> None:
        if not self.has_image():
            return
        self._is_fit = True
        self._pan = QPointF(0.0, 0.0)
        self._zoom = self._fit_zoom()
        self.zoomChanged.emit(self._zoom, True)
        self.update()

    def set_actual_size(self) -> None:
        if not self.has_image():
            return
        self._is_fit = False
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.zoomChanged.emit(self._zoom, False)
        self.update()

    def _image_rect(self) -> QRectF:
        w, h = self._image_size
        draw_w = w * self._zoom
        draw_h = h * self._zoom
        cx = self.width() * 0.5 + self._pan.x()
        cy = self.height() * 0.5 + self._pan.y()
        return QRectF(cx - draw_w * 0.5, cy - draw_h * 0.5, draw_w, draw_h)

    def image_point_at(self, widget_pos: QPointF) -> QPointF:
        """Map a widget coordinate to logical image pixel coordinates (used for zoom-to-cursor)."""
        rect = self._image_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return QPointF(0.0, 0.0)
        ix = (widget_pos.x() - rect.left()) / self._zoom
        iy = (widget_pos.y() - rect.top()) / self._zoom
        return QPointF(ix, iy)

    def _split_line_x(self, target: QRectF) -> float:
        return target.left() + self._split_pos * target.width()

    def _hits_split_divider(self, pos: QPointF) -> bool:
        if not (self._split_compare and self.has_image() and self._source_pixmap is not None):
            return False
        target = self._image_rect()
        if target.width() <= 0 or target.height() <= 0:
            return False
        sx = self._split_line_x(target)
        return (
            abs(pos.x() - sx) <= self._SPLIT_HIT_PX
            and (target.top() - 8.0) <= pos.y() <= (target.bottom() + 8.0)
        )

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if not self.has_image():
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            pixel_dy = event.pixelDelta().y()
            if pixel_dy == 0:
                event.ignore()
                return
            steps = pixel_dy / 48.0
        else:
            steps = delta / 120.0
        factor = 1.15 ** steps
        old_zoom = self._zoom
        new_zoom = max(0.1, min(12.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            event.accept()
            return
        # Smooth cursor-anchored zoom: keep the exact image pixel under `cursor` stationary.
        cursor = event.position()
        center = QPointF(self.width() * 0.5, self.height() * 0.5)
        img_offset = cursor - center - self._pan
        ratio = new_zoom / old_zoom
        self._pan = cursor - center - img_offset * ratio
        self._zoom = new_zoom
        self._is_fit = False
        self.zoomChanged.emit(self._zoom, False)
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.has_image():
            if self._hits_split_divider(event.position()):
                self._dragging_split = True
                self.setCursor(Qt.CursorShape.SplitHCursor)
                event.accept()
                return
            self._drag_origin = event.position()
            self._drag_start_pan = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._dragging_split and self.has_image():
            target = self._image_rect()
            if target.width() > 1.0:
                ratio = (event.position().x() - target.left()) / target.width()
                self.set_split_position(ratio)
            event.accept()
            return
        if self._drag_origin is not None:
            delta = event.position() - self._drag_origin
            self._pan = self._drag_start_pan + delta
            self._is_fit = False
            self.zoomChanged.emit(self._zoom, False)
            self.update()
            event.accept()
            return
        if self._hits_split_divider(event.position()):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        elif self.has_image():
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_split:
            self._dragging_split = False
            if self._hits_split_divider(event.position()):
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_origin is not None:
            self._drag_origin = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.has_image():
            if self._hits_split_divider(event.position()):
                self.set_split_position(0.5)
                event.accept()
                return
            self.fit_to_window()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._is_fit and self.has_image():
            self._recompute_fit_zoom()

    def _draw_pill_label(self, painter: QPainter, text: str, x: float, y: float, bg: QColor, fg: QColor) -> None:
        if not text:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = painter.font()
        font.setPointSize(FONT["caption"] + 1)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        pill_w = fm.horizontalAdvance(text) + 18
        pill_h = max(22, fm.height() + 6)
        pill = QRectF(x, y, pill_w, pill_h)
        path = QPainterPath()
        path.addRoundedRect(pill, pill_h * 0.5, pill_h * 0.5)
        painter.fillPath(path, bg)
        painter.setPen(fg)
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        t = palette(self._theme)
        painter.fillRect(self.rect(), QColor(t["canvas"]))

        split_active = (
            self._split_compare
            and self.has_image()
            and self._pixmap is not None
            and self._source_pixmap is not None
            and not self._source_pixmap.isNull()
        )

        if self.has_image() and self._pixmap is not None:
            # Smooth interpolation when fitting/downscaling; crisp pixels at high zoom.
            painter.setRenderHint(
                QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 3.0
            )
            target = self._image_rect()
            frame_rect = target.adjusted(-0.5, -0.5, 0.5, 0.5)

            if split_active and self._source_pixmap is not None:
                split_x = self._split_line_x(target)
                # Left side: live fused color preview
                painter.save()
                painter.setClipRect(QRectF(target.left(), target.top(), max(0.0, split_x - target.left()), target.height()))
                painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
                painter.restore()

                # Right side: selected source slice aligned pixel-for-pixel
                painter.save()
                painter.setClipRect(QRectF(split_x, target.top(), max(0.0, target.right() - split_x), target.height()))
                painter.drawPixmap(target, self._source_pixmap, QRectF(self._source_pixmap.rect()))
                painter.restore()

                # Hairline frame around image
                painter.setPen(QPen(_qcolor_with_alpha(t["borderStrong"], 110), 1.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(frame_rect)

                # Vertical curtain divider + centered circular drag handle
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                top_y = max(0.0, target.top())
                bot_y = min(float(self.height()), target.bottom())
                painter.setPen(QPen(QColor(0, 0, 0, 120), 3.2))
                painter.drawLine(QPointF(split_x, top_y), QPointF(split_x, bot_y))
                painter.setPen(QPen(QColor(255, 255, 255, 240), 1.6))
                painter.drawLine(QPointF(split_x, top_y), QPointF(split_x, bot_y))

                cy = (top_y + bot_y) * 0.5
                knob_r = 13.0
                painter.setPen(QPen(QColor(t["accent"]), 1.8))
                painter.setBrush(_qcolor_with_alpha(t["surface"], 238))
                painter.drawEllipse(QPointF(split_x, cy), knob_r, knob_r)
                # Crisp left/right chevron triangles inside knob
                arrow_col = QColor(t["text"])
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(arrow_col)
                left_tri = QPainterPath()
                left_tri.moveTo(split_x - 7.5, cy)
                left_tri.lineTo(split_x - 3.0, cy - 3.8)
                left_tri.lineTo(split_x - 3.0, cy + 3.8)
                left_tri.closeSubpath()
                right_tri = QPainterPath()
                right_tri.moveTo(split_x + 7.5, cy)
                right_tri.lineTo(split_x + 3.0, cy - 3.8)
                right_tri.lineTo(split_x + 3.0, cy + 3.8)
                right_tri.closeSubpath()
                painter.drawPath(left_tri)
                painter.drawPath(right_tri)
                painter.restore()

                # Corner labels for left (Fused) & right (Source #i / n)
                if self._split_left_label:
                    self._draw_pill_label(
                        painter,
                        self._split_left_label,
                        12.0,
                        12.0,
                        _qcolor_with_alpha(t["accent"], 225),
                        QColor(t["textOnAccent"]),
                    )
                if self._split_right_label:
                    font = painter.font()
                    font.setPointSize(FONT["caption"] + 1)
                    font.setWeight(QFont.Weight.DemiBold)
                    painter.setFont(font)
                    rw = painter.fontMetrics().horizontalAdvance(self._split_right_label) + 18
                    self._draw_pill_label(
                        painter,
                        self._split_right_label,
                        max(12.0, self.width() - rw - 12.0),
                        12.0,
                        _qcolor_with_alpha(t["surface"], 225),
                        QColor(t["text"]),
                    )
            else:
                painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
                painter.setPen(QPen(_qcolor_with_alpha(t["borderStrong"], 110), 1.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(frame_rect)
        elif self._placeholder_text:
            painter.setPen(QColor(t["canvasText"]))
            painter.drawText(
                self.rect().adjusted(24, 24, -24, -24),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._placeholder_text,
            )

        # Floating top-right pill badge when viewing/peeking a source slice (non-split mode).
        if self._overlay_badge and not split_active:
            font = painter.font()
            font.setPointSize(FONT["caption"] + 1)
            font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(font)
            pill_w = painter.fontMetrics().horizontalAdvance(self._overlay_badge) + 20
            self._draw_pill_label(
                painter,
                self._overlay_badge,
                self.width() - pill_w - 12.0,
                12.0,
                _qcolor_with_alpha(t["accent"], 230),
                QColor(t["textOnAccent"]),
            )

        # Subtle outer hairline border separating canvas from surrounding chrome.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QColor(t["border"]))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class DepthHistogramWidget(QWidget):
    """Custom QPainter depth-contribution histogram with substack range shading."""

    sliceClicked = pyqtSignal(int)

    _PAD_X = 10
    _PAD_TOP = 8
    _PAD_BOTTOM = 16

    def __init__(self, parent: Optional[QWidget] = None, *, theme: str = "light"):
        super().__init__(parent)
        self._theme = theme
        self._values: np.ndarray = np.zeros(0, dtype=np.float32)
        self._start_idx: int = 0
        self._end_idx: int = 0
        self._current_slice: int = -1
        self._empty_text: str = trans.t("live_preview_no_histogram")
        self.setMinimumHeight(74)
        self.setMaximumHeight(82)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self.update()

    def set_empty_text(self, text: str) -> None:
        self._empty_text = text
        self.update()

    def set_histogram(self, values: Optional[Sequence[float]]) -> None:
        if values is None or len(values) == 0:
            self._values = np.zeros(0, dtype=np.float32)
        else:
            arr = np.asarray(values, dtype=np.float32).ravel()
            arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), 0.0)
            self._values = arr
        self.update()

    def set_substack_range(self, start_idx: int, end_idx: int) -> None:
        self._start_idx = int(start_idx)
        self._end_idx = int(end_idx)
        self.update()

    def set_current_slice(self, index: int) -> None:
        if index != self._current_slice:
            self._current_slice = int(index)
            self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(
            self._PAD_X,
            self._PAD_TOP,
            max(1.0, self.width() - 2 * self._PAD_X),
            max(1.0, self.height() - self._PAD_TOP - self._PAD_BOTTOM),
        )

    def _index_for_x(self, x: float) -> int:
        n = len(self._values)
        if n <= 1:
            return 0
        plot = self._plot_rect()
        ratio = (x - plot.left()) / max(1.0, plot.width())
        return max(0, min(n - 1, int(round(ratio * (n - 1)))))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        n = len(self._values)
        if n > 0:
            idx = self._index_for_x(event.position().x())
            total = float(self._values.sum())
            pct = (float(self._values[idx]) / total * 100.0) if total > 1e-12 else 0.0
            self.setToolTip(
                trans.t("live_preview_histogram_tip").format(i=idx + 1, pct=pct)
            )
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and len(self._values) > 0:
            self.sliceClicked.emit(self._index_for_x(event.position().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        t = palette(self._theme)

        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        bg_path = QPainterPath()
        bg_path.addRoundedRect(outer, RADIUS["sm"], RADIUS["sm"])
        painter.fillPath(bg_path, QColor(t["surfaceSunken"]))
        painter.setPen(QPen(QColor(t["border"]), 1.0))
        painter.drawPath(bg_path)

        plot = self._plot_rect()
        n = len(self._values)
        if n == 0:
            painter.setPen(QColor(t["textMuted"]))
            font = painter.font()
            font.setPointSize(FONT["caption"])
            painter.setFont(font)
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, self._empty_text)
            return

        # Subtle horizontal guide lines at 50% and baseline.
        painter.setPen(QPen(_qcolor_with_alpha(t["border"], 140), 1.0, Qt.PenStyle.DotLine))
        mid_y = plot.top() + plot.height() * 0.5
        painter.drawLine(QPointF(plot.left(), mid_y), QPointF(plot.right(), mid_y))
        painter.setPen(QPen(QColor(t["borderStrong"]), 1.0))
        painter.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))

        vmax = float(self._values.max()) if n > 0 else 0.0
        start = max(0, min(n - 1, self._start_idx))
        end = max(start, min(n - 1, self._end_idx))

        # Shade excluded outer regions so active substack pops cleanly.
        if n > 1:
            x_start = plot.left() + (start / (n - 1)) * plot.width()
            x_end = plot.left() + (end / (n - 1)) * plot.width()
        else:
            x_start, x_end = plot.left(), plot.right()

        scrim_color = _qcolor_with_alpha(t["text"], 22 if self._theme == "light" else 56)
        if start > 0 and x_start > plot.left():
            painter.fillRect(
                QRectF(plot.left(), plot.top(), x_start - plot.left(), plot.height()),
                scrim_color,
            )
        if end < n - 1 and x_end < plot.right():
            painter.fillRect(
                QRectF(x_end, plot.top(), plot.right() - x_end, plot.height()),
                scrim_color,
            )

        # Draw per-slice bars.
        bar_step = plot.width() / max(1, n)
        bar_w = max(1.5, bar_step - (1.0 if bar_step >= 4.0 else 0.3))
        active_fill = QColor(t["accent"])
        inactive_fill = _qcolor_with_alpha(t["textMuted"], 85)
        peak_accent = QColor(t["accentHover"])

        if vmax > 1e-12:
            for i in range(n):
                val = float(self._values[i])
                if val <= 0.0:
                    continue
                norm = val / vmax
                bh = max(2.0, norm * (plot.height() - 3.0))
                if n > 1:
                    cx = plot.left() + (i / (n - 1)) * plot.width()
                    bx = max(plot.left(), min(plot.right() - bar_w, cx - bar_w * 0.5))
                else:
                    bx = plot.left() + (plot.width() - bar_w) * 0.5
                rect = QRectF(bx, plot.bottom() - bh, bar_w, bh)
                in_range = start <= i <= end
                color = (peak_accent if norm > 0.85 else active_fill) if in_range else inactive_fill
                painter.fillRect(rect, color)

        # Crisp vertical bracket hairlines at substack boundaries when trimmed.
        if start > 0 or end < n - 1:
            bracket_pen = QPen(QColor(t["accent"]), 1.5)
            painter.setPen(bracket_pen)
            if start > 0:
                painter.drawLine(QPointF(x_start, plot.top()), QPointF(x_start, plot.bottom()))
            if end < n - 1:
                painter.drawLine(QPointF(x_end, plot.top()), QPointF(x_end, plot.bottom()))

        # Current browsed slice hairline indicator.
        if 0 <= self._current_slice < n:
            sx = plot.left() + (self._current_slice / max(1, n - 1)) * plot.width()
            painter.setPen(QPen(QColor(t["text"]), 1.5))
            painter.drawLine(QPointF(sx, plot.top()), QPointF(sx, plot.bottom()))

        # Subtle axis endpoint labels (1 ... N).
        font = painter.font()
        font.setPointSize(max(7, FONT["caption"] - 1))
        painter.setFont(font)
        painter.setPen(QColor(t["textFaint"]))
        label_y = self.height() - 3
        painter.drawText(int(plot.left()), label_y, "1")
        right_txt = str(n)
        rw = painter.fontMetrics().horizontalAdvance(right_txt)
        painter.drawText(int(plot.right() - rw), label_y, right_txt)
        if n >= 3:
            mid_txt = str((n + 1) // 2)
            mw = painter.fontMetrics().horizontalAdvance(mid_txt)
            painter.drawText(int(plot.center().x() - mw * 0.5), label_y, mid_txt)


class SubstackRangeSlider(QWidget):
    """Crisp dual-handle range slider aligned with DepthHistogramWidget."""

    rangeChanged = pyqtSignal(int, int)  # (start_idx, end_idx) inclusive 0-based
    dragStarted = pyqtSignal()
    dragFinished = pyqtSignal()

    _PAD_X = 10
    _HANDLE_R = 7

    def __init__(self, parent: Optional[QWidget] = None, *, theme: str = "light"):
        super().__init__(parent)
        self._theme = theme
        self._minimum: int = 0
        self._maximum: int = 1
        self._start: int = 0
        self._end: int = 1
        self._min_span: int = 1  # inclusive indices -> span >= 1 means >= 2 frames
        self._active_handle: Optional[str] = None  # "start" | "end" | None
        self._hover_handle: Optional[str] = None
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self.update()

    def set_bounds(self, minimum: int, maximum: int, *, min_span: int = 1) -> None:
        self._minimum = int(minimum)
        self._maximum = max(self._minimum, int(maximum))
        self._min_span = min(max(0, int(min_span)), self._maximum - self._minimum)
        self._start = max(self._minimum, min(self._start, self._maximum - self._min_span))
        self._end = max(self._start + self._min_span, min(self._end, self._maximum))
        self.update()

    def set_range(self, start: int, end: int, *, emit: bool = False) -> None:
        s = max(self._minimum, min(int(start), self._maximum - self._min_span))
        e = max(s + self._min_span, min(int(end), self._maximum))
        if s == self._start and e == self._end:
            return
        self._start, self._end = s, e
        self.update()
        if emit:
            self.rangeChanged.emit(self._start, self._end)

    def start_value(self) -> int:
        return self._start

    def end_value(self) -> int:
        return self._end

    def range_values(self) -> tuple[int, int]:
        return (self._start, self._end)

    def _track_left(self) -> float:
        return float(self._PAD_X)

    def _track_width(self) -> float:
        return max(1.0, float(self.width() - 2 * self._PAD_X))

    def _x_for_val(self, val: int) -> float:
        span = max(1, self._maximum - self._minimum)
        return self._track_left() + ((val - self._minimum) / span) * self._track_width()

    def _val_for_x(self, x: float) -> int:
        span = self._maximum - self._minimum
        if span <= 0:
            return self._minimum
        ratio = (x - self._track_left()) / self._track_width()
        return max(self._minimum, min(self._maximum, int(round(self._minimum + ratio * span))))

    def _pick_handle(self, x: float) -> str:
        xs = self._x_for_val(self._start)
        xe = self._x_for_val(self._end)
        ds = abs(x - xs)
        de = abs(x - xe)
        if ds <= de:
            return "start"
        return "end"

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            x = event.position().x()
            self._active_handle = self._pick_handle(x)
            self.dragStarted.emit()
            self._drag_to(x)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        x = event.position().x()
        if self._active_handle is not None:
            self._drag_to(x)
            event.accept()
            return
        xs = self._x_for_val(self._start)
        xe = self._x_for_val(self._end)
        hover = None
        if abs(x - xs) <= self._HANDLE_R + 4:
            hover = "start"
        elif abs(x - xe) <= self._HANDLE_R + 4:
            hover = "end"
        if hover != self._hover_handle:
            self._hover_handle = hover
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if hover else Qt.CursorShape.PointingHandCursor
            )
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._active_handle is not None:
            self._active_handle = None
            self.update()
            self.dragFinished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _drag_to(self, x: float) -> None:
        val = self._val_for_x(x)
        if self._active_handle == "start":
            new_start = max(self._minimum, min(val, self._end - self._min_span))
            if new_start != self._start:
                self._start = new_start
                self.update()
                self.rangeChanged.emit(self._start, self._end)
        elif self._active_handle == "end":
            new_end = min(self._maximum, max(val, self._start + self._min_span))
            if new_end != self._end:
                self._end = new_end
                self.update()
                self.rangeChanged.emit(self._start, self._end)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        t = palette(self._theme)

        cy = self.height() * 0.5
        left = self._track_left()
        right = left + self._track_width()
        xs = self._x_for_val(self._start)
        xe = self._x_for_val(self._end)

        # Background groove.
        groove_h = 4.0
        groove = QRectF(left, cy - groove_h * 0.5, right - left, groove_h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(t["surfaceActive"]))
        painter.drawRoundedRect(groove, 2.0, 2.0)

        # Active substack range highlight.
        active = QRectF(xs, cy - groove_h * 0.5, max(2.0, xe - xs), groove_h)
        painter.setBrush(QColor(t["accent"] if self.isEnabled() else t["borderStrong"]))
        painter.drawRoundedRect(active, 2.0, 2.0)

        # Dual handles.
        for name, hx in (("start", xs), ("end", xe)):
            is_active = self._active_handle == name
            is_hover = self._hover_handle == name
            r = self._HANDLE_R + (1.0 if is_active else 0.0)
            fill = QColor(t["accentSoft"] if is_active else t["surface"])
            border = QColor(
                t["accent"] if (is_active or is_hover) else t["borderStrong"]
            )
            painter.setBrush(fill)
            painter.setPen(QPen(border, 1.8 if (is_active or is_hover) else 1.2))
            painter.drawEllipse(QPointF(hx, cy), r, r)
            # Crisp inner dot for visual precision.
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(t["accent"] if self.isEnabled() else t["textFaint"]))
            painter.drawEllipse(QPointF(hx, cy), 2.2, 2.2)


class _ParamSliderRow(QWidget):
    """Minimalist label + slider + StepperSpinBox parameter row."""

    valueChanged = pyqtSignal(int)

    def __init__(
        self,
        label_text: str,
        *,
        minimum: int,
        maximum: int,
        value: int,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._minimum = int(minimum)
        self._maximum = int(maximum)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.label = QLabel(label_text, self)
        header.addWidget(self.label, 1)

        self.spin = StepperSpinBox(
            self,
            minimum=minimum,
            maximum=maximum,
            step=1,
            decimals=0,
            value=value,
        )
        self.spin.setFixedWidth(74)
        self.spin.setAccessibleName(label_text)
        self.spin.lineEdit().setAccessibleName(label_text)
        header.addWidget(self.spin)
        root.addLayout(header)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(minimum, maximum)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, (maximum - minimum) // 10))
        self.slider.setValue(value)
        self.slider.setAccessibleName(label_text)
        root.addWidget(self.slider)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin.valueChanged.connect(self._on_spin_changed)

    def set_label(self, text: str) -> None:
        clean = (text or "").rstrip(":： ").strip()
        self.label.setText(clean)
        self.slider.setAccessibleName(clean)
        self.spin.setAccessibleName(clean)
        self.spin.lineEdit().setAccessibleName(clean)

    def set_range(self, minimum: int, maximum: int) -> None:
        self._minimum = int(minimum)
        self._maximum = int(maximum)
        with QSignalBlocker(self.slider), QSignalBlocker(self.spin):
            self.slider.setRange(self._minimum, self._maximum)
            self.spin.setRange(self._minimum, self._maximum)

    def set_value(self, value: int) -> None:
        clamped = max(self._minimum, min(self._maximum, int(round(value))))
        with QSignalBlocker(self.slider), QSignalBlocker(self.spin):
            self.slider.setValue(clamped)
            self.spin.setValue(clamped)

    def value(self) -> int:
        return int(self.slider.value())

    def _on_slider_changed(self, value: int) -> None:
        with QSignalBlocker(self.spin):
            self.spin.setValue(value)
        self.valueChanged.emit(int(value))

    def _on_spin_changed(self, value: float) -> None:
        iv = int(round(value))
        with QSignalBlocker(self.slider):
            self.slider.setValue(iv)
        self.valueChanged.emit(iv)


class _LivePreviewWorker(QThread):
    """Single persistent latest-wins worker for proxy preparation and live fusion."""

    prepareProgress = pyqtSignal(int, int, str)  # (done, total, stage)
    prepareReady = pyqtSignal(int, int, int)     # (gen, width, height)
    frameReady = pyqtSignal(int, object, object, float, int, int, str, bool)
    # (request_id, bgr_image, depth_histogram, elapsed_ms, width, height, method, is_draft)
    failed = pyqtSignal(str)

    def __init__(
        self,
        session_factory: Callable[[str], Any],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        configure_native_worker_stack(self)
        self._session_factory = session_factory
        self._cond = threading.Condition()
        self._stop_flag = False
        self._cancel_prepare = threading.Event()
        self._cancel_render = threading.Event()
        self._pending_prepare: Optional[tuple[int, str]] = None
        self._pending_render: Optional[tuple[int, str, dict, Optional[tuple[int, int]], bool]] = None
        self._session_lock = threading.Lock()
        self._session: Any = None
        self._session_size_key: Optional[str] = None
        self._active_gen: int = 0

    def request_prepare(self, gen: int, size_key: str) -> None:
        with self._cond:
            self._active_gen = gen
            self._pending_prepare = (gen, size_key)
            self._cancel_prepare.set()
            self._cancel_render.set()
            self._cond.notify_all()

    def request_render(
        self,
        request_id: int,
        method: str,
        params: Mapping[str, Any],
        substack_range: Optional[tuple[int, int]] = None,
        *,
        draft: bool = False,
    ) -> None:
        """Latest-wins slot: overwrites any unstarted older slider/method request."""
        with self._cond:
            self._pending_render = (
                int(request_id),
                str(method),
                dict(params),
                tuple(substack_range) if substack_range is not None else None,
                bool(draft),
            )
            self._cond.notify_all()

    def get_source_frame(self, index: int, *, aligned: bool = True) -> Optional[np.ndarray]:
        with self._session_lock:
            session = self._session
            if session is None:
                return None
            try:
                return _session_get_source_frame(session, index, aligned=aligned)
            except Exception:
                return None

    def stop_and_close(self, timeout_ms: int = 3000) -> bool:
        """Request shutdown, wait up to timeout_ms, and release session memory."""
        with self._cond:
            self._stop_flag = True
            self._pending_prepare = None
            self._pending_render = None
            self._cancel_prepare.set()
            self._cancel_render.set()
            self._cond.notify_all()

        finished = True
        if self.isRunning():
            finished = self.wait(timeout_ms)
        if finished:
            self._close_session()
        return finished

    def _close_session(self) -> None:
        with self._session_lock:
            session = self._session
            self._session = None
            self._session_size_key = None
        if session is not None:
            close_fn = getattr(session, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

    def run(self) -> None:  # noqa: D401
        try:
            while True:
                with self._cond:
                    while (
                        not self._stop_flag
                        and self._pending_prepare is None
                        and self._pending_render is None
                    ):
                        self._cond.wait()
                    if self._stop_flag:
                        break
                    prep = self._pending_prepare
                    if prep is not None:
                        self._pending_prepare = None
                        self._cancel_prepare.clear()
                        self._cancel_render.clear()
                    else:
                        render_req = self._pending_render
                        self._pending_render = None
                        self._cancel_render.clear()

                if prep is not None:
                    gen, size_key = prep
                    self._run_prepare(gen, size_key)
                else:
                    self._run_render(*render_req)
        finally:
            self._close_session()

    def _run_prepare(self, gen: int, size_key: str) -> None:
        try:
            self._close_session()
            if self._stop_flag or self._cancel_prepare.is_set():
                return

            def _on_progress(*args):
                # Support (done, total), (done, total, stage), or (stage, done, total).
                done, total, stage = 0, 1, "load"
                if len(args) == 2:
                    done, total = int(args[0]), max(1, int(args[1]))
                elif len(args) >= 3:
                    if isinstance(args[0], str):
                        stage, done, total = args[0], int(args[1]), max(1, int(args[2]))
                    else:
                        done, total, stage = int(args[0]), max(1, int(args[1])), str(args[2])
                if not self._stop_flag and not self._cancel_prepare.is_set():
                    self.prepareProgress.emit(done, total, stage)

            cancel_fn = lambda: self._stop_flag or self._cancel_prepare.is_set()
            try:
                session = self._session_factory(
                    size_key,
                    progress_cb=_on_progress,
                    cancel_check=cancel_fn,
                )
            except TypeError:
                session = self._session_factory(size_key)
            _session_prepare(
                session,
                progress_callback=_on_progress,
                should_cancel=cancel_fn,
            )
            if self._stop_flag or self._cancel_prepare.is_set():
                close_fn = getattr(session, "close", None)
                if callable(close_fn):
                    close_fn()
                return
            with self._session_lock:
                self._session = session
                self._session_size_key = size_key
            pw, ph = _session_proxy_size(session)
            self.prepareReady.emit(gen, pw, ph)
        except Exception as exc:
            if not self._stop_flag and not self._cancel_prepare.is_set():
                msg = str(exc)
                if "cancel" not in msg.lower():
                    self.failed.emit(msg)

    def _run_render(
        self,
        request_id: int,
        method: str,
        params: dict,
        substack_range: Optional[tuple[int, int]],
        draft: bool = False,
    ) -> None:
        with self._session_lock:
            session = self._session
        if session is None or self._stop_flag or self._cancel_prepare.is_set():
            return
        try:
            t0 = time.perf_counter()
            result = _session_render(
                session,
                method=method,
                params=params,
                substack_range=substack_range,
                draft=draft,
                should_cancel=lambda: self._stop_flag or self._cancel_prepare.is_set(),
            )
            if self._stop_flag or self._cancel_prepare.is_set() or result is None:
                return
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            bgr, hist, res_ms, is_draft = _unpack_render_result(result, elapsed_ms, fallback_draft=draft)
            if bgr is None:
                return
            h, w = bgr.shape[:2]
            self.frameReady.emit(request_id, bgr, hist, res_ms, w, h, method, is_draft)
        except Exception as exc:
            if not self._stop_flag and not self._cancel_prepare.is_set():
                msg = str(exc)
                if "cancel" not in msg.lower():
                    self.failed.emit(msg)


# ---------------------------------------------------------------------------
# Thin adapters to core.live_preview (kept decoupled & resilient)
# ---------------------------------------------------------------------------

def _session_prepare(session, *, progress_callback, should_cancel):
    prepare_fn = getattr(session, "prepare", None)
    if callable(prepare_fn):
        try:
            return prepare_fn(progress_callback=progress_callback, should_cancel=should_cancel)
        except TypeError:
            return prepare_fn(progress_callback=progress_callback)
    return None


def _session_proxy_size(session) -> tuple[int, int]:
    for attr in ("proxy_size", "frame_size", "resolution"):
        val = getattr(session, attr, None)
        if callable(val):
            val = val()
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return int(val[0]), int(val[1])
    frame = _session_get_source_frame(session, 0, aligned=True)
    if frame is not None and hasattr(frame, "shape"):
        return int(frame.shape[1]), int(frame.shape[0])
    return (0, 0)


def _session_get_source_frame(session, index: int, *, aligned: bool = True) -> Optional[np.ndarray]:
    getter = getattr(session, "get_source_frame", None)
    if callable(getter):
        if not aligned:
            try:
                return getter(index, aligned=False)
            except TypeError:
                return None  # Let dialog fall back to single-frame unaligned proxy decode.
        try:
            return getter(index, aligned=True)
        except TypeError:
            return getter(index)
    frames = getattr(session, "aligned_frames", None) or getattr(session, "frames", None)
    if frames is not None and 0 <= index < len(frames):
        return frames[index]
    return None


def _session_render(session, *, method: str, params: dict, substack_range, draft: bool = False, should_cancel=None):
    render_fn = getattr(session, "render", None)
    if not callable(render_fn):
        raise RuntimeError("LivePreviewSession has no render() method")
    try:
        from core.live_preview import LivePreviewRequest
        req = LivePreviewRequest(
            method=method,
            radius=int(params.get("radius", 8)),
            smoothing=float(params.get("smoothing", 4.0)),
            feather=int(params.get("feather", 3)),
            substack_range=substack_range,
            draft=bool(draft),
        )
        try:
            return render_fn(req, cancel_check=should_cancel)
        except TypeError:
            try:
                return render_fn(req, should_cancel=should_cancel)
            except TypeError:
                return render_fn(req)
    except (ImportError, TypeError):
        try:
            return render_fn(
                method=method,
                radius=int(params.get("radius", 8)),
                smoothing=float(params.get("smoothing", 4.0)),
                feather=int(params.get("feather", 3)),
                substack_range=substack_range,
                draft=bool(draft),
            )
        except TypeError:
            return render_fn(
                method=method,
                radius=int(params.get("radius", 8)),
                smoothing=float(params.get("smoothing", 4.0)),
                feather=int(params.get("feather", 3)),
                substack_range=substack_range,
            )


def _unpack_render_result(
    result, fallback_ms: float, *, fallback_draft: bool = False
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float, bool]:
    if isinstance(result, np.ndarray):
        return result, None, fallback_ms, bool(fallback_draft)
    image = getattr(result, "image_bgr", None)
    if image is None:
        image = getattr(result, "image", None)
    if image is None:
        image = getattr(result, "fused", None)
    hist = getattr(result, "depth_histogram", None)
    elapsed = getattr(result, "elapsed_ms", None)
    ms = float(elapsed) if elapsed is not None else fallback_ms
    is_draft = bool(getattr(result, "is_draft", fallback_draft))
    return image, hist, ms, is_draft


class LivePreviewDialog(QDialog):
    """Standalone interactive Live Color Preview window for OpenFocus A/B/C."""

    parametersApplied = pyqtSignal(str, dict, object)  # (method, params, substack_range)
    enqueue_requested = pyqtSignal(dict)               # {method, params, substack_range}

    def __init__(
        self,
        window_main: Any = None,
        *,
        images: Sequence[Any] = (),
        filenames: Sequence[str] = (),
        initial_method: str = "openfocus_b",
        method_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
        alignment_method: str = "ecc",
        registration_params: Optional[Mapping[str, Any]] = None,
        cached_transforms: Optional[Sequence[Any]] = None,
        aligned_frames: Optional[Sequence[Any]] = None,
        initial_substack: Optional[tuple[int, int]] = None,
        source_revision: int = 0,
        theme: Optional[str] = None,
        session_factory: Optional[Callable[..., Any]] = None,
        parent: Optional[QWidget] = None,
    ):
        qt_parent = parent if isinstance(parent, QWidget) else (
            window_main if isinstance(window_main, QWidget) else None
        )
        super().__init__(qt_parent)
        self.window_main = window_main
        self._images = list(images)
        self._frame_count = len(self._images)
        self._filenames = list(filenames)
        self._source_revision = int(source_revision)
        self._stack_identities = tuple(id(img) for img in self._images)
        self._alignment_method = alignment_method or "none"
        self._registration_params = dict(registration_params or {})
        self._cached_transforms = list(cached_transforms) if cached_transforms is not None else None
        self._aligned_frames = list(aligned_frames) if aligned_frames is not None else None
        self._unaligned_cache: tuple[Optional[tuple[str, int]], Optional[np.ndarray]] = (None, None)
        self._theme = theme or getattr(window_main, "ui_theme", None) or CURRENT_THEME
        self._custom_session_factory = session_factory

        # Active method (strictly OpenFocus A/B/C; default to B).
        self._method = initial_method if initial_method in _METHOD_SHORT_NAMES else "openfocus_b"
        self._method_params: dict[str, dict[str, Any]] = {}
        for mid in ("openfocus_a", "openfocus_b", "openfocus_c"):
            supplied = (method_params or {}).get(mid) if method_params else None
            self._method_params[mid] = resolve_fusion_parameters(mid, supplied)

        # Substack range (0-based inclusive).
        max_idx = max(0, self._frame_count - 1)
        if initial_substack is not None and self._frame_count >= 2:
            s0 = max(0, min(int(initial_substack[0]), max_idx - 1))
            e0 = max(s0 + 1, min(int(initial_substack[1]), max_idx))
            self._substack_range: tuple[int, int] = (s0, e0)
        else:
            self._substack_range = (0, max_idx)
        self._initial_substack = self._substack_range
        self._substack_modified = False

        # View state.
        self._preview_size = "medium"
        self._view_mode = "fused"  # "fused" | "source"
        self._peeking_source = False
        self._split_compare = False
        self._show_aligned_source = True
        self._browse_index = self._substack_range[0]
        self._fused_pixmap: Optional[QPixmap] = None
        self._last_elapsed_ms: float = 0.0
        self._last_resolution: tuple[int, int] = (0, 0)
        self._proxy_logical_size: tuple[int, int] = (0, 0)
        self._last_is_draft: bool = False
        self._is_dragging: bool = False
        self._last_histogram: Optional[np.ndarray] = None
        self._prepare_gen: int = 0
        self._request_seq: int = 0
        self._session_ready: bool = False
        self._closing: bool = False

        # Short settle timer triggers a crisp 100% full-proxy refine if the user pauses mid-drag.
        self._refine_timer = QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.setInterval(140)
        self._refine_timer.timeout.connect(self._on_refine_timeout)

        self.setObjectName("livePreviewDialog")
        self.setWindowTitle(trans.t("live_preview_title"))
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        self._configure_initial_size()
        self._build_ui()
        self.apply_theme(self._theme)
        self._sync_controls_to_state()
        self._update_substack_ui()
        self._update_stack_browser_ui()

        # Persistent single worker thread.
        self._worker = _LivePreviewWorker(self._create_session, parent=self)
        self._worker.prepareProgress.connect(self._on_prepare_progress)
        self._worker.prepareReady.connect(self._on_prepare_ready)
        self._worker.frameReady.connect(self._on_frame_ready)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

        # Install event filter on dialog to capture Space peek cleanly.
        self.installEventFilter(self)

        if self._frame_count >= 2:
            self._start_prepare(self._preview_size)

    def _configure_initial_size(self) -> None:
        target_w, target_h = 1220, 760
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            target_w = min(target_w, max(760, avail.width() - 80))
            target_h = min(target_h, max(540, avail.height() - 100))
        self.resize(target_w, target_h)
        self.setMinimumSize(720, 500)

    def matches_stack(self, raw_images: Sequence[Any], source_revision: int) -> bool:
        return (
            self._source_revision == int(source_revision)
            and len(raw_images) == self._frame_count
            and tuple(id(img) for img in raw_images) == self._stack_identities
        )

    def _create_session(
        self,
        size_key: str,
        *,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Any:
        if self._custom_session_factory is not None:
            return self._custom_session_factory(size_key)
        from core.live_preview import LivePreviewSession

        return LivePreviewSession(
            self._images,
            size=size_key,
            align_method=self._alignment_method,
            cached_transforms=self._cached_transforms,
            aligned_frames=self._aligned_frames,
            cancel_check=cancel_check,
            progress_cb=progress_cb,
        )

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["sm"])
        body.setSpacing(SPACE["md"])

        # === Left Main Preview Area ===
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(SPACE["sm"])

        self.canvas_host = QWidget(self)
        canvas_stack = QVBoxLayout(self.canvas_host)
        canvas_stack.setContentsMargins(0, 0, 0, 0)
        canvas_stack.setSpacing(0)

        self.canvas = LivePreviewCanvas(self.canvas_host, theme=self._theme)
        self.canvas.setProperty("controlId", "live_preview_canvas")
        self.canvas.setToolTip(trans.t("live_preview_fit_hint"))
        self.canvas.zoomChanged.connect(self._on_zoom_changed)
        canvas_stack.addWidget(self.canvas, 1)

        # Clean loading overlay card (positioned inside canvas_host during prepare).
        self.loading_overlay = QFrame(self.canvas_host)
        self.loading_overlay.setObjectName("livePreviewOverlay")
        ov_layout = QVBoxLayout(self.loading_overlay)
        ov_layout.setContentsMargins(20, 14, 20, 14)
        ov_layout.setSpacing(8)
        self.lbl_loading = QLabel(trans.t("live_preview_preparing").format(done=0, total=max(1, self._frame_count)))
        self.lbl_loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ov_layout.addWidget(self.lbl_loading)
        self.progress_bar = QProgressBar(self.loading_overlay)
        self.progress_bar.setRange(0, max(1, self._frame_count))
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(240)
        self.progress_bar.setTextVisible(False)
        ov_layout.addWidget(self.progress_bar, alignment=Qt.AlignmentFlag.AlignCenter)
        self.loading_overlay.hide()

        left_col.addWidget(self.canvas_host, 1)

        # Subtle bottom-left status / size bar.
        bottom_left = QHBoxLayout()
        bottom_left.setContentsMargins(2, 0, 2, 0)
        bottom_left.setSpacing(SPACE["sm"])

        self.lbl_size = QLabel(trans.t("live_preview_size_label"))
        self.lbl_size.setObjectName("mutedLabel")
        bottom_left.addWidget(self.lbl_size)

        self.combo_size = QComboBox(self)
        self.combo_size.setProperty("controlId", "live_preview_size")
        self.combo_size.setMinimumWidth(92)
        for key, label_key in PREVIEW_SIZE_PRESETS:
            self.combo_size.addItem(trans.t(label_key), key)
        self.combo_size.setCurrentIndex(1)  # Medium
        self.combo_size.currentIndexChanged.connect(self._on_preview_size_changed)
        bottom_left.addWidget(self.combo_size)

        self.btn_zoom_fit = QPushButton(trans.t("live_preview_zoom_fit"))
        self.btn_zoom_fit.setObjectName("secondaryButton")
        self.btn_zoom_fit.setProperty("controlId", "live_preview_zoom_fit")
        self.btn_zoom_fit.setToolTip(trans.t("live_preview_fit_hint"))
        self.btn_zoom_fit.clicked.connect(self.canvas.fit_to_window)
        bottom_left.addWidget(self.btn_zoom_fit)

        self.btn_zoom_100 = QPushButton(trans.t("live_preview_zoom_100"))
        self.btn_zoom_100.setObjectName("secondaryButton")
        self.btn_zoom_100.setProperty("controlId", "live_preview_zoom_100")
        self.btn_zoom_100.clicked.connect(self.canvas.set_actual_size)
        bottom_left.addWidget(self.btn_zoom_100)

        self.btn_split_compare = QPushButton(trans.t("live_preview_split_compare"))
        self.btn_split_compare.setCheckable(True)
        self.btn_split_compare.setChecked(False)
        self.btn_split_compare.setObjectName("livePreviewModeBtn")
        self.btn_split_compare.setProperty("controlId", "live_preview_split_compare")
        self.btn_split_compare.setToolTip(trans.t("live_preview_split_compare_hint"))
        self.btn_split_compare.toggled.connect(self._on_split_compare_toggled)
        bottom_left.addWidget(self.btn_split_compare)

        bottom_left.addStretch(1)

        self.lbl_perf_badge = QLabel("")
        self.lbl_perf_badge.setObjectName("livePreviewBadge")
        self.lbl_perf_badge.setProperty("controlId", "live_preview_perf_badge")
        self.lbl_perf_badge.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        bottom_left.addWidget(self.lbl_perf_badge)

        left_col.addLayout(bottom_left)
        body.addLayout(left_col, 1)

        # === Right Control Sidebar (~324px wide, clean cards) ===
        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_scroll.setFixedWidth(324)

        sidebar = QWidget()
        sidebar.setMinimumWidth(0)
        side_lay = QVBoxLayout(sidebar)
        side_lay.setContentsMargins(0, 0, 6, 0)
        side_lay.setSpacing(SPACE["sm"])

        # Card 1: Fusion Method
        method_card, method_lay, self.lbl_method_title = self._make_card(
            trans.t("live_preview_section_method")
        )
        self.method_group = QButtonGroup(self)
        self.method_radios: dict[str, QRadioButton] = {}
        for mid, label_key, _short in LIVE_PREVIEW_METHODS:
            rb = QRadioButton(trans.t(label_key), method_card)
            rb.setProperty("controlId", f"live_preview_method_{mid}")
            rb.setProperty("method_id", mid)
            self.method_group.addButton(rb)
            method_lay.addWidget(rb)
            self.method_radios[mid] = rb
        self.method_group.buttonClicked.connect(self._on_method_radio_clicked)
        side_lay.addWidget(method_card)

        # Card 2: Parameters (Radius / Smoothing / Feather)
        params_card, params_lay, self.lbl_params_title = self._make_card(
            trans.t("live_preview_section_params")
        )
        self.row_radius = _ParamSliderRow(
            trans.t("label_of_radius").rstrip(":： "),
            minimum=1,
            maximum=50,
            value=8,
            parent=params_card,
        )
        self.row_radius.setProperty("controlId", "live_preview_radius")
        self.row_radius.valueChanged.connect(lambda v: self._on_param_changed("radius", v))
        self.row_radius.slider.sliderPressed.connect(self._on_slider_drag_started)
        self.row_radius.slider.sliderReleased.connect(self._on_slider_drag_finished)
        params_lay.addWidget(self.row_radius)

        self.row_smoothing = _ParamSliderRow(
            trans.t("label_of_smoothing").rstrip(":： "),
            minimum=0,
            maximum=34,
            value=4,
            parent=params_card,
        )
        self.row_smoothing.setProperty("controlId", "live_preview_smoothing")
        self.row_smoothing.valueChanged.connect(lambda v: self._on_param_changed("smoothing", float(v)))
        self.row_smoothing.slider.sliderPressed.connect(self._on_slider_drag_started)
        self.row_smoothing.slider.sliderReleased.connect(self._on_slider_drag_finished)
        params_lay.addWidget(self.row_smoothing)

        self.row_feather = _ParamSliderRow(
            trans.t("label_of_feather").rstrip(":： "),
            minimum=0,
            maximum=10,
            value=3,
            parent=params_card,
        )
        self.row_feather.setProperty("controlId", "live_preview_feather")
        self.row_feather.valueChanged.connect(lambda v: self._on_param_changed("feather", v))
        self.row_feather.slider.sliderPressed.connect(self._on_slider_drag_started)
        self.row_feather.slider.sliderReleased.connect(self._on_slider_drag_finished)
        params_lay.addWidget(self.row_feather)

        side_lay.addWidget(params_card)

        # Card 3: Stack Browser
        stack_card, stack_lay, self.lbl_stack_title = self._make_card(
            trans.t("live_preview_section_stack")
        )
        self.btn_view_fused = QPushButton(trans.t("live_preview_view_fused"), stack_card)
        self.btn_view_fused.setCheckable(True)
        self.btn_view_fused.setChecked(True)
        self.btn_view_fused.setObjectName("livePreviewModeBtn")
        self.btn_view_fused.setProperty("controlId", "live_preview_view_fused")
        self.btn_view_fused.clicked.connect(self._switch_to_fused_view)
        stack_lay.addWidget(self.btn_view_fused)

        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(6)
        self.btn_prev_slice = QPushButton("◀", stack_card)
        self.btn_prev_slice.setObjectName("stepButton")
        self.btn_prev_slice.setFixedSize(26, 26)
        self.btn_prev_slice.clicked.connect(lambda: self._step_source_slice(-1))
        nav_row.addWidget(self.btn_prev_slice)

        self.stack_slider = QSlider(Qt.Orientation.Horizontal, stack_card)
        self.stack_slider.setProperty("controlId", "live_preview_stack_slider")
        self.stack_slider.setRange(0, max(0, self._frame_count - 1))
        self.stack_slider.setValue(self._browse_index)
        self.stack_slider.valueChanged.connect(self._on_stack_slider_changed)
        nav_row.addWidget(self.stack_slider, 1)

        self.btn_next_slice = QPushButton("▶", stack_card)
        self.btn_next_slice.setObjectName("stepButton")
        self.btn_next_slice.setFixedSize(26, 26)
        self.btn_next_slice.clicked.connect(lambda: self._step_source_slice(1))
        nav_row.addWidget(self.btn_next_slice)

        self.lbl_frame_counter = QLabel("", stack_card)
        self.lbl_frame_counter.setObjectName("mutedLabel")
        self.lbl_frame_counter.setMinimumWidth(44)
        self.lbl_frame_counter.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        nav_row.addWidget(self.lbl_frame_counter)
        stack_lay.addLayout(nav_row)

        opt_row = QHBoxLayout()
        opt_row.setContentsMargins(0, 0, 0, 0)
        opt_row.setSpacing(6)
        self.chk_show_aligned = QCheckBox(trans.t("live_preview_show_aligned"), stack_card)
        self.chk_show_aligned.setChecked(True)
        self.chk_show_aligned.setToolTip(trans.t("live_preview_show_aligned_hint"))
        self.chk_show_aligned.setProperty("controlId", "live_preview_show_aligned")
        self.chk_show_aligned.toggled.connect(self._on_show_aligned_toggled)
        opt_row.addWidget(self.chk_show_aligned)
        opt_row.addStretch(1)
        stack_lay.addLayout(opt_row)

        self.lbl_peek_hint = QLabel(trans.t("live_preview_shortcuts_hint"), stack_card)
        self.lbl_peek_hint.setObjectName("mutedLabel")
        self.lbl_peek_hint.setWordWrap(True)
        stack_lay.addWidget(self.lbl_peek_hint)
        side_lay.addWidget(stack_card)

        # Card 4: Depth Map Histogram + Substack Dual-Handle Range Slider
        hist_card, hist_lay, self.lbl_hist_title = self._make_card(
            trans.t("live_preview_section_histogram")
        )
        self.histogram_widget = DepthHistogramWidget(hist_card, theme=self._theme)
        self.histogram_widget.setProperty("controlId", "live_preview_histogram")
        self.histogram_widget.sliceClicked.connect(self._on_histogram_slice_clicked)
        hist_lay.addWidget(self.histogram_widget)

        self.substack_slider = SubstackRangeSlider(hist_card, theme=self._theme)
        self.substack_slider.setProperty("controlId", "live_preview_substack_slider")
        self.substack_slider.set_bounds(0, max(0, self._frame_count - 1), min_span=1 if self._frame_count >= 2 else 0)
        self.substack_slider.set_range(*self._substack_range, emit=False)
        self.substack_slider.rangeChanged.connect(self._on_substack_range_changed)
        self.substack_slider.dragStarted.connect(self._on_slider_drag_started)
        self.substack_slider.dragFinished.connect(self._on_slider_drag_finished)
        hist_lay.addWidget(self.substack_slider)

        substack_row = QHBoxLayout()
        substack_row.setContentsMargins(0, 0, 0, 0)
        substack_row.setSpacing(6)
        self.lbl_substack_caption = QLabel(trans.t("live_preview_substack_label"), hist_card)
        self.lbl_substack_caption.setObjectName("mutedLabel")
        substack_row.addWidget(self.lbl_substack_caption)

        self.lbl_substack_range = QLabel("", hist_card)
        self.lbl_substack_range.setProperty("controlId", "live_preview_substack_range")
        substack_row.addWidget(self.lbl_substack_range, 1)

        self.btn_reset_substack = QPushButton(trans.t("live_preview_substack_reset"), hist_card)
        self.btn_reset_substack.setObjectName("secondaryButton")
        self.btn_reset_substack.setProperty("controlId", "live_preview_substack_reset")
        self.btn_reset_substack.setFixedHeight(24)
        self.btn_reset_substack.clicked.connect(self._reset_substack_range)
        substack_row.addWidget(self.btn_reset_substack)
        hist_lay.addLayout(substack_row)

        side_lay.addWidget(hist_card)
        side_lay.addStretch(1)

        sidebar_scroll.setWidget(sidebar)
        body.addWidget(sidebar_scroll, 0)

        outer.addLayout(body, 1)

        # === Bottom-Right Action Bar ===
        footer = QWidget(self)
        footer.setObjectName("dialogButtonBar")
        footer_lay = QHBoxLayout(footer)
        footer_lay.setContentsMargins(SPACE["md"], SPACE["sm"], SPACE["md"], SPACE["sm"])
        footer_lay.setSpacing(SPACE["sm"])

        self.lbl_status = QLabel("", footer)
        self.lbl_status.setObjectName("mutedLabel")
        footer_lay.addWidget(self.lbl_status, 1)

        self.btn_apply = QPushButton(trans.t("live_preview_btn_apply"), footer)
        self.btn_apply.setObjectName("secondaryButton")
        self.btn_apply.setProperty("controlId", "live_preview_btn_apply")
        self.btn_apply.clicked.connect(self._on_apply_parameters)
        footer_lay.addWidget(self.btn_apply)

        self.btn_enqueue = QPushButton(trans.t("live_preview_btn_enqueue"), footer)
        self.btn_enqueue.setObjectName("secondaryButton")
        self.btn_enqueue.setProperty("controlId", "live_preview_btn_enqueue")
        self.btn_enqueue.clicked.connect(self._on_enqueue_render)
        has_queue = callable(getattr(self.window_main, "enqueue_render_job", None))
        self.btn_enqueue.setVisible(has_queue)
        footer_lay.addWidget(self.btn_enqueue)

        self.btn_apply_render = QPushButton(
            trans.t("live_preview_btn_apply_render").replace("&", "&&"), footer
        )
        self.btn_apply_render.setObjectName("primaryButton")
        self.btn_apply_render.setProperty("controlId", "live_preview_btn_apply_render")
        self.btn_apply_render.clicked.connect(self._on_apply_and_render)
        footer_lay.addWidget(self.btn_apply_render)

        self.btn_close = QPushButton(trans.t("btn_close"), footer)
        self.btn_close.setObjectName("secondaryButton")
        self.btn_close.setProperty("controlId", "live_preview_btn_close")
        self.btn_close.clicked.connect(self.close)
        footer_lay.addWidget(self.btn_close)

        outer.addWidget(footer)

    def _make_card(self, title: str) -> tuple[QFrame, QVBoxLayout, QLabel]:
        card = QFrame(self)
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(SPACE["md"], SPACE["sm"], SPACE["md"], SPACE["sm"])
        lay.setSpacing(6)
        header = QLabel(title, card)
        header.setObjectName("sectionLabel")
        lay.addWidget(header)
        return card, lay, header

    # ------------------------------------------------------------------
    # Theme & Localization
    # ------------------------------------------------------------------

    def apply_theme(self, theme: str) -> None:
        self._theme = theme or "light"
        t = palette(self._theme)
        base_qss = build_stylesheet(self._theme)
        extra_qss = f"""
        QFrame#livePreviewOverlay {{
            background-color: {t['surface']};
            border: 1px solid {t['borderStrong']};
            border-radius: {RADIUS['md']}px;
        }}
        QLabel#livePreviewBadge {{
            background-color: {t['surface']};
            color: {t['textMuted']};
            border: 1px solid {t['border']};
            border-radius: {RADIUS['sm']}px;
            padding: 3px 9px;
            font-size: {FONT['caption'] + 1}pt;
            font-weight: {WEIGHT['medium']};
        }}
        QPushButton#livePreviewModeBtn:checked {{
            background-color: {t['accentSoft']};
            color: {t['accentSoftText']};
            border-color: {t['accent']};
            font-weight: {WEIGHT['semibold']};
        }}
        """
        self.setStyleSheet(base_qss + extra_qss)
        self.canvas.set_theme(self._theme)
        self.histogram_widget.set_theme(self._theme)
        self.substack_slider.set_theme(self._theme)
        for spin in self.findChildren(StepperSpinBox):
            spin.reapply_theme()

    def retranslate(self) -> None:
        self.setWindowTitle(trans.t("live_preview_title"))
        self.lbl_size.setText(trans.t("live_preview_size_label"))
        for idx, (_key, label_key) in enumerate(PREVIEW_SIZE_PRESETS):
            self.combo_size.setItemText(idx, trans.t(label_key))
        self.btn_zoom_fit.setText(trans.t("live_preview_zoom_fit"))
        self.btn_zoom_fit.setToolTip(trans.t("live_preview_fit_hint"))
        self.btn_zoom_100.setText(trans.t("live_preview_zoom_100"))
        self.btn_split_compare.setText(trans.t("live_preview_split_compare"))
        self.btn_split_compare.setToolTip(trans.t("live_preview_split_compare_hint"))
        self.canvas.setToolTip(trans.t("live_preview_fit_hint"))
        self.lbl_method_title.setText(trans.t("live_preview_section_method"))
        for mid, label_key, _short in LIVE_PREVIEW_METHODS:
            self.method_radios[mid].setText(trans.t(label_key))
        self.lbl_params_title.setText(trans.t("live_preview_section_params"))
        self.row_radius.set_label(trans.t("label_of_radius"))
        self.row_smoothing.set_label(trans.t("label_of_smoothing"))
        self.row_feather.set_label(trans.t("label_of_feather"))
        self.lbl_stack_title.setText(trans.t("live_preview_section_stack"))
        self.btn_view_fused.setText(trans.t("live_preview_view_fused"))
        self.chk_show_aligned.setText(trans.t("live_preview_show_aligned"))
        self.chk_show_aligned.setToolTip(trans.t("live_preview_show_aligned_hint"))
        self.lbl_peek_hint.setText(trans.t("live_preview_shortcuts_hint"))
        self.lbl_hist_title.setText(trans.t("live_preview_section_histogram"))
        self.histogram_widget.set_empty_text(trans.t("live_preview_no_histogram"))
        self.lbl_substack_caption.setText(trans.t("live_preview_substack_label"))
        self.btn_reset_substack.setText(trans.t("live_preview_substack_reset"))
        self.btn_apply.setText(trans.t("live_preview_btn_apply"))
        self.btn_enqueue.setText(trans.t("live_preview_btn_enqueue"))
        self.btn_apply_render.setText(
            trans.t("live_preview_btn_apply_render").replace("&", "&&")
        )
        self.btn_close.setText(trans.t("btn_close"))
        self._update_substack_ui()
        self._update_stack_browser_ui()
        self._update_perf_badge()

    # ------------------------------------------------------------------
    # State Synchronization & Controls
    # ------------------------------------------------------------------

    def current_method(self) -> str:
        return self._method

    def current_parameters(self) -> dict[str, Any]:
        return dict(self._method_params[self._method])

    def current_substack_range(self) -> tuple[int, int]:
        return self._substack_range

    def _sync_controls_to_state(self) -> None:
        rb = self.method_radios.get(self._method)
        if rb is not None:
            with QSignalBlocker(rb):
                rb.setChecked(True)

        params = self._method_params[self._method]
        spec = get_fusion(self._method)
        spec_by_key = {p.key: p for p in spec.parameters}

        # Radius: applies to OpenFocus A and B; hidden on C.
        has_radius = "radius" in spec_by_key
        self.row_radius.setVisible(has_radius)
        if has_radius:
            p = spec_by_key["radius"]
            self.row_radius.set_range(int(p.minimum), int(p.maximum))
            self.row_radius.set_value(int(params.get("radius", p.default)))

        # Smoothing: hidden on OpenFocus A (no-op), 0..34 on B, 0..14 on C.
        has_smoothing = (
            self._method != "openfocus_a" and "smoothing" in spec_by_key
        )
        self.row_smoothing.setVisible(has_smoothing)
        if has_smoothing:
            p = spec_by_key["smoothing"]
            self.row_smoothing.set_range(int(p.minimum), int(p.maximum))
            self.row_smoothing.set_value(int(round(params.get("smoothing", p.default))))

        # Feather: applies only to OpenFocus B.
        has_feather = "feather" in spec_by_key
        self.row_feather.setVisible(has_feather)
        if has_feather:
            p = spec_by_key["feather"]
            self.row_feather.set_range(int(p.minimum), int(p.maximum))
            self.row_feather.set_value(int(params.get("feather", p.default)))

    def _update_substack_ui(self) -> None:
        s, e = self._substack_range
        n = max(1, self._frame_count)
        self.lbl_substack_range.setText(
            trans.t("live_preview_substack_range").format(start=s + 1, end=e + 1, total=n)
        )
        is_trimmed = (s > 0) or (e < max(0, self._frame_count - 1))
        self.btn_reset_substack.setEnabled(is_trimmed)
        self.histogram_widget.set_substack_range(s, e)
        self.substack_slider.set_range(s, e, emit=False)

    def _update_stack_browser_ui(self) -> None:
        n = max(1, self._frame_count)
        idx = max(0, min(self._browse_index, n - 1))
        self._browse_index = idx
        with QSignalBlocker(self.stack_slider):
            self.stack_slider.setValue(idx)
        self.lbl_frame_counter.setText(
            trans.t("live_preview_frame_counter").format(i=idx + 1, n=n)
        )
        self.btn_prev_slice.setEnabled(idx > 0)
        self.btn_next_slice.setEnabled(idx < n - 1)
        showing_source = self._view_mode == "source" or self._peeking_source
        with QSignalBlocker(self.btn_view_fused):
            self.btn_view_fused.setChecked(not showing_source)
        self.histogram_widget.set_current_slice(idx if showing_source else -1)

    def _update_perf_badge(self) -> None:
        short = _METHOD_SHORT_NAMES.get(self._method, self._method)
        w, h = self._proxy_logical_size if self._proxy_logical_size[0] > 0 else self._last_resolution
        if w > 0 and h > 0:
            draft_suffix = f" · {trans.t('live_preview_draft_tag')}" if self._last_is_draft else ""
            self.lbl_perf_badge.setText(
                f"{short} · {self._last_elapsed_ms:.0f} ms · {w}×{h}{draft_suffix}"
            )
        else:
            self.lbl_perf_badge.setText(short)
        self.lbl_perf_badge.adjustSize()
        self.lbl_perf_badge.updateGeometry()

    def _position_loading_overlay(self) -> None:
        if not self.loading_overlay.isVisible():
            return
        self.loading_overlay.adjustSize()
        ow = self.loading_overlay.width()
        oh = self.loading_overlay.height()
        cw = self.canvas_host.width()
        ch = self.canvas_host.height()
        self.loading_overlay.move(max(8, (cw - ow) // 2), max(8, (ch - oh) // 2))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._position_loading_overlay()

    # ------------------------------------------------------------------
    # Worker Orchestration (Latest-Wins)
    # ------------------------------------------------------------------

    def _start_prepare(self, size_key: str) -> None:
        self._preview_size = size_key
        self._session_ready = False
        self._prepare_gen += 1
        self.progress_bar.setRange(0, max(1, self._frame_count))
        self.progress_bar.setValue(0)
        self.lbl_loading.setText(
            trans.t("live_preview_preparing").format(done=0, total=max(1, self._frame_count))
        )
        self.loading_overlay.show()
        self.loading_overlay.raise_()
        self._position_loading_overlay()
        self._worker.request_prepare(self._prepare_gen, size_key)

    def _submit_render(self, *, draft: bool = False) -> None:
        if not self._session_ready or self._closing:
            return
        self._request_seq += 1
        params = dict(self._method_params[self._method])
        self._worker.request_render(
            self._request_seq,
            self._method,
            params,
            self._substack_range,
            draft=bool(draft),
        )

    def _on_slider_drag_started(self) -> None:
        self._is_dragging = True
        self._refine_timer.stop()

    def _on_slider_drag_finished(self) -> None:
        was_dragging = self._is_dragging
        self._is_dragging = False
        self._refine_timer.stop()
        if was_dragging and self._session_ready and not self._closing:
            # Immediately refine to crisp 100% proxy resolution on mouse release.
            self._submit_render(draft=False)

    def _on_refine_timeout(self) -> None:
        if self._session_ready and not self._closing:
            self._submit_render(draft=False)

    def _on_prepare_progress(self, done: int, total: int, stage: str) -> None:
        if self._closing:
            return
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(min(done, max(1, total)))
        key = "live_preview_aligning" if "align" in (stage or "").lower() else "live_preview_preparing"
        self.lbl_loading.setText(trans.t(key).format(done=done, total=max(1, total)))
        self._position_loading_overlay()

    def _on_prepare_ready(self, gen: int, width: int, height: int) -> None:
        if self._closing or gen != self._prepare_gen:
            return
        self._session_ready = True
        self._last_resolution = (width, height)
        self._proxy_logical_size = (width, height)
        self._last_is_draft = False
        self.loading_overlay.hide()
        self._update_perf_badge()
        if self._split_compare:
            self._refresh_split_source_pixmap()
        self._submit_render(draft=False)

    def _on_frame_ready(
        self,
        request_id: int,
        bgr_image: np.ndarray,
        histogram: Optional[np.ndarray],
        elapsed_ms: float,
        width: int,
        height: int,
        method: str,
        is_draft: bool = False,
    ) -> None:
        if self._closing or request_id < self._request_seq:
            # Drop stale frames if a newer request already superseded this one.
            return
        self._fused_pixmap = cv2_to_pixmap(bgr_image)
        self._last_elapsed_ms = float(elapsed_ms)
        self._last_resolution = (int(width), int(height))
        self._last_is_draft = bool(is_draft)
        if not is_draft and width > 0 and height > 0:
            self._proxy_logical_size = (int(width), int(height))
        if histogram is not None:
            self._last_histogram = np.asarray(histogram, dtype=np.float32)
            self.histogram_widget.set_histogram(self._last_histogram)
        self._update_perf_badge()
        if self._view_mode == "fused" and not self._peeking_source:
            self.canvas.set_overlay_badge("")
            logical = self._proxy_logical_size if self._proxy_logical_size[0] > 0 else None
            self.canvas.set_pixmap(self._fused_pixmap, logical_size=logical)

    def _on_worker_failed(self, message: str) -> None:
        if self._closing:
            return
        self.loading_overlay.hide()
        self.lbl_status.setText(trans.t("live_preview_error").format(error=message))

    # ------------------------------------------------------------------
    # User Interaction Slots
    # ------------------------------------------------------------------

    def _on_preview_size_changed(self, _index: int) -> None:
        size_key = self.combo_size.currentData()
        if size_key and size_key != self._preview_size:
            self._start_prepare(str(size_key))

    def _on_zoom_changed(self, zoom: float, is_fit: bool) -> None:
        if is_fit:
            self.btn_zoom_fit.setText(trans.t("live_preview_zoom_fit"))
        else:
            self.btn_zoom_fit.setText(f"{int(round(zoom * 100))}%")

    def _select_method(self, method_id: str) -> None:
        if not method_id or method_id == self._method or method_id not in _METHOD_SHORT_NAMES:
            return
        self._method = str(method_id)
        self._sync_controls_to_state()
        self._view_mode = "fused"
        self._refine_timer.stop()
        self._update_stack_browser_ui()
        self._update_perf_badge()
        self._submit_render(draft=False)

    def _on_method_radio_clicked(self, button: QRadioButton) -> None:
        method_id = button.property("method_id")
        if method_id:
            self._select_method(str(method_id))

    def _on_param_changed(self, key: str, value: Any) -> None:
        params = self._method_params[self._method]
        if params.get(key) == value:
            return
        params[key] = value
        # Return immediately to live fused view when user adjusts a fusion slider.
        if self._view_mode != "fused":
            self._view_mode = "fused"
            self._update_stack_browser_ui()
        if self._is_dragging:
            self._submit_render(draft=True)
            self._refine_timer.start()
        else:
            self._refine_timer.stop()
            self._submit_render(draft=False)

    def _unaligned_source_frame(self, index: int) -> Optional[np.ndarray]:
        """Decode/downscale one unaligned proxy frame on demand without extra stack RAM."""
        if not 0 <= index < len(self._images):
            return None
        cache_key = (self._preview_size, index)
        if self._unaligned_cache[0] == cache_key and self._unaligned_cache[1] is not None:
            return self._unaligned_cache[1]
        try:
            from core.live_preview import PREVIEW_SIZES, _load_single_proxy
            target_edge = PREVIEW_SIZES.get(self._preview_size, 720)
            frame, _shape = _load_single_proxy(self._images[index], target_edge)
            self._unaligned_cache = (cache_key, frame)
            return frame
        except Exception:
            return None

    def _get_source_pixmap(self, index: int) -> Optional[QPixmap]:
        if not self._session_ready or self._frame_count == 0:
            return None
        idx = max(0, min(int(index), self._frame_count - 1))
        frame = self._worker.get_source_frame(idx, aligned=self._show_aligned_source)
        if frame is None and not self._show_aligned_source:
            frame = self._unaligned_source_frame(idx)
        return cv2_to_pixmap(frame) if frame is not None else None

    def _refresh_split_source_pixmap(self) -> None:
        if not self._session_ready or self._frame_count == 0:
            return
        idx = max(0, min(self._browse_index, self._frame_count - 1))
        pix = self._get_source_pixmap(idx)
        left_lbl = trans.t("live_preview_split_fused_label")
        right_lbl = trans.t("live_preview_peek_badge").format(i=idx + 1, n=self._frame_count)
        self.canvas.set_source_pixmap(pix, left_label=left_lbl, right_label=right_lbl)

    def _on_split_compare_toggled(self, checked: bool) -> None:
        self._split_compare = bool(checked)
        if self._split_compare:
            self._view_mode = "fused"
            self._peeking_source = False
            self.canvas.set_overlay_badge("")
            if self._fused_pixmap is not None:
                logical = self._proxy_logical_size if self._proxy_logical_size[0] > 0 else None
                self.canvas.set_pixmap(self._fused_pixmap, logical_size=logical)
            self._refresh_split_source_pixmap()
            self.canvas.set_split_compare(True)
        else:
            self.canvas.set_split_compare(False)
        self._update_stack_browser_ui()

    def _show_source_slice(self, index: int, *, temporary_peek: bool = False) -> None:
        if not self._session_ready or self._frame_count == 0:
            return
        idx = max(0, min(int(index), self._frame_count - 1))
        self._browse_index = idx
        if self._split_compare and not temporary_peek:
            # In split-compare mode, scrubbing source slices updates the right half live!
            self._refresh_split_source_pixmap()
            self._update_stack_browser_ui()
            return
        pix = self._get_source_pixmap(idx)
        if pix is not None:
            if temporary_peek and self._split_compare:
                self.canvas.set_split_compare(False)
            logical = self._proxy_logical_size if self._proxy_logical_size[0] > 0 else None
            self.canvas.set_pixmap(pix, logical_size=logical)
        badge = trans.t("live_preview_peek_badge").format(i=idx + 1, n=self._frame_count)
        self.canvas.set_overlay_badge(badge)
        if not temporary_peek:
            self._view_mode = "source"
        self._update_stack_browser_ui()

    def _switch_to_fused_view(self) -> None:
        self._view_mode = "fused"
        self._peeking_source = False
        self.canvas.set_overlay_badge("")
        if self._split_compare:
            self.canvas.set_split_compare(True)
            self._refresh_split_source_pixmap()
        if self._fused_pixmap is not None:
            logical = self._proxy_logical_size if self._proxy_logical_size[0] > 0 else None
            self.canvas.set_pixmap(self._fused_pixmap, logical_size=logical)
        self._update_stack_browser_ui()

    def _on_stack_slider_changed(self, value: int) -> None:
        self._browse_index = int(value)
        self._show_source_slice(self._browse_index)

    def _step_source_slice(self, delta: int) -> None:
        target = max(0, min(self._frame_count - 1, self._browse_index + delta))
        if target == self._browse_index and self._session_ready:
            self._show_source_slice(target)
            return
        self.stack_slider.setValue(target)

    def _on_show_aligned_toggled(self, checked: bool) -> None:
        self._show_aligned_source = bool(checked)
        if self._split_compare and not self._peeking_source:
            self._refresh_split_source_pixmap()
        elif self._view_mode == "source" or self._peeking_source:
            self._show_source_slice(self._browse_index, temporary_peek=self._peeking_source)

    def _on_histogram_slice_clicked(self, index: int) -> None:
        self._browse_index = max(0, min(self._frame_count - 1, int(index)))
        self._show_source_slice(self._browse_index)

    def _on_substack_range_changed(self, start_idx: int, end_idx: int) -> None:
        new_range = (int(start_idx), int(end_idx))
        if new_range == self._substack_range:
            return
        self._substack_range = new_range
        self._substack_modified = True
        # Keep browsed slice clamped within the active substack for smooth comparison.
        if self._browse_index < start_idx:
            self._browse_index = start_idx
        elif self._browse_index > end_idx:
            self._browse_index = end_idx
        self._update_substack_ui()
        self._update_stack_browser_ui()
        if self._split_compare:
            self._refresh_split_source_pixmap()
        if self._view_mode != "fused":
            self._view_mode = "fused"
            self._update_stack_browser_ui()
        if self._is_dragging:
            self._submit_render(draft=True)
            self._refine_timer.start()
        else:
            self._refine_timer.stop()
            self._submit_render(draft=False)

    def _reset_substack_range(self) -> None:
        full = (0, max(0, self._frame_count - 1))
        if self._substack_range != full:
            self._substack_range = full
            self._substack_modified = True
            self._update_substack_ui()
            self._submit_render(draft=False)

    # ------------------------------------------------------------------
    # Keyboard Shortcuts (A/B/C, ←/→, 0/1, Space peek, Esc close)
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self and event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            key = event.key()
            mods = event.modifiers()
            if key == Qt.Key.Key_Escape and event.type() == QEvent.Type.KeyPress:
                self.close()
                return True

            focus = self.focusWidget()
            from PyQt6.QtWidgets import QAbstractSpinBox, QLineEdit
            if isinstance(focus, (QLineEdit, QAbstractSpinBox)):
                return super().eventFilter(obj, event)

            if key == Qt.Key.Key_Space and not event.isAutoRepeat():
                if event.type() == QEvent.Type.KeyPress:
                    if not self._peeking_source and self._session_ready:
                        self._peeking_source = True
                        self._show_source_slice(self._browse_index, temporary_peek=True)
                    return True
                if self._peeking_source:
                    self._peeking_source = False
                    if self._view_mode == "fused":
                        self._switch_to_fused_view()
                return True

            if event.type() == QEvent.Type.KeyPress and not (
                mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
            ):
                if key == Qt.Key.Key_A:
                    self._select_method("openfocus_a")
                    return True
                if key == Qt.Key.Key_B:
                    self._select_method("openfocus_b")
                    return True
                if key == Qt.Key.Key_C:
                    self._select_method("openfocus_c")
                    return True
                if key == Qt.Key.Key_0:
                    self.canvas.fit_to_window()
                    return True
                if key == Qt.Key.Key_1:
                    self.canvas.set_actual_size()
                    return True
                if key == Qt.Key.Key_Left:
                    self._step_source_slice(-1)
                    return True
                if key == Qt.Key.Key_Right:
                    self._step_source_slice(1)
                    return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Apply Parameters / Enqueue / Apply & Render
    # ------------------------------------------------------------------

    def _sync_to_main_window(self) -> None:
        window = self.window_main
        method = self._method
        params = dict(self._method_params[method])
        substack = self._substack_range
        if window is not None:
            setter_m = getattr(window, "set_fusion_method", None)
            if callable(setter_m):
                setter_m(method)
            setter_p = getattr(window, "set_fusion_parameters", None)
            if callable(setter_p):
                setter_p(method, params)
            # Apply substack range to main window checked/excluded source frames.
            self._apply_substack_to_main_window(window, substack)
            show_status = getattr(window, "_show_status_message", None)
            if callable(show_status):
                show_status(trans.t("live_preview_applied_status"), 4000)
        self.parametersApplied.emit(method, params, substack)

    def _apply_substack_to_main_window(self, window: Any, substack: tuple[int, int]) -> None:
        if not self._substack_modified and substack == self._initial_substack:
            return
        raw = getattr(window, "raw_images", None) or []
        n = len(raw)
        if n != self._frame_count or n < 2:
            return
        start, end = substack
        new_excluded = {i for i in range(n) if i < start or i > end}
        current_excluded = set(getattr(window, "excluded_frames", set()) or set())
        if new_excluded == current_excluded:
            return
        push_undo = getattr(window, "push_undo", None)
        if callable(push_undo):
            push_undo()
        window.excluded_frames = new_excluded
        sm = getattr(window, "source_manager", None)
        if sm is not None:
            changed = getattr(sm, "processing_selection_changed", None)
            if callable(changed):
                changed()
            else:
                decorate = getattr(sm, "decorate_source_items", None)
                if callable(decorate):
                    decorate()

    def _on_apply_parameters(self) -> None:
        self._sync_to_main_window()
        self.lbl_status.setText(trans.t("live_preview_applied_status"))

    def _on_enqueue_render(self) -> None:
        payload = {
            "method": self._method,
            "params": dict(self._method_params[self._method]),
            "substack_range": self._substack_range,
        }
        self.enqueue_requested.emit(payload)
        window = self.window_main
        enq = getattr(window, "enqueue_render_job", None) if window is not None else None
        if callable(enq):
            start, end = self._substack_range
            frame_indices = list(range(start, end + 1))
            try:
                enq(
                    method=payload["method"],
                    align_method=self._alignment_method,
                    method_params=payload["params"],
                    frame_indices=frame_indices,
                )
            except TypeError:
                try:
                    enq(
                        method=payload["method"],
                        params=payload["params"],
                        frame_indices=frame_indices,
                    )
                except TypeError:
                    enq(payload)
        self.lbl_status.setText(trans.t("live_preview_enqueued_status"))

    def _on_apply_and_render(self) -> None:
        self._sync_to_main_window()
        window = self.window_main
        self.accept()
        if window is not None:
            rm = getattr(window, "render_manager", None)
            start_render = getattr(rm, "start_render", None) if rm is not None else None
            if callable(start_render):
                start_render()

    # ------------------------------------------------------------------
    # Shutdown & Memory Release
    # ------------------------------------------------------------------

    def shutdown_running(self) -> bool:
        """Stop worker and release proxy memory; returns True if still winding down."""
        self._closing = True
        if hasattr(self, "_refine_timer"):
            self._refine_timer.stop()
        worker = getattr(self, "_worker", None)
        if worker is not None:
            stopped = worker.stop_and_close(timeout_ms=2500)
            if not stopped and worker.isRunning():
                worker.setParent(None)
                _ORPHAN_PREVIEW_WORKERS.add(worker)
                worker.finished.connect(
                    lambda w=worker: (_ORPHAN_PREVIEW_WORKERS.discard(w), w.deleteLater())
                )
                return True
        self._fused_pixmap = None
        self._images = []
        self._aligned_frames = None
        self._unaligned_cache = (None, None)
        return False

    def closeEvent(self, event) -> None:  # noqa: N802
        self.shutdown_running()
        super().closeEvent(event)

    def reject(self) -> None:
        self.shutdown_running()
        super().reject()

    def accept(self) -> None:
        self.shutdown_running()
        super().accept()
