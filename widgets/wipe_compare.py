"""A/B wipe comparison widget: two images in one frame with a draggable divider.

Both images share the same zoom/pan transform, so misalignments across the
divider are directly visible. Wheel zooms around the cursor, dragging pans,
dragging the divider (or the handle) changes the split position.
"""
from typing import Optional

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget

DIVIDER_HIT_RADIUS = 10
ZOOM_STEP = 1.15
ZOOM_MIN = 0.05
ZOOM_MAX = 20.0


class WipeCompareWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._left_pixmap: Optional[QPixmap] = None
        self._right_pixmap: Optional[QPixmap] = None
        self._left_title: str = ""
        self._right_title: str = ""
        self._divider_ratio: float = 0.5  # divider x as a fraction of widget width
        self._zoom: float = 1.0
        self._pan = QPointF(0, 0)
        self._drag_mode: str = ""  # "", "divider", "pan"
        self._last_pos = QPointF(0, 0)
        self.setMouseTracking(True)

    # --- public API ---

    def set_images(self, left: Optional[QPixmap], right: Optional[QPixmap],
                   left_title: str = "", right_title: str = "") -> None:
        self._left_pixmap = left
        self._right_pixmap = right
        self._left_title = left_title
        self._right_title = right_title
        self.update()

    def set_titles(self, left_title: str, right_title: str) -> None:
        self._left_title = left_title
        self._right_title = right_title
        self.update()

    def has_images(self) -> bool:
        return self._left_pixmap is not None and self._right_pixmap is not None

    def clear(self) -> None:
        self._left_pixmap = None
        self._right_pixmap = None
        self._left_title = ""
        self._right_title = ""
        self.reset_view()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._divider_ratio = 0.5
        self.update()

    # --- geometry ---

    def _target_rect(self) -> QRectF:
        """Fitted, centered, zoomed and panned rect of the (left) base image."""
        pixmap = self._left_pixmap if self._left_pixmap is not None else self._right_pixmap
        if pixmap is None or pixmap.isNull():
            return QRectF()

        w, h = float(self.width()), float(self.height())
        # Fit the pixmap inside the widget with a small margin
        scale = min((w - 20) / pixmap.width(), (h - 20) / pixmap.height())
        scale = max(0.0001, scale * self._zoom)
        draw_w = pixmap.width() * scale
        draw_h = pixmap.height() * scale
        base_x = (w - draw_w) / 2.0 + self._pan.x()
        base_y = (h - draw_h) / 2.0 + self._pan.y()
        return QRectF(base_x, base_y, draw_w, draw_h)

    def _divider_x(self) -> float:
        return self._divider_ratio * self.width()

    # --- painting ---

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        from ui import styles as _styles
        canvas = "#ffffff" if getattr(_styles, "CURRENT_THEME", "dark") == "light" else "#181818"
        painter.setBackground(QColor(canvas))
        painter.eraseRect(self.rect())

        if not self.has_images():
            painter.setPen(QColor("#666"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._left_title)
            return

        target = self._target_rect()
        divider_x = self._divider_x()

        for pixmap, clip_rect in (
            (self._left_pixmap, QRect(0, 0, int(divider_x), self.height())),
            (self._right_pixmap, QRect(int(divider_x), 0, self.width() - int(divider_x), self.height())),
        ):
            if pixmap is None or pixmap.isNull() or clip_rect.width() <= 0:
                continue
            painter.save()
            painter.setClipRect(clip_rect)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
            painter.restore()

        # Divider line and handle
        pen = QPen(QColor("#ffffff"), 2)
        painter.setPen(pen)
        painter.drawLine(QPointF(divider_x, 0), QPointF(divider_x, self.height()))
        handle_y = self.height() / 2.0
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QPointF(divider_x, handle_y), 7, 7)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(divider_x, handle_y), 11, 11)

        self._draw_title_chip(painter, self._left_title, left=True)
        self._draw_title_chip(painter, self._right_title, left=False)

    @staticmethod
    def _pixmap_bounds(pixmap: QPixmap) -> QRectF:
        return QRectF(pixmap.rect())

    def _draw_title_chip(self, painter: QPainter, text: str, left: bool) -> None:
        if not text:
            return
        # Match the app-wide UI font (set by apply_dark_theme) so the chips
        # render CJK text exactly like the rest of the interface
        painter.setFont(QApplication.instance().font())
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(text) + 16
        text_h = metrics.height() + 6
        x = 8 if left else self.width() - text_w - 8
        chip = QRect(int(x), 8, text_w, text_h)
        painter.fillRect(chip, QColor(0, 0, 0, 150))
        painter.setPen(QColor("#dddddd"))
        painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, text)

    # --- interaction ---

    def wheelEvent(self, event) -> None:  # noqa: N802
        if not self.has_images():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = ZOOM_STEP if delta > 0 else 1.0 / ZOOM_STEP
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self._zoom * factor))

        # Zoom around the mouse cursor: keep the image point under it fixed.
        #
        # Scaling _pan by the zoom ratio is not sufficient, because the
        # drawn rect is also centred by (widget - scaled)/2, which changes
        # with the zoom. Compute the fraction under the cursor, apply the
        # zoom, then shift the pan until that fraction is back under the
        # cursor (measured up to 13% of the image width of drift otherwise
        # when zooming out).
        mouse = QPointF(event.position())
        before = self._target_rect()
        if before.width() > 0 and before.height() > 0:
            fx = (mouse.x() - before.x()) / before.width()
            fy = (mouse.y() - before.y()) / before.height()
            anchored = 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0
        else:
            anchored = False
        self._zoom = new_zoom
        if anchored:
            after = self._target_rect()
            self._pan = QPointF(
                self._pan.x() + (mouse.x() - (after.x() + fx * after.width())),
                self._pan.y() + (mouse.y() - (after.y() + fy * after.height())),
            )
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            if abs(pos.x() - self._divider_x()) <= DIVIDER_HIT_RADIUS:
                self._drag_mode = "divider"
            elif self.has_images():
                self._drag_mode = "pan"
                self._last_pos = pos
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()

        if self._drag_mode == "divider":
            self._divider_ratio = max(0.02, min(0.98, pos.x() / max(1.0, self.width())))
            self.update()
        elif self._drag_mode == "pan":
            delta = pos - self._last_pos
            self._pan = QPointF(self._pan.x() + delta.x(), self._pan.y() + delta.y())
            self._last_pos = pos
            self.update()
        else:
            near_divider = abs(pos.x() - self._divider_x()) <= DIVIDER_HIT_RADIUS
            self.setCursor(
                Qt.CursorShape.SizeHorCursor if near_divider else Qt.CursorShape.OpenHandCursor
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_mode = ""
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_view()
        super().mouseDoubleClickEvent(event)
