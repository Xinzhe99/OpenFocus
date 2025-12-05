from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPixmap, QPen
from PyQt6.QtWidgets import QLabel


class MagnifierLabel(QLabel):
    """Image label that supports zooming and a press-and-hold magnifier."""

    def __init__(self, text: Optional[str] = "", parent=None):
        super().__init__(parent)
        if text:
            self.setText(text)
        self._magnifier_active = False
        self._magnifier_pos = QPointF()
        self._magnifier_size = 160  # pixels
        self._magnifier_zoom = 2.0
        self.setMouseTracking(True)
        self._base_pixmap: Optional[QPixmap] = None
        self._zoom_factor = 1.0
        self._min_zoom = 0.2
        self._max_zoom = 5.0

    def set_magnifier_settings(self, size=None, zoom=None):
        if size is not None and size > 0:
            self._magnifier_size = size
        if zoom is not None and zoom > 0:
            self._magnifier_zoom = zoom

    def set_display_pixmap(self, pixmap: Optional[QPixmap]):
        self._base_pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            super().clear()
            self._magnifier_active = False
            self.update()
            return
        self.setText("")
        self._update_scaled_pixmap()

    def refresh_display(self):
        self._update_scaled_pixmap()

    def reset_view(self):
        self._zoom_factor = 1.0
        if self._base_pixmap is None or self._base_pixmap.isNull():
            super().clear()
            self.update()
            return
        self._update_scaled_pixmap()

    def clear(self):
        self._base_pixmap = None
        self._zoom_factor = 1.0
        self._magnifier_active = False
        self.unsetCursor()
        super().clear()

    def wheelEvent(self, event):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            super().wheelEvent(event)
            return

        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return

        step = 1.1
        if delta > 0:
            self._zoom_factor = min(self._zoom_factor * step, self._max_zoom)
        else:
            self._zoom_factor = max(self._zoom_factor / step, self._min_zoom)

        self._update_scaled_pixmap()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._base_pixmap is not None and not self._base_pixmap.isNull():
            self._update_scaled_pixmap()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            if self._base_pixmap is not None and not self._base_pixmap.isNull():
                self.reset_view()
                event.accept()
                return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.pixmap() is not None
            and not self.pixmap().isNull()
        ):
            self._magnifier_active = True
            self._magnifier_pos = event.position()
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._magnifier_active:
            self._magnifier_pos = event.position()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            if self._base_pixmap is not None and not self._base_pixmap.isNull():
                self.reset_view()
                event.accept()
                return

        if event.button() == Qt.MouseButton.LeftButton and self._magnifier_active:
            self._magnifier_active = False
            self.unsetCursor()
            self.update()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if self._magnifier_active:
            self._magnifier_active = False
            self.unsetCursor()
            self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self._magnifier_active:
            return

        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        pix_rect = self._scaled_pixmap_rect(pixmap)
        if pix_rect.isNull() or not pix_rect.contains(self._magnifier_pos):
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        relative_x = self._magnifier_pos.x() - pix_rect.x()
        relative_y = self._magnifier_pos.y() - pix_rect.y()

        src_size = self._magnifier_size / self._magnifier_zoom
        half_src = src_size / 2.0

        max_x = pixmap.width() - src_size
        max_y = pixmap.height() - src_size

        src_left = max(0.0, min(relative_x - half_src, max_x)) if max_x > 0 else 0.0
        src_top = max(0.0, min(relative_y - half_src, max_y)) if max_y > 0 else 0.0

        source_rect = QRectF(src_left, src_top, min(src_size, pixmap.width()), min(src_size, pixmap.height()))

        target_half = self._magnifier_size / 2.0
        target_rect = QRectF(
            self._magnifier_pos.x() - target_half,
            self._magnifier_pos.y() - target_half,
            self._magnifier_size,
            self._magnifier_size,
        )

        painter.drawPixmap(target_rect, pixmap, source_rect)

        border_color = QColor(255, 255, 255, 220)
        painter.setPen(QPen(border_color, 2))
        painter.drawRect(target_rect)

    def _update_scaled_pixmap(self):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            super().clear()
            self.update()
            return

        label_width = max(1, self.width())
        label_height = max(1, self.height())

        target_width = int(label_width * self._zoom_factor)
        target_height = int(label_height * self._zoom_factor)

        scaled = self._base_pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)
        self.update()

    def _scaled_pixmap_rect(self, pixmap: QPixmap) -> QRectF:
        if pixmap is None or pixmap.isNull():
            return QRectF()

        pm_width = float(pixmap.width())
        pm_height = float(pixmap.height())

        x_offset = (self.width() - pm_width) / 2.0
        y_offset = (self.height() - pm_height) / 2.0

        return QRectF(x_offset, y_offset, pm_width, pm_height)
