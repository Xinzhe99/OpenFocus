"""Source selection and exact result crops for the comparison workbench."""

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QTransform
from PyQt6.QtWidgets import QWidget

from locales import trans


DEFAULT_DETAIL_SIZE = 340


class _Canvas(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title

    def _background(self, painter):
        # Importing widgets must not initialize the application's UI package.
        from ui.styles import CURRENT_THEME, tokens

        theme = tokens(CURRENT_THEME)
        painter.fillRect(self.rect(), QColor(theme["surfaceSunken"]))
        painter.setPen(QColor(theme["border"]))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        return theme

    def _label(self, painter, theme):
        metrics = painter.fontMetrics()
        width = min(self.width() - 12,
                    metrics.horizontalAdvance(self._title) + 16)
        bar = QRect(self.width() - width - 6, 6, width,
                    max(22, metrics.height() + 4))
        painter.fillRect(bar, QColor(theme["accent"]))
        painter.setPen(QColor(theme["textOnAccent"]))
        title = metrics.elidedText(self._title, Qt.TextElideMode.ElideRight,
                                   max(0, width - 16))
        painter.drawText(bar, Qt.AlignmentFlag.AlignCenter, title)

    def _empty(self, painter, theme, status=""):
        painter.setPen(QColor(theme["textMuted"]))
        painter.drawText(self.rect().adjusted(12, 36, -12, -12),
                         Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                         status or trans.t("compare_no_image"))


class SourceCanvas(_Canvas):
    """A fitted source image with a persistent, integer inspection square."""

    regionChanged = pyqtSignal(object)

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self._pixmap = None
        self._region = None
        self._region_centre = None
        self._magnification = 1.0
        self._detail_size = DEFAULT_DETAIL_SIZE
        self._zoom = 1.0
        self._pan = QPointF()
        self._fit_pending = False
        self._drag_region = False
        self._pan_from = None
        self.setMinimumSize(180, 130)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def has_image(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    def set_pixmap(self, pixmap, reset=False):
        old_size = self._pixmap.size() if self.has_image() else None
        self._pixmap = pixmap
        if not self.has_image():
            self._fit_pending = False
            self._drag_region = False
            self._pan_from = None
            self._set_region(None)
        elif reset or old_size != pixmap.size():
            self._fit_pending = True
            self.set_region_centre(QPointF(pixmap.width() / 2,
                                          pixmap.height() / 2))
            if self.isVisible():
                self.fit()
        self.update()

    def region(self):
        """Return (x, y, width, height) in aligned-image pixels, or None."""
        return self._region

    def _set_region(self, region):
        if region is None:
            self._region_centre = None
        if region != self._region:
            self._region = region
            self.update()
            self.regionChanged.emit(region)

    def set_region_centre(self, point: QPointF):
        if not self.has_image():
            return
        width, height = self._pixmap.width(), self._pixmap.height()
        side = min(width, height,
                   max(1, round(self._detail_size / self._magnification)))
        half = side / 2
        # Keep the subpixel centre independently of the integer crop. Reusing
        # the rounded crop's centre accumulates drift while a slider moves.
        self._region_centre = QPointF(
            max(half, min(width - half, point.x())),
            max(half, min(height - half, point.y())))
        x = round(self._region_centre.x() - half)
        y = round(self._region_centre.y() - half)
        self._set_region((x, y, side, side))

    def set_detail_size(self, size: int):
        """Resize the inspected extent while retaining its centre and zoom."""
        self._detail_size = max(1, int(size))
        self.set_magnification(self._magnification)

    def set_magnification(self, factor: float):
        self._magnification = max(0.25, min(8.0, float(factor)))
        if self._region_centre is not None:
            self.set_region_centre(self._region_centre)

    def magnification(self) -> float:
        return self._magnification

    def fit(self):
        if not self.has_image():
            return
        self._zoom = min(max(1, self.width() - 8) / self._pixmap.width(),
                         max(1, self.height() - 8) / self._pixmap.height(), 1.0)
        self._pan = QPointF()
        self._fit_pending = not self.isVisible()
        self.update()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if self._fit_pending:
            self.fit()

    def _image_transform(self):
        if not self.has_image():
            return QTransform()
        x = (self.width() - self._pixmap.width() * self._zoom) / 2
        y = (self.height() - self._pixmap.height() * self._zoom) / 2
        return QTransform(self._zoom, 0, 0, self._zoom,
                          x + self._pan.x(), y + self._pan.y())

    def image_to_widget(self, point: QPointF):
        return self._image_transform().map(point)

    def widget_to_image(self, point: QPointF):
        inverse, _ = self._image_transform().inverted()
        return inverse.map(point)

    def mousePressEvent(self, event):  # noqa: N802
        if not self.has_image():
            return super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_region = True
            self.set_region_centre(self.widget_to_image(event.position()))
            event.accept()
        elif event.button() in (Qt.MouseButton.RightButton,
                                Qt.MouseButton.MiddleButton):
            self._pan_from = QPointF(event.position())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_region and self.has_image():
            self.set_region_centre(self.widget_to_image(event.position()))
            event.accept()
        elif self._pan_from is not None:
            self._pan += event.position() - self._pan_from
            self._pan_from = QPointF(event.position())
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._drag_region:
            self.set_region_centre(self.widget_to_image(event.position()))
            self._drag_region = False
            event.accept()
        elif (event.button() in (Qt.MouseButton.RightButton,
                                 Qt.MouseButton.MiddleButton)
              and self._pan_from is not None):
            self._pan += event.position() - self._pan_from
            self._pan_from = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.update()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):  # noqa: N802
        delta = event.angleDelta().y()
        if not self.has_image() or not delta:
            event.ignore()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            anchor = self.widget_to_image(event.position())
            factor = 1.15 if delta > 0 else 1 / 1.15
            self._zoom = max(0.000001, min(12.0, self._zoom * factor))
            self._pan += event.position() - self.image_to_widget(anchor)
            self.update()
        else:
            factor = 1.2 if delta > 0 else 1 / 1.2
            self.set_magnification(self._magnification * factor)
        event.accept()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        theme = self._background(painter)
        if self.has_image():
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                                  self._zoom < 1.0)
            transform = self._image_transform()
            source = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
            painter.drawPixmap(transform.mapRect(source), self._pixmap, source)
            if self._region is not None:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(transform.mapRect(QRectF(*self._region)))
        else:
            self._empty(painter, theme)
        self._label(painter, theme)
        painter.end()


class DetailCanvas(_Canvas):
    """Display only the requested aligned-image rectangle from stored pixels."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self._pixmap = None
        self._bounds = None
        self._region = None
        self._status = ""
        # "detail" crops to the inspection rectangle; "full" shows the whole
        # computed result. Computing a full frame and only ever displaying a
        # crop read as "the full frame was not really rendered".
        self._view_mode = "detail"
        self._selected = False
        self.setFixedSize(DEFAULT_DETAIL_SIZE, DEFAULT_DETAIL_SIZE)

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self.update()

    def is_selected(self) -> bool:
        return self._selected

    def set_view_mode(self, mode: str) -> None:
        self._view_mode = "full" if mode == "full" else "detail"
        self.update()

    def view_mode(self) -> str:
        return self._view_mode

    def can_show_full(self) -> bool:
        """True when the stored pixmap IS the whole computed frame.

        ROI results carry only a crop, so offering the overview for them would
        present a zoomed crop as the complete image.
        """
        if self._pixmap is None or self._pixmap.isNull():
            return False
        if self._bounds is None:
            return False
        x, y, width, height = self._bounds
        return (x == 0 and y == 0
                and width == self._pixmap.width()
                and height == self._pixmap.height())

    def set_result(self, pixmap, bounds):
        """Store pixels and their (x, y, width, height) image-space bounds."""
        self._pixmap = pixmap
        self._bounds = tuple(bounds) if bounds is not None else None
        self.update()

    def set_region(self, region):
        self._region = tuple(region) if region is not None else None
        self.update()

    def set_status(self, text):
        self._status = text or ""
        self.update()

    def has_region(self) -> bool:
        """Whether stored pixels cover the entire current inspection region."""
        if (self._pixmap is None or self._pixmap.isNull()
                or self._bounds is None or self._region is None):
            return False
        x, y, width, height = self._region
        bx, by, bw, bh = self._bounds
        return (width > 0 and height > 0 and x >= bx and y >= by
                and x + width <= bx + min(bw, self._pixmap.width())
                and y + height <= by + min(bh, self._pixmap.height()))

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        theme = self._background(painter)
        if self._view_mode == "full" and self.can_show_full():
            # Whole computed result, letterboxed to keep its aspect ratio.
            source = QRectF(0, 0, self._pixmap.width(), self._pixmap.height())
            area = QRectF(self.rect())
            scale = min(area.width() / max(1.0, source.width()),
                        area.height() / max(1.0, source.height()))
            width = source.width() * scale
            height = source.height() * scale
            target = QRectF(area.center().x() - width / 2.0,
                            area.center().y() - height / 2.0, width, height)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(target, self._pixmap, source)
        elif self.has_region():
            x, y, width, height = self._region
            bx, by, _, _ = self._bounds
            source = QRectF(x - bx, y - by, width, height)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(QRectF(self.rect()), self._pixmap, source)
        else:
            self._empty(painter, theme, self._status)
        if self._selected:
            # A visible border is the only selection cue; the title bar alone
            # is easy to miss among several equally sized cards.
            painter.setPen(QPen(QColor(theme["accent"]), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect().adjusted(1, 1, -2, -2))
        self._label(painter, theme)
        painter.end()
