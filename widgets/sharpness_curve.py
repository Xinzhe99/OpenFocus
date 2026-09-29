"""Sharpness curve widget: per-frame focus quality drawn above the source slider.

Displays a normalized Laplacian-variance curve over the frame stack so users
can spot out-of-focus or duplicated frames at a glance and jump to them.
"""
from typing import List, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

OUTLIER_RATIO = 0.35  # frames below this fraction of the max are flagged


class SharpnessCurveWidget(QWidget):
    frame_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._values: List[float] = []
        self._current = -1
        self.setMinimumHeight(34)
        self.setMaximumHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("")
        self._tooltip_base = ""

    def set_tooltip_base(self, text: str) -> None:
        self._tooltip_base = text
        self.setToolTip(text)

    def set_data(self, values: List[float]) -> None:
        self._values = [max(0.0, float(v)) for v in values]
        self.setVisible(len(self._values) >= 2)
        self.update()

    def set_current(self, index: int) -> None:
        self._current = index
        self.update()

    def clear(self) -> None:
        self._values = []
        self._current = -1
        self.setVisible(False)
        self.update()

    def _index_from_x(self, x: float) -> int:
        n = len(self._values)
        if n < 2 or self.width() <= 0:
            return 0
        return max(0, min(n - 1, int(round(x / self.width() * (n - 1)))))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        self.frame_clicked.emit(self._index_from_x(pos.x()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        n = len(self._values)
        if n:
            idx = self._index_from_x(event.position().x())
            extra = f" — #{idx + 1}" if self._tooltip_base else f"#{idx + 1}"
            self.setToolTip(self._tooltip_base + extra)
        super().mouseMoveEvent(event)

    def _theme_colors(self):
        """(line, axis, text, alert) for the theme the app actually applies.

        The dark theme is a QSS skin: its background-color never reaches the
        widget palette, so testing palette().color(backgroundRole()) reports
        "light" while the app is dark and the curve text/axis came out
        unreadable. CURRENT_THEME is what apply_theme() installed; the palette
        (whose text role *is* synced in light mode) stays the fallback.
        """
        from ui import styles

        theme = getattr(styles, "CURRENT_THEME", None)
        if theme in ("light", "dark"):
            light = theme == "light"
        else:
            foreground = self.palette().color(self.foregroundRole())
            light = foreground.lightness() < 128

        if light:
            return QColor("#0969da"), QColor("#c8cfd6"), QColor("#57606a"), QColor("#d1242f")
        return QColor("#4da3ff"), QColor("#6b6b6b"), QColor("#c9d1d9"), QColor("#ff6a6a")

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        line, axis, text, alert = self._theme_colors()

        painter.setPen(QPen(axis, 1))
        painter.drawLine(QPointF(0, h - 4), QPointF(w, h - 4))

        n = len(self._values)
        if n < 2:
            return
        vmax = max(self._values) or 1.0

        flagged = []
        poly = QPolygonF()
        for i, v in enumerate(self._values):
            x = i / (n - 1) * (w - 2) + 1
            y = (h - 8) * (1.0 - v / vmax) + 4
            poly.append(QPointF(x, y))
            if v < vmax * OUTLIER_RATIO:
                flagged.append((x, y))

        painter.setPen(QPen(line, 1.6))
        painter.drawPolyline(poly)

        if flagged and n >= 4:
            painter.setPen(QPen(alert, 1))
            painter.setBrush(alert)
            for x, y in flagged:
                painter.drawEllipse(QPointF(x, y), 2.6, 2.6)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        if 0 <= self._current < n:
            x = self._current / (n - 1) * (w - 2) + 1
            painter.setPen(QPen(text, 1))
            painter.drawLine(QPointF(x, 0), QPointF(x, h - 4))
