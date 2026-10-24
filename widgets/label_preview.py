"""Live preview of a label configuration.

The add-label dialog used to be a blind form: you picked a font size, an
x/y offset and a format string, confirmed, and only then saw whether the
label landed inside the frame, how big it was relative to a real frame, or
whether it showed anything at all. Several combinations drew nothing (see
utils/validators.build_label_text), and nothing in the UI said so.

This widget renders one sample frame with the current settings applied, so
the answer is visible before the dialog closes.
"""
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

from utils.validators import LabelAdder, build_label_text


class LabelPreview(QLabel):
    """A scaled preview of a source frame carrying one label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 190)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Ignored)
        self.setObjectName("labelPreview")
        self._adder = LabelAdder()
        self._frame = None
        self._status = ""
        # Family actually used when the request could not be honoured.
        self.substituted_font = ""
        # Which frame the preview pretends to be, so {value} resolves the way
        # the first real frame will.
        self._index = 0
        self._zoom_mode = "fit"
        self._rendered = QPixmap()
        self._pan = (0, 0)
        self._drag_start = None

    def set_preview(self, frame, config, index):
        """Update all dependent state before drawing once."""
        self._frame = None if frame is None else np.asarray(frame)
        self._adder.config.update_config(config)
        self._index = max(0, int(index))
        self.refresh()

    def set_zoom_mode(self, mode):
        self._zoom_mode = mode
        self._pan = (0, 0)
        self._show_rendered()

    def _show_rendered(self):
        if self._rendered.isNull():
            self.clear()
            return
        width, height = max(1, self.width() - 8), max(1, self.height() - 8)
        if self._zoom_mode == "actual":
            x = max(0, min(self._pan[0], max(0, self._rendered.width() - width)))
            y = max(0, min(self._pan[1], max(0, self._rendered.height() - height)))
            self._pan = (x, y)
            pix = self._rendered.copy(x, y, min(width, self._rendered.width()),
                                      min(height, self._rendered.height()))
        else:
            pix = self._rendered.scaled(width, height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pix)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._zoom_mode == "actual":
            self._drag_start = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            delta = event.position() - self._drag_start
            self._pan = (self._pan[0] - int(delta.x()), self._pan[1] - int(delta.y()))
            self._drag_start = event.position()
            self._show_rendered()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------ data
    def set_frame(self, frame) -> None:
        """Use a real frame as the preview canvas (falls back to a placeholder)."""
        self._frame = None if frame is None else np.asarray(frame)
        self.refresh()

    def set_index(self, index: int) -> None:
        self._index = max(0, int(index))
        self.refresh()

    def set_config(self, config: dict) -> None:
        self._adder.config.update_config(config)
        self.refresh()

    def message(self) -> str:
        """Why the preview shows no label, for the dialog's status line."""
        return self._status

    def draw_error(self) -> str:
        """Non-empty when the last draw attempt failed outright."""
        return self._adder.last_error

    # ------------------------------------------------------------------ draw
    def _preview_value(self) -> int:
        """The number this preview's frame would carry.

        The steppers hand back floats, so both terms are rounded: without it
        "{value}" rendered as "1.0" in the preview while the real export drew
        "1" (str() of the substituted value).
        """
        start = int(round(float(self._adder.config.starting_value)))
        step = int(round(float(self._adder.config.interval)))
        return start + self._index * step

    def _canvas(self):
        if self._frame is not None and self._frame.size:
            frame = self._frame
            # Label sizes are absolute pixels, so a 24 MP frame and a 320 px
            # preview must use the same *frame* to look the same. Sample the
            # centre at native scale instead of rescaling the whole image.
            return frame
        # Placeholder: mid grey with a faint grid, so white labels are legible.
        canvas = np.full((360, 540, 3), 70, np.uint8)
        for x in range(0, 540, 60):
            canvas[:, x] = 88
        for y in range(0, 360, 60):
            canvas[y, :] = 88
        return canvas

    def refresh(self) -> None:
        from utils.image_utils import to_display_uint8

        canvas = self._canvas()
        config = self._adder.config
        text = build_label_text(config.format, config.text,
                                self._preview_value())
        # Draw first: the status must describe *this* attempt, not the
        # previous one (out-of-range and draw failures are per-frame).
        try:
            drawn = self._adder.add_label_to_image(canvas, self._index)
            self.substituted_font = self._adder.last_substituted_font
            error = self._adder.last_error
            out_of_range = self._adder.last_out_of_range
        except Exception:
            drawn = canvas
            self.substituted_font = ""
            error = "draw_failed"
            out_of_range = False

        if error:
            self._status = "error"
        elif not text:
            self._status = "empty"
        elif out_of_range:
            self._status = "out_of_range"
        else:
            self._status = ""

        drawn = to_display_uint8(drawn)
        if drawn.ndim == 2:
            drawn = np.repeat(drawn[:, :, None], 3, axis=2)
        h, w = drawn.shape[:2]
        rgb = np.ascontiguousarray(drawn[:, :, ::-1])
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._rendered = QPixmap.fromImage(image)
        bounds = self._adder.last_bounds
        if bounds is not None:
            painter = QPainter(self._rendered)
            pen = QPen(QColor("#40c8d5"))
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(max(1, min(w, h) // 300))
            painter.setPen(pen)
            x, y, text_width, text_height = bounds
            painter.drawRect(x, y, text_width, text_height)
            painter.drawEllipse(x - 3, y + text_height - 3, 6, 6)
            painter.end()
        self._show_rendered()

    def resizeEvent(self, event):  # noqa: N802 (Qt casing)
        super().resizeEvent(event)
        self._show_rendered()
