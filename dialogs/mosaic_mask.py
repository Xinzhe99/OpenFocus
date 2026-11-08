"""Thumbnail-only source mask editing; the parent applies accepted native strokes."""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap, QTransform
from PyQt6.QtWidgets import (
    QButtonGroup, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QVBoxLayout,
)

from dialogs.mosaic_control_points import _SourcePreview
from locales import trans
from utils.file_dialogs import FileDialog
from utils.image_utils import to_display_uint8


class _BrushPreview(_SourcePreview):
    strokeRecorded = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.radius = 50.0
        self.include = False
        self._painting = False
        self._last = None
        self.setMouseTracking(True)

    def _record(self, point):
        # One notification and preview repaint per mouse event; the receiver
        # records native circles without materializing an intermediate array.
        self.strokeRecorded.emit((self._last, point, float(self.radius), bool(self.include)))
        self._last = point

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.native_position(event.position())
            if point is not None:
                self._painting = True
                self._last = None
                self._record(point)
                event.accept()
                return
        # Do not emit the CP view's pointClicked for brush input.
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._painting:
            point = self.native_position(event.position())
            if point is None:
                self._last = None
            else:
                self._record(point)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._painting:
            point = self.native_position(event.position())
            if point is not None and point != self._last:
                self._record(point)
            self._painting = False
            self._last = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MosaicMaskDialog(QDialog):
    """Stage a pending mask without touching source pixels or a full-size mask.

    `strokes` contains (native_x, native_y, native_radius, include_bool).
    `clear_base` discards the original/imported base before those strokes.
    mask previews are bounded, single-channel opacity: zero excludes, full
    dtype scale includes. The parent decodes and validates imports on its worker,
    retains that pending decoded base, then calls set_mask_preview on success or
    set_import_failed on error. Only an Accepted result commits either base or
    strokes; Cancel discards them all. `mask_path` identifies a successful import.
    """

    maskImportRequested = pyqtSignal(str)
    cancelRequested = pyqtSignal()

    def __init__(self, preview, native_shape, mask_preview=None, parent=None):
        super().__init__(parent)
        self.native_shape = tuple(int(value) for value in native_shape[:2])
        self.strokes = []
        self.clear_base = False
        self.mask_path = None
        self._requested_path = None
        self._import_pending = False
        self._close_pending = False
        self.resize(900, 650)
        outer = QVBoxLayout(self)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        outer.addWidget(self.hint)
        self.canvas = _BrushPreview()
        self.canvas.set_preview(preview, self.native_shape)
        outer.addWidget(self.canvas, 1)
        self._preview_shape = tuple(preview.shape[:2])
        self._overlay_item = self.canvas.scene().addPixmap(QPixmap())
        self._overlay_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._overlay_item.setPos(-0.5, -0.5)
        height, width = self.native_shape
        ph, pw = self._preview_shape
        self._overlay_item.setTransform(QTransform.fromScale(width / pw, height / ph))
        self._overlay_item.setZValue(1)
        row = QHBoxLayout()
        self.exclude = QRadioButton()
        self.include = QRadioButton()
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.exclude)
        self.mode_group.addButton(self.include)
        self.exclude.setChecked(True)
        self.include.toggled.connect(self._brush_mode)
        row.addWidget(self.exclude)
        row.addWidget(self.include)
        self.radius_label = QLabel()
        row.addWidget(self.radius_label)
        self.radius = QDoubleSpinBox()
        self.radius.setDecimals(3)
        self.radius.setRange(0.5, max(height, width))
        self.radius.setValue(min(50.0, max(height, width)))
        self.radius.setKeyboardTracking(False)
        self.radius.valueChanged.connect(self._brush_radius)
        self.radius_label.setBuddy(self.radius)
        row.addWidget(self.radius)
        row.addStretch()
        self.import_button = QPushButton()
        self.import_button.clicked.connect(self._import_mask)
        self.clear_button = QPushButton()
        self.clear_button.clicked.connect(self._clear)
        for button in (self.import_button, self.clear_button):
            button.setAutoDefault(False)
            row.addWidget(button)
        outer.addLayout(row)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)
        self.canvas.strokeRecorded.connect(self._stroke)
        trans.languageChanged.connect(self._retranslate)
        self._retranslate()
        self.set_mask_preview(mask_preview)
        self._brush_radius(self.radius.value())

    def _retranslate(self):
        self.setWindowTitle(trans.t('mosaic_mask'))
        self.hint.setText(trans.t('mosaic_mask_hint'))
        self.exclude.setText(trans.t('mosaic_mask_exclude'))
        self.include.setText(trans.t('mosaic_mask_include'))
        self.radius_label.setText(trans.t('mosaic_brush_radius'))
        self.import_button.setText(trans.t('mosaic_mask_import'))
        self.clear_button.setText(trans.t('mosaic_mask_clear'))
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).setText(trans.t('mosaic_apply'))

    def _brush_mode(self, checked):
        self.canvas.include = bool(checked)
        self.canvas._last = None

    def _brush_radius(self, radius):
        self.canvas.radius = float(radius)
        self.canvas._last = None

    def _new_overlay(self):
        height, width = self._preview_shape
        image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        return image

    def set_mask_preview(self, preview):
        """Publish a decoded pending base, never the source-resolution array.

        Supply a preview no larger than the source thumbnail. Import shape and
        PNG/TIFF decoding are parent worker responsibilities, before this call.
        """
        overlay = self._new_overlay()
        if preview is not None:
            if preview.ndim != 2:
                raise ValueError('A mask preview must be single-channel opacity')
            if preview.shape[0] > self._preview_shape[0] or preview.shape[1] > self._preview_shape[1]:
                raise ValueError('A mask preview must not exceed the source thumbnail dimensions')
            if preview.dtype.kind == 'b':
                opacity = preview.astype(np.uint8) * 255
            else:
                opacity = to_display_uint8(preview)
            alpha = np.ascontiguousarray(((255 - opacity).astype(np.uint16) * 110 // 255).astype(np.uint8))
            height, width = alpha.shape
            alpha_image = QImage(alpha.data, width, height, alpha.strides[0], QImage.Format.Format_Grayscale8)
            colored = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
            colored.fill(QColor(255, 0, 0, 255))
            colored.setAlphaChannel(alpha_image)
            painter = QPainter(overlay)
            painter.drawImage(QRectF(0, 0, overlay.width(), overlay.height()), colored)
            painter.end()
        self._overlay = overlay
        self.strokes.clear()
        self.clear_base = False
        if self._requested_path is not None:
            self.mask_path = self._requested_path
        self._requested_path = None
        self._import_pending = False
        self._overlay_item.setPixmap(QPixmap.fromImage(self._overlay))
        self._enable_controls()
        self._finish_close()

    def set_import_failed(self):
        """Re-enable the unchanged pending edit after the parent shows its error."""
        self._requested_path = None
        self._import_pending = False
        self._enable_controls()
        self._finish_close()

    def _stroke(self, segment):
        if self._import_pending:
            return
        previous, point, radius, include = segment
        if previous is None:
            positions = (point,)
        else:
            dx, dy = point[0] - previous[0], point[1] - previous[1]
            count = max(1, math.ceil(math.hypot(dx, dy) / max(0.25, radius / 2)))
            positions = ((previous[0] + dx * step / count, previous[1] + dy * step / count)
                         for step in range(1, count + 1))
        ph, pw = self._preview_shape
        height, width = self.native_shape
        sx, sy = pw / width, ph / height
        painter = QPainter(self._overlay)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        # Source composition replaces opacity instead of accumulating red.
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.setBrush(QColor(0, 0, 0, 0) if include else QColor(255, 0, 0, 110))
        for x, y in positions:
            self.strokes.append((float(x), float(y), radius, include))
            painter.drawEllipse(QRectF((x + 0.5 - radius) * sx, (y + 0.5 - radius) * sy,
                                      radius * 2 * sx, radius * 2 * sy))
        painter.end()
        self._overlay_item.setPixmap(QPixmap.fromImage(self._overlay))

    def _clear(self):
        if self._import_pending:
            return
        self.strokes.clear()
        self.clear_base = True
        self.mask_path = None
        self._overlay = self._new_overlay()
        self._overlay_item.setPixmap(QPixmap.fromImage(self._overlay))

    def _import_mask(self):
        path, _ = FileDialog.getOpenFileName(self, trans.t('mosaic_mask_import'), '', '*.png *.tif *.tiff')
        if path:
            self._requested_path = path
            self._import_pending = True
            self._enable_controls()
            self.maskImportRequested.emit(path)

    def _enable_controls(self):
        for widget in (self.canvas, self.exclude, self.include, self.radius, self.import_button, self.clear_button,
                       self.buttons.button(QDialogButtonBox.StandardButton.Apply)):
            widget.setEnabled(not self._import_pending)

    def accept(self):
        if not self._import_pending:
            super().accept()

    def _finish_close(self):
        if self._close_pending and not self._import_pending:
            self._close_pending = False
            super().reject()

    def _defer_close(self):
        if self._import_pending:
            if not self._close_pending:
                self._close_pending = True
                self.cancelRequested.emit()
            return True
        return False

    def reject(self):
        if not self._defer_close():
            super().reject()

    def closeEvent(self, event):
        if self._defer_close():
            event.ignore()
        else:
            super().closeEvent(event)
