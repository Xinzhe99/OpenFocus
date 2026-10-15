"""Retouch dialog: brush pixels from a source frame onto the fused result.

Left-drag paints from the selected source frame; with "auto" on, each
stroke first picks the frame that is sharpest under the brush, so fixing a
halo is a single drag. Wheel zooms, right/middle-drag pans. The edit is
applied as a new entry in the output history (the original stays).
Core logic (dab blending, tile undo, sharpest-frame pick) is in
utils/retouch.py.
"""
import os

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QGraphicsEllipseItem, QGraphicsItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout,
    QWidget,
)

from locales import trans
from utils.image_utils import to_display_uint8
from utils.retouch import TileUndo, apply_dab, sharpest_frame_at


def _bgr_to_qimage(bgr8):
    bgr8 = np.ascontiguousarray(bgr8)
    if bgr8.ndim == 2:
        h, w = bgr8.shape
        return QImage(bgr8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    h, w = bgr8.shape[:2]
    return QImage(bgr8.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()


class _CanvasItem(QGraphicsItem):
    """Pixmap item whose pixmap is patched in place (dirty rects only):
    re-uploading a 24 MP pixmap per dab would stall the brush."""

    def __init__(self, pixmap):
        super().__init__()
        self.pix = pixmap

    def boundingRect(self):
        return QRectF(0, 0, self.pix.width(), self.pix.height())

    def paint(self, painter, option, widget=None):
        painter.drawPixmap(0, 0, self.pix)

    def patch(self, x, y, bgr8):
        p = QPainter(self.pix)
        p.drawImage(x, y, _bgr_to_qimage(bgr8))
        p.end()
        self.update(QRectF(x, y, bgr8.shape[1], bgr8.shape[0]))

    def replace(self, pixmap):
        self.pix = pixmap
        self.update()


class _View(QGraphicsView):
    def __init__(self, dialog):
        super().__init__()
        self.dlg = dialog
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor(24, 24, 24)))
        self._pan_from = None
        self.cursor_item = QGraphicsEllipseItem()
        pen = QPen(QColor(255, 255, 255, 220))
        pen.setCosmetic(True)
        self.cursor_item.setPen(pen)
        self.cursor_item.setZValue(10)

    def wheelEvent(self, event):
        f = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def _scene_pos(self, event):
        return self.mapToScene(event.position().toPoint())

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_from = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.dlg.stroke_begin(self._scene_pos(event))

    def mouseMoveEvent(self, event):
        pos = self._scene_pos(event)
        r = self.dlg.brush_radius()
        self.cursor_item.setRect(pos.x() - r, pos.y() - r, 2 * r, 2 * r)
        if self._pan_from is not None:
            d = event.position() - self._pan_from
            self._pan_from = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.dlg.stroke_to(pos)

    def mouseReleaseEvent(self, event):
        if self._pan_from is not None and event.button() in (
                Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_from = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.dlg.stroke_end()


class RetouchDialog(QDialog):
    def __init__(self, window_main, result, frames, names=None, parent=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self.setWindowTitle(trans.t("retouch_title"))
        self.setModal(True)
        self.resize(1280, 820)
        self.result = np.ascontiguousarray(result.copy())
        self.frames = frames
        self.names = list(names or [])
        self.undo = TileUndo()
        self._stroke_alpha = None
        self._last_pos = None
        self._edited = False
        self._showing_source = False
        self._build_ui()
        self._apply_style()
        self._show_result()
        self._update_frame_label()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        root = QHBoxLayout(self)
        self.view = _View(self)
        self.view.setCursor(Qt.CursorShape.CrossCursor)
        root.addWidget(self.view, 1)

        side = QWidget()
        side.setFixedWidth(290)
        col = QVBoxLayout(side)
        col.setContentsMargins(6, 0, 0, 0)

        intro = QLabel(trans.t("retouch_intro"))
        intro.setWordWrap(True)
        col.addWidget(intro)

        col.addSpacing(6)
        self.lbl_frame = QLabel()
        self.lbl_frame.setWordWrap(True)
        col.addWidget(self.lbl_frame)
        self.slider_frame = QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.setRange(0, max(0, len(self.frames) - 1))
        self.slider_frame.setValue(len(self.frames) - 1)
        self.slider_frame.valueChanged.connect(self._on_frame_changed)
        col.addWidget(self.slider_frame)

        self.chk_auto = QCheckBox(trans.t("retouch_auto"))
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(trans.t("retouch_auto_hint"))
        col.addWidget(self.chk_auto)

        self.chk_source = QCheckBox(trans.t("retouch_show_source"))
        self.chk_source.setToolTip(trans.t("retouch_show_source_hint"))
        self.chk_source.toggled.connect(self._toggle_source_view)
        col.addWidget(self.chk_source)

        col.addSpacing(8)
        self.lbl_size = QLabel()
        col.addWidget(self.lbl_size)
        short = min(self.result.shape[:2])
        self.slider_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_size.setRange(2, max(8, short // 4))
        self.slider_size.setValue(max(4, min(short // 25, 120)))
        self.slider_size.valueChanged.connect(self._update_labels)
        col.addWidget(self.slider_size)

        self.lbl_hard = QLabel()
        col.addWidget(self.lbl_hard)
        self.slider_hard = QSlider(Qt.Orientation.Horizontal)
        self.slider_hard.setRange(0, 95)
        self.slider_hard.setValue(40)
        self.slider_hard.valueChanged.connect(self._update_labels)
        col.addWidget(self.slider_hard)

        self.lbl_opacity = QLabel()
        col.addWidget(self.lbl_opacity)
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(5, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.valueChanged.connect(self._update_labels)
        col.addWidget(self.slider_opacity)

        col.addSpacing(8)
        row = QHBoxLayout()
        self.btn_undo = QPushButton(trans.t("retouch_undo"))
        self.btn_undo.clicked.connect(self._undo)
        self.btn_redo = QPushButton(trans.t("retouch_redo"))
        self.btn_redo.clicked.connect(self._redo)
        row.addWidget(self.btn_undo)
        row.addWidget(self.btn_redo)
        col.addLayout(row)
        self.btn_fit = QPushButton(trans.t("retouch_fit"))
        self.btn_fit.clicked.connect(self._fit)
        col.addWidget(self.btn_fit)

        keys = QLabel(trans.t("retouch_keys"))
        keys.setWordWrap(True)
        keys.setStyleSheet("color: #8a8a8a; font-size: 11px;")
        col.addSpacing(6)
        col.addWidget(keys)
        col.addStretch()

        bottom = QHBoxLayout()
        self.btn_apply = QPushButton(trans.t("retouch_apply"))
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self.accept)
        btn_cancel = QPushButton(trans.t("btn_cancel"))
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(self.btn_apply)
        bottom.addWidget(btn_cancel)
        col.addLayout(bottom)
        root.addWidget(side)

        for seq, fn in ((QKeySequence.StandardKey.Undo, self._undo),
                        (QKeySequence.StandardKey.Redo, self._redo)):
            QShortcut(QKeySequence(seq), self, activated=fn)
        for key, fn in (("[", lambda: self._bump_size(0.8)),
                        ("]", lambda: self._bump_size(1.25)),
                        (",", lambda: self.slider_frame.setValue(self.slider_frame.value() - 1)),
                        (".", lambda: self.slider_frame.setValue(self.slider_frame.value() + 1)),
                        ("Tab", lambda: self.chk_source.toggle()),
                        ("F", self._fit)):
            QShortcut(QKeySequence(key), self, activated=fn)
        self._update_labels()
        self._update_buttons()

    def _apply_style(self):
        try:
            if self.window_main and getattr(self.window_main, "current_theme", "dark") != "light":
                from ui.styles import DIALOG_CONTROL_DARK_QSS
                self.setStyleSheet(
                    "QDialog { background: #2b2b2b; } QLabel { color: #d0d0d0; }"
                    + DIALOG_CONTROL_DARK_QSS)
        except Exception:
            pass

    # ------------------------------------------------------------- state
    def brush_radius(self):
        return int(self.slider_size.value())

    def _bump_size(self, f):
        self.slider_size.setValue(int(round(self.slider_size.value() * f)) +
                                  (1 if f > 1 else 0))

    def _update_labels(self):
        self.lbl_size.setText(trans.t("retouch_size").format(self.slider_size.value()))
        self.lbl_hard.setText(trans.t("retouch_hardness").format(self.slider_hard.value()))
        self.lbl_opacity.setText(trans.t("retouch_opacity").format(self.slider_opacity.value()))

    def _update_frame_label(self):
        i = self.slider_frame.value()
        name = self.names[i] if i < len(self.names) else ""
        self.lbl_frame.setText(trans.t("retouch_frame").format(
            i=i + 1, n=len(self.frames), name=os.path.basename(str(name))))

    def _update_buttons(self):
        self.btn_undo.setEnabled(self.undo.can_undo)
        self.btn_redo.setEnabled(self.undo.can_redo)

    def _on_frame_changed(self, _v):
        self._update_frame_label()
        if self._showing_source:
            self._show_image(self.frames[self.slider_frame.value()])

    # ----------------------------------------------------------- display
    def _show_image(self, img):
        pix = QPixmap.fromImage(_bgr_to_qimage(to_display_uint8(img)))
        scene = self.view.scene()
        if not hasattr(self, "canvas"):
            self.canvas = _CanvasItem(pix)
            scene.addItem(self.canvas)
            scene.addItem(self.view.cursor_item)
            scene.setSceneRect(self.canvas.boundingRect())
        else:
            self.canvas.replace(pix)

    def _show_result(self):
        self._show_image(self.result)

    def _toggle_source_view(self, on):
        self._showing_source = bool(on)
        if on:
            self._show_image(self.frames[self.slider_frame.value()])
        else:
            self._show_result()

    def _fit(self):
        self.view.fitInView(self.canvas.boundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit()

    # ------------------------------------------------------------ strokes
    def stroke_begin(self, pos: QPointF):
        if self._showing_source:
            self.chk_source.setChecked(False)
        if self.chk_auto.isChecked():
            idx = sharpest_frame_at(self.frames, pos.x(), pos.y(),
                                    max(8, self.brush_radius()))
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(idx)
            self.slider_frame.blockSignals(False)
            self._update_frame_label()
        self.undo.begin()
        self._stroke_alpha = np.zeros(self.result.shape[:2], np.float32)
        self._last_pos = pos
        self._dab(pos)

    def stroke_to(self, pos: QPointF):
        if self._last_pos is None:
            return
        # interpolate dabs so fast drags leave no gaps
        step = max(1.0, self.brush_radius() * 0.3)
        d = pos - self._last_pos
        dist = (d.x() ** 2 + d.y() ** 2) ** 0.5
        n = int(dist // step)
        for k in range(1, n + 1):
            self._dab(self._last_pos + d * (k / max(1, n)))
        if n:
            self._last_pos = pos

    def stroke_end(self):
        if self._last_pos is None:
            return
        self.undo.end()
        self._stroke_alpha = None
        self._last_pos = None
        self._update_buttons()

    def _dab(self, pos):
        rect = apply_dab(self.result, self.frames[self.slider_frame.value()],
                         pos.x(), pos.y(), self.brush_radius(),
                         self.slider_hard.value() / 100.0,
                         self.slider_opacity.value() / 100.0,
                         stroke_alpha=self._stroke_alpha, undo=self.undo)
        if rect is not None:
            self._edited = True
            self._repaint_rect(rect)

    def _repaint_rect(self, rect):
        x0, y0, x1, y1 = rect
        self.canvas.patch(x0, y0, to_display_uint8(self.result[y0:y1, x0:x1]))

    def _undo(self):
        if self.undo.undo(self.result):
            self._show_result()
        self._update_buttons()

    def _redo(self):
        if self.undo.redo(self.result):
            self._show_result()
        self._update_buttons()

    def edited_result(self):
        return self.result if self._edited and self.undo.can_undo else None
