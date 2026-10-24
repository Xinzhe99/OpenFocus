"""Retouch workspace: compare a render's source frames with its result.

The left canvas is read-only; the right canvas is the only writable one.
Both share one zoom/center state, so the same image coordinate always sits
at the same screen position.  Every stroke is locked to the source frame
and brush settings that were active when the pointer went down.

The dialog publishes through `apply_callback(record, image)`: it never
mutates the output list itself.  A failed publish keeps the buffer and the
dialog open.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PyQt6.QtCore import (QEvent, QPointF, QRect, QRectF, QSize, Qt, QThread,
                          QTimer, pyqtSignal)
from PyQt6.QtGui import (QBrush, QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap)
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QDialog,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from locales import trans
from utils.image_utils import thumbnail_bgr, to_display_uint8
from utils.retouch import RetouchDocument

_SHARPNESS_WORKERS = set()


def _qimage_from_bgr(bgr8: np.ndarray, copy: bool = True) -> QImage:
    arr = np.ascontiguousarray(bgr8)
    if arr.ndim == 2:
        h, w = arr.shape
        image = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w = arr.shape[:2]
        image = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_BGR888)
    return image.copy() if copy else image


class _ViewState:
    """Shared image coordinate -> screen coordinate state for both canvases."""

    def __init__(self) -> None:
        self.scale = 1.0
        self.center = QPointF(0.0, 0.0)
        self.fit = True


class _NumberRow(QWidget):
    """A labelled slider plus a numeric input, kept in sync."""

    valueChanged = pyqtSignal(int)

    def __init__(self, label: str, minimum: int, maximum: int, value: int,
                 parent=None):
        super().__init__(parent)
        self._sync = False
        self._label = QLabel(label)
        self._label.setWordWrap(True)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.spin = QSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setKeyboardTracking(False)
        # The stylesheet reserves 22 px on the trailing edge for the stepper;
        # 78 px left three-digit values running underneath it.
        self.spin.setFixedWidth(108)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._label)
        layout.addLayout(row)
        self.setValue(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> int:
        return int(self.spin.value())

    def setValue(self, value: int) -> None:
        self._sync = True
        self.slider.setValue(int(value))
        self.spin.setValue(int(value))
        self._sync = False

    def set_label(self, text: str) -> None:
        self._label.setText(text)

    def _from_slider(self, value: int) -> None:
        if self._sync:
            return
        self._sync = True
        self.spin.setValue(value)
        self._sync = False
        self.valueChanged.emit(int(value))

    def _from_spin(self, value: int) -> None:
        if self._sync:
            return
        self._sync = True
        self.slider.setValue(value)
        self._sync = False
        self.valueChanged.emit(int(value))


class _SharpnessWorker(QThread):
    """Compute one shared-scale detail map for every source frame."""

    ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, frames, request_id: int, preview_side: int = 768):
        super().__init__()
        self.frames = list(frames)
        self.request_id = int(request_id)
        self.preview_side = int(preview_side)

    def run(self) -> None:
        try:
            detail = {}
            for index, frame in enumerate(self.frames):
                if self.isInterruptionRequested():
                    return
                image = frame.image
                if image is None or image.size == 0:
                    continue
                h, w = image.shape[:2]
                scale = min(1.0, self.preview_side / float(max(h, w, 1)))
                if scale < 1.0:
                    image = cv2.resize(
                        image, (max(1, int(round(w * scale))),
                                max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_AREA)
                display = to_display_uint8(image)
                if display.ndim == 2:
                    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
                energy = cv2.GaussianBlur(
                    np.abs(lap), (0, 0), max(1.0, display.shape[1] / 300.0))
                detail[index] = energy
            if not detail:
                self.ready.emit(self.request_id, {})
                return
            peak = max(float(np.percentile(m, 99.5))
                       for m in detail.values())
            peak = max(1e-6, peak)
            maps = {}
            for index, energy in detail.items():
                if self.isInterruptionRequested():
                    return
                norm = np.clip(energy / peak, 0.0, 1.0)
                maps[index] = cv2.applyColorMap(
                    (norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            self.ready.emit(self.request_id, maps)
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))


class _RetouchCanvas(QWidget):
    """One synchronized image canvas with a brush cursor."""

    def __init__(self, dialog: "RetouchDialog", role: str):
        super().__init__()
        self.dlg = dialog
        self.role = role
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMinimumSize(180, 180)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._pan_from = None
        self._drawing = False

    # ------------------------------------------------------------ coords
    def image_rect(self, width: int, height: int,
                   offset_x: int = 0, offset_y: int = 0) -> QRectF:
        s = self.dlg.view.scale
        cx = self.dlg.view.center.x()
        cy = self.dlg.view.center.y()
        x = self.width() / 2.0 - cx * s + offset_x * s
        y = self.height() / 2.0 - cy * s + offset_y * s
        return QRectF(x, y, width * s, height * s)

    def widget_rect_for_image_rect(self, rect, offset_x=0, offset_y=0) -> QRect:
        x0, y0, x1, y1 = rect
        s = self.dlg.view.scale
        cx = self.dlg.view.center.x()
        cy = self.dlg.view.center.y()
        bx = self.width() / 2.0 - cx * s + offset_x * s
        by = self.height() / 2.0 - cy * s + offset_y * s
        return QRect(int(bx + x0 * s) - 2, int(by + y0 * s) - 2,
                     max(1, int((x1 - x0) * s)) + 4,
                     max(1, int((y1 - y0) * s)) + 4)

    def widget_to_image(self, pos: QPointF) -> QPointF:
        s = max(1e-9, self.dlg.view.scale)
        cx = self.dlg.view.center.x()
        cy = self.dlg.view.center.y()
        return QPointF((pos.x() - self.width() / 2.0) / s + cx,
                       (pos.y() - self.height() / 2.0) / s + cy)

    def image_to_widget(self, x: float, y: float,
                        offset_x=0, offset_y=0) -> QPointF:
        rect = self.image_rect(1, 1, offset_x, offset_y)
        return QPointF(rect.x() + x * self.dlg.view.scale,
                       rect.y() + y * self.dlg.view.scale)

    # ------------------------------------------------------------ paint
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(22, 22, 22))
        qimage, width, height, ox, oy = self.dlg.canvas_image(self.role)
        if qimage is not None and width > 0 and height > 0:
            rect = self.image_rect(width, height, ox, oy)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                                  True)
            painter.drawImage(rect, qimage)
            painter.setPen(QPen(QColor(72, 72, 72), 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
            if self.role == "source":
                domain = QPainterPath()
                domain.addRect(self.image_rect(self.dlg.doc.width, self.dlg.doc.height))
                covered = QPainterPath()
                covered.addRect(rect)
                painter.fillPath(domain.subtracted(covered),
                    QBrush(QColor(160, 160, 160, 150), Qt.BrushStyle.DiagCrossPattern))
        self._paint_cursor(painter)
        painter.end()

    def _paint_cursor(self, painter: QPainter) -> None:
        pos = self.dlg.cursor_pos
        if pos is None or self.dlg.preview_active:
            return
        if not self.dlg.image_contains(pos.x(), pos.y()):
            return
        if self.role == "source" and not self.dlg.source_frame_contains(
                pos.x(), pos.y()):
            return
        frame = self.dlg.doc.active_frame
        ox, oy = (frame.offset_x, frame.offset_y) if self.role == "source" else (0, 0)
        # The cursor is expressed in result coordinates; `image_to_widget`
        # expects frame-local coordinates plus the frame offset.
        center = self.image_to_widget(pos.x() - ox, pos.y() - oy, ox, oy)
        diameter = max(2.0, self.dlg.doc.diameter * self.dlg.view.scale)
        valid = self.dlg.cursor_source_valid()
        color = QColor(255, 88, 88, 240) if not valid else QColor(255, 255, 255, 235)
        pen = QPen(color, 1.4)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, diameter / 2.0, diameter / 2.0)
        inner = diameter * float(np.clip(self.dlg.doc.hardness, 0.0, 1.0))
        if inner > 1.0:
            inner_pen = QPen(color, 1.0, Qt.PenStyle.DashLine)
            inner_pen.setCosmetic(True)
            painter.setPen(inner_pen)
            painter.drawEllipse(center, inner / 2.0, inner / 2.0)

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = event.position()
        pan = (event.button() in (Qt.MouseButton.RightButton,
                                  Qt.MouseButton.MiddleButton)
               or (event.button() == Qt.MouseButton.LeftButton
                   and self.dlg.space_down))
        if pan:
            if self.dlg.doc.stroke_active:
                return
            self._pan_from = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            image_pos = self.widget_to_image(pos)
            self.dlg.set_cursor(image_pos.x(), image_pos.y())
            if self.role == "result":
                if self.dlg.begin_stroke(image_pos.x(), image_pos.y()):
                    self._drawing = True

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        image_pos = self.widget_to_image(pos)
        self.dlg.set_cursor(image_pos.x(), image_pos.y())
        if self._pan_from is not None:
            delta = pos - self._pan_from
            self._pan_from = pos
            self.dlg.pan_by(delta.x(), delta.y())
            return
        if self._drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            self.dlg.stroke_to(image_pos.x(), image_pos.y())

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_from is not None:
            self._pan_from = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            # A release can arrive without a preceding move at the same
            # position; explicitly include it so the last segment is not lost.
            pos = self.widget_to_image(event.position())
            self.dlg.stroke_to(pos.x(), pos.y())
            self._drawing = False
            self.dlg.end_stroke()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.dlg.zoom_at(self, event.position(), factor)
        event.accept()

    def leaveEvent(self, event) -> None:
        if not self._drawing and self._pan_from is None:
            self.dlg.clear_cursor()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.dlg.canvas_resized()


class RetouchDialog(QDialog):
    """The retouch workspace described by docs/RETOUCH_SPEC.md."""

    def __init__(self, window_main, *, record, apply_callback, parent=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self.record = record
        self.apply_callback = apply_callback
        self.doc = RetouchDocument(record.image, record.context)
        self.view = _ViewState()
        self.view.center = QPointF(self.doc.width / 2.0, self.doc.height / 2.0)

        self.cursor_pos = None
        self.space_down = False
        self.preview_mode = None
        self._preview_owner = None
        self._applied = False
        self._applying = False
        self._allow_close = False
        self._closing = False
        self.change_requested = False
        self._syncing_frames = False
        self._syncing_tools = False
        self._source_ready = False
        self._source_request = 0
        self._source_qimage = None
        self._pure_source_qimage = None
        self._source_display = None
        self._before_qimage = None
        self._result_display = None
        self._result_qimage = None
        self._sharpness_maps = {}
        self._sharpness_worker = None
        self._sharpness_started = False
        self._sharpness_request = 0
        self._notice = ""
        self._notice_timer = None
        self._eviction_announced = False
        self._layout_user_choice = False
        self._responsive_focus = False
        self._was_maximized = False
        self._normal_geometry = None
        self._stroke_lock_widgets = []

        self.setWindowTitle(trans.t("retouch_title"))
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setMinimumSize(900, 600)
        self.resize(1280, 820)
        self._build_ui()
        self._install_filters()
        self._restore_geometry()
        self._populate_frames()
        self._refresh_before_display()
        self._refresh_result_display()
        self._select_frame(self.doc.frame_index, initial=True)
        self._update_titles()
        self._update_brush_labels()
        self._update_controls()
        self._focus_signal = QApplication.instance().focusChanged
        self._focus_signal.connect(self._on_focus_changed)
        self.finished.connect(lambda _result: self._save_geometry())
        self.finished.connect(self._shutdown_sharpness)

    # ------------------------------------------------------------- build
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 4)
        outer.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(8)
        self.lbl_title = QLabel()
        font = self.lbl_title.font()
        font.setBold(True)
        self.lbl_title.setFont(font)
        self.lbl_title.setSizePolicy(QSizePolicy.Policy.Ignored,
                                     QSizePolicy.Policy.Preferred)
        # Compact provenance line: the source domain must stay visible even
        # in focus mode, where the source canvas is hidden.
        self.lbl_source_summary = QLabel("")
        self.lbl_source_summary.setObjectName("mutedLabel")
        self.lbl_source_summary.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_unsaved = QLabel()
        self.lbl_unsaved.setObjectName("mutedLabel")
        self.btn_change = QPushButton(trans.t("retouch_change_result"))
        self.btn_change.clicked.connect(self._change_result)
        self.btn_fullscreen = QPushButton(trans.t("compare_fullscreen"))
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(0)
        title_column.addWidget(self.lbl_title)
        title_column.addWidget(self.lbl_source_summary)
        top.addLayout(title_column, 1)
        top.addWidget(self.lbl_unsaved)
        top.addWidget(self.btn_change)
        top.addWidget(self.btn_fullscreen)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        self.btn_undo = QPushButton(trans.t("retouch_undo"))
        self.btn_undo.clicked.connect(self._undo)
        self.btn_redo = QPushButton(trans.t("retouch_redo"))
        self.btn_redo.clicked.connect(self._redo)
        self.btn_fit = QPushButton(trans.t("retouch_fit_short"))
        self.btn_fit.setToolTip(trans.t("retouch_fit"))
        self.btn_fit.clicked.connect(self._fit)
        self.btn_100 = QPushButton(trans.t("retouch_zoom_100"))
        self.btn_100.clicked.connect(self._zoom_100)
        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setMinimumWidth(48)
        self.btn_layout = QPushButton(trans.t("retouch_layout_focus_short"))
        self.btn_layout.setToolTip(trans.t("retouch_layout_focus"))
        self.btn_layout.clicked.connect(
            lambda: self._set_focus_layout(self.left_panel.isVisible(), True))
        self.btn_before = QPushButton(trans.t("retouch_before_short"))
        self.btn_before.setToolTip(trans.t("retouch_view_before"))
        self.btn_before.pressed.connect(lambda: self._press_preview("before"))
        self.btn_before.released.connect(lambda: self._release_preview("before"))
        self.btn_peek = QPushButton(trans.t("retouch_peek_short"))
        self.btn_peek.setToolTip(trans.t("retouch_peek_source"))
        self.btn_peek.pressed.connect(lambda: self._press_preview("source"))
        self.btn_peek.released.connect(lambda: self._release_preview("source"))
        for widget in (self.btn_undo, self.btn_redo, self.btn_fit,
                       self.btn_100, self.btn_layout, self.btn_before,
                       self.btn_peek):
            toolbar.addWidget(widget)
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        toolbar.addWidget(divider)
        toolbar.addWidget(self.lbl_zoom)
        toolbar.addStretch(1)
        toolbar.addWidget(self.btn_peek)
        toolbar.addWidget(self.btn_before)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(3)
        self.lbl_source_header = QLabel(trans.t("retouch_stack"))
        self.lbl_source_header.setObjectName("mutedLabel")
        self.lbl_source_header.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.source_canvas = _RetouchCanvas(self, "source")
        left_layout.addWidget(self.lbl_source_header)
        left_layout.addWidget(self.source_canvas, 1)
        self.splitter.addWidget(self.left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(3)
        self.lbl_result_header = QLabel(trans.t("retouch_result_title"))
        self.lbl_result_header.setObjectName("mutedLabel")
        self.lbl_result_header.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.result_canvas = _RetouchCanvas(self, "result")
        right_layout.addWidget(self.lbl_result_header)
        right_layout.addWidget(self.result_canvas, 1)
        self.splitter.addWidget(right_panel)

        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.side_scroll.setMinimumWidth(240)
        self.side_scroll.setMaximumWidth(360)
        self.side_panel = QWidget()
        side = QVBoxLayout(self.side_panel)
        # Leave room for the vertical scrollbar: with widgetResizable and
        # the scrollbar hidden until needed, a cramped window used to clip
        # the right edge of every numeric input.
        side.setContentsMargins(8, 0, 18, 0)
        side.setSpacing(6)

        self.lbl_help = QLabel(trans.t("retouch_help"))
        self.lbl_help.setWordWrap(True)
        self.lbl_help.setObjectName("mutedLabel")
        self.lbl_help.setToolTip(trans.t("retouch_shortcuts"))
        side.addWidget(self.lbl_help)
        # The full key list is reachable as a visible entry, not only by
        # hovering the help paragraph.
        self.btn_shortcuts = QPushButton(trans.t("retouch_shortcuts_help"))
        self.btn_shortcuts.setAutoDefault(False)
        self.btn_shortcuts.clicked.connect(self._show_shortcuts)
        side.addWidget(self.btn_shortcuts)

        lbl_tools = QLabel(trans.t("retouch_tools"))
        font = lbl_tools.font()
        font.setBold(True)
        lbl_tools.setFont(font)
        side.addWidget(lbl_tools)
        self.rb_source = QRadioButton(trans.t("retouch_tool_source"))
        self.rb_restore = QRadioButton(trans.t("retouch_tool_restore"))
        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.rb_source, 0)
        self.tool_group.addButton(self.rb_restore, 1)
        self.rb_source.toggled.connect(
            lambda on: self._set_tool("source") if on else None)
        self.rb_restore.toggled.connect(
            lambda on: self._set_tool("restore") if on else None)
        side.addWidget(self.rb_source)
        side.addWidget(self.rb_restore)

        self.num_diameter = _NumberRow(trans.t("retouch_diameter"),
                                       1, 2048, self.doc.diameter)
        self.num_hardness = _NumberRow("", 0, 100,
                                       int(round(self.doc.hardness * 100)))
        self.num_opacity = _NumberRow("", 1, 100,
                                      int(round(self.doc.opacity * 100)))
        for control in (self.num_diameter, self.num_hardness, self.num_opacity):
            control.valueChanged.connect(self._on_brush_changed)
            side.addWidget(control)

        self.chk_sharp = QCheckBox(trans.t("retouch_show_sharpness"))
        self.chk_sharp.setToolTip(trans.t("retouch_sharpness_legend"))
        self.chk_sharp.toggled.connect(self._toggle_sharpness)
        side.addWidget(self.chk_sharp)
        self.lbl_sharp_state = QLabel("")
        self.lbl_sharp_state.setObjectName("mutedLabel")
        self.lbl_sharp_state.setWordWrap(True)
        self.lbl_sharp_state.setVisible(False)
        side.addWidget(self.lbl_sharp_state)
        self.lbl_heat_legend = QLabel(trans.t("retouch_sharpness_legend"))
        self.lbl_heat_legend.setWordWrap(True)
        self.lbl_heat_legend.setObjectName("mutedLabel")
        self.lbl_heat_legend.setVisible(False)
        side.addWidget(self.lbl_heat_legend)

        self.btn_restore_all = QPushButton(trans.t("retouch_restore_all"))
        self.btn_restore_all.clicked.connect(self._restore_all)
        side.addWidget(self.btn_restore_all)
        side.addStretch(1)

        self.lbl_apply_hint = QLabel(trans.t("retouch_apply_hint"))
        self.lbl_apply_hint.setWordWrap(True)
        self.lbl_apply_hint.setObjectName("mutedLabel")
        self.btn_apply = QPushButton(trans.t("retouch_apply"))
        self.btn_apply.setObjectName("primaryButton")
        # Only the explicit shortcut may apply: an implicit default button
        # would let a plain Enter publish while the visible button is disabled.
        self.btn_apply.setAutoDefault(False)
        self.btn_apply.setDefault(False)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_cancel = QPushButton(trans.t("btn_cancel"))
        self.btn_cancel.setAutoDefault(False)
        self.btn_cancel.clicked.connect(self.reject)

        self.side_scroll.setWidget(self.side_panel)
        self.side_container = QWidget()
        side_outer = QVBoxLayout(self.side_container)
        side_outer.setContentsMargins(0, 0, 0, 0)
        side_outer.setSpacing(4)
        side_outer.addWidget(self.side_scroll, 1)
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(8, 0, 8, 0)
        footer_layout.setSpacing(4)
        footer_layout.addWidget(self.lbl_apply_hint)
        footer_layout.addWidget(self.btn_apply)
        footer_layout.addWidget(self.btn_cancel)
        side_outer.addWidget(footer)
        self.side_container.setMinimumWidth(240)
        self.side_container.setMaximumWidth(360)
        self.splitter.addWidget(self.side_container)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([460, 460, 272])

        strip = QWidget()
        strip_layout = QVBoxLayout(strip)
        strip_layout.setContentsMargins(4, 2, 4, 2)
        strip_layout.setSpacing(2)
        strip_row = QHBoxLayout()
        strip_row.setSpacing(4)
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(32)
        self.btn_prev.clicked.connect(self._prev_frame)
        self.frame_list = QListWidget()
        self.frame_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.frame_list.setFlow(QListWidget.Flow.LeftToRight)
        self.frame_list.setWrapping(False)
        self.frame_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.frame_list.setMovement(QListWidget.Movement.Static)
        self.frame_list.setIconSize(QSize(64, 64))
        self.frame_list.setGridSize(QSize(80, 84))
        self.frame_list.setFixedHeight(84)
        self.frame_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection)
        self.frame_list.currentRowChanged.connect(self._on_frame_selected)
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(32)
        self.btn_next.clicked.connect(self._next_frame)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setKeyboardTracking(False)
        # Room for a two-digit frame number and the " / N" suffix.
        self.frame_spin.setFixedWidth(126)
        self.frame_spin.valueChanged.connect(self._on_frame_spin)
        strip_row.addWidget(self.btn_prev)
        strip_row.addWidget(self.frame_list, 1)
        strip_row.addWidget(self.btn_next)
        strip_row.addWidget(self.frame_spin)
        strip_layout.addLayout(strip_row)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(1)
        self.frame_slider.setToolTip(trans.t("retouch_stack_hint"))
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        strip_layout.addWidget(self.frame_slider)
        strip.setFixedHeight(112)

        status = QHBoxLayout()
        status.setContentsMargins(6, 0, 6, 0)
        self.lbl_zoom_status = QLabel("100%")
        self.lbl_coords = QLabel("")
        self.lbl_frame_status = QLabel("")
        self.lbl_state = QLabel("")
        self.lbl_state.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Preferred)
        status.addWidget(self.lbl_zoom_status)
        status.addWidget(self.lbl_coords)
        status.addWidget(self.lbl_frame_status, 1)
        status.addWidget(self.lbl_state)

        outer.addLayout(top)
        outer.addLayout(toolbar)
        outer.addWidget(self.splitter, 1)
        outer.addWidget(strip)
        outer.addLayout(status)

        self._stroke_lock_widgets = [
            self.btn_undo, self.btn_redo, self.btn_fit, self.btn_100,
            self.btn_layout, self.btn_before, self.btn_peek,
            self.frame_list, self.frame_spin, self.frame_slider,
            self.btn_prev, self.btn_next,
            self.rb_source, self.rb_restore, self.num_diameter,
            self.num_hardness, self.num_opacity, self.chk_sharp,
            self.btn_restore_all, self.btn_apply, self.btn_change,
            self.btn_fullscreen,
        ]

    # ---------------------------------------------------------- lifecycle
    def _restore_geometry(self) -> None:
        from utils.settings_store import get_settings
        settings = get_settings()
        geometry = settings.value("retouch/geometry")
        restored = False
        if geometry is not None:
            restored = self.restoreGeometry(geometry)
        if not restored:
            screen = self.screen() or QApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                self.resize(min(1280, max(900, available.width() - 80)),
                            min(820, max(600, available.height() - 100)))
        state = settings.value("retouch/splitter")
        if state is not None:
            self.splitter.restoreState(state)

    def _save_geometry(self) -> None:
        try:
            from utils.settings_store import get_settings
            settings = get_settings()
            if not self.isFullScreen():
                settings.setValue("retouch/geometry", self.saveGeometry())
            settings.setValue("retouch/splitter", self.splitter.saveState())
        except Exception:
            pass

    def _install_filters(self) -> None:
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)

    def _populate_frames(self) -> None:
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for index, frame in enumerate(self.doc.frames):
            thumb = thumbnail_bgr(frame.image, 64)
            item = QListWidgetItem(
                QIcon(QPixmap.fromImage(_qimage_from_bgr(thumb))),
                str(index + 1))
            item.setToolTip(frame.name)
            item.setSizeHint(QSize(76, 82))
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter
                                  | Qt.AlignmentFlag.AlignBottom)
            self.frame_list.addItem(item)
        self.frame_list.blockSignals(False)
        self.frame_spin.setRange(1, max(1, self.doc.frame_count))
        self.frame_spin.setSuffix(f" / {max(1, self.doc.frame_count)}")
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, self.doc.frame_count - 1))
        self.frame_slider.setValue(self.doc.frame_index)
        self.frame_slider.blockSignals(False)

    def _update_titles(self) -> None:
        depth = "16-bit" if self.doc.base.dtype == np.uint16 else (
            "8-bit" if self.doc.base.dtype == np.uint8 else str(self.doc.base.dtype))
        text = (f"{self.record.label} · {self.doc.width}×{self.doc.height}"
                f" · {depth}")
        self._full_title = text
        self._elide_title()
        self.lbl_title.setToolTip(text)
        self.lbl_result_header.setText(trans.t("retouch_result_title"))
        self.lbl_result_header.setToolTip(self.record.label)
        self._update_source_summary()
        self._update_dirty_ui()

    def _update_source_summary(self) -> None:
        """Compact provenance: method, registration, frame count and domain.

        The information was already recorded in RenderContext, but the header
        only showed the output name, size and depth, so a ROI paste and a full
        registration looked identical.
        """
        label = getattr(self, "lbl_source_summary", None)
        if label is None:
            return
        context = getattr(self.record, "context", None)
        parts = []
        algorithm = getattr(context, "algorithm", "") if context else ""
        if algorithm:
            parts.append(str(algorithm))
        alignment = getattr(context, "alignment", "") if context else ""
        parts.append(trans.t("retouch_source_aligned") if alignment
                     and alignment != "none"
                     else trans.t("retouch_source_unaligned"))
        count = int(getattr(context, "source_count", 0) or 0) if context else 0
        if count:
            parts.append(trans.t("retouch_source_frames").format(n=count))
        roi = getattr(context, "roi_rect", None) if context else None
        parts.append(trans.t("retouch_source_roi") if roi
                     else trans.t("retouch_source_full"))
        label.setText(" · ".join(parts))
        details = [self.record.label]
        if context is not None:
            details.extend(frame.name for frame in context.frames)
            if context.roi_rect:
                details.append(f"ROI: {context.roi_rect}")
        label.setToolTip("\n".join(details))
        if self.doc.frame_count:
            frame = self.doc.active_frame
            self.lbl_source_header.setToolTip(
                f"{os.path.basename(frame.name)}")

    def _elide_title(self) -> None:
        if not hasattr(self, "_full_title"):
            return
        width = max(120, self.lbl_title.width())
        self.lbl_title.setText(self.lbl_title.fontMetrics().elidedText(
            self._full_title, Qt.TextElideMode.ElideMiddle, width))

    # -------------------------------------------------------------- frame
    def _select_frame(self, index: int, initial: bool = False) -> None:
        if not initial and self.doc.stroke_active:
            return
        if not self.doc.set_frame(index):
            return
        index = self.doc.frame_index
        self._syncing_frames = True
        self.frame_list.setCurrentRow(index)
        self.frame_spin.setValue(index + 1)
        self.frame_slider.setValue(index)
        self._syncing_frames = False
        self.btn_prev.setEnabled(index > 0)
        self.btn_next.setEnabled(index < self.doc.frame_count - 1)
        frame = self.doc.frames[index]
        self.lbl_source_header.setText(trans.t("retouch_frame").format(
            i=index + 1, n=self.doc.frame_count,
            name=os.path.basename(frame.name)) + "…")
        self._source_ready = False
        self._source_request += 1
        request = self._source_request
        self._update_controls()
        QTimer.singleShot(
            0, lambda: self._finish_source_request(request, index))

    def _finish_source_request(self, request: int, index: int) -> None:
        if self._closing or request != self._source_request:
            return
        if index != self.doc.frame_index or self.doc.stroke_active:
            return
        self._refresh_source_display()
        self._source_ready = True
        frame = self.doc.frames[index]
        self.lbl_source_header.setText(trans.t("retouch_frame").format(
            i=index + 1, n=self.doc.frame_count,
            name=os.path.basename(frame.name)))
        self._update_controls()
        self._update_status()

    def _on_frame_selected(self, row: int) -> None:
        if self._syncing_frames or row < 0:
            return
        self._select_frame(row)

    def _on_frame_spin(self, value: int) -> None:
        if self._syncing_frames:
            return
        self._select_frame(int(value) - 1)

    def _on_frame_slider(self, value: int) -> None:
        if self._syncing_frames:
            return
        self._select_frame(int(value))

    def _prev_frame(self) -> None:
        self._select_frame(self.doc.frame_index - 1)

    def _next_frame(self) -> None:
        self._select_frame(self.doc.frame_index + 1)

    # ------------------------------------------------------------- display
    def _refresh_source_display(self) -> None:
        frame = self.doc.active_frame
        display = to_display_uint8(frame.image)
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        self._pure_source_qimage = _qimage_from_bgr(display)
        if self.chk_sharp.isChecked() and self._sharpness_maps:
            heat = self._sharpness_maps.get(self.doc.frame_index)
            if heat is not None:
                heat = cv2.resize(heat, (display.shape[1], display.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
                display = cv2.addWeighted(display, 0.45, heat, 0.55, 0)
        valid = self.doc.mask_for(self.doc.frame_index)
        if valid is not None:
            display = display.copy()
            display[~valid] = (80, 80, 160)
        self._source_display = display
        self._source_qimage = _qimage_from_bgr(display)
        self.source_canvas.update()
        if self.preview_mode == "source":
            self.result_canvas.update()

    def _refresh_before_display(self) -> None:
        display = to_display_uint8(self.doc.base)
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        self._before_qimage = _qimage_from_bgr(display)

    def _refresh_result_display(self) -> None:
        self._result_display = to_display_uint8(self.doc.working)
        self._result_qimage = _qimage_from_bgr(self._result_display)
        self.result_canvas.update()
        self._update_status()

    def _patch_result(self, rect) -> None:
        if self._result_qimage is None or self._result_display is None:
            self._refresh_result_display()
            return
        x0, y0, x1, y1 = rect
        patch = self.doc.working[y0:y1, x0:x1]
        display = to_display_uint8(patch)
        if self._result_display.ndim == 3 and display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        self._result_display[y0:y1, x0:x1] = display
        painter = QPainter(self._result_qimage)
        painter.drawImage(x0, y0, _qimage_from_bgr(display))
        painter.end()
        self.result_canvas.update(
            self.result_canvas.widget_rect_for_image_rect(rect))

    def canvas_image(self, role: str):
        """(qimage, width, height, offset_x, offset_y) for one canvas."""
        if role == "source":
            if self._source_qimage is None:
                return None, 0, 0, 0, 0
            frame = self.doc.active_frame
            return (self._source_qimage, frame.width, frame.height,
                    frame.offset_x, frame.offset_y)
        if self.preview_mode == "before":
            return (self._before_qimage, self.doc.width, self.doc.height, 0, 0)
        if self.preview_mode == "source":
            if self._source_qimage is None:
                return None, 0, 0, 0, 0
            frame = self.doc.active_frame
            return (self._pure_source_qimage, frame.width, frame.height,
                    frame.offset_x, frame.offset_y)
        return (self._result_qimage, self.doc.width, self.doc.height, 0, 0)

    # --------------------------------------------------------------- zoom
    def _fit(self) -> None:
        if self.doc.stroke_active:
            return
        self.view.fit = True
        scales = []
        for canvas in (self.source_canvas, self.result_canvas):
            if not canvas.isVisible() or canvas.width() < 2 or canvas.height() < 2:
                continue
            scales.append(min((canvas.width() - 6) / float(self.doc.width),
                              (canvas.height() - 6) / float(self.doc.height)))
        if scales:
            self.view.scale = max(1e-6, min(scales))
        self._update_views()

    def _zoom_100(self) -> None:
        if self.doc.stroke_active:
            return
        dpr = max(1.0, self.result_canvas.devicePixelRatioF())
        self.view.fit = False
        self.view.scale = 1.0 / dpr
        self._update_views()

    def _zoom_step(self, direction: int) -> None:
        """Keyboard zoom around the canvas centre (same clamp as the wheel)."""
        if self.doc.stroke_active:
            return
        canvas = self.result_canvas
        centre = QPointF(canvas.width() / 2.0, canvas.height() / 2.0)
        self.zoom_at(canvas, centre, 1.25 if direction > 0 else 0.8)

    def zoom_at(self, canvas: _RetouchCanvas, pos: QPointF,
                factor: float) -> None:
        if self.doc.stroke_active:
            return
        old = self.view.scale
        dpr = max(1.0, canvas.devicePixelRatioF())
        new = min(max(old * factor, 0.05 / dpr), 16.0 / dpr)
        if abs(new - old) < 1e-9:
            return
        before = canvas.widget_to_image(pos)
        self.view.scale = new
        self.view.fit = False
        after = canvas.widget_to_image(pos)
        self.view.center += before - after
        self._update_views()

    def pan_by(self, dx: float, dy: float) -> None:
        if self.doc.stroke_active or self.view.scale <= 0:
            return
        self.view.center -= QPointF(dx / self.view.scale, dy / self.view.scale)
        self.view.fit = False
        self._update_views()

    def _update_views(self) -> None:
        self.source_canvas.update()
        self.result_canvas.update()
        self._update_status()

    def canvas_resized(self) -> None:
        if self.view.fit:
            self._fit()
        else:
            self._update_status()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide_title()
        QTimer.singleShot(0, self._elide_title)
        if not self._layout_user_choice:
            if self.width() < 1024 and not self._responsive_focus:
                self._set_focus_layout(True, user=False)
                self._responsive_focus = True
            elif self.width() >= 1024 and self._responsive_focus:
                self._set_focus_layout(False, user=False)
                self._responsive_focus = False
        if self.view.fit:
            self._fit()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._fit)

    # -------------------------------------------------------------- cursor
    def set_cursor(self, x: float, y: float) -> None:
        self.cursor_pos = QPointF(x, y)
        self.source_canvas.update()
        self.result_canvas.update()
        self._update_status()

    def clear_cursor(self) -> None:
        if self.cursor_pos is None:
            return
        self.cursor_pos = None
        self.source_canvas.update()
        self.result_canvas.update()
        self._update_status()

    def image_contains(self, x: float, y: float) -> bool:
        return 0 <= x < self.doc.width and 0 <= y < self.doc.height

    def source_frame_contains(self, x: float, y: float) -> bool:
        return self.doc.active_frame.contains(x, y)

    def cursor_source_valid(self) -> bool:
        if self.cursor_pos is None:
            return True
        return self.doc.source_valid_at(self.cursor_pos.x(), self.cursor_pos.y())

    def _update_status(self) -> None:
        dpr = max(1.0, self.result_canvas.devicePixelRatioF())
        percent = int(round(self.view.scale * dpr * 100))
        self.lbl_zoom.setText(f"{percent}%")
        self.lbl_zoom_status.setText(f"{percent}%")
        if self.cursor_pos is None:
            self.lbl_coords.setText("")
        else:
            self.lbl_coords.setText(
                f"x: {int(self.cursor_pos.x())}  y: {int(self.cursor_pos.y())}")
        frame = self.doc.active_frame
        self.lbl_frame_status.setText(trans.t("retouch_frame").format(
            i=self.doc.frame_index + 1, n=self.doc.frame_count,
            name=os.path.basename(frame.name)))
        tool = (trans.t("retouch_tool_source") if self.doc.tool == "source"
                else trans.t("retouch_tool_restore"))
        if self.cursor_pos is not None and self.doc.tool == "source" and not self.cursor_source_valid():
            tool = (trans.t("retouch_source_roi") + " · "
                    if not frame.contains(self.cursor_pos.x(), self.cursor_pos.y()) else "")
            tool += trans.t("retouch_source_unavailable")
        self.lbl_state.setText(self._notice or tool)

    def _set_notice(self, text: str) -> None:
        self._notice = text
        self._update_status()
        if self._notice_timer is not None:
            self._notice_timer.stop()
        self._notice_timer = QTimer(self)
        self._notice_timer.setSingleShot(True)
        self._notice_timer.timeout.connect(self._clear_notice)
        self._notice_timer.start(7000)

    def _clear_notice(self) -> None:
        self._notice = ""
        self._update_status()

    # -------------------------------------------------------------- brush
    def _set_tool(self, tool: str) -> None:
        if self.doc.stroke_active or not self.doc.set_tool(tool):
            self._sync_tool_buttons()
            return
        self._sync_tool_buttons()
        self._refresh_help_text()
        self._update_status()
        self.source_canvas.update()
        self.result_canvas.update()

    def _show_shortcuts(self) -> None:
        from utils.ui_utils import show_message_box
        show_message_box(
            self, trans.t("retouch_shortcuts_help"),
            trans.t("retouch_shortcuts"))

    def _refresh_help_text(self) -> None:
        """The help line follows the active tool (it used to stay on source)."""
        if self.lbl_help is None:
            return
        key = ("retouch_help_restore" if self.doc.tool == "restore"
               else "retouch_help")
        self.lbl_help.setText(trans.t(key))

    def _sync_tool_buttons(self) -> None:
        self._syncing_tools = True
        self.rb_source.setChecked(self.doc.tool == "source")
        self.rb_restore.setChecked(self.doc.tool == "restore")
        self._syncing_tools = False

    def _on_brush_changed(self, _value=None) -> None:
        if self.doc.stroke_active:
            return
        self.doc.set_brush(self.num_diameter.value(),
                           self.num_hardness.value() / 100.0,
                           self.num_opacity.value() / 100.0)
        self._update_brush_labels()
        self.source_canvas.update()
        self.result_canvas.update()

    def _update_brush_labels(self) -> None:
        self.num_hardness.set_label(
            trans.t("retouch_hardness").format(self.num_hardness.value()))
        self.num_opacity.set_label(
            trans.t("retouch_opacity").format(self.num_opacity.value()))

    def _bump_diameter(self, factor: float) -> None:
        """Resize the brush from the keyboard.

        `_NumberRow.setValue` intentionally does not emit valueChanged (it is
        also used to seed the controls), so calling it alone left the document
        at the old diameter: the number changed, the next stroke did not.
        """
        if self.doc.stroke_active:
            return
        value = self.num_diameter.value()
        new = int(round(value * factor))
        if factor > 1.0 and new <= value:
            new = value + 1
        if factor < 1.0 and new >= value:
            new = value - 1
        new = max(1, min(2048, new))
        if new == value:
            return
        self.num_diameter.setValue(new)
        self._on_brush_changed()

    # ------------------------------------------------------------- strokes
    def begin_stroke(self, x: float, y: float) -> bool:
        if (not self._source_ready or self.preview_mode is not None
                or self.doc.stroke_active):
            return False
        if not self.doc.source_valid_at(x, y):
            self._set_notice(trans.t("retouch_source_unavailable"))
            return False
        if not self.doc.begin_stroke(x, y):
            return False
        self._set_stroke_ui(True)
        self._refresh_result_display()
        self._update_dirty_ui()
        return True

    def stroke_to(self, x: float, y: float) -> None:
        if not self.doc.stroke_active:
            return
        dirty = self.doc.stroke_to(x, y)
        if dirty is not None:
            self._patch_result(dirty)

    def end_stroke(self) -> None:
        if not self.doc.stroke_active:
            return
        status = self.doc.end_stroke()
        self._set_stroke_ui(False)
        if status == "oversize":
            self._refresh_result_display()
            self._set_notice(trans.t("retouch_stroke_too_large"))
            QMessageBox.warning(self, trans.t("msg_warning"),
                                trans.t("retouch_stroke_too_large"))
        elif status == "committed" and self.doc.history.evicted:
            if not self._eviction_announced:
                self._eviction_announced = True
                self._set_notice(trans.t("retouch_history_evicted"))
        self._update_dirty_ui()
        self._update_controls()

    def _cancel_stroke(self) -> None:
        if not self.doc.stroke_active:
            return
        self.doc.cancel_stroke()
        self._set_stroke_ui(False)
        self._refresh_result_display()
        self._update_dirty_ui()
        self._update_controls()

    def _set_stroke_ui(self, active: bool) -> None:
        for widget in self._stroke_lock_widgets:
            widget.setEnabled(not active)
        if not active:
            self._update_controls()

    # ------------------------------------------------------------- history
    def _undo(self) -> None:
        if self.preview_mode is not None:
            return
        was_stroke = self.doc.stroke_active
        if self.doc.undo():
            self._refresh_result_display()
        if was_stroke:
            self._set_stroke_ui(False)
        self._update_dirty_ui()
        self._update_controls()

    def _redo(self) -> None:
        if self.preview_mode is not None:
            return
        was_stroke = self.doc.stroke_active
        if self.doc.redo():
            self._refresh_result_display()
        if was_stroke:
            self._set_stroke_ui(False)
        self._update_dirty_ui()
        self._update_controls()

    def _restore_all(self) -> None:
        if (self.doc.stroke_active or self.preview_mode is not None
                or not self.doc.is_dirty):
            return
        box = QMessageBox(self)
        from utils.ui_utils import style_message_box
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(trans.t("retouch_restore_all"))
        box.setText(trans.t("retouch_restore_all_confirm"))
        box.setStandardButtons(QMessageBox.StandardButton.Yes
                               | QMessageBox.StandardButton.Cancel)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return
        if self.doc.restore_all():
            self._refresh_result_display()
        self._update_dirty_ui()
        self._update_controls()

    def _update_dirty_ui(self) -> None:
        dirty = self.doc.is_dirty
        self.lbl_unsaved.setText(
            trans.t("retouch_state_unsaved") if dirty else "")
        self.setWindowTitle(trans.t("retouch_title") + (" ●" if dirty else ""))
        self.btn_apply.setEnabled(dirty and not self.doc.stroke_active
                                  and self.preview_mode is None)
        self.btn_restore_all.setEnabled(dirty and not self.doc.stroke_active
                                        and self.preview_mode is None)
        self.btn_undo.setEnabled(self.doc.can_undo
                                 and not self.doc.stroke_active
                                 and self.preview_mode is None)
        self.btn_redo.setEnabled(self.doc.can_redo
                                 and not self.doc.stroke_active
                                 and self.preview_mode is None)

    def _update_controls(self) -> None:
        blocked = self.doc.stroke_active
        preview = self.preview_mode is not None
        self._sync_tool_buttons()
        self.btn_undo.setEnabled(self.doc.can_undo and not blocked and not preview)
        self.btn_redo.setEnabled(self.doc.can_redo and not blocked and not preview)
        self.btn_restore_all.setEnabled(
            self.doc.is_dirty and not blocked and not preview)
        self.btn_apply.setEnabled(
            self.doc.is_dirty and not blocked and not preview)
        self.btn_change.setEnabled(not blocked)
        self.btn_layout.setEnabled(not blocked)
        self.btn_fit.setEnabled(not blocked)
        self.btn_100.setEnabled(not blocked)
        self.btn_before.setEnabled(not blocked and self.preview_mode != "source")
        self.btn_peek.setEnabled(not blocked and self.preview_mode != "before")
        self.chk_sharp.setEnabled(not blocked)
        self.rb_source.setEnabled(not blocked)
        self.rb_restore.setEnabled(not blocked)
        self.num_diameter.setEnabled(not blocked)
        self.num_hardness.setEnabled(not blocked)
        self.num_opacity.setEnabled(not blocked)
        frame_enabled = not blocked and not preview
        self.frame_list.setEnabled(frame_enabled)
        self.frame_spin.setEnabled(frame_enabled)
        self.frame_slider.setEnabled(frame_enabled)
        self.btn_prev.setEnabled(
            frame_enabled and self.doc.frame_index > 0)
        self.btn_next.setEnabled(
            frame_enabled and self.doc.frame_index < self.doc.frame_count - 1)

    # ------------------------------------------------------------- preview
    def _press_preview(self, mode: str) -> None:
        if self.doc.stroke_active or self.preview_mode is not None:
            return
        self.preview_mode = mode
        self._preview_owner = mode
        self.btn_before.setDown(mode == "before")
        self.btn_peek.setDown(mode == "source")
        self.result_canvas.update()
        self._update_controls()

    def _release_preview(self, mode: str) -> None:
        if self.preview_mode != mode:
            return
        self.preview_mode = None
        self._preview_owner = None
        self.btn_before.setDown(False)
        self.btn_peek.setDown(False)
        self.result_canvas.update()
        self._update_controls()

    def _clear_preview(self) -> None:
        if self.preview_mode is None:
            return
        self.preview_mode = None
        self._preview_owner = None
        self.btn_before.setDown(False)
        self.btn_peek.setDown(False)
        self.result_canvas.update()
        self._update_controls()

    @property
    def preview_active(self) -> bool:
        return self.preview_mode is not None

    # ---------------------------------------------------------- sharpness
    def _toggle_sharpness(self, checked: bool) -> None:
        self.lbl_heat_legend.setVisible(checked)
        if checked and not self._sharpness_started:
            self._start_sharpness_worker()
        self._set_sharp_state(
            "computing" if checked and not self._sharpness_maps
            else ("ready" if checked else ""))
        if self._source_display is not None:
            self._refresh_source_display()

    def _set_sharp_state(self, state: str) -> None:
        """Visible state for the optional detail map: computing/ready/failed."""
        label = self.lbl_sharp_state
        if label is None:
            return
        if not state:
            label.setVisible(False)
            label.setText("")
            return
        key = {"computing": "retouch_sharp_computing",
               "ready": "retouch_sharp_ready",
               "failed": "retouch_sharp_failed"}.get(state)
        if key is None:
            label.setVisible(False)
            return
        label.setText(trans.t(key))
        label.setStyleSheet("color:#c0392b;" if state == "failed" else "")
        label.setVisible(True)

    def _start_sharpness_worker(self) -> None:
        self._sharpness_started = True
        self._sharpness_request += 1
        worker = _SharpnessWorker(self.doc.frames, self._sharpness_request)
        self._sharpness_worker = worker
        worker.ready.connect(self._on_sharpness_ready)
        worker.failed.connect(self._on_sharpness_failed)
        _SHARPNESS_WORKERS.add(worker)
        worker.finished.connect(lambda w=worker: _SHARPNESS_WORKERS.discard(w))
        worker.finished.connect(
            lambda req=self._sharpness_request: self._on_sharpness_finished(req))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_sharpness_failed(self, request: int, error: str) -> None:
        if self._closing or request != self._sharpness_request:
            return
        self._set_sharp_state("failed")
        self.lbl_sharp_state.setToolTip(error)

    def _on_sharpness_finished(self, request: int) -> None:
        if self._closing or request != self._sharpness_request:
            return
        if self._sharpness_worker is not None:
            self._sharpness_worker = None
        if not self._sharpness_maps:
            # A finished worker without maps means the computation failed;
            # the checkbox used to stay ticked with nothing drawn and no
            # explanation, and there was no way to retry.
            self._sharpness_started = False
            self._set_sharp_state("failed")
        else:
            self._set_sharp_state(
                "ready" if self.chk_sharp.isChecked() else "")

    def _on_sharpness_ready(self, request: int, maps) -> None:
        if self._closing or request != self._sharpness_request:
            return
        self._sharpness_maps = maps or {}
        if self.chk_sharp.isChecked() and self._source_display is not None:
            self._set_sharp_state("ready")
            self._refresh_source_display()

    def _on_focus_changed(self, old, new) -> None:
        if self._closing:
            return
        left_canvas = old in (self.source_canvas, self.result_canvas)
        if new is None or (new is not self and not self.isAncestorOf(new)) \
                or left_canvas:
            # A held preview/pan must not survive the move into a numeric
            # field: the modifier release is ignored while typing, so the
            # canvas would stay frozen on the "before" image.
            self._clear_preview()
            self.space_down = False
            self.source_canvas.setCursor(Qt.CursorShape.CrossCursor)
            self.result_canvas.setCursor(Qt.CursorShape.CrossCursor)

    def _shutdown_sharpness(self, *_args) -> None:
        self._closing = True
        try:
            self._focus_signal.disconnect(self._on_focus_changed)
        except (TypeError, RuntimeError):
            pass
        worker = self._sharpness_worker
        self._sharpness_worker = None
        if worker is None:
            return
        try:
            worker.ready.disconnect(self._on_sharpness_ready)
        except (TypeError, RuntimeError):
            pass
        worker.requestInterruption()
        self._sharpness_request += 1

    # ----------------------------------------------------------- layout/fs
    def _set_focus_layout(self, focus: bool, user: bool) -> None:
        if self.doc.stroke_active:
            return
        # A layout switch is not a request to zoom.  Keep the current center
        # and magnification; only a resize that is still in fit mode may
        # recompute the common scale.  The manual button freezes fit mode
        # first so the resulting viewport change cannot refit underneath it.
        if user and self.view.fit:
            self.view.fit = False
        self.left_panel.setVisible(not focus)
        self.btn_layout.setText(
            trans.t("retouch_layout_side_short") if focus
            else trans.t("retouch_layout_focus_short"))
        self.btn_layout.setToolTip(
            trans.t("retouch_layout_side") if focus
            else trans.t("retouch_layout_focus"))
        if user:
            self._layout_user_choice = True
        if self.view.fit:
            self._fit()
        else:
            self._update_views()

    def _toggle_fullscreen(self) -> None:
        if self.doc.stroke_active:
            return
        if self.isFullScreen():
            if self._was_maximized:
                self.showMaximized()
            else:
                self.showNormal()
                if self._normal_geometry is not None:
                    self.setGeometry(self._normal_geometry)
            self.btn_fullscreen.setText(trans.t("compare_fullscreen"))
        else:
            self._normal_geometry = self.geometry()
            self._was_maximized = self.isMaximized()
            self.showFullScreen()
            self.btn_fullscreen.setText(trans.t("compare_exit_fullscreen"))
        # Fit mode follows the new viewport; a manual zoom keeps its scale.
        QTimer.singleShot(0, self.canvas_resized)

    # ------------------------------------------------------------- apply
    def _can_apply(self) -> bool:
        """Single condition shared by the button and the shortcut.

        Ctrl+Enter used to clear a held preview and publish, even though the
        visible Apply button was disabled during preview.
        """
        return (not self._applied and not self._applying and not self._closing
                and not self.doc.stroke_active and self.preview_mode is None
                and self.doc.is_dirty)

    def _apply(self):
        if not self._can_apply():
            return False
        self.btn_apply.setEnabled(False)
        self._applying = True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            image = np.ascontiguousarray(self.doc.working.copy())
            ok, error = self.apply_callback(self.record, image)
        except Exception as exc:
            ok, error = False, str(exc)
        finally:
            QApplication.restoreOverrideCursor()
            self._applying = False
        if ok:
            self._applied = True
            self.accept()
            return True
        self.btn_apply.setEnabled(True)
        QMessageBox.critical(self, trans.t("msg_error"),
                             error or trans.t("msg_proc_error"))
        self._update_controls()
        return False

    def _confirm_close(self) -> bool:
        if self.doc.stroke_active:
            self.end_stroke()
        if not self.doc.is_dirty:
            return True
        box = QMessageBox(self)
        from utils.ui_utils import style_message_box
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(trans.t("retouch_close_title"))
        box.setText(trans.t("retouch_close_text"))
        # A narrow message box clipped the first button's last glyph
        # ("作为新输出应用") even though the action was fully explained.
        box.setMinimumWidth(620)
        apply_btn = box.addButton(trans.t("retouch_apply_close"),
                                  QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton(trans.t("btn_discard_changes"),
                                    QMessageBox.ButtonRole.DestructiveRole)
        continue_btn = box.addButton(trans.t("btn_continue"),
                                     QMessageBox.ButtonRole.RejectRole)
        for button in (apply_btn, discard_btn, continue_btn):
            button.setMinimumWidth(120)
        box.setDefaultButton(continue_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is apply_btn:
            return self._apply()
        return clicked is discard_btn

    def _change_result(self) -> None:
        if self.doc.stroke_active:
            self.end_stroke()
        if not self._confirm_close():
            return
        if self._applied:
            return
        self.change_requested = True
        self._allow_close = True
        self._save_geometry()
        self.done(QDialog.DialogCode.Rejected)

    def reject(self) -> None:
        self.close()

    def closeEvent(self, event) -> None:
        if self._applying:
            event.ignore()
            return
        if self._closing:
            event.accept()
            return
        self._closing = True
        try:
            if self._applied or self._allow_close:
                self._save_geometry()
                self._shutdown_sharpness()
                event.accept()
                return
            if self._confirm_close():
                self._allow_close = True
                self._save_geometry()
                self._shutdown_sharpness()
                event.accept()
            else:
                event.ignore()
                if not self._source_ready:
                    self._select_frame(self.doc.frame_index, initial=True)
        finally:
            self._closing = False

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            if self.doc.stroke_active:
                self.end_stroke()
            self._clear_preview()
            self.space_down = False
            self.source_canvas.setCursor(Qt.CursorShape.CrossCursor)
            self.result_canvas.setCursor(Qt.CursorShape.CrossCursor)
        elif event.type() == QEvent.Type.WindowStateChange:
            self.btn_fullscreen.setText(
                trans.t("compare_exit_fullscreen") if self.isFullScreen()
                else trans.t("compare_fullscreen"))
        super().changeEvent(event)

    # ----------------------------------------------------------- keyboard
    def _editing_text(self) -> bool:
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QAbstractSpinBox)):
            return True
        parent = focused.parent() if focused is not None else None
        return isinstance(parent, QAbstractSpinBox)

    def _canvas_focused(self) -> bool:
        focused = QApplication.focusWidget()
        return focused in (self.source_canvas, self.result_canvas, self)

    def _handle_key(self, event) -> bool:
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & (Qt.KeyboardModifier.ControlModifier
                            | Qt.KeyboardModifier.MetaModifier))
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if event.type() == QEvent.Type.KeyRelease:
            if key == Qt.Key.Key_Tab:
                self._release_preview("before")
                return True
            if key == Qt.Key.Key_Alt:
                self._release_preview("source")
                return True
            if key == Qt.Key.Key_Space:
                self.space_down = False
                self.source_canvas.setCursor(Qt.CursorShape.CrossCursor)
                self.result_canvas.setCursor(Qt.CursorShape.CrossCursor)
                return True
            return False
        if event.type() != QEvent.Type.KeyPress or event.isAutoRepeat():
            return False

        if key == Qt.Key.Key_Escape:
            if self.doc.stroke_active:
                self._cancel_stroke()
            elif self.preview_mode is not None:
                self._clear_preview()
            elif self.isFullScreen():
                self._toggle_fullscreen()
            else:
                self.close()
            return True
        if key == Qt.Key.Key_F11:
            self._toggle_fullscreen()
            return True
        if ctrl and key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._apply()
            return True
        if ctrl and key == Qt.Key.Key_Z:
            self._redo() if shift else self._undo()
            return True
        if ctrl and key == Qt.Key.Key_Y:
            self._redo()
            return True
        # Shift+Tab must remain a normal focus-navigation key: the "hold to
        # see the result before editing" preview is plain Tab on the canvas.
        if key == Qt.Key.Key_Tab and not shift and self._canvas_focused():
            self._press_preview("before")
            return True
        if key == Qt.Key.Key_Alt and self._canvas_focused():
            self._press_preview("source")
            return True
        if key == Qt.Key.Key_Space and self._canvas_focused():
            self.space_down = True
            self.source_canvas.setCursor(Qt.CursorShape.OpenHandCursor)
            self.result_canvas.setCursor(Qt.CursorShape.OpenHandCursor)
            return True
        # Canvas letters and punctuation stay inside the canvas: typing in a
        # numeric field must never resize the brush or change frames.
        if not self._canvas_focused():
            return False
        if key == Qt.Key.Key_F:
            self._fit()
            return True
        if key == Qt.Key.Key_1:
            self._zoom_100()
            return True
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom_step(1)
            return True
        if key == Qt.Key.Key_Minus:
            self._zoom_step(-1)
            return True
        if key == Qt.Key.Key_B:
            self._set_tool("source")
            return True
        if key == Qt.Key.Key_E:
            self._set_tool("restore")
            return True
        if key == Qt.Key.Key_BracketLeft:
            self._bump_diameter(0.8)
            return True
        if key == Qt.Key.Key_BracketRight:
            self._bump_diameter(1.25)
            return True
        if key == Qt.Key.Key_Comma:
            self._prev_frame()
            return True
        if key == Qt.Key.Key_Period:
            self._next_frame()
            return True
        return False

    def eventFilter(self, obj, event) -> bool:
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if event.type() == QEvent.Type.KeyRelease:
                # Releasing a temporarily held key is cleanup, not editing:
                # it must run even when a field grabbed focus meanwhile.
                if event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Alt,
                                   Qt.Key.Key_Space):
                    self._handle_key(event)
                    return True
            elif not self._editing_text() and self._handle_key(event):
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
                event.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                     | Qt.KeyboardModifier.MetaModifier)):
            event.accept()
            return
        if not self._editing_text() and self._handle_key(event):
            event.accept()
            return
        super().keyPressEvent(event)


class RetouchResultChooser(QDialog):
    """Choose which output to retouch when the list has no single selection."""

    def __init__(self, window, parent=None):
        if parent is None and isinstance(window, QWidget):
            parent = window
        super().__init__(parent if isinstance(parent, QWidget) else None)
        self.window = window
        self.setWindowTitle(trans.t("retouch_chooser_title"))
        self.resize(720, 520)
        layout = QVBoxLayout(self)
        hint = QLabel(trans.t("retouch_chooser_hint"))
        hint.setWordWrap(True)
        hint.setObjectName("mutedLabel")
        layout.addWidget(hint)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._update_ok)
        self.list.itemDoubleClicked.connect(lambda _item: self._accept())
        layout.addWidget(self.list, 1)

        self._records = []
        for row in range(window.output_list.count()):
            item = window.output_list.item(row)
            if item is None:
                continue
            record = window.output_manager.record_for_item(item)
            retouchable = record is not None and record.retouchable
            label = item.text()
            if retouchable:
                status = (record.context.algorithm
                          if record.context is not None else "")
                size = (f"{record.context.width}×{record.context.height}"
                        if record.context is not None else "")
                text = f"{label}\n{size} · {status}"
            else:
                text = f"{label}\n{trans.t('retouch_no_source')}"
            entry = QListWidgetItem(item.icon(), text)
            entry.setToolTip(label)
            entry.setData(Qt.ItemDataRole.UserRole, row)
            entry.setData(Qt.ItemDataRole.UserRole + 1, retouchable)
            entry.setSizeHint(QSize(0, 52))
            self.list.addItem(entry)
            self._records.append((row, retouchable))
        first = next((i for i, (_row, ok) in enumerate(self._records) if ok), 0)
        if self.list.count():
            self.list.setCurrentRow(first)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_ok = QPushButton(trans.t("btn_ok"))
        self.btn_ok.clicked.connect(self._accept)
        cancel = QPushButton(trans.t("btn_cancel"))
        cancel.clicked.connect(self.reject)
        buttons.addWidget(self.btn_ok)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        self._update_ok()

    def _current(self):
        entry = self.list.currentItem()
        if entry is None:
            return None, False
        return entry.data(Qt.ItemDataRole.UserRole), bool(
            entry.data(Qt.ItemDataRole.UserRole + 1))

    def _update_ok(self, *_args) -> None:
        # QListWidget may select the first inserted row and emit this signal
        # before the button row is built.
        if not hasattr(self, "btn_ok"):
            return
        _row, ok = self._current()
        self.btn_ok.setEnabled(bool(ok))

    def _accept(self) -> None:
        _row, ok = self._current()
        if ok:
            self.accept()

    def selected_item(self):
        row, ok = self._current()
        if not ok or row is None:
            return None
        return self.window.output_list.item(int(row))
