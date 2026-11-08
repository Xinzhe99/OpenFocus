"""Manual source-coordinate control point editing; optimization belongs to the parent."""
from __future__ import annotations

from dataclasses import replace
import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QTransform
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QStyledItemDelegate, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from core.mosaic_control import MosaicControlPoint
from core.mosaic import measure_camera_control_points
from locales import trans
from utils.image_utils import cv2_to_pixmap


class _SourcePreview(QGraphicsView):
    """A bounded thumbnail displayed in native pixel-center scene coordinates."""

    pointClicked = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(240, 180)
        self._native_shape = (0, 0)
        self._markers = ()
        self._zoomed = False
        self._pixmap_item = None

    def set_preview(self, preview, native_shape):
        self.scene().clear()
        self._markers = ()
        self._native_shape = tuple(native_shape[:2])
        height, width = self._native_shape
        pixmap = cv2_to_pixmap(preview)
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        if not pixmap.isNull():
            self._pixmap_item.setTransform(QTransform.fromScale(width / pixmap.width(), height / pixmap.height()))
            self._pixmap_item.setPos(-0.5, -0.5)
        self.scene().setSceneRect(QRectF(-0.5, -0.5, width, height))
        self._zoomed = False
        self.resetTransform()
        self._fit()

    def _fit(self):
        if self._native_shape[0] and self._native_shape[1]:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._zoomed:
            self._fit()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        scale = self.transform().m11() * factor
        if 1e-8 < scale < 1000:
            self.scale(factor, factor)
            self._zoomed = True
        event.accept()

    def native_position(self, position):
        point = self.mapToScene(position.toPoint())
        height, width = self._native_shape
        if not (-0.5 <= point.x() < width - 0.5 and -0.5 <= point.y() < height - 0.5):
            return None
        return (min(width - 1.0, max(0.0, point.x())),
                min(height - 1.0, max(0.0, point.y())))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.native_position(event.position())
            if point is not None:
                self.pointClicked.emit(point)
                event.accept()
                return
        super().mousePressEvent(event)

    def set_markers(self, markers):
        # (native point, selected, enabled, label) tuples; no image rasterization.
        self._markers = tuple(markers)
        self.viewport().update()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        scale = max(abs(self.transform().m11()), 1e-12)
        radius = 5.0 / scale
        for point, selected, enabled, label in self._markers:
            color = QColor('#ffff00' if selected else '#00ff80' if enabled else '#999999')
            pen = QPen(color, 2 if selected else 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            center = QPointF(*point)
            painter.drawEllipse(center, radius, radius)
            painter.drawLine(center + QPointF(-radius * 1.5, 0), center + QPointF(radius * 1.5, 0))
            painter.drawLine(center + QPointF(0, -radius * 1.5), center + QPointF(0, radius * 1.5))
            # Labels stay in screen pixels at every zoom level.
            painter.save()
            painter.translate(center + QPointF(radius * 1.7, -radius))
            painter.scale(1 / scale, 1 / scale)
            painter.drawText(QPointF(), str(label))
            painter.restore()


class _PointDelegate(QStyledItemDelegate):
    def __init__(self, frames, parent=None):
        super().__init__(parent)
        self.frames = frames

    def createEditor(self, parent, option, index):
        if index.column() in (1, 2):
            editor = QSpinBox(parent)
            editor.setRange(0, max(0, len(self.frames) - 1))
            return editor
        if index.column() in (3, 4):
            editor = QWidget(parent)
            layout = QHBoxLayout(editor)
            layout.setContentsMargins(0, 0, 0, 0)
            editor.coordinates = []
            frame_column = 1 if index.column() == 3 else 2
            frame_index = int(index.siblingAtColumn(frame_column).data(Qt.ItemDataRole.EditRole))
            height, width = self.frames[frame_index].image.shape[:2]
            for maximum in (width - 1, height - 1):
                spin = QDoubleSpinBox(editor)
                spin.setDecimals(6)
                spin.setRange(0, maximum)
                spin.setKeyboardTracking(False)
                layout.addWidget(spin)
                editor.coordinates.append(spin)
            return editor
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        if index.column() in (1, 2):
            editor.setValue(int(index.data(Qt.ItemDataRole.EditRole)))
        elif index.column() in (3, 4):
            values = index.data(Qt.ItemDataRole.UserRole)
            for spin, value in zip(editor.coordinates, values):
                spin.setValue(value)
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if index.column() in (1, 2):
            editor.interpretText()
            model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)
        elif index.column() in (3, 4):
            for spin in editor.coordinates:
                spin.interpretText()
            # One model mutation lets the dialog replace a CP atomically.
            model.setData(index, ', '.join(format(spin.value(), '.6f') for spin in editor.coordinates), Qt.ItemDataRole.EditRole)
        else:
            super().setModelData(editor, model, index)


class MosaicControlPointsDialog(QDialog):
    """`points` is immutable; signals publish tuples for a parent-owned worker.

    selectionChanged emits selected row indices in `points`. optimizeRequested
    emits the current points tuple. A busy close sends cancelRequested once and
    defers the actual close until the parent calls set_busy(False).
    """

    pointsChanged = pyqtSignal(object)
    optimizeRequested = pyqtSignal(object)
    selectionChanged = pyqtSignal(object)
    cancelRequested = pyqtSignal()

    def __init__(self, frames, transforms, control_points, parent=None, *, cameras=()):
        super().__init__(parent)
        self.frames = tuple(frames)
        self.transforms = tuple(transforms)
        self.cameras = tuple(cameras)
        self.points = tuple(control_points)
        self._qa = None
        self._residuals = ()
        self._busy = False
        self._close_pending = False
        self._refreshing = False
        self._pending = [None, None]
        self._buttons = []
        self.resize(1000, 760)
        outer = QVBoxLayout(self)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        outer.addWidget(self.hint)
        choices = QHBoxLayout()
        self.image_a, self.image_b = QComboBox(), QComboBox()
        self.label_a, self.label_b = QLabel(), QLabel()
        for label, combo in ((self.label_a, self.image_a), (self.label_b, self.image_b)):
            choices.addWidget(label)
            choices.addWidget(combo, 1)
            for index, frame in enumerate(self.frames):
                combo.addItem(f'{index}: {frame.name}', index)
        if len(self.frames) > 1:
            self.image_b.setCurrentIndex(1)
        outer.addLayout(choices)
        previews = QSplitter(Qt.Orientation.Horizontal)
        self.preview_a, self.preview_b = _SourcePreview(), _SourcePreview()
        previews.addWidget(self.preview_a)
        previews.addWidget(self.preview_b)
        outer.addWidget(previews, 2)
        self.image_a.currentIndexChanged.connect(self._load_previews)
        self.image_b.currentIndexChanged.connect(self._load_previews)
        self.preview_a.pointClicked.connect(lambda point: self._clicked(0, point))
        self.preview_b.pointClicked.connect(lambda point: self._clicked(1, point))
        self.add_pair = QPushButton()
        self.add_pair.clicked.connect(self._add_pair)
        outer.addWidget(self.add_pair)
        self.table = QTableWidget(0, 7)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setItemDelegate(_PointDelegate(self.frames, self.table))
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        outer.addWidget(self.table, 2)
        row = QHBoxLayout()
        self.delete_selected = self._button(row, 'mosaic_cp_delete', self._delete_selected)
        self.delete_worst = self._button(row, 'mosaic_cp_worst', self._delete_worst)
        self.delete_nonadjacent = self._button(row, 'mosaic_cp_nonadjacent', self._delete_nonadjacent)
        self.optimize = self._button(row, 'mosaic_optimize', self._optimize)
        outer.addLayout(row)
        self.qa_label = QLabel()
        self.qa_label.setWordWrap(True)
        outer.addWidget(self.qa_label)
        self.dialog_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.dialog_buttons.rejected.connect(self.reject)
        outer.addWidget(self.dialog_buttons)
        trans.languageChanged.connect(self._retranslate)
        self._retranslate()
        self._load_previews()
        self._refresh_table()

    def _button(self, row, key, callback):
        button = QPushButton()
        button.setAutoDefault(False)
        button.clicked.connect(callback)
        self._buttons.append((button, key))
        row.addWidget(button)
        return button

    def _retranslate(self):
        self.setWindowTitle(trans.t('mosaic_control_points'))
        self.hint.setText(trans.t('mosaic_cp_click_hint'))
        self.label_a.setText(trans.t('mosaic_cp_a'))
        self.label_b.setText(trans.t('mosaic_cp_b'))
        self.add_pair.setText(trans.t('mosaic_cp_add'))
        self.add_pair.setAutoDefault(False)
        self.table.setHorizontalHeaderLabels([trans.t(key) for key in ('mosaic_cp_active', 'mosaic_cp_a', 'mosaic_cp_b', 'mosaic_cp_point_a', 'mosaic_cp_point_b', 'mosaic_cp_source', 'mosaic_cp_residual')])
        for button, key in self._buttons:
            button.setText(trans.t(key))
        self._refresh_qa()
        if self.table.rowCount():
            self._refresh_table()

    def _load_previews(self, *_):
        self._pending = [None, None]
        for combo, preview in ((self.image_a, self.preview_a), (self.image_b, self.preview_b)):
            index = combo.currentIndex()
            if index >= 0:
                frame = self.frames[index]
                preview.set_preview(frame.preview, frame.image.shape[:2])
        self._markers()
        self._controls()

    def _clicked(self, side, point):
        if self._busy:
            return
        # Clicking an existing marker selects its row; clicking elsewhere stages
        # that side of a new pair until Add is pressed.
        frame_index = (self.image_a, self.image_b)[side].currentIndex()
        preview = (self.preview_a, self.preview_b)[side]
        candidates = []
        for index, cp in enumerate(self.points):
            for source, coordinate in ((cp.frame_a, cp.point_a), (cp.frame_b, cp.point_b)):
                if source == frame_index:
                    distance = math.hypot(point[0] - coordinate[0], point[1] - coordinate[1]) * abs(preview.transform().m11())
                    if distance <= 7:
                        candidates.append((distance, index))
        if candidates:
            self.table.selectRow(min(candidates)[1])
            return
        self._pending[side] = tuple(point)
        self._markers()
        self._controls()

    def _add_pair(self):
        if self._busy or None in self._pending:
            return
        a, b = self.image_a.currentIndex(), self.image_b.currentIndex()
        if a < 0 or b < 0 or a == b:
            return
        cp = MosaicControlPoint(a, b, self._pending[0], self._pending[1], source='manual', weight=2.0)
        self._pending = [None, None]
        self._publish(self.points + (cp,))
        self.table.selectRow(len(self.points) - 1)

    def _selected_rows(self):
        return tuple(sorted({index.row() for index in self.table.selectionModel().selectedRows()}))

    def _selection_changed(self):
        if self._refreshing:
            return
        selected = self._selected_rows()
        if selected:
            cp = self.points[selected[0]]
            if (self.image_a.currentIndex(), self.image_b.currentIndex()) != (cp.frame_a, cp.frame_b):
                self.image_a.blockSignals(True)
                self.image_b.blockSignals(True)
                self.image_a.setCurrentIndex(cp.frame_a)
                self.image_b.setCurrentIndex(cp.frame_b)
                self.image_a.blockSignals(False)
                self.image_b.blockSignals(False)
                self._load_previews()
        self._markers()
        self._controls()
        self.selectionChanged.emit(selected)

    def _markers(self):
        selected = set(self._selected_rows()) if hasattr(self, 'table') else set()
        for side, (combo, preview) in enumerate(((self.image_a, self.preview_a), (self.image_b, self.preview_b))):
            markers = []
            for index, cp in enumerate(self.points):
                for frame_index, point in ((cp.frame_a, cp.point_a), (cp.frame_b, cp.point_b)):
                    if frame_index == combo.currentIndex():
                        markers.append((point, index in selected, cp.enabled, index + 1))
            if self._pending[side] is not None:
                markers.append((self._pending[side], True, True, '+'))
            preview.set_markers(markers)

    def _residual(self, cp):
        if max(cp.frame_a, cp.frame_b) >= len(self.transforms):
            return math.nan
        if self.cameras:
            try:
                return measure_camera_control_points(self.cameras, (cp,)).residuals[0]
            except ValueError:
                return math.inf
        locations = []
        for index, point in ((cp.frame_a, cp.point_a), (cp.frame_b, cp.point_b)):
            matrix = self.transforms[index]
            x, y = point
            denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
            if denominator == 0:
                return math.inf
            locations.append(((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator,
                              (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator))
        return math.hypot(locations[0][0] - locations[1][0], locations[0][1] - locations[1][1])

    def _refresh_table(self):
        selected = self._selected_rows()
        self._refreshing = True
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self.points))
            if self._qa is None:
                if self.cameras:
                    try:
                        self._qa = measure_camera_control_points(self.cameras, self.points)
                    except ValueError:
                        self._qa = None
                self._residuals = (tuple(self._qa.residuals) if self._qa is not None
                                   else tuple(self._residual(cp) for cp in self.points))
            for row, cp in enumerate(self.points):
                enabled = QTableWidgetItem()
                enabled.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
                enabled.setCheckState(Qt.CheckState.Checked if cp.enabled else Qt.CheckState.Unchecked)
                self.table.setItem(row, 0, enabled)
                for column, value in ((1, cp.frame_a), (2, cp.frame_b)):
                    item = QTableWidgetItem()
                    item.setData(Qt.ItemDataRole.EditRole, value)
                    self.table.setItem(row, column, item)
                for column, point in ((3, cp.point_a), (4, cp.point_b)):
                    item = QTableWidgetItem(', '.join(format(value, '.6f') for value in point))
                    item.setData(Qt.ItemDataRole.UserRole, point)
                    self.table.setItem(row, column, item)
                source_key = 'mosaic_cp_auto' if cp.source == 'auto' else 'mosaic_cp_manual'
                for column, text in ((5, trans.t(source_key)), (6, format(self._residuals[row], '.3f'))):
                    item = QTableWidgetItem(text)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    self.table.setItem(row, column, item)
            self.table.clearSelection()
            for row in selected:
                if row < len(self.points):
                    for column in range(self.table.columnCount()):
                        self.table.item(row, column).setSelected(True)
        finally:
            self.table.blockSignals(False)
            self._refreshing = False
        self._markers()
        self._controls()
        self._refresh_qa()

    def _item_changed(self, item):
        if self._refreshing:
            return
        if self._busy:
            self._refresh_table()
            return
        row, column = item.row(), item.column()
        cp = self.points[row]
        try:
            if column == 0:
                cp = replace(cp, enabled=item.checkState() == Qt.CheckState.Checked)
            elif column in (1, 2):
                index = int(item.text())
                if not 0 <= index < len(self.frames):
                    raise ValueError(trans.t('mosaic_cp_frame_outside').format(
                        index=index, last=len(self.frames) - 1))
                cp = replace(cp, **{('frame_a' if column == 1 else 'frame_b'): index})
            elif column in (3, 4):
                cp = replace(cp, **{('point_a' if column == 3 else 'point_b'): tuple(float(value.strip()) for value in item.text().split(','))})
            else:
                return
            for index, point in ((cp.frame_a, cp.point_a), (cp.frame_b, cp.point_b)):
                height, width = self.frames[index].image.shape[:2]
                if not (0 <= point[0] <= width - 1 and 0 <= point[1] <= height - 1):
                    raise ValueError(trans.t('mosaic_cp_point_outside').format(
                        frame=index, x=point[0], y=point[1], width=width, height=height))
        except (ValueError, IndexError) as error:
            self._refresh_table()
            QMessageBox.warning(self, trans.t('mosaic_control_points'), trans.t('mosaic_error').format(error=str(error)))
            return
        points = list(self.points)
        points[row] = cp
        self._publish(tuple(points))

    def _publish(self, points):
        self.points = tuple(points)
        self._qa = None
        self._refresh_table()
        self.pointsChanged.emit(self.points)
        self.selectionChanged.emit(self._selected_rows())

    def _delete_selected(self):
        if not self._busy:
            rows = set(self._selected_rows())
            self.table.clearSelection()
            self._publish(tuple(cp for index, cp in enumerate(self.points) if index not in rows))

    def _delete_worst(self):
        if not self._busy and self.points:
            candidates = [index for index, cp in enumerate(self.points) if cp.enabled and not math.isnan(self._residuals[index])]
            count = max(1, math.ceil(len(candidates) / 10)) if candidates else 0
            rows = set(sorted(candidates, key=lambda index: self._residuals[index], reverse=True)[:count])
            self.table.clearSelection()
            self._publish(tuple(cp for index, cp in enumerate(self.points) if index not in rows))

    def _delete_nonadjacent(self):
        if not self._busy:
            self.table.clearSelection()
            self._publish(tuple(cp for cp in self.points if abs(cp.frame_a - cp.frame_b) <= 1))

    def _optimize(self):
        if not self._busy:
            self.optimizeRequested.emit(self.points)

    def _controls(self):
        if not hasattr(self, 'table'):
            return
        self.add_pair.setEnabled(not self._busy and None not in self._pending and self.image_a.currentIndex() != self.image_b.currentIndex())
        self.delete_selected.setEnabled(not self._busy and bool(self._selected_rows()))
        self.delete_worst.setEnabled(not self._busy and any(cp.enabled for cp in self.points))
        self.delete_nonadjacent.setEnabled(not self._busy and any(abs(cp.frame_a - cp.frame_b) > 1 for cp in self.points))
        self.optimize.setEnabled(not self._busy and len(self.frames) >= 2)

    def _refresh_qa(self):
        if self._qa is None:
            self.qa_label.setText(trans.t('mosaic_qa'))
        else:
            self.qa_label.setText(trans.t('mosaic_qa_status').format(points=self._qa.cp_count,
                weak=sum(pair.weak for pair in self._qa.pairs), disconnected=len(self._qa.disconnected_indices), residual=self._qa.residual_p95))

    def set_result(self, transforms, qa, control_points=None, *, cameras=None):
        if control_points is not None:
            self.points = tuple(control_points)
        self.transforms = tuple(transforms)
        if cameras is not None:
            self.cameras = tuple(cameras)
        self._qa = qa
        self._residuals = tuple(qa.residuals) if qa is not None else ()
        if qa is not None and len(self._residuals) != len(self.points):
            raise ValueError('Alignment residuals must match the current control points')
        self._refresh_table()

    def set_busy(self, busy):
        self._busy = bool(busy)
        for widget in (self.table, self.image_a, self.image_b, self.preview_a, self.preview_b):
            widget.setEnabled(not self._busy)
        self._controls()
        if not self._busy and self._close_pending:
            self._close_pending = False
            super().reject()

    def _defer_close(self):
        if self._busy:
            if not self._close_pending:
                self._close_pending = True
                self.cancelRequested.emit()
            return True
        return False

    def reject(self):
        if not self._defer_close():
            super().reject()

    def accept(self):
        if not self._defer_close():
            super().accept()

    def closeEvent(self, event):
        if self._defer_close():
            event.ignore()
        else:
            super().closeEvent(event)
