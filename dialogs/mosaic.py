"""Independent workspace for arranging and exporting microscope mosaics."""
import os
import threading

import numpy as np
from PyQt6.QtCore import QPointF, QSignalBlocker, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QTransform
from PyQt6.QtWidgets import (
    QApplication, QAbstractItemView, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QListWidget, QMessageBox, QProgressBar,
    QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from core.mosaic import auto_align, canvas_bounds, initial_transforms, make_frame, render_mosaic
from core.registration import RegistrationCancelled
from core.resource_policy import heavy_job, require_memory
from locales import trans
from utils.image_utils import cv2_to_pixmap, imwrite_auto
from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir


class _Worker(QThread):
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(int, int)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self._stop = threading.Event()

    def cancel(self):
        self._stop.set()

    def run(self):
        try:
            with heavy_job(self._stop.is_set):
                value = self.operation(self._stop.is_set, self.progress.emit)
            if self._stop.is_set():
                raise RegistrationCancelled()
        except RegistrationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            if self._stop.is_set():
                self.cancelled.emit()
            else:
                self.error.emit(str(exc))
        else:
            self.result.emit(value)


class _TileItem(QGraphicsPixmapItem):
    def __init__(self, index, pixmap, callback):
        super().__init__(pixmap)
        self.index = index
        self.callback = callback
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)

    def itemChange(self, change, value):
        value = super().itemChange(change, value)
        if change in (QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
                      QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged):
            self.callback(self, change)
        return value


class _Canvas(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(280, 240)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        event.accept()


class MosaicDialog(QDialog):
    """Arrange already-fused images without altering the current source stack."""

    def __init__(self, window_main, parent=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self.setWindowTitle(trans.t('mosaic_title'))
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        screen = self.screen() or QApplication.primaryScreen()
        if screen:
            available = screen.availableGeometry()
            self.resize(min(1280, max(480, available.width() - 80)),
                        min(820, max(360, available.height() - 100)))
        else:
            self.resize(1100, 700)
        self.frames = []
        self.transforms = []
        self.items = []
        self.connected = []
        self.worker = None
        self._close_pending = False
        self._updating = False
        self._build_ui()
        self._refresh_controls()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        intro = QLabel(trans.t('mosaic_intro'))
        intro.setWordWrap(True)
        outer.addWidget(intro)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)
        panel = QWidget()
        panel.setMinimumWidth(220)
        left = QVBoxLayout(panel)
        left.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.add_files = QPushButton(trans.t('mosaic_add_files'))
        self.add_outputs = QPushButton(trans.t('mosaic_add_outputs'))
        row.addWidget(self.add_files)
        row.addWidget(self.add_outputs)
        left.addLayout(row)
        self.remove = QPushButton(trans.t('batch_btn_remove'))
        left.addWidget(self.remove)
        position = QFormLayout()
        self.x_spin, self.y_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.x_spin, self.y_spin):
            spin.setRange(-1000000000, 1000000000)
            spin.setDecimals(1)
            spin.setSingleStep(1)
        position.addRow(trans.t('label_x'), self.x_spin)
        position.addRow(trans.t('label_y'), self.y_spin)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(100)
        position.addRow(trans.t('mosaic_opacity'), self.opacity)
        left.addLayout(position)
        self.auto = QPushButton(trans.t('mosaic_auto'))
        self.fit = QPushButton(trans.t('label_fit'))
        left.addWidget(self.auto)
        left.addWidget(self.fit)
        self.size_label = QLabel('')
        self.size_label.setWordWrap(True)
        left.addWidget(self.size_label)
        splitter.addWidget(panel)
        self.scene = QGraphicsScene(self)
        self.canvas = _Canvas(self.scene)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 900])
        self.status = QLabel(trans.t('mosaic_empty'))
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        outer.addWidget(self.progress)
        row = QHBoxLayout()
        row.addStretch(1)
        self.cancel = QPushButton(trans.t('btn_cancel'))
        self.cancel.setVisible(False)
        self.save = QPushButton(trans.t('mosaic_save'))
        self.close_button = QPushButton(trans.t('btn_close'))
        row.addWidget(self.cancel)
        row.addWidget(self.save)
        row.addWidget(self.close_button)
        outer.addLayout(row)
        self.add_files.clicked.connect(self._add_files)
        self.add_outputs.clicked.connect(self._add_outputs)
        self.remove.clicked.connect(self._remove_selected)
        self.auto.clicked.connect(self._auto_align)
        self.fit.clicked.connect(self._fit)
        self.save.clicked.connect(self._save)
        self.cancel.clicked.connect(self._cancel)
        self.close_button.clicked.connect(self.close)
        self.list.currentRowChanged.connect(self._select_row)
        self.x_spin.valueChanged.connect(self._position_changed)
        self.y_spin.valueChanged.connect(self._position_changed)
        self.opacity.valueChanged.connect(self._opacity_changed)

    def _refresh_controls(self):
        busy = self.worker is not None and self.worker.isRunning()
        selected = 0 <= self.list.currentRow() < len(self.frames)
        self.add_files.setEnabled(not busy)
        self.add_outputs.setEnabled(not busy)
        self.remove.setEnabled(not busy and selected)
        self.x_spin.setEnabled(not busy and selected)
        self.y_spin.setEnabled(not busy and selected)
        self.opacity.setEnabled(not busy and selected)
        self.auto.setEnabled(not busy and len(self.frames) >= 2)
        self.fit.setEnabled(bool(self.frames))
        self.save.setEnabled(not busy and bool(self.frames))
        self.list.setEnabled(not busy)
        for item in self.items:
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not busy)
        self.cancel.setVisible(busy)
        self.cancel.setEnabled(busy)
        self.progress.setVisible(busy)

    def _run(self, operation, status, success):
        self.status.setText(status)
        self.progress.setRange(0, 0)
        worker = _Worker(operation, self)
        self.worker = worker
        worker.progress.connect(self._progress)
        worker.result.connect(success)
        worker.error.connect(self._error)
        worker.cancelled.connect(lambda: self.status.setText(trans.t('compare_cancelled')))
        worker.finished.connect(self._finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._refresh_controls()

    def _progress(self, value, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(value)

    def _finished(self):
        self.worker = None
        self._refresh_controls()
        if self._close_pending:
            self.close()

    def _error(self, error):
        self.status.setText(trans.t('alignment_report_failed'))
        QMessageBox.warning(self, trans.t('mosaic_title'),
                            trans.t('msg_save_failed_info_unexpected').format(error=error))

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, trans.t('mosaic_add_files'), get_last_dialog_dir(),
            trans.t('batch_image_filter'))
        if not paths:
            return
        set_last_dialog_dir(paths[0])
        existing = {os.path.normcase(os.path.abspath(frame.source_path))
                    for frame in self.frames if frame.source_path}
        paths = [path for path in paths if os.path.normcase(os.path.abspath(path)) not in existing]
        if not paths:
            return

        def load(cancel, progress):
            from core.frame_storage import store_frame
            from core.image_loader import (count_frames, iter_decoded_sources,
                                           source_dimensions, _DecodeCancelled,
                                           _is_dng_tiff)
            result = []
            for index, path in enumerate(paths):
                if cancel():
                    raise RegistrationCancelled()
                ext = os.path.splitext(path)[1].lower()
                mapped_tiff = ext in ('.tif', '.tiff') and not _is_dng_tiff(path)
                encoded_bytes = 0 if mapped_tiff else os.path.getsize(path) * 2
                require_memory(encoded_bytes)
                count = count_frames(path)
                largest = 0
                for page in range(count):
                    if cancel():
                        raise RegistrationCancelled()
                    width, height = source_dimensions(path, page)
                    largest = max(largest, width * height)
                require_memory(largest * 32 + encoded_bytes)
                decoder = iter_decoded_sources(
                    [path], 1.0, thread_count=1, should_cancel=cancel,
                    allow_transparent_png=True, disk_backed=True)
                try:
                    item = next(decoder)
                    if item.error is not None:
                        raise item.error
                    for page, image in enumerate(item.pages):
                        name = os.path.basename(path)
                        if len(item.pages) > 1:
                            name += f' [{page + 1}]'
                        mapped = image if isinstance(image, np.memmap) else store_frame(image)
                        result.append(make_frame(name, mapped, path))
                except _DecodeCancelled as exc:
                    raise RegistrationCancelled() from exc
                finally:
                    decoder.close()
                progress(index + 1, len(paths))
            return result

        self._run(load, trans.t('msg_loading_stack'), self._append_frames)

    def _add_outputs(self):
        manager = getattr(self.window_main, 'output_manager', None)
        output_list = getattr(self.window_main, 'output_list', None)
        records = []
        if manager is not None and output_list is not None:
            for row in range(output_list.count()):
                record = manager.record_for_row(row)
                if record is not None and isinstance(record.image, np.ndarray):
                    records.append(record)
        if not records:
            QMessageBox.information(self, trans.t('mosaic_title'), trans.t('msg_export_all_none'))
            return
        picker = QDialog(self)
        picker.setWindowTitle(trans.t('label_outputs_all'))
        layout = QVBoxLayout(picker)
        layout.addWidget(QLabel(trans.t('mosaic_choose_hint')))
        choices = QListWidget()
        choices.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for record in records:
            choices.addItem(record.label)
        layout.addWidget(choices)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(picker.accept)
        buttons.rejected.connect(picker.reject)
        layout.addWidget(buttons)
        if picker.exec() != QDialog.DialogCode.Accepted:
            return
        chosen = [records[choices.row(item)] for item in choices.selectedItems()]
        if not chosen:
            return

        def prepare(cancel, progress):
            result = []
            for index, record in enumerate(chosen):
                if cancel():
                    raise RegistrationCancelled()
                result.append(make_frame(record.label, record.image))
                progress(index + 1, len(chosen))
            return result

        self._run(prepare, trans.t('msg_loading_stack'), self._append_frames)

    def _append_frames(self, frames):
        if not frames:
            return
        old = self._current_transforms()
        self.frames.extend(frames)
        initial = initial_transforms(self.frames)
        self.transforms = old + initial[len(old):]
        self.connected = [False] * len(self.frames)
        self._rebuild()
        self.list.setCurrentRow(len(self.frames) - 1)
        self.status.setText(trans.t('mosaic_manual'))
        QTimer.singleShot(0, self._fit)

    def _remove_selected(self):
        index = self.list.currentRow()
        if not 0 <= index < len(self.frames):
            return
        self.transforms = self._current_transforms()
        del self.frames[index]
        del self.transforms[index]
        del self.connected[index]
        self._rebuild()
        self.list.setCurrentRow(min(index, len(self.frames) - 1))
        self.status.setText(trans.t('mosaic_manual') if self.frames else trans.t('mosaic_empty'))

    def _rebuild(self):
        self._updating = True
        try:
            self.scene.clear()
            self.list.clear()
            self.items = []
            for index, (frame, matrix) in enumerate(zip(self.frames, self.transforms)):
                label = frame.name
                if index > 0 and not self.connected[index]:
                    label += '  ' + trans.t('mosaic_unmatched_mark')
                self.list.addItem(label)
                item = _TileItem(index, cv2_to_pixmap(frame.preview), self._item_changed)
                sx = frame.image.shape[1] / frame.preview.shape[1]
                sy = frame.image.shape[0] / frame.preview.shape[0]
                item.setTransform(QTransform(matrix[0, 0] * sx, matrix[1, 0] * sx, 0,
                                             matrix[0, 1] * sy, matrix[1, 1] * sy, 0,
                                             matrix[0, 2], matrix[1, 2], 1))
                item.setZValue(index)
                self.scene.addItem(item)
                self.items.append(item)
            self._update_size()
        finally:
            self._updating = False
        self._refresh_controls()

    def _current_transforms(self):
        result = []
        for matrix, item in zip(self.transforms, self.items):
            current = matrix.copy()
            current[0, 2] += item.pos().x()
            current[1, 2] += item.pos().y()
            result.append(current)
        return result

    def _item_changed(self, item, change):
        if self._updating:
            return
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged and item.isSelected():
            self.list.setCurrentRow(item.index)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.connected = [False] * len(self.frames)
            for index, frame in enumerate(self.frames):
                suffix = '  ' + trans.t('mosaic_unmatched_mark') if index else ''
                self.list.item(index).setText(frame.name + suffix)
            self.status.setText(trans.t('mosaic_manual'))
            self._update_size()
            if item.index == self.list.currentRow():
                self._sync_position(item.index)

    def _select_row(self, index):
        if self._updating:
            return
        if 0 <= index < len(self.items):
            self._updating = True
            try:
                self.scene.clearSelection()
                self.items[index].setSelected(True)
                self._sync_position(index)
                with QSignalBlocker(self.opacity):
                    self.opacity.setValue(round(self.items[index].opacity() * 100))
            finally:
                self._updating = False
        self._refresh_controls()

    def _sync_position(self, index):
        transform = self._current_transforms()[index]
        with QSignalBlocker(self.x_spin), QSignalBlocker(self.y_spin):
            self.x_spin.setValue(float(transform[0, 2]))
            self.y_spin.setValue(float(transform[1, 2]))

    def _position_changed(self):
        index = self.list.currentRow()
        if self._updating or not 0 <= index < len(self.items):
            return
        matrix = self.transforms[index]
        self.items[index].setPos(QPointF(self.x_spin.value() - matrix[0, 2],
                                         self.y_spin.value() - matrix[1, 2]))

    def _opacity_changed(self, value):
        index = self.list.currentRow()
        if 0 <= index < len(self.items):
            self.items[index].setOpacity(value / 100.0)

    def _update_size(self):
        if not self.frames:
            self.size_label.clear()
            return
        try:
            left, top, right, bottom = canvas_bounds(self.frames, self._current_transforms())
            self.size_label.setText(trans.t('mosaic_size').format(width=right-left, height=bottom-top))
        except (ValueError, OverflowError):
            self.size_label.setText(trans.t('mosaic_invalid'))

    def _fit(self):
        bounds = self.scene.itemsBoundingRect()
        if not bounds.isEmpty():
            self.scene.setSceneRect(bounds.adjusted(-20, -20, 20, 20))
            self.canvas.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _auto_align(self):
        frames = tuple(self.frames)
        transforms = self._current_transforms()

        def align(cancel, progress):
            return auto_align(frames, transforms, cancel, progress)

        def complete(result):
            self.transforms = list(result.transforms)
            self.connected = list(result.connected)
            self._rebuild()
            self.list.setCurrentRow(0)
            unmatched = sum(not value for value in result.connected)
            if unmatched:
                self.status.setText(trans.t('mosaic_unmatched').format(count=unmatched))
            else:
                residual = float(np.median([match[3] for match in result.matches]))
                self.status.setText(trans.t('mosaic_aligned').format(
                    count=len(result.matches), residual=residual))
            self._fit()

        self._run(align, trans.t('render_stage_align'), complete)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, trans.t('mosaic_save'),
                                              get_last_dialog_dir(), trans.t('mosaic_save_filter'))
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += '.tif'
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.tif', '.tiff', '.png'):
            QMessageBox.warning(self, trans.t('mosaic_title'), trans.t('mosaic_save_type'))
            return
        target = os.path.normcase(os.path.abspath(path))
        if any(frame.source_path and os.path.normcase(os.path.abspath(frame.source_path)) == target
               for frame in self.frames):
            QMessageBox.warning(self, trans.t('mosaic_title'), trans.t('mosaic_source_conflict'))
            return
        set_last_dialog_dir(path)
        frames = tuple(self.frames)
        transforms = self._current_transforms()

        def export(cancel, progress):
            image, _origin = render_mosaic(frames, transforms, should_cancel=cancel,
                                           progress=progress)
            if cancel():
                raise RegistrationCancelled()
            if ext == '.png':
                require_memory(image.nbytes * 3)
            if not imwrite_auto(path, image, exif=b'', should_cancel=cancel,
                                stream_tiff=ext in ('.tif', '.tiff')):
                raise OSError(trans.t('msg_save_failed_info_write'))
            return path

        self._run(export, trans.t('mosaic_exporting'),
                  lambda output: self.status.setText(trans.t('depth_saved').format(path=output)))

    def _cancel(self):
        if self.worker is not None:
            self.worker.cancel()
            self.cancel.setEnabled(False)
            self.status.setText(trans.t('compare_stopping'))

    def reject(self):
        if self.worker is not None and self.worker.isRunning():
            self._close_pending = True
            self._cancel()
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self._close_pending = True
            self._cancel()
            event.ignore()
        else:
            super().closeEvent(event)
