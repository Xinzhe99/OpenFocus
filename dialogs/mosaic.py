"""Independent workspace for arranging and exporting microscope mosaics."""
import math
import os
import threading

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QSignalBlocker, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPalette, QTransform
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import (
    QApplication, QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from core.mosaic import (auto_align, canvas_bounds, grid_transforms,
                         initial_transforms, make_frame, render_mosaic)
from core.registration import RegistrationCancelled
from core.resource_policy import heavy_job, require_memory
from locales import trans
from utils.image_utils import cv2_to_pixmap, imwrite_auto
from utils.settings_store import (get_last_dialog_dir, image_save_dir, remember_image_save,
                                  set_last_dialog_dir)
from utils.path_identity import same_path


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


class _MosaicList(QListWidget):
    """Paint the empty-state invitation without intercepting list events."""

    def __init__(self, parent=None):
        super().__init__(parent)
        trans.languageChanged.connect(self._refresh_hint)

    def _refresh_hint(self):
        self.viewport().update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.count():
            return
        painter = QPainter(self.viewport())
        font = self.font()
        if font.pointSizeF() > 0:
            font.setPointSizeF(max(8.0, font.pointSizeF() - 1.0))
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(self.palette().color(QPalette.ColorGroup.Disabled,
                                            QPalette.ColorRole.Text))
        painter.drawText(self.viewport().rect().adjusted(14, 12, -14, -12),
                         Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                         trans.t('mosaic_drop_hint'))


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
        self.preview_opacities = []
        self.worker = None
        self._close_pending = False
        self._updating = False
        self._build_ui()
        self.setAcceptDrops(True)
        for target in (self.list, self.list.viewport(), self.canvas, self.canvas.viewport()):
            target.setAcceptDrops(True)
            target.installEventFilter(self)
        self.scene.selectionChanged.connect(self._scene_selection_changed)
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
        self.list = _MosaicList()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.list.setMinimumHeight(120)
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.add_files = QPushButton(trans.t('mosaic_add_files'))
        self.add_outputs = QPushButton(trans.t('mosaic_add_outputs'))
        row.addWidget(self.add_files)
        row.addWidget(self.add_outputs)
        left.addLayout(row)
        self.remove = QPushButton(trans.t('batch_btn_remove'))
        left.addWidget(self.remove)
        self.add_folder = QPushButton(trans.t('action_open_folder'))
        self.add_folder.clicked.connect(self._add_folder)
        left.addWidget(self.add_folder)
        grid = QFormLayout()
        self.grid_rows, self.grid_cols = QSpinBox(), QSpinBox()
        for spin in (self.grid_rows, self.grid_cols):
            spin.setRange(1, 10000)
        self.grid_cols.setValue(2)
        grid.addRow(trans.t('mosaic_grid_rows'), self.grid_rows)
        grid.addRow(trans.t('mosaic_grid_cols'), self.grid_cols)
        self.grid_order = QComboBox()
        for key, value in (('mosaic_by_row', 'byrow'), ('mosaic_by_column', 'bycolumn')):
            self.grid_order.addItem(trans.t(key), value)
        grid.addRow(trans.t('mosaic_capture_order'), self.grid_order)
        self.grid_direction = QComboBox()
        for key, value in (('mosaic_unidirectional', 'unidirectional'), ('mosaic_zigzag', 'zigzag')):
            self.grid_direction.addItem(trans.t(key), value)
        grid.addRow(trans.t('mosaic_capture_direction'), self.grid_direction)
        self.grid_corner = QComboBox()
        for key, value in (('mosaic_top_left', 'topleft'), ('mosaic_top_right', 'topright'),
                           ('mosaic_bottom_left', 'bottomleft'), ('mosaic_bottom_right', 'bottomright')):
            self.grid_corner.addItem(trans.t(key), value)
        grid.addRow(trans.t('mosaic_start_corner'), self.grid_corner)
        self.grid_overlap = QDoubleSpinBox()
        self.grid_overlap.setRange(0, 95)
        self.grid_overlap.setSuffix(' %')
        self.grid_overlap.setValue(20)
        grid.addRow(trans.t('mosaic_grid_overlap'), self.grid_overlap)
        left.addLayout(grid)
        self.grid_apply = QPushButton(trans.t('mosaic_apply_grid'))
        left.addWidget(self.grid_apply)
        self.grid_hint = QLabel()
        self.grid_hint.setWordWrap(True)
        left.addWidget(self.grid_hint)
        position = QFormLayout()
        self.x_spin, self.y_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.x_spin, self.y_spin):
            spin.setRange(-1000000000, 1000000000)
            spin.setDecimals(1)
            spin.setSingleStep(1)
        position.addRow(trans.t('label_x'), self.x_spin)
        position.addRow(trans.t('label_y'), self.y_spin)
        self.scale_spin, self.rotation_spin = QDoubleSpinBox(), QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 10000)
        self.scale_spin.setDecimals(3)
        self.scale_spin.setSuffix(' %')
        self.scale_spin.setValue(100)
        self.rotation_spin.setRange(-180, 180)
        self.rotation_spin.setDecimals(3)
        self.rotation_spin.setSuffix(' °')
        position.addRow(trans.t('mosaic_scale'), self.scale_spin)
        position.addRow(trans.t('mosaic_rotation'), self.rotation_spin)
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        splitter.addWidget(scroll)
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
        self.grid_apply.clicked.connect(self._apply_grid)
        for control in (self.grid_rows, self.grid_cols):
            control.valueChanged.connect(self._refresh_controls)
        self.auto.clicked.connect(self._auto_align)
        self.fit.clicked.connect(self._fit)
        self.save.clicked.connect(self._save)
        self.cancel.clicked.connect(self._cancel)
        self.close_button.clicked.connect(self.close)
        self.list.currentRowChanged.connect(self._select_row)
        self.x_spin.valueChanged.connect(self._position_changed)
        self.y_spin.valueChanged.connect(self._position_changed)
        self.scale_spin.valueChanged.connect(self._geometry_changed)
        self.rotation_spin.valueChanged.connect(self._geometry_changed)
        self.opacity.valueChanged.connect(self._opacity_changed)
        self.opacity.setToolTip(trans.t('mosaic_preview_hint'))
        preview_hint = QLabel(trans.t('mosaic_preview_hint'))
        preview_hint.setWordWrap(True)
        left.addWidget(preview_hint)
        for button in (self.add_files, self.add_folder, self.add_outputs, self.remove, self.grid_apply,
                       self.auto, self.fit, self.cancel, self.save, self.close_button):
            button.setAutoDefault(False)

    def _refresh_controls(self):
        busy = self.worker is not None
        selected = 0 <= self.list.currentRow() < len(self.frames)
        self.add_files.setEnabled(not busy)
        self.add_outputs.setEnabled(not busy)
        self.add_folder.setEnabled(not busy)
        count = self.grid_rows.value() * self.grid_cols.value()
        self.grid_hint.setText(trans.t('mosaic_grid_hint').format(cells=count, count=len(self.frames)))
        self.grid_apply.setEnabled(not busy and bool(self.frames) and count == len(self.frames))
        for control in (self.grid_rows, self.grid_cols, self.grid_order, self.grid_direction,
                        self.grid_corner, self.grid_overlap):
            control.setEnabled(not busy)
        self.remove.setEnabled(not busy and selected)
        self.x_spin.setEnabled(not busy and selected)
        self.y_spin.setEnabled(not busy and selected)
        self.scale_spin.setEnabled(not busy and selected)
        self.rotation_spin.setEnabled(not busy and selected)
        self.opacity.setEnabled(not busy and selected)
        self.auto.setEnabled(not busy and len(self.frames) >= 2)
        self.fit.setEnabled(bool(self.frames))
        self.save.setEnabled(not busy and bool(self.frames))
        self.list.setEnabled(not busy)
        for item in self.items:
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not busy)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, not busy)
        self.cancel.setVisible(busy)
        self.cancel.setEnabled(busy)
        self.progress.setVisible(busy)

    def _run(self, operation, status, success):
        if self.worker is not None:
            return
        self.status.setText(status)
        self.progress.setRange(0, 0)
        worker = _Worker(operation, self)
        self.worker = worker
        worker.progress.connect(self._progress)
        worker.result.connect(lambda value: success(value)
                              if self.worker is worker and not worker._stop.is_set()
                              and not self._close_pending else None)
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
        self._select_row(self.list.currentRow())
        if self._close_pending:
            self.close()

    def _error(self, error):
        detail = f"{self.status.text()}\n{error}"
        self.status.setText(trans.t('alignment_report_failed'))
        QMessageBox.warning(self, trans.t('mosaic_title'), detail)

    @staticmethod
    def _drop_paths(event):
        return [path for url in event.mimeData().urls()
                if url.isLocalFile() and (path := url.toLocalFile())]

    def dragEnterEvent(self, event):
        if self.worker is None and self._drop_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        paths = self._drop_paths(event)
        if self.worker is not None or not paths:
            event.ignore()
            return
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        self._import_paths(paths, skipped=len(event.mimeData().urls()) - len(paths))

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            self.dragEnterEvent(event)
            return True
        if event.type() == QEvent.Type.Drop:
            self.dropEvent(event)
            return True
        if (event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
                and event.modifiers() == Qt.KeyboardModifier.NoModifier):
            self._remove_selected()
            return True
        return super().eventFilter(watched, event)

    def _add_folder(self):
        if self.worker is not None:
            return
        path = FileDialog.getExistingDirectory(self, trans.t('action_open_folder'),
                                               get_last_dialog_dir())
        if path:
            set_last_dialog_dir(path)
            self._import_paths([path])

    def _add_files(self):
        if self.worker is not None:
            return
        paths, _ = FileDialog.getOpenFileNames(
            self, trans.t('mosaic_add_files'), get_last_dialog_dir(),
            trans.t('batch_image_filter'))
        if not paths:
            return
        set_last_dialog_dir(paths[0])
        self._import_paths(paths)

    def _import_paths(self, sources, skipped=0):
        if self.worker is not None or not sources:
            return
        sources = tuple(sources)
        skipped_count = skipped
        existing = tuple(frame.source_path for frame in self.frames if frame.source_path)

        def load(cancel, progress):
            from core.frame_storage import store_frame
            from core.image_loader import (ImageStackLoader, count_frames, iter_decoded_sources,
                                           source_dimensions, _DecodeCancelled,
                                           _is_dng_tiff)
            from utils.sorting import natural_sort_key
            paths, seen, skipped = [], list(existing), skipped_count

            def include(path):
                nonlocal skipped
                if cancel():
                    raise RegistrationCancelled()
                if (os.path.splitext(path)[1].lower() not in ImageStackLoader.SUPPORTED_FORMATS
                        or any(same_path(path, previous) for previous in seen)):
                    skipped += 1
                else:
                    paths.append(path)
                    seen.append(path)

            for source in sources:
                if cancel():
                    raise RegistrationCancelled()
                if os.path.isdir(source):
                    names = sorted(os.listdir(source), key=natural_sort_key)
                    if not names:
                        skipped += 1
                    for name in names:
                        path = os.path.join(source, name)
                        if os.path.isfile(path):
                            include(path)
                        else:
                            skipped += 1
                elif os.path.isfile(source):
                    include(source)
                else:
                    skipped += 1
            paths.sort(key=natural_sort_key)
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
            return result, skipped

        def complete(result):
            frames, skipped = result
            self._append_frames(frames)
            self.status.setText(trans.t('mosaic_import_status').format(
                added=len(frames), skipped=skipped, total=len(self.frames)))

        self._run(load, trans.t('msg_loading_stack'), complete)

    def _add_outputs(self):
        if self.worker is not None:
            return
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
        self.preview_opacities.extend([1.0] * len(frames))
        self.connected = [False] * len(self.frames)
        self._rebuild()
        self.list.setCurrentRow(len(self.frames) - 1)
        self.status.setText(trans.t('mosaic_manual'))
        QTimer.singleShot(0, self._fit)

    def _remove_selected(self):
        if self.worker is not None:
            return
        index = self.list.currentRow()
        if not 0 <= index < len(self.frames):
            return
        self.transforms = self._current_transforms()
        del self.frames[index]
        del self.transforms[index]
        del self.connected[index]
        del self.preview_opacities[index]
        self._rebuild()
        self.list.setCurrentRow(min(index, len(self.frames) - 1))
        self.status.setText(trans.t('mosaic_manual') if self.frames else trans.t('mosaic_empty'))

    def _apply_grid(self):
        if self.worker is not None or not self.grid_apply.isEnabled():
            return
        self.transforms = grid_transforms(
            self.frames, self.grid_rows.value(), self.grid_cols.value(),
            order=self.grid_order.currentData(), direction=self.grid_direction.currentData(),
            start_corner=self.grid_corner.currentData(), overlap_pct=self.grid_overlap.value())
        self.connected = [False] * len(self.frames)
        self._rebuild()
        self.status.setText(trans.t('mosaic_grid_applied'))
        self._fit()

    def _rebuild(self):
        selected = self.list.currentRow()
        self._updating = True
        try:
            self.scene.clear()
            self.list.clear()
            self.items = []
            for index, (frame, matrix) in enumerate(zip(self.frames, self.transforms)):
                label = frame.name
                if index > 0 and not self.connected[index]:
                    label += '  ' + trans.t('mosaic_unmatched_mark')
                pixmap = cv2_to_pixmap(frame.preview)
                entry = QListWidgetItem(QIcon(pixmap), label)
                entry.setToolTip(frame.source_path or frame.name)
                self.list.addItem(entry)
                item = _TileItem(index, pixmap, self._item_changed)
                item.setToolTip(frame.name)
                item.setOpacity(self.preview_opacities[index])
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
        self.list.setCurrentRow(min(selected, len(self.frames) - 1))
        self._select_row(self.list.currentRow())

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
        if self.worker is not None:
            return
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.connected = [False] * len(self.frames)
            for index, frame in enumerate(self.frames):
                suffix = '  ' + trans.t('mosaic_unmatched_mark') if index else ''
                self.list.item(index).setText(frame.name + suffix)
            self.status.setText(trans.t('mosaic_manual'))
            self._update_size()
            if item.index == self.list.currentRow():
                self._sync_position(item.index)

    def _scene_selection_changed(self):
        if self._updating or self.worker is not None:
            return
        selected = self.scene.selectedItems()
        index = selected[0].index if selected else -1
        self.list.setCurrentRow(index)
        self._select_row(index)

    def _select_row(self, index):
        if self._updating or self.worker is not None:
            return
        self._updating = True
        try:
            self.scene.clearSelection()
            if 0 <= index < len(self.items):
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
        with QSignalBlocker(self.scale_spin), QSignalBlocker(self.rotation_spin):
            self.scale_spin.setValue(math.hypot(transform[0, 0], transform[1, 0]) * 100)
            self.rotation_spin.setValue(math.degrees(math.atan2(transform[1, 0], transform[0, 0])))

    def _position_changed(self):
        index = self.list.currentRow()
        if self._updating or self.worker is not None or not 0 <= index < len(self.items):
            return
        matrix = self.transforms[index]
        self.items[index].setPos(QPointF(self.x_spin.value() - matrix[0, 2],
                                         self.y_spin.value() - matrix[1, 2]))

    def _geometry_changed(self):
        index = self.list.currentRow()
        if self._updating or self.worker is not None or not 0 <= index < len(self.items):
            return
        self.transforms = self._current_transforms()
        angle = math.radians(self.rotation_spin.value())
        scale = self.scale_spin.value() / 100
        a, b = scale * math.cos(angle), scale * math.sin(angle)
        self.transforms[index][:2, :2] = ((a, -b), (b, a))
        self.connected = [False] * len(self.frames)
        self._rebuild()
        self.status.setText(trans.t('mosaic_manual'))

    def _opacity_changed(self, value):
        index = self.list.currentRow()
        if self.worker is None and 0 <= index < len(self.items):
            self.items[index].setOpacity(value / 100.0)
            self.preview_opacities[index] = value / 100.0

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
        if self.worker is not None or len(self.frames) < 2:
            return
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
                scales = [math.hypot(matrix[0, 0], matrix[1, 0]) * 100
                          for matrix in result.transforms]
                self.status.setText(trans.t('mosaic_aligned_geometry').format(
                    count=len(result.matches), residual=residual,
                    minimum=min(scales), maximum=max(scales)))
            self._fit()

        self._run(align, trans.t('render_stage_align'), complete)

    def _save(self):
        if self.worker is not None or not self.frames:
            return
        path, _ = FileDialog.getSaveFileName(self, trans.t('mosaic_save'),
                                              image_save_dir(), trans.t('mosaic_save_filter'))
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += '.tif'
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.tif', '.tiff', '.png'):
            QMessageBox.warning(self, trans.t('mosaic_title'), trans.t('mosaic_save_type'))
            return
        if any(same_path(frame.source_path, path) for frame in self.frames):
            QMessageBox.warning(self, trans.t('mosaic_title'), trans.t('mosaic_source_conflict'))
            return
        remember_image_save(path)
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
        if self.worker is not None:
            self._close_pending = True
            self._cancel()
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self._close_pending = True
            self._cancel()
            event.ignore()
        else:
            super().closeEvent(event)
