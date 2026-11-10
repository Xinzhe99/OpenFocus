"""Source placement, bounded planning, control points and mosaic export."""
from dataclasses import replace
import math
import os
import threading

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QSignalBlocker, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPalette, QTransform
from utils.file_dialogs import FileDialog, filter_suffix
from PyQt6.QtWidgets import (
    QApplication, QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from core.mosaic import (auto_align, canvas_bounds, grid_transforms,
                         initial_transforms, make_frame, plan_mosaic, preview_mosaic, render_mosaic)
from dialogs.mosaic_panels import MosaicSettingsPanel
from dialogs.mosaic_control_points import MosaicControlPointsDialog
from dialogs.mosaic_mask import MosaicMaskDialog
from core.registration import RegistrationCancelled
from core.resource_policy import heavy_job, require_memory
from locales import trans
from utils.image_utils import cv2_to_pixmap, imwrite_auto
from utils.settings_store import (get_last_dialog_dir, image_save_dir, image_save_filter,
                                  remember_image_save, set_last_dialog_dir)
from utils.path_identity import same_path


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise RegistrationCancelled()


def _copy_rows(target, source, cancel):
    for y in range(0, source.shape[0], 128):
        _check_cancel(cancel)
        target[y:y + 128] = source[y:y + 128]
    if isinstance(target, np.memmap):
        target.flush()


def _decode_pages(path, cancel):
    """Use the shared iterator, with the existing descriptor's decode budget.

    Pages in a TIFF are decoded serially; admission is the largest *decoder*
    workspace rather than an inflated estimate for all source pixels together.
    Every returned page owns independent disk-backed storage.
    """
    from core.frame_storage import store_frame
    from core.image_loader import iter_decoded_sources, _DecodeCancelled
    from core.source_frames import scan_source, frame_resolution_bytes
    _check_cancel(cancel)
    headers = scan_source(path, header_only=True, should_cancel=cancel)
    require_memory(max((frame_resolution_bytes(frame) for frame in headers), default=0))
    decoder = iter_decoded_sources([path], 1.0, thread_count=1,
        should_cancel=cancel, allow_transparent_png=True, disk_backed=True)
    try:
        item = next(decoder)
        if item.error is not None:
            raise item.error
        if not item.pages:
            raise ValueError(f'Could not decode source: {path}')
        pages = []
        for image in item.pages:
            _check_cancel(cancel)
            pages.append(image if isinstance(image, np.memmap)
                         else store_frame(image, prefer_ram=False))
        return tuple(pages)
    except _DecodeCancelled as error:
        raise RegistrationCancelled() from error
    finally:
        decoder.close()


def _source_exif(path, page=0):
    from core.image_loader import _read_exif_bytes
    if not page:
        return bytes(_read_exif_bytes(path) or b'')
    from PIL import Image
    from utils.image_utils import export_exif, read_image_exif
    with Image.open(path) as image:
        image.seek(page)
        metadata = read_image_exif(image)
        safe = export_exif(metadata.tobytes()) if metadata else None
        return safe.tobytes() if safe is not None else b''


def _exposure_time(exif):
    if not exif:
        return None
    from PIL import Image
    try:
        metadata = Image.Exif()
        metadata.load(exif)
        value = metadata.get(33434)
        if value is None:
            value = metadata.get_ifd(34665).get(33434)
        if isinstance(value, tuple) and len(value) == 2:
            value = value[0] / value[1]
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds > 0 else None
    except (ValueError, TypeError, KeyError, OSError, ZeroDivisionError, SyntaxError):
        return None


def _mask_preview(mask, shape):
    if mask is None:
        return None
    height, width = shape
    step = max(1, math.ceil(mask.shape[0] / height), math.ceil(mask.shape[1] / width))
    sample = mask[::step, ::step]
    if sample.dtype.kind == 'b':
        sample = sample.astype(np.uint8) * 255
    return cv2.resize(sample, (width, height), interpolation=cv2.INTER_NEAREST)


def _paint_mask(base, shape, strokes, cancel, progress):
    """Commit pending native circles into a new mapping; never mutate the base."""
    if not strokes:
        return base
    from core.frame_storage import create_frame
    dtype = base.dtype if base is not None and base.dtype.kind != 'b' else np.dtype(np.uint8)
    maximum = np.iinfo(dtype).max
    height, width = shape
    require_memory(min(128, height) * width * 24)
    mask = create_frame(shape, dtype)
    for y in range(0, height, 128):
        _check_cancel(cancel)
        if base is None:
            mask[y:y + 128] = maximum
        elif base.dtype.kind == 'b':
            mask[y:y + 128] = base[y:y + 128].astype(dtype) * maximum
        else:
            mask[y:y + 128] = base[y:y + 128]
    total = max(1, len(strokes))
    for index, (cx, cy, radius, include) in enumerate(strokes):
        _check_cancel(cancel)
        left, right = max(0, math.ceil(cx - radius)), min(width, math.floor(cx + radius) + 1)
        top, bottom = max(0, math.ceil(cy - radius)), min(height, math.floor(cy + radius) + 1)
        if left < right and top < bottom:
            xx = (np.arange(left, right, dtype=np.float64) - cx) ** 2
            for y in range(top, bottom, 128):
                _check_cancel(cancel)
                end = min(bottom, y + 128)
                yy = (np.arange(y, end, dtype=np.float64) - cy) ** 2
                inside = yy[:, None] + xx[None, :] <= radius * radius
                strip = mask[y:end, left:right]
                strip[inside] = maximum if include else 0
        progress(index + 1, total)
    mask.flush()
    return mask


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
        except (RegistrationCancelled, InterruptedError):
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
            self.resize(min(1600, max(480, available.width() - 80)),
                        min(950, max(360, available.height() - 100)))
        else:
            self.resize(1500, 900)
        self.frames = []
        self.transforms = []
        self.items = []
        self.connected = []
        self.preview_opacities = []
        self.worker = None
        self._close_pending = False
        self._updating = False
        self._plan = None
        self._alignment = None
        self._revision = 0
        self._plan_revision = -1
        self._preview_item = None
        self.control_points = ()
        self.qa = None
        self._selected_points = ()
        self._cp_dialog = None
        self._mask_dialog = None
        self._mask_imported = None
        self._output_source_paths = []
        self._build_ui()
        self.setAcceptDrops(True)
        for target in (self.list, self.list.viewport(), self.canvas, self.canvas.viewport()):
            target.setAcceptDrops(True)
            target.installEventFilter(self)
        self.scene.selectionChanged.connect(self._scene_selection_changed)
        self._refresh_controls()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        self._source_labels = []
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
        self.mask_button = QPushButton(trans.t('mosaic_mask'))
        self.mask_button.clicked.connect(self._edit_mask)
        left.addWidget(self.mask_button)
        self.add_folder = QPushButton(trans.t('action_open_folder'))
        self.add_folder.clicked.connect(self._add_folder)
        left.addWidget(self.add_folder)
        grid = QFormLayout()
        grid.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.grid_rows, self.grid_cols = QSpinBox(), QSpinBox()
        for spin in (self.grid_rows, self.grid_cols):
            spin.setRange(1, 10000)
        self.grid_cols.setValue(2)
        self._source_row(grid, 'mosaic_grid_rows', self.grid_rows)
        self._source_row(grid, 'mosaic_grid_cols', self.grid_cols)
        self.grid_order = QComboBox()
        for key, value in (('mosaic_by_row', 'byrow'), ('mosaic_by_column', 'bycolumn')):
            self.grid_order.addItem(trans.t(key), value)
        self._source_row(grid, 'mosaic_capture_order', self.grid_order)
        self.grid_direction = QComboBox()
        for key, value in (('mosaic_unidirectional', 'unidirectional'), ('mosaic_zigzag', 'zigzag')):
            self.grid_direction.addItem(trans.t(key), value)
        self._source_row(grid, 'mosaic_capture_direction', self.grid_direction)
        self.grid_corner = QComboBox()
        for key, value in (('mosaic_top_left', 'topleft'), ('mosaic_top_right', 'topright'),
                           ('mosaic_bottom_left', 'bottomleft'), ('mosaic_bottom_right', 'bottomright')):
            self.grid_corner.addItem(trans.t(key), value)
        self._source_row(grid, 'mosaic_start_corner', self.grid_corner)
        self.grid_overlap = QDoubleSpinBox()
        self.grid_overlap.setRange(0, 95)
        self.grid_overlap.setSuffix(' %')
        self.grid_overlap.setValue(20)
        self._source_row(grid, 'mosaic_grid_overlap', self.grid_overlap)
        left.addLayout(grid)
        self.grid_apply = QPushButton(trans.t('mosaic_apply_grid'))
        left.addWidget(self.grid_apply)
        self.grid_hint = QLabel()
        self.grid_hint.setWordWrap(True)
        left.addWidget(self.grid_hint)
        position = QFormLayout()
        position.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.x_spin, self.y_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.x_spin, self.y_spin):
            spin.setRange(-1000000000, 1000000000)
            spin.setDecimals(1)
            spin.setSingleStep(1)
            spin.setToolTip("\n\n".join((trans.t('mosaic_intro'), trans.t('mosaic_manual'))))
        self._source_row(position, 'label_x', self.x_spin)
        self._source_row(position, 'label_y', self.y_spin)
        self.scale_spin, self.rotation_spin = QDoubleSpinBox(), QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 10000)
        self.scale_spin.setDecimals(3)
        self.scale_spin.setSuffix(' %')
        self.scale_spin.setValue(100)
        self.rotation_spin.setRange(-180, 180)
        self.rotation_spin.setDecimals(3)
        self.rotation_spin.setSuffix(' °')
        self._source_row(position, 'mosaic_scale', self.scale_spin)
        self._source_row(position, 'mosaic_rotation', self.rotation_spin)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(100)
        self._source_row(position, 'mosaic_opacity', self.opacity)
        left.addLayout(position)
        self.auto = QPushButton(trans.t('mosaic_auto'))
        self.fit = QPushButton(trans.t('label_fit'))
        self.size_label = QLabel('')
        self.size_label.setWordWrap(True)
        left.addWidget(self.size_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        self.tabs = QTabWidget()
        assistant = QWidget()
        assistant_layout = QVBoxLayout(assistant)
        intro = QLabel(trans.t('mosaic_intro'))
        intro.setWordWrap(True)
        assistant_layout.addWidget(intro)
        self._assistant_intro = intro
        self._assistant_buttons = []
        for key, callback in (('mosaic_add_files', self._add_files),
                              ('action_open_folder', self._add_folder),
                              ('mosaic_add_outputs', self._add_outputs)):
            button = QPushButton(trans.t(key))
            button.clicked.connect(callback)
            button.setAutoDefault(False)
            assistant_layout.addWidget(button)
            self._assistant_buttons.append((button, key))
        assistant_layout.addWidget(self.auto)
        self.cp_button = QPushButton(trans.t('mosaic_control_points'))
        self.cp_button.clicked.connect(self._edit_control_points)
        assistant_layout.addWidget(self.cp_button)
        assistant_layout.addWidget(self.fit)
        assistant_layout.addStretch()
        self.tabs.addTab(assistant, trans.t('mosaic_assistant'))
        self.tabs.addTab(scroll, trans.t('mosaic_sources'))
        self.settings = MosaicSettingsPanel()
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(self.settings)
        self.tabs.addTab(settings_scroll, trans.t('mosaic_editor'))
        self.tabs.setMinimumWidth(300)
        splitter.addWidget(self.tabs)
        self.scene = QGraphicsScene(self)
        self.canvas = _Canvas(self.scene)
        self.canvas.setToolTip("\n\n".join((trans.t('mosaic_intro'), trans.t('mosaic_manual'))))
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([410, 1120])
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
        self.preview_button = QPushButton(trans.t('mosaic_preview'))
        self.preview_button.clicked.connect(self._preview)
        row.addWidget(self.preview_button)
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
        self.settings.changed.connect(self._settings_changed)
        self.settings.overlaysChanged.connect(self._overlay_changed)
        self.settings.sourceSelected.connect(self.list.setCurrentRow)
        trans.languageChanged.connect(self._retranslate)
        for button in (self.add_files, self.add_folder, self.add_outputs, self.remove, self.grid_apply,
                       self.auto, self.fit, self.cancel, self.save, self.close_button,
                       self.mask_button, self.cp_button, self.preview_button):
            button.setAutoDefault(False)

    def _source_row(self, form, key, widget):
        label = QLabel(trans.t(key))
        label.setWordWrap(True)
        label.setBuddy(widget)
        form.addRow(label, widget)
        self._source_labels.append((label, key))

    def _refresh_controls(self):
        busy = self.worker is not None
        selected = 0 <= self.list.currentRow() < len(self.frames)
        planar = self.settings.projection.currentData() is None
        self.add_files.setEnabled(not busy)
        self.add_outputs.setEnabled(not busy)
        self.add_folder.setEnabled(not busy)
        count = self.grid_rows.value() * self.grid_cols.value()
        self.grid_hint.setText(trans.t('mosaic_grid_hint').format(cells=count, count=len(self.frames)))
        self.grid_apply.setEnabled(not busy and planar and bool(self.frames) and count == len(self.frames))
        for control in (self.grid_rows, self.grid_cols, self.grid_order, self.grid_direction,
                        self.grid_corner, self.grid_overlap):
            control.setEnabled(not busy and planar)
        self.remove.setEnabled(not busy and selected)
        self.x_spin.setEnabled(not busy and planar and selected)
        self.y_spin.setEnabled(not busy and planar and selected)
        self.scale_spin.setEnabled(not busy and planar and selected)
        self.rotation_spin.setEnabled(not busy and planar and selected)
        self.opacity.setEnabled(not busy and selected)
        self.auto.setEnabled(not busy and len(self.frames) >= 2)
        self.fit.setEnabled(bool(self.frames))
        self.save.setEnabled(not busy and bool(self.frames))
        self.list.setEnabled(not busy)
        self.settings.setEnabled(not busy)
        self.mask_button.setEnabled(not busy and selected)
        self.cp_button.setEnabled(not busy and len(self.frames) >= 2)
        self.preview_button.setEnabled(not busy and bool(self.frames))
        for button, _key in self._assistant_buttons:
            button.setEnabled(not busy)
        if self._cp_dialog is not None:
            self._cp_dialog.set_busy(busy)
        for item in self.items:
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not busy and planar)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, not busy)
        self.cancel.setVisible(busy)
        self.cancel.setEnabled(busy)
        self.progress.setVisible(busy)

    def _run(self, operation, status, success, settled=None):
        if self.worker is not None:
            return
        self.status.setText(status)
        self.progress.setRange(0, 0)
        worker = _Worker(operation, self)
        self.worker = worker
        worker.progress.connect(self._progress)

        def deliver(value):
            if self.worker is worker and not worker._stop.is_set() and not self._close_pending:
                try:
                    success(value)
                except Exception as error:
                    self._error(str(error))

        def failed(error):
            if not self._close_pending:
                self._error(error)

        def finished():
            if self.worker is worker:
                self.worker = None
                if settled is not None:
                    settled()
                self._refresh_controls()
                self._select_row(self.list.currentRow(), restore_placement=False)
                if self._close_pending:
                    self.close()

        worker.result.connect(deliver)
        worker.error.connect(failed)
        worker.cancelled.connect(lambda: self.status.setText(trans.t('compare_cancelled')))
        worker.finished.connect(finished)
        worker.finished.connect(worker.deleteLater)
        self._refresh_controls()
        worker.start()

    def _progress(self, value, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(value)

    def _error(self, error):
        self.status.setText(trans.t('alignment_report_failed'))
        detail = f"{self.status.text()}\n{error}"
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
            from core.image_loader import ImageStackLoader
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
                pages = _decode_pages(path, cancel)
                from core.image_loader import _read_icc_profile
                for page, image in enumerate(pages):
                    exif = _source_exif(path, page)
                    exposure = _exposure_time(exif)
                    name = os.path.basename(path)
                    if len(pages) > 1:
                        name += f' [{page + 1}]'
                    result.append(make_frame(name, image, path, preview_size=1024,
                        icc_profile=_read_icc_profile(path, page=page),
                        exif=exif, exposure_time=exposure))
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
        choices = QListWidget()
        choices.setToolTip(trans.t('mosaic_choose_hint'))
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

        snapshots = tuple((str(record.label), record.image,
            bytes(getattr(record.context, 'icc_profile', record.icc_profile) or b''),
            str(record.export_path or '') if record.annotations_baked else '',
            tuple(record.source_paths), str(record.export_path or '')) for record in chosen)

        def prepare(cancel, progress):
            from core.frame_storage import create_frame
            result, protected = [], []
            for index, (label, image, icc, metadata_path, source_paths, exported) in enumerate(snapshots):
                exif = _source_exif(metadata_path) if metadata_path else b''
                copied = create_frame(image.shape, image.dtype)
                _copy_rows(copied, image, cancel)
                result.append(make_frame(label, copied, preview_size=1024,
                    icc_profile=icc, exif=exif, exposure_time=_exposure_time(exif)))
                protected.extend(path for path in (*source_paths, exported) if path)
                progress(index + 1, len(snapshots))
            return result, tuple(protected)

        def complete(value):
            frames, protected = value
            self._output_source_paths.extend(protected)
            self._append_frames(frames)

        self._run(prepare, trans.t('msg_loading_stack'), complete)

    def _append_frames(self, frames):
        if not frames:
            return
        old = self._current_transforms()
        self._invalidate_plan()
        self.frames.extend(frames)
        initial = initial_transforms(self.frames)
        self.transforms = old + initial[len(old):]
        self.preview_opacities.extend([1.0] * len(frames))
        self.connected = [False] * len(self.frames)
        self.settings.set_frames(self.frames)
        self._rebuild()
        self.list.setCurrentRow(len(self.frames) - 1)
        self.status.clear()
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
        self.control_points = tuple(replace(cp,
            frame_a=cp.frame_a - (cp.frame_a > index),
            frame_b=cp.frame_b - (cp.frame_b > index))
            for cp in self.control_points if cp.frame_a != index and cp.frame_b != index)
        self._selected_points = ()
        self.qa = None
        self.settings.set_frames(self.frames)
        self._invalidate_plan()
        self._rebuild()
        self.list.setCurrentRow(min(index, len(self.frames) - 1))
        self.status.setText('' if self.frames else trans.t('mosaic_empty'))

    def _apply_grid(self):
        if self.worker is not None or not self.grid_apply.isEnabled():
            return
        self._invalidate_plan()
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
            self._preview_item = None
            self.list.clear()
            self.items = []
            for index, (frame, matrix) in enumerate(zip(self.frames, self.transforms)):
                label = frame.name + ('  ' + trans.t('mosaic_mask_present') if frame.mask is not None else '')
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
            self._invalidate_plan()
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

    def _select_row(self, index, restore_placement=True):
        if self._updating or self.worker is not None:
            return
        if restore_placement:
            self._show_placement()
        self.settings.select_frame(index)
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
        self._invalidate_plan()
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
        bounds = (self._preview_item.sceneBoundingRect()
                  if self._preview_item is not None else self.scene.itemsBoundingRect())
        if not bounds.isEmpty():
            self.scene.setSceneRect(bounds.adjusted(-20, -20, 20, 20))
            self.canvas.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _auto_align(self):
        self._align(tuple(cp for cp in self.control_points if cp.source == 'manual'),
                    detect_features=True)

    def _align(self, points, *, detect_features=False):
        if self.worker is not None or len(self.frames) < 2:
            return
        try:
            frames, transforms, spec, revision = self._snapshot()
            cameras = spec.cameras
            optimize_fov = bool(self.settings.optimize_fov.isChecked()) if cameras else False
        except (ValueError, TypeError) as error:
            self._error(str(error))
            return
        points = tuple(points)

        def align(cancel, progress):
            return auto_align(frames, transforms, cancel, progress,
                control_points=points, cameras=cameras, optimize_fov=optimize_fov,
                detect_features=detect_features)

        def complete(result):
            if revision != self._revision:
                return
            self.transforms = [matrix.copy() for matrix in result.transforms]
            self.connected = list(result.connected)
            self.control_points = tuple(result.control_points)
            self._selected_points = ()
            if result.cameras:
                self.settings.set_cameras(result.cameras)
            self._invalidate_plan()
            self._alignment = result
            self.qa = result.qa
            self._rebuild()
            if self._cp_dialog is not None:
                self._cp_dialog.set_result(self.transforms, self.qa, self.control_points,
                                           cameras=result.cameras)
            if self.qa is not None:
                self.status.setText(trans.t('mosaic_qa_status').format(
                    points=self.qa.cp_count, weak=sum(pair.weak for pair in self.qa.pairs),
                    disconnected=len(self.qa.disconnected_indices), residual=self.qa.residual_p95))
            else:
                self.status.setText(trans.t('mosaic_unmatched').format(
                    count=sum(not value for value in result.connected)))
            self._fit()

        self._run(align, trans.t('render_stage_align'), complete)

    def _snapshot(self):
        # GUI-only: workers receive immutable specs, owned pixels and copied matrices.
        frames = tuple(self.frames)
        transforms = tuple(matrix.copy() for matrix in self._current_transforms())
        spec = self.settings.render_spec(frames)
        return frames, transforms, spec, self._revision

    def _invalidate_plan(self, *, clear_alignment=True):
        if clear_alignment:
            self._alignment = None
            self.qa = None
        self._revision += 1
        self._plan = None
        self._plan_revision = -1
        self._show_placement()

    def _show_placement(self):
        had_preview = self._preview_item is not None
        if had_preview:
            self.scene.removeItem(self._preview_item)
            self._preview_item = None
        for item in self.items:
            item.setVisible(True)
        if had_preview:
            self._fit()

    def _settings_changed(self):
        if not self._updating:
            panorama = self.settings.projection.currentData() is not None
            try:
                changed_camera = bool(self._alignment is not None and (
                    bool(self._alignment.cameras) != panorama or
                    (self._alignment.cameras and
                     tuple(self._alignment.cameras) != self.settings.camera_specs())))
            except (TypeError, ValueError):
                changed_camera = True
            self._invalidate_plan(clear_alignment=changed_camera)
            self._refresh_controls()

    def _overlay_changed(self):
        if self.worker is None and self._plan is not None:
            self._preview()

    def _preview(self):
        if self.worker is not None or not self.frames:
            return
        try:
            frames, transforms, spec, revision = self._snapshot()
            options = self.settings.overlay_options()
        except (ValueError, TypeError) as error:
            self._error(str(error))
            return
        plan = self._plan if self._plan_revision == revision else None
        alignment = self._alignment
        points = tuple(self.control_points[index] for index in self._selected_points
                       if 0 <= index < len(self.control_points))

        def preview(cancel, progress):
            captured = plan if plan is not None else plan_mosaic(frames, transforms,
                spec=spec, alignment=alignment, should_cancel=cancel, progress=progress)
            image = preview_mosaic(captured, max_edge=1024, selected_points=points,
                should_cancel=cancel, **options)
            return captured, image

        def complete(value):
            captured, image = value
            if revision != self._revision:
                return
            self._plan, self._plan_revision = captured, revision
            self._show_placement()
            self._updating = True
            try:
                self.scene.clearSelection()
                for item in self.items:
                    item.setVisible(False)
                self._preview_item = self.scene.addPixmap(cv2_to_pixmap(image))
                self._preview_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
                self._preview_item.setTransform(QTransform.fromScale(
                    captured.size[0] / image.shape[1], captured.size[1] / image.shape[0]))
                self._preview_item.setPos(*captured.origin)
                self._preview_item.setZValue(len(self.items) + 1)
            finally:
                self._updating = False
            self.size_label.setText(trans.t('mosaic_size').format(
                width=captured.size[0], height=captured.size[1]))
            self.status.setText(trans.t('mosaic_preview_ready').format(
                width=captured.size[0], height=captured.size[1]))
            self._fit()

        self._run(preview, trans.t('mosaic_preview'), complete)

    def _edit_control_points(self):
        if self.worker is not None or len(self.frames) < 2 or self._cp_dialog is not None:
            return
        try:
            cameras = (self.settings.camera_specs()
                       if self.settings.projection.currentData() is not None else ())
        except (ValueError, TypeError) as error:
            self._error(str(error))
            return
        dialog = MosaicControlPointsDialog(tuple(self.frames),
            tuple(matrix.copy() for matrix in self._current_transforms()), self.control_points,
            self, cameras=cameras)
        self._cp_dialog = dialog
        dialog.set_result(self._current_transforms(), self.qa, cameras=cameras)
        dialog.pointsChanged.connect(self._points_changed)
        dialog.selectionChanged.connect(self._point_selection_changed)
        dialog.optimizeRequested.connect(self._align)
        dialog.cancelRequested.connect(self._cancel)
        try:
            dialog.exec()
        finally:
            self._cp_dialog = None
            self._selected_points = ()
            dialog.deleteLater()
        self._overlay_changed()

    def _points_changed(self, points):
        self.control_points = tuple(points)
        self.qa = None
        self._selected_points = ()
        self._invalidate_plan()

    def _point_selection_changed(self, indices):
        self._selected_points = tuple(int(index) for index in indices)
        self._overlay_changed()

    def _edit_mask(self):
        index = self.list.currentRow()
        if self.worker is not None or not 0 <= index < len(self.frames):
            return
        frame = self.frames[index]
        dialog = MosaicMaskDialog(frame.preview, frame.image.shape[:2],
                                  _mask_preview(frame.mask, frame.preview.shape[:2]), self)
        self._mask_dialog = dialog
        self._mask_imported = None
        dialog.maskImportRequested.connect(lambda path: self._import_mask(path, frame, dialog))
        dialog.cancelRequested.connect(self._cancel)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        imported = self._mask_imported
        strokes, clear_base = tuple(dialog.strokes), bool(dialog.clear_base)
        self._mask_dialog = None
        self._mask_imported = None
        if self.worker is not None:
            self.worker.finished.connect(dialog.deleteLater)
        else:
            dialog.deleteLater()
        if not accepted or (imported is None and not strokes and not clear_base):
            return
        base = None if clear_base else imported if imported is not None else frame.mask

        def apply(cancel, progress):
            return _paint_mask(base, frame.image.shape[:2], strokes, cancel, progress)

        def complete(mask):
            self.transforms = self._current_transforms()
            self.frames[index] = replace(frame, mask=mask)
            self._invalidate_plan()
            self._rebuild()
            self.list.setCurrentRow(index)

        self._run(apply, trans.t('mosaic_mask'), complete)

    def _import_mask(self, path, frame, dialog):
        if self.worker is not None:
            dialog.set_import_failed()
            return
        shape, preview_shape = frame.image.shape[:2], frame.preview.shape[:2]
        pending = []

        def load(cancel, progress):
            pages = _decode_pages(path, cancel)
            if len(pages) != 1:
                raise ValueError('A source mask must contain exactly one image')
            image = pages[0]
            if image.shape[:2] != shape:
                raise ValueError('Mask dimensions must match the native source dimensions')
            from core.frame_storage import create_frame
            mask = create_frame(shape, image.dtype)
            for y in range(0, shape[0], 128):
                _check_cancel(cancel)
                strip = image[y:y + 128]
                if strip.ndim == 3:
                    if strip.shape[2] == 4:
                        strip = strip[:, :, 3]
                    elif strip.shape[2] == 1:
                        strip = strip[:, :, 0]
                    else:
                        strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
                mask[y:y + 128] = strip
                progress(min(y + 128, shape[0]), shape[0])
            mask.flush()
            return mask, _mask_preview(mask, preview_shape)

        def complete(value):
            pending.append(value)

        def settled():
            if pending:
                mask, preview = pending[0]
                self._mask_imported = mask
                if path not in self._output_source_paths:
                    self._output_source_paths.append(path)
                dialog.set_mask_preview(preview)
            else:
                dialog.set_import_failed()

        self._run(load, trans.t('mosaic_mask_import'), complete,
                  settled=settled)

    def _save(self):
        if self.worker is not None or not self.frames:
            return
        filters = trans.t('mosaic_save_filter')
        path, selected_filter = FileDialog.getSaveFileName(
            self, trans.t('mosaic_save'), image_save_dir(), filters, image_save_filter(filters))
        if not path:
            return
        if not os.path.splitext(path)[1]:
            # The dialog already adds the selected filter's suffix; this only
            # covers a filter without one.
            suffix = filter_suffix(selected_filter)
            path += '.' + suffix if suffix else '.tif'
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.tif', '.tiff', '.png'):
            QMessageBox.warning(self, trans.t('mosaic_title'), trans.t('mosaic_save_type'))
            return
        protected = tuple(frame.source_path for frame in self.frames if frame.source_path)
        protected += tuple(self._output_source_paths)
        if any(same_path(source, path) for source in protected):
            QMessageBox.warning(self, trans.t('mosaic_title'), trans.t('mosaic_source_conflict'))
            return
        try:
            frames, transforms, spec, revision = self._snapshot()
        except (ValueError, TypeError) as error:
            self._error(str(error))
            return
        plan = self._plan if self._plan_revision == revision else None
        alignment = self._alignment
        write_error = trans.t('msg_save_failed_info_write')
        remember_image_save(path, selected_filter)

        def export(cancel, progress):
            captured = plan if plan is not None else plan_mosaic(frames, transforms,
                spec=spec, alignment=alignment, should_cancel=cancel, progress=progress)
            image, _origin = render_mosaic(frames, transforms, plan=captured,
                should_cancel=cancel, progress=progress)
            _check_cancel(cancel)
            if not imwrite_auto(path, image, exif=captured.exif,
                icc_profile=captured.icc_profile, should_cancel=cancel,
                stream_tiff=ext in ('.tif', '.tiff')):
                raise OSError(write_error)
            return captured

        def complete(captured):
            if revision == self._revision:
                self._plan, self._plan_revision = captured, revision
            self.status.setText(trans.t('depth_saved').format(path=path))

        self._run(export, trans.t('mosaic_exporting'), complete)

    def _retranslate(self):
        self.setWindowTitle(trans.t('mosaic_title'))
        for index, key in enumerate(('mosaic_assistant', 'mosaic_sources', 'mosaic_editor')):
            self.tabs.setTabText(index, trans.t(key))
        for button, key in self._assistant_buttons:
            button.setText(trans.t(key))
        for button, key in ((self.add_files, 'mosaic_add_files'), (self.add_folder, 'action_open_folder'),
                (self.add_outputs, 'mosaic_add_outputs'), (self.remove, 'batch_btn_remove'),
                (self.grid_apply, 'mosaic_apply_grid'), (self.auto, 'mosaic_auto'),
                (self.fit, 'label_fit'), (self.cp_button, 'mosaic_control_points'),
                (self.mask_button, 'mosaic_mask'), (self.preview_button, 'mosaic_preview'),
                (self.save, 'mosaic_save'), (self.cancel, 'btn_cancel'), (self.close_button, 'btn_close')):
            button.setText(trans.t(key))
        self._assistant_intro.setText(trans.t('mosaic_intro'))
        self.opacity.setToolTip(trans.t('mosaic_preview_hint'))
        for label, key in self._source_labels:
            label.setText(trans.t(key))
        for combo, keys in ((self.grid_order, ('mosaic_by_row', 'mosaic_by_column')),
                (self.grid_direction, ('mosaic_unidirectional', 'mosaic_zigzag')),
                (self.grid_corner, ('mosaic_top_left', 'mosaic_top_right',
                                    'mosaic_bottom_left', 'mosaic_bottom_right'))):
            for index, key in enumerate(keys):
                combo.setItemText(index, trans.t(key))
        manual_hint = '\n\n'.join((trans.t('mosaic_intro'), trans.t('mosaic_manual')))
        self.canvas.setToolTip(manual_hint)
        self.x_spin.setToolTip(manual_hint)
        self.y_spin.setToolTip(manual_hint)
        for index, frame in enumerate(self.frames):
            label = frame.name
            if frame.mask is not None:
                label += '  ' + trans.t('mosaic_mask_present')
            if index and not self.connected[index]:
                label += '  ' + trans.t('mosaic_unmatched_mark')
            self.list.item(index).setText(label)
        self._refresh_controls()

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
