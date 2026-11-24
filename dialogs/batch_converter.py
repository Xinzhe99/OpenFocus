"""Independent batch still-image conversion workspace."""
import os
from pathlib import Path
from collections import Counter, deque
from threading import Condition
from time import perf_counter

from core.resource_policy import heavy_job
from threadpoolctl import threadpool_limits

from PyQt6.QtCore import QEvent, QItemSelectionModel, QSignalBlocker, QThread, pyqtSignal, QSize, Qt
from PyQt6.QtGui import QIcon, QImage, QPixmap, QShortcut, QKeySequence, QPainter, QColor
from PyQt6.QtWidgets import (QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox,
                             QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from core.batch_converter import (INPUT_EXTENSIONS, OUTPUT_EXTENSIONS, OUTPUT_FORMAT_ORDER,
                                  convert_batch, output_directory, same_output_format, thumbnail_rgb)
from utils.file_dialogs import FileDialog, expand_dialog_path
from utils.settings_store import get_converter_output_format, set_converter_output_format
from utils.sorting import natural_sort_key
from utils.logging_setup import get_logger
from locales import trans


class ConverterWorker(QThread):
    item = pyqtSignal(object)
    progress = pyqtSignal(int)
    complete = pyqtSignal(object)

    def __init__(self, sources, directory, ext, *, output_mode, max_edge, scale):
        super().__init__()
        self.sources, self.directory, self.ext = sources, directory, ext
        self.output_mode, self.max_edge, self.scale = output_mode, max_edge, scale

    def run(self):
        result = convert_batch(self.sources, self.directory, self.ext,
                               self.isInterruptionRequested, self.item.emit, self.progress.emit,
                               output_mode=self.output_mode, max_edge=self.max_edge,
                               scale=self.scale, skip_same_format=True)
        self.complete.emit(result)


class ThumbnailWorker(QThread):
    ready = pyqtSignal(int, QImage, str, str)
    capacity = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition = Condition()
        self._queue = deque()

    def enqueue(self, token, path):
        with self._condition:
            if self.isInterruptionRequested() or len(self._queue) >= self.capacity:
                return False
            self._queue.append((token, path))
            self._condition.notify()
            return True

    def stop(self):
        self.requestInterruption()
        with self._condition:
            self._queue.clear()
            self._condition.notify()

    def discard(self, tokens):
        with self._condition:
            self._queue = deque(entry for entry in self._queue if entry[0] not in tokens)

    def run(self):
        while not self.isInterruptionRequested():
            with self._condition:
                while not self._queue and not self.isInterruptionRequested():
                    self._condition.wait()
                if self.isInterruptionRequested():
                    return
                token, path = self._queue.popleft()
            image, error, tiff_ext = QImage(), '', ''
            started = perf_counter()
            try:
                with heavy_job(self.isInterruptionRequested), threadpool_limits(limits=1):
                    if Path(path).suffix.lower() in ('.tif', '.tiff'):
                        tiff_ext = '.dng' if same_output_format(path, '.dng') else '.tif'
                    pixels = thumbnail_rgb(path)
                height, width = pixels.shape[:2]
                image = QImage(pixels.data, width, height, pixels.strides[0],
                               QImage.Format.Format_RGB888).copy()
            except Exception as exc:
                error = str(exc)
                get_logger(__name__).debug('Thumbnail unavailable for %s: %s', path, error)
            get_logger(__name__).debug('Thumbnail %s elapsed_ms=%.3f cancelled=%s',
                                       path, (perf_counter() - started) * 1000,
                                       self.isInterruptionRequested())
            if not self.isInterruptionRequested():
                self.ready.emit(token, image, error, tiff_ext)


class BatchConverterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('converter_title'))
        self.resize(760, 580)
        self.setAcceptDrops(True)
        self._busy = False
        self.worker = None
        self.sources = []
        self._format_counts = Counter()
        self._close_pending = False
        self._conversion_results = None
        self._terminal_rows = set()
        self.thumbnail_worker = None
        self._thumbnail_items = {}
        # Rows before the cursor have already been submitted to the worker.
        self._thumbnail_cursor = 0
        self._next_token = 0
        self._thumbnail_placeholder = self._thumbnail_icon(QImage())
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        buttons = QHBoxLayout()
        self.add_files = QPushButton(trans.t('converter_add_files'))
        self.add_folder = QPushButton(trans.t('batch_btn_add'))
        for button in (self.add_files, self.add_folder):
            buttons.addWidget(button)
        buttons.addStretch()
        self.queue_count = QLabel()
        buttons.addWidget(self.queue_count)
        layout.addLayout(buttons)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setIconSize(QSize(96, 72))
        self.list.setUniformItemSizes(True)
        self.list.setAcceptDrops(True)
        self.list.viewport().setAcceptDrops(True)
        self.list.viewport().installEventFilter(self)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._queue_menu)
        self.list.itemDoubleClicked.connect(self._show_result_details)
        self.empty_hint = QLabel(trans.t('converter_drop_hint'), self.list.viewport())
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint.setWordWrap(True)
        self.empty_hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self.list)
        self.backspace_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Backspace), self.list)
        for shortcut in (self.delete_shortcut, self.backspace_shortcut):
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._remove)
        self.list.setAccessibleName(trans.t('converter_title'))
        self.list.setToolTip(trans.t('converter_remove_hint'))
        layout.addWidget(self.list, 1)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.addLayout(form)
        self.output_mode = QComboBox()
        for mode, key in (('sibling', 'converter_out_sibling'), ('subfolder', 'batch_out_subfolder'),
                          ('same', 'batch_out_same'), ('custom', 'batch_out_custom')):
            self.output_mode.addItem(trans.t(key), mode)
        form.addRow(trans.t('batch_output_location'), self.output_mode)
        self._directory_values = {'sibling': '_converted', 'subfolder': 'OpenFocus_Output',
                                  'same': '', 'custom': ''}
        self._directory_mode = 'sibling'
        self.directory_label = QLabel()
        self.directory_row = QWidget()
        row = QHBoxLayout(self.directory_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.directory = QLineEdit('_converted')
        self.directory_label.setBuddy(self.directory)
        self.browse = QPushButton(trans.t('batch_btn_browse'))
        row.addWidget(self.directory, 1)
        row.addWidget(self.browse)
        form.addRow(self.directory_label, self.directory_row)
        self.format = QComboBox()
        self.format.addItems(OUTPUT_FORMAT_ORDER)
        self.format.setCurrentText(get_converter_output_format())
        self.format.setAccessibleName(trans.t('batch_output_format'))
        self.format.setToolTip(trans.t('converter_output_scope') + '\n' + trans.t('converter_warning'))
        format_row = QHBoxLayout()
        format_row.addWidget(self.format)
        size_label = QLabel(trans.t('mosaic_output_scale'))
        format_row.addWidget(size_label)
        self.output_size = QComboBox()
        self.output_size.addItem(trans.t('converter_original'), 'original')
        for k in (2, 4, 8):
            self.output_size.addItem(trans.t('converter_long_edge').format(k=k, pixels=k * 1024), k * 1024)
        self.output_size.addItem(trans.t('converter_percentage'), 'percent')
        self._raster_size_index = None
        size_label.setBuddy(self.output_size)
        format_row.addWidget(self.output_size)
        self.percent = QSpinBox()
        self.percent.setRange(1, 100)
        self.percent.setValue(50)
        self.percent.setSuffix(' %')
        self.percent.setAccessibleName(trans.t('converter_percentage'))
        format_row.addWidget(self.percent)
        format_row.addStretch()
        format_label = QLabel(trans.t('batch_output_format'))
        format_label.setBuddy(self.format)
        form.addRow(format_label, format_row)
        self.format_example = QLineEdit()
        self.format_example.setReadOnly(True)
        self.format_example.setFrame(False)
        form.addRow(trans.t('converter_output_example'), self.format_example)
        self.format_warning = QLabel()
        self.format_warning.setWordWrap(True)
        self.format_warning.setObjectName('mutedLabel')
        layout.addWidget(self.format_warning)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.summary = QLabel('')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        row = QHBoxLayout()
        self.start = QPushButton(trans.t('converter_start'))
        self.cancel = QPushButton(trans.t('btn_cancel'))
        self.cancel.setEnabled(False)
        close = QPushButton(trans.t('btn_close'))
        row.addWidget(self.start)
        row.addWidget(self.cancel)
        row.addWidget(close)
        layout.addLayout(row)
        self.controls = (self.add_files, self.add_folder, self.output_mode, self.directory,
                         self.browse, self.format, self.output_size, self.percent, self.start)
        self.add_files.clicked.connect(self._add_files)
        self.add_folder.clicked.connect(self._add_folder)
        self.browse.clicked.connect(self._browse)
        self.start.clicked.connect(self._start)
        self.cancel.clicked.connect(self._cancel)
        close.clicked.connect(self.close)
        self.list.itemSelectionChanged.connect(self._update_controls)
        self.output_mode.currentIndexChanged.connect(self._change_output_mode)
        self.output_size.currentIndexChanged.connect(self._update_controls)
        self.percent.valueChanged.connect(self._update_controls)
        self.directory.textChanged.connect(self._update_controls)
        self.directory.returnPressed.connect(self._start_if_ready)
        self.format.currentTextChanged.connect(self._format_changed)
        for button in (self.add_files, self.add_folder, self.browse, self.start, self.cancel, close):
            button.setAutoDefault(False)
        self._sync_size_for_format()
        self._change_output_mode()

    def _format_changed(self, image_format):
        self._sync_size_for_format()
        self._update_controls()
        set_converter_output_format(image_format)

    def _sync_size_for_format(self):
        if self.format.currentText() == 'DNG':
            self._raster_size_index = self.output_size.currentIndex()
            index = 0
        elif self._raster_size_index is not None:
            index, self._raster_size_index = self._raster_size_index, None
        else:
            return
        self.output_size.blockSignals(True)
        self.output_size.setCurrentIndex(index)
        self.output_size.blockSignals(False)

    def _change_output_mode(self):
        mode = self.output_mode.currentData()
        self._directory_values[self._directory_mode] = self.directory.text()
        self._directory_mode = mode
        self.directory.setText(self._directory_values[mode])
        self.directory.setPlaceholderText({'sibling': '_converted', 'subfolder': 'OpenFocus_Output'}.get(mode, ''))
        key = {'sibling': 'converter_suffix', 'subfolder': 'batch_subfolder_name',
               'same': 'batch_output_location', 'custom': 'batch_output_location'}[mode]
        self.directory_label.setText(trans.t(key))
        self.directory.setAccessibleName(trans.t(key))
        self.directory_label.setVisible(mode != 'same')
        self.directory_row.setVisible(mode != 'same')
        self.browse.setVisible(mode == 'custom')
        self._update_controls()

    def _update_format_warning(self, image_format, notice):
        keys = {'DNG': 'converter_dng_risk', 'JPEG': 'converter_jpeg_risk',
                'BMP': 'converter_8bit_risk', 'WebP': 'converter_8bit_risk',
                'TIFF': 'converter_transparency_risk'}
        key = keys.get(image_format)
        text = trans.t(key).format(format=image_format) if key else ''
        if image_format == 'DNG':
            text += ' ' + trans.t('converter_dng_original_size')
        self.format_warning.setText('\n'.join(filter(None, (notice, text))))
        self.format_warning.setVisible(bool(notice or text))

    def _directory_value(self):
        text = self.directory.text().strip()
        if self.output_mode.currentData() == 'custom' and text:
            return expand_dialog_path(text, os.getcwd())
        return text

    def _resize_options(self):
        size = self.output_size.currentData()
        return (size if isinstance(size, int) else None,
                self.percent.value() / 100 if size == 'percent' else 1.0)

    def _update_controls(self):
        editable = not self._busy and not self._close_pending
        for control in self.controls:
            control.setEnabled(editable)
        selected = self.list.selectedItems()
        self.delete_shortcut.setEnabled(editable and bool(selected))
        self.backspace_shortcut.setEnabled(editable and bool(selected))
        image_format = self.format.currentText()
        dng = image_format == 'DNG'
        self.output_size.setEnabled(editable and not dng)
        self.percent.setVisible(self.output_size.currentData() == 'percent')
        mode = self.output_mode.currentData()
        self.directory.setEnabled(editable and mode != 'same')
        self.browse.setEnabled(editable and mode == 'custom')
        source = self.sources[0] if self.sources else None
        if selected:
            row = self.list.row(selected[0])
            if 0 <= row < len(self.sources):
                source = self.sources[row]
        ext = OUTPUT_EXTENSIONS[image_format]
        preview = notice = ''
        if source:
            try:
                directory = output_directory(source, mode, self._directory_value())
                preview = os.path.join(directory, Path(source).stem + ext)
            except (ValueError, OSError) as exc:
                notice = str(exc)
        self.format_example.setText(preview)
        self.format_example.setToolTip(preview)
        self.directory.setToolTip(notice)
        matching = self._format_counts[ext]
        unchanged = self._resize_options() == (None, 1.0)
        all_unchanged = bool(matching) and matching == len(self.sources) and unchanged
        if not notice and matching:
            notice = trans.t('converter_no_changes' if all_unchanged else 'converter_skip_unchanged')
        self._update_format_warning(image_format, notice)
        self.start.setEnabled(editable and bool(preview) and not all_unchanged)
        self.queue_count.setText(trans.t('converter_queue_count').format(count=len(self.sources)))
        self.empty_hint.setVisible(not self.sources)
        self.empty_hint.setGeometry(self.list.viewport().rect())

    def _start_if_ready(self):
        if self.start.isEnabled():
            self._start()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._start_if_ready()
            event.accept()
            return
        super().keyPressEvent(event)

    @staticmethod
    def _drop_paths(event):
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return []
        return [path for url in mime.urls()
                if url.isLocalFile() and (path := url.toLocalFile())]

    def dragEnterEvent(self, event):
        if not self._busy and not self._close_pending and self._drop_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        paths = self._drop_paths(event)
        if self._busy or self._close_pending or not paths:
            event.ignore()
            return
        self._append(paths, skipped=len(event.mimeData().urls()) - len(paths))
        event.acceptProposedAction()

    def eventFilter(self, watched, event):
        if watched is self.list.viewport():
            if event.type() in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                self.dragEnterEvent(event)
                return True
            if event.type() == QEvent.Type.Drop:
                self.dropEvent(event)
                return True
            if event.type() == QEvent.Type.Resize:
                self.empty_hint.setGeometry(self.list.viewport().rect())
            if (event.type() == QEvent.Type.MouseButtonPress and
                    event.button() == Qt.MouseButton.RightButton):
                # Keep a multi-selection intact until the context menu opens.
                return True
        return super().eventFilter(watched, event)

    def _queue_menu(self, pos):
        if self._busy or self._close_pending:
            return
        item = self.list.itemAt(pos)
        if item is None:
            return
        if not item.isSelected():
            self.list.setCurrentItem(item, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        menu = QMenu(self.list)
        remove = menu.addAction(trans.t('converter_remove_queue'))
        remove.triggered.connect(self._remove)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        menu.popup(self.list.viewport().mapToGlobal(pos))

    def _append(self, paths, skipped=0):
        if self._busy or self._close_pending:
            return
        paths = list(paths)
        if not paths and not skipped:
            return
        candidates = []
        for path in paths:
            path = Path(path)
            try:
                if path.is_dir():
                    # Match Add Folder: import the images directly inside each folder.
                    children = list(path.iterdir())
                    candidates.extend(children)
                    if not children:
                        skipped += 1
                else:
                    candidates.append(path)
            except OSError:
                skipped += 1
        images = [path for path in candidates if path.suffix.lower() in INPUT_EXTENSIONS]
        skipped += len(candidates) - len(images)
        images.sort(key=natural_sort_key)
        seen = {os.path.normcase(os.path.realpath(p)) for p in self.sources}
        added = 0
        for path in images:
            key = os.path.normcase(os.path.realpath(path))
            if key in seen or not path.is_file():
                skipped += 1
                continue
            seen.add(key)
            source = os.path.abspath(path)
            self.sources.append(source)
            item = QListWidgetItem(source)
            item.setSizeHint(QSize(240, 88))
            item.setIcon(self._thumbnail_placeholder)
            item.setToolTip(source)
            token = self._next_token
            self._next_token += 1
            item.setData(Qt.ItemDataRole.UserRole, token)
            # TIFF containers can hold DNG data. Inspect their headers in the
            # bounded preview worker instead of opening files on the GUI thread.
            if path.suffix.lower() not in ('.tif', '.tiff'):
                ext = next((e for e in OUTPUT_EXTENSIONS.values() if same_output_format(source, e)), None)
                if ext:
                    item.setData(Qt.ItemDataRole.UserRole + 3, ext)
                    self._format_counts[ext] += 1
            self._thumbnail_items[token] = item
            self.list.addItem(item)
            added += 1
        self.summary.setText(trans.t('converter_import_status').format(
            added=added, skipped=skipped, total=len(self.sources)))
        self._update_controls()
        self._pump_thumbnails()

    def _add_files(self):
        if self._busy or self._close_pending:
            return
        paths, _ = FileDialog.getOpenFileNames(self, trans.t('converter_add_files'))
        self._append(paths)

    def _add_folder(self):
        if self._busy or self._close_pending:
            return
        folder = FileDialog.getExistingDirectory(self, trans.t('batch_btn_add'))
        if folder:
            self._append([folder])

    def _remove(self):
        if self._busy or self._close_pending:
            return
        rows = sorted((index.row() for index in self.list.selectionModel().selectedRows()), reverse=True)
        tokens = set()
        with QSignalBlocker(self.list):
            for row in rows:
                item = self.list.takeItem(row)
                self.sources.pop(row)
                token = item.data(Qt.ItemDataRole.UserRole)
                tokens.add(token)
                ext = item.data(Qt.ItemDataRole.UserRole + 3)
                if ext:
                    self._format_counts[ext] -= 1
                self._thumbnail_items.pop(token, None)
                if row < self._thumbnail_cursor:
                    self._thumbnail_cursor -= 1
        if self.thumbnail_worker:
            self.thumbnail_worker.discard(tokens)
        if tokens:
            self.summary.clear()
        self._pump_thumbnails()
        self._update_controls()

    def _browse(self):
        if self._busy or self._close_pending:
            return
        if self.output_mode.currentData() != 'custom':
            return
        path = FileDialog.getExistingDirectory(self, trans.t('batch_output_location'))
        if path:
            self.directory.setText(path)

    def _start(self):
        if not self.start.isEnabled():
            return
        self._conversion_results = None
        self._terminal_rows.clear()
        for index, path in enumerate(self.sources):
            self.list.item(index).setText(path)
            self.list.item(index).setToolTip('')
        self._busy = True
        self._update_controls()
        self.cancel.setEnabled(True)
        self.progress.setRange(0, len(self.sources))
        self.progress.setValue(0)
        self.summary.setText('')
        max_edge, scale = self._resize_options()
        self.worker = ConverterWorker(tuple(self.sources), self._directory_value(),
                                      OUTPUT_EXTENSIONS[self.format.currentText()],
                                      output_mode=self.output_mode.currentData(),
                                      max_edge=max_edge, scale=scale)
        self.worker.item.connect(self._item)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.complete.connect(self._complete)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _item(self, event):
        index = event['index']
        if index in self._terminal_rows:
            return
        row = self.list.item(index)
        if row is None:
            return
        status = event['status']
        if status == 'running':
            text = trans.t('converter_item_state').format(
                completed=event['completed'], pages=event['pages'], failed=event['failed'])
            if event['error']:
                text += ' — ' + event['error']
        else:
            self._terminal_rows.add(index)
            text = trans.t('converter_status_' + status)
            if event['error']:
                text += ' — ' + event['error']
        row.setText(Path(self.sources[index]).name + '\n' + text)
        row.setToolTip(self.sources[index])

    def _show_result_details(self, row):
        if self._conversion_results is None:
            return
        index = row.data(Qt.ItemDataRole.UserRole + 2)
        if index is None:
            return
        record = self._conversion_results['items'][index]
        details = [record['source'], trans.t('converter_status_' + record['status'])]
        details.extend(record['outputs'])
        details.extend(record['errors'])
        QMessageBox.information(self, trans.t('converter_title'), '\n'.join(details))

    def _cancel(self):
        if self.worker:
            self.worker.requestInterruption()
            self.cancel.setEnabled(False)
            self.summary.setText(trans.t('depth_cancelling'))

    def _complete(self, result):
        self._conversion_results = result
        for index in range(len(result['items'])):
            self.list.item(index).setData(Qt.ItemDataRole.UserRole + 2, index)
        counts = result['counts']
        summary = ' · '.join(f"{trans.t('converter_status_' + status)}: {counts[status]}"
                             for status in ('success', 'skipped', 'failed', 'partial', 'cancelled', 'not_started'))
        if result['scheduler_error']:
            summary += '\n' + trans.t('converter_scheduler_error').format(error=result['scheduler_error'])
        self.summary.setText(summary)

    def _finished(self):
        self._busy = False
        self._update_controls()
        self.cancel.setEnabled(False)
        if self._close_pending:
            self.close()

    @staticmethod
    def _thumbnail_icon(image):
        canvas = QPixmap(96, 72)
        canvas.fill(QColor('#30343b'))
        if not image.isNull():
            preview = QPixmap.fromImage(image).scaled(96, 72, Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation)
            painter = QPainter(canvas)
            painter.drawPixmap((96 - preview.width()) // 2, (72 - preview.height()) // 2, preview)
            painter.end()
        return QIcon(canvas)

    def _pump_thumbnails(self):
        if self._close_pending or not self.sources:
            return
        if self.thumbnail_worker is None:
            self.thumbnail_worker = ThumbnailWorker(self)
            self.thumbnail_worker.ready.connect(self._thumbnail_ready)
            self.thumbnail_worker.finished.connect(self._thumbnail_finished)
            self.thumbnail_worker.start()
        while self._thumbnail_cursor < self.list.count():
            item = self.list.item(self._thumbnail_cursor)
            token = item.data(Qt.ItemDataRole.UserRole)
            if not self.thumbnail_worker.enqueue(token, self.sources[self._thumbnail_cursor]):
                break
            self._thumbnail_cursor += 1

    def _thumbnail_ready(self, token, image, error, tiff_ext):
        item = self._thumbnail_items.get(token)
        if item is not None:
            item.setIcon(self._thumbnail_placeholder if image.isNull() else self._thumbnail_icon(image))
            if error:
                item.setToolTip(item.toolTip() + '\n' + trans.t('converter_no_preview'))
            if tiff_ext:
                item.setData(Qt.ItemDataRole.UserRole + 3, tiff_ext)
                self._format_counts[tiff_ext] += 1
                self._update_controls()
        self._pump_thumbnails()

    def _thumbnail_finished(self):
        if self._close_pending and not (self.worker and self.worker.isRunning()):
            self.close()

    def _request_close(self):
        self._close_pending = True
        self._update_controls()
        if self.worker and self.worker.isRunning():
            self._cancel()
        if self.thumbnail_worker and self.thumbnail_worker.isRunning():
            self.thumbnail_worker.stop()
        return ((self.worker and self.worker.isRunning()) or
                (self.thumbnail_worker and self.thumbnail_worker.isRunning()))

    def reject(self):
        if not self._request_close():
            super().reject()

    def closeEvent(self, event):
        if self._request_close():
            event.ignore()
        else:
            super().closeEvent(event)
