"""Independent batch still-image conversion workspace."""
import os
from pathlib import Path
from collections import deque
from threading import Condition
from time import perf_counter

from core.resource_policy import heavy_job
from threadpoolctl import threadpool_limits

from PyQt6.QtCore import QThread, pyqtSignal, QSize, Qt
from PyQt6.QtGui import QIcon, QImage, QPixmap, QShortcut, QKeySequence, QPainter, QColor
from PyQt6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
                             QPushButton, QScrollArea, QVBoxLayout, QWidget)

from core.batch_converter import INPUT_EXTENSIONS, OUTPUT_EXTENSIONS, convert_batch, thumbnail_rgb
from utils.file_dialogs import FileDialog
from utils.logging_setup import get_logger
from locales import trans


class ConverterWorker(QThread):
    item = pyqtSignal(int, str)
    progress = pyqtSignal(int)
    complete = pyqtSignal(int, int, bool)

    def __init__(self, sources, directory, ext):
        super().__init__()
        self.sources, self.directory, self.ext = sources, directory, ext

    def run(self):
        result = convert_batch(self.sources, self.directory, self.ext,
                               self.isInterruptionRequested, self.item.emit, self.progress.emit)
        self.complete.emit(*result)


class ThumbnailWorker(QThread):
    ready = pyqtSignal(int, QImage, str)
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
            image, error = QImage(), ''
            started = perf_counter()
            try:
                with heavy_job(self.isInterruptionRequested), threadpool_limits(limits=1):
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
                self.ready.emit(token, image, error)


class BatchConverterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('converter_title'))
        self.resize(760, 580)
        self.worker = None
        self.sources = []
        self._close_pending = False
        self.thumbnail_worker = None
        self._thumbnail_items = {}
        self._thumbnail_cursor = 0
        self._next_token = 0
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        warning = QLabel(trans.t('converter_warning'))
        warning.setWordWrap(True)
        layout.addWidget(warning)
        buttons = QHBoxLayout()
        self.add_files = QPushButton(trans.t('converter_add_files'))
        self.add_folder = QPushButton(trans.t('batch_btn_add'))
        self.remove = QPushButton(trans.t('batch_btn_remove'))
        for button in (self.add_files, self.add_folder, self.remove):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.setIconSize(QSize(96, 72))
        self.list.setUniformItemSizes(True)
        self.delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self.list)
        self.delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.delete_shortcut.activated.connect(self._remove)
        layout.addWidget(self.list)
        layout.addWidget(QLabel(trans.t('batch_output_location')))
        row = QHBoxLayout()
        self.directory = QLineEdit()
        self.directory.setAccessibleName(trans.t('batch_output_location'))
        self.browse = QPushButton(trans.t('batch_btn_browse'))
        row.addWidget(self.directory)
        row.addWidget(self.browse)
        layout.addLayout(row)
        layout.addWidget(QLabel(trans.t('batch_output_format')))
        self.format = QComboBox()
        self.format.addItems(OUTPUT_EXTENSIONS)
        layout.addWidget(self.format)
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
        self.controls = (self.add_files, self.add_folder, self.remove, self.directory,
                         self.browse, self.format, self.start)
        self.add_files.clicked.connect(self._add_files)
        self.add_folder.clicked.connect(self._add_folder)
        self.remove.clicked.connect(self._remove)
        self.browse.clicked.connect(self._browse)
        self.start.clicked.connect(self._start)
        self.cancel.clicked.connect(self._cancel)
        close.clicked.connect(self.close)

    def _append(self, paths):
        seen = {os.path.normcase(os.path.abspath(p)) for p in self.sources}
        for path in paths:
            key = os.path.normcase(os.path.abspath(path))
            if key not in seen and Path(path).suffix.lower() in INPUT_EXTENSIONS:
                seen.add(key)
                self.sources.append(path)
                item = QListWidgetItem(path)
                item.setSizeHint(QSize(240, 88))
                item.setIcon(self._thumbnail_icon(QImage()))
                item.setToolTip(path)
                token = self._next_token
                self._next_token += 1
                item.setData(Qt.ItemDataRole.UserRole, token)
                self._thumbnail_items[token] = item
                self.list.addItem(item)
        self._pump_thumbnails()

    def _add_files(self):
        paths, _ = FileDialog.getOpenFileNames(self, trans.t('converter_add_files'))
        self._append(paths)

    def _add_folder(self):
        folder = FileDialog.getExistingDirectory(self, trans.t('batch_btn_add'))
        if folder:
            self._append([str(p) for p in sorted(Path(folder).iterdir()) if p.is_file()])

    def _remove(self):
        if self.worker and self.worker.isRunning():
            return
        tokens = set()
        for item in sorted(self.list.selectedItems(), key=self.list.row, reverse=True):
            row = self.list.row(item)
            token = item.data(Qt.ItemDataRole.UserRole)
            tokens.add(token)
            self._thumbnail_items.pop(token, None)
            self.list.takeItem(row)
            self.sources.pop(row)
            self._thumbnail_cursor = min(self._thumbnail_cursor, row)
        if self.thumbnail_worker:
            self.thumbnail_worker.discard(tokens)
        self._pump_thumbnails()

    def _browse(self):
        path = FileDialog.getExistingDirectory(self, trans.t('batch_output_location'))
        if path:
            self.directory.setText(path)

    def _start(self):
        directory = self.directory.text().strip()
        if not self.sources or not os.path.isdir(directory):
            QMessageBox.warning(self, trans.t('converter_title'), trans.t('converter_required'))
            return
        for index, path in enumerate(self.sources):
            self.list.item(index).setText(path)
            self.list.item(index).setToolTip('')
        for control in self.controls:
            control.setEnabled(False)
        self.delete_shortcut.setEnabled(False)
        self.cancel.setEnabled(True)
        self.progress.setRange(0, len(self.sources))
        self.progress.setValue(0)
        self.summary.setText('')
        self.worker = ConverterWorker(tuple(self.sources), directory, OUTPUT_EXTENSIONS[self.format.currentText()])
        self.worker.item.connect(self._item)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.complete.connect(self._complete)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _item(self, index, status):
        item = self.list.item(index)
        if item is not None:
            lines = status.splitlines()
            item.setText(Path(self.sources[index]).name + '\n' + (lines[-1] if lines else ''))
            item.setToolTip(self.sources[index] + '\n' + status)

    def _cancel(self):
        if self.worker:
            self.worker.requestInterruption()
            self.cancel.setEnabled(False)
            self.summary.setText(trans.t('depth_cancelling'))

    def _complete(self, success, failed, cancelled):
        self.summary.setText(trans.t('converter_summary').format(success=success, failed=failed,
                             cancelled=trans.t('compare_cancelled') if cancelled else ''))

    def _finished(self):
        for control in self.controls:
            control.setEnabled(True)
        self.cancel.setEnabled(False)
        self.delete_shortcut.setEnabled(True)
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
            if item.data(Qt.ItemDataRole.UserRole + 1):
                self._thumbnail_cursor += 1
                continue
            token = item.data(Qt.ItemDataRole.UserRole)
            if not self.thumbnail_worker.enqueue(token, self.sources[self._thumbnail_cursor]):
                break
            item.setData(Qt.ItemDataRole.UserRole + 1, True)
            self._thumbnail_cursor += 1

    def _thumbnail_ready(self, token, image, error):
        item = self._thumbnail_items.get(token)
        if item is not None:
            item.setIcon(self._thumbnail_icon(image))
            if error:
                item.setToolTip(item.toolTip() + '\n' + trans.t('converter_no_preview'))
        self._pump_thumbnails()

    def _thumbnail_finished(self):
        if self._close_pending and not (self.worker and self.worker.isRunning()):
            self.close()

    def _request_close(self):
        self._close_pending = True
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
