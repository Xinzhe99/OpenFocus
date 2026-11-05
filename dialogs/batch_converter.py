"""Independent batch still-image conversion workspace."""
import os
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QMessageBox, QProgressBar,
                             QPushButton, QScrollArea, QVBoxLayout, QWidget)

from core.batch_converter import INPUT_EXTENSIONS, OUTPUT_EXTENSIONS, convert_one, page_count
from locales import trans


class ConverterWorker(QThread):
    item = pyqtSignal(int, str)
    progress = pyqtSignal(int)
    complete = pyqtSignal(int, int, bool)

    def __init__(self, sources, directory, ext):
        super().__init__()
        self.sources, self.directory, self.ext = sources, directory, ext

    def run(self):
        success = failed = 0
        for index, source in enumerate(self.sources):
            if self.isInterruptionRequested():
                break
            outputs = []
            errors = []
            try:
                count = page_count(source)
                for page in range(count):
                    if self.isInterruptionRequested():
                        break
                    try:
                        outputs.append(convert_one(source, self.directory, self.ext, page, count,
                                                   self.isInterruptionRequested))
                    except InterruptedError:
                        break
                    except Exception as exc:
                        errors.append(f'{page + 1}: {exc}')
            except Exception as exc:
                errors.append(str(exc))
            if errors:
                failed += 1
            elif not self.isInterruptionRequested():
                success += 1
            status = '\n'.join(outputs + errors)
            self.item.emit(index, status or trans.t('compare_cancelled'))
            self.progress.emit(index + 1)
        self.complete.emit(success, failed, self.isInterruptionRequested())


class BatchConverterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('converter_title'))
        self.resize(760, 580)
        self.worker = None
        self.sources = []
        self._close_pending = False
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
                self.list.addItem(path)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, trans.t('converter_add_files'))
        self._append(paths)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, trans.t('batch_btn_add'))
        if folder:
            self._append([str(p) for p in sorted(Path(folder).iterdir()) if p.is_file()])

    def _remove(self):
        for item in sorted(self.list.selectedItems(), key=self.list.row, reverse=True):
            row = self.list.row(item)
            self.list.takeItem(row)
            self.sources.pop(row)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, trans.t('batch_output_location'))
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
        self.list.item(index).setText(self.sources[index] + '\n' + status)
        self.list.item(index).setToolTip(status)

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
        if self._close_pending:
            self.close()

    def reject(self):
        if self.worker and self.worker.isRunning():
            self._close_pending = True
            self._cancel()
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self._close_pending = True
            self._cancel()
            event.ignore()
        else:
            super().closeEvent(event)
