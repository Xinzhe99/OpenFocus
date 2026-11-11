"""Responsive, bounded export of a retouch diagnostic preview and masks."""
from __future__ import annotations

import os
import tempfile

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from locales import trans


class _MissingDataExportWorker(QThread):
    progressed = pyqtSignal(int)

    def __init__(self, image, missing, targets, parent=None):
        super().__init__(parent)
        self.image = image
        self.missing = missing
        self.targets = tuple(targets)
        self.error = None
        self.warning = None
        self.status = "cancelled"

    def run(self):
        from core.background_fill import missing_data_overlay
        from core.registration import RegistrationCancelled
        from core.resource_policy import heavy_job, require_memory
        from utils.image_utils import imwrite_auto

        records = []
        preserve_recovery = False
        committed = False
        try:
            with heavy_job(self.isInterruptionRequested):
                require_memory(self.image.shape[0] * self.image.shape[1] * 96 + 1024 * 1024)
                preview = missing_data_overlay(self.image, self.missing)
                layers = (preview, *(layer for _name, layer, _colour in self.missing.layers()))
                for index, (target, pixels) in enumerate(zip(self.targets, layers)):
                    if self.isInterruptionRequested():
                        return
                    directory = os.path.dirname(os.path.abspath(target))
                    descriptor, temporary = tempfile.mkstemp(
                        prefix=".openfocus-map-", suffix=".png", dir=directory)
                    os.close(descriptor)
                    records.append(dict(temporary=temporary, target=target, backup=None,
                                        existed=False, moved=False, published=False))
                    if not imwrite_auto(temporary, pixels):
                        raise OSError(target)
                    self.progressed.emit(index + 1)

                # Reserve all recovery names and record target state before the
                # first rename, so every completed move has a recovery record.
                for record in records:
                    target = record["target"]
                    record["existed"] = os.path.lexists(target)
                    if record["existed"] and not os.path.isfile(target):
                        raise OSError(f"Unsupported export target: {target}")
                    directory = os.path.dirname(os.path.abspath(target))
                    descriptor, backup = tempfile.mkstemp(
                        prefix=".openfocus-map-backup-", dir=directory)
                    record["backup"] = backup
                    os.close(descriptor)
                if self.isInterruptionRequested():
                    return
                try:
                    for record in records:
                        if record["existed"]:
                            os.replace(record["target"], record["backup"])
                            record["moved"] = True
                    for record in records:
                        os.replace(record["temporary"], record["target"])
                        record["published"] = True
                except Exception as exc:
                    recovery_errors = []
                    for record in reversed(records):
                        try:
                            if record["published"]:
                                os.unlink(record["target"])
                            if record["moved"]:
                                os.replace(record["backup"], record["target"])
                                record["moved"] = False
                        except Exception as recovery_exc:
                            recovery_errors.append(
                                f"{record['target']} (backup {record['backup']}): {recovery_exc}")
                    if recovery_errors:
                        preserve_recovery = True
                        self.error = RuntimeError(
                            f"{exc}; recovery failed: {'; '.join(recovery_errors)}")
                        self.error.__cause__ = exc
                    else:
                        self.error = exc
                    self.status = "failed"
                    return
                committed = True
                self.status = "saved"
        except RegistrationCancelled:
            pass
        except Exception as exc:
            self.error = exc
            self.status = "failed"
        finally:
            cleanup_errors = []
            if not preserve_recovery:
                for record in records:
                    for path in (record["temporary"], record["backup"]):
                        if path is None:
                            continue
                        try:
                            if os.path.lexists(path):
                                os.unlink(path)
                        except OSError as exc:
                            cleanup_errors.append(f"{path}: {exc}")
            if cleanup_errors:
                message = "; ".join(cleanup_errors)
                if committed:
                    self.warning = message
                elif self.error is None:
                    self.error = OSError(message)
                    self.status = "failed"
                else:
                    self.error = RuntimeError(f"{self.error}; cleanup failed: {message}")
            self.image = None
            self.missing = None


class MissingDataExportDialog(QDialog):
    """Own the export worker until all native pixel I/O has finished."""

    def __init__(self, image, missing, targets, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("retouch_missing_save"))
        self.setModal(True)
        self._cancel_requested = False
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, len(targets))
        self.progress.setTextVisible(False)
        self.count = QLabel(f"0 / {len(targets)}")
        row.addWidget(self.progress, 1)
        row.addWidget(self.count)
        layout.addLayout(row)
        self.cancel_button = QPushButton(trans.t("btn_cancel"))
        self.cancel_button.setProperty("controlId", "retouch_map_export_cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        self.worker = _MissingDataExportWorker(image, missing, targets, self)
        self.worker.progressed.connect(self._progress)
        self.worker.finished.connect(self._finished)
        self.resize(400, self.sizeHint().height())
        self._start_timer = QTimer(self)
        self._start_timer.setSingleShot(True)
        self._start_timer.timeout.connect(self._start)
        self._start_timer.start(0)

    def _start(self):
        if self._cancel_requested:
            super().reject()
        else:
            self.worker.start()

    def _progress(self, done):
        self.progress.setValue(done)
        self.count.setText(f"{done} / {self.progress.maximum()}")

    def _finished(self):
        if self.worker.error is None and self.worker.status == "saved":
            self.accept()
        else:
            super().reject()

    def reject(self):
        self._cancel_requested = True
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText(trans.t("compare_stopping"))
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.reject()
            event.ignore()
        else:
            self._cancel_requested = True
            super().closeEvent(event)
