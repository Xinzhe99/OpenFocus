"""Review sanitized diagnostics before explicit transmission."""
import io
import json
import threading
import zipfile

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPlainTextEdit,
                             QPushButton, QVBoxLayout)

from locales import trans
from utils.app_paths import user_data_dir
from utils.report_bundle import build_report_bundle
from utils.report_client import ReportClient
from utils.reporting import report_metadata, send_bundle

_ACTIVE_REPORT_JOBS = set()


class _ReportJob(QObject):
    finished = pyqtSignal(object, bool)

    def __init__(self, call):
        super().__init__()
        self.call = call

    def start(self):
        _ACTIVE_REPORT_JOBS.add(self)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            self.finished.emit(self.call(), True)
        except Exception:
            self.finished.emit(None, False)
        finally:
            _ACTIVE_REPORT_JOBS.discard(self)


class ReportReviewDialog(QDialog):
    def __init__(self, window_main, parent=None, *, crash_text=None, log_text=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self.sent_or_queued = False
        self.locale = trans.current_lang
        self.severity = 'crash' if crash_text else 'error'
        self.bundle = None
        self._job = None
        self._operation = None
        self.setWindowTitle(trans.t('report_review'))
        self.resize(720, 560)
        layout = QVBoxLayout(self)
        scope = QLabel(trans.t('report_scope'))
        scope.setWordWrap(True)
        layout.addWidget(scope)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        layout.addWidget(self.preview, 1)
        self.status = QLabel(trans.t('report_prepare'))
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.send_button = QPushButton(trans.t('report_send'))
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self._send)
        buttons.addWidget(self.send_button)
        buttons.addStretch()
        close = QPushButton(trans.t('btn_close'))
        close.clicked.connect(self.reject)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        def prepare():
            bundle = build_report_bundle(crash_text=crash_text, log_text=log_text)
            metadata = report_metadata(bundle, source='manual', locale=self.locale,
                                       severity=self.severity)
            metadata['installation_id'] = ReportClient(user_data_dir() / 'reports').installation_id
            sections = [json.dumps(metadata, indent=2, ensure_ascii=False)]
            with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
                for name in archive.namelist():
                    sections.append(name + '\n' + json.dumps(
                        json.loads(archive.read(name)), indent=2, ensure_ascii=False))
            return bundle, '\n\n'.join(sections)
        self._start('prepare', prepare)

    def _start(self, operation, call):
        self._operation = operation
        job = _ReportJob(call)
        self._job = job
        job.finished.connect(self._finished)
        job.start()

    def _send(self):
        if self._job is not None or self.bundle is None:
            return
        self.send_button.setEnabled(False)
        self.status.setText(trans.t('report_sending'))
        bundle = self.bundle
        account = getattr(self.window_main, '_account_client', None)
        locale = self.locale
        self._start('send', lambda: send_bundle(bundle, source='manual', locale=locale,
                                                account=account, severity=self.severity))

    def _finished(self, result, ok):
        operation, self._operation = self._operation, None
        self._job = None
        if not ok:
            self.status.setText(trans.t('report_failed'))
            self.send_button.setEnabled(self.bundle is not None)
            return
        if operation == 'prepare':
            self.bundle, text = result
            self.preview.setPlainText(text)
            self.status.clear()
            self.send_button.setEnabled(True)
        else:
            self.status.setText(trans.t('report_' + result))
            self.sent_or_queued = result in ('sent', 'queued')
            self.send_button.setEnabled(result == 'dropped')
            if result == 'queued':
                from ui.report_retry import schedule_report_retries
                schedule_report_retries(self.window_main)
