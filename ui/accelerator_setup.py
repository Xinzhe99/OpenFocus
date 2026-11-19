"""Offer a hardware-matched inference pack without blocking desktop startup."""
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QProgressBar,
                             QPushButton, QVBoxLayout, QWidget)

from locales import trans
from utils.settings_store import get_settings


class _PackJob(QThread):
    progress = pyqtSignal(int, str)

    def __init__(self, install, parent):
        super().__init__(parent)
        self.install = install
        self.status = None
        self.error = ''
        self.cancelled = False

    def run(self):
        try:
            from core import inference_pack
            if not self.install:
                self.status = inference_pack.pack_status()
                return
            last = None

            def progress(done, total, stage):
                nonlocal last
                current = (min(100, int(100 * done / total)) if total else -1, stage)
                if current != last:
                    self.progress.emit(*current)
                    last = current

            try:
                self.status = inference_pack.install_pack(
                    progress=progress, cancelled=self.isInterruptionRequested)
            except inference_pack.PackCancelled:
                self.cancelled = True
                return
            from fusion_methods.neural_runtime import refresh_accelerator_pack
            refresh_accelerator_pack()
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__


def _plain_label(parent):
    label = QLabel(parent)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


class _PackDialog(QDialog):
    def __init__(self, banner):
        super().__init__(banner.window())
        self.banner = banner
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        self.description = _plain_label(self)
        self.feedback = _plain_label(self)
        self.feedback.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.progress = QProgressBar(self)
        layout.addWidget(self.description)
        layout.addWidget(self.feedback)
        layout.addWidget(self.progress)
        row = QHBoxLayout()
        row.addStretch(1)
        self.close_button = QPushButton(self)
        self.close_button.clicked.connect(self.reject)
        self.install_button = QPushButton(self)
        self.install_button.clicked.connect(banner.request_install)
        row.addWidget(self.close_button)
        row.addWidget(self.install_button)
        layout.addLayout(row)

    def reject(self):
        if self.banner.installing():
            self.banner.job.requestInterruption()
            self.close_button.setEnabled(False)
            return
        super().reject()


class AcceleratorBanner(QWidget):
    installed = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.status = None
        self.job = None
        self.dialog = None
        self.error = ''
        self.cancelled = False
        self.offer = False
        self._progress = (-1, '')
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 2, 8, 2)
        self.label = _plain_label(self)
        self.configure = QPushButton(self)
        self.configure.clicked.connect(self.open_settings)
        self.dismiss = QPushButton(self)
        self.dismiss.clicked.connect(self.hide)
        row.addWidget(self.label, 1)
        row.addWidget(self.configure)
        row.addWidget(self.dismiss)
        self.hide()

    def check_at_startup(self):
        if not self.window().isVisible() or self.dialog is not None:
            return
        self.offer = True
        self._start(False)

    def open_settings(self):
        self.offer = False
        self.hide()
        if self.dialog is None:
            self.dialog = _PackDialog(self)
        if self.job is None:
            self._start(False)
        self.retranslate()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def installing(self):
        return self.job is not None and self.job.install

    def request_install(self):
        if self.job is None and self.status and self.status.supported and not self.status.installed:
            self._start(True)

    def _start(self, install):
        if self.job is not None or getattr(self.window(), '_shutting_down', False):
            return
        self.error = ''
        self.cancelled = False
        self._progress = (-1, '')
        self.job = _PackJob(install, self)
        self.job.progress.connect(self._update_progress)
        self.job.finished.connect(self._finished)
        self.job.start()
        self.retranslate()

    def _update_progress(self, percent, stage):
        self._progress = (percent, stage)
        if self.dialog is not None:
            self._show_progress()

    def _show_progress(self):
        percent, stage = self._progress
        self.dialog.progress.setRange(0, 100 if percent >= 0 else 0)
        if percent >= 0:
            self.dialog.progress.setValue(percent)
        self.dialog.feedback.setText(stage)
        self.dialog.feedback.setVisible(bool(stage))

    def _finished(self):
        job, self.job = self.job, None
        self.error, self.cancelled = job.error, job.cancelled
        if job.status is not None or not job.install:
            self.status = job.status
        job.deleteLater()
        if getattr(self.window(), '_shutting_down', False):
            return
        self.retranslate()
        if job.install and self.status and self.status.installed:
            self.installed.emit()
        if self.offer and self.status and self.status.supported and not self.status.installed:
            settings = get_settings()
            key = 'ai/accelerator_offer_seen'
            if settings.value(key) != self.status.version:
                settings.setValue(key, self.status.version)
                self.show()
        self.offer = False

    def shutdown_running(self):
        if self.job is None:
            return False
        self.job.requestInterruption()
        return True

    def retranslate(self):
        self.configure.setText(trans.t('ai_pack_title'))
        self.dismiss.setText(trans.t('report_not_now'))
        status = self.status
        if status and status.supported:
            self.label.setText(trans.t('ai_pack_offer').format(device=status.device))
        dialog = self.dialog
        if dialog is None:
            return
        dialog.setWindowTitle(trans.t('ai_pack_title'))
        busy = self.installing()
        dialog.close_button.setText(trans.t('btn_cancel_download' if busy else 'btn_close'))
        dialog.close_button.setEnabled(not busy or not self.job.isInterruptionRequested())
        dialog.install_button.setText(trans.t('btn_download_install'))
        dialog.install_button.setVisible(bool(self.job is None and status and status.supported and not status.installed))
        dialog.progress.setVisible(self.job is not None)
        if self.job is not None:
            self._show_progress()
        else:
            detail = self.error or (trans.t('converter_status_cancelled') if self.cancelled else '')
            dialog.feedback.setText(detail)
            dialog.feedback.setVisible(bool(detail))
        if status is None:
            description = trans.t('ai_pack_title') if self.error else trans.t('ai_pack_check')
        elif not status.supported:
            description = trans.t('ai_pack_base')
            if status.reason == 'driver':
                description += '\n' + trans.t('ai_pack_driver')
            if status.device:
                description += '\n' + status.device
        elif status.installed:
            description = trans.t('ai_pack_ready').format(device=status.device)
        else:
            description = (trans.t('ai_pack_offer').format(device=status.device) + '\n\n'
                           + trans.t('ai_pack_details').format(size=f'{status.download_bytes / 10**9:.2f} GB',
                                                              disk=f'{status.required_free_bytes / 10**9:.1f} GB'))
        dialog.description.setText(description)
