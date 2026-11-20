"""Offer a hardware-matched inference pack without blocking desktop startup."""
from pathlib import Path

from PyQt6.QtCore import QLocale, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from locales import trans
from ui.right_panel import ElidingButton
from utils.design_tokens import SPACE
from utils.file_dialogs import FileDialog
from utils.settings_store import get_settings
from utils.ui_utils import size_dialog_to_content

_PRIORITY_OPTIONS = (('ai_pack_source_auto', None), ('ai_pack_source_mirrors', True),
                     ('ai_pack_source_official', False))


class _PackJob(QThread):
    progress = pyqtSignal(object)

    def __init__(self, install, artifact_dir, prefer_mirror, parent):
        super().__init__(parent)
        self.install = install
        self.artifact_dir = artifact_dir
        self.prefer_mirror = prefer_mirror
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

            def progress(phase, name, done, total):
                nonlocal last
                current = (phase, name, done, total)
                if current != last:
                    self.progress.emit(current)
                    last = current

            try:
                self.status = inference_pack.install_pack(
                    progress=progress, cancelled=self.isInterruptionRequested,
                    artifact_dir=self.artifact_dir, prefer_mirror=self.prefer_mirror)
            except inference_pack.PackCancelled:
                self.cancelled = True
                return
            from fusion_methods.neural_runtime import refresh_accelerator_pack
            refresh_accelerator_pack()
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__


def _plain_label(parent):
    label = QLabel(parent)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


class _PackDialog(QDialog):
    def __init__(self, banner):
        super().__init__(banner.window())
        self.banner = banner
        self.setProperty('controlId', 'ai_accelerator_dialog')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE['lg'], SPACE['lg'], SPACE['lg'], SPACE['md'])
        layout.setSpacing(SPACE['md'])

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        body = QVBoxLayout(page)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(SPACE['sm'])
        body.setAlignment(Qt.AlignmentFlag.AlignTop)

        card = QFrame(page)
        card.setObjectName('card')
        summary = QVBoxLayout(card)
        summary.setContentsMargins(SPACE['lg'], SPACE['md'], SPACE['lg'], SPACE['md'])
        summary.setSpacing(SPACE['sm'])
        self.heading = _plain_label(card)
        self.heading.setObjectName('acceleratorStatus')
        self.device = _plain_label(card)
        self.device.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.engines = _plain_label(card)
        self.description = _plain_label(card)
        self.description.setObjectName('mutedLabel')
        for label in (self.heading, self.device, self.engines, self.description):
            summary.addWidget(label)
        body.addWidget(card)

        self.feedback = _plain_label(page)
        self.error = _plain_label(page)
        self.error.setObjectName('errorLabel')
        self.error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.progress = QProgressBar(page)
        self.progress.setTextVisible(False)
        body.addWidget(self.error)
        body.addWidget(self.feedback)
        body.addWidget(self.progress)

        self.source = QComboBox(page)
        self.source.setProperty('controlId', 'ai_pack_download_source')
        self.source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        for key, prefer_mirror in _PRIORITY_OPTIONS:
            self.source.addItem(trans.t(key), prefer_mirror)
        body.addWidget(self.source)
        self.install_button = ElidingButton(parent=page)
        self.install_button.setObjectName('primaryButton')
        self.install_button.setAutoDefault(False)
        self.install_button.clicked.connect(self._download)
        self.local_button = ElidingButton(parent=page)
        self.local_button.setAutoDefault(False)
        self.local_button.clicked.connect(self._pick_archives)
        body.addWidget(self.install_button)
        body.addWidget(self.local_button)
        self._scroll.setWidget(page)
        layout.addWidget(self._scroll, 1)

        self.close_button = QPushButton(self)
        self.close_button.setObjectName('secondaryButton')
        self.close_button.setDefault(True)
        self.close_button.clicked.connect(self.reject)
        layout.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignRight)

    def fit(self):
        root = self.layout()
        margins = root.contentsMargins()
        size_dialog_to_content(self, self._scroll, preferred_width=500)
        width = self.width() - margins.left() - margins.right()
        body_height = self._scroll.widget().layout().totalHeightForWidth(max(1, width))
        footer = margins.top() + margins.bottom() + root.spacing() + self.close_button.sizeHint().height()
        self.resize(self.width(), min(self.height(), body_height + footer))

    def _download(self):
        self.banner.request_install(prefer_mirror=self.source.currentData())

    def _pick_archives(self):
        folder = FileDialog.getExistingDirectory(self, trans.t('ai_pack_archive_folder'))
        if folder:
            self.banner.request_install(Path(folder))

    def reject(self):
        if self.banner.installing():
            self.banner.job.requestInterruption()
            self.banner.retranslate()
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
        self._progress = None
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
        self.dialog.fit()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def installing(self):
        return self.job is not None and self.job.install

    def request_install(self, artifact_dir=None, *, prefer_mirror=None):
        if self.job is None and self.status and self.status.supported and not self.status.installed:
            self._start(True, artifact_dir, prefer_mirror)

    def _start(self, install, artifact_dir=None, prefer_mirror=None):
        if self.job is not None or getattr(self.window(), '_shutting_down', False):
            return
        self.error = ''
        self.cancelled = False
        self._progress = None
        self.job = _PackJob(install, artifact_dir, prefer_mirror, self)
        self.job.progress.connect(self._update_progress)
        self.job.finished.connect(self._finished)
        self.job.start()
        self.retranslate()

    def _update_progress(self, progress):
        previous, self._progress = self._progress, progress
        if self.dialog is not None and self.job is not None:
            self._show_progress()
            if self.dialog.isVisible() and (previous is None or previous[:2] != progress[:2]):
                self.dialog.fit()

    def _show_progress(self):
        bar = self.dialog.progress
        if self.job.isInterruptionRequested():
            bar.setRange(0, 0)
            self.dialog.feedback.setText(trans.t('depth_cancelling'))
            self.dialog.feedback.show()
            return
        if self._progress is None:
            bar.setRange(0, 0)
            self.dialog.feedback.hide()
            return
        phase, name, done, total = self._progress
        stage = trans.t(f'ai_pack_progress_{phase}').format(name=name)
        bar.setRange(0, 100 if total else 0)
        if total:
            percent = 100 * done // total
            bar.setValue(percent)
            locale = QLocale(trans.current_lang)
            current = locale.toString(done / 1048576, 'f', 1)
            expected = locale.toString(total / 1048576, 'f', 1)
            stage += f'\n{current} / {expected} MiB ({percent}%)'
        self.dialog.feedback.setText(stage)
        self.dialog.feedback.show()

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
        engines = ''
        if status and status.supported:
            from fusion_methods.neural_runtime import display_name
            engines = ' · '.join(display_name(engine, include_adapter=False) for engine in status.backends)
            self.label.setText(trans.t('ai_pack_offer').format(device=status.device, engines=engines))
        dialog = self.dialog
        if dialog is None:
            return
        dialog.setWindowTitle(trans.t('ai_pack_title'))
        busy = self.installing()
        checking = self.job is not None and not busy
        can_install = self.job is None and status is not None and status.supported and not status.installed
        dialog.close_button.setText(trans.t('btn_cancel' if busy else 'btn_close'))
        dialog.close_button.setEnabled(not busy or not self.job.isInterruptionRequested())
        dialog.install_button.setText(trans.t('btn_download_install'))
        dialog.install_button.setVisible(can_install)
        dialog.local_button.setText(trans.t('ai_pack_install_local'))
        dialog.local_button.setVisible(can_install)
        source_label = trans.t('ai_pack_download_source')
        dialog.source.setAccessibleName(source_label)
        dialog.source.setToolTip(source_label)
        for index, (key, _) in enumerate(_PRIORITY_OPTIONS):
            dialog.source.setItemText(index, trans.t(key))
        dialog.source.setVisible(can_install)
        dialog.progress.setVisible(self.job is not None)
        dialog.error.setText(self.error)
        dialog.error.setVisible(bool(self.error))
        if self.job is not None:
            self._show_progress()
        else:
            feedback = trans.t('compare_cancelled') if self.cancelled else ''
            dialog.feedback.setText(feedback)
            dialog.feedback.setVisible(bool(feedback))

        device = description = ''
        if checking:
            heading = trans.t('ai_pack_check')
        elif status is None:
            heading = trans.t('ai_pack_check_failed')
        else:
            device = status.device
            if busy:
                heading = trans.t('ai_pack_installing')
            elif not status.supported:
                heading = trans.t('ai_pack_bundled')
                reason_key = {'driver': 'ai_pack_driver', 'os_version': 'ai_pack_macos_update'}
                description = trans.t(reason_key.get(status.reason, 'ai_pack_base'))
            elif status.installed:
                heading = trans.t('ai_pack_installed')
                description = trans.t('ai_pack_ready')
            else:
                heading = trans.t('ai_pack_available')
                locale = QLocale(trans.current_lang)
                size = locale.toString(status.download_bytes / 10**9, 'f', 2)
                disk = locale.toString(status.required_free_bytes / 10**9, 'f', 1)
                description = trans.t('ai_pack_details').format(size=f'{size} GB', disk=f'{disk} GB')

        dialog.heading.setText(heading)
        dialog.progress.setAccessibleName(heading)
        for label, text in ((dialog.device, device), (dialog.engines, engines), (dialog.description, description)):
            label.setText(text)
            label.setVisible(bool(text))
        if dialog.isVisible():
            dialog.fit()
