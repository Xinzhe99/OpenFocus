"""Review local logs and optionally send a sanitized diagnostic report."""

import os
import webbrowser

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QFont
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

from constants import app_display_version
from locales import trans
from utils import logging_setup

ISSUES_URL = "https://github.com/Xinzhe99/OpenFocus/issues"

MAX_INLINE_LOG = 40_000





class FeedbackDialog(QDialog):
    """Review local diagnostics before choosing what to submit."""

    def __init__(self, window_main, parent=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self.setWindowTitle(trans.t("feedback_title"))
        self.resize(720, 560)
        self._facts_probe = None
        self._facts_timer = QTimer(self)
        self._facts_timer.setInterval(100)
        self._facts_timer.timeout.connect(self._poll_facts)
        self._account_job = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(6)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        self.log_view.setFont(f)
        self.log_view.setPlainText(
            logging_setup.session_log_excerpt(MAX_INLINE_LOG))
        root.addWidget(self.log_view, 1)

        self.log_hint = QLabel(trans.t("feedback_log_hint").format(
            path=str(logging_setup.session_log_path())))
        self.log_hint.setWordWrap(True)
        self.log_hint.setObjectName("mutedLabel")
        root.addWidget(self.log_hint)
        scope = QLabel(trans.t("feedback_local_export_scope"))
        scope.setWordWrap(True)
        root.addWidget(scope)
        facts_button = QPushButton(trans.t("action_env_info"))
        facts_button.setCheckable(True)
        root.addWidget(facts_button)
        self.facts_view = QPlainTextEdit()
        self.facts_view.setReadOnly(True)
        self.facts_view.setVisible(False)
        root.addWidget(self.facts_view)
        facts_button.toggled.connect(self._show_facts)
        self.open_folder_button = QPushButton(trans.t("action_open_folder"))
        self.open_folder_button.setVisible(False)
        self.open_folder_button.clicked.connect(self._open_export_folder)
        root.addWidget(self.open_folder_button)
        self._export_path = ""

        buttons = QHBoxLayout()
        copy_btn = QPushButton(trans.t("feedback_copy_log"))
        copy_btn.setToolTip(trans.t("feedback_head").format(version=app_display_version()))
        copy_btn.clicked.connect(self._copy_log)
        issues_btn = QPushButton(trans.t("feedback_open_issues"))
        issues_btn.setToolTip(trans.t("feedback_head").format(version=app_display_version()))
        issues_btn.clicked.connect(self._open_issues)
        export_btn = QPushButton(trans.t("feedback_export"))
        export_btn.setToolTip(trans.t("feedback_head").format(version=app_display_version()))
        export_btn.clicked.connect(self._export_package)
        buttons.addWidget(copy_btn)
        buttons.addWidget(issues_btn)
        buttons.addWidget(export_btn)
        client = getattr(window_main, '_account_client', None)
        if client is not None and client.is_signed_in():
            self.account_button = QPushButton(trans.t('feedback_account_center'))
            self.account_button.clicked.connect(self._open_account_center)
            root.addWidget(self.account_button)
        report_btn = QPushButton(trans.t('report_review'))
        report_btn.clicked.connect(self._review_report)
        buttons.addWidget(report_btn)
        buttons.addStretch()
        close_btn = QPushButton(trans.t("btn_close"))
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

    def _show_facts(self, visible):
        self.facts_view.setVisible(visible)
        if visible and self._facts_probe is None:
            from dialogs.about import _EnvironmentProbe
            self.facts_view.setPlainText(trans.t('msg_gif_processing_title'))
            self._facts_probe = _EnvironmentProbe()
            self._facts_probe.start()
            self._facts_timer.start()

    def _poll_facts(self):
        if self._facts_probe.finished.is_set():
            self._facts_timer.stop()
            self.facts_view.setPlainText(self._facts_probe.text)

    def _open_account_center(self):
        client = getattr(self.window_main, '_account_client', None)
        if self._account_job is not None or client is None or not client.is_signed_in():
            return
        from dialogs.account import _AccountJob
        job = _AccountJob(lambda _started: client.open_web_handoff(timeout=8), self.window_main)
        self._account_job = job
        self.account_button.setEnabled(False)
        self.log_hint.setText(trans.t('account_wait_browser'))
        job.finished.connect(self._account_center_finished)
        job.finished.connect(job._release)
        job.start()

    def _account_center_finished(self, result, error):
        from dialogs.account import AccountDialog
        self._account_job = None
        self.account_button.setEnabled(True)
        if error is not None and error.category != 'timeout':
            self.log_hint.setText(AccountDialog._error_text(error))
        elif error is None and not result:
            self.log_hint.setText(trans.t('account_generic_error'))
        else:
            self.log_hint.setText(trans.t('feedback_log_hint').format(
                path=str(logging_setup.session_log_path())))

    def _copy_log(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.log_view.toPlainText())
        self.log_hint.setText(trans.t("feedback_copied"))

    def _open_export_folder(self):
        if self._export_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(self._export_path)))

    # --------------------------------------------------------------- routes
    def _open_issues(self):
        """Just open the tracker: users who file issues write there directly.

        The app's job is to hand over the log, not to host a second form.
        """
        webbrowser.open(ISSUES_URL)

    def _review_report(self):
        from dialogs.report_review import ReportReviewDialog
        ReportReviewDialog(self.window_main, self).exec()

    def _export_package(self):
        dest, _ = FileDialog.getSaveFileName(
            self, trans.t("feedback_export"),
            os.path.join(os.path.expanduser("~"),
                         f"OpenFocus_feedback_"
                         f"{logging_setup.session_id()}.zip"),
            "Zip (*.zip)")
        if not dest:
            return
        try:
            path = logging_setup.export_feedback_package(dest)
        except Exception as exc:
            self.log_hint.setText(trans.t("feedback_export_failed").format(
                error=f"{dest}: {exc}"))
            return
        self._export_path = os.path.abspath(str(path))
        self.log_hint.setText(trans.t("feedback_exported").format(path=self._export_path))
        self.open_folder_button.setVisible(True)
