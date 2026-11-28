"""Review local logs and optionally send a sanitized diagnostic report."""

import os
import webbrowser

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices, QFontDatabase
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import (
    QDialog, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QRadioButton, QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from locales import trans
from utils import logging_setup

ISSUES_URL = "https://github.com/Xinzhe99/OpenFocus/issues"

MAX_INLINE_LOG = 40_000

def _note(text=""):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setObjectName("mutedLabel")
    return label


class FeedbackDialog(QDialog):
    """Review local diagnostics before choosing what to submit."""

    def __init__(self, window_main, parent=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self.setWindowTitle(trans.t("feedback_title"))
        self.resize(660, 420)
        self._facts_probe = None
        self._facts_timer = QTimer(self)
        self._facts_timer.setInterval(100)
        self._facts_timer.timeout.connect(self._poll_facts)
        self._account_job = None
        self._export_path = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(12)
        tabs = QTabWidget()
        tabs.addTab(self._channels_page(), trans.t('feedback_tab_channels'))
        tabs.addTab(self._diagnostics_page(), trans.t('feedback_tab_diagnostics'))
        root.addWidget(tabs, 1)

        footer = QHBoxLayout()
        footer.addStretch()
        logs = QPushButton(trans.t('menu_open_logs'))
        logs.clicked.connect(self._open_logs_folder)
        footer.addWidget(logs)
        close = QPushButton(trans.t('btn_close'))
        close.clicked.connect(self.reject)
        footer.addWidget(close)
        root.addLayout(footer)

    def _channels_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        channels = QGroupBox(trans.t('feedback_channels_title'))
        channels.setToolTip(trans.t('feedback_channels_hint'))
        channel_layout = QVBoxLayout(channels)
        actions = QHBoxLayout()
        issues = QPushButton(trans.t('feedback_open_issues'))
        issues.setToolTip(trans.t('feedback_channels_hint'))
        issues.clicked.connect(self._open_issues)
        client = getattr(self.window_main, '_account_client', None)
        if client is not None and client.is_signed_in():
            self.account_button = QPushButton(trans.t('feedback_account_center'))
            self.account_button.setObjectName('primaryButton')
            self.account_button.clicked.connect(self._open_account_center)
            actions.addWidget(self.account_button)
        else:
            issues.setObjectName('primaryButton')
        actions.addWidget(issues)
        actions.addStretch()
        channel_layout.addLayout(actions)
        self.account_status = _note()
        self.account_status.hide()
        channel_layout.addWidget(self.account_status)
        layout.addWidget(channels)

        report = QGroupBox(trans.t('feedback_error_title'))
        report.setToolTip(trans.t('feedback_error_hint'))
        report_layout = QVBoxLayout(report)
        review = QPushButton(trans.t('report_review'))
        review.setToolTip(trans.t('feedback_error_hint'))
        review.clicked.connect(self._review_report)
        report_layout.addWidget(review, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(report)
        layout.addStretch()
        return page

    def _diagnostics_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        view_controls = QHBoxLayout()
        log = QRadioButton(trans.t('feedback_log_name'))
        facts = QRadioButton(trans.t('action_env_info'))
        log.setChecked(True)
        view_controls.addWidget(log)
        view_controls.addWidget(facts)
        view_controls.addStretch()
        self.copy_button = QPushButton(trans.t('feedback_copy_log'))
        self.copy_button.clicked.connect(self._copy_log)
        view_controls.addWidget(self.copy_button)
        layout.addLayout(view_controls)

        self.views = QStackedWidget()
        self.log_view = QPlainTextEdit()
        self.facts_view = QPlainTextEdit()
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        for view, label in ((self.log_view, log), (self.facts_view, facts)):
            view.setFont(fixed_font)
            view.setReadOnly(True)
            view.setAccessibleName(label.text())
            self.views.addWidget(view)
        self.log_view.setPlainText(logging_setup.session_log_excerpt(MAX_INLINE_LOG))
        layout.addWidget(self.views, 1)
        self.log_path = _note(trans.t('feedback_log_hint').format(path=logging_setup.session_log_path()))
        self.log_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.log_path)
        facts.toggled.connect(self._show_facts)

        export_actions = QHBoxLayout()
        export = QPushButton(trans.t('feedback_export'))
        export.setToolTip(trans.t('feedback_local_export_scope'))
        export.clicked.connect(self._export_package)
        export_actions.addWidget(export)
        self.open_folder_button = QPushButton(trans.t('feedback_open_export_folder'))
        self.open_folder_button.clicked.connect(self._open_export_folder)
        self.open_folder_button.hide()
        export_actions.addWidget(self.open_folder_button)
        export_actions.addStretch()
        layout.addLayout(export_actions)
        self.diagnostic_status = _note()
        self.diagnostic_status.hide()
        layout.addWidget(self.diagnostic_status)
        return page

    def _show_facts(self, visible):
        self.views.setCurrentWidget(self.facts_view if visible else self.log_view)
        self.copy_button.setVisible(not visible)
        self.log_path.setVisible(not visible)
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
        self.account_status.setText(trans.t('account_wait_browser'))
        self.account_status.show()
        job.finished.connect(self._account_center_finished)
        job.finished.connect(job._release)
        job.start()

    def _account_center_finished(self, result, error):
        from dialogs.account import AccountDialog
        self._account_job = None
        self.account_button.setEnabled(True)
        if error is not None and error.category != 'timeout':
            self.account_status.setText(AccountDialog._error_text(error))
        elif error is None and not result:
            self.account_status.setText(trans.t('account_generic_error'))
        else:
            self.account_status.hide()

    def _copy_log(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.log_view.toPlainText())
        self.diagnostic_status.setText(trans.t('feedback_copied'))
        self.diagnostic_status.show()

    def _open_logs_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(logging_setup.logs_dir().absolute())))

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
        self.open_folder_button.hide()
        self._export_path = ""
        try:
            path = logging_setup.export_feedback_package(dest)
        except Exception as exc:
            self.diagnostic_status.setText(trans.t('feedback_export_failed').format(
                error=f"{dest}: {exc}"))
            self.diagnostic_status.show()
            return
        self._export_path = os.path.abspath(str(path))
        self.diagnostic_status.setText(trans.t('feedback_exported').format(path=self._export_path))
        self.diagnostic_status.show()
        self.open_folder_button.setVisible(True)
