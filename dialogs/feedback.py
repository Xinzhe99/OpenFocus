"""Review local logs and optionally send a sanitized diagnostic report."""

import os
import sys
import webbrowser

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QFont
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

from constants import APP_VERSION
from locales import trans
from utils import logging_setup

ISSUES_URL = "https://github.com/Xinzhe99/OpenFocus/issues"

MAX_INLINE_LOG = 40_000


def _system_facts() -> str:
    import platform
    lines = [f"- OpenFocus {APP_VERSION}",
             f"- OS: {platform.platform()}",
             f"- Python: {sys.version.split()[0]}"]
    try:
        from utils import memory
        lines.append(f"- RAM: {memory.total_memory_bytes() / 1024 ** 3:.1f} GB"
                     f" ({memory.available_memory_bytes() / 1024 ** 3:.1f} GB free)")
    except Exception:
        pass
    try:
        from core.multi_focus_fusion import neural_device_kind
        from fusion_methods import neural_runtime
        kind = neural_device_kind()
        status = "not checked" if kind is None else neural_runtime.display_name(kind)
        lines.append(f"- GPU: {status}")
    except Exception:
        pass
    return "\n".join(lines)


class FeedbackDialog(QDialog):
    """Review local diagnostics before choosing what to submit."""

    def __init__(self, window_main, parent=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self.setWindowTitle(trans.t("feedback_title"))
        self.resize(720, 560)

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
        copy_btn.setToolTip(trans.t("feedback_head").format(version=APP_VERSION))
        copy_btn.clicked.connect(self._copy_log)
        issues_btn = QPushButton(trans.t("feedback_open_issues"))
        issues_btn.setToolTip(trans.t("feedback_head").format(version=APP_VERSION))
        issues_btn.clicked.connect(self._open_issues)
        export_btn = QPushButton(trans.t("feedback_export"))
        export_btn.setToolTip(trans.t("feedback_head").format(version=APP_VERSION))
        export_btn.clicked.connect(self._export_package)
        buttons.addWidget(copy_btn)
        buttons.addWidget(issues_btn)
        buttons.addWidget(export_btn)
        report_btn = QPushButton(trans.t('report_review'))
        report_btn.clicked.connect(self._review_report)
        buttons.addWidget(report_btn)
        buttons.addStretch()
        close_btn = QPushButton(trans.t("btn_close"))
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

    def _show_facts(self, visible):
        if visible and not self.facts_view.toPlainText():
            self.facts_view.setPlainText(_system_facts())
        self.facts_view.setVisible(visible)

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
