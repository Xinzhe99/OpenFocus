"""Inspect local logs, copy them, or export a report for manual submission.

Opening the issue tracker sends no report, logs, or images.
"""
import os
import sys
import webbrowser

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

from constants import APP_VERSION
from locales import trans
from utils import logging_setup
from utils.platform_utils import has_compute_gpu

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
        lines.append(f"- CUDA/MPS GPU: {has_compute_gpu()}")
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

        # The log IS the content: the two paragraphs of intro went to the
        # docstring, and the log view fills the page instead of a 150 px
        # sliver under a wall of blank space.
        head = QLabel(trans.t("feedback_head").format(version=APP_VERSION))
        head.setWordWrap(True)
        root.addWidget(head)

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
        scope = QLabel(trans.t("feedback_package_scope"))
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
        copy_btn.clicked.connect(self._copy_log)
        issues_btn = QPushButton(trans.t("feedback_open_issues"))
        issues_btn.clicked.connect(self._open_issues)
        export_btn = QPushButton(trans.t("feedback_export"))
        export_btn.clicked.connect(self._export_package)
        buttons.addWidget(copy_btn)
        buttons.addWidget(issues_btn)
        buttons.addWidget(export_btn)
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

    def _export_package(self):
        dest, _ = QFileDialog.getSaveFileName(
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
