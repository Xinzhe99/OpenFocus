"""User feedback: GitHub issue, e-mail, or a packaged report.

Goal: when something breaks, the report must carry enough context to act on —
app version, OS, memory, and the session log showing what the user actually
did. Typing all of that by hand never happens, so the dialog assembles it and
offers three ways to send it:

1. GitHub issue — opens the tracker with title and body prefilled. The user
   attaches the exported report file.
2. E-mail — opens the mail client with subject and body prefilled. Direct
   SMTP from a desktop app would need a hard-coded account password, which is
   not acceptable in an open-source binary, so the mail client carries the
   send.
3. Export package — writes a zip (description + system facts + session log +
   full logs) the user can attach to either of the above.
"""
import os
import sys
import webbrowser

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel,
    QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout,
)

from constants import APP_VERSION
from locales import trans
from utils import logging_setup
from utils.platform_utils import has_compute_gpu

ISSUES_URL = "https://github.com/Xinzhe99/OpenFocus/issues"
CONTACT_EMAIL = "xiexinzheyyy@163.com"

MAX_INLINE_LOG = 40_000  # characters embedded in the issue/mail body


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
    """Collect a bug report or suggestion and route it to the developer."""

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
            path=str(logging_setup.logs_dir() / "openfocus.log")))
        self.log_hint.setWordWrap(True)
        self.log_hint.setObjectName("mutedLabel")
        root.addWidget(self.log_hint)

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

    # -------------------------------------------------------------- content
    def _compose(self) -> str:
        """Environment facts + session log, ready to paste into an issue."""
        return "\n".join([
            "### Environment",
            _system_facts(),
            "",
            "### Session log (trimmed)",
            "```",
            logging_setup.session_log_excerpt(MAX_INLINE_LOG),
            "```",
        ])

    def _copy_log(self):
        """Copy the same report the issue/mail body would carry.

        The button said "copy log" and copied only the log; the environment
        facts were silently dropped. A status line is enough feedback — the
        modal box demanded an extra click for a completed copy.
        """
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._compose())
        self.log_hint.setText(trans.t("feedback_copied"))

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
                         f"{logging_setup._session_stamp()}.zip"),
            "Zip (*.zip)")
        if not dest:
            return
        try:
            path = logging_setup.export_feedback_package(dest)
        except Exception as exc:  # pragma: no cover - platform dependent
            QMessageBox.warning(
                self, trans.t("feedback_title"),
                trans.t("feedback_export_failed").format(error=str(exc)))
            return
        QMessageBox.information(
            self, trans.t("feedback_title"),
            trans.t("feedback_exported").format(path=str(path)))
