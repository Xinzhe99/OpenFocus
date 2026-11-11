"""Retry explicitly submitted reports without blocking the interface."""
import threading
import weakref

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication

from utils.app_paths import user_data_dir
from utils.report_client import ReportClient
from utils.reporting import retry_reports
from utils.settings_store import get_settings, REPORTS_AUTO_SEND_KEY, REPORTS_CHOICE_KEY


def discard_automatic_reports():
    """Delete unsent automatic reports without holding the UI thread."""
    def discard():
        try:
            ReportClient(user_data_dir() / 'reports').discard_automatic()
        except Exception:
            pass
    threading.Thread(target=discard, daemon=True).start()

_ACTIVE = set()


class _Retry(QObject):
    finished = pyqtSignal(bool)

    def __init__(self, window, app):
        super().__init__(app)
        self.window = weakref.ref(window)
        self.attempts = 0
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._start)
        self.finished.connect(self._finished)

    def begin(self):
        _ACTIVE.add(self)
        self.timer.start(60000)

    def _start(self):
        self.attempts += 1
        window = self.window()
        account = getattr(window, '_account_client', None) if window is not None else None
        settings = get_settings()
        auto_send = settings.value(REPORTS_AUTO_SEND_KEY, False, type=bool)
        account_only = settings.value(REPORTS_CHOICE_KEY) == 'account_default'
        threading.Thread(target=self._run, args=(account, auto_send, account_only), daemon=True).start()

    def _run(self, account, auto_send, account_only):
        try:
            retry_reports(account, include_auto=auto_send, account_only_auto=account_only)
            pending = bool(ReportClient(user_data_dir() / 'reports').pending_count)
        except Exception:
            pending = False
        self.finished.emit(pending)

    def _finished(self, pending):
        if pending and self.attempts < 3:
            self.timer.start(60000 * (2 ** self.attempts))
        else:
            _ACTIVE.discard(self)
            self.deleteLater()


def schedule_report_retries(window):
    app = QApplication.instance()
    if app is None:
        return
    # One timer services the entire queue; each entry has its own backoff.
    if not _ACTIVE:
        _Retry(window, app).begin()
