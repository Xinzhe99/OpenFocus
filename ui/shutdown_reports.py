"""Finish consented error-report delivery after the last window closes."""
import threading
import time

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication

from utils.account_client import AccountClient
from utils.app_paths import user_data_dir
from utils.logging_setup import get_logger
from utils.report_client import ReportClient, automatic_snapshot, revoke_automatic
from utils.reporting import retry_reports
from utils.settings_store import (automatic_reports_allowed, get_settings,
                                  REPORTS_AUTO_SEND_KEY, REPORTS_CHOICE_KEY)

_ACTIVE_JOBS = set()


class _ShutdownReport(QObject):
    finished = pyqtSignal()

    def __init__(self, account, auto_send, account_only, *, opted_out):
        super().__init__()
        self.account = account
        self.auto_send = auto_send
        self.account_only = account_only
        self.opted_out = opted_out
        self.cancel = threading.Event()
        self.deadline = time.monotonic() + 5.0
        self.permission = automatic_snapshot(user_data_dir() / 'reports')

    def start(self):
        _ACTIVE_JOBS.add(self)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            if self.opted_out:
                revoke_automatic(user_data_dir() / 'reports')
                ReportClient(user_data_dir() / 'reports').discard_automatic(only_if_revoked=True)
            account = self.account if self.account is not None else AccountClient()
            enabled = self.auto_send and (not self.account_only or account.is_signed_in())
            retry_reports(account, include_auto=enabled, account_only_auto=self.account_only,
                          permission=self.permission, cancel=self.cancel, deadline=self.deadline)
        except Exception:
            # Keep the pending reports available for the next startup.
            get_logger(__name__).warning('Could not deliver queued reports at shutdown')
        finally:
            self.finished.emit()
            _ACTIVE_JOBS.discard(self)


def start_shutdown_reports(window):
    """Keep only the event loop alive for up to five seconds after UI closure."""
    settings = get_settings()
    enabled = automatic_reports_allowed(settings)
    if not (user_data_dir() / 'reports' / 'reports.json').exists():
        return
    app = QApplication.instance()
    if app is None:
        return
    app.setQuitOnLastWindowClosed(False)
    account_only = settings.value(REPORTS_CHOICE_KEY) == 'account_default'
    opted_out = not settings.value(REPORTS_AUTO_SEND_KEY, True, type=bool)
    job = _ShutdownReport(getattr(window, '_account_client', None), enabled, account_only,
                          opted_out=opted_out)
    timer = QTimer(app)
    timer.setSingleShot(True)
    def expire():
        job.cancel.set()
        app.quit()
    timer.timeout.connect(expire)
    job.finished.connect(app.quit)
    job.finished.connect(timer.stop)
    timer.start(5000)
    job.start()
