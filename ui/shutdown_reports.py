"""Finish consented error-report delivery after the last window closes."""
import threading

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication

from locales import trans
from utils.account_client import AccountClient
from utils.app_paths import user_data_dir
from utils.report_bundle import build_report_bundle, session_has_error
from utils.report_client import ReportClient
from utils.reporting import retry_reports, send_bundle
from utils.settings_store import get_settings, REPORTS_AUTO_SEND_KEY, REPORTS_CHOICE_KEY

_ACTIVE_JOBS = set()


class _ShutdownReport(QObject):
    finished = pyqtSignal()

    def __init__(self, account, locale, auto_send, account_only=False):
        super().__init__()
        self.account = account
        self.locale = locale
        self.auto_send = auto_send
        self.account_only = account_only

    def start(self):
        _ACTIVE_JOBS.add(self)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            account = self.account if self.account is not None else AccountClient()
            enabled = self.auto_send and (not self.account_only or account.is_signed_in())
            if not self.auto_send:
                ReportClient(user_data_dir() / 'reports').discard_automatic()
            if enabled and session_has_error():
                bundle = build_report_bundle()
                if bundle.has_error:
                    send_bundle(bundle, source='auto', locale=self.locale, account=account,
                                require_account=self.account_only)
            if ReportClient(user_data_dir() / 'reports').pending_count:
                retry_reports(account, include_auto=enabled, account_only_auto=self.account_only)
        except Exception:
            # Never expose request contents or prevent shutdown.
            pass
        finally:
            self.finished.emit()
            _ACTIVE_JOBS.discard(self)


def start_shutdown_reports(window):
    """Keep only the event loop alive for up to five seconds after UI closure."""
    settings = get_settings()
    enabled = settings.value(REPORTS_AUTO_SEND_KEY, False, type=bool)
    if not enabled and not (user_data_dir() / 'reports' / 'reports.json').exists():
        return
    app = QApplication.instance()
    if app is None:
        return
    app.setQuitOnLastWindowClosed(False)
    account_only = settings.value(REPORTS_CHOICE_KEY) == 'account_default'
    job = _ShutdownReport(getattr(window, '_account_client', None), trans.current_lang,
                          enabled, account_only)
    timer = QTimer(app)
    timer.setSingleShot(True)
    timer.timeout.connect(app.quit)
    job.finished.connect(app.quit)
    job.finished.connect(timer.stop)
    timer.start(5000)
    job.start()
