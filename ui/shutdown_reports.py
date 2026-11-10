"""Finish consented error-report delivery after the last window closes."""
import threading

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication

from locales import trans
from utils.app_paths import user_data_dir
from utils.report_bundle import build_report_bundle, session_has_error
from utils.report_client import ReportClient
from utils.reporting import retry_reports, send_bundle
from utils.settings_store import get_settings

_ACTIVE_JOBS = set()


class _ShutdownReport(QObject):
    finished = pyqtSignal()

    def __init__(self, account, locale, auto_send):
        super().__init__()
        self.account = account
        self.locale = locale
        self.auto_send = auto_send

    def start(self):
        _ACTIVE_JOBS.add(self)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            if not self.auto_send:
                ReportClient(user_data_dir() / 'reports').discard_automatic()
            if self.auto_send and session_has_error():
                bundle = build_report_bundle()
                if bundle.has_error:
                    send_bundle(bundle, source='auto', locale=self.locale, account=self.account)
            if ReportClient(user_data_dir() / 'reports').pending_count:
                retry_reports(self.account, include_auto=self.auto_send)
        except Exception:
            # Never expose request contents or prevent shutdown.
            pass
        finally:
            self.finished.emit()
            _ACTIVE_JOBS.discard(self)


def start_shutdown_reports(window):
    """Keep only the event loop alive for up to five seconds after UI closure."""
    settings = get_settings()
    enabled = settings.value('reports/auto_send', False, type=bool)
    if not enabled and not (user_data_dir() / 'reports' / 'reports.json').exists():
        return
    app = QApplication.instance()
    if app is None:
        return
    app.setQuitOnLastWindowClosed(False)
    job = _ShutdownReport(getattr(window, '_account_client', None), trans.current_lang, enabled)
    timer = QTimer(app)
    timer.setSingleShot(True)
    timer.timeout.connect(app.quit)
    job.finished.connect(app.quit)
    job.finished.connect(timer.stop)
    timer.start(5000)
    job.start()
