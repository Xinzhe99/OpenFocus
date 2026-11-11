"""Retry explicitly submitted reports without blocking the interface."""
import logging
import threading
import weakref

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox

from locales import trans
from utils.app_paths import user_data_dir
from utils.report_client import ReportClient, automatic_snapshot, revoke_automatic
from utils.reporting import retry_reports
from utils.settings_store import get_settings, REPORTS_AUTO_SEND_KEY, REPORTS_CHOICE_KEY

class _CleanupNotifier(QObject):
    failed = pyqtSignal()

    def __init__(self, app):
        super().__init__(app)
        self.failed.connect(self._show_failure)

    def _show_failure(self):
        QMessageBox.warning(QApplication.activeWindow(), trans.t('msg_error'),
                            trans.t('report_cleanup_failed'))


_NOTIFIER = None


def discard_automatic_reports():
    """Revoke synchronously, then delete queued automatic reports."""
    directory = user_data_dir() / 'reports'
    global _NOTIFIER
    app = QApplication.instance()
    if app is not None and _NOTIFIER is None:
        _NOTIFIER = _CleanupNotifier(app)
    notifier = _NOTIFIER
    revoke_automatic(directory)
    def discard():
        try:
            ReportClient(directory).discard_automatic(only_if_revoked=True)
        except Exception:
            logging.getLogger(__name__).error('Automatic report security cleanup failed')
            if notifier is not None:
                notifier.failed.emit()
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
        permission = automatic_snapshot(user_data_dir() / 'reports')
        threading.Thread(target=self._run, args=(account, auto_send, account_only, permission), daemon=True).start()

    def _run(self, account, auto_send, account_only, permission):
        try:
            retry_reports(account, include_auto=auto_send, account_only_auto=account_only,
                          permission=permission)
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
    settings = get_settings()
    if not settings.value(REPORTS_AUTO_SEND_KEY, False, type=bool):
        discard_automatic_reports()
    # One timer services the entire queue; each entry has its own backoff.
    if not _ACTIVE:
        _Retry(window, app).begin()
