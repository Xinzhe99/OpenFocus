"""Persist automatic reports as errors occur and deliver them off the GUI thread."""
import logging
import threading
import time
import weakref

from PyQt6.QtCore import QObject, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox

from locales import trans
from utils import diagnostics, logging_setup
from utils.app_paths import user_data_dir
from utils.report_bundle import snapshot_failure, snapshot_sign_in_failure
from utils.report_client import ReportClient, automatic_snapshot, revoke_automatic
from utils.reporting import build_channel, report_metadata, retry_reports
from utils.settings_store import automatic_reports_allowed, get_settings, REPORTS_AUTO_SEND_KEY, REPORTS_CHOICE_KEY

_logger = logging_setup.get_logger(__name__)

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
            _logger.error('Automatic report security cleanup failed')
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
        self.in_flight = False
        self.new_report = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._start)
        self.finished.connect(self._finished)

    def begin(self):
        _ACTIVE.add(self)
        self.timer.start(60000)

    def dispatch_soon(self):
        if self.in_flight:
            self.new_report = True
        elif not self.timer.isActive() or self.timer.remainingTime() > 1000:
            self.attempts = 0
            self.timer.start(1000)

    def _start(self):
        self.in_flight = True
        self.attempts += 1
        window = self.window()
        account = getattr(window, '_account_client', None) if window is not None else None
        settings = get_settings()
        auto_send = automatic_reports_allowed(settings)
        account_only = settings.value(REPORTS_CHOICE_KEY) == 'account_default'
        permission = automatic_snapshot(user_data_dir() / 'reports')
        threading.Thread(target=self._run, args=(account, auto_send, account_only, permission), daemon=True).start()

    def _run(self, account, auto_send, account_only, permission):
        try:
            retry_reports(account, include_auto=auto_send, account_only_auto=account_only,
                          permission=permission)
            pending = bool(ReportClient(user_data_dir() / 'reports').pending_count)
        except Exception:
            _logger.warning('Could not retry queued reports')
            pending = False
        self.finished.emit(pending)

    def _finished(self, pending):
        self.in_flight = False
        if self.new_report:
            self.new_report = False
            self.dispatch_soon()
        elif pending and self.attempts < 3:
            self.timer.start(60000 * (2 ** self.attempts))
        else:
            _ACTIVE.discard(self)
            self.deleteLater()


def schedule_report_retries(window, *, urgent=False):
    app = QApplication.instance()
    if app is None:
        return
    settings = get_settings()
    if not settings.value(REPORTS_AUTO_SEND_KEY, True, type=bool):
        discard_automatic_reports()
    # One timer services the entire queue; each entry has its own backoff.
    job = next(iter(_ACTIVE), None)
    if job is None:
        job = _Retry(window, app)
        job.begin()
    if urgent:
        job.dispatch_soon()


class _CaptureTrigger(QObject):
    queued = pyqtSignal()

    def __init__(self, window, app):
        super().__init__(app)
        self.window = weakref.ref(window)
        self.queued.connect(self._deliver, Qt.ConnectionType.QueuedConnection)

    def _deliver(self):
        window = self.window()
        if window is not None and not getattr(window, '_shutting_down', False):
            schedule_report_retries(window, urgent=True)


class _FailureReports(logging.Handler):
    def __init__(self, trigger, settings):
        super().__init__(logging.ERROR)
        self.trigger = trigger
        self.settings_file = settings.fileName()
        self.settings_format = settings.format()
        self.path = user_data_dir() / 'reports'
        self.local = threading.local()
        self.facts = None
        self.reports = ReportClient(self.path)

    def handle(self, record):
        # Logging may already hold the account lock. Avoid reversing the lock
        # order across threads. The report queue serializes its own writes.
        filtered = self.filter(record)
        if filtered:
            self.emit(filtered if isinstance(filtered, logging.LogRecord) else record)
        return filtered

    def emit(self, record):
        if getattr(self.local, 'capturing', False) or getattr(record, 'report_previous_crash', False):
            return
        self.local.capturing = True
        try:
            self._capture(record)
        except Exception:
            _logger.warning('Could not persist an automatic error report')
        finally:
            self.local.capturing = False

    def _capture(self, record):
        permission = automatic_snapshot(self.path)
        settings = QSettings(self.settings_file, self.settings_format)
        enabled = automatic_reports_allowed(settings)
        sign_in = hasattr(record, 'report_sign_in')
        owner = None
        if sign_in:
            if not enabled and (settings.value(REPORTS_CHOICE_KEY) not in (None, 'unselected')
                                or not settings.value(REPORTS_AUTO_SEND_KEY, True, type=bool)):
                return
            bundle = snapshot_sign_in_failure(record)
            if bundle is None:
                return
        else:
            if not enabled:
                return
            window = self.trigger.window()
            account = getattr(window, '_account_client', None) if window is not None else None
            user = account.current_user() if account is not None else None
            owner = user.get('id') if isinstance(user, dict) else None
            if not owner and (settings.value(REPORTS_CHOICE_KEY) == 'account_default'
                              or account is not None and account.is_signed_in()):
                return
            if self.facts is None:
                self.facts = diagnostics.system_facts()
            bundle = snapshot_failure(record, self.facts)
        metadata = report_metadata(bundle, source='auto', locale=trans.current_lang)
        result = self.reports.enqueue(metadata, bundle.data, owner_id=owner, sign_in=sign_in,
                                      permission=permission, deadline=time.monotonic() + .25)
        if result == 'queued' and enabled:
            self.trigger.queued.emit()


def start_failure_capture(window):
    if build_channel() != 'release':
        return
    app = QApplication.instance()
    trigger = _CaptureTrigger(window, app)
    handler = _FailureReports(trigger, get_settings())
    logger = logging_setup.get_logger()
    logger.addHandler(handler)

    def stop():
        logger.removeHandler(handler)
        handler.close()

    app.aboutToQuit.connect(stop)
