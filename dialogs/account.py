"""Optional desktop account access; all account I/O runs off the GUI thread."""
import threading

from PyQt6.QtCore import QObject, QUrl, QUrlQuery, Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPushButton, QVBoxLayout)

from locales import trans
from utils.account_client import AccountClient, AccountError
from utils.settings_store import get_settings

_ACTIVE_JOBS = set()


class _AccountJob(QObject):
    started = pyqtSignal()
    finished = pyqtSignal(object, object)

    def __init__(self, call):
        super().__init__()
        self.call = call
        self._thread = None

    def _release(self, _result, _error):
        self._thread.join()
        _ACTIVE_JOBS.discard(self)
        self.deleteLater()

    def start(self):
        _ACTIVE_JOBS.add(self)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            value = self.call(self.started.emit)
            self.finished.emit(value, None)
        except AccountError as exc:
            self.finished.emit(None, exc)
        except Exception:
            self.finished.emit(None, AccountError('unknown'))


class AccountDialog(QDialog):
    """Sign in voluntarily or continue immediately without an account."""

    def __init__(self, window_main, parent=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self.client = getattr(window_main, '_account_client', None)
        self._job = None
        self._operation = None
        self._cancel = threading.Event()
        self._terms = None
        self.setWindowTitle(trans.t('account_title'))
        self.resize(560, 450)
        root = QVBoxLayout(self)
        explanation = QLabel(trans.t('account_explanation'))
        explanation.setWordWrap(True)
        explanation.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(explanation)
        usage_note = QLabel(trans.t('account_usage_note'))
        usage_note.setWordWrap(True)
        usage_note.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(usage_note)
        self.status = QLabel(trans.t('account_checking'))
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self.status)
        self.email = QLineEdit()
        self.email.setPlaceholderText(trans.t('account_email'))
        self.email.setClearButtonEnabled(True)
        root.addWidget(self.email)
        self.terms = QCheckBox(trans.t('account_terms'))
        root.addWidget(self.terms)
        self.terms_link = QPushButton(trans.t('account_open_terms'))
        self.terms_link.clicked.connect(self._open_terms)
        root.addWidget(self.terms_link)
        self.privacy_link = QPushButton(trans.t('account_open_privacy'))
        self.privacy_link.clicked.connect(self._open_privacy)
        root.addWidget(self.privacy_link)
        self.auto_reports = QCheckBox(trans.t('account_auto_reports'))
        settings = get_settings()
        self.auto_reports.setChecked(settings.value('reports/auto_send', False, type=bool))
        root.addWidget(self.auto_reports)
        self.email_button = QPushButton(trans.t('account_email_send'))
        self.github_button = QPushButton(trans.t('account_github'))
        self.google_button = QPushButton(trans.t('account_google'))
        self.email_button.clicked.connect(lambda: self._sign_in('email'))
        self.github_button.clicked.connect(lambda: self._sign_in('github'))
        self.google_button.clicked.connect(lambda: self._sign_in('google'))
        for button in (self.email_button, self.github_button, self.google_button):
            root.addWidget(button)
        account_actions = QHBoxLayout()
        self.logout_button = QPushButton(trans.t('account_sign_out'))
        self.delete_button = QPushButton(trans.t('account_delete'))
        self.logout_button.clicked.connect(self._logout)
        self.delete_button.clicked.connect(self._delete_account)
        account_actions.addWidget(self.logout_button)
        account_actions.addWidget(self.delete_button)
        root.addLayout(account_actions)
        root.addStretch()
        self.skip_button = QPushButton(trans.t('account_skip'))
        self.skip_button.clicked.connect(self.reject)
        root.addWidget(self.skip_button)
        self._refresh_controls(busy=True)
        if self.client is None:
            self._start('load', lambda _started: AccountClient())
        else:
            self._refresh_controls()

    def _open_terms(self):
        if self._job is None and self.client is not None:
            self._start('terms', lambda _started: self.client.terms())

    def _open_privacy(self):
        url = ('https://openfocus.top/zh/#privacy' if trans.current_lang == 'zh'
               else 'https://openfocus.top/?lang=en#privacy')
        QDesktopServices.openUrl(QUrl(url))

    def _refresh_controls(self, busy=False):
        user = self.client.current_user() if self.client is not None else None
        signed_in = self.client is not None and self.client.is_signed_in()
        for widget in (self.email, self.terms, self.terms_link, self.privacy_link,
                       self.email_button, self.github_button, self.google_button):
            widget.setVisible(not signed_in)
            widget.setEnabled(not busy and self.client is not None)
        self.terms.setEnabled(not busy and self._terms is not None)
        self.auto_reports.setEnabled(not busy)
        self.logout_button.setVisible(signed_in)
        self.delete_button.setVisible(signed_in)
        self.logout_button.setEnabled(not busy)
        self.delete_button.setEnabled(not busy)
        self.skip_button.setText(trans.t('account_continue' if signed_in else 'account_skip'))
        if signed_in:
            email = user.get('email', '') if isinstance(user, dict) else ''
            self.status.setText(trans.t('account_signed_in').format(email=email))
        elif not busy:
            self.status.clear()

    def _start(self, operation, call):
        self._operation = operation
        job = _AccountJob(call)
        self._job = job
        job.started.connect(self._started)
        job.finished.connect(self._finished)
        job.finished.connect(job._release)
        self._refresh_controls(busy=True)
        job.start()

    def _started(self):
        key = 'account_wait_email' if self._operation == 'email' else 'account_wait_browser'
        self.status.setText(trans.t(key))

    def _sign_in(self, provider):
        if self._job is not None or self.client is None:
            return
        if self._terms is None or not self.terms.isChecked():
            self.status.setText(trans.t('account_terms_required'))
            return
        email = self.email.text().strip()
        if provider == 'email' and not email:
            self.status.setText(trans.t('account_email_required'))
            return
        self._cancel.clear()
        client = self.client
        version = self._terms['version']
        language = 'zh' if trans.current_lang == 'zh' else 'en'
        def login(started):
            return client.login(provider, email=email if provider == 'email' else None,
                                cancel=self._cancel, on_started=started,
                                terms_version=version, language=language)
        self._start(provider, login)

    def _logout(self):
        if self._job is None and self.client is not None:
            self._start('logout', lambda _started: self.client.logout())

    def _delete_account(self):
        if self._job is not None or self.client is None:
            return
        if QMessageBox.question(self, trans.t('account_title'), trans.t('account_delete_confirm'),
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self._start('delete', lambda _started: self.client.delete_account())

    def _finished(self, result, error):
        operation = self._operation
        self._job = None
        self._operation = None
        if self._cancel.is_set():
            return
        if error is not None and error.category in ('terms_required', 'invalid_terms_version'):
            self._terms = None
            self.terms.setChecked(False)
        if error is not None:
            self._refresh_controls()
            self.status.setText(self._error_text(error))
            return
        if operation == 'load':
            self.client = result
            self.window_main._account_client = result
        elif operation == 'terms':
            self._terms = result
            url = QUrl(result['url'])
            query = QUrlQuery(url)
            query.removeAllQueryItems('lang')
            query.addQueryItem('lang', 'zh' if trans.current_lang == 'zh' else 'en')
            url.setQuery(query)
            QDesktopServices.openUrl(url)
        elif operation in ('email', 'github', 'google'):
            self._save_report_preference()
        self._refresh_controls()
        if operation == 'logout':
            self.status.setText(trans.t('account_signed_out'))
        elif operation == 'delete':
            self.status.setText(trans.t('account_deleted'))

    @staticmethod
    def _error_text(error):
        if error.category in ('terms_required', 'invalid_terms_version'):
            key = 'account_terms_required'
        elif error.category == 'verified_email_required':
            key = 'account_email_unverified'
        elif error.category == 'account_disabled':
            key = 'account_disabled'
        elif error.category in ('invalid_code', 'sign_in_expired'):
            key = 'account_expired'
        elif error.category == 'terms_unavailable':
            key = 'account_terms_unavailable'
        elif error.category in ('network', 'loopback'):
            key = 'account_network_error'
        elif error.category == 'http' and error.status == 429:
            key = 'account_rate_limited'
        elif error.category == 'http' and error.status == 403:
            key = 'account_disabled'
        elif error.category == 'http' and error.status in (400, 401):
            key = 'account_expired'
        else:
            key = 'account_generic_error'
        return trans.t(key)

    def _save_report_preference(self):
        settings = get_settings()
        previous = settings.value('reports/auto_send', False, type=bool)
        enabled = self.auto_reports.isChecked()
        self.window_main.auto_error_reports = enabled
        settings.setValue('reports/auto_send', enabled)
        settings.sync()
        if previous and not enabled:
            from ui.report_retry import discard_automatic_reports
            discard_automatic_reports()

    def reject(self):
        if self.client is not None and self.client.is_signed_in():
            self._save_report_preference()
        self._cancel.set()
        super().reject()


def cancel_account_jobs(window):
    """Cancel pending sign-ins and report whether account workers remain."""
    for dialog in window.findChildren(AccountDialog):
        dialog._cancel.set()
    return bool(_ACTIVE_JOBS)
