"""Optional desktop account access; all account I/O runs off the GUI thread."""
import sys
import threading
import time
from datetime import date

from PyQt6.QtCore import QObject, QSize, QTimer, QUrl, QUrlQuery, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QVBoxLayout, QWidget)

from locales import trans
from utils.account_client import (BASE_URL, AccountClient, AccountError, _sign_in_stage,
                                  log_sign_in_failure)
from utils.design_tokens import palette
from utils.settings_store import enable_account_default_reports
from utils.ui_utils import resource_path

_ACTIVE_JOBS = set()
_SIGN_IN_OPERATIONS = ('email', 'github', 'google', 'browser')
# Older service terms do not disclose the signed-in reporting default.
_REPORT_TERMS_VERSION = date(2026, 11, 11)

def enable_disclosed_reports(window, version):
    try:
        disclosed = date.fromisoformat(version) >= _REPORT_TERMS_VERSION
    except (TypeError, ValueError):
        disclosed = False
    if disclosed:
        enable_account_default_reports(window)



class _AccountJob(QObject):
    started = pyqtSignal()
    finished = pyqtSignal(object, object)

    def __init__(self, call, owner, *, sign_in_method=None, cancel=None):
        super().__init__()
        self.call = call
        self.owner = owner
        self.sign_in_method = sign_in_method
        self.cancel = cancel
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
        stage = _sign_in_stage.set('unknown')
        started_at = time.monotonic()
        try:
            try:
                value = self.call(self.started.emit)
            except AccountError as exc:
                error = exc
            except Exception:
                error = AccountError('unknown')
            else:
                self.finished.emit(value, None)
                return
            if self.sign_in_method is not None and (self.cancel is None or not self.cancel.is_set()):
                log_sign_in_failure(self.sign_in_method, error,
                                    round((time.monotonic() - started_at) * 1000))
            self.finished.emit(None, error)
        finally:
            _sign_in_stage.reset(stage)


class AccountDialog(QDialog):
    """Sign in, manage the local session, or exit the application."""

    def __init__(self, window_main, parent=None, *, required=False):
        startup = (sys.platform == 'win32' and required and parent is None
                   and not window_main.isVisible())
        super().__init__(None if startup else parent or window_main)
        if startup:
            self.setWindowIcon(window_main.windowIcon())
        self.window_main = window_main
        self.required = required
        self.client = getattr(window_main, '_account_client', None)
        self._job = None
        self._operation = None
        self._cancel = threading.Event()
        self.setWindowTitle(trans.t('account_title'))
        self.setObjectName('accountDialog')
        self.setMinimumWidth(500)
        self._signed_in_view = None
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self.adjustSize)
        self._accept_timer = QTimer(self)
        self._accept_timer.setSingleShot(True)
        self._accept_timer.timeout.connect(self.accept)
        colors = palette(getattr(window_main, 'ui_theme', 'dark'))
        check_icon = resource_path('assets', 'check_white.svg').replace('\\', '/')
        self.setStyleSheet(f'''
            QDialog#accountDialog {{ background: {colors['bg']}; }}
            QFrame#accountCard {{ background: {colors['surface']};
                border: 1px solid {colors['border']}; border-radius: 18px; }}
            QLabel#accountHeading {{ color: {colors['text']}; font-size: 19pt;
                font-weight: 700; }}
            QLabel#accountSubtitle, QLabel#accountSeparator {{
                color: {colors['textMuted']}; font-size: 10pt; }}
            QLabel#accountAgreement {{ color: {colors['textMuted']}; }}
            QLabel#accountStatus {{ color: {colors['textMuted']}; }}
            QCheckBox#accountConsent::indicator {{ width: 16px; height: 16px;
                border: 2px solid {colors['textMuted']}; border-radius: 4px;
                background: {colors['controlBg']}; }}
            QCheckBox#accountConsent::indicator:checked {{ background: {colors['accent']};
                border-color: {colors['accent']}; image: url("{check_icon}"); }}
            QPushButton#accountSocial {{ background: {colors['surfaceHover']};
                color: {colors['text']}; border: 0; border-radius: 10px;
                min-height: 36px; font-weight: 600; }}
            QPushButton#accountSocial:hover {{ background: {colors['surfaceActive']}; }}
            QPushButton#accountEmail {{ background: {colors['accent']};
                color: {colors['textOnAccent']}; border: 0; border-radius: 10px;
                min-height: 36px; font-weight: 600; }}
            QPushButton#accountEmail:hover {{ background: {colors['accentHover']}; }}
            QPushButton#accountClose {{ background: transparent; color: {colors['textMuted']};
                border: 0; padding: 4px; }}
            QPushButton#accountClose:hover {{ color: {colors['accent']}; }}
            QPushButton#accountSocial:disabled, QPushButton#accountEmail:disabled {{
                background: {colors['controlDisabled']}; color: {colors['textFaint']}; }}
            QLineEdit#accountEmailInput {{ background: {colors['controlBg']};
                color: {colors['text']}; border: 1px solid {colors['controlBorder']};
                border-radius: 10px; padding: 0 14px; min-height: 36px; }}
            QLineEdit#accountEmailInput:focus {{ border: 1px solid {colors['accent']}; }}
        ''')

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 18, 32, 16)
        root.setSpacing(8)
        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setAccessibleName('OpenFocus')
        ratio = self.devicePixelRatioF()
        logo_pixmap = QPixmap(resource_path('assets', 'OpenFocus.png')).scaled(
            round(52 * ratio), round(52 * ratio), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        logo_pixmap.setDevicePixelRatio(ratio)
        logo.setPixmap(logo_pixmap)
        root.addWidget(logo)
        self.heading = QLabel(trans.t('account_heading'))
        self.heading.setObjectName('accountHeading')
        self.heading.setWordWrap(True)
        self.heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.heading)
        self.subtitle = QLabel(trans.t('account_subtitle'))
        self.subtitle.setObjectName('accountSubtitle')
        self.subtitle.setWordWrap(True)
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.subtitle)
        root.addSpacing(4)

        card = QFrame()
        card.setObjectName('accountCard')
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 16, 24, 16)
        card_layout.setSpacing(10)
        self.form = QWidget()
        form = QVBoxLayout(self.form)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        self.browser_button = QPushButton(trans.t('account_browser_login'))
        self.browser_button.setObjectName('accountSocial')
        self.browser_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browser_button.clicked.connect(self._sign_in_browser)
        form.addWidget(self.browser_button)
        self.google_button = QPushButton('  ' + trans.t('account_google'))
        self.github_button = QPushButton('  ' + trans.t('account_github'))
        for button, icon in ((self.google_button, 'google.svg'),
                             (self.github_button, 'github.svg')):
            button.setObjectName('accountSocial')
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            pixmap = QPixmap(resource_path('assets', icon))
            if icon == 'github.svg' and not pixmap.isNull():
                painter = QPainter(pixmap)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                painter.fillRect(pixmap.rect(), QColor(colors['text']))
                painter.end()
            button.setIcon(QIcon(pixmap))
            button.setIconSize(QSize(20, 20))
            form.addWidget(button)
        self.google_button.clicked.connect(lambda: self._sign_in('google'))
        self.github_button.clicked.connect(lambda: self._sign_in('github'))

        separator = QLabel(trans.t('account_or'))
        separator.setObjectName('accountSeparator')
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        form.addWidget(separator)
        self.email = QLineEdit()
        self.email.setObjectName('accountEmailInput')
        self.email.setPlaceholderText(trans.t('account_email'))
        self.email.setClearButtonEnabled(True)
        form.addWidget(self.email)
        self.email_button = QPushButton(trans.t('account_email_send'))
        self.email_button.setObjectName('accountEmail')
        self.email_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.email_button.clicked.connect(lambda: self._sign_in('email'))
        form.addWidget(self.email_button)

        agreement_row = QHBoxLayout()
        agreement_row.setSpacing(8)
        self.terms = QCheckBox()
        self.terms.setObjectName('accountConsent')
        self.terms.setAccessibleName(trans.t('account_terms'))
        self.terms.toggled.connect(self._terms_changed)
        agreement_row.addWidget(self.terms, 0, Qt.AlignmentFlag.AlignTop)
        agreement_text = trans.t('account_agreement').replace(
            '<a ', f'<a style="color:{colors["accent"]};" ')
        self.agreement = QLabel(agreement_text)
        self.agreement.setObjectName('accountAgreement')
        self.agreement.setWordWrap(True)
        self.agreement.setTextFormat(Qt.TextFormat.RichText)
        self.agreement.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse |
                                               Qt.TextInteractionFlag.LinksAccessibleByKeyboard)
        self.agreement.linkActivated.connect(self._open_agreement)
        agreement_row.addWidget(self.agreement, 1)
        form.addLayout(agreement_row)
        card_layout.addWidget(self.form)

        self.status = QLabel(trans.t('account_checking'))
        self.status.setObjectName('accountStatus')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        card_layout.addWidget(self.status)
        self.devices_button = QPushButton(trans.t('account_web_center'))
        self.devices_button.setObjectName('accountManageDevices')
        self.devices_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.devices_button.clicked.connect(self._open_account_devices)
        card_layout.addWidget(self.devices_button)
        self.retry_button = QPushButton(trans.t('btn_retry'))
        self.retry_button.setObjectName('accountRetry')
        self.retry_button.clicked.connect(self._retry_account_load)
        card_layout.addWidget(self.retry_button)
        account_actions = QHBoxLayout()
        self.logout_button = QPushButton(trans.t('account_sign_out'))
        self.center_button = QPushButton(trans.t('account_web_center'))
        self.center_button.setObjectName('accountWebCenter')
        self.center_button.clicked.connect(self._open_account_center)
        self.logout_button.clicked.connect(self._logout)
        account_actions.addWidget(self.center_button)
        account_actions.addWidget(self.logout_button)
        card_layout.addLayout(account_actions)
        root.addWidget(card)
        self.close_button = QPushButton(trans.t('action_exit'))
        self.close_button.setObjectName('accountClose')
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        close_font = self.close_button.font()
        close_font.setUnderline(True)
        self.close_button.setFont(close_font)
        self.close_button.clicked.connect(self._close_dialog)
        root.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignHCenter)
        self._refresh_controls(busy=True)
        if self.client is None:
            shared_load = getattr(window_main, '_account_load_job', None)
            if shared_load is None:
                self._start('load', lambda _started: AccountClient())
            else:
                self._operation = 'load'
                self._job = shared_load
                window_main.account_session_loaded.connect(self._finished)
        else:
            self._refresh_controls()
            if self.required and self.client.is_signed_in():
                self._accept_timer.start(0)

    def _retry_account_load(self):
        if self._job is None and self.client is None:
            self._cancel = threading.Event()
            self.status.setText(trans.t('account_checking'))
            self.status.show()
            self._start('load', lambda _started: AccountClient())

    def _close_dialog(self):
        if self.client is not None and self.client.is_signed_in():
            self.accept()
        else:
            self.reject()

    def _open_account_center(self):
        if self._job is None and self.client is not None:
            self._start('web_center', lambda _started: self.client.open_web_handoff(
                cancel=self._cancel, timeout=8))

    def _open_account_devices(self):
        url = QUrl(BASE_URL + '/account/devices')
        query = QUrlQuery()
        query.addQueryItem('lang', 'zh' if trans.current_lang == 'zh' else 'en')
        url.setQuery(query)
        QDesktopServices.openUrl(url)


    def _sign_in_browser(self):
        cancel = self._begin_sign_in()
        if cancel is None:
            return
        client = self.client
        language = 'zh' if trans.current_lang == 'zh' else 'en'
        self._start('browser', lambda started: client.login_with_browser(
            cancel=cancel, on_started=started, language=language))

    def _open_agreement(self, link):
        if link == 'terms':
            url = QUrl('https://account.openfocus.top/terms')
            query = QUrlQuery()
            query.addQueryItem('lang', 'zh' if trans.current_lang == 'zh' else 'en')
            url.setQuery(query)
            QDesktopServices.openUrl(url)
        elif link == 'privacy':
            url = ('https://openfocus.top/zh/#privacy' if trans.current_lang == 'zh'
                   else 'https://openfocus.top/?lang=en#privacy')
            QDesktopServices.openUrl(QUrl(url))

    def _refresh_controls(self, busy=False):
        user = self.client.current_user() if self.client is not None else None
        signed_in = self.client is not None and self.client.is_signed_in()
        pending = self.client is not None and getattr(self.client, 'logout_cleanup_pending', False)
        self.form.setVisible(not signed_in and not pending)
        switching = self._operation in _SIGN_IN_OPERATIONS
        self.form.setEnabled((not busy or switching) and self.client is not None and not pending)
        self.retry_button.setVisible(not busy and self.client is None)
        self.devices_button.hide()
        self.logout_button.setVisible(signed_in or pending)
        self.logout_button.setText(trans.t('account_cleanup_retry' if pending else 'account_sign_out'))
        self.center_button.setVisible(signed_in)
        self.logout_button.setEnabled(not busy)
        self.center_button.setEnabled(not busy)
        self.close_button.setText(trans.t('account_continue' if signed_in else 'action_exit'))
        self.heading.setText(trans.t('account_title' if signed_in else 'account_heading'))
        self.subtitle.setText(trans.t('account_manage_subtitle' if signed_in else 'account_subtitle'))
        if signed_in:
            email = (user.get('email') or user.get('name') or trans.t('account_title')) if isinstance(user, dict) else trans.t('account_title')
            self.status.setText(trans.t('account_signed_in').format(email=email))
            self.status.show()
        elif pending:
            self.status.setText(trans.t('account_cleanup_pending'))
            self.status.show()
        elif not busy:
            self.status.clear()
            self.status.hide()
        if self._signed_in_view != signed_in:
            self._signed_in_view = signed_in
            self.layout().invalidate()
            self._layout_timer.start(0)
        refresh_menu = getattr(self.window_main, '_refresh_account_action', None)
        if refresh_menu is not None:
            refresh_menu()

    def _start(self, operation, call):
        self._operation = operation
        method = operation if operation in _SIGN_IN_OPERATIONS else None
        if operation == 'load':
            method = 'restore'
        job = _AccountJob(call, self.window_main, sign_in_method=method, cancel=self._cancel)
        self._job = job
        job.started.connect(self._started)
        job.finished.connect(self._finished)
        job.finished.connect(job._release)
        self._refresh_controls(busy=True)
        job.start()

    def _terms_changed(self, checked):
        if not checked and self._operation in ('email', 'github', 'google'):
            self._cancel.set()
            self.status.setText(trans.t('account_terms_required'))
            self.status.show()

    def _begin_sign_in(self):
        if self.client is None or (self._job is not None and
                self._operation not in _SIGN_IN_OPERATIONS):
            return None
        self._cancel.set()
        self._cancel = threading.Event()
        return self._cancel

    def _started(self):
        if (self.sender() is not None and self.sender() is not self._job) or self._cancel.is_set():
            return
        key = 'account_wait_email' if self._operation == 'email' else 'account_wait_browser'
        self.status.setText(trans.t(key) + '\n' + trans.t('account_switch_hint'))
        self.status.show()

    def _sign_in(self, provider):
        if self.client is None or (self._job is not None and
                self._operation not in _SIGN_IN_OPERATIONS):
            return
        if provider == 'email' and self._operation == 'email' and not self._cancel.is_set():
            return
        if not self.terms.isChecked():
            self.status.setText(trans.t('account_terms_required'))
            self.status.show()
            return
        email = self.email.text().strip()
        if provider == 'email' and not email:
            self.status.setText(trans.t('account_email_required'))
            self.status.show()
            return
        cancel = self._begin_sign_in()
        if cancel is None:
            return
        client = self.client
        language = 'zh' if trans.current_lang == 'zh' else 'en'
        def login(started):
            _sign_in_stage.set('terms')
            version = client.terms()['version']
            if cancel.is_set():
                return None
            _sign_in_stage.set('unknown')
            user = client.login(provider, email=email if provider == 'email' else None,
                                cancel=cancel, on_started=started,
                                terms_version=version, language=language)
            return user, version
        self.status.setText(trans.t('account_checking'))
        self.status.show()
        self._start(provider, login)

    def _logout(self):
        if self._job is None and self.client is not None:
            self._start('logout', lambda _started: self.client.logout())

    def _finished(self, result, error):
        sender = self.sender()
        if (sender is not None and sender is not self._job and not
                (sender is self.window_main and self._operation == 'load')):
            return
        operation = self._operation
        self._job = None
        self._operation = None
        if self._cancel.is_set():
            if self.isVisible() and self.client is not None and self.client.is_signed_in():
                self._refresh_controls()
            return
        if error is not None and error.category in ('terms_required', 'invalid_terms_version'):
            self.terms.setChecked(False)
        if error is not None:
            if ((operation == 'web_center' and error.category == 'timeout')
                    or (operation == 'browser' and error.category == 'sign_in_cancelled')):
                self._refresh_controls()
                return
            self._refresh_controls()
            pending = getattr(self.client, 'logout_cleanup_pending', False)
            self.status.setText(trans.t('account_cleanup_pending') if pending else self._error_text(error))
            self.status.show()
            if error.category == 'device_limit_reached' and not pending:
                self.devices_button.show()
                self._layout_timer.start(0)
            return
        report_cleanup_failed = False
        if operation == 'load':
            self.client = result
            self.window_main._account_client = result
        elif operation in _SIGN_IN_OPERATIONS:
            version = (self.client.accepted_terms_version() if operation == 'browser'
                       else result[1])
            try:
                enable_disclosed_reports(self.window_main, version)
            except Exception:
                report_cleanup_failed = True
            else:
                self._retry_queued_reports()
        self._refresh_controls()
        if (self.required and not report_cleanup_failed and self.client is not None
                and self.client.is_signed_in() and (operation == 'load' or operation in _SIGN_IN_OPERATIONS)):
            self.accept()
            return
        if report_cleanup_failed:
            self.status.setText(trans.t('report_cleanup_failed'))
            self.status.show()
        if operation == 'web_center' and not result:
            self.status.setText(trans.t('account_generic_error'))
            self.status.show()
        if operation == 'logout':
            self.status.setText(trans.t('account_signed_out'))
            self.status.show()

    def _retry_queued_reports(self):
        from utils.app_paths import user_data_dir
        from utils.report_client import ReportClient, ReportError
        from utils.reporting import build_channel

        if build_channel() != 'release' or not self.client.is_signed_in():
            return
        directory = user_data_dir() / 'reports'
        try:
            pending = directory.is_dir() and ReportClient(directory).pending_count
        except (OSError, ReportError):
            return
        if pending:
            from ui.report_retry import schedule_report_retries
            schedule_report_retries(self.window_main, urgent=True)

    @staticmethod
    def _error_text(error):
        if error.category == 'not_signed_in':
            return trans.t('account_browser_not_signed_in')
        if error.category == 'email_verification_unavailable':
            return trans.t('account_generic_error') + '\n' + trans.t('account_switch_hint')
        if error.category in ('terms_required', 'invalid_terms_version'):
            key = 'account_terms_required'
        elif error.category == 'verified_email_required':
            key = 'account_email_unverified'
        elif error.category == 'account_disabled':
            key = 'account_disabled'
        elif error.category == 'device_limit_reached':
            key = 'account_device_limit'
        elif error.category in ('invalid_code', 'sign_in_expired'):
            key = 'account_expired'
        elif error.category == 'terms_unavailable':
            key = 'account_terms_unavailable'
        elif error.category in ('network', 'loopback'):
            key = 'account_network_error'
        elif error.category == 'rate_limited' or (error.category == 'http' and error.status == 429):
            key = 'account_rate_limited'
        elif error.category == 'http' and error.status == 403:
            key = 'account_disabled'
        elif error.category == 'http' and error.status in (400, 401):
            key = 'account_expired'
        else:
            key = 'account_generic_error'
        return trans.t(key)


    def reject(self):
        cancel = getattr(self, '_cancel', None)
        if cancel is not None and getattr(self, '_operation', None) != 'web_center':
            cancel.set()
        super().reject()


def cancel_account_jobs(window):
    """Cancel pending sign-ins and report whether account workers remain."""
    for dialog in window.findChildren(AccountDialog):
        cancel = getattr(dialog, '_cancel', None)
        if cancel is not None:
            cancel.set()
    return any(job.owner is window for job in _ACTIVE_JOBS)
