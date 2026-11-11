"""Optional desktop account access; all account I/O runs off the GUI thread."""
import threading
from datetime import date

from PyQt6.QtCore import QObject, QSize, QUrl, QUrlQuery, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
                             QWidget)

from locales import trans
from utils.account_client import AccountClient, AccountError
from utils.design_tokens import palette
from utils.settings_store import enable_account_default_reports
from utils.ui_utils import resource_path

_ACTIVE_JOBS = set()
# Older service terms do not disclose the signed-in reporting default.
_REPORT_TERMS_VERSION = date(2026, 11, 11)


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
        self.setWindowTitle(trans.t('account_title'))
        self.setObjectName('accountDialog')
        self.setMinimumWidth(500)
        self.resize(570, 520)
        colors = palette(getattr(window_main, 'ui_theme', 'dark'))
        check_icon = resource_path('assets', 'check_white.svg').replace('\\', '/')
        self.setStyleSheet(f'''
            QDialog#accountDialog {{ background: {colors['bg']}; }}
            QFrame#accountCard {{ background: {colors['surface']};
                border: 1px solid {colors['border']}; border-radius: 18px; }}
            QLabel#accountHeading {{ color: {colors['text']}; font-size: 22pt;
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
                min-height: 44px; font-weight: 600; }}
            QPushButton#accountSocial:hover {{ background: {colors['surfaceActive']}; }}
            QPushButton#accountEmail {{ background: {colors['accent']};
                color: {colors['textOnAccent']}; border: 0; border-radius: 10px;
                min-height: 44px; font-weight: 600; }}
            QPushButton#accountEmail:hover {{ background: {colors['accentHover']}; }}
            QPushButton#accountSkip {{ background: transparent; color: {colors['textMuted']};
                border: 0; padding: 4px; }}
            QPushButton#accountSkip:hover {{ color: {colors['accent']}; }}
            QPushButton#accountSocial:disabled, QPushButton#accountEmail:disabled {{
                background: {colors['controlDisabled']}; color: {colors['textFaint']}; }}
            QLineEdit#accountEmailInput {{ background: {colors['controlBg']};
                color: {colors['text']}; border: 1px solid {colors['controlBorder']};
                border-radius: 10px; padding: 0 14px; min-height: 44px; }}
            QLineEdit#accountEmailInput:focus {{ border: 1px solid {colors['accent']}; }}
        ''')

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(12)
        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setAccessibleName('OpenFocus')
        ratio = self.devicePixelRatioF()
        logo_pixmap = QPixmap(resource_path('assets', 'OpenFocus.png')).scaled(
            round(68 * ratio), round(68 * ratio), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        logo_pixmap.setDevicePixelRatio(ratio)
        logo.setPixmap(logo_pixmap)
        root.addWidget(logo)
        heading = QLabel(trans.t('account_heading'))
        heading.setObjectName('accountHeading')
        heading.setWordWrap(True)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(heading)
        subtitle = QLabel(trans.t('account_subtitle'))
        subtitle.setObjectName('accountSubtitle')
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)
        root.addSpacing(8)

        card = QFrame()
        card.setObjectName('accountCard')
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 26, 28, 26)
        card_layout.setSpacing(14)
        self.form = QWidget()
        form = QVBoxLayout(self.form)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(12)
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
        account_actions = QHBoxLayout()
        self.logout_button = QPushButton(trans.t('account_sign_out'))
        self.delete_button = QPushButton(trans.t('account_delete'))
        self.logout_button.clicked.connect(self._logout)
        self.delete_button.clicked.connect(self._delete_account)
        account_actions.addWidget(self.logout_button)
        account_actions.addWidget(self.delete_button)
        card_layout.addLayout(account_actions)
        root.addWidget(card)
        self.skip_button = QPushButton(trans.t('account_skip'))
        self.skip_button.setObjectName('accountSkip')
        self.skip_button.setCursor(Qt.CursorShape.PointingHandCursor)
        skip_font = self.skip_button.font()
        skip_font.setUnderline(True)
        self.skip_button.setFont(skip_font)
        self.skip_button.clicked.connect(self.reject)
        root.addWidget(self.skip_button, 0, Qt.AlignmentFlag.AlignHCenter)
        root.addStretch()
        self._refresh_controls(busy=True)
        if self.client is None:
            self._start('load', lambda _started: AccountClient())
        else:
            self._refresh_controls()

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
        self.form.setVisible(not signed_in)
        self.form.setEnabled(not busy and self.client is not None)
        self.logout_button.setVisible(signed_in)
        self.delete_button.setVisible(signed_in)
        self.logout_button.setEnabled(not busy)
        self.delete_button.setEnabled(not busy)
        self.skip_button.setText(trans.t('account_continue' if signed_in else 'account_skip'))
        if signed_in:
            email = user.get('email', '') if isinstance(user, dict) else ''
            self.status.setText(trans.t('account_signed_in').format(email=email))
            self.status.show()
        elif not busy:
            self.status.clear()
            self.status.hide()

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
        if self._operation == 'email':
            key = ('account_wait_email_phone'
                   if getattr(self.client, 'email_handoff_mode', None) == 'poll-v1'
                   else 'account_wait_email')
        else:
            key = 'account_wait_browser'
        self.status.setText(trans.t(key))

    def _sign_in(self, provider):
        if self._job is not None or self.client is None:
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
        self._cancel.clear()
        client = self.client
        language = 'zh' if trans.current_lang == 'zh' else 'en'
        def login(started):
            terms = client.terms()
            if self._cancel.is_set():
                return None
            user = client.login(provider, email=email if provider == 'email' else None,
                                cancel=self._cancel, on_started=started,
                                terms_version=terms['version'], language=language)
            return user, terms['version']
        self.status.setText(trans.t('account_checking'))
        self.status.show()
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
            self.terms.setChecked(False)
        if error is not None:
            self._refresh_controls()
            self.status.setText(self._error_text(error))
            self.status.show()
            return
        if operation == 'load':
            self.client = result
            self.window_main._account_client = result
        elif operation in ('email', 'github', 'google'):
            try:
                disclosed = date.fromisoformat(result[1]) >= _REPORT_TERMS_VERSION
            except ValueError:
                disclosed = False
            if disclosed:
                enable_account_default_reports(self.window_main)
        self._refresh_controls()
        if operation == 'logout':
            self.status.setText(trans.t('account_signed_out'))
            self.status.show()
        elif operation == 'delete':
            self.status.setText(trans.t('account_deleted'))
            self.status.show()

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


    def reject(self):
        self._cancel.set()
        super().reject()


def cancel_account_jobs(window):
    """Cancel pending sign-ins and report whether account workers remain."""
    for dialog in window.findChildren(AccountDialog):
        dialog._cancel.set()
    return bool(_ACTIVE_JOBS)
