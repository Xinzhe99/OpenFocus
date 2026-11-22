"""Account and report interactions remain optional and reviewable."""
import json
import time
from threading import Event

import pytest
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QDialog, QPushButton, QWidget


def _wait(qapp, predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline
        QTest.qWait(5)
    qapp.processEvents()


@pytest.fixture
def settings(monkeypatch, tmp_path):
    from utils import settings_store
    value = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, '_settings_singleton', value)
    monkeypatch.setenv('OPENFOCUS_DATA_DIR', str(tmp_path / 'profile'))
    return value


class _Account:
    def __init__(self):
        self.user = None
        self.calls = []

    def current_user(self):
        return self.user

    def is_signed_in(self):
        return self.user is not None

    def terms(self):
        self.calls.append('terms')
        return {'version': '2026-11-11', 'url': 'https://account.openfocus.top/terms',
                'draft': True, 'required': True}

    def login(self, provider, **kwargs):
        self.calls.append((provider, kwargs['terms_version'], kwargs['language']))
        self.user = {'id': 'owner', 'email': 'user@example.test'}
        return self.user

    def open_web_handoff(self, **kwargs):
        self.calls.append('web_handoff')
        return True

    def login_with_browser(self, **kwargs):
        self.calls.append('browser')
        self.user = {'id': 'owner', 'email': 'browser@example.test'}
        return self.user

    def accepted_terms_version(self):
        return '2026-11-11'

    def logout(self):
        self.calls.append('logout')
        self.user = None


@pytest.mark.parametrize('operation', ['email', 'github', 'google', 'browser'])
def test_sign_in_reports_cleanup_failure_without_losing_session(qapp, settings, monkeypatch, operation):
    from dialogs import account as account_dialog
    from locales import trans
    from utils.report_client import ReportError

    def fail_cleanup(window, version):
        raise ReportError('private storage detail')
    monkeypatch.setattr(account_dialog, 'enable_disclosed_reports', fail_cleanup)
    window = QWidget()
    window._account_client = _Account()
    window._account_client.user = {'id': 'owner'}
    dialog = account_dialog.AccountDialog(window)
    try:
        dialog._operation = operation
        dialog._finished((window._account_client.user, '2026-11-11'), None)
        assert window._account_client.is_signed_in()
        assert dialog.status.text() == trans.t('report_cleanup_failed')
        assert 'private storage detail' not in dialog.status.text()
    finally:
        dialog.reject()
        window.deleteLater()


def test_account_menu_and_dialog_follow_sign_in_and_sign_out(qapp, tmp_path, monkeypatch):
    from dialogs.account import AccountDialog
    from locales import trans
    from tests.test_control_channel import _destroy_main_window, _new_main_window

    window = _new_main_window(qapp, tmp_path, monkeypatch)
    _wait(qapp, lambda: getattr(window, '_account_load_job', None) is None)
    client = window._account_client = _Account()
    dialog = AccountDialog(window)
    try:
        window._refresh_account_action()
        action = window.ui_objs['action_account']
        assert action.text() == trans.t('account_heading')
        dialog.show()
        qapp.processEvents()
        signed_out_height = dialog.height()
        dialog.terms.setChecked(True)
        dialog.github_button.click()
        _wait(qapp, lambda: dialog._job is None)
        qapp.processEvents()
        assert 'user@example.test' in action.toolTip()
        assert dialog.center_button.isVisible() and dialog.logout_button.isVisible()
        assert not dialog.form.isVisible()
        assert dialog.height() < signed_out_height - 50
        assert dialog.height() - dialog.close_button.geometry().bottom() < 70
        dialog.center_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert client.calls[-1] == 'web_handoff'
        assert not hasattr(dialog, 'delete_button')
        dialog.logout_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert client.calls[-1] == 'logout'
        assert action.text() == trans.t('account_heading')
        assert dialog.form.isVisible() and not dialog.center_button.isVisible()
    finally:
        dialog.reject()
        _destroy_main_window(qapp, window)


def test_logout_storage_failure_shows_retry(qapp, settings):
    from dialogs.account import AccountDialog
    from locales import trans
    from utils.account_client import AccountClient, AccountError

    class Store:
        value = {'token': 'secret', 'token_type': 'Bearer', 'expires_at': 9999999999,
                 'user': {'id': 'owner'}}
        fail = True
        def load(self):
            return self.value
        def clear(self):
            if self.fail:
                raise AccountError('storage')
            self.value = None

    calls = []
    def request(method, url, body=None, token=None, timeout=None):
        calls.append((url, token))
        return {'ok': True}
    store = Store()
    window = QWidget()
    window._account_client = AccountClient(store=store, request=request)
    dialog = AccountDialog(window)
    try:
        dialog.show()
        dialog.logout_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert window._account_client.logout_cleanup_pending
        assert dialog.status.text() == trans.t('account_cleanup_pending')
        assert dialog.logout_button.isVisible() and not dialog.form.isVisible()
        assert len(calls) == 1 and calls[0][1] == 'secret'
        store.fail = False
        dialog.logout_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert not window._account_client.logout_cleanup_pending
        assert dialog.status.text() == trans.t('account_signed_out')
        assert store.value is None and len(calls) == 1
    finally:
        dialog.reject()
        window.deleteLater()


def test_browser_sign_in_requires_user_action_and_logout_stays_signed_out(qapp, tmp_path, monkeypatch):
    from dialogs.account import AccountDialog
    from tests.test_control_channel import _destroy_main_window, _new_main_window

    window = _new_main_window(qapp, tmp_path, monkeypatch)
    _wait(qapp, lambda: getattr(window, '_account_load_job', None) is None)
    client = window._account_client = _Account()
    dialog = AccountDialog(window)
    try:
        qapp.processEvents()
        assert client.calls == []
        dialog.browser_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert client.calls == ['browser']
        assert 'browser@example.test' in window.ui_objs['action_account'].toolTip()
        dialog.logout_button.click()
        _wait(qapp, lambda: dialog._job is None)
        qapp.processEvents()
        assert client.calls == ['browser', 'logout']
        assert not client.is_signed_in()
    finally:
        dialog.reject()
        _destroy_main_window(qapp, window)


def test_closing_account_dialog_does_not_cancel_opening_web_center(qapp, settings):
    from dialogs.account import AccountDialog

    started, release = Event(), Event()
    class Delayed(_Account):
        def open_web_handoff(self, *, cancel, **kwargs):
            started.set()
            assert release.wait(5)
            return not cancel.is_set()
    window = QWidget()
    client = window._account_client = Delayed()
    client.user = {'id': 'owner', 'email': 'user@example.test'}
    dialog = AccountDialog(window)
    try:
        dialog.center_button.click()
        assert started.wait(5)
        dialog.reject()
        release.set()
        _wait(qapp, lambda: dialog._job is None)
        assert not dialog._cancel.is_set()
    finally:
        release.set()
        window.deleteLater()


@pytest.mark.parametrize('version', ['2026-11-08', None])
def test_browser_sign_in_with_old_terms_does_not_enable_reports(qapp, tmp_path, monkeypatch, version):
    from dialogs.account import AccountDialog
    from tests.test_control_channel import _destroy_main_window, _new_main_window
    from utils.settings_store import get_settings

    window = _new_main_window(qapp, tmp_path, monkeypatch)
    _wait(qapp, lambda: getattr(window, '_account_load_job', None) is None)
    client = window._account_client = _Account()
    client.accepted_terms_version = lambda: version
    dialog = AccountDialog(window)
    try:
        dialog.browser_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert client.is_signed_in()
        assert not get_settings().value('reports/auto_send', False, type=bool)
    finally:
        dialog.reject()
        _destroy_main_window(qapp, window)


def test_account_dialog_shares_pending_session_restore(qapp, tmp_path, monkeypatch):
    from dialogs.account import AccountDialog
    from tests.test_control_channel import _destroy_main_window, _new_main_window
    import utils.account_client as account_module

    entered, release = Event(), Event()
    client = _Account()
    client.user = {'id': 'owner', 'email': 'saved@example.test'}
    def load():
        entered.set()
        assert release.wait(5)
        return client
    monkeypatch.setattr(account_module, 'AccountClient', load)
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    dialog = None
    try:
        assert entered.wait(5)
        shared = window._account_load_job
        release.set()
        shared._thread.join(5)
        assert not shared._thread.is_alive()
        dialog = AccountDialog(window)
        assert dialog._job is shared
        _wait(qapp, lambda: dialog._job is None)
        assert dialog.client is window._account_client is client
        assert 'saved@example.test' in window.ui_objs['action_account'].toolTip()
    finally:
        release.set()
        if dialog is not None:
            dialog.reject()
        _destroy_main_window(qapp, window)


def test_account_requires_consent_without_opening_links_and_defaults_reporting(qapp, settings, monkeypatch):
    from dialogs.account import AccountDialog
    from PyQt6.QtGui import QDesktopServices
    opened = []
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()) or True)
    window = QWidget()
    client = window._account_client = _Account()
    dialog = AccountDialog(window)
    try:
        assert not dialog.terms.isChecked()
        dialog.github_button.click()
        assert client.calls == []
        dialog.terms.setChecked(True)
        dialog.github_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert client.calls[0] == 'terms'
        assert client.calls[1][0:2] == ('github', '2026-11-11')
        assert not opened
        assert settings.value('reports/auto_send', type=bool)
        assert window.auto_error_reports
    finally:
        dialog.reject()
        window.deleteLater()


def test_agreement_links_are_optional_and_open_independently(qapp, settings, monkeypatch):
    from dialogs.account import AccountDialog
    from PyQt6.QtGui import QDesktopServices
    opened = []
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()) or True)
    window = QWidget()
    client = window._account_client = _Account()
    dialog = AccountDialog(window)
    try:
        dialog.agreement.linkActivated.emit('terms')
        dialog.agreement.linkActivated.emit('privacy')
        assert opened[0].startswith('https://account.openfocus.top/terms?')
        assert opened[1].startswith('https://openfocus.top/')
        assert not client.calls and not dialog.terms.isChecked()
    finally:
        dialog.reject()
        window.deleteLater()


def test_sign_in_preserves_existing_reporting_opt_out(qapp, settings):
    from dialogs.account import AccountDialog
    settings.setValue('reports/auto_send', False)
    window = QWidget()
    window.auto_error_reports = False
    window._account_client = _Account()
    dialog = AccountDialog(window)
    try:
        dialog.terms.setChecked(True)
        dialog.github_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert not settings.value('reports/auto_send', type=bool)
        assert not window.auto_error_reports
    finally:
        dialog.reject()
        window.deleteLater()


@pytest.mark.parametrize('version', ['2026-11-08', 'unversioned'])
def test_old_or_unknown_terms_do_not_enable_automatic_reports(qapp, settings, version):
    from dialogs.account import AccountDialog

    class PreviousTerms(_Account):
        def terms(self):
            return {'version': version, 'url': 'https://account.openfocus.top/terms',
                    'draft': True, 'required': False}

    window = QWidget()
    window._account_client = PreviousTerms()
    dialog = AccountDialog(window)
    try:
        dialog.terms.setChecked(True)
        dialog.github_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert window._account_client.is_signed_in()
        assert not settings.value('reports/auto_send', False, type=bool)
    finally:
        dialog.reject()
        window.deleteLater()


@pytest.mark.parametrize(('mode', 'message'), [
    ('poll-v1', 'account_wait_email_phone'),
    ('loopback', 'account_wait_email'),
])
def test_email_status_matches_available_verification_mode(qapp, settings, mode, message):
    from dialogs.account import AccountDialog
    from locales import trans
    release = Event()

    class Waiting(_Account):
        def login(self, provider, **kwargs):
            self.email_handoff_mode = mode
            kwargs['on_started']()
            assert release.wait(5)
            return super().login(provider, **kwargs)

    window = QWidget()
    window._account_client = Waiting()
    dialog = AccountDialog(window)
    try:
        dialog.terms.setChecked(True)
        dialog.email.setText('user@example.test')
        dialog.email_button.click()
        _wait(qapp, lambda: dialog.status.text().startswith(trans.t(message)))
        assert trans.t('account_switch_hint') in dialog.status.text()
    finally:
        release.set()
        _wait(qapp, lambda: dialog._job is None)
        dialog.reject()
        window.deleteLater()


@pytest.mark.parametrize(('first', 'second'), [
    ('github', 'google'), ('github', 'email'), ('github', 'browser'),
    ('github', 'github'), ('email', 'google'), ('browser', 'google'),
])
def test_pending_sign_in_can_switch_methods(qapp, settings, first, second):
    from dialogs.account import AccountDialog, _ACTIVE_JOBS
    from utils.account_client import AccountError

    entered = [Event(), Event()]
    release = [Event(), Event()]
    attempts = []

    class Waiting(_Account):
        def attempt(self, method, **kwargs):
            index = len(attempts)
            attempts.append((method, kwargs['cancel']))
            kwargs['on_started']()
            entered[index].set()
            assert release[index].wait(5)
            if kwargs['cancel'].is_set():
                raise AccountError('cancelled')
            self.user = {'id': method}
            return self.user

        def login(self, provider, **kwargs):
            return self.attempt(provider, **kwargs)

        def login_with_browser(self, **kwargs):
            return self.attempt('browser', **kwargs)

    window = QWidget()
    client = window._account_client = Waiting()
    dialog = AccountDialog(window)
    dialog.show()
    buttons = {'github': dialog.github_button, 'google': dialog.google_button,
               'email': dialog.email_button, 'browser': dialog.browser_button}
    try:
        dialog.terms.setChecked(True)
        dialog.email.setText('user@example.test')
        buttons[first].click()
        _wait(qapp, entered[0].is_set)
        first_job = dialog._job
        assert all(button.isEnabled() for button in buttons.values())
        buttons[second].click()
        _wait(qapp, entered[1].is_set)
        second_job = dialog._job
        assert first_job is not second_job
        assert attempts[0][1].is_set() and not attempts[1][1].is_set()
        release[0].set()
        _wait(qapp, lambda: first_job not in _ACTIVE_JOBS)
        assert dialog._job is second_job
        assert all(button.isEnabled() for button in buttons.values())
        release[1].set()
        _wait(qapp, lambda: dialog._job is None)
        assert client.current_user()['id'] == second
    finally:
        for event in release:
            event.set()
        dialog.reject()
        window.deleteLater()


def test_withdrawing_consent_cancels_pending_sign_in(qapp, settings):
    from dialogs.account import AccountDialog
    from locales import trans
    from utils.account_client import AccountError
    entered, release = Event(), Event()
    cancelled = []

    class Waiting(_Account):
        def login(self, provider, **kwargs):
            cancelled.append(kwargs['cancel'])
            kwargs['on_started']()
            entered.set()
            assert release.wait(5)
            if kwargs['cancel'].is_set():
                raise AccountError('cancelled')
            return super().login(provider, **kwargs)

    window = QWidget()
    client = window._account_client = Waiting()
    dialog = AccountDialog(window)
    try:
        dialog.terms.setChecked(True)
        dialog.github_button.click()
        _wait(qapp, entered.is_set)
        dialog.terms.setChecked(False)
        assert cancelled[0].is_set()
        assert dialog.status.text() == trans.t('account_terms_required')
        release.set()
        _wait(qapp, lambda: dialog._job is None)
        assert not client.is_signed_in()
        assert dialog.github_button.isEnabled()
    finally:
        release.set()
        dialog.reject()
        window.deleteLater()


@pytest.mark.parametrize('failure', ['browser', 'network', 'timeout', 'provider_error'])
def test_failed_sign_in_can_be_retried(qapp, settings, failure):
    from dialogs.account import AccountDialog
    from utils.account_client import AccountError

    class Retry(_Account):
        def login(self, provider, **kwargs):
            if not getattr(self, 'failed', False):
                self.failed = True
                raise AccountError(failure)
            return super().login(provider, **kwargs)

    window = QWidget()
    client = window._account_client = Retry()
    dialog = AccountDialog(window)
    try:
        dialog.terms.setChecked(True)
        dialog.github_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert dialog.github_button.isEnabled() and dialog.google_button.isEnabled()
        dialog.github_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert client.is_signed_in()
    finally:
        dialog.reject()
        window.deleteLater()


_REPORT_EMAIL = 'offline-sign-in@example.test'
_REPORT_TOKEN = 'synthetic-account-bearer'
_REPORT_MESSAGE = 'private exception detail from terms lookup'


class _ReportingAccount(_Account):
    def __init__(self, *, offline):
        super().__init__()
        self.offline = offline

    def terms(self):
        if self.offline:
            from utils.account_client import AccountError
            error = AccountError('network')
            error.args = (_REPORT_MESSAGE, _REPORT_TOKEN)
            raise error
        return super().terms()

    def token(self, *, expected_owner):
        assert self.is_signed_in() and expected_owner == self.user['id']
        return _REPORT_TOKEN


def _submit_startup_email(qapp, window):
    from dialogs.account import AccountDialog
    dialog = AccountDialog(window, required=True)
    try:
        dialog.show()
        dialog.email.setText(_REPORT_EMAIL)
        dialog.terms.setChecked(True)
        dialog.email_button.click()
        _wait(qapp, lambda: dialog._job is None)
        return dialog.result()
    finally:
        dialog.reject()
        dialog.deleteLater()


@pytest.fixture
def sign_in_reports(qapp, settings, tmp_path, monkeypatch, caplog):
    import logging
    from urllib import request

    from tests.test_report_client import Server
    from ui import report_retry
    from utils import logging_setup, reporting
    from utils.report_client import ReportClient

    monkeypatch.setattr(report_retry, 'build_channel', lambda: 'release')
    monkeypatch.setattr(reporting, 'build_channel', lambda: 'release')
    monkeypatch.setattr(report_retry.diagnostics, 'system_facts', lambda: {})
    # Drive the same retry entry point synchronously without the background timer.
    monkeypatch.setattr(report_retry, 'schedule_report_retries', lambda *_args, **_kwargs: None)
    server = Server()
    monkeypatch.setattr(request, 'build_opener', lambda *_args: server)
    logger = logging_setup.get_logger()
    monkeypatch.setattr(logger, 'propagate', True)
    monkeypatch.setattr(logging_setup, '_session_had_error', False)
    caplog.set_level(logging.INFO, logger=logger.name)

    window = QWidget()
    window._account_client = _ReportingAccount(offline=True)
    before = set(logger.handlers)
    report_retry.start_failure_capture(window)
    handler, = set(logger.handlers) - before
    directory = tmp_path / 'profile' / 'reports'
    try:
        yield window, ReportClient(directory), directory / 'reports.json', server
    finally:
        logger.removeHandler(handler)
        handler.close()
        window.deleteLater()


@pytest.mark.parametrize('unselected', [False, True], ids=['fresh', 'unselected'])
def test_startup_sign_in_diagnostics_retry_anonymously_after_accepting_terms(
        qapp, settings, monkeypatch, sign_in_reports, unselected):
    import io
    import zipfile

    from utils import reporting
    from utils.report_client import ReportClient
    from utils.settings_store import (REPORTS_AUTO_SEND_KEY, REPORTS_CHOICE_KEY,
                                      automatic_reports_allowed)

    if unselected:
        settings.setValue(REPORTS_CHOICE_KEY, 'unselected')
        settings.setValue(REPORTS_AUTO_SEND_KEY, True)
        settings.sync()
    assert not automatic_reports_allowed(settings)
    window, queue, path, server = sign_in_reports
    offline_account = window._account_client
    _submit_startup_email(qapp, window)
    assert not offline_account.is_signed_in()
    entry, = json.loads(path.read_text(encoding='utf-8'))['pending']
    assert entry['sign_in'] is True
    assert entry['owner_id'] is None
    assert entry['metadata']['source'] == 'auto'
    assert entry['metadata']['build_channel'] == 'release'

    now = [1000]
    monkeypatch.setattr(reporting, 'ReportClient', lambda location: ReportClient(location, clock=lambda: now[0]))
    assert reporting.retry_reports(offline_account, include_auto=False) == 0
    assert queue.pending_count == 1
    assert not server.requests

    account = window._account_client = _ReportingAccount(offline=False)
    assert _submit_startup_email(qapp, window) == QDialog.DialogCode.Accepted
    assert account.is_signed_in()
    assert account.calls[1][:2] == ('email', '2026-11-11')
    assert automatic_reports_allowed(settings)
    assert settings.value(REPORTS_CHOICE_KEY) == 'account_default'

    def offline_post(request, timeout):
        server.requests.append((request, timeout))
        raise OSError('offline')

    with monkeypatch.context() as offline:
        offline.setattr(server, 'open', offline_post)
        assert reporting.retry_reports(account, include_auto=True, account_only_auto=True) == 0
    failed_post, = [request for request, _ in server.requests]
    assert failed_post.method == 'POST'
    assert failed_post.get_header('Authorization') is None
    assert queue.pending_count == 1

    assert reporting.retry_reports(account, include_auto=True, account_only_auto=True) == 0
    assert len(server.requests) == 1
    now[0] += 60
    assert reporting.retry_reports(account, include_auto=True, account_only_auto=True) == 1
    assert queue.pending_count == 0
    post, put = [request for request, _ in server.requests[1:]]
    assert (post.method, put.method) == ('POST', 'PUT')
    assert post.get_header('Authorization') is None
    assert put.get_header('Authorization') is None

    with zipfile.ZipFile(io.BytesIO(put.data)) as archive:
        assert set(archive.namelist()) == {'capture.json', 'error.json', 'events.json', 'system.json'}
        contents = {name: archive.read(name).decode('utf-8') for name in archive.namelist()}
    error = json.loads(contents['error.json'])
    assert not error.get('stack_lines') and not error.get('stack')
    event, = json.loads(contents['events.json'])
    assert event['event'] == 'account_sign_in.failed'
    evidence = event['parameters']
    assert set(evidence) <= {'method', 'stage', 'category', 'run_id', 'status', 'elapsed_ms', 'edge_request_id'}
    assert {name: evidence[name] for name in ('method', 'stage', 'category')} == {
        'method': 'email', 'stage': 'terms', 'category': 'network',
    }
    for text in (*contents.values(), post.data.decode('utf-8')):
        for private in (_REPORT_EMAIL, _REPORT_TOKEN, _REPORT_MESSAGE, account.user['email']):
            assert private not in text


def test_explicit_reporting_opt_out_keeps_sign_in_failure_local(qapp, settings, caplog, sign_in_reports):
    from utils.settings_store import REPORTS_AUTO_SEND_KEY, REPORTS_CHOICE_KEY

    settings.setValue(REPORTS_AUTO_SEND_KEY, False)
    settings.setValue(REPORTS_CHOICE_KEY, 'chosen')
    settings.sync()
    window, queue, _, server = sign_in_reports
    _submit_startup_email(qapp, window)
    (record,) = [record for record in caplog.records if getattr(record, 'event', None) == 'account_sign_in.failed']
    prefix, _, payload = record.getMessage().partition(' ')
    assert prefix == 'event=account_sign_in.failed'
    evidence = json.loads(payload)
    assert evidence['category'] == 'network' and evidence['stage'] == 'terms'
    assert _REPORT_EMAIL not in payload and _REPORT_MESSAGE not in payload
    assert queue.pending_count == 0
    assert not server.requests


def test_revoking_automatic_reporting_removes_staged_sign_in_report(qapp, settings, sign_in_reports):
    from ui.report_retry import discard_automatic_reports
    from utils.settings_store import REPORTS_AUTO_SEND_KEY, REPORTS_CHOICE_KEY

    window, queue, path, server = sign_in_reports
    _submit_startup_email(qapp, window)
    assert queue.pending_count == 1
    assert queue.enqueue({'source': 'auto'}, b'ordinary automatic report') == 'queued'
    assert queue.enqueue({'source': 'manual'}, b'manual report') == 'queued'

    settings.setValue(REPORTS_AUTO_SEND_KEY, False)
    settings.setValue(REPORTS_CHOICE_KEY, 'chosen')
    settings.sync()
    discard_automatic_reports()
    _wait(qapp, lambda: queue.pending_count == 1)
    remaining, = json.loads(path.read_text(encoding='utf-8'))['pending']
    assert remaining['metadata']['source'] == 'manual'
    assert not server.requests


@pytest.mark.parametrize('language,operation', [('en', 'github'), ('zh', 'browser')])
def test_device_limit_offers_account_devices_recovery(qapp, settings, monkeypatch, language, operation):
    from dialogs.account import AccountDialog, QDesktopServices
    from locales import trans
    from utils.account_client import AccountError

    class FullAccount(_Account):
        full = True

        def login(self, provider, **kwargs):
            if self.full:
                raise AccountError('device_limit_reached', status=409)
            return super().login(provider, **kwargs)

        def login_with_browser(self, **kwargs):
            if self.full:
                raise AccountError('device_limit_reached', status=409)
            return super().login_with_browser(**kwargs)

    opened = []
    monkeypatch.setattr(trans, 'current_lang', language)
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()) or True)
    window = QWidget()
    client = window._account_client = FullAccount()
    dialog = AccountDialog(window)
    try:
        dialog.show()
        recovery = dialog.findChild(QPushButton, 'accountManageDevices')
        assert recovery is not None and recovery.isHidden()
        dialog.terms.setChecked(True)
        sign_in = getattr(dialog, operation + '_button')
        sign_in.click()
        _wait(qapp, lambda: dialog._job is None)
        assert dialog.status.text() == trans.t('account_device_limit')
        assert recovery.isVisible() and recovery.isEnabled()
        assert recovery.text() == trans.t('account_web_center')
        assert not opened and not client.is_signed_in()

        QTest.mouseClick(recovery, Qt.MouseButton.LeftButton)
        assert opened == [f'https://account.openfocus.top/account/devices?lang={language}']
        assert sign_in.isEnabled()

        client.full = False
        sign_in.click()
        _wait(qapp, lambda: dialog._job is None)
        assert client.is_signed_in() and recovery.isHidden()
    finally:
        dialog.reject()
        window.deleteLater()


def test_switch_to_browser_while_terms_are_loading(qapp, settings):
    from dialogs.account import AccountDialog, _ACTIVE_JOBS
    entered, release = Event(), Event()

    class Delayed(_Account):
        def terms(self):
            entered.set()
            assert release.wait(5)
            return super().terms()

    window = QWidget()
    client = window._account_client = Delayed()
    dialog = AccountDialog(window)
    try:
        dialog.terms.setChecked(True)
        dialog.github_button.click()
        _wait(qapp, entered.is_set)
        first_job = dialog._job
        dialog.browser_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert client.current_user()['email'] == 'browser@example.test'
        release.set()
        _wait(qapp, lambda: first_job not in _ACTIVE_JOBS)
        assert not any(isinstance(call, tuple) and call[0] == 'github' for call in client.calls)
        assert client.current_user()['email'] == 'browser@example.test'
    finally:
        release.set()
        dialog.reject()
        window.deleteLater()


def test_skip_while_sign_in_loads_terms_does_not_start_login(qapp, settings, monkeypatch):
    from dialogs.account import AccountDialog
    from PyQt6.QtGui import QDesktopServices
    entered, release = Event(), Event()
    opened = []
    class Delayed(_Account):
        def terms(self):
            entered.set()
            assert release.wait(5)
            return super().terms()
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url) or True)
    window = QWidget()
    window._account_client = Delayed()
    dialog = AccountDialog(window)
    try:
        dialog.terms.setChecked(True)
        dialog.github_button.click()
        assert entered.wait(5)
        dialog.reject()
        release.set()
        _wait(qapp, lambda: dialog._job is None)
        assert not opened
        assert window._account_client.calls == ['terms']
        assert not settings.contains('reports/auto_send')
    finally:
        release.set()
        window.deleteLater()


def test_manual_report_requires_review_and_send(qapp, settings, monkeypatch):
    import dialogs.report_review as review
    from utils.report_bundle import build_report_bundle
    calls = []
    bundle = build_report_bundle(log_text='2026-11-10 01:02:03,000 INFO event=render.started {"frame_count": 3, "path": "/private/image.tif"}',
                                 crash_text='', facts={})
    monkeypatch.setattr(review, 'build_report_bundle', lambda **kwargs: bundle)
    monkeypatch.setattr(review, 'send_bundle', lambda value, **kwargs: calls.append(value) or 'sent')
    window = QWidget()
    dialog = review.ReportReviewDialog(window)
    try:
        _wait(qapp, lambda: dialog._job is None)
        assert not calls
        assert 'frame_count' in dialog.preview.toPlainText()
        assert '/private/image.tif' not in dialog.preview.toPlainText()
        dialog.send_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert calls == [bundle]
        assert dialog.sent_or_queued and not dialog.send_button.isEnabled()
    finally:
        dialog.reject()
        window.deleteLater()


def test_native_report_preview_matches_sent_bundle(qapp, settings, monkeypatch):
    import io
    import json
    import zipfile
    import dialogs.report_review as review
    calls = []
    monkeypatch.setattr(review, 'send_bundle', lambda value, **kwargs: calls.append(value) or 'sent')
    window = QWidget()
    crash = ('Fatal Python error: Segmentation fault\n'
             'Current thread 0x123 (most recent call first):\n'
             '  File "/private/workers.py", line 42 in run\n')
    dialog = review.ReportReviewDialog(window, crash_text=crash, log_text='')
    try:
        _wait(qapp, lambda: dialog._job is None)
        with zipfile.ZipFile(io.BytesIO(dialog.bundle.data)) as archive:
            error = json.loads(archive.read('error.json'))
        preview = dialog.preview.toPlainText()
        assert error['native_kind'] == 'segmentation_fault'
        assert error['stack'][0]['line'] == 42
        assert json.dumps(error, indent=2, ensure_ascii=False) in preview
        assert '/private/' not in preview and not calls
        dialog.send_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert calls == [dialog.bundle]
    finally:
        dialog.reject()
        window.deleteLater()


def test_previous_native_crash_requires_separate_review(tmp_path, monkeypatch):
    from ui import crash_report
    from utils import diagnostics, logging_setup

    logs = tmp_path / 'logs'
    logs.mkdir()
    previous = logs / 'openfocus-20261110T010101_000001-123.crash.txt'
    previous.write_text('Fatal Python error: Segmentation fault\n', encoding='utf-8')
    previous.with_name(previous.name.replace('.crash.txt', '.log')).write_text(
        'event=render.failed {}', encoding='utf-8')
    current = logs / 'openfocus-20261110T020202_000002-456.crash.txt'
    current.write_text('Fatal Python error: current process', encoding='utf-8')
    monkeypatch.setattr(logging_setup, 'logs_dir', lambda: logs)
    monkeypatch.setattr(logging_setup, '_process_alive', lambda pid: False)
    monkeypatch.setattr(diagnostics, 'crash_log_path', lambda: current)
    monkeypatch.setattr(diagnostics, 'fallback_crash_dir', lambda: None)
    pending = crash_report.previous_crash()
    assert pending.crash_text.startswith('Fatal Python error: Segmentation')
    assert 'render.failed' in pending.log_text
    assert crash_report.previous_crash([pending.key]) is None


def test_previous_crash_requires_confirmation_and_cancel_does_not_acknowledge(qapp, settings, monkeypatch):
    import dialogs.report_review as review
    from ui import crash_report
    opened = []
    choice = [False]

    class Box:
        def __init__(self, window):
            self.review = object()
        def setIcon(self, value): pass
        def setWindowTitle(self, value): pass
        def setText(self, value): pass
        def addButton(self, text, role): return self.review
        def exec(self): pass
        def clickedButton(self): return self.review if choice[0] else None

    class Dialog:
        def __init__(self, *args, **kwargs):
            opened.append(kwargs)
            self.sent_or_queued = False
        def exec(self): pass

    Box.Icon = crash_report.QMessageBox.Icon
    Box.ButtonRole = crash_report.QMessageBox.ButtonRole
    monkeypatch.setattr(crash_report, 'QMessageBox', Box)
    pending = crash_report.PreviousCrash('key', 'crash', 'log')
    monkeypatch.setattr(crash_report, 'previous_crash', lambda acknowledged: pending)
    monkeypatch.setattr(review, 'ReportReviewDialog', Dialog)
    monkeypatch.setattr(crash_report, 'get_settings', lambda: settings)
    crash_report.offer_previous_crash(None)
    assert not opened
    choice[0] = True
    crash_report.offer_previous_crash(None)
    assert opened == [{'crash_text': 'crash', 'log_text': 'log',
                       'app_version': 'unknown', 'build_channel': None}]
    assert 'key' not in (settings.value('reports/acknowledged_crashes', []) or [])




def test_previous_crash_discovers_profile_fallback_and_preserves_acknowledgements(tmp_path, monkeypatch):
    import hashlib
    import os
    from ui import crash_report
    from utils import diagnostics, logging_setup

    logs = tmp_path / 'logs'
    fallback = tmp_path / 'fallback'
    logs.mkdir()
    fallback.mkdir()
    name = 'openfocus-20261110T010101_000001-123.crash.txt'
    normal = logs / name
    rescued = fallback / name
    normal.write_text('Fatal Python error: Aborted\n', encoding='utf-8')
    rescued.write_text('Windows fatal exception: access violation\n', encoding='utf-8')
    os.utime(normal, (100, 100))
    os.utime(rescued, (200, 200))
    monkeypatch.setattr(logging_setup, 'logs_dir', lambda: logs)
    monkeypatch.setattr(logging_setup, '_process_alive', lambda pid: False)
    monkeypatch.setattr(diagnostics, 'fallback_crash_dir', lambda: fallback)
    monkeypatch.setattr(diagnostics, 'crash_log_path', lambda: None)
    rescued_report = crash_report.previous_crash()
    assert rescued_report.crash_text.startswith('Windows fatal')
    normal_report = crash_report.previous_crash([rescued_report.key])
    assert normal_report.crash_text.startswith('Fatal Python') and normal_report.key != rescued_report.key
    legacy_key = hashlib.sha256(name.encode('utf-8')).hexdigest()
    assert crash_report.previous_crash([legacy_key]).key == rescued_report.key
    assert crash_report.previous_crash([legacy_key, rescued_report.key]) is None
    monkeypatch.setattr(diagnostics, 'crash_log_path', lambda: rescued)
    assert crash_report.previous_crash().key == normal_report.key


def test_previous_crash_skips_unreadable_candidate(tmp_path, monkeypatch):
    import os
    from ui import crash_report
    from utils import diagnostics, logging_setup
    logs = tmp_path / 'logs'
    logs.mkdir()
    valid = logs / 'openfocus-20261110T010101_000001-101.crash.txt'
    bad = logs / 'openfocus-20261110T010101_000002-102.crash.txt'
    valid.write_text('Fatal Python error: Aborted\n')
    bad.write_text('Fatal Python error: Aborted\n')
    os.utime(valid, (100, 100))
    os.utime(bad, (200, 200))
    monkeypatch.setattr(logging_setup, 'logs_dir', lambda: logs)
    monkeypatch.setattr(logging_setup, '_process_alive', lambda pid: False)
    monkeypatch.setattr(diagnostics, 'fallback_crash_dir', lambda: None)
    monkeypatch.setattr(diagnostics, 'crash_log_path', lambda: None)
    original_tail = crash_report._tail
    monkeypatch.setattr(crash_report, '_tail', lambda path: '' if path == bad else original_tail(path))
    assert crash_report.previous_crash().crash_text.startswith('Fatal Python')
