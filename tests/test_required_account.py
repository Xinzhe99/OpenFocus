"""Required-account desktop behavior with isolated synthetic sessions."""
import time
from threading import Event

from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from tests.test_control_channel import _destroy_main_window, _new_main_window
from utils.account_client import AccountClient, AccountError, TokenStore


_SESSION = {
    'token': 'synthetic-test-token', 'token_type': 'Bearer',
    'expires_at': 1, 'user': {'id': 'synthetic-test-user'},
}


def _wait(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, 'account operation timed out'
        QTest.qWait(5)


def _drive_modal(action):
    """Run an assertion/action once a modal dialog appears, with a deadline."""
    errors = []
    deadline = time.monotonic() + 5

    def drive():
        dialog = QApplication.activeModalWidget()
        if dialog is None:
            if time.monotonic() < deadline:
                QTimer.singleShot(5, drive)
            else:
                errors.append(AssertionError('account dialog did not appear'))
            return
        try:
            action(dialog)
        except BaseException as exc:
            errors.append(exc)
            dialog.reject()

    QTimer.singleShot(0, drive)
    return errors


def test_clean_profile_cannot_enter_without_login(qapp, tmp_path, monkeypatch):
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    window.hide()
    try:
        def close_login(dialog):
            _wait(lambda: dialog.client is not None)
            assert not dialog.client.is_signed_in()
            assert dialog.form.isVisible()
            assert not window.isVisible()
            dialog.close_button.click()

        errors = _drive_modal(close_login)
        assert window._require_account() is False
        assert not errors
        assert not window.isVisible()
    finally:
        _destroy_main_window(qapp, window)


def test_persisted_session_allows_offline_startup_even_after_expiry(qapp, tmp_path, monkeypatch):
    store = TokenStore(path=tmp_path / 'account-session')
    store.save(_SESSION)
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    window.hide()
    try:
        # The window fixture's account client raises on every network request.
        assert window._require_account() is True
        assert window._account_client.current_user()['id'] == 'synthetic-test-user'
    finally:
        _destroy_main_window(qapp, window)


def test_restored_session_applies_accepted_reporting_terms_offline(qapp, tmp_path, monkeypatch):
    from utils.settings_store import automatic_reports_allowed, get_settings

    TokenStore(path=tmp_path / 'account-session').save({
        **_SESSION, 'terms': {'accepted_at': 1, 'version': '2026-11-11'},
    })
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        _wait(lambda: window._account_load_job is None)
        assert automatic_reports_allowed(get_settings())
        assert get_settings().value('reports/auto_send_choice') == 'account_default'
    finally:
        _destroy_main_window(qapp, window)


def test_sign_out_requires_login_again(qapp, tmp_path, monkeypatch):
    TokenStore(path=tmp_path / 'account-session').save(_SESSION)
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    try:
        _wait(lambda: getattr(window, '_account_load_job', None) is None)
        # Sign-out succeeds locally without a real service.
        window._account_client._request = lambda *args, **kwargs: {'ok': True}
        seen = []

        def open_gate():
            def close_gate(dialog):
                seen.append('gate')
                assert not dialog.client.is_signed_in()
                assert dialog.form.isVisible()
                dialog.close_button.click()
            seen.append(_drive_modal(close_gate))

        def sign_out(dialog):
            dialog.logout_button.click()
            _wait(lambda: dialog._job is None)
            assert not window._account_client.is_signed_in()
            assert TokenStore(path=tmp_path / 'account-session').load() is None
            QTimer.singleShot(0, open_gate)
            dialog.close_button.click()

        errors = _drive_modal(sign_out)
        window.show_account_dialog()
        assert not errors
        assert seen and seen[-1] == 'gate'
        assert not seen[0]
        assert not window.isVisible()
    finally:
        _destroy_main_window(qapp, window)


def test_account_storage_initialization_failure_can_be_retried(qapp, tmp_path, monkeypatch):
    from utils import account_client

    attempts = []
    entered, release = Event(), Event()
    from dialogs import account as account_dialog
    store = TokenStore(path=tmp_path / 'account-session')

    def create_client():
        attempts.append(None)
        if len(attempts) == 1:
            entered.set()
            assert release.wait(5)
            raise AccountError('storage')
        return AccountClient(store=store, request=lambda *args, **kwargs: {})

    monkeypatch.setattr(account_client, 'AccountClient', create_client)
    monkeypatch.setattr(account_dialog, 'AccountClient', create_client)
    window = _new_main_window(qapp, tmp_path, monkeypatch)
    assert entered.wait(5)
    window.hide()
    try:
        def retry_then_close(dialog):
            release.set()
            _wait(lambda: dialog._job is None)
            assert dialog.retry_button.isVisible()
            assert not window.isVisible()
            dialog.retry_button.click()
            _wait(lambda: dialog.client is not None)
            assert len(attempts) == 2
            assert dialog.form.isEnabled()
            dialog.close_button.click()

        errors = _drive_modal(retry_then_close)
        assert window._require_account() is False
        assert not errors
    finally:
        release.set()
        _destroy_main_window(qapp, window)
