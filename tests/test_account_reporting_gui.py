"""Account and report interactions remain optional and reviewable."""
import json
import time
from threading import Event

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QWidget


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
        return {'version': '2026-11-08', 'url': 'https://account.openfocus.top/terms',
                'draft': True, 'required': True}

    def login(self, provider, **kwargs):
        self.calls.append((provider, kwargs['terms_version'], kwargs['language']))
        self.user = {'id': 'owner', 'email': 'user@example.test'}
        return self.user


def test_account_requires_review_and_consent_then_remembers_reporting(qapp, settings, monkeypatch):
    from dialogs.account import AccountDialog
    from PyQt6.QtGui import QDesktopServices
    opened = []
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()) or True)
    window = QWidget()
    client = window._account_client = _Account()
    dialog = AccountDialog(window)
    try:
        assert not dialog.auto_reports.isChecked()
        assert not dialog.terms.isEnabled()
        dialog.github_button.click()
        assert client.calls == []
        dialog.terms_link.click()
        _wait(qapp, lambda: dialog._job is None)
        assert opened and opened[0].startswith('https://account.openfocus.top/terms?')
        dialog.terms.setChecked(True)
        dialog.github_button.click()
        _wait(qapp, lambda: dialog._job is None)
        assert client.calls[1][0:2] == ('github', '2026-11-08')
        assert not settings.value('reports/auto_send', type=bool)
        assert not window.auto_error_reports
        dialog.auto_reports.setChecked(True)
        dialog.reject()
        assert settings.value('reports/auto_send', type=bool)
    finally:
        dialog.reject()
        window.deleteLater()


def test_skip_while_terms_load_does_not_open_browser(qapp, settings, monkeypatch):
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
        dialog.terms_link.click()
        assert entered.wait(5)
        dialog.reject()
        release.set()
        _wait(qapp, lambda: dialog._job is None)
        assert not opened
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
    monkeypatch.setattr(diagnostics, 'crash_log_path', lambda: current)
    key, crash, log = crash_report.previous_native_crash()
    assert crash.startswith('Fatal Python error: Segmentation')
    assert 'render.failed' in log
    assert crash_report.previous_native_crash([key]) is None


def test_shutdown_reports_only_send_current_errors(qapp, settings, monkeypatch):
    from ui import shutdown_reports

    sent = []
    class Reports:
        def __init__(self, *_): pass
        @property
        def pending_count(self): return 0
    monkeypatch.setattr(shutdown_reports, 'ReportClient', Reports)
    monkeypatch.setattr(shutdown_reports, 'session_has_error', lambda: False)
    monkeypatch.setattr(shutdown_reports, 'send_bundle', lambda *args, **kwargs: sent.append(args))
    job = shutdown_reports._ShutdownReport(None, 'en', True)
    job._run()
    assert not sent
