"""Isolated tests for default-on reporting; no external requests are made."""
import uuid
import time

import pytest
from PyQt6.QtCore import QSettings

from constants import APP_VERSION
from utils import usage_analytics as usage


class DeferredThread:
    instances = []

    def __init__(self, *, target, daemon, name):
        assert daemon
        self.target = target
        self.instances.append(self)

    def start(self):
        pass

    def run(self):
        self.target()


@pytest.fixture
def setup(qapp, tmp_path, monkeypatch):
    DeferredThread.instances = []
    monkeypatch.setattr(usage.threading, "Thread", DeferredThread)
    monkeypatch.setattr(usage, "_platform", lambda: "linux")
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    calls = []
    now = [1_000_000.0]
    reporter = usage.UsageAnalytics(settings=settings, sender=lambda payload: calls.append(payload) or True,
                                    clock=lambda: now[0])
    return reporter, settings, calls, now


@pytest.mark.parametrize("previous", [None, False])
def test_window_startup_respects_saved_reporting_preference(qapp, tmp_path, monkeypatch, previous):
    from main import OpenFocus
    from utils import settings_store

    settings = QSettings(str(tmp_path / "window-settings.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_store, "_settings_singleton", settings)
    monkeypatch.setattr(usage, "_platform", lambda: "linux")
    if previous is not None:
        settings.setValue(usage.ENABLED_KEY, previous)
    calls = []
    monkeypatch.setattr(usage, "_send", lambda payload: calls.append(payload) or True)
    window = OpenFocus()
    try:
        qapp.processEvents()
        assert window.usage_analytics.enabled() is (previous is None)
        assert 'usage_opt_in' not in window.ui_objs
        assert settings.contains(usage.ID_KEY) is (previous is None)
        if previous is None:
            _wait_for(qapp, lambda: len(calls) == 1)
        else:
            assert not calls
    finally:
        window._force_quit = True
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_default_on_persists_identity_and_payload_is_minimal(setup):
    reporter, settings, calls, now = setup
    assert reporter.enabled()
    reporter.start()
    identifier = settings.value(usage.ID_KEY)
    assert str(uuid.UUID(identifier, version=4)) == identifier
    assert len(DeferredThread.instances) == 1
    reporter.start()
    assert len(DeferredThread.instances) == 1
    DeferredThread.instances[-1].run()
    assert calls == [{"installation_id": identifier, "version": APP_VERSION, "platform": "linux"}]
    assert float(settings.value(usage.LAST_SENT_KEY)) == now[0]
    reporter.start()
    assert len(DeferredThread.instances) == 1
    now[0] += usage.INTERVAL_SECONDS
    reporter.start()
    assert len(DeferredThread.instances) == 2
    assert settings.value(usage.ID_KEY) == identifier


def test_opt_out_cancels_queued_request_and_removes_identity(setup):
    reporter, settings, calls, _ = setup
    reporter.set_enabled(True)
    old_id = settings.value(usage.ID_KEY)
    reporter.set_enabled(False)
    DeferredThread.instances[-1].run()
    assert not calls
    assert not settings.contains(usage.ID_KEY)
    assert not settings.contains(usage.LAST_SENT_KEY)
    reporter.set_enabled(True)
    assert settings.value(usage.ID_KEY) != old_id
    DeferredThread.instances[-1].run()
    assert len(calls) == 1


def test_failed_send_retries_and_never_marks_success(setup):
    reporter, settings, calls, _ = setup
    def offline(payload):
        calls.append(payload)
        raise OSError("offline")
    reporter.sender = offline
    reporter.set_enabled(True)
    DeferredThread.instances[-1].run()
    assert not settings.contains(usage.LAST_SENT_KEY)
    reporter.start()
    DeferredThread.instances[-1].run()
    assert len(calls) == 2
    assert not settings.contains(usage.LAST_SENT_KEY)


def test_opt_out_ignores_inflight_success(setup):
    reporter, settings, calls, _ = setup
    reporter.set_enabled(True)
    old_id = settings.value(usage.ID_KEY)
    reporter.set_enabled(False)
    reporter._on_finished(old_id, True)
    assert not settings.contains(usage.ID_KEY)
    assert not settings.contains(usage.LAST_SENT_KEY)
    assert not calls


@pytest.mark.parametrize("stored", [False, "false", "0"])
def test_existing_explicit_opt_out_remains_disabled(setup, stored):
    reporter, settings, calls, _ = setup
    settings.setValue(usage.ENABLED_KEY, stored)
    reporter.start()
    assert not reporter.enabled()
    assert not DeferredThread.instances and not calls
    assert not settings.contains(usage.ID_KEY)


def test_shutdown_cancels_queued_and_future_reports(setup):
    reporter, settings, calls, _ = setup
    reporter.start()
    identifier = settings.value(usage.ID_KEY)
    reporter.shutdown()
    DeferredThread.instances[-1].run()
    reporter.start()
    assert not calls
    assert not reporter._timer.isActive()
    assert not settings.contains(usage.LAST_SENT_KEY)
    assert settings.value(usage.ID_KEY) == identifier


def test_startup_preserves_existing_identity_and_daily_schedule(setup):
    reporter, settings, calls, now = setup
    identifier = str(uuid.uuid4())
    settings.setValue(usage.ID_KEY, identifier)
    settings.setValue(usage.LAST_SENT_KEY, now[0])
    reporter.start()
    assert not DeferredThread.instances and not calls
    assert reporter._timer.isActive()
    assert settings.value(usage.ID_KEY) == identifier
    reporter.shutdown()

def _wait_for(qapp, predicate, timeout=2.0):
    from PyQt6.QtTest import QTest
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(5)
    assert predicate()


def test_running_application_reports_again_and_opt_out_stops_it(qapp, tmp_path, monkeypatch):
    from PyQt6.QtTest import QTest
    monkeypatch.setattr(usage, 'INTERVAL_SECONDS', 0.08)
    settings = QSettings(str(tmp_path / 'running.ini'), QSettings.Format.IniFormat)
    settings.setValue(usage.ENABLED_KEY, True)
    settings.setValue(usage.ID_KEY, str(uuid.uuid4()))
    settings.setValue(usage.LAST_SENT_KEY, time.time())
    calls = []
    reporter = usage.UsageAnalytics(settings=settings, sender=lambda payload: calls.append(payload) or True)
    try:
        reporter.start()
        assert not calls
        _wait_for(qapp, lambda: len(calls) >= 2)
        reporter.set_enabled(False)
        count = len(calls)
        QTest.qWait(200)
        assert len(calls) == count
        assert not settings.contains(usage.ID_KEY)
    finally:
        reporter.shutdown()


def test_failed_report_retries_while_application_remains_open(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(usage, 'RETRY_INTERVAL_SECONDS', 0.05, raising=False)
    settings = QSettings(str(tmp_path / 'offline.ini'), QSettings.Format.IniFormat)
    attempts = []
    def send(payload):
        attempts.append(payload)
        if len(attempts) == 1:
            raise OSError('offline')
        return True
    reporter = usage.UsageAnalytics(settings=settings, sender=send)
    try:
        reporter.set_enabled(True)
        _wait_for(qapp, lambda: len(attempts) >= 2)
        _wait_for(qapp, lambda: settings.contains(usage.LAST_SENT_KEY))
    finally:
        reporter.set_enabled(False)
        reporter.shutdown()


def test_http_reporting_survives_default_python_agent_block(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            agent = self.headers.get('User-Agent', '')
            self.send_response(403 if agent.startswith('Python-urllib/') else 204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(usage, 'ENDPOINT', f'http://127.0.0.1:{server.server_port}/api/usage')
        assert usage._send({'installation_id': str(uuid.uuid4()),
                            'version': APP_VERSION, 'platform': 'linux'})
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
