"""Regression coverage for previous-session native crash discovery."""
import os

from ui import crash_report
from utils import diagnostics, logging_setup


def test_previous_crash_skips_embedded_header_and_current_session(tmp_path, monkeypatch):
    logs = tmp_path / 'logs'
    logs.mkdir()
    genuine = logs / 'openfocus-earlier.crash.txt'
    genuine.write_text('Windows fatal exception: access violation\n', encoding='utf-8')
    false_marker = logs / 'openfocus-later.crash.txt'
    false_marker.write_text('ValueError: Fatal Python error: mentioned in message\n', encoding='utf-8')
    current = logs / 'openfocus-current.crash.txt'
    current.write_text('Fatal Python error: Aborted\n', encoding='utf-8')
    os.utime(genuine, (100, 100))
    os.utime(false_marker, (200, 200))
    os.utime(current, (300, 300))
    monkeypatch.setattr(logging_setup, 'logs_dir', lambda: logs)
    monkeypatch.setattr(diagnostics, 'fallback_crash_dir', lambda: None)
    monkeypatch.setattr(diagnostics, 'crash_log_path', lambda: current)

    key, crash, _ = crash_report.previous_native_crash()
    assert crash.splitlines() == ['Windows fatal exception: access violation']
    assert crash_report.previous_native_crash([key]) is None


def test_previous_crash_acknowledged_only_after_send_or_queue(tmp_path, monkeypatch):
    from dialogs import report_review

    logs = tmp_path / 'logs'
    logs.mkdir()
    marker = logs / 'openfocus-previous.crash.txt'
    marker.write_text('Fatal Python error: Aborted\n', encoding='utf-8')
    monkeypatch.setattr(logging_setup, 'logs_dir', lambda: logs)
    monkeypatch.setattr(diagnostics, 'fallback_crash_dir', lambda: None)
    monkeypatch.setattr(diagnostics, 'crash_log_path', lambda: None)
    monkeypatch.setattr(crash_report.trans, 't', lambda key: key)
    result = [False]

    class Settings:
        def __init__(self):
            self.values = {}
        def value(self, key, default=None):
            return self.values.get(key, default)
        def setValue(self, key, value):
            self.values[key] = value
        def sync(self):
            pass

    class Box:
        Icon = crash_report.QMessageBox.Icon
        ButtonRole = crash_report.QMessageBox.ButtonRole
        def __init__(self, window):
            self.review = object()
        def setIcon(self, value): pass
        def setWindowTitle(self, value): pass
        def setText(self, value): pass
        def addButton(self, text, role): return self.review
        def exec(self): pass
        def clickedButton(self): return self.review

    class Dialog:
        def __init__(self, window, *, crash_text, log_text):
            assert crash_text.splitlines() == ['Fatal Python error: Aborted']
            self.sent_or_queued = result[0]
        def exec(self): pass

    settings = Settings()
    monkeypatch.setattr(crash_report, 'get_settings', lambda: settings)
    monkeypatch.setattr(crash_report, 'QMessageBox', Box)
    monkeypatch.setattr(report_review, 'ReportReviewDialog', Dialog)
    key = crash_report.previous_native_crash()[0]

    crash_report.offer_previous_crash(None)
    assert crash_report.previous_native_crash(settings.value('reports/acknowledged_crashes', []))[0] == key
    result[0] = True
    crash_report.offer_previous_crash(None)
    assert settings.value('reports/acknowledged_crashes') == [key]
    assert crash_report.previous_native_crash(settings.value('reports/acknowledged_crashes')) is None
