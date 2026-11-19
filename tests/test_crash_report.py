"""Regression coverage for previous-session crash discovery and review."""
import io
import json
import os
import zipfile

import pytest

from ui import crash_report
from utils import diagnostics, logging_setup
from utils.report_bundle import build_report_bundle
from utils.reporting import report_metadata


@pytest.fixture
def crash_logs(tmp_path, monkeypatch):
    logs = tmp_path / 'logs'
    logs.mkdir()
    monkeypatch.setattr(logging_setup, 'logs_dir', lambda: logs)
    monkeypatch.setattr(logging_setup, '_process_alive', lambda pid: False)
    monkeypatch.setattr(diagnostics, 'fallback_crash_dir', lambda: None)
    monkeypatch.setattr(diagnostics, 'crash_log_path', lambda: None)
    return logs


def test_previous_crash_skips_embedded_header_and_current_session(crash_logs, monkeypatch):
    genuine = crash_logs / 'openfocus-20261110T010101_000001-101.crash.txt'
    genuine.write_text('Windows fatal exception: access violation\n', encoding='utf-8')
    false_marker = crash_logs / 'openfocus-20261110T010101_000002-102.crash.txt'
    false_marker.write_text('ValueError: Fatal Python error: mentioned in message\n', encoding='utf-8')
    current = crash_logs / 'openfocus-20261110T010101_000003-103.crash.txt'
    current.write_text('Fatal Python error: Aborted\n', encoding='utf-8')
    os.utime(genuine, (100, 100))
    os.utime(false_marker, (200, 200))
    os.utime(current, (300, 300))
    monkeypatch.setattr(diagnostics, 'crash_log_path', lambda: current)

    pending = crash_report.previous_crash()
    assert pending.crash_text.splitlines() == ['Windows fatal exception: access violation']
    assert crash_report.previous_crash([pending.key]) is None


def test_previous_crash_acknowledged_only_after_send_or_queue(crash_logs, monkeypatch):
    from dialogs import report_review

    marker = crash_logs / 'openfocus-20261110T010101_000001-101.crash.txt'
    session = '20261110T010101_000001-101'
    marker.write_text(f'2026-11-10T01:01:02.000001 session={session} '
                      'fatal unhandled main-thread exception\n'
                      'Traceback (most recent call last):\n'
                      '  File "/synthetic/private/workers.py", line 42, in run\n'
                      'RuntimeError: synthetic private message\n'
                      f'\n2026-11-10T01:01:03.000001 session={session} unraisable exception: finalizer\n'
                      'Traceback (most recent call last):\n'
                      '  File "/synthetic/private/main.py", line 73, in __del__\n'
                      'ValueError: synthetic cleanup failure\n', encoding='utf-8')
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
        def __init__(self, window, *, crash_text, log_text, app_version, build_channel):
            assert 'RuntimeError' in crash_text and 'ValueError' not in crash_text
            assert app_version == 'unknown' and build_channel is None
            bundle = build_report_bundle(crash_text=crash_text, log_text=log_text, facts={},
                                         app_version=app_version, build_channel=build_channel)
            with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
                error = json.loads(archive.read('error.json'))
            assert error['kind'] == 'RuntimeError'
            assert error['stack'] == [{'file': 'workers.py', 'line': 42, 'function': 'run'}]
            assert 'synthetic' not in json.dumps(error)
            self.sent_or_queued = result[0]
        def exec(self): pass

    settings = Settings()
    monkeypatch.setattr(crash_report, 'get_settings', lambda: settings)
    monkeypatch.setattr(crash_report, 'QMessageBox', Box)
    monkeypatch.setattr(report_review, 'ReportReviewDialog', Dialog)
    key = crash_report.previous_crash().key

    crash_report.offer_previous_crash(None)
    assert crash_report.previous_crash(settings.value('reports/acknowledged_crashes', [])).key == key
    result[0] = True
    crash_report.offer_previous_crash(None)
    assert settings.value('reports/acknowledged_crashes') == [key]
    assert crash_report.previous_crash(settings.value('reports/acknowledged_crashes')) is None


def test_previous_crash_defers_occupied_pid_and_skips_normal_exit(crash_logs, monkeypatch):
    session = '20261110T010101_000001-101'
    active = crash_logs / f'openfocus-{session}.crash.txt'
    active.write_text(f'2026-11-10T01:01:02.000001 session={session} '
                      'fatal unhandled main-thread exception\n', encoding='utf-8')
    normal_exit = crash_logs / 'openfocus-20261110T010101_000002-102.crash.txt'
    normal_exit.touch()
    wrong_session = crash_logs / 'openfocus-20261110T010101_000003-103.crash.txt'
    wrong_session.write_text(active.read_text(encoding='utf-8'), encoding='utf-8')
    monkeypatch.setattr(logging_setup, '_process_alive', lambda pid: pid == 101)

    assert crash_report.previous_crash() is None
    monkeypatch.setattr(logging_setup, '_process_alive', lambda pid: False)
    assert crash_report.previous_crash().crash_text.splitlines() == active.read_text(encoding='utf-8').splitlines()


@pytest.mark.parametrize('prefix, banner, version, channel', [
    ('', 'openfocus: event=session.started {"version":"1.12.5-beta.4","frozen":false}',
     '1.12.5-beta.4', 'development'),
    ('', 'openfocus: event=session.started {"version":"1.12.5","frozen":true}',
     '1.12.5', 'release'),
    ('', 'openfocus.worker: reported event=session.started {"version":"42.1","frozen":true}',
     'unknown', None),
    ('', 'openfocus: event=session.started {"version":"/synthetic/private","frozen":true}',
     'unknown', None),
    ('RuntimeError: synthetic message\n',
     'openfocus: event=session.started {"version":"42.1","frozen":true}', 'unknown', None),
])
def test_previous_crash_uses_retained_rotations_and_validated_banner(
        crash_logs, monkeypatch, prefix, banner, version, channel):
    monkeypatch.setattr(crash_report, '_MAX_TAIL', 256)
    session = '20261110T010101_000001-101'
    stem = f'openfocus-{session}'
    (crash_logs / f'{stem}.crash.txt').write_text('Fatal Python error: Aborted\n', encoding='utf-8')
    (crash_logs / f'{stem}.log.2').write_text(
        f'{prefix}2026-11-10 01:01:01,001 INFO    pid=101 thread=MainThread session={session} '
        f'operation=- {banner}\n' + 'old data\n' * 40, encoding='utf-8')
    (crash_logs / f'{stem}.log.1').write_text('retained context\n', encoding='utf-8')
    (crash_logs / f'{stem}.log').write_text('latest context\n', encoding='utf-8')
    (crash_logs / 'openfocus-20261110T010102_000001-101.log').write_text(
        'unrelated context\n', encoding='utf-8')

    pending = crash_report.previous_crash()
    assert pending.log_text.splitlines()[-2:] == ['retained context', 'latest context']
    assert pending.log_text.startswith('...(older lines trimmed;')
    assert 'session.started' not in pending.log_text and 'unrelated' not in pending.log_text
    assert pending.app_version == version and pending.build_channel == channel
    bundle = build_report_bundle(crash_text=pending.crash_text, log_text=pending.log_text,
                                 facts={'OpenFocus': '9.0', 'frozen': True, 'python': '3.13.0'},
                                 app_version=pending.app_version, build_channel=pending.build_channel)
    metadata = report_metadata(bundle, source='manual', locale='en', severity='crash')
    assert metadata['app_version'] == version
    if channel is not None:
        assert metadata['build_channel'] == channel
    with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
        facts = json.loads(archive.read('system.json'))
    assert 'OpenFocus' not in facts and 'python' not in facts
    assert facts.get('frozen') is (None if channel is None else channel == 'release')
