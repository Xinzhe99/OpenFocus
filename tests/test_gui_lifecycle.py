"""Process-level regressions for queued GUI startup during a normal close."""
import os
from pathlib import Path
import site
import subprocess
import sys

import pytest
from tests.process_isolation import sandbox_env

pytest.importorskip("PyQt6")


def test_close_does_not_open_overdue_first_run_dialog(tmp_path, monkeypatch):
    """A slow cooperative stop must not let first-run UI consume Qt's quit."""
    root = Path(__file__).resolve().parents[1]
    discovery = tmp_path / "control.json"
    sentinel = tmp_path / "inherited-data"
    sentinel.mkdir()
    (sentinel / "settings.ini").write_bytes(b"sentinel settings\n")
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(sentinel))
    env = sandbox_env(tmp_path, {"QT_QPA_PLATFORM": "offscreen"},
                      instance_name=f"OpenFocus-lifecycle-{os.getpid()}-{tmp_path.name}")
    if os.name != 'nt':
        # Exercise the real instance listener when even its short name would
        # exceed the Unix socket pathname limit under an isolated TMPDIR.
        long_tmp = tmp_path / ('tmp-' + 'x' * 100)
        long_tmp.mkdir()
        env.update({name: str(long_tmp) for name in ('TMPDIR', 'TMP', 'TEMP')})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        site.getusersitepackages(), env.get("PYTHONPATH"),
    )))
    bootstrap = '''
import faulthandler, runpy, sys, time
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from tests.process_isolation import install_usage_sender
sender, reporters, calls = install_usage_sender()
from utils.usage_analytics import UsageAnalytics
from dialogs.account import AccountDialog
from dialogs.welcome import WelcomeDialog
import webbrowser
browser_calls = []
webbrowser.open = lambda *args, **kwargs: browser_calls.append((args, kwargs)) or False

original_account_exec = AccountDialog.exec
def skip_account(dialog):
    def skip_after_load():
        if getattr(dialog.window_main, '_account_load_job', None) is not None:
            QTimer.singleShot(10, skip_after_load)
        else:
            QTimer.singleShot(100, dialog.reject)
    QTimer.singleShot(0, skip_after_load)
    return original_account_exec(dialog)
AccountDialog.exec = skip_account

def fail_warning(parent, title, text, *args, **kwargs):
    raise RuntimeError(f"Unexpected startup warning: {title}: {text}")
QMessageBox.warning = staticmethod(fail_warning)

welcome_events = []
original_single_shot = QTimer.singleShot
def track_first_run_timer(interval, callback):
    if getattr(callback, '__name__', '') == '_show_welcome':
        welcome_events.append('scheduled')
        def dispatch():
            welcome_events.append('dispatched')
            callback()
        return original_single_shot(interval, dispatch)
    return original_single_shot(interval, callback)
QTimer.singleShot = staticmethod(track_first_run_timer)

def record_welcome(dialog):
    welcome_events.append('opened')
    return 0
WelcomeDialog.exec = record_welcome

faulthandler.dump_traceback_later(15, repeat=False)

QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, sys.argv[2])
original_shutdown = UsageAnalytics.shutdown
original_exec = QApplication.exec

def slow_shutdown(self):
    original_shutdown(self)
    # Keep the GUI thread busy until the real first-run timer is overdue.
    time.sleep(2)
    # Dispatch the overdue callback while closeEvent is still in progress.
    QApplication.processEvents()

def close_main_window():
    windows = [window for window in QApplication.topLevelWidgets()
               if isinstance(window, QMainWindow)]
    assert len(windows) == 1
    windows[0].close()

def run_event_loop():
    QTimer.singleShot(0, close_main_window)
    return original_exec()

UsageAnalytics.shutdown = slow_shutdown
QApplication.exec = staticmethod(run_event_loop)
sys.argv = [sys.argv[1]]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as stopped:
    assert stopped.code == 0
else:
    raise AssertionError("GUI entry point did not exit")
assert reporters and all(captured is sender for captured in reporters)
assert welcome_events == ['scheduled', 'dispatched'], welcome_events
assert not browser_calls, browser_calls
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", bootstrap, str(root / "main.py"),
             str(tmp_path / "settings")],
            cwd=root, env=env, capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail((exc.stderr or b'').decode('utf-8', 'replace'))
    assert result.returncode == 0, result.stdout + result.stderr
    assert not discovery.exists()
    assert {path.name: path.read_bytes() for path in sentinel.iterdir()} == {
        "settings.ini": b"sentinel settings\n"
    }


def test_close_waits_for_account_worker(qapp, tmp_path, monkeypatch):
    from threading import Event
    import time
    from PyQt6.QtTest import QTest
    from dialogs.account import AccountDialog
    from tests.test_control_channel import _new_main_window

    started, release = Event(), Event()

    class Account:
        def current_user(self):
            return None

        def is_signed_in(self):
            return False

        def terms(self):
            started.set()
            assert release.wait(5)
            return {'version': '2026-11-08', 'url': 'https://account.openfocus.top/terms',
                    'draft': True, 'required': False}

    window = _new_main_window(qapp, tmp_path, monkeypatch)
    window._account_client = Account()
    dialog = AccountDialog(window)
    dialog.show()
    try:
        dialog.terms.setChecked(True)
        dialog.github_button.click()
        assert started.wait(5)
        window._force_quit = True
        assert not window.close()
        assert window.isVisible()
        release.set()
        deadline = time.monotonic() + 5
        while window.isVisible() and time.monotonic() < deadline:
            QTest.qWait(10)
        assert not window.isVisible()
    finally:
        release.set()
        window._force_quit = True
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_close_does_not_wait_for_another_windows_account_job(qapp, tmp_path, monkeypatch):
    from threading import Event
    from PyQt6.QtWidgets import QWidget
    from dialogs.account import _AccountJob
    from tests.test_control_channel import _destroy_main_window, _new_main_window, wait_until

    window = _new_main_window(qapp, tmp_path, monkeypatch)
    assert wait_until(lambda: window._account_load_job is None)
    owner = QWidget()
    started, release = Event(), Event()
    def work(_started):
        started.set()
        release.wait(5)
    job = _AccountJob(work, owner)
    job.finished.connect(job._release)
    job.start()
    try:
        assert started.wait(5)
        window._force_quit = True
        assert window.close()
        assert not window.isVisible()
    finally:
        release.set()
        job._thread.join(5)
        qapp.processEvents()
        _destroy_main_window(qapp, window)
        owner.deleteLater()
