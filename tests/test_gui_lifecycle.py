"""Process-level regressions for queued GUI startup during a normal close."""
import os
from pathlib import Path
import site
import subprocess
import sys

import pytest

pytest.importorskip("PyQt6")


def test_close_does_not_open_overdue_first_run_dialog(tmp_path):
    """A slow cooperative stop must not let first-run UI consume Qt's quit."""
    root = Path(__file__).resolve().parents[1]
    discovery = tmp_path / "control.json"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               OPENFOCUS_CONTROL_FILE=str(discovery),
               OPENFOCUS_INSTANCE_NAME=f"OpenFocus-lifecycle-{os.getpid()}-{tmp_path.name}",
               OPENFOCUS_DATA_DIR=str(tmp_path / "data"),
               OPENFOCUS_LOG_DIR=str(tmp_path / "logs"),
               APPDATA=str(tmp_path / "appdata"),
               XDG_CONFIG_HOME=str(tmp_path / "config"),
               XDG_DATA_HOME=str(tmp_path / "data"),
               HOME=str(tmp_path), USERPROFILE=str(tmp_path))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        site.getusersitepackages(), env.get("PYTHONPATH"),
    )))
    bootstrap = '''
import faulthandler, runpy, sys, time
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow
from utils.usage_analytics import UsageAnalytics
from dialogs.account import AccountDialog
import webbrowser
webbrowser.open = lambda *args, **kwargs: False

original_account_exec = AccountDialog.exec
def skip_account(dialog):
    QTimer.singleShot(0, dialog.reject)
    return original_account_exec(dialog)
AccountDialog.exec = skip_account

faulthandler.dump_traceback_later(15, repeat=False)

QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, sys.argv[2])
original_shutdown = UsageAnalytics.shutdown
original_exec = QApplication.exec

def slow_shutdown(self):
    original_shutdown(self)
    # Keep the GUI thread busy until the real first-run timer is overdue.
    time.sleep(2)

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
