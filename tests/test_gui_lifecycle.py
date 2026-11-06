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
import runpy, sys, time
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow
from utils.usage_analytics import UsageAnalytics

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
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, str(root / "main.py"),
         str(tmp_path / "settings")],
        cwd=root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not discovery.exists()
