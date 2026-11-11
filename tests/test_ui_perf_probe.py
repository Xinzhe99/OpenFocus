"""Regression coverage for the standalone synthetic UI performance probe."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("PyQt6")

ROOT = Path(__file__).resolve().parents[1]


def _run(script, tmp_path):
    user = tmp_path / "user"
    logs = tmp_path / "logs"
    user.mkdir()
    logs.mkdir()
    sentinel = user / "settings.ini"
    original = b"[ui]\nlanguage=zh\ntheme=light\nwindow_geometry=sentinel\n"
    sentinel.write_bytes(original)
    recovery = user / "recovery-sentinel"
    recovery.write_bytes(b"unchanged")
    env = os.environ.copy()
    env.update(OPENFOCUS_DATA_DIR=str(user), OPENFOCUS_LOG_DIR=str(logs),
               QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert sentinel.read_bytes() == original
    assert recovery.read_bytes() == b"unchanged"
    assert list(logs.iterdir()) == []
    return result


def test_probe_uses_private_profile_and_real_wipe(tmp_path):
    result = _run('''
import sys
from utils import usage_analytics
calls = []
usage_analytics._platform = lambda: "test"
usage_analytics._send = lambda payload: calls.append(payload)
from tools import ui_perf_probe
sys.argv = ["ui_perf_probe", "--size", "256x192", "--frames", "3"]
ui_perf_probe.main()
assert not calls
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "wipe source step (result fixed)" in result.stdout
    assert "wipe step (both sides)" not in result.stdout


def test_probe_rejects_missing_context_without_wipe_sample(tmp_path):
    result = _run('''
import sys
from core import render_context
render_context.build_render_context = lambda *args, **kwargs: None
from tools import ui_perf_probe
sys.argv = ["ui_perf_probe", "--size", "256x192", "--frames", "3"]
ui_perf_probe.main()
''', tmp_path)
    assert result.returncode != 0
    assert "Could not build synthetic result context" in result.stderr
    assert "wipe source step (result fixed)" not in result.stdout


def test_probe_rejects_preinitialized_settings(tmp_path):
    result = _run('''
import sys
from utils.settings_store import get_settings
get_settings()
from tools import ui_perf_probe
sys.argv = ["ui_perf_probe", "--size", "256x192", "--frames", "3"]
ui_perf_probe.main()
''', tmp_path)
    assert result.returncode != 0
    assert "fresh, independent process" in result.stderr
