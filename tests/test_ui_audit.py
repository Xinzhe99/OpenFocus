"""Regression coverage for the standalone screenshot audit."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import ui_audit


ROOT = Path(__file__).resolve().parents[1]


def test_grab_rejects_failed_screenshot(monkeypatch, tmp_path):
    monkeypatch.setattr(ui_audit, "_settle", lambda app: None)
    widget = SimpleNamespace(grab=lambda: SimpleNamespace(save=lambda path: False))
    with pytest.raises(OSError, match="Could not save screenshot"):
        ui_audit._grab(widget, str(tmp_path / "failed.png"))


@pytest.mark.parametrize("failure", ["RuntimeError", "KeyboardInterrupt"])
def test_private_profile_cleanup_on_failure(tmp_path, failure):
    user = tmp_path / "user"
    user.mkdir()
    sentinel = user / "settings.ini"
    sentinel.write_text("[ui]\nlanguage=zh\ntheme=light\n", encoding="utf-8")
    output = tmp_path / "evidence"
    script = '''
import os, sys
from pathlib import Path
from tools import ui_audit
from utils.settings_store import get_settings
original_data = os.environ["OPENFOCUS_DATA_DIR"]
original_log = os.environ["OPENFOCUS_LOG_DIR"]
sys.argv = ["ui_audit", "--out", sys.argv[1]]
def fail(app, windows, out):
    root = Path(os.environ["OPENFOCUS_DATA_DIR"])
    assert str(root) != original_data
    assert os.environ["OPENFOCUS_LOG_DIR"] != original_log
    assert get_settings().value("usage/enabled", type=bool) is False
    Path(out, "profile-path.txt").write_text(str(root))
    raise EXCEPTION()
ui_audit._main_window = fail
try:
    ui_audit.main()
except EXCEPTION:
    pass
else:
    raise AssertionError("failure was suppressed")
assert os.environ["OPENFOCUS_DATA_DIR"] == original_data
assert os.environ["OPENFOCUS_LOG_DIR"] == original_log
assert not Path(Path(sys.argv[2], "profile-path.txt").read_text()).exists()
'''.replace("EXCEPTION", failure)
    env = dict(os.environ, OPENFOCUS_DATA_DIR=str(user), OPENFOCUS_LOG_DIR=str(user),
               QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, "-c", script, str(output)], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert sentinel.read_text(encoding="utf-8") == "[ui]\nlanguage=zh\ntheme=light\n"


def test_rejects_initialized_settings_before_window(monkeypatch, tmp_path):
    from utils import settings_store
    monkeypatch.setattr(settings_store, "_settings_singleton", object())
    monkeypatch.setattr(sys, "argv", ["ui_audit", "--out", str(tmp_path)])
    with pytest.raises(RuntimeError, match="independent process"):
        ui_audit.main()
