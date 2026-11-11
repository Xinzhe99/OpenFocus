"""Process isolation and shutdown coverage for the language audit."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("language,failure", [
    ("en", ""), ("de", ""), ("en", "error"), ("de", "interrupt"),
])
def test_audit_uses_private_profile_and_cleans_up(tmp_path, language, failure):
    data = tmp_path / "user-data"
    logs = tmp_path / "user-logs"
    data.mkdir()
    logs.mkdir()
    sentinels = {
        data / "settings.ini": b"[ui]\nlanguage=zh\ntheme=light\nlayout=sentinel\n",
        data / "recovery.json": b"recovery sentinel",
        logs / "existing.log": b"log sentinel",
    }
    for path, content in sentinels.items():
        path.write_bytes(content)
    env = os.environ.copy()
    env.update(OPENFOCUS_DATA_DIR=str(data), OPENFOCUS_LOG_DIR=str(logs),
               QT_QPA_PLATFORM="offscreen")
    script = textwrap.dedent('''
        import os, sys, threading
        from pathlib import Path
        from tools import lang_ui_audit as audit
        from utils import usage_analytics

        sends = []
        usage_analytics._send = lambda payload: sends.append(payload)
        original = audit._run_audit
        profile = []
        def run(lang, app):
            from core import frame_storage
            from utils.settings_store import get_settings
            root = Path(os.environ["OPENFOCUS_DATA_DIR"])
            profile.append(root)
            assert root != Path(os.environ["EXPECTED_DATA"])
            assert Path(os.environ["OPENFOCUS_LOG_DIR"]).is_relative_to(root)
            assert Path(frame_storage.storage_directory()) == root
            assert str(get_settings().value("usage/enabled")).lower() == "false"
            if failure:
                def fail(_):
                    if failure == "interrupt":
                        raise KeyboardInterrupt()
                    raise RuntimeError("injected audit failure")
                audit.clipped = fail
            return original(lang, app)
        audit._run_audit = run
        language, failure = sys.argv[1:3]
        sys.argv = ["lang_ui_audit.py", language]
        try:
            audit.main()
            assert not failure
        except (RuntimeError, KeyboardInterrupt) as exc:
            assert failure and (isinstance(exc, KeyboardInterrupt) if failure == "interrupt"
                                else str(exc) == "injected audit failure")
        from core import frame_storage
        from utils import settings_store
        assert profile and not profile[0].exists()
        assert settings_store._settings_singleton is None
        assert frame_storage._storage_directory is None
        assert os.environ["OPENFOCUS_DATA_DIR"] == os.environ["EXPECTED_DATA"]
        assert os.environ["OPENFOCUS_LOG_DIR"] == os.environ["EXPECTED_LOGS"]
        assert not sends
        assert not any(t.name == "usage-analytics" and t.is_alive()
                       for t in threading.enumerate())
    ''')
    env.update(EXPECTED_DATA=str(data), EXPECTED_LOGS=str(logs))
    result = subprocess.run([sys.executable, "-c", script, language, failure],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    if not failure:
        assert f"### {language}:" in result.stdout
    for path, content in sentinels.items():
        assert path.read_bytes() == content
    assert set(data.iterdir()) == {data / "settings.ini", data / "recovery.json"}
    assert set(logs.iterdir()) == {logs / "existing.log"}


def test_audit_rejects_initialized_settings_before_creating_window(tmp_path):
    env = os.environ.copy()
    env.update(OPENFOCUS_DATA_DIR=str(tmp_path), QT_QPA_PLATFORM="offscreen")
    settings = tmp_path / "settings.ini"
    settings.write_bytes(b"[ui]\nlanguage=zh\n")
    script = textwrap.dedent('''
        from utils.settings_store import get_settings
        get_settings()
        from tools.lang_ui_audit import main
        try:
            main()
        except RuntimeError as exc:
            assert "separate process" in str(exc)
        else:
            raise AssertionError("initialized settings were accepted")
        from PyQt6.QtWidgets import QApplication
        assert QApplication.instance() is None
    ''')
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert settings.read_bytes() == b"[ui]\nlanguage=zh\n"
