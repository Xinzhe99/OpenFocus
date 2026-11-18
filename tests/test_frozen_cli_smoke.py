"""Packaged language checks must inspect the controlled application's actions."""
import json
import os
import subprocess
import sys
import pytest

from locales import TranslationManager
from tools import frozen_cli_smoke as smoke


def test_frozen_language_menu_rejects_a_missing_pack(monkeypatch):
    actions = [
        {"key": f"action_lang_{code}", "text": TranslationManager.LANG_NAMES[code],
         "enabled": True, "checkable": True}
        for code in TranslationManager.PACK_ORDER
    ]
    monkeypatch.setattr(smoke, "_call", lambda *args: {"actions": actions})
    assert len(smoke._language_actions(None, {})) == len(TranslationManager.PACK_ORDER)
    actions.pop()
    with pytest.raises(RuntimeError, match="language menu differs"):
        smoke._language_actions(None, {})


def test_synthetic_session_is_shared_with_child_and_cleared(tmp_path, monkeypatch):
    profile = tmp_path / "isolated profile"
    profile.mkdir()
    sentinel = tmp_path / "real-profile"
    sentinel.mkdir()
    (sentinel / "account-session").write_bytes(b"untouched")
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(sentinel))
    environment = os.environ.copy()
    environment["OPENFOCUS_DATA_DIR"] = str(profile)
    command = [sys.executable, "-c", "import json; from utils.account_client import TokenStore; print(json.dumps(TokenStore().load()))"]
    with pytest.raises(RuntimeError, match="exercise cleanup"):
        with smoke._synthetic_session(profile):
            assert os.environ["OPENFOCUS_DATA_DIR"] == str(sentinel)
            result = subprocess.check_output(command, cwd=smoke.ROOT, env=environment, text=True)
            assert json.loads(result)["user"]["id"] == "synthetic-test-user"
            raise RuntimeError("exercise cleanup")
    result = subprocess.check_output(command, cwd=smoke.ROOT, env=environment, text=True)
    assert json.loads(result) is None
    assert (sentinel / "account-session").read_bytes() == b"untouched"
