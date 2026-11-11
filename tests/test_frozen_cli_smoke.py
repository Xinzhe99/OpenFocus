"""Packaged language checks must inspect the controlled application's actions."""
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
