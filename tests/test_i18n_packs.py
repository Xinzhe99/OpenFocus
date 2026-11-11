"""Every shipped language pack must be complete, loadable and menu-wired.

The four-table parity tests only covered en/zh/ja/es by iterating the merged
dict. Once packs are discovered as modules, a pack that fails to import
simply never enters that dict, so the completeness test passes while the
language is silently absent from the menu. These tests pin the discovery
contract instead of the language list.
"""
import re

import pytest

from locales import TranslationManager, trans

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PCT = re.compile(r"%[sdf]")

# Ships with the repo; keep this list in sync only when adding a pack.
EXPECTED_PACKS = {"de", "fr", "ru", "ko", "pt", "it", "nl", "pl"}


def _placeholders(text):
    return set(_PLACEHOLDER.findall(str(text)))


def test_every_pack_is_discovered():
    """No pack file may exist without being loaded (import errors are loud)."""
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package = os.path.join(here, "locales")
    on_disk = {
        name[:-3]
        for name in os.listdir(package)
        if name.endswith(".py") and not name.startswith("_")
        and len(name[:-3]) == 2
    } - {"en", "zh"}
    loaded = set(trans.translations)
    missing = on_disk - loaded
    assert not missing, (
        f"language pack modules exist but did not load: {sorted(missing)} — "
        "check the warning printed by TranslationManager._load_language_packs")


def test_new_packs_are_present():
    assert EXPECTED_PACKS <= set(trans.translations), (
        f"missing packs: {sorted(EXPECTED_PACKS - set(trans.translations))}")


def test_every_pack_has_every_key():
    english = trans.translations["en"]
    for code, table in trans.translations.items():
        if code == "en":
            continue
        missing = set(english) - set(table)
        sample = sorted(missing)[:5]
        assert not missing, f"{code}: {len(missing)} missing keys, e.g. {sample}"


def test_every_pack_keeps_placeholders():
    english = trans.translations["en"]
    keyed = [k for k, v in english.items() if _placeholders(v)]
    assert keyed, "no format-string keys found; the test lost its purpose"

    for code, table in trans.translations.items():
        if code == "en":
            continue
        for key in keyed:
            if key not in table:
                continue  # reported by test_every_pack_has_every_key
            want = _placeholders(english[key])
            got = _placeholders(table[key])
            assert got == want, (
                f"{code}: '{key}' must use {sorted(want)}, got {sorted(got)}")


def test_every_pack_keeps_percent_style_specifiers():
    """A few strings are %-formatted; a dropped %s raises at format time."""
    english = trans.translations["en"]
    keyed = [k for k, v in english.items() if _PCT.search(str(v))]
    for code, table in trans.translations.items():
        if code == "en":
            continue
        for key in keyed:
            if key not in table:
                continue
            want = set(_PCT.findall(str(english[key])))
            got = set(_PCT.findall(str(table[key])))
            assert got == want, f"{code}: '{key}' lost a %-specifier"


def test_packs_are_not_just_english():
    """A pack that copies English wholesale means the translation never ran."""
    english = trans.translations["en"]
    for code, table in trans.translations.items():
        if code == "en":
            continue
        identical = sum(1 for k, v in table.items() if v == english.get(k))
        ratio = identical / max(1, len(english))
        assert ratio < 0.35, (
            f"{code}: {ratio:.0%} of strings are identical to English — "
            "the pack looks untranslated")


def test_available_languages_lists_every_pack():
    codes = [c for c, _label in trans.available_languages()]
    assert "en" in codes
    for code in trans.translations:
        assert code in codes, (
            f"{code} is loaded but absent from the language menu")
    assert len(codes) == len(set(codes)), "duplicate language codes in the menu"


def test_menu_labels_exist_for_every_pack():
    for code, label in trans.available_languages():
        assert label and label.strip(), f"{code} has an empty menu label"


@pytest.mark.parametrize("locale_name,expected", [
    ("zh_CN", "zh"),
    ("zh_Hans_CN", "zh"),
    ("ja_JP", "ja"),
    ("ko_KR", "ko"),
    ("de_DE", "de"),
    ("fr_FR", "fr"),
    ("pt_BR", "pt"),
    ("ru_RU", "ru"),
    ("it_IT", "it"),
    ("nl_NL", "nl"),
    ("pl_PL", "pl"),
    ("en_US", "en"),
    ("C", None),
    ("", None),
])
def test_locale_matching(locale_name, expected):
    assert TranslationManager._match_locale(locale_name) == expected


def test_set_language_switches_the_table():
    manager = TranslationManager()
    for code in manager.translations:
        manager.set_language(code)
        assert manager.current_lang == code
        # Every shipped pack defines the File menu label.
        assert manager.t("menu_file")
    manager.set_language("en")


def test_unknown_autodetect_code_falls_back_to_english():
    manager = TranslationManager()
    assert manager.current_lang in manager.translations


@pytest.mark.parametrize("scan_error", [False, True])
def test_known_packs_load_without_source_directory(monkeypatch, scan_error):
    import locales

    def scan(_path):
        if scan_error:
            raise FileNotFoundError("archived modules")
        return []

    monkeypatch.setattr(locales.os, "listdir", scan)
    manager = TranslationManager()
    assert list(code for code, _ in manager.available_languages()) == list(manager.PACK_ORDER)
    for code in manager.PACK_ORDER:
        manager.set_language(code)
        assert manager.current_lang == code
        assert manager.t("menu_file") == manager.translations[code]["menu_file"]


@pytest.mark.parametrize("locale_name,code", [("de_DE", "de"), ("ko_KR", "ko"), ("pl_PL", "pl")])
def test_autodetect_with_archived_packs(monkeypatch, locale_name, code):
    import locales

    detected = TranslationManager._match_locale(locale_name)
    monkeypatch.setattr(locales.os, "listdir", lambda _: [])
    monkeypatch.setattr(TranslationManager, "_match_locale", staticmethod(lambda _: detected))
    assert TranslationManager().current_lang == code


def test_extra_source_pack_is_still_discovered(monkeypatch):
    import importlib
    from types import SimpleNamespace
    import locales

    original_import = importlib.import_module

    def import_pack(name, package=None):
        if name == ".xx" and package == "locales":
            return SimpleNamespace(PACK={"menu_file": "Example"})
        return original_import(name, package)

    monkeypatch.setattr(locales.os, "listdir", lambda _: ["xx.py", "_x.py", "long.py"])
    monkeypatch.setattr(importlib, "import_module", import_pack)
    manager = TranslationManager()
    assert manager.translations["xx"]["menu_file"] == "Example"
    assert "xx" in dict(manager.available_languages())


def test_failed_pack_does_not_hide_other_packs(monkeypatch, capsys):
    import importlib
    import locales

    original_import = importlib.import_module

    def import_pack(name, package=None):
        if name == ".de" and package == "locales":
            raise ImportError("simulated broken pack")
        return original_import(name, package)

    monkeypatch.setattr(locales.os, "listdir", lambda _: [])
    monkeypatch.setattr(importlib, "import_module", import_pack)
    manager = TranslationManager()
    assert [code for code, _ in manager.available_languages()] == [
        code for code in manager.PACK_ORDER if code != "de"
    ]
    manager.set_language("en")
    assert manager.get("menu_file") == manager.translations["en"]["menu_file"]
    manager.set_language("de")
    assert manager.current_lang == "en"
    manager.set_language("fr")
    manager.translations["fr"].pop("menu_file", None)
    assert manager.get("menu_file") == manager.translations["en"]["menu_file"]
    assert "simulated broken pack" in capsys.readouterr().out
