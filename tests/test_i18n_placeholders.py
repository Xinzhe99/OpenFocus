"""Format-string invariants for the translation tables.

A locale that drops a {placeholder} raises KeyError the moment that dialog
opens, and the missing-key tests cannot see it: the key is present.
"""
import re

from locales import trans

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _placeholders(table, key):
    return set(_PLACEHOLDER.findall(str(table.get(key, ""))))


def test_every_language_keeps_the_placeholders():
    english = trans.translations['en']
    keyed = [k for k, v in english.items() if _PLACEHOLDER.search(str(v))]
    assert keyed, "no format-string keys found; the test has lost its purpose"

    for lang, table in trans.translations.items():
        if lang == 'en':
            continue
        for key in keyed:
            assert _placeholders(table, key) == _placeholders(english, key), (
                f"{lang}: '{key}' must use {_placeholders(english, key)}, "
                f"got {_placeholders(table, key)}")
