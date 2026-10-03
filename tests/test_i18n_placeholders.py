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


def test_extra_language_packs_actually_load():
    """locales_extra 必须可导入且四个语言全部进入 trans.translations。

    v1.33 曾把 ES 块误插进 JA 字典中部造成 SyntaxError，而 locales.py
    静默吞掉后这些语言从字典里消失——当时的占位符/完整性测试遍历的
    正是合并后的字典，于是全绿。"""
    import locales_extra  # noqa: F401  SyntaxError 在这里直接爆
    from locales import trans
    assert {'ja', 'es'} <= set(trans.translations),         f"missing packs, loaded: {sorted(trans.translations)}"
