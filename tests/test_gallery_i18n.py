"""Strict 12-language parity and formatting tests for locales.gallery_i18n."""
from __future__ import annotations

import re

import pytest

from locales import TranslationManager, trans
from locales.gallery_i18n import (
    AI_TAG_KEYS,
    BADGE_KEYS,
    CONTENT_REJECTION_REASON_KEYS,
    ERROR_CODE_KEYS,
    GALLERY_LANGUAGES,
    GALLERY_TRANSLATIONS,
    INTEREST_KEYS,
    LEADERBOARD_PERIOD_KEYS,
    NOTIFICATION_TYPE_KEYS,
    REPORT_REASON_KEYS,
    format_notification,
    t_gallery,
    translate_badge,
    translate_error,
    translate_period,
    translate_rejection_reason,
    translate_report_reason,
    translate_tag,
)

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::[^}]*)?\}")


def _placeholders(text: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(str(text)))


def test_exact_12_openfocus_languages_present():
    assert set(GALLERY_LANGUAGES) == set(TranslationManager.PACK_ORDER)
    assert set(GALLERY_TRANSLATIONS.keys()) == set(GALLERY_LANGUAGES)
    assert len(GALLERY_TRANSLATIONS) == 12


def test_strict_key_parity_and_non_empty_across_all_12_languages():
    english = GALLERY_TRANSLATIONS["en"]
    assert english, "English gallery table must not be empty"

    for lang in GALLERY_LANGUAGES:
        table = GALLERY_TRANSLATIONS[lang]
        missing = set(english) - set(table)
        extra = set(table) - set(english)
        assert not missing, f"{lang}: missing keys {sorted(missing)[:5]}"
        assert not extra, f"{lang}: extra keys {sorted(extra)[:5]}"

        empty = [k for k, v in table.items() if not isinstance(v, str) or not v.strip()]
        assert not empty, f"{lang}: empty translations for {empty[:5]}"


def test_strict_placeholder_parity_across_all_12_languages():
    english = GALLERY_TRANSLATIONS["en"]
    keyed = [k for k, v in english.items() if _placeholders(v)]
    assert keyed, "expected format-string keys in gallery translations"

    for lang in GALLERY_LANGUAGES:
        if lang == "en":
            continue
        table = GALLERY_TRANSLATIONS[lang]
        for key in keyed:
            want = _placeholders(english[key])
            got = _placeholders(table[key])
            assert got == want, (
                f"{lang}: '{key}' placeholder mismatch: expected {sorted(want)}, got {sorted(got)}"
            )


def test_non_english_packs_are_genuinely_translated():
    english = GALLERY_TRANSLATIONS["en"]
    for lang, table in GALLERY_TRANSLATIONS.items():
        if lang == "en":
            continue
        identical = sum(1 for k, v in table.items() if v == english[k])
        ratio = identical / len(english)
        assert ratio < 0.15, (
            f"{lang}: {ratio:.0%} of strings match English verbatim — looks untranslated"
        )


def test_canonical_api_keys_have_all_12_translations():
    assert len(AI_TAG_KEYS) == 13
    assert len(INTEREST_KEYS) == 12
    assert set(INTEREST_KEYS) == set(AI_TAG_KEYS) - {"other"}
    assert len(REPORT_REASON_KEYS) == 7
    assert len(BADGE_KEYS) == 9
    assert len(CONTENT_REJECTION_REASON_KEYS) == 5
    assert len(ERROR_CODE_KEYS) == 35
    assert len(LEADERBOARD_PERIOD_KEYS) == 6

    for lang in GALLERY_LANGUAGES:
        for k in AI_TAG_KEYS:
            val = translate_tag(k, lang=lang)
            assert val and val != f"gallery_tag_{k}"
        for k in REPORT_REASON_KEYS:
            val = translate_report_reason(k, lang=lang)
            assert val and val != f"gallery_report_reason_{k}"
        for k in BADGE_KEYS:
            val = translate_badge(k, lang=lang)
            assert val and val != f"gallery_badge_{k}"
        for k in CONTENT_REJECTION_REASON_KEYS:
            val = translate_rejection_reason(k, lang=lang)
            assert val and val != f"gallery_reject_reason_{k}"
        for k in ERROR_CODE_KEYS:
            val = translate_error(k, lang=lang)
            assert val and val != f"gallery_err_{k}"
        for k in LEADERBOARD_PERIOD_KEYS:
            val = translate_period(k, lang=lang)
            assert val and val != f"gallery_period_{k}"


def test_showcase_danmaku_screensaver_interop_keys_in_all_12_locales():
    required_keys = (
        "gallery_badge_ad",
        "gallery_badge_following",
        "gallery_close",
        "gallery_comments_author_badge",
        "gallery_comments_delete",
        "gallery_comments_empty",
        "gallery_danmaku_all_comments",
        "gallery_danmaku_cancel",
        "gallery_danmaku_cooldown",
        "gallery_danmaku_downvote",
        "gallery_danmaku_placeholder",
        "gallery_danmaku_report",
        "gallery_danmaku_reported",
        "gallery_danmaku_send",
        "gallery_danmaku_upvote",
        "gallery_empty_state",
        "gallery_report_title",
        "gallery_screensaver_press_esc",
    )
    for lang in GALLERY_LANGUAGES:
        for key in required_keys:
            assert key in GALLERY_TRANSLATIONS[lang], f"{lang} missing {key}"
            assert GALLERY_TRANSLATIONS[lang][key].strip()
        assert "{" not in t_gallery("gallery_danmaku_placeholder", lang=lang)
        assert "7" in t_gallery("gallery_danmaku_all_comments", lang=lang, count=7)
        assert "10" in t_gallery("gallery_danmaku_cooldown", lang=lang, seconds=10)


@pytest.mark.parametrize("lang", GALLERY_LANGUAGES)
def test_format_notification_renders_every_type_without_leaked_placeholders(lang):
    for notif_type in NOTIFICATION_TYPE_KEYS:
        payload = {
            "type": notif_type,
            "actor_name": "Ada",
            "actor_count": 4,
            "work_title": "Jumping Spider",
            "period": "7d",
            "rank": 1,
            "points": 200,
            "reason": "nsfw_explicit",
            "detail": "Maintenance window",
        }
        rendered = format_notification(payload, lang=lang)
        assert rendered and rendered.strip()
        assert "{" not in rendered and "}" not in rendered

    # Singular work_liked case
    single = format_notification(
        {"type": "work_liked", "actor_name": "Ada", "actor_count": 1, "work_title": "Fern"},
        lang=lang,
    )
    assert "Ada" in single and "Fern" in single
    assert "{" not in single


def test_t_gallery_delegates_to_trans_when_registered_and_falls_back():
    prev_lang = trans.current_lang
    try:
        trans.set_language("zh")
        assert t_gallery("gallery_tab_square") == GALLERY_TRANSLATIONS["zh"]["gallery_tab_square"]
        assert t_gallery("gallery_tab_square", lang="ja") == GALLERY_TRANSLATIONS["ja"]["gallery_tab_square"]
        assert "500" in t_gallery("gallery_honor_card_reward_points", lang="en", points=500)

        # Registering a custom override in trans should be respected by t_gallery
        trans.translations["zh"]["_test_gallery_override_key"] = "自定义覆盖 {n}"
        try:
            assert t_gallery("_test_gallery_override_key", n=7) == "自定义覆盖 7"
        finally:
            trans.translations["zh"].pop("_test_gallery_override_key", None)
    finally:
        trans.set_language(prev_lang)


def test_lazy_gallery_translations_defers_inactive_packs():
    from locales.gallery_i18n import _LAZY_PACK_BUILDERS, _LazyGalleryTranslations

    built = []

    def make_builder(code):
        def _build():
            built.append(code)
            return {"k": f"val-{code}"}
        return _build

    lazy = _LazyGalleryTranslations(
        {"en": {"k": "val-en"}},
        {lang: make_builder(lang) for lang in GALLERY_LANGUAGES if lang != "en"},
    )
    assert len(lazy) == len(_LAZY_PACK_BUILDERS) + 1
    assert "zh" in lazy
    assert list(lazy.keys()) == list(GALLERY_LANGUAGES)
    assert built == []

    assert lazy.get("zh") == {"k": "val-zh"}
    assert built == ["zh"]
    assert lazy["zh"] == {"k": "val-zh"}
    assert built == ["zh"]


def test_community_naming_across_all_12_locales():
    import re
    from locales import _GALLERY_EXTRA_KEYS, _GALLERY_EXTRA_TEXT

    forbidden = re.compile(
        r"gallery|galerie|galer[ií]a|galleria|galerij|галере|画廊|ギャラリー|갤러리",
        re.I,
    )
    for lang in GALLERY_LANGUAGES:
        table = GALLERY_TRANSLATIONS[lang]
        for key, value in table.items():
            assert not forbidden.search(value), f"Legacy gallery term in {lang}.{key}: {value!r}"
        extra = dict(zip(_GALLERY_EXTRA_KEYS, _GALLERY_EXTRA_TEXT[lang]))
        for key, value in extra.items():
            assert not forbidden.search(value), f"Legacy gallery term in extra {lang}.{key}: {value!r}"

    assert GALLERY_TRANSLATIONS["zh"]["menu_gallery"] == "社区"
    assert GALLERY_TRANSLATIONS["en"]["menu_gallery"] == "Community"
    assert GALLERY_TRANSLATIONS["zh"]["menu_gallery_hub"] == "OpenFocus 社区"
    assert GALLERY_TRANSLATIONS["en"]["menu_gallery_hub"] == "OpenFocus Community"
