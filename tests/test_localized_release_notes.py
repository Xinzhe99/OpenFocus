"""Behavioral regressions for localized, retained in-app release content."""
import pytest

from utils import updater
from utils.release_notes import LANGUAGES, RELEASE_NOTES, release_notes_markdown


@pytest.mark.parametrize("language,phrase", [
    ("en", "Choose a temporary data directory"),
    ("zh", "可为导入帧和处理临时数据选择存储目录"),
    ("ja", "読み込んだフレームと処理用の一時データ"),
    ("es", "Elige un directorio temporal"),
    ("de", "Wählen Sie einen temporären Ordner"),
    ("fr", "Choisissez un répertoire temporaire"),
    ("ru", "Выбирайте временную папку"),
    ("ko", "가져온 프레임과 처리용 임시 데이터"),
    ("pt", "Escolha uma pasta temporária"),
    ("it", "Scegli una cartella temporanea"),
    ("nl", "Kies een tijdelijke map"),
    ("pl", "Wybierz katalog tymczasowy"),
])
def test_current_release_has_actual_localized_content_with_legacy_mirror(language, phrase):
    entries = [{"tag_name": "v1.76", "body": "Legacy English installation boilerplate"}]
    notes = updater.format_release_notes("1.75", "v1.76.0", entries, language)
    assert phrase in notes
    assert "Legacy English installation boilerplate" not in notes
    assert "StackMFF" in notes and "PayPal" in notes
    assert len(notes) > 700
    assert "https://github.com/Xinzhe99/OpenFocus/releases/tag/v1.76" in notes


def test_generated_github_body_round_trips_all_languages(monkeypatch):
    # A future release uses exactly the same generation/consumption protocol.
    monkeypatch.setitem(RELEASE_NOTES, "9.9", dict(RELEASE_NOTES["1.76.0"]))
    body = release_notes_markdown("v9.9") + "\n\n## Downloads\nEnglish installer instructions"
    entries = [{"tag_name": "v9.9", "body": body}]
    for language in LANGUAGES:
        notes = updater.format_release_notes("9.8", "v9.9", entries, language)
        first_line = RELEASE_NOTES["9.9"][language].splitlines()[0][2:]
        assert first_line in notes
        assert "openfocus-notes" not in notes
        assert "English installer instructions" not in notes
        other = "fr" if language != "fr" else "ja"
        assert RELEASE_NOTES["9.9"][other].splitlines()[0][2:] not in notes


def test_release_generation_refuses_untranslated_release(monkeypatch):
    monkeypatch.setitem(RELEASE_NOTES, "9.9", {"en": "An English release note"})
    with pytest.raises(ValueError, match="fr"):
        release_notes_markdown("v9.9")


def test_sort_limit_range_and_padded_version_identity():
    releases = [
        {"tag_name": "v9.2", "body": "middle"},
        {"tag_name": "v9.1.0", "body": "already installed"},
        {"tag_name": "v9.3.0", "body": ""},
        {"tag_name": "v9.3.0", "body": "newest"},
        {"tag_name": "v9.3", "body": "duplicate"},
        {"tag_name": "v9.4", "body": "future"},
        {"tag_name": "not-a-version", "body": "invalid"},
    ]
    notes = updater.format_release_notes("9.1", "v9.3", releases, max_releases=1)
    assert "newest" in notes
    assert all(text not in notes for text in ("middle", "already installed", "duplicate", "future", "invalid"))
    notes = updater.format_release_notes("9.1", "v9.3", releases)
    assert notes.index("newest") < notes.index("middle")
    assert updater.format_release_notes("9.1", "v9.3", releases, max_releases=0) == ""


def test_explicit_translation_fallback_and_source_link_preservation():
    releases = [{"tag_name": "v9.9", "body": "Legacy source text",
                 "body_i18n": {"pt": "Correção de [cores](https://example.org/details)",
                               "en": "Color correction", "fr": 7}}]
    notes = updater.format_release_notes("9.8", "v9.9", releases, "pt_BR")
    assert "Correção de cores (https://example.org/details)" in notes
    assert "Color correction" not in notes
    assert "Color correction" in updater.format_release_notes("9.8", "v9.9", releases, "fr")
    releases[0].pop("body_i18n")
    assert "Legacy source text" in updater.format_release_notes("9.8", "v9.9", releases, "fr")


def test_retained_entries_need_no_network(monkeypatch):
    def offline(*args, **kwargs):
        raise AssertionError("Changing note language must not access the network")
    monkeypatch.setattr(updater, "_api_get", offline)
    entries = updater.release_note_entries("1.75", "v1.76.0", [])
    assert "Choisissez" in updater.fetch_release_notes("1.75", "v1.76.0", releases=entries, language="fr")
    assert "Wybierz" in updater.fetch_release_notes("1.75", "v1.76.0", releases=entries, language="pl")
