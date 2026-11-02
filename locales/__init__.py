
import datetime
import os

from PyQt6.QtCore import QObject, pyqtSignal

from .en import EN
from .zh import ZH

class TranslationManager(QObject):
    languageChanged = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_lang = 'en'
        
        # Auto-detect from the system locale. The lookup is table-driven off
        # the packs that actually loaded, so adding a pack module is enough to
        # make 'xx' selectable AND auto-detected; nothing here needs editing.
        # The legacy UTC+8 fallback stays for hosts where QLocale reports an
        # unusable value (some stripped Linux images answer 'C').
        detected = None
        try:
            from PyQt6.QtCore import QLocale
            name = QLocale.system().name() or ""  # e.g. 'zh_CN', 'pt_BR'
            detected = self._match_locale(name)
        except Exception:
            detected = None
        if detected is None:
            try:
                offset = datetime.datetime.now().astimezone().utcoffset()
                if offset is not None and int(offset.total_seconds()) == 28800:
                    detected = 'zh'
            except Exception:
                pass
        self._pending_autodetect = detected

        self.translations = {"en": dict(EN), "zh": dict(ZH)}

        self._load_language_packs()

        # Apply the detected language only now: the packs had to exist first,
        # otherwise a locale whose pack failed to load would leave
        # current_lang pointing at a table that is not in the dict.
        pending = getattr(self, "_pending_autodetect", None)
        if pending and pending in self.translations:
            self.current_lang = pending
        self._pending_autodetect = None

    # -- language packs ------------------------------------------------------

    # Ordered so the menu lists languages consistently no matter what the
    # filesystem returns. 'en'/'zh' live in en.py/zh.py; 'ja'/'es' ship in
    # _extra.py; everything else is one <code>.py module in this package.
    PACK_ORDER = ("en", "zh", "ja", "es", "de", "fr", "ru", "ko", "pt", "it",
                  "nl", "pl")
    LANG_NAMES = {
        "en": "English", "zh": "中文", "ja": "日本語", "es": "Español",
        "de": "Deutsch", "fr": "Français", "ru": "Русский",
        "ko": "한국어", "pt": "Português", "it": "Italiano",
        "nl": "Nederlands", "pl": "Polski",
    }

    @staticmethod
    def _match_locale(name: str):
        """Map a QLocale name ('pt_BR', 'zh_Hans_CN') onto a pack code."""
        if not name:
            return None
        low = name.lower().replace("-", "_")
        primary = low.split("_")[0]
        # QLocale answers 'C'/'POSIX' when no user locale is configured; there
        # is no such pack, and treating it as a code would blank the UI.
        if primary in ("c", "posix"):
            return None
        # QLocale marks simplified Chinese as 'zh_Hans_CN'; both scripts use
        # the same zh pack, which is Simplified.
        if primary in ("zh", "ja", "ko"):
            return primary
        # Portuguese: one pack covers both, labelled pt-BR in the source text.
        if primary == "pt":
            return "pt"
        return primary or None

    def _load_language_packs(self) -> None:
        """Merge every community pack into self.translations.

        Packs are discovered as modules rather than hardcoded: adding a
        language means dropping a <code>.py file into this package with
        a PACK dict. A broken pack must never be swallowed silently — the
        completeness tests iterate the merged table, so a bare `pass` once let
        the whole Japanese pack vanish while every test stayed green.
        """
        import importlib

        loaded = {"en", "zh"}  # imported at the top of this module

        # _extra.py keeps ja/es in one dict for the original release.
        try:
            from ._extra import EXTRA_TRANSLATIONS
            for lang_code, lang_dict in EXTRA_TRANSLATIONS.items():
                self.translations.setdefault(lang_code, {}).update(lang_dict)
                loaded.add(lang_code)
        except Exception as exc:  # noqa: BLE001 - report, never hide
            print(f"Warning: extra language pack failed to load: {exc}")

        # One module per language: <code>.py inside this package with a PACK dict.
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            files = sorted(os.listdir(here))
        except OSError as exc:
            print(f"Warning: cannot scan for language packs: {exc}")
            return
        for fname in files:
            if not fname.endswith(".py"):
                continue
            code = fname[:-len(".py")]
            # _extra.py holds ja/es; en/zh ship as full tables above;
            # __init__.py is this manager. None is a standalone pack.
            if code.startswith("_") or len(code) != 2 or code in loaded:
                continue
            try:
                module = importlib.import_module(f".{code}", __name__)
                pack = getattr(module, "PACK", None)
                if not isinstance(pack, dict) or not pack:
                    raise ValueError("module has no non-empty PACK dict")
                self.translations.setdefault(code, {}).update(pack)
                loaded.add(code)
            except Exception as exc:  # noqa: BLE001 - report, never hide
                print(f"Warning: language pack '{code}' failed to load: {exc}")

    def available_languages(self):
        """[(code, display_name)] for every fully loaded pack, menu order."""
        order = {c: i for i, c in enumerate(self.PACK_ORDER)}
        codes = [c for c in self.translations if c not in ("en",)]
        codes.sort(key=lambda c: (order.get(c, 999), c))
        return [("en", self.LANG_NAMES["en"])] + [
            (c, self.LANG_NAMES.get(c, c.upper())) for c in codes]

    def set_language(self, lang_code):
        if lang_code in self.translations:
            self.current_lang = lang_code
            self.languageChanged.emit()

    def get(self, key, default=None):
        val = self.translations.get(self.current_lang, {}).get(key)
        if val is None:
            # Fallback to english
            val = self.translations.get('en', {}).get(key)
        return val if val is not None else (default if default is not None else key)
    
    def t(self, key):
        return self.get(key)

# Global instance
trans = TranslationManager()
