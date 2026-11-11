
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
        
        # Auto-detect from the packs that actually loaded. Register shipped
        # languages in PACK_ORDER; source installations may discover extras.
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
        support_text = {
            'en': ('Support Us', 'Thank you for supporting OpenFocus! Include your username in the transfer note to receive permanent acknowledgment in the project.'),
            'zh': ('支持我们', '感谢您支持 OpenFocus！请在转账备注中填写您的用户名，项目将永久鸣谢您。'),
            'ja': ('応援する', 'OpenFocus へのご支援ありがとうございます。送金時のメモにユーザー名を記入すると、プロジェクトに永続的に謝辞が掲載されます。'),
            'es': ('Apóyanos', '¡Gracias por apoyar OpenFocus! Incluye tu nombre de usuario en la nota de la transferencia para recibir un reconocimiento permanente en el proyecto.'),
            'de': ('Unterstützen', 'Vielen Dank für Ihre Unterstützung von OpenFocus! Geben Sie Ihren Benutzernamen im Verwendungszweck an, um dauerhaft im Projekt erwähnt zu werden.'),
            'fr': ('Nous soutenir', 'Merci de soutenir OpenFocus ! Indiquez votre nom d’utilisateur dans la note du transfert pour recevoir une mention permanente dans le projet.'),
            'ru': ('Поддержать нас', 'Спасибо за поддержку OpenFocus! Укажите имя пользователя в комментарии к переводу, чтобы получить постоянную благодарность в проекте.'),
            'ko': ('후원하기', 'OpenFocus를 후원해 주셔서 감사합니다! 송금 메모에 사용자 이름을 적어 주시면 프로젝트에 영구적으로 감사 인사를 남깁니다.'),
            'pt': ('Apoie-nos', 'Obrigado por apoiar o OpenFocus! Inclua seu nome de usuário na nota da transferência para receber um agradecimento permanente no projeto.'),
            'it': ('Sostienici', 'Grazie per sostenere OpenFocus! Inserisci il tuo nome utente nella nota del trasferimento per ricevere un ringraziamento permanente nel progetto.'),
            'nl': ('Steun ons', 'Bedankt voor uw steun aan OpenFocus! Vermeld uw gebruikersnaam in de omschrijving van de overschrijving voor een blijvende vermelding in het project.'),
            'pl': ('Wesprzyj nas', 'Dziękujemy za wsparcie OpenFocus! Podaj nazwę użytkownika w tytule przelewu, aby otrzymać trwałe podziękowanie w projekcie.'),
        }
        for code, (title, message) in support_text.items():
            if code in self.translations:
                self.translations[code].update(support_title=title, support_message=message)
        labels = {
            'en': ('Radius:', 'Smoothing:', 'Feather:', 'OpenFocus A does not currently use smoothing.'),
            'zh': ('半径：', '平滑度：', '羽化：', 'OpenFocus A 目前不使用平滑度。'),
            'ja': ('半径：', '平滑度：', 'ぼかし：', 'OpenFocus A では現在、平滑度は使用されません。'),
            'es': ('Radio:', 'Suavizado:', 'Desvanecimiento:', 'OpenFocus A aún no utiliza el suavizado.'),
            'de': ('Radius:', 'Glättung:', 'Weiche Kante:', 'OpenFocus A verwendet die Glättung derzeit nicht.'),
            'fr': ('Rayon :', 'Lissage :', 'Contour progressif :', 'OpenFocus A n’utilise pas encore le lissage.'),
            'ru': ('Радиус:', 'Сглаживание:', 'Растушёвка:', 'OpenFocus A пока не использует сглаживание.'),
            'ko': ('반경:', '평활화:', '페더:', 'OpenFocus A는 현재 평활화를 사용하지 않습니다.'),
            'pt': ('Raio:', 'Suavização:', 'Difusão:', 'O OpenFocus A ainda não usa a suavização.'),
            'it': ('Raggio:', 'Smussamento:', 'Sfumatura:', 'OpenFocus A non usa ancora lo smussamento.'),
            'nl': ('Straal:', 'Vloeiend maken:', 'Doezelaar:', 'OpenFocus A gebruikt vloeiend maken nog niet.'),
            'pl': ('Promień:', 'Wygładzanie:', 'Wtapianie:', 'OpenFocus A nie używa jeszcze wygładzania.'),
        }
        for code, values in labels.items():
            if code in self.translations:
                self.translations[code].update(dict(zip(
                    ('label_of_radius', 'label_of_smoothing', 'label_of_feather',
                     'label_of_smoothing_a_hint'), values)))

        upper_bounds = {
            'en': 'at most {size}', 'zh': '最多 {size}', 'ja': '最大 {size}',
            'es': 'como máximo {size}', 'de': 'höchstens {size}',
            'fr': 'au maximum {size}', 'ru': 'не более {size}',
            'ko': '최대 {size}', 'pt': 'no máximo {size}', 'it': 'al massimo {size}',
            'nl': 'maximaal {size}', 'pl': 'co najwyżej {size}',
        }
        for code, text in upper_bounds.items():
            if code in self.translations:
                self.translations[code]['storage_available_upper_bound'] = text

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

        Shipped packs are registered in PACK_ORDER and can be imported from
        archives without source files on disk. Source installations may also
        discover extra two-character modules with a PACK dict. Report broken
        packs without preventing the remaining languages from loading.
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

        # Known packs work even when modules are stored in a frozen archive.
        candidates = set(self.PACK_ORDER) - loaded
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            files = os.listdir(here)
        except OSError as exc:
            print(f"Warning: cannot scan for language packs: {exc}")
            files = ()
        for fname in files:
            if not fname.endswith(".py"):
                continue
            code = fname[:-len(".py")]
            if not code.startswith("_") and len(code) == 2 and code not in loaded:
                candidates.add(code)
        for code in sorted(candidates):
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
