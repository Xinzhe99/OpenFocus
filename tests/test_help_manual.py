"""Help opens the published manual in the language selected at click time."""
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QDesktopServices

from locales import trans
from tests.test_menu_translations import _menu_window
from ui.menus import setup_menus


def test_help_opens_localized_github_manual(qapp, monkeypatch):
    opened = []
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()))
    monkeypatch.setattr(trans, 'current_lang', 'zh')
    window = _menu_window(qapp)
    try:
        setup_menus(window)
        actions = window.ui_objs['menu_help'].actions()
        manual = window.ui_objs['menu_github_manual']
        assert manual in actions and manual.isVisible()
        assert manual.text() == '用户手册 (GitHub)'
        assert 'menu_quick_start' not in window.ui_objs
        assert window.ui_objs['menu_guided_tour'] in actions
        assert window.ui_objs['menu_guided_tour'].isVisible()

        for language in (*trans.PACK_ORDER, 'xx'):
            monkeypatch.setattr(trans, 'current_lang', language)
            manual.trigger()
            code = 'EN' if language == 'xx' else language.upper()
            assert opened == [f'https://github.com/Xinzhe99/OpenFocus/blob/main/docs/USER_MANUAL_{code}.md']
            opened.clear()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
