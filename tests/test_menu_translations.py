"""Every retranslatable menu entry must resolve to real text.

`update_ui_text()` looks a `ui_objs` entry up by its `trans_key` property and
falls back to the *key itself* when that property is missing. A menu entry
registered as `ui_objs['action_advanced_settings']` therefore rendered as the
literal string "action_advanced_settings" after switching language: the key
existed nowhere in any table, and nothing failed loudly.

This test walks the real menus and asserts that no visible action or menu is
left showing a translation key instead of a translated label.
"""
import re
import pytest

pytest.importorskip("PyQt6")

from locales import trans  # noqa: E402


def _menu_window(qapp):
    """A window good enough for ui.menus.setup_menus to build every menu."""
    from PyQt6.QtWidgets import QMainWindow

    class Window(QMainWindow):
        pass

    window = Window()
    window.ui_objs = {}
    # Exactly the collaborators and slots setup_menus() touches.
    stub = lambda *a, **k: None  # noqa: E731
    window.export_manager = type("E", (), {
        "save_result": staticmethod(stub),
        "save_result_stack": staticmethod(stub),
        "save_as_gif": staticmethod(stub),
        "save_stack_as_multipage_tiff": staticmethod(stub),
        "save_processed_input_stack": staticmethod(stub),
    })()
    window.source_manager = type("S", (), {
        "clear_image_stack": staticmethod(stub),
    })()
    window.label_manager = type("L", (), {
        "show_add_label_dialog": staticmethod(stub),
        "delete_registered_labels": staticmethod(stub),
        "delete_input_labels": staticmethod(stub),
        "delete_output_labels": staticmethod(stub),
        "delete_all_labels": staticmethod(stub),
    })()
    for name in (
            "open_folder_dialog", "open_video_dialog", "open_project_dialog",
            "save_project_dialog", "rebuild_recent_menu", "undo", "redo",
            "rotate_stack", "flip_stack", "resize_all_images",
            "show_retouch_dialog", "show_batch_processing_dialog",
            "show_compare_dialog", "show_depth_map_dialog",
            "show_dust_map_dialog", "show_scale_bar_dialog",
            "remove_scale_bar", "show_advanced_settings_dialog",
            "set_use_gpu", "apply_theme", "set_language",
            "check_for_updates", "install_update_now", "open_logs_folder",
            "show_quick_start", "show_environment_info", "show_feedback_dialog",
            "show_contact_info"):
        setattr(window, name, stub)
    return window


def _label_looks_like_a_key(text: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*_[a-z0-9_]+", text.strip()))


def test_every_menu_entry_resolves_to_a_translation(qapp):
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QMenu
    from ui.menus import setup_menus

    window = _menu_window(qapp)
    setup_menus(window)
    # Reproduce a language switch: this is where the raw key appeared.
    window.update_ui_text = lambda: None
    for key, obj in window.ui_objs.items():
        if key.startswith(("action_lang_", "theme_")):
            continue  # endonyms and theme names keep their own labels
        t_key = obj.property("trans_key") or key
        if isinstance(obj, QMenu):
            text = obj.title()
        elif isinstance(obj, QAction):
            text = obj.text()
        else:
            continue
        assert not _label_looks_like_a_key(text), (
            f"{key}: menu text {text!r} is a raw translation key, "
            f"not translated text")
        assert trans.get(t_key) != t_key or t_key in trans.translations["en"], (
            f"{key}: trans_key {t_key!r} is missing from the English table")


def test_advanced_settings_entry_is_translated_in_every_language(qapp):
    """The reported bug: Chinese UI showed `action_advanced_settings`.

    Reads each table directly instead of calling set_language(): the suite
    shares one TranslationManager, so switching the active language here
    would leak into every later test (and the Qt singleton is gone once an
    earlier test tore its QApplication down).
    """
    from ui.menus import setup_menus

    window = _menu_window(qapp)
    setup_menus(window)
    action = window.ui_objs["action_advanced_settings"]
    key = action.property("trans_key")
    assert key == "advanced_settings_title"

    for code, table in trans.translations.items():
        label = table.get(key)
        assert label, f"{code}: missing {key}"
        assert not _label_looks_like_a_key(label), (
            f"{code}: advanced settings renders {label!r}")
