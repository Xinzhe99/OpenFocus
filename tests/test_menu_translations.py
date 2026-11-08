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
        "reset_current_stack": staticmethod(stub),
    })()
    window.label_manager = type("L", (), {
        "show_add_label_dialog": staticmethod(stub),
        "delete_registered_labels": staticmethod(stub),
        "delete_input_labels": staticmethod(stub),
        "delete_output_labels": staticmethod(stub),
        "delete_all_labels": staticmethod(stub),
    })()
    window.render_manager = type("R", (), {
        "show_alignment_report": staticmethod(stub),
    })()
    for name in (
            "open_folder_dialog", "open_video_dialog", "open_project_dialog",
            "save_project_dialog", "open_template_dialog", "save_template_dialog",
            "rebuild_recent_menu", "undo", "redo",
            "rotate_stack", "flip_stack", "resize_all_images",
            "show_retouch_dialog", "show_standalone_retouch", "show_batch_processing_dialog",
            "show_compare_dialog", "show_mosaic_dialog", "show_depth_map_dialog",
            "show_dust_map_dialog", "show_scale_bar_dialog",
            "remove_scale_bar", "show_advanced_settings_dialog",
            'restore_default_settings', 'set_use_gpu', 'apply_theme', 'set_language',
            "check_for_updates", "install_update_now", "open_logs_folder",
            "show_quick_start", "show_environment_info", "show_feedback_dialog",
            "show_contact_info"):
        setattr(window, name, stub)
    return window


def test_file_menu_uses_one_folder_entry(qapp):
    from ui.menus import setup_menus

    window = _menu_window(qapp)
    calls = []
    window.open_folder_dialog = lambda: calls.append(True)
    setup_menus(window)
    try:
        assert 'linear_raw_open' not in window.ui_objs
        action = window.ui_objs['action_open_folder']
        assert action in window.ui_objs['menu_file'].actions()
        action.trigger()
        assert calls == [True]
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()



def test_usage_switch_is_only_in_advanced_settings(qapp):
    from ui.menus import setup_menus

    window = _menu_window(qapp)
    setup_menus(window)
    try:
        assert 'usage_opt_in' not in window.ui_objs
        assert window.ui_objs['action_advanced_settings'] in window.ui_objs['menu_settings'].actions()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_usage_disclosure_and_switch_exist_in_every_language():
    for code in trans.PACK_ORDER:
        table = trans.translations[code]
        assert table.get('usage_enabled'), code
        assert table.get('usage_disclosure'), code
        assert 'usage_opt_in' not in table, code


def test_reset_stack_menu_action_calls_source_manager(qapp):
    from ui.menus import setup_menus

    window = _menu_window(qapp)
    calls = []
    window.source_manager.reset_current_stack = lambda: calls.append(True)
    setup_menus(window)
    try:
        action = window.ui_objs["action_reset_stack"]
        actions = window.ui_objs["menu_edit"].actions()
        assert actions.index(action) == actions.index(window.ui_objs["action_redo"]) + 1
        assert action.property("trans_key") == "menu_reset_stack"
        action.trigger()
        assert calls == [True]
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_reset_stack_translations_exist_in_every_language():
    keys = ("menu_reset_stack", "reset_stack_title", "reset_stack_warning",
            "reset_stack_busy", "reset_stack_empty", "reset_stack_done")
    assert set(trans.PACK_ORDER) <= set(trans.translations)
    for code in trans.PACK_ORDER:
        for key in keys:
            assert trans.translations[code].get(key), f"{code}: missing {key}"


@pytest.mark.parametrize("action_index, expected", [
    (0, [[4, 1], [5, 2], [6, 3]]),
    (1, [[3, 6], [2, 5], [1, 4]]),
    (2, [[6, 5, 4], [3, 2, 1]]),
])
def test_rotation_menu_rotates_frames_in_the_named_direction(qapp, action_index, expected):
    import numpy as np
    from controllers.transform_manager import TransformManager
    from ui.menus import setup_menus

    window = _menu_window(qapp)
    original = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    window.raw_images = [original.copy(), original + 1000]
    window.base_images = [image.copy() for image in window.raw_images]
    window.source_transforms = []
    window.current_img_index = 0
    window.current_display_index = 1
    window.push_undo = lambda: None
    manager = TransformManager(window)
    manager._invalidate_processing_results = lambda **kwargs: None
    reloaded = []
    manager.reload_image_stack = lambda **kwargs: reloaded.append(kwargs["initial_index"])
    window.rotate_stack = manager.rotate_stack
    setup_menus(window)

    try:
        window.ui_objs["menu_rotate"].actions()[action_index].trigger()
        expected = np.array(expected, dtype=np.uint16)
        for images in (window.raw_images, window.base_images):
            np.testing.assert_array_equal(images[0], expected)
            np.testing.assert_array_equal(images[1], expected + 1000)
        assert reloaded == [1]
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


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


def test_staged_update_menu_keeps_its_version_after_a_language_sweep(qapp):
    """`menu_install_update` contains {version}; the generic retranslation
    sweep writes that placeholder literally. The format-aware refresh must
    restore the staged tag — including when the download candidate differs.
    """
    import types

    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import (QLabel, QProgressBar, QPushButton,
                                 QStatusBar, QWidget)

    import main
    from controllers.update_manager import STATE_READY

    class _Manager:
        state = STATE_READY
        tag = "v9.8"          # candidate offered by the check
        percent = 100

        @staticmethod
        def staged_tag():
            return "v9.9"     # what is actually downloaded and verified

    class _Host:
        def __init__(self):
            self.update_manager = _Manager()
            self.ui_objs = {
                'action_install_update':
                    QAction(trans.t('menu_install_update'))}
            self.btn_update_action = QPushButton()
            self.lbl_update_status = QLabel()
            self.update_progress = QProgressBar()
            self._update_indicator = QWidget()
            self._bar = QStatusBar()

        def statusBar(self):
            return self._bar

    host = _Host()
    action = host.ui_objs['action_install_update']
    # What update_ui_text's sweep leaves before _refresh_update_indicator runs.
    action.setText(trans.t('menu_install_update'))
    types.MethodType(main.OpenFocus._refresh_update_indicator, host)()

    assert action.text() == trans.t('menu_install_update').format(version="v9.9")
    assert "{version}" not in action.text()
