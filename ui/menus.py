from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QMainWindow
from locales import trans


def setup_menus(window: QMainWindow) -> None:
    """Configure the main menu bar and attach actions to the window."""
    menubar = window.menuBar()
    
    # Initialize the ui_objs container if not present
    if not hasattr(window, 'ui_objs'):
        window.ui_objs = {}

    # --- File Menu ---
    file_menu = menubar.addMenu(trans.t('menu_file'))
    window.ui_objs['menu_file'] = file_menu

    open_action = QAction(trans.t('action_open_folder'), window)
    open_action.setShortcut("Ctrl+O")
    open_action.triggered.connect(window.open_folder_dialog)
    file_menu.addAction(open_action)
    window.ui_objs['action_open_folder'] = open_action

    open_video_action = QAction(trans.t('action_open_video'), window)
    open_video_action.setShortcut("Ctrl+Shift+O")
    open_video_action.triggered.connect(window.open_video_dialog)
    file_menu.addAction(open_video_action)
    window.ui_objs['action_open_video'] = open_video_action

    open_project_action = QAction(trans.t('menu_open_project'), window)
    open_project_action.setShortcut("Ctrl+Alt+O")
    open_project_action.triggered.connect(window.open_project_dialog)
    file_menu.addAction(open_project_action)
    window.ui_objs['menu_open_project'] = open_project_action

    save_project_action = QAction(trans.t('menu_save_project'), window)
    save_project_action.setShortcut("Ctrl+Alt+S")
    save_project_action.triggered.connect(lambda: window.save_project_dialog(save_as=False))
    file_menu.addAction(save_project_action)
    window.ui_objs['menu_save_project'] = save_project_action

    save_project_as_action = QAction(trans.t('menu_save_project_as'), window)
    save_project_as_action.triggered.connect(lambda: window.save_project_dialog(save_as=True))
    file_menu.addAction(save_project_as_action)
    window.ui_objs['menu_save_project_as'] = save_project_as_action

    file_menu.addSeparator()

    # Recent files submenu (items filled in by window.rebuild_recent_menu())
    recent_menu = file_menu.addMenu(trans.t('menu_recent'))
    window.ui_objs['menu_recent'] = recent_menu

    # Recent projects submenu (items filled in by rebuild_recent_projects_menu())
    recent_proj_menu = file_menu.addMenu(trans.t('menu_recent_projects'))
    recent_proj_menu.setProperty("trans_key", 'menu_recent_projects')
    window.ui_objs['menu_recent_projects'] = recent_proj_menu

    file_menu.addSeparator()

    save_action = QAction(trans.t('action_save'), window)
    save_action.setShortcut("Ctrl+S")
    save_action.triggered.connect(window.export_manager.save_result)
    file_menu.addAction(save_action)
    window.ui_objs['action_save'] = save_action

    save_stack_menu = file_menu.addMenu(trans.t('menu_save_stack'))
    window.ui_objs['menu_save_stack'] = save_stack_menu

    registered_stack_menu = save_stack_menu.addMenu(trans.t('menu_registered_stack'))
    window.ui_objs['menu_registered_stack'] = registered_stack_menu

    save_reg_folder_action = QAction(trans.t('action_save_folder'), window)
    save_reg_folder_action.setShortcut("Ctrl+Shift+S")
    save_reg_folder_action.triggered.connect(window.export_manager.save_result_stack)
    save_reg_folder_action.setProperty("trans_key", 'action_save_folder')
    registered_stack_menu.addAction(save_reg_folder_action)
    window.ui_objs['action_save_folder_reg'] = save_reg_folder_action

    save_reg_gif_action = QAction(trans.t('action_save_gif'), window)
    save_reg_gif_action.triggered.connect(lambda: window.export_manager.save_as_gif(target_type="registered"))
    save_reg_gif_action.setProperty("trans_key", 'action_save_gif')
    registered_stack_menu.addAction(save_reg_gif_action)
    window.ui_objs['action_save_gif_reg'] = save_reg_gif_action

    save_reg_mtiff_action = QAction(trans.t('action_save_multipage_tiff'), window)
    save_reg_mtiff_action.triggered.connect(
        lambda: window.export_manager.save_stack_as_multipage_tiff("registered"))
    save_reg_mtiff_action.setProperty("trans_key", 'action_save_multipage_tiff')
    registered_stack_menu.addAction(save_reg_mtiff_action)
    window.ui_objs['action_save_mtiff_reg'] = save_reg_mtiff_action

    input_stack_menu = save_stack_menu.addMenu(trans.t('menu_input_stack'))
    window.ui_objs['menu_input_stack'] = input_stack_menu

    save_input_folder_action = QAction(trans.t('action_save_folder'), window)
    save_input_folder_action.triggered.connect(window.export_manager.save_processed_input_stack)
    save_input_folder_action.setProperty("trans_key", 'action_save_folder')
    input_stack_menu.addAction(save_input_folder_action)
    window.ui_objs['action_save_folder_input'] = save_input_folder_action

    save_input_gif_action = QAction(trans.t('action_save_gif'), window)
    save_input_gif_action.triggered.connect(lambda: window.export_manager.save_as_gif(target_type="input"))
    save_input_gif_action.setProperty("trans_key", 'action_save_gif')
    input_stack_menu.addAction(save_input_gif_action)
    window.ui_objs['action_save_gif_input'] = save_input_gif_action

    save_input_mtiff_action = QAction(trans.t('action_save_multipage_tiff'), window)
    save_input_mtiff_action.triggered.connect(
        lambda: window.export_manager.save_stack_as_multipage_tiff("input"))
    save_input_mtiff_action.setProperty("trans_key", 'action_save_multipage_tiff')
    input_stack_menu.addAction(save_input_mtiff_action)
    window.ui_objs['action_save_mtiff_input'] = save_input_mtiff_action

    file_menu.addSeparator()

    clear_action = QAction(trans.t('action_clear_stack'), window)
    clear_action.setShortcut("Ctrl+W")
    clear_action.triggered.connect(window.source_manager.clear_image_stack)
    file_menu.addAction(clear_action)
    window.ui_objs['action_clear_stack'] = clear_action

    file_menu.addSeparator()

    exit_action = QAction(trans.t('action_exit'), window)
    exit_action.setShortcut("Ctrl+Q")
    exit_action.triggered.connect(window.close)
    file_menu.addAction(exit_action)
    window.ui_objs['action_exit'] = exit_action

    # --- Edit Menu ---
    edit_menu = menubar.addMenu(trans.t('menu_edit'))
    window.ui_objs['menu_edit'] = edit_menu

    undo_action = QAction(trans.t('menu_undo'), window)
    undo_action.setShortcut("Ctrl+Z")
    undo_action.setProperty("trans_key", 'menu_undo')
    undo_action.triggered.connect(window.undo)
    undo_action.setEnabled(False)
    edit_menu.addAction(undo_action)
    window.ui_objs['action_undo'] = undo_action

    redo_action = QAction(trans.t('menu_redo'), window)
    redo_action.setShortcut("Ctrl+Y")
    redo_action.setProperty("trans_key", 'menu_redo')
    redo_action.triggered.connect(window.redo)
    redo_action.setEnabled(False)
    edit_menu.addAction(redo_action)
    window.ui_objs['action_redo'] = redo_action

    edit_menu.addSeparator()

    rotate_menu = edit_menu.addMenu(trans.t('menu_rotate'))
    window.ui_objs['menu_rotate'] = rotate_menu

    rotate_90_cw_action = QAction(trans.t('action_rotate_90_cw'), window)
    rotate_90_cw_action.triggered.connect(lambda: window.rotate_stack(1))
    rotate_menu.addAction(rotate_90_cw_action)
    window.ui_objs['action_rotate_90_cw'] = rotate_90_cw_action

    rotate_90_ccw_action = QAction(trans.t('action_rotate_90_ccw'), window)
    rotate_90_ccw_action.triggered.connect(lambda: window.rotate_stack(2))
    rotate_menu.addAction(rotate_90_ccw_action)
    window.ui_objs['action_rotate_90_ccw'] = rotate_90_ccw_action

    rotate_180_action = QAction(trans.t('action_rotate_180'), window)
    rotate_180_action.triggered.connect(lambda: window.rotate_stack(0))
    rotate_menu.addAction(rotate_180_action)
    window.ui_objs['action_rotate_180'] = rotate_180_action

    flip_menu = edit_menu.addMenu(trans.t('menu_flip'))
    window.ui_objs['menu_flip'] = flip_menu

    flip_horizontal_action = QAction(trans.t('action_flip_h'), window)
    flip_horizontal_action.triggered.connect(lambda: window.flip_stack(1))
    flip_menu.addAction(flip_horizontal_action)
    window.ui_objs['action_flip_h'] = flip_horizontal_action

    flip_vertical_action = QAction(trans.t('action_flip_v'), window)
    flip_vertical_action.triggered.connect(lambda: window.flip_stack(0))
    flip_menu.addAction(flip_vertical_action)
    window.ui_objs['action_flip_v'] = flip_vertical_action

    window.resize_action = QAction(trans.t('action_resize'), window)
    window.resize_action.triggered.connect(window.resize_all_images)
    window.resize_action.setEnabled(False)
    edit_menu.addAction(window.resize_action)
    window.ui_objs['action_resize'] = window.resize_action

    # Retouch brush: paint pixels from a source frame onto the result
    retouch_action = QAction(trans.t('action_retouch'), window)
    retouch_action.setShortcut("Ctrl+T")
    retouch_action.setToolTip(trans.t('action_retouch_hint'))
    retouch_action.triggered.connect(window.show_retouch_dialog)
    edit_menu.addAction(retouch_action)
    window.ui_objs['action_retouch'] = retouch_action
    retouch_action.setProperty("trans_key", 'action_retouch')

    # Labels live under Tools: adding and removing are two halves of one
    # feature, so they share one submenu instead of a loose action here and a
    # separate "delete" menu elsewhere.

    # --- Tools Menu ---
    batch_menu = menubar.addMenu(trans.t('menu_tools'))
    window.ui_objs['menu_tools'] = batch_menu

    batch_action = QAction(trans.t('action_batch_process'), window)
    batch_action.triggered.connect(window.show_batch_processing_dialog)
    batch_menu.addAction(batch_action)
    window.ui_objs['action_batch_process'] = batch_action

    compare_action = QAction(trans.t('action_method_compare'), window)
    compare_action.setToolTip(trans.t('action_method_compare_hint'))
    compare_action.triggered.connect(window.show_compare_dialog)
    batch_menu.addAction(compare_action)
    window.ui_objs['action_method_compare'] = compare_action

    # Rendering group | analysis group | annotation group
    batch_menu.addSeparator()

    depth_action = QAction(trans.t('menu_depth_map'), window)
    depth_action.triggered.connect(window.show_depth_map_dialog)
    depth_action.setProperty("trans_key", 'menu_depth_map')
    batch_menu.addAction(depth_action)
    window.ui_objs['menu_depth_map'] = depth_action

    dust_action = QAction(trans.t('menu_dust_map'), window)
    dust_action.triggered.connect(window.show_dust_map_dialog)
    dust_action.setProperty("trans_key", 'menu_dust_map')
    batch_menu.addAction(dust_action)
    window.ui_objs['menu_dust_map'] = dust_action

    batch_menu.addSeparator()

    # Scale bar mirrors Labels: one submenu with "Add/Edit" and a "Remove"
    # section, instead of a loose dialog entry.
    scale_menu = batch_menu.addMenu(trans.t('menu_scale_bar'))
    scale_menu.setProperty("trans_key", 'menu_scale_bar')
    window.ui_objs['menu_scale_bar'] = scale_menu

    scale_add_action = QAction(trans.t('action_scale_add'), window)
    scale_add_action.triggered.connect(window.show_scale_bar_dialog)
    scale_menu.addAction(scale_add_action)
    window.ui_objs['action_scale_add'] = scale_add_action

    scale_remove_menu = scale_menu.addMenu(trans.t('menu_scale_remove'))
    scale_remove_menu.setProperty("trans_key", 'menu_scale_remove')
    window.ui_objs['menu_scale_remove'] = scale_remove_menu

    def _scale_target_action(text_key, where):
        act = QAction(trans.t(text_key), window)
        act.setProperty("trans_key", text_key)
        act.triggered.connect(
            lambda: window.remove_scale_bar(where))
        scale_remove_menu.addAction(act)
        window.ui_objs[f'action_scale_remove_{where}'] = act
        return act

    _scale_target_action('scalebar_remove_source', 'source')
    _scale_target_action('scalebar_remove_output', 'output')
    _scale_target_action('scalebar_remove_all', 'all')

    batch_menu.addSeparator()
    label_menu = batch_menu.addMenu(trans.t('menu_labels'))
    label_menu.setProperty("trans_key", 'menu_labels')
    window.ui_objs['menu_labels'] = label_menu

    window.add_label_action = QAction(trans.t('action_add_label'), window)
    window.add_label_action.triggered.connect(
        window.label_manager.show_add_label_dialog)
    window.add_label_action.setEnabled(False)
    label_menu.addAction(window.add_label_action)

    # Each target can be cleared independently, including disabled labels.
    remove_menu = label_menu.addMenu(trans.t('menu_remove_labels'))
    remove_menu.setProperty("trans_key", 'menu_remove_labels')
    window.ui_objs['menu_remove_labels'] = remove_menu

    window.del_reg_label_action = QAction(trans.t('label_target_registered'), window)
    window.del_reg_label_action.setProperty("trans_key", 'label_target_registered')
    window.del_reg_label_action.triggered.connect(
        window.label_manager.delete_registered_labels)
    window.del_reg_label_action.setEnabled(False)
    remove_menu.addAction(window.del_reg_label_action)
    window.ui_objs['action_del_reg_label'] = window.del_reg_label_action

    window.del_output_label_action = QAction(trans.t('label_target_output'), window)
    window.del_output_label_action.setProperty("trans_key", 'label_target_output')
    window.del_output_label_action.triggered.connect(
        window.label_manager.delete_output_labels)
    window.del_output_label_action.setEnabled(False)
    remove_menu.addAction(window.del_output_label_action)
    window.ui_objs['action_del_output_label'] = window.del_output_label_action

    window.del_input_label_action = QAction(trans.t('label_target_input'), window)
    window.del_input_label_action.setProperty("trans_key", 'label_target_input')
    window.del_input_label_action.triggered.connect(
        window.label_manager.delete_input_labels)
    window.del_input_label_action.setEnabled(False)
    remove_menu.addAction(window.del_input_label_action)
    window.ui_objs['action_del_input_label'] = window.del_input_label_action

    window.del_all_labels_action = QAction(trans.t('action_del_all_labels'), window)
    window.del_all_labels_action.setProperty("trans_key", 'action_del_all_labels')
    window.del_all_labels_action.triggered.connect(
        window.label_manager.delete_all_labels)
    window.del_all_labels_action.setEnabled(False)
    remove_menu.addAction(window.del_all_labels_action)
    window.ui_objs['action_del_all_labels'] = window.del_all_labels_action

    # --- Settings Menu ---
    settings_menu = menubar.addMenu(trans.t('menu_settings'))
    window.ui_objs['menu_settings'] = settings_menu

    # Everything that is a tuning knob moved to one Advanced page: spread
    # across the menu they were impossible to reason about (which one affects
    # speed, which one affects output?), and each had its own dialog.
    advanced_action = QAction(trans.t("advanced_settings_title"), window)
    advanced_action.setToolTip(trans.t("advanced_settings_hint"))
    advanced_action.triggered.connect(window.show_advanced_settings_dialog)
    settings_menu.addAction(advanced_action)
    # update_ui_text() falls back to the ui_objs key when no trans_key is set,
    # so without this the menu item rendered as "action_advanced_settings"
    # after any language switch.
    advanced_action.setProperty("trans_key", 'advanced_settings_title')
    window.ui_objs['action_advanced_settings'] = advanced_action

    # GPU stays reachable here as a quick toggle; it is the one switch users
    # flip often enough to deserve a click instead of two.
    gpu_action = QAction(trans.t('action_gpu_accel'), window)
    gpu_action.setCheckable(True)
    from utils.platform_utils import has_compute_gpu
    _gpu_present = has_compute_gpu()
    gpu_action.setChecked(_gpu_present and getattr(window, 'use_gpu', True))
    gpu_action.setEnabled(_gpu_present)
    gpu_action.setToolTip(trans.t('msg_gpu_accel_hint') if _gpu_present
                          else trans.t('msg_gpu_accel_unavailable'))
    gpu_action.triggered.connect(window.set_use_gpu)
    settings_menu.addAction(gpu_action)
    window.ui_objs['action_gpu_accel'] = gpu_action

    settings_menu.addSeparator()

    # Theme: light first, it is the fresh-install default. This used to be
    # further down and lost when the Advanced page absorbed the other items.
    theme_menu = settings_menu.addMenu(trans.t('menu_theme'))
    theme_group_theme = QActionGroup(window)
    theme_group_theme.setExclusive(True)
    for theme_code, trans_key in (("light", 'theme_light'),
                                  ("dark", 'theme_dark')):
        act = QAction(trans.t(trans_key), window)
        act.setCheckable(True)
        act.setChecked(getattr(window, 'ui_theme', 'light') == theme_code)
        act.setData(theme_code)
        act.setProperty("trans_key", trans_key)
        act.triggered.connect(
            lambda _checked, code=theme_code: window.apply_theme(code))
        theme_group_theme.addAction(act)
        theme_menu.addAction(act)
        window.ui_objs[f'theme_{theme_code}'] = act
    window.ui_objs['menu_theme'] = theme_menu
    window.ui_objs['menu_theme'].setProperty("trans_key", 'menu_theme')

    # Language Submenu
    lang_menu = settings_menu.addMenu(trans.t('menu_language'))
    window.ui_objs['menu_language'] = lang_menu
    
    lang_group = QActionGroup(window)
    lang_group.setExclusive(True)
    
    # Built from the packs that actually loaded, so adding locales_xx.py puts
    # the language in this menu without touching it. Order comes from
    # TranslationManager.PACK_ORDER.
    for code, label in trans.available_languages():
        act = QAction(label, window)
        act.setCheckable(True)
        act.setChecked(trans.current_lang == code)
        act.setData(code)
        act.triggered.connect(
            lambda _checked=False, c=code: window.set_language(c))
        lang_group.addAction(act)
        lang_menu.addAction(act)
        window.ui_objs[f'action_lang_{code}'] = act

    # --- Help Menu ---
    help_menu = menubar.addMenu(trans.t('menu_help'))
    window.ui_objs['menu_help'] = help_menu

    env_action = QAction(trans.t('action_env_info'), window)
    env_action.triggered.connect(window.show_environment_info)
    help_menu.addAction(env_action)
    window.ui_objs['action_env_info'] = env_action

    feedback_action = QAction(trans.t('action_send_feedback'), window)
    feedback_action.setToolTip(trans.t('action_send_feedback_hint'))
    feedback_action.triggered.connect(window.show_feedback_dialog)
    help_menu.addAction(feedback_action)
    window.ui_objs['action_send_feedback'] = feedback_action

    help_menu.addSeparator()

    contact_action = QAction(trans.t('action_contact'), window)
    contact_action.triggered.connect(window.show_contact_info)
    help_menu.addAction(contact_action)
    window.ui_objs['action_contact'] = contact_action

    quickstart_action = QAction(trans.t('menu_quick_start'), window)
    quickstart_action.triggered.connect(window.show_quick_start)
    help_menu.addAction(quickstart_action)
    window.ui_objs['menu_quick_start'] = quickstart_action

    update_action = QAction(trans.t('menu_check_updates'), window)
    update_action.triggered.connect(window.check_for_updates)
    help_menu.addAction(update_action)
    window.ui_objs['menu_check_updates'] = update_action

    # Only visible once a build is downloaded and verified: "update now"
    # applies it instantly and restarts into the restored session.
    install_action = QAction(trans.t('menu_install_update').format(version=""), window)
    install_action.setProperty("trans_key", 'menu_install_update')
    install_action.triggered.connect(window.install_update_now)
    install_action.setVisible(False)
    help_menu.addAction(install_action)
    window.ui_objs['action_install_update'] = install_action

    logs_action = QAction(trans.t('menu_open_logs'), window)
    logs_action.triggered.connect(window.open_logs_folder)
    help_menu.addAction(logs_action)
    window.ui_objs['menu_open_logs'] = logs_action
