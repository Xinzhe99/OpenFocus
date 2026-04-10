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

    window.add_label_action = QAction(trans.t('action_add_label'), window)
    window.add_label_action.triggered.connect(window.label_manager.show_add_label_dialog)
    window.add_label_action.setEnabled(False)
    edit_menu.addAction(window.add_label_action)
    window.ui_objs['action_add_label'] = window.add_label_action

    delete_label_menu = edit_menu.addMenu(trans.t('menu_del_label'))
    window.ui_objs['menu_del_label'] = delete_label_menu

    window.del_reg_label_action = QAction(trans.t('action_del_reg_label'), window)
    window.del_reg_label_action.triggered.connect(window.label_manager.delete_registered_labels)
    window.del_reg_label_action.setEnabled(False)
    delete_label_menu.addAction(window.del_reg_label_action)
    window.ui_objs['action_del_reg_label'] = window.del_reg_label_action

    window.del_input_label_action = QAction(trans.t('action_del_input_label'), window)
    window.del_input_label_action.triggered.connect(window.label_manager.delete_input_labels)
    window.del_input_label_action.setEnabled(False)
    delete_label_menu.addAction(window.del_input_label_action)
    window.ui_objs['action_del_input_label'] = window.del_input_label_action

    # --- Tools Menu ---
    batch_menu = menubar.addMenu(trans.t('menu_tools'))
    window.ui_objs['menu_tools'] = batch_menu

    batch_action = QAction(trans.t('action_batch_process'), window)
    batch_action.triggered.connect(window.show_batch_processing_dialog)
    batch_menu.addAction(batch_action)
    window.ui_objs['action_batch_process'] = batch_action

    # --- Settings Menu ---
    settings_menu = menubar.addMenu(trans.t('menu_settings'))
    window.ui_objs['menu_settings'] = settings_menu

    # Language Submenu
    lang_menu = settings_menu.addMenu(trans.t('menu_language'))
    window.ui_objs['menu_language'] = lang_menu
    
    lang_group = QActionGroup(window)
    lang_group.setExclusive(True)
    
    lang_en = QAction(trans.t('action_lang_en'), window)
    lang_en.setCheckable(True)
    lang_en.setChecked(trans.current_lang == 'en')
    lang_en.triggered.connect(lambda: window.set_language('en'))
    lang_group.addAction(lang_en)
    lang_menu.addAction(lang_en)
    window.ui_objs['action_lang_en'] = lang_en
    
    lang_zh = QAction(trans.t('action_lang_zh'), window)
    lang_zh.setCheckable(True)
    lang_zh.setChecked(trans.current_lang == 'zh')
    lang_zh.triggered.connect(lambda: window.set_language('zh'))
    lang_group.addAction(lang_zh)
    lang_menu.addAction(lang_zh)
    window.ui_objs['action_lang_zh'] = lang_zh

    tile_action = QAction(trans.t('action_tile_settings'), window)
    # 打开瓦片设置对话框
    tile_action.triggered.connect(lambda: window.show_tile_settings())
    settings_menu.addAction(tile_action)
    window.ui_objs['action_tile_settings'] = tile_action

    registration_action = QAction(trans.t('action_reg_settings'), window)
    registration_action.triggered.connect(lambda: window.show_registration_settings())
    settings_menu.addAction(registration_action)
    window.ui_objs['action_reg_settings'] = registration_action

    thread_settings_action = QAction(trans.t('action_thread_settings'), window)
    thread_settings_action.triggered.connect(lambda: window.show_thread_settings())
    settings_menu.addAction(thread_settings_action)
    window.ui_objs['action_thread_settings'] = thread_settings_action

    stackmffv4_batch_action = QAction(trans.t('action_stackmffv4_batch_settings'), window)
    stackmffv4_batch_action.triggered.connect(lambda: window.show_stackmffv4_batch_settings())
    settings_menu.addAction(stackmffv4_batch_action)
    window.ui_objs['action_stackmffv4_batch_settings'] = stackmffv4_batch_action

    # --- Help Menu ---
    help_menu = menubar.addMenu(trans.t('menu_help'))
    window.ui_objs['menu_help'] = help_menu

    env_action = QAction(trans.t('action_env_info'), window)
    env_action.triggered.connect(window.show_environment_info)
    help_menu.addAction(env_action)
    window.ui_objs['action_env_info'] = env_action

    help_menu.addSeparator()

    contact_action = QAction(trans.t('action_contact'), window)
    contact_action.triggered.connect(window.show_contact_info)
    help_menu.addAction(contact_action)
    window.ui_objs['action_contact'] = contact_action
