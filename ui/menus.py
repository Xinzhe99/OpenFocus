from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu, QMainWindow


def setup_menus(window: QMainWindow) -> None:
    """Configure the main menu bar and attach actions to the window."""
    menubar = window.menuBar()
    file_menu = menubar.addMenu("File")

    open_action = QAction("Open", window)
    open_action.setShortcut("Ctrl+O")
    open_action.triggered.connect(window.open_folder_dialog)
    file_menu.addAction(open_action)

    save_action = QAction("Save", window)
    save_action.setShortcut("Ctrl+S")
    save_action.triggered.connect(window.export_manager.save_result)
    file_menu.addAction(save_action)

    save_stack_menu = QMenu("Save Stack", window)
    file_menu.addMenu(save_stack_menu)

    registered_stack_menu = QMenu("Registered Stack", window)
    save_stack_menu.addMenu(registered_stack_menu)

    save_reg_folder_action = QAction("Save as Folder", window)
    save_reg_folder_action.setShortcut("Ctrl+Shift+S")
    save_reg_folder_action.triggered.connect(window.export_manager.save_result_stack)
    registered_stack_menu.addAction(save_reg_folder_action)

    save_reg_gif_action = QAction("Save as GIF", window)
    save_reg_gif_action.triggered.connect(lambda: window.export_manager.save_as_gif(target_type="registered"))
    registered_stack_menu.addAction(save_reg_gif_action)

    input_stack_menu = QMenu("Input Stack", window)
    save_stack_menu.addMenu(input_stack_menu)

    save_input_folder_action = QAction("Save as Folder", window)
    save_input_folder_action.triggered.connect(window.export_manager.save_processed_input_stack)
    input_stack_menu.addAction(save_input_folder_action)

    save_input_gif_action = QAction("Save as GIF", window)
    save_input_gif_action.triggered.connect(lambda: window.export_manager.save_as_gif(target_type="input"))
    input_stack_menu.addAction(save_input_gif_action)

    file_menu.addSeparator()

    clear_action = QAction("Clear Stack", window)
    clear_action.setShortcut("Ctrl+W")
    clear_action.triggered.connect(window.source_manager.clear_image_stack)
    file_menu.addAction(clear_action)

    file_menu.addSeparator()

    batch_action = QAction("Batch Processing", window)
    batch_action.triggered.connect(window.show_batch_processing_dialog)
    file_menu.addAction(batch_action)

    file_menu.addSeparator()

    exit_action = QAction("Exit", window)
    exit_action.setShortcut("Ctrl+Q")
    exit_action.triggered.connect(window.close)
    file_menu.addAction(exit_action)

    edit_menu = menubar.addMenu("Edit")

    rotate_menu = QMenu("Rotate", window)
    edit_menu.addMenu(rotate_menu)

    rotate_90_cw_action = QAction("90° Clockwise", window)
    rotate_90_cw_action.triggered.connect(lambda: window.rotate_stack(1))
    rotate_menu.addAction(rotate_90_cw_action)

    rotate_90_ccw_action = QAction("90° Counter-Clockwise", window)
    rotate_90_ccw_action.triggered.connect(lambda: window.rotate_stack(2))
    rotate_menu.addAction(rotate_90_ccw_action)

    rotate_180_action = QAction("180°", window)
    rotate_180_action.triggered.connect(lambda: window.rotate_stack(0))
    rotate_menu.addAction(rotate_180_action)

    flip_menu = QMenu("Flip", window)
    edit_menu.addMenu(flip_menu)

    flip_horizontal_action = QAction("Horizontal Flip", window)
    flip_horizontal_action.triggered.connect(lambda: window.flip_stack(1))
    flip_menu.addAction(flip_horizontal_action)

    flip_vertical_action = QAction("Vertical Flip", window)
    flip_vertical_action.triggered.connect(lambda: window.flip_stack(0))
    flip_menu.addAction(flip_vertical_action)

    window.resize_action = QAction("Resize", window)
    window.resize_action.triggered.connect(window.resize_all_images)
    window.resize_action.setEnabled(False)
    edit_menu.addAction(window.resize_action)

    window.add_label_action = QAction("Add Label", window)
    window.add_label_action.triggered.connect(window.label_manager.show_add_label_dialog)
    window.add_label_action.setEnabled(False)
    edit_menu.addAction(window.add_label_action)

    delete_label_menu = QMenu("Delete Label", window)
    edit_menu.addMenu(delete_label_menu)

    window.del_reg_label_action = QAction("Delete Registered Stack Labels", window)
    window.del_reg_label_action.triggered.connect(window.label_manager.delete_registered_labels)
    window.del_reg_label_action.setEnabled(False)
    delete_label_menu.addAction(window.del_reg_label_action)

    window.del_input_label_action = QAction("Delete Input Stack Labels", window)
    window.del_input_label_action.triggered.connect(window.label_manager.delete_input_labels)
    window.del_input_label_action.setEnabled(False)
    delete_label_menu.addAction(window.del_input_label_action)

    help_menu = menubar.addMenu("Help")

    env_action = QAction("Environment Info", window)
    env_action.triggered.connect(window.show_environment_info)
    help_menu.addAction(env_action)

    help_menu.addSeparator()

    contact_action = QAction("Contact Us", window)
    contact_action.triggered.connect(window.show_contact_info)
    help_menu.addAction(contact_action)
