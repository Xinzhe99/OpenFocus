from dataclasses import dataclass

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from dialogs import RegistrationHelpDialog, RenderMethodHelpDialog
from styles import (
    HELP_BUTTON_STYLE,
    HOVER_HIGHLIGHT_BUTTON_STYLE,
    OUTPUT_LIST_STYLE,
    SOURCE_LIST_STYLE,
)


@dataclass
class RightPanelComponents:
    widget: QFrame
    splitter: QSplitter
    btn_reset: QPushButton
    btn_render: QPushButton
    btn_method_help: QPushButton
    btn_reg_help: QPushButton
    rb_a: QRadioButton
    rb_b: QRadioButton
    rb_c: QRadioButton
    rb_d: QRadioButton
    cb_align_homography: QCheckBox
    cb_align_ecc: QCheckBox
    slider_smooth: QSlider
    smooth_value_label: QLabel
    smooth_widget: QWidget
    source_images_label: QLabel
    file_list: QListWidget
    output_label: QLabel
    output_list: QListWidget


def create_right_panel() -> RightPanelComponents:
    right_panel = QFrame()
    right_panel.setMinimumWidth(280)
    right_panel.setStyleSheet("background-color: #2b2b2b; border-left: 1px solid #111;")
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(0)

    right_splitter = QSplitter(Qt.Orientation.Vertical)
    right_layout.addWidget(right_splitter)

    # Configuration group ----------------------------------
    config_widget = QWidget()
    config_layout = QVBoxLayout(config_widget)
    config_layout.setContentsMargins(10, 10, 10, 10)

    method_registration_layout = QHBoxLayout()

    method_group = QGroupBox("Fusion")
    method_layout = QVBoxLayout(method_group)
    rb_a = QRadioButton("Guided Filter")
    rb_a.setChecked(True)
    rb_a.setAutoExclusive(False)
    rb_b = QRadioButton("DCT")
    rb_b.setAutoExclusive(False)
    rb_c = QRadioButton("DTCWT")
    rb_c.setAutoExclusive(False)
    rb_d = QRadioButton("StackMFF-V4")
    rb_d.setAutoExclusive(False)

    method_layout.addWidget(rb_a)
    method_layout.addWidget(rb_b)
    method_layout.addWidget(rb_c)
    method_layout.addWidget(rb_d)
    method_layout.addStretch()

    method_help_layout = QHBoxLayout()
    method_help_layout.addStretch()
    btn_method_help = QPushButton("?")
    btn_method_help.setFixedSize(20, 20)
    btn_method_help.setStyleSheet(HELP_BUTTON_STYLE)
    method_help_layout.addWidget(btn_method_help)
    method_layout.addLayout(method_help_layout)

    registration_group = QGroupBox("Registration")
    registration_layout = QVBoxLayout(registration_group)
    cb_align_homography = QCheckBox("Align (Homography)")
    cb_align_homography.setChecked(True)
    cb_align_ecc = QCheckBox("Align (ECC)")
    cb_align_ecc.setChecked(False)

    registration_layout.addWidget(cb_align_homography)
    registration_layout.addWidget(cb_align_ecc)
    registration_layout.addStretch()

    reg_help_layout = QHBoxLayout()
    reg_help_layout.addStretch()
    btn_reg_help = QPushButton("?")
    btn_reg_help.setFixedSize(20, 20)
    btn_reg_help.setStyleSheet(HELP_BUTTON_STYLE)
    reg_help_layout.addWidget(btn_reg_help)
    registration_layout.addLayout(reg_help_layout)

    method_registration_layout.addWidget(registration_group)
    method_registration_layout.addWidget(method_group)
    config_layout.addLayout(method_registration_layout)

    smooth_widget = QWidget()
    smooth_layout = QVBoxLayout(smooth_widget)
    smooth_layout.setContentsMargins(0, 5, 0, 5)
    smooth_top = QHBoxLayout()
    smooth_top.addWidget(QLabel(" Kernel:"))
    smooth_top.addStretch()
    lbl_smooth_value = QLabel("31")
    smooth_top.addWidget(lbl_smooth_value)
    slider_smooth = QSlider(Qt.Orientation.Horizontal)
    slider_smooth.setRange(1, 51)
    slider_smooth.setSingleStep(2)
    slider_smooth.setPageStep(2)
    slider_smooth.setValue(31)
    smooth_layout.addLayout(smooth_top)
    smooth_layout.addWidget(slider_smooth)
    config_layout.addWidget(smooth_widget)

    button_bar = QHBoxLayout()
    btn_reset = QPushButton("Reset Default")
    btn_reset.setStyleSheet(HOVER_HIGHLIGHT_BUTTON_STYLE)
    btn_render = QPushButton("▶ Start Render")
    btn_render.setFixedHeight(40)
    btn_render.setStyleSheet(HOVER_HIGHLIGHT_BUTTON_STYLE)
    button_bar.addWidget(btn_reset)
    button_bar.addWidget(btn_render)
    config_layout.addLayout(button_bar)

    right_splitter.addWidget(config_widget)

    # Source list -----------------------------------------
    source_list_widget = QWidget()
    source_list_layout = QVBoxLayout(source_list_widget)
    source_list_layout.setContentsMargins(10, 10, 10, 10)
    source_images_label = QLabel("Source Images: 0")
    source_list_layout.addWidget(source_images_label)
    file_list = QListWidget()
    file_list.setIconSize(QSize(40, 40))
    file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    file_list.setStyleSheet(SOURCE_LIST_STYLE)
    source_list_layout.addWidget(file_list)

    right_splitter.addWidget(source_list_widget)

    # Output list -----------------------------------------
    output_list_widget = QWidget()
    output_list_layout = QVBoxLayout(output_list_widget)
    output_list_layout.setContentsMargins(10, 10, 10, 10)
    output_label = QLabel("Output: 0")
    output_list_layout.addWidget(output_label)
    output_list = QListWidget()
    output_list.setIconSize(QSize(40, 40))
    output_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    output_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    output_list.setStyleSheet(OUTPUT_LIST_STYLE)
    output_list_layout.addWidget(output_list)

    right_splitter.addWidget(output_list_widget)

    right_splitter.setStretchFactor(0, 1)
    right_splitter.setStretchFactor(1, 3)
    right_splitter.setStretchFactor(2, 3)

    return RightPanelComponents(
        widget=right_panel,
        splitter=right_splitter,
        btn_reset=btn_reset,
        btn_render=btn_render,
        btn_method_help=btn_method_help,
        btn_reg_help=btn_reg_help,
        rb_a=rb_a,
        rb_b=rb_b,
        rb_c=rb_c,
        rb_d=rb_d,
        cb_align_homography=cb_align_homography,
        cb_align_ecc=cb_align_ecc,
        slider_smooth=slider_smooth,
        smooth_value_label=lbl_smooth_value,
        smooth_widget=smooth_widget,
        source_images_label=source_images_label,
        file_list=file_list,
        output_label=output_label,
        output_list=output_list,
    )


def bind_right_panel(window, components: RightPanelComponents) -> None:
    """Connect signals for the right-panel controls using the provided window."""

    components.rb_a.clicked.connect(lambda: window.handle_method_selection(components.rb_a))
    components.rb_b.clicked.connect(lambda: window.handle_method_selection(components.rb_b))
    components.rb_c.clicked.connect(lambda: window.handle_method_selection(components.rb_c))
    components.rb_d.clicked.connect(lambda: window.handle_method_selection(components.rb_d))

    components.btn_method_help.clicked.connect(lambda: RenderMethodHelpDialog(window).exec())
    components.btn_reg_help.clicked.connect(lambda: RegistrationHelpDialog(window).exec())

    components.btn_render.clicked.connect(window.render_manager.start_render)
    components.btn_reset.clicked.connect(window.reset_to_default)

    components.slider_smooth.valueChanged.connect(window.handle_kernel_slider_change)

    components.rb_a.clicked.connect(window.update_slider_availability)
    components.rb_b.clicked.connect(window.update_slider_availability)
    components.rb_c.clicked.connect(window.update_slider_availability)
    components.rb_d.clicked.connect(window.update_slider_availability)

    components.file_list.customContextMenuRequested.connect(window.source_manager.show_source_context_menu)
    components.file_list.currentRowChanged.connect(window.source_manager.sync_slider_from_list)
    components.output_list.customContextMenuRequested.connect(window.output_manager.show_output_context_menu)
    components.output_list.currentRowChanged.connect(window.output_manager.sync_output_slider_from_list)
    components.output_list.itemClicked.connect(window.output_manager.display_output_image_in_result_view)

    delete_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), components.file_list)
    delete_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
    delete_shortcut.activated.connect(window.source_manager.delete_selected_source_images)
    components.file_list._delete_shortcut = delete_shortcut  # keep reference alive

    delete_output_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), components.output_list)
    delete_output_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
    delete_output_shortcut.activated.connect(window.output_manager.delete_selected_output_images)
    components.output_list._delete_shortcut = delete_output_shortcut

    window.update_slider_availability()
