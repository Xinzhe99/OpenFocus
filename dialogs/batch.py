# dialogs/batch.py
"""Batch processing dialogs."""

import os
from PyQt6.QtCore import Qt, QSize, QSignalBlocker
from PyQt6.QtGui import QFont, QIcon, QTextOption, QShortcut, QKeySequence
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QAbstractItemView,
    QGridLayout,
    QSizePolicy,
    QScrollArea,
    QFrame,
    QWidget,
    QLabel,
    QGroupBox,
    QSpinBox,
    QSlider,
    QTextEdit,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QComboBox,
    QLineEdit,
    QRadioButton,
    QSplitter,
    QTabWidget,
    QCheckBox,
)
from ui.styles import PRIMARY_BLUE, _ARROW_DOWN_DARK
from core.algorithms import (alignment_steps, fusion_algorithms, get_fusion,
    registration_parameter_specs, registration_algorithms, resolve_fusion_parameters,
    desktop_registration_parameters)
from ui.algorithm_parameters import AlgorithmParameterPanel
from ui.algorithm_combo import apply_method_availability, selected_item_enabled
from locales import trans
from utils.output_sharpen import DEFAULT_SHARPEN_STRENGTH, MAX_SHARPEN_STRENGTH
from widgets.stepper import StepperSpinBox
from widgets.workspace import WorkspaceSplitter


BATCH_PICK_DIR_KEY = "batch/last_pick_dir"

# Import workers still running when their dialog is destroyed. Dropping the
# last reference to a running QThread aborts the process ("QThread: Destroyed
# while thread is still running"), and waiting on the GUI thread would freeze
# the window. Track them here until they finish on their own; the cleanup
# connection belongs to the worker, not the dialog, so it still runs after the
# dialog is gone.
_TRACKED_IMPORT_WORKERS: set = set()


def _track_import_worker(worker) -> None:
    _TRACKED_IMPORT_WORKERS.add(worker)

    def release(w=worker):
        _TRACKED_IMPORT_WORKERS.discard(w)
        w.deleteLater()

    worker.finished.connect(release)


def _section_rule(text: str) -> QLabel:
    """A short heading that separates sections inside one group box.

    Batch's groups held six unrelated controls each with no visual hierarchy,
    so the dialog read as one flat list. Sub-headings cost one line and make
    the categories (method / parameters / options) obvious.
    """
    lbl = QLabel(text)
    lbl.setObjectName("sectionRule")
    f = lbl.font()
    f.setBold(True)
    lbl.setFont(f)
    return lbl


class BatchProcessingDialog(QDialog):
    """批处理设置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('batch_title'))
        
        # 存储选中的文件夹路径和对应的缩略图
        self.folder_paths = []
        self.folder_thumbnails = []
        self.single_folder_stacks = []
        self.single_folder_images_with_times = []

        # 获取父窗口的融合和对齐设置
        self.parent_window = parent
        self.load_scale_factor = float(getattr(parent, 'current_scale_factor', 1.0))
        self._runtime_settings = {
            key: getattr(parent, key, default) for key, default in (
                ('reg_downscale_width', None), ('tile_enabled', True),
                ('tile_block_size', None), ('tile_overlap', None),
                ('tile_threshold', None), ('thread_count', 4),
                ('resource_preference', 'balanced'),
                ('ai_batch_size', 2), ('use_gpu', True))}
        # Folder decoding runs on BatchImportWorker; these track the live
        # request so a late result cannot mutate a replaced preview.
        self._import_worker = None
        self._import_generation = 0
        self._import_busy = False
        self._import_mode = None
        self._import_callbacks = {}
        self._pending_thumbnails = []
        self._invalid_thumbnail_paths = set()
        self._applied_thumbnail_paths = set()
        self._active_thumbnail_paths = ()
        self._thumbnail_done = 0
        # Adaptive single-folder split runs off the GUI thread on
        # AdaptiveSplitWorker; these guard against stale results and cache
        # per-frame visual descriptors for instant mode switching.
        self._adaptive_worker = None
        self._adaptive_generation = 0
        self._adaptive_busy = False
        self._adaptive_progress = (0, 0)
        self._adaptive_descriptor_cache = {}
        self._last_split_method = None

        
        # Dark palette only in dark mode: hard-coding it showed a dark slab in
        # light mode.
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        split = WorkspaceSplitter()
        outer.addWidget(split, 1)

        sources = QWidget()
        layout = QVBoxLayout(sources)
        layout.setContentsMargins(14, 10, 10, 10)
        source_scroll = self._settings_page(sources)
        split.addWidget(source_scroll)

        settings_tabs = QTabWidget()
        split.addWidget(settings_tabs)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)

        import_mode_group = QGroupBox(trans.t('batch_import_mode'))
        import_mode_layout = QVBoxLayout(import_mode_group)
        import_mode_layout.setSpacing(6)

        self.rb_multiple_folders = QRadioButton(trans.t('batch_mode_multi'))
        self.rb_multiple_folders.setChecked(True)
        self.rb_multiple_folders.toggled.connect(self.on_import_mode_changed)
        import_mode_layout.addWidget(self.rb_multiple_folders)

        self.rb_single_folder = QRadioButton(trans.t('batch_mode_single'))
        self.rb_single_folder.toggled.connect(self.on_import_mode_changed)
        import_mode_layout.addWidget(self.rb_single_folder)
        import_mode_group.setSizePolicy(QSizePolicy.Policy.Preferred,
                                        QSizePolicy.Policy.Maximum)

        layout.addWidget(import_mode_group)

        self.folder_group = QGroupBox(trans.t('batch_stack_folders'))
        folder_layout = QVBoxLayout(self.folder_group)
        
        # No path text box: it only ever seeded the file dialog's start folder,
        # so the workflow was "paste a path, then also browse to it". The dialog
        # now reopens where you last picked, and the list below IS the state -
        # matching the single-folder group, which already showed a label.
        # Type-or-browse, in one row: the field accepts a path (Enter adds it)
        # and the button opens the picker. Before this the field only seeded
        # the picker's start folder, so typing a path did nothing at all -
        # which is why the folder chooser felt crippled.
        add_row = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText(trans.t('batch_path_placeholder'))
        self.path_input.setClearButtonEnabled(True)
        self.path_input.returnPressed.connect(self.add_path_from_field)
        add_row.addWidget(self.path_input, 1)
        add_folder_btn = QPushButton(trans.t('batch_btn_add'))
        add_folder_btn.clicked.connect(self.add_folders)
        self.btn_add_folder = add_folder_btn
        add_row.addWidget(add_folder_btn)
        folder_layout.addLayout(add_row)
        
        # 文件夹列表。空列表时显示一行提示——否则中间是一大片没有信息
        # 的空白，用户不知道是"还没加"还是"加失败了"。
        self.folder_list = QListWidget()
        # Multi-select: queues are batch by nature, and removing one folder at
        # a time (the old SingleSelection default) was painful.
        self.folder_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.folder_list.setIconSize(QSize(60, 60))
        # Dropping folders anywhere on this dialog queues them as stacks. The
        # list deliberately does NOT accept drops: an accepting widget would
        # consume them (QListWidget handles drag events internally), while a
        # non-accepting one lets Qt bubble them up to the dialog's handler.
        self.folder_list.setAcceptDrops(False)
        self.setAcceptDrops(True)
        self._folder_list_placeholder = QLabel(trans.t('batch_no_folders'))
        self._folder_list_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignCenter)
        self._folder_list_placeholder.setWordWrap(True)
        self._folder_list_placeholder.setObjectName("mutedLabel")
        self._folder_list_placeholder.setContentsMargins(12, 12, 12, 12)
        folder_layout.addWidget(self.folder_list)
        folder_layout.addWidget(self._folder_list_placeholder)
        self._update_folder_placeholder()
        
        # Remove button: only meaningful when there is something to remove.
        # Stretched across the full width under an empty list it read as the
        # primary action of the group.
        self.remove_folder_btn = QPushButton(trans.t('batch_btn_remove'))
        self.remove_folder_btn.clicked.connect(self.remove_selected_folders)
        self.remove_folder_btn.setEnabled(False)
        self._remove_shortcuts = []
        for key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            shortcut = QShortcut(QKeySequence(key), self.folder_list)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self.remove_folder_btn.click)
            self._remove_shortcuts.append(shortcut)
        self.folder_list.itemSelectionChanged.connect(
            lambda: self.remove_folder_btn.setEnabled(
                bool(self.folder_list.selectedItems())))
        self._thumbnail_status_label = QLabel("")
        self._thumbnail_status_label.setObjectName("mutedLabel")
        self._thumbnail_status_label.hide()
        remove_row = QHBoxLayout()
        remove_row.addWidget(self._thumbnail_status_label)
        remove_row.addStretch()
        remove_row.addWidget(self.remove_folder_btn)
        folder_layout.addLayout(remove_row)
        
        layout.addWidget(self.folder_group, 1)

        self.single_folder_group = QGroupBox(trans.t('batch_single_split_settings'))
        self.single_folder_group.setVisible(False)
        single_folder_layout = QVBoxLayout(self.single_folder_group)

        # 显示选取的文件夹路径
        self.single_folder_path_label = QLabel(trans.t('batch_folder_none'))
        self.single_folder_path_label.setObjectName("mutedLabel")
        self.single_folder_path_label.setWordWrap(True)
        single_folder_layout.addWidget(self.single_folder_path_label)

        split_method_layout = QHBoxLayout()
        split_method_layout.addWidget(QLabel(trans.t('batch_split_method')))
        self.split_method_combo = QComboBox()
        self.split_method_combo.addItem(trans.t('batch_split_adaptive'), 'adaptive')
        self.split_method_combo.addItem(trans.t('batch_split_fixed'), 'count')
        self.split_method_combo.addItem(trans.t('batch_split_time'), 'time_threshold')
        self.split_method_combo.addItem(trans.t('batch_split_exposure'), 'exposure')
        self.split_method_combo.setCurrentIndex(0)
        self.split_method_combo.currentIndexChanged.connect(self.on_split_method_changed)
        split_method_layout.addWidget(self.split_method_combo, 1)
        single_folder_layout.addLayout(split_method_layout)

        self.adaptive_hint_label = QLabel(trans.t('batch_adaptive_hint'))
        self.adaptive_hint_label.setObjectName("mutedLabel")
        self.adaptive_hint_label.setWordWrap(True)
        single_folder_layout.addWidget(self.adaptive_hint_label)

        self.param_row_widget = QWidget()
        param_layout = QHBoxLayout(self.param_row_widget)
        param_layout.setContentsMargins(0, 0, 0, 0)
        self.param_label = QLabel(trans.t('batch_images_per_stack'))
        param_layout.addWidget(self.param_label)

        self.param_spinbox = StepperSpinBox(minimum=2, maximum=1000,
                                            decimals=0, step=1, value=5)
        self.param_spinbox.setRange(2, 1000)
        self.param_spinbox.setValue(5)
        
        self.param_spinbox.valueChanged.connect(self.update_single_folder_preview)
        param_layout.addWidget(self.param_spinbox)

        self.param_unit_label = QLabel(trans.t('batch_unit_images'))
        param_layout.addWidget(self.param_unit_label)
        param_layout.addStretch()
        single_folder_layout.addWidget(self.param_row_widget)

        self.preview_label = QLabel(trans.t('batch_preview_default'))
        self.preview_label.setObjectName("mutedLabel")
        self.preview_label.setWordWrap(True)
        single_folder_layout.addWidget(self.preview_label)

        self._stack_import_status_label = QLabel("")
        self._stack_import_status_label.setObjectName("mutedLabel")
        self._stack_import_status_label.hide()
        single_folder_layout.addWidget(self._stack_import_status_label)

        add_single_folder_btn = QPushButton(trans.t('batch_btn_select_split'))
        add_single_folder_btn.clicked.connect(self.add_single_folder)
        self.btn_add_single_folder = add_single_folder_btn
        single_folder_layout.addWidget(add_single_folder_btn)

        layout.addWidget(self.single_folder_group)
        layout.addStretch()
        
        output_group = QWidget()
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(10, 10, 10, 10)
        output_layout.setSpacing(12)

        format_card = QGroupBox(trans.t('batch_output_format'))
        format_card_layout = QVBoxLayout(format_card)
        format_card_layout.setSpacing(8)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel(trans.t('batch_format_label')))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["JPG", "PNG", "BMP", "TIFF", "DNG"])
        self.format_combo.setCurrentText('PNG')
        self.format_combo.setToolTip(trans.t('export_depth_hint'))
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        format_row.addWidget(self.format_combo)

        self.quality_label = QLabel(trans.t('batch_quality_label'))
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(0, 100)
        self.quality_slider.setValue(100)
        self.quality_slider.setFixedWidth(150)
        self.quality_value_label = QLabel("100%")
        self.quality_value_label.setFixedWidth(40)
        self.quality_slider.valueChanged.connect(
            lambda v: self.quality_value_label.setText(f"{v}%"))
        self.quality_layout = QHBoxLayout()
        self.quality_layout.addWidget(self.quality_label)
        self.quality_layout.addWidget(self.quality_slider)
        self.quality_layout.addWidget(self.quality_value_label)
        self.quality_layout.addStretch()
        format_row.addLayout(self.quality_layout)
        format_row.addStretch()
        format_card_layout.addLayout(format_row)
        self.depth_hint = QLabel(trans.t('export_depth_hint'))
        self.depth_hint.setObjectName("mutedLabel")
        self.depth_hint.setWordWrap(True)
        self.depth_hint.hide()
        format_card_layout.addWidget(self.depth_hint)
        self.quality_label.setVisible(False)
        self.quality_slider.setVisible(False)
        self.quality_value_label.setVisible(False)
        output_layout.addWidget(format_card)

        location_card = QGroupBox(trans.t('batch_output_location'))
        location_layout = QVBoxLayout(location_card)
        location_layout.setSpacing(8)

        self.rb_subfolder = QRadioButton(trans.t('batch_out_subfolder'))
        self.rb_subfolder.setChecked(True)
        self.rb_subfolder.toggled.connect(self.on_output_option_changed)
        location_layout.addWidget(self.rb_subfolder)

        self.subfolder_controls = QWidget()
        subfolder_layout = QHBoxLayout(self.subfolder_controls)
        subfolder_layout.setContentsMargins(20, 0, 0, 0)
        subfolder_layout.addWidget(QLabel(trans.t('batch_subfolder_name')))
        self.subfolder_name = QLineEdit("OpenFocus_Output")
        subfolder_layout.addWidget(self.subfolder_name)
        location_layout.addWidget(self.subfolder_controls)

        self.rb_same_folder = QRadioButton(trans.t('batch_out_same'))
        self.rb_same_folder.toggled.connect(self.on_output_option_changed)
        location_layout.addWidget(self.rb_same_folder)

        self.rb_custom_folder = QRadioButton(trans.t('batch_out_custom'))
        self.rb_custom_folder.toggled.connect(self.on_output_option_changed)
        location_layout.addWidget(self.rb_custom_folder)

        self.custom_folder_controls = QWidget()
        custom_folder_layout = QHBoxLayout(self.custom_folder_controls)
        custom_folder_layout.setContentsMargins(20, 0, 0, 0)
        self.custom_folder_path = QLineEdit()
        custom_folder_layout.addWidget(self.custom_folder_path)
        browse_btn = QPushButton(trans.t('batch_btn_browse'))
        browse_btn.clicked.connect(self.browse_output_folder)
        self.browse_btn = browse_btn
        custom_folder_layout.addWidget(browse_btn)
        location_layout.addWidget(self.custom_folder_controls)
        self.custom_folder_controls.hide()

        self._save_aligned_pref = False
        self.save_aligned_cb = QCheckBox(trans.t('batch_save_aligned'))
        self.save_aligned_cb.setChecked(False)
        self.save_aligned_cb.setToolTip(trans.t('batch_save_aligned_hint'))
        self.save_aligned_cb.toggled.connect(self._remember_save_aligned)
        location_layout.addWidget(self.save_aligned_cb)
        # Grouped fusion owns the temporary disk output and cannot also write
        # an aligned copy. The checkbox is unchecked and disabled in that
        # state; this line says why instead of leaving an inert control.
        self.save_aligned_note = QLabel()
        self.save_aligned_note.setObjectName("mutedLabel")
        self.save_aligned_note.setWordWrap(True)
        self.save_aligned_note.hide()
        location_layout.addWidget(self.save_aligned_note)
        output_layout.addWidget(location_card)

        self.options_group = QWidget()
        options_layout = QVBoxLayout(self.options_group)
        options_layout.setContentsMargins(10, 10, 10, 10)
        options_layout.setSpacing(10)

        method_card = QGroupBox(trans.t('batch_section_method'))
        method_card_layout = QVBoxLayout(method_card)
        method_card_layout.setSpacing(10)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel(trans.t('group_fusion')))
        self.fusion_combo = QComboBox()
        from ui.algorithm_combo import fill_fusion_combo
        # The no-op entry means "no fusion" here just like in the main panel;
        # "No alignment" belongs to the registration dropdown on the next row.
        fill_fusion_combo(self.fusion_combo, none_label="radio_no_fusion",
                          none_hint="radio_no_fusion_hint")
        if self.fusion_combo.currentData() is None:
            # A disabled category header is never a method: without a saved
            # choice the batch default is the explicit no-fusion entry.
            self.fusion_combo.setCurrentIndex(
                self.fusion_combo.findData('none'))
        method_row.addWidget(self.fusion_combo, 1)
        method_card_layout.addLayout(method_row)

        align_row = QHBoxLayout()
        align_row.addWidget(QLabel(trans.t('group_registration')))
        self.align_combo = QComboBox()
        self._fill_method_combo(self.align_combo, registration_algorithms())
        align_row.addWidget(self.align_combo, 1)
        method_card_layout.addLayout(align_row)
        options_layout.addWidget(method_card)

        # --- Parameters --------------------------------------------------
        self.params_rule = _section_rule(trans.t('batch_section_params'))
        options_layout.addWidget(self.params_rule)

        self._fusion_drafts = {}
        self._exposure_drafts = {}
        self._registration_drafts = {}
        self._editing_method = None
        self._editing_alignment = None
        self.fusion_parameter_panel = AlgorithmParameterPanel(self)
        self.registration_parameter_panel = AlgorithmParameterPanel(
            self, hidden_parameters=('failure_policy',))
        options_layout.addWidget(self.fusion_parameter_panel)
        options_layout.addWidget(self.registration_parameter_panel)

        # Hidden legacy controls kept off-screen so internal settings APIs and
        # existing callers remain backward-compatible without cluttering the UI.
        self._hidden_legacy_controls = QWidget(self)
        self._hidden_legacy_controls.hide()
        legacy_layout = QVBoxLayout(self._hidden_legacy_controls)

        self.downscale_label = QLabel(trans.t('batch_downscale'))
        self.downscale_label.setToolTip(trans.t('batch_downscale_hint'))
        self.downscale_slider = QSlider(Qt.Orientation.Horizontal)
        self.downscale_slider.setToolTip(trans.t('batch_downscale_hint'))
        self.downscale_slider.setRange(20, 100)
        self.downscale_slider.setSingleStep(5)
        self.downscale_slider.setPageStep(10)
        self.downscale_slider.setValue(100)
        self.downscale_value = QLabel("100%")
        self.downscale_slider.valueChanged.connect(
            lambda v: self.downscale_value.setText(f"{v}%"))
        legacy_layout.addWidget(self.downscale_label)
        legacy_layout.addWidget(self.downscale_slider)
        legacy_layout.addWidget(self.downscale_value)

        self._grouping_pref = False
        self.chk_grouping = QCheckBox(trans.t('batch_group_enable'))
        self.chk_grouping.setToolTip(trans.t('batch_group_hint'))
        legacy_layout.addWidget(self.chk_grouping)
        self.group_controls = QWidget()
        group_row = QHBoxLayout(self.group_controls)
        group_row.setContentsMargins(0, 0, 0, 0)
        group_label = QLabel(trans.t('batch_group_size'))
        group_label.setToolTip(trans.t('batch_group_hint'))
        group_row.addWidget(group_label)
        self.group_size_spin = QSpinBox()
        self.group_size_spin.setRange(2, 9999)
        self.group_size_spin.setKeyboardTracking(False)
        self.group_size_spin.setValue(4)
        self.group_size_spin.setToolTip(trans.t('batch_group_hint'))
        group_row.addWidget(self.group_size_spin)
        group_row.addWidget(QLabel(trans.t('batch_group_overlap')))
        self.group_overlap_spin = QSpinBox()
        self.group_overlap_spin.setRange(0, 2)
        self.group_overlap_spin.setKeyboardTracking(False)
        self._group_overlap_preference = 1
        self.group_overlap_spin.setValue(1)
        self.group_overlap_spin.valueChanged.connect(
            lambda value: setattr(self, '_group_overlap_preference', value))
        group_row.addWidget(self.group_overlap_spin)
        group_row.addStretch()
        legacy_layout.addWidget(self.group_controls)
        self.group_controls.hide()
        self.group_note = QLabel()
        self.group_note.setObjectName("mutedLabel")
        self.group_note.setWordWrap(True)
        self.group_note.hide()
        legacy_layout.addWidget(self.group_note)
        self.chk_grouping.toggled.connect(self._grouping_changed)
        self.group_size_spin.valueChanged.connect(self._group_size_changed)

        self.flat_field_group = QWidget()
        flat_row = QHBoxLayout(self.flat_field_group)
        flat_row.setContentsMargins(0, 0, 0, 0)
        flat_label = QLabel(f"{trans.t('dust_from_file')} ({trans.t('development_label')})")
        flat_label.setToolTip(trans.t('development_unavailable_hint'))
        flat_row.addWidget(flat_label)
        self.flat_field_group.setEnabled(False)
        legacy_layout.addWidget(self.flat_field_group)

        from ui.sensor_cleanup import SensorCleanupControls
        self.sensor_cleanup = SensorCleanupControls('', False, show_scope=False)
        legacy_layout.addWidget(self.sensor_cleanup)
        self.sensor_cleanup.path.textChanged.connect(lambda: self.update_single_folder_preview())
        self.finished.connect(lambda _result: self.sensor_cleanup.stop_inspection())

        # --- Options tab: clean, logically grouped cards -----------------
        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(10, 10, 10, 10)
        advanced_layout.setSpacing(12)
        self.fusion_parameter_panel.valuesChanged.connect(self._remember_fusion_parameters)
        self.registration_parameter_panel.valuesChanged.connect(self._remember_registration_parameters)

        order_card = QGroupBox(trans.t('batch_options_order_brightness'))
        order_layout = QVBoxLayout(order_card)
        order_layout.setSpacing(8)

        self.chk_reverse_order = QCheckBox(trans.t('chk_reverse_order'))
        self.chk_reverse_order.setToolTip(trans.t('chk_reverse_order_hint'))
        order_layout.addWidget(self.chk_reverse_order)

        self.chk_exposure_match = QCheckBox(trans.t('chk_exposure_match'))
        self.chk_exposure_match.setChecked(True)
        self.chk_exposure_match.setToolTip(trans.t('chk_exposure_match_hint'))
        order_layout.addWidget(self.chk_exposure_match)

        self.chk_base_correct = QCheckBox(trans.t('chk_base_correct'))
        self.chk_base_correct.setToolTip(trans.t('chk_base_correct_hint'))
        order_layout.addWidget(self.chk_base_correct)
        advanced_layout.addWidget(order_card)

        postprocess_card = QGroupBox(trans.t('batch_options_postprocess'))
        postprocess_layout = QVBoxLayout(postprocess_card)
        postprocess_layout.setSpacing(8)

        self.chk_sharpen_output = QCheckBox(trans.t('chk_sharpen_output'))
        self.chk_sharpen_output.setToolTip(trans.t('chk_sharpen_output_hint'))
        postprocess_layout.addWidget(self.chk_sharpen_output)
        self.sharpen_strength_spin = QSpinBox()
        self.sharpen_strength_spin.setRange(0, MAX_SHARPEN_STRENGTH)
        self.sharpen_strength_spin.setValue(DEFAULT_SHARPEN_STRENGTH)
        self.sharpen_strength_spin.setSuffix('%')
        self.sharpen_strength_spin.setKeyboardTracking(False)
        strength_label = QLabel(trans.t('render_sharpen_strength'))
        strength_label.setBuddy(self.sharpen_strength_spin)
        self.sharpen_strength_row = QWidget()
        strength_row = QHBoxLayout(self.sharpen_strength_row)
        strength_row.setContentsMargins(20, 0, 0, 0)
        strength_row.addWidget(strength_label)
        strength_row.addWidget(self.sharpen_strength_spin)
        strength_row.addStretch()
        postprocess_layout.addWidget(self.sharpen_strength_row)
        self.chk_sharpen_output.toggled.connect(self._sync_sharpen_strength)
        advanced_layout.addWidget(postprocess_card)

        last_dir = self._last_pick_dir()
        if last_dir:
            self.path_input.setText(last_dir)
        # Reflect the cached availability before seeding so the saved method is
        # applied last and remains the current item even when it is disabled.
        self._apply_fusion_availability()
        self._seed_from_parent()
        self.fusion_combo.currentIndexChanged.connect(self._sync_method_controls)
        self.align_combo.currentIndexChanged.connect(self._sync_registration_controls)
        self._fusion_has_specs = False
        self._registration_has_specs = False
        self._sync_registration_controls()
        self._sync_method_controls()

        settings_tabs.addTab(self._settings_page(self.options_group), trans.t('batch_section_method'))
        settings_tabs.addTab(self._settings_page(output_group), trans.t('batch_output_group'))
        settings_tabs.addTab(self._settings_page(advanced), trans.t('batch_section_options'))
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        start_btn = QPushButton(trans.t('batch_btn_start'))
        start_btn.setObjectName("primaryButton")
        start_btn.clicked.connect(self.start_batch_processing)
        self.btn_start = start_btn
        button_layout.addWidget(start_btn)
        
        cancel_btn = QPushButton(trans.t('batch_btn_cancel'))
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        # Buttons live in the outer (non-scrolling) layout, pinned to the
        # bottom of the dialog.
        button_bar = QWidget()
        button_bar.setLayout(button_layout)
        button_layout.setContentsMargins(14, 8, 14, 10)
        # A separator makes it obvious the bar is fixed while the page above
        # scrolls under it; without it the last row looked clipped by the bar.
        button_bar.setObjectName("dialogButtonBar")
        # A plain QWidget ignores background rules from the stylesheet unless
        # it is told to paint a styled background; without this the bar stayed
        # transparent and the scrolled content showed through it.
        button_bar.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True)
        # Fixed height: the scroll area above takes the remaining space, and
        # without this the bar was laid out ON TOP of the last rows.
        button_bar.setSizePolicy(QSizePolicy.Policy.Preferred,
                                 QSizePolicy.Policy.Fixed)
        outer.addWidget(button_bar, 0)
        
        # 初始化质量滑块显示状态（确保对话框打开时滑块正确显示）
        self.on_format_changed(self.format_combo.currentText())
        self._sync_split_method_controls()

        for _control in ("rb_multiple_folders", "rb_single_folder",
                         "split_method_combo",
                         "btn_add_folder", "btn_add_single_folder",
                         "folder_list", "format_combo", "rb_subfolder",
                         "subfolder_name", "rb_same_folder",
                         "rb_custom_folder", "custom_folder_path",
                         "fusion_combo", "align_combo", "chk_reverse_order",
                         "chk_exposure_match", "chk_base_correct", "chk_grouping",
                         "chk_sharpen_output", "sharpen_strength_spin",
                         "group_size_spin", "group_overlap_spin", "btn_start"):
            _widget = getattr(self, _control, None)
            if _widget is not None:
                _widget.setProperty("controlId", _control)
        split.fit_dialog(960, 620)

    @staticmethod
    def _settings_page(content):
        page = QScrollArea()
        page.setFrameShape(QFrame.Shape.NoFrame)
        page.setWidgetResizable(True)
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.addWidget(content)
        layout.addStretch()
        page.setWidget(host)
        return page


    def _fit_folder_list(self) -> None:
        """Show up to six complete rows, even for paths that scroll sideways."""
        queue = self.folder_list
        rows = min(queue.count(), 6)
        if rows:
            item_height = sum(queue.sizeHintForRow(row) for row in range(rows))
            chrome = 2 * queue.frameWidth() + queue.horizontalScrollBar().sizeHint().height()
            queue.setFixedHeight(item_height + chrome)


    def _update_folder_placeholder(self) -> None:
        """Show the empty-state hint only while the list has no rows."""
        placeholder = getattr(self, "_folder_list_placeholder", None)
        if placeholder is None:
            return
        empty = self.folder_list.count() == 0
        placeholder.setVisible(empty)
        self.folder_list.setVisible(not empty)
        self._fit_folder_list()

    def _drop_drag_enter(self, event) -> None:
        """Accept a drag that carries at least one local directory."""
        if self._dragged_folders(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragEnterEvent(self, event) -> None:
        self._drop_drag_enter(event)

    def dragMoveEvent(self, event) -> None:
        self._drop_drag_enter(event)

    def dropEvent(self, event) -> None:
        self._drop_event(event)

    @staticmethod
    def _dragged_folders(event) -> list:
        """Local directory paths in a drag, in the order they were dropped.

        Files are ignored here on purpose: the folder list is a queue of
        stacks, and silently turning loose images into a stack would make the
        list mean two different things.
        """
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return []
        folders = []
        for url in mime.urls():
            path = url.toLocalFile()
            if path and os.path.isdir(path):
                folders.append(os.path.normpath(path))
        return folders

    def _drop_event(self, event) -> None:
        folders = self._dragged_folders(event)
        if not folders:
            event.ignore()
            return
        event.acceptProposedAction()
        # Dropping folders is the multi-folder workflow; switching the radio
        # here avoids "I dropped five stacks but nothing will be processed"
        # when the dialog was left on single-folder mode.
        if not self.rb_multiple_folders.isChecked():
            self.rb_multiple_folders.setChecked(True)
            self.on_import_mode_changed()
        self.add_folders_to_list(folders)

    def showEvent(self, event):
        """Warm the import worker module before the first drop or pick.

        Importing core.workers pulls in the fusion pipeline (~0.6 s). Doing it
        here, while the dialog is appearing, keeps the first folder drop from
        stalling on module import; the work itself is still on the worker.
        """
        super().showEvent(event)
        if not getattr(self, "_import_module_ready", False):
            self._import_module_ready = True
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self._warm_import_module)

    @staticmethod
    def _warm_import_module() -> None:
        try:
            from core.workers import BatchImportWorker  # noqa: F401
        except Exception:
            pass

    def add_path_from_field(self) -> None:
        """Add whatever path was typed, expanding env vars and quotes.

        Several paths at once are accepted (separated by ';'): pasting a list
        from a script or Explorer's address bar is common.
        """
        raw = self.path_input.text().strip()
        if not raw:
            return
        added, rejected = 0, []
        for chunk in raw.replace('\n', ';').split(';'):
            candidate = chunk.strip().strip('"').strip("'")
            if not candidate:
                continue
            expanded = os.path.expandvars(os.path.expanduser(candidate))
            if os.path.isdir(expanded):
                if expanded not in self.folder_paths:
                    self.add_folder_to_list(expanded)
                    added += 1
            else:
                rejected.append(candidate)
        if added:
            self.path_input.clear()
            from utils.settings_store import get_settings
            get_settings().setValue(BATCH_PICK_DIR_KEY,
                                    os.path.dirname(self.folder_paths[-1]))
        if rejected:
            self.path_input.setText("; ".join(rejected))
            self.path_input.setToolTip(trans.t('batch_path_invalid'))

    def add_folders(self):
        """Pick one or more stack folders."""
        dialog = FileDialog.for_directories(
            self, trans.t('batch_pick_title'), self._last_pick_dir(), multiple=True)
        self._fit_picker_sidebar(dialog)

        # 执行对话框并获取结果
        if dialog.exec():
            selected_dirs = dialog.selectedFiles()
            selected_dirs = [d for d in selected_dirs if os.path.isdir(d)]

            # Rows appear immediately; only the thumbnails decode on the
            # worker, so validation and Start see the selection right away.
            self.add_folders_to_list(selected_dirs)

            if selected_dirs:
                # Remember this pick so the next "Add" opens nearby. Stored
                # under a batch-only key: the shared ui/last_dialog_dir belongs
                # to single-file dialogs, and a batch folder there would make
                # the next "Save As" start somewhere unrelated.
                from utils.settings_store import get_settings
                get_settings().setValue(BATCH_PICK_DIR_KEY,
                                        os.path.dirname(selected_dirs[-1]))


    def _fit_picker_sidebar(self, dialog) -> None:
        """Give the place list just enough width for its longest label."""
        from PyQt6.QtWidgets import QSplitter, QListView

        splitter = dialog.findChild(QSplitter) if hasattr(dialog, "findChild") else None
        if splitter is None or splitter.count() < 2:
            return
        sidebar = splitter.widget(0)
        if not isinstance(sidebar, QListView) or sidebar.model() is None:
            return
        metrics = sidebar.fontMetrics()
        widest = 0
        model = sidebar.model()
        for row in range(model.rowCount()):
            text = model.index(row, 0).data() or ""
            widest = max(widest, metrics.horizontalAdvance(str(text)))
        # icon + padding + a little slack, clamped so a long folder name cannot
        # eat the file view.
        width = max(120, min(230, widest + 58))
        sidebar.setMaximumWidth(width)
        sizes = splitter.sizes()
        if len(sizes) >= 2 and sizes[0] != width:
            sizes[0] = width
            sizes[1] = max(1, sum(sizes) - width)
            splitter.setSizes(sizes)

    def _last_pick_dir(self) -> str:
        from utils.settings_store import get_settings
        value = str(get_settings().value(BATCH_PICK_DIR_KEY, "") or "")
        return value if value and os.path.isdir(value) else ""

    # --- Background folder import ---------------------------------------
    #
    # Every entry point that needs decoded pixels (thumbnails, split previews,
    # drag-and-drop preload) goes through one worker. The dialog keeps only
    # request state; QPixmap/QIcon are built in the slot because Qt GUI
    # objects cannot be created off the GUI thread.

    def _start_import(self, mode: str, on_thumbnails=None, on_stack=None,
                      folder_paths=None, folder_path=None,
                      sort_by: str = 'filename', scale_factor=None) -> bool:
        if mode == 'stack' and self.chk_grouping.isChecked():
            QMessageBox.warning(self, trans.t('msg_error'), trans.t('batch_group_split_error'))
            return False
        if self._import_busy:
            if mode == 'stack' and self._import_mode == 'thumbnails':
                unfinished = [path for path in self._active_thumbnail_paths
                              if path in self.folder_paths
                              and path not in self._invalid_thumbnail_paths
                              and path not in self._applied_thumbnail_paths]
                pending = unfinished + self._pending_thumbnails
                self.cancel_imports()
                self._pending_thumbnails = pending
            else:
                return False
        from core.workers import BatchImportWorker
        self._import_generation += 1
        worker = BatchImportWorker(
            mode,
            folder_paths=folder_paths,
            folder_path=folder_path,
            scale_factor=(self.load_scale_factor if scale_factor is None
                          else scale_factor),
            sort_by=sort_by,
            exclude_paths=([self.sensor_cleanup.path.text().strip()]
                           if mode == 'thumbnails' and self.sensor_cleanup.path.text().strip() else []),
            thread_count=self._runtime_settings.get('thread_count', 0),
            resource_preference=self._runtime_settings.get(
                'resource_preference', 'balanced'),
        )
        worker.import_generation = self._import_generation
        self._import_worker = worker
        self._import_busy = True
        self._import_mode = mode
        self._applied_thumbnail_paths = set()
        self._active_thumbnail_paths = tuple(folder_paths or ()) if mode == 'thumbnails' else ()
        self._thumbnail_done = 0
        self._import_callbacks = {'thumbnails': on_thumbnails,
                                  'stack': on_stack}
        _track_import_worker(worker)
        self._set_import_busy(True)

        # Bound methods of this QObject, not lambdas: Qt disconnects them
        # automatically when the dialog is destroyed, so a worker finishing
        # after the dialog closed cannot call into a deleted window.
        if mode == 'stack':
            worker.stack_loaded.connect(self._on_import_stack_ready)
        else:
            if hasattr(worker, 'folder_loaded'):
                worker.folder_loaded.connect(self._on_import_folder_ready)
            worker.folders_loaded.connect(self._on_import_folders_ready)
        worker.finished.connect(self._on_import_finished)
        worker.start()
        return True

    def _active_import_sender(self):
        worker = self.sender()
        return (worker is not None and worker is self._import_worker and
                getattr(worker, 'import_generation', None) == self._import_generation)

    def _on_import_folder_ready(self, folder_path, image) -> None:
        if not self._active_import_sender() or folder_path in self._applied_thumbnail_paths:
            return
        self._applied_thumbnail_paths.add(folder_path)
        self._thumbnail_done += 1
        self._update_import_status_labels()
        callback = self._import_callbacks.get('thumbnails')
        if callback is not None and folder_path not in self._invalid_thumbnail_paths:
            callback([(folder_path, image)])

    def _on_import_folders_ready(self, results) -> None:
        if not self._active_import_sender():
            return
        callback = self._import_callbacks.pop('thumbnails', None)
        if callback is not None:
            remaining = [
                (path, image) for path, image in results
                if path not in self._invalid_thumbnail_paths
                and path not in self._applied_thumbnail_paths
            ]
            if remaining:
                self._applied_thumbnail_paths.update(path for path, _ in remaining)
                callback(remaining)

    def _on_import_stack_ready(self, folder, entries, message) -> None:
        if not self._active_import_sender():
            return
        callback = self._import_callbacks.pop('stack', None)
        if callback is not None:
            callback(folder, entries, message)

    def _on_import_finished(self) -> None:
        if not self._active_import_sender():
            return
        self._import_worker = None
        self._import_callbacks = {}
        self._import_busy = False
        self._import_mode = None
        self._invalid_thumbnail_paths.clear()
        self._applied_thumbnail_paths.clear()
        self._active_thumbnail_paths = ()
        self._thumbnail_done = 0
        self._pump_folder_thumbnails()
        self._set_import_busy(self._import_busy)

    def _update_import_status_labels(self) -> None:
        thumb_label = getattr(self, '_thumbnail_status_label', None)
        if thumb_label is not None:
            if self._import_busy and self._import_mode == 'thumbnails' and self._active_thumbnail_paths:
                total = len(self._active_thumbnail_paths)
                done = min(self._thumbnail_done, total)
                thumb_label.setText(trans.t('batch_loading_thumbnails').format(done=done, total=total))
                thumb_label.show()
            else:
                thumb_label.clear()
                thumb_label.hide()
        stack_label = getattr(self, '_stack_import_status_label', None)
        if stack_label is not None:
            if self._import_busy and self._import_mode == 'stack':
                stack_label.setText(trans.t('msg_loading_stack'))
                stack_label.show()
            elif self._adaptive_busy:
                done, total = self._adaptive_progress
                stack_label.setText(
                    trans.t('batch_preview_adaptive_analyzing').format(done=done, total=total)
                )
                stack_label.show()
            else:
                stack_label.clear()
                stack_label.hide()

    def _set_import_busy(self, busy: bool) -> None:
        """Disable only actions that require a single-folder stack scan."""
        stack_import_busy = bool(busy and self._import_mode == 'stack')
        stack_busy = bool(stack_import_busy or self._adaptive_busy)
        if hasattr(self, 'chk_grouping'):
            # Recompute rather than blanket-enable: an incompatible mode must
            # keep grouping disabled once the import finishes.
            self._sync_option_states()
        btn_start = getattr(self, "btn_start", None)
        if btn_start is not None:
            btn_start.setEnabled(not stack_busy)
        btn_single = getattr(self, "btn_add_single_folder", None)
        if btn_single is not None:
            btn_single.setEnabled(not stack_import_busy)
        self._update_import_status_labels()

    def _cancel_adaptive_split(self) -> None:
        """Stop any live adaptive split preview worker and drop late signals."""
        worker = self._adaptive_worker
        self._adaptive_generation += 1
        self._adaptive_worker = None
        self._adaptive_busy = False
        self._adaptive_progress = (0, 0)
        if worker is not None:
            worker.cancel()
        self._set_import_busy(self._import_busy)

    def cancel_imports(self) -> None:
        """Ask the live import to stop and drop its result."""
        worker = self._import_worker
        self._import_generation += 1
        self._import_busy = False
        self._import_mode = None
        self._import_worker = None
        self._import_callbacks = {}
        self._pending_thumbnails.clear()
        self._invalid_thumbnail_paths.clear()
        self._applied_thumbnail_paths.clear()
        self._active_thumbnail_paths = ()
        self._thumbnail_done = 0
        self._cancel_adaptive_split()
        if worker is not None:
            # The worker is tracked at module level, so it stays alive until
            # it really stops even if this dialog is being destroyed.
            worker.cancel()

    def done(self, result):
        self.cancel_imports()
        super().done(result)

    def closeEvent(self, event):
        self.cancel_imports()
        super().closeEvent(event)

    def reject(self):
        self.cancel_imports()
        super().reject()

    def accept(self):
        if (self._import_busy and self._import_mode == 'stack') or self._adaptive_busy:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, trans.t('batch_title'),
                                    trans.t('batch_import_in_progress'))
            return
        self.cancel_imports()
        super().accept()

    def _append_folder_row(self, folder_path: str) -> None:
        """Show the folder immediately; its icon arrives with the worker."""
        item = QListWidgetItem(
            f"{os.path.basename(folder_path)}\n{folder_path}")
        self.folder_list.addItem(item)
        self.folder_thumbnails.append(None)
        self._update_folder_placeholder()

    def add_folders_to_list(self, folder_paths) -> None:
        """Register folders now, decode their thumbnails in the background.

        The paths and rows are recorded synchronously so the rest of the
        dialog (validation, Start) sees the selection immediately; only the
        preview icon depends on the worker.
        """
        for folder_path in folder_paths:
            if folder_path in self.folder_paths:
                continue
            self.folder_paths.append(folder_path)
            self._append_folder_row(folder_path)
            self._pending_thumbnails.append(folder_path)
        self._pump_folder_thumbnails()

    def _pump_folder_thumbnails(self):
        if self._import_busy:
            return
        pending = [path for path in self._pending_thumbnails if path in self.folder_paths]
        self._pending_thumbnails.clear()
        if pending:
            self._start_import('thumbnails', folder_paths=pending,
                               on_thumbnails=self._apply_thumbnails)

    def _apply_thumbnails(self, results) -> None:
        """Attach decoded first frames to the rows that already exist."""
        from core.image_loader import ImageStackLoader
        loader = ImageStackLoader()
        for folder_path, first in results:
            if folder_path not in self.folder_paths:
                continue
            if first is None:
                item = self.folder_list.item(self.folder_paths.index(folder_path))
                if item is not None:
                    item.setToolTip(trans.t('converter_no_preview'))
                continue
            index = self.folder_paths.index(folder_path)
            thumbnail = loader.create_thumbnails([first], thumb_size=60)[0]
            if index < len(self.folder_thumbnails):
                self.folder_thumbnails[index] = thumbnail
            item = self.folder_list.item(index)
            if item is not None:
                item.setIcon(QIcon(thumbnail))
        self._fit_folder_list()

    def add_folder_to_list(self, folder_path: str) -> None:
        """添加文件夹到批处理列表（用于拖入场景）"""
        if folder_path in self.folder_paths:
            return
        self.add_folders_to_list([folder_path])

    def add_single_folder_to_list(self, folder_list):
        """添加单个文件夹到列表"""
        
        folder_path = FileDialog.getExistingDirectory(
            self, "Select Image Stack Folder", ""
        )
        
        if folder_path and folder_path not in [folder_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(folder_list.count())]:
            folder_name = os.path.basename(folder_path)
            item = QListWidgetItem(f"{folder_name}\n{folder_path}")
            item.setData(Qt.ItemDataRole.UserRole, folder_path)
            folder_list.addItem(item)
    
    def add_multiple_folders_to_list(self, folder_list):
        """添加多个文件夹到列表"""
        
        # 使用循环添加多个文件夹
        first_time = True
        while True:
            if first_time:
                folder_path = FileDialog.getExistingDirectory(
                    self, "Select Image Stack Folders (click Cancel when done)", ""
                )
                first_time = False
            else:
                folder_path = FileDialog.getExistingDirectory(
                    self, "Select Another Folder or click Cancel when done", ""
                )
            
            if folder_path and folder_path not in [folder_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(folder_list.count())]:
                folder_name = os.path.basename(folder_path)
                item = QListWidgetItem(f"{folder_name}\n{folder_path}")
                item.setData(Qt.ItemDataRole.UserRole, folder_path)
                folder_list.addItem(item)
            else:
                break
            
            # 每次添加后询问是否继续
            reply = QMessageBox.question(
                self, "Continue?", 
                "Do you want to add more folders?", 
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                break
    
    def remove_selected_from_list(self, folder_list):
        """从列表中移除选中的文件夹"""
        selected_items = folder_list.selectedItems()
        for item in selected_items:
            row = folder_list.row(item)
            folder_list.takeItem(row)
    
    def remove_selected_folders(self):
        """移除选中的文件夹"""
        selected_items = self.folder_list.selectedItems()
        if not selected_items:
            return
        
        # Take rows in DESCENDING order: popping the lowest row first shifts
        # every later index, so a Ctrl multi-select removed the wrong folders.
        rows = sorted((self.folder_list.row(it) for it in selected_items),
                      reverse=True)
        for row in rows:
            self.folder_list.takeItem(row)
            path = self.folder_paths.pop(row)
            self._pending_thumbnails = [p for p in self._pending_thumbnails if p != path]
            if self._import_busy:
                self._invalid_thumbnail_paths.add(path)
            if row < len(self.folder_thumbnails):
                self.folder_thumbnails.pop(row)
        self._update_folder_placeholder()

    def on_output_option_changed(self):
        """输出选项改变时的处理"""
        self.subfolder_controls.setVisible(self.rb_subfolder.isChecked())
        self.custom_folder_controls.setVisible(self.rb_custom_folder.isChecked())
    
    def on_import_mode_changed(self):
        """导入模式改变时的处理"""
        is_multiple = self.rb_multiple_folders.isChecked()
        
        self.folder_group.setVisible(is_multiple)
        self.single_folder_group.setVisible(not is_multiple)
        
        # 初始化质量滑块显示状态
        self.on_format_changed(self.format_combo.currentText())
        # Single-folder mode cannot use grouping; the state is disabled and
        # explained here instead of surfacing a modal only after Start.
        self._sync_option_states()
    
    def _current_split_method(self) -> str:
        """Return the active single-folder split method id."""
        data = self.split_method_combo.currentData()
        if data in ('adaptive', 'count', 'time_threshold', 'exposure'):
            return data
        index = self.split_method_combo.currentIndex()
        return ('adaptive', 'count', 'time_threshold', 'exposure')[min(max(index, 0), 3)]

    def _sync_split_method_controls(self) -> None:
        """Show only the controls relevant to the selected split method."""
        method = self._current_split_method()
        is_adaptive = (method == 'adaptive')
        self.param_row_widget.setVisible(not is_adaptive)
        self.adaptive_hint_label.setVisible(is_adaptive)
        if method != self._last_split_method:
            with QSignalBlocker(self.param_spinbox):
                if method == 'count':
                    self.param_label.setText(trans.t('batch_images_per_stack'))
                    self.param_spinbox.setDecimals(0)
                    self.param_spinbox.setSingleStep(1)
                    self.param_spinbox.setRange(2, 1000)
                    self.param_spinbox.setValue(5)
                    self.param_unit_label.setText(trans.t('batch_unit_images'))
                elif method == 'time_threshold':
                    self.param_label.setText(trans.t('batch_time_threshold'))
                    self.param_spinbox.setDecimals(0)
                    self.param_spinbox.setSingleStep(1)
                    self.param_spinbox.setRange(1, 3600)
                    self.param_spinbox.setValue(5)
                    self.param_unit_label.setText(trans.t('batch_unit_seconds'))
                elif method == 'exposure':
                    self.param_label.setText(trans.t('batch_ev_step'))
                    self.param_spinbox.setDecimals(2)
                    self.param_spinbox.setSingleStep(0.33)
                    self.param_spinbox.setRange(0.10, 3.00)
                    self.param_spinbox.setValue(0.33)
                    self.param_unit_label.setText(trans.t('batch_unit_ev'))
            self._last_split_method = method

    def on_split_method_changed(self, _index=None):
        """分割方式改变时的处理"""
        self._sync_split_method_controls()
        self.update_single_folder_preview()

    def on_format_changed(self, format_text):
        """格式改变时的处理 - 控制质量滑块显示"""
        is_jpg = format_text.upper() == "JPG"
        self.quality_label.setVisible(is_jpg)
        self.quality_slider.setVisible(is_jpg)
        self.quality_value_label.setVisible(is_jpg)
        self.depth_hint.setVisible(format_text.upper() in ("JPG", "BMP"))

    def _effective_single_folder_entries(self):
        from utils.path_identity import same_path
        mask = self.sensor_cleanup.path.text().strip()
        return [entry for entry in self.single_folder_images_with_times
                if not mask or not same_path(entry[0], mask)]

    @staticmethod
    def _adaptive_entry_key(entry) -> tuple:
        path = entry[0] if len(entry) > 0 else ""
        frame = entry[1] if len(entry) > 1 else None
        export_name = entry[3] if len(entry) > 3 else path
        page = getattr(frame, 'page', None)
        return (os.path.normcase(os.path.normpath(str(path))), export_name, page, id(frame) if frame is not None else 0)

    @staticmethod
    def _format_adaptive_preview(total_images: int, stacks: list) -> str:
        group_count = len(stacks)
        if group_count <= 1:
            return trans.t('batch_preview_fmt_adaptive').format(total_images, group_count)
        sizes = [str(len(stack)) for stack in stacks]
        if len(sizes) > 6:
            summary = " / ".join(sizes[:5]) + " / …"
        else:
            summary = " / ".join(sizes)
        return trans.t('batch_preview_fmt_adaptive_groups').format(total_images, group_count, summary)

    def _active_adaptive_sender(self) -> bool:
        worker = self.sender()
        return (
            worker is not None
            and worker is self._adaptive_worker
            and getattr(worker, 'adaptive_generation', None) == self._adaptive_generation
        )

    def _start_adaptive_split(self, entries: list) -> None:
        from core.workers import AdaptiveSplitWorker

        self._cancel_adaptive_split()
        self._adaptive_generation += 1
        worker = AdaptiveSplitWorker(
            entries,
            thread_count=self._runtime_settings.get('thread_count', 0),
        )
        worker.adaptive_generation = self._adaptive_generation
        worker.adaptive_entries = list(entries)
        self._adaptive_worker = worker
        self._adaptive_busy = True
        self._adaptive_progress = (0, len(entries))
        self.single_folder_stacks = []
        self.preview_label.setText(
            trans.t('batch_preview_adaptive_analyzing').format(done=0, total=len(entries))
        )
        _track_import_worker(worker)
        self._set_import_busy(self._import_busy)

        worker.progress.connect(self._on_adaptive_progress)
        worker.split_ready.connect(self._on_adaptive_split_ready)
        worker.failed.connect(self._on_adaptive_split_failed)
        worker.finished.connect(self._on_adaptive_split_finished)
        worker.start()

    def _on_adaptive_progress(self, done: int, total: int) -> None:
        if not self._active_adaptive_sender():
            return
        self._adaptive_progress = (done, total)
        self.preview_label.setText(
            trans.t('batch_preview_adaptive_analyzing').format(done=done, total=total)
        )
        self._update_import_status_labels()

    def _on_adaptive_split_ready(self, stacks, descriptors) -> None:
        if not self._active_adaptive_sender():
            return
        worker = self._adaptive_worker
        entries = getattr(worker, 'adaptive_entries', ())
        if descriptors is not None and len(descriptors) == len(entries):
            for entry, desc in zip(entries, descriptors):
                self._adaptive_descriptor_cache[self._adaptive_entry_key(entry)] = desc
        self.single_folder_stacks = list(stacks or [])
        self.preview_label.setText(self._format_adaptive_preview(len(entries), self.single_folder_stacks))

    def _on_adaptive_split_failed(self, _message: str) -> None:
        if not self._active_adaptive_sender():
            return
        worker = self._adaptive_worker
        entries = list(getattr(worker, 'adaptive_entries', ()))
        self.single_folder_stacks = [entries] if entries else []
        self.preview_label.setText(self._format_adaptive_preview(len(entries), self.single_folder_stacks))

    def _on_adaptive_split_finished(self) -> None:
        if not self._active_adaptive_sender():
            return
        self._adaptive_worker = None
        self._adaptive_busy = False
        self._adaptive_progress = (0, 0)
        self._set_import_busy(self._import_busy)

    def update_single_folder_preview(self):
        """更新单文件夹预览"""
        self._cancel_adaptive_split()
        if not self.single_folder_images_with_times:
            self.single_folder_stacks = []
            self.preview_label.setText(trans.t('batch_preview_no_folder'))
            return
        
        split_method = self._current_split_method()
        from core.image_loader import ImageStackLoader
        loader = ImageStackLoader()
        from utils.sorting import natural_sort_key
        entries = self._effective_single_folder_entries()
        if split_method == 'time_threshold':
            entries.sort(key=lambda item: (
                item[2] is None, item[2] or 0,
                natural_sort_key(item[3] if len(item) > 3 else item[0]),
            ))
        else:
            entries.sort(key=lambda item: natural_sort_key(item[3] if len(item) > 3 else item[0]))

        if split_method == 'adaptive':
            if len(entries) <= 1:
                self.single_folder_stacks = [entries] if entries else []
                self.preview_label.setText(self._format_adaptive_preview(len(entries), self.single_folder_stacks))
                return
            keys = [self._adaptive_entry_key(entry) for entry in entries]
            if all(key in self._adaptive_descriptor_cache for key in keys):
                from core.adaptive_split import split_entries_adaptive
                cached = [self._adaptive_descriptor_cache[key] for key in keys]
                stacks, _descs, _sims = split_entries_adaptive(entries, descriptors=cached)
                self.single_folder_stacks = stacks
                self.preview_label.setText(self._format_adaptive_preview(len(entries), stacks))
                return
            self._start_adaptive_split(entries)
            return

        if split_method == 'count':
            param_value = int(self.param_spinbox.value())
            stacks = loader.split_by_count(entries, param_value)
            self.preview_label.setText(trans.t('batch_preview_fmt_count').format(len(entries), len(stacks), param_value))
        elif split_method == 'time_threshold':
            param_value = float(self.param_spinbox.value())
            stacks = loader.split_by_time_threshold(entries, param_value)
            self.preview_label.setText(trans.t('batch_preview_fmt_time').format(len(entries), len(stacks), param_value))
        else:
            from core.exposure_split import read_exposure_infos, split_by_exposure
            ev_step = float(self.param_spinbox.value())
            paths = tuple(item[0] for item in entries)
            cache = getattr(self, '_exposure_info_cache', None)
            if cache is None or cache[0] != paths:
                cache = (paths, read_exposure_infos(paths))
                self._exposure_info_cache = cache
            stacks = split_by_exposure(entries, ev_step=ev_step, exposure_map=cache[1])
            self.preview_label.setText(
                trans.t('batch_preview_fmt_exposure').format(len(entries), len(stacks), f"{ev_step:g}")
            )
        self.single_folder_stacks = stacks
    
    def add_single_folder(self):
        """添加单个文件夹（自动分割）"""

        folder_path = FileDialog.getExistingDirectory(
            self, trans.t('batch_btn_select_split'), ""
        )

        if not folder_path:
            return

        self._start_import(
            'stack', folder_path=folder_path,
            on_stack=self._apply_single_folder_import)
        return

    def _apply_single_folder_import(self, folder_path, entries, message):
        """Show a decoded split preview, or report why it could not load."""
        if not entries:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                trans.t("msg_load_failed"),
                trans.t("msg_load_images_failed_text").format(message=message),
            )
            return
        self._adaptive_descriptor_cache.clear()
        self.single_folder_images_with_times = entries
        self.single_folder_folder_path = folder_path

        # 更新路径显示
        folder_name = os.path.basename(folder_path)
        self.single_folder_path_label.setText(trans.t('import_folder_display').format(folder_name))
        self.single_folder_path_label.setToolTip(folder_path)

        self.update_single_folder_preview()

    
    def get_import_mode(self):
        """获取导入模式"""
        if self.rb_single_folder.isChecked():
            return "single_folder"
        return "multiple_folders"
    
    def get_split_settings(self):
        """获取分割设置"""
        if self.rb_multiple_folders.isChecked():
            return None, None
        
        split_method = self._current_split_method()
        if split_method == 'adaptive':
            return 'adaptive', None
        param_value = self.param_spinbox.value()
        if split_method == 'count':
            return 'count', int(param_value)
        if split_method == 'time_threshold':
            return 'time_threshold', float(param_value)
        return 'exposure', float(param_value)
    

    def browse_output_folder(self):
        """浏览输出文件夹"""
        
        folder_path = FileDialog.getExistingDirectory(
            self, trans.t('batch_out_custom'), ""
        )
        
        if folder_path:
            self.custom_folder_path.setText(folder_path)
    
    def get_output_settings(self):
        """获取输出设置"""
        if self.rb_subfolder.isChecked():
            return "subfolder", self.subfolder_name.text()
        elif self.rb_same_folder.isChecked():
            return "same", None
        else:  # custom folder
            return "custom", self.custom_folder_path.text()
    
    def _fill_method_combo(self, combo, specs, allow_none=False):
        combo.clear()
        if allow_none:
            combo.addItem(trans.t('check_align_none'), 'none')
        for spec in specs:
            combo.addItem(trans.t(spec.label_key), spec.id)
            if spec.hint_key:
                combo.setItemData(combo.count() - 1, trans.t(spec.hint_key),
                                  Qt.ItemDataRole.ToolTipRole)

    def _seed_from_parent(self):
        window = self.parent_window
        if window is None:
            return
        method = window.current_fusion_method()
        if method != 'none':
            get_fusion(method)
        alignment = window.current_alignment_method()
        alignment_steps(alignment)
        self.fusion_combo.setCurrentIndex(self.fusion_combo.findData(method))
        self.align_combo.setCurrentIndex(self.align_combo.findData(alignment))
        self.chk_reverse_order.setChecked(bool(getattr(window, 'stack_reverse_order', False)))
        self.chk_sharpen_output.setChecked(bool(getattr(window, 'sharpen_output', False)))
        self.sharpen_strength_spin.setValue(
            getattr(window, 'sharpen_strength', DEFAULT_SHARPEN_STRENGTH))
        getter = getattr(window, 'brightness_option_for', None)
        if callable(getter):
            method = self.fusion_combo.currentData()
            if method != 'none':
                self.chk_exposure_match.setChecked(bool(getter('exposure_match', method)))
            self.chk_base_correct.setChecked(bool(getter('base_correct', method)))

    def _apply_fusion_availability(self) -> None:
        from core.multi_focus_fusion import is_ai_fusion_available

        apply_method_availability(self.fusion_combo, fusion_algorithms(), self.parent_window,
                                  ai_available=bool(is_ai_fusion_available()))
        apply_method_availability(self.align_combo, registration_algorithms(), self.parent_window,
                                  alignment=True)

    def _method_lock_reason(self):
        self._apply_fusion_availability()
        for combo in (self.fusion_combo, self.align_combo):
            if not selected_item_enabled(combo):
                return combo.currentData(Qt.ItemDataRole.ToolTipRole) or trans.t('msg_error')
        return ''

    def _remember_fusion_parameters(self, values):
        self._fusion_drafts[self._editing_method] = dict(values)

    def _remember_registration_parameters(self, values):
        self._registration_drafts[self._editing_alignment] = dict(values)

    def _sync_registration_controls(self):
        method = self.align_combo.currentData()
        if method is None:
            return
        if method not in self._registration_drafts:
            getter = getattr(self.parent_window, 'registration_parameters_for', None)
            self._registration_drafts[method] = (getter(method) if callable(getter)
                else desktop_registration_parameters(method))
        self._editing_alignment = method
        specs = registration_parameter_specs(method)
        self.registration_parameter_panel.set_parameters(specs, self._registration_drafts[method])
        self._registration_has_specs = method != 'openfocus' and bool(self.registration_parameter_panel.editors)
        self.registration_parameter_panel.setVisible(self._registration_has_specs)
        self._sync_parameters_heading()

    def _sync_parameters_heading(self) -> None:
        """One heading covers both panels; hide it only when both are empty."""
        self.params_rule.setVisible(
            self._fusion_has_specs or self._registration_has_specs)

    def _grouping_changed(self, enabled):
        # Only a user action (or a deliberate toggle in a test) changes the
        # remembered preference; forced off states are set with blocked signals.
        if self.chk_grouping.isEnabled():
            self._grouping_pref = bool(enabled)
        self._sync_option_states()

    def _group_size_changed(self, value):
        with QSignalBlocker(self.group_overlap_spin):
            maximum = value - 2
            self.group_overlap_spin.setMaximum(maximum)
            self.group_overlap_spin.setValue(min(self._group_overlap_preference, maximum))

    def _remember_save_aligned(self, checked: bool) -> None:
        """Track the user's own choice, ignoring programmatic forcing."""
        if self.save_aligned_cb.isEnabled():
            self._save_aligned_pref = bool(checked)
        self._sync_option_states()

    def _set_save_aligned_checked(self, checked: bool) -> None:
        checked = bool(checked)
        if self.save_aligned_cb.isChecked() == checked:
            return
        with QSignalBlocker(self.save_aligned_cb):
            self.save_aligned_cb.setChecked(checked)

    def _set_grouping_checked(self, checked: bool) -> None:
        checked = bool(checked)
        if self.chk_grouping.isChecked() == checked:
            return
        with QSignalBlocker(self.chk_grouping):
            self.chk_grouping.setChecked(checked)

    def _sync_option_states(self) -> None:
        """Reflect the incompatibilities instead of only failing at Start.

        Grouping needs multi-folder input plus a fusion step and cannot share
        the run with the aligned-stack copy; alignment-only output has no
        fusion step, so fusion options have no consumer.
        """
        method = self.fusion_combo.currentData()
        fusion_off = method in (None, 'none')
        spec = None if fusion_off else get_fusion(method)

        if not self.rb_multiple_folders.isChecked():
            grouping_reason = trans.t('batch_group_split_error')
        elif fusion_off or self._save_aligned_pref:
            grouping_reason = trans.t('batch_group_output_error')
        else:
            grouping_reason = ''

        stack_busy = bool(self._import_busy and self._import_mode == 'stack')
        if grouping_reason:
            if self.chk_grouping.isChecked():
                self._grouping_pref = True
            self._set_grouping_checked(False)
        elif (not stack_busy and self._grouping_pref
              and not self.chk_grouping.isChecked()):
            self._set_grouping_checked(True)

        grouping_on = self.chk_grouping.isChecked()
        self.chk_grouping.setEnabled(not grouping_reason and not stack_busy)
        self.group_controls.setVisible(grouping_on)
        self.group_controls.setEnabled(not stack_busy)
        self.chk_grouping.setToolTip(
            grouping_reason or trans.t('batch_group_hint'))
        self.group_note.setText(grouping_reason)
        self.group_note.setVisible(bool(grouping_reason))

        align_only = method == 'none'
        if align_only:
            self.save_aligned_cb.setEnabled(False)
            self._set_save_aligned_checked(True)
            self.save_aligned_cb.setToolTip(trans.t('batch_save_aligned_forced'))
            self.save_aligned_note.setVisible(False)
        elif grouping_on:
            self.save_aligned_cb.setEnabled(False)
            self._set_save_aligned_checked(False)
            self.save_aligned_cb.setToolTip(
                trans.t('batch_group_output_error'))
            self.save_aligned_note.setText(
                trans.t('batch_group_output_error'))
            self.save_aligned_note.setVisible(True)
        else:
            self.save_aligned_cb.setEnabled(True)
            self._set_save_aligned_checked(self._save_aligned_pref)
            self.save_aligned_cb.setToolTip(trans.t('batch_save_aligned_hint'))
            self.save_aligned_note.setVisible(False)

        fusion_only_hint = trans.t('radio_no_fusion_hint')
        for box, hint_key in (
                (self.chk_exposure_match, 'chk_exposure_match_hint'),
                (self.chk_base_correct, 'chk_base_correct_hint'),
                (self.chk_sharpen_output, 'chk_sharpen_output_hint')):
            box.setEnabled(not fusion_off)
            box.setToolTip(fusion_only_hint if fusion_off
                           else trans.t(hint_key))
        self._sync_sharpen_strength()
        locked = bool(spec is not None and spec.order_sensitive)
        self.chk_reverse_order.setEnabled(not fusion_off and not locked)
        if fusion_off:
            self.chk_reverse_order.setToolTip(fusion_only_hint)
        elif locked:
            self.chk_reverse_order.setToolTip(
                trans.t('chk_reverse_order_locked_hint'))
        else:
            self.chk_reverse_order.setToolTip(
                trans.t('chk_reverse_order_hint'))

    def _sync_sharpen_strength(self) -> None:
        self.sharpen_strength_row.setEnabled(
            self.chk_sharpen_output.isEnabled() and self.chk_sharpen_output.isChecked())

    def _sync_method_controls(self) -> None:
        method = self.fusion_combo.currentData()
        previous = self._editing_method
        if previous not in (None, 'none'):
            self._exposure_drafts[previous] = self.chk_exposure_match.isChecked()
        if method not in (None, 'none'):
            if method not in self._exposure_drafts:
                getter = getattr(self.parent_window, 'brightness_option_for', None)
                self._exposure_drafts[method] = (bool(getter('exposure_match', method))
                    if callable(getter) else get_fusion(method).default_exposure_match)
            self.chk_exposure_match.setChecked(self._exposure_drafts[method])
        self._editing_method = method
        specs = () if method in (None, 'none') else get_fusion(method).parameters
        if method not in self._fusion_drafts:
            getter = getattr(self.parent_window, 'fusion_parameters_for', None)
            self._fusion_drafts[method] = (getter(method) if callable(getter) and specs
                else resolve_fusion_parameters(method) if specs else {})
        self.fusion_parameter_panel.set_parameters(specs, self._fusion_drafts[method])
        self.fusion_parameter_panel.setVisible(bool(specs))
        self._fusion_has_specs = bool(specs)
        self._sync_parameters_heading()
        self._sync_option_states()

    def get_processing_settings(self):
        """获取处理设置"""
        reason = self._method_lock_reason()
        if reason:
            raise ValueError(reason)
        format_str = self.format_combo.currentText().lower()
        
        fusion_method = self.fusion_combo.currentData() or "none"
        fusion_params = self.fusion_parameter_panel.values()
        runtime = dict(self._runtime_settings)
        runtime['base_frame'] = ('first' if fusion_method in ('openfocus_a', 'openfocus_b', 'openfocus_c')
                                 else 'last') if self.chk_base_correct.isChecked() else 'none'
        runtime['exposure_match'] = self.chk_exposure_match.isChecked()
        runtime['sharpen_output'] = self.chk_sharpen_output.isChecked()
        runtime['sharpen_strength'] = self.sharpen_strength_spin.value()
        runtime['sort_mode'] = 'desc' if self.chk_reverse_order.isChecked() else 'asc'

        align_method = self.align_combo.currentData() or "none"
        reg_methods = list(alignment_steps(align_method))
        # Registration downscale: a percentage of each stack's own width. 100%
        # keeps the main window's configured width untouched (the historical
        # behaviour); anything lower narrows it proportionally.
        ratio = int(self.downscale_slider.value())

        # 获取JPG质量设置
        jpg_quality = 100
        if format_str == "jpg":
            jpg_quality = self.quality_slider.value()
        
        # Scale-bar preferences: per-stack µm/px is auto-detected by the
        # worker's own loader from each stack's TIFF metadata; the manual
        # value here is the fallback when metadata has no calibration.
        scale_cfg = None
        if self.parent_window and getattr(self.parent_window, "scale_bar_enabled", False):
            manual = float(getattr(self.parent_window, "scale_um_per_px_manual", 0.0) or 0.0)
            scale_cfg = {
                "manual_px_um": manual if manual > 0 else None,
                "position": getattr(self.parent_window, "scale_bar_position", "bottom-right"),
                "color": getattr(self.parent_window, "scale_bar_color", "auto"),
                'font_family': getattr(self.parent_window, 'scale_bar_font_family', ''),
                'font_size': getattr(self.parent_window, 'scale_bar_font_size', 0),
                'font_bold': getattr(self.parent_window, 'scale_bar_font_bold', False),
                'text_align': getattr(self.parent_window, 'scale_bar_text_align', 'auto'),
                'custom_x': getattr(self.parent_window, 'scale_bar_custom_x', None),
                'custom_y': getattr(self.parent_window, 'scale_bar_custom_y', None),
                'target': getattr(self.parent_window, 'scale_bar_target', 'all'),
            }

        # 仅对齐批处理：没有融合步骤，必须保存对齐帧（否则整个任务无输出）
        if fusion_method == "none":
            save_aligned = True
        else:
            save_aligned = self.save_aligned_cb.isChecked()

        # Only alignment: aligned frames exported; watermark scope travels with
        # the frozen processing settings.
        from utils import watermark
        wm_settings = (watermark.settings_from_window(self.parent_window)
                       if self.parent_window is not None else None)

        return {
            "format": format_str,
            "jpg_quality": jpg_quality,
            "fusion_method": fusion_method,
            "fusion_params": fusion_params,
            "reg_methods": reg_methods,
            'alignment': align_method,
            'registration_parameters': self.registration_parameter_panel.values(),
            "reg_scale_percent": ratio,
            'group_size': self.group_size_spin.value() if self.chk_grouping.isChecked() else 0,
            'group_overlap': self.group_overlap_spin.value(),
            "save_aligned": save_aligned,  # 是否保存对齐后的图像栈
            "scale_bar": scale_cfg,
            "watermark": wm_settings,
            'defect_mask_path': self.sensor_cleanup.path.text().strip(),
            'sensor_denoise': self.sensor_cleanup.denoise.isChecked(),
            'load_scale_factor': self.load_scale_factor,
            'runtime_settings': runtime,
            'output_filename_template': str(getattr(self.parent_window, 'output_filename_template', '') or ''),
            'output_name_prefix': str(getattr(self.parent_window, 'output_name_prefix', '') or ''),
            'frozen_stacks': [list(stack) for stack in self.single_folder_stacks],
        }

    def preload_single_folder(self, folder_path: str, scale_factor: float = 1.0) -> None:
        """
        预加载单个文件夹（用于拖入场景）
        自动切换到单文件夹模式、加载文件夹、默认选择自适应（自动）分割
        """
        # 1. 切换到单文件夹模式
        self.rb_single_folder.setChecked(True)
        self.on_import_mode_changed()

        # 2. 默认使用自适应（自动）分割
        adaptive_idx = self.split_method_combo.findData('adaptive')
        self.split_method_combo.setCurrentIndex(adaptive_idx if adaptive_idx >= 0 else 0)
        self._sync_split_method_controls()

        # 3. 加载文件夹并应用缩放（后台，结果到达后更新预览）
        self.load_scale_factor = scale_factor
        self._start_import(
            'stack', folder_path=folder_path, scale_factor=scale_factor,
            on_stack=self._apply_single_folder_import)

    def preload_multiple_folders(self, folder_paths: list[str], scale_factor: float = 1.0) -> None:
        """预加载多个文件夹（用于拖入场景）

        Args:
            folder_paths: 文件夹路径列表
            scale_factor: 缩放因子（统一应用到所有文件夹）
        """
        self.rb_multiple_folders.setChecked(True)
        self.on_import_mode_changed()
        self.load_scale_factor = scale_factor
        self.add_folders_to_list(folder_paths)

    def start_batch_processing(self):
        """开始批处理"""
        from PyQt6.QtWidgets import QMessageBox
        reason = self._method_lock_reason()
        if reason:
            QMessageBox.warning(self, trans.t('msg_error'), reason)
            return
        sensor_error = self.sensor_cleanup.validate()
        if sensor_error:
            QMessageBox.warning(self, trans.t('msg_error'), sensor_error)
            return

        import_mode = self.get_import_mode()
        split_method, split_param = self.get_split_settings()
        group_size = self.group_size_spin.value() if self.chk_grouping.isChecked() else 0
        if group_size:
            if import_mode == 'single_folder':
                QMessageBox.warning(self, trans.t('msg_error'), trans.t('batch_group_split_error'))
                return
            if group_size < 2 or self.group_overlap_spin.value() > group_size - 2:
                QMessageBox.warning(self, trans.t('msg_error'), trans.t('batch_group_range_error'))
                return
            if self.fusion_combo.currentData() in (None, 'none') or self.save_aligned_cb.isChecked():
                QMessageBox.warning(self, trans.t('msg_error'), trans.t('batch_group_output_error'))
                return

        if import_mode == "multiple_folders":
            if not self.folder_paths:
                QMessageBox.warning(self, trans.t("msg_no_folders_title"), trans.t("msg_no_folders_text"))
                return
        else:
            if not self.single_folder_images_with_times:
                QMessageBox.warning(self, trans.t("msg_no_folder_title"), trans.t("msg_no_folder_text"))
                return

        output_type, output_path = self.get_output_settings()
        problem = self._validate_output_target(output_type, output_path)
        if problem:
            QMessageBox.warning(self, trans.t("msg_error"), problem)
            return

        if import_mode == 'single_folder':
            if self._adaptive_busy:
                QMessageBox.information(self, trans.t('batch_title'),
                                        trans.t('batch_import_in_progress'))
                return
            if split_method != 'adaptive' or not self.single_folder_stacks:
                self.update_single_folder_preview()
                if self._adaptive_busy:
                    QMessageBox.information(self, trans.t('batch_title'),
                                            trans.t('batch_import_in_progress'))
                    return
            entries = self._effective_single_folder_entries()
            if not entries or not self.single_folder_stacks:
                QMessageBox.warning(self, trans.t('msg_no_folder_title'), trans.t('msg_no_folder_text'))
                return
            if split_method == 'time_threshold':
                from core.image_loader import ImageStackLoader
                missing = ImageStackLoader.missing_timestamps(entries)
                if missing:
                    QMessageBox.warning(self, trans.t('msg_error'),
                                        trans.t('batch_missing_times').format(n=missing))
                    return
            if (self.fusion_combo.currentData() or 'none') != 'none' and any(
                    len(stack) < 2 for stack in self.single_folder_stacks):
                QMessageBox.warning(self, trans.t('msg_error'),
                                    trans.t('depth_need_two_frames').format(n=1))
                return

        self.accept()

    def _source_output_folders(self) -> list:
        """Folders the results are written into for the 'same'/'subfolder' modes."""
        if self.get_import_mode() == "single_folder":
            return [getattr(self, "single_folder_folder_path", "") or ""]
        return list(self.folder_paths)

    def _validate_output_target(self, output_type: str, output_path: str):
        """Return a message when the batch cannot write its results, else None.

        Checked before the dialog closes: an unwritable or missing target used
        to fail deep inside the worker loop, one opaque error per folder.
        """
        if output_type == "custom":
            target = (output_path or "").strip()
            if not target:
                return trans.t("msg_no_folder_text")
            if os.path.exists(target) and not os.path.isdir(target):
                return f"{trans.t('msg_save_failed_info_write')}\n{target}"
            probe = target if os.path.isdir(target) else os.path.dirname(os.path.abspath(target))
            if not _is_writable_dir(probe):
                return f"{trans.t('msg_save_failed_info_write')}\n{target}"
            return None

        if output_type == "subfolder" and not (output_path or "").strip():
            return trans.t("msg_no_folder_text")

        from utils.export_paths import resolve_output_subfolder
        for folder in self._source_output_folders():
            if not os.path.isdir(folder):
                return f"{trans.t('msg_save_failed_info_write')}\n{folder}"
            if output_type == 'subfolder':
                try:
                    target = resolve_output_subfolder(folder, output_path)
                except ValueError:
                    target = os.path.realpath(os.path.join(folder, output_path))
                    return f"{trans.t('msg_save_failed_info_write')}\n{target}"
            else:
                target = folder
            if os.path.exists(target) and not os.path.isdir(target):
                return f"{trans.t('msg_save_failed_info_write')}\n{target}"
            if not _is_writable_dir(target if os.path.isdir(target) else folder):
                return f"{trans.t('msg_save_failed_info_write')}\n{target}"
        return None


def _is_writable_dir(path: str) -> bool:
    """True when files can actually be created in `path`.

    os.access(W_OK) lies on Windows (it ignores the read-only attribute of
    directories and reports ACL-permitted writes that still fail), so probe
    with a real temporary file.
    """
    import tempfile

    if not path or not os.path.isdir(path):
        return False
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".openfocus_probe_", suffix=".tmp"):
            return True
    except OSError:
        return False


class FolderImportDialog(QDialog):
    """文件夹导入选择对话框 - 让用户选择是单组图像栈还是多组图像栈"""
    
    def __init__(self, folder_path: str, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.parent_window = parent
        self.setWindowTitle(trans.t('import_folder_title'))
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(trans.t('import_choice_label')))
        path = QLineEdit(folder_path)
        path.setReadOnly(True)
        path.setAccessibleName(trans.t('import_folder_display').format(os.path.basename(folder_path)))
        layout.addWidget(path)

        self.rb_single = QRadioButton(trans.t('import_option_single'))
        self.rb_single.setChecked(True)
        self.rb_single.setFocus()
        layout.addWidget(self.rb_single)

        self.rb_batch = QRadioButton(trans.t('import_option_batch'))
        layout.addWidget(self.rb_batch)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.ok_button = QPushButton(trans.t('btn_ok'))
        self.ok_button.setObjectName("primaryButton")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)
        btn_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton(trans.t('btn_cancel'))
        self.cancel_button.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_button)

        layout.addLayout(btn_layout)

    def is_single_stack(self) -> bool:
        """返回 True 表示单组图像栈，False 表示多组图像栈（批处理）"""
        return self.rb_single.isChecked()
