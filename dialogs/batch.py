# dialogs/batch.py
"""Batch processing dialogs."""

import os
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon, QTextOption
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
    QFileDialog,
    QMessageBox,
    QComboBox,
    QLineEdit,
    QRadioButton,
    QCheckBox,
)
from ui.styles import PRIMARY_BLUE, _ARROW_DOWN_DARK
from ui.right_panel import ALIGN_ITEMS, FUSION_ITEMS
from locales import trans
from widgets.stepper import StepperSpinBox


BATCH_PICK_DIR_KEY = "batch/last_pick_dir"


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
        # Height follows the screen, never past it: on a 1080p display with a
        # scaled UI the old fixed 800 px pushed the action buttons under the
        # taskbar and they could not be clicked at all.
        self.resize(800, 800)
        self.setMinimumSize(720, 480)
        
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
                ('ai_batch_size', 2), ('ai_mode', 'soft'), ('use_gpu', True))}
        
        # Dark palette only in dark mode: hard-coding it showed a dark slab in
        # light mode.
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Everything except the action buttons scrolls. The button bar must
        # stay reachable no matter how many groups are shown or how short the
        # screen is.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        import_mode_group = QGroupBox(trans.t('batch_import_mode'))
        # One row: the two modes are a single either/or choice, and stacking
        # them wasted a whole line of the dialog for no extra clarity. The
        # stretch keeps them left-aligned instead of spread across the width.
        import_mode_layout = QHBoxLayout(import_mode_group)
        import_mode_layout.setSpacing(18)

        self.rb_multiple_folders = QRadioButton(trans.t('batch_mode_multi'))
        self.rb_multiple_folders.setChecked(True)
        self.rb_multiple_folders.toggled.connect(self.on_import_mode_changed)
        import_mode_layout.addWidget(self.rb_multiple_folders)

        self.rb_single_folder = QRadioButton(trans.t('batch_mode_single'))
        self.rb_single_folder.toggled.connect(self.on_import_mode_changed)
        import_mode_layout.addWidget(self.rb_single_folder)
        import_mode_layout.addStretch()
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
        # Height follows the content between three and six rows: a fixed
        # minimum left a large empty block under two queued stacks (the list
        # was stretched by the scroll page). _fit_folder_list() reapplies it
        # whenever the list changes.
        self.folder_list.setMinimumHeight(self._folder_row_height() * 3 + 10)
        self._folder_list_placeholder = QLabel(trans.t('batch_no_folders'))
        self._folder_list_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignCenter)
        self._folder_list_placeholder.setWordWrap(True)
        self._folder_list_placeholder.setStyleSheet(
            "color: #888888; background: transparent; padding: 12px;")
        folder_layout.addWidget(self.folder_list)
        folder_layout.addWidget(self._folder_list_placeholder)
        self._update_folder_placeholder()
        
        # Remove button: only meaningful when there is something to remove.
        # Stretched across the full width under an empty list it read as the
        # primary action of the group.
        self.remove_folder_btn = QPushButton(trans.t('batch_btn_remove'))
        self.remove_folder_btn.clicked.connect(self.remove_selected_folders)
        self.remove_folder_btn.setEnabled(False)
        self.folder_list.itemSelectionChanged.connect(
            lambda: self.remove_folder_btn.setEnabled(
                bool(self.folder_list.selectedItems())))
        remove_row = QHBoxLayout()
        remove_row.addStretch()
        remove_row.addWidget(self.remove_folder_btn)
        folder_layout.addLayout(remove_row)
        
        layout.addWidget(self.folder_group)

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
        self.split_method_combo.addItems([trans.t('batch_split_fixed'), trans.t('batch_split_time')])
        self.split_method_combo.currentIndexChanged.connect(self.on_split_method_changed)
        split_method_layout.addWidget(self.split_method_combo)
        split_method_layout.addStretch()
        single_folder_layout.addLayout(split_method_layout)

        param_layout = QHBoxLayout()
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
        single_folder_layout.addLayout(param_layout)

        self.preview_label = QLabel(trans.t('batch_preview_default'))
        self.preview_label.setObjectName("mutedLabel")
        single_folder_layout.addWidget(self.preview_label)

        add_single_folder_btn = QPushButton(trans.t('batch_btn_select_split'))
        add_single_folder_btn.clicked.connect(self.add_single_folder)
        single_folder_layout.addWidget(add_single_folder_btn)

        layout.addWidget(self.single_folder_group)
        
        # Output settings: format, quality, destination and the aligned-stack
        # copy all answer one question ("what gets written where?"), so they
        # live in ONE group. The aligned-stack checkbox used to float loose at
        # the bottom of the dialog, outside every group, where it read as a
        # stray option rather than part of the output.
        output_group = QGroupBox(trans.t('batch_output_group'))
        output_layout = QVBoxLayout(output_group)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel(trans.t('batch_format_label')))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["JPG", "PNG", "BMP", "TIFF"])
        self.format_combo.setCurrentText('PNG')
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
        output_layout.addLayout(format_row)
        depth_hint = QLabel(trans.t('export_depth_hint'))
        depth_hint.setWordWrap(True)
        output_layout.addWidget(depth_hint)
        self.quality_label.setVisible(False)
        self.quality_slider.setVisible(False)
        self.quality_value_label.setVisible(False)

        output_layout.addWidget(_section_rule(trans.t('batch_output_location')))

        self.rb_subfolder = QRadioButton(trans.t('batch_out_subfolder'))
        self.rb_subfolder.setChecked(True)
        self.rb_subfolder.toggled.connect(self.on_output_option_changed)
        output_layout.addWidget(self.rb_subfolder)

        subfolder_layout = QHBoxLayout()
        subfolder_layout.addWidget(QLabel(trans.t('batch_subfolder_name')))
        self.subfolder_name = QLineEdit("OpenFocus_Output")
        subfolder_layout.addWidget(self.subfolder_name)
        output_layout.addLayout(subfolder_layout)

        self.rb_same_folder = QRadioButton(trans.t('batch_out_same'))
        self.rb_same_folder.toggled.connect(self.on_output_option_changed)
        output_layout.addWidget(self.rb_same_folder)

        self.rb_custom_folder = QRadioButton(trans.t('batch_out_custom'))
        self.rb_custom_folder.toggled.connect(self.on_output_option_changed)
        output_layout.addWidget(self.rb_custom_folder)

        custom_folder_layout = QHBoxLayout()
        self.custom_folder_path = QLineEdit()
        self.custom_folder_path.setEnabled(False)
        custom_folder_layout.addWidget(self.custom_folder_path)

        browse_btn = QPushButton(trans.t('batch_btn_browse'))
        browse_btn.clicked.connect(self.browse_output_folder)
        browse_btn.setEnabled(False)
        self.browse_btn = browse_btn
        custom_folder_layout.addWidget(browse_btn)
        output_layout.addLayout(custom_folder_layout)

        self._save_aligned_pref = False
        self.save_aligned_cb = QCheckBox(trans.t('batch_save_aligned'))
        self.save_aligned_cb.setChecked(False)
        self.save_aligned_cb.setToolTip(trans.t('batch_save_aligned_hint'))
        self.save_aligned_cb.toggled.connect(self._remember_save_aligned)
        output_layout.addWidget(self.save_aligned_cb)

        layout.addWidget(output_group)

        # Batch settings. These used to be read-only labels showing whatever the
        # main window happened to be set to, so a batch run could not use a
        # different method, alignment or exposure policy than the last manual
        # render. Every control below feeds get_processing_settings().
        self.options_group = QGroupBox(trans.t('batch_proc_options'))
        options_layout = QVBoxLayout(self.options_group)
        options_layout.setSpacing(8)

        # --- Method ------------------------------------------------------
        options_layout.addWidget(_section_rule(trans.t('batch_section_method')))

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel(trans.t('group_fusion')))
        self.fusion_combo = QComboBox()
        self._fill_method_combo(self.fusion_combo, FUSION_ITEMS)
        method_row.addWidget(self.fusion_combo, 1)
        options_layout.addLayout(method_row)

        align_row = QHBoxLayout()
        align_row.addWidget(QLabel(trans.t('group_registration')))
        self.align_combo = QComboBox()
        self._fill_method_combo(self.align_combo, ALIGN_ITEMS)
        align_row.addWidget(self.align_combo, 1)
        options_layout.addLayout(align_row)

        # --- Parameters --------------------------------------------------
        self.params_rule = _section_rule(trans.t('batch_section_params'))
        options_layout.addWidget(self.params_rule)

        # Labels and controls share one grid so every row's controls start at
        # the same x. Three HBoxes with a trailing stretch put the steppers at
        # the far right and the slider at the far left, so the block had no
        # left edge the eye could follow.
        params = QGridLayout()
        params.setColumnStretch(1, 1)
        params.setHorizontalSpacing(10)
        params.setVerticalSpacing(6)

        self.kernel_label = QLabel(trans.t('label_kernel'))
        params.addWidget(self.kernel_label, 0, 0)
        self.kernel_spin = StepperSpinBox(
            minimum=1, maximum=51, step=2, decimals=0, value=11)
        self.kernel_spin.setMinimumWidth(96)
        params.addWidget(self.kernel_spin, 0, 1,
                         Qt.AlignmentFlag.AlignLeft)

        self.smooth_label = QLabel(trans.t('label_blend_smoothing'))
        params.addWidget(self.smooth_label, 1, 0)
        self.smooth_spin = StepperSpinBox(
            minimum=0.0, maximum=8.0, step=0.2, decimals=1, value=2.0)
        self.smooth_spin.setMinimumWidth(96)
        params.addWidget(self.smooth_spin, 1, 1,
                         Qt.AlignmentFlag.AlignLeft)

        self.downscale_label = QLabel(trans.t('batch_downscale'))
        self.downscale_label.setToolTip(trans.t('batch_downscale_hint'))
        params.addWidget(self.downscale_label, 2, 0)
        scale_row = QHBoxLayout()
        self.downscale_slider = QSlider(Qt.Orientation.Horizontal)
        self.downscale_slider.setRange(20, 100)
        self.downscale_slider.setSingleStep(5)
        self.downscale_slider.setPageStep(10)
        self.downscale_slider.setValue(100)
        self.downscale_value = QLabel("100%")
        self.downscale_value.setMinimumWidth(46)
        self.downscale_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.downscale_slider.valueChanged.connect(
            lambda v: self.downscale_value.setText(f"{v}%"))
        scale_row.addWidget(self.downscale_slider, 1)
        scale_row.addWidget(self.downscale_value)
        params.addLayout(scale_row, 2, 1)
        options_layout.addLayout(params)

        # --- Options -----------------------------------------------------
        options_layout.addWidget(_section_rule(trans.t('batch_section_options')))

        self.chk_reverse_order = QCheckBox(trans.t('chk_reverse_order'))
        self.chk_reverse_order.setToolTip(trans.t('chk_reverse_order_hint'))
        options_layout.addWidget(self.chk_reverse_order)

        self.chk_exposure_match = QCheckBox(trans.t('chk_exposure_match'))
        self.chk_exposure_match.setToolTip(trans.t('chk_exposure_match_hint'))
        options_layout.addWidget(self.chk_exposure_match)

        self.chk_base_correct = QCheckBox(trans.t('chk_base_correct'))
        self.chk_base_correct.setToolTip(trans.t('chk_base_correct_hint'))
        options_layout.addWidget(self.chk_base_correct)

        last_dir = self._last_pick_dir()
        if last_dir:
            self.path_input.setText(last_dir)
        self._seed_from_parent()
        self.fusion_combo.currentIndexChanged.connect(self._sync_method_controls)
        self._sync_method_controls()

        # Order the dialog by workflow: pick inputs -> choose how to process
        # -> decide where the results go. The processing group used to sit
        # after every output control, so the thing you configure first was
        # the last thing on the page.
        layout.insertWidget(layout.indexOf(output_group),
                            self.options_group)
        # Leftover page height collects here. Without a stretch the scroll
        # page stretched the folder group to fill the viewport, leaving a
        # large empty band inside it below the queued stacks.
        layout.addStretch()
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        start_btn = QPushButton(trans.t('batch_btn_start'))
        start_btn.setObjectName("primaryButton")
        start_btn.clicked.connect(self.start_batch_processing)
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
    
    def _fit_folder_list(self) -> None:
        """Size the list to its rows (3..6) instead of stretching it."""
        rows = max(3, min(6, self.folder_list.count()))
        # setFixedHeight first, then relax the minimum: doing it the other way
        # round the minimum won and the list never grew past three rows.
        self.folder_list.setMinimumHeight(self._folder_row_height() * 3 + 10)
        self.folder_list.setFixedHeight(self._folder_row_height() * rows + 10)

    def _folder_row_height(self) -> int:
        """Height of one folder row, from a real item when there is one.

        The old estimate (icon + 14) was 74 px against an actual 44 px row, so
        a three-row list reserved nearly two rows of blank space.
        """
        if self.folder_list.count():
            item = self.folder_list.item(0)
            if item is not None:
                hint = item.sizeHint()
                if hint.height() > 0:
                    return hint.height()
        # Fall back to the widget's own row metric before the first item.
        if self.folder_list.count():
            vh = self.folder_list.visualItemRect(self.folder_list.item(0)).height()
            if vh > 0:
                return vh
        return self.folder_list.iconSize().height() + 14

    def _update_folder_placeholder(self) -> None:
        """Show the empty-state hint only while the list has no rows."""
        placeholder = getattr(self, "_folder_list_placeholder", None)
        if placeholder is None:
            return
        empty = self.folder_list.count() == 0
        placeholder.setVisible(empty)
        self.folder_list.setVisible(not empty)
        self._fit_folder_list()

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
        """Pick one or more stack folders.

        The plain QFileDialog had the flaws of every unstyled directory picker:
        nowhere to type a path, and no shortcuts to where you actually work.
        This one adds an editable path field (Enter jumps, like Explorer) and a
        sidebar of recent and standard locations.
        """
        from PyQt6.QtWidgets import (QFileDialog, QListView, QTreeView,
                                     QAbstractItemView, QSplitter, QListWidget,
                                     QLineEdit, QPushButton)

        dialog = QFileDialog(self, trans.t('batch_pick_title'))
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)

        for view_class in (QListView, QTreeView):
            view = dialog.findChild(view_class)
            if view:
                view.setSelectionMode(
                    QAbstractItemView.SelectionMode.ExtendedSelection)

        start_dir = self._last_pick_dir()
        if start_dir:
            dialog.setDirectory(start_dir)

        # Qt already builds a place list on the left (home / Pictures /
        # Desktop / Documents) as a QStandardItemModel. Appending to it is the
        # supported way to add "last used"; inserting a second QListWidget into
        # the splitter instead produced an empty column beside the real one.
        self._make_picker_path_editable(dialog)

        # 执行对话框并获取结果
        if dialog.exec():
            selected_dirs = dialog.selectedFiles()
            selected_dirs = [d for d in selected_dirs if os.path.isdir(d)]
            
            # 处理选中的文件夹
            if selected_dirs:
                for folder_path in selected_dirs:
                    if folder_path not in self.folder_paths:
                        # 添加到路径列表
                        self.folder_paths.append(folder_path)
                        
                        # 获取文件夹中的第一张图像作为缩略图
                        from core.image_loader import ImageStackLoader
                        loader = ImageStackLoader()
                        success, _, images, _ = loader.load_from_folder(folder_path)
                        
                        if success and images:
                            # 创建缩略图
                            thumbnail = loader.create_thumbnails([images[0]], thumb_size=60)[0]
                            self.folder_thumbnails.append(thumbnail)
                        else:
                            # 如果没有图像，使用空图标
                            self.folder_thumbnails.append(None)
                        
                        # 添加到列表显示
                        folder_name = os.path.basename(folder_path)
                        item_text = f"{folder_name}\n{folder_path}"
                        
                        item = QListWidgetItem(item_text)
                        if self.folder_thumbnails[-1]:
                            item.setIcon(QIcon(self.folder_thumbnails[-1]))
                        
                        self.folder_list.addItem(item)
                        self._update_folder_placeholder()

                # Remember this pick so the next "Add" opens nearby. Stored
                # under a batch-only key: the shared ui/last_dialog_dir belongs
                # to single-file dialogs, and a batch folder there would make
                # the next "Save As" start somewhere unrelated.
                from utils.settings_store import get_settings
                get_settings().setValue(BATCH_PICK_DIR_KEY,
                                        os.path.dirname(selected_dirs[-1]))

    def _make_picker_path_editable(self, dialog) -> None:
        """Let the user type a path into the picker itself.

        Qt's "Look in" combo is a plain dropdown (editable=False), so pasting
        a folder there does nothing. Making it editable turns it into the
        address bar Explorer has: type or paste, press Enter, navigate.
        """
        from PyQt6.QtWidgets import QComboBox

        for combo in dialog.findChildren(QComboBox):
            if combo.objectName() != "lookInCombo":
                continue
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            line = combo.lineEdit()
            if line is None:
                continue
            line.setPlaceholderText(trans.t('batch_pick_path_hint'))
            line.setClearButtonEnabled(True)

            def _navigate(_combo=combo):
                raw = _combo.currentText().strip().strip('"').strip("'")
                target = os.path.expandvars(os.path.expanduser(raw))
                if target and os.path.isdir(target):
                    dialog.setDirectory(target)
                else:
                    _combo.setEditText(dialog.directory().absolutePath())

            line.returnPressed.connect(_navigate)
            combo.activated.connect(lambda _i, _c=combo: _navigate(_c))
            break

        # Narrow the place list to what it actually shows. Qt sizes it to a
        # fixed share of the dialog, which left a ~90 px empty gutter between
        # the sidebar and the file view.
        self._fit_picker_sidebar(dialog)

    def _fit_picker_sidebar(self, dialog) -> None:
        """Give the place list just enough width for its longest label."""
        from PyQt6.QtWidgets import QSplitter, QListView

        splitter = dialog.findChild(QSplitter)
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

    def add_folder_to_list(self, folder_path: str) -> None:
        """添加文件夹到批处理列表（用于拖入场景）"""
        if folder_path in self.folder_paths:
            return

        self.folder_paths.append(folder_path)

        from core.image_loader import ImageStackLoader
        loader = ImageStackLoader()
        success, _, images, _ = loader.load_from_folder(folder_path)

        if success and images:
            thumbnail = loader.create_thumbnails([images[0]], thumb_size=60)[0]
            self.folder_thumbnails.append(thumbnail)
        else:
            self.folder_thumbnails.append(None)

        folder_name = os.path.basename(folder_path)
        item_text = f"{folder_name}\n{folder_path}"

        item = QListWidgetItem(item_text)
        if self.folder_thumbnails[-1]:
            item.setIcon(QIcon(self.folder_thumbnails[-1]))

        self.folder_list.addItem(item)
        self._update_folder_placeholder()

    def add_single_folder_to_list(self, folder_list):
        """添加单个文件夹到列表"""
        from PyQt6.QtWidgets import QFileDialog
        
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Image Stack Folder", ""
        )
        
        if folder_path and folder_path not in [folder_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(folder_list.count())]:
            folder_name = os.path.basename(folder_path)
            item = QListWidgetItem(f"{folder_name}\n{folder_path}")
            item.setData(Qt.ItemDataRole.UserRole, folder_path)
            folder_list.addItem(item)
    
    def add_multiple_folders_to_list(self, folder_list):
        """添加多个文件夹到列表"""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        
        # 使用循环添加多个文件夹
        first_time = True
        while True:
            if first_time:
                folder_path = QFileDialog.getExistingDirectory(
                    self, "Select Image Stack Folders (click Cancel when done)", ""
                )
                first_time = False
            else:
                folder_path = QFileDialog.getExistingDirectory(
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
            self.folder_paths.pop(row)
            if row < len(self.folder_thumbnails):
                self.folder_thumbnails.pop(row)
        self._update_folder_placeholder()

    def on_output_option_changed(self):
        """输出选项改变时的处理"""
        subfolder_enabled = self.rb_subfolder.isChecked()
        same_enabled = self.rb_same_folder.isChecked()
        custom_enabled = self.rb_custom_folder.isChecked()
        
        self.subfolder_name.setEnabled(subfolder_enabled)
        self.custom_folder_path.setEnabled(custom_enabled)
        self.browse_btn.setEnabled(custom_enabled)
    
    def on_import_mode_changed(self):
        """导入模式改变时的处理"""
        is_multiple = self.rb_multiple_folders.isChecked()
        
        self.folder_group.setVisible(is_multiple)
        self.single_folder_group.setVisible(not is_multiple)
        
        # Do NOT pin the dialog to 800 px: on a shorter screen the fixed
        # height put the button bar off-screen. The scroll area above the
        # bar already adapts to the available space.
        self.setMinimumHeight(min(640, self.height()))
        
        # 初始化质量滑块显示状态
        self.on_format_changed(self.format_combo.currentText())
    
    def on_split_method_changed(self, index):
        """分割方式改变时的处理"""
        if index == 0:
            self.param_label.setText(trans.t('batch_images_per_stack'))
            self.param_spinbox.setRange(2, 1000)
            self.param_spinbox.setValue(5)
            self.param_unit_label.setText(trans.t('batch_unit_images'))
        else:
            self.param_label.setText(trans.t('batch_time_threshold'))
            self.param_spinbox.setRange(1, 3600)
            self.param_spinbox.setValue(5)
            self.param_unit_label.setText(trans.t('batch_unit_seconds'))
        
        self.update_single_folder_preview()
    
    def on_format_changed(self, format_text):
        """格式改变时的处理 - 控制质量滑块显示"""
        is_jpg = format_text.upper() == "JPG"
        self.quality_label.setVisible(is_jpg)
        self.quality_slider.setVisible(is_jpg)
        self.quality_value_label.setVisible(is_jpg)
    
    def update_single_folder_preview(self):
        """更新单文件夹预览"""
        if not self.single_folder_images_with_times:
            self.preview_label.setText(trans.t('batch_preview_no_folder'))
            return
        
        split_method = self.split_method_combo.currentIndex()
        param_value = (int(self.param_spinbox.value()) if split_method == 0
                       else float(self.param_spinbox.value()))
        
        from core.image_loader import ImageStackLoader
        loader = ImageStackLoader()
        from utils.sorting import natural_sort_key
        self.single_folder_images_with_times.sort(key=lambda item:
            natural_sort_key(item[3] if len(item) > 3 else item[0]) if split_method == 0 else
            (item[2] is None, item[2] or 0, natural_sort_key(item[3] if len(item) > 3 else item[0])))
        
        if split_method == 0:
            stacks = loader.split_by_count(self.single_folder_images_with_times, param_value)
            self.preview_label.setText(trans.t('batch_preview_fmt_count').format(len(self.single_folder_images_with_times), len(stacks), param_value))
        else:
            stacks = loader.split_by_time_threshold(self.single_folder_images_with_times, param_value)
            self.preview_label.setText(trans.t('batch_preview_fmt_time').format(len(self.single_folder_images_with_times), len(stacks), param_value))
        self.single_folder_stacks = stacks
    
    def add_single_folder(self):
        """添加单个文件夹（自动分割）"""
        from PyQt6.QtWidgets import QFileDialog
        from core.image_loader import ImageStackLoader
        
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder with Multiple Image Stacks", ""
        )
        
        if not folder_path:
            return
        
        loader = ImageStackLoader()
        success, message, images_with_times, filenames = loader.load_images_with_timestamps(
            folder_path, scale_factor=self.load_scale_factor)
        
        if not success or not images_with_times or loader.failed_count:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                trans.t("msg_load_failed"),
                trans.t("msg_load_images_failed_text").format(message=message),
            )
            return
        
        self.single_folder_images_with_times = images_with_times
        self.single_folder_folder_path = folder_path

        # 更新路径显示
        folder_name = os.path.basename(folder_path)
        self.single_folder_path_label.setText(f"Folder: {folder_name}")
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
        
        split_method = self.split_method_combo.currentIndex()
        param_value = self.param_spinbox.value()
        
        if split_method == 0:
            return 'count', int(param_value)
        return 'time_threshold', float(param_value)
    
    def browse_output_folder(self):
        """浏览输出文件夹"""
        from PyQt6.QtWidgets import QFileDialog
        
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ""
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
    
    def _fill_method_combo(self, combo: QComboBox, items) -> None:
        """Populate a method dropdown from the shared right-panel tables.

        Same items, same order and same hint tooltips as the main window, so a
        method cannot mean something different in batch.
        """
        combo.clear()
        for code, key, hint_key in items:
            combo.addItem(trans.t(key), code)
            if hint_key:
                combo.setItemData(combo.count() - 1, trans.t(hint_key),
                                  Qt.ItemDataRole.ToolTipRole)

    def _seed_from_parent(self) -> None:
        """Start from the main window's current settings (it is a batch run by
        default), but let the user change every one of them here."""
        window = self.parent_window
        if window is None:
            self.kernel_spin.setValue(11)
            self.smooth_spin.setValue(2.0)
            return
        try:
            code = window.current_fusion_method()
        except Exception:
            code = "none"
        idx = self.fusion_combo.findData(code)
        self.fusion_combo.setCurrentIndex(max(0, idx))
        try:
            align = window.current_alignment_method()
        except Exception:
            align = "none"
        aidx = self.align_combo.findData(align)
        self.align_combo.setCurrentIndex(max(0, aidx))
        slider = getattr(window, "slider_smooth", None)
        if slider is not None:
            self.kernel_spin.setValue(float(slider.value()))
        smoothing_for = getattr(window, "smoothing_for_method", None)
        if callable(smoothing_for):
            try:
                self.smooth_spin.setValue(float(smoothing_for(code)))
            except Exception:
                pass
        self.chk_reverse_order.setChecked(
            bool(getattr(window, "stack_reverse_order", False)))
        getter = getattr(window, "brightness_option_for", None)
        if callable(getter):
            try:
                self.chk_exposure_match.setChecked(
                    bool(getter("exposure_match", code)))
                self.chk_base_correct.setChecked(
                    bool(getter("base_correct", code)))
            except Exception:
                pass

    def _remember_save_aligned(self, checked: bool) -> None:
        """Track the user's own choice, ignoring programmatic forcing."""
        if self.fusion_combo.currentData() != "none":
            self._save_aligned_pref = bool(checked)

    def _sync_method_controls(self) -> None:
        """Show only the parameters the chosen method actually uses."""
        method = self.fusion_combo.currentData()
        uses_kernel = method in ("guided_filter", "dct", "gfgfgf",
                                 "openfocus_a", "openfocus_b", "openfocus_c")
        uses_smoothing = method in ("openfocus_a", "openfocus_b",
                                    "openfocus_c")
        # Hide the rows outright rather than grey them out: a disabled stepper
        # still shows a value ("Kernel: 31") for a parameter the chosen method
        # never reads, which invites the user to try to change it.
        self.kernel_label.setVisible(uses_kernel)
        self.kernel_spin.setVisible(uses_kernel)
        self.kernel_spin.setEnabled(uses_kernel)
        self.smooth_label.setVisible(uses_smoothing)
        self.smooth_spin.setVisible(uses_smoothing)
        self.smooth_spin.setEnabled(uses_smoothing)
        # The "Parameters" heading only earns its line when a method-specific
        # parameter is on screen; with just the downscale slider it was a
        # section title above a single row.
        rule = getattr(self, "params_rule", None)
        if rule is not None:
            rule.setVisible(uses_kernel or uses_smoothing)
        if uses_smoothing:
            is_b = method == "openfocus_b"
            key = ("label_depth_smoothing" if is_b
                   else "label_blend_smoothing")
            self.smooth_label.setText(trans.t(key))
        # Align-only output IS the aligned frames, so the option is forced on.
        # Reflecting that in the widget keeps the dialog honest: unticking it
        # used to change nothing in the result.
        # `currentData()` is None before the combo is populated, so compare
        # against the literal only when there really is a method.
        align_only = method is not None and method == "none"
        self.save_aligned_cb.setEnabled(not align_only)
        self.save_aligned_cb.setChecked(align_only or
                                        self._save_aligned_pref)
        if align_only:
            self.save_aligned_cb.setToolTip(
                trans.t('batch_save_aligned_forced'))
        else:
            self.save_aligned_cb.setToolTip(
                trans.t('batch_save_aligned_hint'))

    def get_processing_settings(self):
        """获取处理设置"""
        format_str = self.format_combo.currentText().lower()
        
        fusion_method = self.fusion_combo.currentData() or "none"
        fusion_params = {}

        kernel_value = max(1, int(self.kernel_spin.value()))
        if kernel_value % 2 == 0:
            kernel_value = max(1, kernel_value - 1)
        if fusion_method in ("guided_filter", "dct", "gfgfgf",
                             "openfocus_a", "openfocus_b", "openfocus_c"):
            fusion_params["kernel_size"] = kernel_value
        fusion_params["base_frame"] = ("last" if self.chk_base_correct.isChecked()
                                       else "none")
        fusion_params["exposure_match"] = self.chk_exposure_match.isChecked()
        if fusion_method == "openfocus_b":
            fusion_params["depth_smoothing"] = float(self.smooth_spin.value())
        elif fusion_method in ("openfocus_a", "openfocus_c"):
            fusion_params["blend_smoothing"] = float(self.smooth_spin.value())
        fusion_params["sort_mode"] = ("desc" if self.chk_reverse_order.isChecked()
                                      else "asc")

        align_method = self.align_combo.currentData() or "none"
        reg_methods = [] if align_method == "none" else [align_method]
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

        return {
            "format": format_str,
            "jpg_quality": jpg_quality,
            "fusion_method": fusion_method,
            "fusion_params": fusion_params,
            "reg_methods": reg_methods,
            "reg_scale_percent": ratio,
            "save_aligned": save_aligned,  # 是否保存对齐后的图像栈
            "scale_bar": scale_cfg,
            'load_scale_factor': self.load_scale_factor,
            'runtime_settings': dict(self._runtime_settings),
            'frozen_stacks': [list(stack) for stack in self.single_folder_stacks],
        }

    def preload_single_folder(self, folder_path: str, scale_factor: float = 1.0) -> None:
        """
        预加载单个文件夹（用于拖入场景）
        自动切换到单文件夹模式、加载文件夹、设置默认划分参数
        """
        import cv2
        from core.image_loader import ImageStackLoader

        # 1. 切换到单文件夹模式
        self.rb_single_folder.setChecked(True)
        self.on_import_mode_changed()

        # 2. 设置默认划分方式：Fixed Count, 5张/组
        self.split_method_combo.setCurrentIndex(0)  # Fixed Count
        self.param_spinbox.setValue(5)
        self.param_unit_label.setText(trans.t('batch_unit_images'))
        self.on_split_method_changed(0)

        # 3. 加载文件夹并应用缩放
        loader = ImageStackLoader()
        self.load_scale_factor = scale_factor
        success, message, images_with_times, filenames = loader.load_images_with_timestamps(
            folder_path, scale_factor=scale_factor)

        if not success or not images_with_times or loader.failed_count:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                trans.t("msg_load_failed"),
                trans.t("msg_load_images_failed_text").format(message=message),
            )
            return

        self.single_folder_images_with_times = images_with_times

        self.single_folder_folder_path = folder_path

        # 更新路径显示
        folder_name = os.path.basename(folder_path)
        self.single_folder_path_label.setText(f"Folder: {folder_name}")
        self.single_folder_path_label.setToolTip(folder_path)

        # 4. 更新preview
        self.update_single_folder_preview()

    def preload_multiple_folders(self, folder_paths: list[str], scale_factor: float = 1.0) -> None:
        """预加载多个文件夹（用于拖入场景）

        Args:
            folder_paths: 文件夹路径列表
            scale_factor: 缩放因子（统一应用到所有文件夹）
        """
        import cv2
        from core.image_loader import ImageStackLoader

        self.rb_multiple_folders.setChecked(True)
        self.on_import_mode_changed()
        self.load_scale_factor = scale_factor

        loader = ImageStackLoader()
        for folder_path in folder_paths:
            if folder_path in self.folder_paths:
                continue

            self.folder_paths.append(folder_path)

            success, message, full_res_images, filenames = loader.load_from_folder(
                folder_path, scale_factor=scale_factor
            )

            if success and full_res_images:
                thumbnail = loader.create_thumbnails([full_res_images[0]], thumb_size=60)[0]
                self.folder_thumbnails.append(thumbnail)
            else:
                self.folder_thumbnails.append(None)

            folder_name = os.path.basename(folder_path)
            item_text = f"{folder_name}\n{folder_path}"

            item = QListWidgetItem(item_text)
            if self.folder_thumbnails[-1]:
                item.setIcon(QIcon(self.folder_thumbnails[-1]))

            self.folder_list.addItem(item)
        self._update_folder_placeholder()

    def start_batch_processing(self):
        """开始批处理"""
        from PyQt6.QtWidgets import QMessageBox

        import_mode = self.get_import_mode()
        split_method, split_param = self.get_split_settings()

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

        processing_settings = self.get_processing_settings()
        if import_mode == 'single_folder':
            self.update_single_folder_preview()
            if split_method == 'time_threshold':
                from core.image_loader import ImageStackLoader
                missing = ImageStackLoader.missing_timestamps(self.single_folder_images_with_times)
                if missing:
                    QMessageBox.warning(self, trans.t('msg_error'),
                                        trans.t('batch_missing_times').format(n=missing))
                    return
            if processing_settings['fusion_method'] != 'none' and any(
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

        for folder in self._source_output_folders():
            if not os.path.isdir(folder):
                return f"{trans.t('msg_save_failed_info_write')}\n{folder}"
            target = os.path.realpath(os.path.join(folder, output_path)) if output_type == 'subfolder' else folder
            root = os.path.realpath(folder)
            if output_type == 'subfolder' and (os.path.isabs(output_path) or
                    os.path.commonpath((root, target)) != root):
                return f"{trans.t('msg_save_failed_info_write')}\n{target}"
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
        self.resize(500, 200)


        layout = QVBoxLayout(self)

        folder_name = os.path.basename(folder_path)
        folder_display = QLabel(trans.t('import_folder_display').format(folder_name))
        
        layout.addWidget(folder_display)

        path_label = QLabel(folder_path)
        path_label.setObjectName("mutedLabel")
        layout.addWidget(path_label)

        layout.addSpacing(20)

        option_label = QLabel(trans.t('import_choice_label'))
        
        layout.addWidget(option_label)

        self.rb_single = QRadioButton(trans.t('import_option_single'))
        self.rb_single.setChecked(True)
        layout.addWidget(self.rb_single)

        self.rb_batch = QRadioButton(trans.t('import_option_batch'))
        layout.addWidget(self.rb_batch)

        layout.addStretch()

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
