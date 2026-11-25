"""Application preferences, grouped by task and applied together on OK."""
from PyQt6.QtCore import Qt
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QTabWidget, QToolButton,
    QVBoxLayout, QWidget,
)

from constants import (
    TILE_BLOCK_MAX, TILE_BLOCK_MIN, TILE_OVERLAP_MAX, TILE_THRESHOLD_MAX,
    TILE_THRESHOLD_MIN,
)
from core.algorithms import (FUSION_DEFAULT, get_fusion, get_registration, registration_parameter_specs,
                             resolve_registration_parameters)
from core.experimental import alignment_is_experimental
from core.multi_focus_fusion import _DEFAULT_TILE_THRESHOLD
from core.frame_storage import configure_storage_directory, restore_storage_directory
from locales import trans
from utils.export_paths import prefix_output_name, valid_output_prefix
from ui.algorithm_parameters import AlgorithmParameterPanel
from ui.parameter_help import add_help_row
from widgets.stepper import StepperSpinBox

# Registration downscale bounds mirror dialogs/settings.py, the other surface
# that edits the same persisted value.
REG_WIDTH_MIN = 64
REG_WIDTH_MAX = 8192


def _section(parent_layout, title=None) -> QFormLayout:
    if title:
        head = QLabel(title)
        font = head.font()
        font.setBold(True)
        head.setFont(font)
        parent_layout.addWidget(head)
    form = QFormLayout()
    form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    form.setHorizontalSpacing(16)
    form.setVerticalSpacing(6)
    parent_layout.addLayout(form)
    return form


def _note(parent_layout, text: str) -> QLabel:
    note = QLabel(text)
    note.setWordWrap(True)
    note.setObjectName("mutedLabel")
    parent_layout.addWidget(note)
    return note


def _disclosure(layout, title, control_id, content):
    button = QToolButton()
    button.setText(title)
    button.setProperty('controlId', control_id)
    button.setCheckable(True)
    button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

    def expand(opened):
        button.setArrowType(Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow)
        content.setVisible(opened)

    layout.addWidget(button, 0, Qt.AlignmentFlag.AlignLeft)
    layout.addWidget(content)
    button.toggled.connect(expand)
    expand(False)
    return button


class AdvancedSettingsDialog(QDialog):
    """Edit a single draft across independently scrollable preference pages."""

    def __init__(self, window, parent=None):
        super().__init__(parent or window)
        self.window_main = window
        self.setWindowTitle(trans.t("advanced_settings_title"))
        self.resize(min(700, self.screen().availableGeometry().width()), 400)

        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setProperty('controlId', 'advanced_settings_tabs')
        root.addWidget(self.tabs, 1)
        for key, build in (("advanced_tab_general", self._build_general),
                           ("group_fusion", self._build_fusion),
                           ("advanced_tab_registration", self._build_registration),
                           ("advanced_tab_other", self._build_startup),
                           ("gallery_settings_tab", self._build_gallery)):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(14, 10, 14, 10)
            layout.setSpacing(10)
            build(layout)
            layout.addStretch()
            scroll.setWidget(page)
            self.tabs.addTab(scroll, trans.t(key))
        self._initial_fields = self._field_values()

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self.status)
        self.status.hide()
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText(trans.t("btn_ok"))
        ok_btn.setObjectName("primaryButton")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(trans.t("btn_cancel"))
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        ok_btn.setDefault(True)
        self.tabs.currentChanged.connect(self._fit_page)
        self.manual_tile_disclosure.toggled.connect(self._fit_page)
        self.sensor_cleanup_disclosure.toggled.connect(self._fit_page)
        self.registration_options_disclosure.toggled.connect(self._fit_page)
        self.chk_ask_scale.toggled.connect(self._fit_page)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_page()

    def _fit_page(self):
        if not self.isVisible():
            return
        self.layout().activate()
        scroll = self.tabs.currentWidget()
        page = scroll.widget().layout()
        page.activate()
        viewport = scroll.viewport()
        content = page.totalHeightForWidth(viewport.width()) if page.hasHeightForWidth() else page.sizeHint().height()
        available = self.screen().availableGeometry()
        frame_height = self.frameGeometry().height() - self.height()
        height = self.height() - viewport.height() + content
        self.resize(self.width(), min(height, available.height() - frame_height))
        overflow = self.frameGeometry().bottom() - available.bottom()
        if overflow > 0:
            self.move(self.x(), max(available.top(), self.y() - overflow))

    def _build_general(self, page_lay):
        w = self.window_main
        self.chk_experimental_features = QCheckBox(trans.t("experimental_features"))
        self.chk_experimental_features.setProperty("controlId", "chk_experimental_features")
        self.chk_experimental_features.setChecked(bool(getattr(w, "experimental_features", False)))
        self.chk_experimental_features.setToolTip(trans.t("experimental_features_hint"))
        page_lay.addWidget(self.chk_experimental_features)
        _note(page_lay, trans.t("experimental_features_hint"))

        privacy = _section(page_lay, trans.t('advanced_privacy'))
        self.chk_auto_error_reports = QCheckBox(trans.t('report_auto_setting'))
        self.chk_auto_error_reports.setProperty('controlId', 'chk_auto_error_reports')
        self.chk_auto_error_reports.setChecked(bool(getattr(w, 'auto_error_reports', True)))
        self._reports_choice_touched = False
        self.chk_auto_error_reports.toggled.connect(
            lambda _checked: setattr(self, '_reports_choice_touched', True))
        self.chk_auto_error_reports.setToolTip(trans.t('report_auto_hint'))
        privacy.addRow(self.chk_auto_error_reports)
        self.chk_usage = QCheckBox(trans.t('usage_enabled'))
        self.chk_usage.setProperty('controlId', 'chk_usage')
        analytics = getattr(w, 'usage_analytics', None)
        self.chk_usage.setChecked(analytics.enabled() if analytics is not None else True)
        self.chk_usage.setToolTip(trans.t('usage_disclosure'))
        privacy.addRow(self.chk_usage)

        storage = _section(page_lay, trans.t('advanced_working_storage'))
        storage_row = QWidget()
        storage_layout = QHBoxLayout(storage_row)
        storage_layout.setContentsMargins(0, 0, 0, 0)
        self.storage_directory = QLineEdit(getattr(w, "storage_directory", ""))
        self.storage_directory.setPlaceholderText(trans.t("advanced_storage_default"))
        storage_layout.addWidget(self.storage_directory, 1)
        browse = QPushButton(trans.t("advanced_storage_browse"))
        browse.clicked.connect(self._browse_storage_directory)
        storage_layout.addWidget(browse)
        storage.addRow(trans.t("advanced_storage_directory"), storage_row)

        output = _section(page_lay, trans.t("output_prefix_section"))
        self.output_prefix = QLineEdit(getattr(w, "output_name_prefix", "") or "")
        self.output_prefix.setMaxLength(40)
        self.output_prefix.setToolTip(trans.t("output_prefix_hint"))
        output.addRow(trans.t("output_prefix_label"), self.output_prefix)
        self.output_prefix_preview = QLabel()
        self.output_prefix_preview.setWordWrap(True)
        output.addRow(trans.t("output_prefix_preview"), self.output_prefix_preview)
        self.output_prefix.textChanged.connect(self._update_output_prefix_preview)
        self._update_output_prefix_preview()

    def _build_fusion(self, page_lay):
        w = self.window_main
        perf = _section(page_lay)
        self.combo_resources = QComboBox()
        for mode in ("balanced", "responsive", "maximum"):
            self.combo_resources.addItem(
                trans.t(f"advanced_resource_{mode}"), mode)
        current = getattr(w, "resource_preference", "balanced")
        index = self.combo_resources.findData(current)
        self.combo_resources.setCurrentIndex(max(0, index))
        self.combo_resources.setToolTip(
            trans.t("advanced_resource_hint"))
        perf.addRow(trans.t("advanced_resource_mode"), self.combo_resources)
        self.tile_status = _note(page_lay, '')
        self.tile_status.setProperty('controlId', 'effective_tiling_status')
        self.manual_tile_panel = QWidget()
        tile_layout = QVBoxLayout(self.manual_tile_panel)
        tile_layout.setContentsMargins(0, 0, 0, 0)
        self.manual_tile_disclosure = _disclosure(
            page_lay, trans.t('advanced_tile_manual'), 'manual_tile_disclosure', self.manual_tile_panel)
        self.manual_tile_scope_note = _note(tile_layout, trans.t('advanced_tile_experimental_only_note'))
        tiles = _section(tile_layout)
        self.chk_tile = QCheckBox(trans.t('advanced_tile_enabled'))
        self.chk_tile.setChecked(bool(getattr(w, 'tile_enabled', True)))
        tiles.addRow(self.chk_tile)
        self.spin_block = StepperSpinBox(
            minimum=TILE_BLOCK_MIN, maximum=TILE_BLOCK_MAX, step=128,
            decimals=0, value=int(getattr(w, 'tile_block_size', 1024)))
        tiles.addRow(trans.t('advanced_tile_block'), self.spin_block)
        self.spin_overlap = StepperSpinBox(
            minimum=0, maximum=TILE_OVERLAP_MAX, step=32, decimals=0,
            value=int(getattr(w, 'tile_overlap', 256)))
        add_help_row(tiles, trans.t('advanced_tile_overlap'), self.spin_overlap,
                     'tiles', 'parameter_help_tiles')
        self.spin_block.valueChanged.connect(self._sync_overlap_limit)
        self._sync_overlap_limit(self.spin_block.value())
        self.spin_threshold = StepperSpinBox(
            minimum=TILE_THRESHOLD_MIN, maximum=TILE_THRESHOLD_MAX, step=256,
            decimals=0, value=int(getattr(w, 'tile_threshold', _DEFAULT_TILE_THRESHOLD)))
        tiles.addRow(trans.t('advanced_tile_threshold'), self.spin_threshold)
        # Display effective bounds, but never write a clamped legacy value back
        # when the user only visits this dialog or changes another preference.
        self._tile_fields = {
            'tile_block_size': (self.spin_block, self.spin_block.value()),
            'tile_overlap': (self.spin_overlap, self.spin_overlap.value()),
            'tile_threshold': (self.spin_threshold, self.spin_threshold.value()),
        }
        method = getattr(w, 'current_fusion_method', lambda: FUSION_DEFAULT)()
        spec = None if method == 'none' else get_fusion(method)
        supported = (spec is not None and spec.external_tiling_supported and
                     method not in ('openfocus_a', 'openfocus_b', 'openfocus_c'))

        self.manual_tile_scope_note.setVisible(not supported)

        def update_tile_controls(*_args):
            enabled = self.chk_tile.isChecked()
            threshold = int(self.spin_threshold.value())
            spatial = (enabled and supported and
                       (spec.external_tiling_default or threshold != _DEFAULT_TILE_THRESHOLD))
            self.spin_threshold.setEnabled(enabled and method != 'dtcwt')
            self.spin_block.setEnabled(enabled)
            self.spin_overlap.setEnabled(enabled)
            if spec is None:
                key = 'advanced_tile_alignment_only'
            elif spatial and method == 'dtcwt':
                key = 'advanced_tile_memory_status'
            elif spatial and spec.neural:
                key = 'advanced_tile_neural_status'
            elif spatial:
                key = 'advanced_tile_threshold_status'
            else:
                key = 'advanced_tile_full_frame_status'
            self.tile_status.setText(trans.t(key).format(
                method=spec.display_name if spec else '', threshold=threshold,
                block=int(self.spin_block.value())))

        self.chk_tile.toggled.connect(update_tile_controls)
        self.spin_block.valueChanged.connect(update_tile_controls)
        self.spin_threshold.valueChanged.connect(update_tile_controls)
        update_tile_controls()

        from ui.sensor_cleanup import SensorCleanupControls
        self.sensor_cleanup = SensorCleanupControls(
            getattr(w, 'defect_mask_path', ''), getattr(w, 'sensor_denoise', False))
        self.sensor_cleanup_disclosure = _disclosure(
            page_lay, trans.t('sensor_section'), 'sensor_cleanup_disclosure', self.sensor_cleanup)
        self.finished.connect(lambda _result: self.sensor_cleanup.stop_inspection())

    def _build_registration(self, page_lay):
        w = self.window_main
        reg = _section(page_lay)
        self.spin_reg_width = StepperSpinBox(
            minimum=REG_WIDTH_MIN, maximum=REG_WIDTH_MAX, step=64, decimals=0,
            value=int(getattr(w, "reg_downscale_width", 1024)))
        self.spin_reg_width.setMinimumWidth(max(126, self.spin_reg_width.minimumSizeHint().width()))
        self.spin_reg_width.setMaximumWidth(180)
        self.spin_reg_width.setToolTip(trans.t("parameter_help_registration_width"))
        add_help_row(reg, trans.t("advanced_reg_width"), self.spin_reg_width,
                     "registration_width", "parameter_help_registration_width")
        getter = getattr(w, 'registration_parameters_for', None)
        memory = getattr(w, '_registration_parameters_by_method', {})
        self._initial_openfocus = resolve_registration_parameters(
            'openfocus', getter('openfocus') if callable(getter) else memory.get('openfocus'))
        self.openfocus_registration_panel = AlgorithmParameterPanel(
            hidden_parameters=("failure_policy",))
        self.openfocus_registration_panel.setProperty("controlId", "openfocus_registration_panel")
        self.openfocus_registration_panel.set_parameters(
            registration_parameter_specs('openfocus'), self._initial_openfocus)
        reg_title = trans.t(get_registration('openfocus').label_key)
        if alignment_is_experimental('openfocus'):
            reg_title = f"{reg_title} ({trans.t('experimental_label')})"
        self.registration_options_disclosure = _disclosure(
            page_lay, reg_title,
            'registration_options_disclosure', self.openfocus_registration_panel)
        self._initial_registration_fields = self.openfocus_registration_panel.values()
        self._sync_experimental_controls()
        self.chk_experimental_features.toggled.connect(self._sync_experimental_controls)

    def _sync_experimental_controls(self, *_args):
        enabled = self.chk_experimental_features.isChecked()
        self.registration_options_disclosure.setEnabled(enabled)
        self.openfocus_registration_panel.setEnabled(enabled)
        hint = '' if enabled else trans.t('experimental_locked_hint')
        self.registration_options_disclosure.setToolTip(hint)
        if not enabled and self.registration_options_disclosure.isChecked():
            self.registration_options_disclosure.setChecked(False)

    def _build_startup(self, page_lay):
        w = self.window_main
        other = _section(page_lay)
        self.chk_restore = QCheckBox(trans.t('action_restore_last_stack'))
        self.chk_restore.setChecked(bool(getattr(w, "restore_last_stack", False)))
        other.addRow(self.chk_restore)
        self.chk_ask_scale = QCheckBox(trans.t('action_ask_load_scale'))
        self.chk_ask_scale.setChecked(bool(getattr(w, "ask_load_scale", True)))
        self.chk_ask_scale.setToolTip(trans.t("action_ask_load_scale_hint"))
        other.addRow(self.chk_ask_scale)
        self.lbl_load_scale = QLabel(
            f"{float(getattr(w, 'load_scale_default', 1.0)) * 100:g}%")
        other.addRow(trans.t("ds_title"), self.lbl_load_scale)
        self.chk_ask_scale.toggled.connect(
            lambda checked: other.setRowVisible(self.lbl_load_scale, not checked))
        other.setRowVisible(self.lbl_load_scale, not self.chk_ask_scale.isChecked())
        manager = getattr(w, 'update_manager', None)
        self.chk_auto_update = QCheckBox(trans.t('action_auto_download'))
        self.chk_auto_update.setChecked(bool(getattr(manager, 'auto_download', True)))
        other.addRow(self.chk_auto_update)
        self.chk_auto_update.setToolTip(trans.t("action_auto_download_hint"))
        self.chk_include_beta = QCheckBox(trans.t('action_include_beta_with_hint'))
        self.chk_include_beta.setProperty("controlId", "chk_include_beta")
        self.chk_include_beta.setChecked(bool(getattr(manager, 'include_beta', False)))
        other.addRow(self.chk_include_beta)

    def _build_gallery(self, page_lay):
        from controllers.gallery_controller import load_danmaku_settings, read_cache_max_mb
        from core.gallery_cache import (
            GALLERY_CACHE_DIR_KEY,
            GALLERY_DOWNLOAD_DIR_KEY,
            default_gallery_cache_dir,
            default_gallery_download_dir,
        )
        from utils.settings_store import get_settings

        w = self.window_main
        gallery = getattr(w, "gallery_controller", None)
        store = get_settings()
        danmaku_cfg = (
            gallery.danmaku_settings()
            if gallery is not None
            else load_danmaku_settings(store)
        )

        # --- Danmaku Settings ---
        danmaku = _section(page_lay, trans.t("gallery_danmaku_settings"))
        self.chk_danmaku_enabled = QCheckBox(trans.t("gallery_danmaku_enable"))
        self.chk_danmaku_enabled.setProperty("controlId", "chk_danmaku_enabled")
        self.chk_danmaku_enabled.setChecked(bool(danmaku_cfg.enabled))
        danmaku.addRow(self.chk_danmaku_enabled)

        self.combo_danmaku_position = QComboBox()
        self.combo_danmaku_position.setProperty("controlId", "combo_danmaku_position")
        for pos_key in ("top", "middle", "bottom", "full"):
            self.combo_danmaku_position.addItem(trans.t(f"gallery_danmaku_pos_{pos_key}"), pos_key)
        pos_idx = self.combo_danmaku_position.findData(danmaku_cfg.position)
        self.combo_danmaku_position.setCurrentIndex(max(0, pos_idx))
        danmaku.addRow(trans.t("gallery_danmaku_position"), self.combo_danmaku_position)

        self.spin_danmaku_font_size = StepperSpinBox(
            minimum=12, maximum=32, step=1, decimals=0, value=int(danmaku_cfg.font_size)
        )
        self.spin_danmaku_font_size.setProperty("controlId", "spin_danmaku_font_size")
        danmaku.addRow(trans.t("gallery_danmaku_font_size"), self.spin_danmaku_font_size)

        self.spin_danmaku_opacity = StepperSpinBox(
            minimum=10, maximum=100, step=5, decimals=0,
            value=int(round(float(danmaku_cfg.opacity) * 100)),
        )
        self.spin_danmaku_opacity.setProperty("controlId", "spin_danmaku_opacity")
        danmaku.addRow(trans.t("gallery_danmaku_opacity"), self.spin_danmaku_opacity)

        self.combo_danmaku_density = QComboBox()
        self.combo_danmaku_density.setProperty("controlId", "combo_danmaku_density")
        for dens_key in ("low", "medium", "high"):
            self.combo_danmaku_density.addItem(trans.t(f"gallery_danmaku_density_{dens_key}"), dens_key)
        dens_idx = self.combo_danmaku_density.findData(danmaku_cfg.density)
        self.combo_danmaku_density.setCurrentIndex(max(0, dens_idx))
        danmaku.addRow(trans.t("gallery_danmaku_density"), self.combo_danmaku_density)

        # --- Gallery Storage & Cache ---
        storage = _section(page_lay, trans.t("gallery_settings_storage_section"))

        dl_override = str(store.value(GALLERY_DOWNLOAD_DIR_KEY, "") or "").strip()
        dl_row = QWidget()
        dl_lay = QHBoxLayout(dl_row)
        dl_lay.setContentsMargins(0, 0, 0, 0)
        self.gallery_download_dir = QLineEdit(dl_override)
        self.gallery_download_dir.setProperty("controlId", "edit_gallery_download_dir")
        self.gallery_download_dir.setPlaceholderText(str(default_gallery_download_dir()))
        dl_lay.addWidget(self.gallery_download_dir, 1)
        self.btn_browse_gallery_download_dir = QPushButton(trans.t("advanced_storage_browse"))
        self.btn_browse_gallery_download_dir.setProperty("controlId", "btn_browse_gallery_download_dir")
        self.btn_browse_gallery_download_dir.clicked.connect(self._browse_gallery_download_dir)
        dl_lay.addWidget(self.btn_browse_gallery_download_dir)
        storage.addRow(trans.t("gallery_settings_download_dir"), dl_row)

        cache_override = str(store.value(GALLERY_CACHE_DIR_KEY, "") or "").strip()
        cache_row = QWidget()
        cache_lay = QHBoxLayout(cache_row)
        cache_lay.setContentsMargins(0, 0, 0, 0)
        self.gallery_cache_dir = QLineEdit(cache_override)
        self.gallery_cache_dir.setProperty("controlId", "edit_gallery_cache_dir")
        self.gallery_cache_dir.setPlaceholderText(str(default_gallery_cache_dir()))
        cache_lay.addWidget(self.gallery_cache_dir, 1)
        self.btn_browse_gallery_cache_dir = QPushButton(trans.t("advanced_storage_browse"))
        self.btn_browse_gallery_cache_dir.setProperty("controlId", "btn_browse_gallery_cache_dir")
        self.btn_browse_gallery_cache_dir.clicked.connect(self._browse_gallery_cache_dir)
        cache_lay.addWidget(self.btn_browse_gallery_cache_dir)
        storage.addRow(trans.t("gallery_settings_cache_dir"), cache_row)

        initial_limit_mb = (
            gallery.cache_max_mb()
            if gallery is not None
            else read_cache_max_mb(store)
        )
        self.spin_gallery_cache_limit = StepperSpinBox(
            minimum=10, maximum=2048, step=10, decimals=0, value=int(initial_limit_mb)
        )
        self.spin_gallery_cache_limit.setProperty("controlId", "spin_gallery_cache_limit")
        self.spin_gallery_cache_limit.valueChanged.connect(self._refresh_gallery_cache_usage)
        storage.addRow(trans.t("gallery_settings_cache_limit"), self.spin_gallery_cache_limit)

        usage_row = QWidget()
        usage_lay = QHBoxLayout(usage_row)
        usage_lay.setContentsMargins(0, 0, 0, 0)
        self.lbl_gallery_cache_usage = QLabel()
        self.lbl_gallery_cache_usage.setProperty("controlId", "lbl_gallery_cache_usage")
        self.lbl_gallery_cache_usage.setObjectName("mutedLabel")
        usage_lay.addWidget(self.lbl_gallery_cache_usage, 1)
        self.btn_clear_gallery_cache = QPushButton(trans.t("gallery_settings_clear_cache"))
        self.btn_clear_gallery_cache.setProperty("controlId", "btn_clear_gallery_cache")
        self.btn_clear_gallery_cache.clicked.connect(self._clear_gallery_cache)
        usage_lay.addWidget(self.btn_clear_gallery_cache)
        storage.addRow("", usage_row)
        self._refresh_gallery_cache_usage()

    def _browse_gallery_download_dir(self):
        start_dir = self.gallery_download_dir.text().strip() or self.gallery_download_dir.placeholderText()
        path = FileDialog.getExistingDirectory(self, trans.t("gallery_settings_download_dir"), start_dir)
        if path:
            self.gallery_download_dir.setText(path)

    def _browse_gallery_cache_dir(self):
        start_dir = self.gallery_cache_dir.text().strip() or self.gallery_cache_dir.placeholderText()
        path = FileDialog.getExistingDirectory(self, trans.t("gallery_settings_cache_dir"), start_dir)
        if path:
            self.gallery_cache_dir.setText(path)

    def _refresh_gallery_cache_usage(self, *_args):
        gallery = getattr(self.window_main, "gallery_controller", None)
        if gallery is not None:
            used_bytes = gallery.cache_usage_bytes()
        else:
            from core.gallery_cache import GalleryDiskCache
            from utils.settings_store import get_settings
            try:
                used_bytes = GalleryDiskCache(settings=get_settings()).usage_bytes()
            except Exception:
                used_bytes = 0
        used_mb = f"{used_bytes / (1024 * 1024):.1f}"
        limit_mb = int(self.spin_gallery_cache_limit.value())
        self.lbl_gallery_cache_usage.setText(
            trans.t("gallery_settings_cache_usage").format(used_mb=used_mb, limit_mb=limit_mb)
        )

    def _clear_gallery_cache(self):
        gallery = getattr(self.window_main, "gallery_controller", None)
        if gallery is not None:
            removed_files, freed_bytes = gallery.clear_disk_cache()
        else:
            from core.gallery_cache import GalleryDiskCache
            from utils.settings_store import get_settings
            removed_files, freed_bytes = GalleryDiskCache(settings=get_settings()).clear()
        self._refresh_gallery_cache_usage()
        self.status.setText(
            trans.t("gallery_settings_cache_cleared").format(
                files=removed_files,
                size_mb=freed_bytes / (1024 * 1024),
            )
        )
        self.status.show()

    def _browse_storage_directory(self):
        path = FileDialog.getExistingDirectory(
            self, trans.t("advanced_storage_browse"), self.storage_directory.text())
        if path:
            self.storage_directory.setText(path)

    def _sync_overlap_limit(self, block: float) -> None:
        """Keep the dialog's overlap cap in step with the chosen tile size.

        The cap is what the fusion tiler itself applies (`block // 2`), but
        never above the registry-wide maximum, so a stored overlap is only
        reduced when the tiler would reduce it too.
        """
        self.spin_overlap.setMaximum(
            min(TILE_OVERLAP_MAX, max(0, int(block)) // 2))

    def _update_output_prefix_preview(self):
        prefix = self.output_prefix.text()
        if not valid_output_prefix(prefix):
            self.output_prefix_preview.setText(trans.t("output_prefix_hint"))
            return
        self.output_prefix_preview.setText(prefix_output_name(
            prefix, "OpenFocus_20261015_120000_Method_Align.png"))

    def _show_error(self, message, field):
        for panel, disclosure in ((self.manual_tile_panel, self.manual_tile_disclosure),
                                  (self.sensor_cleanup, self.sensor_cleanup_disclosure),
                                  (self.openfocus_registration_panel, self.registration_options_disclosure)):
            if panel is field or panel.isAncestorOf(field):
                disclosure.setChecked(True)
        for index in range(self.tabs.count()):
            page = self.tabs.widget(index)
            if page.isAncestorOf(field):
                self.tabs.setCurrentIndex(index)
                page.widget().layout().activate()
                page.ensureWidgetVisible(field)
                break
        self.status.setText(message)
        self.status.show()
        self._fit_page()
        field.setFocus()

    def _field_values(self):
        return {
            'experimental_features': self.chk_experimental_features.isChecked(),
            'auto_error_reports': self.chk_auto_error_reports.isChecked(),
            'tile_enabled': self.chk_tile.isChecked(),
            'resource_preference': self.combo_resources.currentData(),
            'reg_downscale_width': int(self.spin_reg_width.value()),
            'output_name_prefix': self.output_prefix.text(),
            'restore_last_stack': self.chk_restore.isChecked(),
            'ask_load_scale': self.chk_ask_scale.isChecked(),
            'defect_mask_path': self.sensor_cleanup.path.text().strip(),
            'sensor_denoise': self.sensor_cleanup.denoise.isChecked(),
            'storage_directory': self.storage_directory.text().strip(),
            'auto_download': self.chk_auto_update.isChecked(),
            'include_beta': self.chk_include_beta.isChecked(),
            'share_usage': self.chk_usage.isChecked(),
            'danmaku_enabled': self.chk_danmaku_enabled.isChecked(),
            'danmaku_position': self.combo_danmaku_position.currentData(),
            'danmaku_font_size': int(self.spin_danmaku_font_size.value()),
            'danmaku_opacity_pct': int(self.spin_danmaku_opacity.value()),
            'danmaku_density': self.combo_danmaku_density.currentData(),
            'gallery_download_dir': self.gallery_download_dir.text().strip(),
            'gallery_cache_dir': self.gallery_cache_dir.text().strip(),
            'gallery_cache_max_mb': int(self.spin_gallery_cache_limit.value()),
        }

    def _apply(self):
        w = self.window_main
        spins = [editor for editor, _ in self._tile_fields.values() if editor.isEnabled()]
        spins.extend([
            self.spin_reg_width,
            self.spin_danmaku_font_size,
            self.spin_danmaku_opacity,
            self.spin_gallery_cache_limit,
        ])
        if self.chk_experimental_features.isChecked():
            spins.extend(editor for editor in self.openfocus_registration_panel.editors.values()
                         if isinstance(editor, StepperSpinBox))
        for spin in spins:
            if not spin.lineEdit().hasAcceptableInput():
                self._show_error(trans.t('msg_error'), spin.lineEdit())
                spin.lineEdit().selectAll()
                return
            spin.interpretText()
        if not valid_output_prefix(self.output_prefix.text()):
            self._show_error(trans.t('output_prefix_hint'), self.output_prefix)
            return
        sensor_error = self.sensor_cleanup.validate(w)
        if sensor_error:
            self._show_error(sensor_error, self.sensor_cleanup.path)
            return
        draft = {key: value for key, value in self._field_values().items()
                 if value != self._initial_fields[key]}
        for key, (editor, initial) in self._tile_fields.items():
            if editor.isEnabled() and editor.value() != initial:
                draft[key] = int(editor.value())
        if self.chk_experimental_features.isChecked():
            try:
                openfocus_values = self.openfocus_registration_panel.values()
            except (TypeError, ValueError) as exc:
                self._show_error(str(exc), self.openfocus_registration_panel)
                return
            reg_edits = {key: value for key, value in openfocus_values.items()
                         if value != self._initial_registration_fields[key]}
            if reg_edits:
                draft['_registration_parameters_by_method'] = {
                    **getattr(w, '_registration_parameters_by_method', {}),
                    'openfocus': {**self._initial_openfocus, **reg_edits}}
        if not draft and not self._reports_choice_touched:
            self.accept()
            return
        auto_download = draft.pop('auto_download', None)
        include_beta = draft.pop('include_beta', None)
        share_usage = draft.pop('share_usage', None)
        danmaku_keys = (
            'danmaku_enabled', 'danmaku_position', 'danmaku_font_size',
            'danmaku_opacity_pct', 'danmaku_density',
        )
        danmaku_changed = any(k in draft for k in danmaku_keys)
        for k in danmaku_keys:
            draft.pop(k, None)
        gallery_download_dir = draft.pop('gallery_download_dir', None)
        gallery_cache_dir = draft.pop('gallery_cache_dir', None)
        gallery_cache_max_mb = draft.pop('gallery_cache_max_mb', None)
        gallery_storage_changed = (
            gallery_download_dir is not None
            or gallery_cache_dir is not None
            or gallery_cache_max_mb is not None
        )
        missing = object()
        previous = {key: getattr(w, key, missing) for key in draft}
        manager = getattr(w, "update_manager", None)
        previous_auto = manager.auto_download if manager is not None else None
        previous_beta = getattr(manager, "include_beta", False)
        previous_storage = getattr(w, "storage_directory", "") or ""
        if 'storage_directory' in draft:
            try:
                configure_storage_directory(draft["storage_directory"])
            except OSError as exc:
                self._show_error(trans.t('advanced_storage_invalid').format(error=str(exc)),
                                 self.storage_directory)
                return
        from utils.settings_store import get_settings, REPORTS_CHOICE_KEY
        settings = get_settings() if self._reports_choice_touched else None
        previous_choice = (settings.value(REPORTS_CHOICE_KEY)
                           if settings is not None and settings.contains(REPORTS_CHOICE_KEY)
                           else None)
        enabling_reports = False
        if settings is not None and self.chk_auto_error_reports.isChecked():
            from utils.settings_store import automatic_reports_allowed
            enabling_reports = (not self._initial_fields['auto_error_reports'] or
                                not automatic_reports_allowed(settings))
        if enabling_reports:
            try:
                from utils.app_paths import user_data_dir
                from utils.report_client import ReportClient
                ReportClient(user_data_dir() / 'reports').enable_automatic()
            except Exception:
                if 'storage_directory' in draft:
                    restore_storage_directory(previous_storage)
                self._show_error(trans.t('report_cleanup_failed'), self.chk_auto_error_reports)
                return
        try:
            if manager is not None and auto_download is not None:
                manager.set_auto_download(auto_download)
                if manager.auto_download != auto_download:
                    raise RuntimeError(trans.t('action_auto_download'))
            if manager is not None and include_beta is not None:
                manager.set_include_beta(include_beta)
                if manager.include_beta != include_beta:
                    raise RuntimeError(trans.t('action_include_beta'))
            if settings is not None:
                settings.setValue(REPORTS_CHOICE_KEY, 'chosen')
            for key, value in draft.items():
                setattr(w, key, value)
            if "_registration_parameters_by_method" in draft:
                refresh = getattr(w, "refresh_registration_parameters", None)
                if callable(refresh):
                    refresh()
            if hasattr(w, "persist_settings"):
                w.persist_settings()
            if settings is not None:
                settings.sync()
                from PyQt6.QtCore import QSettings
                if settings.status() != QSettings.Status.NoError:
                    raise OSError('Could not write OpenFocus settings')
        except Exception as exc:
            if enabling_reports:
                from utils.report_client import revoke_automatic
                revoke_automatic(user_data_dir() / 'reports')
            if settings is not None:
                if previous_choice is None:
                    settings.remove(REPORTS_CHOICE_KEY)
                else:
                    settings.setValue(REPORTS_CHOICE_KEY, previous_choice)
            if 'storage_directory' in draft:
                restore_storage_directory(previous_storage)
            for key, value in previous.items():
                if value is missing:
                    if hasattr(w, key):
                        delattr(w, key)
                else:
                    setattr(w, key, value)
            if "_registration_parameters_by_method" in draft:
                refresh = getattr(w, "refresh_registration_parameters", None)
                if callable(refresh):
                    refresh()
            try:
                if manager is not None and manager.auto_download != previous_auto:
                    manager.set_auto_download(previous_auto)
                if manager is not None and getattr(manager, "include_beta", False) != previous_beta:
                    manager.set_include_beta(previous_beta)
                if hasattr(w, "persist_settings"):
                    w.persist_settings()
            except Exception:
                pass
            self.status.setText(str(exc))
            self.status.show()
            self._fit_page()
            return
        apply_experimental = getattr(w, "apply_experimental_features", None)
        if 'experimental_features' in draft and callable(apply_experimental):
            turned_off = self._initial_fields['experimental_features'] and not draft['experimental_features']
            apply_experimental(fallback_to_stable=turned_off)
        if draft.get('auto_error_reports') is False:
            from ui.report_retry import discard_automatic_reports
            discard_automatic_reports()
        analytics = getattr(w, "usage_analytics", None)
        if analytics is not None and share_usage is not None:
            analytics.set_enabled(share_usage)
        gallery = getattr(w, "gallery_controller", None)
        if danmaku_changed:
            from controllers.gallery_controller import persist_danmaku_settings
            from widgets.danmaku_overlay import DanmakuSettings
            new_danmaku = DanmakuSettings(
                enabled=self.chk_danmaku_enabled.isChecked(),
                position=str(self.combo_danmaku_position.currentData() or "top"),
                font_size=int(self.spin_danmaku_font_size.value()),
                opacity=int(self.spin_danmaku_opacity.value()) / 100.0,
                density=str(self.combo_danmaku_density.currentData() or "medium"),
            ).normalized()
            if gallery is not None:
                gallery.apply_danmaku_settings(new_danmaku, persist=True)
            else:
                persist_danmaku_settings(new_danmaku)
        if gallery_storage_changed:
            if gallery is not None:
                gallery.apply_storage_settings(
                    download_dir=gallery_download_dir,
                    cache_dir=gallery_cache_dir,
                    cache_max_mb=gallery_cache_max_mb,
                )
            else:
                from controllers.gallery_controller import persist_gallery_storage_settings
                persist_gallery_storage_settings(
                    download_dir=gallery_download_dir,
                    cache_dir=gallery_cache_dir,
                    cache_max_mb=gallery_cache_max_mb,
                )
        from utils.logging_setup import log_action
        log_action("settings.applied", changed=sorted(
            key for key in draft if previous[key] != draft[key]),
            usage_enabled=self.chk_usage.isChecked())
        processing = set(draft) - {'experimental_features', 'auto_error_reports', 'restore_last_stack',
                                   'ask_load_scale', 'output_name_prefix', 'storage_directory'}
        if any(previous[key] != draft[key] for key in processing):
            mark_dirty = getattr(w, "_mark_project_dirty", None)
            if mark_dirty is not None:
                mark_dirty()
        self.accept()
