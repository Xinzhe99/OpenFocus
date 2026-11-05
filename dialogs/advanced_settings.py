"""One page for the settings that used to be scattered across menus.

Single scroll page with four sections (performance / registration / AI /
startup), not a tab book: the whole point is to see everything at once instead
of hunting through four dialogs and a menu.
"""
from PyQt6.QtCore import Qt
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from constants import (
    TILE_BLOCK_MAX, TILE_BLOCK_MIN, TILE_OVERLAP_MAX, TILE_THRESHOLD_MAX,
    TILE_THRESHOLD_MIN,
)
from core.algorithms import (registration_algorithms, registration_parameter_specs,
                             resolve_registration_parameters)
from core.frame_storage import configure_storage_directory
from locales import trans
from utils.export_paths import prefix_output_name, valid_output_prefix
from widgets.stepper import StepperSpinBox

# Registration downscale bounds mirror dialogs/settings.py, the other surface
# that edits the same persisted value.
REG_WIDTH_MIN = 64
REG_WIDTH_MAX = 8192


def _section(parent_layout, title: str) -> QFormLayout:
    """Bold heading + a form, appended to `parent_layout`."""
    head = QLabel(title)
    f = head.font()
    f.setBold(True)
    head.setFont(f)
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


class AdvancedSettingsDialog(QDialog):
    """Advanced settings in one scrollable page."""

    def __init__(self, window, parent=None):
        super().__init__(parent or window)
        self.window_main = window
        self.setWindowTitle(trans.t("advanced_settings_title"))
        self.resize(640, 620)

        root = QVBoxLayout(self)
        intro = QLabel(trans.t("advanced_settings_intro"))
        intro.setWordWrap(True)
        intro.setObjectName("mutedLabel")
        root.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.setContentsMargins(14, 10, 14, 10)
        page_lay.setSpacing(10)
        scroll.setWidget(page)
        root.addWidget(scroll, 1)

        w = window

        # ---- Performance -------------------------------------------------
        perf = _section(page_lay, trans.t("advanced_tab_performance"))
        self.chk_tile = QCheckBox()
        self.chk_tile.setChecked(bool(getattr(w, "tile_enabled", True)))
        self.chk_tile.setToolTip(trans.t("advanced_tile_enabled_hint"))
        perf.addRow(trans.t("advanced_tile_enabled"), self.chk_tile)
        self.spin_block = StepperSpinBox(
            minimum=TILE_BLOCK_MIN, maximum=TILE_BLOCK_MAX, step=128,
            decimals=0, value=int(getattr(w, "tile_block_size", 1024)))
        perf.addRow(trans.t("advanced_tile_block"), self.spin_block)
        self.spin_overlap = StepperSpinBox(
            minimum=0, maximum=TILE_OVERLAP_MAX, step=32, decimals=0,
            value=int(getattr(w, "tile_overlap", 256)))
        perf.addRow(trans.t("advanced_tile_overlap"), self.spin_overlap)
        self.spin_block.valueChanged.connect(self._sync_overlap_limit)
        self._sync_overlap_limit(self.spin_block.value())
        self.spin_threshold = StepperSpinBox(
            minimum=TILE_THRESHOLD_MIN, maximum=TILE_THRESHOLD_MAX, step=256,
            decimals=0, value=int(getattr(w, "tile_threshold", 2048)))
        perf.addRow(trans.t("advanced_tile_threshold"), self.spin_threshold)
        storage_row = QWidget()
        storage_layout = QHBoxLayout(storage_row)
        storage_layout.setContentsMargins(0, 0, 0, 0)
        self.storage_directory = QLineEdit(getattr(w, "storage_directory", ""))
        self.storage_directory.setPlaceholderText(trans.t("advanced_storage_default"))
        storage_layout.addWidget(self.storage_directory, 1)
        browse = QPushButton(trans.t("advanced_storage_browse"))
        browse.clicked.connect(self._browse_storage_directory)
        storage_layout.addWidget(browse)
        perf.addRow(trans.t("advanced_storage_directory"), storage_row)
        saved_threads = max(0, int(getattr(w, "thread_count", 0)))
        self.spin_threads = StepperSpinBox(
            minimum=0, maximum=max(256, saved_threads), step=1, decimals=0,
            value=saved_threads)
        perf.addRow(trans.t("advanced_threads"), self.spin_threads)
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
        self.tile_note = _note(page_lay, trans.t("advanced_tile_note"))
        def show_tile_controls(enabled):
            for control in (self.spin_block, self.spin_overlap, self.spin_threshold):
                perf.setRowVisible(control, enabled)
            self.tile_note.setVisible(enabled)
        self.chk_tile.toggled.connect(show_tile_controls)
        show_tile_controls(self.chk_tile.isChecked())

        # ---- Registration ------------------------------------------------
        reg = _section(page_lay, trans.t("advanced_tab_registration"))
        self.spin_reg_width = StepperSpinBox(
            minimum=REG_WIDTH_MIN, maximum=REG_WIDTH_MAX, step=64, decimals=0,
            value=int(getattr(w, "reg_downscale_width", 1024)))
        reg.addRow(trans.t("advanced_reg_width"), self.spin_reg_width)
        self.combo_align_failure = QComboBox()
        self.combo_align_failure.setProperty("controlId", "combo_align_failure")
        failure_spec = next(spec for spec in registration_parameter_specs("homography")
                            if spec.key == "failure_policy")
        for value, label in zip(failure_spec.choices, failure_spec.choice_labels):
            self.combo_align_failure.addItem(trans.t(label), value)
        method = getattr(w, "current_alignment_method", lambda: "none")()
        if method == "none":
            method = "homography"
        getter = getattr(w, "registration_parameters_for", None)
        parameters = getter(method) if callable(getter) else {}
        if callable(getter) and parameters.get("failure_policy") not in failure_spec.choices:
            parameters = getter("homography")
        self._initial_align_failure = parameters.get("failure_policy", failure_spec.default)
        self.combo_align_failure.setCurrentIndex(max(
            0, self.combo_align_failure.findData(self._initial_align_failure)))
        reg.addRow(trans.t(failure_spec.label_key), self.combo_align_failure)
        self.chk_align_cache = QCheckBox()
        self.chk_align_cache.setChecked(
            bool(getattr(w, "align_cache_enabled", True)))
        reg.addRow(trans.t("action_align_cache"), self.chk_align_cache)
        self.chk_collect_alignment_report = QCheckBox()
        self.chk_collect_alignment_report.setProperty("controlId", "chk_collect_alignment_report")
        self.chk_collect_alignment_report.setChecked(
            bool(getattr(w, "collect_alignment_report", False)))
        self.chk_collect_alignment_report.setToolTip(trans.t("alignment_report_collect_hint"))
        reg.addRow(trans.t("alignment_report_collect"), self.chk_collect_alignment_report)
        self.btn_alignment_report = QPushButton(trans.t("alignment_report_title"))
        self.btn_alignment_report.setProperty("controlId", "btn_alignment_report")
        self.btn_alignment_report.clicked.connect(
            lambda: w.render_manager.show_alignment_report(parent=self))
        reg.addRow(self.btn_alignment_report)

        from ui.sensor_cleanup import SensorCleanupControls
        sensor = _section(page_lay, trans.t("sensor_section"))
        self.sensor_cleanup = SensorCleanupControls(
            getattr(w, "defect_mask_path", ""), getattr(w, "sensor_denoise", False))
        sensor.addRow(self.sensor_cleanup)

        # Output names are a user preference, independent of project processing.
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

        # ---- AI ----------------------------------------------------------
        ai = _section(page_lay, trans.t("advanced_tab_ai"))
        self.spin_ai_batch = StepperSpinBox(
            minimum=1, maximum=16, step=1, decimals=0,
            value=int(getattr(w, "ai_batch_size", 2)))
        ai.addRow(trans.t("advanced_ai_batch"), self.spin_ai_batch)

        # ---- Startup & updates -------------------------------------------
        other = _section(page_lay, trans.t("advanced_tab_other"))
        self.chk_restore = QCheckBox()
        self.chk_restore.setChecked(bool(getattr(w, "restore_last_stack", False)))
        other.addRow(trans.t("action_restore_last_stack"), self.chk_restore)
        self.chk_ask_scale = QCheckBox()
        self.chk_ask_scale.setChecked(bool(getattr(w, "ask_load_scale", True)))
        self.chk_ask_scale.setToolTip(trans.t("action_ask_load_scale_hint"))
        other.addRow(trans.t("action_ask_load_scale"), self.chk_ask_scale)
        self.lbl_load_scale = QLabel(
            f"{float(getattr(w, 'load_scale_default', 1.0)) * 100:g}%")
        other.addRow(trans.t("ds_title"), self.lbl_load_scale)
        self.chk_ask_scale.toggled.connect(
            lambda checked: other.setRowVisible(self.lbl_load_scale, not checked))
        other.setRowVisible(self.lbl_load_scale, not self.chk_ask_scale.isChecked())
        self.chk_auto_update = QCheckBox()
        try:
            self.chk_auto_update.setChecked(
                bool(w.update_manager.auto_download))
        except Exception:
            self.chk_auto_update.setChecked(True)
        other.addRow(trans.t("action_auto_download"), self.chk_auto_update)
        auto_hint = QLabel(trans.t("action_auto_download_hint"))
        auto_hint.setObjectName("mutedLabel")
        auto_hint.setWordWrap(True)
        other.addRow("", auto_hint)
        self.chk_usage = QCheckBox()
        self.chk_usage.setProperty("controlId", "chk_usage")
        analytics = getattr(w, "usage_analytics", None)
        self.chk_usage.setChecked(analytics.enabled() if analytics is not None else True)
        other.addRow(trans.t("usage_enabled"), self.chk_usage)
        usage_hint = QLabel(trans.t("usage_disclosure"))
        usage_hint.setObjectName("mutedLabel")
        usage_hint.setWordWrap(True)
        other.addRow("", usage_hint)

        page_lay.addStretch()
        self.status = QLabel()
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        # Qt's own button text follows the Qt translation catalog, which this
        # app does not ship: every language showed "OK/Cancel".
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            trans.t("btn_ok"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(
            trans.t("btn_cancel"))
        buttons.accepted.connect(self._apply)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

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

    def _apply(self):
        w = self.window_main
        spins = [self.spin_threads, self.spin_reg_width, self.spin_ai_batch]
        if self.chk_tile.isChecked():
            spins[:0] = [self.spin_block, self.spin_overlap, self.spin_threshold]
        for spin in spins:
            if not spin.lineEdit().hasAcceptableInput():
                self.status.setText(trans.t("msg_error"))
                spin.lineEdit().setFocus()
                spin.lineEdit().selectAll()
                return
            spin.interpretText()
        if not valid_output_prefix(self.output_prefix.text()):
            self.status.setText(trans.t("output_prefix_hint"))
            self.output_prefix.setFocus()
            return
        sensor_error = self.sensor_cleanup.validate(w)
        if sensor_error:
            self.status.setText(sensor_error)
            return
        draft = {
            "tile_enabled": self.chk_tile.isChecked(),
            "tile_block_size": (int(self.spin_block.value()) if self.chk_tile.isChecked()
                                else getattr(w, "tile_block_size", 1024)),
            "tile_overlap": (int(self.spin_overlap.value()) if self.chk_tile.isChecked()
                             else getattr(w, "tile_overlap", 256)),
            "tile_threshold": (int(self.spin_threshold.value()) if self.chk_tile.isChecked()
                               else getattr(w, "tile_threshold", 2048)),
            "thread_count": max(0, int(self.spin_threads.value())),
            "resource_preference": self.combo_resources.currentData(),
            "reg_downscale_width": int(self.spin_reg_width.value()),
            "align_cache_enabled": self.chk_align_cache.isChecked(),
            "collect_alignment_report": self.chk_collect_alignment_report.isChecked(),
            "output_name_prefix": self.output_prefix.text(),
            "ai_batch_size": max(1, int(self.spin_ai_batch.value())),
            "restore_last_stack": self.chk_restore.isChecked(),
            "ask_load_scale": self.chk_ask_scale.isChecked(),
            "defect_mask_path": self.sensor_cleanup.path.text().strip(),
            "sensor_denoise": self.sensor_cleanup.denoise.isChecked(),
            "storage_directory": self.storage_directory.text().strip(),
        }
        failure_policy = self.combo_align_failure.currentData()
        if failure_policy != self._initial_align_failure:
            memory = {method: dict(values) for method, values in
                      getattr(w, "_registration_parameters_by_method", {}).items()}
            for spec in registration_algorithms():
                policy_spec = next((p for p in registration_parameter_specs(spec.id)
                                    if p.key == "failure_policy"), None)
                if policy_spec and failure_policy in policy_spec.choices:
                    values = dict(memory.get(spec.id, {}))
                    values["failure_policy"] = failure_policy
                    memory[spec.id] = resolve_registration_parameters(spec.id, values)
            draft["_registration_parameters_by_method"] = memory
        previous = {key: getattr(w, key, None) for key in draft}
        manager = getattr(w, "update_manager", None)
        previous_auto = manager.auto_download if manager is not None else None
        previous_storage = getattr(w, "storage_directory", "") or ""
        try:
            configure_storage_directory(draft["storage_directory"])
        except OSError as exc:
            self.status.setText(trans.t("advanced_storage_invalid").format(error=str(exc)))
            self.storage_directory.setFocus()
            return
        try:
            if manager is not None:
                manager.set_auto_download(self.chk_auto_update.isChecked())
                if manager.auto_download != self.chk_auto_update.isChecked():
                    raise RuntimeError(trans.t("action_auto_download"))
            for key, value in draft.items():
                setattr(w, key, value)
            if "_registration_parameters_by_method" in draft:
                refresh = getattr(w, "refresh_registration_parameters", None)
                if callable(refresh):
                    refresh()
            if hasattr(w, "persist_settings"):
                w.persist_settings()
        except Exception as exc:
            configure_storage_directory(previous_storage)
            for key, value in previous.items():
                setattr(w, key, value)
            if "_registration_parameters_by_method" in draft:
                refresh = getattr(w, "refresh_registration_parameters", None)
                if callable(refresh):
                    refresh()
            try:
                if manager is not None and manager.auto_download != previous_auto:
                    manager.set_auto_download(previous_auto)
                if hasattr(w, "persist_settings"):
                    w.persist_settings()
            except Exception:
                pass
            self.status.setText(str(exc))
            return
        analytics = getattr(w, "usage_analytics", None)
        if analytics is not None and analytics.enabled() != self.chk_usage.isChecked():
            analytics.set_enabled(self.chk_usage.isChecked())
        from utils.logging_setup import log_action
        log_action("settings.applied", changed=sorted(
            key for key in draft if previous[key] != draft[key]),
            usage_enabled=self.chk_usage.isChecked())
        processing = set(draft) - {"restore_last_stack", "ask_load_scale", "output_name_prefix", "collect_alignment_report", "storage_directory"}
        if any(previous[key] != draft[key] for key in processing):
            mark_dirty = getattr(w, "_mark_project_dirty", None)
            if mark_dirty is not None:
                mark_dirty()
        self.accept()
