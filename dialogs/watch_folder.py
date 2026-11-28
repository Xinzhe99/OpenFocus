"""Non-modal Watch Folder Monitor dialog for OpenFocus."""
from __future__ import annotations

import datetime
import os
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.algorithms import (
    FUSION_DEFAULT,
    fusion_algorithms,
    registration_algorithms,
)
from core.watch_folder import FolderMonitorEngine, WatchFolderConfig
from locales import trans
from ui.algorithm_combo import apply_method_availability, fill_fusion_combo
from ui.context_help import register_help_target
from utils.design_tokens import RADIUS, palette
from utils.file_dialogs import FileDialog
from utils.settings_store import get_settings
from widgets.stepper import StepperSpinBox

WATCH_INPUT_DIR_KEY = "watch_folder/input_dir"
WATCH_OUTPUT_DIR_KEY = "watch_folder/output_dir"
WATCH_SETTLE_KEY = "watch_folder/settle_seconds"
WATCH_FORMAT_KEY = "watch_folder/output_format"


class WatchFolderDialog(QDialog):
    """Non-modal dialog to configure and monitor a watched capture folder."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        engine: Optional[FolderMonitorEngine] = None,
        theme: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.setModal(False)
        self.setObjectName("watchFolderDialog")
        self.setWindowTitle(trans.t("watch_folder_title"))
        self.resize(780, 620)
        self.setMinimumSize(640, 500)

        self.engine = engine if engine is not None else FolderMonitorEngine(self)
        self._theme = theme or getattr(parent, "ui_theme", None) or "light"
        self._status_code = "idle"
        self._log_rows_meta: list[str] = []

        self._build_ui()
        self._wire_engine()
        self._load_preferences()
        self.apply_theme(self._theme)
        self._sync_controls()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # ---- Header banner -----------------------------------------------
        header = QHBoxLayout()
        header.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        self.lbl_title = QLabel(trans.t("watch_folder_title"))
        title_font = QFont(self.lbl_title.font())
        title_font.setPointSize(13)
        title_font.setBold(True)
        self.lbl_title.setFont(title_font)
        self.lbl_title.setObjectName("watchTitleLabel")
        self.lbl_title.setToolTip(trans.t("watch_folder_subtitle"))
        title_col.addWidget(self.lbl_title)

        self.lbl_subtitle = QLabel(trans.t("watch_folder_subtitle"))
        self.lbl_subtitle.setObjectName("mutedLabel")
        self.lbl_subtitle.setWordWrap(True)
        self.lbl_subtitle.hide()
        title_col.addWidget(self.lbl_subtitle)
        header.addLayout(title_col, 1)

        badges = QHBoxLayout()
        badges.setSpacing(8)
        self.lbl_processed = QLabel(
            trans.t("watch_folder_processed_count").format(count=0)
        )
        self.lbl_processed.setObjectName("watchCountBadge")
        self.lbl_processed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badges.addWidget(self.lbl_processed)

        self.lbl_status_badge = QLabel(trans.t("watch_folder_status_idle"))
        self.lbl_status_badge.setObjectName("watchStatusBadge")
        self.lbl_status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badges.addWidget(self.lbl_status_badge)
        header.addLayout(badges)
        root.addLayout(header)

        # ---- Folders group -----------------------------------------------
        self.dirs_group = QGroupBox(trans.t("watch_folder_dirs_group"))
        dirs_grid = QGridLayout(self.dirs_group)
        dirs_grid.setContentsMargins(12, 12, 12, 12)
        dirs_grid.setHorizontalSpacing(10)
        dirs_grid.setVerticalSpacing(8)
        dirs_grid.setColumnStretch(1, 1)

        self.lbl_watch_dir = QLabel(trans.t("watch_folder_input_label"))
        self.watch_dir_edit = QLineEdit()
        self.watch_dir_edit.setPlaceholderText(
            trans.t("watch_folder_input_placeholder")
        )
        self.watch_dir_edit.setProperty("controlId", "watch_dir_edit")
        self.btn_browse_watch = QPushButton(trans.t("batch_btn_browse"))
        self.btn_browse_watch.clicked.connect(self._browse_watch_dir)
        dirs_grid.addWidget(self.lbl_watch_dir, 0, 0)
        dirs_grid.addWidget(self.watch_dir_edit, 0, 1)
        dirs_grid.addWidget(self.btn_browse_watch, 0, 2)

        self.lbl_output_dir = QLabel(trans.t("watch_folder_output_label"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText(
            trans.t("watch_folder_output_placeholder")
        )
        self.output_dir_edit.setProperty("controlId", "watch_output_dir_edit")
        self.btn_browse_output = QPushButton(trans.t("batch_btn_browse"))
        self.btn_browse_output.clicked.connect(self._browse_output_dir)
        dirs_grid.addWidget(self.lbl_output_dir, 1, 0)
        dirs_grid.addWidget(self.output_dir_edit, 1, 1)
        dirs_grid.addWidget(self.btn_browse_output, 1, 2)

        root.addWidget(self.dirs_group)

        # ---- Processing settings group -----------------------------------
        self.settings_group = QGroupBox(trans.t("watch_folder_settings_group"))
        settings_grid = QGridLayout(self.settings_group)
        settings_grid.setContentsMargins(12, 12, 12, 12)
        settings_grid.setHorizontalSpacing(12)
        settings_grid.setVerticalSpacing(8)
        settings_grid.setColumnStretch(1, 1)
        settings_grid.setColumnStretch(3, 1)

        self.lbl_fusion = QLabel(trans.t("group_fusion"))
        self.fusion_combo = QComboBox()
        self.fusion_combo.setProperty("controlId", "watch_fusion_combo")
        fill_fusion_combo(
            self.fusion_combo,
            none_label="radio_no_fusion",
            none_hint="radio_no_fusion_hint",
            window=self.parent_window,
        )
        idx = self.fusion_combo.findData(FUSION_DEFAULT)
        if idx >= 0:
            self.fusion_combo.setCurrentIndex(idx)
        settings_grid.addWidget(self.lbl_fusion, 0, 0)
        settings_grid.addWidget(self.fusion_combo, 0, 1)

        self.lbl_align = QLabel(trans.t("group_registration"))
        self.align_combo = QComboBox()
        self.align_combo.setProperty("controlId", "watch_align_combo")
        for spec in registration_algorithms():
            self.align_combo.addItem(trans.t(spec.label_key), spec.id)
            if spec.hint_key:
                self.align_combo.setItemData(
                    self.align_combo.count() - 1,
                    trans.t(spec.hint_key),
                    Qt.ItemDataRole.ToolTipRole,
                )
        settings_grid.addWidget(self.lbl_align, 0, 2)
        settings_grid.addWidget(self.align_combo, 0, 3)

        self.lbl_format = QLabel(trans.t("watch_folder_format_label"))
        self.format_combo = QComboBox()
        self.format_combo.setProperty("controlId", "watch_format_combo")
        self.format_combo.addItems(["PNG", "TIFF", "JPG", "BMP", "DNG"])
        self.format_combo.setCurrentText("PNG")
        settings_grid.addWidget(self.lbl_format, 1, 0)
        settings_grid.addWidget(self.format_combo, 1, 1)

        self.lbl_settle = QLabel(trans.t("watch_folder_settle_label"))
        self.lbl_settle.setToolTip(trans.t("watch_folder_settle_hint"))
        settle_row = QHBoxLayout()
        settle_row.setContentsMargins(0, 0, 0, 0)
        settle_row.setSpacing(6)
        self.settle_spin = StepperSpinBox(
            minimum=0.5, maximum=60.0, step=0.5, decimals=1, value=2.0
        )
        self.settle_spin.setProperty("controlId", "watch_settle_spin")
        self.settle_spin.setToolTip(trans.t("watch_folder_settle_hint"))
        settle_row.addWidget(self.settle_spin, 1)
        self.lbl_settle_unit = QLabel(trans.t("watch_folder_settle_unit"))
        settle_row.addWidget(self.lbl_settle_unit)
        settings_grid.addWidget(self.lbl_settle, 1, 2)
        settings_grid.addLayout(settle_row, 1, 3)

        self._apply_method_availability()
        self._seed_methods_from_parent()
        root.addWidget(self.settings_group)

        # ---- Controls bar ------------------------------------------------
        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.btn_start = QPushButton(trans.t("watch_folder_btn_start"))
        self.btn_start.setObjectName("primaryButton")
        self.btn_start.setProperty("controlId", "watch_btn_start")
        self.btn_start.clicked.connect(self.start_monitoring)
        controls.addWidget(self.btn_start)

        self.btn_pause = QPushButton(trans.t("watch_folder_btn_pause"))
        self.btn_pause.setProperty("controlId", "watch_btn_pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        controls.addWidget(self.btn_pause)

        self.btn_stop = QPushButton(trans.t("watch_folder_btn_stop"))
        self.btn_stop.setProperty("controlId", "watch_btn_stop")
        self.btn_stop.clicked.connect(self.stop_monitoring)
        controls.addWidget(self.btn_stop)

        controls.addStretch(1)

        self.btn_open_output = QPushButton(trans.t("watch_folder_btn_open_output"))
        self.btn_open_output.setProperty("controlId", "watch_btn_open_output")
        self.btn_open_output.clicked.connect(self._open_output_folder)
        controls.addWidget(self.btn_open_output)

        self.btn_clear_log = QPushButton(trans.t("watch_folder_btn_clear_log"))
        self.btn_clear_log.setProperty("controlId", "watch_btn_clear_log")
        self.btn_clear_log.clicked.connect(self.clear_log)
        controls.addWidget(self.btn_clear_log)

        root.addLayout(controls)

        # ---- Activity Log Table ------------------------------------------
        self.log_group = QGroupBox(trans.t("watch_folder_log_group"))
        log_layout = QVBoxLayout(self.log_group)
        log_layout.setContentsMargins(10, 10, 10, 10)

        self.activity_table = QTableWidget(0, 4)
        self.activity_table.setProperty("controlId", "watch_activity_table")
        self.activity_table.setHorizontalHeaderLabels([
            trans.t("watch_folder_col_time"),
            trans.t("watch_folder_col_stack"),
            trans.t("watch_folder_col_status"),
            trans.t("watch_folder_col_details"),
        ])
        self.activity_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.activity_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.activity_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.activity_table.setAlternatingRowColors(True)
        self.activity_table.setWordWrap(False)
        self.activity_table.verticalHeader().setVisible(False)
        self.activity_table.verticalHeader().setDefaultSectionSize(30)
        self.activity_table.setShowGrid(False)
        hdr = self.activity_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.activity_table.setColumnWidth(0, 82)
        self.activity_table.setColumnWidth(1, 165)
        self.activity_table.setColumnWidth(2, 104)

        log_layout.addWidget(self.activity_table, 1)
        root.addWidget(self.log_group, 1)
        register_help_target(self.dirs_group, "watch_folder")
        register_help_target(self.settings_group, "watch_folder")
        register_help_target(self.fusion_combo, "batch_methods")
        register_help_target(self.align_combo, "batch_methods")
        register_help_target(self.format_combo, "batch_output_format")
        register_help_target(self.settle_spin, "watch_settle")
        register_help_target(self.activity_table, "watch_folder")

    def _apply_method_availability(self) -> None:
        try:
            from core.multi_focus_fusion import is_ai_fusion_available
            ai_available = bool(is_ai_fusion_available())
        except Exception:
            ai_available = False
        apply_method_availability(
            self.fusion_combo,
            fusion_algorithms(),
            self.parent_window,
            ai_available=ai_available,
        )
        apply_method_availability(
            self.align_combo,
            registration_algorithms(),
            self.parent_window,
            alignment=True,
        )

    def _seed_methods_from_parent(self) -> None:
        w = self.parent_window
        if w is None:
            return
        get_f = getattr(w, "current_fusion_method", None)
        if callable(get_f):
            idx = self.fusion_combo.findData(get_f())
            if idx >= 0:
                self.fusion_combo.setCurrentIndex(idx)
        get_a = getattr(w, "current_alignment_method", None)
        if callable(get_a):
            idx = self.align_combo.findData(get_a())
            if idx >= 0:
                self.align_combo.setCurrentIndex(idx)

    def _load_preferences(self) -> None:
        store = get_settings()
        watch_dir = str(store.value(WATCH_INPUT_DIR_KEY, "") or "").strip()
        output_dir = str(store.value(WATCH_OUTPUT_DIR_KEY, "") or "").strip()
        if watch_dir:
            self.watch_dir_edit.setText(watch_dir)
        if output_dir:
            self.output_dir_edit.setText(output_dir)
        try:
            settle = float(store.value(WATCH_SETTLE_KEY, 2.0))
            self.settle_spin.setValue(max(0.5, min(60.0, settle)))
        except (TypeError, ValueError):
            self.settle_spin.setValue(2.0)
        fmt = str(store.value(WATCH_FORMAT_KEY, "PNG") or "PNG").upper()
        if self.format_combo.findText(fmt) >= 0:
            self.format_combo.setCurrentText(fmt)

    def _save_preferences(self) -> None:
        store = get_settings()
        store.setValue(WATCH_INPUT_DIR_KEY, self.watch_dir_edit.text().strip())
        store.setValue(WATCH_OUTPUT_DIR_KEY, self.output_dir_edit.text().strip())
        store.setValue(WATCH_SETTLE_KEY, float(self.settle_spin.value()))
        store.setValue(WATCH_FORMAT_KEY, self.format_combo.currentText().upper())

    def _wire_engine(self) -> None:
        self.engine.started.connect(self._on_engine_started)
        self.engine.stopped.connect(self._on_engine_stopped)
        self.engine.paused_changed.connect(self._on_paused_changed)
        self.engine.status_changed.connect(self._on_status_changed)
        self.engine.stack_detected.connect(self._on_stack_detected)
        self.engine.stack_completed.connect(self._on_stack_completed)
        self.engine.stack_failed.connect(self._on_stack_failed)

    def _browse_watch_dir(self) -> None:
        path = FileDialog.getExistingDirectory(
            self,
            trans.t("watch_folder_input_label"),
            self.watch_dir_edit.text().strip(),
        )
        if path:
            self.watch_dir_edit.setText(path)
            if not self.output_dir_edit.text().strip():
                self.output_dir_edit.setText(os.path.join(path, "OpenFocus_Output"))

    def _browse_output_dir(self) -> None:
        path = FileDialog.getExistingDirectory(
            self,
            trans.t("watch_folder_output_label"),
            self.output_dir_edit.text().strip() or self.watch_dir_edit.text().strip(),
        )
        if path:
            self.output_dir_edit.setText(path)

    def build_config(self) -> WatchFolderConfig:
        w = self.parent_window
        output_dir = self.output_dir_edit.text().strip()
        ledger = os.path.join(output_dir, ".openfocus_watch_ledger.json") if output_dir else None
        return WatchFolderConfig(
            watch_dir=self.watch_dir_edit.text().strip(),
            output_dir=output_dir,
            fusion_method=str(self.fusion_combo.currentData() or FUSION_DEFAULT),
            alignment_method=str(self.align_combo.currentData() or "none"),
            output_format=self.format_combo.currentText().lower(),
            settle_seconds=float(self.settle_spin.value()),
            filename_template=str(getattr(w, "output_filename_template", "") or ""),
            output_name_prefix=str(getattr(w, "output_name_prefix", "") or ""),
            play_sound=bool(getattr(w, "render_complete_sound", False)),
            ledger_path=ledger,
            thread_count=int(getattr(w, "thread_count", 0) or 0),
            resource_preference=str(getattr(w, "resource_preference", "balanced") or "balanced"),
            use_gpu=bool(getattr(w, "use_gpu", True)),
        )

    def start_monitoring(self) -> bool:
        watch_dir = self.watch_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if not watch_dir or not os.path.isdir(watch_dir):
            QMessageBox.warning(
                self,
                trans.t("watch_folder_title"),
                trans.t("watch_folder_err_input"),
            )
            return False
        if not output_dir:
            QMessageBox.warning(
                self,
                trans.t("watch_folder_title"),
                trans.t("watch_folder_err_output"),
            )
            return False
        if os.path.normcase(os.path.realpath(watch_dir)) == os.path.normcase(
            os.path.realpath(output_dir)
        ):
            QMessageBox.warning(
                self,
                trans.t("watch_folder_title"),
                trans.t("watch_folder_err_same_dir"),
            )
            return False
        self._save_preferences()
        try:
            self.engine.start(self.build_config())
        except Exception as exc:
            QMessageBox.critical(self, trans.t("msg_error"), str(exc))
            return False
        self._sync_controls()
        return True

    def toggle_pause(self) -> None:
        if not self.engine.is_running:
            return
        if self.engine.is_paused:
            self.engine.resume()
        else:
            self.engine.pause()
        self._sync_controls()

    def stop_monitoring(self) -> None:
        self.engine.stop()
        self._sync_controls()

    def _open_output_folder(self) -> None:
        target = self.output_dir_edit.text().strip()
        if not target:
            return
        if not os.path.isdir(target):
            try:
                os.makedirs(target, exist_ok=True)
            except OSError:
                return
        try:
            from PyQt6.QtCore import QUrl
            from PyQt6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(target))
        except Exception:
            pass

    def clear_log(self) -> None:
        self.activity_table.setRowCount(0)
        self._log_rows_meta.clear()

    def add_log_entry(
        self,
        stack_name: str,
        status_text: str,
        details: str,
        *,
        level: str = "info",
        timestamp: Optional[str] = None,
    ) -> None:
        ts = timestamp or datetime.datetime.now().strftime("%H:%M:%S")
        row = self.activity_table.rowCount()
        self.activity_table.insertRow(row)
        self._log_rows_meta.append(level)

        items = (
            QTableWidgetItem(ts),
            QTableWidgetItem(stack_name),
            QTableWidgetItem(status_text),
            QTableWidgetItem(details),
        )
        items[0].setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        items[2].setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        for col, item in enumerate(items):
            self.activity_table.setItem(row, col, item)
        self._style_log_row(row, level)
        self.activity_table.scrollToBottom()

    def _style_log_row(self, row: int, level: str) -> None:
        tokens = palette(self._theme)
        color_key = {
            "success": "success",
            "error": "danger",
            "warning": "warning",
            "detected": "accent",
            "info": "textMuted",
        }.get(level, "text")
        status_item = self.activity_table.item(row, 2)
        if status_item is not None:
            status_item.setForeground(QColor(tokens[color_key]))
            font = status_item.font()
            font.setBold(level in ("success", "error", "detected"))
            status_item.setFont(font)

    def _on_engine_started(self, watch_dir: str) -> None:
        self.add_log_entry(
            "—",
            trans.t("watch_folder_status_watching"),
            watch_dir,
            level="info",
        )
        self._sync_controls()

    def _on_engine_stopped(self) -> None:
        self.add_log_entry(
            "—",
            trans.t("watch_folder_status_idle"),
            trans.t("watch_folder_btn_stop"),
            level="info",
        )
        self._sync_controls()

    def _on_paused_changed(self, paused: bool) -> None:
        self.add_log_entry(
            "—",
            trans.t("watch_folder_status_paused" if paused else "watch_folder_status_watching"),
            "",
            level="warning" if paused else "info",
        )
        self._sync_controls()

    def _on_status_changed(self, state: str) -> None:
        self._status_code = state
        self._refresh_status_badge()

    def _on_stack_detected(self, stack_name: str, frame_count: int) -> None:
        self.add_log_entry(
            stack_name,
            trans.t("watch_folder_event_detected"),
            f"{frame_count} frames",
            level="detected",
        )

    def _on_stack_completed(self, stack_name: str, output_path: str, elapsed: float) -> None:
        self.lbl_processed.setText(
            trans.t("watch_folder_processed_count").format(
                count=self.engine.processed_count
            )
        )
        self.add_log_entry(
            stack_name,
            trans.t("watch_folder_event_completed"),
            f"{os.path.basename(output_path)} ({elapsed:.1f}s)",
            level="success",
        )

    def _on_stack_failed(self, stack_name: str, error_message: str) -> None:
        self.add_log_entry(
            stack_name,
            trans.t("watch_folder_event_failed"),
            error_message,
            level="error",
        )

    def _sync_controls(self) -> None:
        running = self.engine.is_running
        paused = self.engine.is_paused
        self.btn_start.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_pause.setText(
            trans.t("watch_folder_btn_resume" if paused else "watch_folder_btn_pause")
        )
        self.btn_stop.setEnabled(running)

        editable = not running
        for widget in (
            self.watch_dir_edit,
            self.output_dir_edit,
            self.btn_browse_watch,
            self.btn_browse_output,
            self.fusion_combo,
            self.align_combo,
            self.format_combo,
            self.settle_spin,
        ):
            widget.setEnabled(editable)
        self._refresh_status_badge()

    def _refresh_status_badge(self) -> None:
        tokens = palette(self._theme)
        key_map = {
            "idle": ("watch_folder_status_idle", tokens["textMuted"], tokens["surfaceActive"], tokens["border"]),
            "watching": ("watch_folder_status_watching", tokens["accentSoftText"], tokens["accentSoft"], tokens["accent"]),
            "paused": ("watch_folder_status_paused", tokens["textMuted"], tokens["surfaceActive"], tokens["border"]),
            "processing": ("watch_folder_status_processing", tokens["success"], tokens["successSoft"], tokens["success"]),
        }
        t_key, fg, bg, border = key_map.get(self._status_code, key_map["idle"])
        self.lbl_status_badge.setText(trans.t(t_key))
        self.lbl_status_badge.setStyleSheet(
            f"QLabel#watchStatusBadge {{"
            f" color: {fg}; background-color: {bg};"
            f" border: 1px solid {border};"
            f" border-radius: {RADIUS['pill']}px;"
            f" padding: 4px 12px; font-weight: 600; }}"
        )

    def apply_theme(self, theme: str = "light") -> None:
        """Apply Light, Dark, or Gray palette styling to this dialog."""
        self._theme = theme if theme in ("light", "dark", "gray") else "light"
        t = palette(self._theme)
        r = RADIUS
        self.setStyleSheet(f"""
            QDialog#watchFolderDialog {{
                background-color: {t['bg']};
                color: {t['text']};
            }}
            QDialog#watchFolderDialog QGroupBox {{
                background-color: {t['surface']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: {r['md']}px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: 600;
            }}
            QDialog#watchFolderDialog QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                top: 2px;
                padding: 0 6px;
                background-color: {t['bg']};
                color: {t['text']};
            }}
            QDialog#watchFolderDialog QLabel {{
                color: {t['text']};
                background: transparent;
            }}
            QDialog#watchFolderDialog QLabel#mutedLabel {{
                color: {t['textMuted']};
            }}
            QDialog#watchFolderDialog QLabel#watchCountBadge {{
                color: {t['textMuted']};
                background-color: {t['surfaceSunken']};
                border: 1px solid {t['border']};
                border-radius: {r['pill']}px;
                padding: 4px 12px;
                font-weight: 500;
            }}
            QDialog#watchFolderDialog QLineEdit,
            QDialog#watchFolderDialog QComboBox,
            QDialog#watchFolderDialog QDoubleSpinBox {{
                background-color: {t['controlBg']};
                color: {t['text']};
                border: 1px solid {t['controlBorder']};
                border-radius: {r['sm']}px;
                padding: 5px 8px;
                min-height: 22px;
            }}
            QDialog#watchFolderDialog QLineEdit:focus,
            QDialog#watchFolderDialog QComboBox:focus,
            QDialog#watchFolderDialog QDoubleSpinBox:focus {{
                border: 1px solid {t['borderFocus']};
            }}
            QDialog#watchFolderDialog QPushButton {{
                background-color: {t['surface']};
                color: {t['text']};
                border: 1px solid {t['controlBorder']};
                border-radius: {r['sm']}px;
                padding: 6px 14px;
                font-weight: 500;
            }}
            QDialog#watchFolderDialog QPushButton:hover {{
                background-color: {t['surfaceHover']};
            }}
            QDialog#watchFolderDialog QPushButton:disabled {{
                background-color: {t['controlDisabled']};
                color: {t['textFaint']};
                border-color: {t['border']};
            }}
            QDialog#watchFolderDialog QPushButton#primaryButton {{
                background-color: {t['accent']};
                color: {t['textOnAccent']};
                border: 1px solid {t['accent']};
                font-weight: 600;
            }}
            QDialog#watchFolderDialog QPushButton#primaryButton:hover {{
                background-color: {t['accentHover']};
            }}
            QDialog#watchFolderDialog QPushButton#primaryButton:disabled {{
                background-color: {t['controlDisabled']};
                color: {t['textFaint']};
                border-color: {t['border']};
            }}
            QDialog#watchFolderDialog QTableWidget {{
                background-color: {t['surfaceSunken']};
                alternate-background-color: {t['surface']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: {r['sm']}px;
                selection-background-color: {t['accentSoft']};
                selection-color: {t['text']};
            }}
            QDialog#watchFolderDialog QHeaderView::section {{
                background-color: {t['surfaceHover']};
                color: {t['textMuted']};
                border: none;
                border-bottom: 1px solid {t['border']};
                padding: 6px 8px;
                font-weight: 600;
            }}
        """)
        self.settle_spin.reapply_theme()
        self._refresh_status_badge()
        for row, level in enumerate(self._log_rows_meta):
            if row < self.activity_table.rowCount():
                self._style_log_row(row, level)

    def retranslate_ui(self) -> None:
        self.setWindowTitle(trans.t("watch_folder_title"))
        self.lbl_title.setText(trans.t("watch_folder_title"))
        self.lbl_subtitle.setText(trans.t("watch_folder_subtitle"))
        self.dirs_group.setTitle(trans.t("watch_folder_dirs_group"))
        self.lbl_watch_dir.setText(trans.t("watch_folder_input_label"))
        self.watch_dir_edit.setPlaceholderText(trans.t("watch_folder_input_placeholder"))
        self.lbl_output_dir.setText(trans.t("watch_folder_output_label"))
        self.output_dir_edit.setPlaceholderText(trans.t("watch_folder_output_placeholder"))
        self.btn_browse_watch.setText(trans.t("batch_btn_browse"))
        self.btn_browse_output.setText(trans.t("batch_btn_browse"))
        self.settings_group.setTitle(trans.t("watch_folder_settings_group"))
        self.lbl_fusion.setText(trans.t("group_fusion"))
        self.lbl_align.setText(trans.t("group_registration"))
        self.lbl_format.setText(trans.t("watch_folder_format_label"))
        self.lbl_settle.setText(trans.t("watch_folder_settle_label"))
        self.lbl_settle_unit.setText(trans.t("watch_folder_settle_unit"))
        self.btn_start.setText(trans.t("watch_folder_btn_start"))
        self.btn_stop.setText(trans.t("watch_folder_btn_stop"))
        self.btn_open_output.setText(trans.t("watch_folder_btn_open_output"))
        self.btn_clear_log.setText(trans.t("watch_folder_btn_clear_log"))
        self.log_group.setTitle(trans.t("watch_folder_log_group"))
        self.activity_table.setHorizontalHeaderLabels([
            trans.t("watch_folder_col_time"),
            trans.t("watch_folder_col_stack"),
            trans.t("watch_folder_col_status"),
            trans.t("watch_folder_col_details"),
        ])
        self.lbl_processed.setText(
            trans.t("watch_folder_processed_count").format(count=self.engine.processed_count)
        )
        self._sync_controls()

    def shutdown_running(self) -> bool:
        self.engine.stop()
        return False

    def closeEvent(self, event) -> None:
        self._save_preferences()
        self.engine.stop()
        super().closeEvent(event)
