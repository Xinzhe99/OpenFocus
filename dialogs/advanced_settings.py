"""One page for the settings that used to be scattered across menus.

Single scroll page with four sections (performance / registration / AI /
startup), not a tab book: the whole point is to see everything at once instead
of hunting through four dialogs and a menu.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame,
    QLabel, QScrollArea, QVBoxLayout, QWidget,
)

from locales import trans
from widgets.stepper import StepperSpinBox


def _section(parent_layout, title: str) -> QFormLayout:
    """Bold heading + a form, appended to `parent_layout`."""
    head = QLabel(title)
    f = head.font()
    f.setBold(True)
    head.setFont(f)
    parent_layout.addWidget(head)
    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    form.setHorizontalSpacing(16)
    form.setVerticalSpacing(6)
    parent_layout.addLayout(form)
    return form


def _note(parent_layout, text: str) -> None:
    note = QLabel(text)
    note.setWordWrap(True)
    note.setObjectName("mutedLabel")
    parent_layout.addWidget(note)


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
            minimum=256, maximum=4096, step=128, decimals=0,
            value=int(getattr(w, "tile_block_size", 1024)))
        perf.addRow(trans.t("advanced_tile_block"), self.spin_block)
        self.spin_overlap = StepperSpinBox(
            minimum=0, maximum=1024, step=32, decimals=0,
            value=int(getattr(w, "tile_overlap", 256)))
        perf.addRow(trans.t("advanced_tile_overlap"), self.spin_overlap)
        self.spin_threshold = StepperSpinBox(
            minimum=256, maximum=20000, step=256, decimals=0,
            value=int(getattr(w, "tile_threshold", 2048)))
        perf.addRow(trans.t("advanced_tile_threshold"), self.spin_threshold)
        self.spin_threads = StepperSpinBox(
            minimum=1, maximum=64, step=1, decimals=0,
            value=int(getattr(w, "thread_count", 4)))
        perf.addRow(trans.t("advanced_threads"), self.spin_threads)
        _note(page_lay, trans.t("advanced_tile_note"))

        # ---- Registration ------------------------------------------------
        reg = _section(page_lay, trans.t("advanced_tab_registration"))
        self.spin_reg_width = StepperSpinBox(
            minimum=256, maximum=8000, step=128, decimals=0,
            value=int(getattr(w, "reg_downscale_width", 1024)))
        reg.addRow(trans.t("advanced_reg_width"), self.spin_reg_width)
        self.chk_align_cache = QCheckBox()
        self.chk_align_cache.setChecked(
            bool(getattr(w, "align_cache_enabled", True)))
        reg.addRow(trans.t("action_align_cache"), self.chk_align_cache)

        # ---- AI ----------------------------------------------------------
        ai = _section(page_lay, trans.t("advanced_tab_ai"))
        self.spin_ai_batch = StepperSpinBox(
            minimum=1, maximum=16, step=1, decimals=0,
            value=int(getattr(w, "ai_batch_size", 2)))
        ai.addRow(trans.t("advanced_ai_batch"), self.spin_ai_batch)
        self.combo_ai_mode = QComboBox()
        self.combo_ai_mode.addItem(trans.t("radio_ai_soft"), "soft")
        self.combo_ai_mode.addItem(trans.t("radio_ai_hard"), "hard")
        idx = self.combo_ai_mode.findData(getattr(w, "ai_mode", "soft"))
        self.combo_ai_mode.setCurrentIndex(max(0, idx))
        ai.addRow(trans.t("advanced_ai_mode"), self.combo_ai_mode)

        # ---- Startup & updates -------------------------------------------
        other = _section(page_lay, trans.t("advanced_tab_other"))
        self.chk_restore = QCheckBox()
        self.chk_restore.setChecked(bool(getattr(w, "restore_last_stack", False)))
        other.addRow(trans.t("action_restore_last_stack"), self.chk_restore)
        self.chk_ask_scale = QCheckBox()
        self.chk_ask_scale.setChecked(bool(getattr(w, "ask_load_scale", True)))
        other.addRow(trans.t("action_ask_load_scale"), self.chk_ask_scale)
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

        page_lay.addStretch()

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

    @staticmethod
    def _rule():
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("mutedLabel")
        return line

    def _apply(self):
        w = self.window_main
        w.tile_enabled = self.chk_tile.isChecked()
        w.tile_block_size = int(self.spin_block.value())
        overlap = int(self.spin_overlap.value())
        w.tile_overlap = min(overlap, max(0, w.tile_block_size // 2))
        w.tile_threshold = int(self.spin_threshold.value())
        w.thread_count = max(1, int(self.spin_threads.value()))
        w.reg_downscale_width = int(self.spin_reg_width.value())
        w.align_cache_enabled = self.chk_align_cache.isChecked()
        w.ai_batch_size = max(1, int(self.spin_ai_batch.value()))
        w.ai_mode = self.combo_ai_mode.currentData()
        w.restore_last_stack = self.chk_restore.isChecked()
        w.ask_load_scale = self.chk_ask_scale.isChecked()
        # `auto_download` is a read-only property; assigning it raised and the
        # broad except swallowed the error, so the checkbox never took effect.
        manager = getattr(w, "update_manager", None)
        if manager is not None:
            manager.set_auto_download(self.chk_auto_update.isChecked())
        if hasattr(w, "persist_settings"):
            w.persist_settings()
        self.accept()
