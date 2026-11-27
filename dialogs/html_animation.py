"""Options dialog for exporting an interactive single-file HTML5 focus-stack viewer."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from core.html_animation_export import HtmlAnimationOptions
from locales import trans
from widgets.stepper import StepperSpinBox


class HtmlAnimationDialog(QDialog):
    """Configure options for single-file interactive HTML5 focus-stack export."""

    def __init__(
        self,
        parent=None,
        *,
        has_registered: bool = False,
        has_input: bool = True,
        has_fused: bool = False,
        default_title: str = "OpenFocus Interactive Focus Stack",
    ):
        super().__init__(parent)
        self.setWindowTitle(trans.t("html_anim_dialog_title"))
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        hint = QLabel(trans.t("html_anim_dialog_hint"))
        hint.setWordWrap(True)
        hint.setObjectName("mutedLabel")
        layout.addWidget(hint)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # Source stack selection
        self.source_combo = QComboBox()
        if has_registered:
            self.source_combo.addItem(trans.t("menu_registered_stack"), "registered")
        if has_input:
            self.source_combo.addItem(trans.t("menu_input_stack"), "input")
        form.addRow(trans.t("html_anim_source_label"), self.source_combo)

        # Max width preset
        self.width_combo = QComboBox()
        self.width_combo.addItem("800 px", 800)
        self.width_combo.addItem("1280 px", 1280)
        self.width_combo.addItem("1920 px", 1920)
        self.width_combo.addItem(trans.t("html_anim_original_size"), 0)
        self.width_combo.setCurrentIndex(1)  # 1280 px default
        form.addRow(trans.t("html_anim_max_width"), self.width_combo)

        # Image format
        self.format_combo = QComboBox()
        self.format_combo.addItem("WebP", "webp")
        self.format_combo.addItem("JPEG", "jpeg")
        form.addRow(trans.t("html_anim_format"), self.format_combo)

        # Quality (75-95)
        q_row = QHBoxLayout()
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(75, 95)
        self.quality_slider.setSingleStep(1)
        self.quality_slider.setValue(85)
        self.quality_value = QLabel("85%")
        self.quality_value.setMinimumWidth(38)
        self.quality_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.quality_slider.valueChanged.connect(lambda v: self.quality_value.setText(f"{v}%"))
        q_row.addWidget(self.quality_slider, 1)
        q_row.addWidget(self.quality_value)
        form.addRow(trans.t("html_anim_quality"), q_row)

        # Default FPS
        self.fps_spin = StepperSpinBox(minimum=1, maximum=60, decimals=0, step=1, value=10)
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.setSuffix(" FPS")
        form.addRow(trans.t("html_anim_fps"), self.fps_spin)

        # Viewer title
        self.title_edit = QLineEdit(default_title or "OpenFocus Interactive Focus Stack")
        form.addRow(trans.t("html_anim_title_label"), self.title_edit)

        # Toggles
        self.ping_pong_chk = QCheckBox(trans.t("html_anim_ping_pong"))
        self.ping_pong_chk.setChecked(True)
        form.addRow("", self.ping_pong_chk)

        self.include_fused_chk = QCheckBox(trans.t("html_anim_include_fused"))
        self.include_fused_chk.setEnabled(bool(has_fused))
        self.include_fused_chk.setChecked(bool(has_fused))
        form.addRow("", self.include_fused_chk)

        self.footer_chk = QCheckBox(trans.t("html_anim_show_footer"))
        self.footer_chk.setChecked(True)
        form.addRow("", self.footer_chk)

        layout.addLayout(form)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton(trans.t("html_anim_export_btn"))
        ok_btn.setObjectName("primaryButton")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def selected_source(self) -> str:
        return str(self.source_combo.currentData() or "input")

    def options(self) -> HtmlAnimationOptions:
        return HtmlAnimationOptions(
            max_width=int(self.width_combo.currentData() or 0),
            quality=int(self.quality_slider.value()),
            image_format=str(self.format_combo.currentData() or "webp"),
            fps=float(self.fps_spin.value()),
            ping_pong=self.ping_pong_chk.isChecked(),
            autoplay=False,
            include_fused_frame=self.include_fused_chk.isEnabled() and self.include_fused_chk.isChecked(),
            title=self.title_edit.text().strip() or "OpenFocus Interactive Focus Stack",
            show_footer=self.footer_chk.isChecked(),
            theme="auto",
        ).normalized()
