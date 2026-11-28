"""Options dialogs for direct 3D stereoscopic & turntable video exports."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from core.stereo_3d_export import LightingSettings, TurntableOptions
from locales import trans
from ui.context_help import register_help_target
from widgets.stepper import StepperSpinBox


class AnaglyphExportDialog(QDialog):
    """Configure Red-Cyan / Red-Green Anaglyph 3D export."""

    def __init__(self, parent=None, *, default_invert: bool = False):
        super().__init__(parent)
        self.setWindowTitle(trans.t("depth_3d_anaglyph_title"))
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(trans.t("depth_3d_anaglyph_red_cyan"), "red_cyan")
        self.mode_combo.addItem(trans.t("depth_3d_anaglyph_red_green"), "red_green")
        form.addRow(trans.t("depth_3d_stereo_mode"), self.mode_combo)

        sep_row = QHBoxLayout()
        self.sep_slider = QSlider(Qt.Orientation.Horizontal)
        self.sep_slider.setRange(5, 100)  # 0.5% to 10.0%
        self.sep_slider.setValue(30)      # 3.0%
        self.sep_value = QLabel("3.0%")
        self.sep_value.setMinimumWidth(44)
        self.sep_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.sep_slider.valueChanged.connect(lambda v: self.sep_value.setText(f"{v / 10.0:.1f}%"))
        sep_row.addWidget(self.sep_slider, 1)
        sep_row.addWidget(self.sep_value)
        form.addRow(trans.t("depth_3d_stereo_separation"), sep_row)

        self.conv_spin = StepperSpinBox(minimum=0.0, maximum=1.0, decimals=2, step=0.05, value=0.50)
        self.conv_spin.setRange(0.0, 1.0)
        self.conv_spin.setValue(0.50)
        form.addRow(trans.t("depth_3d_stereo_convergence"), self.conv_spin)

        self.invert_chk = QCheckBox(trans.t("depth_invert"))
        self.invert_chk.setChecked(bool(default_invert))
        form.addRow("", self.invert_chk)

        layout.addLayout(form)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.setObjectName("primaryButton")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btns.addWidget(cancel_btn)
        btns.addWidget(ok_btn)
        layout.addLayout(btns)
        register_help_target(self, "stereo_3d")

    def options(self) -> dict:
        return {
            "mode": str(self.mode_combo.currentData() or "red_cyan"),
            "separation": self.sep_slider.value() / 1000.0,
            "convergence": float(self.conv_spin.value()),
            "invert_depth": self.invert_chk.isChecked(),
        }


class StereoPairExportDialog(QDialog):
    """Configure Parallel / Cross-Eyed side-by-side stereo pair export."""

    def __init__(self, parent=None, *, default_invert: bool = False):
        super().__init__(parent)
        self.setWindowTitle(trans.t("depth_3d_stereo_pair_title"))
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(trans.t("depth_3d_stereo_parallel"), "parallel")
        self.mode_combo.addItem(trans.t("depth_3d_stereo_cross_eyed"), "cross_eyed")
        form.addRow(trans.t("depth_3d_stereo_layout"), self.mode_combo)

        sep_row = QHBoxLayout()
        self.sep_slider = QSlider(Qt.Orientation.Horizontal)
        self.sep_slider.setRange(5, 100)
        self.sep_slider.setValue(30)
        self.sep_value = QLabel("3.0%")
        self.sep_value.setMinimumWidth(44)
        self.sep_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.sep_slider.valueChanged.connect(lambda v: self.sep_value.setText(f"{v / 10.0:.1f}%"))
        sep_row.addWidget(self.sep_slider, 1)
        sep_row.addWidget(self.sep_value)
        form.addRow(trans.t("depth_3d_stereo_separation"), sep_row)

        self.conv_spin = StepperSpinBox(minimum=0.0, maximum=1.0, decimals=2, step=0.05, value=0.50)
        self.conv_spin.setRange(0.0, 1.0)
        self.conv_spin.setValue(0.50)
        form.addRow(trans.t("depth_3d_stereo_convergence"), self.conv_spin)

        self.divider_chk = QCheckBox(trans.t("depth_3d_stereo_divider"))
        self.divider_chk.setChecked(True)
        form.addRow("", self.divider_chk)

        self.invert_chk = QCheckBox(trans.t("depth_invert"))
        self.invert_chk.setChecked(bool(default_invert))
        form.addRow("", self.invert_chk)

        layout.addLayout(form)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.setObjectName("primaryButton")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btns.addWidget(cancel_btn)
        btns.addWidget(ok_btn)
        layout.addLayout(btns)
        register_help_target(self, "stereo_3d")

    def options(self) -> dict:
        return {
            "mode": str(self.mode_combo.currentData() or "parallel"),
            "separation": self.sep_slider.value() / 1000.0,
            "convergence": float(self.conv_spin.value()),
            "invert_depth": self.invert_chk.isChecked(),
            "divider_width": 4 if self.divider_chk.isChecked() else 0,
        }


class TurntableExportDialog(QDialog):
    """Configure 3D turntable rotation MP4 video / animated GIF export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("depth_3d_turntable_title"))
        self.setMinimumWidth(410)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.format_combo = QComboBox()
        self.format_combo.addItem("MP4 Video (.mp4)", "mp4")
        self.format_combo.addItem("Animated GIF (.gif)", "gif")
        form.addRow(trans.t("depth_3d_video_format"), self.format_combo)

        self.motion_combo = QComboBox()
        self.motion_combo.addItem(trans.t("depth_3d_motion_360"), "turntable_360")
        self.motion_combo.addItem(trans.t("depth_3d_motion_rock"), "rock_25")
        self.motion_combo.addItem(trans.t("depth_3d_motion_orbit_pitch"), "orbit_pitch")
        form.addRow(trans.t("depth_3d_motion_mode"), self.motion_combo)

        self.res_combo = QComboBox()
        self.res_combo.addItem("640 × 480", (640, 480))
        self.res_combo.addItem("800 × 600", (800, 600))
        self.res_combo.addItem("1280 × 720", (1280, 720))
        self.res_combo.setCurrentIndex(1)
        form.addRow(trans.t("depth_3d_video_resolution"), self.res_combo)

        self.frames_spin = StepperSpinBox(minimum=12, maximum=180, decimals=0, step=6, value=48)
        self.frames_spin.setRange(12, 180)
        self.frames_spin.setValue(48)
        form.addRow(trans.t("depth_3d_video_frames"), self.frames_spin)

        self.fps_spin = StepperSpinBox(minimum=6, maximum=60, decimals=0, step=2, value=24)
        self.fps_spin.setRange(6, 60)
        self.fps_spin.setValue(24)
        self.fps_spin.setSuffix(" FPS")
        form.addRow(trans.t("depth_3d_video_fps"), self.fps_spin)

        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        layout.addLayout(form)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.setObjectName("primaryButton")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btns.addWidget(cancel_btn)
        btns.addWidget(ok_btn)
        layout.addLayout(btns)
        register_help_target(self, "stereo_3d")

    def _on_format_changed(self):
        fmt = self.format_combo.currentData()
        if fmt == "gif" and self.fps_spin.value() > 20:
            self.fps_spin.setValue(15)
        elif fmt == "mp4" and self.fps_spin.value() == 15:
            self.fps_spin.setValue(24)

    def build_options(
        self,
        *,
        start_yaw: float,
        base_pitch: float,
        zoom: float,
        relief: float,
        invert: bool,
        mesh_step: int,
        render_mode: str,
        lighting: LightingSettings,
        background_rgb: tuple[int, int, int],
        accent_rgb: tuple[int, int, int],
        wireframe_rgb: tuple[int, int, int] | None = None,
    ) -> TurntableOptions:
        width, height = self.res_combo.currentData() or (800, 600)
        return TurntableOptions(
            output_format=str(self.format_combo.currentData() or "mp4"),
            width=int(width),
            height=int(height),
            frame_count=int(self.frames_spin.value()),
            fps=int(self.fps_spin.value()),
            motion_mode=str(self.motion_combo.currentData() or "turntable_360"),
            start_yaw=start_yaw,
            base_pitch=base_pitch,
            zoom=zoom,
            relief=relief,
            invert=invert,
            mesh_step=mesh_step,
            render_mode=render_mode,
            lighting=lighting,
            background_rgb=background_rgb,
            accent_rgb=accent_rgb,
            wireframe_rgb=wireframe_rgb,
        ).normalized()
