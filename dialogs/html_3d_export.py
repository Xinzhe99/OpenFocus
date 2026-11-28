"""Options dialog for exporting a single-file interactive HTML5 / WebGL 3D viewer."""
from __future__ import annotations

from typing import Optional, Tuple

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

from core.html_3d_export import HTML_3D_MOTION_MODES, Html3DOptions
from core.stereo_3d_export import LightingSettings, RENDER_MODES
from locales import trans


class Html3DExportDialog(QDialog):
    """Configure geometry, texture, and initial viewer options for HTML5 3D export."""

    def __init__(
        self,
        parent=None,
        *,
        source_shape: Optional[Tuple[int, int]] = None,
        has_texture: bool = True,
        default_render_mode: str = "textured",
        default_title: str = "OpenFocus Interactive 3D Model",
    ):
        super().__init__(parent)
        self.setWindowTitle(trans.t("html_3d_dialog_title"))
        self.setMinimumWidth(450)
        self._source_shape = source_shape
        self._has_texture = bool(has_texture)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        hint = QLabel(trans.t("html_3d_dialog_hint"))
        hint.setWordWrap(True)
        hint.setObjectName("mutedLabel")
        layout.addWidget(hint)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # Mesh detail preset (longest grid side)
        self.grid_combo = QComboBox()
        for preset, key in (
            (256, "html_3d_detail_256"),
            (384, "html_3d_detail_384"),
            (512, "html_3d_detail_512"),
            (768, "html_3d_detail_768"),
        ):
            self.grid_combo.addItem(trans.t(key), preset)
        self.grid_combo.setCurrentIndex(1)  # 384 balanced default
        form.addRow(trans.t("html_3d_mesh_detail"), self.grid_combo)

        self.lbl_mesh_estimate = QLabel("")
        self.lbl_mesh_estimate.setObjectName("mutedLabel")
        form.addRow("", self.lbl_mesh_estimate)

        # Texture max width
        self.tex_width_combo = QComboBox()
        self.tex_width_combo.addItem("1024 px", 1024)
        self.tex_width_combo.addItem("1280 px", 1280)
        self.tex_width_combo.addItem("1920 px", 1920)
        self.tex_width_combo.addItem(trans.t("html_anim_original_size"), 0)
        self.tex_width_combo.setCurrentIndex(1)
        form.addRow(trans.t("html_3d_texture_width"), self.tex_width_combo)

        # Image format + quality
        self.format_combo = QComboBox()
        self.format_combo.addItem("WebP", "webp")
        self.format_combo.addItem("JPEG", "jpeg")
        form.addRow(trans.t("html_anim_format"), self.format_combo)

        q_row = QHBoxLayout()
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(70, 96)
        self.quality_slider.setSingleStep(1)
        self.quality_slider.setValue(86)
        self.quality_value = QLabel("86%")
        self.quality_value.setMinimumWidth(38)
        self.quality_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.quality_slider.valueChanged.connect(lambda v: self.quality_value.setText(f"{v}%"))
        q_row.addWidget(self.quality_slider, 1)
        q_row.addWidget(self.quality_value)
        form.addRow(trans.t("html_anim_quality"), q_row)

        # Initial render mode
        self.mode_combo = QComboBox()
        for mode_id, mode_key in (
            ("textured", "depth_3d_mode_textured"),
            ("shaded_clay", "depth_3d_mode_clay"),
            ("depth_color", "depth_3d_mode_depth_color"),
            ("wireframe", "depth_3d_mode_wireframe"),
            ("textured_wireframe", "depth_3d_mode_textured_wireframe"),
        ):
            self.mode_combo.addItem(trans.t(mode_key), mode_id)
        mode_idx = self.mode_combo.findData(default_render_mode)
        if mode_idx >= 0:
            self.mode_combo.setCurrentIndex(mode_idx)
        form.addRow(trans.t("depth_3d_render_mode"), self.mode_combo)

        # Auto motion preset
        self.motion_combo = QComboBox()
        for motion_id, motion_key in (
            ("off", "html_3d_motion_off"),
            ("turntable_360", "depth_3d_motion_360"),
            ("rock_25", "depth_3d_motion_rock"),
            ("orbit_pitch", "depth_3d_motion_orbit_pitch"),
        ):
            self.motion_combo.addItem(trans.t(motion_key), motion_id)
        form.addRow(trans.t("depth_3d_motion_mode"), self.motion_combo)

        # Viewer theme
        self.theme_combo = QComboBox()
        self.theme_combo.addItem(trans.t("html_3d_theme_dark"), "dark")
        self.theme_combo.addItem(trans.t("html_3d_theme_light"), "light")
        self.theme_combo.addItem(trans.t("html_3d_theme_auto"), "auto")
        form.addRow(trans.t("html_3d_theme_label"), self.theme_combo)

        # Viewer title
        self.title_edit = QLineEdit(default_title or "OpenFocus Interactive 3D Model")
        form.addRow(trans.t("html_anim_title_label"), self.title_edit)

        self.footer_chk = QCheckBox(trans.t("html_anim_show_footer"))
        self.footer_chk.setChecked(True)
        form.addRow("", self.footer_chk)

        if not self._has_texture:
            self.tex_width_combo.setEnabled(False)
            self.format_combo.setEnabled(False)
            self.quality_slider.setEnabled(False)

        self.grid_combo.currentIndexChanged.connect(self._update_estimate)
        self._update_estimate()

        layout.addLayout(form)

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

    def _update_estimate(self) -> None:
        preset = int(self.grid_combo.currentData() or 384)
        if self._source_shape and min(self._source_shape) >= 2:
            sh, sw = self._source_shape[:2]
            if sw >= sh:
                gw = min(sw, preset)
                gh = max(2, min(sh, int(round(gw * (sh / float(sw))))))
            else:
                gh = min(sh, preset)
                gw = max(2, min(sw, int(round(gh * (sw / float(sh))))))
            tris = (gw - 1) * (gh - 1) * 2
            self.lbl_mesh_estimate.setText(f"{gw} × {gh} · {tris:,} △")
        else:
            self.lbl_mesh_estimate.setText(f"≤ {preset} × {preset}")

    def build_options(
        self,
        *,
        initial_yaw: float = -35.0,
        initial_pitch: float = 42.0,
        initial_zoom: float = 1.0,
        relief: float = 0.15,
        invert_depth: bool = False,
        lighting: Optional[LightingSettings] = None,
        physical_z_range_um: Optional[Tuple[float, float]] = None,
        pixel_size_um: Optional[float] = None,
        colormap_name: str = "",
    ) -> Html3DOptions:
        mode = str(self.mode_combo.currentData() or "textured")
        if mode not in RENDER_MODES:
            mode = "textured"
        motion = str(self.motion_combo.currentData() or "off")
        if motion not in HTML_3D_MOTION_MODES:
            motion = "off"
        return Html3DOptions(
            max_grid_size=int(self.grid_combo.currentData() or 384),
            max_texture_width=int(self.tex_width_combo.currentData() or 0),
            image_format=str(self.format_combo.currentData() or "webp"),
            quality=int(self.quality_slider.value()),
            render_mode=mode,
            relief=float(relief),
            initial_yaw=float(initial_yaw),
            initial_pitch=float(initial_pitch),
            initial_zoom=float(initial_zoom),
            auto_motion=motion,
            lighting=lighting or LightingSettings(),
            title=self.title_edit.text().strip() or "OpenFocus Interactive 3D Model",
            theme=str(self.theme_combo.currentData() or "dark"),
            show_footer=self.footer_chk.isChecked(),
            invert_depth=bool(invert_depth),
            physical_z_range_um=physical_z_range_um,
            pixel_size_um=pixel_size_um,
            colormap_name=colormap_name,
        ).normalized()
