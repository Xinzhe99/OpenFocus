"""Scale bar settings dialog with a live preview thumbnail.

The dialog edits a local draft. Previously the preview wrote the manual
calibration straight into the main window, so pressing Cancel kept the new
value and later exports used it. It also had no explicit "metadata vs manual"
choice: a manual value of 0 silently meant "use metadata", which is not
discoverable and contradicted the localized hint.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFontComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import cv2
import numpy as np

from locales import trans
from utils import scalebar
from widgets.stepper import StepperSpinBox

CALIBRATION_METADATA = "metadata"
CALIBRATION_MANUAL = "manual"


class ScaleBarDialog(QDialog):
    """Draft editor for calibration, placement, style and target."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("dialog_scalebar_title"))
        self.setMinimumSize(880, 560)
        self._apply_style()

        window = parent
        self._detected = getattr(window, "source_px_um", None)
        self._work_scale = float(
            getattr(window, "current_scale_factor", 1.0) or 1.0)
        stored_manual = float(
            getattr(window, "scale_um_per_px_manual", 0.0) or 0.0)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(8)

        header = QLabel(self._header_text())
        header.setObjectName("mutedLabel")
        header.setWordWrap(True)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        # ---- left: scrollable form --------------------------------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        form_host = QWidget()
        form_col = QVBoxLayout(form_host)
        form_col.setContentsMargins(0, 0, 8, 0)
        form_col.setSpacing(8)

        form_col.addWidget(self._section(trans.t("scalebar_section_calibration")))
        form = QFormLayout()
        form_col.addLayout(form)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem(trans.t("scalebar_mode_metadata"),
                                CALIBRATION_METADATA)
        self.mode_combo.addItem(trans.t("scalebar_mode_manual"),
                                CALIBRATION_MANUAL)
        has_manual = stored_manual > 0
        self.mode_combo.setCurrentIndex(
            self.mode_combo.findData(
                CALIBRATION_MANUAL if has_manual else CALIBRATION_METADATA))
        form.addRow(trans.t("scalebar_calibration_mode"), self.mode_combo)

        self.px_spin = StepperSpinBox(
            minimum=0.0, maximum=1000.0, decimals=4, step=0.01)
        self._adapt_px_decimals(stored_manual)
        self.px_spin.setValue(stored_manual)
        self.px_spin.setToolTip(trans.t("scalebar_manual_hint"))
        self.px_spin.valueChanged.connect(self._adapt_px_decimals)
        form.addRow(trans.t("scalebar_px_um"), self.px_spin)

        if self._detected:
            self.detect_lbl = QLabel(
                trans.t("scalebar_detected").format(value=f"{self._detected:g}"))
        else:
            self.detect_lbl = QLabel(trans.t("scalebar_no_metadata"))
        self.detect_lbl.setWordWrap(True)
        form.addRow("", self.detect_lbl)

        self.effective_lbl = QLabel("")
        self.effective_lbl.setWordWrap(True)
        self.effective_lbl.setObjectName("mutedLabel")
        form.addRow("", self.effective_lbl)

        # ---- placement ---------------------------------------------------
        form_col.addWidget(self._section(trans.t("scalebar_section_place")))
        place_form = QFormLayout()
        form_col.addLayout(place_form)
        self.pos_combo = QComboBox()
        for code, label in (("bottom-right", trans.t("scalebar_pos_br")),
                            ("bottom-left", trans.t("scalebar_pos_bl")),
                            ("top-right", trans.t("scalebar_pos_tr")),
                            ("top-left", trans.t("scalebar_pos_tl")),
                            ("custom", trans.t("scalebar_pos_custom"))):
            self.pos_combo.addItem(label, code)
        pos = getattr(window, "scale_bar_position", "bottom-right")
        self.pos_combo.setCurrentIndex(
            max(0, self.pos_combo.findData(pos)))
        place_form.addRow(trans.t("scalebar_position"), self.pos_combo)

        self.custom_row = QWidget()
        custom_lay = QVBoxLayout(self.custom_row)
        custom_lay.setContentsMargins(0, 0, 0, 0)
        custom_lay.setSpacing(4)
        self.custom_x = self._percent_row(
            custom_lay, "X", float(getattr(window, "scale_bar_custom_x",
                                           0.95) or 0.0))
        self.custom_y = self._percent_row(
            custom_lay, "Y", float(getattr(window, "scale_bar_custom_y",
                                           0.95) or 0.0))
        place_form.addRow("", self.custom_row)
        self.custom_row.setVisible(pos == "custom")

        # ---- appearance --------------------------------------------------
        form_col.addWidget(self._section(trans.t("scalebar_section_text")))
        text_form = QFormLayout()
        form_col.addLayout(text_form)
        self.font_combo = QFontComboBox()
        family = getattr(window, "scale_bar_font_family", "") or ""
        self.font_combo.setCurrentFont(QFont(family or self.font().family()))
        text_form.addRow(trans.t("scalebar_font"), self.font_combo)

        size_row = QWidget()
        size_lay = QHBoxLayout(size_row)
        size_lay.setContentsMargins(0, 0, 0, 0)
        size_lay.setSpacing(8)
        self.size_spin = StepperSpinBox(
            minimum=0, maximum=400, decimals=0, step=2)
        self.size_spin.setValue(
            float(getattr(window, "scale_bar_font_size", 0.0) or 0.0))
        self.size_spin.setMinimumWidth(96)
        size_lay.addWidget(self.size_spin)
        size_lay.addWidget(QLabel(trans.t("scalebar_font_auto")), 1)
        text_form.addRow(trans.t("scalebar_font_size"), size_row)

        self.bold_chk = QCheckBox(trans.t("scalebar_bold"))
        self.bold_chk.setChecked(
            bool(getattr(window, "scale_bar_font_bold", False)))
        text_form.addRow("", self.bold_chk)

        self.align_combo = QComboBox()
        for code, label in (("auto", trans.t("scalebar_align_auto")),
                            ("left", trans.t("scalebar_align_left")),
                            ("center", trans.t("scalebar_align_center")),
                            ("right", trans.t("scalebar_align_right"))):
            self.align_combo.addItem(label, code)
        self.align_combo.setCurrentIndex(max(0, self.align_combo.findData(
            getattr(window, "scale_bar_text_align", "auto"))))
        text_form.addRow(trans.t("scalebar_text_align"), self.align_combo)

        self.color_combo = QComboBox()
        for code, label in (("auto", trans.t("scalebar_color_auto")),
                            ("white", trans.t("scalebar_color_white")),
                            ("black", trans.t("scalebar_color_black"))):
            self.color_combo.addItem(label, code)
        self.color_combo.setCurrentIndex(max(0, self.color_combo.findData(
            getattr(window, "scale_bar_color", "auto"))))
        text_form.addRow(trans.t("scalebar_color"), self.color_combo)

        self.target_combo = QComboBox()
        for code, label in (
                ("all", trans.t("scalebar_target_all")),
                ("source", trans.t("scalebar_target_source")),
                ("output", trans.t("scalebar_target_output"))):
            self.target_combo.addItem(label, code)
        self.target_combo.setCurrentIndex(max(0, self.target_combo.findData(
            getattr(window, "scale_bar_target", "all"))))
        text_form.addRow(trans.t("scalebar_target"), self.target_combo)

        self.font_note = QLabel("")
        self.font_note.setWordWrap(True)
        self.font_note.setObjectName("mutedLabel")
        text_form.addRow("", self.font_note)

        form_col.addStretch()
        scroll.setWidget(form_host)
        body.addWidget(scroll, 1)

        # ---- right: preview ---------------------------------------------
        preview_col = QVBoxLayout()
        preview_col.setSpacing(6)
        title = QLabel(trans.t("label_preview"))
        bold = title.font()
        bold.setBold(True)
        title.setFont(bold)
        preview_col.addWidget(title)

        self.preview_lbl = QLabel(trans.t("scalebar_preview"))
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setMinimumSize(360, 300)
        self.preview_lbl.setStyleSheet(
            "background: #222; border: 1px solid #444;")
        preview_col.addWidget(self.preview_lbl, 1)

        self.preview_meta = QLabel("")
        self.preview_meta.setWordWrap(True)
        self.preview_meta.setObjectName("mutedLabel")
        preview_col.addWidget(self.preview_meta)
        body.addLayout(preview_col, 1)

        # ---- fixed bottom actions ---------------------------------------
        self._error = QLabel("")
        self._error.setWordWrap(True)
        self._error.setStyleSheet("color:#c0392b;")
        root.addWidget(self._error)

        row = QHBoxLayout()
        row.addStretch()
        self._ok_button = QPushButton(trans.t("label_apply_close"))
        self._ok_button.setObjectName("primaryButton")
        self._ok_button.setDefault(True)
        self._ok_button.clicked.connect(self._save_and_accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(self._ok_button)
        row.addWidget(cancel_btn)
        root.addLayout(row)

        self.pos_combo.currentIndexChanged.connect(
            lambda: self.custom_row.setVisible(
                self.pos_combo.currentData() == "custom"))
        for signal in (self.px_spin.valueChanged,
                       self.mode_combo.currentIndexChanged,
                       self.pos_combo.currentIndexChanged,
                       self.color_combo.currentIndexChanged,
                       self.font_combo.currentFontChanged,
                       self.size_spin.valueChanged,
                       self.bold_chk.toggled,
                       self.align_combo.currentIndexChanged,
                       self.target_combo.currentIndexChanged,
                       self.custom_x.valueChanged,
                       self.custom_y.valueChanged):
            signal.connect(self._refresh_preview)
        self._refresh_preview()

    # ------------------------------------------------------------------
    def _header_text(self) -> str:
        window = self.parent()
        frames = len(getattr(window, "raw_images", None) or [])
        outputs = len(getattr(window, "fusion_results", None) or [])
        images = getattr(window, "raw_images", None) or []
        if not images:
            return trans.t("scalebar_header_empty")
        height, width = images[0].shape[:2]
        depth = images[0].dtype.itemsize * 8
        return trans.t("scalebar_header").format(
            frames=frames, outputs=outputs, width=width, height=height,
            depth=depth, scale=f"{self._work_scale * 100:.0f}")

    def _percent_row(self, parent_layout, caption: str, value: float):
        """A slider plus a numeric percentage box (sliders alone cannot
        reproduce an exact stored position)."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(QLabel(caption))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(round(value * 100)))
        spin = StepperSpinBox(minimum=0, maximum=100, decimals=0, step=1)
        spin.setValue(slider.value())
        spin.setMinimumWidth(70)
        slider.valueChanged.connect(
            lambda v: spin.setValue(v) if spin.value() != v else None)
        spin.valueChanged.connect(
            lambda v: slider.setValue(int(v)) if slider.value() != int(v)
            else None)
        row.addWidget(slider, 1)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return slider

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        font = lbl.font()
        font.setBold(True)
        lbl.setFont(font)
        return lbl

    def _adapt_px_decimals(self, value: float) -> None:
        """Show enough decimals for sub-micron calibration, not more."""
        value = abs(float(value or 0.0))
        if value == 0.0 or value >= 10:
            decimals, step = 2, 0.1
        elif value >= 1:
            decimals, step = 3, 0.01
        elif value >= 0.01:
            decimals, step = 4, 0.001
        else:
            decimals, step = 6, 0.0001
        self.px_spin.setDecimals(decimals)
        self.px_spin.setSingleStep(step)

    def _apply_style(self):
        """Styling comes from the generated application stylesheet."""
        self.setStyleSheet("")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibration_mode(self) -> str:
        return self.mode_combo.currentData() or CALIBRATION_METADATA

    def manual_value(self) -> float:
        return float(self.px_spin.value() or 0.0)

    def effective_px_um(self) -> Optional[float]:
        """The value the bar would use for the CURRENT working pixels.

        The manual field is expressed in original pixels; the working image
        may have been loaded or resized at a different scale, so the drawn
        value is divided by that scale. Metadata is already expressed for the
        loaded pixels by the loader.
        """
        if self.calibration_mode() == CALIBRATION_MANUAL:
            manual = self.manual_value()
            if manual <= 0:
                return None
            scale = self._work_scale if self._work_scale > 0 else 1.0
            return manual / scale
        detected = self._detected
        return float(detected) if detected and detected > 0 else None

    def _effective_text(self) -> str:
        effective = self.effective_px_um()
        if effective is None:
            return trans.t("scalebar_not_calibrated")
        parts = [trans.t("scalebar_effective").format(
            value=f"{effective:g}")]
        if self.calibration_mode() == CALIBRATION_MANUAL \
                and abs(self._work_scale - 1.0) > 1e-6:
            parts.append(trans.t("scalebar_original_value").format(
                value=f"{self.manual_value():g}",
                scale=f"{self._work_scale * 100:.0f}"))
        return " · ".join(parts)

    def _font_family(self) -> str:
        """The chosen family, or "" when it is the dialog's own default."""
        family = self.font_combo.currentFont().family()
        return "" if family == self.font().family() else family

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _preview_image(self):
        """A real target image for the selected scope, or a neutral card."""
        window = self.parent()
        target = self.target_combo.currentData()
        if window is not None:
            if target == "source":
                raw = getattr(window, "raw_images", None) or []
                if raw:
                    index = max(0, int(getattr(
                        window, "current_display_index", 0) or 0))
                    return raw[min(index, len(raw) - 1)]
            else:
                result = getattr(window, "fusion_result", None)
                if result is not None:
                    return result
                results = getattr(window, "fusion_results", None) or []
                if results:
                    return results[0]
        return np.full((260, 520, 3), 128, dtype=np.uint8)

    def _refresh_preview(self, *_args):
        image = self._preview_image()
        px_um = self.effective_px_um()
        height, width = image.shape[:2]
        scale = 520.0 / width if width > 520 else 1.0
        small = (cv2.resize(image, (int(width * scale), int(height * scale)),
                            interpolation=cv2.INTER_AREA)
                 if scale < 1.0 else image)
        factor = scale if scale < 1.0 else 1.0
        requested = float(self.size_spin.value() or 0.0)
        font_note = ""
        if px_um:
            drawn, info = scalebar.draw_scale_bar_ex(
                small, px_um / (factor if factor > 0 else 1.0),
                self.pos_combo.currentData(), self.color_combo.currentData(),
                font_family=self._font_family(),
                font_size=requested * factor if requested > 0 else 0.0,
                font_bold=self.bold_chk.isChecked(),
                text_align=self.align_combo.currentData(),
                custom_x=self.custom_x.value() / 100.0,
                custom_y=self.custom_y.value() / 100.0)
            small = drawn
            if info.get("substituted_font"):
                font_note = trans.t("label_font_substituted").format(
                    name=info["substituted_font"])
            if info.get("too_small"):
                self._error.setText(trans.t("scalebar_too_small"))
            else:
                self._error.setText("")
        else:
            self._error.setText(trans.t("scalebar_not_calibrated"))
        self.font_note.setText(font_note)
        if self._ok_button is not None:
            self._ok_button.setEnabled(px_um is not None)

        self.effective_lbl.setText(self._effective_text())
        if px_um:
            # width is the WORKING width, and px_um is already per working
            # pixel, so the preview length matches what an export produces.
            bar_um = scalebar.nice_bar_um(width * px_um * scalebar.BAR_FRACTION)
            self.preview_meta.setText(trans.t("scalebar_preview_meta").format(
                length=scalebar.format_length(bar_um),
                px=int(round(bar_um / px_um)) if px_um else 0))
        else:
            self.preview_meta.setText("")

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                        3 * rgb.shape[1], QImage.Format.Format_RGB888)
        self.preview_lbl.setPixmap(QPixmap.fromImage(qimage).scaled(
            self.preview_lbl.width() or 480,
            self.preview_lbl.height() or 260,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.isVisible():
            self._refresh_preview()

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------
    def _save_and_accept(self):
        parent = self.parent()
        if parent is None:
            self.accept()
            return
        effective = self.effective_px_um()
        if effective is None:
            self._error.setText(trans.t("scalebar_not_calibrated"))
            return
        previous = {
            "enabled": bool(getattr(parent, "scale_bar_enabled", False)),
            "manual": float(getattr(parent, "scale_um_per_px_manual", 0.0)
                            or 0.0),
            "position": getattr(parent, "scale_bar_position", "bottom-right"),
            "color": getattr(parent, "scale_bar_color", "auto"),
            "font_family": getattr(parent, "scale_bar_font_family", ""),
            "font_size": getattr(parent, "scale_bar_font_size", 0.0),
            "font_bold": getattr(parent, "scale_bar_font_bold", False),
            "text_align": getattr(parent, "scale_bar_text_align", "auto"),
            "target": getattr(parent, "scale_bar_target", "all"),
            "custom_x": getattr(parent, "scale_bar_custom_x", 0.95),
            "custom_y": getattr(parent, "scale_bar_custom_y", 0.95),
        }

        parent.scale_bar_enabled = True
        # Store the manual value in ORIGINAL pixel units; the drawing path
        # converts it with the current working scale.
        parent.scale_um_per_px_manual = (
            self.manual_value()
            if self.calibration_mode() == CALIBRATION_MANUAL else 0.0)
        parent.scale_bar_position = self.pos_combo.currentData()
        parent.scale_bar_color = self.color_combo.currentData()
        parent.scale_bar_font_family = self._font_family()
        parent.scale_bar_font_size = float(self.size_spin.value() or 0.0)
        parent.scale_bar_font_bold = self.bold_chk.isChecked()
        parent.scale_bar_text_align = self.align_combo.currentData()
        parent.scale_bar_target = self.target_combo.currentData()
        parent.scale_bar_custom_x = self.custom_x.value() / 100.0
        parent.scale_bar_custom_y = self.custom_y.value() / 100.0

        if hasattr(parent, "persist_settings"):
            parent.persist_settings()
        if hasattr(parent, "push_scalebar_undo"):
            parent.push_scalebar_undo(previous)
        elif hasattr(parent, "_mark_project_dirty"):
            parent._mark_project_dirty()
        try:
            from utils import recovery
            recovery.write_snapshot(parent)
        except Exception:
            pass
        if hasattr(parent, "refresh_annotation_views"):
            parent.refresh_annotation_views()
        elif getattr(parent, "current_display_index", None) is not None:
            parent.update_source_view(parent.current_display_index)
        self.accept()
