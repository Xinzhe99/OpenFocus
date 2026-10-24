"""Scale bar settings dialog with a live preview thumbnail.

The dialog edits a local draft. Previously the preview wrote the manual
calibration straight into the main window, so pressing Cancel kept the new
value and later exports used it. It also had no explicit "metadata vs manual"
choice: a manual value of 0 silently meant "use metadata", which is not
discoverable and contradicted the localized hint.
"""
from __future__ import annotations

from typing import Optional
import math
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QFontComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import cv2
import numpy as np

from locales import trans
from utils import scalebar
from widgets.stepper import StepperSpinBox
from utils.image_utils import cv2_to_pixmap, to_display_uint8
from widgets.magnifier_label import MagnifierLabel

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

        self.header = QLabel(self._header_text())
        self.header.setObjectName("mutedLabel")
        self.header.setWordWrap(True)
        root.addWidget(self.header)

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
        self.target_combo.addItem(trans.t("label_target_registered"), "registered")
        self.target_combo.setCurrentIndex(max(0, self.target_combo.findData(
            getattr(window, "scale_bar_target", "all"))))
        form.insertRow(0, trans.t("scalebar_target"), self.target_combo)
        self.output_combo = QComboBox()
        manager = getattr(window, "output_manager", None)
        outputs = getattr(window, "output_list", None)
        if manager is not None and outputs is not None:
            for index in range(outputs.count()):
                record = manager.record_for_row(index)
                if record is not None and record.kind != "aligned":
                    self.output_combo.addItem(record.label, record)
            current = manager.record_for_item(outputs.currentItem())
            for index in range(self.output_combo.count()):
                if self.output_combo.itemData(index) is current:
                    self.output_combo.setCurrentIndex(index)
                    break
        form.insertRow(1, trans.t("label_target_output"), self.output_combo)
        self._target_form = form

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

        self.preview_lbl = MagnifierLabel(parent=self)
        self.preview_lbl.setMinimumSize(280, 240)
        self.preview_lbl.setSizePolicy(QSizePolicy.Policy.Ignored,
                                      QSizePolicy.Policy.Ignored)
        self.preview_lbl.setObjectName("labelPreview")
        preview_col.addWidget(self.preview_lbl, 1)
        navigation = QHBoxLayout()
        self.frame_index = QSpinBox()
        self.frame_index.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.frame_index.setRange(1, 1)
        self.frame_index.setValue(max(1, int(getattr(window, "current_display_index", 0) or 0) + 1))
        self.prev_button = QPushButton("‹")
        self.prev_button.setToolTip(trans.t("label_preview_prev"))
        self.next_button = QPushButton("›")
        self.next_button.setToolTip(trans.t("label_preview_next"))
        self.prev_button.clicked.connect(lambda: self.frame_index.stepBy(-1))
        self.next_button.clicked.connect(lambda: self.frame_index.stepBy(1))
        navigation.addWidget(self.prev_button)
        navigation.addWidget(self.frame_index)
        navigation.addWidget(self.next_button)
        fit = QPushButton(trans.t("retouch_fit_short").replace(" (F)", "").replace("（F）", ""))
        fit.clicked.connect(self.preview_lbl.reset_view)
        actual = QPushButton(trans.t("retouch_zoom_100"))
        actual.clicked.connect(self._actual_size)
        navigation.addWidget(fit)
        navigation.addWidget(actual)
        preview_col.addLayout(navigation)

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
        self.frame_index.valueChanged.connect(self._refresh_preview)
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        self.output_combo.currentIndexChanged.connect(self._target_changed)
        self._target_changed()
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
            if not math.isfinite(manual) or manual <= 0:
                return None
            scale = self._work_scale if self._work_scale > 0 else 1.0
            return manual / scale
        detected = self._detected
        return float(detected) if detected and math.isfinite(detected) and detected > 0 else None

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
    def _target_record(self):
        if self.target_combo.currentData() in ("all", "output"):
            return self.output_combo.currentData()
        return None

    def _target_changed(self, *_args):
        record = self._target_record()
        context = record.context if record is not None else None
        self._detected = (context.source_px_um if context is not None
                          else getattr(self.parent(), "source_px_um", None)
                          if record is None else None)
        self._work_scale = (context.source_scale_factor if context is not None
                            else 1.0 if record is not None
                            else float(getattr(self.parent(), "current_scale_factor", 1.0) or 1.0))
        if record is not None and record.scale_bar_config:
            cfg = record.scale_bar_config
            controls = (self.mode_combo, self.px_spin, self.pos_combo,
                        self.color_combo, self.font_combo, self.size_spin,
                        self.bold_chk, self.align_combo, self.custom_x, self.custom_y)
            blockers = [QSignalBlocker(control) for control in controls]
            self.mode_combo.setCurrentIndex(self.mode_combo.findData(cfg.get("mode", CALIBRATION_MANUAL)))
            self.px_spin.setValue(cfg.get("manual", cfg["px_um"] * self._work_scale))
            self.pos_combo.setCurrentIndex(self.pos_combo.findData(cfg["position"]))
            self.color_combo.setCurrentIndex(self.color_combo.findData(cfg["color"]))
            self.font_combo.setCurrentFont(QFont(cfg.get("font_family") or self.font().family()))
            self.size_spin.setValue(cfg.get("font_size", 0.0))
            self.bold_chk.setChecked(cfg.get("font_bold", False))
            self.align_combo.setCurrentIndex(self.align_combo.findData(cfg.get("text_align", "auto")))
            self.custom_x.setValue(int(cfg.get("custom_x", 0.95) * 100))
            self.custom_y.setValue(int(cfg.get("custom_y", 0.95) * 100))
            del blockers
        self._target_form.setRowVisible(
            self.output_combo, self.target_combo.currentData() in ("all", "output"))
        frames = getattr(self.parent(), "registration_results" if self.target_combo.currentData() == "registered" else "raw_images", ()) or ()
        self.frame_index.blockSignals(True)
        self.frame_index.setRange(1, max(1, len(frames)))
        self.frame_index.setValue(min(len(frames) or 1, int(getattr(self.parent(), "current_display_index", 0) or 0) + 1))
        self.frame_index.blockSignals(False)
        self.detect_lbl.setText(trans.t("scalebar_detected").format(value=f"{self._detected:g}")
                                if self._detected else trans.t("scalebar_no_metadata"))
        self.preview_lbl.reset_view()
        self._refresh_preview()

    def _preview_image(self):
        target = self.target_combo.currentData()
        record = self._target_record()
        if record is not None:
            return record.image
        if target == "output":
            return None
        attr = "registration_results" if target == "registered" else "raw_images"
        frames = getattr(self.parent(), attr, None) or []
        self.frame_index.blockSignals(True)
        self.frame_index.setRange(1, max(1, len(frames)))
        self.frame_index.blockSignals(False)
        return frames[self.frame_index.value() - 1] if frames else None

    def _draw_config(self):
        return dict(px_um=self.effective_px_um(),
                    position=self.pos_combo.currentData(),
                    color=self.color_combo.currentData(),
                    font_family=self._font_family(),
                    font_size=float(self.size_spin.value()),
                    font_bold=self.bold_chk.isChecked(),
                    text_align=self.align_combo.currentData(),
                    custom_x=self.custom_x.value() / 100.0,
                    custom_y=self.custom_y.value() / 100.0)

    def _refresh_preview(self, *_args):
        image = self._preview_image()
        px_um = self.effective_px_um()
        self.px_spin.setEnabled(self.calibration_mode() == CALIBRATION_MANUAL)
        has_stack = self._target_record() is None and self.target_combo.currentData() != "output"
        for control in (self.frame_index, self.prev_button, self.next_button):
            control.setVisible(has_stack)
        self.prev_button.setEnabled(self.frame_index.value() > 1)
        self.next_button.setEnabled(self.frame_index.value() < self.frame_index.maximum())
        self.effective_lbl.setText(self._effective_text())
        self.font_note.clear()
        self.preview_meta.clear()
        self._ok_button.setEnabled(False)
        if image is None:
            self.preview_lbl.clear()
            self._error.setText(trans.t("label_preview_no_target"))
            return
        height, width = image.shape[:2]
        name = self._target_record().label if self._target_record() else self.target_combo.currentText()
        self.header.setText(name + " · " + trans.t("scalebar_header").format(
            frames=len(getattr(self.parent(), "raw_images", ()) or ()),
            outputs=self.output_combo.count(), width=width, height=height,
            depth=image.dtype.itemsize * 8, scale=f"{self._work_scale * 100:g}"))
        drawn = image
        if px_um is None:
            self._error.setText(trans.t("scalebar_not_calibrated"))
        else:
            drawn, info = scalebar.draw_scale_bar_ex(image, **self._draw_config())
            valid = not info["too_small"] and info["bar_px"] is not None
            self._ok_button.setEnabled(valid)
            self._error.setText("" if valid else trans.t("scalebar_too_small"))
            if info["substituted_font"]:
                self.font_note.setText(trans.t("label_font_substituted").format(name=info["substituted_font"]))
            if valid:
                self.preview_meta.setText(trans.t("scalebar_preview_meta").format(
                    length=scalebar.format_length(info["bar_um"]), px=info["bar_px"]))
        self.preview_lbl.set_display_pixmap(cv2_to_pixmap(to_display_uint8(drawn)))

    def _actual_size(self):
        pixmap = self.preview_lbl._base_pixmap
        if pixmap is None:
            return
        fit = min(self.preview_lbl.width() / pixmap.width(),
                  self.preview_lbl.height() / pixmap.height())
        self.preview_lbl._zoom_factor = 1.0 / max(fit, 1e-9)
        self.preview_lbl.refresh_display()

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------
    def _save_and_accept(self):
        parent = self.parent()
        if parent is None:
            self.accept()
            return
        self._refresh_preview()
        if not self._ok_button.isEnabled():
            return
        effective = self.effective_px_um()
        if effective is None:
            self._error.setText(trans.t("scalebar_not_calibrated"))
            return
        previous = scalebar.capture_state(parent)

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
        manager = getattr(parent, "output_manager", None)
        if manager is not None and self.target_combo.currentData() in ("all", "output"):
            record = self._target_record()
            if record is not None:
                record.scale_bar_config = self._draw_config()
                record.scale_bar_config.update(mode=self.calibration_mode(),
                                                manual=self.manual_value())
                record.scale_bar_initialized = True

        try:
            if hasattr(parent, "persist_settings"):
                parent.persist_settings()
            if hasattr(parent, "push_scalebar_undo"):
                parent.push_scalebar_undo(previous)
            elif hasattr(parent, "_mark_project_dirty"):
                parent._mark_project_dirty()
        except Exception:
            scalebar.restore_state(parent, previous)
            self._error.setText(trans.t("msg_save_failed_text"))
            return
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
