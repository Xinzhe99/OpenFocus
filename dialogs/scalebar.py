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
from PyQt6.QtCore import QSignalBlocker, QThread, QTimer, Qt
from PyQt6.QtGui import QFont
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
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import cv2
import numpy as np

from locales import trans
from utils import scalebar
from widgets.stepper import StepperSpinBox, hide_spin_buttons
from utils.image_utils import cv2_to_pixmap, to_display_uint8
from utils.ui_utils import size_dialog_to_content
from widgets.magnifier_label import MagnifierLabel

CALIBRATION_METADATA = "metadata"
CALIBRATION_MANUAL = "manual"

class _PreviewWorker(QThread):
    """Resolve and develop one bounded preview without touching Qt widgets."""
    def __init__(self, frame, metadata, key):
        super().__init__()
        from utils.platform_utils import configure_native_worker_stack
        configure_native_worker_stack(self)
        self.frame, self.metadata, self.key = frame, metadata, key
        self.pixels = None
        self.error = None

    def run(self):
        try:
            from core.resource_policy import heavy_job
            from core.source_previews import render_source_region
            from core.source_frames import ensure_preview
            with heavy_job(self.isInterruptionRequested):
                pixels = ensure_preview(self.frame, should_cancel=self.isInterruptionRequested)
                if self.isInterruptionRequested():
                    return
                height, width = pixels.shape[:2]
                scale = min(1.0, 1200.0 / max(width, height))
                metadata = (self.metadata if not getattr(self.frame, "preview_is_display", False)
                            and pixels.dtype == np.uint16 and pixels.ndim == 3
                            and pixels.shape[2] == 3 else None)
                self.pixels, _ = render_source_region(
                    pixels, (0, 0, width, height),
                    (max(1, round(width * scale) - 2), max(1, round(height * scale) - 2)), metadata)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.frame = None
            self.metadata = None


class ScaleBarDialog(QDialog):
    """Draft editor for calibration, placement, style and target."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("dialog_scalebar_title"))
        self._apply_style()

        window = parent
        self._detected = getattr(window, "source_px_um", None)
        self._work_scale = float(
            getattr(window, "current_scale_factor", 1.0) or 1.0)
        stored_manual = float(
            getattr(window, "scale_um_per_px_manual", 0.0) or 0.0)
        self._pixel_worker = None
        self._pixel_key = None
        self._preview_cache = None
        self._region_worker = None
        self._region_pending = {}
        self._preview_revision = 0
        self._closing = False
        self._accept_pending = False
        self._displayed_record = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(30)
        self._preview_timer.timeout.connect(self._refresh_preview)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        content_col = QVBoxLayout(content)
        content_col.setContentsMargins(0, 0, 0, 0)
        content_col.setSpacing(8)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        self.header = QLabel(self._header_text())
        self.header.setObjectName("mutedLabel")
        self.header.setWordWrap(True)
        content_col.addWidget(self.header)

        body = QHBoxLayout()
        body.setSpacing(14)
        content_col.addLayout(body, 1)

        # ---- left: form -------------------------------------------------
        form_host = QWidget()
        form_col = QVBoxLayout(form_host)
        form_col.setContentsMargins(0, 0, 8, 0)
        form_col.setSpacing(8)

        form_col.addWidget(self._section(trans.t("scalebar_section_calibration")))
        form = QFormLayout()
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
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
        place_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        place_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
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
        text_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        text_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
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
        self.size_spin.setToolTip(trans.t("scalebar_font_auto"))
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

        self.output_combo = QComboBox()
        manager = getattr(window, "output_manager", None)
        outputs = getattr(window, "output_list", None)
        if manager is not None and outputs is not None:
            for index in range(outputs.count()):
                record = manager.record_for_row(index)
                if record is not None and record.kind != "aligned":
                    suffix = (" · " + trans.t("scalebar_baked_marker")
                              if getattr(record, "annotations_baked", False) else "")
                    self.output_combo.addItem(record.label + suffix, record)
            current = manager.record_for_item(outputs.currentItem())
            for index in range(self.output_combo.count()):
                if self.output_combo.itemData(index) is current:
                    self.output_combo.setCurrentIndex(index)
                    break

        # A target with no frames in this session used to leave the preview
        # blank and OK disabled with no explanation. Grey it out (the label
        # manager already does this for "registered") and fall back to a
        # target that can actually be previewed when the saved one cannot.
        registration = getattr(window, "registration_results", None) or []
        for code, missing in (("registered", not registration),
                              ("output", self.output_combo.count() == 0)):
            if missing:
                item = self.target_combo.model().item(
                    self.target_combo.findData(code))
                item.setEnabled(False)
                item.setToolTip(trans.t("label_preview_no_target"))

        def available(code: str) -> bool:
            index = self.target_combo.findData(code)
            return (index >= 0
                    and self.target_combo.model().item(index).isEnabled())

        stored = getattr(window, "scale_bar_target", "all") or "all"
        if not available(stored):
            order = (("source", "registered", "output", "all")
                     if (getattr(window, "raw_images", None) or [])
                     else ("registered", "output", "all", "source"))
            stored = next((code for code in order if available(code)), "all")
        self.target_combo.setCurrentIndex(
            max(0, self.target_combo.findData(stored)))

        form.insertRow(0, trans.t("scalebar_target"), self.target_combo)
        form.insertRow(1, trans.t("label_target_output"), self.output_combo)
        self._target_form = form

        self.font_note = QLabel("")
        self.font_note.setWordWrap(True)
        self.font_note.setObjectName("mutedLabel")
        text_form.addRow("", self.font_note)

        form_col.addStretch()
        body.addWidget(form_host, 1)

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
        hide_spin_buttons(self.frame_index)
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
            signal.connect(self._schedule_preview)
        self.frame_index.valueChanged.connect(self._schedule_preview)
        self.preview_lbl.sourceRegionRequested.connect(self._request_region)
        self.preview_lbl.sourceRegionsReleased.connect(self._release_regions)
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        self.output_combo.currentIndexChanged.connect(self._target_changed)
        self._drafts = {"stack": self._draft_values()}
        self._draft_owner = None
        self._target_changed()
        if (self.calibration_mode() == CALIBRATION_METADATA
                and self.effective_px_um() is None):
            self.mode_combo.setCurrentIndex(
                self.mode_combo.findData(CALIBRATION_MANUAL))
        self._refresh_preview()
        size_dialog_to_content(self, scroll)

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

    def _preview_metadata(self):
        """Linear calibration for the preview pixels, owned by their source.

        An output record must use its own metadata (or its render context),
        never the currently loaded stack: historical results may come from a
        different camera or may already have been developed. Source and
        registered previews belong to the current stack and use its metadata.
        """
        record = self._target_record()
        if record is not None:
            if getattr(record, "annotations_baked", False):
                return None
            metadata = getattr(record, "linear_dng_metadata", None)
            if metadata is None:
                context = getattr(record, "context", None)
                metadata = getattr(context, "linear_dng_metadata", None)
            return metadata
        if self.target_combo.currentData() == "output":
            return None
        frame = self._preview_image()
        metadata = getattr(frame, "linear_metadata", None)
        return metadata if metadata is not None else getattr(self.parent(), "linear_dng_metadata", None)

    @staticmethod
    def _develop_preview(image, metadata):
        """Develop camera-linear uint16 pixels once; ordinary images pass through."""
        if (metadata is not None and image is not None
                and image.dtype == np.uint16 and image.ndim == 3
                and image.shape[2] == 3):
            from utils.linear_dng import develop_linear
            return develop_linear(image, metadata)
        return image

    def _draft_values(self):
        values = self._draw_config()
        values.update(mode=self.calibration_mode(), manual=self.manual_value(),
                      detected=self._detected, work_scale=self._work_scale)
        return values

    def _target_changed(self, *_args):
        if self._draft_owner is not None:
            self._drafts[self._draft_owner] = self._draft_values()
        record = self._target_record()
        owner = ("output", record.output_id) if record is not None else "stack"
        if owner not in self._drafts:
            context = getattr(record, "context", None)
            detected = getattr(context, "source_px_um", None)
            work_scale = getattr(context, "source_scale_factor", 1.0) or 1.0
            cfg = dict(getattr(record, "scale_bar_config", None) or {})
            draft = dict(self._drafts["stack"])
            draft.update(cfg)
            draft["detected"] = detected
            draft["work_scale"] = work_scale
            draft["manual"] = cfg.get("manual", (cfg.get("px_um") or 0.0) * work_scale)
            draft["mode"] = cfg.get("mode", CALIBRATION_MANUAL if draft["manual"] else CALIBRATION_METADATA)
            self._drafts[owner] = draft
        self._draft_owner = owner
        cfg = self._drafts[owner]
        self._detected = cfg["detected"]
        self._work_scale = cfg["work_scale"]
        controls = (self.mode_combo, self.px_spin, self.pos_combo,
                    self.color_combo, self.font_combo, self.size_spin,
                    self.bold_chk, self.align_combo, self.custom_x, self.custom_y)
        blockers = [QSignalBlocker(control) for control in controls]
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(cfg["mode"]))
        self._adapt_px_decimals(cfg["manual"])
        self.px_spin.setValue(cfg["manual"])
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
            self.output_combo,
            self.output_combo.count() > 0
            and self.target_combo.currentData() in ("all", "output"))
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

    def _stack_draft_valid(self):
        frames = getattr(self.parent(), "raw_images", ()) or ()
        if not frames:
            return False
        draft = self._draft_values() if self._draft_owner == "stack" else self._drafts["stack"]
        if draft["mode"] == CALIBRATION_MANUAL:
            px_um = draft["manual"] / (draft["work_scale"] or 1.0)
        else:
            px_um = draft["detected"]
        if not px_um or not math.isfinite(px_um) or px_um <= 0:
            return False
        height, width = frames[0].shape[:2]
        cfg = self._draw_config()
        cfg["px_um"] = px_um
        _, info = scalebar.draw_scale_bar_ex(
            np.zeros((1, 1, 3), dtype=np.uint8), **cfg,
            canvas_size=(width, height), region_mean=lambda _rect: 128.0)
        return not info["too_small"] and info["bar_px"] is not None

    def _schedule_preview(self, *_args):
        if self._closing:
            return
        self._preview_revision += 1
        if self._region_worker is not None:
            self._region_worker.request_cancel()
        self._region_pending.clear()
        self._ok_button.setEnabled(False)
        self._preview_timer.start()

    def _start_pixels(self, image, key):
        self._pixel_key = key
        if self._pixel_worker is not None:
            if self._pixel_worker.key != key:
                self._pixel_worker.requestInterruption()
            return
        worker = _PreviewWorker(image, self._preview_metadata(), key)
        self._pixel_worker = worker
        def finished():
            self._pixel_worker = None
            if not self._closing and worker.key == self._pixel_key:
                if worker.error:
                    self._preview_cache = (worker.key, None, worker.error)
                elif worker.pixels is not None:
                    self._preview_cache = (worker.key, worker.pixels, None)
            worker.deleteLater()
            if self._closing:
                self._finish_close()
            else:
                self._schedule_preview()
        worker.finished.connect(finished)
        worker.start()

    def _release_regions(self, key):
        if self._region_worker is not None and self._region_worker.key == key:
            self._region_worker.request_cancel()
        self._region_pending.clear()

    def _request_region(self, key, request):
        if self._closing or key != self.preview_lbl._source_key:
            return
        if request[0] == "view" and self.preview_lbl._zoom_factor <= 1.0:
            return
        worker = self._region_worker
        if worker is not None:
            if worker.key == key and not worker.cancelled:
                worker.submit(request)
            else:
                self._region_pending[request[0]] = (key, request)
            return
        image = self._preview_image()
        if image is None or id(image) != key[1]:
            return
        from core.source_previews import SourceRegionWorker
        record = self._target_record()
        baked = bool(record is not None and getattr(record, "annotations_baked", False))
        cfg = None if baked else self._draw_config()
        worker = SourceRegionWorker(image, key, scale_cfg=cfg,
                                    metadata=self._preview_metadata(), develop=not baked)
        self._region_worker = worker
        def ready(source_key, completed, pixels, rect):
            if (not self._closing and pixels is not None
                    and source_key[2] == self._preview_revision):
                self.preview_lbl.set_source_region(cv2_to_pixmap(pixels), rect,
                                                   source_key=source_key, request=completed)
        def failed(source_key, message):
            if (not self._closing and source_key == self.preview_lbl._source_key
                    and source_key[2] == self._preview_revision):
                self._ok_button.setEnabled(False)
                self._error.setText(message)
        def finished():
            self._region_worker = None
            worker.deleteLater()
            if self._closing:
                self._finish_close()
            else:
                pending = list(self._region_pending.values())
                self._region_pending.clear()
                for source_key, item in pending:
                    self._request_region(source_key, item)
        worker.ready.connect(ready)
        worker.failed.connect(failed)
        worker.finished.connect(finished)
        worker.submit(request)
        worker.start()

    def shutdown_running(self):
        self._closing = True
        self._preview_timer.stop()
        self._pixel_key = None
        self._region_pending.clear()
        if self._pixel_worker is not None:
            self._pixel_worker.requestInterruption()
        if self._region_worker is not None:
            self._region_worker.request_cancel()
        return self._pixel_worker is not None or self._region_worker is not None

    def _finish_close(self):
        if self._closing and self._pixel_worker is None and self._region_worker is None:
            if self._accept_pending:
                super().accept()
            else:
                super().reject()

    def reject(self):
        if not self.shutdown_running():
            super().reject()

    def accept(self):
        self._accept_pending = True
        if not self.shutdown_running():
            super().accept()

    def closeEvent(self, event):
        if self.shutdown_running():
            event.ignore()
        else:
            super().closeEvent(event)

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
        key = (self.target_combo.currentData(), id(image),
               getattr(self._target_record(), "output_id", None), id(self._preview_metadata()))
        if self._preview_cache is None or self._preview_cache[0] != key:
            self._start_pixels(image, key)
            return
        _, display, error = self._preview_cache
        if error is not None:
            self._error.setText(error)
            return
        if len(image.shape) == 3 and image.shape[2] == 4:
            self._error.setText(trans.t('transparent_desktop_unsupported').format(path=''))
            return
        native_image = image
        image = display
        name = self._target_record().label if self._target_record() else self.target_combo.currentText()
        self.header.setText(name + " · " + trans.t("scalebar_header").format(
            frames=len(getattr(self.parent(), "raw_images", ()) or ()),
            outputs=self.output_combo.count(), width=width, height=height,
            depth=native_image.dtype.itemsize * 8, scale=f"{self._work_scale * 100:g}"))
        record = self._target_record()
        self._displayed_record = (record, bool(record is not None and
                                                getattr(record, "annotations_baked", False)))
        if record is not None and getattr(record, "annotations_baked", False):
            pixmap = cv2_to_pixmap(image)
            if pixmap.isNull():
                self._error.setText(trans.t("msg_save_failed_text"))
                return
            self.preview_lbl.set_display_pixmap(pixmap, source_size=(width, height),
                source_key=(self.target_combo.currentData(), id(native_image), self._preview_revision))
            self._error.setText(trans.t("scalebar_baked_notice"))
            if self.target_combo.currentData() == "all":
                try:
                    self._ok_button.setEnabled(self._stack_draft_valid())
                except Exception as exc:
                    self._error.setText(str(exc))
            return
        drawn = image
        if px_um is None:
            self._error.setText(trans.t("scalebar_not_calibrated"))
        else:
            scale = image.shape[1] / width
            def preview_mean(rect):
                x, y, w, h = rect
                return scalebar._region_mean(image, round(x * scale), round(y * scale),
                                             round((x + w) * scale), round((y + h) * scale))
            try:
                drawn, info = scalebar.draw_scale_bar_ex(
                    image, **self._draw_config(), canvas_size=(width, height),
                    render_scale=scale, region_mean=preview_mean)
            except Exception as exc:
                self._error.setText(str(exc))
                return
            valid = not info["too_small"] and info["bar_px"] is not None
            if valid and self.target_combo.currentData() == "all":
                try:
                    valid = self._stack_draft_valid()
                except Exception as exc:
                    self._error.setText(str(exc))
                    return
            self._ok_button.setEnabled(valid)
            self._error.setText("" if valid else trans.t("scalebar_too_small"))
            if info["substituted_font"]:
                self.font_note.setText(trans.t("label_font_substituted").format(name=info["substituted_font"]))
            if valid:
                self.preview_meta.setText(trans.t("scalebar_preview_meta").format(
                    length=scalebar.format_length(info["bar_um"]), px=info["bar_px"]))
        pixmap = cv2_to_pixmap(drawn)
        if pixmap.isNull():
            self._ok_button.setEnabled(False)
            self._error.setText(trans.t("msg_save_failed_text"))
            return
        self.preview_lbl.set_display_pixmap(pixmap, source_size=(width, height),
            source_key=(self.target_combo.currentData(), id(native_image), self._preview_revision))

    def _actual_size(self):
        pixmap = self.preview_lbl._base_pixmap
        if pixmap is None:
            return
        source = self.preview_lbl._source_size
        fit = min(self.preview_lbl.width() / source.width(),
                  self.preview_lbl.height() / source.height())
        self.preview_lbl._zoom_factor = 1.0 / max(fit * self.preview_lbl.devicePixelRatioF(), 1e-9)
        self.preview_lbl.refresh_display()

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------
    def _save_and_accept(self):
        parent = self.parent()
        if parent is None:
            self.accept()
            return
        current = self._target_record()
        if (self._displayed_record is not None and current is self._displayed_record[0]
                and not self._displayed_record[1] and current is not None
                and getattr(current, "annotations_baked", False)):
            self._error.setText(trans.t("scalebar_baked_notice"))
            return
        self._refresh_preview()
        if not self._ok_button.isEnabled():
            return
        target = self.target_combo.currentData()
        record = self._target_record()
        baked = bool(record is not None and getattr(record, "annotations_baked", False))
        if target == "output" and (record is None or baked):
            self._error.setText(trans.t("scalebar_baked_notice"))
            return
        if record is not None:
            manager = getattr(parent, "output_manager", None)
            outputs = getattr(parent, "output_list", None)
            if (manager is None or outputs is None or not any(
                    manager.record_for_row(row) is record for row in range(outputs.count()))):
                self._error.setText(trans.t("label_preview_no_target"))
                return
        self._drafts[self._draft_owner] = self._draft_values()
        stack = self._drafts["stack"]
        try:
            previous = scalebar.capture_state(parent)
            candidate = dict(previous)
            if "outputs" in previous:
                candidate["outputs"] = dict(previous["outputs"])
            if target != "output":
                manual = stack["manual"] if stack["mode"] == CALIBRATION_MANUAL else 0.0
                candidate.update(enabled=True, manual=manual,
                                 position=self.pos_combo.currentData(),
                                 color=self.color_combo.currentData(),
                                 font_family=self._font_family(),
                                 font_size=float(self.size_spin.value()),
                                 font_bold=self.bold_chk.isChecked(),
                                 text_align=self.align_combo.currentData(), target=target,
                                 custom_x=self.custom_x.value() / 100.0,
                                 custom_y=self.custom_y.value() / 100.0)
            if target in ("output", "all") and record is not None and not baked:
                cfg = self._draw_config()
                cfg.update(mode=self.calibration_mode(), manual=self.manual_value())
                candidate.setdefault("outputs", {})[record.output_id] = {"config": cfg, "initialized": True}
            if candidate == previous:
                self.accept()
                return
            prepared = (parent.prepare_scalebar_undo(previous)
                        if hasattr(parent, "prepare_scalebar_undo") else None)
            history = {name: list(getattr(parent, name)) for name in ("_undo_stack", "_redo_stack")
                       if hasattr(parent, name)}
            dirty = getattr(parent, "_project_dirty", None)
            published = True
            scalebar.restore_state(parent, candidate)
            if hasattr(parent, "persist_settings"):
                parent.persist_settings()
            if hasattr(parent, "push_scalebar_undo"):
                if hasattr(parent, "prepare_scalebar_undo"):
                    parent.push_scalebar_undo(previous, prepared=prepared)
                else:
                    parent.push_scalebar_undo(previous)
            elif hasattr(parent, "_mark_project_dirty"):
                parent._mark_project_dirty()
        except Exception as exc:
            if locals().get('published', False):
                scalebar.restore_state(parent, previous)
            for name, values in locals().get('history', {}).items():
                getattr(parent, name)[:] = values
            if 'dirty' in locals() and dirty is not None:
                parent._project_dirty = dirty
            if hasattr(parent, "_update_project_title"):
                parent._update_project_title()
            try:
                if locals().get('published', False) and hasattr(parent, "persist_settings"):
                    parent.persist_settings()
            except Exception as restore_exc:
                self._error.setText(f"{trans.t('msg_save_failed_text')} {exc}; {restore_exc}")
                return
            self._error.setText(f"{trans.t('msg_save_failed_text')} {exc}")
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
