"""Watermark editor: a local draft with a live, real-target preview.

Edits commit only on Apply (Cancel/Esc/window close leave every committed
configuration untouched).  The preview is a bounded display copy; exports
always render the watermark at the image's own size and bit depth.
"""
from __future__ import annotations

import cv2
import numpy as np

from PyQt6.QtCore import QSignalBlocker, Qt, QTimer
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFontComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from locales import trans
from utils import watermark
from utils.image_utils import cv2_to_pixmap, to_display_uint8
from utils.ui_utils import size_dialog_to_content
from widgets.magnifier_label import MagnifierLabel
from widgets.stepper import StepperSpinBox, hide_spin_buttons

PREVIEW_MAX_SIDE = 1200
PREVIEW_DEBOUNCE_MS = 25


class WatermarkDialog(QDialog):
    """Draft editor for the tiled text watermark."""

    def __init__(self, parent=None):
        super().__init__(parent)
        window = parent
        self.setWindowTitle(trans.t("dialog_watermark_title"))
        self.setStyleSheet("")
        self._window = window
        self._preview_cache = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self._apply_record = None
        self._initializing = True

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

        form_col.addWidget(self._section(trans.t("watermark_section_text")))
        text_form = QFormLayout()
        text_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        text_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form_col.addLayout(text_form)

        self.text_edit = QLineEdit()
        # QLineEdit counts UTF-16 units; the validator counts code points.
        self.text_edit.setMaxLength(watermark.MAX_TEXT_CHARS * 2)
        self.text_edit.setPlaceholderText(trans.t("watermark_text_placeholder"))
        self.text_edit.setClearButtonEnabled(True)
        self.text_edit.setAccessibleName(trans.t("watermark_text"))
        self.text_edit.setToolTip(trans.t("watermark_text_hint"))
        text_form.addRow(trans.t("watermark_text"), self.text_edit)

        self.copyright_check = QCheckBox(trans.t("watermark_copyright"))
        self.copyright_check.setAccessibleName(trans.t("watermark_copyright"))
        self.copyright_check.setToolTip(trans.t("watermark_copyright_hint"))
        text_form.addRow(self.copyright_check)

        self.copyright_position_combo = QComboBox()
        self.copyright_position_combo.addItem(trans.t("watermark_copyright_prefix"), "prefix")
        self.copyright_position_combo.addItem(trans.t("watermark_copyright_suffix"), "suffix")
        self.copyright_position_combo.setAccessibleName(trans.t("watermark_copyright_position"))
        self.copyright_position_combo.setToolTip(trans.t("watermark_copyright_position_hint"))
        text_form.addRow(trans.t("watermark_copyright_position"), self.copyright_position_combo)

        form_col.addWidget(self._section(trans.t("watermark_section_appearance")))
        appearance = QFormLayout()
        appearance.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        appearance.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form_col.addLayout(appearance)

        self.font_combo = QFontComboBox()
        self.font_combo.setAccessibleName(trans.t("watermark_font"))
        self.font_combo.setToolTip(trans.t("watermark_font_hint"))
        appearance.addRow(trans.t("watermark_font"), self.font_combo)

        self.size_row, self.size_slider, self.size_spin = self._slider_row(
            watermark.MIN_SIZE_PERCENT, watermark.MAX_SIZE_PERCENT,
            watermark.DEFAULT_SIZE_PERCENT, scale=10, decimals=1, suffix="%")
        self.size_slider.setAccessibleName(trans.t("watermark_size"))
        appearance.addRow(trans.t("watermark_size"), self.size_row)

        self.opacity_row, self.opacity_slider, self.opacity_spin = self._slider_row(
            int(watermark.MIN_OPACITY * 100), int(watermark.MAX_OPACITY * 100),
            int(round(watermark.DEFAULT_OPACITY * 100)), scale=1, decimals=0,
            suffix="%")
        self.opacity_slider.setAccessibleName(trans.t("watermark_opacity"))
        self.opacity_spin.setToolTip(trans.t("watermark_opacity_hint"))
        appearance.addRow(trans.t("watermark_opacity"), self.opacity_row)

        self.color_combo = QComboBox()
        self.color_combo.addItem(trans.t("watermark_color_white"), "white")
        self.color_combo.addItem(trans.t("watermark_color_black"), "black")
        self.color_combo.setAccessibleName(trans.t("watermark_color"))
        appearance.addRow(trans.t("watermark_color"), self.color_combo)

        self.angle_row, self.angle_slider, self.angle_spin = self._slider_row(
            int(watermark.MIN_ANGLE), int(watermark.MAX_ANGLE),
            int(watermark.DEFAULT_ANGLE), scale=1, decimals=0, suffix="°")
        self.angle_slider.setAccessibleName(trans.t("watermark_angle"))
        self.angle_spin.setToolTip(trans.t("watermark_angle_hint"))
        appearance.addRow(trans.t("watermark_angle"), self.angle_row)

        self.density_combo = QComboBox()
        for code, key in (("sparse", "watermark_density_sparse"),
                          ("standard", "watermark_density_standard"),
                          ("dense", "watermark_density_dense")):
            self.density_combo.addItem(trans.t(key), code)
        self.density_combo.setAccessibleName(trans.t("watermark_density"))
        self.density_combo.setToolTip(trans.t("watermark_density_hint"))
        appearance.addRow(trans.t("watermark_density"), self.density_combo)

        self.reset_button = QPushButton(trans.t("watermark_reset_defaults"))
        self.reset_button.clicked.connect(self._reset_defaults)
        form_col.addWidget(self.reset_button)

        form_col.addWidget(self._section(trans.t("watermark_section_targets")))
        self.target_checks = {}
        target_col = QVBoxLayout()
        target_col.setContentsMargins(0, 0, 0, 0)
        target_col.setSpacing(4)
        for target, key in (("source", "watermark_target_source"),
                            ("registered", "watermark_target_registered"),
                            ("output", "watermark_target_output")):
            check = QCheckBox(trans.t(key))
            check.setAccessibleName(trans.t(key))
            check.toggled.connect(self._schedule_preview)
            self.target_checks[target] = check
            target_col.addWidget(check)
        hint = QLabel(trans.t("watermark_target_hint"))
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        target_col.addWidget(hint)
        self.export_hint = QLabel(trans.t("watermark_export_hint"))
        self.export_hint.setObjectName("mutedLabel")
        self.export_hint.setWordWrap(True)
        target_col.addWidget(self.export_hint)
        form_col.addLayout(target_col)
        form_col.addStretch()
        body.addWidget(form_host, 1)

        # ---- right: preview ---------------------------------------------
        preview_col = QVBoxLayout()
        preview_col.setSpacing(6)
        title = QLabel(trans.t("watermark_preview"))
        bold = title.font()
        bold.setBold(True)
        title.setFont(bold)
        preview_col.addWidget(title)

        self.preview_target_combo = QComboBox()
        self.preview_target_combo.setAccessibleName(trans.t("watermark_preview_target"))
        self.preview_target_combo.addItem(trans.t("watermark_target_source"), "source")
        self.preview_target_combo.addItem(trans.t("watermark_target_registered"), "registered")
        self.preview_target_combo.addItem(trans.t("watermark_target_output"), "output")
        preview_col.addWidget(self.preview_target_combo)

        self.output_combo = QComboBox()
        self.output_combo.setAccessibleName(trans.t("watermark_target_output"))
        preview_col.addWidget(self.output_combo)

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
        self.preview_meta.setObjectName("mutedLabel")
        self.preview_meta.setWordWrap(True)
        preview_col.addWidget(self.preview_meta)
        self.font_note = QLabel("")
        self.font_note.setObjectName("mutedLabel")
        self.font_note.setWordWrap(True)
        preview_col.addWidget(self.font_note)
        self.baked_note = QLabel("")
        self.baked_note.setObjectName("mutedLabel")
        self.baked_note.setWordWrap(True)
        preview_col.addWidget(self.baked_note)
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

        # Fill output choices before choosing the initial preview target.
        self._rebuild_output_combo()
        manager = getattr(self._window, "output_manager", None)
        items = getattr(self._window, "output_list", None)
        self._apply_record = (manager.record_for_item(items.currentItem())
                              if manager is not None and items is not None and items.currentItem()
                              else None)
        self._populate_from_window()
        self._setup_preview_targets()

        for signal in (self.size_spin.valueChanged, self.opacity_spin.valueChanged,
                       self.angle_spin.valueChanged):
            signal.connect(self._schedule_preview)
        for signal in (self.size_slider.valueChanged, self.opacity_slider.valueChanged,
                       self.angle_slider.valueChanged):
            signal.connect(self._schedule_preview)
        self.text_edit.textEdited.connect(self._schedule_preview)
        self.copyright_check.toggled.connect(self.copyright_position_combo.setEnabled)
        self.copyright_check.toggled.connect(self._schedule_preview)
        self.copyright_position_combo.currentIndexChanged.connect(self._schedule_preview)
        self.font_combo.currentFontChanged.connect(self._schedule_preview)
        self.color_combo.currentIndexChanged.connect(self._schedule_preview)
        self.density_combo.currentIndexChanged.connect(self._schedule_preview)
        self.preview_target_combo.currentIndexChanged.connect(self._preview_target_changed)
        self.output_combo.currentIndexChanged.connect(self._preview_target_changed)
        self.frame_index.valueChanged.connect(self._schedule_preview)

        size_dialog_to_content(self, scroll)
        self._initializing = False
        self._preview_target_changed()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _section(self, text: str) -> QLabel:
        label = QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    def _slider_row(self, minimum, maximum, value, *, scale, decimals, suffix):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(minimum * scale), int(maximum * scale))
        spin = StepperSpinBox(minimum=float(minimum), maximum=float(maximum),
                              decimals=decimals, step=1.0 / scale)
        spin.setSuffix(suffix)
        # Reserve room for the widest numeric value, the suffix and the step
        # buttons; a fixed 88 px cropped "3.5%" into "3.59" on native Windows.
        metrics = QFontMetrics(spin.font())
        widest = max(metrics.horizontalAdvance(
            f"{value:.{decimals}f}{suffix}") for value in (minimum, maximum))
        spin.setMinimumWidth(max(112, widest + 72, spin.sizeHint().width()))
        slider.setValue(int(round(float(value) * scale)))
        spin.setValue(float(value))
        slider.valueChanged.connect(
            lambda raw: spin.setValue(raw / scale)
            if abs(spin.value() - raw / scale) > 1e-9 else None)
        spin.valueChanged.connect(
            lambda number: slider.setValue(int(round(float(number) * scale)))
            if slider.value() != int(round(float(number) * scale)) else None)
        layout.addWidget(slider, 1)
        layout.addWidget(spin)
        return row, slider, spin

    def _header_text(self) -> str:
        window = self._window
        frames = len(getattr(window, "raw_images", ()) or ())
        outputs = len(getattr(window, "fusion_results", ()) or ())
        if not frames and not outputs:
            return trans.t("watermark_header_empty")
        return trans.t("watermark_header").format(
            frames=frames, outputs=outputs)

    def _rebuild_output_combo(self) -> None:
        self.output_combo.blockSignals(True)
        self.output_combo.clear()
        manager = getattr(self._window, "output_manager", None)
        items = getattr(self._window, "output_list", None)
        selected = (manager.record_for_item(items.currentItem())
                    if manager is not None and items is not None and items.currentItem()
                    else None)
        if manager is not None and items is not None:
            for index in range(items.count()):
                record = manager.record_for_row(index)
                if (record is None or record.kind == "aligned"
                        or getattr(record, "image", None) is None):
                    continue
                suffix = f" · {trans.t('watermark_baked_marker')}" if record.watermark_baked else ""
                self.output_combo.addItem(record.label + suffix, record)
        chosen = -1
        if selected is not None:
            for index in range(self.output_combo.count()):
                if self.output_combo.itemData(index) is selected:
                    chosen = index
                    break
        self.output_combo.setCurrentIndex(chosen)
        self.output_combo.blockSignals(False)

    def _load_style(self, cfg: watermark.WatermarkConfig) -> None:
        """Populate the style controls from a committed configuration."""
        blockers = [QSignalBlocker(widget) for widget in (
            self.text_edit, self.copyright_check, self.copyright_position_combo,
            self.font_combo, self.size_slider, self.size_spin,
            self.opacity_slider, self.opacity_spin, self.color_combo,
            self.angle_slider, self.angle_spin, self.density_combo)]
        default_family = self.font().family()
        self.text_edit.setText(cfg.text)
        self.copyright_check.setChecked(cfg.copyright_enabled)
        self.copyright_position_combo.setCurrentIndex(
            max(0, self.copyright_position_combo.findData(cfg.copyright_position)))
        self.copyright_position_combo.setEnabled(cfg.copyright_enabled)
        self.font_combo.setCurrentFont(QFont(cfg.font_family or default_family))
        self.size_slider.setValue(int(round(cfg.size_percent * 10)))
        self.size_spin.setValue(cfg.size_percent)
        self.opacity_slider.setValue(int(round(cfg.opacity * 100)))
        self.opacity_spin.setValue(int(round(cfg.opacity * 100)))
        self.color_combo.setCurrentIndex(max(0, self.color_combo.findData(cfg.color)))
        self.angle_slider.setValue(int(round(cfg.angle)))
        self.angle_spin.setValue(cfg.angle)
        self.density_combo.setCurrentIndex(
            max(0, self.density_combo.findData(cfg.density)))
        del blockers

    def _populate_from_window(self) -> None:
        window = self._window
        targets = watermark.active_targets(window)
        cfg = (watermark.config_for_target(window, "source")
               or watermark.config_from_window(window))
        initial_record = self._apply_record
        if not getattr(window, "raw_images", ()) and initial_record is not None:
            record_cfg = watermark.safe_config(
                getattr(initial_record, "watermark_config", None))
            if getattr(initial_record, "watermark_initialized", False) and record_cfg:
                cfg = record_cfg
        if not targets:
            # First use (or everything was removed): the dialog defaults to
            # source + result, but nothing becomes active until Apply.
            targets = ("source", "output")
        if (self._apply_record is not None
                and getattr(self._apply_record, "kind", "") == "aligned"
                and "registered" not in targets):
            targets = tuple(targets) + ("registered",)
        if cfg is None:
            cfg = watermark.WatermarkConfig(text="", copyright_enabled=True)
        self._load_style(cfg)
        for target, check in self.target_checks.items():
            check.setChecked(target in targets)
        self._sync_target_availability()

    def _setup_preview_targets(self) -> None:
        window = self._window
        registration = getattr(window, "registration_results", ()) or ()
        outputs = self.output_combo.count()
        model = self.preview_target_combo.model()
        for code, missing in (("source", not getattr(window, "raw_images", ())),
                              ("registered", not registration),
                              ("output", outputs == 0)):
            index = self.preview_target_combo.findData(code)
            if index >= 0 and model is not None:
                item = model.item(index)
                if item is not None:
                    item.setEnabled(not missing)
                    if missing:
                        item.setToolTip(trans.t("watermark_preview_no_target"))
        # Prefer the current source frame, then a result, then registered
        # frames.  Switching the preview never changes the apply scope.
        order = ("source", "output", "registered")
        if not outputs:
            order = tuple(code for code in order if code != "output")
        if not getattr(window, "raw_images", ()):
            order = tuple(code for code in order if code != "source")
        if not registration:
            order = tuple(code for code in order if code != "registered")
        chosen = order[0] if order else "source"
        index = self.preview_target_combo.findData(chosen)
        self.preview_target_combo.setCurrentIndex(max(0, index))
        if chosen == "output" and self.output_combo.currentIndex() < 0:
            self.output_combo.setCurrentIndex(0)

    def _sync_preview_navigation(self, reset: bool = False) -> None:
        code = self.preview_target_combo.currentData()
        frames = (getattr(self._window, "registration_results", ()) or ()
                  if code == "registered"
                  else getattr(self._window, "raw_images", ()) or ())
        available = code in ("source", "registered") and bool(frames)
        for widget in (self.frame_index, self.prev_button, self.next_button):
            widget.setVisible(available)
        if available:
            if code == "registered":
                current = int(getattr(self._window, "current_result_index", 0) or 0)
            else:
                current = int(getattr(self._window, "current_display_index", 0) or 0)
            self.frame_index.blockSignals(True)
            self.frame_index.setRange(1, max(1, len(frames)))
            if reset or not 1 <= self.frame_index.value() <= len(frames):
                self.frame_index.setValue(min(len(frames), current + 1))
            self.frame_index.blockSignals(False)
            self.prev_button.setEnabled(self.frame_index.value() > 1)
            self.next_button.setEnabled(self.frame_index.value() < self.frame_index.maximum())
        self.output_combo.setVisible(code == "output")

    def _sync_target_availability(self) -> None:
        window = self._window
        record = self._preview_record() if self.preview_target_combo.currentData() == "output" else None
        baked = bool(record is not None and getattr(record, "watermark_baked", False))
        if (not baked and self.target_checks["output"].isChecked()
                and self._apply_record is not None
                and getattr(self._apply_record, "watermark_baked", False)):
            baked = True
        self.target_checks["source"].setEnabled(bool(getattr(window, "raw_images", ())))
        has_registered = bool(getattr(window, "registration_results", ()))
        self.target_checks["registered"].setEnabled(has_registered)
        # Output stays available even before the first render: the user may
        # pre-select the scope for results produced later.  A baked preview
        # is skipped by the commit path instead of silently changing scope.
        self.baked_note.setText(trans.t("watermark_baked_notice") if baked else "")
        self.baked_note.setVisible(baked)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _schedule_preview(self, *_args) -> None:
        if self._initializing:
            return
        self._preview_timer.start()

    def _preview_target_changed(self, *_args) -> None:
        self._preview_cache = None
        if self.preview_target_combo.currentData() == "output":
            if self.output_combo.currentIndex() < 0 and self.output_combo.count():
                self.output_combo.setCurrentIndex(0)
            record = self.output_combo.currentData()
            record_cfg = watermark.safe_config(
                getattr(record, "watermark_config", None))
            if (record is not None
                    and getattr(record, "watermark_initialized", False)
                    and record_cfg):
                self._load_style(record_cfg)
        self._sync_preview_navigation(reset=True)
        self._sync_target_availability()
        self._refresh_preview()

    def _preview_record(self):
        if self.preview_target_combo.currentData() != "output":
            return None
        record = self.output_combo.currentData()
        return record if record is not None else None

    def _preview_image_and_canvas(self):
        code = self.preview_target_combo.currentData()
        window = self._window
        record = self._preview_record()
        metadata = None
        name = ""
        frame_label = ""
        if record is not None:
            image = getattr(record, "image", None)
            if image is None:
                return None
            key = ("output", record.output_id, id(image))
            if self._preview_cache is not None and self._preview_cache[0] == key:
                return self._preview_cache[1]
            metadata = getattr(record, "linear_dng_metadata", None)
            name = getattr(record, "label", "")
            depth = image.dtype.itemsize * 8
            canvas = tuple(image.shape[:2])
        elif code == "registered":
            frames = getattr(window, "registration_results", ()) or ()
            index = self.frame_index.value() - 1
            if not 0 <= index < len(frames):
                return None
            image = frames[index]
            key = ("registered", index, id(image))
            if self._preview_cache is not None and self._preview_cache[0] == key:
                return self._preview_cache[1]
            metadata = getattr(window, "linear_dng_metadata", None)
            name = trans.t("watermark_target_registered")
            frame_label = trans.t("watermark_frame_label").format(
                index=index + 1, total=len(frames))
            depth = image.dtype.itemsize * 8
            canvas = tuple(image.shape[:2])
        else:
            frames = getattr(window, "raw_images", ()) or ()
            index = self.frame_index.value() - 1
            if not 0 <= index < len(frames):
                return None
            image = frames[index]
            key = ("source", index, id(image))
            if self._preview_cache is not None and self._preview_cache[0] == key:
                return self._preview_cache[1]
            metadata = getattr(image, "linear_metadata", None) or getattr(
                window, "linear_dng_metadata", None)
            name = trans.t("watermark_target_source")
            frame_label = trans.t("watermark_frame_label").format(
                index=index + 1, total=len(frames))
            depth = image.dtype.itemsize * 8
            canvas = tuple(image.shape[:2])

        if metadata is not None and not getattr(image, "preview_is_display", False):
            from utils.frame_pixels import preview_pixels
            display = preview_pixels(image, parent=self)
            from utils.linear_dng import develop_linear
            display = develop_linear(display, metadata)
        else:
            from utils.frame_pixels import preview_pixels
            display = preview_pixels(image, parent=self)
        display = np.array(to_display_uint8(display), copy=True)
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        h, w = display.shape[:2]
        factor = min(1.0, PREVIEW_MAX_SIDE / max(h, w))
        if factor < 1.0:
            display = cv2.resize(display, (max(1, int(round(w * factor))),
                                           max(1, int(round(h * factor)))),
                                 interpolation=cv2.INTER_AREA)
        result = (display, canvas, depth, name, frame_label)
        self._preview_cache = (key, result)
        return result

    def _draft_config(self) -> watermark.WatermarkConfig:
        family = self.font_combo.currentFont().family()
        if family == self.font().family():
            family = ""
        return watermark.normalize_config({
            "text": self.text_edit.text(),
            "copyright_enabled": self.copyright_check.isChecked(),
            "copyright_position": self.copyright_position_combo.currentData(),
            "font_family": family,
            "size_percent": float(self.size_spin.value()),
            "opacity": float(self.opacity_spin.value()) / 100.0,
            "color": self.color_combo.currentData(),
            "angle": float(self.angle_spin.value()),
            "density": self.density_combo.currentData(),
        })

    def _scope_sizes(self):
        """Real (width, height) of every selected target; shape-only."""
        window = self._window
        sizes = []
        if self.target_checks["source"].isChecked():
            for frame in getattr(window, "raw_images", ()) or ():
                sizes.append(tuple(frame.shape[:2]))
        if self.target_checks["registered"].isChecked():
            for frame in getattr(window, "registration_results", ()) or ():
                sizes.append(tuple(frame.shape[:2]))
        if self.target_checks["output"].isChecked():
            record = self._apply_record
            if (record is not None and not getattr(record, "watermark_baked", False)
                    and getattr(record, "image", None) is not None):
                sizes.append(tuple(record.image.shape[:2]))
            manager = getattr(window, "output_manager", None)
            items = getattr(window, "output_list", None)
            if manager is not None and items is not None:
                for row in range(items.count()):
                    other = manager.record_for_row(row)
                    if (other is None or other.watermark_baked
                            or getattr(other, "image", None) is None
                            or other is record):
                        continue
                    if watermark.record_belongs_to_current_stack(window, other):
                        sizes.append(tuple(other.image.shape[:2]))
        return sizes

    def _refresh_preview(self, *_args) -> None:
        self._sync_target_availability()
        self._sync_preview_navigation()
        self._ok_button.setEnabled(False)
        self.font_note.clear()
        self._error.clear()
        if not self.text_edit.text().strip():
            preview = self._preview_image_and_canvas()
            if preview is not None:
                display, canvas, depth, name, frame_label = preview
                self.preview_lbl.set_display_pixmap(cv2_to_pixmap(display))
                self.header.setText(" · ".join(part for part in (
                    name, frame_label,
                    trans.t("watermark_preview_meta").format(
                        width=canvas[1], height=canvas[0], depth=depth)) if part))
                self.preview_meta.clear()
            else:
                self.preview_lbl.clear()
                self._error.setText(trans.t("watermark_preview_no_target"))
            return
        try:
            cfg = self._draft_config()
        except watermark.WatermarkError as exc:
            self.preview_lbl.clear()
            self._error.setText(trans.t(exc.message_key))
            return
        preview = self._preview_image_and_canvas()
        if preview is None:
            self.preview_lbl.clear()
            self._error.setText(trans.t("watermark_preview_no_target"))
            return
        display, canvas, depth, name, frame_label = preview
        canvas_w, canvas_h = int(canvas[1]), int(canvas[0])
        info = watermark.analyze(cfg, canvas_w, canvas_h)
        if not info["ok"]:
            self.preview_lbl.clear()
            self._error.setText(trans.t(info["message_key"]))
            return
        for width, height in self._scope_sizes():
            scope_info = watermark.analyze(cfg, int(width), int(height))
            if not scope_info["ok"]:
                self.preview_lbl.clear()
                self._error.setText(trans.t(scope_info["message_key"]))
                return
        preview_record = self._preview_record()
        drawn = (display if (preview_record is not None
                             and getattr(preview_record, "watermark_baked", False))
                 else watermark.burn(display, cfg, canvas_size=(canvas_w, canvas_h)))
        pixmap = cv2_to_pixmap(drawn)
        self.preview_lbl.set_display_pixmap(pixmap)
        self.header.setText(" · ".join(part for part in (
            name, frame_label,
            trans.t("watermark_preview_meta").format(
                width=canvas_w, height=canvas_h, depth=depth)) if part))
        layout = info["layout"]
        self.preview_meta.setText(trans.t("watermark_preview_layout").format(
            font=layout.font_px, cells=len(layout.cells)))
        if info.get("substituted"):
            self.font_note.setText(trans.t("label_font_substituted").format(
                name=f"{info['substituted']} → {info['actual_family']}"))
        self._ok_button.setEnabled(self._apply_allowed())
        if not any(check.isChecked() for check in self.target_checks.values()):
            self._error.setText(trans.t("watermark_target_required"))

    def _apply_allowed(self) -> bool:
        return any(check.isChecked() for check in self.target_checks.values())

    def _actual_size(self):
        pixmap = self.preview_lbl._base_pixmap
        if pixmap is None:
            return
        fit = min(self.preview_lbl.width() / pixmap.width(),
                  self.preview_lbl.height() / pixmap.height())
        self.preview_lbl._zoom_factor = 1.0 / max(fit, 1e-9)
        self.preview_lbl.refresh_display()

    def _reset_defaults(self) -> None:
        blockers = [QSignalBlocker(widget) for widget in (
            self.size_slider, self.size_spin, self.opacity_slider,
            self.opacity_spin, self.angle_slider, self.angle_spin,
            self.color_combo, self.density_combo, self.font_combo)]
        self.font_combo.setCurrentFont(QFont(self.font().family()))
        self.size_spin.setValue(watermark.DEFAULT_SIZE_PERCENT)
        self.opacity_spin.setValue(int(round(watermark.DEFAULT_OPACITY * 100)))
        self.color_combo.setCurrentIndex(self.color_combo.findData(watermark.DEFAULT_COLOR))
        self.angle_spin.setValue(watermark.DEFAULT_ANGLE)
        self.density_combo.setCurrentIndex(
            self.density_combo.findData(watermark.DEFAULT_DENSITY))
        del blockers
        self._schedule_preview()

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------
    def _save_and_accept(self) -> None:
        parent = self._window
        if parent is None:
            self.accept()
            return
        self._refresh_preview()
        if not self._ok_button.isEnabled():
            return
        try:
            cfg = self._draft_config()
        except watermark.WatermarkError as exc:
            self._error.setText(trans.t(exc.message_key))
            return
        targets = tuple(target for target, check in self.target_checks.items()
                        if check.isChecked())
        if not targets:
            self._error.setText(trans.t("watermark_target_required"))
            return
        # The selected entry is updated under its own scope: results use the
        # output scope, an aligned stack uses the registered one.  Passing it
        # only for a checked "output" left a selected historical/batch aligned
        # record without the registered style the user just applied.
        selected = self._apply_record
        selected_target = ("registered" if getattr(selected, "kind", "") == "aligned"
                           else "output") if selected is not None else None
        previous = watermark.capture_state(parent)
        for key, value in cfg.to_dict().items():
            setattr(parent, watermark.ATTRS[key][0], value)
        for target in targets:
            if (target == "output" and self._apply_record is not None
                    and not watermark.record_belongs_to_current_stack(
                        parent, self._apply_record)):
                # Editing a selected historical result from another stack
                # must not rewrite this stack's default for future results.
                continue
            watermark.set_target_config(parent, target, cfg)
        watermark.apply_to_records(parent, cfg, targets,
                                   selected if selected_target in targets else None)
        if watermark.capture_state(parent) == previous:
            self.accept()
            return
        try:
            if hasattr(parent, "persist_settings"):
                parent.persist_settings()
            if hasattr(parent, "push_watermark_undo"):
                parent.push_watermark_undo(previous)
            elif hasattr(parent, "_mark_project_dirty"):
                parent._mark_project_dirty()
        except Exception:
            watermark.restore_state(parent, previous)
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
