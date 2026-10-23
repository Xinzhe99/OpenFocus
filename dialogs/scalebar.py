"""Scale bar settings dialog with a live preview thumbnail."""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFontComboBox,
    QWidget,
    QSlider,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

import cv2
from locales import trans
from utils import scalebar


from widgets.stepper import StepperSpinBox


class ScaleBarDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("dialog_scalebar_title"))
        self.setMinimumWidth(520)
        self._apply_style()

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(14)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        root.addLayout(layout, 0)

        preview_col = QVBoxLayout()
        preview_col.setContentsMargins(0, 0, 0, 0)
        preview_col.setSpacing(6)


        layout.addWidget(self._section(trans.t("scalebar_section_calibration")))
        form = QFormLayout()
        layout.addLayout(form)
        # µm/px spans several orders of magnitude in practice: 0.0001 for an
        # electron micrograph, 1-10 for light microscopy, up to ~1000 for a
        # macro/landscape shot (1 mm per pixel). A fixed 4-decimals/0-1000
        # spinner is wrong at both ends: it cannot express 0.00005 µm/px and
        # wastes three decimals on "50". The range is fixed to the physical
        # span and the decimals follow the value's magnitude.
        # A StepperSpinBox, not QDoubleSpinBox: Qt lays a spin box's two
        # buttons out on the trailing edge and ignores subcontrol-position, so
        # a "- on the left, + on the right" layout is unreachable from CSS.
        self.px_spin = StepperSpinBox(
            minimum=0.0, maximum=1000.0, decimals=4, step=0.01)
        self.px_spin.setValue(
            float(getattr(parent, "scale_um_per_px_manual", 0.0) or 0.0))
        self.px_spin.setToolTip(trans.t("scalebar_manual_hint"))
        # Adapt BEFORE the first setValue: at the default 2 decimals a
        # 0.00005 µm/px calibration was rounded to 0.00 on entry.
        self._adapt_px_decimals(
            float(getattr(parent, "scale_um_per_px_manual", 0.0) or 0.0))
        self.px_spin.setValue(
            float(getattr(parent, "scale_um_per_px_manual", 0.0) or 0.0))
        self.px_spin.valueChanged.connect(self._adapt_px_decimals)
        form.addRow(trans.t("scalebar_px_um"), self.px_spin)

        detected = getattr(parent, "source_px_um", None)
        if detected:
            self.detect_lbl = QLabel(
                trans.t("scalebar_detected").format(value=f"{detected:g}"))
        else:
            self.detect_lbl = QLabel(trans.t("scalebar_no_metadata"))
        self.detect_lbl.setWordWrap(True)
        form.addRow("", self.detect_lbl)

        # ---- Placement ------------------------------------------------
        layout.addWidget(self._section(trans.t("scalebar_section_place")))
        place_form = QFormLayout()
        layout.addLayout(place_form)
        self.pos_combo = QComboBox()
        for code, label in (("bottom-right", trans.t("scalebar_pos_br")),
                            ("bottom-left", trans.t("scalebar_pos_bl")),
                            ("top-right", trans.t("scalebar_pos_tr")),
                            ("top-left", trans.t("scalebar_pos_tl")),
                            ("custom", trans.t("scalebar_pos_custom"))):
            self.pos_combo.addItem(label, code)
        pos = getattr(parent, "scale_bar_position", "bottom-right")
        idx = self.pos_combo.findData(pos)
        self.pos_combo.setCurrentIndex(max(0, idx))
        place_form.addRow(trans.t("scalebar_position"), self.pos_combo)

        # Custom offsets (0-100 % of the frame), only meaningful for "custom".
        self.custom_row = QWidget()
        custom_lay = QHBoxLayout(self.custom_row)
        custom_lay.setContentsMargins(0, 0, 0, 0)
        custom_lay.setSpacing(8)
        custom_lay.addWidget(QLabel("X"))
        self.custom_x = QSlider(Qt.Orientation.Horizontal)
        self.custom_x.setRange(0, 100)
        self.custom_x.setValue(int(getattr(parent, "scale_bar_custom_x",
                                           0.95) * 100))
        custom_lay.addWidget(self.custom_x, 1)
        custom_lay.addWidget(QLabel("Y"))
        self.custom_y = QSlider(Qt.Orientation.Horizontal)
        self.custom_y.setRange(0, 100)
        self.custom_y.setValue(int(getattr(parent, "scale_bar_custom_y",
                                           0.95) * 100))
        custom_lay.addWidget(self.custom_y, 1)
        place_form.addRow("", self.custom_row)
        self.custom_row.setVisible(pos == "custom")

        # ---- Text style ------------------------------------------------
        layout.addWidget(self._section(trans.t("scalebar_section_text")))
        text_form = QFormLayout()
        layout.addLayout(text_form)
        self.font_combo = QFontComboBox()
        fam = getattr(parent, "scale_bar_font_family", "") or ""
        self.font_combo.setCurrentFont(QFont(fam or self.font().family()))
        text_form.addRow(trans.t("scalebar_font"), self.font_combo)

        size_row = QWidget()
        size_lay = QHBoxLayout(size_row)
        size_lay.setContentsMargins(0, 0, 0, 0)
        size_lay.setSpacing(8)
        self.size_spin = StepperSpinBox(
            minimum=0, maximum=400, decimals=0, step=2)
        self.size_spin.setValue(
            float(getattr(parent, "scale_bar_font_size", 0.0) or 0.0))
        self.size_spin.setMinimumWidth(96)
        self.size_hint = QLabel(trans.t("scalebar_font_auto"))
        self.size_hint.setObjectName("mutedLabel")
        size_lay.addWidget(self.size_spin)
        size_lay.addWidget(self.size_hint, 1)
        text_form.addRow(trans.t("scalebar_font_size"), size_row)

        self.bold_chk = QCheckBox(trans.t("scalebar_bold"))
        self.bold_chk.setChecked(bool(getattr(parent, "scale_bar_font_bold",
                                              False)))
        text_form.addRow("", self.bold_chk)

        self.align_combo = QComboBox()
        for code, label in (("auto", trans.t("scalebar_align_auto")),
                            ("left", trans.t("scalebar_align_left")),
                            ("center", trans.t("scalebar_align_center")),
                            ("right", trans.t("scalebar_align_right"))):
            self.align_combo.addItem(label, code)
        cur_align = getattr(parent, "scale_bar_text_align", "auto")
        self.align_combo.setCurrentIndex(
            max(0, self.align_combo.findData(cur_align)))
        text_form.addRow(trans.t("scalebar_text_align"), self.align_combo)

        self.target_combo = QComboBox()
        for code, label in (
                ("all", trans.t("scalebar_target_all")),
                ("source", trans.t("scalebar_target_source")),
                ("output", trans.t("scalebar_target_output"))):
            self.target_combo.addItem(label, code)
        cur_target = getattr(parent, "scale_bar_target", "all")
        self.target_combo.setCurrentIndex(
            max(0, self.target_combo.findData(cur_target)))
        form.addRow(trans.t("scalebar_target"), self.target_combo)

        self.color_combo = QComboBox()
        for code, label in (("auto", trans.t("scalebar_color_auto")),
                            ("white", trans.t("scalebar_color_white")),
                            ("black", trans.t("scalebar_color_black"))):
            self.color_combo.addItem(label, code)
        col = getattr(parent, "scale_bar_color", "auto")
        idx = self.color_combo.findData(col)
        self.color_combo.setCurrentIndex(max(0, idx))
        text_form.addRow(trans.t("scalebar_color"), self.color_combo)

        self.preview_lbl = QLabel(trans.t("scalebar_preview"))
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setMinimumSize(360, 300)
        self.preview_lbl.setStyleSheet(
            "background: #222; border: 1px solid #444;")
        preview_col.addWidget(self.preview_lbl, 1)
        root.addLayout(preview_col, 1)

        row = QHBoxLayout()
        row.addStretch()
        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.setObjectName("primaryButton")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._save_and_accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(ok_btn)
        row.addWidget(cancel_btn)
        layout.addLayout(row)

        self.pos_combo.currentIndexChanged.connect(
            lambda: self.custom_row.setVisible(
                self.pos_combo.currentData() == "custom"))
        for signal in (self.px_spin.valueChanged,
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

    def _section(self, text: str) -> QLabel:
        """A small bold heading that separates the dialog's three concerns."""
        lbl = QLabel(text)
        f = lbl.font()
        f.setBold(True)
        lbl.setFont(f)
        return lbl

    def _adapt_px_decimals(self, value: float) -> None:
        """Show enough decimals for sub-micron calibration, not more.

        0.0001 µm/px needs 4-5 decimals to stay editable; 50 µm/px needs one.
        Changing the decimals mid-edit would move the caret, so the step is
        derived from the current magnitude instead.
        """
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

    def _preview_image(self):
        parent = self.parent()
        candidates = []
        if parent is not None:
            candidates.append(getattr(parent, "fusion_result", None))
            raw = getattr(parent, "raw_images", None) or []
            if raw:
                candidates.append(raw[0])
        for img in candidates:
            if img is not None:
                return img
        # No image loaded yet: a neutral grey card shows where the bar and its
        # colours land. Random noise (the old fallback) looked like a broken
        # preview and made the "Auto (inverse of background)" colour choice
        # impossible to judge.
        import numpy as np
        return np.full((260, 520, 3), 128, dtype=np.uint8)

    def _effective_px_um(self):
        """Same rule as the export path (utils.scalebar.effective_px_um):
        a manual value wins over metadata, so the preview matches the file."""
        parent = self.parent()
        manual = self.px_spin.value()
        if parent is not None:
            try:
                parent.scale_um_per_px_manual = manual
            except Exception:
                pass
            from utils.scalebar import effective_px_um
            return effective_px_um(parent)
        detected = getattr(parent, "source_px_um", None) if parent else None
        if detected and detected > 0:
            return detected
        return manual if manual > 0 else None

    def _font_family(self) -> str:
        """The chosen family, or "" when it is the dialog's own default."""
        fam = self.font_combo.currentFont().family()
        return "" if fam == self.font().family() else fam

    def _refresh_preview(self):
        img = self._preview_image()
        px_um = self._effective_px_um()
        h, w = img.shape[:2]
        scale = 520.0 / w if w > 520 else 1.0
        small = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else img
        # The preview always draws the bar: it previews the STYLE
        # (font, size, position), while the checkbox only decides
        # whether the app uses the bar at all.
        if px_um:
            # The preview draws on a downscaled copy, so EVERY absolute size
            # must shrink by the same factor: the calibration (bar length) did,
            # but the font did not — a 24 px label on the 520 px preview looked
            # ~2.6x bigger than the same 24 px on the 1344 px export.
            factor = scale if scale < 1.0 else 1.0
            requested = float(self.size_spin.value() or 0.0)
            small = scalebar.draw_scale_bar(
                small, px_um / (factor if factor > 0 else 1.0),
                self.pos_combo.currentData(), self.color_combo.currentData(),
                font_family=self._font_family(),
                font_size=requested * factor if requested > 0 else 0.0,
                font_bold=self.bold_chk.isChecked(),
                text_align=self.align_combo.currentData(),
                custom_x=self.custom_x.value() / 100.0,
                custom_y=self.custom_y.value() / 100.0)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      3 * rgb.shape[1], QImage.Format.Format_RGB888)
        self.preview_lbl.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.preview_lbl.width() or 480, self.preview_lbl.height() or 260,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        """拖大窗口后预览要跟着重算（简单防抖：直接重绘即可，预览是缩略图）。"""
        super().resizeEvent(event)
        if self.isVisible():
            self._refresh_preview()

    def _save_and_accept(self):
        parent = self.parent()
        if parent is not None:
            # Saving a configured bar means it is in use; removal lives in
            # Tools > 比例尺 > 删除比例尺.
            parent.scale_bar_enabled = True
            parent.scale_um_per_px_manual = float(self.px_spin.value() or 0.0)
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
            # Redraw every view NOW: the bar used to appear/disappear only
            # after the next render, which read as "the scale bar does
            # nothing" (and worse, "unchecking left it on the fused view").
            if hasattr(parent, "refresh_annotation_views"):
                parent.refresh_annotation_views()
            elif getattr(parent, "current_display_index", None) is not None:
                parent.update_source_view(parent.current_display_index)
        self.accept()
