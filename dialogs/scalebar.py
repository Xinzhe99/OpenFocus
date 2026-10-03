"""Scale bar settings dialog with a live preview thumbnail."""
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

import cv2
from locales import trans
from utils import scalebar


class ScaleBarDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("dialog_scalebar_title"))
        self.setMinimumWidth(520)
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        self.enable_chk = QCheckBox(trans.t("scalebar_enable"))
        self.enable_chk.setChecked(getattr(parent, "scale_bar_enabled", False))
        layout.addWidget(self.enable_chk)

        form = QFormLayout()
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setDecimals(4)
        self.px_spin.setRange(0.0, 1000.0)
        self.px_spin.setSingleStep(0.01)
        self.px_spin.setValue(float(getattr(parent, "scale_um_per_px_manual", 0.0) or 0.0))
        self.px_spin.setToolTip(trans.t("scalebar_manual_hint"))
        form.addRow(trans.t("scalebar_px_um"), self.px_spin)

        detected = getattr(parent, "source_px_um", None)
        if detected:
            self.detect_lbl = QLabel(
                trans.t("scalebar_detected").format(value=f"{detected:g}"))
        else:
            self.detect_lbl = QLabel(trans.t("scalebar_no_metadata"))
        self.detect_lbl.setWordWrap(True)
        form.addRow("", self.detect_lbl)

        self.pos_combo = QComboBox()
        for code, label in (("bottom-right", trans.t("scalebar_pos_br")),
                            ("bottom-left", trans.t("scalebar_pos_bl")),
                            ("top-right", trans.t("scalebar_pos_tr")),
                            ("top-left", trans.t("scalebar_pos_tl"))):
            self.pos_combo.addItem(label, code)
        pos = getattr(parent, "scale_bar_position", "bottom-right")
        idx = self.pos_combo.findData(pos)
        self.pos_combo.setCurrentIndex(max(0, idx))
        form.addRow(trans.t("scalebar_position"), self.pos_combo)

        self.color_combo = QComboBox()
        for code, label in (("auto", trans.t("scalebar_color_auto")),
                            ("white", trans.t("scalebar_color_white")),
                            ("black", trans.t("scalebar_color_black"))):
            self.color_combo.addItem(label, code)
        col = getattr(parent, "scale_bar_color", "auto")
        idx = self.color_combo.findData(col)
        self.color_combo.setCurrentIndex(max(0, idx))
        form.addRow(trans.t("scalebar_color"), self.color_combo)
        layout.addLayout(form)

        self.preview_lbl = QLabel(trans.t("scalebar_preview"))
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setMinimumHeight(240)
        self.preview_lbl.setStyleSheet("background: #222; border: 1px solid #444;")
        layout.addWidget(self.preview_lbl, 1)

        row = QHBoxLayout()
        row.addStretch()
        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._save_and_accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(ok_btn)
        row.addWidget(cancel_btn)
        layout.addLayout(row)

        for signal in (self.enable_chk.toggled, self.px_spin.valueChanged,
                       self.pos_combo.currentIndexChanged,
                       self.color_combo.currentIndexChanged):
            signal.connect(self._refresh_preview)
        self._refresh_preview()

    def _apply_style(self):
        parent = self.parent()
        theme = getattr(parent, "ui_theme", "dark") if parent else "dark"
        if theme == "light":
            self.setStyleSheet("QDialog { background-color: #ffffff; } QLabel { color: #1f2328; }")
        else:
            self.setStyleSheet("QDialog { background-color: #2b2b2b; border: 1px solid #444; }"
                               "QLabel { color: #d0d0d0; }")

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
        import numpy as np
        return (np.random.rand(400, 600, 3) * 255).astype(np.uint8)

    def _effective_px_um(self):
        parent = self.parent()
        detected = getattr(parent, "source_px_um", None) if parent else None
        if detected and detected > 0:
            return detected
        manual = self.px_spin.value()
        return manual if manual > 0 else None

    def _refresh_preview(self):
        img = self._preview_image()
        px_um = self._effective_px_um()
        h, w = img.shape[:2]
        scale = 520.0 / w if w > 520 else 1.0
        small = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else img
        if px_um:
            small = scalebar.draw_scale_bar(
                small, px_um, self.pos_combo.currentData(),
                self.color_combo.currentData())
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
            parent.scale_bar_enabled = self.enable_chk.isChecked()
            parent.scale_um_per_px_manual = float(self.px_spin.value() or 0.0)
            parent.scale_bar_position = self.pos_combo.currentData()
            parent.scale_bar_color = self.color_combo.currentData()
            if hasattr(parent, "persist_settings"):
                parent.persist_settings()
        self.accept()
