"""Flat-field (dust map) correction dialog.

One defocused shot of a blank/white area records the static gain field of
the optics — dust specks and vignetting. The dialog computes the field
(utils/dust_map.py), previews it as a heatmap with a before/after of the
first frame, and applies the division to the whole stack through the main
window (undoable, alignment caches invalidated).

The white-field reference comes either from a file or from a frame of the
current stack (microscopy folders often contain the white shot). Removing
that frame afterwards is left to the source list (Del) on purpose —
deletion logic already lives in source_manager.
"""
import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QVBoxLayout,
)

from locales import trans
from utils.dust_map import DEFAULT_SIGMA, compute_flat_field, apply_flat_field


def _to_pixmap(img, width=340):
    """8-bit BGR -> QPixmap, downscaled to width."""
    small = cv2.resize(img, (width, max(1, int(img.shape[0] * width / img.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    qimg = QImage(small.data, w, h, 3 * w, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class DustMapDialog(QDialog):
    def __init__(self, window_main, parent=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self._gain = None
        self._white = None
        self.setWindowTitle(trans.t("dust_title"))
        self.setModal(True)
        self.setMinimumWidth(760)
        self._build_ui()
        self._apply_style()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)

        intro = QLabel(trans.t("dust_intro"))
        intro.setWordWrap(True)
        lay.addWidget(intro)

        # 两个来源单选各占一行：并排时第二个按钮的文字在 760px 宽的
        # 对话框里会被挤到第一个按钮上（实测文字互相重叠）。
        self.rb_file = QRadioButton(trans.t("dust_from_file"))
        self.rb_stack = QRadioButton(trans.t("dust_from_stack"))
        self.rb_file.setChecked(True)
        lay.addWidget(self.rb_file)
        lay.addWidget(self.rb_stack)

        pick_row = QHBoxLayout()
        self.btn_browse = QPushButton(trans.t("dust_browse"))
        self.btn_browse.clicked.connect(self._browse)
        pick_row.addWidget(self.btn_browse)
        self.lbl_path = QLabel(trans.t("dust_no_file"))
        pick_row.addWidget(self.lbl_path, 1)
        lay.addLayout(pick_row)

        stack_row = QHBoxLayout()
        stack_row.addWidget(QLabel(trans.t("dust_stack_frame")))
        self.combo_stack = QComboBox()
        self.combo_stack.setEnabled(False)
        stack_row.addWidget(self.combo_stack, 1)
        lay.addLayout(stack_row)

        # 尘点尺度：增益场的提取半径。太小会把白场噪声当成尘点，太大
        # 会把尘点一起抹平（σ=32 时只能去掉约 9% 的尘点对比度）。
        size_row = QHBoxLayout()
        self.lbl_size = QLabel(trans.t("dust_size"))
        self.lbl_size.setToolTip(trans.t("dust_size_hint"))
        size_row.addWidget(self.lbl_size)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(1.0, 40.0)
        self.spin_sigma.setSingleStep(1.0)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setValue(DEFAULT_SIGMA)
        self.spin_sigma.setFixedWidth(80)
        self.spin_sigma.setToolTip(trans.t("dust_size_hint"))
        self.spin_sigma.valueChanged.connect(self._sigma_changed)
        size_row.addStretch()
        size_row.addWidget(self.spin_sigma)
        lay.addLayout(size_row)

        self.chk_remove = QCheckBox(trans.t("dust_remove_hint"))
        lay.addWidget(self.chk_remove)

        self.btn_preview = QPushButton(trans.t("dust_preview"))
        self.btn_preview.clicked.connect(self._preview)
        lay.addWidget(self.btn_preview)

        grid = QHBoxLayout()
        self.lbl_gain = QLabel(trans.t("dust_no_preview"))
        self.lbl_gain.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_gain.setMinimumSize(340, 220)
        self.lbl_gain.setStyleSheet("background:#161616;border:1px solid #444;color:#888;")
        grid.addWidget(self.lbl_gain)
        before_after = QVBoxLayout()
        self.lbl_before = QLabel(trans.t("dust_before"))
        self.lbl_after = QLabel(trans.t("dust_after"))
        for lbl in (self.lbl_before, self.lbl_after):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(340, 104)
            lbl.setStyleSheet("background:#161616;border:1px solid #444;color:#888;")
        before_after.addWidget(self.lbl_before)
        before_after.addWidget(self.lbl_after)
        grid.addLayout(before_after)
        lay.addLayout(grid, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        btns = QHBoxLayout()
        btns.addStretch()
        self.btn_apply = QPushButton(trans.t("dust_apply"))
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply)
        btns.addWidget(self.btn_apply)
        self.btn_close = QPushButton(trans.t("dialog_close"))
        self.btn_close.clicked.connect(self.reject)
        btns.addWidget(self.btn_close)
        lay.addLayout(btns)

        self.rb_file.toggled.connect(self._source_changed)
        self.rb_stack.toggled.connect(self._source_changed)
        self._fill_stack_combo()

    def _apply_style(self):
        try:
            if self.window_main and getattr(self.window_main, "ui_theme", "dark") != "light":
                from ui.styles import DIALOG_CONTROL_DARK_QSS
                self.setStyleSheet(
                    "QDialog { background: #2b2b2b; border: 1px solid #444; }"
                    "QLabel { color: #d0d0d0; }" + DIALOG_CONTROL_DARK_QSS)
        except Exception:
            pass

    # ------------------------------------------------------------ logic
    def _fill_stack_combo(self):
        self.combo_stack.clear()
        frames = getattr(self.window_main, "raw_images", None) or []
        names = getattr(self.window_main, "image_filenames", None) or []
        for i in range(len(frames)):
            name = names[i] if i < len(names) else f"frame_{i + 1}"
            self.combo_stack.addItem(f"{i + 1}. {name}", i)

    def _source_changed(self):
        from_file = self.rb_file.isChecked()
        self.btn_browse.setEnabled(from_file)
        self.lbl_path.setEnabled(from_file)
        self.combo_stack.setEnabled(not from_file)

    def _sigma_changed(self, _value=None):
        """Recompute the field for the new scale (only if a white field exists)."""
        if self._white is not None:
            self._compute()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, trans.t("dust_browse"), "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            self.lbl_path.setText(path)
            self._load_white_from_path(path)

    def _load_white_from_path(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            self._fail(trans.t("dust_err_decode"))
            return
        self._white = img
        self._compute()

    def _current_white(self):
        if self.rb_file.isChecked():
            text = self.lbl_path.text()
            if text in (trans.t("dust_no_file"), ""):
                self._fail(trans.t("dust_err_no_source"))
                return None
            return self._white
        idx = self.combo_stack.currentData()
        frames = getattr(self.window_main, "raw_images", None) or []
        if idx is None or idx >= len(frames):
            self._fail(trans.t("dust_err_no_source"))
            return None
        return frames[idx]

    def _preview(self):
        white = self._current_white()
        if white is not None:
            self._compute(white)

    def _compute(self, white=None):
        if white is None:
            white = self._white
        if white is None:
            self._fail(trans.t("dust_err_no_source"))
            return
        self.lbl_status.setText("")
        try:
            self._gain = compute_flat_field(
                white, sigma=float(self.spin_sigma.value()))
        except ValueError as exc:
            # The library raises English diagnostics; the user sees a
            # localized one and the detail goes to the log.
            import logging
            logging.getLogger("openfocus").warning("flat-field rejected: %s", exc)
            self._fail(trans.t("dust_err_not_white"))
            return
        # 增益热图：0.5–2.0 线性映射到伪彩
        g = self._gain[:, :, 0] if self._gain.ndim == 3 else self._gain
        norm = np.clip((g - 0.5) / 1.5, 0, 1)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
        self.lbl_gain.setPixmap(_to_pixmap(heat))
        # 当前栈第 0 帧的前后对比（无栈时跳过）
        frames = getattr(self.window_main, "raw_images", None) or []
        if frames:
            from utils.image_utils import to_display_uint8
            f0 = frames[0]
            try:
                fixed = apply_flat_field(f0, self._gain)
            except ValueError:
                # white field and stack frames disagree in resolution
                self._fail(trans.t("dust_err_not_white"))
                return
            self.lbl_before.setPixmap(_to_pixmap(to_display_uint8(f0)))
            self.lbl_after.setPixmap(_to_pixmap(to_display_uint8(fixed)))
        self.btn_apply.setEnabled(True)
        self.lbl_status.setStyleSheet("color: #7a7;")
        self.lbl_status.setText(trans.t("dust_ready"))

    def _fail(self, message):
        self._gain = None
        self.btn_apply.setEnabled(False)
        self.lbl_status.setStyleSheet("color: #c66;")
        self.lbl_status.setText(message)

    def _apply(self):
        if self._gain is None:
            return
        remove_idx = None
        if self.rb_stack.isChecked() and self.chk_remove.isChecked():
            remove_idx = self.combo_stack.currentData()
        self.window_main.apply_flat_field_to_stack(self._gain)
        if remove_idx is not None:
            # 删除复用 source_manager 的列表项删除路径
            lw = getattr(self.window_main, "file_list", None)
            if lw is not None and 0 <= remove_idx < lw.count():
                lw.setCurrentRow(remove_idx)
                self.window_main.source_manager.delete_selected_source_images()
        self.accept()
