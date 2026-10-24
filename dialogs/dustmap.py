"""Prepare flat-field previews and atomic stack corrections off the GUI thread."""
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QVBoxLayout,
)

from locales import trans
from widgets.stepper import StepperSpinBox
from utils.dust_map import DEFAULT_SIGMA, compute_flat_field, apply_flat_field


def _to_pixmap(img, width=340):
    """8-bit BGR -> QPixmap, downscaled to width."""
    scale = min(width / img.shape[1], 180 / img.shape[0])
    small = cv2.resize(img, (max(1, round(img.shape[1] * scale)),
                             max(1, round(img.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    qimg = QImage(small.data, w, h, 3 * w, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class _FlatFieldWorker(QThread):
    def __init__(self, operation, revision, raw, base, *, white=None, path="",
                 sigma=DEFAULT_SIGMA, target=0, gain=None):
        super().__init__()
        self.operation, self.revision = operation, revision
        self.raw, self.base = tuple(raw), tuple(base)
        self.white, self.path, self.sigma = white, path, sigma
        self.target, self.gain = target, gain
        self.result = None
        self.error = ""
        self.error_key = "msg_proc_error"

    def _check_cancel(self):
        if self.isInterruptionRequested():
            raise InterruptedError()

    def run(self):
        try:
            self._check_cancel()
            if self.operation == "preview":
                white = self.white
                if self.path:
                    self.error_key = "dust_err_decode"
                    white = cv2.imdecode(np.fromfile(self.path, dtype=np.uint8),
                                         cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                if white is None:
                    raise ValueError(trans.t("dust_err_decode"))
                self._check_cancel()
                self.error_key = "msg_size_mismatch_title"
                if any(frame.shape[:2] != white.shape[:2] for frame in self.raw + self.base):
                    raise ValueError(trans.t("msg_size_mismatch_title"))
                self.error_key = "dust_err_not_white"
                gain = compute_flat_field(white, sigma=self.sigma)
                self._check_cancel()
                self.error_key = "msg_proc_error"
                frame = self.raw[self.target]
                fixed = apply_flat_field(frame, gain)
                self.result = (white, gain, frame, fixed)
            else:
                corrected = []
                for frames in (self.raw, self.base):
                    group = []
                    for frame in frames:
                        self._check_cancel()
                        group.append(apply_flat_field(frame, self.gain))
                    corrected.append(group)
                self.result = tuple(corrected)
            self._check_cancel()
        except InterruptedError:
            self.result = None
        except Exception as exc:
            self.result = None
            self.error = str(exc)


class DustMapDialog(QDialog):
    def __init__(self, window_main, parent=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self._gain = None
        self._white = None
        self._worker = None
        self._revision = 0
        self._gain_revision = -1
        self._close_pending = False
        self._file_path = ""
        self._gain_sources = None
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
        self.lbl_context = QLabel()
        self.lbl_context.setWordWrap(True)
        lay.addWidget(self.lbl_context)

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
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel(trans.t("label_preview")))
        self.combo_target = QComboBox()
        target_row.addWidget(self.combo_target, 1)
        lay.addLayout(target_row)

        # 尘点尺度：增益场的提取半径。太小会把白场噪声当成尘点，太大
        # 会把尘点一起抹平（σ=32 时只能去掉约 9% 的尘点对比度）。
        size_row = QHBoxLayout()
        self.lbl_size = QLabel(trans.t("dust_size"))
        self.lbl_size.setToolTip(trans.t("dust_size_hint"))
        size_row.addWidget(self.lbl_size)
        self.spin_sigma = StepperSpinBox(minimum=1.0, maximum=40.0,
                                       decimals=1, step=1.0, value=DEFAULT_SIGMA)
        self.spin_sigma.setRange(1.0, 40.0)
        self.spin_sigma.setSingleStep(1.0)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setValue(DEFAULT_SIGMA)
        # 80 px was narrower than the widget needs with its arrows and QSS
        # padding (112 px), so "4.0" lost its last digit.
        self.spin_sigma.setMinimumWidth(112)
        self.spin_sigma.setToolTip(trans.t("dust_size_hint"))
        self.spin_sigma.valueChanged.connect(self._sigma_changed)
        size_row.addStretch()
        size_row.addWidget(self.spin_sigma)
        lay.addLayout(size_row)

        self.chk_remove = QCheckBox(trans.t("dust_remove_hint"))
        lay.addWidget(self.chk_remove)
        self.chk_remove.setEnabled(False)

        self.btn_preview = QPushButton(trans.t("dust_preview"))
        self.btn_preview.clicked.connect(self._preview)
        lay.addWidget(self.btn_preview)

        grid = QHBoxLayout()
        self.lbl_gain = QLabel(trans.t("dust_no_preview"))
        self.lbl_gain.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_gain.setMinimumSize(340, 220)
        # Preview placeholders follow the theme: a hard-coded near-black
        # card sat inside the white dialog as three black blocks.
        from ui.styles import CURRENT_THEME
        from utils.design_tokens import palette
        _p = palette(CURRENT_THEME)
        _pv_bg, _pv_border, _pv_fg = (_p["surfaceSunken"], _p["border"],
                                      _p["textMuted"])
        self.lbl_gain.setStyleSheet(
            f"background:{_pv_bg};border:1px solid {_pv_border};color:{_pv_fg};")
        grid.addWidget(self.lbl_gain)
        before_after = QVBoxLayout()
        self.lbl_before = QLabel(trans.t("dust_before"))
        self.lbl_after = QLabel(trans.t("dust_after"))
        for lbl in (self.lbl_before, self.lbl_after):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(340, 104)
            lbl.setStyleSheet(
                f"background:{_pv_bg};border:1px solid {_pv_border};color:{_pv_fg};")
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
        self.btn_cancel = QPushButton(trans.t("btn_cancel_compute"))
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setVisible(False)
        btns.addWidget(self.btn_cancel)
        self.btn_close = QPushButton(trans.t("dialog_close"))
        self.btn_close.clicked.connect(self.reject)
        btns.addWidget(self.btn_close)
        lay.addLayout(btns)

        self.rb_file.toggled.connect(self._source_changed)
        self.rb_stack.toggled.connect(self._source_changed)
        self._fill_stack_combo()
        self.combo_stack.currentIndexChanged.connect(self._source_changed)
        self.combo_target.currentIndexChanged.connect(self._invalidate_gain)

    def _apply_style(self):
        """Styling comes from the generated application stylesheet."""
        self.setStyleSheet("")

    # ------------------------------------------------------------ logic
    def _fill_stack_combo(self):
        self.combo_stack.clear()
        self.combo_target.clear()
        frames = getattr(self.window_main, "raw_images", None) or []
        names = getattr(self.window_main, "image_filenames", None) or []
        for i in range(len(frames)):
            name = names[i] if i < len(names) else ""
            text = trans.t("dialog_roi_frame_named").format(i=i + 1, name=name)
            self.combo_stack.addItem(text, i)
            self.combo_target.addItem(text, i)
        if frames:
            frame = frames[0]
            self.lbl_context.setText(
                f"{trans.t('panel_source_stack')}: {len(frames)} · "
                f"{frame.shape[1]}×{frame.shape[0]} · {frame.dtype}")

    def _source_changed(self):
        from_file = self.rb_file.isChecked()
        self.btn_browse.setEnabled(from_file)
        self.lbl_path.setEnabled(from_file)
        self.combo_stack.setEnabled(not from_file)
        self.chk_remove.setEnabled(not from_file)
        self.chk_remove.setToolTip(self.combo_stack.currentText() if not from_file else "")
        self._invalidate_gain()

    def _invalidate_gain(self, *_args):
        self._revision += 1
        if self._worker is not None:
            self._worker.requestInterruption()
        self._gain = None
        self._gain_revision = -1
        self.btn_apply.setEnabled(False)
        self.lbl_gain.setText(trans.t("dust_no_preview"))
        self.lbl_before.setText(self.combo_target.currentText())
        self.lbl_after.setText(trans.t("dust_after"))
        self.lbl_status.setText("")

    def _sigma_changed(self, _value=None):
        self._invalidate_gain()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, trans.t("dust_browse"), "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            self._load_white_from_path(path)

    def _load_white_from_path(self, path):
        self._file_path = path
        self._white = None
        self.lbl_path.setText(path)
        self.lbl_path.setToolTip(path)
        self._invalidate_gain()
        self._preview()

    def _current_white(self):
        if self.rb_file.isChecked():
            return self._white
        idx = self.combo_stack.currentData()
        frames = getattr(self.window_main, "raw_images", None) or []
        return frames[idx] if idx is not None and 0 <= idx < len(frames) else None

    def _sources(self):
        return (tuple(getattr(self.window_main, "raw_images", ()) or ()),
                tuple(getattr(self.window_main, "base_images", ()) or ()))

    def _sources_match(self, sources):
        return all(len(old) == len(new) and all(a is b for a, b in zip(old, new))
                   for old, new in zip(sources, self._sources()))

    def _preview(self):
        if self._worker is not None:
            return
        raw, base = self._sources()
        white = self._current_white()
        path = self._file_path if self.rb_file.isChecked() else ""
        if not raw or (white is None and not path):
            self._fail(trans.t("dust_err_no_source"))
            return
        self._invalidate_gain()
        worker = _FlatFieldWorker("preview", self._revision, raw, base,
                                 white=white, path=path,
                                 sigma=float(self.spin_sigma.value()),
                                 target=self.combo_target.currentIndex())
        self._start_worker(worker)

    def _start_worker(self, worker):
        self._worker = worker
        self.btn_apply.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setVisible(True)
        self.lbl_status.setText(trans.t("msg_gif_processing_title"))
        worker.finished.connect(self._worker_finished)
        worker.start()

    def _worker_finished(self):
        worker = self._worker
        self._worker = None
        self.btn_cancel.setVisible(False)
        self.btn_preview.setEnabled(True)
        worker.deleteLater()
        if self._close_pending:
            super().reject()
            return
        if worker.revision != self._revision or not self._sources_match((worker.raw, worker.base)):
            self._invalidate_gain()
            return
        if worker.isInterruptionRequested():
            self.lbl_status.setText(trans.t("compare_cancelled"))
            self.btn_apply.setEnabled(self._gain is not None)
            return
        if worker.error:
            self._fail(trans.t(worker.error_key) + "\n" + worker.error)
            return
        if worker.operation == "apply":
            if self.window_main.apply_flat_field_to_stack(
                    worker.gain, remove_index=worker.remove_index,
                    prepared=worker.result) is True:
                self.accept()
            else:
                self.btn_apply.setEnabled(True)
                self.lbl_status.setText(trans.t("msg_proc_error"))
            return
        white, gain, frame, fixed = worker.result
        if self.rb_file.isChecked():
            self._white = white
        self._gain = gain
        self._gain_revision = worker.revision
        self._gain_sources = (worker.raw, worker.base)
        g = gain[:, :, 0] if gain.ndim == 3 else gain
        norm = np.clip((g - 0.5) / 1.5, 0, 1)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        self.lbl_gain.setPixmap(_to_pixmap(heat))
        self.lbl_gain.setToolTip("0.5–2.0")
        from utils.image_utils import to_display_uint8
        self.lbl_before.setPixmap(_to_pixmap(to_display_uint8(frame)))
        self.lbl_after.setPixmap(_to_pixmap(to_display_uint8(fixed)))
        self.btn_apply.setEnabled(True)
        self.lbl_status.setText(trans.t("dust_ready"))

    def _fail(self, message):
        self._invalidate_gain()
        self.lbl_status.setText(message)

    def _apply(self):
        if (self._worker is not None or self._gain is None
                or self._gain_revision != self._revision):
            return
        if not self._sources_match(self._gain_sources):
            self._invalidate_gain()
            return
        raw, base = self._sources()
        worker = _FlatFieldWorker("apply", self._revision, raw, base, gain=self._gain)
        worker.remove_index = (self.combo_stack.currentData()
                               if self.rb_stack.isChecked() and self.chk_remove.isChecked()
                               else None)
        self._start_worker(worker)

    def _cancel(self):
        if self._worker is not None:
            self._worker.requestInterruption()
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText(trans.t("depth_cancelling"))

    def reject(self):
        if self._worker is not None:
            self._close_pending = True
            self._cancel()
            return
        super().reject()

    def closeEvent(self, event):
        if self._worker is not None:
            event.ignore()
            self.reject()
        else:
            super().closeEvent(event)
