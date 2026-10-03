"""Depth-map viewer/exporter with pseudo-color customization.

Computes a focus-position map with the selected algorithm's own activity
measure (core.depth_map), previews it colorized (optionally overlaid on the
fused result), and exports either the colorized image or the raw 16-bit
index map. Custom color schemes are interpolated from user color stops and
persisted in QSettings.
"""
import json

import cv2
import numpy as np
from PyQt6.QtCore import QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from locales import trans
from utils import colormap as cmap
from utils.image_utils import imwrite_auto
from utils.settings_store import get_last_dialog_dir, get_settings, set_last_dialog_dir

_METHOD_KEYS = (
    ("guided_filter", "radio_guided_filter"),
    ("dct", "radio_dct"),
    ("dtcwt", "radio_dtcwt"),
    ("gfgfgf", "radio_gfg"),
    ("stackmffv4", "radio_stackmff"),
)

_DEFAULT_METHOD_COLORMAP = "turbo"

# 打开对话框时自动计算的方法（秒级）；dtcwt/stackmffv4 分钟级，不自动跑
_FAST_AUTO_METHODS = {"guided_filter", "dct", "gfgfgf"}

# 对话框关闭时仍在跑的 worker 放这里保活：QThread 对象被销毁而线程仍在
# 运行会直接 abort 整个进程（AI 推理可达数分钟，用户关窗很正常）。
_ORPHAN_WORKERS = set()


class DepthWorker(QThread):
    progress = pyqtSignal(int, int)
    ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, images, method, use_gpu, parent=None):
        super().__init__(parent)
        self._images = images
        self._method = method
        self._use_gpu = use_gpu
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from core.depth_map import DepthMapCancelled, compute_focus_index
        try:
            model_path = None
            if self._method == "stackmffv4":
                from core.cli import _find_model_path
                model_path = _find_model_path()
            m = compute_focus_index(
                self._images, self._method,
                model_path=model_path, use_gpu=self._use_gpu,
                progress_callback=lambda i, n: self.progress.emit(i, n),
                should_cancel=lambda: self._cancelled,
            )
            if self._cancelled:
                self.failed.emit("cancelled")
                return
            self.ok.emit(m)
        except DepthMapCancelled:
            self.failed.emit("cancelled")
        except Exception as exc:  # surface to the dialog status line
            self.failed.emit(str(exc))


class _GradientBar(QLabel):
    """Thin color legend: LUT gradient with endpoint labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lut = None
        self.setMinimumHeight(26)
        self.setText("")

    def set_lut(self, lut):
        self._lut = lut
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1a1a1a"))
        if self._lut is None:
            return
        m = 12  # side margin for labels
        bar = self.height() - 14
        y = 12
        n = max(1, self.width() - 2 * m)
        for i in range(n):
            c = self._lut[int(i / n * 255)]
            p.setPen(QColor(c[2], c[1], c[0]))
            p.drawLine(m + i, y, m + i, y + bar)
        p.setPen(QColor("#b0b0b0"))
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        p.drawText(2, y + bar, "0")
        p.drawText(self.width() - 24, y + bar, "N-1")


class _CustomSchemeDialog(QDialog):
    """Edit one custom scheme: named list of (position, color) stops."""

    def __init__(self, name, stops, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("depth_custom_scheme_title"))
        self.setMinimumWidth(460)
        self.stops = [list(s) for s in (stops or cmap.default_stops())]

        v = QVBoxLayout(self)
        self.name_edit = QLineEdit(name)
        row = QHBoxLayout()
        row.addWidget(QLabel(trans.t("depth_scheme_name")))
        row.addWidget(self.name_edit, 1)
        v.addLayout(row)

        self.listw = QListWidget()
        self.listw.setMaximumHeight(150)
        v.addWidget(self.listw)

        btns = QHBoxLayout()
        add_btn = QPushButton(trans.t("depth_add_stop"))
        del_btn = QPushButton(trans.t("depth_del_stop"))
        add_btn.clicked.connect(lambda: self._add_stop())
        del_btn.clicked.connect(self._del_stop)
        btns.addWidget(add_btn)
        btns.addWidget(del_btn)
        btns.addStretch()
        v.addLayout(btns)

        self.preview = QLabel()
        self.preview.setMinimumHeight(18)
        v.addWidget(self.preview)

        close_row = QHBoxLayout()
        close_row.addStretch()
        ok_btn = QPushButton(trans.t("btn_ok"))
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton(trans.t("btn_cancel"))
        cancel_btn.clicked.connect(self.reject)
        close_row.addWidget(ok_btn)
        close_row.addWidget(cancel_btn)
        v.addLayout(close_row)
        self._refresh()

    def _add_stop(self):
        pos = 1.0 if not self.stops else round(min(1.0, self.stops[-1][0] + 0.25), 3)
        self.stops.append([pos, "#FFFFFF"])
        self._refresh()

    def _del_stop(self):
        if len(self.stops) > 1 and self.listw.currentRow() >= 0:
            del self.stops[self.listw.currentRow()]
            self._refresh()

    def _refresh(self):
        self.listw.clear()
        for pos, color in self.stops:
            item = QListWidgetItem(f"{pos:.2f}   {color}")
            item.setBackground(QColor(color))
            self.listw.addItem(item)
        try:
            lut = cmap.build_lut_from_stops(self.stops)
            strip = cmap.lut_preview_bgr(lut, width=256, height=16)
            rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                          3 * rgb.shape[1], QImage.Format.Format_RGB888)
            self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(
                256, 16, Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation))
        except Exception:
            self.preview.clear()

    def _edit_stop(self, item):
        """Double-click: edit this stop's position and color in one mini form."""
        row = self.listw.row(item)
        pos, color = self.stops[row]
        d = QDialog(self)
        d.setWindowTitle(trans.t("depth_edit_stop"))
        lay = QFormLayout(d)
        spin = QDoubleSpinBox(d)
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setValue(pos)
        color_btn = QPushButton(color)
        holder = {"color": color}

        def pick():
            c = QColorDialog.getColor(QColor(holder["color"]), d,
                                      trans.t("depth_pick_color"))
            if c.isValid():
                holder["color"] = c.name().upper()
                color_btn.setText(holder["color"])
                color_btn.setStyleSheet(f"background: {holder['color']};")

        color_btn.clicked.connect(pick)
        lay.addRow(trans.t("depth_stop_position"), spin)
        lay.addRow(trans.t("depth_stop_color"), color_btn)
        ok = QPushButton(trans.t("btn_ok"))
        ok.clicked.connect(d.accept)
        lay.addRow(ok)
        if d.exec():
            self.stops[row] = [round(float(spin.value()), 3), holder["color"]]
            self._refresh()


class DepthMapDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("depth_dialog_title"))
        self.resize(880, 620)
        self._apply_style()

        self.window_main = parent
        self._index01 = None
        self._map_method = None   # 生成当前预览图的方法（与方法下拉失步时提示）
        self._worker = None
        self._custom = self._load_custom_schemes()
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(self._refresh_preview)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # ---- preview side ----
        pv = QWidget()
        pvlay = QVBoxLayout(pv)
        pvlay.setContentsMargins(0, 0, 0, 0)
        self.preview_lbl = QLabel(trans.t("depth_no_map_yet"))
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setMinimumSize(420, 320)
        self.preview_lbl.setStyleSheet(
            "background: #161616; border: 1px solid #444; color: #888;")
        pvlay.addWidget(self.preview_lbl, 1)
        self.legend = _GradientBar()
        pvlay.addWidget(self.legend)
        splitter.addWidget(pv)

        # ---- controls side ----
        side = QWidget()
        form = QFormLayout(side)
        form.setContentsMargins(8, 0, 0, 0)

        self.method_combo = QComboBox()
        for code, key in _METHOD_KEYS:
            self.method_combo.addItem(trans.t(key), code)
        default = self._current_window_method()
        i = self.method_combo.findData(default)
        self.method_combo.setCurrentIndex(max(0, i))
        form.addRow(trans.t("depth_method"), self.method_combo)

        self.map_combo = QComboBox()
        self._fill_colormaps()
        form.addRow(trans.t("depth_colormap"), self.map_combo)

        edit_btn = QPushButton(trans.t("depth_edit_custom"))
        edit_btn.clicked.connect(self._edit_custom_scheme)
        row = QHBoxLayout()
        row.addWidget(edit_btn)
        self.invert_chk = QCheckBox(trans.t("depth_invert"))
        row.addWidget(self.invert_chk)
        form.addRow("", row)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setRange(0.4, 2.5)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(1.0)
        form.addRow(trans.t("depth_gamma"), self.gamma_spin)

        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(0, 5)
        self.smooth_spin.setToolTip(trans.t("depth_smooth_hint"))
        form.addRow(trans.t("depth_smooth"), self.smooth_spin)

        self.overlay_chk = QCheckBox(trans.t("depth_overlay"))
        self.overlay_chk.setToolTip(trans.t("depth_overlay_hint"))
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setRange(0.05, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(0.55)
        self.opacity_spin.setEnabled(False)
        self.overlay_chk.toggled.connect(self.opacity_spin.setEnabled)
        row2 = QHBoxLayout()
        row2.addWidget(self.overlay_chk)
        row2.addWidget(self.opacity_spin)
        form.addRow("", row2)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        form.addRow(self.status_lbl)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        layout.addWidget(splitter, 1)

        # ---- actions ----
        btns = QHBoxLayout()
        self.compute_btn = QPushButton(trans.t("depth_compute"))
        self.compute_btn.clicked.connect(self._compute)
        self.cancel_btn = QPushButton(trans.t("btn_cancel_compute"))
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setVisible(False)
        save_color = QPushButton(trans.t("depth_save_color"))
        save_color.clicked.connect(lambda: self._save(colorized=True))
        save_raw = QPushButton(trans.t("depth_save_raw"))
        save_raw.clicked.connect(lambda: self._save(colorized=False))
        close_btn = QPushButton(trans.t("btn_close"))
        close_btn.clicked.connect(self.reject)
        btns.addWidget(self.compute_btn)
        btns.addWidget(self.cancel_btn)
        btns.addStretch()
        btns.addWidget(save_color)
        btns.addWidget(save_raw)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        for w in (self.map_combo, self.invert_chk, self.gamma_spin,
                  self.smooth_spin, self.overlay_chk, self.opacity_spin):
            sig = getattr(w, "currentIndexChanged", None) or getattr(w, "valueChanged", None) or getattr(w, "toggled", None)
            if sig:
                sig.connect(self._refresh_preview)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)

        self._load_prefs()
        self._update_overlay_availability()
        # 打开即出图：经典方法秒级完成，直接算；DTCWT/AI 分钟级，留给用户决定
        if (self._index01 is None and self._images()
                and self.method_combo.currentData() in _FAST_AUTO_METHODS):
            self._compute()

    # ------------------------------------------------------------------
    def _apply_style(self):
        parent = self.parent()
        theme = getattr(parent, "ui_theme", "dark") if parent else "dark"
        if theme == "light":
            self.setStyleSheet("QDialog { background-color: #ffffff; } QLabel { color: #1f2328; }")
        else:
            from ui.styles import DIALOG_CONTROL_DARK_QSS
            self.setStyleSheet("QDialog { background-color: #2b2b2b; border: 1px solid #444; }"
                               "QLabel { color: #d0d0d0; }" + DIALOG_CONTROL_DARK_QSS)

    def _current_window_method(self):
        w = self.window_main
        checks = (("rb_gfg", "gfgfgf"), ("rb_d", "stackmffv4"),
                  ("rb_c", "dtcwt"), ("rb_b", "dct"))
        for attr, code in checks:
            rb = getattr(w, attr, None)
            if rb is not None and rb.isChecked():
                return code
        return "guided_filter"

    def _load_custom_schemes(self):
        try:
            raw = get_settings().value("depthmap/custom_schemes", "")
            data = json.loads(raw) if raw else []
            return {s["name"]: [(float(p), c) for p, c in s["stops"]] for s in data}
        except Exception:
            return {}

    def _save_custom_schemes(self):
        data = [{"name": k, "stops": v} for k, v in self._custom.items()]
        get_settings().setValue("depthmap/custom_schemes", json.dumps(data))

    def _fill_colormaps(self):
        self.map_combo.blockSignals(True)
        self.map_combo.clear()
        # 灰度是论文图的常见需求，但不在 cv2 伪彩表里，手动补上
        entries = [("gray", False), ("gray_r", False)] + cmap.builtin_choices()
        for cid, _rev in entries:
            lut = cmap.get_lut(cid)
            strip = cmap.lut_preview_bgr(lut)
            rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                          3 * rgb.shape[1], QImage.Format.Format_RGB888)
            self.map_combo.addItem(QIcon(QPixmap.fromImage(qimg)), cid, cid)
        for name in sorted(self._custom):
            cid = f"custom:{name}"
            lut = cmap.build_lut_from_stops(self._custom[name])
            strip = cmap.lut_preview_bgr(lut)
            rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                          3 * rgb.shape[1], QImage.Format.Format_RGB888)
            self.map_combo.addItem(QIcon(QPixmap.fromImage(qimg)), name, cid)
        self.map_combo.blockSignals(False)

    def _selected_colormap(self):
        cid = self.map_combo.currentData()
        stops = None
        if cid and cid.startswith("custom:"):
            stops = self._custom.get(cid.split(":", 1)[1])
        return cid or _DEFAULT_METHOD_COLORMAP, stops

    def _edit_custom_scheme(self):
        dlg = _CustomSchemeDialog(trans.t("depth_new_scheme_default"),
                                  cmap.default_stops(), self)
        dlg.listw.itemDoubleClicked.connect(dlg._edit_stop)
        if dlg.exec():
            name = dlg.name_edit.text().strip() or trans.t("depth_new_scheme_default")
            self._custom[name] = [(float(p), c) for p, c in dlg.stops]
            self._save_custom_schemes()
            self._fill_colormaps()
            i = self.map_combo.findData(f"custom:{name}")
            if i >= 0:
                self.map_combo.setCurrentIndex(i)
            self._refresh_preview()

    # ------------------------------------------------------------------
    def _on_method_changed(self):
        """预览图还是旧方法的——明确告诉用户，避免拿错图。"""
        if (self._index01 is not None and self._map_method is not None
                and self.method_combo.currentData() != self._map_method
                and not (self._worker and self._worker.isRunning())):
            self.status_lbl.setText(trans.t("depth_map_stale"))

    def _update_overlay_availability(self):
        base = getattr(self.window_main, "fusion_result", None) if self.window_main else None
        has_result = base is not None
        if not has_result:
            self.overlay_chk.setChecked(False)
        self.overlay_chk.setEnabled(has_result)
        hint = (trans.t("depth_overlay_hint") if has_result
                else trans.t("depth_overlay_missing"))
        self.overlay_chk.setToolTip(hint)
        self.opacity_spin.setEnabled(has_result and self.overlay_chk.isChecked())

    def resizeEvent(self, event):
        """拖大/最大化窗口后预览要跟着重排（防抖 120ms）。"""
        super().resizeEvent(event)
        self._resize_timer.start()

    def _images(self):
        imgs = getattr(self.window_main, "raw_images", None) or []
        return list(imgs) if len(imgs) >= 2 else None

    def _compute(self):
        imgs = self._images()
        if not imgs:
            self.status_lbl.setText(trans.t("depth_need_stack"))
            return
        method = self.method_combo.currentData()
        use_gpu = bool(getattr(self.window_main, "use_gpu", False))
        self._worker = DepthWorker(imgs, method, use_gpu, self)
        self._worker_method = method
        self._worker.progress.connect(lambda i, n: self.status_lbl.setText(
            trans.t("depth_computing").format(i=i, n=n)))
        self._worker.ok.connect(self._computed)
        self._worker.failed.connect(self._failed)
        self.compute_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        self.status_lbl.setText(trans.t("depth_computing").format(i=0, n=len(imgs)))
        self._worker.start()

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancel()

    def _computed(self, index01):
        self._index01 = np.asarray(index01, np.float32)
        self._map_method = getattr(self, "_worker_method", None)
        self._worker = None
        self.compute_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.status_lbl.setText(trans.t("depth_ready"))
        self._refresh_preview()

    def _failed(self, msg):
        self._worker = None
        self.compute_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.status_lbl.setText("" if msg == "cancelled" else msg)

    # ------------------------------------------------------------------
    def _colorized_full(self):
        from core.depth_map import smooth_index_map
        cid, stops = self._selected_colormap()
        m = smooth_index_map(self._index01, self.smooth_spin.value())
        colored = cmap.colorize(m, cid, invert=self.invert_chk.isChecked(),
                                gamma=self.gamma_spin.value(),
                                custom_stops=stops)
        if self.overlay_chk.isChecked():
            base = getattr(self.window_main, "fusion_result", None)
            if base is not None and base.shape[:2] == colored.shape[:2]:
                a = float(self.opacity_spin.value())
                base_f = base.astype(np.float32)
                col_f = colored.astype(np.float32)
                if base_f.ndim == 2:
                    base_f = cv2.cvtColor(base_f, cv2.COLOR_GRAY2BGR)
                out = base_f * (1.0 - a) + col_f * a
                return np.clip(out, 0, 255).astype(np.uint8)
        return colored

    def _refresh_preview(self):
        try:
            lut = cmap.get_lut(*self._selected_colormap())
        except Exception:
            lut = cmap.get_lut(_DEFAULT_METHOD_COLORMAP)
        self.legend.set_lut(lut)
        if self._index01 is None:
            self.legend.update()
            return
        img = self._colorized_full()
        h, w = img.shape[:2]
        target = self.preview_lbl.size()
        if target.width() < 40 or target.height() < 40:
            return  # 布局尚未定型（首次显示前），resizeEvent 会再触发
        scale = min(target.width() / w, target.height() / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      3 * rgb.shape[1], QImage.Format.Format_RGB888)
        self.preview_lbl.setPixmap(QPixmap.fromImage(qimg))

    def _save(self, colorized=True):
        if self._index01 is None:
            self.status_lbl.setText(trans.t("depth_no_map_yet"))
            return
        start_dir = get_last_dialog_dir()
        if colorized:
            path, _ = QFileDialog.getSaveFileName(
                self, trans.t("depth_save_color"),
                self._default_path(start_dir), "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif *.tiff);;BMP (*.bmp)")
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, trans.t("depth_save_raw"),
                self._default_path(start_dir, raw=True), "TIFF 16-bit (*.tif *.tiff)")
        if not path:
            return
        set_last_dialog_dir(path)
        try:
            if colorized:
                imwrite_auto(path, self._colorized_full())
            else:
                u16 = np.clip(self._index01 * 65535.0, 0, 65535).round().astype(np.uint16)
                imwrite_auto(path, u16)
            self.status_lbl.setText(trans.t("depth_saved").format(path=path))
        except Exception as exc:
            self.status_lbl.setText(f"{exc}")

    def _default_path(self, start_dir, raw=False):
        base = "depth_map" + ("_raw16" if raw else "")
        ext = ".tif" if raw else ".png"
        return start_dir.rstrip("/\\") + "/" + base + ext if start_dir else base + ext

    # ------------------------------------------------------------------
    def _load_prefs(self):
        s = get_settings()
        cid = s.value("depthmap/colormap", _DEFAULT_METHOD_COLORMAP)
        i = self.map_combo.findData(cid)
        if i >= 0:
            self.map_combo.setCurrentIndex(i)
        self.gamma_spin.setValue(float(s.value("depthmap/gamma", 1.0)))
        self.smooth_spin.setValue(int(s.value("depthmap/smooth", 0)))
        self.invert_chk.setChecked(s.value("depthmap/invert", False, type=bool))
        self._refresh_preview()

    def _save_prefs(self):
        s = get_settings()
        s.setValue("depthmap/colormap", self.map_combo.currentData())
        s.setValue("depthmap/gamma", self.gamma_spin.value())
        s.setValue("depthmap/smooth", self.smooth_spin.value())
        s.setValue("depthmap/invert", self.invert_chk.isChecked())

    def _detach_worker(self):
        """关闭对话框时后台线程可能正处在长推理中——QThread 被销毁而线程
        仍在运行会让整个进程 abort。让它脱离对话框的生命周期，取消后
        自行跑完/退出并自我清理；信号在接收者销毁后由 Qt 自动断开。"""
        w = self._worker
        self._worker = None
        if w is None or not w.isRunning():
            return
        w.cancel()
        w.setParent(None)
        _ORPHAN_WORKERS.add(w)
        w.finished.connect(w.deleteLater)
        w.finished.connect(lambda ww=w: _ORPHAN_WORKERS.discard(ww))

    def reject(self):
        self._detach_worker()
        self._save_prefs()
        super().reject()

    def closeEvent(self, event):
        self._detach_worker()
        self._save_prefs()
        super().closeEvent(event)
