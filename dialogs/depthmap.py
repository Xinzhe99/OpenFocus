"""Depth-map viewer/exporter with pseudo-color customization.

Computes a focus-position map with the selected algorithm's own activity
measure (core.depth_map), previews it colorized (optionally overlaid on the
fused result), and exports either the colorized image or the raw 16-bit
index map. Custom color schemes are interpolated from user color stops and
persisted in QSettings.
"""
import json
import os

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
    QMessageBox,
    QSlider,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from locales import trans
from widgets.stepper import StepperSpinBox
from utils import colormap as cmap
from utils.image_utils import imwrite_auto
from utils.settings_store import get_last_dialog_dir, get_settings, set_last_dialog_dir

# 仅收录"按索引决策"的融合方法（与融合下拉同序；加权平均类无逐像素
# 决策，不提供深度图——guided_filter / openfocus_a 不在此列）
_METHOD_KEYS = (
    ("stackmffv3", "radio_stackmff"),
    ("openfocus_b", "radio_openfocus_b"),
    ("openfocus_c", "radio_openfocus_c"),
    ("dct", "radio_dct"),
    ("dtcwt", "radio_dtcwt"),
    ("gfgfgf", "radio_gfg"),
)

_DEFAULT_METHOD_COLORMAP = "turbo"

# 打开对话框时自动计算的方法（秒级）；dtcwt/stackmffv3 分钟级，不自动跑
_FAST_AUTO_METHODS = {"dct", "gfgfgf", "openfocus_b", "openfocus_c"}

# 对话框关闭时仍在跑的 worker 放这里保活：QThread 对象被销毁而线程仍在
# 运行会直接 abort 整个进程（AI 推理可达数分钟，用户关窗很正常）。
_ORPHAN_WORKERS = set()


class DepthWorker(QThread):
    progress = pyqtSignal(int, int)
    ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, images, method, use_gpu, kernel_size=7, parent=None):
        super().__init__(parent)
        self._images = images
        self._method = method
        self._use_gpu = use_gpu
        self._kernel_size = int(kernel_size) if kernel_size else 7
        self._cancelled = False
        self.nlevels = 4
        self.source_frames = ()
        self.valid_mask = None
        self.alignment = "none"
        self.registration_settings = {}
        self.reverse_order = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from core.resource_policy import heavy_job
        from core.registration import RegistrationCancelled
        try:
            with heavy_job(lambda: self._cancelled):
                self._run_admitted()
        except RegistrationCancelled:
            self.failed.emit("cancelled")

    def _run_admitted(self):
        from core.depth_map import DepthMapCancelled, compute_focus_index
        from core.registration import RegistrationCancelled
        try:
            if self.alignment != "none":
                from core.compare_worker import align_frames
                self._images = align_frames(self._images, self.alignment,
                    self.registration_settings, lambda: self._cancelled)
                if self._images is None:
                    self.failed.emit("cancelled")
                    return
                from core.render_context import SourceFrame
                self.source_frames = tuple(SourceFrame(str(i), "", image, aligned=True)
                                           for i, image in enumerate(self._images))
            if self.reverse_order:
                self._images = list(reversed(self._images))
                self.source_frames = tuple(reversed(self.source_frames))
            model_path = None
            if self._method == "stackmffv3":
                from core.cli import _find_model_path
                model_path = _find_model_path()
            m = compute_focus_index(
                self._images, self._method,
                # 与融合管线同参数：否则 DCT 的中值窗口 / GFG-FGF 的模糊窗口
                # 永远是默认的 7，而界面上用户调的是别的值
                kernel_size=self._kernel_size,
                model_path=model_path, use_gpu=self._use_gpu,
                progress_callback=lambda i, n: self.progress.emit(i, n),
                should_cancel=lambda: self._cancelled,
                nlevels=self.nlevels,
                thread_count=self.registration_settings.get("thread_count", 0),
                resource_preference=self.registration_settings.get("resource_preference", "balanced"),
            )
            for frame in self.source_frames:
                if self._cancelled:
                    self.failed.emit("cancelled")
                    return
                valid = frame.valid_mask()
                if valid is not None:
                    self.valid_mask = valid if self.valid_mask is None else self.valid_mask & valid
            if self._cancelled:
                self.failed.emit("cancelled")
                return
            self.ok.emit(m)
        except (DepthMapCancelled, RegistrationCancelled):
            self.failed.emit("cancelled")
        except Exception as exc:  # surface to the dialog status line
            self.failed.emit(str(exc))

class MeshExportWorker(QThread):
    ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path, index, texture, options, parent=None):
        super().__init__(parent)
        self.path, self.index, self.texture, self.options = path, index, texture, options

    def cancel(self):
        self.requestInterruption()

    def run(self):
        from utils.mesh_export import export_obj, MeshExportCancelled
        try:
            self.ok.emit(export_obj(self.path, self.index, self.texture,
                should_cancel=self.isInterruptionRequested, **self.options))
        except MeshExportCancelled:
            self.failed.emit("cancelled")
        except Exception as exc:
            self.failed.emit(str(exc))



class _GradientBar(QLabel):
    """Thin color legend: LUT gradient with endpoint labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lut = None
        self._right = "N-1"
        self._left = "1"
        self._middle = ""
        self.setMinimumHeight(36)
        self.setText("")

    def set_lut(self, lut, right_label="N", left_label="1", middle_label=""):
        self._lut = lut
        self._right = right_label
        self._left, self._middle = str(left_label), str(middle_label)
        self.setToolTip("")
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        # Follow the active theme: a hard-coded near-black strip looked like a
        # hole in the light dialog.
        from ui.styles import CURRENT_THEME
        from utils.design_tokens import palette
        p.fillRect(self.rect(), QColor(palette(CURRENT_THEME)["surfaceSunken"]))
        if self._lut is None:
            return
        m = 12  # side margin for labels
        bar = max(1, self.height() - 18)
        y = 0
        n = max(1, self.width() - 2 * m)
        for i in range(n):
            c = self._lut[int(i / n * 255)]
            p.setPen(QColor(c[2], c[1], c[0]))
            p.drawLine(m + i, y, m + i, y + bar)
        p.setPen(QColor(palette(CURRENT_THEME)["textMuted"]))
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        label_y = self.height() - 3
        p.drawText(2, label_y, self._left)
        middle_width = p.fontMetrics().horizontalAdvance(self._middle)
        p.drawText((self.width() - middle_width) // 2, label_y, self._middle)
        w_label = p.fontMetrics().horizontalAdvance(self._right)
        p.drawText(self.width() - 8 - w_label, label_y, self._right)


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
        ok_btn.setObjectName("primaryButton")
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
        self._map_request = 0     # revision of the request that produced the map
        self._request = 0         # latest requested revision
        self._stale = False       # map belongs to older parameters
        self._cancelling = False
        self._worker = None
        self._closing = False
        self._pending_compute = False
        self._worker_outcome = None
        self._export_worker = None
        self._mesh_height = None
        self._mesh_texture = None
        self._map_valid_mask = None
        self._map_frames = []
        self._map_indices = []
        self._capture_sources()
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
        # Preview placeholder follows the palette instead of a fixed near-black.
        from ui.styles import CURRENT_THEME
        from utils.design_tokens import palette
        _p = palette(CURRENT_THEME)
        self.preview_lbl.setStyleSheet(
            f"background: {_p['surfaceSunken']};"
            f" border: 1px solid {_p['border']};"
            f" color: {_p['textMuted']};")
        # The 3D surface lives on its own tab: it answers a different question
        # ("what does this relief look like?") than the colourised map, and it
        # used to require exporting an OBJ and opening another program.
        from PyQt6.QtWidgets import QTabWidget
        from widgets.depth_surface import DepthSurface
        self.tabs = QTabWidget()
        map_page = QWidget()
        map_lay = QVBoxLayout(map_page)
        map_lay.setContentsMargins(0, 0, 0, 0)
        map_lay.addWidget(self.preview_lbl, 1)
        self.tabs.addTab(map_page, trans.t("depth_tab_map"))

        surf_page = QWidget()
        surf_lay = QVBoxLayout(surf_page)
        surf_lay.setContentsMargins(0, 0, 0, 0)
        self.surface = DepthSurface()
        surf_lay.addWidget(self.surface, 1)
        # Detail slider: the surface is sampled at a working resolution to
        # stay interactive; this exposes that trade-off instead of hiding it.
        detail_row = QHBoxLayout()
        detail_row.setContentsMargins(0, 0, 0, 0)
        self.detail_slider = QSlider(Qt.Orientation.Horizontal)
        self.detail_slider.setRange(20, 100)
        self.detail_slider.setSingleStep(5)
        self.detail_slider.setPageStep(10)
        self.detail_slider.setValue(60)
        self.detail_slider.setToolTip(trans.t("depth_3d_detail_hint"))
        self.detail_value = QLabel("0.60x")
        self.detail_value.setMinimumWidth(34)
        self.detail_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.detail_slider.valueChanged.connect(
            lambda v: (self.detail_value.setText(f"{v / 100.0:.2f}x"),
                       self.surface.set_detail(v)))
        self.lbl_surface_state = QLabel("")
        self.lbl_surface_state.setObjectName("mutedLabel")
        self.lbl_surface_state.setWordWrap(True)
        surf_lay.addWidget(self.lbl_surface_state)
        detail_row.addWidget(QLabel(trans.t("depth_3d_detail")))
        detail_row.addWidget(self.detail_slider, 1)
        detail_row.addWidget(self.detail_value)
        surf_lay.addLayout(detail_row)
        self.btn_surface_reset = QPushButton(trans.t("depth_3d_reset"))
        self.btn_surface_reset.clicked.connect(self.surface.reset_view)
        surf_lay.addWidget(self.btn_surface_reset)
        self.tabs.addTab(surf_page, trans.t("depth_tab_3d"))
        pvlay.addWidget(self.tabs, 1)

        self.legend = _GradientBar()
        pvlay.addWidget(self.legend)
        splitter.addWidget(pv)

        # ---- controls side ----
        side = QWidget()
        form = QFormLayout(side)
        form.setContentsMargins(8, 0, 0, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.source_combo = QComboBox()
        self.source_combo.addItem(trans.t("depth_source_current"), "current")
        self.source_combo.addItem(trans.t("depth_source_from_output"), "output")
        self.source_combo.setCurrentIndex(1 if self._output_selected else 0)
        self._source_mode = self.source_combo.currentData()
        form.addRow(trans.t("depth_source_label"), self.source_combo)
        self.align_combo = QComboBox()
        for code, key in (("none", "check_align_none"), ("homography", "check_align_homography"),
                          ("ecc", "check_align_ecc"), ("openfocus", "check_align_of")):
            self.align_combo.addItem(trans.t(key), code)
        self.align_combo.setEnabled(self._source_mode == "current")
        form.addRow(trans.t("group_registration"), self.align_combo)

        self.method_combo = QComboBox()
        for code, key in _METHOD_KEYS:
            self.method_combo.addItem(trans.t(key), code)
        default = self._current_window_method()
        i = self.method_combo.findData(default)
        self.method_combo.setCurrentIndex(max(0, i))
        form.addRow(trans.t("depth_method"), self.method_combo)

        # kernel 含义随方法不同（与主面板滑块同源语义）——只读说明行
        self.lbl_kernel_info = QLabel("")
        # Muted but readable: #999 on white is ~2.8:1, below the 4.5:1 needed
        # Muted but readable: uses the palette's secondary text colour.
        from ui.styles import CURRENT_THEME
        from utils.design_tokens import palette
        self.lbl_kernel_info.setStyleSheet(
            f"color: {palette(CURRENT_THEME)['textMuted']};")
        form.addRow("", self.lbl_kernel_info)
        self._update_kernel_info()

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

        self.gamma_spin = StepperSpinBox(minimum=0.4, maximum=2.5, decimals=2, step=0.05, value=1.0)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setRange(0.4, 2.5)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(1.0)
        form.addRow(trans.t("depth_gamma"), self.gamma_spin)

        self.smooth_spin = StepperSpinBox(minimum=0, maximum=5, decimals=0, step=1)
        self.smooth_spin.setRange(0, 5)
        self.smooth_spin.setToolTip(trans.t("depth_smooth_hint"))
        form.addRow(trans.t("depth_smooth"), self.smooth_spin)

        self.relief_spin = StepperSpinBox(minimum=1, maximum=100, decimals=0, step=1, value=15)
        self.relief_spin.setRange(1, 100)
        self.relief_spin.setSuffix(" %")
        self.relief_spin.setValue(15)
        self.relief_spin.setToolTip(trans.t("depth_relief_hint"))
        self.relief_spin.valueChanged.connect(
            lambda v: self.surface.set_relief(v / 100.0)
            if hasattr(self, "surface") else None)
        form.addRow(trans.t("depth_relief"), self.relief_spin)

        self.physical_chk = QCheckBox(trans.t("depth_physical"))
        form.addRow(self.physical_chk)
        self.z_start = StepperSpinBox(minimum=-1e9, maximum=1e9, decimals=9, step=1)
        self.z_step = StepperSpinBox(minimum=0, maximum=1e9, decimals=9, step=1)
        self.z_unit = QComboBox()
        for unit, factor in (("µm", 1.0), ("nm", 0.001), ("mm", 1000.0)):
            self.z_unit.addItem(unit, factor)
        self.z_direction = QComboBox()
        self.z_direction.addItem("+Z", 1)
        self.z_direction.addItem("−Z", -1)
        form.addRow(trans.t("depth_z_start"), self.z_start)
        form.addRow(trans.t("depth_z_step"), self.z_step)
        form.addRow(self.z_unit, self.z_direction)
        self.z_mapping = QCheckBox(trans.t("depth_z_mapping"))
        self.z_mapping.setToolTip(trans.t("depth_axis_hint"))
        form.addRow(self.z_mapping)
        self.mesh_size = StepperSpinBox(minimum=2, maximum=4096, decimals=0,
                                        step=20, value=600)
        form.addRow(trans.t("depth_mesh_size"), self.mesh_size)
        self.texture_combo = QComboBox()
        self._fill_textures()
        form.addRow(trans.t("depth_texture"), self.texture_combo)
        self.btn_save_physical = QPushButton(trans.t("depth_save_physical"))
        self.btn_save_physical.clicked.connect(self._save_physical)
        form.addRow(self.btn_save_physical)
        for control in (self.physical_chk, self.z_start, self.z_step, self.z_unit,
                        self.z_direction, self.z_mapping, self.mesh_size, self.texture_combo):
            signal = (getattr(control, "currentIndexChanged", None)
                      or getattr(control, "valueChanged", None) or control.toggled)
            signal.connect(self._refresh_preview)
        self.overlay_chk = QCheckBox(trans.t("depth_overlay"))
        self.overlay_chk.setToolTip(trans.t("depth_overlay_hint"))
        self.opacity_spin = StepperSpinBox(minimum=0.05, maximum=1.0, decimals=2, step=0.05, value=0.55)
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

        self.lbl_source = QLabel("")
        self.lbl_source.setObjectName("mutedLabel")
        self.lbl_source.setWordWrap(True)
        form.addRow(self.lbl_source)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        form.addRow(self.status_lbl)
        for combo in side.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)


        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setWidget(side)
        splitter.addWidget(side_scroll)
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
        self.btn_save_color = QPushButton(trans.t("depth_save_color"))
        self.btn_save_color.clicked.connect(lambda: self._save(colorized=True))
        self.btn_save_raw = QPushButton(trans.t("depth_save_raw"))
        self.btn_save_raw.clicked.connect(lambda: self._save(colorized=False))
        self.btn_save_3d = QPushButton(trans.t("depth_export_3d"))
        self.btn_save_3d.setToolTip(trans.t("depth_export_3d_hint"))
        self.btn_save_3d.clicked.connect(self._export_3d)
        close_btn = QPushButton(trans.t("btn_close"))
        close_btn.clicked.connect(self.reject)
        btns.addWidget(self.compute_btn)
        btns.addWidget(self.cancel_btn)
        btns.addStretch()
        btns.addWidget(self.btn_save_color)
        btns.addWidget(self.btn_save_raw)
        btns.addWidget(self.btn_save_3d)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        for w in (self.map_combo, self.invert_chk, self.gamma_spin,
                  self.smooth_spin, self.overlay_chk, self.opacity_spin):
            sig = getattr(w, "currentIndexChanged", None) or getattr(w, "valueChanged", None) or getattr(w, "toggled", None)
            if sig:
                sig.connect(self._refresh_preview)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.align_combo.currentIndexChanged.connect(self._on_method_changed)

        self._load_prefs()
        self._update_overlay_availability()
        self._update_source_summary()
        self._update_export_state()
        # 打开即出图：经典方法秒级完成，直接算；DTCWT/AI 分钟级，留给用户决定
        if (self._index01 is None and self._images()
                and self.method_combo.currentData() in _FAST_AUTO_METHODS):
            self._compute()

    # ------------------------------------------------------------------
    def _apply_style(self):
        """Styling comes from the generated application stylesheet."""
        self.setStyleSheet("")

    def _capture_sources(self):
        win = self.window_main
        effective = getattr(win, "effective_stack", None)
        if callable(effective):
            images, names, indices = effective()
        else:
            images = getattr(win, "raw_images", None) or []
            names = getattr(win, "image_filenames", None) or []
            indices = range(len(images))
        self._current_images = list(images)
        self._current_names = list(names)
        self._current_indices = list(indices)
        self._total_frames = len(getattr(win, "raw_images", None) or [])
        self._reverse_order = bool(getattr(win, "stack_reverse_order", False))
        self._use_gpu = bool(getattr(win, "use_gpu", False))
        self._nlevels = int(getattr(win, "dtcwt_levels", 4))
        self._registration_settings = {"thread_count": int(getattr(win, "thread_count", 4)),
            "resource_preference": getattr(win, "resource_preference", "balanced"),
            "use_gpu": self._use_gpu,
            "reg_downscale_width": int(getattr(win, "reg_downscale_width", 1024)),
            "exposure_match": bool(getattr(win, "exposure_match", False))}
        getter = getattr(win, "kernel_for_method", None)
        from utils.scalebar import effective_px_um
        self._current_pixel_size = effective_px_um(win)
        self._kernels = {method: int(getter(method)) if callable(getter) else 7
                         for method, _key in _METHOD_KEYS}
        output_list = getattr(win, "output_list", None)
        selected = output_list.selectedItems() if output_list is not None else []
        self._output_selected = bool(selected)
        manager = getattr(win, "output_manager", None)
        self._source_record = (manager.record_for_item(selected[0])
                               if manager is not None and len(selected) == 1 else None)
        self._output_context = getattr(self._source_record, "context", None)
        self._output_texture = getattr(self._source_record, "image", None)
        self._source_mode = "output" if selected else "current"

    def _source_context(self):
        return self._output_context if self._source_mode == "output" else None

    def _on_source_changed(self):
        self._source_mode = self.source_combo.currentData()
        self._on_method_changed()
        self.align_combo.setEnabled(self._source_mode == "current")
        self.z_mapping.setChecked(False)
        self._fill_textures()
        self._update_source_summary()
        self._update_overlay_availability()

    def _fill_textures(self):
        self.texture_combo.blockSignals(True)
        self.texture_combo.clear()
        self.texture_combo.addItem(trans.t("depth_colormap"), "color")
        if self._source_context() is not None:
            self.texture_combo.addItem(trans.t("depth_source_from_output"), "output")
        for index in range(len(self._images() or [])):
            self.texture_combo.addItem(trans.t("compare_frame_of").format(
                i=index + 1, n=len(self._images()), name=""), index)
        self.texture_combo.blockSignals(False)

    def _capture_indices(self):
        context = self._source_context()
        return list(context.source_indices) if context is not None else list(self._current_indices)

    def _physical_values(self, index):
        from utils.mesh_export import physical_depth
        if (not self.z_mapping.isChecked() or len(self._map_indices) != len(self._map_frames)
                or self.z_step.value() <= 0):
            raise ValueError(trans.t("depth_physical_missing"))
        unit = self.z_unit.currentData()
        if self._map_valid_mask is not None:
            index = np.where(self._map_valid_mask, index, np.nan)
        return physical_depth(index, self._map_indices, self.z_start.value() * unit,
                              self.z_step.value() * unit, self.z_direction.currentData())

    def _pixel_size(self):
        context = self._source_context()
        return context.pixel_size_um if context is not None else self._current_pixel_size

    def _texture_for_map(self, index):
        choice = self.texture_combo.currentData()
        if choice == "output":
            texture = self._matching_texture()
            if texture is None:
                raise ValueError(trans.t("depth_overlay_missing"))
            return texture
        if isinstance(choice, int):
            return self._map_frames[choice]
        cid, stops = self._selected_colormap()
        return cmap.colorize(index, cid, invert=self.invert_chk.isChecked(),
                             gamma=self.gamma_spin.value(), custom_stops=stops)

    def _save_physical(self):
        if not self._require_current_map():
            return
        try:
            depth, positions = self._physical_values(self._index01)
            context = self._source_context()
            if context is not None and context.roi_mode == "paste":
                frame = context.frames[0]
                full = np.full((context.height, context.width), np.nan, np.float32)
                full[frame.offset_y:frame.offset_y + depth.shape[0],
                     frame.offset_x:frame.offset_x + depth.shape[1]] = depth
                depth = full
            path, _ = QFileDialog.getSaveFileName(self, trans.t("depth_save_physical"),
                self._default_path(get_last_dialog_dir(), raw=True), "TIFF (*.tif *.tiff)")
            if not path:
                return
            if os.path.splitext(path)[1].lower() not in (".tif", ".tiff"):
                raise ValueError(trans.t("depth_raw_needs_tiff"))
            from utils.image_utils import imwrite_float_tiff
            if not imwrite_float_tiff(path, depth, {"unit": "um", "z_positions_um": positions,
                    "missing": "NaN", "source_indices": self._map_indices,
                    "method": self._map_method, "domain": self._domain_metadata()}):
                raise IOError(trans.t("msg_save_failed_text"))
            self.status_lbl.setText(trans.t("depth_saved").format(path=path))
        except Exception as exc:
            self.status_lbl.setText(str(exc))

    def _domain_metadata(self):
        context = self._source_context()
        frame = context.frames[0] if context is not None and context.frames else None
        return {"offset": [frame.offset_x, frame.offset_y] if frame else [0, 0],
                "shape": list(self._index01.shape) if self._index01 is not None else None}

    def _current_window_method(self):
        w = self.window_main
        keys = {code for code, _k in _METHOD_KEYS}
        context = self._source_context()
        if context is not None and context.algorithm in keys:
            return context.algorithm
        # 当前融合方法有深度侧写就用它；否则（仅对齐 / 加权平均类方法 /
        # 无方法）回退到列表第一项
        helper = getattr(w, "current_fusion_method", None)
        if callable(helper):
            code = helper()
            if code in keys:
                return code
        return _METHOD_KEYS[0][0]

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
    def _update_source_summary(self) -> None:
        """Show the frames and output the current map is based on."""
        label = getattr(self, "lbl_source", None)
        if label is None:
            return
        images = self._images() or []
        source = (trans.t("depth_source_from_output") if self._source_mode == "output"
                  else trans.t("depth_source_current"))
        total = self._total_frames if self._source_mode == "current" else len(images)
        parts = [source, f"{len(images)}/{total}"]
        if self._source_mode == "current":
            parts.append(self.align_combo.currentText())
        context = self._source_context()
        if context is not None:
            parts.extend((self._source_record.label, context.alignment))
            label.setToolTip("\n".join(frame.name for frame in context.frames))
        else:
            label.setToolTip("\n".join(self._current_names))
        label.setText(" · ".join(parts) if images else
                      trans.t("depth_need_two_frames").format(n=len(images)))
        if self._source_mode == "output" and context is None:
            label.setText(trans.t("retouch_source_not_recorded"))

    def _mark_stale(self):
        if self._index01 is None:
            return
        self._stale = True
        self.status_lbl.setText(trans.t("depth_stale"))
        if hasattr(self, "surface"):
            try:
                self.surface.set_surface(None, None)
            except Exception:
                pass
        self._update_export_state()

    def _on_method_changed(self):
        self._update_kernel_info()
        self._update_source_summary()
        self._request += 1
        self._mark_stale()
        self._pending_compute = (self.method_combo.currentData() in _FAST_AUTO_METHODS
                                 and bool(self._images()))
        if self._worker is not None:
            self._cancelling = True
            self._worker.cancel()
            self.status_lbl.setText(trans.t("depth_cancelling"))
        elif self._pending_compute:
            self._pending_compute = False
            self._compute()
        self._update_export_state()

    _KERNEL_INFO_KEYS = {"dct": "label_kernel_dct",
                         "gfgfgf": "label_kernel_gfg",
                         "openfocus_b": "label_kernel_of",
                         "openfocus_c": "label_kernel_of"}

    def _update_kernel_info(self):
        """kernel 在不同方法里含义不同（与主面板滑块同源）——只读展示。"""
        method = self.method_combo.currentData()
        if method == "stackmffv3":
            # No kernel to show. State what the map actually is: the network
            # picks a source frame per pixel (a classifier is confident at
            # almost every pixel), so the values are still frame positions.
            self.lbl_kernel_info.setText(trans.t("depth_ai_note"))
            self.lbl_kernel_info.setToolTip(trans.t("depth_ai_note_hint"))
            return
        self.lbl_kernel_info.setToolTip("")
        key = self._KERNEL_INFO_KEYS.get(method)
        if not key:
            self.lbl_kernel_info.setText("")
            return
        k = self._kernel_for(self.method_combo.currentData())
        self.lbl_kernel_info.setText(f"{trans.t(key)} {k}")

    def _kernel_for(self, method):
        """The kernel the user set for *this* method in the main window (each
        method remembers its own slider value), forced odd."""
        k = self._kernels.get(method, 7)
        context = self._source_context()
        parameters = getattr(context, "parameters", {}) or {}
        if context is not None and context.algorithm == method:
            k = int(parameters.get("kernel_size", k))
        return k + 1 if k % 2 == 0 else k

    def _update_overlay_availability(self):
        base = self._matching_texture()
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
        if self._source_mode == "output":
            context = self._source_context()
            frames = [frame.image for frame in context.frames] if context else []
        else:
            frames = self._current_images
        return frames if len(frames) >= 2 else None

    def _compute(self):
        # One worker at a time: a second _compute would overwrite the reference
        # and leave the first worker parented to this dialog, so closing it
        # aborted the process ("QThread destroyed while running").
        if self._closing or self._worker is not None or self._export_worker is not None:
            return
        imgs = self._images()
        if not imgs:
            self.status_lbl.setText(trans.t("depth_need_stack"))
            return
        method = self.method_combo.currentData()
        context = self._source_context()
        reverse = (context.parameters.get("sort_mode") == "desc" if context is not None
                   else self._reverse_order)
        self._request_indices = self._capture_indices()
        self._request_names = ([frame.name for frame in context.frames] if context
                               else list(self._current_names))
        if method in ("openfocus_b", "openfocus_c") and reverse:
            self._request_indices.reverse()
            self._request_names.reverse()
        use_gpu = self._use_gpu
        # The depth map uses the kernel the user set for that method (DCT's
        # median window / GFG-FGF's blur / variance window), never another
        # method's slider value.
        kernel_size = self._kernel_for(method)
        self._request += 1
        request = self._request
        self._cancelling = False
        self._stale = self._index01 is not None
        self._worker_outcome = None
        self._worker = DepthWorker(imgs, method, use_gpu, kernel_size, self)
        self._worker.reverse_order = method in ("openfocus_b", "openfocus_c") and reverse
        self._worker.nlevels = int(context.parameters.get("N", self._nlevels)) if context else self._nlevels
        self._worker.source_frames = context.frames if context is not None else ()
        self._worker.alignment = self.align_combo.currentData() if context is None else "none"
        self._worker.registration_settings = self._registration_settings
        self._worker.request = request
        self._worker_method = method
        self._worker.progress.connect(
            lambda i, n, req=request: self._progress(i, n, req))
        self._worker.ok.connect(lambda index, req=request: self._computed(index, req))
        self._worker.failed.connect(lambda msg, req=request: self._failed(msg, req))
        self._worker.finished.connect(self._worker_finished)
        self.compute_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        if method == "stackmffv3":
            self.status_lbl.setText(trans.t("depth_computing_ai"))
        else:
            self.status_lbl.setText(
                trans.t("depth_computing").format(i=0, n=len(imgs)))
        self._update_export_state()
        self._worker.start()

    def _cancel(self):
        self._pending_compute = False
        if self._export_worker is not None:
            self._export_worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.status_lbl.setText(trans.t("depth_cancelling"))
            return
        if self._worker is None:
            return
        # A native step cannot be interrupted; say so instead of pretending
        # the map is gone. The result is discarded when it returns.
        self._cancelling = True
        self.cancel_btn.setEnabled(False)
        self.status_lbl.setText(trans.t("depth_cancelling"))
        self._worker.cancel()

    def _progress(self, i, n, request):
        if self._closing or self._cancelling or request != self._request:
            return
        self.status_lbl.setText(trans.t("depth_computing_ai")
            if self._worker_method == "stackmffv3" else
            trans.t("depth_computing").format(i=i, n=n))

    def _computed(self, index01, request=None):
        if (self._closing or self._cancelling
                or (request is not None and request != self._request)):
            return
        self._worker_outcome = ("ok", np.asarray(index01, np.float32))

    def _failed(self, msg, request=None):
        if self._closing or (request is not None and request != self._request):
            return
        self._worker_outcome = ("failed", msg)

    def _worker_finished(self):
        worker = self.sender()
        if worker is not self._worker:
            return
        self._worker = None
        worker.deleteLater()
        if self._closing:
            return
        outcome = self._worker_outcome
        if (worker.request == self._request and not self._cancelling
                and outcome is not None and outcome[0] == "ok"):
            self._index01 = outcome[1]
            self._map_valid_mask = worker.valid_mask
            self._map_frames = list(worker._images)
            self._map_indices = self._request_indices
            self._map_names = self._request_names
            self._map_parameters = {"kernel_size": worker._kernel_size, "N": worker.nlevels,
                                    "reverse_order": worker.reverse_order}
            self._map_alignment = worker.alignment if not self._source_context() else self._source_context().alignment
            self.z_mapping.setToolTip(trans.t("depth_axis_hint") + "\n" +
                ", ".join(str(i + 1) for i in self._map_indices))
            self._map_method = self._worker_method
            self._map_request = self._request
            self._stale = False
            self.status_lbl.setText(trans.t("depth_ready"))
            self._refresh_preview()
        elif outcome is not None and outcome[0] == "failed" and outcome[1] != "cancelled":
            self.status_lbl.setText(outcome[1])
        else:
            self.status_lbl.setText(trans.t("depth_cancelled"))
        self._cancelling = False
        self.compute_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(True)
        self._update_export_state()
        self._update_overlay_availability()
        if self._pending_compute:
            self._pending_compute = False
            self._compute()

    def _update_export_state(self):
        """Export only a current, successful map."""
        running = self._worker is not None
        current = (self._index01 is not None and not self._stale
                   and not running and not self._closing and self._export_worker is None)
        for name in ("btn_save_color", "btn_save_raw", "btn_save_3d"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(current)
        self.btn_save_physical.setEnabled(current and self.physical_chk.isChecked())

    # ------------------------------------------------------------------
    def _colorized_full(self):
        from core.depth_map import smooth_index_map
        cid, stops = self._selected_colormap()
        m = smooth_index_map(self._index01, self.smooth_spin.value())
        if self._map_valid_mask is not None:
            m = np.where(self._map_valid_mask, m, np.nan)
        colored = cmap.colorize(m, cid, invert=self.invert_chk.isChecked(),
                                gamma=self.gamma_spin.value(),
                                custom_stops=stops)
        if self.overlay_chk.isChecked():
            base = self._matching_texture()
            if base is not None and base.shape[:2] == colored.shape[:2]:
                from utils.image_utils import to_display_uint8
                a = float(self.opacity_spin.value())
                # 16-bit 结果 (0-65535) 直接当 0-255 混合再 clip，整幅会变白；
                # 先按显示位深换算
                base_f = to_display_uint8(base).astype(np.float32)
                col_f = colored.astype(np.float32)
                if base_f.ndim == 2:
                    base_f = cv2.cvtColor(base_f, cv2.COLOR_GRAY2BGR)
                out = base_f * (1.0 - a) + col_f * a
                if self._map_valid_mask is not None:
                    out[~self._map_valid_mask] = base_f[~self._map_valid_mask]
                return np.clip(out, 0, 255).astype(np.uint8)
        return colored

    def _refresh_preview(self):
        self._refresh_visuals()
        self._update_export_state()

    def _refresh_visuals(self):
        if self._stale:
            return
        try:
            cid, stops = self._selected_colormap()
            lut = cmap.colorize(np.linspace(0, 1, 256, dtype=np.float32)[None, :],
                                cid, invert=self.invert_chk.isChecked(),
                                gamma=self.gamma_spin.value(), custom_stops=stops)[0]
        except Exception:
            lut = cmap.get_lut(_DEFAULT_METHOD_COLORMAP)
        # Both the classical measures and the neural one report a frame
        # index, so the legend keeps the frame count at its right end.
        frames = self._map_frames if self._index01 is not None else (self._images() or [])
        labels = ["1", f"{(len(frames) + 1) / 2:g}", str(len(frames))]
        if (self.physical_chk.isChecked() and self.z_mapping.isChecked()
                and self.z_step.value() > 0 and len(self._map_indices) >= 2):
            unit = self.z_unit.currentData()
            positions = (self.z_start.value() + np.asarray(self._map_indices) *
                         self.z_step.value() * self.z_direction.currentData()) * unit
            values = np.interp([0, (len(positions) - 1) / 2, len(positions) - 1],
                               np.arange(len(positions)), positions)
            labels = [f"{value:g} µm" for value in values]
        self.legend.set_lut(lut, labels[2], labels[0], labels[1])
        self.relief_spin.setEnabled(not self.physical_chk.isChecked())
        if self._index01 is None:
            self.legend.update()
            return
        # Keep the 3D tab in step with the same parameters the map uses, so
        # dragging relief/smoothing/gamma shows the surface immediately.
        self._refresh_surface()
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

    def _refresh_surface(self) -> None:
        """Rebuild the height field for the 3D tab from the live parameters."""
        if self._index01 is None or not hasattr(self, "surface"):
            return
        try:
            from core.depth_map import smooth_index_map
            from utils.mesh_export import build_height_field
            index = smooth_index_map(self._index01, self.smooth_spin.value())
            if self._map_valid_mask is not None:
                index = np.where(self._map_valid_mask, index, np.nan)
            if self.physical_chk.isChecked():
                height, _positions = self._physical_values(index)
                pixel_size = self._pixel_size()
                if pixel_size is None or not np.isfinite(pixel_size) or pixel_size <= 0:
                    raise ValueError(trans.t("depth_physical_missing"))
                if self.invert_chk.isChecked():
                    height = -height
                self._mesh_height = build_height_field(height, self.mesh_size.value())
                longest = max(index.shape[0] - 1, index.shape[1] - 1) * pixel_size
                hf = self._mesh_height / longest
                relief = 1.0
            else:
                height = 1.0 - index if self.invert_chk.isChecked() else index
                hf = build_height_field(height, self.mesh_size.value())
                self._mesh_height = hf
                relief = self.relief_spin.value() / 100.0
            from utils.image_utils import to_display_uint8
            self._mesh_texture = to_display_uint8(self._texture_for_map(index))
            tex = cv2.resize(self._mesh_texture, (hf.shape[1], hf.shape[0]),
                             interpolation=cv2.INTER_AREA)
            self.surface.set_relief(relief)
            self.surface.set_surface(hf, tex, extent=(index.shape[0] - 1, index.shape[1] - 1))
            self._surface_error = ""
            if getattr(self, "lbl_surface_state", None) is not None:
                valid = np.isfinite(self._mesh_height)
                cells = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
                self.lbl_surface_state.setText(
                    f"{hf.shape[1]}×{hf.shape[0]} · V {int(valid.sum())} · △ {int(cells.sum()) * 2}")
        except Exception as exc:
            # Keep the 2D map usable, but never leave the previous surface
            # looking like the current result.
            self._surface_error = str(exc)
            self._mesh_height = None
            self._mesh_texture = None
            if getattr(self, "lbl_surface_state", None) is not None:
                self.lbl_surface_state.setText(
                    trans.t("depth_3d_failed").format(error=exc))
            try:
                self.surface.set_surface(None, None)
            except Exception:
                pass

    def _require_current_map(self) -> bool:
        """Exports are only valid for a map that matches current settings."""
        if self._index01 is None:
            self.status_lbl.setText(trans.t("depth_no_map_yet"))
            return False
        if self._stale:
            self.status_lbl.setText(trans.t("depth_stale"))
            return False
        if self._worker is not None:
            self.status_lbl.setText(trans.t("depth_computing_ai"))
            return False
        return True

    def _save(self, colorized=True):
        if not self._require_current_map():
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
        if not colorized and os.path.splitext(path)[1].lower() not in (
                ".tif", ".tiff"):
            # imwrite_auto legally converts uint16 to 8-bit for JPG/BMP; that
            # is right for ordinary images, wrong for the "raw 16-bit" exit.
            self.status_lbl.setText(trans.t("depth_raw_needs_tiff"))
            return
        set_last_dialog_dir(path)
        try:
            if colorized:
                ok = imwrite_auto(path, self._colorized_full())
            else:
                u16 = np.clip(self._index01 * 65535.0, 0, 65535).round().astype(np.uint16)
                ok = imwrite_auto(path, u16)
            # imwrite_auto returns False (it does not raise) when the target
            # is unwritable — claiming success there was actively misleading.
            if ok:
                self.status_lbl.setText(trans.t("depth_saved").format(path=path))
            else:
                self.status_lbl.setText(trans.t("msg_save_failed_text"))
        except Exception as exc:
            self.status_lbl.setText(f"{exc}")

    def _export_3d(self):
        if not self._require_current_map() or self._export_worker is not None:
            return
        self._refresh_surface()
        if self._mesh_height is None:
            self.status_lbl.setText(self.lbl_surface_state.text())
            return
        from utils.mesh_export import obj_paths
        start_dir = get_last_dialog_dir()
        default = os.path.join(start_dir, "depth_model.obj") if start_dir else "depth_model.obj"
        path, _ = QFileDialog.getSaveFileName(
            self, trans.t("depth_export_3d"), default, "Wavefront OBJ (*.obj)")
        if not path:
            return
        conflicts = [name for name in obj_paths(path) if os.path.exists(name)]
        if conflicts:
            answer = QMessageBox.question(self, trans.t("depth_export_3d"),
                trans.t("depth_replace_group").format(files="\n".join(conflicts)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        metadata = {"method": self._map_method, "source_indices": self._map_indices,
                    "domain": self._domain_metadata(), "smooth": self.smooth_spin.value(),
                    "invert": self.invert_chk.isChecked(), "texture": self.texture_combo.currentText()}
        metadata.update(source_names=getattr(self, "_map_names", []),
                        parameters=getattr(self, "_map_parameters", {}),
                        alignment=getattr(self, "_map_alignment", "none"),
                        request=self._map_request)
        if self.physical_chk.isChecked():
            _depth, metadata["z_positions_um"] = self._physical_values(self._index01)
        options = {"height_field": self._mesh_height,
                   "pixel_size_um": self._pixel_size() if self.physical_chk.isChecked() else None,
                   "relief": self.relief_spin.value() / 100.0,
                   "overwrite": bool(conflicts), "metadata": metadata}
        worker = MeshExportWorker(path, self._index01, self._mesh_texture, options, self)
        self._export_worker = worker
        self._export_outcome = None
        worker.ok.connect(lambda info: self._set_export_outcome((True, info)))
        worker.failed.connect(lambda error: self._set_export_outcome((False, error)))
        worker.finished.connect(self._export_finished)
        self.status_lbl.setText(trans.t("depth_export_3d"))
        self.status_lbl.setToolTip("\n".join(obj_paths(path)))
        self.compute_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self._update_export_state()
        set_last_dialog_dir(path)
        worker.start()

    def _set_export_outcome(self, outcome):
        if not self._closing:
            self._export_outcome = outcome

    def _export_finished(self):
        worker = self.sender()
        if worker is not self._export_worker:
            return
        self._export_worker = None
        worker.deleteLater()
        if self._closing:
            return
        ok, result = self._export_outcome or (False, "cancelled")
        self.status_lbl.setText(trans.t("depth_saved").format(path="\n".join(result["files"]))
            if ok else trans.t("depth_cancelled") if result == "cancelled" else result)
        self.compute_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setEnabled(True)
        self._update_export_state()

    def _detach_export(self):
        worker, self._export_worker = self._export_worker, None
        if worker is None:
            return
        if not worker.isRunning():
            worker.deleteLater()
            return
        worker.cancel()
        worker.setParent(None)
        _ORPHAN_WORKERS.add(worker)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(lambda w=worker: _ORPHAN_WORKERS.discard(w))

    def _matching_texture(self):
        context = self._source_context()
        if context is None or not context.frames:
            return None
        result = self._output_texture
        frame = context.frames[0]
        if any((f.offset_x, f.offset_y, f.width, f.height) !=
               (frame.offset_x, frame.offset_y, frame.width, frame.height)
               for f in context.frames):
            return None
        texture = result[frame.offset_y:frame.offset_y + frame.height,
                         frame.offset_x:frame.offset_x + frame.width]
        if self._index01 is not None and texture.shape[:2] != self._index01.shape:
            return None
        return texture

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
        self.relief_spin.setValue(int(s.value("depthmap/relief", 15)))
        self._refresh_preview()

    def _save_prefs(self):
        s = get_settings()
        s.setValue("depthmap/colormap", self.map_combo.currentData())
        s.setValue("depthmap/gamma", self.gamma_spin.value())
        s.setValue("depthmap/smooth", self.smooth_spin.value())
        s.setValue("depthmap/invert", self.invert_chk.isChecked())
        s.setValue("depthmap/relief", self.relief_spin.value())

    def _detach_worker(self):
        """关闭对话框时后台线程可能正处在长推理中——QThread 被销毁而线程
        仍在运行会让整个进程 abort。让它脱离对话框的生命周期，取消后
        自行跑完/退出并自我清理；信号在接收者销毁后由 Qt 自动断开。"""
        w = self._worker
        self._worker = None
        if w is None:
            return
        if not w.isRunning():
            w.deleteLater()
            return
        w.cancel()
        w.setParent(None)
        _ORPHAN_WORKERS.add(w)
        w.finished.connect(w.deleteLater)
        w.finished.connect(lambda ww=w: _ORPHAN_WORKERS.discard(ww))

    def reject(self):
        self._closing = True
        self._request += 1
        self._pending_compute = False
        self._detach_export()
        self._detach_worker()
        self._save_prefs()
        super().reject()

    def closeEvent(self, event):
        self._closing = True
        self._request += 1
        self._pending_compute = False
        self._detach_export()
        self._detach_worker()
        self._save_prefs()
        super().closeEvent(event)
