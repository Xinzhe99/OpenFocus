"""Interactive comparison using one aligned stack and one inspection region."""
import os

from PyQt6.QtCore import QEvent, QSignalBlocker, Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter,
    QVBoxLayout, QWidget,
)

from core.compare_worker import AlignmentWorker, PreviewWorker
from locales import trans
from utils.image_utils import cv2_to_pixmap
from utils.settings_store import get_settings
from utils.validators import normalize_kernel_size
from widgets.compare_canvas import (
    DEFAULT_DETAIL_SIZE, DetailCanvas, SourceCanvas,
)
from widgets.flow_layout import FlowPanel
from widgets.stepper import StepperSpinBox


METHOD_ORDER = ("stackmffv3", "openfocus_a", "openfocus_b", "openfocus_c",
                "guided_filter", "dct", "dtcwt", "gfgfgf")
METHOD_LABEL_KEYS = {
    "stackmffv3": "radio_stackmff", "openfocus_a": "radio_openfocus_a",
    "openfocus_b": "radio_openfocus_b", "openfocus_c": "radio_openfocus_c",
    "guided_filter": "radio_guided_filter", "dct": "radio_dct",
    "dtcwt": "radio_dtcwt", "gfgfgf": "radio_gfg",
}
# Only parameters consumed by the actual algorithm belong in the editor.
# (key, caption, minimum, maximum, step, decimals, default)
METHOD_PARAMS = {
    "guided_filter": (("kernel_size", "label_kernel_guided", 1, 51, 2, 0, 31),),
    "dct": (("kernel_size", "label_kernel_dct", 1, 51, 2, 0, 7),),
    "gfgfgf": (("kernel_size", "label_kernel_gfg", 1, 51, 2, 0, 7),),
    "dtcwt": (("N", "compare_levels", 1, 6, 1, 0, 4),),
    "openfocus_a": (
        ("kernel_size", "label_kernel_of", 1, 21, 2, 0, 11),
        ("blend_smoothing", "label_blend_smoothing", 0, 8, .2, 1, 2.),
    ),
    "openfocus_b": (
        ("kernel_size", "label_kernel_of", 1, 21, 2, 0, 11),
        ("depth_smoothing", "label_depth_smoothing", 0, 8, .2, 1, 1.2),
    ),
    "openfocus_c": (
        ("blend_smoothing", "label_blend_smoothing", 0, 8, .2, 1, 3.),
    ),
}


class CompareDialog(QDialog):
    """Edit each method independently; publish only current preview requests."""

    def __init__(self, window_main, parent=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self.setWindowTitle(trans.t("compare_title"))
        self.resize(1280, 820)
        # A plain QDialog has only a close button on Windows; allow the title
        # bar to maximise as well as the explicit fullscreen toggle below.
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowType.WindowMaximizeButtonHint)
        self._windowed_geometry = None
        self._raw_frames = list(getattr(window_main, "raw_images", None) or [])
        self._frames = self._raw_frames
        self._aligned_frames = None
        self._epoch = 0
        self._revision = 0
        self._worker = None
        self._worker_kind = None
        self._active_revision = -1
        self._active_keys = {}
        self._results = {}
        self._errors = {}
        self._method_params = {}
        self._param_rows = {}
        self.param_widgets = {}
        self._canvases = {}
        self._layout_state = None
        self._detail_size = max(160, min(960, get_settings().value(
            "compare/preview_size", DEFAULT_DETAIL_SIZE, type=int)))
        self._auto_enabled = False
        self._closing = False
        self._close_code = QDialog.DialogCode.Rejected
        self._ref_index = 0
        self._shared = self._collect_shared_settings()

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._start_request)
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._refresh_layout)
        self._build_ui()
        self._show_frame(0)
        method = (window_main.current_fusion_method()
                  if hasattr(window_main, "current_fusion_method")
                  else "guided_filter")
        if method not in self._checks:
            method = "guided_filter"
        self._checks[method].setChecked(True)

    def _collect_shared_settings(self):
        window = self.window_main
        settings = {
            "use_gpu": bool(getattr(window, "use_gpu", False)),
            "tile_enabled": bool(getattr(window, "tile_enabled", True)),
            "ai_batch_size": max(1, int(getattr(window, "ai_batch_size", 2))),
            "thread_count": max(1, int(getattr(window, "thread_count", 4))),
            "reg_downscale_width": int(getattr(window, "reg_downscale_width", 1024)),
            "exposure_match": bool(getattr(window, "exposure_match", True)),
            "sort_mode": "desc" if getattr(window, "stack_reverse_order", False) else "asc",
        }
        for key in ("tile_block_size", "tile_overlap", "tile_threshold"):
            value = getattr(window, key, None)
            if value is not None:
                settings[key] = value
        return settings

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        methods_row = QHBoxLayout()
        methods_flow = FlowPanel(h_spacing=12, v_spacing=6)
        methods_flow.add(QLabel(trans.t("compare_pick")))
        self._checks = {}
        from core.multi_focus_fusion import is_ai_fusion_available
        for method in METHOD_ORDER:
            if method == "stackmffv3" and not is_ai_fusion_available():
                continue
            box = QCheckBox(trans.t(METHOD_LABEL_KEYS[method]), methods_flow)
            box.toggled.connect(self._methods_changed)
            methods_flow.add(box)
            self._checks[method] = box
        methods_row.addWidget(methods_flow, 1)
        self.btn_run = QPushButton(trans.t("compare_run"))
        self.btn_run.setObjectName("primaryButton")
        self.btn_run.clicked.connect(self.run_comparison)
        methods_row.addWidget(self.btn_run, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(methods_row)

        settings = QHBoxLayout()
        settings.addWidget(QLabel(trans.t("group_registration")))
        self.align_combo = QComboBox()
        for code, key in (("none", "check_align_none"),
                          ("homography", "check_align_homography"),
                          ("ecc", "check_align_ecc"),
                          ("openfocus", "check_align_of")):
            self.align_combo.addItem(trans.t(key), code)
        current_align = (self.window_main.current_alignment_method()
                         if hasattr(self.window_main, "current_alignment_method") else "none")
        self.align_combo.setCurrentIndex(max(0, self.align_combo.findData(current_align)))
        self.align_combo.currentIndexChanged.connect(self._alignment_changed)
        settings.addWidget(self.align_combo)
        settings.addSpacing(12)
        settings.addWidget(QLabel(trans.t("compare_mode")))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(trans.t("compare_roi"), True)
        self.mode_combo.addItem(trans.t("compare_full"), False)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        settings.addWidget(self.mode_combo)
        settings.addStretch()
        self.fit_button = QPushButton(trans.t("retouch_fit"))
        settings.addWidget(self.fit_button)
        self.fullscreen_button = QPushButton(trans.t("compare_fullscreen"))
        self.fullscreen_button.clicked.connect(self._toggle_fullscreen)
        settings.addWidget(self.fullscreen_button)
        root.addLayout(settings)

        self.param_host = FlowPanel(h_spacing=12, v_spacing=6)
        self.param_scroll = QScrollArea()
        self.param_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setWidget(self.param_host)
        self.param_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.param_scroll.viewport().installEventFilter(self)
        root.addWidget(self.param_scroll)
        self._build_param_rows()

        hint = QLabel(trans.t("compare_hint"))
        hint.setWordWrap(True)
        hint.setObjectName("mutedLabel")
        root.addWidget(hint)
        self.preview_note = QLabel(trans.t("compare_preview_note"))
        self.preview_note.setWordWrap(True)
        self.preview_note.setObjectName("mutedLabel")
        root.addWidget(self.preview_note)

        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.split.setChildrenCollapsible(False)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.ref_canvas = SourceCanvas(trans.t("compare_reference"), left)
        self.ref_canvas.set_detail_size(self._detail_size)
        self.ref_canvas.regionChanged.connect(self._region_changed)
        self.fit_button.clicked.connect(self.ref_canvas.fit)
        fit_shortcut = QShortcut(QKeySequence("F"), self)
        fit_shortcut.activated.connect(self.ref_canvas.fit)
        fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        fullscreen_shortcut.activated.connect(self._toggle_fullscreen)
        left_layout.addWidget(self.ref_canvas, 1)
        navigation = QHBoxLayout()
        self.btn_prev_frame = QPushButton("◀")
        self.btn_prev_frame.setFixedWidth(32)
        self.btn_prev_frame.clicked.connect(lambda: self.frame_slider.setValue(self._ref_index - 1))
        navigation.addWidget(self.btn_prev_frame)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, max(0, len(self._frames) - 1))
        self.frame_slider.valueChanged.connect(self._show_frame)
        navigation.addWidget(self.frame_slider, 1)
        self.btn_next_frame = QPushButton("▶")
        self.btn_next_frame.setFixedWidth(32)
        self.btn_next_frame.clicked.connect(lambda: self.frame_slider.setValue(self._ref_index + 1))
        navigation.addWidget(self.btn_next_frame)
        left_layout.addLayout(navigation)
        self.frame_label = QLabel()
        self.frame_label.setObjectName("mutedLabel")
        self.frame_label.setWordWrap(True)
        left_layout.addWidget(self.frame_label)
        self.split.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        size_row = QHBoxLayout()
        size_label = QLabel(trans.t("compare_size"))
        size_row.addWidget(size_label)
        self.preview_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_size_slider.setRange(160, 960)
        self.preview_size_slider.setSingleStep(20)
        self.preview_size_slider.setPageStep(80)
        self.preview_size_slider.setValue(self._detail_size)
        self.preview_size_slider.setAccessibleName(size_label.text())
        size_row.addWidget(self.preview_size_slider, 1)
        self.preview_size_spin = StepperSpinBox(
            minimum=160, maximum=960, step=20, value=self._detail_size)
        self.preview_size_spin.setFixedWidth(110)
        self.preview_size_spin.setAccessibleName(size_label.text())
        self.preview_size_spin.lineEdit().setAccessibleName(size_label.text())
        size_label.setBuddy(self.preview_size_spin.lineEdit())
        size_row.addWidget(self.preview_size_spin)
        right_layout.addLayout(size_row)
        self.preview_size_slider.valueChanged.connect(
            self._preview_size_changed)
        self.preview_size_spin.valueChanged.connect(self._preview_size_changed)

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setWidget(self.grid_host)
        self.result_scroll.viewport().installEventFilter(self)
        right_layout.addWidget(self.result_scroll, 1)
        self.split.addWidget(right)
        self.split.setStretchFactor(0, 2)
        self.split.setStretchFactor(1, 3)
        self.split.setSizes([480, 740])
        self.split.splitterMoved.connect(lambda *_: self._layout_timer.start(0))
        root.addWidget(self.split, 1)
        for method in self._checks:
            canvas = DetailCanvas(trans.t(METHOD_LABEL_KEYS[method]), self.grid_host)
            canvas.setFixedSize(self._detail_size, self._detail_size)
            canvas.hide()
            self._canvases[method] = canvas

        footer = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(180)
        self.progress.hide()
        footer.addWidget(self.progress)
        close_button = QPushButton(trans.t("btn_close"))
        close_button.clicked.connect(self.reject)
        footer.addWidget(close_button)
        root.addLayout(footer)
        self._layout_timer.start(0)

    def _initial_params(self, method):
        window = self.window_main
        params = {}
        for key, _label, lo, hi, _step, decimals, default in METHOD_PARAMS.get(method, ()):
            value = default
            if key == "kernel_size" and hasattr(window, "kernel_for_method"):
                value = window.kernel_for_method(method)
            elif key in ("blend_smoothing", "depth_smoothing") and hasattr(window, "smoothing_for_method"):
                value = window.smoothing_for_method(method)
            value = max(lo, min(hi, value))
            params[key] = float(value) if decimals else int(value)
            if key == "kernel_size":
                params[key] = normalize_kernel_size(params[key])
        if method == "stackmffv3":
            params["mode"] = getattr(window, "ai_mode", "soft")
        for option in ("exposure_match", "base_correct"):
            if hasattr(window, "brightness_option_for"):
                enabled = window.brightness_option_for(option, method)
            else:
                enabled = option == "exposure_match"
            params["base_frame" if option == "base_correct" else option] = (
                ("last" if enabled else "none") if option == "base_correct" else enabled)
        return params

    def _build_param_rows(self):
        for method in self._checks:
            params = self._initial_params(method)
            self._method_params[method] = params
            group = QFrame(self.param_host)
            group.setFrameShape(QFrame.Shape.StyledPanel)
            group.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            row = QHBoxLayout(group)
            row.setContentsMargins(8, 4, 8, 4)
            row.setSpacing(6)
            title = QLabel(trans.t(METHOD_LABEL_KEYS[method]), group)
            font = title.font()
            font.setBold(True)
            title.setFont(font)
            row.addWidget(title)
            self.param_widgets[method] = {}
            for key, label_key, lo, hi, step, decimals, _default in METHOD_PARAMS.get(method, ()):
                caption = QLabel(trans.t(label_key), group)
                spin = StepperSpinBox(group, minimum=lo, maximum=hi, step=step,
                                      decimals=decimals, value=params[key])
                spin.setFixedWidth(110)
                spin.setAccessibleName(f"{title.text()} — {caption.text()}")
                spin.lineEdit().setAccessibleName(spin.accessibleName())
                caption.setBuddy(spin.lineEdit())
                spin.valueChanged.connect(
                    lambda value, m=method, k=key: self._param_changed(m, k, value))
                row.addWidget(caption)
                row.addWidget(spin)
                self.param_widgets[method][key] = spin
            if method == "stackmffv3":
                caption = QLabel(trans.t("label_ai_mode"), group)
                mode = QComboBox(group)
                mode.addItem(trans.t("radio_ai_soft"), "soft")
                mode.addItem(trans.t("radio_ai_hard"), "hard")
                mode.setCurrentIndex(max(0, mode.findData(params["mode"])))
                mode.setAccessibleName(f"{title.text()} — {caption.text()}")
                caption.setBuddy(mode)
                mode.currentIndexChanged.connect(
                    lambda _index, c=mode: self._param_changed("stackmffv3", "mode", c.currentData()))
                row.addWidget(caption)
                row.addWidget(mode)
                self.param_widgets[method]["mode"] = mode
            self.param_host.add(group)
            self._param_rows[method] = group
            group.hide()

    def _selected_methods(self):
        return [m for m in METHOD_ORDER if m in self._checks and self._checks[m].isChecked()]

    def _methods_changed(self, _checked):
        selected = self._selected_methods()
        for method, group in self._param_rows.items():
            group.setVisible(method in selected)
        self.param_scroll.setVisible(bool(selected))
        self.param_host.relayout()
        self._layout_timer.start(0)
        self._request_changed()

    def _param_changed(self, method, key, value):
        if key == "kernel_size":
            value = normalize_kernel_size(int(value))
            spin = self.param_widgets[method][key]
            with QSignalBlocker(spin):
                spin.setValue(value)
        elif key == "N":
            value = int(value)
        if self._method_params[method].get(key) == value:
            return
        self._method_params[method][key] = value
        self._results.pop(method, None)
        self._errors.pop(method, None)
        self._request_changed()

    def _alignment_changed(self, _index):
        self._epoch += 1
        self._aligned_frames = None
        self._frames = self._raw_frames
        self._results.clear()
        self._errors.clear()
        self._show_frame(self._ref_index)
        self._request_changed()

    def _mode_changed(self, _index):
        self.preview_note.setVisible(bool(self.mode_combo.currentData()))
        self._results.clear()
        self._errors.clear()
        self._request_changed()

    def _show_frame(self, index):
        if not self._frames:
            return
        self._ref_index = max(0, min(index, len(self._frames) - 1))
        with QSignalBlocker(self.ref_canvas):
            self.ref_canvas.set_pixmap(cv2_to_pixmap(self._frames[self._ref_index]))
        names = getattr(self.window_main, "image_filenames", None) or []
        name = os.path.basename(str(names[self._ref_index])) if self._ref_index < len(names) else ""
        self.frame_label.setText(trans.t("compare_frame_of").format(
            i=self._ref_index + 1, n=len(self._frames), name=name))
        self.btn_prev_frame.setEnabled(self._ref_index > 0)
        self.btn_next_frame.setEnabled(self._ref_index < len(self._frames) - 1)
        # Switching the reference frame changes no fusion inputs or coordinates.
        self._refresh_panes()

    def _preview_size_changed(self, value):
        size = int(value)
        if size == self._detail_size:
            return
        self._detail_size = size
        with (QSignalBlocker(self.preview_size_slider),
              QSignalBlocker(self.preview_size_spin)):
            self.preview_size_slider.setValue(size)
            self.preview_size_spin.setValue(size)
        for canvas in self._canvases.values():
            canvas.setFixedSize(size, size)
        self.ref_canvas.set_detail_size(size)
        self._layout_timer.start(0)

    def _region_changed(self, _region):
        # A full render and stack registration do not depend on the selection.
        if (self.mode_combo.currentData()
                and self._worker_kind != "alignment"):
            self._request_changed()
        else:
            self._refresh_panes()

    def _result_key(self, method):
        roi = bool(self.mode_combo.currentData())
        return (self._epoch, roi, tuple(sorted(self._method_params[method].items())),
                self.ref_canvas.region() if roi else None)

    def _refresh_panes(self):
        region = self.ref_canvas.region()
        for method in self._selected_methods():
            canvas = self._canvases[method]
            canvas.set_region(region)
            result = self._results.get(method)
            key = self._result_key(method)
            if result is not None and result[0] == key:
                canvas.set_result(result[1], result[2])
                canvas.set_status("")
            else:
                canvas.set_result(None, None)
                error = self._errors.get(method)
                if error is not None and error[0] == key:
                    canvas.set_status(error[1])
                else:
                    canvas.set_status(trans.t("render_stage_fuse") if self._auto_enabled
                                      else trans.t("compare_no_image"))

    def _request_changed(self):
        if self._closing:
            return
        self._revision += 1  # Invalidate old work now, before the debounce expires.
        self._auto_enabled = bool(self._raw_frames and self._selected_methods())
        if self._worker is not None:
            self._worker.cancel()
        self._refresh_panes()
        self.btn_run.setEnabled(bool(self._raw_frames and self._selected_methods()))
        self._preview_timer.start(180)

    def run_comparison(self):
        if self._worker is not None:
            self._auto_enabled = False
            self._revision += 1
            self._preview_timer.stop()
            self._worker.cancel()
            self.status_label.setText(trans.t("msg_render_cancelled"))
            self._refresh_panes()
            return
        for method in self._selected_methods():
            self._results.pop(method, None)
            self._errors.pop(method, None)
        self._request_changed()
        self._preview_timer.start(0)

    def _start_request(self):
        if self._closing or not self._auto_enabled or self._worker is not None:
            return
        methods = self._selected_methods()
        if not methods or not self._raw_frames:
            self._set_idle()
            return
        if self._aligned_frames is None:
            alignment = self.align_combo.currentData()
            if alignment == "none":
                self._aligned_frames = self._raw_frames
                self._frames = self._aligned_frames
                self._show_frame(self._ref_index)
            else:
                worker = AlignmentWorker(self._raw_frames, alignment, self._shared, self)
                worker.ready.connect(self._alignment_ready)
                worker.failed.connect(self._alignment_failed)
                self.status_label.setText(trans.t("render_stage_align"))
                self.progress.setRange(0, 0)
                self._launch_worker(worker, "alignment")
                return
        region = self.ref_canvas.region()
        if region is None:
            return
        pending = []
        for method in methods:
            key = self._result_key(method)
            cached = self._results.get(method)
            failed = self._errors.get(method)
            if (cached is None or cached[0] != key) and (failed is None or failed[0] != key):
                pending.append(method)
        if not pending:
            self._refresh_panes()
            self._set_idle()
            return
        self._active_keys = {method: self._result_key(method) for method in pending}
        worker = PreviewWorker(self._aligned_frames, pending, self._method_params,
                               self._shared, region, bool(self.mode_combo.currentData()), self)
        worker.result_ready.connect(self._preview_ready)
        worker.method_failed.connect(self._preview_failed)
        worker.progress.connect(self._progress_changed)
        self.status_label.setText(trans.t("render_stage_fuse"))
        self.progress.setRange(0, len(pending))
        self.progress.setValue(0)
        self._launch_worker(worker, "preview")

    def _launch_worker(self, worker, kind):
        self._worker = worker
        self._worker_kind = kind
        self._active_revision = self._revision
        worker.finished.connect(self._worker_finished)
        self.btn_run.setText(trans.t("compare_cancel"))
        self.progress.show()
        worker.start()

    def _current_worker(self):
        return (not self._closing and self.sender() is self._worker
                and self._active_revision == self._revision)

    def _alignment_ready(self, frames):
        if not self._current_worker():
            return
        self._aligned_frames = list(frames)
        self._frames = self._aligned_frames
        self._show_frame(self._ref_index)
        self._refresh_panes()

    def _alignment_failed(self, message):
        if not self._current_worker():
            return
        self._auto_enabled = False
        self.status_label.setText(f"{trans.t('msg_error')}: {message}")
        for method in self._selected_methods():
            self._canvases[method].set_status(message)

    def _preview_ready(self, method, image, bounds):
        if not self._current_worker() or method not in self._selected_methods():
            return
        key = self._active_keys.get(method)
        if key != self._result_key(method):
            return
        pixmap = cv2_to_pixmap(image)
        if pixmap.isNull():
            self._preview_failed(method, trans.t("msg_proc_error"))
            return
        self._results[method] = (key, pixmap, tuple(bounds))
        self._errors.pop(method, None)
        self._refresh_panes()

    def _preview_failed(self, method, message):
        if not self._current_worker():
            return
        self._errors[method] = (self._active_keys[method], message)
        self._refresh_panes()

    def _progress_changed(self, done, total, method):
        if not self._current_worker():
            return
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        if method:
            self.status_label.setText(f"{trans.t('render_stage_fuse')} {trans.t(METHOD_LABEL_KEYS[method])}")

    def _worker_finished(self):
        worker = self.sender()
        if worker is not self._worker:
            return
        self._worker = None
        self._worker_kind = None
        worker.deleteLater()
        if self._closing:
            super().done(self._close_code)
            return
        self._set_idle()
        if self._auto_enabled and not self._preview_timer.isActive():
            self._preview_timer.start(0)

    def _set_idle(self):
        self.btn_run.setText(trans.t("compare_run"))
        self.progress.hide()
        if self._auto_enabled:
            self.status_label.setText(trans.t("msg_error") if any(
                self._errors.get(m, (None,))[0] == self._result_key(m)
                for m in self._selected_methods()) else "")

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Resize:
            self._layout_timer.start(0)
        return super().eventFilter(watched, event)

    def _toggle_fullscreen(self):
        self.set_fullscreen(not self.isFullScreen())

    def set_fullscreen(self, enabled: bool):
        """Toggle fullscreen, restoring the windowed geometry on exit."""
        enabled = bool(enabled)
        if enabled:
            if not self.isFullScreen():
                self._windowed_geometry = self.saveGeometry()
                self.showFullScreen()
        elif self.isFullScreen():
            self.showNormal()
            if self._windowed_geometry is not None:
                self.restoreGeometry(self._windowed_geometry)
                self._windowed_geometry = None
        self._sync_fullscreen_button()
        self._layout_timer.start(0)

    def _sync_fullscreen_button(self):
        self.fullscreen_button.setText(trans.t(
            "compare_exit_fullscreen" if self.isFullScreen()
            else "compare_fullscreen"))

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._sync_fullscreen_button()

    def keyPressEvent(self, event):  # noqa: N802
        # Esc leaves fullscreen first; a second Esc still closes the dialog.
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.set_fullscreen(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def _refresh_layout(self):
        if self._closing:
            return
        width = max(1, self.param_scroll.viewport().width())
        height = self.param_host.heightForWidth(width) + 4
        self.param_scroll.setFixedHeight(max(48, min(210, height)))
        selected = tuple(self._selected_methods())
        width = max(1, self.result_scroll.viewport().width())
        spacing = self.grid.spacing()
        columns = max(1, (width + spacing) // (self._detail_size + spacing))
        state = (columns, selected)
        if state == self._layout_state:
            return
        self._layout_state = state
        while self.grid.count():
            self.grid.takeAt(0)
        for method, canvas in self._canvases.items():
            canvas.setVisible(method in selected)
        for index, method in enumerate(selected):
            self.grid.addWidget(self._canvases[method], index // columns, index % columns)

    def done(self, result):
        get_settings().setValue("compare/preview_size", self._detail_size)
        self._closing = True
        self._close_code = result
        self._auto_enabled = False
        self._revision += 1
        self._preview_timer.stop()
        self._layout_timer.stop()
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText(trans.t("compare_stopping"))
            self.setEnabled(False)
            return  # finished closes the dialog; the native call still owns its data.
        super().done(result)

    def closeEvent(self, event):  # noqa: N802
        if self._worker is not None:
            event.ignore()
            self.done(QDialog.DialogCode.Rejected)
        else:
            super().closeEvent(event)
