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
from core.algorithms import (alignment_steps, fusion_algorithms, get_fusion,
    registration_parameter_specs, registration_algorithms, resolve_fusion_parameters,
    resolve_registration_parameters)
from ui.algorithm_parameters import AlgorithmParameterPanel
from widgets.compare_canvas import (
    DEFAULT_DETAIL_SIZE, DetailCanvas, SourceCanvas,
)
from widgets.flow_layout import FlowPanel
from widgets.stepper import StepperSpinBox


def _method_label(method):
    return trans.t(get_fusion(method).label_key)


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
        # Use the same frames a real render would fuse: the source list
        # checkboxes are the "participates in processing" control, and this
        # window used to copy raw_images wholesale, so an excluded frame was
        # still compared here but never in the actual result.
        if hasattr(window_main, "effective_stack"):
            images, names, indices = window_main.effective_stack()
            self._raw_frames = list(images)
            self._source_names = list(names)
            self._source_indices = list(indices)
        else:
            self._raw_frames = list(getattr(window_main, "raw_images", None) or [])
            self._source_names = list(
                getattr(window_main, "image_filenames", None) or [])
            self._source_indices = list(range(len(self._raw_frames)))
        self._total_frames = len(getattr(window_main, "raw_images", None) or [])
        self._frames = self._raw_frames
        self._aligned_frames = None
        self._epoch = 0
        self._revision = 0
        self._worker = None
        self._worker_kind = None
        self._active_revision = -1
        self._active_keys = {}
        self._results = {}
        self._active_method = ""
        self._stale_methods = set()
        self._cancelled_methods = set()
        self._cards = {}
        self._card_status = {}
        self._errors = {}
        # Original (unquantised for display) arrays, keyed by method, so a
        # candidate can be adopted without reading pixels back from a pixmap.
        self._full_results = {}
        self._selected_method = ""
        self._published = False
        self._publishing = False
        self._cancel_requested = False
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
        self._source_revision = int(getattr(window_main, "source_revision", 0))
        manager = getattr(window_main, "label_manager", None)
        frame_ids = manager.frame_ids("input") if manager is not None else []
        self._frame_ids = [frame_ids[i] for i in self._source_indices] if frame_ids else None
        from utils.scalebar import effective_px_um
        self._calibration = {
            "pixel_size_um": effective_px_um(window_main),
            "source_px_um": getattr(window_main, "source_px_um", None),
            "source_scale_factor": getattr(window_main, "current_scale_factor", 1.0),
        }

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._start_request)
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._refresh_layout)
        self._build_ui()
        self._update_source_summary()
        self._show_frame(0)
        method = (window_main.current_fusion_method()
                  if hasattr(window_main, "current_fusion_method")
                  else fusion_algorithms()[0].id)
        if method == 'none':
            method = fusion_algorithms()[0].id
        get_fusion(method)
        self._checks[method].setChecked(True)

    def _update_source_summary(self):
        """State which frames are compared, including excluded ones."""
        label = getattr(self, "source_summary", None)
        if label is None:
            return
        used = len(self._raw_frames)
        total = max(used, int(getattr(self, "_total_frames", used) or used))
        label.setText(trans.t("compare_source_summary").format(
            used=used, total=total))

    def _collect_shared_settings(self):
        window = self.window_main
        settings = {
            "use_gpu": bool(getattr(window, "use_gpu", False)),
            "tile_enabled": bool(getattr(window, "tile_enabled", True)),
            "ai_batch_size": max(1, int(getattr(window, "ai_batch_size", 2))),
            "thread_count": max(0, int(getattr(window, "thread_count", 0))),
            "resource_preference": getattr(window, "resource_preference", "balanced"),
            "reg_downscale_width": int(getattr(window, "reg_downscale_width", 1024)),
            "exposure_match": bool(getattr(window, "exposure_match", True)),
            "sort_mode": "desc" if getattr(window, "stack_reverse_order", False) else "asc",
        }
        for key in ("tile_block_size", "tile_overlap", "tile_threshold"):
            value = getattr(window, key, None)
            if value is not None:
                settings[key] = value
        settings['method_options'] = {}
        getter = getattr(window, 'brightness_option_for', None)
        for spec in fusion_algorithms():
            settings['method_options'][spec.id] = {
                'exposure_match': bool(getter('exposure_match', spec.id)) if callable(getter) else spec.default_exposure_match,
                'base_frame': 'last' if (bool(getter('base_correct', spec.id)) if callable(getter) else spec.default_base_correct) else 'none',
            }
        return settings

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        methods_row = QHBoxLayout()
        methods_flow = FlowPanel(h_spacing=12, v_spacing=6)
        methods_flow.add(QLabel(trans.t("compare_pick")))
        self._checks = {}
        for spec in fusion_algorithms():
            method = spec.id
            box = QCheckBox(trans.t(spec.label_key), methods_flow)
            box.toggled.connect(self._methods_changed)
            methods_flow.add(box)
            self._checks[method] = box
        methods_row.addWidget(methods_flow, 1)
        self.btn_run = QPushButton(trans.t("compare_run"))
        self.btn_run.setObjectName("primaryButton")
        self.btn_run.clicked.connect(self.run_comparison)
        methods_row.addWidget(self.btn_run, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(methods_row)

        self.source_summary = QLabel("")
        self.source_summary.setObjectName("mutedLabel")
        self.source_summary.setWordWrap(True)
        root.addWidget(self.source_summary)

        settings = QHBoxLayout()
        settings.addWidget(QLabel(trans.t("group_registration")))
        self.align_combo = QComboBox()
        for spec in registration_algorithms():
            self.align_combo.addItem(trans.t(spec.label_key), spec.id)
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
        self.mode_combo.setToolTip(trans.t("compare_mode_hint"))
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        settings.addWidget(self.mode_combo)
        settings.addSpacing(12)
        # Viewing is independent of computing: a full result can be inspected
        # as a detail crop or as the whole frame without re-fusing.
        settings.addWidget(QLabel(trans.t("compare_view")))
        self.view_combo = QComboBox()
        self.view_combo.addItem(trans.t("compare_view_detail"), "detail")
        self.view_combo.addItem(trans.t("compare_view_full"), "full")
        self.view_combo.setToolTip(trans.t("compare_view_hint"))
        self.view_combo.currentIndexChanged.connect(self._view_changed)
        settings.addWidget(self.view_combo)
        settings.addStretch()
        self.fit_button = QPushButton(trans.t("retouch_fit"))
        settings.addWidget(self.fit_button)
        self.fullscreen_button = QPushButton(trans.t("compare_fullscreen"))
        self.fullscreen_button.clicked.connect(self._toggle_fullscreen)
        settings.addWidget(self.fullscreen_button)
        root.addLayout(settings)
        self._registration_drafts = {}
        self._editing_alignment = None
        self.registration_parameter_panel = AlgorithmParameterPanel(self)
        self.registration_parameter_panel.valuesChanged.connect(self._registration_parameters_changed)
        root.addWidget(self.registration_parameter_panel)
        self._sync_registration_parameters()

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
            canvas = DetailCanvas(_method_label(method), self.grid_host)
            canvas.setFixedSize(self._detail_size, self._detail_size)
            canvas.hide()
            canvas.installEventFilter(self)
            canvas.setProperty("compare_method", method)
            self._canvases[method] = canvas
            card = QWidget(self.grid_host)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(canvas)
            state = QLabel()
            state.setWordWrap(True)
            state.setFixedWidth(self._detail_size)
            layout.addWidget(state)
            self._cards[method] = card
            self._card_status[method] = state

        footer = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(180)
        self.progress.hide()
        footer.addWidget(self.progress)
        root.addLayout(footer)
        actions = FlowPanel(h_spacing=8, v_spacing=4)
        self.btn_verify = QPushButton(trans.t("compare_verify_full"))
        self.btn_verify.clicked.connect(self._verify_full)
        actions.add(self.btn_verify)
        self.btn_retry = QPushButton(trans.t("compare_retry_method"))
        self.btn_retry.clicked.connect(self._retry_method)
        actions.add(self.btn_retry)
        self.btn_adopt_params = QPushButton(trans.t("compare_adopt_params"))
        self.btn_adopt_params.clicked.connect(self._adopt_params)
        self.btn_adopt_params.setEnabled(False)
        actions.add(self.btn_adopt_params)
        self.btn_adopt_output = QPushButton(trans.t("compare_adopt_output"))
        self.btn_adopt_output.setObjectName("primaryButton")
        self.btn_adopt_output.clicked.connect(self._adopt_output)
        self.btn_adopt_output.setEnabled(False)
        actions.add(self.btn_adopt_output)
        close_button = QPushButton(trans.t("btn_close"))
        close_button.clicked.connect(self.reject)
        actions.add(close_button)
        root.addWidget(actions)
        self._layout_timer.start(0)

    def _initial_params(self, method):
        getter = getattr(self.window_main, 'fusion_parameters_for', None)
        return (getter(method) if callable(getter)
                else resolve_fusion_parameters(method))

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
            title = QLabel(_method_label(method), group)
            font = title.font()
            font.setBold(True)
            title.setFont(font)
            row.addWidget(title)
            panel = AlgorithmParameterPanel(group)
            panel.set_parameters(get_fusion(method).parameters, params)
            panel.valuesChanged.connect(lambda values, m=method: self._params_changed(m, values))
            row.addWidget(panel)
            self.param_widgets[method] = panel.editors
            self.param_host.add(group)
            self._param_rows[method] = group
            group.hide()

    def _selected_methods(self):
        return [m for m, check in self._checks.items() if check.isChecked()]

    def _methods_changed(self, _checked):
        selected = self._selected_methods()
        for method, group in self._param_rows.items():
            group.setVisible(method in selected)
        self.param_scroll.setVisible(bool(selected))
        self.param_host.relayout()
        self._layout_timer.start(0)
        self._request_changed()

    def _params_changed(self, method, values):
        if self._method_params[method] == values:
            return
        self._method_params[method] = dict(values)
        self._results.pop(method, None)
        self._stale_methods.add(method)
        self._full_results.pop(method, None)
        self._errors.pop(method, None)
        self._update_adopt_actions()
        self._request_changed()

    def _sync_registration_parameters(self):
        method = self.align_combo.currentData()
        if method not in self._registration_drafts:
            getter = getattr(self.window_main, 'registration_parameters_for', None)
            self._registration_drafts[method] = (getter(method) if callable(getter)
                else resolve_registration_parameters(method))
        self._editing_alignment = method
        specs = registration_parameter_specs(method)
        self.registration_parameter_panel.set_parameters(specs, self._registration_drafts[method])
        self.registration_parameter_panel.setVisible(bool(specs))
        self._shared['registration_parameters'] = dict(self._registration_drafts[method])

    def _registration_parameters_changed(self, values):
        self._registration_drafts[self._editing_alignment] = dict(values)
        self._alignment_changed(0)

    def _alignment_changed(self, _index):
        self._sync_registration_parameters()
        self._epoch += 1
        self._aligned_frames = None
        self._frames = self._raw_frames
        self._results.clear()
        self._full_results.clear()
        self._errors.clear()
        self._show_frame(self._ref_index)
        self._update_adopt_actions()
        self._request_changed()

    def _mode_changed(self, _index):
        self.preview_note.setVisible(bool(self.mode_combo.currentData()))
        self._stale_methods.update(self._results)
        self._results.clear()
        self._full_results.clear()
        self._errors.clear()
        self._update_adopt_actions()
        self._request_changed()

    def _show_frame(self, index):
        if not self._frames:
            return
        self._ref_index = max(0, min(index, len(self._frames) - 1))
        with QSignalBlocker(self.ref_canvas):
            self.ref_canvas.set_pixmap(cv2_to_pixmap(self._frames[self._ref_index]))
        names = self._source_names
        name = (os.path.basename(str(names[self._ref_index]))
                if self._ref_index < len(names) else "")
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
        for label in self._card_status.values():
            label.setFixedWidth(size)
        self.ref_canvas.set_detail_size(size)
        self._layout_timer.start(0)

    def _view_changed(self, _index):
        """Switch detail/overview without recomputing anything."""
        mode = self.view_combo.currentData()
        for canvas in self._canvases.values():
            canvas.set_view_mode(mode)
        self._refresh_panes()

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
            if self._result_is_current(method):
                status = trans.t("compare_state_current")
            elif self._errors.get(method, (None,))[0] == self._result_key(method):
                status = trans.t("msg_error") + ": " + self._errors[method][1]
            elif self._worker is not None and self._active_revision != self._revision:
                status = trans.t("compare_stopping")
            elif self._worker is not None and method == self._active_method:
                status = trans.t("render_stage_fuse")
            elif self._auto_enabled:
                status = trans.t("compare_state_queued")
            elif method in self._cancelled_methods:
                status = trans.t("compare_cancelled")
            elif method in self._stale_methods:
                status = trans.t("depth_stale")
            else:
                status = trans.t("compare_no_image")
            self._card_status[method].setText(status)
            self._card_status[method].setToolTip(status)
            canvas.set_region(region)
            result = self._results.get(method)
            key = self._result_key(method)
            if result is not None and result[0] == key:
                canvas.set_result(result[1], result[2])
                canvas.set_view_mode(self.view_combo.currentData())
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
        if len(self._raw_frames) < 2:
            self.status_label.setText(trans.t("depth_need_two_frames").format(n=len(self._raw_frames)))
        self._auto_enabled = len(self._raw_frames) >= 2 and bool(self._selected_methods())
        if self._worker is not None:
            self._worker.cancel()
        self._refresh_panes()
        self.btn_run.setEnabled(len(self._raw_frames) >= 2 and bool(self._selected_methods()))
        self._update_adopt_actions()
        self._preview_timer.start(180)

    def run_comparison(self):
        if self._worker is not None:
            self._cancel_requested = True
            self._auto_enabled = False
            self._revision += 1
            self._preview_timer.stop()
            self._worker.cancel()
            # The native step cannot stop instantly; say what is happening and
            # only report "cancelled" when the worker actually finished.
            self.status_label.setText(trans.t("compare_stopping"))
            self.btn_run.setEnabled(False)
            self._refresh_panes()
            return
        for method in self._selected_methods():
            self._errors.pop(method, None)
        self._request_changed()
        self._preview_timer.start(0)

    def _start_request(self):
        if self._closing or not self._auto_enabled or self._worker is not None:
            return
        methods = self._selected_methods()
        if not methods or len(self._raw_frames) < 2:
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
        self._active_method = ""
        self._cancelled_methods.clear()
        self._refresh_panes()
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

    def _select_method(self, method: str) -> None:
        """Mark one result as the adoption candidate."""
        if method not in self._canvases:
            return
        if self._selected_method == method:
            method = ""
        self._selected_method = method
        for name, canvas in self._canvases.items():
            canvas.set_selected(name == method and bool(method))
        self._update_adopt_actions()

    def _update_adopt_actions(self) -> None:
        method = self._selected_method
        available = bool(method) and self._worker is None and not self._closing
        self.btn_verify.setVisible(bool(method) and bool(self.mode_combo.currentData()))
        self.btn_verify.setEnabled(available and self._result_is_current(method))
        self.btn_retry.setVisible(bool(method) and method in self._errors)
        self.btn_retry.setEnabled(available and method in self._errors)
        overview = self.view_combo.model().item(1)
        if overview is not None:
            overview.setEnabled(not bool(self.mode_combo.currentData()))
        if self.mode_combo.currentData() and self.view_combo.currentData() == "full":
            self.view_combo.setCurrentIndex(0)
        eligible = (bool(method) and method in self._selected_methods()
                    and self._result_is_current(method)
                    and self._worker is None and not self._closing
                    and not self._published and not self._publishing)
        self.btn_adopt_params.setEnabled(eligible)
        result = self._full_results.get(method) if method else None
        self.btn_adopt_output.setEnabled(eligible and bool(result)
                                        and result["key"] == self._result_key(method))

    def _retry_method(self):
        method = self._selected_method
        if not method or self._worker is not None:
            return
        self._errors.pop(method, None)
        self._request_changed()
        self._preview_timer.start(0)

    def _verify_full(self):
        method = self._selected_method
        if not method or self._worker is not None or not self._result_is_current(method):
            return
        for name, check in self._checks.items():
            with QSignalBlocker(check):
                check.setChecked(name == method)
        self._methods_changed(True)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(False))
        self._preview_timer.start(0)

    def _result_is_current(self, method: str) -> bool:
        cached = self._results.get(method)
        return bool(cached) and cached[0] == self._result_key(method)

    def _adopt_params(self) -> None:
        """Copy the candidate's parameters to the main window; no render."""
        method = self._selected_method
        if (not method or not self.btn_adopt_params.isEnabled()
                or not self._result_is_current(method)):
            return
        window = self.window_main
        params = dict(self._method_params.get(method, {}))
        previous = window._capture_parameter_state()
        try:
            with window._restore_scope():
                if hasattr(window, 'set_alignment_method'):
                    window.set_alignment_method(self.align_combo.currentData())
                if hasattr(window, 'chk_reverse_order'):
                    window.chk_reverse_order.setChecked(self._shared['sort_mode'] == 'desc')
                if hasattr(window, 'set_fusion_method'):
                    window.set_fusion_method(method)
                window.set_fusion_parameters(method, params)
                window.set_registration_parameters(self.align_combo.currentData(),
                    self.registration_parameter_panel.values())
                if hasattr(window, 'set_brightness_option'):
                    window.set_brightness_option('exposure_match', method,
                        self._shared['method_options'][method]['exposure_match'])
                    window.set_brightness_option('base_correct', method,
                        self._shared['method_options'][method]['base_frame'] != 'none')
            window.persist_settings()
            if hasattr(window, "push_parameter_undo"):
                window.push_parameter_undo(previous)
        except Exception as exc:
            if hasattr(window, "_restore_parameter_state"):
                try:
                    window._restore_parameter_state(previous)
                except Exception as restore_error:
                    self.status_label.setText(f"{exc}\n{restore_error}")
                    return
            self.status_label.setText(str(exc))
            return
        if hasattr(window, "statusBar"):
            window.statusBar().showMessage(
                trans.t("compare_params_adopted"), 6000)
        self.accept()

    def _adopt_output(self) -> None:
        """Publish the selected full-frame result as a new output."""
        method = self._selected_method
        if not method or self._published or self._publishing or self._worker is not None:
            return
        entry = self._full_results.get(method)
        if (not entry or entry["key"] != self._result_key(method)
                or not self._result_is_current(method)):
            return
        image = entry["image"]
        window = self.window_main
        self._publishing = True
        self._update_adopt_actions()
        try:
            from core.render_context import build_render_context
            from controllers.output_manager import OutputRecord
            label = f"{_method_label(method)} · {trans.t('compare_output_suffix')}"
            alignment = self.align_combo.currentData()
            context = build_render_context(
                self._aligned_frames or self._raw_frames, image,
                filenames=self._source_names, algorithm=method,
                alignment_options=alignment_steps(alignment),
                aligned=alignment != "none", source_indices=self._source_indices,
                source_revision=self._source_revision, frame_ids=self._frame_ids,
                parameters={**{key: value for key, value in self._shared.items() if key != 'method_options'},
                    **self._shared['method_options'][method],
                    'fusion_parameters': self._method_params[method]},
                **self._calibration)
            record = OutputRecord.create(
                image, label, context=context, kind="compare",
                exported=False)
            window.output_manager.publish_compare_output(record)
            self._published = True
            self.accept()
        except Exception as exc:
            self.status_label.setText(f"{trans.t('msg_error')}: {exc}")
        finally:
            self._publishing = False
            self._update_adopt_actions()

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
        # Keep the original array for adoption: only a full-frame result may
        # become an output, and it must not round-trip through the pixmap.
        x, y, width, height = tuple(bounds)
        if (not key[1] and x == 0 and y == 0
                and (height, width) == self._raw_frames[0].shape[:2]
                and (height, width) == image.shape[:2]):
            self._full_results[method] = {
                "key": key, "image": image, "bounds": tuple(bounds)}
        else:
            self._full_results.pop(method, None)
        self._errors.pop(method, None)
        self._update_adopt_actions()
        self._refresh_panes()

    def _preview_failed(self, method, message):
        if not self._current_worker():
            return
        self._errors[method] = (self._active_keys[method], message)
        self._refresh_panes()

    def _progress_changed(self, done, total, method):
        if not self._current_worker():
            return
        self._active_method = method
        self._refresh_panes()
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        if method:
            self.status_label.setText(f"{trans.t('render_stage_fuse')} {_method_label(method)}")

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
        if self._cancel_requested:
            self._cancelled_methods.update(m for m in self._selected_methods()
                                           if not self._result_is_current(m))
            self._cancel_requested = False
            self.status_label.setText(trans.t("compare_cancelled"))
            self._set_idle()
            self.btn_run.setEnabled(len(self._raw_frames) >= 2
                                    and bool(self._selected_methods()))
            self._update_adopt_actions()
            self._refresh_panes()
            return
        self.btn_run.setEnabled(len(self._raw_frames) >= 2
                                and bool(self._selected_methods()))
        self._set_idle()
        if self._auto_enabled and not self._preview_timer.isActive():
            self._preview_timer.start(0)

    def _set_idle(self):
        self.btn_run.setText(trans.t("compare_run"))
        self.progress.hide()
        if not self._auto_enabled:
            return
        methods = self._selected_methods()
        failed = [m for m in methods
                  if self._errors.get(m, (None,))[0] == self._result_key(m)]
        done = [m for m in methods
                if self._results.get(m, (None,))[0] == self._result_key(m)]
        if failed:
            self.status_label.setText(
                trans.t("compare_done_summary").format(
                    done=len(done), total=len(methods))
                + " · " + trans.t("msg_error"))
        elif done:
            self.status_label.setText(
                trans.t("compare_done_summary").format(
                    done=len(done), total=len(methods)))
        else:
            self.status_label.setText("")
        self._update_adopt_actions()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Resize:
            self._layout_timer.start(0)
        method = (watched.property("compare_method")
                  if hasattr(watched, "property") else None)
        if method and event.type() == QEvent.Type.MouseButtonPress:
            self._select_method(str(method))
            return True
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
            self._cards[method].setVisible(method in selected)
        for index, method in enumerate(selected):
            self.grid.addWidget(self._cards[method], index // columns, index % columns)

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
