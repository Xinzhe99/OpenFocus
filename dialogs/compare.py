"""Interactive comparison using one aligned stack and one inspection region."""
import os
from collections.abc import Mapping

from PyQt6.QtCore import QEvent, QSize, QSignalBlocker, Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSplitter, QToolTip, QVBoxLayout, QWidget,
)

from controllers.output_manager import workspace_fusions
from core.compare_worker import AlignmentWorker, PreviewWorker, DisplayWorker
from core.source_previews import DISPLAY_REGION_MAX_SIDE, SourcePreviewWorker, SourceRegionWorker
from core.source_frames import PREVIEW_EDGE, ProcessedFrame, SourceFrame, display_image, preview_ready
from locales import trans
from utils.frame_pixels import BoundedPreviewWorker
from utils.image_utils import cv2_to_pixmap
from utils.output_sharpen import DEFAULT_SHARPEN_STRENGTH
from utils.settings_store import get_settings
from core.algorithms import (FUSION_DEFAULT, alignment_steps, fusion_algorithms, get_fusion,
    registration_parameter_specs, registration_algorithms, resolve_fusion_parameters,
    desktop_registration_parameters)
from ui.algorithm_parameters import AlgorithmParameterPanel
from ui.algorithm_combo import (apply_method_availability, fusion_label, fusion_lock_reason,
                                selected_item_enabled)
from core.experimental import alignment_is_experimental, experimental_enabled
from widgets.compare_canvas import (
    DEFAULT_DETAIL_SIZE, DetailCanvas, SourceCanvas,
)
from widgets.flow_layout import FlowPanel
from widgets.stepper import StepperSpinBox


def _method_label(method):
    return trans.t(get_fusion(method).label_key)


def _contains(bounds, region):
    if region is None:
        return False
    bx, by, bw, bh = bounds
    x, y, width, height = region
    return (width > 0 and height > 0 and x >= bx and y >= by
            and x + width <= bx + bw and y + height <= by + bh)


class _ActionFlowPanel(FlowPanel):
    """Prefer one row while retaining the flow layout's wrapping minimum."""

    def sizeHint(self):  # noqa: N802
        items = [self._flow.itemAt(index) for index in range(self._flow.count())]
        hints = [item.sizeHint() for item in items if not item.isEmpty()]
        width = sum(hint.width() for hint in hints) + 8 * max(0, len(hints) - 1)
        return QSize(width, self.heightForWidth(width))


class _MethodFlowPanel(FlowPanel):
    def mousePressEvent(self, event):  # noqa: N802
        child = self.childAt(event.position().toPoint())
        reason = child.property('unavailableReason') if child is not None else None
        if reason:
            QToolTip.showText(event.globalPosition().toPoint(), reason, self)
            return
        super().mousePressEvent(event)

class CompareDialog(QDialog):
    """Edit each method independently; publish only current preview requests."""

    def __init__(self, window_main, parent=None):
        super().__init__(parent or window_main)
        self.window_main = window_main
        self.setWindowTitle(trans.t("compare_title"))
        self.resize(1280, 820)
        # 1280x820 does not fit a 1366x768 work area at 125-150 % scaling; the
        # action bar then sat below the screen. Clamp the first size the same
        # way the retouch window does and let the splitter/scroll areas shrink.
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(1280, max(480, available.width() - 80)),
                        min(820, max(360, available.height() - 100)))
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
        self._reverse_registration = False
        self._epoch = 0
        self._revision = 0
        self._worker = None
        self._source_preview_workers = {}
        self._display_workers = {}
        self._source_preview_generation = 0
        self._reference_worker = None
        self._viewport_tiles = ()
        self._viewport_next = 0
        self._viewport_request = None
        self._viewport_serial = 0
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
        self._edited_parameters = set()
        self._parameter_panels = {}
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
        self._workspace_outputs = [record for record in workspace_fusions(window_main)
            if record.context.matches_stack(self._source_revision, self._source_indices)]
        manager = getattr(window_main, "label_manager", None)
        frame_ids = manager.frame_ids("input") if manager is not None else []
        self._frame_ids = [frame_ids[i] for i in self._source_indices] if frame_ids else None
        from utils.scalebar import effective_px_um
        from core.image_loader import stack_icc_profile
        self._calibration = {
            "pixel_size_um": effective_px_um(window_main),
            "source_px_um": getattr(window_main, "source_px_um", None),
            "source_scale_factor": getattr(window_main, "current_scale_factor", 1.0),
            "icc_profile": stack_icc_profile(window_main, self._source_indices),
            "linear_dng_metadata": getattr(window_main, "linear_dng_metadata", None),
            "source_paths": [window_main.image_source_paths[i] for i in self._source_indices]
                if len(getattr(window_main, "image_source_paths", ())) == self._total_frames else [],
            "frame_exif": [getattr(window_main, "image_source_exifs", ())[i]
                           for i in self._source_indices]
                if len(getattr(window_main, "image_source_exifs", ())) == self._total_frames else [],
            "frame_icc_profiles": [getattr(window_main, "image_icc_profiles", ())[i]
                                   for i in self._source_indices]
                if len(getattr(window_main, "image_icc_profiles", ())) == self._total_frames else [],
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
        method = (window_main.current_fusion_method()
                  if hasattr(window_main, 'current_fusion_method') else FUSION_DEFAULT)
        if method == 'none':
            method = self._first_available_method()
        box = self._checks[method]
        self._reuse_workspace_outputs()
        self._show_frame(0)
        if self._selected_methods():
            self._methods_changed(True)
            self._preview_timer.stop()
            self._set_idle()
        elif box.isEnabled():
            box.setChecked(True)
        else:
            self.param_scroll.setVisible(False)
            self.btn_run.setEnabled(False)
            self.status_label.setText(box.property('unavailableReason'))
        for _control in ("btn_run", "btn_adopt_params", "btn_adopt_output",
                         "btn_retry", "btn_verify", "align_combo",
                         "mode_combo", "view_combo", "registration_parameter_panel"):
            _widget = getattr(self, _control, None)
            if _widget is not None:
                _widget.setProperty("controlId", _control)
        for _method, _box in self._checks.items():
            _box.setProperty("controlId", f"method_{_method}")

    def _first_available_method(self) -> str:
        if self._checks[FUSION_DEFAULT].isEnabled():
            return FUSION_DEFAULT
        return next((name for name, box in self._checks.items() if box.isEnabled()), FUSION_DEFAULT)

    def _apply_method_availability(self) -> None:
        """Keep locked methods visible with their opt-in and dependency guidance."""
        from core.multi_focus_fusion import is_ai_fusion_available

        available = bool(is_ai_fusion_available())
        for spec in fusion_algorithms():
            box = self._checks.get(spec.id)
            if box is None:
                continue
            reason = fusion_lock_reason(spec, self.window_main, available)
            box.setText(fusion_label(spec))
            box.setEnabled(not reason)
            box.setProperty("unavailableReason", reason)
            hint = trans.t(spec.hint_key) if spec.hint_key else ""
            box.setToolTip("\n\n".join(text for text in (hint, reason) if text))

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
            "exposure_match": bool(getattr(window, "exposure_match", False)),
            "sharpen_output": bool(getattr(window, "sharpen_output", False)),
            "sharpen_strength": getattr(window, "sharpen_strength", DEFAULT_SHARPEN_STRENGTH),
            "sort_mode": "desc" if getattr(window, "stack_reverse_order", False) else "asc",
        }
        for key in ("tile_block_size", "tile_overlap", "tile_threshold"):
            value = getattr(window, key, None)
            if value is not None:
                settings[key] = value
        settings['method_options'] = {}
        getter = getattr(window, 'brightness_option_for', None)
        for spec in fusion_algorithms():
            base_correct = (bool(getter('base_correct', spec.id)) if callable(getter)
                            else spec.default_base_correct)
            settings['method_options'][spec.id] = {
                'exposure_match': bool(getter('exposure_match', spec.id)) if callable(getter) else spec.default_exposure_match,
                'base_frame': ('first' if spec.id in ('openfocus_a', 'openfocus_b', 'openfocus_c')
                               else 'last') if base_correct else 'none',
            }
        return settings

    def _build_ui(self):
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        root.setSpacing(8)
        methods_row = QHBoxLayout()
        methods_host = QWidget()
        methods_layout = QVBoxLayout(methods_host)
        methods_layout.setContentsMargins(0, 0, 0, 0)
        methods_layout.setSpacing(4)
        methods_layout.addWidget(QLabel(trans.t("compare_pick")))
        methods_policy = QSizePolicy(QSizePolicy.Policy.Preferred,
                                     QSizePolicy.Policy.Minimum)
        methods_policy.setHeightForWidth(True)
        methods_host.setSizePolicy(methods_policy)
        self._checks = {}
        # Same family order and captions as the main/batch/depth selectors, so
        # a method keeps its familiar neighbour and family.
        from ui.algorithm_combo import fusion_method_groups
        for label, members in fusion_method_groups():
            header = QLabel(label)
            header.setObjectName("sectionLabel")
            methods_layout.addWidget(header)
            flow = _MethodFlowPanel(h_spacing=12, v_spacing=6)
            methods_layout.addWidget(flow)
            for spec in members:
                box = QCheckBox(trans.t(spec.label_key), flow)
                box.toggled.connect(self._methods_changed)
                flow.add(box)
                self._checks[spec.id] = box
        self._apply_method_availability()
        methods_row.addWidget(methods_host, 1)
        self.btn_run = QPushButton(trans.t("compare_run"))
        self.btn_run.setObjectName("primaryButton")
        self.btn_run.clicked.connect(self.run_comparison)
        methods_row.addWidget(self.btn_run, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(methods_row)

        self.source_summary = QLabel("")
        self.source_summary.setObjectName("mutedLabel")
        self.source_summary.setWordWrap(True)
        root.addWidget(self.source_summary)

        settings = FlowPanel(h_spacing=12, v_spacing=6)
        registration_group = QWidget()
        registration_row = QHBoxLayout(registration_group)
        registration_row.setContentsMargins(0, 0, 0, 0)
        registration_label = QLabel(trans.t("group_registration"))
        registration_row.addWidget(registration_label)
        self.align_combo = QComboBox()
        for spec in registration_algorithms():
            self.align_combo.addItem(trans.t(spec.label_key), spec.id)
        apply_method_availability(self.align_combo, registration_algorithms(),
                                  self.window_main, alignment=True)
        current_align = (self.window_main.current_alignment_method()
                         if hasattr(self.window_main, "current_alignment_method") else "none")
        self.align_combo.setCurrentIndex(max(0, self.align_combo.findData(current_align)))
        self.align_combo.currentIndexChanged.connect(self._alignment_changed)
        registration_label.setBuddy(self.align_combo)
        registration_row.addWidget(self.align_combo)
        settings.add(registration_group)
        mode_group = QWidget()
        mode_row = QHBoxLayout(mode_group)
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_label = QLabel(trans.t("compare_mode"))
        mode_row.addWidget(mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(trans.t("compare_full"), False)
        self.mode_combo.addItem(trans.t("compare_roi"), True)
        self.mode_combo.setToolTip(trans.t("compare_mode_hint"))
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        mode_label.setBuddy(self.mode_combo)
        mode_row.addWidget(self.mode_combo)
        settings.add(mode_group)
        # Viewing is independent of computing: a full result can be inspected
        # as a detail crop or as the whole frame without re-fusing.
        view_group = QWidget()
        view_row = QHBoxLayout(view_group)
        view_row.setContentsMargins(0, 0, 0, 0)
        view_label = QLabel(trans.t("compare_view"))
        view_row.addWidget(view_label)
        self.view_combo = QComboBox()
        self.view_combo.addItem(trans.t("compare_view_full"), "full")
        self.view_combo.addItem(trans.t("compare_view_detail"), "detail")
        self.view_combo.setToolTip(trans.t("compare_view_hint"))
        self.view_combo.currentIndexChanged.connect(self._view_changed)
        view_label.setBuddy(self.view_combo)
        view_row.addWidget(self.view_combo)
        settings.add(view_group)
        self.fit_button = QPushButton(trans.t("retouch_fit"))
        settings.add(self.fit_button)
        self.actual_button = QPushButton(trans.t('retouch_zoom_100'))
        settings.add(self.actual_button)
        self.fullscreen_button = QPushButton(trans.t("compare_fullscreen"))
        self.fullscreen_button.clicked.connect(self._toggle_fullscreen)
        settings.add(self.fullscreen_button)
        root.addWidget(settings)
        self._registration_drafts = {}
        self._editing_alignment = None
        self.registration_parameter_panel = AlgorithmParameterPanel(
            self, hidden_parameters=('failure_policy',))
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

        self.preview_note = QLabel(trans.t("compare_preview_note"))
        self.preview_note.setWordWrap(True)
        self.preview_note.setObjectName("mutedLabel")
        self.preview_note.setVisible(False)
        root.addWidget(self.preview_note)

        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.split.setChildrenCollapsible(False)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.ref_canvas = SourceCanvas(trans.t("compare_reference"), left)
        self.ref_canvas.setToolTip(trans.t("compare_hint"))
        self.ref_canvas.set_detail_size(self._detail_size)
        self.ref_canvas.regionChanged.connect(self._region_changed)
        self.ref_canvas.viewportChanged.connect(self._refresh_reference_viewport)
        self.fit_button.clicked.connect(self.ref_canvas.fit)
        self.actual_button.clicked.connect(self.ref_canvas.actual_size)
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
        self.preview_size_spin.setMinimumWidth(110)
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
            canvas.setToolTip(trans.t("compare_candidate_hint"))
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
        actions_row = QHBoxLayout()
        actions = FlowPanel(h_spacing=8, v_spacing=4)
        self.btn_verify = QPushButton(trans.t("compare_verify_full"))
        self.btn_verify.clicked.connect(self._verify_full)
        actions.add(self.btn_verify)
        self.btn_retry = QPushButton(trans.t("compare_retry_method"))
        self.btn_retry.clicked.connect(self._retry_method)
        actions.add(self.btn_retry)
        actions_row.addWidget(actions, 1)
        primary_actions = _ActionFlowPanel(h_spacing=8, v_spacing=4)
        self.btn_adopt_params = QPushButton(trans.t("compare_adopt_params"))
        self.btn_adopt_params.clicked.connect(self._adopt_params)
        self.btn_adopt_params.setEnabled(False)
        primary_actions.add(self.btn_adopt_params)
        self.btn_adopt_output = QPushButton(trans.t("compare_adopt_output"))
        self.btn_adopt_output.setObjectName("primaryButton")
        self.btn_adopt_output.clicked.connect(self._adopt_output)
        self.btn_adopt_output.setEnabled(False)
        primary_actions.add(self.btn_adopt_output)
        close_button = QPushButton(trans.t("btn_close"))
        close_button.clicked.connect(self.reject)
        primary_actions.add(close_button)
        actions_row.addWidget(primary_actions)
        outer.addLayout(actions_row)
        self._layout_timer.start(0)

    def _initial_params(self, method):
        getter = getattr(self.window_main, 'fusion_parameters_for', None)
        return (getter(method) if callable(getter)
                else resolve_fusion_parameters(method))

    def _build_param_rows(self):
        for method in self._checks:
            params = self._initial_params(method)
            self._method_params[method] = params
            specs = get_fusion(method).parameters
            if not specs:
                self.param_widgets[method] = {}
                continue
            group = QFrame(self.param_host)
            group.setFrameShape(QFrame.Shape.StyledPanel)
            group.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            row = QVBoxLayout(group)
            row.setContentsMargins(8, 4, 8, 4)
            row.setSpacing(6)
            title = QLabel(_method_label(method), group)
            font = title.font()
            font.setBold(True)
            title.setFont(font)
            row.addWidget(title)
            panel = AlgorithmParameterPanel(group)
            panel.set_parameters(specs, params)
            self._parameter_panels[method] = panel
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
        for method in set(self._results) - set(selected):
            self._results.pop(method, None)
            self._full_results.pop(method, None)
            self._errors.pop(method, None)
            self._canvases[method].set_result(None, None)
            worker = self._display_workers.get(method)
            if worker is not None:
                worker.cancel()
        reverse = self._registration_is_reversed(selected)
        if reverse != self._reverse_registration:
            self._reverse_registration = reverse
            self._invalidate_alignment()
        for method, group in self._param_rows.items():
            group.setVisible(method in selected)
        self.param_scroll.setVisible(any(method in self._param_rows for method in selected))
        self.param_host.relayout()
        self._layout_timer.start(0)
        self._request_changed()

    def _params_changed(self, method, values):
        if self._method_params[method] == values:
            return
        self._method_params[method] = dict(values)
        self._edited_parameters.add(method)
        self._results.pop(method, None)
        self._stale_methods.add(method)
        self._full_results.pop(method, None)
        self._errors.pop(method, None)
        self._update_adopt_actions()
        self._request_changed()

    def _registration_is_reversed(self, methods):
        # One coordinate system for every candidate. If any selected method
        # requires reverse registration, all candidates use that same stack.
        from core.workers import _ordered_openfocus_registration
        alignment = self.align_combo.currentData()
        return any(_ordered_openfocus_registration(alignment, method, self._shared['sort_mode'])
                   for method in methods)

    def _direction_matches_single_method(self, method):
        return self._reverse_registration == self._registration_is_reversed([method])

    def _sync_registration_parameters(self):
        method = self.align_combo.currentData()
        if method not in self._registration_drafts:
            getter = getattr(self.window_main, 'registration_parameters_for', None)
            self._registration_drafts[method] = (getter(method) if callable(getter)
                else desktop_registration_parameters(method))
        self._editing_alignment = method
        specs = registration_parameter_specs(method)
        self.registration_parameter_panel.set_parameters(specs, self._registration_drafts[method])
        self.registration_parameter_panel.setVisible(method != 'openfocus' and bool(self.registration_parameter_panel.editors))
        self._shared['registration_parameters'] = dict(self._registration_drafts[method])

    def _registration_parameters_changed(self, values):
        self._registration_drafts[self._editing_alignment] = dict(values)
        self._alignment_changed(0)

    def _alignment_changed(self, _index):
        self._sync_registration_parameters()
        self._reverse_registration = self._registration_is_reversed(self._selected_methods())
        self._invalidate_alignment()
        self._request_changed()

    def _invalidate_alignment(self):
        self._stale_methods.update(self._results)
        self._epoch += 1
        self._aligned_frames = None
        self._frames = self._raw_frames
        self._results.clear()
        self._full_results.clear()
        self._errors.clear()
        self._show_frame(self._ref_index)
        self._update_adopt_actions()

    def _mode_changed(self, _index):
        roi = bool(self.mode_combo.currentData())
        self.preview_note.setVisible(roi)
        self._stale_methods.update(self._results)
        self._results.clear()
        self._full_results.clear()
        self._errors.clear()
        self._update_adopt_actions()
        self._request_changed()

    def _show_frame(self, index):
        self._cancel_source_previews()
        if self._closing or not self._frames:
            return
        self._ref_index = max(0, min(index, len(self._frames) - 1))
        frame = self._frames[self._ref_index]
        self.ref_canvas.setToolTip("")
        cached = (isinstance(frame, SourceFrame) and preview_ready(frame)
                  and frame.preview_is_available)
        small_array = (not isinstance(frame, (SourceFrame, ProcessedFrame))
                       and max(frame.shape[:2]) <= PREVIEW_EDGE)
        if cached or small_array:
            self._set_reference_image(frame, display_image(frame), getattr(frame, 'preview_is_display', False))
        else:
            with QSignalBlocker(self.ref_canvas):
                # Geometry is known before the reference pixels arrive.
                # Keep ready candidates inspectable while loading the frame.
                self.ref_canvas.set_pixmap(None, source_size=(frame.shape[1], frame.shape[0]))
            request = self._source_preview_generation
            if isinstance(frame, SourceFrame):
                worker = SourcePreviewWorker(self._frames, request, [self._ref_index], selected_display=True)
                worker.ready.connect(self._source_preview_ready)
                worker.failed.connect(self._source_preview_failed)
                self._source_preview_workers[worker] = worker.request_cancel
            else:
                worker = BoundedPreviewWorker(frame, self._calibration['linear_dng_metadata'], request)
                self._source_preview_workers[worker] = worker.requestInterruption
            worker.finished.connect(self._source_preview_finished)
            worker.start()
        names = self._source_names
        name = (os.path.basename(str(names[self._ref_index]))
                if self._ref_index < len(names) else "")
        self.frame_label.setText(trans.t("compare_frame_of").format(
            i=self._ref_index + 1, n=len(self._frames), name=name))
        self.btn_prev_frame.setEnabled(self._ref_index > 0)
        self.btn_next_frame.setEnabled(self._ref_index < len(self._frames) - 1)
        # Switching the reference frame changes no fusion inputs or coordinates.
        self._refresh_panes()

    def _cancel_source_previews(self):
        self._source_preview_generation += 1
        for cancel in self._source_preview_workers.values():
            cancel()
        self._viewport_tiles = ()
        self._viewport_request = None
        if self._reference_worker is not None:
            self._reference_worker.request_cancel()
        self.ref_canvas.clear_tiles()
        self.ref_canvas.set_native_status('')

    def _set_reference_image(self, frame, image, is_display):
        metadata = self._calibration['linear_dng_metadata']
        if image is not None and metadata is not None and not is_display:
            from utils.linear_dng import develop_linear
            image = develop_linear(image, metadata)
        size = (frame.shape[1], frame.shape[0])
        with QSignalBlocker(self.ref_canvas):
            self.ref_canvas.set_pixmap(cv2_to_pixmap(image) if image is not None else None,
                                       source_size=size, preview_only=isinstance(frame, (SourceFrame, ProcessedFrame)),
                                       preserve_tiles=True)
        self._refresh_reference_viewport()

    def _source_preview_ready(self, request, index, identity, _thumbnail):
        if (self._closing or request != self._source_preview_generation
                or index != self._ref_index or not 0 <= index < len(self._frames)
                or identity != id(self._frames[index])):
            return
        frame = self._frames[index]
        if not preview_ready(frame):
            return
        image = display_image(frame) if frame.preview_is_available else None
        self._set_reference_image(frame, image, frame.preview_is_display)
        self._reference_preview_published()

    def _reference_preview_published(self):
        self._refresh_panes()
        if self._auto_enabled and self._worker is None:
            self._preview_timer.start(0)

    def _source_preview_failed(self, request, message):
        if not self._closing and request == self._source_preview_generation:
            self.ref_canvas.set_native_status(trans.t('msg_display_error_title'))
            self.ref_canvas.setToolTip(message)

    def _source_preview_finished(self):
        worker = self.sender()
        self._source_preview_workers.pop(worker, None)
        if (isinstance(worker, BoundedPreviewWorker) and not self._closing
                and worker.key == self._source_preview_generation):
            if worker.error:
                self._source_preview_failed(worker.key, worker.error)
            else:
                self._set_reference_image(self._frames[self._ref_index], worker.pixels, True)
                self._reference_preview_published()
        worker.deleteLater()
        if self._closing:
            self._finalize_close()

    def _refresh_reference_viewport(self):
        if self._closing:
            return
        tiles = self.ref_canvas.viewport_tiles(DISPLAY_REGION_MAX_SIDE)
        if tiles == self._viewport_tiles:
            return
        self._viewport_tiles = tiles
        self._viewport_next = 0
        self._viewport_request = None
        if tiles:
            self.ref_canvas.stash_tiles()
        else:
            self.ref_canvas.clear_tiles()
        self.ref_canvas.set_native_status(trans.t('native_detail_loading') if tiles else '')
        self.ref_canvas.setToolTip(trans.t('compare_hint'))
        if tiles:
            self._submit_reference_tile()
        elif self._reference_worker is not None:
            self._reference_worker.request_cancel()

    def _submit_reference_tile(self):
        if self._viewport_next >= len(self._viewport_tiles):
            self.ref_canvas.clear_retained_tiles()
            self.ref_canvas.set_native_status('')
            return
        worker = self._reference_worker
        if worker is None:
            worker = SourceRegionWorker(self._frames[self._ref_index], self._source_preview_generation,
                                        metadata=self._calibration['linear_dng_metadata'])
            worker.ready.connect(self._reference_tile_ready)
            worker.failed.connect(self._reference_tile_failed)
            worker.finished.connect(self._reference_worker_finished)
            self._reference_worker = worker
            worker.start()
        elif worker.cancelled:
            return  # Its finished signal starts the current frame's request.
        rect, size = self._viewport_tiles[self._viewport_next]
        self._viewport_serial += 1
        self._viewport_request = ('viewport', self._viewport_serial, rect, size)
        worker.submit(self._viewport_request)

    def _reference_tile_ready(self, generation, request, pixels, bounds):
        if (self._closing or generation != self._source_preview_generation
                or request != self._viewport_request):
            return
        pixmap = cv2_to_pixmap(pixels) if pixels is not None else None
        if pixmap is None or pixmap.isNull():
            self._reference_tile_failed(generation, trans.t('msg_display_error_title'))
            return
        self.ref_canvas.add_tile(bounds, pixmap)
        self._viewport_request = None
        self._viewport_next += 1
        self._submit_reference_tile()

    def _reference_tile_failed(self, generation, message):
        if (self._closing or generation != self._source_preview_generation
                or self._reference_worker is None or self._reference_worker.cancelled):
            return
        self.ref_canvas.clear_retained_tiles()
        self.ref_canvas.set_native_status(trans.t('msg_display_error_title'))
        self.ref_canvas.setToolTip(message)
        self._viewport_next = len(self._viewport_tiles)
        self._viewport_request = None
        self._reference_worker.request_cancel()

    def _reference_worker_finished(self):
        worker = self.sender()
        self._reference_worker = None
        worker.deleteLater()
        if self._closing:
            self._finalize_close()
        elif self._viewport_next < len(self._viewport_tiles):
            self._submit_reference_tile()

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
        self._refresh_panes()

    def _view_changed(self, _index):
        """Switch detail/overview without recomputing anything."""
        mode = self.view_combo.currentData()
        for canvas in self._canvases.values():
            canvas.set_view_mode(mode)
        self._refresh_panes()

    def _region_changed(self, _region):
        if self._closing:
            return
        self._refresh_panes()
        if not self.mode_combo.currentData() or self._worker_kind == 'alignment':
            return
        self._update_adopt_actions()
        methods = self._selected_methods()
        if len(self._raw_frames) >= 2 and any(not self._result_is_current(m) for m in methods):
            self._auto_enabled = True
            # Let the running request finish and take the latest geometry at
            # the next start. Restarting on each mouse event starves the preview.
            if not self._preview_timer.isActive():
                self._preview_timer.start()

    def _reuse_workspace_outputs(self, methods=None):
        if (not self._workspace_outputs or len(self._raw_frames) < 2
                or not selected_item_enabled(self.align_combo)
                or (not experimental_enabled(self.window_main)
                    and alignment_is_experimental(self.align_combo.currentData()))):
            return
        opening = methods is None
        pending = ({m for m, box in self._checks.items() if box.isEnabled()} if opening
                   else {m for m in methods if not self._result_is_current(m)})
        had_alignment = self._aligned_frames is not None
        alignment = self.align_combo.currentData()
        roi = bool(self.mode_combo.currentData())
        strength = self._shared['sharpen_strength'] if self._shared['sharpen_output'] else 0
        records = self._workspace_outputs
        if alignment != 'none' and not had_alignment:
            # Prefer a complete registration grid before considering a result
            # whose fusion may have trimmed the edges of its aligned frames.
            records = sorted(records, key=lambda record: record.context.algorithm == 'dct')
        for record in records:
            context = record.context
            method = context.algorithm
            if method not in pending:
                continue
            options = context.parameters
            brightness = self._shared['method_options'][method]
            saved_strength = (options.get('sharpen_strength', DEFAULT_SHARPEN_STRENGTH)
                              if options.get('sharpen_output') else 0)
            if (bool(options.get('exposure_match')) != brightness['exposure_match']
                    or options.get('base_frame', 'none') != brightness['base_frame']
                    or saved_strength != strength):
                continue
            reverse = (self._registration_is_reversed([method]) if opening and not self._results
                       else self._reverse_registration)
            if not context.matches_registration(alignment, self._shared['registration_parameters'],
                                                self._shared['reg_downscale_width'], reverse):
                continue
            if context.parameters.get('defect_mask_path') or context.parameters.get('sensor_denoise'):
                continue
            spec = get_fusion(method)
            if (spec.sort_frames and
                    context.parameters.get('sort_mode', 'asc') != self._shared['sort_mode']):
                continue
            if (len(context.frames) != len(self._raw_frames)
                    or any(frame.aligned != (alignment != 'none') for frame in context.frames)):
                continue
            candidate = self._workspace_geometry(record, roi)
            if candidate is None:
                continue
            image, bounds, aligned_grid = candidate
            saved = context.parameters.get('fusion_parameters', context.parameters)
            if not isinstance(saved, Mapping) or not all(p.key in saved for p in spec.parameters):
                continue
            try:
                params = resolve_fusion_parameters(method, {p.key: saved[p.key] for p in spec.parameters})
            except ValueError:
                continue
            if method in self._edited_parameters and self._method_params[method] != params:
                continue
            if opening:
                self._reverse_registration = reverse
                with QSignalBlocker(self._checks[method]):
                    self._checks[method].setChecked(True)
            if self._method_params[method] != params:
                self._method_params[method] = params
                self._parameter_panels[method].set_parameters(spec.parameters, params)
            previous = self._results.get(method)
            self._results[method] = {
                'key': self._result_key(method), 'image': image, 'bounds': bounds,
                'full': not roi, 'record': record,
                'display': (previous.get('display') if previous and previous.get('record') is record
                            else None),
            }
            self._errors.pop(method, None)
            pending.remove(method)
            if aligned_grid:
                self._aligned_frames = [frame.image for frame in context.frames]
                self._frames = self._aligned_frames
            if not pending:
                break
        if not opening and not had_alignment and self._aligned_frames is not None:
            self._show_frame(self._ref_index)

    def _workspace_geometry(self, record, roi):
        context = record.context
        image = record.image
        height, width = image.shape[:2]
        if (width, height) != (context.width, context.height) or image.dtype != context.dtype:
            return None
        x = y = 0
        frame_offset = (0, 0)
        aligned = self.align_combo.currentData() != 'none'
        grid_h, grid_w = self._frames[0].shape[:2]
        grid_unknown = aligned and self._aligned_frames is None
        if context.roi_rect is not None:
            if not roi or grid_unknown or context.roi_mode not in ('crop', 'paste'):
                return None
            x, y, rw, rh = context.roi_rect
            if not _contains((0, 0, grid_w, grid_h), context.roi_rect):
                return None
            if context.roi_mode == 'paste':
                if (width, height) != (grid_w, grid_h):
                    return None
                frame_offset = (x, y)
                image = image[y:y + rh, x:x + rw]
                height, width = image.shape[:2]
            if not _contains(context.roi_rect, (x, y, width, height)):
                return None
        elif (not grid_unknown and (width, height) != (grid_w, grid_h)
              and context.algorithm != 'dct'):
            return None
        if any((frame.offset_x, frame.offset_y) != frame_offset
               or frame.image.shape[:2] != (height, width) for frame in context.frames):
            return None
        bounds = (x, y, width, height)
        if not _contains((0, 0, grid_w, grid_h), bounds):
            return None
        if roi and not _contains(bounds, self.ref_canvas.region()):
            return None
        # Registration crops to common coverage. DCT can also trim after
        # registration; use it as a reference only if its input was intact.
        if grid_unknown and context.algorithm == 'dct' and (width, height) != (grid_w, grid_h):
            details = context.fusion_details
            if details is None or details.input_size != (width, height):
                return None
        return image, bounds, grid_unknown

    def _result_key(self, method):
        roi = bool(self.mode_combo.currentData())
        strength = self._shared['sharpen_strength'] if self._shared['sharpen_output'] else 0
        return (self._epoch, roi, tuple(sorted(self._method_params[method].items())),
                strength, self.ref_canvas.region() if roi else None)

    def _refresh_panes(self):
        region = self.ref_canvas.region()
        for method in self._selected_methods():
            canvas = self._canvases[method]
            key = self._result_key(method)
            entry = self._results.get(method)
            current = entry is not None and entry['key'] == key
            if current:
                record = entry.get('record')
                status = (trans.t('compare_workspace_result').format(name=record.label)
                          if record is not None else trans.t('compare_state_current'))
            elif self._errors.get(method, (None,))[0] == key:
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
            canvas.set_view_mode(self.view_combo.currentData())
            same_fusion = entry is not None and entry['key'][:-1] == key[:-1]
            pending = False
            display_error = ''
            if current:
                pending, display_error = self._request_display(method, entry, region)
            elif same_fusion:
                pending = True  # Prior ROI pixels while new coverage is rendered.
            else:
                worker = self._display_workers.get(method)
                if worker is not None:
                    worker.cancel()
            cache = entry.get('display') if same_fusion else None
            if cache is not None:
                canvas.set_result(cache[1], cache[2], full=cache[3], pending=pending)
            else:
                canvas.set_result(None, None)
            if current:
                canvas.set_status(display_error or (trans.t('native_detail_loading') if pending else ''))
            else:
                canvas.set_status(status)

    def _request_display(self, method, entry, region):
        canvas = self._canvases[method]
        full = self.view_combo.currentData() == 'full' and entry['full']
        if not full and not _contains(entry['bounds'], region):
            return False, ''
        size = max(1, round(canvas.width() * canvas.devicePixelRatioF()))
        display_key = (entry['key'], full, None if full else region, size)
        entry['display_target'] = display_key
        cached = entry.get('display')
        if cached is not None and cached[0] == display_key:
            return False, ''
        if entry.get('display_error_key') == display_key:
            return False, entry['display_error']
        worker = self._display_workers.get(method)
        if worker is not None:
            # A size or crop change alone leaves the in-flight sample usable
            # as a temporary display. Read the latest geometry when it retires.
            if worker.entry is not entry:
                worker.cancel()
            return True, ''
        retained = sum(cache['display'][1].width() * cache['display'][1].height() * 4
                       for cache in self._results.values() if cache.get('display'))
        retained += max(0, len(self._selected_methods()) - 1) * size * size * 4
        worker = DisplayWorker(entry['image'], entry['bounds'], region, full, size,
                               self._calibration.get('linear_dng_metadata'), retained)
        worker.entry = entry
        worker.display_key = display_key
        self._display_workers[method] = worker
        worker.ready.connect(lambda pixels, bounds, whole, m=method, w=worker:
                             self._display_ready(m, w, pixels, bounds, whole))
        worker.failed.connect(lambda message, m=method, w=worker:
                              self._display_failed(m, w, message))
        worker.finished.connect(lambda m=method, w=worker: self._display_finished(m, w))
        worker.start()
        return True, ''

    def _display_ready(self, method, worker, pixels, bounds, whole):
        if not self._current_display(method, worker):
            return
        cached = worker.entry.get('display')
        if cached is not None and cached[0] == worker.entry.get('display_target'):
            return  # Geometry returned to the pixels already on screen.
        pixmap = cv2_to_pixmap(pixels)
        if pixmap.isNull():
            self._display_failed(method, worker, trans.t('msg_proc_error'))
            return
        worker.entry['display'] = (worker.display_key, pixmap, tuple(bounds), whole)

    def _current_display(self, method, worker):
        return (not self._closing and self._display_workers.get(method) is worker
                and self._results.get(method) is worker.entry and not worker._should_cancel()
                and worker.entry['key'][:-1] == self._result_key(method)[:-1])

    def _display_failed(self, method, worker, message):
        if self._current_display(method, worker):
            worker.entry['display_error_key'] = worker.display_key
            worker.entry['display_error'] = message

    def _display_finished(self, method, worker):
        if self._display_workers.get(method) is not worker:
            return
        self._display_workers.pop(method, None)
        worker.deleteLater()
        if self._closing:
            self._finalize_close()
        else:
            self._refresh_panes()

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
        if not selected_item_enabled(self.align_combo):
            self._set_idle()
            self.status_label.setText(self.align_combo.currentData(Qt.ItemDataRole.ToolTipRole))
            return
        if (not experimental_enabled(self.window_main)
                and alignment_is_experimental(self.align_combo.currentData())):
            self._set_idle()
            self.status_label.setText(trans.t('experimental_locked_hint'))
            return
        self._apply_method_availability()
        for method in self._selected_methods():
            if not self._checks[method].isEnabled():
                self._set_idle()
                self.status_label.setText(self._checks[method].property('unavailableReason'))
                return
        methods = self._selected_methods()
        if not methods or len(self._raw_frames) < 2:
            self._set_idle()
            return
        self._reuse_workspace_outputs(methods)
        region = self.ref_canvas.region()
        if region is None and self.mode_combo.currentData():
            return
        pending = []
        for method in methods:
            key = self._result_key(method)
            if not self._result_is_current(method) and self._errors.get(method, (None,))[0] != key:
                pending.append(method)
        if not pending:
            self._refresh_panes()
            self._set_idle()
            return
        if self._aligned_frames is None:
            alignment = self.align_combo.currentData()
            if alignment == "none":
                self._aligned_frames = self._raw_frames
                self._frames = self._aligned_frames
            else:
                worker = AlignmentWorker(self._raw_frames, alignment, self._shared, self,
                                         reverse_registration=self._reverse_registration)
                worker.ready.connect(self._alignment_ready)
                worker.failed.connect(self._alignment_failed)
                self.status_label.setText(trans.t("render_stage_align"))
                self.progress.setRange(0, 0)
                self._launch_worker(worker, "alignment")
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
        direction_mismatch = (bool(method) and self._result_is_current(method)
                              and not self._direction_matches_single_method(method))
        self.btn_verify.setVisible(bool(method) and
                                   (bool(self.mode_combo.currentData()) or direction_mismatch))
        reason = trans.t('compare_direction_needs_verify') if direction_mismatch else ''
        self.btn_adopt_params.setToolTip(reason)
        self.btn_verify.setToolTip(reason)
        self.btn_verify.setEnabled(available and self._result_is_current(method))
        self.btn_retry.setVisible(bool(method) and method in self._errors)
        self.btn_retry.setEnabled(available and method in self._errors)
        overview = self.view_combo.model().item(self.view_combo.findData("full"))
        if overview is not None:
            overview.setEnabled(not bool(self.mode_combo.currentData()))
        if self.mode_combo.currentData() and self.view_combo.currentData() == "full":
            self.view_combo.setCurrentIndex(self.view_combo.findData("detail"))
        eligible = (bool(method) and method in self._selected_methods()
                    and self._result_is_current(method)
                    and self._worker is None and not self._closing
                    and not self._published and not self._publishing)
        self.btn_adopt_params.setEnabled(eligible and not direction_mismatch)
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
        return bool(cached) and cached['key'] == self._result_key(method)

    def _adopt_params(self) -> None:
        """Copy the candidate's parameters to the main window; no render."""
        method = self._selected_method
        if (not method or not self.btn_adopt_params.isEnabled()
                or not self._result_is_current(method)
                or not self._direction_matches_single_method(method)):
            return
        window = self.window_main
        params = dict(self._method_params.get(method, {}))
        record = self._results[method].get('record')
        options = record.context.parameters if record is not None else self._shared['method_options'][method]
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
                        options.get('exposure_match', False))
                    window.set_brightness_option('base_correct', method,
                        options.get('base_frame', 'none') != 'none')
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
                fusion_details=entry.get('fusion_details'),
                **self._calibration)
            record = OutputRecord.create(
                image, label, context=context, kind="compare", exported=False,
                fusion_details=entry.get('fusion_details'))
            window.output_manager.publish_compare_output(record)
            self._published = True
        except Exception as exc:
            self.status_label.setText(f"{trans.t('msg_error')}: {exc}")
        finally:
            self._publishing = False
            self._update_adopt_actions()
        if self._published:
            self.accept()

    def _preview_ready(self, method, image, bounds):
        if not self._current_worker() or method not in self._selected_methods():
            return
        key = self._active_keys.get(method)
        if key != self._result_key(method):
            return
        x, y, width, height = tuple(bounds)
        is_full = (not key[1] and x == 0 and y == 0
                   and (height, width) == self._aligned_frames[0].shape[:2]
                   and (height, width) == image.shape[:2])
        previous = self._results.get(method)
        same_fusion = previous is not None and previous['key'][:-1] == key[:-1]
        entry = {'key': key, 'image': image, 'bounds': tuple(bounds),
                 'full': is_full, 'display': previous.get('display') if same_fusion else None,
                 'fusion_details': self._worker.fusion_details.get(method)}
        self._results[method] = entry
        if is_full:
            self._full_results[method] = entry
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
            self._finalize_close()
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
                if self._result_is_current(m)]
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
        if method and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
                if not event.isAutoRepeat():
                    self._select_method(str(method))
                return True
        if method and event.type() in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
            watched.update()
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

    def event(self, event):
        result = super().event(event)
        screen_events = (getattr(QEvent.Type, 'ScreenChangeInternal', None),
                         getattr(QEvent.Type, 'DevicePixelRatioChange', None))
        if (event.type() in screen_events and hasattr(self, 'ref_canvas')
                and not self._closing):
            self._layout_timer.start(0)
            self.ref_canvas.display_density_changed()
        return result

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
        self._refresh_panes()
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
        if self._closing:
            return
        get_settings().setValue("compare/preview_size", self._detail_size)
        self._closing = True
        self._close_code = result
        self._auto_enabled = False
        self._revision += 1
        self._preview_timer.stop()
        self._layout_timer.stop()
        for worker in self._display_workers.values():
            worker.cancel()
        self._cancel_source_previews()
        if self._worker is not None:
            self._worker.cancel()
        if self._has_workers():
            self.status_label.setText(trans.t("compare_stopping"))
            self.setEnabled(False)
            return  # Close after every worker releases its source references.
        self._finalize_close()

    def shutdown_running(self):
        if not self._closing:
            self.reject()
        return self._has_workers()

    def _has_workers(self):
        return bool(self._worker is not None or self._reference_worker is not None
                    or self._source_preview_workers or self._display_workers)

    def _finalize_close(self):
        if self._has_workers():
            return
        self._results.clear()
        self._full_results.clear()
        self._errors.clear()
        self._active_keys.clear()
        for canvas in self._canvases.values():
            canvas.set_result(None, None)
        with QSignalBlocker(self.ref_canvas):
            self.ref_canvas.set_pixmap(None)
        self._aligned_frames = None
        self._frames = []
        self._raw_frames = []
        self._workspace_outputs.clear()
        self._calibration = {}
        super().done(self._close_code)
        self.deleteLater()

    def closeEvent(self, event):  # noqa: N802
        self.done(QDialog.DialogCode.Rejected)
        if self._has_workers():
            event.ignore()
        else:
            event.accept()
