"""Interactive RAW Develop Parameters dialog with live half-size preview.

Allows tuning White Balance, Highlight Recovery, Demosaic Algorithm,
Exposure EV, Auto-Bright, Output Color Space, and optional External DNG
Converter CLI path with non-blocking latest-wins preview rendering.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import numpy as np
from PyQt6.QtCore import QSignalBlocker, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.image_loader import RAW_FORMATS, is_raw_source
from core.raw_develop import (
    DEMOSAIC_ALGORITHMS,
    EXPOSURE_EV_MAX,
    EXPOSURE_EV_MIN,
    HIGHLIGHT_MODES,
    OUTPUT_COLOR_SPACES,
    WB_MODES,
    WB_SHIFT_MAX,
    WB_SHIFT_MIN,
    RawDevelopSettings,
    detect_external_dng_converter,
    load_raw_develop_settings,
    render_raw_develop_preview,
    save_raw_develop_settings,
    validate_dng_converter_path,
)
from locales import trans
from ui.styles import build_stylesheet
from utils.design_tokens import FONT, PALETTES, RADIUS, SPACE, WEIGHT, palette
from utils.file_dialogs import FileDialog

# Keep detached workers alive until finished so closing the dialog mid-decode
# never destroys a running QThread.
_ORPHAN_PREVIEW_WORKERS: set[QThread] = set()


def _orphan_preview_worker(worker: Optional[QThread]) -> None:
    if worker is None:
        return
    worker.setParent(None)
    _ORPHAN_PREVIEW_WORKERS.add(worker)

    def _release() -> None:
        if worker in _ORPHAN_PREVIEW_WORKERS:
            _ORPHAN_PREVIEW_WORKERS.discard(worker)
            worker.deleteLater()

    worker.finished.connect(_release)
    if not worker.isRunning():
        _release()


class _RawPreviewWorker(QThread):
    """Background half-size RAW preview worker."""

    previewReady = pyqtSignal(int, object, float)
    previewFailed = pyqtSignal(int, str)

    def __init__(
        self,
        generation: int,
        sample_path: str,
        settings: RawDevelopSettings,
        max_side: int = 720,
        parent=None,
    ):
        super().__init__(parent)
        self.generation = int(generation)
        self.sample_path = sample_path
        self.settings = settings.normalized()
        self.max_side = int(max_side)

    def run(self) -> None:
        started = time.perf_counter()
        try:
            bgr = render_raw_develop_preview(
                self.sample_path, self.settings, max_side=self.max_side
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.previewReady.emit(self.generation, bgr, elapsed_ms)
        except Exception as exc:  # noqa: BLE001 - surfaced in preview status label
            self.previewFailed.emit(self.generation, str(exc))


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return QPixmap()
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    if rgb.dtype != np.uint8:
        rgb = (rgb >> 8).astype(np.uint8) if rgb.dtype == np.uint16 else np.clip(rgb, 0, 255).astype(np.uint8)
    h, w, _ = rgb.shape
    image = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


def _discover_workspace_raw_paths(window) -> list[str]:
    if window is None:
        return []
    paths = getattr(window, "image_source_paths", None) or []
    result: list[str] = []
    for path in paths:
        if isinstance(path, str) and path and os.path.isfile(path) and is_raw_source(path):
            result.append(path)
    return result


def _resolve_dialog_theme(window, explicit_theme: Optional[str]) -> str:
    if explicit_theme in PALETTES:
        return explicit_theme
    if window is not None:
        resolved = getattr(window, "ui_theme", None) or getattr(window, "theme_mode", None)
        if resolved in PALETTES:
            return resolved
    return "light"


class RawDevelopDialog(QDialog):
    """Interactive dialog for configuring OpenFocus camera RAW develop settings."""

    settingsApplied = pyqtSignal(object)

    def __init__(
        self,
        parent=None,
        *,
        sample_paths: Optional[Sequence[str]] = None,
        initial_settings: Optional[RawDevelopSettings] = None,
        qsettings=None,
        theme: Optional[str] = None,
        preview_max_side: int = 720,
    ):
        super().__init__(parent)
        self._window = parent
        self._qsettings = qsettings
        self._preview_max_side = max(128, int(preview_max_side))
        self._theme = _resolve_dialog_theme(parent, theme)

        self._raw_paths: list[str] = []
        if sample_paths is not None:
            self._raw_paths = [
                os.fspath(p) for p in sample_paths if p and os.path.isfile(os.fspath(p))
            ]
        else:
            self._raw_paths = _discover_workspace_raw_paths(parent)

        self._sample_path: str = ""
        if self._raw_paths:
            idx = getattr(parent, "current_img_index", 0) if parent is not None else 0
            if isinstance(idx, int) and 0 <= idx < len(self._raw_paths):
                self._sample_path = self._raw_paths[idx]
            else:
                self._sample_path = self._raw_paths[0]

        loaded = (
            initial_settings.normalized()
            if isinstance(initial_settings, RawDevelopSettings)
            else load_raw_develop_settings(qsettings)
        )
        self._initial_settings: RawDevelopSettings = loaded
        self.applied_settings: Optional[RawDevelopSettings] = None

        self._preview_generation = 0
        self._preview_worker: Optional[_RawPreviewWorker] = None
        self._pending_preview: Optional[tuple[int, str, RawDevelopSettings]] = None
        self._current_pixmap: Optional[QPixmap] = None
        self._last_elapsed_ms: Optional[float] = None
        self._last_preview_shape: Optional[tuple[int, int]] = None
        self._closing = False

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._dispatch_preview)

        self.setObjectName("rawDevelopDialog")
        self.setProperty("controlId", "raw_develop_dialog")
        self.setWindowTitle(trans.t("raw_develop_title"))
        self.setMinimumSize(920, 620)
        self.resize(1120, 740)

        self._build_ui()
        self._apply_settings_to_controls(loaded)
        self.apply_theme(self._theme)
        self._refresh_sample_selector()
        self._update_summary_bar()
        self._update_dng_status()

        if self._sample_path:
            self.schedule_preview(immediate=True)
        else:
            self._show_empty_preview()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QWidget(self)
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(SPACE["lg"], SPACE["lg"], SPACE["lg"], SPACE["md"])
        body_lay.setSpacing(SPACE["md"])

        self.splitter = QSplitter(Qt.Orientation.Horizontal, body)
        self.splitter.setChildrenCollapsible(False)

        # --- Left Pane: Live Preview Card ---
        preview_card = QFrame(self.splitter)
        preview_card.setObjectName("rawPreviewCard")
        preview_lay = QVBoxLayout(preview_card)
        preview_lay.setContentsMargins(SPACE["md"], SPACE["md"], SPACE["md"], SPACE["md"])
        preview_lay.setSpacing(SPACE["sm"])

        preview_header = QHBoxLayout()
        preview_header.setSpacing(SPACE["sm"])

        self.lbl_preview_title = QLabel(trans.t("raw_develop_preview_heading"), preview_card)
        self.lbl_preview_title.setObjectName("rawPreviewTitle")
        preview_header.addWidget(self.lbl_preview_title)

        self.combo_sample = QComboBox(preview_card)
        self.combo_sample.setProperty("controlId", "combo_raw_sample")
        self.combo_sample.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_sample.currentIndexChanged.connect(self._on_sample_changed)
        preview_header.addWidget(self.combo_sample, 1)

        self.btn_choose_sample = QPushButton(trans.t("raw_develop_choose_sample"), preview_card)
        self.btn_choose_sample.setProperty("controlId", "btn_raw_choose_sample")
        self.btn_choose_sample.clicked.connect(self._browse_sample_raw)
        preview_header.addWidget(self.btn_choose_sample)

        preview_lay.addLayout(preview_header)

        self.preview_stage = QFrame(preview_card)
        self.preview_stage.setObjectName("rawPreviewStage")
        stage_lay = QVBoxLayout(self.preview_stage)
        stage_lay.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])

        self.lbl_preview = QLabel(self.preview_stage)
        self.lbl_preview.setObjectName("rawPreviewCanvas")
        self.lbl_preview.setProperty("controlId", "lbl_raw_preview")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setMinimumSize(400, 360)
        self.lbl_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        stage_lay.addWidget(self.lbl_preview, 1)

        preview_lay.addWidget(self.preview_stage, 1)

        footer_row = QHBoxLayout()
        footer_row.setSpacing(SPACE["sm"])

        self.lbl_preview_summary = QLabel("", preview_card)
        self.lbl_preview_summary.setObjectName("mutedLabel")
        self.lbl_preview_summary.setProperty("controlId", "lbl_raw_preview_summary")
        self.lbl_preview_summary.setMinimumWidth(0)
        self.lbl_preview_summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        footer_row.addWidget(self.lbl_preview_summary, 1)

        self.lbl_preview_status = QLabel("", preview_card)
        self.lbl_preview_status.setObjectName("rawStatusBadge")
        self.lbl_preview_status.setProperty("controlId", "lbl_raw_preview_status")
        footer_row.addWidget(self.lbl_preview_status)

        preview_lay.addLayout(footer_row)

        # --- Right Pane: Grouped Controls in Scroll Area ---
        scroll = QScrollArea(self.splitter)
        scroll.setObjectName("rawControlsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(450)

        controls_container = QWidget()
        controls_lay = QVBoxLayout(controls_container)
        controls_lay.setContentsMargins(0, 0, SPACE["sm"], 0)
        controls_lay.setSpacing(SPACE["sm"])

        controls_lay.addWidget(self._build_wb_card())
        controls_lay.addWidget(self._build_highlight_card())
        controls_lay.addWidget(self._build_demosaic_card())
        controls_lay.addWidget(self._build_tone_color_card())
        controls_lay.addWidget(self._build_dng_converter_card())
        controls_lay.addStretch(1)

        scroll.setWidget(controls_container)

        self.splitter.addWidget(preview_card)
        self.splitter.addWidget(scroll)
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 4)
        self.splitter.setSizes([600, 480])

        body_lay.addWidget(self.splitter, 1)
        root.addWidget(body, 1)

        # --- Inline validation / error banner ---
        self.lbl_error = QLabel("", self)
        self.lbl_error.setObjectName("errorLabel")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setContentsMargins(SPACE["lg"], 0, SPACE["lg"], SPACE["xs"])
        self.lbl_error.hide()
        root.addWidget(self.lbl_error)

        # --- Bottom Button Bar ---
        button_bar = QWidget(self)
        button_bar.setObjectName("dialogButtonBar")
        bar_lay = QHBoxLayout(button_bar)
        bar_lay.setContentsMargins(SPACE["lg"], SPACE["sm"], SPACE["lg"], SPACE["sm"])
        bar_lay.setSpacing(SPACE["sm"])

        self.btn_reset = QPushButton(trans.t("raw_develop_reset_defaults"), button_bar)
        self.btn_reset.setObjectName("secondaryButton")
        self.btn_reset.setProperty("controlId", "btn_raw_reset")
        self.btn_reset.clicked.connect(self._reset_defaults)
        bar_lay.addWidget(self.btn_reset)

        bar_lay.addStretch(1)

        self.btn_cancel = QPushButton(trans.t("btn_cancel"), button_bar)
        self.btn_cancel.setObjectName("secondaryButton")
        self.btn_cancel.setProperty("controlId", "btn_raw_cancel")
        self.btn_cancel.clicked.connect(self.reject)
        bar_lay.addWidget(self.btn_cancel)

        self.btn_apply = QPushButton(trans.t("raw_develop_apply"), button_bar)
        self.btn_apply.setObjectName("primaryButton")
        self.btn_apply.setProperty("controlId", "btn_raw_apply")
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self._apply)
        bar_lay.addWidget(self.btn_apply)

        root.addWidget(button_bar)

    def _make_card(self, title_key: str) -> tuple[QGroupBox, QFormLayout]:
        group = QGroupBox(trans.t(title_key).replace("&", "&&"))
        form = QFormLayout(group)
        form.setContentsMargins(SPACE["sm"], SPACE["sm"], SPACE["sm"], SPACE["sm"])
        form.setHorizontalSpacing(SPACE["sm"])
        form.setVerticalSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        return group, form

    def _build_wb_card(self) -> QGroupBox:
        card, form = self._make_card("raw_develop_group_wb")

        self.combo_wb_mode = QComboBox(card)
        self.combo_wb_mode.setProperty("controlId", "combo_raw_wb_mode")
        self.combo_wb_mode.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for mode in WB_MODES:
            self.combo_wb_mode.addItem(trans.t(f"raw_develop_wb_{mode}"), mode)
        self.combo_wb_mode.currentIndexChanged.connect(self._on_wb_mode_changed)
        form.addRow(trans.t("raw_develop_wb_mode"), self.combo_wb_mode)

        temp_row, self.slider_wb_temp, self.lbl_wb_temp_val = self._make_slider_row(
            card,
            minimum=int(round(WB_SHIFT_MIN * 100)),
            maximum=int(round(WB_SHIFT_MAX * 100)),
            step=5,
            control_id="slider_raw_wb_temp",
            on_change=self._on_slider_param_changed,
        )
        self.lbl_wb_temp_label = QLabel(trans.t("raw_develop_wb_temp"), card)
        form.addRow(self.lbl_wb_temp_label, temp_row)

        tint_row, self.slider_wb_tint, self.lbl_wb_tint_val = self._make_slider_row(
            card,
            minimum=int(round(WB_SHIFT_MIN * 100)),
            maximum=int(round(WB_SHIFT_MAX * 100)),
            step=5,
            control_id="slider_raw_wb_tint",
            on_change=self._on_slider_param_changed,
        )
        self.lbl_wb_tint_label = QLabel(trans.t("raw_develop_wb_tint"), card)
        form.addRow(self.lbl_wb_tint_label, tint_row)

        return card

    def _build_highlight_card(self) -> QGroupBox:
        card, form = self._make_card("raw_develop_group_highlight")

        self.combo_highlight = QComboBox(card)
        self.combo_highlight.setProperty("controlId", "combo_raw_highlight")
        self.combo_highlight.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for mode in HIGHLIGHT_MODES:
            self.combo_highlight.addItem(trans.t(f"raw_develop_hl_{mode}"), mode)
        self.combo_highlight.currentIndexChanged.connect(self._on_discrete_param_changed)
        form.addRow(trans.t("raw_develop_highlight_mode"), self.combo_highlight)

        hint = QLabel(trans.t("raw_develop_highlight_hint"), card)
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        form.addRow(hint)
        return card

    def _build_demosaic_card(self) -> QGroupBox:
        card, form = self._make_card("raw_develop_group_demosaic")

        self.combo_demosaic = QComboBox(card)
        self.combo_demosaic.setProperty("controlId", "combo_raw_demosaic")
        self.combo_demosaic.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for algo in DEMOSAIC_ALGORITHMS:
            self.combo_demosaic.addItem(trans.t(f"raw_develop_demosaic_{algo}"), algo)
        self.combo_demosaic.currentIndexChanged.connect(self._on_discrete_param_changed)
        form.addRow(trans.t("raw_develop_demosaic_algo"), self.combo_demosaic)
        return card

    def _build_tone_color_card(self) -> QGroupBox:
        card, form = self._make_card("raw_develop_group_tone")

        exp_row, self.slider_exposure, self.lbl_exposure_val = self._make_slider_row(
            card,
            minimum=int(round(EXPOSURE_EV_MIN * 100)),
            maximum=int(round(EXPOSURE_EV_MAX * 100)),
            step=5,
            control_id="slider_raw_exposure",
            on_change=self._on_slider_param_changed,
        )
        form.addRow(trans.t("raw_develop_exposure_ev"), exp_row)

        self.combo_output_color = QComboBox(card)
        self.combo_output_color.setProperty("controlId", "combo_raw_output_color")
        self.combo_output_color.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for space in OUTPUT_COLOR_SPACES:
            self.combo_output_color.addItem(trans.t(f"raw_develop_color_{space}"), space)
        self.combo_output_color.currentIndexChanged.connect(self._on_discrete_param_changed)
        form.addRow(trans.t("raw_develop_output_color"), self.combo_output_color)

        self.chk_auto_bright = QCheckBox(trans.t("raw_develop_auto_bright"), card)
        self.chk_auto_bright.setProperty("controlId", "chk_raw_auto_bright")
        self.chk_auto_bright.toggled.connect(self._on_discrete_param_changed)
        form.addRow(self.chk_auto_bright)

        hint = QLabel(trans.t("raw_develop_auto_bright_hint"), card)
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        form.addRow(hint)
        return card

    def _build_dng_converter_card(self) -> QGroupBox:
        card, form = self._make_card("raw_develop_group_dng")

        self.edit_dng_path = QLineEdit(card)
        self.edit_dng_path.setProperty("controlId", "edit_raw_dng_path")
        self.edit_dng_path.setPlaceholderText(trans.t("raw_develop_dng_placeholder"))
        self.edit_dng_path.textChanged.connect(self._update_dng_status)
        form.addRow(trans.t("raw_develop_dng_path"), self.edit_dng_path)

        btn_row = QWidget(card)
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(SPACE["sm"])

        self.btn_browse_dng = QPushButton(trans.t("raw_develop_dng_browse"), btn_row)
        self.btn_browse_dng.setProperty("controlId", "btn_raw_dng_browse")
        self.btn_browse_dng.clicked.connect(self._browse_dng_converter)
        btn_lay.addWidget(self.btn_browse_dng)

        self.btn_detect_dng = QPushButton(trans.t("raw_develop_dng_detect"), btn_row)
        self.btn_detect_dng.setProperty("controlId", "btn_raw_dng_detect")
        self.btn_detect_dng.clicked.connect(self._auto_detect_dng_converter)
        btn_lay.addWidget(self.btn_detect_dng)

        btn_lay.addStretch(1)
        form.addRow("", btn_row)

        self.lbl_dng_status = QLabel("", card)
        self.lbl_dng_status.setObjectName("mutedLabel")
        self.lbl_dng_status.setProperty("controlId", "lbl_raw_dng_status")
        self.lbl_dng_status.setWordWrap(True)
        form.addRow(self.lbl_dng_status)

        hint = QLabel(trans.t("raw_develop_dng_hint"), card)
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        form.addRow(hint)
        return card

    def _make_slider_row(
        self,
        parent: QWidget,
        *,
        minimum: int,
        maximum: int,
        step: int,
        control_id: str,
        on_change,
    ) -> tuple[QWidget, QSlider, QLabel]:
        row = QWidget(parent)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACE["sm"])

        slider = QSlider(Qt.Orientation.Horizontal, row)
        slider.setRange(minimum, maximum)
        slider.setSingleStep(step)
        slider.setPageStep(step * 4)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        slider.setProperty("controlId", control_id)
        slider.valueChanged.connect(on_change)
        lay.addWidget(slider, 1)

        readout = QLabel("0.00", row)
        readout.setObjectName("rawValueLabel")
        readout.setMinimumWidth(56)
        readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(readout)
        return row, slider, readout

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def apply_theme(self, theme: str) -> None:
        """Apply OpenFocus Light, Dark, or Gray design tokens to this dialog."""
        resolved = theme if theme in PALETTES else "light"
        self._theme = resolved
        t = palette(resolved)
        r = RADIUS
        f = FONT
        w = WEIGHT
        extra = f"""
QFrame#rawPreviewCard {{
    background-color: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: {r['md']}px;
}}
QFrame#rawPreviewStage {{
    background-color: {t['canvas']};
    border: 1px solid {t['border']};
    border-radius: {r['sm']}px;
}}
QLabel#rawPreviewTitle {{
    color: {t['text']};
    font-size: {f['subtitle']}pt;
    font-weight: {w['semibold']};
}}
QLabel#rawPreviewCanvas {{
    background: transparent;
    color: {t['canvasText']};
    font-size: {f['body']}pt;
}}
QLabel#rawValueLabel {{
    color: {t['text']};
    font-weight: {w['medium']};
}}
QLabel#rawStatusBadge {{
    background-color: {t['accentSoft']};
    color: {t['accentSoftText']};
    border-radius: {r['xs']}px;
    padding: 2px 8px;
    font-size: {f['caption']}pt;
    font-weight: {w['medium']};
}}
QLabel#rawStatusBadge[state="error"] {{
    background-color: {t['dangerSoft']};
    color: {t['danger']};
}}
QLabel#rawStatusBadge[state="idle"] {{
    background-color: {t['surfaceSunken']};
    color: {t['textMuted']};
}}
"""
        self.setStyleSheet(build_stylesheet(resolved) + extra)

    # ------------------------------------------------------------------
    # Settings <-> Controls Synchronization
    # ------------------------------------------------------------------

    def current_settings(self) -> RawDevelopSettings:
        """Collect current widget values into a normalized ``RawDevelopSettings``."""
        return RawDevelopSettings(
            wb_mode=str(self.combo_wb_mode.currentData() or "camera"),
            wb_temp_shift=self.slider_wb_temp.value() / 100.0,
            wb_tint_shift=self.slider_wb_tint.value() / 100.0,
            highlight_mode=str(self.combo_highlight.currentData() or "clip"),
            demosaic=str(self.combo_demosaic.currentData() or "ahd"),
            exposure_ev=self.slider_exposure.value() / 100.0,
            auto_bright=self.chk_auto_bright.isChecked(),
            output_color=str(self.combo_output_color.currentData() or "srgb"),
            external_dng_converter_path=self.edit_dng_path.text().strip(),
        ).normalized()

    def _apply_settings_to_controls(self, cfg: RawDevelopSettings) -> None:
        norm = cfg.normalized()
        blockers = [
            QSignalBlocker(self.combo_wb_mode),
            QSignalBlocker(self.slider_wb_temp),
            QSignalBlocker(self.slider_wb_tint),
            QSignalBlocker(self.combo_highlight),
            QSignalBlocker(self.combo_demosaic),
            QSignalBlocker(self.slider_exposure),
            QSignalBlocker(self.combo_output_color),
            QSignalBlocker(self.chk_auto_bright),
            QSignalBlocker(self.edit_dng_path),
        ]
        try:
            self.combo_wb_mode.setCurrentIndex(max(0, self.combo_wb_mode.findData(norm.wb_mode)))
            self.slider_wb_temp.setValue(int(round(norm.wb_temp_shift * 100)))
            self.slider_wb_tint.setValue(int(round(norm.wb_tint_shift * 100)))
            self.combo_highlight.setCurrentIndex(max(0, self.combo_highlight.findData(norm.highlight_mode)))
            self.combo_demosaic.setCurrentIndex(max(0, self.combo_demosaic.findData(norm.demosaic)))
            self.slider_exposure.setValue(int(round(norm.exposure_ev * 100)))
            self.combo_output_color.setCurrentIndex(max(0, self.combo_output_color.findData(norm.output_color)))
            self.chk_auto_bright.setChecked(bool(norm.auto_bright))
            self.edit_dng_path.setText(norm.external_dng_converter_path)
        finally:
            del blockers
        self._sync_wb_sliders_enabled()
        self._update_value_readouts()

    def _sync_wb_sliders_enabled(self) -> None:
        is_custom = self.combo_wb_mode.currentData() == "custom"
        for widget in (
            self.lbl_wb_temp_label,
            self.slider_wb_temp,
            self.lbl_wb_temp_val,
            self.lbl_wb_tint_label,
            self.slider_wb_tint,
            self.lbl_wb_tint_val,
        ):
            widget.setEnabled(is_custom)

    def _update_value_readouts(self) -> None:
        temp = self.slider_wb_temp.value() / 100.0
        tint = self.slider_wb_tint.value() / 100.0
        ev = self.slider_exposure.value() / 100.0
        self.lbl_wb_temp_val.setText(f"{temp:+.2f}")
        self.lbl_wb_tint_val.setText(f"{tint:+.2f}")
        self.lbl_exposure_val.setText(f"{ev:+.2f} EV")

    def _update_summary_bar(self) -> None:
        cfg = self.current_settings()
        parts = [
            self.combo_wb_mode.currentText(),
            self.combo_highlight.currentText(),
            cfg.demosaic.upper(),
            f"{cfg.exposure_ev:+.2f} EV",
            self.combo_output_color.currentText(),
        ]
        if self._last_preview_shape is not None and self._last_elapsed_ms is not None:
            w, h = self._last_preview_shape
            parts.append(f"{w}×{h} ({self._last_elapsed_ms:.0f} ms)")
        self.lbl_preview_summary.setText("  ·  ".join(parts))

    def _set_status_badge(self, text: str, state: str = "active") -> None:
        self.lbl_preview_status.setText(text)
        self.lbl_preview_status.setProperty("state", state)
        style = self.lbl_preview_status.style()
        if style is not None:
            style.unpolish(self.lbl_preview_status)
            style.polish(self.lbl_preview_status)

    # ------------------------------------------------------------------
    # Sample Selection & Live Preview Scheduling
    # ------------------------------------------------------------------

    def _refresh_sample_selector(self) -> None:
        blocker = QSignalBlocker(self.combo_sample)
        try:
            self.combo_sample.clear()
            if not self._raw_paths:
                self.combo_sample.addItem(trans.t("raw_develop_no_sample"), "")
                self.combo_sample.setEnabled(False)
            else:
                self.combo_sample.setEnabled(True)
                for path in self._raw_paths:
                    self.combo_sample.addItem(os.path.basename(path), path)
                idx = self.combo_sample.findData(self._sample_path)
                self.combo_sample.setCurrentIndex(max(0, idx))
        finally:
            del blocker

    def set_sample_path(self, path: str) -> None:
        """Select a RAW file for live preview and immediately schedule decoding."""
        cleaned = os.fspath(path) if path else ""
        if cleaned and os.path.isfile(cleaned):
            if cleaned not in self._raw_paths:
                self._raw_paths.insert(0, cleaned)
            self._sample_path = cleaned
            self._refresh_sample_selector()
            self.schedule_preview(immediate=True)
        else:
            self._sample_path = ""
            self._refresh_sample_selector()
            self._show_empty_preview()

    def _on_sample_changed(self, _index: int) -> None:
        data = str(self.combo_sample.currentData() or "")
        if data and os.path.isfile(data):
            self._sample_path = data
            self.schedule_preview(immediate=True)

    def _browse_sample_raw(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(RAW_FORMATS))
        start_dir = os.path.dirname(self._sample_path) if self._sample_path else ""
        chosen, _flt = FileDialog.getOpenFileName(
            self,
            trans.t("raw_develop_choose_sample"),
            start_dir,
            f"RAW ({patterns});;All Files (*)",
        )
        if chosen:
            self.set_sample_path(chosen)

    def _on_wb_mode_changed(self, _index: int) -> None:
        self._sync_wb_sliders_enabled()
        self._update_summary_bar()
        self.schedule_preview(immediate=True)

    def _on_discrete_param_changed(self, *_args) -> None:
        self._update_summary_bar()
        self.schedule_preview(immediate=True)

    def _on_slider_param_changed(self, *_args) -> None:
        self._update_value_readouts()
        self._update_summary_bar()
        self.schedule_preview(immediate=False)

    def schedule_preview(self, *, immediate: bool = False) -> None:
        """Schedule a latest-wins async preview render for the active RAW sample."""
        if self._closing or not self._sample_path:
            return
        self._preview_generation += 1
        self._pending_preview = (
            self._preview_generation,
            self._sample_path,
            self.current_settings(),
        )
        self._set_status_badge(trans.t("raw_develop_preview_rendering"), "active")
        self._preview_timer.stop()
        if immediate:
            self._dispatch_preview()
        else:
            self._preview_timer.start(120)

    def _dispatch_preview(self) -> None:
        if self._closing or self._pending_preview is None:
            return
        if self._preview_worker is not None and self._preview_worker.isRunning():
            # Latest-wins: keep only the newest pending request until current decode finishes.
            return
        gen, sample_path, cfg = self._pending_preview
        self._pending_preview = None
        worker = _RawPreviewWorker(
            gen, sample_path, cfg, max_side=self._preview_max_side, parent=self
        )
        worker.previewReady.connect(self._on_preview_ready)
        worker.previewFailed.connect(self._on_preview_failed)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        self._preview_worker = worker
        worker.start()

    def _on_preview_ready(self, generation: int, bgr: np.ndarray, elapsed_ms: float) -> None:
        if self._closing or generation != self._preview_generation:
            return
        pixmap = _bgr_to_qpixmap(bgr)
        if pixmap.isNull():
            self._set_status_badge(trans.t("raw_develop_preview_failed"), "error")
            return
        self._current_pixmap = pixmap
        self._last_preview_shape = (int(bgr.shape[1]), int(bgr.shape[0]))
        self._last_elapsed_ms = float(elapsed_ms)
        self._render_scaled_pixmap()
        self._set_status_badge(trans.t("raw_develop_preview_ready"), "active")
        self._update_summary_bar()

    def _on_preview_failed(self, generation: int, error: str) -> None:
        if self._closing or generation != self._preview_generation:
            return
        self._current_pixmap = None
        self.lbl_preview.setPixmap(QPixmap())
        self.lbl_preview.setText(
            trans.t("raw_develop_preview_error").format(error=error)
        )
        self._set_status_badge(trans.t("raw_develop_preview_failed"), "error")

    def _on_worker_finished(self, worker: _RawPreviewWorker) -> None:
        if worker is self._preview_worker:
            self._preview_worker = None
        worker.deleteLater()
        if not self._closing and self._pending_preview is not None:
            self._dispatch_preview()

    def _show_empty_preview(self) -> None:
        self._current_pixmap = None
        self._last_preview_shape = None
        self._last_elapsed_ms = None
        self.lbl_preview.setPixmap(QPixmap())
        self.lbl_preview.setText(trans.t("raw_develop_empty_preview"))
        self._set_status_badge(trans.t("raw_develop_no_sample"), "idle")
        self._update_summary_bar()

    def _render_scaled_pixmap(self) -> None:
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        target = self.lbl_preview.contentsRect().size()
        if target.width() <= 4 or target.height() <= 4:
            return
        scaled = self._current_pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_preview.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_scaled_pixmap()

    # ------------------------------------------------------------------
    # External DNG Converter Actions
    # ------------------------------------------------------------------

    def _browse_dng_converter(self) -> None:
        current = self.edit_dng_path.text().strip()
        start_dir = os.path.dirname(current) if current else ""
        flt = "Executable (*.exe);;All Files (*)" if os.name == "nt" else "All Files (*)"
        chosen, _ = FileDialog.getOpenFileName(
            self, trans.t("raw_develop_dng_browse_title"), start_dir, flt
        )
        if chosen:
            self.edit_dng_path.setText(chosen)

    def _auto_detect_dng_converter(self) -> None:
        detected = detect_external_dng_converter()
        if detected:
            self.edit_dng_path.setText(detected)
            self.lbl_dng_status.setText(trans.t("raw_develop_dng_detected").format(path=detected))
        else:
            self.lbl_dng_status.setText(trans.t("raw_develop_dng_not_found"))

    def _update_dng_status(self, *_args) -> None:
        path = self.edit_dng_path.text().strip()
        if not path:
            self.lbl_dng_status.setText(trans.t("raw_develop_dng_optional_status"))
            return
        ok, info = validate_dng_converter_path(path)
        if ok:
            self.lbl_dng_status.setText(trans.t("raw_develop_dng_valid"))
        else:
            self.lbl_dng_status.setText(trans.t("raw_develop_dng_invalid"))

    # ------------------------------------------------------------------
    # Dialog Actions & Cleanup
    # ------------------------------------------------------------------

    def _reset_defaults(self) -> None:
        self.lbl_error.hide()
        self._apply_settings_to_controls(RawDevelopSettings())
        self._update_dng_status()
        self._update_summary_bar()
        if self._sample_path:
            self.schedule_preview(immediate=True)

    def _apply(self) -> None:
        cfg = self.current_settings()
        if cfg.external_dng_converter_path:
            ok, resolved = validate_dng_converter_path(cfg.external_dng_converter_path)
            if not ok:
                self.lbl_error.setText(trans.t("raw_develop_dng_invalid"))
                self.lbl_error.show()
                self.edit_dng_path.setFocus()
                return
            cfg.external_dng_converter_path = resolved
        self.lbl_error.hide()
        saved = save_raw_develop_settings(cfg, settings=self._qsettings)
        self.applied_settings = saved
        if self._window is not None:
            self._window.raw_develop_settings = saved
            try:
                from core.source_frames import SOURCE_CACHE
                SOURCE_CACHE.clear()
            except Exception:
                pass
        self.settingsApplied.emit(saved)
        self.accept()

    def _stop_preview_worker(self) -> None:
        self._closing = True
        self._preview_timer.stop()
        self._pending_preview = None
        worker = self._preview_worker
        self._preview_worker = None
        if worker is not None:
            try:
                worker.previewReady.disconnect(self._on_preview_ready)
                worker.previewFailed.disconnect(self._on_preview_failed)
            except Exception:
                pass
            if worker.isRunning():
                if not worker.wait(300):
                    _orphan_preview_worker(worker)
                    return
            worker.deleteLater()

    def done(self, result: int) -> None:
        self._stop_preview_worker()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_preview_worker()
        super().closeEvent(event)
