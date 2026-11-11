"""Retouch workspace: compare a render's source frames with its result.

The left canvas is read-only; the right canvas is the only writable one.
Both share one zoom/center state, so the same image coordinate always sits
at the same screen position.  Every stroke is locked to the source frame
and brush settings that were active when the pointer went down.

The dialog publishes through `apply_callback(record, image)`: it never
mutates the output list itself.  A failed publish keeps the buffer and the
dialog open.
"""
from __future__ import annotations

import os
import weakref

import cv2
import numpy as np
from PyQt6.QtCore import (QEvent, QPointF, QRect, QRectF, QSize, Qt, QThread,
                          QTimer, pyqtSignal)
from PyQt6.QtGui import (QBrush, QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap)
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QDialog,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar, QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QSplitter, QStyle, QVBoxLayout, QWidget,
)

from locales import trans
from utils.image_utils import thumbnail_bgr, to_display_uint8
from utils.path_identity import same_path
from utils.platform_utils import configure_native_worker_stack
from utils.retouch import RetouchDocument, adjust_tones, prepare_retouch_document
from core.source_frames import display_image
from utils.frame_pixels import run_pixel_worker
from widgets.flow_layout import FlowPanel
from widgets.wipe_compare import WipeCompareWidget
from dialogs.render_settings import _OptionLabel


def _add_wrapped_option(layout, option):
    """Use the render-settings label convention for narrow option columns."""
    text = option.text()
    option.setText("")
    option.setAccessibleName(text)
    label = _OptionLabel(option)
    label.setText(text)
    label.setToolTip(option.toolTip())
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    # Theme padding gives an empty native control a taller sizeHint than a
    # text line. Bound it to the first line so its indicator shares the label's
    # centre even when the translated label wraps onto several lines.
    option.setStyleSheet('QCheckBox, QRadioButton { padding: 0; margin: 0; spacing: 0; }')
    style = option.style()
    radio = isinstance(option, QRadioButton)
    width = style.pixelMetric(QStyle.PixelMetric.PM_ExclusiveIndicatorWidth if radio
                              else QStyle.PixelMetric.PM_IndicatorWidth)
    height = style.pixelMetric(QStyle.PixelMetric.PM_ExclusiveIndicatorHeight if radio
                               else QStyle.PixelMetric.PM_IndicatorHeight)
    # The theme's 16px indicator also has a 1px border on each side.
    option.setFixedSize(max(width, 18), max(height, 18, label.fontMetrics().height()))
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(8)
    row_layout.addWidget(option, 0, Qt.AlignmentFlag.AlignTop)
    row_layout.addWidget(label, 1)
    layout.addWidget(row)
    return row

_SHARPNESS_WORKERS = set()
_SOURCE_WORKERS = set()


class _SourceWorker(QThread):
    """Load just the selected source and prepare its display off the GUI thread."""

    def __init__(self, frame, index, request, metadata):
        super().__init__()
        configure_native_worker_stack(self)
        self.input_frame = frame
        self.index = index
        self.request = request
        self.metadata = metadata
        self.frame = None
        self.display = None
        self.valid = None
        self.error = None

    def run(self):
        try:
            self.frame = RetouchDocument.resolve_source_frame(
                self.input_frame, self.isInterruptionRequested)
            if self.isInterruptionRequested():
                return
            image = self.frame.image
            if self.metadata is not None:
                from utils.linear_dng import develop_linear
                image = develop_linear(image, self.metadata)
            self.display = to_display_uint8(image)
            self.valid = self.frame.valid_mask()
        except Exception as exc:
            self.error = exc


class _StandaloneRetouchWorker(QThread):
    progress = pyqtSignal(int, int)

    def __init__(self, result_path, source_paths):
        super().__init__()
        configure_native_worker_stack(self)
        self.result_path = result_path
        self.source_paths = tuple(source_paths)
        self.result = None
        self.error = None
        self.cancelled = False

    def run(self):
        from utils.standalone_retouch import load_standalone_retouch
        try:
            def progress(index, total):
                self.progress.emit(index, total)
                return not self.isInterruptionRequested()

            self.result = load_standalone_retouch(
                self.result_path, self.source_paths, progress=progress,
                cancelled=self.isInterruptionRequested)
            self.cancelled = self.isInterruptionRequested() or self.result is None
            if self.cancelled:
                self.result = None
        except Exception as exc:
            self.cancelled = self.isInterruptionRequested()
            if not self.cancelled:
                self.error = exc


class StandaloneRetouchLoadDialog(QDialog):
    """Keep the loader and its parent alive until native decoding stops."""

    def __init__(self, result_path, source_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t('standalone_retouch_action'))
        self.setModal(True)
        layout = QVBoxLayout(self)
        self.label = QLabel(trans.t('standalone_retouch_sources'))
        layout.addWidget(self.label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(source_paths) + 1)
        layout.addWidget(self.progress_bar)
        self.cancel_button = QPushButton(trans.t('btn_cancel'))
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        self._cancel_requested = False
        self.worker = _StandaloneRetouchWorker(result_path, source_paths)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._finished)

    def _progress(self, index, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(index)

    def _finished(self):
        if self._cancel_requested:
            self.worker.cancelled = True
            self.worker.result = None
            super().reject()
        else:
            self.accept()

    def start(self):
        if self._cancel_requested:
            self.worker.cancelled = True
            return QDialog.DialogCode.Rejected
        self.worker.start()
        return self.exec()

    def reject(self):
        self._cancel_requested = True
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText(trans.t('compare_stopping'))
            self.label.setText(trans.t('compare_stopping'))
        else:
            self.worker.cancelled = True
            self.worker.result = None
            super().reject()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)

def _release_dialog_buffers(dialog_ref):
    """Drop large Python-owned buffers even if a Qt callback retains the wrapper."""
    dialog = dialog_ref()
    if dialog is None:
        return
    for name in ("doc", "record", "_source_qimage", "_pure_source_qimage",
                 "_source_display", "_source_clean_display", "_source_valid",
                 "_before_qimage", "_result_display", "_result_qimage", "_sharpness_maps"):
        setattr(dialog, name, None)


def _qimage_from_bgr(bgr8: np.ndarray, copy: bool = True) -> QImage:
    arr = np.ascontiguousarray(bgr8)
    if arr.ndim == 2:
        h, w = arr.shape
        image = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w = arr.shape[:2]
        image = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_BGR888)
    return image.copy() if copy else image


class _ViewState:
    """Shared image coordinate -> screen coordinate state for both canvases."""

    def __init__(self) -> None:
        self.scale = 1.0
        self.center = QPointF(0.0, 0.0)
        self.fit = True


class _NumberRow(QWidget):
    """A labelled slider plus a numeric input, kept in sync."""

    valueChanged = pyqtSignal(int)

    def __init__(self, label: str, minimum: int, maximum: int, value: int,
                 parent=None):
        super().__init__(parent)
        self._sync = False
        self._label = QLabel(label)
        self._label.setWordWrap(True)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.spin = QSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setKeyboardTracking(False)
        self.spin.ensurePolished()
        metrics = self.spin.fontMetrics()
        text_width = max(metrics.horizontalAdvance(str(minimum)),
                         metrics.horizontalAdvance(str(maximum)))
        self.spin.setMinimumWidth(max(self.spin.sizeHint().width(), text_width + 54))
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._label)
        layout.addLayout(row)
        self.setValue(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> int:
        return int(self.spin.value())

    def setValue(self, value: int) -> None:
        self._sync = True
        self.slider.setValue(int(value))
        self.spin.setValue(int(value))
        self._sync = False

    def set_label(self, text: str) -> None:
        self._label.setText(text)

    def _from_slider(self, value: int) -> None:
        if self._sync:
            return
        self._sync = True
        self.spin.setValue(value)
        self._sync = False
        self.valueChanged.emit(int(value))

    def _from_spin(self, value: int) -> None:
        if self._sync:
            return
        self._sync = True
        self.slider.setValue(value)
        self._sync = False
        self.valueChanged.emit(int(value))


class _SharpnessWorker(QThread):
    """Compute one shared-scale detail map for every source frame."""

    ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, frames, request_id: int, preview_side: int = 768):
        super().__init__()
        self.frames = list(frames)
        self.request_id = int(request_id)
        self.preview_side = int(preview_side)

    def run(self) -> None:
        try:
            detail = {}
            for index, frame in enumerate(self.frames):
                if self.isInterruptionRequested():
                    return
                from core.source_frames import ProcessedFrame, resolve_frame
                image = (resolve_frame(frame.image, should_cancel=self.isInterruptionRequested)
                         if isinstance(frame.image, ProcessedFrame)
                         else display_image(frame.image))
                if image is None or image.size == 0:
                    continue
                h, w = image.shape[:2]
                scale = min(1.0, self.preview_side / float(max(h, w, 1)))
                if scale < 1.0:
                    image = cv2.resize(
                        image, (max(1, int(round(w * scale))),
                                max(1, int(round(h * scale)))),
                        interpolation=cv2.INTER_AREA)
                display = to_display_uint8(image)
                if display.ndim == 2:
                    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
                energy = cv2.GaussianBlur(
                    np.abs(lap), (0, 0), max(1.0, display.shape[1] / 300.0))
                detail[index] = energy
            if not detail:
                self.ready.emit(self.request_id, {})
                return
            peak = max(float(np.percentile(m, 99.5))
                       for m in detail.values())
            peak = max(1e-6, peak)
            maps = {}
            for index, energy in detail.items():
                if self.isInterruptionRequested():
                    return
                norm = np.clip(energy / peak, 0.0, 1.0)
                maps[index] = cv2.applyColorMap(
                    (norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            self.ready.emit(self.request_id, maps)
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))


class _ToneWorker(QThread):
    def __init__(self, doc, exposure, contrast, **adjustments):
        super().__init__()
        self.doc = doc
        self.exposure = exposure
        self.contrast = contrast
        self.adjustments = adjustments
        self.status = ""
        self.error = ""

    def run(self):
        try:
            self.status = self.doc.adjust_tones(
                self.exposure, self.contrast, self.isInterruptionRequested, **self.adjustments)
        except Exception as exc:
            self.error = str(exc)


class _CheckpointWorker(QThread):
    def __init__(self, doc, operation, path=None):
        super().__init__()
        self.doc = doc
        self.operation = operation
        self.path = path
        self.error = None
        self.cancelled = False

    def run(self):
        try:
            if not self.doc.checkpoint_ready:
                self.doc.initialize_checkpoint(self.isInterruptionRequested)
            if self.operation == "save":
                self.doc.save_checkpoint(self.path, self.isInterruptionRequested)
            else:
                self.doc.load_checkpoint(self.path, self.isInterruptionRequested)
        except Exception as exc:
            self.error = exc
            self.cancelled = self.isInterruptionRequested()


class _CheckpointDialog(QDialog):
    """Keep the workspace modal while checkpoint I/O finishes or cancels."""
    def __init__(self, doc, path, parent, operation="load"):
        super().__init__(parent)
        self.setWindowTitle(trans.t("retouch_checkpoint_" + operation))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(trans.t("retouch_checkpoint_" +
                                        ("saving" if operation == "save" else "loading"))))
        self.cancel_button = QPushButton(trans.t("btn_cancel"))
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        self.worker = _CheckpointWorker(doc, operation, path)
        self.worker.finished.connect(self.accept)

    def start(self):
        self.worker.start()
        self.exec()

    def reject(self):
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText(trans.t("compare_stopping"))
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)

class _ToneDialog(QDialog):
    """Preview a sampled current result, then commit one history entry."""

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.worker = None
        self.setWindowTitle(trans.t("retouch_tones_title"))
        self.setModal(True)
        stride = max(1, int(np.ceil(max(doc.height, doc.width) / 1280)))
        self.sample = np.array(doc.working[::stride, ::stride], copy=True, order="C")
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(40)
        self._preview_timer.timeout.connect(self._refresh_preview)

        layout = QVBoxLayout(self)
        columns = QSplitter(Qt.Orientation.Horizontal)
        columns.setChildrenCollapsible(False)
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 4, 0)
        settings_header = QHBoxLayout()
        settings_title = QLabel(trans.t("retouch_tones_title"))
        font = settings_title.font()
        font.setBold(True)
        settings_title.setFont(font)
        settings_header.addWidget(settings_title, 1)
        self.reset_button = QPushButton(trans.t("btn_reset"))
        self.reset_button.setAutoDefault(False)
        self.reset_button.clicked.connect(self._reset_adjustments)
        settings_header.addWidget(self.reset_button)
        settings_layout.addLayout(settings_header)

        self.controls = QScrollArea()
        self.controls.setWidgetResizable(True)
        self.controls.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        rows = QVBoxLayout(page)
        rows.setContentsMargins(0, 0, 8, 0)
        self.exposure = _NumberRow(trans.t("retouch_exposure_amount"), -200, 200, 0)
        self.contrast = _NumberRow(trans.t("retouch_tones_contrast"), -100, 100, 0)
        self.highlights = _NumberRow(trans.t("retouch_tones_highlights"), -100, 100, 0)
        self.shadows = _NumberRow(trans.t("retouch_tones_shadows"), -100, 100, 0)
        self.saturation = _NumberRow(trans.t("retouch_tones_saturation"), -100, 100, 0)
        self.temperature = _NumberRow(trans.t("retouch_tones_temperature"), -100, 100, 0)
        self.tone_rows = (self.exposure, self.contrast, self.highlights, self.shadows,
                          self.saturation, self.temperature)
        for row in self.tone_rows:
            row.valueChanged.connect(self._schedule_preview)
            rows.addWidget(row)
        rows.addStretch(1)
        self.controls.setWidget(page)
        settings_layout.addWidget(self.controls, 1)
        settings.setMinimumWidth(250)
        settings.setMaximumWidth(400)
        columns.addWidget(settings)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(4, 0, 0, 0)
        header = QHBoxLayout()
        preview_title = QLabel(trans.t("retouch_tones_preview"))
        font = preview_title.font()
        font.setBold(True)
        preview_title.setFont(font)
        header.addWidget(preview_title, 1)
        self.compare_box = QCheckBox(trans.t("retouch_tones_wipe"))
        self.compare_box.setToolTip(trans.t("wipe_hint"))
        header.addWidget(self.compare_box)
        preview_layout.addLayout(header)
        self.preview = WipeCompareWidget()
        self.preview.setMinimumSize(260, 220)
        self.preview.setToolTip(trans.t("wipe_hint"))
        self.preview.set_comparison_enabled(False)
        self.compare_box.toggled.connect(self.preview.set_comparison_enabled)
        preview_layout.addWidget(self.preview, 1)
        columns.addWidget(preview_panel)
        columns.setStretchFactor(0, 0)
        columns.setStretchFactor(1, 1)
        columns.setSizes([300, 780])
        layout.addWidget(columns, 1)

        buttons = QHBoxLayout()
        self.apply_button = QPushButton(trans.t("btn_ok"))
        self.apply_button.setObjectName("primaryButton")
        self.apply_button.setAutoDefault(False)
        self.apply_button.clicked.connect(self._apply)
        self.cancel_button = QPushButton(trans.t("btn_cancel"))
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        from utils.ui_utils import size_dialog_to_content
        size_dialog_to_content(self, self.controls, preferred_width=1080)
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            frame = self.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth, None, self)
            title = self.style().pixelMetric(QStyle.PixelMetric.PM_TitleBarHeight, None, self)
            max_height = max(1, screen.availableGeometry().height() - title - 2 * frame)
            self.resize(self.width(), min(max_height, max(self.height(), 680)))
        self._before_pixmap = self._make_preview(self.sample)
        self._refresh_preview()

    def _make_preview(self, image):
        metadata = getattr(getattr(self.parent(), "record", None), "context", None)
        metadata = getattr(metadata, "linear_dng_metadata", None)
        if metadata is not None:
            from utils.linear_dng import develop_linear
            image = develop_linear(image, metadata)
        display = to_display_uint8(image)
        if display.ndim == 3 and display.shape[2] == 4:
            display = display[..., :3]
        return QPixmap.fromImage(_qimage_from_bgr(display))

    def _schedule_preview(self, *_args):
        if not self._preview_timer.isActive():
            self._preview_timer.start()

    def _refresh_preview(self):
        exposure = self.exposure.value() / 100.0
        contrast = self.contrast.value() / 100.0
        adjustments = self._adjustments()
        if any((exposure, contrast, *adjustments.values())):
            image = adjust_tones(self.sample, exposure, contrast, **adjustments)
            after = self._make_preview(image)
        else:
            after = self._before_pixmap
        self.preview.set_images(self._before_pixmap, after,
                                trans.t("retouch_tones_before"),
                                trans.t("retouch_tones_after"))

    def _reset_adjustments(self):
        self._preview_timer.stop()
        for row in self.tone_rows:
            row.setValue(0)
        self._refresh_preview()

    def _adjustments(self):
        return {name: getattr(self, name).value() / 100.0 for name in
                ("highlights", "shadows", "saturation", "temperature")}

    def _apply(self):
        if self.worker is not None:
            return
        if not any(row.value() for row in self.tone_rows):
            self.accept()
            return
        if self.doc.working.nbytes > self.doc.history.max_bytes:
            QMessageBox.warning(self, trans.t("msg_warning"),
                                trans.t("retouch_tones_oversize"))
            return
        self._preview_timer.stop()
        for row in self.tone_rows:
            row.setEnabled(False)
        self.compare_box.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.worker = _ToneWorker(self.doc, self.exposure.value() / 100.0,
                                  self.contrast.value() / 100.0, **self._adjustments())
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _finished(self):
        worker = self.worker
        self.worker = None
        worker.deleteLater()
        if worker.error:
            QMessageBox.critical(self, trans.t("msg_error"), worker.error)
            for row in self.tone_rows:
                row.setEnabled(True)
            self.compare_box.setEnabled(True)
            self.reset_button.setEnabled(True)
            self.apply_button.setEnabled(True)
            self.cancel_button.setEnabled(True)
            self.cancel_button.setText(trans.t("btn_cancel"))
            self._schedule_preview()
        elif worker.status in ("committed", "empty"):
            self.accept()
        elif worker.status == "record_limit":
            QMessageBox.warning(self, trans.t("msg_warning"),
                                trans.t("retouch_checkpoint_record_limit"))
            super().reject()
        else:
            super().reject()

    def reject(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText(trans.t("compare_stopping"))
        else:
            self._preview_timer.stop()
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.reject()
            event.ignore()
        else:
            self._preview_timer.stop()
            super().closeEvent(event)


class _AutoBackgroundWorker(QThread):
    progressed = pyqtSignal(float)

    def __init__(self, doc, preset, clean, keep_soft_parts=False):
        super().__init__()
        configure_native_worker_stack(self)
        self.doc = doc
        self.preset = preset
        self.clean = clean
        self.keep_soft_parts = keep_soft_parts
        self.status = ""
        self.error = ""

    def run(self):
        try:
            self.status = self.doc.auto_background(
                self.preset, self.clean, self.isInterruptionRequested,
                lambda fraction, _message: self.progressed.emit(float(fraction)),
                keep_soft_parts=self.keep_soft_parts)
        except MemoryError as exc:
            self.error = str(exc) or trans.t("retouch_auto_bg_failed")
        except Exception as exc:
            if self.isInterruptionRequested():
                self.status = "cancelled"
            else:
                self.error = str(exc) or trans.t("retouch_auto_bg_failed")


class _AutoBackgroundDialog(QDialog):
    """Choose the background preset, then replace the background as one
    history entry (core.background_fill via RetouchDocument)."""

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        from utils.settings_store import get_settings
        self.doc = doc
        self.worker = None
        self.setWindowTitle(trans.t("retouch_auto_bg"))
        self.setModal(True)
        layout = QVBoxLayout(self)
        self.options_scroll = QScrollArea()
        self.options_scroll.setWidgetResizable(True)
        self.options_scroll.setFrameShape(QFrame.Shape.NoFrame)
        options = QWidget()
        option_layout = QVBoxLayout(options)
        option_layout.setContentsMargins(0, 0, 0, 0)
        self.options_scroll.setWidget(options)
        layout.addWidget(self.options_scroll, 1)
        settings = get_settings()
        self.rb_auto = QRadioButton(trans.t("retouch_auto_bg_auto_short"))
        self.rb_soft = QRadioButton(trans.t("retouch_auto_bg_soft"))
        self.rb_sharp = QRadioButton(trans.t("retouch_auto_bg_sharp"))
        group = QButtonGroup(self)
        for button, control in ((self.rb_auto, "auto_background_auto"),
                                (self.rb_soft, "auto_background_soft"),
                                (self.rb_sharp, "auto_background_sharp")):
            button.setProperty("controlId", control)
            group.addButton(button)
            hint = trans.t("retouch_auto_bg_hint")
            if button is self.rb_auto:
                hint = trans.t("retouch_auto_bg_auto") + "\n\n" + hint
            button.setToolTip(hint)
            _add_wrapped_option(option_layout, button)
        mode = str(settings.value("retouch/auto_bg_mode", "auto"))
        {"soft": self.rb_soft, "sharp": self.rb_sharp}.get(mode, self.rb_auto).setChecked(True)
        self.chk_clean = QCheckBox(trans.t("retouch_auto_bg_clean"))
        self.chk_clean.setProperty("controlId", "auto_background_clean")
        self.chk_clean.setToolTip(trans.t("retouch_auto_bg_clean_hint"))
        self.chk_clean.setChecked(str(settings.value("retouch/auto_bg_clean", "true")).lower()
                                  not in ("false", "0"))
        _add_wrapped_option(option_layout, self.chk_clean)
        self.chk_soft_parts = QCheckBox(trans.t("retouch_auto_bg_soft_parts"))
        self.chk_soft_parts.setProperty("controlId", "auto_background_keep_soft")
        self.chk_soft_parts.setToolTip(trans.t("retouch_auto_bg_soft_parts_hint"))
        self.chk_soft_parts.setChecked(str(settings.value("retouch/auto_bg_soft_parts", "false")).lower()
                                       in ("true", "1"))
        _add_wrapped_option(option_layout, self.chk_soft_parts)
        option_layout.addStretch(1)
        # The theme draws a thin bar; text inside it was clipped, so the
        # percentage sits beside it instead.
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.lbl_progress = QLabel("")
        self.lbl_progress.setMinimumWidth(self.lbl_progress.fontMetrics().horizontalAdvance("100%") + 4)
        self.lbl_progress.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.lbl_progress)
        self.progress_box = QWidget()
        self.progress_box.setLayout(progress_row)
        progress_row.setContentsMargins(0, 0, 0, 0)
        self.progress_box.setVisible(False)
        layout.addWidget(self.progress_box)
        buttons = QHBoxLayout()
        self.run_button = QPushButton(trans.t("retouch_auto_bg_run"))
        self.run_button.setProperty("controlId", "auto_background_run")
        self.run_button.setObjectName("primaryButton")
        hint = trans.t("retouch_auto_bg_hint")
        if getattr(doc, "stacked_image", None) is not None:
            hint += "\n" + trans.t("retouch_auto_bg_from_stacked")
        self.run_button.setToolTip(hint)
        self.run_button.clicked.connect(self._run)
        self.cancel_button = QPushButton(trans.t("btn_cancel"))
        self.cancel_button.setProperty("controlId", "auto_background_cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.status = ""
        self.preset = ""
        from utils.ui_utils import size_dialog_to_content
        size_dialog_to_content(self, self.options_scroll, preferred_width=480)
        # Wrapped labels report an inflated unconstrained height; size them at
        # the selected width so the dialog does not fill the entire screen.
        margins = layout.contentsMargins()
        content_width = max(1, self.width() - margins.left() - margins.right()
                            - 2 * self.options_scroll.frameWidth())
        content_height = option_layout.totalHeightForWidth(content_width)
        fixed_height = (margins.top() + margins.bottom() + layout.spacing()
                        + buttons.sizeHint().height())
        # Reserve space for the progress row before it becomes visible.
        progress_height = self.progress_box.sizeHint().height() + layout.spacing()
        self.resize(self.width(), min(self.height(), max(240, content_height + fixed_height + progress_height)))

    def _run(self):
        if self.worker is not None:
            return
        from utils.settings_store import get_settings
        preset = ("sharp" if self.rb_sharp.isChecked()
                  else "soft" if self.rb_soft.isChecked() else "auto")
        clean = self.chk_clean.isChecked()
        settings = get_settings()
        settings.setValue("retouch/auto_bg_mode", preset)
        settings.setValue("retouch/auto_bg_clean", "true" if clean else "false")
        soft_parts = self.chk_soft_parts.isChecked()
        settings.setValue("retouch/auto_bg_soft_parts", "true" if soft_parts else "false")
        for widget in (self.rb_auto, self.rb_soft, self.rb_sharp, self.chk_clean,
                       self.chk_soft_parts, self.run_button):
            widget.setEnabled(False)
        self._show_progress(0.0)
        self.progress_box.setVisible(True)
        self.preset = preset
        self.worker = _AutoBackgroundWorker(self.doc, preset, clean, soft_parts)
        self.worker.progressed.connect(self._show_progress)
        self.worker.finished.connect(self._finished)
        self.worker.start()


    def _show_progress(self, fraction):
        self.progress.setValue(int(fraction * 1000))
        self.lbl_progress.setText(f"{int(round(fraction * 100))}%")

    def _finished(self):
        worker = self.worker
        self.worker = None
        self.status = worker.status
        worker.deleteLater()
        if worker.error:
            QMessageBox.critical(self, trans.t("msg_error"),
                                 trans.t("retouch_auto_bg_failed") + "\n\n" + worker.error)
            for widget in (self.rb_auto, self.rb_soft, self.rb_sharp, self.chk_clean,
                           self.chk_soft_parts, self.run_button):
                widget.setEnabled(True)
            self.progress_box.setVisible(False)
            self.cancel_button.setEnabled(True)
            self.cancel_button.setText(trans.t("btn_cancel"))
        elif worker.status in ("committed", "empty"):
            self.accept()
        elif worker.status in ("unsupported", "oversize", "record_limit"):
            key = {"unsupported": "retouch_auto_bg_unsupported",
                   "oversize": "retouch_auto_bg_oversize",
                   "record_limit": "retouch_checkpoint_record_limit"}[worker.status]
            QMessageBox.warning(self, trans.t("msg_warning"), trans.t(key))
            super().reject()
        else:
            super().reject()

    def reject(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText(trans.t("compare_stopping"))
        else:
            super().reject()

    def closeEvent(self, event):
        if self.worker is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)


class _RetouchCanvas(QWidget):
    """One synchronized image canvas with a brush cursor."""

    def __init__(self, dialog: "RetouchDialog", role: str):
        super().__init__()
        self.dlg = dialog
        self.role = role
        self.setProperty("controlId", f"retouch_{role}")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMinimumSize(180, 180)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._pan_from = None
        self._drawing = False

    # ------------------------------------------------------------ coords
    def image_rect(self, width: int, height: int,
                   offset_x: int = 0, offset_y: int = 0) -> QRectF:
        s = self.dlg.view.scale
        cx = self.dlg.view.center.x()
        cy = self.dlg.view.center.y()
        x = self.width() / 2.0 - cx * s + offset_x * s
        y = self.height() / 2.0 - cy * s + offset_y * s
        return QRectF(x, y, width * s, height * s)

    def widget_rect_for_image_rect(self, rect, offset_x=0, offset_y=0) -> QRect:
        x0, y0, x1, y1 = rect
        s = self.dlg.view.scale
        cx = self.dlg.view.center.x()
        cy = self.dlg.view.center.y()
        bx = self.width() / 2.0 - cx * s + offset_x * s
        by = self.height() / 2.0 - cy * s + offset_y * s
        return QRect(int(bx + x0 * s) - 2, int(by + y0 * s) - 2,
                     max(1, int((x1 - x0) * s)) + 4,
                     max(1, int((y1 - y0) * s)) + 4)

    def widget_to_image(self, pos: QPointF) -> QPointF:
        s = max(1e-9, self.dlg.view.scale)
        cx = self.dlg.view.center.x()
        cy = self.dlg.view.center.y()
        return QPointF((pos.x() - self.width() / 2.0) / s + cx,
                       (pos.y() - self.height() / 2.0) / s + cy)

    def image_to_widget(self, x: float, y: float,
                        offset_x=0, offset_y=0) -> QPointF:
        rect = self.image_rect(1, 1, offset_x, offset_y)
        return QPointF(rect.x() + x * self.dlg.view.scale,
                       rect.y() + y * self.dlg.view.scale)

    def cursor_rect(self):
        pos = self.dlg.cursor_pos
        if (pos is None or self.dlg.preview_active
                or not self.dlg.image_contains(pos.x(), pos.y())
                or (self.role == 'source' and not self.dlg.source_frame_contains(pos.x(), pos.y()))):
            return QRect()
        center = self.image_to_widget(pos.x(), pos.y())
        radius = max(2.0, self.dlg.doc.diameter * self.dlg.view.scale) / 2.0 + 3.0
        return QRectF(center.x() - radius, center.y() - radius,
                      radius * 2, radius * 2).toAlignedRect().intersected(self.rect())

    # ------------------------------------------------------------ paint
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(22, 22, 22))
        qimage, width, height, ox, oy = self.dlg.canvas_image(self.role)
        if qimage is not None and width > 0 and height > 0:
            rect = self.image_rect(width, height, ox, oy)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                                  True)
            painter.drawImage(rect, qimage)
            painter.setPen(QPen(QColor(72, 72, 72), 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
            if self.role == "source":
                domain = QPainterPath()
                domain.addRect(self.image_rect(self.dlg.doc.width, self.dlg.doc.height))
                covered = QPainterPath()
                covered.addRect(rect)
                painter.fillPath(domain.subtracted(covered),
                    QBrush(QColor(160, 160, 160, 150), Qt.BrushStyle.DiagCrossPattern))
        self._paint_cursor(painter)
        painter.end()

    def _paint_cursor(self, painter: QPainter) -> None:
        pos = self.dlg.cursor_pos
        if pos is None or self.dlg.preview_active:
            return
        if not self.dlg.image_contains(pos.x(), pos.y()):
            return
        if self.role == "source" and not self.dlg.source_frame_contains(
                pos.x(), pos.y()):
            return
        frame = self.dlg.doc.active_frame
        ox, oy = (frame.offset_x, frame.offset_y) if self.role == "source" else (0, 0)
        # The cursor is expressed in result coordinates; `image_to_widget`
        # expects frame-local coordinates plus the frame offset.
        center = self.image_to_widget(pos.x() - ox, pos.y() - oy, ox, oy)
        diameter = max(2.0, self.dlg.doc.diameter * self.dlg.view.scale)
        valid = self.dlg.cursor_source_valid()
        color = QColor(255, 88, 88, 240) if not valid else QColor(255, 255, 255, 235)
        pen = QPen(color, 1.4)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, diameter / 2.0, diameter / 2.0)
        inner = diameter * float(np.clip(self.dlg.doc.hardness, 0.0, 1.0))
        if inner > 1.0:
            inner_pen = QPen(color, 1.0, Qt.PenStyle.DashLine)
            inner_pen.setCosmetic(True)
            painter.setPen(inner_pen)
            painter.drawEllipse(center, inner / 2.0, inner / 2.0)

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = event.position()
        pan = (event.button() in (Qt.MouseButton.RightButton,
                                  Qt.MouseButton.MiddleButton)
               or (event.button() == Qt.MouseButton.LeftButton
                   and self.dlg.space_down))
        if pan:
            if self.dlg.doc.stroke_active:
                return
            self._pan_from = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            image_pos = self.widget_to_image(pos)
            self.dlg.set_cursor(image_pos.x(), image_pos.y())
            if self.role == "result":
                if self.dlg.begin_stroke(image_pos.x(), image_pos.y()):
                    self._drawing = True

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        image_pos = self.widget_to_image(pos)
        self.dlg.set_cursor(image_pos.x(), image_pos.y())
        if self._pan_from is not None:
            delta = pos - self._pan_from
            self._pan_from = pos
            self.dlg.pan_by(delta.x(), delta.y())
            return
        if self._drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            self.dlg.stroke_to(image_pos.x(), image_pos.y())

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_from is not None:
            self._pan_from = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            # A release can arrive without a preceding move at the same
            # position; explicitly include it so the last segment is not lost.
            pos = self.widget_to_image(event.position())
            self.dlg.stroke_to(pos.x(), pos.y())
            self._drawing = False
            self.dlg.end_stroke()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        modifiers = event.modifiers()
        if modifiers & (Qt.KeyboardModifier.ControlModifier
                        | Qt.KeyboardModifier.MetaModifier):
            # Some trackpads supply pixel deltas instead of angle deltas.
            delta = delta or event.pixelDelta().y()
            if delta:
                self.dlg._bump_diameter(1.25 if delta > 0 else 0.8)
            event.accept()
            return
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.dlg.zoom_at(self, event.position(), factor)
        event.accept()

    def leaveEvent(self, event) -> None:
        if not self._drawing and self._pan_from is None:
            self.dlg.clear_cursor()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.dlg.canvas_resized()


class RetouchDialog(QDialog):
    """The retouch workspace described by docs/RETOUCH_SPEC.md."""

    def __init__(self, window_main, *, record, apply_callback, parent=None,
                 input_paths=()):
        self.cursor_pos = None
        self.space_down = False
        self.preview_mode = None
        self._preview_owner = None
        self._held_keys = set()
        self._applied = False
        self._applying = False
        self._allow_close = False
        self._closing = False
        self._confirming_close = False
        self._confirm_box = None
        self._confirm_apply_button = None
        self._syncing_frames = False
        self._syncing_tools = False
        self._source_ready = False
        self._source_request = 0
        self._source_published = None
        self._source_loading = False
        self._source_worker = None
        self._source_qimage = None
        self._pure_source_qimage = None
        self._source_display = None
        self._source_clean_display = None
        self._source_valid = None
        self._before_qimage = None
        self._result_display = None
        self._result_qimage = None
        self._sharpness_maps = {}
        self._sharpness_worker = None
        self._sharpness_started = False
        self._sharpness_request = 0
        self._notice = ""
        self._notice_timer = None
        self._eviction_announced = False
        self._was_maximized = False
        self._normal_geometry = None
        self._stroke_lock_widgets = []
        self._checkpoint_dialog = None
        self._edit_dialog = None
        self._ui_ready = False
        super().__init__(parent or window_main)
        self.window_main = window_main
        self.record = record
        self.apply_callback = apply_callback
        self.input_paths = tuple(input_paths)
        # Retouching an already retouched output: Auto background works from
        # the stacked result and keeps the brushing on the subject.
        manager = getattr(window_main, "output_manager", None)
        stacked_source = getattr(manager, "stacked_source", None)
        stacked = stacked_source(record) if callable(stacked_source) else None

        def prepare():
            document = prepare_retouch_document(record.image, record.context)
            if stacked is not None:
                from core.source_frames import resolve_frame
                document.stacked_image = resolve_frame(stacked.image)
            return document

        self.doc = run_pixel_worker(self, trans.t("retouch_title"), prepare)
        self.view = _ViewState()
        self.view.center = QPointF(self.doc.width / 2.0, self.doc.height / 2.0)


        self.setWindowTitle(trans.t("retouch_title"))
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setMinimumSize(900, 600)
        self.resize(1280, 820)
        self._build_ui()
        self._ui_ready = True
        for _attribute, _control in (
                ("btn_undo", "retouch_undo"),
                ("btn_redo", "retouch_redo"),
                ("btn_save_checkpoint", "retouch_save_checkpoint"),
                ("btn_load_checkpoint", "retouch_load_checkpoint"),
                ("btn_before", "retouch_before"),
                ("btn_shortcuts", "retouch_shortcuts"),
                ("chk_sharp", "retouch_sharpness"),
                ("btn_auto_bg", "retouch_auto_background"),
                ("chk_missing", "retouch_missing_data"),
                ("btn_save_missing", "retouch_save_missing"),
                ("btn_restore_all", "retouch_restore"),
                ("btn_apply", "retouch_apply"),
                ("btn_cancel", "retouch_cancel"),
                ("frame_slider", "retouch_frame")):
            _widget = getattr(self, _attribute, None)
            if _widget is not None:
                _widget.setProperty("controlId", _control)
        for _name, _row in (("diameter", self.num_diameter),
                            ("hardness", self.num_hardness),
                            ("opacity", self.num_opacity)):
            _row.slider.setProperty("controlId", f"retouch_{_name}_slider")
            _row.spin.setProperty("controlId", f"retouch_{_name}_spin")
        self._install_filters()
        self._restore_geometry()
        self._populate_frames()
        self._refresh_before_display()
        self._refresh_result_display()
        self._select_frame(self.doc.frame_index, initial=True)
        self._update_titles()
        self._update_brush_labels()
        self._update_controls()
        if self.doc.stacked_image is not None:
            self._set_notice(trans.t("retouch_from_retouched"))
        self._focus_signal = QApplication.instance().focusChanged
        self._focus_signal.connect(self._on_focus_changed)
        self.finished.connect(lambda _result: self._save_geometry())
        self.finished.connect(self._shutdown_sharpness)
        self.finished.connect(self.deleteLater)
        self.destroyed.connect(
            lambda _object=None, ref=weakref.ref(self): _release_dialog_buffers(ref))

    # ------------------------------------------------------------- build
    def _build_ui(self) -> None:
        from ui.right_panel import ElidingButton

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 4)
        outer.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(8)
        self.lbl_title = QLabel()
        font = self.lbl_title.font()
        font.setBold(True)
        self.lbl_title.setFont(font)
        self.lbl_title.setSizePolicy(QSizePolicy.Policy.Ignored,
                                     QSizePolicy.Policy.Preferred)
        # Compact provenance line: the source domain must stay visible even
        # in focus mode, where the source canvas is hidden.
        self.lbl_source_summary = QLabel("")
        self.lbl_source_summary.setObjectName("mutedLabel")
        self.lbl_source_summary.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_unsaved = QLabel()
        self.lbl_unsaved.setObjectName("mutedLabel")
        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(0)
        top.addLayout(title_column, 1)
        top.addWidget(self.lbl_unsaved)

        toolbar = FlowPanel(h_spacing=4, v_spacing=4)
        self.btn_undo = QPushButton(trans.t("retouch_undo"))
        self.btn_undo.clicked.connect(self._undo)
        self.btn_redo = QPushButton(trans.t("retouch_redo"))
        self.btn_redo.clicked.connect(self._redo)
        self.btn_save_checkpoint = QPushButton(trans.t("retouch_checkpoint_save"))
        self.btn_save_checkpoint.clicked.connect(self._save_checkpoint)
        self.btn_load_checkpoint = QPushButton(trans.t("retouch_checkpoint_load"))
        self.btn_load_checkpoint.clicked.connect(self._load_checkpoint)
        self.btn_before = QPushButton(trans.t("retouch_before_short"))
        self.btn_before.setToolTip(trans.t("retouch_view_before"))
        self.btn_before.pressed.connect(lambda: self._press_preview("before"))
        self.btn_before.released.connect(lambda: self._release_preview("before"))
        self.btn_shortcuts = ElidingButton(trans.t("retouch_open_shortcuts"))
        self.btn_shortcuts.setAutoDefault(False)
        self.btn_shortcuts.clicked.connect(self._show_shortcuts)
        no_key = trans.t("retouch_no_shortcut")
        self.btn_undo.setToolTip(trans.t("retouch_tip_undo") + " (Ctrl+Z)")
        self.btn_redo.setToolTip(trans.t("retouch_tip_redo") + " (Ctrl+Y / Ctrl+Shift+Z)")
        self.btn_save_checkpoint.setToolTip(trans.t("retouch_tip_save") + "\n" + no_key)
        self.btn_load_checkpoint.setToolTip(trans.t("retouch_tip_load") + "\n" + no_key)
        self.btn_before.setToolTip(trans.t("retouch_shortcut_before") + " (Tab)\n"
                                   + trans.t("retouch_shortcut_scope"))
        self.btn_shortcuts.setToolTip(trans.t("retouch_tip_shortcuts") + "\n" + no_key)
        for widget in (self.btn_undo, self.btn_redo, self.btn_save_checkpoint,
                       self.btn_load_checkpoint, self.btn_before, self.btn_shortcuts):
            toolbar.add(widget)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(3)
        self.lbl_source_header = QLabel(trans.t("retouch_stack"))
        self.lbl_source_header.setObjectName("mutedLabel")
        self.lbl_source_header.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.source_canvas = _RetouchCanvas(self, "source")
        left_layout.addWidget(self.lbl_source_header)
        left_layout.addWidget(self.source_canvas, 1)
        self.splitter.addWidget(self.left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(3)
        self.lbl_result_header = QLabel(trans.t("retouch_result_title"))
        self.lbl_result_header.setObjectName("mutedLabel")
        self.lbl_result_header.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.lbl_zoom = QLabel(trans.t("retouch_zoom_level").format(percent=100))
        self.lbl_zoom.setObjectName("retouchZoom")
        self.lbl_zoom.setProperty("controlId", "retouch_zoom")
        result_heading = QHBoxLayout()
        result_heading.setContentsMargins(0, 0, 0, 0)
        result_heading.addWidget(self.lbl_result_header, 1)
        result_heading.addWidget(self.lbl_zoom, 0, Qt.AlignmentFlag.AlignRight)
        self.result_canvas = _RetouchCanvas(self, "result")
        right_layout.addLayout(result_heading)
        self.lbl_zoom.ensurePolished()
        self.lbl_source_header.setMinimumHeight(self.lbl_zoom.sizeHint().height())
        right_layout.addWidget(self.result_canvas, 1)
        self.splitter.addWidget(right_panel)

        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.side_scroll.setMinimumWidth(240)
        self.side_scroll.setMaximumWidth(360)
        self.side_panel = QWidget()
        side = QVBoxLayout(self.side_panel)
        # Leave room for the vertical scrollbar: with widgetResizable and
        # the scrollbar hidden until needed, a cramped window used to clip
        # the right edge of every numeric input.
        side.setContentsMargins(8, 0, 18, 0)
        side.setSpacing(6)


        lbl_tools = QLabel(trans.t("retouch_tools"))
        font = lbl_tools.font()
        font.setBold(True)
        lbl_tools.setFont(font)
        side.addWidget(lbl_tools)
        self.rb_source = QRadioButton(trans.t("retouch_tool_source"))
        self.rb_restore = QRadioButton(trans.t("retouch_tool_restore"))
        self.rb_exposure = QRadioButton(trans.t("retouch_tool_exposure"))
        self.tool_group = QButtonGroup(self)
        self.tool_group.addButton(self.rb_source, 0)
        self.tool_group.addButton(self.rb_restore, 1)
        self.tool_group.addButton(self.rb_exposure, 2)
        self.rb_source.toggled.connect(
            lambda on: self._set_tool("source") if on else None)
        self.rb_restore.toggled.connect(
            lambda on: self._set_tool("restore") if on else None)
        self.rb_exposure.toggled.connect(
            lambda on: self._set_tool("exposure") if on else None)
        self.rb_source.setToolTip(trans.t("retouch_help") + " (B)")
        self.rb_restore.setToolTip(trans.t("retouch_help_restore") + " (E)")
        self.rb_exposure.setToolTip(trans.t("retouch_exposure_hint") + "\n" + no_key)
        _add_wrapped_option(side, self.rb_source)
        _add_wrapped_option(side, self.rb_restore)
        _add_wrapped_option(side, self.rb_exposure)
        self.num_exposure = _NumberRow(trans.t("retouch_exposure_amount"),
                                       -200, 200, 50)
        self.num_exposure.setToolTip(trans.t("retouch_exposure_hint") + "\n" + no_key)
        self.num_exposure.valueChanged.connect(self._on_exposure_changed)
        side.addWidget(self.num_exposure)

        self.num_diameter = _NumberRow(trans.t("retouch_diameter"),
                                       1, 2048, self.doc.diameter)
        self.num_hardness = _NumberRow("", 0, 100,
                                       int(round(self.doc.hardness * 100)))
        self.num_opacity = _NumberRow("", 1, 100,
                                      int(round(self.doc.opacity * 100)))
        self.num_diameter.setToolTip(trans.t("retouch_brush_wheel_hint") + " ([ / ])")
        self.num_hardness.setToolTip(trans.t("retouch_tip_hardness") + "\n" + no_key)
        self.num_opacity.setToolTip(trans.t("retouch_tip_opacity") + "\n" + no_key)
        for row in (self.num_exposure, self.num_diameter, self.num_hardness, self.num_opacity):
            for child in (row.slider, row.spin, row._label):
                child.setToolTip(row.toolTip())
        for control in (self.num_diameter, self.num_hardness, self.num_opacity):
            control.valueChanged.connect(self._on_brush_changed)
            side.addWidget(control)

        self.chk_sharp = QCheckBox(trans.t("retouch_show_sharpness"))
        self.chk_sharp.setToolTip(trans.t("retouch_show_sharpness_hint") + "\n"
                                  + trans.t("retouch_sharpness_legend") + "\n" + no_key)
        self.chk_sharp.toggled.connect(self._toggle_sharpness)
        _add_wrapped_option(side, self.chk_sharp)
        self.lbl_sharp_state = QLabel("")
        self.lbl_sharp_state.setObjectName("mutedLabel")
        self.lbl_sharp_state.setWordWrap(True)
        self.lbl_sharp_state.setVisible(False)
        side.addWidget(self.lbl_sharp_state)

        self.btn_auto_bg = ElidingButton(trans.t("retouch_auto_bg") + "…")
        self.btn_auto_bg.setToolTip(trans.t("retouch_auto_bg_hint") + "\n" + no_key)
        self.btn_auto_bg.clicked.connect(self._auto_background)
        side.addWidget(self.btn_auto_bg)
        self.chk_missing = QCheckBox(trans.t("retouch_missing_show"))
        self.chk_missing.setToolTip(trans.t("retouch_missing_legend") + "\n" + no_key)
        self.chk_missing.toggled.connect(self._toggle_missing)
        self.missing_option_row = _add_wrapped_option(side, self.chk_missing)
        self.btn_save_missing = ElidingButton(trans.t("retouch_missing_save"))
        self.btn_save_missing.setToolTip(trans.t("retouch_missing_save") + "\n"
                                         + trans.t("retouch_missing_legend") + "\n" + no_key)
        self.btn_save_missing.clicked.connect(self._save_missing)
        self.btn_save_missing.setVisible(False)
        side.addWidget(self.btn_save_missing)
        self.btn_tones = ElidingButton(trans.t("retouch_tones_title") + "…")
        self.btn_tones.setToolTip(trans.t("retouch_tones_hint") + "\n" + no_key)
        self.btn_tones.clicked.connect(self._adjust_tones)
        side.addWidget(self.btn_tones)
        self.btn_restore_all = ElidingButton(trans.t("retouch_restore_all"))
        self.btn_restore_all.setToolTip(trans.t("retouch_restore_all_confirm") + "\n" + no_key)
        self.btn_restore_all.clicked.connect(self._restore_all)
        side.addWidget(self.btn_restore_all)
        side.addStretch(1)

        self.btn_apply = ElidingButton(trans.t("retouch_apply"))
        self.btn_apply.setObjectName("primaryButton")
        self.btn_apply.setToolTip(trans.t("retouch_apply_hint") + " (Ctrl+Enter)")
        # Only the explicit shortcut may apply: an implicit default button
        # would let a plain Enter publish while the visible button is disabled.
        self.btn_apply.setAutoDefault(False)
        self.btn_apply.setDefault(False)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_cancel = QPushButton(trans.t("btn_cancel"))
        self.btn_cancel.setToolTip(trans.t("retouch_tip_cancel") + " (Esc)")
        self.btn_cancel.setAutoDefault(False)
        self.btn_cancel.clicked.connect(self.reject)

        self.side_scroll.setWidget(self.side_panel)
        self.side_container = QWidget()
        side_outer = QVBoxLayout(self.side_container)
        side_outer.setContentsMargins(0, 0, 0, 0)
        side_outer.setSpacing(4)
        side_outer.addWidget(self.side_scroll, 1)
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(8, 0, 8, 0)
        footer_layout.setSpacing(4)
        footer_layout.addWidget(self.btn_apply)
        footer_layout.addWidget(self.btn_cancel)
        side_outer.addWidget(footer)
        self.side_container.setMinimumWidth(240)
        self.side_container.setMaximumWidth(360)
        self.splitter.addWidget(self.side_container)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([460, 460, 272])

        strip = QWidget()
        strip_layout = QVBoxLayout(strip)
        strip_layout.setContentsMargins(4, 2, 4, 2)
        strip_layout.setSpacing(2)
        strip_row = QHBoxLayout()
        strip_row.setSpacing(4)
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(32)
        self.btn_prev.clicked.connect(self._prev_frame)
        self.frame_list = QListWidget()
        self.frame_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.frame_list.setFlow(QListWidget.Flow.LeftToRight)
        self.frame_list.setWrapping(False)
        self.frame_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.frame_list.setMovement(QListWidget.Movement.Static)
        self.frame_list.setIconSize(QSize(64, 64))
        self.frame_list.setGridSize(QSize(80, 84))
        self.frame_list.setFixedHeight(84)
        self.frame_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection)
        self.frame_list.currentRowChanged.connect(self._on_frame_selected)
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(32)
        self.btn_next.clicked.connect(self._next_frame)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.setMinimumWidth(self.frame_spin.sizeHint().width())
        self.frame_spin.valueChanged.connect(self._on_frame_spin)
        strip_row.addWidget(self.btn_prev)
        strip_row.addWidget(self.frame_list, 1)
        strip_row.addWidget(self.btn_next)
        strip_row.addWidget(self.frame_spin)
        strip_layout.addLayout(strip_row)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(1)
        self.frame_slider.setToolTip(trans.t("retouch_stack_hint"))
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        strip_layout.addWidget(self.frame_slider)
        strip.setFixedHeight(112)

        status = QHBoxLayout()
        status.setContentsMargins(6, 0, 6, 0)
        self.lbl_coords = QLabel("")
        self.lbl_frame_status = QLabel("")
        self.lbl_state = QLabel("")
        self.lbl_state.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Preferred)
        status.addWidget(self.lbl_coords)
        status.addWidget(self.lbl_frame_status, 1)
        status.addWidget(self.lbl_state)

        outer.addLayout(top)
        outer.addWidget(toolbar)
        outer.addWidget(self.splitter, 1)
        outer.addWidget(strip)
        outer.addLayout(status)

        self._stroke_lock_widgets = [
            self.btn_undo, self.btn_redo, self.btn_save_checkpoint,
            self.btn_load_checkpoint, self.btn_before, self.btn_shortcuts,
            self.frame_list, self.frame_spin, self.frame_slider,
            self.btn_prev, self.btn_next,
            self.rb_source, self.rb_restore, self.rb_exposure, self.num_exposure, self.num_diameter,
            self.num_hardness, self.num_opacity, self.chk_sharp,
            self.btn_restore_all, self.btn_tones, self.btn_auto_bg, self.chk_missing,
            self.btn_save_missing, self.btn_apply,
        ]

    # ---------------------------------------------------------- lifecycle
    def _restore_geometry(self) -> None:
        from utils.settings_store import get_settings
        settings = get_settings()
        geometry = settings.value("retouch/geometry")
        restored = False
        if geometry is not None:
            restored = self.restoreGeometry(geometry)
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            style = self.style()
            frame = style.pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth, None, self)
            title = style.pixelMetric(QStyle.PixelMetric.PM_TitleBarHeight, None, self)
            max_width = max(1, available.width() - 2 * frame)
            max_height = max(1, available.height() - title - 2 * frame)
            self.setMinimumSize(min(900, max_width), min(600, max_height))
            if not restored:
                self.resize(min(1280, max(900, available.width() - 80)),
                            min(820, max(600, available.height() - 100)))
            self.resize(min(self.width(), max_width), min(self.height(), max_height))
        state = settings.value("retouch/splitter")
        if state is not None:
            self.splitter.restoreState(state)
            sizes = self.splitter.sizes()
            if len(sizes) >= 2 and min(sizes[:2]) == 0:
                self.splitter.setSizes([1, 1, max(240, sizes[2])])

    def _save_geometry(self) -> None:
        try:
            from utils.settings_store import get_settings
            settings = get_settings()
            if not self.isFullScreen():
                settings.setValue("retouch/geometry", self.saveGeometry())
            settings.setValue("retouch/splitter", self.splitter.saveState())
        except Exception:
            pass

    def _install_filters(self) -> None:
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)

    def _populate_frames(self) -> None:
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for index, frame in enumerate(self.doc.frames):
            image = self.doc.source_previews[index]
            item = QListWidgetItem(str(index + 1))
            if image is not None:
                metadata = None if self.doc.source_previews_are_display[index] else ...
                thumb = thumbnail_bgr(self._display_image(image, metadata), 64)
                item.setIcon(QIcon(QPixmap.fromImage(_qimage_from_bgr(thumb))))
            item.setToolTip(frame.name)
            item.setSizeHint(QSize(76, 82))
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter
                                  | Qt.AlignmentFlag.AlignBottom)
            self.frame_list.addItem(item)
        self.frame_list.blockSignals(False)
        self.frame_spin.setRange(1, max(1, self.doc.frame_count))
        self.frame_spin.setSuffix(f" / {max(1, self.doc.frame_count)}")
        self.frame_spin.setMinimumWidth(max(
            self.frame_spin.sizeHint().width(),
            self.frame_spin.fontMetrics().horizontalAdvance(
                f"{max(1, self.doc.frame_count)} / {max(1, self.doc.frame_count)}") + 54))
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, self.doc.frame_count - 1))
        self.frame_slider.setValue(self.doc.frame_index)
        self.frame_slider.blockSignals(False)

    def _update_titles(self) -> None:
        depth = "16-bit" if self.doc.base.dtype == np.uint16 else (
            "8-bit" if self.doc.base.dtype == np.uint8 else str(self.doc.base.dtype))
        text = (f"{self.record.label} · {self.doc.width}×{self.doc.height}"
                f" · {depth}")
        self._full_title = text
        self._elide_title()
        self.lbl_title.setToolTip(text)
        self.lbl_result_header.setText(trans.t("retouch_result_title"))
        self.lbl_result_header.setToolTip(self.record.label)
        self._update_source_summary()
        self._update_dirty_ui()

    def _update_source_summary(self) -> None:
        """Compact provenance: method, registration, frame count and domain.

        The information was already recorded in RenderContext, but the header
        only showed the output name, size and depth, so a ROI paste and a full
        registration looked identical.
        """
        label = getattr(self, "lbl_source_summary", None)
        if label is None:
            return
        context = getattr(self.record, "context", None)
        parts = []
        algorithm = getattr(context, "algorithm", "") if context else ""
        if algorithm:
            parts.append(str(algorithm))
        alignment = getattr(context, "alignment", "") if context else ""
        parts.append(trans.t("retouch_source_aligned") if alignment
                     and alignment != "none"
                     else trans.t("retouch_source_unaligned"))
        count = int(getattr(context, "source_count", 0) or 0) if context else 0
        if count:
            parts.append(trans.t("retouch_source_frames").format(n=count))
        roi = getattr(context, "roi_rect", None) if context else None
        parts.append(trans.t("retouch_source_roi") if roi
                     else trans.t("retouch_source_full"))
        label.setText(" · ".join(parts))
        details = [self.record.label]
        if context is not None:
            details.extend(frame.name for frame in context.frames)
            if context.roi_rect:
                details.append(f"ROI: {context.roi_rect}")
        label.setToolTip("\n".join(details))
        if self.doc.frame_count:
            frame = self.doc.active_frame
            self.lbl_source_header.setToolTip(
                f"{os.path.basename(frame.name)}")

    def _elide_title(self) -> None:
        if not hasattr(self, "_full_title"):
            return
        width = max(120, self.lbl_title.width())
        self.lbl_title.setText(self.lbl_title.fontMetrics().elidedText(
            self._full_title, Qt.TextElideMode.ElideMiddle, width))

    # -------------------------------------------------------------- frame
    def _select_frame(self, index: int, initial: bool = False) -> None:
        if not initial and self.doc.stroke_active:
            return
        if not self.doc.set_frame(index):
            return
        index = self.doc.frame_index
        self._syncing_frames = True
        self.frame_list.setCurrentRow(index)
        self.frame_spin.setValue(index + 1)
        self.frame_slider.setValue(index)
        self._syncing_frames = False
        self.btn_prev.setEnabled(index > 0)
        self.btn_next.setEnabled(index < self.doc.frame_count - 1)
        frame = self.doc.frames[index]
        self.lbl_source_header.setText(trans.t("retouch_frame").format(
            i=index + 1, n=self.doc.frame_count,
            name=os.path.basename(frame.name)) + "…")
        self._source_ready = False
        self._source_published = None
        self._source_loading = True
        self._source_qimage = None
        self._pure_source_qimage = None
        self._source_display = None
        self._source_clean_display = None
        self._source_valid = None
        self.source_canvas.update()
        self.result_canvas.update()
        if self._source_worker is not None:
            self._source_worker.requestInterruption()
        self._source_request += 1
        request = self._source_request
        self._update_controls()
        self._update_status()
        QTimer.singleShot(
            0, lambda: self._finish_source_request(request, index))

    def _finish_source_request(self, request: int, index: int) -> None:
        if self._closing or request != self._source_request or self._source_ready:
            return
        if index != self.doc.frame_index or self.doc.stroke_active:
            return
        self._refresh_source_display()

    def _on_frame_selected(self, row: int) -> None:
        if self._syncing_frames or row < 0:
            return
        self._select_frame(row)

    def _on_frame_spin(self, value: int) -> None:
        if self._syncing_frames:
            return
        self._select_frame(int(value) - 1)

    def _on_frame_slider(self, value: int) -> None:
        if self._syncing_frames:
            return
        self._select_frame(int(value))

    def _prev_frame(self) -> None:
        self._select_frame(self.doc.frame_index - 1)

    def _next_frame(self) -> None:
        self._select_frame(self.doc.frame_index + 1)

    # ------------------------------------------------------------- display
    def _display_image(self, image, metadata=...):
        if metadata is ...:
            metadata = getattr(self.record.context, 'linear_dng_metadata', None)
        if metadata is not None:
            from utils.linear_dng import develop_linear
            image = develop_linear(image, metadata)
        return to_display_uint8(image)

    def _refresh_source_display(self) -> bool:
        if self._closing or self._source_worker is not None:
            return False
        index = self.doc.frame_index
        request = self._source_request
        metadata = getattr(self.record.context, 'linear_dng_metadata', None)
        worker = _SourceWorker(self.doc.frames[index], index, request, metadata)
        self._source_worker = worker
        self._source_loading = True
        self._update_controls()
        _SOURCE_WORKERS.add(worker)
        dialog_ref = weakref.ref(self)
        def finished():
            _SOURCE_WORKERS.discard(worker)
            dialog = dialog_ref()
            if dialog is not None:
                dialog._source_finished(worker)
            worker.deleteLater()
        worker.finished.connect(finished)
        worker.start()
        return True

    def _source_finished(self, worker):
        if self._source_worker is not worker:
            return
        self._source_worker = None
        if self._closing:
            return
        self._source_loading = False
        if worker.request != self._source_request or worker.index != self.doc.frame_index:
            self._finish_source_request(self._source_request, self.doc.frame_index)
            return
        if worker.error is not None:
            self._set_notice(trans.t('retouch_source_failed').format(error=str(worker.error)))
            self._update_controls()
            return
        if worker.display is None:
            self._update_controls()
            return
        self.doc.frames[worker.index] = worker.frame
        if worker.frame.aligned:
            self.doc._mask_cache[worker.index] = worker.valid
            while len(self.doc._mask_cache) > 2:
                self.doc._mask_cache.popitem(last=False)
        display, valid = worker.display, worker.valid
        thumb = thumbnail_bgr(display, 64)
        self.frame_list.item(worker.index).setIcon(QIcon(QPixmap.fromImage(_qimage_from_bgr(thumb))))
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        self._source_clean_display = display
        self._source_valid = valid
        self._pure_source_qimage = _qimage_from_bgr(display)
        self._render_source_overlay()
        self._source_ready = True
        self._source_published = (worker.index, worker.request, worker.frame, valid)
        self.source_canvas.update()
        self.result_canvas.update()
        frame = self.doc.frames[worker.index]
        self.lbl_source_header.setText(trans.t("retouch_frame").format(
            i=worker.index + 1, n=self.doc.frame_count,
            name=os.path.basename(frame.name)))
        self._update_controls()
        self._update_status()

    def _render_source_overlay(self):
        display = self._source_clean_display
        if display is None:
            return
        if self.chk_sharp.isChecked() and self._sharpness_maps:
            heat = self._sharpness_maps.get(self.doc.frame_index)
            if heat is not None:
                heat = cv2.resize(heat, (display.shape[1], display.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
                display = cv2.addWeighted(display, 0.45, heat, 0.55, 0)
        if self._source_valid is not None:
            display = display.copy()
            display[~self._source_valid] = (80, 80, 160)
        self._source_display = display
        self._source_qimage = _qimage_from_bgr(display)
        self.source_canvas.update()
        if self.preview_mode == "source":
            self.result_canvas.update()

    def _refresh_before_display(self) -> None:
        display = self._display_image(self.doc.base)
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        self._before_qimage = _qimage_from_bgr(display)

    def _refresh_result_display(self) -> None:
        self._result_display = self._tint_missing(self._display_image(self.doc.working))
        self._result_qimage = _qimage_from_bgr(self._result_display)
        self.result_canvas.update()
        self._update_status()

    def _patch_result(self, rect) -> None:
        if self._result_qimage is None or self._result_display is None:
            self._refresh_result_display()
            return
        x0, y0, x1, y1 = rect
        patch = self.doc.working[y0:y1, x0:x1]
        display = self._display_image(patch)
        if self._result_display.ndim == 3 and display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        display = self._tint_missing(display, (y0, y1, x0, x1))
        self._result_display[y0:y1, x0:x1] = display
        painter = QPainter(self._result_qimage)
        painter.drawImage(x0, y0, _qimage_from_bgr(display))
        painter.end()
        self.result_canvas.update(
            self.result_canvas.widget_rect_for_image_rect(rect))

    def canvas_image(self, role: str):
        """(qimage, width, height, offset_x, offset_y) for one canvas."""
        if role == "source":
            if self._source_qimage is None:
                return None, 0, 0, 0, 0
            frame = self.doc.active_frame
            return (self._source_qimage, frame.width, frame.height,
                    frame.offset_x, frame.offset_y)
        if self.preview_mode == "before":
            return (self._before_qimage, self.doc.width, self.doc.height, 0, 0)
        if self.preview_mode == "source":
            if self._source_qimage is None:
                return None, 0, 0, 0, 0
            frame = self.doc.active_frame
            return (self._pure_source_qimage, frame.width, frame.height,
                    frame.offset_x, frame.offset_y)
        return (self._result_qimage, self.doc.width, self.doc.height, 0, 0)

    # --------------------------------------------------------------- zoom
    def _fit(self) -> None:
        if self.doc.stroke_active:
            return
        self.view.center = QPointF(self.doc.width / 2.0, self.doc.height / 2.0)
        self.view.fit = True
        scales = []
        for canvas in (self.source_canvas, self.result_canvas):
            if not canvas.isVisible() or canvas.width() < 2 or canvas.height() < 2:
                continue
            scales.append(min((canvas.width() - 6) / float(self.doc.width),
                              (canvas.height() - 6) / float(self.doc.height)))
        if scales:
            self.view.scale = max(1e-6, min(scales))
        self._update_views()

    def _zoom_100(self) -> None:
        if self.doc.stroke_active:
            return
        dpr = max(1.0, self.result_canvas.devicePixelRatioF())
        self.view.fit = False
        self.view.scale = 1.0 / dpr
        self._update_views()

    def _zoom_step(self, direction: int) -> None:
        """Keyboard zoom around the canvas centre (same clamp as the wheel)."""
        if self.doc.stroke_active:
            return
        canvas = self.result_canvas
        centre = QPointF(canvas.width() / 2.0, canvas.height() / 2.0)
        self.zoom_at(canvas, centre, 1.25 if direction > 0 else 0.8)

    def zoom_at(self, canvas: _RetouchCanvas, pos: QPointF,
                factor: float) -> None:
        if self.doc.stroke_active:
            return
        old = self.view.scale
        dpr = max(1.0, canvas.devicePixelRatioF())
        new = min(max(old * factor, 0.05 / dpr), 16.0 / dpr)
        if abs(new - old) < 1e-9:
            return
        before = canvas.widget_to_image(pos)
        self.view.scale = new
        self.view.fit = False
        after = canvas.widget_to_image(pos)
        self.view.center += before - after
        self._update_views()

    def pan_by(self, dx: float, dy: float) -> None:
        if self.doc.stroke_active or self.view.scale <= 0:
            return
        self.view.center -= QPointF(dx / self.view.scale, dy / self.view.scale)
        self.view.fit = False
        self._update_views()

    def _update_views(self) -> None:
        self.source_canvas.update()
        self.result_canvas.update()
        self._update_status()

    def canvas_resized(self) -> None:
        if not self._ui_ready:
            return
        if self.view.fit:
            self._fit()
        else:
            self._update_status()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._ui_ready:
            return
        self._elide_title()
        QTimer.singleShot(0, self._elide_title)
        if self.view.fit:
            self._fit()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._ui_ready:
            return
        QTimer.singleShot(0, self._fit)

    # -------------------------------------------------------------- cursor
    def set_cursor(self, x: float, y: float) -> None:
        old = [canvas.cursor_rect() for canvas in (self.source_canvas, self.result_canvas)]
        self.cursor_pos = QPointF(x, y)
        for canvas, rect in zip((self.source_canvas, self.result_canvas), old):
            dirty = rect.united(canvas.cursor_rect())
            if not dirty.isEmpty():
                canvas.update(dirty)
        self._update_status()

    def clear_cursor(self) -> None:
        if self.cursor_pos is None:
            return
        old = [canvas.cursor_rect() for canvas in (self.source_canvas, self.result_canvas)]
        self.cursor_pos = None
        for canvas, rect in zip((self.source_canvas, self.result_canvas), old):
            if not rect.isEmpty():
                canvas.update(rect)
        self._update_status()

    def image_contains(self, x: float, y: float) -> bool:
        return 0 <= x < self.doc.width and 0 <= y < self.doc.height

    def source_frame_contains(self, x: float, y: float) -> bool:
        return self.doc.active_frame.contains(x, y)

    def cursor_source_valid(self) -> bool:
        if self.cursor_pos is None:
            return True
        x, y = self.cursor_pos.x(), self.cursor_pos.y()
        if not self.image_contains(x, y):
            return False
        if self.doc.tool != 'source':
            return True
        published = self._source_published
        if (not self._source_ready or self._source_loading or published is None
                or published[0] != self.doc.frame_index
                or published[1] != self._source_request
                or published[2] is not self.doc.active_frame):
            return False
        frame, mask = published[2], published[3]
        if not frame.contains(x, y):
            return False
        if mask is None:
            return True
        lx, ly = frame.local_point(x, y)
        return bool(mask[int(ly), int(lx)])

    def _update_status(self) -> None:
        dpr = max(1.0, self.result_canvas.devicePixelRatioF())
        percent = int(round(self.view.scale * dpr * 100))
        self.lbl_zoom.setText(trans.t("retouch_zoom_level").format(percent=percent))
        if self.cursor_pos is None:
            self.lbl_coords.setText("")
        else:
            self.lbl_coords.setText(
                f"x: {int(self.cursor_pos.x())}  y: {int(self.cursor_pos.y())}")
        frame = self.doc.active_frame
        self.lbl_frame_status.setText(trans.t("retouch_frame").format(
            i=self.doc.frame_index + 1, n=self.doc.frame_count,
            name=os.path.basename(frame.name)))
        tool = trans.t({"source": "retouch_tool_source", "restore": "retouch_tool_restore",
                        "exposure": "retouch_tool_exposure"}[self.doc.tool])
        if self.doc.tool == 'source' and self._source_loading:
            tool = self.lbl_source_header.text()
        elif self.cursor_pos is not None and self.doc.tool == "source" and not self.cursor_source_valid():
            tool = (trans.t("retouch_source_roi") + " · "
                    if not frame.contains(self.cursor_pos.x(), self.cursor_pos.y()) else "")
            tool += trans.t("retouch_source_unavailable")
        self.lbl_state.setText(self._notice or tool)

    def _set_notice(self, text: str) -> None:
        self._notice = text
        self._update_status()
        if self._notice_timer is not None:
            self._notice_timer.stop()
        self._notice_timer = QTimer(self)
        self._notice_timer.setSingleShot(True)
        self._notice_timer.timeout.connect(self._clear_notice)
        self._notice_timer.start(7000)

    def _clear_notice(self) -> None:
        self._notice = ""
        self._update_status()

    # -------------------------------------------------------------- brush
    def _set_tool(self, tool: str) -> None:
        if self.doc.stroke_active or not self.doc.set_tool(tool):
            self._sync_tool_buttons()
            return
        self._sync_tool_buttons()
        self._update_status()
        self.source_canvas.update()
        self.result_canvas.update()
        # The exposure amount only applies to the exposure brush; hide it
        # for the source/restore tools instead of leaving a live control
        # whose value the stroke never reads.
        self._update_controls()

    def _show_shortcuts(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(trans.t('retouch_shortcuts_help'))
        dialog.setProperty('controlId', 'retouch_shortcuts_dialog')
        root = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        content = QVBoxLayout(page)
        note = QLabel(trans.t('retouch_shortcut_scope'))
        note.setWordWrap(True)
        content.addWidget(note)
        groups = (
            ('retouch_shortcut_navigation', (
                ('F', 'retouch_fit'), ('1', 'retouch_zoom_100'),
                ('+ / −', 'retouch_shortcut_zoom'),
                (', / .', 'retouch_shortcut_frames'),
                ('F11', 'compare_fullscreen'))),
            ('retouch_tools', (
                ('B', 'retouch_tool_source'), ('E', 'retouch_tool_restore'),
                ('[ / ]', 'retouch_diameter'),
                (trans.t('retouch_shortcut_ctrl_wheel'), 'retouch_diameter'),
                (trans.t('retouch_shortcut_left_drag'), 'retouch_shortcut_paint'))),
            ('retouch_shortcut_preview', (
                ('Tab', 'retouch_shortcut_before'), ('Alt', 'retouch_shortcut_source'),
                (trans.t('retouch_shortcut_wheel'), 'retouch_shortcut_zoom'),
                (trans.t('retouch_shortcut_pan_keys'), 'retouch_shortcut_pan'))),
            ('retouch_shortcut_actions', (
                ('Ctrl+Z', 'retouch_undo'), ('Ctrl+Y / Ctrl+Shift+Z', 'retouch_redo'),
                ('Ctrl+Enter', 'retouch_apply'), ('Esc', 'retouch_shortcut_escape'))),
        )
        for title, entries in groups:
            heading = QLabel(trans.t(title))
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            content.addWidget(heading)
            grid = QGridLayout()
            grid.setHorizontalSpacing(20)
            grid.setVerticalSpacing(8)
            grid.setColumnStretch(1, 1)
            for row, (keys, key) in enumerate(entries):
                shortcut, description = QLabel(keys), QLabel(trans.t(key))
                shortcut.setTextFormat(Qt.TextFormat.PlainText)
                description.setWordWrap(True)
                grid.addWidget(shortcut, row, 0, Qt.AlignmentFlag.AlignTop)
                grid.addWidget(description, row, 1)
            content.addLayout(grid)
        content.addStretch()
        scroll.setWidget(page)
        root.addWidget(scroll, 1)
        close = QPushButton(trans.t('dialog_close'))
        close.clicked.connect(dialog.accept)
        root.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)
        dialog.resize(680, 650)
        dialog.exec()
        dialog.deleteLater()


    def _sync_tool_buttons(self) -> None:
        self._syncing_tools = True
        self.rb_source.setChecked(self.doc.tool == "source")
        self.rb_restore.setChecked(self.doc.tool == "restore")
        self.rb_exposure.setChecked(self.doc.tool == "exposure")
        self._syncing_tools = False

    def _on_exposure_changed(self, value: int) -> None:
        if not self.doc.stroke_active:
            self.doc.exposure = value / 100.0

    def _on_brush_changed(self, _value=None) -> None:
        if self.doc.stroke_active:
            return
        self.doc.set_brush(self.num_diameter.value(),
                           self.num_hardness.value() / 100.0,
                           self.num_opacity.value() / 100.0)
        self._update_brush_labels()
        self.source_canvas.update()
        self.result_canvas.update()

    def _update_brush_labels(self) -> None:
        self.num_hardness.set_label(
            trans.t("retouch_hardness").format(self.num_hardness.value()))
        self.num_opacity.set_label(
            trans.t("retouch_opacity").format(self.num_opacity.value()))

    def _bump_diameter(self, factor: float) -> None:
        """Resize the brush from a shortcut.

        `_NumberRow.setValue` intentionally does not emit valueChanged (it is
        also used to seed the controls), so calling it alone left the document
        at the old diameter: the number changed, the next stroke did not.
        """
        if self.doc.stroke_active:
            return
        value = self.num_diameter.value()
        new = int(round(value * factor))
        if factor > 1.0 and new <= value:
            new = value + 1
        if factor < 1.0 and new >= value:
            new = value - 1
        new = max(1, min(2048, new))
        if new == value:
            return
        self.num_diameter.setValue(new)
        self._on_brush_changed()

    # ------------------------------------------------------------- strokes
    def begin_stroke(self, x: float, y: float) -> bool:
        if (self._applying or self._confirming_close or not self._source_ready
                or self._source_loading or self.preview_mode is not None
                or self.doc.stroke_active or self._edit_dialog is not None
                or self._checkpoint_dialog is not None):
            return False
        if not self.doc.source_valid_at(x, y):
            self._set_notice(trans.t("retouch_source_unavailable"))
            return False
        try:
            started = self.doc.begin_stroke(x, y)
        except (MemoryError, OSError) as exc:
            self._stroke_failed(exc)
            return False
        if not started:
            if self.doc.last_status == "record_limit":
                self._show_record_limit()
            return False
        self._set_stroke_ui(True)
        if self.doc.stroke_start_rect is not None:
            self._patch_result(self.doc.stroke_start_rect)
        self._update_dirty_ui()
        return True

    def stroke_to(self, x: float, y: float) -> None:
        if not self.doc.stroke_active:
            return
        failed_before = self.doc._stroke_failure is not None
        try:
            dirty = self.doc.stroke_to(x, y)
        except (MemoryError, OSError) as exc:
            self._stroke_failed(exc)
            return
        if not failed_before and self.doc._stroke_failure is not None:
            self._refresh_result_display()
        elif dirty is not None:
            self._patch_result(dirty)

    def end_stroke(self) -> None:
        if not self.doc.stroke_active:
            return
        try:
            status = self.doc.end_stroke()
        except (MemoryError, OSError) as exc:
            self._stroke_failed(exc)
            return
        self._set_stroke_ui(False)
        if status == "oversize":
            self._refresh_result_display()
            self._set_notice(trans.t("retouch_stroke_too_large"))
            QMessageBox.warning(self, trans.t("msg_warning"),
                                trans.t("retouch_stroke_too_large"))
        elif status == "record_limit":
            self._refresh_result_display()
            self._show_record_limit()
        elif status == "committed" and self.doc.history.evicted:
            if not self._eviction_announced:
                self._eviction_announced = True
                self._set_notice(trans.t("retouch_history_evicted"))
        self._update_dirty_ui()
        self._update_controls()

    def _stroke_failed(self, error):
        try:
            if self.doc.stroke_active:
                self.doc.cancel_stroke()
        finally:
            self.source_canvas._drawing = False
            self.result_canvas._drawing = False
            self._set_stroke_ui(False)
            self._refresh_result_display()
            self._update_dirty_ui()
            self._update_controls()
        QMessageBox.critical(self, trans.t('msg_error'), str(error))

    def _cancel_stroke(self) -> None:
        if not self.doc.stroke_active:
            return
        self.doc.cancel_stroke()
        self._set_stroke_ui(False)
        self._refresh_result_display()
        self._update_dirty_ui()
        self._update_controls()

    def _set_stroke_ui(self, active: bool) -> None:
        for widget in self._stroke_lock_widgets:
            widget.setEnabled(not active)
        if not active:
            self._update_controls()

    # ------------------------------------------------------------- history
    def _undo(self) -> None:
        if (self.preview_mode is not None or self._edit_dialog is not None
                or self._checkpoint_dialog is not None):
            return
        was_stroke = self.doc.stroke_active
        if self.doc.undo():
            self._refresh_result_display()
        if was_stroke:
            self._set_stroke_ui(False)
        self._update_dirty_ui()
        self._update_controls()

    def _redo(self) -> None:
        if (self.preview_mode is not None or self._edit_dialog is not None
                or self._checkpoint_dialog is not None):
            return
        was_stroke = self.doc.stroke_active
        if self.doc.redo():
            self._refresh_result_display()
        if was_stroke:
            self._set_stroke_ui(False)
        self._update_dirty_ui()
        self._update_controls()

    def _adjust_tones(self) -> None:
        if (self.doc.stroke_active or self.preview_mode is not None
                or self._edit_dialog is not None or self._checkpoint_dialog is not None):
            return
        dialog = _ToneDialog(self.doc, self)
        self._edit_dialog = dialog
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
        finally:
            self._edit_dialog = None
            dialog.deleteLater()
        if accepted:
            self._refresh_result_display()
            if self.doc.history.evicted and not self._eviction_announced:
                self._eviction_announced = True
                self._set_notice(trans.t("retouch_history_evicted"))
            self._update_dirty_ui()
            self._update_controls()

    def _auto_background(self) -> None:
        if (self.doc.stroke_active or self.preview_mode is not None
                or self._edit_dialog is not None or self._checkpoint_dialog is not None):
            return
        dialog = _AutoBackgroundDialog(self.doc, self)
        self._edit_dialog = dialog
        try:
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
        finally:
            self._edit_dialog = None
            dialog.deleteLater()
        if accepted:
            self._refresh_result_display()
            fill = self.doc.background_fill
            if dialog.status == "empty" or (fill is not None and not fill.changed):
                self._set_notice(trans.t("retouch_auto_bg_unchanged"))
            elif fill is not None and dialog.preset == "auto":
                self._set_notice(trans.t("retouch_auto_bg_used_sharp" if fill.preset_used == "sharp"
                                         else "retouch_auto_bg_used_soft"))
            if self.doc.history.evicted and not self._eviction_announced:
                self._eviction_announced = True
                self._set_notice(trans.t("retouch_history_evicted"))
            self._update_dirty_ui()
            self._update_controls()

    def _tint_missing(self, display, region=None):
        """Tint the missing-data classes into an 8-bit display patch."""
        fill = self.doc.background_fill
        if fill is None or not self.chk_missing.isChecked():
            return display
        y0, y1, x0, x1 = region or (0, display.shape[0], 0, display.shape[1])
        out = display.astype(np.float32)
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        for _name, layer, colour in fill.missing.layers():
            amount = layer[y0:y1, x0:x1].astype(np.float32) * (0.6 / 255.0)
            if not amount.any():
                continue
            out += (np.array(colour, np.float32) - out) * amount[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def _toggle_missing(self, checked: bool) -> None:
        self._refresh_result_display()
        self._update_controls()

    def _save_missing(self) -> None:
        fill = self.doc.background_fill
        if (fill is None or self.doc.stroke_active or self.preview_mode is not None
                or self._edit_dialog is not None or self._checkpoint_dialog is not None):
            return
        from utils.settings_store import image_save_default, remember_image_save
        path, _ = FileDialog.getSaveFileName(
            self, trans.t("retouch_missing_save"), image_save_default("missing-data.png"),
            "PNG (*.png)")
        if not path:
            return
        remember_image_save(path)
        selected_path = path
        if not path.lower().endswith(".png"):
            path += ".png"
        # The tinted preview, plus one exact 8-bit mask per class: the masks
        # are the record of where no frame backs the result.
        stem = path[:-4]
        layers = list(fill.missing.layers())
        targets = [path, *(f"{stem}-{name}.png" for name, _layer, _c in layers)]
        try:
            sources = self._checkpoint_sources()
            for target in targets:
                if any(same_path(target, source) for source in sources):
                    raise ValueError(trans.t("retouch_missing_input"))
            existing_siblings = [target for target in targets[1:] if os.path.exists(target)]
            if selected_path != path and os.path.exists(path):
                existing_siblings.insert(0, path)
            if existing_siblings and QMessageBox.question(
                    self, trans.t("retouch_missing_save"),
                    trans.t("retouch_missing_replace").format(
                        files="\n".join(existing_siblings)),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Yes:
                return
            from dialogs.retouch_export import MissingDataExportDialog

            dialog = MissingDataExportDialog(self.doc.working, fill.missing, targets, self)
            self._edit_dialog = dialog
            self._update_controls()
            try:
                accepted = dialog.exec() == QDialog.DialogCode.Accepted
            finally:
                self._edit_dialog = None
                self._update_controls()
                dialog.deleteLater()
            if dialog.worker.error is not None:
                raise dialog.worker.error
            if not accepted:
                return
            warning = getattr(dialog.worker, 'warning', None)
            if warning:
                QMessageBox.warning(self, trans.t('msg_warning'), str(warning))
        except Exception as exc:
            QMessageBox.critical(self, trans.t("msg_error"), str(exc))
            return
        self._set_notice(trans.t("retouch_missing_saved"))

    def _restore_all(self) -> None:
        if (self.doc.stroke_active or self.preview_mode is not None
                or self._edit_dialog is not None or self._checkpoint_dialog is not None
                or not self.doc.is_dirty):
            return
        box = QMessageBox(self)
        from utils.ui_utils import style_message_box
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(trans.t("retouch_restore_all"))
        box.setText(trans.t("retouch_restore_all_confirm"))
        box.setStandardButtons(QMessageBox.StandardButton.Yes
                               | QMessageBox.StandardButton.Cancel)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return
        if self.doc.restore_all():
            self._refresh_result_display()
        elif self.doc.last_status == "record_limit":
            self._show_record_limit()
        elif self.doc.last_status == "oversize":
            QMessageBox.warning(self, trans.t("msg_warning"),
                                trans.t("retouch_stroke_too_large"))
        self._update_dirty_ui()
        self._update_controls()

    def _show_record_limit(self) -> None:
        QMessageBox.warning(self, trans.t("msg_warning"),
                            trans.t("retouch_checkpoint_record_limit"))


    def _checkpoint_sources(self):
        window = self.window_main
        paths = list(self.input_paths)
        paths.extend(getattr(self.record, "source_paths", ()) or ())
        paths.extend(getattr(getattr(self.record, "context", None), "source_paths", ()) or ())
        paths.extend(getattr(window, "image_source_paths", ()) or ())
        paths.extend(getattr(getattr(window, "image_loader", None), "image_paths", ()) or ())
        paths.extend(getattr(window, "_standalone_retouch_paths", ()) or ())
        paths.extend(frame.name for frame in self.doc.frames if os.path.isabs(frame.name))
        return paths

    def _save_checkpoint(self) -> None:
        if (self.doc.stroke_active or self.preview_mode is not None
                or self._edit_dialog is not None or self._checkpoint_dialog is not None):
            return
        path, _ = FileDialog.getSaveFileName(
            self, trans.t("retouch_checkpoint_save"), "",
            trans.t("retouch_checkpoint_filter"))
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".ofretouch"
        try:
            if any(same_path(path, source) for source in self._checkpoint_sources()):
                raise ValueError(trans.t("retouch_checkpoint_input"))
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, trans.t("msg_error"), str(exc))
            return
        dialog = _CheckpointDialog(self.doc, path, self, operation="save")
        self._checkpoint_dialog = dialog
        try:
            dialog.start()
        finally:
            self._checkpoint_dialog = None
        if dialog.worker.error is not None and not dialog.worker.cancelled:
            QMessageBox.critical(self, trans.t("msg_error"),
                                 trans.t("retouch_checkpoint_save_failed") + "\n\n"
                                 + str(dialog.worker.error))

    def _load_checkpoint(self) -> None:
        if (self.doc.stroke_active or self.preview_mode is not None
                or self._edit_dialog is not None or self._checkpoint_dialog is not None):
            return
        path, _ = FileDialog.getOpenFileName(
            self, trans.t("retouch_checkpoint_load"), "",
            trans.t("retouch_checkpoint_filter"))
        if not path:
            return
        if self.doc.is_dirty and QMessageBox.question(
                self, trans.t("retouch_checkpoint_load"),
                trans.t("retouch_checkpoint_replace"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Yes:
            return
        dialog = _CheckpointDialog(self.doc, path, self)
        self._checkpoint_dialog = dialog
        try:
            dialog.start()
        finally:
            self._checkpoint_dialog = None
        error = dialog.worker.error
        if error is not None:
            if not dialog.worker.cancelled:
                QMessageBox.critical(
                    self, trans.t("msg_error"),
                    trans.t("retouch_checkpoint_load_failed") + "\n\n" + str(error))
            return
        self._refresh_result_display()
        self._update_dirty_ui()
        self._update_controls()
        self._update_status()

    def _update_dirty_ui(self) -> None:
        dirty = self.doc.is_dirty
        self.lbl_unsaved.setText(
            trans.t("retouch_state_unsaved") if dirty else "")
        self.setWindowTitle(trans.t("retouch_title") + (" ●" if dirty else ""))
        self.btn_apply.setEnabled(self._can_apply())
        self.btn_restore_all.setEnabled(dirty and not self.doc.stroke_active
                                        and self.preview_mode is None)
        self.btn_undo.setEnabled(self.doc.can_undo
                                 and not self.doc.stroke_active
                                 and self.preview_mode is None)
        self.btn_redo.setEnabled(self.doc.can_redo
                                 and not self.doc.stroke_active
                                 and self.preview_mode is None)

    def _update_controls(self) -> None:
        blocked = self.doc.stroke_active or self._source_loading
        preview = self.preview_mode is not None
        self._sync_tool_buttons()
        self.btn_undo.setEnabled(self.doc.can_undo and not blocked and not preview)
        self.btn_redo.setEnabled(self.doc.can_redo and not blocked and not preview)
        self.btn_save_checkpoint.setEnabled(not blocked and not preview)
        self.btn_load_checkpoint.setEnabled(not blocked and not preview)
        self.btn_tones.setEnabled(not blocked and not preview)
        self.btn_auto_bg.setEnabled(not blocked and not preview
                                    and self.doc.auto_background_supported())
        has_missing = self.doc.background_fill is not None
        self.chk_missing.setVisible(has_missing)
        self.missing_option_row.setVisible(has_missing)
        self.chk_missing.setEnabled(has_missing and not blocked and not preview)
        self.btn_save_missing.setVisible(has_missing and self.chk_missing.isChecked())
        self.btn_save_missing.setEnabled(has_missing and not blocked and not preview)
        self.btn_restore_all.setEnabled(
            self.doc.is_dirty and not blocked and not preview)
        self.btn_apply.setEnabled(self._can_apply())
        if self._confirm_apply_button is not None:
            self._confirm_apply_button.setEnabled(self._can_apply())
            self._confirm_apply_button.setToolTip(
                trans.t('msg_loading_stack') if self._source_loading else '')
        self.btn_before.setEnabled(not blocked and self.preview_mode != "source")
        self.chk_sharp.setEnabled(not blocked)
        self.rb_source.setEnabled(not blocked)
        self.rb_restore.setEnabled(not blocked)
        self.rb_exposure.setEnabled(not blocked)
        exposure_tool = self.doc.tool == "exposure"
        self.num_exposure.setVisible(exposure_tool)
        self.num_exposure.setEnabled(exposure_tool and not blocked)
        self.num_diameter.setEnabled(not blocked)
        self.num_hardness.setEnabled(not blocked)
        self.num_opacity.setEnabled(not blocked)
        frame_enabled = not self.doc.stroke_active and not preview
        self.frame_list.setEnabled(frame_enabled)
        self.frame_spin.setEnabled(frame_enabled)
        self.frame_slider.setEnabled(frame_enabled)
        self.btn_prev.setEnabled(
            frame_enabled and self.doc.frame_index > 0)
        self.btn_next.setEnabled(
            frame_enabled and self.doc.frame_index < self.doc.frame_count - 1)

    # ------------------------------------------------------------- preview
    def _press_preview(self, mode: str) -> None:
        if self.doc.stroke_active or self.preview_mode is not None:
            return
        self.preview_mode = mode
        self._preview_owner = mode
        self.btn_before.setDown(mode == "before")
        self.result_canvas.update()
        self._update_controls()

    def _release_preview(self, mode: str) -> None:
        if self.preview_mode != mode:
            return
        self.preview_mode = None
        self._preview_owner = None
        self.btn_before.setDown(False)
        self.result_canvas.update()
        self._update_controls()

    def _clear_preview(self) -> None:
        self._held_keys.discard(Qt.Key.Key_Tab)
        self._held_keys.discard(Qt.Key.Key_Alt)
        if self.preview_mode is None:
            return
        self.preview_mode = None
        self._preview_owner = None
        self.btn_before.setDown(False)
        self.result_canvas.update()
        self._update_controls()

    @property
    def preview_active(self) -> bool:
        return self.preview_mode is not None

    # ---------------------------------------------------------- sharpness
    def _toggle_sharpness(self, checked: bool) -> None:
        if checked and not self._sharpness_started:
            self._start_sharpness_worker()
        self._set_sharp_state(
            "computing" if checked and not self._sharpness_maps
            else ("ready" if checked else ""))
        self._render_source_overlay()

    def _set_sharp_state(self, state: str) -> None:
        """Show computing or failure; a finished overlay needs no extra label."""
        label = self.lbl_sharp_state
        if label is None:
            return
        if not state or state == "ready":
            label.setVisible(False)
            label.setText("")
            return
        key = {"computing": "retouch_sharp_computing",
               "failed": "retouch_sharp_failed"}.get(state)
        if key is None:
            label.setVisible(False)
            return
        label.setText(trans.t(key))
        label.setStyleSheet("color:#c0392b;" if state == "failed" else "")
        label.setVisible(True)

    def _start_sharpness_worker(self) -> None:
        self._sharpness_started = True
        self._sharpness_request += 1
        worker = _SharpnessWorker(self.doc.frames, self._sharpness_request)
        self._sharpness_worker = worker
        worker.ready.connect(self._on_sharpness_ready)
        worker.failed.connect(self._on_sharpness_failed)
        _SHARPNESS_WORKERS.add(worker)
        worker.finished.connect(lambda w=worker: _SHARPNESS_WORKERS.discard(w))
        dialog_ref = weakref.ref(self)
        request = self._sharpness_request
        def finished():
            dialog = dialog_ref()
            if dialog is not None and not dialog._closing:
                dialog._on_sharpness_finished(request)
        self._sharpness_finished_callback = finished
        worker.finished.connect(self._sharpness_finished_callback)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_sharpness_failed(self, request: int, error: str) -> None:
        if self._closing or request != self._sharpness_request:
            return
        self._set_sharp_state("failed")
        self.lbl_sharp_state.setToolTip(error)

    def _on_sharpness_finished(self, request: int) -> None:
        if self._closing or request != self._sharpness_request:
            return
        if self._sharpness_worker is not None:
            self._sharpness_worker = None
        if not self._sharpness_maps:
            # A finished worker without maps means the computation failed;
            # the checkbox used to stay ticked with nothing drawn and no
            # explanation, and there was no way to retry.
            self._sharpness_started = False
            self._set_sharp_state("failed")
        else:
            self._set_sharp_state(
                "ready" if self.chk_sharp.isChecked() else "")

    def _on_sharpness_ready(self, request: int, maps) -> None:
        if self._closing or request != self._sharpness_request:
            return
        self._sharpness_maps = maps or {}
        if self.chk_sharp.isChecked() and self._source_display is not None:
            self._set_sharp_state("ready")
            self._render_source_overlay()

    def _on_focus_changed(self, old, new) -> None:
        if self._closing:
            return
        left_canvas = old in (self.source_canvas, self.result_canvas)
        if new is None or (new is not self and not self.isAncestorOf(new)) \
                or left_canvas:
            # A held preview/pan must not survive the move into a numeric
            # field: the modifier release is ignored while typing, so the
            # canvas would stay frozen on the "before" image.
            self._clear_preview()
            self._held_keys.clear()
            self.space_down = False
            self.source_canvas.setCursor(Qt.CursorShape.CrossCursor)
            self.result_canvas.setCursor(Qt.CursorShape.CrossCursor)

    def _shutdown_sharpness(self, *_args) -> None:
        self._closing = True
        self._held_keys.clear()
        self.space_down = False
        if self._source_worker is not None:
            self._source_worker.requestInterruption()
        self._source_ready = False
        self._source_published = None
        try:
            self._focus_signal.disconnect(self._on_focus_changed)
        except (TypeError, RuntimeError):
            pass
        worker = self._sharpness_worker
        self._sharpness_worker = None
        if worker is None:
            return
        try:
            worker.ready.disconnect(self._on_sharpness_ready)
        except (TypeError, RuntimeError):
            pass
        try:
            worker.failed.disconnect(self._on_sharpness_failed)
            worker.finished.disconnect(self._sharpness_finished_callback)
        except (TypeError, RuntimeError):
            pass
        worker.requestInterruption()
        self._sharpness_request += 1

    # ----------------------------------------------------------- layout/fs

    def _toggle_fullscreen(self) -> None:
        if self.doc.stroke_active:
            return
        if self.isFullScreen():
            if self._was_maximized:
                self.showMaximized()
            else:
                self.showNormal()
                if self._normal_geometry is not None:
                    self.setGeometry(self._normal_geometry)
        else:
            self._normal_geometry = self.geometry()
            self._was_maximized = self.isMaximized()
            self.showFullScreen()
        # Fit mode follows the new viewport; a manual zoom keeps its scale.
        QTimer.singleShot(0, self.canvas_resized)

    # ------------------------------------------------------------- apply
    def _can_apply(self) -> bool:
        """Single condition shared by the button and the shortcut.

        Ctrl+Enter used to clear a held preview and publish, even though the
        visible Apply button was disabled during preview.
        """
        return (not self._applied and not self._applying and not self._closing
                and not self._source_loading and not self.doc.stroke_active and self.preview_mode is None
                and self._edit_dialog is None and self._checkpoint_dialog is None
                and self.doc.is_dirty)

    def _apply(self, *, close_on_success=True):
        if not self._can_apply():
            return False
        self.btn_apply.setEnabled(False)
        self._applying = True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.setEnabled(False)
            ok, error = self.apply_callback(self.record, self.doc.working)
        except Exception as exc:
            ok, error = False, str(exc)
        finally:
            self.setEnabled(True)
            QApplication.restoreOverrideCursor()
            self._applying = False
        if ok:
            self._applied = True
            if close_on_success:
                self.accept()
            return True
        self.btn_apply.setEnabled(True)
        QMessageBox.critical(self, trans.t("msg_error"),
                             error or trans.t("msg_proc_error"))
        self._update_controls()
        return False

    def _confirm_close(self) -> bool:
        if self.doc.stroke_active:
            self.end_stroke()
        if not self.doc.is_dirty:
            return True
        box = QMessageBox(self)
        from utils.ui_utils import style_message_box
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(trans.t("retouch_close_title"))
        box.setText(trans.t("retouch_close_text"))
        # A narrow message box clipped the first button's last glyph
        # ("作为新输出应用") even though the action was fully explained.
        box.setMinimumWidth(620)
        apply_btn = box.addButton(trans.t("retouch_apply_close"),
                                  QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton(trans.t("btn_discard_changes"),
                                    QMessageBox.ButtonRole.DestructiveRole)
        continue_btn = box.addButton(trans.t("btn_continue"),
                                     QMessageBox.ButtonRole.RejectRole)
        for button in (apply_btn, discard_btn, continue_btn):
            button.setMinimumWidth(120)
        self._confirm_box = box
        self._confirm_apply_button = apply_btn
        apply_btn.setEnabled(self._can_apply())
        if self._source_loading:
            apply_btn.setToolTip(trans.t('msg_loading_stack'))
        box.setDefaultButton(continue_btn)
        try:
            box.exec()
            clicked = box.clickedButton()
            if clicked is apply_btn:
                return self._apply(close_on_success=False)
            return clicked is discard_btn
        finally:
            self._confirm_apply_button = None
            self._confirm_box = None
            box.deleteLater()


    def reject(self) -> None:
        self.close()

    def closeEvent(self, event) -> None:
        if self._edit_dialog is not None:
            self._edit_dialog.reject()
            event.ignore()
            return
        if self._checkpoint_dialog is not None:
            self._checkpoint_dialog.reject()
            event.ignore()
            return
        if self._applying or self._confirming_close:
            event.ignore()
            return
        if self._closing:
            event.accept()
            return
        self._confirming_close = True
        try:
            confirmed = self._applied or self._allow_close or self._confirm_close()
        finally:
            self._confirming_close = False
        if not confirmed:
            event.ignore()
            return
        self._allow_close = True
        self._closing = True
        self._save_geometry()
        self._shutdown_sharpness()
        self.deleteLater()
        event.accept()

    def changeEvent(self, event) -> None:
        if not self._ui_ready:
            super().changeEvent(event)
            return
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            if self.doc.stroke_active:
                self.end_stroke()
            self._clear_preview()
            self._held_keys.clear()
            self.space_down = False
            self.source_canvas.setCursor(Qt.CursorShape.CrossCursor)
            self.result_canvas.setCursor(Qt.CursorShape.CrossCursor)
        super().changeEvent(event)

    # ----------------------------------------------------------- keyboard
    def _editing_text(self) -> bool:
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QAbstractSpinBox)):
            return True
        parent = focused.parent() if focused is not None else None
        return isinstance(parent, QAbstractSpinBox)

    def _canvas_focused(self) -> bool:
        focused = QApplication.focusWidget()
        return focused in (self.source_canvas, self.result_canvas, self)

    def _handle_key(self, event) -> bool:
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & (Qt.KeyboardModifier.ControlModifier
                            | Qt.KeyboardModifier.MetaModifier))
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if self._confirming_close or self._applying:
            return True
        if event.isAutoRepeat() and key in (Qt.Key.Key_Tab, Qt.Key.Key_Alt, Qt.Key.Key_Space):
            return key in self._held_keys
        if event.type() == QEvent.Type.KeyRelease:
            if key not in self._held_keys:
                return False
            self._held_keys.discard(key)
            if key == Qt.Key.Key_Tab:
                self._release_preview("before")
                return True
            if key == Qt.Key.Key_Alt:
                self._release_preview("source")
                return True
            if key == Qt.Key.Key_Space:
                self.space_down = False
                self.source_canvas.setCursor(Qt.CursorShape.CrossCursor)
                self.result_canvas.setCursor(Qt.CursorShape.CrossCursor)
                return True
            return False
        if event.type() != QEvent.Type.KeyPress or event.isAutoRepeat():
            return False

        if key == Qt.Key.Key_Escape:
            if self.doc.stroke_active:
                self._cancel_stroke()
            elif self.preview_mode is not None:
                self._clear_preview()
            elif self.isFullScreen():
                self._toggle_fullscreen()
            else:
                self.close()
            return True
        if key == Qt.Key.Key_F11:
            self._toggle_fullscreen()
            return True
        if self._source_loading:
            if not ctrl and key == Qt.Key.Key_Comma:
                self._prev_frame()
            elif not ctrl and key == Qt.Key.Key_Period:
                self._next_frame()
            return True
        if ctrl and key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._apply()
            return True
        if ctrl and key == Qt.Key.Key_Z:
            self._redo() if shift else self._undo()
            return True
        if ctrl and key == Qt.Key.Key_Y:
            self._redo()
            return True
        # Shift+Tab must remain a normal focus-navigation key: the "hold to
        # see the result before editing" preview is plain Tab on the canvas.
        if key == Qt.Key.Key_Tab and not shift and self._canvas_focused():
            if self.preview_mode is not None or self.doc.stroke_active:
                return False
            self._press_preview("before")
            if self.preview_mode == 'before':
                self._held_keys.add(key)
            return True
        if key == Qt.Key.Key_Alt and self._canvas_focused():
            if self.preview_mode is not None or self.doc.stroke_active:
                return False
            self._press_preview("source")
            if self.preview_mode == 'source':
                self._held_keys.add(key)
            return True
        if key == Qt.Key.Key_Space and self._canvas_focused():
            self._held_keys.add(key)
            self.space_down = True
            self.source_canvas.setCursor(Qt.CursorShape.OpenHandCursor)
            self.result_canvas.setCursor(Qt.CursorShape.OpenHandCursor)
            return True
        # Canvas letters and punctuation stay inside the canvas: typing in a
        # numeric field must never resize the brush or change frames.
        if not self._canvas_focused():
            return False
        if key == Qt.Key.Key_F:
            self._fit()
            return True
        if key == Qt.Key.Key_1:
            self._zoom_100()
            return True
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._zoom_step(1)
            return True
        if key == Qt.Key.Key_Minus:
            self._zoom_step(-1)
            return True
        if key == Qt.Key.Key_B:
            self._set_tool("source")
            return True
        if key == Qt.Key.Key_E:
            self._set_tool("restore")
            return True
        if key == Qt.Key.Key_BracketLeft:
            self._bump_diameter(0.8)
            return True
        if key == Qt.Key.Key_BracketRight:
            self._bump_diameter(1.25)
            return True
        if key == Qt.Key.Key_Comma:
            self._prev_frame()
            return True
        if key == Qt.Key.Key_Period:
            self._next_frame()
            return True
        return False

    def eventFilter(self, obj, event) -> bool:
        if self._edit_dialog is not None or self._checkpoint_dialog is not None:
            return super().eventFilter(obj, event)
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if self._confirming_close or self._applying:
                return True
            if event.type() == QEvent.Type.KeyRelease:
                # Releasing a temporarily held key is cleanup, not editing:
                # it must run even when a field grabbed focus meanwhile.
                if (event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Alt, Qt.Key.Key_Space)
                        and self._handle_key(event)):
                    return True
            elif not self._editing_text() and self._handle_key(event):
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if self._confirming_close or self._applying:
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
                event.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                     | Qt.KeyboardModifier.MetaModifier)):
            event.accept()
            return
        if not self._editing_text() and self._handle_key(event):
            event.accept()
            return
        super().keyPressEvent(event)


