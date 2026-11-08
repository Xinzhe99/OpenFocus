# dialogs/roi.py
"""ROI-related dialogs."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QRadioButton, QGroupBox, QComboBox, QDialogButtonBox, QLabel, QHBoxLayout, QPushButton
)
from PyQt6.QtCore import Qt
from locales import trans
import numpy as np


# A closed dialog may still have an uncancellable native warp finishing.
_ROI_PREVIEW_WORKERS = set()


class ROIRenderOptionsDialog(QDialog):
    """Dialog to ask user how to handle ROI fusion result."""

    def __init__(self, num_frames: int, parent=None, *,
                 frame_names=None, roi_size=None, output_size=None,
                 frames=None, frame_indices=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("dialog_roi_title"))
        self.resize(460, 320)
        self._frame_names = list(frame_names or [])
        self._roi_size = tuple(roi_size) if roi_size else None
        self._output_size = tuple(output_size) if output_size else None
        self._frames = list(frames or [])
        self._frame_indices = list(frame_indices) if frame_indices is not None else list(range(num_frames))
        self._preview_key = None
        self._preview_workers = set()
        self._pending_preview = None
        self.finished.connect(self._cancel_previews)

        self.mode = "crop" # 'crop' or 'paste'
        self.base_frame_index = 0

        layout = QVBoxLayout(self)

        # Options Group
        group = QGroupBox(trans.t("dialog_roi_group_output"))
        group.setToolTip(trans.t("dialog_roi_msg"))
        vbox = QVBoxLayout()

        self.rb_crop = QRadioButton(trans.t("dialog_roi_opt_crop"))
        self.rb_crop.setToolTip(trans.t("dialog_roi_opt_crop_tooltip"))
        self.rb_crop.setChecked(True)
        
        self.rb_paste = QRadioButton(trans.t("dialog_roi_opt_paste"))
        self.rb_paste.setToolTip(trans.t("dialog_roi_opt_paste_tooltip"))

        vbox.addWidget(self.rb_crop)
        vbox.addWidget(self.rb_paste)
        group.setLayout(vbox)
        layout.addWidget(group)

        # Frame Selection (Visible only if Paste selected)
        self.frame_container = QGroupBox(trans.t("dialog_roi_group_base"))
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel(trans.t("dialog_roi_lbl_base")))
        
        self.combo_frames = QComboBox()
        for i in range(num_frames):
            # Name the frame, not just its position: "Frame 3" was ambiguous
            # once frames had been excluded or reordered.
            name = (self._frame_names[i]
                    if i < len(self._frame_names) else "")
            number = self._frame_indices[i] + 1
            label = (trans.t("dialog_roi_frame_named").format(
                i=number, name=name) if name
                else trans.t("dialog_roi_frame_plain").format(i=number))
            self.combo_frames.addItem(label, i)
        self.combo_frames.setCurrentIndex(0)

        hbox.addWidget(self.combo_frames)
        self.frame_container.setLayout(hbox)
        self.frame_container.hide()
        layout.addWidget(self.frame_container)
        self.lbl_preview = QLabel()
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setMinimumHeight(100)
        layout.addWidget(self.lbl_preview)

        # State what the two choices actually produce.
        self.lbl_output = QLabel("")
        self.lbl_output.setWordWrap(True)
        self.lbl_output.setObjectName("mutedLabel")
        layout.addWidget(self.lbl_output)
        layout.addStretch(1)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        
        # Translate buttons
        btn_ok = btns.button(QDialogButtonBox.StandardButton.Ok)
        if btn_ok:
            btn_ok.setText(trans.t('btn_ok'))
            # Same primary weight as every other dialog's confirm action.
            btn_ok.setObjectName("primaryButton")
            
        btn_cancel = btns.button(QDialogButtonBox.StandardButton.Cancel)
        if btn_cancel:
            btn_cancel.setText(trans.t('btn_cancel'))

        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Logic (after every widget exists)
        self.rb_crop.toggled.connect(self._sync_choice)
        self.rb_paste.toggled.connect(self._sync_choice)
        self.combo_frames.currentIndexChanged.connect(
            lambda _i: self._sync_choice())
        self._sync_choice()

    def _sync_choice(self, *_args):
        crop = self.rb_crop.isChecked()
        self.frame_container.setVisible(not crop)
        index = self.combo_frames.currentIndex()
        if 0 <= index < len(self._frames):
            frame = self._frames[index]
            height, width = frame.shape[:2]
            self._output_size = (width, height)
            if not crop:
                self._show_base_preview(index, frame)
        if crop:
            self._cancel_previews()
        self.lbl_preview.setVisible(not crop)
        size = self._roi_size if crop else self._output_size
        text = trans.t("dialog_roi_size_crop") if crop else trans.t(
            "dialog_roi_size_paste")
        if size:
            text += " " + trans.t("dialog_roi_size_px").format(
                w=size[0], h=size[1])
        self.lbl_output.setText(text)

    def _cancel_previews(self, *_args):
        self._preview_key = None
        self._pending_preview = None
        for worker in tuple(self._preview_workers):
            worker.request_cancel()

    def _show_base_preview(self, index, frame):
        key = (index, id(frame))
        if self._preview_key == key:
            return
        self._cancel_previews()
        self._preview_key = key
        self.lbl_preview.clear()
        metadata = getattr(frame, 'linear_metadata', None) or getattr(self.parent(), 'linear_dng_metadata', None)
        if isinstance(frame, np.ndarray):
            # Already owned arrays need no source resolution. Bound development
            # and Qt upload to the thumbnail, as for background region delivery.
            from core.source_previews import render_source_region
            from utils.image_utils import cv2_to_pixmap
            image, _ = render_source_region(frame, (0, 0, frame.shape[1], frame.shape[0]),
                                             (238, 118), metadata)
            self.lbl_preview.setPixmap(cv2_to_pixmap(image))
            return
        if self._preview_workers:
            # A cancelled native operation must retire before the latest
            # selected descriptor can start another QThread.
            self._pending_preview = (index, frame)
            return
        from core.workers import ROIFramePreviewWorker
        from utils.image_utils import cv2_to_pixmap
        request = ('preview', 0, (0, 0, frame.shape[1], frame.shape[0]), (238, 118))
        worker = ROIFramePreviewWorker(frame, key, request, metadata=metadata)
        self._preview_workers.add(worker)
        _ROI_PREVIEW_WORKERS.add(worker)

        def ready(decoded_key, request, image, rect):
            worker.request_cancel()
            if decoded_key == self._preview_key and image is not None:
                self.lbl_preview.setPixmap(cv2_to_pixmap(image))

        def failed(decoded_key, message):
            if decoded_key == self._preview_key:
                self.lbl_preview.setText(message)

        def finished():
            self._preview_workers.discard(worker)
            _ROI_PREVIEW_WORKERS.discard(worker)
            worker.deleteLater()
            pending = self._pending_preview
            self._pending_preview = None
            if pending is not None and self._preview_key == (pending[0], id(pending[1])):
                self._preview_key = None
                self._show_base_preview(*pending)

        worker.ready.connect(ready)
        worker.failed.connect(failed)
        worker.finished.connect(finished)
        worker.start()

    def accept(self):
        if self.rb_paste.isChecked():
            self.mode = "paste"
            data = self.combo_frames.currentData()
            self.base_frame_index = (int(data) if data is not None
                                     else self.combo_frames.currentIndex())
        else:
            self.mode = "crop"
        super().accept()
