# dialogs/roi.py
"""ROI-related dialogs."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QRadioButton, QGroupBox, QComboBox, QDialogButtonBox, QLabel, QHBoxLayout, QPushButton
)
from PyQt6.QtCore import Qt
from locales import trans


class ROIRenderOptionsDialog(QDialog):
    """Dialog to ask user how to handle ROI fusion result."""

    def __init__(self, num_frames: int, parent=None, *,
                 frame_names=None, roi_size=None, output_size=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("dialog_roi_title"))
        self.resize(460, 320)
        self._frame_names = list(frame_names or [])
        self._roi_size = tuple(roi_size) if roi_size else None
        self._output_size = tuple(output_size) if output_size else None

        self.mode = "crop" # 'crop' or 'paste'
        self.base_frame_index = 0

        layout = QVBoxLayout(self)

        lbl = QLabel(trans.t("dialog_roi_msg"))
        layout.addWidget(lbl)

        # Options Group
        group = QGroupBox(trans.t("dialog_roi_group_output"))
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
            label = (trans.t("dialog_roi_frame_named").format(
                i=i + 1, name=name) if name
                else trans.t("dialog_roi_frame_plain").format(i=i + 1))
            self.combo_frames.addItem(label, i)
        self.combo_frames.setCurrentIndex(0)

        hbox.addWidget(self.combo_frames)
        self.frame_container.setLayout(hbox)
        self.frame_container.setEnabled(False) # Default disabled
        layout.addWidget(self.frame_container)

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
        self.frame_container.setEnabled(not crop)
        size = self._roi_size if crop else self._output_size
        text = trans.t("dialog_roi_size_crop") if crop else trans.t(
            "dialog_roi_size_paste")
        if size:
            text += " " + trans.t("dialog_roi_size_px").format(
                w=size[0], h=size[1])
        self.lbl_output.setText(text)

    def accept(self):
        if self.rb_paste.isChecked():
            self.mode = "paste"
            data = self.combo_frames.currentData()
            self.base_frame_index = (int(data) if data is not None
                                     else self.combo_frames.currentIndex())
        else:
            self.mode = "crop"
        super().accept()
