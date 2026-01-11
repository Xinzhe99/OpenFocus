# dialogs/roi.py
"""ROI-related dialogs."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QRadioButton, QGroupBox, QComboBox, QDialogButtonBox, QLabel, QHBoxLayout, QPushButton
)
from PyQt6.QtCore import Qt
from locales import trans


class ROIRenderOptionsDialog(QDialog):
    """Dialog to ask user how to handle ROI fusion result."""

    def __init__(self, num_frames: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(trans.t("dialog_roi_title"))
        self.resize(400, 250)
        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: #ddd; }
            QLabel, QRadioButton { color: #ddd; font-size: 13px; }
            QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 10px; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; color: #aaa; }
            QComboBox { background: #333; color: white; border: 1px solid #555; padding: 5px; }
            QComboBox QAbstractItemView {
                background-color: white;
                color: black;
                selection-background-color: #0078d7;
                selection-color: white;
            }
            QPushButton { background-color: #0078d7; color: white; border: none; padding: 8px 16px; border-radius: 4px; }
            QPushButton:hover { background-color: #0063b1; }
        """)

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
            self.combo_frames.addItem(f"Frame {i+1}")
        self.combo_frames.setCurrentIndex(0) # Default to Frame 1
        
        hbox.addWidget(self.combo_frames)
        self.frame_container.setLayout(hbox)
        self.frame_container.setEnabled(False) # Default disabled
        layout.addWidget(self.frame_container)

        # Logic
        self.rb_crop.toggled.connect(lambda c: self.frame_container.setEnabled(not c))

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        
        # Translate buttons
        btn_ok = btns.button(QDialogButtonBox.StandardButton.Ok)
        if btn_ok:
            btn_ok.setText(trans.t('btn_ok'))
            
        btn_cancel = btns.button(QDialogButtonBox.StandardButton.Cancel)
        if btn_cancel:
            btn_cancel.setText(trans.t('btn_cancel'))

        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def accept(self):
        if self.rb_paste.isChecked():
            self.mode = "paste"
            self.base_frame_index = self.combo_frames.currentIndex()
        else:
            self.mode = "crop"
        super().accept()
