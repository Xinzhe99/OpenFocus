from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFormLayout,
    QFontComboBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from styles import WHITE_COMBOBOX_STYLE
from utils import (
    LabelAdder,
    cv2_to_pixmap,
    pixmap_to_cv2,
    show_success_box,
    show_warning_box,
)


class LabelManager:
    """Encapsulates label configuration dialogs and state toggles."""

    def __init__(self, window: Any) -> None:
        self.window = window

        self._input_adder = LabelAdder()
        self._input_adder.config.target_stack = 0
        self._registered_adder = LabelAdder()
        self._registered_adder.config.target_stack = 1

        self._input_enabled = False
        self._registered_enabled = False

        self._clear_dialog_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def delete_registered_labels(self) -> None:
        window = self.window

        if not self._registered_enabled:
            show_warning_box(window, "No Labels", "Registered stack labels are not enabled.")
            return

        self._registered_enabled = False
        window.del_reg_label_action.setEnabled(False)

        if window.fusion_result is not None:
            window.display_fusion_result()
        elif window.registration_results and window.current_result_index >= 0:
            window.update_result_view(window.current_result_index)

        show_success_box(window, "Success", "Labels removed from Registered Stack.")

    def delete_input_labels(self) -> None:
        window = self.window

        if not self._input_enabled:
            show_warning_box(window, "No Labels", "Input stack labels are not enabled.")
            return

        self._input_enabled = False
        window.del_input_label_action.setEnabled(False)

        if window.stack_images and window.current_display_index >= 0:
            window.update_source_view(window.current_display_index)

        show_success_box(window, "Success", "Labels removed from Input Stack.")

    def reset_labels(self) -> None:
        """Disable all labels and update menu state."""
        self._input_enabled = False
        self._registered_enabled = False
        if hasattr(self.window, "del_input_label_action"):
            self.window.del_input_label_action.setEnabled(False)
        if hasattr(self.window, "del_reg_label_action"):
            self.window.del_reg_label_action.setEnabled(False)

    def apply_labels_to_source_pixmap(self, pixmap, index: int):
        """Return a pixmap with input labels applied when enabled."""
        if not self._input_enabled:
            return pixmap

        image = pixmap_to_cv2(pixmap)
        if image is None:
            return pixmap

        labeled_image = self._input_adder.add_label_to_image(image.copy(), index)
        return cv2_to_pixmap(labeled_image)

    def apply_labels_to_registered_image(self, image, index: int):
        """Return a BGR image with registered labels applied when enabled."""
        if not self._registered_enabled:
            return image

        return self._registered_adder.add_label_to_image(image.copy(), index)

    def prepare_bgr_image(self, target: str, image, index: int):
        """Prepare a copy of the image with labels applied for saving."""
        output = image.copy()
        if target == "registered":
            if self._registered_enabled:
                return self._registered_adder.add_label_to_image(output, index)
            return output
        if target == "input":
            if self._input_enabled:
                return self._input_adder.add_label_to_image(output, index)
            return output
        raise ValueError(f"Unknown label target: {target}")

    def is_enabled_for_target(self, target: str) -> bool:
        """Expose current label enabled state for the requested stack."""
        if target == "registered":
            return self._registered_enabled
        if target == "input":
            return self._input_enabled
        raise ValueError(f"Unknown label target: {target}")

    def show_add_label_dialog(self) -> None:
        window = self.window

        dialog = QDialog(window)
        dialog.setWindowTitle("Add Label Configuration")
        dialog.setFixedSize(500, 650)

        from styles import ADD_LABEL_DIALOG_STYLE  # Local import to avoid circular deps

        dialog.setStyleSheet(ADD_LABEL_DIALOG_STYLE)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form_layout.setHorizontalSpacing(20)
        form_layout.setVerticalSpacing(12)
        form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.target_stack_combo = QComboBox()
        self.target_stack_combo.addItems(["Input Image Stack", "Registered Image Stack"])
        self.target_stack_combo.setCurrentIndex(1)
        self.target_stack_combo.setMinimumWidth(200)
        self.target_stack_combo.setStyleSheet(WHITE_COMBOBOX_STYLE)
        form_layout.addRow("Target Stack:", self.target_stack_combo)

        self.format_edit = QLineEdit("{value}")
        self.format_edit.setMinimumWidth(200)
        self.format_edit.setStyleSheet("background-color: #fff; color: #000; font-weight: normal;")
        form_layout.addRow("Format String:", self.format_edit)

        self.starting_value_spin = QSpinBox()
        self.starting_value_spin.setRange(0, 9999)
        self.starting_value_spin.setValue(1)
        self.starting_value_spin.setMinimumWidth(200)
        self.starting_value_spin.setStyleSheet("background-color: #fff; color: #000; font-weight: normal;")
        form_layout.addRow("Starting Value:", self.starting_value_spin)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 9999)
        self.interval_spin.setValue(1)
        self.interval_spin.setMinimumWidth(200)
        self.interval_spin.setStyleSheet("background-color: #fff; color: #000; font-weight: normal;")
        form_layout.addRow("Interval:", self.interval_spin)

        self.x_location_spin = QSpinBox()
        self.x_location_spin.setRange(0, 9999)
        self.x_location_spin.setValue(20)
        self.x_location_spin.setMinimumWidth(200)
        self.x_location_spin.setStyleSheet("background-color: #fff; color: #000; font-weight: normal;")
        form_layout.addRow("X Location:", self.x_location_spin)

        self.y_location_spin = QSpinBox()
        self.y_location_spin.setRange(0, 9999)
        self.y_location_spin.setValue(80)
        self.y_location_spin.setMinimumWidth(200)
        self.y_location_spin.setStyleSheet("background-color: #fff; color: #000; font-weight: normal;")
        form_layout.addRow("Y Location:", self.y_location_spin)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(1, 100)
        self.font_size_spin.setValue(80)
        self.font_size_spin.setMinimumWidth(200)
        self.font_size_spin.setStyleSheet("background-color: #fff; color: #000; font-weight: normal;")
        form_layout.addRow("Font Size:", self.font_size_spin)

        self.font_family_combo = QFontComboBox()
        self.font_family_combo.setEditable(True)
        self.font_family_combo.setCurrentFont(QFont("Arial"))
        self.font_family_combo.setMinimumWidth(200)
        self.font_family_combo.setStyleSheet(WHITE_COMBOBOX_STYLE)
        form_layout.addRow("Font Family:", self.font_family_combo)

        self.text_edit = QLineEdit("")
        self.text_edit.setMinimumWidth(200)
        self.text_edit.setStyleSheet("background-color: #fff; color: #000; font-weight: normal;")
        form_layout.addRow("Custom Text:", self.text_edit)

        self.range_edit = QLineEdit("All")
        self.range_edit.setMinimumWidth(200)
        self.range_edit.setStyleSheet("background-color: #fff; color: #000; font-weight: normal;")
        form_layout.addRow("Range:", self.range_edit)

        self.transparent_bg_checkbox = QCheckBox("Transparent Background")
        self.transparent_bg_checkbox.setChecked(True)
        form_layout.addRow("", self.transparent_bg_checkbox)

        bg_button_width = 200
        font_button_width = 170
        color_button_height = 32

        bg_color_layout = QHBoxLayout()
        bg_color_layout.setContentsMargins(0, 0, 0, 0)
        bg_color_layout.setSpacing(10)

        self.bg_color_button = QPushButton("Choose Background Color")
        self.bg_color_button.setMinimumWidth(150)
        self.bg_color_button.setMaximumWidth(bg_button_width)
        self.bg_color_button.setFixedHeight(color_button_height)
        self.bg_color_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.bg_color_value = (0, 0, 0)
        self._set_color_button_style(self.bg_color_button, 0, 0, 0)
        self.bg_color_button.clicked.connect(self.choose_background_color)

        bg_color_layout.addWidget(self.bg_color_button)
        bg_color_layout.addStretch()
        form_layout.addRow("Background Color:", bg_color_layout)

        font_color_layout = QHBoxLayout()
        font_color_layout.setContentsMargins(0, 0, 0, 0)
        font_color_layout.setSpacing(10)

        self.font_color_button = QPushButton("Choose Font Color")
        self.font_color_button.setMinimumWidth(130)
        self.font_color_button.setMaximumWidth(font_button_width)
        self.font_color_button.setFixedHeight(color_button_height)
        self.font_color_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.font_color_value = (255, 255, 255)
        self._set_color_button_style(self.font_color_button, 255, 255, 255)
        self.font_color_button.clicked.connect(self.choose_font_color)

        font_color_layout.addWidget(self.font_color_button)
        font_color_layout.addStretch()
        form_layout.addRow("Font Color:", font_color_layout)

        layout.addLayout(form_layout)
        layout.addSpacing(20)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_button = QPushButton("OK")
        ok_button.setMinimumWidth(80)
        cancel_button = QPushButton("Cancel")
        cancel_button.setMinimumWidth(80)

        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        accepted = dialog.exec() == QDialog.DialogCode.Accepted

        if accepted:
            self._apply_label_configuration()

        self._clear_dialog_state()

    def choose_background_color(self) -> None:
        color = QColorDialog.getColor(parent=self.window)

        if color.isValid() and self.bg_color_button is not None:
            self.bg_color_value = (color.blue(), color.green(), color.red())
            self._set_color_button_style(self.bg_color_button, color.red(), color.green(), color.blue())

    def choose_font_color(self) -> None:
        color = QColorDialog.getColor(parent=self.window)

        if color.isValid() and self.font_color_button is not None:
            self.font_color_value = (color.blue(), color.green(), color.red())
            self._set_color_button_style(self.font_color_button, color.red(), color.green(), color.blue())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _set_color_button_style(self, button: QPushButton, red: int, green: int, blue: int) -> None:
        brightness = (red * 299 + green * 587 + blue * 114) / 1000
        text_color = "black" if brightness > 150 else "white"
        button.setStyleSheet(
            f"background-color: rgb({red}, {green}, {blue}); "
            f"color: {text_color}; border: 1px solid #555; padding: 6px 12px; border-radius: 4px;"
        )

    def _apply_label_configuration(self) -> None:
        window = self.window

        config_dict = {
            "target_stack": self.target_stack_combo.currentIndex(),
            "format": self.format_edit.text(),
            "starting_value": self.starting_value_spin.value(),
            "interval": self.interval_spin.value(),
            "x_location": self.x_location_spin.value(),
            "y_location": self.y_location_spin.value(),
            "font_size": self.font_size_spin.value(),
            "font_family": self.font_family_combo.currentFont().family(),
            "text": self.text_edit.text(),
            "range": self.range_edit.text(),
            "transparent_bg": self.transparent_bg_checkbox.isChecked(),
            "bg_color": self.bg_color_value,
            "font_color": self.font_color_value,
        }

        target_stack_index = config_dict["target_stack"]

        if target_stack_index == 1:
            self._registered_adder.config.update_config(config_dict)
            self._registered_enabled = True
            window.del_reg_label_action.setEnabled(True)

            if window.registration_results and window.current_result_index >= 0:
                window.update_result_view(window.current_result_index)
            elif window.fusion_result is not None:
                window.display_fusion_result()
        else:
            self._input_adder.config.update_config(config_dict)
            self._input_enabled = True
            window.del_input_label_action.setEnabled(True)

            if window.stack_images and window.current_display_index >= 0:
                window.update_source_view(window.current_display_index)

        show_success_box(
            window,
            "Configuration Saved",
            "Label configuration saved successfully!",
            "The labels are now visible on the selected stack and will be included when saving.",
        )

    def _clear_dialog_state(self) -> None:
        self.target_stack_combo: Optional[QComboBox] = None
        self.format_edit: Optional[QLineEdit] = None
        self.starting_value_spin: Optional[QSpinBox] = None
        self.interval_spin: Optional[QSpinBox] = None
        self.x_location_spin: Optional[QSpinBox] = None
        self.y_location_spin: Optional[QSpinBox] = None
        self.font_size_spin: Optional[QSpinBox] = None
        self.font_family_combo: Optional[QFontComboBox] = None
        self.text_edit: Optional[QLineEdit] = None
        self.range_edit: Optional[QLineEdit] = None
        self.transparent_bg_checkbox: Optional[QCheckBox] = None
        self.bg_color_button: Optional[QPushButton] = None
        self.font_color_button: Optional[QPushButton] = None
        self.bg_color_value = (0, 0, 0)
        self.font_color_value = (255, 255, 255)