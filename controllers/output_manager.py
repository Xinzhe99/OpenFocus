import os
from typing import Any

import cv2
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QAction, QIcon, QPixmap, QImage
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem, QMenu, QMessageBox

from utils import show_error_box, show_message_box, show_success_box, show_warning_box


class OutputManager:
    """Encapsulates output panel interactions and state updates."""

    def __init__(self, window: Any):
        self.window = window

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def update_output_count(self) -> None:
        count = self.window.output_list.count()
        self.window.output_label.setText(f"Output: {count}")

    def update_output_list_for_fusion(self) -> None:
        window = self.window

        fusion_filename = window.export_manager.generate_default_filename() + ".png"

        if window.fusion_result is not None:
            window.fusion_results.insert(0, window.fusion_result.copy())

        item = QListWidgetItem(fusion_filename)
        pixmap = window.lbl_result_img.pixmap()
        if pixmap is not None:
            icon = QIcon(pixmap.scaled(40, 40, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, transformMode=Qt.TransformationMode.SmoothTransformation))
            item.setIcon(icon)
        window.output_list.insertItem(0, item)
        window.output_list.setCurrentRow(0)
        self.update_output_count()

    def sync_output_slider_from_list(self, row: int) -> None:
        if row >= 0:
            self.window.result_slider.setValue(row)

    # ------------------------------------------------------------------
    # Context menu & list operations
    # ------------------------------------------------------------------
    def show_output_context_menu(self, position: QPoint) -> None:
        window = self.window
        menu = QMenu(window)

        delete_action = QAction("Delete", window)
        delete_action.triggered.connect(self.delete_selected_output_images)
        menu.addAction(delete_action)

        save_as_action = QAction("Save as", window)
        save_as_action.triggered.connect(lambda: self.save_output_image_as(window.output_list.currentItem()))
        save_as_action.setEnabled(len(window.output_list.selectedItems()) == 1)
        menu.addAction(save_as_action)

        menu.exec(window.output_list.mapToGlobal(position))

    def delete_output_image(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.output_list.row(item)
        if row < 0:
            return

        window.output_list.takeItem(row)

        if 0 <= row < len(window.fusion_results):
            window.fusion_results.pop(row)

        self.update_output_count()

        if window.fusion_results:
            new_index = min(row, len(window.fusion_results) - 1)
            window.fusion_result = window.fusion_results[new_index]
            window.output_list.setCurrentRow(new_index)
            self.display_specific_fusion_result(window.fusion_result)
        else:
            window.fusion_result = None
            window.lbl_result_img.clear()
            window.result_control_bar.setVisible(False)

    def save_output_image_as(self, item: QListWidgetItem | None) -> None:
        window = self.window
        if item is None:
            return

        row = window.output_list.row(item)
        if row < 0:
            return

        default_filename = item.text()
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            window,
            "Save as",
            default_filename,
            "PNG Files (*.png);;JPG Files (*.jpg);;Bitmap Files (*.bmp);;TIFF Files (*.tif *.tiff);;All Files (*)",
        )

        if not file_path:
            return

        fallback_ext = os.path.splitext(default_filename)[1]
        file_path = window.export_manager.normalize_export_path(
            file_path,
            fallback_extension=fallback_ext if fallback_ext else None,
        )

        try:
            if row < len(window.fusion_results):
                image_to_save = window.label_manager.prepare_bgr_image(
                    "registered", window.fusion_results[row], 0
                )

                success = cv2.imwrite(file_path, image_to_save)
                if success:
                    show_success_box(window, "Success", "Image saved successfully!", f"Image saved to:\n{file_path}")
                else:
                    show_error_box(window, "Save Failed", "Failed to save the image.", "Unable to write image to the specified file path.")
            elif window.fusion_result is not None and row == 0:
                image_to_save = window.label_manager.prepare_bgr_image(
                    "registered", window.fusion_result, 0
                )

                success = cv2.imwrite(file_path, image_to_save)
                if success:
                    show_success_box(window, "Success", "Image saved successfully!", f"Image saved to:\n{file_path}")
                else:
                    show_error_box(window, "Save Failed", "Failed to save the image.", "Unable to write image to the specified file path.")
            else:
                show_warning_box(window, "Warning", "No valid image to save.", "The selected item does not contain a valid image.")
        except cv2.error as exc:
            show_error_box(window, "Save Failed", "Failed to save the image.", f"OpenCV Error: {str(exc)}\n\nPlease check the file extension and ensure it is supported.")
        except Exception as exc:  # pylint: disable=broad-except
            show_error_box(window, "Save Failed", "Failed to save the image.", f"Unexpected error: {str(exc)}")

    def delete_selected_output_images(self) -> None:
        window = self.window
        selected_items = window.output_list.selectedItems()
        if not selected_items:
            return

        rows = [window.output_list.row(item) for item in selected_items]
        rows.sort(reverse=True)

        for row in rows:
            if row >= 0:
                window.output_list.takeItem(row)
                if 0 <= row < len(window.fusion_results):
                    window.fusion_results.pop(row)

        self.update_output_count()

        if window.fusion_results:
            min_row = min(rows) if rows else 0
            new_index = min(min_row, len(window.fusion_results) - 1)
            window.fusion_result = window.fusion_results[new_index]
            window.output_list.setCurrentRow(new_index)
            self.display_specific_fusion_result(window.fusion_result)
        else:
            window.fusion_result = None
            window.lbl_result_img.clear()
            window.result_control_bar.setVisible(False)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def display_specific_fusion_result(self, fusion_image) -> None:
        if fusion_image is None:
            return

        window = self.window

        try:
            display_image = window.label_manager.prepare_bgr_image("registered", fusion_image, 0)
            rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            height, width, _channels = rgb_image.shape
            bytes_per_line = 3 * width

            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            window.lbl_result_img.set_display_pixmap(pixmap)
            window.result_control_bar.setVisible(False)
            window.lbl_result_info.setText("-- / --")
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                "Display Error",
                "Failed to display the fusion result.",
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )

    def display_output_image_in_result_view(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.output_list.row(item)
        if row < 0:
            return

        if row < len(window.fusion_results):
            self.display_specific_fusion_result(window.fusion_results[row])
        elif window.fusion_result is not None:
            self.show_fusion_result()

    def show_fusion_result(self) -> None:
        window = self.window
        if window.fusion_result is None:
            return

        self.display_specific_fusion_result(window.fusion_result)

    def show_registration_result(self, index: int) -> None:
        window = self.window

        if window.fusion_result is not None:
            return
        if not window.registration_results:
            return
        if not 0 <= index < len(window.registration_results):
            return

        window.current_result_index = index

        try:
            image = window.registration_results[index]
            image = window.label_manager.apply_labels_to_registered_image(image, index)

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width

            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            window.lbl_result_img.set_display_pixmap(pixmap)
            window.lbl_result_info.setText(f"{index + 1} / {len(window.registration_results)}")
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                "Display Error",
                "Failed to display the registration result.",
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )

    def refresh_current_result_view(self) -> None:
        window = self.window
        if window.fusion_result is not None:
            self.show_fusion_result()
        elif window.current_result_index >= 0:
            self.show_registration_result(window.current_result_index)