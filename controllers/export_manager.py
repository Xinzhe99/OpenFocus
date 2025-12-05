import datetime
import os
from typing import Any, Optional

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from dialogs import DurationDialog
from styles import PROGRESS_DIALOG_STYLE
from utils import (
    show_error_box,
    show_message_box,
    show_success_box,
    show_warning_box,
)
from workers import GifSaverWorker

ALLOWED_EXPORT_EXTENSION_MAP = {
    ".png": ".png",
    ".jpg": ".jpg",
    ".bmp": ".bmp",
    ".tif": ".tif",
    ".tiff": ".tiff",
}

EXPORT_EXTENSION_ALIASES = {
    ".jpeg": ".jpg",
    ".jpe": ".jpg",
    ".jfif": ".jpg",
    ".jp2": ".jpg",
}

DEFAULT_EXPORT_EXTENSION = ".png"


class ExportManager:
    """Handles save and export workflows for the main window."""

    def __init__(self, window: Any) -> None:
        self.window = window
        self.gif_progress_dialog: Optional[QProgressDialog] = None
        self.gif_worker: Optional[GifSaverWorker] = None

    def normalize_export_path(self, file_path: str, fallback_extension: str = DEFAULT_EXPORT_EXTENSION) -> str:
        """Map user supplied paths onto the supported export extensions."""
        root, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()

        fallback = (fallback_extension or DEFAULT_EXPORT_EXTENSION).lower()
        if fallback not in ALLOWED_EXPORT_EXTENSION_MAP:
            fallback = DEFAULT_EXPORT_EXTENSION

        if not ext:
            return root + fallback

        mapped = ALLOWED_EXPORT_EXTENSION_MAP.get(ext_lower)
        if mapped:
            return root + mapped

        alias_target = EXPORT_EXTENSION_ALIASES.get(ext_lower)
        if alias_target:
            mapped_alias = ALLOWED_EXPORT_EXTENSION_MAP.get(alias_target)
            if mapped_alias:
                return root + mapped_alias

        return root + fallback

    # ------------------------------------------------------------------
    # Filename helpers
    # ------------------------------------------------------------------
    def generate_default_filename(self) -> str:
        window = self.window
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if window.rb_a.isChecked():
            fusion_method = "GuidedFilter"
        elif window.rb_b.isChecked():
            fusion_method = "DCT"
        elif window.rb_c.isChecked():
            fusion_method = "DTCWT"
        elif window.rb_d.isChecked():
            fusion_method = "StackMFFV4"
        else:
            fusion_method = "None"

        reg_methods = []
        if window.cb_align_homography.isChecked():
            reg_methods.append("Homography")
        if window.cb_align_ecc.isChecked():
            reg_methods.append("ECC")
        reg_method_str = "+".join(reg_methods) if reg_methods else "NoAlign"

        return f"OpenFocus_{timestamp}_{fusion_method}_{reg_method_str}"

    def generate_default_foldername(self) -> str:
        window = self.window
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if window.rb_a.isChecked():
            fusion_method = "GuidedFilter"
        elif window.rb_b.isChecked():
            fusion_method = "DCT"
        elif window.rb_c.isChecked():
            fusion_method = "DTCWT"
        elif window.rb_d.isChecked():
            fusion_method = "StackMFFV4"
        else:
            fusion_method = "None"

        reg_methods = []
        if window.cb_align_homography.isChecked():
            reg_methods.append("Homography")
        if window.cb_align_ecc.isChecked():
            reg_methods.append("ECC")
        reg_method_str = "+".join(reg_methods) if reg_methods else "NoAlign"

        folder_basename = "OpenFocus_Stack"
        if getattr(window, "current_folder_path", None):
            folder_basename = os.path.basename(window.current_folder_path)

        return f"{folder_basename}_{timestamp}_{fusion_method}_{reg_method_str}"

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def save_result(self) -> None:
        window = self.window
        result_to_save = None
        title = ""

        if window.fusion_result is not None:
            result_to_save = window.fusion_result
            title = "Save Fusion Result"
        elif window.registration_results:
            index = window.current_result_index if window.current_result_index >= 0 else 0
            result_to_save = window.registration_results[index]
            title = "Save Registration Result"
        else:
            show_warning_box(window, "No Result", "Please render the result first (registration or fusion).")
            return

        default_filename = self.generate_default_filename()
        file_path, _ = QFileDialog.getSaveFileName(
            window,
            title,
            default_filename,
            "All Supported Formats (*.png *.jpg *.bmp *.tif *.tiff);;"
            "JPG Files (*.jpg);;PNG Files (*.png);;Bitmap Files (*.bmp);;TIFF Files (*.tif *.tiff);;All Files (*)",
        )

        if not file_path:
            return

        file_path = self.normalize_export_path(file_path)

        try:
            index = 0 if window.fusion_result is not None else window.current_result_index
            if index < 0:
                index = 0
            image_to_save = window.label_manager.prepare_bgr_image("registered", result_to_save, index)
            if cv2.imwrite(file_path, image_to_save):
                show_message_box(
                    window,
                    "Success",
                    "Image saved successfully!",
                    f"Image saved to:\n{file_path}",
                    QMessageBox.Icon.Information,
                )
            else:
                show_message_box(
                    window,
                    "Save Failed",
                    "Failed to save the image.",
                    "Unable to write image to the specified file path.",
                    QMessageBox.Icon.Critical,
                )
        except cv2.error as exc:
            show_message_box(
                window,
                "Save Failed",
                "Failed to save the image.",
                f"OpenCV Error: {str(exc)}\n\nPlease check the file extension and ensure it is supported.",
                QMessageBox.Icon.Critical,
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                "Save Failed",
                "Failed to save the image.",
                f"Unexpected error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )

    def save_result_stack(self) -> None:
        window = self.window
        if not window.registration_results:
            show_warning_box(window, "No Stack", "Please perform registration first to save the stack.")
            return

        default_foldername = self.generate_default_foldername()
        folder_path = QFileDialog.getExistingDirectory(
            window,
            "Select Folder to Save Registration Stack",
            default_foldername,
            QFileDialog.Option.ShowDirsOnly,
        )

        if not folder_path:
            return

        try:
            saved_count = 0
            for index, image in enumerate(window.registration_results):
                image_to_save = window.label_manager.prepare_bgr_image("registered", image, index)
                if index < len(window.image_filenames):
                    filename = window.image_filenames[index]
                else:
                    filename = f"registered_{index + 1:04d}{DEFAULT_EXPORT_EXTENSION}"
                file_path = os.path.join(folder_path, filename)
                file_path = self.normalize_export_path(file_path)
                if cv2.imwrite(file_path, image_to_save):
                    saved_count += 1

            show_message_box(
                window,
                "Success",
                "Stack saved successfully!",
                f"Successfully saved {saved_count}/{len(window.registration_results)} images to:\n{folder_path}",
                QMessageBox.Icon.Information,
            )
        except cv2.error as exc:
            show_message_box(
                window,
                "Error",
                "Failed to save stack:",
                f"OpenCV Error: {str(exc)}\n\nPlease check the file path and permissions.",
                QMessageBox.Icon.Critical,
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                "Error",
                "Failed to save stack:",
                f"Unexpected error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )

    def save_as_gif(self, target_type: str = "registered") -> None:
        window = self.window

        if target_type == "registered":
            if not window.registration_results:
                show_warning_box(window, "No Registered Images", "Please perform registration first.")
                return
            images_to_save = window.registration_results
        else:
            if not window.raw_images:
                show_warning_box(window, "No Input Images", "Please load images first.")
                return
            images_to_save = window.raw_images

        duration_dialog = DurationDialog(window)
        if not duration_dialog.exec():
            return

        duration_ms = duration_dialog.get_duration()
        default_filename = self.generate_default_filename() + ".gif"
        file_path, _ = QFileDialog.getSaveFileName(
            window,
            "Save as GIF",
            default_filename,
            "GIF Files (*.gif);;All Files (*)",
        )

        if not file_path:
            return

        self.gif_progress_dialog = QProgressDialog("Saving GIF animation...", "Cancel", 0, 0, window)
        self.gif_progress_dialog.setWindowTitle("Processing")
        self.gif_progress_dialog.setStyleSheet(PROGRESS_DIALOG_STYLE)
        self.gif_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.gif_progress_dialog.setMinimumDuration(0)
        self.gif_progress_dialog.setCancelButton(None)
        self.gif_progress_dialog.show()

        duration_sec = duration_ms / 1000.0
        self.gif_worker = GifSaverWorker(images_to_save, file_path, duration_sec, window.label_manager, target_type)
        self.gif_worker.finished_signal.connect(
            lambda success, msg: self.on_gif_saved(success, msg, duration_ms)
        )
        self.gif_worker.start()

    def on_gif_saved(self, success: bool, message: str, duration_ms: int) -> None:
        window = self.window

        if self.gif_progress_dialog:
            self.gif_progress_dialog.close()
            self.gif_progress_dialog = None

        self.gif_worker = None

        if success:
            show_success_box(
                window,
                "Success",
                "GIF animation saved successfully!",
                f"{message}\nFrame duration: {duration_ms}ms",
            )
        else:
            show_message_box(
                window,
                "Error",
                "Failed to save as GIF animation:",
                f"Error: {message}",
                QMessageBox.Icon.Critical,
            )

    def save_processed_input_stack(self) -> None:
        window = self.window
        if not window.raw_images:
            show_warning_box(window, "No Images", "Please load images first to save the stack.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_foldername = f"Processed_Input_Stack_{timestamp}"
        folder_path = QFileDialog.getExistingDirectory(
            window,
            "Select Folder to Save Processed Input Stack",
            default_foldername,
            QFileDialog.Option.ShowDirsOnly,
        )

        if not folder_path:
            return

        try:
            saved_count = 0
            for index, image in enumerate(window.raw_images):
                image_to_save = window.label_manager.prepare_bgr_image("input", image, index)
                if index < len(window.image_filenames):
                    filename = window.image_filenames[index]
                else:
                    filename = f"processed_{index + 1:04d}{DEFAULT_EXPORT_EXTENSION}"
                file_path = os.path.join(folder_path, filename)
                file_path = self.normalize_export_path(file_path)
                if cv2.imwrite(file_path, image_to_save):
                    saved_count += 1

            show_success_box(
                window,
                "Success",
                "Processed input stack saved successfully!",
                f"Successfully saved {saved_count}/{len(window.raw_images)} images to:\n{folder_path}",
            )
        except cv2.error as exc:
            show_error_box(
                window,
                "Error",
                "Failed to save stack:",
                f"OpenCV Error: {str(exc)}\n\nPlease check the file path and permissions.",
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_error_box(
                window,
                "Error",
                "Failed to save stack:",
                f"Unexpected error: {str(exc)}",
            )
