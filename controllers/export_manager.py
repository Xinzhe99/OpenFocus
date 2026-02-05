import datetime
import os
from typing import Any, Optional

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog

from dialogs import DurationDialog
from ui.styles import PROGRESS_DIALOG_STYLE
from utils import (
    show_error_box,
    show_message_box,
    show_success_box,
    show_warning_box,
)
from core.workers import GifSaverWorker
from locales import trans

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


def get_imwrite_params(extension: str) -> list:
    """Get OpenCV imwrite parameters for maximum quality based on file extension.

    Args:
        extension: File extension (e.g., '.jpg', '.png', '.tif', '.bmp')

    Returns:
        List of parameter tuples for cv2.imwrite, or empty list if no special params needed
    """
    ext = extension.lower()
    if ext in (".jpg", ".jpeg", ".jpe", ".jfif"):
        # JPG: 100 quality (highest, default is ~95)
        return [cv2.IMWRITE_JPEG_QUALITY, 100]
    elif ext in (".png",):
        # PNG: 0 compression (no compression, default is 3)
        return [cv2.IMWRITE_PNG_COMPRESSION, 0]
    elif ext in (".tif", ".tiff"):
        # TIFF: LZW compression disabled (compression flag 1 = no compression)
        return [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    elif ext in (".bmp",):
        # BMP: No quality parameters needed (always lossless)
        return []
    else:
        return []


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
        elif window.rb_gfg.isChecked():
            fusion_method = "GFGFGF"
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
        elif window.rb_gfg.isChecked():
            fusion_method = "GFGFGF"
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
            title = trans.t("action_save")
        elif window.registration_results:
            index = window.current_result_index if window.current_result_index >= 0 else 0
            result_to_save = window.registration_results[index]
            title = trans.t("action_save")
        else:
            show_warning_box(window, trans.t("msg_no_result_title"), trans.t("msg_no_result_text"))
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
            ext = os.path.splitext(file_path)[1].lower()
            params = get_imwrite_params(ext)
            if cv2.imwrite(file_path, image_to_save, params):
                show_message_box(
                    window,
                    trans.t("msg_success"),
                    trans.t("msg_image_saved_text"),
                    trans.t("msg_image_saved_info").format(path=file_path),
                    QMessageBox.Icon.Information,
                )
            else:
                show_message_box(
                    window,
                    trans.t("msg_save_failed_title"),
                    trans.t("msg_save_failed_text"),
                    trans.t("msg_save_failed_info_write"),
                    QMessageBox.Icon.Critical,
                )
        except cv2.error as exc:
            show_message_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_opencv").format(error=str(exc)),
                QMessageBox.Icon.Critical,
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_save_failed_title"),
                trans.t("msg_save_failed_text"),
                trans.t("msg_save_failed_info_unexpected").format(error=str(exc)),
                QMessageBox.Icon.Critical,
            )

    def save_result_stack(self) -> None:
        window = self.window
        if not window.registration_results:
            show_warning_box(window, trans.t("msg_no_stack_title"), trans.t("msg_no_stack_text"))
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
                ext = os.path.splitext(file_path)[1].lower()
                params = get_imwrite_params(ext)
                if cv2.imwrite(file_path, image_to_save, params):
                    saved_count += 1

            show_message_box(
                window,
                trans.t("msg_success"),
                trans.t("msg_stack_saved_text"),
                trans.t("msg_stack_saved_info").format(
                    saved=saved_count,
                    total=len(window.registration_results),
                    folder=folder_path,
                ),
                QMessageBox.Icon.Information,
            )
        except cv2.error as exc:
            show_message_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_save_stack_failed_text"),
                trans.t("msg_save_stack_opencv_info").format(error=str(exc)),
                QMessageBox.Icon.Critical,
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_save_stack_failed_text"),
                trans.t("msg_save_stack_unexpected_info").format(error=str(exc)),
                QMessageBox.Icon.Critical,
            )

    def save_as_gif(self, target_type: str = "registered") -> None:
        window = self.window

        if target_type == "registered":
            if not window.registration_results:
                show_warning_box(window, trans.t("msg_no_registered_images_title"), trans.t("msg_no_registered_images_text"))
                return
            images_to_save = window.registration_results
        else:
            if not window.raw_images:
                show_warning_box(window, trans.t("msg_no_input_images_title"), trans.t("msg_no_input_images_text"))
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

        self.gif_progress_dialog = QProgressDialog(
            trans.t("msg_gif_saving_text"),
            trans.t("btn_cancel"),
            0,
            0,
            window,
        )
        self.gif_progress_dialog.setWindowTitle(trans.t("msg_gif_processing_title"))
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
                trans.t("msg_success"),
                trans.t("msg_gif_saved_text"),
                trans.t("msg_gif_saved_info").format(message=message, duration=duration_ms),
            )
        else:
            show_message_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_gif_save_failed_text"),
                trans.t("msg_gif_save_failed_info").format(message=message),
                QMessageBox.Icon.Critical,
            )

    def save_processed_input_stack(self) -> None:
        window = self.window
        if not window.raw_images:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_no_images_save_stack_text"))
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
                ext = os.path.splitext(file_path)[1].lower()
                params = get_imwrite_params(ext)
                if cv2.imwrite(file_path, image_to_save, params):
                    saved_count += 1

            show_success_box(
                window,
                trans.t("msg_success"),
                trans.t("msg_processed_stack_saved_text"),
                trans.t("msg_processed_stack_saved_info").format(
                    saved=saved_count,
                    total=len(window.raw_images),
                    folder=folder_path,
                ),
            )
        except cv2.error as exc:
            show_error_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_save_stack_failed_text"),
                trans.t("msg_save_stack_opencv_info").format(error=str(exc)),
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_error_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_save_stack_failed_text"),
                trans.t("msg_save_stack_unexpected_info").format(error=str(exc)),
            )
