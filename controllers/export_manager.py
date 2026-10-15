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
from utils.image_utils import imwrite_auto
from utils import scalebar

ALLOWED_EXPORT_EXTENSION_MAP = {
    ".png": ".png",
    ".jpg": ".jpg",
    ".bmp": ".bmp",
    ".tif": ".tif",
    ".tiff": ".tiff",
    ".webp": ".webp",
}

EXPORT_EXTENSION_ALIASES = {
    ".jpeg": ".jpg",
    ".jpe": ".jpg",
    ".jfif": ".jpg",
    ".jp2": ".jpg",
}

DEFAULT_EXPORT_EXTENSION = ".png"


# Re-exported from utils for backward compatibility
from utils.image_utils import get_imwrite_params  # noqa: E402,F401


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
    _FUSION_FILE_TAGS = {
        "guided_filter": "GuidedFilter",
        "dct": "DCT",
        "dtcwt": "DTCWT",
        "gfgfgf": "GFGFGF",
        "stackmffv4": "StackMFFV4",
        "openfocus_a": "OpenFocusA",
        "openfocus_b": "OpenFocusB",
        "openfocus_c": "OpenFocusC",
    }
    _ALIGN_FILE_TAGS = {
        "homography": "Homography",
        "ecc": "ECC",
        "openfocus": "OpenFocusAlign",
    }

    def _fusion_tag(self) -> str:
        return self._FUSION_FILE_TAGS.get(
            self.window.current_fusion_method(), "None")

    def _align_tag(self) -> str:
        return self._ALIGN_FILE_TAGS.get(
            self.window.current_alignment_method(), "NoAlign")

    def generate_default_filename(self) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return (f"OpenFocus_{timestamp}_{self._fusion_tag()}_"
                f"{self._align_tag()}")

    def selected_output(self):
        """(image, default filename) of the output the user has selected in
        the output list, or None. Saving must follow the selection: Ctrl+S
        used to write the newest render even with an older one selected
        and shown, and Save As proposed a name built from the *current*
        method dropdown rather than the method that made that output."""
        window = self.window
        lst = getattr(window, "output_list", None)
        results = getattr(window, "fusion_results", None) or []
        if lst is None or window.fusion_result is None:
            return None
        row = lst.currentRow()
        if row < 0 or row >= len(results):
            return None
        item = lst.item(row)
        name = item.text() if item is not None else ""
        return results[row], self.filename_from_label(name)

    @staticmethod
    def filename_from_label(label: str) -> str:
        """Output-list label -> safe default filename (compare-all labels
        carry ' — Method' suffixes)."""
        import re as _re
        name = (label or "").replace(" — ", "_").strip()
        name = _re.sub(r'[\\/:*?"<>|\s]+', "_", name)
        return name or "OpenFocus_result"

    def generate_default_foldername(self) -> str:
        window = self.window
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_basename = "OpenFocus_Stack"
        if getattr(window, "current_folder_path", None):
            folder_basename = os.path.basename(window.current_folder_path)

        return f"{folder_basename}_{timestamp}_{self._fusion_tag()}_{self._align_tag()}"

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def save_result(self) -> None:
        window = self.window
        result_to_save = None
        title = ""

        selected = self.selected_output()
        default_filename = None
        if selected is not None:
            result_to_save, default_filename = selected
            title = trans.t("action_save")
        elif window.fusion_result is not None:
            result_to_save = window.fusion_result
            title = trans.t("action_save")
        elif window.registration_results:
            index = window.current_result_index if window.current_result_index >= 0 else 0
            result_to_save = window.registration_results[index]
            title = trans.t("action_save")
        else:
            show_warning_box(window, trans.t("msg_no_result_title"), trans.t("msg_no_result_text"))
            return

        if not default_filename:
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
            if imwrite_auto(file_path, scalebar.burn(image_to_save, scalebar.export_cfg(window)), params):
                # Non-blocking confirmation (errors still get a dialog)
                window.statusBar().showMessage(
                    trans.t("msg_image_saved_info").format(path=file_path).replace("\n", " "),
                    8000)
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
                if imwrite_auto(file_path, scalebar.burn(image_to_save, scalebar.export_cfg(window)), params):
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
        self.gif_worker = GifSaverWorker(images_to_save, file_path, duration_sec, window.label_manager, target_type, scale_cfg=scalebar.export_cfg(window))
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

    def save_stack_as_multipage_tiff(self, target_type: str = "registered") -> None:
        """Save a whole stack as ONE multi-page TIFF file (16-bit preserved)."""
        window = self.window
        if target_type == "registered":
            if not window.registration_results:
                show_warning_box(window, trans.t("msg_no_registered_images_title"), trans.t("msg_no_registered_images_text"))
                return
            images = window.registration_results
            label_target = "registered"
            default_name = self.generate_default_filename() + ".tif"
        else:
            if not window.raw_images:
                show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_no_images_save_stack_text"))
                return
            images = window.raw_images
            label_target = "input"
            default_name = f"Input_Stack_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"

        from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir
        file_path, _ = QFileDialog.getSaveFileName(
            window, trans.t("action_save_multipage_tiff"),
            os.path.join(get_last_dialog_dir() or os.path.expanduser("~"), default_name),
            "Multi-page TIFF (*.tif *.tiff)")
        if not file_path:
            return
        if os.path.splitext(file_path)[1].lower() not in ('.tif', '.tiff'):
            file_path += '.tif'
        set_last_dialog_dir(file_path)

        try:
            cfg = scalebar.export_cfg(window)
            labeled = [scalebar.burn(
                           window.label_manager.prepare_bgr_image(label_target, img, i), cfg)
                       for i, img in enumerate(images)]
            from utils.image_utils import imwrite_multi_tiff
            if imwrite_multi_tiff(file_path, labeled):
                show_success_box(
                    window, trans.t("msg_success"),
                    trans.t("msg_multipage_saved_text"),
                    trans.t("msg_multipage_saved_info").format(
                        count=len(labeled), path=file_path))
            else:
                show_error_box(window, trans.t("msg_save_failed_title"),
                               trans.t("msg_save_failed_text"),
                               trans.t("msg_save_failed_info_write"))
        except Exception as exc:  # pylint: disable=broad-except
            show_error_box(window, trans.t("msg_error"),
                           trans.t("msg_save_failed_text"), f"Error: {exc}")

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
                if imwrite_auto(file_path, scalebar.burn(image_to_save, scalebar.export_cfg(window)), params):
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
