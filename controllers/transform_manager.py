import cv2
from typing import Any

from PyQt6.QtGui import QFont

from dialogs import DownsampleDialog
from utils import show_error_box, show_success_box, show_warning_box, get_default_font
from locales import trans


class TransformManager:
    """Handles transformations applied to the currently loaded image stack."""

    def __init__(self, window: Any) -> None:
        self.window = window

    def rotate_stack(self, rotation_code: int) -> None:
        """Rotate every image in the current stack."""
        window = self.window
        if not window.raw_images:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_no_images_rotate"))
            return

        window.raw_images = [cv2.rotate(img, rotation_code) for img in window.raw_images]

        if getattr(window, "base_images", None):
            window.base_images = [cv2.rotate(img, rotation_code) for img in window.base_images]

        self._invalidate_processing_results(clear_output_view=True, preserve_outputs=True)
        self.reload_image_stack()

    def flip_stack(self, flip_code: int) -> None:
        """Flip every image in the current stack."""
        window = self.window
        if not window.raw_images:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_no_images_flip"))
            return

        window.raw_images = [cv2.flip(img, flip_code) for img in window.raw_images]

        if getattr(window, "base_images", None):
            window.base_images = [cv2.flip(img, flip_code) for img in window.base_images]

        self._invalidate_processing_results(clear_output_view=True, preserve_outputs=True)
        self.reload_image_stack()

    def resize_all_images(self) -> None:
        """Resize working images according to the down-sample dialog."""
        window = self.window
        if not window.raw_images:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_no_images_resize"))
            return

        current_scale = getattr(window, "current_scale_factor", 1.0)
        dialog = DownsampleDialog(window, initial_scale=current_scale)
        if not dialog.exec():
            return

        new_scale = dialog.get_scale_factor()
        if abs(new_scale - current_scale) < 0.001:
            return

        try:
            if new_scale == 1.0:
                window.raw_images = [img.copy() for img in window.base_images]
            else:
                new_width = int(window.base_images[0].shape[1] * new_scale)
                new_height = int(window.base_images[0].shape[0] * new_scale)
                resized = []
                for img in window.raw_images:
                    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    resized.append(resized_img)
                window.raw_images = resized

            window.current_scale_factor = new_scale

            self._invalidate_processing_results(clear_output_view=True, preserve_outputs=True)
            self.reload_image_stack()

            show_success_box(
                window,
                trans.t("msg_success"),
                trans.t("msg_resize_success_text").format(percent=int(new_scale * 100)),
            )
        except Exception as exc:  # pylint: disable=broad-except
            show_error_box(
                window,
                trans.t("msg_resize_error_title"),
                trans.t("msg_resize_error_text"),
                f"Error: {str(exc)}",
            )

    def reload_image_stack(self, initial_index: int | None = 0) -> None:
        """Refresh pixmaps and list widgets after a stack mutation."""
        window = self.window
        try:
            if not window.raw_images:
                window.stack_images = []
                window.source_manager.update_file_list([], [])
                window.source_manager.update_slider_range()
                window.stack_slider.blockSignals(True)
                window.stack_slider.setValue(0)
                window.stack_slider.blockSignals(False)
                window.source_control_bar.setVisible(False)
                window.lbl_stack_info.setText("-- / --")
                window.add_label_action.setEnabled(False)
                window.resize_action.setEnabled(False)
                window.lbl_source_img.clear()
                window.lbl_source_img.setText(trans.t('drag_hint'))
                window.lbl_source_img.setFont(QFont(get_default_font(), 16))
                
                if hasattr(window, "update_loaded_status"):
                    window.update_loaded_status()
                return

            pixmaps = window.image_loader.create_pixmaps(window.raw_images, max_size=None)
            thumbnails = window.image_loader.create_thumbnails(window.raw_images, thumb_size=40)

            window.stack_images = pixmaps
            window.source_manager.update_file_list(window.image_filenames, thumbnails)
            window.source_manager.update_slider_range()

            if window.stack_images:
                window.source_control_bar.setVisible(True)
                window.add_label_action.setEnabled(True)
                window.resize_action.setEnabled(True)

                target_index = 0 if initial_index is None else int(initial_index)
                target_index = max(0, min(target_index, len(window.stack_images) - 1))

                window.update_source_view(target_index)
                window.stack_slider.blockSignals(True)
                window.stack_slider.setValue(target_index)
                window.stack_slider.blockSignals(False)
                window.file_list.setCurrentRow(target_index)
                
            if hasattr(window, "update_loaded_status"):
                window.update_loaded_status()

        except Exception as exc:  # pylint: disable=broad-except
            show_error_box(
                window,
                trans.t("msg_reload_error_title"),
                trans.t("msg_reload_error_text"),
                f"Error: {str(exc)}",
            )

    def invalidate_processing_results(self, clear_output_view: bool = False, preserve_outputs: bool = False) -> None:
        """Expose processing reset so other controllers can reuse it."""
        self._invalidate_processing_results(clear_output_view, preserve_outputs)

    def _invalidate_processing_results(self, clear_output_view: bool = False, preserve_outputs: bool = False) -> None:
        """Clear cached processing results and reset related UI elements."""
        window = self.window

        window.aligned_images = []
        window.is_images_aligned = False
        window.last_alignment_options = None
        
        # 清空ROI对齐缓存
        window.roi_aligned_images = []
        window.roi_aligned_raw_count = 0
        window.roi_mode_active = False

        window.registration_results = []
        window.current_result_index = -1

        window.add_label_action.setEnabled(False)
        window.resize_action.setEnabled(False)

        window.result_slider.blockSignals(True)
        window.result_slider.setRange(0, 0)
        window.result_slider.setValue(0)
        window.result_slider.blockSignals(False)
        window.result_slider.setEnabled(False)
        window.result_control_bar.setVisible(False)
        window.lbl_result_info.setText("-- / --")

        if preserve_outputs:
            if window.fusion_results:
                window.fusion_result = window.fusion_results[0]
            if hasattr(window, "output_manager"):
                window.output_manager.update_output_count()
        else:
            window.fusion_result = None
            window.fusion_results = []

            if hasattr(window, "output_list"):
                window.output_list.clear()
            if hasattr(window, "output_manager"):
                window.output_manager.update_output_count()

        if clear_output_view:
            window.lbl_result_img.clear()
