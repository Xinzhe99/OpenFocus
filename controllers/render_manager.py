import traceback
from typing import Any, List, Optional

from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog

from utils import show_custom_message_box, show_message_box, show_warning_box
from multi_focus_fusion import is_stackmffv4_available
from workers import RenderWorker
from roi_dialog import ROIRenderOptionsDialog # Import the new dialog
from locales import trans


class RenderManager:
    """Encapsulates render pipeline orchestration for the main window."""

    def __init__(self, window: Any):
        self.window = window
        self.worker: Optional[RenderWorker] = None

    def start_render(self) -> None:
        window = self.window

        if not window.raw_images or len(window.raw_images) < 2:
            show_warning_box(window, "No Images", "Please load at least 2 images before rendering.")
            return

        window.btn_render.setEnabled(False)
        window.btn_render.setText(trans.t('btn_render_processing'))
        QApplication.processEvents()

        # 禁用在处理过程中不应被修改的 UI 控件
        try:
            window.slider_smooth.setEnabled(False)
        except Exception:
            pass
        try:
            window.rb_a.setEnabled(False)
            window.rb_b.setEnabled(False)
            window.rb_c.setEnabled(False)
            window.rb_gfg.setEnabled(False)
            window.rb_d.setEnabled(False)
        except Exception:
            pass
        try:
            window.cb_align_homography.setEnabled(False)
            window.cb_align_ecc.setEnabled(False)
        except Exception:
            pass
        try:
            window.btn_reset.setEnabled(False)
        except Exception:
            pass

        need_align_homography = window.cb_align_homography.isChecked()
        need_align_ecc = window.cb_align_ecc.isChecked()

        need_fusion = (
            window.rb_a.isChecked()
            or window.rb_b.isChecked()
            or window.rb_c.isChecked()
            or window.rb_gfg.isChecked()
            or window.rb_d.isChecked()
        )

        kernel_slider_value = window.slider_smooth.value()
        if kernel_slider_value <= 0:
            kernel_slider_value = 1
        if kernel_slider_value % 2 == 0:
            kernel_slider_value = max(1, kernel_slider_value - 1)

        if window.rb_d.isChecked() and not is_stackmffv4_available():
            show_warning_box(
                window,
                "StackMFF-V4 Unavailable",
                "StackMFF-V4 requires torch + torchvision. Please install them or choose another fusion method.",
            )
            window.rb_d.setChecked(False)
            window.btn_render.setEnabled(True)
            window.btn_render.setText("▶ Start Render")
            return

        # Handle ROI options
        roi_rect = window.lbl_source_img.get_roi_rect() if window.btn_preview_roi.isChecked() else None
        roi_mode = "crop"
        roi_base_index = 0

        if roi_rect is not None:
             dialog = ROIRenderOptionsDialog(len(window.raw_images), window)
             if dialog.exec() == QDialog.DialogCode.Accepted:
                 roi_mode = dialog.mode
                 roi_base_index = dialog.base_frame_index
             else:
                 # User cancelled the ROI dialog -> cancel render? 
                 # Or just render crop? Let's cancel to be safe.
                 window.btn_render.setEnabled(True)
                 window.btn_render.setText(trans.t('btn_render'))
                 return

        self.worker = RenderWorker(
            window.raw_images,
            window.aligned_images,
            window.is_images_aligned,
            window.last_alignment_options,
            need_align_homography,
            need_align_ecc,
            need_fusion,
            window.rb_a.isChecked(),
            window.rb_b.isChecked(),
            window.rb_c.isChecked(),
            window.rb_gfg.isChecked(),
            window.rb_d.isChecked(),
            kernel_slider_value,
            tile_enabled=getattr(window, "tile_enabled", None),
            tile_block_size=getattr(window, "tile_block_size", None),
            tile_overlap=getattr(window, "tile_overlap", None),
            tile_threshold=getattr(window, "tile_threshold", None),
            reg_downscale_width=getattr(window, "reg_downscale_width", None),
            thread_count=getattr(window, "thread_count", 4),
            roi_rect=roi_rect,
            roi_mode=roi_mode,
            roi_base_index=roi_base_index,
        )

        self.worker.finished_signal.connect(self.on_render_finished)
        self.worker.error_signal.connect(self.on_render_error)
        self.worker.start()

    def on_render_finished(
        self,
        processed_images: List[Any],
        fusion_result: Optional[Any],
        registration_performed: bool,
        alignment_time: float,
        fusion_time: float,
        device_name: str,
    ) -> None:
        window = self.window

        try:
            if fusion_result is not None:
                window.fusion_result = fusion_result
                window.registration_results = processed_images

                window.output_manager.show_fusion_result()
                window.output_manager.update_output_list_for_fusion()

                window.result_control_bar.setVisible(False)
                window.result_slider.setEnabled(False)
                window.current_result_index = -1
                window.add_label_action.setEnabled(True)

                print("Fusion completed successfully!")
            else:
                if registration_performed:
                    window.fusion_result = None
                    window.registration_results = processed_images

                    window.result_slider.setEnabled(True)
                    window.result_slider.setRange(0, len(window.registration_results) - 1)
                    window.result_control_bar.setVisible(True)

                    window.current_result_index = 0
                    window.update_result_view(0)
                    window.add_label_action.setEnabled(True)

                    print("Registration completed successfully!")
                else:
                    print("No operation selected. Please select registration options or fusion method.")

            # Only cache aligned images if we performed a FULL registration (no ROI cropping)
            if registration_performed and not getattr(self.worker, 'roi_rect', None):
                window.aligned_images = processed_images
                window.is_images_aligned = True
                window.last_alignment_options = (
                    self.worker.need_align_homography,
                    self.worker.need_align_ecc,
                )

            total_time = alignment_time + fusion_time

            info_lines = []

            if registration_performed:
                align_methods = []
                if window.cb_align_homography.isChecked():
                    align_methods.append(trans.t("check_align_homography"))
                if window.cb_align_ecc.isChecked():
                    align_methods.append(trans.t("check_align_ecc"))
                align_method_str = ", ".join(align_methods) if align_methods else trans.t("val_none")
                info_lines.append(trans.t("info_align_method").format(align_method_str))
                info_lines.append(trans.t("info_align_time").format(alignment_time))
            else:
                info_lines.append(trans.t("info_align_none_cached"))
                info_lines.append(trans.t("info_align_time").format(0.0))

            if (
                window.rb_a.isChecked()
                or window.rb_b.isChecked()
                or window.rb_c.isChecked()
                or window.rb_gfg.isChecked()
                or window.rb_d.isChecked()
            ):
                if window.rb_a.isChecked():
                    method_name = trans.t("radio_guided_filter")
                elif window.rb_b.isChecked():
                    method_name = trans.t("radio_dct")
                elif window.rb_c.isChecked():
                    method_name = trans.t("radio_dtcwt")
                elif window.rb_gfg.isChecked():
                    method_name = trans.t("radio_gfg")
                elif window.rb_d.isChecked():
                    method_name = trans.t("radio_stackmff")
                else:
                    method_name = trans.t("radio_guided_filter")

                info_lines.append(trans.t("info_fusion_method").format(method_name))
                info_lines.append(trans.t("info_fusion_time").format(fusion_time))
                info_lines.append(trans.t("info_proc_unit").format(device_name))
            else:
                info_lines.append(trans.t("info_fusion_none"))
                info_lines.append(trans.t("info_fusion_time").format(0.0))

            info_lines.append(trans.t("info_total_time").format(total_time))

            show_custom_message_box(
                window,
                trans.t("dialog_completed_title"),
                trans.t("dialog_completed_msg"),
                "\n".join(info_lines),
                QMessageBox.Icon.Information,
            )

        except Exception as exc:
            show_message_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_proc_error"),
                str(exc),
                QMessageBox.Icon.Critical,
            )
            traceback.print_exc()

        finally:
            # 恢复 UI 控件
            try:
                window.slider_smooth.setEnabled(True)
            except Exception:
                pass
            try:
                window.rb_a.setEnabled(True)
                window.rb_b.setEnabled(True)
                window.rb_c.setEnabled(True)
                window.rb_gfg.setEnabled(True)
                window.rb_d.setEnabled(True)
            except Exception:
                pass
            try:
                window.cb_align_homography.setEnabled(True)
                window.cb_align_ecc.setEnabled(True)
            except Exception:
                pass
            try:
                window.btn_reset.setEnabled(True)
            except Exception:
                pass

            window.btn_render.setEnabled(True)
            window.btn_render.setText(trans.t('btn_render'))
            self.worker = None

    def on_render_error(self, error_message: str) -> None:
        window = self.window

        window.btn_render.setEnabled(True)
        window.btn_render.setText(trans.t('btn_render'))

        show_message_box(
            window,
            trans.t("msg_error"),
            trans.t("msg_proc_error"),
            error_message,
            QMessageBox.Icon.Critical,
        )

        traceback.print_exc()
        self.worker = None
