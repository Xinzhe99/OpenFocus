import traceback
from typing import Any, List, Optional

from PyQt6.QtWidgets import QApplication, QMessageBox

from utils import show_custom_message_box, show_message_box, show_warning_box
from multi_focus_fusion import is_stackmffv4_available
from workers import RenderWorker


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
        window.btn_render.setText("⏳ Processing...")
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
        use_gpu: bool,
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

            if registration_performed:
                window.aligned_images = processed_images
                window.is_images_aligned = True
                window.last_alignment_options = (
                    window.cb_align_homography.isChecked(),
                    window.cb_align_ecc.isChecked(),
                )

            total_time = alignment_time + fusion_time

            info_lines = []

            if registration_performed:
                align_methods = []
                if window.cb_align_homography.isChecked():
                    align_methods.append("Homography")
                if window.cb_align_ecc.isChecked():
                    align_methods.append("ECC (Enabled)")
                align_method_str = ", ".join(align_methods) if align_methods else "None"
                info_lines.append(f"Alignment Method: {align_method_str}")
                info_lines.append(f"Alignment Time: {alignment_time:.2f}s")
            else:
                info_lines.append("Alignment Method: None (using cached results)")
                info_lines.append("Alignment Time: 0.00s")

            if (
                window.rb_a.isChecked()
                or window.rb_b.isChecked()
                or window.rb_c.isChecked()
                or window.rb_gfg.isChecked()
                or window.rb_d.isChecked()
            ):
                if window.rb_a.isChecked():
                    method_name = "Guided Filter"
                elif window.rb_b.isChecked():
                    method_name = "DCT"
                elif window.rb_c.isChecked():
                    method_name = "DTCWT"
                elif window.rb_gfg.isChecked():
                    method_name = "GFG-FGF"
                elif window.rb_d.isChecked():
                    method_name = "StackMFF-V4"
                else:
                    method_name = "Guided Filter"

                info_lines.append(f"Fusion Method: {method_name}")
                info_lines.append(f"Fusion Time: {fusion_time:.2f}s")
                info_lines.append(f"Processing Unit: {'GPU' if use_gpu else 'CPU'}")
            else:
                info_lines.append("Fusion Method: None")
                info_lines.append("Fusion Time: 0.00s")

            info_lines.append(f"Total Time: {total_time:.2f}s")

            show_custom_message_box(
                window,
                "Processing Completed",
                "Processing completed successfully!",
                "\n".join(info_lines),
                QMessageBox.Icon.Information,
            )

        except Exception as exc:
            show_message_box(
                window,
                "Error",
                "An error occurred during processing:",
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
            window.btn_render.setText("▶ Start Render")
            self.worker = None

    def on_render_error(self, error_message: str) -> None:
        window = self.window

        window.btn_render.setEnabled(True)
        window.btn_render.setText("▶ Start Render")

        show_message_box(
            window,
            "Error",
            "An error occurred during processing:",
            error_message,
            QMessageBox.Icon.Critical,
        )

        traceback.print_exc()
        self.worker = None
