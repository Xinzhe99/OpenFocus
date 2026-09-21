import traceback
from typing import Any, List, Optional

import cv2

from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog

from utils import show_custom_message_box, show_message_box, show_warning_box
from core.multi_focus_fusion import is_stackmffv4_available
from core.workers import RenderWorker
from dialogs import ROIRenderOptionsDialog  # Import the new dialog
from locales import trans

PREVIEW_MAX_SIDE = 1200


def _downscale_for_preview(img):
    """Scale a frame so its longest side is at most PREVIEW_MAX_SIDE."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= PREVIEW_MAX_SIDE:
        return img
    scale = PREVIEW_MAX_SIDE / float(longest)
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


class RenderManager:
    """Encapsulates render pipeline orchestration for the main window."""

    def __init__(self, window: Any):
        self.window = window
        self.worker: Optional[RenderWorker] = None

    def _restore_ui_controls(self) -> None:
        """Re-enable every control disabled by start_render (safe to call anywhere)."""
        window = self.window
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

    def start_render(self) -> None:
        window = self.window

        if not window.raw_images or len(window.raw_images) < 2:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_render_need_images_text"))
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
                trans.t("msg_stackmff_unavailable_title"),
                trans.t("msg_stackmff_unavailable_text"),
            )
            window.rb_d.setChecked(False)
            self._restore_ui_controls()
            return

        # Handle ROI options - check if ROI mode is active and we have aligned images
        roi_rect = None
        roi_mode = "crop"
        roi_base_index = 0
        use_roi_aligned_images = False
        
        # 检查ROI模式是否激活，并从右侧面板获取ROI区域
        if getattr(window, 'roi_mode_active', False) and window.roi_aligned_images:
            # 从右侧结果面板获取ROI区域（因为ROI是在对齐后的图像上选择的）
            roi_rect = window.lbl_result_img.get_roi_rect() if hasattr(window.lbl_result_img, 'get_roi_rect') else None
            if roi_rect is not None:
                use_roi_aligned_images = True
                dialog = ROIRenderOptionsDialog(len(window.roi_aligned_images), window)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    roi_mode = dialog.mode
                    roi_base_index = dialog.base_frame_index
                else:
                    # User cancelled the ROI dialog -> cancel render
                    self._restore_ui_controls()
                    return
        
        # 确定要使用的图像源
        preview_downscale = getattr(window, 'chk_quick_preview', None)
        self._preview_mode = bool(
            preview_downscale is not None
            and preview_downscale.isChecked()
            and not use_roi_aligned_images
        )
        if use_roi_aligned_images:
            # ROI模式下使用已经对齐的图像栈，跳过额外的配准
            source_images = window.roi_aligned_images
            # 在ROI模式下，图像已经对齐，不需要再次配准
            effective_need_align_homography = False
            effective_need_align_ecc = False
            # 告诉Worker图像已经对齐
            effective_aligned_images = window.roi_aligned_images
            effective_is_aligned = True
            effective_last_alignment_options = (False, True)  # 表示ECC已完成
        elif self._preview_mode:
            # Quick preview: downscale the stack so the draft render finishes
            # in a fraction of the time; alignment always reruns on the small
            # frames (the full-resolution aligned cache does not apply).
            source_images = [_downscale_for_preview(img) for img in window.raw_images]
            effective_need_align_homography = need_align_homography
            effective_need_align_ecc = need_align_ecc
            effective_aligned_images = source_images
            effective_is_aligned = False
            effective_last_alignment_options = (False, False)
        elif (need_align_homography or need_align_ecc) and not window.is_images_aligned \
                and getattr(window, "align_cache_enabled", True):
            # Try the on-disk registration cache before re-aligning
            from utils import align_cache
            cached = align_cache.load_aligned(
                getattr(window, "current_folder_path", ""),
                window.image_filenames,
                (need_align_homography, need_align_ecc),
                getattr(window, "reg_downscale_width", None),
            )
            if cached is not None and len(cached) == len(window.raw_images):
                print("Registration cache hit — skipping alignment")
                source_images = window.raw_images
                effective_need_align_homography = need_align_homography
                effective_need_align_ecc = need_align_ecc
                effective_aligned_images = cached
                effective_is_aligned = True
                effective_last_alignment_options = (need_align_homography, need_align_ecc)
            else:
                source_images = window.raw_images
                effective_need_align_homography = need_align_homography
                effective_need_align_ecc = need_align_ecc
                effective_aligned_images = window.aligned_images
                effective_is_aligned = window.is_images_aligned
                effective_last_alignment_options = window.last_alignment_options
        else:
            source_images = window.raw_images
            effective_need_align_homography = need_align_homography
            effective_need_align_ecc = need_align_ecc
            effective_aligned_images = window.aligned_images
            effective_is_aligned = window.is_images_aligned
            effective_last_alignment_options = window.last_alignment_options

        self.worker = RenderWorker(
            source_images,
            effective_aligned_images,
            effective_is_aligned,
            effective_last_alignment_options,
            effective_need_align_homography,
            effective_need_align_ecc,
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
            stackmffv4_batch_size=getattr(window, "stackmffv4_batch_size", 2),
            roi_rect=roi_rect,
            roi_mode=roi_mode,
            roi_base_index=roi_base_index,
            use_gpu=getattr(window, "use_gpu", True),
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
        preview = getattr(self, '_preview_mode', False)
        self._preview_mode = False

        try:
            if fusion_result is not None:
                window.fusion_result = fusion_result
                # A quick preview is a throwaway draft: show it, but keep it
                # out of the output history so only full-quality renders land
                # there.
                if not preview:
                    window.registration_results = processed_images

                # 如果是ROI模式下的融合，退出ROI模式（但保留对齐图像供复用）
                if getattr(window, 'roi_mode_active', False):
                    window.roi_mode_active = False
                    # 不清空 roi_aligned_images，保留供下次复用
                    if hasattr(window, 'lbl_result_img'):
                        window.lbl_result_img.roi_mode = False
                        window.lbl_result_img.set_roi_rect(None)
                    # 取消ROI按钮的选中状态
                    if hasattr(window, 'btn_preview_roi'):
                        window.btn_preview_roi.blockSignals(True)
                        window.btn_preview_roi.setChecked(False)
                        window.btn_preview_roi.blockSignals(False)

                window.output_manager.show_fusion_result()
                if not preview:
                    window.output_manager.update_output_list_for_fusion()

                # New output: refresh the wipe B-side history, keep selection
                if getattr(window, "wipe_active", False):
                    window.refresh_wipe_controls()

                window.result_control_bar.setVisible(False)
                window.result_slider.setEnabled(False)
                window.current_result_index = -1
                window.add_label_action.setEnabled(True)

                if preview:
                    print("Preview render completed (not added to output history)")
                else:
                    print("Fusion completed successfully!")

                import logging
                logging.getLogger("openfocus").info(
                    "render finished (preview=%s, method fusion, %.1fs)", preview, fusion_time)
            else:
                if registration_performed:
                    if preview:
                        # Preview draft: show the small aligned frames without
                        # touching the full-resolution output history
                        first = processed_images[0] if processed_images else None
                        if first is not None:
                            rgb_image = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
                            h, w, _ch = rgb_image.shape
                            q_image = QImage(rgb_image.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                            window.lbl_result_img.set_display_pixmap(QPixmap.fromImage(q_image))
                        window.result_control_bar.setVisible(False)
                        print("Preview registration completed (not added to output history)")
                    else:
                        window.fusion_result = None
                        window.registration_results = processed_images

                        window.result_slider.setEnabled(True)
                        window.result_slider.setRange(0, len(window.registration_results) - 1)
                        window.result_control_bar.setVisible(True)

                        window.current_result_index = 0
                        window.update_result_view(0)
                        window.add_label_action.setEnabled(True)

                        # Registration results replaced the output history the
                        # wipe view may be scrubbing through
                        if getattr(window, "wipe_active", False):
                            window.refresh_wipe_controls()

                        print("Registration completed successfully!")
                else:
                    print("No operation selected. Please select registration options or fusion method.")

            # Only cache aligned images if we performed a FULL registration (no ROI cropping)
            # Note: self.worker may be None if error occurred, so we check first
            # Preview renders work on downscaled frames — never cache them as
            # the full-resolution aligned stack.
            worker = self.worker
            if (
                registration_performed and not preview
                and worker is not None and not getattr(worker, 'roi_rect', None)
            ):
                window.aligned_images = processed_images
                window.is_images_aligned = True
                window.last_alignment_options = (
                    worker.need_align_homography,
                    worker.need_align_ecc,
                )

                # Persist the alignment for future renders of the same stack
                if getattr(window, "align_cache_enabled", True) and getattr(
                        window, "current_folder_path", None):
                    from utils import align_cache
                    align_cache.save_aligned(
                        window.current_folder_path,
                        window.image_filenames,
                        processed_images,
                        (worker.need_align_homography, worker.need_align_ecc),
                        getattr(window, "reg_downscale_width", None),
                    )

            total_time = alignment_time + fusion_time

            info_lines = []

            if preview:
                info_lines.append(trans.t("info_preview_quality"))
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
            self._restore_ui_controls()
            self.worker = None

    def on_render_error(self, error_message: str) -> None:
        window = self.window

        self._restore_ui_controls()

        import logging
        logging.getLogger("openfocus").error("render failed: %s", error_message)

        show_message_box(
            window,
            trans.t("msg_error"),
            trans.t("msg_proc_error"),
            error_message,
            QMessageBox.Icon.Critical,
        )

        traceback.print_exc()
        self.worker = None
