import traceback
from typing import Any, List, Optional

import cv2

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog

from utils import show_message_box, show_warning_box
from utils.image_utils import to_display_uint8
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

    ALGORITHM_DISPLAY = {
        "guided_filter": "Guided Filter",
        "dct": "DCT",
        "dtcwt": "DTCWT",
        "gfgfgf": "GFG-FGF",
        "openfocus_a": "OpenFocus A",
        "openfocus_b": "OpenFocus B",
        "openfocus_c": "OpenFocus C",
        "stackmffv4": "AI",
    }
    # UI 展示（进度/输出列表/信息行）走 i18n 标签；ALGORITHM_DISPLAY 仅作
    # 非代码兜底与文件名无关
    ALGO_TRANS_KEY = {
        "guided_filter": "radio_guided_filter",
        "dct": "radio_dct",
        "dtcwt": "radio_dtcwt",
        "gfgfgf": "radio_gfg",
        "stackmffv4": "radio_stackmff",
        "openfocus_a": "radio_openfocus_a",
        "openfocus_b": "radio_openfocus_b",
        "openfocus_c": "radio_openfocus_c",
        "none": "radio_no_fusion",
    }

    @classmethod
    def algo_display_name(cls, code: str) -> str:
        key = cls.ALGO_TRANS_KEY.get(code)
        if key:
            return trans.t(key)
        return cls.ALGORITHM_DISPLAY.get(code, code)

    def __init__(self, window: Any):
        self.window = window
        self.worker: Optional[RenderWorker] = None
        self._compare_mode = False
        self._compare_queue: List[str] = []
        self._compare_total = 0
        self._compare_current: Optional[str] = None
        self._preview_mode = False
        # What the *running* render was launched with; see start_render().
        self._render_context: Optional[dict] = None

    def _abort_compare_run(self) -> None:
        """Abort a chained comparison whose next render failed to start."""
        if getattr(self, '_compare_mode', False):
            self._compare_mode = False
            self._compare_queue = []
            self._restore_ui_controls()

    def _restore_ui_controls(self) -> None:
        """Re-enable every control disabled by start_render (safe to call anywhere)."""
        window = self.window
        if getattr(self, '_compare_mode', False):
            # A comparison queue is still running - keep controls locked and
            # show progress instead.
            window.btn_render.setEnabled(False)
            done = self._compare_total - len(self._compare_queue)
            window.btn_render.setText(
                f"{trans.t('msg_compare_progress').format(i=done + 1, n=self._compare_total, method=trans.t('btn_render_processing'))}")
            return
        try:
            window.slider_smooth.setEnabled(True)
        except Exception:
            pass
        try:
            window.combo_fusion.setEnabled(True)
        except Exception:
            pass
        # StackMFF-V4 availability is owned by the torch probe: blanket-enabling
        # it here re-armed an unavailable backend after every render.
        hook = getattr(window, "refresh_method_availability", None)
        if callable(hook):
            hook()
        try:
            window.combo_align.setEnabled(True)
        except Exception:
            pass
        try:
            window.chk_quick_preview.setEnabled(True)
        except Exception:
            pass
        try:
            window.btn_reset.setEnabled(True)
        except Exception:
            pass
        window.btn_render.setEnabled(True)
        window.btn_render.setText(trans.t('btn_render'))

    def start_compare_all(self) -> None:
        """Render the stack once per available fusion method so the results
        can be compared side by side (output history + Wipe side B).

        Clicking again while a run is queued or rendering cancels it."""
        window = self.window

        if self._compare_mode:
            # Cancel: RenderWorker polls is_cancelled between its stages, and
            # the chained _start_next_compare no-ops on the emptied queue.
            self._compare_mode = False
            queued = bool(self._compare_queue)
            self._compare_queue = []
            self._compare_current = None
            if self.worker is not None and self.worker.isRunning():
                self.worker.is_cancelled = True
            elif queued:
                # Between two queued renders nothing is running, so the UI has
                # to be unlocked here instead of by the cancel callback.
                self._restore_ui_controls()
            window.statusBar().showMessage(trans.t('msg_render_cancelled'), 4000)
            return

        if self.worker is not None and self.worker.isRunning():
            return
        if not window.raw_images or len(window.raw_images) < 2:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_render_need_images_text"))
            return

        # 与融合下拉同序（stackmffv4 在 openfocus 族之前）
        methods = ["guided_filter", "dct", "dtcwt", "gfgfgf", "stackmffv4",
                   "openfocus_a", "openfocus_b", "openfocus_c"]
        if not is_stackmffv4_available():
            methods.remove("stackmffv4")

        self._compare_queue = list(methods)
        self._compare_total = len(methods)
        self._compare_mode = True
        self._start_next_compare()

    def _start_next_compare(self) -> None:
        if not self._compare_queue:
            return
        method = self._compare_queue.pop(0)
        self._compare_current = method
        window = self.window
        window.statusBar().showMessage(
            trans.t("msg_compare_progress").format(
                i=self._compare_total - len(self._compare_queue) + 1,
                n=self._compare_total,
                method=self.algo_display_name(method)),
            5000)
        self.start_render(force_algorithm=method)

    def start_render(self, force_algorithm: str | None = None) -> None:
        window = self.window
        # 重入守卫：本方法中途有多次 processEvents()，期间排队的第二次点击
        # 会重入并把 self.worker 覆盖掉——正在运行的 RenderWorker 失去引用
        # 被 C++ 析构，QThread 直接 qFatal abort。守卫必须在任何
        # processEvents 之前设置。
        if getattr(self, "_render_pending", False):
            return
        self._render_pending = True
        try:
            self._start_render_inner(force_algorithm)
        finally:
            self._render_pending = False

    def _start_render_inner(self, force_algorithm: str | None = None) -> None:
        window = self.window
        # A previous render may have aborted without clearing this; it decides
        # whether the result is treated as a throwaway draft.
        self._preview_mode = False

        timer = getattr(self, "_compare_chain_timer", None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
            self._compare_chain_timer = None

        # A render is already running: this click cancels it
        if self.worker is not None and self.worker.isRunning():
            self.worker.is_cancelled = True
            window.btn_render.setEnabled(False)
            window.btn_render.setText(trans.t('btn_cancel_render'))
            QApplication.processEvents()
            return

        if not window.raw_images or len(window.raw_images) < 2:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_render_need_images_text"))
            self._abort_compare_run()
            return

        if force_algorithm:
            # Compare-all mode drives the next renders programmatically; the
            # button doubles as a cancel affordance and everything else stays
            # locked until the queue finishes.
            window.btn_render.setEnabled(False)
            window.btn_render.setText(trans.t('btn_render_processing'))
            QApplication.processEvents()

        window.btn_render.setText(trans.t('btn_cancel_render'))
        QApplication.processEvents()

        # 禁用在处理过程中不应被修改的 UI 控件
        try:
            window.slider_smooth.setEnabled(False)
        except Exception:
            pass
        try:
            window.combo_fusion.setEnabled(False)
        except Exception:
            pass
        try:
            window.combo_align.setEnabled(False)
        except Exception:
            pass
        try:
            window.btn_reset.setEnabled(False)
        except Exception:
            pass
        try:
            # Quick-preview toggling mid-run fragments a comparison: later
            # methods would render as drafts and vanish from the history.
            window.chk_quick_preview.setEnabled(False)
        except Exception:
            pass

        fusion_method = window.current_fusion_method()
        alignment_method = window.current_alignment_method()
        need_align_homography = alignment_method == "homography"
        need_align_ecc = alignment_method == "ecc"
        need_align_openfocus = alignment_method == "openfocus"

        need_fusion = (
            force_algorithm is not None
            or fusion_method != "none"
        )

        kernel_slider_value = window.slider_smooth.value()
        if kernel_slider_value <= 0:
            kernel_slider_value = 1
        if kernel_slider_value % 2 == 0:
            kernel_slider_value = max(1, kernel_slider_value - 1)

        if force_algorithm is None and fusion_method == "stackmffv4" \
                and not is_stackmffv4_available():
            show_warning_box(
                window,
                trans.t("msg_stackmff_unavailable_title"),
                trans.t("msg_stackmff_unavailable_text"),
            )
            window.set_fusion_method("guided_filter")
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
                    self._abort_compare_run()
                    return
        
        # 确定要使用的图像源
        # Frames the user unticked in the source list are excluded here, so
        # registration/fusion only see the included subset. With nothing
        # excluded, effective_stack() returns the live lists unchanged.
        stack_images, stack_filenames, included_idx = window.effective_stack()
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
            effective_last_alignment_options = (False, True, False)  # 表示ECC已完成
        elif self._preview_mode:
            # Quick preview: downscale the stack so the draft render finishes
            # in a fraction of the time; alignment always reruns on the small
            # frames (the full-resolution aligned cache does not apply).
            source_images = [_downscale_for_preview(img) for img in stack_images]
            effective_need_align_homography = need_align_homography
            effective_need_align_ecc = need_align_ecc
            effective_aligned_images = source_images
            effective_is_aligned = False
            effective_last_alignment_options = (False, False, False)
        elif (need_align_homography or need_align_ecc) \
                and not window.is_images_aligned \
                and getattr(window, "align_cache_enabled", True):
            # Try the on-disk registration cache before re-aligning.
            # (OpenFocus-only alignment is excluded: its result is a cheap
            # translation — re-running costs less than a cache-key mismatch.)
            from utils import align_cache
            cached = align_cache.load_aligned(
                getattr(window, "current_folder_path", ""),
                stack_filenames,
                (need_align_homography, need_align_ecc),
                getattr(window, "reg_downscale_width", None),
                getattr(window, "current_scale_factor", 1.0),
                tuple(stack_images[0].shape[:2]) if stack_images else None,
            )
            if cached is not None and len(cached) == len(stack_images):
                print("Registration cache hit — skipping alignment")
                source_images = stack_images
                effective_need_align_homography = need_align_homography
                effective_need_align_ecc = need_align_ecc
                effective_aligned_images = cached
                effective_is_aligned = True
                effective_last_alignment_options = (
                    need_align_homography, need_align_ecc, need_align_openfocus)
            else:
                source_images = stack_images
                effective_need_align_homography = need_align_homography
                effective_need_align_ecc = need_align_ecc
                effective_aligned_images = window.aligned_images
                effective_is_aligned = window.is_images_aligned
                effective_last_alignment_options = window.last_alignment_options
        else:
            source_images = stack_images
            effective_need_align_homography = need_align_homography
            effective_need_align_ecc = need_align_ecc
            effective_aligned_images = window.aligned_images
            effective_is_aligned = window.is_images_aligned
            effective_last_alignment_options = window.last_alignment_options

        # Aligned stacks (cache or reused) cover the full source stack: filter
        # them down to the included subset so they stay 1:1 with source_images.
        if (len(stack_images) != len(window.raw_images)
                and effective_aligned_images is not None
                and len(effective_aligned_images) == len(window.raw_images)):
            effective_aligned_images = [effective_aligned_images[i] for i in included_idx]
            effective_is_aligned = window.is_images_aligned

        # Record exactly what THIS render was launched with. on_render_finished
        # must not read self.worker (a newer render may already have replaced
        # it) and must not publish an alignment that belongs to a different
        # stack than the one the UI shows (quick preview and ROI mode hand the
        # worker downscaled / cropped copies instead of window.raw_images).
        self._render_context = {
            "source": source_images,
            # Live reference of the FULL stack at launch: with exclusions,
            # "source" is a filtered copy, so identity must be checked against
            # this, never against "source".
            "source_ref": window.raw_images,
            "alignment_options": (
                effective_need_align_homography, effective_need_align_ecc,
                need_align_openfocus),
            "preview": self._preview_mode,
            "roi": roi_rect is not None,
            "filenames": list(stack_filenames),
            "included": list(included_idx),
        }

        self.worker = RenderWorker(
            source_images,
            effective_aligned_images,
            effective_is_aligned,
            effective_last_alignment_options,
            effective_need_align_homography,
            effective_need_align_ecc,
            need_fusion,
            fusion_method == "guided_filter",
            fusion_method == "dct",
            fusion_method == "dtcwt",
            fusion_method == "gfgfgf",
            fusion_method == "stackmffv4",
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
            algorithm_override=force_algorithm,
            need_align_openfocus=need_align_openfocus,
            rb_ofa_checked=fusion_method == "openfocus_a",
            rb_ofb_checked=fusion_method == "openfocus_b",
            rb_ofc_checked=fusion_method == "openfocus_c",
            base_frame="last" if getattr(window, "base_frame_correct", False)
                else "none",
            sort_mode="desc" if getattr(window, "stack_reverse_order", False)
                else "asc",
            depth_smoothing=getattr(window, "ofb_depth_smoothing", None),
        )

        self._progress_first_t = None
        self._progress_last_t = None
        self.worker.finished_signal.connect(self.on_render_finished)
        self.worker.error_signal.connect(self.on_render_error)
        self.worker.progress_signal.connect(self.on_render_progress)
        self.worker.stage_signal.connect(self.on_render_stage)
        window.statusBar().clearMessage()  # stale "loading…" etc.
        self._show_progress_bar(trans.t('render_progress'))
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
        self._hide_progress_bar()
        context = self._render_context
        preview = context["preview"] if context else getattr(self, '_preview_mode', False)
        self._preview_mode = False
        compare = getattr(self, '_compare_mode', False)

        # The stack can change while a render runs (a frame deleted, a new
        # folder loaded, rotate/flip/resize). Publishing that result would show
        # — and export — frames of the previous stack under the new stack's
        # filenames, so drop it instead. The alignment cache has its own,
        # stricter guard below.
        stack_changed = (
            context is not None
            and not context["preview"]
            # ROI renders run on their own pre-aligned stack (its staleness is
            # guarded in main.py's ROI flow), so the stack-identity check
            # below does not apply to them.
            and not context.get("roi")
            and (window.raw_images is not context.get("source_ref")
                 or window.included_indices() != context.get("included")
                 or list(window.image_filenames or []) != context["filenames"])
        )
        if stack_changed:
            import logging as _logging
            message = trans.t('msg_render_stack_changed')
            print("Render finished but the stack changed meanwhile — result discarded.")
            _logging.getLogger("openfocus").warning(
                "render result discarded: stack changed while rendering")
            window.statusBar().showMessage(message, 8000)
            self._restore_ui_controls()
            return

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

                # Tag the newest output entry with the method that made it
                method = getattr(self, '_compare_current', None) if compare else None
                if method and window.output_list.count() > 0:
                    item = window.output_list.item(0)
                    if item is not None:
                        item.setText(
                            f"{item.text()} — {self.algo_display_name(method)}")

                # New output: refresh the wipe B-side history, keep selection
                if getattr(window, "wipe_active", False):
                    window.refresh_wipe_controls()

                window.result_control_bar.setVisible(False)
                window.result_slider.setEnabled(False)
                window.current_result_index = -1
                window.add_label_action.setEnabled(True)

                window.statusBar().clearMessage()

                # Bring attention back: flash the taskbar icon
                QApplication.alert(window)

                # Crash auto-recovery: new outputs refresh the snapshot
                from utils import recovery
                recovery.write_snapshot(window)

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
                            first = to_display_uint8(first)
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

            # Publish the alignment only when it belongs to the stack the UI
            # still shows: quick preview and ROI renders run on downscaled /
            # cropped copies, and the stack may have changed (frame deleted,
            # new load, rotate) while this render was in flight. Caching any of
            # those as the full-resolution aligned stack — on memory or on disk
            # — makes the next render fuse frames that never came from the
            # images it shows.
            source_stack = context["source"] if context else None
            alignment_matches_ui = (
                context is not None
                and not context["preview"]
                and not context["roi"]
                and window.raw_images is source_stack
                and list(window.image_filenames or []) == context["filenames"]
                and len(processed_images) == len(window.raw_images)
            )
            if registration_performed and alignment_matches_ui:
                window.aligned_images = processed_images
                window.is_images_aligned = True
                window.last_alignment_options = context["alignment_options"]

                # Persist the alignment for future renders of the same stack.
                # Keyed on the *source* frames (same signature load_aligned()
                # looks up), not on the rendered ones.
                if getattr(window, "align_cache_enabled", True) and getattr(
                        window, "current_folder_path", None):
                    from utils import align_cache
                    align_cache.save_aligned(
                        window.current_folder_path,
                        window.image_filenames,
                        processed_images,
                        context["alignment_options"],
                        getattr(window, "reg_downscale_width", None),
                        getattr(window, "current_scale_factor", 1.0),
                        tuple(source_stack[0].shape[:2]) if source_stack else None,
                    )

            total_time = alignment_time + fusion_time

            info_lines = []

            if preview:
                info_lines.append(trans.t("info_preview_quality"))
            if registration_performed:
                alignment_method = window.current_alignment_method()
                label_map = {
                    "homography": trans.t("check_align_homography"),
                    "ecc": trans.t("check_align_ecc"),
                    "openfocus": trans.t("check_align_of"),
                }
                align_method_str = label_map.get(alignment_method,
                                                 trans.t("val_none"))
                info_lines.append(trans.t("info_align_method").format(align_method_str))
                info_lines.append(trans.t("info_align_time").format(alignment_time))
            else:
                info_lines.append(trans.t("info_align_none_cached"))
                info_lines.append(trans.t("info_align_time").format(0.0))

            fusion_method = window.current_fusion_method()
            if fusion_method == "none" and not compare:
                # 仅对齐模式：只跑配准，融合时间不适用
                info_lines.append(trans.t("info_fusion_none"))
                info_lines.append(trans.t("info_align_time").format(alignment_time))
            else:
                # 对比渲染时显示的是队列里实际渲染的方法，而非当前下拉选择
                compare_method = getattr(self, "_compare_current", None)
                method_name = self.algo_display_name(
                    compare_method or fusion_method)
                info_lines.append(trans.t("info_fusion_method").format(method_name))
                info_lines.append(trans.t("info_fusion_time").format(fusion_time))
                info_lines.append(trans.t("info_proc_unit").format(device_name))

            info_lines.append(trans.t("info_total_time").format(total_time))

            # Non-blocking completion report: a modal "Completed" box after
            # every render (incl. quick previews) cost a click per iteration
            # and covered the result the user wanted to inspect. The summary
            # goes to the status bar; the full timing sits in the output
            # entry's tooltip.
            if not compare:
                window.statusBar().showMessage(
                    trans.t("dialog_completed_title") + " — "
                    + " · ".join(info_lines), 15000)
            if (fusion_result is not None and not preview
                    and window.output_list.count() > 0):
                newest = window.output_list.item(0)
                if newest is not None:
                    # Long names are cut off in the list: lead with the name
                    newest.setToolTip("\n".join([newest.text(), ""] + info_lines))

        except Exception as exc:
            import logging
            logging.getLogger("openfocus").exception("render failed with exception")
            show_message_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_proc_error"),
                str(exc),
                QMessageBox.Icon.Critical,
            )
            traceback.print_exc()

        finally:
            if compare:
                if self._compare_queue:
                    # Keep controls locked and chain the next method. The
                    # timer is tracked so cancelling the run can stop it —
                    # an untracked one fired into a restarted run and
                    # cancelled its in-flight render.
                    compare_continues = True
                    self._compare_chain_timer = QTimer(self.window)
                    self._compare_chain_timer.setSingleShot(True)
                    self._compare_chain_timer.timeout.connect(self._start_next_compare)
                    self._compare_chain_timer.start(400)
                else:
                    self._compare_mode = False
                    self._restore_ui_controls()
                    window.statusBar().showMessage(
                        trans.t('msg_compare_done').format(count=self._compare_total),
                        10000)
                    QApplication.alert(window)
            else:
                # 恢复 UI 控件
                self._restore_ui_controls()
            self.worker = None

    def _progress_widgets(self):
        """Lazily created status-bar progress bar + stage label.

        Before this a render showed no activity at all unless the fusion
        happened to be tiled; long ECC alignments looked like a hang."""
        bar = getattr(self, "_status_bar_progress", None)
        if bar is None:
            from PyQt6.QtWidgets import QLabel, QProgressBar
            sb = self.window.statusBar()
            label = QLabel()
            bar = QProgressBar()
            bar.setFixedWidth(180)
            bar.setFixedHeight(14)
            bar.setTextVisible(False)
            sb.addPermanentWidget(label)
            sb.addPermanentWidget(bar)
            label.hide()
            bar.hide()
            self._status_bar_progress = bar
            self._status_bar_label = label
        return bar, self._status_bar_label

    def _show_progress_bar(self, text: str) -> None:
        try:
            bar, label = self._progress_widgets()
            bar.setRange(0, 0)  # indeterminate until a stage reports counts
            label.setText(text)
            label.show()
            bar.show()
            self.window.statusBar().setVisible(True)
        except Exception:
            pass

    def _hide_progress_bar(self) -> None:
        bar = getattr(self, "_status_bar_progress", None)
        if bar is not None:
            bar.hide()
            self._status_bar_label.hide()


    def on_render_stage(self, stage: str) -> None:
        text = trans.t('render_stage_align') if stage == "align" else trans.t('render_stage_fuse')
        self._progress_first_t = None
        self._show_progress_bar(text)

    def on_render_progress(self, done: int, total: int) -> None:
        """Surface tiled-fusion progress in the status bar with an ETA."""
        import time as _time
        window = self.window
        bar = getattr(self, "_status_bar_progress", None)
        if bar is not None and total > 0:
            bar.setRange(0, int(total))
            bar.setValue(int(done))
        now = _time.time()
        last = getattr(self, '_progress_last_t', None)
        if last is None or now - last > 1.0:  # throttle to 1 Hz
            self._progress_last_t = now
            pct = int(done * 100 / max(1, total))
            eta = ''
            first = getattr(self, '_progress_first_t', None)
            if first is not None and done > 0:
                remaining = (now - first) / done * (total - done)
                eta = f" · ~{int(remaining) + 1}s"
            else:
                self._progress_first_t = now
            window.statusBar().showMessage(
                f"{trans.t('render_progress')} {pct}% ({done}/{total}){eta}", 5000)

    def on_render_error(self, error_message: str) -> None:
        window = self.window
        # Every exit path clears the draft flag: a failed or cancelled render
        # must not leave the controller in preview mode.
        self._preview_mode = False
        self._render_context = None
        self.worker = None
        self._hide_progress_bar()

        if getattr(self, '_compare_mode', False):
            # A failed method aborts the whole comparison run
            self._compare_mode = False
            self._compare_queue = []

        self._restore_ui_controls()

        # Cancellation is not an error: quiet status message, no dialog
        if error_message.startswith("CANCELLED"):
            window.statusBar().showMessage(trans.t('msg_render_cancelled'), 4000)
            return

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
