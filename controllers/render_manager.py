from typing import Any, List, Optional


from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog

from utils import show_message_box, show_warning_box
from utils.image_utils import cv2_to_pixmap
from core.multi_focus_fusion import is_ai_fusion_available
from core.render_context import build_render_context
from core.workers import RenderWorker
from dialogs import ROIRenderOptionsDialog  # Import the new dialog
from locales import trans
from core.algorithms import fusion_algorithms, get_fusion, get_registration, alignment_steps



class RenderManager:
    """Encapsulates render pipeline orchestration for the main window."""

    @classmethod
    def algo_display_name(cls, code: str) -> str:
        return trans.t("radio_no_fusion" if code == "none" else get_fusion(code).label_key)

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

    def _alignment_key(self, included, method=None):
        window = self.window
        method = method or window.current_fusion_method()
        getter = getattr(window, "brightness_option_for", None)
        brightness = (bool(getter("exposure_match", method)) if callable(getter)
                      else bool(getattr(window, "exposure_match", True)))
        alignment = window.current_alignment_method()
        parameters = window.registration_parameters_for(alignment)
        return (getattr(window, "source_revision", 0), tuple(included),
                tuple(id(window.raw_images[i]) for i in included),
                alignment_steps(alignment), tuple(sorted(parameters.items())),
                getattr(window, "reg_downscale_width", None), brightness,
                bool(getattr(window, "stack_reverse_order", False)))

    def _processing_parameters(self, method):
        window = self.window
        values = {key: getattr(window, key, None) for key in (
            "tile_enabled", "tile_block_size", "tile_overlap", "tile_threshold",
            "reg_downscale_width", "ai_batch_size", "stack_reverse_order", "use_gpu")}
        values.update(window.fusion_parameters_for(method))
        values["registration_parameters"] = window.registration_parameters_for(
            window.current_alignment_method())
        values["exposure_match"] = window.brightness_option_for("exposure_match", method)
        values["base_correct"] = window.brightness_option_for("base_correct", method)
        values["base_frame"] = "last" if values["base_correct"] else "none"
        values["sort_mode"] = "desc" if values["stack_reverse_order"] else "asc"
        return values

    def _disk_cache_eligible(self) -> bool:
        window = self.window
        if (getattr(window, "source_transforms", None)
                or getattr(window, "unrecoverable_source_changes", None)):
            return False
        import os
        paths = getattr(window, "image_source_paths", [])
        names = getattr(window, "image_filenames", [])
        folder = getattr(window, "current_folder_path", "")
        return bool(paths) and len(paths) == len(names) and all(
            os.path.normcase(os.path.abspath(path)) ==
            os.path.normcase(os.path.abspath(os.path.join(folder, name)))
            for path, name in zip(paths, names))

    def _confirm_cpu_ai_render(self, force_algorithm: str | None = None) -> bool:
        """Warn before starting a CPU-only AI render; True means go ahead.

        StackMFF V3 is roughly 30-90x slower on a CPU than the classical
        methods, and on a low-spec machine a large stack can leave the whole
        computer unresponsive for a long time. Users with no accelerator get
        told that once, with a "don't show this again" opt-out, instead of
        waiting minutes with no explanation.

        Compare-all drives methods programmatically and must not stop for a
        modal box mid-run, so it is skipped.
        """
        if force_algorithm is not None:
            return True
        window = self.window
        try:
            algorithm = window.current_fusion_method()
        except Exception:
            return True
        if (algorithm == "none" or not get_fusion(algorithm).neural
                or not getattr(window, "use_gpu", True)):
            # A user who explicitly disabled GPU acceleration has already
            # been told; the setting is the opt-out.
            return True

        from utils.settings_store import (AI_CPU_WARNING_DISMISSED_KEY,
                                          get_settings)
        settings = get_settings()
        if settings.value(AI_CPU_WARNING_DISMISSED_KEY, False, type=bool):
            return True

        from core.multi_focus_fusion import is_ai_fusion_available, torch_device_kind
        if not is_ai_fusion_available():
            return True  # torch missing: the render itself will say so
        if torch_device_kind() not in (None, "N/A"):
            return True  # CUDA or MPS available: nothing to warn about

        from PyQt6.QtWidgets import QCheckBox, QMessageBox
        from utils import style_message_box

        box = QMessageBox(window)
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle(trans.t("ai_cpu_warning_title"))
        box.setText(trans.t("ai_cpu_warning_text").format(
            method=get_fusion(algorithm).display_name))
        dont_show = QCheckBox(trans.t("ai_cpu_warning_dont_show"))
        box.setCheckBox(dont_show)
        start_btn = box.addButton(trans.t("btn_render"),
                                  QMessageBox.ButtonRole.AcceptRole)
        box.addButton(trans.t("btn_cancel_generic"),
                      QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(start_btn)
        box.exec()

        if dont_show.isChecked():
            settings.setValue(AI_CPU_WARNING_DISMISSED_KEY, True)
            settings.sync()
        return box.clickedButton() is start_btn

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
        for name in ("fusion_parameter_panel", "registration_parameter_panel", "combo_fusion"):
            widget = getattr(window, name, None)
            if widget is not None:
                widget.setEnabled(True)
        # StackMFF-V3 availability is owned by the torch probe: blanket-enabling
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
            window.statusBar().showMessage(trans.t(
                'depth_cancelling' if self.worker is not None and self.worker.isRunning()
                else 'msg_render_cancelled'), 0)
            return

        if self.worker is not None and self.worker.isRunning():
            return
        if len(window.effective_stack()[0]) < 2:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_render_need_images_text"))
            return

        methods = [spec.id for spec in fusion_algorithms()
                   if not spec.neural or is_ai_fusion_available()]

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
            window.btn_render.setText(trans.t('depth_cancelling'))
            QApplication.processEvents()
            return

        stack_images, stack_filenames, included_idx = window.effective_stack()
        if len(stack_images) < 2:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_render_need_images_text"))
            self._abort_compare_run()
            return

        if not self._confirm_cpu_ai_render(force_algorithm):
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

        for name in ("fusion_parameter_panel", "registration_parameter_panel",
                     "combo_fusion", "combo_align", "btn_reset"):
            widget = getattr(window, name, None)
            if widget is not None:
                widget.setEnabled(False)
        try:
            # Quick-preview toggling mid-run fragments a comparison: later
            # methods would render as drafts and vanish from the history.
            window.chk_quick_preview.setEnabled(False)
        except Exception:
            pass

        def _brightness(win, option, method):
            """Per-method brightness option (duck-typed hosts may not have it)."""
            getter = getattr(win, "brightness_option_for", None)
            if callable(getter):
                return bool(getter(option, method))
            return bool(getattr(win, "exposure_match", True)) if option == (
                "exposure_match") else bool(
                    getattr(win, "base_frame_correct", False))

        fusion_method = window.current_fusion_method()
        alignment_method = window.current_alignment_method()
        registration_parameters = window.registration_parameters_for(alignment_method)
        requested_steps = alignment_steps(alignment_method)

        need_fusion = (
            force_algorithm is not None
            or fusion_method != "none"
        )


        if force_algorithm is None and fusion_method != "none" and get_fusion(fusion_method).neural \
                and not is_ai_fusion_available():
            show_warning_box(
                window,
                trans.t("msg_stackmff_unavailable_title"),
                trans.t("msg_stackmff_unavailable_text"),
            )
            self._restore_ui_controls()
            return

        alignment_key = self._alignment_key(included_idx, force_algorithm or fusion_method)
        disk_alignment_options = requested_steps
        reusable = bool(window.is_images_aligned
                        and getattr(window, "aligned_images", None)
                        and len(window.aligned_images) == len(stack_images)
                        and getattr(window, "alignment_request_key", None) == alignment_key)
        # Handle ROI options - check if ROI mode is active and we have aligned images
        roi_rect = None
        roi_mode = "crop"
        roi_base_index = 0
        use_roi_aligned_images = False
        
        # 检查ROI模式是否激活，并从右侧面板获取ROI区域
        if getattr(window, 'roi_mode_active', False) and window.roi_aligned_images:
            if (getattr(window, "roi_source_revision", None) !=
                    getattr(window, "source_revision", 0)
                    or list(getattr(window, "roi_included_indices", None) or []) != list(included_idx)
                    or getattr(window, "roi_reg_downscale_width", None) != window.reg_downscale_width
                    or len(window.roi_aligned_images) != len(stack_images)):
                window.transform_manager.reset_roi_selection()
                self._restore_ui_controls()
                self._abort_compare_run()
                window.statusBar().showMessage(trans.t("msg_render_stack_changed"), 8000)
                return
            # 从右侧结果面板获取ROI区域（因为ROI是在对齐后的图像上选择的）
            roi_rect = window.lbl_result_img.get_roi_rect() if hasattr(window.lbl_result_img, 'get_roi_rect') else None
            if roi_rect is not None:
                use_roi_aligned_images = True
                names = list(stack_filenames)
                roi = (window.lbl_result_img.get_roi_rect()
                       if hasattr(window.lbl_result_img, "get_roi_rect")
                       else None)
                roi_size = (int(roi.width()), int(roi.height())) if roi else None
                base = (window.roi_aligned_images[0]
                        if getattr(window, "roi_aligned_images", None) else None)
                output_size = (base.shape[1], base.shape[0]) if base is not None else None
                dialog = ROIRenderOptionsDialog(
                    len(window.roi_aligned_images), window,
                    frame_names=names, roi_size=roi_size,
                    output_size=output_size, frames=window.roi_aligned_images,
                    frame_indices=included_idx)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    roi_mode = dialog.mode
                    roi_base_index = dialog.base_frame_index
                else:
                    # User cancelled the ROI dialog -> cancel render
                    self._restore_ui_controls()
                    self._abort_compare_run()
                    return
        
        preview_downscale = getattr(window, 'chk_quick_preview', None)
        # Compare-all is a *producing* operation: the whole point is to fill the
        # output history with one result per method so they can be compared.
        # Honouring the quick-preview checkbox there rendered eight throwaway
        # drafts, so the run completed with nothing added to the history - the
        # user saw a fusion run and no output, with only the last draft left on
        # the canvas. A draft per method is also unusable for comparison: each
        # is downscaled and none survives.
        compare = bool(getattr(self, '_compare_mode', False))
        self._preview_mode = bool(
            preview_downscale is not None
            and preview_downscale.isChecked()
            and not use_roi_aligned_images
            and not compare
        )
        # None = this render neither reads nor writes the alignment cache.
        disk_cache_request = None
        if use_roi_aligned_images:
            # ROI模式下使用已经对齐的图像栈，跳过额外的配准
            source_images = window.roi_aligned_images
            effective_alignment = "none"
            effective_aligned_images = window.roi_aligned_images
            effective_is_aligned = True
            effective_last_alignment_options = (("ecc",), {})
        elif self._preview_mode:
            # Quick preview: downscale the stack so the draft render finishes
            # in a fraction of the time.
            source_images = stack_images
            # Registration is usually the most expensive stage and it does not
            # get cheaper in proportion: on a 1344x1021 stack with ECC the
            # fusion itself is 0.2-0.5 s while ECC takes seconds, so forcing a
            # fresh alignment for every draft made "quick preview" slower than
            # the full render. When the current stack is already aligned, reuse
            # that alignment (downscaled to the draft size) instead of rerunning
            # it; the alignment is a global transform, so scaling it to the
            # preview grid is exact.
            # Reuse requires the exact frame selection and registration parameters.
            if reusable:
                effective_alignment = "none"
                effective_aligned_images = window.aligned_images
                source_images = effective_aligned_images
                effective_is_aligned = True
                effective_last_alignment_options = (requested_steps, registration_parameters)
            else:
                effective_alignment = alignment_method
                effective_aligned_images = source_images
                effective_is_aligned = False
                effective_last_alignment_options = None
        elif requested_steps \
                and not reusable and self._disk_cache_eligible() \
                and getattr(window, "align_cache_enabled", True):
            # Ask the worker to look in the on-disk registration cache.
            # The lookup itself (stat + decode every cached PNG) runs on the
            # worker thread; the GUI only describes the request.
            source_images = stack_images
            effective_alignment = alignment_method
            effective_aligned_images = []
            effective_is_aligned = False
            effective_last_alignment_options = None
            disk_cache_request = {
                "mode": "load",
                "save": bool(need_fusion),
                "folder": getattr(window, "current_folder_path", ""),
                "filenames": list(stack_filenames),
                "align_options": disk_alignment_options,
                "downscale_width": getattr(window, "reg_downscale_width", None),
                "scale_factor": getattr(window, "current_scale_factor", 1.0),
                "frame_shape": (tuple(stack_images[0].shape[:2])
                                if stack_images else None),
            }
        else:
            source_images = stack_images
            effective_alignment = alignment_method
            effective_aligned_images = window.aligned_images if reusable else []
            effective_is_aligned = reusable
            effective_last_alignment_options = window.last_alignment_options if reusable else None
        if disk_cache_request is not None:
            parameters = self._processing_parameters(force_algorithm or fusion_method)
            disk_cache_request["identity"] = {
                "source_revision": getattr(window, "source_revision", 0),
                "included": list(included_idx),
                "frames": [{"name": name, "shape": list(image.shape),
                            "dtype": image.dtype.str}
                           for name, image in zip(stack_filenames, stack_images)],
                "brightness_reference": "last" if parameters["exposure_match"] else "first",
                "exposure_match": parameters["exposure_match"],
                "registration_backend": "opencv",
                "interpolation": "lanczos4",
                "registration_steps": list(requested_steps),
                "registration_parameters": registration_parameters,
            }
            source_ref = window.raw_images
            revision = getattr(window, "source_revision", 0)
            disk_cache_request["is_current"] = lambda: (
                window.raw_images is source_ref
                and getattr(window, "source_revision", 0) == revision)

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
            "alignment_options": (("ecc",) if use_roi_aligned_images else requested_steps),
            "alignment_identity": (requested_steps, registration_parameters),
            "preview": self._preview_mode,
            "roi": roi_rect is not None,
            "roi_rect": ((float(roi_rect.x()), float(roi_rect.y()),
                          float(roi_rect.width()), float(roi_rect.height()))
                         if roi_rect is not None else None),
            "roi_mode": roi_mode,
            "roi_base_index": roi_base_index,
            "algorithm": force_algorithm or fusion_method,
            "aligned": bool(effective_is_aligned),
            "filenames": list(stack_filenames),
            "included": list(included_idx),
            "source_revision": getattr(window, "source_revision", 0),
            "alignment_key": alignment_key,
            "disk_alignment_options": disk_alignment_options,
            "full_filenames": list(window.image_filenames),
            "parameters": self._processing_parameters(force_algorithm or fusion_method),
        }
        from utils.scalebar import effective_px_um
        frame_ids = getattr(getattr(window, "label_manager", None), "frame_ids", None)
        if callable(frame_ids):
            identities = frame_ids("input")
            self._render_context["frame_ids"] = [identities[i] for i in included_idx]
        self._render_context.update(
            pixel_size_um=effective_px_um(window),
            source_px_um=getattr(window, "source_px_um", None),
            source_scale_factor=getattr(window, "current_scale_factor", 1.0))

        from utils.logging_setup import log_action
        log_action("render_start", method=fusion_method,
                   align=alignment_method, preview=self._preview_mode,
                   frames=len(window.raw_images or []))
        self.worker = RenderWorker(
            source_images,
            effective_aligned_images,
            effective_is_aligned,
            effective_last_alignment_options,
            algorithm=(force_algorithm or fusion_method) if need_fusion else None,
            alignment=effective_alignment,
            fusion_parameters=window.fusion_parameters_for(force_algorithm or fusion_method),
            registration_parameters=registration_parameters if effective_alignment != "none" else {},
            need_fusion=need_fusion,
            tile_enabled=getattr(window, "tile_enabled", None),
            tile_block_size=getattr(window, "tile_block_size", None),
            tile_overlap=getattr(window, "tile_overlap", None),
            tile_threshold=getattr(window, "tile_threshold", None),
            reg_downscale_width=getattr(window, "reg_downscale_width", None),
            thread_count=getattr(window, "thread_count", 4),
            resource_preference=getattr(window, "resource_preference", "balanced"),
            ai_batch_size=getattr(window, "ai_batch_size", 2),
            roi_rect=roi_rect,
            roi_mode=roi_mode,
            roi_base_index=roi_base_index,
            use_gpu=getattr(window, "use_gpu", True),
            base_frame="last" if _brightness(window, "base_correct",
                                             force_algorithm or fusion_method) else "none",
            sort_mode="desc" if getattr(window, "stack_reverse_order", False)
                else "asc",
            exposure_match=_brightness(window, "exposure_match", force_algorithm or fusion_method),
            disk_cache=disk_cache_request,
            preview=self._preview_mode,
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
        if self.worker is not None and getattr(self.worker, "is_cancelled", False):
            self.on_render_error("CANCELLED: render cancelled")
            return
        from utils.logging_setup import log_action
        log_action("render_finished", preview=bool(
            (self._render_context or {}).get("preview")),
            seconds=round(fusion_time, 2))
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
            and (window.raw_images is not context.get("source_ref")
                 or window.included_indices() != context.get("included")
                 or getattr(window, "source_revision", 0) != context.get("source_revision", 0)
                 or list(window.image_filenames or []) != context.get("full_filenames", context["filenames"])
                 or ("alignment_key" in context and self._alignment_key(
                     context["included"], context["algorithm"]) != context["alignment_key"])
                 or ("parameters" in context and self._processing_parameters(
                     context["algorithm"]) != context["parameters"]))
        )
        if stack_changed:
            import logging as _logging
            message = trans.t('msg_render_stack_changed')
            print("Render finished but the stack changed meanwhile — result discarded.")
            _logging.getLogger("openfocus").warning(
                "render result discarded: stack changed while rendering")
            window.statusBar().showMessage(message, 8000)
            # Nothing will chain the next comparison method any more (the
            # queue is only advanced from the normal completion path), so an
            # untouched compare run left the render button disabled forever
            # with a stale "-1/0" progress label.
            if getattr(self, "_compare_mode", False):
                self._compare_mode = False
                self._compare_queue = []
                self._compare_current = None
            self._render_context = None
            self._dispose_worker()
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
                    window.registered_source_indices = list((context or {}).get("included", []))
                    window.registration_frame_ids = list((context or {}).get("frame_ids", []))
                    window.registration_filenames = list((context or {}).get("filenames", []))

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
                new_context = None
                if not preview:
                    try:
                        ctx = context or {}
                        new_context = build_render_context(
                            processed_images,
                            fusion_result,
                            filenames=ctx.get("filenames")
                            or list(window.image_filenames or []),
                            algorithm=((self._compare_current if compare else None)
                                       or ctx.get("algorithm")
                                       or window.current_fusion_method()),
                            alignment_options=ctx.get("alignment_options"),
                            # Only claim registration fill when the worker
                            # actually fused warped frames. A stack that was
                            # aligned earlier but rendered with no alignment
                            # request uses the raw frames.
                            aligned=bool(registration_performed
                                         or (ctx.get("roi")
                                             and ctx.get("aligned"))),
                            roi_rect=ctx.get("roi_rect"),
                            frame_ids=ctx.get("frame_ids"),
                            roi_mode=ctx.get("roi_mode"),
                            source_indices=ctx.get("included"),
                            source_revision=ctx.get("source_revision", 0),
                            parameters=ctx.get("parameters"),
                            pixel_size_um=ctx.get("pixel_size_um"),
                            source_px_um=ctx.get("source_px_um"),
                            source_scale_factor=ctx.get("source_scale_factor", 1.0),
                        )
                        window.last_render_context = new_context
                    except Exception:
                        import logging
                        logging.getLogger("openfocus").exception(
                            "could not record render provenance; "
                            "the output will not be retouchable")
                    window.output_manager.update_output_list_for_fusion(
                        context=new_context)

                # Tag the newest output entry with the method that made it
                method = getattr(self, '_compare_current', None) if compare else None
                if method and window.output_list.count() > 0:
                    item = window.output_list.item(0)
                    if item is not None:
                        item.setText(
                            f"{item.text()} — {self.algo_display_name(method)}")
                        window.output_manager.update_record_label(0, item.text())

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
                            window.lbl_result_img.set_display_pixmap(cv2_to_pixmap(first))
                        window.result_control_bar.setVisible(False)
                        print("Preview registration completed (not added to output history)")
                    else:
                        window.fusion_result = None
                        window.registration_results = processed_images
                        window.registered_source_indices = list((context or {}).get("included", []))
                        window.registration_frame_ids = list((context or {}).get("frame_ids", []))
                        window.registration_filenames = list((context or {}).get("filenames", []))

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

                        # Align-only runs have no fused image, so the output
                        # list used to stay empty and the aligned frames were
                        # unreachable. Publish them as a folder entry that can
                        # be re-saved.
                        if not preview:
                            self._publish_aligned_stack(window, processed_images)

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
                and window.raw_images is context.get("source_ref")
                and not stack_changed
                and len(processed_images) == len(context["included"])
            )
            if registration_performed and alignment_matches_ui:
                window.aligned_images = processed_images
                window.is_images_aligned = True
                window.last_alignment_options = context["alignment_identity"]
                window.alignment_request_key = context["alignment_key"]
                # The disk cache was written by the worker (it owns the PNG
                # encode); writing it here again re-encoded the whole stack on
                # the GUI thread for nothing.

            total_time = alignment_time + fusion_time

            info_lines = []

            if preview:
                info_lines.append(trans.t("info_preview_quality"))
            cache_hit = bool(getattr(self.worker, "alignment_reused", False))
            if registration_performed and cache_hit:
                # The frames are aligned, but this run did not align them:
                # reporting 0.00 s under the method name read as "aligned
                # instantly" instead of "reused the cached result".
                info_lines.append(trans.t("info_align_none_cached"))
                info_lines.append(trans.t("info_align_time").format(0.0))
            elif registration_performed:
                alignment_method = window.current_alignment_method()
                align_method_str = trans.t(get_registration(alignment_method).label_key)
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
            self._dispose_worker()

    def _publish_aligned_stack(self, window, frames) -> None:
        """Stage an align-only result as a folder the output list can show."""
        try:
            from controllers.output_manager import publish_aligned_stack

            publish_aligned_stack(window, frames,
                                  filenames=(self._render_context or {}).get("filenames"))
        except Exception as exc:  # never fail a render over the listing
            import logging
            logging.getLogger("openfocus").warning(
                "could not publish the aligned stack: %s", exc)

    def _dispose_worker(self) -> None:
        """Release the finished RenderWorker without destroying a live QThread.

        Dropping the last Python reference to a QThread that is still
        finishing makes Qt destroy the C++ object underneath a running
        thread: the worker emits its result from inside run(), so by the
        time this slot runs the thread may not have returned yet. Qt then
        aborts the process in Qt6Core.dll (exception 0xc0000409) — the crash
        seen after ECC + AI renders. deleteLater() defers destruction to the
        event loop, which only runs once the thread is really gone.
        """
        worker = self.worker
        self.worker = None
        if worker is None:
            return
        dispose = getattr(worker, "deleteLater", None)
        if dispose is None:
            return
        if worker.isRunning():
            retiring = getattr(self, "_retiring_workers", None)
            if retiring is None:
                retiring = self._retiring_workers = set()
            retiring.add(worker)
            def release():
                if worker in retiring:
                    retiring.remove(worker)
                    worker.deleteLater()
            worker.finished.connect(release)
            if not worker.isRunning():
                release()
        else:
            dispose()

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
        key = {"align": "render_stage_align", "fuse": "render_stage_fuse"}.get(stage, "render_progress")
        text = trans.t(key)
        self._progress_first_t = None
        self._show_progress_bar(text)

    def on_render_progress(self, done: int, total: int) -> None:
        """Surface tiled-fusion progress in the status bar with an ETA."""
        import time as _time
        window = self.window
        bar = getattr(self, "_status_bar_progress", None)
        if bar is not None and total > 0:
            # Range first, then value: Qt clamps a value outside the current
            # range, so setting the value while the bar is still in its
            # indeterminate (0..0) mode silently drops it and the bar looks
            # frozen for the rest of the run.
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
        self._dispose_worker()
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

