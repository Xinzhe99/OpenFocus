from dataclasses import dataclass
from typing import Any

import cv2
import os
from PyQt6.QtCore import QObject, QPoint, QThread, Qt
from PyQt6.QtGui import QAction, QIcon, QDragEnterEvent, QDropEvent
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import QListWidgetItem, QMenu, QMessageBox, QDialog

from core.workers import StackLoadWorker
from dialogs import DownsampleDialog
from utils import show_message_box, show_warning_box
from ui import styles as ui_styles
from locales import trans


class StackReviewWorker(QThread):
    """Analyze a snapshot without touching widgets from the worker thread."""

    def __init__(self, frames):
        super().__init__()
        self.frames = tuple(frames)
        self.results = None
        self.error = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from core.stack_review import analyze_frames
            self.results = analyze_frames(self.frames, cancel=lambda: self._cancelled)
        except InterruptedError:
            self._cancelled = True
        except Exception as exc:  # report failures on the UI thread
            self.error = str(exc)


@dataclass
class LoadOptions:
    scale_factor: float
    filenames: list[str]
    base_images: list[Any]
    working_images: list[Any]
    reset_scale_factor: float | None = None


class SourceManager(QObject):
    """Handles source list interactions and bookkeeping for the main window."""

    def __init__(self, window: Any):
        super().__init__()
        self.window = window
        self._review_worker = None
        self._review_frames = ()
        self._review_results = []
        self._review_status_key = None
        self._stack_baseline = None
        window.linear_dng_metadata = None
        self._review_error = ""

    def _clear_history(self):
        window = self.window
        for name in ("_undo_stack", "_redo_stack"):
            history = getattr(window, name, None)
            if history is not None:
                history.clear()
        refresh = getattr(window, "_refresh_undo_actions", None)
        if callable(refresh):
            refresh()

    def _reset_busy(self):
        window = self.window
        render = getattr(window, "render_manager", None)
        batch = getattr(window, "batch_manager", None)
        return (getattr(self, "_load_worker", None) is not None
                or getattr(render, "worker", None) is not None
                or bool(getattr(render, "_compare_mode", False))
                or getattr(window, "roi_alignment_worker", None) is not None
                or getattr(batch, "_worker", None) is not None
                or getattr(batch, "_thread", None) is not None)

    def reset_current_stack(self):
        window = self.window
        baseline = self._stack_baseline
        if not baseline or not getattr(window, "raw_images", None):
            show_warning_box(window, trans.t("reset_stack_title"), trans.t("reset_stack_empty"))
            return
        if self._reset_busy():
            show_warning_box(window, trans.t("reset_stack_title"), trans.t("reset_stack_busy"))
            return
        box = QMessageBox(window)
        from utils.ui_utils import style_message_box
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(trans.t("reset_stack_title"))
        box.setText(trans.t("reset_stack_warning"))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        box.button(QMessageBox.StandardButton.Yes).setText(trans.t("btn_continue"))
        box.button(QMessageBox.StandardButton.Cancel).setText(trans.t("btn_cancel_generic"))
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return
        # A nested event loop in the confirmation can start a new operation.
        if self._reset_busy() or self._stack_baseline is not baseline:
            show_warning_box(window, trans.t("reset_stack_title"), trans.t("reset_stack_busy"))
            return
        self._load_generation = getattr(self, "_load_generation", 0) + 1
        if self._review_worker is not None:
            self._review_worker.cancel()
        self._review_frames = ()
        self._review_results = []
        self._review_status_key = None
        window.raw_images = list(baseline["images"])
        window.base_images = list(baseline["images"])
        window.image_filenames = list(baseline["filenames"])
        window.image_source_paths = list(baseline["paths"])
        window.image_icc_profiles = list(baseline["profiles"])
        window.current_folder_path = baseline["folder"]
        window.current_scale_factor = baseline["scale"]
        window.base_scale_factor = baseline["scale"]
        window.source_px_um = baseline["source_px_um"]
        window.linear_dng_metadata = baseline.get("linear_dng_metadata")
        window.excluded_frames = set()
        window.source_transforms = []
        window.unrecoverable_source_changes = []
        window.scale_bar_enabled = False
        if getattr(window, "wipe_active", False):
            window.btn_wipe.setChecked(False)
        if hasattr(window, "_wipe_source_cache"):
            window._wipe_source_cache = None
        if hasattr(window, "_wipe_preview_record"):
            window._wipe_preview_record = None
        window.transform_manager.invalidate_processing_results(clear_output_view=True)
        window.label_manager.reset_labels()
        self._clear_history()
        invalidate = getattr(window, "invalidate_annotation_caches", None)
        if callable(invalidate):
            invalidate()
        window.transform_manager.reload_image_stack(initial_index=0)
        mark = getattr(window, "_mark_project_dirty", None)
        if callable(mark):
            mark()
        persist = getattr(window, "persist_settings", None)
        if callable(persist):
            persist()
        from utils import recovery
        recovery.write_snapshot(window)
        window._show_status_message(trans.t("reset_stack_done"), 5000)

    def _review_matches_stack(self, frames):
        current = self.window.raw_images
        return len(frames) == len(current) and all(
            before is after for before, after in zip(frames, current))

    def toggle_stack_review(self):
        if self._review_worker is not None:
            self._review_worker.cancel()
            self._review_status_key = "stack_review_cancelling"
            self.refresh_review_text()
            return
        frames = tuple(self.window.raw_images)
        if len(frames) < 2:
            return
        self._review_frames = frames
        self._review_results = []
        self._review_error = ""
        self._review_status_key = "stack_review_working"
        worker = StackReviewWorker(frames)
        self._review_worker = worker
        worker.finished.connect(self._finish_stack_review)
        self.decorate_source_items()
        worker.start()

    def _finish_stack_review(self):
        worker = self._review_worker
        if worker is None:
            return
        self._review_worker = None
        if not self._review_matches_stack(worker.frames):
            self._review_status_key = "stack_review_stale"
        elif worker._cancelled:
            self._review_status_key = "stack_review_cancelled"
        elif worker.error is not None:
            self._review_error = worker.error
            self._review_status_key = "stack_review_error"
        else:
            self._review_frames = worker.frames
            self._review_results = worker.results or []
            self._review_status_key = ("stack_review_found" if self._review_results
                                       else "stack_review_none")
        if self._review_status_key not in ("stack_review_found", "stack_review_none"):
            self._review_frames = ()
            self._review_results = []
        worker.deleteLater()
        self.decorate_source_items()

    def _visible_review_results(self):
        excluded = getattr(self.window, "excluded_frames", set())
        return [entry for entry in self._review_results
                if entry["index"] not in excluded
                and entry["reference_index"] not in excluded]

    def refresh_review_text(self):
        panel = getattr(self.window, "right_panel_components", None)
        if panel is None:
            return
        busy = self._review_worker is not None
        panel.btn_review_stack.setText(trans.t(
            "stack_review_cancel" if busy else "stack_review_button"))
        panel.btn_review_stack.setToolTip(trans.t(
            "stack_review_cancel" if busy else "stack_review_button")
            + "\n" + trans.t("stack_review_hint"))
        panel.btn_review_stack.setEnabled(busy or len(self.window.raw_images) >= 2)
        status_key = self._review_status_key
        visible = self._visible_review_results()
        if status_key in ("stack_review_found", "stack_review_none"):
            status_key = "stack_review_found" if visible else "stack_review_none"
        status = trans.t(status_key) if status_key else ""
        panel.review_status.setText(status.format(count=len(visible),
                                                  error=self._review_error))

    def _invalidate_stale_review(self):
        if self._review_frames and not self._review_matches_stack(self._review_frames):
            if self._review_worker is not None:
                self._review_worker.cancel()
            self._review_frames = ()
            self._review_results = []
            self._review_status_key = "stack_review_stale"

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def update_source_images_count(self) -> None:
        count = self.window.file_list.count()
        self._refresh_source_count()

    def _source_count_text(self, total: int = None) -> str:
        """Header text: plain count, or "11 of 13 participating"."""
        window = self.window
        if total is None:
            total = len(getattr(window, "raw_images", None) or [])
        try:
            included = len(window.included_indices())
        except Exception:
            included = total
        if included == total:
            return trans.t("label_source_images").format(total)
        return trans.t("label_source_images_included").format(
            included=included, total=total)

    def _refresh_source_count(self) -> None:
        label = getattr(self.window, "source_images_label", None)
        if label is not None:
            label.setText(self._source_count_text())

    def _confirm_mixed_sizes(self, images) -> bool:
        """Frames of different sizes in one dropped file set: ask whether to
        continue (fusion needs one size; the user may resize afterwards)."""
        shapes = {(img.shape[0], img.shape[1]) for img in images}
        if len(shapes) <= 1:
            return True
        msg = QMessageBox(self.window)
        from utils.ui_utils import style_message_box
        style_message_box(msg)
        msg.setWindowTitle(trans.t("msg_size_mismatch_title"))
        msg.setText(trans.t("msg_size_mismatch_stack_text"))
        msg.setInformativeText(trans.t("msg_size_mismatch_open_info"))
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        yes_btn = msg.button(QMessageBox.StandardButton.Yes)
        no_btn = msg.button(QMessageBox.StandardButton.No)
        if yes_btn:
            yes_btn.setText(trans.t("btn_continue"))
        if no_btn:
            no_btn.setText(trans.t("btn_cancel_generic"))
        msg.setStyleSheet(ui_styles.MESSAGE_BOX_STYLE)
        return msg.exec() == QMessageBox.StandardButton.Yes

    def _show_linear_profile_notice(self, metadata) -> None:
        if getattr(metadata, "profile_tags", None):
            show_message_box(self.window, trans.t("linear_raw_caption"),
                             trans.t("linear_raw_profile_hint"), "",
                             QMessageBox.Icon.Information)

    def _start_load_worker(self, folder=None, video=None, filepaths=None,
                           scale=1.0, append=False, on_success=None,
                           accept_event=None, prepare_options=None, on_error=None,
                           linear_raw=False) -> None:
        """Run the loader on a background thread and apply results on the
        UI thread when it finishes."""
        window = self.window
        worker = getattr(self, "_load_worker", None)
        if worker is not None:
            show_warning_box(window, trans.t("msg_warning"), trans.t("msg_load_in_progress"))
            if on_error is not None:
                on_error(trans.t("msg_load_in_progress"))
            return
        current_metadata = getattr(window, "linear_dng_metadata", None)
        if append and getattr(window, "raw_images", None):
            if current_metadata is not None:
                linear_raw = True
            elif linear_raw:
                message = trans.t("linear_raw_mismatch")
                if on_error is not None:
                    on_error(message)
                else:
                    show_warning_box(window, trans.t("msg_load_failed"), message)
                return
        if linear_raw and filepaths:
            from core.image_loader import is_raw_source
            if not all(is_raw_source(path) for path in filepaths):
                message = trans.t("linear_raw_invalid")
                if on_error is not None:
                    on_error(message)
                else:
                    show_warning_box(window, trans.t("msg_load_failed"), message)
                return
        working_scale = scale
        if append and getattr(window, "raw_images", None):
            working_scale = float(getattr(window, "current_scale_factor", 1.0) or 1.0)
            baseline = self._stack_baseline
            scale = baseline["scale"] if baseline else working_scale

        self._load_worker = StackLoadWorker(
            window.image_loader, folder=folder, video=video,
            filepaths=filepaths, scale=scale,
            thread_count=getattr(window, "thread_count", 0),
            exclude_paths=([window.defect_mask_path] if prepare_options is None
                           and getattr(window, 'defect_mask_path', '') else []),
            resource_preference=getattr(window, "resource_preference", "balanced"),
            linear_raw=linear_raw,
            expected_metadata=current_metadata if append and window.raw_images else None,
        )
        self._load_worker._load_applied = False
        self._load_accept_event = accept_event
        # Remembered so the frames just decoded can be traced back to the files
        # they came from (project files need that to round-trip).
        self._load_source = (folder, video, filepaths)
        # Load-identity token: a slow in-flight load whose result arrives
        # after the user swapped the stack (e.g. via drag-drop) is discarded
        # instead of silently replacing what they see.
        self._load_generation = getattr(self, "_load_generation", 0) + 1
        generation = self._load_generation

        def load_is_current():
            from PyQt6 import sip
            return (not (isinstance(window, QObject) and sip.isdeleted(window))
                    and not getattr(window, "_shutting_down", False)
                    and generation == getattr(self, "_load_generation", -1))

        def on_done(ok: bool, message: str, images, filenames) -> None:
            accepted = False
            previous_window_state = None
            previous_baseline = None
            try:
                if not load_is_current():
                    return  # superseded by a newer load request
                if not ok:
                    if on_error is not None:
                        on_error(message)
                    elif not message.startswith("CANCELLED") and accept_event is None:
                        show_warning_box(window, trans.t("msg_load_failed"),
                                         trans.t("msg_load_stack_failed_text"), message)
                    return
                load_options = self._build_load_options(images, filenames, scale)
                if append and window.raw_images and working_scale != scale:
                    from core.image_loader import UnsupportedTransparency, _has_transparency
                    if any(_has_transparency(image) for image in images):
                        raise UnsupportedTransparency("Transparent images cannot be downsampled during loading")
                    factor = working_scale / scale
                    from core.source_frames import transform_frame
                    load_options.working_images = [
                        transform_frame(image, 'resize', width=max(1, int(image.shape[1] * factor)),
                                        height=max(1, int(image.shape[0] * factor)))
                        for image in images]
                mixed_ok = self._confirm_mixed_sizes(load_options.working_images)
                if not load_is_current():
                    return
                if not mixed_ok:
                    if on_error is not None:
                        on_error("CANCELLED")
                    return
                append_ok = not append or self._confirm_append_dimensions(load_options.working_images)
                if not load_is_current():
                    return
                if not append_ok:
                    if on_error is not None:
                        on_error("CANCELLED")
                    return
                if prepare_options is not None:
                    load_options = prepare_options(load_options)
                if not load_is_current():
                    return
                state_fields = ("current_folder_path", "source_px_um", "linear_dng_metadata",
                                "base_images", "raw_images", "image_filenames", "image_source_paths",
                                "image_icc_profiles", "current_scale_factor", "base_scale_factor",
                                "excluded_frames", "source_transforms", "unrecoverable_source_changes")
                previous_window_state = {name: getattr(window, name) for name in state_fields
                                         if hasattr(window, name)}
                previous_baseline = self._stack_baseline
                load_folder = folder or (os.path.dirname(video) if video else
                                        os.path.dirname(filepaths[0]) if filepaths else "")
                if not append or not window.raw_images:
                    window.current_folder_path = load_folder
                metadata = getattr(window.image_loader, "px_size_um", None) if not video else None
                if not append or not window.raw_images:
                    window.source_px_um = metadata / scale if metadata else None
                else:
                    incoming_px = metadata / working_scale if metadata else None
                    if getattr(window, "source_px_um", None) != incoming_px:
                        window.source_px_um = None
                    baseline = self._stack_baseline
                    if baseline and baseline["source_px_um"] != (metadata / scale if metadata else None):
                        self._stack_baseline = {**baseline, "source_px_um": None}
                if not append or not window.raw_images:
                    window.linear_dng_metadata = worker.linear_dng_metadata
                self._apply_load_options(load_options, append=append,
                                         profile_map=getattr(worker, "source_icc_profiles", {}))
                accepted = True
                worker._load_stack = window.raw_images
                worker._load_revision = getattr(window, "source_revision", 0)
                if on_success is not None:
                    on_success()
                # Remember for the optional "restore last stack on startup"
                from utils.settings_store import get_settings, LAST_STACK_FOLDER_KEY
                get_settings().setValue(LAST_STACK_FOLDER_KEY, window.current_folder_path or "")
                # Exports inherit the source metadata from now on
                from utils.image_utils import set_source_exif
                set_source_exif(getattr(window.image_loader, "source_exif", b""))

                # Crash auto-recovery: refresh the session snapshot
                from utils import recovery
                recovery.write_snapshot(window)
                self._show_linear_profile_notice(getattr(window, "linear_dng_metadata", None))
                worker._load_applied = load_is_current()
            except Exception as exc:  # pylint: disable=broad-except
                if not accepted and previous_window_state is not None:
                    for name, value in previous_window_state.items():
                        setattr(window, name, value)
                    self._stack_baseline = previous_baseline
                if on_error is not None:
                    on_error(str(exc))
                    return
                show_message_box(
                    window,
                    trans.t("msg_load_error"),
                    trans.t("msg_load_stack_error_text"),
                    f"Error: {str(exc)}",
                    QMessageBox.Icon.Critical,
                )
            finally:
                if ok and not accepted:
                    restore = getattr(worker, "restore_loader_state", None)
                    if restore is not None:
                        restore(images)
                if accepted:
                    previous_state = getattr(worker, "_previous_loader_state", None)
                    if previous_state is not None:
                        previous_state.clear()

        worker = self._load_worker
        worker.progress.connect(self._on_load_progress)
        worker.finished_load.connect(on_done)
        worker.finished.connect(lambda: self._finish_load_worker(worker))
        self._show_load_status()
        worker.start()

    def _finish_load_worker(self, worker) -> None:
        if self._load_worker is worker:
            self._load_worker = None
            self.window.statusBar().clearMessage()
            for name in ("_load_cancel_button", "_load_status_label", "_load_progress_bar"):
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.hide()
            self._start_queued_render(worker)
        worker.deleteLater()

    def _start_queued_render(self, worker) -> None:
        """Start the render the user asked for while this load was running."""
        render = getattr(self.window, "render_manager", None)
        request = getattr(render, "_render_after_load", None) if render is not None else None
        if request is None:
            return
        try:
            ready = (request.get("load_worker") is worker
                     and request.get("load_generation") == getattr(self, "_load_generation", 0)
                     and bool(getattr(worker, "_load_applied", False))
                     and not getattr(worker, "is_cancelled", False)
                     and getattr(worker, "_load_stack", None) is self.window.raw_images
                     and getattr(worker, "_load_revision", None) == getattr(self.window, "source_revision", 0)
                     and len(self.window.effective_stack()[0]) >= 2)
        except Exception:
            ready = False
        if request.get("compare", False) and not getattr(render, "_compare_mode", False):
            ready = False
        if ready:
            from PyQt6.QtCore import QTimer
            source_ref = self.window.raw_images
            revision = getattr(self.window, "source_revision", 0)
            included = list(self.window.effective_stack()[2])

            def start():
                # Keep the request cancellable until this event-loop turn;
                # replacing the stack or starting a new load invalidates it.
                if render._render_after_load is not request:
                    return
                render._render_after_load = None
                from PyQt6 import sip
                current = (not (isinstance(self.window, QObject) and sip.isdeleted(self.window))
                           and not getattr(self.window, "_shutting_down", False)
                           and getattr(self, "_load_worker", None) is None
                           and request["load_generation"] == getattr(self, "_load_generation", 0)
                           and self.window.raw_images is source_ref
                           and getattr(self.window, "source_revision", 0) == revision
                           and list(self.window.effective_stack()[2]) == included)
                compare = request.get("compare", False)
                if compare:
                    current = (current and render._compare_mode
                               and request["compare_token"] is render._compare_token
                               and not render._stack_changed_since(render._compare_context))
                if not current:
                    if compare and request["compare_token"] is render._compare_token:
                        render._abort_compare_run()
                    return
                render.start_render(force_algorithm=request.get("force_algorithm"))

            QTimer.singleShot(0, start)
        else:
            render._render_after_load = None
            if request.get("compare", False):
                render._abort_compare_run()
            self.window.statusBar().showMessage(trans.t("msg_render_queue_cancelled"), 5000)

    def _show_load_status(self) -> None:
        from PyQt6.QtWidgets import QPushButton, QLabel, QProgressBar
        window = self.window
        button = getattr(self, "_load_cancel_button", None)
        if button is None:
            self._load_status_label = QLabel()
            self._load_progress_bar = QProgressBar()
            self._load_progress_bar.setMaximumWidth(140)
            self._load_progress_bar.setFormat("%p%")
            self._load_progress_bar.setRange(0, 0)
            window.statusBar().addPermanentWidget(self._load_status_label)
            window.statusBar().addPermanentWidget(self._load_progress_bar)
            button = QPushButton(trans.t("btn_cancel_generic"))
            button.clicked.connect(self.cancel_load)
            window.statusBar().addPermanentWidget(button)
            self._load_cancel_button = button
        self._load_progress_bar.setRange(0, 0)
        self._load_progress_bar.setValue(0)
        button.setEnabled(True)
        button.show()
        self._load_status_label.setText(trans.t("msg_loading_stack"))
        self._load_status_label.show()
        self._load_progress_bar.show()
        window.statusBar().showMessage(trans.t("msg_loading_stack"))

    def _on_load_progress(self, done: int, total: int) -> None:
        worker = self.sender()
        if (worker is None or worker is not getattr(self, "_load_worker", None)
                or worker.is_cancelled or worker.video):
            return
        text = trans.t("msg_loading_stack")
        if total > 0:
            self._load_progress_bar.setRange(0, total)
            self._load_progress_bar.setValue(done)
            text = f"{text} ({done}/{total})"
        else:
            self._load_progress_bar.setRange(0, 0)
            text = f"{text} ({done})"
        if getattr(getattr(self.window, "render_manager", None), "_render_after_load", False):
            text = f"{text} · {trans.t('msg_render_queued_short')}"
        self._load_status_label.setText(text)
        self.window.statusBar().showMessage(text)

    def cancel_load(self) -> None:
        worker = getattr(self, "_load_worker", None)
        if worker is not None:
            worker.request_cancel()
            self._load_cancel_button.setEnabled(False)
            self._load_status_label.setText(trans.t("depth_cancelling"))
            self.window.statusBar().showMessage(trans.t("depth_cancelling"))

    def restore_last_stack(self) -> None:
        """Re-load the most recently opened stack (skip the downsample dialog,
        reusing the scale from the previous session)."""
        from utils.settings_store import get_settings, LAST_STACK_FOLDER_KEY
        folder = str(get_settings().value(LAST_STACK_FOLDER_KEY, "") or "")
        if folder and os.path.isdir(folder):
            window = self.window
            self._start_load_worker(
                folder=folder,
                scale=getattr(window, "current_scale_factor", 1.0),
                append=False,
            )

    def _ask_load_scale(self):
        """Loading scale for a new import, or None if the user cancelled.

        The scale prompt used to appear before *every* load; users who always
        load at full size can now tick "don't ask again" (re-enable under
        Settings), after which their last choice is applied silently."""
        window = self.window
        if not getattr(window, "ask_load_scale", True):
            return float(getattr(window, "load_scale_default", 1.0) or 1.0)
        current_scale = getattr(window, "current_scale_factor", 1.0)
        dlg = DownsampleDialog(window, initial_scale=current_scale,
                               show_dont_ask=True)
        if not dlg.exec():
            return None
        scale = dlg.get_scale_factor()
        window.load_scale_default = scale
        dont_ask = getattr(dlg, "dont_ask_again", None)
        if callable(dont_ask) and dont_ask():
            window.ask_load_scale = False
        persist = getattr(window, "persist_settings", None)
        if callable(persist):
            persist()
        return scale

    def load_image_stack(self, folder_path: str, append: bool = False) -> None:
        window = self.window
        from utils.logging_setup import log_action
        log_action("load_stack", folder=folder_path, append=append)

        # 同一文件夹再次导入按"重新加载"处理：追加只会得到帧数翻倍的
        # 重复栈（恢复会话后再导入、拖同一文件夹两次都触发过）
        same_folder = (os.path.normcase(os.path.abspath(folder_path)) == os.path.normcase(
            os.path.abspath(getattr(window, "current_folder_path", "") or "")))
        linear_reload = same_folder and getattr(window, "linear_dng_metadata", None) is not None
        if append and same_folder:
            append = False

        scale_factor = self._ask_load_scale()
        if scale_factor is None:
            return

        # Decode on a background thread so the window stays responsive on
        # large stacks; the continuation runs on the UI thread via the signal.
        self._start_load_worker(
            folder=folder_path, scale=scale_factor, append=append, linear_raw=linear_reload,
            on_success=lambda: window.add_recent_file(folder_path) if hasattr(window, "add_recent_file") else None,
        )

    def load_video_stack(self, video_path: str, append: bool = False) -> None:
        """Load image stack from a video file."""
        window = self.window

        scale_factor = self._ask_load_scale()
        if scale_factor is None:
            return

        self._start_load_worker(
            video=video_path, scale=scale_factor, append=append,
            on_success=lambda: window.add_recent_file(video_path) if hasattr(window, "add_recent_file") else None,
        )

    def prompt_and_load_linear_raw(self) -> None:
        from core.image_loader import RAW_FORMATS, is_raw_source
        from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir

        patterns = " ".join(f"*{ext}" for ext in sorted(RAW_FORMATS | {".tif", ".tiff"}))
        paths, _ = FileDialog.getOpenFileNames(
            self.window, trans.t("linear_raw_caption"), get_last_dialog_dir(),
            f"RAW ({patterns});;All Files (*)")
        if not paths:
            return
        if not all(is_raw_source(path) for path in paths):
            show_warning_box(self.window, trans.t("msg_load_failed"), trans.t("linear_raw_invalid"))
            return
        set_last_dialog_dir(paths[0])
        scale = self._ask_load_scale()
        if scale is not None:
            self._start_load_worker(filepaths=paths, scale=scale, linear_raw=True)

    def prompt_and_load_stack(self) -> None:
        from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir
        window = self.window

        dialog = FileDialog.for_directories(
            window, trans.t("action_open_folder"), get_last_dialog_dir())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected = dialog.selectedFiles()
            if selected:
                set_last_dialog_dir(selected[0])
                self.load_image_stack(selected[0])

    def prompt_and_load_video(self) -> None:
        """Open a file dialog to select a video file and load it as image stack."""
        from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir
        window = self.window
        from core.image_loader import ImageStackLoader

        # Build video filter string
        video_exts = " ".join([f"*{ext}" for ext in ImageStackLoader.SUPPORTED_VIDEO_FORMATS])

        video_path, _ = FileDialog.getOpenFileName(
            window,
            trans.t("action_open_video"),
            get_last_dialog_dir(),
            f"Video Files ({video_exts});;All Files (*)",
        )

        if video_path:
            set_last_dialog_dir(video_path)
            self.load_video_stack(video_path)

    def can_accept_drag(self, event: QDragEnterEvent) -> bool:
        if not event.mimeData().hasUrls():
            return False

        urls = event.mimeData().urls()
        if not urls:
            return False

        # Accept if single directory, or one-or-more files
        # We will further validate on drop
        return True

    def handle_drop_event(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        paths = [u.toLocalFile() for u in urls]

        # If a single directory was dropped, ask user how to import
        if len(paths) == 1 and os.path.isdir(paths[0]):
            from dialogs import FolderImportDialog
            dialog = FolderImportDialog(paths[0], self.window)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                if dialog.is_single_stack():
                    self.load_image_stack(paths[0], append=True)
                    event.acceptProposedAction()
                else:
                    # 多组图像栈 - 先弹出下采样设置
                    scale = self._ask_load_scale()
                    if scale is None:
                        event.ignore()
                        return

                    # 打开批处理对话框并预加载文件夹
                    self.window.show_batch_processing_dialog(preload_folder_paths=[paths[0]], scale_factor=scale)
                    event.acceptProposedAction()
            else:
                event.ignore()
            return

        # If multiple directories were dropped, ask user how to import
        elif all(os.path.isdir(p) for p in paths):
            scale = self._ask_load_scale()
            if scale is None:
                event.ignore()
                return

            self.window.show_batch_processing_dialog(preload_folder_paths=paths, scale_factor=scale)
            event.acceptProposedAction()
            return

        # Check if a single video file was dropped
        from core.image_loader import ImageStackLoader
        if len(paths) == 1 and os.path.isfile(paths[0]):
            ext = os.path.splitext(paths[0])[1].lower()
            if (ext in ImageStackLoader.SUPPORTED_VIDEO_FORMATS
                    and getattr(self.window, "linear_dng_metadata", None) is None):
                self.load_video_stack(paths[0], append=True)
                event.acceptProposedAction()
                return

        # Otherwise treat dropped items as a list of files
        filepaths = [p for p in paths if os.path.isfile(p)]
        if not filepaths:
            show_warning_box(self.window, trans.t("msg_warning"), trans.t("msg_drop_no_valid_files_text"))
            event.ignore()
            return

        # Filter by supported image extensions (use ImageStackLoader.SUPPORTED_FORMATS)
        from core.image_loader import ImageStackLoader
        if getattr(self.window, "linear_dng_metadata", None) is not None:
            from core.image_loader import is_raw_source
            if len(filepaths) != len(paths) or any(not is_raw_source(path) for path in paths):
                show_warning_box(self.window, trans.t("msg_load_failed"), trans.t("linear_raw_invalid"))
                event.ignore()
                return

        valid_paths = []
        for p in filepaths:
            ext = os.path.splitext(p)[1].lower()
            if ext in ImageStackLoader.SUPPORTED_FORMATS:
                valid_paths.append(p)

        if not valid_paths:
            show_warning_box(self.window, trans.t("msg_warning"), trans.t("msg_drop_no_supported_images_text"))
            event.ignore()
            return

        # Prompt for downsample (same UX as folder loading)
        try:
            scale = self._ask_load_scale()
            if scale is None:
                # user cancelled
                event.ignore()
                return

            # Decode on the load worker like folder loads: dropping a large
            # file set used to freeze the window while it decoded.
            self._start_load_worker(filepaths=valid_paths, scale=scale, append=True)
            event.acceptProposedAction()
        except Exception as exc:  # pylint: disable=broad-except
            show_message_box(
                self.window,
                trans.t("msg_load_error"),
                trans.t("msg_load_dropped_error_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )
            event.ignore()

    def _build_load_options(
        self,
        full_res_images: list[Any],
        filenames: list[str],
        scale_factor: float,
    ) -> LoadOptions:
        # Sources retain paths and bounded previews rather than original pixel
        # arrays. Share those immutable frames across base/working stacks, while
        # keeping separate lists so delete/reorder cannot mutate both at once.
        return LoadOptions(
            scale_factor=scale_factor,
            filenames=list(filenames),
            base_images=list(full_res_images),
            working_images=list(full_res_images),
        )

    def _source_paths_for(self, filenames: list[str]) -> list[str]:
        """Full path of every freshly loaded frame, aligned with `filenames`.

        Empty when the origin cannot be mapped back to files (video stacks
        decode to synthetic frame names that have no file on disk).
        """
        recorded = list(getattr(getattr(self.window, "image_loader", None), "image_paths", []) or [])
        if len(recorded) == len(filenames):
            return recorded
        folder, video, filepaths = getattr(self, "_load_source", (None, None, None))
        if video:
            return [video] * len(filenames)
        if folder:
            from utils.project_file import _multipage_container_for
            return [_multipage_container_for(name, folder) or os.path.join(folder, name)
                    for name in filenames]
        if filepaths and len(filepaths) == len(filenames):
            return list(filepaths)
        return []

    def _apply_load_options(self, options: LoadOptions, append: bool = False,
                            profile_map=None) -> None:
        window = self.window
        paths = self._source_paths_for(options.filenames)
        previous_paths = list(getattr(window, "image_source_paths", None) or [])
        previous_count = len(window.image_filenames or [])
        if profile_map is None:
            profile_map = getattr(getattr(window, "image_loader", None), "source_icc_profiles", {})
        profiles = [profile_map.get(path, b"") for path in paths]
        if len(profiles) != len(options.working_images):
            profiles = [b""] * len(options.working_images)
        previous_profiles = list(getattr(window, "image_icc_profiles", ()) or ())
        if len(previous_profiles) != len(window.raw_images or []):
            previous_profiles = [b""] * len(window.raw_images or [])
        if (not append or not window.raw_images) and hasattr(options, "source_px_um"):
            window.source_px_um = options.source_px_um
        reset_scale = getattr(options, "reset_scale_factor", None) or options.scale_factor
        source_px_um = getattr(window, "source_px_um", None)
        reset_px_um = (source_px_um * options.scale_factor / reset_scale
                       if source_px_um is not None else None)

        if append and window.raw_images:
            if getattr(window, "source_transforms", None):
                window.unrecoverable_source_changes = list(dict.fromkeys(
                    list(getattr(window, "unrecoverable_source_changes", [])) + [
                        "Frames appended after source transformations"]))
            window.base_images = list(window.raw_images) + list(options.working_images)
            window.base_scale_factor = window.current_scale_factor
            window.raw_images = (window.raw_images or []) + options.working_images
            window.image_icc_profiles = previous_profiles + profiles
            window.image_filenames = (window.image_filenames or []) + options.filenames
            # Keep the paths aligned with the names; anything else is better
            # dropped than left pointing at the wrong frame.
            window.image_source_paths = (
                previous_paths + paths
                if paths and len(previous_paths) == previous_count
                else []
            )
            initial_index = window.current_display_index if window.current_display_index >= 0 else 0
            baseline = self._stack_baseline
            if baseline is not None:
                self._stack_baseline = {
                    **baseline,
                    "images": baseline["images"] + list(getattr(options, "reset_images", options.base_images)),
                    "filenames": baseline["filenames"] + list(options.filenames),
                    "profiles": baseline["profiles"] + profiles,
                    "paths": (baseline["paths"] + paths
                              if len(baseline["paths"]) == len(baseline["images"])
                              and len(paths) == len(options.filenames) else []),
                }
        else:
            window.base_images = options.base_images
            window.current_scale_factor = options.scale_factor
            # The loader already decoded at this scale, so base_images is the
            # sharpest copy available until the source is read again.
            window.base_scale_factor = options.scale_factor
            window.raw_images = options.working_images
            window.image_filenames = options.filenames
            window.image_source_paths = paths
            window.image_icc_profiles = profiles
            window.excluded_frames = set()
            window.source_transforms = []
            window.unrecoverable_source_changes = []
            window.label_manager.reset_labels()
            self._stack_baseline = {
                "images": list(getattr(options, "reset_images", options.working_images)),
                "filenames": list(options.filenames),
                "paths": list(getattr(options, "reset_paths", paths)),
                "profiles": list(profiles),
                "folder": window.current_folder_path,
                "scale": reset_scale,
                "source_px_um": reset_px_um,
                "linear_dng_metadata": getattr(window, "linear_dng_metadata", None),
            }
            initial_index = 0

        window.transform_manager.invalidate_processing_results(
            clear_output_view=True,
            preserve_outputs=True,
        )
        window.transform_manager.reload_image_stack(initial_index=initial_index)

    def _confirm_append_dimensions(self, new_images: list[Any]) -> bool:
        window = self.window
        if not window.raw_images:
            return True

        try:
            existing_shape = window.raw_images[0].shape[:2]
            new_shapes = {(img.shape[0], img.shape[1]) for img in new_images}
            if len(new_shapes) == 1 and existing_shape in new_shapes:
                return True
        except Exception:
            return True

        msg = QMessageBox(window)
        msg.setWindowTitle(trans.t("msg_size_mismatch_title"))
        msg.setText(trans.t("msg_size_mismatch_text"))
        msg.setInformativeText(trans.t("msg_size_mismatch_info"))
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        yes_btn = msg.button(QMessageBox.StandardButton.Yes)
        no_btn = msg.button(QMessageBox.StandardButton.No)
        if yes_btn:
            yes_btn.setText(trans.t("btn_continue"))
        if no_btn:
            no_btn.setText(trans.t("btn_cancel_generic"))
        msg.setStyleSheet(ui_styles.MESSAGE_BOX_STYLE)
        ret = msg.exec()
        return ret == QMessageBox.StandardButton.Yes

    def clear_image_stack(self) -> None:
        window = self.window
        self._stack_baseline = None
        if hasattr(window, "_wipe_source_cache"):
            window._wipe_source_cache = None
        if hasattr(window, "_wipe_preview_record"):
            window._wipe_preview_record = None

        # Also drops the ROI stack, rect and tool button (see reset_roi_selection)
        window.transform_manager.invalidate_processing_results(clear_output_view=True)

        window.raw_images = []
        window.base_images = []
        window.stack_images = []
        window.image_filenames = []
        window.image_source_paths = []
        window.linear_dng_metadata = None
        window.image_icc_profiles = []
        window.excluded_frames = set()
        window.source_transforms = []
        window.unrecoverable_source_changes = []
        window.source_px_um = None
        self._load_generation = getattr(self, "_load_generation", 0) + 1
        window.base_scale_factor = 1.0
        window.current_display_index = -1

        if getattr(window, "scale_bar_enabled", False):
            window.scale_bar_enabled = False
            window.persist_settings()

        window.label_manager.reset_labels()

        # The sharpness strip belongs to the stack that was just dropped; left
        # alone it kept drawing the old frame count's bars and highlight under
        # an empty source list.
        curve = getattr(window, "sharpness_curve", None)
        if curve is not None:
            curve.clear()

        window.transform_manager.reload_image_stack(initial_index=None)

        # No source frames left to compare against: exit wipe mode cleanly
        if getattr(window, "wipe_active", False) and hasattr(window, "btn_wipe"):
            window.btn_wipe.setChecked(False)

    def update_slider_range(self) -> None:
        window = self.window
        if window.stack_images:
            window.stack_slider.setEnabled(True)
            window.stack_slider.setRange(0, len(window.stack_images) - 1)
        else:
            window.stack_slider.setEnabled(False)
            window.stack_slider.setRange(0, 0)

    def update_file_list(self, filenames, thumbnails) -> None:
        window = self.window
        window.file_list.blockSignals(True)
        window.file_list.clear()

        try:
            for filename, thumbnail in zip(filenames, thumbnails):
                item = QListWidgetItem(QIcon(thumbnail), filename)
                window.file_list.addItem(item)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            show_message_box(
                window,
                trans.t("msg_update_error_title"),
                trans.t("msg_update_file_list_text"),
                f"Error: {str(exc)}",
                QMessageBox.Icon.Critical,
            )
        finally:
            self.decorate_source_items()
            window.file_list.blockSignals(False)

        self.update_source_images_count()

    def sync_slider_from_list(self, row: int) -> None:
        if row >= 0:
            self.window.stack_slider.setValue(row)

    # ------------------------------------------------------------------
    # Context menu & deletions
    # ------------------------------------------------------------------
    def show_source_context_menu(self, position: QPoint) -> None:
        window = self.window
        menu = QMenu(window.file_list)

        delete_action = QAction(trans.t("action_delete"), window)
        delete_action.triggered.connect(self.delete_selected_source_images)
        menu.addAction(delete_action)

        menu.exec(window.file_list.mapToGlobal(position))

    def delete_source_image(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.file_list.row(item)
        if row < 0:
            return

        if len(window.raw_images) <= 1:
            self.clear_image_stack()
            return

        window.push_undo()
        window.transform_manager.invalidate_processing_results(clear_output_view=False, preserve_outputs=True)

        self._pop_sequence(window.image_filenames, row)
        self._pop_sequence(window.raw_images, row)
        self._pop_sequence(getattr(window, "base_images", None), row)
        self._pop_sequence(getattr(window, "image_source_paths", None), row)
        self._pop_sequence(getattr(window, "image_icc_profiles", None), row)
        self._remap_exclusions_after_delete([row])

        if window.raw_images:
            new_index = min(row, len(window.raw_images) - 1)
            window.transform_manager.reload_image_stack(initial_index=new_index)
        else:
            self.clear_image_stack()

    def delete_selected_source_images(self) -> None:
        window = self.window
        selected_items = window.file_list.selectedItems()
        if not selected_items:
            return

        rows = [window.file_list.row(item) for item in selected_items]
        rows = [row for row in rows if row >= 0]
        if not rows:
            return

        confirm = QMessageBox(window)
        from utils.ui_utils import style_message_box
        style_message_box(confirm)
        confirm.setIcon(QMessageBox.Icon.Warning)
        confirm.setWindowTitle(trans.t("msg_confirm_delete_title"))
        confirm.setText(trans.t("msg_confirm_delete_source_text").format(count=len(rows)))
        confirm.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Yes:
            return

        remaining = len(window.raw_images) - len(rows)
        if remaining <= 0:
            self.clear_image_stack()
            return

        window.push_undo()
        window.transform_manager.invalidate_processing_results(clear_output_view=False, preserve_outputs=True)

        rows.sort(reverse=True)

        for row in rows:
            self._pop_sequence(window.image_filenames, row)
            self._pop_sequence(window.raw_images, row)
            self._pop_sequence(getattr(window, "base_images", None), row)
            self._pop_sequence(getattr(window, "image_source_paths", None), row)
            self._pop_sequence(getattr(window, "image_icc_profiles", None), row)
        self._remap_exclusions_after_delete(rows)

        if window.raw_images:
            target_index = min(min(rows), len(window.raw_images) - 1)
            window.transform_manager.reload_image_stack(initial_index=target_index)
        else:
            self.clear_image_stack()

    def _remap_exclusions_after_delete(self, rows) -> None:
        removed = set(rows)
        self.window.excluded_frames = {
            i - sum(r < i for r in removed)
            for i in getattr(self.window, "excluded_frames", set()) if i not in removed}

    def _pop_sequence(self, sequence: list[Any] | None, index: int) -> None:
        if sequence is not None and 0 <= index < len(sequence):
            sequence.pop(index)

    # ------------------------------------------------------------------
    # Per-frame exclusion + drag reordering
    # ------------------------------------------------------------------
    def decorate_source_items(self) -> None:
        """Give every list row its checkable "included" state and its original
        position tag (UserRole), so a drag-drop can be mapped back to data."""
        window = self.window
        self._invalidate_stale_review()
        findings = {entry["index"]: entry for entry in self._visible_review_results()}
        excluded = getattr(window, "excluded_frames", set())
        blocked = window.file_list.blockSignals(True)
        for row in range(window.file_list.count()):
            item = window.file_list.item(row)
            if item is None:
                continue
            item.setData(Qt.ItemDataRole.UserRole, row)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Unchecked if row in excluded
                else Qt.CheckState.Checked)
            font = item.font()
            font.setStrikeOut(row in excluded)
            item.setFont(font)
            names = getattr(window, "image_filenames", ())
            if row < len(names):
                name = names[row]
                finding = findings.get(row)
                if finding is not None:
                    label = trans.t("stack_review_row").format(
                        reference=finding["reference_index"] + 1,
                        difference=finding["difference"])
                    item.setText(f"{name} — {label}")
                    item.setToolTip(label)
                else:
                    item.setText(name)
                    item.setToolTip("")
        window.file_list.blockSignals(blocked)
        self.refresh_review_text()

    def on_item_changed(self, item: QListWidgetItem) -> None:
        """Update participation independently of the highlighted frame."""
        window = self.window
        row = window.file_list.row(item)
        if row < 0 or row >= len(window.raw_images):
            return
        excluded = getattr(window, "excluded_frames", None)
        if excluded is None:
            excluded = set()
            window.excluded_frames = excluded
        should_exclude = item.checkState() != Qt.CheckState.Checked
        if (row in excluded) == should_exclude:
            return
        window.push_undo()
        was_blocked = window.file_list.blockSignals(True)
        try:
            if item.checkState() == Qt.CheckState.Checked:
                excluded.discard(row)
            else:
                excluded.add(row)
            font = item.font()
            font.setStrikeOut(item.checkState() == Qt.CheckState.Unchecked)
            item.setFont(font)
        finally:
            window.file_list.blockSignals(was_blocked)
        self.processing_selection_changed()

    def processing_selection_changed(self) -> None:
        window = self.window
        window.source_revision = getattr(window, "source_revision", 0) + 1
        window.aligned_images = []
        window.is_images_aligned = False
        window.last_alignment_options = None
        window.alignment_request_key = None
        window.transform_manager.reset_roi_selection()
        self._refresh_source_count()
        self.decorate_source_items()
        if hasattr(window, "_mark_project_dirty"):
            window._mark_project_dirty()

    def apply_list_order(self) -> None:
        """Reorder the underlying stacks to match a drag-drop in the list.

        Items carry their original index in UserRole; after an InternalMove
        the widget order IS the wanted order.
        """
        window = self.window
        n = len(window.raw_images)
        if window.file_list.count() != n:
            return
        new_order = []
        for row in range(window.file_list.count()):
            item = window.file_list.item(row)
            old = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(old, int) or not (0 <= old < n):
                return  # unmappable: leave the data alone
            new_order.append(old)
        if len(set(new_order)) != n:
            return
        if new_order == list(range(n)):
            return

        window.push_undo()

        def _reorder(seq):
            return [seq[i] for i in new_order]

        window.raw_images = _reorder(window.raw_images)
        window.image_filenames = _reorder(window.image_filenames)
        if getattr(window, "base_images", None):
            window.base_images = _reorder(window.base_images)
        if getattr(window, "image_source_paths", None):
            window.image_source_paths = _reorder(window.image_source_paths)
        if len(getattr(window, "image_icc_profiles", ()) or ()) == n:
            window.image_icc_profiles = _reorder(window.image_icc_profiles)
        excluded = getattr(window, "excluded_frames", None)
        if excluded:
            # old index i ends up at position new_order.index(i)
            window.excluded_frames = {
                new_order.index(i) for i in excluded if i in new_order}

        # The remapped exclusion set survives cache invalidation.
        window.transform_manager.invalidate_processing_results(
            clear_output_view=False, preserve_outputs=True)
        kept = getattr(window, "current_display_index", 0)
        kept = kept if isinstance(kept, int) and 0 <= kept < n else 0
        new_pos = new_order.index(kept) if kept in new_order else 0
        window.transform_manager.reload_image_stack(initial_index=new_pos)

    # === Icon Drag-and-Drop Support Methods ===
    # These methods handle files dropped on app icon/taskbar/dock

    def load_image_stack_from_icon(self, folder_path: str) -> None:
        """Load image stack from folder dropped on app icon / passed on the
        command line. The intent there is unambiguous — one folder is one
        stack — so no import-mode dialog; open the batch dialog explicitly
        when several stacks are wanted. Same folder again = reload."""
        append = bool(getattr(self.window, "raw_images", None))
        same = (os.path.normcase(os.path.abspath(folder_path))
                == os.path.normcase(os.path.abspath(
                    getattr(self.window, "current_folder_path", "") or "")))
        if not append or same:
            self.load_image_stack(folder_path, append=False)
            return
        # 已有其他栈时才需要确认是追加还是……直接追加（与拖放语义一致）
        self.load_image_stack(folder_path, append=True)

    def load_multiple_folders_from_icon(self, folder_paths: list[str]) -> None:
        """Load multiple folders dropped on app icon."""
        scale = self._ask_load_scale()
        if scale is not None:
            self.window.show_batch_processing_dialog(
                preload_folder_paths=folder_paths,
                scale_factor=scale
            )

    def load_video_stack_from_icon(self, video_path: str) -> None:
        """Load video file dropped on app icon."""
        self.load_video_stack(video_path)

    def load_image_files_from_icon(self, file_paths: list[str]) -> None:
        """Load image files dropped on app icon."""
        scale = self._ask_load_scale()
        if scale is None:
            return

        self._start_load_worker(filepaths=list(file_paths), scale=scale, append=True)
