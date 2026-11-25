from bisect import bisect_left
from dataclasses import dataclass
from typing import Any

import cv2
import os
from PyQt6.QtCore import QItemSelectionModel, QObject, QPoint, QThread, QTimer, Qt
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QDragEnterEvent, QDropEvent
from utils.file_dialogs import FileDialog
from PyQt6.QtWidgets import QListWidgetItem, QMenu, QMessageBox, QDialog

from utils import show_message_box, show_warning_box
from ui import styles as ui_styles
from locales import trans

_LAZY_IMPORTS = {
    'StackLoadWorker': ('core.workers', 'StackLoadWorker'),
    'DownsampleDialog': ('dialogs', 'DownsampleDialog'),
}


def _resolve_lazy(name: str):
    value = globals().get(name)
    if value is not None:
        return value
    module_name, attr = _LAZY_IMPORTS[name]
    value = getattr(__import__(module_name, fromlist=[attr]), attr)
    globals()[name] = value
    return value


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        return _resolve_lazy(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    source_exifs: list[bytes] | None = None
    source_icc_profiles: list[bytes] | None = None


class SourceManager(QObject):
    """Handles source list interactions and bookkeeping for the main window."""

    def __init__(self, window: Any):
        super().__init__()
        self.window = window
        self._review_worker = None
        self._review_action = None
        self._review_frames = ()
        self._review_results = []
        self._review_status_key = None
        self._review_notice_timer = QTimer(self)
        self._review_notice_timer.setSingleShot(True)
        self._review_notice_timer.setInterval(5000)
        self._review_notice_timer.timeout.connect(self._expire_review_notice)
        self._render_prewarm_timer = QTimer(self)
        self._render_prewarm_timer.setSingleShot(True)
        self._render_prewarm_timer.setInterval(400)
        self._render_prewarm_timer.timeout.connect(self._start_render_prewarm)
        self._render_prewarm_stack = None
        self._stack_baseline = None
        window.linear_dng_metadata = None
        self._review_error = ""

        self._source_hover = None

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

    def _render_blocks_load(self):
        render = getattr(self.window, "render_manager", None)
        return (getattr(render, "worker", None) is not None
                or getattr(render, "_pending_storage", None) is not None
                or (getattr(render, "_compare_mode", False)
                    and getattr(render, "_render_after_load", None) is None))

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
        window.image_source_exifs = list(baseline.get("exifs", ()))
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
        from utils import watermark
        watermark.reset_for_new_stack(window)
        if getattr(window, "wipe_active", False):
            window.btn_wipe.setChecked(False)
        if hasattr(window, "_wipe_source_cache"):
            window._wipe_source_cache = None
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
        worker.finished.connect(lambda: self._finish_stack_review(worker))
        self.decorate_source_items()
        worker.start()

    def _finish_stack_review(self, worker=None):
        worker = worker if worker is not None else self._review_worker
        if worker is None:
            return
        if worker is not self._review_worker:
            self._retiring_review_workers.discard(worker)
            worker.deleteLater()
            return
        self._review_worker = None
        if not self._review_matches_stack(worker.frames):
            self._review_status_key = "stack_review_stale"
        elif worker._cancelled:
            self._review_status_key = "stack_review_cancelled"
            self._review_notice_timer.start()
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

    def _expire_review_notice(self):
        if self._review_status_key == 'stack_review_cancelled':
            self._review_status_key = None
            self.refresh_review_text()

    def _visible_review_results(self):
        excluded = getattr(self.window, "excluded_frames", set())
        return [entry for entry in self._review_results
                if entry["index"] not in excluded
                and entry["reference_index"] not in excluded]

    def bind_source_actions(self, view):
        # Qt maps Ctrl in portable key sequences to Command on macOS. Keep
        # Ctrl+A for ordinary multi-row selection, independently of inclusion.
        def add_action(keys, callback):
            action = QAction(view)
            action.setShortcut(QKeySequence(keys))
            action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
            action.setAutoRepeat(False)
            action.setShortcutVisibleInContextMenu(True)
            action.triggered.connect(callback)
            view.addAction(action)
            return action

        def set_all(included):
            rows = set(range(len(self.window.raw_images)))
            self._set_frame_participation(rows, included=included)

        self._review_action = add_action('Ctrl+Shift+F', self.toggle_stack_review)
        self._include_all_action = add_action('Ctrl+Shift+I', lambda: set_all(True))
        self._exclude_all_action = add_action('Ctrl+Shift+X', lambda: set_all(False))
        self._refresh_source_actions()

    def _refresh_source_actions(self):
        busy = self._review_worker is not None
        total = len(self.window.raw_images)
        excluded = len(self.window.excluded_frames)
        self._review_action.setText(trans.t('stack_review_cancel' if busy else 'stack_review_action'))
        self._review_action.setToolTip(trans.t('stack_review_hint'))
        self._review_action.setEnabled(busy or total >= 2)
        self._include_all_action.setText(trans.t('source_include_all'))
        self._include_all_action.setEnabled(excluded > 0)
        self._exclude_all_action.setText(trans.t('source_exclude_all'))
        self._exclude_all_action.setEnabled(excluded < total)
        actions = (self._include_all_action, self._exclude_all_action, self._review_action)
        hint = '\n'.join(
            f'{action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)}  {action.text()}'
            for action in actions)
        self.window.source_images_label.setToolTip(hint)
        self.window.file_list.setToolTip(hint)
        self.window.file_list.setAccessibleDescription(hint)

    def refresh_review_text(self):
        if self._review_action is not None:
            self._refresh_source_actions()
        status_key = self._review_status_key
        visible = self._visible_review_results()
        if status_key in ("stack_review_found", "stack_review_none"):
            status_key = "stack_review_found" if visible else "stack_review_none"
        status = trans.t(status_key) if status_key else ""
        set_status = getattr(self.window, "_set_review_status", None)
        if set_status is not None:
            set_status(status.format(count=len(visible), error=self._review_error))

    def _invalidate_stale_review(self):
        if self._review_frames and not self._review_matches_stack(self._review_frames):
            if self._review_worker is not None:
                self._review_worker.cancel()
            self._review_frames = ()
            self._review_results = []
            self._review_status_key = "stack_review_stale"

    def _clear_review(self):
        worker = self._review_worker
        if worker is not None:
            worker.cancel()
            retired = self.__dict__.setdefault('_retiring_review_workers', set())
            retired.add(worker)
            self._review_worker = None
        self._review_frames = ()
        self._review_results = []
        self._review_status_key = None
        self._review_error = ''
        self.refresh_review_text()

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

    def _show_load_issues_notice(self, worker, loaded_count: int) -> None:
        if getattr(worker, "video", None):
            return
        loader = getattr(self.window, "image_loader", None)
        if loader is None:
            return
        failed = int(getattr(loader, "failed_count", 0) or 0)
        skipped = int(getattr(loader, "skipped_derived_count", 0) or 0)
        if failed <= 0 and skipped <= 0:
            return
        if failed > 0 and skipped > 0:
            message = trans.t("msg_load_frames_partial").format(
                loaded=loaded_count, failed=failed, skipped=skipped)
        elif failed > 0:
            message = trans.t("msg_load_frames_failed").format(
                loaded=loaded_count, failed=failed)
        else:
            message = trans.t("msg_load_frames_skipped_derived").format(
                loaded=loaded_count, skipped=skipped)
        status_bar = getattr(self.window, "statusBar", None)
        bar = status_bar() if callable(status_bar) else None
        if bar is not None:
            bar.showMessage(message, 9000)

    def _start_load_worker(self, folder=None, video=None, filepaths=None,
                           scale=1.0, append=False, on_success=None,
                           accept_event=None, prepare_options=None, on_error=None,
                           linear_raw=False, on_finished=None, crash_restore=False) -> None:
        """Run the loader on a background thread and apply results on the
        UI thread when it finishes."""
        window = self.window
        worker = getattr(self, "_load_worker", None)
        if worker is not None:
            show_warning_box(window, trans.t("msg_warning"), trans.t("msg_load_in_progress"))
            if on_error is not None:
                on_error(trans.t("msg_load_in_progress"))
            return
        if self._render_blocks_load():
            message = trans.t("load_render_busy")
            if on_error is not None:
                on_error(message)
            else:
                window.statusBar().showMessage(message, 5000)
            if on_finished is not None:
                on_finished(message)
            return

        def reject_start(message):
            if on_error is not None:
                on_error(message)
            else:
                show_warning_box(window, trans.t("msg_load_failed"), message)
            if on_finished is not None:
                on_finished(message)

        current_metadata = getattr(window, "linear_dng_metadata", None)
        if append and getattr(window, "raw_images", None):
            if current_metadata is not None:
                linear_raw = True
            elif linear_raw:
                reject_start(trans.t("linear_raw_mismatch"))
                return
        if linear_raw and folder and append:
            from core.image_loader import ImageStackLoader, is_raw_source
            from utils.path_identity import same_path
            excluded = getattr(window, "defect_mask_path", "") if prepare_options is None else ""
            try:
                with os.scandir(folder) as entries:
                    mixed = any(
                        entry.is_file()
                        and os.path.splitext(entry.name)[1].lower() in ImageStackLoader.SUPPORTED_FORMATS
                        and not (excluded and same_path(entry.path, excluded))
                        and not is_raw_source(entry.path)
                        for entry in entries
                    )
            except OSError:
                mixed = False  # The worker reports directory errors without changing the stack.
            if mixed:
                reject_start(trans.t("linear_raw_invalid"))
                return
        if linear_raw and filepaths:
            from core.image_loader import is_raw_source
            if not all(is_raw_source(path) for path in filepaths):
                reject_start(trans.t("linear_raw_invalid"))
                return
        working_scale = scale
        if append and getattr(window, "raw_images", None):
            working_scale = float(getattr(window, "current_scale_factor", 1.0) or 1.0)
            baseline = self._stack_baseline
            scale = baseline["scale"] if baseline else working_scale
        append_target = None
        if append and getattr(window, "raw_images", None):
            append_target = (getattr(window, "source_revision", 0), window.raw_images,
                             tuple(window.raw_images), getattr(window, "current_scale_factor", 1.0),
                             getattr(window, "base_scale_factor", 1.0),
                             getattr(window, "base_images", None), self._stack_baseline,
                             getattr(window, "linear_dng_metadata", None))

        def append_target_is_current():
            if append_target is None:
                return True
            revision, raw, frames, current_scale, base_scale, base, baseline, metadata = append_target
            current = getattr(window, "raw_images", None)
            return (getattr(window, "source_revision", 0) == revision and current is raw
                    and len(current) == len(frames)
                    and all(a is b for a, b in zip(current, frames))
                    and getattr(window, "current_scale_factor", 1.0) == current_scale
                    and getattr(window, "base_scale_factor", 1.0) == base_scale
                    and getattr(window, "base_images", None) is base
                    and self._stack_baseline is baseline
                    and getattr(window, "linear_dng_metadata", None) is metadata)

        # A discarded generation never reaches the publication point.
        result_error = "CANCELLED"

        def reject_changed_append():
            nonlocal result_error
            if append_target_is_current():
                return False
            message = trans.t("msg_append_target_changed")
            result_error = message
            if on_error is not None:
                on_error(message)
            else:
                show_warning_box(window, trans.t("msg_load_failed"), message)
            return True

        self._load_worker = _resolve_lazy('StackLoadWorker')(
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
                    and not getattr(worker, "is_cancelled", False)
                    and not getattr(window, "_shutting_down", False)
                    and generation == getattr(self, "_load_generation", -1))

        # Preparing or displaying frames may run a nested Qt event loop. Keep
        # the loader busy until both its thread and its GUI transaction finish.
        result_handled = thread_finished = False
        missing = object()

        def on_thread_finished():
            nonlocal thread_finished
            thread_finished = True
            if result_handled:
                self._finish_load_worker(worker)

        def on_done(ok: bool, message: str, images, filenames) -> None:
            nonlocal result_error, result_handled
            accepted = False
            previous_window_state = None
            previous_baseline = None
            try:
                if not load_is_current():
                    return  # superseded by a newer load request
                if not ok:
                    result_error = message or "Load failed"
                    if on_error is not None:
                        on_error(message)
                    elif not message.startswith("CANCELLED") and accept_event is None:
                        show_warning_box(window, trans.t("msg_load_failed"),
                                         trans.t("msg_load_stack_failed_text"), message)
                    return
                if reject_changed_append():
                    return
                load_options = (worker.prepared_options if crash_restore else
                                self._build_load_options(images, filenames, scale))
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
                mixed_ok = self._confirm_mixed_sizes(images if crash_restore else load_options.working_images)
                if not load_is_current():
                    return
                if reject_changed_append():
                    return
                if not mixed_ok:
                    if on_error is not None:
                        on_error("CANCELLED")
                    return
                append_ok = not append or self._confirm_append_dimensions(load_options.working_images)
                if not load_is_current():
                    return
                if reject_changed_append():
                    return
                if not append_ok:
                    if on_error is not None:
                        on_error("CANCELLED")
                    return
                if prepare_options is not None and not crash_restore:
                    load_options = prepare_options(load_options)
                if not load_is_current():
                    return
                if reject_changed_append():
                    return
                if crash_restore:
                    # Cancellation applies to decode and replay, before Qt
                    # publishes the stack and applies the project settings.
                    self._load_cancel_button.setEnabled(False)
                from utils import watermark
                state_fields = ("current_folder_path", "source_px_um", "linear_dng_metadata",
                                "base_images", "raw_images", "image_filenames", "image_source_paths",
                                "image_source_exifs", "image_icc_profiles", "current_scale_factor", "base_scale_factor",
                                "excluded_frames", "source_transforms", "unrecoverable_source_changes",
                                *(attr for attr, _ in watermark.ATTRS.values()),
                                *(attr for attr, _ in watermark.SCOPE_ATTRS.values()),
                                *watermark.TARGET_CONFIG_ATTRS.values(), watermark.STACK_LINEAGE_ATTR)
                # Rotating the lineage is what isolates the replaced stack's
                # outputs; a rolled-back load must restore the old one.
                watermark.stack_lineage(window)
                previous_window_state = {name: getattr(window, name, missing)
                                         for name in state_fields}
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
                if not append:
                    # First paint runs inside _apply_load_options. A watermark
                    # validated for the outgoing stack may fail on the incoming
                    # canvas and open a modal before the load can finish.
                    watermark.reset_for_new_stack(window)
                    invalidate = getattr(window, "invalidate_annotation_caches", None)
                    if callable(invalidate):
                        invalidate()
                self._apply_load_options(load_options, append=append,
                                         profile_map=getattr(worker, "source_icc_profiles", ()))
                if not append:
                    history = getattr(window, "output_list", None)
                    if history is not None:
                        history.setCurrentRow(-1)
                    # A project load restores its own settings in on_success.
                    persist = getattr(window, "persist_settings", None)
                    if callable(persist):
                        try:
                            persist()
                        except Exception:
                            pass
                    # Update history metadata without repainting the old result
                    # the stack replacement just cleared from the result view.
                    refresh = getattr(getattr(window, "output_manager", None),
                                      "refresh_annotation_export_state", None)
                    if callable(refresh):
                        try:
                            refresh()
                        except Exception:
                            pass
                accepted = True
                worker._load_stack = window.raw_images
                worker._load_revision = getattr(window, "source_revision", 0)
                if on_success is not None:
                    on_success()
                # Remember for the optional "restore last stack on startup"
                from utils.settings_store import get_settings, LAST_STACK_FOLDER_KEY, LAST_STACK_MODE_KEY
                settings = get_settings()
                settings.setValue(LAST_STACK_FOLDER_KEY, window.current_folder_path or "")
                settings.setValue(LAST_STACK_MODE_KEY,
                                  "linear" if getattr(window, "linear_dng_metadata", None) is not None
                                  else "standard")
                # Exports inherit the source metadata from now on
                from utils.image_utils import set_source_exif
                set_source_exif(getattr(window.image_loader, "source_exif", b""))

                # Recovery already has a validated snapshot of this exact
                # session. Avoid probing its sources again on the GUI thread.
                if not crash_restore:
                    from utils import recovery
                    recovery.write_snapshot(window)
                self._show_linear_profile_notice(getattr(window, "linear_dng_metadata", None))
                self._show_load_issues_notice(worker, len(images))
                worker._load_applied = load_is_current()
            except Exception as exc:  # pylint: disable=broad-except
                result_error = str(exc) or "Load failed"
                if not accepted and previous_window_state is not None:
                    for name, value in previous_window_state.items():
                        if value is not missing:
                            setattr(window, name, value)
                        elif hasattr(window, name):
                            delattr(window, name)
                    self._stack_baseline = previous_baseline
                    persist = getattr(window, "persist_settings", None)
                    if callable(persist):
                        try:
                            persist()
                        except Exception:
                            pass
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
                if on_finished is not None:
                    on_finished(None if worker._load_applied else result_error)
                result_handled = True
                if thread_finished:
                    self._finish_load_worker(worker)

        worker = self._load_worker
        if crash_restore:
            worker.prepare_result = lambda images, names: prepare_options(
                self._build_load_options(images, names, scale), lambda: worker.is_cancelled)
            worker.preparing.connect(self._show_restore_progress)
        worker.progress.connect(self._on_load_progress)
        if video:
            from core.video_import import bind_video_progress_callback
            bind_video_progress_callback(
                getattr(window.image_loader, "should_cancel", None),
                worker.progress.emit,
            )
        worker.finished_load.connect(on_done)
        worker.finished.connect(on_thread_finished)
        self._show_load_status()
        worker.start()

    def _start_render_prewarm(self) -> None:
        from PyQt6 import sip

        window = self.window
        stack = self._render_prewarm_stack
        self._render_prewarm_stack = None
        if (isinstance(window, QObject) and sip.isdeleted(window)) or getattr(window, "_shutting_down", False):
            return
        if (os.environ.get("QT_QPA_PLATFORM") == "offscreen"
                or os.environ.get("OPENFOCUS_DISABLE_PERSISTENT_RENDER_WORKER")):
            return
        render = getattr(window, "render_manager", None)
        if (render is None or getattr(render, "_closing", False) or self._reset_busy()
                or getattr(render, "_pending_storage", None) is not None):
            return
        if stack != (getattr(self, "_load_generation", 0), id(window.raw_images),
                     getattr(window, "source_revision", 0)):
            return
        from core.render_process import prewarm_persistent_worker
        prewarm_persistent_worker()

    def _finish_load_worker(self, worker) -> None:
        if self._load_worker is worker:
            self._load_worker = None
            for name in ("_load_cancel_button", "_load_status_label", "_load_progress_bar"):
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.hide()
            self._start_queued_render(worker)
            if (getattr(worker, "_load_applied", False)
                    and os.environ.get("QT_QPA_PLATFORM") != "offscreen"
                    and not os.environ.get("OPENFOCUS_DISABLE_PERSISTENT_RENDER_WORKER")):
                self._render_prewarm_stack = (getattr(self, "_load_generation", 0),
                                              id(worker._load_stack), worker._load_revision)
                self._render_prewarm_timer.start()
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
                if isinstance(self.window, QObject) and sip.isdeleted(self.window):
                    return
                render._hide_progress_bar()
                render.refresh_action_labels()
                current = (not getattr(self.window, '_shutting_down', False)
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
                    render.last_state = 'cancelled'
                    self.window.statusBar().showMessage(trans.t("msg_render_queue_cancelled"), 5000)
                    return
                render.start_render(force_algorithm=request.get("force_algorithm"))

            QTimer.singleShot(0, start)
            render.refresh_action_labels()
        else:
            render._render_after_load = None
            render.last_state = 'cancelled'
            render._hide_progress_bar()
            render.refresh_action_labels()
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
            window.statusBar().add_activity(self._load_status_label)
            window.statusBar().add_activity(self._load_progress_bar)
            button = QPushButton(trans.t("btn_cancel_generic"))
            button.clicked.connect(self.cancel_load)
            window.statusBar().add_activity(button)
            self._load_cancel_button = button
        self._load_progress_bar.setRange(0, 0)
        self._load_progress_bar.setValue(0)
        button.setEnabled(True)
        button.show()
        self._load_status_label.setText(trans.t("msg_loading_stack"))
        self._load_status_label.show()
        self._load_progress_bar.show()

    def _show_restore_progress(self) -> None:
        worker = self.sender()
        if worker is not None and worker is getattr(self, '_load_worker', None) and not worker.is_cancelled:
            self._load_status_label.setText(trans.t('update_restore_running'))
            self._load_progress_bar.setRange(0, 0)
            self._load_progress_bar.setValue(0)

    def _on_load_progress(self, done: int, total: int) -> None:
        worker = self.sender()
        if (worker is None or worker is not getattr(self, "_load_worker", None)
                or worker.is_cancelled):
            return
        text = trans.t("msg_loading_stack")
        if total > 0:
            bar_total = max(total, done)
            self._load_progress_bar.setRange(0, bar_total)
            self._load_progress_bar.setValue(done)
            text = f"{text} ({done}/{total})"
        else:
            self._load_progress_bar.setRange(0, 0)
            text = f"{text} ({done})"
        self._load_status_label.setText(text)

    def cancel_load(self) -> None:
        worker = getattr(self, "_load_worker", None)
        if worker is not None:
            worker.request_cancel()
            self._load_cancel_button.setEnabled(False)
            self._load_status_label.setText(trans.t("depth_cancelling"))

    def restore_last_stack(self) -> None:
        """Re-load the most recently opened stack (skip the downsample dialog,
        reusing the scale from the previous session)."""
        from utils.settings_store import get_settings, LAST_STACK_FOLDER_KEY, LAST_STACK_MODE_KEY
        settings = get_settings()
        folder = str(settings.value(LAST_STACK_FOLDER_KEY, "") or "")
        linear_raw = settings.value(LAST_STACK_MODE_KEY, "standard") == "linear"
        if folder and os.path.isdir(folder):
            window = self.window
            self._start_load_worker(
                folder=folder,
                scale=getattr(window, "current_scale_factor", 1.0),
                append=False,
                linear_raw=linear_raw,
            )

    def _ask_load_scale(self, *, paths=(), folder=None, video=None, linear_raw=False):
        """Loading scale for a new import, or None if the user cancelled.

        The scale prompt used to appear before *every* load; users who always
        load at full size can now tick "don't ask again" (re-enable under
        Settings), after which their last choice is applied silently."""
        window = self.window
        if not getattr(window, "ask_load_scale", True):
            return float(getattr(window, "load_scale_default", 1.0) or 1.0)
        current_scale = getattr(window, "current_scale_factor", 1.0)
        dlg = _resolve_lazy('DownsampleDialog')(window, initial_scale=current_scale,
                               show_dont_ask=True, source_paths=paths,
                               source_folder=folder, source_video=video,
                               linear_raw=linear_raw)
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

    def _choose_raw_import_mode(self, paths, append=False):
        """Return the import mode, or None when the user cancels."""
        if append and getattr(self.window, "raw_images", None):
            return getattr(self.window, "linear_dng_metadata", None) is not None
        from core.image_loader import is_raw_source
        if not paths or not all(is_raw_source(path) for path in paths):
            return False
        from utils.ui_utils import style_message_box
        box = QMessageBox(self.window)
        style_message_box(box)
        box.setWindowTitle(trans.t("raw_import_mode_title"))
        box.setText(trans.t("raw_import_mode_prompt"))
        box.setInformativeText(trans.t("raw_import_mode_explanation"))
        box.setIcon(QMessageBox.Icon.Question)
        linear = box.addButton(trans.t("raw_import_mode_linear"), QMessageBox.ButtonRole.AcceptRole)
        standard = box.addButton(trans.t("raw_import_mode_standard"), QMessageBox.ButtonRole.ActionRole)
        cancel = box.addButton(trans.t("btn_cancel_generic"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        if box.clickedButton() is linear:
            return True
        if box.clickedButton() is standard:
            return False
        return None

    def _folder_import_paths(self, folder, *, strict=False):
        """Match the ordinary folder loader's supported and excluded sources."""
        from core.image_loader import ImageStackLoader
        from utils.path_identity import same_path
        loader = ImageStackLoader()
        excluded = getattr(self.window, "defect_mask_path", "")
        try:
            with os.scandir(folder) as entries:
                files = [(entry.name, entry.path) for entry in entries
                         if entry.is_file()
                         and os.path.splitext(entry.name)[1].lower() in loader.SUPPORTED_FORMATS
                         and not (excluded and same_path(entry.path, excluded))]
            return [path for _, path in loader._drop_derived_aligned_copies(files)]
        except OSError:
            if strict:
                raise
            return []  # Let the worker report inaccessible folders.

    def load_image_stack(self, folder_path: str, append: bool = False) -> None:
        window = self.window
        from utils.logging_setup import log_action
        log_action("load_stack", folder=folder_path, append=append)

        # 同一文件夹再次导入按"重新加载"处理：追加只会得到帧数翻倍的
        # 重复栈（恢复会话后再导入、拖同一文件夹两次都触发过）
        current_folder = getattr(window, "current_folder_path", "") or ""
        same_folder = bool(current_folder) and (
            os.path.normcase(os.path.abspath(folder_path))
            == os.path.normcase(os.path.abspath(current_folder))
        )
        if same_folder and getattr(window, "linear_dng_metadata", None) is not None:
            linear_raw = True
        else:
            # Appending to a stack inherits its import mode regardless of the folder contents.
            paths = () if append and getattr(window, 'raw_images', None) else self._folder_import_paths(folder_path)
            linear_raw = self._choose_raw_import_mode(paths, append=append)
        if linear_raw is None:
            return
        if append and same_folder:
            append = False

        scale_factor = self._ask_load_scale(folder=folder_path, linear_raw=linear_raw)
        if scale_factor is None:
            return

        # Decode on a background thread so the window stays responsive on
        # large stacks; the continuation runs on the UI thread via the signal.
        self._start_load_worker(
            folder=folder_path, scale=scale_factor, append=append, linear_raw=linear_raw,
            on_success=lambda: window.add_recent_file(folder_path) if hasattr(window, "add_recent_file") else None,
        )

    def load_video_stack(self, video_path: str, append: bool = False) -> None:
        """Load image stack from a video file."""
        window = self.window

        scale_factor = self._ask_load_scale(video=video_path)
        if scale_factor is None:
            return

        self._start_load_worker(
            video=video_path, scale=scale_factor, append=append,
            on_success=lambda: window.add_recent_file(video_path) if hasattr(window, "add_recent_file") else None,
        )

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

    def prompt_and_load_frames(self) -> None:
        """Select image files and load them together as a new stack."""
        from core.image_loader import ImageStackLoader
        from utils.settings_store import get_last_dialog_dir, set_last_dialog_dir
        from utils.sorting import natural_sort_key

        image_exts = " ".join(f"*{ext}" for ext in sorted(ImageStackLoader.SUPPORTED_FORMATS))
        paths, _ = FileDialog.getOpenFileNames(
            self.window, trans.t("action_open_frames"), get_last_dialog_dir(),
            f"{trans.t('action_open_frames')} ({image_exts});;All Files (*)",
            "", FileDialog.Option.HideNameFilterDetails,
        )
        if not paths:
            return
        set_last_dialog_dir(paths[0])
        paths = sorted(paths, key=lambda path: natural_sort_key(os.path.basename(path)))
        linear_raw = self._choose_raw_import_mode(paths, append=False)
        if linear_raw is None:
            return
        scale = self._ask_load_scale(paths=paths, linear_raw=linear_raw)
        if scale is None:
            return
        self._start_load_worker(filepaths=paths, scale=scale, append=False, linear_raw=linear_raw)

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
        from core.app import dispatch_project_open
        if dispatch_project_open(self.window, paths):
            event.acceptProposedAction()
            return

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
                    scale = self._ask_load_scale(folder=paths[0])
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
            scale = self._ask_load_scale(folder=paths[0])
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
            linear_raw = self._choose_raw_import_mode(valid_paths, append=True)
            if linear_raw is None:
                event.ignore()
                return
            scale = self._ask_load_scale(paths=valid_paths, linear_raw=linear_raw)
            if scale is None:
                # user cancelled
                event.ignore()
                return

            # Decode on the load worker like folder loads: dropping a large
            # file set used to freeze the window while it decoded.
            self._start_load_worker(filepaths=valid_paths, scale=scale, append=True, linear_raw=linear_raw)
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
        source_exifs = list(getattr(self.window.image_loader, "source_exifs", ()) or ())
        if len(source_exifs) != len(full_res_images):
            source_exifs = [b""] * len(full_res_images)
        profiles = list(getattr(self.window.image_loader, "source_icc_profiles", ()) or ())
        if len(profiles) != len(full_res_images):
            profiles = [b""] * len(full_res_images)
        return LoadOptions(
            scale_factor=scale_factor,
            filenames=list(filenames),
            base_images=list(full_res_images),
            working_images=list(full_res_images),
            source_exifs=source_exifs,
            source_icc_profiles=profiles,
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
        option_profiles = getattr(options, "source_icc_profiles", None)
        if option_profiles is not None:
            profile_map = option_profiles
        elif profile_map is None:
            profile_map = getattr(getattr(window, "image_loader", None), "source_icc_profiles", ())
        profiles = list(profile_map or ())
        if len(profiles) != len(options.working_images):
            profiles = [b""] * len(options.working_images)
        exifs = list(getattr(options, "source_exifs", ()) or ())
        if len(exifs) != len(options.working_images):
            exifs = [b""] * len(options.working_images)
        previous_profiles = list(getattr(window, "image_icc_profiles", ()) or ())
        if len(previous_profiles) != len(window.raw_images or []):
            previous_profiles = [b""] * len(window.raw_images or [])
        previous_exifs = list(getattr(window, "image_source_exifs", ()) or ())
        if len(previous_exifs) != len(window.raw_images or []):
            previous_exifs = [b""] * len(window.raw_images or [])
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
            window.image_source_exifs = (previous_exifs + exifs
                                         if len(exifs) == len(options.working_images)
                                         else [])
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
                    "exifs": baseline.get("exifs", []) + exifs,
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
            window.image_source_exifs = exifs
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
                "exifs": list(exifs),
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
        if getattr(window, "project_path", None):
            window._mark_project_dirty()

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
        worker = getattr(self, "_load_worker", None)
        if worker is not None:
            worker.request_cancel()
        render = getattr(window, 'render_manager', None)
        if render is not None and render._render_after_load is not None:
            render._render_after_load = None
            render.last_state = 'cancelled'
            render._abort_compare_run()
            render._hide_progress_bar()
            render.refresh_action_labels()
        # A cancelled load can still roll its own loader back after this call.
        # Give the empty workspace a fresh loader so neither that rollback nor
        # the last decoded stack can keep cleared source frames alive here.
        from core.image_loader import ImageStackLoader
        window.image_loader = ImageStackLoader()
        self._stack_baseline = None
        if hasattr(window, "_wipe_source_cache"):
            window._wipe_source_cache = None

        # Also drops the ROI stack, rect and tool button (see reset_roi_selection)
        window.transform_manager.invalidate_processing_results(clear_output_view=True)

        window.raw_images = []
        window.base_images = []
        window.stack_images = []
        window.image_filenames = []
        window.image_source_paths = []
        window.image_source_exifs = []
        window.linear_dng_metadata = None
        window.image_icc_profiles = []
        window.excluded_frames = set()
        window.source_transforms = []
        window.unrecoverable_source_changes = []
        window.source_px_um = None
        self._load_generation = getattr(self, "_load_generation", 0) + 1
        window.base_scale_factor = 1.0
        window.current_display_index = -1
        self._clear_review()

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
        if getattr(window, "project_path", None):
            window._mark_project_dirty()
        from utils import recovery
        recovery.write_snapshot(window)

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
        if self._source_hover is None or self._source_hover.view is not self.window.file_list:
            from controllers.source_hover import SourceHover
            self._source_hover = SourceHover(self, self.window.file_list)
        self._source_hover.invalidate()
        window = self.window
        has_updates_toggle = hasattr(window.file_list, "setUpdatesEnabled")
        if has_updates_toggle:
            window.file_list.setUpdatesEnabled(False)
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
            if has_updates_toggle:
                window.file_list.setUpdatesEnabled(True)

        self.update_source_images_count()

    def sync_slider_from_list(self, row: int) -> None:
        if row >= 0:
            self.window.stack_slider.setValue(row)

    # ------------------------------------------------------------------
    # Context menu & deletions
    # ------------------------------------------------------------------
    def _selected_source_rows(self) -> set[int]:
        window = self.window
        count = len(window.raw_images)
        return {row for item in window.file_list.selectedItems()
                if 0 <= (row := window.file_list.row(item)) < count}

    def show_source_context_menu(self, position: QPoint) -> None:
        window = self.window
        view = window.file_list
        target = view.itemAt(position)
        if target is None:
            return
        if not target.isSelected():
            view.setCurrentItem(target, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        rows = self._selected_source_rows()
        if not rows:
            return
        excluded = getattr(window, "excluded_frames", set())
        menu = QMenu(view)
        menu.setToolTipsVisible(True)
        menu.addAction(self._review_action)
        menu.addSeparator()

        include_action = menu.addAction(trans.t("source_include_selected"))
        include_action.setEnabled(not rows.isdisjoint(excluded))
        include_action.triggered.connect(lambda: self._set_frame_participation(
            self._selected_source_rows(), included=True))
        exclude_action = menu.addAction(trans.t("source_exclude_selected"))
        exclude_action.setEnabled(not rows.issubset(excluded))
        exclude_action.triggered.connect(lambda: self._set_frame_participation(
            self._selected_source_rows(), included=False))
        menu.addSeparator()
        menu.addAction(trans.t("action_delete"), self.delete_selected_source_images)

        menu.exec(view.viewport().mapToGlobal(position))
        menu.deleteLater()

    def delete_source_image(self, item: QListWidgetItem) -> None:
        window = self.window
        row = window.file_list.row(item)
        if row < 0 or row >= len(window.raw_images):
            return
        window.push_undo()

        if len(window.raw_images) <= 1:
            self.clear_image_stack()
            return

        window.transform_manager.invalidate_processing_results(clear_output_view=False, preserve_outputs=True)

        self._pop_sequence(window.image_filenames, row)
        self._pop_sequence(window.raw_images, row)
        self._pop_sequence(getattr(window, "base_images", None), row)
        self._pop_sequence(getattr(window, "image_source_paths", None), row)
        self._pop_sequence(getattr(window, "image_source_exifs", None), row)
        self._pop_sequence(getattr(window, "image_icc_profiles", None), row)
        self._remap_exclusions_after_delete([row])

        if window.raw_images:
            new_index = min(row, len(window.raw_images) - 1)
            window.transform_manager.reload_image_stack(initial_index=new_index)
        else:
            self.clear_image_stack()

    def delete_selected_source_images(self) -> None:
        window = self.window
        rows = sorted(self._selected_source_rows())
        if not rows:
            return

        raw = window.raw_images
        revision = getattr(window, "source_revision", 0)
        frames = tuple(raw)

        confirm = QMessageBox(window)
        from utils.ui_utils import style_message_box
        style_message_box(confirm)
        confirm.setIcon(QMessageBox.Icon.Warning)
        confirm.setWindowTitle(trans.t("msg_confirm_delete_title"))
        confirm.setText(trans.t("msg_confirm_delete_source_text").format(count=len(rows)))
        confirm.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if confirm.exec() != QMessageBox.StandardButton.Yes:
            return
        current = window.raw_images
        if (getattr(window, "source_revision", 0) != revision or current is not raw
                or len(current) != len(frames)
                or any(a is not b for a, b in zip(current, frames))):
            window.statusBar().showMessage(trans.t("msg_delete_source_changed"), 5000)
            return
        window.push_undo()

        remaining = len(window.raw_images) - len(rows)
        if remaining <= 0:
            self.clear_image_stack()
            return

        window.transform_manager.invalidate_processing_results(clear_output_view=False, preserve_outputs=True)

        rows.sort(reverse=True)

        for row in rows:
            self._pop_sequence(window.image_filenames, row)
            self._pop_sequence(window.raw_images, row)
            self._pop_sequence(getattr(window, "base_images", None), row)
            self._pop_sequence(getattr(window, "image_source_paths", None), row)
            self._pop_sequence(getattr(window, "image_source_exifs", None), row)
            self._pop_sequence(getattr(window, "image_icc_profiles", None), row)
        self._remap_exclusions_after_delete(rows)

        if window.raw_images:
            target_index = min(min(rows), len(window.raw_images) - 1)
            window.transform_manager.reload_image_stack(initial_index=target_index)
        else:
            self.clear_image_stack()

    def _remap_exclusions_after_delete(self, rows) -> None:
        removed = set(rows)
        removed_rows = sorted(removed)
        self.window.excluded_frames = {
            i - bisect_left(removed_rows, i)
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
        self._set_frame_participation({row}, included=item.checkState() == Qt.CheckState.Checked)

    def _set_frame_participation(self, rows: set[int], *, included: bool) -> None:
        if not rows:
            return
        window = self.window
        excluded = getattr(window, "excluded_frames", set())
        if (rows.isdisjoint(excluded) if included else rows.issubset(excluded)):
            return
        window.push_undo()
        if included:
            excluded.difference_update(rows)
        else:
            excluded.update(rows)
        window.excluded_frames = excluded
        self.processing_selection_changed()

    def processing_selection_changed(self) -> None:
        window = self.window
        window.source_revision = getattr(window, "source_revision", 0) + 1
        # Prepared fusion inputs belong to the old selection. Dropping the
        # manager's reference leaves any in-flight reader's snapshots alive.
        clear_prepared = getattr(getattr(window, 'render_manager', None),
                                 'clear_prepared_cache', None)
        if callable(clear_prepared):
            clear_prepared()
        # The last completed alignment still belongs to its original ordered
        # selection. Restore it if those same frames are checked again.
        key = getattr(window, 'alignment_request_key', None)
        if key is None:
            window.aligned_images = []
            window.last_alignment_options = None
        window.is_images_aligned = bool(key and tuple(window.included_indices()) == key[1])
        window.transform_manager.reset_roi_selection()
        self._refresh_source_count()
        self.decorate_source_items()
        # Inclusion changes the processing revision but leaves the source
        # frames in place. Replace any preview request invalidated by it.
        if getattr(window, '_source_preview_workers', None):
            window._start_source_previews()
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

        selected_rows = self._selected_source_rows()
        kept = getattr(window, "current_display_index", 0)
        kept = kept if isinstance(kept, int) and 0 <= kept < n else 0
        new_pos = new_order.index(kept)

        window.push_undo()

        def _reorder(seq):
            return [seq[i] for i in new_order]

        window.raw_images = _reorder(window.raw_images)
        window.image_filenames = _reorder(window.image_filenames)
        if getattr(window, "base_images", None):
            window.base_images = _reorder(window.base_images)
        if getattr(window, "image_source_paths", None):
            window.image_source_paths = _reorder(window.image_source_paths)
        if len(getattr(window, "image_source_exifs", ()) or ()) == n:
            window.image_source_exifs = _reorder(window.image_source_exifs)
        if len(getattr(window, "image_icc_profiles", ()) or ()) == n:
            window.image_icc_profiles = _reorder(window.image_icc_profiles)
        excluded = getattr(window, "excluded_frames", None)
        if excluded:
            window.excluded_frames = {row for row, old in enumerate(new_order) if old in excluded}

        # The remapped exclusion set survives cache invalidation.
        window.transform_manager.invalidate_processing_results(
            clear_output_view=False, preserve_outputs=True)
        if window.transform_manager.reload_image_stack(initial_index=new_pos):
            for row in range(n):
                window.file_list.item(row).setSelected(row in selected_rows)

    # === Icon Drag-and-Drop Support Methods ===
    # These methods handle files dropped on app icon/taskbar/dock

    def load_image_stack_from_icon(self, folder_path: str) -> None:
        """Load a folder dropped on the app icon or passed on the command line."""
        self.load_image_stack(folder_path, append=bool(getattr(self.window, "raw_images", None)))

    def load_multiple_folders_from_icon(self, folder_paths: list[str]) -> None:
        """Load multiple folders dropped on app icon."""
        scale = self._ask_load_scale(folder=folder_paths[0] if folder_paths else None)
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
        linear_raw = self._choose_raw_import_mode(file_paths, append=True)
        if linear_raw is None:
            return
        scale = self._ask_load_scale(paths=file_paths, linear_raw=linear_raw)
        if scale is None:
            return

        self._start_load_worker(filepaths=list(file_paths), scale=scale, append=True, linear_raw=linear_raw)
