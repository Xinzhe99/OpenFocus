from typing import Any, List, Optional
from copy import deepcopy
from math import ceil
from time import monotonic


from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from utils import show_message_box, show_warning_box
from utils.image_utils import cv2_to_pixmap
from utils.output_sharpen import DEFAULT_SHARPEN_STRENGTH
from core.multi_focus_fusion import is_ai_fusion_available
from core.render_context import RegistrationDetails, build_render_context
from locales import trans
from core.algorithms import fusion_algorithms, get_fusion, get_registration, alignment_steps
from core.experimental import alignment_is_locked
from ui.algorithm_combo import desktop_fusion_lock_reason, fusion_lock_reason



_STORAGE_UNSET = object()
_RENDER_STAGE_KEYS = {
    'waiting': 'render_stage_waiting', 'prepare': 'render_stage_prepare',
    'align': 'render_stage_align', 'fuse': 'render_stage_fuse',
    'finalize': 'render_stage_finalize', 'publish': 'render_stage_publish',
}


def __getattr__(name):
    """Keep render dependencies importable without loading them at GUI startup."""
    if name == 'RenderProcessWorker':
        from core.render_process import RenderProcessWorker as dependency
    elif name == 'ROIRenderOptionsDialog':
        from dialogs import ROIRenderOptionsDialog as dependency
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = dependency
    return dependency


def _base_frame(method, enabled):
    if not enabled:
        return "none"
    return "first" if method in ("openfocus_a", "openfocus_b", "openfocus_c") else "last"


def _failed_alignment_detail(entries, names, indices):
    reason_keys = {
        'low detector quality': 'alignment_reason_low_quality',
        'motion outside limits': 'alignment_reason_outside_limits',
        'horizontal shift outside limits': 'alignment_reason_shift_x_limit',
        'vertical shift outside limits': 'alignment_reason_shift_y_limit',
        'rotation outside limits': 'alignment_reason_rotation_limit',
        'scale change outside limits': 'alignment_reason_scale_limit',
        'reference alignment fell back': 'alignment_reason_reference_fallback',
        'insufficient features': 'alignment_reason_insufficient_features',
        'insufficient matches': 'alignment_reason_insufficient_matches',
        'homography estimation failed': 'alignment_reason_homography_failed',
        'ECC failed to converge': 'alignment_reason_ecc_failed',
    }
    for entry in entries:
        index = entry.get('frame_index')
        if entry.get('status') != 'failed' or not isinstance(index, int) or not 0 <= index < len(names):
            continue
        number = indices[index] + 1 if index < len(indices) else index + 1
        description = trans.t('alignment_report_frame').format(
            number=number, name=names[index], method=entry.get('method', ''),
            status=trans.t('alignment_report_failed'))
        reason = entry.get('reason')
        if reason:
            text = trans.t(reason_keys[reason]) if reason in reason_keys else reason
            description += f"\n{trans.t('alignment_detail_reason')}: {text}"
        return description
    return ''


class _StorageProbe(QObject):
    completed = pyqtSignal(object)

    def start(self, estimate):
        import threading

        def run():
            from core.frame_storage import assess_scratch, scratch_directory
            try:
                with scratch_directory(separate=True, explicit_directory=None):
                    result = assess_scratch(*estimate)
            except Exception as exc:
                from utils.logging_setup import get_logger
                get_logger(__name__).info('storage preflight unavailable', exc_info=True)
                result = {'status': 'unknown', 'directory': None, 'explicit': False,
                          'reason': str(exc)}
            try:
                self.completed.emit(result)
            except RuntimeError:
                pass  # The window was destroyed while discovery was running.

        threading.Thread(target=run, name='openfocus-storage-preflight', daemon=True).start()


class _StorageCancellation:
    def __init__(self, manager, request):
        self.manager, self.request = manager, request

    def cancel(self):
        self.manager.cancel_storage_preflight(self.request)


class RenderSignals(QObject):
    """Lightweight render lifecycle notifications for UI observers."""

    render_started = pyqtSignal()
    # True when the run produced a result the user can look at.
    render_finished = pyqtSignal(bool)


class RenderManager:
    """Encapsulates render pipeline orchestration for the main window."""

    @classmethod
    def algo_display_name(cls, code: str) -> str:
        return trans.t("radio_no_fusion" if code == "none" else get_fusion(code).label_key)

    def __init__(self, window: Any):
        self.window = window
        self.signals = RenderSignals()
        self._render_active = False
        self.worker: Optional[Any] = None
        self.context_worker = None
        self._context_candidate = None
        self._retiring_context_workers = set()
        self._prepared_cache = None
        self._prepared_cache_key = None
        self._prepared_cache_source = None
        self._compare_mode = False
        self._compare_cancelling = False
        self._compare_queue: List[str] = []
        self._compare_total = 0
        self._compare_current: Optional[str] = None
        # An unaffordable method must not cancel the comparison queue.
        self._compare_failures: List[tuple] = []
        self._compare_token = object()
        self._compare_context = None
        self._substack_mode = False
        self._substack_cancelling = False
        self._substack_queue: List[Any] = []
        self._substack_total = 0
        self._substack_current = None
        self._substack_auto_restack = False
        self._substack_interactive = True
        self._substack_confirmed = False
        self._substack_results: List[Any] = []
        self._substack_token = object()
        self._substack_context = None
        self._substack_chain_timer = None
        self._preview_mode = False
        self._live_preview_timer = None
        # What the *running* render was launched with; see start_render().
        self._render_context: Optional[dict] = None
        # Control-channel observability: the last outcome and tile progress.
        self.last_error: Optional[str] = None
        self.last_result: Optional[dict] = None
        self.last_state: Optional[str] = None
        self._render_interactive = True
        self._control_progress: Optional[tuple] = None
        self.alignment_diagnostics = None
        # Temporary-storage assessment of the last render start, if computed.
        self.last_storage_assessment: Optional[dict] = None
        # Most recent render performance report (memory only, this session).
        self.performance_report: Optional[dict] = None
        # An interactive request made while source frames are loading.  Keep
        # the forced algorithm and comparison ownership, not just a Boolean,
        # so the deferred render cannot be mislabeled or outlive cancellation.
        self._render_after_load: Optional[dict] = None
        self._pending_storage = None
        self._active_queue_job: Optional[Any] = None
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.clear_prepared_cache)
        destroyed = getattr(window, 'destroyed', None)
        if destroyed is not None:
            destroyed.connect(self.clear_prepared_cache)

    def clear_prepared_cache(self):
        """Release the retained temporary snapshots for an obsolete stack."""
        self._prepared_cache = None
        self._prepared_cache_key = None
        self._prepared_cache_source = None

    def _storage_request_identity(self):
        window = self.window
        frames, names, included = window.effective_stack()
        method = window.current_fusion_method()
        from core import frame_storage
        return (id(window.raw_images), getattr(window, 'source_revision', 0),
                tuple(id(frame) for frame in frames), tuple(names), tuple(included),
                tuple(getattr(window, 'image_source_paths', ()) or ()),
                method, window.current_alignment_method(),
                deepcopy(self._processing_parameters(method)),
                tuple(getattr(window, name, None) for name in (
                    'thread_count', 'resource_preference', 'current_scale_factor',
                    'align_cache_enabled', 'collect_alignment_report', 'roi_mode_active',
                    'alignment_roi', 'current_folder_path', 'tile_enabled', 'profile_performance')),
                bool(window.chk_quick_preview.isChecked()), frame_storage._storage_directory,
                self._compare_token)

    def _begin_storage_preflight(self, frames, force_algorithm, interactive, confirmed, roi_choice,
                                 cpu_confirmed):
        request = {'identity': self._storage_request_identity(), 'completion': None,
                   'force_algorithm': force_algorithm, 'interactive': interactive,
                   'confirmed': confirmed, 'cpu_confirmed': cpu_confirmed,
                   'roi_choice': deepcopy(roi_choice), 'previous_state': self.last_state,
                   'previous_error': self.last_error}
        estimate = self._storage_estimate(frames)
        probe = _StorageProbe(self.window)
        request['probe'] = probe
        self._pending_storage = request
        self.last_storage_assessment = None
        probe.completed.connect(lambda result: self._finish_storage_preflight(request, result),
                                Qt.ConnectionType.QueuedConnection)
        self.last_state = 'running'
        self.last_error = None
        self.refresh_action_labels()
        try:
            probe.start(estimate)
        except Exception:
            self._pending_storage = None
            probe.deleteLater()
            self._hide_progress_bar()
            self.refresh_action_labels()
            raise

    def set_storage_completion(self, completion):
        request = self._pending_storage
        if request is None:
            raise RuntimeError('Storage preflight is no longer pending')
        request['completion'] = completion
        return _StorageCancellation(self, request)

    def cancel_storage_preflight(self, request=None):
        pending = self._pending_storage
        if pending is None or (request is not None and request is not pending):
            return False
        self._pending_storage = None
        self.last_state = 'cancelled'
        self._hide_progress_bar()
        self.refresh_action_labels()
        completion = pending['completion']
        if completion is not None:
            completion('cancelled')
        return True

    def cancel_pending_start(self):
        """Invalidate queued and in-progress render starts at their safe boundary."""
        queued = self._render_after_load is not None
        self._stop_compare_timer()
        self._compare_token = object()
        self._compare_queue = []
        self._compare_context = None
        self._compare_current = None
        self._compare_mode = False
        self._stop_substack_timer()
        self._substack_token = object()
        self._substack_queue = []
        self._substack_context = None
        self._substack_current = None
        self._substack_mode = False
        self._substack_cancelling = False
        self._substack_auto_restack = False
        self._substack_results = []
        self._render_after_load = None
        self.cancel_storage_preflight()
        if queued:
            self.last_state = "cancelled"
            self._hide_progress_bar()
            self.refresh_action_labels()

    def _finish_storage_preflight(self, request, assessment):
        request['probe'].deleteLater()
        if self._pending_storage is not request:
            return
        self._pending_storage = None
        try:
            stale = (getattr(self.window, '_shutting_down', False)
                     or self._storage_request_identity() != request['identity']
                     or getattr(getattr(self.window, 'source_manager', None), '_load_worker', None) is not None)
            if stale:
                self.last_state = 'discarded'
                self._hide_progress_bar()
                self.window.statusBar().showMessage(trans.t('msg_render_stack_changed'), 8000)
                self.refresh_action_labels()
                status = 'stack_changed'
            else:
                status = self.start_render(request['force_algorithm'], interactive=request['interactive'],
                                           confirmed=request['confirmed'], roi_choice=request['roi_choice'],
                                           _storage_assessment=assessment,
                                           _cpu_confirmed=request['cpu_confirmed'])
                if status != 'started':
                    if status in ('confirm_storage', 'insufficient_storage'):
                        self.last_state = request['previous_state']
                        self.last_error = request['previous_error']
                    else:
                        self.last_state = 'cancelled' if status == 'cancelled' else None
                    self._hide_progress_bar()
                    self.refresh_action_labels()
        except Exception as exc:
            self.last_state = 'error'
            self.last_error = str(exc)
            self._hide_progress_bar()
            self.refresh_action_labels()
            status = 'error'
        completion = request['completion']
        if completion is not None:
            completion(status)

    def _capture_performance_report(self) -> None:
        """Keep the finished worker's report, including a partial one."""
        report = getattr(self.worker, 'performance_report', None)
        if report:
            self.performance_report = report

    def _alignment_key(self, included, method=None, *, queue_item=None):
        window = self.window
        if queue_item is not None:
            method = queue_item.method
            opts = queue_item.options or {}
            brightness = bool(opts.get("exposure_match", False))
            alignment = queue_item.align_method
            parameters = deepcopy(queue_item.align_params or {})
            reverse_order = bool(opts.get("reverse_order", False))
        else:
            method = method or window.current_fusion_method()
            getter = getattr(window, "brightness_option_for", None)
            brightness = (bool(getter("exposure_match", method)) if callable(getter)
                          else bool(getattr(window, "exposure_match", False)))
            alignment = window.current_alignment_method()
            parameters = window.registration_parameters_for(alignment)
            reverse_order = bool(getattr(window, "stack_reverse_order", False))
        from core.workers import _ordered_openfocus_registration
        reverse_registration = _ordered_openfocus_registration(
            alignment, method, 'desc' if reverse_order else 'asc')
        # Checkbox edits advance source_revision to reject in-flight work.
        # The ordered inclusion below already identifies those edits for reuse.
        return (getattr(window, "source_content_revision", 0), tuple(included),
                tuple(id(window.raw_images[i]) for i in included),
                alignment_steps(alignment), tuple(sorted(parameters.items())),
                getattr(window, "alignment_roi", None),
                getattr(window, "reg_downscale_width", None), brightness,
                reverse_order,
                bool(getattr(window, "defect_mask_path", "") or getattr(window, "sensor_denoise", False)),
                'reverse-registration-v1' if reverse_registration else None)

    def _source_exif_snapshot(self, included) -> tuple:
        """Per-included-frame EXIF, aligned with the current source list."""
        window = self.window
        raw = getattr(window, "raw_images", ()) or ()
        exifs = tuple(getattr(window, "image_source_exifs", ()) or ())
        if len(exifs) != len(raw):
            return ()
        return tuple(bytes(exifs[index]) if 0 <= index < len(exifs) else b""
                     for index in included)

    def _processing_parameters(self, method, *, queue_item=None):
        window = self.window
        values = {key: getattr(window, key, None) for key in (
            "tile_enabled", "tile_block_size", "tile_overlap", "tile_threshold",
            "reg_downscale_width", "ai_batch_size", "stack_reverse_order", "use_gpu")}
        if queue_item is not None:
            opts = queue_item.options or {}
            values.update(deepcopy(queue_item.method_params or {}))
            values["registration_parameters"] = deepcopy(queue_item.align_params or {})
            values["exposure_match"] = bool(opts.get("exposure_match", False))
            values["base_correct"] = bool(opts.get("base_correct", False))
            values["stack_reverse_order"] = bool(opts.get("reverse_order", False))
            values["sharpen_output"] = bool(opts.get("sharpen_output", False))
            values["sharpen_strength"] = opts.get("sharpen_strength", DEFAULT_SHARPEN_STRENGTH)
        else:
            values.update(window.fusion_parameters_for(method))
            values["registration_parameters"] = window.registration_parameters_for(
                window.current_alignment_method())
            values["exposure_match"] = window.brightness_option_for("exposure_match", method)
            values["base_correct"] = window.brightness_option_for("base_correct", method)
            values["sharpen_output"] = bool(getattr(window, "sharpen_output", False))
            values["sharpen_strength"] = getattr(window, "sharpen_strength", DEFAULT_SHARPEN_STRENGTH)
        if method in ('openfocus_a', 'openfocus_b', 'openfocus_c'):
            values['tile_enabled'] = False
        elif method != 'none':
            from core.multi_focus_fusion import _DEFAULT_TILE_THRESHOLD
            values['tile_enabled'] = bool(values['tile_enabled'] and (
                get_fusion(method).external_tiling_default
                or values['tile_threshold'] != _DEFAULT_TILE_THRESHOLD))
        values["defect_mask_path"] = getattr(window, "defect_mask_path", "")
        values["sensor_denoise"] = bool(getattr(window, "sensor_denoise", False))
        values["base_frame"] = _base_frame(method, values["base_correct"])
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

    def _ai_confirmation_needed(self, force_algorithm: str | None = None) -> bool:
        """True when starting a CPU-only AI render would ask the user first.

        Kept separate from the dialog so the control channel can answer
        ``confirm_required`` without opening a modal box.
        """
        if force_algorithm is not None:
            return False
        window = self.window
        try:
            algorithm = window.current_fusion_method()
        except Exception:
            return False
        if (algorithm == "none" or not get_fusion(algorithm).neural
                or not getattr(window, "use_gpu", True)):
            # A user who explicitly disabled GPU acceleration has already
            # been told; the setting is the opt-out.
            return False

        from utils.settings_store import (AI_CPU_WARNING_DISMISSED_KEY,
                                          get_settings)
        settings = get_settings()
        if settings.value(AI_CPU_WARNING_DISMISSED_KEY, False, type=bool):
            return False

        from core.multi_focus_fusion import is_ai_fusion_available, neural_device_kind
        if not is_ai_fusion_available():
            return False  # The render itself reports a missing runtime.
        if neural_device_kind() not in (None, "N/A"):
            return False  # an accelerator is available: nothing to warn about
        return True

    def _confirm_cpu_ai_render(self, force_algorithm: str | None = None) -> bool:
        """Warn before starting a CPU-only AI render; True means go ahead.

        StackMFF can take a long time on CPU with a high-resolution or deep
        stack. Users with no accelerator are told once, with a
        "don't show this again" opt-out, before they start a long render.

        Compare-all drives methods programmatically and must not stop for a
        modal box mid-run, so it is skipped.
        """
        if not self._ai_confirmation_needed(force_algorithm):
            return True
        window = self.window
        algorithm = window.current_fusion_method()
        from utils.settings_store import (AI_CPU_WARNING_DISMISSED_KEY,
                                          get_settings)
        settings = get_settings()
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

    def _storage_preflight(self, force_algorithm: str | None, frames) -> Optional[dict]:
        """Assess temporary storage for a full render; no UI, no allocation.

        Drafts, ROI renders and compare-all runs are skipped: they are small or
        driven programmatically. An estimate that cannot be made returns None
        (logged), so a render never prompts on a guess.
        """
        window = self.window
        preview = getattr(window, 'chk_quick_preview', None)
        if (force_algorithm is not None or getattr(window, 'roi_mode_active', False)
                or (preview is not None and preview.isChecked())):
            return None
        try:
            from core.frame_storage import assess_scratch
            return assess_scratch(*self._storage_estimate(frames))
        except Exception:
            from utils.logging_setup import get_logger
            get_logger(__name__).info('storage preflight unavailable', exc_info=True)
            return None

    def _storage_estimate(self, frames):
        """(minimum, expected, optional, source paths) for the current settings."""
        from core.storage_preflight import render_storage_estimate
        window = self.window
        fusion = window.current_fusion_method()
        getter = getattr(window, 'brightness_option_for', None)
        exposure = (bool(getter('exposure_match', fusion)) if callable(getter)
                    else bool(getattr(window, 'exposure_match', False)))
        return render_storage_estimate(
            frames, algorithm=None if fusion == 'none' else fusion,
            alignment=window.current_alignment_method(), need_fusion=fusion != 'none',
            exposure_match=exposure,
            cleanup=bool(getattr(window, 'defect_mask_path', '')
                         or getattr(window, 'sensor_denoise', False)))

    def _storage_message(self, assessment: dict) -> str:
        from core.frame_storage import readable_bytes
        neutral = assessment['directory'] is None
        text = trans.t('storage_low_text').format(
            path=assessment['directory'] or '—',
            free='—' if neutral else readable_bytes(assessment['free_bytes']),
            available=(trans.t('storage_available_upper_bound').format(
                size=readable_bytes(assessment['available_bytes'])) if neutral
                else readable_bytes(assessment['available_bytes'])),
            required=readable_bytes(assessment['required_bytes']))
        if assessment['status'] == 'critical':
            text += '\n\n' + trans.t('storage_low_critical').format(
                minimum=readable_bytes(assessment['minimum_bytes']))
        return text

    def _confirm_storage(self, assessment: dict, frames) -> Optional[dict]:
        """Ask how to proceed with low temporary storage; None cancels the render.

        Returns ``{'directory': path or None, 'keep': bool}``. A chosen
        directory is validated and re-assessed; a critical result never offers
        Continue, because the job's own admission would reject it anyway.
        """
        from PyQt6.QtWidgets import QMessageBox
        from core.frame_storage import assess_scratch, validate_storage_directory
        from utils import style_message_box
        from utils.file_dialogs import FileDialog
        window = self.window
        chosen = None
        while True:
            box = QMessageBox(window)
            style_message_box(box)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle(trans.t('storage_low_title'))
            box.setText(self._storage_message(assessment))
            choose = box.addButton(trans.t('storage_choose'), QMessageBox.ButtonRole.ActionRole)
            choose.setProperty('controlId', 'btn_storage_choose')
            proceed = None
            if assessment['status'] != 'critical':
                proceed = box.addButton(trans.t('storage_continue'), QMessageBox.ButtonRole.AcceptRole)
                proceed.setProperty('controlId', 'btn_storage_continue')
            cancel = box.addButton(trans.t('storage_cancel'), QMessageBox.ButtonRole.RejectRole)
            cancel.setProperty('controlId', 'btn_storage_cancel')
            box.setDefaultButton(choose)
            box.exec()
            clicked = box.clickedButton()
            if proceed is not None and clicked is proceed:
                return {'directory': chosen, 'keep': chosen is not None and self._keep_storage(chosen)}
            if clicked is not choose:
                return None
            path = FileDialog.getExistingDirectory(window, trans.t('storage_choose'),
                                                   assessment['directory'] or '')
            if not path:
                continue
            try:
                path = validate_storage_directory(path)
            except OSError as exc:
                show_warning_box(window, trans.t('storage_low_title'),
                                 trans.t('advanced_storage_invalid').format(error=str(exc)))
                continue
            chosen, assessment = path, assess_scratch(*self._storage_estimate(frames),
                                                      directory=path)
            self.last_storage_assessment = assessment
            if assessment['status'] in ('ok', 'unknown'):
                return {'directory': chosen, 'keep': self._keep_storage(chosen)}

    def _keep_storage(self, path: str) -> bool:
        from PyQt6.QtWidgets import QMessageBox
        from utils import style_message_box
        box = QMessageBox(self.window)
        style_message_box(box)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(trans.t('storage_keep_title'))
        box.setText(trans.t('storage_keep_text').format(path=path))
        keep = box.addButton(trans.t('storage_keep_yes'), QMessageBox.ButtonRole.YesRole)
        keep.setProperty('controlId', 'btn_storage_keep')
        once = box.addButton(trans.t('storage_keep_no'), QMessageBox.ButtonRole.NoRole)
        once.setProperty('controlId', 'btn_storage_once')
        box.setDefaultButton(once)
        box.exec()
        return box.clickedButton() is keep

    def _apply_storage_choice(self, choice: dict) -> Optional[str]:
        """Persist a kept directory like Advanced Settings; return a one-render override."""
        directory = choice.get('directory')
        if directory is None or not choice.get('keep'):
            return directory
        from core.frame_storage import configure_storage_directory
        configure_storage_directory(directory)
        self.window.storage_directory = directory
        persist = getattr(self.window, 'persist_settings', None)
        if callable(persist):
            persist()
        return None

    def _memory_summary(self, estimate):
        return trans.t('render_memory_summary').format(
            method=self.algo_display_name(estimate.algorithm), frames=estimate.frames,
            width=estimate.width, height=estimate.height)

    def _memory_details(self, estimate):
        from core.frame_storage import readable_bytes
        from core.resource_policy import ram_ceiling, vram_budget
        from fusion_methods.neural_runtime import display_name

        snapshot = estimate.snapshot
        unknown = trans.t('render_memory_unknown')
        lines = [self._memory_summary(estimate)]
        lines.append(trans.t('render_memory_ram').format(
            available=readable_bytes(snapshot.available_ram) if snapshot.ram_known else unknown,
            budget=readable_bytes(ram_ceiling(snapshot)) if snapshot.ram_known else unknown))
        if snapshot.commit_headroom is not None:
            lines.append(trans.t('render_memory_commit').format(
                size=readable_bytes(snapshot.commit_headroom)))
        for key, amount in (('source', estimate.source_ram),
                            ('registration', estimate.registration_ram),
                            ('fusion', estimate.fusion_ram)):
            if amount:
                lines.append(trans.t('render_memory_' + key).format(size=readable_bytes(amount)))
        if estimate.tile:
            lines.append(trans.t('render_memory_tile').format(
                tile=estimate.tile, whole=readable_bytes(estimate.whole_ram)))
        if estimate.vram_required:
            lines.append(trans.t('render_memory_vram').format(
                device=display_name(estimate.accelerator),
                available=readable_bytes(snapshot.free_vram) if snapshot.total_vram else unknown,
                budget=readable_bytes(vram_budget(snapshot)) if snapshot.total_vram else unknown,
                required=readable_bytes(estimate.vram_required)))
        if estimate.gpu_fallback:
            lines.append(trans.t('render_memory_fallback_plan').format(
                device=display_name(estimate.accelerator)))
        elif estimate.device is not None:
            lines.append(trans.t('render_memory_device').format(device=display_name(estimate.device)))
        lines.append(trans.t('render_memory_snapshot'))
        return '\n'.join(lines)

    def _memory_guidance(self, estimate):
        text = trans.t('render_memory_guidance')
        if estimate and estimate.tiling_supported:
            text += '\n' + trans.t('render_memory_tiling_guidance')
        if estimate and estimate.gpu_fallback:
            text += '\n' + trans.t('render_memory_gpu_guidance')
        return text

    def _confirm_memory(self, estimate):
        from utils import style_message_box

        box = style_message_box(QMessageBox(self.window))
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(trans.t('render_memory_title'))
        box.setText(trans.t('render_memory_warning' if estimate.ram_short
                            else 'render_memory_fallback_warning'))
        box.setInformativeText(self._memory_details(estimate) + '\n\n' + self._memory_guidance(estimate))
        proceed = box.addButton(trans.t('btn_render'), QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(trans.t('btn_cancel_generic'), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is proceed

    def _estimate_memory(self, frames, method, parameters, roi_rect, register, alignment=None):
        from core.render_preflight import estimate_render_memory
        from utils.logging_setup import get_logger

        roi = (tuple(int(value) for value in
                     (roi_rect.x(), roi_rect.y(), roi_rect.width(), roi_rect.height()))
               if roi_rect is not None else None)
        window = self.window
        try:
            return estimate_render_memory(
                frames, method, parameters, preview=self._preview_mode, roi=roi, register=register,
                alignment=alignment,
                use_gpu=getattr(window, 'use_gpu', True),
                tile_enabled=getattr(window, 'tile_enabled', None),
                tile_block=getattr(window, 'tile_block_size', None),
                tile_threshold=getattr(window, 'tile_threshold', None))
        except Exception:
            get_logger(__name__).info('render memory estimate unavailable', exc_info=True)
            return None

    def _compare_button(self):
        return getattr(self.window, "btn_compare_all", None)

    def _set_compare_button(self, mode: str, reason: str = '') -> None:
        """Paint the Compare-all button for the state it is actually in.

        While a comparison runs it is the cancel affordance, so it must not
        keep saying "Render all methods…"; while a single render owns the
        pipeline it is disabled instead of silently returning when clicked.
        """
        button = self._compare_button()
        if button is None:
            return
        if mode == "cancel":
            button.setEnabled(True)
            button.setText(trans.t('compare_cancel'))
            button.setToolTip(trans.t('btn_cancel_render'))
        elif mode == "cancelling":
            button.setEnabled(False)
            button.setText(trans.t('depth_cancelling'))
            button.setToolTip(trans.t('depth_cancelling'))
        elif mode == "disabled":
            button.setEnabled(False)
            button.setText(trans.t('btn_compare_all'))
            button.setToolTip(
                f"{trans.t('btn_compare_all')}\n"
                f"{reason or trans.t('btn_compare_all_hint')}")
        else:
            button.setEnabled(True)
            button.setText(trans.t('btn_compare_all'))
            button.setToolTip(
                f"{trans.t('btn_compare_all')}\n"
                f"{trans.t('btn_compare_all_hint')}")
        more_btn = getattr(self.window, "btn_render_more", None)
        if more_btn is not None and hasattr(more_btn, "set_active_indicator"):
            more_btn.set_active_indicator(mode in ("cancel", "cancelling"))

    def _alignment_lock_reason(self):
        method = getattr(self.window, 'current_alignment_method', lambda: 'none')()
        return trans.t('experimental_locked_hint') if alignment_is_locked(self.window, method) else ''

    def _render_lock_reason(self, force_algorithm=None):
        method = force_algorithm or getattr(self.window, 'current_fusion_method', lambda: 'none')()
        return desktop_fusion_lock_reason(self.window, method) or self._alignment_lock_reason()

    def _render_worker_active(self) -> bool:
        worker = self.worker
        # Keep ownership until the GUI processes the queued result or error.
        return worker is not None and (getattr(self, '_render_active', False) or worker.isRunning())

    def is_pipeline_busy(self) -> bool:
        """True when a render, storage preflight, compare run, or context worker owns the pipeline."""
        return bool(
            getattr(self, "_render_pending", False)
            or self._pending_storage is not None
            or self._compare_mode
            or getattr(self, "_substack_mode", False)
            or self._render_after_load is not None
            or getattr(self, "context_worker", None) is not None
            or self._render_worker_active()
        )

    def start_queued_render(self, item) -> str:
        """Execute a frozen RenderQueueItem without blocking parameter controls."""
        status = self.start_render(
            force_algorithm=item.method,
            interactive=False,
            confirmed=True,
            queue_item=item,
        )
        return status

    def cancel_queued_render(self, item=None) -> None:
        """Cancel the currently running queued render job if active."""
        if item is not None and self._active_queue_job is not item:
            if (self._active_queue_job is None
                    or getattr(self._active_queue_job, "job_id", None) != getattr(item, "job_id", None)):
                return
        context_worker = getattr(self, "context_worker", None)
        if context_worker is not None and not getattr(context_worker, "is_cancelled", False):
            context_worker.cancel()
            self.refresh_action_labels()
            self._show_progress_bar(trans.t('depth_cancelling'))
            return
        if self._render_worker_active() and not getattr(self.worker, "is_cancelled", False):
            self._cancel_worker()
            self.dismiss_live_preview()
            self.refresh_action_labels()
            self._show_progress_bar(trans.t('depth_cancelling'))
            return
        # No active worker to acknowledge cancellation: notify the queue directly.
        active = self._active_queue_job
        if active is not None:
            self._active_queue_job = None
            queue = getattr(self.window, "render_queue", None)
            if queue is not None:
                queue.notify_job_cancelled(active.job_id)

    def _notify_queue_idle(self) -> None:
        if self._compare_mode or getattr(self, "_substack_mode", False) or self._active_queue_job is not None:
            return
        queue = getattr(self.window, "render_queue", None)
        if queue is not None and hasattr(queue, "notify_pipeline_idle"):
            queue.notify_pipeline_idle()

    def _render_is_cancelling(self) -> bool:
        return bool(getattr(self, '_compare_cancelling', False)
                    or getattr(self, '_substack_cancelling', False)
                    or getattr(self.worker, 'is_cancelled', False)
                    or getattr(getattr(self, 'context_worker', None), 'is_cancelled', False))

    def refresh_action_labels(self) -> None:
        """Re-apply render/compare captions after a language or state change.

        A language switch used to reset a running comparison's cancel button
        to "Render all methods…" (and re-enable it over a single render).
        """
        window = self.window
        render = getattr(window, "btn_render", None)
        busy = self._render_worker_active() or getattr(self, 'context_worker', None) is not None
        if self._render_after_load is not None or self._pending_storage is not None:
            self._show_progress_bar(trans.t('render_stage_loading'))
            if render is not None:
                render.setEnabled(True)
                render.setText(trans.t('btn_cancel_render'))
            self._set_compare_button('cancel' if self._compare_mode else 'disabled')
        elif self._render_is_cancelling():
            if render is not None:
                render.setEnabled(False)
                render.setText(trans.t('depth_cancelling'))
            self._set_compare_button('cancelling' if getattr(self, '_compare_cancelling', False) else 'disabled')
        elif getattr(self, '_compare_mode', False):
            if render is not None:
                render.setEnabled(False)
                render.setText(trans.t('btn_render_processing'))
            self._set_compare_button("cancel")
        elif getattr(self, '_substack_mode', False):
            if render is not None:
                render.setEnabled(True)
                render.setText(trans.t('btn_cancel_render'))
            self._set_compare_button("disabled")
        elif busy:
            if render is not None:
                render.setEnabled(True)
                render.setText(trans.t('btn_cancel_render'))
            self._set_compare_button("disabled")
        else:
            if render is not None:
                reason = self._render_lock_reason()
                render.setEnabled(not reason)
                render.setToolTip(reason)
                render.setText(trans.t('btn_render'))
            reason = self._alignment_lock_reason()
            self._set_compare_button('disabled' if reason else 'idle', reason)

    def _stop_compare_timer(self) -> None:
        timer = getattr(self, "_compare_chain_timer", None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
            self._compare_chain_timer = None

    def _queue_next_compare(self) -> None:
        self._stop_compare_timer()
        token = self._compare_token
        from PyQt6.QtCore import QObject
        timer = self._compare_chain_timer = QTimer(
            self.window if isinstance(self.window, QObject) else None)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._start_next_compare()
                              if token is self._compare_token else None)
        timer.start(400)

    def _abort_compare_run(self) -> None:
        """Abort a chained comparison whose next render failed to start."""
        self._stop_compare_timer()
        self._compare_token = object()
        self._compare_context = None
        if getattr(self, '_compare_mode', False):
            self._compare_mode = False
            self._compare_queue = []
            self._compare_current = None
            self._restore_ui_controls()

    _RENDER_SETTING_NAMES = (
        "chk_quick_preview", "chk_exposure_match", "chk_base_correct",
        "chk_reverse_order", "chk_sharpen_output", "btn_render_settings",
    )

    def _render_setting_widgets(self):
        """Yield the render-setting widgets even on duck-typed test hosts."""
        window = self.window
        components = getattr(window, "right_panel_components", None)
        for name in self._RENDER_SETTING_NAMES:
            widget = getattr(window, name, None)
            if widget is None and components is not None:
                widget = getattr(components, name, None)
            if widget is not None:
                yield widget

    def _set_render_settings_enabled(self, enabled: bool) -> None:
        """Lock every setting that would invalidate an in-flight result."""
        for widget in self._render_setting_widgets():
            try:
                widget.setEnabled(bool(enabled))
            except Exception:
                pass

    def _stop_substack_timer(self) -> None:
        timer = getattr(self, "_substack_chain_timer", None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
            self._substack_chain_timer = None

    def _queue_next_substack(self) -> None:
        self._stop_substack_timer()
        token = self._substack_token
        timer = self._substack_chain_timer = QTimer(
            self.window if isinstance(self.window, QObject) else None)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._start_next_substack()
                              if token is self._substack_token else None)
        timer.start(50)

    def _abort_substack_run(self) -> None:
        """Abort a queued substack run and unlock UI controls."""
        self._stop_substack_timer()
        self._substack_token = object()
        self._substack_context = None
        was_active = bool(getattr(self, '_substack_mode', False))
        self._substack_mode = False
        self._substack_cancelling = False
        self._substack_queue = []
        self._substack_current = None
        self._substack_auto_restack = False
        self._substack_results = []
        if was_active:
            self._restore_ui_controls()
            self._notify_queue_idle()

    def cancel_substack_render(self) -> None:
        """Cancel an active or queued substack render cleanly."""
        if not getattr(self, '_substack_mode', False):
            return
        self._substack_queue = []
        self._substack_auto_restack = False
        self._stop_substack_timer()
        self._substack_token = object()
        context_worker = getattr(self, "context_worker", None)
        running = context_worker is not None or self._render_worker_active()
        if running:
            self._substack_cancelling = True
            if context_worker is not None:
                context_worker.cancel()
            else:
                self._cancel_worker()
            self.dismiss_live_preview()
            self.refresh_action_labels()
            self._show_progress_bar(trans.t('depth_cancelling'))
        else:
            self._abort_substack_run()
            self.last_state = 'cancelled'
            self._hide_progress_bar()
            self.window.statusBar().showMessage(trans.t('msg_render_cancelled'), 4000)

    def start_substack_render(
        self,
        ranges,
        *,
        auto_restack: bool = False,
        interactive: bool = True,
        confirmed: bool = False,
    ) -> str:
        """Sequentially render overlapping substacks and keep every output."""
        window = self.window
        if (getattr(self, "_render_pending", False) or self._render_after_load is not None
                or self._pending_storage is not None
                or getattr(self, 'context_worker', None) is not None
                or self._render_worker_active()
                or getattr(self, '_compare_mode', False)
                or getattr(self, '_substack_mode', False)):
            if interactive:
                window.statusBar().showMessage(trans.t('substack_busy'), 3000)
            return "busy"

        queue = list(ranges or ())
        if not queue:
            return "no_frames"

        if len(window.effective_stack()[0]) < 2:
            if interactive:
                show_warning_box(window, trans.t("substack_dialog_title"), trans.t("substack_need_frames"))
            return "no_frames"

        if window.current_fusion_method() == "none":
            if interactive:
                show_warning_box(window, trans.t("substack_dialog_title"), trans.t("substack_need_fusion_method"))
            return "no_fusion_method"

        reason = self._render_lock_reason()
        if reason:
            if interactive:
                show_warning_box(window, trans.t("msg_error"), reason)
            return "unavailable"

        self._substack_queue = queue
        self._substack_total = len(queue)
        self._substack_current = None
        self._substack_auto_restack = bool(auto_restack)
        self._substack_interactive = bool(interactive)
        self._substack_confirmed = bool(confirmed)
        self._substack_results = []
        self._stop_substack_timer()
        self._substack_token = object()
        self._substack_context = None
        self._substack_cancelling = False
        self._substack_mode = True
        self.refresh_action_labels()
        return self._start_next_substack()

    def _start_next_substack(self) -> str:
        if not self._substack_mode or not self._substack_queue:
            return "idle"
        if self._stack_changed_since(self._substack_context):
            self._abort_substack_run()
            self.window.statusBar().showMessage(trans.t('msg_render_stack_changed'), 8000)
            return "stack_changed"
        sr = self._substack_queue.pop(0)
        self._substack_current = sr
        self.window.statusBar().showMessage(
            trans.t("substack_progress").format(
                i=sr.index, n=sr.total, start=sr.display_start, end=sr.display_end
            ),
            5000,
        )
        return self.start_render(
            interactive=self._substack_interactive,
            confirmed=self._substack_confirmed,
            _substack_range=sr,
        )

    def _finish_substack_run(self) -> None:
        """Complete a substack run and optionally stack the kept substack outputs."""
        window = self.window
        results = list(self._substack_results)
        auto_restack = bool(self._substack_auto_restack)
        interactive = bool(self._substack_interactive)
        self._substack_mode = False
        self._substack_cancelling = False
        self._substack_queue = []
        self._substack_current = None
        self._substack_auto_restack = False
        self._substack_results = []
        self._substack_context = None
        self._stop_substack_timer()
        self._substack_token = object()
        self._restore_ui_controls()

        if auto_restack and len(results) >= 2:
            method_name = self.algo_display_name(window.current_fusion_method())
            composite_label = trans.t("substack_composite_label").format(
                count=len(results), method=method_name
            )
            self.start_stack_outputs(
                results,
                composite_label=composite_label,
                interactive=interactive,
                confirmed=True,
                _from_substack_count=len(results),
            )
            return

        window.statusBar().showMessage(
            trans.t("substack_done").format(count=len(results)), 10000
        )
        from PyQt6.QtWidgets import QWidget
        if isinstance(window, QWidget):
            QApplication.alert(window)
        self._notify_queue_idle()

    def start_stack_outputs(
        self,
        records,
        *,
        composite_label: str | None = None,
        interactive: bool = True,
        confirmed: bool = False,
        _from_substack_count: int | None = None,
    ) -> str:
        """Directly fuse 2+ selected OutputRecord pixel buffers into a composite output."""
        window = self.window
        if (getattr(self, "_render_pending", False) or self._render_after_load is not None
                or self._pending_storage is not None
                or getattr(self, 'context_worker', None) is not None
                or self._render_worker_active()
                or getattr(self, '_compare_mode', False)
                or getattr(self, '_substack_mode', False)):
            if interactive:
                window.statusBar().showMessage(trans.t('btn_render_processing'), 3000)
            return "busy"

        valid_records = [
            rec for rec in (records or ())
            if rec is not None and getattr(rec, "kind", "fusion") != "aligned"
            and getattr(rec, "image", None) is not None
        ]
        if len(valid_records) < 2:
            if interactive:
                show_warning_box(
                    window,
                    trans.t("action_stack_selected_outputs"),
                    trans.t("stack_outputs_need_two"),
                )
            return "no_frames"

        method = window.current_fusion_method()
        if method == "none":
            if interactive:
                show_warning_box(
                    window,
                    trans.t("action_stack_selected_outputs"),
                    trans.t("stack_outputs_need_fusion_method"),
                )
            return "no_fusion_method"

        reason = desktop_fusion_lock_reason(window, method)
        if reason:
            if interactive:
                show_warning_box(window, trans.t("msg_error"), reason)
            return "unavailable"

        # Preserve shared linear RAW metadata when every selected output shares
        # linear metadata; otherwise develop linear inputs to sRGB before fusing.
        linear_metas = [getattr(rec, "linear_dng_metadata", None) for rec in valid_records]
        raw_images = []
        if all(meta is not None for meta in linear_metas):
            shared_linear_metadata = deepcopy(linear_metas[0])
            raw_images = [rec.image for rec in valid_records]
        else:
            shared_linear_metadata = None
            from core.source_frames import resolve_frame
            for rec, meta in zip(valid_records, linear_metas):
                if meta is not None:
                    from utils.linear_dng import develop_linear
                    raw_images.append(develop_linear(resolve_frame(rec.image), meta))
                else:
                    raw_images.append(rec.image)

        from core.substack_utils import prepare_output_images_for_stacking
        prepared_images, needs_alignment = prepare_output_images_for_stacking(raw_images)

        if needs_alignment:
            ws_align = window.current_alignment_method()
            effective_alignment = (
                ws_align
                if ws_align != "none" and not alignment_is_locked(window, ws_align)
                else "ecc"
            )
        else:
            effective_alignment = "none"

        source_names = tuple(
            rec.label or f"output_{idx + 1}" for idx, rec in enumerate(valid_records)
        )
        source_ids = tuple(
            str(getattr(rec, "output_id", "") or "")
            for rec in valid_records
            if getattr(rec, "output_id", "")
        )
        label = composite_label or trans.t("stack_outputs_label").format(
            count=len(valid_records),
            method=self.algo_display_name(method),
        )
        spec = {
            "images": prepared_images,
            "filenames": list(source_names),
            "source_output_names": source_names,
            "source_output_ids": source_ids,
            "alignment": effective_alignment,
            "composite_label": label,
            "linear_dng_metadata": shared_linear_metadata,
            "icc_profile": getattr(valid_records[0], "icc_profile", b"") or b"",
            "from_substack_count": _from_substack_count,
        }
        window.statusBar().showMessage(
            trans.t("stack_outputs_progress").format(count=len(valid_records)), 5000
        )
        return self.start_render(
            interactive=interactive,
            confirmed=confirmed,
            _stack_outputs=spec,
        )

    def _restore_ui_controls(self) -> None:
        """Re-enable every control disabled by start_render (safe to call anywhere)."""
        window = self.window
        self._compare_cancelling = False
        self._substack_cancelling = False
        if getattr(self, '_compare_mode', False):
            # A comparison queue is still running - keep controls locked and
            # show progress instead. The Compare button stays enabled because
            # it is the only way to stop the run.
            self._set_render_settings_enabled(False)
            window.btn_render.setEnabled(False)
            done = self._compare_total - len(self._compare_queue)
            window.btn_render.setText(
                f"{trans.t('msg_compare_progress').format(i=done + 1, n=self._compare_total, method=trans.t('btn_render_processing'))}")
            self._set_compare_button("cancel")
            return
        if getattr(self, '_substack_mode', False):
            self._set_render_settings_enabled(False)
            window.btn_render.setEnabled(True)
            window.btn_render.setText(trans.t('btn_cancel_render'))
            self._set_compare_button("disabled")
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
        self._set_render_settings_enabled(True)
        try:
            window.btn_reset.setEnabled(True)
        except Exception:
            pass
        self.refresh_action_labels()

    def start_compare_all(self) -> None:
        """Render the stack once per available fusion method so the results
        can be compared side by side (output history + Wipe side B).

        Clicking again while a run is queued or rendering cancels it."""
        window = self.window

        if self._compare_mode:
            # Cancel: RenderWorker polls is_cancelled between its stages, and
            # the chained _start_next_compare no-ops on the emptied queue.
            self._compare_mode = False
            self._compare_queue = []
            self._compare_current = None
            self._stop_compare_timer()
            self._compare_token = object()
            self._compare_context = None
            queued = self._render_after_load
            if queued is not None and queued.get("compare", False):
                self._render_after_load = None
                self.last_state = 'cancelled'
                self._hide_progress_bar()
            context_worker = getattr(self, "context_worker", None)
            running = context_worker is not None or self._render_worker_active()
            if running:
                # Keep the truth visible until the worker acknowledges the
                # cancel: no enabled button that would silently do nothing.
                self._compare_cancelling = True
                if context_worker is not None:
                    context_worker.cancel()
                else:
                    self._cancel_worker()
                self.refresh_action_labels()
                self._show_progress_bar(trans.t('depth_cancelling'))
            else:
                # Between two queued renders nothing is running, so the UI has
                # to be unlocked here instead of by the cancel callback.
                self._restore_ui_controls()
                window.statusBar().showMessage(trans.t('msg_render_cancelled'), 4000)
            return

        if (getattr(self, "_render_pending", False) or self._render_after_load is not None
                or self._pending_storage is not None):
            key = 'msg_render_queued_after_load' if self._render_after_load is not None else 'btn_render_processing'
            window.statusBar().showMessage(trans.t(key), 3000)
            return

        if getattr(self, 'context_worker', None) is not None or self._render_worker_active() or getattr(self, '_substack_mode', False):
            # A single render owns the pipeline. The button is disabled in
            # that state, but keep the programmatic path truthful too.
            window.statusBar().showMessage(trans.t('btn_render_processing'), 3000)
            return
 
        if len(window.effective_stack()[0]) < 2:
            show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_render_need_images_text"))
            return

        reason = self._alignment_lock_reason()
        if reason:
            show_warning_box(window, trans.t('experimental_label'), reason)
            return
        ai_available = is_ai_fusion_available()
        methods = [spec.id for spec in fusion_algorithms()
                   if not fusion_lock_reason(spec, window, ai_available)]

        self._compare_queue = list(methods)
        self._compare_total = len(methods)
        self._compare_failures = []
        self._stop_compare_timer()
        self._compare_token = object()
        self._compare_context = None
        self._compare_cancelling = False
        self._compare_mode = True
        self._set_compare_button("cancel")
        self._start_next_compare()

    def _start_next_compare(self) -> None:
        if not self._compare_mode or not self._compare_queue:
            return
        if self._stack_changed_since(self._compare_context):
            self._abort_compare_run()
            self.window.statusBar().showMessage(trans.t('msg_render_stack_changed'), 8000)
            return
        method = self._compare_queue.pop(0)
        self._compare_current = method
        window = self.window
        window.statusBar().showMessage(
            trans.t("msg_compare_progress").format(
                i=self._compare_total - len(self._compare_queue),
                n=self._compare_total,
                method=self.algo_display_name(method)),
            5000)
        self.start_render(force_algorithm=method)

    def start_render(self, force_algorithm: str | None = None, *,
                     interactive: bool = True, confirmed: bool = False,
                     roi_choice: dict | None = None, _storage_assessment=_STORAGE_UNSET,
                     _cpu_confirmed: bool = False, _substack_range=None,
                     _stack_outputs: dict | None = None, queue_item=None):
        """Start a render and report why it did or did not start.

        ``interactive=False`` is the control-channel path: it never opens a
        modal box. A decision that would have shown one is returned as a
        status instead, so the client gets a reply before anything blocks.
        """
        # 重入守卫：本方法中途有多次 processEvents()，期间排队的第二次点击
        # 会重入并把 self.worker 覆盖掉——正在运行的 RenderWorker 失去引用
        # 被 C++ 析构，QThread 直接 qFatal abort。守卫必须在任何
        # processEvents 之前设置。
        if self._pending_storage is not None:
            if interactive:
                self.cancel_storage_preflight()
                return 'cancelled'
            return 'busy'
        if getattr(self, "_render_pending", False):
            return "busy"
        if (self._compare_mode and self._render_after_load is None
                and force_algorithm != self._compare_current):
            return "busy"
        if (self._substack_mode and _substack_range is None and _stack_outputs is None):
            if interactive:
                self.cancel_substack_render()
                return 'cancelled'
            return 'busy'
        self._render_pending = True
        try:
            stop_warmup = getattr(self.window, '_cancel_depth_warmup', None)
            if stop_warmup is not None:
                stop_warmup()
            self._start_render_inner(force_algorithm, interactive=interactive,
                                     confirmed=confirmed, roi_choice=roi_choice,
                                     storage_assessment=_storage_assessment, cpu_confirmed=_cpu_confirmed,
                                     substack_range=_substack_range, stack_outputs=_stack_outputs,
                                     queue_item=queue_item)
        except Exception as exc:
            self.last_state = "error"
            self.last_error = str(exc)
            self.last_result = None
            self._render_context = None
            self._preview_mode = False
            self._active_queue_job = None
            self._abort_compare_run()
            self._abort_substack_run()
            self._restore_ui_controls()
            self._hide_progress_bar()
            raise
        finally:
            self._render_pending = False
        return self._start_status

    def _start_render_inner(self, force_algorithm: str | None = None, *,
                            interactive: bool = True, confirmed: bool = False,
                            roi_choice: dict | None = None, storage_assessment=_STORAGE_UNSET,
                            cpu_confirmed: bool = False, substack_range=None,
                            stack_outputs: dict | None = None, queue_item=None) -> None:
        window = self.window
        is_substack = substack_range is not None
        is_stack_outputs = stack_outputs is not None
        # "started" means the request was accepted, even for a cache hit.
        self._start_status = "started"

        # A render is already running: this click cancels it
        context_worker = getattr(self, "context_worker", None)
        if context_worker is not None:
            self._start_status = "busy"
            if interactive and not getattr(context_worker, 'is_cancelled', False):
                context_worker.cancel()
                window.btn_render.setEnabled(False)
                window.btn_render.setText(trans.t('depth_cancelling'))
                self._show_progress_bar(trans.t('depth_cancelling'))
            return
        if self._render_worker_active():
            self._start_status = "busy"
            if interactive and not getattr(self.worker, 'is_cancelled', False):
                self._cancel_worker()
                self.dismiss_live_preview()
                window.btn_render.setEnabled(False)
                window.btn_render.setText(trans.t('depth_cancelling'))
                self._show_progress_bar(trans.t('depth_cancelling'))
            return
        if self._render_after_load is not None:
            if not interactive:
                self._start_status = "loading"
                return
            queued = self._render_after_load
            self._render_after_load = None
            self.last_state = 'cancelled'
            self._hide_progress_bar()
            self.refresh_action_labels()
            if queued.get("compare", False):
                self._abort_compare_run()
            self._start_status = "loading" if getattr(
                getattr(window, "source_manager", None), "_load_worker", None) is not None else "cancelled"
            window.statusBar().showMessage(trans.t("msg_render_queue_cancelled"), 5000)
            return
        if queue_item is not None:
            reason = desktop_fusion_lock_reason(window, queue_item.method) or (
                trans.t('experimental_locked_hint')
                if alignment_is_locked(window, queue_item.align_method) else ''
            )
        elif is_stack_outputs and stack_outputs.get("alignment") == "none":
            reason = desktop_fusion_lock_reason(
                window, force_algorithm or window.current_fusion_method()
            )
        else:
            reason = self._render_lock_reason(force_algorithm)
        if reason:
            self._start_status = 'unavailable'
            self._abort_compare_run()
            if interactive:
                show_warning_box(window, trans.t('msg_error'), reason)
                return
            raise ValueError(reason)
        # Frames are still loading (a render clicked right after opening a
        # stack used to say "load at least 2 images"): queue the render and
        # start it when the load finishes.  A second click cancels the queue.
        source = getattr(window, "source_manager", None)
        if getattr(source, "_load_worker", None) is not None:
            self._start_status = "loading"
            if interactive:
                self._render_after_load = {
                    "force_algorithm": force_algorithm,
                    "compare": bool(self._compare_mode),
                    "compare_token": self._compare_token,
                    "load_worker": source._load_worker,
                    "load_generation": getattr(source, "_load_generation", 0),
                }
                self.last_state = 'queued'
                self.refresh_action_labels()
            return
        # A previous render may have aborted without clearing this; it decides
        # whether the result is treated as a throwaway draft.
        self._preview_mode = False

        self._stop_compare_timer()

        source_ref = window.raw_images
        source_revision = getattr(window, "source_revision", 0)
        source_content_revision = getattr(window, "source_content_revision", 0)
        compare_token = self._compare_token
        substack_token = self._substack_token
        full_included_idx = None
        full_alignment_key = None
        reusable_aligned_slice = []
        if is_stack_outputs:
            stack_images = list(stack_outputs["images"])
            stack_filenames = list(stack_outputs["filenames"])
            included_idx = list(range(len(stack_images)))
        elif is_substack:
            full_images, full_filenames, full_included_idx = window.effective_stack()
            stack_images = list(full_images[substack_range.start:substack_range.end])
            stack_filenames = list(full_filenames[substack_range.start:substack_range.end])
            included_idx = list(full_included_idx[substack_range.start:substack_range.end])
        elif queue_item is not None and queue_item.frame_indices is not None:
            raw_list = list(window.raw_images or [])
            all_names = list(window.image_filenames or [])
            if any(i < 0 or i >= len(raw_list) for i in queue_item.frame_indices):
                self._start_status = "no_frames"
                return
            included_idx = list(queue_item.frame_indices)
            stack_images = [raw_list[i] for i in included_idx]
            stack_filenames = [
                all_names[i] if i < len(all_names) else f"frame_{i + 1}"
                for i in included_idx
            ]
        else:
            stack_images, stack_filenames, included_idx = window.effective_stack()
        if len(stack_images) < 2:
            if interactive:
                show_warning_box(window, trans.t("msg_no_images_title"), trans.t("msg_render_need_images_text"))
            self._start_status = "no_frames"
            self._abort_compare_run()
            self._abort_substack_run()
            return
        cleanup = (
            False if is_stack_outputs
            else bool(getattr(window, "defect_mask_path", "")
                      or getattr(window, "sensor_denoise", False))
        )
        if cleanup:
            if (getattr(window, "source_transforms", None)
                    or getattr(window, "unrecoverable_source_changes", None)):
                message = trans.t("sensor_history_unsupported")
                self._start_status = "sensor_history_unsupported"
            elif (getattr(window, "roi_mode_active", False)
                  and getattr(window, "roi_aligned_images", None)):
                message = trans.t("sensor_roi_unsupported")
                self._start_status = "sensor_roi_unsupported"
            else:
                message = None
            if message:
                self.last_state = "error"
                self.last_error = message
                self.last_result = None
                self._abort_compare_run()
                self._restore_ui_controls()
                if interactive:
                    show_warning_box(window, trans.t("msg_error"), message)
                return


        if self._ai_confirmation_needed(force_algorithm) and not (confirmed or cpu_confirmed):
            if not interactive:
                self._start_status = "confirm_ai"
                self._abort_compare_run()
                return
            if not self._confirm_cpu_ai_render(force_algorithm):
                self._start_status = "cancelled"
                self._abort_compare_run()
                return
            cpu_confirmed = True
        if compare_token is not self._compare_token:
            self._start_status = "cancelled"
            self.last_state = "cancelled"
            self._restore_ui_controls()
            return

        if queue_item is not None:
            fusion_method = queue_item.method
            method = queue_item.method
            alignment_method = queue_item.align_method
            registration_parameters = deepcopy(queue_item.align_params or {})
            need_fusion = method != "none"
        else:
            fusion_method = window.current_fusion_method()
            method = force_algorithm or fusion_method
            alignment_method = stack_outputs["alignment"] if is_stack_outputs else window.current_alignment_method()
            registration_parameters = window.registration_parameters_for(alignment_method)
            need_fusion = force_algorithm is not None or method != "none"
        requested_steps = alignment_steps(alignment_method)
        collect_report = bool(getattr(window, 'collect_alignment_report', False) and requested_steps and not is_stack_outputs)
        if is_stack_outputs:
            alignment_key = None
            reusable = False
        elif is_substack:
            alignment_key = self._alignment_key(included_idx, method)
            full_alignment_key = self._alignment_key(full_included_idx, method)
            full_reusable = bool(
                window.is_images_aligned
                and getattr(window, "aligned_images", None)
                and len(window.aligned_images) == len(full_included_idx)
                and getattr(window, "alignment_request_key", None) == full_alignment_key
                and not cleanup
            )
            reusable = full_reusable
            if full_reusable:
                reusable_aligned_slice = list(
                    window.aligned_images[substack_range.start:substack_range.end]
                )
        else:
            alignment_key = self._alignment_key(included_idx, method, queue_item=queue_item)
            reusable = bool(window.is_images_aligned
                            and getattr(window, "aligned_images", None)
                            and len(window.aligned_images) == len(stack_images)
                            and getattr(window, "alignment_request_key", None) == alignment_key
                            and not cleanup)
        preview = getattr(window, 'chk_quick_preview', None)
        queue_preview = bool((queue_item.options or {}).get("preview", False)) if queue_item is not None else False
        preview_requested = queue_preview if queue_item is not None else (preview is not None and preview.isChecked())
        publish_cached_alignment = (
            not need_fusion and requested_steps and reusable and not collect_report
            and not is_substack and not is_stack_outputs
            and not getattr(window, 'roi_mode_active', False)
            and not preview_requested)

        # One storage assessment per render start, before any UI is locked.
        storage_override = None
        from core.frame_storage import automatic_preflight_may_block
        assessment = None
        if not publish_cached_alignment and not is_substack and not is_stack_outputs:
            if (storage_assessment is _STORAGE_UNSET and force_algorithm is None
                    and queue_item is None
                    and not getattr(window, 'roi_mode_active', False)
                    and not preview_requested
                    and automatic_preflight_may_block()):
                self._begin_storage_preflight(stack_images, force_algorithm, interactive, confirmed, roi_choice,
                                              cpu_confirmed)
                self._start_status = 'preflight'
                return
            assessment = (self._storage_preflight(force_algorithm, stack_images)
                          if storage_assessment is _STORAGE_UNSET else storage_assessment)
        self.last_storage_assessment = assessment
        status = assessment['status'] if assessment else 'ok'
        if status == 'critical' or (status == 'advisory' and not confirmed):
            if not interactive:
                self._start_status = ("insufficient_storage" if status == 'critical'
                                      else "confirm_storage")
                self._abort_compare_run()
                return
            choice = self._confirm_storage(assessment, stack_images)
            if choice is None:
                self._start_status = "cancelled"
                self._abort_compare_run()
                return
            try:
                storage_override = self._apply_storage_choice(choice)
            except OSError as exc:
                show_warning_box(window, trans.t('storage_low_title'),
                                 trans.t('advanced_storage_invalid').format(error=str(exc)))
                self._start_status = "cancelled"
                self._abort_compare_run()
                return

        if queue_item is not None:
            # Queued background render: keep method, alignment, parameters, and
            # render settings unlocked so the user can prepare additional jobs
            # or inspect existing outputs while this job runs.
            if getattr(window, 'btn_render', None) is not None:
                window.btn_render.setEnabled(True)
                window.btn_render.setText(trans.t('btn_cancel_render'))
            self._set_compare_button("disabled")
        elif force_algorithm:
            # Compare-all mode drives the next renders programmatically; the
            # Render button stays disabled (the Compare button is the cancel
            # affordance) and everything else stays locked until the queue
            # finishes. Painting "Cancel render" here made a disabled button
            # look like the way to stop the run.
            window.btn_render.setEnabled(False)
            window.btn_render.setText(trans.t('btn_render_processing'))
        else:
            window.btn_render.setEnabled(True)
            window.btn_render.setText(trans.t('btn_cancel_render'))
        # Preparation stays on this event-loop turn; modal choices below can
        # still dispatch cancellation or a source replacement.

        if queue_item is None:
            for name in ("fusion_parameter_panel", "registration_parameter_panel",
                         "combo_fusion", "combo_align", "btn_reset"):
                widget = getattr(window, name, None)
                if widget is not None:
                    widget.setEnabled(False)
            if not getattr(self, '_compare_mode', False):
                # A single render owns the pipeline; Compare-all cannot start
                # until it ends, so do not leave an enabled silent no-op.
                self._set_compare_button("disabled")
            # Every control that changes the input to the running result must stay
            # locked until completion, cancellation or failure restores it. A
            # mid-run change used to discard the finished render as stale.
            self._set_render_settings_enabled(False)

        def _brightness(win, option, method):
            """Per-method brightness option (duck-typed hosts may not have it)."""
            getter = getattr(win, "brightness_option_for", None)
            if callable(getter):
                return bool(getter(option, method))
            return bool(getattr(win, "exposure_match", False)) if option == (
                "exposure_match") else bool(
                    getattr(win, "base_frame_correct", False))

        if (queue_item is not None or force_algorithm is None) and method != "none" and get_fusion(method).neural \
                and not is_ai_fusion_available():
            if interactive:
                show_warning_box(
                    window,
                    trans.t("msg_stackmff_unavailable_title"),
                    trans.t("msg_stackmff_unavailable_text"),
                )
            self._start_status = "unavailable"
            self._restore_ui_controls()
            return

        disk_alignment_options = requested_steps
        # Handle ROI options - check if ROI mode is active and we have aligned images
        roi_rect = None
        roi_mode = "crop"
        roi_base_index = 0
        use_roi_aligned_images = False
        
        # 检查ROI模式是否激活，并从右侧面板获取ROI区域
        if not is_substack and not is_stack_outputs and getattr(window, 'roi_mode_active', False) and window.roi_aligned_images:
            if (getattr(window, "roi_source_revision", None) !=
                    getattr(window, "source_revision", 0)
                    or list(getattr(window, "roi_included_indices", None) or []) != list(included_idx)
                    or getattr(window, "roi_reg_downscale_width", None) != window.reg_downscale_width
                    or getattr(window, "roi_alignment_key", None) != alignment_key
                    or len(window.roi_aligned_images) != len(stack_images)):
                window.transform_manager.reset_roi_selection()
                self._restore_ui_controls()
                self._abort_compare_run()
                self._start_status = "stack_changed"
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
                if roi_choice is not None:
                    roi_mode = str(roi_choice.get("mode", "crop"))
                    if roi_mode not in ("crop", "paste"):
                        raise ValueError(f"ROI mode must be crop or paste: {roi_mode}")
                    roi_base_index = int(roi_choice.get("base_index", 0))
                    if not 0 <= roi_base_index < len(window.roi_aligned_images):
                        raise ValueError("ROI base_index is outside the included stack")
                elif not interactive:
                    # The client must decide crop/paste up front; exec() here
                    # would leave its request unanswered.
                    self._start_status = "confirm_roi"
                    self._restore_ui_controls()
                    self._abort_compare_run()
                    return
                else:
                    dialog_cls = globals().get('ROIRenderOptionsDialog')
                    if dialog_cls is None:
                        dialog_cls = __getattr__('ROIRenderOptionsDialog')
                    dialog = dialog_cls(
                        len(window.roi_aligned_images), window,
                        frame_names=names, roi_size=roi_size,
                        output_size=output_size, frames=window.roi_aligned_images,
                        frame_indices=included_idx)
                    if dialog.exec() == QDialog.DialogCode.Accepted:
                        roi_mode = dialog.mode
                        roi_base_index = dialog.base_frame_index
                    else:
                        # User cancelled the ROI dialog -> cancel render
                        self._start_status = "cancelled"
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
            preview_requested
            and not use_roi_aligned_images
            and not compare
            and not is_substack
            and not is_stack_outputs
        )
        if (not use_roi_aligned_images and not self._preview_mode
                and not getattr(window, 'roi_mode_active', False)
                and getattr(window, 'roi_aligned_images', None)):
            # The ROI preview's aligned stack is a second full-resolution
            # copy of every frame (1.8 GB for 15 x 20 MP), kept only to make
            # re-entering ROI mode instant.  A full render builds or reuses
            # its own aligned stack, so holding both doubled the memory and
            # starved later stages (Auto background).  Re-entering ROI mode
            # simply aligns again.
            roi_frames = window.roi_aligned_images
            window.roi_aligned_images = []
            # After an ROI render the review/export stack is that same ROI
            # stack; release it too, or it stays alive until this render ends.
            reviewed = getattr(window, 'registration_results', None) or []
            if reviewed and any(a is b for a, b in zip(reviewed, roi_frames)):
                window.registration_results = []
            del roi_frames, reviewed
        # None = this render neither reads nor writes the alignment cache.
        disk_cache_request = None
        if use_roi_aligned_images:
            # ROI模式下使用已经对齐的图像栈，跳过额外的配准
            source_images = window.roi_aligned_images
            effective_alignment = "none"
            effective_aligned_images = window.roi_aligned_images
            effective_is_aligned = bool(requested_steps)
            effective_last_alignment_options = ((requested_steps, registration_parameters)
                                                if requested_steps else None)
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
            if reusable and not collect_report:
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
                and not reusable and not is_stack_outputs and self._disk_cache_eligible() \
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
            if reusable:
                effective_aligned_images = (
                    reusable_aligned_slice if is_substack else window.aligned_images
                )
            else:
                effective_aligned_images = []
            effective_is_aligned = reusable
            effective_last_alignment_options = window.last_alignment_options if reusable else None
        if disk_cache_request is not None:
            parameters = self._processing_parameters(method, queue_item=queue_item)
            interp = registration_parameters.get("interpolation", "lanczos")
            alignment_roi = getattr(window, "alignment_roi", None)
            disk_cache_request["identity"] = {
                "source_revision": alignment_key[0],
                "included": list(included_idx),
                "frames": [{"name": name, "shape": list(image.shape),
                            "dtype": image.dtype.str}
                           for name, image in zip(stack_filenames, stack_images)],
                "brightness_reference": "last" if parameters["exposure_match"] else "first",
                "exposure_match": parameters["exposure_match"],
                "registration_backend": "opencv",
                "interpolation": "lanczos4" if interp == "lanczos" else interp,
                "registration_steps": list(requested_steps),
                "registration_parameters": registration_parameters,
            }
            if alignment_roi is not None:
                disk_cache_request["identity"]["alignment_roi"] = list(alignment_roi)
            revision = getattr(window, "source_revision", 0)
            disk_cache_request["is_current"] = lambda: (
                window.raw_images is source_ref
                and (queue_item is not None
                     or getattr(window, "source_revision", 0) == revision))

        if queue_item is not None:
            from core.algorithms import resolve_fusion_parameters
            fusion_parameters = (
                resolve_fusion_parameters(method, queue_item.method_params or {})
                if method != "none" else {}
            )
        else:
            fusion_parameters = window.fusion_parameters_for(method)
        # Reusable aligned descriptors are what the fusion stage actually reads.
        estimate_frames = effective_aligned_images if effective_is_aligned else source_images
        memory_estimate = None
        if not publish_cached_alignment:
            memory_estimate = self._estimate_memory(
                estimate_frames, method, fusion_parameters, roi_rect,
                register=bool(requested_steps and effective_alignment != 'none'
                              and (not effective_is_aligned or collect_report)),
                alignment=effective_alignment)
        if (memory_estimate and (memory_estimate.ram_short or memory_estimate.gpu_fallback)
                and interactive and not confirmed and force_algorithm is None
                and not self._confirm_memory(memory_estimate)):
            self._start_status = 'cancelled'
            self._preview_mode = False
            self._restore_ui_controls()
            return

        if compare_token is not self._compare_token or substack_token is not self._substack_token:
            self._start_status = "cancelled"
            self.last_state = "cancelled"
            self._restore_ui_controls()
            return
        expected_included = full_included_idx if is_substack else included_idx
        if (not is_stack_outputs and (
                window.raw_images is not source_ref
                or (queue_item is None and (
                    getattr(window, "source_revision", 0) != source_revision
                    or window.included_indices() != expected_included))
                or (queue_item is not None
                    and getattr(window, "source_content_revision", 0) != source_content_revision)
                or getattr(getattr(window, "source_manager", None), "_load_worker", None) is not None)):
            self._start_status = "stack_changed"
            self._abort_compare_run()
            self._abort_substack_run()
            self._restore_ui_controls()
            return

        # Record exactly what THIS render was launched with. on_render_finished
        # must not read self.worker (a newer render may already have replaced
        # it) and must not publish an alignment that belongs to a different
        # stack than the one the UI shows (quick preview and ROI mode hand the
        # worker downscaled / cropped copies instead of window.raw_images).
        from core.image_loader import stack_icc_profile
        custom_output_label = None
        if is_stack_outputs:
            custom_output_label = stack_outputs["composite_label"]
        elif is_substack:
            custom_output_label = trans.t("substack_output_label").format(
                index=substack_range.index,
                total=substack_range.total,
                start=substack_range.display_start,
                end=substack_range.display_end,
            )
        self._render_context = {
            "source": source_images,
            # Preserve the original input frames even when alignment reuse
            # supplies already-aligned arrays to the worker. Pixels follow the
            # stack's copy-on-write invariant; only the container is copied.
            "comparison_source": tuple(stack_images),
            # Live reference of the FULL stack at launch: with exclusions,
            # "source" is a filtered copy, so identity must be checked against
            # this, never against "source".
            "source_ref": window.raw_images,
            "alignment_method": alignment_method,
            "alignment_options": requested_steps,
            "alignment_identity": ((requested_steps, registration_parameters,
                                    bool(alignment_key and alignment_key[-1] == 'reverse-registration-v1'))
                                   if 'openfocus' in requested_steps
                                   else (requested_steps, registration_parameters)),
            "preview": self._preview_mode,
            "roi": roi_rect is not None,
            "roi_rect": ((float(roi_rect.x()), float(roi_rect.y()),
                          float(roi_rect.width()), float(roi_rect.height()))
                         if roi_rect is not None else None),
            "roi_mode": roi_mode,
            "roi_base_index": roi_base_index,
            "algorithm": method,
            "queue_job_id": queue_item.job_id if queue_item is not None else None,
            "memory_estimate": memory_estimate,
            "aligned": bool(effective_is_aligned),
            "registration_reuse": ('roi' if use_roi_aligned_images and effective_is_aligned
                                   else 'session' if (self._preview_mode or is_substack) and effective_is_aligned
                                   else None),
            "cached_registration_dispatch": (
                getattr(getattr(window, 'roi_registration_details', None), 'dispatch', ())
                if use_roi_aligned_images else
                getattr(window, 'alignment_registration_dispatch', ()) if reusable else ()),
            "filenames": list(stack_filenames),
            "included": list(included_idx),
            "source_revision": getattr(window, "source_revision", 0),
            "source_content_revision": source_content_revision,
            "alignment_key": alignment_key,
            "disk_alignment_options": disk_alignment_options,
            "full_filenames": list(window.image_filenames),
            "icc_profile": (stack_outputs["icc_profile"] if is_stack_outputs
                            else stack_icc_profile(window, included_idx)),
            "linear_dng_metadata": (
                deepcopy(stack_outputs["linear_dng_metadata"]) if is_stack_outputs
                else deepcopy(getattr(window, "linear_dng_metadata", None))
            ),
            # Match the frames actually sent to the worker, not every loaded
            # file: EXIF/source provenance is indexed by processed_images.
            "source_paths": (() if is_stack_outputs else tuple(
                window.image_source_paths[i]
                for i in included_idx
                if 0 <= i < len(getattr(window, "image_source_paths", ()) or ())
            )),
            "frame_exif": (() if is_stack_outputs else self._source_exif_snapshot(included_idx)),
            "frame_icc_profiles": (() if is_stack_outputs else tuple(
                (getattr(window, "image_icc_profiles", ()) or ())[i]
                if i < len(getattr(window, "image_icc_profiles", ()) or ()) else b""
                for i in included_idx)),
            "parameters": self._processing_parameters(method, queue_item=queue_item),
            "substack": is_substack,
            "substack_range": substack_range,
            "full_included": list(full_included_idx) if is_substack else None,
            "full_alignment_key": full_alignment_key if is_substack else None,
            "stack_outputs": is_stack_outputs,
            "output_label": custom_output_label,
            "source_output_names": (stack_outputs["source_output_names"] if is_stack_outputs else ()),
            "source_output_ids": (stack_outputs["source_output_ids"] if is_stack_outputs else ()),
            "from_substack_count": (stack_outputs["from_substack_count"] if is_stack_outputs else None),
        }
        if compare:
            self._compare_context = self._render_context
        if is_substack:
            self._substack_context = self._render_context
        from utils.scalebar import effective_px_um
        frame_ids = getattr(getattr(window, "label_manager", None), "frame_ids", None)
        if callable(frame_ids) and not is_stack_outputs:
            identities = frame_ids("input")
            self._render_context["frame_ids"] = [identities[i] for i in included_idx]
        self._render_context.update(
            pixel_size_um=effective_px_um(window),
            source_px_um=getattr(window, "source_px_um", None),
            source_scale_factor=getattr(window, "current_scale_factor", 1.0))

        from utils.logging_setup import log_action
        processing = self._render_context["parameters"]
        log_action("render_start", method=force_algorithm or fusion_method,
                   align=alignment_method, preview=self._preview_mode,
                   frames=len(stack_images), loaded_frames=len(window.raw_images or []),
                   sharpen_output=processing["sharpen_output"],
                   sharpen_strength=processing["sharpen_strength"])
        self.last_error = None
        self.last_result = None
        self._control_progress = None
        self.last_state = "running"
        self._active_queue_job = queue_item
        self._render_context['started_at'] = monotonic()
        self._render_interactive = interactive
        if publish_cached_alignment:
            self._render_context["registration_reuse"] = 'session'
            self.on_render_stage('waiting')
            self.on_render_finished(effective_aligned_images, None, True, 0.0, 0.0, "CPU")
            return

        all_source_paths = list(getattr(window, "image_source_paths", None) or [])
        source_paths = (
            [] if is_stack_outputs
            else ([all_source_paths[i] for i in included_idx]
                  if len(all_source_paths) == len(window.raw_images) else [])
        )
        # Retain one prepared stack. Methods that do not build snapshots can
        # leave it in place as long as the full-resolution alignment is valid.
        from core.source_frames import SourceFrame
        descriptor = get_fusion(method) if need_fusion else None
        native_preparation = method == 'openfocus_b'
        preparation_key = (alignment_key,)
        if native_preparation:
            from core.frame_storage import storage_directory
            preparation_key += (method, storage_override or storage_directory())
        retain_preparation = bool(
            not is_substack and not is_stack_outputs
            and descriptor and descriptor.streaming_sources
            and (native_preparation or (
                descriptor.neural and getattr(window, 'tile_enabled', None) is not False
                and not (descriptor.sort_frames and processing["stack_reverse_order"])))
            and not self._preview_mode and roi_rect is None and not use_roi_aligned_images
            and not collect_report and not cleanup
            and all(isinstance(frame, SourceFrame) and
                    not (frame.ndim == 3 and frame.shape[2] == 4)
                    for frame in stack_images))
        replaces_alignment = (not is_substack and not is_stack_outputs
                              and requested_steps and not self._preview_mode
                              and roi_rect is None and not use_roi_aligned_images
                              and (not reusable or collect_report))
        if (not is_substack and not is_stack_outputs and self._prepared_cache is not None and (
                self._prepared_cache_source is not source_ref
                or self._prepared_cache_key[0] != alignment_key
                or replaces_alignment
                or (retain_preparation and self._prepared_cache_key != preparation_key))):
            self.clear_prepared_cache()
        prepared_cache = self._prepared_cache if retain_preparation else None
        self._render_context['preparation_key'] = preparation_key if retain_preparation else None
        live_preview = (
            interactive and not compare and not is_substack and not is_stack_outputs
            and queue_item is None and need_fusion
            and method in ('openfocus_a', 'openfocus_b', 'openfocus_c')
            and not self._preview_mode and roi_rect is None and not use_roi_aligned_images
            and getattr(window, 'live_preview_page', None) is not None
            and not window.wipe_active and not getattr(window, 'roi_mode_active', False))
        if live_preview:
            self._render_context['live_source_rows'] = tuple(
                reversed(included_idx) if processing["stack_reverse_order"]
                else included_idx)
        worker_cls = globals().get('RenderProcessWorker')
        if worker_cls is None:
            worker_cls = __getattr__('RenderProcessWorker')
        self.worker = worker_cls(
            source_images,
            effective_aligned_images,
            effective_is_aligned,
            effective_last_alignment_options,
            algorithm=method if need_fusion else None,
            alignment=effective_alignment,
            fusion_parameters=fusion_parameters,
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
            base_frame=processing["base_frame"],
            sort_mode="asc" if is_stack_outputs else processing["sort_mode"],
            exposure_match=processing["exposure_match"],
            sharpen_output=processing["sharpen_output"],
            sharpen_strength=processing["sharpen_strength"],
            defect_mask_path="" if is_stack_outputs else getattr(window, "defect_mask_path", ""),
            sensor_denoise=False if is_stack_outputs else bool(getattr(window, "sensor_denoise", False)),
            source_paths=source_paths,
            load_scale_factor=getattr(window, "current_scale_factor", 1.0),
            disk_cache=disk_cache_request,
            collect_alignment_report=collect_report and not use_roi_aligned_images,
            prepared_cache=prepared_cache,
            # Jobs that cannot consume the retained snapshots may keep their
            # aligned pixels, but must not grow the shared retention budget.
            retain_aligned_pixels=self._prepared_cache is None or retain_preparation,
            retain_preparation=retain_preparation,
            preview=self._preview_mode,
            profile_performance=bool(getattr(window, "profile_performance", False)),
            alignment_roi=getattr(window, "alignment_roi", None),
        )
        self.worker.storage_directory = storage_override
        self.worker.live_preview_requested = live_preview

        self.worker.finished_signal.connect(self.on_render_finished)
        self.worker.error_signal.connect(self.on_render_error)
        self.worker.progress_signal.connect(self.on_render_progress)
        self.worker.stage_signal.connect(self.on_render_stage)
        window.statusBar().clearMessage()
        gallery = getattr(window, "gallery_controller", None)
        if gallery is not None:
            gallery.on_render_started(
                show_fusion=interactive and need_fusion and not compare and not self._preview_mode,
            )
        self.on_render_stage('waiting')
        if memory_estimate:
            summary = self._memory_summary(memory_estimate)
            if memory_estimate.gpu_fallback:
                from fusion_methods.neural_runtime import display_name
                summary += ' · ' + trans.t('render_memory_fallback_plan').format(
                    device=display_name(memory_estimate.accelerator))
            window.statusBar().showMessage(summary, 8000)
        self.worker.start()
        if live_preview:
            self._start_live_preview()

    def _start_live_preview(self):
        self._live_preview_row = self.window.file_list.currentRow()
        self._live_preview_user_selected = False
        if self._live_preview_timer is None:
            timer = self._live_preview_timer = QTimer(self.window.live_preview_page)
            timer.setInterval(80)
            timer.timeout.connect(self._poll_live_preview)
        self._live_preview_timer.start()

    def _poll_live_preview(self):
        if not self._live_preview_timer.isActive():
            return
        worker, context, window = self.worker, self._render_context, self.window
        if (worker is None or worker.is_cancelled or context is None
                or getattr(self, '_closing', False)
                or window.raw_images is not context['source_ref']
                or getattr(window, 'source_revision', 0) != context['source_revision']
                or window.wipe_active or getattr(window, 'roi_mode_active', False)):
            self.dismiss_live_preview()
            return
        source_list = window.file_list
        current_row = source_list.currentRow()
        if current_row != self._live_preview_row:
            self._live_preview_user_selected = True
        frame = worker.take_live_preview()
        if frame is not None:
            page = window.live_preview_page
            page.show_frame(*frame)
            window.view_stack.setCurrentWidget(page)
            # Fusion order follows the included, possibly reversed subset.
            rows = context['live_source_rows']
            index = frame[3] - 1
            # Do not replace the pane's source during a held loupe or drag.
            if (not self._live_preview_user_selected and 0 <= index < len(rows)
                    and not QApplication.mouseButtons()):
                row = rows[index]
                item = source_list.item(row)
                if item is not None and row != current_row:
                    self._live_preview_row = row
                    source_list.setCurrentRow(row)
                    source_list.scrollToItem(item)

    def inspect_previous_output(self, *_args):
        # History stays usable while preparing frames or aligning. Once the
        # live page is visible, an explicit selection takes precedence.
        page = self.window.live_preview_page
        if self.window.view_stack.currentWidget() is page:
            self.dismiss_live_preview()

    def dismiss_live_preview(self):
        """Restore the current output and keep it visible for this job."""
        self._live_preview_row = None
        self._live_preview_user_selected = False
        timer = self._live_preview_timer
        if timer is None or not timer.isActive():
            return
        timer.stop()
        page, stack = self.window.live_preview_page, self.window.view_stack
        if stack.currentWidget() is page:
            stack.setCurrentIndex(0)
        page.clear()

    def _stack_changed_since(self, context) -> bool:
        """True when the stack (or the settings it was rendered with) is no
        longer the one a render context was started for."""
        if context is None:
            return False
        if context.get("stack_outputs"):
            return False
        window = self.window
        if context.get("queue_job_id"):
            # Queued jobs carry their own frozen method, parameter, option,
            # and frame-index snapshots, so live UI edits for the next job
            # must not invalidate an in-flight background job. Only a load,
            # clear, or destructive stack transform invalidates it.
            return bool(
                window.raw_images is not context.get("source_ref")
                or getattr(window, "source_content_revision", 0) != context.get("source_content_revision", 0)
                or list(window.image_filenames or []) != context.get("full_filenames", context["filenames"])
            )
        if context.get("substack"):
            full_included = context.get("full_included")
            return (
                window.raw_images is not context.get("source_ref")
                or (full_included is not None and window.included_indices() != full_included)
                or getattr(window, "source_revision", 0) != context.get("source_revision", 0)
                or list(window.image_filenames or []) != context.get("full_filenames", context["filenames"])
                or (context.get("full_alignment_key") is not None
                    and full_included is not None
                    and self._alignment_key(full_included, context["algorithm"]) != context["full_alignment_key"])
                or ("parameters" in context and self._processing_parameters(
                    context["algorithm"]) != context["parameters"])
            )
        return (
            window.raw_images is not context.get("source_ref")
            or window.included_indices() != context.get("included")
            or getattr(window, "source_revision", 0) != context.get("source_revision", 0)
            or list(window.image_filenames or []) != context.get("full_filenames", context["filenames"])
            or ("alignment_key" in context and context["alignment_key"] is not None
                and self._alignment_key(context["included"], context["algorithm"]) != context["alignment_key"])
            or ("parameters" in context and self._processing_parameters(
                context["algorithm"]) != context["parameters"])
        )

    def on_render_finished(
        self,
        processed_images: List[Any],
        fusion_result: Optional[Any],
        registration_performed: bool,
        alignment_time: float,
        fusion_time: float,
        device_name: str,
    ) -> None:
        self.dismiss_live_preview()
        window = self.window
        if self.worker is not None and getattr(self.worker, "is_cancelled", False):
            self.on_render_error("CANCELLED: render cancelled")
            return
        # The render stays busy until the output and its provenance are published.
        context = self._render_context
        preview = context["preview"] if context else getattr(self, '_preview_mode', False)
        # Preserve the request while a context worker is active.
        compare = getattr(self, '_compare_mode', False)
        substack = bool(getattr(self, '_substack_mode', False) and (context or {}).get('substack'))

        # The stack can change while a render runs (a frame deleted, a new
        # folder loaded, rotate/flip/resize). Publishing that result would show
        # — and export — frames of the previous stack under the new stack's
        # filenames, so drop it instead. The alignment cache has its own,
        # stricter guard below.
        stack_changed = self._stack_changed_since(context)
        if stack_changed:
            import logging as _logging
            message = trans.t('msg_render_stack_changed')
            print("Render finished but the stack changed meanwhile — result discarded.")
            _logging.getLogger("openfocus").warning(
                "render result discarded: stack changed while rendering")
            self.last_state = "discarded"
            self.last_error = message
            self.last_result = None
            self._control_progress = None
            window.statusBar().showMessage(message, 8000)
            # Nothing will chain the next comparison method any more (the
            # queue is only advanced from the normal completion path), so an
            # untouched compare run left the render button disabled forever
            # with a stale "-1/0" progress label.
            self._abort_compare_run()
            self._abort_substack_run()
            self._render_context = None
            discarded_queue_job = self._active_queue_job
            self._active_queue_job = None
            self._hide_progress_bar()
            self._dispose_worker()
            self._restore_ui_controls()
            gallery = getattr(window, "gallery_controller", None)
            if gallery is not None:
                gallery.on_render_ended()
            if discarded_queue_job is not None:
                queue = getattr(window, "render_queue", None)
                if queue is not None:
                    queue.notify_job_failed(discarded_queue_job.job_id, message)
            else:
                self._notify_queue_idle()
            return
        if getattr(self, '_render_progress_stage', None) != 'publish':
            self.on_render_stage('publish')
        ctx = context or {}
        registration_details = getattr(self.worker, 'registration_details', None)
        reuse = ctx.get('registration_reuse') or getattr(registration_details, 'reuse', None)
        if not reuse and getattr(self.worker, 'alignment_reused', False):
            reuse = 'session'
        if reuse in ('session', 'roi'):
            registration_details = RegistrationDetails(ctx.get('cached_registration_dispatch', ()), reuse)
        elif registration_details is None and registration_performed:
            registration_details = RegistrationDetails()
        effective_registration = bool(registration_details is not None)
        if context is not None:
            context['registration_details'] = registration_details
        candidate = getattr(self, "_context_candidate", None)
        if candidate is None and fusion_result is not None and not preview:
            import numpy as np
            needs_copy = any(isinstance(image, np.ndarray) and
                             ((context or {}).get("roi_rect") is not None or
                              image.shape[:2] != fusion_result.shape[:2])
                             for image in processed_images if image is not None)
            if needs_copy:
                from core.workers import RenderContextWorker
                kwargs = self._render_context_kwargs(context, compare, effective_registration)
                worker = RenderContextWorker(processed_images, fusion_result, **kwargs)
                self._retiring_context_workers.add(worker)
                self.context_worker = worker
                self._context_candidate = (processed_images, fusion_result, registration_performed,
                                           alignment_time, fusion_time, device_name)
                worker.ready.connect(lambda result: self._context_ready(worker, result, None))
                worker.failed.connect(lambda error: self._context_ready(worker, None, error))
                worker.finished.connect(lambda: self._context_worker_finished(worker))
                worker.start()
                return
        self._preview_mode = False
        if context and context.get('preparation_key') is not None:
            self._prepared_cache = getattr(self.worker, 'prepared_cache', None)
            self._prepared_cache_key = context['preparation_key'] if self._prepared_cache is not None else None
            self._prepared_cache_source = context['source_ref'] if self._prepared_cache is not None else None


        self.alignment_diagnostics = (
            list(getattr(self.worker, 'alignment_report', None) or []),
            list((context or {}).get('filenames', [])),
            list((context or {}).get('included', [])),
        )
        self._capture_performance_report()

        previous_preview_record = getattr(window, "_wipe_preview_record", None)
        display_committed = False
        result_info = None
        try:
            window._wipe_preview_record = None
            # Draft ownership changes only after the replacement pixels are displayed.
            if fusion_result is not None:
                if preview and not (context or {}).get("roi"):
                    try:
                        from controllers.output_manager import OutputRecord
                        ctx = context or {}
                        preview_context = build_render_context(
                            processed_images, fusion_result,
                            filenames=ctx.get("filenames"),
                            algorithm=ctx.get("algorithm", "unknown"),
                            aligned=effective_registration,
                            source_indices=ctx.get("included"),
                            frame_ids=ctx.get("frame_ids"),
                            source_revision=ctx.get("source_revision", 0),
                            source_content_revision=ctx.get("source_content_revision"),
                            parameters=ctx.get("parameters"),
                            fusion_details=getattr(self.worker, 'fusion_details', None),
                            registration_details=registration_details,
                            linear_dng_metadata=ctx.get("linear_dng_metadata"),
                            source_exif=((ctx.get("frame_exif") or (b"",))[0]),
                            frame_exif=ctx.get("frame_exif"),
                            frame_icc_profiles=ctx.get("frame_icc_profiles"),
                            source_paths=ctx.get("source_paths"))
                        if preview_context is not None:
                            window._wipe_preview_record = OutputRecord.create(
                                fusion_result, "", context=preview_context)
                    except Exception:
                        import logging
                        logging.getLogger("openfocus").exception("could not record preview provenance")
                previous_fusion = getattr(window, "fusion_result", None)
                window.fusion_result = fusion_result
                result_info = {
                    "shape": list(fusion_result.shape),
                    "dtype": str(fusion_result.dtype),
                    "device": device_name,
                    "alignment_seconds": round(float(alignment_time), 3),
                    "fusion_seconds": round(float(fusion_time), 3),
                    "registration_performed": effective_registration,
                    "preview": bool(preview),
                    "method": (context or {}).get("algorithm"),
                }
                # A quick preview is a throwaway draft: show it, but keep it
                # out of the output history so only full-quality renders land
                # there.
                if not preview:
                    window.registration_results = processed_images
                    window.registered_source_indices = list((context or {}).get("included", []))
                    window.registration_frame_ids = list((context or {}).get("frame_ids", []))
                    window.registration_filenames = list((context or {}).get("filenames", []))


                try:
                    displayed = window.output_manager.display_specific_fusion_result(fusion_result)
                    if not displayed:
                        raise RuntimeError("Could not display render result")
                except Exception:
                    window.fusion_result = previous_fusion
                    raise
                display_committed = True
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
                if preview:
                    window.preview_result = {"kind": "fusion", "image": fusion_result}
                    # The draft is on the canvas, not in the history. Clear
                    # the old row so Ctrl+S cannot mistake a previous formal
                    # result for the draft the user is looking at.
                    try:
                        window.output_list.clearSelection()
                        window.output_list.setCurrentRow(-1)
                    except Exception:
                        pass
                else:
                    window.preview_result = None
                new_context = None
                if not preview:
                    try:
                        if candidate is not None:
                            new_context = candidate["context"]
                            if candidate["error"]:
                                raise RuntimeError(candidate["error"])
                        else:
                            new_context = build_render_context(
                                processed_images, fusion_result,
                                **self._render_context_kwargs(context, compare, effective_registration))
                        window.last_render_context = new_context
                    except Exception:
                        window.last_render_context = None
                        import logging
                        logging.getLogger("openfocus").exception(
                            "could not record render provenance; "
                            "the output will not be retouchable")
                    method = (context or {}).get("algorithm")
                    custom_label = (context or {}).get("output_label")
                    label = custom_label or window.export_manager.generate_default_filename(fusion_method=method)
                    window.output_manager.update_output_list_for_fusion(
                        label,
                        context=new_context,
                        fusion_details=getattr(self.worker, 'fusion_details', None),
                        registration_details=registration_details,
                        linear_dng_metadata=(context or {}).get("linear_dng_metadata"),
                        source_paths=(context or {}).get("source_paths", ()),
                        source_output_names=(context or {}).get("source_output_names", ()),
                        source_output_ids=(context or {}).get("source_output_ids", ()),
                    )

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
                if not preview and not compare:
                    from utils.sound_notify import play_completion_sound
                    play_completion_sound(bool(getattr(window, "render_complete_sound", False)))

                # Crash auto-recovery: new outputs refresh the snapshot
                from utils import recovery
                recovery.write_snapshot(window)

                if preview:
                    print("Preview render completed (not added to output history)")
                else:
                    print("Fusion completed successfully!")

                import logging
                logging.getLogger("openfocus").info(
                    "render finished (preview=%s, method fusion, align %.1fs, fuse %.1fs)",
                    preview, alignment_time, fusion_time)
            else:
                if effective_registration:
                    result_info = {
                        "device": device_name,
                        "alignment_seconds": round(float(alignment_time), 3),
                        "fusion_seconds": round(float(fusion_time), 3),
                        "registration_performed": True,
                        "preview": bool(preview),
                        "method": (context or {}).get("algorithm"),
                        "frame_count": len(processed_images),
                    }
                    if preview:
                        # Preview draft: show the small aligned frames without
                        first = processed_images[0] if processed_images else None
                        if first is None:
                            raise ValueError("Registration preview contains no frames")
                        display = first
                        metadata = (context or {}).get('linear_dng_metadata')
                        if metadata is not None:
                            from utils.linear_dng import develop_linear
                            display = develop_linear(display, metadata)
                        pixmap = cv2_to_pixmap(display)
                        window.lbl_result_img.set_display_pixmap(pixmap)
                        window.preview_result = {"kind": "registration", "image": first}
                        display_committed = True
                        window.fusion_result = None
                        window.registration_results = []
                        window.registered_source_indices = []
                        window.registration_frame_ids = []
                        window.registration_filenames = []
                        window.current_result_index = -1
                        window.aligned_stack_dir = None
                        window.output_list.clearSelection()
                        window.output_list.setCurrentRow(-1)
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
                        window.preview_result = None
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
                and not context.get("substack")
                and not context.get("stack_outputs")
                and window.raw_images is context.get("source_ref")
                and not stack_changed
                and len(processed_images) == len(context["included"])
            )
            if registration_performed and alignment_matches_ui:
                window.aligned_images = processed_images
                window.is_images_aligned = True
                window.last_alignment_options = context["alignment_identity"]
                window.alignment_request_key = context["alignment_key"]
                window.alignment_registration_dispatch = getattr(registration_details, 'dispatch', ())
                refresh_aligned = getattr(window, '_on_cached_alignment_updated', None)
                if callable(refresh_aligned):
                    refresh_aligned()
                # The disk cache was written by the worker (it owns the PNG
                # encode); writing it here again re-encoded the whole stack on
                # the GUI thread for nothing.

            started_at = (context or {}).get('started_at')
            total_time = monotonic() - started_at if started_at is not None else alignment_time + fusion_time
            if result_info is not None:
                result_info['elapsed_seconds'] = round(total_time, 3)

            info_lines = []
            if fusion_result is not None:
                height, width = fusion_result.shape[:2]
                info_lines.append(trans.t("status_res_fmt").format(width, height))

            if preview:
                info_lines.append(trans.t("info_preview_quality"))
            if effective_registration and reuse:
                info_lines.append(trans.t("info_align_time").format(alignment_time))
            elif effective_registration:
                alignment_method = (context or {}).get("alignment_method") or window.current_alignment_method()
                align_method_str = trans.t(get_registration(alignment_method).label_key)
                info_lines.append(trans.t("info_align_method").format(align_method_str))
                info_lines.append(trans.t("info_align_time").format(alignment_time))
            else:
                info_lines.append(trans.t("info_align_none"))
                info_lines.append(trans.t("info_align_time").format(0.0))
            from controllers.output_manager import registration_tooltip_lines
            info_lines.extend(registration_tooltip_lines(registration_details))

            fusion_method = (context or {}).get('algorithm') or window.current_fusion_method()
            if fusion_method == "none" and not compare:
                # 仅对齐模式：只跑配准，融合时间不适用
                info_lines.append(trans.t("info_fusion_none"))
            else:
                # 对比渲染时显示的是队列里实际渲染的方法，而非当前下拉选择
                compare_method = getattr(self, "_compare_current", None)
                method_name = self.algo_display_name(
                    compare_method or fusion_method)
                info_lines.append(trans.t("info_fusion_method").format(method_name))
                info_lines.append(trans.t("info_fusion_time").format(fusion_time))
                info_lines.append(trans.t("info_fusion_device").format(device_name))
                estimate = (context or {}).get('memory_estimate')
                if estimate and estimate.accelerator and device_name == 'CPU':
                    from fusion_methods.neural_runtime import display_name
                    info_lines.append(trans.t('render_memory_fallback_actual').format(
                        device=display_name(estimate.accelerator)))

            info_lines.append(trans.t("info_total_time").format(total_time))
            if self.alignment_diagnostics[0]:
                fallback_count = len({entry['frame_index'] for entry in self.alignment_diagnostics[0]
                                      if entry.get('status') == 'fallback' and 'frame_index' in entry})
                if fallback_count:
                    info_lines.append(f"{trans.t('alignment_report_fallback')}: {fallback_count}")

            # Non-blocking completion report: a modal "Completed" box after
            # every render (incl. quick previews) cost a click per iteration
            # and covered the result the user wanted to inspect. The summary
            # goes to the status bar; the full timing sits in the output
            # entry's tooltip.
            if not compare and not substack:
                if (context or {}).get("stack_outputs"):
                    from_substack_count = (context or {}).get("from_substack_count")
                    if from_substack_count:
                        summary_prefix = trans.t("substack_done_with_restack").format(
                            count=from_substack_count)
                    else:
                        summary_prefix = trans.t("stack_outputs_done").format(
                            count=len((context or {}).get("source_output_names") or ()))
                    window.statusBar().showMessage(
                        summary_prefix + " — " + " · ".join(info_lines), 15000)
                else:
                    window.statusBar().showMessage(
                        trans.t("dialog_completed_title") + " — "
                        + " · ".join(info_lines), 15000)
            if (fusion_result is not None and not preview
                    and window.output_list.count() > 0):
                newest = window.output_list.item(0)
                if newest is not None:
                    record_for_row = getattr(window.output_manager, "record_for_row", None)
                    record = record_for_row(0) if callable(record_for_row) else None
                    refresh = getattr(window.output_manager, "_refresh_output_tooltip", None)
                    if record is not None and callable(refresh):
                        record.tooltip_details = tuple(info_lines)
                        refresh(0)
                    else:
                        newest.setToolTip("\n".join([newest.text(), ""] + info_lines))
            self.last_result = result_info
            self.last_state = "done" if self.last_result is not None else "error"
            if self.last_state == "error":
                self.last_error = "Render completed without producing a result"
            else:
                from utils.logging_setup import log_action
                log_action('render_finished', preview=bool(preview), seconds=round(total_time, 2),
                           alignment_seconds=round(alignment_time, 2), fusion_seconds=round(fusion_time, 2))

        except Exception as exc:
            if not display_committed:
                window._wipe_preview_record = previous_preview_record
            import logging
            logging.getLogger("openfocus").exception("render failed with exception")
            self.last_state = "error"
            self.last_error = str(exc)
            self.last_result = None
            if getattr(self, "_render_interactive", True):
                show_message_box(
                    window,
                    trans.t("msg_error"),
                    trans.t("msg_proc_error"),
                    str(exc),
                    QMessageBox.Icon.Critical,
                )

        finally:
            finished_queue_job = self._active_queue_job
            self._active_queue_job = None
            self._dispose_worker()
            if not compare and not substack:
                self._restore_ui_controls()
            produced_output = self.last_state == 'done' and (
                fusion_result is not None or (effective_registration and not preview))
            self._hide_progress_bar(produced_output=produced_output)
            gallery = getattr(window, "gallery_controller", None)
            if gallery is not None:
                if self.last_state == 'done' and fusion_result is not None:
                    gallery.on_fusion_completed(
                        fusion_result,
                        interactive=getattr(self, "_render_interactive", True),
                        is_preview=bool(preview),
                        is_compare=bool(compare or substack),
                    )
                else:
                    gallery.on_render_ended()
            if finished_queue_job is not None:
                queue = getattr(window, "render_queue", None)
                if queue is not None:
                    if self.last_state == 'done':
                        result_index = None
                        result_name = ""
                        result_output_id = ""
                        if not preview and getattr(window, "output_list", None) is not None and window.output_list.count() > 0:
                            result_index = 0
                            first_item = window.output_list.item(0)
                            if first_item is not None:
                                result_name = first_item.text()
                            record_for_row = getattr(getattr(window, "output_manager", None), "record_for_row", None)
                            record = record_for_row(0) if callable(record_for_row) else None
                            if record is not None:
                                result_name = getattr(record, "label", "") or result_name
                                result_output_id = getattr(record, "output_id", "") or ""
                        queue.notify_job_completed(
                            finished_queue_job.job_id,
                            result_index=result_index,
                            result_name=result_name,
                            result_output_id=result_output_id,
                        )
                    else:
                        queue.notify_job_failed(
                            finished_queue_job.job_id,
                            self.last_error or trans.t("msg_proc_error"),
                        )
            elif compare:
                if self._compare_queue:
                    # Keep controls locked and chain the next method. The
                    # timer is tracked so cancelling the run can stop it —
                    # an untracked one fired into a restarted run and
                    # cancelled its in-flight render.
                    self._queue_next_compare()
                else:
                    self._finish_compare_run()
            elif substack:
                if self.last_state == 'done' and fusion_result is not None and window.output_list.count() > 0:
                    rec = window.output_manager.record_for_row(0)
                    if rec is not None:
                        self._substack_results.append(rec)
                if self._substack_cancelling:
                    self._abort_substack_run()
                elif self._substack_queue:
                    self._queue_next_substack()
                else:
                    self._finish_substack_run()
            else:
                self._notify_queue_idle()

    def _render_context_kwargs(self, context, compare, registration_performed):
        ctx = context or {}
        return dict(
            filenames=ctx.get("filenames") or list(self.window.image_filenames or []),
            algorithm=((self._compare_current if compare else None) or ctx.get("algorithm")
                       or self.window.current_fusion_method()),
            alignment_options=ctx.get("alignment_options"),
            aligned=bool(registration_performed or (ctx.get("roi") and ctx.get("aligned"))),
            roi_rect=ctx.get("roi_rect"), frame_ids=ctx.get("frame_ids"),
            roi_mode=ctx.get("roi_mode"), source_indices=ctx.get("included"),
            source_revision=ctx.get("source_revision", 0),
            source_content_revision=ctx.get("source_content_revision"),
            parameters=ctx.get("parameters"),
            pixel_size_um=ctx.get("pixel_size_um"), source_px_um=ctx.get("source_px_um"),
            source_scale_factor=ctx.get("source_scale_factor", 1.0),
            icc_profile=ctx.get("icc_profile", b""),
            source_exif=((ctx.get("frame_exif") or (b"",))[0]),
            frame_exif=ctx.get("frame_exif"), frame_icc_profiles=ctx.get("frame_icc_profiles"),
            rendered_depth=getattr(self.worker, 'rendered_depth', None),
            fusion_details=getattr(self.worker, 'fusion_details', None),
            registration_details=ctx.get('registration_details'),
            linear_dng_metadata=ctx.get("linear_dng_metadata"), source_paths=ctx.get("source_paths"))

    def _context_ready(self, worker, result, error):
        if worker is not self.context_worker or not isinstance(self._context_candidate, tuple):
            return
        if getattr(self, "_closing", False):
            self._context_candidate = None
            self.context_worker = None
            return
        args = self._context_candidate
        if self._stack_changed_since(self._render_context):
            self._context_candidate = {"context": None, "error": None}
            self.context_worker = None
            try:
                self.on_render_finished(*args)
            finally:
                self._context_candidate = None
            return
        if getattr(worker, "is_cancelled", False) or (error and error.startswith("CANCELLED")):
            self._context_candidate = None
            self.context_worker = None
            self.on_render_error("CANCELLED: render cancelled")
            return
        self._context_candidate = {"context": result, "error": error}
        self.context_worker = None
        try:
            self.on_render_finished(*args)
        finally:
            self._context_candidate = None

    def _context_worker_finished(self, worker):
        self._retiring_context_workers.discard(worker)
        worker.deleteLater()

    def _publish_aligned_stack(self, window, frames) -> None:
        """Stage an align-only result as a folder the output list can show."""
        try:
            from controllers.output_manager import publish_aligned_stack

            ctx = self._render_context or {}
            frame_exif = tuple(ctx.get("frame_exif") or ())
            publish_aligned_stack(window, frames,
                                  filenames=ctx.get("filenames"),
                                  icc_profile=ctx.get("icc_profile", b""),
                                  linear_dng_metadata=ctx.get("linear_dng_metadata"),
                                  source_exif=(frame_exif[0] if frame_exif else b""),
                                  frame_exif=frame_exif,
                                  frame_icc_profiles=ctx.get("frame_icc_profiles"),
                                  registration_details=ctx.get('registration_details'),
                                  source_frames=ctx.get("comparison_source"),
                                  source_indices=(self._render_context or {}).get("included"),
                                  alignment_options=(self._render_context or {}).get("alignment_options"),
                                  source_revision=(self._render_context or {}).get("source_revision", 0))
            if getattr(window, "wipe_active", False):
                window.refresh_wipe_controls()
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
            sb.add_activity(label)
            sb.add_activity(bar)
            label.hide()
            bar.hide()
            self._status_bar_progress = bar
            self._status_bar_label = label
        return bar, self._status_bar_label

    def _show_progress_bar(self, text: str) -> None:
        if not getattr(self, "_render_active", False):
            self._render_active = True
            signals = getattr(self, "signals", None)
            if signals is not None:
                signals.render_started.emit()
        try:
            bar, label = self._progress_widgets()
            bar.setRange(0, 0)  # indeterminate until a stage reports counts
            label.setText(text)
            label.show()
            estimate = (self._render_context or {}).get('memory_estimate')
            label.setToolTip(self._memory_details(estimate) if estimate else '')
            bar.show()
            self.window.statusBar().setVisible(True)
        except Exception:
            pass

    def _hide_progress_bar(self, *, produced_output: bool = False) -> None:
        bar = getattr(self, "_status_bar_progress", None)
        if bar is not None:
            bar.hide()
            self._status_bar_label.hide()
        if getattr(self, "_render_active", False):
            self._render_active = False
            signals = getattr(self, "signals", None)
            if signals is not None:
                signals.render_finished.emit(produced_output)

    def _cancel_worker(self) -> None:
        """Ask the running worker to stop, waking its cancel event at once."""
        worker = self.worker
        if worker is None:
            return
        cancel = getattr(worker, "cancel", None)
        if callable(cancel):
            cancel()
        else:
            worker.is_cancelled = True


    def on_render_stage(self, stage: str) -> None:
        self._render_progress_stage = stage
        if self._render_is_cancelling():
            text = trans.t('depth_cancelling')
        else:
            text = trans.t(_RENDER_STAGE_KEYS.get(stage, 'render_progress'))
        self._control_progress = None
        self._progress_first_t = monotonic()
        self._progress_last_t = None
        self._show_progress_bar(text)
        gallery = getattr(self.window, "gallery_controller", None)
        if gallery is not None:
            gallery.on_render_progress(0, 0, text)
        if self._active_queue_job is not None:
            queue = getattr(self.window, "render_queue", None)
            if queue is not None:
                queue.notify_job_progress(
                    self._active_queue_job.job_id,
                    self._active_queue_job.progress,
                    text,
                )

    def on_render_progress(self, done: int, total: int) -> None:
        """Surface completed preparation frames or fusion tiles with an ETA."""
        if total <= 0:
            return
        if self._render_is_cancelling():
            self.on_render_stage('cancelling')
            return
        self._control_progress = (int(done), int(total))
        done = min(max(0, done), total)
        bar = getattr(self, "_status_bar_progress", None)
        if bar is not None:
            from PyQt6 import sip
            if sip.isdeleted(bar):
                return
            # Range first, then value: Qt clamps a value outside the current
            # range, so setting the value while the bar is still in its
            # indeterminate (0..0) mode silently drops it and the bar looks
            # frozen for the rest of the run.
            bar.setRange(0, int(total))
            bar.setValue(int(done))
        now = monotonic()
        last = getattr(self, '_progress_last_t', None)
        if last is None or now - last > 0.25 or done == total:
            self._progress_last_t = now
            pct = done * 100 // total
            eta = ''
            first = getattr(self, '_progress_first_t', None)
            # The first frame carries worker/OpenCL start-up time, so an ETA
            # derived from it would flash a wildly inflated estimate.
            if first is not None and 2 <= done < total and now - first >= 0.8:
                remaining = (now - first) / done * (total - done)
                eta = f" · ~{ceil(remaining)}s"
            key = _RENDER_STAGE_KEYS.get(getattr(self, '_render_progress_stage', None), 'render_progress')
            text = f"{trans.t(key)} {pct}% ({done}/{total}){eta}"
            label = getattr(self, '_status_bar_label', None)
            if label is not None:
                label.setText(text)
            gallery = getattr(self.window, "gallery_controller", None)
            if gallery is not None and hasattr(gallery, "on_render_progress"):
                gallery.on_render_progress(done, total, text)
            if self._active_queue_job is not None:
                queue = getattr(self.window, "render_queue", None)
                if queue is not None:
                    queue.notify_job_progress(self._active_queue_job.job_id, pct, text)

    def _finish_compare_run(self) -> None:
        """End a comparison run: unlock the UI and report skipped methods."""
        window = self.window
        self._compare_mode = False
        self._stop_compare_timer()
        self._compare_token = object()
        self._compare_context = None
        self._compare_current = None
        self._restore_ui_controls()
        failures = list(getattr(self, "_compare_failures", []))
        done = self._compare_total - len(failures)
        window.statusBar().showMessage(
            trans.t('msg_compare_done').format(count=done), 10000)
        from PyQt6.QtWidgets import QWidget
        if isinstance(window, QWidget):
            QApplication.alert(window)
        from utils.sound_notify import play_completion_sound
        play_completion_sound(bool(getattr(window, "render_complete_sound", False)))
        if failures and getattr(self, "_render_interactive", True):
            details = "\n".join(f"{self.algo_display_name(method)}: {reason}"
                                 for method, reason in failures)
            show_message_box(window, trans.t("msg_warning"),
                             trans.t("msg_compare_skipped").format(
                                 done=done, total=self._compare_total),
                             details, QMessageBox.Icon.Warning)
        self._notify_queue_idle()

    def on_render_error(self, error_message: str) -> None:
        self.dismiss_live_preview()
        window = self.window
        gallery = getattr(window, "gallery_controller", None)
        if gallery is not None:
            gallery.on_render_ended()
        if (getattr(self, "_compare_cancelling", False)
                or getattr(self, "_substack_cancelling", False)
                or bool(getattr(self.worker, "is_cancelled", False))):
            error_message = "CANCELLED: render cancelled"
        try:
            stack_changed = self._stack_changed_since(self._render_context)
        except Exception:      # a partial window (tests, early startup): cannot tell
            stack_changed = False
        context = self._render_context or {}
        self.alignment_diagnostics = (
            list(getattr(self.worker, 'alignment_report', None) or [])
            if not error_message.startswith('CANCELLED') else [],
            list(context.get('filenames', [])), list(context.get('included', [])),
        )
        self._capture_performance_report()
        if (not error_message.startswith('CANCELLED') and
                getattr(self.worker, 'failure_category', None) in ('resource_limit', 'memory')):
            estimate = context.get('memory_estimate')
            if estimate:
                error_message += '\n\n' + self._memory_details(estimate)
            error_message += '\n\n' + self._memory_guidance(estimate)
        # Every exit path clears the draft flag: a failed or cancelled render
        # must not leave the controller in preview mode.
        self._preview_mode = False
        self._render_context = None
        failed_queue_job = self._active_queue_job
        self._active_queue_job = None
        self._dispose_worker()

        if (getattr(self, '_compare_mode', False)
                and not error_message.startswith("CANCELLED") and stack_changed):
            # The stack changed under the run: the remaining methods would
            # render a different stack than the results already published.
            self._abort_compare_run()
            self.last_state = "discarded"
            self.last_error = trans.t('msg_render_stack_changed')
            self.last_result = None
            self._control_progress = None
            self._restore_ui_controls()
            self._hide_progress_bar()
            window.statusBar().showMessage(self.last_error, 8000)
            return
        if (getattr(self, '_compare_mode', False)
                and not error_message.startswith("CANCELLED")):
            # One method that cannot run (not enough memory for a whole-frame
            # network, a missing model...) is skipped; the rest still render
            # and the skipped ones are listed once at the end.
            import logging
            method = self._compare_current or "?"
            logging.getLogger("openfocus").warning(
                "compare run: %s failed, continuing: %s", method, error_message)
            self._compare_failures.append((method, str(error_message)))
            self.last_error = str(error_message)
            self._control_progress = None
            if self._compare_queue:
                self._hide_progress_bar()
                self._queue_next_compare()
            else:
                self.last_state = "error"
                self.last_result = None
                self._hide_progress_bar()
                self._finish_compare_run()
            return
        if getattr(self, '_compare_mode', False):
            # Cancelled: stop the whole comparison run
            self._compare_mode = False
            self._compare_queue = []
            self._stop_compare_timer()
            self._compare_token = object()
            self._compare_context = None
            self._compare_current = None

        if getattr(self, '_substack_mode', False):
            # Stop the substack queue cleanly; already-published substack
            # outputs remain in the Output list.
            self._substack_mode = False
            self._substack_cancelling = False
            self._substack_queue = []
            self._stop_substack_timer()
            self._substack_token = object()
            self._substack_context = None
            self._substack_current = None
            self._substack_auto_restack = False
            self._substack_results = []

        self._restore_ui_controls()

        self._control_progress = None
        # Cancellation is not an error: quiet status message, no dialog
        if error_message.startswith("CANCELLED"):
            self.last_state = "cancelled"
            self.last_result = None
            self.last_error = None
            self._hide_progress_bar()
            window.statusBar().showMessage(trans.t('msg_render_cancelled'), 4000)
            if failed_queue_job is not None:
                queue = getattr(window, "render_queue", None)
                if queue is not None:
                    queue.notify_job_cancelled(failed_queue_job.job_id)
            else:
                self._notify_queue_idle()
            return
        self.last_error = str(error_message)
        self.last_state = "error"
        self.last_result = None
        self._hide_progress_bar()

        import logging
        logging.getLogger("openfocus").error("render failed: %s", error_message)

        if failed_queue_job is not None:
            detail = _failed_alignment_detail(*self.alignment_diagnostics) or str(error_message)
            window.statusBar().showMessage(f"{trans.t('msg_error')}: {detail}", 8000)
            queue = getattr(window, "render_queue", None)
            if queue is not None:
                queue.notify_job_failed(failed_queue_job.job_id, detail)
            return

        if getattr(self, "_render_interactive", True):
            # Source names belong in the local dialog, outside error logging and reporting.
            error_message = _failed_alignment_detail(*self.alignment_diagnostics) or error_message
            show_message_box(
                window,
                trans.t("msg_error"),
                trans.t("msg_proc_error"),
                error_message,
                QMessageBox.Icon.Critical,
            )
        self._notify_queue_idle()

