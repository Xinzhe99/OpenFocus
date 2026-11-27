"""Non-blocking FIFO background render queue for OpenFocus.

Allows users to enqueue snapshots of fusion methods, parameters, alignment
choices, options, and optional frame subsets without locking the main
workspace controls. Jobs run sequentially through the application's existing
render pipeline and report progress and outcomes via Qt signals.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Optional
import uuid

from PyQt6.QtCore import QObject, pyqtSignal

from core.algorithms import get_fusion, get_registration
from locales import trans

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
FINISHED_STATUSES = (STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED)

_SHORT_PARAM_NAMES = {
    "radius": "r",
    "smoothing": "s",
    "feather": "f",
    "kernel_size": "k",
    "levels": "L",
    "mode": "mode",
    "reference": "ref",
}


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def format_queue_job_label(
    method: str,
    method_params: Optional[dict] = None,
    *,
    align_method: str = "none",
    frame_count: Optional[int] = None,
    options: Optional[dict] = None,
) -> str:
    """Build a concise human-readable label for a queued OpenFocus job."""
    if method == "none":
        base = trans.t("radio_no_fusion")
    else:
        try:
            base = trans.t(get_fusion(method).label_key)
        except Exception:
            base = str(method)

    params = method_params or {}
    parts = []
    if method != "none":
        try:
            spec = get_fusion(method)
            ordered_keys = [p.key for p in spec.parameters]
        except Exception:
            ordered_keys = list(params.keys())
        for key in ordered_keys:
            if key not in params or params[key] is None:
                continue
            short = _SHORT_PARAM_NAMES.get(key, key)
            parts.append(f"{short}={_format_param_value(params[key])}")
        for key, value in params.items():
            if key in ordered_keys or value is None:
                continue
            short = _SHORT_PARAM_NAMES.get(key, key)
            parts.append(f"{short}={_format_param_value(value)}")

    if parts:
        title = f"{base} ({', '.join(parts)})"
    else:
        title = base

    suffixes = []
    if frame_count is not None and frame_count > 0:
        suffixes.append(trans.t("render_queue_frames_count").format(count=int(frame_count)))
    if options and options.get("preview"):
        suffixes.append(trans.t("render_queue_preview_tag"))
    if method == "none" and align_method and align_method != "none":
        try:
            align_name = trans.t(get_registration(align_method).label_key)
        except Exception:
            align_name = str(align_method)
        suffixes.insert(0, align_name)

    if suffixes:
        return f"{title} · {' · '.join(suffixes)}"
    return title


@dataclass
class RenderQueueItem:
    """Immutable-input snapshot plus mutable execution state for one queued job."""

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    label: str = ""
    method: str = "openfocus_b"
    align_method: str = "ecc"
    method_params: dict = field(default_factory=dict)
    align_params: dict = field(default_factory=dict)
    frame_indices: Optional[tuple[int, ...]] = None
    options: dict = field(default_factory=dict)
    status: str = STATUS_QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    elapsed_seconds: float = 0.0
    result_index: Optional[int] = None
    result_name: str = ""
    error_message: str = ""
    result_output_id: str = ""
    progress: int = 0
    progress_text: str = ""
    frame_count: int = 0
    source_content_revision: Optional[int] = None

    def snapshot_copy(self) -> "RenderQueueItem":
        """Normalize mutable fields so later UI edits cannot mutate this job."""
        indices = (
            tuple(int(i) for i in self.frame_indices)
            if self.frame_indices is not None
            else None
        )
        frame_count = self.frame_count or (len(indices) if indices is not None else 0)
        method_params = deepcopy(dict(self.method_params or {}))
        align_params = deepcopy(dict(self.align_params or {}))
        options = deepcopy(dict(self.options or {}))
        label = self.label or format_queue_job_label(
            self.method,
            method_params,
            align_method=self.align_method,
            frame_count=frame_count or None,
            options=options,
        )
        return RenderQueueItem(
            job_id=self.job_id or uuid.uuid4().hex[:12],
            label=label,
            method=self.method,
            align_method=self.align_method,
            method_params=method_params,
            align_params=align_params,
            frame_indices=indices,
            options=options,
            status=STATUS_QUEUED,
            created_at=self.created_at or time.time(),
            started_at=None,
            finished_at=None,
            elapsed_seconds=0.0,
            result_index=None,
            result_name="",
            error_message="",
            result_output_id="",
            progress=0,
            progress_text="",
            frame_count=frame_count,
            source_content_revision=self.source_content_revision,
        )


class RenderQueueManager(QObject):
    """Manages FIFO background execution of queued OpenFocus render jobs."""

    queue_changed = pyqtSignal()
    job_started = pyqtSignal(str)
    job_progress = pyqtSignal(str, int, str)
    job_completed = pyqtSignal(str, object)
    job_failed = pyqtSignal(str, str)
    job_cancelled = pyqtSignal(str)
    all_jobs_finished = pyqtSignal()

    def __init__(
        self,
        *,
        runner: Optional[Callable[[RenderQueueItem], Any]] = None,
        cancel_runner: Optional[Callable[[RenderQueueItem], None]] = None,
        is_pipeline_busy: Optional[Callable[[], bool]] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._runner = runner
        self._cancel_runner = cancel_runner
        self._is_pipeline_busy = is_pipeline_busy
        self._items: list[RenderQueueItem] = []
        self._active_job_id: Optional[str] = None
        self._paused: bool = False
        self._draining: bool = False

    def set_handlers(
        self,
        *,
        runner: Optional[Callable[[RenderQueueItem], Any]] = None,
        cancel_runner: Optional[Callable[[RenderQueueItem], None]] = None,
        is_pipeline_busy: Optional[Callable[[], bool]] = None,
    ) -> None:
        if runner is not None:
            self._runner = runner
        if cancel_runner is not None:
            self._cancel_runner = cancel_runner
        if is_pipeline_busy is not None:
            self._is_pipeline_busy = is_pipeline_busy

    # ------------------------------------------------------------------
    # Public queue operations
    # ------------------------------------------------------------------

    def enqueue(self, item: RenderQueueItem) -> str:
        """Append a frozen job snapshot and start immediately if idle."""
        frozen = item.snapshot_copy()
        while self._find(frozen.job_id) is not None:
            frozen.job_id = uuid.uuid4().hex[:12]
        self._items.append(frozen)
        self.queue_changed.emit()
        self._drain_next()
        return frozen.job_id

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job immediately or request cancellation of the running job."""
        item = self._find(job_id)
        if item is None:
            return False
        if item.status == STATUS_QUEUED:
            now = time.time()
            item.status = STATUS_CANCELLED
            item.finished_at = now
            item.elapsed_seconds = 0.0
            self.job_cancelled.emit(job_id)
            self.queue_changed.emit()
            return True
        if item.status == STATUS_RUNNING and self._active_job_id == job_id:
            if callable(self._cancel_runner):
                self._cancel_runner(item)
            else:
                self.notify_job_cancelled(job_id)
            return True
        return False

    def remove_job(self, job_id: str) -> bool:
        """Remove a non-running job from the queue history."""
        for idx, item in enumerate(self._items):
            if item.job_id == job_id:
                if item.status == STATUS_RUNNING:
                    return False
                self._items.pop(idx)
                self.queue_changed.emit()
                return True
        return False

    def move_job(self, job_id: str, delta: int) -> bool:
        """Reorder a queued job up (delta < 0) or down (delta > 0) among queued items."""
        if delta == 0:
            return False
        queued_positions = [
            idx for idx, item in enumerate(self._items) if item.status == STATUS_QUEUED
        ]
        target_pos = None
        for q_idx, list_idx in enumerate(queued_positions):
            if self._items[list_idx].job_id == job_id:
                target_pos = q_idx
                break
        if target_pos is None:
            return False
        new_pos = max(0, min(len(queued_positions) - 1, target_pos + int(delta)))
        if new_pos == target_pos:
            return False

        # Reorder only within the slots occupied by queued items so running and
        # finished items never jump out of order.
        queued_items = [self._items[i] for i in queued_positions]
        moved = queued_items.pop(target_pos)
        queued_items.insert(new_pos, moved)
        for slot, item in zip(queued_positions, queued_items):
            self._items[slot] = item
        self.queue_changed.emit()
        return True

    def clear_finished(self) -> int:
        """Remove completed, failed, and cancelled jobs from the queue list."""
        before = len(self._items)
        self._items = [
            item for item in self._items if item.status not in FINISHED_STATUSES
        ]
        removed = before - len(self._items)
        if removed:
            self.queue_changed.emit()
        return removed

    def pause(self) -> None:
        if not self._paused:
            self._paused = True
            self.queue_changed.emit()

    def resume(self) -> None:
        if self._paused:
            self._paused = False
            self.queue_changed.emit()
            self._drain_next()

    def is_paused(self) -> bool:
        return self._paused

    def items(self) -> list[RenderQueueItem]:
        return list(self._items)

    def get_job(self, job_id: str) -> Optional[RenderQueueItem]:
        return self._find(job_id)

    def active_job(self) -> Optional[RenderQueueItem]:
        if self._active_job_id is None:
            return None
        return self._find(self._active_job_id)

    def pending_count(self) -> int:
        return sum(1 for item in self._items if item.status == STATUS_QUEUED)

    def active_and_pending_count(self) -> int:
        return sum(
            1 for item in self._items if item.status in (STATUS_QUEUED, STATUS_RUNNING)
        )

    def is_running(self) -> bool:
        return self._active_job_id is not None

    def cancel_pending_jobs(self) -> int:
        """Cancel every currently queued job (for shutdown or stack clear)."""
        cancelled = []
        now = time.time()
        for item in self._items:
            if item.status == STATUS_QUEUED:
                item.status = STATUS_CANCELLED
                item.finished_at = now
                item.elapsed_seconds = 0.0
                cancelled.append(item.job_id)
        for job_id in cancelled:
            self.job_cancelled.emit(job_id)
        if cancelled:
            self.queue_changed.emit()
        return len(cancelled)

    # ------------------------------------------------------------------
    # Pipeline callbacks
    # ------------------------------------------------------------------

    def notify_job_progress(
        self, job_id: str, percent: int, stage_text: str = ""
    ) -> None:
        item = self._find(job_id)
        if item is None or item.status != STATUS_RUNNING:
            return
        pct = max(0, min(100, int(percent)))
        item.progress = pct
        if stage_text:
            item.progress_text = str(stage_text)
        self.job_progress.emit(job_id, pct, item.progress_text)

    def notify_job_completed(
        self,
        job_id: str,
        *,
        result_index: Optional[int] = None,
        result_name: str = "",
        result_output_id: str = "",
    ) -> None:
        item = self._find(job_id)
        if item is None:
            return
        now = time.time()
        item.status = STATUS_COMPLETED
        item.progress = 100
        item.finished_at = now
        if item.started_at is not None:
            item.elapsed_seconds = max(0.0, now - item.started_at)
        item.result_index = result_index
        item.result_name = str(result_name or "")
        item.result_output_id = str(result_output_id or "")
        item.error_message = ""
        if self._active_job_id == job_id:
            self._active_job_id = None
        self.job_completed.emit(job_id, item)
        self.queue_changed.emit()
        self._after_job_ended()

    def notify_job_failed(self, job_id: str, error_message: str) -> None:
        item = self._find(job_id)
        if item is None:
            return
        now = time.time()
        item.status = STATUS_FAILED
        item.finished_at = now
        if item.started_at is not None:
            item.elapsed_seconds = max(0.0, now - item.started_at)
        item.error_message = str(error_message or "")
        if self._active_job_id == job_id:
            self._active_job_id = None
        self.job_failed.emit(job_id, item.error_message)
        self.queue_changed.emit()
        self._after_job_ended()

    def notify_job_cancelled(self, job_id: str) -> None:
        item = self._find(job_id)
        if item is None:
            return
        now = time.time()
        item.status = STATUS_CANCELLED
        item.finished_at = now
        if item.started_at is not None:
            item.elapsed_seconds = max(0.0, now - item.started_at)
        if self._active_job_id == job_id:
            self._active_job_id = None
        self.job_cancelled.emit(job_id)
        self.queue_changed.emit()
        self._after_job_ended()

    def notify_pipeline_idle(self) -> None:
        """Kick off queued jobs once an external manual render completes."""
        self._drain_next()

    # ------------------------------------------------------------------
    # Internal FIFO scheduler
    # ------------------------------------------------------------------

    def _find(self, job_id: str) -> Optional[RenderQueueItem]:
        for item in self._items:
            if item.job_id == job_id:
                return item
        return None

    def _next_queued(self) -> Optional[RenderQueueItem]:
        for item in self._items:
            if item.status == STATUS_QUEUED:
                return item
        return None

    def _pipeline_busy(self) -> bool:
        if callable(self._is_pipeline_busy):
            try:
                return bool(self._is_pipeline_busy())
            except Exception:
                return False
        return False

    def _after_job_ended(self) -> None:
        if self._draining:
            return
        if self.pending_count() == 0:
            self.all_jobs_finished.emit()
            return
        if not self._paused:
            self._drain_next()

    def _drain_next(self) -> None:
        if self._draining:
            return
        self._draining = True
        try:
            ran_any = False
            while not self._paused and self._active_job_id is None:
                if self._pipeline_busy():
                    break
                item = self._next_queued()
                if item is None:
                    break
                if self._runner is None:
                    # Without a runner attached (e.g. UI preview), keep items queued.
                    break
                item.status = STATUS_RUNNING
                item.started_at = time.time()
                item.progress = 0
                item.progress_text = ""
                self._active_job_id = item.job_id
                ran_any = True
                self.job_started.emit(item.job_id)
                self.queue_changed.emit()
                try:
                    outcome = self._runner(item)
                except Exception as exc:
                    if self._active_job_id == item.job_id and item.status == STATUS_RUNNING:
                        self.notify_job_failed(item.job_id, str(exc))
                    continue

                if outcome == "busy":
                    # Pipeline became busy right before launch: keep job queued.
                    if self._active_job_id == item.job_id and item.status == STATUS_RUNNING:
                        item.status = STATUS_QUEUED
                        item.started_at = None
                        self._active_job_id = None
                        self.queue_changed.emit()
                    break
                if outcome in (None, True, "started", "preflight", "loading"):
                    # Async runner is active, OR sync runner already called
                    # notify_job_completed/failed/cancelled during _runner(item).
                    if self._active_job_id is not None:
                        break
                    continue
                # Any other string/False return represents an immediate start failure.
                if self._active_job_id == item.job_id and item.status == STATUS_RUNNING:
                    reason = str(outcome) if outcome is not False else "Render failed to start"
                    self.notify_job_failed(item.job_id, reason)
            if ran_any and self._active_job_id is None and self.pending_count() == 0:
                self.all_jobs_finished.emit()
        finally:
            self._draining = False
