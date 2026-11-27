"""Tests for the non-blocking Background Render Queue, dialog, and main window integration."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QObject, pyqtSignal

from core.render_queue import (
    RenderQueueItem,
    RenderQueueManager,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    format_queue_job_label,
)
from dialogs.render_queue import RenderQueueDialog
from locales import trans


class _FakeAsyncRunner:
    """Controllable async runner to test non-blocking FIFO execution deterministically."""

    def __init__(self, queue: RenderQueueManager):
        self.queue = queue
        self.started: list[RenderQueueItem] = []
        self.cancelled: list[str] = []
        self.active: RenderQueueItem | None = None

    def start(self, item: RenderQueueItem) -> str:
        self.started.append(item)
        self.active = item
        return "started"

    def cancel(self, item: RenderQueueItem) -> None:
        self.cancelled.append(item.job_id)
        if self.active is item:
            self.active = None
            self.queue.notify_job_cancelled(item.job_id)

    def finish_active(self, *, result_index: int = 0, result_name: str = "out.png", output_id: str = ""):
        assert self.active is not None
        job = self.active
        self.active = None
        self.queue.notify_job_completed(
            job.job_id,
            result_index=result_index,
            result_name=result_name,
            result_output_id=output_id or f"id-{job.job_id}",
        )

    def fail_active(self, error: str):
        assert self.active is not None
        job = self.active
        self.active = None
        self.queue.notify_job_failed(job.job_id, error)


def test_fifo_execution_with_distinct_methods_params_and_subsets(qapp):
    queue = RenderQueueManager()
    runner = _FakeAsyncRunner(queue)
    queue.set_handlers(runner=runner.start, cancel_runner=runner.cancel)

    events: list[tuple] = []
    queue.job_started.connect(lambda jid: events.append(("started", jid)))
    queue.job_completed.connect(lambda jid, it: events.append(("completed", jid, it.result_name)))
    queue.all_jobs_finished.connect(lambda: events.append(("all_done",)))

    # Mutating caller dicts after enqueue must NOT affect the frozen snapshot.
    params_b = {"radius": 8, "smoothing": 4}
    opts = {"exposure_match": True, "base_correct": False, "reverse_order": False,
            "sharpen_output": False, "sharpen_strength": 50, "preview": False}

    id1 = queue.enqueue(RenderQueueItem(
        method="openfocus_a", align_method="ecc", method_params={}, align_params={},
        frame_indices=(0, 1, 2, 3), options=opts,
    ))
    # Job 1 is now running; queue jobs 2 and 3 while job 1 is in flight!
    assert queue.is_running()
    assert queue.active_job().job_id == id1
    assert queue.pending_count() == 0

    id2 = queue.enqueue(RenderQueueItem(
        method="openfocus_b", align_method="ecc", method_params=params_b, align_params={},
        frame_indices=(0, 2, 3), options=opts,
    ))
    params_b["radius"] = 99  # mutate original dict after enqueue
    opts["reverse_order"] = True

    id3 = queue.enqueue(RenderQueueItem(
        method="openfocus_c", align_method="none", method_params={"smoothing": 2}, align_params={},
        frame_indices=(1, 2), options={"exposure_match": False, "reverse_order": True},
    ))

    assert queue.pending_count() == 2
    assert queue.active_and_pending_count() == 3
    assert queue.get_job(id2).method_params["radius"] == 8
    assert queue.get_job(id2).options["reverse_order"] is False
    assert queue.get_job(id2).frame_indices == (0, 2, 3)
    assert queue.get_job(id3).frame_indices == (1, 2)

    # Progress reporting on active job 1
    progress_seen = []
    queue.job_progress.connect(lambda jid, pct, txt: progress_seen.append((jid, pct, txt)))
    queue.notify_job_progress(id1, 45, "Fusing 45%")
    assert progress_seen == [(id1, 45, "Fusing 45%")]
    assert queue.get_job(id1).progress == 45

    # Finish job 1 -> job 2 immediately starts
    runner.finish_active(result_index=0, result_name="OpenFocus_A_001.png")
    assert queue.get_job(id1).status == STATUS_COMPLETED
    assert queue.get_job(id1).result_name == "OpenFocus_A_001.png"
    assert queue.active_job().job_id == id2
    assert queue.pending_count() == 1

    # Finish job 2 -> job 3 immediately starts
    runner.finish_active(result_index=0, result_name="OpenFocus_B_002.png")
    assert queue.active_job().job_id == id3
    assert queue.pending_count() == 0

    # Finish job 3 -> all_jobs_finished fires
    runner.finish_active(result_index=0, result_name="OpenFocus_C_003.png")
    assert not queue.is_running()
    assert queue.pending_count() == 0
    assert [it.job_id for it in runner.started] == [id1, id2, id3]
    assert events[-1] == ("all_done",)


def test_reorder_queued_jobs_changes_execution_order(qapp):
    queue = RenderQueueManager()
    runner = _FakeAsyncRunner(queue)
    queue.set_handlers(runner=runner.start, cancel_runner=runner.cancel)

    id1 = queue.enqueue(RenderQueueItem(method="openfocus_a", align_method="none"))
    id2 = queue.enqueue(RenderQueueItem(method="openfocus_b", align_method="none"))
    id3 = queue.enqueue(RenderQueueItem(method="openfocus_c", align_method="none"))
    id4 = queue.enqueue(RenderQueueItem(method="guided_filter", align_method="none"))

    # Cannot move the currently running job
    assert not queue.move_job(id1, 1)
    # Move job 4 up twice so order of queued becomes [id4, id2, id3]
    assert queue.move_job(id4, -1)
    assert queue.move_job(id4, -1)
    assert not queue.move_job(id4, -1)  # already first queued
    assert [it.job_id for it in queue.items()] == [id1, id4, id2, id3]

    # Move job 2 down -> [id4, id3, id2]
    assert queue.move_job(id2, 1)
    assert [it.job_id for it in queue.items()] == [id1, id4, id3, id2]

    for _ in range(4):
        runner.finish_active()
    assert [it.job_id for it in runner.started] == [id1, id4, id3, id2]


def test_cancel_queued_vs_cancel_running_job(qapp):
    queue = RenderQueueManager()
    runner = _FakeAsyncRunner(queue)
    queue.set_handlers(runner=runner.start, cancel_runner=runner.cancel)

    cancelled_signals = []
    queue.job_cancelled.connect(cancelled_signals.append)

    id1 = queue.enqueue(RenderQueueItem(method="openfocus_a", align_method="none"))
    id2 = queue.enqueue(RenderQueueItem(method="openfocus_b", align_method="none"))
    id3 = queue.enqueue(RenderQueueItem(method="openfocus_c", align_method="none"))

    # Cancel queued job 2: running job 1 stays active, job 2 marked cancelled
    assert queue.cancel_job(id2)
    assert queue.get_job(id2).status == STATUS_CANCELLED
    assert queue.active_job().job_id == id1
    assert queue.pending_count() == 1
    assert cancelled_signals == [id2]

    # Cancel running job 1: cancel_runner called and job 3 starts immediately
    assert queue.cancel_job(id1)
    assert runner.cancelled == [id1]
    assert queue.get_job(id1).status == STATUS_CANCELLED
    assert queue.active_job().job_id == id3

    runner.finish_active(result_name="job3.png")
    assert queue.get_job(id3).status == STATUS_COMPLETED
    assert [it.job_id for it in runner.started] == [id1, id3]


def test_failure_isolation_continues_to_subsequent_jobs(qapp):
    queue = RenderQueueManager()
    runner = _FakeAsyncRunner(queue)
    queue.set_handlers(runner=runner.start, cancel_runner=runner.cancel)

    failures = []
    all_done = []
    queue.job_failed.connect(lambda jid, err: failures.append((jid, err)))
    queue.all_jobs_finished.connect(lambda: all_done.append(True))

    id1 = queue.enqueue(RenderQueueItem(method="openfocus_a", align_method="none"))
    id2 = queue.enqueue(RenderQueueItem(method="openfocus_b", align_method="none"))
    id3 = queue.enqueue(RenderQueueItem(method="openfocus_c", align_method="none"))

    runner.finish_active(result_name="ok1.png")
    # Job 2 fails -> job 3 MUST still execute!
    runner.fail_active("Simulated fusion error")
    assert queue.get_job(id2).status == STATUS_FAILED
    assert queue.get_job(id2).error_message == "Simulated fusion error"
    assert failures == [(id2, "Simulated fusion error")]
    assert queue.active_job().job_id == id3

    runner.finish_active(result_name="ok3.png")
    assert queue.get_job(id3).status == STATUS_COMPLETED
    assert not queue.is_running()
    assert all_done == [True]


def test_pause_resume_remove_and_clear_finished(qapp):
    queue = RenderQueueManager()
    runner = _FakeAsyncRunner(queue)
    queue.set_handlers(runner=runner.start, cancel_runner=runner.cancel)

    id1 = queue.enqueue(RenderQueueItem(method="openfocus_a", align_method="none"))
    id2 = queue.enqueue(RenderQueueItem(method="openfocus_b", align_method="none"))

    queue.pause()
    assert queue.is_paused()
    # Finishing job 1 while paused must NOT auto-start job 2
    runner.finish_active(result_name="ok1.png")
    assert not queue.is_running()
    assert queue.get_job(id2).status == STATUS_QUEUED

    # Clear finished removes job 1 but keeps queued job 2
    assert queue.clear_finished() == 1
    assert [it.job_id for it in queue.items()] == [id2]

    # Resuming kicks off job 2
    queue.resume()
    assert not queue.is_paused()
    assert queue.active_job().job_id == id2

    # Cannot remove a running job
    assert not queue.remove_job(id2)
    runner.finish_active(result_name="ok2.png")
    assert queue.remove_job(id2)
    assert queue.items() == []


def test_label_formatting_includes_params_and_frames():
    label = format_queue_job_label(
        "openfocus_b", {"radius": 8, "smoothing": 4}, align_method="ecc", frame_count=92
    )
    assert "r=8" in label and "s=4" in label and "92" in label


@pytest.mark.parametrize("theme", ["light", "dark", "gray"])
def test_render_queue_dialog_interactions_and_themes(qapp, theme):
    queue = RenderQueueManager()
    runner = _FakeAsyncRunner(queue)
    queue.set_handlers(runner=runner.start, cancel_runner=runner.cancel)

    id1 = queue.enqueue(RenderQueueItem(
        label="OpenFocus - A · 24 frames", method="openfocus_a", align_method="ecc",
        options={"exposure_match": True},
    ))
    id2 = queue.enqueue(RenderQueueItem(
        label="OpenFocus - B (r=8, s=4) · 92 frames", method="openfocus_b", align_method="ecc",
        method_params={"radius": 8, "smoothing": 4}, options={"exposure_match": True},
    ))
    id3 = queue.enqueue(RenderQueueItem(
        label="OpenFocus - C (s=2) · 45 frames", method="openfocus_c", align_method="none",
        method_params={"smoothing": 2},
    ))

    # Finish job 1 so we have Completed + Running + Queued simultaneously
    runner.finish_active(result_index=0, result_name="OpenFocus_A_001.png")
    queue.notify_job_progress(id2, 45, "Fusing 45% (41/92)")

    dialog = RenderQueueDialog(queue, theme=theme)
    viewed = []
    enqueued_clicks = []
    dialog.view_result_requested.connect(viewed.append)
    dialog.enqueue_current_requested.connect(lambda: enqueued_clicks.append(True))
    try:
        dialog.show()
        qapp.processEvents()

        # Verify all 3 cards rendered with expected controls
        card1 = dialog.card_for(id1)
        card2 = dialog.card_for(id2)
        card3 = dialog.card_for(id3)
        assert card1 is not None and card1.btn_view is not None
        assert card2 is not None and card2.progress_bar is not None
        assert card2.progress_bar.value() == 45
        assert "45%" in card2.status_pill.text()
        assert card3 is not None and card3.btn_cancel is not None
        assert "radio_no_align" not in card3.subtitle_label.text()
        assert trans.t("check_align_none") in card3.subtitle_label.text()

        # Live progress update updates active card in place
        queue.notify_job_progress(id2, 72, "Fusing 72% (66/92)")
        assert card2.progress_bar.value() == 72
        assert "72%" in card2.status_pill.text()

        # Clicking View Result emits job id
        card1.btn_view.click()
        assert viewed == [id1]

        # Clicking Pause / Resume toggles queue pause
        dialog.btn_pause_resume.click()
        assert queue.is_paused()
        dialog.btn_pause_resume.click()
        assert not queue.is_paused()

        # Clicking Enqueue Current Settings emits signal
        dialog.btn_enqueue_current.click()
        assert enqueued_clicks == [True]

        # Clicking Clear Finished removes completed job 1
        dialog.btn_clear_finished.click()
        assert dialog.card_for(id1) is None
        assert dialog.card_for(id2) is not None
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_main_window_enqueue_render_job_non_blocking_e2e(qapp, tmp_path, monkeypatch):
    """End-to-end main window wiring: non-blocking UI, FIFO snapshots, output insertion & View Result."""
    from tests.test_control_channel import _destroy_main_window, _new_main_window
    import controllers.render_manager as rm_mod

    created_workers = []

    class _MockRenderWorker(QObject):
        finished_signal = pyqtSignal(object, object, bool, float, float, str)
        error_signal = pyqtSignal(str)
        progress_signal = pyqtSignal(int, int)
        stage_signal = pyqtSignal(str)

        def __init__(self, source_images, *args, **kwargs):
            super().__init__()
            self.source_images = list(source_images)
            self.algorithm = kwargs.get("algorithm")
            self.alignment = kwargs.get("alignment")
            self.fusion_parameters = dict(kwargs.get("fusion_parameters") or {})
            self.is_cancelled = False
            self._running = False
            self.alignment_reused = False
            self.registration_details = None
            self.fusion_details = None
            self.rendered_depth = None
            self.alignment_report = []
            self.performance_report = None
            self.prepared_cache = None
            created_workers.append(self)

        def isRunning(self):
            return self._running

        def start(self):
            self._running = True
            self.stage_signal.emit("fuse")
            self.progress_signal.emit(1, 2)

        def cancel(self):
            self.is_cancelled = True
            self._running = False
            self.error_signal.emit("CANCELLED: render cancelled")

        def complete(self, fill_value: int):
            self._running = False
            h, w = self.source_images[0].shape[:2]
            out = np.full((h, w, 3), fill_value, dtype=np.uint8)
            self.finished_signal.emit(self.source_images, out, False, 0.0, 0.12, "CPU")

        def fail(self, msg: str):
            self._running = False
            self.error_signal.emit(msg)

    monkeypatch.setattr(rm_mod, "RenderProcessWorker", _MockRenderWorker)

    window = _new_main_window(qapp, tmp_path, monkeypatch, use_gpu=False)
    try:
        # Load 4 synthetic frames
        frames = [np.full((32, 32, 3), 20 * (i + 1), dtype=np.uint8) for i in range(4)]
        window.raw_images = frames
        window.image_filenames = [f"img_{i}.png" for i in range(4)]
        window.source_content_revision = 1
        window.source_revision = 1
        window.set_alignment_method("none")
        window.set_fusion_method("openfocus_a")

        # Enqueue Job 1 (OpenFocus A, all 4 frames)
        job1 = window.enqueue_render_job()
        assert job1 is not None
        assert len(created_workers) == 1
        assert created_workers[0].algorithm == "openfocus_a"
        assert len(created_workers[0].source_images) == 4

        # NON-BLOCKING CHECK: right-panel method/alignment/parameter controls MUST remain enabled!
        assert window.combo_fusion.isEnabled()
        assert window.combo_align.isEnabled()
        assert window.fusion_parameter_panel.isEnabled()
        assert window.btn_reset.isEnabled()
        assert window.btn_enqueue.isEnabled()
        assert window.btn_render_queue.isEnabled()
        assert "1" in window.btn_render_queue.full_text()

        # While Job 1 is running, switch method, edit parameters, and change frame exclusion for Job 2!
        window.set_fusion_method("openfocus_b")
        window.set_fusion_parameters("openfocus_b", {"radius": 8, "smoothing": 4})
        window.excluded_frames = {3}  # only frames 0, 1, 2 for Job 2
        window.source_revision += 1

        job2 = window.enqueue_render_job()
        assert job2 is not None
        assert "2" in window.btn_render_queue.full_text()

        # Queue Job 3 with explicit substack & method override while Job 1 still runs
        job3 = window.enqueue_render_job(
            method="openfocus_c",
            align_method="none",
            method_params={"smoothing": 2},
            frame_indices=(1, 2, 3),
        )
        assert job3 is not None
        assert "3" in window.btn_render_queue.full_text()
        assert len(created_workers) == 1  # Still only Job 1 running

        # Complete Job 1: must NOT be discarded despite UI edits, must publish output, and start Job 2!
        created_workers[0].complete(111)
        qapp.processEvents()
        assert window.render_queue.get_job(job1).status == STATUS_COMPLETED
        assert window.output_list.count() == 1
        assert len(created_workers) == 2
        assert created_workers[1].algorithm == "openfocus_b"
        assert created_workers[1].fusion_parameters["radius"] == 8
        assert created_workers[1].fusion_parameters["smoothing"] == 4
        assert len(created_workers[1].source_images) == 3

        # Fail Job 2 -> failure isolation must still start Job 3!
        created_workers[1].fail("Job 2 failure test")
        qapp.processEvents()
        assert window.render_queue.get_job(job2).status == STATUS_FAILED
        assert len(created_workers) == 3
        assert created_workers[2].algorithm == "openfocus_c"
        assert len(created_workers[2].source_images) == 3

        # Complete Job 3 -> output list now has 2 outputs, badge returns to idle label
        created_workers[2].complete(222)
        qapp.processEvents()
        assert window.render_queue.get_job(job3).status == STATUS_COMPLETED
        assert window.output_list.count() == 2
        assert window.btn_render_queue.full_text() == trans.t("btn_render_queue")

        # Verify View Result jumps back to Job 1's output (which shifted from row 0 to row 1!)
        assert window.output_list.currentRow() == 0  # currently showing Job 3
        assert window._view_queue_result(job1)
        assert window.output_list.currentRow() == 1
        assert int(window.fusion_result[0, 0, 0]) == 111

        # Verify Tools menu action opens the non-modal Render Queue dialog
        assert "action_render_queue" in window.ui_objs
        window.ui_objs["action_render_queue"].trigger()
        qapp.processEvents()
        assert window._render_queue_dialog is not None
        assert window._render_queue_dialog.isVisible()
        assert not window._render_queue_dialog.isModal()
    finally:
        _destroy_main_window(qapp, window)
