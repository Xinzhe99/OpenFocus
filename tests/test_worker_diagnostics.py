"""Behavioral diagnostics for worker failures, cancellation and partial output."""
import json
import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from core.compare_worker import AlignmentWorker, PreviewWorker
from core.workers import BatchWorker, PartialWriteError, RenderWorker, StackLoadWorker
from utils.image_utils import imwrite_auto
from utils.logging_setup import get_logger


@pytest.fixture
def diagnostics(qapp):
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = get_logger()
    previous_level = logger.level
    handler = Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        handler.close()


def _events(records, name):
    return [record for record in records
            if getattr(record, "event", "").startswith(name + ".")]


def _details(record):
    return json.loads(record.getMessage().split(" ", 1)[1])


def _assert_outcome(records, name, status):
    events = _events(records, name)
    assert [record.event for record in events] == [name + ".started", name + "." + status]
    assert events[0].operation_id == events[1].operation_id != "-"
    assert events[0].session_id == events[1].session_id
    assert _details(events[1])["elapsed_ms"] >= 0
    return events


def _frames():
    return [np.full((8, 12, 3), value, np.uint16) for value in (1234, 5678)]


def _render():
    return RenderWorker(_frames(), algorithm="openfocus_a", alignment="none",
                        fusion_parameters={"kernel_size": 5}, use_gpu=False,
                        tile_enabled=False, thread_count=1)


def _capture_render(worker):
    results, errors = [], []
    worker.finished_signal.connect(lambda *args: results.append(args))
    worker.error_signal.connect(errors.append)
    return results, errors


def test_render_failure_retains_traceback_and_error_signal(monkeypatch, diagnostics):
    worker = _render()
    results, errors = _capture_render(worker)

    def fail(*args, **kwargs):
        raise RuntimeError("fusion backend failed")

    monkeypatch.setattr(worker, "_run_fusion", fail)
    worker.run()

    assert results == []
    assert errors == ["fusion backend failed"]
    started, failed = _assert_outcome(diagnostics, "render", "failed")
    assert failed.exc_info[0] is RuntimeError
    assert str(failed.exc_info[1]) == "fusion backend failed"
    assert failed.exc_info[2] is not None
    details = _details(started)
    assert details["method"] == "openfocus_a"
    assert details["parameters"]["kernel_size"] == 5
    assert details["registration"] == []
    assert details["frame_count"] == 2
    assert details["first_shape"] == [8, 12, 3]
    assert details["first_dtype"] == "uint16"


@pytest.mark.parametrize("cancel_before_run", [True, False])
def test_cancelled_render_has_no_success_or_failure(monkeypatch, diagnostics, cancel_before_run):
    worker = _render()
    results, errors = _capture_render(worker)
    calls = []

    def fuse(images, **kwargs):
        calls.append(True)
        worker.cancel()
        return images[0], "CPU"

    monkeypatch.setattr(worker, "_run_fusion", fuse)
    if cancel_before_run:
        worker.cancel()
    worker.run()

    assert results == []
    assert errors == ["CANCELLED: render cancelled"]
    assert bool(calls) is not cancel_before_run
    _, cancelled = _assert_outcome(diagnostics, "render", "cancelled")
    assert cancelled.exc_info is None


def test_comparison_alignment_failure_keeps_signal_and_traceback(monkeypatch, diagnostics):
    import core.compare_worker as comparison

    worker = AlignmentWorker(_frames(), "openfocus", {"thread_count": 1})
    results, errors = [], []
    worker.ready.connect(results.append)
    worker.failed.connect(errors.append)

    def fail(*args, **kwargs):
        raise ValueError("registration failed")

    monkeypatch.setattr(comparison, "align_frames", fail)
    worker.run()

    assert results == []
    assert errors == ["registration failed"]
    _, failed = _assert_outcome(diagnostics, "comparison_alignment", "failed")
    assert failed.exc_info[0] is ValueError
    assert failed.exc_info[2] is not None


def test_comparison_partial_failure_continues_and_fails_overall(monkeypatch, diagnostics):
    worker = PreviewWorker(_frames(), ["openfocus_a", "openfocus_b"], {}, {}, None, False)
    results, errors, progress = [], [], []
    worker.result_ready.connect(lambda *args: results.append(args))
    worker.method_failed.connect(lambda *args: errors.append(args))
    worker.progress.connect(lambda *args: progress.append(args))
    expected = worker.images[0]

    def fuse(method):
        if method == "openfocus_a":
            raise RuntimeError("first method failed")
        return expected, (0, 0, 12, 8)

    monkeypatch.setattr(worker, "_fuse", fuse)
    worker.run()

    assert errors == [("openfocus_a", "first method failed")]
    assert len(results) == 1 and results[0][0] == "openfocus_b"
    assert results[0][1] is expected
    assert results[0][2] == (0, 0, 12, 8)
    assert progress == [(0, 2, "openfocus_a"), (1, 2, "openfocus_b"), (2, 2, "")]
    _assert_outcome(diagnostics, "comparison_preview", "failed")
    methods = _events(diagnostics, "comparison_method")
    assert [record.event for record in methods] == [
        "comparison_method.started", "comparison_method.failed",
        "comparison_method.started", "comparison_method.completed"]
    assert methods[1].exc_info[0] is RuntimeError
    assert _details(methods[0])["method"] == "openfocus_a"
    assert _details(methods[2])["method"] == "openfocus_b"
    parent = _events(diagnostics, "comparison_preview")[0].operation_id
    assert _details(methods[0])["parent_operation_id"] == parent
    assert _details(methods[2])["parent_operation_id"] == parent


@pytest.mark.parametrize("raise_on_write", [False, True])
def test_partial_batch_retains_artifacts_continues_and_records_failure(
        monkeypatch, tmp_path, diagnostics, raise_on_write):
    import utils.image_utils as image_utils

    folders = [tmp_path / name for name in ("partial", "complete")]
    for folder in folders:
        folder.mkdir()
        for index, image in enumerate(_frames(), 1):
            assert imwrite_auto(str(folder / f"frame{index}.png"), image, exif=b"")
    write = image_utils.imwrite_auto

    def partial_write(path, *args, **kwargs):
        destination = Path(path)
        if destination.parent.name == "partial_aligned" and destination.name == "frame2.png":
            if raise_on_write:
                raise OSError("output disk failed")
            return False
        return write(path, *args, **kwargs)

    monkeypatch.setattr(image_utils, "imwrite_auto", partial_write)
    worker = BatchWorker([str(folder) for folder in folders], "custom", str(tmp_path / "out"),
                         {"format": "png", "fusion_method": "none", "reg_methods": []},
                         use_gpu=False, thread_count=1)
    results, errors = [], []
    worker.finished.connect(results.append)
    worker.error.connect(errors.append)
    worker.run()

    assert errors == [] and len(results) == 1
    result = results[0]
    assert result["counts"] == {"success": 1, "partial": 1, "failed": 0,
                                "cancelled": 0, "not_started": 0}
    assert [task["status"] for task in result["tasks"]] == ["partial", "success"]
    assert len(result["tasks"][0]["artifacts"]) == 1
    assert len(result["tasks"][1]["artifacts"]) == 2
    assert all(Path(path).is_file() for task in result["tasks"] for path in task["artifacts"])
    assert len(result["tasks"][0]["failures"]) == 1
    _assert_outcome(diagnostics, "batch", "failed")
    folders_logged = _events(diagnostics, "batch_folder")
    assert [record.event for record in folders_logged] == [
        "batch_folder.started", "batch_folder.failed",
        "batch_folder.started", "batch_folder.completed"]
    assert folders_logged[1].exc_info[0] is PartialWriteError
    stacks = _events(diagnostics, "batch_stack")
    assert [record.event for record in stacks] == [
        "batch_stack.started", "batch_stack.failed", "batch_stack.started", "batch_stack.completed"]
    assert _details(stacks[0])["frame_count"] == 2
    assert _details(stacks[0])["first_dtype"] == "uint16"
    assert _details(stacks[0])["parent_operation_id"] == folders_logged[0].operation_id
    writes = [record for record in _events(diagnostics, "batch_image_write")
              if record.event.endswith(".failed")]
    assert len(writes) == 1
    if raise_on_write:
        assert writes[0].exc_info[0] is OSError
        assert str(writes[0].exc_info[1]) == "output disk failed"
    else:
        assert writes[0].exc_info is None


def test_invalid_frame_still_reaches_processing_error_signal(diagnostics):
    worker = RenderWorker([None], algorithm="openfocus_a", alignment="none",
                          use_gpu=False, tile_enabled=False, thread_count=1)
    results, errors = _capture_render(worker)
    worker.run()

    assert results == [] and len(errors) == 1
    _, failed = _assert_outcome(diagnostics, "render", "failed")
    assert failed.exc_info[0] is AttributeError
    assert errors[0] == str(failed.exc_info[1])
    assert failed.exc_info[2] is not None


def test_successful_video_load_after_corrupt_image_is_not_a_failure(tmp_path, diagnostics):
    import cv2
    from core.image_loader import ImageStackLoader

    good = tmp_path / "good.png"
    broken = tmp_path / "broken.png"
    assert imwrite_auto(str(good), np.full((48, 64, 3), 30, np.uint8), exif=b"")
    broken.write_bytes(b"invalid image")
    loader = ImageStackLoader()
    ok, _, _, _ = loader.load_from_filepaths([str(good), str(broken)], thread_count=1)
    assert ok and loader.failed_count == 1
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    if not writer.isOpened():
        writer.release()
        pytest.skip("MJPG encoder unavailable")
    try:
        for value in (60, 100):
            writer.write(np.full((48, 64, 3), value, np.uint8))
    finally:
        writer.release()
    worker = StackLoadWorker(loader, video=str(video))
    results = []
    worker.finished_load.connect(lambda *args: results.append(args))
    worker.run()

    assert len(results) == 1 and results[0][0] is True
    assert len(results[0][2]) == 2
    assert [round(float(image.mean())) for image in results[0][2]] == [60, 100]
    _assert_outcome(diagnostics, "stack_load", "completed")
