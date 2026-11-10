"""Diagnostic behavior: isolation, retained evidence and failure boundaries."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import zipfile

import pytest

from utils import logging_setup as logs


@pytest.fixture
def log_folder(monkeypatch, tmp_path):
    logs.shutdown_logging()
    monkeypatch.setenv("OPENFOCUS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("OPENFOCUS_LOG_LEVEL", raising=False)
    logs.setup_logging()
    yield logs.logs_dir()
    logs.shutdown_logging()


def test_frame_details_records_each_distinct_geometry_without_pixels():
    class Frame:
        def __init__(self, shape, dtype, size):
            self.shape, self.dtype, self.nbytes = shape, dtype, size

        def __array__(self):
            raise AssertionError('diagnostics must not read image pixels')

    first = Frame((40, 60, 3), 'uint8', 7200)
    second = Frame((20, 30), 'uint16', 1200)
    result = logs.frame_details([first, second, first])
    assert result['frame_count'] == 3
    assert result['total_bytes'] == 15600
    assert result['frame_shapes'] == [
        {'shape': (40, 60, 3), 'dtype': 'uint8', 'count': 2},
        {'shape': (20, 30), 'dtype': 'uint16', 'count': 1},
    ]


def test_repeated_setup_preserves_records_and_debug_selection(log_folder):
    logs.log_session_start()
    logs.setup_logging()
    logs.log_session_start()
    logs.log_action("unicode_action", path="试验/a b=c\nsecond line")
    logger = logs.get_logger(__name__)
    logger.debug("hidden debug entry")
    logs.setup_logging(level="DEBUG")
    logger.debug("visible debug entry")
    text = logs.session_log_excerpt()
    assert text.count("event=session.started") == 1
    assert text.count("event=unicode_action") == 1
    assert "hidden debug entry" not in text
    assert "visible debug entry" in text
    action = next(line for line in text.splitlines() if "event=unicode_action" in line)
    assert json.loads(action.split("event=unicode_action ", 1)[1])["path"] == "试验/a b=c\nsecond line"


def test_concurrent_and_nested_tasks_keep_their_own_context(log_folder):
    barrier = threading.Barrier(2)
    logger = logs.get_logger(__name__)

    def task(name):
        with logs.log_operation(name):
            barrier.wait(timeout=5)
            logger.info("outer %s before", name)
            with logs.log_operation(f"{name}_child"):
                logger.info("child %s", name)
            logger.info("outer %s after", name)
        logger.info("outside %s", name)

    threads = [threading.Thread(target=task, args=(name,)) for name in ("left", "right")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    text = logs.session_log_excerpt()

    def operation_for(fragment):
        line = next(line for line in text.splitlines() if fragment in line)
        return re.search(r"operation=(\S+)", line).group(1)

    left, right = (operation_for(f"outer {name} before") for name in ("left", "right"))
    assert left != right
    for name, parent in (("left", left), ("right", right)):
        assert operation_for(f"outer {name} after") == parent
        assert operation_for(f"child {name}") not in (parent, "-")
        assert operation_for(f"outside {name}") == "-"
        child_start = next(line for line in text.splitlines() if f"event={name}_child.started" in line)
        assert json.loads(child_start.split(".started ", 1)[1])["parent_operation_id"] == parent


def test_caught_failure_retains_traceback_and_cancel_is_not_an_error(log_folder):
    def backend_boundary():
        raise RuntimeError("focus backend failure")

    with logs.log_operation("failed_render") as operation:
        try:
            backend_boundary()
        except RuntimeError:
            operation.fail()
    with logs.log_operation("cancelled_render") as operation:
        operation.cancel()
    with logs.log_operation("rejected_task") as operation:
        operation.fail()  # A reported false result has no fabricated traceback.
    text = logs.session_log_excerpt()
    assert "event=failed_render.failed" in text
    assert "backend_boundary" in text
    assert "RuntimeError: focus backend failure" in text
    assert "event=failed_render.completed" not in text
    cancelled = next(line for line in text.splitlines() if "event=cancelled_render.cancelled" in line)
    assert "INFO" in cancelled
    assert "event=cancelled_render.failed" not in text
    assert "event=rejected_task.failed" in text
    assert "NoneType: None" not in text


def test_escaped_exception_propagates_and_explicit_cancellation_survives(log_folder):
    with pytest.raises(ValueError, match="invalid backend"):
        with logs.log_operation("invalid"):
            raise ValueError("invalid backend")
    with pytest.raises(InterruptedError):
        with logs.log_operation("stopped") as operation:
            operation.cancel()
            raise InterruptedError("cooperative stop")
    text = logs.session_log_excerpt()
    assert "event=invalid.failed" in text
    assert "ValueError: invalid backend" in text
    assert "event=stopped.cancelled" in text
    assert "event=stopped.failed" not in text


def test_unwritable_log_location_does_not_replace_processing_exceptions(log_folder, tmp_path):
    blocked = tmp_path / "file_instead_of_directory"
    blocked.write_text("occupied", encoding="utf-8")
    logs.setup_logging(log_dir=blocked)
    with pytest.raises(ValueError, match="original failure"):
        with logs.log_operation("backend"):
            raise ValueError("original failure")
    assert logs.session_log_excerpt() == "(log unavailable)"
    assert blocked.read_text(encoding="utf-8") == "occupied"


def test_excerpt_reads_this_sessions_retained_rotations_in_order(log_folder, monkeypatch):
    logs.shutdown_logging()
    monkeypatch.setattr(logs, "MAX_BYTES", 1200)
    logs.setup_logging()
    for index in range(20):
        logs.log_action("rotation", marker=f"marker-{index:02d}|", padding="x" * 700)
    other = log_folder / "openfocus-20000101T000000_000000-1.log"
    other.write_text("foreign session must not appear", encoding="utf-8")
    text = logs.session_log_excerpt(20_000)
    assert text.index("marker-16|") < text.index("marker-17|") < text.index("marker-18|") < text.index("marker-19|")
    assert "marker-00|" not in text
    assert "foreign session" not in text
    short = logs.session_log_excerpt(200)
    assert len(short.encode("utf-8")) < 400
    assert logs.session_log_excerpt(0) == ""


def test_retention_keeps_active_and_unrelated_files(log_folder):
    stale = log_folder / "openfocus-20000101T000000_000000-2147483647.log"
    active = log_folder / f"openfocus-20000101T000000_000000-{os.getpid()}.log"
    unrelated = log_folder / "notes.txt"
    for path in (stale, active, unrelated):
        path.write_text(path.name, encoding="utf-8")
        old = time.time() - (logs.RETENTION_DAYS + 1) * 86400
        os.utime(path, (old, old))
    logs.shutdown_logging()
    logs.setup_logging()
    assert not stale.exists()
    assert active.read_text(encoding="utf-8") == active.name
    assert unrelated.read_text(encoding="utf-8") == unrelated.name


def test_parallel_processes_rotate_without_mixing_sessions(log_folder):
    code = """
import sys
from utils.logging_setup import setup_logging, log_action, session_log_path, session_log_excerpt
setup_logging()
for i in range(30):
    log_action('parallel', owner=sys.argv[1], padding='x' * 70000)
log_action('owner_finished', owner=sys.argv[1])
print(session_log_path())
"""
    processes = [subprocess.Popen([sys.executable, "-c", code, own, other],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, encoding="utf-8")
                 for own, other in (("alpha", "beta"), ("beta", "alpha"))]
    outputs = [process.communicate(timeout=30) for process in processes]
    paths = [Path(stdout.splitlines()[0]) for stdout, _ in outputs]
    assert paths[0] != paths[1]
    for process, (stdout, stderr), path, owner in zip(processes, outputs, paths, ("alpha", "beta")):
        assert process.returncode == 0, stderr
        assert not stderr
        current = path.read_text(encoding="utf-8")
        assert f'"owner": "{owner}"' in current
        other = "beta" if owner == "alpha" else "alpha"
        assert f'"owner": "{other}"' not in current
        assert "Logging error" not in stdout


def test_feedback_includes_current_crash_and_tolerates_vanished_history(log_folder, monkeypatch, tmp_path):
    from utils import diagnostics
    crash = log_folder / "openfocus-20000101T000000_000000-1.crash.txt"
    crash.write_text("native traceback marker", encoding="utf-8")
    monkeypatch.setattr(diagnostics, "_crash_path", crash)
    monkeypatch.setattr(logs, "all_log_files", lambda: [log_folder / "already_removed.log"])
    logs.log_action("feedback_marker", value="report evidence")
    report = logs.export_feedback_package(tmp_path / "report.zip")
    with zipfile.ZipFile(report) as archive:
        assert b"report evidence" in archive.read("session_log.txt")
        assert archive.read(f"logs/{crash.name}") == b"native traceback marker"
        facts = archive.read("system.txt").decode("utf-8")
        assert logs.session_id() in facts
        assert "python:" in facts
        assert "already_removed.log" not in archive.namelist()


@pytest.mark.parametrize("kind,expected", [
    (None, "not checked"),
    ("CPU", "False"),
    ("DIRECTML", "True"),
])
def test_feedback_export_reports_neural_gpu_status(log_folder, monkeypatch, tmp_path,
                                                   kind, expected):
    from core import multi_focus_fusion
    from fusion_methods import neural_runtime

    monkeypatch.setattr(neural_runtime, "_DEVICE", kind)
    monkeypatch.setattr(multi_focus_fusion, "_NEURAL_DEVICE_KIND", None)
    report = logs.export_feedback_package(tmp_path / "report.zip")
    with zipfile.ZipFile(report) as archive:
        facts = archive.read("system.txt").decode("utf-8")
    assert f"gpu (neural acceleration): {expected}" in facts


@pytest.mark.parametrize("code,outcome", [(0, "completed"), (2, "failed")])
def test_system_exit_keeps_code_without_inventing_a_cancellation(log_folder, code, outcome):
    with pytest.raises(SystemExit) as caught:
        with logs.log_operation("command"):
            raise SystemExit(code)
    assert caught.value.code == code
    text = logs.session_log_excerpt()
    assert f"event=command.{outcome}" in text
    assert "event=command.cancelled" not in text
    assert "Traceback" not in text


def test_feedback_bounds_log_tails_and_explains_omissions(log_folder, monkeypatch, tmp_path):
    from utils import diagnostics
    monkeypatch.setattr(diagnostics, '_crash_path', None)
    monkeypatch.setattr(logs, 'MAX_BYTES', 100)
    monkeypatch.setattr(logs, 'FEEDBACK_MAX_LOG_BYTES', 150)
    monkeypatch.setattr(logs, 'FEEDBACK_MAX_LOG_FILES', 2)
    paths = [log_folder / f'history-{index}.log' for index in range(3)]
    for index, path in enumerate(paths):
        path.write_bytes(b'old:' + bytes([65 + index]) * 200)
    monkeypatch.setattr(logs, 'all_log_files', lambda: paths)
    report = logs.export_feedback_package(tmp_path / 'bounded.zip')
    with zipfile.ZipFile(report) as archive:
        assert archive.read('logs/history-0.log') == b'A' * 100
        assert archive.read('logs/history-1.log') == b'B' * 50
        assert 'logs/history-2.log' not in archive.namelist()
        index = archive.read('diagnostic_index.txt').decode('utf-8')
        assert 'older bytes trimmed' in index
        assert 'Omitted 1 older log files' in index
        assert 'operation=<id>' in index
        assert '.cancelled' in index


def test_feedback_file_limit_also_bounds_unavailable_index(log_folder, monkeypatch, tmp_path):
    from utils import diagnostics
    monkeypatch.setattr(diagnostics, '_crash_path', None)
    monkeypatch.setattr(logs, 'FEEDBACK_MAX_LOG_FILES', 2)
    paths = [log_folder / f'missing-{index}.log' for index in range(10)]
    monkeypatch.setattr(logs, 'all_log_files', lambda: paths)
    report = logs.export_feedback_package(tmp_path / 'missing.zip')
    with zipfile.ZipFile(report) as archive:
        index = archive.read('diagnostic_index.txt').decode('utf-8')
        assert index.count('Unavailable:') == 2
        assert 'Omitted 8 older log files' in index


@pytest.mark.parametrize('view', ['contiguous', 'columns', 'reversed', 'transposed'])
def test_image_result_fingerprint_matches_c_order_pixels(view):
    import hashlib
    import numpy as np

    image = np.arange(7 * 9 * 3, dtype=np.uint16).reshape(7, 9, 3)
    if view == 'columns':
        image = image[:, ::2]
    elif view == 'reversed':
        image = image[::-1]
    elif view == 'transposed':
        image = image.transpose(1, 0, 2)
    details = logs.image_result_details(image)
    assert details == {'shape': image.shape, 'dtype': 'uint16',
                       'sha256': hashlib.sha256(image.tobytes(order='C')).hexdigest()}
