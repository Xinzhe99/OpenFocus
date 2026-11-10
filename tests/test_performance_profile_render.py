"""Render-level profiling: reports from in-process and spawned renders."""
import json
import os
import time

import numpy as np
import pytest

from conftest import make_two_focus_stack
from core.multi_focus_fusion import is_ai_fusion_available, _ai_effective_tiling
from utils import performance_profile as pp

pytestmark = pytest.mark.skipif(not is_ai_fusion_available(),
                                reason="ONNX Runtime not installed")


def _stack(count=3, size=(150, 110)):
    first, second = make_two_focus_stack(size)
    frames = [first, second]
    while len(frames) < count:
        frames.append(((first.astype(np.uint16) + second) // 2).astype(np.uint8))
    return frames


def _worker(frames, **options):
    from core.workers import RenderWorker
    return RenderWorker(frames, algorithm='stackmffv3', use_gpu=False, thread_count=1,
                        exposure_match=False, tile_enabled=True, tile_block_size=64,
                        tile_overlap=16, tile_threshold=32, **options)


def test_render_worker_keeps_report_and_logs_compact_event(monkeypatch):
    from core import workers
    events = []
    monkeypatch.setattr(workers, 'log_action', lambda event, **details: events.append((event, details)))
    frames = _stack()
    reference, profiled = _worker(frames), _worker(frames, profile_performance=True)
    results = {}
    for name, worker in (('off', reference), ('on', profiled)):
        worker.finished_signal.connect(lambda *value, name=name: results.__setitem__(name, value))
        worker.run()
    np.testing.assert_array_equal(results['on'][1], results['off'][1])
    assert reference.performance_report is None
    report = profiled.performance_report
    assert report['outcome'] == 'completed'
    assert report['operation_id']
    assert f"Finished: {report['finished_at']}" in pp.summary(report)
    assert report['threads']['requested'] == 1
    assert report['render']['algorithm'] == 'stackmffv3'
    assert report['stages']['render']['peak_rss_bytes'] > 0
    assert report['render']['output_shape'] == list(results['on'][1].shape)
    assert {'commit', 'versions', 'physical_cores', 'automatic_threads'} <= set(report['runtime'])
    assert any('saving/encoding' in note for note in report['unmeasured'])
    assert 'render/fusion/fuse/tiled/tile' in report['stages']
    (event, details), = [item for item in events if item[0] == 'render.performance']
    assert 'coords' not in details['tiling']
    assert len(details['slowest_tiles']) <= pp.SLOWEST_TILES
    json.dumps(details)


def test_cancelled_render_returns_partial_report():
    worker = _worker(_stack(), profile_performance=True)
    errors = []
    worker.error_signal.connect(errors.append)
    worker.progress_signal.connect(
        lambda done, total: setattr(worker, 'is_cancelled', done >= 1))
    worker.run()
    assert errors == ['CANCELLED: render cancelled']
    report = worker.performance_report
    assert report['outcome'] == 'cancelled'
    assert report['stages']['render/fusion/fuse/tiled/tile']['count'] >= 1


def test_failed_render_keeps_original_error(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError('original failure')
    worker = _worker(_stack(), profile_performance=True)
    monkeypatch.setattr(worker, '_run_fusion', broken)
    errors = []
    worker.error_signal.connect(errors.append)
    worker.run()
    assert errors == ['original failure']
    assert worker.performance_report['outcome'] == 'failed'


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    from core import frame_storage
    directory = tmp_path / 'scratch'
    directory.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', str(directory))
    return directory


def _wait(worker, qapp):
    from PyQt6.QtTest import QTest
    deadline = time.monotonic() + 60
    while worker.isRunning() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert not worker.isRunning()
    worker.wait(1000)
    qapp.processEvents()


def test_spawned_render_relays_report_to_the_gui_process(qapp, scratch):
    pytest.importorskip('PyQt6')
    from core.render_process import RenderProcessWorker
    worker = RenderProcessWorker(
        _stack(), algorithm='stackmffv3', use_gpu=False, thread_count=1,
        exposure_match=False, tile_enabled=True, tile_block_size=64, tile_overlap=16,
        tile_threshold=32, profile_performance=True)
    results, errors = [], []
    worker.finished_signal.connect(lambda *value: results.append(value))
    worker.error_signal.connect(errors.append)
    worker.start()
    _wait(worker, qapp)
    assert not errors and results
    report = worker.performance_report
    assert report['outcome'] == 'completed'
    assert report['threads']['requested'] == 1
    assert report['tiling']['count'] == len(_ai_effective_tiling(3, 64, 150, 110, 16, 1)[2])


def _crash_child(*args):
    os._exit(23)


def test_crashed_child_reports_the_existing_error_and_no_report(qapp, scratch, monkeypatch):
    from core import render_process
    monkeypatch.setattr(render_process, '_render_child', _crash_child)
    worker = render_process.RenderProcessWorker(
        _stack(), algorithm='stackmffv3', use_gpu=False, thread_count=1,
        profile_performance=True)
    errors = []
    worker.error_signal.connect(errors.append)
    worker.start()
    _wait(worker, qapp)
    assert len(errors) == 1 and 'exit code 23' in errors[0]
    assert worker.performance_report is None
