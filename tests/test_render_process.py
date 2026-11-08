"""Consumer-visible guarantees of crash-isolated GUI rendering."""
import gc
import os
from pathlib import Path
import time

import numpy as np
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtTest import QTest


def _crash_child(connection, cancel_event, request, budget, directory, language):
    from core import frame_storage as storage
    from core.render_process import _Relay
    storage.restore_storage_directory(directory)
    relay = _Relay(connection, cancel_event)
    with storage.observe_temp_files(relay.temporary_file), storage.scratch_directory():
        frame = storage.create_frame((8, 9), np.uint16)
        frame[0] = 1001
        os._exit(23)


def _abort_child(connection, cancel_event, request, budget, directory, language):
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(3)
    os.abort()


def _wait(worker, qapp):
    deadline = time.monotonic() + 35
    while worker.isRunning() and time.monotonic() < deadline:
        QTest.qWait(10)
    if worker.isRunning():
        worker.cancel()
        worker.wait(6000)
        pytest.fail('Isolated render did not finish within 35 seconds')
    worker.wait(1000)
    qapp.processEvents()


def _start(worker):
    results, errors = [], []
    worker.finished_signal.connect(lambda *value: results.append(value))
    worker.error_signal.connect(errors.append)
    worker.start()
    return results, errors


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    from core import frame_storage as storage
    directory = tmp_path / 'scratch'
    directory.mkdir()
    monkeypatch.setattr(storage, '_storage_directory', str(directory))
    yield directory
    gc.collect()


def _sources(tmp_path):
    from core.source_frames import scan_source
    from tests.conftest import make_two_focus_stack
    from utils.image_utils import imwrite_auto
    frames = []
    for index, image in enumerate(make_two_focus_stack((48, 64))):
        image = image.astype(np.uint16) * 211 + 1003
        path = tmp_path / f'source-{index}.tif'
        assert imwrite_auto(str(path), image)
        frames.append(scan_source(str(path))[0])
    return frames


@pytest.mark.parametrize('alignment', ['none', 'openfocus'])
def test_spawned_render_preserves_uint16_and_lazy_alignment(qapp, tmp_path, scratch, alignment):
    from core.render_process import RenderProcessWorker
    from core.source_frames import ProcessedFrame, SourceFrame, resolve_frame
    from core.workers import RenderWorker
    frames = _sources(tmp_path)
    options = dict(algorithm='openfocus_a', alignment=alignment, tile_enabled=False,
                   thread_count=1, use_gpu=False, exposure_match=False,
                   collect_alignment_report=True)
    baseline, baseline_errors = [], []
    reference = RenderWorker(frames, **options)
    reference.finished_signal.connect(lambda *value: baseline.append(value))
    reference.error_signal.connect(baseline_errors.append)
    reference.run()
    assert not baseline_errors
    worker = RenderProcessWorker(frames, **options)
    results, errors = _start(worker)
    _wait(worker, qapp)
    assert not errors
    assert worker.child_pid is not None and worker.child_pid != os.getpid()
    processed, fused, registered, *_ = results[0]
    assert fused.dtype == np.uint16
    np.testing.assert_array_equal(fused, baseline[0][1])
    assert np.any(fused % 257 != 0)
    assert not fused.flags.writeable
    expected_type = SourceFrame if alignment == 'none' else ProcessedFrame
    assert all(isinstance(frame, expected_type) for frame in processed)
    for frame, expected in zip(processed, baseline[0][0]):
        np.testing.assert_array_equal(resolve_frame(frame), resolve_frame(expected))
    assert registered == (alignment != 'none')
    assert worker.alignment_report == reference.alignment_report
    # The map must survive the child's exit, then disappear with its last GUI view.
    root = fused
    while getattr(root, 'base', None) is not None:
        if isinstance(root, np.memmap):
            path = Path(root.filename)
        root = root.base
    assert path.exists()
    view = fused[1:-1, 1:-1]
    del fused
    results.clear()
    assert path.exists()
    del view
    gc.collect()
    assert not path.exists()


@pytest.mark.parametrize('target', [_crash_child, _abort_child])
def test_native_child_failure_keeps_application_alive_and_allows_retry(
        qapp, tmp_path, scratch, monkeypatch, target):
    from core import render_process
    frames = _sources(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(render_process, '_render_child', target)
        worker = render_process.RenderProcessWorker(
            frames, algorithm='openfocus_a', thread_count=1, use_gpu=False)
        results, errors = _start(worker)
        _wait(worker, qapp)
    assert not results
    assert len(errors) == 1
    assert qapp is not None and not qapp.closingDown()
    assert not list(scratch.glob('openfocus-frame-*'))
    retry = render_process.RenderProcessWorker(
        frames, algorithm='openfocus_a', tile_enabled=False, thread_count=1,
        use_gpu=False, exposure_match=False)
    results, errors = _start(retry)
    _wait(retry, qapp)
    assert not errors
    assert results[0][1].dtype == np.uint16


def test_cancel_during_child_startup_publishes_no_result(qapp, tmp_path, scratch):
    from PyQt6.QtCore import QTimer
    from core.render_process import RenderProcessWorker
    worker = RenderProcessWorker(_sources(tmp_path), algorithm='openfocus_a',
                                 thread_count=1, use_gpu=False)
    results, errors = _start(worker)
    timer = QTimer()
    def cancel_started_child():
        if worker.child_pid is not None:
            # RenderManager also sets the flag directly when Cancel is clicked.
            worker.is_cancelled = True
            timer.stop()
    timer.timeout.connect(cancel_started_child)
    timer.start(10)
    _wait(worker, qapp)
    timer.stop()
    assert not results
    assert len(errors) == 1 and errors[0].startswith('CANCELLED:')
    assert not list(scratch.glob('openfocus-frame-*'))


def test_changed_source_is_reported_without_publishing_stale_pixels(qapp, tmp_path, scratch):
    from core.render_process import RenderProcessWorker
    from utils.image_utils import imwrite_auto
    frames = _sources(tmp_path)
    path = frames[0].path
    assert imwrite_auto(path, np.full((48, 64, 3), 1777, dtype=np.uint16))
    worker = RenderProcessWorker(frames, algorithm='openfocus_a', tile_enabled=False,
                                 thread_count=1, use_gpu=False)
    results, errors = _start(worker)
    _wait(worker, qapp)
    assert not results
    assert len(errors) == 1 and path in errors[0]


def test_heap_input_snapshot_survives_parent_staging_release(qapp, scratch):
    from core.render_process import RenderProcessWorker
    image = (np.arange(32 * 40 * 3, dtype=np.uint16).reshape(32, 40, 3) * 7 + 1003)
    source = image[::-1, ::2]
    expected = source.copy()
    worker = RenderProcessWorker([source, source], algorithm='openfocus_a',
        thread_count=1, use_gpu=False, tile_enabled=False, exposure_match=False)
    results, errors = _start(worker)
    _wait(worker, qapp)
    assert not errors
    image.fill(0)
    for frame in results[0][0]:
        np.testing.assert_array_equal(frame, expected)
    np.testing.assert_array_equal(results[0][1], expected)


def test_registered_extension_and_composite_alignment_execute_in_child(qapp, scratch, monkeypatch):
    from core import algorithms
    from core.render_process import RenderProcessWorker
    monkeypatch.setattr(algorithms, '_FUSION', dict(algorithms._FUSION))
    monkeypatch.setattr(algorithms, '_REGISTRATION', dict(algorithms._REGISTRATION))
    parameters = (
        algorithms.ParameterSpec('fraction', 'test_fraction', .25, 0, 1, .05, 2),
        algorithms.ParameterSpec('direction', 'test_direction', 'forward',
                                 choices=('forward', 'reverse')),
    )
    algorithms.register_fusion(algorithms.FusionAlgorithm(
        'isolated_extension', 'test_extension', 'Extension', 'Extension',
        'tests.test_algorithms:Extension.blend', parameters))
    algorithms.register_registration(algorithms.RegistrationAlgorithm(
        'isolated_shift', 'test_shift', 'Shift', 'tests.test_algorithms:Extension.shift',
        (algorithms.ParameterSpec('delta', 'test_delta', 3, -20, 20),)))
    algorithms.register_registration(algorithms.RegistrationAlgorithm(
        'isolated_double', 'test_double', 'Double', 'tests.test_algorithms:Extension.double'))
    algorithms.register_registration(algorithms.RegistrationAlgorithm(
        'isolated_composite', 'test_composite', 'Composite',
        steps=('isolated_shift', 'isolated_double')))
    images = [np.full((16, 20, 3), value, np.uint8) for value in (20, 100)]
    worker = RenderProcessWorker(images, algorithm='isolated_extension',
        alignment='isolated_composite', registration_parameters={'delta': 7},
        fusion_parameters={'fraction': .25, 'direction': 'reverse'},
        thread_count=1, use_gpu=False, tile_enabled=False, exposure_match=False)
    results, errors = _start(worker)
    _wait(worker, qapp)
    assert not errors
    np.testing.assert_array_equal(results[0][1], np.full((16, 20, 3), 174, np.uint8))
