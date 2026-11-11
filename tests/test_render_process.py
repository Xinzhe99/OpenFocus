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


def _render_with_prepared_cache(connection, cancel_event, request, budget, directory, language, failure):
    """Run the real child pipeline with a mapped cache produced after fusion."""
    from core import frame_storage, render_process, workers
    from core.multi_focus_fusion import MultiFocusFusion
    from core.resource_policy import ResourceLimitError
    from utils import output_sharpen

    fuse = MultiFocusFusion.fuse
    run = render_process._ChildRenderWorker.run

    def fuse_with_cache(self, *args, **kwargs):
        result = fuse(self, *args, **kwargs)
        self.prepared_cache = [frame_storage.store_frame(
            np.full((8, 9), 231, np.uint16), prefer_ram=False)]
        if failure == 'store':
            def fail_store(*args, **kwargs):
                raise OSError('injected output store failure')
            workers.store_frame = fail_store
        return result

    def run_with_diagnostics(self):
        run(self)
        self.alignment_report = [{'probe': 'alignment diagnostic'}]
        self.alignment_report_provenance = {'probe': 'child'}

    def fail_sharpen(*args, **kwargs):
        raise ResourceLimitError('injected sharpen memory failure')

    MultiFocusFusion.fuse = fuse_with_cache
    render_process._ChildRenderWorker.run = run_with_diagnostics
    if failure == 'sharpen':
        output_sharpen.sharpen_output = fail_sharpen
    render_process._render_child(connection, cancel_event, request, budget, directory, language)


def _sharpen_failure_child(*args):
    _render_with_prepared_cache(*args, 'sharpen')


def _store_failure_child(*args):
    _render_with_prepared_cache(*args, 'store')


def _successful_cache_child(*args):
    _render_with_prepared_cache(*args, None)


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
def scratch(tmp_path, monkeypatch, isolated_render_workers):
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


def test_isolated_render_reports_preparation_after_admission(qapp, monkeypatch):
    from core.render_process import RenderProcessWorker, _Relay

    worker = RenderProcessWorker([], algorithm='openfocus_a')
    stages = []
    worker.stage_signal.connect(stages.append)

    def run_process():
        assert stages == ['waiting', 'prepare']
        return None

    monkeypatch.setattr(worker, '_run_process', run_process)
    worker.run()
    assert stages == ['waiting', 'prepare']

    messages = []
    relay = _Relay(None, None)
    monkeypatch.setattr(relay, 'send', messages.append)
    relay.stage('waiting')
    relay.stage('prepare')
    assert messages == [('stage', 'prepare')]


def test_relay_keeps_stage_order_and_first_progress_per_stage(monkeypatch):
    from core import render_process

    messages = []
    relay = render_process._Relay(None, None)
    monkeypatch.setattr(relay, 'send', messages.append)
    monkeypatch.setattr(render_process.time, 'monotonic', lambda: 10.0)
    relay.stage('prepare')
    relay.progress(1, 4)
    relay.progress(2, 4)
    relay.stage('align')
    relay.stage('fuse')
    relay.progress(1, 3)
    relay.progress(3, 3)
    relay.stage('finish')
    assert messages == [('stage', 'prepare'), ('progress', 1, 4),
                        ('stage', 'align'), ('stage', 'fuse'),
                        ('progress', 1, 3), ('progress', 3, 3),
                        ('stage', 'finish')]


@pytest.mark.parametrize('target, message', [
    (_sharpen_failure_child, 'injected sharpen memory failure'),
    (_store_failure_child, 'injected output store failure'),
])
def test_failed_render_preserves_parent_cache_and_cleans_child_scratch(
        qapp, tmp_path, scratch, monkeypatch, target, message):
    from core import frame_storage, render_process

    parent_cache = [frame_storage.store_frame(
        np.full((8, 9), 137, np.uint16), prefer_ram=False)]
    parent_path = Path(frame_storage.describe_frame(parent_cache[0])['path'])
    opened, cleaned, directories, warnings = [], set(), set(), []
    open_frame = frame_storage.open_frame
    cleanup = render_process._cleanup_files
    warning = render_process.logger.warning

    def record_open(description, **kwargs):
        opened.append(Path(description['path']))
        return open_frame(description, **kwargs)

    def record_cleanup(paths, folders):
        cleaned.update(map(Path, paths))
        directories.update(map(Path, folders))
        return cleanup(paths, folders)

    def record_warning(message, *args, **kwargs):
        warnings.append(message)
        return warning(message, *args, **kwargs)

    monkeypatch.setattr(frame_storage, 'open_frame', record_open)
    monkeypatch.setattr(render_process, '_cleanup_files', record_cleanup)
    monkeypatch.setattr(render_process.logger, 'warning', record_warning)
    frames = _sources(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(render_process, '_render_child', target)
        worker = render_process.RenderProcessWorker(
            frames, algorithm='guided_filter', thread_count=1, use_gpu=False,
            tile_enabled=False, exposure_match=False, prepared_cache=parent_cache,
            retain_preparation=True, sharpen_output=(target is _sharpen_failure_child),
            collect_alignment_report=True, profile_performance=True)
        results, errors = _start(worker)
        _wait(worker, qapp)

    assert not results and len(errors) == 1 and message in errors[0]
    assert worker.prepared_cache is parent_cache
    assert worker.alignment_report == [{'probe': 'alignment diagnostic'}]
    assert worker.alignment_report_provenance == {'probe': 'child'}
    assert worker.performance_report['outcome'] == 'failed'
    assert cleaned and not (cleaned & set(opened))
    assert all(not path.exists() for path in cleaned)
    assert all(not path.exists() for path in directories)
    assert not any('could not remove render scratch file' in text for text in warnings)
    assert parent_path.exists()
    np.testing.assert_array_equal(parent_cache[0], np.full((8, 9), 137, np.uint16))
    parent_cache.clear()
    gc.collect()
    assert not parent_path.exists()

    retry = render_process.RenderProcessWorker(
        frames, algorithm='guided_filter', thread_count=1, use_gpu=False,
        tile_enabled=False, exposure_match=False)
    retried, retry_errors = _start(retry)
    _wait(retry, qapp)
    assert not retry_errors and len(retried) == 1


def test_successful_render_adopts_prepared_cache(qapp, tmp_path, scratch, monkeypatch):
    from core import frame_storage, render_process

    monkeypatch.setattr(render_process, '_render_child', _successful_cache_child)
    worker = render_process.RenderProcessWorker(
        _sources(tmp_path), algorithm='guided_filter', thread_count=1, use_gpu=False,
        tile_enabled=False, exposure_match=False, retain_preparation=True)
    results, errors = _start(worker)
    _wait(worker, qapp)
    assert not errors and len(results) == 1
    cache = worker.prepared_cache
    assert len(cache) == 1
    path = Path(frame_storage.describe_frame(cache[0])['path'])
    assert path.exists()
    np.testing.assert_array_equal(cache[0], np.full((8, 9), 231, np.uint16))
    worker.prepared_cache = None
    cache.clear()
    gc.collect()
    assert not path.exists()


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
    assert list((tmp_path / 'logs').glob('openfocus-*.log'))
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


def test_fusion_preparation_progress_transitions_to_tile_progress():
    from types import SimpleNamespace
    from core.workers import RenderWorker

    events = []
    worker = SimpleNamespace(
        stage_signal=SimpleNamespace(emit=lambda stage: events.append(('stage', stage))),
        progress_signal=SimpleNamespace(emit=lambda done, total: events.append(('progress', done, total))))
    progress = RenderWorker._fusion_progress(worker)
    progress.preparation(1, 92)
    progress.preparation(92, 92)
    progress(0, 228)
    progress(1, 228)
    assert events == [
        ('stage', 'prepare'), ('progress', 1, 92), ('progress', 92, 92),
        ('stage', 'fuse'), ('progress', 0, 228), ('progress', 1, 228),
    ]


@pytest.mark.parametrize('change', ['selection', 'transform', 'close'])
def test_prepared_cache_invalidation_preserves_inflight_reader(qapp, scratch, change):
    from types import SimpleNamespace
    from controllers.render_manager import RenderManager
    from controllers.source_manager import SourceManager
    from controllers.transform_manager import TransformManager
    from core import frame_storage

    class Signal:
        def connect(self, slot):
            self.slot = slot

    class Control:
        def __getattr__(self, name):
            return lambda *args: None

    window = SimpleNamespace(destroyed=Signal(), add_label_action=Control(),
        resize_action=Control(), result_slider=Control(), result_control_bar=Control(),
        lbl_result_info=Control(), raw_images=[])
    manager = RenderManager(window)
    window.render_manager = manager
    window.transform_manager = TransformManager(window)
    frame = frame_storage.store_frame(np.ones((8, 9), np.uint8), prefer_ram=False)
    path = Path(frame_storage.describe_frame(frame)['path'])
    manager._prepared_cache = [frame]
    manager._prepared_cache_key = ('old',)
    manager._prepared_cache_source = window.raw_images
    reader = [frame]
    del frame

    if change == 'selection':
        source = SourceManager(window)
        source._refresh_source_count = lambda: None
        source.decorate_source_items = lambda: None
        source.processing_selection_changed()
    elif change == 'transform':
        window.transform_manager.invalidate_processing_results()
    else:
        window.destroyed.slot()
    assert manager._prepared_cache is None
    assert manager._prepared_cache_key is None
    assert path.exists()
    np.testing.assert_array_equal(reader[0], np.ones((8, 9), np.uint8))
    reader.clear()
    gc.collect()
    assert not path.exists()
