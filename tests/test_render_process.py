"""Consumer-visible guarantees of crash-isolated GUI rendering."""
import gc
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtTest import QTest
from core.resource_policy import ResourceLimitError


def _crash_child(connection, cancel_event, request, budget, directory, language):
    from core import frame_storage as storage
    from core.render_process import _Relay
    storage.restore_storage_directory(directory)
    relay = _Relay(connection, cancel_event)
    relay.stage('fuse')
    with storage.observe_temp_files(relay.temporary_file), storage.scratch_directory():
        frame = storage.create_frame((8, 9), np.uint16)
        frame[0] = 1001
        os._exit(23)


def _abort_child(connection, cancel_event, request, budget, directory, language):
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(3)
    os.abort()


def _silent_child(connection, cancel_event, request, budget, directory, language):
    connection.send(('stage', 'align'))
    connection.close()


def _diagnostic_error_child(connection, cancel_event, request, budget, directory, language):
    connection.send(('error', 'user-visible private detail', None,
                     ('value', 'child_setup_or_transfer', 'unknown')))
    connection.close()


def _runtime_error_child(connection, cancel_event, request, budget, directory, language):
    from core import render_process

    def fail_render(self):
        raise RuntimeError('fixture private detail synthetic/source.tif')

    render_process._ChildRenderWorker._run_admitted_body = fail_render
    render_process._render_child(connection, cancel_event, request, budget, directory, language)


def _diagnostic_cancel_child(connection, cancel_event, request, budget, directory, language):
    connection.send(('error', 'CANCELLED: render cancelled', None, None))
    connection.close()


def _legacy_source_child(connection, cancel_event, request, budget, directory, language):
    from core.frame_transport import FrameDecoder
    try:
        FrameDecoder().decode(request)
    except ValueError as exc:
        connection.send(('error', str(exc), None, ('value', 'child_setup_or_transfer', 'prepare')))
    finally:
        connection.close()


def _changed_source_child(connection, cancel_event, request, budget, directory, language):
    from core import render_process
    render_process._windows_source_run = lambda: True
    render_process._LOADED_SOURCE_REVISION = b'new worker revision'
    render_process._render_child(connection, cancel_event, request, budget, directory, language)



def _render_with_prepared_cache(connection, cancel_event, request, budget, directory, language, failure):
    """Run the real child pipeline with a mapped cache produced after fusion."""
    from core import frame_storage, render_process
    from core.multi_focus_fusion import MultiFocusFusion
    from core.resource_policy import ResourceSnapshot, require_memory
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
            frame_storage.create_transient_frame = fail_store
        return result

    def run_with_diagnostics(self):
        run(self)
        self.alignment_report = [{'probe': 'alignment diagnostic'}]
        self.alignment_report_provenance = {'probe': 'child'}

    def fail_sharpen(image, should_cancel=None, *, strength):
        assert strength == 175
        snapshot = ResourceSnapshot(available_ram=8 * 1024**3, total_ram=16 * 1024**3,
                                    commit_headroom=9 * 1024**2,
                                    process_commit_headroom=5 * 1024**2)
        require_memory(6 * 1024**2, snapshot, user_limit=3 * 1024**2)

    MultiFocusFusion.fuse = fuse_with_cache
    render_process._ChildRenderWorker.run = run_with_diagnostics
    if failure == 'sharpen':
        output_sharpen.sharpen_output = fail_sharpen
    # The memory rejection fixture controls the complete admission snapshot.
    render_process._render_child(connection, cancel_event, request,
                                 None if failure == 'sharpen' else budget, directory, language)


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


def test_spawn_failure_reports_pending_dispatch_and_releases_worker(qapp, scratch, monkeypatch, caplog):
    from threading import Event
    from types import SimpleNamespace
    from core import render_process

    real_context = render_process.multiprocessing.get_context('spawn')
    entered, release = Event(), Event()
    closed = []
    private_detail = 'private/source-video.mp4'

    class BlockedProcess:
        pid = None

        def start(self):
            entered.set()
            release.wait()
            raise OSError(private_detail)

        def close(self):
            closed.append(True)

    context = SimpleNamespace(Event=real_context.Event, Pipe=real_context.Pipe,
                              Process=lambda **kwargs: BlockedProcess())
    monkeypatch.setattr(render_process.multiprocessing, 'get_context', lambda method: context)
    caplog.set_level(logging.INFO, logger='openfocus')
    worker = render_process.RenderProcessWorker([], algorithm=None, need_fusion=False)
    results, errors = _start(worker)
    try:
        assert entered.wait(5)
        dispatch = [r for r in caplog.records
                    if getattr(r, 'event', '').startswith('render_dispatch.')]
        assert any(r.event == 'render_dispatch.spawn_started' for r in dispatch)
        assert not any(r.event == 'render_dispatch.spawn_completed' for r in dispatch)
        assert worker.isRunning()
    finally:
        release.set()
        _wait(worker, qapp)
    assert not results and errors == [private_detail]
    assert closed and not worker.isRunning()
    dispatch = [r for r in caplog.records
                if getattr(r, 'event', '').startswith('render_dispatch.')]
    assert not any(r.event == 'render_dispatch.spawn_completed' for r in dispatch)
    assert all(r.getMessage().endswith(' {}') for r in dispatch)
    assert all(private_detail not in r.getMessage() for r in dispatch)


def test_changed_render_source_reports_restart_before_spawn(qapp, tmp_path, monkeypatch):
    from core import render_process

    source_file = tmp_path / 'source_frames.py'
    source_file.write_text('revision = 1\n', encoding='utf-8')
    monkeypatch.setattr(render_process, '_windows_source_run', lambda: True)
    monkeypatch.setattr(render_process, '_RENDER_SOURCE_FILES', (source_file,))
    monkeypatch.setattr(render_process, '_LOADED_SOURCE_REVISION', render_process._render_source_revision())
    monkeypatch.setattr(render_process.trans, 'current_lang', 'en')
    monkeypatch.setattr(render_process.multiprocessing, 'get_context',
                        lambda method: pytest.fail('Spawn attempted with changed source code'))
    source_file.write_text('revision = 2\n', encoding='utf-8')

    worker = render_process.RenderProcessWorker([], algorithm=None, need_fusion=False)
    errors = []
    worker.error_signal.connect(errors.append)
    worker.run()
    assert errors == [render_process.trans.t('render_source_changed')]
    assert 'Restart OpenFocus' in errors[0]
    assert worker.child_pid is None


@pytest.mark.parametrize('exception, category', [
    (MemoryError('allocation failed'), 'memory'),
    (ResourceLimitError('render limit exceeded'), 'resource_limit'),
])
def test_render_process_classifies_parent_memory_failure(qapp, monkeypatch, exception, category):
    from core.render_process import RenderProcessWorker

    worker = RenderProcessWorker([], algorithm=None, need_fusion=False)
    errors = []
    worker.error_signal.connect(errors.append)

    def fail():
        raise exception

    monkeypatch.setattr(worker, '_run_process', fail)
    worker.run()
    assert errors == [str(exception)]
    assert worker.failure_category == category


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

@pytest.mark.parametrize('exception, category', [
    (lambda: ResourceLimitError('private'), 'resource_limit'),
    (lambda: MemoryError('private'), 'memory'),
    (lambda: OSError('private'), 'os'),
    (lambda: ValueError('private'), 'value'),
    (lambda: RuntimeError('private'), 'runtime'),
    (lambda: TypeError('private'), 'other'),
])
def test_worker_captures_exception_type_without_changing_error(qapp, monkeypatch, exception, category):
    from core.workers import RenderWorker

    worker = RenderWorker([], algorithm=None, need_fusion=False)
    errors = []
    worker.error_signal.connect(errors.append)

    def fail():
        raise exception()

    monkeypatch.setattr(worker, '_run_admitted', fail)
    worker.run()
    assert errors == ['private']
    assert worker._render_error_category == category


def test_worker_cancellation_has_no_error_category(qapp, monkeypatch):
    from core.registration import RegistrationCancelled
    from core.workers import RenderWorker

    worker = RenderWorker([], algorithm=None, need_fusion=False)
    errors = []
    worker.error_signal.connect(errors.append)

    def cancel():
        raise RegistrationCancelled()

    monkeypatch.setattr(worker, '_run_admitted', cancel)
    worker.run()
    assert errors == ['CANCELLED: render cancelled']
    assert worker._render_error_category is None


def test_source_child_rejects_legacy_gui_before_decoding(monkeypatch):
    from threading import Event
    from core import render_process
    from utils import diagnostics, logging_setup

    request = render_process.FrameEncoder().encode({'args': []})
    receiver, sender = render_process.multiprocessing.Pipe(duplex=False)
    monkeypatch.setattr(logging_setup, 'setup_logging', lambda: None)
    monkeypatch.setattr(diagnostics, 'install_exception_hooks', lambda: None)
    monkeypatch.setattr(render_process, '_windows_source_run', lambda: True)
    monkeypatch.setattr(render_process, '_LOADED_SOURCE_REVISION', b'worker revision')
    monkeypatch.setattr(render_process.multiprocessing, 'parent_process', object)
    monkeypatch.setattr(render_process.trans, 'current_lang', 'en')
    monkeypatch.setattr(render_process, 'FrameDecoder',
                        lambda **kwargs: pytest.fail('Legacy descriptors reached the decoder'))

    with receiver, sender:
        render_process._render_child(sender, Event(), request, None, None, 'zh')
        assert receiver.poll()
        kind, message, state, metadata, details = receiver.recv()
        assert (kind, message, state, metadata) == (
            'error', render_process.trans.t('render_source_changed'), None,
            ('runtime', 'child_setup_or_transfer', 'prepare'))
        assert details['error']['kind'] == 'RuntimeError'
        assert details['error']['stack'][0]['function'] == '_render_child'
        assert '重启 OpenFocus' in render_process.trans.t('render_source_changed')


@pytest.mark.parametrize('frozen', [False, True])
def test_child_setup_failure_and_cancellation_packets(monkeypatch, frozen):
    from contextlib import nullcontext
    from threading import Event
    from types import SimpleNamespace
    from core import render_process
    from core.registration import RegistrationCancelled
    from utils import diagnostics, logging_setup

    monkeypatch.setattr(render_process, 'sys', SimpleNamespace(platform='win32', frozen=frozen))
    monkeypatch.setattr(render_process.multiprocessing, 'parent_process', lambda: frozen or None)

    class Connection:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(message)

        def close(self):
            pass

    monkeypatch.setattr(logging_setup, 'setup_logging', lambda: None)
    monkeypatch.setattr(diagnostics, 'install_exception_hooks', lambda: None)
    monkeypatch.setattr(render_process.storage, 'restore_storage_directory', lambda directory: None)
    monkeypatch.setattr(render_process, 'process_budget', lambda budget: nullcontext())
    monkeypatch.setattr(render_process.storage, 'observe_temp_files', lambda callback: nullcontext())

    class Decoder:
        def __init__(self, **kwargs):
            pass

        def decode(self, request):
            raise request

    monkeypatch.setattr(render_process, 'FrameDecoder', Decoder)
    connection = Connection()
    render_process._render_child(connection, Event(), ValueError('private path'), None, None, 'en')
    (kind, message, state, metadata, details), = connection.messages
    assert (kind, message, state, metadata) == (
        'error', 'private path', None, ('value', 'child_setup_or_transfer', 'prepare'))
    assert details['error']['kind'] == 'ValueError'
    assert details['error']['stack'][0]['function'] == '_render_child'
    assert 'private path' not in repr(details)
    connection = Connection()
    render_process._render_child(connection, Event(), RegistrationCancelled(), None, None, 'en')
    assert connection.messages == [('error', 'CANCELLED: render cancelled', None, None)]


@pytest.mark.parametrize('target, expected_event, expected_error', [
    (_diagnostic_error_child, {'error_category': 'value', 'error_origin': 'child_setup_or_transfer',
                               'render_stage': 'unknown'}, 'user-visible private detail'),
    (_diagnostic_cancel_child, None, 'CANCELLED: render cancelled'),
])
def test_child_error_event_and_user_message(qapp, scratch, monkeypatch, target, expected_event, expected_error):
    from core import render_process

    events = []
    monkeypatch.setattr(render_process, '_render_child', target)
    monkeypatch.setattr(render_process, 'log_action', lambda event, **fields:
                        events.append((event, fields)) if event == 'render_child_error' else None)
    worker = render_process.RenderProcessWorker([], algorithm=None, need_fusion=False)
    results, errors = _start(worker)
    _wait(worker, qapp)
    assert not results and errors == [expected_error]
    assert events == ([] if expected_event is None else [('render_child_error', expected_event)])
    assert worker.failure_category == (expected_event['error_category'] if expected_event else None)


def _child_evidence(caplog):
    # Pytest can observe the same record on both the application and root logger.
    records = dict.fromkeys(caplog.records)
    events = [record for record in records if getattr(record, 'event', None) == 'render_child_error']
    failures = [record for record in records if hasattr(record, 'report_child_evidence')]
    event, = events
    failure, = failures
    fields = json.loads(event.getMessage().split(' ', 1)[1])
    assert fields == failure.report_child_evidence
    assert failure.exc_info[0] is RuntimeError
    return fields


def test_worker_error_reports_original_stack_through_ipc(qapp, scratch, monkeypatch, caplog):
    from core import render_process

    caplog.set_level(logging.INFO, logger='openfocus')
    monkeypatch.setattr(logging.getLogger('openfocus'), 'propagate', True)
    monkeypatch.setattr(render_process, '_render_child', _runtime_error_child)
    worker = render_process.RenderProcessWorker([], algorithm=None, need_fusion=False)
    results, errors = _start(worker)
    _wait(worker, qapp)

    assert not results and errors == ['fixture private detail synthetic/source.tif']
    fields = _child_evidence(caplog)
    assert fields['error_category'] == 'runtime'
    assert fields['error_origin'] == 'child_worker'
    assert fields['render_stage'] == 'prepare'
    assert fields['error']['kind'] == 'RuntimeError'
    assert [(frame['file'], frame['function']) for frame in fields['error']['stack']] == [
        ('workers.py', 'run'), ('workers.py', '_run_admitted')]
    assert all(type(line) is int and line > 0 for line in fields['error']['stack_lines'])
    assert 'synthetic/source.tif' not in json.dumps(fields)


def test_exchange_filters_malformed_child_details():
    from core.render_process import _RenderExchange

    exchange = _RenderExchange(None)
    exchange.receive(('error', 'fixture UI message', None, ('resource_limit', 'child_worker', 'finish'), {
        'error': {'kind': 'RuntimeError', 'stack_lines': [4], 'stack': [
            {'file': 'synthetic/workers.py', 'line': 4, 'function': 'run'}]},
        'admission': {'resource_status': 'insufficient', 'required_bytes': 8,
                      'ram_ceiling': True, 'available_ram_bytes': 10**16,
                      'available_commit_bytes': -1, 'message': 'fixture private detail'},
        'exit_code': 12,
    }), None)
    assert exchange.error == 'fixture UI message'
    assert exchange.error_metadata == {
        'error_category': 'resource_limit', 'error_origin': 'child_worker', 'render_stage': 'finish',
        'admission': {'resource_status': 'insufficient', 'required_bytes': 8},
    }


@pytest.mark.parametrize('target', [_legacy_source_child, _changed_source_child])
@pytest.mark.parametrize('live_preview', [False, True])
def test_spawn_revision_mismatch_relays_restart(qapp, scratch, monkeypatch, target, live_preview):
    from core import render_process

    monkeypatch.setattr(render_process, '_windows_source_run', lambda: True)
    monkeypatch.setattr(render_process, '_LOADED_SOURCE_REVISION', render_process._render_source_revision())
    monkeypatch.setattr(render_process, '_render_child', target)
    monkeypatch.setattr(render_process.trans, 'current_lang', 'en')
    worker = render_process.RenderProcessWorker([], algorithm=None, need_fusion=False)
    worker.live_preview_requested = live_preview
    results, errors = _start(worker)
    _wait(worker, qapp)
    assert worker.child_pid is not None
    assert not results and errors == [render_process.trans.t('render_source_changed')]
    assert not list(scratch.glob('openfocus-frame-*'))


def test_relay_sanitizes_unknown_stage(monkeypatch):
    from core.render_process import _Relay

    relay = _Relay(None, None)
    monkeypatch.setattr(relay, 'send', lambda message: None)
    relay.stage('align')
    assert relay.last_stage == 'align'
    relay.stage('private stage')
    assert relay.last_stage == 'unknown'
    relay.stage('waiting')
    assert relay.last_stage == 'unknown'


@pytest.mark.parametrize('target, message, category', [
    (_sharpen_failure_child, 'Insufficient memory: need 6 MB', 'resource_limit'),
    (_store_failure_child, 'injected output store failure', 'os'),
])
def test_failed_render_preserves_parent_cache_and_cleans_child_scratch(
        qapp, tmp_path, scratch, monkeypatch, caplog, target, message, category):
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
    caplog.set_level(logging.INFO, logger='openfocus')
    monkeypatch.setattr(logging.getLogger('openfocus'), 'propagate', True)
    monkeypatch.setattr(render_process.logger, 'warning', record_warning)
    frames = _sources(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(render_process, '_render_child', target)
        worker = render_process.RenderProcessWorker(
            frames, algorithm='guided_filter', thread_count=1, use_gpu=False,
            tile_enabled=False, exposure_match=False, prepared_cache=parent_cache,
            retain_preparation=True, sharpen_output=(target is _sharpen_failure_child),
            sharpen_strength=175,
            collect_alignment_report=True, profile_performance=True)
        results, errors = _start(worker)
        _wait(worker, qapp)

    assert not results and len(errors) == 1 and message in errors[0]
    fields = _child_evidence(caplog)
    assert fields['error_category'] == category
    assert fields['error_origin'] == 'child_worker'
    assert fields['render_stage'] in ('prepare', 'align', 'fuse', 'finalize')
    if target is _sharpen_failure_child:
        assert fields['error']['kind'] == 'ResourceLimitError'
        assert any(frame['function'] == 'require_memory' for frame in fields['error']['stack'])
        assert fields['admission'] == {
            'resource_status': 'insufficient', 'required_bytes': 6 * 1024**2,
            'ram_ceiling': 3 * 1024**2, 'available_ram_bytes': 8 * 1024**3,
            'total_ram_bytes': 16 * 1024**3, 'available_commit_bytes': 9 * 1024**2,
            'process_commit_headroom_bytes': 5 * 1024**2,
        }
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


def test_spawned_tiled_fusion_publishes_frozen_tooltip_details(qapp, tmp_path, scratch, monkeypatch):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QLabel, QListWidget
    from controllers.output_manager import OutputManager
    from core.render_context import build_render_context
    from core.render_process import RenderProcessWorker
    from locales import trans

    monkeypatch.setattr(trans, 'current_lang', 'en')
    parameters = {'kernel_size': 9}
    worker = RenderProcessWorker(_sources(tmp_path), algorithm='guided_filter',
        fusion_parameters=parameters, tile_enabled=True, tile_threshold=32,
        tile_block_size=100, tile_overlap=64, thread_count=1, use_gpu=False)
    results, errors = _start(worker)
    _wait(worker, qapp)
    assert not errors and len(results) == 1
    assert worker.performance_report is None
    details = worker.fusion_details
    assert (details.input_size, details.tile_grid) == ((64, 48), (48, 24, 6))
    assert dict(details.parameters) == {'kernel_size': 9}

    frames, image, *_ = results[0]
    context = build_render_context(frames, image, algorithm='guided_filter',
                                   parameters=parameters, fusion_details=details)
    panel = SimpleNamespace(output_list=QListWidget(), output_label=QLabel(),
        fusion_results=[], fusion_result=image)
    manager = OutputManager(panel)
    manager.update_output_list_for_fusion('result', context=context)
    parameters['kernel_size'] = 31
    manager._refresh_output_tooltip(0)
    tooltip = panel.output_list.item(0).toolTip()
    assert 'spatial tiles (64 × 48 px input)' in tooltip
    assert 'Block: 48 × 48 px · Grid overlap setting: 24 px · Tile passes: 6' in tooltip
    assert f'{trans.t("label_kernel_guided").rstrip(":： ")} 9' in tooltip
    assert trans.t('retouch_unexported') in tooltip


def test_legacy_tooltip_uses_saved_choice_without_guessing_tiling(monkeypatch):
    from types import SimpleNamespace
    from controllers.output_manager import OutputRecord, _fusion_tooltip_lines
    from locales import trans

    monkeypatch.setattr(trans, 'current_lang', 'zh')
    context = SimpleNamespace(algorithm='stackmffv3', parameters={
        'fusion_parameters': {'mode': 'hard'}, 'tile_enabled': True,
        'tile_block_size': 1024, 'defect_mask_path': 'private/location.tif'})
    record = OutputRecord.create(None, 'archived', context=context)
    tooltip = '\n'.join(_fusion_tooltip_lines(record))
    assert trans.t('radio_ai_hard') in tooltip
    assert trans.t('sensor_mask') in tooltip
    assert '1024' not in tooltip and 'private' not in tooltip
    assert trans.t('output_fusion_whole').split('（')[0] not in tooltip
    assert trans.t('output_fusion_tiled').split('（')[0] not in tooltip


@pytest.mark.parametrize('kind', ['fusion', 'aligned'])
def test_output_tooltip_discloses_registration_accelerator_and_reuse(qapp, monkeypatch, kind):
    from types import SimpleNamespace
    from PyQt6.QtWidgets import QLabel, QListWidget
    from controllers.output_manager import OutputManager, OutputRecord
    from core.render_context import RegistrationDetails, RegistrationDispatch
    from locales import trans

    monkeypatch.setattr(trans, 'current_lang', 'en')
    panel = SimpleNamespace(output_list=QListWidget(), output_label=QLabel(), fusion_results=[])
    manager = OutputManager(panel)
    image = np.zeros((8, 9), np.uint8) if kind == 'fusion' else None
    dispatch = RegistrationDispatch('ecc', 'CPU', 'CPU', 'OpenCL: Test GPU', True)
    record = OutputRecord.create(image, 'result', kind=kind,
        registration_details=RegistrationDetails((dispatch,)))
    if kind == 'fusion':
        record.tooltip_details = ('Fusion Device: OpenCL: Test GPU',)
    item = manager._insert_output(image, 'result', record)
    tooltip = item.toolTip().splitlines()
    assert 'Registration Device: CPU' in tooltip
    assert 'ECC Resize Accelerator: OpenCL: Test GPU' in tooltip
    assert 'ECC Resize: CPU fallback used' in tooltip
    assert ('Fusion Device: OpenCL: Test GPU' in tooltip) == (kind == 'fusion')

    legacy = OutputRecord.create(image, 'cached', kind=kind,
        registration_details=RegistrationDetails(reuse='disk_plan'))
    tooltip = manager._insert_output(image, 'cached', legacy).toolTip().splitlines()
    assert 'Registration: reused cached transforms (applied on CPU)' in tooltip
    assert 'Original Registration Device: unavailable' in tooltip
    assert 'Original Registration Device: CPU' not in tooltip


@pytest.mark.parametrize('alignment', ['none', 'openfocus'])
def test_spawned_render_preserves_uint16_and_lazy_alignment(qapp, tmp_path, scratch, alignment):
    from core.render_process import RenderProcessWorker
    from core.render_context import RegistrationDetails, RegistrationDispatch
    from core.source_frames import ProcessedFrame, SourceFrame, resolve_frame
    from core.workers import RenderWorker
    frames = _sources(tmp_path)
    options = dict(algorithm='openfocus_a', alignment=alignment, tile_enabled=True, tile_threshold=1,
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
    expected_dispatch = (None if alignment == 'none' else
                         RegistrationDetails((RegistrationDispatch('openfocus', 'CPU', 'CPU'),)))
    assert worker.registration_details == reference.registration_details == expected_dispatch
    assert worker.fusion_details == reference.fusion_details
    assert worker.fusion_details.tile_grid is None
    assert worker.fusion_details.input_size == tuple(processed[0].shape[1::-1])
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
        qapp, tmp_path, scratch, monkeypatch, caplog, target):
    from core import render_process

    caplog.set_level(logging.INFO, logger='openfocus')
    monkeypatch.setattr(logging.getLogger('openfocus'), 'propagate', True)
    frames = _sources(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(render_process, '_render_child', target)
        worker = render_process.RenderProcessWorker(
            frames, algorithm='openfocus_a', thread_count=1, use_gpu=False)
        results, errors = _start(worker)
        _wait(worker, qapp)
    assert not results
    assert len(errors) == 1
    fields = _child_evidence(caplog)
    assert fields['error_origin'] == 'child_exit'
    assert fields['error_category'] == worker.failure_category == 'runtime'
    assert 'error' not in fields
    if target is _crash_child:
        assert fields['render_stage'] == 'fuse' and fields['exit_code'] == 23
    else:
        assert fields['render_stage'] == 'prepare' and fields['exit_code'] != 0
    assert qapp is not None and not qapp.closingDown()
    assert not list(scratch.glob('openfocus-frame-*'))
    retry = render_process.RenderProcessWorker(
        frames, algorithm='openfocus_a', tile_enabled=False, thread_count=1,
        use_gpu=False, exposure_match=False)
    results, errors = _start(retry)
    _wait(retry, qapp)
    assert not errors
    assert results[0][1].dtype == np.uint16


def test_child_exit_without_terminal_packet_still_reports_failure(qapp, scratch, monkeypatch, caplog):
    from core import render_process

    caplog.set_level(logging.INFO, logger='openfocus')
    monkeypatch.setattr(logging.getLogger('openfocus'), 'propagate', True)
    monkeypatch.setattr(render_process, '_render_child', _silent_child)
    worker = render_process.RenderProcessWorker([], algorithm=None, need_fusion=False)
    results, errors = _start(worker)
    _wait(worker, qapp)

    assert not results and len(errors) == 1
    assert _child_evidence(caplog) == {'error_category': 'runtime', 'error_origin': 'child_exit',
                                      'render_stage': 'align', 'exit_code': 0}


@pytest.mark.parametrize('platform, code, expected', [
    ('nt', -1073741819, 0xC0000005), ('nt', 0xFFFFFFFF, 0xFFFFFFFF), ('nt', 2**32, None),
    ('posix', -6, -6), ('posix', 2**31, None), ('posix', True, None),
])
def test_exit_evidence_bounds(platform, code, expected, monkeypatch):
    from types import SimpleNamespace
    from core import render_process

    monkeypatch.setattr(render_process, 'os', SimpleNamespace(name=platform))
    assert render_process._bounded_exit_code(code) == expected


def test_cancel_during_child_startup_publishes_no_result(qapp, tmp_path, scratch, monkeypatch, caplog):
    from PyQt6.QtCore import QTimer
    from core.render_process import RenderProcessWorker

    caplog.set_level(logging.INFO, logger='openfocus')
    monkeypatch.setattr(logging.getLogger('openfocus'), 'propagate', True)
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
    assert worker.fusion_details is None
    assert len(errors) == 1 and errors[0].startswith('CANCELLED:')
    assert all(getattr(record, 'event', None) != 'render_child_error' for record in caplog.records)
    assert all(not hasattr(record, 'report_child_evidence') for record in caplog.records)
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
    from PyQt6.QtCore import QRect

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
        fusion_parameters={'fraction': .25, 'direction': 'reverse'}, roi_rect=QRect(4, 3, 5, 6),
        thread_count=1, use_gpu=False, tile_enabled=False, exposure_match=False)
    results, errors = _start(worker)
    _wait(worker, qapp)
    assert not errors
    np.testing.assert_array_equal(results[0][1], np.full((6, 5, 3), 174, np.uint8))
    assert worker.fusion_details.input_size == (20, 16)
    assert worker.fusion_details.tile_grid is None
    assert dict(worker.fusion_details.parameters) == {'fraction': .25, 'direction': 'reverse'}


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
    assert manager._prepared_cache_source is None
    assert path.exists()
    np.testing.assert_array_equal(reader[0], np.ones((8, 9), np.uint8))
    reader.clear()
    gc.collect()
    assert not path.exists()


@pytest.mark.parametrize('isolated', [False, True])
def test_nonstreaming_reuse_preserves_provenance_and_releases_history(
        qapp, tmp_path, isolated_process_env, monkeypatch, isolated):
    from PyQt6.QtCore import QCoreApplication, QEvent
    from core import algorithms, frame_storage, source_frames
    from core.render_context import build_render_context
    from core.render_process import RenderProcessWorker
    from core.workers import RenderWorker

    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', str(scratch))

    monkeypatch.setattr(algorithms, '_FUSION', dict(algorithms._FUSION))
    algorithms.register_fusion(algorithms.FusionAlgorithm(
        'retention_extension', 'test_extension', 'Extension', 'Extension',
        'tests.test_algorithms:Extension.blend'))
    frames = _sources(tmp_path)
    options = dict(alignment='openfocus', thread_count=1, tile_enabled=False,
                   use_gpu=False, sort_mode='off', exposure_match=False)
    first = RenderProcessWorker(frames, algorithm='openfocus_a', **options)
    initial, errors = _start(first)
    _wait(first, qapp)
    assert not errors and len(initial) == 1
    aligned = initial[0][0]
    assert all(isinstance(frame, source_frames.RetainedFrame) for frame in aligned)
    paths = [Path(frame_storage.describe_frame(frame.pixels)['path']) for frame in aligned]
    cls = RenderProcessWorker if isolated else RenderWorker
    second = cls(frames, aligned_images=aligned, is_images_aligned=True,
                 last_alignment_options=(first.alignment, first.registration_parameters,
                                         first._reverse_registration),
                 algorithm='retention_extension', **options)
    results, errors = [], []
    second.finished_signal.connect(lambda *value: results.append(value))
    second.error_signal.connect(errors.append)
    if isolated:
        second.start()
        _wait(second, qapp)
    else:
        second.run()
    assert not errors and len(results) == 1 and second.alignment_reused
    assert all(isinstance(frame, source_frames.RetainedFrame) for frame in results[0][0])
    context = build_render_context(results[0][0], results[0][1])
    assert context is not None
    Path(frames[0].path).write_bytes(b'changed source')
    with pytest.raises(OSError, match='Source changed'):
        source_frames.resolve_frame(results[0][0][0])

    initial.clear()
    results.clear()
    del aligned
    first.deleteLater()
    second.deleteLater()
    del first, second
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    source_frames.SOURCE_CACHE.clear()
    gc.collect()
    assert all(not path.exists() for path in paths)
    assert not any(isinstance(frame.image, source_frames.RetainedFrame)
                   for frame in context.frames)


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('fits', [False, True])
def test_aligned_and_prepared_caches_share_retention_budget(
        qapp, tmp_path, scratch, monkeypatch, existing, fits):
    from core import frame_resolution_cache as cache, source_frames
    from core.workers import RenderWorker

    frames = _sources(tmp_path)
    aligned = [source_frames.crop_frame(frame, 0, 0, frame.shape[1], frame.shape[0])
               for frame in frames]
    if existing:
        with cache.render_scope(frames):
            for frame in aligned:
                source_frames.resolve_frame(frame)
            aligned = cache.retain_aligned_frames(aligned)
        assert all(isinstance(frame, source_frames.RetainedFrame) for frame in aligned)
    aligned_bytes = sum(frame.nbytes for frame in aligned)
    prepared = [np.zeros((16,), np.uint8)]
    monkeypatch.setattr(cache, 'RETAINED_FRAME_LIMIT_BYTES',
                        aligned_bytes + prepared[0].nbytes - (0 if fits else 1))
    options = dict(algorithm='openfocus_a', alignment='openfocus', thread_count=1,
                   tile_enabled=False, use_gpu=False, sort_mode='off',
                   exposure_match=False, retain_aligned_pixels=True)
    worker = RenderWorker(frames, aligned_images=aligned if existing else None,
                          is_images_aligned=existing, **options)
    worker.last_alignment_options = (worker.alignment, worker.registration_parameters,
                                     worker._reverse_registration)
    if not existing:
        monkeypatch.setattr(worker, '_run_registration',
                            lambda images, should_cancel=None: (aligned, 0.0))

    def fuse(images, **kwargs):
        for image in images:
            source_frames.resolve_frame(image)
        worker.prepared_cache = prepared
        return np.zeros(frames[0].shape, frames[0].dtype), 'CPU', 0.0

    monkeypatch.setattr(worker, '_run_fusion', fuse)
    results = []
    worker.finished_signal.connect(lambda *value: results.append(value))
    worker._run_admitted()
    assert len(results) == 1
    output = results[0][0]
    if existing or fits:
        assert all(isinstance(frame, source_frames.RetainedFrame) for frame in output)
    else:
        assert all(not isinstance(frame, source_frames.RetainedFrame) for frame in output)
    assert (worker.prepared_cache is None) == (existing and not fits)
    retained_bytes = sum(frame.nbytes for frame in output
                         if isinstance(frame, source_frames.RetainedFrame))
    prepared_bytes = sum(frame.nbytes for frame in worker.prepared_cache or ())
    assert retained_bytes + prepared_bytes <= cache.RETAINED_FRAME_LIMIT_BYTES


def test_nonstreaming_reuse_keeps_mixed_frame_positions(qapp, tmp_path, scratch, monkeypatch):
    from core import algorithms, frame_resolution_cache as cache, source_frames
    from core.workers import RenderWorker

    monkeypatch.setattr(algorithms, '_FUSION', dict(algorithms._FUSION))
    algorithms.register_fusion(algorithms.FusionAlgorithm(
        'mixed_extension', 'test_extension', 'Extension', 'Extension',
        'tests.test_algorithms:Extension.blend'))
    frames = _sources(tmp_path)
    descriptor = source_frames.crop_frame(frames[0], 0, 0, frames[0].shape[1], frames[0].shape[0])
    with cache.render_scope(frames):
        source_frames.resolve_frame(descriptor)
        retained = cache.retain_aligned_frames([descriptor])[0]
    assert isinstance(retained, source_frames.RetainedFrame)
    ordinary = np.full(frames[0].shape, 321, frames[0].dtype)
    aligned = [ordinary, retained]
    worker = RenderWorker(frames, aligned_images=aligned, is_images_aligned=True,
        algorithm='mixed_extension', alignment='openfocus', thread_count=1,
        tile_enabled=False, use_gpu=False, sort_mode='off', exposure_match=False)
    worker.last_alignment_options = (worker.alignment, worker.registration_parameters,
                                     worker._reverse_registration)
    results = []
    worker.finished_signal.connect(lambda *value: results.append(value))
    worker._run_admitted()
    output = results[0][0]
    np.testing.assert_array_equal(output[0], ordinary)
    assert output[1] is retained


def test_isolated_render_adopts_readonly_focus_map(qapp, scratch):
    from core.frame_storage import describe_frame, owns_temp_file
    from core.render_process import RenderProcessWorker

    shape = (32, 39)
    sharp = (np.indices(shape).sum(axis=0) % 2 * 240 + 8).astype(np.uint8)
    flat = np.full(shape, 128, np.uint8)
    worker = RenderProcessWorker([sharp, flat], algorithm='openfocus_c', use_gpu=False,
        sort_mode='desc', fusion_parameters={'smoothing': 0}, thread_count=1, exposure_match=False)
    stages = []
    worker.stage_signal.connect(stages.append)
    results, errors = _start(worker)
    _wait(worker, qapp)
    assert len(results) == 1 and not errors
    assert stages[-2:] == ['finalize', 'publish']
    retained = worker.rendered_depth
    index = retained['index']
    assert retained['frame_count'] == 2 and retained['reverse_order']
    assert retained['valid_rect'] == (0, 0, shape[1], shape[0])
    assert index.dtype == np.float32 and not index.flags.writeable
    mapping = describe_frame(index)
    assert Path(mapping['path']).parent.is_relative_to(scratch)
    assert owns_temp_file(mapping['path'])
    np.testing.assert_array_equal(index[1:-1, 1:-1], 1)
