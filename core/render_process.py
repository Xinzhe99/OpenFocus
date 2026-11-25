"""One crash-isolated process per GUI render, using the existing pipeline."""
from __future__ import annotations

import atexit
import gc
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback
from hashlib import sha256
from pathlib import Path

from PyQt6.QtCore import Qt

from core import frame_storage as storage
from core.algorithms import (fusion_algorithms, registration_algorithms,
                             get_fusion, get_registration,
                             register_fusion, register_registration)
from core.frame_transport import (FrameDecoder, FrameEncoder, LEGACY_SOURCE_REQUEST_ERROR,
                                  unwrap_source_request, wrap_source_request)
from core.resource_policy import (ResourceLimitError, _ADMISSION_BYTES_LIMIT,
                                  _ADMISSION_BYTE_FIELDS, heavy_job,
                                  make_process_budget, process_budget)
from core.registration import RegistrationCancelled
from core.workers import RenderWorker, _render_error_category
from locales import trans
from utils.logging_setup import get_logger, log_action
from utils.report_bundle import _MAX_FRAMES, safe_exception

logger = get_logger(__name__)
_CANCEL_GRACE_SECONDS = 3.0
_LIVE_PREVIEW_EDGE = 512
_ERROR_STATE_FIELDS = ('alignment_reused', 'alignment_computed', 'alignment_report',
                       'alignment_report_provenance', 'performance_report')
_SUCCESS_STATE_FIELDS = (*_ERROR_STATE_FIELDS, 'prepared_cache', 'rendered_depth', 'fusion_details',
                         'registration_details')
_RENDER_STAGES = ('prepare', 'align', 'fuse', 'finish', 'finalize', 'publish')
_ERROR_CATEGORIES = ('resource_limit', 'memory', 'os', 'value', 'runtime', 'other')
_ERROR_ORIGINS = ('child_worker', 'child_setup_or_transfer')
_CHILD_EXIT_ORIGIN = 'child_exit'
_RENDER_SOURCE_FILES = tuple(Path(__file__).with_name(f'{name}.py') for name in (
    'render_process', 'frame_transport', 'source_frames', 'frame_storage',
    'frame_resolution_cache', 'workers', 'registration', 'resource_policy',
    'algorithms', 'multi_focus_fusion', 'render_context',
    'fusion_live_preview', '_perspective_gpu',
))


def _windows_source_run():
    return sys.platform == 'win32' and not getattr(sys, 'frozen', False)


def _render_source_revision():
    digest = sha256()
    try:
        for path in _RENDER_SOURCE_FILES:
            digest.update(sha256(path.read_bytes()).digest())
    except OSError:
        return None
    return digest.digest()


# Capture the transport and pipeline boundary files when this process loads them.
# Reading only at dispatch would give an old GUI the revision of the new files.
_LOADED_SOURCE_REVISION = _render_source_revision() if _windows_source_run() else None


def _exit_reason(code):
    if os.name == 'nt' and code is not None and (code < 0 or code > 0x7fffffff):
        return f'NTSTATUS 0x{code & 0xffffffff:08X}'
    if code is not None and code < 0:
        number = -code
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = str(number)
        return f'{name} ({signal.strsignal(number)})'
    return f'exit code {code}'


def _bounded_exit_code(code):
    if type(code) is not int or code < -(2**31):
        return None
    if os.name == 'nt':
        return code & 0xffffffff if code < 2**32 else None
    return code if code < 2**31 else None


def _capture_error_details(exc):
    details = {}
    if isinstance(exc, ResourceLimitError):
        admission = _validated_admission(exc.admission)
        if admission is not None:
            details['admission'] = admission
    try:
        frames = ((frame.f_code.co_filename, line, frame.f_code.co_name)
                  for frame, line in traceback.walk_tb(exc.__traceback__))
        error = safe_exception(type(exc).__name__, frames)
        if error['stack_lines']:
            details['error'] = error
    except Exception:
        # Optional diagnostics must not replace the original error signal.
        pass
    return details


def _validated_error(error):
    if type(error) is not dict:
        return None
    kind, lines, stack = error.get('kind'), error.get('stack_lines'), error.get('stack', [])
    if (type(kind) is not str or len(kind) > 64 or type(lines) is not list
            or not 0 < len(lines) <= _MAX_FRAMES
            or any(type(line) is not int or not 0 < line < 10**9 for line in lines)
            or type(stack) is not list or len(stack) > len(lines)):
        return None
    frames = []
    remaining_lines = iter(lines)
    for frame in stack:
        if type(frame) is not dict:
            return None
        file, line, function = frame.get('file'), frame.get('line'), frame.get('function')
        if (type(file) is not str or len(file) > 64 or type(function) is not str
                or len(function) > 64 or type(line) is not int or line not in remaining_lines):
            return None
        frames.append((file, line, function))
    safe = safe_exception(kind, frames)
    if safe['kind'] != kind or safe.get('stack', []) != stack:
        return None
    safe['stack_lines'] = list(lines)
    return safe


def _validated_admission(admission):
    if type(admission) is not dict:
        return None
    status = admission.get('resource_status')
    if type(status) is not str or status not in ('unknown', 'insufficient'):
        return None
    safe = {'resource_status': status}
    for name in _ADMISSION_BYTE_FIELDS:
        value = admission.get(name)
        if type(value) is int and 0 <= value <= _ADMISSION_BYTES_LIMIT:
            safe[name] = value
    return safe


class _LivePreviewMailbox:
    """One latest grayscale map shared with the spawned child, outside the pipe."""

    def __init__(self, shared):
        self._pixels, self._metadata, self._lock = self.shared = shared
        self._last_read = (0, 0)

    @classmethod
    def create(cls, context):
        return cls((context.RawArray('B', _LIVE_PREVIEW_EDGE ** 2),
                    context.RawArray('q', 5),  # width, height, done, total, blending
                    context.Lock()))

    def reset(self):
        with self._lock:
            self._metadata[:] = (0, 0, 0, 0, 0)
            self._last_read = (0, 0)

    def publish(self, gray, done, total, blending):
        height, width = gray.shape
        if not (0 < width <= _LIVE_PREVIEW_EDGE and 0 < height <= _LIVE_PREVIEW_EDGE
                and 0 < done <= total):
            return
        if not self._lock.acquire(False):
            return
        try:
            memoryview(self._pixels).cast('B')[:width * height] = memoryview(gray).cast('B')
            self._metadata[:] = (width, height, done, total, blending)
        finally:
            self._lock.release()

    def take(self):
        if not self._lock.acquire(False):
            return None
        try:
            width, height, done, total, blending = self._metadata[:]
            position = (blending, done)
            if position <= self._last_read:
                return None
            gray = bytes(memoryview(self._pixels)[:width * height])
            self._last_read = position
            return gray, width, height, done, total, bool(blending)
        finally:
            self._lock.release()


class _ChildRenderWorker(RenderWorker):
    def __init__(self, cancel_event, *args, **kwargs):
        self._cancel_event = cancel_event
        self._local_cancelled = False
        self._error_details = None
        super().__init__(*args, **kwargs)

    @property
    def is_cancelled(self):
        return self._local_cancelled or self._cancel_event.is_set()

    @is_cancelled.setter
    def is_cancelled(self, value):
        self._local_cancelled = bool(value)

    def _record_render_error(self, exc):
        super()._record_render_error(exc)
        self._error_details = _capture_error_details(exc)


class _Relay:
    def __init__(self, connection, cancel_event):
        self.connection = connection
        self.cancel_event = cancel_event
        self.lock = threading.RLock()
        self.last_progress = None
        self.last_stage = 'prepare'
        self.directories = set()

    def send(self, message):
        with self.lock:
            try:
                self.connection.send(message)
                return True
            except (EOFError, OSError):
                self.cancel_event.set()
                return False

    def progress(self, done, total):
        with self.lock:
            now = time.monotonic()
            if self.last_progress is None or now - self.last_progress >= .1 or done >= total:
                self.last_progress = now
                self.send(('progress', int(done), int(total)))

    def stage(self, stage):
        # The parent already owns the shared lease. The child's independent
        # process-local lease does not represent a wait for another GUI job.
        if stage != 'waiting':
            with self.lock:
                self.last_stage = stage if stage in _RENDER_STAGES else 'unknown'
                self.last_progress = None
                self.send(('stage', stage))

    def shared_memory(self, name):
        self.send(('shm', name))

    def temporary_file(self, path):
        state = storage.capture_scratch()
        directory = state.directory if state is not None and state.owned else None
        if directory is not None:
            with self.lock:
                self.directories.add(directory)
        self.send(('scratch', path, directory))


def _restore_algorithms(fusions, registrations):
    for specs, lookup, register in (
            (fusions, get_fusion, register_fusion),
            (registrations, get_registration, register_registration)):
        for spec in specs:
            try:
                lookup(spec.id)
            except ValueError:
                register(spec)


def _send_child_error(relay, exc):
    logger.exception('isolated render failed')
    relay.send(('error', str(exc), None,
                (_render_error_category(exc), 'child_setup_or_transfer', relay.last_stage),
                _capture_error_details(exc)))


def _run_decoded_render_job(connection, cancel_event, relay, request, budget, directory,
                            language, *, mailbox=None, live_preview_requested=False):
    """Execute one isolated render job and relay results, errors, and cleanup."""
    from utils.logging_setup import set_log_level
    trans.current_lang = language
    worker = None
    inputs = None
    encoder = None
    results, errors = [], []
    try:
        if _windows_source_run() and multiprocessing.parent_process() is not None:
            request = unwrap_source_request(request, _LOADED_SOURCE_REVISION)
            if request is None:
                raise RuntimeError(trans.t('render_source_changed'))
        if mailbox is None and isinstance(request, tuple) and len(request) == 3 and request[0] == 'live_preview':
            _, request, shared = request
            mailbox = _LivePreviewMailbox(shared)
            live_preview_requested = True
        storage.restore_storage_directory(directory)
        with process_budget(budget), storage.observe_temp_files(relay.temporary_file):
            inputs = FrameDecoder(should_cancel=cancel_event.is_set).decode(request)
            set_log_level(inputs['log_level'])
            _restore_algorithms(inputs['fusions'], inputs['registrations'])
            worker = _ChildRenderWorker(cancel_event, *inputs['args'], **inputs['kwargs'])
            if live_preview_requested and mailbox is not None:
                worker.live_preview_sink = mailbox.publish
            direct = Qt.ConnectionType.DirectConnection
            worker.progress_signal.connect(relay.progress, direct)
            worker.stage_signal.connect(relay.stage, direct)
            worker.finished_signal.connect(lambda *result: results.append(result), direct)
            worker.error_signal.connect(errors.append, direct)
            worker.run()
            if errors:
                state = {name: getattr(worker, name) for name in _ERROR_STATE_FIELDS}
                # Error packets have no receiver acknowledgement.
                encoder = FrameEncoder(allow_shared_memory=False)
                metadata = ((worker._render_error_category, 'child_worker', relay.last_stage)
                            if worker._render_error_category is not None else None)
                message = ('error', errors[-1], encoder.encode(state), metadata)
                # Keep the metadata triple readable by older source-run parents.
                if metadata is not None and worker._error_details:
                    message += (worker._error_details,)
                relay.send(message)
            elif not results:
                raise RuntimeError('Render worker produced neither a result nor an error')
            else:
                relay.stage('publish')
                state = {name: getattr(worker, name) for name in _SUCCESS_STATE_FIELDS}
                encoder = FrameEncoder(cancel_event.is_set, on_shared_memory=relay.shared_memory)
                payload = encoder.encode({'result': results[-1], 'state': state})
                paths = encoder.owned_paths
                if relay.send(('completed', payload, paths, relay.directories)):
                    # Retain every source, map and preview until the GUI has opened it.
                    while not connection.poll(.05):
                        if cancel_event.is_set():
                            break
                    if not cancel_event.is_set() and connection.recv() == 'accepted':
                        storage.disown_temp_files(paths)
                        from core import scratch_registry
                        scratch_registry.forget([*paths, *relay.directories])
    finally:
        if encoder is not None:
            encoder.close(unlink=True)
            encoder = None
        if worker is not None:
            for sig in (worker.progress_signal, worker.stage_signal,
                        worker.finished_signal, worker.error_signal):
                try:
                    sig.disconnect()
                except TypeError:
                    pass
            for attr in ('raw_images', 'aligned_images', 'prepared_cache',
                         'rendered_depth', 'live_preview_sink'):
                if hasattr(worker, attr):
                    setattr(worker, attr, None)
        results.clear()
        errors.clear()
        if isinstance(inputs, dict):
            inputs.clear()


def _render_child(connection, cancel_event, request, budget, directory, language):
    """Top-level spawn target; image computation never runs in the GUI process."""
    from utils.logging_setup import setup_logging
    from utils.diagnostics import install_exception_hooks
    setup_logging()
    install_exception_hooks()
    relay = _Relay(connection, cancel_event)
    try:
        _run_decoded_render_job(connection, cancel_event, relay, request,
                                budget, directory, language)
    except RegistrationCancelled:
        relay.send(('error', 'CANCELLED: render cancelled', None, None))
    except Exception as exc:
        _send_child_error(relay, exc)
    finally:
        connection.close()


_DEFAULT_GET_CONTEXT = multiprocessing.get_context
_DEFAULT_RENDER_CHILD = _render_child
_PERSISTENT_LOCK = threading.Lock()
_PERSISTENT_HANDLE = None
_PERSISTENT_GENERATION = 0
_PREWARM = None


class _PersistentWorkerHandle:
    """Long-lived isolated render child reused across compatible GUI renders."""

    def __init__(self, process, parent_conn, cancel_event, directory, source_revision, mailbox, generation):
        self.process = process
        self.parent_conn = parent_conn
        self.cancel_event = cancel_event
        self.directory = directory
        self.source_revision = source_revision
        self.mailbox = mailbox
        self.generation = generation

    def compatible(self, directory, source_revision):
        return (self.process is not None
                and self.process.is_alive()
                and self.directory == directory
                and self.source_revision == source_revision)

    def close(self, *, bounded=False):
        proc = self.process
        conn = self.parent_conn
        cancel_event = self.cancel_event
        self.process = None
        self.parent_conn = None
        self.cancel_event = None
        self.mailbox = None
        if proc is not None and proc.pid is not None:
            if proc.is_alive() and conn is not None:
                try:
                    conn.send('shutdown')
                except (BrokenPipeError, EOFError, OSError):
                    pass
                proc.join(1.0)
            if not bounded and proc.is_alive() and cancel_event is not None:
                try:
                    cancel_event.set()
                except Exception:
                    pass
                proc.join(_CANCEL_GRACE_SECONDS)
            if proc.is_alive():
                proc.terminate()
                proc.join(1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(1.0 if bounded else None)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if proc is not None:
            try:
                proc.close()
            except ValueError:
                pass


def _create_persistent_worker(context, directory, source_revision, generation):
    cancel_event = context.Event()
    parent, child = context.Pipe()
    try:
        mailbox = _LivePreviewMailbox.create(context)
        process = context.Process(
            target=_persistent_render_child,
            args=(child, cancel_event, directory, mailbox.shared),
            name='OpenFocus render',
            daemon=True,
        )
        return _PersistentWorkerHandle(process, parent, cancel_event, directory,
                                       source_revision, mailbox, generation), child
    except BaseException:
        child.close()
        parent.close()
        raise


def shutdown_persistent_worker():
    """Retire the cached child and cancel any background warmup."""
    global _PERSISTENT_HANDLE, _PERSISTENT_GENERATION
    with _PERSISTENT_LOCK:
        _PERSISTENT_GENERATION += 1
        handle = _PERSISTENT_HANDLE
        _PERSISTENT_HANDLE = None
        prewarm = _PREWARM
    try:
        if handle is not None:
            handle.close(bounded=True)
    finally:
        if prewarm is not None:
            owner, finished = prewarm
            if owner is not threading.current_thread():
                # Process.start() can still be inside the OS. Warmup owns its
                # child until it exits and cannot publish it after shutdown.
                finished.wait(1.0)


def prewarm_persistent_worker(directory=None, *, min_ram_bytes=512 * 1024**2, block: bool = False) -> bool:
    """Schedule an idle render-child warmup, or report its result if blocking."""
    global _PREWARM
    if not _persistent_worker_enabled():
        return False
    scratch = storage.capture_scratch()
    if directory is None and scratch is not None:
        directory = scratch.explicit_directory
    elif directory is not None:
        directory = os.path.abspath(os.fsdecode(directory))
    inherit_directory = directory is None and scratch is None
    warmed = False

    def cancelled():
        with _PERSISTENT_LOCK:
            return generation != _PERSISTENT_GENERATION

    def run():
        global _PREWARM
        nonlocal warmed
        try:
            # Share render/load admission so a user job cannot start a second
            # child while this one is warming and holding its RAM budget.
            with heavy_job(cancelled):
                with _PERSISTENT_LOCK:
                    if _PERSISTENT_HANDLE is not None:
                        return
                source_revision = None
                if _windows_source_run():
                    source_revision = _LOADED_SOURCE_REVISION
                    if source_revision is None or _render_source_revision() != source_revision:
                        return
                from core.resource_policy import probe_memory, ram_budget
                if ram_budget(probe_memory()) < int(min_ram_bytes) or cancelled():
                    return
                target_dir = directory
                if inherit_directory:
                    with storage.scratch_directory() as scratch_dir:
                        target_dir = scratch_dir.explicit_directory
                warmed = _warm_persistent_worker(target_dir, source_revision, generation, cancelled)
        except RegistrationCancelled:
            pass
        except Exception:
            logger.debug('could not prewarm isolated render worker', exc_info=True)
        finally:
            with _PERSISTENT_LOCK:
                _PREWARM = None
                finished.set()

    with _PERSISTENT_LOCK:
        if _PERSISTENT_HANDLE is not None or _PREWARM is not None:
            return False
        generation = _PERSISTENT_GENERATION
        finished = threading.Event()
        if block:
            _PREWARM = threading.current_thread(), finished
        else:
            thread = threading.Thread(target=run, name='openfocus-render-prewarm', daemon=True)
            _PREWARM = thread, finished
            try:
                thread.start()
            except RuntimeError:
                _PREWARM = None
                return False
    if block:
        run()
        return warmed
    return True


def _warm_persistent_worker(directory, source_revision, generation, cancelled):
    handle, child = _create_persistent_worker(multiprocessing.get_context('spawn'), directory,
                                             source_revision, generation)
    try:
        if cancelled():
            return False
        handle.process.start()
        child.close()
        child = None
        if cancelled():
            return False
        parent = handle.parent_conn
        parent.send('warmup')
        deadline = time.monotonic() + 15.0
        while not cancelled() and time.monotonic() < deadline and handle.process.is_alive():
            if parent.poll(0.1):
                if parent.recv() == ('idle',):
                    checked_in = _checkin_persistent_worker(handle, bounded=True)
                    handle = None
                    return checked_in
                break
        return False
    finally:
        if child is not None:
            child.close()
        if handle is not None:
            handle.close(bounded=True)


atexit.register(shutdown_persistent_worker)


def _persistent_worker_enabled():
    return (multiprocessing.get_context is _DEFAULT_GET_CONTEXT
            and _render_child is _DEFAULT_RENDER_CHILD
            and not os.environ.get('OPENFOCUS_DISABLE_PERSISTENT_RENDER_WORKER'))


def _checkout_persistent_worker(directory, source_revision):
    global _PERSISTENT_HANDLE
    with _PERSISTENT_LOCK:
        handle = _PERSISTENT_HANDLE
        if handle is not None:
            _PERSISTENT_HANDLE = None
            if handle.compatible(directory, source_revision):
                return handle
    if handle is not None:
        handle.close()
    return None


def _checkin_persistent_worker(handle, *, bounded=False):
    global _PERSISTENT_HANDLE
    with _PERSISTENT_LOCK:
        if (handle.generation == _PERSISTENT_GENERATION and _PERSISTENT_HANDLE is None
                and handle.process is not None and handle.process.is_alive()):
            _PERSISTENT_HANDLE = handle
            return True
    handle.close(bounded=bounded)
    return False


def _run_one_persistent_job(connection, cancel_event, request, budget, job_directory, language,
                            mailbox, live_preview_requested):
    """Run a single render job in a dedicated stack frame so all mmaps close on return."""
    relay = _Relay(connection, cancel_event)
    try:
        _run_decoded_render_job(connection, cancel_event, relay, request,
                                budget, job_directory, language,
                                mailbox=mailbox, live_preview_requested=live_preview_requested)
    except RegistrationCancelled:
        relay.send(('error', 'CANCELLED: render cancelled', None, None))
    except Exception as exc:
        _send_child_error(relay, exc)


def _persistent_render_child(connection, cancel_event, directory, mailbox_shared):
    """Persistent top-level spawn target handling sequential isolated renders."""
    from utils.logging_setup import setup_logging
    from utils.diagnostics import install_exception_hooks
    setup_logging()
    install_exception_hooks()
    storage.restore_storage_directory(directory)
    mailbox = _LivePreviewMailbox(mailbox_shared) if mailbox_shared is not None else None
    try:
        while True:
            try:
                command = connection.recv()
            except (EOFError, OSError):
                break
            if command is None or command == 'shutdown':
                break
            if command == 'warmup':
                import core.openfocus_alignment  # noqa: F401
                import fusion_methods.openfocus_bc  # noqa: F401
                try:
                    connection.send(('idle',))
                except (EOFError, OSError):
                    break
                continue
            _, request, budget, job_directory, language, live_preview_requested = command
            _run_one_persistent_job(
                connection, cancel_event, request, budget, job_directory,
                language, mailbox, live_preview_requested,
            )
            command = request = budget = None
            from core import source_frames
            source_frames.SOURCE_CACHE.clear()
            source_frames.PREVIEW_CACHE.clear()
            gc.collect()
            gc.collect()
            if cancel_event.is_set():
                break
            try:
                connection.send(('idle',))
            except (EOFError, OSError):
                break
    finally:
        connection.close()


def _cleanup_files(paths, directories):
    for path in paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning('could not remove render scratch file %s', path, exc_info=True)
    for directory in directories:
        try:
            os.rmdir(directory)
        except OSError:
            pass


class _RenderExchange:
    """Parent-side job messages and tentative result ownership."""

    def __init__(self, worker):
        self.worker = worker
        self.files = set()
        self.directories = set()
        self.shared_memory = set()
        self.packet = None
        self.decoder = None
        self.error = None
        self.error_metadata = None
        self.last_stage = 'prepare'

    def restore_state(self, state, fields=_SUCCESS_STATE_FIELDS):
        for name in fields:
            setattr(self.worker, name, state[name])

    def discard_shared_memory(self):
        from core.frame_transport import discard_shared_memory

        for name in self.shared_memory:
            try:
                discard_shared_memory(name)
            except OSError:
                logger.warning('could not release render shared memory', exc_info=True)
        self.shared_memory.clear()

    def receive(self, message, connection):
        kind = message[0]
        if kind == 'scratch':
            from core import scratch_registry
            scratch_registry.register(message[1])
            if message[2] is not None and message[2] not in self.directories:
                scratch_registry.register(message[2])
            self.files.add(message[1])
            if message[2] is not None:
                self.directories.add(message[2])
        elif kind == 'shm':
            self.shared_memory.add(message[1])
        elif kind == 'progress':
            self.worker.progress_signal.emit(*message[1:])
        elif kind == 'stage':
            stage = message[1]
            self.last_stage = stage if type(stage) is str and stage in _RENDER_STAGES else 'unknown'
            self.worker.stage_signal.emit(message[1])
        elif kind == 'error':
            self.error_metadata = None
            self.error = message[1]
            metadata = message[3] if len(message) > 3 else None
            if type(metadata) is tuple and len(metadata) == 3:
                category, origin, stage = metadata
                self.error_metadata = {
                    'error_category': (category if type(category) is str and category in _ERROR_CATEGORIES
                                       else 'other'),
                    'error_origin': (origin if type(origin) is str and origin in _ERROR_ORIGINS
                                     else 'child_setup_or_transfer'),
                    'render_stage': (stage if type(stage) is str and stage in _RENDER_STAGES
                                     else 'unknown'),
                }
                details = message[4] if len(message) > 4 else None
                if type(details) is dict:
                    error = _validated_error(details.get('error'))
                    if error is not None:
                        self.error_metadata['error'] = error
                    admission = _validated_admission(details.get('admission'))
                    if admission is not None:
                        self.error_metadata['admission'] = admission
            if message[2] is not None:
                self.restore_state(FrameDecoder().decode(message[2]), _ERROR_STATE_FIELDS)
        elif kind == 'completed':
            if self.worker.is_cancelled:
                connection.send('cancelled')
            else:
                self.decoder = FrameDecoder(owned_paths=message[2],
                                            owned_directories=message[3])
                self.packet = self.decoder.decode(message[1])
                connection.send('accepted')


class RenderProcessWorker(RenderWorker):
    """Retain RenderWorker's signal/lifecycle API, isolating its native work."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('retain_aligned_pixels', True)
        super().__init__(*args, **kwargs)
        # The controller sets process-only options before starting the worker.
        # Only RenderWorker's constructor arguments go to the child.
        self.storage_directory = None
        self.live_preview_requested = False
        self._render_args = args
        self._render_kwargs = kwargs
        self._live_preview_mailbox = None
        self._process = None
        self._cancel_event = None
        self.child_pid = None
        self.failure_category = None
        self._child_error_evidence = None

    def take_live_preview(self):
        """Poll from the GUI thread without waiting for the child."""
        mailbox = self._live_preview_mailbox
        return mailbox.take() if mailbox is not None else None

    def cancel(self):
        super().cancel()
        if self._cancel_event is not None:
            self._cancel_event.set()

    def _check_current(self):
        current = self.disk_cache.get('is_current') if self.disk_cache else None
        if callable(current) and not current():
            self.cancel()
        return self.is_cancelled

    def run(self):
        self.failure_category = None
        self.fusion_details = None
        self.registration_details = None
        self._child_error_evidence = None
        result = None
        error = None
        try:
            log_action('render_dispatch.worker_started')
            self.stage_signal.emit('waiting')
            with heavy_job(self._check_current), storage.scratch_directory() as scratch:
                log_action('render_dispatch.admitted')
                if self.storage_directory is not None:
                    scratch.explicit_directory = self.storage_directory
                self.stage_signal.emit('prepare')
                result = self._run_process()
        except RegistrationCancelled:
            error = 'CANCELLED: render cancelled'
        except Exception as exc:
            evidence = self._child_error_evidence
            if evidence is not None:
                if not self.is_cancelled:
                    self.failure_category = evidence['error_category']
                    try:
                        log_action('render_child_error', **evidence)
                    except Exception:
                        pass
                    logger.exception('could not complete isolated render', extra={
                        'report_child_evidence': evidence,
                        'report_workload': {'method': self.algorithm, 'registration': self.alignment,
                                            'frame_count': len(self.raw_images), 'preview': self.preview},
                    })
            else:
                if isinstance(exc, MemoryError):
                    self.failure_category = _render_error_category(exc)
                logger.exception('could not complete isolated render')
            error = str(exc)
        self._live_preview_mailbox = None
        if self.is_cancelled:
            self.failure_category = None
            result = None
            error = 'CANCELLED: render cancelled'
        if error is not None:
            self.fusion_details = None
            self.registration_details = None
            self.error_signal.emit(error)
        elif result is not None:
            self.finished_signal.emit(*result)
        self._child_error_evidence = None

    def _encode_request_payload(self, encoder, source_revision, *, live_preview_mailbox=None):
        request = encoder.encode({
            'args': self._render_args,
            'kwargs': self._render_kwargs,
            'fusions': fusion_algorithms(),
            'registrations': registration_algorithms(),
            'log_level': logger.getEffectiveLevel(),
        })
        if live_preview_mailbox is not None:
            # Standard multiprocessing handles keep the same spawn signature
            # so a mismatched Windows child can still reject the revision.
            request = ('live_preview', request, live_preview_mailbox.shared)
        if source_revision is not None:
            request = wrap_source_request(request, source_revision)
        return request

    def _poll_child_exchange(self, process, parent, cancel_event, exchange, *, stop_on_idle=False):
        """Relay messages until the child exits or signals idle; enforce cancel grace."""
        cancel_deadline = None
        idle_confirmed = False
        while True:
            if self._check_current():
                cancel_event.set()
                if cancel_deadline is None:
                    cancel_deadline = time.monotonic() + _CANCEL_GRACE_SECONDS
                elif process.is_alive() and time.monotonic() >= cancel_deadline:
                    process.terminate()
                    process.join(1)
                    if process.is_alive():
                        process.kill()
            try:
                available = parent.poll(.05)
                message = parent.recv() if available else None
            except (EOFError, OSError):
                message = None
                available = False
            if message is not None:
                if stop_on_idle and message[0] == 'idle':
                    idle_confirmed = True
                    break
                exchange.receive(message, parent)
            if not process.is_alive() and not available:
                break
        return idle_confirmed

    def _raise_child_crash(self, exit_code_raw, exchange):
        evidence = dict(exchange.error_metadata or {'error_category': 'runtime'})
        evidence.update(error_origin=_CHILD_EXIT_ORIGIN, render_stage=exchange.last_stage)
        exit_code = _bounded_exit_code(exit_code_raw)
        if exit_code is not None:
            evidence['exit_code'] = exit_code
        self._child_error_evidence = evidence
        raise RuntimeError(trans.t('render_worker_crashed').format(
            reason=_exit_reason(exit_code_raw)))

    def _finalize_exchange_result(self, exchange, source_revision):
        if exchange.error is not None:
            self._child_error_evidence = exchange.error_metadata
            if source_revision is not None and exchange.error == LEGACY_SOURCE_REQUEST_ERROR:
                raise RuntimeError(trans.t('render_source_changed'))
            raise RuntimeError(exchange.error)
        exchange.restore_state(exchange.packet['state'])
        return exchange.packet['result'], exchange.decoder.adopted_paths

    def _run_process(self):
        source_revision = None
        if _windows_source_run():
            source_revision = _LOADED_SOURCE_REVISION
            if source_revision is None or _render_source_revision() != source_revision:
                raise RuntimeError(trans.t('render_source_changed'))
        if _persistent_worker_enabled():
            return self._run_persistent_process(source_revision)
        context = multiprocessing.get_context('spawn')
        self._cancel_event = context.Event()
        parent, child = context.Pipe()
        process = None
        exchange = _RenderExchange(self)
        adopted = set()
        encoder = None
        try:
            log_action('render_dispatch.ipc_ready')
            encoder = FrameEncoder(self._check_current)
            mailbox = None
            if self.live_preview_requested:
                mailbox = self._live_preview_mailbox = _LivePreviewMailbox.create(context)
            request = self._encode_request_payload(encoder, source_revision,
                                                   live_preview_mailbox=mailbox)
            log_action('render_dispatch.request_encoded')
            budget = make_process_budget(storage.ram_snapshot_bytes())
            scratch = storage.capture_scratch()
            directory = scratch.explicit_directory if scratch is not None else None
            process = context.Process(target=_render_child,
                args=(child, self._cancel_event, request, budget,
                      directory, trans.current_lang),
                name='OpenFocus render', daemon=True)
            self._process = process
            log_action('render_dispatch.spawn_started')
            process.start()
            log_action('render_dispatch.spawn_completed')
            self.child_pid = process.pid
            child.close()
            self._poll_child_exchange(process, parent, self._cancel_event, exchange)
            process.join()
            if self.is_cancelled:
                raise RegistrationCancelled()
            if process.exitcode != 0 or (exchange.packet is None and exchange.error is None):
                self._raise_child_crash(process.exitcode, exchange)
            result, adopted = self._finalize_exchange_result(exchange, source_revision)
            return result
        finally:
            # A failed decode/start/connection must not leave a child or scratch behind.
            if process is not None and process.pid is not None:
                if process.is_alive():
                    self._cancel_event.set()
                    process.join(_CANCEL_GRACE_SECONDS)
                if process.is_alive():
                    process.terminate()
                    process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join()
            parent.close()
            child.close()
            if not adopted:
                exchange.packet = exchange.decoder = None
            # Tracebacks retained by logging may outlive this worker.  The input
            # encoder owns borrowed arrays until the child exits, but must not
            # keep them alive through a retained _run_process frame afterwards.
            if encoder is not None:
                encoder.close(unlink=True)
            encoder = None
            exchange.discard_shared_memory()
            _cleanup_files(exchange.files - adopted, exchange.directories)
            if process is not None:
                process.close()
            self._process = None
            self._cancel_event = None

    def _run_persistent_process(self, source_revision):
        context = multiprocessing.get_context('spawn')
        scratch = storage.capture_scratch()
        directory = scratch.explicit_directory if scratch is not None else None
        with _PERSISTENT_LOCK:
            generation = _PERSISTENT_GENERATION
        handle = _checkout_persistent_worker(directory, source_revision)
        child = None
        if handle is None:
            handle, child = _create_persistent_worker(context, directory, source_revision, generation)
        parent = handle.parent_conn
        process = handle.process
        cancel_event = handle.cancel_event
        mailbox = handle.mailbox
        mailbox.reset()
        self._live_preview_mailbox = mailbox if self.live_preview_requested else None
        self._cancel_event = cancel_event
        self._process = process
        exchange = _RenderExchange(self)
        adopted = set()
        encoder = None
        reusable = False
        try:
            log_action('render_dispatch.ipc_ready')
            encoder = FrameEncoder(self._check_current)
            request = self._encode_request_payload(encoder, source_revision)
            log_action('render_dispatch.request_encoded')
            budget = make_process_budget(storage.ram_snapshot_bytes())
            if process.pid is None:
                log_action('render_dispatch.spawn_started')
                process.start()
                log_action('render_dispatch.spawn_completed')
            if child is not None:
                child.close()
                child = None
            self.child_pid = process.pid
            parent.send(('run', request, budget, directory, trans.current_lang, bool(self.live_preview_requested)))
            idle_confirmed = self._poll_child_exchange(
                process, parent, cancel_event, exchange, stop_on_idle=True)
            if not idle_confirmed and not process.is_alive():
                process.join()
            if self.is_cancelled:
                raise RegistrationCancelled()
            if (not idle_confirmed and process.exitcode not in (0, None)) or (
                exchange.packet is None and exchange.error is None
            ):
                exit_code_raw = process.exitcode if not process.is_alive() else -1
                self._raise_child_crash(exit_code_raw, exchange)
            category = (exchange.error_metadata or {}).get('error_category')
            # A retry after memory exhaustion needs fresh native allocators
            # and device contexts.
            reusable = (idle_confirmed and process.is_alive() and not cancel_event.is_set()
                        and category not in ('resource_limit', 'memory'))
            result, adopted = self._finalize_exchange_result(exchange, source_revision)
            return result
        finally:
            if child is not None:
                try:
                    child.close()
                except OSError:
                    pass
            if not adopted:
                exchange.packet = exchange.decoder = None
            if reusable:
                _checkin_persistent_worker(handle)
            else:
                handle.close()
            if encoder is not None:
                encoder.close(unlink=True)
            encoder = None
            exchange.discard_shared_memory()
            _cleanup_files(exchange.files - adopted, exchange.directories)
            self._process = None
            self._cancel_event = None
