"""One crash-isolated process per GUI render, using the existing pipeline."""
from __future__ import annotations

import multiprocessing
import os
import signal
import threading
import time

from PyQt6.QtCore import Qt

from core import frame_storage as storage
from core.algorithms import (fusion_algorithms, registration_algorithms,
                             get_fusion, get_registration,
                             register_fusion, register_registration)
from core.frame_transport import FrameDecoder, FrameEncoder
from core.resource_policy import heavy_job, make_process_budget, process_budget
from core.registration import RegistrationCancelled
from core.workers import RenderWorker
from locales import trans
from utils.logging_setup import debug_enabled, get_logger

logger = get_logger(__name__)
_CANCEL_GRACE_SECONDS = 3.0
_ERROR_STATE_FIELDS = ('alignment_reused', 'alignment_computed', 'alignment_report',
                       'alignment_report_provenance', 'performance_report')
_SUCCESS_STATE_FIELDS = (*_ERROR_STATE_FIELDS, 'prepared_cache')


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


class _ChildRenderWorker(RenderWorker):
    def __init__(self, cancel_event, *args, **kwargs):
        self._cancel_event = cancel_event
        self._local_cancelled = False
        super().__init__(*args, **kwargs)

    @property
    def is_cancelled(self):
        return self._local_cancelled or self._cancel_event.is_set()

    @is_cancelled.setter
    def is_cancelled(self, value):
        self._local_cancelled = bool(value)


class _Relay:
    def __init__(self, connection, cancel_event):
        self.connection = connection
        self.cancel_event = cancel_event
        self.lock = threading.RLock()
        self.last_progress = None
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
                self.last_progress = None
                self.send(('stage', stage))

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


def _render_child(connection, cancel_event, request, budget, directory, language):
    """Top-level spawn target; image computation never runs in the GUI process."""
    from utils.logging_setup import set_debug_logging, setup_logging
    from utils.diagnostics import install_exception_hooks
    setup_logging()
    install_exception_hooks()
    storage.restore_storage_directory(directory)
    trans.current_lang = language
    relay = _Relay(connection, cancel_event)
    try:
        with process_budget(budget), storage.observe_temp_files(relay.temporary_file):
            inputs = FrameDecoder(should_cancel=cancel_event.is_set).decode(request)
            if inputs.get('detailed_logging'):
                # Record the same detail as the window that started this render.
                set_debug_logging(True)
            _restore_algorithms(inputs['fusions'], inputs['registrations'])
            worker = _ChildRenderWorker(cancel_event, *inputs['args'], **inputs['kwargs'])
            results, errors = [], []
            direct = Qt.ConnectionType.DirectConnection
            worker.progress_signal.connect(relay.progress, direct)
            worker.stage_signal.connect(relay.stage, direct)
            worker.finished_signal.connect(lambda *result: results.append(result), direct)
            worker.error_signal.connect(errors.append, direct)
            worker.run()
            if errors:
                state = {name: getattr(worker, name) for name in _ERROR_STATE_FIELDS}
                encoder = FrameEncoder()
                relay.send(('error', errors[-1], encoder.encode(state)))
                return
            if not results:
                raise RuntimeError('Render worker produced neither a result nor an error')
            state = {name: getattr(worker, name) for name in _SUCCESS_STATE_FIELDS}
            encoder = FrameEncoder(cancel_event.is_set)
            payload = encoder.encode({'result': results[-1], 'state': state})
            paths = encoder.owned_paths
            if not relay.send(('completed', payload, paths, relay.directories)):
                return
            # Retain every source, map and preview until the GUI has opened it.
            while not connection.poll(.05):
                if cancel_event.is_set():
                    return
            if connection.recv() == 'accepted':
                storage.disown_temp_files(paths)
                from core import scratch_registry
                scratch_registry.forget([*paths, *relay.directories])
    except RegistrationCancelled:
        relay.send(('error', 'CANCELLED: render cancelled', None))
    except Exception as exc:
        logger.exception('isolated render failed')
        relay.send(('error', str(exc), None))
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
        self.packet = None
        self.decoder = None
        self.error = None

    def restore_state(self, state, fields=_SUCCESS_STATE_FIELDS):
        for name in fields:
            setattr(self.worker, name, state[name])

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
        elif kind == 'progress':
            self.worker.progress_signal.emit(*message[1:])
        elif kind == 'stage':
            self.worker.stage_signal.emit(message[1])
        elif kind == 'error':
            self.error = message[1]
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
        # A directory chosen for this render only (set before start). It binds
        # the parent's job scope and therefore the child (see _run_process).
        # Not a constructor argument: the arguments are RenderWorker's own.
        self.storage_directory = None
        self._render_args = args
        self._render_kwargs = kwargs
        self._process = None
        self._cancel_event = None
        self.child_pid = None

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
        result = None
        error = None
        try:
            self.stage_signal.emit('waiting')
            with heavy_job(self._check_current), storage.scratch_directory() as scratch:
                if self.storage_directory is not None:
                    scratch.explicit_directory = self.storage_directory
                self.stage_signal.emit('prepare')
                result = self._run_process()
        except RegistrationCancelled:
            error = 'CANCELLED: render cancelled'
        except Exception as exc:
            logger.exception('could not complete isolated render')
            error = str(exc)
        if self.is_cancelled:
            result = None
            error = 'CANCELLED: render cancelled'
        if error is not None:
            self.error_signal.emit(error)
        elif result is not None:
            self.finished_signal.emit(*result)

    def _run_process(self):
        context = multiprocessing.get_context('spawn')
        self._cancel_event = context.Event()
        parent, child = context.Pipe()
        process = None
        exchange = _RenderExchange(self)
        adopted = set()
        encoder = None
        try:
            encoder = FrameEncoder(self._check_current)
            request = encoder.encode({'args': self._render_args, 'kwargs': self._render_kwargs,
                'fusions': fusion_algorithms(), 'registrations': registration_algorithms(),
                'detailed_logging': debug_enabled()})
            budget = make_process_budget(storage.ram_snapshot_bytes())
            scratch = storage.capture_scratch()
            directory = scratch.explicit_directory if scratch is not None else None
            process = context.Process(target=_render_child,
                args=(child, self._cancel_event, request, budget,
                      directory, trans.current_lang),
                name='OpenFocus render', daemon=True)
            self._process = process
            process.start()
            self.child_pid = process.pid
            child.close()
            cancel_deadline = None
            while True:
                if self._check_current():
                    self._cancel_event.set()
                    if cancel_deadline is None:
                        cancel_deadline = time.monotonic() + _CANCEL_GRACE_SECONDS
                    elif process.is_alive() and time.monotonic() >= cancel_deadline:
                        process.terminate()
                        process.join(1)
                        if process.is_alive():
                            process.kill()
                try:
                    available = parent.poll(.05)
                    if available:
                        message = parent.recv()
                    else:
                        message = None
                except (EOFError, OSError):
                    message = None
                    available = False
                if message is not None:
                    exchange.receive(message, parent)
                if not process.is_alive() and not available:
                    break
            process.join()
            if self.is_cancelled:
                raise RegistrationCancelled()
            if process.exitcode != 0 or (exchange.packet is None and exchange.error is None):
                raise RuntimeError(trans.t('render_worker_crashed').format(
                    reason=_exit_reason(process.exitcode)))
            if exchange.error is not None:
                raise RuntimeError(exchange.error)
            exchange.restore_state(exchange.packet['state'])
            adopted = exchange.decoder.adopted_paths
            return exchange.packet['result']
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
            encoder = None
            _cleanup_files(exchange.files - adopted, exchange.directories)
            if process is not None:
                process.close()
            self._process = None
            self._cancel_event = None
