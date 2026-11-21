"""Admit and dispatch StackMFF inference to installed native and ONNX engines."""
import atexit
from collections import OrderedDict
from contextlib import contextmanager
import json
import hashlib
import os
import sys
import tempfile
import uuid
import threading
import time
import traceback

import cv2
import numpy as np

from utils.logging_setup import get_logger, log_action
from utils.performance_profile import current as _profile

_LOG = get_logger(__name__)
_SESSIONS = OrderedDict()
_PROVIDER = {'CPU': 'CPUExecutionProvider', 'DIRECTML': 'DmlExecutionProvider',
             'CUDA': 'CUDAExecutionProvider', 'ROCM': 'ROCMExecutionProvider',
             'OPENVINO': 'OpenVINOExecutionProvider', 'COREML': 'CoreMLExecutionProvider'}
_ORT_ORDER = ('CUDA', 'ROCM', 'DIRECTML', 'OPENVINO', 'COREML')
_NATIVE_ORDER = ('TENSORRT', 'TORCH_CUDA', 'TORCH_MPS')
# A refresh publishes one immutable installation snapshot. A live job holds
# its own reference, so an installer cannot switch its interpreter mid-job.
_PACK = None
_JOB_PACK = None
_WORKER = None
_JOB_RAM_LIMIT = _JOB_VRAM_LIMIT = None
_WORKER_FAILURE = None
_FAILED_SHAPES = {}
_FALLBACK_REASONS = []
_JOB_DEVICES = []
_DEVICE = None
_DIRECTML_ADAPTER_INDEX = 0
_ADAPTER_NAME = ''
_LOCK = threading.RLock()
_ACTUAL_DEVICE = None
_JOB_ACCELERATOR = None
_FALLBACK_CPU = False
_ACTUAL_CHECKPOINT = None
_AVAILABLE_PROVIDERS = ()
_HARDWARE_ADAPTERS = ()
_PROBE_ERRORS = ()
# Verified accelerator that logits() will not admit because its free memory
# cannot be read; AI jobs then run on CPU.
_UNADMITTED = None
# Providers whose jobs logits() admits only against known free device memory.
VRAM_GATED_KINDS = ('DIRECTML', 'CUDA', 'ROCM', 'OPENVINO', 'TENSORRT', 'TORCH_CUDA')
UNIFIED_MEMORY_KINDS = ('COREML', 'TORCH_MPS')


def native_startup_bytes(kind):
    return 512 * 1024**2 if kind in _NATIVE_ORDER else 0


def device_diagnostics():
    """Cached hardware, provider capability and job execution; never probes."""
    return {'hardware': _HARDWARE_ADAPTERS, 'providers': _AVAILABLE_PROVIDERS,
            'ort_selected': _DEVICE,
            'selected': cached_device(), 'adapter': _adapter_name(cached_device()),
            'eligible_native': _PACK[0] if _PACK else (),
            'actual_devices': tuple(_JOB_DEVICES), 'fallback_reasons': tuple(_FALLBACK_REASONS),
            'actual': _ACTUAL_DEVICE, 'checkpoint': _ACTUAL_CHECKPOINT,
            'fallback_cpu': _FALLBACK_CPU, 'probe_errors': _PROBE_ERRORS,
            'unadmitted': _UNADMITTED}


def unadmitted_accelerator():
    """The verified accelerator AI jobs cannot use (free memory unreadable), or None."""
    return _UNADMITTED


def admitted_kind(kind):
    """The device AI jobs will actually run on: CPU for an unadmitted accelerator.

    A provider that passes the probe but whose free memory cannot be read is
    sent to CPU by every job, so it must not be shown as ready.
    """
    if kind is not None and kind == _UNADMITTED:
        return 'CPU'
    return kind


def _vram_snapshot(kind):
    """Probe kind's free memory; None when the probe itself fails.

    Provider selection and every job read it the same way, so a probe error
    means unknown memory (CPU) in both places instead of failing the job.
    """
    from core.resource_policy import probe_resources
    try:
        return probe_resources(kind)
    except Exception:
        _LOG.warning('%s memory probe failed; treating free memory as unknown',
                     kind, exc_info=True)
        return None


def _vram_unknown(snapshot):
    return snapshot is None or not snapshot.vram_known


def _memory_unreadable(kind):
    return kind in VRAM_GATED_KINDS and _vram_unknown(_vram_snapshot(kind))


def _record_admission(kind, snapshot):
    """Keep the reported admission tied to the latest probe a job acted on."""
    global _UNADMITTED
    with _LOCK:
        if _vram_unknown(snapshot):
            _UNADMITTED = kind
        elif _UNADMITTED == kind:
            _UNADMITTED = None


def finish_inference_job():
    global _WORKER
    worker, _WORKER = _WORKER, None
    if worker is not None:
        worker.close()


atexit.register(finish_inference_job)


@contextmanager
def inference_job(owned=True):
    """Enter after acquiring the heavy-job lease. Nested callers borrow it."""
    if owned:
        start_inference_job()
    try:
        yield
    finally:
        if owned:
            finish_inference_job()


def start_inference_job():
    """Reset actual-device reporting and fallback for a new serialized job."""
    global _ACTUAL_DEVICE, _JOB_ACCELERATOR, _FALLBACK_CPU, _ACTUAL_CHECKPOINT
    global _JOB_PACK, _WORKER_FAILURE, _JOB_RAM_LIMIT, _JOB_VRAM_LIMIT
    finish_inference_job()
    _JOB_PACK = _WORKER_FAILURE = None
    _JOB_RAM_LIMIT = _JOB_VRAM_LIMIT = None
    _FAILED_SHAPES.clear()
    _FALLBACK_REASONS.clear()
    _JOB_DEVICES.clear()
    _ACTUAL_DEVICE = _JOB_ACCELERATOR = _ACTUAL_CHECKPOINT = None
    _FALLBACK_CPU = False


def actual_device():
    """Device actually used by the current job, or None before inference."""
    return _ACTUAL_DEVICE


def actual_device_report():
    """Name the device used by this job, retaining successful GPU work on fallback."""
    if len(_JOB_DEVICES) > 1:
        return ' + '.join(display_name(kind) for kind in sorted(_JOB_DEVICES, key=lambda k: k != 'CPU'))
    if _ACTUAL_DEVICE == 'CPU' and _JOB_ACCELERATOR is not None:
        return 'CPU + ' + display_name(_JOB_ACCELERATOR)
    return display_name(_ACTUAL_DEVICE or 'CPU')


def cached_device():
    """Return the latest verified device, or None before probing."""
    return _PACK[0][0] if _PACK and _PACK[0] else _DEVICE



def is_accelerator(kind):
    return kind in _ORT_ORDER or kind in _NATIVE_ORDER


def _adapter_name(kind):
    if kind in _NATIVE_ORDER:
        pack = _JOB_PACK if _JOB_PACK is not None else _PACK
        return pack[2] if pack and kind in pack[0] else ''
    return _ADAPTER_NAME if kind == _DEVICE else ''


def display_name(kind, *, include_adapter=True):
    provider = {'DIRECTML': 'DirectML', 'CUDA': 'CUDA', 'ROCM': 'ROCm',
                'OPENVINO': 'OpenVINO GPU', 'COREML': 'CoreML',
                'TENSORRT': 'TensorRT', 'TORCH_CUDA': 'PyTorch CUDA',
                'TORCH_MPS': 'PyTorch MPS'}.get(kind, str(kind))
    adapter = _adapter_name(kind) if include_adapter else ''
    return f'{provider} ({adapter})' if adapter else provider


def gpu_toggle_state(kind, use_gpu, available):
    enabled = is_accelerator(kind) or (kind is None and bool(available))
    return enabled, bool(enabled and use_gpu)


def _driver_adapter_name(kind, index):
    """Query the driver ordinal configured as the execution provider's device_id."""
    # OpenVINO's auto-selected GPU and CoreML's compute units are opaque
    # here. Keep the provider label when we cannot identify its adapter.
    if kind not in ('CUDA', 'ROCM'):
        return ''
    import ctypes

    name = ctypes.create_string_buffer(256)
    try:
        if kind == 'CUDA':
            driver = ctypes.CDLL('nvcuda.dll' if sys.platform == 'win32' else 'libcuda.so.1')
            device = ctypes.c_int()
            if driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device), index):
                return ''
            status = driver.cuDeviceGetName(name, len(name), device.value)
        else:
            driver = ctypes.CDLL('amdhip64.dll' if sys.platform == 'win32' else 'libamdhip64.so')
            if driver.hipInit(0):
                return ''
            status = driver.hipDeviceGetName(name, len(name), index)
        if status == 0:
            return name.value.decode('utf-8', errors='replace').strip()
    except (OSError, AttributeError, ValueError):
        _LOG.debug('Could not identify the %s adapter', kind, exc_info=True)
    return ''


class _ProfiledSession:
    """Inspect placement once per model/shape session, then reuse it normally."""

    def __init__(self, session, device):
        self.session = session
        self.device = device
        self.actual_device = None
        self._profiling = True

    def _finish_profile(self, inspect):
        self._profiling = False
        profile = self.session.end_profiling()
        try:
            if not inspect:
                return
            with open(profile, encoding='utf-8') as stream:
                events = json.load(stream)
            providers = {event.get('args', {}).get('provider') for event in events}
            if _PROVIDER[self.device] in providers:
                return self.device
            if _PROVIDER['CPU'] in providers:
                return 'CPU'
            raise RuntimeError('ONNX profile contains no verified execution provider')
        finally:
            if profile:
                os.remove(profile)

    def run(self, outputs, inputs):
        try:
            result = self.session.run(outputs, inputs)
        except Exception:
            if self._profiling:
                try:
                    self._finish_profile(False)
                except Exception:
                    _LOG.warning('Could not remove failed inference profile\n%s', traceback.format_exc())
            raise
        if self._profiling:
            self.actual_device = self._finish_profile(True)
        return result


def refresh_accelerator_pack():
    """Refresh local pack eligibility without importing its engines or running a model."""
    global _PACK
    from core.inference_pack import pack_status, worker_command
    with _LOCK:
        try:
            command = worker_command()
            status = pack_status() if command else None
        except (OSError, KeyError, ValueError, TypeError):
            _LOG.warning('Optional accelerator installation could not be read', exc_info=True)
            command, status = None, None
        adapter = status.device if status and status.supported else ''
        engines = tuple(k for k in _NATIVE_ORDER if adapter and k in status.backends)
        _PACK = (engines, tuple(command or ()), adapter)
        return engines


def resolve_device(use_gpu):
    if not use_gpu:
        return 'CPU'
    if _PACK is None:
        refresh_accelerator_pack()
    if _PACK[0]:
        return _PACK[0][0]
    return resolve_ort_device()


def resolve_ort_device():
    """Probe the compact-runtime provider independently of native pack eligibility."""
    global _DEVICE, _DIRECTML_ADAPTER_INDEX, _ADAPTER_NAME
    global _AVAILABLE_PROVIDERS, _HARDWARE_ADAPTERS, _PROBE_ERRORS, _UNADMITTED
    with _LOCK:
        if _DEVICE is None:
            _UNADMITTED = None
            _ADAPTER_NAME = ''
            import onnxruntime as ort
            from utils.weights import weights_path
            available = ort.get_available_providers()
            from core.resource_policy import directml_adapters
            _AVAILABLE_PROVIDERS = tuple(available)
            _HARDWARE_ADAPTERS = tuple(directml_adapters())
            errors = []
            log_action('neural_provider_probe', providers=available,
                       hardware=_HARDWARE_ADAPTERS)
            device = 'CPU'
            for kind in _ORT_ORDER:
                if _PROVIDER[kind] not in available:
                    continue
                if kind == 'DIRECTML':
                    candidates = _HARDWARE_ADAPTERS or [(0, '', 0, 0)]
                else:
                    candidates = [(0, '', 0, 0)]
                for adapter_index, adapter_name, _vendor, _memory in candidates:
                    try:
                        options = ort.SessionOptions()
                        options.profile_file_prefix = os.path.join(tempfile.gettempdir(), 'openfocus-ort-' + uuid.uuid4().hex)
                        options.enable_profiling = True
                        options.intra_op_num_threads = 1
                        for name, value in zip(('layers', 'height', 'width'), (2, 64, 64)):
                            options.add_free_dimension_override_by_name(name, value)
                        if kind == 'DIRECTML':
                            options.enable_mem_pattern = False
                            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                        provider = _provider(kind, adapter_index)
                        session = ort.InferenceSession(weights_path('stackmff_v3_final.onnx'),
                            sess_options=options, providers=[provider, 'CPUExecutionProvider'])
                        session.disable_fallback()
                        session = _ProfiledSession(session, kind)
                        session.run(['logits'], {'input': np.zeros((1, 2, 64, 64), dtype=np.float32)})
                        if session.actual_device != kind:
                            raise RuntimeError(f'{kind} did not execute any model nodes')
                        if kind == 'DIRECTML':
                            from core.resource_policy import set_directml_adapter
                            set_directml_adapter(adapter_index)
                            _DIRECTML_ADAPTER_INDEX = adapter_index
                        _ADAPTER_NAME = (adapter_name.strip() if kind == 'DIRECTML'
                                         else _driver_adapter_name(kind, adapter_index))
                        if _memory_unreadable(kind):
                            _UNADMITTED = kind
                        device = kind
                        break
                    except Exception as error:
                        errors.append(f'{kind} adapter {adapter_index}: {type(error).__name__}: {error}')
                        _LOG.warning('%s provider probe failed for adapter %s', kind,
                                     adapter_index, exc_info=True)
                if device != 'CPU':
                    break
            _DEVICE = device
            _PROBE_ERRORS = tuple(errors)
            log_action('neural_provider_selected', provider=_PROVIDER[device],
                       adapter=_ADAPTER_NAME or None,
                       adapter_index=adapter_index if is_accelerator(device) else None,
                       fallback=device == 'CPU', probe_errors=errors)
        return _DEVICE


def _provider(kind, index=0):
    if kind in ('DIRECTML', 'CUDA', 'ROCM'):
        return _PROVIDER[kind], {'device_id': index}
    if kind == 'OPENVINO':
        return _PROVIDER[kind], {'device_type': 'GPU'}
    return _PROVIDER[kind]

def estimate_inference_bytes(n, h, w, kind):
    hp, wp = h + (-h) % 64, w + (-w) % 64
    pixels = hp * wp
    feature_base = 7 if kind == 'CPU' else 12
    workspace = 0
    if kind == 'TENSORRT':
        from core.inference_tensorrt import workspace_bytes
        workspace = workspace_bytes(n, hp, wp)
    return (int(n * pixels * 3 * 4 * (feature_base + 4))
            + int(pixels / 16 * 3 * 4 * (n * 64 + n * n))
            + n * pixels * 4 * 6 + 4 * 1024 * 1024 + workspace)


def _session_key(path, shape, device, threads):
    path = os.path.abspath(path)
    stat = os.stat(path)
    adapter = _DIRECTML_ADAPTER_INDEX if device == 'DIRECTML' else 0
    return (path, stat.st_mtime_ns, stat.st_size, shape, device, adapter, int(threads or 1))


# Optional ONNX Runtime operator traces (Chrome trace JSON), CPU sessions only.
# Accelerated sessions already profile their first run to verify placement.
OPERATOR_PROFILE_ENV = 'OPENFOCUS_ORT_OPERATOR_PROFILE'
OPERATOR_PROFILE_DIR_ENV = 'OPENFOCUS_ORT_PROFILE_DIR'
_TRACED = {}


def operator_trace_directory():
    """Configured trace directory, or None when operator tracing is off or unusable."""
    if os.environ.get(OPERATOR_PROFILE_ENV) != '1':
        return None
    directory = os.environ.get(OPERATOR_PROFILE_DIR_ENV)
    if not directory:
        from utils.logging_setup import logs_dir
        directory = os.path.join(str(logs_dir()), 'ort-traces')
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        # A diagnostic must not fail the render it observes.
        _LOG.warning('ONNX Runtime operator traces disabled: cannot use %s', directory,
                     exc_info=True)
        return None
    return directory


def _end_trace(key):
    session = _TRACED.pop(key, None)
    if session is None:
        return
    try:
        path = session.end_profiling()
        log_action('ort_operator_trace', file=os.path.basename(path) if path else None,
                   shape=list(key[3]), threads=key[6])
    except Exception:
        _LOG.warning('Could not finish an ONNX Runtime operator trace\n%s', traceback.format_exc())


def finish_operator_traces():
    """End every open operator trace so ORT writes its file."""
    for key in list(_TRACED):
        _end_trace(key)


atexit.register(finish_operator_traces)


def _describe_session(key):
    """Path-free session identity for performance reports."""
    return (os.path.basename(key[0]), tuple(int(v) for v in key[3]), key[4], key[6])


def _clear_sessions():
    profile = _profile()
    if profile.enabled and _SESSIONS:
        profile.sessions_cleared([_describe_session(key) for key in _SESSIONS])
    for key in _SESSIONS:
        _end_trace(key)
    _SESSIONS.clear()


def _session(path, shape, device, threads):
    import onnxruntime as ort
    global _SESSIONS
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'ONNX model not found: {path}')
    key = _session_key(path, shape, device, threads)
    adapter = key[5]
    profile = _profile()
    if profile.enabled:
        profile.session_lookup(_describe_session(key), key in _SESSIONS)
    if key not in _SESSIONS:
        # Two shapes keep the full and reduced inference scales compiled.
        # Admission in logits accounts for the retained accelerator session.
        while len(_SESSIONS) >= 2:
            evicted, session = _SESSIONS.popitem(last=False)
            _end_trace(evicted)
            del session
            _LOG.debug('Neural session evicted shape=%s device=%s threads=%s for shape=%s',
                       evicted[3], evicted[4], evicted[6], shape)
            if profile.enabled:
                profile.session_evicted(_describe_session(evicted))
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads or 1))
        options.enable_cpu_mem_arena = False
        trace_directory = operator_trace_directory() if device == 'CPU' else None
        if device != 'CPU':
            options.enable_profiling = True
            options.profile_file_prefix = os.path.join(
                tempfile.gettempdir(), 'openfocus-ort-' + uuid.uuid4().hex)
        elif trace_directory is not None:
            options.enable_profiling = True
            options.profile_file_prefix = os.path.join(
                trace_directory, 'openfocus-ort-{}-t{}-{}'.format(
                    'x'.join(str(int(v)) for v in shape), options.intra_op_num_threads,
                    uuid.uuid4().hex[:8]))
        # DirectML needs concrete dimensions for attention MatMul.
        for name, value in zip(('layers', 'height', 'width'), shape):
            options.add_free_dimension_override_by_name(name, int(value))
        if device == 'DIRECTML':
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        provider = _provider(device, _DIRECTML_ADAPTER_INDEX if device == 'DIRECTML' else 0)
        constructed = time.perf_counter_ns() if profile.enabled else 0
        session = ort.InferenceSession(path, sess_options=options,
                                       providers=[provider, 'CPUExecutionProvider']
                                       if device != 'CPU' else ['CPUExecutionProvider'])
        session.disable_fallback()
        if profile.enabled:
            constructed = time.perf_counter_ns() - constructed
        if device != 'CPU' and _PROVIDER[device] not in session.get_providers():
            try:
                _ProfiledSession(session, device)._finish_profile(False)
            finally:
                raise RuntimeError(f'{device} provider did not initialize')
        hashed = time.perf_counter_ns() if profile.enabled else 0
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(64 * 1024), b''):
                digest.update(block)
        if profile.enabled:
            profile.session_built(time.perf_counter_ns() - hashed, constructed)
        log_action('neural_session', checkpoint=os.path.basename(path),
                   checkpoint_sha256=digest.hexdigest(), provider=_PROVIDER[device],
                   adapter=display_name(device), adapter_index=adapter,
                   shape=shape, threads=options.intra_op_num_threads)
        if device != 'CPU':
            session = _ProfiledSession(session, device)
        elif trace_directory is not None:
            _TRACED[key] = session
        _SESSIONS[key] = session
    _SESSIONS.move_to_end(key)
    return _SESSIONS[key]


def _log_session_failure(shape, session, device):
    """Record bounded failure context without paths or exception text."""
    try:
        from utils.memory import failure_memory_snapshot
        details = failure_memory_snapshot()
        details['shape'] = [int(value) for value in shape]
        if device in _PROVIDER:
            details['requested_provider'] = _PROVIDER[device]
        actual = ('CPU' if session is not None and device == 'CPU'
                  else getattr(session, 'actual_device', None))
        if actual in _PROVIDER:
            details['provider'] = _PROVIDER[actual]
        log_action('neural_session_failure', **details)
    except Exception:
        pass


def apply_inference_budget(plan):
    """Carry a serialized stage's frozen limits into actual shape admission."""
    global _JOB_RAM_LIMIT, _JOB_VRAM_LIMIT, _FALLBACK_CPU
    _JOB_RAM_LIMIT = plan.ram_budget
    _JOB_VRAM_LIMIT = plan.vram_budget or None
    if plan.device_fallback and plan.device_fallback not in _FALLBACK_REASONS:
        _FALLBACK_REASONS.append(plan.device_fallback)
        _FALLBACK_CPU |= plan.device_kind == 'CPU'


def candidate_devices(device, *, probe_ort=True):
    """Yield increasingly compatible engines, deferring ORT probes until needed."""
    if device == 'CPU':
        yield 'CPU'
        return
    if device in _NATIVE_ORDER:
        pack = _JOB_PACK if _JOB_PACK is not None else _PACK
        yield device
        if pack:
            yield from (kind for kind in _NATIVE_ORDER[_NATIVE_ORDER.index(device) + 1:]
                        if kind in pack[0])
        device = resolve_ort_device() if probe_ort else _DEVICE
    if device in _ORT_ORDER:
        yield device
    for kind in _ORT_ORDER:
        if kind != device and _PROVIDER[kind] in _AVAILABLE_PROVIDERS:
            yield kind
    yield 'CPU'


def resolve_job_device(use_gpu):
    global _FALLBACK_CPU
    device = resolve_device(use_gpu)
    if device == 'CPU' and use_gpu:
        # A provider declined the bundled probe graph, but a custom graph may
        # be supported. Try it with real placement verification in logits().
        device = next((k for k in _ORT_ORDER if _PROVIDER[k] in _AVAILABLE_PROVIDERS), device)
        if device == 'CPU':
            reason = 'no eligible GPU inference backend'
            if reason not in _FALLBACK_REASONS:
                _FALLBACK_REASONS.append(reason)
            _FALLBACK_CPU = True
    return device


def _memory_refusal(path, shape, device, threads, *, reclaim=True):
    if device == 'TORCH_MPS':
        from core.resource_policy import probe_memory, ram_ceiling
        n, hp, wp = shape
        # The staged planes already exist. Admit the extra mmap plus the GPU
        # working set in RAM. A live reading already reflects a started worker.
        required = estimate_inference_bytes(*shape, device) + n * hp * wp * 4
        startup = native_startup_bytes(device)
        if _JOB_RAM_LIMIT is not None and required + startup > _JOB_RAM_LIMIT:
            return f'Host memory plan budget {_JOB_RAM_LIMIT} bytes, requires {required + startup}'
        if _WORKER is None:
            required += startup
        snapshot = probe_memory()
        if not snapshot.ram_known:
            return 'available host memory could not be read'
        budget = ram_ceiling(snapshot)
        if required > budget:
            return f'Host memory budget {budget} bytes, requires {required}'
        return None
    from core.resource_policy import vram_budget
    if device not in VRAM_GATED_KINDS:
        return None
    if device in _PROVIDER:
        try:
            session = _SESSIONS.get(_session_key(path, shape, device, threads))
        except OSError:
            session = None
        if session is not None and session.actual_device == device:
            return None
        del session
    n, hp, wp = shape
    required = estimate_inference_bytes(*shape, device) + n * hp * wp * 4 * 6
    snapshot = _vram_snapshot(device)
    if reclaim and (_vram_unknown(snapshot) or required > vram_budget(snapshot)) and (_SESSIONS or _WORKER):
        _clear_sessions()
        finish_inference_job()
        snapshot = _vram_snapshot(device)
    _record_admission(device, snapshot)
    if _vram_unknown(snapshot):
        return 'free GPU memory could not be read'
    budget = vram_budget(snapshot)
    if _JOB_VRAM_LIMIT is not None:
        budget = min(budget, _JOB_VRAM_LIMIT)
    if required > budget:
        return f'GPU memory budget {budget} bytes, requires {required}'
    return None


def _run_ort(planes, path, device, threads, should_cancel, profile):
    from fusion_methods.neural_common import cancel
    from core.registration import RegistrationCancelled
    session = None
    try:
        with profile.stage('session.acquire'):
            session = _session(path, planes.shape[1:], device, threads)
        cancel(should_cancel)
        with profile.stage('session.run'):
            result = session.run(['logits'], {'input': planes})[0]
    except RegistrationCancelled:
        raise
    except Exception:
        _log_session_failure(planes.shape[1:], session, device)
        raise
    return result, 'CPU' if device == 'CPU' else session.actual_device


def _run_native(planes, path, device, should_cancel, profile):
    global _WORKER, _WORKER_FAILURE, _JOB_PACK
    from fusion_methods.inference_client import Worker, WorkerLost, WorkerError
    from core.inference_pack import pack_root
    if _WORKER_FAILURE:
        raise WorkerLost(_WORKER_FAILURE)
    if _JOB_PACK is None:
        if _PACK is None:
            refresh_accelerator_pack()
        _JOB_PACK = _PACK
    if not _JOB_PACK[1]:
        raise WorkerLost('Optional accelerator pack is unavailable')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        with profile.stage('session.acquire'):
            if _WORKER is None:
                if device in VRAM_GATED_KINDS:
                    # Leave physical/commit headroom for importing CUDA libraries.
                    from core.resource_policy import require_memory
                    require_memory(native_startup_bytes(device) + planes.nbytes, None, _JOB_RAM_LIMIT)
                _WORKER = Worker(_JOB_PACK[1], should_cancel)
                # Worker startup reduces whichever memory pool admits the model.
                refusal = _memory_refusal(path, planes.shape[1:], device, 1, reclaim=False)
                if refusal:
                    raise WorkerError(refusal)
        if device not in _WORKER.backends:
            raise WorkerError('Engine unavailable on this GPU/driver')
        with profile.stage('session.run'):
            result = _WORKER.run(device, path, planes, pack_root() / 'engine-cache', should_cancel)
        return result, device
    except (WorkerLost, OSError) as exc:
        _WORKER_FAILURE = str(exc)
        finish_inference_job()
        raise WorkerLost(_WORKER_FAILURE) from None


def _record_fallback(device, shape, reason):
    message = f'{device} {tuple(shape)}: {reason}'
    if message not in _FALLBACK_REASONS:
        _FALLBACK_REASONS.append(message)
        _LOG.warning('Neural fallback: %s', message)


def logits(planes, path, device, threads, should_cancel=None):
    from fusion_methods.neural_common import cancel
    from core.registration import RegistrationCancelled
    cancel(should_cancel)
    h, w = planes.shape[-2:]
    ph, pw = (-h) % 64, (-w) % 64
    if ph or pw:
        mode = 'reflect' if ph < h and pw < w else 'edge'
        planes = np.pad(planes, ((0, 0), (0, 0), (0, ph), (0, pw)), mode=mode)
    planes = np.ascontiguousarray(planes, dtype=np.float32)
    shape = planes.shape[1:]
    global _ACTUAL_DEVICE, _JOB_ACCELERATOR, _FALLBACK_CPU, _ACTUAL_CHECKPOINT
    requested = device
    model = os.path.abspath(path)
    profile = _profile()
    placed_on_cpu = None
    for device in candidate_devices(requested):
        cancel(should_cancel)
        key = (model, shape, device)
        if key in _FAILED_SHAPES:
            continue
        if device == 'CPU' and placed_on_cpu is not None:
            result = placed_on_cpu
            break
        if placed_on_cpu is not None:
            # Trying another GPU must not retain an extra full stack of logits.
            placed_on_cpu = result = None
        if device == 'CPU' and requested != 'CPU':
            from core.resource_policy import require_memory
            require_memory(estimate_inference_bytes(*shape, device), None, _JOB_RAM_LIMIT)
        reason = _memory_refusal(path, shape, device, threads)
        if reason is None:
            try:
                if device in _NATIVE_ORDER:
                    result, actual = _run_native(planes, model, device, should_cancel, profile)
                else:
                    result, actual = _run_ort(planes, path, device, threads, should_cancel, profile)
                if actual == device:
                    break
                # Keep usable CPU-placed logits in case another GPU cannot run.
                placed_on_cpu = result
                reason = 'all graph nodes were placed on CPU'
            except (RegistrationCancelled, FileNotFoundError, PermissionError):
                raise
            except Exception as exc:
                if device == 'CPU':
                    raise
                from fusion_methods.inference_client import WorkerError
                reason = str(exc) if isinstance(exc, WorkerError) else f'inference failed ({type(exc).__name__})'
                _LOG.warning('%s inference failed for %s %s\n%s', device, os.path.basename(path),
                             shape, traceback.format_exc())
        cancel(should_cancel)
        _record_fallback(device, shape, reason)
        _FAILED_SHAPES[key] = reason
        _clear_sessions()
        if device in _NATIVE_ORDER and _WORKER is not None:
            # Release native allocations before admission on another engine.
            finish_inference_job()
    _FALLBACK_CPU = _FALLBACK_CPU or (requested != 'CPU' and device == 'CPU')
    if device not in _JOB_DEVICES:
        _JOB_DEVICES.append(device)
    _ACTUAL_DEVICE = device
    if is_accelerator(device):
        _JOB_ACCELERATOR = device
    _ACTUAL_CHECKPOINT = os.path.basename(model)
    if profile.enabled:
        profile.session_run((_ACTUAL_CHECKPOINT, tuple(int(v) for v in shape), device,
                             int(threads or 1)))
        profile.add('threads', 'session_intra_op', max(1, int(threads or 1)))
        profile.add('render', 'actual_provider', _PROVIDER.get(device, device))
    _LOG.debug('Neural inference checkpoint=%s provider=%s adapter=%s shape=%s output=%s fallback=%s',
               _ACTUAL_CHECKPOINT, _PROVIDER.get(device, device), display_name(device), shape,
               result.shape, _FALLBACK_CPU)
    cancel(should_cancel)
    return result[..., :h, :w]


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=1, keepdims=True)
    return x


def _resize_antialias(planes, scale, should_cancel=None):
    from fusion_methods.neural_common import cancel
    # PyTorch bilinear antialias uses a triangle filter widened for downsampling.
    result = planes
    for axis in (2, 3):
        cancel(should_cancel)
        old = result.shape[axis]
        new = int(old * scale)
        if new < 1:
            raise ValueError('Input is too small for the inference scale')
        src = np.moveaxis(result, axis, -1)
        dst = np.empty(src.shape[:-1] + (new,), dtype=np.float32)
        for i in range(new):
            cancel(should_cancel)
            center = (i + 0.5) / scale - 0.5
            radius = max(1.0, 1.0 / scale)
            first = max(0, int(np.floor(center - radius)) + 1)
            last = min(old, int(np.ceil(center + radius)))
            weights = np.maximum(0, 1 - np.abs(np.arange(first, last) - center) / radius).astype(np.float32)
            weights /= weights.sum()
            dst[..., i] = np.sum(src[..., first:last] * weights, axis=-1)
        result = np.moveaxis(dst, -1, axis)
    return result


def probabilities(planes, path, device, threads, sigma, cancel=None, *, dual_scale=True):
    from fusion_methods.neural_common import cancel as check
    profile = _profile()
    with profile.stage('logits.full'):
        p = logits(planes, path, device, threads, cancel)
    with profile.stage('softmax.full'):
        p = softmax(p)
    check(cancel)
    if dual_scale:
        h, w = planes.shape[-2:]
        with profile.stage('resize.0.75'):
            small = _resize_antialias(planes, 0.75, cancel)
        with profile.stage('logits.reduced'):
            q = logits(small, path, device, threads, cancel)
        del small
        with profile.stage('softmax.reduced'):
            q = softmax(q)
        with profile.stage('upsample_combine'):
            for n in range(p.shape[1]):
                check(cancel)
                p[0, n] += cv2.resize(q[0, n], (w, h), interpolation=cv2.INTER_LINEAR)
            del q
            p *= 0.5
    if sigma:
        with profile.stage('gaussian'):
            radius = int(3 * sigma + 0.5)
            for n in range(p.shape[1]):
                check(cancel)
                p[0, n] = cv2.GaussianBlur(p[0, n], (radius * 2 + 1,) * 2, sigma,
                                            borderType=cv2.BORDER_REPLICATE)
            check(cancel)
            p /= p.sum(axis=1, keepdims=True)
    check(cancel)
    return p
