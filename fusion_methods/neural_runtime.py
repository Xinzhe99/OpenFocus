"""ONNX Runtime inference for the bundled dynamic StackMFF graphs."""
import atexit
from collections import OrderedDict
import json
import hashlib
import os
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
_DEVICE = None
_DIRECTML_ADAPTER_INDEX = 0
_DIRECTML_ADAPTER_NAME = ''
_LOCK = threading.RLock()
_ACTUAL_DEVICE = None
_FALLBACK_CPU = False
_ACTUAL_CHECKPOINT = None
_AVAILABLE_PROVIDERS = ()
_HARDWARE_ADAPTERS = ()
_PROBE_ERRORS = ()
# Verified accelerator that logits() will not admit because its free memory
# cannot be read; AI jobs then run on CPU.
_UNADMITTED = None
# Providers whose jobs logits() admits only against known free device memory.
VRAM_GATED_KINDS = ('DIRECTML', 'CUDA', 'ROCM', 'OPENVINO')


def device_diagnostics():
    """Cached hardware, provider capability and job execution; never probes."""
    return {'hardware': _HARDWARE_ADAPTERS, 'providers': _AVAILABLE_PROVIDERS,
            'selected': _DEVICE, 'adapter': _DIRECTML_ADAPTER_NAME,
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


def start_inference_job():
    """Reset actual-device reporting and fallback for a new serialized job."""
    global _ACTUAL_DEVICE, _FALLBACK_CPU, _ACTUAL_CHECKPOINT
    _ACTUAL_DEVICE = _ACTUAL_CHECKPOINT = None
    _FALLBACK_CPU = False


def actual_device():
    """Device actually used by the current job, or None before inference."""
    return _ACTUAL_DEVICE

def cached_device():
    """Return the latest verified device, or None before probing."""
    return _DEVICE



def is_accelerator(kind):
    return kind in ('DIRECTML', 'CUDA', 'ROCM', 'OPENVINO', 'COREML')


def display_name(kind):
    if kind == 'DIRECTML' and _DIRECTML_ADAPTER_NAME:
        return f'DirectML ({_DIRECTML_ADAPTER_NAME})'
    return {'DIRECTML': 'DirectML', 'CUDA': 'CUDA', 'ROCM': 'ROCm',
            'OPENVINO': 'OpenVINO GPU', 'COREML': 'CoreML', 'CPU': 'CPU',
            'N/A': 'N/A'}.get(kind, str(kind))


def gpu_toggle_state(kind, use_gpu, available):
    enabled = is_accelerator(kind) or (kind is None and bool(available))
    return enabled, bool(enabled and use_gpu)


def execution_status(neural, use_gpu, checkpoint, capability):
    """Return (device, state) without probing or claiming planned work ran.

    Capability is the completed background probe; actual placement belongs to
    the last inference of the selected checkpoint. Explicit CPU and classical
    methods always describe CPU mode, even after an earlier accelerated job.
    """
    if not neural or not use_gpu:
        return 'CPU', 'selected'
    checkpoint = os.path.basename(os.path.splitext(checkpoint or '')[0] + '.onnx')
    if (_ACTUAL_DEVICE is not None and _ACTUAL_CHECKPOINT == checkpoint
            and (_ACTUAL_DEVICE == 'CPU' or _ACTUAL_DEVICE == capability)):
        return _ACTUAL_DEVICE, 'actual'
    if capability is None:
        return None, 'pending'
    if is_accelerator(capability):
        return capability, 'ready'
    return 'CPU', 'selected'


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


def resolve_device(use_gpu):
    if not use_gpu:
        return 'CPU'
    global _DEVICE, _DIRECTML_ADAPTER_INDEX, _DIRECTML_ADAPTER_NAME
    global _AVAILABLE_PROVIDERS, _HARDWARE_ADAPTERS, _PROBE_ERRORS, _UNADMITTED
    with _LOCK:
        if _DEVICE is None:
            _UNADMITTED = None
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
            for kind in ('CUDA', 'ROCM', 'DIRECTML', 'OPENVINO', 'COREML'):
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
                        if kind == 'DIRECTML':
                            _DIRECTML_ADAPTER_INDEX, _DIRECTML_ADAPTER_NAME = adapter_index, adapter_name
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
                       adapter=_DIRECTML_ADAPTER_NAME or None,
                       adapter_index=_DIRECTML_ADAPTER_INDEX if device == 'DIRECTML' else None,
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
    return (int(n * pixels * 3 * 4 * (feature_base + 4))
            + int(pixels / 16 * 3 * 4 * (n * 64 + n * n))
            + n * pixels * 4 * 6 + 4 * 1024 * 1024)


def _session_key(path, shape, device, threads):
    path = os.path.abspath(os.path.splitext(path)[0] + '.onnx')
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
    path = os.path.abspath(os.path.splitext(path)[0] + '.onnx')
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
    global _ACTUAL_DEVICE, _FALLBACK_CPU, _ACTUAL_CHECKPOINT
    if _FALLBACK_CPU:
        device = 'CPU'
    cached = None
    if device in VRAM_GATED_KINDS:
        from core.resource_policy import vram_budget
        try:
            cached = _SESSIONS.get(_session_key(path, shape, device, threads))
        except OSError:
            cached = None
        # A completed session already owns its workspace; charging its full
        # estimate against the remaining free VRAM would force a false fallback.
        reusable = cached is not None and cached.actual_device == device
        n, hp, wp = shape
        required = estimate_inference_bytes(n, hp, wp, device) + n * hp * wp * 4 * 6

        def refused(snapshot):
            return _vram_unknown(snapshot) or required > vram_budget(snapshot)

        if not reusable:
            snapshot = _vram_snapshot(device)
            if refused(snapshot) and _SESSIONS:
                cached = None
                _clear_sessions()
                snapshot = _vram_snapshot(device)
            # What the status shows follows the probe this job acted on.
            _record_admission(device, snapshot)
            if refused(snapshot):
                known = not _vram_unknown(snapshot)
                _LOG.warning('%s checkpoint=%s shape=%s memory required=%s budget=%s known=%s; using CPU for this job',
                             device, os.path.basename(path), shape, required,
                             vram_budget(snapshot) if known else 0, known)
                _FALLBACK_CPU = True
                _clear_sessions()
                device = 'CPU'
    if _FALLBACK_CPU:
        from core.resource_policy import require_memory
        require_memory(estimate_inference_bytes(*shape, 'CPU'))
    session = None
    profile = _profile()
    device_failed = False
    try:
        with profile.stage('session.acquire'):
            session = _session(path, shape, device, threads)
        cancel(should_cancel)
        with profile.stage('session.run'):
            result = session.run(['logits'], {'input': planes})[0]
        if device != 'CPU':
            device = session.actual_device
            if device == 'CPU':
                # ORT can initialize a provider yet place this graph entirely
                # on CPU. Reuse its valid output and keep later scales on CPU.
                _FALLBACK_CPU = True
                _clear_sessions()
    except RegistrationCancelled:
        raise
    except Exception:
        if device == 'CPU':
            raise
        _LOG.warning('%s checkpoint=%s shape=%s adapter=%s inference failed; falling back to CPU for this job\n%s',
                     device, os.path.basename(path), shape, display_name(device), traceback.format_exc())
        device_failed = True
    if device_failed:
        session = None
        cached = None
        _clear_sessions()
        cancel(should_cancel)
        with _LOCK:
            _FALLBACK_CPU = True
        from core.resource_policy import require_memory
        require_memory(estimate_inference_bytes(*shape, 'CPU'))
        with profile.stage('session.acquire'):
            session = _session(path, shape, 'CPU', threads)
        cancel(should_cancel)
        with profile.stage('session.run'):
            result = session.run(['logits'], {'input': planes})[0]
        device = 'CPU'
    _ACTUAL_DEVICE = device
    _ACTUAL_CHECKPOINT = os.path.basename(os.path.splitext(path)[0] + '.onnx')
    if profile.enabled:
        profile.session_run((_ACTUAL_CHECKPOINT, tuple(int(v) for v in shape), device,
                             int(threads or 1)))
        profile.add('threads', 'session_intra_op', max(1, int(threads or 1)))
        profile.add('render', 'actual_provider', _PROVIDER[device])
    _LOG.debug('Neural inference checkpoint=%s provider=%s adapter=%s shape=%s output=%s fallback=%s',
               _ACTUAL_CHECKPOINT, _PROVIDER[device], display_name(device), shape,
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


def probabilities(planes, path, device, threads, sigma, cancel=None):
    from fusion_methods.neural_common import cancel as check
    profile = _profile()
    h, w = planes.shape[-2:]
    with profile.stage('logits.full'):
        p = logits(planes, path, device, threads, cancel)
    with profile.stage('softmax.full'):
        p = softmax(p)
    check(cancel)
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
