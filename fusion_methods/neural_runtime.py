"""ONNX Runtime inference for the bundled dynamic StackMFF graphs."""
from collections import OrderedDict
import json
import hashlib
import os
import tempfile
import uuid
import threading

import cv2
import numpy as np

from utils.logging_setup import get_logger, log_action

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


def device_diagnostics():
    """Cached hardware, provider capability and job execution; never probes."""
    return {'hardware': _HARDWARE_ADAPTERS, 'providers': _AVAILABLE_PROVIDERS,
            'selected': _DEVICE, 'adapter': _DIRECTML_ADAPTER_NAME,
            'actual': _ACTUAL_DEVICE, 'checkpoint': _ACTUAL_CHECKPOINT,
            'fallback_cpu': _FALLBACK_CPU, 'probe_errors': _PROBE_ERRORS}


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
                    _LOG.warning('Could not remove failed inference profile', exc_info=True)
            raise
        if self._profiling:
            self.actual_device = self._finish_profile(True)
        return result


def resolve_device(use_gpu):
    if not use_gpu:
        return 'CPU'
    global _DEVICE, _DIRECTML_ADAPTER_INDEX, _DIRECTML_ADAPTER_NAME
    global _AVAILABLE_PROVIDERS, _HARDWARE_ADAPTERS, _PROBE_ERRORS
    with _LOCK:
        if _DEVICE is None:
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


def _session(path, shape, device, threads):
    import onnxruntime as ort
    global _SESSIONS
    path = os.path.abspath(os.path.splitext(path)[0] + '.onnx')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'ONNX model not found: {path}')
    stat = os.stat(path)
    adapter = _DIRECTML_ADAPTER_INDEX if device == 'DIRECTML' else 0
    key = (path, stat.st_mtime_ns, stat.st_size, shape, device, adapter, int(threads or 1))
    if key not in _SESSIONS:
        # Accelerators retain one session so another graph cannot keep VRAM
        # alive across scales. CPU retains both scales without ORT's arena.
        limit = 2 if device == 'CPU' else 1
        while len(_SESSIONS) >= limit:
            _SESSIONS.popitem(last=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads or 1))
        options.enable_cpu_mem_arena = False
        if device != 'CPU':
            options.enable_profiling = True
            options.profile_file_prefix = os.path.join(
                tempfile.gettempdir(), 'openfocus-ort-' + uuid.uuid4().hex)
        # DirectML needs concrete dimensions for attention MatMul.
        for name, value in zip(('layers', 'height', 'width'), shape):
            options.add_free_dimension_override_by_name(name, int(value))
        if device == 'DIRECTML':
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        provider = _provider(device, _DIRECTML_ADAPTER_INDEX if device == 'DIRECTML' else 0)
        session = ort.InferenceSession(path, sess_options=options,
                                       providers=[provider, 'CPUExecutionProvider']
                                       if device != 'CPU' else ['CPUExecutionProvider'])
        session.disable_fallback()
        if device != 'CPU' and _PROVIDER[device] not in session.get_providers():
            try:
                _ProfiledSession(session, device)._finish_profile(False)
            finally:
                raise RuntimeError(f'{device} provider did not initialize')
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(64 * 1024), b''):
                digest.update(block)
        log_action('neural_session', checkpoint=os.path.basename(path),
                   checkpoint_sha256=digest.hexdigest(), provider=_PROVIDER[device],
                   adapter=display_name(device), adapter_index=adapter,
                   shape=shape, threads=options.intra_op_num_threads)
        if device != 'CPU':
            session = _ProfiledSession(session, device)
        _SESSIONS[key] = session
    _SESSIONS.move_to_end(key)
    return _SESSIONS[key]


def logits(planes, path, device, threads, should_cancel=None):
    from fusion_methods.neural_common import cancel
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
    if device in ('DIRECTML', 'CUDA', 'ROCM', 'OPENVINO'):
        from core.resource_policy import probe_resources, vram_budget
        snapshot = probe_resources(device)
        n, hp, wp = shape
        required = estimate_inference_bytes(n, hp, wp, device) + n * hp * wp * 4 * 6
        if not snapshot.vram_known or required > vram_budget(snapshot):
            _LOG.warning('%s checkpoint=%s shape=%s memory required=%s budget=%s known=%s; using CPU for this job',
                         device, os.path.basename(path), shape, required,
                         vram_budget(snapshot), snapshot.vram_known)
            _FALLBACK_CPU = True
            _SESSIONS.clear()
            device = 'CPU'
    if _FALLBACK_CPU:
        from core.resource_policy import require_memory
        require_memory(estimate_inference_bytes(*shape, 'CPU'))
    session = None
    try:
        session = _session(path, shape, device, threads)
        result = session.run(['logits'], {'input': planes})[0]
        if device != 'CPU':
            device = session.actual_device
            if device == 'CPU':
                # ORT can initialize a provider yet place this graph entirely
                # on CPU. Reuse its valid output and keep later scales on CPU.
                _FALLBACK_CPU = True
                _SESSIONS.clear()
    except Exception:
        if device == 'CPU':
            raise
        with _LOCK:
            _FALLBACK_CPU = True
        _LOG.warning('%s checkpoint=%s shape=%s adapter=%s inference failed; falling back to CPU for this job',
                     device, os.path.basename(path), shape, display_name(device), exc_info=True)
        session = None
        _SESSIONS.clear()
        from core.resource_policy import require_memory
        require_memory(estimate_inference_bytes(*shape, 'CPU'))
        session = _session(path, shape, 'CPU', threads)
        result = session.run(['logits'], {'input': planes})[0]
        device = 'CPU'
    _ACTUAL_DEVICE = device
    _ACTUAL_CHECKPOINT = os.path.basename(os.path.splitext(path)[0] + '.onnx')
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


def _resize_antialias(planes, scale):
    # PyTorch bilinear antialias uses a triangle filter widened for downsampling.
    result = planes
    for axis in (2, 3):
        old = result.shape[axis]
        new = int(old * scale)
        if new < 1:
            raise ValueError('Input is too small for the inference scale')
        src = np.moveaxis(result, axis, -1)
        dst = np.empty(src.shape[:-1] + (new,), dtype=np.float32)
        for i in range(new):
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
    h, w = planes.shape[-2:]
    p = softmax(logits(planes, path, device, threads, cancel))
    check(cancel)
    small = _resize_antialias(planes, 0.75)
    q = softmax(logits(small, path, device, threads, cancel))
    for n in range(p.shape[1]):
        p[0, n] += cv2.resize(q[0, n], (w, h), interpolation=cv2.INTER_LINEAR)
    p *= 0.5
    if sigma:
        radius = int(3 * sigma + 0.5)
        for n in range(p.shape[1]):
            p[0, n] = cv2.GaussianBlur(p[0, n], (radius * 2 + 1,) * 2, sigma,
                                        borderType=cv2.BORDER_REPLICATE)
        p /= p.sum(axis=1, keepdims=True)
    check(cancel)
    return p
