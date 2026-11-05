"""ONNX Runtime inference for the bundled dynamic StackMFF graphs."""
from collections import OrderedDict
import json
import logging
import os
import tempfile
import uuid
import threading

import cv2
import numpy as np

_LOG = logging.getLogger(__name__)
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


def start_inference_job():
    """Reset actual-device reporting and fallback for a new serialized job."""
    global _ACTUAL_DEVICE, _FALLBACK_CPU
    _ACTUAL_DEVICE = None
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


def resolve_device(use_gpu):
    if not use_gpu:
        return 'CPU'
    global _DEVICE, _DIRECTML_ADAPTER_INDEX, _DIRECTML_ADAPTER_NAME
    with _LOCK:
        if _DEVICE is None:
            import onnxruntime as ort
            from utils.weights import weights_path
            available = ort.get_available_providers()
            _DEVICE = 'CPU'
            for kind in ('CUDA', 'ROCM', 'DIRECTML', 'OPENVINO', 'COREML'):
                if _PROVIDER[kind] not in available:
                    continue
                if kind == 'DIRECTML':
                    from core.resource_policy import directml_adapters
                    candidates = directml_adapters()
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
                        session.run(['logits'], {'input': np.zeros((1, 2, 64, 64), dtype=np.float32)})
                        profile = session.end_profiling()
                        try:
                            with open(profile, encoding='utf-8') as stream:
                                events = json.load(stream)
                            if not any(event.get('args', {}).get('provider') == _PROVIDER[kind]
                                       for event in events):
                                raise RuntimeError(f'{kind} did not execute any model nodes')
                        finally:
                            os.remove(profile)
                        if kind == 'DIRECTML':
                            from core.resource_policy import set_directml_adapter
                            set_directml_adapter(adapter_index)
                        if kind in ('DIRECTML', 'CUDA', 'ROCM', 'OPENVINO'):
                            from core.resource_policy import probe_resources
                            if not probe_resources(kind).vram_known:
                                _LOG.warning('%s adapter %s has no reliable free-VRAM reading',
                                             kind, adapter_index)
                                continue
                        if kind == 'DIRECTML':
                            _DIRECTML_ADAPTER_INDEX, _DIRECTML_ADAPTER_NAME = adapter_index, adapter_name
                        _DEVICE = kind
                        break
                    except Exception:
                        _LOG.warning('%s provider probe failed for adapter %s', kind,
                                     adapter_index, exc_info=True)
                if _DEVICE != 'CPU':
                    break
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
    key = (path, stat.st_mtime_ns, stat.st_size, shape, device, int(threads or 1))
    if key not in _SESSIONS:
        # Accelerators retain one session so another graph cannot keep VRAM
        # alive across scales. CPU retains both scales without ORT's arena.
        limit = 2 if device == 'CPU' else 1
        while len(_SESSIONS) >= limit:
            _SESSIONS.popitem(last=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads or 1))
        options.enable_cpu_mem_arena = False
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
            raise RuntimeError(f'{device} provider did not initialize')
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
    global _ACTUAL_DEVICE, _FALLBACK_CPU
    if _FALLBACK_CPU:
        device = 'CPU'
    if device in ('DIRECTML', 'CUDA', 'ROCM', 'OPENVINO'):
        from core.resource_policy import probe_resources, vram_budget
        snapshot = probe_resources(device)
        n, hp, wp = shape
        required = estimate_inference_bytes(n, hp, wp, device) + n * hp * wp * 4 * 6
        if not snapshot.vram_known or required > vram_budget(snapshot):
            _LOG.warning('%s memory budget is unknown or insufficient; using CPU for this job', device)
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
    except Exception:
        if device == 'CPU':
            raise
        with _LOCK:
            _FALLBACK_CPU = True
        _LOG.warning('%s inference failed; falling back to CPU for this job', device, exc_info=True)
        session = None
        _SESSIONS.clear()
        from core.resource_policy import require_memory
        require_memory(estimate_inference_bytes(*shape, 'CPU'))
        session = _session(path, shape, 'CPU', threads)
        result = session.run(['logits'], {'input': planes})[0]
        device = 'CPU'
    _ACTUAL_DEVICE = device
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
