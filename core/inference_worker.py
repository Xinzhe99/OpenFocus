"""Isolated native inference worker. The desktop never imports its GPU libraries."""
import hashlib
import io
import json
import math
import mmap
import os
from pathlib import Path
import platform
import stat
import sys
import warnings


ROOT = Path(__file__).resolve().parents[1]
# Only exact releases that passed ONNX/forward_logits parity have a Torch adapter.
# A renamed copy is equivalent. A replacement with the same name is not.
_TORCH_EXPORTS = {
    'f7ba388afcefb988f38707b349d5609b4a56770d81c8405d881e50f605b18a19': (
        'stackmff_final.pth', '8a530228077a4dc0476258f55dd6e00d0b00d04861b1177bdbe4767eb449fb7d'),
    '045c7db9052f2d31c78633ced5eb299e841b2974c63eebcfbbf36f4ad2610614': (
        'stackmff_v2_final.pth', 'f7f2ee36453d6073d62b89289133012b95116c07b208e76f5ba6d00a1f3f2cda'),
    '7b92bf31c984eac2b93d6000158b4802b7d62c23140d52e28a47676f58d5c4f7': (
        'stackmff_v3_final.pth', '39b2ea3738e8fea1676c379238d6be81a87988fc9952dbd6ac3825d5d7249f46'),
}


def _load_torch_model(digest, device, *, bounded_workspace=False):
    import torch

    from core.models.stackmff_v1.network import StackMFF
    from core.models.stackmff_v2.network import StackMFF_V2
    from core.models.stackmff_v3.model_io import load_checkpoint, load_variant_checkpoint
    from core.models.stackmff_v3.network import InterLayerAttention, IntraLayerWindowAttention

    if digest not in _TORCH_EXPORTS:
        raise ValueError('This ONNX graph has no verified native PyTorch adapter')
    name, expected = _TORCH_EXPORTS[digest]
    checkpoint = (ROOT / 'weights' / name).read_bytes()
    if hashlib.sha256(checkpoint).hexdigest() != expected:
        raise ValueError(f'Checkpoint does not match the selected ONNX export: {name}')
    # Pass the same verified bytes to the same strict loaders as the ONNX exporter.
    source = io.BytesIO(checkpoint)
    if name == 'stackmff_final.pth':
        model = load_variant_checkpoint(source, StackMFF, 'stackmff_v3backbone')
    elif name == 'stackmff_v2_final.pth':
        model = load_variant_checkpoint(source, StackMFF_V2, 'v2_v3backbone')
    else:
        model = load_checkpoint(source)
    if bounded_workspace:
        for module in model.modules():
            if isinstance(module, (IntraLayerWindowAttention, InterLayerAttention)):
                module.bounded_workspace = True
    return model.to(device=device, dtype=torch.float32).forward_logits


def _absolute_path(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('Worker paths must be absolute')
    return path


class InferenceWorker:
    @staticmethod
    def _region(request, size):
        if 'mapping' in request:
            import re
            name = request['mapping']
            if os.name != 'nt' or not re.fullmatch(r'Local\\OpenFocus-[0-9a-f]{32}', name):
                raise ValueError('Invalid Windows shared memory name')
            return mmap.mmap(-1, size, tagname=name, access=mmap.ACCESS_WRITE)
        with _absolute_path(request['tensor']).open('r+b') as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                raise ValueError('Shared tensor byte count differs from its float32 shape')
            return mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_WRITE)

    def __init__(self):
        is_apple = sys.platform == 'darwin' and platform.machine() == 'arm64'
        if is_apple:
            # The worker inherits this environment variable even under Python -I.
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
        import numpy
        import torch

        self.torch = torch
        self.numpy = numpy
        if is_apple:
            if not torch.backends.mps.is_available():
                raise RuntimeError('Apple Silicon MPS is unavailable')
            # A few operators register CPU fallbacks even with fallback disabled.
            # Treat their warnings as failures on every request in this process.
            torch.set_warn_always(True)
            warnings.filterwarnings('error', message=r'The operator .*MPS backend.*run on the CPU.*',
                                    category=UserWarning)
            self.torch_device = 'mps'
            self.device = 'Apple Silicon GPU'
            self.backends = ['TORCH_MPS']
        else:
            if (sys.platform not in ('win32', 'linux') or not torch.version.cuda
                    or torch.version.hip or not torch.cuda.is_available()):
                raise RuntimeError('NVIDIA CUDA is unavailable')
            torch.cuda.set_device(0)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.torch_device = 'cuda:0'
            self.device = torch.cuda.get_device_name(0)
            self.backends = ['TORCH_CUDA']
        # Device enumeration alone does not prove that kernels execute.
        if torch.ones(1, device=self.torch_device, dtype=torch.float32).add_(1).item() != 2:
            raise RuntimeError('GPU startup operation returned an incorrect result')
        self.trt = None
        if sys.platform == 'win32':
            try:
                from core.inference_tensorrt import TensorRT
                self.trt = TensorRT(torch)
                self.backends.insert(0, 'TENSORRT')
            except Exception as exc:
                print(f'TensorRT unavailable: {exc}', file=sys.stderr, flush=True)
        self.active_key = None
        self.sessions = {}

    def probe_models(self):
        """Execute the released checkpoints before accepting a Unix pack install."""
        # 128px exercises shifted attention at both feature scales. Multiple
        # layers also exercise inter-layer attention with the desktop's padding.
        shape = (1, 3, 128, 128)
        with self.torch.inference_mode():
            x = self.torch.zeros(shape, device=self.torch_device, dtype=self.torch.float32)
            for expected, (name, _) in _TORCH_EXPORTS.items():
                graph = (ROOT / 'weights' / name).with_suffix('.onnx')
                try:
                    digest = hashlib.sha256(graph.read_bytes()).hexdigest()
                    if digest != expected:
                        raise ValueError('ONNX graph does not match the verified release')
                    predict = _load_torch_model(digest, self.torch_device, bounded_workspace=True)
                    self._validate_logits(predict(x), x.device, shape)
                    del predict
                except Exception as exc:
                    raise RuntimeError(f'{graph.name}: {exc}') from exc

    def clear(self):
        self.sessions.clear()
        self.active_key = None
        if self.torch_device == 'mps':
            self.torch.mps.empty_cache()
        else:
            self.torch.cuda.empty_cache()

    def _select(self, backend, model, shape, cache):
        # Read through a single handle so replacement cannot mix a digest and graph.
        # Hash on every selection, including warm hits: users can overwrite imports.
        graph = model.read_bytes()
        digest = hashlib.sha256(graph).hexdigest()
        if backend == 'TENSORRT' and digest not in _TORCH_EXPORTS:
            from core.inference_tensorrt import resolve_weights
            graph, digest = resolve_weights(graph, digest, model.parent)
        key = (backend, digest)
        if self.active_key != key:
            self.clear()
            self.active_key = key
        scale = shape if backend == 'TENSORRT' else None
        if scale in self.sessions:
            predict = self.sessions.pop(scale)
        else:
            # Full and reduced scales alternate across tiles. Admission in the host
            # checks free VRAM before each call, including the second build.
            if len(self.sessions) == 2:
                del self.sessions[next(iter(self.sessions))]
                self.torch.cuda.empty_cache()
            if backend != 'TENSORRT':
                predict = _load_torch_model(digest, self.torch_device, bounded_workspace=True)
            else:
                verified = digest in _TORCH_EXPORTS
                load_reference = (lambda: _load_torch_model(digest, self.torch_device)) if verified else None
                predict = self.trt.load(graph, digest, shape, cache, reference=load_reference)
        self.sessions[scale] = predict
        return predict

    def run(self, request):
        if request.get('op') != 'run' or request.get('backend') not in self.backends:
            raise ValueError('Unsupported worker operation or unavailable backend')
        backend = request['backend']
        shape = tuple(request['shape'])
        if (len(shape) != 4 or shape[0] != 1 or
                any(type(n) is not int or n <= 0 for n in shape)):
            raise ValueError('Expected a positive [1, layers, height, width] shape')
        model = _absolute_path(request['model'])
        cache = _absolute_path(request['cache'])
        with self._region(request, 4 * math.prod(shape)) as region:
            predict = self._select(backend, model, shape, cache)
            with self.torch.inference_mode():
                self._predict_into(region, shape, predict)
        return {'ok': True, 'backend': backend}

    def _predict_into(self, region, shape, predict):
        # numpy.ndarray (unlike frombuffer) doesn't hold an exported memoryview.
        # The mmap remains open until the synchronous D2H copy has completed.
        shared = self.torch.from_numpy(self.numpy.ndarray(shape, self.numpy.float32, buffer=region))
        x = shared.to(self.torch_device)
        y = predict(x)
        self._validate_logits(y, x.device, shape)
        shared.copy_(y, non_blocking=False)

    def _validate_logits(self, y, device, shape):
        if y.device != device or y.dtype != self.torch.float32 or tuple(y.shape) != shape:
            raise ValueError(f'Expected float32 logits on {device} with the input shape; '
                             f'got {y.dtype} on {y.device} with shape {tuple(y.shape)}')
        # The reductions propagate NaN and cover both signs of infinity without
        # allocating a tensor-sized mask. Leave the shared input intact on failure.
        low, high = self.torch.aminmax(y)
        if not (math.isfinite(low.item()) and math.isfinite(high.item())):
            raise ValueError('GPU inference returned non-finite logits')


def main():
    # Preserve the protocol even when a native dependency writes to fd 1.
    with os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8') as output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

        def reply(payload):
            print(json.dumps(payload, ensure_ascii=True), file=output, flush=True)

        # -I intentionally removes the working directory. Add only the known app root.
        sys.path.append(str(ROOT))
        try:
            worker = InferenceWorker()
            if sys.argv[1:] == ['--probe']:
                worker.probe_models()
        except Exception as exc:
            reply({'error': str(exc)})
            return 1
        reply({'ready': True, 'backends': worker.backends, 'device': worker.device})
        for line in sys.stdin.buffer:
            try:
                result = worker.run(json.loads(line))
            except Exception as exc:
                result = {'error': str(exc) or type(exc).__name__}
            if 'error' in result:
                worker.clear()  # After the traceback releases any GPU tensors.
            reply(result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
