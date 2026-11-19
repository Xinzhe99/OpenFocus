"""Isolated NVIDIA inference worker. The desktop never imports its GPU libraries."""
import hashlib
import io
import json
import math
import mmap
import os
from pathlib import Path
import stat
import sys


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


def _load_torch_model(digest):
    from core.models.stackmff_v1.network import StackMFF
    from core.models.stackmff_v2.network import StackMFF_V2
    from core.models.stackmff_v3.model_io import load_checkpoint, load_variant_checkpoint

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
    return model.to('cuda:0').eval().forward_logits


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
        import numpy
        import torch

        self.torch = torch
        self.numpy = numpy
        if not torch.version.cuda or torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('NVIDIA CUDA is unavailable')
        torch.cuda.set_device(0)
        # Driver enumeration alone does not prove the wheel has usable kernels.
        torch.ones(1, device='cuda:0').add_(1).item()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.cuda.get_device_name(0)
        self.backends = ['TORCH_CUDA']
        self.trt = None
        try:
            from core.inference_tensorrt import TensorRT
            self.trt = TensorRT(torch)
            self.backends.insert(0, 'TENSORRT')
        except Exception as exc:
            print(f'TensorRT unavailable: {exc}', file=sys.stderr, flush=True)
        self.active_key = None
        self.sessions = {}

    def clear(self):
        self.sessions.clear()
        self.active_key = None
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
            if backend == 'TORCH_CUDA':
                predict = _load_torch_model(digest)
            else:
                verified = digest in _TORCH_EXPORTS
                load_reference = (lambda: _load_torch_model(digest)) if verified else None
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
        x = shared.to('cuda:0')
        y = predict(x)
        if y.device != x.device or y.dtype != x.dtype or tuple(y.shape) != shape:
            raise ValueError('GPU inference did not return float32 logits with the input shape')
        # The reductions propagate NaN and cover both signs of infinity without
        # allocating a tensor-sized mask. Leave the shared input intact on failure.
        low, high = self.torch.aminmax(y)
        if not (math.isfinite(low.item()) and math.isfinite(high.item())):
            raise ValueError('GPU inference returned non-finite logits')
        shared.copy_(y, non_blocking=False)


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
