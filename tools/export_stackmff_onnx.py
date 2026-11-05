"""Export the shipped StackMFF checkpoints to dynamic ONNX inference graphs.

Development-only: it needs PyTorch (to rebuild and re-export the networks) and
``onnx`` (for the graph checker and metadata), while the application runs the
resulting ``weights/*.onnx`` files through ONNX Runtime.

The exported graph is the pure per-layer-logit network
(``forward_logits``) with ``layers``, ``height`` and ``width`` dynamic:

* ``layers`` is whatever the stack contains at run time (never fixed here);
* ``height`` and ``width`` must be multiples of 64, because the network pads
  its input to that alignment internally and a traced export cannot branch on
  the values of dynamic shapes. ``fusion_methods.neural_common.stage_luma``
  performs that padding before every call.

Multi-scale averaging, Gaussian smoothing and the V2 depth normalisation stay
in the NumPy post-processing of the adapters, so one graph per model serves
every inference scale and both V3 output modes.

Usage (from the repository root)::

    python tools/export_stackmff_onnx.py
    python tools/export_stackmff_onnx.py --no-verify   # skip the CPU parity run
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The verification shapes deliberately differ from the export shape and from
# each other: a graph that only works at the shape it was traced with (the
# failure mode of the legacy tracer) cannot pass this list.
VERIFY_SHAPES = ((2, 64, 64), (3, 128, 192), (5, 192, 128), (8, 128, 128))
PARITY_TOLERANCE = 1e-3


def _require(module, hint):
    try:
        return __import__(module)
    except ImportError:
        raise SystemExit(f'{module} is required to export the ONNX models: {hint}')


def _source_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _variants():
    """Name -> (checkpoint filename, loader) for the three shipped models."""
    from core.models.stackmff_v1.network import StackMFF
    from core.models.stackmff_v2.network import StackMFF_V2
    from core.models.stackmff_v3.model_io import (load_checkpoint,
                                                  load_variant_checkpoint)
    return {
        'stackmff': ('stackmff_final.pth',
                     lambda path: load_variant_checkpoint(
                         str(path), StackMFF, 'stackmff_v3backbone')),
        'stackmff_v2': ('stackmff_v2_final.pth',
                        lambda path: load_variant_checkpoint(
                            str(path), StackMFF_V2, 'v2_v3backbone')),
        'stackmff_v3': ('stackmff_v3_final.pth',
                        lambda path: load_checkpoint(str(path))),
    }


def _export(model, destination, opset):
    import torch

    class Logits(torch.nn.Module):
        """Only the logit head is exported; fusion maths stays in NumPy."""

        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, stack):
            return self.network.forward_logits(stack)

    network = Logits(model.eval())
    example = torch.zeros(1, 2, 64, 64, dtype=torch.float32)
    torch.onnx.export(
        network, (example,), str(destination),
        input_names=['input'], output_names=['logits'],
        opset_version=opset, dynamo=True,
        dynamic_shapes={'stack': {
            1: torch.export.Dim('layers', min=2),
            2: torch.export.Dim('height', min=64),
            3: torch.export.Dim('width', min=64),
        }},
        external_data=False,
    )


def _annotate(path, algorithm, source_path, opset):
    import onnx
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    metadata = {
        'openfocus_algorithm': algorithm,
        'openfocus_export': 'tools/export_stackmff_onnx.py',
        'openfocus_opset': str(opset),
        'openfocus_source': source_path.name,
        'openfocus_source_sha256': _source_digest(source_path),
        'openfocus_input': 'float32 [1, layers, height, width], luma 0-1, '
                           'height and width multiples of 64',
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))
    return model


def _verify(path, model, onnxruntime):
    import numpy as np
    import torch

    session = onnxruntime.InferenceSession(
        str(path), providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != 'input':
        raise SystemExit(f'{path.name}: unexpected graph inputs '
                         f'{[(i.name, i.shape) for i in inputs]}')
    outputs = session.get_outputs()
    if len(outputs) != 1 or outputs[0].name != 'logits':
        raise SystemExit(f'{path.name}: unexpected graph outputs '
                         f'{[(o.name, o.shape) for o in outputs]}')
    for layers, height, width in VERIFY_SHAPES:
        rng = np.random.default_rng(layers * 1000 + height)
        stack = rng.random((1, layers, height, width), dtype=np.float32)
        with torch.no_grad():
            reference = model.forward_logits(
                torch.from_numpy(stack)).numpy()
        actual = session.run(['logits'], {'input': stack})[0]
        if actual.shape != reference.shape:
            raise SystemExit(f'{path.name}: shape {actual.shape} != '
                             f'{reference.shape} at N={layers} '
                             f'{height}x{width}')
        error = float(np.abs(actual - reference).max())
        if not np.isfinite(error) or error > PARITY_TOLERANCE:
            raise SystemExit(f'{path.name}: max |ONNX - torch| = {error:.2e} '
                             f'at N={layers} {height}x{width}')
        print(f'  N={layers:<3} {height}x{width:<4} '
              f'max |ONNX - torch| = {error:.1e}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-dir', default=str(ROOT / 'weights'),
                        help='destination directory (default: weights/)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset (default: 17, the oldest opset the '
                             'dynamo exporter and DirectML both accept)')
    parser.add_argument('--verify', dest='verify', action='store_true',
                        default=True,
                        help='run the CPU parity check (default)')
    parser.add_argument('--no-verify', dest='verify', action='store_false')
    args = parser.parse_args(argv)

    torch = _require('torch', 'pip install torch')
    print(f'torch {torch.__version__}')
    onnxruntime = _require('onnxruntime', 'pip install onnxruntime') \
        if args.verify else None
    if sys.version_info < (3, 10):
        raise SystemExit('Python 3.10+ is required')

    sys.path.insert(0, str(ROOT))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for algorithm, (checkpoint, load) in _variants().items():
        source = ROOT / 'weights' / checkpoint
        if not source.is_file():
            raise SystemExit(f'missing checkpoint {source}')
        destination = out_dir / (source.stem + '.onnx')
        print(f'{algorithm}: {source.name} -> {destination.name}')
        model = load(source)
        _export(model, destination, args.opset)
        _annotate(destination, algorithm, source, args.opset)
        if onnxruntime is not None:
            _verify(destination, model, onnxruntime)
        print(f'  {destination.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
