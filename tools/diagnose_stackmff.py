"""Small, real CPU inference diagnostic for shipped StackMFF models.

Run from a checkout with ONNX Runtime and OpenCV installed::

    python tools/diagnose_stackmff.py
    python tools/diagnose_stackmff.py --torch
    python tools/diagnose_stackmff.py --weights-dir PATH_TO_BUNDLE_WEIGHTS --torch
    python tools/diagnose_stackmff.py --gpu

PyTorch and ONNX are development-only dependencies for --torch. No source
images are opened; the deterministic focus stack is two 64x80 colour frames.
With --gpu the public consumers request the verified accelerator, report actual
fallback/device, and alternate V1/V3 twice to exercise session isolation. The
logit/checkpoint parity comparison always uses the CPU execution provider.
Similarity of a particular image is not evidence that checkpoints are identical.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _difference(a, b):
    import numpy as np
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {'max_abs': float(difference.max()),
            'mean_abs': float(difference.mean()),
            'different_values': int(np.count_nonzero(difference))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--weights-dir', type=Path, default=ROOT / 'weights')
    parser.add_argument('--torch', action='store_true', help='also compare original PyTorch checkpoints')
    parser.add_argument('--gpu', action='store_true', help='request GPU for the public fusion consumers')
    args = parser.parse_args(argv)
    import cv2
    import numpy as np
    import onnxruntime as ort
    from core.multi_focus_fusion import MultiFocusFusion
    from core.resource_policy import directml_adapters
    from fusion_methods import neural_runtime
    from utils.logging_setup import image_result_details

    report = {'runtime_version': ort.__version__,
              'available_providers': ort.get_available_providers(),
              'hardware_dxgi': directml_adapters(),
              'logit_provider': 'CPUExecutionProvider', 'models': {}}
    stack = np.random.default_rng(7).random((1, 2, 64, 64), dtype=np.float32)
    outputs = {}
    states = {}
    if args.torch:
        import torch
        import onnx
        from tools.export_stackmff_onnx import _variants
        torch.set_num_threads(1)
        report['torch_version'] = torch.__version__
        report['torch_cuda_available'] = torch.cuda.is_available()
        report['torch_cuda_devices'] = [torch.cuda.get_device_name(i)
                                        for i in range(torch.cuda.device_count())]
        variants = _variants()
    for method in ('stackmff', 'stackmff_v2', 'stackmff_v3'):
        stem = {'stackmff': 'stackmff_final', 'stackmff_v2': 'stackmff_v2_final',
                'stackmff_v3': 'stackmff_v3_final'}[method]
        path = args.weights_dir / (stem + '.onnx')
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        outputs[method] = session.run(['logits'], {'input': stack})[0]
        details = {'onnx_sha256': _sha256(path),
                   'inputs': [(item.name, item.shape) for item in session.get_inputs()],
                   'outputs': [(item.name, item.shape) for item in session.get_outputs()],
                   'metadata': session.get_modelmeta().custom_metadata_map}
        if args.torch:
            details['checkpoint_sha256'] = _sha256(path.with_suffix('.pth'))
            checkpoint = torch.load(path.with_suffix('.pth'), map_location='cpu', weights_only=True)
            states[method] = {key.removeprefix('backbone.'): value
                              for key, value in checkpoint['model'].items()}
            details['checkpoint_args'] = checkpoint.get('args')
            graph = onnx.load(str(path))
            details['initializer_count'] = len(graph.graph.initializer)
            details['initializer_sha256'] = hashlib.sha256(
                b''.join(item.raw_data for item in graph.graph.initializer)).hexdigest()
            model = variants[method][1](path.with_suffix('.pth'))
            with torch.no_grad():
                reference = model.forward_logits(torch.from_numpy(stack)).numpy()
            details['onnx_vs_torch_logits'] = _difference(outputs[method], reference)
        report['models'][method] = details
    report['v1_vs_v3_logits'] = _difference(outputs['stackmff'], outputs['stackmff_v3'])
    if args.torch:
        report['v1_v3_parameters'] = {
            'total': len(states['stackmff']),
            'identical': sum(torch.equal(value, states['stackmff_v3'][key])
                             for key, value in states['stackmff'].items())}

    texture = np.random.default_rng(37).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(texture, (9, 9), 2)
    top = (np.indices((64, 80))[0] < 32)[..., None]
    frames = [np.where(top, texture, blurred), np.where(top, blurred, texture)]
    report['consumers'] = {}
    for tiled in (False, True):
        fused = {}
        identities = {}
        for method in ('stackmff', 'stackmffv3', 'stackmff', 'stackmffv3'):
            fusion = MultiFocusFusion(method, use_gpu=args.gpu, exposure_match=False,
                tile_enabled=tiled, tile_block_size=64, tile_overlap=16, tile_threshold=32)
            stem = 'stackmff_final' if method == 'stackmff' else 'stackmff_v3_final'
            result = fusion.fuse(frames, thread_count=1,
                model_path=str((args.weights_dir / (stem + '.onnx')).resolve()))
            if method in fused and not np.array_equal(result, fused[method]):
                raise RuntimeError(f'{method} changed after switching checkpoints')
            fused[method] = result
            identities[method] = {**image_result_details(result), 'device': fusion.get_info()['device']}
        report['consumers']['tiled' if tiled else 'whole'] = {
            'v1_vs_v3': _difference(fused['stackmff'], fused['stackmffv3']),
            'results': identities, 'repeat_equal': True}
    report['cached_device_diagnostics'] = neural_runtime.device_diagnostics()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
