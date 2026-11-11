"""Visual A/B for the Issue #21 feather gate.

Renders hard | feather | GT crops where ground-truth error improves most.
For a real stack, frames use natural filename order and the crop shows the
largest hard/feather difference; that difference is not a quality judgment.

Usage:
    python tools/_proto_feather_visual.py            # synthetic demo
    python tools/_proto_feather_visual.py <dir>      # real stack (gate run)
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
import glob
import os
import tempfile
import numpy as np
import cv2
from fusion_methods.openfocus_a import openfocus_a_impl
from utils.sorting import natural_sort_key


def load_stack(path):
    files = glob.glob(path + '/*.png') + glob.glob(path + '/*.jpg') + glob.glob(path + '/*.tif') + glob.glob(path + '/*.tiff')
    files.sort(key=lambda f: (natural_sort_key(os.path.basename(f)), os.path.basename(f)))
    if not files:
        raise ValueError('no images found in ' + path)
    print('input order:', ', '.join(os.path.basename(f) for f in files))
    frames = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in files]
    for filename, frame in zip(files, frames):
        if frame is None:
            raise ValueError('could not decode image: ' + filename)
    return frames


def select_crop(plane, k=48):
    best = None
    h, w = plane.shape
    for y in range(0, h - k, 8):
        for x in range(0, w - k, 8):
            score = plane[y:y + k, x:x + k].sum()
            if best is None or score > best[0]:
                best = (score, y, x)
    if best is None:
        raise ValueError('image must be larger than 48 pixels in both dimensions')
    return best


def render(source):
    real = isinstance(source, list)
    if real:
        frames = [f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                  for f in source]
        # normalize 16-bit to the float32 0-255 domain the impl expects
        norm = []
        for f in frames:
            if f.dtype == np.uint16:
                norm.append(f.astype(np.float32) * (255.0 / 65535.0))
            else:
                norm.append(f.astype(np.float32))
        frames = norm
        gt8 = None
    else:
        from synthetic_benchmark import make_stack, SCENES
        h, w = source.shape[:2]
        depth = SCENES['default'](h, w)
        frames = [f.copy() for f in make_stack(source.astype(np.float64), depth, 32, noise=3.0)]
        gt8 = source.clip(0, 255).astype('uint8')

    out_hard = openfocus_a_impl([f.copy() for f in frames], thread_count=1)
    out_feather = openfocus_a_impl([f.copy() for f in frames], thread_count=1,
                                   blend_mode='feather')
    difference = np.abs(out_feather.astype(np.float32) - out_hard.astype(np.float32)).max(axis=2)
    if gt8 is None:
        plane = difference
        mode = 'absolute output difference (not a quality judgment)'
    else:
        plane = (np.abs(out_hard.astype(np.float32) - gt8.astype(np.float32)).max(axis=2)
                 - np.abs(out_feather.astype(np.float32) - gt8.astype(np.float32)).max(axis=2))
        mode = 'ground-truth error improvement'
    score_value, cy, cx = select_crop(plane)
    k = 48
    h, w = plane.shape
    if not np.any(difference):
        print('No output difference')
    pad = 10
    y0, x0 = max(0, cy - pad), max(0, cx - pad)
    y1, x1 = min(h, cy + k + pad), min(w, cx + k + pad)
    panels = [out_hard, out_feather] + ([gt8] if gt8 is not None else [])
    row = np.hstack([p[y0:y1, x0:x1] for p in panels])
    row = cv2.resize(row, (row.shape[1] * 4, row.shape[0] * 4),
                     interpolation=cv2.INTER_NEAREST)
    fd, temporary = tempfile.mkstemp(suffix='.png', dir='.')
    os.close(fd)
    try:
        if not cv2.imwrite(temporary, row):
            raise OSError('failed to save feather audit image')
        os.replace(temporary, '.tmp_feather_gate.png')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if gt8 is not None:
        from synthetic_benchmark import score
        print('PSNR hard %.2f feather %.2f' % (score(out_hard, gt8)[0],
                                               score(out_feather, gt8)[0]))
    print('%s crop at (%d, %d), score %.3f, saved .tmp_feather_gate.png' % (mode, cy, cx, score_value))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        stack = load_stack(sys.argv[1])
        assert stack, 'no images found in ' + sys.argv[1]
        print('loaded %d frames' % len(stack))
        render(stack)
    else:
        gt = cv2.imread('tests/golden/demo_guided_filter.png')
        if gt is None:
            raise ValueError('could not decode synthetic ground truth')
        gt = cv2.resize(gt, (gt.shape[1] // 2, gt.shape[0] // 2))
        render(gt)
