"""Soft weight blending prototype for Issue #21 (halos at depth transitions).

Hard argmax selection (current OpenFocus A) creates binary boundaries that
show as halos at crossings. Power-softmax weights w_i = r_i^g / sum_j r_j^g
blend all frames with per-pixel confidence; gamma controls hardness
(gamma -> inf recovers hard selection; gamma small -> uniform average).
Deterministic single-thread comparison against the hard baseline.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'tools')
import numpy as np
import cv2
from synthetic_benchmark import make_texture, make_stack, SCENES, score
from fusion_methods._openfocus_pixels import frame_codes
from fusion_methods.openfocus_a import _response
from fusion_methods.openfocus_ab import _check_cancel


def responses(frames, kernel_size=11):
    radius = max(1, int(kernel_size) // 2)
    out = []
    for f in frames:
        code = frame_codes(f, None, None)
        out.append(_response(code, radius, None, _check_cancel,
                             u8=f.dtype == np.uint8))
    return np.stack(out)


def fuse_hard(frames, resp):
    idx = resp.argmax(axis=0)
    stack = np.stack(frames, 0)
    return np.take_along_axis(stack, idx[None, :, :, None], axis=0)[0]


def fuse_soft(frames, resp, gamma):
    r = np.log(np.maximum(resp.astype(np.float64), 1e-12))
    r = r - r.max(axis=0, keepdims=True)
    w = np.exp(gamma * r)
    w /= w.sum(axis=0, keepdims=True)
    stack = np.stack(frames, 0).astype(np.float64)
    return np.einsum('fhw,fhwc->hwc', w, stack)


def bench(scene, noise, n, gamma):
    h, w = 768, 1024
    gt = make_texture(h, w)
    depth = SCENES[scene](h, w)
    frames = make_stack(gt, depth, n, noise=noise)
    resp = responses(frames)
    gt8 = gt.clip(0, 255).astype('uint8')
    res = {}
    res['hard'] = score(fuse_hard(frames, resp), gt8)
    for g in gamma:
        res[f'soft{g:g}'] = score(fuse_soft(frames, resp, g), gt8)
    return res


def transition_band_error(out, gt, band=6):
    """Mean |out-gt| inside a dilated band around GT depth discontinuities."""
    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY) if gt.ndim == 3 else gt
    edges = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    mask = (np.abs(edges) > np.percentile(np.abs(edges), 99)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((2 * band + 1, 2 * band + 1), np.uint8))
    err = np.abs(out.astype(np.float32) - gt.astype(np.float32)).max(axis=2)
    return err[mask > 0].mean() if mask.any() else float('nan')


if __name__ == '__main__':
    print(f"{'scene':>7} {'noise':>5} {'frames':>6} " + ' '.join(f"{k:>10}" for k in ['hard', 'soft1', 'soft2', 'soft4', 'soft8']) + '   (transition-band MAE, lower=better)')
    for scene in ("default", "small"):
        for noise in (0.0, 3.0):
            for n in (16, 32):
                h, w = 768, 1024
                gt = make_texture(h, w)
                depth = SCENES[scene](h, w)
                frames = make_stack(gt, depth, n, noise=noise)
                resp = responses(frames)
                gt8 = gt.clip(0, 255).astype('uint8')
                res = {'hard': fuse_hard(frames, resp)}
                for g in [1.0, 2.0, 4.0, 8.0]:
                    res[f'soft{g:g}'] = fuse_soft(frames, resp, g)
                cells = ' '.join(f"{transition_band_error(res[k], gt8):>10.3f}" for k in ['hard', 'soft1', 'soft2', 'soft4', 'soft8'])
                print(f"{scene:>7} {noise:>5.1f} {n:>6} {cells}")

    print(f"\n{'scene':>7} {'noise':>5} {'frames':>6} " + ' '.join(f"{k:>8}" for k in ['hard', 'soft1', 'soft2', 'soft4', 'soft8']) + '   (PSNR)')
    for scene in ("default", "small"):
        for noise in (0.0, 3.0):
            for n in (16, 32):
                r = bench(scene, noise, n, [1.0, 2.0, 4.0, 8.0])
                print(f"{scene:>7} {noise:>5.1f} {n:>6} " + ' '.join(f"{r[k][0]:>8.2f}" for k in ['hard', 'soft1', 'soft2', 'soft4', 'soft8']))
