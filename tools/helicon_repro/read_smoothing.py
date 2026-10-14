"""Re-read the saved probe dmaps and measure the CONTINUOUS depth profile
(the smoothing kernel width), which the first pass of probe_smoothing.py
missed by rounding to a hard index first."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from probe_measure import RES

H, W, EDGE = 240, 480, 240

def continuous_profile(tag):
    p = os.path.join(RES, 'dmap_%s.tif' % tag)
    if not os.path.exists(p):
        return None
    a = np.asarray(Image.open(p), dtype=np.float64)
    if a.ndim == 3:
        a = a[..., 0]
    dh, dw = a.shape
    ox = (W - dw) // 2
    d = a / 65535.0 * 1.0          # n=2: depth in [0,1], 1 = first frame (fA)
    prof = d.mean(axis=0)          # column mean, in dmap coords
    edge = EDGE - ox
    return prof, edge

def widths(prof, edge, half=30):
    x = np.arange(int(edge) - half, int(edge) + half)
    seg = prof[int(edge) - half:int(edge) + half]
    if len(seg) < 4:
        return None
    # depth goes 1 (left, fA sharp) -> 0 (right, fB sharp): descending
    def at(level):
        for j in range(len(seg) - 1):
            a, b = seg[j], seg[j + 1]
            if (a - level) * (b - level) <= 0 and a != b:
                return x[j] + (level - a) * (x[j + 1] - x[j]) / (b - a)
        return None
    a10, a50, a90 = at(0.9), at(0.5), at(0.1)
    if None in (a10, a50, a90):
        return None
    return a90 - a10, a50 - edge

print('%-10s %12s %12s' % ('probe', 'width10-90', 'offset50'))
for sp in [0, 1, 2, 3, 5, 8]:
    r = continuous_profile('sm_sp%d' % sp)
    if r is None:
        print('sp=%-8d MISSING' % sp); continue
    prof, edge = r
    w = widths(prof, edge)
    if w is None:
        print('sp=%-8d no crossing' % sp); continue
    print('sp=%-8d %12.2f %12.2f' % (sp, w[0], w[1]))
for dmf in [0, 5, 10]:
    r = continuous_profile('sm_dmf%d' % dmf)
    if r is None:
        print('dmf=%-8d MISSING' % dmf); continue
    prof, edge = r
    w = widths(prof, edge)
    if w is None:
        print('dmf=%-8d no crossing' % dmf); continue
    print('dmf=%-7d %12.2f %12.2f' % (dmf, w[0], w[1]))

# also: how much does the continuous map deviate from a pure step far from the
# edge? (tests whether smoothing leaks across the whole image)
print('\nflatness away from edge (std of continuous depth, |x-edge|>60):')
for sp in [0, 3, 8]:
    r = continuous_profile('sm_sp%d' % sp)
    if r is None: continue
    prof, edge = r
    far = np.concatenate([prof[:int(edge)-60], prof[int(edge)+60:]])
    print('  sp=%d  std=%.4f  min=%.3f max=%.3f' % (sp, far.std(), far.min(), far.max()))
