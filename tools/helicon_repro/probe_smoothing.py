"""§7.1/7.2: map Smoothing (-sp) and DepthMapFeathering (-dmf) to the depth-map
transition profile, using a stack with one perfectly known depth step.

Geometry: left half sharp in f0, right half sharp in f1 (same texture class,
equal variance) -> the true depth is a step at x=EDGE. The exported dmap's
10-90% transition width measures how each control smooths the depth map, and
the 50% point's offset measures any bias.
"""
import sys, os, subprocess, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from probe_measure import ROOT, OUT, RES, HF

rng = np.random.default_rng(2026)
H, W, EDGE = 240, 480, 240

def box(a, r):
    if r <= 0: return a.astype(np.float64)
    k = 2*r+1
    p = np.pad(a, ((r,r),(0,0)), mode='edge'); c = np.cumsum(p, 0)
    a = (c[k-1:] - np.concatenate([np.zeros((1,)+a.shape[1:]), c[:-k]]))/k
    p = np.pad(a, ((0,0),(r,r)), mode='edge'); c = np.cumsum(p, 1)
    return (c[:, k-1:] - np.concatenate([np.zeros((a.shape[0],1)), c[:, :-k]], 1))/k

def band_var(g, x0, x1, r=5):
    b = g[:, x0:x1]
    return float(box((b - box(b, r))**2, r).mean())

def build():
    """f0: texture left / flat right ; f1: flat left / texture right.
    Both textures same sigma -> depth is a clean step at EDGE."""
    s = 26.0
    texL = 128 + rng.normal(0, s, (H, EDGE)); flatL = 128 + rng.normal(0, 1.5, (H, EDGE))
    flatR = 128 + rng.normal(0, 1.5, (H, W-EDGE)); texR = 128 + rng.normal(0, s, (H, W-EDGE))
    fA = np.concatenate([texL, flatR], axis=1)
    fB = np.concatenate([flatL, texR], axis=1)
    d = os.path.join(ROOT, 'probeSmooth'); os.makedirs(d, exist_ok=True)
    for i, f in enumerate([fA, fB]):
        Image.fromarray(np.repeat(np.clip(f,0,255)[...,None].astype(np.uint8), 3, axis=2)
                        ).save(os.path.join(d, 'sm_%d.tif' % i))
    a = np.asarray(Image.open(os.path.join(d,'sm_0.tif')).convert('RGB'), np.float64).mean(2)
    b = np.asarray(Image.open(os.path.join(d,'sm_1.tif')).convert('RGB'), np.float64).mean(2)
    vA_l, vA_r = band_var(a, 20, EDGE-20), band_var(a, EDGE+20, W-20)
    vB_l, vB_r = band_var(b, 20, EDGE-20), band_var(b, EDGE+20, W-20)
    return d, (vA_l, vA_r, vB_l, vB_r)

def render(d, tag, sp, dmf=None):
    before = set(glob.glob(OUT + '\\*_dmap.tiff'))
    out = OUT + '\\' + tag + '.tif'
    if os.path.exists(out): os.remove(out)
    args = [HF, '-silent', '-save:' + out, '-mp:1', '-rp:5', '-sp:%d' % sp,
            '-va:0', '-ha:0', '-ra:0', '-ma:0', '-j:100', '-dmap']
    if dmf is not None: args.insert(-4, '-dmf:%d' % dmf)
    subprocess.run(args + [d], capture_output=True, timeout=300)
    t0 = time.time()
    while time.time() - t0 < 20:
        new = set(glob.glob(OUT + '\\*_dmap.tiff')) - before
        if new:
            g = sorted(new)[0]; s1 = os.path.getsize(g); time.sleep(0.3)
            if s1 == os.path.getsize(g):
                dm = os.path.join(RES, 'dmap_' + tag + '.tif'); os.replace(g, dm); return dm
        time.sleep(0.3)
    return None

def profile(dm_path):
    """P(fB wins) per column (rows averaged), in dmap coords."""
    a = np.asarray(Image.open(dm_path), dtype=np.float64)
    if a.ndim == 3: a = a[..., 0]
    dh, dw = a.shape
    step = 65535.0
    idx = np.clip(1 - np.rint(a/step).astype(int), 0, 1)   # 1 = fB (right sharp)
    return idx.mean(axis=0), dw, EDGE - (W - dw)//2

def metrics(p, edge):
    seg = p[max(0,int(edge)-40):int(edge)+40]
    x = np.arange(max(0,int(edge)-40), max(0,int(edge)-40)+len(seg))
    lo, hi = np.percentile(seg, 10), np.percentile(seg, 90)
    if hi - lo < 1e-6: return 0.0, 0.0
    i10 = x[np.argmax(seg >= lo)] if (seg >= lo).any() else x[0]
    # 10->90 width via interpolation on the monotone-ish profile
    def at(level):
        for j in range(len(seg)-1):
            if (seg[j]-level)*(seg[j+1]-level) <= 0 and seg[j] != seg[j+1]:
                return x[j] + (level-seg[j])*(x[j+1]-x[j])/(seg[j+1]-seg[j])
        return None
    a10, a50, a90 = at(lo), at(0.5), at(hi)
    w = (a90-a10) if (a10 is not None and a90 is not None) else float('nan')
    off = (a50-edge) if a50 is not None else float('nan')
    return w, off

if __name__ == '__main__':
    d, v = build()
    print('measured band variances (luma): fA(L,R)=(%.1f, %.1f)  fB(L,R)=(%.1f, %.1f)' % v)
    print('\n%-6s %-6s %10s %10s' % ('sp', 'dmf', 'width10-90', 'offset50'))
    for sp in [0, 1, 2, 3, 5, 8]:
        dm = render(d, 'sm_sp%d' % sp, sp)
        if dm is None: print('%-6d %-6s RENDER FAIL' % (sp, '-')); continue
        p, dw, e = profile(dm)
        w, off = metrics(p, e)
        print('%-6d %-6s %10.1f %10.1f' % (sp, '-', w, off))
    for dmf in [0, 5, 10]:
        dm = render(d, 'sm_dmf%d' % dmf, 3, dmf=dmf)
        if dm is None: print('%-6d %-6s RENDER FAIL' % (3, dmf)); continue
        p, dw, e = profile(dm)
        w, off = metrics(p, e)
        print('%-6d %-6d %10.1f %10.1f' % (3, dmf, w, off))
    os.system('taskkill /F /IM _HeliconFocus.exe >nul 2>&1')
