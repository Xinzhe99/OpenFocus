"""Pin the measure window half-width r, free of any feathering offset.

Geometry: a straight vertical edge. fA = noise(30) left of the edge, noise(sa)
right of it. fB = noise(sb) everywhere, with Va < Vb < VD so the flat side is
decisively fB (no tie ambiguity).

A box window of half-width r centred at distance d beyond the edge covers the
high-variance side for frac = (r-d)/(2r):

    V_fA(d) = frac*VD + (1-frac)*Va ,  V_fB = Vb
    flip where  frac* = (Vb - Va)/(VD - Va)   =>   d* = r*(1 - 2*frac*) + off

Three probes with different (Va,Vb) give three d* values; least squares then
over-determines r and the constant feathering/alignment offset `off`.

Lossless PNG input and alignment disabled (-va/-ha/-ra/-ma:0) so the only
mechanism left is the measure window itself.
"""
import sys, os, subprocess, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from probe_measure import ROOT, OUT, RES, HF

rng = np.random.default_rng(4242)
H, W = 240, 480
EDGE = 240                       # x < EDGE is the high-variance side

def box(a, r):
    if r <= 0: return a.astype(np.float64)
    k = 2*r+1
    p = np.pad(a, ((r,r),(0,0)), mode='edge'); c = np.cumsum(p, 0)
    a = (c[k-1:] - np.concatenate([np.zeros((1,)+a.shape[1:]), c[:-k]]))/k
    p = np.pad(a, ((0,0),(r,r)), mode='edge'); c = np.cumsum(p, 1)
    return (c[:, k-1:] - np.concatenate([np.zeros((a.shape[0],1)), c[:, :-k]], 1))/k

def var_of(img, x0, x1, r=3):
    """measured box-variance (luma) of a column band of the saved image."""
    g = img[..., :3].mean(axis=2) if img.ndim == 3 else img
    band = g[:, x0:x1]
    return float(box((band - box(band, r))**2, r).mean())

def build(sa, sb, tag):
    hi = 128 + rng.normal(0, 30, (H, EDGE))
    loA = 128 + rng.normal(0, sa, (H, W - EDGE))
    fA = np.concatenate([hi, loA], axis=1)
    fB = 128 + rng.normal(0, sb, (H, W))
    d = os.path.join(ROOT, tag); os.makedirs(d, exist_ok=True)
    # lossless 8-bit RGB TIFF (HF's CLI decodes TIFF; PNG is not in its list)
    for i, f in enumerate([fA, fB]):
        rgb = np.repeat(np.clip(f, 0, 255)[..., None].astype(np.uint8), 3, axis=2)
        Image.fromarray(rgb).save(os.path.join(d, 'w_%d.tif' % i))
    a = np.asarray(Image.open(os.path.join(d, 'w_0.tif')).convert('RGB'), dtype=np.float64)
    b = np.asarray(Image.open(os.path.join(d, 'w_1.tif')).convert('RGB'), dtype=np.float64)
    return d, var_of(a, EDGE+20, EDGE+120), var_of(b, 40, W-40), var_of(a, 40, EDGE-20)

def render(d, tag):
    before = set(glob.glob(OUT + '\\*_dmap.tiff'))
    out = OUT + '\\' + tag + '.tif'
    if os.path.exists(out): os.remove(out)
    subprocess.run([HF, '-silent', '-save:' + out, '-mp:1', '-rp:3', '-sp:0',
                    '-va:0', '-ha:0', '-ra:0', '-ma:0', '-j:100', '-dmap', d],
                   capture_output=True, timeout=300)
    t0 = time.time()
    while time.time() - t0 < 15:
        new = set(glob.glob(OUT + '\\*_dmap.tiff')) - before
        if new:
            g = sorted(new)[0]; s1 = os.path.getsize(g); time.sleep(0.3)
            if s1 == os.path.getsize(g):
                dm = os.path.join(RES, 'dmap_' + tag + '.tif'); os.replace(g, dm); return dm
        time.sleep(0.3)
    return None

def flip_distance(dm_path, band=40):
    a = np.asarray(Image.open(dm_path), dtype=np.float64)
    if a.ndim == 3: a = a[..., 0]
    idx = np.clip(1 - np.rint(a/65535.0).astype(int), 0, 1)   # 0 = fA wins
    dh, dw = idx.shape
    ox = (W - dw)//2
    edge = EDGE - ox
    lo = max(0, (dh//2)-band); hi = min(dh, (dh//2)+band)
    p = (1 - idx[lo:hi, :]).mean(axis=0)
    p = np.convolve(p, np.ones(3)/3, mode='same')
    seg = p[int(edge):]
    below = np.where(seg < 0.5)[0]
    return (int(below[0]) if len(below) else None), p, edge

if __name__ == '__main__':
    rows = []
    for tag, sa, sb in [('probeWin_p1', 5.0, 6.0), ('probeWin_p2', 5.0, 12.0),
                        ('probeWin_p3', 5.0, 20.0)]:
        d, va, vb, vd = build(sa, sb, tag)
        dm = render(d, tag)
        b, prof, edge = flip_distance(dm)
        frac = (vb - va) / (vd - va)
        k = 1 - 2*frac
        rows.append((tag, b, k, va, vb, vd, frac))
        print('%-14s Va=%6.1f Vb=%6.1f VD=%6.1f frac*=%.4f k=%.4f d*=%s'
              % (tag, va, vb, vd, frac, k, b))
    good = [(b, k) for _, b, k, *_ in rows if b is not None]
    if len(good) >= 2:
        B = np.array([g[0] for g in good], float); K = np.array([g[1] for g in good], float)
        A = np.stack([K, np.ones_like(K)], axis=1)
        (r, off), *_ = np.linalg.lstsq(A, B, rcond=None)
        pred = A @ np.array([r, off])
        print('\n=> window half-width r = %.2f px  (box %dx%d)' % (r, 2*r+1, 2*r+1))
        print('=> constant offset = %+.2f px' % off)
        print('=> residuals (px): %s' % np.round(pred - B, 2).tolist())
        print('   (OpenFocus: kernel_size=11 -> radius 5)')
    for tag, _, _, *_ in rows:
        _, prof, edge = flip_distance(os.path.join(RES, 'dmap_' + tag + '.tif'))
        print('%s P(fA) d=-10..+14: %s' % (tag,
              [round(float(prof[int(edge)+d]), 2) for d in range(-10, 15)]))
