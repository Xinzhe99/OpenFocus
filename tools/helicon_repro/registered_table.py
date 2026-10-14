"""Re-measure the 11-stack table with crop-offset registration (Helicon's crop
is asymmetric; naive center-crop costs ~25 levels of pure misalignment)."""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
from real_compare import load_rgb, DATA, OUT

def box(a, r):
    if r <= 0: return a.astype(np.float64)
    k = 2*r+1
    p = np.pad(a, ((r,r),(0,0)), mode='edge'); c = np.cumsum(p, 0)
    a = (c[k-1:] - np.concatenate([np.zeros((1,)+a.shape[1:]), c[:-k]]))/k
    p = np.pad(a, ((0,0),(r,r)), mode='edge'); c = np.cumsum(p, 1)
    return (c[:, k-1:] - np.concatenate([np.zeros((a.shape[0],1)), c[:, :-k]], 1))/k

def gauss(a, s):
    if s <= 0: return a.astype(np.float64)
    r = max(1, int(3*s)); k = np.exp(-0.5*(np.arange(-r,r+1)/s)**2); k/=k.sum()
    p0 = ((r,r),(0,0)) + ((0,0),)*(a.ndim-2)
    a = sum(k[i]*np.pad(a, p0, mode='edge')[i:i+a.shape[0]] for i in range(2*r+1))
    p1 = ((0,0),(r,r)) + ((0,0),)*(a.ndim-2)
    return sum(k[i]*np.pad(a, p1, mode='edge')[:, i:i+a.shape[1]] for i in range(2*r+1))

def var_energy(f, r, luma):
    if luma:
        g = f.mean(axis=2); m = box(g, r); return box((g-m)**2, r)
    e = np.zeros(f.shape[:2])
    for c in range(3):
        m = box(f[..., c], r); e += box((f[..., c]-m)**2, r)
    return e

def align(frames):
    from real_compare import estimate_translation_subpix
    gray = frames.mean(axis=3); ref = gray[0]
    out = [frames[0]]; sh = [(0.0, 0.0)]
    for k in range(1, len(frames)):
        dy, dx = estimate_translation_subpix(ref, gray[k])
        if abs(dy) < 0.05 and abs(dx) < 0.05:
            out.append(frames[k]); sh.append((0.0, 0.0))
        else:
            moved = []
            for c in range(3):
                a = frames[k, ..., c]
                y0, x0 = int(np.floor(dy)), int(np.floor(dx)); fy, fx = dy-y0, dx-x0
                t = lambda yy, xx: np.roll(np.roll(a, yy, 0), xx, 1)
                moved.append((1-fy)*(1-fx)*t(y0, x0) + fy*(1-fx)*t(y0+1, x0)
                             + (1-fy)*fx*t(y0, x0+1) + fy*fx*t(y0+1, x0+1))
            out.append(np.stack(moved, axis=2)); sh.append((dy, dx))
    return out

def make(frames, r, luma, base=True):
    al = align(frames)
    idx = np.stack([var_energy(f, r, luma) for f in al]).argmax(0)
    n = len(frames)
    d = idx.astype(float)
    if n > 2:
        d = gauss(box(d, 1), 1.2)
    sel = np.clip(np.rint(d), 0, n-1).astype(int)
    H, W = sel.shape
    S = np.stack([f[:H, :W] for f in al])
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    out = S[sel, yy, xx]
    if base:
        out = out + gauss(S[-1], 32.0) - gauss(out, 32.0)
    return out

def registered_mae(mine, ref, R=5, m=6):
    """min MAE over integer crop offsets (Helicon's crop is asymmetric).

    Overlap length is recomputed per offset: a fixed length silently drops
    every |offset| > m (shapes stop matching), which under-sweeps the range.
    (tests/test_helicon_regression.py::_metrics carries the same fix.)
    """
    best = (1e18, 0, 0)
    H0, W0 = mine.shape[:2]; H1, W1 = ref.shape[:2]
    for dy in range(-R, R+1):
        for dx in range(-R, R+1):
            y0a, y0b = max(dy, 0), max(-dy, 0)
            x0a, x0b = max(dx, 0), max(-dx, 0)
            h = min(H0 - y0a, H1 - y0b) - 2*m
            w = min(W0 - x0a, W1 - x0b) - 2*m
            if h <= 0 or w <= 0:
                continue
            a = mine[m+y0a:m+y0a+h, m+x0a:m+x0a+w]
            b = ref[m+y0b:m+y0b+h, m+x0b:m+x0b+w]
            e = float(np.abs(a-b).mean())
            if e < best[0]: best = (e, dy, dx)
    return best

STACKS = ['kitchen','balls','keyboard','bottles','bucket','fruits','metals',
          'plants','telephone','window','zeromotion']

print('%-11s %3s | %-22s | %-22s' % ('stack', 'n', 'r2 perchan (v1.39)', 'r5 luma (recommended)'))
print('%-11s %3s | %6s %6s %7s | %6s %6s %7s' % ('', '', 'naive', 'reg', 'off', 'naive', 'reg', 'off'))
rows = []
for stack in STACKS:
    files = sorted(glob.glob(os.path.join(DATA, stack, '*.jpg')))
    frames = np.stack([load_rgb(f) for f in files])
    ref = np.asarray(Image.open(OUT + r'\real_%s_r3_s3.tif' % stack).convert('RGB'), dtype=np.float64)
    out = []
    for r, luma in [(2, False), (5, True)]:
        mine = make(frames, r, luma)
        h = min(mine.shape[0], ref.shape[0]); w = min(mine.shape[1], ref.shape[1])
        a = mine[(mine.shape[0]-h)//2:(mine.shape[0]-h)//2+h, (mine.shape[1]-w)//2:(mine.shape[1]-w)//2+w]
        b = ref[(ref.shape[0]-h)//2:(ref.shape[0]-h)//2+h, (ref.shape[1]-w)//2:(ref.shape[1]-w)//2+w]
        naive = float(np.abs(a[6:-6,6:-6] - b[6:-6,6:-6]).mean())
        reg, dy, dx = registered_mae(mine, ref)
        out += [naive, reg, '%d,%d' % (dy, dx)]
    print('%-11s %3d | %6.2f %6.2f %7s | %6.2f %6.2f %7s' % (stack, len(files), *out))
    rows.append((stack, out[1], out[4]))
rows.sort(key=lambda t: t[2])
print('\nmedian registered MAE: v1.39=%.2f  recommended=%.2f  (8-bit levels, range ~255)'
      % (np.median([r[1] for r in rows]), np.median([r[2] for r in rows])))
