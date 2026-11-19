"""Exact CPU kernels for GFG-FGF (Gradient-domain Guided Filter Fusion).

Uses bounded float64 prefix/sliding sums and fused single-pass neighborhood
operators to reproduce the reference OpenCV/NumPy arithmetic bit-for-bit while
eliminating redundant intermediate frame allocations and filter2D passes.
"""
import numpy as np

from fusion_methods._depth_kernel import njit


@njit(nogil=True)
def _reflect(i, n):
    """cv2.BORDER_REFLECT coordinate mapping."""
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i - 1
        if i >= n:
            i = 2 * n - 1 - i
    return i


@njit(nogil=True)
def _reflect101(i, n):
    """cv2.BORDER_REFLECT_101 (BORDER_DEFAULT) coordinate mapping."""
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        if i >= n:
            i = 2 * (n - 1) - i
    return i


@njit(nogil=True)
def extract_channel_u8(raw, ch, lut, out):
    """Extract normalized [0, 1] float32 single channel from uint8 frame."""
    h, w = out.shape
    if raw.ndim == 2:
        for y in range(h):
            for x in range(w):
                out[y, x] = lut[raw[y, x]]
    else:
        for y in range(h):
            for x in range(w):
                out[y, x] = lut[raw[y, x, ch]]


@njit(nogil=True)
def extract_channel_u16(raw, ch, out):
    """Extract normalized [0, 1] float32 single channel from uint16 frame."""
    h, w = out.shape
    scale = np.float32(255.0 / 65535.0)
    inv255 = np.float32(255.0)
    if raw.ndim == 2:
        for y in range(h):
            for x in range(w):
                out[y, x] = (np.float32(raw[y, x]) * scale) / inv255
    else:
        for y in range(h):
            for x in range(w):
                out[y, x] = (np.float32(raw[y, x, ch]) * scale) / inv255


@njit(nogil=True)
def extract_channel_f32(raw, ch, out):
    """Extract normalized [0, 1] float32 single channel from float frame."""
    h, w = out.shape
    inv255 = np.float32(255.0)
    if raw.ndim == 2:
        for y in range(h):
            for x in range(w):
                out[y, x] = np.float32(raw[y, x]) / inv255
    else:
        for y in range(h):
            for x in range(w):
                out[y, x] = np.float32(raw[y, x, ch]) / inv255


@njit(nogil=True)
def scharr_focus_score_exact(gray):
    """Interior mean(|Gx|^2 + |Gy|^2) using 3x3 Scharr-like operators."""
    h, w = gray.shape
    if h <= 2 or w <= 2:
        return np.float32(0.0)
    c3 = np.float32(3.0)
    c10 = np.float32(10.0)
    total = 0.0
    for y in range(1, h - 1):
        row_sum = 0.0
        for x in range(1, w - 1):
            a00 = gray[y - 1, x - 1]
            a01 = gray[y - 1, x]
            a02 = gray[y - 1, x + 1]
            a10 = gray[y, x - 1]
            a12 = gray[y, x + 1]
            a20 = gray[y + 1, x - 1]
            a21 = gray[y + 1, x]
            a22 = gray[y + 1, x + 1]
            gx = (-c3 * a00 - c10 * a01 - c3 * a02) + (c3 * a20 + c10 * a21 + c3 * a22)
            gy = (-c3 * a00 + c3 * a02) + (-c10 * a10 + c10 * a12) + (-c3 * a20 + c3 * a22)
            row_sum += float(gx * gx + gy * gy)
        total += row_sum
    count = (h - 2) * (w - 2)
    return np.float32(total / count)


@njit(nogil=True)
def gfg_gradient_exact(gray, threshold):
    """Exact bit-for-bit GFG four-neighborhood gradient (BORDER_REFLECT)."""
    h, w = gray.shape
    out = np.empty((h, w), np.float32)
    t1 = np.float32(1.0 / 16.0)
    t2 = np.float32(2.0 / 16.0)
    t4 = np.float32(4.0 / 16.0)
    half = np.float32(0.5)
    zero = np.float32(0.0)
    thr = np.float32(threshold)

    xprev = np.empty(w, np.int32)
    xnext = np.empty(w, np.int32)
    for x in range(w):
        xprev[x] = _reflect(x - 1, w)
        xnext[x] = _reflect(x + 1, w)

    for y in range(h):
        ym = _reflect(y - 1, h)
        yp = _reflect(y + 1, h)
        for x in range(w):
            xm = xprev[x]
            xp = xnext[x]
            p = (gray[ym, xm] * t1 + gray[ym, x] * t2) + gray[ym, xp] * t1
            p = ((p + gray[y, xm] * t2) + gray[y, x] * t4) + gray[y, xp] * t2
            p = ((p + gray[yp, xm] * t1) + gray[yp, x] * t2) + gray[yp, xp] * t1

            g = abs(gray[y, xp] - p)
            g += abs(gray[y, xm] - p)
            g += abs(gray[yp, x] - p)
            g += abs(gray[ym, x] - p)
            g *= half
            out[y, x] = zero if g < thr else g
    return out


@njit(nogil=True)
def _blur1_step_row(gray, sy, xidx, k, w, pref, ring, slot, csum, replace):
    acc = 0.0
    pref[0] = 0.0
    for i in range(xidx.size):
        acc += float(gray[sy, xidx[i]])
        pref[i + 1] = acc
    for x in range(w):
        s = pref[x + k] - pref[x]
        if replace:
            csum[x] += s - ring[slot, x]
        else:
            csum[x] += s
        ring[slot, x] = s


@njit(nogil=True)
def gfg_focus_ia_exact(gray, mean_ksize, threshold):
    """Fused exact box-diff + GFG gradient -> IA focus map before FGF."""
    h, w = gray.shape
    ia = np.empty((h, w), np.float32)

    t1 = np.float32(1.0 / 16.0)
    t2 = np.float32(2.0 / 16.0)
    t4 = np.float32(4.0 / 16.0)
    half = np.float32(0.5)
    zero = np.float32(0.0)
    thr = np.float32(threshold)

    xprev = np.empty(w, np.int32)
    xnext = np.empty(w, np.int32)
    for x in range(w):
        xprev[x] = _reflect(x - 1, w)
        xnext[x] = _reflect(x + 1, w)

    if mean_ksize <= 1:
        for y in range(h):
            ym = _reflect(y - 1, h)
            yp = _reflect(y + 1, h)
            for x in range(w):
                xm = xprev[x]
                xp = xnext[x]
                p = (gray[ym, xm] * t1 + gray[ym, x] * t2) + gray[ym, xp] * t1
                p = ((p + gray[y, xm] * t2) + gray[y, x] * t4) + gray[y, xp] * t2
                p = ((p + gray[yp, xm] * t1) + gray[yp, x] * t2) + gray[yp, xp] * t1
                g = abs(gray[y, xp] - p) + abs(gray[y, xm] - p)
                g = (g + abs(gray[yp, x] - p)) + abs(gray[ym, x] - p)
                g *= half
                ia[y, x] = zero if g < thr else g
        return ia

    rad = mean_ksize // 2
    k = 2 * rad + 1
    inv = 1.0 / (k * k)
    ring = np.empty((k, w), np.float64)
    csum = np.zeros(w, np.float64)
    pref = np.empty(w + k, np.float64)
    xidx = np.empty(w + 2 * rad, np.int32)
    for i in range(w + 2 * rad):
        xidx[i] = _reflect101(i - rad, w)

    for d in range(-rad, rad + 1):
        _blur1_step_row(gray, _reflect101(d, h), xidx, k, w, pref, ring, d + rad, csum, False)

    head = 0
    for y in range(h):
        if y > 0:
            _blur1_step_row(gray, _reflect101(y + rad, h), xidx, k, w, pref, ring, head, csum, True)
            head += 1
            if head == k:
                head = 0
        ym = _reflect(y - 1, h)
        yp = _reflect(y + 1, h)
        for x in range(w):
            xm = xprev[x]
            xp = xnext[x]
            p = (gray[ym, xm] * t1 + gray[ym, x] * t2) + gray[ym, xp] * t1
            p = ((p + gray[y, xm] * t2) + gray[y, x] * t4) + gray[y, xp] * t2
            p = ((p + gray[yp, xm] * t1) + gray[yp, x] * t2) + gray[yp, xp] * t1
            g = abs(gray[y, xp] - p)
            g += abs(gray[y, xm] - p)
            g += abs(gray[yp, x] - p)
            g += abs(gray[ym, x] - p)
            g *= half
            if g >= thr:
                ia[y, x] = g
            else:
                ia[y, x] = abs(gray[y, x] - np.float32(csum[x] * inv))
    return ia


@njit(nogil=True)
def area4_mask_from_idx(idm_map, frame_i, h_sub, w_sub):
    """Exact 4x4 INTER_AREA downsample of binary mask (idm_map == frame_i)."""
    out = np.empty((h_sub, w_sub), np.float32)
    inv16 = np.float32(1.0 / 16.0)
    for ys in range(h_sub):
        y0 = ys * 4
        for xs in range(w_sub):
            x0 = xs * 4
            cnt = 0
            for dy in range(4):
                yy = y0 + dy
                for dx in range(4):
                    if idm_map[yy, x0 + dx] == frame_i:
                        cnt += 1
            out[ys, xs] = np.float32(cnt) * inv16
    return out


@njit(nogil=True)
def _sub_stage1_row(I_sub, p_sub, sy, xidx, k, w, pI, pp, pIp, pII, ring4, slot, csum4, replace):
    aI = ap = aIp = aII = 0.0
    pI[0] = pp[0] = pIp[0] = pII[0] = 0.0
    for i in range(xidx.size):
        xx = xidx[i]
        iv = I_sub[sy, xx]
        pv = p_sub[sy, xx]
        aI += float(iv)
        ap += float(pv)
        aIp += float(iv * pv)
        aII += float(iv * iv)
        pI[i + 1] = aI
        pp[i + 1] = ap
        pIp[i + 1] = aIp
        pII[i + 1] = aII
    for x in range(w):
        s0 = pI[x + k] - pI[x]
        s1 = pp[x + k] - pp[x]
        s2 = pIp[x + k] - pIp[x]
        s3 = pII[x + k] - pII[x]
        if replace:
            csum4[x, 0] += s0 - ring4[slot, x, 0]
            csum4[x, 1] += s1 - ring4[slot, x, 1]
            csum4[x, 2] += s2 - ring4[slot, x, 2]
            csum4[x, 3] += s3 - ring4[slot, x, 3]
        else:
            csum4[x, 0] += s0
            csum4[x, 1] += s1
            csum4[x, 2] += s2
            csum4[x, 3] += s3
        ring4[slot, x, 0] = s0
        ring4[slot, x, 1] = s1
        ring4[slot, x, 2] = s2
        ring4[slot, x, 3] = s3


@njit(nogil=True)
def _sub_stage2_row(a_buf, b_buf, sy, xidx, k, w, pa, pb, ring2, slot, csum2, replace):
    aa = ab = 0.0
    pa[0] = pb[0] = 0.0
    for i in range(xidx.size):
        xx = xidx[i]
        aa += float(a_buf[sy, xx])
        ab += float(b_buf[sy, xx])
        pa[i + 1] = aa
        pb[i + 1] = ab
    for x in range(w):
        s0 = pa[x + k] - pa[x]
        s1 = pb[x + k] - pb[x]
        if replace:
            csum2[x, 0] += s0 - ring2[slot, x, 0]
            csum2[x, 1] += s1 - ring2[slot, x, 1]
        else:
            csum2[x, 0] += s0
            csum2[x, 1] += s1
        ring2[slot, x, 0] = s0
        ring2[slot, x, 1] = s1


@njit(nogil=True)
def fgf_coeffs_exact(I_sub, p_sub, r_sub, eps):
    """Exact two-pass box-filtered guided-filter coefficients (mean_a, mean_b)."""
    h, w = I_sub.shape
    rad = max(0, int(r_sub))
    k = 2 * rad + 1
    inv = 1.0 / (k * k)
    eps32 = np.float32(eps)

    a_buf = np.empty((h, w), np.float32)
    b_buf = np.empty((h, w), np.float32)

    ring4 = np.empty((k, w, 4), np.float64)
    csum4 = np.zeros((w, 4), np.float64)
    pI = np.empty(w + k, np.float64)
    pp = np.empty(w + k, np.float64)
    pIp = np.empty(w + k, np.float64)
    pII = np.empty(w + k, np.float64)
    xidx = np.empty(w + 2 * rad, np.int32)
    for i in range(w + 2 * rad):
        xidx[i] = _reflect101(i - rad, w)

    for d in range(-rad, rad + 1):
        _sub_stage1_row(I_sub, p_sub, _reflect101(d, h), xidx, k, w,
                        pI, pp, pIp, pII, ring4, d + rad, csum4, False)
    for x in range(w):
        mI = np.float32(csum4[x, 0] * inv)
        mp = np.float32(csum4[x, 1] * inv)
        mIp = np.float32(csum4[x, 2] * inv)
        mII = np.float32(csum4[x, 3] * inv)
        a = (mIp - mI * mp) / ((mII - mI * mI) + eps32)
        a_buf[0, x] = a
        b_buf[0, x] = mp - a * mI

    head = 0
    for y in range(1, h):
        _sub_stage1_row(I_sub, p_sub, _reflect101(y + rad, h), xidx, k, w,
                        pI, pp, pIp, pII, ring4, head, csum4, True)
        head += 1
        if head == k:
            head = 0
        for x in range(w):
            mI = np.float32(csum4[x, 0] * inv)
            mp = np.float32(csum4[x, 1] * inv)
            mIp = np.float32(csum4[x, 2] * inv)
            mII = np.float32(csum4[x, 3] * inv)
            a = (mIp - mI * mp) / ((mII - mI * mI) + eps32)
            a_buf[y, x] = a
            b_buf[y, x] = mp - a * mI

    mean_a = np.empty((h, w), np.float32)
    mean_b = np.empty((h, w), np.float32)
    ring2 = np.empty((k, w, 2), np.float64)
    csum2 = np.zeros((w, 2), np.float64)
    pa = np.empty(w + k, np.float64)
    pb = np.empty(w + k, np.float64)

    for d in range(-rad, rad + 1):
        _sub_stage2_row(a_buf, b_buf, _reflect101(d, h), xidx, k, w,
                        pa, pb, ring2, d + rad, csum2, False)
    for x in range(w):
        mean_a[0, x] = np.float32(csum2[x, 0] * inv)
        mean_b[0, x] = np.float32(csum2[x, 1] * inv)

    head = 0
    for y in range(1, h):
        _sub_stage2_row(a_buf, b_buf, _reflect101(y + rad, h), xidx, k, w,
                        pa, pb, ring2, head, csum2, True)
        head += 1
        if head == k:
            head = 0
        for x in range(w):
            mean_a[y, x] = np.float32(csum2[x, 0] * inv)
            mean_b[y, x] = np.float32(csum2[x, 1] * inv)

    return mean_a, mean_b


@njit(nogil=True)
def fgf_apply_q(mean_a, I, mean_b):
    """Compute q = mean_a * I + mean_b without intermediate allocations."""
    h, w = I.shape
    out = np.empty((h, w), np.float32)
    for y in range(h):
        for x in range(w):
            out[y, x] = mean_a[y, x] * I[y, x] + mean_b[y, x]
    return out


@njit(nogil=True)
def gfg_decision_update(afm, gx, gy, val, sob_best, idx, frame_i, tie_eps):
    """Exact streaming GfgDecision update with 3x3 Sobel tie-breaker."""
    h, w = afm.shape
    eps = np.float32(tie_eps)
    for y in range(h):
        for x in range(w):
            v = afm[y, x]
            s = abs(gx[y, x]) + abs(gy[y, x])
            diff = v - val[y, x]
            if (diff > eps) or (abs(diff) <= eps and s > sob_best[y, x]):
                val[y, x] = v
                sob_best[y, x] = s
                idx[y, x] = frame_i


@njit(nogil=True)
def sobel_abs_sum(gx, gy, out):
    """Exact |gx| + |gy| Sobel gradient magnitude."""
    h, w = gx.shape
    for y in range(h):
        for x in range(w):
            out[y, x] = abs(gx[y, x]) + abs(gy[y, x])


@njit(nogil=True)
def bincount_i32(idx, num_imgs):
    """Count winning pixels per frame index in one pass."""
    h, w = idx.shape
    counts = np.zeros(num_imgs, np.int64)
    for y in range(h):
        for x in range(w):
            counts[idx[y, x]] += 1
    return counts


@njit(nogil=True)
def accumulate_gfg_u8(raw, lut, weight, acc, sum_w):
    """Accumulate normalized uint8 BGR frame weighted by GFG weight map."""
    h, w = weight.shape
    if raw.ndim == 2:
        for y in range(h):
            for x in range(w):
                wt = weight[y, x]
                v = lut[raw[y, x]] * wt
                sum_w[y, x] += wt
                acc[y, x, 0] += v
                acc[y, x, 1] += v
                acc[y, x, 2] += v
    else:
        for y in range(h):
            for x in range(w):
                wt = weight[y, x]
                sum_w[y, x] += wt
                acc[y, x, 0] += lut[raw[y, x, 0]] * wt
                acc[y, x, 1] += lut[raw[y, x, 1]] * wt
                acc[y, x, 2] += lut[raw[y, x, 2]] * wt


@njit(nogil=True)
def accumulate_gfg_u16(raw, weight, acc, sum_w):
    """Accumulate normalized uint16 BGR frame weighted by GFG weight map."""
    h, w = weight.shape
    scale = np.float32(255.0 / 65535.0)
    inv255 = np.float32(255.0)
    if raw.ndim == 2:
        for y in range(h):
            for x in range(w):
                wt = weight[y, x]
                v = ((np.float32(raw[y, x]) * scale) / inv255) * wt
                sum_w[y, x] += wt
                acc[y, x, 0] += v
                acc[y, x, 1] += v
                acc[y, x, 2] += v
    else:
        for y in range(h):
            for x in range(w):
                wt = weight[y, x]
                sum_w[y, x] += wt
                acc[y, x, 0] += ((np.float32(raw[y, x, 0]) * scale) / inv255) * wt
                acc[y, x, 1] += ((np.float32(raw[y, x, 1]) * scale) / inv255) * wt
                acc[y, x, 2] += ((np.float32(raw[y, x, 2]) * scale) / inv255) * wt


@njit(nogil=True)
def accumulate_gfg_f32(raw, weight, acc, sum_w):
    """Accumulate normalized float BGR frame weighted by GFG weight map."""
    h, w = weight.shape
    inv255 = np.float32(255.0)
    if raw.ndim == 2:
        for y in range(h):
            for x in range(w):
                wt = weight[y, x]
                v = (np.float32(raw[y, x]) / inv255) * wt
                sum_w[y, x] += wt
                acc[y, x, 0] += v
                acc[y, x, 1] += v
                acc[y, x, 2] += v
    else:
        for y in range(h):
            for x in range(w):
                wt = weight[y, x]
                sum_w[y, x] += wt
                acc[y, x, 0] += (np.float32(raw[y, x, 0]) / inv255) * wt
                acc[y, x, 1] += (np.float32(raw[y, x, 1]) / inv255) * wt
                acc[y, x, 2] += (np.float32(raw[y, x, 2]) / inv255) * wt


@njit(nogil=True)
def normalize_gfg_rows(acc, sum_w, start, stop):
    """Divide non-zero rows in place; return True if any pixel has sum_w <= 1e-6."""
    w = sum_w.shape[1]
    eps = np.float32(1e-6)
    has_zero = False
    for y in range(start, stop):
        for x in range(w):
            den = sum_w[y, x]
            if den > eps:
                acc[y, x, 0] /= den
                acc[y, x, 1] /= den
                acc[y, x, 2] /= den
            else:
                has_zero = True
    return has_zero


@njit(nogil=True)
def fill_zero_weight_rows_u8(raw, lut, idm_map, sum_w, frame_i, acc, start, stop):
    w = sum_w.shape[1]
    eps = np.float32(1e-6)
    for y in range(start, stop):
        for x in range(w):
            if sum_w[y, x] <= eps and idm_map[y, x] == frame_i:
                if raw.ndim == 2:
                    v = lut[raw[y, x]]
                    acc[y, x, 0] = v
                    acc[y, x, 1] = v
                    acc[y, x, 2] = v
                else:
                    acc[y, x, 0] = lut[raw[y, x, 0]]
                    acc[y, x, 1] = lut[raw[y, x, 1]]
                    acc[y, x, 2] = lut[raw[y, x, 2]]


@njit(nogil=True)
def fill_zero_weight_rows_u16(raw, idm_map, sum_w, frame_i, acc, start, stop):
    w = sum_w.shape[1]
    eps = np.float32(1e-6)
    scale = np.float32(255.0 / 65535.0)
    inv255 = np.float32(255.0)
    for y in range(start, stop):
        for x in range(w):
            if sum_w[y, x] <= eps and idm_map[y, x] == frame_i:
                if raw.ndim == 2:
                    v = (np.float32(raw[y, x]) * scale) / inv255
                    acc[y, x, 0] = v
                    acc[y, x, 1] = v
                    acc[y, x, 2] = v
                else:
                    acc[y, x, 0] = (np.float32(raw[y, x, 0]) * scale) / inv255
                    acc[y, x, 1] = (np.float32(raw[y, x, 1]) * scale) / inv255
                    acc[y, x, 2] = (np.float32(raw[y, x, 2]) * scale) / inv255


@njit(nogil=True)
def fill_zero_weight_rows_f32(raw, idm_map, sum_w, frame_i, acc, start, stop):
    w = sum_w.shape[1]
    eps = np.float32(1e-6)
    inv255 = np.float32(255.0)
    for y in range(start, stop):
        for x in range(w):
            if sum_w[y, x] <= eps and idm_map[y, x] == frame_i:
                if raw.ndim == 2:
                    v = np.float32(raw[y, x]) / inv255
                    acc[y, x, 0] = v
                    acc[y, x, 1] = v
                    acc[y, x, 2] = v
                else:
                    acc[y, x, 0] = np.float32(raw[y, x, 0]) / inv255
                    acc[y, x, 1] = np.float32(raw[y, x, 1]) / inv255
                    acc[y, x, 2] = np.float32(raw[y, x, 2]) / inv255


@njit(nogil=True)
def quantize_gfg_rows(acc, out, start, stop):
    """Scale [0, 1] -> [0, 255], clamp, and write into uint8/float32 out."""
    w = acc.shape[1]
    scale = np.float32(255.0)
    zero = np.float32(0.0)
    for y in range(start, stop):
        for x in range(w):
            for c in range(3):
                v = acc[y, x, c] * scale
                if v < zero:
                    v = zero
                elif v > scale:
                    v = scale
                out[y, x, c] = v
