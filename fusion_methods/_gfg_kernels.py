"""Exact CPU kernels for GFG-FGF (Gradient-domain Guided Filter Fusion).

Uses bounded float64 prefix/sliding sums, AVX2-friendly interior loops,
fused Sobel decision updates, active bounding-box ROI mask refinement, and
row-strip parallel kernels to reproduce the reference arithmetic bit-for-bit
across CPU cores with bounded RAM.
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
def extract_channel_u8_rows(raw, ch, lut, out, y0, y1):
    """Extract normalized [0, 1] float32 single channel for rows [y0:y1] from uint8."""
    w = out.shape[1]
    if raw.ndim == 2:
        for y in range(y0, y1):
            for x in range(w):
                out[y, x] = lut[raw[y, x]]
    else:
        for y in range(y0, y1):
            for x in range(w):
                out[y, x] = lut[raw[y, x, ch]]


@njit(nogil=True)
def extract_channel_u8(raw, ch, lut, out):
    """Extract normalized [0, 1] float32 single channel from uint8 frame."""
    extract_channel_u8_rows(raw, ch, lut, out, 0, out.shape[0])


@njit(nogil=True)
def extract_channel_u16_rows(raw, ch, out, y0, y1):
    """Extract normalized [0, 1] float32 single channel for rows [y0:y1] from uint16."""
    w = out.shape[1]
    scale = np.float32(255.0 / 65535.0)
    inv255 = np.float32(255.0)
    if raw.ndim == 2:
        for y in range(y0, y1):
            for x in range(w):
                out[y, x] = (np.float32(raw[y, x]) * scale) / inv255
    else:
        for y in range(y0, y1):
            for x in range(w):
                out[y, x] = (np.float32(raw[y, x, ch]) * scale) / inv255


@njit(nogil=True)
def extract_channel_u16(raw, ch, out):
    """Extract normalized [0, 1] float32 single channel from uint16 frame."""
    extract_channel_u16_rows(raw, ch, out, 0, out.shape[0])


@njit(nogil=True)
def extract_channel_f32_rows(raw, ch, out, y0, y1):
    """Extract normalized [0, 1] float32 single channel for rows [y0:y1] from float."""
    w = out.shape[1]
    inv255 = np.float32(255.0)
    if raw.ndim == 2:
        for y in range(y0, y1):
            for x in range(w):
                out[y, x] = np.float32(raw[y, x]) / inv255
    else:
        for y in range(y0, y1):
            for x in range(w):
                out[y, x] = np.float32(raw[y, x, ch]) / inv255


@njit(nogil=True)
def extract_channel_f32(raw, ch, out):
    """Extract normalized [0, 1] float32 single channel from float frame."""
    extract_channel_f32_rows(raw, ch, out, 0, out.shape[0])


@njit(nogil=True)
def scharr_focus_score_rows(gray, y0, y1):
    """Interior sum(|Gx|^2 + |Gy|^2) over rows [y0:y1] (clamped to [1, h-1))."""
    h, w = gray.shape
    if h <= 2 or w <= 2:
        return 0.0
    lo = max(1, y0)
    hi = min(h - 1, y1)
    c3 = np.float32(3.0)
    c10 = np.float32(10.0)
    total = 0.0
    for y in range(lo, hi):
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
    return total


@njit(nogil=True)
def scharr_focus_score_exact(gray):
    """Interior mean(|Gx|^2 + |Gy|^2) using 3x3 Scharr-like operators."""
    h, w = gray.shape
    if h <= 2 or w <= 2:
        return np.float32(0.0)
    total = scharr_focus_score_rows(gray, 1, h - 1)
    return np.float32(total / ((h - 2) * (w - 2)))


@njit(nogil=True)
def _gfg_grad_pixel(g00, g01, g02, g10, g11, g12, g20, g21, g22):
    t1 = np.float32(1.0 / 16.0)
    t2 = np.float32(2.0 / 16.0)
    t4 = np.float32(4.0 / 16.0)
    half = np.float32(0.5)
    p = (g00 * t1 + g01 * t2) + g02 * t1
    p = ((p + g10 * t2) + g11 * t4) + g12 * t2
    p = ((p + g20 * t1) + g21 * t2) + g22 * t1
    g = abs(g12 - p)
    g += abs(g10 - p)
    g += abs(g21 - p)
    g += abs(g01 - p)
    return g * half


@njit(nogil=True)
def gfg_gradient_exact(gray, threshold):
    """Exact bit-for-bit GFG four-neighborhood gradient (BORDER_REFLECT)."""
    h, w = gray.shape
    out = np.empty((h, w), np.float32)
    zero = np.float32(0.0)
    thr = np.float32(threshold)

    for y in range(h):
        ym = _reflect(y - 1, h)
        yp = _reflect(y + 1, h)
        if w <= 2:
            for x in range(w):
                xm = _reflect(x - 1, w)
                xp = _reflect(x + 1, w)
                g = _gfg_grad_pixel(
                    gray[ym, xm], gray[ym, x], gray[ym, xp],
                    gray[y, xm], gray[y, x], gray[y, xp],
                    gray[yp, xm], gray[yp, x], gray[yp, xp],
                )
                out[y, x] = zero if g < thr else g
        else:
            # Left border x == 0
            g = _gfg_grad_pixel(
                gray[ym, 0], gray[ym, 0], gray[ym, 1],
                gray[y, 0], gray[y, 0], gray[y, 1],
                gray[yp, 0], gray[yp, 0], gray[yp, 1],
            )
            out[y, 0] = zero if g < thr else g
            # Interior [1, w - 1): direct indexing for SIMD auto-vectorization
            t1 = np.float32(1.0 / 16.0)
            t2 = np.float32(2.0 / 16.0)
            t4 = np.float32(4.0 / 16.0)
            half = np.float32(0.5)
            for x in range(1, w - 1):
                p = (gray[ym, x - 1] * t1 + gray[ym, x] * t2) + gray[ym, x + 1] * t1
                p = ((p + gray[y, x - 1] * t2) + gray[y, x] * t4) + gray[y, x + 1] * t2
                p = ((p + gray[yp, x - 1] * t1) + gray[yp, x] * t2) + gray[yp, x + 1] * t1
                g = abs(gray[y, x + 1] - p)
                g += abs(gray[y, x - 1] - p)
                g += abs(gray[yp, x] - p)
                g += abs(gray[ym, x] - p)
                g *= half
                out[y, x] = zero if g < thr else g
            # Right border x == w - 1
            wm1 = w - 1
            wm2 = w - 2
            g = _gfg_grad_pixel(
                gray[ym, wm2], gray[ym, wm1], gray[ym, wm1],
                gray[y, wm2], gray[y, wm1], gray[y, wm1],
                gray[yp, wm2], gray[yp, wm1], gray[yp, wm1],
            )
            out[y, wm1] = zero if g < thr else g
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
def gfg_focus_ia_rows(gray, mean_ksize, threshold, ia, y0, y1):
    """Exact box-diff + GFG gradient -> IA for rows [y0:y1]."""
    if y0 >= y1:
        return
    h, w = gray.shape
    t1 = np.float32(1.0 / 16.0)
    t2 = np.float32(2.0 / 16.0)
    t4 = np.float32(4.0 / 16.0)
    half = np.float32(0.5)
    zero = np.float32(0.0)
    thr = np.float32(threshold)

    if mean_ksize <= 1:
        for y in range(y0, y1):
            ym = _reflect(y - 1, h)
            yp = _reflect(y + 1, h)
            if w <= 2:
                for x in range(w):
                    xm = _reflect(x - 1, w)
                    xp = _reflect(x + 1, w)
                    g = _gfg_grad_pixel(
                        gray[ym, xm], gray[ym, x], gray[ym, xp],
                        gray[y, xm], gray[y, x], gray[y, xp],
                        gray[yp, xm], gray[yp, x], gray[yp, xp],
                    )
                    ia[y, x] = zero if g < thr else g
            else:
                g = _gfg_grad_pixel(
                    gray[ym, 0], gray[ym, 0], gray[ym, 1],
                    gray[y, 0], gray[y, 0], gray[y, 1],
                    gray[yp, 0], gray[yp, 0], gray[yp, 1],
                )
                ia[y, 0] = zero if g < thr else g
                for x in range(1, w - 1):
                    p = (gray[ym, x - 1] * t1 + gray[ym, x] * t2) + gray[ym, x + 1] * t1
                    p = ((p + gray[y, x - 1] * t2) + gray[y, x] * t4) + gray[y, x + 1] * t2
                    p = ((p + gray[yp, x - 1] * t1) + gray[yp, x] * t2) + gray[yp, x + 1] * t1
                    g = abs(gray[y, x + 1] - p) + abs(gray[y, x - 1] - p)
                    g = (g + abs(gray[yp, x] - p)) + abs(gray[ym, x] - p)
                    g *= half
                    ia[y, x] = zero if g < thr else g
                wm1 = w - 1
                wm2 = w - 2
                g = _gfg_grad_pixel(
                    gray[ym, wm2], gray[ym, wm1], gray[ym, wm1],
                    gray[y, wm2], gray[y, wm1], gray[y, wm1],
                    gray[yp, wm2], gray[yp, wm1], gray[yp, wm1],
                )
                ia[y, wm1] = zero if g < thr else g
        return

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
        _blur1_step_row(gray, _reflect101(y0 + d, h), xidx, k, w, pref, ring, d + rad, csum, False)

    head = 0
    for y in range(y0, y1):
        if y > y0:
            _blur1_step_row(gray, _reflect101(y + rad, h), xidx, k, w, pref, ring, head, csum, True)
            head += 1
            if head == k:
                head = 0
        ym = _reflect(y - 1, h)
        yp = _reflect(y + 1, h)
        if w <= 2:
            for x in range(w):
                xm = _reflect(x - 1, w)
                xp = _reflect(x + 1, w)
                g = _gfg_grad_pixel(
                    gray[ym, xm], gray[ym, x], gray[ym, xp],
                    gray[y, xm], gray[y, x], gray[y, xp],
                    gray[yp, xm], gray[yp, x], gray[yp, xp],
                )
                ia[y, x] = g if g >= thr else abs(gray[y, x] - np.float32(csum[x] * inv))
        else:
            g = _gfg_grad_pixel(
                gray[ym, 0], gray[ym, 0], gray[ym, 1],
                gray[y, 0], gray[y, 0], gray[y, 1],
                gray[yp, 0], gray[yp, 0], gray[yp, 1],
            )
            ia[y, 0] = g if g >= thr else abs(gray[y, 0] - np.float32(csum[0] * inv))
            for x in range(1, w - 1):
                p = (gray[ym, x - 1] * t1 + gray[ym, x] * t2) + gray[ym, x + 1] * t1
                p = ((p + gray[y, x - 1] * t2) + gray[y, x] * t4) + gray[y, x + 1] * t2
                p = ((p + gray[yp, x - 1] * t1) + gray[yp, x] * t2) + gray[yp, x + 1] * t1
                g = abs(gray[y, x + 1] - p)
                g += abs(gray[y, x - 1] - p)
                g += abs(gray[yp, x] - p)
                g += abs(gray[ym, x] - p)
                g *= half
                if g >= thr:
                    ia[y, x] = g
                else:
                    ia[y, x] = abs(gray[y, x] - np.float32(csum[x] * inv))
            wm1 = w - 1
            wm2 = w - 2
            g = _gfg_grad_pixel(
                gray[ym, wm2], gray[ym, wm1], gray[ym, wm1],
                gray[y, wm2], gray[y, wm1], gray[y, wm1],
                gray[yp, wm2], gray[yp, wm1], gray[yp, wm1],
            )
            ia[y, wm1] = g if g >= thr else abs(gray[y, wm1] - np.float32(csum[wm1] * inv))


@njit(nogil=True)
def gfg_focus_ia_exact(gray, mean_ksize, threshold):
    """Fused exact box-diff + GFG gradient -> IA focus map before FGF."""
    h, w = gray.shape
    ia = np.empty((h, w), np.float32)
    gfg_focus_ia_rows(gray, mean_ksize, threshold, ia, 0, h)
    return ia


@njit(nogil=True)
def gfg_focus_ia_area4_rows(gray, mean_ksize, threshold, I_sub, p_sub, ys0, ys1):
    """Fused IA focus map + 4x4 INTER_AREA downsample for sub-rows [ys0:ys1] without full IA buffer."""
    if ys0 >= ys1:
        return
    h, w = gray.shape
    ws = w // 4
    t1 = np.float32(1.0 / 16.0)
    t2 = np.float32(2.0 / 16.0)
    t4 = np.float32(4.0 / 16.0)
    half = np.float32(0.5)
    zero = np.float32(0.0)
    thr = np.float32(threshold)
    inv16 = np.float32(1.0 / 16.0)

    y_start = ys0 * 4
    y_stop = ys1 * 4
    ia_row = np.empty(w, np.float32)
    accI = np.empty(ws, np.float32)
    accP = np.empty(ws, np.float32)

    use_blur = mean_ksize > 1
    if use_blur:
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
            _blur1_step_row(gray, _reflect101(y_start + d, h), xidx, k, w, pref, ring, d + rad, csum, False)
        head = 0
    else:
        rad = 0
        k = 1
        inv = 1.0
        ring = np.empty((1, 1), np.float64)
        csum = np.zeros(1, np.float64)
        pref = np.empty(1, np.float64)
        xidx = np.empty(1, np.int32)
        head = 0

    for ys in range(ys0, ys1):
        for xs in range(ws):
            accI[xs] = zero
            accP[xs] = zero
        for dy in range(4):
            y = ys * 4 + dy
            if use_blur and y > y_start:
                _blur1_step_row(gray, _reflect101(y + rad, h), xidx, k, w, pref, ring, head, csum, True)
                head += 1
                if head == k:
                    head = 0
            ym = _reflect(y - 1, h)
            yp = _reflect(y + 1, h)
            # x == 0
            g = _gfg_grad_pixel(
                gray[ym, 0], gray[ym, 0], gray[ym, 1],
                gray[y, 0], gray[y, 0], gray[y, 1],
                gray[yp, 0], gray[yp, 0], gray[yp, 1],
            )
            if g >= thr:
                ia_row[0] = g
            elif use_blur:
                ia_row[0] = abs(gray[y, 0] - np.float32(csum[0] * inv))
            else:
                ia_row[0] = zero
            # Interior x in [1, w - 1)
            for x in range(1, w - 1):
                p = (gray[ym, x - 1] * t1 + gray[ym, x] * t2) + gray[ym, x + 1] * t1
                p = ((p + gray[y, x - 1] * t2) + gray[y, x] * t4) + gray[y, x + 1] * t2
                p = ((p + gray[yp, x - 1] * t1) + gray[yp, x] * t2) + gray[yp, x + 1] * t1
                g = abs(gray[y, x + 1] - p)
                g += abs(gray[y, x - 1] - p)
                g += abs(gray[yp, x] - p)
                g += abs(gray[ym, x] - p)
                g *= half
                if g >= thr:
                    ia_row[x] = g
                elif use_blur:
                    ia_row[x] = abs(gray[y, x] - np.float32(csum[x] * inv))
                else:
                    ia_row[x] = zero
            # x == w - 1
            wm1 = w - 1
            wm2 = w - 2
            g = _gfg_grad_pixel(
                gray[ym, wm2], gray[ym, wm1], gray[ym, wm1],
                gray[y, wm2], gray[y, wm1], gray[y, wm1],
                gray[yp, wm2], gray[yp, wm1], gray[yp, wm1],
            )
            if g >= thr:
                ia_row[wm1] = g
            elif use_blur:
                ia_row[wm1] = abs(gray[y, wm1] - np.float32(csum[wm1] * inv))
            else:
                ia_row[wm1] = zero

            for xs in range(ws):
                x0 = xs * 4
                accI[xs] += (gray[y, x0] + gray[y, x0 + 1]) + (gray[y, x0 + 2] + gray[y, x0 + 3])
                accP[xs] += (ia_row[x0] + ia_row[x0 + 1]) + (ia_row[x0 + 2] + ia_row[x0 + 3])
        for xs in range(ws):
            I_sub[ys, xs] = accI[xs] * inv16
            p_sub[ys, xs] = accP[xs] * inv16


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
def area4_channel_and_mask_roi_u8(raw, lut, ch, idm_map, frame_i,
                                  sy0, sx0, sh, sw, I_sub, p_sub, ry0, ry1):
    """Fused 4x4 INTER_AREA downsample of channel + binary mask on sub-ROI rows [ry0:ry1]."""
    inv16 = np.float32(1.0 / 16.0)
    is_gray = raw.ndim == 2
    for ry in range(ry0, ry1):
        y0 = (sy0 + ry) * 4
        for rx in range(sw):
            x0 = (sx0 + rx) * 4
            sI = np.float32(0.0)
            cnt = 0
            for dy in range(4):
                yy = y0 + dy
                for dx in range(4):
                    xx = x0 + dx
                    if is_gray:
                        sI += lut[raw[yy, xx]]
                    else:
                        sI += lut[raw[yy, xx, ch]]
                    if idm_map[yy, xx] == frame_i:
                        cnt += 1
            I_sub[ry, rx] = sI * inv16
            p_sub[ry, rx] = np.float32(cnt) * inv16


@njit(nogil=True)
def area4_channel_and_mask_roi_u16(raw, ch, idm_map, frame_i,
                                   sy0, sx0, sh, sw, I_sub, p_sub, ry0, ry1):
    """Fused 4x4 INTER_AREA downsample of uint16 channel + binary mask on sub-ROI rows."""
    inv16 = np.float32(1.0 / 16.0)
    scale = np.float32(255.0 / 65535.0)
    inv255 = np.float32(255.0)
    is_gray = raw.ndim == 2
    for ry in range(ry0, ry1):
        y0 = (sy0 + ry) * 4
        for rx in range(sw):
            x0 = (sx0 + rx) * 4
            sI = np.float32(0.0)
            cnt = 0
            for dy in range(4):
                yy = y0 + dy
                for dx in range(4):
                    xx = x0 + dx
                    if is_gray:
                        sI += (np.float32(raw[yy, xx]) * scale) / inv255
                    else:
                        sI += (np.float32(raw[yy, xx, ch]) * scale) / inv255
                    if idm_map[yy, xx] == frame_i:
                        cnt += 1
            I_sub[ry, rx] = sI * inv16
            p_sub[ry, rx] = np.float32(cnt) * inv16


@njit(nogil=True)
def area4_channel_and_mask_roi_f32(raw, ch, idm_map, frame_i,
                                   sy0, sx0, sh, sw, I_sub, p_sub, ry0, ry1):
    """Fused 4x4 INTER_AREA downsample of float channel + binary mask on sub-ROI rows."""
    inv16 = np.float32(1.0 / 16.0)
    inv255 = np.float32(255.0)
    is_gray = raw.ndim == 2
    for ry in range(ry0, ry1):
        y0 = (sy0 + ry) * 4
        for rx in range(sw):
            x0 = (sx0 + rx) * 4
            sI = np.float32(0.0)
            cnt = 0
            for dy in range(4):
                yy = y0 + dy
                for dx in range(4):
                    xx = x0 + dx
                    if is_gray:
                        sI += np.float32(raw[yy, xx]) / inv255
                    else:
                        sI += np.float32(raw[yy, xx, ch]) / inv255
                    if idm_map[yy, xx] == frame_i:
                        cnt += 1
            I_sub[ry, rx] = sI * inv16
            p_sub[ry, rx] = np.float32(cnt) * inv16


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
    zero = np.float32(0.0)

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
        if csum4[x, 1] == 0.0:
            a_buf[0, x] = zero
            b_buf[0, x] = zero
        else:
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
            if csum4[x, 1] == 0.0:
                a_buf[y, x] = zero
                b_buf[y, x] = zero
            else:
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
def fgf_coeffs_exact_roi(I_sub, p_sub, full_hs, full_ws,
                         sub_y0, sub_x0, ay0, ay1, ax0, ax1,
                         my0, my1, mx0, mx1, r_sub, eps):
    """Exact two-pass guided-filter coefficients on ROI [my0:my1, mx0:mx1] of (full_hs, full_ws)."""
    rad = max(0, int(r_sub))
    k = 2 * rad + 1
    inv = 1.0 / (k * k)
    eps32 = np.float32(eps)
    zero = np.float32(0.0)

    ah = ay1 - ay0
    aw = ax1 - ax0
    a_buf = np.empty((ah, aw), np.float32)
    b_buf = np.empty((ah, aw), np.float32)

    ring4 = np.empty((k, aw, 4), np.float64)
    csum4 = np.zeros((aw, 4), np.float64)
    pI = np.empty(aw + k, np.float64)
    pp = np.empty(aw + k, np.float64)
    pIp = np.empty(aw + k, np.float64)
    pII = np.empty(aw + k, np.float64)
    xidx = np.empty(aw + 2 * rad, np.int32)
    for i in range(aw + 2 * rad):
        xidx[i] = _reflect101(ax0 + i - rad, full_ws) - sub_x0

    for d in range(-rad, rad + 1):
        sy = _reflect101(ay0 + d, full_hs) - sub_y0
        _sub_stage1_row(I_sub, p_sub, sy, xidx, k, aw,
                        pI, pp, pIp, pII, ring4, d + rad, csum4, False)
    for x in range(aw):
        if csum4[x, 1] == 0.0:
            a_buf[0, x] = zero
            b_buf[0, x] = zero
        else:
            mI = np.float32(csum4[x, 0] * inv)
            mp = np.float32(csum4[x, 1] * inv)
            mIp = np.float32(csum4[x, 2] * inv)
            mII = np.float32(csum4[x, 3] * inv)
            a = (mIp - mI * mp) / ((mII - mI * mI) + eps32)
            a_buf[0, x] = a
            b_buf[0, x] = mp - a * mI

    head = 0
    for ry in range(1, ah):
        sy = _reflect101(ay0 + ry + rad, full_hs) - sub_y0
        _sub_stage1_row(I_sub, p_sub, sy, xidx, k, aw,
                        pI, pp, pIp, pII, ring4, head, csum4, True)
        head += 1
        if head == k:
            head = 0
        for x in range(aw):
            if csum4[x, 1] == 0.0:
                a_buf[ry, x] = zero
                b_buf[ry, x] = zero
            else:
                mI = np.float32(csum4[x, 0] * inv)
                mp = np.float32(csum4[x, 1] * inv)
                mIp = np.float32(csum4[x, 2] * inv)
                mII = np.float32(csum4[x, 3] * inv)
                a = (mIp - mI * mp) / ((mII - mI * mI) + eps32)
                a_buf[ry, x] = a
                b_buf[ry, x] = mp - a * mI

    mh = my1 - my0
    mw = mx1 - mx0
    mean_a = np.empty((mh, mw), np.float32)
    mean_b = np.empty((mh, mw), np.float32)
    ring2 = np.empty((k, mw, 2), np.float64)
    csum2 = np.zeros((mw, 2), np.float64)
    pa = np.empty(mw + k, np.float64)
    pb = np.empty(mw + k, np.float64)
    xloc = np.empty(mw + 2 * rad, np.int32)
    for i in range(mw + 2 * rad):
        fx = _reflect101(mx0 + i - rad, full_ws)
        xloc[i] = fx - ax0 if (ax0 <= fx < ax1) else -1

    for d in range(-rad, rad + 1):
        fy = _reflect101(my0 + d, full_hs)
        if ay0 <= fy < ay1:
            ry = fy - ay0
            aa = ab = 0.0
            pa[0] = pb[0] = 0.0
            for i in range(xloc.size):
                lx = xloc[i]
                if lx >= 0:
                    aa += float(a_buf[ry, lx])
                    ab += float(b_buf[ry, lx])
                pa[i + 1] = aa
                pb[i + 1] = ab
            for ix in range(mw):
                sa = pa[ix + k] - pa[ix]
                sb = pb[ix + k] - pb[ix]
                csum2[ix, 0] += sa
                csum2[ix, 1] += sb
                ring2[d + rad, ix, 0] = sa
                ring2[d + rad, ix, 1] = sb
        else:
            for ix in range(mw):
                ring2[d + rad, ix, 0] = 0.0
                ring2[d + rad, ix, 1] = 0.0

    for ix in range(mw):
        mean_a[0, ix] = np.float32(csum2[ix, 0] * inv)
        mean_b[0, ix] = np.float32(csum2[ix, 1] * inv)

    head = 0
    for ry_m in range(1, mh):
        fy = _reflect101(my0 + ry_m + rad, full_hs)
        if ay0 <= fy < ay1:
            ry = fy - ay0
            aa = ab = 0.0
            pa[0] = pb[0] = 0.0
            for i in range(xloc.size):
                lx = xloc[i]
                if lx >= 0:
                    aa += float(a_buf[ry, lx])
                    ab += float(b_buf[ry, lx])
                pa[i + 1] = aa
                pb[i + 1] = ab
            for ix in range(mw):
                sa = pa[ix + k] - pa[ix]
                sb = pb[ix + k] - pb[ix]
                csum2[ix, 0] += sa - ring2[head, ix, 0]
                csum2[ix, 1] += sb - ring2[head, ix, 1]
                ring2[head, ix, 0] = sa
                ring2[head, ix, 1] = sb
        else:
            for ix in range(mw):
                csum2[ix, 0] -= ring2[head, ix, 0]
                csum2[ix, 1] -= ring2[head, ix, 1]
                ring2[head, ix, 0] = 0.0
                ring2[head, ix, 1] = 0.0
        head += 1
        if head == k:
            head = 0
        for ix in range(mw):
            mean_a[ry_m, ix] = np.float32(csum2[ix, 0] * inv)
            mean_b[ry_m, ix] = np.float32(csum2[ix, 1] * inv)

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
def _eval_upsample4_row(mean_a_sub, mean_b_sub, gray, hs, ws, y, dst_row):
    """Evaluate exact 4x INTER_LINEAR upsample(mean_a)*gray + upsample(mean_b) for row y."""
    ys = y >> 2
    dy = y & 3
    if dy == 0:
        y0 = ys - 1 if ys > 0 else 0
        y1 = ys
        wy1 = np.float32(0.625) if ys > 0 else np.float32(1.0)
    elif dy == 1:
        y0 = ys - 1 if ys > 0 else 0
        y1 = ys
        wy1 = np.float32(0.875) if ys > 0 else np.float32(1.0)
    elif dy == 2:
        y0 = ys
        y1 = ys + 1 if ys + 1 < hs else hs - 1
        wy1 = np.float32(0.125) if ys + 1 < hs else np.float32(0.0)
    else:
        y0 = ys
        y1 = ys + 1 if ys + 1 < hs else hs - 1
        wy1 = np.float32(0.375) if ys + 1 < hs else np.float32(0.0)
    wy0 = np.float32(1.0) - wy1

    c0625 = np.float32(0.625)
    c0375 = np.float32(0.375)
    c0875 = np.float32(0.875)
    c0125 = np.float32(0.125)

    for xs in range(ws):
        xm = xs - 1 if xs > 0 else 0
        xp = xs + 1 if xs + 1 < ws else ws - 1
        a_l = mean_a_sub[y0, xm] * wy0 + mean_a_sub[y1, xm] * wy1
        a_c = mean_a_sub[y0, xs] * wy0 + mean_a_sub[y1, xs] * wy1
        a_r = mean_a_sub[y0, xp] * wy0 + mean_a_sub[y1, xp] * wy1
        b_l = mean_b_sub[y0, xm] * wy0 + mean_b_sub[y1, xm] * wy1
        b_c = mean_b_sub[y0, xs] * wy0 + mean_b_sub[y1, xs] * wy1
        b_r = mean_b_sub[y0, xp] * wy0 + mean_b_sub[y1, xp] * wy1

        x0 = xs * 4
        if xs > 0:
            a0 = a_l * c0375 + a_c * c0625
            b0 = b_l * c0375 + b_c * c0625
            a1 = a_l * c0125 + a_c * c0875
            b1 = b_l * c0125 + b_c * c0875
        else:
            a0 = a1 = a_c
            b0 = b1 = b_c
        if xs + 1 < ws:
            a2 = a_c * c0875 + a_r * c0125
            b2 = b_c * c0875 + b_r * c0125
            a3 = a_c * c0625 + a_r * c0375
            b3 = b_c * c0625 + b_r * c0375
        else:
            a2 = a3 = a_c
            b2 = b3 = b_c

        dst_row[x0] = a0 * gray[y, x0] + b0
        dst_row[x0 + 1] = a1 * gray[y, x0 + 1] + b1
        dst_row[x0 + 2] = a2 * gray[y, x0 + 2] + b2
        dst_row[x0 + 3] = a3 * gray[y, x0 + 3] + b3


@njit(nogil=True)
def fgf_upsample4_sobel_decision_rows(mean_a_sub, mean_b_sub, gray,
                                      val, sob_best, idx,
                                      frame_i, tie_eps, first_valid, y_start, y_stop):
    """Fused 4x bilinear upsample + 3x3 Sobel + GfgDecision update on rows [y_start:y_stop]."""
    if y_start >= y_stop:
        return
    h, w = gray.shape
    hs, ws = mean_a_sub.shape
    eps = np.float32(tie_eps)
    two = np.float32(2.0)

    r_prev = np.empty(w, np.float32)
    r_curr = np.empty(w, np.float32)
    r_next = np.empty(w, np.float32)

    _eval_upsample4_row(mean_a_sub, mean_b_sub, gray, hs, ws, _reflect101(y_start - 1, h), r_prev)
    _eval_upsample4_row(mean_a_sub, mean_b_sub, gray, hs, ws, y_start, r_curr)

    for y in range(y_start, y_stop):
        _eval_upsample4_row(mean_a_sub, mean_b_sub, gray, hs, ws, _reflect101(y + 1, h), r_next)
        # x == 0
        xm = _reflect101(-1, w)
        xp = _reflect101(1, w)
        gx = ((r_prev[xp] - r_prev[xm]) + (r_next[xp] - r_next[xm])) + two * (r_curr[xp] - r_curr[xm])
        gy = ((r_next[xm] + r_next[xp]) + two * r_next[0]) - ((r_prev[xm] + r_prev[xp]) + two * r_prev[0])
        s = abs(gx) + abs(gy)
        v = r_curr[0]
        if first_valid:
            val[y, 0] = v
            sob_best[y, 0] = s
            idx[y, 0] = frame_i
        else:
            diff = v - val[y, 0]
            if (diff > eps) or (abs(diff) <= eps and s > sob_best[y, 0]):
                val[y, 0] = v
                sob_best[y, 0] = s
                idx[y, 0] = frame_i

        # Interior x in [1, w - 1)
        if first_valid:
            for x in range(1, w - 1):
                gx = ((r_prev[x + 1] - r_prev[x - 1]) + (r_next[x + 1] - r_next[x - 1])) + two * (r_curr[x + 1] - r_curr[x - 1])
                gy = ((r_next[x - 1] + r_next[x + 1]) + two * r_next[x]) - ((r_prev[x - 1] + r_prev[x + 1]) + two * r_prev[x])
                val[y, x] = r_curr[x]
                sob_best[y, x] = abs(gx) + abs(gy)
                idx[y, x] = frame_i
        else:
            for x in range(1, w - 1):
                gx = ((r_prev[x + 1] - r_prev[x - 1]) + (r_next[x + 1] - r_next[x - 1])) + two * (r_curr[x + 1] - r_curr[x - 1])
                gy = ((r_next[x - 1] + r_next[x + 1]) + two * r_next[x]) - ((r_prev[x - 1] + r_prev[x + 1]) + two * r_prev[x])
                s = abs(gx) + abs(gy)
                v = r_curr[x]
                diff = v - val[y, x]
                if (diff > eps) or (abs(diff) <= eps and s > sob_best[y, x]):
                    val[y, x] = v
                    sob_best[y, x] = s
                    idx[y, x] = frame_i

        # x == w - 1
        wm1 = w - 1
        xm = _reflect101(wm1 - 1, w)
        xp = _reflect101(wm1 + 1, w)
        gx = ((r_prev[xp] - r_prev[xm]) + (r_next[xp] - r_next[xm])) + two * (r_curr[xp] - r_curr[xm])
        gy = ((r_next[xm] + r_next[xp]) + two * r_next[wm1]) - ((r_prev[xm] + r_prev[xp]) + two * r_prev[wm1])
        s = abs(gx) + abs(gy)
        v = r_curr[wm1]
        if first_valid:
            val[y, wm1] = v
            sob_best[y, wm1] = s
            idx[y, wm1] = frame_i
        else:
            diff = v - val[y, wm1]
            if (diff > eps) or (abs(diff) <= eps and s > sob_best[y, wm1]):
                val[y, wm1] = v
                sob_best[y, wm1] = s
                idx[y, wm1] = frame_i

        tmp = r_prev
        r_prev = r_curr
        r_curr = r_next
        r_next = tmp


@njit(nogil=True)
def gfg_sobel_decision_rows(afm, val, sob_best, idx, frame_i, tie_eps, first_valid, y0, y1):
    """Fused 3x3 Sobel + GfgDecision update directly on afm without gx/gy buffers."""
    if y0 >= y1:
        return
    h, w = afm.shape
    eps = np.float32(tie_eps)
    two = np.float32(2.0)

    for y in range(y0, y1):
        ym = _reflect101(y - 1, h)
        yp = _reflect101(y + 1, h)
        if w <= 2:
            for x in range(w):
                xm = _reflect101(x - 1, w)
                xp = _reflect101(x + 1, w)
                gx = ((afm[ym, xp] - afm[ym, xm]) + (afm[yp, xp] - afm[yp, xm])) + two * (afm[y, xp] - afm[y, xm])
                gy = ((afm[yp, xm] + afm[yp, xp]) + two * afm[yp, x]) - ((afm[ym, xm] + afm[ym, xp]) + two * afm[ym, x])
                s = abs(gx) + abs(gy)
                v = afm[y, x]
                if first_valid:
                    val[y, x] = v
                    sob_best[y, x] = s
                    idx[y, x] = frame_i
                else:
                    diff = v - val[y, x]
                    if (diff > eps) or (abs(diff) <= eps and s > sob_best[y, x]):
                        val[y, x] = v
                        sob_best[y, x] = s
                        idx[y, x] = frame_i
        else:
            # x == 0
            xm = _reflect101(-1, w)
            xp = 1
            gx = ((afm[ym, xp] - afm[ym, xm]) + (afm[yp, xp] - afm[yp, xm])) + two * (afm[y, xp] - afm[y, xm])
            gy = ((afm[yp, xm] + afm[yp, xp]) + two * afm[yp, 0]) - ((afm[ym, xm] + afm[ym, xp]) + two * afm[ym, 0])
            s = abs(gx) + abs(gy)
            v = afm[y, 0]
            if first_valid:
                val[y, 0] = v
                sob_best[y, 0] = s
                idx[y, 0] = frame_i
            else:
                diff = v - val[y, 0]
                if (diff > eps) or (abs(diff) <= eps and s > sob_best[y, 0]):
                    val[y, 0] = v
                    sob_best[y, 0] = s
                    idx[y, 0] = frame_i

            if first_valid:
                for x in range(1, w - 1):
                    gx = ((afm[ym, x + 1] - afm[ym, x - 1]) + (afm[yp, x + 1] - afm[yp, x - 1])) + two * (afm[y, x + 1] - afm[y, x - 1])
                    gy = ((afm[yp, x - 1] + afm[yp, x + 1]) + two * afm[yp, x]) - ((afm[ym, x - 1] + afm[ym, x + 1]) + two * afm[ym, x])
                    val[y, x] = afm[y, x]
                    sob_best[y, x] = abs(gx) + abs(gy)
                    idx[y, x] = frame_i
            else:
                for x in range(1, w - 1):
                    gx = ((afm[ym, x + 1] - afm[ym, x - 1]) + (afm[yp, x + 1] - afm[yp, x - 1])) + two * (afm[y, x + 1] - afm[y, x - 1])
                    gy = ((afm[yp, x - 1] + afm[yp, x + 1]) + two * afm[yp, x]) - ((afm[ym, x - 1] + afm[ym, x + 1]) + two * afm[ym, x])
                    s = abs(gx) + abs(gy)
                    v = afm[y, x]
                    diff = v - val[y, x]
                    if (diff > eps) or (abs(diff) <= eps and s > sob_best[y, x]):
                        val[y, x] = v
                        sob_best[y, x] = s
                        idx[y, x] = frame_i

            wm1 = w - 1
            xm = wm1 - 1
            xp = _reflect101(wm1 + 1, w)
            gx = ((afm[ym, xp] - afm[ym, xm]) + (afm[yp, xp] - afm[yp, xm])) + two * (afm[y, xp] - afm[y, xm])
            gy = ((afm[yp, xm] + afm[yp, xp]) + two * afm[yp, wm1]) - ((afm[ym, xm] + afm[ym, xp]) + two * afm[ym, wm1])
            s = abs(gx) + abs(gy)
            v = afm[y, wm1]
            if first_valid:
                val[y, wm1] = v
                sob_best[y, wm1] = s
                idx[y, wm1] = frame_i
            else:
                diff = v - val[y, wm1]
                if (diff > eps) or (abs(diff) <= eps and s > sob_best[y, wm1]):
                    val[y, wm1] = v
                    sob_best[y, wm1] = s
                    idx[y, wm1] = frame_i


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
def active_bboxes_i32_rows(idx, num_imgs, y0, y1):
    """Return per-frame pixel counts and active bounding boxes for rows [y0:y1]."""
    h, w = idx.shape
    wins = np.zeros(num_imgs, np.int64)
    ymin = np.full(num_imgs, h, np.int32)
    ymax = np.full(num_imgs, -1, np.int32)
    xmin = np.full(num_imgs, w, np.int32)
    xmax = np.full(num_imgs, -1, np.int32)
    for y in range(y0, y1):
        for x in range(w):
            k = idx[y, x]
            wins[k] += 1
            if y < ymin[k]:
                ymin[k] = y
            if y > ymax[k]:
                ymax[k] = y
            if x < xmin[k]:
                xmin[k] = x
            if x > xmax[k]:
                xmax[k] = x
    return wins, ymin, ymax, xmin, xmax


@njit(nogil=True)
def _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, ys, xs):
    if my0 <= ys < my1 and mx0 <= xs < mx1:
        return mean_a[ys - my0, xs - mx0], mean_b[ys - my0, xs - mx0]
    return np.float32(0.0), np.float32(0.0)


@njit(nogil=True)
def upsample4_accum_roi_u8_rows(raw, lut, ch, mean_a, mean_b,
                                hs, ws, my0, my1, mx0, mx1,
                                out_x0, out_x1, acc, sum_w, y_start, y_stop):
    """Fused 4x bilinear upsample of (mean_a, mean_b) + uint8 RGB accumulation on ROI rows."""
    if y_start >= y_stop:
        return
    is_gray = raw.ndim == 2
    zero = np.float32(0.0)
    c0625 = np.float32(0.625)
    c0375 = np.float32(0.375)
    c0875 = np.float32(0.875)
    c0125 = np.float32(0.125)
    xs_lo = out_x0 >> 2
    xs_hi = min(ws, (out_x1 + 3) >> 2)

    for y in range(y_start, y_stop):
        ys = y >> 2
        dy = y & 3
        if dy == 0:
            y0 = ys - 1 if ys > 0 else 0
            y1 = ys
            wy1 = c0625 if ys > 0 else np.float32(1.0)
        elif dy == 1:
            y0 = ys - 1 if ys > 0 else 0
            y1 = ys
            wy1 = c0875 if ys > 0 else np.float32(1.0)
        elif dy == 2:
            y0 = ys
            y1 = ys + 1 if ys + 1 < hs else hs - 1
            wy1 = c0125 if ys + 1 < hs else zero
        else:
            y0 = ys
            y1 = ys + 1 if ys + 1 < hs else hs - 1
            wy1 = c0375 if ys + 1 < hs else zero
        wy0 = np.float32(1.0) - wy1

        for xs in range(xs_lo, xs_hi):
            xm = xs - 1 if xs > 0 else 0
            xp = xs + 1 if xs + 1 < ws else ws - 1

            a0l, b0l = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y0, xm)
            a1l, b1l = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y1, xm)
            a0c, b0c = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y0, xs)
            a1c, b1c = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y1, xs)
            a0r, b0r = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y0, xp)
            a1r, b1r = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y1, xp)

            a_l = a0l * wy0 + a1l * wy1
            b_l = b0l * wy0 + b1l * wy1
            a_c = a0c * wy0 + a1c * wy1
            b_c = b0c * wy0 + b1c * wy1
            a_r = a0r * wy0 + a1r * wy1
            b_r = b0r * wy0 + b1r * wy1

            if (a_l == zero and b_l == zero and a_c == zero and b_c == zero
                    and a_r == zero and b_r == zero):
                continue

            if xs > 0:
                a0 = a_l * c0375 + a_c * c0625
                b0 = b_l * c0375 + b_c * c0625
                a1 = a_l * c0125 + a_c * c0875
                b1 = b_l * c0125 + b_c * c0875
            else:
                a0 = a1 = a_c
                b0 = b1 = b_c
            if xs + 1 < ws:
                a2 = a_c * c0875 + a_r * c0125
                b2 = b_c * c0875 + b_r * c0125
                a3 = a_c * c0625 + a_r * c0375
                b3 = b_c * c0625 + b_r * c0375
            else:
                a2 = a3 = a_c
                b2 = b3 = b_c

            x_base = xs * 4
            for dx in range(4):
                x = x_base + dx
                if x < out_x0 or x >= out_x1:
                    continue
                if dx == 0:
                    aa, bb = a0, b0
                elif dx == 1:
                    aa, bb = a1, b1
                elif dx == 2:
                    aa, bb = a2, b2
                else:
                    aa, bb = a3, b3
                if aa == zero and bb == zero:
                    continue
                if is_gray:
                    v = lut[raw[y, x]]
                    wt = aa * v + bb
                    sum_w[y, x] += wt
                    vw = v * wt
                    acc[y, x, 0] += vw
                    acc[y, x, 1] += vw
                    acc[y, x, 2] += vw
                else:
                    c0 = lut[raw[y, x, 0]]
                    c1 = lut[raw[y, x, 1]]
                    c2 = lut[raw[y, x, 2]]
                    g = c0 if ch == 0 else (c1 if ch == 1 else c2)
                    wt = aa * g + bb
                    sum_w[y, x] += wt
                    acc[y, x, 0] += c0 * wt
                    acc[y, x, 1] += c1 * wt
                    acc[y, x, 2] += c2 * wt


@njit(nogil=True)
def upsample4_accum_roi_u16_rows(raw, ch, mean_a, mean_b,
                                 hs, ws, my0, my1, mx0, mx1,
                                 out_x0, out_x1, acc, sum_w, y_start, y_stop):
    """Fused 4x bilinear upsample of (mean_a, mean_b) + uint16 RGB accumulation on ROI rows."""
    if y_start >= y_stop:
        return
    is_gray = raw.ndim == 2
    zero = np.float32(0.0)
    scale = np.float32(255.0 / 65535.0)
    inv255 = np.float32(255.0)
    c0625 = np.float32(0.625)
    c0375 = np.float32(0.375)
    c0875 = np.float32(0.875)
    c0125 = np.float32(0.125)
    xs_lo = out_x0 >> 2
    xs_hi = min(ws, (out_x1 + 3) >> 2)

    for y in range(y_start, y_stop):
        ys = y >> 2
        dy = y & 3
        if dy == 0:
            y0 = ys - 1 if ys > 0 else 0
            y1 = ys
            wy1 = c0625 if ys > 0 else np.float32(1.0)
        elif dy == 1:
            y0 = ys - 1 if ys > 0 else 0
            y1 = ys
            wy1 = c0875 if ys > 0 else np.float32(1.0)
        elif dy == 2:
            y0 = ys
            y1 = ys + 1 if ys + 1 < hs else hs - 1
            wy1 = c0125 if ys + 1 < hs else zero
        else:
            y0 = ys
            y1 = ys + 1 if ys + 1 < hs else hs - 1
            wy1 = c0375 if ys + 1 < hs else zero
        wy0 = np.float32(1.0) - wy1

        for xs in range(xs_lo, xs_hi):
            xm = xs - 1 if xs > 0 else 0
            xp = xs + 1 if xs + 1 < ws else ws - 1

            a0l, b0l = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y0, xm)
            a1l, b1l = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y1, xm)
            a0c, b0c = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y0, xs)
            a1c, b1c = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y1, xs)
            a0r, b0r = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y0, xp)
            a1r, b1r = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y1, xp)

            a_l = a0l * wy0 + a1l * wy1
            b_l = b0l * wy0 + b1l * wy1
            a_c = a0c * wy0 + a1c * wy1
            b_c = b0c * wy0 + b1c * wy1
            a_r = a0r * wy0 + a1r * wy1
            b_r = b0r * wy0 + b1r * wy1

            if (a_l == zero and b_l == zero and a_c == zero and b_c == zero
                    and a_r == zero and b_r == zero):
                continue

            if xs > 0:
                a0 = a_l * c0375 + a_c * c0625
                b0 = b_l * c0375 + b_c * c0625
                a1 = a_l * c0125 + a_c * c0875
                b1 = b_l * c0125 + b_c * c0875
            else:
                a0 = a1 = a_c
                b0 = b1 = b_c
            if xs + 1 < ws:
                a2 = a_c * c0875 + a_r * c0125
                b2 = b_c * c0875 + b_r * c0125
                a3 = a_c * c0625 + a_r * c0375
                b3 = b_c * c0625 + b_r * c0375
            else:
                a2 = a3 = a_c
                b2 = b3 = b_c

            x_base = xs * 4
            for dx in range(4):
                x = x_base + dx
                if x < out_x0 or x >= out_x1:
                    continue
                if dx == 0:
                    aa, bb = a0, b0
                elif dx == 1:
                    aa, bb = a1, b1
                elif dx == 2:
                    aa, bb = a2, b2
                else:
                    aa, bb = a3, b3
                if aa == zero and bb == zero:
                    continue
                if is_gray:
                    v = (np.float32(raw[y, x]) * scale) / inv255
                    wt = aa * v + bb
                    sum_w[y, x] += wt
                    vw = v * wt
                    acc[y, x, 0] += vw
                    acc[y, x, 1] += vw
                    acc[y, x, 2] += vw
                else:
                    c0 = (np.float32(raw[y, x, 0]) * scale) / inv255
                    c1 = (np.float32(raw[y, x, 1]) * scale) / inv255
                    c2 = (np.float32(raw[y, x, 2]) * scale) / inv255
                    g = c0 if ch == 0 else (c1 if ch == 1 else c2)
                    wt = aa * g + bb
                    sum_w[y, x] += wt
                    acc[y, x, 0] += c0 * wt
                    acc[y, x, 1] += c1 * wt
                    acc[y, x, 2] += c2 * wt


@njit(nogil=True)
def upsample4_accum_roi_f32_rows(raw, ch, mean_a, mean_b,
                                 hs, ws, my0, my1, mx0, mx1,
                                 out_x0, out_x1, acc, sum_w, y_start, y_stop):
    """Fused 4x bilinear upsample of (mean_a, mean_b) + float RGB accumulation on ROI rows."""
    if y_start >= y_stop:
        return
    is_gray = raw.ndim == 2
    zero = np.float32(0.0)
    inv255 = np.float32(255.0)
    c0625 = np.float32(0.625)
    c0375 = np.float32(0.375)
    c0875 = np.float32(0.875)
    c0125 = np.float32(0.125)
    xs_lo = out_x0 >> 2
    xs_hi = min(ws, (out_x1 + 3) >> 2)

    for y in range(y_start, y_stop):
        ys = y >> 2
        dy = y & 3
        if dy == 0:
            y0 = ys - 1 if ys > 0 else 0
            y1 = ys
            wy1 = c0625 if ys > 0 else np.float32(1.0)
        elif dy == 1:
            y0 = ys - 1 if ys > 0 else 0
            y1 = ys
            wy1 = c0875 if ys > 0 else np.float32(1.0)
        elif dy == 2:
            y0 = ys
            y1 = ys + 1 if ys + 1 < hs else hs - 1
            wy1 = c0125 if ys + 1 < hs else zero
        else:
            y0 = ys
            y1 = ys + 1 if ys + 1 < hs else hs - 1
            wy1 = c0375 if ys + 1 < hs else zero
        wy0 = np.float32(1.0) - wy1

        for xs in range(xs_lo, xs_hi):
            xm = xs - 1 if xs > 0 else 0
            xp = xs + 1 if xs + 1 < ws else ws - 1

            a0l, b0l = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y0, xm)
            a1l, b1l = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y1, xm)
            a0c, b0c = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y0, xs)
            a1c, b1c = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y1, xs)
            a0r, b0r = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y0, xp)
            a1r, b1r = _sample_mab_roi(mean_a, mean_b, my0, my1, mx0, mx1, y1, xp)

            a_l = a0l * wy0 + a1l * wy1
            b_l = b0l * wy0 + b1l * wy1
            a_c = a0c * wy0 + a1c * wy1
            b_c = b0c * wy0 + b1c * wy1
            a_r = a0r * wy0 + a1r * wy1
            b_r = b0r * wy0 + b1r * wy1

            if (a_l == zero and b_l == zero and a_c == zero and b_c == zero
                    and a_r == zero and b_r == zero):
                continue

            if xs > 0:
                a0 = a_l * c0375 + a_c * c0625
                b0 = b_l * c0375 + b_c * c0625
                a1 = a_l * c0125 + a_c * c0875
                b1 = b_l * c0125 + b_c * c0875
            else:
                a0 = a1 = a_c
                b0 = b1 = b_c
            if xs + 1 < ws:
                a2 = a_c * c0875 + a_r * c0125
                b2 = b_c * c0875 + b_r * c0125
                a3 = a_c * c0625 + a_r * c0375
                b3 = b_c * c0625 + b_r * c0375
            else:
                a2 = a3 = a_c
                b2 = b3 = b_c

            x_base = xs * 4
            for dx in range(4):
                x = x_base + dx
                if x < out_x0 or x >= out_x1:
                    continue
                if dx == 0:
                    aa, bb = a0, b0
                elif dx == 1:
                    aa, bb = a1, b1
                elif dx == 2:
                    aa, bb = a2, b2
                else:
                    aa, bb = a3, b3
                if aa == zero and bb == zero:
                    continue
                if is_gray:
                    v = np.float32(raw[y, x]) / inv255
                    wt = aa * v + bb
                    sum_w[y, x] += wt
                    vw = v * wt
                    acc[y, x, 0] += vw
                    acc[y, x, 1] += vw
                    acc[y, x, 2] += vw
                else:
                    c0 = np.float32(raw[y, x, 0]) / inv255
                    c1 = np.float32(raw[y, x, 1]) / inv255
                    c2 = np.float32(raw[y, x, 2]) / inv255
                    g = c0 if ch == 0 else (c1 if ch == 1 else c2)
                    wt = aa * g + bb
                    sum_w[y, x] += wt
                    acc[y, x, 0] += c0 * wt
                    acc[y, x, 1] += c1 * wt
                    acc[y, x, 2] += c2 * wt


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
