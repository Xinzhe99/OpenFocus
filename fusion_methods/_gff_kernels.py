"""Exact CPU kernels for Guided Filter fusion.

Uses bounded float64 prefix/sliding sums, active bounding-box ROI cropping,
zero-mask early skips, and row-strip parallel kernels to reproduce the reference
OpenCV/NumPy arithmetic bit-for-bit across all CPU cores with bounded RAM.
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
def sum_u8_to_f32_rows(raw, lut, out, y0, y1):
    """Sum normalized BGR channels of uint8 2D/3-ch/4-ch rows [y0:y1] into out."""
    w = out.shape[1]
    if raw.ndim == 2:
        for y in range(y0, y1):
            for x in range(w):
                v = lut[raw[y, x]]
                out[y, x] = (v + v) + v
    else:
        for y in range(y0, y1):
            for x in range(w):
                b = lut[raw[y, x, 0]]
                g = lut[raw[y, x, 1]]
                r = lut[raw[y, x, 2]]
                out[y, x] = (b + g) + r


@njit(nogil=True)
def sum_u8_to_f32(raw, lut, out):
    """Sum normalized BGR channels of a uint8 2D/3-ch/4-ch frame into out."""
    sum_u8_to_f32_rows(raw, lut, out, 0, out.shape[0])


@njit(nogil=True)
def sum_u16_to_f32_rows(raw, out, y0, y1):
    """Sum normalized BGR channels of uint16 2D/3-ch/4-ch rows [y0:y1] into out."""
    w = out.shape[1]
    scale = np.float32(255.0 / 65535.0)
    inv255 = np.float32(255.0)
    if raw.ndim == 2:
        for y in range(y0, y1):
            for x in range(w):
                v = (np.float32(raw[y, x]) * scale) / inv255
                out[y, x] = (v + v) + v
    else:
        for y in range(y0, y1):
            for x in range(w):
                b = (np.float32(raw[y, x, 0]) * scale) / inv255
                g = (np.float32(raw[y, x, 1]) * scale) / inv255
                r = (np.float32(raw[y, x, 2]) * scale) / inv255
                out[y, x] = (b + g) + r


@njit(nogil=True)
def sum_u16_to_f32(raw, out):
    """Sum normalized BGR channels of a uint16 2D/3-ch/4-ch frame into out."""
    sum_u16_to_f32_rows(raw, out, 0, out.shape[0])


@njit(nogil=True)
def sum_f32_to_f32_rows(raw, out, y0, y1):
    """Sum normalized BGR channels of float 2D/3-ch/4-ch rows [y0:y1] into out."""
    w = out.shape[1]
    inv255 = np.float32(255.0)
    if raw.ndim == 2:
        for y in range(y0, y1):
            for x in range(w):
                v = np.float32(raw[y, x]) / inv255
                out[y, x] = (v + v) + v
    else:
        for y in range(y0, y1):
            for x in range(w):
                b = np.float32(raw[y, x, 0]) / inv255
                g = np.float32(raw[y, x, 1]) / inv255
                r = np.float32(raw[y, x, 2]) / inv255
                out[y, x] = (b + g) + r


@njit(nogil=True)
def sum_f32_to_f32(raw, out):
    """Sum normalized BGR channels of a float 2D/3-ch/4-ch frame into out."""
    sum_f32_to_f32_rows(raw, out, 0, out.shape[0])


@njit(nogil=True)
def prep_u8_bgr_roi_rows(raw, lut, out, roi_y0, roi_x0, ry0, ry1):
    """Convert uint8 ROI rows [ry0:ry1] to float32 [0, 1] BGR via LUT."""
    rw = out.shape[1]
    if raw.ndim == 2:
        for ry in range(ry0, ry1):
            sy = roi_y0 + ry
            for rx in range(rw):
                v = lut[raw[sy, roi_x0 + rx]]
                out[ry, rx, 0] = v
                out[ry, rx, 1] = v
                out[ry, rx, 2] = v
    else:
        for ry in range(ry0, ry1):
            sy = roi_y0 + ry
            for rx in range(rw):
                sx = roi_x0 + rx
                out[ry, rx, 0] = lut[raw[sy, sx, 0]]
                out[ry, rx, 1] = lut[raw[sy, sx, 1]]
                out[ry, rx, 2] = lut[raw[sy, sx, 2]]


@njit(nogil=True)
def prep_u16_bgr_roi_rows(raw, out, roi_y0, roi_x0, ry0, ry1):
    """Convert uint16 ROI rows [ry0:ry1] to float32 [0, 1] BGR."""
    rw = out.shape[1]
    scale = np.float32(255.0 / 65535.0)
    inv255 = np.float32(255.0)
    if raw.ndim == 2:
        for ry in range(ry0, ry1):
            sy = roi_y0 + ry
            for rx in range(rw):
                v = (np.float32(raw[sy, roi_x0 + rx]) * scale) / inv255
                out[ry, rx, 0] = v
                out[ry, rx, 1] = v
                out[ry, rx, 2] = v
    else:
        for ry in range(ry0, ry1):
            sy = roi_y0 + ry
            for rx in range(rw):
                sx = roi_x0 + rx
                out[ry, rx, 0] = (np.float32(raw[sy, sx, 0]) * scale) / inv255
                out[ry, rx, 1] = (np.float32(raw[sy, sx, 1]) * scale) / inv255
                out[ry, rx, 2] = (np.float32(raw[sy, sx, 2]) * scale) / inv255


@njit(nogil=True)
def prep_f32_bgr_roi_rows(raw, out, roi_y0, roi_x0, ry0, ry1):
    """Convert float ROI rows [ry0:ry1] to float32 [0, 1] BGR."""
    rw = out.shape[1]
    inv255 = np.float32(255.0)
    if raw.ndim == 2:
        for ry in range(ry0, ry1):
            sy = roi_y0 + ry
            for rx in range(rw):
                v = np.float32(raw[sy, roi_x0 + rx]) / inv255
                out[ry, rx, 0] = v
                out[ry, rx, 1] = v
                out[ry, rx, 2] = v
    else:
        for ry in range(ry0, ry1):
            sy = roi_y0 + ry
            for rx in range(rw):
                sx = roi_x0 + rx
                out[ry, rx, 0] = np.float32(raw[sy, sx, 0]) / inv255
                out[ry, rx, 1] = np.float32(raw[sy, sx, 1]) / inv255
                out[ry, rx, 2] = np.float32(raw[sy, sx, 2]) / inv255


@njit(nogil=True)
def prep_u8_bgr(raw, lut, out):
    """Convert uint8 2D/3-ch/4-ch frame to float32 [0, 1] BGR via LUT."""
    prep_u8_bgr_roi_rows(raw, lut, out, 0, 0, 0, out.shape[0])


@njit(nogil=True)
def argmax_update_rows(sal, best, max_indices, k, start, stop):
    """Update winner index using strict `>` to preserve first-frame ties."""
    width = sal.shape[1]
    for y in range(start, stop):
        for x in range(width):
            val = sal[y, x]
            if val > best[y, x]:
                best[y, x] = val
                max_indices[y, x] = k


@njit(nogil=True)
def active_bboxes_rows(max_indices, num_images, y0, y1):
    """Return per-frame pixel counts and active bounding boxes for rows [y0:y1]."""
    h, w = max_indices.shape
    wins = np.zeros(num_images, np.int64)
    ymin = np.full(num_images, h, np.int32)
    ymax = np.full(num_images, -1, np.int32)
    xmin = np.full(num_images, w, np.int32)
    xmax = np.full(num_images, -1, np.int32)
    for y in range(y0, y1):
        for x in range(w):
            k = max_indices[y, x]
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
def active_bboxes(max_indices, num_images):
    """Return per-frame pixel counts and active bounding boxes."""
    return active_bboxes_rows(max_indices, num_images, 0, max_indices.shape[0])


@njit(nogil=True)
def _box3_step_row_roi(img_roi, sy_local, xidx_local, k, rw,
                       p0, p1, p2, ring, slot, csum, replace):
    """Update ring row and vertical column sums for one 3-channel ROI row."""
    a0 = a1 = a2 = 0.0
    p0[0] = p1[0] = p2[0] = 0.0
    for i in range(xidx_local.size):
        xx = xidx_local[i]
        a0 += float(img_roi[sy_local, xx, 0])
        a1 += float(img_roi[sy_local, xx, 1])
        a2 += float(img_roi[sy_local, xx, 2])
        p0[i + 1] = a0
        p1[i + 1] = a1
        p2[i + 1] = a2
    for ix in range(rw):
        s0 = p0[ix + k] - p0[ix]
        s1 = p1[ix + k] - p1[ix]
        s2 = p2[ix + k] - p2[ix]
        if replace:
            csum[ix, 0] += s0 - ring[slot, ix, 0]
            csum[ix, 1] += s1 - ring[slot, ix, 1]
            csum[ix, 2] += s2 - ring[slot, ix, 2]
        else:
            csum[ix, 0] += s0
            csum[ix, 1] += s1
            csum[ix, 2] += s2
        ring[slot, ix, 0] = s0
        ring[slot, ix, 1] = s1
        ring[slot, ix, 2] = s2


@njit(nogil=True)
def box3_exact_region_roi_rows(img_roi, full_h, full_w, roi_y0, roi_x0,
                               rad, y0, x0, rw, out, ry_start, ry_stop):
    """Exact cv2.blur(..., BORDER_REFLECT) for rows [ry_start:ry_stop] of [y0:y1, x0:x1]."""
    if ry_start >= ry_stop:
        return
    if rad <= 0:
        for ry in range(ry_start, ry_stop):
            sy = y0 + ry - roi_y0
            for rx in range(rw):
                sx = x0 + rx - roi_x0
                out[ry, rx, 0] = img_roi[sy, sx, 0]
                out[ry, rx, 1] = img_roi[sy, sx, 1]
                out[ry, rx, 2] = img_roi[sy, sx, 2]
        return

    k = 2 * rad + 1
    inv = 1.0 / (k * k)
    ring = np.empty((k, rw, 3), np.float64)
    csum = np.zeros((rw, 3), np.float64)
    p0 = np.empty(rw + k, np.float64)
    p1 = np.empty(rw + k, np.float64)
    p2 = np.empty(rw + k, np.float64)
    xidx_local = np.empty(rw + 2 * rad, np.int32)
    for i in range(rw + 2 * rad):
        xidx_local[i] = _reflect(x0 + i - rad, full_w) - roi_x0

    for d in range(-rad, rad + 1):
        _box3_step_row_roi(
            img_roi, _reflect(y0 + ry_start + d, full_h) - roi_y0, xidx_local,
            k, rw, p0, p1, p2, ring, d + rad, csum, False,
        )

    for ix in range(rw):
        out[ry_start, ix, 0] = np.float32(csum[ix, 0] * inv)
        out[ry_start, ix, 1] = np.float32(csum[ix, 1] * inv)
        out[ry_start, ix, 2] = np.float32(csum[ix, 2] * inv)

    head = 0
    for ry in range(ry_start + 1, ry_stop):
        _box3_step_row_roi(
            img_roi, _reflect(y0 + ry + rad, full_h) - roi_y0, xidx_local,
            k, rw, p0, p1, p2, ring, head, csum, True,
        )
        head += 1
        if head == k:
            head = 0
        for ix in range(rw):
            out[ry, ix, 0] = np.float32(csum[ix, 0] * inv)
            out[ry, ix, 1] = np.float32(csum[ix, 1] * inv)
            out[ry, ix, 2] = np.float32(csum[ix, 2] * inv)


@njit(nogil=True)
def box3_exact_region(img, rad, y0, y1, x0, x1):
    """Exact cv2.blur(img, (2*rad+1, 2*rad+1), BORDER_REFLECT) on [y0:y1, x0:x1]."""
    h, w, _ = img.shape
    rh = y1 - y0
    rw = x1 - x0
    out = np.empty((rh, rw, 3), np.float32)
    box3_exact_region_roi_rows(img, h, w, 0, 0, rad, y0, x0, rw, out, 0, rh)
    return out


@njit(nogil=True)
def _gf_stage1_row_roi(gray_roi, max_indices, frame_k, sy_global, roi_y0, roi_x0,
                       xidx_global, k, aw, pI, pp, pIp, pII, ring4, slot, csum4, replace):
    """Update stage-1 4-plane ring row and column sums for one ROI row."""
    aI = ap = aIp = aII = 0.0
    pI[0] = pp[0] = pIp[0] = pII[0] = 0.0
    sy_local = sy_global - roi_y0
    for i in range(xidx_global.size):
        gx = xidx_global[i]
        iv = gray_roi[sy_local, gx - roi_x0]
        aI += float(iv)
        aII += float(iv * iv)
        if max_indices[sy_global, gx] == frame_k:
            ap += 1.0
            aIp += float(iv)
        pI[i + 1] = aI
        pp[i + 1] = ap
        pIp[i + 1] = aIp
        pII[i + 1] = aII
    for ix in range(aw):
        s0 = pI[ix + k] - pI[ix]
        s1 = pp[ix + k] - pp[ix]
        s2 = pIp[ix + k] - pIp[ix]
        s3 = pII[ix + k] - pII[ix]
        if replace:
            csum4[ix, 0] += s0 - ring4[slot, ix, 0]
            csum4[ix, 1] += s1 - ring4[slot, ix, 1]
            csum4[ix, 2] += s2 - ring4[slot, ix, 2]
            csum4[ix, 3] += s3 - ring4[slot, ix, 3]
        else:
            csum4[ix, 0] += s0
            csum4[ix, 1] += s1
            csum4[ix, 2] += s2
            csum4[ix, 3] += s3
        ring4[slot, ix, 0] = s0
        ring4[slot, ix, 1] = s1
        ring4[slot, ix, 2] = s2
        ring4[slot, ix, 3] = s3


@njit(nogil=True)
def _gf_stage1_emit_ab(csum4, inv, eps32, aw, a_buf, b_buf, row_out):
    """Compute local linear coefficients (a, b) for one row with zero-p skip."""
    zero = np.float32(0.0)
    for ix in range(aw):
        sp = csum4[ix, 1]
        if sp == 0.0:
            a_buf[row_out, ix] = zero
            b_buf[row_out, ix] = zero
        else:
            mI = np.float32(csum4[ix, 0] * inv)
            mp = np.float32(sp * inv)
            mIp = np.float32(csum4[ix, 2] * inv)
            mII = np.float32(csum4[ix, 3] * inv)
            a = (mIp - mI * mp) / ((mII - mI * mI) + eps32)
            a_buf[row_out, ix] = a
            b_buf[row_out, ix] = mp - a * mI


@njit(nogil=True)
def gf_stage1_roi_rows(gray_roi, max_indices, frame_k, rad, eps,
                       full_h, full_w, roi_y0, roi_x0,
                       ay0, ax0, aw, a_buf, b_buf, ry_start, ry_stop):
    """Exact Stage-1 guided-filter (a, b) for rows [ry_start:ry_stop] of [ay0:ay1, ax0:ax1]."""
    if ry_start >= ry_stop:
        return
    k = 2 * rad + 1
    inv = 1.0 / (k * k)
    eps32 = np.float32(eps)

    ring4 = np.empty((k, aw, 4), np.float64)
    csum4 = np.zeros((aw, 4), np.float64)
    pI = np.empty(aw + k, np.float64)
    pp = np.empty(aw + k, np.float64)
    pIp = np.empty(aw + k, np.float64)
    pII = np.empty(aw + k, np.float64)
    xidx_global = np.empty(aw + 2 * rad, np.int32)
    for i in range(aw + 2 * rad):
        xidx_global[i] = _reflect101(ax0 + i - rad, full_w)

    for d in range(-rad, rad + 1):
        _gf_stage1_row_roi(
            gray_roi, max_indices, frame_k, _reflect101(ay0 + ry_start + d, full_h),
            roi_y0, roi_x0, xidx_global, k, aw, pI, pp, pIp, pII,
            ring4, d + rad, csum4, False,
        )
    _gf_stage1_emit_ab(csum4, inv, eps32, aw, a_buf, b_buf, ry_start)

    head = 0
    for ry in range(ry_start + 1, ry_stop):
        _gf_stage1_row_roi(
            gray_roi, max_indices, frame_k, _reflect101(ay0 + ry + rad, full_h),
            roi_y0, roi_x0, xidx_global, k, aw, pI, pp, pIp, pII,
            ring4, head, csum4, True,
        )
        head += 1
        if head == k:
            head = 0
        _gf_stage1_emit_ab(csum4, inv, eps32, aw, a_buf, b_buf, ry)


@njit(nogil=True)
def _gf_stage2_row(a_buf, b_buf, active_row, ry, xloc, k, qw,
                   pa, pb, ring2, slot, csum2, replace):
    """Update stage-2 2-plane ring row and column sums for one row."""
    if active_row:
        aa = ab = 0.0
        pa[0] = pb[0] = 0.0
        for i in range(xloc.size):
            lx = xloc[i]
            if lx >= 0:
                aa += float(a_buf[ry, lx])
                ab += float(b_buf[ry, lx])
            pa[i + 1] = aa
            pb[i + 1] = ab
        for ix in range(qw):
            sa = pa[ix + k] - pa[ix]
            sb = pb[ix + k] - pb[ix]
            if replace:
                csum2[ix, 0] += sa - ring2[slot, ix, 0]
                csum2[ix, 1] += sb - ring2[slot, ix, 1]
            else:
                csum2[ix, 0] += sa
                csum2[ix, 1] += sb
            ring2[slot, ix, 0] = sa
            ring2[slot, ix, 1] = sb
    elif replace:
        for ix in range(qw):
            csum2[ix, 0] -= ring2[slot, ix, 0]
            csum2[ix, 1] -= ring2[slot, ix, 1]
            ring2[slot, ix, 0] = 0.0
            ring2[slot, ix, 1] = 0.0
    else:
        for ix in range(qw):
            ring2[slot, ix, 0] = 0.0
            ring2[slot, ix, 1] = 0.0


@njit(nogil=True)
def gf_stage2_roi_rows(gray_roi, a_buf, b_buf, rad,
                       full_h, full_w, roi_y0, roi_x0,
                       ay0, ay1, ax0, ax1,
                       qy0, qx0, qw, q_out, ry_start, ry_stop):
    """Exact Stage-2 guided-filter weight q for rows [ry_start:ry_stop] of [qy0:qy1, qx0:qx1]."""
    if ry_start >= ry_stop:
        return
    k = 2 * rad + 1
    inv = 1.0 / (k * k)

    ring2 = np.empty((k, qw, 2), np.float64)
    csum2 = np.zeros((qw, 2), np.float64)
    pa = np.empty(qw + k, np.float64)
    pb = np.empty(qw + k, np.float64)
    xloc = np.empty(qw + 2 * rad, np.int32)
    for i in range(qw + 2 * rad):
        fx = _reflect101(qx0 + i - rad, full_w)
        xloc[i] = fx - ax0 if (ax0 <= fx < ax1) else -1

    for d in range(-rad, rad + 1):
        fy = _reflect101(qy0 + ry_start + d, full_h)
        active = ay0 <= fy < ay1
        _gf_stage2_row(a_buf, b_buf, active, fy - ay0 if active else 0,
                       xloc, k, qw, pa, pb, ring2, d + rad, csum2, False)

    gy_local = qy0 + ry_start - roi_y0
    gx_off = qx0 - roi_x0
    for ix in range(qw):
        q_out[ry_start, ix] = (np.float32(csum2[ix, 0] * inv) * gray_roi[gy_local, gx_off + ix]
                               + np.float32(csum2[ix, 1] * inv))

    head = 0
    for ry_q in range(ry_start + 1, ry_stop):
        fy = _reflect101(qy0 + ry_q + rad, full_h)
        active = ay0 <= fy < ay1
        _gf_stage2_row(a_buf, b_buf, active, fy - ay0 if active else 0,
                       xloc, k, qw, pa, pb, ring2, head, csum2, True)
        head += 1
        if head == k:
            head = 0
        gy_local = qy0 + ry_q - roi_y0
        for ix in range(qw):
            q_out[ry_q, ix] = (np.float32(csum2[ix, 0] * inv) * gray_roi[gy_local, gx_off + ix]
                               + np.float32(csum2[ix, 1] * inv))


@njit(nogil=True)
def gf_weight_exact_region(gray, max_indices, frame_k, rad, eps,
                           ymin, ymax, xmin, xmax):
    """Exact two-pass guided-filter weight restricted to active support."""
    h, w = gray.shape
    k = 2 * rad + 1
    if h <= k:
        ay0, ay1, qy0, qy1 = 0, h, 0, h
    else:
        ay0 = max(0, ymin - rad)
        ay1 = min(h, ymax + rad + 1)
        qy0 = max(0, ay0 - rad)
        qy1 = min(h, ay1 + rad)

    if w <= k:
        ax0, ax1, qx0, qx1 = 0, w, 0, w
    else:
        ax0 = max(0, xmin - rad)
        ax1 = min(w, xmax + rad + 1)
        qx0 = max(0, ax0 - rad)
        qx1 = min(w, ax1 + rad)

    ah = ay1 - ay0
    aw = ax1 - ax0
    qh = qy1 - qy0
    qw = qx1 - qx0

    a_buf = np.empty((ah, aw), np.float32)
    b_buf = np.empty((ah, aw), np.float32)
    gf_stage1_roi_rows(gray, max_indices, frame_k, rad, eps,
                       h, w, 0, 0, ay0, ax0, aw, a_buf, b_buf, 0, ah)
    q_out = np.empty((qh, qw), np.float32)
    gf_stage2_roi_rows(gray, a_buf, b_buf, rad, h, w, 0, 0,
                       ay0, ay1, ax0, ax1, qy0, qx0, qw, q_out, 0, qh)
    return q_out, qy0, qy1, qx0, qx1


@njit(nogil=True)
def accumulate_frame_region_roi_rows(bgr_roi, roi_y0, roi_x0,
                                     base_patch, base_y0, base_x0,
                                     wb, by0, by1, bx0, bx1,
                                     wd, dy0, dy1, dx0, dx1,
                                     bn, bd, dn, dd, y_start, y_stop):
    """Accumulate base and detail weighted contributions for global rows [y_start:y_stop]."""
    zero = np.float32(0.0)
    b_lo = max(by0, y_start)
    b_hi = min(by1, y_stop)
    for y in range(b_lo, b_hi):
        wy = y - by0
        py = y - base_y0
        for x in range(bx0, bx1):
            w = wb[wy, x - bx0]
            if w != zero:
                px = x - base_x0
                bd[y, x] += w
                bn[y, x, 0] += base_patch[py, px, 0] * w
                bn[y, x, 1] += base_patch[py, px, 1] * w
                bn[y, x, 2] += base_patch[py, px, 2] * w

    d_lo = max(dy0, y_start)
    d_hi = min(dy1, y_stop)
    for y in range(d_lo, d_hi):
        wy = y - dy0
        py = y - base_y0
        ry = y - roi_y0
        for x in range(dx0, dx1):
            w = wd[wy, x - dx0]
            if w != zero:
                px = x - base_x0
                rx = x - roi_x0
                dd[y, x] += w
                dn[y, x, 0] += (bgr_roi[ry, rx, 0] - base_patch[py, px, 0]) * w
                dn[y, x, 1] += (bgr_roi[ry, rx, 1] - base_patch[py, px, 1]) * w
                dn[y, x, 2] += (bgr_roi[ry, rx, 2] - base_patch[py, px, 2]) * w


@njit(nogil=True)
def accumulate_frame_region(img, base_patch, base_y0, base_x0,
                            wb, by0, by1, bx0, bx1,
                            wd, dy0, dy1, dx0, dx1,
                            bn, bd, dn, dd):
    """Accumulate base and detail weighted contributions in frame order."""
    accumulate_frame_region_roi_rows(
        img, 0, 0, base_patch, base_y0, base_x0,
        wb, by0, by1, bx0, bx1, wd, dy0, dy1, dx0, dx1,
        bn, bd, dn, dd, min(by0, dy0), max(by1, dy1),
    )


@njit(nogil=True)
def reconstruct_rows(bn, bd, dn, dd, out, start, stop):
    """Normalize and clamp [start:stop] rows into uint8 or float32 output."""
    width = out.shape[1]
    eps = np.float32(1e-6)
    scale = np.float32(255.0)
    zero = np.float32(0.0)
    for y in range(start, stop):
        for x in range(width):
            bden = bd[y, x]
            if bden < eps:
                bden = eps
            dden = dd[y, x]
            if dden < eps:
                dden = eps
            for c in range(3):
                val = ((bn[y, x, c] / bden) + (dn[y, x, c] / dden)) * scale
                if val < zero:
                    val = zero
                elif val > scale:
                    val = scale
                out[y, x, c] = val
