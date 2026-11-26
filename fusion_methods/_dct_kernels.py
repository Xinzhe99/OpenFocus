"""Compiled Numba CPU kernels for DCT block-variance multi-focus fusion."""
import numpy as np

from fusion_methods._depth_kernel import njit


@njit(nogil=True)
def block_variance_u8_band(gray, orig_h, orig_w, block_size, out_var, by_start, by_stop):
    """Compute per-block spatial variance for block-rows [by_start, by_stop)."""
    map_w = out_var.shape[1]
    full_by = orig_h // block_size
    full_bx = orig_w // block_size
    inv = np.float32(1.0) / np.float32(block_size * block_size)
    max_y = orig_h - 1
    max_x = orig_w - 1

    if block_size <= 16:
        sx = np.empty(map_w, dtype=np.uint32)
        sx2 = np.empty(map_w, dtype=np.uint32)
        for by in range(by_start, by_stop):
            for bx in range(map_w):
                sx[bx] = np.uint32(0)
                sx2[bx] = np.uint32(0)
            y0 = by * block_size
            interior_y = by < full_by
            for dy in range(block_size):
                sy = y0 + dy if interior_y else (y0 + dy if y0 + dy < orig_h else max_y)
                row = gray[sy]
                for bx in range(full_bx):
                    x0 = bx * block_size
                    rsx = np.uint32(0)
                    rsx2 = np.uint32(0)
                    for dx in range(block_size):
                        v = np.uint32(row[x0 + dx])
                        rsx += v
                        rsx2 += v * v
                    sx[bx] += rsx
                    sx2[bx] += rsx2
                for bx in range(full_bx, map_w):
                    x0 = bx * block_size
                    rsx = np.uint32(0)
                    rsx2 = np.uint32(0)
                    for dx in range(block_size):
                        x = x0 + dx
                        sxx = x if x < orig_w else max_x
                        v = np.uint32(row[sxx])
                        rsx += v
                        rsx2 += v * v
                    sx[bx] += rsx
                    sx2[bx] += rsx2
            out_row = out_var[by]
            for bx in range(map_w):
                mean = np.float32(sx[bx]) * inv
                mean_sq = np.float32(sx2[bx]) * inv
                out_row[bx] = mean_sq - mean * mean
    else:
        sx_f = np.empty(map_w, dtype=np.float64)
        sx2_f = np.empty(map_w, dtype=np.float64)
        for by in range(by_start, by_stop):
            for bx in range(map_w):
                sx_f[bx] = 0.0
                sx2_f[bx] = 0.0
            y0 = by * block_size
            for dy in range(block_size):
                y = y0 + dy
                sy = y if y < orig_h else max_y
                row = gray[sy]
                for bx in range(full_bx):
                    x0 = bx * block_size
                    rsx = 0.0
                    rsx2 = 0.0
                    for dx in range(block_size):
                        v = np.float64(row[x0 + dx])
                        rsx += v
                        rsx2 += v * v
                    sx_f[bx] += rsx
                    sx2_f[bx] += rsx2
                for bx in range(full_bx, map_w):
                    x0 = bx * block_size
                    rsx = 0.0
                    rsx2 = 0.0
                    for dx in range(block_size):
                        x = x0 + dx
                        sxx = x if x < orig_w else max_x
                        v = np.float64(row[sxx])
                        rsx += v
                        rsx2 += v * v
                    sx_f[bx] += rsx
                    sx2_f[bx] += rsx2
            out_row = out_var[by]
            for bx in range(map_w):
                mean = np.float32(sx_f[bx]) * inv
                mean_sq = np.float32(sx2_f[bx]) * inv
                out_row[bx] = mean_sq - mean * mean


@njit(nogil=True)
def block_variance_u8(gray, orig_h, orig_w, block_size, out_var):
    """Compute per-block spatial variance matching padded INTER_AREA box averages."""
    block_variance_u8_band(gray, orig_h, orig_w, block_size, out_var, 0, out_var.shape[0])


@njit(nogil=True)
def block_variance_bgr_u8_band(src, orig_h, orig_w, block_size, out_var, by_start, by_stop):
    """Fused OpenCV-exact BGR/BGRA->gray + per-block variance over block-rows [by_start, by_stop)."""
    map_w = out_var.shape[1]
    full_by = orig_h // block_size
    full_bx = orig_w // block_size
    inv = np.float32(1.0) / np.float32(block_size * block_size)
    max_y = orig_h - 1
    max_x = orig_w - 1
    c_b = np.uint32(3735)
    c_g = np.uint32(19235)
    c_r = np.uint32(9798)
    c_half = np.uint32(16384)
    c_shift = np.uint32(15)

    if block_size <= 16:
        sx = np.empty(map_w, dtype=np.uint32)
        sx2 = np.empty(map_w, dtype=np.uint32)
        for by in range(by_start, by_stop):
            for bx in range(map_w):
                sx[bx] = np.uint32(0)
                sx2[bx] = np.uint32(0)
            y0 = by * block_size
            interior_y = by < full_by
            for dy in range(block_size):
                sy = y0 + dy if interior_y else (y0 + dy if y0 + dy < orig_h else max_y)
                row = src[sy]
                for bx in range(full_bx):
                    x0 = bx * block_size
                    rsx = np.uint32(0)
                    rsx2 = np.uint32(0)
                    for dx in range(block_size):
                        xx = x0 + dx
                        v = (
                            np.uint32(row[xx, 0]) * c_b
                            + np.uint32(row[xx, 1]) * c_g
                            + np.uint32(row[xx, 2]) * c_r
                            + c_half
                        ) >> c_shift
                        rsx += v
                        rsx2 += v * v
                    sx[bx] += rsx
                    sx2[bx] += rsx2
                for bx in range(full_bx, map_w):
                    x0 = bx * block_size
                    rsx = np.uint32(0)
                    rsx2 = np.uint32(0)
                    for dx in range(block_size):
                        x = x0 + dx
                        sxx = x if x < orig_w else max_x
                        v = (
                            np.uint32(row[sxx, 0]) * c_b
                            + np.uint32(row[sxx, 1]) * c_g
                            + np.uint32(row[sxx, 2]) * c_r
                            + c_half
                        ) >> c_shift
                        rsx += v
                        rsx2 += v * v
                    sx[bx] += rsx
                    sx2[bx] += rsx2
            out_row = out_var[by]
            for bx in range(map_w):
                mean = np.float32(sx[bx]) * inv
                mean_sq = np.float32(sx2[bx]) * inv
                out_row[bx] = mean_sq - mean * mean
    else:
        sx_f = np.empty(map_w, dtype=np.float64)
        sx2_f = np.empty(map_w, dtype=np.float64)
        for by in range(by_start, by_stop):
            for bx in range(map_w):
                sx_f[bx] = 0.0
                sx2_f[bx] = 0.0
            y0 = by * block_size
            for dy in range(block_size):
                y = y0 + dy
                sy = y if y < orig_h else max_y
                row = src[sy]
                for bx in range(full_bx):
                    x0 = bx * block_size
                    rsx = 0.0
                    rsx2 = 0.0
                    for dx in range(block_size):
                        xx = x0 + dx
                        v = np.float64(
                            (
                                np.uint32(row[xx, 0]) * c_b
                                + np.uint32(row[xx, 1]) * c_g
                                + np.uint32(row[xx, 2]) * c_r
                                + c_half
                            ) >> c_shift
                        )
                        rsx += v
                        rsx2 += v * v
                    sx_f[bx] += rsx
                    sx2_f[bx] += rsx2
                for bx in range(full_bx, map_w):
                    x0 = bx * block_size
                    rsx = 0.0
                    rsx2 = 0.0
                    for dx in range(block_size):
                        x = x0 + dx
                        sxx = x if x < orig_w else max_x
                        v = np.float64(
                            (
                                np.uint32(row[sxx, 0]) * c_b
                                + np.uint32(row[sxx, 1]) * c_g
                                + np.uint32(row[sxx, 2]) * c_r
                                + c_half
                            ) >> c_shift
                        )
                        rsx += v
                        rsx2 += v * v
                    sx_f[bx] += rsx
                    sx2_f[bx] += rsx2
            out_row = out_var[by]
            for bx in range(map_w):
                mean = np.float32(sx_f[bx]) * inv
                mean_sq = np.float32(sx2_f[bx]) * inv
                out_row[bx] = mean_sq - mean * mean


@njit(nogil=True)
def block_variance_bgr_u8(src, orig_h, orig_w, block_size, out_var):
    """Fused OpenCV-exact BGR/BGRA->gray + per-block variance without allocating gray."""
    block_variance_bgr_u8_band(src, orig_h, orig_w, block_size, out_var, 0, out_var.shape[0])


@njit(nogil=True)
def block_variance_f32(gray, orig_h, orig_w, block_size, out_var):
    """Compute per-block spatial variance on a float32 grayscale frame."""
    map_h = out_var.shape[0]
    map_w = out_var.shape[1]
    full_by = orig_h // block_size
    full_bx = orig_w // block_size
    inv_side = np.float32(1.0) / np.float32(block_size)
    max_y = orig_h - 1
    max_x = orig_w - 1

    col_s = np.empty(map_w, dtype=np.float32)
    col_s2 = np.empty(map_w, dtype=np.float32)

    for by in range(map_h):
        for bx in range(map_w):
            col_s[bx] = np.float32(0.0)
            col_s2[bx] = np.float32(0.0)
        y0 = by * block_size
        interior_y = by < full_by
        for dy in range(block_size):
            sy = y0 + dy if interior_y else (y0 + dy if y0 + dy < orig_h else max_y)
            row = gray[sy]
            for bx in range(full_bx):
                x0 = bx * block_size
                row_s = np.float32(0.0)
                row_s2 = np.float32(0.0)
                for dx in range(block_size):
                    v = np.float32(row[x0 + dx])
                    row_s += v
                    row_s2 += v * v
                col_s[bx] += row_s * inv_side
                col_s2[bx] += row_s2 * inv_side
            for bx in range(full_bx, map_w):
                x0 = bx * block_size
                row_s = np.float32(0.0)
                row_s2 = np.float32(0.0)
                for dx in range(block_size):
                    x = x0 + dx
                    sxx = x if x < orig_w else max_x
                    v = np.float32(row[sxx])
                    row_s += v
                    row_s2 += v * v
                col_s[bx] += row_s * inv_side
                col_s2[bx] += row_s2 * inv_side
        out_row = out_var[by]
        for bx in range(map_w):
            mean = col_s[bx] * inv_side
            mean_sq = col_s2[bx] * inv_side
            out_row[bx] = mean_sq - mean * mean


@njit(nogil=True)
def argmax_update_blocks(var_map, max_var, best_idx, frame_idx):
    """Update best frame indices in-place using strict greater-than comparison."""
    h, w = var_map.shape
    for y in range(h):
        v_row = var_map[y]
        m_row = max_var[y]
        b_row = best_idx[y]
        for x in range(w):
            v = v_row[x]
            if v > m_row[x]:
                m_row[x] = v
                b_row[x] = frame_idx


@njit(nogil=True)
def _quickselect_i32(buf, n, k):
    """In-place Hoare quickselect returning exact k-th smallest int32."""
    lo = 0
    hi = n - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        pivot = buf[mid]
        i = lo
        j = hi
        while i <= j:
            while buf[i] < pivot:
                i += 1
            while buf[j] > pivot:
                j -= 1
            if i <= j:
                tmp = buf[i]
                buf[i] = buf[j]
                buf[j] = tmp
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            return buf[k]
    return buf[k]


@njit(nogil=True)
def median_filter_i32_band(src, dst, ksize, y_start, y_stop):
    """Exact odd-kernel nearest-border median filter over rows [y_start, y_stop)."""
    h, w = src.shape
    if ksize <= 1:
        for y in range(y_start, y_stop):
            for x in range(w):
                dst[y, x] = src[y, x]
        return
    r = ksize >> 1
    win_n = ksize * ksize
    mid = win_n >> 1
    buf = np.empty(win_n, dtype=np.int32)
    y_lo = r if h > 2 * r else h
    y_hi = max(y_lo, h - r)
    x_lo = r if w > 2 * r else w
    x_hi = max(x_lo, w - r)

    for y in range(y_start, y_stop):
        out_row = dst[y]
        interior_y = y_lo <= y < y_hi
        for x in range(w):
            center = src[y, x]
            if interior_y and x_lo <= x < x_hi:
                uniform = True
                for dy in range(-r, r + 1):
                    s_row = src[y + dy]
                    for dx in range(-r, r + 1):
                        if s_row[x + dx] != center:
                            uniform = False
                            break
                    if not uniform:
                        break
                if uniform:
                    out_row[x] = center
                    continue
                pos = 0
                for dy in range(-r, r + 1):
                    s_row = src[y + dy]
                    for dx in range(-r, r + 1):
                        buf[pos] = s_row[x + dx]
                        pos += 1
                out_row[x] = _quickselect_i32(buf, win_n, mid)
            else:
                uniform = True
                for dy in range(-r, r + 1):
                    sy = y + dy
                    if sy < 0:
                        sy = 0
                    elif sy >= h:
                        sy = h - 1
                    s_row = src[sy]
                    for dx in range(-r, r + 1):
                        sx_i = x + dx
                        if sx_i < 0:
                            sx_i = 0
                        elif sx_i >= w:
                            sx_i = w - 1
                        if s_row[sx_i] != center:
                            uniform = False
                            break
                    if not uniform:
                        break
                if uniform:
                    out_row[x] = center
                    continue
                pos = 0
                for dy in range(-r, r + 1):
                    sy = y + dy
                    if sy < 0:
                        sy = 0
                    elif sy >= h:
                        sy = h - 1
                    s_row = src[sy]
                    for dx in range(-r, r + 1):
                        sx_i = x + dx
                        if sx_i < 0:
                            sx_i = 0
                        elif sx_i >= w:
                            sx_i = w - 1
                        buf[pos] = s_row[sx_i]
                        pos += 1
                out_row[x] = _quickselect_i32(buf, win_n, mid)


@njit(nogil=True)
def median_filter_i32(src, dst, ksize):
    """Exact odd-kernel nearest-border median filter matching scipy.ndimage.median_filter."""
    median_filter_i32_band(src, dst, ksize, 0, src.shape[0])


@njit(nogil=True)
def copy_selected_blocks_u8_band(src, final_idx, dst, orig_h, orig_w, block_size, frame_idx, by_start, by_stop):
    """Copy BGR pixels belonging to frame_idx row-sequentially over block-rows [by_start, by_stop)."""
    map_w = final_idx.shape[1]
    for by in range(by_start, by_stop):
        idx_row = final_idx[by]
        first_bx = -1
        last_bx = -1
        for bx in range(map_w):
            if idx_row[bx] == frame_idx:
                if first_bx < 0:
                    first_bx = bx
                last_bx = bx
        if first_bx < 0:
            continue
        y0 = by * block_size
        y1 = orig_h if y0 + block_size > orig_h else y0 + block_size
        for y in range(y0, y1):
            s_row = src[y]
            d_row = dst[y]
            bx = first_bx
            while bx <= last_bx:
                if idx_row[bx] != frame_idx:
                    bx += 1
                    continue
                run_start = bx
                bx += 1
                while bx <= last_bx and idx_row[bx] == frame_idx:
                    bx += 1
                x0 = run_start * block_size
                x1 = bx * block_size
                if x1 > orig_w:
                    x1 = orig_w
                for x in range(x0, x1):
                    d_row[x, 0] = s_row[x, 0]
                    d_row[x, 1] = s_row[x, 1]
                    d_row[x, 2] = s_row[x, 2]


@njit(nogil=True)
def copy_selected_blocks_u8(src, final_idx, dst, orig_h, orig_w, block_size, frame_idx):
    """Copy BGR pixels belonging to frame_idx row-sequentially from uint8 source into dst."""
    copy_selected_blocks_u8_band(
        src, final_idx, dst, orig_h, orig_w, block_size, frame_idx, 0, final_idx.shape[0]
    )


@njit(nogil=True)
def copy_selected_blocks_u16(src, final_idx, dst, orig_h, orig_w, block_size, frame_idx):
    """Copy BGR pixels belonging to frame_idx from uint16 source into float32 dst."""
    map_h, map_w = final_idx.shape
    scale = np.float32(255.0 / 65535.0)
    for by in range(map_h):
        idx_row = final_idx[by]
        first_bx = -1
        last_bx = -1
        for bx in range(map_w):
            if idx_row[bx] == frame_idx:
                if first_bx < 0:
                    first_bx = bx
                last_bx = bx
        if first_bx < 0:
            continue
        y0 = by * block_size
        y1 = orig_h if y0 + block_size > orig_h else y0 + block_size
        for y in range(y0, y1):
            s_row = src[y]
            d_row = dst[y]
            bx = first_bx
            while bx <= last_bx:
                if idx_row[bx] != frame_idx:
                    bx += 1
                    continue
                run_start = bx
                bx += 1
                while bx <= last_bx and idx_row[bx] == frame_idx:
                    bx += 1
                x0 = run_start * block_size
                x1 = bx * block_size
                if x1 > orig_w:
                    x1 = orig_w
                for x in range(x0, x1):
                    d_row[x, 0] = np.float32(s_row[x, 0]) * scale
                    d_row[x, 1] = np.float32(s_row[x, 1]) * scale
                    d_row[x, 2] = np.float32(s_row[x, 2]) * scale


@njit(nogil=True)
def copy_selected_blocks_f32(src, final_idx, dst, orig_h, orig_w, block_size, frame_idx):
    """Copy BGR pixels belonging to frame_idx from float source into dst."""
    map_h, map_w = final_idx.shape
    for by in range(map_h):
        idx_row = final_idx[by]
        first_bx = -1
        last_bx = -1
        for bx in range(map_w):
            if idx_row[bx] == frame_idx:
                if first_bx < 0:
                    first_bx = bx
                last_bx = bx
        if first_bx < 0:
            continue
        y0 = by * block_size
        y1 = orig_h if y0 + block_size > orig_h else y0 + block_size
        for y in range(y0, y1):
            s_row = src[y]
            d_row = dst[y]
            bx = first_bx
            while bx <= last_bx:
                if idx_row[bx] != frame_idx:
                    bx += 1
                    continue
                run_start = bx
                bx += 1
                while bx <= last_bx and idx_row[bx] == frame_idx:
                    bx += 1
                x0 = run_start * block_size
                x1 = bx * block_size
                if x1 > orig_w:
                    x1 = orig_w
                for x in range(x0, x1):
                    d_row[x, 0] = np.float32(s_row[x, 0])
                    d_row[x, 1] = np.float32(s_row[x, 1])
                    d_row[x, 2] = np.float32(s_row[x, 2])
