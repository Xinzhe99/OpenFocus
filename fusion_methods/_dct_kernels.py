"""Compiled Numba CPU kernels for DCT block-variance multi-focus fusion."""
import numpy as np

from fusion_methods._depth_kernel import njit


@njit(nogil=True)
def block_variance_u8(gray, orig_h, orig_w, block_size, out_var):
    """Compute per-block spatial variance matching padded INTER_AREA box averages."""
    map_h = out_var.shape[0]
    map_w = out_var.shape[1]
    full_by = orig_h // block_size
    full_bx = orig_w // block_size
    inv = np.float32(1.0) / np.float32(block_size * block_size)
    max_y = orig_h - 1
    max_x = orig_w - 1

    if block_size <= 16:
        for by in range(map_h):
            y0 = by * block_size
            interior_y = by < full_by
            for bx in range(map_w):
                x0 = bx * block_size
                sx = np.int64(0)
                sx2 = np.int64(0)
                if interior_y and bx < full_bx:
                    for dy in range(block_size):
                        y = y0 + dy
                        for dx in range(block_size):
                            v = np.int64(gray[y, x0 + dx])
                            sx += v
                            sx2 += v * v
                else:
                    for dy in range(block_size):
                        y = y0 + dy
                        sy = y if y < orig_h else max_y
                        for dx in range(block_size):
                            x = x0 + dx
                            sxx = x if x < orig_w else max_x
                            v = np.int64(gray[sy, sxx])
                            sx += v
                            sx2 += v * v
                mean = np.float32(sx) * inv
                mean_sq = np.float32(sx2) * inv
                out_var[by, bx] = mean_sq - mean * mean
    else:
        for by in range(map_h):
            y0 = by * block_size
            for bx in range(map_w):
                x0 = bx * block_size
                sx = 0.0
                sx2 = 0.0
                for dy in range(block_size):
                    y = y0 + dy
                    sy = y if y < orig_h else max_y
                    for dx in range(block_size):
                        x = x0 + dx
                        sxx = x if x < orig_w else max_x
                        v = np.float64(gray[sy, sxx])
                        sx += v
                        sx2 += v * v
                mean = np.float32(sx) * inv
                mean_sq = np.float32(sx2) * inv
                out_var[by, bx] = mean_sq - mean * mean


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

    for by in range(map_h):
        y0 = by * block_size
        interior_y = by < full_by
        for bx in range(map_w):
            x0 = bx * block_size
            col_s = np.float32(0.0)
            col_s2 = np.float32(0.0)
            if interior_y and bx < full_bx:
                for dy in range(block_size):
                    y = y0 + dy
                    row_s = np.float32(0.0)
                    row_s2 = np.float32(0.0)
                    for dx in range(block_size):
                        v = np.float32(gray[y, x0 + dx])
                        row_s += v
                        row_s2 += v * v
                    col_s += row_s * inv_side
                    col_s2 += row_s2 * inv_side
            else:
                for dy in range(block_size):
                    y = y0 + dy
                    sy = y if y < orig_h else max_y
                    row_s = np.float32(0.0)
                    row_s2 = np.float32(0.0)
                    for dx in range(block_size):
                        x = x0 + dx
                        sxx = x if x < orig_w else max_x
                        v = np.float32(gray[sy, sxx])
                        row_s += v
                        row_s2 += v * v
                    col_s += row_s * inv_side
                    col_s2 += row_s2 * inv_side
            mean = col_s * inv_side
            mean_sq = col_s2 * inv_side
            out_var[by, bx] = mean_sq - mean * mean


@njit(nogil=True)
def argmax_update_blocks(var_map, max_var, best_idx, frame_idx):
    """Update best frame indices in-place using strict greater-than comparison."""
    h, w = var_map.shape
    for y in range(h):
        for x in range(w):
            v = var_map[y, x]
            if v > max_var[y, x]:
                max_var[y, x] = v
                best_idx[y, x] = frame_idx


@njit(nogil=True)
def copy_selected_blocks_u8(src, final_idx, dst, orig_h, orig_w, block_size, frame_idx):
    """Copy BGR pixels belonging to frame_idx directly from uint8 source into dst."""
    map_h, map_w = final_idx.shape
    for by in range(map_h):
        y0 = by * block_size
        y1 = min(orig_h, y0 + block_size)
        for bx in range(map_w):
            if final_idx[by, bx] != frame_idx:
                continue
            x0 = bx * block_size
            x1 = min(orig_w, x0 + block_size)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    dst[y, x, 0] = src[y, x, 0]
                    dst[y, x, 1] = src[y, x, 1]
                    dst[y, x, 2] = src[y, x, 2]


@njit(nogil=True)
def copy_selected_blocks_u16(src, final_idx, dst, orig_h, orig_w, block_size, frame_idx):
    """Copy BGR pixels belonging to frame_idx from uint16 source into float32 dst."""
    map_h, map_w = final_idx.shape
    scale = np.float32(255.0 / 65535.0)
    for by in range(map_h):
        y0 = by * block_size
        y1 = min(orig_h, y0 + block_size)
        for bx in range(map_w):
            if final_idx[by, bx] != frame_idx:
                continue
            x0 = bx * block_size
            x1 = min(orig_w, x0 + block_size)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    dst[y, x, 0] = np.float32(src[y, x, 0]) * scale
                    dst[y, x, 1] = np.float32(src[y, x, 1]) * scale
                    dst[y, x, 2] = np.float32(src[y, x, 2]) * scale


@njit(nogil=True)
def copy_selected_blocks_f32(src, final_idx, dst, orig_h, orig_w, block_size, frame_idx):
    """Copy BGR pixels belonging to frame_idx from float source into dst."""
    map_h, map_w = final_idx.shape
    for by in range(map_h):
        y0 = by * block_size
        y1 = min(orig_h, y0 + block_size)
        for bx in range(map_w):
            if final_idx[by, bx] != frame_idx:
                continue
            x0 = bx * block_size
            x1 = min(orig_w, x0 + block_size)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    dst[y, x, 0] = np.float32(src[y, x, 0])
                    dst[y, x, 1] = np.float32(src[y, x, 1])
                    dst[y, x, 2] = np.float32(src[y, x, 2])
