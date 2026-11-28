"""Unified inference, smooth probability blending, and seamless tiled / sparse 100MP engine."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import math
import torch
from torch.nn import functional as F

from .network import StreamFocusNet


def optimize_model(
    model: StreamFocusNet,
    device: torch.device,
    dtype: torch.dtype = None,
    channels_last: bool = False,
    compile_mode: str = None,
) -> StreamFocusNet:
    """Prepare StreamFocusNet for fast inference with optional dtype, channels_last, and torch.compile."""
    model = model.to(device=device, dtype=dtype) if dtype is not None else model.to(device=device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    if compile_mode and device.type == 'cuda' and hasattr(torch, 'compile'):
        model = torch.compile(model, mode=compile_mode)
    return model


def smooth_probs(probs: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """Separable Gaussian smoothing over [B, N, H, W] probabilities, renormalized over N."""
    if sigma <= 0:
        return probs
    r = int(3 * sigma + 0.5)
    t = torch.arange(-r, r + 1, dtype=probs.dtype, device=probs.device)
    k = torch.exp(-t ** 2 / (2 * sigma * sigma))
    k = k / k.sum()
    N = probs.shape[1]
    q = F.conv2d(F.pad(probs, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1).repeat(N, 1, 1, 1), groups=N)
    q = F.conv2d(F.pad(q, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1).repeat(N, 1, 1, 1), groups=N)
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)


@lru_cache(maxsize=64)
def _feather_cached(tile_h: int, tile_w: int, top: int, bottom: int, left: int, right: int, dev_str: str) -> torch.Tensor:
    device = torch.device(dev_str)
    wy = torch.ones(tile_h, dtype=torch.float32, device=device)
    wx = torch.ones(tile_w, dtype=torch.float32, device=device)
    if top > 0:
        t = (torch.arange(top, dtype=torch.float32, device=device) + 0.5) / top
        wy[:top] = 0.5 - 0.5 * torch.cos(math.pi * t)
    if bottom > 0:
        t = (torch.arange(bottom, dtype=torch.float32, device=device) + 0.5) / bottom
        wy[-bottom:] = 0.5 + 0.5 * torch.cos(math.pi * t)
    if left > 0:
        t = (torch.arange(left, dtype=torch.float32, device=device) + 0.5) / left
        wx[:left] = 0.5 - 0.5 * torch.cos(math.pi * t)
    if right > 0:
        t = (torch.arange(right, dtype=torch.float32, device=device) + 0.5) / right
        wx[-right:] = 0.5 + 0.5 * torch.cos(math.pi * t)
    return wy[:, None] * wx[None, :]


def make_2d_feather(tile_h: int, tile_w: int, top: int, bottom: int, left: int, right: int, device) -> torch.Tensor:
    """Raised-cosine/linear separable window weights that sum strictly to 1.0 across overlapping tiles."""
    return _feather_cached(tile_h, tile_w, top, bottom, left, right, str(device))


def tile_starts(length: int, tile_size: int, overlap: int):
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError('overlap must be strictly smaller than tile_size')
    starts = list(range(0, length - tile_size, stride))
    if not starts or starts[-1] + tile_size < length:
        starts.append(length - tile_size)
    return starts


def compute_tile_layout(height: int, width: int, tile_size: int, overlap: int):
    """Yield (y0, y1, x0, x1, top_ov, bot_ov, left_ov, right_ov) with exact partition-of-unity overlaps."""
    ys = tile_starts(height, tile_size, overlap)
    xs = tile_starts(width, tile_size, overlap)
    tiles = []
    for yi, y0 in enumerate(ys):
        y1 = min(height, y0 + tile_size)
        top_ov = (ys[yi - 1] + tile_size - y0) if yi > 0 else 0
        bot_ov = (y1 - ys[yi + 1]) if yi + 1 < len(ys) else 0
        for xi, x0 in enumerate(xs):
            x1 = min(width, x0 + tile_size)
            left_ov = (xs[xi - 1] + tile_size - x0) if xi > 0 else 0
            right_ov = (x1 - xs[xi + 1]) if xi + 1 < len(xs) else 0
            tiles.append((y0, y1, x0, x1, top_ov, bot_ov, left_ov, right_ov))
    return tiles


@torch.inference_mode()
def fuse_stack(
    model: StreamFocusNet,
    luma_frames: torch.Tensor,
    color_frames: torch.Tensor,
    device: torch.device,
    chunk_size: int = 8,
    max_mem_layers: int = 32,
    smooth_sigma: float = 0.0,
    tile_size: int = 0,
    tile_overlap: int = 128,
    amp_dtype: torch.dtype = None,
):
    """Unified fusion for arbitrary N and (H, W).

    luma_frames: [N, H, W] float32 in [0, 1] (CPU or GPU)
    color_frames: [N, C, H, W] float32 (CPU or GPU)
    Returns (fused [C, H, W] fp32 on CPU, winner_idx [H, W] int64 on CPU).
    """
    N, H, W = luma_frames.shape
    C = color_frames.shape[1]
    use_amp = device.type == 'cuda' and amp_dtype is not None

    if tile_size > 0 and max(H, W) > tile_size:
        out_acc = torch.zeros((C, H, W), dtype=torch.float32)
        w_acc = torch.zeros((H, W), dtype=torch.float32)
        winner_out = torch.zeros((H, W), dtype=torch.int64)
        best_w = torch.full((H, W), -1.0, dtype=torch.float32)
        for y0, y1, x0, x1, top_ov, bot_ov, left_ov, right_ov in compute_tile_layout(H, W, tile_size, tile_overlap):
            tile_fused, tile_win = fuse_stack(
                model,
                luma_frames[:, y0:y1, x0:x1],
                color_frames[:, :, y0:y1, x0:x1],
                device=device,
                chunk_size=chunk_size,
                max_mem_layers=max_mem_layers,
                smooth_sigma=smooth_sigma,
                tile_size=0,
                amp_dtype=amp_dtype,
            )
            w2d = make_2d_feather(y1 - y0, x1 - x0, top_ov, bot_ov, left_ov, right_ov, 'cpu')
            out_acc[:, y0:y1, x0:x1] += tile_fused * w2d.unsqueeze(0)
            w_acc[y0:y1, x0:x1] += w2d
            mask = w2d > best_w[y0:y1, x0:x1]
            best_w[y0:y1, x0:x1] = torch.where(mask, w2d, best_w[y0:y1, x0:x1])
            winner_out[y0:y1, x0:x1] = torch.where(mask, tile_win, winner_out[y0:y1, x0:x1])
        return out_acc / w_acc.clamp_min(1e-12).unsqueeze(0), winner_out

    if N <= max_mem_layers and H * W <= 2048 * 2048:
        param_dtype = model.enc1[0].weight.dtype if hasattr(model, 'enc1') else torch.float32
        x = luma_frames.to(device=device, dtype=param_dtype, non_blocking=True).unsqueeze(0)
        rgb = color_frames.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model.forward_logits(x).float()
        probs = logits.softmax(dim=1)
        if smooth_sigma > 0:
            probs = smooth_probs(probs, smooth_sigma)
        fused = (probs[0].unsqueeze(1) * rgb).sum(dim=0)
        winner = probs[0].argmax(dim=0)
        return fused.cpu(), winner.cpu()

    if smooth_sigma > 0:
        param_dtype = model.enc1[0].weight.dtype if hasattr(model, 'enc1') else torch.float32
        x = luma_frames.to(device=device, dtype=param_dtype, non_blocking=True).unsqueeze(0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits_all = model.stream_logits(x, chunk_size=chunk_size)
        probs = smooth_probs(logits_all.softmax(dim=1), smooth_sigma)[0]
        fused = torch.zeros((C, H, W), device=device, dtype=torch.float32)
        for s in range(0, N, chunk_size):
            cnt = min(chunk_size, N - s)
            rgb_c = color_frames[s:s + cnt].to(device=device, dtype=torch.float32, non_blocking=True)
            fused += (probs[s:s + cnt].unsqueeze(1) * rgb_c).sum(dim=0)
        return fused.cpu(), probs.argmax(dim=0).cpu()

    fused, winner, _ = model.stream_fuse(
        lambda s, cnt: luma_frames[s:s + cnt],
        lambda s, cnt: color_frames[s:s + cnt],
        num_layers=N,
        hw=(H, W),
        device=device,
        chunk_size=chunk_size,
        out_channels=C,
        amp_dtype=amp_dtype,
    )
    return fused.cpu(), winner.cpu()


def _resolve_stream_chunks(device: torch.device, chunk_size, native_chunk_size, low_vram: bool = False):
    """Select fastest O(1)-VRAM chunk sizes (~0.9-1.25 GiB) with silent fallback to ~0.4 GiB if free VRAM < 2 GiB."""
    if low_vram or device.type != 'cuda':
        def_p, def_n = 2, 1
    else:
        def_p, def_n = 8, 4
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if free_bytes < 2 * 1024 ** 3:
                def_p, def_n = 2, 1
        except Exception:
            def_p, def_n = 2, 1
    return (def_p if chunk_size is None else int(chunk_size)), (def_n if native_chunk_size is None else int(native_chunk_size))


@torch.inference_mode()
def fuse_large_stack_sparse(
    model: StreamFocusNet,
    read_preview_luma_chunk,
    read_native_tile,
    num_layers: int,
    native_hw: tuple,
    preview_hw: tuple,
    device: torch.device,
    tile_size: int = 1024,
    tile_overlap: int = 128,
    top_k: int = 16,
    chunk_size: int = None,
    native_chunk_size: int = None,
    amp_dtype: torch.dtype = None,
    low_vram: bool = False,
    prefetch_tiles: bool = True,
    shortcut_prob: float = 0.0,
):
    """Preview-guided sparse Top-K native tiled streaming using the SAME StreamFocusNet model.

    1) Runs O(1)-VRAM stream passes on previews (1 summary pass/round + 1 single decode pass)
       to rank Top-K active layers per native tile via an O(1)-VRAM coarse Top-M buffer.
    2) For each native tile, reads ONLY the Top-K active layers and runs stream_fuse natively.
    """
    chunk_size, native_chunk_size = _resolve_stream_chunks(device, chunk_size, native_chunk_size, low_vram=low_vram)
    H, W = native_hw
    ph, pw = preview_hw
    h2, w2 = (ph + 1) // 2, (pw + 1) // 2
    h4, w4 = (h2 + 1) // 2, (w2 + 1) // 2
    param_dtype = model.enc1[0].weight.dtype
    use_amp = device.type == 'cuda' and amp_dtype is not None

    half_state = model.half_cross_blk.init_state(1, h2, w2, device) if model.half_cross_blk is not None else None
    half_summary = None

    # Pass 1..R: preview summaries (O(1) VRAM)
    summaries = []
    for r_idx, layer_blk in enumerate(model.cross_layers):
        state = layer_blk.init_state(1, h4, w4, device)
        for s in range(0, num_layers, chunk_size):
            cnt = min(chunk_size, num_layers - s)
            luma = read_preview_luma_chunk(s, cnt).to(device=device, dtype=param_dtype, non_blocking=True).unsqueeze(0)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if r_idx == 0 and half_state is not None:
                    e2_c, f4 = model.encode_e2_f4(luma)
                    model.half_cross_blk.accumulate(half_state, e2_c)
                    del e2_c
                else:
                    f4 = model.encode_f4_only(luma)
                for prev in range(r_idx):
                    f4 = model.cross_layers[prev].apply_chunk(f4, summaries[prev])
                layer_blk.accumulate(state, f4)
            del luma, f4
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            summaries.append(layer_blk.finalize_summary(state))
            if r_idx == 0 and half_state is not None:
                half_summary = model.half_cross_blk.finalize_summary(half_state)
                del half_state
        del state

    # Single Pass R+1: streaming coarse Top-M score buffer on 1/4 preview grid (O(1) VRAM)
    tiles = compute_tile_layout(H, W, tile_size, tile_overlap)
    k_eff = min(num_layers, top_k)
    pool_s = 4
    gh, gw = (ph + pool_s - 1) // pool_s, (pw + pool_s - 1) // pool_s
    m_keep = min(num_layers, max(16, k_eff))

    top_vals = torch.full((m_keep, gh, gw), -torch.inf, device=device, dtype=torch.float32)
    top_ids = torch.zeros((m_keep, gh, gw), device=device, dtype=torch.int64)

    for s in range(0, num_layers, chunk_size):
        cnt = min(chunk_size, num_layers - s)
        luma = read_preview_luma_chunk(s, cnt).to(device=device, dtype=param_dtype, non_blocking=True).unsqueeze(0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            e1, e2, f4 = model.encode_chunk(luma)
            for r_idx, layer_blk in enumerate(model.cross_layers):
                f4 = layer_blk.apply_chunk(f4, summaries[r_idx])
            sc = model.decode_chunk(f4, e2, e1, half_summary)[0].float()
        del luma, e1, e2, f4

        sc_g = F.avg_pool2d(sc.unsqueeze(1), pool_s, stride=pool_s, ceil_mode=True).squeeze(1)
        del sc
        cand_vals = torch.cat([top_vals, sc_g], dim=0)
        cand_ids = torch.cat(
            [top_ids, torch.arange(s, s + cnt, device=device)[:, None, None].expand(-1, gh, gw)], dim=0
        )
        top_vals, pos = cand_vals.topk(m_keep, dim=0)
        top_ids = cand_ids.gather(0, pos)
        del sc_g, cand_vals, cand_ids, pos

    top_probs_cpu = top_vals.softmax(dim=0).cpu()
    top_ids_cpu = top_ids.cpu()
    del top_vals, top_ids, summaries, half_summary

    tile_top_ids_cpu = []
    tile_shortcut_winners = []
    for (y0, y1, x0, x1, *_) in tiles:
        gy0 = max(0, min(gh - 1, int(y0 * gh / H)))
        gy1 = max(gy0 + 1, min(gh, int(math.ceil(y1 * gh / H))))
        gx0 = max(0, min(gw - 1, int(x0 * gw / W)))
        gx1 = max(gx0 + 1, min(gw, int(math.ceil(x1 * gw / W))))
        shortcut_win = None
        if shortcut_prob > 0.0:
            top1_p_min = float(top_probs_cpu[0, gy0:gy1, gx0:gx1].min())
            top1_ids_patch = top_ids_cpu[0, gy0:gy1, gx0:gx1]
            top1_first = int(top1_ids_patch.reshape(-1)[0])
            if top1_p_min >= shortcut_prob and bool((top1_ids_patch == top1_first).all()):
                shortcut_win = top1_first
        tile_shortcut_winners.append(shortcut_win)
        if shortcut_win is not None:
            tile_top_ids_cpu.append([shortcut_win])
        else:
            tp = top_probs_cpu[:, gy0:gy1, gx0:gx1].reshape(-1)
            ti = top_ids_cpu[:, gy0:gy1, gx0:gx1].reshape(-1)
            layer_mass = torch.zeros(num_layers, dtype=torch.float32)
            layer_mass.scatter_add_(0, ti, tp)
            tile_top_ids_cpu.append(sorted(layer_mass.topk(k_eff).indices.tolist()))

    del top_probs_cpu, top_ids_cpu

    out_acc = torch.zeros((3, H, W), dtype=torch.float32)
    w_acc = torch.zeros((H, W), dtype=torch.float32)
    winner_out = torch.zeros((H, W), dtype=torch.int64)
    best_w = torch.full((H, W), -1.0, dtype=torch.float32)

    def _fetch(t_idx: int):
        y0, y1, x0, x1, *_ = tiles[t_idx]
        return read_native_tile(tile_top_ids_cpu[t_idx], y0, y1, x0, x1)

    def _run_tile(t_idx: int, tile_data):
        y0, y1, x0, x1, top_ov, bot_ov, left_ov, right_ov = tiles[t_idx]
        ids = tile_top_ids_cpu[t_idx]
        luma_tile, rgb_tile = tile_data
        shortcut_win = tile_shortcut_winners[t_idx]
        if shortcut_win is not None:
            tile_fused = rgb_tile[0].float().cpu()
            tile_win = torch.full((y1 - y0, x1 - x0), shortcut_win, dtype=torch.int64)
        else:
            tile_fused, tile_local_win, _ = model.stream_fuse(
                lambda s, cnt: luma_tile[s:s + cnt],
                lambda s, cnt: rgb_tile[s:s + cnt],
                num_layers=len(ids),
                hw=(y1 - y0, x1 - x0),
                device=device,
                chunk_size=native_chunk_size,
                out_channels=3,
                amp_dtype=amp_dtype,
            )
            tile_fused = tile_fused.cpu()
            id_tensor = torch.tensor(ids, dtype=torch.int64)
            tile_win = id_tensor[tile_local_win.cpu()]
        w2d = make_2d_feather(y1 - y0, x1 - x0, top_ov, bot_ov, left_ov, right_ov, 'cpu')
        out_acc[:, y0:y1, x0:x1].add_(tile_fused * w2d.unsqueeze(0))
        w_acc[y0:y1, x0:x1].add_(w2d)
        mask = w2d > best_w[y0:y1, x0:x1]
        best_w[y0:y1, x0:x1] = torch.where(mask, w2d, best_w[y0:y1, x0:x1])
        winner_out[y0:y1, x0:x1] = torch.where(mask, tile_win, winner_out[y0:y1, x0:x1])

    if prefetch_tiles and len(tiles) > 1:
        with ThreadPoolExecutor(max_workers=1) as io_pool:
            fut = io_pool.submit(_fetch, 0)
            for t_idx in range(len(tiles)):
                tile_data = fut.result()
                if t_idx + 1 < len(tiles):
                    fut = io_pool.submit(_fetch, t_idx + 1)
                _run_tile(t_idx, tile_data)
    else:
        for t_idx in range(len(tiles)):
            _run_tile(t_idx, _fetch(t_idx))

    return out_acc / w_acc.clamp_min(1e-12).unsqueeze(0), winner_out
