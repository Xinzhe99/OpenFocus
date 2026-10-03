"""Focus-position (pseudo-depth) index maps.

Computes, per pixel, which frame of the stack is in focus — the "depth"
signal every fusion algorithm derives internally — using each method's own
activity measure, so the exported map matches the fusion the user sees:

- guided_filter: argmax of Gaussian(|Laplacian(channel sum)|) (gff saliency)
- dct:           block-variance winner + double median filter (dct's own
                 consistency pass), upsampled from the block grid
- gfgfgf:        argmax of the guided-filtered blur-difference AFM
- dtcwt:         argmax of the per-pixel DTCWT high-pass energy (the wavelet
                 activity level the fusion uses, on the gray channel)
- stackmffv4:    the model's own focus-index output (tiled with the same
                 memory budget as AI fusion)

The fusion pipeline itself is untouched; these are faithful mirrors of the
measures, factored out so they are testable and reusable (dialog, CLI).

All functions take images in the 0-255 BGR domain (uint8 or float) and
return a float32 [H, W] map in [0, 1] — the normalized frame index.
"""
import os
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

SUPPORTED_METHODS = ("guided_filter", "dct", "dtcwt", "gfgfgf", "stackmffv4")


class DepthMapCancelled(Exception):
    """Raised when should_cancel returns True mid-computation."""


def _check(images: Sequence[np.ndarray]) -> None:
    if len(images) < 2:
        raise ValueError("A focus index map needs at least 2 frames")
    first = images[0]
    if first.ndim != 3 or first.shape[2] < 3:
        raise ValueError("Depth map expects BGR images")


def _norm_index(argmax_map: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(argmax_map.shape, np.float32)
    return (argmax_map.astype(np.float32) / (n - 1)).clip(0.0, 1.0)


def _tick(progress: Optional[Callable], i: int, n: int,
          should_cancel: Optional[Callable[[], bool]]) -> None:
    if should_cancel is not None and should_cancel():
        raise DepthMapCancelled()
    if progress is not None:
        try:
            progress(i, n)
        except Exception:
            pass


# --------------------------------------------------------------------------
# guided_filter — gff's saliency measure
# --------------------------------------------------------------------------
def _measure_gff(images: Sequence[np.ndarray], progress, should_cancel):
    sigma = 5.0  # gff.DEFAULT_SIGMA_R
    # 流式 argmax：不物化 n×H×W 的显著性栈（40×6K×4K = 3.8GB），只保留
    # 逐像素当前最优值/索引；严格 > 保持与 np.argmax 相同的首胜语义
    best_val = best_idx = None
    for idx, bgr in enumerate(images):
        _tick(progress, idx, len(images), should_cancel)
        img_sum = np.sum(bgr.astype(np.float32), axis=2)
        lap = np.abs(cv2.Laplacian(img_sum, cv2.CV_32F, ksize=1,
                                   borderType=cv2.BORDER_REFLECT))
        sal = cv2.GaussianBlur(lap, (0, 0), sigmaX=sigma, sigmaY=sigma,
                               borderType=cv2.BORDER_REFLECT)
        if best_val is None:
            best_val, best_idx = sal, np.zeros(sal.shape, np.int32)
        else:
            hit = sal > best_val
            best_val[hit] = sal[hit]
            best_idx[hit] = idx
    return best_idx


# --------------------------------------------------------------------------
# dct — block variance + double median consistency
# --------------------------------------------------------------------------
def _measure_dct(images, progress, should_cancel, kernel_size=7, block_size=8):
    from scipy.ndimage import median_filter

    if kernel_size % 2 == 0:
        kernel_size += 1
    block_size = max(2, int(block_size))
    h, w = images[0].shape[:2]
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    map_h, map_w = h_trim // block_size, w_trim // block_size
    if map_h == 0 or map_w == 0:
        raise ValueError("Block size too large for the image")

    max_var = np.full((map_h, map_w), -1.0, np.float32)
    best = np.zeros((map_h, map_w), np.int32)
    for idx, bgr in enumerate(images):
        _tick(progress, idx, len(images), should_cancel)
        gray = cv2.cvtColor(bgr[:h_trim, :w_trim].astype(np.float32),
                            cv2.COLOR_BGR2GRAY)
        mean_sq = cv2.resize(gray ** 2, (map_w, map_h),
                             interpolation=cv2.INTER_AREA)
        mean_val = cv2.resize(gray, (map_w, map_h),
                              interpolation=cv2.INTER_AREA)
        var = mean_sq - mean_val ** 2
        mask = var > max_var
        max_var[mask] = var[mask]
        best[mask] = idx

    best = median_filter(best, size=kernel_size, mode="nearest")
    best = median_filter(best, size=kernel_size, mode="nearest")
    # 块级索引回到全分辨率（与 dct 融合重建的掩膜放大方式一致）；
    # 融合输出会裁到 block 的整数倍，但深度图必须与栈同尺寸，否则
    # 叠加/保存侧的形状守卫会静默失效
    best = cv2.resize(best.astype(np.float32), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return best


# --------------------------------------------------------------------------
# gfgfgf — blur-difference + guided-filter AFM
# --------------------------------------------------------------------------
def _measure_gfgfgf(images, progress, should_cancel, kernel_size=7,
                    thread_count=None):
    from fusion_methods.gfg_fgf import _run_guided_filter

    if kernel_size % 2 == 0:
        kernel_size += 1
    g_msz, g_gsz, g_eps, threshold = kernel_size, 5, 0.3, 0.005
    n = len(images)
    # 流式 argmax（见 _measure_gff）：AFM 栈不物化
    best_val = best_idx = None
    for idx, bgr in enumerate(images):
        _tick(progress, idx, n, should_cancel)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        blur = cv2.blur(g, (g_msz, g_msz))
        diff = cv2.absdiff(g, blur)
        _, gfg = cv2.threshold(diff, threshold, 0, cv2.THRESH_TOZERO)
        afm = _run_guided_filter(g, gfg, g_gsz, g_eps)
        if best_val is None:
            best_val, best_idx = afm, np.zeros(afm.shape, np.int32)
        else:
            hit = afm > best_val
            best_val[hit] = afm[hit]
            best_idx[hit] = idx
    return best_idx


# --------------------------------------------------------------------------
# dtcwt — per-frame wavelet high-pass energy (its activity level)
# --------------------------------------------------------------------------
def _measure_dtcwt(images, progress, should_cancel, nlevels=3):
    try:
        import dtcwt as dtcwt_lib
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("DTCWT depth maps need the dtcwt package") from exc

    transform = dtcwt_lib.Transform2d()
    n = len(images)
    h, w = images[0].shape[:2]
    # 流式 argmax（见 _measure_gff）：能量栈不物化
    best_val = best_idx = None
    for idx, bgr in enumerate(images):
        _tick(progress, idx, n, should_cancel)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        coeffs = transform.forward(gray, nlevels=nlevels)
        acc = np.zeros((h, w), np.float32)
        for level_yh in coeffs.highpasses:
            # (H', W', 6) -> 每方向幅值平方求和，上采样回全分辨率累加
            mag2 = np.sum(np.abs(level_yh) ** 2, axis=2).astype(np.float32)
            lvl = cv2.resize(mag2, (w, h), interpolation=cv2.INTER_LINEAR)
            acc += lvl
        if best_val is None:
            best_val, best_idx = acc, np.zeros(acc.shape, np.int32)
        else:
            hit = acc > best_val
            best_val[hit] = acc[hit]
            best_idx[hit] = idx
    return best_idx


# --------------------------------------------------------------------------
# stackmffv4 — the model's own focus indices, tiled within the AI budget
# --------------------------------------------------------------------------
def _tile_weight(x0, y0, x1, y1, fw, fh, W, H, overlap):
    """Linear ramp over the overlap band, zero outside — same feathering
    idea as tiled fusion, applied to the index maps. Bands clamp to the
    tile size so tiny tiles (small images) stay valid."""
    wgt = np.ones((fh, fw), np.float32)
    oy = min(max(0, overlap), fh)
    ox = min(max(0, overlap), fw)
    if oy > 0:
        ry = np.linspace(0.0, 1.0, oy, dtype=np.float32)
        if y0 > 0:
            wgt[:oy, :] *= ry[:, None]
        if y1 < H:
            wgt[fh - oy:, :] *= ry[::-1][:, None]
    if ox > 0:
        rx = np.linspace(0.0, 1.0, ox, dtype=np.float32)
        if x0 > 0:
            wgt[:, :ox] *= rx[None, :]
        if x1 < W:
            wgt[:, fw - ox:] *= rx[::-1][None, :]
    return wgt


def _measure_stackmffv4(images, progress, should_cancel,
                        model_path=None, use_gpu=False):
    from fusion_methods.stackmffv4 import _get_model_and_device

    if model_path is None:
        from utils import resource_path
        model_path = resource_path("weights", "stackmffv4.pth")
    model, device = _get_model_and_device(model_path, use_gpu)

    import torch

    h, w = images[0].shape[:2]
    n = len(images)
    from core.multi_focus_fusion import STACKMFFV4_IMAGE_BUDGET
    budget_px = STACKMFFV4_IMAGE_BUDGET * 1024 * 1024  # 与融合路径同一预算

    def _gray01(f, y0, y1, x0, x1):
        crop = images[f][y0:y1, x0:x1]
        return torch.from_numpy(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)

    def forward_and_stitch(coords, block, batch, ov):
        acc = np.zeros((h, w), np.float32)
        wsum = np.zeros((h, w), np.float32)
        padded = ((block - 1) // 128 + 1) * 128  # 网络要求 128 倍数
        done = 0
        for batch_start in range(0, len(coords), batch):
            _tick(progress, done, len(coords), should_cancel)
            chunk = coords[batch_start:batch_start + batch]
            inp = torch.zeros(len(chunk), n, padded, padded)
            for i, (x0, y0, x1, y1) in enumerate(chunk):
                for f in range(n):
                    inp[i, f, :y1 - y0, :x1 - x0] = _gray01(f, y0, y1, x0, x1)
            with torch.no_grad():
                _, focus = model(inp.to(device))
            focus = focus.cpu().numpy()  # [B, Hp, Wp]
            for i, (x0, y0, x1, y1) in enumerate(chunk):
                fh, fw = y1 - y0, x1 - x0
                fmap = focus[i, :fh, :fw].astype(np.float32)
                wgt = _tile_weight(x0, y0, x1, y1, fw, fh, w, h, ov)
                acc[y0:y1, x0:x1] += fmap * wgt
                wsum[y0:y1, x0:x1] += wgt
            done += len(chunk)
            del inp, focus
        wsum[wsum == 0] = 1.0
        return acc / wsum

    # 整栈一次前向就装得下预算时不要分块：小图上瓦片化还会让 overlap
    # 逼近瓦片尺寸、step 退化成逐像素
    if n * h * w <= budget_px:
        _tick(progress, 0, 1, should_cancel)
        padded_h = ((h - 1) // 128 + 1) * 128
        padded_w = ((w - 1) // 128 + 1) * 128
        inp = torch.zeros(1, n, padded_h, padded_w)
        for f in range(n):
            inp[0, f, :h, :w] = _gray01(f, 0, h, 0, w)
        with torch.no_grad():
            _, focus = model(inp.to(device))
        idx = focus[0, :h, :w].cpu().numpy().astype(np.float32)
        del inp, focus
    else:
        from core.multi_focus_fusion import _stackmffv4_effective_tiling
        block, batch, coords = _stackmffv4_effective_tiling(n, 1024, h, w, 256, 2)
        ov = max(0, min(256, block // 2))
        idx = forward_and_stitch(coords, block, batch, ov)
    return np.clip(idx.round(), 0, n - 1).astype(np.int32)


# --------------------------------------------------------------------------
# public entry
# --------------------------------------------------------------------------
def compute_focus_index(images: Sequence[np.ndarray],
                         method: str,
                         kernel_size: int = 7,
                         block_size: int = 8,
                         model_path: Optional[str] = None,
                         use_gpu: bool = False,
                         progress_callback: Optional[Callable] = None,
                         should_cancel: Optional[Callable[[], bool]] = None,
                         ) -> np.ndarray:
    """Focus-position map: float32 [H, W] in [0, 1] (normalized frame index).

    `images` are 0-255-domain BGR frames (uint8 or float). The output is
    always full stack size — dct decides winners on its block grid and the
    map is upscaled back, mirroring how dct fusion upscales its masks.
    """
    _check(images)
    if images[0].dtype == np.uint16:
        # 与融合管线同口径：0-65535 -> 0-255 float（详见 normalize_fuse_input）
        images = [f.astype(np.float32) * (255.0 / 65535.0) for f in images]
    n = len(images)
    if method == "guided_filter":
        idx = _measure_gff(images, progress_callback, should_cancel)
    elif method == "dct":
        idx = _measure_dct(images, progress_callback, should_cancel,
                           kernel_size=kernel_size, block_size=block_size)
    elif method == "gfgfgf":
        idx = _measure_gfgfgf(images, progress_callback, should_cancel,
                              kernel_size=kernel_size)
    elif method == "dtcwt":
        idx = _measure_dtcwt(images, progress_callback, should_cancel)
    elif method == "stackmffv4":
        idx = _measure_stackmffv4(images, progress_callback, should_cancel,
                                  model_path=model_path, use_gpu=use_gpu)
    else:
        raise ValueError(f"Unsupported method for depth maps: {method}")
    return _norm_index(idx, n)


def smooth_index_map(index01: np.ndarray, strength: int = 0) -> np.ndarray:
    """Edge-aware smoothing for display: median first (kills speckle),
    then a small bilateral pass if strength >= 3. Returns float32 [0,1]."""
    m = index01.astype(np.float32)
    if strength <= 0:
        return m
    k = min(2 * int(strength) + 1, 15)
    u8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
    u8 = cv2.medianBlur(u8, k if k % 2 == 1 else k + 1)
    m = u8.astype(np.float32) / 255.0
    if strength >= 3:
        m = cv2.bilateralFilter(m, 9, 0.15, 9)
    return np.clip(m, 0.0, 1.0)
