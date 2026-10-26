"""Focus-position estimates, returned as float32 normalized frame indices.

Hard selectors estimate a dominant source frame; wavelet and pyramid methods
estimate activity rather than an exact pixel provenance map. These maps are
not physical surface measurements. Input normalization is shared with fusion.
"""
import os
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

# 与融合下拉同序（仅保留"按索引决策"的方法；加权平均类无逐像素决策）
SUPPORTED_METHODS = ("dct", "dtcwt", "gfgfgf", "stackmffv3",
                     "openfocus_b", "openfocus_c")


class DepthMapCancelled(Exception):
    """Raised when should_cancel returns True mid-computation."""


def _check(images: Sequence[np.ndarray]) -> None:
    if len(images) < 2:
        raise ValueError("A focus index map needs at least 2 frames")
    first = images[0]
    if first.ndim != 3 or first.shape[2] < 3:
        raise ValueError("Depth map expects BGR images")
    if any(image.shape != first.shape for image in images):
        raise ValueError("Depth map frames must have matching shapes")
    if min(first.shape[:2]) <= 0:
        raise ValueError("Empty image dimensions")


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
# dct — block variance + double median consistency
# --------------------------------------------------------------------------
def _measure_dct(images, progress, should_cancel, kernel_size=7, block_size=8):
    from scipy.ndimage import median_filter

    if kernel_size % 2 == 0:
        kernel_size += 1
    block_size = max(2, int(block_size))
    h, w = images[0].shape[:2]
    h_trim = ((h + block_size - 1) // block_size) * block_size
    w_trim = ((w + block_size - 1) // block_size) * block_size
    map_h, map_w = h_trim // block_size, w_trim // block_size
    if map_h == 0 or map_w == 0:
        raise ValueError("Block size too large for the image")

    max_var = np.full((map_h, map_w), -1.0, np.float32)
    best = np.zeros((map_h, map_w), np.int32)
    for idx, bgr in enumerate(images):
        _tick(progress, idx, len(images), should_cancel)
        if (h_trim, w_trim) != (h, w):
            bgr = cv2.copyMakeBorder(bgr, 0, h_trim - h, 0, w_trim - w,
                                     cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
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
    best = cv2.resize(best.astype(np.float32), (w_trim, h_trim),
                      interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return best[:h, :w]


# --------------------------------------------------------------------------
# gfgfgf — GFG (Gaussian-like four-neighbourhood gradient) + FGF AFM
# --------------------------------------------------------------------------
def _measure_gfgfgf(images, progress, should_cancel, kernel_size=7,
                    thread_count=None):
    from fusion_methods.gfg_fgf import gfg_focus_map, gfg_decision_index

    if kernel_size % 2 == 0:
        kernel_size += 1
    n = len(images)
    # 与融合同一聚焦图与并列决策（Sobel 小区域度量），深度图才与融合一致
    afms = None
    for idx, bgr in enumerate(images):
        _tick(progress, idx, n, should_cancel)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        afm = gfg_focus_map(g, kernel_size)
        if afms is None:
            afms = np.empty((n,) + afm.shape, np.float32)
        afms[idx] = afm
    best_idx = gfg_decision_index(afms)
    return best_idx


# --------------------------------------------------------------------------
# dtcwt — per-frame wavelet high-pass energy (its activity level)
# --------------------------------------------------------------------------
def _measure_dtcwt(images, progress, should_cancel, nlevels=4):
    """DTCWT high-pass energy per frame.

    nlevels mirrors the fusion's own decomposition depth
    (MultiFocusFusion._fuse_dtcwt defaults to N=4): a map computed at a
    different depth is not "the method's own activity measure".
    """
    try:
        import dtcwt as dtcwt_lib
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("DTCWT depth maps need the dtcwt package") from exc

    transform = dtcwt_lib.Transform2d()
    n = len(images)
    h, w = images[0].shape[:2]
    # 流式 argmax：能量栈不物化
    best_val = best_idx = None
    for idx, bgr in enumerate(images):
        _tick(progress, idx, n, should_cancel)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        coeffs = transform.forward(gray, nlevels=nlevels)
        acc = np.zeros((h, w), np.float32)
        for level_yh in coeffs.highpasses:
            _tick(None, idx, n, should_cancel)
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
# stackmffv3 — the model's own focus indices, tiled within the AI budget
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


# This is the historical whole-frame/tile decision, a result parameter.
# Available RAM must never move this boundary or shrink its spatial tiles.
_NEURAL_WHOLE_FRAME_PIXELS = 80 * 1024 * 1024


def _neural_depth_tiles(images):
    h, w = images[0].shape[:2]
    if len(images) * h * w <= _NEURAL_WHOLE_FRAME_PIXELS:
        return 0, [(0, 0, w, h)]
    from core.multi_focus_fusion import _ai_effective_tiling
    block, _batch, coords = _ai_effective_tiling(len(images), 1024, h, w, 256, 1)
    return max(0, min(256, block // 2)), coords


def _plan_neural_depth(images, model_path, use_gpu, thread_count,
                       resource_preference):
    from fusion_methods.stackmffv3 import _prepare_inference
    if model_path is None:
        from utils import resource_path
        model_path = resource_path("weights", "stackmff_v3_final.pth")
    h, w = images[0].shape[:2]
    _overlap, coords = _neural_depth_tiles(images)
    x0, y0, x1, y1 = max(coords, key=lambda xy: (xy[2] - xy[0]) * (xy[3] - xy[1]))
    crops = [image[y0:y1, x0:x1] for image in images]
    normalization = sum(image.size * 4 for image in images) if any(
        image.dtype == np.uint16 for image in images) else 0
    # Map accumulator, weight, normalization result and clipped publication.
    fixed = normalization + h * w * 4 * 4
    _model, _device, plan = _prepare_inference(
        crops, model_path, use_gpu, thread_count=thread_count,
        resource_preference=resource_preference, fixed_bytes=fixed)
    return plan, model_path


def _measure_stackmffv3(images, progress, should_cancel,
                       model_path=None, use_gpu=False, execution_plan=None,
                       mode="soft"):
    """Soft: expected frame position. Hard: per-pixel most probable frame."""
    from fusion_methods.stackmffv3 import model_probabilities, normalize_mode

    mode = normalize_mode(mode)
    h, w = images[0].shape[:2]
    n = len(images)
    overlap, coords = _neural_depth_tiles(images)
    indices = np.arange(n, dtype=np.float32)[:, None, None]

    def measure(crops):
        probabilities = model_probabilities(
            crops, model_path, use_gpu, sigma=0.0 if mode == "hard" else 0.5,
            should_cancel=should_cancel, execution_plan=execution_plan)
        if mode == "hard":
            return probabilities.argmax(0).astype(np.float32)
        return (probabilities * indices).sum(0).astype(np.float32)

    _tick(progress, 0, len(coords), should_cancel)
    if len(coords) == 1:
        result = measure(images)
        _tick(progress, 1, 1, should_cancel)
        return np.clip(result, 0, n - 1).astype(np.float32)

    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    for done, (x0, y0, x1, y1) in enumerate(coords):
        _tick(None, done, len(coords), should_cancel)
        crops = [img[y0:y1, x0:x1] for img in images]
        fmap = measure(crops)
        fh, fw = y1 - y0, x1 - x0
        weight = _tile_weight(x0, y0, x1, y1, fw, fh, w, h, overlap)
        acc[y0:y1, x0:x1] += fmap * weight
        wsum[y0:y1, x0:x1] += weight
        _tick(progress, done + 1, len(coords), should_cancel)
    wsum[wsum == 0] = 1.0
    result = np.clip(acc / wsum, 0, n - 1).astype(np.float32)
    _tick(None, len(coords), len(coords), should_cancel)
    return result


# --------------------------------------------------------------------------
# openfocus_a/b/c — 各自融合管线的"焦点位置"可视化（三种不同的图）
# --------------------------------------------------------------------------
def _measure_openfocus(images, progress, should_cancel, kernel_size=11):
    """B：argmax of blur(x²) − blur(x)² on the gray channel.

    与 fusion_methods/openfocus_ab.py 的 B 方法度量一致——硬拾取的
    逐像素获胜帧。kernel 滑块即方差窗口（默认半径 5）；流式 argmax
    （严格 > 保持 np.argmax 首胜语义）。
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = max(1, int(kernel_size))
    best_val = best_idx = None
    for idx, bgr in enumerate(images):
        _tick(progress, idx, len(images), should_cancel)
        gray = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2GRAY)
        mean = cv2.blur(gray, (k, k))
        var = cv2.blur(gray * gray, (k, k)) - mean * mean
        if best_val is None:
            best_val, best_idx = var, np.zeros(var.shape, np.int32)
        else:
            hit = var > best_val
            best_val[hit] = var[hit]
            best_idx[hit] = idx
    return best_idx


def _measure_openfocus_c(images, progress, should_cancel, kernel_size=11):
    """C：最细带通层能量 argmax（金字塔法按频带决策的深度侧写）。

    band = gray − blur(gray, σ1)（拉普拉斯金字塔最细层的同款构造），
    W = blur(|band|, window)，逐像素取能量最大帧——与 B 的窗口方差
    （二阶统计）不同，这是带通能量的选择行为。
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    k = max(1, int(kernel_size))
    best_val = best_idx = None
    for idx, bgr in enumerate(images):
        _tick(progress, idx, len(images), should_cancel)
        gray = cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2GRAY)
        band = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
        energy = cv2.blur(np.abs(band), (k, k))
        if best_val is None:
            best_val, best_idx = energy, np.zeros(energy.shape, np.int32)
        else:
            hit = energy > best_val
            best_val[hit] = energy[hit]
            best_idx[hit] = idx
    return best_idx


# --------------------------------------------------------------------------
# public entry
# --------------------------------------------------------------------------
# The depth map must use the same window as the fusion it explains:
# DCT/GFG-FGF fuse with 7, the OpenFocus family with 11 (Radius 5).
_DEPTH_KERNEL_DEFAULTS = {"dct": 7, "gfgfgf": 7,
                          "openfocus_b": 11, "openfocus_c": 11}


def compute_focus_index(images: Sequence[np.ndarray],
                         method: str,
                         kernel_size: Optional[int] = None,
                         block_size: int = 8,
                         model_path: Optional[str] = None,
                         use_gpu: bool = False,
                         progress_callback: Optional[Callable] = None,
                         should_cancel: Optional[Callable[[], bool]] = None,
                         nlevels: int = 4,
                         ai_mode: str = "soft",
                         thread_count: Optional[int] = None,
                         resource_preference: str = 'balanced',
                         ) -> np.ndarray:
    """Focus-position map: float32 [H, W] in [0, 1] (normalized frame index).

    `images` are 0-255-domain BGR frames (uint8 or float). The output is
    always full stack size — dct decides winners on its block grid and the
    map is upscaled back, mirroring how dct fusion upscales its masks.
    `kernel_size=None` uses the fusion default for that method (7 for
    DCT/GFG-FGF, 11 for OpenFocus B/C).
    """
    from core.resource_policy import Workload, heavy_job, plan_execution
    from core.registration import RegistrationCancelled
    from utils.image_utils import normalize_fuse_input

    _check(images)
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method for depth maps: {method}")
    try:
        with heavy_job(should_cancel):
            n = len(images)
            h, w = images[0].shape[:2]
            if method == 'stackmffv3':
                plan, model_path = _plan_neural_depth(
                    images, model_path, use_gpu, thread_count, resource_preference)
            else:
                # The shared CPU model includes normalization, map storage,
                # method scratch and the final normalized output.
                plan = plan_execution(
                    Workload(algorithm=method, frames=n, height=h, width=w,
                             channels=images[0].shape[2],
                             itemsize=max(image.dtype.itemsize for image in images),
                             parallel_layer='native'),
                    preference=resource_preference, user_threads=thread_count)
            _tick(None, 0, n, should_cancel)
            images, _is_16bit = normalize_fuse_input(images)
            kernel = (kernel_size if kernel_size is not None
                      else _DEPTH_KERNEL_DEFAULTS.get(method))
            if method == "dct":
                idx = _measure_dct(images, progress_callback, should_cancel,
                                   kernel_size=kernel, block_size=block_size)
            elif method == "gfgfgf":
                idx = _measure_gfgfgf(images, progress_callback, should_cancel,
                                      kernel_size=kernel)
            elif method == "dtcwt":
                idx = _measure_dtcwt(images, progress_callback, should_cancel,
                                     nlevels=nlevels)
            elif method == "stackmffv3":
                idx = _measure_stackmffv3(
                    images, progress_callback, should_cancel, model_path=model_path,
                    use_gpu=use_gpu, execution_plan=plan, mode=ai_mode)
            elif method == "openfocus_b":
                idx = _measure_openfocus(images, progress_callback, should_cancel,
                                         kernel_size=kernel)
            else:
                idx = _measure_openfocus_c(images, progress_callback, should_cancel,
                                           kernel_size=kernel)
            _tick(None, n, n, should_cancel)
            result = _norm_index(idx, n)
            _tick(None, n, n, should_cancel)
            return result
    except RegistrationCancelled as exc:
        raise DepthMapCancelled() from exc


def smooth_index_map(index01: np.ndarray, strength: int = 0) -> np.ndarray:
    """Edge-aware smoothing for display: median first (kills speckle),
    then a small bilateral pass if strength >= 3. Returns float32 [0,1]."""
    m = index01.astype(np.float32)
    if strength <= 0:
        return m
    k = min(2 * int(strength) + 1, 15)
    from scipy.ndimage import median_filter
    m = median_filter(m, size=k, mode="nearest")
    if strength >= 3:
        m = cv2.bilateralFilter(m, 9, 0.15, 9)
    return np.clip(m, 0.0, 1.0)
