"""Focus-position estimates, returned as float32 normalized frame indices.

Hard selectors estimate a dominant source frame; wavelet and pyramid methods
estimate activity rather than an exact pixel provenance map. These maps are
not physical surface measurements. Input normalization is shared with fusion.
"""
import os
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from core.algorithms import get_fusion, resolve_fusion_parameters, resolve_runner
from core.frame_storage import create_frame, store_frame, scratch_directory


class DepthMapCancelled(Exception):
    """Raised when should_cancel returns True mid-computation."""


def _check(images: Sequence[np.ndarray], allow_grayscale=False, should_cancel=None) -> None:
    if len(images) < 2:
        raise ValueError("A focus index map needs at least 2 frames")
    from core.image_loader import _has_transparency
    from core.source_frames import resolve_frame
    for image in images:
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()
        if image.ndim == 3 and image.shape[2] == 4:
            if _has_transparency(resolve_frame(image, should_cancel=should_cancel)):
                raise ValueError("Depth maps of transparent images are unsupported")
    first = images[0]
    if not ((first.ndim == 3 and first.shape[2] >= 3) or
            (allow_grayscale and (first.ndim == 2 or
                                  (first.ndim == 3 and first.shape[2] == 1)))):
        raise ValueError("Depth map expects BGR images or supported grayscale images")
    if any(image.shape != first.shape for image in images):
        raise ValueError("Depth map frames must have matching shapes")
    if min(first.shape[:2]) <= 0:
        raise ValueError("Empty image dimensions")


def _norm_index(argmax_map: np.ndarray, n: int) -> np.ndarray:
    result = create_frame(argmax_map.shape, np.float32)
    if n <= 1:
        result[...] = 0
        return result
    for start in range(0, result.shape[0], 128):
        rows = slice(start, min(result.shape[0], start + 128))
        result[rows] = (argmax_map[rows].astype(np.float32) / (n - 1)).clip(0.0, 1.0)
    return result


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
    from fusion_methods.gfg_fgf import gfg_focus_map, GfgDecision

    if kernel_size % 2 == 0:
        kernel_size += 1
    n = len(images)
    # 与融合同一聚焦图与并列决策（Sobel 小区域度量），深度图才与融合一致
    decision = GfgDecision()
    for idx, bgr in enumerate(images):
        _tick(progress, idx, n, should_cancel)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        decision.update(idx, gfg_focus_map(g, kernel_size))
    _tick(None, n, n, should_cancel)
    return decision.idx


# --------------------------------------------------------------------------
# dtcwt — per-frame wavelet high-pass energy (its activity level)
# --------------------------------------------------------------------------
def _measure_dtcwt(images, progress, should_cancel, nlevels=4):
    """DTCWT high-pass energy per frame.

    nlevels mirrors the fusion's own decomposition depth
    (MultiFocusFusion._fuse_dtcwt defaults to N=4): a map computed at a
    different depth is not "the method's own activity measure".
    """
    # Reuse the fusion adapter's NumPy compatibility setup, including when
    # depth is the first wavelet operation in a fresh process.
    from fusion_methods.dtcwt import dtcwt as dtcwt_lib
    if dtcwt_lib is None:  # pragma: no cover - env dependent
        raise RuntimeError("DTCWT depth maps need dtcwt, scipy and six")

    transform = dtcwt_lib.Transform2d()
    n = len(images)
    h, w = images[0].shape[:2]
    # 流式 argmax：能量栈不物化
    best_val = best_idx = None
    for idx, bgr in enumerate(images):
        _tick(progress, idx, n, should_cancel)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        coeffs = transform.forward(gray, nlevels=nlevels)
        acc = create_frame((h, w), np.float32)
        acc[...] = 0
        for level_yh in coeffs.highpasses:
            _tick(None, idx, n, should_cancel)
            # (H', W', 6) -> 每方向幅值平方求和，上采样回全分辨率累加
            mag2 = np.sum(np.abs(level_yh) ** 2, axis=2).astype(np.float32)
            lvl = cv2.resize(mag2, (w, h), interpolation=cv2.INTER_LINEAR)
            acc += lvl
        if best_val is None:
            best_val = acc
            best_idx = create_frame(acc.shape, np.int32)
            best_idx[...] = 0
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


_NEURAL_WHOLE_FRAME_PIXELS = 80 * 1024 * 1024


def _neural_depth_tiles(images, tile_size):
    h, w = images[0].shape[:2]
    if ((max(h, w) + 63) // 64) * 64 <= ((tile_size + 63) // 64) * 64:
        return 0, [(0, 0, w, h)]
    from core.multi_focus_fusion import _ai_effective_tiling
    block, _batch, coords = _ai_effective_tiling(len(images), tile_size, h, w, 256, 1)
    return min(256, block // 2), coords


def _plan_depth_tiles(images, use_gpu, thread_count, resource_preference):
    from dataclasses import replace
    from core.resource_policy import plan_neural_tiles
    from fusion_methods.neural_runtime import resolve_device, estimate_inference_bytes

    h, w = images[0].shape[:2]
    channels = images[0].shape[2] if images[0].ndim == 3 else 1
    requested = max(h, w) if len(images) * h * w <= _NEURAL_WHOLE_FRAME_PIXELS else 1024
    tile_size, plan = plan_neural_tiles(
        len(images), h, w, channels, requested, resolve_device(use_gpu),
        estimate_inference_bytes, itemsize=max(image.dtype.itemsize for image in images),
        preference=resource_preference, user_threads=thread_count)
    return replace(plan, tile_size=tile_size)


def _plan_neural_depth(images, model_path, use_gpu, thread_count,
                       resource_preference):
    if model_path is None:
        from utils import resource_path
        model_path = resource_path("weights", "stackmff_v3_final.onnx")
    return _plan_depth_tiles(images, use_gpu, thread_count, resource_preference), model_path


def _measure_stackmffv2(images, progress, should_cancel,
                        model_path=None, use_gpu=False, execution_plan=None,
                        thread_count=None, resource_preference='balanced'):
    """Sampled focal layer from the V2 network."""
    from fusion_methods.stackmff_v2 import DEFAULT_WEIGHTS, depth_positions
    from utils.weights import weights_path

    if model_path is None:
        model_path = weights_path(os.path.basename(DEFAULT_WEIGHTS))
    n = len(images)
    tile_size = getattr(execution_plan, 'tile_size', None)
    overlap = min(256, tile_size // 2) if tile_size is not None else 0
    result = depth_positions(
        images, model_path, use_gpu,
        should_cancel=should_cancel, execution_plan=execution_plan,
        thread_count=thread_count, resource_preference=resource_preference,
        tile_size=tile_size, tile_overlap=overlap,
        progress_callback=lambda done, total: _tick(progress, done, total, should_cancel))
    return np.clip(result, 0, n - 1).astype(np.float32)


def _plan_stackmffv2_depth(images, model_path, use_gpu, thread_count,
                           resource_preference):
    """Admit the V2 network for one depth-map pass."""
    from fusion_methods.stackmff_v2 import DEFAULT_WEIGHTS
    from utils.weights import weights_path

    if model_path is None:
        model_path = weights_path(os.path.basename(DEFAULT_WEIGHTS))
    return _plan_depth_tiles(images, use_gpu, thread_count, resource_preference), model_path


def _measure_stackmffv3(images, progress, should_cancel,
                       model_path=None, use_gpu=False, execution_plan=None,
                       mode="soft"):
    """Soft: expected frame position. Hard: per-pixel most probable frame."""
    from fusion_methods.stackmffv3 import model_probabilities, normalize_mode

    mode = normalize_mode(mode)
    h, w = images[0].shape[:2]
    n = len(images)
    tile_size = getattr(execution_plan, 'tile_size', None) or max(h, w)
    overlap, coords = _neural_depth_tiles(images, tile_size)
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
# public entry
# --------------------------------------------------------------------------


def compute_focus_index(images: Sequence[np.ndarray], method: str, parameters=None, *,
                         model_path=None, use_gpu=False, progress_callback=None,
                         should_cancel=None, thread_count=None,
                         resource_preference='balanced', **legacy_parameters):
    """Return normalized focus positions using the descriptor's depth runner.

    Legacy named numeric arguments are translated only at this input boundary.
    """
    from core.resource_policy import Workload, heavy_job, plan_execution
    from core.registration import RegistrationCancelled
    from utils.image_utils import normalize_fuse_input

    spec = get_fusion(method)
    values = resolve_fusion_parameters(method, parameters, depth=True)
    declared_keys = {parameter.key for parameter in spec.depth.parameters}
    for key, value in legacy_parameters.items():
        target = {'nlevels': 'N', 'ai_mode': 'mode'}.get(key, key)
        if target not in declared_keys:
            # ``block_size`` selects the DCT depth grid without being a declared
            # descriptor parameter; it is forwarded to the DCT runner below.
            if key == 'block_size' and method == 'dct':
                continue
            raise ValueError(f'Unknown depth parameter: {key}')
        if value is not None:
            values[target] = value
    values = resolve_fusion_parameters(method, values, depth=True)
    depth_block = legacy_parameters.get('block_size')
    try:
        with heavy_job(should_cancel), scratch_directory():
            if method != 'openfocus_b':
                from core.source_frames import materialize_frames
                images = materialize_frames(images, should_cancel,
                                            thread_count=thread_count,
                                            resource_preference=resource_preference)
            _check(images, allow_grayscale=method in ('openfocus_b', 'openfocus_c'),
                   should_cancel=should_cancel)
            if spec.neural:
                from fusion_methods.neural_runtime import start_inference_job
                start_inference_job()
            n = len(images)
            h, w = images[0].shape[:2]
            if spec.depth_planner:
                plan, model_path = resolve_runner(spec.depth_planner)(
                    images, model_path, use_gpu, thread_count, resource_preference)
            else:
                # The shared CPU model includes normalization, map storage,
                # method scratch and the final normalized output.
                plan = plan_execution(
                    Workload(algorithm=method, frames=n, height=h, width=w,
                             channels=images[0].shape[2] if images[0].ndim == 3 else 1,
                             itemsize=max(image.dtype.itemsize for image in images),
                             parallel_layer='native'),
                    preference=resource_preference, user_threads=thread_count)
            _tick(None, 0, n, should_cancel)
            # Neural runners normalize one spatial tile at a time. B/C prepare
            # one frame at a time, using fusion's stack-wide bit-depth rule.
            if method not in ('openfocus_b', 'openfocus_c', 'stackmffv2', 'stackmffv3'):
                if any(image.dtype == np.uint16 for image in images):
                    normalized = []
                    for image in images:
                        _tick(None, 0, n, should_cancel)
                        converted = normalize_fuse_input([image])[0][0]
                        if converted.dtype != np.float32:
                            converted = converted.astype(np.float32)
                        normalized.append(image if converted is image else store_frame(converted))
                    images = normalized
            runtime = {'progress_callback': progress_callback, 'should_cancel': should_cancel,
                       'thread_count': plan.native_threads, 'use_gpu': use_gpu,
                       'model_path': model_path, 'execution_plan': plan}
            if depth_block is not None:
                runtime['block_size'] = depth_block
            idx = resolve_runner(spec.depth.runner)(images, **runtime, **values)
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
