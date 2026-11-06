"""Shared input preparation and frame-order helpers for OpenFocus fusion."""
import os
from typing import List

import cv2
import numpy as np

from utils.sorting import natural_sort_key
from core.frame_storage import create_frame, store_frame


def _collect_images(input_source, *, normalize=True, should_cancel=None) -> List[np.ndarray]:
    """Load frames; public callers retain the historical 0–255 normalization."""
    _check_cancel(should_cancel)
    if isinstance(input_source, (list, tuple)):
        from core.source_frames import materialize_frames
        frames = materialize_frames([f for f in input_source if f is not None], should_cancel)
        if len(frames) < 2:
            raise ValueError("OpenFocus A/B fusion needs at least 2 frames")
        # Same 0-255 domain as the directory loader: a uint16 list used to
        # stay in 0-65535 and saturate on the final np.clip(..., 0, 255).
        if not normalize:
            return frames
        return [store_frame(_prepare_frame(f, None, should_cancel)) if f.dtype == np.uint16
                else f for f in frames]

    if isinstance(input_source, str) and os.path.isdir(input_source):
        from core.image_loader import admit_image_decode, decode_image

        exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp', '.tif', '.tiff', '.bmp')
        files = [f for f in sorted(os.listdir(input_source),
                                    key=natural_sort_key)
                 if f.lower().endswith(exts)]
        if len(files) < 2:
            raise ValueError(
                f"OpenFocus A/B fusion needs at least 2 images in {input_source}")
        frames = []
        for name in files:
            _check_cancel(should_cancel)
            path = os.path.join(input_source, name)
            admit_image_decode(path)
            img = decode_image(path)
            if img is None:
                raise RuntimeError(f"Unable to read image: {name}")
            if normalize and img.dtype == np.uint16:
                img = (img.astype(np.float32) * (255.0 / 65535.0))
            frames.append(store_frame(img))
        return frames

    raise ValueError("input_source must be a list of frames or a directory")


def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def _frame_pass_workers(frames, img_resize, algorithm, thread_count, *, activity_only=False):
    """Admit complete independent producers, including their retained result."""
    if thread_count == 1 or len(frames) < 2:
        return 1
    from core import resource_policy as policy
    from core.multi_focus_fusion import memory_budget_bytes
    h, w = frames[0].shape[:2]
    if img_resize:
        w, h = map(int, img_resize)
    # Parallel producers must fit as complete resident planes. The public
    # estimator admits serial mappings/row scratch and cannot size this window.
    fixed = 4 * 1024 * 1024 + h * w * (12 if activity_only else 16)
    per_worker = h * w * 80
    if img_resize:
        # _prepare_frame must convert uint16 BEFORE resizing to preserve its
        # fractional interpolation. A small destination cannot hide multiple
        # concurrently converted full-size sources from the worker budget.
        for source in frames:
            if source.dtype == np.uint16:
                sh, sw = source.shape[:2]
                channels = source.shape[2] if len(source.shape) == 3 else 1
                per_worker = max(per_worker, sh * sw * channels * 8 + h * w * channels * 4)
    budget = memory_budget_bytes()
    # Serial mapping/strip execution remains the low-RAM path. Parallel
    # producers are admitted only when their full resident envelopes fit.
    if fixed + 2 * per_worker > budget:
        return 1
    plan = policy.plan_execution(policy.Workload(
        algorithm=algorithm, frames=len(frames), height=h, width=w,
        fixed_bytes=fixed, per_worker_bytes=per_worker, outer_tasks=len(frames)),
        preference='maximum' if thread_count is not None else 'balanced', user_threads=thread_count,
        ram_budget_override=budget)
    return plan.workers


def _ordered_frame_results(count, produce, workers, should_cancel):
    """Produce a worker-sized window and yield only in source-index order.

    Producers own their temporary maps. The consumer alone updates global
    accumulators; a completed later frame never changes winner/tie ordering.
    A consumer error or cancellation also stops running producers at their
    next cancellable row/native boundary before their mappings are released.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor, wait
    from threading import Event
    stop = Event()

    def cancelled():
        return stop.is_set() or (should_cancel is not None and should_cancel())

    if workers <= 1:
        for index in range(count):
            _check_cancel(cancelled)
            yield index, produce(index, cancelled)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        window = deque()
        next_index = 0
        try:
            while next_index < min(workers, count):
                window.append(pool.submit(produce, next_index, cancelled))
                next_index += 1
            index = 0
            while window:
                _check_cancel(cancelled)
                for pending in window:
                    if pending.done() and pending.exception() is not None:
                        pending.result()
                if not window[0].done():
                    wait((window[0],), timeout=0.05)
                    continue
                future = window.popleft()
                result = future.result()
                del future
                _check_cancel(cancelled)
                yield index, result
                del result
                index += 1
                _check_cancel(cancelled)
                if next_index < count:
                    window.append(pool.submit(produce, next_index, cancelled))
                    next_index += 1
        finally:
            stop.set()
            for future in window:
                future.cancel()
            window.clear()


def _prepare_frame(frame, img_resize, should_cancel):
    """Convert one frame in rows; retain conversion-before-resize semantics."""
    _check_cancel(should_cancel)
    frame = np.asarray(frame)
    if img_resize is None:
        if frame.dtype == np.float32:
            return frame
        result = create_frame(frame.shape, np.float32)
        for start in range(0, frame.shape[0], 128):
            _check_cancel(should_cancel)
            rows = slice(start, min(frame.shape[0], start + 128))
            if frame.dtype == np.uint16:
                result[rows] = frame[rows].astype(np.float32) * (255.0 / 65535.0)
            else:
                result[rows] = frame[rows]
        return result
    if frame.dtype == np.uint16:
        frame = _prepare_frame(frame, None, should_cancel)
    w, h = img_resize
    frame = cv2.resize(frame, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    return frame if frame.dtype == np.float32 else frame.astype(np.float32)


def _luminance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    take = min(img.shape[2], 3)
    if take == 1:
        return img[:, :, 0]
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)[:take]
    out = cv2.transform(img[:, :, :take], weights.reshape(1, -1))
    return out if out.ndim == 2 else out[:, :, 0]


def _window_variance(gray: np.ndarray, radius: int) -> np.ndarray:
    """窗口局部方差：var = blur(x²) − blur(x)²，box 滤波可分离、O(N)。"""
    k = max(1, int(radius)) * 2 + 1
    g = gray
    mean = cv2.blur(g, (k, k))
    mean_sq = cv2.blur(g * g, (k, k))
    return np.maximum(mean_sq - mean * mean, 0.0)


# ------------------------------------------------------------------
# 帧序检测（排序行为）
# ------------------------------------------------------------------
def frame_sharpness_curve(frames, radius: int = 3, max_dim: int = 256):
    """逐帧标量清晰度：降采样 luma 上的窗口方差均值（诊断用）。

    用与 A/B/C 融合一致的窗口方差度量，在 ≤256px 的小图上计算，
    单帧毫秒级。
    注意：该曲线与帧序**方向**无关（见上方实验结论），仅用于诊断与
    validate_autosort.py 的证据复跑，不参与产品路径的方向决策。
    """
    curve = []
    for f in frames:
        g = _luminance(np.asarray(f).astype(np.float32))
        h, w = g.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / float(max(h, w))
            g = cv2.resize(g, (max(1, int(round(w * s))),
                               max(1, int(round(h * s)))),
                           interpolation=cv2.INTER_AREA)
        curve.append(float(_window_variance(g, radius).mean()))
    return np.asarray(curve, np.float64)


def detect_stack_flip(curve, min_frames: int = 5, threshold: float = -0.7,
                      min_range: float = 0.10) -> bool:
    """清晰度曲线随帧序明确单调递减时返回 True（应当翻转）。

    坍缩保护：帧数不足、曲线平坦（无单调证据）时一律不翻——典型焦点栈
    的全局清晰度是"钟形"（两端模糊中间清晰），秩相关近 0，天然不触发。
    """
    c = np.asarray(curve, np.float64)
    n = int(c.size)
    if n < min_frames:
        return False
    med = float(np.median(c))
    if med <= 0.0:
        return False
    if float(c.max() - c.min()) / med < min_range:
        return False
    # Spearman 秩相关 = 秩上的 Pearson
    ranks = np.argsort(np.argsort(c)).astype(np.float64)
    idx = np.arange(n, dtype=np.float64)
    idx -= idx.mean()
    ranks -= ranks.mean()
    denom = float(np.sqrt((idx * idx).sum() * (ranks * ranks).sum()))
    if denom <= 0.0:
        return False
    rho = float((idx * ranks).sum() / denom)
    return rho <= threshold


# ------------------------------------------------------------------
# 帧序方向（Sort）
# ------------------------------------------------------------------
# 全局清晰度曲线与帧序方向无关——11 个真实栈的 Spearman rho 散布在
# -0.98…+1.00，"单调递减检测"只对 1/11 正确。根本原因：内容是
# 反转变换协变的（反转后的栈同样是合法焦点栈），任何内容信号都
# 无法判定方向。因此方向控制是显式的：
# asc（默认，保持文件序）/ desc（倒序栈由用户翻转）。

def sort_frames_for_openfocus(frames, mode: str = "asc"):
    """按 mode 规范帧序，返回 (排序后帧列表, 是否翻转)。

    mode:
      - "asc" : 保持文件序（默认——所见即所得，方向由用户保证）
      - "desc": 倒序栈——翻转为正序，让末帧先验（A 的平坦退化）、
                基底参考（消除亮度漂移）、深度图标签落在确定的一端
      - "off" : 等同 asc（保留给 UI/CLI 的显式关闭语义）
    """
    frames = list(frames)
    if mode == "desc" and len(frames) > 1:
        return frames[::-1], True
    return frames, False
