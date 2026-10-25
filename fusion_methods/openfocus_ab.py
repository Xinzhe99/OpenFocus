"""OpenFocus A/B/C —— 三条互补的融合管线。

- 清晰度度量 = 窗口局部方差（blur(x²) − blur(x)²）。
- Method B（深度图选择）：argmax(var) → 深度图平滑 → 按图硬拾取；
  硬拾取优于软混合，平局取首帧。
- Method A（加权平均）：w = gauss(var)，分母带 +1 的末帧先验
  （平坦区权重全零时精确退化为末帧）。
- Method C（金字塔）：7 层拉普拉斯金字塔逐带 max-pick 硬拾取
  （|带|高斯 σ3 权重取 argmax；max-pick 优于软归一混合），
  bicubic ×2 + σ0.6 展开，最粗带（基带）整带取末帧
  （低频来自参考帧 + 高频按频段独立决策）。

约定：输入为 0-255 域的 BGR 帧列表（uint8 或 float32），
输出经 fuse_output_dtype 归一（与 gff/dct 相同的位深契约）。
"""
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.image_utils import fuse_output_dtype
from utils.sorting import natural_sort_key


def _collect_images(input_source) -> List[np.ndarray]:
    """接受帧列表或目录路径，返回 0-255 域 BGR 帧列表。"""
    if isinstance(input_source, (list, tuple)):
        frames = [f for f in input_source if f is not None]
        if len(frames) < 2:
            raise ValueError("OpenFocus A/B fusion needs at least 2 frames")
        # Same 0-255 domain as the directory loader: a uint16 list used to
        # stay in 0-65535 and saturate on the final np.clip(..., 0, 255).
        return [f.astype(np.float32) * (255.0 / 65535.0)
                if f.dtype == np.uint16 else f for f in frames]

    if isinstance(input_source, str) and os.path.isdir(input_source):
        exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp', '.tif', '.tiff', '.bmp')
        files = [f for f in sorted(os.listdir(input_source),
                                    key=natural_sort_key)
                 if f.lower().endswith(exts)]
        if len(files) < 2:
            raise ValueError(
                f"OpenFocus A/B fusion needs at least 2 images in {input_source}")
        frames = []
        for name in files:
            data = np.fromfile(os.path.join(input_source, name), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Unable to read image: {name}")
            if img.dtype == np.uint16:
                img = (img.astype(np.float32) * (255.0 / 65535.0))
            frames.append(img)
        return frames

    raise ValueError("input_source must be a list of frames or a directory")


def _as_float(frames) -> List[np.ndarray]:
    out = []
    for f in frames:
        arr = f if f.dtype == np.float32 else f.astype(np.float32)
        out.append(arr)
    return out


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


def openfocus_a_impl(input_source, img_resize=None, kernel_size: int = 11,
                     thread_count: int = None, blend_smoothing: float = 2.0):
    """OpenFocus - A（方差加权平均，末帧先验）。

    w_i = gauss(var_i, σ=blend_smoothing) ；
    out = (Σ w_i·I_i + I_last) / (Σ w_i + 1)。
    +1 的末帧先验保证平坦区（权重全零）精确退化为末帧——这正是
    Method A 平局行为（输出恒等于末帧）的成因。

    `blend_smoothing` 是 Helicon 的 "Smoothing"：权重图的平滑尺度，
    决定清晰区之间的过渡宽度（大 = 更柔、细节更软）。
    """
    frames = _collect_images(input_source)
    if img_resize is not None:
        w, h = img_resize
        frames = [cv2.resize(f, (int(w), int(h)),
                             interpolation=cv2.INTER_AREA) for f in frames]
    frames_f = _as_float(frames)

    radius = max(1, int(kernel_size) // 2) if kernel_size else 3
    # Smoothing：对权重图做轻度高斯（深度过渡平滑参数）
    sigma = max(0.0, float(2.0 if blend_smoothing is None
                          else blend_smoothing))

    h, w = frames_f[0].shape[:2]
    channels = frames_f[0].shape[2] if frames_f[0].ndim == 3 else 1
    acc = np.zeros((h, w, channels), dtype=np.float32)
    wsum = np.zeros((h, w, 1), dtype=np.float32)
    # One frame at a time: the variance map, its smoothed weights and the
    # float frame are added and dropped inside the loop. Holding the whole
    # (N, H, W) variance stack plus the whole weight stack (what this used to
    # do) is three extra full-stack float32 buffers for no result change —
    # the arithmetic per frame and the accumulation order are identical.
    for frame in frames_f:
        w2d = _window_variance(_luminance(frame), radius)
        if sigma > 1e-3:
            w2d = cv2.GaussianBlur(w2d, (0, 0), sigmaX=sigma)
        w2d = w2d[:, :, np.newaxis]
        # 2D 灰度帧统一到 (h, w, 1)：直接 (h,w)*(h,w,1) 会广播成 (h,w,w)
        f3 = frame[:, :, np.newaxis] if frame.ndim == 2 else frame
        acc += f3 * w2d
        wsum += w2d
    # 末帧先验（+1）：平坦区权重全零时输出精确等于末帧
    last = frames_f[-1]
    acc += last[:, :, np.newaxis] if last.ndim == 2 else last
    wsum += 1.0
    fused = acc / wsum
    if last.ndim == 2:
        fused = fused[:, :, 0]
    return np.clip(fused, 0, 255).astype(fuse_output_dtype(frames))


DEPTH_SMOOTHING_DEFAULT = 1.2  # σ of the depth-map transition (default)


def openfocus_b_impl(input_source, img_resize=None, kernel_size: int = 11,
                     thread_count: int = None,
                     depth_smoothing: float = DEPTH_SMOOTHING_DEFAULT):
    """OpenFocus - B（方差深度图选择）。

    两遍结构：argmax(var) → 深度图平滑（边界 ±3 帧线性过渡）→ 硬拾取。
    np.argmax 天然平局取首帧。
    """
    frames = _collect_images(input_source)
    if img_resize is not None:
        w, h = img_resize
        frames = [cv2.resize(f, (int(w), int(h)),
                             interpolation=cv2.INTER_AREA) for f in frames]
    frames_f = _as_float(frames)

    radius = max(1, int(kernel_size) // 2) if kernel_size else 3
    n = len(frames_f)
    h, w = frames_f[0].shape[:2]

    # Pass 1: running argmax. ``strict`` greater-than keeps np.argmax's
    # first-frame tie-break, and only the current maximum and its index stay
    # alive instead of the whole (N, H, W) variance stack.
    best = np.full((h, w), -np.inf, dtype=np.float32)
    winner = np.zeros((h, w), dtype=np.int32)
    for i, frame in enumerate(frames_f):
        var = _window_variance(_luminance(frame), radius)
        win = var > best
        best[win] = var[win]
        winner[win] = i
    del best

    depth = winner.astype(np.float32)
    # 深度图平滑：先中值去噪点，再高斯做边界过渡（约 ±3 帧），
    # 最后取整——硬拾取（软混合会降低局部对比度）。
    # 注意用 float32 做中值：转 uint8 会在帧数 >256 时截断深度索引
    # （显微深栈可达数百帧）；对整数输入两者结果逐位一致。
    depth = cv2.medianBlur(np.ascontiguousarray(depth, np.float32), 3)
    # depth_smoothing = 过渡带高斯 σ（过渡宽度随其近线性增宽）。
    # 默认 1.2；
    # 0 = 只做中值去噪点（细节最锐，真值基准上 +0.5 dB），调大可压制
    # 噪声栈在深度跳变处的斑点/光晕。
    sigma = max(0.0, float(depth_smoothing if depth_smoothing is not None
                           else DEPTH_SMOOTHING_DEFAULT))
    if sigma > 0.05:
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=sigma)
    idx = np.clip(np.rint(depth), 0, n - 1).astype(np.int32)
    del depth, winner

    # Pass 2: hard pick straight out of the selected source frame. Building
    # an (N, H, W, C) float stack just to gather one pixel per position cost
    # a full-stack buffer; this touches each needed frame once.
    out_dtype = fuse_output_dtype(frames)
    fused = np.empty((h, w, frames_f[0].shape[2]
                      if frames_f[0].ndim == 3 else 1), dtype=np.float32)
    for i in np.unique(idx):
        sel = idx == i
        frame = frames_f[i]
        src = frame if frame.ndim == 3 else frame[:, :, np.newaxis]
        fused[sel] = src[sel]
    if frames_f[0].ndim == 2:
        fused = fused[:, :, 0]
    return np.clip(fused, 0, 255).astype(out_dtype)


# ------------------------------------------------------------------
# OpenFocus - C（拉普拉斯金字塔融合）
# ------------------------------------------------------------------

def _pyramid_gauss(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """高斯金字塔：σ=1 模糊 + 2× 抽点（与复现管线一致的 Burt 型分解）。"""
    g = [img.astype(np.float32)]
    for _ in range(levels - 1):
        b = cv2.GaussianBlur(g[-1], (0, 0), 1.0)
        g.append(b[::2, ::2])
    return g


def _pyramid_expand(band: np.ndarray, shape) -> np.ndarray:
    """2× 上采样（bicubic 插值 + σ=0.6 模糊），裁到目标形状。

    最终验证配方（reproduce_and_compare.repro_C）：PIL BICUBIC + gauss
    σ0.6；此处用 cv2.INTER_CUBIC 等价实现（审计实测三栈 MAE 与 PIL 版
    逐点一致），避免融合热路径依赖 Pillow。
    band 可以是 (h, w) 或 (h, w, C)。
    """
    up = cv2.resize(band, (band.shape[1] * 2, band.shape[0] * 2),
                    interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(up[:shape[0], :shape[1]], (0, 0), 0.6)


def openfocus_c_impl(input_source, img_resize=None, kernel_size: int = 7,
                     thread_count: int = None, levels: int = 7,
                     blend_smoothing: float = 3.0):
    """OpenFocus - C：拉普拉斯金字塔融合。

    每个带通层的权重 = |亮度带| 的高斯平滑（σ=blend_smoothing），逐像素
    取 argmax 硬拾取（max-pick）——实测 max-pick 明显优于软归一混合
    （stackA 上 0.51% vs 1.35% MAE）；最粗的基带整带取末帧（亮度基准）。
    层数自适应：min(levels, log2(min(H,W))-1)，小图不产生退化带。

    `blend_smoothing` 是 Helicon 的 "Smoothing"（Method C 唯一的可调
    参数）：决定各帧带通在多宽的尺度上被比较/过渡。σ 越大过渡越柔、
    细节越软；σ→0 等价于逐像素裸 argmax（最锐、最易斑点）。`kernel_size`
    仅为兼容 A/B 的签名保留，C 不使用空间窗口。
    """
    frames = _collect_images(input_source)
    if img_resize is not None:
        w, h = img_resize
        frames = [cv2.resize(f, (int(w), int(h)),
                             interpolation=cv2.INTER_AREA) for f in frames]
    frames_f = _as_float(frames)

    h, w = frames_f[0].shape[:2]
    # 小图保护：每层减半，至少留 2px；levels 至少 2（一层带通 + 基带）
    usable = max(2, min(levels, int(np.log2(max(2, min(h, w)))) - 1))
    sigma = max(0.0, float(3.0 if blend_smoothing is None
                          else blend_smoothing))

    # 逐帧流式决策：每层只保留当前最优带与最优能量，而不是整栈金字塔 +
    # 整栈带通堆栈 + 整栈权重堆栈。严格 `>` 更新保持 np.argmax 的首帧
    # 平局行为；每帧的分解、层间减法和 σ 平滑与原来逐位相同。
    best_energy = [None] * usable
    out_bands = [None] * usable
    for frame_index, frame in enumerate(frames_f):
        pyr = _pyramid_gauss(frame, usable)
        for k in range(usable):
            if k < usable - 1:
                up = _pyramid_expand(pyr[k + 1], pyr[k].shape[:2])
                band = pyr[k] - up
                del up
                # 权重：亮度域带通幅值，σ 平滑（opacity 平滑）。
                if band.ndim == 3:
                    lum = (band * np.array([0.114, 0.587, 0.299],
                                           np.float32)).sum(axis=2)
                else:
                    lum = band
                energy = (cv2.GaussianBlur(np.abs(lum), (0, 0), sigma)
                          if sigma > 1e-3 else np.abs(lum))
                if best_energy[k] is None:
                    best_energy[k] = energy
                    out_bands[k] = band.astype(np.float32, copy=True)
                else:
                    win = energy > best_energy[k]
                    if win.any():
                        best_energy[k][win] = energy[win]
                        out_bands[k][win] = band[win]
                    del win
            elif frame_index == len(frames_f) - 1:
                # 基带：整带取末帧（低频来自参考帧）
                out_bands[k] = np.array(pyr[k], dtype=np.float32, copy=True)
        del pyr
    del best_energy

    # 自顶向下重建：expand + 累加
    img = out_bands[-1]
    for k in range(usable - 2, -1, -1):
        img = _pyramid_expand(img, out_bands[k].shape[:2]) + out_bands[k]

    return np.clip(img, 0, 255).astype(fuse_output_dtype(frames))


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
