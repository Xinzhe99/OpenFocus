# Reference:
# 付宏语, 巩岩, 汪路涵, 等. 多聚焦显微图像融合算法[J]. Laser & Optoelectronics Progress, 2024, 61(6): 0618022-0618022-9.


import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fusion_methods.dct import _ensure_color_image
from utils.image_utils import fuse_output_dtype
from core.frame_storage import create_frame, store_frame, capture_scratch, run_with_scratch

_U8_NORM_LUT = np.arange(256, dtype=np.float32) / np.float32(255.0)


def _load_kernels():
    """Load cached Numba kernels when the startup compiler budget allows."""
    from core.resource_policy import ResourceLimitError
    try:
        from fusion_methods import _gfg_kernels
        return _gfg_kernels
    except (ImportError, ResourceLimitError):
        return None


# -----------------------------------------------------------------------------
# 全局辅助与核心算法实现
# -----------------------------------------------------------------------------

def _fast_guided_filter_impl(I, p, r, eps, s=4):
    """
    Python 实现的快速导向滤波 (He et al. 2015)。
    通过下采样 (s) 显著加速系数计算。
    """
    # 1. 下采样
    h, w = I.shape[:2]
    h_sub = int(h / s)
    w_sub = int(w / s)

    # 避免尺寸过小
    if h_sub < 1 or w_sub < 1:
        I_sub = I
        p_sub = p
        r_sub = r
    else:
        # INTER_AREA, not NEAREST: nearest-neighbour subsampling aliases fine
        # texture into the box statistics (ground-truth benchmark: 39.7 dB
        # vs 40.8 dB with area averaging, which matches opencv-contrib's
        # guidedFilter). This fallback is what pip/installer users run —
        # neither ships opencv-contrib.
        I_sub = cv2.resize(I, (w_sub, h_sub), interpolation=cv2.INTER_AREA)
        p_sub = cv2.resize(p, (w_sub, h_sub), interpolation=cv2.INTER_AREA)
        r_sub = max(1, int(r / s))

    kernels = _load_kernels()
    if kernels is not None and I_sub.dtype == np.float32 and p_sub.dtype == np.float32 and I.dtype == np.float32:
        mean_a, mean_b = kernels.fgf_coeffs_exact(I_sub, p_sub, r_sub, eps)
        if h_sub < 1 or w_sub < 1:
            return kernels.fgf_apply_q(mean_a, I, mean_b)
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
        return kernels.fgf_apply_q(mean_a, I, mean_b)

    ksize = (2 * r_sub + 1, 2 * r_sub + 1)

    # 2. 计算统计量 (Box Filter)
    # 使用 cv2.CV_32F 明确指定深度，避免不必要的推断
    mean_I = cv2.boxFilter(I_sub, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p_sub, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I_sub * p_sub, cv2.CV_32F, ksize)
    mean_II = cv2.boxFilter(I_sub * I_sub, cv2.CV_32F, ksize)

    # 3. 计算线性系数 a, b
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 4. 系数平滑
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    # 5. 上采样回原始尺寸
    if h_sub < 1 or w_sub < 1:
        q = mean_a * I + mean_b
    else:
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
        q = mean_a * I + mean_b

    return q

# 尝试获取 OpenCV 的快速实现，否则使用上面的 Python 优化版
try:
    # 检查是否有 ximgproc 模块
    _cv_guided_filter = cv2.ximgproc.guidedFilter
    _HAS_XIMGPROC = True
    def _run_guided_filter(guide, src, radius, eps):
        return _cv_guided_filter(guide, src, radius, eps)
except AttributeError:
    _HAS_XIMGPROC = False
    # 使用 Python 优化版，默认 s=4 进行加速
    def _run_guided_filter(guide, src, radius, eps):
        return _fast_guided_filter_impl(guide, src, radius, eps, s=4)

# 论文式 (4) 的类高斯卷积模板 T
_GFG_TEMPLATE = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
# 四邻域位移核：filter2D 求 I(x±1, y) / I(x, y±1)
_SHIFT_KERNELS = [np.zeros((3, 3), np.float32) for _ in range(4)]
for _k, (_r, _c) in zip(_SHIFT_KERNELS, ((1, 2), (1, 0), (2, 1), (0, 1))):
    _k[_r, _c] = 1.0
GFG_THRESHOLD = 0.005  # 论文 T0


def gfg_gradient(gray):
    """类高斯四邻域梯度（GFG）算子，论文式 (3)–(8)。

    P = I ⊗ T；G∥ = (|I(x+1,y)−P| + |I(x−1,y)−P|)/2，G⊥ 同理沿 y；
    G = G∥ + G⊥，低于 T0 的置 0。输入为 [0,1] float32 单通道。
    """
    kernels = _load_kernels()
    if kernels is not None and gray.dtype == np.float32:
        return kernels.gfg_gradient_exact(gray, GFG_THRESHOLD)
    p = cv2.filter2D(gray, cv2.CV_32F, _GFG_TEMPLATE, borderType=cv2.BORDER_REFLECT)
    g = np.zeros_like(p)
    for k in _SHIFT_KERNELS:
        nb = cv2.filter2D(gray, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
        g += np.abs(nb - p)
    g *= 0.5
    g[g < GFG_THRESHOLD] = 0.0
    return g


def gfg_focus_map(gray, mean_ksize, gf_radius=5, gf_eps=0.3):
    """单帧最终聚焦信息图 IF（论文式 (1)–(10)）。

    IR = |I − I∗fm|（粗略聚焦图）；G = GFG(I)；IA 在 G ≥ T0 处取 G（边缘
    高频），其余取 IR；IF = FGF(I, IA)。融合与深度图共用这一实现。
    """
    kernels = _load_kernels()
    if kernels is not None and gray.dtype == np.float32 and (mean_ksize <= 1 or mean_ksize % 2 == 1):
        ia = kernels.gfg_focus_ia_exact(gray, int(mean_ksize), GFG_THRESHOLD)
        return _run_guided_filter(gray, ia, gf_radius, gf_eps)
    blur = cv2.blur(gray, (mean_ksize, mean_ksize))
    ia = cv2.absdiff(gray, blur)
    g = gfg_gradient(gray)
    edge = g > 0
    ia[edge] = g[edge]
    return _run_guided_filter(gray, ia, gf_radius, gf_eps)


def _sobel_energy(fm):
    """论文式 (11)–(13)：G3×3 = |Gx| + |Gy|（Sobel 作用于聚焦信息图）。"""
    gx = cv2.Sobel(fm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(fm, cv2.CV_32F, 0, 1, ksize=3)
    kernels = _load_kernels()
    if kernels is not None:
        out = np.empty_like(gx)
        kernels.sobel_abs_sum(gx, gy, out)
        return out
    return np.abs(gx) + np.abs(gy)


class GfgDecision:
    """流式逐像素决策：取聚焦度量最大的帧；并列（差值 ≤ tie_eps，如平坦背景
    上全为 0）时按论文 §2.2 的小区域聚焦度量——比较聚焦信息图的 3×3 Sobel
    梯度 G3×3，取梯度大的帧。只保留 3 张图，内存与帧数无关。"""

    def __init__(self, tie_eps=1e-6):
        self.tie_eps = float(tie_eps)
        self.val = self.idx = self.sob = None

    def update(self, i, afm):
        gx = cv2.Sobel(afm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(afm, cv2.CV_32F, 0, 1, ksize=3)
        kernels = _load_kernels()
        if self.val is None:
            if kernels is not None:
                sob = np.empty_like(gx)
                kernels.sobel_abs_sum(gx, gy, sob)
            else:
                sob = np.abs(gx) + np.abs(gy)
            self.val = store_frame(afm.astype(np.float32, copy=False))
            self.idx = create_frame(afm.shape, np.int32)
            self.idx[...] = i
            self.sob = store_frame(sob)
            return
        if kernels is not None and afm.dtype == np.float32:
            kernels.gfg_decision_update(afm, gx, gy, self.val, self.sob, self.idx, i, self.tie_eps)
            return
        sob = np.abs(gx) + np.abs(gy)
        diff = afm - self.val
        win = diff > self.tie_eps
        win |= (np.abs(diff) <= self.tie_eps) & (sob > self.sob)
        self.val[win] = afm[win]
        self.sob[win] = sob[win]
        self.idx[win] = i


def gfg_decision_index(afms, tie_eps=1e-6):
    """afms 可迭代（按帧序）→ 决策索引图（见 GfgDecision）。"""
    dec = GfgDecision(tie_eps)
    for i, afm in enumerate(afms):
        dec.update(i, afm)
    return dec.idx


def gfgfgf_impl(input_source, img_resize=None, kernel_size=7, thread_count: int = None,
                min_frame_focus=0.0, should_cancel=None, progress_callback=None,
                use_gpu: bool = False, device_callback=None):
    """
    GFG-FGF 优化实现版（支持 OpenCL GPU 加速与 CPU 回退）。
    """
    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    check_cancel()
    scratch = capture_scratch()
    # -------------------------------------------------------------------------
    # 1. 图像加载与预处理
    # -------------------------------------------------------------------------
    if isinstance(input_source, str):
        from fusion_methods.openfocus_ab import _collect_directory_sources

        # 文件夹读取逻辑
        if not os.path.exists(input_source):
            raise ValueError(f"Path not found: {input_source}")

        stack_ori, img_paths = _collect_directory_sources(input_source, should_cancel)
        if not img_paths:
            raise ValueError("No images found in folder")
    else:
        # 列表输入
        stack_ori = input_source

    if not stack_ori:
        raise ValueError("Input stack is empty")

    if progress_callback is not None:
        callback = progress_callback
        reported = -1

        def progress_callback(index, total):
            nonlocal reported
            if index > reported:
                callback(index, total)
                reported = index

    if use_gpu:
        from core.source_frames import SOURCE_CACHE
        from fusion_methods._openfocus_gpu import GpuError, log_fallback
        from fusion_methods._openfocus_gpu_gfg import GpuGfg

        try:
            gpu = GpuGfg()
            result = gpu.fuse(
                stack_ori, img_resize, kernel_size, min_frame_focus,
                should_cancel=should_cancel,
                progress_callback=progress_callback,
            )
            if device_callback is not None:
                device_callback(gpu.device_name)
            return result
        except GpuError as exc:
            log_fallback(exc)
        finally:
            SOURCE_CACHE.clear()

    from core.registration import carry_warp_accelerator
    from core.source_frames import process_frame, resolve_frame, without_source_cache_growth

    resize = (img_resize is not None
              and (stack_ori[0].shape[1], stack_ori[0].shape[0]) != img_resize)
    if resize:
        prepared = []

        def prepare(image):
            image = _ensure_color_image(image)
            return cv2.resize(image, img_resize)

        for img in stack_ori:
            check_cancel()
            shape = (img_resize[1], img_resize[0], 3)
            prepared.append(process_frame(img, prepare, shape=shape, should_cancel=should_cancel))
        stack_ori = prepared
        del prepared

    # Retain source frames in their existing representation and convert per pass,
    # rather than retaining a second full float32 color stack.
    num_imgs = len(stack_ori)
    total_steps = num_imgs * 2
    if progress_callback is not None:
        progress_callback(0, total_steps)

    h, w = stack_ori[0].shape[:2]
    out_dtype = fuse_output_dtype(stack_ori)
    kernels = _load_kernels()

    @carry_warp_accelerator
    def _resolve_raw(index):
        check_cancel()
        with without_source_cache_growth():
            raw = resolve_frame(stack_ori[index], should_cancel=should_cancel)
        check_cancel()
        return raw

    def normalized_pixels(img, channel=None, rows=None):
        if rows is not None:
            img = img[rows]
        if channel is not None:
            img = img[:, :, channel]
        if img.dtype == np.uint16:
            img = img.astype(np.float32) * (255.0 / 65535.0)
        return img.astype(np.float32) / 255.0

    def _extract_channel(index, ch):
        raw = _resolve_raw(index)
        if kernels is not None and (raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] in (3, 4))):
            if raw.shape[:2] != (h, w):
                raise ValueError("GFG-FGF requires frames with matching shapes")
            contig = np.ascontiguousarray(raw)
            del raw
            if contig.dtype == np.uint8:
                out = np.empty((h, w), dtype=np.float32)
                kernels.extract_channel_u8(contig, ch, _U8_NORM_LUT, out)
                return out
            if contig.dtype == np.uint16:
                out = np.empty((h, w), dtype=np.float32)
                kernels.extract_channel_u16(contig, ch, out)
                return out
            if contig.dtype in (np.float32, np.float64):
                out = np.empty((h, w), dtype=np.float32)
                f32 = contig if contig.dtype == np.float32 else contig.astype(np.float32)
                kernels.extract_channel_f32(f32, ch, out)
                return out
            raw = contig
        color = _ensure_color_image(raw)
        del raw
        return normalized_pixels(color, ch)

    first_raw = _resolve_raw(0)
    first_bgr = _ensure_color_image(first_raw)
    del first_raw
    first_norm = normalized_pixels(first_bgr)
    del first_bgr
    channel_idx = int(np.argmax(first_norm.sum(axis=(0, 1))))
    del first_norm

    # -------------------------------------------------------------------------
    # 3. 计算焦点值 (Focus Measure) - 阶段 1
    # -------------------------------------------------------------------------
    kx = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)
    ky = kx.T.copy()
    focus_vals = np.zeros(num_imgs, dtype=np.float32)

    def compute_focus_score(idx):
        check_cancel()
        gray_img = _extract_channel(idx, channel_idx)
        if kernels is not None:
            return idx, float(kernels.scharr_focus_score_exact(gray_img))
        gx = cv2.filter2D(gray_img, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT)
        gy = cv2.filter2D(gray_img, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT)
        sub_gx = gx[1:-1, 1:-1]
        sub_gy = gy[1:-1, 1:-1]
        score = np.mean(sub_gx**2 + sub_gy**2)
        return idx, score

    # 决定线程池大小：优先使用传入 thread_count，否则使用原先的策略（上限8）
    if thread_count is None:
        max_workers = min(8, os.cpu_count() or 4)
    else:
        try:
            max_workers = max(1, int(thread_count))
        except Exception:
            max_workers = min(8, os.cpu_count() or 4)

    if min_frame_focus > 0:  # 整帧分数只服务于可选的跳帧
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start in range(0, num_imgs, max_workers):
                check_cancel()
                for i, val in executor.map(partial(run_with_scratch, scratch, compute_focus_score),
                                           range(start, min(num_imgs, start + max_workers))):
                    check_cancel()
                    focus_vals[i] = val

    max_focus = np.max(focus_vals) if num_imgs > 0 else 1.0
    if max_focus == 0:
        max_focus = 1.0

    # -------------------------------------------------------------------------
    # 4. 计算初始决策图 (AFMs) - 阶段 2
    # -------------------------------------------------------------------------
    scale = float(min_frame_focus)
    g_msz = kernel_size
    g_gsz = 5
    g_eps = 0.3

    def compute_afm_map(i):
        check_cancel()
        if scale > 0 and focus_vals[i] < scale * max_focus:
            return i, None
        return i, gfg_focus_map(_extract_channel(i, channel_idx), g_msz, g_gsz, g_eps)

    decision = GfgDecision()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, num_imgs, max_workers):
            check_cancel()
            for i, res in executor.map(partial(run_with_scratch, scratch, compute_afm_map),
                                       range(start, min(num_imgs, start + max_workers))):
                check_cancel()
                if res is not None:
                    decision.update(i, res)
                del res
                if progress_callback is not None:
                    progress_callback(i + 1, total_steps)
    if decision.idx is None:  # 全部被跳过：退化为首帧
        decision.update(0, np.zeros((h, w), np.float32))

    # -------------------------------------------------------------------------
    # 5. 决策与融合 (Fusion) - 阶段 3
    # -------------------------------------------------------------------------
    idm_map = decision.idx
    del decision

    sum_fdms = create_frame((h, w), np.float32)
    imfu_result = create_frame((h, w, 3), np.float32)
    sum_fdms[...] = 0
    imfu_result[...] = 0

    if kernels is not None:
        wins = kernels.bincount_i32(idm_map, num_imgs)
    else:
        wins = np.bincount(idm_map.ravel(), minlength=num_imgs)

    h_sub = h // 4
    w_sub = w // 4
    use_fast_sub_mask = (kernels is not None and not _HAS_XIMGPROC
                         and h_sub >= 1 and w_sub >= 1 and (h % 4 == 0) and (w % 4 == 0))

    def compute_fusion_component(i):
        check_cancel()
        if not wins[i]:
            return None
        guide = _extract_channel(i, channel_idx)
        if use_fast_sub_mask:
            I_sub = cv2.resize(guide, (w_sub, h_sub), interpolation=cv2.INTER_AREA)
            p_sub = kernels.area4_mask_from_idx(idm_map, i, h_sub, w_sub)
            r_sub = max(1, int(g_gsz / 4))
            mean_a, mean_b = kernels.fgf_coeffs_exact(I_sub, p_sub, r_sub, g_eps)
            mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
            mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
            fdm = kernels.fgf_apply_q(mean_a, guide, mean_b)
        else:
            maxfm_binary = (idm_map == i).astype(np.float32)
            fdm = _run_guided_filter(guide, maxfm_binary, g_gsz, g_eps)
        return i, fdm

    def _accumulate_raw(raw, fdm_weight):
        if kernels is not None and (raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] in (3, 4))):
            contig = np.ascontiguousarray(raw)
            if contig.dtype == np.uint8:
                kernels.accumulate_gfg_u8(contig, _U8_NORM_LUT, fdm_weight, imfu_result, sum_fdms)
                return
            if contig.dtype == np.uint16:
                kernels.accumulate_gfg_u16(contig, fdm_weight, imfu_result, sum_fdms)
                return
            if contig.dtype in (np.float32, np.float64):
                f32 = contig if contig.dtype == np.float32 else contig.astype(np.float32)
                kernels.accumulate_gfg_f32(f32, fdm_weight, imfu_result, sum_fdms)
                return
        np.add(sum_fdms, fdm_weight, out=sum_fdms)
        image = _ensure_color_image(raw)
        for row_s in range(0, h, 128):
            check_cancel()
            rows = slice(row_s, min(h, row_s + 128))
            imfu_result[rows] += normalized_pixels(image, rows=rows) * fdm_weight[rows, :, None]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start in range(0, num_imgs, max_workers):
            check_cancel()
            stop = min(num_imgs, start + max_workers)
            active = [i for i in range(start, stop) if wins[i]]
            for res in executor.map(partial(run_with_scratch, scratch, compute_fusion_component), active):
                check_cancel()
                if res is None:
                    continue
                i, fdm_weight = res
                raw = _resolve_raw(i)
                _accumulate_raw(raw, fdm_weight)
                del raw, res, fdm_weight
            if progress_callback is not None:
                progress_callback(num_imgs + stop, total_steps)

    # -------------------------------------------------------------------------
    # 6. 归一化与输出
    # -------------------------------------------------------------------------
    out = create_frame((h, w, 3), out_dtype)

    if kernels is not None:
        has_zero = False
        for row_s in range(0, h, 128):
            check_cancel()
            if kernels.normalize_gfg_rows(imfu_result, sum_fdms, row_s, min(h, row_s + 128)):
                has_zero = True
        if has_zero:
            zero_mask = sum_fdms <= 1e-6
            for i in np.unique(idm_map[zero_mask]):
                raw = _resolve_raw(int(i))
                if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] in (3, 4)):
                    contig = np.ascontiguousarray(raw)
                    for row_s in range(0, h, 128):
                        check_cancel()
                        stop_r = min(h, row_s + 128)
                        if contig.dtype == np.uint8:
                            kernels.fill_zero_weight_rows_u8(contig, _U8_NORM_LUT, idm_map, sum_fdms, int(i), imfu_result, row_s, stop_r)
                        elif contig.dtype == np.uint16:
                            kernels.fill_zero_weight_rows_u16(contig, idm_map, sum_fdms, int(i), imfu_result, row_s, stop_r)
                        else:
                            f32 = contig if contig.dtype == np.float32 else contig.astype(np.float32)
                            kernels.fill_zero_weight_rows_f32(f32, idm_map, sum_fdms, int(i), imfu_result, row_s, stop_r)
                else:
                    image = _ensure_color_image(raw)
                    for row_s in range(0, h, 128):
                        check_cancel()
                        rows = slice(row_s, min(h, row_s + 128))
                        sel = zero_mask[rows] & (idm_map[rows] == i)
                        np.copyto(imfu_result[rows], normalized_pixels(image, rows=rows),
                                  where=sel[:, :, None])
                del raw
        for row_s in range(0, h, 128):
            check_cancel()
            kernels.quantize_gfg_rows(imfu_result, out, row_s, min(h, row_s + 128))
    else:
        nonzero_mask = create_frame((h, w), np.bool_)
        for row_s in range(0, h, 128):
            check_cancel()
            rows = slice(row_s, min(h, row_s + 128))
            nonzero_mask[rows] = sum_fdms[rows] > 1e-6

        for c in range(3):
            check_cancel()
            ch_data = imfu_result[:, :, c]
            np.divide(ch_data, sum_fdms, out=ch_data, where=nonzero_mask)
        if not nonzero_mask.all():
            for i in np.unique(idm_map[~nonzero_mask]):
                image = _ensure_color_image(_resolve_raw(int(i)))
                for row_s in range(0, h, 128):
                    check_cancel()
                    rows = slice(row_s, min(h, row_s + 128))
                    sel = (~nonzero_mask[rows]) & (idm_map[rows] == i)
                    np.copyto(imfu_result[rows], normalized_pixels(image, rows=rows),
                              where=sel[:, :, None])
                del image
        for c in range(3):
            ch_data = imfu_result[:, :, c]
            for row_s in range(0, h, 128):
                check_cancel()
                rows = slice(row_s, min(h, row_s + 128))
                np.multiply(ch_data[rows], 255.0, out=ch_data[rows])
                np.clip(ch_data[rows], 0, 255, out=ch_data[rows])
                out[rows, :, c] = ch_data[rows]

    check_cancel()
    if progress_callback is not None:
        progress_callback(total_steps, total_steps)
    if device_callback is not None:
        device_callback('CPU')
    return out

# -----------------------------------------------------------------------------
# 测试入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) > 1:
        folder = sys.argv[1]
        t0 = time.time()
        out = gfgfgf_impl(folder, None)
        t1 = time.time()
        print(f"Fusion done in {t1 - t0:.4f}s")
        cv2.imwrite('gfg_fused_opt.png', out)
    else:
        # 生成较大的测试数据
        print("Running synthesis test...")
        h, w = 1024, 1024
        img1 = np.zeros((h, w, 3), dtype=np.uint8)
        img2 = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 绘制图案
        cv2.circle(img1, (300, 300), 100, (255, 255, 255), -1)
        cv2.rectangle(img2, (600, 600), (900, 900), (0, 255, 0), -1)
        
        # 模糊模拟散焦
        blur1 = cv2.GaussianBlur(img1, (51, 51), 0)
        blur2 = cv2.GaussianBlur(img2, (51, 51), 0)
        
        # 输入源：左边清晰，右边模糊 vs 左边模糊，右边清晰
        src1 = img1.copy()
        src1[:, 512:] = blur1[:, 512:]
        
        src2 = img2.copy()
        src2[:, :512] = blur2[:, :512]
        
        input_list = [src1, src2]
        
        t0 = time.time()
        fused = gfgfgf_impl(input_list, None)
        t1 = time.time()
        
        print(f"Optimization Fusion Time: {t1 - t0:.4f}s for size {w}x{h}")
        # cv2.imshow("Fused", fused)
        # cv2.waitKey(0)
        cv2.imwrite("test_fused.png", fused)