# Reference:
# 付宏语, 巩岩, 汪路涵, 等. 多聚焦显微图像融合算法[J]. Laser & Optoelectronics Progress, 2024, 61(6): 0618022-0618022-9.


import concurrent.futures
import math
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from fusion_methods.dct import _ensure_color_image
from utils.image_utils import fuse_output_dtype
from core.frame_storage import create_frame, store_frame, capture_scratch, run_with_scratch

_U8_NORM_LUT = np.arange(256, dtype=np.float32) / np.float32(255.0)
_ROW_PARALLEL_PIXELS = 512 * 512


def _load_kernels():
    """Load cached Numba kernels when the startup compiler budget allows."""
    from core.resource_policy import ResourceLimitError
    try:
        from fusion_methods import _gfg_kernels
        return _gfg_kernels
    except (ImportError, ResourceLimitError):
        return None


def _reflect101_span(lo, hi, rad, n):
    if rad <= 0 or n <= 1:
        return lo, hi
    if n <= 2 * rad + 1:
        return 0, n
    r_lo = max(0, lo - rad)
    r_hi = min(n, hi + rad)
    if lo < rad:
        r_hi = max(r_hi, min(n, rad - lo + 1))
    if hi > n - rad:
        r_lo = min(r_lo, max(0, 2 * (n - 1) - (hi - 1 + rad)))
    return r_lo, r_hi


def _gf_bounds(lo_win, hi_win, rad, n):
    k = 2 * rad + 1
    if n <= k:
        return 0, n, 0, n
    a0 = max(0, lo_win - rad)
    a1 = min(n, hi_win + rad + 1)
    q0 = max(0, a0 - rad)
    q1 = min(n, a1 + rad)
    return a0, a1, q0, q1


def _dispatch_strips(executor, workers, start, stop, fn, check_cancel, min_rows=32):
    total = stop - start
    if total <= 0:
        return
    if workers <= 1 or executor is None or total < min_rows * 2:
        fn(start, stop)
        return
    strips = min(workers, max(1, total // min_rows))
    if strips <= 1:
        fn(start, stop)
        return
    step = (total + strips - 1) // strips
    futures = []
    for y in range(start, stop, step):
        e = min(stop, y + step)
        futures.append(executor.submit(fn, y, e))
    pending = set(futures)
    try:
        while pending:
            check_cancel()
            done, pending = concurrent.futures.wait(
                pending, timeout=0.05, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                fut.result()
    except BaseException:
        for fut in futures:
            fut.cancel()
        concurrent.futures.wait(futures)
        raise


# -----------------------------------------------------------------------------
# 全局辅助与核心算法实现
# -----------------------------------------------------------------------------

def _fast_guided_filter_impl(I, p, r, eps, s=4):
    """
    Python 实现的快速导向滤波 (He et al. 2015)。
    通过下采样 (s) 显著加速系数计算。
    """
    h, w = I.shape[:2]
    h_sub = int(h / s)
    w_sub = int(w / s)

    if h_sub < 1 or w_sub < 1:
        I_sub = I
        p_sub = p
        r_sub = r
    else:
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

    mean_I = cv2.boxFilter(I_sub, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p_sub, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I_sub * p_sub, cv2.CV_32F, ksize)
    mean_II = cv2.boxFilter(I_sub * I_sub, cv2.CV_32F, ksize)

    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    if h_sub < 1 or w_sub < 1:
        q = mean_a * I + mean_b
    else:
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
        q = mean_a * I + mean_b

    return q


try:
    _cv_guided_filter = cv2.ximgproc.guidedFilter
    _HAS_XIMGPROC = True
    def _run_guided_filter(guide, src, radius, eps):
        return _cv_guided_filter(guide, src, radius, eps)
except AttributeError:
    _HAS_XIMGPROC = False
    def _run_guided_filter(guide, src, radius, eps):
        return _fast_guided_filter_impl(guide, src, radius, eps, s=4)

_DEFAULT_RUN_GUIDED_FILTER = _run_guided_filter

_GFG_TEMPLATE = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
_SHIFT_KERNELS = [np.zeros((3, 3), np.float32) for _ in range(4)]
for _k, (_r, _c) in zip(_SHIFT_KERNELS, ((1, 2), (1, 0), (2, 1), (0, 1))):
    _k[_r, _c] = 1.0
GFG_THRESHOLD = 0.005  # 论文 T0


def gfg_gradient(gray):
    """类高斯四邻域梯度（GFG）算子，论文式 (3)–(8)。"""
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
    """单帧最终聚焦信息图 IF（论文式 (1)–(10)）。"""
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


_DEFAULT_GFG_FOCUS_MAP = gfg_focus_map


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
    """流式逐像素决策：取聚焦度量最大的帧；并列（差值 ≤ tie_eps）时比较 3×3 Sobel 梯度。"""

    def __init__(self, tie_eps=1e-6):
        self.tie_eps = float(tie_eps)
        self.val = self.idx = self.sob = None

    def update(self, i, afm):
        kernels = _load_kernels()
        if kernels is not None and afm.dtype == np.float32:
            afm_c = np.ascontiguousarray(afm)
            if self.val is None:
                self.val = create_frame(afm_c.shape, np.float32)
                self.sob = create_frame(afm_c.shape, np.float32)
                self.idx = create_frame(afm_c.shape, np.int32)
                kernels.gfg_sobel_decision_rows(
                    afm_c, self.val, self.sob, self.idx,
                    int(i), self.tie_eps, True, 0, afm_c.shape[0],
                )
                return
            kernels.gfg_sobel_decision_rows(
                afm_c, self.val, self.sob, self.idx,
                int(i), self.tie_eps, False, 0, afm_c.shape[0],
            )
            return

        gx = cv2.Sobel(afm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(afm, cv2.CV_32F, 0, 1, ksize=3)
        sob = np.abs(gx) + np.abs(gy)
        if self.val is None:
            self.val = store_frame(afm.astype(np.float32, copy=False))
            self.idx = create_frame(afm.shape, np.int32)
            self.idx[...] = i
            self.sob = store_frame(sob)
            return
        diff = afm - self.val
        win = diff > self.tie_eps
        win |= (np.abs(diff) <= self.tie_eps) & (sob > self.sob)
        self.val[win] = afm[win]
        self.sob[win] = sob[win]
        self.idx[win] = i


_DEFAULT_GFG_DECISION_UPDATE = GfgDecision.update


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
    GFG-FGF 优化实现版（支持 OpenCL GPU 加速与多核低内存流式 CPU 回退）。
    """
    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    check_cancel()
    scratch = capture_scratch()
    if isinstance(input_source, str):
        from fusion_methods.openfocus_ab import _collect_directory_sources

        if not os.path.exists(input_source):
            raise ValueError(f"Path not found: {input_source}")

        stack_ori, img_paths = _collect_directory_sources(input_source, should_cancel)
        if not img_paths:
            raise ValueError("No images found in folder")
    else:
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
    from fusion_methods._openfocus_gpu import _iter_prefetched

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

    num_imgs = len(stack_ori)
    total_steps = num_imgs * 2
    if progress_callback is not None:
        progress_callback(0, total_steps)

    h, w = stack_ori[0].shape[:2]
    out_dtype = fuse_output_dtype(stack_ori)
    kernels = _load_kernels()

    if thread_count is None:
        max_workers = min(12, max(1, os.cpu_count() or 4))
    else:
        try:
            max_workers = max(1, int(thread_count))
        except Exception:
            max_workers = min(12, max(1, os.cpu_count() or 4))

    pixels = h * w
    hooks_unpatched = (
        gfg_focus_map is _DEFAULT_GFG_FOCUS_MAP
        and GfgDecision.update is _DEFAULT_GFG_DECISION_UPDATE
        and _run_guided_filter is _DEFAULT_RUN_GUIDED_FILTER
        and not _HAS_XIMGPROC
    )
    h_sub = h // 4
    w_sub = w // 4
    exact_div4 = h_sub >= 1 and w_sub >= 1 and (h % 4 == 0) and (w % 4 == 0)
    odd_or_one_ksize = (kernel_size <= 1 or kernel_size % 2 == 1)

    # Row-strip parallelism keeps peak RAM bounded (~1-2 live frames) on large images
    # while saturating all CPU cores.
    row_parallel = kernels is not None and pixels >= _ROW_PARALLEL_PIXELS and max_workers > 1
    row_workers = max_workers if row_parallel else 1
    frame_workers = 1 if (row_parallel or pixels >= 4_000_000) else min(max_workers, 8)
    prefetch_slots = 0
    if row_parallel and num_imgs >= 2:
        from core import resource_policy as _rp
        from fusion_methods._openfocus_gpu import _prefetch_window
        _snap = _rp.probe_memory()
        _budget = _rp.ram_ceiling(_snap)
        _frame_est = max(1, pixels * 3 * 2)
        _staging_est = pixels * 20 + _frame_est
        prefetch_slots = _prefetch_window(
            _budget, _staging_est, _frame_est, num_imgs,
            max_slots=2, thread_count=max_workers,
        )

    @carry_warp_accelerator
    def _resolve_raw(index, cancel_fn=None):
        cancel = cancel_fn if cancel_fn is not None else should_cancel
        if cancel is not None and cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()
        with without_source_cache_growth():
            raw = resolve_frame(stack_ori[index], should_cancel=cancel)
        if cancel is not None and cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()
        return raw

    def normalized_pixels(img, channel=None, rows=None):
        if rows is not None:
            img = img[rows]
        if channel is not None:
            img = img[:, :, channel]
        if img.dtype == np.uint16:
            img = img.astype(np.float32) * (255.0 / 65535.0)
        return img.astype(np.float32) / 255.0

    def _valid_raw_channels(raw):
        return raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] in (3, 4))

    def _extract_channel_from_raw(raw, ch, out_buf=None, strip_exec=None):
        if kernels is not None and _valid_raw_channels(raw):
            if raw.shape[:2] != (h, w):
                raise ValueError("GFG-FGF requires frames with matching shapes")
            contig = np.ascontiguousarray(raw)
            out = out_buf if out_buf is not None else np.empty((h, w), dtype=np.float32)
            if contig.dtype == np.uint8:
                _dispatch_strips(
                    strip_exec, row_workers, 0, h,
                    lambda s, e, c=contig, o=out: kernels.extract_channel_u8_rows(c, ch, _U8_NORM_LUT, o, s, e),
                    check_cancel,
                )
                return out
            if contig.dtype == np.uint16:
                _dispatch_strips(
                    strip_exec, row_workers, 0, h,
                    lambda s, e, c=contig, o=out: kernels.extract_channel_u16_rows(c, ch, o, s, e),
                    check_cancel,
                )
                return out
            if contig.dtype in (np.float32, np.float64):
                f32 = contig if contig.dtype == np.float32 else contig.astype(np.float32)
                _dispatch_strips(
                    strip_exec, row_workers, 0, h,
                    lambda s, e, c=f32, o=out: kernels.extract_channel_f32_rows(c, ch, o, s, e),
                    check_cancel,
                )
                return out
        color = _ensure_color_image(raw)
        res = normalized_pixels(color, ch)
        if out_buf is not None:
            out_buf[...] = res
            return out_buf
        return res

    def _extract_channel(index, ch):
        raw = _resolve_raw(index)
        return _extract_channel_from_raw(raw, ch)

    # Dominant channel selection from frame 0 without retaining a full float32 BGR copy on uint8.
    first_raw = _resolve_raw(0)
    if first_raw.ndim == 2:
        channel_idx = 0
    elif first_raw.ndim == 3 and first_raw.shape[2] >= 3 and first_raw.dtype == np.uint8:
        sums = [int(first_raw[:, :, c].sum(dtype=np.uint64)) for c in range(3)]
        channel_idx = int(np.argmax(sums))
    else:
        first_bgr = _ensure_color_image(first_raw)
        first_norm = normalized_pixels(first_bgr)
        del first_bgr
        channel_idx = int(np.argmax(first_norm.sum(axis=(0, 1), dtype=np.float64)))
        del first_norm
    del first_raw

    strip_pool = ThreadPoolExecutor(max_workers=row_workers) if row_parallel else None
    try:
        # ---------------------------------------------------------------------
        # 3. Optional frame focus score pre-filter
        # ---------------------------------------------------------------------
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
            score = np.mean(gx[1:-1, 1:-1] ** 2 + gy[1:-1, 1:-1] ** 2)
            return idx, score

        if min_frame_focus > 0:
            if row_parallel:
                gray_scratch = np.empty((h, w), dtype=np.float32)
                for idx in range(num_imgs):
                    raw = _resolve_raw(idx)
                    _extract_channel_from_raw(raw, channel_idx, gray_scratch, strip_pool)
                    del raw
                    if h > 2 and w > 2:
                        step = max(64, (h - 2 + row_workers - 1) // row_workers)
                        futs = [
                            strip_pool.submit(kernels.scharr_focus_score_rows, gray_scratch, y, min(h - 1, y + step))
                            for y in range(1, h - 1, step)
                        ]
                        focus_vals[idx] = np.float32(sum(f.result() for f in futs) / ((h - 2) * (w - 2)))
                del gray_scratch
            else:
                with ThreadPoolExecutor(max_workers=frame_workers) as executor:
                    for start in range(0, num_imgs, frame_workers):
                        check_cancel()
                        for i, val in executor.map(partial(run_with_scratch, scratch, compute_focus_score),
                                                   range(start, min(num_imgs, start + frame_workers))):
                            check_cancel()
                            focus_vals[i] = val

        max_focus = np.max(focus_vals) if num_imgs > 0 else 1.0
        if max_focus == 0:
            max_focus = 1.0

        # ---------------------------------------------------------------------
        # 4. Focus maps + streaming GfgDecision (Pass 2)
        # ---------------------------------------------------------------------
        scale = float(min_frame_focus)
        g_msz = int(kernel_size)
        g_gsz = 5
        g_eps = 0.3
        r_sub = max(1, int(g_gsz / 4))

        decision = GfgDecision()
        use_fused_pass2 = (
            kernels is not None
            and hooks_unpatched
            and exact_div4
            and odd_or_one_ksize
            and row_parallel
            and out_dtype == np.uint8
        )

        if use_fused_pass2:
            # Fused Pass 2: zero full-frame IA/mean_a/mean_b/afm/gx/gy allocations!
            gray_buf = np.empty((h, w), dtype=np.float32)
            I_sub = np.empty((h_sub, w_sub), dtype=np.float32)
            p_sub = np.empty((h_sub, w_sub), dtype=np.float32)
            val_buf = create_frame((h, w), np.float32)
            sob_buf = create_frame((h, w), np.float32)
            idx_buf = create_frame((h, w), np.int32)

            valid_k = [i for i in range(num_imgs) if not (scale > 0 and focus_vals[i] < scale * max_focus)]
            valid_src = [stack_ori[i] for i in valid_k]
            v_slots = min(prefetch_slots, max(0, len(valid_k) - 1)) if valid_k else 0

            def _prefetch_valid(pos, _src, cancelled):
                return run_with_scratch(scratch, _resolve_raw, valid_k[pos], cancelled)

            pref = _iter_prefetched(
                valid_src, _prefetch_valid, enabled=v_slots > 0,
                should_cancel=should_cancel, max_in_flight=max(1, v_slots),
            )
            first_valid = True
            next_p = 0
            try:
                for pos, raw in pref:
                    i = valid_k[pos]
                    while next_p < i:
                        if progress_callback is not None:
                            progress_callback(next_p + 1, total_steps)
                        next_p += 1
                    check_cancel()
                    _extract_channel_from_raw(raw, channel_idx, gray_buf, strip_pool)
                    del raw
                    _dispatch_strips(
                        strip_pool, row_workers, 0, h_sub,
                        lambda s, e: kernels.gfg_focus_ia_area4_rows(
                            gray_buf, g_msz, GFG_THRESHOLD, I_sub, p_sub, s, e,
                        ),
                        check_cancel, min_rows=16,
                    )
                    mean_a_sub, mean_b_sub = kernels.fgf_coeffs_exact(I_sub, p_sub, r_sub, g_eps)
                    fv = first_valid
                    _dispatch_strips(
                        strip_pool, row_workers, 0, h,
                        lambda s, e, ma=mean_a_sub, mb=mean_b_sub, fi=i, f=fv:
                            kernels.fgf_upsample4_sobel_decision_rows(
                                ma, mb, gray_buf, val_buf, sob_buf, idx_buf,
                                fi, decision.tie_eps, f, s, e,
                            ),
                        check_cancel, min_rows=64,
                    )
                    del mean_a_sub, mean_b_sub
                    first_valid = False
                    if progress_callback is not None:
                        progress_callback(i + 1, total_steps)
                    next_p = i + 1
                while next_p < num_imgs:
                    if progress_callback is not None:
                        progress_callback(next_p + 1, total_steps)
                    next_p += 1
            finally:
                pref.close()
            del gray_buf, I_sub, p_sub, val_buf, sob_buf
            if first_valid:
                idx_buf[...] = 0
            idm_map = idx_buf
            del decision
        elif row_parallel and hooks_unpatched and odd_or_one_ksize:
            # Non-div4 large frame: row-parallel IA + Sobel decision, 1 frame live at a time.
            gray_buf = np.empty((h, w), dtype=np.float32)
            ia_buf = np.empty((h, w), dtype=np.float32)
            val_buf = create_frame((h, w), np.float32)
            sob_buf = create_frame((h, w), np.float32)
            idx_buf = create_frame((h, w), np.int32)

            valid_k = [i for i in range(num_imgs) if not (scale > 0 and focus_vals[i] < scale * max_focus)]
            valid_src = [stack_ori[i] for i in valid_k]
            v_slots = min(prefetch_slots, max(0, len(valid_k) - 1)) if valid_k else 0

            def _prefetch_valid2(pos, _src, cancelled):
                return run_with_scratch(scratch, _resolve_raw, valid_k[pos], cancelled)

            pref = _iter_prefetched(
                valid_src, _prefetch_valid2, enabled=v_slots > 0,
                should_cancel=should_cancel, max_in_flight=max(1, v_slots),
            )
            first_valid = True
            next_p = 0
            try:
                for pos, raw in pref:
                    i = valid_k[pos]
                    while next_p < i:
                        if progress_callback is not None:
                            progress_callback(next_p + 1, total_steps)
                        next_p += 1
                    check_cancel()
                    _extract_channel_from_raw(raw, channel_idx, gray_buf, strip_pool)
                    del raw
                    _dispatch_strips(
                        strip_pool, row_workers, 0, h,
                        lambda s, e: kernels.gfg_focus_ia_rows(gray_buf, g_msz, GFG_THRESHOLD, ia_buf, s, e),
                        check_cancel, min_rows=64,
                    )
                    afm = _run_guided_filter(gray_buf, ia_buf, g_gsz, g_eps)
                    fv = first_valid
                    _dispatch_strips(
                        strip_pool, row_workers, 0, h,
                        lambda s, e, a=afm, fi=i, f=fv:
                            kernels.gfg_sobel_decision_rows(
                                a, val_buf, sob_buf, idx_buf, fi, decision.tie_eps, f, s, e,
                            ),
                        check_cancel, min_rows=64,
                    )
                    del afm
                    first_valid = False
                    if progress_callback is not None:
                        progress_callback(i + 1, total_steps)
                    next_p = i + 1
                while next_p < num_imgs:
                    if progress_callback is not None:
                        progress_callback(next_p + 1, total_steps)
                    next_p += 1
            finally:
                pref.close()
            del gray_buf, ia_buf, val_buf, sob_buf
            if first_valid:
                idx_buf[...] = 0
            idm_map = idx_buf
            del decision
        else:
            def compute_afm_map(i):
                check_cancel()
                if scale > 0 and focus_vals[i] < scale * max_focus:
                    return i, None
                return i, gfg_focus_map(_extract_channel(i, channel_idx), g_msz, g_gsz, g_eps)

            with ThreadPoolExecutor(max_workers=frame_workers) as executor:
                for start in range(0, num_imgs, frame_workers):
                    check_cancel()
                    for i, res in executor.map(partial(run_with_scratch, scratch, compute_afm_map),
                                               range(start, min(num_imgs, start + frame_workers))):
                        check_cancel()
                        if res is not None:
                            decision.update(i, res)
                        del res
                        if progress_callback is not None:
                            progress_callback(i + 1, total_steps)
            if decision.idx is None:
                decision.update(0, np.zeros((h, w), np.float32))
            idm_map = decision.idx
            del decision

        # ---------------------------------------------------------------------
        # 5. Decision refinement & weighted fusion (Pass 3)
        # ---------------------------------------------------------------------
        sum_fdms = create_frame((h, w), np.float32)
        imfu_result = create_frame((h, w, 3), np.float32)
        sum_fdms[...] = 0
        imfu_result[...] = 0

        if kernels is not None:
            if row_parallel:
                step = max(64, (h + row_workers - 1) // row_workers)
                futs = [
                    strip_pool.submit(kernels.active_bboxes_i32_rows, idm_map, num_imgs, y, min(h, y + step))
                    for y in range(0, h, step)
                ]
                wins = np.zeros(num_imgs, np.int64)
                ymin = np.full(num_imgs, h, np.int32)
                ymax = np.full(num_imgs, -1, np.int32)
                xmin = np.full(num_imgs, w, np.int32)
                xmax = np.full(num_imgs, -1, np.int32)
                for fut in futs:
                    w_p, y0_p, y1_p, x0_p, x1_p = fut.result()
                    wins += w_p
                    np.minimum(ymin, y0_p, out=ymin)
                    np.maximum(ymax, y1_p, out=ymax)
                    np.minimum(xmin, x0_p, out=xmin)
                    np.maximum(xmax, x1_p, out=xmax)
            else:
                wins, ymin, ymax, xmin, xmax = kernels.active_bboxes_i32_rows(idm_map, num_imgs, 0, h)
        else:
            wins = np.bincount(idm_map.ravel(), minlength=num_imgs)

        use_fast_roi_pass3 = (
            kernels is not None
            and not _HAS_XIMGPROC
            and _run_guided_filter is _DEFAULT_RUN_GUIDED_FILTER
            and exact_div4
            and out_dtype == np.uint8
        )

        def _accumulate_raw(raw, fdm_weight):
            if kernels is not None and _valid_raw_channels(raw):
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
                rows_sl = slice(row_s, min(h, row_s + 128))
                imfu_result[rows_sl] += normalized_pixels(image, rows=rows_sl) * fdm_weight[rows_sl, :, None]

        def _fuse_active_roi_frame(i, raw, strip_exec):
            if not (_valid_raw_channels(raw) and raw.dtype in (np.uint8, np.uint16, np.float32, np.float64)):
                guide = _extract_channel_from_raw(raw, channel_idx, strip_exec=strip_exec)
                fdm = _run_guided_filter(guide, (idm_map == i).astype(np.float32), g_gsz, g_eps)
                del guide
                _accumulate_raw(raw, fdm)
                del fdm
                return

            contig = np.ascontiguousarray(raw)
            y0_k, y1_k = int(ymin[i]), int(ymax[i])
            x0_k, x1_k = int(xmin[i]), int(xmax[i])
            win_sy0 = y0_k >> 2
            win_sy1 = (y1_k >> 2) + 1
            win_sx0 = x0_k >> 2
            win_sx1 = (x1_k >> 2) + 1

            say0, say1, smy0, smy1 = _gf_bounds(win_sy0, win_sy1 - 1, r_sub, h_sub)
            sax0, sax1, smx0, smx1 = _gf_bounds(win_sx0, win_sx1 - 1, r_sub, w_sub)
            sub_y0, sub_y1 = _reflect101_span(say0, say1, r_sub, h_sub)
            sub_x0, sub_x1 = _reflect101_span(sax0, sax1, r_sub, w_sub)
            sub_h, sub_w = sub_y1 - sub_y0, sub_x1 - sub_x0

            I_sub = np.empty((sub_h, sub_w), dtype=np.float32)
            p_sub = np.empty((sub_h, sub_w), dtype=np.float32)
            if contig.dtype == np.uint8:
                _dispatch_strips(
                    strip_exec, row_workers, 0, sub_h,
                    lambda s, e: kernels.area4_channel_and_mask_roi_u8(
                        contig, _U8_NORM_LUT, channel_idx, idm_map, i,
                        sub_y0, sub_x0, sub_h, sub_w, I_sub, p_sub, s, e,
                    ),
                    check_cancel, min_rows=16,
                )
            elif contig.dtype == np.uint16:
                _dispatch_strips(
                    strip_exec, row_workers, 0, sub_h,
                    lambda s, e: kernels.area4_channel_and_mask_roi_u16(
                        contig, channel_idx, idm_map, i,
                        sub_y0, sub_x0, sub_h, sub_w, I_sub, p_sub, s, e,
                    ),
                    check_cancel, min_rows=16,
                )
            else:
                f32 = contig if contig.dtype == np.float32 else contig.astype(np.float32)
                _dispatch_strips(
                    strip_exec, row_workers, 0, sub_h,
                    lambda s, e: kernels.area4_channel_and_mask_roi_f32(
                        f32, channel_idx, idm_map, i,
                        sub_y0, sub_x0, sub_h, sub_w, I_sub, p_sub, s, e,
                    ),
                    check_cancel, min_rows=16,
                )

            mean_a, mean_b = kernels.fgf_coeffs_exact_roi(
                I_sub, p_sub, h_sub, w_sub,
                sub_y0, sub_x0, say0, say1, sax0, sax1,
                smy0, smy1, smx0, smx1, r_sub, g_eps,
            )
            del I_sub, p_sub

            out_y0 = max(0, smy0 * 4 - 4)
            out_y1 = min(h, smy1 * 4 + 4)
            out_x0 = max(0, smx0 * 4 - 4)
            out_x1 = min(w, smx1 * 4 + 4)

            if contig.dtype == np.uint8:
                _dispatch_strips(
                    strip_exec, row_workers, out_y0, out_y1,
                    lambda s, e: kernels.upsample4_accum_roi_u8_rows(
                        contig, _U8_NORM_LUT, channel_idx, mean_a, mean_b,
                        h_sub, w_sub, smy0, smy1, smx0, smx1,
                        out_x0, out_x1, imfu_result, sum_fdms, s, e,
                    ),
                    check_cancel, min_rows=64,
                )
            elif contig.dtype == np.uint16:
                _dispatch_strips(
                    strip_exec, row_workers, out_y0, out_y1,
                    lambda s, e: kernels.upsample4_accum_roi_u16_rows(
                        contig, channel_idx, mean_a, mean_b,
                        h_sub, w_sub, smy0, smy1, smx0, smx1,
                        out_x0, out_x1, imfu_result, sum_fdms, s, e,
                    ),
                    check_cancel, min_rows=64,
                )
            else:
                f32 = contig if contig.dtype == np.float32 else contig.astype(np.float32)
                _dispatch_strips(
                    strip_exec, row_workers, out_y0, out_y1,
                    lambda s, e: kernels.upsample4_accum_roi_f32_rows(
                        f32, channel_idx, mean_a, mean_b,
                        h_sub, w_sub, smy0, smy1, smx0, smx1,
                        out_x0, out_x1, imfu_result, sum_fdms, s, e,
                    ),
                    check_cancel, min_rows=64,
                )
            del contig, mean_a, mean_b

        if use_fast_roi_pass3:
            active_k = [i for i in range(num_imgs) if wins[i]]
            active_src = [stack_ori[i] for i in active_k]
            act_slots = min(prefetch_slots, max(0, len(active_k) - 1)) if active_k else 0

            def _prefetch_active(pos, _src, cancelled):
                return run_with_scratch(scratch, _resolve_raw, active_k[pos], cancelled)

            pref = _iter_prefetched(
                active_src, _prefetch_active, enabled=act_slots > 0,
                should_cancel=should_cancel, max_in_flight=max(1, act_slots),
            )
            next_p = 0
            try:
                for pos, raw in pref:
                    i = active_k[pos]
                    while next_p < i:
                        if progress_callback is not None:
                            progress_callback(num_imgs + next_p + 1, total_steps)
                        next_p += 1
                    check_cancel()
                    _fuse_active_roi_frame(i, raw, strip_pool)
                    del raw
                    if progress_callback is not None:
                        progress_callback(num_imgs + i + 1, total_steps)
                    next_p = i + 1
                while next_p < num_imgs:
                    if progress_callback is not None:
                        progress_callback(num_imgs + next_p + 1, total_steps)
                    next_p += 1
            finally:
                pref.close()
        else:
            def compute_fusion_component(i):
                check_cancel()
                if not wins[i]:
                    return None
                raw = _resolve_raw(i)
                guide = _extract_channel_from_raw(raw, channel_idx)
                maxfm_binary = (idm_map == i).astype(np.float32)
                fdm = _run_guided_filter(guide, maxfm_binary, g_gsz, g_eps)
                del guide, maxfm_binary
                return i, raw, fdm

            with ThreadPoolExecutor(max_workers=frame_workers) as executor:
                for start in range(0, num_imgs, frame_workers):
                    check_cancel()
                    stop = min(num_imgs, start + frame_workers)
                    active = [i for i in range(start, stop) if wins[i]]
                    for res in executor.map(partial(run_with_scratch, scratch, compute_fusion_component), active):
                        check_cancel()
                        if res is None:
                            continue
                        _i, raw, fdm_weight = res
                        _accumulate_raw(raw, fdm_weight)
                        del raw, res, fdm_weight
                    if progress_callback is not None:
                        progress_callback(num_imgs + stop, total_steps)

        # ---------------------------------------------------------------------
        # 6. Normalization & output
        # ---------------------------------------------------------------------
        out = create_frame((h, w, 3), out_dtype)

        if kernels is not None:
            has_zero = False
            if row_parallel:
                step = max(64, (h + row_workers - 1) // row_workers)
                futs = [
                    strip_pool.submit(kernels.normalize_gfg_rows, imfu_result, sum_fdms, y, min(h, y + step))
                    for y in range(0, h, step)
                ]
                has_zero = any(f.result() for f in futs)
            else:
                for row_s in range(0, h, 128):
                    check_cancel()
                    if kernels.normalize_gfg_rows(imfu_result, sum_fdms, row_s, min(h, row_s + 128)):
                        has_zero = True
            if has_zero:
                zero_mask = sum_fdms <= 1e-6
                for i in np.unique(idm_map[zero_mask]):
                    raw = _resolve_raw(int(i))
                    if _valid_raw_channels(raw):
                        contig = np.ascontiguousarray(raw)
                        for row_s in range(0, h, 128):
                            check_cancel()
                            stop_r = min(h, row_s + 128)
                            if contig.dtype == np.uint8:
                                kernels.fill_zero_weight_rows_u8(
                                    contig, _U8_NORM_LUT, idm_map, sum_fdms, int(i), imfu_result, row_s, stop_r)
                            elif contig.dtype == np.uint16:
                                kernels.fill_zero_weight_rows_u16(
                                    contig, idm_map, sum_fdms, int(i), imfu_result, row_s, stop_r)
                            else:
                                f32 = contig if contig.dtype == np.float32 else contig.astype(np.float32)
                                kernels.fill_zero_weight_rows_f32(
                                    f32, idm_map, sum_fdms, int(i), imfu_result, row_s, stop_r)
                    else:
                        image = _ensure_color_image(raw)
                        for row_s in range(0, h, 128):
                            check_cancel()
                            rows_sl = slice(row_s, min(h, row_s + 128))
                            sel = zero_mask[rows_sl] & (idm_map[rows_sl] == i)
                            np.copyto(imfu_result[rows_sl], normalized_pixels(image, rows=rows_sl),
                                      where=sel[:, :, None])
                    del raw
            if row_parallel:
                _dispatch_strips(
                    strip_pool, row_workers, 0, h,
                    lambda s, e: kernels.quantize_gfg_rows(imfu_result, out, s, e),
                    check_cancel, min_rows=64,
                )
            else:
                for row_s in range(0, h, 128):
                    check_cancel()
                    kernels.quantize_gfg_rows(imfu_result, out, row_s, min(h, row_s + 128))
        else:
            nonzero_mask = create_frame((h, w), np.bool_)
            for row_s in range(0, h, 128):
                check_cancel()
                rows_sl = slice(row_s, min(h, row_s + 128))
                nonzero_mask[rows_sl] = sum_fdms[rows_sl] > 1e-6

            for c in range(3):
                check_cancel()
                ch_data = imfu_result[:, :, c]
                np.divide(ch_data, sum_fdms, out=ch_data, where=nonzero_mask)
            if not nonzero_mask.all():
                for i in np.unique(idm_map[~nonzero_mask]):
                    image = _ensure_color_image(_resolve_raw(int(i)))
                    for row_s in range(0, h, 128):
                        check_cancel()
                        rows_sl = slice(row_s, min(h, row_s + 128))
                        sel = (~nonzero_mask[rows_sl]) & (idm_map[rows_sl] == i)
                        np.copyto(imfu_result[rows_sl], normalized_pixels(image, rows=rows_sl),
                                  where=sel[:, :, None])
                    del image
            for c in range(3):
                ch_data = imfu_result[:, :, c]
                for row_s in range(0, h, 128):
                    check_cancel()
                    rows_sl = slice(row_s, min(h, row_s + 128))
                    np.multiply(ch_data[rows_sl], 255.0, out=ch_data[rows_sl])
                    np.clip(ch_data[rows_sl], 0, 255, out=ch_data[rows_sl])
                    out[rows_sl, :, c] = ch_data[rows_sl]
    finally:
        if strip_pool is not None:
            strip_pool.shutdown(wait=True, cancel_futures=True)

    check_cancel()
    if progress_callback is not None:
        progress_callback(total_steps, total_steps)
    if device_callback is not None:
        device_callback('CPU')
    return out


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
        print("Running synthesis test...")
        h, w = 1024, 1024
        img1 = np.zeros((h, w, 3), dtype=np.uint8)
        img2 = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(img1, (300, 300), 100, (255, 255, 255), -1)
        cv2.rectangle(img2, (600, 600), (900, 900), (0, 255, 0), -1)
        blur1 = cv2.GaussianBlur(img1, (51, 51), 0)
        blur2 = cv2.GaussianBlur(img2, (51, 51), 0)
        src1 = img1.copy()
        src1[:, 512:] = blur1[:, 512:]
        src2 = img2.copy()
        src2[:, :512] = blur2[:, :512]
        t0 = time.time()
        fused = gfgfgf_impl([src1, src2], None)
        t1 = time.time()
        print(f"Optimization Fusion Time: {t1 - t0:.4f}s for size {w}x{h}")
        cv2.imwrite("test_fused.png", fused)
