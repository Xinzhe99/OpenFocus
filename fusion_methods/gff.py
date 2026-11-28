import concurrent.futures
from functools import partial
import os

import cv2
import numpy as np

from core.frame_storage import capture_scratch, create_frame, run_with_scratch, store_frame
from fusion_methods.dct import _ensure_color_image
from utils.image_utils import fuse_output_dtype

# Reference:
# https://github.com/RCharradi/Image-fusion-with-guided-filtering
# Li S, Kang X, Hu J. Image fusion with guided filtering[J]. IEEE Transactions on Image processing, 2013, 22(7): 2864-2875.

_U8_NORM_LUT = np.arange(256, dtype=np.float32) / np.float32(255.0)
_ROW_PARALLEL_PIXELS = 512 * 512


def _load_kernels():
    """Load cached Numba kernels when the startup compiler budget allows."""
    from core.resource_policy import ResourceLimitError
    try:
        from fusion_methods import _gff_kernels
        return _gff_kernels
    except (ImportError, ResourceLimitError):
        return None


def _reflect_span(lo, hi, rad, n):
    if rad <= 0 or n <= 1:
        return lo, hi
    if n <= 2 * rad + 1:
        return 0, n
    r_lo = max(0, lo - rad)
    r_hi = min(n, hi + rad)
    if lo < rad:
        r_hi = max(r_hi, min(n, rad - lo))
    if hi > n - rad:
        r_lo = min(r_lo, max(0, 2 * n - hi - rad))
    return r_lo, r_hi


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
    """Run fn(s, e) across disjoint row strips [start:stop] using executor."""
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


def gff_impl(input_source, img_resize, kernel_size=31, thread_count: int = None,
             should_cancel=None, progress_callback=None,
             use_gpu: bool = False, device_callback=None):
    """
    基于引导滤波的多焦点图像栈融合算法实现 (支持 OpenCL GPU 加速与多核流式 CPU 回退)
    不依赖 opencv-contrib (ximgproc)，内置引导滤波实现。
    """
    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    check_cancel()
    scratch = capture_scratch()
    if thread_count is None:
        max_workers = min(12, max(1, os.cpu_count() or 4))
    else:
        try:
            max_workers = max(1, int(thread_count))
        except (TypeError, ValueError):
            max_workers = min(12, max(1, os.cpu_count() or 4))

    DEFAULT_R1 = 45
    DEFAULT_R2 = 7
    DEFAULT_EPS1 = 0.3
    DEFAULT_EPS2 = 10e-6
    DEFAULT_SIGMA_R = 5

    average_filter_size = kernel_size if kernel_size is not None and kernel_size > 0 else 31
    if average_filter_size % 2 == 0:
        average_filter_size += 1

    if isinstance(input_source, str):
        from fusion_methods.openfocus_ab import _collect_directory_sources

        stack_ori, paths = _collect_directory_sources(input_source, should_cancel)
        if not paths and not os.listdir(input_source):
            raise ValueError("Input folder is empty or no images were found")
    else:
        stack_ori = input_source

    if not stack_ori:
        raise ValueError("No image data was loaded")

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
        from fusion_methods._openfocus_gpu_gff import GpuGff

        try:
            gpu = GpuGff()
            result = gpu.fuse(
                stack_ori, img_resize, average_filter_size,
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
    from fusion_methods._openfocus_gpu import _iter_prefetched

    out_dtype = fuse_output_dtype(stack_ori)
    kernels = _load_kernels()

    @carry_warp_accelerator
    def _resolve_raw(index, cancel_fn=None):
        from core.source_frames import resolve_frame, without_source_cache_growth

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

    def prepare_frame(index, raw=None):
        if raw is None:
            raw = _resolve_raw(index)
        image = _ensure_color_image(np.ascontiguousarray(raw))
        del raw
        if image.dtype == np.uint16:
            image = image.astype(np.float32) * (255.0 / 65535.0)
        if img_resize:
            image = cv2.resize(image, img_resize)
        return image.astype(np.float32) / 255.0

    def guided_filter_fusion_stack():
        num_images = len(stack_ori)
        if num_images == 0:
            return None

        total_steps = num_images * 2
        if progress_callback is not None:
            progress_callback(0, total_steps)

        if img_resize is not None:
            rows, cols = int(img_resize[1]), int(img_resize[0])
            channels = 3
        else:
            first_shape = stack_ori[0].shape
            if len(first_shape) == 2 or (len(first_shape) == 3 and first_shape[2] in (3, 4)):
                rows, cols, channels = int(first_shape[0]), int(first_shape[1]), 3
            else:
                first = prepare_frame(0)
                rows, cols, channels = first.shape
                del first

        pixels = rows * cols
        # Use row-strip parallelism for large frames so peak RAM stays bounded
        # (~1-2 live frames) while saturating all CPU cores.
        row_parallel = kernels is not None and pixels >= _ROW_PARALLEL_PIXELS and max_workers > 1
        frame_workers = 1 if (row_parallel or pixels >= 4_000_000) else min(max_workers, 8)
        row_workers = max_workers if row_parallel else 1
        prefetch_slots = 0
        if row_parallel and num_images >= 2:
            from core import resource_policy as _rp
            from fusion_methods._openfocus_gpu import _prefetch_window
            _snap = _rp.probe_memory()
            _budget = _rp.ram_ceiling(_snap)
            _frame_est = max(1, pixels * 3 * 2)
            _staging_est = pixels * 24 + _frame_est
            prefetch_slots = _prefetch_window(
                _budget, _staging_est, _frame_est, num_images,
                max_slots=2, thread_count=max_workers,
            )

        def _check_shape(shape):
            if shape[:2] != (rows, cols):
                raise ValueError("Guided Filter requires frames with matching shapes")

        def _valid_raw_channels(raw):
            return raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] in (3, 4))

        def _compute_saliency_from_raw(index, raw, strip_exec):
            if kernels is not None and img_resize is None:
                if _valid_raw_channels(raw) and raw.dtype in (np.uint8, np.uint16, np.float32, np.float64):
                    _check_shape(raw.shape)
                    contig = np.ascontiguousarray(raw)
                    del raw
                    img_sum = np.empty((rows, cols), dtype=np.float32)
                    if contig.dtype == np.uint8:
                        _dispatch_strips(
                            strip_exec, row_workers, 0, rows,
                            lambda s, e, c=contig, o=img_sum: kernels.sum_u8_to_f32_rows(c, _U8_NORM_LUT, o, s, e),
                            check_cancel,
                        )
                    elif contig.dtype == np.uint16:
                        _dispatch_strips(
                            strip_exec, row_workers, 0, rows,
                            lambda s, e, c=contig, o=img_sum: kernels.sum_u16_to_f32_rows(c, o, s, e),
                            check_cancel,
                        )
                    else:
                        f32 = contig if contig.dtype == np.float32 else contig.astype(np.float32)
                        _dispatch_strips(
                            strip_exec, row_workers, 0, rows,
                            lambda s, e, c=f32, o=img_sum: kernels.sum_f32_to_f32_rows(c, o, s, e),
                            check_cancel,
                        )
                    del contig
                else:
                    img = prepare_frame(index, raw)
                    del raw
                    _check_shape(img.shape)
                    img_sum = np.sum(img, axis=2)
                    del img
            else:
                img = prepare_frame(index, raw)
                del raw
                _check_shape(img.shape)
                img_sum = np.sum(img, axis=2)
                del img

            lap = np.abs(cv2.Laplacian(img_sum, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REFLECT))
            del img_sum
            sal = cv2.GaussianBlur(
                lap, (0, 0), sigmaX=DEFAULT_SIGMA_R, sigmaY=DEFAULT_SIGMA_R,
                borderType=cv2.BORDER_REFLECT)
            del lap
            return sal

        def process_saliency(index):
            raw = _resolve_raw(index)
            return _compute_saliency_from_raw(index, raw, None)

        max_indices = create_frame((rows, cols), np.int32)
        max_indices[...] = 0
        best_saliency = None

        strip_pool = (concurrent.futures.ThreadPoolExecutor(max_workers=row_workers)
                      if row_parallel else None)
        try:
            if row_parallel:
                def _prefetch_raw(index, _src, cancelled):
                    return run_with_scratch(scratch, _resolve_raw, index, cancelled)

                pref = _iter_prefetched(
                    stack_ori, _prefetch_raw, enabled=prefetch_slots > 0,
                    should_cancel=should_cancel, max_in_flight=max(1, prefetch_slots),
                )
                try:
                    for k, raw in pref:
                        check_cancel()
                        sal = _compute_saliency_from_raw(k, raw, strip_pool)
                        del raw
                        check_cancel()
                        if best_saliency is None:
                            best_saliency = store_frame(sal)
                        else:
                            _dispatch_strips(
                                strip_pool, row_workers, 0, rows,
                                lambda s, e, sl=sal, bs=best_saliency, mi=max_indices, fk=k:
                                    kernels.argmax_update_rows(sl, bs, mi, fk, s, e),
                                check_cancel, min_rows=64,
                            )
                        del sal
                        if progress_callback is not None:
                            progress_callback(k + 1, total_steps)
                finally:
                    pref.close()
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=frame_workers) as executor:
                    for start in range(0, num_images, frame_workers):
                        check_cancel()
                        stop = min(num_images, start + frame_workers)
                        for k, sal in zip(range(start, stop), executor.map(
                                partial(run_with_scratch, scratch, process_saliency), range(start, stop))):
                            check_cancel()
                            if best_saliency is None:
                                best_saliency = store_frame(sal)
                            elif kernels is not None:
                                for row in range(0, rows, 128):
                                    check_cancel()
                                    kernels.argmax_update_rows(
                                        sal, best_saliency, max_indices, k, row, min(rows, row + 128))
                            else:
                                for row in range(0, rows, 128):
                                    check_cancel()
                                    block = slice(row, min(rows, row + 128))
                                    win = sal[block] > best_saliency[block]
                                    np.copyto(best_saliency[block], sal[block], where=win)
                                    np.copyto(max_indices[block], k, where=win)
                                del win
                            del sal
                            if progress_callback is not None:
                                progress_callback(k + 1, total_steps)
            del best_saliency

            fused_base_numerator = create_frame((rows, cols, channels), np.float32)
            fused_base_denominator = create_frame((rows, cols), np.float32)
            fused_detail_numerator = create_frame((rows, cols, channels), np.float32)
            fused_detail_denominator = create_frame((rows, cols), np.float32)
            for buffer in (fused_base_numerator, fused_base_denominator,
                           fused_detail_numerator, fused_detail_denominator):
                buffer[...] = 0

            check_cancel()
            if kernels is not None:
                if row_parallel:
                    step = max(64, (rows + row_workers - 1) // row_workers)
                    futs = [
                        strip_pool.submit(kernels.active_bboxes_rows, max_indices, num_images, y, min(rows, y + step))
                        for y in range(0, rows, step)
                    ]
                    wins = np.zeros(num_images, np.int64)
                    ymin = np.full(num_images, rows, np.int32)
                    ymax = np.full(num_images, -1, np.int32)
                    xmin = np.full(num_images, cols, np.int32)
                    xmax = np.full(num_images, -1, np.int32)
                    for fut in futs:
                        w_p, y0_p, y1_p, x0_p, x1_p = fut.result()
                        wins += w_p
                        np.minimum(ymin, y0_p, out=ymin)
                        np.maximum(ymax, y1_p, out=ymax)
                        np.minimum(xmin, x0_p, out=xmin)
                        np.maximum(xmax, x1_p, out=xmax)
                else:
                    wins, ymin, ymax, xmin, xmax = kernels.active_bboxes(max_indices, num_images)
            else:
                wins = np.bincount(max_indices.ravel(), minlength=num_images)

            def contributes(k):
                if wins[k]:
                    return True
                from core.source_frames import resolve_frame, without_source_cache_growth

                with without_source_cache_growth():
                    frame = resolve_frame(stack_ori[k], should_cancel=should_cancel)
                if frame.dtype in (np.uint8, np.uint16):
                    return False
                if frame.dtype.kind == 'f':
                    return not (-255.0 <= np.min(frame) and np.max(frame) <= 255.0)
                return True

            def _prep_bgr_roi(index, raw, roi_y0, roi_y1, roi_x0, roi_x1, strip_exec):
                rh = roi_y1 - roi_y0
                rw = roi_x1 - roi_x0
                if (kernels is not None and img_resize is None
                        and _valid_raw_channels(raw)
                        and raw.dtype in (np.uint8, np.uint16, np.float32, np.float64)):
                    _check_shape(raw.shape)
                    contig = np.ascontiguousarray(raw)
                    out_bgr = np.empty((rh, rw, 3), dtype=np.float32)
                    if contig.dtype == np.uint8:
                        _dispatch_strips(
                            strip_exec, row_workers, 0, rh,
                            lambda s, e, c=contig, o=out_bgr:
                                kernels.prep_u8_bgr_roi_rows(c, _U8_NORM_LUT, o, roi_y0, roi_x0, s, e),
                            check_cancel,
                        )
                    elif contig.dtype == np.uint16:
                        _dispatch_strips(
                            strip_exec, row_workers, 0, rh,
                            lambda s, e, c=contig, o=out_bgr:
                                kernels.prep_u16_bgr_roi_rows(c, o, roi_y0, roi_x0, s, e),
                            check_cancel,
                        )
                    else:
                        f32 = contig if contig.dtype == np.float32 else contig.astype(np.float32)
                        _dispatch_strips(
                            strip_exec, row_workers, 0, rh,
                            lambda s, e, c=f32, o=out_bgr:
                                kernels.prep_f32_bgr_roi_rows(c, o, roi_y0, roi_x0, s, e),
                            check_cancel,
                        )
                    del contig
                    return out_bgr
                full_img = prepare_frame(index, raw)
                return np.ascontiguousarray(full_img[roi_y0:roi_y1, roi_x0:roi_x1])

            def _process_full_opencv(k, raw=None):
                check_cancel()
                mask_k = (max_indices == k).astype(np.float32)
                img_k = prepare_frame(k, raw)
                base = cv2.blur(img_k, (average_filter_size, average_filter_size),
                                borderType=cv2.BORDER_REFLECT)
                detail = img_k - base
                I = cv2.cvtColor(img_k, cv2.COLOR_BGR2GRAY)
                p = mask_k
                Ip = I * p
                I2 = I * I

                def run_gf(r, eps):
                    ksize = (2 * r + 1, 2 * r + 1)
                    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
                    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
                    mean_Ip = cv2.boxFilter(Ip, cv2.CV_32F, ksize)
                    cov_Ip = mean_Ip - mean_I * mean_p
                    mean_II = cv2.boxFilter(I2, cv2.CV_32F, ksize)
                    var_I = mean_II - mean_I * mean_I
                    a = cov_Ip / (var_I + eps)
                    b = mean_p - a * mean_I
                    return cv2.boxFilter(a, cv2.CV_32F, ksize) * I + cv2.boxFilter(b, cv2.CV_32F, ksize)

                weight_base_k = run_gf(DEFAULT_R1, DEFAULT_EPS1)
                check_cancel()
                weight_detail_k = run_gf(DEFAULT_R2, DEFAULT_EPS2)
                base_num = base * weight_base_k[:, :, np.newaxis]
                detail_num = detail * weight_detail_k[:, :, np.newaxis]
                return ('full', base_num, weight_base_k, detail_num, weight_detail_k)

            def _process_and_accumulate_region(k, raw, strip_exec):
                check_cancel()
                y0_k, y1_k = int(ymin[k]), int(ymax[k])
                x0_k, x1_k = int(xmin[k]), int(xmax[k])

                ay0_b, ay1_b, by0, by1 = _gf_bounds(y0_k, y1_k, DEFAULT_R1, rows)
                ax0_b, ax1_b, bx0, bx1 = _gf_bounds(x0_k, x1_k, DEFAULT_R1, cols)
                ay0_d, ay1_d, dy0, dy1 = _gf_bounds(y0_k, y1_k, DEFAULT_R2, rows)
                ax0_d, ax1_d, dx0, dx1 = _gf_bounds(x0_k, x1_k, DEFAULT_R2, cols)

                base_y0 = min(by0, dy0)
                base_y1 = max(by1, dy1)
                base_x0 = min(bx0, dx0)
                base_x1 = max(bx1, dx1)
                rad_avg = average_filter_size // 2
                bw = base_x1 - base_x0
                wb_w = bx1 - bx0
                wd_w = dx1 - dx0
                aw_b = ax1_b - ax0_b
                aw_d = ax1_d - ax0_d

                g1_x0, g1_x1 = _reflect101_span(ax0_b, ax1_b, DEFAULT_R1, cols)
                b_x0, b_x1 = _reflect_span(base_x0, base_x1, rad_avg, cols)
                roi_x0 = min(g1_x0, b_x0, base_x0)
                roi_x1 = max(g1_x1, b_x1, base_x1)

                band_h = 512 if (base_y1 - base_y0) > 512 else (base_y1 - base_y0)
                for seg_y0 in range(base_y0, base_y1, band_h):
                    check_cancel()
                    seg_y1 = min(base_y1, seg_y0 + band_h)
                    seg_h = seg_y1 - seg_y0

                    # Exact Stage-1 & input row spans for base GF on [seg_y0:seg_y1].
                    sb_y0, sb_y1 = _reflect101_span(seg_y0, seg_y1, DEFAULT_R1, rows)
                    ab_y0 = max(ay0_b, sb_y0)
                    ab_y1 = min(ay1_b, sb_y1)
                    if ab_y0 < ab_y1:
                        g1_y0, g1_y1 = _reflect101_span(ab_y0, ab_y1, DEFAULT_R1, rows)
                    else:
                        g1_y0, g1_y1 = seg_y0, seg_y1

                    # Exact Stage-1 & input row spans for detail GF on [db_y0:db_y1].
                    db_y0 = max(seg_y0, dy0)
                    db_y1 = min(seg_y1, dy1)
                    has_detail = db_y0 < db_y1
                    if has_detail:
                        sd_y0, sd_y1 = _reflect101_span(db_y0, db_y1, DEFAULT_R2, rows)
                        dab_y0 = max(ay0_d, sd_y0)
                        dab_y1 = min(ay1_d, sd_y1)
                        if dab_y0 < dab_y1:
                            dg_y0, dg_y1 = _reflect101_span(dab_y0, dab_y1, DEFAULT_R2, rows)
                        else:
                            dg_y0, dg_y1 = db_y0, db_y1
                            has_detail = False
                    else:
                        dab_y0 = dab_y1 = dg_y0 = dg_y1 = seg_y0

                    bl_y0, bl_y1 = _reflect_span(seg_y0, seg_y1, rad_avg, rows)
                    roi_y0 = min(g1_y0, dg_y0, bl_y0, seg_y0)
                    roi_y1 = max(g1_y1, dg_y1, bl_y1, seg_y1)

                    bgr_roi = _prep_bgr_roi(k, raw, roi_y0, roi_y1, roi_x0, roi_x1, strip_exec)
                    gray_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)

                    def _run_gf_band(rad, eps, ay0, ay1, ax0, ax1, aw, qy0, qy1, qx0, qw):
                        ah = ay1 - ay0
                        qh = qy1 - qy0
                        a_buf = np.empty((ah, aw), np.float32)
                        b_buf = np.empty((ah, aw), np.float32)
                        _dispatch_strips(
                            strip_exec, row_workers, 0, ah,
                            lambda s, e: kernels.gf_stage1_roi_rows(
                                gray_roi, max_indices, k, rad, eps,
                                rows, cols, roi_y0, roi_x0,
                                ay0, ax0, aw, a_buf, b_buf, s, e,
                            ),
                            check_cancel,
                        )
                        q_out = np.empty((qh, qw), np.float32)
                        _dispatch_strips(
                            strip_exec, row_workers, 0, qh,
                            lambda s, e: kernels.gf_stage2_roi_rows(
                                gray_roi, a_buf, b_buf, rad,
                                rows, cols, roi_y0, roi_x0,
                                ay0, ay1, ax0, ax1,
                                qy0, qx0, qw, q_out, s, e,
                            ),
                            check_cancel,
                        )
                        return q_out

                    if ab_y0 < ab_y1:
                        wb = _run_gf_band(
                            DEFAULT_R1, DEFAULT_EPS1, ab_y0, ab_y1, ax0_b, ax1_b, aw_b,
                            seg_y0, seg_y1, bx0, wb_w,
                        )
                        wb_y0, wb_y1 = seg_y0, seg_y1
                    else:
                        wb = np.empty((0, wb_w), np.float32)
                        wb_y0 = wb_y1 = seg_y0
                    check_cancel()

                    if has_detail:
                        wd = _run_gf_band(
                            DEFAULT_R2, DEFAULT_EPS2, dab_y0, dab_y1, ax0_d, ax1_d, aw_d,
                            db_y0, db_y1, dx0, wd_w,
                        )
                        wd_y0, wd_y1 = db_y0, db_y1
                    else:
                        wd = np.empty((0, wd_w), np.float32)
                        wd_y0 = wd_y1 = seg_y0
                    del gray_roi

                    base_patch = np.empty((seg_h, bw, 3), np.float32)
                    _dispatch_strips(
                        strip_exec, row_workers, 0, seg_h,
                        lambda s, e: kernels.box3_exact_region_roi_rows(
                            bgr_roi, rows, cols, roi_y0, roi_x0,
                            rad_avg, seg_y0, base_x0, bw, base_patch, s, e,
                        ),
                        check_cancel,
                    )
                    _dispatch_strips(
                        strip_exec, row_workers, seg_y0, seg_y1,
                        lambda s, e: kernels.accumulate_frame_region_roi_rows(
                            bgr_roi, roi_y0, roi_x0, base_patch, seg_y0, base_x0,
                            wb, wb_y0, wb_y1, bx0, bx1, wd, wd_y0, wd_y1, dx0, dx1,
                            fused_base_numerator, fused_base_denominator,
                            fused_detail_numerator, fused_detail_denominator,
                            s, e,
                        ),
                        check_cancel, min_rows=64,
                    )
                    del bgr_roi, base_patch, wb, wd

            def _run_region_frame(k, raw, strip_exec):
                if kernels is None or wins[k] == 0:
                    return _process_full_opencv(k, raw)
                check_cancel()
                y0_k, y1_k = int(ymin[k]), int(ymax[k])
                x0_k, x1_k = int(xmin[k]), int(xmax[k])

                ay0_b, ay1_b, by0, by1 = _gf_bounds(y0_k, y1_k, DEFAULT_R1, rows)
                ax0_b, ax1_b, bx0, bx1 = _gf_bounds(x0_k, x1_k, DEFAULT_R1, cols)
                ay0_d, ay1_d, dy0, dy1 = _gf_bounds(y0_k, y1_k, DEFAULT_R2, rows)
                ax0_d, ax1_d, dx0, dx1 = _gf_bounds(x0_k, x1_k, DEFAULT_R2, cols)

                base_y0 = min(by0, dy0)
                base_y1 = max(by1, dy1)
                base_x0 = min(bx0, dx0)
                base_x1 = max(bx1, dx1)
                rad_avg = average_filter_size // 2

                g1_y0, g1_y1 = _reflect101_span(ay0_b, ay1_b, DEFAULT_R1, rows)
                g1_x0, g1_x1 = _reflect101_span(ax0_b, ax1_b, DEFAULT_R1, cols)
                b_y0, b_y1 = _reflect_span(base_y0, base_y1, rad_avg, rows)
                b_x0, b_x1 = _reflect_span(base_x0, base_x1, rad_avg, cols)
                roi_y0 = min(g1_y0, b_y0, base_y0)
                roi_y1 = max(g1_y1, b_y1, base_y1)
                roi_x0 = min(g1_x0, b_x0, base_x0)
                roi_x1 = max(g1_x1, b_x1, base_x1)

                bgr_roi = _prep_bgr_roi(k, raw, roi_y0, roi_y1, roi_x0, roi_x1, strip_exec)
                del raw
                gray_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)

                def _run_gf_roi(rad, eps, ay0, ay1, ax0, ax1, qy0, qy1, qx0, qx1):
                    ah, aw = ay1 - ay0, ax1 - ax0
                    qh, qw = qy1 - qy0, qx1 - qx0
                    a_buf = np.empty((ah, aw), np.float32)
                    b_buf = np.empty((ah, aw), np.float32)
                    kernels.gf_stage1_roi_rows(
                        gray_roi, max_indices, k, rad, eps,
                        rows, cols, roi_y0, roi_x0,
                        ay0, ax0, aw, a_buf, b_buf, 0, ah,
                    )
                    q_out = np.empty((qh, qw), np.float32)
                    kernels.gf_stage2_roi_rows(
                        gray_roi, a_buf, b_buf, rad,
                        rows, cols, roi_y0, roi_x0,
                        ay0, ay1, ax0, ax1,
                        qy0, qx0, qw, q_out, 0, qh,
                    )
                    return q_out

                wb = _run_gf_roi(DEFAULT_R1, DEFAULT_EPS1, ay0_b, ay1_b, ax0_b, ax1_b, by0, by1, bx0, bx1)
                check_cancel()
                wd = _run_gf_roi(DEFAULT_R2, DEFAULT_EPS2, ay0_d, ay1_d, ax0_d, ax1_d, dy0, dy1, dx0, dx1)
                del gray_roi

                bh, bw = base_y1 - base_y0, base_x1 - base_x0
                base_patch = np.empty((bh, bw, 3), np.float32)
                kernels.box3_exact_region_roi_rows(
                    bgr_roi, rows, cols, roi_y0, roi_x0,
                    rad_avg, base_y0, base_x0, bw, base_patch, 0, bh,
                )
                return ('region', bgr_roi, roi_y0, roi_x0, base_patch, base_y0, base_x0,
                        wb, by0, by1, bx0, bx1, wd, dy0, dy1, dx0, dx1)

            if row_parallel:
                active_k = [k for k in range(num_images) if contributes(k)]
                active_src = [stack_ori[k] for k in active_k]
                from core import resource_policy as _rp
                from fusion_methods._openfocus_gpu import _prefetch_window
                _snap2 = _rp.probe_memory()
                _budget2 = _rp.ram_ceiling(_snap2)
                _frame_est2 = max(1, pixels * 3 * 2)
                _band_est2 = min(rows, 692) * cols * 48 + _frame_est2
                act_slots = _prefetch_window(
                    _budget2, _band_est2, _frame_est2, len(active_k),
                    max_slots=1, thread_count=max_workers,
                ) if active_k else 0

                def _prefetch_active(pos, _src, cancelled):
                    return run_with_scratch(scratch, _resolve_raw, active_k[pos], cancelled)

                pref = _iter_prefetched(
                    active_src, _prefetch_active, enabled=act_slots > 0,
                    should_cancel=should_cancel, max_in_flight=max(1, act_slots),
                )
                next_p = 0
                try:
                    for pos, raw in pref:
                        k = active_k[pos]
                        while next_p < k:
                            if progress_callback is not None:
                                progress_callback(num_images + next_p + 1, total_steps)
                            next_p += 1
                        if kernels is not None and wins[k] > 0:
                            _process_and_accumulate_region(k, raw, strip_pool)
                        else:
                            _, bn, bd, dn, dd = _process_full_opencv(k, raw)
                            fused_base_numerator += bn
                            fused_base_denominator += bd
                            fused_detail_numerator += dn
                            fused_detail_denominator += dd
                        del raw
                        if progress_callback is not None:
                            progress_callback(num_images + k + 1, total_steps)
                        next_p = k + 1
                    while next_p < num_images:
                        if progress_callback is not None:
                            progress_callback(num_images + next_p + 1, total_steps)
                        next_p += 1
                finally:
                    pref.close()
            else:
                def process_weight_fusion(k):
                    raw = _resolve_raw(k)
                    return _run_region_frame(k, raw, None)

                with concurrent.futures.ThreadPoolExecutor(max_workers=frame_workers) as executor:
                    for start in range(0, num_images, frame_workers):
                        check_cancel()
                        stop = min(num_images, start + frame_workers)
                        batch = [k for k in range(start, stop) if contributes(k)]
                        for item in executor.map(
                                partial(run_with_scratch, scratch, process_weight_fusion), batch):
                            check_cancel()
                            if item[0] == 'region':
                                (_, bgr_roi, roi_y0, roi_x0, base_patch, base_y0, base_x0,
                                 wb, by0, by1, bx0, bx1, wd, dy0, dy1, dx0, dx1) = item
                                kernels.accumulate_frame_region_roi_rows(
                                    bgr_roi, roi_y0, roi_x0, base_patch, base_y0, base_x0,
                                    wb, by0, by1, bx0, bx1, wd, dy0, dy1, dx0, dx1,
                                    fused_base_numerator, fused_base_denominator,
                                    fused_detail_numerator, fused_detail_denominator,
                                    min(by0, dy0), max(by1, dy1),
                                )
                            else:
                                _, bn, bd, dn, dd = item
                                fused_base_numerator += bn
                                fused_base_denominator += bd
                                fused_detail_numerator += dn
                                fused_detail_denominator += dd
                            del item
                        if progress_callback is not None:
                            progress_callback(num_images + stop, total_steps)

            check_cancel()
            fused_img = create_frame((rows, cols, channels), out_dtype)
            if kernels is not None and row_parallel:
                _dispatch_strips(
                    strip_pool, row_workers, 0, rows,
                    lambda s, e: kernels.reconstruct_rows(
                        fused_base_numerator, fused_base_denominator,
                        fused_detail_numerator, fused_detail_denominator,
                        fused_img, s, e,
                    ),
                    check_cancel, min_rows=64,
                )
            else:
                for start in range(0, rows, 128):
                    check_cancel()
                    stop_row = min(rows, start + 128)
                    if kernels is not None:
                        kernels.reconstruct_rows(
                            fused_base_numerator, fused_base_denominator,
                            fused_detail_numerator, fused_detail_denominator,
                            fused_img, start, stop_row)
                    else:
                        block = slice(start, stop_row)
                        bd = fused_base_denominator[block]
                        dd = fused_detail_denominator[block]
                        bd[bd < 1e-6] = 1e-6
                        dd[dd < 1e-6] = 1e-6
                        base = fused_base_numerator[block] / bd[:, :, None]
                        detail = fused_detail_numerator[block] / dd[:, :, None]
                        fused_img[block] = np.clip((base + detail) * 255, 0, 255)
        finally:
            if strip_pool is not None:
                strip_pool.shutdown(wait=True, cancel_futures=True)

        check_cancel()
        if progress_callback is not None:
            progress_callback(total_steps, total_steps)
        if device_callback is not None:
            device_callback('CPU')
        return fused_img

    return guided_filter_fusion_stack()
