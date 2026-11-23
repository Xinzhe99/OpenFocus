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


def _load_kernels():
    """Load cached Numba kernels when the startup compiler budget allows."""
    from core.resource_policy import ResourceLimitError
    try:
        from fusion_methods import _gff_kernels
        return _gff_kernels
    except (ImportError, ResourceLimitError):
        return None


def gff_impl(input_source, img_resize, kernel_size=31, thread_count: int = None,
             should_cancel=None, progress_callback=None,
             use_gpu: bool = False, device_callback=None):
    """
    基于引导滤波的多焦点图像栈融合算法实现 (支持 OpenCL GPU 加速与 CPU 回退)
    不依赖 opencv-contrib (ximgproc)，内置引导滤波实现。
    """
    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    check_cancel()
    scratch = capture_scratch()
    try:
        max_workers = max(1, int(thread_count)) if thread_count is not None else 1
    except (TypeError, ValueError):
        max_workers = 1

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

    out_dtype = fuse_output_dtype(stack_ori)
    kernels = _load_kernels()

    @carry_warp_accelerator
    def _resolve_raw(index):
        from core.source_frames import resolve_frame, without_source_cache_growth

        check_cancel()
        with without_source_cache_growth():
            raw = resolve_frame(stack_ori[index], should_cancel=should_cancel)
        check_cancel()
        return raw

    def prepare_frame(index):
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

        def _check_shape(shape):
            if shape[:2] != (rows, cols):
                raise ValueError("Guided Filter requires frames with matching shapes")

        def _valid_raw_channels(raw):
            return raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] in (3, 4))

        def process_saliency(index):
            if kernels is not None and img_resize is None:
                raw = _resolve_raw(index)
                if _valid_raw_channels(raw) and raw.dtype in (np.uint8, np.uint16, np.float32, np.float64):
                    _check_shape(raw.shape)
                    contig = np.ascontiguousarray(raw)
                    del raw
                    img_sum = np.empty((rows, cols), dtype=np.float32)
                    if contig.dtype == np.uint8:
                        kernels.sum_u8_to_f32(contig, _U8_NORM_LUT, img_sum)
                    elif contig.dtype == np.uint16:
                        kernels.sum_u16_to_f32(contig, img_sum)
                    else:
                        f32 = contig if contig.dtype == np.float32 else contig.astype(np.float32)
                        kernels.sum_f32_to_f32(f32, img_sum)
                    del contig
                else:
                    del raw
                    img = prepare_frame(index)
                    _check_shape(img.shape)
                    img_sum = np.sum(img, axis=2)
                    del img
            else:
                img = prepare_frame(index)
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

        max_indices = create_frame((rows, cols), np.int32)
        max_indices[...] = 0
        best_saliency = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start in range(0, num_images, max_workers):
                check_cancel()
                stop = min(num_images, start + max_workers)
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

        def _prep_bgr_k(index):
            if kernels is not None and img_resize is None:
                raw = _resolve_raw(index)
                if raw.dtype == np.uint8 and _valid_raw_channels(raw):
                    _check_shape(raw.shape)
                    img_k = np.empty((rows, cols, 3), dtype=np.float32)
                    kernels.prep_u8_bgr(np.ascontiguousarray(raw), _U8_NORM_LUT, img_k)
                    del raw
                    return img_k
                del raw
            return prepare_frame(index)

        def _process_full_opencv(k):
            check_cancel()
            mask_k = (max_indices == k).astype(np.float32)
            img_k = prepare_frame(k)
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

        def process_weight_fusion(k):
            if kernels is None or wins[k] == 0:
                return _process_full_opencv(k)
            check_cancel()
            img_k = _prep_bgr_k(k)
            gray = cv2.cvtColor(img_k, cv2.COLOR_BGR2GRAY)
            y0_k, y1_k, x0_k, x1_k = int(ymin[k]), int(ymax[k]), int(xmin[k]), int(xmax[k])
            wb, by0, by1, bx0, bx1 = kernels.gf_weight_exact_region(
                gray, max_indices, k, DEFAULT_R1, DEFAULT_EPS1, y0_k, y1_k, x0_k, x1_k)
            check_cancel()
            wd, dy0, dy1, dx0, dx1 = kernels.gf_weight_exact_region(
                gray, max_indices, k, DEFAULT_R2, DEFAULT_EPS2, y0_k, y1_k, x0_k, x1_k)
            del gray
            base_y0 = min(by0, dy0)
            base_y1 = max(by1, dy1)
            base_x0 = min(bx0, dx0)
            base_x1 = max(bx1, dx1)
            base_patch = kernels.box3_exact_region(
                img_k, average_filter_size // 2, base_y0, base_y1, base_x0, base_x1)
            return ('region', img_k, base_patch, base_y0, base_x0,
                    wb, by0, by1, bx0, bx1, wd, dy0, dy1, dx0, dx1)

        # Empty masks contribute zero only for bounded finite images. Preserve
        # the original NaN/overflow behaviour for other floating-point data.
        check_cancel()
        if kernels is not None:
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start in range(0, num_images, max_workers):
                check_cancel()
                stop = min(num_images, start + max_workers)
                batch = [k for k in range(start, stop) if contributes(k)]
                for item in executor.map(
                        partial(run_with_scratch, scratch, process_weight_fusion), batch):
                    check_cancel()
                    if item[0] == 'region':
                        _, img_k, base_patch, base_y0, base_x0, wb, by0, by1, bx0, bx1, wd, dy0, dy1, dx0, dx1 = item
                        kernels.accumulate_frame_region(
                            img_k, base_patch, base_y0, base_x0,
                            wb, by0, by1, bx0, bx1,
                            wd, dy0, dy1, dx0, dx1,
                            fused_base_numerator, fused_base_denominator,
                            fused_detail_numerator, fused_detail_denominator,
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
        check_cancel()
        if progress_callback is not None:
            progress_callback(total_steps, total_steps)
        if device_callback is not None:
            device_callback('CPU')
        return fused_img

    return guided_filter_fusion_stack()
