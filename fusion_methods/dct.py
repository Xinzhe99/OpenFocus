# Reference:
# Haghighat M B A, Aghagolzadeh A, Seyedarabi H. Multi-focus image fusion for visual sensor networks in DCT domain[J]. Computers & Electrical Engineering, 2011, 37(5): 789-797.
import os
import time
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import median_filter

from utils.image_utils import fuse_output_dtype
from core.frame_storage import create_frame

ArraySource = Sequence[np.ndarray]

_ORIGINAL_CVT_COLOR = cv2.cvtColor


def _ensure_color_image(image: np.ndarray) -> np.ndarray:
    """确保输入图像为三通道BGR格式。"""
    if image is None:
        raise ValueError("Input image is None")
    from core.source_frames import resolve_frame
    image = resolve_frame(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _collect_images_from_folder(source_folder: str, check_cancel=None) -> Tuple[List[np.ndarray], List[str]]:
    from fusion_methods.openfocus_ab import _collect_directory_sources

    if not os.path.isdir(source_folder):
        if check_cancel is not None:
            check_cancel()
        return [], []
    return _collect_directory_sources(source_folder, check_cancel)


def _normalize_image_stack(images: Sequence[np.ndarray], check_cancel=None) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """统一图像尺寸并返回BGR列表。"""
    if not images:
        raise ValueError("图像栈为空")

    # 获取基准尺寸
    ref_img = images[0]
    target_h, target_w = ref_img.shape[:2]

    from core.source_frames import process_frame

    normalized = []
    for img in images:
        if check_cancel is not None:
            check_cancel()
        if getattr(img, "ndim", 0) == 3 and img.shape == (target_h, target_w, 3):
            normalized.append(img)
            continue

        def prepare(image):
            image = _ensure_color_image(image)
            if image.shape[:2] != (target_h, target_w):
                image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return image

        normalized.append(process_frame(img, prepare, shape=(target_h, target_w, 3),
                                        should_cancel=check_cancel))

    if target_h < 8 or target_w < 8:
        raise ValueError("图像尺寸过小")

    return normalized, (target_h, target_w)


def _load_kernels():
    """Load cached Numba kernels when the startup compiler budget allows."""
    from core.resource_policy import ResourceLimitError
    try:
        from fusion_methods import _dct_kernels
        return _dct_kernels
    except (ImportError, ResourceLimitError):
        return None


def _warmup_dct_kernels(kernels, normalized_images, block_size, out_dtype, use_fused_bgr):
    """Pre-compile Numba signatures on the main thread before spawning worker threads."""
    if kernels is None:
        return
    dummy_var = np.empty((1, 1), dtype=np.float32)
    dummy_max = np.full((1, 1), -1.0, dtype=np.float32)
    dummy_idx = np.zeros((1, 1), dtype=np.int32)
    dummy_dst = np.zeros((1, 1, 3), dtype=out_dtype)
    seen_dtypes = set()
    for img in normalized_images:
        dt = np.dtype(img.dtype)
        if dt in seen_dtypes:
            continue
        seen_dtypes.add(dt)
        if dt == np.uint8:
            if use_fused_bgr:
                kernels.block_variance_bgr_u8(
                    np.zeros((1, 1, 3), dtype=np.uint8), 1, 1, block_size, dummy_var
                )
            else:
                kernels.block_variance_u8(
                    np.zeros((1, 1), dtype=np.uint8), 1, 1, block_size, dummy_var
                )
            if out_dtype == np.uint8:
                kernels.copy_selected_blocks_u8(
                    np.zeros((1, 1, 3), dtype=np.uint8), dummy_idx, dummy_dst, 1, 1, block_size, 0
                )
            else:
                kernels.copy_selected_blocks_f32(
                    np.zeros((1, 1, 3), dtype=np.uint8), dummy_idx, dummy_dst, 1, 1, block_size, 0
                )
        elif dt == np.uint16:
            kernels.block_variance_f32(
                np.zeros((1, 1), dtype=np.float32), 1, 1, block_size, dummy_var
            )
            kernels.copy_selected_blocks_u16(
                np.zeros((1, 1, 3), dtype=np.uint16), dummy_idx, dummy_dst, 1, 1, block_size, 0
            )
        else:
            kernels.block_variance_f32(
                np.zeros((1, 1), dtype=np.float32), 1, 1, block_size, dummy_var
            )
            kernels.copy_selected_blocks_f32(
                np.zeros((1, 1, 3), dtype=dt), dummy_idx, dummy_dst, 1, 1, block_size, 0
            )
    kernels.argmax_update_blocks(dummy_var, dummy_max, dummy_idx, 0)


def dct_focus_stack_fusion(
    source: Union[str, ArraySource],
    output_path: str = None,
    block_size: int = 8,
    kernel_size: int = 7,
    thread_count: int = None,
    should_cancel=None,
    progress_callback=None,
    use_gpu: bool = False,
    device_callback=None,
) -> np.ndarray:
    """
    高度优化的 DCT/方差 图像融合算法 (支持 Numba CPU 并行与 OpenCL GPU 加速/CPU 回退)。

    利用 Parseval 定理，DCT 域的高频能量(方差)等价于空间域的像素方差。
    """
    from collections import deque
    import concurrent.futures
    from core.frame_storage import capture_scratch, run_with_scratch

    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    check_cancel()
    # --- 1. 参数校验与准备 ---
    if kernel_size % 2 == 0:
        kernel_size += 1
    block_size = max(2, int(block_size))

    if isinstance(source, str):
        images, _ = _collect_images_from_folder(source, check_cancel=check_cancel)
    elif isinstance(source, (list, tuple)):
        images = [img for img in source if img is not None]
    else:
        raise TypeError("Source must be folder path or image list.")

    if len(images) < 2:
        raise ValueError("需要至少 2 张图像进行融合。")

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
        from fusion_methods._openfocus_gpu_dct import GpuDct

        try:
            gpu = GpuDct()
            result = gpu.fuse(
                images, block_size, kernel_size,
                should_cancel=should_cancel,
                progress_callback=progress_callback,
            )
            if device_callback is not None:
                device_callback(gpu.device_name)
            if output_path:
                from utils.image_utils import imwrite_auto
                imwrite_auto(output_path, result)
            return result
        except GpuError as exc:
            log_fallback(exc)
        finally:
            SOURCE_CACHE.clear()

    # 归一化并获取尺寸
    normalized_images, (orig_h, orig_w) = _normalize_image_stack(images, check_cancel=check_cancel)

    pad_h = ((orig_h + block_size - 1) // block_size) * block_size
    pad_w = ((orig_w + block_size - 1) // block_size) * block_size
    pad_rows, pad_cols = pad_h - orig_h, pad_w - orig_w
    map_h = pad_h // block_size
    map_w = pad_w // block_size

    if map_h == 0 or map_w == 0:
        raise ValueError("Block size is too large for image size.")

    kernels = _load_kernels()
    use_fused_bgr = kernels is not None and cv2.cvtColor is _ORIGINAL_CVT_COLOR
    scratch = capture_scratch()
    count = len(normalized_images)
    total_steps = count * 2
    if progress_callback is not None:
        progress_callback(0, total_steps)

    if thread_count is None:
        max_workers = min(8, max(1, os.cpu_count() or 1), count)
    else:
        try:
            max_workers = max(1, min(int(thread_count), count))
        except (TypeError, ValueError):
            max_workers = 1

    # On large lazy stacks (>= 1 MP), MultiFocusFusion's generic estimator may
    # cap thread_count=1 even though streaming DCT holds only ~135 MB per
    # active frame. Scale workers safely within current RAM ceiling headroom.
    has_lazy = any(not isinstance(img, np.ndarray) for img in normalized_images)
    if max_workers == 1 and has_lazy and orig_h * orig_w >= 1024 * 1024 and count >= 2:
        from core import resource_policy
        from fusion_methods._openfocus_execution import _current

        job = _current.get()
        if job is None or getattr(job, 'thread_count', None) != 1:
            snap = resource_policy.probe_memory()
            ceiling = resource_policy.ram_ceiling(snap) if snap.ram_known else 0
            per_frame = max(1, int(normalized_images[0].nbytes) * 2)
            headroom = max(0, ceiling - 768 * 1024 * 1024)
            cpu_cap = min(6, max(1, (os.cpu_count() or 2) // 2))
            admitted = min(cpu_cap, count, int(headroom // per_frame))
            if admitted > 1:
                max_workers = admitted

    out_dtype = fuse_output_dtype(normalized_images)
    if kernels is not None and max_workers > 1:
        _warmup_dct_kernels(kernels, normalized_images, block_size, out_dtype, use_fused_bgr)

    # --- 2. 快速计算方差图 ---
    max_variance_map = np.full((map_h, map_w), -1.0, dtype=np.float32)
    best_index_map = np.zeros((map_h, map_w), dtype=np.int32)

    from core.source_frames import resolve_frame, without_source_cache_growth

    def frame_variance(index):
        check_cancel()
        with without_source_cache_growth():
            image = resolve_frame(normalized_images[index], should_cancel=should_cancel)
        check_cancel()

        if use_fused_bgr and image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] >= 3:
            var_map = np.empty((map_h, map_w), dtype=np.float32)
            kernels.block_variance_bgr_u8(image, orig_h, orig_w, block_size, var_map)
            del image
            return var_map

        if image.dtype == np.uint16:
            image = image.astype(np.float32) * (255.0 / 65535.0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        del image

        if kernels is not None:
            var_map = np.empty((map_h, map_w), dtype=np.float32)
            if gray.dtype == np.uint8:
                kernels.block_variance_u8(gray, orig_h, orig_w, block_size, var_map)
            else:
                gray_f32 = gray if gray.dtype == np.float32 else gray.astype(np.float32)
                kernels.block_variance_f32(gray_f32, orig_h, orig_w, block_size, var_map)
            del gray
            return var_map

        if pad_rows or pad_cols:
            gray = cv2.copyMakeBorder(gray, 0, pad_rows, 0, pad_cols, cv2.BORDER_REPLICATE)
        gray_f32 = gray.astype(np.float32)
        del gray
        mean_val = cv2.resize(gray_f32, (map_w, map_h), interpolation=cv2.INTER_AREA)
        np.square(mean_val, out=mean_val)
        np.square(gray_f32, out=gray_f32)
        mean_sq = cv2.resize(gray_f32, (map_w, map_h), interpolation=cv2.INTER_AREA)
        del gray_f32
        mean_sq -= mean_val
        del mean_val
        return mean_sq

    def reduce_variance(idx, var_map):
        if kernels is not None:
            kernels.argmax_update_blocks(var_map, max_variance_map, best_index_map, idx)
        else:
            mask = var_map > max_variance_map
            max_variance_map[mask] = var_map[mask]
            best_index_map[mask] = idx

    if max_workers == 1:
        for idx in range(count):
            var_map = frame_variance(idx)
            check_cancel()
            reduce_variance(idx, var_map)
            del var_map
            if progress_callback is not None:
                progress_callback(idx + 1, total_steps)
    else:
        from core.registration import carry_warp_accelerator

        prepare_variance = carry_warp_accelerator(frame_variance)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            window = deque()
            next_submit = 0
            try:
                while next_submit < count or window:
                    check_cancel()
                    while next_submit < count and len(window) < max_workers:
                        window.append((
                            next_submit,
                            pool.submit(run_with_scratch, scratch, prepare_variance, next_submit),
                        ))
                        next_submit += 1
                    for _, pending in window:
                        if pending.done() and pending.exception() is not None:
                            pending.result()
                    idx, fut = window[0]
                    if not fut.done():
                        concurrent.futures.wait((fut,), timeout=0.05)
                        continue
                    window.popleft()
                    var_map = fut.result()
                    del fut
                    check_cancel()
                    reduce_variance(idx, var_map)
                    del var_map
                    if progress_callback is not None:
                        progress_callback(idx + 1, total_steps)
            finally:
                for _, fut in window:
                    fut.cancel()
                window.clear()

    del max_variance_map

    # --- 3. 一致性验证 (中值滤波) ---
    check_cancel()
    if kernel_size <= 1:
        final_index_map = best_index_map
    elif kernels is not None:
        tmp_map = np.empty_like(best_index_map)
        kernels.median_filter_i32(best_index_map, tmp_map, kernel_size)
        check_cancel()
        kernels.median_filter_i32(tmp_map, best_index_map, kernel_size)
        del tmp_map
        final_index_map = best_index_map
    else:
        filtered_map = median_filter(best_index_map, size=kernel_size, mode="nearest")
        del best_index_map
        check_cancel()
        filtered_map = median_filter(filtered_map, size=kernel_size, mode="nearest")
        final_index_map = filtered_map.astype(np.int32, copy=False)
        del filtered_map

    # --- 4. 快速重建 ---
    fused_image = create_frame((orig_h, orig_w, 3), out_dtype)
    fused_image[...] = 0

    unique_indices = set(int(v) for v in np.unique(final_index_map))

    def _copy_layer(idx, source_layer):
        if kernels is not None and source_layer.ndim == 3 and source_layer.shape[2] >= 3:
            if out_dtype == np.uint8 and source_layer.dtype == np.uint8:
                kernels.copy_selected_blocks_u8(
                    source_layer, final_index_map, fused_image, orig_h, orig_w, block_size, idx)
            elif source_layer.dtype == np.uint16:
                kernels.copy_selected_blocks_u16(
                    source_layer, final_index_map, fused_image, orig_h, orig_w, block_size, idx)
            else:
                kernels.copy_selected_blocks_f32(
                    source_layer, final_index_map, fused_image, orig_h, orig_w, block_size, idx)
            return
        if source_layer.dtype == np.uint16:
            source_layer = source_layer.astype(np.float32) * (255.0 / 65535.0)
        block_rows = max(1, 128 // block_size)
        for start in range(0, map_h, block_rows):
            check_cancel()
            stop = min(map_h, start + block_rows)
            y0 = start * block_size
            y1 = min(orig_h, stop * block_size)
            rows = slice(y0, y1)
            mask_full = np.repeat(
                np.repeat(final_index_map[start:stop] == idx, block_size, axis=0),
                block_size, axis=1,
            )[:y1 - y0, :orig_w]
            np.copyto(fused_image[rows], source_layer[rows],
                      where=mask_full[:, :, None], casting='unsafe')

    active_indices = [idx for idx in range(count) if idx in unique_indices]

    if max_workers == 1 or len(active_indices) <= 1:
        for idx in range(count):
            if idx not in unique_indices:
                if progress_callback is not None:
                    progress_callback(count + idx + 1, total_steps)
                continue
            check_cancel()
            with without_source_cache_growth():
                source_layer = resolve_frame(normalized_images[idx], should_cancel=should_cancel)
            check_cancel()
            _copy_layer(idx, source_layer)
            del source_layer
            if progress_callback is not None:
                progress_callback(count + idx + 1, total_steps)
    else:
        from core.registration import carry_warp_accelerator

        # Each frame writes strictly disjoint block coordinates in fused_image,
        # so Numba block copies can execute concurrently with frame decoding.
        def reconstruct_worker(idx):
            check_cancel()
            with without_source_cache_growth():
                source_layer = resolve_frame(normalized_images[idx], should_cancel=should_cancel)
            check_cancel()
            if kernels is not None and source_layer.ndim == 3 and source_layer.shape[2] >= 3:
                _copy_layer(idx, source_layer)
                del source_layer
                return None
            return source_layer

        run_reconstruct = carry_warp_accelerator(reconstruct_worker)
        reconstruct_workers = min(max_workers, len(active_indices))
        with concurrent.futures.ThreadPoolExecutor(max_workers=reconstruct_workers) as pool:
            window = deque()
            next_pos = 0
            next_progress = 0
            try:
                while next_pos < len(active_indices) or window:
                    check_cancel()
                    while next_pos < len(active_indices) and len(window) < reconstruct_workers:
                        f_idx = active_indices[next_pos]
                        window.append((
                            f_idx,
                            pool.submit(run_with_scratch, scratch, run_reconstruct, f_idx),
                        ))
                        next_pos += 1
                    for _, pending in window:
                        if pending.done() and pending.exception() is not None:
                            pending.result()
                    idx, fut = window[0]
                    if not fut.done():
                        concurrent.futures.wait((fut,), timeout=0.05)
                        continue
                    window.popleft()
                    source_layer = fut.result()
                    del fut
                    check_cancel()
                    if source_layer is not None:
                        _copy_layer(idx, source_layer)
                        del source_layer
                    if progress_callback is not None:
                        while next_progress <= idx:
                            progress_callback(count + next_progress + 1, total_steps)
                            next_progress += 1
                if progress_callback is not None:
                    while next_progress < count:
                        progress_callback(count + next_progress + 1, total_steps)
                        next_progress += 1
            finally:
                for _, fut in window:
                    fut.cancel()
                window.clear()

    check_cancel()
    if device_callback is not None:
        device_callback('CPU')

    if output_path:
        from utils.image_utils import imwrite_auto
        imwrite_auto(output_path, fused_image)

    return fused_image


# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    t_start = time.time()

    # 修改这里的路径为你实际的图片文件夹
    TARGET_DIR = os.path.join("images", "stack")
    OUTPUT_FILE = os.path.join(TARGET_DIR, "Fused_Result_Optimized.tif")

    BLOCK_SIZE = 8
    KERNEL_SIZE = 7

    if os.path.exists(TARGET_DIR):
        try:
            print(f"开始处理: {TARGET_DIR}")
            result = dct_focus_stack_fusion(
                source=TARGET_DIR,  # 修正参数名
                output_path=OUTPUT_FILE,
                block_size=BLOCK_SIZE,
                kernel_size=KERNEL_SIZE
            )

            elapsed = time.time() - t_start
            print(f"处理完成，耗时: {elapsed:.4f} 秒")

            if result is not None:
                # 显示结果 (限制最大显示尺寸)
                h, w = result.shape[:2]
                max_dim = 800
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    show_w, show_h = int(w * scale), int(h * scale)
                    show_img = cv2.resize(result, (show_w, show_h))
                else:
                    show_img = result

                cv2.imshow("Optimized Fusion Result", show_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("文件夹路径不存在，请检查配置。")
