"""
图像序列配准统一接口

提供三种配准方法的统一调用接口:
1. Homography (单应性对齐)
2. ECC (ECC对齐)
3. Both (组合配准: Homography + ECC)

所有算法实现都包含在此脚本中，无需外部依赖

# 只进行单应性对齐
python Registration.py --mode homography

# 只进行ECC配准
python Registration.py --mode ecc

# 组合配准（homography + ecc）
python Registration.py --mode both

"""


import argparse
import concurrent.futures
from collections import deque
from contextlib import contextmanager
import time
import cv2
import numpy as np
import os
import glob
import sys
from typing import Union, List, Optional

from utils.sorting import natural_sort_key as _image_sort_key
from core.image_loader import apply_exif_orientation
from core.frame_storage import capture_scratch, create_frame, run_with_scratch, store_frame
from utils.logging_setup import log_action, log_operation



# ========== 基于变换矩阵的精确裁切函数 ==========

from contextvars import ContextVar

# Grouped fusion records crop offsets in the reference image's pixel grid.
grouped_crop_offsets = ContextVar('grouped_crop_offsets', default=None)

# Optional per-frame diagnostics, collected on the calling thread.
alignment_report = ContextVar('alignment_report', default=None)


def _report_alignment(frame_index, method, status, **details):
    report = alignment_report.get()
    if report is not None:
        details = {key: value.tolist() if isinstance(value, np.ndarray) else value
                   for key, value in details.items()}
        report.append({'frame_index': frame_index, 'method': method,
                       'status': status, **details})


def _report_reference(method, index=0):
    _report_alignment(index, method, 'reference', transform=np.eye(3).tolist())


def _record_grouped_crop(top, bottom, left, right):
    offsets = grouped_crop_offsets.get()
    if offsets is not None:
        if top >= bottom or left >= right:
            raise ValueError('Registration has no common valid region; disable grouping or use better aligned inputs')
        offsets.append((left, top, right, bottom))


def _compute_valid_region_from_transforms(H_matrices, img_shape, margin=2):
    """
    通过变换矩阵计算所有图像的公共有效区域
    
    Args:
        H_matrices: 变换矩阵列表 (3x3)
        img_shape: 图像尺寸 (h, w)
        margin: 安全边距，用于处理透视变形导致的边缘不精确问题
    
    Returns:
        (top, bottom, left, right): 公共有效区域的边界
    """
    if grouped_crop_offsets.get() is not None:
        # Lanczos4 samples a radius of four source pixels. Transform an inset
        # rectangle, then retain an axis-aligned rectangle inside every warp.
        h, w = img_shape
        if min(h, w) <= 8:
            raise ValueError('Grouped registration requires images larger than 8 pixels in each dimension')
        corners = np.array([[4, w - 4, w - 4, 4],
                            [4, 4, h - 4, h - 4], [1, 1, 1, 1]], dtype=np.float64)
        left, top, right, bottom = 0.0, 0.0, float(w), float(h)
        for matrix in H_matrices:
            transformed = np.asarray(matrix, dtype=np.float64) @ corners
            denominator = transformed[2]
            if (not np.all(np.isfinite(transformed)) or np.any(np.abs(denominator) < 1e-10)
                    or not (np.all(denominator > 0) or np.all(denominator < 0))):
                raise ValueError('Grouped registration cannot safely crop this projective transform; disable grouping')
            x, y = transformed[:2] / denominator
            if not (x[0] < x[1] and x[3] < x[2] and y[0] < y[3] and y[1] < y[2]):
                raise ValueError('Grouped registration cannot safely crop this rotated or projective transform; disable grouping')
            left = max(left, x[0], x[3])
            right = min(right, x[1], x[2])
            top = max(top, y[0], y[1])
            bottom = min(bottom, y[2], y[3])
        return int(np.ceil(top)), int(np.floor(bottom)), int(np.ceil(left)), int(np.floor(right))

    h, w = img_shape
    
    # 图像四个角点
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ], dtype=np.float32).T  # 3x4 矩阵
    
    # 初始化有效区域为整个图像
    top, bottom, left, right = 0.0, float(h), 0.0, float(w)
    
    for H in H_matrices:
        if H is None:
            continue
        
        # 变换角点
        transformed = H @ corners  # 3x4
        # 齐次坐标转换
        w_coords = transformed[2:3, :]
        # 防止除以零
        w_coords = np.where(np.abs(w_coords) < 1e-10, 1e-10, w_coords)
        transformed = transformed[:2, :] / w_coords  # 2x4
        
        x_coords = transformed[0, :]
        y_coords = transformed[1, :]
        
        # 对于透视变换，我们需要考虑四边形的每条边
        # 上边：连接左上和右上角，找最大y值（要裁切的区域）
        # 下边：连接左下和右下角，找最小y值
        # 左边：连接左上和左下角，找最大x值
        # 右边：连接右上和右下角，找最小x值
        
        # 角点索引: 0=左上, 1=右上, 2=右下, 3=左下
        # 上边的有效y：max(y_左上, y_右上)
        valid_top_edge = max(y_coords[0], y_coords[1])
        # 下边的有效y：min(y_左下, y_右下)
        valid_bottom_edge = min(y_coords[2], y_coords[3])
        # 左边的有效x：max(x_左上, x_左下)
        valid_left_edge = max(x_coords[0], x_coords[3])
        # 右边的有效x：min(x_右上, x_右下)
        valid_right_edge = min(x_coords[1], x_coords[2])
        
        # 与画布边界取交集
        valid_left_edge = max(0, valid_left_edge)
        valid_right_edge = min(w, valid_right_edge)
        valid_top_edge = max(0, valid_top_edge)
        valid_bottom_edge = min(h, valid_bottom_edge)
        
        # 更新公共有效区域（所有图像的交集）
        left = max(left, valid_left_edge)
        right = min(right, valid_right_edge)
        top = max(top, valid_top_edge)
        bottom = min(bottom, valid_bottom_edge)
    
    # 添加安全边距
    top += margin
    bottom -= margin
    left += margin
    right -= margin
    
    # 转换为整数，向内取整以确保安全
    top = int(np.ceil(top))
    bottom = int(np.floor(bottom))
    left = int(np.ceil(left))
    right = int(np.floor(right))
    
    return top, bottom, left, right



# ========== 单应性对齐算法实现（非线性） ==========


def _align_homography_impl(input_source, output_path=None, img_filenames=None, downscale_width=1600, thread_count: int = 4, should_cancel=None, failure_policy='continue'):
    """
    商业级图像对齐算法优化版
    特性：
    1. 传递对齐 (Sequential Alignment)：解决大景深下的特征丢失问题
    2. 金字塔加速 (Downscale Processing)：大幅提升特征检测速度
    3. 矩阵累积 (Matrix Chaining)：减少累积误差
    4. Lanczos插值：保证画质清晰度
    5. 并行计算优化 (Parallel Processing)：利用多核加速特征提取和图像变换
    """
    import concurrent.futures

    # --- 1. 数据加载与预处理 ---
    if img_filenames is None and isinstance(input_source, str):
        # 支持常见格式，过滤非图片
        valid_exts = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        img_paths = sorted(
            (os.path.join(input_source, f) for f in os.listdir(input_source)
             if os.path.splitext(f)[1].lower() in valid_exts),
            key=_image_sort_key
        )
        # 注意：这里为了内存考虑，商业软件通常不会一次性读入所有大图
        # 但为了保持接口一致，这里先全部读入。更好的做法是建立生成器。
        images = [cv2.imread(path) for path in img_paths]
        img_filenames = [os.path.basename(path) for path in img_paths]
    else:
        images = input_source

    num_images = len(images)
    if num_images:
        _report_reference('homography')
    if num_images < 2:
        return images

    # --- 初始化 ---
    h_orig, w_orig = images[0].shape[:2]
    # 如果单张图像过大（任一边 >= 2048），强制将用于下采样的目标宽度设为 1024
    max_dim = max(h_orig, w_orig)
    if max_dim >= 2048:
        try:
            # 记录之前的值以便调试
            prev_down = downscale_width
        except NameError:
            prev_down = None
        downscale_width = 1024
        print(f"[Registration] Large image detected ({h_orig}x{w_orig}), setting downscale_width {prev_down} -> {downscale_width}")
    
    # 全局累积矩阵 (用于将当前帧直接映射回第0帧)
    H_global = np.eye(3, dtype=np.float32)
    
    # 收集所有变换矩阵用于精确裁切
    H_matrices = [np.eye(3, dtype=np.float32)]  # 第一张是基准，单位矩阵

    print(f"Aligning {num_images} images using Sequential Homography (Parallel Optimized)...")

    # --- 2. 并行特征提取 ---
    print("  - Step 1/3: Extracting features concurrently...")

    def get_features_task(img):
        # 在线程中创建独立的检测器，确保线程安全
        local_detector = cv2.SIFT_create()
        h, w = img.shape[:2]
        scale = downscale_width / float(w) if w > downscale_width else 1.0
        if scale < 1.0:
            # 使用 INTER_LINEAR 速度更快，对于特征检测通常足够
            img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            img_small = img
        if img_small.dtype != np.uint8:
            # SIFT expects 8-bit input. Scale by dynamic range: uint16 by
            # 65535, float inputs (already 0-255 from normalize_fuse_input)
            # by 1 — the old dtype-based branch multiplied floats by 255
            # again and the uint8 cast wrapped them mod 256.
            if img_small.dtype == np.uint16:
                divisor = 65535.0
            elif np.issubdtype(img_small.dtype, np.floating):
                maximum = float(np.max(img_small))
                # Keep legacy 0-1 float inputs scaled to the SIFT byte domain.
                # Floats already in 0-255 need no additional gain.
                divisor = (maximum if maximum > 255.0 else
                           1.0 if maximum <= 1.0 else 255.0)
            else:
                divisor = 1.0
            img_small = (img_small.astype(np.float32) / divisor * 255.0).round().clip(0, 255).astype(np.uint8)
        kps, des = local_detector.detectAndCompute(img_small, None)
        return kps, des, scale

    # 使用线程池并行提取特征
    # OpenCV 的大部分操作释放 GIL，因此多线程可以有效加速
    try:
        max_workers = max(1, int(thread_count))
    except Exception:
        max_workers = min(8, os.cpu_count() or 1)

    # Matching retains its last accepted reference, including failed frames.
    bf = cv2.BFMatcher(cv2.NORM_L2)
    with _bounded_registration_map(get_features_task, images, max_workers,
                                   should_cancel) as features:
        last_kps, last_des, last_scale = next(features)
        last_index = 0
        trajectory_failed = False
        for idx, (curr_kps, curr_des, curr_scale) in enumerate(features, 1):
            _check_cancel(should_cancel)
            if curr_des is None or len(curr_kps) < 4 or last_des is None:
                print(f"Warning: Frame {idx} features insufficient. Keeping original position.")
                if failure_policy == 'fail':
                    _report_alignment(idx, 'homography', 'failed', reason='insufficient features', reference_index=last_index)
                    raise ValueError(f'Homography alignment failed at frame {idx + 1}: insufficient features')
                _report_alignment(idx, 'homography', 'fallback', reason='insufficient features',
                                  reference_index=last_index, transform=H_global)
                H_matrices.append(H_global.copy())
                continue

            matches = bf.knnMatch(curr_des, last_des, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.70 * n.distance:
                        good_matches.append(m)
            if len(good_matches) < 6:
                print(f"Warning: Frame {idx} poor matches ({len(good_matches)}). Keeping previous trajectory.")
                if failure_policy == 'fail':
                    _report_alignment(idx, 'homography', 'failed', reason='insufficient matches', reference_index=last_index, match_count=len(good_matches))
                    raise ValueError(f'Homography alignment failed at frame {idx + 1}: insufficient matches')
                _report_alignment(idx, 'homography', 'fallback', reason='insufficient matches',
                                  reference_index=last_index, transform=H_global,
                                  match_count=len(good_matches))
                H_matrices.append(H_global.copy())
                continue

            pts_curr = np.float32([curr_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / curr_scale
            pts_last = np.float32([last_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) / last_scale
            H_local, mask = cv2.findHomography(pts_curr, pts_last, cv2.RANSAC, 5.0)
            _check_cancel(should_cancel)
            estimated = H_local is not None
            if H_local is None:
                print(f"Frame {idx} alignment failed.")
                if failure_policy == 'fail':
                    _report_alignment(idx, 'homography', 'failed', reason='homography estimation failed', reference_index=last_index, match_count=len(good_matches))
                    raise ValueError(f'Homography alignment failed at frame {idx + 1}: homography estimation failed')
                _report_alignment(idx, 'homography', 'fallback', reason='homography estimation failed',
                                  reference_index=last_index, transform=H_global,
                                  match_count=len(good_matches))
                trajectory_failed = True
                H_local = np.eye(3)
            H_global = np.matmul(H_global, H_local)
            H_matrices.append(H_global.copy())
            if estimated:
                details = {'match_count': len(good_matches)}
                if mask is not None:
                    details['inlier_count'] = int(np.count_nonzero(mask))
                if trajectory_failed:
                    details['reason'] = 'reference alignment fell back'
                _report_alignment(idx, 'homography', 'fallback' if trajectory_failed else 'aligned',
                                  reference_index=last_index, transform=H_global, **details)
            last_kps, last_des, last_scale = curr_kps, curr_des, curr_scale
            last_index = idx
    del last_kps, last_des, curr_kps, curr_des

    # --- 4. 并行应用变换与裁切 ---
    print("  - Step 3/3: Warping images concurrently...")
    
    # 预先计算裁切区域，直接变换到目标区域，避免先变换后裁切的浪费
    top, bottom, left, right = _compute_valid_region_from_transforms(H_matrices, (h_orig, w_orig))
    _record_grouped_crop(top, bottom, left, right)
    
    do_crop = True
    if top >= bottom or left >= right:
        print("Warning: Invalid crop region, skipping crop.")
        do_crop = False
        target_w, target_h = w_orig, h_orig
        offset_x, offset_y = 0, 0
    else:
        target_w = right - left
        target_h = bottom - top
        offset_x = -left
        offset_y = -top
        print(f"    Optimized: Warping directly to cropped region ({target_w}x{target_h})...")

    # 构造裁切平移矩阵
    T_crop = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    def warp_task(args):
        img, H = args
        # 合并裁切变换
        if do_crop:
            H_final = T_crop @ H
        else:
            H_final = H

        return cv2.warpPerspective(img, H_final, (target_w, target_h),
                                 flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

    with _bounded_registration_map(warp_task, zip(images, H_matrices),
                                   max_workers, should_cancel) as results:
        aligned_images = [store_frame(image) for image in results]
    _save_registration(aligned_images, output_path, img_filenames, should_cancel)

    return aligned_images


class RegistrationCancelled(Exception):
    """Raised when a render cancellation flag fires mid-registration."""

def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise RegistrationCancelled()


@contextmanager
def _bounded_registration_map(function, values, workers, should_cancel=None):
    """Ordered results with at most one task per worker retained or running.

    Leaving the scope cancels queued work and waits for native calls before
    the enclosing heavy-job lease can be released.
    """
    _check_cancel(should_cancel)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    pending = deque()
    source = iter(values)
    scratch = capture_scratch()

    def submit_one():
        _check_cancel(should_cancel)
        try:
            value = next(source)
        except StopIteration:
            return
        pending.append(pool.submit(run_with_scratch, scratch, run, value))

    def run(value):
        _check_cancel(should_cancel)
        result = function(value)
        _check_cancel(should_cancel)
        return result

    def consume():
        for _ in range(workers):
            submit_one()
        while pending:
            _check_cancel(should_cancel)
            # A later failure must stop replenishment even when an earlier
            # native call is still running.
            for future in pending:
                if future.done() and future.exception() is not None:
                    future.result()
            future = pending[0]
            done, _ = concurrent.futures.wait((future,), timeout=0.05)
            if not done:
                continue
            result = future.result()
            del done
            pending.popleft()
            del future
            _check_cancel(should_cancel)
            yield result
            del result
            submit_one()

    results = consume()
    try:
        yield results
    finally:
        results.close()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        pending.clear()


def _save_registration(images, output_path, filenames, should_cancel):
    """Publish only after all transforms succeed; each file commits atomically."""
    _check_cancel(should_cancel)
    if not output_path:
        return
    import tempfile
    from utils.image_utils import imwrite_auto
    os.makedirs(output_path, exist_ok=True)
    for index, image in enumerate(images):
        _check_cancel(should_cancel)
        name = filenames[index] if filenames else f'frame_{index:04d}.png'
        destination = os.path.join(output_path, name)
        fd, temporary = tempfile.mkstemp(prefix='.align-',
                                         suffix=os.path.splitext(name)[1],
                                         dir=output_path)
        os.close(fd)
        try:
            if not imwrite_auto(temporary, image):
                raise OSError(f"Could not save aligned frame: {name}")
            _check_cancel(should_cancel)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    _check_cancel(should_cancel)



def _report_helicon_transform(transform, reference_index):
    if alignment_report.get() is None:
        return
    _report_alignment(transform.source_index, 'openfocus',
        'reference' if transform.source_index == reference_index else
        'fallback' if transform.failure else 'aligned',
        transform=np.linalg.inv(transform.inverse_matrix),
        detector_parameters=list(transform.f8), pair_parameters=list(transform.pair_f8),
        reference_index=transform.reference_index, rounds=transform.rounds,
        **({'reason': transform.failure} if transform.failure else {}))


def apply_alignment_plan(sources, plan, *, defer_warps=False, should_cancel=None):
    """Apply a validated cached or fresh plan, preserving original source order."""
    from functools import partial
    from core.hf9_alignment import apply_alignment, alignment_resolution_bytes
    from core.source_frames import resolve_frame
    plan.validate_sources(sources, should_cancel=should_cancel)
    for transform in plan.transforms:
        _check_cancel(should_cancel)
        _report_helicon_transform(transform, plan.reference_index)
    h, w = sources[0].shape[:2]
    left, top, right, bottom = 0, 0, w, h
    if grouped_crop_offsets.get() is not None:
        bounds = [transform.valid_bounds for transform in plan.transforms]
        left = max(bound[0] for bound in bounds)
        top = max(bound[1] for bound in bounds)
        right = min(bound[2] for bound in bounds)
        bottom = min(bound[3] for bound in bounds)
        if top >= bottom or left >= right:
            raise ValueError('Registration transforms have no common valid crop')
        _record_grouped_crop(top, bottom, left, right)
    result = []
    for source, transform in zip(sources, plan.transforms):
        _check_cancel(should_cancel)
        processor = partial(apply_alignment, transform=transform, should_cancel=should_cancel)
        if defer_warps:
            from core.source_frames import process_frame, crop_frame
            aligned = process_frame(source, processor, should_cancel=should_cancel,
                                    scratch_bytes=alignment_resolution_bytes(source))
            h, w = source.shape[:2]
            if (left, top, right, bottom) != (0, 0, w, h):
                aligned = crop_frame(aligned, left, top, right - left, bottom - top,
                                     should_cancel=should_cancel)
        else:
            image = processor(resolve_frame(source))
            aligned = store_frame(image[top:bottom, left:right])
        result.append(aligned)
    _check_cancel(should_cancel)
    return result


def _align_helicon_impl(input_source, output_path=None, img_filenames=None, *,
                        should_cancel=None, max_shift_ratio=.05, failure_policy='continue',
                        reference_frame=1, rotation=0., magnification=3.,
                        defer_warps=False, plan_callback=None):
    """Verified HF9 wide detector; no alternate estimator or native fit."""
    from core.hf9_alignment import estimate_alignment_plan, AlignmentEstimationError
    images = input_source
    if isinstance(input_source, str):
        from core.source_frames import scratch_source_headers
        extensions = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        paths = sorted((os.path.join(input_source, name) for name in os.listdir(input_source)
                        if os.path.splitext(name)[1].lower() in extensions), key=_image_sort_key)
        images = scratch_source_headers(paths, should_cancel)
        img_filenames = [os.path.basename(path) for path in paths]
    if images is None or not len(images):
        return []
    try:
        plan = estimate_alignment_plan(images, max_shift_ratio=max_shift_ratio,
            reference_frame=reference_frame, rotation=rotation, magnification=magnification,
            should_cancel=should_cancel,
            failure_policy=failure_policy)
    except AlignmentEstimationError as error:
        for transform in getattr(error, 'completed_transforms', ()):
            _report_helicon_transform(transform, reference_frame - 1)
        _report_alignment(getattr(error, 'frame_index', None), 'openfocus', 'failed', reason=str(error))
        raise
    if plan_callback is not None:
        plan_callback(plan)
    result = apply_alignment_plan(images, plan, defer_warps=defer_warps, should_cancel=should_cancel)
    if output_path:
        from core.source_frames import resolve_frame
        _save_registration((resolve_frame(frame) for frame in result), output_path,
                           img_filenames, should_cancel)
    return result


# ========== ECC对齐算法实现（高精度） ==========

def _align_ecc_impl(input_source, output_path=None, img_filenames=None, downscale_width=1000, thread_count: int = 4, should_cancel=None, use_gpu: bool = False, failure_policy='continue'):
    """
    基于 ECC (增强相关系数) 的高精度图像栈对齐算法
    适用于：显微摄影、微距摄影中伴随呼吸效应的图像栈
    优势：亚像素精度，自动处理缩放中心偏移，不依赖特征点
    优化：并行预处理、并行变换、合并裁切操作
    """
    import concurrent.futures

    # --- 1. 数据加载 ---
    if img_filenames is None and isinstance(input_source, str):
        img_paths = sorted(
            (os.path.join(input_source, f) for f in os.listdir(input_source)
             if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp', '.tif', '.tiff', '.webp'}),
            key=_image_sort_key
        )
        images = [cv2.imread(path) for path in img_paths]
        img_filenames = [os.path.basename(path) for path in img_paths]
    else:
        images = input_source
    if len(images):
        _report_reference('ecc')

    if len(images) < 2:
        return images

    # --- 2. 初始化 ---
    h_orig, w_orig = images[0].shape[:2]
    # 如果单张图像过大（任一边 >= 2048），强制将用于下采样的目标宽度设为 1024
    max_dim = max(h_orig, w_orig)
    if max_dim >= 2048:
        try:
            prev_down = downscale_width
        except NameError:
            prev_down = None
        downscale_width = 1024
        print(f"[Registration][ECC] Large image detected ({h_orig}x{w_orig}), setting downscale_width {prev_down} -> {downscale_width}")
    
    # 全局变换矩阵 (3x3 单位矩阵)
    H_global = np.eye(3, dtype=np.float32)
    
    # 收集所有变换矩阵用于精确裁切
    H_matrices = [np.eye(3, dtype=np.float32)]  # 第一张是基准
    
    # 定义 ECC 变换类型
    warp_mode = cv2.MOTION_HOMOGRAPHY 
    
    # ECC 终止条件
    number_of_iterations = 50
    termination_eps = 1e-4
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # 预处理函数：转灰度 + 降采样 + 高斯模糊 + 归一化
    def preprocess(img):
        h, w = img.shape[:2]
        scale = downscale_width / float(w) if w > downscale_width else 1.0
        if scale < 1.0:
            small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small_img = img # 引用即可，无需拷贝

        if small_img.ndim == 2:
            gray = small_img
        elif small_img.shape[2] == 1:
            gray = small_img[:, :, 0]
        else:
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # findTransformECC 的数值稳定性依赖归一化的浮点输入：uint8 的
        # 大幅值梯度会让优化步长过大，常见于小位移场景直接不收敛
        gray = gray.astype(np.float32) / (65535.0 if gray.dtype == np.uint16 else 255.0)
        return gray, scale

    print(f"Aligning {len(images)} images using ECC (Parallel Optimized)...")

    # --- 3. 并行预处理 ---
    # print("  - Step 1/3: Preprocessing images concurrently...")
    try:
        max_workers = max(1, int(thread_count))
    except Exception:
        max_workers = min(8, os.cpu_count() or 1)

    with _bounded_registration_map(preprocess, images, max_workers,
                                   should_cancel) as results:
        preprocessed_data = [(store_frame(gray), scale) for gray, scale in results]

    # 所有帧同尺寸，缩放比例一致
    scale_factor = preprocessed_data[0][1]


    # --- 4. 计算矩阵 (相邻帧对彼此独立 -> 并行计算, 串行累积) ---
    # print("  - Step 2/3: Calculating ECC matrices...")

    def _compute_pair_transform(idx: int) -> np.ndarray:
        """ECC transform of frame idx relative to frame idx-1 (scale-restored)."""
        prev_gray = preprocessed_data[idx - 1][0]
        curr_gray, _ = preprocessed_data[idx]

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            cc, warp_matrix = cv2.findTransformECC(
                prev_gray,  # template
                curr_gray,  # input
                warp_matrix,
                warp_mode,
                criteria,
                None,
                1
            )

            # 尺度还原
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix[0, 2] /= scale_factor
                warp_matrix[1, 2] /= scale_factor
                warp_matrix[2, 0] *= scale_factor
                warp_matrix[2, 1] *= scale_factor
            else:
                warp_matrix[0, 2] /= scale_factor
                warp_matrix[1, 2] /= scale_factor
        except cv2.error:
            cc = None
            print(f"Warning: ECC failed to converge at frame {idx}. Assuming no motion.")
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
        return warp_matrix, cc

    with _bounded_registration_map(_compute_pair_transform,
                                   range(1, len(images)), max_workers,
                                   should_cancel) as results:
        pair_transforms = list(results)
    del preprocessed_data

    trajectory_failed = False
    for idx, (warp_matrix, cc) in enumerate(pair_transforms, start=1):
        _check_cancel(should_cancel)
        if cc is None and failure_policy == 'fail':
            _report_alignment(idx, 'ecc', 'failed', reason='ECC failed to converge',
                              reference_index=idx - 1)
            raise ValueError(f'ECC alignment failed at frame {idx + 1}: ECC failed to converge')
        # 矩阵累积：warp_matrix maps frame k-1 -> frame k, so the map
        # frame0 -> frame_k is Wk @ ... @ W1 — compose on the LEFT.
        # (The old right-side product W1 @ W2 @ ... left frame 3+ misaligned
        # for any rotation/scale; pure translations commute and hid it.)
        if warp_mode == cv2.MOTION_AFFINE:
            row = np.array([[0, 0, 1]], dtype=np.float32)
            H_local_3x3 = np.vstack([warp_matrix, row])
            H_global = np.matmul(H_local_3x3, H_global)
        else:
            H_global = np.matmul(warp_matrix, H_global)

        # 记录变换矩阵（使用逆矩阵）
        H_inv = np.linalg.inv(H_global)
        H_matrices.append(H_inv.copy())
        reason = ('ECC failed to converge' if cc is None else
                  'reference alignment fell back' if trajectory_failed else None)
        trajectory_failed = trajectory_failed or cc is None
        details = {'reference_index': idx - 1, 'transform': H_inv}
        if cc is not None:
            details['quality'] = float(cc)
        if reason is not None:
            details['reason'] = reason
        _report_alignment(idx, 'ecc', 'fallback' if reason else 'aligned', **details)

    # --- 5. 并行应用变换与裁切 ---
    # print("  - Step 3/3: Warping and saving concurrently...")

    # 计算公共有效区域
    top, bottom, left, right = _compute_valid_region_from_transforms(H_matrices, (h_orig, w_orig))
    _record_grouped_crop(top, bottom, left, right)
    
    do_crop = True
    if top >= bottom or left >= right:
        print("Warning: Invalid crop region, skipping crop.")
        do_crop = False
        target_w, target_h = w_orig, h_orig
        offset_x, offset_y = 0, 0
    else:
        target_w = right - left
        target_h = bottom - top
        offset_x = -left
        offset_y = -top
        # print(f"    Optimized: Warping directly to cropped region ({target_w}x{target_h})...")

    # 构造裁切平移矩阵
    T_crop = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    # 尝试导入 cupy。仅当调用方显式要求 GPU 时才使用——以前只要机器上
    # 装了 CuPy 就静默切到 GPU，`use_gpu=False` / CLI `--cpu` 无法关闭它，
    # 而且这条路径用双线性插值、CPU 路径用 Lanczos4，两者本来就不等价。
    HAS_CUPY = False
    cp = None
    if use_gpu:
        try:
            import cupy as cp
            import cupyx.scipy.ndimage
            from core.resource_policy import ResourceLimitError, ResourceSnapshot, vram_budget
            free, total = cp.cuda.runtime.memGetInfo()
            budget = vram_budget(ResourceSnapshot(free_vram=int(free),
                                                  total_vram=int(total),
                                                  device_kind="CUDA"))
            # Float64 grids, homogeneous coordinates, channel output and
            # interpolation workspace coexist in this legacy backend.
            required = (target_h * target_w * 192
                        + max(image.nbytes for image in images))
            if required > budget:
                raise ResourceLimitError("Insufficient VRAM for explicit CuPy alignment")
            HAS_CUPY = True
            print("    [Info] Explicit CuPy alignment uses linear interpolation.")
        except ImportError as error:
            raise RuntimeError("Explicit CuPy alignment backend is unavailable") from error
    else:
        print("    [Info] GPU warping disabled. Using CPU for warping.")
    log_action('registration_warp', method='ecc',
               provider='CuPy' if HAS_CUPY else 'OpenCV',
               device='CUDA' if HAS_CUPY else 'CPU', shape=(target_h, target_w),
               frames=len(images), gpu_requested=use_gpu)

    def warp_task(args):
        idx, img, H = args
        
        # 合并裁切变换：先变换 H，再平移 T_crop
        if do_crop:
            H_final = T_crop @ H
        else:
            H_final = H
            
        # 如果有 Cupy，使用 GPU 加速
        if HAS_CUPY:
            # 将图像传输到 GPU
            img_gpu = cp.asarray(img)
            
            # Cupy 的 affine_transform 需要逆变换矩阵
            # cv2.warpPerspective 使用的是 H_final (前向映射矩阵的逆，即从目标到源)
            # 但 ndimage.affine_transform 也需要从输出坐标映射回输入坐标的矩阵
            # 注意：ndimage.affine_transform 对矩阵的定义可能与 OpenCV 不同
            # OpenCV: dst(x,y) = src(M * [x,y,1])
            # ndimage: output[i, j] = input[matrix @ [i,j] + offset]
            
            # 对于透视变换 (Homography)，affine_transform 不够用，因为它只支持仿射
            # 我们需要手动构建坐标网格并使用 map_coordinates
            
            # 创建目标网格
            y_grid, x_grid = cp.meshgrid(cp.arange(target_h), cp.arange(target_w), indexing='ij')
            
            # 展平网格
            ones = cp.ones_like(x_grid)
            coords = cp.stack([x_grid, y_grid, ones]) # 3 x N
            coords = coords.reshape(3, -1)
            
            # 应用变换矩阵 (H_final 已经是 H_inv，即从目标到源的映射)
            # src_coords = H_final @ dst_coords
            # cv2.warpPerspective treats M as the forward src->dst map and
            # inverts it internally; manual grid sampling must therefore use
            # the inverse so the CuPy path matches the CPU path.
            H_gpu = cp.asarray(np.linalg.inv(H_final))
            src_coords_homo = cp.matmul(H_gpu, coords)
            
            # 归一化齐次坐标
            w_coords = src_coords_homo[2, :]
            w_coords = cp.where(cp.abs(w_coords) < 1e-10, 1e-10, w_coords)
            src_x = src_coords_homo[0, :] / w_coords
            src_y = src_coords_homo[1, :] / w_coords
            
            # 重塑回图像形状
            src_x = src_x.reshape(target_h, target_w)
            src_y = src_y.reshape(target_h, target_w)
            
            # 对每个通道进行插值
            channels = []
            planes = ((img_gpu,) if img.ndim == 2 else
                      (img_gpu[:, :, c] for c in range(img.shape[2])))
            for plane in planes:
                # order=1 (linear) 速度最快, order=3 (cubic) 质量更好
                # 这里使用 order=1 以获得最大加速，如果追求质量可用 order=3
                channel_out = cupyx.scipy.ndimage.map_coordinates(
                    plane,
                    cp.stack([src_y, src_x]), 
                    order=1, 
                    mode='constant', 
                    cval=0
                )
                channels.append(channel_out)
            
            aligned_img_gpu = (channels[0] if len(channels) == 1 else
                               cp.stack(channels, axis=2))
            
            # 传回 CPU
            aligned_img = cp.asnumpy(aligned_img_gpu).astype(img.dtype)
            
        else:
            # CPU 版本 (OpenCV)
            aligned_img = cv2.warpPerspective(
                img,
                H_final,
                (target_w, target_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT
            )
        
        return aligned_img

    task_args = ((i, image, H_matrices[i]) for i, image in enumerate(images))
    # The explicit legacy GPU backend is serialized; never mix CPU/GPU frames.
    with _bounded_registration_map(warp_task, task_args,
                                   1 if HAS_CUPY else max_workers,
                                   should_cancel) as results:
        aligned_images = [store_frame(image) for image in results]
    _save_registration(aligned_images, output_path, img_filenames, should_cancel)

    return aligned_images


# ========== 稳定配准算法实现 ==========

def _stabilisation_impl(input_source, output_path=None, filenames=None):
    """
    Stabilize images from directory path or image list
    Args:
        input_source: string (directory path) or list of images
        output_path: string, directory path to save results (optional, None means no saving)
        filenames: list of original filenames (optional)
    Returns:
        list of stabilized images (always returns processed images regardless of output_path)
    """
    # Process input source
    img_filenames = filenames  # Use passed filenames or detect from input
    if isinstance(input_source, str):
        img_paths = sorted(
            [os.path.join(input_source, file) for file in os.listdir(input_source)
             if os.path.splitext(file)[1].lower() in ['.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp', '.tif', '.tiff', '.webp']],
            key=_image_sort_key
        )
        images = [cv2.imread(path) for path in img_paths]
        # Store original filenames with extensions
        img_filenames = [os.path.basename(path) for path in img_paths]
    else:
        images = input_source

    n_frames = len(images)
    img_first = images[0]
    h, w = img_first.shape[:2]

    prev_gray = cv2.cvtColor(img_first, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames, 3), np.float32)

    # Feature detection parameters
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate transforms (frame i to frame i+1)
    for i in range(n_frames - 1):
        curr = images[i + 1]
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
        if prev_pts is None:
            # 如果没有找到特征点，使用上一次的变换
            if i > 0:
                transforms[i] = transforms[i-1]
            continue
            
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)

        idx = status.ravel() == 1
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        if len(prev_pts) < 4:
            # 如果匹配点太少，使用上一次的变换
            if i > 0:
                transforms[i] = transforms[i-1]
            continue

        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        if m is None:
            # 如果估计失败，使用上一次的变换
            if i > 0:
                transforms[i] = transforms[i-1]
            continue
            
        dx, dy = m[0, 2], m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]
        prev_gray = curr_gray

    # 最后一帧使用与倍数第二帧相同的变换
    transforms[-1] = transforms[-2]

    # Smooth trajectory
    def smooth(trajectory, radius=30):
        smoothed_trajectory = np.copy(trajectory)
        kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
        padding = np.pad(trajectory, ((radius, radius), (0, 0)), 'edge')
        for i in range(3):
            smoothed_trajectory[:, i] = np.convolve(padding[:, i], kernel, mode='valid')
        return smoothed_trajectory

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory

    # Apply transforms
    stabilized_images = []
    for i, frame in enumerate(images):
        # 每一帧应用对应的 difference 校正量
        dx, dy, da = difference[i]
        
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da), np.cos(da), dy]
        ], dtype=np.float32)

        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.04)
        frame_stabilized = cv2.warpAffine(frame_stabilized, T, (w, h))
        stabilized_images.append(store_frame(frame_stabilized))

        # Save results if output path is provided
        if output_path:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # Use original filename if available, otherwise use default naming
            if img_filenames:
                output_file = os.path.join(output_path, img_filenames[i])
            else:
                output_file = os.path.join(output_path, f'frame_{i:04d}.png')
            cv2.imwrite(output_file, frame_stabilized)

    return stabilized_images


# ========== 统一接口类 ==========

class ImageRegistration:
    """
    图像序列配准统一接口类
    
    支持的方法:
    - 'homography': 单应性对齐配准（非线性）
    - 'ecc': ECC对齐配准（高精度、亚像素级）
    - 'both': 组合配准（先 Homography 后 ECC）
    
    示例:
        # 使用单应性对齐
        registration = ImageRegistration(method='homography')
        result = registration.process('./images', './output')
        
        # 使用ECC对齐
        registration = ImageRegistration(method='ecc')
        result = registration.process(image_list, './output')

        # 使用组合对齐
        registration = ImageRegistration(method='both')
        result = registration.process(image_list, './output')
    """
    
    def __init__(self, method='homography', downscale_width=1024,
                 max_shift_ratio=None, brightness_reference='first',
                 use_gpu=False, parameters=None, *, defer_warps=False):
        from core.algorithms import (alignment_steps, registration_parameter_specs,
                                     resolve_registration_parameters)
        self.method = method
        self.steps = alignment_steps(method)
        values = resolve_registration_parameters(method, parameters)
        if max_shift_ratio is not None:
            # Historical constructor argument: only the OpenFocus aligner
            # consumes it, and it used to be ignored for other methods.
            if 'max_shift_ratio' in {item.key for item in registration_parameter_specs(method)}:
                values['max_shift_ratio'] = max_shift_ratio
        self.parameters = resolve_registration_parameters(method, values)
        self.downscale_width = int(downscale_width) if downscale_width is not None else 1024
        self.brightness_reference = brightness_reference if brightness_reference in ('first', 'last') else 'first'
        self.use_gpu = bool(use_gpu)
        self.defer_warps = bool(defer_warps)
        self.alignment_plan = None
    
    def process(self, 
                input_source: Union[str, List[np.ndarray]], 
                output_path: Optional[str] = None,
                thread_count: int = 4,
                should_cancel=None, resource_preference="balanced") -> List[np.ndarray]:
        """
        执行图像配准
        
        Args:
            input_source (str or list): 图像目录路径或预加载的图像列表
            output_path (str, optional): 输出目录路径。
                - 如果提供路径，会将配准后的图像保存到磁盘
                - 如果为 None，只返回图像列表不保存，节省磁盘空间和 I/O 开销
        
        Returns:
            list: 配准后的图像列表（始终返回，无论是否保存到磁盘）
        """
        from core.resource_policy import Workload, heavy_job, plan_execution
        from core.frame_storage import estimate_registration_space, scratch_directory, select_scratch

        with heavy_job(should_cancel), scratch_directory():
            _check_cancel(should_cancel)
            self.alignment_plan = None
            if self.steps == ('openfocus',):
                from core.algorithms import get_registration, resolve_runner
                sources = input_source if input_source is not None else []
                if not isinstance(sources, str) and not hasattr(sources, '__len__'):
                    sources = tuple(sources)
                return resolve_runner(get_registration('openfocus').runner)(self, sources,
                    thread_count=thread_count, should_cancel=should_cancel,
                    output_path=output_path, **self.parameters)
            filenames = None
            images = input_source
            if isinstance(input_source, str):
                extensions = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png',
                              '.bmp', '.tif', '.tiff', '.webp'}
                paths = sorted((os.path.join(input_source, name)
                                for name in os.listdir(input_source)
                                if os.path.splitext(name)[1].lower() in extensions),
                               key=_image_sort_key)
                from core.image_loader import admit_image_decode, decode_image
                from core.source_frames import scratch_source_headers
                headers = scratch_source_headers(paths, should_cancel)
                if headers:
                    input_bytes = sum(frame.nbytes for frame in headers)
                    select_scratch(input_bytes, paths, preferred_size=(
                        input_bytes + estimate_registration_space(headers, len(self.steps))))
                images = []
                for path in paths:
                    _check_cancel(should_cancel)
                    admit_image_decode(path)
                    image = (decode_image(path, preserve_grayscale=True)
                             if self.steps == ('openfocus',) else decode_image(path))
                    if image is None:
                        raise ValueError(f"Could not decode registration frame: {path}")
                    images.append(store_frame(image))
                filenames = [os.path.basename(path) for path in paths]
            else:
                from collections.abc import Sequence
                from core.source_frames import materialize_frames
                staging = (estimate_registration_space(input_source, len(self.steps))
                           if isinstance(input_source, Sequence) else 0)
                images = (materialize_frames(input_source, should_cancel,
                                             preferred_staging_bytes=staging)
                          if input_source is not None else [])
            if images is not None and len(images) and 'openfocus' in self.steps and self.parameters['reference_frame'] > len(images):
                raise ValueError(f"OpenFocus reference frame {self.parameters['reference_frame']} is outside the stack of {len(images)} frames")

            from core.image_loader import _has_transparency, UnsupportedTransparency
            for image in images or ():
                _check_cancel(should_cancel)
                if _has_transparency(image):
                    raise UnsupportedTransparency("Registration of transparent images is unsupported")

            if images is None or len(images) < 2:
                _check_cancel(should_cancel)
                if images:
                    for step in self.steps:
                        _report_reference(step)
                return images if images is not None else []
            if not self.steps:
                _save_registration(images, output_path, filenames, should_cancel)
                return images
            height, width = images[0].shape[:2]
            channels = images[0].shape[2] if images[0].ndim == 3 else 1
            # Grayscale caches and aligned frames are disk-backed. Charge only
            # the active frame scratch; worker slots account for concurrent ECC.
            gray_bytes = height * width * 4
            plan = plan_execution(Workload(
                algorithm='registration', frames=len(images), height=height,
                width=width, channels=channels, itemsize=images[0].dtype.itemsize,
                fixed_bytes=gray_bytes,
                per_worker_bytes=max(image.size for image in images) * 4 * 4,
                outer_tasks=len(images), parallel_layer='outer'),
                preference=resource_preference, user_threads=thread_count)
            workers = min(plan.workers, plan.max_in_flight)
            log_action('registration_plan', method=self.method, frames=len(images),
                       shape=images[0].shape, workers=workers,
                       native_threads=plan.native_threads, ram_budget=plan.ram_budget,
                       vram_budget=plan.vram_budget, gpu_requested=self.use_gpu)
            from core.algorithms import get_registration, resolve_runner
            result = images
            for step in self.steps:
                _check_cancel(should_cancel)
                spec = get_registration(step)
                parameters = {item.key: self.parameters[item.key] for item in spec.parameters}
                with log_operation('registration_stage', method=step,
                                   parameters=parameters, provider='OpenCV',
                                   estimation_device='CPU', frames=len(result),
                                   shape=result[0].shape) as operation:
                    try:
                        result = resolve_runner(spec.runner)(self, result,
                            thread_count=workers, should_cancel=should_cancel, **parameters)
                        _check_cancel(should_cancel)
                    except RegistrationCancelled:
                        operation.cancel()
                        raise
            _save_registration(result, output_path, filenames, should_cancel)
            return result
    
    def _process_homography(self, 
                           input_source: Union[str, List[np.ndarray]], 
                           output_path: Optional[str] = None,
                           thread_count: int = 4,
                           should_cancel=None, failure_policy='continue') -> List[np.ndarray]:
        """
        单应性对齐配准（非线性）
        
        Args:
            input_source: 图像源
            output_path: 输出路径
        
        Returns:
            配准后的图像列表
        """
        return _align_homography_impl(input_source, output_path, downscale_width=self.downscale_width, thread_count=thread_count, should_cancel=should_cancel, failure_policy=failure_policy)
    
    def _process_ecc(self, 
                    input_source: Union[str, List[np.ndarray]], 
                    output_path: Optional[str] = None,
                    thread_count: int = 4,
                    should_cancel=None, failure_policy='continue') -> List[np.ndarray]:
        """
        ECC对齐配准（高精度、亚像素级）
        
        Args:
            input_source: 图像源
            output_path: 输出路径
        
        Returns:
            配准后的图像列表
        """
        return _align_ecc_impl(input_source, output_path, downscale_width=self.downscale_width, thread_count=thread_count, should_cancel=should_cancel, use_gpu=self.use_gpu, failure_policy=failure_policy)


    def set_method(self, method):
        from core.algorithms import alignment_steps, resolve_registration_parameters
        steps = alignment_steps(method)
        parameters = resolve_registration_parameters(method)
        self.method = method
        self.steps = steps
        self.parameters = parameters
    
    def get_info(self) -> dict:
        """
        获取当前配准器信息
        
        Returns:
            dict: 包含配准方法等信息
        """
        return {
            'method': self.method
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"ImageRegistration(method='{self.method}')"


# ========== 便捷函数 ==========

def register_images(input_source: Union[str, List[np.ndarray]],
                    method: str = 'homography',
                    output_path: Optional[str] = None) -> List[np.ndarray]:
    """
    便捷函数:一次性完成图像配准
    
    Args:
        input_source: 图像源(目录路径或图像列表)
        method: 配准方法 ('homography', 'ecc', 'both')
        output_path: 输出路径 (可选，None 表示只返回列表不保存)
    
    Returns:
        配准后的图像列表
    
    示例:
        # 保存到磁盘并返回列表
        result = register_images('./images', method='homography', output_path='./output')
        
        # 只返回列表不保存，节省磁盘空间
        result = register_images('./images', method='ecc')
        
        # 组合对齐
        result = register_images(image_list, method='both')
    """
    registration = ImageRegistration(method=method)
    return registration.process(input_source, output_path=output_path)


def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description='Image Stack Registration Tool')
    parser.add_argument('--input_path', default='./coral_best_zoom', help='Input image directory path')
    parser.add_argument('--output', default=r'E:\FinishedProjects\XuChuang\regi_test', help='Output directory path')
    from core.algorithms import registration_algorithms
    parser.add_argument('--mode', default='homography',
                        choices=[spec.id for spec in registration_algorithms()],
                        help='Registered alignment method')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        print(f"\n=== {args.mode.upper()} Alignment ===")
        output_dir = os.path.join(args.output, f'{args.mode}_result')
        registration = ImageRegistration(method=args.mode)
        processed_images = registration.process(args.input_path, output_dir)
        print(f"Alignment completed. Results saved to: {output_dir}")
        print(f"Processed {len(processed_images)} images")
        
        end_time = time.time()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
