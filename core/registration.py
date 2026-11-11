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
from dataclasses import asdict, dataclass
import json
import threading
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
from utils.performance_profile import current as _profile



# ========== 基于变换矩阵的精确裁切函数 ==========

from contextvars import ContextVar

# Grouped fusion records crop offsets in the reference image's pixel grid.
grouped_crop_offsets = ContextVar('grouped_crop_offsets', default=None)

# Optional per-frame diagnostics, collected on the calling thread.
alignment_report = ContextVar('alignment_report', default=None)

# OpenCV's thread setting is process-wide. All temporary exchanges share this
# lock; the owner budget is local to the source-producing context.
_warp_threads_lock = threading.Lock()
_deferred_warp_threads = ContextVar('deferred_warp_threads', default=1)


@contextmanager
def deferred_warp_threads(threads):
    token = _deferred_warp_threads.set(threads)
    try:
        yield
    finally:
        _deferred_warp_threads.reset(token)


def exchange_opencv_threads(threads):
    with _warp_threads_lock:
        previous = cv2.getNumThreads()
        cv2.setNumThreads(threads)
        return previous


def _warp_thread_budget():
    # Build information alone does not identify the active backend.
    current_backend = getattr(cv2, 'currentParallelFramework', None)
    if current_backend is None:
        return 1
    try:
        backend = current_backend()
        if not isinstance(backend, str) or backend.upper() not in ('TBB', 'OPENMP', 'PTHREADS'):
            return 1
        threads = _deferred_warp_threads.get()
        return max(1, int(threads() if callable(threads) else threads))
    except Exception:
        return 1


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


def _align_homography_impl(input_source, output_path=None, img_filenames=None, downscale_width=1600, thread_count: int = 4, should_cancel=None, failure_policy='continue', *, defer_warps=False, plan_callback=None):
    """
    商业级图像对齐算法优化版
    特性：
    1. 传递对齐 (Sequential Alignment)：解决大景深下的特征丢失问题
    2. 金字塔加速 (Downscale Processing)：大幅提升特征检测速度
    3. 矩阵累积 (Matrix Chaining)：减少累积误差
    4. Lanczos插值：保证画质清晰度
    5. 并行计算优化 (Parallel Processing)：利用多核加速特征提取和图像变换
    """
    # Directory loading and source policy belong to ImageRegistration.process.
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
        from core.source_frames import resolve_frame
        source = img
        img = resolve_frame(img, should_cancel=should_cancel)
        identity = _perspective_source(source, img, should_cancel)
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
        return kps, des, scale, identity

    # 使用线程池并行提取特征
    # OpenCV 的大部分操作释放 GIL，因此多线程可以有效加速
    try:
        max_workers = max(1, int(thread_count))
    except Exception:
        max_workers = min(8, os.cpu_count() or 1)

    # Matching retains its last accepted reference, including failed frames.
    bf = cv2.BFMatcher(cv2.NORM_L2)
    with _bounded_registration_map(get_features_task, images, max_workers,
                                   should_cancel, stage='registration.features') as features:
        last_kps, last_des, last_scale, identity = next(features)
        source_identities = [identity]
        last_index = 0
        trajectory_failed = False
        for idx, (curr_kps, curr_des, curr_scale, identity) in enumerate(features, 1):
            source_identities.append(identity)
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

    aligned_images = _apply_perspective_warps(
        images, H_matrices, T_crop if do_crop else None, (target_w, target_h),
        max_workers, should_cancel, defer_warps, method='homography', sources=source_identities,
        bounds=(top, bottom, left, right), plan_callback=plan_callback)
    _save_registration(aligned_images, output_path, img_filenames, should_cancel)

    return aligned_images


class RegistrationCancelled(Exception):
    """Raised when a render cancellation flag fires mid-registration."""

def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise RegistrationCancelled()


def _perspective_source(frame, image, should_cancel):
    from core.openfocus_alignment import (SourceFingerprint, _source_file_identity,
                                    _pixel_digest, _source_context)
    path, page, stamp = _source_file_identity(frame)
    source = SourceFingerprint(tuple(image.shape), image.dtype.str,
                              _pixel_digest(image, should_cancel), path, page,
                              stamp, _source_context(frame))
    source.validate_file()
    return source


@dataclass(frozen=True)
class PerspectiveWarp:
    source: object
    matrix: tuple
    matrix_dtype: str
    dsize: tuple
    should_cancel: object = None

    def __call__(self, image):
        self.source.validate_pixels(image, self.should_cancel)
        matrix = np.asarray(self.matrix, dtype=self.matrix_dtype).reshape(3, 3)
        threads = _warp_thread_budget()
        if threads <= 1:
            return cv2.warpPerspective(image, matrix, self.dsize,
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        with _warp_threads_lock:
            previous = cv2.getNumThreads()
            cv2.setNumThreads(threads)
            try:
                return cv2.warpPerspective(image, matrix, self.dsize,
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
            finally:
                cv2.setNumThreads(0 if previous == 1 else previous)


def _perspective_output_shape(shape, dsize):
    # OpenCV drops a singleton channel axis when producing warp output.
    channels = () if shape[2:] == (1,) else shape[2:]
    return (dsize[1], dsize[0]) + channels


@dataclass(frozen=True)
class PerspectiveStage:
    method: str
    sources: tuple
    matrices: tuple
    matrix_dtypes: tuple
    bounds: tuple
    dsize: tuple

    def __post_init__(self):
        count = len(self.sources)
        if (self.method not in ('homography', 'ecc') or count < 2
                or not all(isinstance(value, tuple) for value in
                           (self.sources, self.matrices, self.matrix_dtypes, self.bounds, self.dsize))
                or len(self.matrices) != count or len(self.matrix_dtypes) != count
                or len(self.bounds) != 4 or len(self.dsize) != 2
                or any(type(value) is not int or value <= 0 for value in self.dsize)
                or any(type(value) is not int for value in self.bounds)):
            raise ValueError('Invalid perspective stage geometry')
        top, bottom, left, right = self.bounds
        height, width = self.sources[0].shape[:2]
        if top < bottom and left < right:
            if not (0 <= top < bottom <= height and 0 <= left < right <= width
                    and self.dsize == (right - left, bottom - top)):
                raise ValueError('Invalid perspective crop')
        elif self.dsize != (width, height):
            raise ValueError('Invalid uncropped perspective size')
        for source, matrix, dtype in zip(self.sources, self.matrices, self.matrix_dtypes):
            if (not isinstance(matrix, tuple) or len(matrix) != 9
                    or np.dtype(dtype) not in (np.dtype('float32'), np.dtype('float64'))
                    or not np.all(np.isfinite(matrix))
                    or len(source.shape) not in (2, 3) or any(value <= 0 for value in source.shape)
                    or np.dtype(source.dtype).kind not in 'uf'
                    or len(source.sha256) != 64
                    or any(value not in '0123456789abcdef' for value in source.sha256)):
                raise ValueError('Invalid perspective transform or source')

    def validate_sources(self, sources, *, should_cancel=None):
        from core.openfocus_alignment import _source_context, _source_file_identity
        from core.source_frames import ProcessedFrame
        if len(sources) != len(self.sources):
            raise ValueError('Perspective source count changed')
        for frame, source in zip(sources, self.sources):
            _check_cancel(should_cancel)
            source.validate_file()
            if (tuple(frame.shape) != source.shape or np.dtype(frame.dtype).str != source.dtype
                    or _source_file_identity(frame) != (source.path, source.page, source.file_stamp)
                    or (not (isinstance(frame, ProcessedFrame) and
                             isinstance(frame.processor, PerspectiveWarp))
                        and _source_context(frame) != source.context)):
                raise ValueError('Perspective source identity changed')
            if isinstance(frame, np.ndarray):
                source.validate_pixels(frame, should_cancel)

    def apply(self, images, *, defer_warps, should_cancel=None, workers=1):
        from core.source_frames import process_frame, resolve_frame
        self.validate_sources(images, should_cancel=should_cancel)
        frames = [process_frame(image, PerspectiveWarp(source, matrix, dtype, self.dsize, should_cancel),
                    shape=_perspective_output_shape(image.shape, self.dsize), should_cancel=should_cancel)
                  for image, source, matrix, dtype in
                  zip(images, self.sources, self.matrices, self.matrix_dtypes)]
        if defer_warps:
            return frames
        with _bounded_registration_map(resolve_frame, frames, workers, should_cancel,
                                       stage='registration.warp') as results:
            return [store_frame(image) for image in results]


@dataclass(frozen=True)
class PerspectiveAlignmentPlan:
    """Ordered OpenCV resamplings; independent of the wide alignment plan schema."""
    stages: tuple
    parameters: str
    downscale_width: int

    def __post_init__(self):
        if (not isinstance(self.stages, tuple) or not self.stages
                or any(not isinstance(stage, PerspectiveStage) for stage in self.stages)
                or type(self.downscale_width) is not int or self.downscale_width <= 0
                or not isinstance(json.loads(self.parameters), dict)):
            raise ValueError('Invalid perspective alignment plan')
        for previous, current in zip(self.stages, self.stages[1:]):
            if len(previous.sources) != len(current.sources):
                raise ValueError('Perspective stage source count changed')
            for before, after in zip(previous.sources, current.sources):
                if (after.shape != _perspective_output_shape(before.shape, previous.dsize)
                        or before.dtype != after.dtype
                        or (before.path, before.page, before.file_stamp) !=
                           (after.path, after.page, after.file_stamp)):
                    raise ValueError('Perspective stage source chain changed')

    def validate_sources(self, sources, *, should_cancel=None):
        self.stages[0].validate_sources(sources, should_cancel=should_cancel)

    def to_dict(self):
        return {'version': 1, 'backend': 'opencv-perspective-v1', **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        from core.openfocus_alignment import SourceFingerprint
        if data['version'] != 1 or data['backend'] != 'opencv-perspective-v1':
            raise ValueError('Unsupported perspective plan version')
        stages = []
        for item in data['stages']:
            sources = tuple(SourceFingerprint(tuple(source['shape']), source['dtype'],
                source['sha256'], source['path'], source['page'], tuple(source['file_stamp']),
                source['context']) for source in item['sources'])
            stages.append(PerspectiveStage(item['method'], sources,
                tuple(tuple(matrix) for matrix in item['matrices']), tuple(item['matrix_dtypes']),
                tuple(item['bounds']), tuple(item['dsize'])))
        return cls(tuple(stages), data['parameters'], data['downscale_width'])


def _apply_perspective_warps(images, matrices, crop, dsize, workers,
                             should_cancel, defer_warps, *, method, sources,
                             bounds, plan_callback=None):
    """Keep every stage's Lanczos resampling, matrix precision and crop."""
    final = tuple(crop @ matrix if crop is not None else matrix.copy() for matrix in matrices)
    stage = PerspectiveStage(method, tuple(sources),
        tuple(tuple(float(value) for value in matrix.flat) for matrix in final),
        tuple(matrix.dtype.str for matrix in final), tuple(int(value) for value in bounds), dsize)
    if plan_callback is not None:
        plan_callback(stage)
    return stage.apply(images, defer_warps=defer_warps, should_cancel=should_cancel, workers=workers)

@contextmanager
def _bounded_registration_map(function, values, workers, should_cancel=None, *,
                              stage=None, max_in_flight=None):
    """Consume ordered results with a bounded number of retained tasks.

    Leaving the scope cancels queued work and waits for native calls before
    the enclosing heavy-job lease can be released. ``stage`` names the calls in
    a performance profile (pool threads: summed under ``concurrent``).
    """
    profile = _profile()
    _check_cancel(should_cancel)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    in_flight = workers if max_in_flight is None else max(workers, int(max_in_flight))
    pending = deque()
    source = iter(values)
    scratch = capture_scratch()

    def submit_one():
        _check_cancel(should_cancel)
        for future in pending:
            if future.done() and future.exception() is not None:
                future.result()
        try:
            value = next(source)
        except StopIteration:
            return
        _check_cancel(should_cancel)
        for future in pending:
            if future.done() and future.exception() is not None:
                future.result()
        pending.append(pool.submit(run_with_scratch, scratch, run, value))

    def run(value):
        _check_cancel(should_cancel)
        if stage is not None and profile.enabled:
            with profile.stage(stage):
                result = function(value)
        else:
            result = function(value)
        _check_cancel(should_cancel)
        return result

    def consume():
        for _ in range(in_flight):
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
        close = getattr(source, 'close', None)
        if close is not None:
            close()


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
        from core.source_frames import resolve_frame
        image = resolve_frame(image, should_cancel=should_cancel)
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



def _report_openfocus_transform(transform, reference_index):
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
    if isinstance(plan, PerspectiveAlignmentPlan):
        plan.validate_sources(sources, should_cancel=should_cancel)
        result = sources
        for stage in plan.stages:
            _record_grouped_crop(*stage.bounds)
            result = stage.apply(result, defer_warps=True, should_cancel=should_cancel)
        if defer_warps:
            return result
        from core.source_frames import resolve_frame
        return [store_frame(resolve_frame(frame, should_cancel=should_cancel)) for frame in result]
    from functools import partial
    from core.openfocus_alignment import apply_alignment, alignment_resolution_bytes
    from core.source_frames import resolve_frame
    plan.validate_sources(sources, should_cancel=should_cancel)
    for transform in plan.transforms:
        _check_cancel(should_cancel)
        _report_openfocus_transform(transform, plan.reference_index)
    h, w = sources[0].shape[:2]
    left, top, right, bottom = 0, 0, w, h
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
            image = processor(resolve_frame(source, should_cancel=should_cancel))
            aligned = store_frame(image[top:bottom, left:right])
        result.append(aligned)
    _check_cancel(should_cancel)
    return result


def _align_openfocus_impl(input_source, output_path=None, img_filenames=None, *,
                        should_cancel=None, max_shift_ratio=.05, failure_policy='continue',
                        reference_frame=1, rotation=0., magnification=3.,
                        defer_warps=False, plan_callback=None):
    """Wide detector using the verified search and sampling behavior."""
    from core.openfocus_alignment import estimate_alignment_plan, AlignmentEstimationError
    images = input_source
    if isinstance(input_source, str):
        from dataclasses import replace
        from core.source_frames import scratch_source_headers
        extensions = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        paths = sorted((os.path.join(input_source, name) for name in os.listdir(input_source)
                        if os.path.splitext(name)[1].lower() in extensions), key=_image_sort_key)
        images = scratch_source_headers(paths, should_cancel)
        images = [replace(frame, shape=frame.shape[:2]) if frame.grayscale else frame
                  for frame in images]
        img_filenames = [os.path.basename(path) for path in paths]
    if images is None or not len(images):
        return []
    from core.frame_storage import scratch_directory
    from core.source_frames import select_frame_scratch
    with scratch_directory():
        select_frame_scratch(images, should_cancel=should_cancel)
        try:
            plan = estimate_alignment_plan(images, max_shift_ratio=max_shift_ratio,
                reference_frame=reference_frame, rotation=rotation, magnification=magnification,
                should_cancel=should_cancel,
                failure_policy=failure_policy)
        except AlignmentEstimationError as error:
            for transform in getattr(error, 'completed_transforms', ()):
                _report_openfocus_transform(transform, reference_frame - 1)
            _report_alignment(getattr(error, 'frame_index', None), 'openfocus', 'failed', reason=str(error))
            raise
        if plan_callback is not None:
            plan_callback(plan)
        result = apply_alignment_plan(images, plan, defer_warps=defer_warps, should_cancel=should_cancel)
        if output_path:
            from core.source_frames import resolve_frame
            _save_registration((resolve_frame(frame, should_cancel=should_cancel) for frame in result), output_path,
                               img_filenames, should_cancel)
        return result


# ========== ECC对齐算法实现（高精度） ==========

def _align_ecc_impl(input_source, output_path=None, img_filenames=None, downscale_width=1000, thread_count: int = 4, should_cancel=None, use_gpu: bool = False, failure_policy='continue', *, defer_warps=False, plan_callback=None):
    """
    基于 ECC (增强相关系数) 的高精度图像栈对齐算法
    适用于：显微摄影、微距摄影中伴随呼吸效应的图像栈
    优势：亚像素精度，自动处理缩放中心偏移，不依赖特征点
    优化：并行预处理、并行变换、合并裁切操作
    """
    # Directory loading and source policy belong to ImageRegistration.process.
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
        from core.source_frames import resolve_frame
        source = img
        img = resolve_frame(img, should_cancel=should_cancel)
        identity = _perspective_source(source, img, should_cancel)
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
        return gray, scale, identity

    print(f"Aligning {len(images)} images using ECC (Parallel Optimized)...")

    # --- 3. 并行预处理 ---
    # print("  - Step 1/3: Preprocessing images concurrently...")
    try:
        max_workers = max(1, int(thread_count))
    except Exception:
        max_workers = min(8, os.cpu_count() or 1)

    source_identities = []
    def pairs():
        # Preparation runs on the submitting thread and consumes one slot.
        profile = _profile()
        def prepare(image):
            _check_cancel(should_cancel)
            with deferred_warp_threads(1):
                if profile.enabled:
                    with profile.stage('registration.prepare'):
                        result = preprocess(image)
                else:
                    result = preprocess(image)
            _check_cancel(should_cancel)
            return result

        source = iter(images)
        previous, scale_factor, identity = prepare(next(source))
        source_identities.append(identity)
        for idx, image in enumerate(source, 1):
            current, _, identity = prepare(image)
            source_identities.append(identity)
            yield idx, previous, current, scale_factor
            previous = current

    def _compute_pair_transform(pair):
        """ECC transform of frame idx relative to frame idx-1 (scale-restored)."""
        idx, prev_gray, curr_gray, scale_factor = pair

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

    if max_workers == 1:
        pair_transforms = []
        profile = _profile()
        for pair in pairs():
            _check_cancel(should_cancel)
            if profile.enabled:
                with profile.stage('registration.estimate'):
                    transform = _compute_pair_transform(pair)
            else:
                transform = _compute_pair_transform(pair)
            _check_cancel(should_cancel)
            pair_transforms.append(transform)
    else:
        with _bounded_registration_map(_compute_pair_transform, pairs(), max_workers - 1,
                                       should_cancel, stage='registration.estimate',
                                       max_in_flight=max_workers) as results:
            pair_transforms = list(results)

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

    if not HAS_CUPY:
        aligned_images = _apply_perspective_warps(
            images, H_matrices, T_crop if do_crop else None, (target_w, target_h),
            max_workers, should_cancel, defer_warps, method='ecc', sources=source_identities,
            bounds=(top, bottom, left, right), plan_callback=plan_callback)
        _save_registration(aligned_images, output_path, img_filenames, should_cancel)
        return aligned_images

    def warp_task(args):
        from core.source_frames import resolve_frame
        img, H = args
        img = resolve_frame(img, should_cancel=should_cancel)
        H_final = T_crop @ H if do_crop else H
        img_gpu = cp.asarray(img)
        y_grid, x_grid = cp.meshgrid(cp.arange(target_h), cp.arange(target_w), indexing='ij')
        coords = cp.stack([x_grid, y_grid, cp.ones_like(x_grid)]).reshape(3, -1)
        # Preserve the explicitly selected legacy GPU linear interpolation.
        src_coords = cp.asarray(np.linalg.inv(H_final)) @ coords
        divisor = cp.where(cp.abs(src_coords[2]) < 1e-10, 1e-10, src_coords[2])
        src_x = (src_coords[0] / divisor).reshape(target_h, target_w)
        src_y = (src_coords[1] / divisor).reshape(target_h, target_w)
        planes = ((img_gpu,) if img.ndim == 2 else
                  (img_gpu[:, :, c] for c in range(img.shape[2])))
        channels = [cupyx.scipy.ndimage.map_coordinates(
            plane, cp.stack([src_y, src_x]), order=1, mode='constant', cval=0)
            for plane in planes]
        result = channels[0] if len(channels) == 1 else cp.stack(channels, axis=2)
        return cp.asnumpy(result).astype(img.dtype)

    # The explicit legacy GPU backend is serialized; never mix CPU/GPU frames.
    with _bounded_registration_map(warp_task, zip(images, H_matrices), 1,
                                   should_cancel, stage='registration.warp') as results:
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
        self._perspective_stages = []
    
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
            self._perspective_stages = []
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
            deferred = self.defer_warps and not self.use_gpu
            if isinstance(input_source, str):
                extensions = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png',
                              '.bmp', '.tif', '.tiff', '.webp'}
                paths = sorted((os.path.join(input_source, name)
                                for name in os.listdir(input_source)
                                if os.path.splitext(name)[1].lower() in extensions),
                               key=_image_sort_key)
                from core.image_loader import admit_image_decode, decode_image
                if deferred:
                    from core.source_frames import scan_source, process_frame
                    images = []
                    for path in paths:
                        _check_cancel(should_cancel)
                        for source in scan_source(path, header_only=True, page_index=-1,
                                                  should_cancel=should_cancel):
                            # Match decode_image's directory contract: validate
                            # alpha before retaining an opaque BGR producer.
                            if source.ndim == 3 and source.shape[2] == 4:
                                from core.source_frames import resolve_frame
                                from core.image_loader import _has_transparency, UnsupportedTransparency
                                if _has_transparency(resolve_frame(source, should_cancel=should_cancel)):
                                    raise UnsupportedTransparency('Registration of transparent images is unsupported')
                                _check_cancel(should_cancel)
                                source = process_frame(source,
                                    lambda pixels: cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR),
                                    shape=source.shape[:2] + (3,), should_cancel=should_cancel)
                            images.append(source)
                else:
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
                        image = decode_image(path)
                        if image is None:
                            raise ValueError(f"Could not decode registration frame: {path}")
                        images.append(store_frame(image))
                filenames = [os.path.basename(path) for path in paths]
            else:
                from collections.abc import Sequence
                from core.source_frames import materialize_frames
                if deferred:
                    images = list(input_source) if input_source is not None else []
                else:
                    staging = (estimate_registration_space(input_source, len(self.steps))
                               if isinstance(input_source, Sequence) else 0)
                    images = (materialize_frames(input_source, should_cancel,
                                                 preferred_staging_bytes=staging,
                                                 thread_count=thread_count,
                                                 resource_preference=resource_preference)
                              if input_source is not None else [])
            if deferred and images and self.steps:
                from core.source_frames import select_frame_scratch
                select_frame_scratch(images,
                    preferred_size=estimate_registration_space(images[:2], len(self.steps)),
                    should_cancel=should_cancel)
            if images is not None and len(images) and 'openfocus' in self.steps and self.parameters['reference_frame'] > len(images):
                raise ValueError(f"OpenFocus reference frame {self.parameters['reference_frame']} is outside the stack of {len(images)} frames")

            from core.image_loader import _has_transparency, UnsupportedTransparency
            for image in images or ():
                _check_cancel(should_cancel)
                from core.source_frames import resolve_frame
                if image.ndim == 3 and image.shape[2] == 4 and _has_transparency(resolve_frame(image, should_cancel=should_cancel)):
                    raise UnsupportedTransparency("Registration of transparent images is unsupported")

            if images is None or len(images) < 2:
                _check_cancel(should_cancel)
                if images:
                    for step in self.steps:
                        _report_reference(step)
                    _save_registration(images, output_path, filenames, should_cancel)
                return images if images is not None else []
            if not self.steps:
                _save_registration(images, output_path, filenames, should_cancel)
                return images
            height, width = images[0].shape[:2]
            channels = images[0].shape[2] if images[0].ndim == 3 else 1
            # Charge active source resolution and bounded concurrent estimation;
            # deferred output descriptors retain no full-resolution warp stack.
            gray_bytes = height * width * 4
            from core.source_frames import frame_resolution_bytes
            plan = plan_execution(Workload(
                algorithm='registration', frames=len(images), height=height,
                width=width, channels=channels, itemsize=images[0].dtype.itemsize,
                fixed_bytes=gray_bytes,
                per_worker_bytes=max(image.size * 16 + frame_resolution_bytes(image)
                                     for image in images),
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
                if deferred and step not in ('homography', 'ecc', 'openfocus'):
                    # Extensions retain their existing array input contract and
                    # receive only their declared parameters/execution keywords.
                    from core.source_frames import materialize_frames
                    result = materialize_frames(result, should_cancel,
                        thread_count=thread_count, resource_preference=resource_preference)
                parameters = {item.key: self.parameters[item.key] for item in spec.parameters}
                with log_operation('registration_stage', method=step,
                                   parameters=parameters, provider='OpenCV',
                                   estimation_device='CPU', frames=len(result),
                                   shape=result[0].shape) as operation, \
                        _profile().stage(f'step.{step}'):
                    try:
                        result = resolve_runner(spec.runner)(self, result,
                            thread_count=workers, should_cancel=should_cancel, **parameters)
                        _check_cancel(should_cancel)
                    except RegistrationCancelled:
                        operation.cancel()
                        raise
            if len(self._perspective_stages) == len(self.steps):
                self.alignment_plan = PerspectiveAlignmentPlan(tuple(self._perspective_stages),
                    json.dumps(self.parameters, sort_keys=True, separators=(',', ':'), allow_nan=False),
                    self.downscale_width)
            else:
                self.alignment_plan = None
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
        return _align_homography_impl(input_source, output_path, downscale_width=self.downscale_width, thread_count=thread_count, should_cancel=should_cancel, failure_policy=failure_policy, defer_warps=self.defer_warps and not self.use_gpu, plan_callback=self._perspective_stages.append)
    
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
        return _align_ecc_impl(input_source, output_path, downscale_width=self.downscale_width, thread_count=thread_count, should_cancel=should_cancel, use_gpu=self.use_gpu, failure_policy=failure_policy, defer_warps=self.defer_warps, plan_callback=self._perspective_stages.append)


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
