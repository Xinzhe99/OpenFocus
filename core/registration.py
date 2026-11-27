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
from collections import OrderedDict, deque
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import mmap
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
from core.registration_errors import RejectedAlignmentFrame
from core.frame_storage import capture_scratch, create_frame, run_with_scratch, store_frame
from utils.logging_setup import log_action, log_operation
from utils.performance_profile import current as _profile

_ORIGINAL_FIND_TRANSFORM_ECC = cv2.findTransformECC
_ecc_native_lock = threading.Lock()
_ecc_native_solver = None
_ecc_native_loaded = False


def _load_ecc_native():
    """Load the native CPU homography ECC solver once, falling back cleanly if unavailable."""
    global _ecc_native_solver, _ecc_native_loaded
    if _ecc_native_loaded:
        return _ecc_native_solver
    with _ecc_native_lock:
        if not _ecc_native_loaded:
            try:
                from core._ecc_native import find_transform_ecc_homography
                _ecc_native_solver = find_transform_ecc_homography
            except Exception:
                _ecc_native_solver = None
            _ecc_native_loaded = True
    return _ecc_native_solver



# ========== 基于变换矩阵的精确裁切函数 ==========

from contextvars import ContextVar

# Grouped fusion records crop offsets in the reference image's pixel grid.
grouped_crop_offsets = ContextVar('grouped_crop_offsets', default=None)

# Optional per-frame diagnostics, collected on the calling thread.
alignment_report = ContextVar('alignment_report', default=None)

# OpenCV's thread setting is process-wide. All temporary exchanges share this
# lock; the owner budget is local to the source-producing context.
_warp_threads_lock = threading.Lock()
_deferred_warp_threads = ContextVar('deferred_warp_threads', default=None)
_deferred_ecc_gpu = ContextVar('deferred_ecc_gpu', default=None)

_deferred_alignment_timers_lock = threading.Lock()
_deferred_alignment_timers = []


class DeferredAlignmentTimer:
    """Measure wall-clock time spent inside deferred alignment warps.

    Uses active-depth reference counting so concurrent producer threads and
    nested warp stages accumulate the wall-clock union of warp intervals
    without double-counting or exceeding the enclosing stage's elapsed time.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self._started = 0.0
        self._elapsed = 0.0
        self._warps = 0

    def _enter(self):
        with self._lock:
            if self._active == 0:
                self._started = time.monotonic()
            self._active += 1
            self._warps += 1

    def _exit(self):
        with self._lock:
            self._active -= 1
            if self._active == 0:
                self._elapsed += max(0.0, time.monotonic() - self._started)

    @property
    def elapsed(self) -> float:
        with self._lock:
            if self._active > 0:
                return self._elapsed + max(0.0, time.monotonic() - self._started)
            return self._elapsed

    @property
    def warps(self) -> int:
        with self._lock:
            return self._warps


@contextmanager
def deferred_alignment_timer():
    """Observe wall-clock time spent applying deferred alignment transforms."""
    timer = DeferredAlignmentTimer()
    with _deferred_alignment_timers_lock:
        _deferred_alignment_timers.append(timer)
    try:
        yield timer
    finally:
        with _deferred_alignment_timers_lock:
            try:
                _deferred_alignment_timers.remove(timer)
            except ValueError:
                pass


@contextmanager
def measure_deferred_alignment():
    """Attribute this block's wall-clock span to all active alignment timers."""
    if not _deferred_alignment_timers:
        yield
        return
    with _deferred_alignment_timers_lock:
        timers = tuple(_deferred_alignment_timers)
    if not timers:
        yield
        return
    for timer in timers:
        timer._enter()
    try:
        yield
    finally:
        for timer in timers:
            timer._exit()


@contextmanager
def deferred_warp_threads(threads):
    token = _deferred_warp_threads.set(threads)
    try:
        yield
    finally:
        _deferred_warp_threads.reset(token)


@contextmanager
def deferred_ecc_warp_accelerator(enabled):
    """Share one admitted OpenCL workspace across a prepared stack."""
    backend = None
    if enabled:
        from core._perspective_gpu import ExactPerspectiveGpu

        backend = ExactPerspectiveGpu()
    token = _deferred_ecc_gpu.set(backend)
    try:
        yield backend
    finally:
        _deferred_ecc_gpu.reset(token)
        if backend is not None:
            backend.close()


def carry_warp_accelerator(function):
    """Run ``function`` on a pool thread with the submitting thread's warp accelerator."""
    backend = _deferred_ecc_gpu.get()
    if backend is None:
        return function

    def carried(*args, **kwargs):
        token = _deferred_ecc_gpu.set(backend)
        try:
            return function(*args, **kwargs)
        finally:
            _deferred_ecc_gpu.reset(token)
    return carried


def exchange_opencv_threads(threads):
    with _warp_threads_lock:
        previous = cv2.getNumThreads()
        cv2.setNumThreads(threads)
        return previous


@contextmanager
def _serial_ecc_native_threads():
    from threadpoolctl import threadpool_limits

    # Zero disables OpenCV parallel regions on every backend. Limit BLAS for
    # the native solver's small linear systems until all pair workers join.
    previous = exchange_opencv_threads(0)
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            yield
    finally:
        exchange_opencv_threads(0 if previous == 1 else previous)


# OpenCV parallel backends whose setNumThreads(N) caps a parallel region at N
# workers. GCD only switches parallelism on or off, and HPX and WinRT are not
# known to cap it, so those keep warps serial. The names are those reported by
# currentParallelFramework() and the build summary's "Parallel framework".
_BOUNDED_PARALLEL_BACKENDS = ('TBB', 'ONETBB', 'OPENMP', 'PTHREADS',
                              'CONCURRENCY', 'MS-CONCURRENCY')
# The only parallel plugins OpenCV registers itself; each caps its workers.
_BOUNDED_PARALLEL_PLUGINS = ('TBB', 'ONETBB', 'OPENMP')


def _bounded_backend_name(name):
    if not isinstance(name, str) or not name.strip():
        return False
    return name.strip().split()[0].upper() in _BOUNDED_PARALLEL_BACKENDS


def _active_backend_bounds_threads():
    """True when OpenCV's active parallel backend caps a call's workers.

    Bindings that expose currentParallelFramework() answer directly. The
    published Python packages do not, so the compiled framework is read from
    the build summary. A runtime plugin can replace it, but OpenCV registers
    only oneTBB, TBB and OpenMP plugins, which all cap their workers; a plugin
    named through the environment that is not one of those keeps warps serial.
    """
    current = getattr(cv2, 'currentParallelFramework', None)
    if current is not None:
        return _bounded_backend_name(current())
    for variable in ('OPENCV_PARALLEL_BACKEND', 'OPENCV_PARALLEL_PRIORITY_LIST'):
        for name in os.environ.get(variable, '').replace(';', ',').split(','):
            if name.strip() and name.strip().upper() not in _BOUNDED_PARALLEL_PLUGINS:
                return False
    for line in cv2.getBuildInformation().splitlines():
        label, _, value = line.partition(':')
        if label.strip() == 'Parallel framework':
            return _bounded_backend_name(value)
    return False


def _warp_thread_budget():
    """Threads lent to a warp on this thread; one outside a lending scope."""
    try:
        threads = _deferred_warp_threads.get()
        threads = threads() if callable(threads) else threads
        if threads is None:
            return 1
        threads = max(1, int(threads))
        if threads == 1 or not _active_backend_bounds_threads():
            return 1
        return threads
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



_INTERPOLATION_FLAGS = {
    'lanczos': cv2.INTER_LANCZOS4,
    'cubic': cv2.INTER_CUBIC,
    'linear': cv2.INTER_LINEAR,
}


def interpolation_flag(interpolation: str) -> int:
    """Resolve OpenFocus interpolation name to the corresponding OpenCV flag."""
    key = 'lanczos' if interpolation is None else str(interpolation).lower()
    if key not in _INTERPOLATION_FLAGS:
        raise ValueError(f"Unsupported interpolation: {interpolation!r} (expected 'lanczos', 'cubic', or 'linear')")
    return _INTERPOLATION_FLAGS[key]


def normalize_alignment_roi(alignment_roi, img_shape):
    """Validate and clamp an (x, y, w, h) alignment detection area to source dimensions."""
    if alignment_roi is None:
        return None
    if len(alignment_roi) != 4:
        raise ValueError('alignment_roi must be a 4-tuple (x, y, w, h)')
    h, w = int(img_shape[0]), int(img_shape[1])
    rx, ry, rw, rh = (int(round(float(v))) for v in alignment_roi)
    rx = max(0, min(rx, max(0, w - 1)))
    ry = max(0, min(ry, max(0, h - 1)))
    rw = max(0, min(rw, w - rx))
    rh = max(0, min(rh, h - ry))
    if rw < 8 or rh < 8:
        raise ValueError('alignment_roi must be at least 8x8 pixels within the source image')
    return (rx, ry, rw, rh)


def conjugate_roi_matrix(matrix, rx: int, ry: int):
    """Translate a 3x3 transform estimated in ROI coordinates back to full-frame coordinates.

    Computes ``T(rx, ry) @ M_roi @ T(-rx, -ry)`` in float64 precision.
    """
    m64 = np.asarray(matrix, dtype=np.float64)
    if m64.shape == (2, 3):
        m64 = np.vstack([m64, np.array([[0.0, 0.0, 1.0]], dtype=np.float64)])
        is_affine = True
    else:
        is_affine = False
    t_pos = np.array([[1.0, 0.0, float(rx)], [0.0, 1.0, float(ry)], [0.0, 0.0, 1.0]], dtype=np.float64)
    t_neg = np.array([[1.0, 0.0, float(-rx)], [0.0, 1.0, float(-ry)], [0.0, 0.0, 1.0]], dtype=np.float64)
    full = t_pos @ m64 @ t_neg
    if abs(full[2, 2]) > 1e-12:
        full = full / full[2, 2]
    out = full[:2, :] if is_affine else full
    return out.astype(np.asarray(matrix).dtype, copy=False)


def validate_transform_bounds(
    matrix,
    img_shape,
    *,
    max_shift_x_ratio=None,
    max_shift_y_ratio=None,
    max_rotation_deg=None,
    max_scale_change=None,
    center=None,
):
    """Return a failure reason string if a 3x3 transform violates geometric bounds, else None."""
    if (max_shift_x_ratio is None and max_shift_y_ratio is None
            and max_rotation_deg is None and max_scale_change is None):
        return None
    h, w = float(img_shape[0]), float(img_shape[1])
    cx, cy = (w / 2.0, h / 2.0) if center is None else (float(center[0]), float(center[1]))
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(m)) or abs(m[2, 2]) < 1e-12:
        return 'motion outside limits'
    m = m / m[2, 2]
    p = m @ np.array([cx, cy, 1.0], dtype=np.float64)
    if not np.all(np.isfinite(p)) or p[2] <= 1e-6:
        return 'motion outside limits'
    px, py = p[0] / p[2], p[1] / p[2]
    dx = px - cx
    dy = py - cy

    # Local 2x2 Jacobian of projective map (H0.p / H2.p, H1.p / H2.p) at (cx, cy).
    inv_w = 1.0 / p[2]
    j00 = (m[0, 0] - px * m[2, 0]) * inv_w
    j01 = (m[0, 1] - px * m[2, 1]) * inv_w
    j10 = (m[1, 0] - py * m[2, 0]) * inv_w
    j11 = (m[1, 1] - py * m[2, 1]) * inv_w
    jac = np.array([[j00, j01], [j10, j11]], dtype=np.float64)
    if not np.all(np.isfinite(jac)) or np.linalg.det(jac) <= 1e-6:
        return 'motion outside limits'

    if max_shift_x_ratio is not None:
        rx = float(max_shift_x_ratio)
        tol_x = 0.25 if rx == 0.0 else 1e-4
        if abs(dx) > w * rx + tol_x:
            return 'horizontal shift outside limits'
    if max_shift_y_ratio is not None:
        ry = float(max_shift_y_ratio)
        tol_y = 0.25 if ry == 0.0 else 1e-4
        if abs(dy) > h * ry + tol_y:
            return 'vertical shift outside limits'

    u, s, vt = np.linalg.svd(jac)
    rot_mat = u @ vt
    if np.linalg.det(rot_mat) < 0:
        u[:, -1] *= -1
        rot_mat = u @ vt
    rot_deg = abs(math.degrees(math.atan2(rot_mat[1, 0], rot_mat[0, 0])))
    scale_change = max(abs(float(s[0]) - 1.0), abs(float(s[1]) - 1.0))

    if max_rotation_deg is not None:
        rlim = float(max_rotation_deg)
        tol_rot = 0.05 if rlim == 0.0 else 1e-4
        if rot_deg > rlim + tol_rot:
            return 'rotation outside limits'
    if max_scale_change is not None:
        slim = float(max_scale_change)
        tol_scale = 0.002 if slim == 0.0 else 1e-4
        if scale_change > slim + tol_scale:
            return 'scale change outside limits'
    return None


def _registration_gray(image, *, is_rgb=False):
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    from core.openfocus_alignment import _is_bgr_rev

    if _is_bgr_rev(image):
        image = image[..., ::-1]
        is_rgb = True
    if image.dtype == np.float64:
        image = image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY)


# ========== 单应性对齐算法实现（非线性） ==========


def _align_homography_impl(input_source, output_path=None, img_filenames=None, downscale_width=1600, thread_count: int = 4, should_cancel=None, failure_policy='continue', *, defer_warps=False, plan_callback=None, dispatch_callback=None, accelerate_warp=True, alignment_roi=None, max_shift_x_ratio=None, max_shift_y_ratio=None, max_rotation_deg=None, max_scale_change=None, auto_crop=True, interpolation='lanczos'):
    """
    商业级图像对齐算法优化版
    特性：
    1. 传递对齐 (Sequential Alignment)：解决大景深下的特征丢失问题
    2. 金字塔加速 (Downscale Processing)：大幅提升特征检测速度
    3. 矩阵累积 (Matrix Chaining)：减少累积误差
    4. 可选插值（Lanczos / Bicubic / Bilinear）：兼顾画质与速度
    5. 并行计算优化 (Parallel Processing)：利用多核加速特征提取和图像变换
    """
    # Directory loading and source policy belong to ImageRegistration.process.
    interpolation_flag(interpolation)
    images = input_source

    num_images = len(images)
    if num_images:
        _report_reference('homography')
    if num_images < 2:
        return images

    # --- 初始化 ---
    h_orig, w_orig = images[0].shape[:2]
    roi = normalize_alignment_roi(alignment_roi, (h_orig, w_orig))
    roi_center = ((roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0) if roi is not None else None)
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

    from core.frame_resolution_cache import within_retention_limit
    large_stack = not within_retention_limit(images)

    def get_features_task(img):
        from core.source_frames import resolve_frame, without_source_cache_growth
        from core.openfocus_alignment import _can_use_file_digest
        source = img
        use_file_digest = _can_use_file_digest(source)
        identity = _perspective_source_file(source, should_cancel) if use_file_digest else None
        cache_scope = without_source_cache_growth() if large_stack else nullcontext()
        with cache_scope:
            img = resolve_frame(img, should_cancel=should_cancel)
        if identity is None:
            identity = _perspective_source(source, img, should_cancel)
        if roi is not None:
            rx, ry, rw, rh = roi
            img = img[ry:ry + rh, rx:rx + rw]
        # 在线程中创建独立的检测器，确保线程安全
        local_detector = cv2.SIFT_create()
        h, w = img.shape[:2]
        scale = downscale_width / float(w) if w > downscale_width else 1.0
        if scale < 1.0:
            # 使用 INTER_LINEAR 速度更快，对于特征检测通常足够
            img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            # Release full-resolution source pixels before quantization and SIFT.
            del img
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
        # SIFT converts to gray after resize and byte quantization.
        kps, des = local_detector.detectAndCompute(_registration_gray(img_small), None)
        return kps, des, scale, identity

    # 使用线程池并行提取特征
    # OpenCV 的大部分操作释放 GIL，因此多线程可以有效加速
    try:
        max_workers = max(1, int(thread_count))
    except Exception:
        max_workers = min(8, os.cpu_count() or 1)

    # Matching retains its last accepted reference, including failed frames.
    bf = cv2.BFMatcher(cv2.NORM_L2)
    flann_index_params = {'algorithm': 1, 'trees': 4}
    flann_search_params = {'checks': 64}
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
                if failure_policy == 'fail':
                    _report_alignment(idx, 'homography', 'failed', reason='insufficient features', reference_index=last_index)
                    raise RejectedAlignmentFrame('Homography', idx, 'insufficient features')
                print(f"Warning: Frame {idx} features insufficient. Keeping original position.")
                _report_alignment(idx, 'homography', 'fallback', reason='insufficient features',
                                  reference_index=last_index, transform=H_global)
                H_matrices.append(H_global.copy())
                continue

            if len(curr_des) < 2 or len(last_des) < 2:
                matches = ()
            elif len(curr_des) * len(last_des) > 400_000_000:
                try:
                    matcher = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
                    matches = matcher.knnMatch(curr_des, last_des, k=2)
                except cv2.error:
                    matches = bf.knnMatch(curr_des, last_des, k=2)
            else:
                matches = bf.knnMatch(curr_des, last_des, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.70 * n.distance:
                        good_matches.append(m)
            if len(good_matches) < 6:
                if failure_policy == 'fail':
                    _report_alignment(idx, 'homography', 'failed', reason='insufficient matches', reference_index=last_index, match_count=len(good_matches))
                    raise RejectedAlignmentFrame('Homography', idx, 'insufficient matches')
                print(f"Warning: Frame {idx} poor matches ({len(good_matches)}). Keeping previous trajectory.")
                _report_alignment(idx, 'homography', 'fallback', reason='insufficient matches',
                                  reference_index=last_index, transform=H_global,
                                  match_count=len(good_matches))
                H_matrices.append(H_global.copy())
                continue

            pts_curr = np.float32([curr_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / curr_scale
            pts_last = np.float32([last_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) / last_scale
            H_local, mask = cv2.findHomography(pts_curr, pts_last, cv2.RANSAC, 5.0)
            _check_cancel(should_cancel)
            if H_local is not None and roi is not None:
                H_local = conjugate_roi_matrix(H_local, roi[0], roi[1])
            bound_reason = (
                validate_transform_bounds(
                    H_local, (h_orig, w_orig),
                    max_shift_x_ratio=max_shift_x_ratio,
                    max_shift_y_ratio=max_shift_y_ratio,
                    max_rotation_deg=max_rotation_deg,
                    max_scale_change=max_scale_change,
                    center=roi_center,
                )
                if H_local is not None else 'homography estimation failed'
            )
            estimated = bound_reason is None
            if not estimated:
                print(f"Frame {idx} alignment rejected ({bound_reason}).")
                if failure_policy == 'fail':
                    _report_alignment(idx, 'homography', 'failed', reason=bound_reason, reference_index=last_index, match_count=len(good_matches))
                    raise RejectedAlignmentFrame('Homography', idx, bound_reason)
                _report_alignment(idx, 'homography', 'fallback', reason=bound_reason,
                                  reference_index=last_index, transform=H_global,
                                  match_count=len(good_matches))
                trajectory_failed = True
                H_local = np.eye(3, dtype=np.float32)
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
    
    if auto_crop:
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
    else:
        top, bottom, left, right = 0, h_orig, 0, w_orig
        _record_grouped_crop(top, bottom, left, right)
        do_crop = False
        target_w, target_h = w_orig, h_orig
        offset_x, offset_y = 0, 0

    # 构造裁切平移矩阵
    T_crop = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    with deferred_ecc_warp_accelerator(accelerate_warp and interpolation == 'lanczos' and
                                       (not defer_warps or output_path)) as warp:
        aligned_images = _apply_perspective_warps(
            images, H_matrices, T_crop if do_crop else None, (target_w, target_h),
            max_workers, should_cancel, defer_warps, method='homography', sources=source_identities,
            bounds=(top, bottom, left, right), plan_callback=plan_callback,
            interpolation=interpolation)
        _save_registration(aligned_images, output_path, img_filenames, should_cancel)
    warp_device = 'CPU'
    if warp is not None and warp.gpu_warps:
        warp_device = ('CPU / ' if warp.cpu_warps else '') + warp.device_name
    if dispatch_callback is not None:
        dispatch_callback('homography', estimation='CPU', warp=warp_device)

    return aligned_images


class RegistrationCancelled(Exception):
    """Raised when a render cancellation flag fires mid-registration."""

def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise RegistrationCancelled()


def _perspective_source_file(frame, should_cancel, *, sha256=None):
    """Build a file-backed SourceFingerprint without decoding full-resolution pixels."""
    from core.openfocus_alignment import (SourceFingerprint, _file_digest,
                                          _source_context, _source_file_identity)
    path, page, stamp = _source_file_identity(frame)
    dtype_str = np.dtype(frame.dtype).str
    shape = tuple(frame.shape)
    if sha256 is None:
        sha256 = _file_digest(path, should_cancel)
    source = SourceFingerprint(shape, dtype_str, sha256, path, page,
                               stamp, _source_context(frame), digest_kind='file')
    source.validate_file()
    return source


@dataclass(frozen=True)
class _JpegEccPreparation:
    size: tuple[int, int]
    scale: float
    flags: int
    encoded_bytes: int
    workspace_bytes: int


def _plan_jpeg_ecc(source, downscale_width, *, metadata_only=False):
    """Plan direct grayscale decoding for a full-size JPEG source."""
    from core.source_frames import SourceFrame, _jpeg_full_buffer_bytes

    if (type(source) is not SourceFrame or source.transforms or source.linear_metadata is not None
            or source.scale != 1.0 or not source.path or source.dtype != np.uint8
            or os.path.splitext(source.path)[1].lower() not in ('.jpg', '.jpeg', '.jpe', '.jfif')):
        return None
    height, width, scale = _ecc_resize_geometry(source.shape, downscale_width)
    if scale >= 1.0 or min(height, width) < 1:
        return None
    rotated = source.orientation in (5, 6, 7, 8)
    geometry = source.shape[:2][::-1] if rotated else source.shape[:2]
    if metadata_only:
        bands = source._jpeg_components
        if bands not in (1, 3):
            return None
        raw_height, raw_width = geometry
    else:
        from PIL import Image

        try:
            with Image.open(source.path) as header:
                if header.format != 'JPEG' or header.mode not in ('L', 'RGB'):
                    return None
                raw_width, raw_height = header.size
                bands = len(header.getbands())
        except (OSError, Image.DecompressionBombError):
            return None
        if (raw_height, raw_width) != geometry:
            return None
    requested_width, requested_height = (height, width) if rotated else (width, height)
    for divisor, flag in ((8, cv2.IMREAD_REDUCED_GRAYSCALE_8),
                          (4, cv2.IMREAD_REDUCED_GRAYSCALE_4),
                          (2, cv2.IMREAD_REDUCED_GRAYSCALE_2),
                          (1, cv2.IMREAD_GRAYSCALE)):
        decoded_width = (raw_width + divisor - 1) // divisor
        decoded_height = (raw_height + divisor - 1) // divisor
        if decoded_width >= requested_width and decoded_height >= requested_height:
            break
    # A reduced output still needs native-size coefficients for multiple scans.
    coefficients = (0 if source._jpeg_single_scan else
                    _jpeg_full_buffer_bytes(raw_width, raw_height, bands))
    encoded = source.fingerprint[0]
    workspace = (encoded + coefficients + 2 * decoded_width * decoded_height
                 + 2 * width * height + 128 * raw_width + 128 * 1024)
    return _JpegEccPreparation((width, height), scale, flag, encoded, workspace)


def _decode_jpeg_ecc(source, plan, should_cancel):
    from core.source_frames import _admit_source_decode, _orient

    source.validate_source()
    with _admit_source_decode(plan.workspace_bytes, should_cancel):
        with open(source.path, 'rb') as stream:
            encoded = stream.read(plan.encoded_bytes + 1)
        if len(encoded) != plan.encoded_bytes:
            raise OSError(f'Source changed since it was opened; reopen the stack: {source.path}')
        _check_cancel(should_cancel)
        # Bind the fingerprint to the same encoded bytes used for estimation.
        identity = _perspective_source_file(source, should_cancel,
                                            sha256=hashlib.sha256(encoded).hexdigest())
        _check_cancel(should_cancel)
        gray = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8),
                            plan.flags | cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_ANYDEPTH)
        del encoded
        if gray is None:
            raise ValueError(f'Could not decode source: {source.path}')
        if source.orientation != 1:
            gray = _orient(gray, source.orientation)
        if gray.shape[::-1] != plan.size:
            gray = cv2.resize(gray, plan.size, interpolation=cv2.INTER_AREA)
        _check_cancel(should_cancel)
    return gray, identity


def _perspective_source(frame, image, should_cancel):
    from core.openfocus_alignment import (SourceFingerprint, _source_file_identity,
                                    _pixel_digest, _source_context)
    path, page, stamp = _source_file_identity(frame)
    source = SourceFingerprint(tuple(image.shape), image.dtype.str,
                              _pixel_digest(image, should_cancel, source=frame), path, page,
                              stamp, _source_context(frame))
    source.validate_file()
    return source


def _merge_cancel_callbacks(primary, secondary):
    if primary is None:
        return secondary
    if secondary is None or secondary is primary:
        return primary
    return lambda: bool(primary() or secondary())


def _perspective_near_lanczos_boundary(inverse, x, y, width, height, sx0, sy0,
                                       min_denom, should_cancel):
    from core import _alignment_kernels

    x1, y1 = x + width, y + height
    magnitude = np.abs(inverse) @ (x1 - 1, y1 - 1, 1)
    # Bound both evaluations of the rational coordinates, including matrix
    # rebasing and OpenCV's block/FMA evaluation order. The sums of absolute
    # terms also cover cancellation in the numerator or denominator.
    eps = 64 * np.finfo(np.float64).eps
    denom = min_denom - eps * magnitude[2]
    if denom <= 0:
        return True
    numerator = max(magnitude[0] + sx0 * magnitude[2],
                    magnitude[1] + sy0 * magnitude[2])
    bins = float(cv2.INTER_TAB_SIZE)
    tolerance = bins * eps * numerator / denom * (1 + magnitude[2] / denom)
    if not math.isfinite(tolerance) or tolerance >= 0.25:
        return True

    if (inverse[2, 0] == inverse[2, 1] == inverse[0, 1] == inverse[1, 0] == 0
            and inverse[0, 0] == inverse[1, 1] == inverse[2, 2]):
        # Integer destination steps leave a translation's fraction unchanged.
        x, y, x1, y1 = 0, 0, 1, 1
    elif _alignment_kernels.available():
        for start in range(y, y1, 128):
            _check_cancel(should_cancel)
            if _alignment_kernels.perspective_has_rounding_edge(
                    inverse, x, x1, start, min(start + 128, y1), bins, tolerance):
                return True
        return False

    # Optional-Numba fallback needs only row-sized workspaces.
    from core.resource_policy import require_memory
    require_memory(6 * (x1 - x) * np.dtype(np.float64).itemsize)
    xs = np.arange(x, x1, dtype=np.float64)
    ax, ay, aw = inverse[:, 0, None] * xs
    for row in range(y, y1):
        _check_cancel(should_cancel)
        shift = inverse[:, 1] * row + inverse[:, 2]
        scale = bins / (aw + shift[2])
        for values, offset in ((ax, shift[0]), (ay, shift[1])):
            coords = (values + offset) * scale
            if np.any(np.abs(coords - np.floor(coords) - 0.5) <= tolerance):
                return True
    return False


@dataclass(frozen=True)
class PerspectiveWarp:
    source: object
    matrix: tuple
    matrix_dtype: str
    dsize: tuple
    should_cancel: object = None
    method: str = ''
    interpolation: str = 'lanczos'

    def _is_integer_crop(self, src_shape, src_dtype):
        a, b, tx, c, d, ty, p, q, r = self.matrix
        width, height = self.dsize
        return (
            a == d == r == 1 and b == c == p == q == 0
            and src_dtype in (np.uint8, np.uint16)
            and (len(src_shape) == 2 or (len(src_shape) == 3 and 1 <= src_shape[2] <= 4))
            and max(src_shape[0], src_shape[1], width, height) < 32767
            and 0 <= -tx <= src_shape[1] - width
            and 0 <= -ty <= src_shape[0] - height
            and int(tx) == tx and int(ty) == ty
        )

    def _run_cv_warp(self, src, M, dsize, flags, threads, *, destination=None):
        args = (src, M, dsize) if destination is None else (src, M, dsize, destination)
        if threads <= 1:
            return cv2.warpPerspective(*args, flags=flags, borderMode=cv2.BORDER_CONSTANT)
        with _warp_threads_lock:
            previous = cv2.getNumThreads()
            cv2.setNumThreads(threads)
            try:
                return cv2.warpPerspective(*args, flags=flags, borderMode=cv2.BORDER_CONSTANT)
            finally:
                cv2.setNumThreads(0 if previous == 1 else previous)

    def _apply_to_image(self, image, *, should_cancel=None, destination=None):
        # At an integer sampling location Lanczos returns the source code
        # exactly for integer pixels. Keep the allocating/native path for
        # floats, where zero-weight taps can propagate NaNs or affect -0.
        from core import _alignment_kernels as _native_kernels
        from core.openfocus_alignment import _is_bgr_rev, _run_row_bands
        cancel_fn = self.should_cancel if should_cancel is None else should_cancel
        _check_cancel(cancel_fn)
        width, height = self.dsize
        if self._is_integer_crop(image.shape, image.dtype):
            _, _, tx, _, _, ty, _, _, _ = self.matrix
            left, top = -int(tx), -int(ty)
            crop = image[top:top + height, left:left + width]
            if crop.shape[2:] == (1,):
                crop = crop[:, :, 0]
            if destination is not None:
                for start in range(0, height, 128):
                    _check_cancel(cancel_fn)
                    destination[start:start + 128] = crop[start:start + 128]
                return destination
            if (crop.ndim == 3 and crop.shape[2] == 3
                    and _is_bgr_rev(image) and _native_kernels.available()):
                out = np.empty((height, width, 3), dtype=image.dtype)
                swap_fn = (_native_kernels.swap_rgb_to_bgr_rows_u8 if image.dtype == np.uint8
                           else _native_kernels.swap_rgb_to_bgr_rows_u16)
                _run_row_bands(
                    height, 128, cancel_fn, swap_fn, crop[..., ::-1], out,
                    parallel_pixels=height * width, workers=_warp_thread_budget(),
                )
                return out
            # A resolved warp owns its pixels even without a render cache.
            return crop.copy(order='C')
        matrix = np.asarray(self.matrix, dtype=self.matrix_dtype).reshape(3, 3)
        backend = _deferred_ecc_gpu.get()
        if backend is not None and self.interpolation == 'lanczos':
            # Direct RGB TIFF frames arrive as a reversed view of contiguous storage.
            bgr_rev = _is_bgr_rev(image)
            warp_src = image[..., ::-1] if bgr_rev else image
            if destination is None:
                result = backend.warp(warp_src, matrix, self.dsize,
                                      cancel_fn, bgr_rev=bgr_rev, method=self.method)
            else:
                result = backend.warp(warp_src, matrix, self.dsize,
                                      cancel_fn, bgr_rev=bgr_rev, method=self.method,
                                      destination=destination)
            _check_cancel(cancel_fn)
            if result is not None:
                return result
        bgr_rev = _is_bgr_rev(image) and _native_kernels.available()
        src = image[..., ::-1] if bgr_rev else image
        threads = _warp_thread_budget()
        warped = self._run_cv_warp(src, matrix, self.dsize, interpolation_flag(self.interpolation), threads,
                                   destination=destination)
        _check_cancel(cancel_fn)
        if bgr_rev:
            swap_fn = (_native_kernels.swap_bgr_inplace_rows_u8 if warped.dtype == np.uint8
                       else _native_kernels.swap_bgr_inplace_rows_u16)
            _run_row_bands(
                warped.shape[0], 128, cancel_fn, swap_fn, warped,
                parallel_pixels=warped.shape[0] * warped.shape[1], workers=threads,
            )
        if backend is not None:
            backend.note_cpu_warp(self.method)
        return warped

    def __call__(self, image):
        with measure_deferred_alignment():
            self.source.validate_pixels(image, self.should_cancel)
            return self._apply_to_image(image)

    def call_with_source(self, source_frame, image, *, should_cancel=None):
        """Validate against the original source descriptor before applying the warp."""
        cancel_fn = _merge_cancel_callbacks(self.should_cancel, should_cancel)
        with measure_deferred_alignment():
            self.source.validate_pixels(image, cancel_fn, source=source_frame)
            return self._apply_to_image(image, should_cancel=cancel_fn)

    def _call_mapped_jpeg(self, source_frame, image, *, should_cancel=None):
        """Warp packed BGR pixels into an owned mapping with bounded CPU workspace."""
        from core.frame_storage import create_frame, frame_region
        from core.resource_policy import ResourceLimitError, require_memory

        cancel_fn = _merge_cancel_callbacks(self.should_cancel, should_cancel)
        with measure_deferred_alignment():
            integer_crop = self._is_integer_crop(image.shape, image.dtype)
            required = 128 * 1024 + (0 if integer_crop else _perspective_cpu_tile_bytes(self.dsize))
            try:
                require_memory(required)
            except ResourceLimitError:
                from core.source_frames import SOURCE_CACHE

                SOURCE_CACHE.clear()
                gc.collect()
                require_memory(required)
            with frame_region(image) as pixels:
                self.source.validate_pixels(pixels, cancel_fn, source=source_frame)
                output = create_frame(_perspective_output_shape(image.shape, self.dsize), image.dtype)
                # Close the written view before the root mapping enters the staged stack.
                with frame_region(output, write=True, writeback=True) as destination:
                    _check_cancel(cancel_fn)
                    warped = self._apply_to_image(pixels, should_cancel=cancel_fn, destination=destination)
                    if warped is not destination:
                        np.copyto(destination, warped)
            _check_cancel(cancel_fn)
            return output

    def _full_warp_crop(self, image, x, y, width, height, channels, should_cancel):
        from core import _alignment_kernels
        from core.openfocus_alignment import _is_bgr_rev
        from core.resource_policy import require_memory

        src_channels = image.shape[2] if image.ndim == 3 else 1
        packed = (0 if image.flags.c_contiguous or
                  (_is_bgr_rev(image) and _alignment_kernels.available()) else image.nbytes)
        full_bytes = math.prod(self.dsize) * src_channels * image.dtype.itemsize
        crop_bytes = width * height * (channels or src_channels) * image.dtype.itemsize
        require_memory(full_bytes + max(packed, crop_bytes))
        _check_cancel(should_cancel)
        full = self._apply_to_image(image, should_cancel=should_cancel)
        _check_cancel(should_cancel)
        if full.shape[:2] == (height, width) and (channels is None or full.ndim == 2):
            return full
        cropped = full[y:y + height, x:x + width]
        if channels is not None and cropped.ndim == 3:
            cropped = cropped[:, :, :channels]
        return cropped.copy(order='C')

    def warp_region(self, source_or_image, x, y, width, height, *,
                    channels=None, should_cancel=None):
        """Compute one destination rectangle bit-identical to full warp[y:y+h, x:x+w]."""
        from core.source_frames import is_direct_rgb_tiff, resolve_frame

        cancel_fn = _merge_cancel_callbacks(self.should_cancel, should_cancel)
        _check_cancel(cancel_fn)
        dsize_w, dsize_h = self.dsize
        if x < 0 or y < 0 or width < 0 or height < 0 or x + width > dsize_w or y + height > dsize_h:
            raise ValueError('Requested region does not fit the frame')

        direct_tiff = is_direct_rgb_tiff(source_or_image)
        image = None
        if not direct_tiff:
            image = (source_or_image if isinstance(source_or_image, np.ndarray)
                     else resolve_frame(source_or_image, should_cancel=cancel_fn))
        with measure_deferred_alignment():
            return self._warp_region_resolved(
                source_or_image, image, direct_tiff, x, y, width, height,
                channels=channels, cancel_fn=cancel_fn,
            )

    def _warp_region_resolved(self, source_or_image, image, direct_tiff, x, y, width, height, *,
                              channels=None, cancel_fn=None):
        from core import _alignment_kernels as _native_kernels
        from core.openfocus_alignment import _is_bgr_rev, _run_row_bands, _validate_direct_rgb_source
        from core.resource_policy import require_memory
        from core.source_frames import _read_direct_rgb_tiff_region, resolve_frame

        if direct_tiff:
            _validate_direct_rgb_source(self.source, source_or_image, cancel_fn)
        elif isinstance(source_or_image, np.ndarray):
            self.source.validate_pixels(image, cancel_fn)
        else:
            self.source.validate_pixels(image, cancel_fn, source=source_or_image)

        src_shape = tuple(source_or_image.shape)
        src_dtype = np.dtype(source_or_image.dtype)
        drop_singleton = len(src_shape) == 3 and src_shape[2] == 1
        out_ch = None if (len(src_shape) == 2 or drop_singleton) else (
            channels if channels is not None else src_shape[2])
        out_shape = (height, width) if out_ch is None else (height, width, out_ch)
        if width == 0 or height == 0:
            return np.empty(out_shape, dtype=src_dtype)

        if self._is_integer_crop(src_shape, src_dtype):
            _, _, tx, _, _, ty, _, _, _ = self.matrix
            left, top = -int(tx) + x, -int(ty) + y
            if direct_tiff:
                result = _read_direct_rgb_tiff_region(
                    source_or_image, left, top, width, height,
                    channels=channels if not drop_singleton else None,
                    should_cancel=cancel_fn,
                )
                return result
            require_memory(math.prod(out_shape) * src_dtype.itemsize)
            crop = image[top:top + height, left:left + width]
            if crop.shape[2:] == (1,):
                crop = crop[:, :, 0]
            if channels is not None and crop.ndim == 3:
                crop = crop[:, :, :channels]
            if (crop.ndim == 3 and crop.shape[2] == 3
                    and _is_bgr_rev(image) and _native_kernels.available()):
                out = np.empty(crop.shape, dtype=src_dtype)
                swap_fn = (_native_kernels.swap_rgb_to_bgr_rows_u8 if src_dtype == np.uint8
                           else _native_kernels.swap_rgb_to_bgr_rows_u16)
                _run_row_bands(
                    height, 128, cancel_fn, swap_fn, crop[..., ::-1], out,
                    parallel_pixels=height * width, workers=_warp_thread_budget(),
                )
                return out
            return crop.copy(order='C')

        M64 = np.asarray(self.matrix, dtype=self.matrix_dtype).astype(np.float64).reshape(3, 3)
        ok, M_inv = cv2.invert(M64)
        corners = np.array([
            [float(x), float(y), 1.0],
            [float(x + width - 1), float(y), 1.0],
            [float(x + width - 1), float(y + height - 1), 1.0],
            [float(x), float(y + height - 1), 1.0],
        ], dtype=np.float64).T
        sc = (M_inv @ corners) if ok else None
        unsafe = (not ok or not np.all(np.isfinite(sc)) or np.any(sc[2] <= 1e-6)
                  or max(src_shape[:2]) >= 32767)
        if not unsafe:
            sx = sc[0] / sc[2]
            sy = sc[1] / sc[2]
            # The full OpenCV map saturates to signed 16-bit coordinates.
            unsafe = max(np.max(np.abs(sx)), np.max(np.abs(sy))) >= 32760
        if unsafe:
            if direct_tiff:
                image = resolve_frame(source_or_image, should_cancel=cancel_fn)
            return self._full_warp_crop(image, x, y, width, height, channels, cancel_fn)

        halo = 8
        sx0 = max(0, int(math.floor(float(np.min(sx)))) - halo)
        sy0 = max(0, int(math.floor(float(np.min(sy)))) - halo)
        sx1 = min(src_shape[1], int(math.ceil(float(np.max(sx)))) + halo + 1)
        sy1 = min(src_shape[0], int(math.ceil(float(np.max(sy)))) + halo + 1)
        if sx0 >= sx1 or sy0 >= sy1:
            require_memory(math.prod(out_shape) * src_dtype.itemsize)
            return np.zeros(out_shape, dtype=src_dtype)
        backend = _deferred_ecc_gpu.get()
        if backend is not None and self.interpolation == 'lanczos':
            if direct_tiff:
                with source_or_image._open_direct_tiff() as stream:
                    with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        rgb = np.ndarray(src_shape, dtype=src_dtype, buffer=mm,
                                         offset=source_or_image._rgb_tiff_offset)
                        try:
                            warped = backend.warp(rgb, M64, self.dsize, cancel_fn,
                                                  bgr_rev=True, region=(x, y, width, height),
                                                  method=self.method)
                        finally:
                            del rgb
            else:
                bgr_rev = _is_bgr_rev(image)
                warped = backend.warp(image[..., ::-1] if bgr_rev else image, M64, self.dsize,
                                      cancel_fn, bgr_rev=bgr_rev, region=(x, y, width, height),
                                      method=self.method)
            if warped is not None:
                if channels is not None and warped.ndim == 3 and warped.shape[2] != channels:
                    return np.array(warped[:, :, :channels], copy=True, order='C')
                return warped

        if _perspective_near_lanczos_boundary(
                M_inv, x, y, width, height, sx0, sy0, float(np.min(sc[2])), cancel_fn):
            if direct_tiff:
                image = resolve_frame(source_or_image, should_cancel=cancel_fn)
            return self._full_warp_crop(image, x, y, width, height, channels, cancel_fn)

        T_shift = np.array([[1.0, 0.0, float(x)], [0.0, 1.0, float(y)], [0.0, 0.0, 1.0]], dtype=np.float64)
        T_src = np.array([[1.0, 0.0, float(-sx0)], [0.0, 1.0, float(-sy0)], [0.0, 0.0, 1.0]], dtype=np.float64)
        M_sub_inv = T_src @ (M_inv @ T_shift)

        threads = _warp_thread_budget()
        flags = interpolation_flag(self.interpolation) | cv2.WARP_INVERSE_MAP
        if direct_tiff:
            have_native = _native_kernels.available()
            src_ch = src_shape[2] if len(src_shape) == 3 else 1
            sub_bytes = (sy1 - sy0) * (sx1 - sx0) * src_ch * src_dtype.itemsize
            out_bytes = height * width * src_ch * src_dtype.itemsize
            require_memory(out_bytes + (0 if have_native else sub_bytes))
            with source_or_image._open_direct_tiff() as stream:
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    rgb_full = np.ndarray(
                        src_shape, dtype=src_dtype,
                        buffer=mm, offset=source_or_image._rgb_tiff_offset,
                    )
                    sub_src = (rgb_full[sy0:sy1, sx0:sx1] if have_native
                               else np.ascontiguousarray(rgb_full[sy0:sy1, sx0:sx1, ::-1]))
                    try:
                        warped = self._run_cv_warp(sub_src, M_sub_inv, (width, height), flags, threads)
                    finally:
                        del sub_src, rgb_full
            if have_native and warped.ndim == 3 and warped.shape[2] == 3:
                swap_fn = (_native_kernels.swap_bgr_inplace_rows_u8 if src_dtype == np.uint8
                           else _native_kernels.swap_bgr_inplace_rows_u16)
                _run_row_bands(
                    warped.shape[0], 128, cancel_fn, swap_fn, warped,
                    parallel_pixels=warped.shape[0] * warped.shape[1], workers=threads,
                )
        else:
            bgr_rev = _is_bgr_rev(image) and _native_kernels.available()
            sub_src = (image[..., ::-1] if bgr_rev else image)[sy0:sy1, sx0:sx1]
            src_ch = src_shape[2] if len(src_shape) == 3 else 1
            require_memory(height * width * src_ch * src_dtype.itemsize)
            warped = self._run_cv_warp(sub_src, M_sub_inv, (width, height), flags, threads)
            if bgr_rev and warped.ndim == 3 and warped.shape[2] == 3:
                swap_fn = (_native_kernels.swap_bgr_inplace_rows_u8 if warped.dtype == np.uint8
                           else _native_kernels.swap_bgr_inplace_rows_u16)
                _run_row_bands(
                    warped.shape[0], 128, cancel_fn, swap_fn, warped,
                    parallel_pixels=warped.shape[0] * warped.shape[1], workers=threads,
                )

        if backend is not None:
            backend.note_cpu_warp(self.method)
        if channels is not None and warped.ndim == 3 and warped.shape[2] != channels:
            return np.array(warped[:, :, :channels], copy=True, order='C')
        return warped


_ORIGINAL_PERSPECTIVE_CALL = PerspectiveWarp.__call__


def _perspective_output_shape(shape, dsize):
    # OpenCV drops a singleton channel axis when producing warp output.
    channels = () if shape[2:] == (1,) else shape[2:]
    return (dsize[1], dsize[0]) + channels


def _perspective_cpu_tile_bytes(dsize):
    # OpenCV uses 32x32 mapping tiles within each stripe.
    width, height = dsize
    stripes = min(height, (width * height + 65535) // 65536)
    return stripes * 16 * 1024


def _perspective_scratch_bytes(frame, warp, shape):
    from core.source_frames import SourceFrame

    if (type(frame) is not SourceFrame or frame._rgb_tiff_offset is not None
            or not (frame._jpeg_components or frame.raw_sensor_shape)
            or frame.scale != 1.0 or frame.transforms):
        return None
    # Decoded JPEG and RAW sources resolve to packed C-contiguous BGR arrays.
    # OpenCV can borrow that input and write one independent output.
    output = math.prod(shape) * frame.dtype.itemsize
    hash_bytes = 128 * 1024
    if warp._is_integer_crop(frame.shape, frame.dtype):
        return output + hash_bytes
    from core._perspective_gpu import _MAX_SIDE, _MAX_WORKSPACE_BYTES, _MIN_PIXELS

    # The exact GPU path packs at most one source band, bounded by its total
    # device workspace.
    width, height = warp.dsize
    band_bytes = (min(frame.nbytes, _MAX_WORKSPACE_BYTES)
                  if width * height >= _MIN_PIXELS and max(width, height, *frame.shape[:2]) <= _MAX_SIDE
                  else 0)
    return output + hash_bytes + max(_perspective_cpu_tile_bytes(warp.dsize), band_bytes)


@dataclass(frozen=True)
class PerspectiveStage:
    method: str
    sources: tuple
    matrices: tuple
    matrix_dtypes: tuple
    bounds: tuple
    dsize: tuple
    interpolation: str = 'lanczos'

    def __post_init__(self):
        count = len(self.sources)
        if (self.method not in ('homography', 'ecc') or count < 2
                or not all(isinstance(value, tuple) for value in
                           (self.sources, self.matrices, self.matrix_dtypes, self.bounds, self.dsize))
                or len(self.matrices) != count or len(self.matrix_dtypes) != count
                or len(self.bounds) != 4 or len(self.dsize) != 2
                or any(type(value) is not int or value <= 0 for value in self.dsize)
                or any(type(value) is not int for value in self.bounds)
                or self.interpolation not in _INTERPOLATION_FLAGS):
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
        frames = []
        for image, source, matrix, dtype in zip(images, self.sources, self.matrices, self.matrix_dtypes):
            warp = PerspectiveWarp(
                source, matrix, dtype, self.dsize, should_cancel,
                method=self.method, interpolation=self.interpolation,
            )
            shape = _perspective_output_shape(image.shape, self.dsize)
            frames.append(process_frame(image, warp, shape=shape, should_cancel=should_cancel,
                                        scratch_bytes=_perspective_scratch_bytes(image, warp, shape)))
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
                source['context'], source.get('digest_kind', 'pixel')) for source in item['sources'])
            stages.append(PerspectiveStage(item['method'], sources,
                tuple(tuple(matrix) for matrix in item['matrices']), tuple(item['matrix_dtypes']),
                tuple(item['bounds']), tuple(item['dsize']),
                interpolation=item.get('interpolation', 'lanczos')))
        return cls(tuple(stages), data['parameters'], data['downscale_width'])


def _apply_perspective_warps(images, matrices, crop, dsize, workers,
                             should_cancel, defer_warps, *, method, sources,
                             bounds, plan_callback=None, interpolation='lanczos'):
    """Keep every stage's resampling, matrix precision and crop."""
    final = tuple(crop @ matrix if crop is not None else matrix.copy() for matrix in matrices)
    stage = PerspectiveStage(
        method, tuple(sources),
        tuple(tuple(float(value) for value in matrix.flat) for matrix in final),
        tuple(matrix.dtype.str for matrix in final),
        tuple(int(value) for value in bounds), dsize,
        interpolation=interpolation,
    )
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
    ecc_gpu = _deferred_ecc_gpu.get()

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
        token = _deferred_ecc_gpu.set(ecc_gpu) if ecc_gpu is not None else None
        try:
            _check_cancel(should_cancel)
            if stage is not None and profile.enabled:
                with profile.stage(stage):
                    result = function(value)
            else:
                result = function(value)
            _check_cancel(should_cancel)
            return result
        finally:
            if token is not None:
                _deferred_ecc_gpu.reset(token)

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


def _prepare_and_estimate_pairs(values, prepare, estimate, workers, should_cancel=None, *,
                                on_prepared=None, window=None, budget=None,
                                progress_callback=None):
    """Prepare each value and estimate each adjacent pair in one bounded pool.

    ``prepare(value)`` returns a tuple whose last item is the source identity;
    ``estimate(index, previous, current)`` receives the prepared tuples of
    values ``index - 1`` and ``index``. Both kinds of task share ``workers``
    threads, so preparation (source decoding) of several frames overlaps
    without exceeding the stage's thread budget. At most ``window`` distinct
    prepared values are retained, including those still held by unfinished
    estimates. Returns the estimates in pair order and the identities in
    value order.
    If given, ``budget`` limits the sum of queued and running task workspaces.
    Its gray retention window is reserved separately.

    Leaving early, through cancellation or a failed task, cancels queued work
    and waits for running native calls before returning.
    """
    profile = _profile()
    _check_cancel(should_cancel)
    count = len(values)
    if count < 2:
        raise ValueError('pair estimation needs at least two values')
    window = max(2, int(window if window is not None else workers + 1))
    prepare_bytes = budget.prepare_bytes if budget is not None else 0
    solve_bytes = budget.solve_bytes if budget is not None else 0
    task_bytes = budget.task_bytes if budget is not None else 0
    # If memory cannot overlap the two kinds of tasks, fill the window before
    # handing it to the solvers. Alternating decode and solve would serialize
    # every pair even when concurrent solvers fit easily.
    batch_prepares = prepare_bytes + solve_bytes > task_bytes
    prepare_until = min(window, count) if batch_prepares else count
    scratch = capture_scratch()

    def staged(stage, function, *args):
        # The pool's work item keeps this task alive until after its future
        # completes. Drop the arguments (prepared frames) before returning so
        # the retention window is not exceeded while the worker lets go.
        held = list(args)

        def run():
            try:
                _check_cancel(should_cancel)
                if profile.enabled:
                    with profile.stage(stage):
                        result = function(*held)
                else:
                    result = function(*held)
                _check_cancel(should_cancel)
                return result
            finally:
                held.clear()
        return run

    from threadpoolctl import ThreadpoolController

    def init_worker():
        # Some OpenMP runtimes apply limits per thread, so each worker needs
        # its own limit even when the owner is already limited.
        ThreadpoolController().select(user_api='openmp').limit(limits=1)

    # Other OpenMP runtimes use a process-wide limit. Restore the owner's
    # previous setting only after all worker calls have stopped.
    with ThreadpoolController().select(user_api='openmp').limit(limits=1):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers, initializer=init_worker)
        prepared = {}
        # Preparation hands its result over here and returns None: the pool's
        # worker keeps a task's return value referenced for a moment after its
        # future completes, which would let a released frame outlive the window.
        ready = {}
        estimates = {}
        identities = [None] * count
        results = [None] * (count - 1)
        completed_estimates = 0
        next_prepare = 0
        next_estimate = 1
        try:
            while next_estimate < count or estimates:
                _check_cancel(should_cancel)
                for future in (*prepared.values(), *estimates.values()):
                    if future.done() and future.exception() is not None:
                        future.result()
                for index in [index for index, future in estimates.items() if future.done()]:
                    results[index - 1] = estimates.pop(index).result()
                    completed_estimates += 1
                    if progress_callback is not None:
                        progress_callback(completed_estimates, count - 1)
                        _check_cancel(should_cancel)
                if batch_prepares and not estimates and next_estimate == prepare_until:
                    prepare_until = min(count, prepare_until + window - 1)
                work_left = task_bytes - solve_bytes * len(estimates)
                if prepare_bytes:
                    work_left -= prepare_bytes * sum(not future.done() for future in prepared.values())
                # Pairs go first: they release prepared frames.
                while (work_left >= solve_bytes and next_estimate < count
                       and (not batch_prepares or next_prepare == prepare_until)
                       and next_estimate in prepared and prepared[next_estimate - 1].done()
                       and prepared[next_estimate].done()):
                    # A future can fail after the scan above and before this pair
                    # becomes ready. Propagate its error before accessing pixels.
                    prepared[next_estimate - 1].result()
                    prepared[next_estimate].result()
                    prepared.pop(next_estimate - 1)
                    previous = ready.pop(next_estimate - 1)
                    current = ready[next_estimate]
                    estimates[next_estimate] = pool.submit(
                        run_with_scratch, scratch,
                        staged('registration.estimate', estimate, next_estimate, previous, current))
                    del previous, current
                    work_left -= solve_bytes
                    next_estimate += 1
                # Count every distinct frame still referenced: a pending estimate
                # holds both of its frames, and estimates can finish out of
                # order, so its later frame may already have left ``prepared``.
                retained = set(prepared)
                for index in estimates:
                    retained.update((index - 1, index))
                while (next_prepare < prepare_until and len(retained) < window
                       and work_left >= prepare_bytes):
                    index = next_prepare

                    def prepare_one(index=index, value=values[index]):
                        result = prepare(value)
                        identities[index] = result[-1]
                        if on_prepared is not None:
                            on_prepared(index, result)
                        ready[index] = result

                    prepared[index] = pool.submit(
                        run_with_scratch, scratch,
                        staged('registration.prepare', prepare_one))
                    work_left -= prepare_bytes
                    retained.add(index)
                    next_prepare += 1
                if next_estimate >= count and not estimates:
                    break
                # Completed preparation stays available for its adjacent pairs.
                # Including it in FIRST_COMPLETED would return immediately and
                # spin while the remaining native decode/estimate is running.
                waiting = [future for future in (*prepared.values(), *estimates.values())
                           if not future.done()]
                if waiting:
                    concurrent.futures.wait(waiting, timeout=0.05,
                                            return_when=concurrent.futures.FIRST_COMPLETED)
                waiting.clear()
        finally:
            for future in list(prepared.values()) + list(estimates.values()):
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            prepared.clear()
            ready.clear()
            estimates.clear()
    _check_cancel(should_cancel)
    return results, identities


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
        result = sources
        for stage in plan.stages:
            result = stage.apply(result, defer_warps=True, should_cancel=should_cancel)
            _record_grouped_crop(*stage.bounds)
        if defer_warps:
            return result
        from core.source_frames import resolve_frame
        return [store_frame(resolve_frame(frame, should_cancel=should_cancel)) for frame in result]
    from functools import partial
    from core import openfocus_alignment as _oa
    from core.openfocus_alignment import (
        _ORIGINAL_APPLY_ALIGNMENT,
        alignment_resolution_bytes,
        apply_alignment,
        apply_alignment_region,
    )
    from core.source_frames import resolve_frame
    plan.validate_sources(sources, should_cancel=should_cancel)
    for transform in plan.transforms:
        _check_cancel(should_cancel)
        _report_openfocus_transform(transform, plan.reference_index)
    h, w = sources[0].shape[:2]
    left, top, right, bottom = 0, 0, w, h
    if getattr(plan, 'auto_crop', True):
        bounds = [transform.valid_bounds for transform in plan.transforms]
        left = max(bound[0] for bound in bounds)
        top = max(bound[1] for bound in bounds)
        right = min(bound[2] for bound in bounds)
        bottom = min(bound[3] for bound in bounds)
        if top >= bottom or left >= right:
            raise ValueError('Registration transforms have no common valid crop')
    _record_grouped_crop(top, bottom, left, right)
    plan_interp = getattr(plan, 'interpolation', None)
    use_direct_region = apply_alignment is _ORIGINAL_APPLY_ALIGNMENT and plan_interp is None
    result = []
    for source, transform in zip(sources, plan.transforms):
        _check_cancel(should_cancel)
        if plan_interp is not None:
            processor = partial(apply_alignment, transform=transform, should_cancel=should_cancel,
                                interpolation=plan_interp)
        else:
            processor = partial(apply_alignment, transform=transform, should_cancel=should_cancel)
        if defer_warps:
            from core.source_frames import process_frame, crop_frame
            aligned = process_frame(source, processor, should_cancel=should_cancel,
                                    scratch_bytes=alignment_resolution_bytes(source))
            h, w = source.shape[:2]
            if (left, top, right, bottom) != (0, 0, w, h):
                aligned = crop_frame(aligned, left, top, right - left, bottom - top,
                                     should_cancel=should_cancel, store_source=False)
        elif use_direct_region:
            image = apply_alignment_region(
                source, transform, left, top, right - left, bottom - top,
                should_cancel=should_cancel,
            )
            aligned = store_frame(image)
        else:
            image = processor(resolve_frame(source, should_cancel=should_cancel))
            aligned = store_frame(image[top:bottom, left:right])
        result.append(aligned)
    _check_cancel(should_cancel)
    return result


def _align_openfocus_impl(input_source, output_path=None, img_filenames=None, *,
                        should_cancel=None, max_shift_ratio=.05, max_shift_x_ratio=None,
                        max_shift_y_ratio=None, failure_policy='continue',
                        reference_frame=1, rotation=0., magnification=3.,
                        alignment_roi=None, auto_crop=True, interpolation=None,
                        defer_warps=False, plan_callback=None, dispatch_callback=None):
    """Wide detector using the verified search and sampling behavior."""
    from core.openfocus_alignment import _alignment_spill_bytes, estimate_alignment_plan
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
    from contextlib import nullcontext
    from core.frame_resolution_cache import ScopedCache, current, within_retention_limit
    from core.frame_storage import scratch_directory
    from core.source_frames import select_frame_scratch, without_source_cache_growth

    cache = current()
    large_stack = not within_retention_limit(images)
    # Estimation only needs one color frame at a time. Once the stack exceeds
    # the retention limit, reuse existing mappings and let fusion stage its own.
    cache_reads = cache.reuse_only() if large_stack and isinstance(cache, ScopedCache) else nullcontext()
    source_reads = without_source_cache_growth() if large_stack else nullcontext()
    with scratch_directory():
        height, width = images[0].shape[:2]
        select_frame_scratch(
            images, working_size=_alignment_spill_bytes(height, width, len(images), reference_frame),
            should_cancel=should_cancel,
        )
        try:
            with cache_reads, source_reads:
                plan = estimate_alignment_plan(
                    images, max_shift_ratio=max_shift_ratio,
                    max_shift_x_ratio=max_shift_x_ratio,
                    max_shift_y_ratio=max_shift_y_ratio,
                    reference_frame=reference_frame, rotation=rotation, magnification=magnification,
                    alignment_roi=alignment_roi, auto_crop=auto_crop, interpolation=interpolation,
                    should_cancel=should_cancel,
                    failure_policy=failure_policy)
        except RejectedAlignmentFrame as error:
            for transform in error.completed_transforms:
                _report_openfocus_transform(transform, reference_frame - 1)
            _report_alignment(error.frame_index, 'openfocus', 'failed', reason=error.reason)
            raise
        if plan_callback is not None:
            plan_callback(plan)
        result = apply_alignment_plan(images, plan, defer_warps=defer_warps, should_cancel=should_cancel)
        if output_path:
            from core.source_frames import resolve_frame
            _save_registration((resolve_frame(frame, should_cancel=should_cancel) for frame in result), output_path,
                               img_filenames, should_cancel)
        if dispatch_callback is not None and len(images) > 1:
            dispatch_callback('openfocus', estimation='CPU', warp='CPU')
        return result


# ========== ECC对齐算法实现（高精度） ==========

def _ecc_width(shape, requested):
    return 1024 if max(shape[:2]) >= 2048 else requested


def _ecc_resize_geometry(shape, downscale_width):
    h, w = shape[:2]
    scale = downscale_width / float(w) if w > downscale_width else 1.0
    return (round(h * scale), round(w * scale), scale) if scale < 1.0 else (h, w, scale)


def _ecc_solve_bytes(pixels):
    # An unsuccessful native solve can retry with OpenCV, whose homography
    # solver needs a Jacobian, grids, gradients and masks of this geometry.
    return (30 * 4 + 4) * pixels + 128 * 1024


@dataclass(frozen=True)
class _EccStageBudget:
    workers: int
    task_bytes: int
    prepare_bytes: int
    solve_bytes: int
    jpeg_inputs: dict[int, _JpegEccPreparation]
    accelerate_ecc: bool = True


def _plan_deferred_ecc(images, downscale_width, accelerate_ecc, preference, thread_count,
                       should_cancel, *, alignment_roi=None):
    from core.openfocus_alignment import _hash_scratch_bytes, _is_bgr_rev
    from core.resource_policy import cpu_capacity, probe_resources, ram_ceiling, require_memory
    from core.source_frames import frame_resolution_bytes, is_direct_rgb_tiff

    if accelerate_ecc:
        from core._ecc_area_gpu import _AXIS, _TILE_ROWS

    width = _ecc_width(images[0].shape, downscale_width)
    max_pixels = prepare_bytes = prepare_bytes_cpu = 0
    jpeg_inputs = {}
    for frame in images:
        _check_cancel(should_cancel)
        h, w, scale = _ecc_resize_geometry(frame.shape, width)
        pixels = max(1, h) * max(1, w)
        max_pixels = max(max_pixels, pixels)
        jpeg = None if alignment_roi is not None else _plan_jpeg_ecc(frame, width)
        if jpeg is not None:
            jpeg_inputs[id(frame)] = jpeg
            req = max(jpeg.workspace_bytes, 2 * pixels + 64 * 1024)
            prepare_bytes = max(prepare_bytes, req)
            prepare_bytes_cpu = max(prepare_bytes_cpu, req)
            continue
        channels = frame.shape[2] if frame.ndim == 3 else 1
        itemsize = frame.dtype.itemsize
        decode = frame_resolution_bytes(frame)
        # Source hashing uses one band, independently of any pack that OpenCV
        # needs for resize or conversion. A direct TIFF still faults in pixels.
        from core.source_frames import SourceFrame
        decoded = frame.nbytes if decode else 0
        packed = (0 if ((isinstance(frame, np.ndarray)
                         and (frame.flags.c_contiguous or _is_bgr_rev(frame)))
                        or (type(frame) is SourceFrame and frame._rgb_tiff_offset is None)
                        or (is_direct_rgb_tiff(frame)
                            and frame._rgb_tiff_offset % frame.dtype.alignment == 0))
                  else frame.nbytes)
        color, gray = pixels * channels * itemsize, pixels * itemsize
        prepare = max(decode, decoded + _hash_scratch_bytes(frame), 2 * gray + 64 * 1024)
        prepare_cpu = prepare
        if scale < 1.0:
            # INTER_AREA uses axis tables and a pair of accumulator rows. The
            # OpenCL path stages bounded _TILE_ROWS stripes rather than full
            # frames, and gracefully falls back if that headroom is absent.
            resize_scratch = 32 * (sum(frame.shape[:2]) + h + w * channels)
            prepare_cpu = max(prepare, decoded + packed + color + resize_scratch, color + gray)
            if accelerate_ecc and frame.dtype == np.uint8:
                stripe_src_rows = min(frame.shape[0], math.ceil(_TILE_ROWS / max(scale, 1e-6)) + 2)
                gpu_stripe = (
                    min(frame.nbytes, stripe_src_rows * frame.shape[1] * channels * itemsize)
                    + min(color, _TILE_ROWS * w * channels * itemsize)
                    + _AXIS.itemsize * (h + w)
                )
                resize_scratch = max(resize_scratch, gpu_stripe)
            prepare = max(prepare, decoded + packed + color + resize_scratch, color + gray)
        else:
            prepare = max(prepare, decoded + packed + gray)
            prepare_cpu = prepare
        prepare_bytes = max(prepare_bytes, prepare)
        prepare_bytes_cpu = max(prepare_bytes_cpu, prepare_cpu)

    # Each prepared float32 plane can retain its lower-resolution pyramid levels.
    gray_bytes = 6 * max_pixels
    solve_bytes = _ecc_solve_bytes(max_pixels)
    snapshot = probe_resources()
    ceiling_ram = ram_ceiling(snapshot)
    if (accelerate_ecc
            and 2 * gray_bytes + max(prepare_bytes, solve_bytes) > ceiling_ram
            and 2 * gray_bytes + max(prepare_bytes_cpu, solve_bytes) <= ceiling_ram):
        accelerate_ecc = False
        prepare_bytes = prepare_bytes_cpu
    ram = require_memory(2 * gray_bytes + max(prepare_bytes, solve_bytes), snapshot)
    ceiling = cpu_capacity(snapshot, preference)
    if thread_count is not None and thread_count > 0:
        ceiling = min(ceiling, int(thread_count))
    # Reserve a window for all the prepared inputs. With that window full,
    # either all solvers or at least one preparation must fit on its own.
    workers = max(1, min(ceiling, len(images) - 1,
                         (ram - gray_bytes) // (solve_bytes + gray_bytes),
                         (ram - gray_bytes - prepare_bytes) // gray_bytes))
    task_bytes = ram - (workers + 1) * gray_bytes
    log_action('registration_plan', method='ecc', frames=len(images), shape=images[0].shape,
               workers=workers, native_threads=1, ram_budget=ram, vram_budget=0,
               gpu_requested=False, accelerated_resize_requested=accelerate_ecc,
               prepare_bytes=prepare_bytes,
               solve_bytes=solve_bytes, retained_gray_bytes=(workers + 1) * gray_bytes,
               reduced_jpeg_frames=len(jpeg_inputs))
    return _EccStageBudget(workers, task_bytes, prepare_bytes, solve_bytes, jpeg_inputs, accelerate_ecc)



def _align_ecc_impl(input_source, output_path=None, img_filenames=None, downscale_width=1000,
                    thread_count: int = 4, should_cancel=None, use_gpu: bool = False,
                    failure_policy='continue', *, defer_warps=False, plan_callback=None,
                    accelerate_ecc=True, stage_budget=None, dispatch_callback=None,
                    progress_callback=None, alignment_roi=None,
                    max_shift_x_ratio=None, max_shift_y_ratio=None,
                    max_rotation_deg=None, max_scale_change=None,
                    auto_crop=True, interpolation='lanczos'):
    """
    基于 ECC (增强相关系数) 的高精度图像栈对齐算法
    适用于：显微摄影、微距摄影中伴随呼吸效应的图像栈
    优势：亚像素精度，自动处理缩放中心偏移，不依赖特征点
    优化：并行预处理、并行变换、合并裁切操作
    """
    # Directory loading and source policy belong to ImageRegistration.process.
    interpolation_flag(interpolation)
    images = input_source
    if len(images):
        _report_reference('ecc')

    if len(images) < 2:
        return images

    accelerate_resize = accelerate_ecc and (stage_budget is None or stage_budget.accelerate_ecc)
    area_gpu = None
    if accelerate_resize:
        from core._ecc_area_gpu import ExactAreaGpu
        area_gpu = ExactAreaGpu(should_cancel=should_cancel)

    # --- 2. 初始化 ---
    h_orig, w_orig = images[0].shape[:2]
    roi = normalize_alignment_roi(alignment_roi, (h_orig, w_orig))
    roi_center = ((roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0) if roi is not None else None)
    # 如果单张图像过大（任一边 >= 2048），强制将用于下采样的目标宽度设为 1024
    requested_width = downscale_width
    downscale_width = _ecc_width(images[0].shape, requested_width)
    if max(h_orig, w_orig) >= 2048:
        print(f"[Registration][ECC] Large image detected ({h_orig}x{w_orig}), setting downscale_width {requested_width} -> {downscale_width}")
    
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
    ecc_native = None
    use_pyr_reuse = False
    pyr_builder = None
    pyr_solver = None
    if (warp_mode == cv2.MOTION_HOMOGRAPHY
            and cv2.findTransformECC is _ORIGINAL_FIND_TRANSFORM_ECC):
        ecc_native = _load_ecc_native()
        if ecc_native is not None:
            from core._ecc_native import (
                build_ecc_pyramid as _build_ecc_pyramid,
                find_transform_ecc_homography as _default_ecc_native,
                find_transform_ecc_homography_pyramids as _solve_ecc_pyramids,
            )
            if ecc_native is _default_ecc_native:
                use_pyr_reuse = True
                pyr_builder = _build_ecc_pyramid
                pyr_solver = _solve_ecc_pyramids

    # A stack can revisit the same adjacent pair. Cache only small, successful
    # solve results, keyed by exact prepared pixels in their original order.
    recent_pairs = OrderedDict() if len(images) > 2 else None
    recent_pairs_lock = threading.Lock()

    from core.frame_resolution_cache import within_retention_limit
    large_stack = not within_retention_limit(images)

    # 预处理函数：转灰度 + 降采样 + 高斯模糊 + 归一化
    def preprocess(source):
        from core.openfocus_alignment import _can_use_file_digest, _is_bgr_rev
        from core.source_frames import resolve_frame, source_area_resizer, without_source_cache_growth
        # Resizing during JPEG decode avoids loading and transferring full-size
        # color pixels. Keep OpenCL area resizing for sources that need it.
        jpeg = None if roi is not None else (
            stage_budget.jpeg_inputs.get(id(source)) if stage_budget is not None
            else _plan_jpeg_ecc(source, downscale_width))
        if jpeg is not None:
            pixels, identity = _decode_jpeg_ecc(source, jpeg, should_cancel)
            scale = jpeg.scale
        else:
            # Replace each intermediate so a deferred full-size decode can be
            # released as soon as resize or color conversion no longer needs it.
            use_file_digest = _can_use_file_digest(source)
            identity = _perspective_source_file(source, should_cancel) if use_file_digest else None
            cache_scope = without_source_cache_growth() if large_stack else nullcontext()
            with cache_scope, source_area_resizer(area_gpu):
                pixels = resolve_frame(source, should_cancel=should_cancel)
            if identity is None:
                identity = _perspective_source(source, pixels, should_cancel)
            is_rgb = False
            if roi is not None:
                rx, ry, rw, rh = roi
                pixels = pixels[ry:ry + rh, rx:rx + rw]
            target_height, _, scale = _ecc_resize_geometry(pixels.shape, downscale_width)
            if scale < 1.0:
                resized = None
                # ExactAreaGpu only beats parallel CPU INTER_AREA on >=16MP 8-bit inputs;
                # smaller or 16-bit stacks incur OpenCL compile + queue-lock overhead.
                if (area_gpu is not None and target_height > 0
                        and target_height / pixels.shape[0] == scale
                        and pixels.dtype == np.uint8
                        and pixels.shape[0] * pixels.shape[1] >= 16_000_000):
                    # An explicit size preserves OpenCV's scale only at this geometry.
                    resized = area_gpu.resize(pixels, (downscale_width, target_height))
                if resized is not None:
                    pixels = resized
                else:
                    is_rgb = _is_bgr_rev(pixels)
                    if is_rgb:
                        pixels = pixels[..., ::-1]
                    # Resize color first to preserve exact integer rounding in the ECC inputs.
                    pixels = cv2.resize(pixels, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                del resized
            pixels = _registration_gray(pixels, is_rgb=is_rgb)
        pixels = cv2.GaussianBlur(pixels, (5, 5), 0)
        # findTransformECC 的数值稳定性依赖归一化的浮点输入：uint8 的
        # 大幅值梯度会让优化步长过大，常见于小位移场景直接不收敛
        divisor = 65535.0 if pixels.dtype == np.uint16 else 255.0
        gray = np.divide(pixels, divisor, dtype=np.float32)
        del pixels
        key = None
        if recent_pairs is not None:
            key = gray.shape, hashlib.blake2b(memoryview(gray), digest_size=16).digest()
        prepared_data = pyr_builder(gray) if use_pyr_reuse else gray
        return prepared_data, scale, key, identity

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
        previous, scale_factor, previous_key, identity = prepare(next(source))
        source_identities.append(identity)
        for idx, image in enumerate(source, 1):
            current, _, current_key, identity = prepare(image)
            source_identities.append(identity)
            yield idx, previous, current, scale_factor, previous_key, current_key
            previous, previous_key = current, current_key

    def _compute_pair_transform(pair):
        """ECC transform of frame idx relative to frame idx-1 (scale-restored)."""
        idx, prev_gray, curr_gray, scale_factor, previous_key, current_key = pair
        key = previous_key, current_key
        if recent_pairs is not None:
            with recent_pairs_lock:
                result = recent_pairs.get(key)
                if result is not None:
                    recent_pairs.move_to_end(key)
                    return result

        cc, warp_matrix = None, None
        try:
            if use_pyr_reuse:
                cc, warp_matrix = pyr_solver(
                    prev_gray,
                    curr_gray,
                    max_iter=number_of_iterations,
                    eps=termination_eps,
                    num_threads=max_workers if len(images) == 2 else 1,
                )
            elif ecc_native is not None:
                cc, warp_matrix = ecc_native(
                    prev_gray,
                    curr_gray,
                    max_iter=number_of_iterations,
                    eps=termination_eps,
                )
            if cc is None or warp_matrix is None:
                _check_cancel(should_cancel)
                tpl_full = prev_gray[0] if use_pyr_reuse else prev_gray
                src_full = curr_gray[0] if use_pyr_reuse else curr_gray
                rows = 3 if warp_mode == cv2.MOTION_HOMOGRAPHY else 2
                warp_matrix = np.eye(rows, 3, dtype=np.float32)
                cc, warp_matrix = cv2.findTransformECC(
                    tpl_full,  # template
                    src_full,  # input
                    warp_matrix,
                    warp_mode,
                    criteria,
                    None,
                    1
                )
            if failure_policy == 'fail' and (cc is None or not np.isfinite(cc)):
                raise RejectedAlignmentFrame('ECC', idx, 'ECC failed to converge')

            # 尺度还原
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix[0, 2] /= scale_factor
                warp_matrix[1, 2] /= scale_factor
                warp_matrix[2, 0] *= scale_factor
                warp_matrix[2, 1] *= scale_factor
            else:
                warp_matrix[0, 2] /= scale_factor
                warp_matrix[1, 2] /= scale_factor
            if roi is not None:
                warp_matrix = conjugate_roi_matrix(warp_matrix, roi[0], roi[1])
        except cv2.error as exc:
            _check_cancel(should_cancel)
            if failure_policy == 'fail':
                raise RejectedAlignmentFrame('ECC', idx, 'ECC failed to converge') from exc
            cc = None
            print(f"Warning: ECC failed to converge at frame {idx}. Assuming no motion.")
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
        result = warp_matrix, cc
        if recent_pairs is not None and cc is not None and np.isfinite(cc):
            with recent_pairs_lock:
                recent_pairs[key] = result
                recent_pairs.move_to_end(key)
                if len(recent_pairs) > 32:
                    recent_pairs.popitem(last=False)
        return result

    from contextlib import closing, nullcontext
    with closing(area_gpu) if area_gpu is not None else nullcontext():
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
                if progress_callback is not None:
                    progress_callback(len(pair_transforms), len(images) - 1)
                    _check_cancel(should_cancel)
                del pair  # Release the old gray before decoding the next frame.
        else:
            def prepare_frame(image):
                with deferred_warp_threads(1):
                    return preprocess(image)

            def estimate_pair(index, previous, current):
                return _compute_pair_transform((index, previous[0], current[0], scale_of[0],
                                                previous[2], current[2]))

            scale_of = []

            def remember_scale(index, prepared):
                if index == 0:
                    scale_of.append(prepared[1])

            pair_transforms, prepared_identities = _prepare_and_estimate_pairs(
                images, prepare_frame, estimate_pair, max_workers, should_cancel,
                on_prepared=remember_scale, budget=stage_budget,
                progress_callback=progress_callback)
            source_identities.extend(prepared_identities)

        if area_gpu is not None:
            log_action('registration_ecc_resize', device=area_gpu.device_name or 'CPU',
                       gpu_resizes=area_gpu.frames_resized, frames=len(images))

    trajectory_failed = False
    for idx, (warp_matrix, cc) in enumerate(pair_transforms, start=1):
        _check_cancel(should_cancel)
        # Validate per-pair transform against configured geometric limits.
        bound_reason = None
        if cc is not None:
            if warp_mode == cv2.MOTION_AFFINE:
                local_3x3 = np.vstack([warp_matrix, np.array([[0, 0, 1]], dtype=np.float32)])
            else:
                local_3x3 = warp_matrix
            try:
                pair_inv = np.linalg.inv(local_3x3.astype(np.float64))
            except np.linalg.LinAlgError:
                pair_inv = None
            bound_reason = (
                validate_transform_bounds(
                    pair_inv, (h_orig, w_orig),
                    max_shift_x_ratio=max_shift_x_ratio,
                    max_shift_y_ratio=max_shift_y_ratio,
                    max_rotation_deg=max_rotation_deg,
                    max_scale_change=max_scale_change,
                    center=roi_center,
                )
                if pair_inv is not None else 'motion outside limits'
            )
            if bound_reason is not None:
                if failure_policy == 'fail':
                    raise RejectedAlignmentFrame('ECC', idx, bound_reason)
                print(f"Warning: ECC frame {idx} rejected ({bound_reason}). Assuming no motion.")
                cc = None
                warp_matrix = (np.eye(3, dtype=np.float32) if warp_mode == cv2.MOTION_HOMOGRAPHY
                               else np.eye(2, 3, dtype=np.float32))
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
        reason = (bound_reason if bound_reason is not None else
                  'ECC failed to converge' if cc is None else
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

    if auto_crop:
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
    else:
        top, bottom, left, right = 0, h_orig, 0, w_orig
        _record_grouped_crop(top, bottom, left, right)
        do_crop = False
        target_w, target_h = w_orig, h_orig
        offset_x, offset_y = 0, 0

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
    if defer_warps:
        log_action('registration_warp_deferred', method='ecc', frames=len(images),
                   shape=(target_h, target_w))
    elif HAS_CUPY:
        log_action('registration_warp', method='ecc',
                   provider='CuPy', device='CUDA', shape=(target_h, target_w),
                   frames=len(images), gpu_requested=True)

    if not HAS_CUPY:
        with deferred_ecc_warp_accelerator(accelerate_ecc and interpolation == 'lanczos' and
                                           (not defer_warps or output_path)) as warp:
            aligned_images = _apply_perspective_warps(
                images, H_matrices, T_crop if do_crop else None, (target_w, target_h),
                max_workers, should_cancel, defer_warps, method='ecc', sources=source_identities,
                bounds=(top, bottom, left, right), plan_callback=plan_callback,
                interpolation=interpolation)
            _save_registration(aligned_images, output_path, img_filenames, should_cancel)
        warp_device = 'CPU'
        if warp is not None and warp.gpu_warps:
            warp_device = ('CPU / ' if warp.cpu_warps else '') + warp.device_name
        if not defer_warps or output_path:
            log_action('registration_warp', method='ecc', device=warp_device,
                       provider='OpenCL' if warp_device != 'CPU' else 'OpenCV',
                       shape=(target_h, target_w), frames=len(images), gpu_requested=accelerate_ecc,
                       gpu_warps=warp.gpu_warps if warp is not None else 0)
        if dispatch_callback is not None:
            dispatch_callback('ecc', estimation='CPU', warp=warp_device,
                resize_accelerator=area_gpu.device_name if area_gpu and area_gpu.frames_resized else None,
                resize_fallback=bool((accelerate_ecc and not accelerate_resize)
                                     or (area_gpu and area_gpu.fell_back)))
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
    if dispatch_callback is not None:
        dispatch_callback('ecc', estimation='CPU', warp='CUDA (CuPy)',
            resize_accelerator=area_gpu.device_name if area_gpu and area_gpu.frames_resized else None,
            resize_fallback=bool(area_gpu and area_gpu.fell_back))

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
        images = []
        for path in img_paths:
            if os.name == 'nt' and not path.isascii():
                data = np.fromfile(path, dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
                del data
            else:
                frame = cv2.imread(path)
            images.append(frame)
        # Store original filenames with extensions
        img_filenames = [os.path.basename(path) for path in img_paths]
    else:
        images = input_source

    n_frames = len(images)
    img_first = images[0]
    h, w = img_first.shape[:2]

    prev_gray = img_first if img_first.ndim == 2 else cv2.cvtColor(img_first, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames, 3), np.float32)

    # Feature detection parameters
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Calculate transforms (frame i to frame i+1)
    for i in range(n_frames - 1):
        curr = images[i + 1]
        curr_gray = curr if curr.ndim == 2 else cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

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
            from utils.image_utils import imwrite_auto
            imwrite_auto(output_file, frame_stabilized, exif=b'')

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

    _UNSET_STAGE_ROI = object()

    def __init__(self, method='homography', downscale_width=1024,
                 max_shift_ratio=None, brightness_reference='first',
                 use_gpu=False, parameters=None, *, defer_warps=False, accelerate_ecc=True,
                 alignment_roi=None, max_shift_x_ratio=None, max_shift_y_ratio=None,
                 max_rotation_deg=None, max_scale_change=None, auto_crop=None,
                 interpolation=None):
        from core.algorithms import (alignment_steps, registration_parameter_specs,
                                     resolve_registration_parameters)
        self.method = method
        self.steps = alignment_steps(method)
        spec_keys = {item.key for item in registration_parameter_specs(method)}
        raw_params = dict(parameters) if parameters is not None else {}

        # Extract optional geometric/resampling overrides even on methods whose
        # declared parameter schema uses legacy keys (e.g. 'openfocus').
        self.alignment_roi = alignment_roi
        self.max_shift_x_ratio = max_shift_x_ratio
        self.max_shift_y_ratio = max_shift_y_ratio
        self.auto_crop = True if auto_crop is None else bool(auto_crop)
        self.openfocus_interpolation = interpolation

        if 'alignment_roi' in raw_params and 'alignment_roi' not in spec_keys:
            if self.alignment_roi is None:
                self.alignment_roi = raw_params.pop('alignment_roi')
            else:
                raw_params.pop('alignment_roi')

        if 'max_shift_ratio' in spec_keys:
            if 'max_shift_x_ratio' in raw_params:
                val = float(raw_params.pop('max_shift_x_ratio'))
                if self.max_shift_x_ratio is None:
                    self.max_shift_x_ratio = val
            if 'max_shift_y_ratio' in raw_params:
                val = float(raw_params.pop('max_shift_y_ratio'))
                if self.max_shift_y_ratio is None:
                    self.max_shift_y_ratio = val
            if 'max_rotation_deg' in raw_params:
                val = float(raw_params.pop('max_rotation_deg'))
                raw_params.setdefault('rotation', val)
            if 'max_scale_change' in raw_params:
                val = float(raw_params.pop('max_scale_change'))
                raw_params.setdefault('magnification', val * 100.0)
            if 'auto_crop' in raw_params:
                val = bool(raw_params.pop('auto_crop'))
                if auto_crop is None:
                    self.auto_crop = val
            if 'interpolation' in raw_params:
                val = raw_params.pop('interpolation')
                if self.openfocus_interpolation is None:
                    self.openfocus_interpolation = val

        values = resolve_registration_parameters(method, raw_params)
        if max_shift_ratio is not None:
            if 'max_shift_ratio' in spec_keys:
                values['max_shift_ratio'] = max_shift_ratio
            elif 'max_shift_x_ratio' in spec_keys and 'max_shift_y_ratio' in spec_keys:
                values['max_shift_x_ratio'] = max_shift_ratio
                values['max_shift_y_ratio'] = max_shift_ratio
        if max_shift_x_ratio is not None and 'max_shift_x_ratio' in spec_keys:
            values['max_shift_x_ratio'] = max_shift_x_ratio
        if max_shift_y_ratio is not None and 'max_shift_y_ratio' in spec_keys:
            values['max_shift_y_ratio'] = max_shift_y_ratio
        if max_rotation_deg is not None:
            if 'max_rotation_deg' in spec_keys:
                values['max_rotation_deg'] = max_rotation_deg
            elif 'rotation' in spec_keys:
                values['rotation'] = max_rotation_deg
        if max_scale_change is not None:
            if 'max_scale_change' in spec_keys:
                values['max_scale_change'] = max_scale_change
            elif 'magnification' in spec_keys:
                values['magnification'] = float(max_scale_change) * 100.0
        if auto_crop is not None and 'auto_crop' in spec_keys:
            values['auto_crop'] = bool(auto_crop)
        if interpolation is not None and 'interpolation' in spec_keys:
            values['interpolation'] = interpolation

        self.parameters = resolve_registration_parameters(method, values)
        if self.openfocus_interpolation is not None:
            interpolation_flag(self.openfocus_interpolation)
        self.downscale_width = int(downscale_width) if downscale_width is not None else 1024
        self.brightness_reference = brightness_reference if brightness_reference in ('first', 'last') else 'first'
        self.use_gpu = bool(use_gpu)
        # use_gpu explicitly selects the legacy CuPy warp. The application
        # preference admits only pixel-equivalent OpenCL resize and warp work.
        self.accelerate_ecc = bool(accelerate_ecc)
        self.defer_warps = bool(defer_warps)
        self.alignment_plan = None
        self._perspective_stages = []
        self._dispatch = []
        self._active_stage_roi = self._UNSET_STAGE_ROI

    @property
    def dispatch(self):
        return tuple(self._dispatch)

    def _record_dispatch(self, method, **devices):
        from core.render_context import RegistrationDispatch
        self._dispatch.append(RegistrationDispatch(method, **devices))
    
    def process(self, 
                input_source: Union[str, List[np.ndarray]], 
                output_path: Optional[str] = None,
                thread_count: int = 4,
                should_cancel=None, resource_preference="balanced",
                progress_callback=None) -> List[np.ndarray]:
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
            from core.source_frames import SOURCE_CACHE
            SOURCE_CACHE.adapt_to_memory()
            self.alignment_plan = None
            self._perspective_stages = []
            self._dispatch.clear()
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
            workers = 1
            if not deferred or self.steps != ('ecc',):
                height, width = images[0].shape[:2]
                channels = images[0].shape[2] if images[0].ndim == 3 else 1
                from core.resource_policy import registration_working_set
                from core.source_frames import frame_resolution_bytes
                if deferred and all(step in ('homography', 'ecc') for step in self.steps):
                    # Deferred homography/combined estimation downscales each frame to
                    # at most downscale_width (1024px for >=2048px sources) for SIFT
                    # and never materializes full-resolution warps during registration.
                    dw = 1024 if max(height, width) >= 2048 else self.downscale_width
                    scale = min(1.0, dw / float(max(1, width)))
                    scaled_px = max(1, round(height * scale) * round(width * scale))
                    feature_scratch = scaled_px * 32 + 16 * 1024 * 1024
                    fixed = 0
                    per_worker = max(
                        max(frame_resolution_bytes(image), image.nbytes) + feature_scratch
                        for image in images
                    )
                else:
                    # Other aligners and eager warps retain their existing shared budget.
                    fixed, per_worker = registration_working_set(height, width,
                        max(image.size * 16 + frame_resolution_bytes(image) for image in images))
                plan = plan_execution(Workload(
                    algorithm='registration', frames=len(images), height=height,
                    width=width, channels=channels, itemsize=images[0].dtype.itemsize,
                    fixed_bytes=fixed, per_worker_bytes=per_worker,
                    outer_tasks=len(images), parallel_layer='outer'),
                    preference=resource_preference, user_threads=thread_count)
                workers = min(plan.workers, plan.max_in_flight)
                log_action('registration_plan', method=self.method, frames=len(images),
                           shape=images[0].shape, workers=workers,
                           native_threads=plan.native_threads, ram_budget=plan.ram_budget,
                           vram_budget=plan.vram_budget, gpu_requested=self.use_gpu)
            from core.algorithms import get_registration, resolve_runner
            result = images
            stage_roi = self.alignment_roi
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
                self._active_stage_roi = stage_roi
                stage_input_shape = result[0].shape[:2]
                stage_count_before = len(self._perspective_stages)
                with log_operation('registration_stage', method=step,
                                   parameters=parameters, provider='OpenCV',
                                   estimation_device='CPU', frames=len(result),
                                   shape=result[0].shape) as operation, \
                        _profile().stage(f'step.{step}'):
                    try:
                        execution = {'thread_count': workers, 'should_cancel': should_cancel}
                        if step == 'ecc' and progress_callback is not None:
                            execution['progress_callback'] = progress_callback
                        if deferred and step == 'ecc':
                            budget = _plan_deferred_ecc(
                                result, self.downscale_width,
                                self.accelerate_ecc, resource_preference, thread_count,
                                should_cancel, alignment_roi=stage_roi)
                            execution.update(thread_count=budget.workers, stage_budget=budget)
                        result = resolve_runner(spec.runner)(self, result,
                            **execution, **parameters)
                        _check_cancel(should_cancel)
                    except RegistrationCancelled:
                        operation.cancel()
                        raise
                    finally:
                        self._active_stage_roi = self._UNSET_STAGE_ROI
                if stage_roi is not None and len(self._perspective_stages) > stage_count_before:
                    top_b, bottom_b, left_b, right_b = self._perspective_stages[-1].bounds
                    if top_b < bottom_b and left_b < right_b:
                        rx, ry, rw, rh = normalize_alignment_roi(stage_roi, stage_input_shape)
                        stage_h, stage_w = result[0].shape[:2]
                        nx = max(0, rx - left_b)
                        ny = max(0, ry - top_b)
                        nx1 = min(stage_w, rx + rw - left_b)
                        ny1 = min(stage_h, ry + rh - top_b)
                        nw = nx1 - nx
                        nh = ny1 - ny
                        stage_roi = (nx, ny, nw, nh) if nw >= 8 and nh >= 8 else None
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
                           should_cancel=None, failure_policy='continue',
                           max_shift_x_ratio=None, max_shift_y_ratio=None,
                           max_rotation_deg=None, max_scale_change=None,
                           auto_crop=True, interpolation='lanczos') -> List[np.ndarray]:
        """
        单应性对齐配准（非线性）
        
        Args:
            input_source: 图像源
            output_path: 输出路径
        
        Returns:
            配准后的图像列表
        """
        roi = self.alignment_roi if self._active_stage_roi is self._UNSET_STAGE_ROI else self._active_stage_roi
        return _align_homography_impl(
            input_source, output_path, downscale_width=self.downscale_width,
            thread_count=thread_count, should_cancel=should_cancel, failure_policy=failure_policy,
            defer_warps=self.defer_warps and not self.use_gpu, plan_callback=self._perspective_stages.append,
            dispatch_callback=self._record_dispatch, accelerate_warp=self.accelerate_ecc,
            alignment_roi=roi,
            max_shift_x_ratio=max_shift_x_ratio, max_shift_y_ratio=max_shift_y_ratio,
            max_rotation_deg=max_rotation_deg, max_scale_change=max_scale_change,
            auto_crop=auto_crop, interpolation=interpolation)
    
    def _process_ecc(self, 
                    input_source: Union[str, List[np.ndarray]], 
                    output_path: Optional[str] = None,
                    thread_count: int = 4,
                    should_cancel=None, failure_policy='continue', *, stage_budget=None,
                    progress_callback=None,
                    max_shift_x_ratio=None, max_shift_y_ratio=None,
                    max_rotation_deg=None, max_scale_change=None,
                    auto_crop=True, interpolation='lanczos') -> List[np.ndarray]:
        """
        ECC对齐配准（高精度、亚像素级）
        
        Args:
            input_source: 图像源
            output_path: 输出路径
        
        Returns:
            配准后的图像列表
        """
        roi = self.alignment_roi if self._active_stage_roi is self._UNSET_STAGE_ROI else self._active_stage_roi
        try:
            with _serial_ecc_native_threads() if stage_budget is not None else nullcontext():
                return _align_ecc_impl(
                    input_source, output_path, downscale_width=self.downscale_width,
                    thread_count=thread_count, should_cancel=should_cancel, use_gpu=self.use_gpu,
                    failure_policy=failure_policy, defer_warps=self.defer_warps,
                    plan_callback=self._perspective_stages.append, accelerate_ecc=self.accelerate_ecc,
                    stage_budget=stage_budget, dispatch_callback=self._record_dispatch,
                    progress_callback=progress_callback,
                    alignment_roi=roi,
                    max_shift_x_ratio=max_shift_x_ratio, max_shift_y_ratio=max_shift_y_ratio,
                    max_rotation_deg=max_rotation_deg, max_scale_change=max_scale_change,
                    auto_crop=auto_crop, interpolation=interpolation)
        except RejectedAlignmentFrame as error:
            # Pair estimation may run on a pool thread. Record it in the caller's report.
            _report_alignment(error.frame_index, 'ecc', 'failed', reason=error.reason,
                              reference_index=error.frame_index - 1)
            raise


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
