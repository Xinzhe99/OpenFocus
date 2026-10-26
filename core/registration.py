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
import io
from typing import Union, List, Optional

from utils.sorting import natural_sort_key as _image_sort_key

# Windowed/Windows console mojibake fix — frozen builds only. Rebinding
# sys.stdout at import time closes the ORIGINAL wrapper once it is collected
# (TextIOWrapper.__del__ closes its buffer), which under pytest silently kills
# the terminal output mid-session (the ubuntu CI job failed exactly there) and
# made any redirected run lose its report tail.
if getattr(sys, "frozen", False) and sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# # ========== 缩放对齐算法实现（线性） ==========



# def _align_zoom_impl(input_source, output_path=None, img_filenames=None):
#     """
#     Align images from directory path or image list
#     Args:
#         input_source: string (directory path) or list of images
#         output_path: string, directory path to save results (optional, None means no saving)
#     Returns:
#         list of aligned images (always returns processed images regardless of output_path)
#     """
#     # Process input source
#     if img_filenames is None and isinstance(input_source, str):
#         # Pre-compile regular expression
#         num_pattern = re.compile(r"\d+")
#         # Use generator expression and list comprehension for optimized file filtering and sorting
#         img_paths = sorted(
#             (os.path.join(input_source, f) for f in os.listdir(input_source)
#              if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp', '.tif', '.tiff', '.webp'}),
#             key=lambda x: int(num_pattern.findall(os.path.basename(x))[-1])
#         )
#         # Use list comprehension to read all images at once
#         images = [cv2.imread(path) for path in img_paths]
#         # Store original filenames with extensions
#         img_filenames = [os.path.basename(path) for path in img_paths]
#     else:
#         images = input_source

#     # Cache image dimensions and feature detector
#     img_first = images[0]
#     img_last = images[-1]
#     h_first, w_first = img_first.shape[:2]
#     h_last, w_last = img_last.shape[:2]

#     # SIFT feature detection and matching
#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(img_first, None)
#     kp2, des2 = sift.detectAndCompute(img_last, None)

#     # Use BFMatcher for feature matching (SIFT uses L2 norm)
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

#     # Find valid matching points
#     min_distance_threshold = w_first / 2
#     p1 = p2 = q1 = q2 = None

#     # Use numpy array operations to optimize distance calculation
#     for i in range(len(matches) - 1):
#         p1 = np.array(kp1[matches[i].queryIdx].pt)
#         p2 = np.array(kp1[matches[i + 1].queryIdx].pt)
#         q1 = np.array(kp2[matches[i].trainIdx].pt)
#         q2 = np.array(kp2[matches[i + 1].trainIdx].pt)

#         if np.linalg.norm(p1 - p2) > min_distance_threshold:
#             break
#     else:
#         raise ValueError("Unable to find matching point pairs that meet the criteria")

#     # Calculate scaling factor
#     dist = np.linalg.norm(p1 - p2)
#     dist2 = np.linalg.norm(q1 - q2)
#     c = max(dist2 / dist, dist / dist2)

#     # Pre-calculate all scaling factors
#     aug_list = np.linspace(1, c, len(images))
#     is_forward = dist2 / dist > 1
#     target_shape = (h_last, w_last) if is_forward else (h_first, w_first)
#     aug_factors = aug_list[::-1] if is_forward else aug_list

#     # Pre-create output directory
#     if output_path:
#         os.makedirs(output_path, exist_ok=True)

#     # Process images
#     aligned_images = []
#     target_h, target_w = target_shape

#     for idx, (img, aug_factor) in enumerate(zip(images, aug_factors)):
#         # Use cv2.INTER_AREA for scaling
#         img_resized = cv2.resize(img, None, fx=aug_factor, fy=aug_factor, interpolation=cv2.INTER_AREA)

#         # Calculate crop position
#         h, w = img_resized.shape[:2]
#         x = (w - target_w) // 2
#         y = (h - target_h) // 2

#         # Crop image
#         img_crop = img_resized[y:y + target_h, x:x + target_w]
#         aligned_images.append(img_crop)

#         # Save results
#         if output_path:
#             # Use original filename if available, otherwise use default naming
#             if img_filenames:
#                 output_file = os.path.join(output_path, img_filenames[idx])
#             else:
#                 output_file = os.path.join(output_path, f'frame_{idx:04d}.png')
#             cv2.imwrite(output_file, img_crop)

#     return aligned_images


# ========== 基于变换矩阵的精确裁切函数 ==========

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


def _crop_with_transforms(images, H_matrices):
    """
    基于变换矩阵精确裁切图像
    
    Args:
        images: 图像列表
        H_matrices: 变换矩阵列表
    
    Returns:
        裁切后的图像列表
    """
    if not images or len(images) == 0:
        return images
    
    h, w = images[0].shape[:2]
    top, bottom, left, right = _compute_valid_region_from_transforms(H_matrices, (h, w))
    
    # 确保裁切区域有效
    if top >= bottom or left >= right:
        print("Warning: Invalid crop region, returning original images.")
        return images
    
    # 裁切所有图像
    cropped_images = []
    for img in images:
        cropped = img[top:bottom, left:right].copy()
        cropped_images.append(cropped)
    
    print(f"Cropped by transform matrices: ({left}, {top}) to ({right}, {bottom}), new size: {right-left}x{bottom-top}")
    
    return cropped_images


# ========== 单应性对齐算法实现（非线性） ==========


def _align_homography_impl(input_source, output_path=None, img_filenames=None, downscale_width=1600, thread_count: int = 4, should_cancel=None):
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
                divisor = float(np.max(img_small)) if float(np.max(img_small)) > 255.0 else 1.0
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
        for idx, (curr_kps, curr_des, curr_scale) in enumerate(features, 1):
            _check_cancel(should_cancel)
            if curr_des is None or len(curr_kps) < 4 or last_des is None:
                print(f"Warning: Frame {idx} features insufficient. Keeping original position.")
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
                H_matrices.append(H_global.copy())
                continue

            pts_curr = np.float32([curr_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / curr_scale
            pts_last = np.float32([last_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) / last_scale
            H_local, mask = cv2.findHomography(pts_curr, pts_last, cv2.RANSAC, 5.0)
            _check_cancel(should_cancel)
            if H_local is None:
                print(f"Frame {idx} alignment failed.")
                H_local = np.eye(3)
            H_global = np.matmul(H_global, H_local)
            H_matrices.append(H_global.copy())
            last_kps, last_des, last_scale = curr_kps, curr_des, curr_scale
    del last_kps, last_des, curr_kps, curr_des

    # --- 4. 并行应用变换与裁切 ---
    print("  - Step 3/3: Warping images concurrently...")
    
    # 预先计算裁切区域，直接变换到目标区域，避免先变换后裁切的浪费
    top, bottom, left, right = _compute_valid_region_from_transforms(H_matrices, (h_orig, w_orig))
    
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
        aligned_images = list(results)
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

    def submit_one():
        _check_cancel(should_cancel)
        try:
            value = next(source)
        except StopIteration:
            return
        pending.append(pool.submit(run, value))

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



# ========== OpenFocus 对齐：有界相似变换（平移+旋转+缩放） ==========

_OF_EST_WIDTH = 480  # 估计阶段长边上限（旋转/缩放网格在低分辨率上搜索）
# 位移上限（帧尺寸比例）：较大的位移估计不可靠，强行校正会得到半错位
# 并让有效区交集被裁掉一大块（25% 上限 = 单边最多丢 1/4 画面）。取 8%——
# 本对齐器有 NCC 置信门限，可用 ImageRegistration(max_shift_ratio=) /
# CLI --align-shift-limit 按需放宽。
_OF_MAX_SHIFT_RATIO = 0.08
# 旋转/放大率容差：Helicon 把它们与位移分开暴露为独立参数，我们的实现
# 此前只有位移上限，旋转和缩放可以取到搜索边界（8° / ±5%）而不被拒——
# 一帧被误估出 5° 旋转会以角点错位 0.087·对角线的方式污染整幅画面。
_OF_MAX_ROTATION_DEG = 8.0
_OF_MAX_SCALE_DELTA = 0.05


def _of_gray_est(img, est_width):
    """灰度 + 降采样到估计分辨率（估计分辨率以长边计）。

    这里曾额外减去 σ2.5 低通（函数旧名 `_of_gray_hp`）。高通会丢掉
    低频，而"清晰 vs 失焦"帧对的相关性主要就在低频——实测 σ11 失焦时，
    高通输入把 16 px 位移估成误差 22 px，普通灰度误差 0.27 px。旋转/
    缩放的判别评分同样不依赖高频。
    """
    if img.ndim == 2:
        g = img.astype(np.float32)
    else:
        take = min(img.shape[2], 3)
        if take == 1:
            g = img[:, :, 0].astype(np.float32)
        else:
            m = np.array([0.114, 0.587, 0.299], np.float32)
            g = cv2.transform(img[:, :, :take].astype(np.float32),
                              m.reshape(1, -1))
            g = g if g.ndim == 2 else g[:, :, 0]
    h, w = g.shape[:2]
    if max(h, w) > est_width:
        s = est_width / float(max(h, w))
        g = cv2.resize(g, (max(16, int(round(w * s))),
                           max(16, int(round(h * s)))),
                       interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(g, np.float32)


def _of_translation_ncc(ref_est, cur_est, patch_frac=0.7):
    """Translation cur→ref by normalized cross-correlation.

    Phase correlation silently fails on "sharp vs defocused" pairs: measured
    on a synthetic stack with a true 16 px shift it reported 145 px, while
    NCC found 15.93 px. Focus stacking always compares differently blurred
    frames, so the translation estimate must not rely on the phase peak.

    A central patch of `cur_est` is matched inside `ref_est`; the returned
    shift is where the patch content moved. Sub-pixel peak by parabola.

    Take the global peak as-is (no tie-breaking): a defocused frame smears
    the correlation peak into a broad plateau, and preferring the smallest
    displacement among near-ties then walks off the true peak (σ9 blur:
    peak 0.805 at the correct −16 px, but shifts within 0.01 of it span
    several pixels, and "smallest" picked +5 px). Restricting the search
    window is just as harmful: when the true shift lies outside it, the
    best remaining in-window peak is a spurious match that silently
    translates the frame. The caller owns the acceptance range instead.
    """
    h, w = ref_est.shape[:2]
    ph = max(16, int(h * patch_frac))
    pw = max(16, int(w * patch_frac))
    y0, x0 = (h - ph) // 2, (w - pw) // 2
    patch = np.ascontiguousarray(cur_est[y0:y0 + ph, x0:x0 + pw])
    # 平坦块（全黑/全白的边框帧、无纹理区域）：NCC 无定义，模板匹配会
    # 在角落给出 1.0 的假峰。报告无对应关系，由调用方回退到恒等。
    if float(patch.std()) < 1e-3 or float(ref_est.std()) < 1e-3:
        return -1.0, 0.0, 0.0
    try:
        res = cv2.matchTemplate(ref_est, patch, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return -1.0, 0.0, 0.0
    px = int(np.argmax(res.max(axis=0)))
    py = int(np.argmax(res[:, px]))

    def _parabola(v0, v1, v2):
        den = v0 - 2.0 * v1 + v2
        return 0.0 if abs(den) < 1e-9 else 0.5 * (v0 - v2) / den

    sx = sy = 0.0
    if 0 < px < res.shape[1] - 1:
        sx = float(np.clip(_parabola(res[py, px - 1], res[py, px],
                                     res[py, px + 1]), -1.0, 1.0))
    if 0 < py < res.shape[0] - 1:
        sy = float(np.clip(_parabola(res[py - 1, px], res[py, px],
                                     res[py + 1, px]), -1.0, 1.0))
    # Content moved from (x0, y0) in cur to (px, py) in ref.
    dx = (x0 + sx) - px
    dy = (y0 + sy) - py
    return float(res[py, px]), dx, dy


def _of_ncc_score(ref_est, fixed, patch_frac=0.7):
    """重叠区归一化互相关：旋转/缩放候选的判别评分。

    平移用 NCC 模板搜索（相位相关在清晰/失焦帧对上有假峰，见
    `_of_translation_ncc`）；NCC 在正确校正时 →1，旋转错 0.5° 以上
    急剧下降。返回 (ncc, dx, dy)。
    """
    ncc, dx, dy = _of_translation_ncc(ref_est, fixed, patch_frac=patch_frac)
    if not np.isfinite(ncc):
        return -1.0, 0.0, 0.0
    return ncc, dx, dy


def _of_estimate_similarity(ref_est, cur_est, patch_frac=0.7,
                            rot_bound=8.0, scale_range=(0.95, 1.05),
                            should_cancel=None):
    """有界网格搜索 cur→ref 的相似变换（旋转+缩放），NCC 评分。

    返回 (ncc, theta_deg, scale, dx, dy)：
    theta/scale 满足 warpAffine(cur, getRotationMatrix2D(center, theta,
    scale)) ≈ ref（内容层面），(dx, dy) 是残差平移（内容仍需移动的量，
    由 `_of_translation_ncc` 的 NCC 峰给出）。
    """
    h, w = ref_est.shape[:2]
    cy, cx = h / 2.0, w / 2.0

    def _eval(theta, scale):
        _check_cancel(should_cancel)
        M = cv2.getRotationMatrix2D((cx, cy), theta, scale)
        fixed = cv2.warpAffine(cur_est, M, (w, h))
        result = _of_ncc_score(ref_est, fixed, patch_frac=patch_frac)
        _check_cancel(should_cancel)
        return result

    best = (-1.0, 0.0, 1.0, 0.0, 0.0)
    # 平移-only 基准：模糊与缩放在 NCC 上天然相似，无显著增益时坚持
    # s=1 的简约解（否则失焦帧会被误判出假旋转/假缩放）
    base_ncc, base_dx, base_dy = _eval(0.0, 1.0)
    best = (base_ncc, 0.0, 1.0, base_dx, base_dy)
    for th in np.arange(-rot_bound, rot_bound + 1e-6, 1.0):
        for sc in (0.95, 1.0, 1.05):
            if not (scale_range[0] - 1e-9 <= sc <= scale_range[1] + 1e-9):
                continue
            ncc, dx, dy = _eval(th, sc)
            if ncc > best[0]:
                best = (ncc, th, sc, dx, dy)
    # 非平凡 (θ, s) 必须显著优于平移-only 基准（≥0.05 NCC）才采纳
    if (best[1] != 0.0 or best[2] != 1.0) and best[0] < base_ncc + 0.05:
        best = (base_ncc, 0.0, 1.0, base_dx, base_dy)
    if best[1] == 0.0 and best[2] == 1.0 and best[0] >= 0.97:
        # 常见情形：纯平移已高度吻合——跳过两级精修（省一半耗时）
        return best
    # 两级精修：角度 0.25°、缩放 0.01
    for _ in range(2):
        th0, sc0 = best[1], best[2]
        improved = False
        for th in np.arange(max(-rot_bound, th0 - 1.0),
                            min(rot_bound, th0 + 1.0) + 1e-9, 0.25):
            for sc in np.arange(max(scale_range[0], sc0 - 0.02),
                                min(scale_range[1], sc0 + 0.02) + 1e-9, 0.01):
                ncc, dx, dy = _eval(th, sc)
                if ncc > best[0] + 1e-6:
                    best = (ncc, th, sc, dx, dy)
                    improved = True
        if not improved:
            break
    # 精修后复检：非平凡解仍须显著优于平移-only 基准
    if (best[1] != 0.0 or best[2] != 1.0) and best[0] < base_ncc + 0.05:
        best = (base_ncc, 0.0, 1.0, base_dx, base_dy)
    return best


def _align_openfocus_impl(input_source, output_path=None, img_filenames=None,
                          downscale_width=None, thread_count: int = 4,
                          should_cancel=None, equalize_brightness=True,
                          max_shift_ratio=None, brightness_reference="first",
                          max_rotation_deg=None, max_scale_delta=None):
    """
    OpenFocus 快速对齐：有界相似变换（平移+旋转+缩放）+ 亮度均衡 + 自动裁边。

    几何参考帧是**首帧**（固定锚点）。旋转/缩放用有界
    网格搜索（NCC 判别评分），平移用 NCC 模板搜索的亚像素峰（相位相关
    在清晰/失焦帧对上有假峰）；低置信帧放弃该帧对齐；位移超过
    max_shift_ratio 时也放弃（强行钳制会得到一半错位 + 污染有效区），
    最后按有效区交集自动裁边。
    """
    if max_shift_ratio is None:
        max_shift_ratio = _OF_MAX_SHIFT_RATIO
    if max_rotation_deg is None:
        max_rotation_deg = _OF_MAX_ROTATION_DEG
    if max_scale_delta is None:
        max_scale_delta = _OF_MAX_SCALE_DELTA
    if img_filenames is None and isinstance(input_source, str):
        valid_exts = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        img_paths = sorted(
            (os.path.join(input_source, f) for f in os.listdir(input_source)
             if os.path.splitext(f)[1].lower() in valid_exts),
            key=_image_sort_key
        )
        images = [cv2.imread(path) for path in img_paths]
        img_filenames = [os.path.basename(path) for path in img_paths]
    else:
        images = input_source
    if images is None or len(images) < 2:
        return images if images is not None else []

    n = len(images)
    h, w = images[0].shape[:2]
    max_shift = (int(w * max_shift_ratio), int(h * max_shift_ratio))
    est_width = min(_OF_EST_WIDTH, max(320, w))

    def _gray_float(img):
        if img.ndim == 2:
            g = img
        else:
            take = min(img.shape[2], 3)
            if take == 1:
                g = img[:, :, 0]
            else:
                m = np.array([0.114, 0.587, 0.299], np.float32)
                g = cv2.transform(img[:, :, :take].astype(np.float32),
                                  m.reshape(1, -1))
                g = g if g.ndim == 2 else g[:, :, 0]
        return g.astype(np.float32)

    # Geometry is always estimated against the FIRST frame: one fixed anchor
    # keeps the valid-region crop minimal and the per-frame shift budget
    # independent of the exposure target. The brightness target is separate
    # (the fusion pipeline calibrates to the LAST frame by default), so a
    # stack is not pulled toward two different ends.
    ref_idx = 0
    bright_gray = (_gray_float(images[-1])
                   if brightness_reference == "last" else _gray_float(images[0]))
    ref_low = cv2.GaussianBlur(bright_gray, (0, 0), 32.0)
    ref_low_mean = float(ref_low.mean()) or 1.0
    del bright_gray, ref_low
    gains = [1.0] * n

    ref_est = _of_gray_est(images[ref_idx], est_width)
    eh, ew = ref_est.shape[:2]
    patch_frac = min(0.7, max(0.34, 1.0 - 2.0 * float(max_shift_ratio) - 0.02))

    def _prepare(i):
        """Everything about frame i that does not depend on the state machine.

        Greyscale, the σ32 brightness low-pass and the estimation-resolution
        grey are per-frame and read-only, so they are computed for a bounded
        window in parallel. The acceptance logic below still consumes the
        results strictly in frame order.
        """
        gain = 1.0
        if equalize_brightness:
            cur_gray = _gray_float(images[i])
            cur_low = cv2.GaussianBlur(cur_gray, (0, 0), 32.0)
            cur_mean = float(cur_low.mean())
            if cur_mean > 1e-3:
                gain = float(np.clip(ref_low_mean / cur_mean, 0.5, 2.0))
        return i, gain, _of_gray_est(images[i], est_width)

    def _compose_full(ncc_pack):
        """估计分辨率下的校正参数 → 全分辨率 3x3（warpAffine 直接可用）。

        warpAffine 的 M 是内容正向变换（实测：[I|+10] 把内容右移 10px；
        getRotationMatrix2D(θ,s) 的搜索语义已在网格中验证）。完整校正 =
        先施加旋转/缩放 Rs，再把内容平移 −d（d 为 NCC 峰给出的残差内容
        位移）。矩阵复合：M_total = [I|−d]·Rs_full；d 从估计分辨率按
        比例换算回全分辨率。
        """
        _ncc, theta, scale, dx, dy = ncc_pack
        Rs = np.eye(3)
        Rs[:2, :] = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), theta, scale)
        sx, sy = ew / float(w), eh / float(h)
        d_full = np.array([-dx / sx, -dy / sy])
        M = np.eye(3)
        M[:2, :2] = Rs[:2, :2]
        M[:2, 2] = Rs[:2, :2] @ d_full + Rs[:2, 2]
        return M

    # 死区：低于这些分量视为无运动（固定机位栈的噪声保护）
    dead_t, dead_rot, dead_scale = 0.15, 0.05, 1e-3
    min_ncc = 0.06          # 低于此视为无可靠对应关系
    chain_ncc = 0.10        # 对参考帧置信不足 → 链到上一帧重估

    corrections = [None] * n  # cur→ref 的校正矩阵；参考帧为 None
    corrections[ref_idx] = np.eye(3, dtype=np.float64)
    prev_est = ref_est
    print("Aligning %d images using OpenFocus (reference: first frame)..." % n)
    with _bounded_registration_map(_prepare, range(n), max(1, int(thread_count)),
                                   should_cancel) as prepared:
        for i, gain, cur_est in prepared:
            _check_cancel(should_cancel)
            # 亮度标定对参考帧同样生效：几何锚点在前，曝光锚点可以在另一端
            # （融合默认标定到末帧），参考帧的 gain 不是恒等。
            gains[i] = gain
            if i == ref_idx:
                continue

            # 模板必须给可接受的位移留出搜索余量：patch_frac=0.7 最多覆盖
            # 单边 15% 的位移；用户把上限放宽到 25% 时若不缩小模板，真位移
            # 落在搜索窗外，估计器只能返回窗内的伪峰（并真的平移该帧）。
            est = _of_estimate_similarity(ref_est, cur_est,
                                          patch_frac=patch_frac,
                                          should_cancel=should_cancel)
            chained = False
            if est[0] < chain_ncc and i >= 2:
                # 失焦/弱纹理帧：与相邻帧（模糊程度接近）更相关——链式估计
                est2 = _of_estimate_similarity(prev_est, cur_est,
                                               patch_frac=patch_frac,
                                               should_cancel=should_cancel)
                if est2[0] > est[0] + 0.02:
                    M = corrections[i - 1] @ _compose_full(est2)
                    chained = True
                    est = (est2[0],) + est[1:]
            if not chained:
                M = _compose_full(est)

            ncc = est[0]
            if ncc < min_ncc:
                M = np.eye(3)
            # 平移分量取“画面中心的净位移”：矩阵的 M[:,2] 还包含绕中心旋转/缩放
            # 诱导的项（480×360 上 2.5° 就有 ~8-11px），直接拿它判上限会把合法
            # 的旋转校正误杀成“位移超限”。t_c = b + (A − I)·c。
            A2 = M[:2, :2]
            tc = M[:2, 2] + A2 @ np.array([w / 2.0, h / 2.0]) - np.array(
                [w / 2.0, h / 2.0])
            tx = float(tc[0])
            ty = float(tc[1])
            # 位移上限：超限**放弃该帧对齐**——较大的位移估计不可靠，
            # 旧的"钳制到上限"会把大位移校正成半错位，还让有效区交集被裁掉
            # 一大块（25% 上限意味着单边最多裁 1/4 画面）。
            over = (abs(tx) > max_shift[0] or abs(ty) > max_shift[1])
            # 旋转/放大率各自独立设限（Helicon 的 Rotation / Magnification
            # adjustment）：搜索范围本身就是这两个值，超出即说明估计不可信，
            # 与位移超限同样放弃整帧而不是半校正。
            rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
            scale_now = float(np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])))
            if (abs(rot_deg) > max_rotation_deg + 1e-6
                    or abs(scale_now - 1.0) > max_scale_delta + 1e-6):
                over = True
            if over:
                M = np.eye(3)
                tx = ty = 0.0
                rot_deg, scale_now = 0.0, 1.0
            # 死区：整体运动可忽略 → 恒等（不产生裁边/插值损失）
            if (abs(tx) < dead_t and abs(ty) < dead_t
                    and abs(rot_deg) < dead_rot
                    and abs(scale_now - 1.0) < dead_scale):
                M = np.eye(3)
            corrections[i] = M
            prev_est = cur_est
            print("  frame %d/%d: rot %+.2f° scale %.3f shift (%+.2f, %+.2f)"
                  " ncc %.2f%s%s" % (i + 1, n, rot_deg, scale_now, tx, ty, ncc,
                                     " (chained)" if chained else "",
                                     " (DROPPED: shift over %.0f%% of frame)"
                                     % (100.0 * max_shift_ratio) if over else ""))

    def _warp(args):
        idx, img, M, gain = args
        warped = cv2.warpAffine(img, M[:2, :], (w, h),
                                flags=cv2.INTER_LANCZOS4
                                | cv2.WARP_FILL_OUTLIERS,
                                borderMode=cv2.BORDER_REFLECT)
        if gain != 1.0:
            # 亮度均衡同步应用到输出帧：均衡后的帧进入融合，消除曝光漂移
            warped = np.clip(warped.astype(np.float32) * gain, 0,
                             255 if img.dtype == np.uint8 else 65535)
            warped = warped.astype(img.dtype)
        return idx, warped

    aligned = [None] * n
    if gains[ref_idx] == 1.0:
        aligned[ref_idx] = images[ref_idx]
    else:
        ref = images[ref_idx]
        scaled = np.clip(ref.astype(np.float32) * gains[ref_idx], 0,
                         255 if ref.dtype == np.uint8 else 65535)
        aligned[ref_idx] = scaled.astype(ref.dtype)
    warp_args = ((i, images[i], corrections[i], gains[i])
                 for i in range(n) if i != ref_idx)
    with _bounded_registration_map(_warp, warp_args, max(1, int(thread_count)),
                                   should_cancel) as results:
        for idx, warped in results:
            aligned[idx] = warped

    # 有效区 = 各帧"正向映射矩形"（corrections 的逆）的交集内接框：
    # dst(x) = cur(A·x) 有效 ⟺ A·x ∈ 源图 ⟺ x ∈ A⁻¹(源图矩形)
    fwd = [np.linalg.inv(M) if M is not None else None for M in corrections]
    top, bottom, left, right = _compute_valid_region_from_transforms(
        fwd, (h, w))
    if top >= bottom or left >= right:
        top, bottom, left, right = 0, h, 0, w
    aligned = [img[top:bottom, left:right] for img in aligned]

    _save_registration(aligned, output_path, img_filenames, should_cancel)
    return aligned

# ========== ECC对齐算法实现（高精度） ==========

def _align_ecc_impl(input_source, output_path=None, img_filenames=None, downscale_width=1000, thread_count: int = 4, should_cancel=None, use_gpu: bool = False):
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
        preprocessed_data = list(results)

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
            print(f"Warning: ECC failed to converge at frame {idx}. Assuming no motion.")
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)
        return warp_matrix

    with _bounded_registration_map(_compute_pair_transform,
                                   range(1, len(images)), max_workers,
                                   should_cancel) as results:
        pair_transforms = list(results)
    del preprocessed_data

    for idx, warp_matrix in enumerate(pair_transforms, start=1):
        _check_cancel(should_cancel)
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

    # --- 5. 并行应用变换与裁切 ---
    # print("  - Step 3/3: Warping and saving concurrently...")

    # 计算公共有效区域
    top, bottom, left, right = _compute_valid_region_from_transforms(H_matrices, (h_orig, w_orig))
    
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
            for c in range(img.shape[2]):
                # order=1 (linear) 速度最快, order=3 (cubic) 质量更好
                # 这里使用 order=1 以获得最大加速，如果追求质量可用 order=3
                channel_out = cupyx.scipy.ndimage.map_coordinates(
                    img_gpu[:, :, c], 
                    cp.stack([src_y, src_x]), 
                    order=1, 
                    mode='constant', 
                    cval=0
                )
                channels.append(channel_out)
            
            aligned_img_gpu = cp.stack(channels, axis=2)
            
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
        aligned_images = list(results)
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
        stabilized_images.append(frame_stabilized)

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


# # ========== 组合配准算法实现 ==========

# def _registration_impl(input_source, output_path=None):
#     """
#     组合配准算法内部实现
#     Args:
#         input_source: string (directory path) or list of images
#         output_path: string, directory path to save results (optional, None means no saving)
#     Returns:
#         list of registered images (always returns processed images regardless of output_path)
#     """
#     # Store original filenames if input is a directory
#     img_filenames = None
#     if isinstance(input_source, str):
#         num_pattern = re.compile(r"\d+")
#         img_paths = sorted(
#             (os.path.join(input_source, f) for f in os.listdir(input_source)
#              if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.jpe', '.jfif', '.png', '.bmp', '.tif', '.tiff', '.webp'}),
#             key=lambda x: int(num_pattern.findall(os.path.basename(x))[-1])
#         )
#         img_filenames = [os.path.basename(path) for path in img_paths]
    
#     # Step 1: Coarse alignment using zoom
#     zoom_aligned_images = _align_zoom_impl(input_source)
    
#     # Step 2: Fine registration using stabilisation
#     final_images = _stabilisation_impl(zoom_aligned_images, output_path, img_filenames)
    
#     return final_images


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
                 use_gpu=False, parameters=None):
        from core.algorithms import alignment_steps, resolve_registration_parameters
        self.method = method
        self.steps = alignment_steps(method)
        values = resolve_registration_parameters(method, parameters)
        if max_shift_ratio is not None:
            values['max_shift_ratio'] = max_shift_ratio
        self.parameters = resolve_registration_parameters(method, values)
        self.downscale_width = int(downscale_width) if downscale_width is not None else 1024
        self.brightness_reference = brightness_reference if brightness_reference in ('first', 'last') else 'first'
        self.use_gpu = bool(use_gpu)
    
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
        from core.resource_policy import Workload, heavy_job, plan_execution, require_memory

        with heavy_job(should_cancel):
            _check_cancel(should_cancel)
            filenames = None
            images = input_source
            if isinstance(input_source, str):
                # Header inspection admits the decoded stack before allocation;
                # cv2.imread retains the directory API's existing color contract.
                from PIL import Image
                extensions = {'.jpg', '.jpeg', '.jpe', '.jfif', '.png',
                              '.bmp', '.tif', '.tiff', '.webp'}
                paths = sorted((os.path.join(input_source, name)
                                for name in os.listdir(input_source)
                                if os.path.splitext(name)[1].lower() in extensions),
                               key=_image_sort_key)
                decoded_bytes = 0
                largest_frame = 0
                for path in paths:
                    _check_cancel(should_cancel)
                    with Image.open(path) as header:
                        frame_bytes = header.width * header.height * 3
                    decoded_bytes += frame_bytes
                    largest_frame = max(largest_frame, frame_bytes)
                if paths:
                    require_memory(decoded_bytes + largest_frame * 2)
                images = []
                for path in paths:
                    _check_cancel(should_cancel)
                    image = cv2.imread(path)
                    _check_cancel(should_cancel)
                    if image is None:
                        raise ValueError(f"Could not decode registration frame: {path}")
                    images.append(image)
                filenames = [os.path.basename(path) for path in paths]

            if images is None or len(images) < 2:
                _check_cancel(should_cancel)
                return images if images is not None else []
            if not self.steps:
                _save_registration(images, output_path, filenames, should_cancel)
                return images
            height, width = images[0].shape[:2]
            channels = images[0].shape[2] if images[0].ndim == 3 else 1
            output_bytes = sum(image.nbytes for image in images)
            # ECC retains its normalized thumbnails until pair estimation ends.
            # Full-resolution planes are a conservative bound even with resize.
            gray_bytes = sum(image.shape[0] * image.shape[1] * 4 for image in images)
            plan = plan_execution(Workload(
                algorithm='registration', frames=len(images), height=height,
                width=width, channels=channels, itemsize=images[0].dtype.itemsize,
                fixed_bytes=output_bytes * min(2, len(self.steps)) + gray_bytes,
                per_worker_bytes=max(image.size for image in images) * 4 * 16,
                outer_tasks=len(images), parallel_layer='outer'),
                preference=resource_preference, user_threads=thread_count)
            workers = min(plan.workers, plan.max_in_flight)
            from core.algorithms import get_registration, resolve_runner
            result = images
            for step in self.steps:
                _check_cancel(should_cancel)
                spec = get_registration(step)
                parameters = {item.key: self.parameters[item.key] for item in spec.parameters}
                result = resolve_runner(spec.runner)(self, result,
                    thread_count=workers, should_cancel=should_cancel, **parameters)
                _check_cancel(should_cancel)
            _save_registration(result, output_path, filenames, should_cancel)
            return result
    
    def _process_homography(self, 
                           input_source: Union[str, List[np.ndarray]], 
                           output_path: Optional[str] = None,
                           thread_count: int = 4,
                           should_cancel=None) -> List[np.ndarray]:
        """
        单应性对齐配准（非线性）
        
        Args:
            input_source: 图像源
            output_path: 输出路径
        
        Returns:
            配准后的图像列表
        """
        return _align_homography_impl(input_source, output_path, downscale_width=self.downscale_width, thread_count=thread_count, should_cancel=should_cancel)
    
    def _process_ecc(self, 
                    input_source: Union[str, List[np.ndarray]], 
                    output_path: Optional[str] = None,
                    thread_count: int = 4,
                    should_cancel=None) -> List[np.ndarray]:
        """
        ECC对齐配准（高精度、亚像素级）
        
        Args:
            input_source: 图像源
            output_path: 输出路径
        
        Returns:
            配准后的图像列表
        """
        return _align_ecc_impl(input_source, output_path, downscale_width=self.downscale_width, thread_count=thread_count, should_cancel=should_cancel, use_gpu=self.use_gpu)


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


# ========== 向后兼容的函数别名 ==========

# def image_stack_align_zoom(input_source, output_path=None):
#     """向后兼容的函数别名"""
#     return _align_zoom_impl(input_source, output_path)

# def image_stack_stabilisation(input_source, output_path=None, filenames=None):
#     """向后兼容的函数别名"""
#     return _stabilisation_impl(input_source, output_path, filenames)

# def image_stack_registration(input_source, output_path=None):
#     """向后兼容的函数别名"""
#     return _registration_impl(input_source, output_path)

# def process_image_stack(input_path, output_path=None):
#     """向后兼容的函数别名"""
#     return _registration_impl(input_path, output_path)

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
