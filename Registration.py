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
import re
import time
import cv2
import numpy as np
import os
import glob
import sys
import io
from typing import Union, List, Optional

# Set standard output encoding to utf-8 when a console stream exists
if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
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
#              if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}),
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

def _align_homography_impl(input_source, output_path=None, img_filenames=None, downscale_width=1600, thread_count: int = 4):
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
        num_pattern = re.compile(r"\d+")
        # 支持常见格式，过滤非图片
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        img_paths = sorted(
            (os.path.join(input_source, f) for f in os.listdir(input_source)
             if os.path.splitext(f)[1].lower() in valid_exts),
            key=lambda x: int(num_pattern.findall(os.path.basename(x))[-1]) if num_pattern.findall(os.path.basename(x)) else x
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
        kps, des = local_detector.detectAndCompute(img_small, None)
        return kps, des, scale

    # 使用线程池并行提取特征
    # OpenCV 的大部分操作释放 GIL，因此多线程可以有效加速
    try:
        max_workers = max(1, int(thread_count))
    except Exception:
        max_workers = min(8, os.cpu_count() or 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        features_list = list(executor.map(get_features_task, images))

    # --- 3. 序列矩阵计算 (必须串行) ---
    print("  - Step 2/3: Calculating transform matrices...")
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    # 获取第一帧特征
    last_kps, last_des, last_scale = features_list[0]

    for idx in range(1, num_images):
        curr_kps, curr_des, curr_scale = features_list[idx]

        # 异常处理：特征不足
        if curr_des is None or len(curr_kps) < 4 or last_des is None:
            print(f"Warning: Frame {idx} features insufficient. Keeping original position.")
            H_matrices.append(H_global.copy())
            continue

        # 特征匹配 (Current vs Last)
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

        # 提取坐标
        pts_curr = np.float32([curr_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) / curr_scale
        pts_last = np.float32([last_kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) / last_scale

        # RANSAC 计算单应性矩阵
        H_local, mask = cv2.findHomography(pts_curr, pts_last, cv2.RANSAC, 5.0)

        if H_local is None:
            print(f"Frame {idx} alignment failed.")
            H_local = np.eye(3)

        # 矩阵链乘
        H_global = np.matmul(H_global, H_local)
        H_matrices.append(H_global.copy())

        # 更新引用
        last_kps = curr_kps
        last_des = curr_des
        last_scale = curr_scale

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

    # 准备参数
    warp_args = zip(images, H_matrices)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        aligned_images = list(executor.map(warp_task, warp_args))

    # 如果有输出路径，保存图像
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        print("  - Saving results...")
        for idx, img in enumerate(aligned_images):
            fname = img_filenames[idx] if img_filenames else f'frame_{idx:04d}.png'
            cv2.imwrite(os.path.join(output_path, fname), img)

    return aligned_images


# ========== ECC对齐算法实现（高精度） ==========

def _align_ecc_impl(input_source, output_path=None, img_filenames=None, downscale_width=1000, thread_count: int = 4):
    """
    基于 ECC (增强相关系数) 的高精度图像栈对齐算法
    适用于：显微摄影、微距摄影中伴随呼吸效应的图像栈
    优势：亚像素精度，自动处理缩放中心偏移，不依赖特征点
    优化：并行预处理、并行变换、合并裁切操作
    """
    import concurrent.futures

    # --- 1. 数据加载 ---
    if img_filenames is None and isinstance(input_source, str):
        num_pattern = re.compile(r"\d+")
        img_paths = sorted(
            (os.path.join(input_source, f) for f in os.listdir(input_source)
             if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}),
            key=lambda x: int(num_pattern.findall(os.path.basename(x))[-1]) if num_pattern.findall(os.path.basename(x)) else x
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

    # 预处理函数：转灰度 + 降采样 + 高斯模糊
    def preprocess(img):
        h, w = img.shape[:2]
        scale = downscale_width / float(w) if w > downscale_width else 1.0
        if scale < 1.0:
            small_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small_img = img # 引用即可，无需拷贝
        
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray, scale

    print(f"Aligning {len(images)} images using ECC (Parallel Optimized)...")

    # --- 3. 并行预处理 ---
    # print("  - Step 1/3: Preprocessing images concurrently...")
    try:
        max_workers = max(1, int(thread_count))
    except Exception:
        max_workers = min(8, os.cpu_count() or 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map 保证结果顺序与输入一致
        preprocessed_data = list(executor.map(preprocess, images))

    # 准备第一张图作为参考
    last_gray, scale_factor = preprocessed_data[0]

    # 预先创建输出目录
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    # --- 4. 逐帧计算矩阵 (必须串行) ---
    # print("  - Step 2/3: Calculating ECC matrices...")
    
    for idx in range(1, len(images)):
        curr_gray, _ = preprocessed_data[idx]
        
        # 初始化当前变换矩阵
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            # 核心：计算当前帧相对于上一帧的变换
            cc, warp_matrix = cv2.findTransformECC(
                last_gray,  # template
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

        except cv2.error as e:
            print(f"Warning: ECC failed to converge at frame {idx}. Assuming no motion.")
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

        # 矩阵累积
        if warp_mode == cv2.MOTION_AFFINE:
            row = np.array([[0, 0, 1]], dtype=np.float32)
            H_local_3x3 = np.vstack([warp_matrix, row])
            H_global = np.matmul(H_global, H_local_3x3)
        else:
            H_global = np.matmul(H_global, warp_matrix)

        # 记录变换矩阵（使用逆矩阵）
        H_inv = np.linalg.inv(H_global)
        H_matrices.append(H_inv.copy())

        # 更新上一帧
        last_gray = curr_gray
        
        # if idx % 5 == 0:
            # print(f"    Calculated matrix {idx}/{len(images)}...")

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

    # 尝试导入 cupy
    try:
        import cupy as cp
        import cupyx.scipy.ndimage
        HAS_CUPY = True
        print("    [Info] Cupy detected. GPU acceleration enabled for warping.")
    except ImportError:
        HAS_CUPY = False
        print("    [Info] Cupy not found. Using CPU for warping.")

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
            H_gpu = cp.asarray(H_final)
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
            aligned_img = cp.asnumpy(aligned_img_gpu).astype(np.uint8)
            
        else:
            # CPU 版本 (OpenCV)
            aligned_img = cv2.warpPerspective(
                img,
                H_final,
                (target_w, target_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT
            )
        
        # 如果有输出路径，直接在线程中保存
        if output_path:
            fname = img_filenames[idx] if img_filenames else f'frame_{idx:04d}.png'
            cv2.imwrite(os.path.join(output_path, fname), aligned_img)
            
        return aligned_img

    # 准备参数
    task_args = []
    for i in range(len(images)):
        task_args.append((i, images[i], H_matrices[i]))

    # 如果有 Cupy，不使用多线程，因为 GPU 操作本身是并行的且受限于 PCIe 带宽
    # 多线程同时向 GPU 传输数据可能会导致争用
    if HAS_CUPY:
        aligned_images = []
        for args in task_args:
            aligned_images.append(warp_task(args))
    else:
        # CPU 模式下继续使用多线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            aligned_images = list(executor.map(warp_task, task_args))

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
             if os.path.splitext(file)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']],
            key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[-1])
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
#              if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}),
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
    
    SUPPORTED_METHODS = ['homography', 'ecc', 'both']
    
    def __init__(self, method: str = 'homography', downscale_width: int = 1024):
        """
        初始化配准器
        
        Args:
            method (str): 配准方法名称，可选 'homography', 'ecc', 'both'
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"不支持的配准方法: {method}. "
                f"支持的方法: {', '.join(self.SUPPORTED_METHODS)}"
            )
        
        self.method = method
        # 用户可配置的下采样宽度，用于特征提取等预处理阶段
        self.downscale_width = int(downscale_width) if downscale_width is not None else 1024
    
    def process(self, 
                input_source: Union[str, List[np.ndarray]], 
                output_path: Optional[str] = None,
                thread_count: int = 4) -> List[np.ndarray]:
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
        if self.method == 'homography':
            return self._process_homography(input_source, output_path, thread_count=thread_count)
        elif self.method == 'ecc':
            return self._process_ecc(input_source, output_path, thread_count=thread_count)
        elif self.method == 'both':
            # 组合模式：先 Homography，后 ECC
            # 第一步：Homography (不保存中间结果，除非只做这一步)
            print("=== Step 1: Homography Alignment ===")
            # 如果是 both 模式，第一步不需要保存到 output_path，只在内存中传递
            homography_result = self._process_homography(input_source, output_path=None, thread_count=thread_count)
            
            print("\n=== Step 2: ECC Alignment ===")
            # 第二步：ECC (保存最终结果)
            return self._process_ecc(homography_result, output_path, thread_count=thread_count)
    
    def _process_homography(self, 
                           input_source: Union[str, List[np.ndarray]], 
                           output_path: Optional[str] = None,
                           thread_count: int = 4) -> List[np.ndarray]:
        """
        单应性对齐配准（非线性）
        
        Args:
            input_source: 图像源
            output_path: 输出路径
        
        Returns:
            配准后的图像列表
        """
        return _align_homography_impl(input_source, output_path, downscale_width=self.downscale_width, thread_count=thread_count)
    
    def _process_ecc(self, 
                    input_source: Union[str, List[np.ndarray]], 
                    output_path: Optional[str] = None,
                    thread_count: int = 4) -> List[np.ndarray]:
        """
        ECC对齐配准（高精度、亚像素级）
        
        Args:
            input_source: 图像源
            output_path: 输出路径
        
        Returns:
            配准后的图像列表
        """
        return _align_ecc_impl(input_source, output_path, downscale_width=self.downscale_width, thread_count=thread_count)
    
    def set_method(self, method: str):
        """
        切换配准方法
        
        Args:
            method (str): 新的配准方法名称
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"不支持的配准方法: {method}. "
                f"支持的方法: {', '.join(self.SUPPORTED_METHODS)}"
            )
        
        self.method = method
    
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
    parser.add_argument('--mode', default='homography', choices=['homography', 'ecc', 'both'],
                        help='Processing mode: homography, ecc, both (homography + ecc)')
    
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