# Reference:
# Haghighat M B A, Aghagolzadeh A, Seyedarabi H. Multi-focus image fusion for visual sensor networks in DCT domain[J]. Computers & Electrical Engineering, 2011, 37(5): 789-797.
import glob
import os
import time
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np


ArraySource = Sequence[np.ndarray]


def _ensure_color_image(image: np.ndarray) -> np.ndarray:
    """确保输入图像为三通道BGR格式。"""
    if image is None:
        raise ValueError("Input image is None")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.ndim == 3 and image.shape[2] == 3:
        return image

    raise ValueError("Unsupported image shape for DCT fusion: " + str(image.shape))


def _collect_images_from_folder(source_folder: str) -> Tuple[List[np.ndarray], List[str]]:
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    img_paths: List[str] = []

    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(source_folder, ext)))

    img_paths.sort()

    images: List[np.ndarray] = []
    for path in img_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    return images, img_paths


def _normalize_image_stack(images: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    if not images:
        raise ValueError("图像栈为空，无法执行DCT融合。")

    normalized: List[np.ndarray] = []
    ref_img = _ensure_color_image(np.ascontiguousarray(images[0]))
    target_h, target_w = ref_img.shape[:2]

    for idx, img in enumerate(images):
        color_img = _ensure_color_image(np.ascontiguousarray(img))
        if color_img.shape[:2] != (target_h, target_w):
            color_img = cv2.resize(color_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        normalized.append(color_img)

    if target_h < 8 or target_w < 8:
        raise ValueError("输入图像尺寸过小，无法执行DCT融合。")

    return normalized, (target_h, target_w)


def _compute_variance_map(gray_image: np.ndarray, block_size: int, map_h: int, map_w: int) -> np.ndarray:
    var_map = np.zeros((map_h, map_w), dtype=np.float64)
    img_float = gray_image.astype(np.float64) - 128.0

    for i in range(map_h):
        row_start = i * block_size
        row_end = row_start + block_size
        for j in range(map_w):
            col_start = j * block_size
            col_end = col_start + block_size

            block = img_float[row_start:row_end, col_start:col_end]
            dct_block = cv2.dct(block)
            norm_dct = dct_block / float(block_size)
            mean_val = norm_dct[0, 0]
            variance = np.sum(norm_dct ** 2) - (mean_val ** 2)
            var_map[i, j] = variance

    return var_map


def dct_focus_stack_fusion(
    source: Union[str, ArraySource],
    output_path: str = None,
    block_size: int = 8,
    kernel_size: int = 7,
) -> np.ndarray:
    """
    使用 DCT+Variance+CV 算法融合文件夹中的所有图像。
    
    参数:
        source (str | Sequence[np.ndarray]): 图片文件夹路径或图像数组序列
        output_path (str, optional): 结果保存路径
        block_size (int): DCT 变换块大小 (建议 8 或 16)
        kernel_size (int): 一致性验证滤波器的核大小 (必须为奇数, 建议 5, 7, 9)
    
    返回:
        np.ndarray: 融合后的图像
    """
    
    # --- 参数校验 ---
    if kernel_size % 2 == 0:
        print(f"警告: kernel_size 必须是奇数，已自动调整为 {kernel_size + 1}")
        kernel_size += 1

    block_size = max(1, int(block_size))
    if block_size < 2:
        block_size = 2

    # --- 1. 准备图像栈 ---
    if isinstance(source, str):
        images, _ = _collect_images_from_folder(source)
        if len(images) < 2:
            raise ValueError(f"文件夹 {source} 中图片少于 2 张，无法融合。")
    elif isinstance(source, (list, tuple)):
        images = [np.ascontiguousarray(img) for img in source if img is not None]
        if len(images) < 2:
            raise ValueError("提供的图像栈少于 2 张，无法融合。")
    else:
        raise TypeError("source 必须是文件夹路径或图像数组序列。")

    normalized_images, (target_h, target_w) = _normalize_image_stack(images)

    h_trim = (target_h // block_size) * block_size
    w_trim = (target_w // block_size) * block_size
    map_h = h_trim // block_size
    map_w = w_trim // block_size

    if h_trim == 0 or w_trim == 0:
        raise ValueError("图像尺寸不足以按照当前 block_size 进行分块。")

    max_variance_map = np.full((map_h, map_w), -1.0, dtype=np.float64)
    best_index_map = np.zeros((map_h, map_w), dtype=np.int32)

    # --- 2. 计算方差图 ---
    for idx, color_img in enumerate(normalized_images):
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        gray = gray[:h_trim, :w_trim]
        var_map = _compute_variance_map(gray, block_size, map_h, map_w)

        mask_better = var_map > max_variance_map
        max_variance_map[mask_better] = var_map[mask_better]
        best_index_map[mask_better] = idx

    # --- 3. 一致性验证 (Median Filter) ---
    if len(normalized_images) < 255:
        cv_map_processing = best_index_map.astype(np.uint8)
    else:
        cv_map_processing = best_index_map.astype(np.float32)

    filtered_map = cv2.medianBlur(cv_map_processing, kernel_size)
    filtered_map = cv2.medianBlur(filtered_map, kernel_size)
    final_index_map = filtered_map.astype(np.int32)

    # --- 4. 重建融合结果 ---
    fused_image = np.zeros((h_trim, w_trim, 3), dtype=np.uint8)
    unique_indices = np.unique(final_index_map)

    for idx in unique_indices:
        mask_small = (final_index_map == idx).astype(np.uint8)
        mask_full = cv2.resize(mask_small, (w_trim, h_trim), interpolation=cv2.INTER_NEAREST)
        if not np.any(mask_full):
            continue

        src_img = normalized_images[idx][:h_trim, :w_trim]
        mask_bool = mask_full.astype(bool)
        fused_image[mask_bool] = src_img[mask_bool]

    if output_path:
        cv2.imwrite(output_path, fused_image)

    return fused_image

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    # 配置
    t_start = time.time()
    TARGET_DIR = r"C:\Users\dell\Pictures\Helicon Focus\StackMFF V2 Used\Bug"
    OUTPUT_FILE = os.path.join(TARGET_DIR, "Fused_Result_Fast.tif")
    
    # 参数设置
    BLOCK_SIZE = 8   # DCT 块大小
    KERNEL_SIZE = 7  # 滤波核大小
    
    if os.path.exists(TARGET_DIR):
        try:
            # 调用函数
            result = dct_focus_stack_fusion(
                source_folder=TARGET_DIR,
                output_path=OUTPUT_FILE,
                block_size=BLOCK_SIZE,
                kernel_size=KERNEL_SIZE
            )
            print(f"总耗时: {time.time() - t_start:.2f}s")
            # 显示缩略图
            if result is not None:
                h, w = result.shape[:2]
                scale = 800 / h
                show_img = cv2.resize(result, (int(w*scale), int(h*scale)))
                cv2.imshow("Fast Fusion Result", show_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("路径不存在。")