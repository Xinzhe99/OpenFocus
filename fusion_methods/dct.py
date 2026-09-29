# Reference:
# Haghighat M B A, Aghagolzadeh A, Seyedarabi H. Multi-focus image fusion for visual sensor networks in DCT domain[J]. Computers & Electrical Engineering, 2011, 37(5): 789-797.
import glob
import os
import time
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import median_filter

from utils.image_utils import fuse_output_dtype

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
    raise ValueError(f"Unsupported image shape: {image.shape}")

def _collect_images_from_folder(source_folder: str) -> Tuple[List[np.ndarray], List[str]]:
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(source_folder, ext)))
    img_paths.sort()
    
    images = []
    for path in img_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return images, img_paths

def _normalize_image_stack(images: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """统一图像尺寸并返回BGR列表。"""
    if not images:
        raise ValueError("图像栈为空")
    
    # 获取基准尺寸
    ref_img = images[0]
    target_h, target_w = ref_img.shape[:2]
    
    normalized = []
    for img in images:
        img_bgr = _ensure_color_image(np.ascontiguousarray(img))
        if img_bgr.shape[:2] != (target_h, target_w):
            img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        normalized.append(img_bgr)
        
    if target_h < 8 or target_w < 8:
        raise ValueError("图像尺寸过小")
        
    return normalized, (target_h, target_w)

def dct_focus_stack_fusion(
    source: Union[str, ArraySource],
    output_path: str = None,
    block_size: int = 8,
    kernel_size: int = 7,
) -> np.ndarray:
    """
    高度优化的 DCT/方差 图像融合算法。
    
    优化说明:
    利用 Parseval 定理，DCT 域的高频能量(方差)等价于空间域的像素方差。
    通过 cv2.resize(INTER_AREA) 快速计算块均值和平方均值，替代了原本极其缓慢的
    逐块 DCT 循环。
    """
    
    # --- 1. 参数校验与准备 ---
    if kernel_size % 2 == 0:
        kernel_size += 1
    block_size = max(2, int(block_size))

    if isinstance(source, str):
        images, _ = _collect_images_from_folder(source)
    elif isinstance(source, (list, tuple)):
        images = [img for img in source if img is not None]
    else:
        raise TypeError("Source must be folder path or image list.")

    if len(images) < 2:
        raise ValueError("需要至少 2 张图像进行融合。")

    # 归一化并获取尺寸
    normalized_images, (h, w) = _normalize_image_stack(images)

    # 计算对齐后的尺寸 (必须是 block_size 的整数倍)
    h_trim = (h // block_size) * block_size
    w_trim = (w // block_size) * block_size
    map_h = h_trim // block_size
    map_w = w_trim // block_size

    if map_h == 0 or map_w == 0:
        raise ValueError("Block size is too large for image size.")

    # --- 2. 快速计算方差图 (核心优化) ---
    # 预分配空间
    max_variance_map = np.full((map_h, map_w), -1.0, dtype=np.float32)
    # 赢家索引图统一用 int32：帧数可以超过 255，不能为了迁就 cv2.medianBlur
    # 而存成 uint8
    best_index_map = np.zeros((map_h, map_w), dtype=np.int32)

    for idx, bgr_img in enumerate(normalized_images):
        # 裁剪边缘以匹配 block 分块
        img_trim = bgr_img[:h_trim, :w_trim]
        
        # 转灰度并转 float32 以防止平方溢出
        gray = cv2.cvtColor(img_trim, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 1. 计算 E[X^2] (平方的均值)
        # cv2.resize 使用 INTER_AREA 实际上就是在做块平均，速度极快
        mean_sq = cv2.resize(gray ** 2, (map_w, map_h), interpolation=cv2.INTER_AREA)
        
        # 2. 计算 (E[X])^2 (均值的平方)
        mean_val = cv2.resize(gray, (map_w, map_h), interpolation=cv2.INTER_AREA)
        sq_mean = mean_val ** 2
        
        # 3. 方差 Var(X) = E[X^2] - (E[X])^2
        # 这在数学上严格等价于 DCT 交流分量的能量和
        var_map = mean_sq - sq_mean
        
        # 更新最大方差图
        mask = var_map > max_variance_map
        max_variance_map[mask] = var_map[mask]
        best_index_map[mask] = idx

    # --- 3. 一致性验证 (中值滤波) ---
    # cv2.medianBlur 在 ksize > 5 时只接受 CV_8U，帧数 >= 256 的索引图会直接报错。
    # mode="nearest" 与 cv2.medianBlur 的补边行为逐像素一致，8-bit 结果保持不变。
    filtered_map = median_filter(best_index_map, size=kernel_size, mode="nearest")
    filtered_map = median_filter(filtered_map, size=kernel_size, mode="nearest")

    # 索引图本身就是整数，无需再转换
    final_index_map = filtered_map.astype(np.int32, copy=False)

    # --- 4. 快速重建 ---
    # 输出位深跟随输入：16-bit 栈在这里被压成 uint8 会丢掉调用方要保留的深度
    fused_image = np.zeros((h_trim, w_trim, 3), dtype=fuse_output_dtype(normalized_images))

    # 仅遍历用到的源图像索引进行填充
    unique_indices = np.unique(final_index_map)
    
    for idx in unique_indices:
        # 生成掩膜：哪里需要这张图，哪里就是 True
        # 放大"每个索引的掩膜"而不是索引图本身：把索引图 resize 成 uint8 会把
        # 第 256 张之后的帧截断成错误的小索引（长栈会取错源图）。
        block_mask = (final_index_map == idx).astype(np.uint8)
        mask = cv2.resize(block_mask, (w_trim, h_trim),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # 即使这里是 Python 循环，也是针对整张图的掩膜操作，速度很快
        # 裁剪源图像以匹配尺寸
        source_layer = normalized_images[idx][:h_trim, :w_trim]
        
        # 赋值
        fused_image[mask] = source_layer[mask]

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
    TARGET_DIR = r"C:\Users\dell\Pictures\Helicon Focus\StackMFF V2 Used\Bug"
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