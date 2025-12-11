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
    # 使用较小的数据类型存储索引，节省内存
    idx_dtype = np.uint8 if len(images) < 256 else np.int32
    best_index_map = np.zeros((map_h, map_w), dtype=idx_dtype)

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
    # 必须转回适合滤波的类型，虽然 uint8 也可以，但为了稳健转一下
    if idx_dtype == np.uint8:
        map_to_filter = best_index_map
    else:
        map_to_filter = best_index_map.astype(np.float32)

    # 两次中值滤波去除噪点
    filtered_map = cv2.medianBlur(map_to_filter, kernel_size)
    filtered_map = cv2.medianBlur(filtered_map, kernel_size)
    
    # 转回整数索引
    if filtered_map.dtype != np.int32 and filtered_map.dtype != np.uint8:
        final_index_map = filtered_map.astype(np.int32)
    else:
        final_index_map = filtered_map

    # --- 4. 快速重建 ---
    # 将小尺寸的索引图一次性放大回原图尺寸 (Nearest Neighbor)
    full_size_indices = cv2.resize(
        final_index_map.astype(np.uint8), # resize 对 uint8 最快
        (w_trim, h_trim), 
        interpolation=cv2.INTER_NEAREST
    )

    fused_image = np.zeros((h_trim, w_trim, 3), dtype=np.uint8)
    
    # 仅遍历用到的源图像索引进行填充
    unique_indices = np.unique(final_index_map)
    
    for idx in unique_indices:
        # 生成掩膜：哪里需要这张图，哪里就是 True
        mask = (full_size_indices == idx)
        
        # 即使这里是 Python 循环，也是针对整张图的掩膜操作，速度很快
        # 裁剪源图像以匹配尺寸
        source_layer = normalized_images[idx][:h_trim, :w_trim]
        
        # 赋值
        fused_image[mask] = source_layer[mask]

    if output_path:
        cv2.imwrite(output_path, fused_image)

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