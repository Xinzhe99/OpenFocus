import os
import glob
import re
import cv2
import numpy as np
import concurrent.futures

from fusion_methods.dct import _ensure_color_image
from utils.image_utils import fuse_output_dtype

# Reference:
# https://github.com/RCharradi/Image-fusion-with-guided-filtering
# Li S, Kang X, Hu J. Image fusion with guided filtering[J]. IEEE Transactions on Image processing, 2013, 22(7): 2864-2875.

def gff_impl(input_source, img_resize, kernel_size=31, thread_count: int = None,
             should_cancel=None):
    """
    基于引导滤波的多焦点图像栈融合算法实现 (CPU版本)
    不依赖 opencv-contrib (ximgproc)，内置引导滤波实现。
    """
    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    check_cancel()
    try:
        max_workers = max(1, int(thread_count)) if thread_count is not None else 1
    except (TypeError, ValueError):
        max_workers = 1
    
    # ========== 参数设置 ==========
    # 默认参数 (参考原脚本)
    DEFAULT_R1 = 45
    DEFAULT_R2 = 7
    DEFAULT_EPS1 = 0.3
    DEFAULT_EPS2 = 10e-6
    DEFAULT_SIGMA_R = 5
    
    # 如果传入了 kernel_size 且有效，则使用它作为均值滤波尺寸
    average_filter_size = kernel_size if kernel_size is not None and kernel_size > 0 else 31
    
    # 确保滤波器大小为奇数
    if average_filter_size % 2 == 0:
        average_filter_size += 1


    # ========== 数据加载 ==========
    if isinstance(input_source, str):
        def get_image_suffix(input_stack_path):
            filenames = os.listdir(input_stack_path)
            if len(filenames) == 0:
                return None
            suffixes = [os.path.splitext(filename)[1] for filename in filenames]
            return suffixes[0]

        img_ext = get_image_suffix(input_source)
        if img_ext is None:
            raise ValueError("Input folder is empty or no images were found")
            
        glob_format = '*' + img_ext
        img_stack_path_list = glob.glob(os.path.join(input_source, glob_format))
        # 尝试按数字排序
        try:
            img_stack_path_list.sort(key=lambda x: int(str(re.findall(r"\d+", x.split(os.sep)[-1])[-1])))
        except IndexError:
            img_stack_path_list.sort() # 回退到字典序
            
        stack_ori = []
        for img_path in img_stack_path_list:
            check_cancel()
            stack_ori.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
    else:
        stack_ori = input_source
    
    if not stack_ori:
        raise ValueError("No image data was loaded")

    out_dtype = fuse_output_dtype(stack_ori)

    def prepare_frame(index):
        check_cancel()
        image = _ensure_color_image(np.ascontiguousarray(stack_ori[index]))
        if image.dtype == np.uint16:
            image = image.astype(np.float32) * (255.0 / 65535.0)
        if img_resize:
            image = cv2.resize(image, img_resize)
        return image.astype(np.float32) / 255.0
    
    # ========== 核心算法实现 ==========
    
    def guided_filter_fusion_stack():
        """
        对图像堆栈执行基于引导滤波的融合
        """
        num_images = len(stack_ori)
        if num_images == 0:
            return None
        
        first = prepare_frame(0)
        rows, cols, channels = first.shape
        del first
            
        # 2. 计算显著性图 (Saliency Map)
        # Saliency = Gaussian(abs(Laplacian(Sum_Channels)))
        
        def process_saliency(index):
            img = prepare_frame(index)
            # 将三通道相加用于计算拉普拉斯
            img_sum = np.sum(img, axis=2)
            # 使用 cv2.Laplacian 替代 scipy.ndimage.laplace
            lap = np.abs(cv2.Laplacian(img_sum, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REFLECT))
            # 使用 cv2.GaussianBlur 替代 scipy.ndimage.gaussian_filter
            sal = cv2.GaussianBlur(lap, (0, 0), sigmaX=DEFAULT_SIGMA_R, sigmaY=DEFAULT_SIGMA_R, borderType=cv2.BORDER_REFLECT)
            return sal

        # 3. 初始决策图 (Decision Map)：逐帧在线求 argmax。
        # 严格 `>` 更新保持 np.argmax 的首帧平局行为，且不再物化整栈
        # (N, H, W) 显著性堆栈——那是一个只用于一次 argmax 的全栈 float32。
        max_indices = np.zeros((rows, cols), dtype=np.int32)
        best_saliency = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start in range(0, num_images, max_workers):
                check_cancel()
                stop = min(num_images, start + max_workers)
                for k, sal in zip(range(start, stop), executor.map(process_saliency, range(start, stop))):
                    check_cancel()
                    if best_saliency is None:
                        best_saliency = sal
                    else:
                        win = sal > best_saliency
                        best_saliency[win] = sal[win]
                        max_indices[win] = k
                        del win
                    del sal
        del best_saliency

        # 4. 权重图细化与融合 (Weight Refinement & Fusion)
        # 初始化累加器
        fused_base_numerator = np.zeros((rows, cols, channels), dtype=np.float32)
        fused_base_denominator = np.zeros((rows, cols), dtype=np.float32) # 优化为单通道
        
        fused_detail_numerator = np.zeros((rows, cols, channels), dtype=np.float32)
        fused_detail_denominator = np.zeros((rows, cols), dtype=np.float32) # 优化为单通道

        def process_weight_fusion(k):
            check_cancel()
            # 生成第 k 张图的二值掩膜
            mask_k = (max_indices == k).astype(np.float32) # (H, W)
            
            img_k = prepare_frame(k)
            base = cv2.blur(img_k, (average_filter_size, average_filter_size),
                            borderType=cv2.BORDER_REFLECT)
            detail = img_k - base
            
            # 使用内置的引导滤波细化掩膜
            img_k_gray = cv2.cvtColor(img_k, cv2.COLOR_BGR2GRAY)
            
            # 预计算 I*p 和 I*I，避免在两次 guided_filter 调用中重复计算
            I = img_k_gray
            p = mask_k
            Ip = I * p
            I2 = I * I
            
            # 内部优化版 guided_filter，复用预计算结果
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
                
                mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
                mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
                
                q = mean_a * I + mean_b
                return q

            # 针对 Base 层
            weight_base_k = run_gf(DEFAULT_R1, DEFAULT_EPS1)
            
            check_cancel()
            # 针对 Detail 层
            weight_detail_k = run_gf(DEFAULT_R2, DEFAULT_EPS2)

            # 优化：不再使用 np.repeat 扩展为 3 通道，而是利用广播机制
            # weight_base_k 和 weight_detail_k 都是 (H, W)
            
            # 计算分子项 (H, W, 3) * (H, W, 1) -> (H, W, 3)
            base_num = base * weight_base_k[:, :, np.newaxis]
            detail_num = detail * weight_detail_k[:, :, np.newaxis]
            
            # 分母项直接返回单通道权重即可
            return base_num, weight_base_k, detail_num, weight_detail_k

        # Submit only a worker-sized window; consume and release in frame order.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for start in range(0, num_images, max_workers):
                check_cancel()
                stop = min(num_images, start + max_workers)
                for bn, bd, dn, dd in executor.map(process_weight_fusion, range(start, stop)):
                    check_cancel()
                    fused_base_numerator += bn
                    fused_base_denominator += bd
                    fused_detail_numerator += dn
                    fused_detail_denominator += dd
                    del bn, bd, dn, dd

            
        # 5. 重建图像
        check_cancel()
        # 避免除以零
        fused_base_denominator[fused_base_denominator < 1e-6] = 1e-6
        fused_detail_denominator[fused_detail_denominator < 1e-6] = 1e-6
        
        # 广播除法 (H, W, 3) / (H, W, 1)
        fused_base = fused_base_numerator / fused_base_denominator[:, :, np.newaxis]
        fused_detail = fused_detail_numerator / fused_detail_denominator[:, :, np.newaxis]
        
        fused_img = fused_base + fused_detail
        
        # 裁剪并转换回源位深
        fused_img = np.clip(fused_img * 255, 0, 255).astype(out_dtype)
        check_cancel()
        
        return fused_img

    return guided_filter_fusion_stack()