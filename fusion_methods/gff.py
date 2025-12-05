import os
import glob
import re
import time
import cv2
import numpy as np
import concurrent.futures

# Reference:
# https://github.com/RCharradi/Image-fusion-with-guided-filtering
# Li S, Kang X, Hu J. Image fusion with guided filtering[J]. IEEE Transactions on Image processing, 2013, 22(7): 2864-2875.

def gff_impl(input_source, img_resize, kernel_size=31):
    """
    基于引导滤波的多焦点图像栈融合算法实现 (CPU版本)
    不依赖 opencv-contrib (ximgproc)，内置引导滤波实现。
    """
    
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

    # ========== 内置引导滤波实现 ==========
    def guided_filter(I, p, r, eps):
        """
        快速引导滤波实现 (基于 Box Filter)
        I: 引导图像 (Guide Image), 单通道或三通道
        p: 输入图像 (Input Image), 单通道
        r: 滤波半径 (radius)
        eps: 正则化参数
        """
        # 确保 I 和 p 类型一致
        if I.dtype != np.float32:
            I = I.astype(np.float32)
        if p.dtype != np.float32:
            p = p.astype(np.float32)
            
        # 滤波器直径
        ksize = (2 * r + 1, 2 * r + 1)
        
        # 均值计算
        mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
        mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
        
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
        
        q = mean_a * I + mean_b
        return q

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
            raise ValueError("输入目录为空或未找到图像文件")
            
        glob_format = '*' + img_ext
        img_stack_path_list = glob.glob(os.path.join(input_source, glob_format))
        # 尝试按数字排序
        try:
            img_stack_path_list.sort(key=lambda x: int(str(re.findall(r"\d+", x.split(os.sep)[-1])[-1])))
        except IndexError:
            img_stack_path_list.sort() # 回退到字典序
            
        stack_ori = [cv2.imread(img_path) for img_path in img_stack_path_list]
    else:
        stack_ori = input_source
    
    if not stack_ori:
        raise ValueError("没有加载到图像数据")

    if img_resize:
        stack_ori = [cv2.resize(img, img_resize) for img in stack_ori]

    # 转换为 float32 并归一化到 [0, 1]
    stack_flt = [img.astype(np.float32) / 255.0 for img in stack_ori]
    
    # ========== 核心算法实现 ==========
    
    def guided_filter_fusion_stack(images):
        """
        对图像堆栈执行基于引导滤波的融合
        """
        num_images = len(images)
        if num_images == 0:
            return None
        
        rows, cols, channels = images[0].shape
        
        # 1. 图像分解 (Base Layer & Detail Layer)
        # 使用 uniform_filter 提取 Base 层
        
        def process_decompose(img):
            # 使用 cv2.blur 替代 uniform_filter，速度更快
            base = cv2.blur(img, (average_filter_size, average_filter_size), borderType=cv2.BORDER_REFLECT)
            detail = img - base
            return base, detail

        # 并行处理图像分解
        with concurrent.futures.ThreadPoolExecutor() as executor:
            decompose_results = list(executor.map(process_decompose, images))
        
        base_layers = [res[0] for res in decompose_results]
        detail_layers = [res[1] for res in decompose_results]
            
        # 2. 计算显著性图 (Saliency Map)
        # Saliency = Gaussian(abs(Laplacian(Sum_Channels)))
        
        def process_saliency(img):
            # 将三通道相加用于计算拉普拉斯
            img_sum = np.sum(img, axis=2)
            # 使用 cv2.Laplacian 替代 scipy.ndimage.laplace
            lap = np.abs(cv2.Laplacian(img_sum, cv2.CV_32F, ksize=1, borderType=cv2.BORDER_REFLECT))
            # 使用 cv2.GaussianBlur 替代 scipy.ndimage.gaussian_filter
            sal = cv2.GaussianBlur(lap, (0, 0), sigmaX=DEFAULT_SIGMA_R, sigmaY=DEFAULT_SIGMA_R, borderType=cv2.BORDER_REFLECT)
            return sal

        # 并行处理显著性图计算
        with concurrent.futures.ThreadPoolExecutor() as executor:
            saliency_maps = list(executor.map(process_saliency, images))
            
        # 3. 生成初始决策图 (Decision Map)
        # 在每个像素位置选择显著性最大的图像索引
        saliency_stack = np.stack(saliency_maps, axis=0) # (N, H, W)
        max_indices = np.argmax(saliency_stack, axis=0)  # (H, W)
        
        # 4. 权重图细化与融合 (Weight Refinement & Fusion)
        # 初始化累加器
        fused_base_numerator = np.zeros((rows, cols, channels), dtype=np.float32)
        fused_base_denominator = np.zeros((rows, cols), dtype=np.float32) # 优化为单通道
        
        fused_detail_numerator = np.zeros((rows, cols, channels), dtype=np.float32)
        fused_detail_denominator = np.zeros((rows, cols), dtype=np.float32) # 优化为单通道
        
        def process_weight_fusion(k):
            # 生成第 k 张图的二值掩膜
            mask_k = (max_indices == k).astype(np.float32) # (H, W)
            
            img_k = images[k] # Guide image (H, W, 3)
            
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
            
            # 针对 Detail 层
            weight_detail_k = run_gf(DEFAULT_R2, DEFAULT_EPS2)

            # 优化：不再使用 np.repeat 扩展为 3 通道，而是利用广播机制
            # weight_base_k 和 weight_detail_k 都是 (H, W)
            
            # 计算分子项 (H, W, 3) * (H, W, 1) -> (H, W, 3)
            base_num = base_layers[k] * weight_base_k[:, :, np.newaxis]
            detail_num = detail_layers[k] * weight_detail_k[:, :, np.newaxis]
            
            # 分母项直接返回单通道权重即可
            return base_num, weight_base_k, detail_num, weight_detail_k

        # 并行处理权重计算与融合
        # 使用 as_completed 模式，处理完一个就累加一个，避免一次性持有所有结果导致内存爆炸
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_weight_fusion, k): k for k in range(num_images)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    bn, bd, dn, dd = future.result()
                    fused_base_numerator += bn
                    fused_base_denominator += bd
                    fused_detail_numerator += dn
                    fused_detail_denominator += dd
                    # 显式删除引用，帮助垃圾回收
                    del bn, bd, dn, dd
                except Exception as exc:
                    print(f'Image {futures[future]} generated an exception: {exc}')

            
        # 5. 重建图像
        # 避免除以零
        fused_base_denominator[fused_base_denominator < 1e-6] = 1e-6
        fused_detail_denominator[fused_detail_denominator < 1e-6] = 1e-6
        
        # 广播除法 (H, W, 3) / (H, W, 1)
        fused_base = fused_base_numerator / fused_base_denominator[:, :, np.newaxis]
        fused_detail = fused_detail_numerator / fused_detail_denominator[:, :, np.newaxis]
        
        fused_img = fused_base + fused_detail
        
        # 裁剪并转换回 uint8
        fused_img = np.clip(fused_img * 255, 0, 255).astype(np.uint8)
        
        return fused_img

    return guided_filter_fusion_stack(stack_flt)
