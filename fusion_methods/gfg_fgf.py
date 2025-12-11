# Reference:
# 付宏语, 巩岩, 汪路涵, 等. 多聚焦显微图像融合算法[J]. Laser & Optoelectronics Progress, 2024, 61(6): 0618022-0618022-9.


import os
import glob
import re
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# 全局辅助与核心算法实现
# -----------------------------------------------------------------------------

def _fast_guided_filter_impl(I, p, r, eps, s=4):
    """
    Python 实现的快速导向滤波 (He et al. 2015)。
    通过下采样 (s) 显著加速系数计算。
    """
    # 1. 下采样
    h, w = I.shape[:2]
    h_sub = int(h / s)
    w_sub = int(w / s)
    
    # 避免尺寸过小
    if h_sub < 1 or w_sub < 1:
        I_sub = I
        p_sub = p
        r_sub = r
    else:
        I_sub = cv2.resize(I, (w_sub, h_sub), interpolation=cv2.INTER_NEAREST)
        p_sub = cv2.resize(p, (w_sub, h_sub), interpolation=cv2.INTER_NEAREST)
        r_sub = max(1, int(r / s))

    ksize = (2 * r_sub + 1, 2 * r_sub + 1)

    # 2. 计算统计量 (Box Filter)
    # 使用 cv2.CV_32F 明确指定深度，避免不必要的推断
    mean_I = cv2.boxFilter(I_sub, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p_sub, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I_sub * p_sub, cv2.CV_32F, ksize)
    mean_II = cv2.boxFilter(I_sub * I_sub, cv2.CV_32F, ksize)

    # 3. 计算线性系数 a, b
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 4. 系数平滑
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    # 5. 上采样回原始尺寸
    if h_sub < 1 or w_sub < 1:
        q = mean_a * I + mean_b
    else:
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
        q = mean_a * I + mean_b

    return q

# 尝试获取 OpenCV 的快速实现，否则使用上面的 Python 优化版
try:
    # 检查是否有 ximgproc 模块
    _cv_guided_filter = cv2.ximgproc.guidedFilter
    def _run_guided_filter(guide, src, radius, eps):
        return _cv_guided_filter(guide, src, radius, eps)
except AttributeError:
    # 使用 Python 优化版，默认 s=4 进行加速
    def _run_guided_filter(guide, src, radius, eps):
        return _fast_guided_filter_impl(guide, src, radius, eps, s=4)

def gfgfgf_impl(input_source, img_resize=None, kernel_size=7, thread_count: int = None):
    """
    GFG-FGF 优化实现版。
    """
    # -------------------------------------------------------------------------
    # 1. 图像加载与预处理
    # -------------------------------------------------------------------------
    if isinstance(input_source, str):
        # 文件夹读取逻辑
        if not os.path.exists(input_source):
             raise ValueError(f"Path not found: {input_source}")
             
        files = os.listdir(input_source)
        # 简单过滤图片扩展名
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        img_paths = [
            os.path.join(input_source, f) for f in files 
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        
        if not img_paths:
            raise ValueError("No images found in folder")

        # 尝试按数字排序
        try:
            img_paths.sort(key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[-1]))
        except Exception:
            img_paths.sort()

        stack_ori = [cv2.imread(p) for p in img_paths]
    else:
        # 列表输入
        stack_ori = input_source

    if not stack_ori:
        raise ValueError("Input stack is empty")

    # Resize 处理
    if img_resize is not None:
        # 检查是否需要 resize，避免无用操作
        if (stack_ori[0].shape[1], stack_ori[0].shape[0]) != img_resize:
            stack_ori = [cv2.resize(img, img_resize) for img in stack_ori]

    # -------------------------------------------------------------------------
    # 2. 数据准备：统一转为 Float32 (0.0 - 1.0)
    #    这能显著减少后续多次除法运算，并利用 SIMD 优化
    # -------------------------------------------------------------------------
    num_imgs = len(stack_ori)
    h, w = stack_ori[0].shape[:2]

    # imgs_f32: (N, H, W, 3) range [0, 1]
    # 使用列表存储，避免一次性分配巨大的 numpy 数组导致内存溢出
    imgs_f32 = []
    for img in stack_ori:
        if img.dtype == np.uint8:
            imgs_f32.append(img.astype(np.float32) / 255.0)
        else:
            # 假设已经是 float 且范围 0-1 或 0-255，这里做安全处理
            temp = img.astype(np.float32)
            if temp.max() > 1.0:
                temp /= 255.0
            imgs_f32.append(temp)
    
    # 原始 uint8 数据不再需要，释放引用（如果外部未持有）
    del stack_ori

    # 确定用于计算焦点的通道 (Channel Selection)
    # 计算第一张图各通道和，取最大的通道索引
    # 利用 sum(axis=(0,1)) 快速求和
    ch_sum = imgs_f32[0].sum(axis=(0, 1))
    channel_idx = int(np.argmax(ch_sum))

    # 提取所有图的灰度/单通道用于引导 (N, H, W)
    # 直接引用切片，不拷贝数据 (View)
    grays = [img[:, :, channel_idx] for img in imgs_f32]

    # -------------------------------------------------------------------------
    # 3. 计算焦点值 (Focus Measure) - 阶段 1
    # -------------------------------------------------------------------------
    # 定义卷积核 (保持 float32)
    kx = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)
    ky = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)

    focus_vals = np.zeros(num_imgs, dtype=np.float32)

    def compute_focus_score(idx):
        gray_img = grays[idx]
        # filter2D 内部并行优化较好，但在大量小图时 ThreadPool 仍有优势
        gx = cv2.filter2D(gray_img, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT)
        gy = cv2.filter2D(gray_img, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT)
        
        # 忽略边界 1 像素，避免边界伪影
        # 使用 np.mean 之前先做 slice
        sub_gx = gx[1:-1, 1:-1]
        sub_gy = gy[1:-1, 1:-1]
        
        # 向量化计算平方和均值
        score = np.mean(sub_gx**2 + sub_gy**2)
        return idx, score

    # 决定线程池大小：优先使用传入 thread_count，否则使用原先的策略（上限8）
    if thread_count is None:
        max_workers = min(8, os.cpu_count() or 4)
    else:
        try:
            max_workers = max(1, int(thread_count))
        except Exception:
            max_workers = min(8, os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_focus_score, i) for i in range(num_imgs)]
        for fut in as_completed(futures):
            i, val = fut.result()
            focus_vals[i] = val

    max_focus = np.max(focus_vals) if num_imgs > 0 else 1.0
    if max_focus == 0: max_focus = 1.0 # 避免除零

    # -------------------------------------------------------------------------
    # 4. 计算初始决策图 (AFMs) - 阶段 2
    # -------------------------------------------------------------------------
    scale = 0.15
    g_msz = kernel_size
    g_gsz = 5
    g_eps = 0.3
    threshold = 0.005
    
    # 预分配 stack 空间 (N, H, W)
    afms_stack = np.zeros((num_imgs, h, w), dtype=np.float32)

    def compute_afm_map(i):
        # 焦点值过低直接跳过，保留全0
        if focus_vals[i] < scale * max_focus:
            return i, None
        
        g = grays[i]
        # 局部平均
        src_blur = cv2.blur(g, (g_msz, g_msz))
        # 差分
        src_diff = cv2.absdiff(g, src_blur)
        
        # 阈值处理：原地操作优化
        # gfg_map = src_diff if src_diff > threshold else 0
        _, gfg_map = cv2.threshold(src_diff, threshold, 0, cv2.THRESH_TOZERO)
        
        # 导向滤波
        afm = _run_guided_filter(g, gfg_map, g_gsz, g_eps)
        return i, afm

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_afm_map, i) for i in range(num_imgs)]
        for fut in as_completed(futures):
            i, res = fut.result()
            if res is not None:
                afms_stack[i] = res

    # -------------------------------------------------------------------------
    # 5. 决策与融合 (Fusion) - 阶段 3
    # -------------------------------------------------------------------------
    # 计算 IDM (Argmax)
    # idm_map: (H, W) 每个像素对应最大响应的图片索引
    idm_map = np.argmax(afms_stack, axis=0).astype(np.int32)
    
    # 释放 afms_stack 内存
    del afms_stack

    # 累加器
    sum_fdms = np.zeros((h, w), dtype=np.float32)
    imfu_result = np.zeros((h, w, 3), dtype=np.float32)

    # 锁用于累加（虽然 Python GIL 存在，但 += numpy array 不是原子的，需注意线程安全）
    # 但为了性能，我们让每个线程返回结果，主线程累加，或者分配独立的 buffer
    # 考虑到内存，我们直接串行累加或者分块。
    # 由于 GuidedFilter 比较耗时，我们还是并行计算 Filter，主线程累加。
    
    def compute_fusion_component(i):
        if focus_vals[i] < scale * max_focus:
            return None
            
        # 生成二值 Mask (float)
        # 这一步比较快
        maxfm_binary = (idm_map == i).astype(np.float32)
        
        # 再次导向滤波平滑权重
        # 这里的 maxfm_binary 也是 0-1，guided filter 输出也是 0-1
        fdm = _run_guided_filter(grays[i], maxfm_binary, g_gsz, g_eps)
        
        return i, fdm

    # 启动并行计算权重
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_fusion_component, i) for i in range(num_imgs)]
        
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            
            i, fdm_weight = res
            
            # 累加权重 (H, W)
            sum_fdms += fdm_weight
            
            # 累加加权图像 (H, W, 3)
            # 利用广播机制：(H,W,3) * (H,W,1) -> (H,W,3)
            # 这里是主要的计算量之一
            imfu_result += imgs_f32[i] * fdm_weight[:, :, None]

    # -------------------------------------------------------------------------
    # 6. 归一化与输出
    # -------------------------------------------------------------------------
    # 避免除以零
    # 创建掩码，只处理权重不为0的区域
    nonzero_mask = sum_fdms > 1e-6
    
    # 原地归一化
    # 只需对 nonzero 区域处理，其他区域保持 0
    # 由于 imfu_result 已经是 float32，直接操作
    
    # 将 sum_fdms 扩展维度以便广播
    sum_fdms_expanded = sum_fdms[:, :, None]
    
    # 利用 np.divide 的 where 参数或者 boolean indexing
    # Boolean indexing 在大数组下可能会产生临时拷贝，但比迭代快
    # 考虑到内存效率，逐通道处理
    
    out = np.zeros((h, w, 3), dtype=np.uint8)
    
    for c in range(3):
        # 提取通道
        ch_data = imfu_result[:, :, c]
        # 除法
        # 使用 np.divide 的 out 参数
        np.divide(ch_data, sum_fdms, out=ch_data, where=nonzero_mask)
        # Clip 并转 uint8
        np.clip(ch_data * 255.0, 0, 255, out=ch_data)
        out[:, :, c] = ch_data.astype(np.uint8)

    return out

# -----------------------------------------------------------------------------
# 测试入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) > 1:
        folder = sys.argv[1]
        t0 = time.time()
        out = gfgfgf_impl(folder, None)
        t1 = time.time()
        print(f"Fusion done in {t1 - t0:.4f}s")
        cv2.imwrite('gfg_fused_opt.png', out)
    else:
        # 生成较大的测试数据
        print("Running synthesis test...")
        h, w = 1024, 1024
        img1 = np.zeros((h, w, 3), dtype=np.uint8)
        img2 = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 绘制图案
        cv2.circle(img1, (300, 300), 100, (255, 255, 255), -1)
        cv2.rectangle(img2, (600, 600), (900, 900), (0, 255, 0), -1)
        
        # 模糊模拟散焦
        blur1 = cv2.GaussianBlur(img1, (51, 51), 0)
        blur2 = cv2.GaussianBlur(img2, (51, 51), 0)
        
        # 输入源：左边清晰，右边模糊 vs 左边模糊，右边清晰
        src1 = img1.copy()
        src1[:, 512:] = blur1[:, 512:]
        
        src2 = img2.copy()
        src2[:, :512] = blur2[:, :512]
        
        input_list = [src1, src2]
        
        t0 = time.time()
        fused = gfgfgf_impl(input_list, None)
        t1 = time.time()
        
        print(f"Optimization Fusion Time: {t1 - t0:.4f}s for size {w}x{h}")
        # cv2.imshow("Fused", fused)
        # cv2.waitKey(0)
        cv2.imwrite("test_fused.png", fused)