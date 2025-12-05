# 条件导入，避免在不需要时产生错误
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

import os
import re
import glob
import numpy as np
from typing import Union, List, Tuple, Optional
import cv2

def _stackmffv4_impl(input_source, img_resize, model_path, use_gpu):
    """
    基于 StackMFF-V4 神经网络的图像融合算法
    
    Args:
        input_source: 图像源(目录路径或图像列表)
        img_resize: 目标尺寸 (width, height)
        model_path: 模型权重文件路径
        use_gpu: 是否使用GPU
    
    Returns:
        融合后的图像 (BGR格式, uint8)
    """
    if torch is None or F is None:
        raise ImportError("PyTorch not installed")
    from network import StackMFF_V4
    
    # 设置设备
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用 GPU 进行 AI 融合...")
    else:
        device = torch.device('cpu')
        print("使用 CPU 进行 AI 融合...")
    
    # 加载图像
    if isinstance(input_source, str):
        def get_image_suffix(input_stack_path):
            filenames = os.listdir(input_stack_path)
            if len(filenames) == 0:
                return None
            suffixes = [os.path.splitext(filename)[1] for filename in filenames]
            return suffixes[0]

        img_ext = get_image_suffix(input_source)
        glob_format = '*' + img_ext
        img_stack_path_list = glob.glob(os.path.join(input_source, glob_format))
        img_stack_path_list.sort(
            key=lambda x: int(str(re.findall(r"\d+", x.split(os.sep)[-1])[-1])))
        
        color_images = []
        gray_tensors = []
        for img_path in img_stack_path_list:
            bgr_img = cv2.imread(img_path)
            if img_resize:
                bgr_img = cv2.resize(bgr_img, img_resize)
            color_images.append(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            gray_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0)
            gray_tensors.append(gray_tensor)
    else:
        color_images = []
        gray_tensors = []
        for img in input_source:
            if img_resize:
                img = cv2.resize(img, img_resize)
            color_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0)
            gray_tensors.append(gray_tensor)
    
    num_images = len(color_images)
    if num_images < 2:
        raise ValueError("至少需要2张图像进行融合")
    
    print(f"加载了 {num_images} 张图像")
    
    # 堆叠灰度图像张量
    image_stack = torch.stack(gray_tensors)  # [N, H, W]
    original_size = image_stack.shape[-2:]
    
    # 加载模型
    model = StackMFF_V4()
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # Older PyTorch versions do not support the weights_only flag
        state_dict = torch.load(model_path, map_location=device)
    # 处理 DataParallel 的 'module.' 前缀
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 将图像尺寸调整为32的倍数
    def resize_to_multiple_of_32(image):
        h, w = image.shape[-2:]
        new_h = ((h - 1) // 32 + 1) * 32
        new_w = ((w - 1) // 32 + 1) * 32
        resized = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return resized, (h, w)
    
    # 模型推理
    with torch.no_grad():
        resized_stack, _ = resize_to_multiple_of_32(image_stack.unsqueeze(0))
        resized_stack = resized_stack.to(device)
        
        fused_image, focus_indices = model(resized_stack)
        
        # 转换为 numpy 并调整到原始尺寸
        fused_image = cv2.resize(
            fused_image.cpu().numpy().squeeze(),
            (original_size[1], original_size[0])
        )
        focus_indices = cv2.resize(
            focus_indices.cpu().numpy().squeeze().astype(np.float32),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(int)
    
    # 根据焦点索引生成彩色融合图像
    height, width = fused_image.shape
    focus_indices = np.clip(focus_indices, 0, num_images - 1)
    color_array = np.stack(color_images, axis=0)  # [N, H, W, 3]
    fused_color = color_array[focus_indices, np.arange(height)[:, None], np.arange(width)]
    fused_color_bgr = cv2.cvtColor(fused_color.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return fused_color_bgr