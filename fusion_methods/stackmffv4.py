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
import cv2

# ================= 全局缓存变量 =================
_GLOBAL_MODEL = None
_GLOBAL_DEVICE = None

# 缓存可用的加速器类型（模块加载时检测一次）
_MPS_AVAILABLE = None
_CUDA_AVAILABLE = None


def _detect_accelerators():
    """检测系统可用的加速器类型，模块加载时执行一次"""
    global _MPS_AVAILABLE, _CUDA_AVAILABLE
    try:
        import torch
        _MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        _CUDA_AVAILABLE = torch.cuda.is_available()
    except Exception:
        _MPS_AVAILABLE = False
        _CUDA_AVAILABLE = False


_detect_accelerators()

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

    if use_gpu and _MPS_AVAILABLE:
        device = torch.device('mps')
    elif use_gpu and _CUDA_AVAILABLE:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    device_name = device.type.upper()
    if device.type == 'cuda':
        device_name = 'CUDA'

    print(f"Running AI fusion on {device_name}...")

    color_images = []
    gray_tensors = []

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

        for img_path in img_stack_path_list:
            bgr_img = cv2.imread(img_path)
            if img_resize:
                bgr_img = cv2.resize(bgr_img, img_resize)
            color_images.append(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            gray_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0)
            gray_tensors.append(gray_tensor)
    else:
        bgr_images = list(input_source)
        if not bgr_images:
            raise ValueError("Empty image list")
        if img_resize:
            bgr_images = [cv2.resize(img, img_resize) for img in bgr_images]
        for img in bgr_images:
            color_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0)
            gray_tensors.append(gray_tensor)
    
    num_images = len(color_images)
    if num_images < 2:
        raise ValueError("At least two images are required for fusion")

    print(f"Loaded {num_images} images")

    image_stack = torch.stack(gray_tensors)
    original_size = image_stack.shape[-2:]

    global _GLOBAL_MODEL, _GLOBAL_DEVICE

    if _GLOBAL_MODEL is None:
        print("Loading StackMFF-V4 model (once)...")
        model = StackMFF_V4()
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        _GLOBAL_MODEL = model
        _GLOBAL_DEVICE = device
    else:
        model = _GLOBAL_MODEL
        if _GLOBAL_DEVICE != device:
            model.to(device)
            _GLOBAL_DEVICE = device
    
    def resize_to_multiple_of_32(image):
        h, w = image.shape[-2:]
        new_h = ((h - 1) // 32 + 1) * 32
        new_w = ((w - 1) // 32 + 1) * 32
        if new_h == h and new_w == w:
            return image, (h, w)
        resized = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return resized, (h, w)

    with torch.no_grad():
        input_tensor = image_stack.unsqueeze(0).to(device)
        resized_input, _ = resize_to_multiple_of_32(input_tensor)

        _, focus_indices = model(resized_input)

        focus_indices = focus_indices.squeeze().cpu().numpy()

        del input_tensor, resized_input

    h, w = original_size
    focus_map = cv2.resize(
        focus_indices.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_NEAREST
    ).astype(int)

    focus_map = np.clip(focus_map, 0, num_images - 1)
    color_array = np.stack(color_images, axis=0)
    fused_color = color_array[focus_map, np.arange(h)[:, None], np.arange(w)]
    fused_color_bgr = cv2.cvtColor(fused_color.astype(np.uint8), cv2.COLOR_RGB2BGR)

    return fused_color_bgr