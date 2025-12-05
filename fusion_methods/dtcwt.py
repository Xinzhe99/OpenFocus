# Reference:
# Lewis J J, O’Callaghan R J, Nikolov S G, et al. Pixel-and region-based image fusion with complex wavelets[J]. Information fusion, 2007, 8(2): 119-130.
import os
import re
import glob
import numpy as np
from typing import Union, List, Tuple, Optional
import cv2
from fusion_methods.gff import gff_impl
from fusion_methods.stackmffv4 import _stackmffv4_impl

# 条件导入，避免在不需要时产生错误
# try:
#     import torch
#     import torch.nn.functional as F
# except ImportError:
#     torch = None
#     F = None

torch = None
F = None

# try:
#     from pytorch_wavelets import DTCWTForward, DTCWTInverse
# except ImportError:
#     DTCWTForward = None
#     DTCWTInverse = None

DTCWTForward = None
DTCWTInverse = None

try:
    import dtcwt
    from scipy.ndimage import maximum_filter, convolve
except ImportError:
    dtcwt = None
    maximum_filter = None
    convolve = None

# try:
#     import cupy as cp
#     from cupyx.scipy.ndimage import gaussian_filter, maximum_filter as cp_maximum_filter
# except ImportError:
#     cp = None
#     gaussian_filter = None
#     cp_maximum_filter = None

cp = None
gaussian_filter = None
cp_maximum_filter = None

def _dtcwt_impl(input_source, img_resize, N, use_gpu):
    """
    DTCWT融合算法内部实现
    
    优先使用 pytorch_wavelets（需要PyTorch），
    如果PyTorch不可用，则回退到 dtcwt 库（纯NumPy实现）
    """
    # pytorch_available = torch is not None and DTCWTForward is not None and DTCWTInverse is not None
    dtcwt_numpy_available = dtcwt is not None

    if not dtcwt_numpy_available:
        raise RuntimeError(
            "DTCWT融合当前仅支持CPU实现，请安装 dtcwt 库: pip install dtcwt scipy"
        )

    if use_gpu:
        print("提示: DTCWT融合仅支持CPU执行，已强制切换到CPU")

    # if pytorch_available:
    #     return _dtcwt_fusion_pytorch(input_source, img_resize, N, use_gpu)

    print("提示: 使用dtcwt库(纯NumPy实现)进行DTCWT融合")
    return _dtcwt_fusion_numpy(input_source, img_resize, N, False)


# def _dtcwt_fusion_pytorch(input_source, img_resize, N, use_gpu):
#     """使用 pytorch_wavelets 的 DTCWT 融合实现（需要PyTorch）"""
#     # 原GPU/CuPy实现已注释停用，以确保仅使用CPU版本。

GPU_DTCWT_IMPLEMENTATION = r"""
def _dtcwt_fusion_pytorch(input_source, img_resize, N, use_gpu):
    '''
    使用 pytorch_wavelets 的 DTCWT 融合实现（需要PyTorch）
    '''
    if torch is None or F is None or DTCWTForward is None or DTCWTInverse is None:
        raise ImportError('PyTorch or pytorch_wavelets not installed')

    if use_gpu and not torch.cuda.is_available():
        print('警告: CUDA不可用，自动降级到CPU进行计算')
        use_gpu = False

    cupy_available = False
    cp = None
    cp_maximum_filter = None

    if use_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import maximum_filter as cp_maximum_filter
            cupy_available = True
        except ImportError:
            print('提示: CuPy未安装，将使用PyTorch进行高频融合（速度较慢）')

    device = torch.device('cuda' if use_gpu else 'cpu')

    def fuse_highfreq_coeffs(level_coeffs, device, cupy_available, cp, cp_maximum_filter, window_size=3):
        num_images = len(level_coeffs)
        if num_images != 2:
            fused = level_coeffs[0]
            for i in range(1, num_images):
                fused = fuse_highfreq_coeffs([fused, level_coeffs[i]], device, cupy_available, cp, cp_maximum_filter, window_size)
            return fused

        batch, channels, directions, height, width, ri = level_coeffs[0].shape

        coeff1_reshaped = level_coeffs[0].reshape(batch * channels * directions, height, width, ri)
        coeff2_reshaped = level_coeffs[1].reshape(batch * channels * directions, height, width, ri)

        mag1_all = torch.hypot(coeff1_reshaped[:, :, :, 0], coeff1_reshaped[:, :, :, 1])
        mag2_all = torch.hypot(coeff2_reshaped[:, :, :, 0], coeff2_reshaped[:, :, :, 1])

        use_cupy = cupy_available and device.type == 'cuda'

        if use_cupy:
            mag1_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(mag1_all.contiguous()))
            mag2_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(mag2_all.contiguous()))

            try:
                from cupyx.scipy.ndimage import maximum_filter
                A1_cp = cp.empty_like(mag1_cp)
                A2_cp = cp.empty_like(mag2_cp)

                for i in range(batch * channels * directions):
                    A1_cp[i] = maximum_filter(mag1_cp[i], size=window_size, mode='reflect')
                    A2_cp[i] = maximum_filter(mag2_cp[i], size=window_size, mode='reflect')
            except Exception:
                A1_cp = cp.empty_like(mag1_cp)
                A2_cp = cp.empty_like(mag2_cp)
                for i in range(batch * channels * directions):
                    A1_cp[i] = cp_maximum_filter(mag1_cp[i], size=window_size, mode='reflect')
                    A2_cp[i] = cp_maximum_filter(mag2_cp[i], size=window_size, mode='reflect')

            A1 = torch.from_dlpack(A1_cp.toDlpack())
            A2 = torch.from_dlpack(A2_cp.toDlpack())
        else:
            pad_size = window_size // 2

            mag1_padded = F.pad(mag1_all.unsqueeze(1),
                               (pad_size, pad_size, pad_size, pad_size),
                               mode='reflect')
            mag2_padded = F.pad(mag2_all.unsqueeze(1),
                               (pad_size, pad_size, pad_size, pad_size),
                               mode='reflect')

            A1 = F.max_pool2d(mag1_padded, kernel_size=window_size, stride=1, padding=0).squeeze(1)
            A2 = F.max_pool2d(mag2_padded, kernel_size=window_size, stride=1, padding=0).squeeze(1)

        initial_mask = (A1 > A2)

        kernel = torch.ones(1, 1, window_size, window_size, device=device, dtype=torch.float32)
        conv_result = F.conv2d(
            initial_mask.unsqueeze(1).float(),
            kernel,
            padding=0
        ).squeeze(1)

        threshold = (window_size * window_size) // 2
        W_small = (conv_result > threshold)

        pad_total = window_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        W = F.pad(
            W_small.unsqueeze(1).float(),
            (pad_left, pad_right, pad_left, pad_right),
            mode='replicate'
        ).squeeze(1)

        W_expanded = W.reshape(batch * channels * directions, height, width, 1)
        fused_coeff_flat = torch.lerp(coeff2_reshaped, coeff1_reshaped, W_expanded)
        fused_coeff = fused_coeff_flat.reshape(batch, channels, directions, height, width, ri)

        return fused_coeff

    if use_gpu:
        dtcwt_forward = DTCWTForward(J=N, biort='legall', qshift='qshift_06').cuda()
        dtcwt_inverse = DTCWTInverse(biort='legall', qshift='qshift_06').cuda()
    else:
        dtcwt_forward = DTCWTForward(J=N, biort='legall', qshift='qshift_06')
        dtcwt_inverse = DTCWTInverse(biort='legall', qshift='qshift_06')

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

        images = []
        for img_path in img_stack_path_list:
            img = cv2.imread(img_path)
            if img_resize:
                img = cv2.resize(img, img_resize)
            images.append(img)
    else:
        images = []
        for img in input_source:
            if img_resize:
                img = cv2.resize(img, img_resize)
            images.append(img)

    num_images = len(images)
    if num_images < 2:
        raise ValueError('至少需要2张图像进行融合')

    images_np = np.stack([
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for img in images
    ], axis=0)

    images_batch = torch.from_numpy(images_np.transpose(0, 3, 1, 2))
    if use_gpu:
        images_batch = images_batch.pin_memory().cuda(non_blocking=True)

    all_Yl = []
    all_Yh = []

    with torch.no_grad():
        for i in range(num_images):
            img_tensor = images_batch[i:i+1]
            Yl, Yh = dtcwt_forward(img_tensor)
            all_Yl.append(Yl)
            all_Yh.append(Yh)

        fused_Yl = torch.mean(torch.stack(all_Yl), dim=0)

        fused_Yh = []
        for level in range(N):
            level_coeffs = [Yh[level] for Yh in all_Yh]
            fused_level = fuse_highfreq_coeffs(level_coeffs, device, cupy_available, cp, cp_maximum_filter)
            fused_Yh.append(fused_level)

        fused_img = dtcwt_inverse((fused_Yl, fused_Yh))

    fused_img = fused_img.squeeze(0).cpu().numpy()
    fused_img = fused_img.transpose(1, 2, 0)
    fused_img = np.clip(fused_img * 255.0, 0, 255).astype(np.uint8)
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)

    return fused_img
"""


def _dtcwt_fusion_numpy(input_source, img_resize, N, use_gpu):
    """
    使用 dtcwt 库的 DTCWT 融合实现（纯NumPy，不依赖PyTorch）
    
    注意: 此实现不支持GPU加速，use_gpu参数会被忽略
    """
    if dtcwt is None or maximum_filter is None or convolve is None:
        raise ImportError("dtcwt or scipy not installed")
    
    if use_gpu:
        print("警告: dtcwt纯NumPy实现不支持GPU加速，将使用CPU")
    
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

        images = []
        for img_path in img_stack_path_list:
            img = cv2.imread(img_path)
            if img_resize:
                img = cv2.resize(img, img_resize)
            images.append(img)
    else:
        images = []
        for img in input_source:
            if img_resize:
                img = cv2.resize(img, img_resize)
            images.append(img)
    
    num_images = len(images)
    if num_images < 2:
        raise ValueError("至少需要2张图像进行融合")
    
    # 转换为RGB浮点格式
    images_rgb = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for img in images
    ]
    
    # 高频系数融合函数（纯NumPy实现）
    def fuse_highfreq_numpy(coeffs_list, window_size=3):
        """
        融合多张图像的高频系数
        coeffs_list: 每张图像该层的高频系数，形状为 (height, width, 6) 复数数组
        """
        num_imgs = len(coeffs_list)
        if num_imgs == 1:
            return coeffs_list[0]
        
        # 两两融合
        if num_imgs > 2:
            fused = coeffs_list[0]
            for i in range(1, num_imgs):
                fused = fuse_highfreq_numpy([fused, coeffs_list[i]], window_size)
            return fused
        
        coeff1, coeff2 = coeffs_list[0], coeffs_list[1]
        height, width, num_dirs = coeff1.shape
        
        fused = np.zeros_like(coeff1)
        
        for d in range(num_dirs):
            # 计算幅值
            mag1 = np.abs(coeff1[:, :, d])
            mag2 = np.abs(coeff2[:, :, d])
            
            # 局部最大幅值
            A1 = maximum_filter(mag1, size=window_size, mode='reflect')
            A2 = maximum_filter(mag2, size=window_size, mode='reflect')
            
            # 初始掩码
            initial_mask = (A1 > A2).astype(np.float32)
            
            # 一致性验证
            kernel = np.ones((window_size, window_size), dtype=np.float32)
            conv_result = convolve(initial_mask, kernel, mode='constant', cval=0)
            
            threshold = (window_size * window_size) / 2
            W = (conv_result > threshold).astype(np.float32)
            
            # 融合
            fused[:, :, d] = W * coeff1[:, :, d] + (1 - W) * coeff2[:, :, d]
        
        return fused
    
    # 创建DTCWT变换器
    transform = dtcwt.Transform2d()
    
    # 对每个通道分别进行融合
    fused_channels = []
    
    for channel in range(3):  # RGB三通道
        # 对每张图像的该通道进行DTCWT变换
        all_transforms = []
        for img in images_rgb:
            t = transform.forward(img[:, :, channel], nlevels=N)
            all_transforms.append(t)
        
        # 融合低频系数（取平均）
        lowpass_stack = np.stack([t.lowpass for t in all_transforms], axis=0)
        fused_lowpass = np.mean(lowpass_stack, axis=0)
        
        # 融合高频系数
        fused_highpasses = []
        for level in range(N):
            level_coeffs = [t.highpasses[level] for t in all_transforms]
            fused_level = fuse_highfreq_numpy(level_coeffs)
            fused_highpasses.append(fused_level)
        
        # 构建融合后的变换结果并逆变换
        fused_pyramid = dtcwt.Pyramid(fused_lowpass, tuple(fused_highpasses))
        fused_channel = transform.inverse(fused_pyramid)
        fused_channels.append(fused_channel)
    
    # 合并通道
    fused_img = np.stack(fused_channels, axis=-1)
    fused_img = np.clip(fused_img * 255.0, 0, 255).astype(np.uint8)
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
    
    return fused_img