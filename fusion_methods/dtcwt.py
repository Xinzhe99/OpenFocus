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

# NumPy 2.0 removed np.asfarray; restore a compatible shim for dtcwt.
if not hasattr(np, "asfarray"):
    def _asfarray_compat(arr, dtype=None):
        target_dtype = dtype or np.float_
        kind = np.dtype(target_dtype).kind
        if kind not in ("f", "c"):
            target_dtype = np.float_
        return np.asarray(arr, dtype=target_dtype)

    np.asfarray = _asfarray_compat

if not hasattr(np, "issubsctype"):
    def _issubsctype_compat(arg1, arg2):
        try:
            dtype1 = np.dtype(arg1) if not isinstance(arg1, np.dtype) else arg1
        except TypeError:
            dtype1 = np.asarray(arg1).dtype
        dtype2 = np.dtype(arg2) if not isinstance(arg2, np.dtype) else arg2
        return np.issubdtype(dtype1, dtype2)

    np.issubsctype = _issubsctype_compat

# Conditional imports to avoid errors when optional dependencies are missing.
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
    Internal implementation of the DTCWT fusion algorithm.

    Prefers the pytorch_wavelets backend (requires PyTorch).
    Falls back to the dtcwt library (NumPy implementation) when PyTorch is unavailable.
    """
    # pytorch_available = torch is not None and DTCWTForward is not None and DTCWTInverse is not None
    dtcwt_numpy_available = dtcwt is not None

    if not dtcwt_numpy_available:
        raise RuntimeError(
            "DTCWT fusion currently only supports the CPU implementation. Install dtcwt: pip install dtcwt scipy"
        )

    if use_gpu:
        print("Notice: DTCWT fusion only runs on the CPU; forcing CPU execution.")

    # if pytorch_available:
    #     return _dtcwt_fusion_pytorch(input_source, img_resize, N, use_gpu)

    print("Using the dtcwt library (NumPy implementation) for DTCWT fusion.")
    return _dtcwt_fusion_numpy(input_source, img_resize, N, False)


# def _dtcwt_fusion_pytorch(input_source, img_resize, N, use_gpu):
#     """DTCWT fusion using pytorch_wavelets (requires PyTorch)."""
#     # Original GPU/CuPy implementation is disabled so we only ship the CPU path.

GPU_DTCWT_IMPLEMENTATION = r"""
def _dtcwt_fusion_pytorch(input_source, img_resize, N, use_gpu):
    '''
    DTCWT fusion using pytorch_wavelets (requires PyTorch)
    '''
    if torch is None or F is None or DTCWTForward is None or DTCWTInverse is None:
        raise ImportError('PyTorch or pytorch_wavelets not installed')

    if use_gpu and not torch.cuda.is_available():
        print('Warning: CUDA is unavailable; falling back to CPU execution.')
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
            print('Notice: CuPy is not installed; using PyTorch for high-frequency fusion (slower).')

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
        raise ValueError('At least two images are required for fusion.')

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
    DTCWT fusion implementation using the dtcwt library (pure NumPy, no PyTorch dependency).

    Note: GPU acceleration is not supported; the use_gpu flag is ignored.
    """
    if dtcwt is None or maximum_filter is None or convolve is None:
        raise ImportError("dtcwt or scipy not installed")
    
    if use_gpu:
        print("Warning: The pure NumPy dtcwt implementation does not support GPU acceleration; using the CPU.")

    # Load images
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
        raise ValueError("At least two images are required for fusion.")

    # Convert to float RGB format
    images_rgb = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for img in images
    ]
    
    # High-frequency coefficient fusion (pure NumPy implementation)
    def fuse_highfreq_numpy(coeffs_list, window_size=3):
        """
        Fuse the high-frequency coefficients from multiple images.
        coeffs_list: high-frequency coefficients for this level, shaped (height, width, 6) as complex arrays.
        """
        num_imgs = len(coeffs_list)
        if num_imgs == 1:
            return coeffs_list[0]
        
        # Fuse two images at a time if more than two inputs are provided
        if num_imgs > 2:
            fused = coeffs_list[0]
            for i in range(1, num_imgs):
                fused = fuse_highfreq_numpy([fused, coeffs_list[i]], window_size)
            return fused
        
        coeff1, coeff2 = coeffs_list[0], coeffs_list[1]
        height, width, num_dirs = coeff1.shape
        
        fused = np.zeros_like(coeff1)
        
        for d in range(num_dirs):
            # Compute magnitudes
            mag1 = np.abs(coeff1[:, :, d])
            mag2 = np.abs(coeff2[:, :, d])
            
            # Local maximum magnitudes
            A1 = maximum_filter(mag1, size=window_size, mode='reflect')
            A2 = maximum_filter(mag2, size=window_size, mode='reflect')
            
            # Initial mask
            initial_mask = (A1 > A2).astype(np.float32)
            
            # Consistency check
            kernel = np.ones((window_size, window_size), dtype=np.float32)
            conv_result = convolve(initial_mask, kernel, mode='constant', cval=0)
            
            threshold = (window_size * window_size) / 2
            W = (conv_result > threshold).astype(np.float32)
            
            # Blend using the binary mask
            fused[:, :, d] = W * coeff1[:, :, d] + (1 - W) * coeff2[:, :, d]
        
        return fused
    
    # Create the DTCWT transformer
    transform = dtcwt.Transform2d()
    
    # Fuse each channel independently
    fused_channels = []
    
    for channel in range(3):  # RGB channels
        # Perform DTCWT on the current channel
        all_transforms = []
        for img in images_rgb:
            t = transform.forward(img[:, :, channel], nlevels=N)
            all_transforms.append(t)
        
        # Fuse low-frequency coefficients by averaging
        lowpass_stack = np.stack([t.lowpass for t in all_transforms], axis=0)
        fused_lowpass = np.mean(lowpass_stack, axis=0)
        
        # Fuse high-frequency coefficients
        fused_highpasses = []
        for level in range(N):
            level_coeffs = [t.highpasses[level] for t in all_transforms]
            fused_level = fuse_highfreq_numpy(level_coeffs)
            fused_highpasses.append(fused_level)
        
        # Reconstruct the fused pyramid and perform the inverse transform
        fused_pyramid = dtcwt.Pyramid(fused_lowpass, tuple(fused_highpasses))
        fused_channel = transform.inverse(fused_pyramid)
        fused_channels.append(fused_channel)
    
    # Merge the channels back into a single image
    fused_img = np.stack(fused_channels, axis=-1)
    fused_img = np.clip(fused_img * 255.0, 0, 255).astype(np.uint8)
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
    
    return fused_img