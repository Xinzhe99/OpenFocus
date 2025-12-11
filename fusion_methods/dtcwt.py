# Reference:
# Lewis J J, O’Callaghan R J, Nikolov S G, et al. Pixel-and region-based image fusion with complex wavelets[J]. Information fusion, 2007, 8(2): 119-130.

import os
import re
import glob
import numpy as np
from typing import Union, List, Tuple, Optional
import cv2

# NumPy 2.0 compatibility shims
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

# Optional imports with checks
try:
    import dtcwt
    from scipy.ndimage import maximum_filter, convolve
except ImportError:
    dtcwt = None
    maximum_filter = None
    convolve = None


def _dtcwt_impl(input_source, img_resize, N, use_gpu):
    """
    DTCWT fusion implementation (Optimized CPU version).
    
    Args:
        input_source: List of images or path to image directory.
        img_resize: Tuple (width, height) or None.
        N: Number of wavelet decomposition levels.
        use_gpu: Ignored in this optimized NumPy version (CPU is used).
    """
    if dtcwt is None or maximum_filter is None or convolve is None:
        raise RuntimeError(
            "DTCWT fusion requires 'dtcwt' and 'scipy'. "
            "Please install them via: pip install dtcwt scipy"
        )

    # 1. Load and Preprocess Images
    images = _load_images(input_source, img_resize)
    
    if len(images) < 2:
        raise ValueError("At least two images are required for fusion.")

    # Convert to float32 RGB [0, 1]
    # Processing images as a batch is not easily possible with standard dtcwt library 
    # (which expects 2D inputs), so we prepare them for channel-wise processing.
    images_rgb = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for img in images
    ]

    # 2. Define Optimized Fusion Function (Vectorized)
    def fuse_highfreq_vectorized(coeffs_list, window_size=3):
        """
        Optimized high-frequency fusion.
        Vectorizes operations over the 6 wavelet directions to avoid Python loops.
        """
        num_imgs = len(coeffs_list)
        if num_imgs == 1:
            return coeffs_list[0]
        
        # Recursive pairwise fusion if more than 2 images
        if num_imgs > 2:
            fused = coeffs_list[0]
            for i in range(1, num_imgs):
                fused = fuse_highfreq_vectorized([fused, coeffs_list[i]], window_size)
            return fused
        
        # Pairwise Fusion
        c1, c2 = coeffs_list[0], coeffs_list[1]
        
        # c1 shape is typically (H, W, 6).
        # We want to filter spatially (H, W) but independently for each direction (6).
        # We construct a 3D kernel: (window_size, window_size, 1).
        footprint = np.ones((window_size, window_size, 1), dtype=bool)

        # 1. Compute Magnitudes
        mag1 = np.abs(c1)
        mag2 = np.abs(c2)

        # 2. Activity Level Measurement (Max Filter)
        # Applying 3D filter with size (3,3,1) acts as 2D filter on each of the 6 slices in parallel.
        A1 = maximum_filter(mag1, footprint=footprint, mode='reflect')
        A2 = maximum_filter(mag2, footprint=footprint, mode='reflect')

        # 3. Initial Mask Generation
        initial_mask = A1 > A2  # Boolean array (H, W, 6)

        # 4. Consistency Verification (Majority Filter / Convolution)
        # Using a float kernel of ones to count neighbors
        kernel_weights = np.ones((window_size, window_size, 1), dtype=np.float32)
        
        # Convolve input mask (converted to float) with kernel
        count_map = convolve(initial_mask.astype(np.float32), kernel_weights, mode='constant', cval=0.0)

        # Threshold: if more than half the window supports source 1, use source 1
        threshold = (window_size * window_size) / 2.0
        W = count_map > threshold  # Final Boolean Mask (H, W, 6)

        # 5. Final Blending
        # Use boolean indexing or multiplication. 
        # W is boolean: True -> c1, False -> c2
        fused = np.where(W, c1, c2)
        
        return fused

    # 3. Perform DTCWT and Fusion
    transform = dtcwt.Transform2d()
    fused_channels = []

    # Process R, G, B channels sequentially
    # (Parallelizing this loop via ThreadPool gives diminishing returns due to GIL, 
    # vectorizing the inner fusion is the most effective optimization)
    for channel_idx in range(3):
        # Forward Transform
        transforms = [
            transform.forward(img[:, :, channel_idx], nlevels=N) 
            for img in images_rgb
        ]
        
        # Fuse Low-pass (Average)
        # Stack low-passes to (Num_Images, H, W) then mean
        lowpass_stack = np.stack([t.lowpass for t in transforms], axis=0)
        fused_lowpass = np.mean(lowpass_stack, axis=0)
        
        # Fuse High-pass (Rule-based)
        fused_highpasses = []
        for level in range(N):
            # Extract coefficients for this level from all images
            # Each t.highpasses[level] is usually (H, W, 6) complex array
            level_coeffs = [t.highpasses[level] for t in transforms]
            
            # Apply vectorized fusion
            fused_level = fuse_highfreq_vectorized(level_coeffs)
            fused_highpasses.append(fused_level)
        
        # Inverse Transform
        fused_pyramid = dtcwt.Pyramid(fused_lowpass, tuple(fused_highpasses))
        fused_channel = transform.inverse(fused_pyramid)
        fused_channels.append(fused_channel)

    # 4. Reconstruct Final Image
    fused_img = np.stack(fused_channels, axis=-1)
    
    # Clip and Convert
    fused_img = np.clip(fused_img * 255.0, 0, 255).astype(np.uint8)
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
    
    return fused_img


def _load_images(input_source: Union[str, List[np.ndarray]], img_resize: Optional[Tuple[int, int]]) -> List[np.ndarray]:
    """Helper to load images from a path or list."""
    images = []
    if isinstance(input_source, str):
        if not os.path.exists(input_source):
             raise ValueError(f"Input path does not exist: {input_source}")
             
        filenames = os.listdir(input_source)
        if not filenames:
            return []
            
        # Infer extension from first file
        suffixes = [os.path.splitext(f)[1] for f in filenames if os.path.splitext(f)[1]]
        img_ext = suffixes[0] if suffixes else ''
        
        glob_pattern = os.path.join(input_source, '*' + img_ext)
        img_paths = glob.glob(glob_pattern)
        
        # Sort numerically if possible
        def sort_key(x):
            nums = re.findall(r"\d+", os.path.basename(x))
            return int(nums[-1]) if nums else x
            
        img_paths.sort(key=sort_key)

        for p in img_paths:
            img = cv2.imread(p)
            if img is not None:
                if img_resize:
                    img = cv2.resize(img, img_resize)
                images.append(img)
    else:
        for img in input_source:
            if img_resize:
                img = cv2.resize(img, img_resize)
            images.append(img)
            
    return images