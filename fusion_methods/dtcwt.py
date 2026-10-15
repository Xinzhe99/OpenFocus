# Reference:
# Lewis J J, O’Callaghan R J, Nikolov S G, et al. Pixel-and region-based image fusion with complex wavelets[J]. Information fusion, 2007, 8(2): 119-130.

import os
import re
import glob
import numpy as np
from typing import Union, List, Tuple, Optional
import cv2

from utils.image_utils import fuse_output_dtype

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
    out_dtype = fuse_output_dtype(images)
    images_rgb = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for img in images
    ]

    h, w = images_rgb[0].shape[:2]
    n_frames = len(images_rgb)
    transform = dtcwt.Transform2d()

    # 2. Activity per frame and level from the luminance transform. Only the
    # (real, |coeff| summed over the 6 orientations) activity maps are kept —
    # never all frames' complex pyramids, so memory stays near one pyramid.
    def _luma(img):
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    activities = [[] for _ in range(N)]
    for img in images_rgb:
        pyr = transform.forward(_luma(img), nlevels=N)
        for level in range(N):
            activities[level].append(
                maximum_filter(np.abs(pyr.highpasses[level]).sum(axis=2).astype(np.float32),
                               size=3, mode='reflect'))
        del pyr

    def _consistent_argmax(act):
        """N-way argmax + 3x3 majority consistency check on the index map.

        One decision per pixel and level, shared by all six orientations and
        all three channels: deciding each orientation/channel separately (the
        previous pairwise rule) mixed frames inside one pixel — colour fringes
        and ghosted edges."""
        idx = np.argmax(np.stack(act), axis=0)
        votes = np.stack([convolve((idx == i).astype(np.float32),
                                   np.ones((3, 3), np.float32), mode='nearest')
                          for i in range(n_frames)])
        return np.argmax(votes, axis=0)

    decisions = [_consistent_argmax(activities[level]) for level in range(N)]
    del activities

    # 3. Low-pass weights: soft selection driven by the coarsest decision. A
    # plain mean over all frames (the previous rule) washed the defocused
    # frames' low frequencies into every region — veiling glare / low
    # contrast on deep stacks.
    coarse_idx = decisions[-1]
    lp_weights = None

    # 4. Per channel, stream the frames: forward, keep the selected
    # coefficients, accumulate the weighted low-pass.
    fused_channels = []
    for c in range(3):
        fused_high = None
        lp = None
        for i, img in enumerate(images_rgb):
            pyr = transform.forward(img[:, :, c], nlevels=N)
            if fused_high is None:
                fused_high = [np.zeros_like(hp) for hp in pyr.highpasses]
                lp = np.zeros_like(pyr.lowpass)
                if lp_weights is None:
                    lp_shape = pyr.lowpass.shape
                    ws = []
                    for k in range(n_frames):
                        m = cv2.resize((coarse_idx == k).astype(np.float32),
                                       (lp_shape[1], lp_shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
                        ws.append(cv2.GaussianBlur(m, (0, 0), 1.5))
                    ws = np.stack(ws)
                    lp_weights = ws / np.maximum(ws.sum(axis=0, keepdims=True), 1e-6)
            for level in range(N):
                sel = decisions[level] == i
                fused_high[level][sel] = pyr.highpasses[level][sel]
            lp += lp_weights[i] * pyr.lowpass
            del pyr
        rec = transform.inverse(dtcwt.Pyramid(lp, tuple(fused_high)))
        # dtcwt pads odd sizes by duplicating the last row/column
        fused_channels.append(rec[:h, :w])

    # 5. Reconstruct Final Image
    fused_img = np.stack(fused_channels, axis=-1)

    # Clip and Convert
    fused_img = np.clip(fused_img * 255.0, 0, 255).astype(out_dtype)
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