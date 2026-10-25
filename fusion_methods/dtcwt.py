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


# The dtcwt package is pure NumPy: its column convolution runs in a Python
# loop but spends its time inside NumPy ufuncs, which release the GIL. Frames
# are transformed independently, so a small thread pool overlaps them and the
# result is unchanged (each frame's coefficients are computed by exactly the
# same call as before).
_DTCWT_MAX_WORKERS = min(4, max(1, (os.cpu_count() or 4) - 1))
# The activity pass keeps one full pyramid per in-flight frame, so its worker
# count trades memory for speed (measured: 3 workers = 1.55x but +68 MB peak).
# The channel pass streams frames inside each channel, so running the three
# channels concurrently costs no meaningful extra memory.
_DTCWT_ACTIVITY_WORKERS = 2


def _iter_forward(transform, planes, nlevels, workers=None):
    """Yield transform.forward() for each plane, overlapping a few at a time.

    A generator on purpose: the caller consumes one pyramid, derives its
    activity maps and drops it. Returning a list would keep every frame's
    pyramid alive at once (~21 MB each at 720x960, so 5 frames = +84 MB peak
    for no reason). Results stay in input order.
    """
    if workers is None:
        workers = _DTCWT_MAX_WORKERS
    if len(planes) < 2 or workers < 2:
        for plane in planes:
            yield transform.forward(plane, nlevels=nlevels)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded submission: at most `workers` transforms are in flight, so at
        # most that many pyramids exist at once. pool.map() would submit every
        # plane immediately and hold all finished results, which raised peak
        # memory by 67 MB on a 5-frame 720x960 stack.
        window = deque()
        it = iter(planes)
        for _ in range(workers):
            nxt = next(it, None)
            if nxt is None:
                break
            window.append(pool.submit(
                lambda p=nxt: transform.forward(p, nlevels=nlevels)))
        while window:
            fut = window.popleft()
            nxt = next(it, None)
            if nxt is not None:
                window.append(pool.submit(
                    lambda p=nxt: transform.forward(p, nlevels=nlevels)))
            yield fut.result()


def _map_forward(transform, planes, nlevels, workers=None):
    """transform.forward() over independent 2-D planes, in parallel."""
    return list(_iter_forward(transform, planes, nlevels, workers=workers))


def _dtcwt_impl(input_source, img_resize, N, use_gpu, thread_count=None,
                should_cancel=None):
    """
    DTCWT fusion implementation (Optimized CPU version).
    
    Args:
        input_source: List of images or path to image directory.
        img_resize: Tuple (width, height) or None.
        N: Number of wavelet decomposition levels.
        use_gpu: Ignored in this optimized NumPy version (CPU is used).
        thread_count: Upper bound on concurrent frame transforms.
        should_cancel: Cooperative cancellation callback.
    """
    if dtcwt is None or maximum_filter is None or convolve is None:
        raise RuntimeError(
            "DTCWT fusion requires 'dtcwt' and 'scipy'. "
            "Please install them via: pip install dtcwt scipy "
            "(dtcwt has no wheels for Python 3.13+; it installs on 3.10-3.12)"
        )

    def _check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

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

    # Respect the caller's concurrency cap instead of the module default: a
    # tiled render already runs one transform per worker, so the activity
    # pass must not open its own pool on top of that.
    if thread_count is None:
        activity_workers = _DTCWT_ACTIVITY_WORKERS
        max_workers = _DTCWT_MAX_WORKERS
    else:
        try:
            cap = max(1, int(thread_count))
        except (TypeError, ValueError):
            cap = _DTCWT_ACTIVITY_WORKERS
        activity_workers = max(1, min(_DTCWT_ACTIVITY_WORKERS, cap))
        max_workers = max(1, min(_DTCWT_MAX_WORKERS, cap))

    # 2. Activity per frame and level from the luminance transform. Only the
    # (real, |coeff| summed over the 6 orientations) activity maps are kept —
    # never all frames' complex pyramids, so memory stays near one pyramid.
    def _luma(img):
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    activities = [[] for _ in range(N)]
    _luma_planes = [_luma(img) for img in images_rgb]
    for pyr in _iter_forward(transform, _luma_planes, N,
                             workers=activity_workers):
        _check_cancel()
        for level in range(N):
            activities[level].append(
                maximum_filter(np.abs(pyr.highpasses[level]).sum(axis=2).astype(np.float32),
                               size=3, mode='reflect'))
        del pyr
    del _luma_planes

    # np.argmax over the stacked maps stays: a running max with boolean
    # fancy-index assignment measured no faster than stacking (the extra pass
    # over the mask costs about what the stack costs) and the stack is what
    # ``np.argmax``'s first-frame tie-break is defined on.
    frame_indices = [np.argmax(np.stack(act), axis=0) for act in activities]
    del activities

    def _consistent_argmax(idx, count):
        """3x3 majority consistency check on the index map.

        One decision per pixel and level, shared by all six orientations and
        all three channels: deciding each orientation/channel separately (the
        previous pairwise rule) mixed frames inside one pixel — colour fringes
        and ghosted edges.

        The per-frame vote maps are consumed one at a time: only the best
        count and its label stay alive, which is the same argmax result as
        stacking all N vote maps (strict ``>`` keeps the first-label tie) at
        one map instead of N.
        """
        best_count = None
        best_label = None
        kernel = np.ones((3, 3), np.float32)
        for i in range(count):
            votes = convolve((idx == i).astype(np.float32), kernel,
                             mode='nearest')
            if best_count is None:
                best_count = votes
                best_label = np.full(idx.shape, i, np.int32)
                continue
            win = votes > best_count
            if win.any():
                best_count[win] = votes[win]
                best_label[win] = i
            del win, votes
        return best_label

    decisions = []
    for level in range(N):
        _check_cancel()
        decisions.append(_consistent_argmax(frame_indices[level], n_frames))
        frame_indices[level] = None
    del frame_indices

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
            _check_cancel()
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