# Reference:
# Lewis J J, O’Callaghan R J, Nikolov S G, et al. Pixel-and region-based image fusion with complex wavelets[J]. Information fusion, 2007, 8(2): 119-130.

import os
import re
import glob
import numpy as np
from typing import Union, List, Tuple, Optional
import cv2

from utils.image_utils import fuse_output_dtype
from fusion_methods.dct import _ensure_color_image

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


# Activity transforms are the only parallel layer. Each worker owns its
# transform and pyramid; channel reconstruction remains serial.
_DTCWT_MAX_WORKERS = min(4, max(1, (os.cpu_count() or 4) - 1))
_DTCWT_ACTIVITY_WORKERS = 2


def _iter_forward(transform, planes, nlevels, workers=None, should_cancel=None,
                  transform_factory=None):
    """Yield ordered pyramids from a bounded, cancellation-aware window."""
    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    workers = max(1, int(_DTCWT_MAX_WORKERS if workers is None else workers))
    if workers == 1:
        for plane in planes:
            check_cancel()
            yield transform.forward(plane, nlevels=nlevels)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from threading import local

    state = local()

    def forward(plane):
        check_cancel()
        if not hasattr(state, "transform"):
            state.transform = transform_factory() if transform_factory else transform
        return state.transform.forward(plane, nlevels=nlevels)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        window = deque()
        it = iter(planes)
        try:
            for _ in range(workers):
                check_cancel()
                plane = next(it, None)
                if plane is None:
                    break
                window.append(pool.submit(forward, plane))
            while window:
                check_cancel()
                future = window.popleft()
                result = future.result()
                del future
                check_cancel()
                yield result
                del result
                check_cancel()
                plane = next(it, None)
                if plane is not None:
                    window.append(pool.submit(forward, plane))
        finally:
            for future in window:
                future.cancel()
            window.clear()




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

    _check_cancel()
    # 1. Load and preprocess one frame at a time.
    images = _load_images(input_source, img_resize, should_cancel=should_cancel)
    
    if len(images) < 2:
        raise ValueError("At least two images are required for fusion.")

    out_dtype = fuse_output_dtype(images)

    def prepare_frame(index):
        _check_cancel()
        image = _ensure_color_image(images[index])
        if image.dtype == np.uint16:
            image = image.astype(np.float32) * (255.0 / 65535.0)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    h, w = images[0].shape[:2]
    n_frames = len(images)
    transform = dtcwt.Transform2d()

    # Respect the caller's concurrency cap instead of the module default: a
    # tiled render already runs one transform per worker, so the activity
    # pass must not open its own pool on top of that.
    try:
        cap = max(1, int(thread_count)) if thread_count is not None else _DTCWT_MAX_WORKERS
    except (TypeError, ValueError):
        cap = 1
    activity_workers = max(1, min(_DTCWT_ACTIVITY_WORKERS, _DTCWT_MAX_WORKERS, cap))

    # 2. Activity per frame and level from the luminance transform. Only the
    # (real, |coeff| summed over the 6 orientations) activity maps are kept —
    # never all frames' complex pyramids, so memory stays near one pyramid.
    def _luma(img):
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    best_activity = [None] * N
    frame_indices = [None] * N
    planes = (_luma(prepare_frame(i)) for i in range(n_frames))
    pyramids = _iter_forward(transform, planes, N, workers=activity_workers,
                              should_cancel=should_cancel,
                              transform_factory=dtcwt.Transform2d)
    try:
        for i, pyr in enumerate(pyramids):
            for level in range(N):
                _check_cancel()
                activity = maximum_filter(
                    np.abs(pyr.highpasses[level]).sum(axis=2).astype(np.float32),
                    size=3, mode='reflect')
                if best_activity[level] is None:
                    best_activity[level] = activity
                    frame_indices[level] = np.zeros(activity.shape, np.int32)
                else:
                    win = activity > best_activity[level]
                    best_activity[level][win] = activity[win]
                    frame_indices[level][win] = i
                    del win
                del activity
            del pyr
    finally:
        pyramids.close()
    del best_activity

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
            _check_cancel()
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
        _check_cancel()
        for i in range(n_frames):
            img = prepare_frame(i)
            pyr = transform.forward(img[:, :, c], nlevels=N)
            if fused_high is None:
                fused_high = [np.zeros_like(hp) for hp in pyr.highpasses]
                lp = np.zeros_like(pyr.lowpass)
                if lp_weights is None:
                    lp_shape = pyr.lowpass.shape
                    ws = []
                    for k in range(n_frames):
                        _check_cancel()
                        m = cv2.resize((coarse_idx == k).astype(np.float32),
                                       (lp_shape[1], lp_shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
                        ws.append(cv2.GaussianBlur(m, (0, 0), 1.5))
                    ws = np.stack(ws)
                    lp_weights = ws / np.maximum(ws.sum(axis=0, keepdims=True), 1e-6)
                    del ws
            for level in range(N):
                _check_cancel()
                sel = decisions[level] == i
                fused_high[level][sel] = pyr.highpasses[level][sel]
            lp += lp_weights[i] * pyr.lowpass
            del pyr
        _check_cancel()
        rec = transform.inverse(dtcwt.Pyramid(lp, tuple(fused_high)))
        # dtcwt pads odd sizes by duplicating the last row/column
        fused_channels.append(rec[:h, :w])

    # 5. Reconstruct Final Image
    _check_cancel()
    fused_img = np.stack(fused_channels, axis=-1)

    # Clip and Convert
    fused_img = np.clip(fused_img * 255.0, 0, 255).astype(out_dtype)
    fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
    
    return fused_img


def _load_images(input_source: Union[str, List[np.ndarray]], img_resize: Optional[Tuple[int, int]],
                 should_cancel=None) -> List[np.ndarray]:
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
            if should_cancel is not None and should_cancel():
                from core.registration import RegistrationCancelled
                raise RegistrationCancelled()
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img_resize:
                    img = cv2.resize(img, img_resize)
                images.append(img)
    else:
        for img in input_source:
            if should_cancel is not None and should_cancel():
                from core.registration import RegistrationCancelled
                raise RegistrationCancelled()
            if img_resize:
                img = cv2.resize(img, img_resize)
            images.append(img)
            
    return images