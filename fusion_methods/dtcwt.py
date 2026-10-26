# Reference:
# Lewis J J, O’Callaghan R J, Nikolov S G, et al. Pixel-and region-based image fusion with complex wavelets[J]. Information fusion, 2007, 8(2): 119-130.

import os
import re
import glob
import math
import concurrent.futures
import numpy as np
from typing import Union, List, Tuple, Optional
import cv2

from utils.image_utils import fuse_output_dtype
from fusion_methods.dct import _ensure_color_image
from utils.sorting import natural_sort_key

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


def pyramid_shape_envelope(height: int, width: int, nlevels: int) -> tuple:
    """Conservative high-pass and low-pass shapes for a float32 pyramid.

    dtcwt duplicates the last row/column of an odd-sized image, so the first
    level starts from the next even size; each following level halves. The
    "+2" covers the symmetric-extension taps. Verified against the installed
    dtcwt for levels 1-6 and odd/even sizes; an underestimate must be fixed
    here rather than worked around at the call site.
    """
    height = max(1, int(height))
    width = max(1, int(width))
    levels = max(1, int(nlevels))
    even_h, even_w = height + (height % 2), width + (width % 2)
    shapes = []
    for level in range(1, levels + 1):
        shapes.append((math.ceil(even_h / 2 ** level) + 2,
                       math.ceil(even_w / 2 ** level) + 2))
    low_h, low_w = 2 * shapes[-1][0], 2 * shapes[-1][1]
    return tuple(shapes), (low_h, low_w)


def estimate_working_set(height: int, width: int, nlevels: int,
                         n_frames: int, channels: int = 3,
                         new_source_bytes: int = 0) -> tuple:
    """(fixed_bytes, per_worker_bytes) for one DTCWT stage.

    Counts the new allocations this module actually makes: activity/winner
    maps, the low-pass weight block built once per call, one pyramid per live
    worker, and the inverse/output overlap. Resident input arrays are the
    caller's business and are not charged twice.
    """
    height = max(1, int(height))
    width = max(1, int(width))
    frames = max(1, int(n_frames))
    channels = max(1, int(channels))
    shapes, (low_h, low_w) = pyramid_shape_envelope(height, width, nlevels)
    float32 = np.dtype(np.float32).itemsize
    complex64 = np.dtype(np.complex64).itemsize
    plane = height * width * float32
    activity = sum(rh * rw * float32 for rh, rw in shapes)
    winner = sum(rh * rw * np.dtype(np.int32).itemsize for rh, rw in shapes)
    pyramid = (sum(rh * rw * 6 * complex64 for rh, rw in shapes)
               + low_h * low_w * float32)
    weights = frames * low_h * low_w * float32
    # The first channel keeps several copies of the weight block alive while
    # it stacks and normalises them, plus a blur/resize temporary per frame.
    fixed = (int(new_source_bytes) + 2 * winner + 2 * activity
             + 3 * weights + 2 * pyramid + 12 * plane * channels)
    # A live worker holds the prepared RGB plane (a channel view keeps the
    # whole frame), its pyramid, native workspace and the selected
    # coefficients of the channel being reduced.
    per_worker = 9 * plane * channels + 3 * pyramid + 2 * activity
    return int(fixed), int(per_worker)



def _iter_forward(transform, planes, nlevels, workers=None, should_cancel=None,
                  transform_factory=None, prepare=None):
    """Yield ordered pyramids from a bounded, cancellation-aware window.

    ``prepare`` runs inside the worker that will transform the item, so the
    main thread never holds a prepared frame for a future it has not consumed
    yet. Items stay in input order, which keeps tie-breaking and the low-pass
    accumulation order identical to the serial path.
    """
    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    def ready(item):
        return prepare(item) if prepare is not None else item

    workers = max(1, int(_DTCWT_MAX_WORKERS if workers is None else workers))
    if workers == 1:
        for item in planes:
            check_cancel()
            yield transform.forward(ready(item), nlevels=nlevels)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from threading import local

    state = local()

    def forward(item):
        check_cancel()
        if not hasattr(state, "transform"):
            state.transform = transform_factory() if transform_factory else transform
        return state.transform.forward(ready(item), nlevels=nlevels)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        window = deque()
        it = iter(planes)
        try:
            for _ in range(workers):
                check_cancel()
                item = next(it, None)
                if item is None:
                    break
                window.append(pool.submit(forward, item))
            while window:
                check_cancel()
                future = window[0]
                # A later failure must stop replenishment even while an
                # earlier native call is still running.
                for pending in window:
                    if pending.done() and pending.exception() is not None:
                        pending.result()
                if not future.done():
                    concurrent.futures.wait((future,), timeout=0.05)
                    continue
                window.popleft()
                result = future.result()
                del future
                check_cancel()
                yield result
                del result
                check_cancel()
                item = next(it, None)
                if item is not None:
                    window.append(pool.submit(forward, item))
        finally:
            for future in window:
                future.cancel()
            window.clear()




def _dtcwt_impl(input_source, img_resize, N, use_gpu, thread_count=None,
                should_cancel=None, execution_plan=None):
    """
    DTCWT fusion implementation (Optimized CPU version).

    Args:
        input_source: List of images or path to image directory.
        img_resize: Tuple (width, height) or None.
        N: Number of wavelet decomposition levels.
        use_gpu: Ignored in this optimized NumPy version (CPU is used).
        thread_count: Upper bound on concurrent frame transforms.
        should_cancel: Cooperative cancellation callback.
        execution_plan: The resource plan its owner already admitted. Tiled
            callers pass a serial child plan; without one this function takes
            its own lease and plans from the explicit working-set estimate.
    """
    if execution_plan is not None:
        # The owner holds the lease; a pool thread must not re-acquire it.
        return _dtcwt_impl_planned(
            input_source, img_resize, N, thread_count, should_cancel,
            execution_plan)

    from core import resource_policy as policy
    from core.registration import RegistrationCancelled

    with policy.heavy_job(should_cancel):
        _check_cancelled(should_cancel)
        height, width, frames, new_bytes = _peek_source(
            input_source, img_resize, should_cancel)
        fixed, per_worker = estimate_working_set(
            height, width, N, frames, new_source_bytes=new_bytes)
        plan = policy.plan_execution(
            policy.Workload(
                algorithm="dtcwt", frames=frames, height=height, width=width,
                channels=3, itemsize=1,
                fixed_bytes=fixed, per_worker_bytes=per_worker,
                outer_tasks=frames),
            preference="balanced", user_threads=thread_count)
        for reason in plan.reasons:
            print(f"Info: resource plan: {reason}")
        _check_cancelled(should_cancel)
        return _dtcwt_impl_planned(
            input_source, img_resize, N, thread_count, should_cancel, plan)


def _check_cancelled(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def _peek_source(input_source, img_resize, should_cancel=None):
    """(height, width, frames, new bytes this call will allocate) up front."""
    import os
    if isinstance(input_source, str) and os.path.isdir(input_source):
        from core.image_loader import decode_image
        names = sorted(
            (f for f in os.listdir(input_source)
             if os.path.splitext(f)[1].lower() in
             ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.bmp', '.tif',
              '.tiff', '.webp')),
            key=natural_sort_key)
        if not names:
            raise ValueError(f"No image files found in directory: {input_source}")
        first = decode_image(os.path.join(input_source, names[0]))
        if first is None:
            raise RuntimeError(f"Unable to read image: {names[0]}")
        height, width = first.shape[:2]
        frames = len(names)
        # Every frame is decoded into a new array inside the stage.
        new_bytes = frames * height * width * np.dtype(first.dtype).itemsize
        if first.ndim == 2:
            new_bytes *= 3
    else:
        frames = len(input_source)
        if not frames:
            raise ValueError("Fusion requires a non-empty image stack")
        first = input_source[0]
        height, width = first.shape[:2]
        new_bytes = 0
    if img_resize:
        height, width = int(img_resize[1]), int(img_resize[0])
    return max(1, height), max(1, width), max(1, frames), int(new_bytes)


def _dtcwt_impl_planned(input_source, img_resize, N, thread_count,
                        should_cancel, execution_plan):
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
    if execution_plan is not None:
        # The owner already admitted this stage; use its decision verbatim.
        cap = max(1, min(cap, int(execution_plan.workers)))
    activity_workers = max(1, min(_DTCWT_ACTIVITY_WORKERS, _DTCWT_MAX_WORKERS, cap))
    channel_workers = max(1, min(_DTCWT_MAX_WORKERS, cap, n_frames))


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
    #
    # The three channels stay in order and each one's frames are consumed in
    # frame order, so every float32 addition happens in the same sequence as
    # the serial version. Only the independent forward transforms run in a
    # bounded pool; one channel finishes and is inverted before the next
    # channel starts, which keeps the inverse workspace out of the forward
    # working set.
    fused_channels = []
    for c in range(3):
        fused_high = None
        lp = None
        _check_cancel()

        def prepare_channel(index, _channel=c):
            return prepare_frame(index)[:, :, _channel]

        channel_pyramids = _iter_forward(
            transform, range(n_frames), N, workers=channel_workers,
            should_cancel=should_cancel,
            transform_factory=dtcwt.Transform2d,
            prepare=prepare_channel)
        try:
            for i, pyr in enumerate(channel_pyramids):
                _check_cancel()
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
                    del sel
                lp += lp_weights[i] * pyr.lowpass
                del pyr
        finally:
            channel_pyramids.close()
        _check_cancel()
        rec = transform.inverse(dtcwt.Pyramid(lp, tuple(fused_high)))
        # dtcwt pads odd sizes by duplicating the last row/column
        fused_channels.append(rec[:h, :w])
        del fused_high, lp, rec


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