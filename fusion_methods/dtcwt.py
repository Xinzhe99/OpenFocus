# Reference:
# Lewis J J, O’Callaghan R J, Nikolov S G, et al. Pixel-and region-based image fusion with complex wavelets[J]. Information fusion, 2007, 8(2): 119-130.

import os
import math
import concurrent.futures
import numpy as np
from typing import Union, List, Tuple, Optional
import cv2

from utils.image_utils import fuse_output_dtype
from fusion_methods.dct import _ensure_color_image
from core.frame_storage import create_frame, store_frame, capture_scratch, run_with_scratch

# NumPy 2.0 compatibility shims
if not hasattr(np, "asfarray"):
    def _asfarray_compat(arr, dtype=None):
        target_dtype = dtype or np.float64
        kind = np.dtype(target_dtype).kind
        if kind not in ("f", "c"):
            target_dtype = np.float64
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


class _MonotonicProgress:
    """Wrap progress_callback so percentage and step counts never regress on fallback."""

    def __init__(self, callback):
        self._cb = callback
        self._best_frac = -1.0
        self._best_done = 0

    def __call__(self, done, total):
        if self._cb is None:
            return
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        frac = float(done) / float(total)
        if frac < self._best_frac:
            return
        if done < self._best_done:
            done = self._best_done
        self._best_frac = frac
        self._best_done = done
        self._cb(done, total)


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

    Counts active transform and reconstruction scratch. Source descriptors do
    not retain decoded pixels; decision and low-pass weight planes are mapped.
    Source decoding and cache growth are added by the stage owner.
    """
    height = max(1, int(height))
    width = max(1, int(width))
    channels = max(1, int(channels))
    shapes, (low_h, low_w) = pyramid_shape_envelope(height, width, nlevels)
    float32 = np.dtype(np.float32).itemsize
    complex64 = np.dtype(np.complex64).itemsize
    plane = height * width * float32
    activity = sum(rh * rw * float32 for rh, rw in shapes)
    pyramid = (sum(rh * rw * 6 * complex64 for rh, rw in shapes)
               + low_h * low_w * float32)
    weights = low_h * low_w * float32
    # Weight generation and normalization stream one mapped plane at a time.
    fixed = (int(new_source_bytes) + 2 * activity
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
    scratch = capture_scratch()

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
                window.append(pool.submit(run_with_scratch, scratch, forward, item))
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
                    window.append(pool.submit(run_with_scratch, scratch, forward, item))
        finally:
            for future in window:
                future.cancel()
            window.clear()




def _dtcwt_impl(input_source, img_resize, N, use_gpu, thread_count=None,
                should_cancel=None, execution_plan=None,
                progress_callback=None, device_callback=None):
    """
    DTCWT fusion implementation (OpenCL GPU when enabled/admitted, CPU fallback).

    Args:
        input_source: List of images or path to image directory.
        img_resize: Tuple (width, height) or None.
        N: Number of wavelet decomposition levels.
        use_gpu: Whether to attempt OpenCL GPU acceleration before CPU fallback.
        thread_count: Upper bound on concurrent frame transforms.
        should_cancel: Cooperative cancellation callback.
        execution_plan: The resource plan its owner already admitted. Tiled
            callers pass a serial child plan; without one this function takes
            its own lease and plans from the explicit working-set estimate.
        progress_callback: Optional ``(current, total)`` progress reporter.
        device_callback: Optional callback receiving selected device string.
    """
    progress = (
        progress_callback
        if isinstance(progress_callback, _MonotonicProgress)
        else (_MonotonicProgress(progress_callback) if progress_callback is not None else None)
    )
    if execution_plan is not None:
        # The owner holds the lease; a pool thread must not re-acquire it.
        return _dtcwt_impl_planned(
            input_source, img_resize, N, thread_count, should_cancel,
            execution_plan, use_gpu=use_gpu, progress_callback=progress,
            device_callback=device_callback)

    from core import resource_policy as policy

    with policy.heavy_job(should_cancel):
        _check_cancelled(should_cancel)
        input_source = _load_images(input_source, img_resize, should_cancel)
        img_resize = None
        if not input_source:
            raise ValueError('Fusion requires a non-empty image stack')
        if use_gpu and dtcwt is not None:
            result = _try_gpu(input_source, N, should_cancel, progress, device_callback)
            if result is not None:
                return result
        height, width = input_source[0].shape[:2]
        frames = len(input_source)
        fixed, per_worker = estimate_working_set(height, width, N, frames)
        from core.source_frames import frame_resolution_bytes, source_cache_growth_bytes
        fixed += source_cache_growth_bytes(input_source)
        per_worker += max(frame_resolution_bytes(frame) for frame in input_source)
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
            input_source, img_resize, N, thread_count, should_cancel, plan,
            use_gpu=False, progress_callback=progress, device_callback=device_callback)


def _check_cancelled(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def _try_gpu(images, nlevels, should_cancel, progress, device_callback):
    from core.source_frames import SOURCE_CACHE
    from fusion_methods._openfocus_gpu import GpuError, log_fallback
    from fusion_methods._openfocus_gpu_dtcwt import GpuDtcwt

    if len(images) < 2:
        raise ValueError('At least two images are required for fusion.')
    _check_cancelled(should_cancel)
    try:
        gpu = GpuDtcwt()
        try:
            result = gpu.fuse(images, nlevels, should_cancel, progress)
        finally:
            SOURCE_CACHE.clear()
    except GpuError as exc:
        _check_cancelled(should_cancel)
        log_fallback(exc)
        return None
    _check_cancelled(should_cancel)
    if device_callback is not None:
        device_callback(gpu.device_name)
    return result


def _dtcwt_impl_planned(input_source, img_resize, N, thread_count,
                        should_cancel, execution_plan,
                        use_gpu=False, progress_callback=None,
                        device_callback=None):
    if dtcwt is None or maximum_filter is None or convolve is None:
        raise RuntimeError(
            "DTCWT fusion requires dtcwt, scipy and six. On Python 3.13+ "
            "install six, then install dtcwt==0.14.0 with --no-deps; "
            "see the user manual for the upstream NumPy constraint."
        )

    def _check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    progress = (
        progress_callback
        if isinstance(progress_callback, _MonotonicProgress)
        else (_MonotonicProgress(progress_callback) if progress_callback is not None else None)
    )

    _check_cancel()
    # 1. Load and preprocess one frame at a time.
    images = _load_images(input_source, img_resize, should_cancel=should_cancel)

    if len(images) < 2:
        raise ValueError("At least two images are required for fusion.")

    if use_gpu:
        result = _try_gpu(images, N, should_cancel, progress, device_callback)
        if result is not None:
            return result

    out_dtype = fuse_output_dtype(images)
    h, w = images[0].shape[:2]
    n_frames = len(images)

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

    from core.source_frames import resolve_frame

    total_steps = 2 * n_frames + 1  # final inverse transform and packing
    if progress is not None:
        progress(0, total_steps)

    def prepare_frame(index):
        _check_cancel()
        image = resolve_frame(images[index], should_cancel=should_cancel)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        image = _ensure_color_image(image)
        if image.dtype == np.uint16:
            image = image.astype(np.float32) * (255.0 / 65535.0)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    transform = dtcwt.Transform2d()

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
                    best_activity[level] = store_frame(activity)
                    frame_indices[level] = create_frame(activity.shape, np.int32)
                    frame_indices[level][...] = 0
                else:
                    win = activity > best_activity[level]
                    best_activity[level][win] = activity[win]
                    frame_indices[level][win] = i
                    del win
                del activity
            del pyr
            if progress is not None:
                progress(i + 1, total_steps)
    finally:
        pyramids.close()
    del best_activity

    # This exact integer vote is independent of stack length. Keep upstream
    # floating-point transforms and accumulation order for CPU parity.
    from fusion_methods._dtcwt_kernels import consistent_vote

    vote = consistent_vote()

    decisions = []
    for level in range(N):
        _check_cancel()
        labels = np.asarray(frame_indices[level])
        decision = create_frame(labels.shape, np.int32)
        output = np.asarray(decision)
        for start in range(0, labels.shape[0], 128):
            _check_cancel()
            vote(labels, output, start, min(start + 128, labels.shape[0]))
        del labels, output
        decisions.append(decision)
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
    fused_img = create_frame((h, w, 3), out_dtype)
    for c in range(3):
        fused_high = None
        lp = None
        _check_cancel()

        def prepare_channel(index, _channel=c):
            _check_cancel()
            image = resolve_frame(images[index], should_cancel=should_cancel)
            # BGR->RGB is a channel permutation. Select the plane before
            # float conversion so a view cannot pin a full RGB float frame.
            if image.ndim == 2:
                plane = image
            elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
                plane = image[:, :, 0 if image.shape[2] == 1 else 2 - _channel]
            else:
                raise ValueError(f'Unsupported image shape: {image.shape}')
            if plane.dtype == np.uint16:
                plane = plane.astype(np.float32) * (255.0 / 65535.0)
            return plane.astype(np.float32) / 255.0

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
                        lp_weights = []
                        total_weight = create_frame(lp_shape, np.float32)
                        total_weight[...] = 0
                        for k in range(n_frames):
                            _check_cancel()
                            m = cv2.resize((coarse_idx == k).astype(np.float32),
                                           (lp_shape[1], lp_shape[0]),
                                           interpolation=cv2.INTER_LINEAR)
                            weight = store_frame(cv2.GaussianBlur(m, (0, 0), 1.5))
                            lp_weights.append(weight)
                            total_weight += weight
                        np.maximum(total_weight, 1e-6, out=total_weight)
                        for weight in lp_weights:
                            _check_cancel()
                            np.divide(weight, total_weight, out=weight)
                        del m, weight, total_weight
                for level in range(N):
                    _check_cancel()
                    sel = decisions[level] == i
                    fused_high[level][sel] = pyr.highpasses[level][sel]
                    del sel
                lp += lp_weights[i] * pyr.lowpass
                del pyr
                if progress is not None:
                    step = n_frames + ((c * n_frames + i + 1) * n_frames) // (3 * n_frames)
                    progress(step, total_steps)
        finally:
            channel_pyramids.close()
        _check_cancel()
        rec = transform.inverse(dtcwt.Pyramid(lp, tuple(fused_high)))
        # dtcwt pads odd sizes by duplicating the last row/column
        for start in range(0, h, 128):
            _check_cancel()
            rows = slice(start, min(h, start + 128))
            fused_img[rows, :, 2 - c] = np.clip(rec[rows, :w] * 255.0, 0, 255)
        del fused_high, lp, rec

    if progress is not None:
        progress(total_steps, total_steps)
    if device_callback is not None:
        device_callback('CPU')
    _check_cancel()

    return fused_img


def _load_images(input_source: Union[str, List[np.ndarray]], img_resize: Optional[Tuple[int, int]],
                 should_cancel=None) -> List[np.ndarray]:
    """Helper to load images from a path or list."""
    from core.source_frames import process_frame
    images = []
    if isinstance(input_source, str):

        if not os.path.exists(input_source):
             raise ValueError(f"Input path does not exist: {input_source}")
             
        from fusion_methods.openfocus_ab import _collect_directory_sources
        images, _ = _collect_directory_sources(input_source, should_cancel)
    else:
        for img in input_source:
            if should_cancel is not None and should_cancel():
                from core.registration import RegistrationCancelled
                raise RegistrationCancelled()
            images.append(img)
            
    if img_resize:
        images = [process_frame(img, lambda image: cv2.resize(image, img_resize),
                                shape=(img_resize[1], img_resize[0]) + (img.shape[2:]
                                    if img.ndim != 3 or img.shape[2] != 1 else ()),
                                should_cancel=should_cancel) for img in images]
    return images