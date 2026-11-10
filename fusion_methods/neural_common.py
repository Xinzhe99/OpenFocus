"""Shared staging, cancellation and resource planning for neural fusion."""
import os

import cv2
import numpy as np

from utils.sorting import natural_sort_key
ATTENTION_PAD_ALIGN = 64



def cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()




def read_stack(input_source, img_resize, should_cancel=None, execution_plan=None,
               *, preserve_grayscale=True):
    """Read source descriptors and defer each exact backend resize to its consumer."""
    from core.source_frames import scan_source, process_frame
    if isinstance(input_source, str):
        exts = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp',
                '.tif', '.tiff', '.bmp')
        names = sorted((name for name in os.listdir(input_source)
                        if name.lower().endswith(exts)), key=natural_sort_key)
        frames = []
        for name in names:
            cancel(should_cancel)
            frames.extend(scan_source(os.path.join(input_source, name), header_only=True,
                                      should_cancel=lambda: cancel(should_cancel)))
        frames = [process_frame(frame, lambda image: image[:, :, 0], shape=frame.shape[:2],
                                should_cancel=should_cancel, scratch_bytes=0)
                  if preserve_grayscale and frame.grayscale else frame for frame in frames]
    else:
        frames = list(input_source)
    if not frames:
        raise ValueError("Empty image list")
    if img_resize:
        frames = [process_frame(frame, lambda image: cv2.resize(image, img_resize),
                    shape=(img_resize[1], img_resize[0]) + (frame.shape[2:]
                        if frame.ndim != 3 or frame.shape[2] != 1 else ()),
                    should_cancel=should_cancel) for frame in frames]
    if execution_plan is not None:
        from core.source_frames import frame_resolution_bytes, source_cache_growth_bytes
        from core.resource_policy import require_memory
        required = max(frame_resolution_bytes(frame) for frame in frames) + source_cache_growth_bytes(frames)
        if required:
            require_memory(required, user_limit=execution_plan.ram_budget)
    return frames


def working_frame(frame, normalize_stack):
    """Convert a single frame exactly as 16-bit stack normalization does."""
    from core.source_frames import resolve_frame
    frame = resolve_frame(frame)
    if not normalize_stack or frame.dtype == np.float32:
        return frame
    result = frame.astype(np.float32)
    if frame.dtype == np.uint16:
        result *= 255.0 / 65535.0
    return result


def _plane(frame):
    """One 0-255 luma plane as float32, independent of source bit depth."""
    if frame.ndim == 2:
        gray = frame
    elif frame.shape[2] == 1:
        gray = frame[:, :, 0]
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 65535.0 if gray.dtype == np.uint16 else 255.0
    return gray.astype(np.float32, copy=False) / scale


def stage_planes(frames, align, should_cancel=None):
    """Stage a padded image grid as float32 luma at 0-1.

    Pad before native colour conversion so whole and tiled inference use the
    same conversion grid, including scalar/SIMD rounding at image edges.
    """
    h, w = frames[0].shape[:2]
    hp, wp = h + (-h) % align, w + (-w) % align
    ph, pw = hp - h, wp - w
    mode = 'reflect' if ph < h and pw < w else 'edge'
    batch = np.empty((1, len(frames), hp, wp), dtype=np.float32)
    normalize_stack = any(frame.dtype == np.uint16 for frame in frames)
    for index, frame in enumerate(frames):
        cancel(should_cancel)
        frame = working_frame(frame, normalize_stack)
        if ph or pw:
            padding = ((0, ph), (0, pw)) + ((0, 0),) * (frame.ndim - 2)
            frame = np.pad(frame, padding, mode=mode)
        batch[0, index] = _plane(frame)
    cancel(should_cancel)
    return batch


def normalized_frames(frames, should_cancel=None):
    """Describe normalization without retaining processed full colour sources."""
    from core.source_frames import normalize_frames
    cancel(should_cancel)
    return normalize_frames(frames, should_cancel)


def plan_attention_inference(algorithm, frames, model_path, device,
                             thread_count=None,
                             resource_preference='balanced', execution_plan=None,
                             result_count=1, fixed_bytes=0, height=None, width=None):
    """Admit the compact runtime's working set before creating a session."""
    from core.resource_policy import Workload, plan_execution, probe_resources
    from fusion_methods.neural_runtime import estimate_inference_bytes

    if execution_plan is not None:
        device = getattr(execution_plan, 'device_kind', device)
        thread_count = min(thread_count or execution_plan.native_threads,
                           execution_plan.native_threads)
    h, w = frames[0].shape[:2]
    h = int(height) if height is not None else h
    w = int(width) if width is not None else w
    n = len(frames)
    c = frames[0].shape[2] if frames[0].ndim == 3 else 1
    hp, wp = h + (-h) % 64, w + (-w) % 64
    inference = estimate_inference_bytes(n, h, w, device)
    host = (fixed_bytes + n * hp * wp * 4 * (2 * c + 3)
            + result_count * h * w * c * 4 + inference)
    from core.source_frames import frame_resolution_bytes, source_cache_growth_bytes
    host += max(frame_resolution_bytes(frame) for frame in frames) + source_cache_growth_bytes(frames)
    workload = Workload(algorithm=algorithm, frames=n, height=h, width=w,
                        channels=c, itemsize=max(f.dtype.itemsize for f in frames),
                        parallel_layer='native', fixed_bytes=host, per_worker_bytes=0)
    plan = plan_execution(workload, probe_resources(), preference=resource_preference,
                          user_threads=thread_count,
                          ram_budget_override=execution_plan.ram_budget if execution_plan else None)
    from dataclasses import replace
    return model_path, device, replace(plan, device_kind=device)


def _ycbcr(frame):
    """BT.601 Y/Cb/Cr in 0-255, independent of cv2's float input range."""
    data = frame.astype(np.float32, copy=False)
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]
    if data.ndim == 2:
        data = np.repeat(data[:, :, None], 3, axis=2)
    blue, green, red = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    return luma, (blue - luma) * 0.564 + 128.0, (red - luma) * 0.713 + 128.0


def _ycbcr_to_bgr(luma, cb, cr):
    luma = luma.astype(np.float32, copy=False)
    cb = cb - 128.0
    cr = cr - 128.0
    blue = luma + 1.773 * cb
    green = luma - 0.714 * cr - 0.344 * cb
    red = luma + 1.403 * cr
    return np.stack((blue, green, red), axis=-1)


def tensor_to_bgr(fused_luma, frames, should_cancel=None):
    """Colour reconstruction from the frame whose luma matches the prediction.

    This mirrors the StackMFF reference: the fused luma is kept and Cb/Cr are
    taken from the source frame closest to it, so no new colours are invented.
    The conversion is written out because cv2's float32 YCrCb path expects
    0-1 inputs, while every buffer here is in the project's 0-255 range.
    """
    cancel(should_cancel)
    normalize_stack = any(frame.dtype == np.uint16 for frame in frames)
    target = np.clip(fused_luma, 0.0, 255.0)
    best = cb = cr = None
    for frame in frames:
        cancel(should_cancel)
        luma, frame_cb, frame_cr = _ycbcr(working_frame(frame, normalize_stack))
        distance = np.abs(luma - target)
        if best is None:
            best, cb, cr = distance, frame_cb, frame_cr
            continue
        better = (distance < best) | (np.isnan(distance) & ~np.isnan(best))
        np.copyto(best, distance, where=better)
        np.copyto(cb, frame_cb, where=better)
        np.copyto(cr, frame_cr, where=better)
    cancel(should_cancel)
    return np.clip(_ycbcr_to_bgr(target, cb, cr), 0.0, 255.0)
