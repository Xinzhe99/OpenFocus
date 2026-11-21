"""Shared staging, cancellation and resource planning for neural fusion."""
import os
from contextlib import contextmanager

import cv2
import numpy as np

from utils.sorting import natural_sort_key
ATTENTION_PAD_ALIGN = 64



def cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()




@contextmanager
def neural_lease(execution_plan, should_cancel):
    if execution_plan is not None:
        yield
        return
    from core.resource_policy import heavy_job
    from fusion_methods.neural_runtime import inference_job
    with heavy_job(should_cancel), inference_job():
        yield


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


def copy_tile_colors(pixels, normalize_stack, gain=1.0):
    """Pack one color tile in its backend's working domain."""
    from utils.image_utils import apply_exposure_gain

    crop = np.array(pixels, dtype=np.float32 if normalize_stack else pixels.dtype,
                    copy=True, order='C')
    if pixels.dtype == np.uint16:
        crop *= 255.0 / 65535.0
    if gain == 1.0:
        return crop
    if crop.dtype == np.float32:
        # The pack copy is owned by this tile.
        if crop.ndim == 3 and crop.shape[2] == 4:
            color = crop[..., :3]
        else:
            color = crop
        color *= float(gain)
        np.clip(color, 0.0, 255.0, out=color)
        return crop
    return apply_exposure_gain(crop, gain)


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


def write_luma_plane(frame, destination):
    """Write BT.601 luma at 0–1 into one float32 model input plane."""
    if frame.ndim == 2:
        gray = frame
    elif frame.shape[2] == 1:
        gray = frame[:, :, 0]
    else:
        if frame.dtype == np.float32:
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=destination)
            gray = destination
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = np.float32(65535 if gray.dtype == np.uint16 else 255)
    np.divide(gray, scale, out=destination, dtype=np.float32)


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
        write_luma_plane(frame, batch[0, index])
    cancel(should_cancel)
    return batch


def normalized_frames(frames, should_cancel=None):
    """Describe normalization without retaining processed full colour sources."""
    from core.source_frames import normalize_frames
    cancel(should_cancel)
    return normalize_frames(frames, should_cancel)


def attention_inference_bytes(frames, height, width, channels, device, result_count=1):
    """Host working set and new VRAM requirement for a whole inference."""
    from fusion_methods.neural_runtime import (UNIFIED_MEMORY_KINDS, VRAM_GATED_KINDS,
                                               estimate_inference_bytes, native_startup_bytes)
    hp = height + (-height) % ATTENTION_PAD_ALIGN
    wp = width + (-width) % ATTENTION_PAD_ALIGN
    inference = estimate_inference_bytes(frames, height, width, device)
    host = (frames * hp * wp * 4 * (2 * channels + 3)
            + result_count * height * width * channels * 4 + inference)
    if device in UNIFIED_MEMORY_KINDS:
        host += native_startup_bytes(device)
    vram = inference + frames * hp * wp * 4 * 6 if device in VRAM_GATED_KINDS else 0
    return host, vram


def plan_attention_inference(algorithm, frames, model_path, device,
                             thread_count=None,
                             resource_preference='balanced', execution_plan=None,
                             result_count=1, fixed_bytes=0, height=None, width=None):
    """Admit the compact runtime's working set before creating a session."""
    from dataclasses import replace
    from core.resource_policy import ResourceLimitError, Workload, plan_execution, probe_resources
    from core.source_frames import frame_resolution_bytes, source_cache_growth_bytes
    from fusion_methods.neural_runtime import apply_inference_budget, candidate_devices

    preferred = device
    if execution_plan is not None:
        if device != 'CPU':
            device = getattr(execution_plan, 'device_kind', device)
        thread_count = min(thread_count or execution_plan.native_threads,
                           execution_plan.native_threads)
    h, w = frames[0].shape[:2]
    h = int(height) if height is not None else h
    w = int(width) if width is not None else w
    n = len(frames)
    c = frames[0].shape[2] if frames[0].ndim == 3 else 1
    fixed_bytes += max(frame_resolution_bytes(frame) for frame in frames) + source_cache_growth_bytes(frames)
    itemsize = max(frame.dtype.itemsize for frame in frames)
    snapshot = probe_resources()
    requested = device
    for device in candidate_devices(requested):
        host, _ = attention_inference_bytes(n, h, w, c, device, result_count)
        workload = Workload(algorithm=algorithm, frames=n, height=h, width=w,
                            channels=c, itemsize=itemsize, parallel_layer='native',
                            fixed_bytes=host + fixed_bytes, per_worker_bytes=0)
        try:
            plan = plan_execution(workload, snapshot, preference=resource_preference,
                                  user_threads=thread_count,
                                  ram_budget_override=execution_plan.ram_budget if execution_plan else None)
        except ResourceLimitError:
            if device == 'CPU':
                raise
            continue
        fallback = getattr(execution_plan, 'device_fallback', None)
        if preferred != 'CPU' and requested == 'CPU' and not fallback:
            fallback = 'execution plan selected CPU'
        if device != requested:
            fallback = f'{requested} did not fit the host memory budget'
        plan = replace(plan, device_kind=device, device_fallback=fallback,
                       vram_budget=getattr(execution_plan, 'vram_budget', 0))
        apply_inference_budget(plan)
        return model_path, device, plan



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
    best = np.empty(target.shape, dtype=np.float32)
    cb = np.empty_like(best)
    cr = np.empty_like(best)
    luma = np.empty_like(best)
    work = np.empty_like(best)
    better = np.empty(best.shape, dtype=bool)
    for index, frame in enumerate(frames):
        cancel(should_cancel)
        source = working_frame(frame, normalize_stack).astype(np.float32, copy=False)
        if source.ndim == 3 and source.shape[2] == 1:
            source = source[:, :, 0]
        blue, green, red = ((source,) * 3 if source.ndim == 2 else
                            (source[:, :, 0], source[:, :, 1], source[:, :, 2]))
        # Keep the reference's float32 arithmetic order.
        np.multiply(red, np.float32(0.299), out=luma)
        np.multiply(green, np.float32(0.587), out=work)
        np.add(luma, work, out=luma)
        np.multiply(blue, np.float32(0.114), out=work)
        np.add(luma, work, out=luma)
        np.subtract(luma, target, out=work)
        np.abs(work, out=work)
        if index == 0:
            better.fill(True)
        else:
            np.less(work, best, out=better)
            better |= np.isnan(work) & ~np.isnan(best)
        np.copyto(best, work, where=better)
        np.subtract(blue, luma, out=cb, where=better)
        np.subtract(red, luma, out=cr, where=better)
        del source, blue, green, red
    cancel(should_cancel)
    cb *= np.float32(0.564)
    cb += np.float32(128)
    cr *= np.float32(0.713)
    cr += np.float32(128)
    result = _ycbcr_to_bgr(target, cb, cr)
    return np.clip(result, 0.0, 255.0, out=result)
