"""Shared staging, cancellation and resource planning for neural fusion."""
import os
from contextlib import contextmanager
from functools import partial
from itertools import repeat

import cv2
import numpy as np

from core.source_frames import ProcessedFrame
from utils.performance_profile import current as _profile
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


def luma_plane(frame):
    """Return one 0-1 float32 luma plane from a grayscale or BGR frame."""
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
        batch[0, index] = luma_plane(frame)
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


class _TileCrop(ProcessedFrame):
    """A packed crop from a source already admitted by the tile planner."""

    def full_resolution(self, *, should_cancel=None, cache_result=True, store_result=True):
        cancel(self.should_cancel)
        cancel(should_cancel)
        with _profile().stage('resolve_region'):
            source = self.source
            if isinstance(source, ProcessedFrame):
                source = source.full_resolution(should_cancel=should_cancel or self.should_cancel,
                                                cache_result=False, store_result=False)
            image = self.processor(source)
        cancel(self.should_cancel)
        cancel(should_cancel)
        return image


def _packed_crop(image, *, rows, columns, normalize, gain):
    # The packed grid preserves OpenCV's SIMD/scalar luma rounding at tile edges.
    image = np.ascontiguousarray(image[rows, columns])
    if gain is None:
        return image
    from utils.image_utils import apply_exposure_gain
    return apply_exposure_gain(working_frame(image, normalize), gain)


def tile_sources(frames, coords, *, exposure_gains=None, should_cancel=None):
    """Describe one packed crop per frame, resolving only the crop being used."""
    x0, y0, x1, y1 = coords
    normalize = exposure_gains is not None and any(frame.dtype == np.uint16 for frame in frames)
    gains = repeat(None) if exposure_gains is None else exposure_gains
    sources = []
    for frame, gain in zip(frames, gains):
        cancel(should_cancel)
        if frame.shape[0] < y1 or frame.shape[1] < x1:
            raise ValueError('All frames must have the same dimensions')
        shape = (y1 - y0, x1 - x0) + frame.shape[2:]
        dtype = np.dtype(np.float32) if normalize else frame.dtype
        processor = partial(_packed_crop, rows=slice(y0, y1), columns=slice(x0, x1),
                            normalize=normalize, gain=gain)
        sources.append(_TileCrop(frame, processor, shape, dtype, should_cancel))
    return sources


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
    from core.source_frames import resolve_frame
    height, width = target.shape
    band_rows = max(1, min(128, (64 * 1024) // width))
    best = np.empty((height, width), dtype=np.result_type(target.dtype, np.float32))
    cb = np.empty((height, width), dtype=np.float32)
    cr = np.empty_like(cb)
    for index, frame in enumerate(frames):
        cancel(should_cancel)
        image = resolve_frame(frame, should_cancel=should_cancel)
        for start in range(0, height, band_rows):
            cancel(should_cancel)
            rows = slice(start, start + band_rows)
            luma, frame_cb, frame_cr = _ycbcr(working_frame(image[rows], normalize_stack))
            distance = np.abs(luma - target[rows])
            if index == 0:
                best[rows], cb[rows], cr[rows] = distance, frame_cb, frame_cr
            else:
                previous = best[rows]
                better = (distance < previous) | (np.isnan(distance) & ~np.isnan(previous))
                np.copyto(previous, distance, where=better)
                np.copyto(cb[rows], frame_cb, where=better)
                np.copyto(cr[rows], frame_cr, where=better)
        del image, luma, frame_cb, frame_cr, distance
    fused = np.empty((height, width, 3), dtype=np.float32)
    for start in range(0, height, band_rows):
        cancel(should_cancel)
        rows = slice(start, start + band_rows)
        color = _ycbcr_to_bgr(target[rows], cb[rows], cr[rows])
        np.clip(color, 0.0, 255.0, out=fused[rows])
        del color
    cancel(should_cancel)
    return fused
