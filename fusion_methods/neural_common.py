"""Shared staging, cancellation and resource planning for neural fusion."""
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import cv2
import numpy as np

from core.source_frames import ProcessedFrame, borrow_frame_region
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


def is_bgr_rev_contiguous(image):
    """True when image is a contiguous C-order RGB buffer viewed via ::-1 as BGR."""
    return (isinstance(image, np.ndarray)
            and image.ndim == 3 and image.shape[2] == 3
            and image.strides[2] == -image.dtype.itemsize
            and image[:, :, ::-1].flags.c_contiguous)


def is_bgr_rev_strided(image):
    """True when image views an RGB buffer via ::-1 with positive row/col strides."""
    itemsize = getattr(getattr(image, 'dtype', None), 'itemsize', 0)
    return (isinstance(image, np.ndarray)
            and image.ndim == 3 and image.shape[2] == 3
            and itemsize > 0
            and image.strides[0] > 0
            and image.strides[1] == 3 * itemsize
            and image.strides[2] == -itemsize)


def copy_tile_colors(pixels, normalize_stack, gain=1.0):
    """Pack one color tile in its backend's working domain."""
    from utils.image_utils import apply_exposure_gain

    if is_bgr_rev_strided(pixels) and (pixels.dtype == np.uint16 or pixels.dtype == np.uint8):
        kernels = load_stackmff_kernels()
        if kernels is not None:
            rgb = pixels[:, :, ::-1]
            h, w = pixels.shape[:2]
            try:
                if normalize_stack:
                    dst = np.empty((h, w, 3), dtype=np.float32)
                    if pixels.dtype == np.uint16:
                        kernels.pack_rgb_u16_to_bgr_f32(rgb, 0, 0, h, w, dst, float(gain))
                    else:
                        kernels.pack_rgb_u8_to_bgr_f32(rgb, 0, 0, h, w, dst, float(gain))
                    return dst
                if pixels.dtype == np.uint8 and float(gain) == 1.0:
                    dst = np.empty((h, w, 3), dtype=np.uint8)
                    kernels.pack_rgb_u8_to_bgr_u8(rgb, 0, 0, h, w, dst)
                    return dst
            except StackMFFKernelUnavailable:
                pass

    crop = np.array(pixels, dtype=np.float32 if normalize_stack else pixels.dtype,
                    copy=True, order='C')
    if pixels.dtype == np.uint16:
        crop *= 255.0 / 65535.0
    return apply_exposure_gain(crop, gain, copy=False)


@dataclass(frozen=True, eq=False, kw_only=True)
class _TileCrop(ProcessedFrame):
    """Pack a single admitted crop on demand without caching its pixels."""

    region: tuple

    def full_resolution(self, *, should_cancel=None, cache_result=True, store_result=True):
        def cancelled():
            return ((should_cancel is not None and should_cancel()) or
                    (self.should_cancel is not None and self.should_cancel()))

        cancel(cancelled)
        with _profile().stage('resolve_region'):
            with borrow_frame_region(self.source, self.region, should_cancel=cancelled) as pixels:
                if pixels.shape != self.shape:
                    raise ValueError('All frames must have the same dimensions')
                crop = self.processor(pixels)
        cancel(cancelled)
        return crop


def tile_sources(frames, coords, *, exposure_gains=None, should_cancel=None):
    """Describe packed crops in the same working domain as eager tile staging."""
    x0, y0, x1, y1 = coords
    region = np.s_[y0:y1, x0:x1]
    normalize_stack = any(frame.dtype == np.uint16 for frame in frames)
    sources = []
    for index, frame in enumerate(frames):
        cancel(should_cancel)
        if frame.shape[0] < y1 or frame.shape[1] < x1:
            raise ValueError('All frames must have the same dimensions')
        gain = exposure_gains[index] if exposure_gains is not None else 1.0
        shape = (y1 - y0, x1 - x0) + frame.shape[2:]
        converted = normalize_stack or (gain != 1.0 and frame.dtype != np.uint8)
        dtype = np.dtype(np.float32) if converted else frame.dtype
        pack = partial(copy_tile_colors, normalize_stack=normalize_stack, gain=gain)
        sources.append(_TileCrop(frame, pack, shape, dtype, should_cancel, region=region))
    return sources


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


def _pad_plane_2d(dst_plane, h, w, ph, pw, pad_mode):
    """Pad dst_plane[0:h, 0:w] in-place out to (h + ph, w + pw)."""
    if pw:
        if pad_mode == 'reflect' and pw < w:
            dst_plane[:h, w:] = dst_plane[:h, w - 2::-1][:, :pw]
        else:
            dst_plane[:h, w:] = dst_plane[:h, w - 1:w]
    if ph:
        if pad_mode == 'reflect' and ph < h:
            dst_plane[h:, :] = dst_plane[h - 2::-1, :][:ph, :]
        else:
            dst_plane[h:, :] = dst_plane[h - 1:h, :]


def _pad_band_width_3ch(scratch_band, h_band, w, pw, pad_mode):
    if not pw:
        return
    if pad_mode == 'reflect' and pw < w:
        scratch_band[:h_band, w:] = scratch_band[:h_band, w - 2::-1][:, :pw]
    else:
        scratch_band[:h_band, w:] = scratch_band[:h_band, w - 1:w]


def write_luma_plane_fast(raw, dst_plane, normalize_stack, *, ph=0, pw=0,
                          pad_mode='edge', threads=1, should_cancel=None,
                          kernels=None, scratch=None):
    """Write BT.601 normalized [0-1] luma + padding bit-exact with reference."""
    h, w = raw.shape[:2]
    wp = w + pw
    workers = max(1, min(8, int(threads or 1)))
    inv255 = np.float32(255.0)
    use_band_threads = workers > 1 and h * w >= 262144 and h >= 2
    # Only the first row range may borrow the caller's scratch. Concurrent
    # ranges must keep their packed pixels until their conversion finishes.

    if raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] == 1):
        gray = raw if raw.ndim == 2 else raw[:, :, 0]
        if normalize_stack and gray.dtype == np.uint16:
            # Mirror working_frame (float32 * (255/65535)) followed by / float32(255).
            band_rows = min(h, max(1, (256 * 1024) // max(1, w * 4)))
            scale16 = np.float32(255.0 / 65535.0)

            def _gray_u16_band(rng):
                buf_arr = (scratch[:band_rows, :w, 0]
                           if (rng[0] == 0 and scratch is not None
                               and scratch.shape[0] >= band_rows and scratch.shape[1] >= w)
                           else np.empty((band_rows, w), dtype=np.float32))
                for y0 in range(rng[0], rng[1], band_rows):
                    cancel(should_cancel)
                    y1 = min(rng[1], y0 + band_rows)
                    bh = y1 - y0
                    buf = buf_arr[:bh]
                    np.multiply(gray[y0:y1], scale16, out=buf, dtype=np.float32)
                    np.divide(buf, inv255, out=dst_plane[y0:y1, :w], dtype=np.float32)

            if use_band_threads:
                from fusion_methods.neural_runtime import run_neural_tasks
                parts = min(workers, h)
                step = (h + parts - 1) // parts
                run_neural_tasks(parts, [(y0, min(h, y0 + step)) for y0 in range(0, h, step)],
                                 _gray_u16_band)
            else:
                _gray_u16_band((0, h))
        else:
            scale = np.float32(65535 if gray.dtype == np.uint16 else 255)
            np.divide(gray, scale, out=dst_plane[:h, :w], dtype=np.float32)
        _pad_plane_2d(dst_plane, h, w, ph, pw, pad_mode)
        return

    if raw.ndim == 3 and raw.shape[2] == 3:
        is_c = raw.flags.c_contiguous
        is_rev = False if is_c else is_bgr_rev_contiguous(raw)
        if raw.dtype == np.float32 and is_c and not pw and not use_band_threads:
            cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY, dst=dst_plane[:h, :w])
            np.divide(dst_plane[:h, :w], inv255, out=dst_plane[:h, :w], dtype=np.float32)
            if ph:
                _pad_plane_2d(dst_plane, h, wp, ph, 0, pad_mode)
            return

        if raw.dtype == np.uint8 and not normalize_stack and (is_c or is_rev):
            if kernels is None:
                kernels = load_stackmff_kernels()
            if kernels is not None:
                kfn = kernels.bgr_u8_to_luma_rows if is_c else kernels.rgb_u8_to_luma_rows
                src = raw if is_c else raw[:, :, ::-1]
                try:
                    if use_band_threads:
                        from fusion_methods.neural_runtime import run_neural_tasks
                        parts = min(workers, h)
                        step = (h + parts - 1) // parts
                        run_neural_tasks(parts, [(y0, min(h, y0 + step)) for y0 in range(0, h, step)],
                                         lambda rng: kfn(src, dst_plane, rng[0], rng[1]))
                    else:
                        kfn(src, dst_plane, 0, h)
                except StackMFFKernelUnavailable:
                    pass
                else:
                    _pad_plane_2d(dst_plane, h, w, ph, pw, pad_mode)
                    return

        if raw.dtype == np.float32 and is_c:
            band_rows = min(h, max(1, (256 * 1024) // max(1, wp * 12))) if pw else min(h, max(64, (256 * 1024) // max(1, w * 4)))

            def _f32_band(rng):
                s_arr = (scratch[:band_rows, :wp]
                         if (rng[0] == 0 and pw and scratch is not None
                             and scratch.shape[0] >= band_rows and scratch.shape[1] == wp)
                         else (np.empty((band_rows, wp, 3), dtype=np.float32) if pw else None))
                for y0 in range(rng[0], rng[1], band_rows):
                    cancel(should_cancel)
                    y1 = min(rng[1], y0 + band_rows)
                    bh = y1 - y0
                    if pw:
                        s_view = s_arr[:bh]
                        s_view[:, :w] = raw[y0:y1]
                        _pad_band_width_3ch(s_view, bh, w, pw, pad_mode)
                        cv2.cvtColor(s_view, cv2.COLOR_BGR2GRAY, dst=dst_plane[y0:y1, :wp])
                        np.divide(dst_plane[y0:y1, :wp], inv255, out=dst_plane[y0:y1, :wp], dtype=np.float32)
                    else:
                        cv2.cvtColor(raw[y0:y1], cv2.COLOR_BGR2GRAY, dst=dst_plane[y0:y1, :w])
                        np.divide(dst_plane[y0:y1, :w], inv255, out=dst_plane[y0:y1, :w], dtype=np.float32)

            if use_band_threads:
                from fusion_methods.neural_runtime import run_neural_tasks
                parts = min(workers, h)
                step = (h + parts - 1) // parts
                run_neural_tasks(parts, [(y0, min(h, y0 + step)) for y0 in range(0, h, step)],
                                 _f32_band)
            else:
                _f32_band((0, h))
            if ph:
                _pad_plane_2d(dst_plane, h, wp, ph, 0, pad_mode)
            return

        if (raw.dtype == np.uint16 or (raw.dtype == np.uint8 and normalize_stack)) and (is_c or is_rev):
            if kernels is None:
                kernels = load_stackmff_kernels()
            if kernels is not None:
                if is_c:
                    pack_fn = (kernels.pack_bgr_u16_to_bgr_f32_rows if raw.dtype == np.uint16
                               else kernels.pack_bgr_u8_to_bgr_f32_rows)
                    src = raw
                else:
                    pack_fn = (kernels.pack_rgb_u16_to_bgr_f32_rows if raw.dtype == np.uint16
                               else kernels.pack_rgb_u8_to_bgr_f32_rows)
                    src = raw[:, :, ::-1]
                band_rows = min(h, max(1, (256 * 1024) // max(1, wp * 12)))

                def _pack_band(rng):
                    s_arr = (scratch[:band_rows, :wp]
                             if (rng[0] == 0 and scratch is not None
                                 and scratch.shape[0] >= band_rows and scratch.shape[1] == wp)
                             else np.empty((band_rows, wp, 3), dtype=np.float32))
                    for y0 in range(rng[0], rng[1], band_rows):
                        cancel(should_cancel)
                        y1 = min(rng[1], y0 + band_rows)
                        bh = y1 - y0
                        pack_fn(src, y0, y1, s_arr)
                        s_view = s_arr[:bh]
                        if pw:
                            _pad_band_width_3ch(s_view, bh, w, pw, pad_mode)
                        cv2.cvtColor(s_view, cv2.COLOR_BGR2GRAY, dst=dst_plane[y0:y1, :wp])
                        np.divide(dst_plane[y0:y1, :wp], inv255, out=dst_plane[y0:y1, :wp], dtype=np.float32)

                try:
                    if use_band_threads:
                        from fusion_methods.neural_runtime import run_neural_tasks
                        parts = min(workers, h)
                        step = (h + parts - 1) // parts
                        run_neural_tasks(parts, [(y0, min(h, y0 + step)) for y0 in range(0, h, step)],
                                         _pack_band)
                    else:
                        _pack_band((0, h))
                except StackMFFKernelUnavailable:
                    pass
                else:
                    if ph:
                        _pad_plane_2d(dst_plane, h, wp, ph, 0, pad_mode)
                    return

    # Fallback: preserve exact padding before conversion
    work = working_frame(raw, normalize_stack)
    if ph or pw:
        padding = ((0, ph), (0, pw)) + ((0, 0),) * (work.ndim - 2)
        work = np.pad(work, padding, mode=pad_mode)
    write_luma_plane(work, dst_plane)


def stage_planes(frames, align, should_cancel=None, *, threads=1):
    """Stage a padded image grid as float32 luma at 0-1.

    Pad before native colour conversion so whole and tiled inference use the
    same conversion grid, including scalar/SIMD rounding at image edges.
    """
    from core.source_frames import resolve_frame
    from fusion_methods.neural_runtime import run_neural_tasks

    h, w = frames[0].shape[:2]
    hp, wp = h + (-h) % align, w + (-w) % align
    ph, pw = hp - h, wp - w
    mode = 'reflect' if ph < h and pw < w else 'edge'
    batch = np.empty((1, len(frames), hp, wp), dtype=np.float32)
    normalize_stack = any(frame.dtype == np.uint16 for frame in frames)
    workers = max(1, min(8, int(threads or 1)))
    kernels = load_stackmff_kernels()
    need_scratch = bool(pw) or (kernels is not None and any(
        f.dtype == np.uint16 or (f.dtype == np.uint8 and normalize_stack) for f in frames
    ))
    band_rows = min(h, max(1, (256 * 1024) // max(1, wp * 12)))
    total_px = len(frames) * h * w
    layer_workers = min(workers, max(1, total_px // 4194304)) if h * w >= 32768 else 1
    if (layer_workers > 1 and len(frames) >= 4
            and all(isinstance(frame, np.ndarray) for frame in frames)):
        step = (len(frames) + layer_workers - 1) // layer_workers
        ranges = [(i0, min(len(frames), i0 + step)) for i0 in range(0, len(frames), step)]

        def _stage_range(rng):
            scratch = np.empty((band_rows, wp, 3), dtype=np.float32) if need_scratch else None
            for idx in range(rng[0], rng[1]):
                cancel(should_cancel)
                write_luma_plane_fast(
                    frames[idx], batch[0, idx], normalize_stack,
                    ph=ph, pw=pw, pad_mode=mode, threads=1, should_cancel=should_cancel,
                    kernels=kernels, scratch=scratch,
                )

        run_neural_tasks(layer_workers, ranges, _stage_range)
    else:
        scratch = np.empty((band_rows, wp, 3), dtype=np.float32) if need_scratch else None
        for index, frame in enumerate(frames):
            cancel(should_cancel)
            raw = resolve_frame(frame, should_cancel=should_cancel)
            write_luma_plane_fast(
                raw, batch[0, index], normalize_stack,
                ph=ph, pw=pw, pad_mode=mode, threads=workers, should_cancel=should_cancel,
                kernels=kernels, scratch=scratch,
            )
            del raw
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
        required_ram = host + fixed_bytes
        workload = Workload(algorithm=algorithm, frames=n, height=h, width=w,
                            channels=c, itemsize=itemsize, parallel_layer='native',
                            fixed_bytes=required_ram, per_worker_bytes=0)
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
                       required_ram=required_ram,
                       vram_budget=getattr(execution_plan, 'vram_budget', 0))
        apply_inference_budget(plan)
        return model_path, device, plan


def plan_rgb_staging(plan, tile_count, tile_bytes, scratch_bytes, byte_cap):
    """Admit extra RGB tile data alongside the full inference working set."""
    limit = min(tile_count, min(byte_cap, plan.ram_budget // 4) // tile_bytes)
    reserved = plan.required_ram + scratch_bytes
    if limit < 2 or reserved + tile_bytes > plan.ram_budget:
        return None, 1
    kernels = load_stackmff_kernels()
    if kernels is None:
        return None, 1
    from core.resource_policy import probe_memory, ram_ceiling

    headroom = min(plan.ram_budget, ram_ceiling(probe_memory()))
    size = min(limit, 1 + (headroom - reserved) // tile_bytes)
    return (kernels, size) if size > 1 else (None, 1)




_STACKMFF_KERNELS = None
_STACKMFF_KERNELS_DISABLED = False


class StackMFFKernelUnavailable(MemoryError):
    """An optional specialization ran out of memory before touching its output."""


def disable_stackmff_kernels():
    global _STACKMFF_KERNELS_DISABLED
    _STACKMFF_KERNELS_DISABLED = True


def reset_stackmff_kernels():
    global _STACKMFF_KERNELS_DISABLED
    _STACKMFF_KERNELS_DISABLED = False


def load_stackmff_kernels():
    """Load the optional native dispatchers for this inference job."""
    global _STACKMFF_KERNELS
    if _STACKMFF_KERNELS_DISABLED:
        return None
    if _STACKMFF_KERNELS is not None:
        return _STACKMFF_KERNELS
    from core.algorithm_adapters import DEPTH_KERNEL_STARTUP_BYTES
    from core.resource_policy import probe_memory, require_memory

    try:
        require_memory(DEPTH_KERNEL_STARTUP_BYTES, probe_memory())
        from fusion_methods import _stackmff_kernels
    except MemoryError:
        disable_stackmff_kernels()
        return None
    except ImportError:
        return None
    _STACKMFF_KERNELS = _stackmff_kernels
    return _STACKMFF_KERNELS


def _ycbcr_to_bgr(luma, cb, cr):
    luma = luma.astype(np.float32, copy=False)
    cb = cb - 128.0
    cr = cr - 128.0
    blue = luma + 1.773 * cb
    green = luma - 0.714 * cr - 0.344 * cb
    red = luma + 1.403 * cr
    return np.stack((blue, green, red), axis=-1)


def _select_v1_nearest_kernel(kernels, raw, target_shape):
    """Return (kernel_rows_fn, src_array) for a single frame, or (None, None)."""
    if (kernels is None or raw.ndim != 3 or raw.shape[2] < 3
            or raw.shape[:2] != target_shape):
        return None, None
    if raw.flags.c_contiguous:
        if raw.dtype == np.float32:
            return kernels.v1_update_nearest_f32_rows, raw
        if raw.dtype == np.uint16:
            return kernels.v1_update_nearest_u16_rows, raw
        if raw.dtype == np.uint8:
            return kernels.v1_update_nearest_u8_rows, raw
    elif is_bgr_rev_contiguous(raw) and raw.shape[2] == 3:
        rgb = raw[:, :, ::-1]
        if raw.dtype == np.uint16:
            return kernels.v1_update_nearest_rgb_u16_rows, rgb
        if raw.dtype == np.uint8:
            return kernels.v1_update_nearest_rgb_u8_rows, rgb
    return None, None


def tensor_to_bgr(fused_luma, frames, should_cancel=None, *, threads=1):
    """Colour reconstruction from the frame whose luma matches the prediction.

    This mirrors the StackMFF reference: the fused luma is kept and Cb/Cr are
    taken from the source frame closest to it, so no new colours are invented.
    The conversion is written out because cv2's float32 YCrCb path expects
    0-1 inputs, while every buffer here is in the project's 0-255 range.
    """
    from core.source_frames import resolve_frame
    from fusion_methods.neural_runtime import run_neural_tasks

    cancel(should_cancel)
    normalize_stack = any(frame.dtype == np.uint16 for frame in frames)
    target = np.clip(fused_luma, 0.0, 255.0).astype(np.float32, copy=False)
    best = np.empty(target.shape, dtype=np.float32)
    cb = np.empty_like(best)
    cr = np.empty_like(best)
    kernels = load_stackmff_kernels()
    workers = max(1, min(8, int(threads or 1)))
    h, w = target.shape
    luma = work = better = None
    target_c = np.ascontiguousarray(target) if kernels is not None else None

    resident_pairs = None
    row_workers = min(workers, h, max(1, (h * w) // 16384))
    if (row_workers > 1 and kernels is not None and h >= 2 and h * w >= 65536
            and all(isinstance(frame, np.ndarray) for frame in frames)):
        pairs = [_select_v1_nearest_kernel(kernels, frame, (h, w)) for frame in frames]
        if all(pair[0] is not None for pair in pairs):
            resident_pairs = pairs

    if resident_pairs is not None:
        parts = row_workers
        step = (h + parts - 1) // parts
        ranges = [(y0, min(h, y0 + step)) for y0 in range(0, h, step)]

        def _nearest_band(rng):
            y0, y1 = rng
            for idx, (kfn, src) in enumerate(resident_pairs):
                cancel(should_cancel)
                kfn(src, target_c, best, cb, cr, idx == 0, y0, y1)

        try:
            run_neural_tasks(row_workers, ranges, _nearest_band)
        except StackMFFKernelUnavailable:
            kernels = target_c = resident_pairs = None
        else:
            cancel(should_cancel)
            result = np.empty((h, w, 3), dtype=np.float32)
            try:
                run_neural_tasks(
                    row_workers, ranges,
                    lambda rng: kernels.v1_ycbcr_to_bgr_rows(target_c, cb, cr, result, rng[0], rng[1]),
                )
                return result
            except StackMFFKernelUnavailable:
                kernels = target_c = None

    if resident_pairs is None:
        for index, frame in enumerate(frames):
            cancel(should_cancel)
            raw = resolve_frame(frame, should_cancel=should_cancel)
            used_kernel = False
            kfn, src = _select_v1_nearest_kernel(kernels, raw, (h, w))
            if kfn is not None:
                try:
                    if workers > 1 and h * w >= 262144 and h >= 2:
                        parts = min(workers, h)
                        step = (h + parts - 1) // parts
                        ranges = [(y0, min(h, y0 + step)) for y0 in range(0, h, step)]
                        run_neural_tasks(
                            workers, ranges,
                            lambda rng, _fn=kfn, _src=src, _first=(index == 0): _fn(
                                _src, target_c, best, cb, cr, _first, rng[0], rng[1]),
                        )
                    else:
                        kfn(src, target_c, best, cb, cr, index == 0, 0, h)
                except StackMFFKernelUnavailable:
                    kernels = target_c = None
                else:
                    used_kernel = True
            if not used_kernel:
                if luma is None:
                    luma = np.empty_like(best)
                    work = np.empty_like(best)
                    better = np.empty(best.shape, dtype=bool)
                source = working_frame(raw, normalize_stack).astype(np.float32, copy=False)
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
            del kfn, src, raw

    cancel(should_cancel)
    if kernels is not None and target_c is not None:
        result = np.empty((h, w, 3), dtype=np.float32)
        try:
            if workers > 1 and h * w >= 65536 and h >= 2:
                parts = min(workers, h)
                step = (h + parts - 1) // parts
                ranges = [(y0, min(h, y0 + step)) for y0 in range(0, h, step)]
                run_neural_tasks(
                    workers, ranges,
                    lambda rng: kernels.v1_ycbcr_to_bgr_rows(target_c, cb, cr, result, rng[0], rng[1]),
                )
            else:
                kernels.v1_ycbcr_to_bgr_rows(target_c, cb, cr, result, 0, h)
            return result
        except StackMFFKernelUnavailable:
            pass
    cb *= np.float32(0.564)
    cb += np.float32(128)
    cr *= np.float32(0.713)
    cr += np.float32(128)
    result = _ycbcr_to_bgr(target, cb, cr)
    return np.clip(result, 0.0, 255.0, out=result)
