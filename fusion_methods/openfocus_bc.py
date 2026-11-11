"""Bounded OpenFocus B depth fusion and C weighted-pyramid fusion."""
import numpy as np

from fusion_methods.openfocus_depth import depth_map, _validate_feather
from core.frame_storage import create_transient_frame as create_frame
from fusion_methods._openfocus_execution import (
    execution_job, configure_rows, run_bands,
)


def _inputs(input_source, img_resize, should_cancel):
    from fusion_methods.openfocus_ab import _collect_images
    frames = _collect_images(input_source, normalize=False, should_cancel=should_cancel)
    if len(frames) > 65536:
        raise ValueError('OpenFocus depth requires between 2 and 65536 source frames')
    shape = frames[0].shape
    if img_resize is not None:
        shape = (int(img_resize[1]), int(img_resize[0])) + shape[2:]
    return frames, shape



def _depth(frames, shape, img_resize, kernel_size, smoothing, should_cancel,
           depth_override, thread_count, progress_callback=None, *, return_codes=False,
           half_resolution=False, preserve_detail=False, feather=0, gpu=None):
    if depth_override is None:
        return depth_map(frames, img_resize, kernel_size, smoothing, should_cancel,
                         progress_callback=progress_callback, thread_count=thread_count,
                         return_codes=return_codes, half_resolution=half_resolution,
                         preserve_detail=preserve_detail, feather=feather, _gpu=gpu)
    depth = np.asarray(depth_override, dtype=np.float32)
    if depth.shape != shape[:2]:
        raise ValueError("depth_override must match the source height and width")
    if return_codes:
        step = 65535 // max(1, len(frames) - 1)
        return np.rint(depth.astype(np.float64) * step).astype(np.int64)
    return depth


def _source_codes(source, img_resize, should_cancel):
    """Borrow native samples; allocate code planes only at the legacy boundary."""
    from core.source_frames import resolve_frame
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._openfocus_pixels import frame_codes
    _check_cancel(should_cancel)
    frame = resolve_frame(source)
    _check_cancel(should_cancel)
    if img_resize is None and frame.dtype in (np.uint8, np.uint16):
        return frame, 256 if frame.dtype == np.uint8 else 1
    return frame_codes(frame, img_resize, should_cancel), 1


def _merge_codes(accumulator, current, depth, index, count, should_cancel,
                 *, source_scale=1, reverse=False, gpu=None):
    """Ordered MergeWeighted bands, skipping inactive samples before color reads."""
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._openfocus_kernels import merge_rows
    step = 65535 // max(1, count - 1)
    first = index == (count - 1 if reverse else 0)
    if gpu is not None:
        from fusion_methods._openfocus_gpu import GpuError, log_fallback
        for start in range(0, current.shape[0], gpu.rows):
            _check_cancel(should_cancel)
            stop = min(start + gpu.rows, current.shape[0])
            if gpu is not None:
                gpu.acquire(should_cancel)
                try:
                    try:
                        gpu.band(accumulator[start:stop], current[start:stop],
                                 depth[start:stop], index, step, source_scale, first)
                        gpu.used = True
                        continue
                    except GpuError as exc:
                        gpu.failed = True
                        log_fallback(exc)
                finally:
                    gpu.release()
                gpu = None
            merge_rows(accumulator, current, depth, index, step, start, stop,
                       source_scale, first, reverse)
        return gpu
    def merge(start, stop):
        _check_cancel(should_cancel)
        merge_rows(accumulator, current, depth, index, step, start, stop,
                   source_scale, first, reverse)
    run_bands(current.shape[0], 128, merge)
    return None



def _output_codes(codes, frames, should_cancel, *, source_scale=1):
    from fusion_methods.openfocus_ab import _check_cancel
    is_8bit = all(frame.dtype == np.uint8 for frame in frames)
    output = create_frame(codes.shape, np.uint8 if is_8bit else np.float32)
    def convert(start, stop):
        _check_cancel(should_cancel)
        rows = slice(start, stop)
        if is_8bit:
            output[rows] = codes[rows] if source_scale == 256 else codes[rows] >> 8
        else:
            block = codes[rows].astype(np.float32)
            if source_scale != 1:
                block *= np.float32(source_scale)
            output[rows] = block * np.float32(255.0 / 65535.0)
    run_bands(codes.shape[0], 128, convert,
              scratch_bytes=min(128, codes.shape[0]) * int(np.prod(codes.shape[1:])) * 8)
    return output




def _active_sources(depth, count, should_cancel):
    """Reject whole sources before decoding when no code band uses them."""
    from fusion_methods.openfocus_ab import _check_cancel
    active = np.zeros(count, dtype=bool)
    active[0] = True  # The first merge initializes every output sample.
    step = 65535 // (count - 1)
    for start in range(0, depth.shape[0], 128):
        _check_cancel(should_cancel)
        codes = depth[start:start + 128].astype(np.int64)
        lower = np.clip(codes // step, 0, count - 1)
        active |= np.bincount(lower.ravel(), minlength=count).astype(bool)
        fraction = codes % step != 0
        upper = np.minimum(lower[fraction] + 1, count - 1)
        active |= np.bincount(upper, minlength=count).astype(bool)
    return active


@execution_job
def openfocus_b_impl(input_source, img_resize=None, kernel_size: int = 17,
                     thread_count: int = None, depth_smoothing: float = 1.44,
                     should_cancel=None, depth_override=None, progress_callback=None,
                     *, feather: int = 0, use_gpu: bool = False,
                     device_callback=None):
    """Fuse source codes, optionally accelerating ordered merge on an OpenCL GPU."""
    _validate_feather(feather)
    from core import source_frames
    from fusion_methods.openfocus_ab import _check_cancel
    frames, shape = _inputs(input_source, img_resize, should_cancel)
    configure_rows(frames, img_resize, 'openfocus_b', thread_count, should_cancel)
    smoothing = 1.44 if depth_smoothing is None else float(depth_smoothing)
    count = len(frames)
    activity_steps = count if depth_override is None else 0
    total = activity_steps + count + 1
    if progress_callback is not None:
        progress_callback(0, total)
    _check_cancel(should_cancel)
    gpu = None
    if use_gpu:
        from fusion_methods._openfocus_gpu import Merge, GpuError, log_fallback
        try:
            gpu = Merge(shape[1], 1 if len(shape) == 2 else shape[2])
        except GpuError as exc:
            log_fallback(exc)
    gpu_state = gpu
    def depth_progress(index, _total):
        if progress_callback is not None:
            progress_callback(index + 1, total)
    depth = _depth(frames, shape, img_resize, kernel_size,
                   smoothing * (5.0 / 1.2), should_cancel, depth_override,
                   thread_count, depth_progress, return_codes=True,
                   half_resolution=True, preserve_detail=True, feather=feather,
                   gpu=gpu)
    active = _active_sources(depth, count, should_cancel)
    if gpu is not None and gpu.failed:
        gpu = None
    result = create_frame(shape, np.uint16)
    merge_progress = activity_steps
    def merge_sources(resident):
        nonlocal gpu, merge_progress
        for index, source in enumerate(frames):
            _check_cancel(should_cancel)
            if img_resize is None and source.shape != shape:
                raise ValueError('OpenFocus B requires frames with matching shapes')
            if not active[index]:
                if isinstance(source, source_frames.ProcessedFrame) or (
                        source.ndim == 3 and source.shape[2] == 4):
                    source_frames.resolve_frame(source)
                elif isinstance(source, source_frames.SourceFrame):
                    source.validate_source()
            else:
                frame, source_scale = _source_codes(source, img_resize, should_cancel)
                if frame.shape != shape:
                    raise ValueError('OpenFocus B requires frames with matching shapes')
                if resident:
                    try:
                        gpu.resident_frame(frame, index, count, source_scale)
                    except GpuError as exc:
                        return exc
                else:
                    gpu = _merge_codes(result, frame, depth, index, count, should_cancel,
                                       source_scale=source_scale, gpu=gpu)
                del frame
            current = activity_steps + index + 1
            if progress_callback is not None and current > merge_progress:
                progress_callback(current, total)
                merge_progress = current
        return None

    resident = False
    if gpu is not None:
        try:
            resident = gpu.begin_merge(depth, shape, should_cancel)
        except GpuError as exc:
            gpu.failed = True
            log_fallback(exc)
            gpu = None
    if resident:
        try:
            failure = merge_sources(True)
            if failure is None:
                try:
                    gpu.finish_merge(result)
                except GpuError as exc:
                    failure = exc
        finally:
            gpu.end_merge()
        if failure is not None:
            gpu.failed = True
            log_fallback(failure)
            gpu = None
            merge_sources(False)
    else:
        merge_sources(False)
    del depth
    source_frames.SOURCE_CACHE.clear()
    _check_cancel(should_cancel)
    output = _output_codes(result, frames, should_cancel)
    if progress_callback is not None:
        progress_callback(total, total)
    _check_cancel(should_cancel)
    if device_callback is not None:
        from fusion_methods._openfocus_gpu import device_status
        device_callback(device_status(gpu_state))
    return output


@execution_job
def openfocus_c_impl(input_source, img_resize=None, thread_count: int = None,
                     blend_smoothing: float = 2.4, should_cancel=None,
                     progress_callback=None):
    """Fuse source frames using a streaming weighted detail pyramid."""
    from core import source_frames
    from fusion_methods._openfocus_pyramid import fuse_pyramid

    frames, shape = _inputs(input_source, img_resize, should_cancel)
    configure_rows(frames, img_resize, 'openfocus_c', thread_count, should_cancel)
    smoothing = 2.4 if blend_smoothing is None else float(blend_smoothing)
    try:
        return fuse_pyramid(frames, shape, img_resize, smoothing * (5.0 / 3.0),
                            should_cancel, progress_callback)
    finally:
        source_frames.SOURCE_CACHE.clear()
