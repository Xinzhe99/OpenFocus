"""Bounded OpenFocus B depth fusion and C weighted-pyramid fusion."""
import numpy as np

from fusion_methods.openfocus_depth import depth_map, _validate_feather
from core.frame_storage import create_transient_frame as create_frame
from core.frame_storage import frame_region
from fusion_methods._openfocus_execution import (
    execution_job, configure_rows, run_bands,
)

_MERGE_TILE = 128
_MAX_MERGE_MASK_BYTES = 4 * 1024 * 1024


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
           half_resolution=False, preserve_detail=False, feather=0, use_gpu=True, gpu=None,
           live_preview=None):
    if depth_override is None:
        return depth_map(frames, img_resize, kernel_size, smoothing, should_cancel,
                         progress_callback=progress_callback, thread_count=thread_count,
                         return_codes=return_codes, half_resolution=half_resolution,
                         preserve_detail=preserve_detail, feather=feather, use_gpu=use_gpu,
                         _gpu=gpu, _live_preview=live_preview)
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
    frame = resolve_frame(source, should_cancel=should_cancel)
    _check_cancel(should_cancel)
    if img_resize is None and frame.dtype in (np.uint8, np.uint16):
        return frame, 256 if frame.dtype == np.uint8 else 1
    return frame_codes(frame, img_resize, should_cancel), 1


def _merge_codes(accumulator, current, depth, index, count, should_cancel,
                 *, source_scale=1, gpu=None, tiles=None):
    """Ordered MergeWeighted bands, skipping inactive samples before color reads."""
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._openfocus_kernels import merge_rows
    step = 65535 // max(1, count - 1)
    first = index == 0
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
                       source_scale, first, False)
        return gpu
    if tiles is not None and not first:
        from fusion_methods._openfocus_depth_rows import merge_active_rows

        def merge_tiles(start, stop):
            _check_cancel(should_cancel)
            merge_active_rows(accumulator[start:stop], current[start:stop], depth[start:stop],
                              index, step, source_scale, tiles[start // _MERGE_TILE], _MERGE_TILE)

        run_bands(current.shape[0], _MERGE_TILE, merge_tiles)
        return None
    def merge(start, stop):
        _check_cancel(should_cancel)
        merge_rows(accumulator, current, depth, index, step, start, stop,
                   source_scale, first, False)
    run_bands(current.shape[0], 128, merge)
    return None



def _output_codes(codes, frames, should_cancel, *, source_scale=1):
    from fusion_methods.openfocus_ab import _check_cancel
    is_8bit = all(frame.dtype == np.uint8 for frame in frames)
    output = create_frame(codes.shape, np.uint8 if is_8bit else np.float32)
    if (not is_8bit) and codes.dtype == np.uint16 and source_scale == 1:
        from fusion_methods._openfocus_kernels import output_u16_to_f32_rows
        factor = np.float32(255.0 / 65535.0)

        def convert_fast(start, stop):
            _check_cancel(should_cancel)
            output_u16_to_f32_rows(codes, output, start, stop, factor)

        run_bands(codes.shape[0], 128, convert_fast)
        return output

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




def _active_sources(depth, count, should_cancel, *, tiles=None):
    """Reject whole sources before decoding when no code band uses them."""
    import threading
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._openfocus_kernels import active_source_rows
    active = np.zeros(count, dtype=bool)
    active[0] = True  # The first merge initializes every output sample.
    step = 65535 // (count - 1)
    if tiles is None:
        lock = threading.Lock()

        def mark_bands(start, stop):
            _check_cancel(should_cancel)
            local = np.zeros(count, dtype=bool)
            active_source_rows(depth, step, local, start, stop)
            with lock:
                np.logical_or(active, local, out=active)

        run_bands(depth.shape[0], _MERGE_TILE, mark_bands)
    else:
        def mark(start, stop):
            _check_cancel(should_cancel)
            for col, left in enumerate(range(0, depth.shape[1], _MERGE_TILE)):
                tile = depth[start:stop, left:left + _MERGE_TILE]
                active_source_rows(tile, step, tiles[start // _MERGE_TILE, col], 0, stop - start)

        run_bands(depth.shape[0], _MERGE_TILE, mark)
        active |= tiles.any(axis=(0, 1))
    return active


@execution_job
def openfocus_b_impl(input_source, img_resize=None, kernel_size: int = 17,
                     thread_count: int = None, depth_smoothing: float = 1.44,
                     should_cancel=None, depth_override=None, progress_callback=None,
                     *, feather: int = 0, use_gpu: bool = True,
                     device_callback=None, focus_callback=None, live_preview=None):
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
                   use_gpu=use_gpu, gpu=gpu, live_preview=live_preview)
    if gpu is not None and gpu.failed:
        gpu = None
    is_8bit = all(frame.dtype == np.uint8 for frame in frames)
    active = tiles = None
    result = output = None
    merge_progress = activity_steps
    def merge_sources(resident_flag):
        nonlocal gpu, merge_progress
        from math import prod
        from core import resource_policy
        from core.multi_focus_fusion import memory_budget_bytes
        from fusion_methods._openfocus_gpu import _iter_prefetched, _prefetch_window

        host_budget = min(memory_budget_bytes(),
                          resource_policy.ram_ceiling(resource_policy.probe_memory()))
        samples = prod(shape)
        per_slot = max(
            source_frames.frame_resolution_bytes(frame, metadata_only=True)
            + (2 * frame.nbytes if img_resize is None and frame.dtype in (np.uint8, np.uint16)
               else 8 * (frame.size + samples))
            for frame in frames
        )

        def prepare_merge_source(index, source, cancelled):
            _check_cancel(cancelled)
            if img_resize is None and source.shape != shape:
                raise ValueError('OpenFocus B requires frames with matching shapes')
            if not active[index]:
                if depth_override is not None and (
                        isinstance(source, source_frames.ProcessedFrame)
                        or (source.ndim == 3 and source.shape[2] == 4)):
                    source_frames.resolve_frame(source, should_cancel=cancelled)
                else:
                    source_frames.validate_frame_source(source)
                return None
            with source_frames.without_source_cache_growth():
                frame, source_scale = _source_codes(source, img_resize, cancelled)
            if frame.shape != shape:
                raise ValueError('OpenFocus B requires frames with matching shapes')
            return frame, source_scale

        if resident_flag:
            # Charge the current decode and any other upload still holding its host buffer.
            fixed_host = per_slot * (2 if getattr(gpu, '_merge_ping_pong', False) else 1)
            fixed_host += getattr(gpu, '_merge_unified_bytes', 0)
            if result is not None:
                fixed_host += result.nbytes
            slots = _prefetch_window(
                host_budget, fixed_host, per_slot, count, thread_count=thread_count,
            )
            prepared_iter = _iter_prefetched(
                frames, prepare_merge_source, enabled=slots > 0,
                should_cancel=should_cancel, max_in_flight=max(1, slots),
            )
            try:
                for index, prepared in prepared_iter:
                    _check_cancel(should_cancel)
                    if prepared is not None:
                        frame, source_scale = prepared
                        try:
                            gpu.resident_frame(frame, index, count, source_scale)
                        except GpuError as exc:
                            return exc
                        finally:
                            del frame, prepared
                    if live_preview is not None:
                        live_preview.show_blend(depth, index, count)
                    current = activity_steps + index + 1
                    if progress_callback is not None and current > merge_progress:
                        progress_callback(current, total)
                        merge_progress = current
            finally:
                prepared_iter.close()
            return None

        fixed_host = per_slot
        if result is not None:
            fixed_host += result.nbytes
        if depth is not None:
            fixed_host += depth.nbytes
        if tiles is not None:
            fixed_host += tiles.nbytes
        slots = _prefetch_window(
            host_budget, fixed_host, per_slot, count,
            max_slots=2, thread_count=thread_count,
        )
        prepared_iter = _iter_prefetched(
            frames, prepare_merge_source, enabled=slots > 0,
            should_cancel=should_cancel, max_in_flight=max(1, slots),
        )
        try:
            for index, prepared in prepared_iter:
                _check_cancel(should_cancel)
                if prepared is not None:
                    frame, source_scale = prepared
                    try:
                        with frame_region(frame) as pixels:
                            gpu = _merge_codes(result, pixels, depth, index, count, should_cancel,
                                               source_scale=source_scale, gpu=gpu,
                                               tiles=None if tiles is None else tiles[:, :, index])
                    finally:
                        del frame, prepared
                if live_preview is not None:
                    live_preview.show_blend(depth, index, count)
                current = activity_steps + index + 1
                if progress_callback is not None and current > merge_progress:
                    progress_callback(current, total)
                    merge_progress = current
        finally:
            prepared_iter.close()
        return None

    resident = False
    failure = None
    if gpu is not None:
        try:
            resident = gpu.begin_merge(depth, shape, should_cancel)
        except GpuError as exc:
            gpu.failed = True
            log_fallback(exc)
            gpu = None
    if resident:
        try:
            active = _active_sources(depth, count, should_cancel)
            source_frames.SOURCE_CACHE.clear()
            direct_u8 = is_8bit and hasattr(gpu, 'finish_merge_u8')
            if not direct_u8:
                result = create_frame(shape, np.uint16)
            failure = merge_sources(True)
            if failure is None:
                try:
                    if direct_u8:
                        output = create_frame(shape, np.uint8)
                        gpu.finish_merge_u8(output)
                    else:
                        gpu.finish_merge(result)
                except GpuError as exc:
                    output = None
                    failure = exc
        finally:
            gpu.end_merge()
        if failure is not None:
            gpu.failed = True
            log_fallback(failure)
            gpu = None
    if not resident or failure is not None:
        bands, columns = ((side + _MERGE_TILE - 1) // _MERGE_TILE for side in depth.shape)
        # Limit the auxiliary mask for deep stacks.
        tiles = (np.zeros((bands, columns, count), dtype=bool)
                 if bands * columns * count <= _MAX_MERGE_MASK_BYTES else None)
        active = _active_sources(depth, count, should_cancel, tiles=tiles)
        source_frames.SOURCE_CACHE.clear()
        if result is None:
            result = create_frame(shape, np.uint16)
        merge_sources(False)
    if focus_callback is not None:
        focus_callback(depth, (65535 // (count - 1)) * (count - 1))
    del depth
    source_frames.SOURCE_CACHE.clear()
    _check_cancel(should_cancel)
    if output is None:
        output = _output_codes(result, frames, should_cancel)
    del result
    if progress_callback is not None:
        progress_callback(total, total)
    _check_cancel(should_cancel)
    if device_callback is not None:
        from fusion_methods._openfocus_gpu import device_status
        device_callback(device_status(gpu_state))
    return output


def _pyramid_levels(height, width):
    levels = [(height, width)]
    while True:
        next_level = ((height + 1) // 2, (width + 1) // 2)
        # Thin images still need a detail level for focus selection.
        if min(next_level) < 16 and (len(levels) > 1 or next_level == levels[-1]):
            return levels
        levels.append(next_level)
        height, width = next_level


@execution_job
def openfocus_c_impl(input_source, img_resize=None, thread_count: int = None,
                     blend_smoothing: float = 2.4, should_cancel=None,
                     progress_callback=None, use_gpu: bool = True,
                     device_callback=None, focus_callback=None, *, live_preview=None):
    """Fuse source frames using a streaming weighted detail pyramid."""
    from core import source_frames

    frames, shape = _inputs(input_source, img_resize, should_cancel)
    configure_rows(frames, img_resize, 'openfocus_c', thread_count, should_cancel)
    smoothing = 2.4 if blend_smoothing is None else float(blend_smoothing)
    if progress_callback is not None:
        callback = progress_callback
        reported = -1
        def progress_callback(index, total):
            nonlocal reported
            if index > reported:
                callback(index, total)
                reported = index
    try:
        if use_gpu:
            from fusion_methods._openfocus_gpu import GpuError
            from fusion_methods._openfocus_gpu_c import GpuC
            from utils.logging_setup import get_logger

            gpu = None
            try:
                gpu = GpuC()
                args = (frames, shape, img_resize, smoothing * (5.0 / 3.0),
                        should_cancel, progress_callback, focus_callback)
                result = (gpu.fuse(*args) if live_preview is None else
                          gpu.fuse(*args, live_preview=live_preview))
            except GpuError as exc:
                cause = f': {exc.__cause__}' if exc.__cause__ is not None else ''
                get_logger(__name__).warning(
                    'OpenFocus C GPU fallback to CPU for shape=%s device=%s: %s%s',
                    shape, getattr(gpu, 'device_name', 'OpenCL'), exc, cause,
                )
            else:
                if device_callback is not None:
                    device_callback(gpu.device_name)
                return result
        from fusion_methods._openfocus_pyramid import fuse_pyramid
        result = fuse_pyramid(frames, shape, img_resize, smoothing * (5.0 / 3.0),
                              should_cancel, progress_callback, focus_callback,
                              live_preview=live_preview)
        if device_callback is not None:
            device_callback('CPU')
        return result
    finally:
        source_frames.SOURCE_CACHE.clear()
