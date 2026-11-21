"""Bounded spatial inference for StackMFF V2.

Each inference sees the complete focal stack.  Tiles contribute expected layer
positions to a single feathered image, which is normalized once before any
source pixels are selected.
"""


import cv2
import numpy as np

from fusion_methods import neural_common as common
from utils.image_utils import apply_exposure_gain
from core.frame_storage import create_frame
from fusion_methods.stackmff_v2 import _STRIP_ROWS, _expected_positions, _sample


class _Source:
    def __init__(self, source, resize, gains, should_cancel):
        self.gains = gains
        self.should_cancel = should_cancel
        self.directory = isinstance(source, str)
        self.items = common.read_stack(source, None, should_cancel, preserve_grayscale=False)
        if len(self.items) < 2:
            raise ValueError('At least two images are required for fusion')
        self.mixed_16 = not self.directory and any(x.dtype == np.uint16 for x in self.items)
        first = self.items[0]
        tail = first.shape[2:] if first.ndim != 3 or first.shape[2] != 1 else ()
        self.shape = ((resize[1], resize[0]) + tail) if resize else first.shape
        self.dtype = np.dtype(np.float32) if gains is not None and self.mixed_16 else first.dtype
        self.original_items = None
        if resize:
            from core.source_frames import (SourceFrame, ProcessedFrame, process_frame,
                                            materialize_frames, frame_resolution_bytes,
                                            source_cache_growth_bytes)
            from core.frame_storage import scratch_directory, select_scratch
            from core.resource_policy import require_memory
            mixed_16 = self.mixed_16

            def prepare(image, index):
                if gains is not None:
                    image = common.working_frame(image, mixed_16)
                    image = apply_exposure_gain(image, gains[index])
                return cv2.resize(image, resize)

            prepared = []
            paths = []
            for index, frame in enumerate(self.items):
                common.cancel(should_cancel)
                shape = (resize[1], resize[0]) + (frame.shape[2:]
                    if frame.ndim != 3 or frame.shape[2] != 1 else ())
                dtype = np.float32 if gains is not None and mixed_16 else frame.dtype
                prepared.append(process_frame(frame,
                    lambda image, index=index: prepare(image, index), shape=shape, dtype=dtype,
                    should_cancel=should_cancel,
                    scratch_bytes=frame.nbytes * 4 + int(np.prod(shape)) * np.dtype(dtype).itemsize * 2))
                original = frame
                while isinstance(original, ProcessedFrame):
                    original = original.source
                if isinstance(original, SourceFrame):
                    paths.append(original.path)
            require_memory(max(frame_resolution_bytes(frame) for frame in prepared)
                           + source_cache_growth_bytes(prepared))
            self.original_items = self.items
            with scratch_directory():
                select_scratch(sum(frame.nbytes for frame in prepared), paths)
                self.items = materialize_frames(prepared, should_cancel, thread_count=1,
                                                prefer_ram=False)
            self.gains = None

    def get(self, index, region=None):
        common.cancel(self.should_cancel)
        from core.source_frames import SourceFrame, ProcessedFrame, resolve_frame
        if self.original_items is not None:
            original = self.original_items[index]
            while isinstance(original, ProcessedFrame):
                original = original.source
            if isinstance(original, SourceFrame):
                original.validate_source()
        item = self.items[index]
        if item.shape[:2] != self.shape[:2]:
            raise ValueError('All frames must have the same dimensions')
        frame = (resolve_frame(item, should_cancel=self.should_cancel) if region is None
                 else self._crop(item, region, self.should_cancel))
        if self.gains is not None:
            if self.mixed_16:
                if frame.dtype == np.uint16:
                    frame = frame.astype(np.float32) * (255.0 / 65535.0)
                elif frame.dtype != np.float32:
                    frame = frame.astype(np.float32)
            frame = apply_exposure_gain(frame, self.gains[index])
        return frame

    @staticmethod
    def _crop(frame, region, should_cancel):
        x, y, width, height = region
        h, w = frame.shape[:2]
        from core.frame_storage import is_mapped_frame
        from core.source_frames import borrow_frame_region

        if x + width <= w and y + height <= h:
            region = np.s_[y:y + height, x:x + width]
            if isinstance(frame, np.ndarray) and not is_mapped_frame(frame):
                return frame[region]
            with borrow_frame_region(frame, region, should_cancel=should_cancel) as crop:
                return crop.copy(order='C')
        ph, pw = (-h) % common.ATTENTION_PAD_ALIGN, (-w) % common.ATTENTION_PAD_ALIGN
        mode = 'reflect' if ph < h and pw < w else 'edge'
        rows = np.pad(np.arange(h), (0, ph), mode=mode)[y:y + height]
        cols = np.pad(np.arange(w), (0, pw), mode=mode)[x:x + width]
        top, left = rows.min(), cols.min()
        key = np.s_[top:rows.max() + 1, left:cols.max() + 1]
        with borrow_frame_region(frame, key, should_cancel=should_cancel) as crop:
            return crop[(rows - top)[:, None], cols[None, :] - left]




def tiled_output_bytes(height, width, channels, result_count):
    """Bound the full-canvas maps and strip workspace used by V2."""
    padded_h = height + (-height) % common.ATTENTION_PAD_ALIGN
    padded_w = width + (-width) % common.ATTENTION_PAD_ALIGN
    pixels = padded_h * padded_w
    strip_pixels = min(padded_h, _STRIP_ROWS) * padded_w
    depth_bytes = pixels * 12 + strip_pixels * 12
    if not result_count:
        return depth_bytes
    # Depth is released before sampling. Gathering and exposure adjustment
    # need at most two float32 strips in addition to the selection mask.
    output_bytes = pixels * 8 + height * width * channels * 4 * result_count
    return max(depth_bytes, output_bytes + strip_pixels * max(17, 1 + channels * 8))


def _plan(source, model_path, use_gpu, tile_size, execution_plan,
          thread_count, resource_preference, result_count):
    from core.resource_policy import ResourceLimitError, plan_neural_tiles
    from fusion_methods.neural_runtime import resolve_job_device, estimate_inference_bytes
    from fusion_methods.stackmff_v2 import _prepare_inference
    h, w = source.shape[:2]
    channels = source.shape[2] if len(source.shape) == 3 else 1
    device = (execution_plan.device_kind if execution_plan is not None and use_gpu
              else resolve_job_device(use_gpu))
    from core.source_frames import frame_resolution_bytes, source_cache_growth_bytes
    source_resolution = max(frame_resolution_bytes(frame) for frame in source.items)
    source_cache = source_cache_growth_bytes(source.items)
    block, outer = plan_neural_tiles(
        len(source.items), h, w, channels, tile_size or min(h, w), device,
        estimate_inference_bytes, directory=False,
        source_resolution_bytes=source_resolution, source_cache_bytes=source_cache,
        itemsize=source.dtype.itemsize, preference=resource_preference,
        user_threads=thread_count)
    if execution_plan is not None:
        from dataclasses import replace
        outer = replace(outer, ram_budget=min(outer.ram_budget, execution_plan.ram_budget),
                        vram_budget=min(outer.vram_budget, execution_plan.vram_budget)
                        if execution_plan.vram_budget else outer.vram_budget,
                        device_fallback=outer.device_fallback or execution_plan.device_fallback,
                        native_threads=min(outer.native_threads, execution_plan.native_threads))
    fixed = tiled_output_bytes(h, w, channels, result_count)
    while True:
        try:
            model, device, plan = _prepare_inference(
                source.items, model_path, use_gpu, height=block, width=block,
                thread_count=thread_count, resource_preference=resource_preference,
                execution_plan=outer, result_count=result_count, fixed_bytes=fixed)
            return block, model, device, plan
        except ResourceLimitError:
            if block <= min(64, h, w):
                raise
            block = max(min(64, h, w), block // 2)


def _weights(x0, y0, x1, y1, width, height, overlap):
    from core.multi_focus_fusion import MultiFocusFusion
    return MultiFocusFusion._compute_tile_weights(
        None, x0, y0, x1, y1, x1-x0, y1-y0, width, height, overlap)


def _stage_tile_planes(source, region, should_cancel):
    """Stage an already padded tile, releasing each crop before the next read."""
    _, _, width, height = region
    count = len(source.items)
    planes = np.empty((1, count, height, width), dtype=np.float32)
    for index in range(count):
        common.cancel(should_cancel)
        frame = source.get(index, region)
        if frame.shape[:2] != (height, width):
            raise ValueError('All frames must have the same dimensions')
        frame = common.working_frame(frame, source.mixed_16 or frame.dtype == np.uint16)
        # Keep the packed conversion grid for OpenCV's edge rounding.
        common.write_luma_plane(np.ascontiguousarray(frame), planes[0, index])
        del frame
    common.cancel(should_cancel)
    return planes


_MAX_V2_CHUNK_BYTES = 512 * 1024 ** 2


def _stage_direct_rgb_chunk(source, y, chunk_xs, block, source_h, source_w,
                            all_rows, all_cols, kernels, io_workers, should_cancel,
                            *, worker_buffers=None):
    """Stage one horizontal chunk of direct RGB TIFF tiles into a batch of luma planes."""
    from concurrent.futures import ThreadPoolExecutor
    from core.source_frames import read_rgb_tiff_rows

    count = len(source.items)
    k_count = len(chunk_xs)
    chunk_planes = np.empty((k_count, count, block, block), dtype=np.float32)
    if y + block <= source_h:
        y_top, y_bot, local_rows = y, y + block, None
    else:
        rows_slice = all_rows[y:y + block]
        y_top, y_bot = int(rows_slice.min()), int(rows_slice.max()) + 1
        local_rows = (rows_slice - y_top).astype(np.int32)

    tile_specs = []
    for x in chunk_xs:
        if local_rows is None and x + block <= source_w:
            tile_specs.append((True, x, None))
        else:
            r_loc = (local_rows if local_rows is not None
                     else np.arange(y - y_top, y - y_top + block, dtype=np.int32))
            c_loc = all_cols[x:x + block].astype(np.int32)
            tile_specs.append((False, r_loc, c_loc))

    has_u16 = any(item.dtype == np.uint16 for item in source.items)
    has_u8 = any(item.dtype == np.uint8 for item in source.items)
    mixed_16 = bool(source.mixed_16 or has_u16)
    gains = source.gains
    if worker_buffers is None:
        worker_u16 = [np.empty((block, source_w, 3), dtype=np.uint16) if has_u16 else None
                      for _ in range(io_workers)]
        worker_u8 = [np.empty((block, source_w, 3), dtype=np.uint8) if has_u8 else None
                     for _ in range(io_workers)]
        worker_bgr_f32 = [np.empty((block, block, 3), dtype=np.float32) if mixed_16 else None
                          for _ in range(io_workers)]
        worker_bgr_u8 = [np.empty((block, block, 3), dtype=np.uint8)
                         if (has_u8 and not mixed_16 and gains is not None) else None
                         for _ in range(io_workers)]
    else:
        worker_u16, worker_u8, worker_bgr_f32, worker_bgr_u8 = worker_buffers

    inv255 = np.float32(255.0)

    def _stage_worker(worker_id):
        b_u16 = worker_u16[worker_id]
        b_u8 = worker_u8[worker_id]
        bgr_f32 = worker_bgr_f32[worker_id]
        bgr_u8 = worker_bgr_u8[worker_id]
        for idx in range(worker_id, count, io_workers):
            common.cancel(should_cancel)
            item = source.items[idx]
            gain = float(gains[idx]) if gains is not None else 1.0
            if item.dtype == np.uint16:
                band = read_rgb_tiff_rows(item, y_top, y_bot, b_u16, should_cancel=should_cancel)
                for k_idx, (fast, a, b_idx) in enumerate(tile_specs):
                    if fast:
                        kernels.pack_rgb_u16_to_bgr_f32(band, 0, a, block, block, bgr_f32, gain)
                    else:
                        kernels.pack_rgb_u16_to_bgr_f32_indexed(band, a, b_idx, bgr_f32, gain)
                    dst = chunk_planes[k_idx, idx]
                    cv2.cvtColor(bgr_f32, cv2.COLOR_BGR2GRAY, dst=dst)
                    np.divide(dst, inv255, out=dst, dtype=np.float32)
            else:
                band = read_rgb_tiff_rows(item, y_top, y_bot, b_u8, should_cancel=should_cancel)
                for k_idx, (fast, a, b_idx) in enumerate(tile_specs):
                    dst = chunk_planes[k_idx, idx]
                    if mixed_16:
                        if fast:
                            kernels.pack_rgb_u8_to_bgr_f32(band, 0, a, block, block, bgr_f32, gain)
                        else:
                            kernels.pack_rgb_u8_to_bgr_f32_indexed(band, a, b_idx, bgr_f32, gain)
                        cv2.cvtColor(bgr_f32, cv2.COLOR_BGR2GRAY, dst=dst)
                        np.divide(dst, inv255, out=dst, dtype=np.float32)
                    elif gain == 1.0:
                        if fast:
                            kernels.rgb_u8_to_luma(band, 0, a, block, block, dst)
                        else:
                            kernels.rgb_u8_to_luma_indexed(band, a, b_idx, dst)
                    else:
                        if fast:
                            kernels.pack_rgb_u8_to_bgr_u8(band, 0, a, block, block, bgr_u8)
                        else:
                            kernels.pack_rgb_u8_to_bgr_u8_indexed(band, a, b_idx, bgr_u8)
                        adjusted = apply_exposure_gain(bgr_u8, gain)
                        common.write_luma_plane(adjusted, dst)

    if io_workers > 1:
        with ThreadPoolExecutor(max_workers=io_workers) as executor:
            futures = [executor.submit(_stage_worker, wid) for wid in range(io_workers)]
            for fut in futures:
                fut.result()
    else:
        _stage_worker(0)
    common.cancel(should_cancel)
    return chunk_planes


def tiled_depth(source, model_path, use_gpu, should_cancel=None,
                execution_plan=None, thread_count=None,
                resource_preference='balanced', tile_size=None, tile_overlap=0,
                progress_callback=None, result_count=0):
    """Return globally normalized depth and sampled indices for a source."""
    with common.neural_lease(execution_plan, should_cancel):
        return _tiled_depth_under_lease(
            source, model_path, use_gpu, should_cancel, execution_plan,
            thread_count, resource_preference, tile_size, tile_overlap,
            progress_callback, result_count)


def _tiled_depth_under_lease(source, model_path, use_gpu, should_cancel=None,
                             execution_plan=None, thread_count=None,
                             resource_preference='balanced', tile_size=None, tile_overlap=0,
                             progress_callback=None, result_count=0, tile_grid_callback=None):
    from fusion_methods import neural_runtime
    source_h, source_w = source.shape[:2]
    h = source_h + (-source_h) % common.ATTENTION_PAD_ALIGN
    w = source_w + (-source_w) % common.ATTENTION_PAD_ALIGN
    count = len(source.items)
    block, model, device, plan = _plan(source, model_path, use_gpu, tile_size,
                                      execution_plan, thread_count,
                                      resource_preference, result_count)
    block = max(block, min(common.ATTENTION_PAD_ALIGN, h, w))
    if block >= common.ATTENTION_PAD_ALIGN:
        block = (block // common.ATTENTION_PAD_ALIGN) * common.ATTENTION_PAD_ALIGN
    overlap = max(0, min(int(tile_overlap), block // 2))
    from core.multi_focus_fusion import _tile_axis_starts
    ys = sorted(set(_tile_axis_starts(h, block, overlap, common.ATTENTION_PAD_ALIGN)))
    xs = sorted(set(_tile_axis_starts(w, block, overlap, common.ATTENTION_PAD_ALIGN)))
    total = len(xs) * len(ys)
    if tile_grid_callback is not None:
        tile_grid_callback(block, overlap, total)

    from core.source_frames import is_direct_rgb_tiff

    kernels = None
    if len(xs) > 1 and all(is_direct_rgb_tiff(item) for item in source.items):
        for item in source.items:
            if item.shape[:2] != (source_h, source_w):
                raise ValueError('All frames must have the same dimensions')
        has_u16 = any(item.dtype == np.uint16 for item in source.items)
        has_u8 = any(item.dtype == np.uint8 for item in source.items)
        mixed_16 = source.mixed_16 or has_u16
        io_workers = min(max(1, int(plan.native_threads or 1)), 4, count)
        tile_pixels = block * block
        worker_bytes = block * source_w * 3 * (2 * has_u16 + has_u8)
        if mixed_16:
            worker_bytes += tile_pixels * 3 * 4
        elif source.gains is not None:
            worker_bytes += tile_pixels * (3 * 7 + 1)
        kernels, max_k = common.plan_rgb_staging(plan, len(xs), count * tile_pixels * 4,
                                                io_workers * worker_bytes, _MAX_V2_CHUNK_BYTES)

    expected = create_frame((h, w), np.float32)
    # Edge tiles can overlap even at zero requested overlap. Only a grid
    # that divides the padded canvas exactly writes every pixel once.
    weight = None
    if overlap or h % block or w % block:
        weight = create_frame((h, w), np.float32)
        expected.fill(0)
        weight.fill(0)
    if progress_callback is not None:
        progress_callback(0, total)
    done = 0

    def _accumulate_tile(x_pos, y_pos, tile_planes):
        nonlocal done
        logits = neural_runtime.logits(tile_planes, model, device, plan.native_threads, should_cancel)
        positions = _expected_positions(logits, should_cancel, threads=plan.native_threads)[:block, :block]
        del logits
        rows_sl, cols_sl = slice(y_pos, y_pos + block), slice(x_pos, x_pos + block)
        if weight is None:
            expected[rows_sl, cols_sl] = positions
        else:
            weights = _weights(x_pos, y_pos, x_pos + block, y_pos + block, w, h, overlap)
            expected[rows_sl, cols_sl] += positions * weights
            weight[rows_sl, cols_sl] += weights
        done += 1
        if progress_callback is not None:
            progress_callback(done, total)

    if kernels is not None:
        ph = (-source_h) % common.ATTENTION_PAD_ALIGN
        pw = (-source_w) % common.ATTENTION_PAD_ALIGN
        pad_mode = 'reflect' if ph < source_h and pw < source_w else 'edge'
        all_rows = (np.pad(np.arange(source_h, dtype=np.int32), (0, ph), mode=pad_mode)
                    if ph else np.arange(source_h, dtype=np.int32))
        all_cols = (np.pad(np.arange(source_w, dtype=np.int32), (0, pw), mode=pad_mode)
                    if pw else np.arange(source_w, dtype=np.int32))
        worker_buffers = (
            [np.empty((block, source_w, 3), dtype=np.uint16) if has_u16 else None
             for _ in range(io_workers)],
            [np.empty((block, source_w, 3), dtype=np.uint8) if has_u8 else None
             for _ in range(io_workers)],
            [np.empty((block, block, 3), dtype=np.float32) if mixed_16 else None
             for _ in range(io_workers)],
            [np.empty((block, block, 3), dtype=np.uint8)
             if (has_u8 and not mixed_16 and source.gains is not None) else None
             for _ in range(io_workers)],
        )

        for y in ys:
            common.cancel(should_cancel)
            for c_start in range(0, len(xs), max_k):
                common.cancel(should_cancel)
                chunk_xs = xs[c_start:c_start + max_k]
                if len(chunk_xs) == 1:
                    x = chunk_xs[0]
                    planes = _stage_tile_planes(source, (x, y, block, block), should_cancel)
                    _accumulate_tile(x, y, planes)
                    del planes
                    continue
                chunk_planes = _stage_direct_rgb_chunk(
                    source, y, chunk_xs, block, source_h, source_w,
                    all_rows, all_cols, kernels, io_workers, should_cancel,
                    worker_buffers=worker_buffers)
                for k_idx, x in enumerate(chunk_xs):
                    common.cancel(should_cancel)
                    _accumulate_tile(x, y, chunk_planes[k_idx:k_idx + 1])
                del chunk_planes
    else:
        for y in ys:
            for x in xs:
                common.cancel(should_cancel)
                planes = _stage_tile_planes(source, (x, y, block, block), should_cancel)
                _accumulate_tile(x, y, planes)
                del planes
    common.cancel(should_cancel)
    if weight is not None:
        for start in range(0, h, 128):
            common.cancel(should_cancel)
            rows = slice(start, min(h, start + 128))
            np.divide(expected[rows], weight[rows], out=expected[rows], where=weight[rows] != 0)
        del weight
    lo, hi = expected.min(), expected.max()
    expected -= lo
    expected /= hi - lo + 1e-8
    index = create_frame((h, w), np.int64)
    for start in range(0, h, 128):
        common.cancel(should_cancel)
        rows = slice(start, min(h, start + 128))
        index[rows] = np.clip(np.rint(expected[rows] * (count - 1)), 0, count - 1).astype(np.int64)
    return expected[:source_h, :source_w], index[:source_h, :source_w]


def tiled_fusion(input_source, img_resize, model_path, use_gpu,
                 should_cancel=None, execution_plan=None, thread_count=None,
                 resource_preference='balanced', tile_size=None, tile_overlap=0,
                 progress_callback=None, exposure_gains=None, source_all_uint8=None):
    """Fuse a list or directory without materializing a full-frame stack."""
    with common.neural_lease(execution_plan, should_cancel):
        return _tiled_fusion_under_lease(
            input_source, img_resize, model_path, use_gpu, should_cancel,
            execution_plan, thread_count, resource_preference, tile_size,
            tile_overlap, progress_callback, exposure_gains, source_all_uint8)


def _tiled_fusion_under_lease(input_source, img_resize, model_path, use_gpu,
                              should_cancel=None, execution_plan=None, thread_count=None,
                              resource_preference='balanced', tile_size=None, tile_overlap=0,
                              progress_callback=None, exposure_gains=None, source_all_uint8=None,
                              focus_callback=None, tile_grid_callback=None):
    source = _Source(input_source, img_resize, exposure_gains, should_cancel)
    total_tiles = 0

    def report_tile(done, total):
        nonlocal total_tiles
        total_tiles = total
        progress_callback(done, total + 1)

    depth, index = _tiled_depth_under_lease(source, model_path, use_gpu, should_cancel,
                               execution_plan, thread_count, resource_preference,
                               tile_size, tile_overlap, report_tile if progress_callback else None,
                               1, tile_grid_callback)
    del depth
    if source_all_uint8 is None:
        source_all_uint8 = (source.items[0].dtype == np.uint8 if source.directory else
                            all(item.dtype == np.uint8 for item in source.items))
    output = _sample(source.items, index, should_cancel,
                     dtype=np.uint8 if source_all_uint8 else np.float32,
                     gains=source.gains, normalize_for_exposure=source.mixed_16,
                     original_frames=source.original_items)
    if focus_callback is not None:
        focus_callback(index, len(source.items) - 1)
    if progress_callback is not None:
        progress_callback(total_tiles + 1, total_tiles + 1)
    common.cancel(should_cancel)
    return output
