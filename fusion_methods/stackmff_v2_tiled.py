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
        frame = resolve_frame(self.items[index])
        if frame.shape[:2] != self.shape[:2]:
            raise ValueError('All frames must have the same dimensions')
        if region is not None:
            frame = self._crop(frame, region)
        if self.gains is not None:
            if self.mixed_16:
                if frame.dtype == np.uint16:
                    frame = frame.astype(np.float32) * (255.0 / 65535.0)
                elif frame.dtype != np.float32:
                    frame = frame.astype(np.float32)
            frame = apply_exposure_gain(frame, self.gains[index])
        return frame

    @staticmethod
    def _crop(frame, region):
        x, y, width, height = region
        h, w = frame.shape[:2]
        if x + width <= w and y + height <= h:
            return frame[y:y + height, x:x + width]
        ph, pw = (-h) % common.ATTENTION_PAD_ALIGN, (-w) % common.ATTENTION_PAD_ALIGN
        mode = 'reflect' if ph < h and pw < w else 'edge'
        rows = np.pad(np.arange(h), (0, ph), mode=mode)[y:y + height]
        cols = np.pad(np.arange(w), (0, pw), mode=mode)[x:x + width]
        return frame[rows[:, None], cols[None, :]]


def _normalized(frame):
    if frame.dtype == np.uint16:
        return frame.astype(np.float32) * (255.0 / 65535.0)
    return frame


def _plan(source, model_path, use_gpu, tile_size, execution_plan,
          thread_count, resource_preference, result_count):
    from core.resource_policy import ResourceLimitError, plan_neural_tiles
    from fusion_methods.neural_runtime import resolve_device, estimate_inference_bytes
    from fusion_methods.stackmff_v2 import _prepare_inference
    h, w = source.shape[:2]
    channels = source.shape[2] if len(source.shape) == 3 else 1
    device = (execution_plan.device_kind if execution_plan is not None
              else resolve_device(use_gpu))
    from core.source_frames import frame_resolution_bytes, source_cache_growth_bytes
    source_resolution = max(frame_resolution_bytes(frame) for frame in source.items)
    source_cache = source_cache_growth_bytes(source.items)
    if source.gains is not None:
        source_resolution += h * w * channels * max(4, source.dtype.itemsize) * 4
    block, outer = plan_neural_tiles(
        len(source.items), h, w, channels, tile_size or min(h, w), device,
        estimate_inference_bytes, directory=False,
        source_resolution_bytes=source_resolution, source_cache_bytes=source_cache,
        itemsize=source.dtype.itemsize, preference=resource_preference,
        user_threads=thread_count)
    if execution_plan is not None:
        from dataclasses import replace
        outer = replace(outer, ram_budget=min(outer.ram_budget, execution_plan.ram_budget),
                        native_threads=min(outer.native_threads, execution_plan.native_threads))
    first = source.get(0)
    padded_h = h + (-h) % common.ATTENTION_PAD_ALIGN
    padded_w = w + (-w) % common.ATTENTION_PAD_ALIGN
    fixed = padded_h * padded_w * (32 + channels * 8 * result_count)
    fixed += source_resolution + source_cache
    while True:
        crop = first[:block, :block]
        try:
            model, device, plan = _prepare_inference(
                [crop] * len(source.items), model_path, use_gpu,
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


def tiled_depth(source, model_path, use_gpu, should_cancel=None,
                execution_plan=None, thread_count=None,
                resource_preference='balanced', tile_size=None, tile_overlap=0,
                progress_callback=None, result_count=0):
    """Return globally normalized depth and sampled indices for a source."""
    from fusion_methods import neural_runtime
    source_h, source_w = source.shape[:2]
    h = source_h + (-source_h) % common.ATTENTION_PAD_ALIGN
    w = source_w + (-source_w) % common.ATTENTION_PAD_ALIGN
    count = len(source.items)
    block, model, device, plan = _plan(source, model_path, use_gpu, tile_size,
                                      execution_plan, thread_count,
                                      resource_preference, result_count)
    block = max(block, min(common.ATTENTION_PAD_ALIGN, h, w))
    overlap = max(0, min(int(tile_overlap), block // 2))
    step = block - overlap
    ys = sorted({min(y, h - block) for y in range(0, h, step)})
    xs = sorted({min(x, w - block) for x in range(0, w, step)})
    total = len(xs) * len(ys)
    expected = create_frame((h, w), np.float32)
    weight = create_frame((h, w), np.float32)
    expected[...] = 0
    weight[...] = 0
    if progress_callback is not None:
        progress_callback(0, total)
    done = 0
    for y in ys:
        for x in xs:
            common.cancel(should_cancel)
            crops = []
            for i in range(count):
                frame = source.get(i, (x, y, block, block))
                if frame.shape[:2] != (block, block):
                    raise ValueError('All frames must have the same dimensions')
                crops.append(common.working_frame(
                    frame, source.mixed_16 or frame.dtype == np.uint16).copy())
            planes = common.stage_planes(crops, common.ATTENTION_PAD_ALIGN, should_cancel)
            del crops
            probs = neural_runtime.softmax(neural_runtime.logits(
                planes, model, device, plan.native_threads, should_cancel))
            del planes
            positions = (probs * np.arange(1, count + 1, dtype=np.float32)
                         [None, :, None, None]).sum(axis=1)[0, :block, :block]
            del probs
            weights = _weights(x, y, x + block, y + block, w, h, overlap)
            expected[y:y+block, x:x+block] += positions * weights
            weight[y:y+block, x:x+block] += weights
            done += 1
            if progress_callback is not None:
                progress_callback(done, total)
    common.cancel(should_cancel)
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
    source = _Source(input_source, img_resize, exposure_gains, should_cancel)
    depth, index = tiled_depth(source, model_path, use_gpu, should_cancel,
                               execution_plan, thread_count, resource_preference,
                               tile_size, tile_overlap, progress_callback, 1)
    del depth
    first = _normalized(source.get(0))
    if source_all_uint8 is None:
        source_all_uint8 = all(item.dtype == np.uint8 for item in source.items) if not source.directory else first.dtype == np.uint8
    output = create_frame(first.shape, np.uint8 if source_all_uint8 else np.float32)
    for i in range(len(source.items)):
        common.cancel(should_cancel)
        frame = first if i == 0 else _normalized(source.get(i))
        for start in range(0, output.shape[0], 128):
            common.cancel(should_cancel)
            rows = slice(start, min(output.shape[0], start + 128))
            mask = index[rows] == i
            np.copyto(output[rows], frame[rows],
                      where=mask[:, :, None] if frame.ndim == 3 else mask,
                      casting='unsafe')
    if output.dtype != np.uint8:
        np.clip(output, 0, 255, out=output)
    common.cancel(should_cancel)
    return output[..., 0] if output.ndim == 3 and output.shape[-1] == 1 else output
