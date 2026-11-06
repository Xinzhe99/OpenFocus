"""StackMFF V2 back-end using its retrained attention backbone.

Frame order defines depth positions. The expected layer is min-max normalised
over the padded inference, then the nearest source frame is sampled.
"""
from contextlib import nullcontext

import numpy as np

from fusion_methods import neural_common as common
from utils.image_utils import fuse_output_dtype

PAD_ALIGN = common.ATTENTION_PAD_ALIGN
DEFAULT_WEIGHTS = 'weights/stackmff_v2_final.onnx'
_ALGORITHM = 'stackmffv2'


def _prepare_inference(frames, model_path, use_gpu, thread_count=None,
                       resource_preference='balanced', execution_plan=None,
                       result_count=1, fixed_bytes=0):
    """Admit the retrained model and its attention workspace."""
    from fusion_methods import neural_runtime
    device = neural_runtime.resolve_device(use_gpu)
    return common.plan_attention_inference(
        _ALGORITHM, frames, model_path, device,
        thread_count=thread_count, resource_preference=resource_preference,
        execution_plan=execution_plan, result_count=result_count,
        fixed_bytes=fixed_bytes)


def _inference_depth(planes, model, device, should_cancel=None, threads=1):
    """Return depth normalized over the complete padded inference."""
    from fusion_methods import neural_runtime
    probs = neural_runtime.softmax(neural_runtime.logits(planes, model, device, threads, should_cancel))
    n = probs.shape[1]
    depth = (probs * np.arange(1, n + 1, dtype=np.float32)[None, :, None, None]).sum(axis=1)[0]
    lo, hi = depth.min(), depth.max()
    depth = (depth - lo) / (hi - lo + 1e-8)
    index = np.clip(np.rint(depth * (n - 1)), 0, n - 1).astype(np.int64)
    return depth, index




def _sample(frames, index, should_cancel=None):
    """Reference output: copy the sampled source frame at every pixel."""
    common.cancel(should_cancel)
    from core.frame_storage import create_frame
    out = create_frame(frames[0].shape, np.float32)
    for layer, frame in enumerate(frames):
        common.cancel(should_cancel)
        for start in range(0, out.shape[0], 128):
            common.cancel(should_cancel)
            rows = slice(start, min(out.shape[0], start + 128))
            mask = index[rows] == layer
            np.copyto(out[rows], frame[rows],
                      where=mask[:, :, None] if frame.ndim == 3 else mask,
                      casting='unsafe')
    common.cancel(should_cancel)
    return out[..., 0] if out.ndim == 3 and out.shape[-1] == 1 else out


def infer(frames, model_path, use_gpu, should_cancel=None,
          execution_plan=None, thread_count=None,
          resource_preference='balanced'):
    """Return (fused pixels, depth 0-1, index 0..N-1) for one whole tile."""
    if execution_plan is None:
        from fusion_methods.neural_runtime import start_inference_job
        start_inference_job()
    common.cancel(should_cancel)
    out_dtype = np.float32 if any(f.dtype == np.uint16 for f in frames) else fuse_output_dtype(frames)
    model, device, plan = _prepare_inference(
        frames, model_path, use_gpu, thread_count=thread_count,
        resource_preference=resource_preference, execution_plan=execution_plan)
    normalized = common.normalized_frames(frames, should_cancel)
    planes = common.stage_planes(normalized, PAD_ALIGN, should_cancel)
    depth_full, index_full = _inference_depth(planes, model, device, should_cancel, plan.native_threads)
    del planes
    h, w = normalized[0].shape[:2]
    # The model normalises across the padded map before we crop.
    depth = depth_full[:h, :w]
    index = index_full[:h, :w]
    fused = _sample(normalized, index, should_cancel)
    del depth_full, index_full
    if out_dtype is np.uint8:
        fused = np.clip(np.round(fused), 0, 255).astype(np.uint8)
    else:
        np.clip(fused, 0.0, 255.0, out=fused)
    common.cancel(should_cancel)
    return fused, depth, index


def stackmffv2_impl(input_source, img_resize, model_path, use_gpu,
                    should_cancel=None, progress_callback=None,
                    thread_count=None, resource_preference='balanced',
                    execution_plan=None, tile_size=None, tile_overlap=0,
                    exposure_gains=None, source_all_uint8=None, **kwargs):
    """Whole-stack fusion; the caller may already hold the heavy-job lease."""
    from core.resource_policy import heavy_job

    lease = heavy_job(should_cancel) if execution_plan is None else nullcontext()
    with lease:
        common.cancel(should_cancel)
        if tile_size is not None:
            from fusion_methods.stackmff_v2_tiled import tiled_fusion
            return tiled_fusion(
                input_source, img_resize, model_path, use_gpu, should_cancel,
                execution_plan, thread_count, resource_preference, tile_size,
                tile_overlap, progress_callback, exposure_gains, source_all_uint8)
        if isinstance(input_source, str):
            from core.frame_storage import store_frame
            from fusion_methods.stackmff_v2_tiled import _Source
            source = _Source(input_source, img_resize, None, should_cancel)
            frames = []
            for index in range(len(source.items)):
                common.cancel(should_cancel)
                frames.append(store_frame(source.get(index)))
        else:
            frames = common.read_stack(input_source, img_resize, should_cancel,
                                       execution_plan)
        if len(frames) < 2:
            raise ValueError("At least two images are required for fusion")
        if progress_callback is not None:
            progress_callback(0, 1)
        fused, _depth, _index = infer(
            frames, model_path, use_gpu,
            should_cancel=should_cancel, execution_plan=execution_plan,
            thread_count=thread_count,
            resource_preference=resource_preference)
        if progress_callback is not None:
            progress_callback(1, 1)
        common.cancel(should_cancel)
        return fused


def depth_positions(images, model_path, use_gpu, should_cancel=None,
                    execution_plan=None, thread_count=None,
                    resource_preference='balanced', tile_size=None,
                    tile_overlap=0, progress_callback=None):
    """Sampled layer positions, with adaptive whole-stack spatial tiling."""
    from core.resource_policy import ResourceLimitError
    from fusion_methods.stackmff_v2_tiled import _Source, tiled_depth
    images = list(images)
    if tile_size is None:
        try:
            _prepare_inference(images, model_path, use_gpu,
                               thread_count=thread_count,
                               resource_preference=resource_preference,
                               execution_plan=execution_plan)
        except ResourceLimitError:
            pass
        else:
            if progress_callback is not None:
                progress_callback(0, 1)
            fused, _depth, index = infer(
                images, model_path, use_gpu, should_cancel=should_cancel,
                execution_plan=execution_plan, thread_count=thread_count,
                resource_preference=resource_preference)
            if progress_callback is not None:
                progress_callback(1, 1)
            return index.astype(np.float32)
    source = _Source(images, None, None, should_cancel)
    _depth, index = tiled_depth(
        source, model_path, use_gpu, should_cancel, execution_plan,
        thread_count, resource_preference, tile_size, tile_overlap,
        progress_callback)
    return index.astype(np.float32)
