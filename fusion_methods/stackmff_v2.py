"""StackMFF V2 back-end using its retrained attention backbone.

Frame order defines depth positions. The expected layer is min-max normalised
over the padded inference, then the nearest source frame is sampled.
"""
import os
from contextlib import nullcontext

import numpy as np

from fusion_methods import neural_common as common
from utils.image_utils import fuse_output_dtype

PAD_ALIGN = common.ATTENTION_PAD_ALIGN
DEFAULT_WEIGHTS = 'weights/stackmff_v2_final.pth'
_ALGORITHM = 'stackmffv2'
_CACHE = None


def _model_and_device(model_path, use_gpu):
    common.ensure_torch()
    device = common.resolve_device(use_gpu)
    path = os.path.abspath(model_path)
    key = (path, common.checkpoint_digest(path), str(device))
    global _CACHE
    if _CACHE is None or _CACHE[0] != key:
        from core.models.stackmff_v2.network import StackMFF_V2
        from core.models.stackmff_v3.model_io import load_variant_checkpoint
        model = load_variant_checkpoint(path, StackMFF_V2, 'v2_v3backbone',
                                        device=device)
        _CACHE = (key, model, device)
    return _CACHE[1], _CACHE[2]


def _prepare_inference(frames, model_path, use_gpu, thread_count=None,
                       resource_preference='balanced', execution_plan=None,
                       result_count=1, fixed_bytes=0):
    """Admit the retrained model and its attention workspace."""
    common.ensure_torch()
    device = common.resolve_device(use_gpu)
    return common.plan_attention_inference(
        _ALGORITHM, frames, model_path, device, common.torch,
        lambda: _model_and_device(model_path, use_gpu),
        thread_count=thread_count, resource_preference=resource_preference,
        execution_plan=execution_plan, result_count=result_count,
        fixed_bytes=fixed_bytes)


def _inference_depth(planes, model, device, should_cancel=None):
    """Return the reference model's padded depth and source indices."""
    common.cancel(should_cancel)
    tensor = planes.to(device)
    try:
        with common.torch.no_grad():
            _luma, depth, index = model(tensor)
        common.cancel(should_cancel)
    finally:
        # Wait for queued kernels before releasing the heavy-job lease.
        common.synchronize(device)
    return depth[0, 0].cpu().numpy(), index[0].cpu().numpy()




def _sample(frames, index, should_cancel=None):
    """Reference output: copy the sampled source frame at every pixel."""
    common.cancel(should_cancel)
    stack = np.stack(frames).astype(np.float32, copy=False)
    if stack.ndim == 3:
        stack = stack[..., None]
    h, w = stack.shape[1:3]
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    out = stack[index, rows, cols]
    common.cancel(should_cancel)
    return out[..., 0] if out.shape[-1] == 1 else out


def infer(frames, model_path, use_gpu, should_cancel=None,
          execution_plan=None, thread_count=None,
          resource_preference='balanced'):
    """Return (fused pixels, depth 0-1, index 0..N-1) for one whole tile."""
    common.cancel(should_cancel)
    out_dtype = fuse_output_dtype(frames)
    model, device, _plan = _prepare_inference(
        frames, model_path, use_gpu, thread_count=thread_count,
        resource_preference=resource_preference, execution_plan=execution_plan)
    normalized = common.normalized_frames(frames)
    planes = common.stage_planes(normalized, PAD_ALIGN, should_cancel)
    depth_full, index_full = _inference_depth(planes, model, device, should_cancel)
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
                    execution_plan=None, **kwargs):
    """Whole-stack fusion; the caller may already hold the heavy-job lease."""
    from core.resource_policy import heavy_job

    lease = heavy_job(should_cancel) if execution_plan is None else nullcontext()
    with lease:
        common.cancel(should_cancel)
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
                    resource_preference='balanced'):
    """Sampled layer positions in 0..N-1 for the depth-map pipeline."""
    fused, _depth, index = infer(
        list(images), model_path, use_gpu,
        should_cancel=should_cancel, execution_plan=execution_plan,
        thread_count=thread_count, resource_preference=resource_preference)
    del fused
    return index.astype(np.float32)
