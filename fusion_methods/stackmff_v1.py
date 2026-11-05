"""StackMFF back-end using its retrained attention backbone.

The network fuses stack luma; colour is reconstructed from the closest source.
The model has no user-tunable parameters.
"""
import os
from contextlib import nullcontext

import numpy as np

from fusion_methods import neural_common as common
from utils.image_utils import fuse_output_dtype

DEFAULT_WEIGHTS = 'weights/stackmff_final.pth'
INFER_SIGMA = 0.5
INFER_SCALES = (1.0, 0.75)
_ALGORITHM = 'stackmff'
_CACHE = None


def _model_and_device(model_path, use_gpu):
    """Load one checkpoint per (path, digest, device) and keep it resident."""
    common.ensure_torch()
    device = common.resolve_device(use_gpu)
    path = os.path.abspath(model_path)
    key = (path, common.checkpoint_digest(path), str(device))
    global _CACHE
    if _CACHE is None or _CACHE[0] != key:
        from core.models.stackmff_v1.network import StackMFF
        from core.models.stackmff_v3.model_io import load_variant_checkpoint
        model = load_variant_checkpoint(path, StackMFF, 'stackmff_v3backbone',
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


def _fuse_tile(frames, model_path, use_gpu, should_cancel=None,
               execution_plan=None, thread_count=None,
               resource_preference='balanced', **_kwargs):
    """Fuse one tile; returns 0-255 pixels in the caller's dtype."""
    common.cancel(should_cancel)
    out_dtype = fuse_output_dtype(frames)
    model, device, _plan = _prepare_inference(
        frames, model_path, use_gpu, thread_count=thread_count,
        resource_preference=resource_preference, execution_plan=execution_plan)
    normalized = common.normalized_frames(frames)
    # The backbone pads internally. Keep the original dimensions here so the
    # second inference scale matches the released benchmark preprocessing.
    planes = common.stage_planes(normalized, 1, should_cancel)
    tensor = planes.unsqueeze(0)  # (B, 1, D, H, W)
    try:
        with common.torch.no_grad():
            output = model(tensor.to(device), sigma=INFER_SIGMA,
                           scales=INFER_SCALES, should_cancel=should_cancel)
        common.cancel(should_cancel)
    finally:
        # Device calls enqueue work. Cancellation must not release the heavy
        # lease while kernels from this invocation still use its resources.
        common.synchronize(device)
    h, w = normalized[0].shape[:2]
    luma = output.squeeze().float().clamp(0, 1).cpu().numpy()[:h, :w] * 255.0
    del tensor, planes
    fused = common.tensor_to_bgr(luma, normalized, should_cancel)
    if out_dtype is np.uint8:
        np.clip(np.round(fused), 0, 255, out=fused)
        return fused.astype(np.uint8)
    return np.clip(fused, 0.0, 255.0, out=fused)


def stackmff_impl(input_source, img_resize, model_path, use_gpu,
                  should_cancel=None, progress_callback=None,
                  thread_count=None, resource_preference='balanced',
                  execution_plan=None, **kwargs):
    """Whole-stack fusion used by both the direct and tiled paths."""
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
        result = _fuse_tile(
            frames, model_path, use_gpu, should_cancel=should_cancel,
            execution_plan=execution_plan, thread_count=thread_count,
            resource_preference=resource_preference)
        common.cancel(should_cancel)
        if progress_callback is not None:
            progress_callback(1, 1)
        common.cancel(should_cancel)
        return result
