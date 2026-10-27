"""StackMFF (2025) back-end: one 3D pass over the whole focus stack.

The network reads the stack's luma planes and predicts the fused luma; colour
is reconstructed from the source frame closest to that prediction, matching
the upstream reference. The model has no user-tunable parameters.
"""
import os
from contextlib import nullcontext

import numpy as np

from fusion_methods import neural_common as common
from utils.image_utils import fuse_output_dtype

PAD_ALIGN = 16
DEFAULT_WEIGHTS = 'weights/stackmff_final.pth'
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
        state = common.torch.load(path, map_location='cpu')
        state = {name[7:] if name.startswith('module.') else name: value
                 for name, value in state.items()}
        model = StackMFF()
        model.load_state_dict(state)
        model.to(device).eval()
        _CACHE = (key, model, device)
    return _CACHE[1], _CACHE[2]


def _prepare_inference(frames, model_path, use_gpu, thread_count=None,
                       resource_preference='balanced', execution_plan=None,
                       result_count=1, fixed_bytes=0):
    """Admit the checkpoint plus this architecture's 3D convolution workspace."""
    h, w = frames[0].shape[:2]
    hp, wp = h + (-h) % PAD_ALIGN, w + (-w) % PAD_ALIGN
    n = len(frames)
    # Five encoder stages at full/2/4/8/16 resolution plus the decoder mirrors.
    # Coefficients are conservative allocation terms, not calibrated RSS.
    workspace = int(64.0 * n * hp * wp * 4)
    host = int(n * hp * wp * 4 * 6 + result_count * h * w * 3 * 4 + fixed_bytes)
    device, plan = common.plan_inference(
        _ALGORITHM, frames, model_path, use_gpu, thread_count,
        resource_preference, execution_plan,
        workspace_bytes=workspace, host_bytes=host)
    model, device = _model_and_device(model_path, use_gpu)
    return model, device, plan


def _fuse_tile(frames, model_path, use_gpu, should_cancel=None,
               execution_plan=None, **_kwargs):
    """Fuse one tile; returns 0-255 pixels in the caller's dtype."""
    common.cancel(should_cancel)
    out_dtype = fuse_output_dtype(frames)
    normalized = common.normalized_frames(frames)
    planes = common.stage_planes(normalized, PAD_ALIGN, should_cancel)
    model, device, _plan = _prepare_inference(
        normalized, model_path, use_gpu, execution_plan=execution_plan)
    tensor = planes.unsqueeze(0)  # (B, 1, D, Hp, Wp)
    try:
        with common.torch.no_grad():
            output = model(tensor.to(device))
        common.cancel(should_cancel)
    finally:
        if getattr(device, 'type', '') == 'cuda':
            common.torch.cuda.synchronize(device)
        elif getattr(device, 'type', '') == 'mps':
            common.torch.mps.synchronize()
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
