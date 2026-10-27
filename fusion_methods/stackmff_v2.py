"""StackMFF V2 back-end: per-layer focus estimation plus cross-layer regression.

The stack's focus order is meaningful: a bidirectional recurrent layer reads
it, so the frames must stay in acquisition order. Fusion follows the reference
rule exactly: the expected layer is min-max normalised over the padded
inference, scaled to the stack length and the nearest source frame is sampled.
The same sampled layer feeds the depth-map measure.
"""
import os
from contextlib import nullcontext

import numpy as np

from fusion_methods import neural_common as common
from utils.image_utils import fuse_output_dtype

PAD_ALIGN = 32
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
        state = common.torch.load(path, map_location='cpu')
        state = {name.replace('module.', ''): value
                 for name, value in state.items()}
        model = StackMFF_V2()
        model.load_state_dict(state)
        model.to(device).eval()
        _CACHE = (key, model, device)
    return _CACHE[1], _CACHE[2]


def _prepare_inference(frames, model_path, use_gpu, thread_count=None,
                       resource_preference='balanced', execution_plan=None,
                       result_count=1, fixed_bytes=0):
    """Admit the checkpoint plus the encoder/GRU workspace for this tile."""
    h, w = frames[0].shape[:2]
    hp, wp = h + (-h) % PAD_ALIGN, w + (-w) % PAD_ALIGN
    n = len(frames)
    # 64-channel half-resolution encoder path and the 16-channel GRU are the
    # dominant terms; feature maps stay per layer, so they scale with N.
    workspace = int(96.0 * n * hp * wp * 4)
    host = int(n * hp * wp * 4 * 4 + result_count * h * w * 3 * 4 + fixed_bytes)
    device, plan = common.plan_inference(
        _ALGORITHM, frames, model_path, use_gpu, thread_count,
        resource_preference, execution_plan,
        workspace_bytes=workspace, host_bytes=host)
    model, device = _model_and_device(model_path, use_gpu)
    return model, device, plan


def _probabilities(planes, model, device, should_cancel=None):
    """[N, Hp, Wp] focus probabilities; sums to one along the stack."""
    common.cancel(should_cancel)
    tensor = planes.to(device)
    try:
        with common.torch.no_grad():
            features = model.feature_extraction(tensor)
            focus = model.layer_interaction(features)
            probs = common.torch.softmax(focus, dim=1)
        common.cancel(should_cancel)
    finally:
        if getattr(device, 'type', '') == 'cuda':
            common.torch.cuda.synchronize(device)
        elif getattr(device, 'type', '') == 'mps':
            common.torch.mps.synchronize()
    return probs[0].cpu().numpy()


def _depth_from(probs, n):
    """Reference depth: expected layer 1..N, min-max normalised to 0..1."""
    indices = np.arange(1, n + 1, dtype=np.float32)[:, None, None]
    expected = (probs * indices).sum(axis=0)
    low, high = float(expected.min()), float(expected.max())
    span = high - low
    if span <= 1e-12:
        # A flat tile has no focus variation; a constant map keeps every
        # caller deterministic instead of dividing by zero.
        return np.zeros(expected.shape, np.float32)
    return ((expected - low) / span).astype(np.float32)


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
    normalized = common.normalized_frames(frames)
    n = len(normalized)
    planes = common.stage_planes(normalized, PAD_ALIGN, should_cancel)
    model, device, _plan = _prepare_inference(
        normalized, model_path, use_gpu, thread_count=thread_count,
        resource_preference=resource_preference, execution_plan=execution_plan)
    probs = _probabilities(planes, model, device, should_cancel)
    del planes
    h, w = normalized[0].shape[:2]
    # The reference normalises the depth map over the *padded* inference and
    # crops afterwards. Normalising the crop instead shifts every index when
    # the edge padding changes the extrema, so keep the reference order.
    depth_full = _depth_from(probs, n)
    index_full = np.clip(np.round(depth_full * (n - 1)).astype(np.int64), 0, n - 1)
    depth = depth_full[:h, :w]
    index = index_full[:h, :w]
    fused = _sample(normalized, index, should_cancel)
    del probs, depth_full, index_full
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
