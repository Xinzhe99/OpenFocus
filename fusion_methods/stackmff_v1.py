"""StackMFF back-end using its retrained attention backbone.

The network fuses stack luma; colour is reconstructed from the closest source.
The model has no user-tunable parameters.
"""

import numpy as np

from fusion_methods import neural_common as common
from utils.image_utils import fuse_output_dtype

DEFAULT_WEIGHTS = 'weights/stackmff_final.onnx'
INFER_SIGMA = 0.5
INFER_SCALES = (1.0, 0.75)
_ALGORITHM = 'stackmff'


def _prepare_inference(frames, model_path, use_gpu, thread_count=None,
                       resource_preference='balanced', execution_plan=None,
                       result_count=1, fixed_bytes=0):
    """Admit the retrained model and its attention workspace."""
    from fusion_methods import neural_runtime
    device = neural_runtime.resolve_job_device(use_gpu)
    return common.plan_attention_inference(
        _ALGORITHM, frames, model_path, device,
        thread_count=thread_count, resource_preference=resource_preference,
        execution_plan=execution_plan, result_count=result_count,
        fixed_bytes=fixed_bytes)


def _fuse_tile(frames, model_path, use_gpu, should_cancel=None,
               execution_plan=None, thread_count=None,
               resource_preference='balanced', *, tile_plan=None):
    """Fuse one tile; returns 0-255 pixels in the caller's dtype."""
    common.cancel(should_cancel)
    out_dtype = np.float32 if any(f.dtype == np.uint16 for f in frames) else fuse_output_dtype(frames)
    if tile_plan is None:
        model, device, plan = _prepare_inference(
            frames, model_path, use_gpu, thread_count=thread_count,
            resource_preference=resource_preference, execution_plan=execution_plan)
    else:
        model, device, plan = model_path, tile_plan.device_kind, tile_plan
    # The backbone pads internally. Keep the original dimensions here so the
    # second inference scale matches the released benchmark preprocessing.
    planes = common.stage_planes(frames, 1, should_cancel)
    from fusion_methods import neural_runtime
    probs = neural_runtime.probabilities(planes, model, device, plan.native_threads,
                                         INFER_SIGMA, should_cancel)
    # The staged planes belong to this tile and inference has finished reading them.
    np.multiply(probs, planes, out=planes)
    del probs
    luma = planes.sum(axis=1)[0]
    del planes
    np.clip(luma, 0, 1, out=luma)
    luma *= 255.0
    fused = common.tensor_to_bgr(luma, frames, should_cancel)
    if out_dtype is np.uint8:
        np.clip(np.round(fused), 0, 255, out=fused)
        return fused.astype(np.uint8)
    return np.clip(fused, 0.0, 255.0, out=fused)


def stackmff_impl(input_source, img_resize, model_path, use_gpu,
                  should_cancel=None, progress_callback=None,
                  thread_count=None, resource_preference='balanced',
                  execution_plan=None, *, tile_plan=None):
    """Whole-stack fusion used by both the direct and tiled paths."""
    with common.neural_lease(tile_plan or execution_plan, should_cancel):
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
            resource_preference=resource_preference, tile_plan=tile_plan)
        common.cancel(should_cancel)
        if progress_callback is not None:
            progress_callback(1, 1)
        common.cancel(should_cancel)
        return result
