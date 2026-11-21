"""StackMFF V2 back-end using its retrained attention backbone.

Frame order defines depth positions. The expected layer is min-max normalised
over the padded inference, then the nearest source frame is sampled.
"""

import numpy as np

from fusion_methods import neural_common as common
from utils.image_utils import apply_exposure_gain, fuse_output_dtype

PAD_ALIGN = common.ATTENTION_PAD_ALIGN
DEFAULT_WEIGHTS = 'weights/stackmff_v2_final.onnx'
_ALGORITHM = 'stackmffv2'
_STRIP_ROWS = 128


def _prepare_inference(frames, model_path, use_gpu, thread_count=None,
                       resource_preference='balanced', execution_plan=None,
                       result_count=1, fixed_bytes=0, height=None, width=None, device=None):
    """Admit the retrained model and its attention workspace."""
    from fusion_methods import neural_runtime
    device = (device if device is not None and use_gpu
              else neural_runtime.resolve_job_device(use_gpu))
    return common.plan_attention_inference(
        _ALGORITHM, frames, model_path, device,
        thread_count=thread_count, resource_preference=resource_preference,
        execution_plan=execution_plan, result_count=result_count,
        fixed_bytes=fixed_bytes, height=height, width=width)


def _expected_positions(logits, should_cancel):
    """Reuse private logits for softmax and the ordered layer reduction."""
    from fusion_methods.neural_runtime import softmax_inplace

    common.cancel(should_cancel)
    probs = softmax_inplace(logits)
    weights = np.arange(1, probs.shape[1] + 1, dtype=np.float32)[None, :, None, None]
    np.multiply(probs, weights, out=probs)
    common.cancel(should_cancel)
    return probs.sum(axis=1)[0]


def _whole_depth(frames, model, device, threads, should_cancel):
    from fusion_methods.neural_runtime import logits

    planes = common.stage_planes(frames, PAD_ALIGN, should_cancel)
    scores = logits(planes, model, device, threads, should_cancel)
    del planes
    depth = _expected_positions(scores, should_cancel)
    del scores
    n = len(frames)
    lo, hi = depth.min(), depth.max()
    depth -= lo
    depth /= hi - lo + 1e-8
    index = np.clip(np.rint(depth * (n - 1)), 0, n - 1).astype(np.int64)
    h, w = frames[0].shape[:2]
    return depth[:h, :w], index[:h, :w]




def _sample(frames, index, should_cancel=None, *, dtype=np.float32, round_uint8=False,
            gains=None, normalize_for_exposure=False, original_frames=None):
    """Read each contributing frame once and only visit its selected row strips."""
    from core.frame_storage import create_frame, frame_region
    from core.source_frames import resolve_frame, validate_frame_source

    shape = frames[0].shape
    strips = [[] for _ in frames]
    count = len(frames)
    for start in range(0, index.shape[0], _STRIP_ROWS):
        common.cancel(should_cancel)
        values = index[start:start + _STRIP_ROWS].ravel()
        # A non-finite depth can produce an invalid index. Leave those pixels
        # at their initial value just as the full mask scan did.
        if values.min() < 0 or values.max() >= count:
            values = values[(values >= 0) & (values < count)]
        for layer in np.flatnonzero(np.bincount(values, minlength=count)):
            strips[layer].append(start)
        del values

    out = create_frame(shape, dtype)
    for layer, frame in enumerate(frames):
        common.cancel(should_cancel)
        if frame.shape[:2] != shape[:2]:
            raise ValueError('All frames must have the same dimensions')
        if len(frame.shape) != len(shape) or (len(shape) == 3 and frame.shape[2] not in (1, shape[2])):
            raise ValueError('All frames must have compatible channel counts')
        if original_frames is not None:
            validate_frame_source(original_frames[layer])
        if not strips[layer]:
            validate_frame_source(frame)
            continue
        image = resolve_frame(frame, should_cancel=should_cancel)
        gain = gains[layer] if gains is not None else 1.0
        for start in strips[layer]:
            common.cancel(should_cancel)
            rows = slice(start, start + _STRIP_ROWS)
            mask = index[rows] == layer
            with frame_region(image, rows) as pixels:
                if (gain == 1.0 and pixels.dtype != np.uint16
                        and (not round_uint8 or pixels.dtype == np.uint8)):
                    np.copyto(out[rows], pixels, where=mask[..., None] if pixels.ndim == 3 else mask,
                              casting='unsafe')
                    continue
                selected = pixels[mask]
                if gain != 1.0:
                    if normalize_for_exposure:
                        selected = common.working_frame(selected, True)
                    # Preserve the channel axis for exposure handling of alpha.
                    selected = apply_exposure_gain(selected[None], gain)[0]
                if selected.dtype == np.uint16:
                    selected = common.working_frame(selected, True)
                if round_uint8:
                    np.rint(selected, out=selected)
                    np.clip(selected, 0, 255, out=selected)
                out[rows][mask] = selected
                del selected
        del image
    if out.dtype != np.uint8:
        for start in range(0, out.shape[0], _STRIP_ROWS):
            common.cancel(should_cancel)
            strip = out[start:start + _STRIP_ROWS]
            np.clip(strip, 0.0, 255.0, out=strip)
    common.cancel(should_cancel)
    return out[..., 0] if out.ndim == 3 and out.shape[-1] == 1 else out


def infer(frames, model_path, use_gpu, should_cancel=None,
          execution_plan=None, thread_count=None,
          resource_preference='balanced'):
    """Return (fused pixels, depth 0-1, index 0..N-1) for one whole tile."""
    with common.neural_lease(execution_plan, should_cancel):
        return _infer_under_lease(frames, model_path, use_gpu, should_cancel,
                                  execution_plan, thread_count, resource_preference)


def _infer_under_lease(frames, model_path, use_gpu, should_cancel=None,
                       execution_plan=None, thread_count=None,
                       resource_preference='balanced'):
    common.cancel(should_cancel)
    out_dtype = np.float32 if any(f.dtype == np.uint16 for f in frames) else fuse_output_dtype(frames)
    model, device, plan = _prepare_inference(
        frames, model_path, use_gpu, thread_count=thread_count,
        resource_preference=resource_preference, execution_plan=execution_plan)
    depth, index = _whole_depth(frames, model, device, plan.native_threads, should_cancel)
    fused = _sample(frames, index, should_cancel, dtype=out_dtype,
                    round_uint8=out_dtype is np.uint8)
    return fused, depth, index


def stackmffv2_impl(input_source, img_resize, model_path, use_gpu,
                    should_cancel=None, progress_callback=None,
                    thread_count=None, resource_preference='balanced',
                    execution_plan=None, tile_size=None, tile_overlap=0,
                    exposure_gains=None, source_all_uint8=None, focus_callback=None,
                    tile_grid_callback=None, **kwargs):
    """Whole-stack fusion; the caller may already hold the heavy-job lease."""
    with common.neural_lease(execution_plan, should_cancel):
        common.cancel(should_cancel)
        if tile_size is not None:
            from fusion_methods.stackmff_v2_tiled import _tiled_fusion_under_lease
            return _tiled_fusion_under_lease(
                input_source, img_resize, model_path, use_gpu, should_cancel,
                execution_plan, thread_count, resource_preference, tile_size,
                tile_overlap, progress_callback, exposure_gains, source_all_uint8,
                focus_callback, tile_grid_callback)
        frames = common.read_stack(input_source, img_resize, should_cancel,
                                   execution_plan, preserve_grayscale=False)
        if len(frames) < 2:
            raise ValueError("At least two images are required for fusion")
        if progress_callback is not None:
            progress_callback(0, 1)
        fused, _depth, _index = _infer_under_lease(
            frames, model_path, use_gpu,
            should_cancel=should_cancel, execution_plan=execution_plan,
            thread_count=thread_count,
            resource_preference=resource_preference)
        if progress_callback is not None:
            progress_callback(1, 1)
        common.cancel(should_cancel)
        if focus_callback is not None:
            focus_callback(_index, len(frames) - 1)
        return fused


def depth_positions(images, model_path, use_gpu, should_cancel=None,
                    execution_plan=None, thread_count=None,
                    resource_preference='balanced', tile_size=None,
                    tile_overlap=0, progress_callback=None):
    """Sampled layer positions, with adaptive whole-stack spatial tiling."""
    with common.neural_lease(execution_plan, should_cancel):
        return _depth_positions_under_lease(
            images, model_path, use_gpu, should_cancel, execution_plan,
            thread_count, resource_preference, tile_size, tile_overlap, progress_callback)


def _depth_positions_under_lease(images, model_path, use_gpu, should_cancel=None,
                                 execution_plan=None, thread_count=None,
                                 resource_preference='balanced', tile_size=None,
                                 tile_overlap=0, progress_callback=None):
    from core.resource_policy import ResourceLimitError
    from fusion_methods.stackmff_v2_tiled import _Source, _tiled_depth_under_lease
    images = list(images)
    if tile_size is None:
        try:
            model, device, plan = _prepare_inference(
                images, model_path, use_gpu, thread_count=thread_count,
                resource_preference=resource_preference, execution_plan=execution_plan)
        except ResourceLimitError:
            pass
        else:
            if progress_callback is not None:
                progress_callback(0, 1)
            _depth, index = _whole_depth(images, model, device, plan.native_threads, should_cancel)
            if progress_callback is not None:
                progress_callback(1, 1)
            return index.astype(np.float32)
    source = _Source(images, None, None, should_cancel)
    _depth, index = _tiled_depth_under_lease(
        source, model_path, use_gpu, should_cancel, execution_plan,
        thread_count, resource_preference, tile_size, tile_overlap,
        progress_callback)
    return index.astype(np.float32)
