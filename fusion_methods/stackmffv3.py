# -*- coding: utf-8 -*-
"""StackMFF V3 back-end: pixel-wise source classification over the stack.

The network predicts, for every pixel, a probability distribution over the N
input frames. Two output modes consume that distribution:

* ``soft`` — probability-weighted blend of the observed frame values. This is
  the paper's default: probabilities are averaged over the inference scales
  and lightly smoothed, then used as weights.
* ``hard`` — the value of the most probable frame (argmax), a pure pixel pick
  with no interpolation between frames. Same scales, no smoothing (the
  reference implementation's ``--hard``).

The network sees luma only; the resulting per-pixel weights or indices are
applied to the three colour channels, exactly as in the reference code.

The exported attention model is shipped in ``weights/``.
"""
from contextlib import nullcontext
import cv2
import numpy as np

from utils.image_utils import fuse_output_dtype
from utils.performance_profile import current as _profile
from fusion_methods import neural_common as common


# 网络以 4x4 卷积下采样 4 倍，模型内部把输入 pad 到 64 的倍数；这里用同一
# 对齐量预处理批内瓦片，让同批张量形状一致。
PAD_ALIGN = common.ATTENTION_PAD_ALIGN

# 推理尺度（论文配置）：soft 在全分辨率与 0.75 倍上平均概率；hard 用同一
# 组尺度、不做平滑，然后取 argmax。
INFER_SCALES = (1.0, 0.75)
SOFT_SIGMA = 0.5

VALID_MODES = ("soft", "hard")


def _resolve_device(use_gpu):
    from fusion_methods.neural_runtime import resolve_device
    return resolve_device(use_gpu)


def normalize_mode(mode):
    """Coerce a UI/CLI value to a supported output mode."""
    mode = str(mode or 'soft').strip().lower()
    return mode if mode in VALID_MODES else 'soft'




def _cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def _gray01(bgr):
    """Luma in [0, 1] (BT.601, as cv2.COLOR_BGR2GRAY) for one frame."""
    gray = (bgr if bgr.ndim == 2 else bgr[:, :, 0]
            if bgr.shape[2] == 1 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    maxv = 65535.0 if gray.dtype == np.uint16 else 255.0
    return gray.astype(np.float32) / maxv


def _forward_probabilities(batch_input, model, device, sigma, should_cancel, threads=1):
    from fusion_methods.neural_runtime import probabilities
    return probabilities(batch_input, model, device, threads, sigma, should_cancel)


def _probabilities(batch_input, model, device, mode, should_cancel=None, threads=1):
    return _forward_probabilities(batch_input, model, device,
                                  0.0 if mode == 'hard' else SOFT_SIGMA,
                                  should_cancel, threads)


def _blend_tile(probs, frames, mode, should_cancel=None):
    """Combine probabilities with one source frame at a time in 0-255."""
    _cancel(should_cancel)
    height, width = frames[0].shape[:2]
    channels = frames[0].shape[2] if frames[0].ndim == 3 else 1
    result = np.zeros((height, width, channels), dtype=np.float32)
    normalize_stack = any(frame.dtype == np.uint16 for frame in frames)
    selected_indices = None
    if mode == 'hard':
        from core.frame_storage import create_frame
        selected_indices = create_frame((height, width), np.intp)
        for row in range(0, height, 256):
            _cancel(should_cancel)
            stop = min(row + 256, height)
            selected_indices[row:stop] = probs[:, row:stop].argmax(0)
    from core.source_frames import resolve_frame
    for index, frame in enumerate(frames):
        _cancel(should_cancel)
        image = resolve_frame(frame)
        for row in range(0, height, 256):
            _cancel(should_cancel)
            stop = min(row + 256, height)
            source = common.working_frame(image[row:stop], normalize_stack)
            if source.ndim == 2:
                source = source[:, :, None]
            output = result[row:stop]
            if selected_indices is None:
                output += probs[index, row:stop, :, None] * source
            else:
                np.copyto(output, source,
                          where=(selected_indices[row:stop] == index)[:, :, None])
            del source
        del image
    _cancel(should_cancel)
    return result


def _validate_tiles(tiles_list):
    first = tiles_list[0]
    if not first:
        raise ValueError("Empty image list")
    shape = first[0].shape
    if len(shape) not in (2, 3) or (len(shape) == 3 and shape[2] not in (1, 3)):
        raise ValueError("Expected grayscale or BGR frames")
    if min(shape[:2]) <= 0:
        raise ValueError("Empty image dimensions")
    for tile in tiles_list:
        if len(tile) != len(first):
            raise ValueError("Inconsistent number of images in tile batch")
        if any(frame.shape != shape for frame in tile):
            raise ValueError("Inconsistent frame shapes in tile batch")


def _stage_gray(frames, should_cancel):
    h, w = frames[0].shape[:2]
    hp, wp = h + (-h) % PAD_ALIGN, w + (-w) % PAD_ALIGN
    batch = np.empty((1, len(frames), hp, wp), dtype=np.float32)
    normalize_stack = any(frame.dtype == np.uint16 for frame in frames)
    for f, frame in enumerate(frames):
        _cancel(should_cancel)
        batch[0, f, :h, :w] = _gray01(common.working_frame(frame, normalize_stack))
    if wp != w:
        batch[:, :, :h, w:] = batch[:, :, :h, w - 1:w]
    if hp != h:
        batch[:, :, h:, :] = batch[:, :, h - 1:h, :]
    return batch


def _prepare_inference(frames, model_path, use_gpu, thread_count=None,
                       resource_preference='balanced', execution_plan=None,
                       result_count=1, fixed_bytes=0):
    """Admit model loading and the attention workspace for this stack."""
    device = _resolve_device(use_gpu)
    return common.plan_attention_inference(
        'stackmffv3', frames, model_path, device,
        thread_count=thread_count, resource_preference=resource_preference,
        execution_plan=execution_plan, result_count=result_count,
        fixed_bytes=fixed_bytes)


def _stackmffv3_batch_impl(tiles_list, model_path, use_gpu, mode='soft',
                          should_cancel=None, thread_count=None,
                          resource_preference='balanced', execution_plan=None):
    """Fuse tiles in order with B=1 until true batching passes GPU gates.

    A supplied plan means the caller owns the heavy-job lease, including when
    dispatching to a child thread. Actual inference allocations are rechecked.
    """
    from core.resource_policy import heavy_job

    lease = heavy_job(should_cancel) if execution_plan is None else nullcontext()
    with lease:
        if execution_plan is None:
            from fusion_methods.neural_runtime import start_inference_job
            start_inference_job()
        _cancel(should_cancel)
        if not tiles_list:
            return []
        _validate_tiles(tiles_list)
        mode = normalize_mode(mode)
        profile = _profile()
        with profile.stage('plan_attention_inference'):
            model, device, plan = _prepare_inference(
                tiles_list[0], model_path, use_gpu, thread_count,
                resource_preference, execution_plan, len(tiles_list))
        if profile.enabled:
            profile.add('threads', 'plan_attention_inference', plan.native_threads)
        results = []
        for tile in tiles_list:
            _cancel(should_cancel)
            frames = tile
            is_16bit = any(frame.dtype == np.uint16 for frame in frames)
            with profile.stage('stage_gray'):
                batch_input = _stage_gray(frames, should_cancel)
            with profile.stage('probabilities'):
                probs = _probabilities(batch_input, model, device, mode, should_cancel,
                                       plan.native_threads)
            del batch_input
            h, w = frames[0].shape[:2]
            # Color mixing does not need an N-frame color stack.
            grayscale = frames[0].ndim == 2
            with profile.stage('blend_tile'):
                fused = _blend_tile(probs[0, :, :h, :w], frames, mode, should_cancel)
            del probs
            if not is_16bit and fuse_output_dtype(frames) is np.uint8:
                fused = np.clip(np.round(fused), 0, 255).astype(np.uint8)
            else:
                np.clip(fused, 0.0, 255.0, out=fused)
            if grayscale:
                fused = fused[..., 0]
            results.append(fused)
        _cancel(should_cancel)
        return results


def _read_stack(input_source, img_resize, should_cancel=None, execution_plan=None):
    """Capture source descriptors using shared cancellable per-frame processing."""
    return common.read_stack(input_source, img_resize, should_cancel, execution_plan)


def _stackmffv3_impl(input_source, img_resize, model_path, use_gpu,
                     mode='soft', progress_callback=None, should_cancel=None,
                     thread_count=None, resource_preference='balanced',
                     execution_plan=None):
    """Whole-frame fusion with the same resource contract as tiled fusion."""
    from core.resource_policy import heavy_job

    lease = heavy_job(should_cancel) if execution_plan is None else nullcontext()
    with lease:
        _cancel(should_cancel)
        frames = _read_stack(input_source, img_resize, should_cancel, execution_plan)
        if len(frames) < 2:
            raise ValueError("At least two images are required for fusion")
        if progress_callback is not None:
            progress_callback(0, 1)
        out = _stackmffv3_batch_impl(
            [frames], model_path, use_gpu, mode,
            should_cancel=should_cancel, thread_count=thread_count,
            resource_preference=resource_preference, execution_plan=execution_plan)[0]
        _cancel(should_cancel)
        if progress_callback is not None:
            progress_callback(1, 1)
        _cancel(should_cancel)
        return out


def model_probabilities(frames, model_path, use_gpu, sigma=0.0,
                        should_cancel=None, thread_count=None,
                        resource_preference='balanced', execution_plan=None):
    """Per-pixel [N,H,W] probabilities, using fusion's single replicated pad."""
    from core.resource_policy import heavy_job

    lease = heavy_job(should_cancel) if execution_plan is None else nullcontext()
    with lease:
        if execution_plan is None:
            from fusion_methods.neural_runtime import start_inference_job
            start_inference_job()
        _cancel(should_cancel)
        _validate_tiles([frames])
        model, device, plan = _prepare_inference(
            frames, model_path, use_gpu, thread_count,
            resource_preference, execution_plan)
        batch = _stage_gray(frames, should_cancel)
        probabilities = _forward_probabilities(batch, model, device, sigma,
                                               should_cancel, plan.native_threads)
        h, w = frames[0].shape[:2]
        result = probabilities[0, :, :h, :w]
        _cancel(should_cancel)
        return result
