# -*- coding: utf-8 -*-
"""StackMFF V3 back-end: pixel-wise source classification over the stack.

The network predicts, for every pixel, a probability distribution over the N
input frames. Two output modes consume that distribution:

* ``soft`` — probability-weighted blend of the observed frame values, with
  light spatial smoothing of the probabilities.
* ``hard`` — the value of the most probable frame (argmax), without smoothing.

Inference runs at the working size by default. Dual-scale inference averages
the probabilities from the working size and 75% of that size for either mode.

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

SOFT_SIGMA = 0.5

VALID_MODES = ("soft", "hard")


def _resolve_device(use_gpu):
    from fusion_methods.neural_runtime import resolve_job_device
    return resolve_job_device(use_gpu)


def normalize_mode(mode):
    """Coerce a UI/CLI value to a supported output mode."""
    mode = str(mode or 'soft').strip().lower()
    return mode if mode in VALID_MODES else 'soft'




def _cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def _forward_probabilities(batch_input, model, device, sigma, should_cancel,
                           threads, inference_scale):
    from fusion_methods.neural_runtime import probabilities
    return probabilities(batch_input, model, device, threads, sigma, should_cancel,
                         dual_scale=inference_scale == 'dual')


def _probabilities(batch_input, model, device, mode, should_cancel, threads,
                   inference_scale):
    return _forward_probabilities(batch_input, model, device,
                                  0.0 if mode == 'hard' else SOFT_SIGMA,
                                  should_cancel, threads, inference_scale)


def _blend_tile(probs, frames, mode, should_cancel=None, focus_callback=None):
    """Combine probabilities with one source frame at a time in 0-255."""
    _cancel(should_cancel)
    height, width = frames[0].shape[:2]
    channels = frames[0].shape[2] if frames[0].ndim == 3 else 1
    result = np.zeros((height, width, channels), dtype=np.float32)
    normalize_stack = any(frame.dtype == np.uint16 for frame in frames)
    selected_indices = None
    expected = None
    if mode == 'hard':
        from core.frame_storage import create_frame
        selected_indices = create_frame((height, width), np.intp)
        # NumPy packs strided argmax input. Bound that copy by stack depth.
        row_bytes = len(frames) * width * probs.dtype.itemsize
        scan_rows = max(1, min(height, (8 * 1024 ** 2) // row_bytes))
        for row in range(0, height, scan_rows):
            _cancel(should_cancel)
            stop = min(row + scan_rows, height)
            np.argmax(probs[:, row:stop], axis=0, out=selected_indices[row:stop])
    elif focus_callback is not None:
        from core.frame_storage import create_transient_frame
        expected = create_transient_frame((height, width), np.float32)
        expected[...] = 0
    from core.source_frames import resolve_frame, validate_frame_source
    kernels = common.load_stackmff_kernels()
    band_rows = min(height, 256, max(1, (64 * 1024) // (width * channels)))
    selected_counts = None
    if selected_indices is None:
        source_dtypes = {np.float32} if normalize_stack else {frame.dtype for frame in frames}
        scratch = None
    else:
        selected_counts = np.bincount(selected_indices.ravel(), minlength=len(frames))
    for index, frame in enumerate(frames):
        _cancel(should_cancel)
        if selected_counts is not None and not selected_counts[index]:
            validate_frame_source(frame)
            continue
        image = resolve_frame(frame, should_cancel=should_cancel)
        blend = None
        if (kernels is not None and result.flags.c_contiguous and image.flags.c_contiguous
                and ((image.ndim == 3 and image.shape[2] == channels)
                     or (image.ndim == 2 and channels == 1))):
            if selected_indices is None:
                if image.dtype == np.float32:
                    blend = kernels.v3_accumulate_soft_f32
                elif image.dtype == np.uint16:
                    blend = kernels.v3_accumulate_soft_u16
                elif image.dtype == np.uint8:
                    blend = kernels.v3_accumulate_soft_u8
            elif image.dtype == np.float32:
                blend = kernels.v3_scatter_hard_f32
            elif image.dtype == np.uint16:
                blend = kernels.v3_scatter_hard_u16
            elif image.dtype == np.uint8:
                blend = kernels.v3_scatter_hard_u8
        for row in range(0, height, band_rows):
            _cancel(should_cancel)
            stop = min(row + band_rows, height)
            output = result[row:stop]
            img_band = image[row:stop]
            if blend is not None:
                src_band = img_band[:, :, None] if img_band.ndim == 2 else img_band
                try:
                    if selected_indices is None:
                        exp_band = expected[row:stop] if expected is not None else output[:, :, 0]
                        blend(src_band, np.ascontiguousarray(probs[index, row:stop]), output,
                              exp_band, float(index), bool(expected is not None and index))
                    else:
                        blend(src_band, np.ascontiguousarray(selected_indices[row:stop]), index, output)
                except common.StackMFFKernelUnavailable:
                    kernels = blend = None
                else:
                    continue
                finally:
                    del src_band
            source = common.working_frame(img_band, normalize_stack)
            if source.ndim == 2:
                source = source[:, :, None]
            if selected_indices is None:
                if scratch is None:
                    scratch = np.empty((band_rows, width, channels),
                                       dtype=np.result_type(probs.dtype, np.float32, *source_dtypes))
                probability = probs[index, row:stop]
                product = scratch[:stop - row]
                np.multiply(probability[:, :, None], source, out=product)
                np.add(output, product, out=output)
                if expected is not None and index:
                    depth_product = product[:, :, 0]
                    np.multiply(probability, np.float32(index), out=depth_product)
                    np.add(expected[row:stop], depth_product, out=expected[row:stop])
            else:
                np.copyto(output, source,
                          where=(selected_indices[row:stop] == index)[:, :, None])
            del source
        del img_band, image
    _cancel(should_cancel)
    if focus_callback is not None:
        focus_callback(selected_indices if expected is None else expected, len(frames) - 1)
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
        common.write_luma_plane(common.working_frame(frame, normalize_stack), batch[0, f, :h, :w])
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
                          resource_preference='balanced', execution_plan=None,
                          focus_callback=None, *, tile_plan=None, inference_scale='single'):
    """Fuse tiles in order with B=1 until true batching passes GPU gates.

    The spatial scheduler supplies a tile plan covering the complete grid.
    Other callers still admit each batch before inference.
    """
    with common.neural_lease(tile_plan or execution_plan, should_cancel):
        _cancel(should_cancel)
        if not tiles_list:
            return []
        _validate_tiles(tiles_list)
        mode = normalize_mode(mode)
        profile = _profile()
        if tile_plan is None:
            with profile.stage('plan_attention_inference'):
                model, device, plan = _prepare_inference(
                    tiles_list[0], model_path, use_gpu, thread_count,
                    resource_preference, execution_plan, len(tiles_list))
            if profile.enabled:
                profile.add('threads', 'plan_attention_inference', plan.native_threads)
        else:
            model, device, plan = model_path, tile_plan.device_kind, tile_plan
        results = []
        for tile in tiles_list:
            _cancel(should_cancel)
            frames = tile
            is_16bit = any(frame.dtype == np.uint16 for frame in frames)
            with profile.stage('stage_gray'):
                batch_input = _stage_gray(frames, should_cancel)
            with profile.stage('probabilities'):
                probs = _probabilities(batch_input, model, device, mode, should_cancel,
                                       plan.native_threads, inference_scale)
            del batch_input
            h, w = frames[0].shape[:2]
            # Color mixing does not need an N-frame color stack.
            grayscale = frames[0].ndim == 2
            with profile.stage('blend_tile'):
                fused = _blend_tile(probs[0, :, :h, :w], frames, mode, should_cancel,
                                   focus_callback)
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
                     execution_plan=None, focus_callback=None, *, tile_plan=None,
                     inference_scale='single'):
    """Whole-frame fusion with the same resource contract as tiled fusion."""
    from core.resource_policy import heavy_job

    lease = heavy_job(should_cancel) if execution_plan is None and tile_plan is None else nullcontext()
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
            resource_preference=resource_preference, execution_plan=execution_plan,
            focus_callback=focus_callback, tile_plan=tile_plan,
            inference_scale=inference_scale)[0]
        _cancel(should_cancel)
        if progress_callback is not None:
            progress_callback(1, 1)
        _cancel(should_cancel)
        return out


def model_probabilities(frames, model_path, use_gpu, sigma=0.0,
                        should_cancel=None, thread_count=None,
                        resource_preference='balanced', execution_plan=None, *,
                        inference_scale='single'):
    """Per-pixel [N,H,W] probabilities with the selected fusion inference scale."""
    with common.neural_lease(execution_plan, should_cancel):
        _cancel(should_cancel)
        _validate_tiles([frames])
        model, device, plan = _prepare_inference(
            frames, model_path, use_gpu, thread_count,
            resource_preference, execution_plan)
        batch = _stage_gray(frames, should_cancel)
        probabilities = _forward_probabilities(batch, model, device, sigma,
                                               should_cancel, plan.native_threads, inference_scale)
        h, w = frames[0].shape[:2]
        result = probabilities[0, :, :h, :w]
        _cancel(should_cancel)
        return result
