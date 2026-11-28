# -*- coding: utf-8 -*-
"""StackMFF V4 (StreamFocusNet Flash) back-end: order-invariant streaming fusion.

Runs the single-file ``weights/stackmff_v4_flash.onnx`` model (or native CUDA/MPS
``weights/stackmff_v4_flash.pth`` via the isolated accelerator worker) at native
single scale ``(1.0,)`` with O(N)-compute, grid-artifact-free probability maps.
"""
from contextlib import nullcontext
import numpy as np

from utils.image_utils import fuse_output_dtype
from utils.performance_profile import current as _profile
from fusion_methods import neural_common as common
from fusion_methods.stackmffv3 import (
    _blend_tile,
    _cancel,
    _read_stack,
    _resolve_device,
    _stage_gray,
    _validate_tiles,
    normalize_mode,
)

DEFAULT_WEIGHTS = 'weights/stackmff_v4_flash.onnx'
PAD_ALIGN = common.ATTENTION_PAD_ALIGN
INFER_SCALES = (1.0,)
SOFT_SIGMA = 0.5
VALID_MODES = ("soft", "hard")


def _forward_probabilities(batch_input, model, device, sigma, should_cancel,
                           threads=1):
    from fusion_methods.neural_runtime import probabilities
    return probabilities(batch_input, model, device, threads, sigma,
                         should_cancel, scales=INFER_SCALES)


def _probabilities(batch_input, model, device, mode, should_cancel=None,
                   threads=1):
    sigma = 0.0 if normalize_mode(mode) == 'hard' else SOFT_SIGMA
    return _forward_probabilities(batch_input, model, device, sigma,
                                  should_cancel, threads)


def _prepare_inference(frames, model_path, use_gpu, thread_count=None,
                       resource_preference='balanced', execution_plan=None,
                       result_count=1, fixed_bytes=0):
    """Admit model loading and O(N) inference workspace for StackMFF V4."""
    device = _resolve_device(use_gpu)
    return common.plan_attention_inference(
        'stackmffv4', frames, model_path, device,
        thread_count=thread_count, resource_preference=resource_preference,
        execution_plan=execution_plan, result_count=result_count,
        fixed_bytes=fixed_bytes)


def _stackmffv4_batch_impl(tiles_list, model_path, use_gpu, mode='soft',
                           should_cancel=None, thread_count=None,
                           resource_preference='balanced', execution_plan=None,
                           focus_callback=None, *, tile_plan=None):
    """Fuse ordered tiles for StackMFF V4 using single-scale O(N) inference."""
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
        for frames in tiles_list:
            _cancel(should_cancel)
            is_16bit = any(frame.dtype == np.uint16 for frame in frames)
            with profile.stage('stage_gray'):
                batch_input = _stage_gray(frames, should_cancel, threads=plan.native_threads)
            with profile.stage('probabilities'):
                probs = _probabilities(batch_input, model, device, mode, should_cancel,
                                       plan.native_threads)
            del batch_input
            h, w = frames[0].shape[:2]
            grayscale = frames[0].ndim == 2
            with profile.stage('blend_tile'):
                fused = _blend_tile(probs[0, :, :h, :w], frames, mode, should_cancel,
                                    focus_callback, threads=plan.native_threads)
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


def _stackmffv4_impl(input_source, img_resize, model_path, use_gpu,
                     mode='soft', progress_callback=None, should_cancel=None,
                     thread_count=None, resource_preference='balanced',
                     execution_plan=None, focus_callback=None, *, tile_plan=None):
    """Whole-frame StackMFF V4 fusion with the same resource contract as tiled fusion."""
    from core.resource_policy import heavy_job

    lease = heavy_job(should_cancel) if execution_plan is None and tile_plan is None else nullcontext()
    with lease:
        _cancel(should_cancel)
        frames = _read_stack(input_source, img_resize, should_cancel, execution_plan)
        if len(frames) < 2:
            raise ValueError("At least two images are required for fusion")
        if progress_callback is not None:
            progress_callback(0, 1)
        out = _stackmffv4_batch_impl(
            [frames], model_path, use_gpu, mode,
            should_cancel=should_cancel, thread_count=thread_count,
            resource_preference=resource_preference, execution_plan=execution_plan,
            focus_callback=focus_callback, tile_plan=tile_plan)[0]
        _cancel(should_cancel)
        if progress_callback is not None:
            progress_callback(1, 1)
        _cancel(should_cancel)
        return out


def model_probabilities(frames, model_path, use_gpu, sigma=0.0,
                        should_cancel=None, thread_count=None,
                        resource_preference='balanced', execution_plan=None):
    """Per-pixel [N, H, W] probabilities from StackMFF V4 at native single scale."""
    with common.neural_lease(execution_plan, should_cancel):
        _cancel(should_cancel)
        _validate_tiles([frames])
        model, device, plan = _prepare_inference(
            frames, model_path, use_gpu, thread_count,
            resource_preference, execution_plan)
        batch = _stage_gray(frames, should_cancel, threads=plan.native_threads)
        probs = _forward_probabilities(batch, model, device, sigma,
                                       should_cancel, plan.native_threads)
        h, w = frames[0].shape[:2]
        result = probs[0, :, :h, :w]
        _cancel(should_cancel)
        return result
