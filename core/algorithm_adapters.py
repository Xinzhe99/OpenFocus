"""Lazy backend translations and execution capability hooks."""
from math import ceil, log2


# Native compiler startup is resident state, even for a very small image.
DEPTH_KERNEL_STARTUP_BYTES = 256 * 1024 * 1024


def roi_guided(parameters, width, height):
    return max(96, 111 + parameters['kernel_size'] // 2), 1


def roi_dct(parameters, width, height):
    return max(96, 8 * (parameters['kernel_size'] + 1)), 8


def roi_dtcwt(parameters, width, height):
    grid = 2 ** parameters['N']
    return max(96, 16 * grid), grid


def roi_gfg(parameters, width, height):
    return max(96, parameters['kernel_size'] // 2 + 48), 4


def roi_neural(parameters, width, height):
    return 256, 128


def roi_openfocus_a(parameters, width, height):
    return max(96, parameters['kernel_size'] // 2 + ceil(4 * parameters['blend_smoothing']) + 2), 1


def roi_openfocus_b(parameters, width, height):
    return max(96, parameters['kernel_size'] // 2 + ceil(4 * parameters['depth_smoothing']) + 2), 1


def roi_openfocus_c(parameters, width, height):
    levels = max(2, min(7, int(log2(max(2, min(width, height)))) - 1))
    grid = 2 ** (levels - 1)
    return max(96, ceil((8 + 4 * parameters['blend_smoothing']) * grid)), grid


# OpenFocus retains fixed global maps/pyramids and a bounded set of independent
# source producers. Charge their live planes, not N float colour frames.
# These envelopes include mapped planes: fitting them in RAM avoids relying on
# paging for the preferred whole-image path. The fallback still uses mappings.
def openfocus_a_resources(height, width, parameters, frames):
    pixels = height * width
    # Shared output/best planes (16 B/px), independent frame/response producer
    # and completed result (80 B/px). Sum preserves the serial envelope.
    return 4 * 1024 * 1024 + pixels * 16, pixels * 80


def openfocus_b_resources(height, width, parameters, frames):
    pixels = height * width
    # Global selection/smoothing/reconstruction plus independent activity work.
    return 4 * 1024 * 1024 + pixels * 160, pixels * 80


def openfocus_c_resources(height, width, parameters, frames):
    pixels = height * width
    return 4 * 1024 * 1024 + pixels * 304, pixels * 80


def guided_resources(height, width, parameters, frames):
    # Four retained accumulators plus the decision map; one guided-filter
    # worker, including its ordered result, has a bounded float workspace.
    return height * width * 40, height * width * 192


def gfg_resources(height, width, parameters, frames):
    return height * width * 40, height * width * 128


def dct_resources(height, width, parameters, frames):
    # Include block padding and median-filter scratch at the coarser grid.
    return 4 * 1024 * 1024, (height + 8) * (width + 8) * 64


def _openfocus_needs_tiles(algorithm, frames, threshold):
    from core.multi_focus_fusion import (
        _DEFAULT_TILE_THRESHOLD, estimate_peak_bytes, memory_budget_bytes)
    if not frames:
        return False
    # A non-default threshold remains an explicit spatial cap (also useful for
    # reproducible tiled renders). The default chooses by actual RAM headroom.
    forced = threshold != _DEFAULT_TILE_THRESHOLD and max(frames[0].shape[:2]) > threshold
    return forced or estimate_peak_bytes(algorithm, frames) > memory_budget_bytes()


def openfocus_a_needs_tiles(frames, threshold):
    return _openfocus_needs_tiles('openfocus_a', frames, threshold)


def openfocus_b_needs_tiles(frames, threshold):
    return _openfocus_needs_tiles('openfocus_b', frames, threshold)


def openfocus_c_needs_tiles(frames, threshold):
    return _openfocus_needs_tiles('openfocus_c', frames, threshold)


def dtcwt_resources(height, width, parameters, frames):
    from fusion_methods.dtcwt import estimate_working_set
    return estimate_working_set(height, width, parameters['N'], frames)


def dtcwt_needs_tiles(frames, threshold):
    from core.multi_focus_fusion import estimate_peak_bytes, memory_budget_bytes
    # Prefer whole transforms to avoid introducing tile-boundary wavelets.
    return estimate_peak_bytes('dtcwt', frames) > memory_budget_bytes()


def neural_needs_tiles(frames, threshold):
    from core.multi_focus_fusion import _stack_footprint_exceeds
    return max(frames[0].shape[:2]) > threshold or _stack_footprint_exceeds(frames, threshold)


def registration_homography(registration, images, thread_count=4, should_cancel=None, **parameters):
    return registration._process_homography(images, thread_count=thread_count, should_cancel=should_cancel, failure_policy=parameters['failure_policy'])


def registration_ecc(registration, images, thread_count=4, should_cancel=None, **parameters):
    return registration._process_ecc(images, thread_count=thread_count, should_cancel=should_cancel, failure_policy=parameters['failure_policy'])


def registration_openfocus(registration, images, thread_count=4, should_cancel=None, **parameters):
    from core.registration import _align_openfocus_impl
    return _align_openfocus_impl(images, downscale_width=registration.downscale_width,
        thread_count=thread_count, should_cancel=should_cancel,
        max_shift_ratio=parameters['max_shift_ratio'], brightness_reference=registration.brightness_reference,
        failure_policy=parameters['failure_policy'], reference_frame=parameters['reference_frame'])


def depth_dct(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_dct
    return _measure_dct(images, progress_callback, should_cancel,
                        kernel_size=parameters['kernel_size'],
                        block_size=parameters.get('block_size', 8))


def depth_gfg(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_gfgfgf
    return _measure_gfgfgf(images, progress_callback, should_cancel, kernel_size=parameters['kernel_size'], thread_count=parameters.get('thread_count'))


def depth_dtcwt(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_dtcwt
    return _measure_dtcwt(images, progress_callback, should_cancel, nlevels=parameters['N'])


def depth_neural(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_stackmffv3
    return _measure_stackmffv3(images, progress_callback, should_cancel,
        model_path=parameters.get('model_path'), use_gpu=parameters.get('use_gpu', False),
        execution_plan=parameters.get('execution_plan'), mode=parameters['mode'])


def depth_stackmffv2(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_stackmffv2
    return _measure_stackmffv2(images, progress_callback, should_cancel,
        model_path=parameters.get('model_path'), use_gpu=parameters.get('use_gpu', False),
        execution_plan=parameters.get('execution_plan'),
        thread_count=parameters.get('thread_count'),
        resource_preference=parameters.get('resource_preference', 'balanced'))


def stackmffv2_needs_tiles(frames, threshold):
    """Tile when geometry or the attention working set exceeds its budget."""
    from core.multi_focus_fusion import _stack_footprint_exceeds
    from core.resource_policy import ResourceLimitError
    from fusion_methods.stackmff_v2 import _prepare_inference
    h, w = frames[0].shape[:2]
    if max(h, w) > threshold or _stack_footprint_exceeds(frames, threshold):
        return True
    try:
        _prepare_inference(frames, None, False)
    except ResourceLimitError:
        return True
    return False


def depth_openfocus_b(images, progress_callback=None, should_cancel=None, **parameters):
    from fusion_methods.openfocus_depth import depth_map
    smoothing = parameters['depth_smoothing'] * (5.0 / 1.2)
    return depth_map(images, None, parameters['kernel_size'], smoothing,
                     should_cancel, progress_callback)


def depth_openfocus_c(images, progress_callback=None, should_cancel=None, **parameters):
    from fusion_methods.openfocus_depth import depth_map
    smoothing = parameters['blend_smoothing'] * (5.0 / 3.0)
    return depth_map(images, None, 11, smoothing, should_cancel, progress_callback)
