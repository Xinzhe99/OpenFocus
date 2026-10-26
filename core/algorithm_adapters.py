"""Lazy backend translations and execution capability hooks."""
from math import ceil, log2


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


def dtcwt_resources(height, width, parameters, frames):
    from fusion_methods.dtcwt import estimate_working_set
    return estimate_working_set(height, width, parameters['N'], frames)


def dtcwt_needs_tiles(frames, threshold):
    from core.multi_focus_fusion import _dtcwt_activity_bytes, DTCWT_ACTIVITY_BUDGET
    return _dtcwt_activity_bytes(frames) > DTCWT_ACTIVITY_BUDGET


def neural_needs_tiles(frames, threshold):
    from core.multi_focus_fusion import _stack_footprint_exceeds
    return max(frames[0].shape[:2]) > threshold or _stack_footprint_exceeds(frames, threshold)


def registration_homography(registration, images, thread_count=4, should_cancel=None, **parameters):
    return registration._process_homography(images, thread_count=thread_count, should_cancel=should_cancel)


def registration_ecc(registration, images, thread_count=4, should_cancel=None, **parameters):
    return registration._process_ecc(images, thread_count=thread_count, should_cancel=should_cancel)


def registration_openfocus(registration, images, thread_count=4, should_cancel=None, **parameters):
    from core.registration import _align_openfocus_impl
    return _align_openfocus_impl(images, downscale_width=registration.downscale_width,
        thread_count=thread_count, should_cancel=should_cancel,
        max_shift_ratio=parameters['max_shift_ratio'], brightness_reference=registration.brightness_reference)


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


def depth_openfocus_b(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_openfocus
    return _measure_openfocus(images, progress_callback, should_cancel, kernel_size=parameters['kernel_size'])


def depth_openfocus_c(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_openfocus_c
    return _measure_openfocus_c(images, progress_callback, should_cancel, kernel_size=parameters['kernel_size'])
