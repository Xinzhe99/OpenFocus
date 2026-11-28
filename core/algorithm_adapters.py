"""Lazy backend translations and execution capability hooks."""
from math import ceil


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
    return max(96, parameters['radius'] + 2), 1


def roi_openfocus_b(parameters, width, height):
    return max(96, parameters['radius'] + ceil(4 * parameters['smoothing'] * .24) + 2), 1


def roi_openfocus_c(parameters, width, height):
    # Global depth recurrences require the full context. Half-scale gray
    # sampling uses a two-pixel grid; C does not reconstruct RGB pyramids.
    return max(width, height), 2


# Serial OpenFocus passes retain global planes in mappings and use bounded
# row scratch. A's background accumulators spill to mapped row windows when
# they exceed 32 MiB. Concurrent producers are admitted separately.
def openfocus_a_resources(height, width, parameters, frames):
    from core._native import native_available

    native = native_available()
    background_ram = min(height * width * 24, 32 * 1024 * 1024)
    fixed = (4 * 1024 * 1024 + background_ram
             + min(height, 128) * width * (64 if native else 120))
    return fixed, height * width * (12 if native else 16)


def openfocus_b_resources(height, width, parameters, frames):
    from core._native import native_available

    native = native_available()
    fixed = (4 * 1024 * 1024
             + min(height, 128) * width * (64 if native else 160))
    half_planes = 6 * ((height + 1) // 2) * ((width + 1) // 2) * 2
    return fixed, max(height * width * 8, half_planes)


def openfocus_c_resources(height, width, parameters, frames):
    from core._native import native_available

    native = native_available()
    fixed = (4 * 1024 * 1024
             + min(height, 128) * width * (96 if native else 192))
    return fixed, height * width * (12 if native else 16)


def guided_resources(height, width, parameters, frames):
    # Band-streamed Pass 2 and OpenCL workspace hold bounded accumulators
    # (~24*N fixed, ~8*N per worker; GPU Peak RSS ~0.8 GiB, CPU ~2.1 GiB at 45 MP).
    return height * width * 24, height * width * 8


def gfg_resources(height, width, parameters, frames):
    # Row-parallel streaming and OpenCL workspace (~16*N fixed, ~8*N per worker;
    # GPU Peak RSS ~0.8 GiB, CPU ~1.25 GiB at 45 MP).
    return height * width * 16, height * width * 8


def dct_resources(height, width, parameters, frames):
    # Coarse block variance/index maps plus active frame strip and output buffers
    # (~8*N fixed, ~8*N per worker; Peak RSS ~1.0-1.2 GiB at 45 MP).
    padded = (height + 8) * (width + 8)
    return padded * 8, padded * 8


def _openfocus_needs_tiles(frames, threshold, *, height=None, width=None, **_kwargs):
    from core.multi_focus_fusion import _DEFAULT_TILE_THRESHOLD
    # Global maps are streamed through bounded rows/mappings. Memory pressure
    # must not silently replace that context with independently fused crops.
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    h = int(height) if height is not None else h
    w = int(width) if width is not None else w
    return threshold != _DEFAULT_TILE_THRESHOLD and max(h, w) > threshold


def openfocus_a_needs_tiles(frames, threshold, **kwargs):
    return _openfocus_needs_tiles(frames, threshold, **kwargs)


def openfocus_b_needs_tiles(frames, threshold, **kwargs):
    return _openfocus_needs_tiles(frames, threshold, **kwargs)


def _spatial_needs_tiles(algorithm, frames, threshold, *, height=None, width=None, **_kwargs):
    from core.multi_focus_fusion import (_DEFAULT_TILE_THRESHOLD,
                                         estimate_peak_bytes,
                                         memory_budget_bytes)
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    h = int(height) if height is not None else h
    w = int(width) if width is not None else w
    if threshold != _DEFAULT_TILE_THRESHOLD and max(h, w) > threshold:
        return True
    return estimate_peak_bytes(algorithm, frames, height=h, width=w) > memory_budget_bytes()


def guided_needs_tiles(frames, threshold, **kwargs):
    return _spatial_needs_tiles('guided_filter', frames, threshold, **kwargs)


def gfg_needs_tiles(frames, threshold, **kwargs):
    return _spatial_needs_tiles('gfgfgf', frames, threshold, **kwargs)


def dct_needs_tiles(frames, threshold, **kwargs):
    return _spatial_needs_tiles('dct', frames, threshold, **kwargs)


def dtcwt_resources(height, width, parameters, frames):
    from fusion_methods.dtcwt import estimate_working_set
    return estimate_working_set(height, width, parameters['N'], frames)


def dtcwt_needs_tiles(frames, threshold, *, height=None, width=None, **_kwargs):
    from core.multi_focus_fusion import estimate_peak_bytes, memory_budget_bytes
    # Prefer whole transforms to avoid introducing tile-boundary wavelets.
    if not frames:
        return False
    return estimate_peak_bytes('dtcwt', frames, height=height, width=width) > memory_budget_bytes()


def neural_needs_tiles(frames, threshold, *, height=None, width=None, **_kwargs):
    from core.multi_focus_fusion import _stack_footprint_exceeds
    h, w = frames[0].shape[:2]
    h = int(height) if height is not None else h
    w = int(width) if width is not None else w
    return max(h, w) > threshold or _stack_footprint_exceeds(
        frames, threshold, height=h, width=w)


def registration_homography(registration, images, thread_count=4, should_cancel=None, **parameters):
    return registration._process_homography(
        images, thread_count=thread_count, should_cancel=should_cancel,
        failure_policy=parameters['failure_policy'],
        max_shift_x_ratio=parameters.get('max_shift_x_ratio'),
        max_shift_y_ratio=parameters.get('max_shift_y_ratio'),
        max_rotation_deg=parameters.get('max_rotation_deg'),
        max_scale_change=parameters.get('max_scale_change'),
        auto_crop=parameters.get('auto_crop', True),
        interpolation=parameters.get('interpolation', 'lanczos'),
    )


def registration_ecc(registration, images, thread_count=4, should_cancel=None, **parameters):
    return registration._process_ecc(
        images, thread_count=thread_count, should_cancel=should_cancel,
        failure_policy=parameters['failure_policy'],
        stage_budget=parameters.get('stage_budget'),
        progress_callback=parameters.get('progress_callback'),
        max_shift_x_ratio=parameters.get('max_shift_x_ratio'),
        max_shift_y_ratio=parameters.get('max_shift_y_ratio'),
        max_rotation_deg=parameters.get('max_rotation_deg'),
        max_scale_change=parameters.get('max_scale_change'),
        auto_crop=parameters.get('auto_crop', True),
        interpolation=parameters.get('interpolation', 'lanczos'),
    )


def registration_openfocus(registration, images, thread_count=4, should_cancel=None, **parameters):
    from core.registration import _align_openfocus_impl, deferred_warp_threads

    with deferred_warp_threads(thread_count if thread_count and thread_count > 0 else None):
        return _align_openfocus_impl(
            images, should_cancel=should_cancel,
            max_shift_ratio=parameters['max_shift_ratio'],
            max_shift_x_ratio=getattr(registration, 'max_shift_x_ratio', None),
            max_shift_y_ratio=getattr(registration, 'max_shift_y_ratio', None),
            failure_policy=parameters['failure_policy'],
            reference_frame=parameters['reference_frame'],
            rotation=parameters['rotation'],
            magnification=parameters['magnification'],
            alignment_roi=getattr(registration, 'alignment_roi', None),
            auto_crop=getattr(registration, 'auto_crop', True),
            interpolation=getattr(registration, 'openfocus_interpolation', None),
            defer_warps=registration.defer_warps,
            output_path=parameters.get('output_path'),
            plan_callback=lambda plan: setattr(registration, 'alignment_plan', plan),
            dispatch_callback=registration._record_dispatch,
        )


def depth_dct(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_dct
    return _measure_dct(images, progress_callback, should_cancel,
                        kernel_size=parameters['kernel_size'],
                        block_size=parameters.get('block_size', 8),
                        thread_count=parameters.get('thread_count'))


def depth_gfg(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_gfgfgf
    return _measure_gfgfgf(images, progress_callback, should_cancel, kernel_size=parameters['kernel_size'], thread_count=parameters.get('thread_count'))


def depth_dtcwt(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_dtcwt
    return _measure_dtcwt(images, progress_callback, should_cancel,
                          nlevels=parameters['N'],
                          thread_count=parameters.get('thread_count'))


def depth_stackmffv4(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_stackmffv4
    return _measure_stackmffv4(
        images, progress_callback, should_cancel,
        model_path=parameters.get('model_path'), use_gpu=parameters.get('use_gpu', True),
        execution_plan=parameters.get('execution_plan'), mode=parameters.get('mode', 'soft'))


def depth_neural(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_stackmffv3
    return _measure_stackmffv3(images, progress_callback, should_cancel,
        model_path=parameters.get('model_path'), use_gpu=parameters.get('use_gpu', True),
        execution_plan=parameters.get('execution_plan'), mode=parameters['mode'],
        inference_scale=parameters['inference_scale'])


def depth_stackmffv2(images, progress_callback=None, should_cancel=None, **parameters):
    from core.depth_map import _measure_stackmffv2
    return _measure_stackmffv2(images, progress_callback, should_cancel,
        model_path=parameters.get('model_path'), use_gpu=parameters.get('use_gpu', True),
        execution_plan=parameters.get('execution_plan'),
        thread_count=parameters.get('thread_count'),
        resource_preference=parameters.get('resource_preference', 'balanced'))


def stackmffv2_needs_tiles(frames, threshold, *, height=None, width=None,
                           device=None, **_kwargs):
    """Tile when geometry or the attention working set exceeds its budget."""
    from core.multi_focus_fusion import _stack_footprint_exceeds
    from core.resource_policy import ResourceLimitError
    from fusion_methods.stackmff_v2 import _prepare_inference
    h, w = frames[0].shape[:2]
    h = int(height) if height is not None else h
    w = int(width) if width is not None else w
    if max(h, w) > threshold or _stack_footprint_exceeds(frames, threshold, height=h, width=w):
        return True
    try:
        _prepare_inference(frames, None, False, height=h, width=w, device=device or 'CPU')
    except ResourceLimitError:
        return True
    return False


def depth_openfocus_b(images, progress_callback=None, should_cancel=None, **parameters):
    from fusion_methods.openfocus_depth import depth_map
    return depth_map(images, None, 2 * parameters['radius'] + 1, parameters['smoothing'],
                     should_cancel, progress_callback,
                     thread_count=parameters.get('thread_count'), half_resolution=True,
                     preserve_detail=True, feather=parameters['feather'],
                     use_gpu=parameters.get('use_gpu', True))


def depth_openfocus_c(images, progress_callback=None, should_cancel=None, **parameters):
    from fusion_methods.openfocus_depth import depth_map
    return depth_map(images, None, 17, parameters['smoothing'], should_cancel, progress_callback,
                     thread_count=parameters.get('thread_count'), half_resolution=True,
                     use_gpu=parameters.get('use_gpu', True))
