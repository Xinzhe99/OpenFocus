"""Band-limited OpenCL order statistics for float32 background focus energies."""
import logging

import numpy as np

from fusion_methods._openfocus_gpu import (
    GpuError, _device, _device_lock, _free_vram, opencl_device_name,
)

log = logging.getLogger(__name__)
_PROGRAM_CACHE = {}

_KERNEL = r'''
// Each work item selects one pixel's median in its private column of the
// device buffer. Neighbouring pixels read consecutive values in each frame.
__kernel void reduce_focus(__global float *energy, __global float *peak,
                           __global float *median, __global uchar *indices,
                           const int frames, const int pixels)
{
    size_t pixel = get_global_id(0);
    #define VALUE(frame) energy[(size_t)(frame) * pixels + pixel]
    float best = VALUE(0);
    int index = 0;
    for (int frame = 1; frame < frames; ++frame) {
        float value = VALUE(frame);
        if (value > best || (isnan(value) && !isnan(best))) {
            best = value;
            index = frame;
        }
    }
    peak[pixel] = best;
    if (frames <= 256) indices[pixel] = (uchar)index;
    else ((__global ushort *)indices)[pixel] = (ushort)index;
    if (isnan(best)) { median[pixel] = best; return; }

    // Partition until the upper middle value is in its sorted position.
    int middle = frames / 2, left = 0, right = frames - 1;
    while (left < right) {
        float pivot = VALUE(left + (right - left) / 2);
        int i = left, j = right;
        while (i <= j) {
            while (VALUE(i) < pivot) ++i;
            while (VALUE(j) > pivot) --j;
            if (i <= j) {
                float a = VALUE(i), b = VALUE(j);
                VALUE(i) = b;
                VALUE(j) = a;
                ++i;
                --j;
            }
        }
        if (middle <= j) right = j;
        else if (middle >= i) left = i;
        else break;
    }
    float upper = VALUE(middle);
    if (frames & 1) median[pixel] = upper;
    else {
        float lower = VALUE(0);
        for (int k = 1; k < middle; ++k) lower = fmax(lower, VALUE(k));
        median[pixel] = (lower + upper) * 0.5f;
    }
    #undef VALUE
}
'''


def _band_pixels(device, frames, index_bytes):
    from dataclasses import replace
    from core import resource_policy

    ram = resource_policy.probe_memory()
    host_limit = resource_policy.ram_budget(ram)
    if device.host_unified_memory:
        limit = host_limit // 2
    else:
        memory = _free_vram(device)
        if memory is None:
            raise GpuError('GPU free memory is unknown')
        limit = resource_policy.vram_budget(replace(ram, free_vram=memory[0], total_vram=memory[1]))
    # Leave space for the context. Allow a native host transfer buffer as
    # large as the device workspace and respect the individual allocation cap.
    limit = min(limit - (16 << 20), host_limit // 2, 32 << 20, int(device.global_mem_size) // 4)
    pixels = min(limit // (4 * frames + 8 + index_bytes), int(device.max_mem_alloc_size) // (4 * frames))
    if pixels < 1:
        raise GpuError('Focus energy reduction exceeds the resource budget')
    return pixels


def _get_or_build_kernel(cl, device):
    """Return cached (ctx, queue, kernel) for this OpenCL device."""
    from fusion_methods._openfocus_gpu import _build_cached_program, _get_shared_context

    ctx, queue = _get_shared_context(cl, device)
    key = device.int_ptr
    cached = _PROGRAM_CACHE.get(key)
    if cached is not None and cached[0] is ctx:
        return cached
    program = _build_cached_program(cl, device, ctx, _KERNEL)
    kernel = cl.Kernel(program, 'reduce_focus')
    entry = (ctx, queue, kernel)
    _PROGRAM_CACHE[key] = entry
    return entry


def reduce_focus(energy, check_cancel):
    """Return peak, median and first argmax, or raise GpuError to retry on CPU."""
    frames, height, width = energy.shape
    indices_type = np.uint8 if frames <= 256 else np.uint16
    while not _device_lock.acquire(timeout=0.1):
        check_cancel()
    try:
        check_cancel()
        device = None
        try:
            import pyopencl as cl
            device = _device(cl)
            if device is None:
                raise GpuError('No available OpenCL GPU')
            required = cl.device_fp_config.DENORM | cl.device_fp_config.INF_NAN | cl.device_fp_config.ROUND_TO_NEAREST
            if device.single_fp_config & required != required:
                raise GpuError('OpenCL device lacks IEEE float32 arithmetic')
            band = min(height * width, _band_pixels(device, frames, np.dtype(indices_type).itemsize))
            ctx, queue, kernel = _get_or_build_kernel(cl, device)
            workspace = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, band * frames * 4)
            peak = np.empty(height * width, np.float32)
            median = np.empty_like(peak)
            indices = np.empty(height * width, indices_type)
            outputs = [(a, cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, band * a.itemsize)) for a in (peak, median, indices)]
        except GpuError:
            raise
        except Exception as exc:
            if device is not None:
                _PROGRAM_CACHE.pop(device.int_ptr, None)
            from fusion_methods._openfocus_gpu import _invalidate_shared_context
            _invalidate_shared_context()
            raise GpuError('Could not initialize background focus reduction') from exc
        completed = False
        try:
            for start in range(0, peak.size, band):
                check_cancel()
                count = min(band, peak.size - start)
                try:
                    # One rectangular DMA reads this band directly from each
                    # contiguous energy plane, without packing a second host stack.
                    cl.enqueue_copy(queue, workspace, energy, buffer_origin=(0, 0), host_origin=(start * 4, 0),
                                    region=(count * 4, frames, 1), host_pitches=(peak.nbytes,))
                    kernel(queue, (count,), None, workspace, *(buf for _, buf in outputs), np.int32(frames), np.int32(count))
                    for array, buf in outputs:
                        cl.enqueue_copy(queue, array[start:start + count], buf)
                except Exception as exc:
                    _PROGRAM_CACHE.pop(device.int_ptr, None)
                    from fusion_methods._openfocus_gpu import _invalidate_shared_context
                    _invalidate_shared_context()
                    raise GpuError('OpenCL background focus reduction failed') from exc
            completed = True
        finally:
            try:
                queue.finish()
            except Exception as exc:
                _PROGRAM_CACHE.pop(device.int_ptr, None)
                from fusion_methods._openfocus_gpu import _invalidate_shared_context
                _invalidate_shared_context()
                if completed:
                    raise GpuError('OpenCL background focus synchronization failed') from exc
            finally:
                del workspace, outputs
        log.info('Background focus reduction completed on %s (frames=%d, pixels=%d, band_pixels=%d)',
                 opencl_device_name(device), frames, peak.size, band)
        return peak.reshape(height, width), median.reshape(height, width), indices.reshape(height, width)
    finally:
        _device_lock.release()
