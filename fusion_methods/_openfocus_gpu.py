"""Bounded OpenCL depth activity and ordered merge for OpenFocus B."""
import logging
from threading import Lock

import numpy as np

_device_lock = Lock()

_MERGE_KERNEL = r'''
__kernel void activity(__global const ushort *gray, __global const ushort *fine,
                       __global ushort *out)
{
    size_t i = get_global_id(0);
    ulong g = gray[i], f = fine[i];
    ulong maximum = max(g, f);
    ulong denominator = ((maximum * 100ul * 0x80008001ul) >> 32) >> 15;
    ulong difference = g > f ? g - f : f - g;
    out[i] = (ushort)min(65535ul, 110ul * difference / (denominator + 10ul));
}

__kernel void merge(__global ushort *acc, __global const ushort *img,
                    __global const long *depth, const int width,
                    const int channels, const int step, const int index,
                    const int scale, const int first)
{
    size_t pixel = get_global_id(0);
    int weight = 1024;
    if (!first) {
        long code = depth[pixel];
        if (code < 0 || code > 131070) return;
        long delta = code - (long)step * index;
        if (delta >= 0) {
            if (delta > step) return;
        } else {
            if (delta <= -step) return;
            weight = 1024 - (int)(((-delta) << 10) / step);
            if (weight <= 0) return;
        }
    }
    for (int c = 0; c < channels; ++c) {
        size_t k = pixel * channels + c;
        int value = (int)img[k] * scale;
        if (weight == 1024) acc[k] = (ushort)value;
        else acc[k] = (ushort)(((int)acc[k] * (1024 - weight)
                                + value * weight) >> 10);
    }
}
'''


def _device(cl):
    for platform in cl.get_platforms():
        try:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
        except cl.Error:
            continue
        for device in devices:
            if (device.type & cl.device_type.GPU and device.available
                    and device.compiler_available):
                return device
    return None


def gpu_available():
    """Return whether an available OpenCL GPU with a compiler is present."""
    try:
        import pyopencl as cl
        return _device(cl) is not None
    except Exception:
        return False


class GpuError(RuntimeError):
    """A device operation failed; the caller may continue on the CPU."""


def log_fallback(error):
    reason = error.__cause__ or error
    logging.getLogger(__name__).warning('OpenFocus GPU fallback to CPU: %s', reason)


def device_status(gpu):
    if gpu is None or not gpu.used:
        return 'CPU'
    if gpu.failed:
        return 'CPU + ' + gpu.device_name
    return gpu.device_name


class Merge:
    """A job-local OpenCL context and bounded merge workspace."""

    def __init__(self, width, channels):
        try:
            import pyopencl as cl
            from core import resource_policy
            from core.multi_focus_fusion import memory_budget_bytes

            device = _device(cl)
            if device is None:
                raise GpuError('No available OpenCL GPU')
            # Account for host staging and device allocations.
            row = int(width) * (8 + 5 * int(channels))
            budget = min(int(memory_budget_bytes()),
                         int(resource_policy.frame_ram_budget(resource_policy.probe_resources())),
                         int(device.global_mem_size) // 4, 16 * 1024 * 1024)
            largest_row = int(width) * max(8, 2 * int(channels))
            rows = min(128, budget // max(1, 2 * row),
                       int(device.max_mem_alloc_size) // max(1, largest_row))
            if rows < 1 or width <= 0 or channels <= 0:
                raise GpuError('GPU merge workspace exceeds the resource budget')
            self.cl = cl
            self.context = cl.Context(devices=[device])
            self.queue = cl.CommandQueue(self.context, device=device)
            self.program = cl.Program(self.context, _MERGE_KERNEL).build()
            self.device_name = 'OpenCL: ' + str(device.name).strip()
            self.activity_kernel = cl.Kernel(self.program, 'activity')
            self.kernel = cl.Kernel(self.program, 'merge')
            self.rows = rows
            self.failed = False
            self.used = False
        except GpuError:
            raise
        except Exception as exc:
            raise GpuError('OpenCL GPU initialization failed') from exc

    def acquire(self, should_cancel):
        from fusion_methods.openfocus_ab import _check_cancel
        while not _device_lock.acquire(timeout=0.1):
            _check_cancel(should_cancel)
        try:
            _check_cancel(should_cancel)
        except BaseException:
            _device_lock.release()
            raise

    def release(self):
        _device_lock.release()

    def activity(self, gray, fine, output):
        try:
            cl = self.cl
            source = np.ascontiguousarray(gray, dtype=np.uint16)
            filtered = np.ascontiguousarray(fine, dtype=np.uint16)
            result = np.empty(source.shape, dtype=np.uint16)
            flags = cl.mem_flags
            gb = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=source)
            fb = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=filtered)
            ob = cl.Buffer(self.context, flags.WRITE_ONLY, result.nbytes)
            self.activity_kernel(self.queue, (source.size,), None, gb, fb, ob)
            cl.enqueue_copy(self.queue, result, ob, is_blocking=True)
        except Exception as exc:
            raise GpuError('OpenCL GPU depth activity failed') from exc
        output[...] = result

    def band(self, accumulator, image, depth, index, step, scale, first):
        """Merge a band transactionally; failure leaves the accumulator intact."""
        try:
            cl = self.cl
            # Initial accumulation is deliberately not read from an
            # uninitialized transient frame.
            acc = (np.zeros(accumulator.shape, dtype=np.uint16) if first else
                   np.ascontiguousarray(accumulator, dtype=np.uint16))
            img = np.ascontiguousarray(image, dtype=np.uint16)
            dep = np.ascontiguousarray(depth, dtype=np.int64)
            flags = cl.mem_flags
            ab = cl.Buffer(self.context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=acc)
            ib = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=img)
            db = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=dep)
            channels = 1 if image.ndim == 2 else image.shape[2]
            self.kernel(self.queue, (depth.size,), None, ab, ib, db,
                        np.int32(depth.shape[1]), np.int32(channels),
                        np.int32(step), np.int32(index), np.int32(scale),
                        np.int32(first))
            # Copy to private staging before committing to the output plane.
            result = np.empty_like(acc)
            cl.enqueue_copy(self.queue, result, ab, is_blocking=True)
        except Exception as exc:
            raise GpuError('OpenCL GPU merge failed') from exc
        accumulator[...] = result
