"""Optional job-local OpenCL implementation of the fractional CPU INTER_AREA path."""
import ctypes
import logging
import math
import os
import sys
from threading import Lock

import cv2
import numpy as np


# OpenCV 5.0.0 modules/imgproc/src/resize.cpp, computeResizeAreaTab and
# ResizeArea_Invoker: ordered binary32 products and sums on the MSVC SSE
# baseline, followed by saturate_cast (nearest even for integer pixels).
# Integral scale factors dispatch to a different CPU implementation.
_TAPS = 5
_AXIS = np.dtype([('start', '<i4'), ('count', '<i4'), ('weight', '<f4', _TAPS)])
_TILE_ROWS = 128
_MIN_SOURCE_BYTES = 8 * 1024**2
_MAX_SIDE = 32768
_KERNEL_NAMES = ((1, 'area8'), (2, 'area16'))

_cache_lock = Lock()
_queue_lock = Lock()
_PROGRAM_CACHE = None


def _get_or_build_program(cl, device):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import _device_cache_key

    key = _device_cache_key(device)
    with _cache_lock:
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key:
            return _PROGRAM_CACHE[1:]
        context = cl.Context([device])
        queue = cl.CommandQueue(context, device=device)
        program = cl.Program(context, _KERNEL).build()
        kernels = {size: cl.Kernel(program, name) for size, name in _KERNEL_NAMES}
        _PROGRAM_CACHE = (key, context, queue, kernels)
        return context, queue, kernels


def _invalidate_program_cache():
    global _PROGRAM_CACHE
    with _cache_lock:
        _PROGRAM_CACHE = None

_KERNEL = r'''
#pragma OPENCL FP_CONTRACT OFF

typedef struct { int start, count; float weight[5]; } Span;

#define AREA(NAME, TYPE, STORE) \
__kernel void NAME(__global const TYPE *src, __global TYPE *dst, \
                   __global const Span *xs, __global const Span *ys, \
                   int channels, int src_stride, int first_src_y, int first_dst_y) \
{ \
    int k = get_global_id(0), y = get_global_id(1); \
    Span xspan = xs[k / channels], yspan = ys[first_dst_y + y]; \
    int i = (yspan.start - first_src_y) * src_stride + xspan.start * channels + k % channels; \
    float total = 0.0f; \
    for (int dy = 0; dy < yspan.count; ++dy, i += src_stride) { \
        float row = 0.0f; \
        for (int dx = 0; dx < xspan.count; ++dx) \
            row += (float)src[i + dx * channels] * xspan.weight[dx]; \
        total += row * yspan.weight[dy]; \
    } \
    dst[y * get_global_size(0) + k] = STORE(total); \
}
AREA(area8, uchar, convert_uchar_sat_rte)
AREA(area16, ushort, convert_ushort_sat_rte)
'''


def _cpu_contract():
    """Admit only the audited generic CPU path and its rounding semantics."""
    if sys.platform != 'win32' or cv2.__version__ != '5.0.0':
        return False
    fields = dict(line.strip().split(':', 1) for line in cv2.getBuildInformation().splitlines()
                  if ':' in line)
    fields = {key: value.strip() for key, value in fields.items()}
    flags = fields.get('C++ flags (Release)', '').lower().split()
    return (fields.get('Version control') == '5.0.0'
            and fields.get('Host', '').endswith('AMD64')
            and fields.get('Baseline') == 'SSE SSE2 SSE3'
            and '/cl.exe ' in fields.get('C++ Compiler', '').lower()
            and '/fp:precise' in flags and '/fp:fast' not in flags and '/fp:contract' not in flags
            and not any(flag.startswith('/arch:') for flag in flags)
            and fields.get('Custom HAL') in ('NO', 'YES (ipp (ver 0.0.1))'))


def _axis_table(source, dest):
    # Match OpenCV's double reciprocal, 1e-3 edge cutoff and final partial
    # cell. Store only the final coefficient in binary32, exactly once.
    scale = 1.0 / (dest / source)
    table = np.zeros(dest, _AXIS)
    for d in range(dest):
        left = d * scale
        right = left + scale
        cell = min(scale, source - left)
        end = min(math.floor(right), source - 1)
        start = min(math.ceil(left), end)
        weights = []
        first = start
        if start - left > 1e-3:
            first -= 1
            weights.append((start - left) / cell)
        weights.extend([1.0 / cell] * (end - start))
        if right - end > 1e-3:
            weights.append(min(right - end, 1.0, cell) / cell)
        table['start'][d] = first
        table['count'][d] = len(weights)
        table['weight'][d, :len(weights)] = weights
    return table


class ExactAreaGpu:
    """One stack's lazy kernel and stripe buffers. None requests CPU resize."""

    def __init__(self):
        self._lock = Lock()
        self._disabled = False
        self._cl = None
        self._shape = None
        self._buffers = None
        self._context = None
        self._queue = None
        self._kernels = None
        self.frames_resized = 0
        self.device_name = ''

    def _select_device(self):
        if not _cpu_contract() or os.environ.get('PYOPENCL_BUILD_OPTIONS'):
            return False
        from fusion_methods._openfocus_gpu import _device
        import pyopencl as cl

        device = _device(cl)
        if (device is None or not device.endian_little or
                not (device.single_fp_config & cl.device_fp_config.ROUND_TO_NEAREST)):
            return False
        self._round_mode = ctypes.CDLL('ucrtbase').fegetround
        self._device = device
        self._cl = cl
        return True

    def _workspace(self, image, width, height, channels):
        from core import resource_policy as policy
        from fusion_methods._openfocus_gpu import _free_vram

        self._buffers = self._shape = None
        source_h, source_w = image.shape[:2]
        itemsize = image.itemsize
        rows = min(_TILE_ROWS, height)
        src_bytes = min(source_h, math.ceil(rows * source_h / height) + 2) * source_w * channels * itemsize
        dst_bytes = rows * width * channels * itemsize
        table_bytes = (width + height) * _AXIS.itemsize
        workspace = src_bytes + dst_bytes + table_bytes
        memory = _free_vram(self._device)
        if memory is None:
            return False
        budget = policy.vram_budget(policy.ResourceSnapshot(free_vram=memory[0], total_vram=memory[1]))
        # Also allow for one stripe of pageable transfer staging on the host.
        if (workspace > budget or max(src_bytes, dst_bytes, table_bytes) > self._device.max_mem_alloc_size
                or workspace + height * width * channels * itemsize >
                policy.frame_ram_budget(policy.probe_resources())):
            return False
        if self._context is None:
            self._context, self._queue, self._kernels = _get_or_build_program(
                self._cl, self._device,
            )
        xs, ys = _axis_table(source_w, width), _axis_table(source_h, height)
        cl, context = self._cl, self._context
        flags = cl.mem_flags
        self._buffers = (cl.Buffer(context, flags.READ_ONLY, src_bytes),
                         cl.Buffer(context, flags.WRITE_ONLY, dst_bytes),
                         cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=xs),
                         cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=ys))
        self._ys = ys
        self._shape = image.shape, image.dtype, width, height
        return True

    def resize(self, image, dsize):
        """Return CPU-equivalent cv2.resize(image, dsize, interpolation=INTER_AREA), or None.

        dsize is explicit (OpenCV may use different coefficients with fx/fy).
        Large contiguous uint8/uint16 gray, BGR and BGRA images are eligible.
        Both axes must shrink by at most four, with a fractional factor on at
        least one axis. All other calls remain ordinary CPU operations.
        """
        if (self._disabled or image.dtype not in (np.uint8, np.uint16)
                or image.ndim not in (2, 3) or not image.flags.c_contiguous
                or image.nbytes < _MIN_SOURCE_BYTES):
            return None
        h, w = image.shape[:2]
        width, height = dsize
        channels = image.shape[2] if image.ndim == 3 else 1
        if (channels not in (1, 3, 4) or max(h, w) > _MAX_SIDE
                or not (0 < width < w <= 4 * width and 0 < height < h <= 4 * height)
                or (w % width == 0 and h % height == 0)):
            return None
        with self._lock, _queue_lock:
            if self._disabled:
                return None
            try:
                if self._cl is None and not self._select_device():
                    self._disabled = True
                    return None
                if cv2.ipp.useIPP_NotExact() or self._round_mode() != 0:
                    return None
                if self._shape != (image.shape, image.dtype, width, height):
                    if not self._workspace(image, width, height, channels):
                        self._disabled = True
                        return None
                shape = (height, width) if channels == 1 else (height, width, channels)
                output = np.empty(shape, image.dtype)
                src, dst, xs, ys = self._buffers
                kernel = self._kernels[image.itemsize]
                starts, counts = self._ys['start'], self._ys['count']
                for row in range(0, height, _TILE_ROWS):
                    stop = min(row + _TILE_ROWS, height)
                    first = starts[row]
                    last = starts[stop - 1] + counts[stop - 1]
                    band = image[first:last]
                    self._cl.enqueue_copy(self._queue, src, band, is_blocking=False)
                    kernel(self._queue, (width * channels, stop - row), None, src, dst, xs, ys,
                           np.int32(channels), np.int32(w * channels), first, np.int32(row))
                    self._cl.enqueue_copy(self._queue, output[row:stop], dst, is_blocking=True)
                if self.frames_resized == 0:
                    from fusion_methods._openfocus_gpu import opencl_device_name
                    self.device_name = opencl_device_name(self._device)
                    logging.getLogger(__name__).info('ECC working-frame resize on %s', self.device_name)
                self.frames_resized += 1
                return output
            except Exception as exc:
                drain_failed = False
                queue = self._queue
                if queue is not None:
                    try:
                        queue.finish()
                    except Exception:
                        drain_failed = True
                err_types = tuple(
                    t for t in (getattr(self._cl, 'Error', None), MemoryError)
                    if isinstance(t, type)
                )
                if drain_failed or (err_types and isinstance(exc, err_types)):
                    _invalidate_program_cache()
                self._disabled = True
                self._buffers = None
                self._context = None
                self._queue = None
                self._kernels = None
                logging.getLogger(__name__).info('ECC area resize using CPU: %s', exc)
                return None
