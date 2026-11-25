"""Optional job-local OpenCL implementation of audited CPU INTER_AREA paths."""
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
# Whole blocks use resizeAreaFast_Invoker's sum followed by one binary32
# reciprocal multiplication. ResizeAreaFastVec's special 2x2 stays on CPU.
_MAX_SCALE = 8
_TAPS = _MAX_SCALE + 1
_AXIS = np.dtype([('start', '<i4'), ('count', '<i4'), ('weight', '<f4', _TAPS)])
_TILE_ROWS = 128
_MIN_SOURCE_BYTES = 8 * 1024**2
_MAX_SIDE = 32768
_KERNEL_NAMES = (
    (1, False, False, 'area8'),
    (2, False, False, 'area16'),
    (1, True, False, 'area8_bgr_rev'),
    (2, True, False, 'area16_bgr_rev'),
    (1, False, True, 'block8'),
    (2, False, True, 'block16'),
    (1, True, True, 'block8_bgr_rev'),
    (2, True, True, 'block16_bgr_rev'),
)

_cache_lock = Lock()
_queue_lock = Lock()
_PROGRAM_CACHE = None


def _is_bgr_rev(image):
    return (not image.flags.c_contiguous and image.ndim == 3 and image.shape[2] == 3
            and image.dtype in (np.uint8, np.uint16) and image[..., ::-1].flags.c_contiguous)


def _get_or_build_program(cl, device):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import (
        _build_cached_program, _device_cache_key, _get_shared_context,
    )

    key = _device_cache_key(device)
    with _cache_lock:
        context = _get_shared_context(cl, device)
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key and _PROGRAM_CACHE[1] is context:
            return _PROGRAM_CACHE[1:]
        queue = cl.CommandQueue(context, device=device)
        program = _build_cached_program(cl, device, context, _KERNEL)
        kernels = {(size, rev, block): cl.Kernel(program, name)
                   for size, rev, block, name in _KERNEL_NAMES}
        _PROGRAM_CACHE = (key, context, queue, kernels)
        return context, queue, kernels


def _invalidate_program_cache(context=None):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import _invalidate_shared_context

    with _cache_lock:
        _PROGRAM_CACHE = None
        _invalidate_shared_context(context)

_KERNEL = f'#define AREA_TAPS {_TAPS}\n' + r'''
#pragma OPENCL FP_CONTRACT OFF

typedef struct { int start, count; float weight[AREA_TAPS]; } Span;

#define AREA(NAME, TYPE, STORE, REVERSE) \
__kernel void NAME(__global const TYPE *src, __global TYPE *dst, \
                   __global const Span *xs, __global const Span *ys, \
                   int channels, int src_stride, int first_src_y, int first_dst_y) \
{ \
    int k = get_global_id(0), y = get_global_id(1); \
    Span xspan = xs[k / channels], yspan = ys[first_dst_y + y]; \
    int c = k % channels; \
    int i = (yspan.start - first_src_y) * src_stride + xspan.start * channels \
            + (REVERSE ? 2 - c : c); \
    float total = 0.0f; \
    for (int dy = 0; dy < yspan.count; ++dy, i += src_stride) { \
        float row = 0.0f; \
        for (int dx = 0; dx < xspan.count; ++dx) \
            row += (float)src[i + dx * channels] * xspan.weight[dx]; \
        total += row * yspan.weight[dy]; \
    } \
    dst[y * get_global_size(0) + k] = STORE(total); \
}
AREA(area8, uchar, convert_uchar_sat_rte, 0)
AREA(area16, ushort, convert_ushort_sat_rte, 0)
AREA(area8_bgr_rev, uchar, convert_uchar_sat_rte, 1)
AREA(area16_bgr_rev, ushort, convert_ushort_sat_rte, 1)

#define BLOCK(NAME, TYPE, STORE, REVERSE) \
__kernel void NAME(__global const TYPE *src, __global TYPE *dst, \
                   int channels, int src_stride, int sx, int sy, float scale) \
{ \
    int k = get_global_id(0), y = get_global_id(1); \
    int c = k % channels; \
    int i = y * sy * src_stride + (k - c) * sx + (REVERSE ? 2 - c : c); \
    uint sum = 0; \
    for (int dy = 0; dy < sy; ++dy, i += src_stride) { \
        for (int dx = 0; dx < sx; ++dx) \
            sum += src[i + dx * channels]; \
    } \
    dst[y * get_global_size(0) + k] = STORE((float)sum * scale); \
}
BLOCK(block8, uchar, convert_uchar_sat_rte, 0)
BLOCK(block16, ushort, convert_ushort_sat_rte, 0)
BLOCK(block8_bgr_rev, uchar, convert_uchar_sat_rte, 1)
BLOCK(block16_bgr_rev, ushort, convert_ushort_sat_rte, 1)
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

    def __init__(self, *, should_cancel=None):
        self._should_cancel = should_cancel
        self._lock = Lock()
        self._disabled = False
        self._cl = None
        self._shape = None
        self._buffers = self._ys = None
        self._context = None
        self._queue = None
        self._kernels = None
        self.frames_resized = 0
        self.device_name = ''
        self.fell_back = False

    def close(self):
        from fusion_methods._openfocus_gpu import _release_buffers

        _release_buffers(self._buffers)
        self._buffers = self._shape = self._ys = None

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

    def _workspace(self, image, width, height, channels, block):
        from core import resource_policy as policy
        from fusion_methods._openfocus_gpu import _free_vram

        self.close()
        source_h, source_w = image.shape[:2]
        itemsize = image.itemsize
        # Preserve the source-stripe row bound of the fourfold path. At
        # larger ratios this reduces transfer staging and device residency.
        rows = min(_TILE_ROWS, height, _TILE_ROWS * 4 * height // source_h)
        source_rows = (rows * block[1] if block else
                       min(source_h, math.ceil(rows * source_h / height) + 2))
        src_bytes = source_rows * source_w * channels * itemsize
        dst_bytes = rows * width * channels * itemsize
        table_bytes = 0 if block else (width + height) * _AXIS.itemsize
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
        cl, context = self._cl, self._context
        flags = cl.mem_flags
        self._buffers = (cl.Buffer(context, flags.READ_ONLY, src_bytes),
                         cl.Buffer(context, flags.WRITE_ONLY, dst_bytes))
        if not block:
            xs, ys = _axis_table(source_w, width), _axis_table(source_h, height)
            self._buffers += (cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=xs),
                              cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=ys))
            self._ys = ys
        self._rows = rows
        self._shape = image.shape, image.dtype, width, height
        return True

    def resize(self, image, dsize):
        """Return CPU-equivalent cv2.resize(image, dsize, interpolation=INTER_AREA), or None.

        dsize is explicit (OpenCV may use different coefficients with fx/fy).
        Large contiguous uint8/uint16 gray, BGR, BGRA and bgr_rev images are eligible.
        Both axes must shrink by at most eight. The special 2x2 path stays on CPU.
        """
        bgr_rev = _is_bgr_rev(image)
        if (self._disabled or image.dtype not in (np.uint8, np.uint16)
                or image.ndim not in (2, 3) or not (image.flags.c_contiguous or bgr_rev)
                or image.nbytes < _MIN_SOURCE_BYTES):
            return None
        h, w = image.shape[:2]
        width, height = dsize
        channels = image.shape[2] if image.ndim == 3 else 1
        if (channels not in (1, 3, 4) or max(h, w) > _MAX_SIDE
                or not (0 < width < w <= _MAX_SCALE * width
                        and 0 < height < h <= _MAX_SCALE * height)):
            return None
        block = None
        if w % width == 0 and h % height == 0:
            block = (w // width, h // height)
        if block == (2, 2):
            return None
        from core.registration import RegistrationCancelled, _check_cancel

        should_cancel = self._should_cancel
        with self._lock, _queue_lock:
            _check_cancel(should_cancel)
            if self._disabled:
                return None
            try:
                if self._cl is None and not self._select_device():
                    self._disabled = True
                    self.fell_back = True
                    return None
                if cv2.ipp.useIPP_NotExact() or self._round_mode() != 0:
                    self.fell_back = True
                    return None
                if self._shape != (image.shape, image.dtype, width, height):
                    if not self._workspace(image, width, height, channels, block):
                        self._disabled = True
                        self.fell_back = True
                        return None
                shape = (height, width) if channels == 1 else (height, width, channels)
                output = np.empty(shape, image.dtype)
                src, dst = self._buffers[:2]
                kernel = self._kernels[(image.itemsize, bgr_rev, block is not None)]
                layout = (np.int32(channels), np.int32(w * channels))
                if block:
                    sx, sy = block
                    # Even uint16 block sums are below 2**24, so float32
                    # accumulation on the CPU is exact before the multiply.
                    args = (*layout, np.int32(sx), np.int32(sy), np.float32(1.0 / (sx * sy)))
                else:
                    args = (*self._buffers[2:], *layout)
                    starts, counts = self._ys['start'], self._ys['count']
                source_view = image[..., ::-1] if bgr_rev else image
                for row in range(0, height, self._rows):
                    _check_cancel(should_cancel)
                    stop = min(row + self._rows, height)
                    if block:
                        first, last = row * sy, stop * sy
                        stripe_args = args
                    else:
                        first = starts[row]
                        last = starts[stop - 1] + counts[stop - 1]
                        stripe_args = (*args, first, np.int32(row))
                    band = source_view[first:last]
                    # Keep the host buffer export alive through the blocking readback.
                    upload = self._cl.enqueue_copy(self._queue, src, band, is_blocking=False)
                    kernel(self._queue, (width * channels, stop - row), None, src, dst, *stripe_args)
                    self._cl.enqueue_copy(self._queue, output[row:stop], dst, is_blocking=True)
                    del upload
                _check_cancel(should_cancel)
                if self.frames_resized == 0:
                    from fusion_methods._openfocus_gpu import opencl_device_name
                    self.device_name = opencl_device_name(self._device)
                    logging.getLogger(__name__).info('ECC working-frame resize on %s', self.device_name)
                self.frames_resized += 1
                return output
            except RegistrationCancelled:
                # All cancellation checks are at stripe boundaries with no work in flight.
                raise
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
                    _invalidate_program_cache(self._context)
                self._disabled = True
                self.fell_back = True
                self.close()
                self._context = None
                self._queue = None
                self._kernels = None
                logging.getLogger(__name__).info('ECC area resize using CPU: %s', exc)
                return None
