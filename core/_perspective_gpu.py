"""Bounded, optional OpenCL Lanczos4 warping for large BGR8 frames."""
import ctypes
import logging
import math
import os
from threading import Lock

import cv2
import numpy as np

from core._ecc_area_gpu import _cpu_contract
from fusion_methods._openfocus_gpu import _device, _device_cache_key, _free_vram, opencl_device_name

_MIN_PIXELS = 1024 * 1024
_BAND_ROWS = 256
_REPAIR_LIMIT = 32
_REPAIR_BYTES = (_REPAIR_LIMIT + 1) * np.dtype(np.int32).itemsize
_MAX_SIDE = 32766
_queue_lock = Lock()
_program = None
_log = logging.getLogger(__name__)

# The coordinates are quantized before interpolation, as in OpenCV's
# warpPerspective/remap. The fixed-point coefficient sum fits in int32.
_SOURCE = r'''
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF

__kernel void warp(__global const uchar *src, __global uchar *dst,
                   __global const short *weights, volatile __global int *repair,
                   int sw, int sh, int dw, int y0, double tolerance,
                   double a, double b, double c, double d, double e, double f,
                   double g, double h, double i) {
    int x = get_global_id(0), row = get_global_id(1), y = y0 + row;
    double scale = 32.0 / ((h * y + i) + g * x);
    double fx = ((b * y + c) + a * x) * scale;
    double fy = ((e * y + f) + d * x) * scale;
    if (fabs(fx - floor(fx) - 0.5) <= tolerance ||
        fabs(fy - floor(fy) - 0.5) <= tolerance) {
        int slot = atomic_inc(repair);
        if (slot < REPAIR_LIMIT)
            repair[slot + 1] = y * dw + x;
        return;
    }
    int qx = convert_int_rte(fx), qy = convert_int_rte(fy);
    int sx = (qx >> 5) - 3, sy = (qy >> 5) - 3;
    __global const short *coeffs = weights + (((qy & 31) * 32 + (qx & 31)) * 64);
    int3 sum = (int3)(1 << 14);
    if (sx >= 0 && sx + 7 < sw && sy >= 0 && sy + 7 < sh) {
        for (int j = 0; j < 8; ++j) {
            __global const uchar *pixels = src + (size_t)((sy + j) * sw + sx) * 3;
            for (int k = 0; k < 8; ++k)
                sum += convert_int3(vload3(k, pixels)) * coeffs[j * 8 + k];
        }
    } else {
        for (int j = 0; j < 8; ++j) {
            if (sy + j < 0 || sy + j >= sh) continue;
            for (int k = 0; k < 8; ++k) {
                if (sx + k < 0 || sx + k >= sw) continue;
                int index = (sy + j) * sw + sx + k;
                sum += convert_int3(vload3(index, src)) * coeffs[j * 8 + k];
            }
        }
    }
    vstore3(convert_uchar3_sat(sum >> 15), row * dw + x, dst);
}
'''


def _coefficients():
    # Read the 32 one-dimensional phases from the installed OpenCV build.
    impulse = np.zeros((25, 25), np.float32)
    impulse[12, 12] = 1
    coords = np.empty((32, 8, 2), np.int16)
    coords[..., 0] = np.arange(15, 7, -1)
    coords[..., 1] = 12
    phases = np.broadcast_to(np.arange(32, dtype=np.uint16)[:, None], (32, 8)).copy()
    one = cv2.remap(impulse, coords, phases, cv2.INTER_LANCZOS4)
    table = np.empty((32, 32, 8, 8), np.int16)
    for y in range(32):
        for x in range(32):
            block = np.clip(np.rint(np.outer(one[y], one[x]) * 32768), -32768, 32767)
            delta = 32768 - int(block.sum())
            if delta:
                # OpenCV corrects the quantization error using the first
                # appropriate extremum inside the central 2x2 correction window.
                center = block[4:6, 4:6]
                index = (center.argmax() if delta > 0 else center.argmin())
                center[index // 2, index % 2] += delta
            table[y, x] = block
    return table


def _warp_geometry(matrix, width, height):
    ok, inverse = cv2.invert(matrix.astype(np.float64))
    if not ok or not np.isfinite(inverse).all():
        return None
    corners = np.array(((0, 0, 1), (width - 1, 0, 1),
                        (0, height - 1, 1), (width - 1, height - 1, 1)))
    points = inverse @ corners.T
    min_w = points[2].min()
    if min_w <= 1e-6 or np.max(np.abs(points[:2] / points[2])) >= _MAX_SIDE:
        return None
    # Enclose the host's block/FMA evaluation and OpenCL's double division.
    # The bound includes cancellation in both sums. Rounding can differ only
    # if a coordinate is within this bound of a half-integral bin boundary.
    magnitude = np.abs(inverse) @ corners[-1]
    eps = 16 * np.finfo(np.float64).eps
    denom = min_w - eps * magnitude[2]
    if denom <= 0:
        return None
    tolerance = 32 * eps * max(magnitude[:2]) / denom * (1 + magnitude[2] / denom)
    if not math.isfinite(tolerance) or tolerance >= 0.25:
        return None
    return inverse, tolerance


def _repair_rounding_edges(image, inverse, result, coordinates, should_cancel):
    from core.registration import _check_cancel

    height, width = result.shape[:2]
    # Match WarpPerspectiveInvoker's horizontal blocks in OpenCV 5.0.0.
    # Rebase each row/block using its exact first-pixel sums. OpenCV then
    # evaluates the same local x positions with its active SIMD or scalar path.
    block_width = min(1024 // min(16, height), width)
    blocks = set()
    for offset in coordinates:
        y, x = divmod(int(offset), width)
        blocks.add((y, x // block_width * block_width))
    shifted = inverse.copy()
    shifted[:, 1] = 0
    for y, x in blocks:
        _check_cancel(should_cancel)
        shifted[:, 2] = (inverse[:, 0] * x + inverse[:, 1] * y) + inverse[:, 2]
        cols = min(block_width, width - x)
        cv2.warpPerspective(image, shifted, (cols, 1), dst=result[y:y + 1, x:x + cols],
                            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)


def _verified_program(cl, device):
    global _program
    key = _device_cache_key(device)
    if _program is not None and _program[0] == key:
        return _program[1]
    context = cl.Context(devices=[device])
    queue = cl.CommandQueue(context)
    program = cl.Program(context, _SOURCE).build(options=[f'-DREPAIR_LIMIT={_REPAIR_LIMIT}'])
    kernel = cl.Kernel(program, 'warp')
    flags = cl.mem_flags
    weights = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=_coefficients())

    # Prove the active interpolation path, including border, phase and channel
    # arithmetic. Crop and real perspective transforms share the same kernel.
    sample = np.random.default_rng(4916).integers(0, 256, (71, 97, 3), np.uint8)
    height, width = sample.shape[:2]
    source = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=sample)
    target = cl.Buffer(context, flags.WRITE_ONLY, sample.nbytes)
    repair = cl.Buffer(context, flags.READ_WRITE, _REPAIR_BYTES)
    sample_out = np.empty_like(sample)
    edges = np.empty(_REPAIR_LIMIT + 1, np.int32)
    matrices = (
        np.array(((1.002, -.004, 5.37), (.003, .998, -3.81), (1e-5, -2e-5, 1))),
        np.array(((.94, .08, -7.4), (-.06, 1.05, 6.7), (4e-4, -3e-4, 1))),
    )
    try:
        for matrix in matrices:
            geometry = _warp_geometry(matrix, width, height)
            inv, tol = geometry
            cl.enqueue_fill_buffer(queue, repair, np.int32(0), 0, 4)
            kernel(queue, (width, height), None, source, target, weights, repair,
                   np.int32(width), np.int32(height), np.int32(width), np.int32(0),
                   np.float64(tol), *inv.flat)
            cl.enqueue_copy(queue, sample_out, target)
            cl.enqueue_copy(queue, edges, repair)
            if 0 < edges[0] <= _REPAIR_LIMIT:
                _repair_rounding_edges(sample, inv, sample_out, edges[1:1 + edges[0]], None)
            expected = cv2.warpPerspective(sample, matrix, (width, height), flags=cv2.INTER_LANCZOS4)
            if edges[0] > _REPAIR_LIMIT or not np.array_equal(sample_out, expected):
                raise RuntimeError('OpenCL Lanczos4 self-check differed from OpenCV')
    finally:
        queue.finish()
    _program = (key, (context, queue, kernel, weights))
    return _program[1]


class ExactPerspectiveGpu:
    """One preparation pass's buffers and dispatch counts."""

    def __init__(self):
        self._cl = self._device = self._buffers = self._shape = None
        self._disabled = False
        self._round_mode = None
        self.device_name = ''
        self.gpu_warps = 0
        self.cpu_warps = 0

    def close(self):
        self._buffers = self._shape = None

    def _initialize(self):
        if not _cpu_contract() or os.environ.get('PYOPENCL_BUILD_OPTIONS'):
            return False
        import pyopencl as cl

        device = _device(cl)
        if (device is None or not device.endian_little or
                'cl_khr_fp64' not in device.extensions.split() or
                not (device.double_fp_config & cl.device_fp_config.ROUND_TO_NEAREST)):
            return False
        self._round_mode = ctypes.CDLL('ucrtbase').fegetround
        self._cl, self._device = cl, device
        return True

    def _workspace(self, image, dsize):
        from core import resource_policy as policy

        width, height = dsize
        rows = min(_BAND_ROWS, height)
        target_bytes = width * rows * 3
        staging = (min(image.shape[0], _BAND_ROWS) * image.shape[1] * 3 + target_bytes)
        policy.require_memory(width * height * 3 + staging)
        self.close()
        memory = _free_vram(self._device)
        if memory is None:
            return False
        budget = policy.vram_budget(policy.ResourceSnapshot(free_vram=memory[0], total_vram=memory[1]))
        if (image.nbytes + target_bytes + 128 * 1024 + _REPAIR_BYTES > budget or
                max(image.nbytes, target_bytes) > self._device.max_mem_alloc_size):
            return False
        context, _, _, _ = _verified_program(self._cl, self._device)
        cl = self._cl
        self._buffers = (cl.Buffer(context, cl.mem_flags.READ_ONLY, image.nbytes),
                         cl.Buffer(context, cl.mem_flags.WRITE_ONLY, target_bytes),
                         cl.Buffer(context, cl.mem_flags.READ_WRITE, _REPAIR_BYTES))
        self._shape = image.shape, dsize
        return True

    def note_cpu_warp(self):
        with _queue_lock:
            self.cpu_warps += 1

    def warp(self, image, matrix, dsize, should_cancel):
        """Return exact pixels when the device, layout and geometry are admitted."""
        global _program
        from core.registration import RegistrationCancelled, _check_cancel

        width, height = dsize
        if (self._disabled or image.dtype != np.uint8 or image.shape[2:] != (3,)
                or not image.flags.c_contiguous or width * height < _MIN_PIXELS
                or max(width, height, *image.shape[:2]) > _MAX_SIDE):
            return None
        geometry = _warp_geometry(matrix, width, height)
        if geometry is None:
            return None
        inverse, tolerance = geometry
        # A translation exactly on a bin boundary would flag every pixel.
        if (inverse[0, 1] == inverse[1, 0] == inverse[2, 0] == inverse[2, 1] == 0
                and inverse[0, 0] == inverse[1, 1] == inverse[2, 2]
                and np.any(np.abs((32 * inverse[:2, 2] / inverse[2, 2]) % 1 - .5) <= tolerance)):
            return None
        with _queue_lock:
            _check_cancel(should_cancel)
            if self._disabled:
                return None
            queue = None
            try:
                if self._cl is None and not self._initialize():
                    self._disabled = True
                    return None
                if cv2.ipp.useIPP_NotExact() or self._round_mode() != 0:
                    return None
                if self._shape != (image.shape, dsize) and not self._workspace(image, dsize):
                    self._disabled = True
                    return None
                cl = self._cl
                _, queue, kernel, weights = _verified_program(cl, self._device)
                source, target, repair = self._buffers
                for row in range(0, image.shape[0], _BAND_ROWS):
                    _check_cancel(should_cancel)
                    band = image[row:row + _BAND_ROWS]
                    cl.enqueue_copy(queue, source, band, dst_offset=row * image.shape[1] * 3)
                result = np.empty((height, width, 3), np.uint8)
                cl.enqueue_fill_buffer(queue, repair, np.int32(0), 0, 4)
                edges = np.empty(_REPAIR_LIMIT + 1, np.int32)
                for row in range(0, height, _BAND_ROWS):
                    _check_cancel(should_cancel)
                    band = result[row:row + _BAND_ROWS]
                    kernel(queue, (width, len(band)), None, source, target, weights, repair,
                           np.int32(image.shape[1]), np.int32(image.shape[0]), np.int32(width),
                           np.int32(row), np.float64(tolerance), *inverse.flat)
                    cl.enqueue_copy(queue, band, target)
                cl.enqueue_copy(queue, edges, repair)
                _check_cancel(should_cancel)
                if edges[0] > _REPAIR_LIMIT:
                    return None
                if edges[0]:
                    _repair_rounding_edges(image, inverse, result, edges[1:1 + edges[0]], should_cancel)
                if self.gpu_warps == 0:
                    self.device_name = opencl_device_name(self._device)
                    _log.info('Deferred Lanczos4 warp on %s', self.device_name)
                self.gpu_warps += 1
                return result
            except Exception as exc:
                if queue is not None:
                    try:
                        queue.finish()
                    except Exception:
                        _program = None
                if isinstance(exc, RegistrationCancelled):
                    raise
                self.close()
                self._disabled = True
                _log.info('Deferred Lanczos4 warp using OpenCV: %s', exc)
                return None
