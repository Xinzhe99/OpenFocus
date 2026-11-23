"""Bounded, optional OpenCL Lanczos4 warping for large 8-bit and 16-bit frames."""
import ctypes
import logging
import math
import os
from collections import Counter
from string import Template
from threading import Lock

import cv2
import numpy as np

from core._ecc_area_gpu import _cpu_contract
from fusion_methods._openfocus_gpu import _device, _device_cache_key, _free_vram, opencl_device_name

_MIN_PIXELS = 1024 * 1024
_BAND_ROWS = (512, 256, 128, 64, 32, 16)
_REPAIR_LIMIT = 32
_REPAIR_BYTES = (_REPAIR_LIMIT + 1) * np.dtype(np.int32).itemsize
_MAX_SIDE = 32766
_MAX_WORKSPACE_BYTES = 64 * 1024 ** 2
# Source rows and columns kept around the samples of one band.
_HALO = 8
_queue_lock = Lock()
_program = None
_log = logging.getLogger(__name__)

# The coordinates are quantized before interpolation, as in OpenCV's
# warpPerspective/remap. OpenCV sums an interior sample one source row at a
# time and rounds each row before adding it. A sample touching the image
# border is summed flat over its in-image taps. The 8-bit sums are integers
# and do not depend on the order.
_BODY = '''
__kernel void $name(__global const $pixel *src, __global $pixel *dst,
                    __global const $coeff *weights, volatile __global int *repair,
                    int sw, int sh, int bw, int bx, int by,
                    int dw, int full_w, int y0, int x0, double tolerance,
                    double a, double b, double c, double d, double e, double f,
                    double g, double h, double i) {
    int lx = get_global_id(0), ly = get_global_id(1), x = x0 + lx, y = y0 + ly;
    double scale = 32.0 / ((h * y + i) + g * x);
    double fx = ((b * y + c) + a * x) * scale;
    double fy = ((e * y + f) + d * x) * scale;
    if (fabs(fx - floor(fx) - 0.5) <= tolerance ||
        fabs(fy - floor(fy) - 0.5) <= tolerance) {
        int slot = atomic_inc(repair);
        if (slot < REPAIR_LIMIT)
            repair[slot + 1] = y * full_w + x;
        return;
    }
    int qx = convert_int_rte(fx), qy = convert_int_rte(fy);
    int sx = (qx >> 5) - 3, sy = (qy >> 5) - 3;
    __global const $coeff *coeffs = weights + (((qy & 31) * 32 + (qx & 31)) * 64);
    __global $pixel *out = dst + ((size_t)(ly * dw + lx) * $ch);
    $vec sum = ($vec)($zero);
    if (sx >= 0 && sx + 7 < sw && sy >= 0 && sy + 7 < sh) {
        __global const $pixel *row = src + ((size_t)(sy - by) * bw + (sx - bx)) * $ch;
        for (int j = 0; j < 8; ++j, row += (size_t)bw * $ch) {
            __global const $coeff *w = coeffs + j * 8;
$rows
            sum += r;
        }
    } else {
        for (int j = 0; j < 8; ++j) {
            int yy = sy + j;
            if (yy < 0 || yy >= sh) continue;
            __global const $pixel *row = src + (size_t)(yy - by) * bw * $ch;
            for (int k = 0; k < 8; ++k) {
                int xx = sx + k;
                if (xx < 0 || xx >= sw) continue;
                sum += convert_$vec($sample)$sw * coeffs[j * 8 + k];
            }
        }
    }
    $store
}
'''

# Kernel name, pixel, accumulator, coefficient type, channels, reversed storage.
_VARIANTS = tuple(
    (f'warp_{label}_c{channels}{"_rev" if reverse else ""}', pixel, acc, coeff, channels, reverse)
    for label, pixel, acc, coeff in (('u8', 'uchar', 'int', 'short'),
                                     ('u16', 'ushort', 'float', 'float'))
    for channels, reverse in ((1, False), (3, False), (3, True), (4, False)))


def _kernel_source(name, pixel, acc, coeff, channels, reverse):
    """Build one kernel. Its channels are compiled in and moved as one vector."""
    suffix = '' if channels == 1 else str(channels)
    vec = acc + suffix
    swizzle = '.zyx' if reverse else ''
    sample = ((lambda index: f'vload{suffix}({index}, row)') if suffix
              else (lambda index: f'row[{index}]'))
    result = ('convert_' + ('uchar' if pixel == 'uchar' else 'ushort') + suffix + '_sat'
              + ('_rte' if acc == 'float' else ''))
    shift = ' >> 15' if acc == 'int' else ''
    rows = '\n'.join(
        ('            ' + vec + ' r = ' if k == 0 else '            r += ')
        + f'convert_{vec}({sample(k)}){swizzle} * w[{k}];' for k in range(8))
    store = (f'vstore{suffix}({result}(sum{shift}), 0, out);' if suffix
             else f'out[0] = {result}(sum{shift});')
    return Template(_BODY).substitute(
        name=name, pixel=pixel, coeff=coeff, ch=channels, vec=vec, sw=swizzle,
        sample=sample('xx - bx'), zero='1 << 14' if acc == 'int' else '0.0f',
        rows=rows, store=store)


_SOURCE = ('#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n'
           '#pragma OPENCL FP_CONTRACT OFF\n'
           + ''.join(_kernel_source(*variant) for variant in _VARIANTS))

# Interior, rotated and partly out-of-image transforms proved on each device.
_CHECK_MATRICES = (
    np.array(((1.002, -.004, 5.37), (.003, .998, -3.81), (1e-5, -2e-5, 1))),
    np.array(((.94, .08, -7.4), (-.06, 1.05, 6.7), (4e-4, -3e-4, 1))),
    np.array(((1., 0., 61.), (0., 1., 23.), (0., 0., 1.))),
)


def _phases():
    # Read the 32 one-dimensional phases from the installed OpenCV build.
    impulse = np.zeros((25, 25), np.float32)
    impulse[12, 12] = 1
    coords = np.empty((32, 8, 2), np.int16)
    coords[..., 0] = np.arange(15, 7, -1)
    coords[..., 1] = 12
    phases = np.broadcast_to(np.arange(32, dtype=np.uint16)[:, None], (32, 8)).copy()
    return cv2.remap(impulse, coords, phases, cv2.INTER_LANCZOS4)


def _coefficients():
    # OpenCV resamples 8-bit samples through a 15-bit fixed-point table.
    one = _phases()
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


def _weights():
    # OpenCV resamples 16-bit samples through the products of its float phases.
    one = _phases()
    table = np.empty((32, 32, 8, 8), np.float32)
    for y in range(32):
        for x in range(32):
            table[y, x] = np.outer(one[y], one[x])
    return table


def _warp_geometry(matrix, x0, y0, width, height):
    inverse = cv2.invert(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    if not inverse[0] or not np.isfinite(inverse[1]).all():
        return None
    corners = np.array(((x0, y0, 1), (x0 + width - 1, y0, 1),
                        (x0, y0 + height - 1, 1), (x0 + width - 1, y0 + height - 1, 1)))
    points = inverse[1] @ corners.T
    min_w = points[2].min()
    if min_w <= 1e-6 or np.max(np.abs(points[:2] / points[2])) >= _MAX_SIDE:
        return None
    # Enclose the host's block/FMA evaluation and OpenCL's double division.
    # The bound includes cancellation in both sums. Rounding can differ only
    # if a coordinate is within this bound of a half-integral bin boundary.
    magnitude = np.abs(inverse[1]) @ corners[-1]
    eps = 16 * np.finfo(np.float64).eps
    denom = min_w - eps * magnitude[2]
    if denom <= 0:
        return None
    tolerance = 32 * eps * max(magnitude[:2]) / denom * (1 + magnitude[2] / denom)
    if not math.isfinite(tolerance) or tolerance >= 0.25:
        return None
    return inverse[1], tolerance


def _source_window(inverse, rect, src_w, src_h):
    """Source rows and columns that a destination rectangle can sample."""
    x0, y0, width, height = rect
    corners = np.array(((x0, y0, 1), (x0 + width - 1, y0, 1),
                        (x0, y0 + height - 1, 1), (x0 + width - 1, y0 + height - 1, 1)))
    points = inverse @ corners.T
    sx = points[0] / points[2]
    sy = points[1] / points[2]
    left = max(0, int(math.floor(sx.min())) - _HALO)
    top = max(0, int(math.floor(sy.min())) - _HALO)
    # A rectangle mapped outside the image has no samples and stays empty.
    return (left, top,
            max(left, min(src_w, int(math.ceil(sx.max())) + 1 + _HALO)),
            max(top, min(src_h, int(math.ceil(sy.max())) + 1 + _HALO)))


def _band_plan(inverse, rect, src_size, channels, itemsize, budget, max_alloc, fixed_bytes):
    """Split the destination into bands whose source windows fit the VRAM budget."""
    x0, y0, width, height = rect
    src_h, src_w = src_size
    for rows in _BAND_ROWS:
        bands = []
        for start in range(0, height, rows):
            count = min(rows, height - start)
            window = _source_window(inverse, (x0, y0 + start, width, count), src_w, src_h)
            bands.append((start, count, *window))
        areas = [(window[2] - window[0]) * (window[3] - window[1]) for _, _, *window in bands]
        # A window narrower than the source is copied into a staging band.
        staged = max(area if window[0] > 0 or window[2] < src_w else 0
                     for area, (_, _, *window) in zip(areas, bands))
        src_bytes = max(areas) * channels * itemsize
        rows_bytes = min(rows, height) * width * channels * itemsize
        if (src_bytes + rows_bytes + fixed_bytes <= budget
                and max(src_bytes, rows_bytes) <= max_alloc):
            return bands, src_bytes, rows_bytes, staged * channels
    return None


def _repair_rounding_edges(image, inverse, result, coordinates, rect, dsize, should_cancel,
                           *, bgr_rev=False):
    """Evaluate flagged samples with OpenCV's own block arithmetic."""
    from core.registration import _check_cancel

    x0, y0, width, _ = rect
    full_w, full_h = dsize
    # Match WarpPerspectiveInvoker's horizontal blocks in OpenCV 5.0.0.
    # Rebase each row/block using its exact first-pixel sums. OpenCV then
    # evaluates the same local x positions with its active SIMD or scalar path.
    block_width = min(1024 // min(16, full_h), full_w)
    blocks = set()
    for offset in coordinates:
        y, x = divmod(int(offset), full_w)
        blocks.add((y, x // block_width * block_width))
    shifted = inverse.copy()
    shifted[:, 1] = 0
    for y, x in blocks:
        _check_cancel(should_cancel)
        shifted[:, 2] = (inverse[:, 0] * x + inverse[:, 1] * y) + inverse[:, 2]
        cols = min(block_width, full_w - x)
        row = cv2.warpPerspective(image, shifted, (cols, 1),
                                  flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
        if bgr_rev:
            row = row[..., ::-1]
        left, right = max(x, x0), min(x + cols, x0 + width)
        result[y - y0, left - x0:right - x0] = row[0, left - x:right - x]


def _self_check(cl, device, program):
    """Prove the active interpolation paths, or raise.

    Covers borders, bands, phases, channel counts, reversed storage and
    destination regions. The host repair is proved on block edges too.
    """
    rng = np.random.default_rng(4916)
    dsize = (417, 293)
    rects = ((0, 0, *dsize), (151, 97, 233, 151))
    block = min(1024 // min(16, dsize[1]), dsize[0])
    columns = (0, block - 1, block, 150, 151, 383, 416)
    probe = ExactPerspectiveGpu()
    probe._cl, probe._device = cl, device
    try:
        for dtype, high in ((np.uint8, 256), (np.uint16, 65536)):
            for channels, reverse in ((1, False), (3, False), (3, True), (4, False)):
                shape = (317, 461) if channels == 1 else (317, 461, channels)
                sample = rng.integers(0, high, shape, dtype)
                logical = np.ascontiguousarray(sample[..., ::-1]) if reverse else sample
                for matrix in _CHECK_MATRICES:
                    geometry = _warp_geometry(matrix, 0, 0, *dsize)
                    if geometry is None:
                        raise RuntimeError('OpenCL Lanczos4 self-check geometry was declined')
                    inverse, tolerance = geometry
                    expected = cv2.warpPerspective(logical, matrix, dsize,
                                                   flags=cv2.INTER_LANCZOS4)
                    for rect in rects:
                        x0, y0, width, height = rect
                        wanted = expected[y0:y0 + height, x0:x0 + width]
                        actual = probe._dispatch(program, sample, inverse, tolerance,
                                                 reverse, rect, dsize, None)
                        if actual is None or not np.array_equal(actual, wanted):
                            raise RuntimeError('OpenCL Lanczos4 self-check differed from OpenCV')
                        offsets = [y * dsize[0] + x for y in (y0, y0 + height - 1, y0 + 150)
                                   for x in columns if x0 <= x < x0 + width]
                        _repair_rounding_edges(logical, inverse, actual, offsets, rect, dsize, None)
                        if not np.array_equal(actual, wanted):
                            raise RuntimeError('OpenCL Lanczos4 self-check differed from OpenCV')
    finally:
        probe.close()


def _verified_program(cl, device):
    global _program
    key = _device_cache_key(device)
    if _program is not None and _program[0] == key:
        return _program[1]
    context = cl.Context(devices=[device])
    queue = cl.CommandQueue(context)
    program = cl.Program(context, _SOURCE).build(options=[f'-DREPAIR_LIMIT={_REPAIR_LIMIT}'])
    kernels = {(2 if pixel == 'ushort' else 1, channels, reverse): cl.Kernel(program, name)
               for name, pixel, _, _, channels, reverse in _VARIANTS}
    flags = cl.mem_flags
    weights = (cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=_coefficients()),
               cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=_weights()))
    built = (context, queue, kernels, *weights)
    try:
        _self_check(cl, device, built)
        queue.finish()
    except Exception:
        for table in weights:
            table.release()
        raise
    _program = (key, built)
    return built


class ExactPerspectiveGpu:
    """One scope's device buffers and dispatch counts."""

    def __init__(self):
        self._cl = self._device = self._buffers = None
        self._disabled = False
        self._round_mode = None
        self.device_name = ''
        self.gpu_warps = 0
        self.cpu_warps = 0
        self._gpu_by_method = Counter()
        self._cpu_by_method = Counter()

    def close(self):
        buffers, self._buffers = self._buffers, None
        for buffer in (buffers or ())[2:]:
            buffer.release()

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

    def _workspace(self, context, cl, src_bytes, dst_bytes):
        """Reuse this scope's device buffers while the admitted bands fit."""
        # OpenCL rejects empty buffers, and no source band is in view for
        # a region mapped outside the image.
        src_bytes, dst_bytes = max(1, src_bytes), max(1, dst_bytes)
        buffers = self._buffers
        if buffers is None or buffers[0] < src_bytes or buffers[1] < dst_bytes:
            self.close()
            flags = cl.mem_flags
            self._buffers = (src_bytes, dst_bytes,
                             cl.Buffer(context, flags.READ_ONLY, src_bytes),
                             cl.Buffer(context, flags.WRITE_ONLY, dst_bytes),
                             cl.Buffer(context, flags.READ_WRITE, _REPAIR_BYTES))
            buffers = self._buffers
        return buffers[2:]

    @property
    def devices_by_method(self):
        return {method: ('CPU / ' if self._cpu_by_method[method] else '') + self.device_name
                for method in self._gpu_by_method if method}

    def note_cpu_warp(self, method=''):
        with _queue_lock:
            self.cpu_warps += 1
            self._cpu_by_method[method] += 1

    def warp(self, image, matrix, dsize, should_cancel, *, bgr_rev=False, region=None, method=''):
        """Return exact pixels when the device, layout and geometry are admitted.

        ``bgr_rev`` reads a contiguous RGB frame as its reversed BGR view.
        ``region`` selects a rectangle of the full-frame warp.
        """
        global _program
        from core.registration import RegistrationCancelled, _check_cancel

        width, height = dsize
        x0, y0, rw, rh = region if region is not None else (0, 0, width, height)
        channels = image.shape[2] if image.ndim == 3 else 1
        if (self._disabled or image.dtype not in (np.uint8, np.uint16)
                or image.ndim not in (2, 3) or channels not in (1, 3, 4)
                or not image.flags.c_contiguous or (bgr_rev and channels != 3)
                or rw * rh < _MIN_PIXELS
                or max(width, height, rw, rh, *image.shape[:2]) > _MAX_SIDE
                or x0 < 0 or y0 < 0 or x0 + rw > width or y0 + rh > height):
            return None
        geometry = _warp_geometry(matrix, x0, y0, rw, rh)
        if geometry is None:
            return None
        inverse, tolerance = geometry
        # A translation exactly on a bin boundary would flag every pixel.
        if (inverse[0, 1] == inverse[1, 0] == inverse[2, 0] == inverse[2, 1] == 0
                and inverse[0, 0] == inverse[1, 1] == inverse[2, 2]
                and np.any(np.abs((32 * (inverse[:2, 2] + inverse[0, 0] * np.array((x0, y0)))
                                   / inverse[2, 2]) % 1 - .5) <= tolerance)):
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
                program = _verified_program(self._cl, self._device)
                queue = program[1]
                result = self._dispatch(program, image, inverse, tolerance, bgr_rev,
                                        (x0, y0, rw, rh), (width, height), should_cancel)
                if result is None:
                    return None
                if self.gpu_warps == 0:
                    self.device_name = opencl_device_name(self._device)
                    _log.info('Deferred Lanczos4 warp on %s', self.device_name)
                self.gpu_warps += 1
                self._gpu_by_method[method] += 1
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

    def _dispatch(self, program, image, inverse, tolerance, bgr_rev, rect, dsize, should_cancel):
        from core import resource_policy as policy
        from core.registration import _check_cancel

        context, queue, kernels, weights8, weights16 = program
        cl = self._cl
        x0, y0, width, height = rect
        src_h, src_w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        itemsize = image.dtype.itemsize
        weights = weights8 if itemsize == 1 else weights16
        memory = _free_vram(self._device)
        if memory is None:
            self._disabled = True
            return None
        snapshot = policy.ResourceSnapshot(free_vram=memory[0], total_vram=memory[1])
        budget = min(_MAX_WORKSPACE_BYTES, policy.vram_budget(snapshot))
        plan = _band_plan(inverse, rect, image.shape[:2], channels, itemsize, budget,
                          int(self._device.max_mem_alloc_size), weights8.size + weights16.size + _REPAIR_BYTES)
        if plan is None:
            self._disabled = True
            return None
        bands, src_bytes, dst_bytes, staged = plan
        policy.require_memory(width * height * channels * itemsize + staged * itemsize)
        _check_cancel(should_cancel)
        result = np.empty((height, width) if channels == 1 else (height, width, channels), image.dtype)
        stage = np.empty(staged, image.dtype) if staged else None
        d_src, d_dst, d_repair = self._workspace(context, cl, src_bytes, dst_bytes)
        kernel = kernels[(itemsize, channels, bgr_rev)]
        cl.enqueue_fill_buffer(queue, d_repair, np.int32(0), 0, 4)
        for ly0, rows, sx0, sy0, sx1, sy1 in bands:
            _check_cancel(should_cancel)
            if sx0 >= sx1 or sy0 >= sy1:
                result[ly0:ly0 + rows] = 0
                continue
            band = image[sy0:sy1]
            if sx0 > 0 or sx1 < src_w:
                count = (sy1 - sy0) * (sx1 - sx0) * channels
                shape = ((sy1 - sy0, sx1 - sx0, channels) if image.ndim == 3
                         else (sy1 - sy0, sx1 - sx0))
                band = stage[:count].reshape(shape)
                np.copyto(band, image[sy0:sy1, sx0:sx1])
            upload = cl.enqueue_copy(queue, d_src, band, is_blocking=False)
            kernel(
                queue, (width, rows), None, d_src, d_dst, weights, d_repair,
                np.int32(src_w), np.int32(src_h),
                np.int32(sx1 - sx0), np.int32(sx0), np.int32(sy0),
                np.int32(width), np.int32(dsize[0]), np.int32(y0 + ly0), np.int32(x0),
                np.float64(tolerance), *inverse.flat,
            )
            cl.enqueue_copy(queue, result[ly0:ly0 + rows], d_dst, is_blocking=True)
            del upload, band
        edges = np.empty(_REPAIR_LIMIT + 1, np.int32)
        cl.enqueue_copy(queue, edges, d_repair, is_blocking=True)
        _check_cancel(should_cancel)
        if edges[0] > _REPAIR_LIMIT:
            return None
        if edges[0]:
            _repair_rounding_edges(image, inverse, result, edges[1:1 + edges[0]], rect,
                                   dsize, should_cancel, bgr_rev=bgr_rev)
        return result
