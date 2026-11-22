"""Native Numba kernels and bounded band execution for image pipeline utilities.

Import lazily after memory admission; callers keep a pure-NumPy/OpenCV fallback
when compilation or startup memory cannot be admitted.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
import sys
from threading import Event

from llvmlite import ir
from numba import njit as _numba_njit, types
from numba.extending import intrinsic
import numpy as np


_BAND_ROWS = 128
_MIN_PARALLEL_ELEMENTS = 500_000


def _njit(*args, cache=True, **kwargs):
    """Compile with optional on-disk caching, falling back to in-memory JIT."""
    def decorate(function):
        if cache:
            try:
                return _numba_njit(*args, cache=True, **kwargs)(function)
            except (RuntimeError, OSError):
                pass
        return _numba_njit(*args, **kwargs)(function)
    return decorate


@intrinsic
def _rint_f32(typing_context, value):
    """Round float32 to nearest even integer via LLVM llvm.rint."""
    if value != types.float32:
        return None
    signature = types.float32(types.float32)

    def codegen(context, builder, sig, arguments):
        f32 = ir.FloatType()
        fn = builder.module.declare_intrinsic(
            'llvm.rint', [f32], ir.FunctionType(f32, [f32]))
        return builder.call(fn, arguments)

    return signature, codegen


@intrinsic
def _rint_f64(typing_context, value):
    """Round float64 to nearest even integer via LLVM llvm.rint."""
    if value != types.float64:
        return None
    signature = types.float64(types.float64)

    def codegen(context, builder, sig, arguments):
        f64 = ir.DoubleType()
        fn = builder.module.declare_intrinsic(
            'llvm.rint', [f64], ir.FunctionType(f64, [f64]))
        return builder.call(fn, arguments)

    return signature, codegen


# ---------------------------------------------------------------------------
# Exposure mean reductions
# ---------------------------------------------------------------------------
@_njit(nogil=True, fastmath=False)
def sum_u16_2d_u64(arr, start, stop):
    total = np.uint64(0)
    width = arr.shape[1]
    if width <= 65535:
        for y in range(start, stop):
            row = arr[y]
            part = np.uint32(0)
            for x in range(width):
                part = np.uint32(part + np.uint32(row[x]))
            total = np.uint64(total + np.uint64(part))
    else:
        chunk = 32768
        for y in range(start, stop):
            row = arr[y]
            for base in range(0, width, chunk):
                sub = row[base:min(base + chunk, width)]
                part = np.uint32(0)
                for x in range(sub.shape[0]):
                    part = np.uint32(part + np.uint32(sub[x]))
                total = np.uint64(total + np.uint64(part))
    return total


@_njit(nogil=True, fastmath=False)
def sum_u8_2d_u64(arr, start, stop):
    total = np.uint64(0)
    width = arr.shape[1]
    if width <= 16777215:
        for y in range(start, stop):
            row = arr[y]
            part = np.uint32(0)
            for x in range(width):
                part = np.uint32(part + np.uint32(row[x]))
            total = np.uint64(total + np.uint64(part))
    else:
        chunk = 65536
        for y in range(start, stop):
            row = arr[y]
            for base in range(0, width, chunk):
                sub = row[base:min(base + chunk, width)]
                part = np.uint32(0)
                for x in range(sub.shape[0]):
                    part = np.uint32(part + np.uint32(sub[x]))
                total = np.uint64(total + np.uint64(part))
    return total


@_njit(nogil=True, fastmath=False)
def sum_u16_3ch_u64(arr, start, stop):
    total = np.uint64(0)
    width = arr.shape[1]
    if width <= 21845:
        for y in range(start, stop):
            row = arr[y]
            part = np.uint32(0)
            for x in range(width):
                part = np.uint32(
                    part
                    + np.uint32(row[x, 0])
                    + np.uint32(row[x, 1])
                    + np.uint32(row[x, 2])
                )
            total = np.uint64(total + np.uint64(part))
    else:
        chunk = 10922
        for y in range(start, stop):
            row = arr[y]
            for base in range(0, width, chunk):
                sub = row[base:min(base + chunk, width)]
                part = np.uint32(0)
                for x in range(sub.shape[0]):
                    part = np.uint32(
                        part
                        + np.uint32(sub[x, 0])
                        + np.uint32(sub[x, 1])
                        + np.uint32(sub[x, 2])
                    )
                total = np.uint64(total + np.uint64(part))
    return total


@_njit(nogil=True, fastmath=False)
def sum_u8_3ch_u64(arr, start, stop):
    total = np.uint64(0)
    width = arr.shape[1]
    if width <= 5592405:
        for y in range(start, stop):
            row = arr[y]
            part = np.uint32(0)
            for x in range(width):
                part = np.uint32(
                    part
                    + np.uint32(row[x, 0])
                    + np.uint32(row[x, 1])
                    + np.uint32(row[x, 2])
                )
            total = np.uint64(total + np.uint64(part))
    else:
        chunk = 32768
        for y in range(start, stop):
            row = arr[y]
            for base in range(0, width, chunk):
                sub = row[base:min(base + chunk, width)]
                part = np.uint32(0)
                for x in range(sub.shape[0]):
                    part = np.uint32(
                        part
                        + np.uint32(sub[x, 0])
                        + np.uint32(sub[x, 1])
                        + np.uint32(sub[x, 2])
                    )
                total = np.uint64(total + np.uint64(part))
    return total


@_njit(nogil=True, fastmath=False)
def sum_u16_2d_norm_f64(arr, start, stop):
    scale = np.float32(255.0 / 65535.0)
    total = np.float64(0.0)
    width = arr.shape[1]
    width4 = width & ~3
    for y in range(start, stop):
        row = arr[y]
        s0 = np.float64(0.0)
        s1 = np.float64(0.0)
        s2 = np.float64(0.0)
        s3 = np.float64(0.0)
        for x in range(0, width4, 4):
            s0 += np.float64(np.float32(row[x]) * scale)
            s1 += np.float64(np.float32(row[x + 1]) * scale)
            s2 += np.float64(np.float32(row[x + 2]) * scale)
            s3 += np.float64(np.float32(row[x + 3]) * scale)
        row_sum = (s0 + s1) + (s2 + s3)
        for x in range(width4, width):
            row_sum += np.float64(np.float32(row[x]) * scale)
        total += row_sum
    return total


@_njit(nogil=True, fastmath=False)
def sum_u16_3ch_norm_f64(arr, start, stop):
    scale = np.float32(255.0 / 65535.0)
    total = np.float64(0.0)
    width = arr.shape[1]
    for y in range(start, stop):
        row = arr[y]
        s0 = np.float64(0.0)
        s1 = np.float64(0.0)
        s2 = np.float64(0.0)
        for x in range(width):
            s0 += np.float64(np.float32(row[x, 0]) * scale)
            s1 += np.float64(np.float32(row[x, 1]) * scale)
            s2 += np.float64(np.float32(row[x, 2]) * scale)
        total += (s0 + s1) + s2
    return total


@_njit(nogil=True, fastmath=False)
def sum_f32_2d_f64(arr, start, stop):
    total = np.float64(0.0)
    width = arr.shape[1]
    width4 = width & ~3
    for y in range(start, stop):
        row = arr[y]
        s0 = np.float64(0.0)
        s1 = np.float64(0.0)
        s2 = np.float64(0.0)
        s3 = np.float64(0.0)
        for x in range(0, width4, 4):
            s0 += np.float64(row[x])
            s1 += np.float64(row[x + 1])
            s2 += np.float64(row[x + 2])
            s3 += np.float64(row[x + 3])
        row_sum = (s0 + s1) + (s2 + s3)
        for x in range(width4, width):
            row_sum += np.float64(row[x])
        total += row_sum
    return total


@_njit(nogil=True, fastmath=False)
def sum_f32_3ch_f64(arr, start, stop):
    total = np.float64(0.0)
    width = arr.shape[1]
    for y in range(start, stop):
        row = arr[y]
        s0 = np.float64(0.0)
        s1 = np.float64(0.0)
        s2 = np.float64(0.0)
        for x in range(width):
            s0 += np.float64(row[x, 0])
            s1 += np.float64(row[x, 1])
            s2 += np.float64(row[x, 2])
        total += (s0 + s1) + s2
    return total


# ---------------------------------------------------------------------------
# Exposure gain kernels
# ---------------------------------------------------------------------------
@_njit(nogil=True, fastmath=False)
def apply_lut_2d_rows(src, lut, dst, start, stop):
    width = src.shape[1]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            drow[x] = lut[srow[x]]


@_njit(nogil=True, fastmath=False)
def apply_lut_bgr_rev_rows(rgb_src, lut, bgr_dst, start, stop):
    width = rgb_src.shape[1]
    for y in range(start, stop):
        srow = rgb_src[y]
        drow = bgr_dst[y]
        for x in range(width):
            drow[x, 0] = lut[srow[x, 2]]
            drow[x, 1] = lut[srow[x, 1]]
            drow[x, 2] = lut[srow[x, 0]]


@_njit(nogil=True, fastmath=False)
def apply_lut_3d_rows(src, lut, dst, start, stop):
    width = src.shape[1]
    channels = src.shape[2]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            for c in range(channels):
                drow[x, c] = lut[srow[x, c]]


@_njit(nogil=True, fastmath=False)
def apply_lut_bgra_rows(src, lut, dst, start, stop):
    width = src.shape[1]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            drow[x, 0] = lut[srow[x, 0]]
            drow[x, 1] = lut[srow[x, 1]]
            drow[x, 2] = lut[srow[x, 2]]
            drow[x, 3] = srow[x, 3]


@_njit(nogil=True, fastmath=False)
def scale_clip_f32_2d_rows(src, gain, limit, dst, start, stop):
    zero = np.float32(0.0)
    gain32 = np.float32(gain)
    limit32 = np.float32(limit)
    width = src.shape[1]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            v = np.float32(srow[x] * gain32)
            if v < zero:
                v = zero
            elif v > limit32:
                v = limit32
            drow[x] = v


@_njit(nogil=True, fastmath=False)
def scale_clip_f32_bgra_rows(src, gain, limit, dst, start, stop):
    zero = np.float32(0.0)
    gain32 = np.float32(gain)
    limit32 = np.float32(limit)
    width = src.shape[1]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            for c in range(3):
                v = np.float32(srow[x, c] * gain32)
                if v < zero:
                    v = zero
                elif v > limit32:
                    v = limit32
                drow[x, c] = v
            drow[x, 3] = srow[x, 3]


# ---------------------------------------------------------------------------
# Output quantization kernels (cache=False because of @intrinsic _rint_*)
# ---------------------------------------------------------------------------
@_njit(nogil=True, fastmath=False, cache=False)
def quantize_f32_to_u16_2d(src, dst, start, stop):
    zero = np.float32(0.0)
    hi = np.float32(255.0)
    scale = np.float32(65535.0 / 255.0)
    width = src.shape[1]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            v = srow[x]
            if v <= zero or v != v:
                drow[x] = np.uint16(0)
            elif v >= hi:
                drow[x] = np.uint16(65535)
            else:
                drow[x] = np.uint16(_rint_f32(np.float32(v * scale)))


@_njit(nogil=True, fastmath=False, cache=False)
def quantize_f32_to_u16_3d(src, dst, start, stop):
    zero = np.float32(0.0)
    hi = np.float32(255.0)
    scale = np.float32(65535.0 / 255.0)
    width = src.shape[1]
    channels = src.shape[2]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            for c in range(channels):
                v = srow[x, c]
                if v <= zero or v != v:
                    drow[x, c] = np.uint16(0)
                elif v >= hi:
                    drow[x, c] = np.uint16(65535)
                else:
                    drow[x, c] = np.uint16(_rint_f32(np.float32(v * scale)))


@_njit(nogil=True, fastmath=False, cache=False)
def quantize_f32_to_u8_2d(src, dst, start, stop):
    zero = np.float32(0.0)
    hi = np.float32(255.0)
    width = src.shape[1]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            v = srow[x]
            if v <= zero or v != v:
                drow[x] = np.uint8(0)
            elif v >= hi:
                drow[x] = np.uint8(255)
            else:
                drow[x] = np.uint8(_rint_f32(v))


@_njit(nogil=True, fastmath=False, cache=False)
def quantize_f32_to_u8_3d(src, dst, start, stop):
    zero = np.float32(0.0)
    hi = np.float32(255.0)
    width = src.shape[1]
    channels = src.shape[2]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            for c in range(channels):
                v = srow[x, c]
                if v <= zero or v != v:
                    drow[x, c] = np.uint8(0)
                elif v >= hi:
                    drow[x, c] = np.uint8(255)
                else:
                    drow[x, c] = np.uint8(_rint_f32(v))


@_njit(nogil=True, fastmath=False, cache=False)
def quantize_f64_to_u16_2d(src, dst, start, stop):
    scale = np.float64(65535.0 / 255.0)
    width = src.shape[1]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            v = srow[x]
            if v <= 0.0 or v != v:
                drow[x] = np.uint16(0)
            elif v >= 255.0:
                drow[x] = np.uint16(65535)
            else:
                drow[x] = np.uint16(_rint_f64(v * scale))


@_njit(nogil=True, fastmath=False, cache=False)
def quantize_f64_to_u16_3d(src, dst, start, stop):
    scale = np.float64(65535.0 / 255.0)
    width = src.shape[1]
    channels = src.shape[2]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            for c in range(channels):
                v = srow[x, c]
                if v <= 0.0 or v != v:
                    drow[x, c] = np.uint16(0)
                elif v >= 255.0:
                    drow[x, c] = np.uint16(65535)
                else:
                    drow[x, c] = np.uint16(_rint_f64(v * scale))


@_njit(nogil=True, fastmath=False, cache=False)
def quantize_f64_to_u8_2d(src, dst, start, stop):
    width = src.shape[1]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            v = srow[x]
            if v <= 0.0 or v != v:
                drow[x] = np.uint8(0)
            elif v >= 255.0:
                drow[x] = np.uint8(255)
            else:
                drow[x] = np.uint8(_rint_f64(v))


@_njit(nogil=True, fastmath=False, cache=False)
def quantize_f64_to_u8_3d(src, dst, start, stop):
    width = src.shape[1]
    channels = src.shape[2]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            for c in range(channels):
                v = srow[x, c]
                if v <= 0.0 or v != v:
                    drow[x, c] = np.uint8(0)
                elif v >= 255.0:
                    drow[x, c] = np.uint8(255)
                else:
                    drow[x, c] = np.uint8(_rint_f64(v))


# ---------------------------------------------------------------------------
# Display conversion kernels (exact match for cv2.convertScaleAbs(1/257))
# ---------------------------------------------------------------------------
@_njit(nogil=True, fastmath=False)
def u16_to_display_u8_2d(src, dst, start, stop):
    width = src.shape[1]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            drow[x] = np.uint8((np.uint32(srow[x]) + np.uint32(128)) // np.uint32(257))


@_njit(nogil=True, fastmath=False)
def u16_bgr_rev_to_display_u8(rgb_src, bgr_dst, start, stop):
    width = rgb_src.shape[1]
    for y in range(start, stop):
        srow = rgb_src[y]
        drow = bgr_dst[y]
        for x in range(width):
            drow[x, 0] = np.uint8((np.uint32(srow[x, 2]) + np.uint32(128)) // np.uint32(257))
            drow[x, 1] = np.uint8((np.uint32(srow[x, 1]) + np.uint32(128)) // np.uint32(257))
            drow[x, 2] = np.uint8((np.uint32(srow[x, 0]) + np.uint32(128)) // np.uint32(257))


@_njit(nogil=True, fastmath=False)
def u16_to_display_u8_3d(src, dst, start, stop):
    width = src.shape[1]
    channels = src.shape[2]
    for y in range(start, stop):
        srow = src[y]
        drow = dst[y]
        for x in range(width):
            for c in range(channels):
                drow[x, c] = np.uint8((np.uint32(srow[x, c]) + np.uint32(128)) // np.uint32(257))


# ---------------------------------------------------------------------------
# Worker sizing & cancellable band runner
# ---------------------------------------------------------------------------
def worker_count(elements: int, height: int, band: int = _BAND_ROWS,
                 max_workers: int = 4, min_elements: int = _MIN_PARALLEL_ELEMENTS) -> int:
    """Choose worker count while respecting active serial/row execution scopes."""
    bands = (height + band - 1) // band
    if bands <= 1 or elements < min_elements:
        return 1
    exec_mod = sys.modules.get('fusion_methods._openfocus_execution')
    if exec_mod is not None:
        job = exec_mod._current.get()
        if job is not None:
            if not job.row_mode:
                return 1
            return max(1, min(max_workers, int(job.workers), bands))
    try:
        from utils.platform_utils import cpu_limits
        physical, logical = cpu_limits()[:2]
        cap = physical or max(1, (logical or 1) // 2)
    except Exception:
        cap = max(1, (os.cpu_count() or 1) // 2)
    return max(1, min(max_workers, cap, bands))


def run_image_bands(height: int, band: int, fn, *, workers: int = 1, should_cancel=None):
    """Run fn(start, stop) across disjoint row bands, returning per-slot values."""
    def check():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    check()
    if workers <= 1:
        results = []
        for start in range(0, height, band):
            check()
            results.append(fn(start, min(height, start + band)))
        check()
        return results

    stop = Event()

    def work(slot: int):
        out = []
        for start in range(slot * band, height, workers * band):
            if stop.is_set():
                from core.registration import RegistrationCancelled
                raise RegistrationCancelled()
            check()
            out.append(fn(start, min(height, start + band)))
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, slot) for slot in range(workers)]
        pending = set(futures)
        try:
            while pending:
                check()
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for fut in done:
                    fut.result()
            check()
        except BaseException:
            stop.set()
            for fut in futures:
                fut.cancel()
            wait(futures)
            raise
        return [item for fut in futures for item in fut.result()]
