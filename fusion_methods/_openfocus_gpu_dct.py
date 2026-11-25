"""OpenCL GPU fusion for DCT block-variance fusion (opt-in via use_gpu=True).

Ports BGR->grayscale conversion, edge-replicated block spatial variance,
streaming strict greater-than block argmax, and masked block reconstruction onto
OpenCL 1.2 device kernels with single-frame streaming, compact host
consistency filtering, and automatic CPU fallback.
"""
from __future__ import annotations

from threading import Lock

import cv2
import numpy as np
from scipy.ndimage import median_filter

from fusion_methods.dct import _ensure_color_image
from fusion_methods.openfocus_ab import _check_cancel
from utils.image_utils import fuse_output_dtype

_DCT_K = r'''
#pragma OPENCL FP_CONTRACT OFF

static inline uint cv_bgr_to_gray_u8(__global const uchar* src, size_t pos, int C) {
    if (C == 1) {
        return (uint)src[pos];
    }
    size_t b = pos * (size_t)C;
    uint c0 = (uint)src[b];
    uint c1 = (uint)src[b + 1];
    uint c2 = (uint)src[b + 2];
    return (c0 * 3735u + c1 * 19235u + c2 * 9798u + 16384u) >> 15;
}

__kernel void dct_block_var_argmax_u8(
    __global const uchar* src,
    __global float* max_var,
    __global int* best_idx,
    const int orig_h,
    const int orig_w,
    const int map_h,
    const int map_w,
    const int block_size,
    const int C,
    const int frame_idx,
    const int first_frame)
{
    int bx = get_global_id(0);
    int by = get_global_id(1);
    if (bx >= map_w || by >= map_h) return;

    int y0 = by * block_size;
    int x0 = bx * block_size;
    int max_y = orig_h - 1;
    int max_x = orig_w - 1;
    float var;

    if (block_size <= 16) {
        uint sx = 0u;
        uint sx2 = 0u;
        if (y0 + block_size <= orig_h && x0 + block_size <= orig_w) {
            for (int dy = 0; dy < block_size; ++dy) {
                size_t row = (size_t)(y0 + dy) * (size_t)orig_w + (size_t)x0;
                for (int dx = 0; dx < block_size; ++dx) {
                    uint v = cv_bgr_to_gray_u8(src, row + (size_t)dx, C);
                    sx += v;
                    sx2 += v * v;
                }
            }
        } else {
            for (int dy = 0; dy < block_size; ++dy) {
                int sy = y0 + dy;
                if (sy > max_y) sy = max_y;
                size_t row = (size_t)sy * (size_t)orig_w;
                for (int dx = 0; dx < block_size; ++dx) {
                    int sx_i = x0 + dx;
                    if (sx_i > max_x) sx_i = max_x;
                    uint v = cv_bgr_to_gray_u8(src, row + (size_t)sx_i, C);
                    sx += v;
                    sx2 += v * v;
                }
            }
        }
        float inv = 1.0f / (float)(block_size * block_size);
        float m = (float)sx * inv;
        float msq = (float)sx2 * inv;
        var = msq - m * m;
    } else {
        float inv_side = 1.0f / (float)block_size;
        float col_s = 0.0f;
        float col_s2 = 0.0f;
        for (int dy = 0; dy < block_size; ++dy) {
            int sy = min(y0 + dy, max_y);
            size_t row = (size_t)sy * (size_t)orig_w;
            float row_s = 0.0f;
            float row_s2 = 0.0f;
            for (int dx = 0; dx < block_size; ++dx) {
                int sx_i = min(x0 + dx, max_x);
                float v = (float)cv_bgr_to_gray_u8(src, row + (size_t)sx_i, C);
                row_s = row_s + v;
                row_s2 = row_s2 + v * v;
            }
            col_s = col_s + row_s * inv_side;
            col_s2 = col_s2 + row_s2 * inv_side;
        }
        float m = col_s * inv_side;
        var = col_s2 * inv_side - m * m;
    }

    size_t pos = (size_t)by * (size_t)map_w + (size_t)bx;
    if (first_frame || var > max_var[pos]) {
        max_var[pos] = var;
        best_idx[pos] = frame_idx;
    }
}

#define DEFINE_BLOCK_VAR_ARGMAX(NAME, SRC_T, CONV_EXPR) \
__kernel void NAME( \
    __global const SRC_T* src, \
    __global float* max_var, \
    __global int* best_idx, \
    const int orig_h, \
    const int orig_w, \
    const int map_h, \
    const int map_w, \
    const int block_size, \
    const int C, \
    const int frame_idx, \
    const int first_frame) \
{ \
    int bx = get_global_id(0); \
    int by = get_global_id(1); \
    if (bx >= map_w || by >= map_h) return; \
    int y0 = by * block_size; \
    int x0 = bx * block_size; \
    int max_y = orig_h - 1; \
    int max_x = orig_w - 1; \
    float inv_side = 1.0f / (float)block_size; \
    float col_s = 0.0f; \
    float col_s2 = 0.0f; \
    for (int dy = 0; dy < block_size; ++dy) { \
        int sy = min(y0 + dy, max_y); \
        size_t row = (size_t)sy * (size_t)orig_w; \
        float row_s = 0.0f; \
        float row_s2 = 0.0f; \
        for (int dx = 0; dx < block_size; ++dx) { \
            int sx_i = min(x0 + dx, max_x); \
            size_t p = row + (size_t)sx_i; \
            float v; \
            if (C == 1) { \
                SRC_T raw0 = src[p]; \
                v = (CONV_EXPR); \
            } else { \
                size_t b = p * (size_t)C; \
                SRC_T raw0 = src[b];     float c0 = (CONV_EXPR); \
                raw0       = src[b + 1]; float c1 = (CONV_EXPR); \
                raw0       = src[b + 2]; float c2 = (CONV_EXPR); \
                v = (0.114f * c0 + 0.587f * c1) + 0.299f * c2; \
            } \
            row_s = row_s + v; \
            row_s2 = row_s2 + v * v; \
        } \
        col_s = col_s + row_s * inv_side; \
        col_s2 = col_s2 + row_s2 * inv_side; \
    } \
    float m = col_s * inv_side; \
    float var = col_s2 * inv_side - m * m; \
    size_t pos = (size_t)by * (size_t)map_w + (size_t)bx; \
    if (first_frame || var > max_var[pos]) { \
        max_var[pos] = var; \
        best_idx[pos] = frame_idx; \
    } \
}

DEFINE_BLOCK_VAR_ARGMAX(dct_block_var_argmax_u16, ushort, ((float)raw0 * (255.0f / 65535.0f)))
DEFINE_BLOCK_VAR_ARGMAX(dct_block_var_argmax_f32, float, raw0)

#define DEFINE_COPY_BLOCKS(NAME, SRC_T, DST_T, CONV_EXPR) \
__kernel void NAME( \
    __global const SRC_T* src, \
    __global const int* final_idx, \
    __global DST_T* out, \
    const int orig_h, \
    const int orig_w, \
    const int map_w, \
    const int block_size, \
    const int C, \
    const int frame_idx) \
{ \
    int x = get_global_id(0); \
    int y = get_global_id(1); \
    if (x >= orig_w || y >= orig_h) return; \
    int bx = x / block_size; \
    int by = y / block_size; \
    if (final_idx[(size_t)by * (size_t)map_w + (size_t)bx] != frame_idx) return; \
    size_t p = (size_t)y * (size_t)orig_w + (size_t)x; \
    size_t ob = p * 3; \
    if (C == 1) { \
        SRC_T raw0 = src[p]; \
        DST_T v = (CONV_EXPR); \
        out[ob] = v; \
        out[ob + 1] = v; \
        out[ob + 2] = v; \
    } else { \
        size_t ib = p * (size_t)C; \
        SRC_T raw0 = src[ib];     out[ob]     = (CONV_EXPR); \
        raw0       = src[ib + 1]; out[ob + 1] = (CONV_EXPR); \
        raw0       = src[ib + 2]; out[ob + 2] = (CONV_EXPR); \
    } \
}

DEFINE_COPY_BLOCKS(dct_copy_u8_u8, uchar, uchar, raw0)
DEFINE_COPY_BLOCKS(dct_copy_u8_f32, uchar, float, (float)raw0)
DEFINE_COPY_BLOCKS(dct_copy_u16_f32, ushort, float, ((float)raw0 * (255.0f / 65535.0f)))
DEFINE_COPY_BLOCKS(dct_copy_f32_f32, float, float, raw0)
'''

_KERNEL_NAMES = (
    'dct_block_var_argmax_u8',
    'dct_block_var_argmax_u16',
    'dct_block_var_argmax_f32',
    'dct_copy_u8_u8',
    'dct_copy_u8_f32',
    'dct_copy_u16_f32',
    'dct_copy_f32_f32',
)

_PROGRAM_CACHE = None
_cache_lock = Lock()


def _device_cache_key(device):
    return (
        str(getattr(device, 'name', '')),
        str(getattr(device, 'vendor', '')),
        str(getattr(device, 'driver_version', '')),
        int(getattr(device, 'global_mem_size', 0)),
    )


def _get_or_build_program(cl, device):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import (
        _build_cached_program, _fp32_build_options, _get_shared_context,
    )

    key = _device_cache_key(device)
    with _cache_lock:
        ctx = _get_shared_context(cl, device)
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key and _PROGRAM_CACHE[1] is ctx:
            return _PROGRAM_CACHE[1:]
        queue = cl.CommandQueue(ctx, device=device)
        program = _build_cached_program(
            cl, device, ctx, _DCT_K,
            options=_fp32_build_options(cl, device),
        )
        kernels = {name: cl.Kernel(program, name) for name in _KERNEL_NAMES}
        _PROGRAM_CACHE = (key, ctx, queue, kernels)
        return ctx, queue, kernels


def _invalidate_program_cache(context=None):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import _invalidate_shared_context

    with _cache_lock:
        _PROGRAM_CACHE = None
        _invalidate_shared_context(context)


def _upload_kind(dtype):
    if dtype == np.uint8:
        return 'u8'
    if dtype == np.uint16:
        return 'u16'
    return 'f32'


def _source_staging_bytes(frame, shape):
    from core.source_frames import frame_has_mapped_pixels, frame_resolution_bytes

    resolution = frame_resolution_bytes(frame)
    if frame_has_mapped_pixels(frame):
        resolution = max(resolution, frame.nbytes)
    if tuple(frame.shape[:2]) != tuple(shape[:2]):
        return resolution + frame.nbytes * 4 + int(shape[0]) * int(shape[1]) * 12
    packed = 0 if isinstance(frame, np.ndarray) and frame.flags.c_contiguous else frame.nbytes
    itemsize = min(4, np.dtype(frame.dtype).itemsize)
    upload = frame.size * itemsize
    return resolution + packed + upload


def _frame_upload_bytes(frame, shape):
    if tuple(frame.shape[:2]) != tuple(shape[:2]):
        itemsize = min(4, np.dtype(frame.dtype).itemsize)
        return int(shape[0]) * int(shape[1]) * 3 * itemsize
    channels = frame.shape[2] if frame.ndim == 3 else 1
    itemsize = 4 if frame.dtype == np.float64 else frame.dtype.itemsize
    return int(shape[0]) * int(shape[1]) * channels * itemsize


class GpuDct:
    """OpenCL 1.2 device workspace and kernels for DCT block-variance fusion."""

    def __init__(self):
        from fusion_methods._openfocus_gpu import GpuError, _device, opencl_device_name
        try:
            import pyopencl as cl
            device = _device(cl)
            if device is None:
                raise GpuError('No available OpenCL GPU')
            self.cl = cl
            self.device = device
            self.device_name = opencl_device_name(device)
            self.mf = cl.mem_flags
            self.ctx, self.q, self.k = _get_or_build_program(cl, device)
        except GpuError:
            raise
        except Exception as exc:
            raise GpuError('OpenCL GPU DCT initialization failed') from exc

    def _run(self, kernel, gsize, *args):
        g = gsize if isinstance(gsize, tuple) else (gsize,)
        kernel(self.q, g, None, *args)

    def _drain_queue(self):
        q = getattr(self, 'q', None)
        if q is not None:
            try:
                q.finish()
            except Exception:
                _invalidate_program_cache(getattr(self, 'ctx', None))

    def fuse(self, frames, block_size, kernel_size,
             should_cancel=None, progress_callback=None):
        from fusion_methods._openfocus_gpu import GpuError, _device_lock

        class CallbackFailure(BaseException):
            pass

        def progress(index, total):
            try:
                progress_callback(index, total)
            except BaseException as exc:
                raise CallbackFailure() from exc

        while not _device_lock.acquire(timeout=0.1):
            _check_cancel(should_cancel)
        try:
            try:
                return self._fuse_locked(
                    frames, block_size, kernel_size, should_cancel,
                    progress if progress_callback is not None else None)
            except CallbackFailure as exc:
                self._drain_queue()
                raise exc.__cause__
            except (self.cl.Error, MemoryError) as exc:
                _invalidate_program_cache(getattr(self, 'ctx', None))
                self._drain_queue()
                raise GpuError('OpenCL GPU DCT execution failed') from exc
            except BaseException:
                self._drain_queue()
                raise
        finally:
            self.k = None
            self.q = None
            self.ctx = None
            _device_lock.release()

    def _fuse_locked(self, frames, block_size, kernel_size,
                     should_cancel, progress_callback):
        from dataclasses import replace
        from core import resource_policy
        from core.source_frames import resolve_frame, without_source_cache_growth
        from fusion_methods._openfocus_gpu import GpuError, _free_vram

        if not frames or len(frames) < 2:
            raise ValueError('需要至少 2 张图像进行融合。')

        block_size = max(2, int(block_size))
        ksize = int(kernel_size)
        if ksize % 2 == 0:
            ksize += 1

        orig_h, orig_w = int(frames[0].shape[0]), int(frames[0].shape[1])
        if orig_h < 8 or orig_w < 8:
            raise ValueError('图像尺寸过小')

        pad_h = ((orig_h + block_size - 1) // block_size) * block_size
        pad_w = ((orig_w + block_size - 1) // block_size) * block_size
        map_h = pad_h // block_size
        map_w = pad_w // block_size
        if map_h == 0 or map_w == 0:
            raise ValueError('Block size is too large for image size.')

        shape = (orig_h, orig_w, 3)
        n = orig_h * orig_w
        n_blocks = map_h * map_w
        count = len(frames)
        total_steps = count * 2
        if progress_callback is not None:
            progress_callback(0, total_steps)
        _check_cancel(should_cancel)

        out_dtype = fuse_output_dtype(frames)
        is_u8_out = out_dtype == np.uint8
        out_bytes = n * 3 * (1 if is_u8_out else 4)

        snapshot = resource_policy.probe_memory()
        host_budget = resource_policy.ram_ceiling(snapshot)
        source_peak = 0
        src_bytes = 0
        for frame in frames:
            _check_cancel(should_cancel)
            source_peak = max(source_peak, _source_staging_bytes(frame, shape))
            src_bytes = max(src_bytes, _frame_upload_bytes(frame, shape))
        staging = max(source_peak, out_bytes * 2) + 16 * 1024 * 1024
        if staging > host_budget:
            raise GpuError(f'DCT GPU staging exceeds the host memory budget '
                           f'({staging >> 20} MiB needed, {host_budget >> 20} MiB available)')

        need = (src_bytes
                + out_bytes
                + 8 * n_blocks
                + 16 * 1024 * 1024)
        limit = int(self.device.global_mem_size) // 2
        if self.device.host_unified_memory:
            limit = min(limit, host_budget - staging)
        else:
            memory = _free_vram(self.device)
            if memory is None:
                raise GpuError('DCT GPU free memory is unknown')
            snapshot = replace(snapshot, free_vram=memory[0], total_vram=memory[1])
            limit = min(limit, resource_policy.vram_budget(snapshot))
        single = int(self.device.max_mem_alloc_size)
        largest = max(src_bytes, out_bytes, 4 * n_blocks)
        if need > limit or largest > single:
            raise GpuError('DCT GPU workspace exceeds the device budget')

        def prepare_upload(index):
            _check_cancel(should_cancel)
            with without_source_cache_growth():
                raw = resolve_frame(frames[index], should_cancel=should_cancel)
            _check_cancel(should_cancel)
            if raw.dtype.kind == 'f':
                if not np.isfinite(raw).all():
                    raise GpuError('Non-finite floating-point frame uses CPU fallback')
                mn, mx = float(np.min(raw)), float(np.max(raw))
                if mn < -255.0 or mx > 255.0:
                    raise GpuError('Unbounded floating-point frame uses CPU fallback')
            if raw.shape[:2] != (orig_h, orig_w):
                image = _ensure_color_image(np.ascontiguousarray(raw))
                del raw
                # Like the CPU path, resize native integer samples before normalization.
                if image.dtype == np.float64:
                    image = image.astype(np.float32)
                host_src = np.ascontiguousarray(
                    cv2.resize(image, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR))
                del image
                return host_src, 3, _upload_kind(host_src.dtype)
            if raw.ndim == 2:
                channels = 1
            elif raw.ndim == 3 and raw.shape[2] in (3, 4):
                channels = int(raw.shape[2])
            else:
                raise ValueError(f'Unsupported image shape: {raw.shape}')
            dtype = np.float32 if raw.dtype == np.float64 else raw.dtype
            if dtype not in (np.uint8, np.uint16, np.float32):
                raise GpuError(f'Unsupported GPU pixel dtype: {raw.dtype}')
            host_src = np.ascontiguousarray(raw, dtype=dtype)
            del raw
            return host_src, channels, _upload_kind(dtype)

        src_buf = self.cl.Buffer(self.ctx, self.mf.READ_ONLY, max(1, src_bytes))
        max_var_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n_blocks))
        idx_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n_blocks))

        in_flight = False
        try:
            # Pass 1: streaming variance + strict '>' argmax per block.
            for index in range(count):
                _check_cancel(should_cancel)
                h_src, ch, kind = prepare_upload(index)
                in_flight = True
                self.cl.enqueue_copy(self.q, src_buf, h_src, is_blocking=False)
                self._run(
                    self.k[f'dct_block_var_argmax_{kind}'],
                    (map_w, map_h),
                    src_buf, max_var_buf, idx_buf,
                    np.int32(orig_h), np.int32(orig_w),
                    np.int32(map_h), np.int32(map_w),
                    np.int32(block_size), np.int32(ch),
                    np.int32(index), np.int32(1 if index == 0 else 0),
                )
                self.q.finish()
                in_flight = False
                del h_src
                if progress_callback is not None:
                    progress_callback(index + 1, total_steps)

            del max_var_buf

            # Pass 2: consistency verification via two median filters on compact block index grid.
            best_index_map = np.empty((map_h, map_w), dtype=np.int32)
            in_flight = True
            self.cl.enqueue_copy(self.q, best_index_map, idx_buf, is_blocking=True)
            in_flight = False
            _check_cancel(should_cancel)
            filtered = median_filter(best_index_map, size=ksize, mode='nearest')
            _check_cancel(should_cancel)
            filtered = median_filter(filtered, size=ksize, mode='nearest')
            final_index_map = np.ascontiguousarray(filtered, dtype=np.int32)
            del best_index_map, filtered

            unique_set = set(int(v) for v in np.unique(final_index_map))
            in_flight = True
            self.cl.enqueue_copy(self.q, idx_buf, final_index_map, is_blocking=False)

            # Pass 3: reconstruct selected blocks for active source frames only.
            out_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, max(1, out_bytes))
            self.cl.enqueue_fill_buffer(
                self.q, out_buf, np.uint8(0) if is_u8_out else np.float32(0.0), 0, out_bytes
            )
            out_suffix = 'u8' if is_u8_out else 'f32'

            for index in range(count):
                _check_cancel(should_cancel)
                if index not in unique_set:
                    if progress_callback is not None:
                        progress_callback(count + index + 1, total_steps)
                    continue
                h_src, ch, kind = prepare_upload(index)
                in_flight = True
                self.cl.enqueue_copy(self.q, src_buf, h_src, is_blocking=False)
                self._run(
                    self.k[f'dct_copy_{kind}_{out_suffix}'],
                    (orig_w, orig_h),
                    src_buf, idx_buf, out_buf,
                    np.int32(orig_h), np.int32(orig_w),
                    np.int32(map_w), np.int32(block_size),
                    np.int32(ch), np.int32(index),
                )
                self.q.finish()
                in_flight = False
                del h_src
                if progress_callback is not None:
                    progress_callback(count + index + 1, total_steps)

            if in_flight:
                self.q.finish()
                in_flight = False
            del final_index_map
            del src_buf, idx_buf

            _check_cancel(should_cancel)
            out = np.empty(shape, dtype=out_dtype)
            in_flight = True
            self.cl.enqueue_copy(self.q, out, out_buf, is_blocking=True)
            in_flight = False
            if progress_callback is not None:
                progress_callback(total_steps, total_steps)
            _check_cancel(should_cancel)
            return out
        except BaseException:
            if in_flight:
                self._drain_queue()
            raise
