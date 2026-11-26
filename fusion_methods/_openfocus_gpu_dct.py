"""OpenCL GPU fusion for DCT block-variance fusion (opt-in via use_gpu=True).

Ports BGR->grayscale conversion, coalesced edge-replicated block spatial
variance, streaming strict greater-than block argmax, device-side consistency
median filtering, and masked block reconstruction onto OpenCL 1.2 device
kernels with bounded background frame prefetch, ping-pong async DMA, and
automatic CPU fallback.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
import contextvars
import os
from threading import Event, Lock

import cv2
import numpy as np
from scipy.ndimage import median_filter

from core.frame_storage import async_upload_view
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

__kernel void dct_block_var_argmax_u8_b8(
    __global const uchar* src,
    __global float* max_var,
    __global int* best_idx,
    const int orig_h,
    const int orig_w,
    const int map_h,
    const int map_w,
    const int C,
    const int frame_idx,
    const int first_frame)
{
    int bx = get_group_id(0);
    int by = get_group_id(1);
    if (bx >= map_w || by >= map_h) return;

    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int tid = (ty << 3) | tx;

    int sy = by * 8 + ty;
    int sx_i = bx * 8 + tx;
    if (sy >= orig_h) sy = orig_h - 1;
    if (sx_i >= orig_w) sx_i = orig_w - 1;

    uint v = cv_bgr_to_gray_u8(src, (size_t)sy * (size_t)orig_w + (size_t)sx_i, C);

    __local uint l_sx[64];
    __local uint l_sx2[64];
    l_sx[tid] = v;
    l_sx2[tid] = v * v;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32) {
        l_sx[tid] += l_sx[tid + 32];
        l_sx2[tid] += l_sx2[tid + 32];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16) {
        l_sx[tid] += l_sx[tid + 16];
        l_sx2[tid] += l_sx2[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 8) {
        l_sx[tid] += l_sx[tid + 8];
        l_sx2[tid] += l_sx2[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 4) {
        l_sx[tid] += l_sx[tid + 4];
        l_sx2[tid] += l_sx2[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 2) {
        l_sx[tid] += l_sx[tid + 2];
        l_sx2[tid] += l_sx2[tid + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        uint total_s = l_sx[0] + l_sx[1];
        uint total_s2 = l_sx2[0] + l_sx2[1];
        const float inv = 1.0f / 64.0f;
        float m = (float)total_s * inv;
        float msq = (float)total_s2 * inv;
        float var = msq - m * m;
        size_t pos = (size_t)by * (size_t)map_w + (size_t)bx;
        if (first_frame || var > max_var[pos]) {
            max_var[pos] = var;
            best_idx[pos] = frame_idx;
        }
    }
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

__kernel void dct_median_filter_i32(
    __global const int* src,
    __global int* dst,
    const int h,
    const int w,
    const int ksize)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    int r = ksize >> 1;
    int center = src[(size_t)y * (size_t)w + (size_t)x];
    int buf[49];
    int pos = 0;
    int uniform = 1;

    if (y >= r && y < h - r && x >= r && x < w - r) {
        for (int dy = -r; dy <= r; ++dy) {
            size_t row = (size_t)(y + dy) * (size_t)w;
            for (int dx = -r; dx <= r; ++dx) {
                int v = src[row + (size_t)(x + dx)];
                if (v != center) uniform = 0;
                buf[pos++] = v;
            }
        }
    } else {
        for (int dy = -r; dy <= r; ++dy) {
            int sy = clamp(y + dy, 0, h - 1);
            size_t row = (size_t)sy * (size_t)w;
            for (int dx = -r; dx <= r; ++dx) {
                int sx_i = clamp(x + dx, 0, w - 1);
                int v = src[row + (size_t)sx_i];
                if (v != center) uniform = 0;
                buf[pos++] = v;
            }
        }
    }

    if (uniform) {
        dst[(size_t)y * (size_t)w + (size_t)x] = center;
        return;
    }

    // Hoare quickselect for exact median of odd-sized window (ksize <= 7).
    int n = pos;
    int k = n >> 1;
    int lo = 0;
    int hi = n - 1;
    while (lo < hi) {
        int pivot = buf[(lo + hi) >> 1];
        int i = lo;
        int j = hi;
        while (i <= j) {
            while (buf[i] < pivot) ++i;
            while (buf[j] > pivot) --j;
            if (i <= j) {
                int tmp = buf[i];
                buf[i] = buf[j];
                buf[j] = tmp;
                ++i;
                --j;
            }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    dst[(size_t)y * (size_t)w + (size_t)x] = buf[k];
}

__kernel void dct_mark_active_i32(
    __global const int* final_idx,
    volatile __global uint* active_flags,
    const int n_blocks,
    const int count)
{
    int gid = get_global_id(0);
    if (gid >= n_blocks) return;
    int f = final_idx[gid];
    if (gid > 0 && final_idx[gid - 1] == f) return;
    if (f >= 0 && f < count) {
        atomic_or(&active_flags[f], 1u);
    }
}

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
    'dct_block_var_argmax_u8_b8',
    'dct_block_var_argmax_u8',
    'dct_block_var_argmax_u16',
    'dct_block_var_argmax_f32',
    'dct_median_filter_i32',
    'dct_mark_active_i32',
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


def _dct_prefetch_slots(host_budget: int, host_staging: int, frame_bytes: int, count: int) -> int:
    """Return bounded background prefetch depth for DCT host frame decode."""
    from fusion_methods._openfocus_execution import _current

    job = _current.get()
    if job is not None and getattr(job, 'thread_count', None) == 1:
        return 0
    if count < 2 or frame_bytes <= 0 or host_staging + frame_bytes > host_budget:
        return 0
    headroom = max(0, int(host_budget) - int(host_staging))
    cpu_cap = max(1, min(6, (os.cpu_count() or 2) // 2))
    return max(0, min(cpu_cap, int(headroom // int(frame_bytes)), int(count) - 1))


def _iter_prefetched_indices(indices, prepare_fn, *, slots: int, should_cancel=None):
    """Yield (frame_index, prepared_tuple) in order with bounded multi-thread decode."""
    total = len(indices)
    if total == 0:
        return
    if slots <= 0 or total < 2:
        for frame_index in indices:
            _check_cancel(should_cancel)
            result = prepare_fn(frame_index, should_cancel)
            _check_cancel(should_cancel)
            yield frame_index, result
            del result
        return

    from fusion_methods._openfocus_execution import _worker_native_limit

    stop = Event()

    def cancelled():
        return stop.is_set() or (should_cancel is not None and should_cancel())

    def submit_item(pool, frame_index):
        ctx = contextvars.copy_context()
        return pool.submit(ctx.run, prepare_fn, frame_index, cancelled)

    _check_cancel(should_cancel)
    workers = min(slots, 6)
    with ThreadPoolExecutor(max_workers=workers, initializer=_worker_native_limit) as pool:
        pending = deque()
        next_pos = 0
        try:
            while next_pos < min(slots, total):
                pending.append((indices[next_pos], submit_item(pool, indices[next_pos])))
                next_pos += 1
            while pending:
                frame_index, future = pending[0]
                while not future.done():
                    _check_cancel(should_cancel)
                    wait((future,), timeout=0.05)
                _check_cancel(should_cancel)
                pending.popleft()
                result = future.result()
                while next_pos < total and len(pending) < slots:
                    pending.append((indices[next_pos], submit_item(pool, indices[next_pos])))
                    next_pos += 1
                yield frame_index, result
                del result
        finally:
            stop.set()
            for _, future in pending:
                future.cancel()


def _supports_b8_local(cl, device, kernel) -> bool:
    try:
        max_wg = int(kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
        item_sizes = getattr(device, 'max_work_item_sizes', (8, 8))
        local_mem = int(getattr(device, 'local_mem_size', 1024))
        return max_wg >= 64 and len(item_sizes) >= 2 and min(item_sizes[:2]) >= 8 and local_mem >= 512
    except Exception:
        return False


class GpuDct:
    """OpenCL 1.2 device workspace and kernels for DCT block-variance fusion."""

    def __init__(self):
        from fusion_methods._openfocus_gpu import (
            GpuError, _device, _get_transfer_queue, opencl_device_name,
        )
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
            self.tq = _get_transfer_queue(cl, device, self.ctx)
            self._b8_local = _supports_b8_local(cl, device, self.k['dct_block_var_argmax_u8_b8'])
        except GpuError:
            raise
        except Exception as exc:
            raise GpuError('OpenCL GPU DCT initialization failed') from exc

    def _run(self, kernel, gsize, *args, lsize=None, wait_for=None):
        from fusion_methods._openfocus_gpu import _enqueue_kernel

        g = gsize if isinstance(gsize, tuple) else (gsize,)
        return _enqueue_kernel(kernel, self.q, g, lsize, *args, wait_for=wait_for)

    def _drain_queue(self):
        from fusion_methods._openfocus_gpu import _drain_command_queue

        ctx = getattr(self, 'ctx', None)
        tq = getattr(self, 'tq', None)
        q = getattr(self, 'q', None)
        if tq is not None and tq is not q:
            _drain_command_queue(tq, ctx)
        if q is not None:
            try:
                q.finish()
            except Exception:
                _invalidate_program_cache(ctx)

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
            self.tq = None
            self.q = None
            self.ctx = None
            _device_lock.release()

    def _fuse_locked(self, frames, block_size, kernel_size,
                     should_cancel, progress_callback):
        from dataclasses import replace
        from core import resource_policy
        from core.source_frames import resolve_frame, without_source_cache_growth
        from fusion_methods._openfocus_gpu import (
            GpuError, _PinnedUploader, _event_wait_list, _free_vram,
            _release_buffers, _wait_event,
        )

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
        unified_memory = self.device.host_unified_memory
        if unified_memory:
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

        def prepare_upload(index, cancelled):
            _check_cancel(cancelled)
            with without_source_cache_growth():
                raw = resolve_frame(frames[index], should_cancel=cancelled)
            _check_cancel(cancelled)
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

        src_bufs = ()
        max_var_buf = idx_buf = active_buf = out_buf = None
        uploader = pass1_iter = pass3_iter = None
        in_flight = False
        cancel_fn = lambda: _check_cancel(should_cancel)
        try:
            prefetch_budget = host_budget - need if unified_memory else host_budget
            prefetch_slots = _dct_prefetch_slots(
                prefetch_budget, staging, source_peak, count,
            )
            ping_pong = (
                count >= 2
                and (need + src_bytes) <= limit
                and (staging + prefetch_slots * source_peak + 2 * src_bytes
                     + (src_bytes if unified_memory else 0)) <= prefetch_budget
            )
            src_bufs = tuple(
                self.cl.Buffer(self.ctx, self.mf.READ_ONLY, max(1, src_bytes))
                for _ in range(2 if ping_pong else 1)
            )
            max_var_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n_blocks))
            idx_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, 4 * n_blocks))
            if ping_pong:
                uploader = _PinnedUploader(
                    self.cl, self.ctx, self.q, getattr(self, 'tq', None),
                    2, src_bytes,
                )

            use_b8 = (block_size == 8) and bool(getattr(self, '_b8_local', False))

            # Pass 1: pipelined variance + strict '>' argmax per block.
            pass1_iter = _iter_prefetched_indices(
                range(count), prepare_upload, slots=prefetch_slots,
                should_cancel=should_cancel,
            )
            slot_events = [None, None]
            slot_hosts = [None, None]
            for index, (h_src, ch, kind) in pass1_iter:
                slot = index & 1 if ping_pong else 0
                prev_slot_ev = slot_events[slot]
                if prev_slot_ev is not None and (uploader is None or not uploader.separate_queue):
                    _wait_event(prev_slot_ev, cancel_fn)
                    slot_events[slot] = None
                    slot_hosts[slot] = None
                    prev_slot_ev = None
                _check_cancel(should_cancel)
                src_buf = src_bufs[slot]
                in_flight = True
                if uploader is not None:
                    upload = uploader.upload(
                        slot, src_buf, h_src,
                        wait_for=_event_wait_list(prev_slot_ev), cancel=cancel_fn,
                    )
                    wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
                else:
                    upload = self.cl.enqueue_copy(
                        self.q, src_buf, async_upload_view(h_src), is_blocking=False,
                    )
                    wait_upload = None

                if use_b8 and kind == 'u8':
                    last_ev = self._run(
                        self.k['dct_block_var_argmax_u8_b8'],
                        (map_w * 8, map_h * 8),
                        src_buf, max_var_buf, idx_buf,
                        np.int32(orig_h), np.int32(orig_w),
                        np.int32(map_h), np.int32(map_w),
                        np.int32(ch),
                        np.int32(index), np.int32(1 if index == 0 else 0),
                        lsize=(8, 8), wait_for=wait_upload,
                    )
                else:
                    last_ev = self._run(
                        self.k[f'dct_block_var_argmax_{kind}'],
                        (map_w, map_h),
                        src_buf, max_var_buf, idx_buf,
                        np.int32(orig_h), np.int32(orig_w),
                        np.int32(map_h), np.int32(map_w),
                        np.int32(block_size), np.int32(ch),
                        np.int32(index), np.int32(1 if index == 0 else 0),
                        wait_for=wait_upload,
                    )
                if ping_pong and last_ev is not None and hasattr(last_ev, 'wait'):
                    if hasattr(self.q, 'flush'):
                        self.q.flush()
                    slot_events[slot] = last_ev
                    slot_hosts[slot] = (h_src, upload)
                else:
                    self.q.finish()
                    in_flight = False
                    del h_src, upload
                if progress_callback is not None:
                    progress_callback(index + 1, total_steps)

            pass1_iter.close()
            pass1_iter = None
            if ping_pong:
                if uploader is not None:
                    uploader.finish(cancel_fn)
                for slot in (0, 1):
                    if slot_events[slot] is not None:
                        _wait_event(slot_events[slot], cancel_fn)
                        slot_events[slot] = None
                    slot_hosts[slot] = None
                self.q.finish()
                in_flight = False

            # Pass 2: consistency verification via two median filters on compact block index grid.
            _check_cancel(should_cancel)
            if ksize <= 7:
                # Run both passes on device, reusing max_var_buf (4 * n_blocks bytes) as scratch.
                in_flight = True
                if ksize > 1:
                    self._run(
                        self.k['dct_median_filter_i32'], (map_w, map_h),
                        idx_buf, max_var_buf,
                        np.int32(map_h), np.int32(map_w), np.int32(ksize),
                    )
                    self._run(
                        self.k['dct_median_filter_i32'], (map_w, map_h),
                        max_var_buf, idx_buf,
                        np.int32(map_h), np.int32(map_w), np.int32(ksize),
                    )
                active_bytes = max(4, count * 4)
                active_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, active_bytes)
                self.cl.enqueue_fill_buffer(self.q, active_buf, np.uint32(0), 0, active_bytes)
                self._run(
                    self.k['dct_mark_active_i32'], (n_blocks,),
                    idx_buf, active_buf, np.int32(n_blocks), np.int32(count),
                )
                active_flags = np.empty(count, dtype=np.uint32)
                self.cl.enqueue_copy(self.q, active_flags, active_buf, is_blocking=True)
                in_flight = False
                _release_buffers(active_buf, max_var_buf)
                active_buf = max_var_buf = None
                active_indices = [i for i in range(count) if active_flags[i]]
            else:
                _release_buffers(max_var_buf)
                max_var_buf = None
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
                active_indices = [i for i in range(count) if i in unique_set]
                in_flight = True
                self.cl.enqueue_copy(self.q, idx_buf, final_index_map, is_blocking=True)
                in_flight = False
                del final_index_map

            # Pass 3: pipelined reconstruction for active source frames only.
            out_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, max(1, out_bytes))
            in_flight = True
            self.cl.enqueue_fill_buffer(
                self.q, out_buf, np.uint8(0) if is_u8_out else np.float32(0.0), 0, out_bytes
            )
            out_suffix = 'u8' if is_u8_out else 'f32'

            pass3_slots = _dct_prefetch_slots(
                prefetch_budget, staging, source_peak, len(active_indices),
            )
            pass3_iter = _iter_prefetched_indices(
                active_indices, prepare_upload, slots=pass3_slots,
                should_cancel=should_cancel,
            )
            slot_events = [None, None]
            slot_hosts = [None, None]
            next_progress = 0
            for step_pos, (index, (h_src, ch, kind)) in enumerate(pass3_iter):
                slot = step_pos & 1 if ping_pong else 0
                prev_slot_ev = slot_events[slot]
                if prev_slot_ev is not None and (uploader is None or not uploader.separate_queue):
                    _wait_event(prev_slot_ev, cancel_fn)
                    slot_events[slot] = None
                    slot_hosts[slot] = None
                    prev_slot_ev = None
                _check_cancel(should_cancel)
                src_buf = src_bufs[slot]
                in_flight = True
                if uploader is not None:
                    upload = uploader.upload(
                        slot, src_buf, h_src,
                        wait_for=_event_wait_list(prev_slot_ev), cancel=cancel_fn,
                    )
                    wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
                else:
                    upload = self.cl.enqueue_copy(
                        self.q, src_buf, async_upload_view(h_src), is_blocking=False,
                    )
                    wait_upload = None

                last_ev = self._run(
                    self.k[f'dct_copy_{kind}_{out_suffix}'],
                    (orig_w, orig_h),
                    src_buf, idx_buf, out_buf,
                    np.int32(orig_h), np.int32(orig_w),
                    np.int32(map_w), np.int32(block_size),
                    np.int32(ch), np.int32(index),
                    wait_for=wait_upload,
                )
                if ping_pong and last_ev is not None and hasattr(last_ev, 'wait'):
                    if hasattr(self.q, 'flush'):
                        self.q.flush()
                    slot_events[slot] = last_ev
                    slot_hosts[slot] = (h_src, upload)
                else:
                    self.q.finish()
                    in_flight = False
                    del h_src, upload
                if progress_callback is not None:
                    while next_progress <= index:
                        progress_callback(count + next_progress + 1, total_steps)
                        next_progress += 1

            pass3_iter.close()
            pass3_iter = None
            if ping_pong:
                if uploader is not None:
                    uploader.finish(cancel_fn)
                for slot in (0, 1):
                    if slot_events[slot] is not None:
                        _wait_event(slot_events[slot], cancel_fn)
                        slot_events[slot] = None
                    slot_hosts[slot] = None
            if in_flight:
                self.q.finish()
                in_flight = False
            if uploader is not None:
                uploader.close()
                uploader = None
            _release_buffers(src_bufs, idx_buf)
            src_bufs = ()
            idx_buf = None

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
        finally:
            if uploader is not None:
                uploader.close()
            if pass1_iter is not None:
                pass1_iter.close()
            if pass3_iter is not None:
                pass3_iter.close()
            _release_buffers(src_bufs, max_var_buf, idx_buf, active_buf, out_buf)
