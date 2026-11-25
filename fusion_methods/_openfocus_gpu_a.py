"""OpenFocus A GPU fusion (opt-in via use_gpu=True on openfocus_a_impl).

Bit-exact port of OpenFocus A's default (hard) Burt-pyramid response,
winner-take-all selection with low-response background accumulation,
low-response blending, and output code exit onto OpenCL kernels.
"""
from __future__ import annotations

import math
from threading import Lock

import numpy as np

from fusion_methods._openfocus_pixels import frame_codes
from fusion_methods.openfocus_ab import _check_cancel
from fusion_methods.openfocus_a import _hard_pyramid_shapes

_READBACK_BUFFER_BYTES = 8 * 1024 * 1024

_KERNEL_NAMES = (
    'a_down2_u8', 'a_down2_u16', 'a_down2', 'a_laplacian',
    'a_select_u8', 'a_select_u16',
    'a_select_u8_bgr_rev', 'a_select_u16_bgr_rev',
    'a_blend_low', 'a_out_u8', 'a_out_f32',
    'sample_focus_plane',
)

_cache_lock = Lock()
_PROGRAM_CACHE = None


def _get_or_build_program(cl, device):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import (
        _PREVIEW_KERNEL, _build_cached_program, _device_cache_key,
        _fp32_build_options, _get_shared_context,
    )

    key = _device_cache_key(device)
    with _cache_lock:
        ctx = _get_shared_context(cl, device)
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key and _PROGRAM_CACHE[1] is ctx:
            return _PROGRAM_CACHE[1:]
        queue = cl.CommandQueue(ctx, device=device)
        program = _build_cached_program(
            cl, device, ctx, _A_K + _PREVIEW_KERNEL.format(type='int'),
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

_A_K = r'''
#pragma OPENCL FP_CONTRACT OFF

static inline int gray_u8(__global const uchar* src, size_t i, int channels, int u8_gray)
{
    if (channels == 1) return (int)src[i] << 8;
    size_t base = i * channels;
    int total = (int)src[base] + (int)src[base + 1]
              + (channels >= 3 ? (int)src[base + 2] : 0);
    return (total / 3) << 8;
}

static inline int gray_u16(__global const ushort* src, size_t i, int channels, int u8_gray)
{
    if (channels == 1) return (int)src[i];
    size_t base = i * channels;
    int total = (int)src[base] + (int)src[base + 1]
              + (channels >= 3 ? (int)src[base + 2] : 0);
    return u8_gray ? (((total >> 8) / 3) << 8) : (total / 3);
}

#define DEFINE_A_DOWN2(NAME, PIXEL_T, GRAY) \
__kernel void NAME(__global const PIXEL_T* src, __global int* out, \
                   const int H, const int W, const int OH, const int OW, \
                   const int channels, const int u8_gray) \
{ \
    int x = get_global_id(0); \
    int y = get_global_id(1); \
    if (x >= OW || y >= OH) return; \
    size_t top = (size_t)(2 * y) * W; \
    size_t bottom = (size_t)min(2 * y + 1, H - 1) * W; \
    int left = 2 * x; \
    int right = min(left + 1, W - 1); \
    int total = GRAY(src, top + left, channels, u8_gray) \
              + GRAY(src, top + right, channels, u8_gray) \
              + GRAY(src, bottom + left, channels, u8_gray) \
              + GRAY(src, bottom + right, channels, u8_gray); \
    out[(size_t)y * OW + x] = total >> 2; \
}

DEFINE_A_DOWN2(a_down2_u8, uchar, gray_u8)
DEFINE_A_DOWN2(a_down2_u16, ushort, gray_u16)

__kernel void a_down2(__global const int* gray, __global int* out,
                      const int H, const int W, const int OH, const int OW)
{
    int x = get_global_id(0);
    if (x >= OW) return;
    int y = get_global_id(1);
    if (y >= OH) return;
    int top = 2 * y;
    int bottom = min(top + 1, H - 1);
    int left = 2 * x;
    int right = min(left + 1, W - 1);
    int total = gray[(size_t)top * W + left] + gray[(size_t)top * W + right]
              + gray[(size_t)bottom * W + left] + gray[(size_t)bottom * W + right];
    out[(size_t)y * OW + x] = total >> 2;
}

static inline int burt_expand2(__global const int* reduced, int y, int x, int H, int W)
{
    int y0 = (y >> 1) - ((y & 1) == 0);
    int y1 = min(y0 + 1, H - 1);
    y0 = max(y0, 0);
    int wy0 = (y & 1) == 0 ? 1 : 3;
    int wy1 = 4 - wy0;
    int x0 = (x >> 1) - ((x & 1) == 0);
    int x1 = min(x0 + 1, W - 1);
    x0 = max(x0, 0);
    int wx0 = (x & 1) == 0 ? 1 : 3;
    int wx1 = 4 - wx0;
    size_t row0 = (size_t)y0 * W;
    size_t row1 = (size_t)y1 * W;
    int top = reduced[row0 + x0] * wx0 + reduced[row0 + x1] * wx1;
    int bottom = reduced[row1 + x0] * wx0 + reduced[row1 + x1] * wx1;
    return top * wy0 + bottom * wy1;
}

__kernel void a_laplacian(__global const int* gray, __global const int* reduced,
                          __global int* lap, const int offset, const int GH, const int GW,
                          const int RH, const int RW)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= GW || y >= GH) return;
    size_t i = (size_t)y * GW + x;
    lap[offset + i] = (int)abs(gray[i] * 16 - burt_expand2(reduced, y, x, RH, RW));
}

static inline int burt_expand_laplacian(__global const int* lap, int y, int x,
                                        int H, int W, int level)
{
    if (level == 1) return (burt_expand2(lap, y, x, H, W) + 8) >> 4;

    int shift = level + 1;
    int factor = 1 << level;
    int den = 1 << shift;
    int py = 2 * y + 1 - factor;
    int px = 2 * x + 1 - factor;
    int y0 = py >> shift;
    int x0 = px >> shift;
    int fy = py - y0 * den;
    int fx = px - x0 * den;
    int ya = max(0, min(y0, H - 1));
    int yb = max(0, min(y0 + 1, H - 1));
    int xa = max(0, min(x0, W - 1));
    int xb = max(0, min(x0 + 1, W - 1));
    size_t row0 = (size_t)ya * W;
    size_t row1 = (size_t)yb * W;
    int a = lap[row0 + xa], b = lap[row1 + xa];
    int c = lap[row0 + xb], d = lap[row1 + xb];
    int bits = shift * 2;
    // Q4 Laplacians are at most 65535*16. For den <= 32 the
    // weighted numerator and rounding both fit a signed int.
    if (shift <= 5) {
        int left = a * (den - fy) + b * fy;
        int right = c * (den - fy) + d * fy;
        return (left * (den - fx) + right * fx + (1 << (bits - 1))) >> bits;
    }
    long left = (long)a * (den - fy) + (long)b * fy;
    long right = (long)c * (den - fy) + (long)d * fy;
    return (int)((left * (den - fx) + right * fx + (1L << (bits - 1))) >> bits);
}

#define DEFINE_A_SELECT(NAME, PIXEL_T, SCALE, GRAY, BGR_REV) \
__kernel void NAME(__global const PIXEL_T* image, __global const int* half_gray, \
                   __global const int* lap, __global int* best, __global ushort* output, \
                   __global float* accumulated, __global float* weights, \
                   __global ushort* stats, const int H, const int W, const int C, \
                   const int levels, const int first, const int u8_gray) \
{ \
    int x = get_global_id(0); \
    int y = get_global_id(1); \
    if (x >= W || y >= H) return; \
    size_t k = (size_t)y * W + x; \
    int previous = first ? 0 : best[k]; \
    if (previous == 65535 * 16) return; \
    int h = (H + 1) >> 1; \
    int w = (W + 1) >> 1; \
    int score = (int)abs(GRAY(image, k, C, u8_gray) * 16 - burt_expand2(half_gray, y, x, h, w)); \
    for (int level = 1; level < levels && score < 65535 * 16; ++level) { \
        score += burt_expand_laplacian(lap, y, x, h, w, level); \
        lap += (size_t)h * w; \
        h = (h + 1) >> 1; \
        w = (w + 1) >> 1; \
    } \
    score = min(score, 65535 * 16); \
    int take = first || score > previous; \
    int collect = previous < 65536 && score < 65536; \
    if (!take && !collect) return; \
    float factor = 1.0f; \
    float weight = 0.0f; \
    if (collect) { \
        ushort current_code = (ushort)(score >> 4); \
        size_t sk = (size_t)k * 2; \
        if (!first && score == 0) { \
            if (stats[sk]) stats[sk] = (ushort)0; \
            return; \
        } \
        if (first) { \
            stats[sk] = current_code; \
            stats[sk + 1] = (ushort)0; \
        } else { \
            ushort minimum = stats[sk]; \
            if (current_code < minimum) minimum = current_code; \
            ushort second = stats[sk + 1]; \
            if (score > previous) { \
                second = (ushort)(previous >> 4); \
            } else if (current_code > second) { \
                second = current_code; \
            } \
            stats[sk] = minimum; \
            stats[sk + 1] = second; \
        } \
        if (first) { \
            weight = score ? 1.0f : 0.0f; \
            weights[k] = weight; \
        } else if (score > previous) { \
            float ratio = (float)previous / (float)score; \
            float r2 = ratio * ratio; \
            factor = r2 * r2; \
            weight = 1.0f; \
            float wf = weights[k] * factor; \
            weights[k] = wf + weight; \
        } else { \
            float ratio = previous ? ((float)score / (float)previous) : 0.0f; \
            float r2 = ratio * ratio; \
            weight = r2 * r2; \
            weights[k] = weights[k] + weight; \
        } \
    } \
    for (int c = 0; c < C; ++c) { \
        size_t idx = (size_t)k * C + c; \
        size_t sidx = BGR_REV ? ((size_t)k * 3 + (2 - c)) : idx; \
        int value = (int)image[sidx] * SCALE; \
        if (collect) { \
            if (first) { \
                accumulated[idx] = (float)value * weight; \
            } else { \
                float a1 = accumulated[idx] * factor; \
                float a2 = (float)value * weight; \
                accumulated[idx] = a1 + a2; \
            } \
        } \
        if (take) { \
            output[idx] = (ushort)value; \
        } \
    } \
    if (take) { \
        best[k] = (int)score; \
    } \
}

DEFINE_A_SELECT(a_select_u8, uchar, 256, gray_u8, 0)
DEFINE_A_SELECT(a_select_u16, ushort, 1, gray_u16, 0)
DEFINE_A_SELECT(a_select_u8_bgr_rev, uchar, 256, gray_u8, 1)
DEFINE_A_SELECT(a_select_u16_bgr_rev, ushort, 1, gray_u16, 1)

__kernel void a_blend_low(__global ushort* output, __global const int* best,
                          __global const float* accumulated, __global const float* weights,
                          __global const ushort* stats, const int n, const int C)
{
    int k = get_global_id(0);
    if (k >= n) return;
    long score = (long)best[k];
    size_t sk = (size_t)k * 2;
    if (score >= 65536L || (score >> 4) == (long)stats[sk]) return;
    long second = (long)stats[sk + 1] << 4;
    long gap = max(0L, score - second);
    if (gap * 2L >= score) return;
    long confidence = min(score, 5L * score - 10L * gap);
    float s1 = (float)min(32768L, 65536L - score) / 32768.0f;
    float s2 = s1 * (float)confidence;
    float strength = s2 / (float)score;
    float denominator = weights[k];
    if (denominator <= 0.0f) return;
    float inv_strength = 1.0f - strength;
    for (int c = 0; c < C; ++c) {
        size_t idx = (size_t)k * C + c;
        float average = accumulated[idx] / denominator;
        float v1 = (float)output[idx] * inv_strength;
        float v2 = average * strength;
        float value = v1 + v2;
        int rounded = (int)(value + 0.5f);
        output[idx] = (ushort)min(65535, max(0, rounded));
    }
}

__kernel void a_out_u8(__global const ushort* src, __global uchar* dst,
                       const int start, const int n, const int C)
{
    int i = get_global_id(0);
    if (i >= n) return;
    int sample = start + i;
    int v = (int)src[sample] >> 8;
    if (C != 4 || (sample % 4) != 3) v = max(v - 1, 0);
    dst[i] = (uchar)v;
}

__kernel void a_out_f32(__global const ushort* src, __global float* dst,
                        const int start, const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    dst[i] = (float)src[start + i] * (1.0f / 257.0f);
}
'''


def _source_staging_bytes(frame, shape, img_resize, native_upload):
    from core.source_frames import frame_has_mapped_pixels, frame_resolution_bytes

    # Resolution accounts for decode and deferred processors (including ECC).
    # Mapped input may also fault in pages while it is being read.
    resolution = frame_resolution_bytes(frame)
    if frame_has_mapped_pixels(frame):
        resolution = max(resolution, frame.nbytes)
    if native_upload:
        packed = 0 if isinstance(frame, np.ndarray) and frame.flags.c_contiguous else frame.nbytes
        return resolution + packed + frame.nbytes  # packing and native upload staging

    source_strip = min(frame.shape[0], 128) * math.prod(frame.shape[1:])
    samples = math.prod(shape)
    # Upload can hold int32 codes, packed uint16 codes and native staging.
    upload = samples * 8
    conversion = frame.size * 4
    if img_resize is not None:
        output_strip = min(shape[0], 128) * math.prod(shape[1:])
        source_strip = max(source_strip, output_strip)
        # Area resize can hold both integer code planes and the float32
        # OpenCV input/output; keep room for conversion strips as well.
        conversion = 8 * (frame.size + samples)
    return resolution + max(upload, conversion + source_strip * 16)


class GpuA:
    """Device-resident OpenFocus A fusion with exact integer Burt responses."""

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
            raise GpuError('OpenCL GPU A initialization failed') from exc

    def _run(self, kernel, gsize, *args):
        return kernel(self.q, gsize if isinstance(gsize, tuple) else (gsize,), None, *args)

    def _drain_queue(self):
        q = getattr(self, 'q', None)
        if q is not None:
            try:
                q.finish()
            except Exception:
                _invalidate_program_cache(getattr(self, 'ctx', None))

    def fuse(self, frames, img_resize, radius, should_cancel=None, progress_callback=None,
             *, live_preview=None):
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
                args = (frames, img_resize, radius, should_cancel,
                        progress if progress_callback is not None else None)
                return (self._fuse_locked(*args) if live_preview is None else
                        self._fuse_locked(*args, live_preview=live_preview))
            except CallbackFailure as exc:
                self._drain_queue()
                raise exc.__cause__
            except (self.cl.Error, MemoryError) as exc:
                _invalidate_program_cache(getattr(self, 'ctx', None))
                self._drain_queue()
                raise GpuError('OpenCL GPU A execution failed') from exc
            except BaseException:
                self._drain_queue()
                raise
        finally:
            self.k = None
            self.q = None
            self.ctx = None
            _device_lock.release()

    def _fuse_locked(self, frames, img_resize, radius, should_cancel, progress_callback,
                     *, live_preview=None):
        from dataclasses import replace
        from core import resource_policy
        from core.source_frames import resolve_frame, without_source_cache_growth
        from fusion_methods._openfocus_gpu import (
            GpuError, _PreviewPlaneReader, _free_vram, _iter_prefetched, _prepare_gpu_upload,
            _release_buffers,
        )

        shape = frames[0].shape
        if img_resize is not None:
            tail = () if len(shape) == 2 or shape[2] == 1 else shape[2:]
            shape = (int(img_resize[1]), int(img_resize[0])) + tail
        height, width = int(shape[0]), int(shape[1])
        if min(height, width) < 1:
            raise GpuError('OpenFocus A GPU requires non-empty frames')
        channels = 1 if len(shape) == 2 else int(shape[2])
        n = height * width
        total = len(frames)
        if progress_callback is not None:
            progress_callback(0, total)
        _check_cancel(should_cancel)

        pyr_shapes = _hard_pyramid_shapes((height, width), radius)
        levels_count = len(pyr_shapes) - 1

        is_u8 = all(getattr(source, 'dtype', None) == np.uint8 for source in frames)
        native_upload = img_resize is None

        snapshot = resource_policy.probe_memory()
        # This serial stage can spend the reserve-protected headroom, including
        # the child job's hard limit. Managed frame residency has its own cap.
        host_budget = resource_policy.ram_ceiling(snapshot)
        samples = n * channels
        out_dtype = np.dtype(np.uint8 if is_u8 else np.float32)
        out_bytes = samples * out_dtype.itemsize
        readback_bytes = min(out_bytes, _READBACK_BUFFER_BYTES)
        source_peak = 0
        for frame in frames:
            _check_cancel(should_cancel)
            use_native = native_upload and frame.dtype in (np.uint8, np.uint16)
            source_peak = max(source_peak, _source_staging_bytes(frame, shape, img_resize, use_native))
        # Upload and readback never overlap. Readback holds the host output
        # and at most one transfer buffer, with driver overhead separate.
        staging = max(source_peak, out_bytes + readback_bytes) + 16 * 1024 * 1024
        if staging > host_budget:
            raise GpuError(f'OpenFocus A GPU staging exceeds the host memory budget '
                           f'({staging >> 20} MiB needed, {host_budget >> 20} MiB available)')

        # Keep the first reduced gray plane for selection. Deeper levels use
        # two alternating gray buffers and one packed coarse Laplacian buffer.
        gray_sizes = [h * w * 4 for h, w in pyr_shapes[1:4]]
        lap_bytes = sum(h * w * 4 for h, w in pyr_shapes[1:-1])
        src_bytes = samples * (1 if native_upload and is_u8 else 2)
        need = (sum(gray_sizes) + lap_bytes
                + n * 4 * 3          # best, weight, statistics
                + n * channels * 2   # output codes
                + n * channels * 4   # accumulated sums
                + max(src_bytes, readback_bytes)  # upload and readback are disjoint
                + 16 * 1024 * 1024)
        limit = int(self.device.global_mem_size) // 2
        unified_memory = self.device.host_unified_memory
        if unified_memory:
            limit = min(limit, host_budget - staging)
        else:
            memory = _free_vram(self.device)
            if memory is None:
                raise GpuError('OpenFocus A GPU free memory is unknown')
            snapshot = replace(snapshot, free_vram=memory[0], total_vram=memory[1])
            limit = min(limit, resource_policy.vram_budget(snapshot))
        single = int(self.device.max_mem_alloc_size)
        largest = max(n * channels * 4, n * 4)
        if need > limit or largest > single:
            raise GpuError('OpenFocus A GPU workspace exceeds the device budget')

        gray_bufs = lap_buf = best_buf = out_buf = acc_buf = wt_buf = stat_buf = dst_buf = None
        src_bufs = ()
        reader = prepared_iter = None
        in_flight = False
        try:
            from fusion_methods._openfocus_gpu import _prefetch_window, _wait_event

            gray_bufs = [self.cl.Buffer(self.ctx, self.mf.READ_WRITE, size) for size in gray_sizes]
            lap_buf = (self.cl.Buffer(self.ctx, self.mf.READ_WRITE, lap_bytes)
                       if lap_bytes else gray_bufs[0])
            best_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, n * 4))
            out_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, n * channels * 2))
            acc_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, n * channels * 4))
            wt_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, n * 4))
            stat_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, n * 4))

            prefetch_budget = host_budget - need if unified_memory else host_budget
            prefetch_slots = _prefetch_window(
                prefetch_budget, staging, source_peak, total,
            )
            ping_pong = (
                total >= 2
                and (need + src_bytes) <= limit
                and (staging + prefetch_slots * source_peak + 2 * src_bytes
                     + (src_bytes if unified_memory else 0)) <= prefetch_budget
            )
            src_bufs = tuple(
                self.cl.Buffer(self.ctx, self.mf.READ_ONLY, src_bytes)
                for _ in range(2 if ping_pong else 1)
            )
            if live_preview is not None:
                reader = _PreviewPlaneReader(self.cl, self.ctx, self.q, self.k['sample_focus_plane'],
                                             (height, width), np.int32)

            def prepare_frame(_index, source, cancelled):
                _check_cancel(cancelled)
                with without_source_cache_growth():
                    raw = resolve_frame(source, should_cancel=cancelled)
                if native_upload and source.dtype in (np.uint8, np.uint16):
                    _check_cancel(cancelled)
                    if tuple(raw.shape) != tuple(shape):
                        raise ValueError('OpenFocus A requires frames with matching shapes')
                    host_src, bgr_rev = _prepare_gpu_upload(raw)
                    del raw
                    src_is_u8 = host_src.dtype == np.uint8
                    return host_src, src_is_u8, 0, bgr_rev
                raw_code = frame_codes(raw, img_resize, cancelled)
                del raw
                if tuple(raw_code.shape) != tuple(shape):
                    raise ValueError('OpenFocus A requires frames with matching shapes')
                host_src, bgr_rev = _prepare_gpu_upload(raw_code, np.uint16)
                del raw_code
                u8_gray = 1 if getattr(source, 'dtype', None) == np.uint8 else 0
                return host_src, False, u8_gray, bgr_rev

            prepared_iter = _iter_prefetched(
                frames, prepare_frame, enabled=prefetch_slots > 0,
                should_cancel=should_cancel, max_in_flight=max(1, prefetch_slots),
            )
            slot_events = [None, None]
            slot_hosts = [None, None]
            for index, (host_src, src_is_u8, u8_gray, bgr_rev) in prepared_iter:
                slot = index & 1 if ping_pong else 0
                if slot_events[slot] is not None:
                    _wait_event(slot_events[slot], lambda: _check_cancel(should_cancel))
                    slot_events[slot] = None
                    slot_hosts[slot] = None
                _check_cancel(should_cancel)
                src_buf = src_bufs[slot]
                in_flight = True
                upload = self.cl.enqueue_copy(self.q, src_buf, host_src, is_blocking=False)
                _check_cancel(should_cancel)
                rh, rw = pyr_shapes[1]
                down2 = self.k['a_down2_u8'] if src_is_u8 else self.k['a_down2_u16']
                self._run(down2, (rw, rh), src_buf, gray_bufs[0], np.int32(height),
                          np.int32(width), np.int32(rh), np.int32(rw), np.int32(channels),
                          np.int32(u8_gray))

                offset = 0
                gray_slot = 0
                for level in range(1, levels_count):
                    _check_cancel(should_cancel)
                    gh, gw = pyr_shapes[level]
                    rh, rw = pyr_shapes[level + 1]
                    reduced_slot = 1 + (level - 1) % 2
                    self._run(self.k['a_down2'], (rw, rh), gray_bufs[gray_slot],
                              gray_bufs[reduced_slot], np.int32(gh), np.int32(gw),
                              np.int32(rh), np.int32(rw))
                    self._run(self.k['a_laplacian'], (gw, gh), gray_bufs[gray_slot],
                              gray_bufs[reduced_slot], lap_buf, np.int32(offset),
                              np.int32(gh), np.int32(gw), np.int32(rh), np.int32(rw))
                    offset += gh * gw
                    gray_slot = reduced_slot

                _check_cancel(should_cancel)
                if src_is_u8:
                    kern_name = 'a_select_u8_bgr_rev' if bgr_rev else 'a_select_u8'
                else:
                    kern_name = 'a_select_u16_bgr_rev' if bgr_rev else 'a_select_u16'
                last_event = self._run(
                    self.k[kern_name], (width, height), src_buf, gray_bufs[0], lap_buf,
                    best_buf, out_buf, acc_buf, wt_buf, stat_buf, np.int32(height),
                    np.int32(width), np.int32(channels), np.int32(levels_count),
                    np.int32(index == 0), np.int32(u8_gray),
                )
                if ping_pong and last_event is not None and hasattr(last_event, 'wait'):
                    slot_events[slot] = last_event
                    slot_hosts[slot] = (host_src, upload)
                    if reader is not None and live_preview.tracks():
                        _wait_event(last_event, lambda: _check_cancel(should_cancel))
                        slot_events[slot] = None
                        slot_hosts[slot] = None
                        sample = reader.read(best_buf, live_preview)
                        if sample is not None:
                            live_preview.update_best(sample, index, total, sampled=True)
                else:
                    self.q.finish()
                    in_flight = False
                    if reader is not None and live_preview.tracks():
                        _check_cancel(should_cancel)
                        sample = reader.read(best_buf, live_preview)
                        if sample is not None:
                            live_preview.update_best(sample, index, total, sampled=True)
                    del host_src, upload
                if progress_callback is not None:
                    progress_callback(index + 1, total)

            if ping_pong:
                for slot in (0, 1):
                    if slot_events[slot] is not None:
                        _wait_event(slot_events[slot], lambda: _check_cancel(should_cancel))
                        slot_events[slot] = None
                    slot_hosts[slot] = None
                self.q.finish()
                in_flight = False

            _release_buffers(src_bufs, gray_bufs, lap_buf)
            src_bufs = gray_bufs = lap_buf = None
            _check_cancel(should_cancel)
            in_flight = True
            self._run(self.k['a_blend_low'], n, out_buf, best_buf,
                      acc_buf, wt_buf, stat_buf, np.int32(n), np.int32(channels))

            out = np.empty(shape, dtype=out_dtype)
            out_samples = out.reshape(-1)
            dst_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, readback_bytes)
            chunk = readback_bytes // out_dtype.itemsize
            kernel = self.k['a_out_u8'] if is_u8 else self.k['a_out_f32']
            extra = (np.int32(channels),) if is_u8 else ()
            for start in range(0, samples, chunk):
                _check_cancel(should_cancel)
                region = out_samples[start:min(start + chunk, samples)]
                self._run(kernel, region.size, out_buf, dst_buf, np.int32(start),
                          np.int32(region.size), *extra)
                self.cl.enqueue_copy(self.q, region, dst_buf, is_blocking=True)
            in_flight = False
            _check_cancel(should_cancel)
            return out
        except BaseException:
            if in_flight:
                self._drain_queue()
            raise
        finally:
            if prepared_iter is not None:
                prepared_iter.close()
            if reader is not None:
                reader.close()
            _release_buffers(gray_bufs, lap_buf, best_buf, out_buf, acc_buf, wt_buf, stat_buf, src_bufs, dst_buf)
