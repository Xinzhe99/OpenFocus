"""OpenFocus A GPU fusion (opt-in via use_gpu=True on openfocus_a_impl).

Bit-exact port of OpenFocus A's default (hard) Burt-pyramid response,
winner-take-all selection with low-response background accumulation,
low-response blending, and output code exit onto OpenCL kernels.
"""
from __future__ import annotations

import numpy as np

from fusion_methods._openfocus_pixels import frame_codes
from fusion_methods.openfocus_ab import _check_cancel

_A_K = r'''
#pragma OPENCL FP_CONTRACT OFF

__kernel void a_gray_u8(__global const uchar* src, __global int* gray,
                        const int n, const int C)
{
    int i = get_global_id(0);
    if (i >= n) return;
    if (C == 1) {
        gray[i] = (int)src[i] << 8;
        return;
    }
    size_t base = (size_t)i * C;
    int total = (int)src[base] + (int)src[base + 1] + (C >= 3 ? (int)src[base + 2] : 0);
    gray[i] = (total / 3) << 8;
}

__kernel void a_gray_u16(__global const ushort* src, __global int* gray,
                         const int n, const int C, const int u8_gray)
{
    int i = get_global_id(0);
    if (i >= n) return;
    if (C == 1) {
        gray[i] = (int)src[i];
        return;
    }
    size_t base = (size_t)i * C;
    int total = (int)src[base] + (int)src[base + 1] + (C >= 3 ? (int)src[base + 2] : 0);
    gray[i] = u8_gray ? (((total >> 8) / 3) << 8) : (total / 3);
}

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
    long total = (long)gray[(size_t)top * W + left]
               + (long)gray[(size_t)top * W + right]
               + (long)gray[(size_t)bottom * W + left]
               + (long)gray[(size_t)bottom * W + right];
    out[(size_t)y * OW + x] = (int)(total >> 2);
}

__kernel void a_laplacian0(__global const int* gray, __global const int* reduced,
                           __global int* resp, const int GH, const int GW,
                           const int RH, const int RW)
{
    int x = get_global_id(0);
    if (x >= GW) return;
    int y = get_global_id(1);
    if (y >= GH) return;
    int y0 = (y >> 1) - ((y & 1) == 0 ? 1 : 0);
    int y1 = min(y0 + 1, RH - 1);
    y0 = max(y0, 0);
    long wy0 = (y & 1) == 0 ? 1L : 3L;
    long wy1 = 4L - wy0;
    int x0 = (x >> 1) - ((x & 1) == 0 ? 1 : 0);
    int x1 = min(x0 + 1, RW - 1);
    x0 = max(x0, 0);
    long wx0 = (x & 1) == 0 ? 1L : 3L;
    long wx1 = 4L - wx0;
    long top = (long)reduced[(size_t)y0 * RW + x0] * wx0 + (long)reduced[(size_t)y0 * RW + x1] * wx1;
    long bottom = (long)reduced[(size_t)y1 * RW + x0] * wx0 + (long)reduced[(size_t)y1 * RW + x1] * wx1;
    long value = (long)gray[(size_t)y * GW + x] * 16L - top * wy0 - bottom * wy1;
    long abs_val = value < 0L ? -value : value;
    resp[(size_t)y * GW + x] = (int)min(abs_val, 65535L * 16L);
}

__kernel void a_laplacian(__global const int* gray, __global const int* reduced,
                          __global int* out, const int GH, const int GW,
                          const int RH, const int RW)
{
    int x = get_global_id(0);
    if (x >= GW) return;
    int y = get_global_id(1);
    if (y >= GH) return;
    int y0 = (y >> 1) - ((y & 1) == 0 ? 1 : 0);
    int y1 = min(y0 + 1, RH - 1);
    y0 = max(y0, 0);
    long wy0 = (y & 1) == 0 ? 1L : 3L;
    long wy1 = 4L - wy0;
    int x0 = (x >> 1) - ((x & 1) == 0 ? 1 : 0);
    int x1 = min(x0 + 1, RW - 1);
    x0 = max(x0, 0);
    long wx0 = (x & 1) == 0 ? 1L : 3L;
    long wx1 = 4L - wx0;
    long top = (long)reduced[(size_t)y0 * RW + x0] * wx0 + (long)reduced[(size_t)y0 * RW + x1] * wx1;
    long bottom = (long)reduced[(size_t)y1 * RW + x0] * wx0 + (long)reduced[(size_t)y1 * RW + x1] * wx1;
    long value = (long)gray[(size_t)y * GW + x] * 16L - top * wy0 - bottom * wy1;
    out[(size_t)y * GW + x] = (int)(value < 0L ? -value : value);
}

__kernel void a_accumulate(__global const int* lap, __global int* resp,
                           const int H, const int W, const int LH, const int LW,
                           const int level)
{
    int x = get_global_id(0);
    if (x >= W) return;
    int y = get_global_id(1);
    if (y >= H) return;
    int shift = level + 1;
    int factor = 1 << level;
    long den = 1L << shift;
    long rounding = 1L << (2 * shift - 1);
    int py = 2 * y + 1 - factor;
    int y0 = (py >= 0) ? (py >> shift) : -((-py + (int)den - 1) >> shift);
    long fy = (long)py - (long)y0 * den;
    int ya = max(0, min(y0, LH - 1));
    int yb = max(0, min(y0 + 1, LH - 1));
    int px = 2 * x + 1 - factor;
    int x0 = (px >= 0) ? (px >> shift) : -((-px + (int)den - 1) >> shift);
    long fx = (long)px - (long)x0 * den;
    int xa = max(0, min(x0, LW - 1));
    int xb = max(0, min(x0 + 1, LW - 1));
    long left = (long)lap[(size_t)ya * LW + xa] * (den - fy) + (long)lap[(size_t)yb * LW + xa] * fy;
    long right = (long)lap[(size_t)ya * LW + xb] * (den - fy) + (long)lap[(size_t)yb * LW + xb] * fy;
    long value = ((left << shift) + (right - left) * fx + rounding) >> (2 * shift);
    size_t idx = (size_t)y * W + x;
    long total = (long)resp[idx] + value;
    resp[idx] = (int)min(total, 65535L * 16L);
}

#define DEFINE_A_SELECT(NAME, PIXEL_T, SCALE) \
__kernel void NAME(__global const PIXEL_T* image, __global const int* response, \
                   __global int* best, __global ushort* output, \
                   __global float* accumulated, __global float* weights, \
                   __global ushort* stats, const int n, const int C, \
                   const int first) \
{ \
    int k = get_global_id(0); \
    if (k >= n) return; \
    long score = (long)response[k]; \
    long previous = first ? 0L : (long)best[k]; \
    int take = first || (score > previous); \
    int collect = first || (previous < 65536L); \
    float factor = 1.0f; \
    float weight = 0.0f; \
    if (collect) { \
        ushort current_code = (ushort)(score >> 4); \
        size_t sk = (size_t)k * 2; \
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
        int value = (int)image[idx] * SCALE; \
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

DEFINE_A_SELECT(a_select_u8, uchar, 256)
DEFINE_A_SELECT(a_select_u16, ushort, 1)

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
                       const int n, const int C)
{
    int i = get_global_id(0);
    if (i >= n) return;
    int v = (int)src[i] >> 8;
    if (C != 4 || (i % 4) != 3) v = max(v - 1, 0);
    dst[i] = (uchar)v;
}

__kernel void a_out_f32(__global const ushort* src, __global float* dst, const int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    dst[i] = (float)src[i] * (1.0f / 257.0f);
}
'''


class GpuA:
    """Device-resident OpenFocus A fusion; mirrors openfocus_a_impl (hard) step for step."""

    def __init__(self):
        from fusion_methods._openfocus_gpu import GpuError, _device
        try:
            import pyopencl as cl
            device = _device(cl)
            if device is None:
                raise GpuError('No available OpenCL GPU')
            self.cl = cl
            self.device = device
            self.device_name = 'OpenCL: ' + str(device.name).strip()
            self.ctx = cl.Context(devices=[device])
            self.q = cl.CommandQueue(self.ctx, device=device)
            self.mf = cl.mem_flags
            prog = cl.Program(self.ctx, _A_K).build(
                options=['-cl-fp32-correctly-rounded-divide-sqrt'])
            self.k = {name: cl.Kernel(prog, name) for name in (
                'a_gray_u8', 'a_gray_u16', 'a_down2', 'a_laplacian0',
                'a_laplacian', 'a_accumulate', 'a_select_u8', 'a_select_u16',
                'a_blend_low', 'a_out_u8', 'a_out_f32',
            )}
        except GpuError:
            raise
        except Exception as exc:
            raise GpuError('OpenCL GPU A initialization failed') from exc

    def _run(self, kernel, gsize, *args):
        kernel(self.q, gsize if isinstance(gsize, tuple) else (gsize,), None, *args)

    def fuse(self, frames, img_resize, radius, should_cancel=None, progress_callback=None):
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
                return self._fuse_locked(frames, img_resize, radius, should_cancel,
                                         progress if progress_callback is not None else None)
            except CallbackFailure as exc:
                raise exc.__cause__
            except (self.cl.Error, MemoryError) as exc:
                raise GpuError('OpenCL GPU A execution failed') from exc
        finally:
            self.k = None
            self.q = None
            self.ctx = None
            _device_lock.release()

    def _fuse_locked(self, frames, img_resize, radius, should_cancel, progress_callback):
        from dataclasses import replace
        from core import resource_policy
        from core.multi_focus_fusion import memory_budget_bytes
        from core.source_frames import frame_resolution_bytes, resolve_frame, source_cache_growth_bytes
        from fusion_methods._openfocus_gpu import GpuError, _free_vram

        shape = frames[0].shape
        if img_resize is not None:
            tail = () if len(shape) == 2 or shape[2] == 1 else shape[2:]
            shape = (int(img_resize[1]), int(img_resize[0])) + tail
        height, width = int(shape[0]), int(shape[1])
        if min(height, width) < 1:
            raise GpuError('OpenFocus A GPU requires non-empty frames')
        is_gray = len(shape) == 2
        channels = 1 if is_gray else int(shape[2])
        n = height * width
        total = len(frames)
        if progress_callback is not None:
            progress_callback(0, total)
        _check_cancel(should_cancel)

        levels_count = int(np.floor(np.log2(max(1, radius)))) + 3
        pyr_shapes = [(height, width)]
        for _ in range(levels_count):
            ph, pw = pyr_shapes[-1]
            pyr_shapes.append(((ph + 1) // 2, (pw + 1) // 2))

        is_u8 = all(getattr(source, 'dtype', None) == np.uint8 for source in frames)
        fast_hard = img_resize is None and all(
            getattr(source, 'dtype', None) in (np.uint8, np.uint16) for source in frames)

        snapshot = resource_policy.probe_resources()
        host_budget = min(int(memory_budget_bytes()),
                          int(resource_policy.frame_ram_budget(snapshot)))
        out_sample_bytes = 1 if is_u8 else 4
        if fast_hard:
            in_sample_bytes = 2 if any(getattr(f, 'dtype', None) == np.uint16 for f in frames) else 1
        else:
            in_sample_bytes = 6
        staging = (n * channels * (out_sample_bytes + in_sample_bytes)
                   + max((frame_resolution_bytes(frame) for frame in frames), default=0)
                   + source_cache_growth_bytes(frames) + 16 * 1024 * 1024)
        if staging > host_budget:
            raise GpuError('OpenFocus A GPU staging exceeds the host memory budget')

        lap_bytes = max(1, pyr_shapes[1][0] * pyr_shapes[1][1]) * 4 if levels_count > 1 else 0
        need = (sum(max(1, h * w) * 4 for h, w in pyr_shapes)
                + lap_bytes
                + n * 4 * 4          # resp, best, wt, stat
                + n * channels * 2   # output codes
                + n * channels * 4   # accumulated sums
                + n * channels * 4   # per-frame source / final output buffer
                + 16 * 1024 * 1024)
        limit = int(self.device.global_mem_size) // 2
        if self.device.host_unified_memory:
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

        gray_bufs = [self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, h * w) * 4)
                     for h, w in pyr_shapes]
        lap_buf = (self.cl.Buffer(self.ctx, self.mf.READ_WRITE, lap_bytes)
                   if lap_bytes else None)
        resp_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, n * 4))
        best_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, n * 4))
        out_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, n * channels * 2))
        acc_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, n * channels * 4))
        wt_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, n * 4))
        stat_buf = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, max(1, n * 4))

        for index, source in enumerate(frames):
            _check_cancel(should_cancel)
            if fast_hard:
                raw = resolve_frame(source)
                _check_cancel(should_cancel)
                if tuple(raw.shape) != tuple(shape):
                    raise ValueError('OpenFocus A requires frames with matching shapes')
                host_src = np.ascontiguousarray(raw)
                src_buf = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                         hostbuf=host_src)
                src_is_u8 = host_src.dtype == np.uint8
                if src_is_u8:
                    self._run(self.k['a_gray_u8'], n, src_buf, gray_bufs[0],
                              np.int32(n), np.int32(channels))
                else:
                    self._run(self.k['a_gray_u16'], n, src_buf, gray_bufs[0],
                              np.int32(n), np.int32(channels), np.int32(0))
            else:
                raw_code = frame_codes(source, img_resize, should_cancel)
                if tuple(raw_code.shape) != tuple(shape):
                    raise ValueError('OpenFocus A requires frames with matching shapes')
                host_src = np.ascontiguousarray(raw_code, dtype=np.uint16)
                src_buf = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                         hostbuf=host_src)
                u8_gray = 1 if getattr(source, 'dtype', None) == np.uint8 else 0
                src_is_u8 = False
                self._run(self.k['a_gray_u16'], n, src_buf, gray_bufs[0],
                          np.int32(n), np.int32(channels), np.int32(u8_gray))

            for level in range(levels_count):
                _check_cancel(should_cancel)
                gh, gw = pyr_shapes[level]
                rh, rw = pyr_shapes[level + 1]
                self._run(self.k['a_down2'], (rw, rh),
                          gray_bufs[level], gray_bufs[level + 1],
                          np.int32(gh), np.int32(gw), np.int32(rh), np.int32(rw))
                if level == 0:
                    self._run(self.k['a_laplacian0'], (gw, gh),
                              gray_bufs[level], gray_bufs[level + 1], resp_buf,
                              np.int32(gh), np.int32(gw), np.int32(rh), np.int32(rw))
                else:
                    self._run(self.k['a_laplacian'], (gw, gh),
                              gray_bufs[level], gray_bufs[level + 1], lap_buf,
                              np.int32(gh), np.int32(gw), np.int32(rh), np.int32(rw))
                    self._run(self.k['a_accumulate'], (width, height),
                              lap_buf, resp_buf,
                              np.int32(height), np.int32(width),
                              np.int32(gh), np.int32(gw), np.int32(level))

            select_kern = self.k['a_select_u8'] if src_is_u8 else self.k['a_select_u16']
            self._run(select_kern, n, src_buf, resp_buf, best_buf, out_buf,
                      acc_buf, wt_buf, stat_buf, np.int32(n), np.int32(channels),
                      np.int32(1 if index == 0 else 0))
            # Bound queued frame uploads to one source, even for very deep stacks.
            self.q.finish()
            del src_buf, host_src
            if progress_callback is not None:
                progress_callback(index + 1, total)

        _check_cancel(should_cancel)
        self._run(self.k['a_blend_low'], n, out_buf, best_buf,
                  acc_buf, wt_buf, stat_buf, np.int32(n), np.int32(channels))

        out_dtype = np.uint8 if is_u8 else np.float32
        out = np.empty(shape, dtype=out_dtype)
        dst_buf = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, max(1, out.nbytes))
        total_samples = n * channels
        if is_u8:
            self._run(self.k['a_out_u8'], total_samples, out_buf, dst_buf,
                      np.int32(total_samples), np.int32(channels))
        else:
            self._run(self.k['a_out_f32'], total_samples, out_buf, dst_buf,
                      np.int32(total_samples))
        self.cl.enqueue_copy(self.q, out, dst_buf, is_blocking=True)
        _check_cancel(should_cancel)
        return out
