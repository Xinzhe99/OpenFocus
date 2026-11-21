# OpenFocus C GPU fusion (opt-in via use_gpu=True on openfocus_c_impl).
#
# Bit-exact port of fusion_methods._openfocus_pyramid.fuse_pyramid onto
# OpenCL kernels. The program builds with -cl-fp32-correctly-rounded-divide-sqrt
# and every mul+add pair is split across statements so device results match
# the numba CPU path bit for bit (verified: 92-frame stack ndiff=0).
# The per-level weight filter reuses the proven filt4 IIR kernel from
# _openfocus_gpu via device-side transposes, replicating
# openfocus_depth._filter's axis order (1 then 0) and per-axis counts.
from __future__ import annotations

from threading import Lock

import numpy as np

from fusion_methods.openfocus_bc import _pyramid_levels, _source_codes
from fusion_methods.openfocus_ab import _check_cancel

_KERNEL_NAMES = (
    'c_norm', 'c_norm_u8', 'c_down_h', 'c_down_v', 'c_up_h',
    'c_detail_w', 'c_collapse', 'c_merge', 'c_merge_focus',
    'c_out_u8', 'c_out_f32', 'transpose_u16', 'filt4',
)

_cache_lock = Lock()
_PROGRAM_CACHE = None


def _get_or_build_program(cl, device):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import _FILTER_K, _device_cache_key

    key = _device_cache_key(device)
    with _cache_lock:
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key:
            return _PROGRAM_CACHE[1:]
        ctx = cl.Context(devices=[device])
        queue = cl.CommandQueue(ctx, device=device)
        program = cl.Program(ctx, _FILTER_K + _C_K).build(
            options=['-cl-fp32-correctly-rounded-divide-sqrt'],
        )
        kernels = {name: cl.Kernel(program, name) for name in _KERNEL_NAMES}
        _PROGRAM_CACHE = (key, ctx, queue, kernels)
        return ctx, queue, kernels


def _invalidate_program_cache():
    global _PROGRAM_CACHE
    with _cache_lock:
        _PROGRAM_CACHE = None

_C_K = r'''
#pragma OPENCL FP_CONTRACT OFF
__constant float C_DOWN[4]={0.125f,0.375f,0.375f,0.125f};
__constant float C_EVEN[5]={0.019027505f,0.318410337f,0.559359729f,0.103202470f,0.0f};
__constant float C_ODD[5]={0.0f,0.103202470f,0.559359729f,0.318410337f,0.019027505f};

__kernel void c_norm(__global const ushort* src, __global float* dst,
                     const float factor, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    dst[i]=(float)src[i]*factor;
}

__kernel void c_norm_u8(__global const uchar* src, __global float* dst,
                        const float factor, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    dst[i]=(float)src[i]*factor;
}

__kernel void c_down_h(__global const float* s, __global float* d,
                       const int W, const int H, const int OW, const int C)
{
    int x=get_global_id(0); if(x>=OW)return;
    int y=get_global_id(1); if(y>=H)return;
    for(int c=0;c<C;c++){
        float v=0.0f;
        for(int k=0;k<4;k++){
            int sx=min(W-1,max(0,2*x+k-1));
            float p=s[(y*W+sx)*C+c]*C_DOWN[k]; v=v+p;
        }
        d[(y*OW+x)*C+c]=v;
    }
}

__kernel void c_down_v(__global const float* s, __global float* d,
                       const int W, const int H, const int OW, const int OH,
                       const int C)
{
    int x=get_global_id(0); if(x>=OW)return;
    int y=get_global_id(1); if(y>=OH)return;
    for(int c=0;c<C;c++){
        float v=0.0f;
        for(int k=0;k<4;k++){
            int sy=min(H-1,max(0,2*y+k-1));
            float p=s[(sy*W+x)*C+c]*C_DOWN[k]; v=v+p;
        }
        d[(y*OW+x)*C+c]=v;
    }
}

static inline float c_up_total(const __constant float* w, int origin, int length)
{
    float total=0.0f;
    for(int k=0;k<5;k++){
        int i=origin+k;
        if(i>=0&&i<length) total=total+w[k];
    }
    return total;
}

static inline float c_up_value(__global const float* s, const __constant float* w,
                               int offset, int stride, int origin, int length, float total)
{
    float value=0.0f;
    for(int k=0;k<5;k++){
        int i=origin+k;
        if(i>=0&&i<length){float p=s[offset+i*stride]*w[k]; value=value+p;}
    }
    return value/total;
}

__kernel void c_up_h(__global const float* s, __global float* d,
                     const int SW, const int H, const int DW, const int C)
{
    int x=get_global_id(0); if(x>=DW)return;
    int y=get_global_id(1); if(y>=H)return;
    // Interleave both horizontal phases in fine-width pixel order.
    const __constant float* w=(x&1)?C_ODD:C_EVEN;
    int sx=(x>>1)-2;
    float total=c_up_total(w,sx,SW);
    for(int c=0;c<C;c++)
        d[(y*DW+x)*C+c]=c_up_value(s,w,y*SW*C+c,C,sx,SW,total);
}

__kernel void c_detail_w(__global float* fine, __global const float* horizontal,
                         __global ushort* weight, const int W, const int H,
                         const int SH, const int C)
{
    int x=get_global_id(0); if(x>=W)return;
    int y=get_global_id(1); if(y>=H)return;
    const __constant float* w=(y&1)?C_ODD:C_EVEN;
    int sy=(y>>1)-2;
    float total=c_up_total(w,sy,SH);
    float st=0.0f;
    for(int c=0;c<C;c++){
        int i=(y*W+x)*C+c;
        float lo=c_up_value(horizontal,w,x*C+c,W*C,sy,SH,total);
        float v=(fine[i]-lo)/(lo+0.1f);
        fine[i]=v;
        if(c<3) st=st+fabs(v);
    }
    if(C==1) st=st*3.0f;
    else if(C==2) st=st*1.5f;
    int border=(H>2&&(y==0||y==H-1))||(W>2&&(x==0||x==W-1));
    int wi=border?0:min(65535,(int)(st*21845.0f));
    weight[y*W+x]=(ushort)wi;
}

__kernel void c_merge(__global const float* detail, __global const ushort* weight,
                      __global float* acc, __global ushort* best,
                      const int first, const int W, const int H, const int C)
{
    int x=get_global_id(0); if(x>=W)return;
    int y=get_global_id(1); if(y>=H)return;
    int k=y*W+x;
    ushort w=weight[k];
    if(first||w>best[k]){
        best[k]=w;
        for(int c=0;c<C;c++) acc[k*C+c]=detail[k*C+c];
    }
}

__kernel void c_merge_focus(__global const float* detail, __global const ushort* weight,
                            __global float* acc, __global ushort* best,
                            __global ushort* winner, const int index,
                            const int W, const int H, const int C)
{
    int x=get_global_id(0); if(x>=W)return;
    int y=get_global_id(1); if(y>=H)return;
    int k=y*W+x;
    ushort w=weight[k];
    if(index==0||w>best[k]){
        best[k]=w;
        winner[k]=(ushort)index;
        for(int c=0;c<C;c++) acc[k*C+c]=detail[k*C+c];
    }
}

__kernel void c_collapse(__global float* detail, __global const float* horizontal,
                         const int W, const int H, const int SH, const int C)
{
    int x=get_global_id(0); if(x>=W)return;
    int y=get_global_id(1); if(y>=H)return;
    const __constant float* w=(y&1)?C_ODD:C_EVEN;
    int sy=(y>>1)-2;
    float total=c_up_total(w,sy,SH);
    for(int c=0;c<C;c++){
        int i=(y*W+x)*C+c;
        float v=c_up_value(horizontal,w,x*C+c,W*C,sy,SH,total);
        float t=(v+0.1f)*detail[i]; float value=t+v;
        detail[i]=fmin(1.0f,fmax(0.0f,value));
    }
}

__kernel void c_out_u8(__global const float* src, __global uchar* dst, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    float v=fmin(1.0f,fmax(0.0f,src[i]));
    int code=(int)(v*65535.0f);
    dst[i]=(uchar)(code>>8);
}

__kernel void c_out_f32(__global const float* src, __global float* dst, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    float v=fmin(1.0f,fmax(0.0f,src[i]));
    float code=v*65535.0f;
    dst[i]=rint(code)*(1.0f/257.0f);
}
'''


class GpuC:
    """Device-resident C pyramid fusion with bounded, in-order scratch reuse."""

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
            raise GpuError('OpenCL GPU C initialization failed') from exc

    def _f32(self, pixels, channels):
        return self.cl.Buffer(self.ctx, self.mf.READ_WRITE, pixels * channels * 4)

    def _u16(self, pixels):
        return self.cl.Buffer(self.ctx, self.mf.READ_WRITE, pixels * 2)

    def _run(self, kernel, gsize, *args):
        kernel(self.q, gsize if isinstance(gsize, tuple) else (gsize,), None, *args)

    def _filt4(self, src, dst, first, h, w, count, q, t):
        step = (w + 3) // 4
        self.k['filt4'](self.q, (step,), None, src, dst, first,
                        np.int32(h), np.int32(w), np.int32(count),
                        np.uint32(q), np.uint32(t), np.int32(step))

    def _filter_w(self, weight, scratch, h, w, config):
        """Device-side openfocus_depth._filter: axis 1 (horizontal) then 0."""
        a, first = scratch
        count_h, count_v, q, t = config
        # filt4 reads src[y,x] before writing that cell, so src==dst is safe.
        self._run(self.k['transpose_u16'], (w, h), weight, a,
                  np.int32(w), np.int32(h))
        self._filt4(a, a, first, w, h, count_h, q, t)
        self._run(self.k['transpose_u16'], (h, w), a, weight,
                  np.int32(h), np.int32(w))
        self._filt4(weight, weight, first, h, w, count_v, q, t)

    def _synthesize(self, kernel, coarse, horizontal, target, sh, sw, dh, dw, c,
                    *extra):
        self._run(self.k['c_up_h'], (dw, sh), coarse, horizontal,
                  np.int32(sw), np.int32(sh), np.int32(dw), np.int32(c))
        self._run(kernel, (dw, dh), target, horizontal, *extra,
                  np.int32(dw), np.int32(dh), np.int32(sh), np.int32(c))

    def _drain_queue(self):
        q = getattr(self, 'q', None)
        if q is not None:
            try:
                q.finish()
            except Exception:
                _invalidate_program_cache()

    def fuse(self, frames, shape, img_resize, smoothing,
             should_cancel=None, progress_callback=None, focus_callback=None):
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
                args = (frames, shape, img_resize, smoothing, should_cancel,
                        progress if progress_callback is not None else None)
                if focus_callback is not None:
                    return self._fuse_locked(*args, focus_callback=focus_callback)
                return self._fuse_locked(*args)
            except CallbackFailure as exc:
                self._drain_queue()
                raise exc.__cause__
            except (self.cl.Error, MemoryError) as exc:
                _invalidate_program_cache()
                self._drain_queue()
                raise GpuError('OpenCL GPU C execution failed') from exc
            except BaseException:
                self._drain_queue()
                raise
        finally:
            self.k = self.q = self.ctx = None
            _device_lock.release()

    def _fuse_locked(self, frames, shape, img_resize, smoothing,
                     should_cancel, progress_callback, focus_callback=None):
        height, width = shape[0], shape[1]
        is_gray = len(shape) == 2
        channels = 1 if is_gray else shape[2]
        is_u8 = all(getattr(f, 'dtype', None) == np.uint8 for f in frames)
        levels = _pyramid_levels(height, width)
        detail_levels = list(zip(levels, levels[1:]))
        count = len(frames)
        total = count + 1
        if progress_callback is not None:
            progress_callback(0, total)
        parameter = np.float32(max(0.0, (float(smoothing) - 1.0) / 5.0))
        out_dtype = np.uint8 if is_u8 else np.float32
        out_bytes = 1 if is_u8 else 4
        from dataclasses import replace
        from core import resource_policy
        from core.source_frames import (
            frame_has_mapped_pixels, frame_resolution_bytes, without_source_cache_growth,
        )
        from fusion_methods._openfocus_gpu import GpuError, _free_vram

        snapshot = resource_policy.probe_resources()
        host_budget = resource_policy.ram_ceiling(snapshot)
        in_bytes = 1 if (img_resize is None and is_u8) else 2
        full_size = height * width
        n = full_size * channels
        source_peak = 0
        for frame in frames:
            _check_cancel(should_cancel)
            resolution = frame_resolution_bytes(frame)
            if frame_has_mapped_pixels(frame):
                resolution = max(resolution, frame.nbytes)
            if img_resize is None and frame.dtype in (np.uint8, np.uint16):
                packed = isinstance(frame, np.ndarray) and frame.flags.c_contiguous
                working = frame.nbytes * (1 if packed else 2)
            else:
                # Legacy conversion can hold int32 codes and OpenCV's float32
                # inputs/outputs. The packed upload may also need a driver copy.
                row_samples = max(min(frame.shape[0], 128) * int(np.prod(frame.shape[1:])),
                                  min(height, 128) * width * channels)
                conversion = (frame.size * 4 if img_resize is None
                              else 8 * (frame.size + n))
                working = max(n * 8, conversion + row_samples * 16)
            source_peak = max(source_peak, resolution + working)
        readback = max(2 * n * out_bytes,
                       n * out_bytes + 4 * full_size if focus_callback is not None else 0)
        staging = max(source_peak, readback) + 16 * 1024 * 1024
        if staging > host_budget:
            raise GpuError('OpenFocus C GPU staging exceeds the host memory budget')
        # Each size matches an allocation below. The queue is in order, so
        # a smaller pyramid level can reuse a prefix of the same scratch plane.
        detail_sizes = [h * w for (h, w), _ in detail_levels]
        coarsest_size = levels[-1][0] * levels[-1][1]
        rolling_sizes = (full_size, levels[1][0] * levels[1][1]) if detail_levels else ()
        horizontal_size = (max(height * levels[1][1], levels[1][0] * width)
                           if detail_levels else 0)
        filter_planes = 2 if detail_levels and parameter > 0 else 0
        keep_winner = focus_callback is not None and bool(detail_levels)
        need = 4 * channels * (sum(rolling_sizes) + horizontal_size + coarsest_size)
        need += sum(detail_sizes) * (4 * channels + 2)
        need += 2 * full_size * (bool(detail_levels) + filter_planes + keep_winner)
        need += n * (in_bytes + out_bytes) + 16 * 1024 * 1024
        limit = int(self.device.global_mem_size) // 2
        if self.device.host_unified_memory:
            limit = min(limit, host_budget - staging)
        else:
            memory = _free_vram(self.device)
            if memory is None:
                raise GpuError('OpenFocus C GPU free memory is unknown')
            snapshot = replace(snapshot, free_vram=memory[0], total_vram=memory[1])
            limit = min(limit, resource_policy.vram_budget(snapshot))
        single = int(self.device.max_mem_alloc_size)
        if need > limit or n * 4 > single:
            raise GpuError('OpenFocus C GPU workspace exceeds the device budget')

        accumulators = [self._f32(size, channels) for size in detail_sizes]
        maxima = [self._u16(size) for size in detail_sizes]
        coarsest = self._f32(coarsest_size, channels)
        rolling = tuple(self._f32(size, channels) for size in rolling_sizes)
        horizontal = self._f32(horizontal_size, channels) if detail_levels else None
        weight = self._u16(full_size) if detail_levels else None
        winner = self._u16(full_size) if keep_winner else None
        filter_scratch = tuple(self._u16(full_size) for _ in range(filter_planes))
        filter_configs = []
        if filter_scratch:
            from fusion_methods.openfocus_depth import _weight
            for (h, w), _ in detail_levels:
                radius = float(np.float32(np.float32(max(h, w)) * parameter / np.float32(100)))
                q = int(_weight(radius))
                filter_configs.append((max(1, min(int(radius), w)), max(1, min(int(radius), h)),
                                       q, (4096 - q) >> 1))
        src = self.cl.Buffer(self.ctx, self.mf.READ_ONLY, n * in_bytes)
        in_flight = False
        try:
            for index, frame in enumerate(frames):
                _check_cancel(should_cancel)
                with without_source_cache_growth():
                    pixels, scale = _source_codes(frame, img_resize, should_cancel)
                if tuple(pixels.shape) != tuple(shape):
                    raise ValueError('OpenFocus C requires frames with matching shapes')
                if not detail_levels and index > 0:
                    del pixels
                    if progress_callback is not None:
                        progress_callback(index + 1, total)
                    continue
                pix_u8 = pixels.dtype == np.uint8
                codes = np.ascontiguousarray(pixels, dtype=np.uint8 if pix_u8 else np.uint16)
                del pixels
                factor = np.float32(scale / (65280.0 if is_u8 else 65535.0))
                fine = rolling[0] if detail_levels else coarsest
                in_flight = True
                self.cl.enqueue_copy(self.q, src, codes, is_blocking=False)
                self._run(self.k['c_norm_u8' if pix_u8 else 'c_norm'], n, src, fine,
                          factor, np.int32(n))

                for level, ((h, w), (ch, cw)) in enumerate(detail_levels):
                    _check_cancel(should_cancel)
                    # Coarsest weights depended only on geometry, so every
                    # frame tied. Write the first frame directly to its own plane.
                    coarse = (coarsest if index == 0 and level == len(detail_levels) - 1
                              else rolling[(level + 1) % 2])
                    self._run(self.k['c_down_h'], (cw, h), fine, horizontal,
                              np.int32(w), np.int32(h), np.int32(cw), np.int32(channels))
                    self._run(self.k['c_down_v'], (cw, ch), horizontal, coarse,
                              np.int32(cw), np.int32(h), np.int32(cw), np.int32(ch),
                              np.int32(channels))
                    self._synthesize(self.k['c_detail_w'], coarse, horizontal, fine,
                                     ch, cw, h, w, channels, weight)
                    if filter_scratch:
                        self._filter_w(weight, filter_scratch, h, w, filter_configs[level])
                    buffers = fine, weight, accumulators[level], maxima[level]
                    dimensions = np.int32(w), np.int32(h), np.int32(channels)
                    if level == 0 and winner is not None:
                        self._run(self.k['c_merge_focus'], (w, h), *buffers, winner,
                                  np.int32(index), *dimensions)
                    else:
                        self._run(self.k['c_merge'], (w, h), *buffers,
                                  np.int32(index == 0), *dimensions)
                    fine = coarse
                # Bound queued frame uploads to one source, even for very deep stacks.
                self.q.finish()
                in_flight = False
                del codes
                if progress_callback is not None:
                    progress_callback(index + 1, total)

            result = coarsest
            in_flight = True
            for ((h, w), (ch, cw)), detail in zip(reversed(detail_levels), reversed(accumulators)):
                _check_cancel(should_cancel)
                self._synthesize(self.k['c_collapse'], result, horizontal, detail,
                                 ch, cw, h, w, channels)
                result = detail

            out = np.empty((height, width, channels), out_dtype)
            dst = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, out.nbytes)
            kern = self.k['c_out_u8'] if is_u8 else self.k['c_out_f32']
            self._run(kern, n, result, dst, np.int32(n))
            self.cl.enqueue_copy(self.q, out, dst, is_blocking=True)
            if focus_callback is not None:
                from core.frame_storage import create_frame
                positions = create_frame((height, width), np.uint16)
                if winner is None:
                    positions.fill(0)
                else:
                    self.cl.enqueue_copy(self.q, positions, winner, is_blocking=True)
                in_flight = False
                _check_cancel(should_cancel)
                focus_callback(positions, count - 1)
            else:
                in_flight = False
            if progress_callback is not None:
                progress_callback(total, total)
            _check_cancel(should_cancel)
            return out[:, :, 0] if is_gray else out
        except BaseException:
            if in_flight:
                self._drain_queue()
            raise


