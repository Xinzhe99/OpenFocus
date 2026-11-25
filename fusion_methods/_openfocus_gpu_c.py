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

_READBACK_BUFFER_BYTES = 8 * 1024 * 1024
_WORKSPACE_RESERVE_BYTES = 16 * 1024 * 1024

_KERNEL_NAMES = (
    'c_norm', 'c_down_h_packed', 'c_down_h', 'c_down_v', 'c_up_h',
    'c_weight_packed', 'c_merge_packed', 'c_detail_w', 'c_collapse', 'c_merge',
    'c_out_u8', 'c_out_f32', 'transpose_u16', 'filt4',
    'sample_focus_plane',
)

_cache_lock = Lock()
_PROGRAM_CACHE = None


def _get_or_build_program(cl, device):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import (
        _PREVIEW_KERNEL, _build_filter_program, _device_cache_key, _get_shared_context,
    )

    key = _device_cache_key(device)
    with _cache_lock:
        ctx = _get_shared_context(cl, device)
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key and _PROGRAM_CACHE[1] is ctx:
            return _PROGRAM_CACHE[1:]
        queue = cl.CommandQueue(ctx, device=device)
        program, tiled_transpose = _build_filter_program(
            cl, device, ctx, _C_K + _PREVIEW_KERNEL.format(type='ushort'),
            options=['-cl-fp32-correctly-rounded-divide-sqrt'],
        )
        kernels = {name: cl.Kernel(program, name) for name in _KERNEL_NAMES}
        _PROGRAM_CACHE = (key, ctx, queue, kernels, tiled_transpose)
        return ctx, queue, kernels, tiled_transpose


def _invalidate_program_cache(context=None):
    global _PROGRAM_CACHE
    from fusion_methods._openfocus_gpu import _invalidate_shared_context

    with _cache_lock:
        _PROGRAM_CACHE = None
        _invalidate_shared_context(context)

_C_K = r'''
#pragma OPENCL FP_CONTRACT OFF
__constant float C_DOWN[4]={0.125f,0.375f,0.375f,0.125f};
__constant float C_EVEN[5]={0.019027505f,0.318410337f,0.559359729f,0.103202470f,0.0f};
__constant float C_ODD[5]={0.0f,0.103202470f,0.559359729f,0.318410337f,0.019027505f};

static inline float c_source_value(__global const uchar* src, int k, int c, int C,
                                   int u8, int reverse, float factor)
{
    int i=k*C+(reverse?2-c:c);
    float code=u8?(float)src[i]:(float)((__global const ushort*)src)[i];
    return code*factor;
}

__kernel void c_norm(__global const uchar* src, __global float* dst,
                     const float factor, const int u8, const int reverse, const int C)
{
    int k=get_global_id(0);
    for(int c=0;c<C;c++) dst[k*C+c]=c_source_value(src,k,c,C,u8,reverse,factor);
}

__kernel void c_down_h_packed(__global const uchar* src, __global float* dst,
                             const float factor, const int u8, const int reverse,
                             const int W, const int H, const int OW, const int C)
{
    int x=get_global_id(0); if(x>=OW)return;
    int y=get_global_id(1); if(y>=H)return;
    for(int c=0;c<C;c++){
        float v=0.0f;
        for(int k=0;k<4;k++){
            int sx=min(W-1,max(0,2*x+k-1));
            float s=c_source_value(src,y*W+sx,c,C,u8,reverse,factor);
            float p=s*C_DOWN[k]; v=v+p;
        }
        dst[(y*OW+x)*C+c]=v;
    }
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

static inline ushort c_weight(float st, int x, int y, int W, int H, int C)
{
    if(C==1) st=st*3.0f;
    else if(C==2) st=st*1.5f;
    int border=(H>2&&(y==0||y==H-1))||(W>2&&(x==0||x==W-1));
    return border?0:(ushort)min(65535,(int)(st*21845.0f));
}

static inline float c_packed_detail(__global const uchar* src,
                                    __global const float* horizontal,
                                    const __constant float* w, int x, int k,
                                    int c, int W, int C, int sy, int SH, float total,
                                    float factor, int u8, int reverse)
{
    float lo=c_up_value(horizontal,w,x*C+c,W*C,sy,SH,total);
    float fine=c_source_value(src,k,c,C,u8,reverse,factor);
    return (fine-lo)/(lo+0.1f);
}

__kernel void c_weight_packed(__global const uchar* src, __global const float* horizontal,
                              __global ushort* weight, const float factor, const int u8,
                              const int reverse, const int W, const int H, const int SH,
                              const int C)
{
    int x=get_global_id(0); if(x>=W)return;
    int y=get_global_id(1); if(y>=H)return;
    int k=y*W+x;
    const __constant float* w=(y&1)?C_ODD:C_EVEN;
    int sy=(y>>1)-2;
    float total=c_up_total(w,sy,SH);
    float st=0.0f;
    for(int c=0;c<min(C,3);c++){
        float v=c_packed_detail(src,horizontal,w,x,k,c,W,C,sy,SH,total,factor,u8,reverse);
        st=st+fabs(v);
    }
    weight[k]=c_weight(st,x,y,W,H,C);
}

__kernel void c_merge_packed(__global const uchar* src, __global const float* horizontal,
                             __global const ushort* weight, __global float* acc,
                             __global ushort* best, __global ushort* winner,
                             const int index, const float factor, const int u8,
                             const int reverse, const int W, const int H, const int SH,
                             const int C)
{
    int x=get_global_id(0); if(x>=W)return;
    int y=get_global_id(1); if(y>=H)return;
    int k=y*W+x;
    const __constant float* w=(y&1)?C_ODD:C_EVEN;
    int sy=(y>>1)-2;
    float total=c_up_total(w,sy,SH);
    int colors=0;
    float detail[3];
    ushort score;
    if(weight){
        score=weight[k];
    }else{
        // Without filtering, compare the weight immediately and reuse the detail.
        colors=min(C,3);
        float st=0.0f;
        for(int c=0;c<colors;c++){
            detail[c]=c_packed_detail(src,horizontal,w,x,k,c,W,C,sy,SH,total,factor,u8,reverse);
            st=st+fabs(detail[c]);
        }
        score=c_weight(st,x,y,W,H,C);
    }
    if(index!=0&&score<=best[k]) return;
    best[k]=score;
    if(winner) winner[k]=(ushort)index;
    for(int c=0;c<colors;c++) acc[k*C+c]=detail[c];
    for(int c=colors;c<C;c++){
        acc[k*C+c]=c_packed_detail(src,horizontal,w,x,k,c,W,C,sy,SH,total,factor,u8,reverse);
    }
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
    weight[y*W+x]=c_weight(st,x,y,W,H,C);
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

__kernel void c_out_u8(__global const float* src, __global uchar* dst,
                       const int start, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    float v=fmin(1.0f,fmax(0.0f,src[start+i]));
    int code=(int)(v*65535.0f);
    dst[i]=(uchar)(code>>8);
}

__kernel void c_out_f32(__global const float* src, __global float* dst,
                        const int start, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    float v=fmin(1.0f,fmax(0.0f,src[start+i]));
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
            self.ctx, self.q, self.k, self._tiled_transpose = _get_or_build_program(cl, device)
        except GpuError:
            raise
        except Exception as exc:
            raise GpuError('OpenCL GPU C initialization failed') from exc

    def _f32(self, pixels, channels):
        return self.cl.Buffer(self.ctx, self.mf.READ_WRITE, pixels * channels * 4)

    def _u16(self, pixels):
        return self.cl.Buffer(self.ctx, self.mf.READ_WRITE, pixels * 2)

    def _run(self, kernel, gsize, *args):
        return kernel(self.q, gsize if isinstance(gsize, tuple) else (gsize,), None, *args)

    def _filt4(self, src, dst, first, h, w, count, q, t):
        step = (w + 3) // 4
        self.k['filt4'](self.q, (step,), None, src, dst, first,
                        np.int32(h), np.int32(w), np.int32(count),
                        np.uint32(q), np.uint32(t), np.int32(step))

    def _transpose(self, src, dst, w, h):
        if self._tiled_transpose:
            gw, gh = ((w + 15) // 16) * 16, ((h + 15) // 16) * 16
            self.k['transpose_u16'](self.q, (gw, gh), (16, 16), src, dst,
                                    np.int32(w), np.int32(h))
        else:
            self._run(self.k['transpose_u16'], (w, h), src, dst,
                      np.int32(w), np.int32(h))

    def _filter_w(self, weight, scratch, h, w, config):
        """Device-side openfocus_depth._filter: axis 1 (horizontal) then 0."""
        a, first = scratch
        count_h, count_v, q, t = config
        # filt4 reads src[y,x] before writing that cell, so src==dst is safe.
        self._transpose(weight, a, w, h)
        self._filt4(a, a, first, w, h, count_h, q, t)
        self._transpose(a, weight, h, w)
        self._filt4(weight, weight, first, h, w, count_v, q, t)

    def _up_horizontal(self, coarse, horizontal, sh, sw, dw, c):
        self._run(self.k['c_up_h'], (dw, sh), coarse, horizontal,
                  np.int32(sw), np.int32(sh), np.int32(dw), np.int32(c))

    def _synthesize(self, kernel, coarse, horizontal, target, sh, sw, dh, dw, c,
                    *extra):
        self._up_horizontal(coarse, horizontal, sh, sw, dw, c)
        self._run(kernel, (dw, dh), target, horizontal, *extra,
                  np.int32(dw), np.int32(dh), np.int32(sh), np.int32(c))

    def _drain_queue(self):
        q = getattr(self, 'q', None)
        if q is not None:
            try:
                q.finish()
            except Exception:
                _invalidate_program_cache(getattr(self, 'ctx', None))

    def fuse(self, frames, shape, img_resize, smoothing,
             should_cancel=None, progress_callback=None, focus_callback=None,
             *, live_preview=None):
        from fusion_methods._openfocus_gpu import GpuError, _device_lock

        class CallbackFailure(BaseException):
            pass

        def progress(index, total):
            try:
                progress_callback(index, total)
            except BaseException as exc:
                raise CallbackFailure() from exc

        def focus(positions, scale):
            try:
                focus_callback(positions, scale)
            except BaseException as exc:
                raise CallbackFailure() from exc

        while not _device_lock.acquire(timeout=0.1):
            _check_cancel(should_cancel)
        try:
            try:
                args = (frames, shape, img_resize, smoothing, should_cancel,
                        progress if progress_callback is not None else None)
                callbacks = {}
                if focus_callback is not None:
                    callbacks['focus_callback'] = focus
                if live_preview is not None:
                    callbacks['live_preview'] = live_preview
                return self._fuse_locked(*args, **callbacks)
            except CallbackFailure as exc:
                self._drain_queue()
                raise exc.__cause__
            except (self.cl.Error, MemoryError) as exc:
                _invalidate_program_cache(getattr(self, 'ctx', None))
                self._drain_queue()
                raise GpuError('OpenCL GPU C execution failed') from exc
            except BaseException:
                self._drain_queue()
                raise
        finally:
            self.k = self.q = self.ctx = None
            _device_lock.release()

    def _fuse_locked(self, frames, shape, img_resize, smoothing,
                     should_cancel, progress_callback, focus_callback=None, *, live_preview=None):
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
        from fusion_methods._openfocus_gpu import (
            GpuError, _PreviewPlaneReader, _free_vram, _iter_prefetched, _prepare_gpu_upload,
            _release_buffers,
        )
        from utils.logging_setup import log_debug

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
                packed = isinstance(frame, np.ndarray) and (
                    frame.flags.c_contiguous
                    or (frame.ndim == 3 and frame.shape[2] == 3
                        and frame[..., ::-1].flags.c_contiguous)
                )
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
        readback_bytes = min(max(1, n * out_bytes), _READBACK_BUFFER_BYTES)
        readback = max(n * out_bytes + readback_bytes,
                       n * out_bytes + 4 * full_size if focus_callback is not None else 0)
        staging = max(source_peak, readback) + _WORKSPACE_RESERVE_BYTES
        if staging > host_budget:
            raise GpuError(f'OpenFocus C GPU staging exceeds the host memory budget '
                           f'(need {staging:,} bytes, limit {host_budget:,} bytes)')
        # The finest level reads packed source codes directly. Only the two
        # largest reduced levels need rolling float32 scratch.
        detail_sizes = [h * w for (h, w), _ in detail_levels]
        coarsest_size = levels[-1][0] * levels[-1][1]
        rolling_sizes = tuple(h * w for h, w in levels[1:3])
        horizontal_size = (max(height * levels[1][1], levels[1][0] * width)
                           if detail_levels else 0)
        filter_planes = 2 if detail_levels and parameter > 0 else 0
        keep_winner = focus_callback is not None and bool(detail_levels)
        weight_size = full_size if filter_planes else 0
        if not filter_planes and len(detail_sizes) > 1:
            weight_size = detail_sizes[1]
        upload_bytes = n * in_bytes
        # Upload and chunked readback do not coexist. Reserve separately for
        # the context, driver overhead and the bounded live preview sample.
        buffers = (
            *(4 * channels * size for size in (*detail_sizes, *rolling_sizes,
                                              horizontal_size, coarsest_size)),
            *(2 * size for size in detail_sizes),
            2 * weight_size,
            *((2 * full_size,) * (filter_planes + keep_winner)),
            max(upload_bytes, readback_bytes),
        )
        need = sum(buffers) + _WORKSPACE_RESERVE_BYTES
        largest = max(buffers)
        single = int(self.device.max_mem_alloc_size)
        if largest > single:
            raise GpuError(f'OpenFocus C GPU workspace needs a {largest:,}-byte buffer, '
                           f'but the OpenCL single allocation limit is {single:,} bytes')
        device_cap = int(self.device.global_mem_size) // 2
        unified_memory = self.device.host_unified_memory
        free_vram = total_vram = None
        if unified_memory:
            policy_limit = host_budget - staging
        else:
            memory = _free_vram(self.device)
            if memory is None:
                raise GpuError(f'OpenFocus C GPU free memory is unknown '
                               f'(need {need:,} bytes of device workspace)')
            free_vram, total_vram = memory
            snapshot = replace(snapshot, free_vram=free_vram, total_vram=total_vram)
            policy_limit = resource_policy.vram_budget(snapshot)
        limit = min(device_cap, policy_limit)
        if need > limit:
            if unified_memory:
                policy_reason = (f'shared host memory after staging {policy_limit:,} bytes, '
                                 f'host budget {host_budget:,} bytes, staging {staging:,} bytes')
            else:
                policy_reason = (f'verified free VRAM {free_vram:,} of {total_vram:,} bytes, '
                                 f'VRAM policy budget {policy_limit:,} bytes')
            raise GpuError(f'OpenFocus C GPU workspace exceeds the device budget '
                           f'(need {need:,} bytes, limit {limit:,} bytes, {policy_reason}, '
                           f'OpenCL half-total cap {device_cap:,} bytes)')
        log_debug('openfocus_c.gpu_admitted', shape=shape, workspace_bytes=need,
                  device_budget_bytes=limit, largest_buffer_bytes=largest,
                  max_allocation_bytes=single, free_vram_bytes=free_vram,
                  total_vram_bytes=total_vram, staging_bytes=staging,
                  host_budget_bytes=host_budget)

        accumulators = maxima = coarsest = rolling = horizontal = weight = winner = filter_scratch = dst = None
        srcs = ()
        reader = prepared_iter = None
        in_flight = False
        try:
            from fusion_methods._openfocus_gpu import _prefetch_window, _wait_event

            accumulators = [self._f32(size, channels) for size in detail_sizes]
            maxima = [self._u16(size) for size in detail_sizes]
            coarsest = self._f32(coarsest_size, channels)
            rolling = tuple(self._f32(size, channels) for size in rolling_sizes)
            horizontal = self._f32(horizontal_size, channels) if detail_levels else None
            weight = self._u16(weight_size) if weight_size else None
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
            prefetch_budget = host_budget - need if unified_memory else host_budget
            prefetch_slots = (
                _prefetch_window(
                    prefetch_budget, staging, source_peak, total,
                )
                if detail_levels else 0
            )
            ping_pong = (
                bool(detail_levels) and total >= 2
                and (need + upload_bytes) <= limit
                and (staging + prefetch_slots * source_peak + 2 * upload_bytes
                     + (upload_bytes if unified_memory else 0)) <= prefetch_budget
            )
            srcs = tuple(
                self.cl.Buffer(self.ctx, self.mf.READ_ONLY, upload_bytes)
                for _ in range(2 if ping_pong else 1)
            )
            if live_preview is not None and detail_levels:
                reader = _PreviewPlaneReader(self.cl, self.ctx, self.q, self.k['sample_focus_plane'],
                                             (height, width), np.uint16)

            def prepare_frame(index, frame, cancelled):
                _check_cancel(cancelled)
                with without_source_cache_growth():
                    pixels, scale = _source_codes(frame, img_resize, cancelled)
                if tuple(pixels.shape) != tuple(shape):
                    raise ValueError('OpenFocus C requires frames with matching shapes')
                if not detail_levels and index > 0:
                    del pixels
                    return None
                pix_u8 = pixels.dtype == np.uint8
                codes, bgr_rev = _prepare_gpu_upload(pixels, np.uint8 if pix_u8 else np.uint16)
                del pixels
                return codes, scale, pix_u8, bgr_rev

            prepared_iter = _iter_prefetched(
                frames, prepare_frame, enabled=prefetch_slots > 0,
                should_cancel=should_cancel, max_in_flight=max(1, prefetch_slots),
            )
            slot_events = [None, None]
            slot_hosts = [None, None]
            for index, item in prepared_iter:
                _check_cancel(should_cancel)
                if item is None:
                    if progress_callback is not None:
                        progress_callback(index + 1, total)
                    continue
                slot = index & 1 if ping_pong else 0
                if slot_events[slot] is not None:
                    _wait_event(slot_events[slot], lambda: _check_cancel(should_cancel))
                    slot_events[slot] = None
                    slot_hosts[slot] = None
                src = srcs[slot]
                codes, scale, pix_u8, bgr_rev = item
                packing = (np.float32(scale / (65280.0 if is_u8 else 65535.0)),
                           np.int32(pix_u8), np.int32(bgr_rev))
                in_flight = True
                upload = self.cl.enqueue_copy(self.q, src, codes, is_blocking=False)
                last_event = None
                if not detail_levels:
                    last_event = self._run(self.k['c_norm'], full_size, src, coarsest,
                                           *packing, np.int32(channels))

                for level, ((h, w), (ch, cw)) in enumerate(detail_levels):
                    _check_cancel(should_cancel)
                    # Coarsest weights depended only on geometry, so every
                    # frame tied. Write the first frame directly to its own plane.
                    coarse = (coarsest if index == 0 and level == len(detail_levels) - 1
                              else rolling[level % 2])
                    w_arg, h_arg, cw_arg, ch_arg, c_arg = map(np.int32, (w, h, cw, ch, channels))
                    if level == 0:
                        self._run(self.k['c_down_h_packed'], (cw, h), src, horizontal,
                                  *packing, w_arg, h_arg, cw_arg, c_arg)
                    else:
                        fine = rolling[(level - 1) % 2]
                        self._run(self.k['c_down_h'], (cw, h), fine, horizontal,
                                  w_arg, h_arg, cw_arg, c_arg)
                    self._run(self.k['c_down_v'], (cw, ch), horizontal, coarse,
                              cw_arg, h_arg, cw_arg, ch_arg, c_arg)
                    if level == 0:
                        # The filtered scores depend on whole rows and columns.
                        # After filtering, recompute detail only where a source
                        # wins, avoiding a full-size current-frame float plane.
                        self._up_horizontal(coarse, horizontal, ch, cw, w, channels)
                        up_dims = w_arg, h_arg, ch_arg, c_arg
                        filtered_weight = weight if filter_scratch else None
                        if filter_scratch:
                            self._run(self.k['c_weight_packed'], (w, h), src, horizontal,
                                      weight, *packing, *up_dims)
                            self._filter_w(weight, filter_scratch, h, w, filter_configs[level])
                        last_event = self._run(
                            self.k['c_merge_packed'], (w, h), src, horizontal,
                            filtered_weight, accumulators[0], maxima[0], winner,
                            np.int32(index), *packing, *up_dims,
                        )
                    else:
                        self._synthesize(self.k['c_detail_w'], coarse, horizontal, fine,
                                         ch, cw, h, w, channels, weight)
                        if filter_scratch:
                            self._filter_w(weight, filter_scratch, h, w, filter_configs[level])
                        last_event = self._run(
                            self.k['c_merge'], (w, h), fine, weight, accumulators[level],
                            maxima[level], np.int32(index == 0), w_arg, h_arg, c_arg,
                        )
                if ping_pong and last_event is not None and hasattr(last_event, 'wait'):
                    slot_events[slot] = last_event
                    slot_hosts[slot] = (codes, item, upload)
                    if reader is not None and live_preview.tracks():
                        _wait_event(last_event, lambda: _check_cancel(should_cancel))
                        slot_events[slot] = None
                        slot_hosts[slot] = None
                        sample = reader.read(maxima[0], live_preview)
                        if sample is not None:
                            live_preview.update_best(sample, index, count, sampled=True)
                else:
                    self.q.finish()
                    in_flight = False
                    if reader is not None and live_preview.tracks():
                        _check_cancel(should_cancel)
                        sample = reader.read(maxima[0], live_preview)
                        if sample is not None:
                            live_preview.update_best(sample, index, count, sampled=True)
                    elif reader is None and live_preview is not None and not detail_levels:
                        live_preview.show_positions(np.zeros((height, width), np.uint8), index, count, 1)
                    del codes, item, upload
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

            _release_buffers(srcs)
            srcs = None
            result = coarsest
            in_flight = True
            for ((h, w), (ch, cw)), detail in zip(reversed(detail_levels), reversed(accumulators)):
                _check_cancel(should_cancel)
                self._synthesize(self.k['c_collapse'], result, horizontal, detail,
                                 ch, cw, h, w, channels)
                result = detail

            out = np.empty((height, width, channels), out_dtype)
            out_samples = out.reshape(-1)
            dst = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, readback_bytes)
            chunk = max(1, readback_bytes // out_bytes)
            kern = self.k['c_out_u8'] if is_u8 else self.k['c_out_f32']
            for start in range(0, n, chunk):
                _check_cancel(should_cancel)
                region = out_samples[start:min(start + chunk, n)]
                self._run(kern, region.size, result, dst, np.int32(start), np.int32(region.size))
                self.cl.enqueue_copy(self.q, region, dst, is_blocking=True)
            if focus_callback is not None:
                from core.frame_storage import create_transient_frame
                positions = create_transient_frame((height, width), np.uint16)
                if winner is None:
                    positions.fill(0)
                else:
                    self.cl.enqueue_copy(self.q, positions, winner, is_blocking=True)
            in_flight = False
            if progress_callback is not None:
                progress_callback(total, total)
            _check_cancel(should_cancel)
            if focus_callback is not None:
                focus_callback(positions, count - 1)
            return out[:, :, 0] if is_gray else out
        except BaseException:
            if in_flight:
                self._drain_queue()
            raise
        finally:
            if prepared_iter is not None:
                prepared_iter.close()
            if reader is not None:
                reader.close()
            _release_buffers(
                accumulators, maxima, coarsest, rolling, horizontal,
                weight, winner, filter_scratch, srcs, dst,
            )


