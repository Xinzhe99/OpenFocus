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

from fusion_methods.openfocus_bc import _source_codes
from fusion_methods.openfocus_ab import _check_cancel

_KERNEL_NAMES = (
    'c_norm', 'c_norm_u8', 'c_down_h', 'c_down_v', 'c_up_h', 'c_up_v',
    'c_detail_w', 'c_last_w', 'c_merge', 'c_collapse',
    'c_merge_focus',
    'c_out_u8', 'c_out_f32', 'c_transpose', 'filt4',
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

__kernel void c_up_h(__global const float* s, __global float* d,
                     const int W, const int H, const int C, const int odd)
{
    int x=get_global_id(0); if(x>=W)return;
    int y=get_global_id(1); if(y>=H)return;
    const __constant float* w=odd?C_ODD:C_EVEN;
    float total=0.0f;
    for(int k=0;k<5;k++){int sx=x+k-2; if(sx>=0&&sx<W) total=total+w[k];}
    for(int c=0;c<C;c++){
        float v=0.0f;
        for(int k=0;k<5;k++){int sx=x+k-2; if(sx>=0&&sx<W){float p=s[(y*W+sx)*C+c]*w[k]; v=v+p;}}
        d[(y*W+x)*C+c]=v/total;
    }
}

__kernel void c_up_v(__global const float* s, __global float* d,
                     const int SW, const int SH, const int DW, const int DH,
                     const int parity, const int C)
{
    int x=get_global_id(0); if(x>=SW)return;
    int y=get_global_id(1); if(y>=DH)return;
    int sy0=y>>1;
    const __constant float* w=(y&1)?C_ODD:C_EVEN;
    float total=0.0f;
    for(int k=0;k<5;k++){int sy=sy0+k-2; if(sy>=0&&sy<SH) total=total+w[k];}
    int ox=2*x+parity; if(ox>=DW)return;
    for(int c=0;c<C;c++){
        float v=0.0f;
        for(int k=0;k<5;k++){int sy=sy0+k-2; if(sy>=0&&sy<SH){float p=s[(sy*SW+x)*C+c]*w[k]; v=v+p;}}
        d[(y*DW+ox)*C+c]=v/total;
    }
}

__kernel void c_detail_w(__global const float* fine, __global const float* up,
                         __global float* detail, __global ushort* weight,
                         const int W, const int H, const int C)
{
    int x=get_global_id(0); if(x>=W)return;
    int y=get_global_id(1); if(y>=H)return;
    float st=0.0f;
    for(int c=0;c<C;c++){
        int i=(y*W+x)*C+c;
        float lo=up[i];
        float v=(fine[i]-lo)/(lo+0.1f);
        detail[i]=v;
        if(c<3) st=st+fabs(v);
    }
    if(C==1) st=st*3.0f;
    else if(C==2) st=st*1.5f;
    int border=(H>2&&(y==0||y==H-1))||(W>2&&(x==0||x==W-1));
    int wi=border?0:min(65535,(int)(st*21845.0f));
    weight[y*W+x]=(ushort)wi;
}

__kernel void c_last_w(__global ushort* weight, const int W, const int H)
{
    int x=get_global_id(0); if(x>=W)return;
    int y=get_global_id(1); if(y>=H)return;
    weight[y*W+x]=(y==0||x==0||y==H-1||x==W-1)?(ushort)0:(ushort)65535;
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

__kernel void c_collapse(__global const float* coarse, __global const float* detail,
                         __global float* out, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    float v=coarse[i];
    float t=(v+0.1f)*detail[i]; float value=t+v;
    out[i]=fmin(1.0f,fmax(0.0f,value));
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

__kernel void c_transpose(__global const ushort* src, __global ushort* dst,
                          const int H, const int W)
{
    int x=get_global_id(0); if(x>=W)return;
    int y=get_global_id(1); if(y>=H)return;
    dst[x*H+y]=src[y*W+x];
}
'''


class GpuC:
    """Device-resident C pyramid fusion; mirrors fuse_pyramid step for step."""

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
        self._planes = {}
        self._weights = {}
        self._filt = {}

    # -- buffer helpers ---------------------------------------------------------

    def _f32(self, h, w, c):
        return self.cl.Buffer(self.ctx, self.mf.READ_WRITE, h * w * c * 4)

    def _u16(self, h, w):
        return self.cl.Buffer(self.ctx, self.mf.READ_WRITE, h * w * 2)

    def _plane(self, key, h, w, c):
        k = (key, h, w, c)
        if k not in self._planes:
            self._planes[k] = self._f32(h, w, c)
        return self._planes[k]

    def _weight_plane(self, h, w):
        k = ('w', h, w)
        if k not in self._weights:
            self._weights[k] = self._u16(h, w)
        return self._weights[k]

    def _run(self, kernel, gsize, *args):
        kernel(self.q, gsize if isinstance(gsize, tuple) else (gsize,), None, *args)

    def _filt4(self, src, dst, first, h, w, count, q, t):
        step = (w + 3) // 4
        self.k['filt4'](self.q, (step,), None, src, dst, first,
                        np.int32(h), np.int32(w), np.int32(count),
                        np.uint32(q), np.uint32(t), np.int32(step))

    def _filter_w(self, weight, h, w, radius):
        """Device-side openfocus_depth._filter: axis 1 (horizontal) then 0."""
        from fusion_methods.openfocus_depth import _weight
        q = int(_weight(radius))
        t = (4096 - q) >> 1
        count_h = max(1, min(int(radius), w))   # along original x
        count_v = max(1, min(int(radius), h))   # along original y
        if (h, w) not in self._filt:
            # a: transposed (w,h) work plane; first: shared history (h*w elems).
            # filt4 reads src[y,x] before writing out[y,x] in reverse row order,
            # so src==dst aliasing is safe and saves a copy per axis.
            self._filt[(h, w)] = (self._u16(w, h), self._u16(h, w))
        a, first = self._filt[(h, w)]
        # horizontal (axis 1): transpose -> filt4 in place -> transpose back
        self._run(self.k['c_transpose'], (w, h), weight, a,
                  np.int32(h), np.int32(w))
        self._filt4(a, a, first, w, h, count_h, q, t)
        self._run(self.k['c_transpose'], (h, w), a, weight,
                  np.int32(w), np.int32(h))
        # vertical (axis 0): filt4 in place on (h,w)
        self._filt4(weight, weight, first, h, w, count_v, q, t)

    def _up(self, src, dst, sh, sw, dh, dw, c):
        """Mirror _up(): horizontal keeps coarse shape; two vertical parity passes."""
        tmp = self._plane(('uh', sh, sw, c), sh, sw, c)
        for parity, odd in ((0, 0), (1, 1)):
            self._run(self.k['c_up_h'], (sw, sh), src, tmp,
                      np.int32(sw), np.int32(sh), np.int32(c), np.int32(odd))
            self._run(self.k['c_up_v'], (sw, dh), tmp, dst,
                      np.int32(sw), np.int32(sh), np.int32(dw), np.int32(dh),
                      np.int32(parity), np.int32(c))
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
            for cache in ('_planes', '_weights', '_filt'):
                bufs = getattr(self, cache, None)
                if bufs is not None:
                    bufs.clear()
            self.k = self.q = self.ctx = None
            _device_lock.release()

    def _fuse_locked(self, frames, shape, img_resize, smoothing,
                     should_cancel, progress_callback, focus_callback=None):
        height, width = shape[0], shape[1]
        is_gray = len(shape) == 2
        channels = 1 if is_gray else shape[2]
        is_u8 = all(getattr(f, 'dtype', None) == np.uint8 for f in frames)
        levels = [(height, width)]
        while True:
            nxt = ((levels[-1][0] + 1) // 2, (levels[-1][1] + 1) // 2)
            if min(nxt) < 16 and (len(levels) > 1 or nxt == levels[-1]):
                break
            levels.append(nxt)
        count = len(frames)
        total = count + 1
        if progress_callback is not None:
            progress_callback(0, total)
        parameter = np.float32(max(0.0, (float(smoothing) - 1.0) / 5.0))
        from dataclasses import replace
        from core import resource_policy
        from core.multi_focus_fusion import memory_budget_bytes
        from core.source_frames import frame_resolution_bytes, source_cache_growth_bytes
        from fusion_methods._openfocus_gpu import GpuError, _free_vram

        snapshot = resource_policy.probe_resources()
        host_budget = min(int(memory_budget_bytes()),
                          int(resource_policy.frame_ram_budget(snapshot)))
        in_bytes = 1 if (img_resize is None and is_u8) else 2
        staging = (height * width * channels * (in_bytes + (1 if is_u8 else 4))
                   + max((frame_resolution_bytes(frame) for frame in frames), default=0)
                   + source_cache_growth_bytes(frames) + 16 * 1024 * 1024)
        if staging > host_budget:
            raise GpuError('OpenFocus C GPU staging exceeds the host memory budget')
        # Includes all cached pyramid planes, input/output buffers and context headroom.
        need = sum(h * w * (channels * 4 * 8 + 2 * 4) for h, w in levels)
        need += height * width * channels * 6 + 16 * 1024 * 1024
        if focus_callback is not None:
            need += height * width * 2
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
        level0 = height * width * channels * 4
        if need > limit or level0 > single:
            raise GpuError('OpenFocus C GPU workspace exceeds the device budget')


        accumulators, maxima = [], []
        for h, w in levels:
            acc = self._plane(('acc', h, w, channels), h, w, channels)
            best = self._u16(h, w)
            accumulators.append(acc)
            maxima.append(best)
        winner = self._u16(height, width) if focus_callback is not None else None

        src = None
        src_bytes = 0
        in_flight = False
        try:
            for index, frame in enumerate(frames):
                _check_cancel(should_cancel)
                pixels, scale = _source_codes(frame, img_resize, should_cancel)
                if tuple(pixels.shape) != tuple(shape):
                    raise ValueError('OpenFocus C requires frames with matching shapes')
                pix_u8 = pixels.dtype == np.uint8
                codes = np.ascontiguousarray(pixels, dtype=np.uint8 if pix_u8 else np.uint16)
                factor = np.float32(scale / (65280.0 if is_u8 else 65535.0))
                n = height * width * channels
                base = self._plane(('g0', height, width, channels), height, width, channels)
                if src is None or src_bytes != codes.nbytes:
                    src = None
                    src_bytes = 0
                    src = self.cl.Buffer(self.ctx, self.mf.READ_ONLY, max(1, codes.nbytes))
                    src_bytes = codes.nbytes
                in_flight = True
                self.cl.enqueue_copy(self.q, src, codes, is_blocking=False)
                self._run(self.k['c_norm_u8' if pix_u8 else 'c_norm'], (n,), src, base,
                          np.float32(factor), np.int32(n))

                gaussian = [base]
                for level in range(1, len(levels)):
                    ph, pw = levels[level - 1]
                    ch, cw = levels[level]
                    tmp = self._plane(('dt', level, ph, cw, channels), ph, cw, channels)
                    out = self._plane(('g', level, ch, cw, channels), ch, cw, channels)
                    self._run(self.k['c_down_h'], (cw, ph), gaussian[-1], tmp,
                              np.int32(pw), np.int32(ph), np.int32(cw), np.int32(channels))
                    self._run(self.k['c_down_v'], (cw, ch), tmp, out,
                              np.int32(cw), np.int32(ph), np.int32(cw), np.int32(ch),
                              np.int32(channels))
                    gaussian.append(out)

                for level, (h, w) in enumerate(levels):
                    _check_cancel(should_cancel)
                    c = channels
                    if level + 1 == len(levels):
                        detail = gaussian[level]
                        weight = self._weight_plane(h, w)
                        self._run(self.k['c_last_w'], (w, h), weight,
                                  np.int32(w), np.int32(h))
                    else:
                        th, tw = levels[level + 1]
                        up = self._plane(('up', level, h, w, c), h, w, c)
                        detail = self._plane(('det', level, h, w, c), h, w, c)
                        weight = self._weight_plane(h, w)
                        self._up(gaussian[level + 1], up, th, tw, h, w, c)
                        self._run(self.k['c_detail_w'], (w, h), gaussian[level], up,
                                  detail, weight, np.int32(w), np.int32(h), np.int32(c))
                    if parameter > 0:
                        radius = np.float32(np.float32(max(h, w)) * parameter / np.float32(100))
                        self._filter_w(weight, h, w, float(radius))
                    buffers = detail, weight, accumulators[level], maxima[level]
                    dimensions = np.int32(w), np.int32(h), np.int32(c)
                    if level == 0 and winner is not None:
                        self._run(self.k['c_merge_focus'], (w, h), *buffers, winner,
                                  np.int32(index), *dimensions)
                    else:
                        self._run(self.k['c_merge'], (w, h), *buffers,
                                  np.int32(index == 0), *dimensions)
                # Bound queued frame uploads to one source, even for very deep stacks.
                self.q.finish()
                in_flight = False
                del codes, pixels
                if progress_callback is not None:
                    progress_callback(index + 1, total)

            result = accumulators[-1]
            in_flight = True
            for level in range(len(levels) - 2, -1, -1):
                _check_cancel(should_cancel)
                h, w = levels[level]
                th, tw = levels[level + 1]
                coarse = self._plane(('cc', level, h, w, channels), h, w, channels)
                out = self._plane(('co', level, h, w, channels), h, w, channels)
                self._up(result, coarse, th, tw, h, w, channels)
                self._run(self.k['c_collapse'], (h * w * channels,),
                          coarse, accumulators[level], out, np.int32(h * w * channels))
                result = out

            oh, ow = height, width
            out_dtype = np.uint8 if is_u8 else np.float32
            out = np.empty((oh, ow, channels), out_dtype)
            dst = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, out.nbytes)
            n = oh * ow * channels
            kern = self.k['c_out_u8'] if is_u8 else self.k['c_out_f32']
            self._run(kern, (n,), result, dst, np.int32(n))
            self.cl.enqueue_copy(self.q, out, dst, is_blocking=True)
            if winner is not None:
                from core.frame_storage import create_frame
                positions = create_frame((height, width), np.uint16)
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


