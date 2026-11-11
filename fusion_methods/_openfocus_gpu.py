"""Optional OpenCL acceleration for OpenFocus B depth fusion and merge.

This module is imported lazily by ``openfocus_b_impl``. ``pyopencl`` is an
optional dependency: when it or an OpenCL device is unavailable, callers keep
the CPU path. The GPU path reproduces the CPU ``depth_map`` (half-resolution,
preserve-detail) and ordered MergeWeighted stages bit-exactly on supported
inputs; unsupported inputs fall back to CPU before any GPU work starts.

Supported GPU inputs (v1): ``img_resize=None``, 3-channel uint8 or uint16
sources, ``depth_override=None``. Everything else uses the CPU implementation.
"""
import numpy as np

from fusion_methods._openfocus_pixels import frame_gray_codes

# ---------------------------------------------------------------------------
# Kernel sources (ported from the verified research probe; semantics identical
# to the CPU njit rows in _openfocus_depth_rows / _openfocus_kernels).
# ---------------------------------------------------------------------------

_FILTER_K = r'''
__kernel void filt(__global const ushort* src, __global ushort* out,
                   __global ushort* first, const int H, const int W,
                   const int count, const uint q, const uint t)
{
    int x = get_global_id(0);
    if (x >= W) return;
    uint mask = 0xffffffffu;
    uint acc = 0;
    for (int y = 0; y < count; ++y) acc += (uint)src[(size_t)y*W + x];
    acc = (acc << 4) / (uint)count;
    acc *= (t + q);
    for (int y = 0; y < H; ++y) {
        uint value = (uint)src[(size_t)y*W + x];
        uint innovation = (value * 16u * t) & mask;
        uint state = ((acc + 2048u + innovation) & mask) >> 12;
        first[(size_t)y*W + x] = (ushort)(((state + 4u) >> 4) & 0xffffu);
        acc = (state * q + innovation) & mask;
    }
    acc = (uint)first[(size_t)(H-1)*W + x] << 4;
    acc *= (t + q);
    for (int y = H - 1; y >= 0; --y) {
        uint value = (uint)src[(size_t)y*W + x];
        uint innovation = (value * 16u * t) & mask;
        uint state = ((acc + 2048u + innovation) & mask) >> 12;
        out[(size_t)y*W + x] = (ushort)(((((state + 3u) >> 4) + 1u
            + (uint)first[(size_t)y*W + x]) >> 1) & 0xffffu);
        acc = (state * q + innovation) & mask;
    }
}

__kernel void filt_guided(__global const ushort* src, __global const ushort* guide,
                          __global ushort* out, __global uint* first,
                          const int H, const int W, const int count,
                          const uint tm, const uint tl)
{
    int x=get_global_id(0); if(x>=W)return;
    uint mask=0xffffffffu;
    uint acc=0;
    for(int y=0;y<count;++y) acc += (uint)src[(size_t)y*W+x];
    acc=(acc<<4)/(uint)count;
    for(int y=0;y<H;++y){
        uint value=(uint)src[(size_t)y*W+x];
        uint strength=(uint)guide[(size_t)y*W+x];
        uint amount=((((65536u-strength)*tl+strength*tm)&mask)>>16);
        uint previous=(y==0)?value:(uint)src[(size_t)(y-1)*W+x];
        uint state=((((4096u-2u*amount)*(uint)acc+2048u
            +(value+previous)*(amount<<4))&mask)>>12);
        acc=state;
        first[(size_t)y*W+x]=(state+4u)>>4;
    }
    acc=(uint)first[(size_t)(H-1)*W+x]<<4;
    for(int y=H-1;y>=0;--y){
        uint value=(uint)src[(size_t)y*W+x];
        uint strength=(uint)guide[(size_t)y*W+x];
        uint amount=((((65536u-strength)*tl+strength*tm)&mask)>>16);
        uint previous=(y==H-1)?(uint)first[(size_t)(H-1)*W+x]:(uint)src[(size_t)(y+1)*W+x];
        uint state=((((4096u-2u*amount)*(uint)acc+2048u
            +(value+previous)*(amount<<4))&mask)>>12);
        acc=state;
        out[(size_t)y*W+x]=(ushort)(((((state+3u)>>4)+1u
            +first[(size_t)y*W+x])>>1)&0xffffu);
    }
}

__kernel void filt4(__global const ushort* src, __global ushort* out,
                    __global ushort* first, const int H, const int W,
                    const int count, const uint q, const uint t, const int step4)
{
    int x0=get_global_id(0); if(x0>=step4) return;
    uint mask=0xffffffffu;
    uint acc[4]={0u,0u,0u,0u};
    int xs[4]={x0,x0+step4,x0+2*step4,x0+3*step4};
    for(int k=0;k<4;k++){int x=xs[k]; if(x>=W)continue;
        for(int y=0;y<count;++y) acc[k]+=(uint)src[(size_t)y*W+x];
        acc[k]=(acc[k]<<4)/(uint)count; acc[k]*=(t+q);
    }
    for(int y=0;y<H;++y){
        for(int k=0;k<4;k++){int x=xs[k]; if(x>=W)continue;
            uint value=(uint)src[(size_t)y*W+x];
            uint innovation=(value*16u*t)&mask;
            uint state=((acc[k]+2048u+innovation)&mask)>>12;
            first[(size_t)y*W+x]=(ushort)(((state+4u)>>4)&0xffffu);
            acc[k]=(state*q+innovation)&mask;
        }
    }
    for(int k=0;k<4;k++){int x=xs[k]; if(x>=W)continue;
        acc[k]=(uint)first[(size_t)(H-1)*W+x]<<4; acc[k]*=(t+q);
    }
    for(int y=H-1;y>=0;--y){
        for(int k=0;k<4;k++){int x=xs[k]; if(x>=W)continue;
            uint value=(uint)src[(size_t)y*W+x];
            uint innovation=(value*16u*t)&mask;
            uint state=((acc[k]+2048u+innovation)&mask)>>12;
            out[(size_t)y*W+x]=(ushort)(((((state+3u)>>4)+1u
                +(uint)first[(size_t)y*W+x])>>1)&0xffffu);
            acc[k]=(state*q+innovation)&mask;
        }
    }
}
'''

_DEPTH_K = r'''
__kernel void activity(__global const ushort* gray, __global const ushort* fine,
                       __global ushort* out, const int n)
{
    int i=get_global_id(0); if(i>=n) return;
    ulong g=gray[i], f=fine[i];
    ulong maximum=max(g,f);
    ulong denom=((maximum*100ul*0x80008001ul)>>32)>>15;
    ulong v=(110ul*abs((long)(g-f)))/(denom+10ul);
    out[i]=(ushort)min(65535ul,v);
}
__kernel void init_activity(__global const ushort* a, __global ushort* depth,
                            __global ushort* votes, const uint count, const int n)
{
    int i=get_global_id(0); if(i>=n) return;
    depth[i]=0; votes[i]=(ushort)((uint)a[i]/count);
}
__kernel void select_activity(__global const ushort* a, __global ushort* best,
                              __global ushort* depth, __global ushort* votes,
                              const ushort code, const uint count, const int n)
{
    int i=get_global_id(0); if(i>=n) return;
    uint v=a[i];
    votes[i]=(ushort)(((uint)votes[i]+v/count)&65535u);
    if(v>(uint)best[i]){best[i]=(ushort)v;depth[i]=code;}
}
__kernel void guide(__global const ushort* best, __global const float* lut,
                    __global ushort* out, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    float ratio=4095.0f/65535.0f;
    float position=(float)best[i]*ratio;
    int index=min(4094,max(0,(int)position));
    float fraction=position-(float)index;
    float left=lut[index]*(1.0f-fraction);
    float right=lut[index+1]*fraction;
    float value=left+right;
    out[i]=(ushort)min(65535.0f,max(0.0f,value));
}
__kernel void snap(__global ushort* depth, __global const long* lut,
                   const int step, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    int hh=step>>1;
    long block=(long)depth[i];
    long total=(long)hh+block;
    long snapped=total-total%(long)step;
    long distance=block-snapped;
    long signed_val=((abs(distance)<<10)/(long)hh)*((distance<0)?-1L:1L);
    long ai=abs(signed_val); if(ai>1024L) ai=1024L; int index=(int)ai;
    long correction=(long)hh*lut[index]*((signed_val>0)?1L:-1L);
    correction=(correction>=0)?(correction>>10):-((-correction)>>10);
    depth[i]=(ushort)((snapped+correction)&65535L);
}
__kernel void detail_pool(__global const ushort* gray, __global ushort* out,
                          const int H, const int W, const int OH, const int OW)
{
    int x=get_global_id(0), y=get_global_id(1);
    if(x>=OW||y>=OH)return;
    int y0=min(2*y,H-1), y1=min(2*y+1,H-1), x0=min(2*x,W-1), x1=min(2*x+1,W-1);
    long a=gray[(size_t)y0*W+x0], b=gray[(size_t)y0*W+x1];
    long c=gray[(size_t)y1*W+x0], d=gray[(size_t)y1*W+x1];
    long mean=(a+b+c+d)/4;
    long vals[4]={a,b,c,d}; long energy=0;
    for(int k=0;k<4;k++){
        long v=vals[k], maximum=max(v,mean);
        long denom=((maximum*100L*0x80008001L)>>32)>>15;
        long ev=110L*abs(v-mean)/(denom+10L); if(ev>65535L)ev=65535L; energy+=ev;
    }
    out[(size_t)y*OW+x]=(ushort)(energy/4);
}
inline int refl(int p,int n){ if(n==1)return 0; int period=2*(n-1); p%=period; if(p<0)p+=period; return p>=n?period-p:p; }
__kernel void detail_smooth(__global const ushort* src, __global ushort* out,
                            const int H, const int W)
{
    int x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H)return;
    long total=0;
    for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++)
        total += (long)src[(size_t)refl(y+dy,H)*W+refl(x+dx,W)];
    out[(size_t)y*W+x]=(ushort)(total/25);
}
__kernel void gray_u8(__global const uchar* src, __global ushort* out,
                      const int H, const int W, const int C, const int scale)
{
    int x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H)return;
    ulong total=0;
    for(int c=0;c<3&&c<C;c++) total+=(ulong)src[((size_t)y*W+x)*C+c];
    out[(size_t)y*W+x]=(ushort)((total*(ulong)scale*85ul)>>8);
}
__kernel void detail_select(__global const ushort* a, __global ushort* best,
                            __global ushort* second, __global ushort* winner,
                            const ushort code, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    uint v=a[i];
    if(v>(uint)best[i]){second[i]=best[i];best[i]=(ushort)v;winner[i]=code;}
    else if(v>(uint)second[i]) second[i]=(ushort)v;
}
'''

_RESAMPLE_K = r'''
__kernel void vert(__global const ushort* src, __global float* scratch,
                   __global const long* idx, __global const float* wt,
                   __global const long* cnt, const int OH, const int SW)
{
    int x=get_global_id(0), y=get_global_id(1);
    if(x>=SW||y>=OH)return;
    float acc=0.0f;
    int n=(int)cnt[y];
    for(int t=0;t<n;t++) acc=fma(wt[y*8+t],(float)src[idx[y*8+t]*SW+x],acc);
    scratch[(size_t)y*SW+x]=acc;
}
__kernel void horiz(__global const float* scratch, __global ushort* out,
                    __global const long* idx, __global const float* wt,
                    __global const long* cnt, const int OH, const int OW, const int SW)
{
    int x=get_global_id(0), y=get_global_id(1);
    if(x>=OW||y>=OH)return;
    float acc=0.0f;
    int n=(int)cnt[x];
    for(int t=0;t<n;t++) acc=fma(wt[x*8+t],scratch[(size_t)y*SW+idx[x*8+t]],acc);
    float rounded=acc+0.5f;
    out[(size_t)y*OW+x]=(ushort)(rounded<=0.0f?0:(rounded>=65535.0f?65535:(int)rounded));
}
'''

_MERGE_K = r'''
__kernel void merge(__global ushort* acc, __global const ushort* img,
                    __global const int* depth, const int H, const int W,
                    const int C, const int step, const int index,
                    const int scale, const int first, const int reverse)
{
    int x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H)return;
    int weight=1024;
    if(!first){
        int delta=(int)depth[(size_t)y*W+x]-step*index;
        if(reverse) delta=-delta;
        if(delta>=0){ if(delta>step) return; }
        else { if(delta<=-step) return; weight=1024-(((-delta)<<10)/step); if(weight<=0) return; }
    }
    for(int c=0;c<C;c++){
        size_t k=((size_t)y*W+x)*C+c;
        int value=(int)img[k]*scale;
        if(weight==1024) acc[k]=(ushort)value;
        else acc[k]=(ushort)((((int)acc[k]*(1024-weight))+value*weight)>>10);
    }
}

__kernel void merge_u8(__global ushort* acc, __global const uchar* img,
                       __global const int* depth, const int H, const int W,
                       const int C, const int step, const int index,
                       const int first, const int reverse)
{
    int x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H)return;
    int weight=1024;
    if(!first){
        int delta=depth[(size_t)y*W+x]-step*index;
        if(reverse) delta=-delta;
        if(delta>=0){ if(delta>step) return; }
        else { if(delta<=-step) return; weight=1024-(((-delta)<<10)/step); if(weight<=0) return; }
    }
    for(int c=0;c<C;c++){
        size_t k=((size_t)y*W+x)*C+c;
        int value=(int)img[k]*256;
        if(weight==1024) acc[k]=(ushort)value;
        else acc[k]=(ushort)((((int)acc[k]*(1024-weight))+value*weight)>>10);
    }
}
'''


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

_G = None


def _gpu():
    """Return (ctx, queue, mem_flags, programs) or None when unavailable."""
    global _G
    if _G is False:
        return None
    if _G is None:
        try:
            import pyopencl as cl
            ctx = cl.create_some_context(interactive=False)
            q = cl.CommandQueue(ctx)
            mf = cl.mem_flags
            pk = cl.Program(ctx, _FILTER_K + _DEPTH_K + _RESAMPLE_K + _MERGE_K).build()
            _G = (cl, ctx, q, mf, pk)
        except Exception:
            _G = False
            return None
    return _G


def gpu_available():
    return _gpu() is not None


def gpu_supported(frames, shape, img_resize, depth_override):
    """Inputs the GPU path can reproduce bit-exactly (v1 scope)."""
    if img_resize is not None or depth_override is not None:
        return False
    if len(shape) != 3 or shape[2] != 3:
        return False
    # Only plain ndarrays with one dtype: lazy sources and mixed
    # uint8/uint16 stacks keep the CPU path for v1.
    dtypes = {f.dtype for f in frames} if frames else set()
    return (len(dtypes) == 1 and dtypes.pop() in (np.uint8, np.uint16)
            and all(isinstance(f, np.ndarray) and f.ndim == 3 for f in frames))


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

class _Filter:
    """Unguided/guided IIR axis filter; mirrors gpu_filter.Filter + device ctx."""

    def __init__(self, cl, ctx, q, mf, prg):
        self.cl = cl
        self.ctx = ctx
        self.q = q
        self.mf = mf
        self.p = prg
        self._bufs = {}

    def _bufs_for(self, h, w, guided):
        key = (h, w, guided)
        if key not in self._bufs:
            n = h * w
            if guided:
                # filt_guided keeps uint32 forward state in `first`.
                self._bufs[key] = (self.cl.Buffer(self.ctx, self.mf.READ_WRITE, n * 4),)
            else:
                self._bufs[key] = (
                    self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, n * 2),
                    self.cl.Buffer(self.ctx, self.mf.READ_WRITE, n * 2))
        return self._bufs[key]

    def _upload(self, arr):
        return self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                              hostbuf=np.ascontiguousarray(arr, dtype=np.uint16))

    def _download(self, buf, shape):
        out = np.empty(shape, np.uint16)
        self.cl.enqueue_copy(self.q, out, buf, is_blocking=True)
        return out

    def _axis(self, src, count, q, t, axis=0):
        data = np.ascontiguousarray(src if axis == 0 else src.T, np.uint16)
        h, w = data.shape
        ib = self._upload(data)
        step = (w + 3) // 4
        ob, fb = self._bufs_for(h, w, False)
        self.cl.Kernel(self.p, 'filt4')(self.q, (step,), None, ib, ob, fb,
                                        np.int32(h), np.int32(w), np.int32(count),
                                        np.uint32(q), np.uint32(t), np.int32(step))
        out = self._download(ob, (h, w))
        return out if axis == 0 else out.T.copy()

    def run(self, src, radius):
        from fusion_methods.openfocus_depth import _weight
        q = int(_weight(radius))
        t = (4096 - q) >> 1
        h, w = src.shape
        r = self._axis(src, max(1, min(int(radius), w)), q, t, 1)
        return self._axis(r, max(1, min(int(radius), h)), q, t, 0)

    def axis_guided(self, src, guide, main, low, axis=0):
        from fusion_methods.openfocus_depth import _weight
        data = np.ascontiguousarray(src if axis == 0 else src.T, np.uint16)
        gd = np.ascontiguousarray(guide if axis == 0 else guide.T, np.uint16)
        h, w = data.shape
        count = max(1, min(int(main), h))
        tm = (4096 - (int(_weight(main)) or 0)) >> 1
        tl = (4096 - int(_weight(low))) >> 1
        ib = self._upload(data)
        gb = self._upload(gd)
        (fb,) = self._bufs_for(h, w, True)
        ob = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, data.nbytes)
        self.cl.Kernel(self.p, 'filt_guided')(self.q, (w,), None, ib, gb, ob, fb,
                                              np.int32(h), np.int32(w),
                                              np.int32(count), np.uint32(tm),
                                              np.uint32(tl))
        out = self._download(ob, (h, w))
        return out if axis == 0 else out.T.copy()


class _HalfResample:
    def __init__(self, cl, ctx, q, mf, prg):
        self.cl = cl
        self.ctx = ctx
        self.q = q
        self.mf = mf
        self.p = prg
        self._cache = {}

    def _axes(self, shape):
        if shape not in self._cache:
            from fusion_methods._openfocus_resample import half_gray_axes
            v, h = half_gray_axes(*shape)

            def mk(a, dt):
                return self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                      hostbuf=np.ascontiguousarray(a, dtype=dt))
            self._cache[shape] = (v, h, mk(v.indices, np.int64), mk(v.weights, np.float32),
                                  mk(v.counts, np.int64), mk(h.indices, np.int64),
                                  mk(h.weights, np.float32), mk(h.counts, np.int64))
        return self._cache[shape]

    def run(self, gray):
        v, h, vix, vw, vc, hix, hw, hc = self._axes(gray.shape)
        H, W = gray.shape
        OH, OW = v.indices.shape[0], h.indices.shape[0]
        ib = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(gray))
        vb = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, OH * W * 4)
        self.cl.Kernel(self.p, 'vert')(self.q, (W, OH), None, ib, vb, vix, vw, vc,
                                       np.int32(OH), np.int32(W))
        ob = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, OH * OW * 2)
        self.cl.Kernel(self.p, 'horiz')(self.q, (OW, OH), None, vb, ob, hix, hw, hc,
                                        np.int32(OH), np.int32(OW), np.int32(W))
        self.q.finish()
        out = np.empty((OH, OW), np.uint16)
        self.cl.enqueue_copy(self.q, out, ob, is_blocking=True)
        return out


class _ExpandDepth(_HalfResample):
    def run(self, depth, shape):
        from fusion_methods._openfocus_resample import _axis_plan
        v = _axis_plan(depth.shape[0], int(shape[0]), linear=True)
        h = _axis_plan(depth.shape[1], int(shape[1]), linear=True)

        def pad(a, taps=8):
            b = np.zeros((a.shape[0], taps), a.dtype)
            b[:, :a.shape[1]] = a
            return b
        H, W = depth.shape
        OH, OW = int(shape[0]), int(shape[1])
        ib = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(depth))
        vb = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, OH * W * 4)
        vix = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                             hostbuf=pad(v.indices))
        vw = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=pad(v.weights))
        vc = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=v.counts.astype(np.int64))
        self.cl.Kernel(self.p, 'vert')(self.q, (W, OH), None, ib, vb, vix, vw, vc,
                                       np.int32(OH), np.int32(W))
        ob = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, OH * OW * 2)
        hix = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                             hostbuf=pad(h.indices))
        hw = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=pad(h.weights))
        hc = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=h.counts.astype(np.int64))
        self.cl.Kernel(self.p, 'horiz')(self.q, (OW, OH), None, vb, ob, hix, hw, hc,
                                        np.int32(OH), np.int32(OW), np.int32(W))
        self.q.finish()
        out = np.empty((OH, OW), np.uint16)
        self.cl.enqueue_copy(self.q, out, ob, is_blocking=True)
        return out


class _GpuB:
    """GPU depth_map (half-res + preserve-detail) + MergeWeighted stream."""

    def __init__(self):
        cl, ctx, q, mf, pk = _gpu()
        self.cl = cl
        self.ctx = ctx
        self.q = q
        self.mf = mf
        self.p = pk
        self.filt = _Filter(cl, ctx, q, mf, pk)
        self.half = _HalfResample(cl, ctx, q, mf, pk)
        self.expand = _ExpandDepth(cl, ctx, q, mf, pk)
        self._cbuf = {}

    # -- per-frame stages -------------------------------------------------
    def _gray(self, frame):
        H, W, C = frame.shape
        if frame.dtype == np.uint8 and C >= 3:
            key = ('gray', H, W)
            if key not in self._cbuf:
                self._cbuf[key] = self.cl.Buffer(
                    self.ctx, self.mf.WRITE_ONLY, H * W * 2)
            ib = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                hostbuf=np.ascontiguousarray(frame))
            self.cl.Kernel(self.p, 'gray_u8')(self.q, (W, H), None, ib,
                                              self._cbuf[key], np.int32(H),
                                              np.int32(W), np.int32(C), np.int32(256))
            self.q.finish()
            out = np.empty((H, W), np.uint16)
            self.cl.enqueue_copy(self.q, out, self._cbuf[key], is_blocking=True)
            return out
        return frame_gray_codes(frame, None, lambda: None, depth_color=True)

    def _detail(self, gray):
        h, w = gray.shape
        oh, ow = h // 2, w // 2
        key = ('detail', h, w)
        if key not in self._cbuf:
            self._cbuf[key] = (
                self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, oh * ow * 2),
                self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, oh * ow * 2))
        pb, sb = self._cbuf[key]
        gb = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(gray))
        self.cl.Kernel(self.p, 'detail_pool')(self.q, (ow, oh), None, gb, pb,
                                              np.int32(h), np.int32(w),
                                              np.int32(oh), np.int32(ow))
        self.cl.Kernel(self.p, 'detail_smooth')(self.q, (ow, oh), None, pb, sb,
                                                np.int32(oh), np.int32(ow))
        self.q.finish()
        out = np.empty((oh, ow), np.uint16)
        self.cl.enqueue_copy(self.q, out, sb, is_blocking=True)
        return out

    def _act(self, half, fine, coarse):
        h, w = half.shape
        key = ('act', h, w)
        if key not in self._cbuf:
            self._cbuf[key] = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY,
                                             half.nbytes)
        fb = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(fine))
        gb = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(half))
        self.cl.Kernel(self.p, 'activity')(self.q, (h * w,), None, gb, fb,
                                           self._cbuf[key], np.int32(h * w))
        self.q.finish()
        out = np.empty_like(half)
        self.cl.enqueue_copy(self.q, out, self._cbuf[key], is_blocking=True)
        return self.filt.run(out, coarse)

    def _select_init(self, a, count):
        depth = np.zeros_like(a)
        votes = np.empty_like(a)
        ab = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(a))
        db = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, a.nbytes)
        vb = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, a.nbytes)
        self.cl.Kernel(self.p, 'init_activity')(self.q, (a.size,), None, ab, db,
                                                vb, np.uint32(count),
                                                np.int32(a.size))
        self.cl.enqueue_copy(self.q, depth, db)
        self.cl.enqueue_copy(self.q, votes, vb)
        self.q.finish()
        return a.copy(), depth, votes

    def _select(self, a, best, depth, votes, code, count):
        ab = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(a))
        bb = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf=best)
        db = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf=depth)
        vb = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf=votes)
        self.cl.Kernel(self.p, 'select_activity')(self.q, (a.size,), None, ab,
                                                  bb, db, vb, np.uint16(code),
                                                  np.uint32(count), np.int32(a.size))
        self.cl.enqueue_copy(self.q, best, bb)
        self.cl.enqueue_copy(self.q, depth, db)
        self.cl.enqueue_copy(self.q, votes, vb)
        self.q.finish()

    # -- depth_map equivalent ---------------------------------------------
    def depth(self, frames, kernel_size, smoothing, feather, shape, count,
              should_cancel, progress_callback=None):
        from fusion_methods.openfocus_depth import _snap_lut
        from fusion_methods._openfocus_depth_rows import (
            confidence_rows, detail_confidence_rows,
            apply_detail_rows, select_detail_rows)
        from fusion_methods.openfocus_ab import _check_cancel

        def cancel():
            _check_cancel(should_cancel)

        radius = max(1, int(kernel_size or 17) // 2)
        parameter = float(np.float64(radius) * np.float64(np.float32(0.075)))
        fine_p = np.float32(parameter * 0.5)
        coarse_p = np.float32(parameter * 2.0)
        step = 65535 // (count - 1)
        best = depth = votes = None
        d_best = d_second = d_winner = None
        for idx, frame in enumerate(frames):
            cancel()
            gray = self._gray(frame)
            detail = self._detail(gray)
            half = self.half.run(gray)
            extent = np.float32(max(half.shape))
            fine_r = np.float32(extent * fine_p / np.float32(100))
            coarse_r = np.float32(extent * coarse_p / np.float32(100))
            fine = self.filt.run(half, fine_r)
            activity = self._act(half, fine, coarse_r)
            code = idx * step
            if best is None:
                best, depth, votes = self._select_init(activity, count)
                d_best = detail.copy()
                d_second = np.zeros_like(detail)
                d_winner = np.zeros_like(detail)
            else:
                self._select(activity, best, depth, votes, code, count)
                select_detail_rows(detail, d_best, d_second, d_winner, code,
                                   0, detail.shape[0])
            if progress_callback is not None:
                progress_callback(idx, count)
        cancel()
        detail_confidence_rows(best, d_best, d_second, 0, d_best.shape[0])
        confidence_rows(best, votes, 0, best.shape[0])
        hist = np.bincount(best.ravel(), minlength=65536)
        threshold = int(np.float32(np.float32(best.size)
                                   * np.float32(0.009999999776482582)))
        lo = int(np.searchsorted(np.cumsum(hist), threshold)) if threshold else 0
        hi = (65535 - int(np.searchsorted(np.cumsum(hist[::-1]), threshold))
              if threshold else 65535)
        positions = np.arange(4096, dtype=np.float64) * 65535.0 / 4095.0
        span = max(hi - lo, 1)
        lut = (65535.0 * np.sqrt(np.clip((positions - lo) / span, 0, 1))
               ).astype(np.float32)
        g = np.empty_like(best)
        bb = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=best)
        lb = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=lut)
        ob = self.cl.Buffer(self.ctx, self.mf.WRITE_ONLY, best.nbytes)
        self.cl.Kernel(self.p, 'guide')(self.q, (best.size,), None, bb, lb, ob,
                                        np.int32(best.size))
        self.cl.enqueue_copy(self.q, g, ob)
        self.q.finish()
        smooth = np.float32(np.float32(smoothing) * np.float32(0.75)
                            * np.float32(0.03125))
        extent = np.float32(max(depth.shape))
        scale = np.float32(extent * smooth / np.float32(100))
        low = np.float32(extent * np.float32(smooth * np.float32(64))
                         / np.float32(100))
        for _ in range(2):
            depth = self.filt.axis_guided(depth, g, scale, low, 1)
            depth = self.filt.axis_guided(depth, g, scale, low, 0)
        half_step = step >> 1
        if half_step:
            sl = _snap_lut(feather).astype(np.int64)
            db = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                hostbuf=depth)
            sb = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                hostbuf=sl)
            self.cl.Kernel(self.p, 'snap')(self.q, (depth.size,), None, db, sb,
                                           np.int32(step), np.int32(depth.size))
            self.cl.enqueue_copy(self.q, depth, db)
            self.q.finish()
        apply_detail_rows(depth, d_second, d_winner, 0, depth.shape[0])
        return self.expand.run(depth, shape)

    # -- merge ------------------------------------------------------------
    def merge(self, frames, depth, shape, should_cancel, progress_callback=None):
        from fusion_methods.openfocus_ab import _check_cancel
        from fusion_methods.openfocus_bc import _active_sources

        def cancel():
            _check_cancel(should_cancel)

        count = len(frames)
        H, W, C = shape
        step = 65535 // max(1, count - 1)
        active = _active_sources(depth.astype(np.int64), count, should_cancel)
        acc = self.cl.Buffer(self.ctx, self.mf.READ_WRITE,
                             H * W * C * 2)
        self.cl.enqueue_fill_buffer(self.q, acc, np.uint16(0), 0, H * W * C * 2)
        db = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(depth, dtype=np.int32))
        u8 = frames[0].dtype == np.uint8
        kernel = 'merge_u8' if u8 else 'merge'
        for index, frame in enumerate(frames):
            cancel()
            if not active[index]:
                if progress_callback is not None:
                    progress_callback(index, count)
                continue
            img = np.ascontiguousarray(frame)
            ib = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                hostbuf=img)
            first = 1 if index == 0 else 0
            if u8:
                self.cl.Kernel(self.p, kernel)(
                    self.q, (W, H), None, acc, ib, db, np.int32(H), np.int32(W),
                    np.int32(C), np.int32(step), np.int32(index), np.int32(first),
                    np.int32(0))
            else:
                self.cl.Kernel(self.p, kernel)(
                    self.q, (W, H), None, acc, ib, db, np.int32(H), np.int32(W),
                    np.int32(C), np.int32(step), np.int32(index), np.int32(1),
                    np.int32(first), np.int32(0))
            if progress_callback is not None:
                progress_callback(index, count)
        cancel()
        out = np.empty((H, W, C), np.uint16)
        self.cl.enqueue_copy(self.q, out, acc, is_blocking=True)
        return out


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def openfocus_b_gpu(frames, shape, img_resize, kernel_size, thread_count,
                    depth_smoothing, should_cancel, depth_override,
                    progress_callback, feather):
    """GPU equivalent of openfocus_b_impl for supported inputs."""
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods.openfocus_bc import _output_codes
    from fusion_methods._openfocus_execution import configure_rows

    def cancel():
        _check_cancel(should_cancel)

    count = len(frames)
    smoothing = 1.44 if depth_smoothing is None else float(depth_smoothing)
    configure_rows(frames, img_resize, 'openfocus_b', thread_count, should_cancel)
    total = count + count + 1
    if progress_callback is not None:
        progress_callback(0, total)
    cancel()

    def depth_progress(index, _total):
        if progress_callback is not None:
            progress_callback(index + 1, total)

    eng = _GpuB()
    depth = eng.depth(frames, kernel_size,
                      smoothing * (5.0 / 1.2), feather, shape, count,
                      should_cancel, depth_progress)
    cancel()

    def merge_progress(index, _total):
        if progress_callback is not None:
            progress_callback(count + index + 1, total)

    result = eng.merge(frames, depth, shape, should_cancel, merge_progress)
    cancel()
    del depth
    output = _output_codes(result, frames, should_cancel)
    if progress_callback is not None:
        progress_callback(total, total)
    cancel()
    return output
