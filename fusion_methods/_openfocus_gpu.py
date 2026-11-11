"""Bounded OpenCL depth activity and ordered merge for OpenFocus B."""
import logging
from threading import Lock

import numpy as np

_device_lock = Lock()

_MERGE_KERNEL = r'''
__kernel void activity(__global const ushort *gray, __global const ushort *fine,
                       __global ushort *out)
{
    size_t i = get_global_id(0);
    ulong g = gray[i], f = fine[i];
    ulong maximum = max(g, f);
    ulong denominator = ((maximum * 100ul * 0x80008001ul) >> 32) >> 15;
    ulong difference = g > f ? g - f : f - g;
    out[i] = (ushort)min(65535ul, 110ul * difference / (denominator + 10ul));
}

__kernel void merge(__global ushort *acc, __global const ushort *img,
                    __global const long *depth, const int width,
                    const int channels, const int step, const int index,
                    const int scale, const int first)
{
    size_t pixel = get_global_id(0);
    int weight = 1024;
    if (!first) {
        long code = depth[pixel];
        if (code < 0 || code > 131070) return;
        long delta = code - (long)step * index;
        if (delta >= 0) {
            if (delta > step) return;
        } else {
            if (delta <= -step) return;
            weight = 1024 - (int)(((-delta) << 10) / step);
            if (weight <= 0) return;
        }
    }
    for (int c = 0; c < channels; ++c) {
        size_t k = pixel * channels + c;
        int value = (int)img[k] * scale;
        if (weight == 1024) acc[k] = (ushort)value;
        else acc[k] = (ushort)(((int)acc[k] * (1024 - weight)
                                + value * weight) >> 10);
    }
}
'''


def _device(cl):
    for platform in cl.get_platforms():
        try:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
        except cl.Error:
            continue
        for device in devices:
            if (device.type & cl.device_type.GPU and device.available
                    and device.compiler_available):
                return device
    return None


def gpu_available():
    """Return whether an available OpenCL GPU with a compiler is present."""
    try:
        import pyopencl as cl
        return _device(cl) is not None
    except Exception:
        return False


class GpuError(RuntimeError):
    """A device operation failed; the caller may continue on the CPU."""


def log_fallback(error):
    reason = error.__cause__ or error
    logging.getLogger(__name__).warning('OpenFocus GPU fallback to CPU: %s', reason)


def device_status(gpu):
    if gpu is None or not gpu.used:
        return 'CPU'
    if gpu.failed:
        return 'CPU + ' + gpu.device_name
    return gpu.device_name


def _free_vram(device):
    """Read free memory from the selected OpenCL GPU's matching driver."""
    import subprocess
    if (int(device.vendor_id) == 0x1002
            and 'cl_amd_device_attribute_query' in device.extensions.split()):
        try:
            free_kib = int(device.get_info(0x4039)[0])
            if 0 < free_kib * 1024 <= int(device.global_mem_size):
                return free_kib * 1024, int(device.global_mem_size)
        except Exception:
            pass
    if int(device.vendor_id) != 0x10de:
        return None
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.free,memory.total',
             '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=3, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        matches = []
        for line in result.stdout.splitlines():
            name, free, total = (part.strip() for part in line.rsplit(',', 2))
            total_bytes = int(total) * 1024 * 1024
            if (name == str(device.name).strip()
                    and abs(total_bytes - int(device.global_mem_size)) <= 64 * 1024 * 1024):
                matches.append((int(free) * 1024 * 1024, total_bytes))
        return matches[0] if len(matches) == 1 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class Merge:
    """A job-local OpenCL context and bounded merge workspace."""

    def __init__(self, width, channels):
        try:
            import pyopencl as cl
            from core import resource_policy
            from core.multi_focus_fusion import memory_budget_bytes

            device = _device(cl)
            if device is None:
                raise GpuError('No available OpenCL GPU')
            # Account for host staging and device allocations.
            row = int(width) * (8 + 5 * int(channels))
            budget = min(int(memory_budget_bytes()),
                         int(resource_policy.frame_ram_budget(resource_policy.probe_resources())),
                         int(device.global_mem_size) // 4, 16 * 1024 * 1024)
            largest_row = int(width) * max(8, 2 * int(channels))
            rows = min(512, budget // max(1, 2 * row),
                       int(device.max_mem_alloc_size) // max(1, largest_row))
            if rows < 1 or width <= 0 or channels <= 0:
                raise GpuError('GPU merge workspace exceeds the resource budget')
            self.cl = cl
            self.device = device
            self.context = cl.Context(devices=[device])
            self.queue = cl.CommandQueue(self.context, device=device)
            self.program = cl.Program(self.context, _MERGE_KERNEL).build()
            self.device_name = 'OpenCL: ' + str(device.name).strip()
            self.activity_kernel = cl.Kernel(self.program, 'activity')
            self.kernel = cl.Kernel(self.program, 'merge')
            self.rows = rows
            self.failed = False
            self.used = False
        except GpuError:
            raise
        except Exception as exc:
            raise GpuError('OpenCL GPU initialization failed') from exc

    def acquire(self, should_cancel):
        from fusion_methods.openfocus_ab import _check_cancel
        while not _device_lock.acquire(timeout=0.1):
            _check_cancel(should_cancel)
        try:
            _check_cancel(should_cancel)
        except BaseException:
            _device_lock.release()
            raise

    def release(self):
        _device_lock.release()

    def full_depth(self, frames, shape, img_resize, kernel_size, smoothing,
                   feather, should_cancel, progress_callback):
        """Run the PR depth stages only when all resident planes fit the job."""
        from core import resource_policy
        from core.multi_focus_fusion import memory_budget_bytes
        from core.source_frames import frame_resolution_bytes, source_cache_growth_bytes
        height, width = shape[:2]
        if min(height, width) < 2:
            return None
        pixels = int(height) * int(width)
        host_budget = min(int(memory_budget_bytes()),
                          int(resource_policy.frame_ram_budget(resource_policy.probe_resources())))
        source_bytes = max((frame_resolution_bytes(frame) for frame in frames), default=0)
        source_bytes += source_cache_growth_bytes(frames)
        if (pixels * 112 + source_bytes + 16 * 1024 * 1024 > host_budget
                or pixels * 8 > int(self.device.max_mem_alloc_size)):
            logging.getLogger(__name__).info('OpenFocus full GPU depth exceeds the job memory budget')
            return None

        class CallbackFailure(BaseException):
            pass

        def progress(index, total):
            if progress_callback is not None:
                try:
                    progress_callback(index, total)
                except BaseException as exc:
                    raise CallbackFailure() from exc

        self.acquire(should_cancel)
        try:
            if self.device.host_unified_memory:
                # Device allocations share the same physical RAM as staging.
                if pixels * 160 + source_bytes + 32 * 1024 * 1024 > host_budget:
                    logging.getLogger(__name__).info('OpenFocus full GPU depth exceeds unified memory budget')
                    return None
            else:
                from dataclasses import replace
                memory = _free_vram(self.device)
                if memory is None:
                    logging.getLogger(__name__).info('OpenFocus full GPU depth has no verified free VRAM reading')
                    return None
                snapshot = replace(resource_policy.probe_resources(),
                                   free_vram=memory[0], total_vram=memory[1])
                if pixels * 48 + 16 * 1024 * 1024 > resource_policy.vram_budget(snapshot):
                    logging.getLogger(__name__).info('OpenFocus full GPU depth exceeds the free VRAM budget')
                    return None
            try:
                self.depth_program = self.cl.Program(
                    self.context, _FILTER_K + _DEPTH_K + _RESAMPLE_K).build()
                engine = _GpuB(self, img_resize, should_cancel)
                return engine.depth(frames, kernel_size, smoothing, feather, shape,
                                    len(frames), should_cancel, progress)
            except CallbackFailure as exc:
                raise exc.__cause__
            except (self.cl.Error, MemoryError) as exc:
                raise GpuError('OpenCL GPU depth failed') from exc
        finally:
            self.release()

    def activity(self, gray, fine, output):
        try:
            cl = self.cl
            source = np.ascontiguousarray(gray, dtype=np.uint16)
            filtered = np.ascontiguousarray(fine, dtype=np.uint16)
            result = np.empty(source.shape, dtype=np.uint16)
            flags = cl.mem_flags
            gb = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=source)
            fb = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=filtered)
            ob = cl.Buffer(self.context, flags.WRITE_ONLY, result.nbytes)
            self.activity_kernel(self.queue, (source.size,), None, gb, fb, ob)
            cl.enqueue_copy(self.queue, result, ob, is_blocking=True)
        except Exception as exc:
            raise GpuError('OpenCL GPU depth activity failed') from exc
        output[...] = result

    def begin_merge(self, depth, shape, should_cancel):
        """Reserve a whole-image merge, or leave the bounded band path in use."""
        from core import resource_policy
        from core.multi_focus_fusion import memory_budget_bytes
        from dataclasses import replace
        height, width = shape[:2]
        channels = 1 if len(shape) == 2 else shape[2]
        pixels = int(height) * int(width)
        required = pixels * (8 + 4 * channels)
        host = min(int(memory_budget_bytes()),
                   int(resource_policy.frame_ram_budget(resource_policy.probe_resources())))
        if required * 2 + 16 * 1024 * 1024 > host:
            return False
        if pixels * max(8, 2 * channels) > int(self.device.max_mem_alloc_size):
            return False
        self.acquire(should_cancel)
        try:
            if self.device.host_unified_memory:
                admitted = required * 3 + 16 * 1024 * 1024 <= host
            else:
                memory = _free_vram(self.device)
                snapshot = (None if memory is None else
                            replace(resource_policy.probe_resources(),
                                    free_vram=memory[0], total_vram=memory[1]))
                admitted = (snapshot is not None and required + 16 * 1024 * 1024
                            <= resource_policy.vram_budget(snapshot))
            if not admitted:
                self.release()
                return False
            cl = self.cl
            codes = np.ascontiguousarray(depth, dtype=np.int64)
            self._merge_depth = cl.Buffer(self.context, cl.mem_flags.READ_ONLY |
                                          cl.mem_flags.COPY_HOST_PTR, hostbuf=codes)
            self._merge_acc = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                        pixels * channels * 2)
            self._merge_shape = shape
            return True
        except Exception as exc:
            self._merge_acc = self._merge_depth = None
            self.release()
            raise GpuError('OpenCL GPU merge allocation failed') from exc

    def resident_frame(self, frame, index, count, scale):
        try:
            cl = self.cl
            image = np.ascontiguousarray(frame, dtype=np.uint16)
            ib = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=image)
            height, width = self._merge_shape[:2]
            channels = 1 if image.ndim == 2 else image.shape[2]
            event = self.kernel(self.queue, (height * width,), None, self._merge_acc,
                                ib, self._merge_depth, np.int32(width), np.int32(channels),
                                np.int32(65535 // max(1, count - 1)), np.int32(index),
                                np.int32(scale), np.int32(index == 0))
            event.wait()
            self.used = True
        except Exception as exc:
            raise GpuError('OpenCL GPU merge failed') from exc

    def finish_merge(self, result):
        try:
            output = np.empty(self._merge_shape, dtype=np.uint16)
            self.cl.enqueue_copy(self.queue, output, self._merge_acc, is_blocking=True)
        except Exception as exc:
            raise GpuError('OpenCL GPU merge download failed') from exc
        result[...] = output

    def end_merge(self):
        self._merge_acc = self._merge_depth = None
        self.release()

    def band(self, accumulator, image, depth, index, step, scale, first):
        """Merge a band transactionally; failure leaves the accumulator intact."""
        try:
            cl = self.cl
            # Initial accumulation is deliberately not read from an
            # uninitialized transient frame.
            acc = (np.zeros(accumulator.shape, dtype=np.uint16) if first else
                   np.ascontiguousarray(accumulator, dtype=np.uint16))
            img = np.ascontiguousarray(image, dtype=np.uint16)
            dep = np.ascontiguousarray(depth, dtype=np.int64)
            flags = cl.mem_flags
            ab = cl.Buffer(self.context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=acc)
            ib = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=img)
            db = cl.Buffer(self.context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=dep)
            channels = 1 if image.ndim == 2 else image.shape[2]
            self.kernel(self.queue, (depth.size,), None, ab, ib, db,
                        np.int32(depth.shape[1]), np.int32(channels),
                        np.int32(step), np.int32(index), np.int32(scale),
                        np.int32(first))
            # Copy to private staging before committing to the output plane.
            result = np.empty_like(acc)
            cl.enqueue_copy(self.queue, result, ab, is_blocking=True)
        except Exception as exc:
            raise GpuError('OpenCL GPU merge failed') from exc
        accumulator[...] = result


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
#pragma OPENCL FP_CONTRACT OFF
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


class _Filter:
    """Unguided/guided IIR axis filter; mirrors gpu_filter.Filter + device ctx."""

    def __init__(self, cl, ctx, q, mf, prg):
        self.cl = cl
        self.ctx = ctx
        self.q = q
        self.mf = mf
        self.p = prg
        self.kernels = {name: cl.Kernel(self.p, name) for name in ('filt4', 'filt_guided')}
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
                    self.cl.Buffer(self.ctx, self.mf.READ_WRITE, n * 2),
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
        self.kernels['filt4'](self.q, (step,), None, ib, ob, fb,
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
        ob = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, data.nbytes)
        self.kernels['filt_guided'](self.q, (w,), None, ib, gb, ob, fb,
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
        self.kernels = {name: cl.Kernel(self.p, name) for name in ('vert', 'horiz')}
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
        vb = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, OH * W * 4)
        self.kernels['vert'](self.q, (W, OH), None, ib, vb, vix, vw, vc,
                                       np.int32(OH), np.int32(W))
        ob = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, OH * OW * 2)
        self.kernels['horiz'](self.q, (OW, OH), None, vb, ob, hix, hw, hc,
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
        vb = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, OH * W * 4)
        vix = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                             hostbuf=pad(v.indices))
        vw = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=pad(v.weights))
        vc = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=v.counts.astype(np.int64))
        self.kernels['vert'](self.q, (W, OH), None, ib, vb, vix, vw, vc,
                                       np.int32(OH), np.int32(W))
        ob = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, OH * OW * 2)
        hix = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                             hostbuf=pad(h.indices))
        hw = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=pad(h.weights))
        hc = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=h.counts.astype(np.int64))
        self.kernels['horiz'](self.q, (OW, OH), None, vb, ob, hix, hw, hc,
                                        np.int32(OH), np.int32(OW), np.int32(W))
        self.q.finish()
        out = np.empty((OH, OW), np.uint16)
        self.cl.enqueue_copy(self.q, out, ob, is_blocking=True)
        return out


class _GpuB:
    """GPU depth_map (half-res + preserve-detail) + MergeWeighted stream."""

    def __init__(self, gpu, img_resize, should_cancel):
        self.gpu = gpu
        cl, ctx, q, mf, pk = gpu.cl, gpu.context, gpu.queue, gpu.cl.mem_flags, gpu.depth_program
        self.img_resize = img_resize
        self.should_cancel = should_cancel
        self.cl = cl
        self.ctx = ctx
        self.q = q
        self.mf = mf
        self.p = pk
        self.kernels = {name: cl.Kernel(self.p, name) for name in ('gray_u8', 'detail_pool', 'detail_smooth', 'activity', 'init_activity', 'select_activity', 'guide', 'snap')}
        self.filt = _Filter(cl, ctx, q, mf, pk)
        self.half = _HalfResample(cl, ctx, q, mf, pk)
        self.expand = _ExpandDepth(cl, ctx, q, mf, pk)
        self._cbuf = {}

    # -- per-frame stages -------------------------------------------------
    def _gray(self, frame):
        from fusion_methods._openfocus_pixels import frame_gray_codes
        from core.source_frames import resolve_frame
        pixels = resolve_frame(frame, should_cancel=self.should_cancel)
        if (self.img_resize is None and pixels.dtype == np.uint8
                and pixels.ndim == 3 and pixels.shape[2] == 3):
            height, width, channels = pixels.shape
            source = np.ascontiguousarray(pixels)
            gray = np.empty((height, width), dtype=np.uint16)
            ib = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                hostbuf=source)
            ob = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, gray.nbytes)
            self.kernels['gray_u8'](self.q, (width, height), None, ib, ob,
                                              np.int32(height), np.int32(width),
                                              np.int32(channels), np.int32(256))
            self.cl.enqueue_copy(self.q, gray, ob, is_blocking=True)
            self.gpu.used = True
            return gray
        gray = frame_gray_codes(pixels, self.img_resize, self.should_cancel, depth_color=True)
        return np.ascontiguousarray(gray, dtype=np.uint16)

    def _detail(self, gray):
        h, w = gray.shape
        oh, ow = h // 2, w // 2
        key = ('detail', h, w)
        if key not in self._cbuf:
            self._cbuf[key] = (
                self.cl.Buffer(self.ctx, self.mf.READ_WRITE, oh * ow * 2),
                self.cl.Buffer(self.ctx, self.mf.READ_WRITE, oh * ow * 2))
        pb, sb = self._cbuf[key]
        gb = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(gray))
        self.kernels['detail_pool'](self.q, (ow, oh), None, gb, pb,
                                              np.int32(h), np.int32(w),
                                              np.int32(oh), np.int32(ow))
        self.kernels['detail_smooth'](self.q, (ow, oh), None, pb, sb,
                                                np.int32(oh), np.int32(ow))
        self.q.finish()
        out = np.empty((oh, ow), np.uint16)
        self.cl.enqueue_copy(self.q, out, sb, is_blocking=True)
        return out

    def _act(self, half, fine, coarse):
        h, w = half.shape
        key = ('act', h, w)
        if key not in self._cbuf:
            self._cbuf[key] = self.cl.Buffer(self.ctx, self.mf.READ_WRITE,
                                             half.nbytes)
        fb = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(fine))
        gb = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(half))
        self.kernels['activity'](self.q, (h * w,), None, gb, fb,
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
        db = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
        vb = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, a.nbytes)
        self.kernels['init_activity'](self.q, (a.size,), None, ab, db,
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
        self.kernels['select_activity'](self.q, (a.size,), None, ab,
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
            if gray.shape != tuple(shape[:2]):
                raise ValueError("OpenFocus depth requires frames with matching shapes")
            cancel()
            detail = self._detail(gray)
            self.gpu.used = True
            cancel()
            half = self.half.run(gray)
            cancel()
            extent = np.float32(max(half.shape))
            fine_r = np.float32(extent * fine_p / np.float32(100))
            coarse_r = np.float32(extent * coarse_p / np.float32(100))
            fine = self.filt.run(half, fine_r)
            cancel()
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
        ob = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, best.nbytes)
        self.kernels['guide'](self.q, (best.size,), None, bb, lb, ob,
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
            cancel()
            depth = self.filt.axis_guided(depth, g, scale, low, 1)
            cancel()
            depth = self.filt.axis_guided(depth, g, scale, low, 0)
        half_step = step >> 1
        if half_step:
            sl = _snap_lut(feather).astype(np.int64)
            db = self.cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                hostbuf=depth)
            sb = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                hostbuf=sl)
            self.kernels['snap'](self.q, (depth.size,), None, db, sb,
                                           np.int32(step), np.int32(depth.size))
            self.cl.enqueue_copy(self.q, depth, db)
            self.q.finish()
        apply_detail_rows(depth, d_second, d_winner, 0, depth.shape[0])
        return self.expand.run(depth, shape)

