"""Bounded OpenCL depth activity and ordered merge for OpenFocus B."""
import atexit
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
import contextvars
import hashlib
import logging
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from threading import Event, Lock

import numpy as np

_device_lock = Lock()
_context_lock = Lock()
_cache_lock = Lock()
_transfer_queue_lock = Lock()
_SHARED_CONTEXT = None
_PROGRAM_CACHE = None
_DEPTH_PROGRAM_CACHE = None
_TRANSFER_QUEUE_CACHE = None

_nvml_lock = Lock()
_nvml_state = None
_nvml_atexit_registered = False

_MERGE_KERNEL_NAMES = (
    'activity', 'merge', 'merge_u8_i64', 'merge_u16_u16', 'merge_u8_u16',
    'merge_bgr_rev', 'merge_u8_i64_bgr_rev', 'merge_u16_u16_bgr_rev', 'merge_u8_u16_bgr_rev',
    'merge_out_u8',
)
_DEPTH_KERNEL_NAMES = (
    'gray_u8', 'gray_u16', 'detail_pool', 'detail_smooth', 'detail_smooth_h',
    'detail_smooth_v', 'detail_select',
    'init_detail', 'detail_confidence', 'confidence', 'zero_u32', 'confidence_hist',
    'apply_detail',
    'activity', 'init_activity', 'select_activity', 'guide', 'snap',
    'sample_focus_plane',
)
_FILTER_KERNEL_NAMES = ('filt', 'filt_guided', 'transpose_u16')
_RESAMPLE_KERNEL_NAMES = ('vert', 'horiz')

_PREVIEW_KERNEL = r'''
__kernel void sample_focus_plane(__global const {type}* plane, __global {type}* sample,
                                 const int height, const int width,
                                 const int rows, const int cols)
{{
    int col = get_global_id(0), row = get_global_id(1);
    size_t y = ((2 * row + 1) * (size_t)height) / (2 * rows);
    size_t x = ((2 * col + 1) * (size_t)width) / (2 * cols);
    sample[row * cols + col] = plane[y * width + x];
}}
'''


class _PreviewPlaneReader:
    """Gather a bounded focus plane on-device, then read only the sample."""

    def __init__(self, cl, context, queue, kernel, shape, dtype):
        self._cl, self._context, self._queue = cl, context, queue
        self._kernel, self._shape, self._dtype = kernel, shape, dtype
        self._buffer = self._pixels = None

    def read(self, source, preview):
        from core.fusion_live_preview import preview_shape

        try:
            if self._pixels is None:
                self._pixels = np.empty(preview_shape(*self._shape), self._dtype)
                self._buffer = self._cl.Buffer(self._context, self._cl.mem_flags.WRITE_ONLY,
                                               self._pixels.nbytes)
            rows, cols = self._pixels.shape
            self._kernel(self._queue, (cols, rows), None, source, self._buffer,
                         *map(np.int32, (*self._shape, rows, cols)))
            self._cl.enqueue_copy(self._queue, self._pixels, self._buffer, is_blocking=True)
        except (self._cl.Error, MemoryError):
            preview.unavailable()
            return None
        return self._pixels

    def close(self):
        _release_buffers(self._buffer)
        self._buffer = None


def _release_buffers(*items):
    """Explicitly release any OpenCL buffer objects without raising."""
    seen = set()

    def _walk(item):
        if item is None:
            return
        if isinstance(item, dict):
            for value in item.values():
                _walk(value)
            return
        if isinstance(item, (list, tuple)):
            for value in item:
                _walk(value)
            return
        release = getattr(item, 'release', None)
        if callable(release):
            obj_id = id(item)
            if obj_id in seen:
                return
            seen.add(obj_id)
            try:
                release()
            except Exception:
                pass

    for entry in items:
        _walk(entry)


def _prepare_gpu_upload(array, dtype=None):
    """Return (contiguous_host_buffer, bgr_rev) without copying reversed RGB views."""
    if dtype is not None and array.dtype != dtype:
        return np.ascontiguousarray(array, dtype=dtype), False
    if array.flags.c_contiguous:
        return array, False
    if array.ndim == 3 and array.shape[2] == 3:
        reversed_rgb = array[..., ::-1]
        if reversed_rgb.flags.c_contiguous:
            return reversed_rgb, True
    return np.ascontiguousarray(array), False


def _supports_tiled_transpose(device):
    item_sizes = getattr(device, 'max_work_item_sizes', (16, 16))
    return (getattr(device, 'max_work_group_size', 256) >= 256
            and len(item_sizes) >= 2 and min(item_sizes[:2]) >= 16
            and getattr(device, 'local_mem_size', 16 * 17 * 2) >= 16 * 17 * 2)


def _opencl_cache_dir() -> Path | None:
    """Return the persistent directory for compiled OpenCL binaries."""
    try:
        override = os.environ.get('OPENFOCUS_OPENCL_CACHE_DIR')
        if override:
            path = Path(override)
        else:
            local = (os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CACHE_HOME')
                     or (Path.home() / '.cache'))
            path = Path(local) / 'OpenFocus' / 'opencl_cache'
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        return None


def _program_cache_path(device, source: str, options=None) -> Path | None:
    cache_dir = _opencl_cache_dir()
    if cache_dir is None:
        return None
    opts = tuple(str(opt) for opt in options) if options else ()
    platform = getattr(device, 'platform', None)
    key = '|'.join((
        str(getattr(device, 'name', '')),
        str(getattr(device, 'vendor', '')),
        str(getattr(device, 'driver_version', '')),
        str(getattr(device, 'version', '')),
        str(getattr(platform, 'name', '')),
        str(getattr(platform, 'vendor', '')),
        str(getattr(platform, 'version', '')),
        str(opts),
        os.environ.get('PYOPENCL_BUILD_OPTIONS', ''),
    )).encode('utf-8', errors='replace')
    digest = hashlib.sha256(key + b'\0' + source.encode('utf-8')).hexdigest()
    return cache_dir / f'{digest}.clbin'


def _build_cached_program(cl, device, context, source, *, options=None):
    """Build an OpenCL program using an on-disk binary cache keyed by device/driver/source."""
    build_opts = list(options) if options else []
    cache_path = _program_cache_path(device, source, build_opts)
    can_binary = (
        cache_path is not None
        and hasattr(cl, 'program_info')
        and hasattr(cl.program_info, 'BINARIES')
    )
    if can_binary:
        try:
            blob = cache_path.read_bytes()
            if blob:
                prg = cl.Program(context, [device], [blob])
                return prg.build(options=build_opts) if build_opts else prg.build()
        except Exception:
            pass  # Missing, unreadable, or stale cache entries are rebuilt from source.
    prg = cl.Program(context, source)
    program = prg.build(options=build_opts) if build_opts else prg.build()
    if can_binary:
        tmp_path = None
        try:
            binaries = program.get_info(cl.program_info.BINARIES)
            if isinstance(binaries, (list, tuple)) and binaries:
                blob = binaries[0]
                if isinstance(blob, (bytes, bytearray)) and blob:
                    with NamedTemporaryFile(dir=cache_path.parent, prefix=cache_path.stem,
                                            suffix='.tmp', delete=False) as stream:
                        tmp_path = Path(stream.name)
                        stream.write(blob)
                    tmp_path.replace(cache_path)
        except Exception:
            pass
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
    return program


def _build_filter_program(cl, device, context, extra_source, *, options=None):
    """Build the program and return whether its transpose can launch tiled."""
    tiled = _supports_tiled_transpose(device)
    source = _TRANSPOSE_TILED_K if tiled else _TRANSPOSE_NAIVE_K
    program = _build_cached_program(
        cl, device, context, _FILTER_K_BODY + source + extra_source, options=options,
    )
    if tiled:
        group_size = cl.Kernel(program, 'transpose_u16').get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE, device,
        )
        if group_size < 256:
            program = _build_cached_program(
                cl, device, context, _FILTER_K_BODY + _TRANSPOSE_NAIVE_K + extra_source,
                options=options,
            )
            tiled = False
    return program, tiled


def _prefetch_window(host_budget: int, host_staging: int, frame_bytes: int, count: int,
                     *, max_slots: int = 3, thread_count=None) -> int:
    """Return bounded background prefetch depth permitted by host memory headroom."""
    if thread_count is None:
        from fusion_methods._openfocus_execution import _current
        job = _current.get()
        thread_count = getattr(job, 'thread_count', None)
    if count < 2 or thread_count == 1 or frame_bytes <= 0:
        return 0
    if host_staging + frame_bytes > host_budget:
        return 0
    headroom = max(0, int(host_budget) - int(host_staging))
    cpu_cap = max(1, (os.cpu_count() or 2) // 2)
    slot_cap = min(int(max_slots), cpu_cap)
    if thread_count is not None and thread_count > 0:
        slot_cap = min(slot_cap, thread_count - 1)
    return max(0, min(slot_cap, int(headroom // int(frame_bytes)), int(count) - 1))


def _wait_event(event, cancel=None) -> None:
    """Wait on an OpenCL event if present and cancel-check before and after."""
    if cancel is not None:
        cancel()
    if event is not None and hasattr(event, 'wait'):
        event.wait()
    if cancel is not None:
        cancel()


def _event_wait_list(event):
    if event is not None and hasattr(event, 'wait'):
        return [event]
    return None


def _enqueue_kernel(kernel, queue, gsize, lsize, *args, wait_for=None):
    if wait_for:
        try:
            return kernel(queue, gsize, lsize, *args, wait_for=wait_for)
        except TypeError:
            for ev in wait_for:
                _wait_event(ev)
    return kernel(queue, gsize, lsize, *args)


def _get_transfer_queue(cl, device, context):
    """Return a cached dedicated transfer queue on ``context`` for async H2D DMA."""
    global _TRANSFER_QUEUE_CACHE
    key = (_device_cache_key(device), id(context))
    with _transfer_queue_lock:
        if _TRANSFER_QUEUE_CACHE is not None and _TRANSFER_QUEUE_CACHE[0] == key:
            return _TRANSFER_QUEUE_CACHE[1]
        try:
            tq = cl.CommandQueue(context, device=device)
        except Exception:
            return None
        _TRANSFER_QUEUE_CACHE = (key, tq)
        return tq


def _drain_command_queue(queue, context=None):
    if queue is not None:
        try:
            queue.finish()
        except Exception:
            _invalidate_program_cache(context)


class _PinnedUploader:
    """Overlapped async H2D uploader using a dedicated OpenCL transfer queue."""

    def __init__(self, cl, ctx, compute_q, transfer_q, slots, max_bytes, *, use_pinned=False):
        self._cl = cl
        self._ctx = ctx
        self._compute_q = compute_q
        self._tq = transfer_q if transfer_q is not None else compute_q
        self._slots = max(1, int(slots))
        self._max_bytes = max(0, int(max_bytes))
        self._dma_events = [None] * self._slots
        self.pinned = False

    @property
    def separate_queue(self):
        return self._tq is not self._compute_q

    def upload(self, slot, dev_buf, host_array, *, wait_for=None, cancel=None):
        """Enqueue async H2D copy on the transfer queue and return its event."""
        prev_dma = self._dma_events[slot]
        if prev_dma is not None:
            _wait_event(prev_dma, cancel)
            self._dma_events[slot] = None
        wait_list = wait_for if (wait_for and self.separate_queue) else None
        try:
            ev = self._cl.enqueue_copy(
                self._tq, dev_buf, host_array, is_blocking=False, wait_for=wait_list,
            )
        except TypeError:
            if wait_list:
                for item in wait_list:
                    _wait_event(item, cancel)
            ev = self._cl.enqueue_copy(self._tq, dev_buf, host_array, is_blocking=False)
        self._dma_events[slot] = ev
        if self.separate_queue and hasattr(self._tq, 'flush'):
            self._tq.flush()
        return ev

    def finish(self, cancel=None):
        for slot in range(len(self._dma_events)):
            ev = self._dma_events[slot]
            if ev is not None:
                _wait_event(ev, cancel)
                self._dma_events[slot] = None

    def close(self):
        for slot in range(len(self._dma_events)):
            ev = self._dma_events[slot]
            if ev is not None and hasattr(ev, 'wait'):
                try:
                    ev.wait()
                except Exception:
                    pass
            self._dma_events[slot] = None


def _iter_prefetched(items, prepare_fn, *, enabled: bool, should_cancel=None,
                     max_in_flight: int = 1):
    """Yield (index, prepare_fn(index, item, cancelled)) with bounded background prefetch."""
    from fusion_methods.openfocus_ab import _check_cancel
    from fusion_methods._openfocus_execution import _current, _worker_native_limit

    total = len(items)
    if total == 0:
        return
    job = _current.get() if enabled else None
    if job is not None and job.thread_count == 1:
        enabled = False
    slots = max(0, min(int(max_in_flight), total - 1)) if enabled else 0
    if slots <= 0 or total < 2:
        for index, item in enumerate(items):
            _check_cancel(should_cancel)
            result = prepare_fn(index, item, should_cancel)
            _check_cancel(should_cancel)
            yield index, result
            del result
        return

    stop = Event()

    def cancelled():
        return stop.is_set() or (should_cancel is not None and should_cancel())

    def submit_item(pool, index):
        ctx = contextvars.copy_context()
        return pool.submit(ctx.run, prepare_fn, index, items[index], cancelled)

    _check_cancel(should_cancel)
    workers = min(slots, 3)
    with ThreadPoolExecutor(max_workers=workers, initializer=_worker_native_limit) as pool:
        pending = deque()
        next_submit = 0
        try:
            while next_submit < min(slots, total):
                pending.append(submit_item(pool, next_submit))
                next_submit += 1
            for index in range(total):
                future = pending[0]
                while not future.done():
                    _check_cancel(should_cancel)
                    wait((future,), timeout=0.05)
                _check_cancel(should_cancel)
                pending.popleft()
                result = future.result()
                while next_submit < total and len(pending) < slots:
                    pending.append(submit_item(pool, next_submit))
                    next_submit += 1
                yield index, result
                del result
        finally:
            stop.set()
            for future in pending:
                future.cancel()


def _device_cache_key(device):
    return (
        getattr(device, 'int_ptr', None),
        str(getattr(device, 'name', '')),
        str(getattr(device, 'vendor', '')),
        str(getattr(device, 'driver_version', '')),
        int(getattr(device, 'global_mem_size', 0)),
    )


def _get_shared_context(cl, device):
    """Share the selected device's context while each stage owns its queue."""
    global _SHARED_CONTEXT
    key = _device_cache_key(device)
    factory = cl.Context
    with _context_lock:
        if _SHARED_CONTEXT is not None:
            cached_key, cached_cl, cached_factory, cached_device, context = _SHARED_CONTEXT
            if (cached_key == key and cached_cl is cl and cached_factory is factory
                    and (key[0] is not None or cached_device is device)):
                return context
        try:
            context = factory(devices=[device])
        except TypeError:
            context = factory([device])
        _SHARED_CONTEXT = (key, cl, factory, device, context)
        return context


def _invalidate_shared_context(context):
    global _SHARED_CONTEXT, _TRANSFER_QUEUE_CACHE
    if context is not None:
        with _context_lock:
            if _SHARED_CONTEXT is not None and _SHARED_CONTEXT[-1] is context:
                _SHARED_CONTEXT = None
        with _transfer_queue_lock:
            if _TRANSFER_QUEUE_CACHE is not None and _TRANSFER_QUEUE_CACHE[0][1] == id(context):
                _TRANSFER_QUEUE_CACHE = None


def _get_or_build_merge_program(cl, device):
    global _PROGRAM_CACHE, _DEPTH_PROGRAM_CACHE
    key = _device_cache_key(device)
    with _cache_lock:
        ctx = _get_shared_context(cl, device)
        if _PROGRAM_CACHE is not None and _PROGRAM_CACHE[0] == key and _PROGRAM_CACHE[1] is ctx:
            return _PROGRAM_CACHE[1:]
        queue = cl.CommandQueue(ctx, device=device)
        program = _build_cached_program(cl, device, ctx, _MERGE_KERNEL)
        kernels = {name: cl.Kernel(program, name) for name in _MERGE_KERNEL_NAMES}
        _PROGRAM_CACHE = (key, ctx, queue, program, kernels)
        _DEPTH_PROGRAM_CACHE = None
        return ctx, queue, program, kernels


def _get_or_build_depth_program(cl, device, context):
    global _DEPTH_PROGRAM_CACHE
    key = (_device_cache_key(device), id(context))
    with _cache_lock:
        if _DEPTH_PROGRAM_CACHE is not None and _DEPTH_PROGRAM_CACHE[0] == key:
            return _DEPTH_PROGRAM_CACHE[1:]
        source = _DEPTH_K + _RESAMPLE_K + _PREVIEW_KERNEL.format(type='ushort')
        program, tiled = _build_filter_program(cl, device, context, source)
        depth_kernels = {name: cl.Kernel(program, name) for name in _DEPTH_KERNEL_NAMES}
        filter_kernels = {name: cl.Kernel(program, name) for name in _FILTER_KERNEL_NAMES}
        resample_kernels = {name: cl.Kernel(program, name) for name in _RESAMPLE_KERNEL_NAMES}
        _DEPTH_PROGRAM_CACHE = (
            key, program, depth_kernels, filter_kernels, resample_kernels, tiled,
        )
        return program, depth_kernels, filter_kernels, resample_kernels, tiled


def _invalidate_program_cache(context=None):
    global _PROGRAM_CACHE, _DEPTH_PROGRAM_CACHE, _TRANSFER_QUEUE_CACHE
    with _cache_lock:
        _PROGRAM_CACHE = None
        _DEPTH_PROGRAM_CACHE = None
    with _transfer_queue_lock:
        _TRANSFER_QUEUE_CACHE = None
    _invalidate_shared_context(context)

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

#define DEFINE_MERGE(NAME, IMG_T, DEPTH_T, CHECK_BOUNDS, BGR_REV) \
__kernel void NAME(__global ushort *acc, __global const IMG_T *img, \
                   __global const DEPTH_T *depth, const int width, \
                   const int channels, const int step, const int index, \
                   const int scale, const int first) \
{ \
    size_t pixel = get_global_id(0); \
    int weight = 1024; \
    if (!first) { \
        long code = (long)depth[pixel]; \
        if (CHECK_BOUNDS && (code < 0 || code > 131070)) return; \
        long delta = code - (long)step * index; \
        if (delta >= 0) { \
            if (delta > step) return; \
        } else { \
            if (delta <= -step) return; \
            weight = 1024 - (int)(((-delta) << 10) / step); \
            if (weight <= 0) return; \
        } \
    } \
    for (int c = 0; c < channels; ++c) { \
        size_t k = pixel * channels + c; \
        size_t sk = BGR_REV ? (pixel * 3 + (2 - c)) : k; \
        int value = (int)img[sk] * scale; \
        if (weight == 1024) acc[k] = (ushort)value; \
        else acc[k] = (ushort)(((int)acc[k] * (1024 - weight) \
                                + value * weight) >> 10); \
    } \
}

DEFINE_MERGE(merge, ushort, long, 1, 0)
DEFINE_MERGE(merge_u8_i64, uchar, long, 1, 0)
DEFINE_MERGE(merge_u16_u16, ushort, ushort, 0, 0)
DEFINE_MERGE(merge_u8_u16, uchar, ushort, 0, 0)
DEFINE_MERGE(merge_bgr_rev, ushort, long, 1, 1)
DEFINE_MERGE(merge_u8_i64_bgr_rev, uchar, long, 1, 1)
DEFINE_MERGE(merge_u16_u16_bgr_rev, ushort, ushort, 0, 1)
DEFINE_MERGE(merge_u8_u16_bgr_rev, uchar, ushort, 0, 1)

__kernel void merge_out_u8(__global const ushort *acc, __global uchar *dst,
                           const ulong offset, const ulong count)
{
    size_t i = get_global_id(0);
    if (i < count) dst[i] = (uchar)(acc[offset + i] >> 8);
}
'''


def _device_rank(device, index):
    try:
        unified = bool(getattr(device, 'host_unified_memory', False))
    except Exception:
        unified = False
    try:
        mem = int(getattr(device, 'global_mem_size', 0))
    except Exception:
        mem = 0
    try:
        cus = int(getattr(device, 'max_compute_units', 0))
    except Exception:
        cus = 0
    return (0 if unified else 1, mem, cus, -index)


def _is_weak_integrated_gpu(device) -> bool:
    """Return True for low-CU unified-memory GPUs slower than multi-core native CPU."""
    try:
        unified = bool(getattr(device, 'host_unified_memory', False))
        cus = int(getattr(device, 'max_compute_units', 0) or 0)
    except Exception:
        return False
    if not unified or cus <= 0 or cus > 8 or (os.cpu_count() or 0) < 8:
        return False
    from core._native import native_available
    return native_available()


def _device(cl):
    candidates = []
    index = 0
    for platform in cl.get_platforms():
        try:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
        except cl.Error:
            continue
        for device in devices:
            if (device.type & cl.device_type.GPU and device.available
                    and device.compiler_available):
                if _is_weak_integrated_gpu(device):
                    continue
                candidates.append((index, device))
                index += 1
    if not candidates:
        return None
    return max(candidates, key=lambda item: _device_rank(item[1], item[0]))[1]


def opencl_device_name(device):
    try:
        name = str(device.name or '').strip()
    except Exception:
        # A missing descriptive field does not invalidate a compatible GPU.
        name = ''
    return f'OpenCL: {name}' if name else 'OpenCL'


def probe_device_name():
    """Return the OpenCL backend and model of the preferred compatible GPU."""
    try:
        import pyopencl as cl
        device = _device(cl)
        return opencl_device_name(device) if device is not None else ''
    except Exception:
        return ''


def gpu_available():
    return bool(probe_device_name())


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


def _shutdown_nvml_locked():
    global _nvml_state
    state = _nvml_state
    _nvml_state = None
    if state is not None:
        shutdown = state[0]
        try:
            shutdown()
        except Exception:
            pass


def _shutdown_nvml():
    with _nvml_lock:
        _shutdown_nvml_locked()


def _load_nvml_state_locked():
    global _nvml_state, _nvml_atexit_registered
    if _nvml_state is not None:
        return _nvml_state

    import ctypes
    import os
    import sys

    lib = None
    for candidate in (('nvml.dll',) if sys.platform == 'win32'
                      else ('libnvidia-ml.so.1', 'libnvidia-ml.so')):
        try:
            lib = ctypes.CDLL(candidate)
            break
        except OSError:
            continue
    if lib is None and sys.platform == 'win32':
        nvsmi = os.path.join(os.environ.get('ProgramFiles', r'C:\Program Files'),
                             'NVIDIA Corporation', 'NVSMI', 'nvml.dll')
        if os.path.isfile(nvsmi):
            try:
                lib = ctypes.CDLL(nvsmi)
            except OSError:
                lib = None
    if lib is None:
        return None

    class _MemInfo(ctypes.Structure):
        _fields_ = [('total', ctypes.c_uint64),
                    ('free', ctypes.c_uint64),
                    ('used', ctypes.c_uint64)]

    try:
        init = lib.nvmlInit_v2
        shutdown = lib.nvmlShutdown
        get_count = lib.nvmlDeviceGetCount_v2
        get_handle = lib.nvmlDeviceGetHandleByIndex_v2
        get_name = lib.nvmlDeviceGetName
        get_mem = lib.nvmlDeviceGetMemoryInfo
    except AttributeError:
        return None
    get_count.argtypes = (ctypes.POINTER(ctypes.c_uint),)
    get_handle.argtypes = (ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
    get_name.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint)
    get_mem.argtypes = (ctypes.c_void_p, ctypes.POINTER(_MemInfo))
    try:
        if init() != 0:
            return None
    except Exception:
        return None
    if not _nvml_atexit_registered:
        atexit.register(_shutdown_nvml)
        _nvml_atexit_registered = True
    _nvml_state = (shutdown, get_count, get_handle, get_name, get_mem, _MemInfo)
    return _nvml_state


def _nvml_free_vram(device):
    """Query free/total bytes via cached in-process NVML session."""
    import ctypes

    target_name = str(device.name).strip()
    target_total = int(device.global_mem_size)
    with _nvml_lock:
        for _ in range(2):
            state = _load_nvml_state_locked()
            if state is None:
                return None
            _, get_count, get_handle, get_name, get_mem, mem_info_cls = state
            try:
                count = ctypes.c_uint()
                if get_count(ctypes.byref(count)) != 0:
                    _shutdown_nvml_locked()
                    continue
                matches = []
                buf = ctypes.create_string_buffer(256)
                info = mem_info_cls()
                for idx in range(count.value):
                    handle = ctypes.c_void_p()
                    if (get_handle(idx, ctypes.byref(handle)) != 0
                            or get_name(handle, buf, len(buf)) != 0
                            or get_mem(handle, ctypes.byref(info)) != 0):
                        break
                    name = buf.value.decode('utf-8', errors='replace').strip()
                    total_bytes = int(info.total)
                    if name == target_name and abs(total_bytes - target_total) <= 64 * 1024 * 1024:
                        matches.append((int(info.free), total_bytes))
                else:
                    return matches[0] if len(matches) == 1 else None
                # A failed query may hide a second identical adapter.
                _shutdown_nvml_locked()
            except Exception:
                _shutdown_nvml_locked()
        return None


def _fp32_build_options(cl, device):
    """Pass correctly rounded FP32 divide/sqrt only when advertised by the device."""
    try:
        rounded = cl.device_fp_config.CORRECTLY_ROUNDED_DIVIDE_SQRT
        single_fp = device.single_fp_config
    except Exception:
        return []
    if (isinstance(rounded, int) and not isinstance(rounded, bool) and rounded > 0
            and isinstance(single_fp, int) and not isinstance(single_fp, bool)
            and single_fp >= 0 and (single_fp & rounded) == rounded):
        return ['-cl-fp32-correctly-rounded-divide-sqrt']
    return []


def _dxgi_free_vram(device):
    """Read free/total DXGI LOCAL budget for a unique matching physical adapter."""
    if sys.platform != 'win32':
        return None
    try:
        vid = int(getattr(device, 'vendor_id', 0) or 0)
    except (TypeError, ValueError):
        return None
    if vid <= 0:
        return None
    from core import resource_policy

    try:
        dev_mem = int(getattr(device, 'global_mem_size', 0) or 0)
    except (TypeError, ValueError):
        dev_mem = 0
    dev_name = str(getattr(device, 'name', '') or '').strip().lower()

    try:
        adapters = resource_policy._dxgi_adapter_budgets(vid)
    except Exception:
        return None

    identity = _opencl_dxgi_identity(device)
    if identity is not None:
        luid, node_mask = identity
        # The DXGI budget probe covers node 0 on the matched adapter.
        if node_mask != 1:
            return None
        return next((reading for adap_luid, _, _, reading in adapters if adap_luid == luid), None)

    seen_luids = set()
    exact_matches = []
    mem_matches = []
    for luid, name, dedicated, reading in adapters:
        if luid in seen_luids:
            continue
        seen_luids.add(luid)
        if dev_name and name.strip().lower() == dev_name:
            exact_matches.append(reading)
        elif dev_mem > 0 and dedicated > 0 and abs(dedicated - dev_mem) <= max(512 * 1024**2, dev_mem // 8):
            mem_matches.append(reading)
    matches = exact_matches or mem_matches
    return matches[0] if len(matches) == 1 else None


def _opencl_device_query(device):
    """Query extension fields through the ICD that owns a PyOpenCL device."""
    import ctypes

    try:
        import pyopencl as cl
    except ImportError:
        return None

    try:
        if not isinstance(device, cl.Device) or 'cl_khr_icd' not in device.platform.extensions.split():
            return None
        handle = device.int_ptr
        if not handle:
            return None
        # cl_khr_icd puts the dispatch table at the front of a device ID.
        # clGetDeviceInfo is the fourth member of cl_icd_dispatch.
        table = ctypes.cast(handle, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        convention = ctypes.WINFUNCTYPE if sys.platform == 'win32' else ctypes.CFUNCTYPE
        get_info = convention(ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t,
                              ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t))(table[3])
    except (cl.Error, AttributeError, TypeError, ValueError):
        return None

    def read_info(field, result):
        size = ctypes.sizeof(result)
        written = ctypes.c_size_t()
        return (get_info(handle, field, size, ctypes.byref(result), ctypes.byref(written)) == 0
                and written.value == size)

    return read_info


def _opencl_dxgi_identity(device):
    """Return the reported Windows adapter LUID and its device-node mask."""
    if 'cl_khr_device_uuid' not in getattr(device, 'extensions', '').split():
        return None
    query = _opencl_device_query(device)
    if query is None:
        return None
    import ctypes

    valid = ctypes.c_uint32()
    # CL_DEVICE_LUID_VALID_KHR, CL_DEVICE_LUID_KHR, CL_DEVICE_NODE_MASK_KHR.
    if not query(0x106C, valid) or not valid.value:
        return None
    luid = (ctypes.c_uint32 * 2)()
    if not query(0x106D, luid):
        return None
    node_mask = ctypes.c_uint32()
    return tuple(luid), node_mask.value if query(0x106E, node_mask) else 0


def _opencl_pci_address(device):
    """Return (domain, bus, slot, function), with domain unknown on AMD's older extension."""
    if not sys.platform.startswith('linux'):
        return None
    import ctypes

    try:
        extensions = device.extensions.split()
    except Exception:
        return None
    khr = 'cl_khr_pci_bus_info' in extensions
    amd = 'cl_amd_device_attribute_query' in extensions
    if not (khr or amd):
        return None

    query = _opencl_device_query(device)
    if query is None:
        return None

    if khr:
        address = (ctypes.c_uint32 * 4)()
        # CL_DEVICE_PCI_BUS_INFO_KHR
        if query(0x410F, address):
            domain, bus, slot, function = address
            if bus <= 0xff and slot <= 0x1f and function <= 7:
                return domain, bus, slot, function
    if amd:
        class AmdTopology(ctypes.Structure):
            _fields_ = [('type', ctypes.c_uint32), ('unused', ctypes.c_uint8 * 17),
                        ('bus', ctypes.c_uint8), ('slot', ctypes.c_uint8), ('function', ctypes.c_uint8)]

        topology = AmdTopology()
        # CL_DEVICE_TOPOLOGY_AMD and CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD
        if (query(0x4037, topology)
                and topology.type == 1 and topology.slot <= 0x1f and topology.function <= 7):
            return None, topology.bus, topology.slot, topology.function
    return None


def _drm_counter(device_path, name, base=10):
    try:
        return int((device_path / name).read_text(encoding='ascii'), base)
    except (OSError, ValueError):
        return None


def _linux_drm_free_vram(device, *, drm_root='/sys/class/drm'):
    """Read free/total bytes from the matching amdgpu adapter's sysfs VRAM counters."""
    try:
        if int(getattr(device, 'vendor_id', 0) or 0) != 0x1002:
            return None
    except (TypeError, ValueError):
        return None

    pci_address = _opencl_pci_address(device)
    if pci_address is None:
        try:
            dev_mem = int(getattr(device, 'global_mem_size', 0) or 0)
        except (TypeError, ValueError):
            return None
        if dev_mem <= 0:
            return None
        tol = max(256 * 1024**2, dev_mem // 8)
    else:
        domain, bus, slot, function = pci_address
        bdf = f':{bus:02x}:{slot:02x}.{function:x}'
        full_address = f'{domain:04x}{bdf}' if domain is not None else None
    try:
        entries = sorted(os.listdir(drm_root))
    except OSError:
        return None
    seen_devs = set()
    matches = []
    for name in entries:
        if not (name.startswith('card') and name[4:].isdigit()):
            continue
        dev_dir = Path(drm_root) / name / 'device'
        real_dev = Path(os.path.realpath(os.fspath(dev_dir)))
        if real_dev in seen_devs:
            continue
        seen_devs.add(real_dev)
        if pci_address is not None:
            if full_address is not None and real_dev.name != full_address:
                continue
            if full_address is None and not real_dev.name.endswith(bdf):
                continue
        vendor = _drm_counter(dev_dir, 'vendor', 16)
        if vendor is None and pci_address is not None:
            return None
        if vendor != 0x1002:
            continue
        total = _drm_counter(dev_dir, 'mem_info_vram_total')
        if (pci_address is None and total is not None and total > 0
                and abs(total - dev_mem) > tol):
            continue
        if matches:
            return None
        matches.append((dev_dir, total))
    if not matches:
        return None
    dev_dir, total = matches[0]
    if total is None or total <= 0:
        return None
    used = _drm_counter(dev_dir, 'mem_info_vram_used')
    if used is None or not 0 <= used <= total:
        return None
    return total - used, total


def _free_vram(device):
    """Read free memory from the selected OpenCL GPU's matching driver."""
    try:
        vid = int(getattr(device, 'vendor_id', 0) or 0)
    except (TypeError, ValueError):
        vid = 0
    if vid == 0x1002:
        try:
            if 'cl_amd_device_attribute_query' in getattr(device, 'extensions', '').split():
                free_kib = int(device.get_info(0x4039)[0])
                if 0 < free_kib * 1024 <= int(device.global_mem_size):
                    return free_kib * 1024, int(device.global_mem_size)
        except Exception:
            pass
    if vid == 0x10de:
        nvml = _nvml_free_vram(device)
        if nvml is not None:
            return nvml
    if sys.platform == 'win32':
        dxgi = _dxgi_free_vram(device)
        if dxgi is not None:
            return dxgi
    if vid == 0x10de:
        import subprocess

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
            if len(matches) == 1:
                return matches[0]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if sys.platform.startswith('linux'):
        return _linux_drm_free_vram(device)
    return None


class Merge:
    """A bounded merge workspace on the process's OpenCL context."""

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
            self.context, self.queue, self.program, kernels = _get_or_build_merge_program(
                cl, device,
            )
            self.transfer_queue = _get_transfer_queue(cl, device, self.context)
            self.device_name = opencl_device_name(device)
            self.activity_kernel = kernels['activity']
            self.kernel = kernels['merge']
            self.merge_u8_i64_kernel = kernels['merge_u8_i64']
            self.merge_u16_u16_kernel = kernels['merge_u16_u16']
            self.merge_u8_u16_kernel = kernels['merge_u8_u16']
            self.kernel_bgr_rev = kernels['merge_bgr_rev']
            self.merge_u8_i64_bgr_rev_kernel = kernels['merge_u8_i64_bgr_rev']
            self.merge_u16_u16_bgr_rev_kernel = kernels['merge_u16_u16_bgr_rev']
            self.merge_u8_u16_bgr_rev_kernel = kernels['merge_u8_u16_bgr_rev']
            self.merge_out_u8_kernel = kernels['merge_out_u8']
            self.depth_program = None
            self._merge_acc = None
            self._merge_depth = None
            self._merge_src = None
            self._merge_src_bytes = 0
            self._merge_src_alt = None
            self._merge_src_alt_bytes = 0
            self._merge_uploader = None
            self._merge_slot_events = [None, None]
            self._merge_slot_hosts = [None, None]
            self._merge_slot = 0
            self._merge_ping_pong = False
            self._merge_unified_bytes = 0
            self._merge_u16_depth = False
            self._merge_needs_drain = False
            self.rows = rows
            self.failed = False
            self.used = False
        except GpuError:
            raise
        except Exception as exc:
            raise GpuError('OpenCL GPU initialization failed') from exc

    def _drain_queue(self):
        ctx = getattr(self, 'context', None)
        _drain_command_queue(getattr(self, 'transfer_queue', None), ctx)
        _drain_command_queue(getattr(self, 'queue', None), ctx)

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
                   feather, should_cancel, progress_callback, *, live_preview=None,
                   preserve_detail: bool = True):
        """Run the PR depth stages only when all resident planes fit the job."""
        from core import resource_policy
        from core.multi_focus_fusion import memory_budget_bytes
        from core.source_frames import frame_has_mapped_pixels, frame_resolution_bytes
        height, width = map(int, shape[:2])
        if min(height, width) < 2:
            return None
        pixels = height * width
        oh, ow = height // 2, width // 2
        # Both axis plans retain 8-tap indices, weights, and counts on the device.
        axes_bytes = (height + width + oh + ow) * (8 * (8 + 4) + 8)
        snapshot = resource_policy.probe_resources()
        host_budget = min(int(memory_budget_bytes()),
                          int(resource_policy.ram_ceiling(snapshot)))
        source_peak = 0
        max_src_bytes = 0
        for frame in frames:
            resolution = frame_resolution_bytes(frame)
            mapped = frame_has_mapped_pixels(frame)
            if mapped:
                resolution = max(resolution, frame.nbytes)
            raw_native = img_resize is None and getattr(frame, 'dtype', None) in (np.uint8, np.uint16)
            if raw_native:
                if isinstance(frame, np.ndarray):
                    packed = (frame.flags.c_contiguous
                              or (frame.ndim == 3 and frame.shape[2] == 3
                                  and frame[..., ::-1].flags.c_contiguous))
                    working = frame.nbytes * (1 if packed else 2)
                else:
                    # Mapped descriptors may resolve to strided views requiring a copy.
                    working = frame.nbytes * (1 if getattr(frame, '_rgb_tiff_offset', None) is not None else 2)
            else:
                channels = 1 if frame.ndim == 2 else int(frame.shape[2])
                source_codes = int(frame.size) * 4
                target_codes = pixels * channels * 4
                working = max(target_codes + pixels * 4, pixels * 6)
                if img_resize is not None and frame.dtype.kind != 'f':
                    # The resize holds int32 codes, a float32 input copy, and float32 output.
                    working = max(working, 2 * source_codes + target_codes,
                                  source_codes + 2 * target_codes)
                else:
                    # Float conversion holds two arithmetic row temporaries.
                    resized = img_resize is not None
                    rows = min(128, height if resized else frame.shape[0])
                    row_size = (width if resized else frame.shape[1]) * channels
                    row_scratch = 2 * rows * row_size * max(4, frame.dtype.itemsize)
                    float_input = target_codes if resized else 0
                    working = max(working, target_codes + float_input + row_scratch)
                    if resized:
                        working = max(working, source_codes + target_codes)
            source_peak = max(source_peak, resolution + working)
            if raw_native:
                max_src_bytes = max(max_src_bytes, int(frame.nbytes))
        # bincount casts the half-resolution uint16 confidence plane to native ints.
        final_staging = max(pixels * 2 + oh * ow * 2, oh * ow * (2 + np.dtype(np.intp).itemsize))
        host_staging = max(source_peak, final_staging + axes_bytes) + 16 * 1024 * 1024
        # The guided path retains both filter workspaces through expansion.
        vram_need = pixels * (24 if preserve_detail else 17) + max_src_bytes + axes_bytes + 16 * 1024 * 1024
        largest_alloc = max(pixels * 8, max_src_bytes, max(height, width) * 8 * 8)
        if host_staging > host_budget or largest_alloc > int(self.device.max_mem_alloc_size):
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
                vram_cap = int(self.device.global_mem_size)
                host_staging += vram_need
                if host_staging > host_budget or vram_need > vram_cap:
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
                vram_cap = min(int(self.device.global_mem_size) // 2,
                               resource_policy.vram_budget(snapshot))
                if vram_need > vram_cap:
                    logging.getLogger(__name__).info('OpenFocus full GPU depth exceeds the free VRAM budget')
                    return None
            try:
                depth_prog, depth_k, filt_k, resamp_k, tiled = _get_or_build_depth_program(
                    self.cl, self.device, self.context,
                )
                self.depth_program = depth_prog
                prefetch_slots = _prefetch_window(
                    host_budget, host_staging, source_peak, len(frames),
                )
                extra_host = max(max_src_bytes, pixels * 2)
                extra_vram = max_src_bytes + pixels * 2
                unified_extra_vram = extra_vram if self.device.host_unified_memory else 0
                ping_pong = (
                    len(frames) >= 2
                    and (host_staging + prefetch_slots * source_peak + 2 * extra_host
                         + unified_extra_vram) <= host_budget
                    and (vram_need + extra_vram) <= vram_cap
                )
                engine = _GpuB(
                    self, img_resize, depth_kernels=depth_k, filter_kernels=filt_k,
                    resample_kernels=resamp_k, tiled_transpose=tiled,
                )
                engine._prefetch_slots = prefetch_slots
                engine._ping_pong = ping_pong
                engine._max_upload_bytes = extra_host
                engine._unified_memory = bool(self.device.host_unified_memory)
                depth_kwargs = {
                    'prefetch': prefetch_slots > 0,
                    'live_preview': live_preview,
                }
                if not preserve_detail:
                    depth_kwargs['preserve_detail'] = False
                return engine.depth(
                    frames, kernel_size, smoothing, feather, shape, len(frames),
                    should_cancel, progress, **depth_kwargs,
                )
            except CallbackFailure as exc:
                self._drain_queue()
                raise exc.__cause__
            except (self.cl.Error, MemoryError) as exc:
                _invalidate_program_cache(self.context)
                self._drain_queue()
                raise GpuError('OpenCL GPU depth failed') from exc
            except BaseException:
                self._drain_queue()
                raise
        finally:
            self.depth_program = None
            self.release()

    def activity(self, gray, fine, output):
        gb = fb = ob = None
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
            if hasattr(self, 'cl') and isinstance(exc, (self.cl.Error, MemoryError)):
                _invalidate_program_cache(getattr(self, 'context', None))
            self._drain_queue()
            raise GpuError('OpenCL GPU depth activity failed') from exc
        finally:
            _release_buffers(gb, fb, ob)
        output[...] = result

    def begin_merge(self, depth, shape, should_cancel):
        """Reserve a whole-image merge, or leave the bounded band path in use."""
        from core import resource_policy
        from core.multi_focus_fusion import memory_budget_bytes
        from dataclasses import replace
        height, width = shape[:2]
        channels = 1 if len(shape) == 2 else shape[2]
        pixels = int(height) * int(width)
        u16_depth = getattr(depth, 'dtype', None) == np.uint16
        depth_bytes = 2 if u16_depth else 8
        required = pixels * (depth_bytes + 4 * channels)
        host = min(int(memory_budget_bytes()),
                   int(resource_policy.ram_ceiling(resource_policy.probe_resources())))
        # Code conversion can retain int32 source samples beside their uint16 upload.
        host_staging = pixels * max(depth_bytes, 6 * channels) + 16 * 1024 * 1024
        if host_staging > host:
            return False
        if pixels * max(depth_bytes, 2 * channels) > int(self.device.max_mem_alloc_size):
            return False
        self.acquire(should_cancel)
        try:
            extra_src = pixels * channels * 2
            ping_pong = False
            if self.device.host_unified_memory:
                admitted = (required + host_staging <= host
                            and required <= int(self.device.global_mem_size))
                ping_pong = (
                    admitted
                    and (required + extra_src + host_staging + extra_src <= host)
                    and (required + extra_src <= int(self.device.global_mem_size))
                )
            else:
                memory = _free_vram(self.device)
                snapshot = (None if memory is None else
                            replace(resource_policy.probe_resources(),
                                    free_vram=memory[0], total_vram=memory[1]))
                vram_cap = (min(int(self.device.global_mem_size) // 2,
                                resource_policy.vram_budget(snapshot))
                            if snapshot is not None else 0)
                admitted = (snapshot is not None and required + 16 * 1024 * 1024 <= vram_cap)
                ping_pong = (
                    admitted
                    and (required + extra_src + 16 * 1024 * 1024 <= vram_cap)
                    and (host_staging + extra_src <= host)
                )
            if not admitted:
                self.release()
                return False
            cl = self.cl
            codes = np.ascontiguousarray(depth, dtype=np.uint16 if u16_depth else np.int64)
            self._merge_src = None
            self._merge_src_bytes = 0
            self._merge_src_alt = None
            self._merge_src_alt_bytes = 0
            self._merge_uploader = None
            self._merge_slot_events = [None, None]
            self._merge_slot_hosts = [None, None]
            self._merge_slot = 0
            self._merge_ping_pong = ping_pong
            # Unified memory allocates the upload buffers from host RAM after the merge
            # sizes its prefetch window, unlike the depth and accumulator buffers below.
            self._merge_unified_bytes = (extra_src * (2 if ping_pong else 1)
                                         if self.device.host_unified_memory else 0)
            self._merge_needs_drain = False
            self._merge_depth = cl.Buffer(self.context, cl.mem_flags.READ_ONLY |
                                          cl.mem_flags.COPY_HOST_PTR, hostbuf=codes)
            self._merge_acc = cl.Buffer(self.context, cl.mem_flags.READ_WRITE,
                                        pixels * channels * 2)
            self._merge_shape = shape
            self._merge_u16_depth = u16_depth
            return True
        except BaseException as exc:
            if hasattr(self, 'cl') and isinstance(exc, (self.cl.Error, MemoryError)):
                _invalidate_program_cache(getattr(self, 'context', None))
            self._drain_queue()
            uploader = getattr(self, '_merge_uploader', None)
            if uploader is not None:
                uploader.close()
                self._merge_uploader = None
            _release_buffers(self._merge_acc, self._merge_depth, self._merge_src,
                             getattr(self, '_merge_src_alt', None))
            self._merge_acc = self._merge_depth = self._merge_src = self._merge_src_alt = None
            self._merge_src_bytes = self._merge_src_alt_bytes = 0
            self._merge_slot_events = [None, None]
            self._merge_slot_hosts = [None, None]
            self._merge_slot = 0
            self._merge_ping_pong = False
            self._merge_unified_bytes = 0
            self._merge_u16_depth = False
            self._merge_needs_drain = False
            self.release()
            if not isinstance(exc, Exception):
                raise
            raise GpuError('OpenCL GPU merge allocation failed') from exc

    def _flush_merge_slots(self):
        uploader = getattr(self, '_merge_uploader', None)
        if uploader is not None:
            uploader.finish()
        events = getattr(self, '_merge_slot_events', None)
        hosts = getattr(self, '_merge_slot_hosts', None)
        if events is not None:
            for slot in (0, 1):
                if events[slot] is not None:
                    _wait_event(events[slot])
                    events[slot] = None
                if hosts is not None:
                    hosts[slot] = None

    def resident_frame(self, frame, index, count, scale):
        try:
            cl = self.cl
            is_u8 = getattr(frame, 'dtype', None) == np.uint8
            target_dtype = np.uint8 if is_u8 else np.uint16
            image, bgr_rev = _prepare_gpu_upload(frame, target_dtype)
            if self._merge_u16_depth:
                rev_attr = 'merge_u8_u16_bgr_rev_kernel' if is_u8 else 'merge_u16_u16_bgr_rev_kernel'
                norm_attr = 'merge_u8_u16_kernel' if is_u8 else 'merge_u16_u16_kernel'
            else:
                rev_attr = 'merge_u8_i64_bgr_rev_kernel' if is_u8 else 'kernel_bgr_rev'
                norm_attr = 'merge_u8_i64_kernel' if is_u8 else 'kernel'
            kern = getattr(self, rev_attr if bgr_rev else norm_attr)
            ping_pong = bool(getattr(self, '_merge_ping_pong', False))
            slot = self._merge_slot if ping_pong else 0
            prev_compute_ev = None
            if ping_pong:
                self._merge_slot ^= 1
                prev_compute_ev = self._merge_slot_events[slot]
                if prev_compute_ev is not None:
                    uploader_cur = getattr(self, '_merge_uploader', None)
                    if uploader_cur is None or not uploader_cur.separate_queue:
                        _wait_event(prev_compute_ev)
                        self._merge_slot_events[slot] = None
                        self._merge_slot_hosts[slot] = None
                        prev_compute_ev = None
            src_attr = '_merge_src' if slot == 0 else '_merge_src_alt'
            bytes_attr = '_merge_src_bytes' if slot == 0 else '_merge_src_alt_bytes'
            src_buf = getattr(self, src_attr, None)
            src_bytes = getattr(self, bytes_attr, 0)
            if src_buf is None or src_bytes != image.nbytes:
                if prev_compute_ev is not None:
                    _wait_event(prev_compute_ev)
                    self._merge_slot_events[slot] = None
                    self._merge_slot_hosts[slot] = None
                    prev_compute_ev = None
                _release_buffers(src_buf)
                setattr(self, src_attr, None)
                setattr(self, bytes_attr, 0)
                src_buf = cl.Buffer(
                    self.context, cl.mem_flags.READ_ONLY, max(1, image.nbytes),
                )
                setattr(self, src_attr, src_buf)
                setattr(self, bytes_attr, image.nbytes)
            uploader = getattr(self, '_merge_uploader', None)
            if ping_pong and uploader is None:
                uploader = _PinnedUploader(
                    cl, self.context, self.queue, getattr(self, 'transfer_queue', None),
                    2, image.nbytes,
                )
                self._merge_uploader = uploader
            self._merge_needs_drain = True
            if uploader is not None:
                upload = uploader.upload(
                    slot, src_buf, image, wait_for=_event_wait_list(prev_compute_ev),
                )
                wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
            else:
                upload = cl.enqueue_copy(self.queue, src_buf, image, is_blocking=False)
                wait_upload = None
            height, width = self._merge_shape[:2]
            channels = 1 if image.ndim == 2 else image.shape[2]
            event = _enqueue_kernel(
                kern, self.queue, (height * width,), None, self._merge_acc,
                src_buf, self._merge_depth, np.int32(width),
                np.int32(channels), np.int32(65535 // max(1, count - 1)),
                np.int32(index), np.int32(scale), np.int32(index == 0),
                wait_for=wait_upload,
            )
            if ping_pong and event is not None and hasattr(event, 'wait'):
                if hasattr(self.queue, 'flush'):
                    self.queue.flush()
                self._merge_slot_events[slot] = event
                self._merge_slot_hosts[slot] = (
                    (upload,) if (uploader is not None and uploader.pinned) else (image, upload)
                )
            else:
                event.wait()
                del upload
                self._merge_needs_drain = False
            self.used = True
        except Exception as exc:
            if hasattr(self, 'cl') and isinstance(exc, (self.cl.Error, MemoryError)):
                _invalidate_program_cache(getattr(self, 'context', None))
            self._drain_queue()
            if getattr(self, '_merge_slot_events', None) is not None:
                self._merge_slot_events = [None, None]
            if getattr(self, '_merge_slot_hosts', None) is not None:
                self._merge_slot_hosts = [None, None]
            self._merge_needs_drain = False
            raise GpuError('OpenCL GPU merge failed') from exc

    def finish_merge(self, result):
        try:
            self._flush_merge_slots()
            direct = (isinstance(result, np.ndarray)
                      and result.dtype == np.uint16
                      and result.shape == tuple(self._merge_shape)
                      and result.flags.c_contiguous
                      and result.flags.writeable)
            output = result if direct else np.empty(self._merge_shape, dtype=np.uint16)
            self.cl.enqueue_copy(self.queue, output, self._merge_acc, is_blocking=True)
            self._merge_needs_drain = False
        except Exception as exc:
            if hasattr(self, 'cl') and isinstance(exc, (self.cl.Error, MemoryError)):
                _invalidate_program_cache(getattr(self, 'context', None))
            self._drain_queue()
            self._merge_needs_drain = False
            raise GpuError('OpenCL GPU merge download failed') from exc
        if not direct:
            result[...] = output

    def finish_merge_u8(self, output):
        dst_buf = None
        try:
            self._flush_merge_slots()
            _release_buffers(self._merge_src, self._merge_src_alt)
            self._merge_src = self._merge_src_alt = None
            self._merge_src_bytes = self._merge_src_alt_bytes = 0
            direct = (isinstance(output, np.ndarray)
                      and output.dtype == np.uint8
                      and output.shape == tuple(self._merge_shape)
                      and output.flags.c_contiguous
                      and output.flags.writeable)
            dest = output if direct else np.empty(self._merge_shape, dtype=np.uint8)
            flat = dest.reshape(-1)
            total = int(flat.size)
            chunk_bytes = min(max(1, total), 8 * 1024 * 1024)
            dst_buf = self.cl.Buffer(self.context, self.cl.mem_flags.WRITE_ONLY, chunk_bytes)
            kern = self.merge_out_u8_kernel
            self._merge_needs_drain = True
            for start in range(0, total, chunk_bytes):
                region = flat[start:min(start + chunk_bytes, total)]
                kern(self.queue, (region.size,), None, self._merge_acc, dst_buf,
                     np.uint64(start), np.uint64(region.size))
                self.cl.enqueue_copy(self.queue, region, dst_buf, is_blocking=True)
            self._merge_needs_drain = False
        except Exception as exc:
            if hasattr(self, 'cl') and isinstance(exc, (self.cl.Error, MemoryError)):
                _invalidate_program_cache(getattr(self, 'context', None))
            self._drain_queue()
            self._merge_needs_drain = False
            raise GpuError('OpenCL GPU merge download failed') from exc
        finally:
            _release_buffers(dst_buf)
        if not direct:
            output[...] = dest

    def end_merge(self):
        if getattr(self, '_merge_needs_drain', False):
            self._drain_queue()
        uploader = getattr(self, '_merge_uploader', None)
        if uploader is not None:
            uploader.close()
            self._merge_uploader = None
        if getattr(self, '_merge_slot_events', None) is not None:
            self._merge_slot_events = [None, None]
        if getattr(self, '_merge_slot_hosts', None) is not None:
            self._merge_slot_hosts = [None, None]
        _release_buffers(self._merge_acc, self._merge_depth, self._merge_src,
                         getattr(self, '_merge_src_alt', None))
        self._merge_acc = self._merge_depth = self._merge_src = self._merge_src_alt = None
        self._merge_src_bytes = self._merge_src_alt_bytes = 0
        self._merge_slot = 0
        self._merge_ping_pong = False
        self._merge_unified_bytes = 0
        self._merge_u16_depth = False
        self._merge_needs_drain = False
        self.release()

    def band(self, accumulator, image, depth, index, step, scale, first):
        """Merge a band transactionally; failure leaves the accumulator intact."""
        ab = ib = db = None
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
            if hasattr(self, 'cl') and isinstance(exc, (self.cl.Error, MemoryError)):
                _invalidate_program_cache(getattr(self, 'context', None))
            self._drain_queue()
            raise GpuError('OpenCL GPU merge failed') from exc
        finally:
            _release_buffers(ab, ib, db)
        accumulator[...] = result


_FILTER_K_BODY = r'''
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
'''

_TRANSPOSE_TILED_K = r'''
__kernel void transpose_u16(__global const ushort* s, __global ushort* d,
                            const int W, const int H)
{
    __local ushort tile[16][17];
    int gx = get_group_id(0) * 16 + get_local_id(0);
    int gy = get_group_id(1) * 16 + get_local_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    if (gx < W && gy < H) tile[ly][lx] = s[(size_t)gy * W + gx];
    barrier(CLK_LOCAL_MEM_FENCE);
    int ox = get_group_id(1) * 16 + lx;
    int oy = get_group_id(0) * 16 + ly;
    if (ox < H && oy < W) d[(size_t)oy * H + ox] = tile[lx][ly];
}
'''

_TRANSPOSE_NAIVE_K = r'''
__kernel void transpose_u16(__global const ushort* s, __global ushort* d,
                            const int W, const int H)
{
    int x = get_global_id(0); if (x >= W) return;
    int y = get_global_id(1); if (y >= H) return;
    d[(size_t)x * H + y] = s[(size_t)y * W + x];
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
__kernel void detail_smooth_h(__global const ushort* src, __global uint* tmp,
                              const int H, const int W)
{
    int x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H)return;
    size_t row=(size_t)y*W;
    uint sum;
    if(x>=2 && x+2<W){
        sum=(uint)src[row+x-2]+(uint)src[row+x-1]+(uint)src[row+x]
           +(uint)src[row+x+1]+(uint)src[row+x+2];
    }else{
        sum=(uint)src[row+refl(x-2,W)]+(uint)src[row+refl(x-1,W)]+(uint)src[row+x]
           +(uint)src[row+refl(x+1,W)]+(uint)src[row+refl(x+2,W)];
    }
    tmp[row+x]=sum;
}
__kernel void detail_smooth_v(__global const uint* tmp, __global ushort* out,
                              const int H, const int W)
{
    int x=get_global_id(0), y=get_global_id(1);
    if(x>=W||y>=H)return;
    uint total;
    if(y>=2 && y+2<H){
        total=tmp[(size_t)(y-2)*W+x]+tmp[(size_t)(y-1)*W+x]+tmp[(size_t)y*W+x]
             +tmp[(size_t)(y+1)*W+x]+tmp[(size_t)(y+2)*W+x];
    }else{
        total=tmp[(size_t)refl(y-2,H)*W+x]+tmp[(size_t)refl(y-1,H)*W+x]+tmp[(size_t)y*W+x]
             +tmp[(size_t)refl(y+1,H)*W+x]+tmp[(size_t)refl(y+2,H)*W+x];
    }
    out[(size_t)y*W+x]=(ushort)(total/25u);
}
#define DEFINE_GRAY(NAME, SRC_T) \
__kernel void NAME(__global const SRC_T* src, __global ushort* out, \
                   const int H, const int W, const int C, const int scale) \
{ \
    int x = get_global_id(0), y = get_global_id(1); \
    if (x >= W || y >= H) return; \
    size_t p = (size_t)y * W + x; \
    if (C == 1) { \
        out[p] = (ushort)((ulong)src[p] * (ulong)scale); \
        return; \
    } \
    size_t base = p * C; \
    if (C >= 3) { \
        ulong total = (ulong)src[base] + (ulong)src[base + 1] + (ulong)src[base + 2]; \
        out[p] = (ushort)((total * (ulong)scale * 85ul) >> 8); \
    } else { \
        ulong total = (ulong)src[base] + (ulong)src[base + 1]; \
        out[p] = (ushort)((total / 3ul) * (ulong)scale); \
    } \
}
DEFINE_GRAY(gray_u8, uchar)
DEFINE_GRAY(gray_u16, ushort)
__kernel void init_detail(__global ushort* second, __global ushort* winner, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    second[i]=0; winner[i]=0;
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
__kernel void detail_confidence(__global const ushort* coarse, __global const ushort* best,
                                __global ushort* second, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    long top=(long)best[i];
    long gap=top-(long)second[i];
    second[i]=(ushort)(top>=128L && top>4L*((long)coarse[i]+1L) && 2L*gap>top);
}
__kernel void confidence(__global ushort* best, __global const ushort* votes, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    long diff=(long)best[i]-(long)votes[i];
    best[i]=(ushort)(diff<0L?-diff:diff);
}
__kernel void zero_u32(__global uint* buf, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    buf[i]=0u;
}
__kernel void confidence_hist(__global ushort* best, __global const ushort* votes,
                              __global uint* hist, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    int diff=(int)best[i]-(int)votes[i];
    uint c=(uint)(diff<0?-diff:diff);
    best[i]=(ushort)c;
    atomic_inc(&hist[c]);
}
__kernel void apply_detail(__global ushort* depth, __global const ushort* mask,
                           __global const ushort* winner, const int n)
{
    int i=get_global_id(0); if(i>=n)return;
    if(mask[i]) depth[i]=winner[i];
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

    def __init__(self, cl, ctx, q, mf, prg, kernels=None, *, tiled_transpose=True):
        self.cl = cl
        self.ctx = ctx
        self.q = q
        self.mf = mf
        self.p = prg
        self.kernels = (kernels if kernels is not None else
                        {name: cl.Kernel(self.p, name) for name in _FILTER_KERNEL_NAMES})
        self._bufs = {}
        self._tiled_transpose = tiled_transpose

    def _transpose(self, src_buf, dst_buf, w, h):
        if self._tiled_transpose:
            gw, gh = ((w + 15) // 16) * 16, ((h + 15) // 16) * 16
            self.kernels['transpose_u16'](self.q, (gw, gh), (16, 16), src_buf, dst_buf,
                                          np.int32(w), np.int32(h))
        else:
            self.kernels['transpose_u16'](self.q, (w, h), None, src_buf, dst_buf,
                                          np.int32(w), np.int32(h))

    def _bufs_for(self, h, w, guided):
        key = (h, w, guided)
        if key not in self._bufs:
            n = h * w
            if guided:
                self._bufs[key] = (
                    self.cl.Buffer(self.ctx, self.mf.READ_WRITE, n * 4),
                    self.cl.Buffer(self.ctx, self.mf.READ_WRITE, n * 2),
                    self.cl.Buffer(self.ctx, self.mf.READ_WRITE, n * 2),
                    self.cl.Buffer(self.ctx, self.mf.READ_WRITE, n * 2))
            else:
                self._bufs[key] = (
                    self.cl.Buffer(self.ctx, self.mf.READ_WRITE, n * 2),
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

    def run_buf(self, src_buf, out_buf, h, w, radius):
        from fusion_methods.openfocus_depth import _weight
        if radius == 0:
            if src_buf is not out_buf:
                self.cl.enqueue_copy(self.q, out_buf, src_buf)
            return
        q = int(_weight(radius))
        t = (4096 - q) >> 1
        count_h = max(1, min(int(radius), w))
        count_v = max(1, min(int(radius), h))
        ta, tb, first = self._bufs_for(h, w, False)
        # One work-item per column keeps every compute unit busy; each column
        # runs the same ordered recurrence, so results do not depend on width.
        self._transpose(src_buf, ta, w, h)
        self.kernels['filt'](self.q, (h,), None, ta, tb, first,
                             np.int32(w), np.int32(h), np.int32(count_h),
                             np.uint32(q), np.uint32(t))
        self._transpose(tb, ta, h, w)
        self.kernels['filt'](self.q, (w,), None, ta, out_buf, first,
                             np.int32(h), np.int32(w), np.int32(count_v),
                             np.uint32(q), np.uint32(t))

    def axis_guided_buf(self, src_buf, guide_buf, out_buf, h, w, main, low, axis=0):
        from fusion_methods.openfocus_depth import _weight
        if main == 0:
            if src_buf is not out_buf:
                self.cl.enqueue_copy(self.q, out_buf, src_buf)
            return
        first, ta, tb, tg = self._bufs_for(h, w, True)
        tm = (4096 - (int(_weight(main)) or 0)) >> 1
        tl = (4096 - int(_weight(low))) >> 1
        if axis == 1:
            self._transpose(src_buf, ta, w, h)
            self._transpose(guide_buf, tg, w, h)
            count = max(1, min(int(main), w))
            self.kernels['filt_guided'](self.q, (h,), None, ta, tg, tb, first,
                                        np.int32(w), np.int32(h), np.int32(count),
                                        np.uint32(tm), np.uint32(tl))
            self._transpose(tb, out_buf, h, w)
        else:
            count = max(1, min(int(main), h))
            self.kernels['filt_guided'](self.q, (w,), None, src_buf, guide_buf, out_buf, first,
                                        np.int32(h), np.int32(w), np.int32(count),
                                        np.uint32(tm), np.uint32(tl))

    def run(self, src, radius):
        h, w = src.shape
        sb = self._upload(src)
        ob = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, h * w * 2)
        self.run_buf(sb, ob, h, w, radius)
        return self._download(ob, (h, w))

    def axis_guided(self, src, guide, main, low, axis=0):
        h, w = src.shape
        sb = self._upload(src)
        gb = self._upload(guide)
        ob = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, h * w * 2)
        self.axis_guided_buf(sb, gb, ob, h, w, main, low, axis=axis)
        return self._download(ob, (h, w))


class _HalfResample:
    def __init__(self, cl, ctx, q, mf, prg, kernels=None):
        self.cl = cl
        self.ctx = ctx
        self.q = q
        self.mf = mf
        self.p = prg
        self.kernels = (kernels if kernels is not None else
                        {name: cl.Kernel(self.p, name) for name in _RESAMPLE_KERNEL_NAMES})
        self._cache = {}
        self._scratch = {}

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

    def _vb(self, OH, W):
        key = (OH, W)
        if key not in self._scratch:
            self._scratch[key] = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, OH * W * 4)
        return self._scratch[key]

    def run_buf(self, gray_buf, out_buf, H, W, *, wait_for=None):
        v, h, vix, vw, vc, hix, hw, hc = self._axes((H, W))
        OH, OW = v.indices.shape[0], h.indices.shape[0]
        vb = self._vb(OH, W)
        _enqueue_kernel(self.kernels['vert'], self.q, (W, OH), None, gray_buf, vb, vix, vw, vc,
                        np.int32(OH), np.int32(W), wait_for=wait_for)
        self.kernels['horiz'](self.q, (OW, OH), None, vb, out_buf, hix, hw, hc,
                              np.int32(OH), np.int32(OW), np.int32(W))
        return OH, OW

    def run(self, gray):
        H, W = gray.shape
        v, h, _, _, _, _, _, _ = self._axes((H, W))
        OH, OW = v.indices.shape[0], h.indices.shape[0]
        ib = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(gray))
        ob = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, OH * OW * 2)
        self.run_buf(ib, ob, H, W)
        self.q.finish()
        out = np.empty((OH, OW), np.uint16)
        self.cl.enqueue_copy(self.q, out, ob, is_blocking=True)
        return out


class _ExpandDepth(_HalfResample):
    def run_buf(self, depth_buf, src_shape, dst_shape):
        from core.frame_storage import create_transient_frame
        from fusion_methods._openfocus_resample import _axis_plan
        v = _axis_plan(src_shape[0], int(dst_shape[0]), linear=True)
        h = _axis_plan(src_shape[1], int(dst_shape[1]), linear=True)

        def pad(a, taps=8):
            b = np.zeros((a.shape[0], taps), a.dtype)
            b[:, :a.shape[1]] = a
            return b
        H, W = src_shape
        OH, OW = int(dst_shape[0]), int(dst_shape[1])
        vb = vix = vw = vc = ob = hix = hw = hc = None
        try:
            vb = self.cl.Buffer(self.ctx, self.mf.READ_WRITE, OH * W * 4)
            vix = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                 hostbuf=pad(v.indices))
            vw = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                hostbuf=pad(v.weights))
            vc = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                hostbuf=v.counts.astype(np.int64))
            self.kernels['vert'](self.q, (W, OH), None, depth_buf, vb, vix, vw, vc,
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
            out = create_transient_frame((OH, OW), np.uint16)
            self.cl.enqueue_copy(self.q, out, ob, is_blocking=True)
            return out
        finally:
            _release_buffers(vb, vix, vw, vc, ob, hix, hw, hc)

    def run(self, depth, shape):
        ib = self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(depth))
        try:
            return self.run_buf(ib, depth.shape, shape)
        finally:
            _release_buffers(ib)


class _GpuB:
    """GPU depth_map (half-res + preserve-detail) + MergeWeighted stream."""

    def __init__(self, gpu, img_resize, depth_kernels=None, filter_kernels=None,
                 resample_kernels=None, *, tiled_transpose=True):
        self.gpu = gpu
        cl, ctx, q, mf, pk = gpu.cl, gpu.context, gpu.queue, gpu.cl.mem_flags, gpu.depth_program
        self.img_resize = img_resize
        self.cl = cl
        self.ctx = ctx
        self.q = q
        self.mf = mf
        self.p = pk
        self.kernels = (depth_kernels if depth_kernels is not None else
                        {name: cl.Kernel(self.p, name) for name in _DEPTH_KERNEL_NAMES})
        self.filt = _Filter(
            cl, ctx, q, mf, pk, kernels=filter_kernels,
            tiled_transpose=tiled_transpose,
        )
        self.half = _HalfResample(cl, ctx, q, mf, pk, kernels=resample_kernels)
        self.expand = _ExpandDepth(cl, ctx, q, mf, pk, kernels=resample_kernels)
        self._cbuf = {}

    def _source_buffer(self, nbytes, slot=0, *, wait_for=None, cancel=None):
        # Ping-pong frames alternate across slots so the next host upload
        # never collides with an in-flight kernel reading the prior slot.
        key = f'src_in_{slot}'
        capacity, buf = self._cbuf.get(key, (0, None))
        if buf is None or nbytes > capacity:
            if buf is not None and wait_for:
                # The old allocation still counts against VRAM while a queued
                # kernel uses it. Admission budgets one source buffer per slot.
                for event in wait_for:
                    _wait_event(event, cancel)
            self._cbuf.pop(key, None)
            del buf
            capacity = max(1, nbytes)
            buf = self.cl.Buffer(self.ctx, self.mf.READ_ONLY, capacity)
            self._cbuf[key] = capacity, buf
        return buf

    # -- per-frame stages -------------------------------------------------
    def _prepare_gray_input(self, frame, cancelled, height, width):
        from fusion_methods._openfocus_pixels import frame_gray_codes
        from core.source_frames import resolve_frame, without_source_cache_growth
        with without_source_cache_growth():
            pixels = resolve_frame(frame, should_cancel=cancelled)
        if self.img_resize is None:
            if pixels.shape[:2] != (height, width):
                raise ValueError("OpenFocus depth requires frames with matching shapes")
            if pixels.dtype in (np.uint8, np.uint16):
                channels = 1 if pixels.ndim == 2 else int(pixels.shape[2])
                is_u8 = pixels.dtype == np.uint8
                source, _ = _prepare_gpu_upload(pixels)
                del pixels
                return True, source, channels, is_u8
        gray = frame_gray_codes(pixels, self.img_resize, cancelled, depth_color=True)
        del pixels
        if gray.shape != (height, width):
            raise ValueError("OpenFocus depth requires frames with matching shapes")
        host_gray = np.ascontiguousarray(gray, dtype=np.uint16)
        return False, host_gray, 1, False

    def _gray_into(self, prepared, gray_buf, height, width, slot=0,
                   *, uploader=None, wait_for=None, cancel=None):
        is_raw, host_buf, channels, is_u8 = prepared
        if is_raw:
            ib = self._source_buffer(host_buf.nbytes, slot=slot, wait_for=wait_for, cancel=cancel)
            if uploader is not None:
                upload = uploader.upload(slot, ib, host_buf, wait_for=wait_for, cancel=cancel)
                wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
            else:
                upload = self.cl.enqueue_copy(self.q, ib, host_buf, is_blocking=False)
                wait_upload = None
            kern = self.kernels['gray_u8' if is_u8 else 'gray_u16']
            _enqueue_kernel(kern, self.q, (width, height), None, ib, gray_buf,
                            np.int32(height), np.int32(width), np.int32(channels),
                            np.int32(256 if is_u8 else 1), wait_for=wait_upload)
            self.gpu.used = True
            return upload, None
        if uploader is not None:
            upload = uploader.upload(slot, gray_buf, host_buf, wait_for=wait_for, cancel=cancel)
            wait_upload = _event_wait_list(upload) if uploader.separate_queue else None
            return upload, wait_upload
        return self.cl.enqueue_copy(self.q, gray_buf, host_buf, is_blocking=False), None

    # -- depth_map equivalent ---------------------------------------------
    def depth(self, frames, kernel_size, smoothing, feather, shape, count,
              should_cancel, progress_callback=None, *, prefetch=False, live_preview=None,
              preserve_detail: bool = True):
        prefetch_slots = getattr(self, '_prefetch_slots', 1 if prefetch else 0)
        ping_pong = bool(getattr(self, '_ping_pong', False))
        from fusion_methods.openfocus_depth import _snap_lut
        from fusion_methods.openfocus_ab import _check_cancel

        def cancel():
            _check_cancel(should_cancel)

        height, width = int(shape[0]), int(shape[1])
        oh, ow = height // 2, width // 2
        m = oh * ow
        m_bytes = m * 2

        radius = max(1, int(kernel_size or 17) // 2)
        parameter = float(np.float64(radius) * np.float64(np.float32(0.075)))
        fine_p = np.float32(parameter * 0.5)
        coarse_p = np.float32(parameter * 2.0)
        extent = np.float32(max(oh, ow))
        fine_r = float(np.float32(extent * fine_p / np.float32(100)))
        coarse_r = float(np.float32(extent * coarse_p / np.float32(100)))
        smooth = np.float32(np.float32(smoothing) * np.float32(0.75) * np.float32(0.03125))
        scale = float(np.float32(extent * smooth / np.float32(100)))
        low = float(np.float32(extent * np.float32(smooth * np.float32(64)) / np.float32(100)))
        step = 65535 // (count - 1)

        completed = False
        prepared_iter = None
        reader = None
        uploader = None
        allocated = []
        try:
            cancel()
            def _alloc(buf):
                allocated.append(buf)
                return buf

            gray_bufs = tuple(
                _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, height * width * 2))
                for _ in range(2 if ping_pong else 1)
            )
            pb = (_alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))
                  if (preserve_detail or scale != 0.0) else None)
            sb = (_alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))
                  if (preserve_detail or scale != 0.0) else None)
            if preserve_detail:
                d_best_buf = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))
                d_second_buf = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))
                d_winner_buf = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))
            else:
                d_best_buf = d_second_buf = d_winner_buf = None
            half_buf = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))
            fine_buf = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))
            act_raw_buf = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))
            act_buf = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))
            best_buf = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))
            depth_buf = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))
            votes_buf = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, m_bytes))

            max_upload = int(getattr(self, '_max_upload_bytes', height * width * 2))
            if ping_pong:
                uploader = _PinnedUploader(
                    self.cl, self.ctx, self.q, getattr(self.gpu, 'transfer_queue', None),
                    2, max_upload,
                )

            if live_preview is not None:
                reader = _PreviewPlaneReader(self.cl, self.ctx, self.q,
                                             self.kernels['sample_focus_plane'], (oh, ow), np.uint16)

            def prepare(_index, frame, cancelled):
                return self._prepare_gray_input(frame, cancelled, height, width)

            effective_slots = max(1, int(prefetch_slots)) if prefetch else 0
            prepared_iter = _iter_prefetched(
                frames, prepare, enabled=prefetch, should_cancel=should_cancel,
                max_in_flight=max(1, effective_slots),
            )
            detail_tmp_buf = self.half._vb(oh, width) if preserve_detail else None
            use_sep_smooth = (
                preserve_detail
                and 'detail_smooth_h' in self.kernels
                and 'detail_smooth_v' in self.kernels
            )
            slot_events = [None, None]
            slot_hosts = [None, None]
            for idx, prepared in prepared_iter:
                slot = idx & 1 if ping_pong else 0
                prev_slot_ev = slot_events[slot]
                if prev_slot_ev is not None and (uploader is None or not uploader.separate_queue):
                    _wait_event(prev_slot_ev, cancel)
                    slot_events[slot] = None
                    slot_hosts[slot] = None
                    prev_slot_ev = None
                cancel()
                gray_buf = gray_bufs[slot]
                upload, wait_upload = self._gray_into(
                    prepared, gray_buf, height, width, slot=slot,
                    uploader=uploader, wait_for=_event_wait_list(prev_slot_ev), cancel=cancel,
                )
                cancel()
                if preserve_detail:
                    _enqueue_kernel(
                        self.kernels['detail_pool'], self.q, (ow, oh), None, gray_buf, pb,
                        np.int32(height), np.int32(width), np.int32(oh), np.int32(ow),
                        wait_for=wait_upload,
                    )
                    wait_upload = None
                    if use_sep_smooth:
                        self.kernels['detail_smooth_h'](self.q, (ow, oh), None, pb, detail_tmp_buf,
                                                        np.int32(oh), np.int32(ow))
                        self.kernels['detail_smooth_v'](self.q, (ow, oh), None, detail_tmp_buf, sb,
                                                        np.int32(oh), np.int32(ow))
                    else:
                        self.kernels['detail_smooth'](self.q, (ow, oh), None, pb, sb,
                                                      np.int32(oh), np.int32(ow))
                self.gpu.used = True
                cancel()
                self.half.run_buf(gray_buf, half_buf, height, width, wait_for=wait_upload)
                cancel()
                self.filt.run_buf(half_buf, fine_buf, oh, ow, fine_r)
                cancel()
                self.kernels['activity'](self.q, (m,), None, half_buf, fine_buf,
                                         act_raw_buf, np.int32(m))
                self.filt.run_buf(act_raw_buf, act_buf, oh, ow, coarse_r)
                code = idx * step
                if idx == 0:
                    self.cl.enqueue_copy(self.q, best_buf, act_buf)
                    last_event = self.kernels['init_activity'](self.q, (m,), None, act_buf, depth_buf,
                                                               votes_buf, np.uint32(count), np.int32(m))
                    if preserve_detail:
                        self.cl.enqueue_copy(self.q, d_best_buf, sb)
                        last_event = self.kernels['init_detail'](
                            self.q, (m,), None, d_second_buf, d_winner_buf, np.int32(m),
                        )
                else:
                    last_event = self.kernels['select_activity'](self.q, (m,), None, act_buf, best_buf,
                                                                 depth_buf, votes_buf, np.uint16(code),
                                                                 np.uint32(count), np.int32(m))
                    if preserve_detail:
                        last_event = self.kernels['detail_select'](
                            self.q, (m,), None, sb, d_best_buf, d_second_buf, d_winner_buf,
                            np.uint16(code), np.int32(m),
                        )
                if ping_pong and last_event is not None and hasattr(last_event, 'wait'):
                    if hasattr(self.q, 'flush'):
                        self.q.flush()
                    slot_events[slot] = last_event
                    slot_hosts[slot] = (
                        (upload,) if (uploader is not None and uploader.pinned) else (prepared, upload)
                    )
                    if reader is not None and live_preview.ready():
                        _wait_event(last_event, cancel)
                        slot_events[slot] = None
                        slot_hosts[slot] = None
                        sample = reader.read(depth_buf, live_preview)
                        if sample is not None:
                            live_preview.show_positions(sample, idx, count, step * (count - 1),
                                                        sampled=True)
                else:
                    self.q.finish()
                    if reader is not None and live_preview.ready():
                        cancel()
                        sample = reader.read(depth_buf, live_preview)
                        if sample is not None:
                            live_preview.show_positions(sample, idx, count, step * (count - 1),
                                                        sampled=True)
                del upload, prepared
                if progress_callback is not None:
                    progress_callback(idx, count)

            if ping_pong:
                if uploader is not None:
                    uploader.finish(cancel)
                for slot in (0, 1):
                    if slot_events[slot] is not None:
                        _wait_event(slot_events[slot], cancel)
                        slot_events[slot] = None
                    slot_hosts[slot] = None
                self.q.finish()
            if uploader is not None:
                uploader.close()
                uploader = None

            cancel()
            if preserve_detail:
                self.kernels['detail_confidence'](self.q, (m,), None, best_buf, d_best_buf,
                                                  d_second_buf, np.int32(m))
            if scale != 0.0:
                if 'zero_u32' in self.kernels and 'confidence_hist' in self.kernels:
                    hist_buf = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_WRITE, 65536 * 4))
                    self.kernels['zero_u32'](self.q, (65536,), None, hist_buf, np.int32(65536))
                    self.kernels['confidence_hist'](self.q, (m,), None, best_buf, votes_buf,
                                                    hist_buf, np.int32(m))
                    hist_u32 = np.empty(65536, dtype=np.uint32)
                    self.cl.enqueue_copy(self.q, hist_u32, hist_buf, is_blocking=True)
                    hist = hist_u32.astype(np.int64, copy=False)
                else:
                    self.kernels['confidence'](self.q, (m,), None, best_buf, votes_buf, np.int32(m))
                    best = np.empty((oh, ow), dtype=np.uint16)
                    self.cl.enqueue_copy(self.q, best, best_buf, is_blocking=True)
                    hist = np.bincount(best.ravel(), minlength=65536)
                threshold = int(np.float32(np.float32(m)
                                           * np.float32(0.009999999776482582)))
                lo = int(np.searchsorted(np.cumsum(hist), threshold)) if threshold else 0
                hi = (65535 - int(np.searchsorted(np.cumsum(hist[::-1]), threshold))
                      if threshold else 65535)
                positions = np.arange(4096, dtype=np.float64) * 65535.0 / 4095.0
                span = max(hi - lo, 1)
                lut = (65535.0 * np.sqrt(np.clip((positions - lo) / span, 0, 1))
                       ).astype(np.float32)

                lb = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                           hostbuf=lut))
                guide_buf = pb
                self.kernels['guide'](self.q, (m,), None, best_buf, lb, guide_buf, np.int32(m))
                tmp_depth_buf = sb
                for _ in range(2):
                    cancel()
                    self.filt.axis_guided_buf(depth_buf, guide_buf, tmp_depth_buf, oh, ow,
                                              scale, low, axis=1)
                    cancel()
                    self.filt.axis_guided_buf(tmp_depth_buf, guide_buf, depth_buf, oh, ow,
                                              scale, low, axis=0)
            half_step = step >> 1
            if half_step:
                sl = _snap_lut(feather).astype(np.int64)
                sb_lut = _alloc(self.cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                               hostbuf=sl))
                self.kernels['snap'](self.q, (m,), None, depth_buf, sb_lut,
                                     np.int32(step), np.int32(m))
            if preserve_detail:
                self.kernels['apply_detail'](self.q, (m,), None, depth_buf, d_second_buf,
                                             d_winner_buf, np.int32(m))
            cancel()
            out = self.expand.run_buf(depth_buf, (oh, ow), shape)
            completed = True
            return out
        finally:
            if not completed:
                self.gpu._drain_queue()
            if uploader is not None:
                uploader.close()
            if prepared_iter is not None:
                prepared_iter.close()
            if reader is not None:
                reader.close()
            _release_buffers(
                allocated,
                self._cbuf,
                self.filt._bufs,
                self.half._cache,
                self.half._scratch,
                self.expand._cache,
                self.expand._scratch,
            )
            self._cbuf.clear()
            self.filt._bufs.clear()
            self.half._cache.clear()
            self.half._scratch.clear()
            self.expand._cache.clear()
            self.expand._scratch.clear()

