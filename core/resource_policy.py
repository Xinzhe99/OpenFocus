"""One execution plan for every heavy pipeline stage.

The fusion orchestrator, the tile workers, the AI adapter and the alignment
passes each used to decide their own concurrency and memory limits, from
different numbers (a 45% free-RAM share here, a hard-coded 80 "image units"
there, an assumed core count from ``logical // 2``). Those decisions could
disagree: the planner shrank a tile while the per-tile call started its own
thread pool, or the neural path accepted a batch the conv buffers could not
hold.

This module is the single place that turns a *resource snapshot* and a
*workload* into an immutable *execution plan*: how many outer workers, how
many native threads each may use, how many results may be in flight, and how
much RAM/VRAM the stage may allocate on top of what is already resident.

Deliberately boring: plain dataclasses, no plugin registry, no per-algorithm
policy classes. Ratios here are conservative starting points to be calibrated
by the soak harness, not measured constants.
"""
import math
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from constants import FRAME_RAM_BUDGET_BYTES
from utils import memory
from utils.logging_setup import debug_enabled, log_debug
from utils.platform_utils import cpu_limits
from core.algorithms import get_fusion
from core import torch_backend

# Share of currently-available physical memory one stage may plan to allocate.
BUDGET_RATIO = 0.45
# Never plan to leave the machine with less than this for the desktop.
RESERVE_BYTES = 1024 ** 3
# The proportional reserve (15% of total RAM) is capped so large machines do
# not set aside ~10 GB for the desktop and starve every stage.
RESERVE_RATIO = 0.15
RESERVE_CAP_BYTES = 4 * 1024 ** 3

# GPU reserves: the allocator keeps freed blocks, so free VRAM is what matters.
VRAM_RESERVE_RATIO = 0.15
VRAM_RESERVE_BYTES = 512 * 1024 ** 2
VRAM_BUDGET_RATIO = 0.70

PREFERENCE_BALANCED = "balanced"
PREFERENCE_RESPONSIVE = "responsive"
PREFERENCE_MAXIMUM = "maximum"
PREFERENCES = (PREFERENCE_BALANCED, PREFERENCE_RESPONSIVE, PREFERENCE_MAXIMUM)


def raw_sensor_shape(sizes):
    """Return the validated, aspect-expanded sensor height and width."""
    aspect = float(sizes.pixel_aspect)
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError('Invalid RAW pixel aspect ratio')
    width = math.ceil(max(sizes.raw_width, sizes.width) * max(1, aspect, 1 / aspect))
    height = max(sizes.raw_height, sizes.height)
    if width <= 0 or height <= 0:
        raise ValueError('Invalid RAW dimensions')
    return int(height), int(width)


def raw_decode_bytes(sensor_width, sensor_height, output_nbytes=0, encoded_bytes=0):
    """Conservative full-sensor native workspace plus output and encoded input."""
    return (int(sensor_width) * int(sensor_height) * 64
            + int(output_nbytes) + int(encoded_bytes) * 2)


@dataclass(frozen=True)
class ResourceSnapshot:
    """What the machine can give right now."""

    available_ram: int = 0
    total_ram: int = 0
    process_rss: int = 0
    physical_cores: int = 1
    logical_cpus: int = 1
    free_vram: int = 0
    total_vram: int = 0
    device_kind: str = "CPU"
    cpu_limit: float = 0
    physical_cores_known: bool = False

    @property
    def ram_known(self) -> bool:
        return self.available_ram > 0 and self.total_ram > 0

    @property
    def vram_known(self) -> bool:
        return self.free_vram > 0 and self.total_vram > 0


@dataclass(frozen=True)
class Workload:
    """The shape of the work, independent of the machine running it."""

    algorithm: str = ""
    frames: int = 1
    height: int = 0
    width: int = 0
    channels: int = 3
    itemsize: int = 1
    block_size: Optional[int] = None
    tile_overlap: int = 0
    # Newly allocated bytes only: resident inputs belong to the snapshot.
    fixed_bytes: Optional[int] = None
    per_worker_bytes: Optional[int] = None
    fixed_vram_bytes: int = 0
    per_worker_vram_bytes: int = 0
    outer_tasks: Optional[int] = None
    parallel_layer: str = "outer"


@dataclass(frozen=True)
class ExecutionPlan:
    """Decisions for one stage. Frozen: a running job never changes shape."""

    workers: int = 1
    native_threads: int = 1
    max_in_flight: int = 1
    ram_budget: int = 0
    vram_budget: int = 0
    device_kind: str = "CPU"
    reasons: tuple = field(default_factory=tuple)
    tile_size: Optional[int] = None

    def describe(self) -> str:
        return (f"workers={self.workers}, threads={self.native_threads}, "
                f"in-flight={self.max_in_flight}, "
                f"ram={memory.mb(self.ram_budget):.0f}MB, "
                f"vram={memory.mb(self.vram_budget):.0f}MB, "
                f"device={self.device_kind}")


@dataclass(frozen=True)
class ProcessBudget:
    """Envelope for one isolated child job, frozen when the job is admitted.

    Values are newly allocated bytes, like the snapshot budgets they come
    from. Zero means unknown or exhausted, never unlimited. ``frame_ram`` is
    the slice of the job-wide managed frame RAM this process may hold, so a
    child never re-grants the part its parent is still using.
    """

    comfortable_ram: int = 0
    maximum_ram: int = 0
    frame_ram: int = 0

    def __post_init__(self):
        if min(self.comfortable_ram, self.maximum_ram, self.frame_ram) < 0:
            raise ValueError("Process budget limits must be nonnegative")
        if self.comfortable_ram > self.maximum_ram:
            raise ValueError("comfortable_ram must not exceed maximum_ram")


_DIRECTML_ADAPTER_INDEX = 0


def set_directml_adapter(index):
    """Use the same DXGI adapter for DirectML execution and its budget."""
    global _DIRECTML_ADAPTER_INDEX
    _DIRECTML_ADAPTER_INDEX = max(0, int(index))


def directml_adapters():
    """Enumerate hardware DXGI adapters, preferring discrete GPU memory."""
    if sys.platform != 'win32':
        return []
    import ctypes
    import uuid

    class AdapterDesc(ctypes.Structure):
        _fields_ = [('name', ctypes.c_wchar * 128), ('vendor', ctypes.c_uint32),
                    ('device', ctypes.c_uint32), ('subsystem', ctypes.c_uint32),
                    ('revision', ctypes.c_uint32), ('dedicated', ctypes.c_size_t),
                    ('system', ctypes.c_size_t), ('shared', ctypes.c_size_t),
                    ('luid', ctypes.c_uint64)]

    def method(obj, index, restype, *argtypes):
        table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[index])

    factory = ctypes.c_void_p()
    found = []
    try:
        create = ctypes.WinDLL('dxgi').CreateDXGIFactory1
        create.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
        create.restype = ctypes.c_int32
        iid = (ctypes.c_ubyte * 16).from_buffer_copy(
            uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
        if create(ctypes.byref(iid), ctypes.byref(factory)) < 0 or not factory.value:
            return []
        enum = method(factory, 7, ctypes.c_int32, ctypes.c_uint32,
                      ctypes.POINTER(ctypes.c_void_p))
        for index in range(64):
            adapter = ctypes.c_void_p()
            if enum(factory, index, ctypes.byref(adapter)) < 0 or not adapter.value:
                break
            try:
                desc = AdapterDesc()
                get_desc = method(adapter, 8, ctypes.c_int32, ctypes.POINTER(AdapterDesc))
                if get_desc(adapter, ctypes.byref(desc)) >= 0 and desc.vendor != 0x1414:
                    found.append((index, desc.name, int(desc.vendor), int(desc.dedicated)))
            finally:
                method(adapter, 2, ctypes.c_uint32)(adapter)
    except (OSError, AttributeError, ValueError):
        return []
    finally:
        if factory.value:
            method(factory, 2, ctypes.c_uint32)(factory)
    return sorted(found, key=lambda item: (item[3] > 512 * 1024 ** 2,
                                          item[2] in (0x10de, 0x1002, 0x8086),
                                          item[3], -item[0]), reverse=True)


def _directml_memory() -> tuple:
    """Free and total LOCAL budget for the selected ORT DirectML adapter.

    DXGI's budget and usage are process-specific and change over time. LOCAL
    also describes the GPU-accessible pool on unified-memory adapters; host
    admission remains necessary in both cases. Failure is an unknown budget.
    """
    if sys.platform != 'win32':
        return 0, 0
    import ctypes
    import uuid

    class VideoMemoryInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ('budget', 'usage', 'available_reservation', 'reservation')]

    def guid(value):
        return (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le)

    def method(obj, index, restype, *argtypes):
        table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[index])

    factory = ctypes.c_void_p()
    adapter = ctypes.c_void_p()
    adapter3 = ctypes.c_void_p()
    try:
        dxgi = ctypes.WinDLL('dxgi')
        create = dxgi.CreateDXGIFactory1
        create.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
        create.restype = ctypes.c_int32
        factory_id = guid('770aae78-f26f-4dba-a829-253c83d1b387')
        if create(ctypes.byref(factory_id), ctypes.byref(factory)) < 0 or not factory.value:
            return 0, 0
        enumerate_adapter = method(factory, 7, ctypes.c_int32, ctypes.c_uint32,
                                   ctypes.POINTER(ctypes.c_void_p))
        if enumerate_adapter(factory, _DIRECTML_ADAPTER_INDEX, ctypes.byref(adapter)) < 0 or not adapter.value:
            return 0, 0
        query_interface = method(adapter, 0, ctypes.c_int32, ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_void_p))
        adapter3_id = guid('645967a4-1392-4310-a798-8053ce3e93fd')
        if query_interface(adapter, ctypes.byref(adapter3_id), ctypes.byref(adapter3)) < 0 or not adapter3.value:
            return 0, 0
        query_memory = method(adapter3, 14, ctypes.c_int32, ctypes.c_uint32,
                              ctypes.c_int32, ctypes.POINTER(VideoMemoryInfo))
        info = VideoMemoryInfo()
        # Node 0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL.
        if query_memory(adapter3, 0, 0, ctypes.byref(info)) < 0 or not info.budget:
            return 0, 0
        return max(0, int(info.budget) - int(info.usage)), int(info.budget)
    except (OSError, AttributeError, ValueError):
        return 0, 0
    finally:
        for obj in (adapter3, adapter, factory):
            if obj.value:
                method(obj, 2, ctypes.c_uint32)(obj)


def _gpu_memory(kind: str) -> tuple:
    """Return free and total bytes without importing a full ML framework."""
    reading = torch_backend.device_memory(kind)
    if reading is not None:
        return reading
    if kind not in ('CUDA', 'ROCM'):
        return 0, 0
    import ctypes

    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    try:
        if kind == 'ROCM':
            library = ctypes.CDLL('amdhip64.dll' if sys.platform == 'win32' else 'libamdhip64.so')
            previous = ctypes.c_int()
            if library.hipInit(0) or library.hipGetDevice(ctypes.byref(previous)):
                return 0, 0
            if library.hipSetDevice(0):
                return 0, 0
            try:
                if library.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)):
                    return 0, 0
            finally:
                library.hipSetDevice(previous.value)
        else:
            library = ctypes.CDLL('nvcuda.dll' if sys.platform == 'win32' else 'libcuda.so.1')
            device, context = ctypes.c_int(), ctypes.c_void_p()
            if library.cuInit(0) or library.cuDeviceGet(ctypes.byref(device), 0):
                return 0, 0
            retain = library.cuDevicePrimaryCtxRetain
            retain.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_int)
            if retain(ctypes.byref(context), device.value):
                return 0, 0
            try:
                push = library.cuCtxPushCurrent_v2
                push.argtypes = (ctypes.c_void_p,)
                if push(context):
                    return 0, 0
                try:
                    if library.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)):
                        return 0, 0
                finally:
                    popped = ctypes.c_void_p()
                    library.cuCtxPopCurrent_v2(ctypes.byref(popped))
            finally:
                library.cuDevicePrimaryCtxRelease_v2(device.value)
        return int(free.value), int(total.value)
    except (OSError, AttributeError, ValueError):
        return 0, 0


def probe_memory() -> ResourceSnapshot:
    """Read current RAM for admission without CPU topology or GPU queries."""
    available, total = memory.system_memory_bytes()
    return ResourceSnapshot(
        available_ram=int(available or 0),
        total_ram=int(total or 0),
        process_rss=int(memory.process_rss_bytes() or 0),
    )


def probe_resources(device_kind: Optional[str] = None) -> ResourceSnapshot:
    """Read the machine once. Every field may be zero when unknowable."""
    cores, logical, limit, topology_known = cpu_limits()

    ram = probe_memory()
    kind = str(device_kind or "CPU").upper()
    free_vram = total_vram = 0
    if kind == 'DIRECTML':
        free_vram, total_vram = _directml_memory()
    elif kind in ('CUDA', 'ROCM', 'XPU', 'OPENVINO'):
        free_vram, total_vram = _gpu_memory(kind)
    return replace(
        ram,
        physical_cores=max(1, int(cores)),
        logical_cpus=max(1, int(logical)),
        free_vram=free_vram,
        total_vram=total_vram,
        device_kind=kind,
        cpu_limit=limit,
        physical_cores_known=topology_known,
    )


def ram_budget(snapshot: ResourceSnapshot) -> int:
    """Bytes a stage may allocate *on top of* what is already resident.

    The source stack is already counted in ``available_ram``'s complement, so
    it is not subtracted a second time here; what this bounds is the new
    allocation the stage is about to make. An active process budget can only
    lower this share, never raise it.
    """
    return _apply_comfortable(_detected_ram_budget(snapshot))


def _detected_ram_budget(snapshot: ResourceSnapshot) -> int:
    """Comfortable share the machine offers, before any child-job envelope."""
    if not snapshot.ram_known:
        return 0
    return max(0, int(min(BUDGET_RATIO * snapshot.available_ram,
                          _detected_ram_ceiling(snapshot))))


def vram_budget(snapshot: ResourceSnapshot) -> int:
    """Bytes of new VRAM allocation allowed, or 0 when there is no VRAM."""
    if not snapshot.vram_known:
        return 0
    reserve = max(VRAM_RESERVE_BYTES,
                  int(VRAM_RESERVE_RATIO * snapshot.total_vram))
    headroom = snapshot.free_vram - reserve
    return max(0, int(min(VRAM_BUDGET_RATIO * snapshot.free_vram, headroom)))


def cpu_capacity(snapshot: ResourceSnapshot, preference: str) -> int:
    """Usable capacity, retaining at least one spare core in balanced mode."""
    cores = min(max(1, snapshot.physical_cores), max(1, snapshot.logical_cpus))
    if snapshot.cpu_limit > 0:
        cores = min(cores, max(1, math.floor(snapshot.cpu_limit)))
    if preference == PREFERENCE_MAXIMUM:
        return cores
    if preference == PREFERENCE_RESPONSIVE:
        return max(1, math.floor(cores * 0.5))
    return max(1, min(cores - 1, math.floor(cores * 0.8)))


def native_thread_ceiling(preference=PREFERENCE_BALANCED, user_threads=0):
    """CPU allowance for a serial owner independent of outer task count."""
    capacity = cpu_capacity(probe_resources(), preference)
    return min(capacity, int(user_threads)) if user_threads and int(user_threads) > 0 else capacity


def cpu_facts(preference: str = PREFERENCE_BALANCED) -> dict:
    """CPU topology and what automatic threads resolve to, for diagnostics."""
    snapshot = probe_resources()
    return {'physical_cores': snapshot.physical_cores, 'logical_cpus': snapshot.logical_cpus,
            'cpu_limit': snapshot.cpu_limit,
            'physical_cores_known': snapshot.physical_cores_known,
            'automatic_threads': {name: cpu_capacity(snapshot, name) for name in PREFERENCES},
            'resource_preference': preference}


def log_render_resources(phase: str, preference: str = PREFERENCE_BALANCED) -> None:
    """Record process memory around a render, and the CPU facts at its start.

    Detailed logging only, and never a reason for a render to fail.
    """
    if not debug_enabled():
        return
    try:
        log_debug('render_resources', phase=phase, rss_bytes=memory.process_rss_bytes(),
                  peak_rss_bytes=memory.peak_rss_bytes(),
                  available_ram_bytes=memory.available_memory_bytes(),
                  total_ram_bytes=memory.total_memory_bytes(),
                  **(cpu_facts(preference) if phase == 'start' else {}))
    except Exception:
        log_debug('render_resources', phase=phase, unavailable=True)


class ResourceLimitError(MemoryError):
    """The minimum working set cannot safely be admitted."""


# One isolated child job owns its process, so every producer thread reads the
# same envelope from here instead of carrying a per-thread copy.
_ACTIVE_PROCESS_BUDGET: Optional[ProcessBudget] = None


def _active_process_budget() -> Optional[ProcessBudget]:
    """Envelope currently in force, if this process is a child job.

    A single reference to an immutable object, so a producer thread observes
    either the previous or the new envelope, never a partly applied one.
    """
    return _ACTIVE_PROCESS_BUDGET


def _apply_comfortable(limit: int) -> int:
    active = _active_process_budget()
    if active is None:
        return limit
    return min(limit, max(0, int(active.comfortable_ram)))


def _apply_maximum(limit: int) -> int:
    active = _active_process_budget()
    if active is None:
        return limit
    return min(limit, max(0, int(active.maximum_ram)))


def ram_ceiling(snapshot: ResourceSnapshot) -> int:
    """Most new RAM one stage may take: everything above the desktop reserve.

    ``ram_budget`` is the comfortable share used to size concurrency; this is
    the hard limit for admitting a single minimum working set that does not
    fit in that share. An active process budget can only lower it.
    """
    return _apply_maximum(_detected_ram_ceiling(snapshot))


def _detected_ram_ceiling(snapshot: ResourceSnapshot) -> int:
    """Reserve-protected headroom the machine offers, before any envelope."""
    if not snapshot.ram_known:
        return 0
    reserve = max(RESERVE_BYTES,
                  min(int(RESERVE_RATIO * snapshot.total_ram), RESERVE_CAP_BYTES))
    return max(0, int(snapshot.available_ram - reserve))


def require_memory(required_bytes: int, snapshot: Optional[ResourceSnapshot] = None,
                   user_limit: Optional[int] = None) -> int:
    """Admit a known new allocation; return the RAM budget the stage may use.

    Within the comfortable share the whole share is returned, so callers can
    add workers. A larger minimum working set is still admitted up to the
    reserve-protected ceiling, but then the budget is exactly that working
    set, leaving no room for extra concurrency. Both limits, and any user
    limit, are clamped again by an active process budget.
    """
    if required_bytes < 0:
        raise ValueError("required_bytes must be nonnegative")
    snapshot = snapshot or probe_resources()
    budget = ram_budget(snapshot)
    ceiling = ram_ceiling(snapshot)
    if user_limit is not None:
        budget = min(budget, max(0, int(user_limit)))
        ceiling = min(ceiling, max(0, int(user_limit)))
    if not snapshot.ram_known:
        raise ResourceLimitError("Available physical memory could not be determined")
    if budget > 0 and required_bytes <= budget:
        return budget
    if ceiling > 0 and required_bytes <= ceiling:
        return max(1, int(required_bytes))
    raise ResourceLimitError(
        f"Insufficient memory: need {memory.mb(required_bytes):.0f} MB "
        f"of new allocations; at most {memory.mb(ceiling):.0f} MB is available")


def make_process_budget(parent_frame_bytes: int = 0,
                        snapshot: Optional[ResourceSnapshot] = None) -> ProcessBudget:
    """Freeze the envelope for one isolated child job before it starts work.

    ``snapshot`` is the admission reading, possibly taken before this process
    existed. Whatever this process has loaded since is already missing from a
    fresh reading, so the current one may only tighten the snapshot: a child
    never gains headroom its parent did not admit against. ``parent_frame_bytes``
    is the managed frame RAM the parent still holds; the job-wide frame cap is
    shared, so the child's slice is only the remainder.
    """
    if parent_frame_bytes < 0:
        raise ValueError("parent_frame_bytes must be nonnegative")
    if snapshot is None:
        snapshot = probe_memory()
    else:
        snapshot = _account_consumed_headroom(snapshot)
    comfortable = _detected_ram_budget(snapshot)
    maximum = _detected_ram_ceiling(snapshot)
    frame = (min(FRAME_RAM_BUDGET_BYTES, comfortable // 2)
             if snapshot.ram_known else 0)
    return ProcessBudget(comfortable_ram=comfortable, maximum_ram=maximum,
                         frame_ram=max(0, frame - int(parent_frame_bytes)))


def _account_consumed_headroom(snapshot: ResourceSnapshot) -> ResourceSnapshot:
    """Lower an admission snapshot by the headroom this process already used.

    ``available_ram`` excludes everything already resident, so a current
    reading covers this process's own RSS growth (imports, decoders) without
    subtracting RSS a second time. It is a tightening pass only: a machine
    that freed memory meanwhile does not widen the admitted envelope.
    """
    if not snapshot.ram_known:
        return snapshot
    current = probe_memory()
    if not current.ram_known or current.available_ram >= snapshot.available_ram:
        return snapshot
    return replace(snapshot, available_ram=int(current.available_ram))


@contextmanager
def process_budget(budget: Optional[ProcessBudget] = None):
    """Apply one child job's envelope to every thread in this process.

    Entered by the job's owner thread before producers start; producer threads
    read the active envelope directly, so no per-thread plumbing is needed.
    ``None`` leaves the ordinary policy in place. Exit restores the previous
    envelope, which is the ordinary policy for a job-scoped context.
    """
    global _ACTIVE_PROCESS_BUDGET
    if budget is None:
        yield None
        return
    previous = _ACTIVE_PROCESS_BUDGET
    _ACTIVE_PROCESS_BUDGET = budget
    try:
        yield budget
    finally:
        _ACTIVE_PROCESS_BUDGET = previous


def frame_ram_budget(snapshot: ResourceSnapshot) -> int:
    """Managed frame RAM this process may hold in RAM right now.

    Snapshots and numerical scratch share ``FRAME_RAM_BUDGET_BYTES`` and at
    most half the comfortable budget; an active child job's slice is the
    tighter bound. Unknown memory keeps the disk-mapping fallback (zero).
    """
    if not snapshot.ram_known:
        return 0
    cap = min(FRAME_RAM_BUDGET_BYTES, max(0, ram_budget(snapshot)) // 2)
    active = _active_process_budget()
    if active is not None:
        cap = min(cap, max(0, int(active.frame_ram)))
    return max(0, int(cap))


def estimate_working_set(workload: Workload) -> tuple:
    """Conservative fixed and per-worker allocation estimates for CPU stages."""
    pixels = max(0, workload.height) * max(0, workload.width)
    channels = max(1, workload.channels)
    frames = max(1, workload.frames)
    frame_float = pixels * channels * np.dtype(np.float32).itemsize
    if workload.algorithm in ('openfocus_b', 'openfocus_c'):
        # The depth estimator retains several int64 maps, filter intermediates,
        # and histogram/LUT scratch. C additionally holds pyramid reconstructions.
        scratch = 192 + channels * (64 if workload.algorithm == 'openfocus_c' else 16)
        if workload.block_size:
            tile_pixels = int(workload.block_size) ** 2
            fixed = pixels * ((channels + 1) * 4 + 4 + channels * 4)
            per_worker = tile_pixels * (scratch + frames * channels * (max(1, workload.itemsize) + 4))
        else:
            fixed = pixels * (scratch + frames * channels * 4) + 4 * 1024 * 1024
            per_worker = max(1, pixels * channels * 4)
    elif workload.block_size:
        fixed = pixels * (channels + 1) * 4
        per_worker = math.ceil(frames * int(workload.block_size) ** 2 * channels
                               * (max(1, workload.itemsize) + 4) * 5.6)
        # A completed tile and its weight remain live until ordered reduction.
        per_worker += int(workload.block_size) ** 2 * (channels + 1) * 4
    elif workload.algorithm == "registration":
        fixed = frames * pixels * channels * max(1, workload.itemsize)
        per_worker = frame_float * 6
    else:
        fixed = (frames + 1) * frame_float
        per_worker = frame_float * 6
    if workload.fixed_bytes is not None:
        fixed = int(workload.fixed_bytes)
    if workload.per_worker_bytes is not None:
        per_worker = int(workload.per_worker_bytes)
    if fixed < 0 or per_worker < 0:
        raise ValueError("Working-set estimates must be nonnegative")
    if fixed + per_worker <= 0:
        raise ResourceLimitError("The stage's minimum working set is unknown")
    return fixed, per_worker


def plan_execution(workload: Workload, snapshot: Optional[ResourceSnapshot] = None,
                   preference: str = PREFERENCE_BALANCED,
                   user_threads: Optional[int] = None,
                   user_batch: Optional[int] = None,
                   ram_budget_override: Optional[int] = None) -> ExecutionPlan:
    """Admit fixed buffers plus concurrent workers without changing geometry.

    Explicit footprints replace the default CPU estimates. Native stages such
    as AI must provide their own estimates and set ``parallel_layer='native'``.
    Thread and batch values are upper bounds; None/zero selects automatic.
    A RAM override is an additional cap, never permission to exceed headroom.
    """
    if workload.parallel_layer not in ("outer", "native"):
        raise ValueError("parallel_layer must be 'outer' or 'native'")
    try:
        requires_estimate = get_fusion(workload.algorithm).neural
    except ValueError:
        requires_estimate = False
    if requires_estimate:
        if workload.fixed_bytes is None or workload.per_worker_bytes is None:
            raise ResourceLimitError("AI requires an explicit working-set estimate")
    snapshot = snapshot or probe_resources()
    capacity = cpu_capacity(snapshot, preference)
    ceiling = (min(capacity, int(user_threads))
               if user_threads is not None and user_threads > 0 else capacity)
    fixed, per_worker = estimate_working_set(workload)
    ram = require_memory(fixed + per_worker, snapshot, ram_budget_override)
    tasks = workload.outer_tasks
    if tasks is None:
        tasks = ceiling if workload.block_size else max(1, workload.frames)
    workers = min(ceiling, max(1, int(tasks)))
    if workload.parallel_layer == "native":
        workers = 1
    if user_batch is not None and user_batch > 0:
        workers = min(workers, int(user_batch))
    reasons = []
    if per_worker:
        by_memory = (ram - fixed) // per_worker
        if by_memory < workers:
            workers = by_memory
            reasons.append(f"memory: {workers} worker(s) fit after fixed buffers")

    vram = vram_budget(snapshot) if torch_backend.has_device_memory(
        snapshot.device_kind) else 0
    gpu_fixed = workload.fixed_vram_bytes
    gpu_worker = workload.per_worker_vram_bytes
    if gpu_fixed < 0 or gpu_worker < 0:
        raise ValueError("VRAM estimates must be nonnegative")
    if torch_backend.has_device_memory(snapshot.device_kind):
        if gpu_fixed + gpu_worker <= 0:
            raise ResourceLimitError(
                f"{snapshot.device_kind} requires an explicit working-set estimate")
        if not fits_in_budget(gpu_fixed + gpu_worker, vram):
            raise ResourceLimitError("Insufficient or unknown available GPU memory")
        if gpu_worker:
            workers = min(workers, (vram - gpu_fixed) // gpu_worker)
        # GPU inference has one owner and one stream; no parallel submissions.
        workers = 1
    native = ceiling if workers == 1 else 1
    plan = ExecutionPlan(workers=workers, native_threads=native,
                         max_in_flight=workers, ram_budget=ram, vram_budget=vram,
                         device_kind=snapshot.device_kind, reasons=tuple(reasons))
    log_debug('execution_plan', algorithm=workload.algorithm, frames=workload.frames,
              shape=(workload.height, workload.width, workload.channels),
              block_size=workload.block_size, layer=workload.parallel_layer, tasks=tasks,
              fixed_bytes=fixed, per_worker_bytes=per_worker,
              available_ram=snapshot.available_ram, ram_budget=ram, vram_budget=vram,
              preference=preference, user_threads=user_threads, cpu_capacity=capacity,
              workers=plan.workers, native_threads=plan.native_threads,
              device=plan.device_kind, reasons=plan.reasons)
    return plan


def plan_neural_tiles(frames: int, height: int, width: int, channels: int,
                      block_size: int, device_kind: str, estimate_inference,
                      *, directory: bool = False, itemsize: int = 1,
                      source_resolution_bytes: int = 0, source_cache_bytes: int = 0,
                      snapshot: Optional[ResourceSnapshot] = None,
                      preference: str = PREFERENCE_BALANCED,
                      user_threads: Optional[int] = None):
    """Choose a spatial tile before allocating crops or output buffers.

    CoreML allocations share host RAM. DirectML additionally requires a
    trustworthy free-VRAM reading; without one, select CPU instead.
    The tile estimate includes crop copies, staging, inference and reduction.
    """
    if source_resolution_bytes < 0 or source_cache_bytes < 0:
        raise ValueError('Source workspace sizes must be nonnegative')
    snapshot = snapshot or probe_resources(device_kind)
    maximum = max(1, min(int(block_size), height, width))
    minimum = min(maximum, 64)
    pixels = height * width
    fixed = pixels * ((channels + 1) * 4 + channels * 4)
    fixed += source_cache_bytes
    if source_resolution_bytes:
        fixed += source_resolution_bytes
    elif directory:
        # Legacy directories lack descriptor geometry/decoder information.
        fixed += pixels * channels * max(4, itemsize) * 2
    ceiling = ram_ceiling(snapshot)
    if not snapshot.ram_known:
        raise ResourceLimitError('Available physical memory could not be determined')
    kinds = [str(device_kind).upper()]
    if kinds[0] != 'CPU':
        kinds.append('CPU')
    for kind in kinds:
        # Dedicated devices need the selected provider's real free VRAM.
        dedicated = kind in ('DIRECTML', 'CUDA', 'ROCM', 'OPENVINO')
        if dedicated and (snapshot.device_kind.upper() != kind or not snapshot.vram_known):
            continue
        if kind not in ('CPU', 'COREML', 'DIRECTML', 'CUDA', 'ROCM', 'OPENVINO'):
            continue
        block = maximum
        while True:
            padded = ((block + 63) // 64) * 64
            tile_pixels = padded * padded
            # Matched/normalized crops may coexist with their predecessors.
            crops = frames * block * block * channels * 8
            staging = frames * tile_pixels * 4 * (2 * channels + 3)
            reduction = tile_pixels * (channels * 4 * 3 + 8)
            inference = estimate_inference(frames, block, block, kind)
            required = fixed + crops + staging + reduction + inference
            device_required = inference + staging + reduction
            device_fits = (not dedicated or
                           fits_in_budget(device_required, vram_budget(snapshot)))
            if required <= ceiling and ceiling > 0 and device_fits:
                capacity = cpu_capacity(snapshot, preference)
                threads = (min(capacity, int(user_threads))
                           if user_threads is not None and user_threads > 0 else capacity)
                # Why a tile is smaller than requested: the first size whose
                # working set fits the ceiling, halving from the image clamp.
                log_debug('neural_tile_plan', frames=frames, shape=(height, width, channels),
                          block_requested=int(block_size), block_after_image_clamp=maximum,
                          block=block, device=kind, required_bytes=required,
                          fixed_bytes=fixed, inference_bytes=inference, ram_ceiling=ceiling,
                          available_ram=snapshot.available_ram, preference=preference,
                          user_threads=user_threads, cpu_capacity=capacity,
                          native_threads=max(1, threads))
                return block, ExecutionPlan(
                    workers=1, native_threads=max(1, threads), max_in_flight=1,
                    ram_budget=ceiling,
                    vram_budget=vram_budget(snapshot) if dedicated else 0,
                    device_kind=kind,
                    reasons=(f'neural tile {block}px, {kind}',))
            if block <= minimum:
                break
            block = max(minimum, block // 2)
    raise ResourceLimitError(
        f'Insufficient memory for neural fusion: minimum {minimum}px tile '
        f'and output require more than {memory.mb(ceiling):.0f} MB available')


def fits_in_budget(required_bytes: int, budget: int) -> bool:
    """True when a stage's minimum working set can be allocated at all.

    Callers use this to refuse work *before* allocating, with a message the
    user can act on, instead of being killed by the OS mid-fusion.
    """
    return required_bytes >= 0 and (required_bytes == 0 or
                                    budget > 0 and required_bytes <= budget)


_HEAVY_JOB_LOCK = threading.RLock()


@contextmanager
def heavy_job(should_cancel=None):
    """Serialize heavy job lifetimes, allowing same-thread nested stages.

    Waiting is cooperative. Cancellation never revokes a running owner's
    lease: callers must finish/join native work before leaving this context.
    Nested pool workers must use their owner's plan, not acquire this lease.
    """
    def check_cancel():
        if should_cancel is not None and should_cancel():
            from core.registration import RegistrationCancelled
            raise RegistrationCancelled()

    waited = 0
    while True:
        check_cancel()
        if _HEAVY_JOB_LOCK.acquire(timeout=0.05):
            break
        waited += 1
    if waited:
        # Another heavy job held the lease; the wait is not processing time.
        log_debug('heavy_job.waited', waited_ms=waited * 50)
    try:
        check_cancel()
        yield
    finally:
        _HEAVY_JOB_LOCK.release()
