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
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils import memory
from utils.platform_utils import cpu_limits

# Share of currently-available physical memory one stage may plan to allocate.
BUDGET_RATIO = 0.45
# Never plan to leave the machine with less than this for the desktop.
RESERVE_BYTES = 1024 ** 3

# GPU reserves: the allocator keeps freed blocks, so free VRAM is what matters.
VRAM_RESERVE_RATIO = 0.15
VRAM_RESERVE_BYTES = 512 * 1024 ** 2
VRAM_BUDGET_RATIO = 0.70

PREFERENCE_BALANCED = "balanced"
PREFERENCE_RESPONSIVE = "responsive"
PREFERENCE_MAXIMUM = "maximum"
PREFERENCES = (PREFERENCE_BALANCED, PREFERENCE_RESPONSIVE, PREFERENCE_MAXIMUM)


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

    def describe(self) -> str:
        return (f"workers={self.workers}, threads={self.native_threads}, "
                f"in-flight={self.max_in_flight}, "
                f"ram={memory.mb(self.ram_budget):.0f}MB, "
                f"vram={memory.mb(self.vram_budget):.0f}MB, "
                f"device={self.device_kind}")


def _gpu_memory() -> tuple:
    """(free, total) VRAM bytes for the active accelerator, else (0, 0).

    Never *imports* torch: this runs on the UI thread and in the CLI, where a
    multi-second import just to size a budget would be worse than reporting no
    VRAM. If torch is already loaded (the render path loaded it), its allocator
    knows the real numbers; otherwise there is no usable accelerator anyway.
    """
    import sys
    torch = sys.modules.get("torch")
    if torch is None:
        return 0, 0
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return int(free), int(total)
    except Exception:
        return 0, 0
    # MPS shares physical memory; there is no separate VRAM pool to budget.
    return 0, 0


def probe_resources(device_kind: Optional[str] = None) -> ResourceSnapshot:
    """Read the machine once. Every field may be zero when unknowable."""
    cores, logical, limit, topology_known = cpu_limits()

    available = memory.available_memory_bytes()
    total = memory.total_memory_bytes()
    rss = memory.process_rss_bytes()
    kind = device_kind or "CPU"
    free_vram = total_vram = 0
    if kind == "CUDA":
        free_vram, total_vram = _gpu_memory()
    return ResourceSnapshot(
        available_ram=int(available or 0),
        total_ram=int(total or 0),
        process_rss=int(rss or 0),
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
    allocation the stage is about to make.
    """
    if not snapshot.ram_known:
        return 0
    reserve = max(RESERVE_BYTES, int(0.15 * snapshot.total_ram))
    headroom = snapshot.available_ram - reserve
    return max(0, int(min(BUDGET_RATIO * snapshot.available_ram, headroom)))


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


class ResourceLimitError(MemoryError):
    """The minimum working set cannot safely be admitted."""


def require_memory(required_bytes: int, snapshot: Optional[ResourceSnapshot] = None,
                   user_limit: Optional[int] = None) -> int:
    """Admit a known new allocation; return its capped available RAM budget."""
    if required_bytes < 0:
        raise ValueError("required_bytes must be nonnegative")
    snapshot = snapshot or probe_resources()
    budget = ram_budget(snapshot)
    if user_limit is not None:
        budget = min(budget, max(0, int(user_limit)))
    if not snapshot.ram_known:
        raise ResourceLimitError("Available physical memory could not be determined")
    if budget <= 0 or required_bytes > budget:
        raise ResourceLimitError(
            f"Insufficient memory: need {memory.mb(required_bytes):.0f} MB "
            f"of new allocations; budget is {memory.mb(budget):.0f} MB")
    return budget


def estimate_working_set(workload: Workload) -> tuple:
    """Conservative fixed and per-worker allocation estimates for CPU stages."""
    pixels = max(0, workload.height) * max(0, workload.width)
    channels = max(1, workload.channels)
    frames = max(1, workload.frames)
    frame_float = pixels * channels * np.dtype(np.float32).itemsize
    if workload.block_size:
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
    if workload.algorithm in ("ai", "stackmff", "stackmff_v3", "stackmffv3"):
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

    vram = vram_budget(snapshot) if snapshot.device_kind == "CUDA" else 0
    gpu_fixed = workload.fixed_vram_bytes
    gpu_worker = workload.per_worker_vram_bytes
    if gpu_fixed < 0 or gpu_worker < 0:
        raise ValueError("VRAM estimates must be nonnegative")
    if snapshot.device_kind == "CUDA":
        if gpu_fixed + gpu_worker <= 0:
            raise ResourceLimitError("CUDA requires an explicit working-set estimate")
        if not fits_in_budget(gpu_fixed + gpu_worker, vram):
            raise ResourceLimitError("Insufficient or unknown available GPU memory")
        if gpu_worker:
            workers = min(workers, (vram - gpu_fixed) // gpu_worker)
        # GPU inference has one owner and one stream; no parallel submissions.
        workers = 1
    native = ceiling if workers == 1 else 1
    return ExecutionPlan(workers=workers, native_threads=native,
                         max_in_flight=workers, ram_budget=ram, vram_budget=vram,
                         device_kind=snapshot.device_kind, reasons=tuple(reasons))


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

    while True:
        check_cancel()
        if _HEAVY_JOB_LOCK.acquire(timeout=0.05):
            break
    try:
        check_cancel()
        yield
    finally:
        _HEAVY_JOB_LOCK.release()
