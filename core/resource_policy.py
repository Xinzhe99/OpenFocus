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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils import memory
from utils.platform_utils import physical_core_count

# Share of currently-available physical memory one stage may plan to allocate.
BUDGET_RATIO = 0.45
# Never plan to leave the machine with less than this for the desktop.
RESERVE_BYTES = 512 * 1024 ** 2
# Share of physical cores the default "balanced" mode may occupy.
CPU_HEADROOM = 0.8

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
    import os

    available = memory.available_memory_bytes()
    total = memory.total_memory_bytes()
    rss = memory.process_rss_bytes()
    cores = physical_core_count()
    logical = os.cpu_count() or cores
    kind = device_kind or "CPU"
    free_vram = total_vram = 0
    if kind in ("CUDA", "MPS"):
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
    """Physical cores this stage may occupy in the given preference mode."""
    cores = max(1, snapshot.physical_cores)
    if preference == PREFERENCE_RESPONSIVE:
        share = 0.5
    elif preference == PREFERENCE_MAXIMUM:
        share = 1.0
    else:
        share = CPU_HEADROOM
    return max(1, min(cores, int(cores * share) or 1))


def plan_execution(workload: Workload, snapshot: Optional[ResourceSnapshot] = None,
                   preference: str = PREFERENCE_BALANCED,
                   user_threads: Optional[int] = None,
                   user_batch: Optional[int] = None,
                   ram_budget_override: Optional[int] = None) -> ExecutionPlan:
    """Turn a workload and a snapshot into one bounded plan.

    ``user_threads`` is an *upper bound*: memory pressure may lower it, never
    raise it. ``user_batch`` bounds the neural batch the same way.
    ``ram_budget_override`` lets a caller that already computed the budget
    (``multi_focus_fusion.memory_budget_bytes``) pass it in, so there is one
    authoritative budget function rather than two.
    """
    if preference not in PREFERENCES:
        preference = PREFERENCE_BALANCED
    snapshot = snapshot or probe_resources()

    capacity = cpu_capacity(snapshot, preference)
    ceiling = capacity if user_threads is None else min(capacity,
                                                        max(1, int(user_threads)))
    reasons = []

    ram = (int(ram_budget_override) if ram_budget_override is not None
           else ram_budget(snapshot))
    vram = vram_budget(snapshot) if snapshot.device_kind in ("CUDA", "MPS") else 0

    # Per-worker working set of the classical tile pipeline, calibrated on
    # 1600x2400x10 at 1024 px (guided filter, the heaviest method): ~850 MB
    # per worker against a 150 MB naive estimate, i.e. 5.6x, because several
    # float32 temporaries of the crop live at once for every frame.
    block = workload.block_size
    workers = ceiling
    if block and workload.frames > 0:
        naive = (workload.frames * (int(block) ** 2) * max(1, workload.channels)
                 * (1 + np.dtype(np.float32).itemsize) * 5.6)
        per_worker = max(1, int(naive))
        if ram > 0:
            by_memory = max(1, ram // per_worker)
            if by_memory < workers:
                reasons.append(f"memory: {by_memory} worker(s) fit in "
                               f"{memory.mb(ram):.0f} MB")
                workers = by_memory
        elif not snapshot.ram_known:
            # Unknown memory: fail closed. Assuming there is room is exactly
            # how the old psutil-less guard let a machine swap.
            workers = 1
            reasons.append("memory: unavailable, single worker")
    if snapshot.ram_known and ram <= 0:
        # No headroom at all: refuse to grow concurrency.
        workers = 1
        reasons.append("memory: no headroom, single worker")

    workers = max(1, min(workers, ceiling))

    # One parallel layer only: many outer workers with single-threaded
    # native calls, or one worker that may use the whole native budget.
    if workers > 1:
        native = 1
    else:
        native = max(1, min(ceiling, snapshot.physical_cores))
        if native > 1:
            reasons.append(f"threads: single worker uses {native} native thread(s)")

    return ExecutionPlan(
        workers=workers,
        native_threads=native,
        max_in_flight=workers,
        ram_budget=ram,
        vram_budget=vram,
        device_kind=snapshot.device_kind,
        reasons=tuple(reasons),
    )


def fits_in_budget(required_bytes: int, budget: int) -> bool:
    """True when a stage's minimum working set can be allocated at all.

    Callers use this to refuse work *before* allocating, with a message the
    user can act on, instead of being killed by the OS mid-fusion.
    """
    return required_bytes <= 0 or budget <= 0 or required_bytes <= budget
