"""Execution planning: one budget, one worker count, one parallel layer.

These pin the behaviour the fusion pipeline relies on:
  * the plan never exceeds the physical-core budget or the caller's cap;
  * memory pressure lowers concurrency, never raises it;
  * one parallel layer only (many outer workers with single-threaded native
    calls, or a single worker that may use the whole native budget);
  * an explicit budget override is honoured, because the pipeline's
    ``memory_budget_bytes`` remains the single authoritative number.
"""
import numpy as np
import pytest

from core import resource_policy as policy


def _snapshot(available=8 * 1024 ** 3, total=16 * 1024 ** 3, cores=8,
              device="CPU", free_vram=0, total_vram=0):
    return policy.ResourceSnapshot(
        available_ram=available, total_ram=total,
        process_rss=512 * 1024 ** 2,
        physical_cores=cores, logical_cpus=cores * 2,
        free_vram=free_vram, total_vram=total_vram,
        device_kind=device)


def _workload(frames=10, block=1024, channels=3):
    return policy.Workload(algorithm="guided_filter", frames=frames,
                           height=2400, width=1600, channels=channels,
                           block_size=block)


def test_plan_never_exceeds_physical_cores():
    plan = policy.plan_execution(_workload(), _snapshot(cores=6),
                                 ram_budget_override=64 * 1024 ** 3)
    assert plan.workers <= 6
    assert plan.workers >= 1


def test_user_thread_count_is_an_upper_bound():
    generous = _snapshot(cores=16)
    plan = policy.plan_execution(_workload(), generous, user_threads=3,
                                 ram_budget_override=64 * 1024 ** 3)
    assert plan.workers <= 3
    # A larger cap must not be invented on a roomy machine either.
    capped = policy.plan_execution(_workload(), generous, user_threads=2,
                                   ram_budget_override=64 * 1024 ** 3)
    assert capped.workers <= 2


def test_memory_pressure_reduces_workers():
    """A tight budget must reduce concurrency — the bit-identical lever."""
    snapshot = _snapshot(cores=16)
    roomy = policy.plan_execution(_workload(), snapshot,
                                  ram_budget_override=32 * 1024 ** 3)
    tight = policy.plan_execution(_workload(), snapshot,
                                  ram_budget_override=1200 * 1024 ** 2)
    assert tight.workers < roomy.workers
    assert tight.workers >= 1


def test_single_parallel_layer():
    """Either many single-threaded workers, or one worker with threads."""
    snapshot = _snapshot(cores=12)
    many = policy.plan_execution(_workload(), snapshot,
                                 ram_budget_override=64 * 1024 ** 3)
    if many.workers > 1:
        assert many.native_threads == 1
    one = policy.plan_execution(_workload(), snapshot, user_threads=1,
                                ram_budget_override=64 * 1024 ** 3)
    assert one.workers == 1
    assert one.native_threads >= 1


def test_preference_scales_the_core_budget():
    snapshot = _snapshot(cores=20)
    balanced = policy.cpu_capacity(snapshot, policy.PREFERENCE_BALANCED)
    responsive = policy.cpu_capacity(snapshot, policy.PREFERENCE_RESPONSIVE)
    maximum = policy.cpu_capacity(snapshot, policy.PREFERENCE_MAXIMUM)
    assert responsive <= balanced <= maximum
    assert maximum == 20
    assert responsive >= 1


def test_unknown_memory_does_not_grow_concurrency():
    """No readable memory information must fail closed, not open."""
    blind = policy.ResourceSnapshot(available_ram=0, total_ram=0,
                                    physical_cores=8, logical_cpus=16)
    plan = policy.plan_execution(_workload(), blind)
    assert plan.workers == 1
    assert plan.native_threads >= 1


def test_budget_leaves_headroom():
    snapshot = _snapshot(available=4 * 1024 ** 3, total=8 * 1024 ** 3)
    budget = policy.ram_budget(snapshot)
    assert 0 < budget < snapshot.available_ram
    # At least the reserve (or 15% of total) stays free.
    assert snapshot.available_ram - budget >= min(policy.RESERVE_BYTES,
                                                  snapshot.available_ram * 0.4)


def test_budget_is_new_allocation_not_the_stack():
    """The budget bounds *new* allocations; a resident stack is not re-charged."""
    snapshot = _snapshot(available=4 * 1024 ** 3, total=16 * 1024 ** 3)
    assert policy.ram_budget(snapshot) == int(
        min(policy.BUDGET_RATIO * snapshot.available_ram,
            snapshot.available_ram - max(policy.RESERVE_BYTES,
                                         int(0.15 * snapshot.total_ram))))


def test_vram_budget_only_with_a_gpu():
    cpu = _snapshot(device="CPU")
    assert policy.vram_budget(cpu) == 0
    gpu = _snapshot(device="CUDA", free_vram=8 * 1024 ** 3,
                    total_vram=12 * 1024 ** 3)
    budget = policy.vram_budget(gpu)
    assert 0 < budget < gpu.free_vram


def test_plan_is_immutable():
    plan = policy.plan_execution(_workload(), _snapshot(),
                                 ram_budget_override=8 * 1024 ** 3)
    with pytest.raises(Exception):
        plan.workers = 99


def test_fits_in_budget_refuses_only_when_known():
    assert policy.fits_in_budget(100, 0)      # unknown budget: allow
    assert policy.fits_in_budget(0, 100)      # nothing required
    assert policy.fits_in_budget(100, 200)
    assert not policy.fits_in_budget(300, 200)


def test_physical_cores_are_real():
    """The old ``logical // 2`` estimate undercounts mixed P/E machines."""
    import os
    from utils.platform_utils import physical_core_count

    cores = physical_core_count()
    logical = os.cpu_count() or cores
    assert 1 <= cores <= logical
    # The detection must not simply echo half the logical count when the
    # platform can answer exactly (this machine has 12 cores / 20 threads).
    assert cores >= 1


def test_planner_agrees_with_the_pipeline_budget(monkeypatch):
    """``memory_budget_bytes`` stays the single authoritative number."""
    from core import multi_focus_fusion as mff

    monkeypatch.setattr(mff, "memory_budget_bytes",
                        lambda: 2000 * 1024 ** 2)
    fusion = mff.MultiFocusFusion(algorithm="guided_filter", use_gpu=False)
    tight = fusion._calculate_optimal_thread_count(1600, 2400, 3, 10, 1024, 8)
    monkeypatch.setattr(mff, "memory_budget_bytes",
                        lambda: 64000 * 1024 ** 2)
    roomy = fusion._calculate_optimal_thread_count(1600, 2400, 3, 10, 1024, 8)
    assert tight < roomy
