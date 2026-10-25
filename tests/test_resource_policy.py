"""Admission and lease regressions using deterministic resource snapshots."""
import io
import queue
import struct
import threading
from dataclasses import replace

import pytest

from core import resource_policy as policy
from core.registration import RegistrationCancelled
from utils import memory, platform_utils

GiB = 1024 ** 3


def _snapshot(**changes):
    snapshot = policy.ResourceSnapshot(available_ram=8 * GiB, total_ram=16 * GiB,
                                       physical_cores=8, logical_cpus=16)
    return replace(snapshot, **changes)


def _workload(**changes):
    workload = policy.Workload(algorithm="guided_filter", frames=10,
                               height=200, width=300, fixed_bytes=200,
                               per_worker_bytes=300, outer_tasks=10)
    return replace(workload, **changes)


@pytest.mark.parametrize("cores,expected", [(1, 1), (2, 1), (3, 2), (4, 3),
                                            (5, 4), (8, 6), (12, 9), (20, 16)])
def test_balanced_preserves_spare_core(cores, expected):
    assert policy.cpu_capacity(_snapshot(physical_cores=cores, logical_cpus=cores),
                               policy.PREFERENCE_BALANCED) == expected


def test_affinity_and_fractional_quota_limit_all_preferences():
    limited = _snapshot(physical_cores=12, logical_cpus=4, cpu_limit=2.5)
    assert policy.cpu_capacity(limited, policy.PREFERENCE_MAXIMUM) == 2
    assert policy.cpu_capacity(limited, policy.PREFERENCE_BALANCED) == 1
    assert policy.cpu_capacity(limited, policy.PREFERENCE_RESPONSIVE) == 1


def test_fixed_allocations_reduce_concurrency_before_workers():
    plan = policy.plan_execution(_workload(), _snapshot(), ram_budget_override=1100)
    assert (plan.workers, plan.native_threads, plan.max_in_flight) == (3, 1, 3)
    tighter = policy.plan_execution(_workload(), _snapshot(), ram_budget_override=500)
    assert (tighter.workers, tighter.max_in_flight) == (1, 1)
    with pytest.raises(policy.ResourceLimitError):
        policy.plan_execution(_workload(), _snapshot(), ram_budget_override=499)


def test_threads_and_batch_are_caps_and_zero_means_auto():
    auto = policy.plan_execution(_workload(), _snapshot(), user_threads=0)
    assert auto.workers == 6
    capped = policy.plan_execution(_workload(), _snapshot(), user_threads=3,
                                    user_batch=2)
    assert (capped.workers, capped.native_threads) == (2, 1)
    native = policy.plan_execution(_workload(parallel_layer="native"), _snapshot(),
                                    user_threads=3, user_batch=8)
    assert (native.workers, native.native_threads) == (1, 3)


@pytest.mark.parametrize("snapshot", [policy.ResourceSnapshot(),
    _snapshot(available_ram=GiB, total_ram=4 * GiB)])
def test_unknown_or_no_headroom_refuses_even_with_large_override(snapshot):
    with pytest.raises(policy.ResourceLimitError):
        policy.plan_execution(_workload(), snapshot, ram_budget_override=100 * GiB)
    with pytest.raises(policy.ResourceLimitError):
        policy.require_memory(1, snapshot, user_limit=100 * GiB)


def test_override_cannot_raise_detected_headroom():
    snapshot = _snapshot(available_ram=2 * GiB, total_ram=4 * GiB)
    plan = policy.plan_execution(_workload(), snapshot, ram_budget_override=100 * GiB)
    assert plan.ram_budget == int(0.9 * GiB)
    with pytest.raises(policy.ResourceLimitError):
        policy.plan_execution(_workload(fixed_bytes=GiB), snapshot,
                              ram_budget_override=100 * GiB)


def test_one_gib_reserve_and_resident_buffers_not_recharged():
    snapshot = _snapshot(available_ram=GiB + GiB // 4, total_ram=2 * GiB)
    assert policy.ram_budget(snapshot) == GiB // 4
    assert policy.ram_budget(replace(snapshot, process_rss=GiB)) == GiB // 4


@pytest.mark.parametrize("required,budget,expected", [(100, 0, False),
    (100, -1, False), (0, 0, True), (100, 100, True), (101, 100, False)])
def test_budget_check_never_treats_zero_as_unlimited(required, budget, expected):
    assert policy.fits_in_budget(required, budget) is expected


def test_whole_frame_admission_accounts_for_worker_scratch():
    workload = policy.Workload(algorithm="registration", frames=4, height=10,
                               width=20, channels=3, itemsize=2)
    # Output stack: 4*10*20*3*2=4800; each worker's six float buffers: 14400.
    plan = policy.plan_execution(workload, _snapshot(), ram_budget_override=33600)
    assert plan.workers == 2
    with pytest.raises(policy.ResourceLimitError):
        policy.plan_execution(workload, _snapshot(), ram_budget_override=19199)


def test_tile_estimate_includes_fixed_accumulators_and_retained_result():
    workload = policy.Workload(frames=2, height=100, width=100, channels=3,
                               block_size=10, itemsize=2)
    fixed, worker = policy.estimate_working_set(workload)
    assert fixed == 160000
    assert worker == 21760
    with pytest.raises(policy.ResourceLimitError):
        policy.plan_execution(workload, _snapshot(), ram_budget_override=181759)
    assert workload.block_size == 10


def test_cuda_requires_known_vram_and_accounts_for_fixed_model():
    workload = _workload(algorithm="stackmff", parallel_layer="native",
                         fixed_vram_bytes=GiB, per_worker_vram_bytes=GiB)
    cuda = _snapshot(device_kind="CUDA", free_vram=4 * GiB, total_vram=8 * GiB)
    plan = policy.plan_execution(workload, cuda, user_threads=3)
    assert (plan.workers, plan.native_threads) == (1, 3)
    with pytest.raises(policy.ResourceLimitError):
        policy.plan_execution(workload, replace(cuda, free_vram=2 * GiB))
    with pytest.raises(policy.ResourceLimitError):
        policy.plan_execution(workload, replace(cuda, free_vram=0))


def test_ai_requires_explicit_estimate():
    with pytest.raises(policy.ResourceLimitError):
        policy.plan_execution(policy.Workload(algorithm="ai", frames=2,
                                               height=20, width=30), _snapshot())


def _thread_call(function):
    result = queue.Queue()
    def run():
        try:
            result.put((True, function()))
        except BaseException as error:
            result.put((False, error))
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def _result(call):
    thread, result = call
    thread.join(2)
    assert not thread.is_alive(), "heavy-job waiter failed to finish"
    ok, value = result.get_nowait()
    if not ok:
        raise value
    return value


def test_heavy_job_contention_holds_until_outermost_owner_exits():
    waiting, entered = threading.Event(), threading.Event()
    def contender():
        with policy.heavy_job(lambda: waiting.set() or False):
            entered.set()
    with policy.heavy_job():
        with policy.heavy_job():
            call = _thread_call(contender)
            assert waiting.wait(2)
            assert not entered.wait(0.1)
        assert not entered.wait(0.1)
    _result(call)
    assert entered.is_set()


def test_waiting_cancellation_does_not_enter_or_release_running_owner():
    waiting, cancelled, entered = threading.Event(), threading.Event(), threading.Event()
    def waiter():
        def should_cancel():
            waiting.set()
            return cancelled.is_set()
        with policy.heavy_job(should_cancel):
            entered.set()
    with policy.heavy_job():
        call = _thread_call(waiter)
        assert waiting.wait(2)
        cancelled.set()
        with pytest.raises(RegistrationCancelled):
            _result(call)
        assert not entered.is_set()
        next_call = _thread_call(lambda: _enter(entered))
        assert not entered.wait(0.1)
    _result(next_call)
    assert entered.is_set()


def _enter(event):
    with policy.heavy_job():
        event.set()


def test_heavy_job_releases_on_exception_and_preentry_cancellation():
    with pytest.raises(RuntimeError, match="native failure"):
        with policy.heavy_job():
            with policy.heavy_job():
                raise RuntimeError("native failure")
    with pytest.raises(RegistrationCancelled):
        with policy.heavy_job(lambda: True):
            pytest.fail("cancelled work was admitted")
    entered = threading.Event()
    _result(_thread_call(lambda: _enter(entered)))
    assert entered.is_set()


def test_cancellation_after_acquiring_lease_releases_it():
    checks = iter((False, True))
    with pytest.raises(RegistrationCancelled):
        with policy.heavy_job(lambda: next(checks)):
            pytest.fail("work started after cancellation at admission")
    entered = threading.Event()
    _result(_thread_call(lambda: _enter(entered)))
    assert entered.is_set()


def _files(monkeypatch, contents):
    def fake_open(path, *args, **kwargs):
        if str(path) not in contents:
            raise FileNotFoundError(path)
        return io.StringIO(contents[str(path)])
    monkeypatch.setattr("builtins.open", fake_open)


def test_linux_affinity_counts_smt_siblings_once(monkeypatch):
    _files(monkeypatch, {"/proc/cpuinfo":
        "processor : 0\nphysical id : 0\ncore id : 0\n\n"
        "processor : 1\nphysical id : 0\ncore id : 0\n\n"
        "processor : 2\nphysical id : 0\ncore id : 1\n\n"
        "processor : 3\nphysical id : 1\ncore id : 0"})
    monkeypatch.setattr(platform_utils.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    assert platform_utils._linux_physical_cores() == 1
    monkeypatch.setattr(platform_utils.os, "sched_getaffinity", lambda pid: {1, 3}, raising=False)
    assert platform_utils._linux_physical_cores() == 2


def test_windows_topology_respects_affinity_and_hybrid_cores():
    pointer = struct.calcsize("P")
    def record(mask):
        body = bytes(22) + struct.pack("<H", 1)
        body += mask.to_bytes(pointer, "little") + bytes(8)
        return struct.pack("<II", 0, len(body) + 8) + body
    raw = record(0b11) + record(0b1100) + record(0b10000)
    assert platform_utils._windows_core_records(raw) == 3
    assert platform_utils._windows_core_records(raw, 0b11, 0) == 1
    assert platform_utils._windows_core_records(raw, 0b10010, 0) == 2
    assert platform_utils._windows_core_records(raw[:-1]) == 0


def test_cgroup_v2_nested_quota_uses_tightest_parent(monkeypatch):
    _files(monkeypatch, {
        "/proc/self/cgroup": "0::/tenant/job\n",
        "/proc/self/mountinfo": "30 20 0:27 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        "/sys/fs/cgroup/job/cpu.max": "350000 100000\n",
        "/sys/fs/cgroup/cpu.max": "150000 100000\n",
    })
    assert platform_utils._linux_cpu_quota() == 1.5


def test_cgroup_v1_quota_and_unlimited_parent(monkeypatch):
    _files(monkeypatch, {
        "/proc/self/cgroup": "4:cpu,cpuacct:/job\n",
        "/proc/self/mountinfo": "30 20 0:27 / /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n",
        "/sys/fs/cgroup/cpu/job/cpu.cfs_quota_us": "250000",
        "/sys/fs/cgroup/cpu/job/cpu.cfs_period_us": "100000",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
    })
    assert platform_utils._linux_cpu_quota() == 2.5


def test_cpu_probe_rechecks_limits_and_marks_unknown_topology(monkeypatch):
    monkeypatch.setattr(platform_utils, "is_windows", lambda: False)
    monkeypatch.setattr(platform_utils, "is_linux", lambda: True)
    monkeypatch.setattr(platform_utils.os, "cpu_count", lambda: 20)
    monkeypatch.setattr(platform_utils, "_affinity_cpu_count", lambda: 8)
    monkeypatch.setattr(platform_utils, "_linux_physical_cores", lambda: 6)
    monkeypatch.setattr(platform_utils, "_linux_cpu_quota", lambda: 2.5)
    assert platform_utils.cpu_limits() == (2, 8, 2.5, True)
    monkeypatch.setattr(platform_utils, "_linux_cpu_quota", lambda: 0)
    assert platform_utils.cpu_limits() == (6, 8, 0, True)
    monkeypatch.setattr(platform_utils, "_linux_physical_cores", lambda: 0)
    assert platform_utils.cpu_limits() == (1, 8, 0, False)


def test_linux_available_memory_is_bounded_by_cgroup_occupancy(monkeypatch):
    monkeypatch.setattr(memory.sys, "platform", "linux")
    _files(monkeypatch, {
        "/proc/meminfo": "MemTotal: 16777216 kB\nMemAvailable: 8388608 kB\n",
        "/proc/self/cgroup": "0::/job\n",
        "/proc/self/mountinfo": "30 20 0:27 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        "/sys/fs/cgroup/job/memory.max": str(4 * GiB),
        "/sys/fs/cgroup/job/memory.current": str(3 * GiB),
        "/sys/fs/cgroup/memory.max": "max",
    })
    assert memory._posix_memory() == (GiB, 4 * GiB)


def test_missing_memory_availability_is_never_invented(monkeypatch):
    monkeypatch.setattr(memory.sys, "platform", "linux")
    _files(monkeypatch, {"/proc/meminfo": "MemTotal: 8388608 kB\n"})
    assert memory.available_memory_bytes() == 0
    assert memory.total_memory_bytes() == 8 * GiB
