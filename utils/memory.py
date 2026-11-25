"""Process and system memory, without third-party dependencies.

`psutil` is not a dependency of this project (it is not in pyproject and is
not installed with the app), yet the fusion memory guard imported it inside a
try/except and silently fell back to a hard-coded 4096 MB. On a machine with
less free memory than that, the guard allowed far too much work and the app
was killed by the OS — the "it freezes on big stacks" report.

Everything here uses the standard library and platform APIs. Unknown readings
remain zero so admission can refuse work rather than inventing spare RAM.
"""
import os
import sys


def _windows_memory_status():
    """Return one Win32 physical and commit-memory reading."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise OSError("GlobalMemoryStatusEx failed")
    return stat


def _windows_memory():
    stat = _windows_memory_status()
    # Public memory readings remain physical; admission also checks commit.
    return int(stat.ullAvailPhys), int(stat.ullTotalPhys)


def _windows_job_process_limit():
    """Return the immediate job's process commit limit, if exposed."""
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", wintypes.DWORD), ("minimum_working_set", ctypes.c_size_t),
                    ("maximum_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "read_operations", "write_operations", "other_operations",
            "read_bytes", "write_bytes", "other_bytes")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD)]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        limits = ExtendedLimits()
        if not kernel32.QueryInformationJobObject(None, 9, ctypes.byref(limits),
                                                  ctypes.sizeof(limits), None):
            return None
        if limits.basic.flags & 0x100:  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
            return int(limits.process_memory)
    except (OSError, AttributeError):
        return None
    return None


def _macos_memory():
    """Physical total and reclaimable pages from Darwin's native VM counters."""
    import ctypes
    libc = ctypes.CDLL(None)
    libc.sysctlbyname.argtypes = (
        ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p, ctypes.c_size_t,
    )
    libc.sysctlbyname.restype = ctypes.c_int
    total = ctypes.c_uint64()
    size = ctypes.c_size_t(ctypes.sizeof(total))
    if (libc.sysctlbyname(b"hw.memsize", ctypes.byref(total),
                          ctypes.byref(size), None, 0) != 0
            or size.value != ctypes.sizeof(total)):
        return 0, 0
    total_bytes = total.value
    libc.mach_host_self.argtypes = ()
    libc.mach_host_self.restype = ctypes.c_uint
    libc.host_page_size.argtypes = (ctypes.c_uint, ctypes.POINTER(ctypes.c_size_t))
    libc.host_page_size.restype = ctypes.c_int
    libc.mach_port_deallocate.argtypes = (ctypes.c_uint, ctypes.c_uint)
    libc.mach_port_deallocate.restype = ctypes.c_int
    # mach_task_self() uses this cached port. Resolve it before acquiring a right.
    task = ctypes.c_uint.in_dll(libc, "mach_task_self_").value
    host = libc.mach_host_self()
    if not host:
        return 0, total_bytes
    try:
        # vm_size_t is pointer-sized on Darwin.
        page_size = ctypes.c_size_t()
        if libc.host_page_size(host, ctypes.byref(page_size)) != 0 or not page_size.value:
            return 0, total_bytes
        statistics_argtypes = (
            ctypes.c_uint, ctypes.c_int, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        )
        stat64 = getattr(libc, "host_statistics64", None)
        if stat64 is not None:
            stat64.argtypes = statistics_argtypes
            stat64.restype = ctypes.c_int
            # HOST_VM_INFO64 (4) takes 40 integer_t words in the current revision.
            # Earlier revisions still include the free and inactive counts (0 and 2).
            stats64 = (ctypes.c_uint * 40)()
            count64 = ctypes.c_uint(len(stats64))
            if (stat64(host, 4, stats64, ctypes.byref(count64)) == 0
                    and 3 <= count64.value <= len(stats64)):
                available = (stats64[0] + stats64[2]) * page_size.value
                return min(total_bytes, available), total_bytes
        # HOST_VM_INFO (2) reports the same page counts in its 15-word layout.
        libc.host_statistics.argtypes = statistics_argtypes
        libc.host_statistics.restype = ctypes.c_int
        stats = (ctypes.c_uint * 15)()
        count = ctypes.c_uint(len(stats))
        if (libc.host_statistics(host, 2, stats, ctypes.byref(count)) != 0
                or not 3 <= count.value <= len(stats)):
            return 0, total_bytes
        available = (stats[0] + stats[2]) * page_size.value
        return min(total_bytes, available), total_bytes
    finally:
        libc.mach_port_deallocate(task, host)


def _linux_memory_limit(available, total):
    from utils.platform_utils import _linux_cgroup_paths
    for path, version in _linux_cgroup_paths("memory"):
        limit_file = "memory.max" if version == "cgroup2" else "memory.limit_in_bytes"
        usage_file = "memory.current" if version == "cgroup2" else "memory.usage_in_bytes"
        limit = 0
        try:
            with open(path + "/" + limit_file, encoding="ascii") as fh:
                value = fh.read().strip()
            if value == "max":
                continue
            limit = int(value)
            if limit <= 0:
                continue
            total = min(total, limit)
            with open(path + "/" + usage_file, encoding="ascii") as fh:
                usage = int(fh.read().strip())
            available = min(available, max(0, limit - usage))
        except (OSError, ValueError):
            # A known limit with unknown occupancy cannot prove headroom.
            if limit > 0:
                available = 0
    return available, total


def _posix_memory():
    if sys.platform == "darwin":
        return _macos_memory()
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="ascii") as fh:
                fields = dict(line.split(":", 1) for line in fh if ":" in line)
            total = int(fields["MemTotal"].split()[0]) * 1024
            # MemFree is conservative on older kernels without MemAvailable.
            avail = int(fields.get("MemAvailable", fields.get("MemFree", "0"))
                        .split()[0]) * 1024
            return _linux_memory_limit(avail, total)
        except (OSError, ValueError, KeyError, IndexError):
            return 0, 0
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = max(0, os.sysconf("SC_PHYS_PAGES")) * page
        avail = max(0, os.sysconf("SC_AVPHYS_PAGES")) * page
        return avail, total
    except (ValueError, OSError, AttributeError):
        return 0, 0


def system_memory_bytes():
    """(available_bytes, total_bytes) from a single OS reading."""
    try:
        if sys.platform == "win32":
            return _windows_memory()
        return _posix_memory()
    except Exception:
        return 0, 0


def admission_memory_bytes():
    """Return physical bytes, reported commit and immediate-job headroom."""
    try:
        if sys.platform == "win32":
            stat = _windows_memory_status()
            commit = int(stat.ullAvailPageFile)
            limit = _windows_job_process_limit()
            job_headroom = None
            if limit is not None:
                counters = _process_counters(extended=True)
                job_headroom = max(0, limit - int(counters.PrivateUsage)) if counters else 0
            return int(stat.ullAvailPhys), int(stat.ullTotalPhys), commit, job_headroom
        available, total = system_memory_bytes()
        return available, total, None, None
    except Exception:
        return 0, 0, None, None


def available_memory_bytes() -> int:
    """Physical memory the OS says it can hand out right now."""
    return system_memory_bytes()[0]


def total_memory_bytes() -> int:
    return system_memory_bytes()[1]


def _process_counters(extended=False):
    """Win32 process memory counters for this process, or None."""
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counter_type = PROCESS_MEMORY_COUNTERS
    if extended:
        class PROCESS_MEMORY_COUNTERS_EX(PROCESS_MEMORY_COUNTERS):
            _fields_ = [("PrivateUsage", ctypes.c_size_t)]
        counter_type = PROCESS_MEMORY_COUNTERS_EX

    counters = counter_type()
    counters.cb = ctypes.sizeof(counter_type)
    # GetCurrentProcess must be typed: its pseudo-handle is -1, and without an
    # explicit restype ctypes truncates it to a 32-bit int, the call fails and
    # every memory reading silently came back as 0.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    # argtypes are required as well: without them ctypes passes the 64-bit
    # pseudo-handle as a 32-bit int and the call fails silently.
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(counter_type),
        wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters),
                                    counters.cb)
    return counters if ok else None


def failure_memory_snapshot():
    """Available Windows memory counters; omit readings that cannot be obtained."""
    if sys.platform != 'win32':
        return {}
    import ctypes
    from ctypes import wintypes

    class PERFORMANCE_INFORMATION(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal',
                'PhysicalAvailable', 'SystemCache', 'KernelTotal',
                'KernelPaged', 'KernelNonpaged', 'PageSize')
        ] + [(name, wintypes.DWORD) for name in (
            'HandleCount', 'ProcessCount', 'ThreadCount')]

    result = {}
    try:
        info = PERFORMANCE_INFORMATION()
        info.cb = ctypes.sizeof(info)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetPerformanceInfo.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), wintypes.DWORD]
        psapi.GetPerformanceInfo.restype = wintypes.BOOL
        if psapi.GetPerformanceInfo(ctypes.byref(info), info.cb) and info.PageSize:
            result.update(commit_total_bytes=int(info.CommitTotal * info.PageSize),
                          commit_limit_bytes=int(info.CommitLimit * info.PageSize),
                          available_ram_bytes=int(info.PhysicalAvailable * info.PageSize))
    except Exception:
        pass
    try:
        counters = _process_counters(extended=True)
        if counters is not None:
            result.update(private_commit_bytes=int(counters.PrivateUsage),
                          working_set_bytes=int(counters.WorkingSetSize))
    except Exception:
        pass
    return {key: value for key, value in result.items() if 0 <= value <= sys.maxsize}


def process_rss_bytes() -> int:
    """Resident size of this process (what the task manager shows)."""
    try:
        if sys.platform == "win32":
            counters = _process_counters()
            if counters is not None:
                return int(counters.WorkingSetSize)
            return 0
        try:
            with open("/proc/self/statm", "r", encoding="ascii") as fh:
                pages = int(fh.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError, AttributeError):
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(usage) * (1024 if sys.platform != "darwin" else 1)
    except Exception:
        return 0


def peak_rss_bytes() -> int:
    """Largest resident size this process has reached."""
    try:
        if sys.platform == "win32":
            counters = _process_counters()
            if counters is not None:
                return int(counters.PeakWorkingSetSize)
            return 0
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage) * (1024 if sys.platform != "darwin" else 1)
    except Exception:
        return 0


def mb(value_bytes: int) -> float:
    return value_bytes / (1024.0 ** 2)


def configure_opencv_memory() -> None:
    """Bound OpenCV's own parallelism.

    Measured on a 1600x2400x8 guided-filter fusion: with OpenCV's default
    (one thread per core, OpenCL on) the process peaked at 1.5 GB; with a
    single OpenCV thread and OpenCL off it peaked at 186 MB — an 8x reduction,
    and it was FASTER (2.1 s vs 2.9 s), because the fusion already parallelises
    over tiles and the two levels of threading only fought each other.

    OpenCL is disabled deliberately: this project runs the classical
    algorithms on the CPU by design (see AGENTS.md), so an OpenCL context only
    adds per-thread buffers and driver overhead.

    Safe to call more than once.
    """
    try:
        import cv2
    except Exception:
        return
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    try:
        # Zero explicitly disables OpenCV parallel regions; the backend's
        # reported worker-pool capacity is not the requested thread limit.
        cv2.setNumThreads(0)
    except Exception:
        pass
