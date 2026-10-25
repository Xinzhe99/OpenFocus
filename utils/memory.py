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


def _windows_memory():
    """(available_bytes, total_bytes) via the Win32 GlobalMemoryStatusEx."""
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
    # "Available" must be the physical figure: the page-file numbers include
    # disk-backed virtual memory, and allocating into that is what makes a
    # fusion appear to hang while the machine swaps.
    return int(stat.ullAvailPhys), int(stat.ullTotalPhys)


def _macos_memory():
    """Physical total and reclaimable pages from Darwin's native VM counters."""
    import ctypes
    libc = ctypes.CDLL(None)
    total = ctypes.c_uint64()
    size = ctypes.c_size_t(ctypes.sizeof(total))
    if libc.sysctlbyname(b"hw.memsize", ctypes.byref(total),
                         ctypes.byref(size), None, 0) != 0:
        return 0, 0
    libc.mach_host_self.restype = ctypes.c_uint
    host = libc.mach_host_self()
    try:
        page_size = ctypes.c_uint()
        if libc.host_page_size(host, ctypes.byref(page_size)) != 0:
            return 0, int(total.value)
        # HOST_VM_INFO uses integer_t fields: free, active, inactive, wired,
        # then fault/pageout counters. Speculative pages are already free.
        stats = (ctypes.c_uint * 15)()
        count = ctypes.c_uint(15)
        if libc.host_statistics(host, 2, stats, ctypes.byref(count)) != 0:
            return 0, int(total.value)
        available = (int(stats[0]) + int(stats[2])) * page_size.value
        return available, int(total.value)
    finally:
        # mach_init.h defines mach_task_self() as this cached exported port.
        task = ctypes.c_uint.in_dll(libc, "mach_task_self_").value
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


def available_memory_bytes() -> int:
    """Physical memory the OS says it can hand out right now."""
    try:
        if sys.platform == "win32":
            return _windows_memory()[0]
        return _posix_memory()[0]
    except Exception:
        return 0


def total_memory_bytes() -> int:
    try:
        if sys.platform == "win32":
            return _windows_memory()[1]
        return _posix_memory()[1]
    except Exception:
        return 0


def _process_counters():
    """Win32 PROCESS_MEMORY_COUNTERS for this process, or None."""
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

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    # GetCurrentProcess must be typed: its pseudo-handle is -1, and without an
    # explicit restype ctypes truncates it to a 32-bit int, the call fails and
    # every memory reading silently came back as 0.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    # argtypes are required as well: without them ctypes passes the 64-bit
    # pseudo-handle as a 32-bit int and the call fails silently.
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters),
                                    counters.cb)
    return counters if ok else None


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
        cv2.setNumThreads(1)
    except Exception:
        pass
