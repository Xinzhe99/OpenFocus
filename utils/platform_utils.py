"""
跨平台工具模块 - 处理 Windows/macOS/Linux 兼容性
"""

import os
import platform
import struct
import sys


def get_os_type() -> str:
    """
    获取当前操作系统类型
    
    Returns:
        'windows' | 'macos' | 'linux'
    """
    system = platform.system().lower()
    if system == 'darwin':
        return 'macos'
    elif system == 'windows':
        return 'windows'
    elif system == 'linux':
        return 'linux'
    else:
        return 'unknown'


def get_ui_font_family() -> str:
    """
    获取适合当前操作系统的 UI 字体族
    
    Returns:
        字体族名字符串
    """
    os_type = get_os_type()
    
    if os_type == 'macos':
        # macOS 系统字体
        return '"SF Pro", "Helvetica Neue", Arial, sans-serif'
    elif os_type == 'windows':
        # Windows 系统字体
        return '"Segoe UI", "Microsoft YaHei", Arial, sans-serif'
    else:
        # Linux 系统字体
        return '"Ubuntu", "DejaVu Sans", Arial, sans-serif'


def get_monospace_font_family() -> str:
    """
    获取适合当前操作系统的等宽字体族
    
    Returns:
        字体族名字符串
    """
    os_type = get_os_type()
    
    if os_type == 'macos':
        # macOS 等宽字体
        return '"SF Mono", "Monaco", "Menlo", Consolas, monospace'
    elif os_type == 'windows':
        # Windows 等宽字体
        return 'Consolas, "Courier New", monospace'
    else:
        # Linux 等宽字体
        return '"Ubuntu Mono", "DejaVu Sans Mono", Consolas, monospace'


def get_default_font() -> str:
    """
    获取适合当前操作系统的默认字体
    
    Returns:
        字体名字符串
    """
    os_type = get_os_type()
    
    if os_type == 'macos':
        return 'SF Pro'
    elif os_type == 'windows':
        return 'Microsoft YaHei'
    else:
        return 'Ubuntu'


def is_windows() -> bool:
    """检查是否为 Windows 系统"""
    return get_os_type() == 'windows'


def is_macos() -> bool:
    """检查是否为 macOS 系统"""
    return get_os_type() == 'macos'


def is_linux() -> bool:
    """检查是否为 Linux 系统"""
    return get_os_type() == 'linux'


# 导出常用函数
__all__ = [
    'get_os_type',
    'get_ui_font_family',
    'get_monospace_font_family',
    'get_default_font',
    'is_windows',
    'is_macos',
    'is_linux',
]


_GPU_PRESENT_CACHE = None


def has_compute_gpu() -> bool:
    """True when this machine has a usable GPU for the neural path.

    Asked from the menu, so it must be cheap: importing torch costs ~2.1 s and
    pins ~193 MB (see core.multi_focus_fusion.is_ai_fusion_available). This
    probes the operating system instead, taking a few milliseconds, and caches
    the answer.

    A "GPU" here means CUDA or Apple-silicon MPS - the two back-ends the
    fusion actually uses. Merely having a display adapter is not enough, which
    is why the check shells out to nvidia-smi rather than enumerating devices.
    """
    global _GPU_PRESENT_CACHE
    if _GPU_PRESENT_CACHE is not None:
        return _GPU_PRESENT_CACHE

    found = False
    try:
        if is_macos():
            # Every Apple-silicon Mac has a usable MPS device.
            found = platform.machine().lower() in ("arm64", "aarch64")
        elif is_windows() or is_linux():
            import shutil
            import subprocess
            exe = shutil.which("nvidia-smi")
            if exe:
                try:
                    out = subprocess.run(
                        [exe, "--query-gpu=name", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=2.0,
                        creationflags=(0x08000000 if is_windows() else 0),
                    )
                    found = out.returncode == 0 and bool(out.stdout.strip())
                except (OSError, subprocess.SubprocessError):
                    found = False
    except Exception:
        found = False

    _GPU_PRESENT_CACHE = found
    return found


def reset_gpu_probe_cache() -> None:
    """Forget the cached answer (tests, or a driver installed mid-session)."""
    global _GPU_PRESENT_CACHE
    _GPU_PRESENT_CACHE = None


def _windows_physical_cores() -> int:
    """Count RelationProcessorCore records via GetLogicalProcessorInformationEx.

    Each record starts with ``DWORD Relationship; DWORD Size`` followed by the
    relationship-specific union; walking by the record's own Size avoids
    hard-coding that union layout. (Reading the header as two WORDs — the
    mistake behind the older SYSTEM_LOGICAL_PROCESSOR_INFORMATION attempt —
    yields Size 0 and misreads the strides on x64, reporting 10 cores for a
    12-core machine.)
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    func = kernel32.GetLogicalProcessorInformationEx
    func.argtypes = [wintypes.DWORD, ctypes.c_void_p,
                     ctypes.POINTER(wintypes.DWORD)]
    func.restype = wintypes.BOOL
    RELATION_PROCESSOR_CORE = 0
    length = wintypes.DWORD(0)
    # First call sizes the buffer; ERROR_INSUFFICIENT_BUFFER is expected.
    func(RELATION_PROCESSOR_CORE, None, ctypes.byref(length))
    if length.value <= 0:
        return 0
    buffer = ctypes.create_string_buffer(length.value)
    if not func(RELATION_PROCESSOR_CORE, ctypes.byref(buffer),
                ctypes.byref(length)):
        return 0
    return _windows_core_records(bytes(buffer[:length.value]), *_windows_affinity())


def _windows_core_records(raw, affinity_mask=0, affinity_group=0):
    """Decode processor-core records, intersecting the process affinity mask."""
    cores = 0
    offset = 0
    while offset + 8 <= len(raw):
        relationship, size = struct.unpack_from("<II", raw, offset)
        if size < 8 or offset + size > len(raw):
            return 0
        if relationship == 0:
            if not affinity_mask:
                cores += 1
            elif size >= 32:
                groups = struct.unpack_from("<H", raw, offset + 30)[0]
                pointer_size = struct.calcsize("P")
                stride = pointer_size + 8
                for index in range(groups):
                    start = offset + 32 + index * stride
                    if start + stride > offset + size:
                        return 0
                    mask = int.from_bytes(raw[start:start + pointer_size], "little")
                    group = struct.unpack_from("<H", raw, start + pointer_size)[0]
                    if group == affinity_group and mask & affinity_mask:
                        cores += 1
                        break
        offset += size
    return cores


def _linux_physical_cores() -> int:
    """Count physical/core pairs only for processors in this process's affinity."""
    try:
        affinity = set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
            records = fh.read().strip().split("\n\n")
        pairs = set()
        for record in records:
            fields = dict(line.split(":", 1) for line in record.splitlines()
                          if ":" in line)
            fields = {key.strip(): value.strip() for key, value in fields.items()}
            processor = int(fields.get("processor", "-1"))
            if affinity is not None and processor not in affinity:
                continue
            if "physical id" not in fields or "core id" not in fields:
                return 0
            pairs.add((fields["physical id"], fields["core id"]))
        return len(pairs)
    except (OSError, ValueError):
        return 0


def _macos_physical_cores() -> int:
    """sysctl hw.physicalcpu (Apple silicon reports P+E cores)."""
    import ctypes
    try:
        libc = ctypes.CDLL(None)
        size = ctypes.c_size_t(0)
        if libc.sysctlbyname(b"hw.physicalcpu", None,
                             ctypes.byref(size), None, 0) != 0:
            return 0
        value = ctypes.c_uint32(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if libc.sysctlbyname(b"hw.physicalcpu", ctypes.byref(value),
                             ctypes.byref(size), None, 0) != 0:
            return 0
        return int(value.value)
    except (AttributeError, OSError):
        return 0


def _linux_cgroup_paths(controller):
    """Yield this cgroup and its ancestors on matching v1/v2 mounts."""
    from pathlib import PurePosixPath
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as fh:
            memberships = [line.strip().split(":", 2) for line in fh]
        with open("/proc/self/mountinfo", encoding="utf-8") as fh:
            mounts = fh.readlines()
    except OSError:
        return
    seen = set()
    for line in mounts:
        before, separator, after = line.partition(" - ")
        left, right = before.split(), after.split()
        if not separator or len(left) < 5 or len(right) < 3:
            continue
        version = right[0]
        if version not in ("cgroup", "cgroup2"):
            continue
        if version == "cgroup" and controller not in right[2].split(","):
            continue
        # mountinfo encodes whitespace and backslashes as octal escapes.
        def unescape(value):
            for encoded, decoded in (("\\040", " "), ("\\011", "\t"),
                                     ("\\012", "\n"), ("\\134", "\\")):
                value = value.replace(encoded, decoded)
            return value
        root = PurePosixPath(unescape(left[3]))
        mount = PurePosixPath(unescape(left[4]))
        for membership in memberships:
            if len(membership) != 3:
                continue
            _, controllers, location = membership
            if (version == "cgroup2" and controllers or
                    version == "cgroup" and controller not in controllers.split(",")):
                continue
            location = PurePosixPath(location)
            try:
                relative = location.relative_to(root)
            except ValueError:
                # A cgroup namespace may expose its mount root as '/'.
                if location != PurePosixPath("/"):
                    continue
                relative = PurePosixPath(".")
            current = mount / relative
            if ".." in current.parts:
                continue
            while True:
                key = (str(current), version)
                if key not in seen:
                    seen.add(key)
                    yield key
                if current == mount:
                    break
                current = current.parent


def _linux_cpu_quota():
    """Tightest inherited CPU quota in cores, or zero when unrestricted/unknown."""
    limits = []
    for path, version in _linux_cgroup_paths("cpu"):
        try:
            if version == "cgroup2":
                with open(path + "/cpu.max", encoding="ascii") as fh:
                    quota, period = fh.read().split()
                if quota == "max":
                    continue
            else:
                with open(path + "/cpu.cfs_quota_us", encoding="ascii") as fh:
                    quota = fh.read().strip()
                with open(path + "/cpu.cfs_period_us", encoding="ascii") as fh:
                    period = fh.read().strip()
            quota, period = int(quota), int(period)
            if quota > 0 and period > 0:
                limits.append(quota / period)
        except (OSError, ValueError):
            continue
    return min(limits) if limits else 0


def _windows_affinity():
    import ctypes
    from ctypes import wintypes

    class GroupAffinity(ctypes.Structure):
        _fields_ = [("mask", ctypes.c_size_t), ("group", wintypes.WORD),
                    ("reserved", wintypes.WORD * 3)]

    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetCurrentThread.restype = wintypes.HANDLE
        kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.GetProcessAffinityMask.restype = wintypes.BOOL
        kernel.GetThreadGroupAffinity.argtypes = [wintypes.HANDLE,
                                                   ctypes.POINTER(GroupAffinity)]
        kernel.GetThreadGroupAffinity.restype = wintypes.BOOL
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        group = GroupAffinity()
        if (kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),
                ctypes.byref(process_mask), ctypes.byref(system_mask)) and
                kernel.GetThreadGroupAffinity(kernel.GetCurrentThread(),
                                               ctypes.byref(group))):
            return int(process_mask.value), int(group.group)
    except (OSError, AttributeError):
        pass
    return 0, 0


def _affinity_cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    if is_windows():
        return _windows_affinity()[0].bit_count()
    return 0


def cpu_limits():
    """Return physical capacity, allowed logical CPUs, quota and topology validity.

    Probe per job: affinity and cgroup limits can change during a session.
    Unknown topology uses one core, never presents logical//2 as measured.
    """
    logical = max(1, os.cpu_count() or 1)
    affinity = _affinity_cpu_count()
    if affinity > 0:
        logical = min(logical, affinity)
    detected = 0
    quota = 0
    try:
        if is_windows():
            detected = _windows_physical_cores()
        elif is_linux():
            detected = _linux_physical_cores()
            quota = _linux_cpu_quota()
        elif is_macos():
            detected = _macos_physical_cores()
    except (OSError, AttributeError, ValueError):
        detected = 0
    known = detected > 0
    cores = min(logical, detected) if known else 1
    if quota > 0:
        cores = min(cores, max(1, int(quota)))
    return cores, logical, quota, known


def physical_core_count() -> int:
    """Usable physical-core capacity; one conservative core if topology is unknown."""
    return cpu_limits()[0]
