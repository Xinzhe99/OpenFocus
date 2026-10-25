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


_PHYSICAL_CORES = None


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
    raw = bytes(buffer[:length.value])
    cores = 0
    offset = 0
    while offset + 8 <= len(raw):
        relationship, size = struct.unpack_from("<II", raw, offset)
        if size < 8 or offset + size > len(raw):
            break
        if relationship == RELATION_PROCESSOR_CORE:
            cores += 1
        offset += size
    return cores


def _linux_physical_cores() -> int:
    """/proc/cpuinfo (physical id, core id) pairs, else the affinity count."""
    try:
        pairs = set()
        physical_id = core_id = None
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                key = key.strip()
                if key == "physical id":
                    physical_id = value.strip()
                elif key == "core id":
                    core_id = value.strip()
                elif not line.strip():
                    if physical_id is not None and core_id is not None:
                        pairs.add((physical_id, core_id))
                    physical_id = core_id = None
        if pairs:
            return len(pairs)
    except OSError:
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
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


def physical_core_count() -> int:
    """Physical cores (SMT siblings counted once).

    Tile fusion is memory-bandwidth-bound: hyper-threads add peak memory
    without adding throughput, which is why Helicon-style engines size their
    worker pool from physical cores. A mixed P/E laptop has no SMT on its
    efficiency cores, so ``logical // 2`` undercounts it (an i7-12700 has 12
    physical cores behind 20 logical processors); the platform API answers
    the real topology and the halved count is only the fallback.
    """
    global _PHYSICAL_CORES
    if _PHYSICAL_CORES is not None:
        return _PHYSICAL_CORES
    logical = os.cpu_count() or 4
    detected = 0
    try:
        if is_windows():
            detected = _windows_physical_cores()
        elif is_linux():
            detected = _linux_physical_cores()
        elif is_macos():
            detected = _macos_physical_cores()
    except Exception:
        detected = 0
    if detected <= 0:
        detected = max(1, logical // 2)
    _PHYSICAL_CORES = max(1, min(int(detected), logical))
    return _PHYSICAL_CORES
