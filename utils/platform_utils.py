"""
跨平台工具模块 - 处理 Windows/macOS/Linux 兼容性
"""

import os
from functools import lru_cache
import platform
import struct
import sys

from constants import NATIVE_WORKER_STACK_BYTES


def get_os_type() -> str:
    """
    获取当前操作系统类型
    
    Returns:
        'windows' | 'macos' | 'linux'
    """
    if sys.platform.startswith(('win32', 'cygwin')):
        return 'windows'
    if sys.platform == 'darwin':
        return 'macos'
    if sys.platform.startswith('linux'):
        return 'linux'
    system = platform.system().lower()
    if system == 'darwin':
        return 'macos'
    elif system == 'windows':
        return 'windows'
    elif system == 'linux':
        return 'linux'
    else:
        return 'unknown'


_UI_FONT_FAMILIES = {
    'macos': ('SF Pro', 'Helvetica Neue', 'Arial'),
    'windows': ('Segoe UI', 'Microsoft YaHei', 'Arial'),
    'linux': ('Ubuntu', 'DejaVu Sans', 'Arial'),
}

_MONOSPACE_FONT_FAMILIES = {
    'macos': ('SF Mono', 'Monaco', 'Menlo', 'Consolas'),
    'windows': ('Consolas', 'Courier New'),
    'linux': ('Ubuntu Mono', 'DejaVu Sans Mono', 'Consolas'),
}


def _format_css_font_stack(families: tuple[str, ...], generic: str) -> str:
    quoted = [f'"{family}"' if family not in ('Arial', 'Consolas') else family for family in families]
    return ', '.join((*quoted, generic))


def get_ui_font_families() -> tuple[str, ...]:
    """Return ordered QFont-compatible UI font families for the active OS."""
    return _UI_FONT_FAMILIES.get(get_os_type(), _UI_FONT_FAMILIES['linux'])


def get_monospace_font_families() -> tuple[str, ...]:
    """Return ordered QFont-compatible monospace font families for the active OS."""
    return _MONOSPACE_FONT_FAMILIES.get(get_os_type(), _MONOSPACE_FONT_FAMILIES['linux'])


def get_ui_font_family() -> str:
    """
    获取适合当前操作系统的 UI 字体族
    
    Returns:
        字体族名字符串
    """
    return _format_css_font_stack(get_ui_font_families(), 'sans-serif')


def get_monospace_font_family() -> str:
    """
    获取适合当前操作系统的等宽字体族
    
    Returns:
        字体族名字符串
    """
    return _format_css_font_stack(get_monospace_font_families(), 'monospace')


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


def configure_native_worker_stack(worker) -> None:
    """Provide macOS workers enough stack for native numerical libraries."""
    if is_macos():
        worker.setStackSize(max(worker.stackSize(), NATIVE_WORKER_STACK_BYTES))


# 导出常用函数
__all__ = [
    'get_os_type',
    'get_ui_font_families',
    'get_monospace_font_families',
    'get_ui_font_family',
    'get_monospace_font_family',
    'get_default_font',
    'is_windows',
    'is_macos',
    'is_linux',
    'configure_native_worker_stack',
    'cached_compute_gpu',
    'has_compute_gpu',
    'cpu_limits',
    'physical_core_count',
]


def _neural_accelerator_present():
    """Return verified AI-provider capability, or None before probing."""
    from core.multi_focus_fusion import neural_device_kind
    from fusion_methods import neural_runtime
    kind = neural_device_kind()
    return None if kind is None else neural_runtime.is_accelerator(kind)


def cached_compute_gpu():
    """Nonblocking verified AI capability; hardware presence is independent.

    A display adapter, vendor tool or installed execution provider does not
    prove that the bundled model can run on that provider. In particular,
    integrated and discrete adapters must pass the same inference probe.
    """
    try:
        return _neural_accelerator_present()
    except Exception:
        return None


def has_compute_gpu() -> bool:
    """Probe supported AI execution when necessary; never use GPU-name hints.

    This can initialize a driver and run a small ONNX model. UI and diagnostic
    paths must use cached_compute_gpu() and the existing background probe.
    True describes provider capability, not execution of the current task.
    """
    present = cached_compute_gpu()
    if present is not None:
        return present
    from core.multi_focus_fusion import ensure_neural_device_kind
    from fusion_methods import neural_runtime
    return neural_runtime.is_accelerator(ensure_neural_device_kind())


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


def _parse_cpu_list(text: str) -> tuple[int, ...]:
    """Parse Linux sysfs cpulist syntax (e.g. '0-3,6,8-9') into CPU IDs."""
    # Binary sysfs topology attributes can include the kernel's trailing NUL.
    text = text.removesuffix("\0").strip()
    if not text:
        return ()
    cpus = set()
    for part in text.split(","):
        bounds = part.strip().split("-", 1)
        if any(not (bound.isascii() and bound.isdecimal()) for bound in bounds):
            raise ValueError(f"Invalid CPU list entry: {part!r}")
        start, end = int(bounds[0]), int(bounds[-1])
        if end < start:
            raise ValueError(f"Invalid CPU range: {part!r}")
        cpus.update(range(start, end + 1))
    return tuple(sorted(cpus))


def _linux_sysfs_physical_cores(affinity=None, sys_cpu_root: str = "/sys/devices/system/cpu") -> int:
    """Count usable physical cores from Linux sysfs when /proc lacks IDs."""
    if affinity is None:
        try:
            with open(os.path.join(sys_cpu_root, "online"), encoding="ascii") as fh:
                cpus = set(_parse_cpu_list(fh.read()))
        except OSError:
            try:
                entries = os.listdir(sys_cpu_root)
            except OSError:
                return 0
            cpus = {int(name[3:]) for name in entries
                    if name.startswith("cpu") and name[3:].isascii() and name[3:].isdecimal()}
        except ValueError:
            return 0
    else:
        cpus = set(affinity)

    remaining = cpus.copy()
    cores = 0
    for cpu_id in cpus:
        if cpu_id not in remaining:
            continue
        topo = os.path.join(sys_cpu_root, f"cpu{cpu_id}", "topology")
        for sibling_file in ("core_cpus_list", "thread_siblings_list"):
            try:
                with open(os.path.join(topo, sibling_file), encoding="ascii") as fh:
                    siblings = cpus.intersection(_parse_cpu_list(fh.read()))
            except (OSError, ValueError):
                continue
            if cpu_id not in siblings:
                continue
            if not siblings <= remaining:
                return 0
            remaining.difference_update(siblings)
            cores += 1
            break

    # A sibling list represents the whole core, even if a sibling's own
    # topology files are unreadable. Use IDs only for CPUs still uncovered.
    id_groups = set()
    for cpu_id in remaining:
        topo = os.path.join(sys_cpu_root, f"cpu{cpu_id}", "topology")
        try:
            with open(os.path.join(topo, "physical_package_id"), encoding="ascii") as fh:
                package = int(fh.read())
            with open(os.path.join(topo, "core_id"), encoding="ascii") as fh:
                core = int(fh.read())
            if package < 0 or core < 0:
                return 0
            domain = []
            for attribute in ("die_id", "cluster_id"):
                try:
                    with open(os.path.join(topo, attribute), encoding="ascii") as fh:
                        domain.append(int(fh.read()))
                except FileNotFoundError:
                    domain.append(-1)
        except (OSError, ValueError):
            return 0
        id_groups.add((package, *domain, core))
    return cores + len(id_groups)


def _linux_physical_cores() -> int:
    """Count physical cores available to the calling thread's affinity."""
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
            package = int(fields["physical id"])
            core = int(fields["core id"])
            if package < 0 or core < 0:
                break
            pairs.add((package, core))
        else:
            if pairs:
                return len(pairs)
    except (KeyError, OSError, ValueError):
        pass
    return _linux_sysfs_physical_cores(affinity)


def _macos_physical_cores() -> int:
    """sysctl hw.physicalcpu (Apple silicon reports P+E cores)."""
    import ctypes
    try:
        libc = ctypes.CDLL(None)
        libc.sysctlbyname.argtypes = (
            ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p, ctypes.c_size_t,
        )
        libc.sysctlbyname.restype = ctypes.c_int
        value = ctypes.c_int()
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if (libc.sysctlbyname(b"hw.physicalcpu", ctypes.byref(value),
                              ctypes.byref(size), None, 0) != 0
                or size.value != ctypes.sizeof(value)):
            return 0
        return max(0, value.value)
    except (AttributeError, OSError):
        return 0


def _linux_cgroup_paths(controller):
    """Yield this cgroup and its ancestors on matching v1/v2 mounts."""
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as fh:
            memberships = fh.read()
        with open("/proc/self/mountinfo", encoding="utf-8") as fh:
            mounts = fh.read()
    except OSError:
        return
    yield from _parse_linux_cgroup_paths(controller, memberships, mounts)


@lru_cache(maxsize=8)
def _parse_linux_cgroup_paths(controller, membership_text, mount_text):
    """Reuse parsing only for identical live membership and mount snapshots.

    Limits and occupancy remain uncached. Including both proc snapshots in the
    key also observes cgroup migration and mount changes on the next probe.
    """
    from pathlib import PurePosixPath
    # Only "\n" ends a record: cgroup names may hold the other characters
    # str.splitlines() treats as line boundaries.
    memberships = [line.strip().split(":", 2) for line in membership_text.split("\n")]
    mounts = mount_text.split("\n")
    paths = []
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
                    paths.append(key)
                if current == mount:
                    break
                current = current.parent

    return tuple(paths)


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
