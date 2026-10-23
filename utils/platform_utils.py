"""
跨平台工具模块 - 处理 Windows/macOS/Linux 兼容性
"""

import os
import sys
import platform


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


def physical_core_count() -> int:
    """Physical cores (SMT siblings counted once).

    Tile fusion is memory-bandwidth-bound: hyper-threads add peak memory
    without adding throughput, so this application sizes its
    worker pool from physical cores. Modern multi-core desktops are all SMT,
    so the logical count halved is the reliable estimate (win32 structure
    walking was tried and misread stride sizes on x64); floor 1.
    """
    global _PHYSICAL_CORES
    if _PHYSICAL_CORES is not None:
        return _PHYSICAL_CORES
    logical = os.cpu_count() or 4
    _PHYSICAL_CORES = max(1, logical // 2)
    return _PHYSICAL_CORES
