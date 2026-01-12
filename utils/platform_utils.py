"""
跨平台工具模块 - 处理 Windows/macOS/Linux 兼容性
"""

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
