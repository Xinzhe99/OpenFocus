"""Best-effort precompilation of the shared CPU depth recurrence."""
import os
from pathlib import Path
import sys


def depth_cache_revision():
    """Identify the cache warmup for this installation and compiler setup."""
    program = (Path(sys.executable) if getattr(sys, 'frozen', False)
               else Path(__file__).with_name('_depth_kernel.py'))
    try:
        stat = program.stat()
    except OSError:
        return None
    options = tuple(os.environ.get(key) for key in (
        'NUMBA_CACHE_DIR', 'NUMBA_CPU_NAME', 'NUMBA_CPU_FEATURES'))
    return repr((str(program), stat.st_mtime_ns, stat.st_size,
                 sys.implementation.cache_tag, options))


def warm_depth_cache():
    """Run only in the disposable desktop warmup process."""
    try:
        if sys.platform == 'win32':
            from ctypes import windll
            from ctypes.wintypes import BOOL, DWORD, HANDLE
            kernel = windll.kernel32
            kernel.GetCurrentProcess.restype = HANDLE
            kernel.SetPriorityClass.argtypes = (HANDLE, DWORD)
            kernel.SetPriorityClass.restype = BOOL
            if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
                return False
        else:
            os.nice(10)
        from core.algorithm_adapters import DEPTH_KERNEL_STARTUP_BYTES
        from core.resource_policy import probe_memory, ram_budget
        if ram_budget(probe_memory()) < 2 * DEPTH_KERNEL_STARTUP_BYTES:
            return False
        from fusion_methods._depth_kernel import axis_block, prepare_depth_kernel
        if getattr(axis_block._cache, 'cache_path', None) is None:
            return False
        prepare_depth_kernel()
        return True
    except (OSError, ImportError, MemoryError, RuntimeError):
        return False
