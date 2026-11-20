"""Launch an isolated interpreter from a source or PyInstaller process."""
import os
import subprocess
import sys
import threading


_DLL_LOCK = threading.Lock()


def _spawn_posix(command, kwargs):
    import fcntl

    # Keep a shared lock on the interpreter open in the worker itself so a
    # concurrent repair cannot discard the generation or its lazy imports.
    with open(command[0], 'rb') as binary:
        fd = binary.fileno()
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        pass_fds = (*kwargs.pop('pass_fds', ()), fd)
        return subprocess.Popen(command, pass_fds=pass_fds, **kwargs)


def spawn(command, **kwargs):
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith(('PYTHON', '_PYI_')) or key in ('VIRTUAL_ENV', '__PYVENV_LAUNCHER__'):
            del env[key]
    bundle = getattr(sys, '_MEIPASS', None)
    if bundle:
        root = os.path.normcase(os.path.abspath(bundle))
        for key in ('PATH', 'DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH'):
            if key + '_ORIG' in env:
                env[key] = env.pop(key + '_ORIG')
            elif key in env:
                paths = env[key].split(os.pathsep)
                env[key] = os.pathsep.join(p for p in paths if p and not (
                    os.path.normcase(os.path.abspath(p)) + os.sep).startswith(root + os.sep))
    kwargs.setdefault('cwd', os.path.dirname(command[0]))
    kwargs.setdefault('env', env)
    if sys.platform != 'win32':
        return _spawn_posix(command, kwargs)
    kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
    if not bundle:
        return subprocess.Popen(command, **kwargs)

    # PyInstaller's Win32 DLL directory is inherited independently of PATH.
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    with _DLL_LOCK:
        size = kernel.GetDllDirectoryW(0, None)
        previous = ctypes.create_unicode_buffer(size or 1)
        kernel.GetDllDirectoryW(len(previous), previous)
        if not kernel.SetDllDirectoryW(None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return subprocess.Popen(command, **kwargs)
        finally:
            kernel.SetDllDirectoryW(previous.value or None)
