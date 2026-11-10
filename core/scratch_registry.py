"""Crash-safe bookkeeping for scratch files and folders.

Frame mappings, preview spills and job folders are normally removed when
their owners are released. A process killed by a native fault (for example
SIGBUS when a disk fills) never runs those finalizers, so its scratch used to
stay on disk until the user found it by hand.

Each process that creates scratch keeps one registry file under
``user_data_dir()/scratch-sessions`` and holds an exclusive OS lock on it for
its lifetime. Every scratch path is appended to that file before pixels are
written. At startup, a registry whose lock can be taken belongs to a process
that has exited, so the paths it lists are removed. A live process holds its
lock, so concurrent GUI, CLI and control-tool processes never remove each
other's scratch, regardless of PID namespaces.

Scratch written by builds that predate the registry is removed only when it
provably belongs to no running process: it was last modified before the
system booted, or (on Linux) no process has it open or mapped.
"""
from __future__ import annotations

import atexit
import errno
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger("openfocus")

_SUFFIX = ".scratch"
# Lossless for any POSIX path bytes, including names that are not valid UTF-8.
_ENCODING = dict(encoding="utf-8", errors="surrogateescape")
# Only paths with these basenames are ever removed, whatever a registry says.
_PREFIXES = ("openfocus-frame-", "openfocus-preview-", "openfocus-grouped-",
             "openfocus_video_", "openfocus-")
# Loose files written by builds without a registry.
_LEGACY_PREFIXES = ("openfocus-frame-", "openfocus-preview-")
_ALWAYS_OPEN_PREFIXES = ("openfocus-frame-",)

_lock = threading.Lock()
_handle = None          # open, locked registry file of this process
_handle_path = None
_disabled = False       # registry unavailable: scratch still works, unrecorded
_directory_override = None  # tests


def registry_directory() -> Path:
    if _directory_override is not None:
        return Path(_directory_override)
    from utils.app_paths import user_data_dir
    return user_data_dir() / "scratch-sessions"


# Windows byte-range locks also block reads of the locked range, so lock one
# byte far beyond any data; other processes can still read the paths.
_WINDOWS_LOCK_OFFSET = 0x7FFFFFF0


def _try_lock(handle) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False
    finally:
        try:
            handle.seek(0)
        except (OSError, ValueError):
            pass


def _open_registry():
    """Create and lock this process's registry, or disable recording."""
    global _handle, _handle_path, _disabled
    directory = registry_directory()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{os.getpid()}-{uuid.uuid4().hex[:12]}{_SUFFIX}"
    handle = open(path, "a+", **_ENCODING)
    if not _try_lock(handle):
        handle.close()
        path.unlink(missing_ok=True)
        raise OSError(errno.EAGAIN, "could not lock new scratch registry", str(path))
    _handle, _handle_path = handle, path


def register(path) -> None:
    """Record a scratch file or folder so a crash cannot strand it.

    Call immediately after the path is created, before data is written.
    Failure to record never prevents processing.
    """
    global _disabled
    if _disabled:
        return
    line = os.path.abspath(os.fsdecode(path))
    if "\n" in line or "\r" in line:
        return
    with _lock:
        try:
            if _handle is None:
                _open_registry()
            _handle.seek(0, os.SEEK_END)
            _handle.write(line + "\n")
            _handle.flush()
        except (OSError, ValueError) as exc:
            _disabled = True
            log.warning("scratch registry unavailable; crash leftovers will not be "
                        "removed automatically: %s", exc)


def forget(paths) -> None:
    """Stop recording paths that another process has taken over.

    A file handed to another process (a render worker's results, adopted by
    the window) must leave this registry: after a clean exit the registry
    stays on disk, unlocked, while anything it lists still exists, and the
    next startup would remove scratch the new owner is still using.  The new
    owner records the paths in its own registry.
    """
    if isinstance(paths, (str, bytes, os.PathLike)):
        paths = (paths,)
    drop = {os.path.abspath(os.fsdecode(path)) for path in paths}
    if not drop:
        return
    with _lock:
        if _handle is None:
            return
        try:
            _handle.seek(0)
            kept = [line for line in _handle.read().splitlines() if line and line not in drop]
            _handle.seek(0)
            _handle.truncate()
            _handle.write("".join(line + "\n" for line in kept))
            _handle.flush()
        except (OSError, ValueError) as exc:
            log.warning("scratch registry: could not forget handed-over scratch: %s", exc)


def _safe_target(path: str):
    """Return a Path that may be removed, or None for anything unexpected."""
    if not path or not os.path.isabs(path):
        return None
    target = Path(path)
    if not target.name.startswith(_PREFIXES):
        return None
    try:
        info = target.lstat()
    except OSError:
        return None
    import stat
    if stat.S_ISLNK(info.st_mode):
        return None
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        return None
    return target


def _remove(target: Path) -> int:
    """Remove a file or folder, returning bytes freed (best effort)."""
    try:
        if target.is_dir():
            size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()
                       and not p.is_symlink())
            shutil.rmtree(target)
        else:
            size = target.stat().st_size
            target.unlink()
        return size
    except OSError:
        return 0


def _live_registry_paths(directory: Path) -> set:
    """Paths recorded by registries whose owner is still running."""
    live = set()
    for entry in directory.glob(f"*{_SUFFIX}"):
        if entry != _handle_path:
            try:
                probe = open(entry, "r+", **_ENCODING)
            except OSError:
                continue
            try:
                if _try_lock(probe):
                    continue  # stale, not live
            finally:
                probe.close()
        try:
            live.update(line.strip() for line in entry.read_text(**_ENCODING).splitlines())
        except (OSError, ValueError):
            pass
    return live


def _sweep_stale_registries(directory: Path):
    removed, freed = 0, 0
    for entry in directory.glob(f"*{_SUFFIX}"):
        if _handle_path is not None and entry == _handle_path:
            continue
        try:
            probe = open(entry, "r+", **_ENCODING)
        except OSError:
            continue
        leftover = False
        try:
            if not _try_lock(probe):
                continue  # owner is alive
            lines = probe.read().splitlines()
            # Most recent first: folders are recorded before files inside them.
            for line in reversed(lines):
                target = _safe_target(line.strip())
                if target is not None:
                    size = _remove(target)
                    if target.exists():
                        leftover = True  # e.g. locked by antivirus; retry next time
                    else:
                        removed += 1
                        freed += size
        except (OSError, ValueError):
            leftover = True
        finally:
            probe.close()
        if leftover:
            continue
        try:
            entry.unlink()
        except OSError:
            pass
    return removed, freed


def _boot_time():
    """Wall-clock system boot time, or None when unknown."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/stat", encoding="ascii") as stat_file:
                for line in stat_file:
                    if line.startswith("btime "):
                        return float(line.split()[1])
        elif os.name == "nt":
            import ctypes
            ticks = ctypes.windll.kernel32.GetTickCount64()
            return time.time() - ticks / 1000.0
        elif sys.platform == "darwin":
            import ctypes
            import ctypes.util

            class _Timeval(ctypes.Structure):
                _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_int32)]

            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            value = _Timeval()
            size = ctypes.c_size_t(ctypes.sizeof(value))
            if libc.sysctlbyname(b"kern.boottime", ctypes.byref(value),
                                 ctypes.byref(size), None, 0) == 0:
                return float(value.tv_sec)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _open_by_any_process():
    """Inodes of files open or mapped by any visible process (Linux only).

    Device numbers are deliberately ignored: btrfs reports different devices
    in /proc/*/maps and stat(). An inode shared by two filesystems can only
    make a file look busy, which keeps it.
    """
    if not sys.platform.startswith("linux") or not os.path.isdir("/proc/self/fd"):
        return None
    busy = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        base = f"/proc/{pid}"
        try:
            for fd in os.listdir(f"{base}/fd"):
                try:
                    busy.add(os.stat(f"{base}/fd/{fd}").st_ino)
                except OSError:
                    pass
            with open(f"{base}/maps", encoding="utf-8", errors="replace") as maps:
                for line in maps:
                    parts = line.split(None, 5)
                    if len(parts) >= 5 and parts[4] != "0":
                        busy.add(int(parts[4]))
        except (OSError, ValueError):
            continue  # exited, or another user's process
    return busy


def _sweep_legacy(directories, live):
    """Loose scratch from builds that predate the registry."""
    candidates = []
    for directory in directories:
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            if (entry.name.startswith(_LEGACY_PREFIXES) and entry.is_file(follow_symlinks=False)
                    and os.path.abspath(entry.path) not in live):
                candidates.append(entry.path)
    if not candidates:
        return 0, 0
    boot = _boot_time()
    busy = _open_by_any_process()
    removed, freed = 0, 0
    for path in candidates:
        target = _safe_target(os.path.abspath(path))
        if target is None:
            continue
        try:
            info = target.stat()
        except OSError:
            continue
        before_boot = boot is not None and info.st_mtime < boot - 60
        # Frame files stay open or mapped for their whole life, so "not open
        # anywhere" proves abandonment. Preview spills are reopened by path on
        # demand, so only a reboot proves that; they are small.
        unused = (busy is not None and target.name.startswith(_ALWAYS_OPEN_PREFIXES)
                  and info.st_ino not in busy)
        if before_boot or unused:
            size = _remove(target)
            if size or not target.exists():
                removed += 1
                freed += size
    return removed, freed


def sweep_abandoned(extra_directories=()):
    """Remove scratch left by processes that ended without cleaning up.

    Safe to call while other OpenFocus processes run. Returns (count, bytes).
    """
    try:
        directory = registry_directory()
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return 0, 0
    removed, freed = _sweep_stale_registries(directory)
    folders = {os.path.abspath(tempfile.gettempdir())}
    folders.update(os.path.abspath(os.fsdecode(d)) for d in extra_directories if d)
    legacy = _sweep_legacy(sorted(folders), _live_registry_paths(directory))
    removed, freed = removed + legacy[0], freed + legacy[1]
    if removed:
        log.info("removed %d abandoned scratch item(s), %.1f MiB, left by an "
                 "earlier OpenFocus process", removed, freed / 2 ** 20)
    return removed, freed


def sweep_in_background(extra_directories=()):
    thread = threading.Thread(target=_sweep_quietly, args=(tuple(extra_directories),),
                              name="scratch-sweep", daemon=True)
    thread.start()
    return thread


def _sweep_quietly(extra_directories):
    try:
        sweep_abandoned(extra_directories)
    except Exception:
        log.exception("scratch sweep failed")


def _close_at_exit():
    """Drop the registry when everything it lists is already gone.

    Anything still present (for example a mapping Windows would not let us
    delete) keeps the file, now unlocked, so the next startup removes it.
    """
    global _handle, _handle_path
    with _lock:
        if _handle is None:
            return
        try:
            _handle.seek(0)
            remaining = [line for line in _handle.read().splitlines()
                         if line and os.path.lexists(line)]
            _handle.close()
            if not remaining:
                _handle_path.unlink(missing_ok=True)
        except OSError:
            pass
        _handle = _handle_path = None


atexit.register(_close_at_exit)
