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
import re
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


def _still_published(handle, path) -> bool:
    """True while *path* names the very file *handle* has open."""
    try:
        opened, named = os.fstat(handle.fileno()), os.stat(path)
    except OSError:
        return False
    return (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)


_OPEN_ATTEMPTS = 5


def _open_registry():
    """Create and lock this process's registry, or disable recording.

    A new registry is visible to other processes' sweeps before it is
    locked. A sweep that wins that race sees an unlocked, empty registry and
    removes it (or, on Windows, merely holds its lock for a moment), so the
    lock is retried under a fresh name until this process holds the lock on
    a file that is still published.
    """
    global _handle, _handle_path
    directory = registry_directory()
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in range(_OPEN_ATTEMPTS):
        path = directory / f"{os.getpid()}-{uuid.uuid4().hex[:12]}{_SUFFIX}"
        handle = open(path, "a+", **_ENCODING)
        if _try_lock(handle) and _still_published(handle, path):
            _handle, _handle_path = handle, path
            return
        handle.close()
        if os.path.exists(path):
            try:
                path.unlink()
            except OSError:
                pass
        time.sleep(0.01 * (attempt + 1))
    raise OSError(errno.EAGAIN, "could not lock a new scratch registry", str(directory))


def register(path) -> None:
    """Record a scratch file or folder so a crash cannot strand it.

    Call immediately after the path is created, before data is written.
    Failure to record never prevents processing.
    """
    global _disabled
    if _disabled:
        return
    line = os.fsdecode(path)
    if not os.path.isabs(line):
        return
    line = os.path.abspath(line)
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


def _safe_target(path: str, registered_directories=frozenset()):
    """Return a Path that may be removed, or None for anything unexpected."""
    if not path or not os.path.isabs(path):
        return None
    target = Path(path)
    if not target.name.startswith(_PREFIXES):
        parent = target.parent
        if (not re.fullmatch(r"video_[0-9]{5,}\.png", target.name)
                or not parent.name.startswith("openfocus_video_")
                or _norm(str(parent)) not in registered_directories
                or _safe_target(str(parent)) is None
                or not parent.is_dir()):
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
            target.rmdir()
            size = 0
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
            except FileNotFoundError:
                continue
            except OSError:
                return None
            try:
                if _try_lock(probe):
                    continue  # stale, not live
            finally:
                probe.close()
        try:
            live.update(line for line in entry.read_text(**_ENCODING).splitlines() if line)
        except (OSError, ValueError):
            return None
    return live


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _owned_by_live(target: Path, live) -> bool:
    """True when a live registry lists target, a folder holding it, or a path inside it."""
    path = _norm(str(target))
    for owned in live:
        if path == owned:
            return True
        inner, outer = (path, owned) if len(path) > len(owned) else (owned, path)
        if inner.startswith(outer.rstrip(os.sep) + os.sep):
            return True
    return False


def _sweep_stale_registries(directory: Path, live=frozenset()):
    """Remove what exited processes recorded, except what a live one owns.

    A path in both a stale and a live registry was handed over (a render
    worker's results adopted by the window) without the stale registry
    forgetting it; the live owner records and removes it itself.
    """
    live = {_norm(path) for path in live if path}
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
            directories = {_norm(line) for line in lines
                           if os.path.isabs(line) and
                           Path(line).name.startswith("openfocus_video_")}
            # Most recent first: folders are recorded before files inside them.
            for line in reversed(lines):
                target = _safe_target(line, directories)
                if target is not None and not _owned_by_live(target, live):
                    size = _remove(target)
                    if target.exists():
                        leftover = True  # e.g. locked by antivirus; retry next time
                    else:
                        removed += 1
                        freed += size
            if not leftover and os.name != "nt":
                # Unlink before unlocking: once the lock is released, a new
                # owner could lock this file just before it disappears.
                # (Windows cannot delete an open file; there the owner keeps
                # it open, so the unlink below fails for a live registry.)
                _unlink_quietly(entry)
        except (OSError, ValueError):
            leftover = True
        finally:
            probe.close()
        if not leftover and os.name == "nt":
            _unlink_quietly(entry)
    return removed, freed


def _unlink_quietly(entry: Path) -> None:
    try:
        entry.unlink()
    except OSError:
        pass


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
            return _windows_boot_time(ctypes.windll.kernel32)
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


def _windows_boot_time(kernel32):
    """Boot time from GetTickCount64.

    ctypes assumes a signed 32-bit int unless told otherwise, which turns the
    uptime negative after about 25 days (and wraps every 49.7), putting
    "boot" in the future so every preview spill looks pre-boot.
    """
    import ctypes
    function = kernel32.GetTickCount64
    function.restype = ctypes.c_ulonglong
    function.argtypes = ()
    return time.time() - function() / 1000.0


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
                except FileNotFoundError:
                    continue
                except OSError:
                    return None
            with open(f"{base}/maps", encoding="utf-8", errors="replace") as maps:
                for line in maps:
                    parts = line.split(None, 5)
                    if len(parts) >= 5 and parts[4] != "0":
                        busy.add(int(parts[4]))
        except FileNotFoundError:
            continue  # process exited during the scan
        except (OSError, ValueError):
            return None  # incomplete visibility cannot prove abandonment
    return busy


def _sweep_legacy(directories, live, registry=None):
    """Loose scratch from builds that predate the registry.

    *live* is read before the folders are listed, so it can miss a frame a
    running process creates and records during the sweep. With *registry*,
    the live registries are read again after the process scan; see below.
    """
    candidates = []
    for directory in directories:
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            if (entry.name.startswith(_LEGACY_PREFIXES) and entry.is_file(follow_symlinks=False)
                    and not _owned_by_live(Path(os.path.abspath(entry.path)), live)):
                candidates.append(entry.path)
    if not candidates:
        return 0, 0
    boot = _boot_time()
    busy = _open_by_any_process()
    if registry is not None:
        # A new frame's descriptor is closed before the file is mapped, so a
        # process scan that lands in between finds it open nowhere. Its owner
        # records it before closing the descriptor, so registries read after
        # the scan always list it.
        recorded = _live_registry_paths(registry)
        if recorded is None:
            return 0, 0
        live = set(live) | {_norm(path) for path in recorded if path}
    removed, freed = 0, 0
    for path in candidates:
        target = _safe_target(os.path.abspath(path))
        if target is None or _owned_by_live(target, live):
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
    live = _live_registry_paths(directory)
    if live is None:
        return 0, 0
    live = {_norm(path) for path in live if path}
    removed, freed = _sweep_stale_registries(directory, live)
    folders = {os.path.abspath(tempfile.gettempdir())}
    folders.update(os.path.abspath(os.fsdecode(d)) for d in extra_directories if d)
    legacy = _sweep_legacy(sorted(folders), live, directory)
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
