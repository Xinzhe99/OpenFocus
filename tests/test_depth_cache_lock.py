"""Regression coverage for optional compiler cache lock startup."""
import errno
import os
import sys
from types import SimpleNamespace

from fusion_methods import _depth_kernel


class _WindowsOS:
    name = 'nt'

    def __getattr__(self, name):
        return getattr(os, name)


def test_cache_lock_retries_startup_and_initializes_after_lock(tmp_path, monkeypatch):
    platform_os = _WindowsOS()
    attempts = {'open': 0, 'lock': 0}
    held = []

    def open_file(*args):
        attempts['open'] += 1
        if attempts['open'] == 1:
            raise PermissionError(errno.EACCES, 'creation race')
        return os.open(*args)

    def locking(fd, mode, length):
        assert length == 1
        if mode == 1:
            attempts['lock'] += 1
            if attempts['lock'] == 1:
                raise OSError(errno.EACCES, 'contended')
        held.append(mode)

    def write(fd, data):
        assert held == [1]
        return os.write(fd, data)

    platform_os.open = open_file
    platform_os.write = write
    monkeypatch.setattr(_depth_kernel, 'os', platform_os)
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(
        LK_NBLCK=1, LK_UNLCK=2, locking=locking))
    monkeypatch.setattr(_depth_kernel.time, 'sleep', lambda _: None)

    with _depth_kernel._cache_access(tmp_path) as locked:
        assert locked
        assert held == [1]
    assert held == [1, 2]
    assert attempts == {'open': 2, 'lock': 2}
    assert (tmp_path / '.openfocus-numba.lock').read_bytes() == b'\0'


def test_cache_initialization_failure_skips_cache_and_releases_lock(tmp_path, monkeypatch):
    platform_os = _WindowsOS()
    modes = []
    cache_calls = []

    def fail_write(fd, data):
        raise OSError(errno.EIO, 'write failed')

    platform_os.write = fail_write
    monkeypatch.setattr(_depth_kernel, 'os', platform_os)
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(
        LK_NBLCK=1, LK_UNLCK=2,
        locking=lambda fd, mode, length: modes.append(mode)))
    cache = SimpleNamespace(cache_path=tmp_path,
                            load_overload=lambda *args: cache_calls.append(args))
    assert _depth_kernel._GuardedCache(cache).load_overload('key') is None
    assert cache_calls == []
    assert modes == [1, 2]


def test_cache_lock_contention_is_bounded_and_never_unlocks_unowned_byte(tmp_path, monkeypatch):
    platform_os = _WindowsOS()
    modes = []

    def locking(fd, mode, length):
        modes.append(mode)
        if mode == 1:
            raise OSError(errno.EACCES, 'contended')

    monkeypatch.setattr(_depth_kernel, 'os', platform_os)
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(
        LK_NBLCK=1, LK_UNLCK=2, locking=locking))
    monkeypatch.setattr(_depth_kernel.time, 'sleep', lambda _: None)
    with _depth_kernel._cache_access(tmp_path) as locked:
        assert not locked
    assert modes == [1] * 51
