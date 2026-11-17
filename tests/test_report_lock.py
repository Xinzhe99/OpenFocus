"""Regression coverage for report lock creation and ownership."""
import errno
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from utils import report_client


class _WindowsOS:
    name = 'nt'

    def __getattr__(self, name):
        return getattr(os, name)


def test_lock_retries_first_creation_and_initializes_under_lock(tmp_path, monkeypatch):
    directory = tmp_path / 'state'
    real_mkdir = type(directory).mkdir
    real_open = os.open
    attempts = {'mkdir': 0, 'open': 0}
    held = []
    platform_os = _WindowsOS()

    def mkdir(path, *args, **kwargs):
        if path == directory:
            attempts['mkdir'] += 1
            if attempts['mkdir'] == 1:
                raise PermissionError(errno.EACCES, 'creation race')
        return real_mkdir(path, *args, **kwargs)

    def open_file(*args):
        attempts['open'] += 1
        if attempts['open'] == 1:
            raise PermissionError(errno.EACCES, 'creation race')
        return real_open(*args)

    def locking(fd, mode, length):
        assert length == 1
        held.append(mode)

    def fstat(fd):
        assert held == [1]
        return os.fstat(fd)

    platform_os.open = open_file
    platform_os.fstat = fstat
    monkeypatch.setattr(type(directory), 'mkdir', mkdir)
    monkeypatch.setattr(report_client, 'os', platform_os)
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(
        LK_NBLCK=1, LK_UNLCK=2, locking=locking))
    monkeypatch.setattr(report_client.time, 'sleep', lambda _: None)

    with report_client._file_lock(directory, 'queue.lock'):
        assert held == [1]
    assert held == [1, 2]
    assert attempts == {'mkdir': 2, 'open': 2}
    assert (directory / 'queue.lock').read_bytes() == b'\0'


@pytest.mark.parametrize('stage', ['mkdir', 'open'])
def test_creation_retry_honors_cancellation(tmp_path, monkeypatch, stage):
    cancelled = threading.Event()
    platform_os = _WindowsOS()
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        cancelled.set()
        raise PermissionError(errno.EACCES, 'creation race')

    if stage == 'mkdir':
        monkeypatch.setattr(type(tmp_path), 'mkdir', fail)
    else:
        platform_os.open = fail
    monkeypatch.setattr(report_client, 'os', platform_os)
    with pytest.raises(report_client.ReportError, match='cancelled'):
        with report_client._file_lock(tmp_path, 'queue.lock', cancel=cancelled):
            pytest.fail('cancelled operation acquired the lock')
    assert len(calls) == 1


def test_creation_permission_failure_is_bounded(tmp_path, monkeypatch):
    platform_os = _WindowsOS()
    calls = []

    def fail(*args):
        calls.append(True)
        raise PermissionError(errno.EACCES, 'permission denied')

    platform_os.open = fail
    monkeypatch.setattr(report_client, 'os', platform_os)
    monkeypatch.setattr(report_client.time, 'sleep', lambda _: None)
    with pytest.raises(report_client.ReportError, match='Could not lock'):
        with report_client._file_lock(tmp_path, 'queue.lock'):
            pytest.fail('unavailable lock was acquired')
    assert len(calls) == 11


def test_creation_retry_honors_deadline(tmp_path, monkeypatch):
    platform_os = _WindowsOS()
    clock = [0]

    def fail(*args):
        clock[0] = 2
        raise PermissionError(errno.EACCES, 'creation race')

    platform_os.open = fail
    monkeypatch.setattr(report_client, 'os', platform_os)
    monkeypatch.setattr(report_client, 'time', SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda _: pytest.fail('deadline expired')))
    with pytest.raises(report_client.ReportError, match='cancelled'):
        with report_client._file_lock(tmp_path, 'queue.lock', deadline=1):
            pytest.fail('expired operation acquired the lock')


def test_initialization_failure_releases_file_and_local_locks(tmp_path, monkeypatch):
    platform_os = _WindowsOS()
    modes = []
    calls = []

    def fstat(fd):
        calls.append(True)
        if len(calls) == 1:
            raise OSError(errno.EIO, 'initialization failed')
        return os.fstat(fd)

    platform_os.fstat = fstat
    monkeypatch.setattr(report_client, 'os', platform_os)
    monkeypatch.setitem(sys.modules, 'msvcrt', SimpleNamespace(
        LK_NBLCK=1, LK_UNLCK=2,
        locking=lambda fd, mode, length: modes.append(mode)))
    with pytest.raises(report_client.ReportError, match='Could not lock'):
        with report_client._file_lock(tmp_path, 'queue.lock'):
            pytest.fail('initialization failed')
    with report_client._file_lock(tmp_path, 'queue.lock'):
        assert modes == [1, 2, 1]
    assert modes == [1, 2, 1, 2]
