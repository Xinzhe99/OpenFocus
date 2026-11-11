"""Offline concurrency tests for report permission and queue transactions."""
import json
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from utils.report_client import (ReportClient, ReportError, automatic_snapshot,
                                 revoke_automatic)


class Response:
    def __init__(self, status, value):
        self.status = status
        self.value = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return self.value[:limit]


class Server:
    def __init__(self, block=None, entered=None, release=None, fail_put=False):
        self.requests = []
        self.block, self.entered, self.release = block, entered, release
        self.fail_put = fail_put

    def open(self, request, timeout):
        self.requests.append((request.method, request.data))
        if request.method == self.block:
            self.entered.set()
            if not self.release.wait(5):
                raise AssertionError('request barrier timed out')
        if request.method == 'POST':
            return Response(201, {'id': 'report', 'upload_token': 'secret'})
        if self.fail_put:
            raise OSError('simulated upload failure')
        return Response(200, {'ok': True, 'bytes': len(request.data)})


def _enqueue_process(directory, ready, start, result, marker):
    try:
        client = ReportClient(directory)
        ready.put(True)
        if not start.wait(10):
            raise RuntimeError('start barrier timed out')
        client.enqueue({'source': 'manual', 'subject': marker}, marker.encode())
        result.put(('ok', client.installation_id))
    except Exception as exc:
        result.put(('error', repr(exc)))


def _retry_process(directory, ready, start, requests, result):
    class ProcessServer:
        def open(self, request, timeout):
            requests.put(request.method)
            if request.method == 'POST':
                return Response(201, {'id': 'report', 'upload_token': 'secret'})
            return Response(200, {'ok': True, 'bytes': len(request.data)})
    try:
        client = ReportClient(directory)
        client._opener = ProcessServer()
        ready.put(True)
        if not start.wait(10):
            raise RuntimeError('start barrier timed out')
        result.put(('ok', client.retry_pending()))
    except Exception as exc:
        result.put(('error', repr(exc)))


def _hold_delivery(directory, acquired, release):
    client = ReportClient(directory)
    with client._delivery_lock():
        acquired.set()
        release.wait(10)


def _blocked_retry_process(directory, entered, release, requests, result):
    class ProcessServer:
        def open(self, request, timeout):
            requests.put(request.method)
            if request.method == 'POST':
                entered.set()
                if not release.wait(10):
                    raise RuntimeError('request barrier timed out')
                return Response(201, {'id': 'report', 'upload_token': 'secret'})
            return Response(200, {'ok': True, 'bytes': len(request.data)})
    try:
        client = ReportClient(directory)
        client._opener = ProcessServer()
        result.put(('ok', client.retry_pending()))
    except Exception as exc:
        result.put(('error', repr(exc)))


def _join(processes):
    for process in processes:
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
            pytest.fail('report worker did not finish')
        assert process.exitcode == 0


def test_revoke_during_post_prevents_put_and_preserves_manual(tmp_path):
    client = ReportClient(tmp_path)
    permission = automatic_snapshot(tmp_path)
    client.enqueue({'source': 'auto', 'subject': 'first'}, b'first', permission=permission)
    client.enqueue({'source': 'auto', 'subject': 'second'}, b'second', permission=permission)
    entered, release = threading.Event(), threading.Event()
    server = client._opener = Server('POST', entered, release)
    with ThreadPoolExecutor(max_workers=2) as pool:
        retry = pool.submit(client.retry_pending, permission=permission)
        try:
            assert entered.wait(5)
            revoke_automatic(tmp_path)
            cleanup = pool.submit(client.discard_automatic)
            assert cleanup.result(timeout=2) == 2
            client.enqueue({'source': 'manual'}, b'manual')
        finally:
            release.set()
        assert retry.result(timeout=5) == 0
    assert [method for method, _ in server.requests] == ['POST']
    assert client.retry_pending() == 1
    assert [method for method, _ in server.requests] == ['POST', 'POST', 'PUT']
    assert client.pending_count == 0


@pytest.mark.parametrize('fail_put', [False, True])
def test_revoke_during_started_put_never_restores_entry(tmp_path, fail_put):
    client = ReportClient(tmp_path)
    permission = automatic_snapshot(tmp_path)
    client.enqueue({'source': 'auto'}, b'one', permission=permission)
    client.enqueue({'source': 'auto'}, b'two', permission=permission)
    entered, release = threading.Event(), threading.Event()
    server = client._opener = Server('PUT', entered, release, fail_put)
    with ThreadPoolExecutor(max_workers=2) as pool:
        retry = pool.submit(client.retry_pending, permission=permission)
        try:
            assert entered.wait(5)
            revoke_automatic(tmp_path)
            assert pool.submit(client.discard_automatic).result(timeout=2) == 2
        finally:
            release.set()
        assert retry.result(timeout=5) == 0
    assert [method for method, _ in server.requests] == ['POST', 'PUT']
    assert client.pending_count == 0


def test_revoke_between_post_response_and_put_permission(tmp_path, monkeypatch):
    client = ReportClient(tmp_path)
    permission = automatic_snapshot(tmp_path)
    server = client._opener = Server()
    client.enqueue({'source': 'auto'}, b'zip', permission=permission)
    entered, release = threading.Event(), threading.Event()
    eligible = client._eligible
    count = 0
    def paused(*args):
        nonlocal count
        count += 1
        if count == 2:
            entered.set()
            assert release.wait(5)
        return eligible(*args)
    monkeypatch.setattr(client, '_eligible', paused)
    with ThreadPoolExecutor(max_workers=2) as pool:
        retry = pool.submit(client.retry_pending, permission=permission)
        try:
            assert entered.wait(5)
            revoke_automatic(tmp_path)
            assert pool.submit(client.discard_automatic).result(timeout=2) == 1
        finally:
            release.set()
        assert retry.result(timeout=5) == 0
    assert [method for method, _ in server.requests] == ['POST']


def test_reenable_requires_successful_cleanup_and_rejects_old_snapshot(tmp_path, monkeypatch):
    client = ReportClient(tmp_path)
    old = automatic_snapshot(tmp_path)
    client.enqueue({'source': 'auto'}, b'old', permission=old)
    revoke_automatic(tmp_path)
    original_save = client._save
    def fail_save(state):
        raise ReportError('Could not store report queue')
    monkeypatch.setattr(client, '_save', fail_save)
    with pytest.raises(ReportError):
        client.discard_automatic()
    with pytest.raises(ReportError):
        client.enable_automatic()
    assert client.enqueue({'source': 'auto'}, b'blocked', permission=old) == 'dropped'
    monkeypatch.setattr(client, '_save', original_save)
    new = client.enable_automatic()
    assert client.pending_count == 0
    assert client.enqueue({'source': 'auto'}, b'stale', permission=old) == 'dropped'
    assert client.enqueue({'source': 'auto'}, b'new', permission=new) == 'queued'


def test_two_processes_enqueue_without_losing_state(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    ready, result, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    processes = [ctx.Process(target=_enqueue_process, args=(str(tmp_path), ready, start, result, marker))
                 for marker in ('one', 'two')]
    for process in processes:
        process.start()
    try:
        assert ready.get(timeout=10) and ready.get(timeout=10)
        start.set()
        outcomes = [result.get(timeout=10), result.get(timeout=10)]
    finally:
        start.set()
        _join(processes)
    assert all(status == 'ok' for status, _ in outcomes), outcomes
    assert outcomes[0][1] == outcomes[1][1]
    state = json.loads((tmp_path / 'reports.json').read_text(encoding='utf-8'))
    assert state['installation_id'] == outcomes[0][1]
    assert {entry['metadata']['subject'] for entry in state['pending']} == {'one', 'two'}
    assert len({entry['id'] for entry in state['pending']}) == 2
    if os.name != 'nt':
        assert (tmp_path / 'reports.json').stat().st_mode & 0o077 == 0


def test_two_processes_retry_same_entry_only_once(tmp_path):
    ReportClient(tmp_path).enqueue({'source': 'manual'}, b'zip')
    ctx = multiprocessing.get_context('spawn')
    ready, result, requests, start = ctx.Queue(), ctx.Queue(), ctx.Queue(), ctx.Event()
    processes = [ctx.Process(target=_retry_process, args=(str(tmp_path), ready, start, requests, result))
                 for _ in range(2)]
    for process in processes:
        process.start()
    try:
        assert ready.get(timeout=10) and ready.get(timeout=10)
        start.set()
        outcomes = [result.get(timeout=10), result.get(timeout=10)]
    finally:
        start.set()
        _join(processes)
    assert all(status == 'ok' for status, _ in outcomes), outcomes
    assert sorted(value for _, value in outcomes) == [0, 1]
    assert [requests.get(timeout=5), requests.get(timeout=5)] == ['POST', 'PUT']
    assert requests.empty()
    assert ReportClient(tmp_path).pending_count == 0


@pytest.mark.parametrize('stop', ['cancel', 'deadline'])
def test_waiting_for_process_delivery_lock_can_be_cancelled(tmp_path, stop):
    client = ReportClient(tmp_path)
    client.enqueue({'source': 'manual'}, b'zip')
    server = client._opener = Server()
    ctx = multiprocessing.get_context('spawn')
    acquired, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_delivery, args=(str(tmp_path), acquired, release))
    holder.start()
    cancel = threading.Event()
    try:
        assert acquired.wait(10)
        with ThreadPoolExecutor(max_workers=1) as pool:
            deadline = time.monotonic() + 0.2 if stop == 'deadline' else None
            retry = pool.submit(client.retry_pending, cancel=cancel, deadline=deadline)
            if stop == 'cancel':
                cancel.set()
            try:
                assert retry.result(timeout=5) == 0
            except ReportError:
                pass
        assert server.requests == []
        assert client.pending_count == 1
    finally:
        release.set()
        _join([holder])
    assert client.retry_pending() == 1


def test_other_process_can_revoke_while_post_is_blocked(tmp_path):
    client = ReportClient(tmp_path)
    client.enqueue({'source': 'auto'}, b'auto')
    ctx = multiprocessing.get_context('spawn')
    entered, release, requests, result = ctx.Event(), ctx.Event(), ctx.Queue(), ctx.Queue()
    worker = ctx.Process(target=_blocked_retry_process,
                         args=(str(tmp_path), entered, release, requests, result))
    worker.start()
    try:
        assert entered.wait(10)
        with ThreadPoolExecutor(max_workers=1) as pool:
            def update():
                client.enqueue({'source': 'manual'}, b'manual')
                return client.discard_automatic()
            assert pool.submit(update).result(timeout=2) == 1
    finally:
        release.set()
        _join([worker])
    assert result.get(timeout=5) == ('ok', 0)
    assert requests.get(timeout=5) == 'POST'
    assert requests.empty()
    state = json.loads((tmp_path / 'reports.json').read_text(encoding='utf-8'))
    assert [entry['metadata']['source'] for entry in state['pending']] == ['manual']


def test_deadline_after_post_prevents_upload(tmp_path, monkeypatch):
    from utils import report_client

    client = ReportClient(tmp_path)
    client.enqueue({'source': 'manual'}, b'zip')
    entered, release, expired = threading.Event(), threading.Event(), threading.Event()
    server = client._opener = Server('POST', entered, release)
    clock = SimpleNamespace(monotonic=lambda: 2 if expired.is_set() else 0,
                            sleep=time.sleep)
    monkeypatch.setattr(report_client, 'time', clock)
    with ThreadPoolExecutor(max_workers=1) as pool:
        retry = pool.submit(client.retry_pending, deadline=1)
        try:
            assert entered.wait(5)
            expired.set()
        finally:
            release.set()
        assert retry.result(timeout=5) == 0
    assert [method for method, _ in server.requests] == ['POST']
    assert client.pending_count == 1


def test_delivery_lock_released_after_process_termination(tmp_path):
    client = ReportClient(tmp_path)
    client.enqueue({'source': 'manual'}, b'zip')
    client._opener = Server()
    ctx = multiprocessing.get_context('spawn')
    acquired, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_delivery, args=(str(tmp_path), acquired, release))
    holder.start()
    try:
        assert acquired.wait(10)
        holder.terminate()
        holder.join(10)
        assert not holder.is_alive()
        assert client.retry_pending(deadline=time.monotonic() + 5) == 1
    finally:
        if holder.is_alive():
            holder.terminate()
            holder.join(5)
