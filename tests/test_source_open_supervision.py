from types import SimpleNamespace
import multiprocessing
import sys

import pytest

from tools.source_open_benchmark import _run_child


class Reader:
    def __init__(self, messages, error=EOFError):
        self.messages = iter(messages)
        self.error = error
        self.next_message = None
    def poll(self, timeout=0):
        try:
            self.next_message = next(self.messages)
        except StopIteration:
            raise self.error()
        return True
    def recv(self):
        return self.next_message
    def close(self):
        pass


class Process:
    def __init__(self, exitcode):
        self.exitcode = exitcode
        self.terminated = False
    def start(self): pass
    def is_alive(self): return False
    def join(self, timeout=None): pass
    def terminate(self): self.terminated = True
    def close(self): pass


def run(messages, exitcode=0, error=EOFError):
    reader, process = Reader(messages, error), Process(exitcode)
    context = SimpleNamespace(Pipe=lambda **kw: (reader, SimpleNamespace(close=lambda: None)),
                              Process=lambda **kw: process)
    report = {'errors': [], 'sampled_peak_rss': 0, 'process_high_water_rss': 0}
    _run_child(context, lambda: None, (), SimpleNamespace(set=lambda: None),
               {'max_rss_mib': None, 'max_load_seconds': None}, report)
    return report, process


def test_result_then_closed_pipe_is_normal():
    report, process = run([{'event': 'result', 'report': {'done': True}}], error=BrokenPipeError)
    assert report['done'] and not report['errors'] and not process.terminated


def test_missing_result_or_nonzero_exit_fails():
    assert 'no result' in run([])[0]['errors'][0]
    assert 'code 3' in run([{'event': 'result'}], exitcode=3)[0]['errors'][0]




def _send_result(connection):
    connection.send({'event': 'result', 'report': {'done': True}})
    connection.close()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows pipe closure regression')
def test_windows_spawn_result_then_close():
    context = multiprocessing.get_context('spawn')
    report = {'errors': [], 'sampled_peak_rss': 0, 'process_high_water_rss': 0}
    _run_child(context, _send_result, (), context.Event(),
               {'max_rss_mib': None, 'max_load_seconds': None}, report)
    assert report['done'] and not report['errors']
def test_other_io_error_propagates():
    with pytest.raises(PermissionError):
        run([], error=PermissionError)
