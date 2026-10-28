"""Executable diagnostics are observable in real subprocess failures."""
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(code, tmp_path, *args):
    environment = dict(os.environ, OPENFOCUS_LOG_DIR=str(tmp_path / "logs"),
                       PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-c", code, *map(str, args)],
                          cwd=ROOT, env=environment, capture_output=True,
                          text=True, encoding="utf-8", timeout=30)


@pytest.mark.parametrize("kind", ["main", "thread", "unraisable"])
def test_real_unhandled_failures_have_tracebacks_in_logs_and_crash_report(tmp_path, kind):
    result = _run("""
import gc
import sys
import threading
from utils.logging_setup import setup_logging, log_session_start, export_feedback_package
from utils.diagnostics import install_exception_hooks
setup_logging()
log_session_start()
install_exception_hooks()
def failing_backend():
    raise RuntimeError('diagnostic failure marker')
if sys.argv[1] == 'main':
    failing_backend()
elif sys.argv[1] == 'thread':
    worker = threading.Thread(target=failing_backend, name='diagnostic-worker')
    worker.start()
    worker.join()
else:
    class BrokenFinalizer:
        def __del__(self):
            failing_backend()
    broken = BrokenFinalizer()
    del broken
    gc.collect()
export_feedback_package(sys.argv[2], include_all_logs=False)
""", tmp_path, kind, tmp_path / "report.zip")
    assert result.returncode == (1 if kind == "main" else 0), result.stderr
    files = list((tmp_path / "logs").glob("*.log"))
    crashes = list((tmp_path / "logs").glob("*.crash.txt"))
    assert len(files) == len(crashes) == 1
    for path in (*files, *crashes):
        text = path.read_text(encoding="utf-8")
        assert "RuntimeError: diagnostic failure marker" in text
        assert "failing_backend" in text
        assert "Traceback (most recent call last)" in text
        assert "session=" in text
    if kind == "thread":
        assert "diagnostic-worker" in files[0].read_text(encoding="utf-8")
    if kind != "main":
        with zipfile.ZipFile(tmp_path / "report.zip") as archive:
            assert b"diagnostic failure marker" in archive.read("session_log.txt")
            assert b"failing_backend" in archive.read(f"logs/{crashes[0].name}")
    assert "diagnostic failure marker" in result.stderr  # Original console hook survives.


def test_import_and_hook_lifecycle_leave_embedding_hooks_in_control(tmp_path):
    result = _run("""
import sys
import threading
from utils.logging_setup import setup_logging
previous = (sys.excepthook, threading.excepthook, sys.unraisablehook)
from utils.diagnostics import install_exception_hooks, shutdown_exception_hooks
assert previous == (sys.excepthook, threading.excepthook, sys.unraisablehook)
setup_logging()
first = install_exception_hooks()
assert install_exception_hooks() == first
shutdown_exception_hooks()
assert previous == (sys.excepthook, threading.excepthook, sys.unraisablehook)
install_exception_hooks()
def embedding_hook(*args):
    pass
sys.excepthook = embedding_hook
shutdown_exception_hooks()
assert sys.excepthook is embedding_hook
assert previous[1:] == (threading.excepthook, sys.unraisablehook)
print('embedding hooks retained')
""", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "embedding hooks retained"


def test_diagnostics_start_without_gui_or_numerical_dependencies(tmp_path):
    result = _run("""
import importlib.abc
import sys
class MissingRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PyQt6', 'numpy', 'cv2', 'torch', 'scipy', 'ui', 'controllers', 'dialogs'}:
            raise ModuleNotFoundError('runtime unavailable', name=fullname)
sys.meta_path.insert(0, MissingRuntime())
from utils.logging_setup import setup_logging, log_session_start, log_action, session_log_excerpt
from utils.diagnostics import install_exception_hooks, system_facts
setup_logging()
log_session_start()
install_exception_hooks()
log_action('dependency_failure', module='numpy')
assert 'dependency_failure' in session_log_excerpt()
assert system_facts()['OpenFocus']
assert not any(name in sys.modules for name in ('numpy', 'PyQt6', 'torch', 'ui', 'controllers'))
print('diagnostics available before runtime imports')
""", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "diagnostics available before runtime imports" in result.stdout


def test_main_cli_runs_without_gui_dependencies(tmp_path):
    result = _run("""
import importlib.abc
import runpy
import sys
class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PyQt6', 'ui', 'controllers', 'dialogs'}:
            raise ModuleNotFoundError('GUI unavailable', name=fullname)
sys.meta_path.insert(0, NoGui())
sys.argv = ['main.py', '--help']
runpy.run_path('main.py', run_name='__main__')
""", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "--param" in result.stdout
    assert "--input" in result.stdout


def test_failure_during_gui_import_is_captured_before_window_startup(tmp_path):
    result = _run("""
import importlib.abc
import runpy
import sys
class BrokenGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'PyQt6':
            raise ModuleNotFoundError('Qt startup failure marker', name=fullname)
sys.meta_path.insert(0, BrokenGui())
sys.argv = ['main.py']
runpy.run_path('main.py', run_name='__main__')
""", tmp_path)
    assert result.returncode == 1
    text = next((tmp_path / "logs").glob("*.log")).read_text(encoding="utf-8")
    assert "ModuleNotFoundError: Qt startup failure marker" in text
    assert 'from PyQt6.QtWidgets import' in text
    assert "Traceback (most recent call last)" in text


def test_host_hook_wrappers_keep_predecessors_after_shutdown_and_reinstall(tmp_path):
    result = _run("""
import sys
import threading
from types import SimpleNamespace
from utils.logging_setup import setup_logging, session_log_excerpt
from utils.diagnostics import install_exception_hooks, shutdown_exception_hooks
targets = ((sys, 'excepthook'), (threading, 'excepthook'), (sys, 'unraisablehook'))
predecessor_calls = [0, 0, 0]
host_calls = [0, 0, 0]
for index, (module, name) in enumerate(targets):
    def predecessor(*args, i=index):
        predecessor_calls[i] += 1
    setattr(module, name, predecessor)
setup_logging()
install_exception_hooks()
callbacks = tuple(getattr(module, name) for module, name in targets)
for index, (module, name) in enumerate(targets):
    def host(*args, i=index):
        host_calls[i] += 1
        callbacks[i](*args)
    setattr(module, name, host)
error = RuntimeError('hook chain marker')
thread_args = threading.ExceptHookArgs((RuntimeError, error, None, None))
unraisable = SimpleNamespace(exc_type=RuntimeError, exc_value=error,
                            exc_traceback=None, err_msg=None)
def report_all():
    sys.excepthook(RuntimeError, error, None)
    threading.excepthook(thread_args)
    sys.unraisablehook(unraisable)
shutdown_exception_hooks()
before = session_log_excerpt()
report_all()
assert predecessor_calls == [1, 1, 1]
assert session_log_excerpt() == before
install_exception_hooks()
report_all()
assert predecessor_calls == [2, 2, 2]
assert host_calls == [2, 2, 2]
assert session_log_excerpt().count('RuntimeError: hook chain marker') == 3
shutdown_exception_hooks()
print('hook delegation retained exactly once')
""", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "hook delegation retained exactly once"


def test_cli_interrupt_during_decode_keeps_exit_message_and_cancel_outcome(tmp_path):
    result = _run("""
from pathlib import Path
import sys
import numpy as np
import core.cli as cli
from utils.image_utils import imwrite_auto
source = Path(sys.argv[1]) / 'input'
source.mkdir()
for index in range(2):
    assert imwrite_auto(str(source / f'{index}.png'), np.zeros((12, 16, 3), np.uint8))
def interrupt(*args, **kwargs):
    raise KeyboardInterrupt()
cli._load_images = interrupt
assert cli.run_cli(['-i', str(source), '-o', str(source / 'out.png'), '-m', 'dct']) == 1
assert not (source / 'out.png').exists()
""", tmp_path, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "interrupted" in result.stderr
    assert "Loaded" in result.stderr  # human progress moved to stderr
    text = next((tmp_path / "logs").glob("*.log")).read_text(encoding="utf-8")
    assert "event=cli.cancelled" in text
    assert "event=cli.failed" not in text
    assert "Traceback" not in text
