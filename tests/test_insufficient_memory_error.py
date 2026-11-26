"""Memory limits the application reports are safe to handle on Python 3.10."""
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.memory import InsufficientMemoryError

ROOT = Path(__file__).resolve().parents[1]


def _raised_type(call):
    """The type of the MemoryError a call raises, without keeping the instance."""
    try:
        call()
    except MemoryError:
        return sys.exc_info()[0]
    raise AssertionError('the call did not report a memory limit')


def test_handlers_for_memory_error_catch_it():
    assert issubclass(InsufficientMemoryError, MemoryError)
    assert InsufficientMemoryError is not MemoryError
    with pytest.raises(MemoryError, match='needs more'):
        raise InsufficientMemoryError('needs more')


@pytest.mark.parametrize('available,estimate', [(0, 1), (10, 6)])
def test_converter_admission_reports_the_safe_type(monkeypatch, tmp_path, available, estimate):
    from core import batch_converter

    monkeypatch.setattr(batch_converter, 'estimate_job', lambda *args: (estimate, 0))
    monkeypatch.setattr(batch_converter, 'available_memory_bytes', lambda: available)
    raised = _raised_type(lambda: batch_converter._admit(tmp_path / 'source.png', 0, tmp_path))
    assert raised is InsufficientMemoryError


def test_tensorrt_workspace_limit_reports_the_safe_type():
    from core.inference_tensorrt import TensorRT

    backend = object.__new__(TensorRT)
    backend.trt = None
    backend.logger = SimpleNamespace(recent=[])
    backend.torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device: (64 << 20, 128 << 20)))
    raised = _raised_type(lambda: backend._build(None, None, False, 48 << 20))
    assert raised is InsufficientMemoryError


def test_flat_field_preview_failure_reports_the_safe_type(qapp, monkeypatch):
    from PyQt6.QtGui import QPixmap
    from dialogs import dustmap

    monkeypatch.setattr(dustmap.QPixmap, 'fromImage', staticmethod(lambda image: QPixmap()))
    raised = _raised_type(lambda: dustmap._to_pixmap(np.zeros((4, 4, 3), np.uint8)))
    assert raised is InsufficientMemoryError


def test_mosaic_preview_failure_reports_the_safe_type(qapp, monkeypatch):
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import QWidget
    from core.mosaic import make_frame
    from dialogs import mosaic

    parent = QWidget()
    dialog = mosaic.MosaicDialog(parent)
    try:
        monkeypatch.setattr(mosaic, 'cv2_to_pixmap', lambda _: QPixmap())
        frame = make_frame('first', np.zeros((8, 8, 3), np.uint8))
        raised = _raised_type(lambda: dialog._append_frames([frame]))
        assert raised is InsufficientMemoryError
    finally:
        dialog.close()
        parent.deleteLater()


_TEARDOWN = textwrap.dedent('''
    import gc
    import sys
    import weakref

    from utils.memory import InsufficientMemoryError


    class Tile:
        pass


    def failing_call(kind):
        tile = Tile()                        # kept alive by the traceback
        weakref.finalize(tile, gc.collect)   # a finalizer that starts a collection
        raise kind('no memory')


    def handle(kind):
        try:
            failing_call(kind)
        except MemoryError as error:
            return str(error)


    kind = InsufficientMemoryError if sys.argv[1] == 'application' else MemoryError
    print(handle(kind), flush=True)
''')


def _teardown(kind):
    return subprocess.run([sys.executable, '-c', _TEARDOWN, kind], cwd=ROOT,
                          capture_output=True, text=True, timeout=60)


def test_collection_during_teardown_does_not_crash():
    """A handler drops the error while a finalizer in its traceback collects."""
    result = _teardown('application')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'no memory'


@pytest.mark.skipif(sys.version_info[:2] != (3, 10),
                    reason='only Python 3.10 frees a plain MemoryError twice')
def test_the_teardown_check_detects_the_interpreter_fault():
    assert _teardown('plain').returncode != 0
