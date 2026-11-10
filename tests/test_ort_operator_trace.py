"""Optional ONNX Runtime operator traces: off by default, closed on every exit path."""
from collections import OrderedDict
import json
import os

import pytest

from core.multi_focus_fusion import is_ai_fusion_available

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(PROJECT_ROOT, 'weights', 'stackmff_v3_final.onnx')

pytestmark = pytest.mark.skipif(not is_ai_fusion_available(),
                                reason="ONNX Runtime not installed")


@pytest.fixture
def runtime(monkeypatch):
    from fusion_methods import neural_runtime
    monkeypatch.setattr(neural_runtime, '_SESSIONS', OrderedDict())
    monkeypatch.setattr(neural_runtime, '_TRACED', {})
    return neural_runtime


def _run(runtime, shape):
    import numpy as np
    session = runtime._session(MODEL, shape, 'CPU', 1)
    session.run(['logits'], {'input': np.zeros((1, *shape), np.float32)})


def test_tracing_is_off_unless_requested(runtime, tmp_path, monkeypatch):
    monkeypatch.delenv(runtime.OPERATOR_PROFILE_ENV, raising=False)
    monkeypatch.setenv(runtime.OPERATOR_PROFILE_DIR_ENV, str(tmp_path))
    _run(runtime, (2, 64, 64))
    assert runtime.operator_trace_directory() is None
    assert not runtime._TRACED
    runtime.finish_operator_traces()
    assert not list(tmp_path.iterdir())


def test_an_unusable_trace_folder_disables_tracing_not_inference(runtime, tmp_path, monkeypatch):
    blocker = tmp_path / 'not-a-folder'
    blocker.write_text('x')
    monkeypatch.setenv(runtime.OPERATOR_PROFILE_ENV, '1')
    monkeypatch.setenv(runtime.OPERATOR_PROFILE_DIR_ENV, str(blocker / 'traces'))
    _run(runtime, (2, 64, 64))                      # the session still runs
    assert runtime.operator_trace_directory() is None
    assert not runtime._TRACED


def test_traces_are_written_on_eviction_clear_and_finish(runtime, tmp_path, monkeypatch):
    monkeypatch.setenv(runtime.OPERATOR_PROFILE_ENV, '1')
    monkeypatch.setenv(runtime.OPERATOR_PROFILE_DIR_ENV, str(tmp_path))
    for shape in ((2, 64, 64), (2, 64, 128), (2, 128, 64)):   # third evicts the first
        _run(runtime, shape)
    assert len(list(tmp_path.glob('openfocus-ort-2x64x64-t1-*.json'))) == 1
    runtime._clear_sessions()                                  # ends the other two
    assert len(list(tmp_path.glob('openfocus-ort-*.json'))) == 3
    _run(runtime, (2, 64, 64))
    runtime.finish_operator_traces()                           # normal completion
    traces = sorted(tmp_path.glob('openfocus-ort-*.json'))
    assert len(traces) == 4 and not runtime._TRACED
    events = json.loads(traces[0].read_text())
    assert any(event.get('cat') == 'Node' for event in events)
