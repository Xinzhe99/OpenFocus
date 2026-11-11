"""CLI --profile-performance: stderr summary, --json embedding, FILE output."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import make_two_focus_stack
from core.multi_focus_fusion import is_ai_fusion_available

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = [pytest.mark.usefixtures("isolated_process_env"),
              pytest.mark.skipif(not is_ai_fusion_available(),
                                 reason="ONNX Runtime not installed")]


def _stack(count=3, size=(150, 110)):
    first, second = make_two_focus_stack(size)
    frames = [first, second]
    while len(frames) < count:
        frames.append(((first.astype(np.uint16) + second) // 2).astype(np.uint8))
    return frames


def _run_cli(*argv, env=None):
    return subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'main.py'), *argv],
                          capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300,
                          env={**os.environ, **(env or {}),
                               'OPENFOCUS_DATA_DIR': os.environ['OPENFOCUS_DATA_DIR'],
                               'OPENFOCUS_LOG_DIR': os.environ['OPENFOCUS_LOG_DIR']})


def _cli_stack(tmp_path):
    from utils.image_utils import imwrite_auto
    stack = tmp_path / 'stack'
    stack.mkdir()
    for index, image in enumerate(_stack()):
        assert imwrite_auto(str(stack / f'f{index}.png'), image)
    return stack


_TILES = ('--method', 'stackmffv3', '--cpu', '--threads', '1', '--tile-size', '64',
          '--tile-overlap', '16', '--tile-threshold', '32')


def test_cli_embeds_report_in_json_and_writes_file(tmp_path):
    stack = _cli_stack(tmp_path)
    plain, profiled = tmp_path / 'plain.png', tmp_path / 'profiled.png'
    assert _run_cli('--input', str(stack), '--output', str(plain), *_TILES).returncode == 0
    result = _run_cli('--input', str(stack), '--output', str(profiled), *_TILES,
                      '--json', str(tmp_path / 'summary.json'),
                      '--profile-performance', str(tmp_path / 'profile.json'))
    assert result.returncode == 0, result.stderr
    assert 'Where the time went' in result.stderr
    assert 'Stages in execution order (total / self):' in result.stderr
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert {'mode', 'method', 'output', 'outputs'} <= set(summary)
    written = json.loads((tmp_path / 'profile.json').read_text())
    assert summary['performance']['tiling'] == written['tiling']
    assert written['outcome'] == 'completed'
    assert written['threads']['requested'] == 1
    import cv2
    np.testing.assert_array_equal(cv2.imread(str(profiled)), cv2.imread(str(plain)))


def test_cli_json_progress_keeps_stdout_machine_readable(tmp_path):
    stack = _cli_stack(tmp_path)
    result = _run_cli('--input', str(stack), '--output', str(tmp_path / 'out.png'), *_TILES,
                      '--json-progress', '--profile-performance')
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines and all(isinstance(json.loads(line), dict) for line in lines)
    assert 'Stages in execution order (total / self):' in result.stderr


def test_cli_rejects_profiling_batch_runs(tmp_path):
    stack = _cli_stack(tmp_path)
    result = _run_cli('--input', str(stack), '--output-dir', str(tmp_path / 'out'),
                      '--profile-performance')
    assert result.returncode == 2
    assert '--profile-performance requires a single' in result.stderr
