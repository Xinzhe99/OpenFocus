"""Profiling the shared tiled path: every tiled method reports geometry and tile work."""
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import make_two_focus_stack
from core.multi_focus_fusion import MultiFocusFusion
from utils import performance_profile as pp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stack(count=3, size=(150, 110)):
    first, second = make_two_focus_stack(size)
    frames = [first, second]
    while len(frames) < count:
        frames.append(((first.astype(np.uint16) + second) // 2).astype(np.uint8))
    return frames


def _tiled(algorithm, profile, threads=2):
    fusion = MultiFocusFusion(algorithm=algorithm, use_gpu=False, tile_enabled=True,
                              tile_block_size=64, tile_overlap=16, tile_threshold=32)
    with pp.active(profile):
        return fusion.fuse(input_source=_stack(), img_resize=None, thread_count=threads)


@pytest.mark.parametrize('algorithm', ['guided_filter', 'dct'])
def test_tiled_classical_fusion_reports_geometry_and_pool_work(algorithm):
    reference = _tiled(algorithm, pp.NULL_PROFILE)
    profile = pp.PerformanceProfile()
    profiled = _tiled(algorithm, profile)
    np.testing.assert_array_equal(profiled, reference)
    report = profile.report('completed')
    tiling = report['tiling']
    assert tiling['count'] == len(tiling['coords']) > 1
    assert (tiling['block_requested'], tiling['overlap_requested']) == (64, 16)
    assert tiling['clipped_area_ratio'] > 1
    assert tiling['padded_inference_area_ratio'] is None      # no padding backend
    assert 1 <= report['threads']['tile_workers'] <= 2
    # Tiles run on pool threads: summed under ``concurrent``, never in the tree.
    for name in ('tile.crop', 'tile.fuse'):
        assert report['concurrent'][name]['count'] == tiling['count']
    stages = report['stages']
    assert stages['tiled/wait_for_tile']['count'] == tiling['count']
    assert stages['tiled/accumulate']['count'] == tiling['count']
    assert stages['tiled/normalize']['count'] == 1
    assert not any(path.endswith('tile.fuse') for path in stages)


def test_cli_profiles_a_classical_method(tmp_path):
    from utils.image_utils import imwrite_auto
    import json
    stack = tmp_path / 'stack'
    stack.mkdir()
    for index, image in enumerate(_stack()):
        assert imwrite_auto(str(stack / f'f{index}.png'), image)
    report = tmp_path / 'reports' / 'profile.json'       # the folder is created
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, 'main.py'), '--input', str(stack),
         '--output', str(tmp_path / 'out.png'), '--method', 'guided_filter', '--threads', '1',
         '--tile-size', '64', '--tile-overlap', '16', '--tile-threshold', '32',
         '--align', 'ecc', '--profile-performance', str(report)],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300)
    assert result.returncode == 0, result.stderr
    written = json.loads(report.read_text(encoding='utf-8'))
    stages = written['stages']
    assert written['render']['algorithm'] == 'guided_filter'
    assert stages['render/fusion/tiled/normalize']['count'] == 1
    assert stages['render']['order'] < stages['render/registration']['order'] \
        < stages['render/fusion']['order']
    # Registration is broken down: the step on the render thread, its pool work beside it.
    assert stages['render/registration/step.ecc']['count'] == 1
    assert written['concurrent']['registration.estimate']['count'] == 2    # adjacent pairs
    assert written['concurrent']['registration.prepare']['count'] == 3     # frames
    assert 'Where the time went' in result.stderr


def test_cli_diagnostics_need_no_gui_packages(tmp_path):
    """`pip install openfocus` has no PyQt6; profiling and detailed logs still work."""
    from utils.image_utils import imwrite_auto
    import json
    stack = tmp_path / 'stack'
    stack.mkdir()
    for index, image in enumerate(_stack()):
        assert imwrite_auto(str(stack / f'f{index}.png'), image)
    report = tmp_path / 'profile.json'
    argv = ['--input', str(stack), '--output', str(tmp_path / 'out.png'),
            '--method', 'guided_filter', '--threads', '1', '--log-level', 'debug',
            '--profile-performance', str(report)]
    code = ("import sys; sys.modules['PyQt6'] = None\n"
            "from core.cli import run_cli\n"
            f"sys.exit(run_cli({argv!r}))\n")
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            cwd=PROJECT_ROOT, timeout=300,
                            env={**os.environ, 'OPENFOCUS_LOG_DIR': str(tmp_path / 'logs')})
    assert result.returncode == 0, result.stderr
    assert 'warning: --profile-performance' not in result.stderr
    runtime = json.loads(report.read_text(encoding='utf-8'))['runtime']
    assert runtime['automatic_threads']['maximum'] >= 1
    log = next((tmp_path / 'logs').glob('openfocus-*.log')).read_text(encoding='utf-8')
    assert 'event=render_resources ' in log


@pytest.mark.parametrize('target', ['frame', 'output', 'depth'])
def test_cli_refuses_a_profile_file_that_is_an_input_or_another_output(tmp_path, capsys, target):
    """The report is written last, even after a failure: it must never replace a
    source frame (also one found by expanding an input folder) or another output."""
    from core.cli import EXIT_USAGE, run_cli
    from utils.image_utils import imwrite_auto
    stack = tmp_path / 'stack'
    stack.mkdir()
    for index, image in enumerate(_stack()):
        assert imwrite_auto(str(stack / f'f{index}.png'), image)
    output = tmp_path / 'out.png'
    destination = {'frame': stack / 'f1.png', 'output': output,
                   'depth': tmp_path / 'out_depth.png'}[target]
    before = (stack / 'f1.png').read_bytes()
    code = run_cli(['--input', str(stack), '--output', str(output), '--method', 'dct',
                    '--threads', '1', '--depth-map', '--profile-performance', str(destination)])
    assert code == EXIT_USAGE
    message = capsys.readouterr().err
    assert '--profile-performance' in message
    assert ('overwrite an input image' in message) == (target == 'frame')
    assert (stack / 'f1.png').read_bytes() == before
    assert not output.exists()
