"""The StackMFF CPU benchmark tool: smoke runs are reproducible; full runs ask first."""
import json
import os
import subprocess
import sys

import pytest

from core.multi_focus_fusion import is_ai_fusion_available

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(PROJECT_ROOT, 'tools', 'stackmff_cpu_benchmark.py')

pytestmark = pytest.mark.skipif(not is_ai_fusion_available(),
                                reason="ONNX Runtime not installed")


def _run(*argv):
    return subprocess.run([sys.executable, TOOL, *argv], capture_output=True, text=True,
                          cwd=PROJECT_ROOT, timeout=85)


def test_full_mode_prints_the_plan_and_does_not_run():
    result = _run('tiling', '--full')
    assert result.returncode == 0
    assert result.stdout == ''
    plan = json.loads(result.stderr[:result.stderr.rindex('}') + 1])['plan']
    assert plan['estimated_seconds'] > 0 and plan['estimated_ram_bytes'] > 0
    assert plan['historical_peak_rss_bytes'] == int(9.7e9)
    assert '--confirm' in result.stderr
    assert '1.6-1.7x' in plan['ram_measured_note']
    assert plan['description'].startswith('6 tiled renders')


def test_tile_sizes_replace_the_default_comparison_sizes():
    result = _run('tiling', '--full', '--tile-sizes', '256')
    plan = json.loads(result.stderr[:result.stderr.rindex('}') + 1])['plan']
    # historical + three overlaps at the 448 base + 256; 640 is no longer added.
    assert plan['description'].startswith('5 tiled renders')


def test_overhead_plan_counts_warm_ups_and_every_arm(tmp_path):
    def plan(*extra):
        result = _run('overhead', '--full', *extra)
        return json.loads(result.stderr[:result.stderr.rindex('}') + 1])['plan']
    two_arms = plan()
    assert two_arms['description'].startswith('10 timed renders (profiling off, on;')
    assert '20 renders in all' in two_arms['description']
    three_arms = plan('--baseline-root', str(tmp_path))
    assert three_arms['description'].startswith('15 timed renders (profiling off, on, baseline;')
    assert '30 renders in all' in three_arms['description']
    assert three_arms['estimated_seconds'] == pytest.approx(
        1.5 * two_arms['estimated_seconds'], abs=2)


def test_each_finished_configuration_is_checkpointed(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('stackmff_cpu_benchmark', TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    checkpoint = tmp_path / 'out.json.partial.jsonl'
    monkeypatch.setattr(tool, '_CHECKPOINT', checkpoint)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    result = tool._run_worker(tool.Path(PROJECT_ROOT), scratch, {'kind': 'no-such-kind'}, 60)
    assert 'error' in result
    lines = checkpoint.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])['config'] == {'kind': 'no-such-kind'}


def test_historical_session_sequence_reproduces_the_logged_churn():
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tools'))
    try:
        import stackmff_cpu_benchmark as bench
    finally:
        sys.path.pop(0)
    sequence, tiles = bench.historical_sequence(82, 1040, 778, 389, 256)
    assert tiles == 20 and len(sequence) == 40
    assert len(set(sequence)) == 7          # 384x384 is both a reduced and a corner shape
    cache, created = [], 0
    for key in sequence:                      # the runtime's 2-entry LRU
        if key in cache:
            cache.remove(key)
        else:
            created += 1
            if len(cache) == 2:
                cache.pop(0)
        cache.append(key)
    assert created == 20                      # neural_session events in the historical log


def test_smoke_tiling_and_sessions(tmp_path):
    for name in ('tiling', 'sessions'):
        output = tmp_path / f'{name}.json'
        result = _run(name, '--smoke', '--threads', '1', '--json-output', str(output))
        assert result.returncode == 0, result.stderr
        payload = json.loads(output.read_text())
        assert not (tmp_path / f'{name}.json.partial.jsonl').exists()
        assert payload['benchmark'] == name and payload['mode'] == 'smoke'
        assert {'commit', 'dirty', 'versions', 'cpu', 'seed'} <= set(payload)
        if name == 'tiling':
            rows = {row['label']: row for row in payload['output']['rows']}
            reference = rows[payload['output']['reference']]
            assert reference['vs_reference']['identical']
            assert all('error' not in row for row in rows.values())
            assert all(row['tiles'] == len(row['coords']) for row in rows.values())
        else:
            assert payload['output']['synthetic_eviction']['evictions'] > 0
