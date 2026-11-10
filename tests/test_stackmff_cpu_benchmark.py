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


def _load_tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location('stackmff_cpu_benchmark', TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_plan_and_interrupted_repeated_runs_preserve_checkpoints(tmp_path, monkeypatch):
    tool = _load_tool()
    output = tmp_path / 'result.json'
    old = tmp_path / 'result.json.partial.jsonl'
    old.write_text('earlier run\n')
    monkeypatch.setattr(tool, '_git', lambda root: (None, None))
    assert tool.main(['sessions', '--full', '--json-output', str(output)]) == 0
    assert old.read_text() == 'earlier run\n'

    calls = 0

    def run_worker(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls % 2 == 0:
            raise KeyboardInterrupt
        return subprocess.CompletedProcess(args[0], 0, stdout='{"measurement": 42}\n', stderr='')

    monkeypatch.setattr(tool.subprocess, 'run', run_worker)
    for suffix in (1, 2):
        with pytest.raises(KeyboardInterrupt):
            tool.main(['sessions', '--json-output', str(output)])
        checkpoint = tmp_path / f'result.json.partial.jsonl.{suffix}'
        assert json.loads(checkpoint.read_text().splitlines()[0])['result']['measurement'] == 42
    assert old.read_text() == 'earlier run\n'


def test_stdout_worker_failure_keeps_successful_results_and_checkpoint(tmp_path, monkeypatch, capsys):
    tool = _load_tool()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tool, '_git', lambda root: (None, None))
    existing = tmp_path / 'sessions.partial.jsonl'
    existing.write_text('previous data\n')
    responses = iter([
        subprocess.CompletedProcess([], 1, stdout='', stderr='worker failed'),
        subprocess.CompletedProcess([], 0, stdout='{"measurement": 42}\n', stderr=''),
    ])
    monkeypatch.setattr(tool.subprocess, 'run', lambda *args, **kwargs: next(responses))
    assert tool.main(['sessions']) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload['output']['recorded']['error'] == 'worker failed'
    assert payload['output']['synthetic_eviction']['measurement'] == 42
    checkpoint = tmp_path / 'sessions.partial.jsonl.1'
    assert str(checkpoint) in captured.err
    entries = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert len(entries) == 2
    assert entries[1]['result']['measurement'] == 42
    assert existing.read_text() == 'previous data\n'


def test_json_worker_timeout_retains_completed_results(tmp_path, monkeypatch):
    tool = _load_tool()
    monkeypatch.setattr(tool, '_git', lambda root: (None, None))
    calls = 0

    def run_worker(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
        return subprocess.CompletedProcess(args[0], 0, stdout='{"measurement": 42}\n', stderr='')

    monkeypatch.setattr(tool.subprocess, 'run', run_worker)
    output = tmp_path / 'result.json'
    assert tool.main(['sessions', '--json-output', str(output)]) == 1
    payload = json.loads(output.read_text())
    assert 'timed out' in payload['output']['recorded']['error']
    assert payload['output']['synthetic_eviction']['measurement'] == 42
    checkpoint = tmp_path / 'result.json.partial.jsonl'
    assert len(checkpoint.read_text().splitlines()) == 2


def test_successful_stdout_run_keeps_checkpoint(tmp_path, monkeypatch, capsys):
    tool = _load_tool()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tool, '_git', lambda root: (None, None))
    monkeypatch.setattr(tool.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, stdout='{"measurement": 42}\n',
                                                    stderr=''))
    assert tool.main(['sessions']) == 0
    captured = capsys.readouterr()
    checkpoint = tmp_path / 'sessions.partial.jsonl'
    assert str(checkpoint) in captured.err
    assert len(checkpoint.read_text().splitlines()) == 2
    assert json.loads(captured.out)['output']['recorded']['measurement'] == 42


def test_overhead_continues_after_failure(tmp_path, monkeypatch):
    tool = _load_tool()
    monkeypatch.setattr(tool, '_git', lambda root: (None, None))
    calls = 0

    def run_worker(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(args[0], 1, stdout='', stderr='failed')
        return subprocess.CompletedProcess(
            args[0], 0, stdout='{"wall_s": 1, "output_sha256": "same"}\n', stderr='')

    monkeypatch.setattr(tool.subprocess, 'run', run_worker)
    output = tmp_path / 'overhead.json'
    assert tool.main(['overhead', '--json-output', str(output)]) == 1
    result = json.loads(output.read_text())['output']
    assert calls == 10
    assert result['off_s']['n'] == 4
    assert result['on_s']['n'] == 5
    assert result['failures']['off'][0]['result']['error'] == 'failed'
    assert result['failures']['off'][0]['repetition'] == 1
    assert result['failures']['on'] == []
    assert result['outputs_identical'] is True
    assert len((tmp_path / 'overhead.json.partial.jsonl').read_text().splitlines()) == 10


def test_overhead_missing_arm_is_indeterminate_and_checkpoints_identify_arms(tmp_path, monkeypatch):
    tool = _load_tool()
    monkeypatch.setattr(tool, '_git', lambda root: (None, None))

    def run_worker(command, **kwargs):
        config = json.loads(command[3])
        if config['arm'] == 'off':
            return subprocess.CompletedProcess(command, 1, stdout='', stderr='off failed')
        return subprocess.CompletedProcess(
            command, 0, stdout='{"wall_s": 1, "output_sha256": "same"}\n', stderr='')

    monkeypatch.setattr(tool.subprocess, 'run', run_worker)
    output = tmp_path / 'overhead.json'
    assert tool.main(['overhead', '--baseline-root', str(tmp_path), '--json-output', str(output)]) == 1
    result = json.loads(output.read_text())['output']
    assert result['off_s'] is None
    assert result['on_s']['n'] == result['baseline_s']['n'] == 5
    assert result['outputs_identical'] is None
    assert [failure['repetition'] for failure in result['failures']['off']] == [1, 2, 3, 4, 5]
    assert all(failure['result']['error'] == 'off failed' for failure in result['failures']['off'])
    checkpoint = tmp_path / 'overhead.json.partial.jsonl'
    entries = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    assert len(entries) == 15
    assert {entry['config']['arm'] for entry in entries} == {'off', 'on', 'baseline'}
    assert sum(entry['config']['arm'] == 'baseline' for entry in entries) == 5
    assert all(not entry['config']['profile'] for entry in entries
               if entry['config']['arm'] in ('off', 'baseline'))


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
