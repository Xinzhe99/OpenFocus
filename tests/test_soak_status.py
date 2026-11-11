import json
import sys
from types import SimpleNamespace

from tools import soak_test


def test_no_fixtures_is_not_success(tmp_path, monkeypatch, capsys):
    report = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['soak_test', '--quick', '--json', str(report)])
    monkeypatch.setattr(soak_test, 'load_stack', lambda path: [])
    monkeypatch.setattr(soak_test, 'fusion_algorithms', lambda: [SimpleNamespace(id='example')])
    assert soak_test.main() == 1
    data = json.loads(report.read_text())
    assert not data['passed'] and data['executed_count'] == 0 and data['skipped']
    assert 'No soak cases executed' in capsys.readouterr().out


def test_interrupted_second_case_preserves_first(tmp_path, monkeypatch):
    report = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['soak_test', '--quick', '--json', str(report), '--only', 'hetao40/'])
    monkeypatch.setattr(soak_test, 'load_stack', lambda path: [SimpleNamespace(shape=(2, 2))])
    monkeypatch.setattr(soak_test, 'fusion_algorithms', lambda: [SimpleNamespace(id='one'), SimpleNamespace(id='two')])
    def run(label, frames, algo, failures, results, reg=None):
        if algo == 'two':
            raise KeyboardInterrupt()
        results.append({'case': label})
    monkeypatch.setattr(soak_test, 'run_case', run)
    assert soak_test.main() == 1
    data = json.loads(report.read_text())
    assert data['executed_count'] == 1 and data['interrupted'] and not data['passed']
    assert 'hetao40/two' in data['unfinished']
