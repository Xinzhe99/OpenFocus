"""The parallel test runner cuts the suite into consecutive, complete slices."""
import json
from pathlib import Path

import pytest

from tools import parallel_tests
from tools.parallel_tests import collect, exit_code, split_files


def _counts(*sizes):
    return {f"tests/test_{index:03d}.py": size for index, size in enumerate(sizes)}


@pytest.mark.parametrize("sizes,slices", [
    ((5, 5, 5, 5), 2),
    ((100, 1, 1, 1, 1, 1), 3),
    ((1, 1, 1, 1, 1, 100), 3),
    ((0, 0, 7, 0, 3), 4),
    ((3,), 8),
    (tuple(range(1, 40)), 8),
])
def test_slices_are_consecutive_complete_and_never_empty(sizes, slices):
    counts = _counts(*sizes)
    groups = split_files(counts, slices)
    assert [name for group in groups for name in group] == list(counts)
    assert all(groups)
    assert len(groups) == min(slices, len(counts))


def test_slices_hold_similar_numbers_of_tests():
    counts = _counts(*([10] * 80))
    totals = [sum(counts[name] for name in group) for group in split_files(counts, 8)]
    assert totals == [100] * 8


def test_one_large_file_does_not_starve_later_slices():
    counts = _counts(1000, 10, 10, 10, 10)
    groups = split_files(counts, 4)
    assert groups[0] == ["tests/test_000.py"]
    assert len(groups) == 4


@pytest.mark.parametrize("returncode,expected", [
    (0, 0), (1, 1), (5, 5), (-11, 139), (-6, 134), (3221225477, 133)])
def test_exit_code_reports_a_crash_as_a_signal(returncode, expected):
    assert exit_code(returncode) == expected


@pytest.fixture
def small_suite(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(parallel_tests.ROOT))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.setattr(parallel_tests, "ROOT", tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    return tests


def test_parallel_run_stops_when_a_file_fails_collection(small_suite, capsys):
    (small_suite / "test_ready.py").write_text("def test_ready():\n    pass\n", encoding="utf-8")
    (small_suite / "test_broken.py").write_text("raise RuntimeError('broken collection')\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        parallel_tests.main(["--jobs", "2"])

    assert error.value.code == 2
    assert "broken collection" in capsys.readouterr().out


def test_parallel_run_preserves_node_selection_and_inherited_basetemp(small_suite, monkeypatch):
    first = small_suite / "test_first.py"
    second = small_suite / "test_second.py"
    record = """from pathlib import Path

def test_selected(tmp_path):
    with Path(__file__).with_suffix('.log').open('x', encoding='utf-8') as stream:
        stream.write(str(tmp_path))
"""
    first.write_text(record + "\ndef test_not_selected():\n    raise AssertionError\n", encoding="utf-8")
    second.write_text(record, encoding="utf-8")
    basetemp = small_suite.parent / "worker scratch"
    monkeypatch.setenv("PYTEST_ADDOPTS", f'--basetemp="{basetemp.as_posix()}"')

    assert parallel_tests.main(["--jobs", "2", "--slices", "2",
                                "tests/test_first.py::test_selected", "tests/test_second.py"]) == 0

    first_path = Path(first.with_suffix(".log").read_text(encoding="utf-8"))
    second_path = Path(second.with_suffix(".log").read_text(encoding="utf-8"))
    assert first_path.is_relative_to(basetemp / "slice-1")
    assert second_path.is_relative_to(basetemp / "slice-2")


def test_crash_report_contains_the_failed_slice_targets(small_suite):
    (small_suite / "test_assertion.py").write_text("def test_failure():\n    assert False\n", encoding="utf-8")
    crash = small_suite / "test_crash.py"
    crash.write_text("import os\n\ndef test_exit():\n    os._exit(132)\n", encoding="utf-8")
    report = small_suite.parent / "crash.json"

    assert parallel_tests.main(["--jobs", "2", "--slices", "2", "--crash-selection", str(report)]) == 132

    assert json.loads(report.read_text(encoding="utf-8")) == [str(crash)]


@pytest.mark.parametrize("options", [[], ["-vv"], ["-q"], ["-v", "--verbosity=2"]])
def test_selection_keeps_only_files_with_selected_tests_at_any_verbosity(options):
    counts, _, _ = collect(["-k", "test_slices_hold_similar_numbers_of_tests", *options])
    assert counts == {"tests/test_parallel_tests_tool.py": 1}
