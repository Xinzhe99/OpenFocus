"""The parallel test runner cuts the suite into consecutive, complete slices."""
import pytest

from tools.parallel_tests import exit_code, split_files


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
