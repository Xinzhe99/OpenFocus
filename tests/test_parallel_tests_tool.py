"""The parallel test runner cuts the suite into consecutive, complete slices."""
import os

import pytest

from tools.parallel_tests import (collect, exit_code, parse_listing, slice_options,
                                  split_files, without_option)


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


def test_listing_counts_tests_per_file_in_either_format():
    listing = "\n".join([
        "tests/test_a.py::test_one",
        "tests/test_a.py::TestGroup::test_two[case 1]",
        "tests\\test_b.py::test_three",
        "tests/test_c.py: 12",
        "",
        "3 tests collected in 0.10s",
    ])
    assert parse_listing(listing) == ({"tests/test_a.py": 2, "tests/test_b.py": 1, "tests/test_c.py": 12}, [])


def test_listing_names_files_that_failed_to_collect_once():
    listing = "\n".join([
        "tests/test_a.py::test_one",
        "==== ERRORS ====",
        "ERROR tests/test_broken.py - ModuleNotFoundError: No module named 'missing'",
        "ERROR tests/test_broken.py",
        "ERROR collecting something else",
    ])
    assert parse_listing(listing) == ({"tests/test_a.py": 1}, ["tests/test_broken.py"])


@pytest.mark.parametrize("options,expected", [
    (["--basetemp=/tmp/run", "-x"], ["--basetemp=" + os.path.join("/tmp/run", "slice-3"), "-x"]),
    (["-x", "--basetemp", "/tmp/run"], ["-x", "--basetemp", os.path.join("/tmp/run", "slice-3")]),
    (["--timeout=180", "-k", "name"], ["--timeout=180", "-k", "name"]),
])
def test_each_slice_gets_its_own_base_temp(options, expected):
    assert slice_options(options, 3) == expected


def test_collection_ignores_the_base_temp_option():
    assert without_option(["--basetemp=/tmp/run", "-x", "--basetemp", "/tmp/run", "-q"],
                          "--basetemp") == ["-x", "-q"]


@pytest.mark.parametrize("options", [[], ["-vv"], ["-q"], ["-v", "--verbosity=2"]])
def test_selection_keeps_only_files_with_selected_tests_at_any_verbosity(options):
    counts = collect(["-k", "test_slices_hold_similar_numbers_of_tests", *options])
    assert counts == {"tests/test_parallel_tests_tool.py": 1}
