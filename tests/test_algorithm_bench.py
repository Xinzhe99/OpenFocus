import sys
import types

import numpy as np
import pytest

from tools import algorithm_bench as bench


DIGEST = "a" * 64


def row(method="a", elapsed="1.0", digest=DIGEST, shape="(2, 3)", dtype="uint8"):
    return "\t".join((method, elapsed, digest, shape, dtype)) + "\n"


def compare(tmp_path, before, after):
    left, right = tmp_path / "before.txt", tmp_path / "after.txt"
    left.write_text(before, encoding="utf-8")
    right.write_text(after, encoding="utf-8")
    return bench.compare_reports(left, right)


def test_identical_complete_reports_pass(tmp_path, capsys):
    assert compare(tmp_path, row() + row("b"), row() + row("b")) == 0
    assert "all outputs bit-identical" in capsys.readouterr().out


@pytest.mark.parametrize("changed", [
    row(digest="b" * 64),
    row(shape="(3, 2)"),
    row(dtype="int8"),
])
def test_changed_output_or_metadata_fails(tmp_path, capsys, changed):
    assert compare(tmp_path, row(), changed) == 1
    assert "CHANGED" in capsys.readouterr().out


@pytest.mark.parametrize("before,after,missing", [
    (row(), row() + row("b"), "missing from before"),
    (row() + row("b"), row(), "missing from after"),
])
def test_method_set_must_match(tmp_path, capsys, before, after, missing):
    assert compare(tmp_path, before, after) == 1
    output = capsys.readouterr().out
    assert "b" in output and missing in output


@pytest.mark.parametrize("report", [
    "", row() + row(), row(elapsed="nan"), row(elapsed="-1"),
    row(elapsed="inf"), "a\t1\t" + DIGEST + "\n", row()[:-1],
    row(digest="z" * 64), row(shape=""), row(dtype=""),
])
def test_invalid_reports_fail(tmp_path, report):
    assert compare(tmp_path, report, row()) == 1


@pytest.mark.parametrize("before,after", [
    (row(elapsed="None", digest="SKIP", shape="", dtype=""),
     row(elapsed="None", digest="SKIP", shape="", dtype="")),
    (row(), row(elapsed="None", digest="SKIP", shape="ImportError: missing", dtype="")),
    (row(digest="a" * 16), row(digest="a" * 16)),
])
def test_unavailable_results_cannot_pass(tmp_path, capsys, before, after):
    assert compare(tmp_path, before, after) == 1
    output = capsys.readouterr().out
    assert "unavailable" in output
    assert "all outputs bit-identical" not in output


def test_failed_run_publishes_all_rows_and_returns_failure(tmp_path, monkeypatch, capsys):
    stack = tmp_path / "stack"
    stack.mkdir()
    monkeypatch.setattr(bench, "STACK_DIR", str(stack))
    monkeypatch.setattr(bench.cv2, "imread", lambda path: np.zeros((2, 2), dtype=np.uint8))
    monkeypatch.setattr(bench, "fusion_algorithms", lambda: [types.SimpleNamespace(id=x) for x in ("good", "bad")])

    class Fuser:
        def __init__(self, algorithm, **kwargs):
            self.algorithm = algorithm

        def fuse(self, frames):
            if self.algorithm == "bad":
                raise RuntimeError("backend broke")
            return frames[0]

    monkeypatch.setitem(sys.modules, "core.multi_focus_fusion", types.SimpleNamespace(MultiFocusFusion=Fuser))
    assert bench.run_benchmark("run") == 1
    report = bench.load_report(tmp_path / "run.txt")
    assert len(report["good"][1]) == 64
    assert report["bad"][1:3] == ("SKIP", "RuntimeError: backend broke")
    assert "1 succeeded, 1 failed" in capsys.readouterr().out


def test_interruption_preserves_existing_report(tmp_path, monkeypatch):
    stack = tmp_path / "stack"
    stack.mkdir()
    destination = tmp_path / "run.txt"
    destination.write_text(row(), encoding="utf-8")
    monkeypatch.setattr(bench, "STACK_DIR", str(stack))
    monkeypatch.setattr(bench.cv2, "imread", lambda path: np.zeros((2, 2), dtype=np.uint8))
    monkeypatch.setattr(bench, "fusion_algorithms", lambda: [types.SimpleNamespace(id="a")])

    class Fuser:
        def __init__(self, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setitem(sys.modules, "core.multi_focus_fusion", types.SimpleNamespace(MultiFocusFusion=Fuser))
    with pytest.raises(KeyboardInterrupt):
        bench.run_benchmark("run")
    assert destination.read_text(encoding="utf-8") == row()
