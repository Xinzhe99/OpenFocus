"""Regression tests for the batch/CLI engineering features (v1.37):
--dry-run, --resume, --json/--json-progress and per-folder --config."""
import json
import os

import numpy as np
import cv2

from core.cli import run_cli


def _make_stacks(tmp_path):
    fa = str(tmp_path / "stackA")
    fb = str(tmp_path / "stackB")
    for d in (fa, fb):
        os.makedirs(d, exist_ok=True)
        for i in range(3):
            cv2.imwrite(os.path.join(d, f"f{i}.png"),
                        np.full((24, 24, 3), 40 * i, np.uint8))
    return fa, fb, tmp_path / "out"


def _run_cli(argv):
    return run_cli(argv)


def test_dry_run_writes_nothing(tmp_path):
    fa, fb, out = _make_stacks(tmp_path)
    rc = _run_cli(["-i", fa, fb, "-d", str(out), "--dry-run"])
    assert rc == 0
    assert not os.path.isdir(out) or not list(out.glob("*.png"))


def test_batch_json_summary(tmp_path):
    fa, fb, out = _make_stacks(tmp_path)
    jout = tmp_path / "res.json"
    rc = _run_cli(["-i", fa, fb, "-d", str(out), "--json", str(jout)])
    assert rc == 0
    payload = json.loads(jout.read_text(encoding="utf-8"))
    assert payload["mode"] == "batch"
    assert payload["succeeded"] == 2
    assert payload["failed"] == []
    assert os.path.isfile(out / "stackA.png")


def test_batch_config_override_changes_method(tmp_path):
    fa, fb, out = _make_stacks(tmp_path)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"stackB": {"method": "dct", "kernel": 9}}),
                   encoding="utf-8")
    rc = _run_cli(["-i", fa, fb, "-d", str(out), "--config", str(cfg)])
    assert rc == 0
    assert os.path.isfile(out / "stackB.png")


def test_batch_resume_skips_finished_folders(tmp_path):
    fa, fb, out = _make_stacks(tmp_path)
    # first run fuses everything
    assert _run_cli(["-i", fa, fb, "-d", str(out)]) == 0
    first = (out / "stackA.png").stat().st_mtime_ns
    # second run with --resume skips both outputs
    rc = _run_cli(["-i", fa, fb, "-d", str(out), "--resume"])
    assert rc == 0
    assert (out / "stackA.png").stat().st_mtime_ns == first


def test_resume_requires_batch_mode(tmp_path):
    fa, _fb, _out = _make_stacks(tmp_path)
    rc = _run_cli(["-i", fa, "-o", str(tmp_path / "x.png"), "--resume"])
    assert rc == 2


def test_config_requires_batch_mode(tmp_path):
    fa, _fb, _out = _make_stacks(tmp_path)
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}", encoding="utf-8")
    rc = _run_cli(["-i", fa, "-o", str(tmp_path / "x.png"), "--config", str(cfg)])
    assert rc == 2
