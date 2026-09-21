"""End-to-end CLI tests: exit codes and a full synthetic fusion run."""
import os
import subprocess
import sys

import pytest
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(PROJECT_ROOT, "main.py")


def run_cli(*argv):
    return subprocess.run(
        [sys.executable, MAIN, *argv],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300,
    )


def test_help_exits_zero(tmp_path):
    r = run_cli("--help")
    assert r.returncode == 0
    assert "--input" in r.stdout


def test_usage_error_exit_code(tmp_path):
    r = run_cli("--input", "x", "--output", str(tmp_path / "y.txt"))
    assert r.returncode == 2


def test_missing_input_exit_code(tmp_path):
    r = run_cli("--input", str(tmp_path / "nope"), "--output", str(tmp_path / "y.png"))
    assert r.returncode == 1


def test_bad_downscale_exit_code(tmp_path):
    r = run_cli("--downscale", "0", "--input", "x", "--output", str(tmp_path / "y.png"))
    assert r.returncode == 2


def test_full_pipeline_synthetic_stack(tmp_path):
    stack = tmp_path / "stack"
    stack.mkdir()
    xx, yy = np.meshgrid(np.arange(160), np.arange(120))
    pattern = ((np.sin(xx / 5.0) * np.sin(yy / 5.0) + 1) / 2 * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(pattern, (31, 31), 0)
    for i, frame in enumerate((pattern, blurred)):
        bgr = frame[..., None].repeat(3, axis=2)
        cv2.imwrite(str(stack / f"frame_{i:02d}.png"), bgr)

    out = tmp_path / "out" / "fused.png"
    r = run_cli("--input", str(stack), "--output", str(out),
                "--method", "guided_filter", "--align", "ecc", "--threads", "2")
    assert r.returncode == 0, r.stderr
    assert out.exists() and out.stat().st_size > 0


def _write_stack(folder, n=3):
    import numpy as np, cv2
    xx, yy = np.meshgrid(np.arange(140), np.arange(110))
    pattern = ((np.sin(xx / 6.0) * np.sin(yy / 6.0) + 1) / 2 * 255).astype(np.uint8)
    for i in range(n):
        shifted = np.roll(pattern, i * 2, axis=1)
        cv2.imwrite(str(folder / f"f{i}.png"), shifted[..., None].repeat(3, axis=2))


def test_batch_output_dir(tmp_path):
    import numpy as np
    stacks = tmp_path / "stacks"
    out = tmp_path / "out"
    for name in ("stackA", "stackB"):
        d = stacks / name
        d.mkdir(parents=True)
        _write_stack(d)

    r = run_cli("--input", str(stacks / "stackA"), str(stacks / "stackB"),
                "--output-dir", str(out), "--method", "guided_filter",
                "--threads", "2")
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(out)) == ["stackA.png", "stackB.png"]


def test_batch_no_folders_exit_code(tmp_path):
    r = run_cli("--input", str(tmp_path / "nofolder"), "--output-dir", str(tmp_path / "out"))
    assert r.returncode == 2
