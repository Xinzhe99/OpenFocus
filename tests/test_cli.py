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
