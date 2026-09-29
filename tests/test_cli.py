"""End-to-end CLI tests: exit codes and a full synthetic fusion run."""
import os
import subprocess
import sys

import pytest
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(PROJECT_ROOT, "main.py")

QUrl = pytest.importorskip("PyQt6.QtCore").QUrl


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


# --- file:// URI normalisation ------------------------------------------------

def test_dropped_file_uri_maps_back_to_the_file(tmp_path):
    """A file:// URI (Dock drop, single-instance forwarding) must name a path
    that exists — that only holds if the scheme *and* the URI quoting go away."""
    from core.app import PlatformFileHandler

    target = tmp_path / "My Stacks" / "f 0.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"png")

    uri = QUrl.fromLocalFile(str(target)).toString()
    assert uri.startswith("file://")
    assert os.path.normcase(PlatformFileHandler.normalize(uri)) == os.path.normcase(str(target))


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="drive letters are Windows paths")
def test_windows_file_uri_has_no_phantom_leading_slash():
    from core.app import PlatformFileHandler

    # Stripping the scheme by hand left '/C:/...', which no file API accepts
    assert PlatformFileHandler.normalize("file:///C:/stacks/demo/f0.png") == "C:/stacks/demo/f0.png"
    assert PlatformFileHandler.normalize("file:///C:/Users/me/My%20Stacks/f0.png") == r"C:/Users/me/My Stacks/f0.png"


def test_posix_uris_and_mac_triple_slash():
    from core.app import PlatformFileHandler

    assert PlatformFileHandler.normalize("file:///Users/me/stack/f0.png") == "/Users/me/stack/f0.png"
    assert PlatformFileHandler.normalize("///Users/me/stack/f0.png") == "/Users/me/stack/f0.png"


def test_plain_paths_keep_a_literal_percent():
    """Command-line arguments are paths, not URIs: unquoting them corrupts
    folders really named '100%'."""
    from core.app import PlatformFileHandler

    assert PlatformFileHandler.normalize(r"C:\stacks\100%\f0.png") == r"C:\stacks\100%\f0.png"
    assert PlatformFileHandler.normalize("/stacks/100%/f0.png") == "/stacks/100%/f0.png"
    assert PlatformFileHandler.normalize("") == ""
