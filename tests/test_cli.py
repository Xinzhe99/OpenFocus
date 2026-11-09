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

@pytest.mark.parametrize('method', ['openfocus_a', 'openfocus_b', 'openfocus_c'])
def test_desc_orders_sources_before_openfocus_registration(monkeypatch, method):
    from core import cli, registration, multi_focus_fusion
    from utils import image_utils

    frames = [np.full((4, 4), value, np.uint8) for value in (10, 20, 30)]
    observed = {}

    class Registration:
        def __init__(self, **kwargs):
            pass

        def process(self, images, **kwargs):
            observed['registration'] = list(images)
            return images

    class Fusion:
        def __init__(self, **kwargs):
            observed['sort'] = kwargs['sort_mode']

        def fuse(self, input_source, **kwargs):
            observed['fusion'] = list(input_source)
            return input_source[0]

        def get_info(self):
            return {'device': 'test'}

    monkeypatch.setattr(registration, 'ImageRegistration', Registration)
    monkeypatch.setattr(multi_focus_fusion, 'MultiFocusFusion', Fusion)
    monkeypatch.setattr(cli, '_transparent_region', lambda images, args: (images, None))
    monkeypatch.setattr(image_utils, 'prepare_fusion_input', lambda images, cancel: (images, False))
    args = cli.build_parser().parse_args(['-i', 'in', '-o', 'out.png', '-m', method,
                                           '--align', 'openfocus', '--sort', 'desc'])
    cli._fuse_stack_admitted_body(frames, args)
    assert [int(frame[0, 0]) for frame in observed['registration']] == [30, 20, 10]
    assert [int(frame[0, 0]) for frame in observed['fusion']] == [30, 20, 10]
    assert [int(frame[0, 0]) for frame in args._depth_frames] == [30, 20, 10]
    assert observed['sort'] == 'asc'
    assert [int(frame[0, 0]) for frame in frames] == [10, 20, 30]


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


def test_grouped_cli_preserves_16_bit_and_reports_progress(tmp_path):
    import json

    stack = tmp_path / "stack"
    stack.mkdir()
    y, x = np.indices((64, 80))
    for index in range(7):
        pattern = ((x * 439 + y * 271 + index * 101) % 60000).astype(np.uint16)
        assert cv2.imwrite(str(stack / f"frame{index}.png"),
                           np.repeat(pattern[..., None], 3, axis=2))
    out = tmp_path / "fused.tif"
    result = run_cli("--input", str(stack), "--output", str(out),
                     "--method", "guided_filter", "--cpu", "--threads", "1",
                     "--no-exposure-match", "--group-size", "3",
                     "--group-overlap", "1", "--json-progress")
    assert result.returncode == 0, result.stderr
    fused = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert fused is not None and fused.dtype == np.uint16 and fused.shape == (64, 80, 3)
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert any(event.get("event") == "group" for event in events)
    assert "approximate" in result.stderr and "temporary disk" in result.stderr


def test_grouped_cli_rejects_depth_and_nonreducing_overlap(tmp_path):
    stack = tmp_path / "stack"
    stack.mkdir()
    output = tmp_path / "fused.png"
    depth = run_cli("--input", str(stack), "--output", str(output),
                    "--method", "guided_filter", "--group-size", "3", "--depth-map")
    assert depth.returncode == 2 and "original frame indices" in depth.stderr
    overlap = run_cli("--input", str(stack), "--output", str(output),
                      "--group-size", "3", "--group-overlap", "2")
    assert overlap.returncode == 2 and "group-overlap" in overlap.stderr


def test_grouped_cli_rejects_nonlocal_alignment_reference(tmp_path):
    stack = tmp_path / "stack"
    stack.mkdir()
    _write_stack(stack, n=5)
    output = tmp_path / "fused.png"
    result = run_cli("--input", str(stack), "--output", str(output),
                     "--method", "guided_filter", "--cpu", "--group-size", "3",
                     "--align", "openfocus", "--align-param", "reference_frame=2")
    assert result.returncode != 0
    assert "stack-wide alignment reference" in result.stderr
    assert not output.exists()


def test_grouped_cli_batch_processes_each_folder(tmp_path):
    folders = [tmp_path / "a", tmp_path / "b"]
    for folder in folders:
        folder.mkdir()
        _write_stack(folder, n=5)
    output = tmp_path / "results"
    result = run_cli("--input", *(str(folder) for folder in folders),
                     "--output-dir", str(output), "--method", "guided_filter",
                     "--cpu", "--threads", "1", "--group-size", "3")
    assert result.returncode == 0, result.stderr
    for name in ("a.png", "b.png"):
        image = cv2.imread(str(output / name), cv2.IMREAD_UNCHANGED)
        assert image is not None and image.dtype == np.uint8

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
