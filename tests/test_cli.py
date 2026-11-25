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

pytestmark = pytest.mark.usefixtures("isolated_process_env")


def run_cli(*argv):
    return subprocess.run(
        [sys.executable, MAIN, *argv],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300,
    )


def test_cli_ignores_inherited_data_and_log_overrides(tmp_path, monkeypatch):
    from tests.process_isolation import sandbox_env

    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "settings.ini"
    sentinel.write_bytes(b"unchanged user data\n")
    log = outside / "existing.log"
    log.write_bytes(b"unchanged diagnostic log\n")
    monkeypatch.setenv("OPENFOCUS_DATA_DIR", str(outside))
    monkeypatch.setenv("OPENFOCUS_LOG_DIR", str(outside))
    env = sandbox_env(tmp_path / "child", {
        "OPENFOCUS_DATA_DIR": str(outside), "OPENFOCUS_LOG_DIR": str(outside),
    })
    result = subprocess.run([sys.executable, MAIN, "--help"], cwd=PROJECT_ROOT,
                            env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == b"unchanged user data\n"
    assert log.read_bytes() == b"unchanged diagnostic log\n"
    assert {path.name for path in outside.iterdir()} == {"settings.ini", "existing.log"}
    assert list((tmp_path / "child" / "logs").glob("openfocus-*.log"))

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


@pytest.mark.parametrize('scale', ['single', 'dual'])
def test_cli_stackmff_inference_scale_reaches_fusion_and_depth(tmp_path, monkeypatch, scale):
    from core import cli, depth_map, multi_focus_fusion

    seen = {}

    def fuse(fusion, *, input_source, **kwargs):
        seen['fusion'] = dict(fusion.fusion_parameters)
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def focus_index(frames, method, *, parameters, **kwargs):
        seen['depth'] = dict(parameters)
        return np.zeros((8, 8), dtype=np.float32)

    monkeypatch.setattr(multi_focus_fusion.MultiFocusFusion, '_validate_environment', lambda self: None)
    monkeypatch.setattr(multi_focus_fusion.MultiFocusFusion, 'fuse', fuse)
    monkeypatch.setattr(depth_map, 'compute_focus_index', focus_index)
    output = tmp_path / 'out.png'
    argv = ['-i', 'in', '-o', str(output), '-m', 'stackmffv3', '--align', 'none',
            '--cpu', '--ai-mode', 'hard', '--depth-map']
    if scale == 'dual':
        argv.extend(['--param', 'inference_scale="dual"'])
    args = cli.build_parser().parse_args(argv)
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (20, 120)]
    cli._fuse_stack_admitted_body(frames, args)
    cli._save_depth(args, str(output))
    expected = {'mode': 'hard', 'inference_scale': scale}
    assert seen == {'fusion': expected, 'depth': expected}
    assert (tmp_path / 'out_depth.png').is_file()


def test_cli_rejects_invalid_stackmff_inference_scale():
    from core import cli

    args = cli.build_parser().parse_args([
        '-i', 'in', '-o', 'out.png', '-m', 'stackmffv3', '--param', 'inference_scale="triple"'])
    with pytest.raises(ValueError, match='inference_scale'):
        cli._resolve_algorithm_arguments(args)


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


def test_cli_defaults_to_openfocus_b_and_requires_experimental_opt_in(tmp_path):
    import json

    stack = tmp_path / 'stack'
    stack.mkdir()
    for index in range(2):
        assert cv2.imwrite(str(stack / f'{index}.png'), np.zeros((8, 8), np.uint8))
    output = tmp_path / 'out.png'
    args = ('--input', str(stack), '--output', str(output), '--dry-run')
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr
    assert 'method=openfocus_b' in result.stdout
    result = run_cli(*args, '--method', 'dct')
    assert result.returncode == 2
    assert 'DCT' in result.stderr and '--experimental' in result.stderr
    result = run_cli(*args, '--method', 'dct', '--experimental')
    assert result.returncode == 0, result.stderr
    assert 'method=dct' in result.stdout
    assert not output.exists()

    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'stack': {'method': 'dct'}}), encoding='utf-8')
    batch = ('--input', str(stack), '--output-dir', str(tmp_path / 'out'),
             '--config', str(config), '--dry-run')
    result = run_cli(*batch)
    assert result.returncode == 2
    assert 'config stack' in result.stderr and '--experimental' in result.stderr
    result = run_cli(*batch, '--experimental')
    assert result.returncode == 0, result.stderr
    assert 'method=dct' in result.stdout


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
                "--experimental", "--method", "guided_filter", "--align", "ecc", "--threads", "2")
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
                "--output-dir", str(out), "--experimental", "--method", "guided_filter",
                "--threads", "2")
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(out)) == ["stackA.png", "stackB.png"]


@pytest.mark.parametrize("cancel_during", ["watermark", "encode"])
def test_cli_batch_cancels_watermarked_export(tmp_path, monkeypatch, capsys, cancel_during):
    import json
    from core import cli
    from utils import image_utils, watermark

    folders = [tmp_path / "first", tmp_path / "second"]
    for folder in folders:
        folder.mkdir()
        _write_stack(folder, n=2)
    output = tmp_path / "out"
    cancelled = False
    run_admitted = cli._run_cli_admitted

    def run_with_cancellation(args):
        args._should_cancel = lambda: cancelled
        return run_admitted(args)

    module, function = ((watermark, "burn") if cancel_during == "watermark"
                        else (image_utils, "imwrite_auto"))
    original = getattr(module, function)

    def cancel_on_export(*args, **kwargs):
        nonlocal cancelled
        cancelled = True
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "_run_cli_admitted", run_with_cancellation)
    monkeypatch.setattr(module, function, cancel_on_export)
    result = cli.run_cli(["-i", *(str(folder) for folder in folders), "-d", str(output),
                          "--watermark", "OpenFocus", "--watermark-size", "7",
                          "--cpu", "--threads", "1", "--json-progress"])
    assert result == cli.EXIT_ERROR
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["name"] for event in events if event["event"] == "folder"] == ["first"]
    assert events[-1]["event"] == "done"
    assert events[-1]["cancelled"] is True
    assert not list(output.iterdir())


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
                     "--experimental", "--method", "guided_filter", "--cpu", "--threads", "1",
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
                    "--experimental", "--method", "guided_filter", "--group-size", "3", "--depth-map")
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
                     "--experimental", "--method", "guided_filter", "--cpu", "--group-size", "3",
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
                     "--output-dir", str(output), "--experimental", "--method", "guided_filter",
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
