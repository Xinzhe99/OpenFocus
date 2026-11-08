"""Regression tests for the batch/CLI engineering features (v1.37):
--dry-run, --resume, --json/--json-progress and per-folder --config."""
import json
import os

import numpy as np
import cv2
import pytest


def _extension_fuse(fusion, images, img_resize=None, *, gain=1.0, selection="first", **execution):
    frame = images[0] if selection == "first" else images[-1]
    return np.asarray(frame, dtype=np.float32) * gain


def _extension_depth(images, *, gain=1.0, selection="first", **execution):
    index = 0 if selection == "first" else len(images) - 1
    return np.full(images[0].shape[:2], index * gain, dtype=np.float32)


def _extension_align(registration, images, *, offset=0, **execution):
    return [image + offset for image in images]


@pytest.fixture
def cli_extension(monkeypatch):
    from core import algorithms
    monkeypatch.setattr(algorithms, "_FUSION", dict(algorithms._FUSION))
    monkeypatch.setattr(algorithms, "_REGISTRATION", dict(algorithms._REGISTRATION))
    parameters = (algorithms.ParameterSpec("gain", "gain", 1.0, 0.0, 2.0, .1, 2),
                  algorithms.ParameterSpec("selection", "selection", "first", choices=("first", "last")))
    algorithms.register_fusion(algorithms.FusionAlgorithm(
        "cli_extension", "cli_extension", "CLI extension", "CE",
        "tests.test_cli_engineering:_extension_fuse", parameters=parameters,
        depth=algorithms.DepthSpec("tests.test_cli_engineering:_extension_depth", parameters)))
    algorithms.register_registration(algorithms.RegistrationAlgorithm(
        "cli_alignment", "cli_alignment", "CA", "tests.test_cli_engineering:_extension_align",
        parameters=(algorithms.ParameterSpec("offset", "offset", 0, 0, 10),)))


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

def test_batch_dry_run_summary_counts_validated_stacks(tmp_path):
    fa, fb, out = _make_stacks(tmp_path)
    summary = tmp_path / "summary.json"
    assert _run_cli(["-i", fa, fb, "-d", str(out), "--dry-run",
                     "--json", str(summary)]) == 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["succeeded"] == 2
    assert payload["failed"] == []
    assert not out.exists()


@pytest.mark.parametrize('batch', [False, True])
def test_grouped_dry_run_rejects_alpha_before_processing(tmp_path, capsys, batch):
    folder = tmp_path / 'stack'
    folder.mkdir()
    frame = np.zeros((24, 24, 4), dtype=np.uint8)
    frame[2:-2, 2:-2, 3] = 255
    for index in range(2):
        assert cv2.imwrite(str(folder / f'{index}.png'), frame)
    destination = ['-d', str(tmp_path / 'out')] if batch else ['-o', str(tmp_path / 'out.png')]
    assert _run_cli(['-i', str(folder), *destination, '--group-size', '2', '--dry-run']) != 0
    assert 'alpha-channel' in capsys.readouterr().err
    assert not (tmp_path / 'out.png').exists()
    assert not (tmp_path / 'out').exists()


def test_grouped_cli_real_registration_keeps_singleton_coordinates(tmp_path):
    folder = tmp_path / 'stack'
    folder.mkdir()
    rng = np.random.default_rng(131)
    frame = rng.integers(0, 256, (64, 68, 3), dtype=np.uint8)
    for index in range(5):
        assert cv2.imwrite(str(folder / f'{index}.png'), frame)
    output = tmp_path / 'out.png'
    assert _run_cli(['-i', str(folder), '-o', str(output), '--group-size', '2',
                     '--group-overlap', '0', '--align', 'openfocus',
                     '--method', 'openfocus_c', '--cpu']) == 0
    np.testing.assert_array_equal(cv2.imread(str(output)), frame)

def test_cli_alignment_report_records_registration_and_restores_context(tmp_path):
    import core.registration as registration

    folder = str(tmp_path / 'report-stack')
    os.makedirs(folder)
    for index in range(3):
        assert cv2.imwrite(os.path.join(folder, f'f{index}.png'),
                           np.full((24, 24, 3), 80, np.uint8))
    report = tmp_path / 'report.json'
    prior = []
    token = registration.alignment_report.set(prior)
    try:
        result = _run_cli(['-i', folder, '-o', str(tmp_path / 'fused.png'),
                           '--align', 'openfocus', '--method', 'guided_filter',
                           '--cpu', '--alignment-report', str(report)])
        assert result == 0
        assert registration.alignment_report.get() is prior
        assert prior == []
    finally:
        registration.alignment_report.reset(token)
    payload = json.loads(report.read_text(encoding='utf-8'))
    assert payload['method'] == 'openfocus'
    assert payload['input'] == [folder]
    assert payload['frame_count'] == 3
    assert {entry['frame_index'] for entry in payload['entries']} == {0, 1, 2}
    assert all(entry['method'] == 'openfocus' for entry in payload['entries'])
    assert any(entry['status'] == 'reference' for entry in payload['entries'])


@pytest.mark.parametrize('extra', [
    ['--group-size', '2'], ['--dry-run'], ['--suggest-duplicates'],
])
def test_alignment_report_rejects_unsupported_modes(tmp_path, extra):
    folder, _, _ = _make_stacks(tmp_path)
    report = tmp_path / 'report.json'
    assert _run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'),
                     '--alignment-report', str(report), *extra]) == 2
    assert not report.exists()


def test_alignment_report_rejects_batch(tmp_path):
    folder, _, out = _make_stacks(tmp_path)
    assert _run_cli(['-i', folder, '-d', str(out), '--alignment-report',
                     str(tmp_path / 'report.json')]) == 2
    assert not out.exists()


@pytest.mark.parametrize('target', ['frame', 'output', 'depth', 'summary', 'dust'])
def test_alignment_report_never_overwrites_other_paths(tmp_path, target):
    folder, _, _ = _make_stacks(tmp_path)
    frame = os.path.join(folder, 'f0.png')
    output = str(tmp_path / 'out.png')
    depth = str(tmp_path / 'out_depth.png')
    summary = str(tmp_path / 'summary.json')
    paths = {'frame': frame, 'output': output, 'depth': depth, 'summary': summary, 'dust': frame}
    before = open(frame, 'rb').read()
    args = ['-i', folder, '-o', output, '--align', 'openfocus', '--depth-map', '--json', summary,
            '--alignment-report', paths[target]]
    if target == 'dust':
        args += ['--dust-map', frame]
    assert _run_cli(args) != 0
    assert open(frame, 'rb').read() == before
    assert not os.path.exists(output)


@pytest.mark.parametrize('hardlink', [False, True])
def test_alignment_report_does_not_overwrite_sensor_mask(tmp_path, hardlink):
    folder, _, _ = _make_stacks(tmp_path)
    mask = tmp_path / 'mask.png'
    assert cv2.imwrite(str(mask), np.zeros((24, 24), dtype=np.uint8))
    report = mask
    if hardlink:
        report = tmp_path / 'linked.json'
        try:
            os.link(mask, report)
        except OSError:
            pytest.skip('hard links are unavailable')
    original = mask.read_bytes()
    result = _run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'),
                       '--align', 'openfocus', '--defect-mask-path', str(mask),
                       '--alignment-report', str(report)])
    assert result == 2
    assert mask.read_bytes() == original
    assert not (tmp_path / 'out.png').exists()


def test_alignment_report_requires_alignment(tmp_path):
    folder, _, _ = _make_stacks(tmp_path)
    report = tmp_path / 'report.json'
    assert _run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'),
                     '--alignment-report', str(report)]) == 2
    assert not report.exists()


def test_alignment_report_rejects_hardlinked_input(tmp_path):
    folder, _, _ = _make_stacks(tmp_path)
    frame = os.path.join(folder, 'f0.png')
    report = tmp_path / 'linked.json'
    try:
        os.link(frame, report)
    except OSError:
        pytest.skip('hard links are unavailable')
    original = report.read_bytes()
    assert _run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'), '--align', 'openfocus',
                     '--alignment-report', str(report)]) != 0
    assert report.read_bytes() == original
    assert not (tmp_path / 'out.png').exists()



def test_alignment_report_failure_restores_context_and_preserves_report(tmp_path, monkeypatch):
    import core.cli as cli
    import core.registration as registration

    folder, _, _ = _make_stacks(tmp_path)
    report = tmp_path / 'report.json'
    report.write_text('previous', encoding='utf-8')
    prior = []
    token = registration.alignment_report.set(prior)
    def fail_fusion(images, args):
        registration.alignment_report.get().append({'frame_index': 0, 'status': 'aligned'})
        raise RuntimeError('registration failed')
    monkeypatch.setattr(cli, '_fuse_stack', fail_fusion)
    try:
        assert _run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'), '--align', 'openfocus',
                         '--alignment-report', str(report)]) == 1
        assert registration.alignment_report.get() is prior
        assert prior == []
    finally:
        registration.alignment_report.reset(token)
    assert report.read_text(encoding='utf-8') == 'previous'
    assert not (tmp_path / 'out.png').exists()

def test_alignment_report_atomic_write_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    from core.cli import _write_alignment_report

    report = tmp_path / 'report.json'
    report.write_text('existing', encoding='utf-8')
    def fail_replace(*args):
        raise OSError('replace failed')
    monkeypatch.setattr(os, 'replace', fail_replace)
    with pytest.raises(OSError):
        _write_alignment_report(str(report), {'entries': []})
    assert report.read_text(encoding='utf-8') == 'existing'
    assert list(tmp_path.iterdir()) == [report]


def test_batch_json_summary(tmp_path):
    fa, fb, out = _make_stacks(tmp_path)
    jout = tmp_path / "res.json"
    rc = _run_cli(["-i", fa, fb, "-d", str(out), "-m", "guided_filter", "--json", str(jout)])
    assert rc == 0
    payload = json.loads(jout.read_text(encoding="utf-8"))
    assert payload["mode"] == "batch"
    assert payload["succeeded"] == 2
    assert payload["failed"] == []
    assert os.path.isfile(out / "stackA.png")


def test_strict_alignment_reports_rejected_frames_in_single_and_batch(tmp_path, capsys):
    fa, fb, out = _make_stacks(tmp_path)
    option = ["--align", "homography", "--align-param", 'failure_policy="fail"',
              "-m", "guided_filter", "--cpu"]
    single = tmp_path / "single.png"
    assert _run_cli(["-i", fa, "-o", str(single), *option]) == 1
    capsys.readouterr()
    assert not single.exists()
    assert _run_cli(["-i", fa, fb, "-d", str(out), *option]) == 1
    error = capsys.readouterr().err
    assert "stackA" in error and "stackB" in error
    assert not list(out.glob("*.png"))


def test_cli_reference_frame_outside_stack_reports_failure(tmp_path, capsys):
    fa, fb, out = _make_stacks(tmp_path)
    option = ["--align", "openfocus", "--align-param", "reference_frame=4"]
    assert _run_cli(["-i", fa, "-o", str(tmp_path / "single.png"), *option]) == 1
    assert "reference frame 4" in capsys.readouterr().err
    assert _run_cli(["-i", fa, fb, "-d", str(out), *option]) == 1
    assert "reference frame 4" in capsys.readouterr().err


def test_batch_config_override_changes_method(tmp_path):
    fa, fb, out = _make_stacks(tmp_path)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"stackB": {"method": "dct", "kernel": 9}}),
                   encoding="utf-8")
    rc = _run_cli(["-i", fa, fb, "-d", str(out), "-m", "guided_filter", "--config", str(cfg)])
    assert rc == 0
    assert os.path.isfile(out / "stackB.png")


def test_batch_resume_skips_finished_folders(tmp_path):
    fa, fb, out = _make_stacks(tmp_path)
    # first run fuses everything
    assert _run_cli(["-i", fa, fb, "-d", str(out), "-m", "guided_filter"]) == 0
    first = (out / "stackA.png").stat().st_mtime_ns
    # second run with --resume skips both outputs
    rc = _run_cli(["-i", fa, fb, "-d", str(out), "-m", "guided_filter", "--resume"])
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


def test_multiple_folders_require_output_dir(tmp_path, capsys):
    fa, fb, _out = _make_stacks(tmp_path)
    rc = _run_cli(["-i", fa, fb, "-o", str(tmp_path / "single.png")])
    assert rc == 2
    assert "--output-dir" in capsys.readouterr().err


def test_exposure_match_defaults_on_and_can_be_disabled(tmp_path):
    """v1.57: the pipeline-level exposure match is on by default, and
    --no-exposure-match restores the old brightness behaviour."""
    from core.cli import build_parser
    parser = build_parser()
    assert parser.parse_args(["-i", "x", "-o", "y.png"]).exposure_match is True
    assert parser.parse_args(["-i", "x", "-o", "y.png",
                              "--no-exposure-match"]).exposure_match is False
    assert parser.parse_args(["-i", "x", "-o", "y.png",
                              "--exposure-match"]).exposure_match is True


def test_generic_parameters_override_legacy_flags():
    from core.cli import build_parser, _resolve_algorithm_arguments
    args = build_parser().parse_args(["-i", "x", "-o", "x.png", "-m", "dct",
                                      "--kernel", "11", "--param", "kernel_size=9"])
    _resolve_algorithm_arguments(args)
    assert args._fusion_parameters == {"kernel_size": 9}


@pytest.mark.parametrize("option,value", [
    ("--param", "unknown=1"), ("--param", "kernel_size=true"),
    ("--param", "kernel_size=NaN"), ("--param", "kernel_size=8"),
    ("--param", "kernel_size=oops"), ("--align-param", "unknown=1"),
])
def test_invalid_generic_parameter_is_usage_error(tmp_path, option, value):
    assert run_cli(["-i", "missing", "-o", str(tmp_path / "out.png"), "-m", "dct",
                    option, value]) == 2


def test_new_method_numeric_enum_alignment_depth_unicode_and_u16(tmp_path, cli_extension):
    from utils.image_utils import imwrite_auto
    source = tmp_path / "图像"
    source.mkdir()
    for i, value in enumerate((10000, 40000)):
        assert imwrite_auto(str(source / f"帧{i}.tif"), np.full((24, 24, 3), value, np.uint16))
    outputs = []
    for gain, selection in ((1.0, "first"), (.5, "last")):
        out = tmp_path / f"结果{selection}.tif"
        assert run_cli(["-i", str(source), "-o", str(out), "-m", "cli_extension",
                        "--align", "cli_alignment", "--align-param", "offset=1",
                        "--param", f"gain={gain}", "--param", f'selection="{selection}"',
                        "--no-exposure-match", "--depth-map", "--depth-colormap", "gray"]) == 0
        pixels = cv2.imdecode(np.fromfile(str(out), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert pixels.dtype == np.uint16
        outputs.append(pixels)
        depth = out.with_name(out.stem + "_depth.png")
        assert depth.is_file()
        depth_pixels = cv2.imdecode(np.fromfile(str(depth), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert abs(float(depth_pixels.mean()) - (0 if selection == "first" else 127.5)) <= 1
    assert outputs[0].mean() < outputs[1].mean()


def test_non_ascii_path_survives_a_legacy_stdout_encoding(tmp_path):
    """A cp1252 stdout must not abort a render over a Chinese folder name.

    Windows keeps the locale code page when output is redirected, and the
    progress line names the source path. The CLI reconfigured only redirected
    streams, so this runs the real entry point in a subprocess.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    from utils.image_utils import imwrite_auto
    source = tmp_path / "图像栈"
    source.mkdir()
    for index, value in enumerate((90, 170)):
        assert imwrite_auto(str(source / f"帧{index}.png"),
                            np.full((24, 24, 3), value, np.uint8))
    output = tmp_path / "结果.png"
    environment = dict(os.environ, PYTHONIOENCODING="cp1252",
                       OPENFOCUS_LOG_DIR=str(tmp_path / "logs"),
                       PYTHONPATH=str(root), KMP_DUPLICATE_LIB_OK="TRUE")
    result = subprocess.run(
        [sys.executable, str(root / "main.py"), "-i", str(source),
         "-o", str(output), "-m", "dct", "--threads", "1"],
        cwd=str(root), env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180)
    assert result.returncode == 0, result.stderr
    assert output.is_file()


def test_batch_config_generic_dictionaries_execute(tmp_path, cli_extension):
    fa, fb, out = _make_stacks(tmp_path)
    cfg = tmp_path / "generic.json"
    cfg.write_text(json.dumps({"stackB": {
        "method": "cli_extension", "align": "cli_alignment", "exposure_match": False,
        "fusion_parameters": {"gain": .5, "selection": "last"},
        "registration_parameters": {"offset": 2}}}), encoding="utf-8")
    assert run_cli(["-i", fa, fb, "-d", str(out), "-m", "guided_filter",
                    "--config", str(cfg)]) == 0
    pixels = cv2.imread(str(out / "stackB.png"))
    assert np.all(pixels == 41)


def test_explicit_unavailable_method_does_not_change_id(tmp_path, monkeypatch, capsys):
    from core import algorithms
    from dataclasses import replace
    monkeypatch.setattr(algorithms, "_FUSION", dict(algorithms._FUSION))
    spec = algorithms.get_fusion("guided_filter")
    algorithms._FUSION["guided_filter"] = replace(spec, environment="tests.test_cli_engineering:_unavailable")
    fa, _fb, _out = _make_stacks(tmp_path)
    assert run_cli(["-i", fa, "-o", str(tmp_path / "out.png"), "-m", "guided_filter"]) == 1
    assert "unavailable test backend" in capsys.readouterr().err


def _unavailable(fusion):
    raise RuntimeError("unavailable test backend")


def test_suggest_duplicates_reports_ordered_files_without_modifying_them(tmp_path, capsys):
    image = np.random.default_rng(42).integers(0, 256, (80, 80, 3), dtype=np.uint8)
    other = np.random.default_rng(43).integers(0, 256, (80, 80, 3), dtype=np.uint8)
    for name, pixels in (("f2.png", image), ("f10.png", other), ("f1.png", image)):
        assert cv2.imwrite(str(tmp_path / name), pixels)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result_file = tmp_path / "result.json"

    assert run_cli(["--suggest-duplicates", "-i", str(tmp_path), "--json", str(result_file)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert json.loads(result_file.read_text(encoding="utf-8")) == payload
    assert [os.path.basename(path) for path in payload["files"]] == ["f1.png", "f2.png", "f10.png"]
    assert payload["mode"] == "suggest-duplicates"
    assert [(item["index"], item["reference_index"]) for item in payload["suggestions"]] == [(1, 0)]
    assert all((tmp_path / name).read_bytes() == contents for name, contents in before.items())


def test_suggest_duplicates_keeps_explicit_file_order(tmp_path, capsys, monkeypatch):
    from core import stack_review
    paths = [tmp_path / "b.png", tmp_path / "a.png"]
    for index, path in enumerate(paths):
        assert cv2.imwrite(str(path), np.full((16, 16, 3), index * 50, np.uint8))

    def analyze(frames, cancel=None):
        assert [int(frame[0, 0, 0]) for frame in frames] == [0, 50]
        return [{"index": 1, "reference_index": 0, "difference": 0.01}]

    monkeypatch.setattr(stack_review, "analyze_frames", analyze)
    assert run_cli(["--suggest-duplicates", "-i", *map(str, paths)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files"] == list(map(str, paths))
    assert payload["suggestions"][0]["difference"] == 0.01


def test_suggest_duplicates_rejects_json_hardlink_to_input(tmp_path, capsys):
    folder, _, _ = _make_stacks(tmp_path)
    source = os.path.join(folder, "f0.png")
    alias = tmp_path / "alias.json"
    try:
        os.link(source, alias)
    except OSError:
        pytest.skip("hard links unavailable")
    before = alias.read_bytes()
    assert run_cli(["--suggest-duplicates", "-i", folder, "--json", str(alias)]) == 1
    captured = capsys.readouterr()
    assert "overwrite" in captured.err and not captured.out
    assert alias.read_bytes() == before


def test_suggest_duplicates_rejects_unreadable_files_and_output_options(tmp_path, capsys):
    folder, _, _ = _make_stacks(tmp_path)
    (tmp_path / "stackA" / "f1.png").write_bytes(b"not an image")
    assert run_cli(["--suggest-duplicates", "-i", folder]) == 1
    captured = capsys.readouterr()
    assert "decode" in captured.err and not captured.out
    output = tmp_path / "output.png"
    assert run_cli(["--suggest-duplicates", "-i", folder, "-o", str(output)]) == 2
    assert not output.exists()


def test_suggest_duplicates_rejects_files_changed_during_analysis(tmp_path, capsys, monkeypatch):
    from core import stack_review
    folder, _, _ = _make_stacks(tmp_path)
    result_file = tmp_path / "result.json"

    def analyze(frames, cancel=None):
        list(frames)
        with open(os.path.join(folder, "f1.png"), "ab") as source:
            source.write(b"changed")
        return []

    monkeypatch.setattr(stack_review, "analyze_frames", analyze)
    assert run_cli(["--suggest-duplicates", "-i", folder, "--json", str(result_file)]) == 1
    captured = capsys.readouterr()
    assert "changed during review" in captured.err and not captured.out
    assert not result_file.exists()


def test_suggest_duplicates_reports_interruption(tmp_path, capsys, monkeypatch):
    from core import stack_review
    folder, _, _ = _make_stacks(tmp_path)

    def analyze(frames, cancel=None):
        raise InterruptedError("review cancelled")

    monkeypatch.setattr(stack_review, "analyze_frames", analyze)
    assert run_cli(["--suggest-duplicates", "-i", folder]) == 1
    captured = capsys.readouterr()
    assert "review cancelled" in captured.err and not captured.out


def test_suggest_duplicates_rejects_multipage_input(tmp_path, capsys):
    from PIL import Image
    pages = [Image.fromarray(np.full((16, 16, 3), value, np.uint8)) for value in (20, 40)]
    stack = tmp_path / "stack.tiff"
    pages[0].save(stack, save_all=True, append_images=pages[1:])
    single = tmp_path / "single.png"
    pages[0].save(single)
    assert run_cli(["--suggest-duplicates", "-i", str(stack), str(single)]) == 1
    captured = capsys.readouterr()
    assert "single-frame" in captured.err and not captured.out


def test_suggest_duplicates_handles_keyboard_interrupt(tmp_path, capsys, monkeypatch):
    from core import stack_review
    folder, _, _ = _make_stacks(tmp_path)

    def analyze(frames, cancel=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(stack_review, "analyze_frames", analyze)
    assert run_cli(["--suggest-duplicates", "-i", folder]) == 1
    captured = capsys.readouterr()
    assert "interrupted" in captured.err and not captured.out


@pytest.mark.parametrize('enabled', [False, True])
def test_registration_receives_cleaned_native_depth_pixels(tmp_path, monkeypatch, enabled):
    from core import cli, registration
    mask = np.zeros((16, 16), np.uint8)
    mask[8, 8] = 255
    mask_path = tmp_path / 'mask.png'
    assert cv2.imwrite(str(mask_path), mask)
    image = np.full((16, 16, 3), 1000, np.uint16)
    image[8, 8] = 50000
    argv = ['-i', 'unused', '-o', 'unused.png', '-m', 'dct', '--align', 'ecc']
    if enabled:
        argv += ['--defect-mask-path', str(mask_path)]
    args = cli.build_parser().parse_args(argv)
    registered = []

    class Registration:
        def __init__(self, **kwargs):
            pass

        def process(self, images, **kwargs):
            registered.append(np.array(images[0], copy=True))
            return images


    monkeypatch.setattr(registration, 'ImageRegistration', Registration)
    fused = cli._fuse_stack([image, image.copy()], args)
    expected = image.copy()
    if enabled:
        expected[8, 8] = 1000
    assert registered[0].dtype == np.uint16
    np.testing.assert_array_equal(registered[0], expected)
    assert fused.dtype == np.uint16
    assert fused[8, 8, 0] == (1000 if enabled else 50000)
    assert image[8, 8, 0] == 50000


def test_grouped_sensor_cleanup_only_processes_original_frames(tmp_path, monkeypatch):
    from core import cli
    from utils import sensor_cleanup
    folder = tmp_path / 'stack'
    folder.mkdir()
    paths = []
    for index in range(4):
        path = folder / f'{index}.png'
        assert cv2.imwrite(str(path), np.full((16, 16, 3), index + 30, np.uint8))
        paths.append(str(path))
    mask_path = tmp_path / 'mask.png'
    assert cv2.imwrite(str(mask_path), np.zeros((16, 16), np.uint8))
    args = cli.build_parser().parse_args(['-i', str(folder), '-o', str(tmp_path / 'out.png'),
                                         '-m', 'dct', '--group-size', '2',
                                         '--defect-mask-path', str(mask_path)])
    calls = []
    cleanup = sensor_cleanup.cleanup_sensor_image
    def record(image, *args, **kwargs):
        calls.append(int(image[0, 0, 0]))
        return cleanup(image, *args, **kwargs)
    monkeypatch.setattr(sensor_cleanup, 'cleanup_sensor_image', record)
    monkeypatch.setattr(cli, '_fuse_stack_admitted', lambda images, args: images[0])
    cli._fuse_grouped(paths, args)
    assert calls == [30, 31, 32, 33]


@pytest.mark.parametrize('grouped', [False, True])
def test_sensor_cleanup_admission_precedes_mask_and_frame_processing(tmp_path, monkeypatch, grouped):
    from core import cli, grouped_fusion, resource_policy
    from utils import sensor_cleanup

    mask_path = tmp_path / 'mask.png'
    assert cv2.imwrite(str(mask_path), np.zeros((16, 16), np.uint8))
    args = cli.build_parser().parse_args(['-i', 'unused', '-o', 'unused.png',
                                         '-m', 'dct', '--defect-mask-path', str(mask_path)])
    image = np.zeros((16, 16, 3), np.uint8)
    requests = []

    def reject(amount):
        requests.append(amount)
        raise resource_policy.ResourceLimitError('full')

    monkeypatch.setattr(resource_policy, 'require_memory', reject)
    monkeypatch.setattr(sensor_cleanup, 'load_sensor_defect_mask',
                        lambda *a, **k: pytest.fail('mask loaded before admission'))
    monkeypatch.setattr(sensor_cleanup, 'cleanup_sensor_image',
                        lambda *a, **k: pytest.fail('frame cleaned before admission'))
    if grouped:
        monkeypatch.setattr(cli, '_inspect_grouped_cli', lambda *a: None)
        monkeypatch.setattr(grouped_fusion, 'grouped_fuse',
                            lambda *a, **k: k['preprocess'](image))
        action = lambda: cli._fuse_grouped(['unused'], args)
    else:
        action = lambda: cli._fuse_stack_admitted([image, image], args)
    with pytest.raises(resource_policy.ResourceLimitError):
        action()
    assert requests and requests[0] >= image.nbytes + mask_path.stat().st_size + 16 * 16 * 16


@pytest.mark.parametrize('batch', [False, True])
def test_sensor_mask_cannot_be_output(tmp_path, batch):
    folder = tmp_path / 'stack'
    folder.mkdir()
    for index in range(2):
        assert cv2.imwrite(str(folder / f'{index}.png'), np.zeros((16, 16, 3), np.uint8))
    mask = tmp_path / ('stack.png' if batch else 'mask.png')
    assert cv2.imwrite(str(mask), np.zeros((16, 16), np.uint8))
    original = mask.read_bytes()
    destination = ['-d', str(tmp_path)] if batch else ['-o', str(mask)]
    assert run_cli(['-i', str(folder), *destination, '-m', 'dct',
                    '--defect-mask-path', str(mask)]) != 0
    assert mask.read_bytes() == original


@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('grouped', [False, True])
@pytest.mark.parametrize('dry_run', [False, True])
def test_cli_rejects_rotated_square_source_with_sensor_mask(tmp_path, capsys, batch, grouped, dry_run):
    from PIL import Image

    folder = tmp_path / 'stack'
    folder.mkdir()
    for index in range(2):
        image = Image.fromarray(np.full((24, 24, 3), 40 + index, np.uint8))
        exif = image.getexif()
        exif[274] = 6 if index else 1
        image.save(folder / f'{index}.jpg', exif=exif)
    mask = tmp_path / 'mask.png'
    assert cv2.imwrite(str(mask), np.zeros((24, 24), np.uint8))
    destination = ['-d', str(tmp_path / 'out')] if batch else ['-o', str(tmp_path / 'out.png')]
    options = (['--group-size', '2'] if grouped else []) + (['--dry-run'] if dry_run else [])
    assert run_cli(['-i', str(folder), *destination, *options,
                    '--defect-mask-path', str(mask)]) != 0
    assert 'orientation' in capsys.readouterr().err.lower()
    assert not (tmp_path / 'out.png').exists()
    assert not (tmp_path / 'out' / 'stack.png').exists()


@pytest.mark.parametrize('option', [[], ['--sensor-denoise']])
def test_cli_oriented_sources_remain_available_without_mask(tmp_path, option):
    from PIL import Image

    folder = tmp_path / 'stack'
    folder.mkdir()
    for index in range(2):
        image = Image.fromarray(np.zeros((24, 24, 3), np.uint8))
        exif = image.getexif()
        exif[274] = 6
        image.save(folder / f'{index}.jpg', exif=exif)
    assert run_cli(['-i', str(folder), '-o', str(tmp_path / 'out.png'),
                    '--dry-run', *option]) == 0


def test_cli_rejects_video_before_materializing_masked_frames(tmp_path, capsys):
    video = tmp_path / 'source.mp4'
    video.write_bytes(b'')
    mask = tmp_path / 'mask.png'
    assert cv2.imwrite(str(mask), np.zeros((16, 16), np.uint8))
    assert run_cli(['-i', str(video), '-o', str(tmp_path / 'out.png'),
                    '--defect-mask-path', str(mask)]) != 0
    assert 'video' in capsys.readouterr().err.lower()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("summary_is_parent", [False, True])
def test_cli_rejects_output_file_used_as_a_directory(tmp_path, batch, summary_is_parent):
    folder, _, _ = _make_stacks(tmp_path)
    output = tmp_path / "result.png"
    summary = output / "summary.json"
    if batch:
        output = tmp_path / "out" / "stackA.png"
        summary = output / "summary.json"
    if summary_is_parent:
        summary = tmp_path / "summary.json"
        output = summary / "stackA.png" if batch else summary / "result.png"
    destination = ["-d", str(output.parent)] if batch else ["-o", str(output)]
    assert run_cli(["-i", folder, *destination, "-m", "dct", "--json", str(summary)]) == 2
    assert not output.exists()
    assert not summary.exists()
    if batch or summary_is_parent:
        assert not output.parent.exists()
