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


def test_cli_imagej_without_units_requires_explicit_scale_for_bar(tmp_path, capsys):
    import tifffile

    folder = tmp_path / 'imagej'
    folder.mkdir()
    for index in range(2):
        tifffile.imwrite(folder / f'{index}.tif',
                         np.full((240, 400), 120, np.uint8), imagej=True)
    plain = tmp_path / 'plain.png'
    annotated = tmp_path / 'annotated.png'
    missing = tmp_path / 'missing.png'
    base = ['-i', str(folder), '--method', 'guided_filter', '--align', 'none', '--cpu']

    assert _run_cli([*base, '-o', str(plain)]) == 0
    assert _run_cli([*base, '-o', str(missing), '--scale-bar']) != 0
    assert 'no pixel size available' in capsys.readouterr().err
    assert not missing.exists()
    assert _run_cli([*base, '-o', str(annotated), '--scale-um', '0.5']) == 0
    assert not np.array_equal(cv2.imread(str(plain)), cv2.imread(str(annotated)))


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


@pytest.mark.parametrize('destination', ['file', 'stderr', 'off'])
def test_interrupted_render_finalizes_profile(tmp_path, capsys, monkeypatch, destination):
    from core import cli
    folder, _, _ = _make_stacks(tmp_path)
    profile_path = tmp_path / 'profile.json'
    events = []

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, '_load_images', interrupt)
    monkeypatch.setattr(cli, 'log_action', lambda name, **fields: events.append((name, fields)))
    argv = ['-i', folder, '-o', str(tmp_path / 'out.png'), '-m', 'dct']
    if destination != 'off':
        argv.append('--profile-performance')
        if destination == 'file':
            argv.append(str(profile_path))
    assert run_cli(argv) == cli.EXIT_ERROR
    stderr = capsys.readouterr().err
    assert 'interrupted' in stderr
    reports = [fields for name, fields in events if name == 'render.performance']
    if destination == 'off':
        assert not reports
        assert 'Outcome:' not in stderr
    else:
        assert len(reports) == 1
        assert reports[0]['outcome'] == 'cancelled'
        assert reports[0]['stages']['load']['count'] == 1
        assert 'Outcome: cancelled' in stderr
    if destination == 'file':
        report = json.loads(profile_path.read_text(encoding='utf-8'))
        assert report['outcome'] == 'cancelled'
        assert report['stages']['load']['count'] == 1
    else:
        assert not profile_path.exists()


def test_profile_does_not_finalize_before_render_starts(tmp_path, capsys, monkeypatch):
    from core import cli
    events = []
    monkeypatch.setattr(cli, 'log_action', lambda name, **fields: events.append(name))
    profile_path = tmp_path / 'profile.json'
    assert run_cli(['-i', str(tmp_path), '-o', str(tmp_path / 'out.png'),
                    '--dry-run', '--profile-performance', str(profile_path)]) == cli.EXIT_USAGE
    assert 'render.performance' not in events
    assert 'Outcome:' not in capsys.readouterr().err
    assert not profile_path.exists()


def test_interruption_before_render_does_not_finalize_profile(tmp_path, capsys, monkeypatch):
    from core import cli
    folder, _, _ = _make_stacks(tmp_path)
    profile_path = tmp_path / 'profile.json'
    events = []

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, '_resolve_stack', interrupt)
    monkeypatch.setattr(cli, 'log_action', lambda name, **fields: events.append(name))
    assert run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'), '-m', 'dct',
                    '--profile-performance', str(profile_path)]) == cli.EXIT_ERROR
    stderr = capsys.readouterr().err
    assert 'interrupted' in stderr
    assert 'Outcome:' not in stderr
    assert 'render.performance' not in events
    assert not profile_path.exists()


def test_profile_finalizes_failed_render(tmp_path, capsys, monkeypatch):
    from core import cli
    folder, _, _ = _make_stacks(tmp_path)
    profile_path = tmp_path / 'profile.json'

    def fail(*args, **kwargs):
        raise RuntimeError('load failed')

    monkeypatch.setattr(cli, '_load_images', fail)
    assert run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'), '-m', 'dct',
                    '--profile-performance', str(profile_path)]) == cli.EXIT_ERROR
    assert json.loads(profile_path.read_text(encoding='utf-8'))['outcome'] == 'failed'
    assert 'Outcome: failed' in capsys.readouterr().err


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


@pytest.mark.parametrize('options', [[], ['--dry-run'], ['--group-size', '2']])
def test_summary_missing_input_never_writes_source(tmp_path, monkeypatch, options):
    from core import cli

    source = tmp_path / 'source.png'
    source.write_bytes(b'original image')
    writes = []
    monkeypatch.setattr(cli, '_replace_text_file', lambda *a, **k: writes.append(a))
    rc = run_cli(['-i', str(source), str(tmp_path / 'missing.png'),
                  '-o', str(tmp_path / 'out.png'), '-m', 'dct',
                  '--json', str(source), *options])
    assert rc != 0
    assert source.read_bytes() == b'original image'
    assert not writes


@pytest.mark.parametrize('kind', ['one_frame', 'count_error', 'video_error'])
def test_summary_admission_precedes_stack_decoding(tmp_path, monkeypatch, kind):
    from core import image_loader

    folder = tmp_path / 'stack'
    folder.mkdir()
    frame = folder / 'frame.png'
    frame.write_bytes(b'frame contents')
    source = folder
    if kind == 'count_error':
        def fail_count(path):
            raise OSError('count failed')
        monkeypatch.setattr(image_loader, 'count_frames', fail_count)
    elif kind == 'video_error':
        source = tmp_path / 'source.mp4'
        source.write_bytes(b'video contents')
        monkeypatch.setattr(image_loader.ImageStackLoader, 'load_from_video',
                            lambda self, path: (False, 'decode failed', [], []))
    summary = tmp_path / 'summary.json'
    assert run_cli(['-i', str(source), '-o', str(tmp_path / 'out.png'),
                    '-m', 'dct', '--json', str(summary)]) == 1
    assert json.loads(summary.read_text(encoding='utf-8'))['mode'] == 'single'
    assert frame.read_bytes() == b'frame contents'
    if kind == 'video_error':
        assert source.read_bytes() == b'video contents'
        assert run_cli(['-i', str(source), '-o', str(tmp_path / 'out.png'),
                        '-m', 'dct', '--json', str(source)]) == 2
        assert source.read_bytes() == b'video contents'


@pytest.mark.parametrize('alias', ['same', 'hardlink', 'symlink', 'symlink_parent'])
def test_summary_refuses_frame_aliases_before_writing(tmp_path, alias):
    folder = tmp_path / 'stack'
    folder.mkdir()
    frame = folder / 'frame.png'
    frame.write_bytes(b'original')
    summary = frame
    try:
        if alias == 'hardlink':
            summary = tmp_path / 'alias.json'
            os.link(frame, summary)
        elif alias == 'symlink':
            summary = tmp_path / 'alias.json'
            summary.symlink_to(frame)
        elif alias == 'symlink_parent':
            link = tmp_path / 'linked'
            link.symlink_to(folder, target_is_directory=True)
            summary = link / frame.name
    except OSError as exc:
        pytest.skip(f'links unavailable: {exc}')
    assert run_cli(['-i', str(folder), '-o', str(tmp_path / 'out.png'),
                    '-m', 'dct', '--json', str(summary)]) == 2
    assert frame.read_bytes() == b'original'
    assert not (tmp_path / 'out.png').exists()


@pytest.mark.parametrize('target', ['dust', 'defect', 'output', 'depth', 'profile'])
def test_summary_refuses_references_and_outputs(tmp_path, target):
    folder, _, _ = _make_stacks(tmp_path)
    output = tmp_path / 'out.png'
    reference = tmp_path / 'reference.png'
    reference.write_bytes(b'reference')
    options = []
    if target == 'dust':
        summary, options = reference, ['--dust-map', str(reference)]
    elif target == 'defect':
        summary, options = reference, ['--defect-mask-path', str(reference)]
    elif target == 'output':
        summary = output
    elif target == 'depth':
        summary, options = tmp_path / 'depth.png', ['--depth-map', str(tmp_path / 'depth.png')]
    else:
        summary, options = tmp_path / 'profile.json', ['--profile-performance', str(tmp_path / 'profile.json')]
    if target in ('output', 'depth', 'profile'):
        summary.write_bytes(b'previous result')
    before = summary.read_bytes()
    assert run_cli(['-i', folder, '-o', str(output), '-m', 'dct',
                    '--json', str(summary), *options]) == 2
    assert summary.read_bytes() == before
    assert reference.read_bytes() == b'reference'


def test_batch_summary_refuses_config_and_unavailable_inputs(tmp_path, monkeypatch):
    from core import cli

    folder, _, out = _make_stacks(tmp_path)
    config = tmp_path / 'config.json'
    config.write_text('{}', encoding='utf-8')
    assert run_cli(['-i', folder, '-d', str(out), '--config', str(config),
                    '--json', str(config)]) == 2
    assert config.read_text(encoding='utf-8') == '{}'
    summary = tmp_path / 'summary-dir' / 'summary.json'
    def unavailable(*args):
        raise PermissionError('cannot enumerate')
    monkeypatch.setattr(cli, '_batch_inputs', unavailable)
    assert run_cli(['-i', folder, '-d', str(out), '--json', str(summary)]) == 2
    assert not out.exists()
    assert not summary.parent.exists()


@pytest.mark.parametrize('after_admission', [False, True])
def test_batch_outer_failure_only_writes_after_admission(tmp_path, monkeypatch, after_admission):
    from core import cli

    folder, _, out = _make_stacks(tmp_path)
    summary = tmp_path / 'summary.json'
    if after_admission:
        original = cli.os.makedirs
        def fail_output(path, *args, **kwargs):
            if os.fspath(path) == os.fspath(out):
                raise OSError('output directory failed')
            return original(path, *args, **kwargs)
        monkeypatch.setattr(cli.os, 'makedirs', fail_output)
    else:
        def fail_plan(*args):
            raise OSError('planning failed')
        monkeypatch.setattr(cli, '_result_outputs', fail_plan)
    assert run_cli(['-i', folder, '-d', str(out), '-m', 'dct', '--json', str(summary)]) == 1
    if after_admission:
        payload = json.loads(summary.read_text(encoding='utf-8'))
        assert payload['mode'] == 'batch' and 'output directory failed' in payload['error']
    else:
        assert not summary.exists()
    assert not out.exists()


@pytest.mark.parametrize('failure', ['replace', 'interrupt', 'changed_source'])
def test_summary_atomic_failure_preserves_previous_file(tmp_path, monkeypatch, failure):
    from core import cli

    folder, _, _ = _make_stacks(tmp_path)
    source = os.path.join(folder, 'f0.png')
    before = open(source, 'rb').read()
    summary = tmp_path / 'summary.json'
    summary.write_bytes(b'previous summary')
    if failure in ('replace', 'interrupt'):
        def fail_replace(*args):
            if failure == 'interrupt':
                raise KeyboardInterrupt()
            raise OSError('replace failed')
        monkeypatch.setattr(cli.os, 'replace', fail_replace)
    else:
        original = cli._admitted_input_keys
        calls = 0
        def unavailable(candidates):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError('source became inaccessible')
            return original(candidates)
        monkeypatch.setattr(cli, '_admitted_input_keys', unavailable)
    rc = run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'), '-m', 'dct',
                  '--dry-run', '--json', str(summary)])
    assert rc == (1 if failure == 'interrupt' else 0)
    assert summary.read_bytes() == b'previous summary'
    assert not list(tmp_path.glob('.openfocus_summary_*.tmp'))
    assert open(source, 'rb').read() == before


def test_summary_scanning_interrupt_never_writes(tmp_path, monkeypatch):
    from core import cli

    folder, _, _ = _make_stacks(tmp_path)
    summary = tmp_path / 'new' / 'summary.json'
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(cli, '_list_folder_images', interrupt)
    assert run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'), '-m', 'dct',
                    '--json', str(summary)]) == 1
    assert not summary.parent.exists()


def test_summary_source_stat_failure_does_not_create_directory(tmp_path, monkeypatch):
    from core import cli

    folder, _, _ = _make_stacks(tmp_path)
    frame = os.path.join(folder, 'f0.png')
    summary = tmp_path / 'new' / 'summary.json'
    original = cli.os.stat
    def unavailable(path, *args, **kwargs):
        if os.fspath(path) == frame:
            raise PermissionError('source inaccessible')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(cli.os, 'stat', unavailable)
    assert run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'), '-m', 'dct',
                    '--json', str(summary)]) == 1
    assert not summary.parent.exists()


@pytest.mark.parametrize('valid', [False, True])
def test_native_summary_preflight_and_processing_failure(tmp_path, monkeypatch, valid):
    from core import cli

    source = tmp_path / 'source.dng'
    source.write_bytes(b'native source')
    summary = tmp_path / 'summary.json'
    def validate(paths):
        if not valid:
            raise ValueError('invalid native sources')
    monkeypatch.setattr(cli, '_validate_native_paths', validate)
    def fail_stack(*args):
        raise RuntimeError('native processing failed')
    monkeypatch.setattr(cli, '_resolve_stack', fail_stack)
    result = run_cli(['-i', str(source), '-o', str(tmp_path / 'out.dng'),
                      '-m', 'dct', '--native-raw', '--json', str(summary)])
    assert result == (1 if valid else 2)
    assert source.read_bytes() == b'native source'
    if valid:
        assert 'native processing failed' in json.loads(summary.read_text(encoding='utf-8'))['error']
    else:
        assert not summary.exists()


def test_batch_processing_failure_writes_batch_summary(tmp_path, monkeypatch):
    from core import cli

    folder, _, out = _make_stacks(tmp_path)
    summary = tmp_path / 'summary.json'
    def fail_load(*args, **kwargs):
        raise RuntimeError('load failed')
    monkeypatch.setattr(cli, '_load_images', fail_load)
    assert run_cli(['-i', folder, '-d', str(out), '-m', 'dct', '--json', str(summary)]) == 1
    payload = json.loads(summary.read_text(encoding='utf-8'))
    assert payload['mode'] == 'batch' and payload['failed'] == ['stackA']


@pytest.mark.parametrize('dry_run', [False, True])
def test_single_safe_summary_for_success(tmp_path, monkeypatch, dry_run):
    from core import cli

    folder, _, _ = _make_stacks(tmp_path)
    summary = tmp_path / 'summary.json'
    output = tmp_path / 'out.png'
    if not dry_run:
        monkeypatch.setattr(cli, '_load_images', lambda *a, **k: [np.zeros((24, 24, 3), np.uint8)] * 3)
        monkeypatch.setattr(cli, '_fuse_stack', lambda images, args: images[0])
    options = ['--dry-run'] if dry_run else []
    assert run_cli(['-i', folder, '-o', str(output), '-m', 'dct',
                    '--json', str(summary), *options]) == 0
    payload = json.loads(summary.read_text(encoding='utf-8'))
    assert payload['mode'] == 'single'
    assert payload['output'] == str(output)
    assert output.exists() != dry_run


def test_batch_summary_conflict_does_not_create_outputs(tmp_path):
    folder, _, out = _make_stacks(tmp_path)
    summary = out / 'stackA.png'
    assert run_cli(['-i', folder, '-d', str(out), '-m', 'dct',
                    '--json', str(summary)]) == 2
    assert not out.exists()


def test_summary_refuses_sidecar_alias(tmp_path):
    folder, _, _ = _make_stacks(tmp_path)
    sidecar = tmp_path / 'stacked.png'
    sidecar.write_bytes(b'existing sidecar')
    assert run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'), '-m', 'dct',
                    '--auto-background', 'auto', '--save-stacked', str(sidecar),
                    '--json', str(sidecar)]) == 2
    assert sidecar.read_bytes() == b'existing sidecar'


@pytest.mark.parametrize('counts,still', [((2, 2), False), ((3, 2), False), ((2, 2), True)])
def test_video_frames_remain_unique_and_ordered(tmp_path, monkeypatch, counts, still):
    from core import cli, image_loader

    videos = [tmp_path / 'a.mp4', tmp_path / 'b.mp4']
    for video in videos:
        video.write_bytes(b'video')
    frames = {str(video): [np.full((8, 8, 3), 20 * (index + 1) + i, np.uint8)
                           for i in range(count)]
              for index, (video, count) in enumerate(zip(videos, counts))}
    monkeypatch.setattr(image_loader.ImageStackLoader, 'load_from_video',
                        lambda self, path: (True, '', frames[path], []))
    inputs = [str(path) for path in videos]
    expected = [frame for video in videos for frame in frames[str(video)]]
    if still:
        image = tmp_path / 'still.png'
        pixels = np.full((8, 8, 3), 7, np.uint8)
        assert cv2.imwrite(str(image), pixels)
        inputs.insert(0, str(image))
        expected.insert(0, pixels)
    args = cli.build_parser().parse_args(['-i', *inputs, '-o', str(tmp_path / 'out.png')])
    scratch = tmp_path / 'frames'
    scratch.mkdir()
    _, paths = cli._resolve_stack(args, str(scratch))
    assert len(paths) == len(expected) == len(set(paths))
    for path, pixels in zip(paths, expected):
        np.testing.assert_array_equal(cv2.imread(path), pixels)


@pytest.mark.parametrize('failure', ['false', 'oserror', 'interrupt'])
def test_video_write_failure_cleans_temporary_frames(tmp_path, monkeypatch, failure):
    from core import cli, image_loader
    from utils import image_utils

    videos = [tmp_path / 'a.mp4', tmp_path / 'b.mp4']
    for video in videos:
        video.write_bytes(b'original video')
    monkeypatch.setattr(image_loader.ImageStackLoader, 'load_from_video',
                        lambda self, path: (True, '', [np.zeros((8, 8, 3), np.uint8)] * 2, []))
    original = image_utils.imwrite_auto
    written = []
    def write_frame(path, frame, *args, **kwargs):
        written.append(path)
        if len(written) == 3:
            if failure == 'oserror':
                raise OSError('frame write failed')
            if failure == 'interrupt':
                raise KeyboardInterrupt()
            return False
        return original(path, frame, *args, **kwargs)
    monkeypatch.setattr(image_utils, 'imwrite_auto', write_frame)
    monkeypatch.setattr(cli, '_fuse_stack', lambda *args: pytest.fail('unexpected fusion'))
    output = tmp_path / 'out.png'
    summary = tmp_path / 'summary.json'
    assert run_cli(['-i', *map(str, videos), '-o', str(output), '-m', 'dct',
                    '--json', str(summary)]) == 1
    assert all(video.read_bytes() == b'original video' for video in videos)
    assert not output.exists() and written and not os.path.exists(os.path.dirname(written[0]))
    assert summary.exists() == (failure != 'interrupt')


@pytest.mark.parametrize('override', [
    {'threads': -1}, {'threads': True}, {'batch_size': 0}, {'tile_size': 63},
    {'resource_preference': 'unknown'}, {'cpu': 'false'}, {'base_frame': 'typo'},
    {'align_shift_limit': True}, {'fusion_parameters': []},
])
def test_config_rejects_invalid_overrides_before_processing(tmp_path, monkeypatch, override):
    from core import cli

    folder, _, out = _make_stacks(tmp_path)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'unmatched': override}), encoding='utf-8')
    monkeypatch.setattr(cli, '_load_images', lambda *a, **k: pytest.fail('unexpected load'))
    assert run_cli(['-i', folder, '-d', str(out), '-m', 'dct', '--config', str(config)]) == 2
    assert not out.exists()


@pytest.mark.parametrize('config_text', ['{"stackA":{"threads":1,"threads":2}}',
                                         '{"stackA":{"threads":NaN}}',
                                         '{"stackA":{"threads":Infinity}}'])
def test_config_rejects_duplicate_and_nonfinite_json(tmp_path, config_text):
    folder, _, out = _make_stacks(tmp_path)
    config = tmp_path / 'config.json'
    config.write_text(config_text, encoding='utf-8')
    assert run_cli(['-i', folder, '-d', str(out), '-m', 'dct', '--config', str(config)]) == 2
    assert not out.exists()


@pytest.mark.parametrize('options', [['--group-size', '2'], ['--native-raw']])
def test_config_rejects_incompatible_background(tmp_path, options):
    folder, _, out = _make_stacks(tmp_path)
    config = tmp_path / 'config.json'
    config.write_text('{"stackA":{"auto_background":"auto"}}', encoding='utf-8')
    assert run_cli(['-i', folder, '-d', str(out), '-m', 'dct', '--config', str(config), *options]) == 2
    assert not out.exists()


def test_config_cannot_disable_background_with_inherited_sidecar(tmp_path):
    folder, _, out = _make_stacks(tmp_path)
    config = tmp_path / 'config.json'
    config.write_text('{"stackA":{"auto_background":null}}', encoding='utf-8')
    assert run_cli(['-i', folder, '-d', str(out), '-m', 'dct', '--dry-run',
                    '--auto-background', '--save-stacked', '--config', str(config)]) == 2
    assert not out.exists()


def test_batch_override_resources_reach_loading_and_fusion(tmp_path, monkeypatch):
    from core import cli

    first, second, out = _make_stacks(tmp_path)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'stackA': {'threads': 1, 'resource_preference': 'responsive'},
                                  'stackB': {'threads': 3, 'resource_preference': 'maximum'}}), encoding='utf-8')
    loads, fusions = [], []
    def load(paths, **kwargs):
        loads.append((kwargs['thread_count'], kwargs['resource_preference']))
        return [np.zeros((24, 24, 3), np.uint8)] * 3
    def fuse(images, args):
        fusions.append((args.threads, args.resource_preference))
        return images[0]
    monkeypatch.setattr(cli, '_load_images', load)
    monkeypatch.setattr(cli, '_fuse_stack', fuse)
    assert run_cli(['-i', first, second, '-d', str(out), '-m', 'dct', '--config', str(config)]) == 0
    assert loads == fusions == [(1, 'responsive'), (3, 'maximum')]


@pytest.mark.parametrize('state', ['missing', 'empty', 'directory', 'complete'])
def test_resume_checks_every_requested_depth_output(tmp_path, monkeypatch, state):
    from core import cli

    folder, _, out = _make_stacks(tmp_path)
    out.mkdir()
    result = out / 'stackA.png'
    result.write_bytes(b'result')
    depth = out / 'stackA_depth.png'
    if state == 'empty':
        depth.write_bytes(b'')
    elif state == 'directory':
        depth.mkdir()
    elif state == 'complete':
        depth.write_bytes(b'depth')
    mtime = result.stat().st_mtime_ns
    scans = []
    original = cli._folder_stack_paths
    def scan(*args):
        scans.append(args[0])
        return original(*args)
    monkeypatch.setattr(cli, '_folder_stack_paths', scan)
    summary = tmp_path / 'summary.json'
    assert run_cli(['-i', folder, '-d', str(out), '-m', 'dct', '--depth-map',
                    '--resume', '--dry-run', '--json', str(summary)]) == 0
    payload = json.loads(summary.read_text(encoding='utf-8'))
    assert bool(scans) == (state != 'complete')
    assert payload['skipped_existing'] == (['stackA'] if state == 'complete' else [])
    assert result.stat().st_mtime_ns == mtime


def test_resume_checks_all_background_masks(tmp_path, monkeypatch):
    from core import cli

    folder, _, out = _make_stacks(tmp_path)
    out.mkdir()
    parser = cli.build_parser()
    args = parser.parse_args(['-i', folder, '-d', str(out), '-m', 'dct',
                              '--auto-background', '--missing-data-map'])
    results = cli._result_outputs(args, str(out / 'stackA.png'))
    assert len(results) > 2
    for _, path in results[:-1]:
        with open(path, 'wb') as stream:
            stream.write(b'existing')
    scans = []
    original = cli._folder_stack_paths
    def scan(*args):
        scans.append(args[0])
        return original(*args)
    monkeypatch.setattr(cli, '_folder_stack_paths', scan)
    assert run_cli(['-i', folder, '-d', str(out), '-m', 'dct', '--auto-background',
                    '--missing-data-map', '--resume', '--dry-run']) == 0
    assert scans


def test_resume_stat_error_fails_folder_and_continues(tmp_path, monkeypatch):
    from core import cli

    first, second, out = _make_stacks(tmp_path)
    out.mkdir()
    first_output = out / 'stackA.png'
    second_output = out / 'stackB.png'
    first_output.write_bytes(b'first')
    second_output.write_bytes(b'second')
    original = cli.os.stat
    def stat_path(path, *args, **kwargs):
        if os.fspath(path) == str(first_output):
            raise PermissionError('inaccessible')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(cli.os, 'stat', stat_path)
    assert run_cli(['-i', first, second, '-d', str(out), '-m', 'dct', '--resume', '--dry-run']) == 1
    assert second_output.read_bytes() == b'second'


@pytest.mark.parametrize('scenario', ['single_dry', 'batch_dry', 'batch_partial',
                                      'resume', 'usage', 'conflict', 'interrupt'])
def test_machine_progress_always_has_one_terminal_event(tmp_path, monkeypatch, capsys, scenario):
    from core import cli

    first, second, out = _make_stacks(tmp_path)
    args = ['-i', first, '-o', str(tmp_path / 'out.png'), '-m', 'dct', '--json-progress']
    expected = 0
    if scenario in ('batch_dry', 'batch_partial', 'resume', 'conflict', 'interrupt'):
        args = ['-i', first, second, '-d', str(out), '-m', 'dct', '--json-progress', '--dry-run']
    if scenario == 'single_dry':
        args.append('--dry-run')
    elif scenario == 'batch_partial':
        for name in os.listdir(second):
            os.unlink(os.path.join(second, name))
        expected = 1
    elif scenario == 'resume':
        out.mkdir()
        (out / 'stackA.png').write_bytes(b'a')
        (out / 'stackB.png').write_bytes(b'b')
        args.append('--resume')
    elif scenario == 'usage':
        args.append('--resume')
        expected = 2
    elif scenario == 'conflict':
        args += ['--json', str(out / 'stackA.png')]
        expected = 2
    elif scenario == 'interrupt':
        original = cli._folder_stack_paths
        def interrupt(folder, *rest):
            if folder == os.path.abspath(second):
                raise KeyboardInterrupt()
            return original(folder, *rest)
        monkeypatch.setattr(cli, '_folder_stack_paths', interrupt)
        expected = 1
    assert run_cli(args) == expected
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sum(event['event'] == 'start' for event in events) == 1
    assert sum(event['event'] == 'done' for event in events) == 1
    done = events[-1]
    assert done['event'] == 'done' and done['ok'] == (expected == 0)
    assert done['exit_code'] == expected
    if scenario == 'interrupt':
        assert done['cancelled'] and done['succeeded'] == 1 and done['failed'] == []
    if scenario == 'resume':
        assert done['succeeded'] == 2 and done['skipped_existing'] == ['stackA', 'stackB']


def test_machine_progress_before_start_usage_has_no_events(tmp_path, capsys):
    folder, _, _ = _make_stacks(tmp_path)
    assert run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'), '-m', 'dct',
                    '--threads', '-1', '--json-progress']) == 2
    assert capsys.readouterr().out == ''


def test_resume_retries_after_depth_write_failure(tmp_path, monkeypatch):
    from core import cli

    folder, _, out = _make_stacks(tmp_path)
    monkeypatch.setattr(cli, '_load_images', lambda *a, **k: [np.zeros((24, 24, 3), np.uint8)] * 3)
    fusions = []
    def fuse(images, args):
        fusions.append(1)
        return images[0]
    monkeypatch.setattr(cli, '_fuse_stack', fuse)
    attempts = []
    def save_depth(args, output):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError('depth write failed')
        with open(cli._depth_out_path(args, output), 'wb') as stream:
            stream.write(b'depth')
    monkeypatch.setattr(cli, '_save_depth', save_depth)
    summary = tmp_path / 'summary.json'
    argv = ['-i', folder, '-d', str(out), '-m', 'dct', '--depth-map', '--resume', '--json', str(summary)]
    assert run_cli(argv) == 1
    assert (out / 'stackA.png').is_file() and not (out / 'stackA_depth.png').exists()
    assert run_cli(argv) == 0
    assert (out / 'stackA_depth.png').read_bytes() == b'depth'
    assert len(fusions) == 2
    assert json.loads(summary.read_text(encoding='utf-8'))['skipped_existing'] == []


def test_broken_progress_pipe_does_not_repeat_processing(tmp_path, monkeypatch):
    import sys

    folder, _, _ = _make_stacks(tmp_path)
    class BrokenStream:
        encoding = 'utf-8'
        def __init__(self):
            self.attempts = 0
        def write(self, text):
            self.attempts += 1
            raise BrokenPipeError()
        def flush(self):
            pass
    stream = BrokenStream()
    monkeypatch.setattr(sys, 'stdout', stream)
    assert run_cli(['-i', folder, '-o', str(tmp_path / 'out.png'), '-m', 'dct',
                    '--dry-run', '--json-progress']) == 0
    assert stream.attempts == 1


@pytest.mark.parametrize('scenario', ['single_cancel', 'batch_scan_failure'])
def test_progress_terminal_on_cancellation_or_scan_failure(tmp_path, monkeypatch, capsys, scenario):
    from core import cli

    folder, _, out = _make_stacks(tmp_path)
    if scenario == 'single_cancel':
        def interrupt(*args):
            raise KeyboardInterrupt()
        monkeypatch.setattr(cli, '_resolve_stack', interrupt)
        args = ['-i', folder, '-o', str(tmp_path / 'out.png'), '-m', 'dct', '--json-progress']
        expected = 1
    else:
        def fail(*args):
            raise PermissionError('scan failed')
        monkeypatch.setattr(cli, '_batch_inputs', fail)
        args = ['-i', folder, '-d', str(out), '-m', 'dct', '--json-progress']
        expected = 2
    assert run_cli(args) == expected
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event['event'] for event in events].count('start') == 1
    assert [event['event'] for event in events].count('done') == 1
    assert events[-1]['event'] == 'done' and events[-1]['exit_code'] == expected
    if scenario == 'single_cancel':
        assert events[-1]['cancelled'] and events[-1]['code'] == 'cancelled'
    else:
        assert events[-1]['total'] == 1 and events[-1]['succeeded'] == 0


def test_cli_registers_video_scratch_and_sweeps_before_work(tmp_path, monkeypatch):
    from core import cli, image_loader, scratch_registry

    video = tmp_path / 'source.mp4'
    video.write_bytes(b'video')
    registered, forgotten, sweeps = [], [], []
    monkeypatch.setattr(scratch_registry, 'register', lambda path: registered.append(path))
    monkeypatch.setattr(scratch_registry, 'forget', lambda paths: forgotten.extend(paths))
    monkeypatch.setattr(scratch_registry, 'sweep_in_background', lambda paths: sweeps.append(paths))
    monkeypatch.setattr(image_loader.ImageStackLoader, 'load_from_video',
                        lambda self, path: (True, '', [np.zeros((8, 8, 3), np.uint8)] * 2, []))
    assert run_cli(['-i', str(video), '-o', str(tmp_path / 'out.png'), '-m', 'dct', '--dry-run']) == 0
    video_directories = [path for path in registered if os.path.basename(path).startswith('openfocus_video_')]
    assert sweeps and len(video_directories) == 1
    frames = [path for path in registered if os.path.basename(path).startswith('video_')]
    assert len(frames) == 2 and all(path in forgotten for path in [*frames, *video_directories])
    assert all(not os.path.exists(path) for path in [*frames, *video_directories])
