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
