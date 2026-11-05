"""Packaging contract: one install, no capability tiers.

`pip install openfocus` supplies the runtime and model files for the default
neural fusion method, along with the supported input decoders. These tests pin
the dependency and package-data contract without requiring a GPU.
"""
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

# tomllib is 3.11+; CI still tests 3.10, so fall back to tomli when present.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on Python 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

pytestmark = pytest.mark.skipif(
    tomllib is None, reason="no TOML parser available (pip install tomli)")


def _project():
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def test_there_are_no_optional_dependency_groups():
    extras = _project().get("optional-dependencies") or {}
    assert not extras, (
        f"extras reappeared ({sorted(extras)}); every capability must be in "
        "the base dependencies so `pip install openfocus` is complete")


def test_the_ai_stack_is_a_base_dependency():
    deps = _project()["dependencies"]
    names = {d.split(";")[0].split("=")[0].split(">")[0].split("<")[0].strip().lower()
             for d in deps}
    for required in ("onnxruntime", "onnxruntime-directml", "numpy", "opencv-python-headless", "scipy"):
        assert required in names, f"{required} must be declared for the supported platforms"
    runtime = [Requirement(d) for d in deps if d.startswith("onnxruntime")]
    for platform, machine, expected in (
        ("win32", "AMD64", "onnxruntime-directml"),
        ("win32", "ARM64", "onnxruntime"),
        ("darwin", "arm64", "onnxruntime"),
        ("linux", "x86_64", "onnxruntime"),
    ):
        environment = {"sys_platform": platform, "platform_machine": machine}
        selected = [req.name for req in runtime if req.marker.evaluate(environment)]
        assert selected == [expected], (environment, selected)


def test_dynamic_onnx_checkpoints_are_packaged():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]
    assert "*.onnx" in data["weights"]
    for name in ("stackmff_final.onnx", "stackmff_v2_final.onnx", "stackmff_v3_final.onnx"):
        assert (ROOT / "weights" / name).is_file()


def test_optional_input_decoders_ship_by_default():
    names = {d.split("=")[0].split(">")[0].split("<")[0].strip().lower()
             for d in _project()["dependencies"]}
    assert "pillow-heif" in names, "HEIC input would be unavailable"
    assert "rawpy" in names, "camera RAW input would be unavailable"


def test_pytorch_is_not_a_production_dependency():
    names = {d.split(";")[0].split("=")[0].split(">")[0].split("<")[0].strip().lower()
             for d in _project()["dependencies"]}
    assert not {"torch", "torchvision"} & names


def test_python_floor_matches_the_ci_matrix():
    floor = _project()["requires-python"]
    assert "3.10" in floor, (
        "CI and both release builds run 3.10; advertising anything lower "
        "installs on interpreters the app cannot import on")


def test_console_script_is_exposed():
    assert "openfocus" in _project()["scripts"]



def test_docs_do_not_advertise_extras():
    """A stale `pip install openfocus[ai]` line sends users to a package
    that has no such extra."""
    stale = []
    for path in [ROOT / "README.md", *ROOT.glob("docs/*.md")]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for extra in ("openfocus[ai]", "openfocus[raw]", "openfocus[heic]"):
            if extra in text:
                stale.append(f"{path.name}: {extra}")
    assert not stale, f"docs still advertise removed extras: {stale}"
