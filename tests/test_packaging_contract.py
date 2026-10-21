"""Packaging contract: one install, no capability tiers.

`pip install openfocus` must be a complete install. The AI method is the
default fusion choice, so an install that lacked torch was broken the moment
the window opened, and the `[ai]`/`[heic]`/`[raw]` extras meant a user could
not tell which capabilities they actually had. These tests pin the single
install so it cannot quietly fragment again.
"""
from pathlib import Path

import pytest

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
    names = {d.split("=")[0].split(">")[0].split("<")[0].strip() for d in deps}
    for required in ("torch", "numpy", "opencv-python-headless", "scipy"):
        assert required in names, (
            f"{required} must be installed by default; the default fusion "
            "method is the neural one")


def test_optional_input_decoders_ship_by_default():
    names = {d.split("=")[0].split(">")[0].split("<")[0].strip().lower()
             for d in _project()["dependencies"]}
    assert "pillow-heif" in names, "HEIC input would be unavailable"
    assert "rawpy" in names, "camera RAW input would be unavailable"


def test_no_unused_pytorch_companion_is_required():
    """Nothing imports torchvision; requiring it only inflated the install."""
    names = {d.split("=")[0].split(">")[0].split("<")[0].strip().lower()
             for d in _project()["dependencies"]}
    assert "torchvision" not in names


def test_python_floor_matches_the_ci_matrix():
    floor = _project()["requires-python"]
    assert "3.10" in floor, (
        "CI and both release builds run 3.10; advertising anything lower "
        "installs on interpreters the app cannot import on")


def test_console_script_is_exposed():
    assert "openfocus" in _project()["scripts"]


def test_every_locale_pack_is_packaged():
    """The packs are top-level modules; unlisted ones vanish from the wheel."""
    listed = set(tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
                 ["tool"]["setuptools"]["py-modules"])
    on_disk = {p.stem for p in ROOT.glob("locales_*.py")}
    missing = on_disk - listed
    assert not missing, (
        f"locale modules absent from py-modules: {sorted(missing)}")


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
