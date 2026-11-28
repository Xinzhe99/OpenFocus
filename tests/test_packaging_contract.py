"""Packaging contract: one install, no capability tiers.

`pip install openfocus` supplies the runtime and model files for the default
neural fusion method, along with the supported input decoders. These tests pin
the dependency and package-data contract without requiring a GPU.
"""
import json
import shutil

from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

# tomllib is 3.11+; keep the tomli fallback for Python 3.10 environments.
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
    import hashlib
    from core.inference_worker import _SHIPPED_TORCH_EXPORTS, _TORCH_EXPORTS

    scikit_build = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["scikit-build"]
    assert "weights" in scikit_build["wheel"]["packages"]
    assert "core" in scikit_build["wheel"]["packages"]
    shipped = {}
    for name in ("stackmff_final.onnx", "stackmff_v2_final.onnx", "stackmff_v3_final.onnx"):
        graph = ROOT / "weights" / name
        checkpoint = graph.with_suffix(".pth")
        shipped[hashlib.sha256(graph.read_bytes()).hexdigest()] = (
            checkpoint.name, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    assert _SHIPPED_TORCH_EXPORTS == shipped
    assert all(_TORCH_EXPORTS[digest] == release for digest, release in shipped.items())
    previous_v3 = '7b92bf31c984eac2b93d6000158b4802b7d62c23140d52e28a47676f58d5c4f7'
    current_v3 = next(release for release in shipped.values() if release[0] == 'stackmff_v3_final.pth')
    assert _TORCH_EXPORTS[previous_v3] == current_v3


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
        "OpenFocus requires Python 3.10+; advertising anything lower "
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


def test_frozen_resource_manifest_covers_runtime_files():
    manifest_path = ROOT / 'packaging/runtime_resources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['version'] == 1
    included = set()
    for pattern in manifest['include']:
        paths = {path.relative_to(ROOT).as_posix() for path in ROOT.glob(pattern)
                 if path.is_file()}
        assert paths, pattern
        included.update(paths)
    expected = {
        'LICENSE', 'packaging/runtime_resources.json', 'docs/THIRD_PARTY_NOTICES.md',
        'assets/OpenFocus.ico', 'assets/OpenFocus.png',
        *(f'assets/demo_stack/demo_{n:02d}.jpg' for n in range(1, 7)),
        *(f'docs/USER_MANUAL_{lang}.md' for lang in
          ('DE', 'EN', 'ES', 'FR', 'IT', 'JA', 'KO', 'NL', 'PL', 'PT', 'RU', 'ZH')),
        *(f'weights/{model}.onnx' for model in
          ('stackmff_final', 'stackmff_v2_final', 'stackmff_v3_final')),
        *(f'weights/{model}.pth' for model in
          ('stackmff_final', 'stackmff_v2_final', 'stackmff_v3_final')),
        'core/inference_worker.py', 'core/inference_tensorrt.py',
        *(f'core/accelerator_{tag}.json' for tag in ('win64', 'linux_x86_64', 'macos_arm64')),
        'core/__init__.py', 'core/models/__init__.py', 'core/models/stackmff_v3/model_io.py',
        *(f'core/models/stackmff_v{version}/network.py' for version in (1, 2, 3)),
    }
    assert expected <= included
    assert {path.relative_to(ROOT).as_posix() for path in (ROOT / 'assets').glob('*.svg')} <= included
    assert {'assets/launch.mp4', 'assets/readme/**', 'assets/Examples/**',
            'docs/ux-specs/**', 'ui/*.py'} <= set(manifest['exclude'])


def test_frozen_resource_verifier_checks_hashes_and_exclusions(tmp_path):
    from tools.verify_frozen_resources import verify_resources

    source, bundle = tmp_path / 'source', tmp_path / 'bundle'
    (source / 'packaging').mkdir(parents=True)
    (source / 'assets').mkdir()
    manifest = {'include': ['packaging/runtime_resources.json', 'assets/icon.svg'],
                'exclude': ['assets/readme/**', 'ui/*.py']}
    (source / 'packaging/runtime_resources.json').write_text(json.dumps(manifest), encoding='utf-8')
    (source / 'assets/icon.svg').write_text('icon', encoding='utf-8')
    shutil.copytree(source, bundle)
    assert verify_resources(bundle, source) == []
    (bundle / 'assets/icon.svg').write_text('modified', encoding='utf-8')
    (bundle / 'ui').mkdir()
    (bundle / 'ui/source.py').write_text('', encoding='utf-8')
    errors = verify_resources(bundle, source)
    assert 'hash mismatch assets/icon.svg' in errors
    assert 'excluded path present: ui/source.py' in errors
    (bundle / 'packaging/runtime_resources.json').unlink()
    assert 'missing packaging/runtime_resources.json' in verify_resources(bundle, source)
