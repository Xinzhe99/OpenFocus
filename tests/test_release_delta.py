"""Round-trip and rejection scenarios for desktop release delta staging."""
import hashlib
import os
from pathlib import Path
import zipfile

import pytest

from tools.build_update_delta import build_delta
from utils import release_delta as delta
from utils.updater import DownloadCancelled


@pytest.fixture
def bundles(tmp_path):
    platform = "windows-x64"
    base = tmp_path / "installed"
    target = tmp_path / "target"
    for root, tag, exe in ((base, "v2.0.0", b"old"), (target, "v2.0.1", b"new")):
        runtime = root / f"_openfocus_runtime_{tag}"
        runtime.mkdir(parents=True)
        (root / "OpenFocus.exe").write_bytes(exe)
        (runtime / "shared.dll").write_bytes(hashlib.shake_256(b"identical runtime").digest(1024 * 1024))
    (base / "old.dll").write_bytes(b"obsolete runtime")
    (base / "OpenFocusData").mkdir()
    (base / "OpenFocusData" / "project.ofproj").write_bytes(b"local project")
    (target / "OpenFocus.portable").touch()
    (target / "docs.txt").write_bytes(b"new docs")
    patch = tmp_path / "update.zip"
    opts = dict(base_tag="v2.0.0", target_tag="v2.0.1", platform=platform)
    build_delta(base, target, patch, **opts)
    return base, target, patch, opts


def test_reconstructs_new_release_without_downloading_renamed_runtime(tmp_path, bundles):
    base, target, patch, opts = bundles
    candidate = tmp_path / "staged"
    delta.stage_delta(patch, base, candidate, **opts)

    assert patch.stat().st_size < (target / "_openfocus_runtime_v2.0.1/shared.dll").stat().st_size // 100
    expected = {p.relative_to(target): p.read_bytes() for p in target.rglob("*")
                if p.is_file() and p.name != "OpenFocus.portable"}
    actual = {p.relative_to(candidate): p.read_bytes() for p in candidate.rglob("*") if p.is_file()}
    assert actual == expected
    assert (base / "OpenFocusData/project.ofproj").read_bytes() == b"local project"


@pytest.mark.parametrize("path,problem", [
    ("OpenFocus.exe", "does not match the patch base"),
    ("_openfocus_runtime_v2.0.0/shared.dll", "checksum differs"),
])
def test_rejects_changed_installation_before_ready(tmp_path, bundles, path, problem):
    base, _, patch, opts = bundles
    source = base / path
    with source.open("r+b") as file:
        file.write(b"!")
    with pytest.raises(ValueError, match=problem):
        delta.stage_delta(patch, base, tmp_path / "staged", **opts)
    assert source.read_bytes().startswith(b"!")


def test_cancellation_never_touches_installed_files(tmp_path, bundles):
    base, _, patch, opts = bundles
    cancelled = False

    def progress(done, _total):
        nonlocal cancelled
        cancelled = done > 1

    with pytest.raises(DownloadCancelled):
        delta.stage_delta(patch, base, tmp_path / "staged", **opts,
                          should_cancel=lambda: cancelled, on_progress=progress)
    assert (base / "OpenFocus.exe").read_bytes() == b"old"


@pytest.mark.skipif(os.name == "nt", reason="needs Unix symlinks and modes")
def test_reconstructs_bundle_symlinks_and_execution_mode(tmp_path):
    base, target, candidate = (tmp_path / name for name in ("base", "target", "staged"))
    for root in (base, target):
        root.mkdir()
        exe = root / "OpenFocus"
        exe.write_bytes(b"#!/bin/sh\nexit 0\n")
        exe.chmod(0o755)
        (root / "runtime").symlink_to("OpenFocus")
    (target / "runtime-data").write_bytes(b"new release data")
    patch = tmp_path / "delta.zip"
    opts = dict(base_tag="v2.0.0", target_tag="v2.0.1", platform="linux-x86_64")
    build_delta(base, target, patch, **opts)
    delta.stage_delta(patch, base, candidate, **opts)
    assert os.readlink(candidate / "runtime") == "OpenFocus"
    assert (candidate / "runtime").read_bytes() == b"#!/bin/sh\nexit 0\n"
    assert os.access(candidate / "OpenFocus", os.X_OK)


def test_rejects_archive_written_outside_manifest(tmp_path, bundles):
    base, _, patch, opts = bundles
    with zipfile.ZipFile(patch, "a") as archive:
        archive.writestr("../other", "unexpected")
    with pytest.raises(ValueError, match="manifest differ"):
        delta.stage_delta(patch, base, tmp_path / "staged", **opts)
    assert not (tmp_path / "other").exists()
