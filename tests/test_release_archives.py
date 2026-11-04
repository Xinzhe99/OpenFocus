"""Small-file boundary and corruption tests for split Windows release assets."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tools import release_archives as ra

LABEL = "v1.2.3"


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stage(tmp_path: Path, zip_size: int = 21, setup_size: int = 19,
           limit: int = 20, part_size: int = 8):
    zip_path = _write(tmp_path / f"OpenFocus-{LABEL}-windows-x64.zip",
                      b"z" * zip_size)
    setup_path = _write(tmp_path / f"OpenFocus-{LABEL}-setup.exe",
                        b"s" * setup_size)
    recovery = _write(tmp_path / "join.ps1", b"# recovery")
    out = tmp_path / "out"
    manifest = ra.stage_windows(LABEL, zip_path, setup_path, out, recovery,
                                limit=limit, part_size=part_size)
    return zip_path, setup_path, out, manifest


def test_under_limit_and_boundary_assets(tmp_path):
    for size, split in ((19, False), (20, True), (21, True)):
        source = _write(tmp_path / f"asset-{size}.bin", b"x" * size)
        info = ra.split_asset(source, tmp_path / f"out-{size}", 20, 8)
        assert bool(info["parts"]) is split
        assert info["size"] == size
        assert info["sha256"] == _digest(b"x" * size)


def test_stage_and_reassemble_roundtrip(tmp_path):
    zip_path, setup_path, out, manifest = _stage(tmp_path)
    ra.verify_stage(manifest, out, limit=20)
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    zip_info = data["assets"]["windows-zip"]
    setup_info = data["assets"]["windows-setup"]
    assert [part["size"] for part in zip_info["parts"]] == [8, 8, 5]
    assert zip_info["sha256"] == _digest(b"z" * 21)
    assert setup_info["parts"] == []
    assert setup_info["sha256"] == _digest(b"s" * 19)
    assert (out / ra.RECOVERY_SCRIPT).is_file()

    joined = tmp_path / "joined"
    ra.reassemble(manifest, out, joined, limit=20)
    assert (joined / zip_path.name).read_bytes() == zip_path.read_bytes()
    assert (joined / setup_path.name).read_bytes() == setup_path.read_bytes()


def test_corrupt_or_missing_part_fails_without_output(tmp_path):
    _, _, out, manifest = _stage(tmp_path)
    zip_part = next(out.glob("*windows-x64.zip.part*"))
    payload = bytearray(zip_part.read_bytes())
    payload[0] ^= 0xFF
    zip_part.write_bytes(bytes(payload))

    joined = tmp_path / "joined"
    with pytest.raises(ra.ReleaseArchiveError):
        ra.reassemble(manifest, out, joined, limit=20)
    assert not (joined / f"OpenFocus-{LABEL}-windows-x64.zip").exists()

    _, _, out2, manifest2 = _stage(tmp_path / "missing")
    next(out2.glob("*windows-x64.zip.part*")).unlink()
    with pytest.raises(ra.ReleaseArchiveError):
        ra.reassemble(manifest2, out2, joined, limit=20)


def test_windows_recovery_preserves_existing_file_on_corruption(tmp_path):
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is required to exercise Windows recovery")

    zip_path, _, out, manifest = _stage(tmp_path)
    part = next(out.glob("*windows-x64.zip.part*"))
    original = part.read_bytes()
    part.write_bytes(b"!" + original[1:])
    target_dir = tmp_path / "recovered"
    target_dir.mkdir()
    target = _write(target_dir / zip_path.name, b"previous valid file")
    script = Path(__file__).resolve().parents[1] / "tools" / "join_openfocus_parts.ps1"
    command = [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
               "Bypass", "-File", str(script), "-Manifest", str(manifest),
               "-OutputDirectory", str(target_dir)]
    environment = os.environ.copy()
    if sys.platform == "win32":
        environment["PSModulePath"] = str(tmp_path / "missing-modules")

    failed = subprocess.run(command, capture_output=True, timeout=30, env=environment)
    assert failed.returncode != 0
    assert target.read_bytes() == b"previous valid file"
    assert sorted(p.name for p in target_dir.iterdir()) == [zip_path.name]

    part.write_bytes(original)
    result = subprocess.run(command, capture_output=True, timeout=30, env=environment)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert target.read_bytes() == zip_path.read_bytes()


def test_manifest_tampering_and_extra_parts_are_rejected(tmp_path):
    _, _, out, manifest = _stage(tmp_path)
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    data["assets"]["windows-zip"]["parts"][0]["name"] += ".bad"
    Path(manifest).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ra.ReleaseArchiveError):
        ra.load_manifest(manifest, limit=20)

    _, _, out2, manifest2 = _stage(tmp_path / "extra")
    _write(out2 / f"OpenFocus-{LABEL}-windows-x64.zip.part999", b"x")
    with pytest.raises(ra.ReleaseArchiveError):
        ra.verify_stage(manifest2, out2, limit=20)


def test_split_full_file_is_rejected_by_stage_verification(tmp_path):
    _, _, out, manifest = _stage(tmp_path)
    _write(out / f"OpenFocus-{LABEL}-windows-x64.zip", b"z" * 21)
    with pytest.raises(ra.ReleaseArchiveError):
        ra.verify_stage(manifest, out, limit=20)


def test_release_asset_name_checks(tmp_path):
    _, _, out, manifest = _stage(tmp_path)
    names = []
    for part in json.loads(Path(manifest).read_text(encoding="utf-8"))[
            "assets"]["windows-zip"]["parts"]:
        names.append(part["name"])
    names += [f"OpenFocus-{LABEL}-setup.exe",
              f"OpenFocus-{LABEL}-macos.dmg",
              f"OpenFocus-{LABEL}-macos-arm64.zip",
              f"OpenFocus-{LABEL}-windows-assets.json",
              ra.RECOVERY_SCRIPT]
    asset_list = _write(tmp_path / "assets.txt", ("\n".join(names) + "\n").encode())
    ra.check_release_assets(manifest, asset_list, LABEL, limit=20)

    _write(tmp_path / "extra.txt",
           ("\n".join(names + [f"OpenFocus-{LABEL}-windows-x64.zip.part999"])
            + "\n").encode())
    with pytest.raises(ra.ReleaseArchiveError):
        ra.check_release_assets(manifest, tmp_path / "extra.txt", LABEL, limit=20)

    _write(tmp_path / "full.txt",
           ("\n".join(names + [f"OpenFocus-{LABEL}-windows-x64.zip"]) + "\n").encode())
    with pytest.raises(ra.ReleaseArchiveError):
        ra.check_release_assets(manifest, tmp_path / "full.txt", LABEL, limit=20)

    _write(tmp_path / "missing.txt",
           ("\n".join(names[1:]) + "\n").encode())
    with pytest.raises(ra.ReleaseArchiveError):
        ra.check_release_assets(manifest, tmp_path / "missing.txt", LABEL, limit=20)


def test_mirror_manifest_marks_split_windows_r2_only(tmp_path):
    zip_path, setup_path, out, manifest = _stage(tmp_path)
    r2_dir = tmp_path / "r2"
    ra.reassemble(manifest, out, r2_dir, limit=20)
    setup_bytes = setup_path.read_bytes()
    zip_bytes = zip_path.read_bytes()
    release = {
        "tag_name": LABEL,
        "published_at": "2026-01-01T00:00:00Z",
        "assets": [
            {"name": f"OpenFocus-{LABEL}-setup.exe", "size": len(setup_bytes),
             "digest": "sha256:" + _digest(setup_bytes),
             "browser_download_url": f"https://github.com/o/r/releases/download/{LABEL}/setup.exe"},
            {"name": f"OpenFocus-{LABEL}-macos.dmg", "size": 3,
             "digest": "sha256:" + _digest(b"dmg"),
             "browser_download_url": f"https://github.com/o/r/releases/download/{LABEL}/macos.dmg"},
            {"name": f"OpenFocus-{LABEL}-macos-arm64.zip", "size": 4,
             "digest": "sha256:" + _digest(b"zip!"),
             "browser_download_url": f"https://github.com/o/r/releases/download/{LABEL}/macos.zip"},
        ],
    }
    _write(r2_dir / f"OpenFocus-{LABEL}-macos.dmg", b"dmg")
    _write(r2_dir / f"OpenFocus-{LABEL}-macos-arm64.zip", b"zip!")
    release_json = _write(tmp_path / "release.json", json.dumps(release).encode())
    releases_json = _write(tmp_path / "releases.json", json.dumps([]).encode())
    output = tmp_path / "latest.json"
    ra.write_mirror_manifest(LABEL, release_json, r2_dir, releases_json,
                             output, windows_manifest=manifest, limit=20)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["tag"] == LABEL
    assert data["assets"]["windows-zip"] == {
        "name": f"OpenFocus-{LABEL}-windows-x64.zip",
        "size": len(zip_bytes),
        "digest": "sha256:" + _digest(zip_bytes),
        "source": "r2",
        "url": None,
    }
    assert data["assets"]["windows-setup"]["url"].startswith("https://github.com/")
    assert "source" not in data["assets"]["windows-setup"]
    assert data["assets"]["macos-dmg"]["url"].startswith("https://github.com/")


def test_zip_tree_keeps_openfocus_root(tmp_path):
    source = tmp_path / "bundle"
    _write(source / "OpenFocus.exe", b"MZ")
    _write(source / "docs" / "README.txt", b"readme")
    output = tmp_path / "out.zip"
    ra.zip_tree(source, output)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "OpenFocus/OpenFocus.exe", "OpenFocus/docs/README.txt"}
        assert archive.read("OpenFocus/OpenFocus.exe") == b"MZ"


def test_mirror_manifest_accepts_legacy_four_complete_assets(tmp_path):
    r2_dir = tmp_path / "r2"
    files = {
        f"OpenFocus-{LABEL}-setup.exe": b"setup",
        f"OpenFocus-{LABEL}-windows-x64.zip": b"winzip",
        f"OpenFocus-{LABEL}-macos.dmg": b"dmg",
        f"OpenFocus-{LABEL}-macos-arm64.zip": b"maczip",
    }
    for name, payload in files.items():
        _write(r2_dir / name, payload)
    assets = []
    for name, payload in files.items():
        assets.append({
            "name": name,
            "size": len(payload),
            "digest": "sha256:" + _digest(payload),
            "browser_download_url": (
                f"https://github.com/o/r/releases/download/{LABEL}/{name}"
            ),
        })
    release = {"tag_name": LABEL, "published_at": "2026-01-01T00:00:00Z",
               "assets": assets}
    release_json = _write(tmp_path / "legacy-release.json",
                          json.dumps(release).encode())
    releases_json = _write(tmp_path / "legacy-releases.json",
                           json.dumps([]).encode())
    output = tmp_path / "legacy-latest.json"
    ra.write_mirror_manifest(LABEL, release_json, r2_dir, releases_json, output)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert set(data["assets"]) == {
        "windows-setup", "windows-zip", "macos-dmg", "macos-zip"}
    for entry in data["assets"].values():
        assert entry["url"].startswith("https://github.com/")
        assert "source" not in entry


def test_gpu_preview_stage_reuses_split_contract(tmp_path):
    label = "v1.2.3"
    recovery = _write(tmp_path / "join.ps1", b"# recovery")
    out = tmp_path / "gpu"
    source = _write(tmp_path / f"OpenFocus-{label}-windows-x64-rocm.zip",
                    b"r" * 21)
    manifest = ra.stage_gpu_preview(
        label, source, out, recovery, "rocm", limit=20, part_size=8)
    ra.verify_stage(manifest, out, limit=20, kind=ra.GPU_KIND)
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    assert data["kind"] == ra.GPU_KIND
    assert data["tag"] == label
    assert data["assets"]["gpu-zip"]["sha256"] == _digest(b"r" * 21)
    assert [part["size"] for part in data["assets"]["gpu-zip"]["parts"]] == [
        8, 8, 5]
    assert (out / ra.RECOVERY_SCRIPT).is_file()

    whole = _write(tmp_path / f"OpenFocus-{label}-windows-x64-xpu.zip",
                   b"x" * 19)
    out2 = tmp_path / "gpu-whole"
    manifest2 = ra.stage_gpu_preview(
        label, whole, out2, recovery, "xpu", limit=20, part_size=8)
    data2 = json.loads(Path(manifest2).read_text(encoding="utf-8"))
    assert data2["assets"]["gpu-zip"]["parts"] == []
    assert (out2 / whole.name).is_file()
    ra.verify_stage(manifest2, out2, limit=20, kind=ra.GPU_KIND)
