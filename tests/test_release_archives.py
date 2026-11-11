"""Small-file boundary and corruption tests for split Windows release assets."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
import textwrap
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


@pytest.mark.parametrize("existing", [False, True])
def test_reassemble_replace_failure_preserves_target(tmp_path, monkeypatch, existing):
    source, _, staged, manifest = _stage(tmp_path)
    target = tmp_path / "joined" / source.name
    if existing:
        _write(target, b"old release")
    original = Path.replace

    def fail_joining(path, destination):
        if path == target.with_name(target.name + ".joining") and destination == target:
            raise OSError("replace failed")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", fail_joining)
    with pytest.raises(OSError, match="replace failed"):
        ra.reassemble(manifest, staged, target.parent, limit=20)
    assert target.read_bytes() == b"old release" if existing else not target.exists()
    assert not target.with_name(target.name + ".joining").exists()


def test_reassemble_replaces_existing_target(tmp_path):
    source, _, staged, manifest = _stage(tmp_path)
    target = _write(tmp_path / "joined" / source.name, b"old release")
    outputs = ra.reassemble(manifest, staged, target.parent, limit=20)
    assert target in outputs
    assert target.read_bytes() == source.read_bytes()
    assert not target.with_name(target.name + ".joining").exists()


@pytest.mark.parametrize("failure", ["corrupt", "missing", "interrupt"])
def test_reassemble_preserves_existing_on_read_failure(tmp_path, monkeypatch, failure):
    source, _, staged, manifest = _stage(tmp_path)
    target = _write(tmp_path / "joined" / source.name, b"old release")
    part_name = json.loads(Path(manifest).read_text(encoding="utf-8"))["assets"]["windows-zip"]["parts"][0]["name"]
    part = staged / part_name
    if failure == "corrupt":
        part.write_bytes(b"x" * part.stat().st_size)
    elif failure == "missing":
        part.unlink()
    else:
        original = ra.sha256_file

        def interrupt(path):
            if path == part:
                raise KeyboardInterrupt()
            return original(path)

        monkeypatch.setattr(ra, "sha256_file", interrupt)
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else ra.ReleaseArchiveError):
        ra.reassemble(manifest, staged, target.parent, limit=20)
    assert target.read_bytes() == b"old release"
    assert not target.with_name(target.name + ".joining").exists()


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


def _mirror_fixture(tmp_path, tag=LABEL):
    directory = tmp_path / "assets"
    files = {ra.expected_name(tag, key): key.encode()
             for key in (*ra.WINDOWS_KEYS, *ra.MACOS_NAMES)}
    assets = []
    for name, content in files.items():
        _write(directory / name, content)
        assets.append({"name": name, "size": len(content),
                       "digest": "sha256:" + _digest(content),
                       "browser_download_url": f"https://github.com/o/r/releases/download/{tag}/{name}"})
    release = {"tag_name": tag, "draft": False, "prerelease": False,
               "assets": assets}
    release_path = _write(tmp_path / "release.json", json.dumps(release).encode())
    releases_path = _write(tmp_path / "releases.json", b"[]")
    return directory, files, release, release_path, releases_path


def test_beta_mirror_requires_explicit_opt_in_and_verified_assets(tmp_path):
    tag = "v1.77.0-beta.3"
    directory, files, release, release_path, releases_path = _mirror_fixture(tmp_path, tag)
    release["prerelease"] = True
    release_path.write_text(json.dumps(release), encoding="utf-8")
    output = tmp_path / "beta.json"
    with pytest.raises(ra.ReleaseArchiveError):
        ra.write_mirror_manifest(tag, release_path, directory, releases_path, output)
    ra.write_mirror_manifest(tag, release_path, directory, releases_path, output,
                             allow_beta=True)
    assert set(item["name"] for item in json.loads(output.read_text())["assets"].values()) == set(files)
    output.unlink()
    name, content = next(iter(files.items()))
    _write(directory / name, b"x" * len(content))
    with pytest.raises(ra.ReleaseArchiveError, match="SHA-256"):
        ra.write_mirror_manifest(tag, release_path, directory, releases_path, output,
                                 allow_beta=True)
    assert not output.exists()




def test_beta_workflow_run_requires_matching_build_identity(tmp_path):
    tag = "v1.77.0-beta.3"
    directory, files, release, release_path, releases_path = _mirror_fixture(tmp_path, tag)
    release["prerelease"] = True
    identity = {"schema_version": 1, "tag": tag, "version": "1.77.0",
                "sha": "a" * 40, "run_id": 123, "run_attempt": 1,
                "assets": [{"name": name, "size": len(content), "sha256": _digest(content)}
                           for name, content in files.items()]}
    identity_path = _write(tmp_path / "identity.json", json.dumps(identity).encode())
    data = identity_path.read_bytes()
    release["assets"].append({"name": f"OpenFocus-{tag}-build.json", "size": len(data),
                              "digest": "sha256:" + _digest(data)})
    release_path.write_text(json.dumps(release), encoding="utf-8")
    output = tmp_path / "beta.json"
    with pytest.raises(ra.ReleaseArchiveError, match="workflow run SHA"):
        ra.write_mirror_manifest(tag, release_path, directory, releases_path, output,
                                 build_identity=identity_path, head_sha="b" * 40, allow_beta=True)
    assert not output.exists()
    ra.write_mirror_manifest(tag, release_path, directory, releases_path, output,
                             build_identity=identity_path, head_sha="a" * 40, allow_beta=True)
    assert json.loads(output.read_text())["tag"] == tag
    output.unlink()
    identity["version"] = tag[1:]
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    data = identity_path.read_bytes()
    release["assets"][-1].update(size=len(data), digest="sha256:" + _digest(data))
    release_path.write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(ra.ReleaseArchiveError, match="invalid build identity"):
        ra.write_mirror_manifest(tag, release_path, directory, releases_path, output,
                                 build_identity=identity_path, head_sha="a" * 40, allow_beta=True)


@pytest.mark.parametrize("tag,prerelease", [
    ("v1.77.0", False), ("v1.77.0-beta.0", True),
    ("v1.77.0-beta.03", True), ("v1.77.0-rc.1", True),
    ("gpu-v1.77.0-beta.3", True), ("v1.77.0-beta.3", False),
])
def test_beta_mirror_rejects_other_tags_and_non_prereleases(tmp_path, tag, prerelease):
    directory, _, release, release_path, releases_path = _mirror_fixture(tmp_path, tag)
    release["prerelease"] = prerelease
    release_path.write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(ra.ReleaseArchiveError):
        ra.write_mirror_manifest(tag, release_path, directory, releases_path,
                                 tmp_path / "out.json", allow_beta=True)


@pytest.mark.parametrize("candidate,previous,expected", [
    ("v1.77.0-beta.3", "v1.77.0-beta.2", "publish"),
    ("v1.77.0-beta.2", "v1.77.0-beta.10", "skip"),
    ("v1.78-beta.1", "v1.77.99-beta.9", "publish"),
    ("v1.77-beta.3", "v1.77.0-beta.3", "publish"),
    ("v1.9-beta.99", "v1.10-beta.1", "skip"),
])
def test_beta_metadata_never_regresses(tmp_path, candidate, previous, expected):
    path = _write(tmp_path / "previous.json", json.dumps({"tag": previous}).encode())
    assert ra.beta_metadata_action(candidate, path) == expected


def test_beta_metadata_rejects_invalid_previous_tag(tmp_path):
    path = _write(tmp_path / "previous.json", b'{"tag":"v1.77.0"}')
    with pytest.raises(ra.ReleaseArchiveError):
        ra.beta_metadata_action("v1.77.0-beta.3", path)


@pytest.mark.parametrize("failure", ["corrupt", "missing", "no-digest", "draft", "prerelease"])
def test_mirror_preflight_rejects_unverified_assets(tmp_path, failure):
    directory, files, release, release_path, releases_path = _mirror_fixture(tmp_path)
    name = next(iter(files))
    if failure == "corrupt":
        _write(directory / name, b"x" * len(files[name]))
    elif failure == "missing":
        (directory / name).unlink()
    elif failure == "no-digest":
        release["assets"][0]["digest"] = None
    else:
        release[failure] = True
    release_path.write_text(json.dumps(release), encoding="utf-8")
    output = tmp_path / "latest.json"
    with pytest.raises(ra.ReleaseArchiveError):
        ra.write_mirror_manifest(LABEL, release_path, directory, releases_path, output)
    assert not output.exists()


def test_mirror_identity_supplies_missing_digest_and_binds_sha(tmp_path):
    directory, files, release, release_path, releases_path = _mirror_fixture(tmp_path)
    for asset in release["assets"]:
        asset["digest"] = None
    identity = {"schema_version": 1, "tag": LABEL, "version": LABEL[1:],
                "sha": "a" * 40, "run_id": 123, "run_attempt": 1,
                "assets": [{"name": name, "size": len(content), "sha256": _digest(content)}
                           for name, content in files.items()]}
    identity_path = _write(tmp_path / "identity.json", json.dumps(identity).encode())
    identity_bytes = identity_path.read_bytes()
    release["assets"].append({"name": f"OpenFocus-{LABEL}-build.json",
                              "size": len(identity_bytes),
                              "digest": "sha256:" + _digest(identity_bytes)})
    release_path.write_text(json.dumps(release), encoding="utf-8")
    output = tmp_path / "latest.json"
    ra.write_mirror_manifest(LABEL, release_path, directory, releases_path, output,
                             build_identity=identity_path, head_sha="a" * 40)
    assert len(json.loads(output.read_text(encoding="utf-8"))["assets"]) == 4
    output.unlink()
    with pytest.raises(ra.ReleaseArchiveError, match="workflow run SHA"):
        ra.write_mirror_manifest(LABEL, release_path, directory, releases_path, output,
                                 build_identity=identity_path, head_sha="b" * 40)
    release["assets"].append({"name": "unexpected.bin"})
    release_path.write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(ra.ReleaseArchiveError, match="assets differ from build identity"):
        ra.write_mirror_manifest(LABEL, release_path, directory, releases_path, output,
                                 build_identity=identity_path)
    release["assets"].pop()
    release_path.write_text(json.dumps(release), encoding="utf-8")
    name, content = next(iter(files.items()))
    _write(directory / name, b"x" * len(content))
    with pytest.raises(ra.ReleaseArchiveError, match="build identity"):
        ra.write_mirror_manifest(LABEL, release_path, directory, releases_path, output,
                                 build_identity=identity_path)
    assert not output.exists()


def test_mirror_rejects_corrupt_reassembled_split_asset(tmp_path):
    zip_path, setup_path, staged, manifest = _stage(tmp_path)
    directory = tmp_path / "mirror"
    ra.reassemble(manifest, staged, directory, limit=20)
    release_assets = []
    for name, content in ((setup_path.name, setup_path.read_bytes()),
                          (ra.expected_name(LABEL, "macos-dmg"), b"dmg"),
                          (ra.expected_name(LABEL, "macos-zip"), b"zip")):
        _write(directory / name, content)
        release_assets.append({"name": name, "size": len(content),
                               "digest": "sha256:" + _digest(content),
                               "browser_download_url": f"https://github.com/o/r/releases/download/{LABEL}/{name}"})
    release = _write(tmp_path / "release.json", json.dumps({"tag_name": LABEL, "assets": release_assets}).encode())
    releases = _write(tmp_path / "releases.json", b"[]")
    _write(directory / zip_path.name, b"x" * zip_path.stat().st_size)
    with pytest.raises(ra.ReleaseArchiveError, match="SHA-256"):
        ra.write_mirror_manifest(LABEL, release, directory, releases, tmp_path / "result.json",
                                 windows_manifest=manifest, limit=20)
    assert not (tmp_path / "result.json").exists()


def test_mirror_accepts_published_metadata_as_independent_digest(tmp_path):
    directory, files, release, release_path, releases_path = _mirror_fixture(tmp_path)
    output = tmp_path / "previous.json"
    ra.write_mirror_manifest(LABEL, release_path, directory, releases_path, output)
    for asset in release["assets"]:
        asset["digest"] = None
    release_path.write_text(json.dumps(release), encoding="utf-8")
    new_output = tmp_path / "new.json"
    ra.write_mirror_manifest(LABEL, release_path, directory, releases_path, new_output,
                             trusted_metadata=output)
    assert len(json.loads(new_output.read_text(encoding="utf-8"))["assets"]) == 4
    name, content = next(iter(files.items()))
    _write(directory / name, b"x" * len(content))
    with pytest.raises(ra.ReleaseArchiveError, match="build identity"):
        ra.write_mirror_manifest(LABEL, release_path, directory, releases_path, new_output,
                                 trusted_metadata=output)


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="requires POSIX shell")
@pytest.mark.parametrize("corrupt", [False, True])
def test_mirror_workflow_preflight_precedes_r2_writes(tmp_path, corrupt):
    directory, files, _, _, _ = _mirror_fixture(tmp_path)
    if corrupt:
        name, content = next(iter(files.items()))
        _write(directory / name, b"x" * len(content))
    _write(tmp_path / "assets.txt", "\n".join(files).encode())
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "tools").mkdir()
    shutil.copy(root / "tools" / "release_archives.py", tmp_path / "tools")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def command(name, source):
        path = _write(bin_dir / name, (f"#!{sys.executable}\n" + textwrap.dedent(source)).encode())
        path.chmod(0o755)

    command("gh", """
        import json
        print(json.dumps([[]]))
    """)
    command("jq", """
        import json, sys
        args = sys.argv[1:]
        if '-r' in args and '.assets[].name' in args:
            with open(args[-1]) as stream:
                data = json.load(stream)
            print('\\n'.join(item['name'] for item in data['assets'].values()))
        elif '--arg' in args and 'name' in args:
            name = args[args.index('name') + 1]
            with open(args[-1]) as stream:
                data = json.load(stream)
            print(next(item['digest'] for item in data['assets'].values() if item['name'] == name))
        else:
            print('[]')
    """)
    command("aws", """
        import os, pathlib, shutil, sys
        args = sys.argv[1:]
        root = pathlib.Path(os.environ['RUNNER_TEMP'])
        if args[:2] == ['s3', 'cp'] and args[2].endswith('latest-release.json'):
            sys.exit(1)
        if args[:2] == ['s3', 'cp'] and args[2].startswith('assets/'):
            with open(root / 'writes', 'a') as stream:
                stream.write(args[3] + '\\n')
            shutil.copyfile(args[2], root / pathlib.Path(args[3]).name)
        elif args[:2] == ['s3api', 'head-object']:
            print((root / args[args.index('--key') + 1]).stat().st_size)
        elif args[:2] == ['s3', 'cp'] and args[2].startswith('s3://'):
            sys.stdout.buffer.write((root / pathlib.Path(args[2]).name).read_bytes())
        else:
            sys.exit(1)
    """)
    workflow = (root / ".github" / "workflows" / "r2-mirror.yml").read_text(encoding="utf-8")

    def step(name, next_name):
        block = workflow.split(f"      - name: {name}\n", 1)[1].split(
            f"      - name: {next_name}\n", 1)[0]
        script = block.split("        run: |\n", 1)[1]
        return textwrap.dedent(script).replace("${{ steps.release.outputs.tag }}", LABEL)

    script = step("Verify all assets before writing to R2", "Upload to R2")
    script += "\n" + step("Upload to R2", "Publish latest mirrored release metadata")
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
               RUNNER_TEMP=str(tmp_path), GITHUB_REPOSITORY="o/r", R2_ACCOUNT_ID="test",
               HEAD_SHA="", WINDOWS_MANIFEST="")
    result = subprocess.run(["bash", "-e", "-c", script], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    writes = tmp_path / "writes"
    if corrupt:
        assert result.returncode != 0
        assert not writes.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert {line.rsplit('/', 1)[-1] for line in writes.read_text().splitlines()} == set(files)


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
