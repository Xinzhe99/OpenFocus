"""Split and reassemble Windows release assets around GitHub's 2 GiB limit.

GitHub rejects a release asset of 2 GiB or more. A complete Windows zip/setup
that reaches the limit is staged as raw ``<name>.partNNN`` chunks plus a JSON
manifest carrying exact sizes and SHA-256 values. The R2 mirror downloads those
strict names, verifies every part and the joined file, and uploads the canonical
full name. Assets below the limit stay as one file.

The same manifest drives the Windows recovery script, so joining the GitHub
parts yields exactly the file the updater verifies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

GITHUB_ASSET_LIMIT = 2_147_483_648
DEFAULT_PART_BYTES = 1_900_000_000
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STABLE_TAG_RE = re.compile(r"^v[0-9]+(?:\.[0-9]+)*$")
BETA_TAG_RE = re.compile(r"^v([0-9]+(?:\.[0-9]+)*)-beta\.([1-9][0-9]*)$")
WINDOWS_KEYS = ("windows-zip", "windows-setup")
GPU_KEY = "gpu-zip"
GPU_KIND = "gpu-preview"
WINDOWS_NAMES = {
    "windows-zip": "windows-x64.zip",
    "windows-setup": "setup.exe",
}
MACOS_NAMES = {
    "macos-dmg": "macos.dmg",
    "macos-zip": "macos-arm64.zip",
}
RECOVERY_SCRIPT = "join-openfocus-parts.ps1"


class ReleaseArchiveError(ValueError):
    """A split manifest or staged file does not satisfy the release contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_basename(name: str) -> str:
    name = str(name)
    if (not name or name != PurePosixPath(name).name
            or name in (".", "..") or not SAFE_NAME_RE.fullmatch(name)):
        raise ReleaseArchiveError(f"unsafe release asset name: {name!r}")
    return name


def _digest_ok(value: str) -> bool:
    return bool(SHA256_RE.fullmatch(str(value or "")))


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ReleaseArchiveError(f"expected a JSON object: {path}")
    return data


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def expected_name(label: str, key: str) -> str:
    _safe_basename(label)
    if key in WINDOWS_NAMES:
        return f"OpenFocus-{label}-{WINDOWS_NAMES[key]}"
    if key in MACOS_NAMES:
        return f"OpenFocus-{label}-{MACOS_NAMES[key]}"
    raise ReleaseArchiveError(f"unknown logical asset: {key}")


def split_asset(source: Path, out_dir: Path, limit: int,
                part_size: int) -> dict:
    """Copy a whole asset, or write ordered raw parts and return its record."""
    if limit <= 0 or part_size <= 0 or part_size >= limit:
        raise ReleaseArchiveError("part size must be positive and below the limit")
    source = Path(source)
    if not source.is_file() or source.stat().st_size <= 0:
        raise ReleaseArchiveError(f"missing or empty asset: {source}")
    _safe_basename(source.name)
    size = source.stat().st_size
    full_sha = sha256_file(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    if size < limit:
        shutil.copy2(source, out_dir / source.name)
        return {"name": source.name, "size": size, "sha256": full_sha,
                "parts": []}

    count = (size + part_size - 1) // part_size
    width = max(3, len(str(count)))
    parts = []
    with open(source, "rb") as stream:
        for index in range(1, count + 1):
            expected = min(part_size, size - (index - 1) * part_size)
            name = f"{source.name}.part{index:0{width}d}"
            target = out_dir / name
            digest = hashlib.sha256()
            written = 0
            with open(target, "wb") as part_file:
                while written < expected:
                    chunk = stream.read(min(1024 * 1024, expected - written))
                    if not chunk:
                        raise ReleaseArchiveError(
                            f"short read while splitting {source.name}")
                    part_file.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
            if written >= limit:
                raise ReleaseArchiveError(f"part is not below the limit: {name}")
            parts.append({"name": name, "size": written,
                          "sha256": digest.hexdigest()})
    if sum(part["size"] for part in parts) != size:
        raise ReleaseArchiveError(f"part sizes do not add up for {source.name}")
    return {"name": source.name, "size": size, "sha256": full_sha,
            "parts": parts}


def _validate_asset_record(info: dict, key: str, limit: int,
                           full_name: str = "") -> None:
    if not isinstance(info, dict):
        raise ReleaseArchiveError(f"{key}: asset record is missing")
    name = _safe_basename(str(info.get("name") or ""))
    if full_name and name != full_name:
        raise ReleaseArchiveError(f"{key}: unexpected full name {name!r}")
    size = info.get("size")
    if type(size) is not int or size <= 0:
        raise ReleaseArchiveError(f"{key}: invalid full size")
    if not _digest_ok(info.get("sha256")):
        raise ReleaseArchiveError(f"{key}: invalid full SHA-256")
    parts = info.get("parts")
    if not isinstance(parts, list):
        raise ReleaseArchiveError(f"{key}: parts must be a list")
    expected_index = 1
    part_total = 0
    part_names = set()
    for part in parts:
        if not isinstance(part, dict):
            raise ReleaseArchiveError(f"{key}: invalid part record")
        part_name = _safe_basename(str(part.get("name") or ""))
        match = re.fullmatch(re.escape(name) + r"\.part(\d{3,})", part_name)
        if not match or int(match.group(1)) != expected_index:
            raise ReleaseArchiveError(
                f"{key}: part order/name is invalid: {part_name!r}")
        expected_index += 1
        if part_name in part_names:
            raise ReleaseArchiveError(f"{key}: duplicate part {part_name}")
        part_names.add(part_name)
        part_size = part.get("size")
        if type(part_size) is not int or part_size <= 0 or part_size >= limit:
            raise ReleaseArchiveError(
                f"{key}: part size is not below the limit: {part_name}")
        if not _digest_ok(part.get("sha256")):
            raise ReleaseArchiveError(f"{key}: invalid part SHA-256")
        part_total += part_size
    if parts and part_total != size:
        raise ReleaseArchiveError(f"{key}: part sizes do not add up")
    if not parts and size >= limit:
        raise ReleaseArchiveError(
            f"{key}: a full asset at or above the limit must be split")


def _validate_manifest(manifest: dict, limit: int,
                       kind: str = "stable") -> dict:
    if manifest.get("schema") != 1:
        raise ReleaseArchiveError("unsupported split manifest schema")
    tag = _safe_basename(str(manifest.get("tag") or ""))
    assets = manifest.get("assets")
    if kind == "stable":
        if not isinstance(assets, dict) or set(assets) != set(WINDOWS_KEYS):
            raise ReleaseArchiveError(
                "manifest must contain windows-zip and windows-setup")
        keys = WINDOWS_KEYS
    elif kind == GPU_KIND:
        if manifest.get("kind") != GPU_KIND:
            raise ReleaseArchiveError("not a GPU preview manifest")
        if not isinstance(assets, dict) or set(assets) != {GPU_KEY}:
            raise ReleaseArchiveError("GPU manifest must contain gpu-zip")
        keys = (GPU_KEY,)
    else:
        raise ReleaseArchiveError(f"unknown manifest kind: {kind}")
    for key in keys:
        info = assets[key]
        if not isinstance(info, dict):
            raise ReleaseArchiveError(f"{key}: asset record is missing")
        full_name = expected_name(tag, key) if kind == "stable" else ""
        if kind == GPU_KIND:
            candidate = _safe_basename(str(info.get("name") or ""))
            if (not candidate.startswith(f"OpenFocus-{tag}-windows-x64-")
                    or not candidate.endswith(".zip")):
                raise ReleaseArchiveError(
                    f"{key}: unexpected GPU preview name {candidate!r}")
        _validate_asset_record(info, key, limit, full_name)
    return manifest


def load_manifest(path: Path, limit: int = GITHUB_ASSET_LIMIT,
                  kind: str = "stable") -> dict:
    return _validate_manifest(_load_json(Path(path)), limit, kind)


def stage_windows(label: str, zip_path: Path, setup_path: Path, out_dir: Path,
                  recovery_script: Path, limit: int = GITHUB_ASSET_LIMIT,
                  part_size: int = DEFAULT_PART_BYTES) -> Path:
    """Create the release staging directory for both Windows logical assets."""
    label = _safe_basename(label)
    expected = {
        "windows-zip": Path(zip_path),
        "windows-setup": Path(setup_path),
    }
    records = {}
    for key, source in expected.items():
        if source.name != expected_name(label, key):
            raise ReleaseArchiveError(
                f"{key}: expected {expected_name(label, key)!r}, got {source.name!r}")
        records[key] = split_asset(source, Path(out_dir), limit, part_size)
    manifest = {"schema": 1, "tag": label, "assets": records}
    manifest_path = Path(out_dir) / f"OpenFocus-{label}-windows-assets.json"
    _write_json(manifest_path, manifest)
    script = Path(recovery_script)
    if not script.is_file():
        raise ReleaseArchiveError(f"missing recovery script: {script}")
    shutil.copy2(script, Path(out_dir) / RECOVERY_SCRIPT)
    load_manifest(manifest_path, limit)
    return manifest_path


def stage_gpu_preview(label: str, source: Path, out_dir: Path,
                      recovery_script: Path, variant: str,
                      limit: int = GITHUB_ASSET_LIMIT,
                      part_size: int = DEFAULT_PART_BYTES) -> Path:
    """Stage one GPU preview zip with the same split/recovery format."""
    label = _safe_basename(label)
    variant = _safe_basename(variant)
    source = Path(source)
    if (not source.name.startswith(f"OpenFocus-{label}-windows-x64-")
            or not source.name.endswith(".zip")):
        raise ReleaseArchiveError(
            f"unexpected GPU preview asset name: {source.name!r}")
    info = split_asset(source, Path(out_dir), limit, part_size)
    manifest = {"schema": 1, "tag": label, "kind": GPU_KIND,
                "assets": {GPU_KEY: info}}
    manifest_path = (Path(out_dir) /
                     f"OpenFocus-{label}-gpu-{variant}-assets.json")
    _write_json(manifest_path, manifest)
    script = Path(recovery_script)
    if not script.is_file():
        raise ReleaseArchiveError(f"missing recovery script: {script}")
    shutil.copy2(script, Path(out_dir) / RECOVERY_SCRIPT)
    load_manifest(manifest_path, limit, kind=GPU_KIND)
    return manifest_path


def _verify_part_files(info: dict, directory: Path, label: str,
                       limit: int) -> list:
    errors = []
    parts = info["parts"]
    full_name = info["name"]
    if parts:
        if (directory / full_name).exists():
            errors.append(f"{label}: full asset must not be staged when split")
        expected = {part["name"] for part in parts}
        for part in parts:
            path = directory / part["name"]
            if not path.is_file():
                errors.append(f"{label}: missing part {part['name']}")
                continue
            if path.stat().st_size != part["size"]:
                errors.append(f"{label}: size differs for {part['name']}")
            elif sha256_file(path) != part["sha256"]:
                errors.append(f"{label}: SHA-256 differs for {part['name']}")
        for path in directory.glob(full_name + ".part*"):
            if path.name not in expected:
                errors.append(f"{label}: unexpected part {path.name}")
    else:
        path = directory / full_name
        if not path.is_file():
            errors.append(f"{label}: missing full asset {full_name}")
        else:
            if path.stat().st_size >= limit:
                errors.append(f"{label}: full asset is not below the limit")
            if path.stat().st_size != info["size"]:
                errors.append(f"{label}: size differs for {full_name}")
            elif sha256_file(path) != info["sha256"]:
                errors.append(f"{label}: SHA-256 differs for {full_name}")
        for path in directory.glob(full_name + ".part*"):
            errors.append(f"{label}: unexpected part {path.name}")
    return errors


def verify_stage(manifest_path: Path, directory: Path,
                 limit: int = GITHUB_ASSET_LIMIT,
                 kind: str = "stable") -> None:
    manifest = load_manifest(manifest_path, limit, kind=kind)
    keys = WINDOWS_KEYS if kind == "stable" else (GPU_KEY,)
    directory = Path(directory)
    errors = []
    for key in keys:
        errors.extend(_verify_part_files(
            manifest["assets"][key], directory, key, limit))
    expected_parts = {
        part["name"]
        for key in keys for part in manifest["assets"][key]["parts"]
    }
    for path in directory.glob("*.part*"):
        if path.name not in expected_parts:
            errors.append(f"unexpected staged part: {path.name}")
    if errors:
        raise ReleaseArchiveError("; ".join(errors))


def check_release_assets(manifest_path: Path, asset_list: Path,
                         tag: str, limit: int = GITHUB_ASSET_LIMIT) -> None:
    manifest = load_manifest(manifest_path, limit)
    if manifest["tag"] != tag:
        raise ReleaseArchiveError(
            f"manifest tag {manifest['tag']!r} != release tag {tag!r}")
    names = []
    for line in Path(asset_list).read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            names.append(_safe_basename(name))
    counts = Counter(names)
    duplicates = [name for name, count in counts.items() if count > 1]
    if duplicates:
        raise ReleaseArchiveError(f"duplicate release assets: {duplicates}")
    present = set(names)
    errors = []
    expected_parts = set()
    for key in WINDOWS_KEYS:
        info = manifest["assets"][key]
        full_name = info["name"]
        if info["parts"]:
            if full_name in present:
                errors.append(f"{key}: full asset must not be on GitHub")
            expected_parts.update(part["name"] for part in info["parts"])
            for part in info["parts"]:
                if part["name"] not in present:
                    errors.append(f"{key}: missing part {part['name']}")
            for name in present:
                if name.startswith(full_name + ".part") and name not in expected_parts:
                    errors.append(f"{key}: unexpected part {name}")
        else:
            if full_name not in present:
                errors.append(f"{key}: missing full asset {full_name}")
            for name in present:
                if name.startswith(full_name + ".part"):
                    errors.append(f"{key}: unexpected part {name}")
    if errors:
        raise ReleaseArchiveError("; ".join(errors))


def list_downloads(manifest_path: Path,
                   limit: int = GITHUB_ASSET_LIMIT) -> list:
    manifest = load_manifest(manifest_path, limit)
    names = []
    for key in WINDOWS_KEYS:
        info = manifest["assets"][key]
        names.extend(part["name"] for part in info["parts"])
        if not info["parts"]:
            names.append(info["name"])
    return names


def reassemble(manifest_path: Path, source_dir: Path, output_dir: Path,
               limit: int = GITHUB_ASSET_LIMIT) -> list:
    manifest = load_manifest(manifest_path, limit)
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for key in WINDOWS_KEYS:
        info = manifest["assets"][key]
        target = output_dir / info["name"]
        if info["parts"]:
            temp = target.with_name(target.name + ".joining")
            if temp.exists():
                temp.unlink()
            try:
                digest = hashlib.sha256()
                written = 0
                with open(temp, "wb") as out:
                    for part in info["parts"]:
                        path = source_dir / part["name"]
                        if not path.is_file():
                            raise ReleaseArchiveError(
                                f"{key}: missing part {part['name']}")
                        if path.stat().st_size != part["size"]:
                            raise ReleaseArchiveError(
                                f"{key}: part size differs: {part['name']}")
                        if sha256_file(path) != part["sha256"]:
                            raise ReleaseArchiveError(
                                f"{key}: part SHA-256 differs: {part['name']}")
                        with open(path, "rb") as stream:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                out.write(chunk)
                                digest.update(chunk)
                                written += len(chunk)
                if written != info["size"] or digest.hexdigest() != info["sha256"]:
                    raise ReleaseArchiveError(
                        f"{key}: joined file does not match the manifest")
                temp.replace(target)
            except BaseException:
                if temp.exists():
                    temp.unlink()
                raise
        else:
            path = source_dir / info["name"]
            if not path.is_file() or path.stat().st_size != info["size"]:
                raise ReleaseArchiveError(f"{key}: missing or wrong full asset")
            if sha256_file(path) != info["sha256"]:
                raise ReleaseArchiveError(f"{key}: full SHA-256 differs")
            shutil.copy2(path, target)
        outputs.append(target)
    return outputs


def _github_asset(release: dict, name: str, r2_dir: Path,
                  trusted: dict = None) -> dict:
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise ReleaseArchiveError("GitHub release has no asset list")
    matches = [item for item in assets
               if isinstance(item, dict) and item.get("name") == name]
    if len(matches) != 1:
        raise ReleaseArchiveError(f"expected exactly one GitHub asset {name!r}")
    asset = matches[0]
    size = asset.get("size")
    url = str(asset.get("browser_download_url") or "")
    if type(size) is not int or size <= 0:
        raise ReleaseArchiveError(f"invalid GitHub asset size: {name}")
    if not url.startswith("https://github.com/") or "/releases/download/" not in url:
        raise ReleaseArchiveError(f"invalid GitHub asset URL: {name}")
    local = Path(r2_dir) / name
    if not local.is_file() or local.stat().st_size != size:
        raise ReleaseArchiveError(f"mirrored file is missing or wrong size: {name}")
    local_digest = "sha256:" + sha256_file(local)
    github_digest = str(asset.get("digest") or "").lower()
    expected = (trusted or {}).get(name)
    if not github_digest and not expected:
        raise ReleaseArchiveError(f"no trusted SHA-256 for GitHub asset: {name}")
    if github_digest and github_digest != local_digest:
        raise ReleaseArchiveError(f"mirrored SHA-256 differs from GitHub: {name}")
    if expected and (expected["size"] != size or
                     "sha256:" + expected["sha256"] != local_digest):
        raise ReleaseArchiveError(f"mirrored SHA-256 differs from build identity: {name}")
    return {"name": name, "size": size, "url": url, "digest": local_digest}


def _mirror_identity(path: Path, tag: str, release: dict, head_sha: str) -> dict:
    identity = _load_json(path)
    identity_name = f"OpenFocus-{tag}-build.json"
    matches = [a for a in release.get("assets", [])
               if isinstance(a, dict) and a.get("name") == identity_name]
    if (len(matches) != 1 or type(matches[0].get("size")) is not int or
            matches[0]["size"] <= 0 or path.stat().st_size != matches[0]["size"]):
        raise ReleaseArchiveError("build identity release asset is missing or wrong size")
    digest = matches[0].get("digest")
    if digest and digest.lower() != "sha256:" + sha256_file(path):
        raise ReleaseArchiveError("build identity GitHub digest differs")
    beta = BETA_TAG_RE.fullmatch(tag)
    version = beta.group(1) if beta else tag.removeprefix("v")
    if (identity.get("schema_version") != 1 or identity.get("tag") != tag or
            identity.get("version") != version or
            not re.fullmatch(r"[0-9a-f]{40}", str(identity.get("sha", ""))) or
            not str(identity.get("run_id", "")).isdigit() or
            not str(identity.get("run_attempt", "")).isdigit() or
            int(identity["run_id"]) < 1 or int(identity["run_attempt"]) < 1):
        raise ReleaseArchiveError("invalid build identity")
    if head_sha and identity["sha"] != head_sha.lower():
        raise ReleaseArchiveError("workflow run SHA differs from build identity")
    records = identity.get("assets")
    if not isinstance(records, list) or not records:
        raise ReleaseArchiveError("build identity has no assets")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ReleaseArchiveError("invalid build identity asset")
        name = _safe_basename(record.get("name", ""))
        if (name in result or type(record.get("size")) is not int or
                record["size"] <= 0 or not _digest_ok(record.get("sha256"))):
            raise ReleaseArchiveError(f"invalid build identity asset: {name}")
        matches = [a for a in release.get("assets", [])
                   if isinstance(a, dict) and a.get("name") == name]
        if len(matches) != 1 or matches[0].get("size") != record["size"]:
            raise ReleaseArchiveError(f"build identity release asset differs: {name}")
        digest = matches[0].get("digest")
        if digest and digest.lower() != "sha256:" + record["sha256"]:
            raise ReleaseArchiveError(f"build identity GitHub digest differs: {name}")
        result[name] = record
    release_assets = release.get("assets")
    if (not isinstance(release_assets, list) or
            any(not isinstance(asset, dict) or not isinstance(asset.get("name"), str)
                for asset in release_assets)):
        raise ReleaseArchiveError("invalid GitHub release asset inventory")
    release_names = [asset["name"] for asset in release_assets]
    if (len(release_names) != len(set(release_names)) or
            set(release_names) != set(result) | {identity_name}):
        raise ReleaseArchiveError("GitHub release assets differ from build identity")
    return result


def beta_version(tag: str) -> tuple:
    match = BETA_TAG_RE.fullmatch(tag)
    if not match:
        raise ReleaseArchiveError(f"invalid beta release tag: {tag}")
    numbers = tuple(int(part) for part in match.group(1).split("."))
    while len(numbers) > 1 and numbers[-1] == 0:
        numbers = numbers[:-1]
    return numbers, int(match.group(2))


def beta_metadata_action(tag: str, previous: Path) -> str:
    candidate = beta_version(tag)
    prior = _load_json(Path(previous))
    old_tag = prior.get("tag")
    if not isinstance(old_tag, str):
        raise ReleaseArchiveError("invalid previous beta metadata tag")
    return "publish" if candidate >= beta_version(old_tag) else "skip"


def write_mirror_manifest(tag: str, release_json: Path, r2_dir: Path,
                          releases_json: Path, output: Path,
                          windows_manifest: Path = None,
                          limit: int = GITHUB_ASSET_LIMIT,
                          build_identity: Path = None, head_sha: str = "",
                          trusted_metadata: Path = None,
                          allow_beta: bool = False) -> None:
    tag = _safe_basename(tag)
    release = _load_json(Path(release_json))
    if release.get("tag_name") != tag:
        raise ReleaseArchiveError("GitHub release tag does not match")
    if allow_beta:
        beta_version(tag)
    elif not STABLE_TAG_RE.fullmatch(tag):
        raise ReleaseArchiveError("invalid stable release tag")
    if release.get("draft", False) is not False or release.get("prerelease", False) is not allow_beta:
        raise ReleaseArchiveError("release must be published and match the requested channel")
    if head_sha and not build_identity:
        raise ReleaseArchiveError("workflow run requires a build identity")
    trusted = _mirror_identity(Path(build_identity), tag, release, head_sha) if build_identity else {}
    if trusted_metadata:
        previous = _load_json(Path(trusted_metadata))
        if previous.get("tag") != tag or not isinstance(previous.get("assets"), dict):
            raise ReleaseArchiveError("trusted mirror metadata tag or assets differ")
        for key in (*WINDOWS_KEYS, *MACOS_NAMES):
            entry = previous["assets"].get(key)
            name = expected_name(tag, key)
            if (not isinstance(entry, dict) or entry.get("name") != name or
                    type(entry.get("size")) is not int or entry["size"] <= 0 or
                    not str(entry.get("digest", "")).startswith("sha256:") or
                    not _digest_ok(entry["digest"][7:])):
                raise ReleaseArchiveError(f"invalid trusted mirror metadata: {name}")
            record = {"name": name, "size": entry["size"], "sha256": entry["digest"][7:]}
            if build_identity and name not in trusted and not (
                    windows_manifest and key in WINDOWS_KEYS):
                raise ReleaseArchiveError(f"build identity missing asset: {name}")
            if name in trusted and trusted[name] != record:
                raise ReleaseArchiveError(f"trusted mirror metadata differs from build identity: {name}")
            trusted[name] = record
    r2_dir = Path(r2_dir)
    assets = {}
    if windows_manifest:
        manifest = load_manifest(windows_manifest, limit)
        if manifest["tag"] != tag:
            raise ReleaseArchiveError("Windows manifest tag does not match")
        if build_identity:
            manifest_name = f"OpenFocus-{tag}-windows-assets.json"
            record = trusted.get(manifest_name)
            if (not record or Path(windows_manifest).stat().st_size != record["size"] or
                    sha256_file(Path(windows_manifest)) != record["sha256"]):
                raise ReleaseArchiveError("Windows manifest differs from build identity")
            for info in manifest["assets"].values():
                for item in info["parts"] or [info]:
                    record = trusted.get(item["name"])
                    if not record or record["size"] != item["size"] or record["sha256"] != item["sha256"]:
                        raise ReleaseArchiveError(f"Windows asset differs from build identity: {item['name']}")
        for key in WINDOWS_KEYS:
            info = manifest["assets"][key]
            if info["parts"]:
                full = r2_dir / info["name"]
                if not full.is_file() or full.stat().st_size != info["size"]:
                    raise ReleaseArchiveError(f"{key}: R2 full asset is missing")
                if sha256_file(full) != info["sha256"]:
                    raise ReleaseArchiveError(f"{key}: R2 full SHA-256 differs")
                assets[key] = {"name": info["name"], "size": info["size"],
                               "digest": "sha256:" + info["sha256"],
                               "source": "r2", "url": None}
            else:
                assets[key] = _github_asset(release, info["name"], r2_dir, trusted)
    else:
        for key in WINDOWS_KEYS:
            name = expected_name(tag, key)
            if build_identity and name not in trusted:
                raise ReleaseArchiveError(f"build identity missing asset: {name}")
            assets[key] = _github_asset(release, name, r2_dir, trusted)
    for key in MACOS_NAMES:
        name = expected_name(tag, key)
        if build_identity and name not in trusted:
            raise ReleaseArchiveError(f"build identity missing asset: {name}")
        assets[key] = _github_asset(release, name, r2_dir, trusted)
    with open(releases_json, encoding="utf-8") as stream:
        releases = json.load(stream)
    if isinstance(releases, dict):
        releases = releases.get("releases")
    if not isinstance(releases, list):
        raise ReleaseArchiveError("releases JSON must be a list or contain one")
    payload = {"tag": tag, "published": release.get("published_at"),
               "assets": assets, "releases": releases}
    _write_json(Path(output), payload)


def zip_tree(source: Path, output: Path) -> None:
    source = Path(source)
    if not source.is_dir():
        raise ReleaseArchiveError(f"missing bundle directory: {source}")
    output = Path(output)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname="OpenFocus/" + path.relative_to(source).as_posix())
    print(output)


def _cmd_stage(args) -> None:
    print(stage_windows(args.label, Path(args.zip), Path(args.setup),
                        Path(args.out), Path(args.recovery),
                        args.limit, args.part_size))


def _cmd_stage_gpu(args) -> None:
    print(stage_gpu_preview(args.label, Path(args.zip), Path(args.out),
                            Path(args.recovery), args.variant,
                            args.limit, args.part_size))


def _cmd_verify_stage(args) -> None:
    verify_stage(Path(args.manifest), Path(args.dir), args.limit, args.kind)
    print("staged assets are complete")


def _cmd_check_release(args) -> None:
    check_release_assets(Path(args.manifest), Path(args.asset_list),
                         args.tag, args.limit)
    print("GitHub release asset names match the split manifest")


def _cmd_list_downloads(args) -> None:
    for name in list_downloads(Path(args.manifest), args.limit):
        print(name)


def _cmd_reassemble(args) -> None:
    for path in reassemble(Path(args.manifest), Path(args.source_dir),
                           Path(args.output_dir), args.limit):
        print(path)


def _cmd_write_mirror(args) -> None:
    write_mirror_manifest(
        args.tag, Path(args.release_json), Path(args.r2_dir),
        Path(args.releases_json), Path(args.output),
        Path(args.windows_manifest) if args.windows_manifest else None,
        args.limit, Path(args.build_identity) if args.build_identity else None,
        args.head_sha, Path(args.trusted_metadata) if args.trusted_metadata else None,
        allow_beta=args.allow_beta)


def _cmd_beta_metadata_action(args) -> None:
    print(beta_metadata_action(args.tag, Path(args.previous)))


def _cmd_zip_tree(args) -> None:
    zip_tree(Path(args.source), Path(args.output))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("zip-tree")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_zip_tree)

    p = sub.add_parser("stage-windows")
    p.add_argument("--label", required=True)
    p.add_argument("--zip", required=True)
    p.add_argument("--setup", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--recovery", required=True)
    p.add_argument("--limit", type=int, default=GITHUB_ASSET_LIMIT)
    p.add_argument("--part-size", type=int, default=DEFAULT_PART_BYTES)
    p.set_defaults(func=_cmd_stage)

    p = sub.add_parser("stage-gpu")
    p.add_argument("--label", required=True)
    p.add_argument("--zip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--recovery", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--limit", type=int, default=GITHUB_ASSET_LIMIT)
    p.add_argument("--part-size", type=int, default=DEFAULT_PART_BYTES)
    p.set_defaults(func=_cmd_stage_gpu)

    p = sub.add_parser("verify-stage")
    p.add_argument("--manifest", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--limit", type=int, default=GITHUB_ASSET_LIMIT)
    p.add_argument("--kind", choices=("stable", GPU_KIND), default="stable")
    p.set_defaults(func=_cmd_verify_stage)

    p = sub.add_parser("check-release")
    p.add_argument("--manifest", required=True)
    p.add_argument("--asset-list", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=GITHUB_ASSET_LIMIT)
    p.set_defaults(func=_cmd_check_release)

    p = sub.add_parser("list-downloads")
    p.add_argument("--manifest", required=True)
    p.add_argument("--limit", type=int, default=GITHUB_ASSET_LIMIT)
    p.set_defaults(func=_cmd_list_downloads)

    p = sub.add_parser("reassemble")
    p.add_argument("--manifest", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=GITHUB_ASSET_LIMIT)
    p.set_defaults(func=_cmd_reassemble)

    p = sub.add_parser("write-mirror-manifest")
    p.add_argument("--tag", required=True)
    p.add_argument("--release-json", required=True)
    p.add_argument("--windows-manifest", default="")
    p.add_argument("--r2-dir", required=True)
    p.add_argument("--releases-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--build-identity", default="")
    p.add_argument("--head-sha", default="")
    p.add_argument("--trusted-metadata", default="")
    p.add_argument("--allow-beta", action="store_true")
    p.add_argument("--limit", type=int, default=GITHUB_ASSET_LIMIT)
    p.set_defaults(func=_cmd_write_mirror)

    p = sub.add_parser("beta-metadata-action")
    p.add_argument("--tag", required=True)
    p.add_argument("--previous", required=True)
    p.set_defaults(func=_cmd_beta_metadata_action)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ReleaseArchiveError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
