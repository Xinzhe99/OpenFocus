"""Build checked, file-level deltas between official desktop release bundles."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import release_delta as delta
from utils.self_update import extract_linux_tar, extract_zip

_ASSET_LIMIT = 2_147_483_648
_BRIDGE_NAMES = {"windows-x64": "OpenFocus-Patch.exe",
                 "macos-arm64": "OpenFocus-Patch", "linux-x86_64": "OpenFocus-Patch"}
_ROOT_NAMES = {"windows-x64": "OpenFocus", "macos-arm64": "OpenFocus.app", "linux-x86_64": "OpenFocus"}


def _inventory(root):
    entries = {}

    def visit(directory, parents=()):
        with os.scandir(directory) as children:
            for child in sorted(children, key=lambda entry: entry.name):
                if not parents and delta.local_entry(child.name):
                    continue
                parts = (*parents, child.name)
                relative = "/".join(parts)
                if not delta._valid_relative(relative):
                    raise ValueError(f"Unsupported bundle path: {relative}")
                info = child.stat(follow_symlinks=False)
                entry = {"path": relative, "mode": info.st_mode & 0o777}
                entries[relative] = entry
                if stat.S_ISLNK(info.st_mode):
                    entry.update(type="link", target=os.readlink(child.path))
                elif stat.S_ISDIR(info.st_mode) and not delta._reparse(info):
                    entry["type"] = "dir"
                    visit(child.path, parts)
                elif stat.S_ISREG(info.st_mode) and not delta._reparse(info):
                    with open(child.path, "rb") as source:
                        checksum = delta._transfer(source, size=info.st_size)
                    entry.update(type="file", size=info.st_size, sha256=checksum)
                else:
                    raise ValueError(f"Unsupported bundle entry: {relative}")

    visit(root)
    return entries


def build_delta(base_root, target_root, patch_path, *, base_tag, target_tag, platform):
    """Reuse identical files by contents even when their paths change."""
    base = _inventory(base_root)
    target = _inventory(target_root)
    entrypoint = delta.EXECUTABLES[platform]
    if base.get(entrypoint, {}).get("type") != "file" or target.get(entrypoint, {}).get("type") != "file":
        raise ValueError("Both bundles must contain the official executable")
    sources = {}
    for path, entry in base.items():
        if entry["type"] == "file":
            sources.setdefault((entry["sha256"], entry["size"]), path)
    entries = list(target.values())
    with zipfile.ZipFile(patch_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, entry in enumerate(entries):
            if entry["type"] != "file":
                continue
            path = entry["path"]
            key = (entry["sha256"], entry["size"])
            existing = base.get(path, {})
            source = path if key == (existing.get("sha256"), existing.get("size")) else sources.get(key)
            if source is not None:
                entry["source"] = source
            else:
                archive.write(target_root / path, delta._payload(index))
        archive.writestr("manifest.json", json.dumps({
            "schema": 1, "from": base_tag, "to": target_tag, "platform": platform,
            "base_entry_sha256": base[entrypoint]["sha256"], "entries": entries,
        }, separators=(",", ":")))


def _extract(path, root, platform):
    if platform == "linux-x86_64":
        extract_linux_tar(path, root)
    else:
        extract_zip(path, root)
    return root / _ROOT_NAMES[platform]


def build_archive_delta(base, target, output, *, base_tag, target_tag, platform, verify_tree=None):
    if not delta._valid_tag(base_tag) or not delta._valid_tag(target_tag, preview=True) or base_tag == target_tag:
        raise ValueError("Invalid delta release versions")
    output.mkdir(parents=True, exist_ok=True)
    final = output / delta.delta_name(target_tag, base_tag, platform)
    if final.exists() or verify_tree is not None and verify_tree.exists():
        raise ValueError("Update delta output already exists")
    if delta.platform_key() != platform:
        raise ValueError("Build deltas on the target desktop platform to preserve bundle metadata")
    with tempfile.TemporaryDirectory(prefix="openfocus-delta-", dir=output) as path:
        temp = Path(path)
        base_root = _extract(base, temp / "base", platform)
        target_root = _extract(target, temp / "target", platform)
        patch = temp / "candidate.zip"
        build_delta(base_root, target_root, patch, base_tag=base_tag, target_tag=target_tag, platform=platform)
        delta_size = patch.stat().st_size
        full_size = target.stat().st_size
        if delta_size >= min(full_size, _ASSET_LIMIT):
            print(f"Skipping {final.name}: {delta_size} bytes for a {full_size}-byte full package")
            return None
        reconstructed = temp / "reconstructed"
        delta.stage_delta(patch, base_root, reconstructed, base_tag=base_tag,
                          target_tag=target_tag, platform=platform)
        if verify_tree is not None:
            shutil.move(str(reconstructed), verify_tree)
        patch.replace(final)
    print(f"{final.name}: {delta_size:,} of {full_size:,} bytes ({delta_size / full_size:.1%})")
    return final


def build_bridge(patch, helper, output, *, limit=_ASSET_LIMIT):
    manifest = delta.read_manifest(patch)
    target, platform = manifest["to"], manifest["platform"]
    if manifest["from"] != delta.PATCH_REQUIRED_FROM or patch.name != delta.delta_name(target, manifest["from"], platform):
        raise ValueError("Only the published bootstrap base is eligible for a manual patch")
    helper_name = _BRIDGE_NAMES[platform]
    if helper.name != helper_name or not helper.is_file():
        raise ValueError(f"Missing platform patch helper: {helper_name}")
    output.mkdir(parents=True, exist_ok=True)
    final = output / delta.legacy_patch_name(target, platform)
    instructions = f"""OpenFocus {target} patch for {delta.PATCH_REQUIRED_FROM} ({platform})

Close every OpenFocus window and quit any background OpenFocus processes.
Extract this ZIP to a separate folder. Keep {helper_name} and the
.ofdelta.zip file together. Run the helper and enter the full path of your
existing OpenFocus folder (Windows or Linux) or OpenFocus.app (macOS).

Windows: double-click {helper_name}, or run it in Command Prompt.
macOS/Linux: use Terminal in the extracted folder:
    chmod +x ./OpenFocus-Patch
    ./OpenFocus-Patch

Keep the existing application where it is until the patch finishes.
If files in the original installation differ from the official release,
use the full installer or archive on the same release page.
"""
    with tempfile.TemporaryDirectory(prefix="openfocus-patch-", dir=output) as work:
        path = Path(work) / "bridge.zip"
        with zipfile.ZipFile(path, "x") as archive:
            archive.write(helper, helper_name, compress_type=zipfile.ZIP_STORED)
            archive.write(patch, patch.name, compress_type=zipfile.ZIP_STORED)
            archive.writestr("README.txt", instructions.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)
        if path.stat().st_size >= min(limit, _ASSET_LIMIT):
            print(f"Skipping {final.name}: the helper and patch together reach {path.stat().st_size:,} bytes")
            return
        if final.exists():
            raise ValueError(f"Patch output already exists: {final}")
        path.replace(final)
    print(f"{final.name}: {final.stat().st_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--base", type=Path, required=True)
    build.add_argument("--target", type=Path, required=True)
    build.add_argument("--from-tag", required=True)
    build.add_argument("--to-tag", required=True)
    build.add_argument("--platform", choices=tuple(delta.EXECUTABLES), required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--verify-tree", type=Path)
    bridge = commands.add_parser("bridge")
    bridge.add_argument("--patch", type=Path, required=True)
    bridge.add_argument("--helper", type=Path, required=True)
    bridge.add_argument("--output", type=Path, required=True)
    bridge.add_argument("--limit", type=int, default=_ASSET_LIMIT)
    args = parser.parse_args()
    if args.command == "build":
        build_archive_delta(args.base, args.target, args.output, base_tag=args.from_tag,
                            target_tag=args.to_tag, platform=args.platform, verify_tree=args.verify_tree)
    else:
        build_bridge(args.patch, args.helper, args.output, limit=args.limit)


if __name__ == "__main__":
    main()
