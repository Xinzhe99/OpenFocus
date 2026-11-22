"""Apply a release delta to a new tree using only verified program files.

A delta ZIP lists the complete target tree. Each regular file either references
an identical file in the original bundle or has a payload in the ZIP. A
staging caller verifies the ZIP's release digest before using this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform as host_platform
import posixpath
import re
import stat
import sys
import unicodedata
import zipfile

from utils.updater import DownloadCancelled

PATCH_REQUIRED_FROM = "v2.0.0"
EXECUTABLES = {
    "windows-x64": "OpenFocus.exe",
    "macos-arm64": "Contents/MacOS/OpenFocus",
    "linux-x86_64": "OpenFocus",
}
_TAG = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-beta\.[1-9][0-9]*)?")
_PREVIEW_TAG = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+-preview\.[A-Za-z0-9]+")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_MANIFEST = "manifest.json"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024


def platform_key() -> str:
    arch = host_platform.machine().lower()
    if sys.platform == "win32" and arch in ("amd64", "x86_64"):
        return "windows-x64"
    if sys.platform == "darwin" and arch in ("arm64", "aarch64"):
        return "macos-arm64"
    if sys.platform == "linux" and arch in ("amd64", "x86_64"):
        return "linux-x86_64"
    return ""


def delta_name(target_tag: str, base_tag: str, platform: str) -> str:
    return f"OpenFocus-{target_tag}-{platform}-from-{base_tag}.ofdelta.zip"


def legacy_patch_name(target_tag: str, platform: str) -> str:
    return f"OpenFocus-{target_tag}-{platform}-patch-from-{PATCH_REQUIRED_FROM}.zip"


def local_entry(name: str) -> bool:
    """Paths owned by the user, the installer, or the update transaction."""
    key = name.casefold()
    return (key in {"openfocus.portable", "openfocusdata", ".staging_complete"}
            or bool(re.fullmatch(r"unins[0-9]{3,}\.(?:exe|dat|msg)", key)))


def _valid_relative(path: str) -> bool:
    if not isinstance(path, str) or not path or len(path) > 4096:
        return False
    if any(ord(ch) < 32 or ch in '\\:<>"|?*' for ch in path):
        return False
    parts = path.split("/")
    return not local_entry(parts[0]) and all(
        part and part.rstrip(" .") == part
        and not re.fullmatch(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part, re.I)
        for part in parts
    )


def _valid_link(path: str, target: str) -> bool:
    if (not isinstance(target, str) or not target or len(target) > 4096
            or target.startswith("/") or any(ord(ch) < 32 or ch in "\\:" for ch in target)):
        return False
    return _valid_relative(posixpath.normpath(posixpath.join(posixpath.dirname(path), target)))


def _valid_tag(tag, *, preview=False):
    return (isinstance(tag, str) and len(tag) <= 32
            and (_TAG.fullmatch(tag) or preview and _PREVIEW_TAG.fullmatch(tag)))


def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise DownloadCancelled("patch preparation cancelled")


def _transfer(source, destination=None, *, size=None, should_cancel=None):
    digest = hashlib.sha256()
    seen = 0
    while True:
        _check_cancel(should_cancel)
        chunk = source.read(_CHUNK_BYTES)
        if not chunk:
            break
        seen += len(chunk)
        if size is not None and seen > size:
            raise ValueError("Patch member exceeds its declared size")
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
    if size is not None and seen != size:
        raise ValueError("Patch member is truncated")
    return digest.hexdigest()


def _reparse(stat_result):
    return bool(getattr(stat_result, "st_file_attributes", 0)
                & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _source_file(root, relative):
    """Open a regular installed file without following bundle links or junctions."""
    path = root
    parts = relative.split("/")
    for part in parts[:-1]:
        path /= part
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or _reparse(info):
            raise ValueError(f"Invalid installed directory: {relative}")
    path /= parts[-1]
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or _reparse(info):
        raise ValueError(f"Invalid installed file: {relative}")
    return os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0)
                             | getattr(os, "O_NOFOLLOW", 0)), "rb")


def _payload(index):
    return f"files/{index}"


def _read_manifest(archive):
    members = {info.filename: info for info in archive.infolist()}
    manifest_info = members.pop(_MANIFEST, None)
    if (len(members) + 1 != len(archive.infolist()) or manifest_info is None
            or manifest_info.file_size > _MAX_MANIFEST_BYTES):
        raise ValueError("Invalid delta ZIP directory")
    manifest = json.loads(archive.read(manifest_info))
    if not isinstance(manifest, dict):
        raise ValueError("Invalid delta manifest")
    base, target, platform = (manifest.get(key) for key in ("from", "to", "platform"))
    base_hash = manifest.get("base_entry_sha256")
    entries = manifest.get("entries")
    if (manifest.get("schema") != 1 or not _valid_tag(base) or not _valid_tag(target, preview=True)
            or base == target or not isinstance(platform, str) or platform not in EXECUTABLES
            or not isinstance(base_hash, str) or not _SHA256.fullmatch(base_hash)
            or not isinstance(entries, list) or not entries):
        raise ValueError("Invalid delta identity")
    paths = {}
    folded = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("Invalid delta entry")
        path, kind = entry.get("path"), entry.get("type")
        if not _valid_relative(path) or path in paths:
            raise ValueError(f"Invalid or duplicate delta path: {path!r}")
        if platform != "linux-x86_64":
            key = unicodedata.normalize("NFD", path).casefold()
            if key in folded:
                raise ValueError(f"Conflicting delta path: {path!r}")
            folded.add(key)
        mode = entry.get("mode")
        if type(mode) is not int or not 0 <= mode <= 0o777:
            raise ValueError(f"Invalid delta permissions: {path}")
        if kind == "file":
            size, digest = entry.get("size"), entry.get("sha256")
            if type(size) is not int or size < 0 or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise ValueError(f"Invalid delta file: {path}")
            if "source" in entry:
                if not _valid_relative(entry["source"]):
                    raise ValueError(f"Invalid delta source: {path}")
            else:
                info = members.pop(_payload(index), None)
                if info is None or info.file_size != size:
                    raise ValueError(f"Missing or incompatible delta payload: {path}")
        elif kind == "link":
            if platform == "windows-x64" or not _valid_link(path, entry.get("target")):
                raise ValueError(f"Invalid delta symlink: {path}")
        elif kind != "dir":
            raise ValueError(f"Unsupported delta member: {path}")
        paths[path] = kind
    for path in paths:
        parent = posixpath.dirname(path)
        if parent and paths.get(parent) != "dir":
            raise ValueError(f"Invalid delta parent directory: {path}")
    if members or paths.get(EXECUTABLES[platform]) != "file":
        raise ValueError("Delta payload and target manifest differ")
    return manifest


def read_manifest(patch_path):
    """Read and validate the release identity and the ZIP member inventory."""
    try:
        with zipfile.ZipFile(patch_path) as archive:
            return _read_manifest(archive)
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid delta ZIP") from exc


def stage_delta(patch_path, installed_root, candidate_root, *, base_tag, target_tag,
                platform, should_cancel=None, on_progress=None):
    """Stage the verified target. The caller owns cleanup and the final swap."""
    installed_root = Path(installed_root).resolve(strict=True)
    candidate_root = Path(candidate_root)
    if candidate_root.is_symlink() or candidate_root.resolve().is_relative_to(installed_root):
        raise ValueError("Patch staging must be outside the installation")
    if candidate_root.exists() and (not candidate_root.is_dir() or any(candidate_root.iterdir())):
        raise ValueError("Patch staging directory must be empty")
    try:
        with zipfile.ZipFile(patch_path) as archive:
            manifest = _read_manifest(archive)
            if (manifest["from"], manifest["to"], manifest["platform"]) != (base_tag, target_tag, platform):
                raise ValueError("Patch does not match the installation and target release")
            with _source_file(installed_root, EXECUTABLES[platform]) as entrypoint:
                if _transfer(entrypoint, should_cancel=should_cancel) != manifest["base_entry_sha256"]:
                    raise ValueError("Installed executable does not match the patch base")
            entries = manifest["entries"]
            candidate_root.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                _check_cancel(should_cancel)
                if entry["type"] == "dir":
                    (candidate_root / entry["path"]).mkdir(parents=True, exist_ok=True)
            for index, entry in enumerate(entries):
                _check_cancel(should_cancel)
                if entry["type"] == "file":
                    source = (_source_file(installed_root, entry["source"]) if "source" in entry
                              else archive.open(_payload(index)))
                    filename = candidate_root / entry["path"]
                    with source, filename.open("xb") as output:
                        digest = _transfer(source, output, size=entry["size"], should_cancel=should_cancel)
                    if digest != entry["sha256"]:
                        raise ValueError(f"Program file checksum differs: {entry['path']}")
                    if os.name != "nt":
                        filename.chmod(entry["mode"])
                if on_progress is not None:
                    on_progress(index + 1, len(entries))
            root = candidate_root.resolve(strict=True)
            links = [entry for entry in entries if entry["type"] == "link"]
            for entry in links:
                _check_cancel(should_cancel)
                (root / entry["path"]).symlink_to(entry["target"])
            for entry in links:
                _check_cancel(should_cancel)
                link = root / entry["path"]
                if not link.resolve(strict=True).is_relative_to(root):
                    raise ValueError(f"Patch symlink escapes the target: {entry['path']}")
            if os.name != "nt":
                dirs = (entry for entry in entries if entry["type"] == "dir")
                for entry in sorted(dirs, key=lambda item: item["path"].count("/"), reverse=True):
                    (root / entry["path"]).chmod(entry["mode"])
            _check_cancel(should_cancel)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError("Invalid delta payload") from exc
