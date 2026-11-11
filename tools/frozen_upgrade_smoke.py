#!/usr/bin/env python3
"""Exercise a Windows frozen build overlaid on a published portable release."""

import argparse
import hashlib
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


def _extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract regular ZIP entries without allowing Windows path escapes."""
    seen = {}
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            name = entry.filename
            parts = PurePosixPath(name).parts
            if (not parts or name.startswith("/") or "\\" in name
                    or PureWindowsPath(name).drive
                    or any(part in (".", "..") or ":" in part
                           or part.rstrip(" .") != part
                           or PureWindowsPath(part).is_reserved() for part in parts)):
                raise ValueError(f"Unsafe ZIP entry: {name!r}")
            mode = entry.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Unsupported ZIP entry: {name!r}")
            is_dir = entry.is_dir()
            if kind == stat.S_IFDIR and not is_dir:
                raise ValueError(f"Invalid directory entry: {name!r}")
            key = tuple(part.casefold() for part in parts)
            if key in seen and (not is_dir or not seen[key]):
                raise ValueError(f"Duplicate ZIP entry: {name!r}")
            seen[key] = is_dir
            target = destination.joinpath(*parts)
            if is_dir:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def _digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def run(executable: Path, previous_zip: Path, runtime_name: str) -> None:
    if sys.platform != "win32":
        raise RuntimeError("Frozen upgrade smoke requires Windows and robocopy")
    executable = executable.resolve(strict=True)
    previous_zip = previous_zip.resolve(strict=True)
    if not executable.is_file() or executable.name.casefold() != "openfocus.exe":
        raise ValueError("--executable must point to the new OpenFocus.exe")
    if (not runtime_name or runtime_name.casefold() in (".", "..", "_internal")
            or PureWindowsPath(runtime_name).name != runtime_name
            or "/" in runtime_name or "\\" in runtime_name or ":" in runtime_name
            or runtime_name.rstrip(" .") != runtime_name):
        raise ValueError("--runtime-name must be the current runtime directory basename")
    runtime = executable.parent / runtime_name
    if not runtime.is_dir():
        raise FileNotFoundError(f"Current runtime directory not found: {runtime}")
    legacy_marker = executable.parent / "_internal" / "legacy-layout.marker"
    if not legacy_marker.is_file() or legacy_marker.stat().st_size == 0:
        raise RuntimeError(f"New build requires a nonempty legacy marker: {legacy_marker}")

    with tempfile.TemporaryDirectory(prefix="openfocus upgrade smoke ") as temp:
        root = Path(temp)
        extracted = root / "previous release"
        extracted.mkdir()
        _extract_archive(previous_zip, extracted)
        executables = [path for path in extracted.rglob("*")
                       if path.is_file() and path.name.casefold() == "openfocus.exe"]
        if len(executables) != 1:
            raise RuntimeError(f"Previous ZIP must contain one OpenFocus.exe; found {len(executables)}")
        install = executables[0].parent
        old_internal = install / "_internal"
        if not old_internal.is_dir():
            raise RuntimeError("Previous ZIP does not contain _internal beside OpenFocus.exe")
        if (install / runtime_name).exists():
            raise RuntimeError("Previous ZIP already contains the current runtime directory")

        # Add conflicting source data so the updater's exclusions are exercised.
        staged = root / "new release"
        shutil.copytree(executable.parent, staged)
        staged_data = staged / "OpenFocusData"
        staged_data.mkdir(exist_ok=True)
        (staged_data / "upgrade-smoke-sentinel.bin").write_bytes(b"new data must not overwrite old data")
        (staged / "OpenFocus.portable").write_bytes(b"new marker must not overwrite old marker")
        data = install / "OpenFocusData"
        data.mkdir(exist_ok=True)
        sentinel = data / "upgrade-smoke-sentinel.bin"
        old_data = b"representative previous portable data"
        sentinel.write_bytes(old_data)
        marker = install / "OpenFocus.portable"
        old_marker = b"representative previous portable marker"
        marker.write_bytes(old_marker)

        # Pick a real old runtime file that the new compatibility directory does
        # not supply, so retaining stale runtime contents is part of the test.
        old_file = next((path for path in old_internal.rglob("*")
                         if path.is_file() and not (staged / "_internal" /
                                                   path.relative_to(old_internal)).exists()), None)
        if old_file is None:
            raise RuntimeError("Previous _internal has no old-only file to verify")
        old_digest = _digest(old_file)
        result = subprocess.run([
            "robocopy", str(staged), str(install), "/E", "/IS", "/IT",
            "/R:2", "/W:1", "/XF", "OpenFocus.portable", "/XD", "OpenFocusData",
        ], check=False)
        if result.returncode >= 8 or result.returncode < 0:
            raise RuntimeError(f"Upgrade robocopy failed with exit code {result.returncode}")
        overlaid_executable = install / "OpenFocus.exe"
        overlaid_runtime = install / runtime_name
        if _digest(overlaid_executable) != _digest(executable):
            raise RuntimeError("Overlay did not install the new executable")
        if not overlaid_runtime.is_dir():
            raise RuntimeError("Overlay did not install the current runtime")
        if not old_file.is_file() or _digest(old_file) != old_digest:
            raise RuntimeError("Overlay did not retain the old _internal contents")
        if sentinel.read_bytes() != old_data or marker.read_bytes() != old_marker:
            raise RuntimeError("Overlay changed portable user data or the portable marker")

        # The CLI smoke owns its temporary portable profile and requires a
        # clean directory. Preserve the overlaid fixtures while it runs.
        saved_data = root / "saved portable data"
        saved_marker = root / "saved portable marker"
        data.rename(saved_data)
        marker.rename(saved_marker)
        smoke = Path(__file__).resolve().with_name("frozen_cli_smoke.py")
        try:
            subprocess.run([
                sys.executable, str(smoke), "--executable", str(overlaid_executable),
                "--resources", str(overlaid_runtime),
            ], check=True)
        finally:
            saved_data.rename(data)
            saved_marker.rename(marker)
        if sentinel.read_bytes() != old_data or marker.read_bytes() != old_marker:
            raise RuntimeError("Smoke run changed portable user data or the portable marker")
        print("Frozen upgrade smoke passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--previous-zip", required=True, type=Path)
    parser.add_argument("--runtime-name", required=True)
    args = parser.parse_args()
    run(args.executable, args.previous_zip, args.runtime_name)


if __name__ == "__main__":
    main()
