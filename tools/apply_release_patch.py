"""Install the published OpenFocus patch for an existing official 2.0.0 bundle."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import release_delta, self_update, updater


def _running():
    if sys.platform == "win32":
        command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                   "if (Get-Process -Name OpenFocus -ErrorAction SilentlyContinue) { exit 0 }; exit 1"]
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                timeout=15, check=False)
        if result.returncode not in (0, 1):
            raise ValueError("Could not check whether OpenFocus is running")
        return result.returncode == 0
    try:
        result = subprocess.run(["pgrep", "-ix", "OpenFocus"], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, timeout=15, check=False)
    except FileNotFoundError:
        proc = Path("/proc")
        if not proc.is_dir():
            raise ValueError("Could not check whether OpenFocus is running")
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if (entry / "comm").read_text(encoding="utf-8", errors="ignore").strip().lower() == "openfocus":
                    return True
            except OSError:
                continue
        return False
    if result.returncode not in (0, 1):
        raise ValueError("Could not check whether OpenFocus is running")
    return result.returncode == 0


def install(patch, installed, checksum, *, restart=True):
    platform = release_delta.platform_key()
    if not platform or not self_update.verify_update_candidate(str(installed), platform):
        raise ValueError("Select the folder or .app bundle of your official OpenFocus installation")
    if _running():
        raise ValueError("Quit OpenFocus before installing this patch")
    if updater.verify_file_digest(patch, checksum) is not True:
        raise ValueError("The patch is incomplete or its SHA-256 does not match the release")
    manifest = release_delta.read_manifest(patch)
    base, tag = manifest["from"], manifest["to"]
    version = tag.split("-preview.", 1)[0]
    if (base != release_delta.PATCH_REQUIRED_FROM or manifest["platform"] != platform
            or not (updater.is_newer(version, base) or version == base != tag)):
        raise ValueError("This patch does not match the published platform and base version")
    staging = Path(tempfile.mkdtemp(prefix="OpenFocus_release_patch_")).resolve()
    candidate = staging / ("OpenFocus.app" if platform == "macos-arm64" else "OpenFocus")
    process = None
    try:
        print(f"Preparing OpenFocus {tag}…", flush=True)
        release_delta.stage_delta(patch, installed, candidate, base_tag=base, target_tag=tag, platform=platform)
        if not self_update.verify_update_candidate(str(candidate), platform, patch=True):
            raise ValueError("Could not verify the reconstructed release bundle")
        if _running():
            raise ValueError("Quit OpenFocus before installing this patch")
        print("Replacing the existing application…", flush=True)
        process = self_update.handoff_update(str(installed), str(candidate), tag,
                                              staging_root=str(staging), restart=restart)
        if process.wait() != 0:
            result = Path(self_update.result_path(tag))
            detail = ""
            if result.is_file():
                detail = result.read_text(encoding="utf-8", errors="replace").strip()
                try:
                    result.unlink()
                except OSError:
                    pass
            raise ValueError(detail or "The patch could not replace the application")
    finally:
        if process is None or process.poll() is not None:
            shutil.rmtree(staging, ignore_errors=True)
    print(f"Installed OpenFocus {tag}.")


def main():
    frozen = self_update.is_frozen()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path)
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--no-restart", action="store_true")
    if not frozen:
        parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    interactive = args.install_dir is None
    try:
        if frozen:
            path = Path(sys._MEIPASS) / "OpenFocus-Patch.sha256"
            checksum = path.read_text(encoding="ascii").strip()
            patch_dir = Path(sys.executable).parent
        else:
            checksum = args.sha256
            patch_dir = Path.cwd()
        patch = args.patch
        if patch is None:
            patches = sorted(patch_dir.glob("OpenFocus-*.ofdelta.zip"))
            if len(patches) != 1:
                raise ValueError("Place the helper beside its matching .ofdelta.zip or specify --patch")
            patch = patches[0]
        installed = args.install_dir
        if installed is None:
            value = input("Path to your existing OpenFocus folder or OpenFocus.app: ").strip().strip('"').strip("'")
            if not value:
                raise ValueError("No installation selected")
            installed = Path(value).expanduser()
        install(patch.resolve(strict=True), installed.resolve(strict=True), checksum,
                restart=not args.no_restart)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError, EOFError, KeyboardInterrupt) as exc:
        print(f"OpenFocus patch failed: {exc}", file=sys.stderr, flush=True)
        if interactive:
            try:
                input("Press Enter to close.")
            except EOFError:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
