"""Self-update: download, stage and apply release builds in place.

Designed for the PyInstaller onedir layout (app folder with the exe plus
_internal). Flow:

1. download the release portable zip (progress callback)
2. extract into a staging folder next to the installation
3. verify the staged executable exists
4. write a small apply script that - after the app quits - mirrors the
   staged folder over the installation and restarts the app
5. quit the app; the detached script finishes the swap

Elevation: when the install directory is not writable (Program Files),
the apply script relaunches its robocopy step via PowerShell -Verb RunAs
(one UAC prompt).
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from typing import Callable, List, Optional, Tuple

PORTABLE_MARKER = "OpenFocus.portable"


def app_install_dir() -> str:
    """Directory containing the running executable (frozen) or the project root (source run)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def mac_app_bundle() -> Optional[str]:
    """Path of the running .app bundle on macOS, None elsewhere/ unbundled."""
    if sys.platform != "darwin":
        return None
    exe = os.path.abspath(sys.executable)
    idx = exe.find(".app")
    if idx == -1:
        return None
    return exe[: idx + len(".app")]


def staging_dir_for(app_dir: str, tag: str) -> str:
    return app_dir.rstrip("/\\") + f"_update_{tag}"


def extract_zip(zip_path: str, dest_root: str,
                on_progress: Optional[Callable[[int, int], None]] = None) -> None:
    """Extract zip into dest_root with per-entry progress (done, total)."""
    with zipfile.ZipFile(zip_path) as z:
        entries = z.infolist()
        total = len(entries)
        for i, entry in enumerate(entries):
            z.extract(entry, dest_root)
            if on_progress is not None:
                try:
                    on_progress(i + 1, total)
                except Exception:
                    pass


def verify_staged_app(staging_root: str) -> bool:
    """True when the staged folder contains a runnable app entry."""
    exe_name = "OpenFocus.exe" if sys.platform == "win32" else "OpenFocus"
    candidates = [
        os.path.join(staging_root, "OpenFocus", exe_name),
        os.path.join(staging_root, "OpenFocus.app", "Contents", "MacOS", "OpenFocus"),
        os.path.join(staging_root, exe_name),
    ]
    return any(os.path.isfile(c) for c in candidates)


def write_apply_script_windows(app_dir: str, staging_app_dir: str, tag: str) -> str:
    """Write a detached batch script: wait -> mirror staging over install ->
    cleanup -> restart. Falls back to an elevated robocopy when the install
    directory is not writable (Program Files)."""
    app_dir = os.path.abspath(app_dir)
    staging_app_dir = os.path.abspath(staging_app_dir)
    script_path = os.path.join(os.path.dirname(staging_app_dir), f"apply_update_{tag}.bat")
    exe = os.path.join(app_dir, "OpenFocus.exe")
    content = (
        "@echo off\r\n"
        "rem Wait for OpenFocus to exit\r\n"
        "timeout /t 2 /nobreak >nul\r\n"
        f'robocopy "{staging_app_dir}" "{app_dir}" /E /IS /IT /NFL /NDL /NJH /NJS /NP\r\n'
        "if errorlevel 8 (\r\n"
        f"  echo Elevated copy required\r\n"
        f'  powershell -NoProfile -Command "Start-Process -FilePath \'%~f0\' -ArgumentList \'/elevated\' -Verb RunAs -Wait"\r\n'
        ")\r\n"
        "if \"%1\"==\"/elevated\" (\r\n"
        f'  robocopy "{staging_app_dir}" "{app_dir}" /E /IS /IT /NFL /NDL /NJH /NJS /NP\r\n'
        ")\r\n"
        f'rmdir /S /Q "{staging_app_dir}"\r\n'
        f'start "" "{exe}"\r\n'
        'del "%~f0"\r\n'
    )
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    return script_path


def write_apply_script_macos(app_bundle: str, staged_bundle: str, tag: str) -> str:
    app_bundle = os.path.abspath(app_bundle)
    staged_bundle = os.path.abspath(staged_bundle)
    parent = os.path.dirname(app_bundle)
    script_path = os.path.join(os.path.dirname(staged_bundle), f"apply_update_{tag}.sh")
    content = (
        "#!/bin/bash\n"
        "sleep 2\n"
        f'rm -rf "{app_bundle}"\n'
        f'ditto "{staged_bundle}" "{app_bundle}"\n'
        f'open "{app_bundle}"\n'
        f'rm -rf "{staged_bundle}"\n'
        'rm -f "$0"\n'
    )
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(script_path, 0o755)
    return script_path


def launch_detached_windows(script_path: str) -> None:
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(["cmd", "/c", script_path], close_fds=True,
                     creationflags=flags, cwd=os.path.dirname(script_path))


def launch_detached_unix(script_path: str) -> None:
    subprocess.Popen(["/bin/bash", script_path], start_new_session=True)


def apply_update_and_restart(app_dir: str, staged_app_dir: str, tag: str) -> None:
    """Write the apply script, launch it detached and return (caller quits next)."""
    if sys.platform == "win32":
        script = write_apply_script_windows(app_dir, staged_app_dir, tag)
        launch_detached_windows(script)
    elif sys.platform == "darwin":
        script = write_apply_script_macos(app_dir, staged_app_dir, tag)
        launch_detached_unix(script)
    else:
        raise OSError("self-update is not supported on this platform")
