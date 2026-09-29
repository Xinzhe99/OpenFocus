"""Self-update: download, stage and apply release builds in place.

Designed for the PyInstaller onedir layout (app folder with the exe plus
_internal). Flow:

1. download the release portable zip (progress callback)
2. extract into a staging folder under the user-writable temp dir
3. verify the staged executable exists
4. write a small apply script that - after the app quits - mirrors the
   staged folder over the installation and restarts the app
5. quit the app; the detached script finishes the swap

Staging deliberately lives in the temp dir, never next to the installation:
an install under Program Files is not writable by the running (unelevated)
process, so staging beside it would fail before the swap could even be
offered. Elevation is handled by the apply script relaunching its robocopy
step via PowerShell -Verb RunAs (one UAC prompt).

The release tag arrives from the GitHub API, so it is treated as untrusted
input: it is sanitised before reaching any filesystem path or script text.
"""
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from typing import Callable, Optional

PORTABLE_MARKER = "OpenFocus.portable"

_SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9._-]")
STAGING_PREFIX = "OpenFocus_update_"
APP_NAME = "OpenFocus"


def app_install_dir() -> str:
    """Directory containing the running executable (frozen) or the project root (source run)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def sanitize_tag(tag: str) -> str:
    """Strip a remote-supplied tag down to path-safe characters."""
    cleaned = _SAFE_TAG_RE.sub("", str(tag or "").strip())
    cleaned = cleaned.strip(".") or "latest"
    return cleaned[:32]


def mac_app_bundle() -> Optional[str]:
    """Path of the running .app bundle on macOS, None elsewhere/unbundled."""
    if sys.platform != "darwin":
        return None
    # Match a real path component ending in .app - a plain substring search
    # would happily cut through directories merely named ".app..."
    parts = os.path.abspath(sys.executable).split(os.sep)
    for i, part in enumerate(parts):
        if part.endswith(".app"):
            return os.sep.join(parts[:i + 1]) or os.sep
    return None


def staging_root_for(tag: str) -> str:
    """User-writable staging directory for this update."""
    return os.path.join(tempfile.gettempdir(), STAGING_PREFIX + sanitize_tag(tag))


def _has_executable(candidate: str) -> bool:
    exe = APP_NAME + (".exe" if sys.platform == "win32" else "")
    return os.path.isfile(os.path.join(candidate, exe))


def find_staged_app_dir(staging_root: str) -> str:
    """Directory inside staging_root holding the unpacked onedir app, '' if absent."""
    for candidate in (os.path.join(staging_root, APP_NAME), staging_root):
        if _has_executable(candidate):
            return candidate
    try:
        entries = sorted(os.listdir(staging_root))
    except OSError:
        return ""
    for name in entries:
        path = os.path.join(staging_root, name)
        if os.path.isdir(path) and _has_executable(path):
            return path
    return ""


def find_staged_bundle(staging_root: str) -> str:
    """.app bundle inside staging_root, '' if absent."""
    for candidate in (os.path.join(staging_root, APP_NAME + ".app"),
                      os.path.join(staging_root, APP_NAME, APP_NAME + ".app")):
        if os.path.isdir(os.path.join(candidate, "Contents", "MacOS")):
            return candidate
    try:
        entries = sorted(os.listdir(staging_root))
    except OSError:
        return ""
    for name in entries:
        path = os.path.join(staging_root, name)
        if name.endswith(".app") and os.path.isdir(
                os.path.join(path, "Contents", "MacOS")):
            return path
    return ""


def verify_staged_app(staging_root: str) -> bool:
    """True when the staged folder contains a runnable app entry."""
    if sys.platform == "darwin" and find_staged_bundle(staging_root):
        return True
    return bool(find_staged_app_dir(staging_root))


def extract_zip(zip_path: str, dest_root: str,
                on_progress: Optional[Callable[[int, int], None]] = None) -> None:
    """Extract zip into dest_root with per-entry progress (done, total)."""
    os.makedirs(dest_root, exist_ok=True)
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


def _bat_path(path: str) -> str:
    """Quote a path for cmd, refusing anything that could escape the quotes."""
    abspath = os.path.abspath(path)
    if '"' in abspath or "\n" in abspath or "\r" in abspath:
        raise ValueError("Unsafe path for the update script")
    return f'"{abspath}"'


def write_apply_script_windows(app_dir: str, staging_app_dir: str, tag: str,
                               staging_root: str = "", zip_path: str = "") -> str:
    """Write a detached batch script: wait for the app to exit -> mirror the
    staged folder over the installation -> clean up -> restart. Retries the
    copy elevated when the install directory is not writable.

    The elevated retry re-invokes this script with '/elevated'; that branch
    only copies, so a failing unelevated first pass cannot loop into repeated
    UAC prompts.
    """
    app_dir = os.path.abspath(app_dir)
    staging_app_dir = os.path.abspath(staging_app_dir)
    staging_root = os.path.abspath(staging_root or os.path.dirname(staging_app_dir))
    script_dir = os.path.dirname(staging_root)
    script_path = os.path.join(
        script_dir, f"apply_update_{sanitize_tag(tag)}.bat")
    exe = os.path.join(app_dir, APP_NAME + ".exe")
    copy_cmd = (
        f"robocopy {_bat_path(staging_app_dir)} {_bat_path(app_dir)}"
        " /E /IS /IT /NFL /NDL /NJH /NJS /NP"
    )
    content = (
        "@echo off\r\n"
        "setlocal enableextensions\r\n"
        'if "%~1"=="/elevated" goto elevatedcopy\r\n'
        "call :waitforexit\r\n"
        f"{copy_cmd}\r\n"
        "if errorlevel 8 (\r\n"
        '  powershell -NoProfile -Command "Start-Process -FilePath \'%~f0\''
        ' -ArgumentList \'/elevated\' -Verb RunAs -Wait"\r\n'
        ")\r\n"
        f'rmdir /S /Q {_bat_path(staging_root)} >nul 2>&1\r\n'
        f'if exist {_bat_path(zip_path)} del /Q {_bat_path(zip_path)} >nul 2>&1'
        "\r\n"
        f'start "" {_bat_path(exe)}\r\n'
        'del "%~f0"\r\n'
        "exit /b\r\n"
        ":elevatedcopy\r\n"
        "call :waitforexit\r\n"
        f"{copy_cmd}\r\n"
        "exit /b\r\n"
        ":waitforexit\r\n"
        "set /a TRIES=0\r\n"
        ":waitloop\r\n"
        "timeout /t 1 /nobreak >nul 2>&1\r\n"
        f'tasklist /FI "IMAGENAME eq {APP_NAME}.exe" 2>NUL'
        f' | find /I "{APP_NAME}.exe" >NUL\r\n'
        "if errorlevel 1 exit /b 0\r\n"
        "set /a TRIES+=1\r\n"
        "if %TRIES% LSS 20 goto waitloop\r\n"
        "exit /b 0\r\n"
    )
    with open(script_path, "w", encoding="ascii") as f:
        f.write(content)
    return script_path


def write_apply_script_macos(app_bundle: str, staged_bundle: str, tag: str,
                             staging_root: str = "", zip_path: str = "",
                             wait_pid: int = 0) -> str:
    app_bundle = os.path.abspath(app_bundle)
    staged_bundle = os.path.abspath(staged_bundle)
    staging_root = os.path.abspath(staging_root or os.path.dirname(staged_bundle))
    if not app_bundle.endswith(".app") or not os.path.isdir(app_bundle):
        raise ValueError("Refusing to replace a non-existent .app bundle")
    if not os.path.isdir(os.path.join(staged_bundle, "Contents", "MacOS")):
        raise ValueError("Staged bundle is incomplete")
    pid = int(wait_pid) if str(wait_pid).lstrip("-").isdigit() else 0
    script_path = os.path.join(
        os.path.dirname(staging_root), f"apply_update_{sanitize_tag(tag)}.sh")
    content = (
        "#!/bin/bash\n"
        "for _ in $(seq 1 20); do\n"
        f"  [ {pid} -gt 0 ] && kill -0 {pid} 2>/dev/null || break\n"
        "  sleep 1\n"
        "done\n"
        f'if ditto "{staged_bundle}" "{app_bundle}.new"; then\n'
        '  rm -rf "' + app_bundle + '.old"\n'
        f'  mv "{app_bundle}" "{app_bundle}.old" 2>/dev/null\n'
        f'  mv "{app_bundle}.new" "{app_bundle}"\n'
        f'  rm -rf "{app_bundle}.old"\n'
        f'  open "{app_bundle}"\n'
        "fi\n"
        f'rm -rf "{staging_root}"\n'
        f'rm -f "{zip_path}"\n'
        'rm -f "$0"\n'
    )
    if '"' in app_bundle or '"' in staged_bundle or "\n" in app_bundle:
        raise ValueError("Unsafe path for the update script")
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
