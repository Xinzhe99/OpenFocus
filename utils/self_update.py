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
    # 非 Windows 也要接受 OpenFocus.exe：Windows onedir 的 zip 在任何系统上
    # 解开都只有这个入口（Linux CI 的端到端用例就是这种场景）。
    names = ([APP_NAME + ".exe"] if sys.platform == "win32"
             else [APP_NAME, APP_NAME + ".exe"])
    return any(os.path.isfile(os.path.join(candidate, n)) for n in names)


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


def _short_path(path: str) -> str:
    """ASCII-safe 8.3 path via GetShortPathNameW; falls back to the input.

    A .bat is parsed in the OEM codepage, so non-ASCII paths (Chinese
    usernames under %TEMP%) must be converted to their short form before
    being embedded — an encoding error here used to kill the one-click
    update silently after the download had already finished.
    """
    if sys.platform != "win32" or path.isascii():
        return path
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(
            os.path.abspath(path), buf, len(buf))
        if 0 < n < len(buf):
            return buf.value
    except Exception:
        pass
    return path


def _bat_path(path: str) -> str:
    """Quote a path for cmd, refusing anything that could escape the quotes."""
    abspath = _short_path(os.path.abspath(path))
    if '"' in abspath or "\n" in abspath or "\r" in abspath:
        raise ValueError("Unsafe path for the update script")
    if not abspath.isascii():
        raise ValueError("Path cannot be made ASCII-safe for the update script")
    return f'"{abspath}"'


def portable_zip_path(tag: str) -> str:
    """Canonical download target for a release's portable zip."""
    return os.path.join(tempfile.gettempdir(),
                        f"{APP_NAME}-{sanitize_tag(tag)}-portable.zip")


def result_path(tag: str) -> str:
    return os.path.join(tempfile.gettempdir(),
                        f"{APP_NAME}_update_result_{sanitize_tag(tag)}.txt")


def write_result(tag: str, ok: bool, detail: str = "") -> None:
    """Record how the swap ended so the next launch can report it.

    Without this a failed swap was indistinguishable from "the user closed
    the app": the old build came back with no explanation at all.
    """
    try:
        with open(result_path(tag), "w", encoding="utf-8") as f:
            f.write(("ok" if ok else "failed") + "\n" + str(detail or ""))
    except OSError:
        pass


def consume_results() -> list:
    """Read and delete every pending apply result. [(tag, ok, detail)]."""
    import glob
    out = []
    pattern = os.path.join(tempfile.gettempdir(), f"{APP_NAME}_update_result_*.txt")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                first, _, rest = f.read().partition("\n")
            out.append((os.path.basename(path)
                        .replace(f"{APP_NAME}_update_result_", "")
                        .replace(".txt", ""),
                        first.strip() == "ok", rest.strip()))
        except OSError:
            continue
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    return out


def staging_complete_marker(tag: str) -> str:
    """Path of the marker that says "this staging folder is a whole build"."""
    return os.path.join(staging_root_for(tag), ".staging_complete")


def mark_staging_complete(tag: str) -> bool:
    """Called only after extract + verify succeeded.

    `verify_staged_app` can only see that an entry point exists, so a staging
    folder truncated by a cancelled extraction (or a full disk) looked exactly
    like a good build and would have been swapped in over a working install.
    """
    try:
        with open(staging_complete_marker(tag), "w", encoding="utf-8") as f:
            f.write("ok")
        return True
    except OSError:
        return False


def discard_staging(tag: str) -> None:
    """Remove an incomplete staging folder (never leave it half-unpacked)."""
    import shutil
    try:
        shutil.rmtree(staging_root_for(tag), ignore_errors=True)
    except Exception:
        pass


def staged_ready(tag: str) -> bool:
    """True when a *complete* staged build for this tag is on disk."""
    root = staging_root_for(tag)
    if not os.path.isdir(root):
        return False
    if not os.path.isfile(staging_complete_marker(tag)):
        return False
    return verify_staged_app(root)


def write_apply_script_windows(app_dir: str, staging_app_dir: str, tag: str,
                               staging_root: str = "", zip_path: str = "",
                               wait_pid: int = 0) -> str:
    """Write a detached batch script: wait for the app to exit -> mirror the
    staged folder over the installation -> clean up -> restart. Retries the
    copy elevated when the install directory is not writable.

    The elevated retry re-invokes this script with '/elevated'; that branch
    only copies, so a failing unelevated first pass cannot loop into repeated
    UAC prompts.

    Everything here runs without a visible console (the launcher uses
    CREATE_NO_WINDOW): an earlier version combined DETACHED_PROCESS with
    per-second `timeout`/`tasklist` polling, which allocated a fresh console
    window per call — the "flashing black windows" users saw — and robocopy
    without /R:/W: retried a locked exe a million times in silence, so the
    update never finished and the app never came back.
    """
    app_dir = os.path.abspath(app_dir)
    staging_app_dir = os.path.abspath(staging_app_dir)
    staging_root = os.path.abspath(staging_root or os.path.dirname(staging_app_dir))
    script_dir = os.path.dirname(staging_root)
    script_path = os.path.join(
        script_dir, f"apply_update_{sanitize_tag(tag)}.bat")
    try:
        pid = max(0, int(wait_pid))
    except (TypeError, ValueError):
        pid = 0

    def _quote_policy(allow_ansi: bool):
        """Quote a path for cmd, refusing anything that could escape quotes."""
        def q(path: str) -> str:
            abspath = _short_path(os.path.abspath(path))
            if not abspath.isascii():
                if not allow_ansi:
                    raise ValueError(
                        "Path cannot be made ASCII-safe for the update script")
                # 8.3 短名保不住 ASCII（CJK 目录常见）。退回原始路径，
                # 整个脚本改用 ANSI 代码页写入——中文系统的 cp936/GBK
                # 同时是 cmd 的解析代码页，可正确解析
                abspath = os.path.abspath(path)
            if '"' in abspath or '\n' in abspath or '\r' in abspath:
                raise ValueError("Unsafe path for the update script")
            # cmd expands %VAR% inside double quotes: a '%' in the temp or
            # install path silently rewrote the robocopy arguments (and the
            # result path), so the swap failed in a way nobody could see.
            # Doubling it is the escape cmd understands.
            abspath = abspath.replace("%", "%%")
            return f'"{abspath}"'
        return q

    def _build_content(_bp):
        exe = os.path.join(app_dir, APP_NAME + ".exe")
        sentinel = os.path.join(script_dir, f"update_ok_{sanitize_tag(tag)}.flg")
        result = result_path(tag)
        copy_cmd = (
            f"robocopy {_bp(staging_app_dir)} {_bp(app_dir)}"
            # Bounded retries: the defaults (/R:1000000 /W:30) turned one
            # locked binary into a silent multi-hour stall.
            " /E /IS /IT /R:2 /W:1 /NFL /NDL /NJH /NJS /NP"
        )
        content = (
            "@echo off\r\n"
            "setlocal enableextensions\r\n"
            f"set APP_PID={pid}\r\n"
            # A sentinel left by an interrupted earlier attempt would make this
            # run report success without a successful copy.
            f'if exist {_bp(sentinel)} del /Q {_bp(sentinel)} >nul 2>&1\r\n'
            'if "%~1"=="/elevated" goto elevatedcopy\r\n'
            "call :waitforexit\r\n"
            f"{copy_cmd}\r\n"
            "if errorlevel 8 (\r\n"
            # Re-invoke this script elevated. Keep `%~f0` inside PowerShell
            # SINGLE quotes: doubling it with quotes would close cmd's own
            # double-quoted -Command region and mangle the whole line.
            '  powershell -NoProfile -Command "Start-Process -FilePath \'%~f0\''
            ' -ArgumentList \'/elevated\' -Verb RunAs -Wait -WindowStyle Hidden"\r\n'
            ") else (\r\n"
            # 非特权复制成功同样要写哨兵：只有特权分支写的话，可写安装
            # （便携版）每次更新都会跳过下面的清理，~1GB 暂存永久留在 %TEMP%
            f'  if not errorlevel 8 type NUL > {_bp(sentinel)}\r\n'
            ")\r\n"
            # Clean up + relaunch only after a copy that verifiably succeeded:
            # the elevated branch writes the sentinel on success, so a declined
            # UAC prompt (or a still-failing copy) keeps the staged folder as
            # the rollback instead of deleting it and relaunching a mixed
            # install. A failure is recorded so the next launch can say so.
            f"if not exist {_bp(sentinel)} (\r\n"
            f'  echo failed> {_bp(result)}\r\n'
            f'  echo robocopy errorlevel %ERRORLEVEL%>> {_bp(result)}\r\n'
            f'  start "" {_bp(exe)}\r\n'
            "  exit /b\r\n"
            ")\r\n"            f"rmdir /S /Q {_bp(staging_root)} >nul 2>&1\r\n"
            f"if exist {_bp(zip_path)} del /Q {_bp(zip_path)} >nul 2>&1\r\n"
            f"if exist {_bp(sentinel)} del /Q {_bp(sentinel)} >nul 2>&1\r\n"
            # The relaunched instance restores the session the updater
            # snapshotted before quitting (see utils.recovery).
            f'echo ok> {_bp(result)}\r\n'
            f'start "" {_bp(exe)} --restore-session\r\n'
            # Detached self-delete: cmd reads a .bat incrementally, so
            # deleting it inline can truncate the remaining lines.
            f'start "" /b cmd /c del /Q "%~f0"\r\n'
            "exit /b\r\n"
            ":elevatedcopy\r\n"
            "call :waitforexit\r\n"
            f"{copy_cmd}\r\n"
            f'if not errorlevel 8 type NUL > {_bp(sentinel)}\r\n'
            "exit /b\r\n"
            ":waitforexit\r\n"
            # No PID (0/unknown): nothing to wait for — otherwise the filter
            # matches the System Idle Process row and burns the full timeout.
            "if %APP_PID% LEQ 0 exit /b 0\r\n"
            "set /a TRIES=0\r\n"
            ":waitloop\r\n"
            # 用 PID 等待，不用镜像名：更新期间用户手动再开一个实例时，
            # 按 IMAGENAME 过滤会把新实例当成"老进程还在"而干等到超时。
            f'tasklist /FI "PID eq %APP_PID%" 2>NUL | find " %APP_PID% " >NUL\r\n'
            "if errorlevel 1 exit /b 0\r\n"
            "set /a TRIES+=1\r\n"
            "if %TRIES% GEQ 60 exit /b 0\r\n"
            # ping 而不是 timeout：隐藏控制台下 timeout 直接报错退出（不等待）
            "ping -n 2 127.0.0.1 >NUL 2>&1\r\n"
            "goto waitloop\r\n"
        )

        return content


    try:
        content = _build_content(_quote_policy(False))
        encoding = "ascii"
    except ValueError:
        content = _build_content(_quote_policy(True))
        encoding = "mbcs"

    with open(script_path, "w", encoding=encoding) as f:
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
    result_file = result_path(tag)
    script_path = os.path.join(
        os.path.dirname(staging_root), f"apply_update_{sanitize_tag(tag)}.sh")
    content = (
        "#!/bin/bash\n"
        "for _ in $(seq 1 20); do\n"
        f"  [ {pid} -gt 0 ] && kill -0 {pid} 2>/dev/null || break\n"
        "  sleep 1\n"
        "done\n"
        # Swap guarded step by step: keep the old bundle until the new one
        # is verified in place, restore it if the second mv fails, and
        # always relaunch something (old or new) so the app never just
        # disappears after quitting for an update. Every branch records the
        # outcome so the next launch can report a failed swap.
        f'if ditto "{staged_bundle}" "{app_bundle}.new"; then\n'
        '  rm -rf "' + app_bundle + '.old"\n'
        f'  if mv "{app_bundle}" "{app_bundle}.old" 2>/dev/null; then\n'
        f'    if mv "{app_bundle}.new" "{app_bundle}"; then\n'
        f'      rm -rf "{app_bundle}.old"\n'
        f'      echo ok > "{result_file}"\n'
        f'      open "{app_bundle}" --args --restore-session\n'
        "    else\n"
        f'      mv "{app_bundle}.old" "{app_bundle}" 2>/dev/null\n'
        f'      echo failed > "{result_file}"\n'
        f'      open "{app_bundle}"\n'
        "    fi\n"
        "  else\n"
        f'    echo failed > "{result_file}"\n'
        f'    open "{app_bundle}"\n'
        "  fi\n"
        "else\n"
        f'  echo failed > "{result_file}"\n'
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
    # CREATE_NO_WINDOW, NOT DETACHED_PROCESS: a detached cmd has no console,
    # so every console child it spawns (tasklist, ping, robocopy, powershell)
    # allocates its own — which is what flashed a burst of black windows over
    # the screen while the update ran. With a hidden console they all inherit
    # it and stay invisible.
    flags = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(["cmd", "/c", script_path], close_fds=True,
                     creationflags=flags, cwd=os.path.dirname(script_path))


def launch_detached_unix(script_path: str) -> None:
    subprocess.Popen(["/bin/bash", script_path], start_new_session=True)
