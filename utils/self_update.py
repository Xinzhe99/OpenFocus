"""Self-update: download, stage and apply release builds in place.

Designed for the PyInstaller onedir layout (app folder with the exe plus
_internal). Flow:

1. download the release portable zip (progress callback)
2. extract into a staging folder under the user-writable temp dir
3. verify the staged executable exists
4. write a small apply script that switches a complete candidate into place
   after the app quits, preserving portable data and rolling back on failure
5. quit the app; the detached script finishes the swap

Staging deliberately lives in the temp dir, never next to the installation:
an install under Program Files is not writable by the running (unelevated)
process, so staging beside it would fail before the swap could even be
offered. The Windows apply script can retry the complete transaction with
PowerShell -Verb RunAs (one UAC prompt).

The release tag arrives from the download mirror, so it is treated as untrusted
input: it is sanitised before reaching any filesystem path or script text.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from typing import Callable, Optional

PORTABLE_MARKER = "OpenFocus.portable"
# Present only in the independent CUDA/ROCm/XPU preview builds. They carry a
# different torch runtime, so the stable-channel updater must not replace them.
GPU_VARIANT_MARKER = "OpenFocus.gpu-variant"

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


def gpu_variant() -> dict:
    """Return the build-time accelerator record; {} only for a stable build.

    Fail closed: once the marker path exists, a malformed, oversized or
    unreadable file still means "not a stable build" so the stable updater can
    never replace a vendor runtime. Only a confirmed absent marker (or a
    source run, which cannot self-update) opens the normal stable path.
    """
    if not is_frozen():
        return {}
    try:
        path = os.path.join(app_install_dir(), GPU_VARIANT_MARKER)
        os.lstat(path)
    except FileNotFoundError:
        return {}
    except OSError:
        return {"variant": "unknown", "invalid": True}
    try:
        if os.path.getsize(path) > 4096:
            raise ValueError("marker is too large")
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        if isinstance(data, dict) and str(data.get("variant") or ""):
            return data
    except (OSError, ValueError):
        pass
    return {"variant": "unknown", "invalid": True}


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


def _valid_macos_bundle(bundle: str) -> bool:
    """Check the declared executable, its mode, and native Mach-O architecture."""
    import platform
    import plistlib
    import struct
    try:
        with open(os.path.join(bundle, "Contents", "Info.plist"), "rb") as stream:
            name = plistlib.load(stream)["CFBundleExecutable"]
        if not name or os.path.basename(name) != name:
            return False
        entry = os.path.join(bundle, "Contents", "MacOS", name)
        if not os.path.isfile(entry) or not os.access(entry, os.X_OK):
            return False
        expected = 0x0100000C if platform.machine().lower() in ("arm64", "aarch64") else 0x01000007
        with open(entry, "rb") as stream:
            header = stream.read(8)
            if len(header) != 8:
                return False
            magic = header[:4]
            if magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):
                return struct.unpack("<I", header[4:])[0] == expected
            if magic in (b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce"):
                return struct.unpack(">I", header[4:])[0] == expected
            if magic in (b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"):
                count = struct.unpack(">I", header[4:])[0]
                size = 32 if magic[-1] == 0xBF else 20
                if count > 32:
                    return False
                for _ in range(count):
                    record = stream.read(size)
                    if len(record) != size:
                        return False
                    if struct.unpack(">I", record[:4])[0] == expected:
                        return True
        return False
    except (OSError, ValueError, KeyError, TypeError, plistlib.InvalidFileException):
        return False


def verify_staged_app(staging_root: str) -> bool:
    """True when the staged folder contains a runnable app entry."""
    if sys.platform == "darwin":
        bundle = find_staged_bundle(staging_root)
        return bool(bundle and _valid_macos_bundle(bundle))
    return bool(find_staged_app_dir(staging_root))


def extract_zip(zip_path: str, dest_root: str,
                on_progress: Optional[Callable[[int, int], None]] = None,
                should_cancel: Optional[Callable[[], bool]] = None) -> None:
    """Extract in bounded blocks, preserving Unix modes and safe bundle links.

    Cancellation is checked between reads and writes; an individual system IO
    operation can still block until the operating system returns.
    """
    import stat
    from utils.updater import DownloadCancelled

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise DownloadCancelled("staging cancelled")

    os.makedirs(dest_root, exist_ok=True)
    root = os.path.realpath(dest_root)
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.infolist()
        for i, entry in enumerate(entries):
            check_cancel()
            target = os.path.join(root, entry.filename)
            if os.path.commonpath((root, os.path.realpath(target))) != root:
                raise ValueError("Archive entry escapes the staging directory")
            mode = entry.external_attr >> 16
            if entry.is_dir():
                os.makedirs(target, exist_ok=True)
            elif stat.S_ISLNK(mode):
                if entry.file_size > 4096:
                    raise ValueError("Archive symlink target is too long")
                check_cancel()
                with archive.open(entry) as source:
                    chunks = []
                    remaining = 4097
                    while remaining:
                        check_cancel()
                        chunk = source.read(remaining)
                        check_cancel()
                        if not chunk:
                            break
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    link_bytes = b"".join(chunks)
                if len(link_bytes) > 4096:
                    raise ValueError("Archive symlink target is too long")
                link = link_bytes.decode("utf-8")
                resolved = os.path.realpath(os.path.join(os.path.dirname(target), link))
                if os.path.isabs(link) or os.path.commonpath((root, resolved)) != root:
                    raise ValueError("Archive symlink escapes the staging directory")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.symlink(link, target)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with archive.open(entry) as source, open(target, "wb") as dest:
                    while True:
                        check_cancel()
                        chunk = source.read(256 * 1024)
                        check_cancel()
                        if not chunk:
                            break
                        dest.write(chunk)
                check_cancel()
                if mode and os.name != "nt":
                    os.chmod(target, stat.S_IMODE(mode))
            if on_progress is not None:
                on_progress(i + 1, len(entries))
        check_cancel()




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
    """Write a PowerShell transaction with encoded, uninterpreted paths."""
    import base64

    app_dir = os.path.abspath(app_dir)
    staging_app_dir = os.path.abspath(staging_app_dir)
    staging_root = os.path.abspath(staging_root or os.path.dirname(staging_app_dir))
    if any('"' in p or '\r' in p or '\n' in p for p in (app_dir, staging_app_dir, staging_root, zip_path)):
        raise ValueError("Unsafe path for the update script")
    try:
        pid = max(0, int(wait_pid))
    except (TypeError, ValueError):
        pid = 0
    config = base64.b64encode(json.dumps({
        "app": app_dir, "staged": staging_app_dir, "root": staging_root,
        "zip": os.path.abspath(zip_path) if zip_path else "",
        "result": result_path(tag), "pid": pid, "tag": sanitize_tag(tag),
    }, ensure_ascii=True).encode("utf-8")).decode("ascii")
    template = r'''
$ErrorActionPreference = 'Stop'
$c = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CONFIG__')) | ConvertFrom-Json
$elevated = __ELEVATED__
$id = [Guid]::NewGuid().ToString('N')
$candidate = $c.app + '.update-new-' + $c.tag + '-' + $id
$backup = $c.app + '.update-old-' + $c.tag + '-' + $id
$oldMoved = $false
$newMoved = $false
$moved = @()
$stage = 'preparation failed'
function Result([string]$value) {
    try { [IO.File]::WriteAllText($c.result, $value, (New-Object Text.UTF8Encoding $false)) } catch {}
}
function Restart-App {
    try { Start-Process -FilePath (Join-Path $c.app 'OpenFocus.exe') -ArgumentList '--restore-session' } catch {}
}
try {
    $stage = 'process exit timed out'
    if ($c.pid -gt 0) {
        $process = Get-Process -Id $c.pid -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            try { $process.WaitForExit(60000) | Out-Null } catch {}
            if (Get-Process -Id $c.pid -ErrorAction SilentlyContinue) { throw 'process still running' }
        }
    }
    $stage = 'preparation failed'
    New-Item -ItemType Directory -Path $candidate -ErrorAction Stop | Out-Null
    & robocopy $c.staged $candidate /E /R:10 /W:5 /NFL /NDL /NJH /NJS /NP /XF OpenFocus.portable /XD OpenFocusData | Out-Null
    if ($LASTEXITCODE -ge 8) { throw 'candidate copy failed' }
    if (!(Test-Path -LiteralPath (Join-Path $candidate 'OpenFocus.exe') -PathType Leaf) -or
        !(Test-Path -LiteralPath (Join-Path $candidate '_internal') -PathType Container)) {
        throw 'candidate is incomplete'
    }
    $stage = 'process exit timed out'
    if ($c.pid -gt 0 -and (Get-Process -Id $c.pid -ErrorAction SilentlyContinue)) {
        throw 'process still running'
    }
    $stage = 'switch failed'
    $preserve = @('OpenFocus.portable', 'OpenFocusData') | Where-Object {
        Test-Path -LiteralPath (Join-Path $c.app $_)
    }
    Move-Item -LiteralPath $c.app -Destination $backup
    $oldMoved = $true
    Move-Item -LiteralPath $candidate -Destination $c.app
    $newMoved = $true
    foreach ($name in $preserve) {
        Move-Item -LiteralPath (Join-Path $backup $name) -Destination (Join-Path $c.app $name)
        $moved += $name
    }
} catch {
    $rollbackOK = $true
    try {
        foreach ($name in $moved) {
            Move-Item -LiteralPath (Join-Path $c.app $name) -Destination (Join-Path $backup $name)
        }
        if ($newMoved) { Move-Item -LiteralPath $c.app -Destination $candidate }
        if ($oldMoved) { Move-Item -LiteralPath $backup -Destination $c.app }
    } catch { $rollbackOK = $false }
    if (!$rollbackOK) {
        Result ("failed`nrollback failed; backup: " + $backup)
        exit 1
    }
    if (Test-Path -LiteralPath $candidate) {
        try { Remove-Item -LiteralPath $candidate -Recurse -Force } catch {}
    }
    if ($stage -ne 'process exit timed out' -and !$elevated) {
        try {
            $p = Start-Process -FilePath 'powershell.exe' -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand','__INNER__')
            exit $p.ExitCode
        } catch { $stage = 'elevation cancelled or failed' }
    }
    Result ("failed`n" + $stage + $(if ($oldMoved) { '; rollback succeeded' } else { '' }))
    if ($stage -ne 'process exit timed out') { Restart-App }
    exit 1
}
Result 'ok'
foreach ($path in @($backup, $c.root, $c.zip)) {
    if ($path) { try { Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop } catch {} }
}
Restart-App
'''
    inner = template.replace('__CONFIG__', config).replace('__ELEVATED__', '$true').replace('__INNER__', '')
    encoded_inner = base64.b64encode(inner.encode('utf-16le')).decode('ascii')
    script = (template.replace('__CONFIG__', config)
              .replace('__ELEVATED__', '$false').replace('__INNER__', encoded_inner))
    script_path = os.path.join(os.path.dirname(staging_root),
                               f"apply_update_{sanitize_tag(tag)}.ps1")
    with open(script_path, 'w', encoding='utf-8-sig', newline='\n') as f:
        f.write(script)
    return script_path


def write_apply_script_macos(app_bundle: str, staged_bundle: str, tag: str,
                             staging_root: str = "", zip_path: str = "",
                             wait_pid: int = 0) -> str:
    app_bundle = os.path.abspath(app_bundle)
    staged_bundle = os.path.abspath(staged_bundle)
    staging_root = os.path.abspath(staging_root or os.path.dirname(staged_bundle))
    if not app_bundle.endswith(".app") or not os.path.isdir(app_bundle):
        raise ValueError("Refusing to replace a non-existent .app bundle")
    if not _valid_macos_bundle(staged_bundle):
        raise ValueError("Staged bundle is incomplete or incompatible")
    pid = int(wait_pid) if str(wait_pid).lstrip("-").isdigit() else 0
    result_file = result_path(tag)
    script_path = os.path.join(
        os.path.dirname(staging_root), f"apply_update_{sanitize_tag(tag)}.sh")
    import shlex
    q = shlex.quote
    content = (
        "#!/bin/bash\n"
        "for _ in $(seq 1 20); do\n"
        f"  [ {pid} -gt 0 ] && kill -0 {pid} 2>/dev/null || break\n"
        "  sleep 1\n"
        "done\n"
        f"if [ {pid} -gt 0 ] && kill -0 {pid} 2>/dev/null; then exit 1; fi\n"
        "ok=0\n"
        f"if rm -rf {q(app_bundle + '.new')} && ditto {q(staged_bundle)} {q(app_bundle + '.new')}; then\n"
        f"  rm -rf {q(app_bundle + '.old')}\n"
        f"  if mv {q(app_bundle)} {q(app_bundle + '.old')}; then\n"
        f"    if mv {q(app_bundle + '.new')} {q(app_bundle)}; then\n"
        "      ok=1\n"
        f"      rm -rf {q(app_bundle + '.old')}\n"
        "    else\n"
        f"      mv {q(app_bundle + '.old')} {q(app_bundle)}\n"
        "    fi\n"
        "  fi\n"
        "fi\n"
        'if [ "$ok" = 1 ]; then\n'
        f"  echo ok > {q(result_file)}\n"
        f"  rm -rf {q(staging_root)}\n"
        f"  rm -f {q(zip_path)}\n"
        "else\n"
        f"  echo failed > {q(result_file)}\n"
        "fi\n"
        f"open {q(app_bundle)} --args --restore-session\n"
        'rm -f "$0"\n'
    )
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(script_path, 0o755)
    return script_path


def launch_detached_windows(script_path: str) -> None:
    flags = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                      "-File", script_path], close_fds=True,
                     creationflags=flags, cwd=os.path.dirname(script_path))


def launch_detached_unix(script_path: str) -> None:
    subprocess.Popen(["/bin/bash", script_path], start_new_session=True)
