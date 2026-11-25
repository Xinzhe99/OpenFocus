"""Self-update: download, stage and apply release builds in place.

Designed for the PyInstaller onedir layout (app folder with the exe plus
_internal). Flow:

1. download and verify a release package (or an exact-base patch)
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
import stat
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


def linux_app_bundle() -> Optional[str]:
    """Path of the running official Linux onedir bundle, if present."""
    if sys.platform != "linux" or not is_frozen() or os.environ.get("APPIMAGE"):
        return None
    from utils.release_delta import platform_key
    if platform_key() != "linux-x86_64":
        return None
    executable = os.path.realpath(sys.executable)
    bundle = os.path.dirname(executable)
    if (os.path.basename(executable) == APP_NAME
            and verify_update_candidate(bundle, "linux-x86_64")):
        return bundle
    return None


def installed_update_root() -> str:
    if sys.platform == "win32":
        return app_install_dir()
    return mac_app_bundle() or linux_app_bundle() or ""


def _update_work_dir() -> str:
    root = tempfile.gettempdir()
    if sys.platform != "linux":
        return root
    # This cache and its READY markers must not be writable by other /tmp users.
    directory = os.path.join(root, f"OpenFocus-updates-{os.getuid()}")
    try:
        os.mkdir(directory, 0o700)
    except FileExistsError:
        pass
    info = os.lstat(directory)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise OSError("Unsafe OpenFocus update workspace")
    return directory


def staging_root_for(tag: str) -> str:
    """User-writable staging directory for this update."""
    return os.path.join(_update_work_dir(), STAGING_PREFIX + sanitize_tag(tag))


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



def verify_update_candidate(candidate_root: str, platform: str, *, patch: bool = False) -> bool:
    """Check the bundle layout and validate the signature of a patched macOS app."""
    if platform == "macos-arm64":
        if not _valid_macos_bundle(candidate_root):
            return False
        if not patch:
            return True
        try:
            return subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", candidate_root],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  timeout=60).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
    if platform not in ("windows-x64", "linux-x86_64"):
        return False
    entry = os.path.join(candidate_root, APP_NAME + (".exe" if platform == "windows-x64" else ""))
    return (os.path.isfile(entry) and not os.path.islink(entry)
            and os.path.isdir(os.path.join(candidate_root, "_internal"))
            and (platform == "windows-x64" or os.access(entry, os.X_OK)))


def verify_staged_app(staging_root: str) -> bool:
    """True when the staged folder contains a runnable app entry."""
    if sys.platform == "darwin":
        bundle = find_staged_bundle(staging_root)
        return bool(bundle and _valid_macos_bundle(bundle))
    if sys.platform == "linux":
        candidate = find_staged_app_dir(staging_root)
        return bool(candidate and verify_update_candidate(candidate, "linux-x86_64"))
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
            name = entry.filename
            if (not name or "\\" in name or name.startswith("/")
                    or (len(name) > 1 and name[1] == ":")):
                raise ValueError("Archive entry escapes the staging directory")
            target = os.path.join(root, name)
            if os.path.commonpath((root, os.path.realpath(target))) != root:
                raise ValueError("Archive entry escapes the staging directory")
            if os.path.realpath(target) == root:
                continue
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
                if os.path.lexists(target) and not os.path.isdir(target):
                    os.unlink(target)
                os.symlink(link, target)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if os.path.islink(target):
                    os.unlink(target)
                elif os.path.isfile(target) and not os.access(target, os.W_OK):
                    os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
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



def extract_linux_tar(tar_path: str, dest_root: str,
                      on_progress: Optional[Callable[[int, int], None]] = None,
                      should_cancel: Optional[Callable[[], bool]] = None) -> None:
    """Extract the official onedir bundle, preserving safe links and modes."""
    import tarfile
    from utils.updater import DownloadCancelled

    def check_cancel():
        if should_cancel is not None and should_cancel():
            raise DownloadCancelled("staging cancelled")

    os.makedirs(dest_root, exist_ok=True)
    root = os.path.realpath(dest_root)
    bundle = os.path.join(root, APP_NAME)

    def member_path(name):
        parts = name.rstrip("/").split("/")
        if (parts[0] != APP_NAME or any(part in ("", ".", "..") for part in parts)
                or "\\" in name):
            raise ValueError("Archive entry escapes the application bundle")
        path = os.path.join(root, *parts)
        if os.path.realpath(path) != path:
            raise ValueError("Archive entry overlaps a symlink")
        return path

    directories, hardlinks = [], []
    done = 0
    with tarfile.open(tar_path, "r|gz") as archive:
        for entry in archive:
            check_cancel()
            target = member_path(entry.name)
            if entry.isdir():
                os.makedirs(target, exist_ok=True)
                directories.append((target, entry.mode))
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                if entry.issym():
                    link = entry.linkname
                    resolved = os.path.realpath(os.path.join(os.path.dirname(target), link))
                    if os.path.isabs(link) or os.path.commonpath((bundle, resolved)) != bundle:
                        raise ValueError("Archive symlink escapes the application bundle")
                    os.symlink(link, target)
                elif entry.islnk():
                    hardlinks.append((target, member_path(entry.linkname)))
                elif entry.isfile():
                    with archive.extractfile(entry) as source, open(target, "xb") as dest:
                        while True:
                            check_cancel()
                            chunk = source.read(256 * 1024)
                            check_cancel()
                            if not chunk:
                                break
                            dest.write(chunk)
                    os.chmod(target, entry.mode & 0o777)
                else:
                    raise ValueError("Unsupported Linux archive entry")
            done += 1
            if on_progress is not None:
                on_progress(done, 0)
    for target, source in hardlinks:
        check_cancel()
        if (os.path.realpath(target) != target or os.path.realpath(source) != source
                or not os.path.isfile(source)):
            raise ValueError("Invalid Linux archive hardlink")
        os.link(source, target)
    for target, mode in reversed(directories):
        os.chmod(target, mode & 0o777)
    check_cancel()


def full_package_path(tag: str) -> str:
    if sys.platform == "linux":
        return os.path.join(_update_work_dir(), f"{APP_NAME}-{sanitize_tag(tag)}-linux-x86_64.tar.gz")
    return portable_zip_path(tag)


def delta_package_path(name: str) -> str:
    if not re.fullmatch(r"OpenFocus-[A-Za-z0-9._-]+\.ofdelta\.zip", name):
        raise ValueError("Invalid patch archive name")
    return os.path.join(_update_work_dir(), name)


def portable_zip_path(tag: str) -> str:
    """Canonical download target for a release's portable zip."""
    return os.path.join(tempfile.gettempdir(),
                        f"{APP_NAME}-{sanitize_tag(tag)}-portable.zip")


def result_path(tag: str) -> str:
    return os.path.join(_update_work_dir(), f"{APP_NAME}_update_result_{sanitize_tag(tag)}.txt")


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
    try:
        pattern = os.path.join(_update_work_dir(), f"{APP_NAME}_update_result_*.txt")
    except OSError:
        return out
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

    def _remove_readonly_file(func, path, exc_info):
        if func is not os.unlink or not isinstance(exc_info[1], PermissionError):
            raise exc_info[1]
        if not stat.S_ISREG(os.lstat(path).st_mode):
            raise exc_info[1]
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        func(path)

    def _make_directory_writable(path):
        mode = os.lstat(path).st_mode
        if stat.S_ISDIR(mode) and mode & stat.S_IRWXU != stat.S_IRWXU:
            os.chmod(path, stat.S_IMODE(mode) | stat.S_IRWXU)

    try:
        root = staging_root_for(tag)
        if os.name == "nt":
            shutil.rmtree(root, onerror=_remove_readonly_file)
            return
        try:
            shutil.rmtree(root)
        except PermissionError:
            # Unlink permission on POSIX belongs to the enclosing directory.
            _make_directory_writable(root)
            for parent, directories, _ in os.walk(root):
                for directory in directories:
                    _make_directory_writable(os.path.join(parent, directory))
            shutil.rmtree(root)
    except OSError:
        pass


def staged_ready(tag: str) -> bool:
    """True when a *complete* staged build for this tag is on disk."""
    try:
        root = staging_root_for(tag)
    except OSError:
        return False
    if not os.path.isdir(root):
        return False
    if not os.path.isfile(staging_complete_marker(tag)):
        return False
    return verify_staged_app(root)


def write_apply_script_windows(app_dir: str, staging_app_dir: str, tag: str,
                               staging_root: str = "", zip_path: str = "",
                               wait_pid: int = 0, *, restart: bool = True) -> str:
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
        "restart": bool(restart),
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
    if (!$c.restart -or $elevated) { return }
    $env:PYINSTALLER_RESET_ENVIRONMENT = '1'
    try { Start-Process -FilePath (Join-Path $c.app 'OpenFocus.exe') -ArgumentList '--restore-session' } catch {}
}
function Move-WithRetry([string]$Source, [string]$Destination) {
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        try {
            Move-Item -LiteralPath $Source -Destination $Destination
            return
        } catch {
            if ($attempt -lt 9 -and $_.Exception -is [System.IO.IOException] -and
                -not ($_.Exception -is [System.IO.FileNotFoundException]) -and
                -not ($_.Exception -is [System.IO.DirectoryNotFoundException])) {
                Start-Sleep -Milliseconds 250
                continue
            }
            throw
        }
    }
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
    & robocopy $c.staged $candidate /E /R:10 /W:5 /NFL /NDL /NJH /NJS /NP /XF OpenFocus.portable unins*.exe unins*.dat unins*.msg /XD OpenFocusData /XJ | Out-Null
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
    $oldEntries = @(Get-ChildItem -LiteralPath $c.app -Force)
    $preserve = @($oldEntries | Where-Object {
        $_.Name -in @('OpenFocus.portable', 'OpenFocusData') -or
        (!$_.PSIsContainer -and $_.Name -match '^unins[0-9]{3,}\.(exe|dat|msg)$')
    } | ForEach-Object { $_.Name })
    $keepBackup = @($oldEntries | Where-Object {
        $_.Name -notin $preserve -and
        !($_.PSIsContainer -and $_.Name -match '^_openfocus_runtime_v[0-9.]+(?:-beta\.[0-9]+)?$') -and
        !(Test-Path -LiteralPath (Join-Path $candidate $_.Name))
    }).Count -gt 0
    Move-WithRetry $c.app $backup
    $oldMoved = $true
    Move-WithRetry $candidate $c.app
    $newMoved = $true
    foreach ($name in $preserve) {
        Move-WithRetry (Join-Path $backup $name) (Join-Path $c.app $name)
        $moved += $name
    }
} catch {
    $rollbackOK = $true
    try {
        foreach ($name in $moved) {
            Move-WithRetry (Join-Path $c.app $name) (Join-Path $backup $name)
        }
        if ($newMoved) { Move-WithRetry $c.app $candidate }
        if ($oldMoved) { Move-WithRetry $backup $c.app }
    } catch { $rollbackOK = $false }
    if (!$rollbackOK) {
        Result ("failed`nrollback failed; backup: " + $backup)
        exit 2
    }
    if (Test-Path -LiteralPath $candidate) {
        try { Remove-Item -LiteralPath $candidate -Recurse -Force } catch {}
    }
    if ($stage -ne 'process exit timed out' -and !$elevated) {
        try {
            $p = Start-Process -FilePath 'powershell.exe' -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand','__INNER__')
            # The helper may have different Windows credentials. Relaunch only
            # from the original user's process, including after safe rollback.
            if ($p.ExitCode -in @(0, 1)) { Restart-App }
            exit $p.ExitCode
        } catch { $stage = 'elevation cancelled or failed' }
    }
    Result ("failed`n" + $stage + $(if ($oldMoved) { '; rollback succeeded' } else { '' }))
    if ($stage -eq 'process exit timed out') { exit 3 }
    Restart-App
    exit 1
} finally {
    try { if (!$elevated -and $PSCommandPath) { Remove-Item -LiteralPath $PSCommandPath -Force } } catch {}
}
if ($keepBackup) { Result ("ok`n" + $backup) } else { Result 'ok' }
foreach ($path in @($(if (!$keepBackup) { $backup }), $c.root, $c.zip)) {
    if ($path) { try { Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop } catch {} }
}
Restart-App
exit 0
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
                             wait_pid: int = 0, *, restart: bool = True) -> str:
    if not app_bundle.endswith(".app") or not os.path.isdir(app_bundle):
        raise ValueError("Refusing to replace a non-existent .app bundle")
    if not _valid_macos_bundle(staged_bundle):
        raise ValueError("Staged bundle is incomplete or incompatible")
    return _write_apply_script_unix(app_bundle, staged_bundle, tag, staging_root,
                                    zip_path, wait_pid, restart, macos=True)


def _write_apply_script_unix(installed_root, candidate_root, tag, staging_root,
                             package_path, wait_pid, restart, *, macos):
    import shlex

    app = os.path.abspath(installed_root)
    staged = os.path.abspath(candidate_root)
    root = os.path.abspath(staging_root or os.path.dirname(staged))
    q = shlex.quote
    pid = max(0, int(wait_pid))
    config = (f"app={q(app)}\nstaged={q(staged)}\nroot={q(root)}\n"
              f"package={q(os.path.abspath(package_path) if package_path else '')}\n"
              f"result={q(result_path(tag))}\npid={pid}\nrestart={int(bool(restart))}\n"
              f"macos={int(macos)}\n")
    script = r'''
trap 'rm -f -- "$0"' EXIT

restart_app() {
    [ "$restart" = 1 ] || return 0
    if [ "$macos" = 1 ]; then
        open "$app" --args --restore-session
    else
        PYINSTALLER_RESET_ENVIRONMENT=1 "$app/OpenFocus" --restore-session </dev/null >/dev/null 2>&1 &
    fi
}

exists() { [ -e "$1" ] || [ -L "$1" ]; }

for ((attempt=0; pid>0 && attempt<60; attempt++)); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
done
if [ "$pid" -gt 0 ] && kill -0 "$pid" 2>/dev/null; then
    printf 'failed\nprocess exit timed out\n' > "$result"
    exit 3
fi

transaction=$(mktemp -d "${app}.update-XXXXXXXX") || {
    printf 'failed\npreparation failed\n' > "$result"
    restart_app
    exit 1
}
candidate="$transaction/new"
backup="$transaction/previous"
old_moved=0
new_moved=0
preserved=()

rollback() {
    local name
    for name in "${preserved[@]}"; do
        mv -- "$app/$name" "$backup/$name" || return 1
    done
    if [ "$new_moved" = 1 ]; then mv -- "$app" "$candidate" || return 1; fi
    if [ "$old_moved" = 1 ]; then
        ! exists "$app" && mv -- "$backup" "$app" || return 1
    fi
}

fail() {
    if rollback; then
        rm -rf -- "$transaction"
        printf 'failed\n%s\n' "$1" > "$result"
        restart_app
        exit 1
    fi
    printf 'failed\nrollback failed; backup: %s\n' "$backup" > "$result"
    exit 2
}

if [ "$macos" = 1 ]; then
    ditto "$staged" "$candidate" && [ -d "$candidate/Contents/MacOS" ] || fail 'candidate copy failed'
else
    cp -a -- "$staged" "$candidate" && [ -x "$candidate/OpenFocus" ] &&
        [ -d "$candidate/_internal" ] || fail 'candidate copy failed'
    chmod u+rwx -- "$candidate" || fail 'candidate copy failed'
    rm -rf -- "$candidate/OpenFocus.portable" "$candidate/OpenFocusData" || fail 'candidate copy failed'
fi
[ -d "$app" ] && [ ! -L "$app" ] && mv -- "$app" "$backup" || fail 'switch failed'
old_moved=1
! exists "$app" && mv -- "$candidate" "$app" || fail 'switch failed'
new_moved=1

keep_backup=0
if [ "$macos" = 0 ]; then
    for name in OpenFocus.portable OpenFocusData; do
        if exists "$backup/$name"; then
            mv -- "$backup/$name" "$app/$name" || fail 'preserving local data failed'
            preserved+=("$name")
        fi
    done
    shopt -s dotglob nullglob
    for entry in "$backup"/*; do
        if ! exists "$app/${entry##*/}"; then keep_backup=1; break; fi
    done
fi
if [ "$keep_backup" = 1 ]; then
    printf 'ok\n%s\n' "$backup" > "$result"
else
    printf 'ok\n' > "$result"
    rm -rf -- "$transaction"
fi
rm -rf -- "$root"
if [ -n "$package" ]; then rm -f -- "$package"; fi
restart_app
exit 0
'''
    fd, script_path = tempfile.mkstemp(prefix=f"apply_update_{sanitize_tag(tag)}_", suffix=".sh",
                                       dir=os.path.dirname(root), text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("#!/bin/bash\n" + config + script)
    return script_path


def launch_detached_windows(script_path: str) -> subprocess.Popen:
    flags = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                             "-File", script_path], close_fds=True,
                            creationflags=flags, cwd=os.path.dirname(script_path))


def launch_detached_unix(script_path: str) -> subprocess.Popen:
    env = os.environ.copy()
    if sys.platform == "linux" and is_frozen():
        # The shell and its copy commands need the system libraries, not the
        # libraries PyInstaller prepended for the running build.
        if "LD_LIBRARY_PATH_ORIG" in env:
            env["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH_ORIG"]
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return subprocess.Popen(["/bin/bash", script_path], start_new_session=True,
                            cwd=os.path.dirname(script_path), env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def handoff_update(installed_root: str, candidate_root: str, tag: str, *,
                   staging_root: str, package_path: str = "", wait_pid: int = 0,
                   restart: bool = True) -> subprocess.Popen:
    """Start the shared swap transaction and return its waitable process."""
    if not all((installed_root, candidate_root, staging_root)):
        raise ValueError("Missing update directories")
    app, candidate, root = tuple(os.path.realpath(path) for path in
                                 (installed_root, candidate_root, staging_root))
    for raw, resolved in zip((installed_root, candidate_root, staging_root), (app, candidate, root)):
        path = os.path.abspath(raw)
        canonical_leaf = os.path.join(os.path.realpath(os.path.dirname(path)), os.path.basename(path))
        if (resolved == os.path.dirname(resolved)
                or os.path.normcase(canonical_leaf) != os.path.normcase(resolved)):
            raise ValueError("Unsafe update directories")

    def is_within(path, parent):
        try:
            return os.path.normcase(os.path.commonpath((path, parent))) == os.path.normcase(parent)
        except ValueError:
            return False  # Separate Windows drives.

    if (not is_within(candidate, root) or is_within(app, root) or is_within(root, app)
            or (package_path and is_within(os.path.realpath(package_path), app))):
        raise ValueError("Installation, staging and package paths overlap")
    if sys.platform == "win32":
        if not verify_update_candidate(candidate, "windows-x64"):
            raise ValueError("Staged Windows build is incomplete")
        return launch_detached_windows(write_apply_script_windows(
            app, candidate, tag, root, package_path, wait_pid, restart=restart))
    if sys.platform == "darwin":
        return launch_detached_unix(write_apply_script_macos(
            app, candidate, tag, root, package_path, wait_pid, restart=restart))
    if sys.platform == "linux":
        if not (verify_update_candidate(app, "linux-x86_64") and
                verify_update_candidate(candidate, "linux-x86_64")):
            raise ValueError("Expected complete Linux onedir bundles")
        return launch_detached_unix(_write_apply_script_unix(
            app, candidate, tag, root, package_path, wait_pid, restart, macos=False))
    raise ValueError("Unsupported update platform")
