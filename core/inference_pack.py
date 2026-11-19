"""Install the optional Windows inference worker into the current user's profile.

All artifact URLs, byte counts and SHA-256 hashes come from the application.
A working generation remains selected until its replacement passes the worker's
GPU startup probe. The desktop's Python environment is never modified.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from utils.app_paths import is_portable_mode, portable_data_dir, user_data_dir

_CORE = Path(__file__).resolve().parent
_WORKER = _CORE / 'inference_worker.py'
_GENERATION = re.compile(r'gen-[0-9a-f]{32}')
_BLOCK = 1024 * 1024


class PackCancelled(Exception):
    """An explicit cancellation before activation."""


@dataclass(frozen=True)
class PackStatus:
    supported: bool
    device: str
    reason: str
    download_bytes: int
    installed: bool
    backends: tuple[str, ...]
    version: str
    required_free_bytes: int


@lru_cache(maxsize=1)
def _manifest():
    return json.loads((_CORE / 'accelerator_win64.json').read_text(encoding='utf-8'))


def pack_root() -> Path:
    """Keep portable and explicitly isolated profiles self-contained."""
    if sys.platform != 'win32' or os.environ.get('OPENFOCUS_DATA_DIR'):
        base = user_data_dir()
    elif is_portable_mode():
        base = Path(portable_data_dir())
    else:
        base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local') / 'OpenFocus'
    return base / 'inference-packs'


def _hardware() -> tuple[str, str]:
    if sys.platform != 'win32' or platform.machine().upper() not in ('AMD64', 'X86_64') or sys.maxsize <= 2**32:
        return '', 'platform'
    try:
        # Load the display driver's copy even in a frozen application.
        driver = ctypes.WinDLL('nvcuda.dll', winmode=0x800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
        device = ctypes.c_int()
        driver_version = ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        major, minor = ctypes.c_int(), ctypes.c_int()
        if driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device), 0):
            return '', 'no_cuda_device'
        driver.cuDeviceGetName(name, len(name), device)
        gpu = name.value.decode('utf-8', errors='replace')
        if driver.cuDriverGetVersion(ctypes.byref(driver_version)) or driver_version.value < 12040:
            return gpu, 'driver'
        if (driver.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device)
                or (major.value, minor.value) not in {(7, 5), (8, 0), (8, 6), (8, 9), (9, 0)}):
            return gpu, 'architecture'
        return gpu, ''
    except (OSError, AttributeError):
        return '', 'no_cuda_device'


def _active(root: Path):
    try:
        receipt = json.loads((root / 'active.json').read_text(encoding='utf-8'))
        generation = receipt['generation']
        if isinstance(generation, str) and _GENERATION.fullmatch(generation):
            return receipt
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {}


def worker_command() -> list[str] | None:
    """Resolve only an activated private runtime and this application's worker."""
    if sys.platform != 'win32' or not _WORKER.is_file():
        return None
    root = pack_root()
    receipt = _active(root)
    if receipt.get('version') != _manifest()['version']:
        return None
    runtime = root / receipt['generation']
    if not all((runtime / name).is_file() for name in ('python.exe', 'python310.dll', 'python310._pth')):
        return None
    return [str(runtime / 'python.exe'), '-I', '-u', str(_WORKER)]


def pack_status() -> PackStatus:
    """Offline recommendation for device zero, the same device used by the worker."""
    device, reason = _hardware()
    manifest = _manifest()
    return PackStatus(not reason, device, reason, sum(a['size'] for a in manifest['artifacts']),
                      worker_command() is not None, tuple(manifest['backends']), manifest['version'],
                      manifest['required_free_bytes'])


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise PackCancelled()


@contextmanager
def _install_lock(root):
    # OS-owned byte range locks are released when an installer crashes.
    import msvcrt
    with (root / 'install.lock').open('a+b') as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('An inference pack installation is already running.') from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def _remove_retired(root, keep):
    for path in root.glob('gen-*'):
        if path.name == keep or not _GENERATION.fullmatch(path.name):
            continue
        try:
            if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                continue
            # Windows prevents deletion of a running process's image. Leave the
            # whole retired generation intact while another render is using it.
            (path / 'python.exe').unlink(missing_ok=True)
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)


def _blocks(artifact, local, cancelled, prefer_mirror):
    if local is not None:
        with local.open('rb') as source:
            while chunk := source.read(_BLOCK):
                yield chunk
        return

    size = artifact['size']
    received = 0
    official = artifact['url']
    mirrors = tuple(artifact.get('mirrors', ())) if prefer_mirror else ()
    urls = (*mirrors, official, official, official, official)
    for attempt, url in enumerate(urls):
        _check(cancelled)
        headers = {'Range': f'bytes={received}-'} if received else {}
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=20) as source:
                if received and (source.status != 206 or source.headers.get('Content-Range') !=
                                 f'bytes {received}-{size - 1}/{size}'):
                    if url != official:
                        continue
                    raise ValueError('The download server did not honor the requested byte range.')
                # read1 returns available bytes, including on a slow connection.
                while chunk := source.read1(_BLOCK):
                    received += len(chunk)
                    yield chunk
                if received == size:
                    return
                raise EOFError('The download ended before its pinned byte count.')
        except (OSError, HTTPException, EOFError) as exc:
            _check(cancelled)
            if (attempt == len(urls) - 1
                    or url == official and isinstance(exc, HTTPError) and exc.code < 500):
                raise


def _fetch(artifact, archive, local, cancelled, report, *, prefer_mirror=False):
    expected = artifact['size']
    if shutil.disk_usage(archive.parent).free < expected:
        raise OSError(f'Insufficient disk space for {archive.name} ({expected:,} bytes).')
    digest = hashlib.sha256()
    count = 0
    last_update = 0.0
    with archive.open('xb') as dest:
        for chunk in _blocks(artifact, local, cancelled, prefer_mirror):
            _check(cancelled)
            count += len(chunk)
            if count > expected:
                raise ValueError(f'Download exceeds the pinned size for {archive.name}.')
            digest.update(chunk)
            dest.write(chunk)
            now = time.monotonic()
            if now - last_update >= 0.15:
                report(count)
                last_update = now
    if count != expected or digest.hexdigest() != artifact['sha256']:
        raise ValueError(f'Download size or SHA-256 mismatch for {archive.name}.')
    report(count)


def _extract(archive, runtime, cancelled, report):
    wheel = archive.suffix == '.whl'
    site = runtime / 'Lib' / 'site-packages'
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        required = sum(item.file_size for item in members)
        if shutil.disk_usage(runtime).free < required:
            raise OSError(f'Insufficient disk space to unpack {archive.name} ({required:,} bytes).')
        written = 0
        last_update = time.monotonic()
        report(0, required)
        for item in members:
            _check(cancelled)
            if item.is_dir():
                continue
            parts = PurePosixPath(item.filename).parts
            if not parts or any(p in ('/', '..') or ':' in p or '\\' in p for p in parts):
                raise ValueError(f'Invalid archive path in {archive.name}: {item.filename}')
            base = site if wheel else runtime
            if wheel and parts[0].endswith('.data'):
                scheme = {'purelib': site, 'platlib': site, 'data': runtime,
                          'scripts': runtime / 'Scripts', 'headers': runtime / 'Include'}
                base = scheme[parts[1]]
                parts = parts[2:]
            dest = base.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as inp, dest.open('xb') as out:
                while chunk := inp.read(_BLOCK):
                    _check(cancelled)
                    out.write(chunk)
                    written += len(chunk)
                    now = time.monotonic()
                    if now - last_update >= 0.15:
                        report(written, required)
                        last_update = now
        report(written, required)


def _probe(runtime, backends, cancelled):
    from core.inference_process import spawn
    command = [str(runtime / 'python.exe'), '-I', '-u', str(_WORKER)]
    # EOF makes the worker exit immediately after its startup probe. Use files
    # so diagnostics from a native library can never fill a pipe and hang it.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        with spawn(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr) as process:
            deadline = time.monotonic() + 120
            try:
                while process.poll() is None:
                    _check(cancelled)
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Inference worker startup timed out.')
                    time.sleep(0.12)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        stdout.seek(0)
        line = stdout.readline(8192)
        try:
            hello = json.loads(line)
            if (process.returncode == 0 and hello.get('ready') is True
                    and set(backends) <= set(hello.get('backends', ()))):
                return
        except (ValueError, TypeError, AttributeError):
            pass
        stderr.seek(max(0, stderr.tell() - 2048))
        detail = stderr.read().decode('utf-8', errors='replace').strip()
        raise RuntimeError(f'GPU worker verification failed ({process.returncode}): '
                           f'{line.decode("utf-8", errors="replace").strip()} {detail}')


def install_pack(*, progress=None, cancelled=None, artifact_dir: Path | None = None,
                 prefer_mirror=False) -> PackStatus:
    """Install or repair, reporting progress as (phase, name, done, total).

    Download and local verification counts are cumulative archive bytes.
    Unpack counts are per archive's uncompressed bytes. GPU verification has
    no known total.

    ``artifact_dir`` optionally supplies original archives for an offline install.
    Every local archive undergoes the same size and digest checks as a download.
    ``prefer_mirror`` tries a listed mirror first, with the same digest check.
    Cancellation and errors preserve the previously active generation.
    """
    status = pack_status()
    if not status.supported:
        raise RuntimeError(f'Unsupported inference pack hardware: {status.reason}')
    if not _WORKER.is_file():
        raise FileNotFoundError(_WORKER)
    _check(cancelled)
    root = pack_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _install_lock(root):
        _remove_retired(root, _active(root).get('generation'))
        if shutil.disk_usage(root).free < status.required_free_bytes:
            raise OSError(f'Installation requires {status.required_free_bytes:,} free bytes at {root}.')
        runtime = root / f'gen-{uuid.uuid4().hex}'
        runtime.mkdir(mode=0o700)
        complete = 0
        transfer_phase = 'download' if artifact_dir is None else 'local'

        def notify(phase, name='', done=0, total=0):
            if progress is not None:
                progress(phase, name, done, total)

        try:
            for artifact in _manifest()['artifacts']:
                _check(cancelled)
                name = unquote(urlsplit(artifact['url']).path.rsplit('/', 1)[1])
                archive = runtime / name
                def transfer_progress(count):
                    notify(transfer_phase, name, complete + count, status.download_bytes)

                transfer_progress(0)
                local = Path(artifact_dir) / name if artifact_dir is not None else None
                _fetch(artifact, archive, local, cancelled, transfer_progress,
                       prefer_mirror=prefer_mirror)
                complete += artifact['size']
                _extract(archive, runtime, cancelled,
                         lambda done, total: notify('unpack', name, done, total))
                archive.unlink()
            # Preserve the desktop's app-local C++ runtime for a portable PC
            # without a system-wide redistributable, including after upgrades.
            if getattr(sys, 'frozen', False):
                for pattern in ('msvcp140*.dll', 'vcruntime140*.dll'):
                    for library in _CORE.parent.glob(pattern):
                        shutil.copyfile(library, runtime / library.name)
            (runtime / 'python310._pth').write_text('python310.zip\n.\nLib/site-packages\n', encoding='ascii')
            notify('probe')
            _probe(runtime, status.backends, cancelled)
            pointer = runtime / 'activation.json'
            pointer.write_text(json.dumps({'version': status.version, 'generation': runtime.name}), encoding='utf-8')
            _check(cancelled)
            os.replace(pointer, root / 'active.json')
        except BaseException:
            if _active(root).get('generation') != runtime.name:
                shutil.rmtree(runtime, ignore_errors=True)
            raise
        _remove_retired(root, runtime.name)
    return pack_status()
