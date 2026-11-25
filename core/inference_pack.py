"""Install the optional inference worker into the current user's profile.

Official URLs, byte counts and SHA-256 hashes are pinned by the application.
A working generation remains selected until its replacement passes the worker's
GPU startup probe. The desktop's Python environment is never modified.
"""
from __future__ import annotations

import ctypes
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import zipfile
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from constants import app_display_version
from utils.app_paths import is_portable_mode, portable_data_dir, user_data_dir
from utils.logging_setup import get_logger
from utils.tls import https_handler

_CORE = Path(__file__).resolve().parent
_WORKER = _CORE / 'inference_worker.py'
_GENERATION = re.compile(r'gen-[0-9a-f]{32}')
_BLOCK = 1024 * 1024
_MIRROR_CATALOG = 'https://account.openfocus.top/v1/runtime-mirrors'
_CATALOG_MAX_BYTES = 64 * 1024
_COUNTRY = re.compile(r'[A-Z]{2}')
_SHA256 = re.compile(r'[0-9a-f]{64}')
_MANIFESTS = {
    'win32': 'accelerator_win64.json',
    'linux': 'accelerator_linux_x86_64.json',
    'darwin': 'accelerator_macos_arm64.json',
}
_UNIX_PYTHON = Path('python/bin/python3.11')
_TAR_MAX_BYTES = 256 * _BLOCK
_TAR_MAX_MEMBERS = 32768
_log = get_logger(__name__)


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


def _platform() -> str:
    machine = platform.machine().lower()
    if sys.maxsize > 2**32:
        if sys.platform in ('win32', 'linux') and machine in ('amd64', 'x86_64'):
            return sys.platform
        if sys.platform == 'darwin' and machine == 'arm64':
            return 'darwin'
    return ''


@lru_cache(maxsize=len(_MANIFESTS))
def _manifest(target):
    return json.loads((_CORE / _MANIFESTS[target]).read_text(encoding='utf-8'))


def pack_root() -> Path:
    """Keep portable and explicitly isolated profiles self-contained."""
    if sys.platform != 'win32' or os.environ.get('OPENFOCUS_DATA_DIR'):
        base = user_data_dir()
    elif is_portable_mode():
        base = Path(portable_data_dir())
    else:
        base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local') / 'OpenFocus'
    return base / 'inference-packs'


def _hardware(target) -> tuple[str, str]:
    if target == 'darwin':
        version = re.match(r'[0-9]+', platform.mac_ver()[0])
        return ('Apple Silicon', '') if version and int(version[0]) >= 13 else ('', 'os_version')
    if target == 'linux':
        try:
            libc = os.confstr('CS_GNU_LIBC_VERSION') or ''
        except (AttributeError, OSError, ValueError):
            libc = ''
        version = re.fullmatch(r'glibc ([0-9]+)\.([0-9]+)', libc)
        if not version or tuple(map(int, version.groups())) < (2, 17):
            return '', 'libc'
    try:
        if target == 'win32':
            # Load the display driver's copy even in a frozen application.
            driver = ctypes.WinDLL('nvcuda.dll', winmode=0x800)  # LOAD_LIBRARY_SEARCH_SYSTEM32
        else:
            driver = ctypes.CDLL('libcuda.so.1')
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
        if driver.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device):
            return gpu, 'architecture'
        capability = major.value, minor.value
        if target == 'win32':
            supported = capability in {(7, 5), (8, 0), (8, 6), (8, 9), (9, 0)}
        else:
            # PyTorch 2.6 cu124 builds for sm_50, sm_60, sm_70, sm_75,
            # sm_80, sm_86 and sm_90. CUDA cubins support newer minors.
            supported = 5 <= major.value <= 8 or capability == (9, 0)
        if not supported:
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


def _runtime_command(runtime, target):
    if not _WORKER.is_file():
        return None
    try:
        meta = runtime.lstat()
        if not stat.S_ISDIR(meta.st_mode) or (
            getattr(meta, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            return None
        if target == 'win32':
            required = tuple(runtime / name for name in ('python.exe', 'python310.dll', 'python310._pth'))
        else:
            required = (runtime / _UNIX_PYTHON,)
        for path in required:
            if not path.is_file() or not path.resolve().is_relative_to(runtime):
                return None
        python = required[0]
        if target != 'win32' and not os.access(python, os.X_OK):
            return None
    except (OSError, RuntimeError):
        return None
    return [str(python), '-I', '-u', str(_WORKER)]


def _activated_command(target, manifest):
    root = pack_root().resolve()
    receipt = _active(root)
    if receipt.get('version') != manifest['version']:
        return None
    return _runtime_command(root / receipt['generation'], target)


def worker_command() -> list[str] | None:
    """Resolve only an activated private runtime and this application's worker."""
    target = _platform()
    return _activated_command(target, _manifest(target)) if target else None


def pack_status() -> PackStatus:
    """Offline recommendation for device zero, the same device used by the worker."""
    target = _platform()
    if not target:
        return PackStatus(False, '', 'platform', 0, False, (), '', 0)
    device, reason = _hardware(target)
    manifest = _manifest(target)
    return PackStatus(not reason, device, reason, sum(a['size'] for a in manifest['artifacts']),
                      _activated_command(target, manifest) is not None, tuple(manifest['backends']),
                      manifest['version'], manifest['required_free_bytes'])


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise PackCancelled()


@contextmanager
def _install_lock(root):
    # OS-owned locks are released even if an installer crashes.
    with (root / 'install.lock').open('a+b') as lock:
        try:
            if _platform() == 'win32':
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('An inference pack installation is already running.') from exc
        yield


def _remove_retired(root, keep, target):
    for path in root.glob('gen-*'):
        if path.name == keep or not _GENERATION.fullmatch(path.name):
            continue
        try:
            if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                continue
            if target == 'win32':
                # Windows won't delete the executable while a worker is using it.
                (path / 'python.exe').unlink(missing_ok=True)
            else:
                import fcntl
                try:
                    binary = (path / _UNIX_PYTHON).open('rb')
                except FileNotFoundError:
                    pass  # An interrupted extraction never produced the executable.
                else:
                    with binary:
                        # The worker holds a shared lock for its whole lifetime.
                        fcntl.flock(binary.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        shutil.rmtree(path, ignore_errors=True)
                    continue
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)


def _windows_region() -> str:
    if sys.platform != 'win32':
        return ''
    try:
        get_region = ctypes.windll.kernel32.GetUserDefaultGeoName
        region = ctypes.create_unicode_buffer(16)
        return region.value if get_region(region, len(region)) else ''
    except (AttributeError, OSError):
        return ''


def _https_url(url):
    if not isinstance(url, str) or '\\' in url or any(not '!' <= c <= '~' for c in url):
        return False
    try:
        parts = urlsplit(url)
        return (parts.scheme == 'https' and bool(parts.hostname) and parts.username is None
                and '%' not in parts.netloc and parts.port != 0 and not parts.fragment)
    except ValueError:
        return False


class _HTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _https_url(newurl):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@lru_cache(maxsize=1)
def _opener():
    return build_opener(_HTTPSRedirect(), https_handler())


def _network_failure(exc):
    # Exception messages can contain signed URLs and proxy configuration.
    if isinstance(exc, HTTPError):
        exc.close()
        return f'HTTP {exc.code}'
    reason = getattr(exc, 'reason', exc)
    return 'connection failure' if isinstance(reason, str) else type(reason).__name__


def _mirror_catalog(cancelled):
    _check(cancelled)
    try:
        request = Request(_MIRROR_CATALOG, headers={
            'Accept': 'application/json',
            'User-Agent': f'OpenFocus/{app_display_version()}',
        })
        with _opener().open(request, timeout=3) as response:
            if response.status != 200:
                raise ValueError('unexpected catalog status')
            body = response.read(_CATALOG_MAX_BYTES + 1)
        if len(body) > _CATALOG_MAX_BYTES:
            raise ValueError('catalog too large')
        data = json.loads(body.decode('utf-8'))
        country, mirrors = data['country'], data['mirrors']
        if country is not None and (not isinstance(country, str) or not _COUNTRY.fullmatch(country)):
            raise ValueError('invalid country')
        if not isinstance(mirrors, dict) or not all(
            _SHA256.fullmatch(sha) and isinstance(urls, list) and all(map(_https_url, urls))
            for sha, urls in mirrors.items()
        ):
            raise ValueError('invalid mirrors')
        _check(cancelled)
        return country, mirrors
    except (OSError, HTTPException) as exc:
        failure = _network_failure(exc)
    except (KeyError, TypeError, ValueError, RecursionError):
        failure = 'invalid catalog response'
    _check(cancelled)
    _log.warning('Runtime mirror catalog at %s failed (%s); using bundled mirrors',
                 urlsplit(_MIRROR_CATALOG).hostname, failure)
    return None


def _download_sources(artifacts, catalog, prefer_mirror):
    country, mirrors = catalog if catalog is not None else (None, None)
    if prefer_mirror is not None:
        mirror_first, basis = prefer_mirror, 'manual selection'
    elif country in (None, 'XX'):
        mirror_first, basis = _windows_region() == 'CN', 'Windows region'
    else:
        mirror_first, basis = country == 'CN', 'network region'
    _log.info('Runtime download priority: %s first (%s)',
              'mirror' if mirror_first else 'official', basis)
    for artifact in artifacts:
        official = artifact['url']
        urls = artifact.get('mirrors', ()) if mirrors is None else mirrors.get(artifact['sha256'], ())
        distinct_mirrors = tuple(dict.fromkeys(url for url in urls if url != official))
        yield artifact, (*distinct_mirrors, official) if mirror_first else (official, *distinct_mirrors)


class _InvalidTransfer(ValueError):
    """Response metadata or body conflicts with the requested byte range."""


class _ArchiveWriteError(OSError):
    """A local filesystem failure must not cause retries against other hosts."""


class _Transfer:
    def __init__(self, dest, expected, cancelled, report):
        self.dest = dest
        self.expected = expected
        self.cancelled = cancelled
        self.report = report
        self.count = 0
        self.digest = hashlib.sha256()
        self.last_update = 0.0

    def rewind(self, count=0, digest=None):
        if self.count != count:
            self.dest.seek(count)
            self.dest.truncate()
            self.count = count
            self.digest = hashlib.sha256() if digest is None else digest
            self.report(count)

    def copy(self, read):
        while True:
            _check(self.cancelled)
            chunk = read(min(_BLOCK, self.expected - self.count + 1))
            _check(self.cancelled)
            if not chunk:
                return
            if len(chunk) > self.expected - self.count:
                raise _InvalidTransfer('body exceeds the pinned byte count')
            try:
                self.dest.write(chunk)
            except OSError as exc:
                raise _ArchiveWriteError(f'Could not write {Path(self.dest.name).name}: {exc.strerror}') from exc
            self.digest.update(chunk)
            self.count += len(chunk)
            now = time.monotonic()
            if now - self.last_update >= 0.15:
                self.report(self.count)
                self.last_update = now


def _download(artifact, urls, transfer):
    pending = deque(urls)
    no_range = deque()
    official = artifact['url']
    official_retries = 3
    size = artifact['size']
    while pending or no_range:
        _check(transfer.cancelled)
        if pending:
            url = pending.popleft()
        else:
            url = no_range.popleft()
            transfer.rewind()
        # A failed EOF check after the last byte still requires a fresh response.
        if transfer.count == size:
            transfer.rewind()
        offset = transfer.count
        prefix_digest = transfer.digest.copy()
        host = urlsplit(url).hostname
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with _opener().open(Request(url, headers=headers), timeout=20) as source:
                final_host = urlsplit(source.geturl()).hostname
                if final_host != host:
                    _log.info('Runtime download redirected from %s to %s', host, final_host)
                    host = final_host
                if offset and source.status == 200:
                    # Keep the existing prefix while other hosts can still resume it.
                    if url not in no_range:
                        no_range.append(url)
                    raise _InvalidTransfer('server ignored the byte range')
                if offset:
                    if (source.status != 206 or source.headers.get('Content-Range') !=
                            f'bytes {offset}-{size - 1}/{size}'):
                        raise _InvalidTransfer('unexpected status or Content-Range for a resumed download')
                elif source.status != 200 or source.headers.get('Content-Range') is not None:
                    raise _InvalidTransfer('unexpected status or Content-Range for a full download')
                length = source.headers.get('Content-Length')
                if length is not None and length.strip().lstrip('0') != str(size - offset):
                    raise _InvalidTransfer('Content-Length does not match the requested byte count')
                if source.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                    raise _InvalidTransfer('server encoded the artifact bytes')
                transfer.copy(source.read1)
                if transfer.count != size:
                    raise EOFError()
        except _InvalidTransfer as exc:
            transfer.rewind(offset, prefix_digest)
            failure = str(exc)
        except _ArchiveWriteError:
            raise
        except (OSError, HTTPException, EOFError) as exc:
            if isinstance(exc, HTTPError):
                final_host = urlsplit(exc.geturl()).hostname
                if final_host and final_host != host:
                    host = f'{host} via {final_host}'
            failure = ('response ended before the requested byte count' if isinstance(exc, EOFError)
                       else _network_failure(exc))
            if (url == official and official_retries
                    and (not isinstance(exc, HTTPError) or exc.code >= 500)):
                pending.append(url)
                official_retries -= 1
        else:
            if transfer.digest.hexdigest() == artifact['sha256']:
                return
            failure = 'SHA-256 mismatch'
            transfer.rewind()
            if offset:
                # The corrupt bytes may be from an earlier host; try this one on its own.
                pending.appendleft(url)
        _check(transfer.cancelled)
        if pending or no_range:
            next_url = pending[0] if pending else no_range[0]
            _log.warning('Runtime download from %s failed (%s); %s %s', host, failure,
                         'retrying at' if next_url == url else 'trying', urlsplit(next_url).hostname)
        else:
            _log.warning('Runtime download from %s failed (%s); no sources left', host, failure)
            raise ValueError(f'Download failed for {Path(transfer.dest.name).name} '
                             f'from {host}: {failure}.')


def _fetch(artifact, archive, local, cancelled, report, *, urls):
    expected = artifact['size']
    if shutil.disk_usage(archive.parent).free < expected:
        raise OSError(f'Insufficient disk space for {archive.name} ({expected:,} bytes).')
    with archive.open('xb') as dest:
        transfer = _Transfer(dest, expected, cancelled, report)
        if local is None:
            _download(artifact, urls, transfer)
        else:
            with local.open('rb') as source:
                try:
                    transfer.copy(source.read)
                except _InvalidTransfer as exc:
                    raise ValueError(f'Archive exceeds the pinned size for {archive.name}.') from exc
            if transfer.count != expected or transfer.digest.hexdigest() != artifact['sha256']:
                raise ValueError(f'Archive size or SHA-256 mismatch for {archive.name}.')
    report(transfer.count)


def _archive_parts(name, *, relative_link=False):
    parts = PurePosixPath(name).parts
    if not parts or name.startswith('/') or any(':' in p or '\\' in p or '\x00' in p for p in parts):
        raise ValueError(f'Invalid archive path: {name!r}')
    if not relative_link and '..' in parts:
        raise ValueError(f'Invalid archive path: {name!r}')
    return parts


class _TarStream:
    """Bound the decompressor itself, including tar metadata and padding."""

    def __init__(self, source, cancelled):
        self.source = source
        self.cancelled = cancelled
        self.remaining = _TAR_MAX_BYTES

    def read(self, size):
        _check(self.cancelled)
        data = self.source.read(min(size, self.remaining + 1))
        self.remaining -= len(data)
        if self.remaining < 0:
            raise ValueError('Standalone Python archive exceeds the unpack limit.')
        return data


def _extract_python(archive, runtime, required, cancelled, report):
    if not 0 < required <= _TAR_MAX_BYTES:
        raise ValueError('Invalid standalone Python unpack size.')
    if shutil.disk_usage(runtime).free < required:
        raise OSError(f'Insufficient disk space to unpack {archive.name} ({required:,} bytes).')
    prefix = runtime / 'python'
    seen = set()
    links = []
    written = 0
    last_update = time.monotonic()
    report(0, required)
    with gzip.open(archive, 'rb') as compressed:
        stream = _TarStream(compressed, cancelled)
        with tarfile.open(fileobj=stream, mode='r|') as source:
            for member in source:
                _check(cancelled)
                parts = _archive_parts(member.name)
                if parts[0] != 'python' or parts in seen or len(seen) >= _TAR_MAX_MEMBERS:
                    raise ValueError(f'Invalid standalone Python archive entry: {member.name!r}')
                seen.add(parts)
                dest = runtime.joinpath(*parts)
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif member.issym() and not member.size:
                    links.append((member.name, dest, _archive_parts(member.linkname, relative_link=True)))
                elif member.isreg() and not member.issparse() and 0 <= member.size <= required - written:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as inp, dest.open('xb') as out:
                        while chunk := inp.read(_BLOCK):
                            _check(cancelled)
                            out.write(chunk)
                            written += len(chunk)
                            now = time.monotonic()
                            if now - last_update >= 0.15:
                                report(written, required)
                                last_update = now
                    if member.mode & 0o111:
                        dest.chmod(0o755)
                else:
                    raise ValueError(f'Invalid standalone Python archive entry: {member.name!r}')
        while stream.read(_BLOCK):
            pass
    if written != required:
        raise ValueError('Standalone Python archive does not match the pinned unpack size.')
    # Create links last so an archive member cannot write through one. The
    # standalone builds only need relative links to other files in the prefix.
    for name, dest, target in links:
        _check(cancelled)
        if dest.parent.resolve() != dest.parent:
            raise ValueError(f'Invalid standalone Python link parent: {name!r}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Check both lexical traversal and the final physical resolution.
        if dest == prefix:
            raise ValueError(f'Invalid standalone Python symlink: {name!r}')
        level = len(dest.parent.relative_to(prefix).parts)
        for part in target:
            level += -1 if part == '..' else 1
            if level < 0:
                raise ValueError(f'Invalid standalone Python symlink: {name!r}')
        dest.symlink_to(Path(*target))
    for name, dest, _ in links:
        _check(cancelled)
        try:
            real = dest.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f'Invalid standalone Python link: {name!r}') from exc
        if not real.is_relative_to(prefix) or not real.is_file():
            raise ValueError(f'Invalid standalone Python link: {name!r}')
    report(written, required)


def _extract_zip(archive, runtime, target, cancelled, report, max_bytes):
    wheel = archive.suffix == '.whl'
    windows = target == 'win32'
    prefix = runtime if windows else runtime / 'python'
    site = prefix / ('Lib/site-packages' if windows else 'lib/python3.11/site-packages')
    scheme = {'purelib': site, 'platlib': site, 'data': prefix,
              'scripts': prefix / ('Scripts' if windows else 'bin'),
              'headers': prefix / ('Include' if windows else 'include')}
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        required = sum(item.file_size for item in members)
        if required > max_bytes:
            raise ValueError(f'Archive {archive.name} exceeds the unpack limit.')
        if shutil.disk_usage(runtime).free < required:
            raise OSError(f'Insufficient disk space to unpack {archive.name} ({required:,} bytes).')
        written = 0
        last_update = time.monotonic()
        report(0, required)
        for item in members:
            _check(cancelled)
            if item.is_dir():
                continue
            parts = _archive_parts(item.filename)
            base = site if wheel else runtime
            if wheel and parts[0].endswith('.data'):
                if len(parts) < 3 or parts[1] not in scheme:
                    raise ValueError(f'Invalid wheel path in {archive.name}: {item.filename}')
                base = scheme[parts[1]]
                parts = parts[2:]
            mode = item.external_attr >> 16
            if not windows and stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ValueError(f'Invalid wheel file type in {archive.name}: {item.filename}')
            dest = base.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if target == 'linux' and dest == site / 'nvidia/__init__.py' and item.file_size == 0:
                # The pinned NVIDIA wheels each ship the same empty namespace file.
                if dest.is_file() and not dest.is_symlink() and dest.stat().st_size == 0:
                    continue
            with source.open(item) as inp, dest.open('xb') as out:
                while chunk := inp.read(_BLOCK):
                    _check(cancelled)
                    out.write(chunk)
                    written += len(chunk)
                    now = time.monotonic()
                    if now - last_update >= 0.15:
                        report(written, required)
                        last_update = now
            if not windows and mode & 0o111:
                dest.chmod(0o755)
        report(written, required)


def _probe(runtime, target, backends, cancelled):
    from core.inference_process import spawn
    command = _runtime_command(runtime, target)
    if command is None:
        raise RuntimeError('The inference worker runtime is incomplete.')
    if target != 'win32':
        command.append('--probe')
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
                 prefer_mirror: bool | None = None) -> PackStatus:
    """Install or repair, reporting progress as (phase, name, done, total).

    Download and local verification counts are cumulative archive bytes.
    Unpack counts are per archive's uncompressed bytes. GPU verification has
    no known total.

    ``artifact_dir`` optionally supplies original archives for an offline install.
    Every local archive undergoes the same size and digest checks as a download.
    ``prefer_mirror`` defaults to automatic region selection, using Windows when
    the network region is unknown. True prioritizes mirrors; False prioritizes
    official sources.
    Cancellation and errors preserve the previously active generation.
    """
    status = pack_status()
    if not status.supported:
        raise RuntimeError(f'Unsupported inference pack hardware: {status.reason}')
    target = _platform()
    if not _WORKER.is_file():
        raise FileNotFoundError(_WORKER)
    _check(cancelled)
    root = pack_root().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _install_lock(root):
        _remove_retired(root, _active(root).get('generation'), target)
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
            artifacts = _manifest(target)['artifacts']
            if artifact_dir is None:
                sources = _download_sources(artifacts, _mirror_catalog(cancelled), prefer_mirror)
            else:
                sources = ((artifact, ()) for artifact in artifacts)
            for artifact, urls in sources:
                _check(cancelled)
                name = unquote(urlsplit(artifact['url']).path.rsplit('/', 1)[1])
                archive = runtime / name

                def transfer_progress(count):
                    notify(transfer_phase, name, complete + count, status.download_bytes)

                transfer_progress(0)
                local = Path(artifact_dir) / name if artifact_dir is not None else None
                _fetch(artifact, archive, local, cancelled, transfer_progress, urls=urls)
                complete += artifact['size']

                def unpack_progress(done, total):
                    notify('unpack', name, done, total)

                if target != 'win32' and name.endswith('.tar.gz'):
                    _extract_python(archive, runtime, artifact['unpacked_size'], cancelled, unpack_progress)
                else:
                    _extract_zip(archive, runtime, target, cancelled, unpack_progress, status.required_free_bytes)
                archive.unlink()
            if target == 'win32':
                # Preserve the desktop's app-local C++ runtime for portable PCs.
                if getattr(sys, 'frozen', False):
                    for pattern in ('msvcp140*.dll', 'vcruntime140*.dll'):
                        for library in _CORE.parent.glob(pattern):
                            shutil.copyfile(library, runtime / library.name)
                (runtime / 'python310._pth').write_text('python310.zip\n.\nLib/site-packages\n', encoding='ascii')
            notify('probe')
            _probe(runtime, target, status.backends, cancelled)
            pointer = runtime / 'activation.json'
            pointer.write_text(json.dumps({'version': status.version, 'generation': runtime.name}), encoding='utf-8')
            _check(cancelled)
            os.replace(pointer, root / 'active.json')
        except BaseException:
            if _active(root).get('generation') != runtime.name:
                shutil.rmtree(runtime, ignore_errors=True)
            raise
        _remove_retired(root, runtime.name, target)
    return pack_status()
