"""Managed pack transactions with tiny local archives and no GPU or network."""
from contextlib import nullcontext
import gzip
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import stat
import subprocess
import tarfile
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
import zipfile

import pytest

from core import inference_pack as pack

_OFFICIAL = 'https://download.pytorch.org/whl/example.whl'
_BUNDLED = 'https://mirrors.aliyun.com/pytorch-wheels/example.whl'
_MIRROR = 'https://mirror.example/torch.whl'
_SECOND = 'https://backup.example/torch.whl'
_PAYLOAD = b'abcdefghij'


def _response(url, payload, status=200, **headers):
    stream = io.BytesIO(payload)
    stream.status = status
    stream.headers = headers
    stream.geturl = lambda: url
    return stream


@pytest.fixture
def artifact():
    return {'url': _OFFICIAL, 'mirrors': [_BUNDLED], 'size': len(_PAYLOAD),
            'sha256': hashlib.sha256(_PAYLOAD).hexdigest()}


@pytest.fixture
def transport(monkeypatch):
    def configure(handler):
        monkeypatch.setattr(pack, '_opener', lambda: SimpleNamespace(open=handler))
    return configure


@pytest.fixture
def network_log(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger('openfocus'), 'propagate', True)
    return caplog


def test_production_pack_has_pinned_binary_sources():
    manifest = pack._manifest('win32')
    origins = {'www.python.org', 'download.pytorch.org', 'pypi.nvidia.com', 'files.pythonhosted.org'}
    for artifact in manifest['artifacts']:
        url = urlsplit(artifact['url'])
        assert url.scheme == 'https' and url.hostname in origins
        assert url.path.endswith(('.whl', '.zip'))
        assert artifact['size'] > 0 and len(bytes.fromhex(artifact['sha256'])) == 32
        assert 'onnxruntime' not in url.path
    assert tuple(manifest['backends']) == ('TENSORRT', 'TORCH_CUDA')
    torch, = (a for a in manifest['artifacts'] if urlsplit(a['url']).hostname == 'download.pytorch.org')
    assert torch['size'] == 2510729365
    assert torch['sha256'] == '6f99d8459369cfd6661c2aee14787592fe50156a33faf9ef643ba04e42d6543f'
    wheel = 'torch-2.5.1%2Bcu124-cp310-cp310-win_amd64.whl'
    assert torch['mirrors'] == [
        f'https://mirrors.aliyun.com/pytorch-wheels/cu124/{wheel}',
        f'https://mirror.nju.edu.cn/pytorch/whl/cu124/{wheel}',
        f'https://mirror.sjtu.edu.cn/pytorch-wheels/cu124/{wheel}',
    ]


@pytest.mark.parametrize(('target', 'backend', 'abi', 'torch_size', 'total'), [
    ('linux', 'TORCH_CUDA', 'cp311-cp311-linux_x86_64', 768460593, 3128517661),
    ('darwin', 'TORCH_MPS', 'cp311-none-macosx_11_0_arm64', 66530713, 116468949),
])
def test_unix_manifests_pin_platform_artifacts(target, backend, abi, torch_size, total):
    manifest = pack._manifest(target)
    assert manifest['backends'] == [backend]
    artifacts = manifest['artifacts']
    assert sum(a['size'] for a in artifacts) == total
    assert urlsplit(artifacts[0]['url']).hostname == 'github.com'
    assert artifacts[0]['url'].endswith('-install_only.tar.gz')
    assert artifacts[0]['unpacked_size'] > artifacts[0]['size']
    for artifact in artifacts:
        url = urlsplit(artifact['url'])
        assert url.scheme == 'https'
        assert url.hostname in {'github.com', 'files.pythonhosted.org', 'download-r2.pytorch.org'}
        assert url.path.endswith(('.tar.gz', '.whl'))
        assert artifact['size'] > 0 and len(bytes.fromhex(artifact['sha256'])) == 32
    torch, = (a for a in artifacts if '/torch-' in a['url'])
    assert torch['size'] == torch_size and torch['url'].endswith(f'-{abi}.whl')
    assert torch['mirrors']
    cuda_libraries = [a for a in artifacts if '/nvidia_' in a['url']]
    assert len(cuda_libraries) == (13 if target == 'linux' else 0)


@pytest.mark.parametrize(('host', 'machine', 'bits', 'target'), [
    ('win32', 'AMD64', 64, 'win32'),
    ('win32', 'AMD64', 32, ''),
    ('linux', 'x86_64', 64, 'linux'),
    ('linux', 'aarch64', 64, ''),
    ('darwin', 'arm64', 64, 'darwin'),
    ('darwin', 'x86_64', 64, ''),
    ('freebsd14', 'amd64', 64, ''),
])
def test_pack_rules_select_only_supported_builds(monkeypatch, host, machine, bits, target):
    monkeypatch.setattr(pack, 'sys', SimpleNamespace(platform=host, maxsize=2 ** (bits - 1) - 1))
    monkeypatch.setattr(pack.platform, 'machine', lambda: machine)
    assert pack._platform() == target
    if not target:
        monkeypatch.setattr(pack, '_manifest', lambda *_: pytest.fail('unsupported platform selected a manifest'))
        status = pack.pack_status()
        assert not status.supported and not status.installed and status.reason == 'platform'
        assert status.download_bytes == 0 and status.backends == ()
        assert pack.worker_command() is None
        with pytest.raises(RuntimeError, match='platform'):
            pack.install_pack()


@pytest.mark.parametrize(('version', 'supported'), [('12.7.6', False), ('13.0', True), ('26.1', True)])
def test_macos_mps_floor(monkeypatch, version, supported):
    monkeypatch.setattr(pack.platform, 'mac_ver', lambda: (version, (), 'arm64'))
    assert (pack._hardware('darwin')[1] == '') == supported


@pytest.mark.parametrize('libc', ['glibc 2.16', 'musl 1.2', ''])
def test_linux_rejects_unsupported_libc_before_loading_cuda(monkeypatch, libc):
    monkeypatch.setattr(pack.os, 'confstr', lambda _: libc, raising=False)
    monkeypatch.setattr(pack.ctypes, 'CDLL', lambda _: pytest.fail('unsupported OS tried loading CUDA'))
    assert pack._hardware('linux') == ('', 'libc')


@pytest.mark.parametrize(('driver_version', 'capability', 'result'), [
    (12030, (8, 9), 'driver'),
    (12040, (3, 7), 'architecture'),
    (12040, (5, 2), ''),
    (12040, (8, 9), ''),
    (12080, (9, 0), ''),
    (12080, (12, 0), 'architecture'),
])
def test_linux_uses_cuda_driver_device_zero(monkeypatch, driver_version, capability, result):
    monkeypatch.setattr(pack.os, 'confstr', lambda _: 'glibc 2.17', raising=False)

    def get_device(_device, which):
        assert which == 0
        return 0

    def get_version(out):
        out._obj.value = driver_version
        return 0

    def get_capability(major, minor, _device):
        major._obj.value, minor._obj.value = capability
        return 0

    def get_name(out, *_):
        out.value = b'Test GPU'
        return 0

    def load(library):
        assert library == 'libcuda.so.1'
        return SimpleNamespace(cuInit=lambda _: 0, cuDeviceGet=get_device, cuDriverGetVersion=get_version,
                               cuDeviceComputeCapability=get_capability, cuDeviceGetName=get_name)

    monkeypatch.setattr(pack.ctypes, 'CDLL', load)
    assert pack._hardware('linux') == ('Test GPU', result)


@pytest.fixture
def transaction(tmp_path, monkeypatch):
    sources = tmp_path / 'archives'
    sources.mkdir()
    archive = sources / 'embed.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name in ('python.exe', 'python310.dll', 'python310._pth'):
            z.writestr(name, 'input')
    data = archive.read_bytes()
    manifest = {'version': 'test', 'backends': ['TORCH_CUDA'], 'required_free_bytes': 4096,
                'artifacts': [{'url': 'https://www.python.org/embed.zip', 'size': len(data),
                               'sha256': hashlib.sha256(data).hexdigest()}]}
    root = tmp_path / 'packs'
    monkeypatch.setattr(pack, '_platform', lambda: 'win32')
    monkeypatch.setattr(pack, '_manifest', lambda _: manifest)
    monkeypatch.setattr(pack, '_hardware', lambda _: ('test device', ''))
    monkeypatch.setattr(pack, 'pack_root', lambda: root)
    monkeypatch.setattr(pack, '_WORKER', Path(__file__).resolve())
    monkeypatch.setattr(pack, '_install_lock', lambda _: nullcontext())
    monkeypatch.setattr(pack, '_probe', lambda *args: None)
    return root, sources


@pytest.mark.parametrize('failure', ['cancel', 'hash', 'probe'])
def test_failed_repair_keeps_working_generation(transaction, monkeypatch, failure):
    root, sources = transaction
    pack.install_pack(artifact_dir=sources)
    before = (root / 'active.json').read_bytes()
    generation = json.loads(before)['generation']
    if failure == 'hash':
        (sources / 'embed.zip').write_bytes(b'corrupt')
    if failure == 'probe':
        def fail(*args):
            raise RuntimeError('worker unavailable')
        monkeypatch.setattr(pack, '_probe', fail)
    cancelling = False

    def progress(phase, name, done, total):
        nonlocal cancelling
        cancelling = failure == 'cancel' and phase == 'unpack'

    with pytest.raises({'cancel': pack.PackCancelled, 'hash': ValueError, 'probe': RuntimeError}[failure]):
        pack.install_pack(artifact_dir=sources, progress=progress, cancelled=lambda: cancelling)
    assert (root / 'active.json').read_bytes() == before
    assert list(root.glob('gen-*')) == [root / generation]
    assert (root / generation / 'python.exe').is_file()


def test_receipt_stays_inside_pack_root(tmp_path):
    (tmp_path / 'active.json').write_text(json.dumps({'version': 'test', 'generation': '../other'}))
    assert pack._active(tmp_path) == {}


def test_successful_repair_replaces_generation(transaction):
    root, sources = transaction
    pack.install_pack(artifact_dir=sources)
    old = pack._active(root)['generation']
    pack.install_pack(artifact_dir=sources)
    new = pack._active(root)['generation']
    assert new != old
    assert not (root / old).exists()
    assert (root / new / 'python310._pth').read_text().splitlines() == [
        'python310.zip', '.', 'Lib/site-packages']
    assert pack.worker_command() == [str(root / new / 'python.exe'), '-I', '-u', str(pack._WORKER)]


def test_unsupported_hardware_does_not_start_install(transaction, monkeypatch):
    root, sources = transaction
    monkeypatch.setattr(pack, '_hardware', lambda _: ('unsupported device', 'architecture'))
    with pytest.raises(RuntimeError, match='architecture'):
        pack.install_pack(artifact_dir=sources)
    assert not root.exists()


def test_offline_install_never_opens_network(transaction, transport):
    _, sources = transaction
    transport(lambda *args, **kwargs: pytest.fail('offline install attempted a network request'))
    phases = []
    pack.install_pack(artifact_dir=sources, progress=lambda phase, *_: phases.append(phase))
    assert {'local', 'unpack', 'probe'} <= set(phases)


def test_online_install_uses_catalog_and_keeps_pinned_archive_name(transaction, transport):
    root, sources = transaction
    artifact, = pack._manifest('win32')['artifacts']
    requests = []
    archive = (sources / 'embed.zip').read_bytes()

    def open_transfer(request, timeout):
        url = request.full_url
        requests.append(url)
        if url == pack._MIRROR_CATALOG:
            body = json.dumps({'country': 'CN', 'mirrors': {artifact['sha256']: [_MIRROR]}}).encode()
            return _response(url, body)
        assert url == _MIRROR
        return _response(url, archive)

    transport(open_transfer)
    pack.install_pack()
    assert requests == [pack._MIRROR_CATALOG, _MIRROR]
    generation = pack._active(root)['generation']
    assert (root / generation / 'python.exe').is_file()


def _tar_file(archive, name, contents, mode=0o644):
    member = tarfile.TarInfo(name)
    member.mode = mode
    member.size = len(contents)
    archive.addfile(member, io.BytesIO(contents))


def _tar_entry(archive, name, kind, target=''):
    member = tarfile.TarInfo(name)
    member.type = kind
    member.linkname = target
    archive.addfile(member)


def _pin(archive, **metadata):
    data = archive.read_bytes()
    return {'url': f'https://example.com/{archive.name}', 'size': len(data),
            'sha256': hashlib.sha256(data).hexdigest(), **metadata}


@pytest.fixture(params=['linux', 'darwin'])
def unix_transaction(request, tmp_path, monkeypatch):
    platform = request.param
    sources = tmp_path / 'offline'
    sources.mkdir()
    interpreter = b'#!/bin/sh\nread line\n'
    archive = sources / 'python.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        _tar_file(tar, 'python/bin/python3.11', interpreter, mode=0o4755)
    wheel = sources / 'array.whl'
    with zipfile.ZipFile(wheel, 'w') as z:
        z.writestr('array_module/__init__.py', 'pinned dependency')
        z.writestr('array_module-1.0.data/purelib/array_extra.py', 'pinned Python data')
    backend = 'TORCH_MPS' if platform == 'darwin' else 'TORCH_CUDA'
    manifest = {'version': f'test-{platform}', 'backends': [backend], 'required_free_bytes': 4096,
                'artifacts': [_pin(archive, unpacked_size=len(interpreter)), _pin(wheel)]}
    root = (tmp_path / 'packs').resolve()
    monkeypatch.setattr(pack, '_platform', lambda: platform)
    monkeypatch.setattr(pack, '_manifest', lambda _: manifest)
    monkeypatch.setattr(pack, '_hardware', lambda _: ('test device', ''))
    monkeypatch.setattr(pack, 'pack_root', lambda: root)
    if os.name == 'nt':
        # Only these two host APIs depend on the native OS in the Unix transaction.
        monkeypatch.setattr(pack, '_install_lock', lambda _: nullcontext())
        monkeypatch.setattr(pack, '_remove_retired', lambda *_: None)
    return root, sources, backend


def test_unix_offline_install_uses_same_worker_command(unix_transaction, monkeypatch, transport):
    from core import inference_process

    root, sources, backend = unix_transaction
    commands = []

    def spawn(command, **kwargs):
        commands.append(command)
        assert kwargs['stdin'] == subprocess.DEVNULL
        kwargs['stdout'].write(json.dumps({'ready': True, 'backends': [backend]}).encode())
        return nullcontext(SimpleNamespace(returncode=0, poll=lambda: 0))

    monkeypatch.setattr(inference_process, 'spawn', spawn)
    transport(lambda *args, **kwargs: pytest.fail('offline install attempted a network request'))
    status = pack.install_pack(artifact_dir=sources)
    assert status.installed and status.backends == (backend,)
    runtime, = root.glob('gen-*')
    expected = [str(runtime / 'python/bin/python3.11'), '-I', '-u', str(pack._WORKER)]
    assert pack.worker_command() == expected
    assert commands == [expected + ['--probe']]
    site = runtime / 'python/lib/python3.11/site-packages'
    assert (site / 'array_module/__init__.py').read_text() == 'pinned dependency'
    assert (site / 'array_extra.py').read_text() == 'pinned Python data'
    if os.name != 'nt':
        assert stat.S_IMODE(Path(expected[0]).stat().st_mode) == 0o755


@pytest.mark.skipif(os.name == 'nt', reason='POSIX interpreter symlinks')
def test_unix_receipt_cannot_launch_external_interpreter(unix_transaction, monkeypatch):
    root, sources, _ = unix_transaction
    monkeypatch.setattr(pack, '_probe', lambda *args: None)
    pack.install_pack(artifact_dir=sources)
    runtime = root / pack._active(root)['generation']
    binary = runtime / 'python/bin/python3.11'
    external = root.parent / 'external-python'
    binary.rename(external)
    binary.symlink_to(external)
    assert pack.worker_command() is None
    assert not pack.pack_status().installed


@pytest.mark.skipif(os.name == 'nt', reason='POSIX worker leases')
def test_unix_repair_preserves_live_worker_generation(unix_transaction, monkeypatch):
    from core.inference_process import spawn

    root, sources, _ = unix_transaction
    monkeypatch.setattr(pack, '_probe', lambda *args: None)
    pack.install_pack(artifact_dir=sources)
    old = pack._active(root)['generation']
    with spawn(pack.worker_command(), stdin=subprocess.PIPE) as worker:
        pack.install_pack(artifact_dir=sources)
        assert worker.poll() is None
        assert (root / old / 'python/lib/python3.11/site-packages/array_extra.py').is_file()
        assert pack._active(root)['generation'] != old
        worker.communicate(b'release\n', timeout=5)
    pack.install_pack(artifact_dir=sources)
    assert not (root / old).exists()
    assert len(list(root.glob('gen-*'))) == 1


@pytest.mark.skipif(os.name == 'nt', reason='POSIX installer locks')
def test_posix_installer_lock_is_exclusive_and_released(unix_transaction):
    root, _, _ = unix_transaction
    root.mkdir()
    with pack._install_lock(root):
        with pytest.raises(RuntimeError, match='installation is already running'):
            with pack._install_lock(root):
                pytest.fail('second installer claimed the lock')
    with pack._install_lock(root):
        assert (root / 'install.lock').is_file()


@pytest.mark.parametrize(('name', 'kind', 'target'), [
    ('/outside', tarfile.DIRTYPE, ''),
    ('//outside', tarfile.DIRTYPE, ''),
    ('python/../../outside', tarfile.DIRTYPE, ''),
    ('python/device', tarfile.CHRTYPE, ''),
    ('python/fifo', tarfile.FIFOTYPE, ''),
    ('python/bin/link', tarfile.SYMTYPE, '../../../outside'),
    ('python/bin/link', tarfile.SYMTYPE, '/outside'),
    ('python/bin/link', tarfile.LNKTYPE, 'outside'),
])
def test_tar_rejects_unsafe_paths_and_entry_types(tmp_path, name, kind, target):
    archive = tmp_path / 'runtime.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        _tar_file(tar, 'python/bin/python3.11', b'abc')
        _tar_entry(tar, name, kind, target)
    runtime = (tmp_path / 'runtime').resolve()
    runtime.mkdir()
    with pytest.raises(ValueError, match='Invalid'):
        pack._extract_python(archive, runtime, 3, None, lambda *_: None)
    assert not (tmp_path / 'outside').exists()


@pytest.mark.skipif(os.name == 'nt', reason='POSIX archive symlinks')
def test_tar_keeps_confined_relative_file_links(tmp_path):
    archive = tmp_path / 'runtime.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        _tar_entry(tar, 'python/bin/link', tarfile.SYMTYPE, '../lib/library')
        _tar_file(tar, 'python/lib/library', b'abc')
    runtime = (tmp_path / 'runtime').resolve()
    runtime.mkdir()
    pack._extract_python(archive, runtime, 3, None, lambda *_: None)
    link = runtime / 'python/bin/link'
    assert link.is_symlink() and link.read_bytes() == b'abc'


def test_tar_limits_decompressed_metadata_and_padding(tmp_path, monkeypatch):
    archive = tmp_path / 'runtime.tar.gz'
    with gzip.open(archive, 'wb') as compressed:
        compressed.write(b'\0' * 15000)
    monkeypatch.setattr(pack, '_TAR_MAX_BYTES', 10240)
    with pytest.raises(ValueError, match='unpack limit'):
        pack._extract_python(archive, tmp_path.resolve(), 3, None, lambda *_: None)


def test_linux_wheels_share_empty_namespace_only(tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    for i, content in enumerate((b'', b'', b'conflict')):
        archive = tmp_path / f'cuda{i}.whl'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('nvidia/__init__.py', content)
            z.writestr(f'nvidia/lib{i}/__init__.py', b'abc')
        if content:
            with pytest.raises(FileExistsError):
                pack._extract_zip(archive, runtime, 'linux', None, lambda *_: None, 4096)
        else:
            pack._extract_zip(archive, runtime, 'linux', None, lambda *_: None, 4096)
    site = runtime / 'python/lib/python3.11/site-packages'
    assert (site / 'nvidia/__init__.py').read_bytes() == b''
    assert (site / 'nvidia/lib1/__init__.py').read_bytes() == b'abc'


@pytest.mark.parametrize(('country', 'windows', 'override', 'expected'), [
    ('CN', 'US', None, (_MIRROR, _SECOND, _OFFICIAL)),
    ('US', 'CN', None, (_OFFICIAL, _MIRROR, _SECOND)),
    ('TW', 'CN', None, (_OFFICIAL, _MIRROR, _SECOND)),
    (None, 'CN', None, (_MIRROR, _SECOND, _OFFICIAL)),
    ('XX', 'US', None, (_OFFICIAL, _MIRROR, _SECOND)),
    ('XX', 'CN', None, (_MIRROR, _SECOND, _OFFICIAL)),
    (None, '', None, (_OFFICIAL, _MIRROR, _SECOND)),
    ('US', 'CN', True, (_MIRROR, _SECOND, _OFFICIAL)),
    ('CN', 'US', False, (_OFFICIAL, _MIRROR, _SECOND)),
])
def test_catalog_country_and_manual_priority(artifact, transport, monkeypatch, country, windows,
                                             override, expected):
    def open_catalog(request, timeout):
        assert request.full_url == 'https://account.openfocus.top/v1/runtime-mirrors'
        assert request.get_method() == 'GET'
        assert request.get_header('Authorization') is None
        assert request.get_header('Cookie') is None
        body = json.dumps({'country': country, 'mirrors': {
            artifact['sha256']: [_MIRROR, _SECOND, _MIRROR, _OFFICIAL],
        }})
        return _response(request.full_url, body.encode('utf-8'))

    transport(open_catalog)
    monkeypatch.setattr(pack, '_windows_region', lambda: windows)
    catalog = pack._mirror_catalog(None)
    assert list(pack._download_sources([artifact], catalog, override)) == [(artifact, expected)]


@pytest.mark.parametrize(('region', 'override', 'expected'), [
    ('CN', None, (_BUNDLED, _OFFICIAL)), ('US', None, (_OFFICIAL, _BUNDLED)),
    ('', None, (_OFFICIAL, _BUNDLED)), ('CN', False, (_OFFICIAL, _BUNDLED)),
    ('US', True, (_BUNDLED, _OFFICIAL)),
])
def test_unreachable_catalog_uses_bundled_sources(artifact, transport, monkeypatch, network_log, region,
                                                  override, expected):
    def fail(request, timeout):
        raise URLError(TimeoutError('private proxy configuration'))

    transport(fail)
    monkeypatch.setattr(pack, '_windows_region', lambda: region)
    catalog = pack._mirror_catalog(None)
    assert catalog is None
    assert list(pack._download_sources([artifact], catalog, override)) == [(artifact, expected)]
    assert 'account.openfocus.top failed (TimeoutError); using bundled mirrors' in network_log.text
    assert 'private proxy configuration' not in network_log.text


@pytest.mark.parametrize('entry', ['missing', 'empty', 'other-artifact'])
@pytest.mark.parametrize('country', ['CN', None])
def test_valid_catalog_omission_or_delisting_removes_bundled_mirrors(artifact, transport, country, entry):
    mirrors = {}
    if entry == 'empty':
        mirrors[artifact['sha256']] = []
    elif entry == 'other-artifact':
        mirrors['a' * 64] = [_MIRROR]
    body = json.dumps({'country': country, 'mirrors': mirrors}).encode()
    transport(lambda request, timeout: _response(request.full_url, body))
    catalog = pack._mirror_catalog(None)
    assert catalog == (country, mirrors)
    assert list(pack._download_sources([artifact], catalog, True)) == [(artifact, (_OFFICIAL,))]


@pytest.mark.parametrize('body', [
    b'<html>', b'{}', b'{"country":"CHN","mirrors":{}}', b'{"country":"cn","mirrors":{}}',
    b'{"country":10,"mirrors":{}}', b'{"country":"CN","mirrors":[]}',
    json.dumps({'country': 'CN', 'mirrors': {'x' * 64: []}}).encode(),
    json.dumps({'country': 'CN', 'mirrors': {'a' * 64: _MIRROR}}).encode(),
    json.dumps({'country': 'CN', 'mirrors': {'a' * 64: ['http://mirror.example/torch.whl']}}).encode(),
    json.dumps({'country': 'CN', 'mirrors': {'a' * 64: ['https://user@mirror.example/torch.whl']}}).encode(),
    b' ' * pack._CATALOG_MAX_BYTES + b'{}',
])
def test_invalid_catalog_uses_bundled_sources(artifact, transport, monkeypatch, body):
    transport(lambda request, timeout: _response(request.full_url, body))
    monkeypatch.setattr(pack, '_windows_region', lambda: 'US')
    catalog = pack._mirror_catalog(None)
    assert catalog is None
    assert list(pack._download_sources([artifact], catalog, None)) == [(artifact, (_OFFICIAL, _BUNDLED))]


def test_catalog_requires_full_success_response(transport):
    body = json.dumps({'country': 'CN', 'mirrors': {}}).encode()
    transport(lambda request, timeout: _response(request.full_url, body, status=206))
    assert pack._mirror_catalog(None) is None


@pytest.mark.parametrize('first', [_OFFICIAL, _MIRROR])
def test_interrupted_transfer_resumes_at_first_missing_byte(tmp_path, artifact, transport, first):
    requests = []
    urls = (first, _OFFICIAL if first == _MIRROR else _MIRROR)

    def open_transfer(request, timeout):
        requested = request.get_header('Range')
        requests.append((request.full_url, requested))
        if requested is None:
            return _response(request.full_url, _PAYLOAD[:3])
        return _response(request.full_url, _PAYLOAD[3:], 206, **{'Content-Range': 'bytes 3-9/10'})

    transport(open_transfer)
    archive = tmp_path / 'example.whl'
    progress = []
    pack._fetch(artifact, archive, None, None, progress.append, urls=urls)
    assert requests == [(urls[0], None), (urls[1], 'bytes=3-')]
    assert progress[0] == 3 and progress[-1] == 10
    assert archive.read_bytes() == _PAYLOAD


def test_official_retry_continues_after_short_reads(tmp_path, artifact, transport):
    requests = []

    def open_transfer(request, timeout):
        requested = request.get_header('Range')
        requests.append(requested)
        if requested is None:
            return _response(request.full_url, _PAYLOAD[:3])
        if requested == 'bytes=3-':
            return _response(request.full_url, _PAYLOAD[3:5], 206, **{'Content-Range': 'bytes 3-9/10'})
        return _response(request.full_url, _PAYLOAD[5:], 206, **{'Content-Range': 'bytes 5-9/10'})

    transport(open_transfer)
    archive = tmp_path / 'example.whl'
    pack._fetch(artifact, archive, None, None, lambda _: None, urls=(_OFFICIAL,))
    assert requests == [None, 'bytes=3-', 'bytes=5-']
    assert archive.read_bytes() == _PAYLOAD


@pytest.mark.parametrize(('status', 'headers', 'body'), [
    (200, {}, _PAYLOAD),
    (206, {'Content-Range': 'bytes 2-9/10'}, _PAYLOAD[2:]),
    (206, {'Content-Range': 'bytes 3-8/10'}, _PAYLOAD[3:9]),
    (206, {'Content-Range': 'bytes 3-9/11'}, _PAYLOAD[3:]),
    (206, {'Content-Range': 'bytes 3-9/10', 'Content-Length': '8'}, _PAYLOAD[3:]),
    (206, {'Content-Range': 'bytes 3-9/10'}, _PAYLOAD[3:] + b'extra'),
])
def test_mirror_failover_keeps_prefix_across_bad_ranges(tmp_path, artifact, transport,
                                                      status, headers, body):
    requests = []

    def open_transfer(request, timeout):
        url = request.full_url
        requests.append((url, request.get_header('Range')))
        if url == _MIRROR:
            return _response(url, _PAYLOAD[:3])
        if url == _SECOND:
            return _response(url, body, status, **headers)
        return _response(url, _PAYLOAD[3:], 206, **{'Content-Range': 'bytes 3-9/10'})

    transport(open_transfer)
    archive = tmp_path / 'example.whl'
    pack._fetch(artifact, archive, None, None, lambda _: None, urls=(_MIRROR, _SECOND, _OFFICIAL))
    assert requests == [(_MIRROR, None), (_SECOND, 'bytes=3-'), (_OFFICIAL, 'bytes=3-')]
    assert archive.read_bytes() == _PAYLOAD


def test_full_download_only_restarts_when_remaining_source_ignores_range(tmp_path, artifact, transport):
    requests = []

    def open_transfer(request, timeout):
        url = request.full_url
        requests.append((url, request.get_header('Range')))
        return _response(url, _PAYLOAD[:3] if url == _MIRROR else _PAYLOAD)

    transport(open_transfer)
    archive = tmp_path / 'example.whl'
    pack._fetch(artifact, archive, None, None, lambda _: None, urls=(_MIRROR, _OFFICIAL))
    assert requests == [(_MIRROR, None), (_OFFICIAL, 'bytes=3-'), (_OFFICIAL, None)]
    assert archive.read_bytes() == _PAYLOAD


def test_resumed_sha_mismatch_retries_without_corrupt_prefix(tmp_path, artifact, transport):
    requests = []

    def open_transfer(request, timeout):
        url = request.full_url
        requested = request.get_header('Range')
        requests.append((url, requested))
        if url == _MIRROR:
            return _response(url, b'BAD')
        if requested:
            return _response(url, _PAYLOAD[3:], 206, **{'Content-Range': 'bytes 3-9/10'})
        return _response(url, _PAYLOAD)

    transport(open_transfer)
    archive = tmp_path / 'example.whl'
    pack._fetch(artifact, archive, None, None, lambda _: None, urls=(_MIRROR, _OFFICIAL))
    assert requests == [(_MIRROR, None), (_OFFICIAL, 'bytes=3-'), (_OFFICIAL, None)]
    assert archive.read_bytes() == _PAYLOAD


def test_download_exhausts_sources_with_pinned_sha_and_safe_diagnostics(tmp_path, artifact, transport, network_log):
    def open_transfer(request, timeout):
        if request.full_url == _MIRROR:
            raise HTTPError('https://storage.example/asset?token=private', 403, 'private body', {}, None)
        return _response(request.full_url, b'0123456789')

    transport(open_transfer)
    with pytest.raises(ValueError, match='download.pytorch.org: SHA-256 mismatch'):
        pack._fetch(artifact, tmp_path / 'example.whl', None, None, lambda _: None,
                    urls=(_MIRROR, _OFFICIAL))
    assert 'mirror.example via storage.example failed (HTTP 403); trying download.pytorch.org' in network_log.text
    assert 'private' not in network_log.text


def test_cancellation_stops_download_before_failover(tmp_path, artifact, transport):
    cancelled = False

    def open_transfer(request, timeout):
        assert request.full_url == _MIRROR
        return _response(request.full_url, _PAYLOAD[:3])

    def notify(count):
        nonlocal cancelled
        cancelled = True

    transport(open_transfer)
    with pytest.raises(pack.PackCancelled):
        pack._fetch(artifact, tmp_path / 'example.whl', None, lambda: cancelled, notify,
                    urls=(_MIRROR, _OFFICIAL))
