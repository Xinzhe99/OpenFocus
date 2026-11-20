"""Managed pack transactions with tiny local archives and no GPU or network."""
from contextlib import nullcontext
import hashlib
import io
import json
import logging
from pathlib import Path
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
    manifest = pack._manifest()
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


@pytest.fixture
def transaction(tmp_path, monkeypatch):
    sources = tmp_path / 'archives'
    sources.mkdir()
    archive = sources / 'embed.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name in ('python.exe', 'python310.dll', 'python310._pth'):
            z.writestr(name, 'input')
    data = archive.read_bytes()
    manifest = {'version': 'test', 'backends': ['TORCH_CUDA'], 'required_free_bytes': 1,
                'artifacts': [{'url': 'https://www.python.org/embed.zip', 'size': len(data),
                               'sha256': hashlib.sha256(data).hexdigest()}]}
    root = tmp_path / 'packs'
    monkeypatch.setattr(pack, '_manifest', lambda: manifest)
    monkeypatch.setattr(pack, '_hardware', lambda: ('test device', ''))
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
    if pack.sys.platform == 'win32':
        assert pack.worker_command() == [str(root / new / 'python.exe'), '-I', '-u', str(pack._WORKER)]


def test_unsupported_hardware_does_not_start_install(transaction, monkeypatch):
    root, sources = transaction
    monkeypatch.setattr(pack, '_hardware', lambda: ('unsupported device', 'architecture'))
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
    artifact, = pack._manifest()['artifacts']
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
        assert dict(request.header_items()) == {'Accept': 'application/json'}
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
