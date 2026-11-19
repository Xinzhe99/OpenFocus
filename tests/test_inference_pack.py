"""Managed pack transactions with tiny local archives and no GPU or network."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
import zipfile

import pytest

from core import inference_pack as pack


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

    def progress(done, total, phase):
        nonlocal cancelling
        cancelling = failure == 'cancel' and phase.startswith('Unpacking ')

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


def test_interrupted_transfer_resumes_at_first_missing_byte(tmp_path, monkeypatch):
    import io

    payload = b'abcdefghij'
    ranges = []

    def open_transfer(request, timeout):
        requested = request.get_header('Range')
        ranges.append(requested)
        stream = io.BytesIO(payload[:3] if requested is None else payload[3:])
        stream.status = 200 if requested is None else 206
        stream.headers = {'Content-Range': 'bytes 3-9/10'}
        return stream

    monkeypatch.setattr(pack, 'urlopen', open_transfer)
    artifact = {'url': 'https://www.python.org/example.zip', 'size': len(payload),
                'sha256': hashlib.sha256(payload).hexdigest()}
    archive = tmp_path / 'example.zip'
    pack._fetch(artifact, archive, None, None, lambda count: None)
    assert ranges == [None, 'bytes=3-']
    assert archive.read_bytes() == payload
