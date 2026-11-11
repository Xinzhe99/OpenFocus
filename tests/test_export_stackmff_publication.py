import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import export_stackmff_onnx as exporter


def test_failed_publication_restores_models(tmp_path, monkeypatch):
    weights = tmp_path / 'weights'
    weights.mkdir()
    for name in ('a', 'b', 'c'):
        (weights / f'{name}.pt').write_bytes(b'checkpoint')
        (weights / f'{name}.onnx').write_bytes(('old-' + name).encode())
    monkeypatch.setattr(exporter, 'ROOT', tmp_path)
    monkeypatch.setattr(exporter, '_require', lambda *args: SimpleNamespace(__version__='test'))
    monkeypatch.setattr(exporter, '_variants', lambda: {name: (f'{name}.pt', lambda path: None) for name in ('a', 'b', 'c')})
    monkeypatch.setattr(exporter, '_export', lambda model, path, opset: path.write_bytes(b'new'))
    monkeypatch.setattr(exporter, '_annotate', lambda *args: None)
    original_replace = exporter.os.replace
    def replace(source, destination):
        if Path(destination) == weights / 'b.onnx':
            raise OSError('publication failed')
        return original_replace(source, destination)
    monkeypatch.setattr(exporter.os, 'replace', replace)
    with pytest.raises(OSError, match='publication failed'):
        exporter.main(['--out-dir', str(weights), '--no-verify'])
    for name in ('a', 'b', 'c'):
        assert (weights / f'{name}.onnx').read_bytes() == ('old-' + name).encode()
    assert not list(weights.glob('.stackmff-export-*'))
