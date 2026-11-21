"""Numerical and workspace coverage for native StackMFF attention."""
import hashlib

import pytest

from core.inference_worker import ROOT, _load_torch_model
from core.models import attention_workspace as workspace


@pytest.fixture(scope='module')
def torch_cpu():
    torch = pytest.importorskip('torch')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('graph', ['stackmff_final', 'stackmff_v2_final', 'stackmff_v3_final'])
def test_released_logits_agree_with_dense_attention(torch_cpu, monkeypatch, graph):
    torch = torch_cpu
    from core.models.stackmff_v3 import network

    digest = hashlib.sha256((ROOT / 'weights' / f'{graph}.onnx').read_bytes()).hexdigest()
    dense = _load_torch_model(digest, 'cpu')
    bounded = _load_torch_model(digest, 'cpu', bounded_workspace=True)
    # The rectangular tile exercises shifted masks and multiple window blocks.
    calls = set()
    original = network._score_attention

    def record(q, k, v, mask=None):
        assert q.numel() // q.shape[-1] * k.shape[-2] <= workspace.SCORE_ELEMENTS
        assert q.dtype == k.dtype == v.dtype == torch.float32
        calls.add(q.ndim)
        return original(q, k, v, mask)

    x = torch.rand((1, 9, 64, 128), generator=torch.Generator().manual_seed(217))
    with torch.inference_mode():
        reference = dense(x)
        torch.testing.assert_close(bounded(x), reference, rtol=0, atol=0)
        monkeypatch.setattr(workspace, 'SCORE_ELEMENTS', 2 * 3 * 64**2)
        monkeypatch.setattr(workspace, 'FOCUS_QUERY_LIMIT', 4)
        monkeypatch.setattr(network, '_score_attention', record)
        torch.testing.assert_close(bounded(x), reference, rtol=0, atol=1e-3)
    assert calls == {3, 5}  # Cross-layer sites and intra-layer windows.


@pytest.mark.parametrize('layers', [100, 500])
def test_cross_layer_blocks_keep_every_key_and_query(torch_cpu, monkeypatch, layers):
    torch = torch_cpu
    from core.models.stackmff_v3 import network

    block = network.InterLayerAttention(48, 3, pool=2).eval()
    x = torch.rand((1, layers, 4, 6, 48), generator=torch.Generator().manual_seed(64))
    calls = []
    original = network._score_attention

    def record(q, k, v, mask=None):
        groups, queries, _ = q.shape
        assert k.shape[-2] == v.shape[-2] == layers
        assert groups * queries * layers <= workspace.SCORE_ELEMENTS
        assert queries < layers
        calls.append((groups, queries))
        return original(q, k, v, mask)

    with torch.inference_mode():
        reference = block(x)
        block.bounded_workspace = True
        monkeypatch.setattr(workspace, 'SCORE_ELEMENTS', 3 * layers * 64 * 2)
        monkeypatch.setattr(workspace, 'FOCUS_QUERY_LIMIT', 64)
        monkeypatch.setattr(network, '_score_attention', record)
        torch.testing.assert_close(block(x), reference, rtol=0, atol=1e-5)
    assert len(calls) > 1
    assert sum(groups * queries for groups, queries in calls) == 6 * 3 * layers
