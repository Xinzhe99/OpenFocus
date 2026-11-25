"""Parity tests between C++ native OpenFocus C pyramid kernels and Numba fallbacks."""
import math
import numpy as np
import pytest


def _synthetic_stack(shape, dtype, *, count=4, bgr_rev=False, seed=77):
    rng = np.random.default_rng(seed)
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frames = []
    for idx in range(count):
        base = 0.30 + 0.25 * np.sin(yy * 0.06 + idx * 0.3) * np.cos(xx * 0.05)
        cx = (idx + 0.5) * (w / count)
        env = np.exp(-((xx - cx) ** 2) / (2.0 * (max(w, 8) / (2.5 * count)) ** 2))
        detail = 0.35 * np.sin(xx * (1.1 + 0.12 * idx)) * np.cos(yy * 0.9) * env
        noise = rng.normal(0.0, 0.015, size=(h, w)).astype(np.float32)
        plane = np.clip(base + detail + noise, 0.0, 1.0)
        if len(shape) == 3:
            c = shape[2]
            arr = np.empty((h, w, c), dtype=np.float32)
            arr[:, :, 0] = plane
            if c > 1:
                arr[:, :, 1] = np.clip(plane * 0.85 + 0.08 * np.sin(yy * 0.2), 0.0, 1.0)
            if c > 2:
                arr[:, :, 2] = np.clip(plane * 0.70 + 0.12 * np.cos(xx * 0.15), 0.0, 1.0)
            if c > 3:
                arr[:, :, 3] = 0.5 + 0.5 * np.sin(xx * 0.3 + yy * 0.1 + idx)
        else:
            arr = plane
        if dtype == np.uint8:
            q = np.rint(arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            q = np.rint(arr * 65535.0).clip(0, 65535).astype(np.uint16)
        if bgr_rev and q.ndim == 3 and q.shape[2] >= 3:
            q = q[:, :, ::-1].copy()[:, :, ::-1]
        frames.append(q)
    return frames


@pytest.mark.parametrize('shape,bgr_rev', [
    ((64, 64, 3), False),
    ((97, 83, 3), True),
    ((73, 101), False),
    ((131, 67, 4), False),
    ((20, 22, 3), False),
    ((12, 14, 3), False),
])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('smoothing', [0.0, 2.4, 4.0])
def test_cpp_openfocus_c_matches_numba(monkeypatch, shape, bgr_rev, dtype, smoothing):
    from core import _native
    from fusion_methods import openfocus_bc

    if _native.load_native() is None:
        pytest.skip('Native module not built')

    frames = _synthetic_stack(shape, dtype, count=4, bgr_rev=bgr_rev, seed=77)
    winners_cpp = []
    winners_nb = []

    out_cpp = openfocus_bc.openfocus_c_impl(
        frames, thread_count=2, blend_smoothing=smoothing, use_gpu=False,
        focus_callback=lambda w, mx: winners_cpp.append(w.copy()),
    )

    monkeypatch.setattr(_native, 'load_native', lambda: None)
    out_nb = openfocus_bc.openfocus_c_impl(
        frames, thread_count=2, blend_smoothing=smoothing, use_gpu=False,
        focus_callback=lambda w, mx: winners_nb.append(w.copy()),
    )

    assert out_cpp.dtype == out_nb.dtype
    assert out_cpp.shape == out_nb.shape

    if dtype == np.uint8:
        diff = np.abs(out_cpp.astype(np.int64) - out_nb.astype(np.int64))
        assert int(diff.max()) <= 1
        mse = float(np.mean((out_cpp.astype(np.float64) - out_nb.astype(np.float64)) ** 2))
        if mse > 0:
            assert 10.0 * math.log10((255.0 * 255.0) / mse) > 55.0
    else:
        diff = np.abs(out_cpp.astype(np.float64) - out_nb.astype(np.float64))
        assert float(diff.max()) <= (1.0 / 257.0) + 1e-4
        mse = float(np.mean(diff * diff))
        if mse > 0:
            assert 10.0 * math.log10((255.0 * 255.0) / mse) > 70.0

    assert len(winners_cpp) == 1 and len(winners_nb) == 1
    np.testing.assert_array_equal(winners_cpp[0], winners_nb[0])
