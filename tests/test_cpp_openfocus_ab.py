"""Parity tests between C++ native OpenFocus A/B kernels and Numba fallbacks."""
import numpy as np
import pytest


def _synthetic_stack(shape, dtype, *, count=4, bgr_rev=False, seed=42):
    rng = np.random.default_rng(seed)
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frames = []
    for idx in range(count):
        base = 0.35 + 0.25 * np.sin(yy * 0.05 + idx * 0.4) * np.cos(xx * 0.04)
        # Sharp high-frequency band moving across depth index.
        band_center = (idx + 0.5) * (w / count)
        mask = np.exp(-((xx - band_center) ** 2) / (2.0 * (w / (2.5 * count)) ** 2))
        texture = 0.35 * np.sin(xx * (1.2 + 0.1 * idx)) * np.cos(yy * 1.1) * mask
        noise = rng.normal(0.0, 0.015, size=(h, w)).astype(np.float32)
        plane = np.clip(base + texture + noise, 0.0, 1.0)
        if len(shape) == 3:
            c = shape[2]
            arr = np.empty((h, w, c), dtype=np.float32)
            arr[:, :, 0] = plane
            arr[:, :, 1] = np.clip(plane * 0.85 + 0.08 * np.sin(yy * 0.2), 0.0, 1.0)
            arr[:, :, 2] = np.clip(plane * 0.7 + 0.12 * np.cos(xx * 0.15), 0.0, 1.0)
            if c == 4:
                arr[:, :, 3] = 1.0
        else:
            arr = plane
        if dtype == np.uint8:
            q = np.rint(arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            q = np.rint(arr * 65535.0).clip(0, 65535).astype(np.uint16)
        if bgr_rev and q.ndim == 3 and q.shape[2] == 3:
            q = q[:, :, ::-1].copy()[:, :, ::-1]
        frames.append(q)
    return frames


@pytest.mark.parametrize('shape,bgr_rev', [
    ((64, 64, 3), False),
    ((97, 83, 3), True),
    ((73, 101), False),
])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('disable_bg', [False, True])
def test_cpp_openfocus_a_matches_numba(monkeypatch, shape, bgr_rev, dtype, disable_bg):
    from core import _native
    from fusion_methods import openfocus_a

    if _native.load_native() is None:
        pytest.skip('Native module not built')

    frames = _synthetic_stack(shape, dtype, count=4, bgr_rev=bgr_rev, seed=101)
    if disable_bg:
        monkeypatch.setattr(openfocus_a, '_BACKGROUND_RAM_LIMIT_BYTES', 0)

    out_cpp = openfocus_a.openfocus_a_impl(frames, thread_count=2)

    monkeypatch.setattr(_native, 'load_native', lambda: None)
    out_nb = openfocus_a.openfocus_a_impl(frames, thread_count=2)

    assert out_cpp.dtype == out_nb.dtype
    assert out_cpp.shape == out_nb.shape
    diff = np.abs(out_cpp.astype(np.int64) - out_nb.astype(np.int64))
    assert int(diff.max()) <= 1


@pytest.mark.parametrize('shape,bgr_rev', [
    ((64, 64, 3), False),
    ((97, 83, 3), True),
    ((73, 101), False),
])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('smoothing,feather', [
    (0.0, 0),
    (4.0 * 0.24, 3),
])
def test_cpp_openfocus_b_matches_numba(monkeypatch, shape, bgr_rev, dtype, smoothing, feather):
    from core import _native
    from fusion_methods import openfocus_bc

    if _native.load_native() is None:
        pytest.skip('Native module not built')

    frames = _synthetic_stack(shape, dtype, count=5, bgr_rev=bgr_rev, seed=202)
    out_cpp = openfocus_bc.openfocus_b_impl(
        frames, thread_count=2, kernel_size=7,
        depth_smoothing=smoothing, feather=feather,
    )

    monkeypatch.setattr(_native, 'load_native', lambda: None)
    out_nb = openfocus_bc.openfocus_b_impl(
        frames, thread_count=2, kernel_size=7,
        depth_smoothing=smoothing, feather=feather,
    )

    assert out_cpp.dtype == out_nb.dtype
    assert out_cpp.shape == out_nb.shape
    diff = np.abs(out_cpp.astype(np.int64) - out_nb.astype(np.int64))
    assert int(diff.max()) == 0


@pytest.mark.parametrize('src_shape,target_half', [
    ((64, 64), (32, 32)),
    ((97, 83), (48, 41)),
])
def test_cpp_resample_matches_numba_exact(monkeypatch, src_shape, target_half):
    from core import _native
    from fusion_methods import _openfocus_resample as resample

    if _native.load_native() is None:
        pytest.skip('Native module not built')

    rng = np.random.default_rng(303)
    gray = rng.integers(0, 65536, size=src_shape, dtype=np.uint16)

    half_cpp = resample.half_gray(gray)
    assert half_cpp.shape == target_half
    full_cpp = resample.expand_depth(half_cpp, src_shape)
    assert full_cpp.shape == src_shape

    monkeypatch.setattr(_native, 'load_native', lambda: None)
    half_nb = resample.half_gray(gray)
    full_nb = resample.expand_depth(half_cpp, src_shape)

    np.testing.assert_array_equal(half_cpp, half_nb)
    np.testing.assert_array_equal(full_cpp, full_nb)


@pytest.mark.parametrize('shape', [(65, 79), (128, 128)])
@pytest.mark.parametrize('guided', [False, True])
def test_cpp_depth_iir_matches_numba_exact(monkeypatch, shape, guided):
    from core import _native
    from fusion_methods import openfocus_depth

    if _native.load_native() is None:
        pytest.skip('Native module not built')

    rng = np.random.default_rng(404)
    src = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    guide = rng.integers(0, 65536, size=shape, dtype=np.uint16) if guided else None

    out_cpp = np.empty(shape, dtype=np.uint16)
    openfocus_depth._axis(src, guide, 2.5, 2.5, 2.5, 1, lambda: None, np.empty, output=out_cpp)
    out0_cpp = np.empty(shape, dtype=np.uint16)
    openfocus_depth._axis(out_cpp, guide, 2.5, 2.5, 2.5, 0, lambda: None, np.empty, output=out0_cpp)

    monkeypatch.setattr(_native, 'load_native', lambda: None)
    out_nb = np.empty(shape, dtype=np.uint16)
    openfocus_depth._axis(src, guide, 2.5, 2.5, 2.5, 1, lambda: None, np.empty, output=out_nb)
    out0_nb = np.empty(shape, dtype=np.uint16)
    openfocus_depth._axis(out_nb, guide, 2.5, 2.5, 2.5, 0, lambda: None, np.empty, output=out0_nb)

    np.testing.assert_array_equal(out_cpp, out_nb)
    np.testing.assert_array_equal(out0_cpp, out0_nb)
