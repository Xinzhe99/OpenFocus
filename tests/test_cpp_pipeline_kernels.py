"""Parity and zero-Numba-JIT tests for PR 6 pipeline C++17 kernels."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
@pytest.mark.parametrize('shape,bgr_rev', [
    ((37, 43), False),
    ((41, 53, 3), False),
    ((41, 53, 3), True),
    ((29, 31, 4), False),
])
def test_cpp_table_rows_matches_numba(monkeypatch, dtype, shape, bgr_rev):
    from core import _native
    from fusion_methods import _openfocus_kernels as kernels

    if _native.load_native() is None:
        pytest.skip('Native module not built')

    rng = np.random.default_rng(11)
    hi = 256 if dtype == np.uint8 else 65536
    image = rng.integers(0, hi, size=shape, dtype=dtype)
    if bgr_rev and image.ndim == 3:
        image = image[:, :, ::-1].copy()[:, :, ::-1]
    tcols = 1 if len(shape) == 2 else shape[2]
    table = rng.integers(0, hi, size=(hi, tcols), dtype=dtype)

    out_cpp = np.empty(shape, dtype=dtype)
    kernels.table_rows(image, table, out_cpp, 0, shape[0])

    monkeypatch.setattr(_native, 'load_native', lambda: None)
    out_nb = np.empty(shape, dtype=dtype)
    kernels.table_rows(image, table, out_nb, 0, shape[0])

    np.testing.assert_array_equal(out_cpp, out_nb)


@pytest.mark.parametrize('src_len,dst_len,linear', [
    (64, 32, False),
    (97, 48, False),
    (129, 64, False),
    (32, 64, True),
    (48, 97, True),
    (50, 101, True),
])
def test_cpp_axis_plan_fill_matches_numba_exact(monkeypatch, src_len, dst_len, linear):
    from core import _native
    from fusion_methods import _openfocus_resample as resample

    if _native.load_native() is None:
        pytest.skip('Native module not built')

    plan_cpp = resample._axis_plan(src_len, dst_len, linear=linear)
    monkeypatch.setattr(_native, 'load_native', lambda: None)
    plan_nb = resample._axis_plan(src_len, dst_len, linear=linear)

    np.testing.assert_array_equal(plan_cpp.indices, plan_nb.indices)
    np.testing.assert_array_equal(plan_cpp.weights, plan_nb.weights)
    np.testing.assert_array_equal(plan_cpp.counts, plan_nb.counts)


@pytest.mark.parametrize('dtype,norm', [
    (np.uint8, False),
    (np.uint16, False),
    (np.uint16, True),
])
@pytest.mark.parametrize('shape,bgr_rev', [
    ((257, 400, 3), False),
    ((256, 300, 3), True),
    ((200, 220, 4), False),
])
def test_cpp_exposure_mean_matches_fallback(monkeypatch, dtype, norm, shape, bgr_rev):
    from core import _native
    from utils import image_utils

    if _native.load_native() is None:
        pytest.skip('Native module not built')

    rng = np.random.default_rng(22)
    hi = 256 if dtype == np.uint8 else 65536
    img = rng.integers(0, hi, size=shape, dtype=dtype)
    if bgr_rev:
        img = img[:, :, ::-1].copy()[:, :, ::-1]

    mean_cpp = image_utils._exposure_mean(img, normalize_uint16=norm)
    mean_cancellable = image_utils._exposure_mean(img, lambda: False, normalize_uint16=norm)
    monkeypatch.setattr(_native, 'load_native', lambda: None)
    monkeypatch.setattr(image_utils, '_load_image_kernels', lambda: None)
    mean_fb = image_utils._exposure_mean(img, normalize_uint16=norm)

    assert mean_cpp == pytest.approx(mean_fb, rel=1e-13)
    assert mean_cancellable == pytest.approx(mean_fb, rel=1e-13)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_cpp_exposure_mean_cancels_between_strips(monkeypatch, dtype):
    from core import _native
    from core.multi_focus_fusion import MultiFocusFusion
    from core.registration import RegistrationCancelled

    native = _native.load_native()
    if native is None:
        pytest.skip('Native module not built')

    frame = np.full((1025, 31, 3), 120, dtype)
    original = native.of_exposure_mean
    reduced = []

    def exposure_mean(strip, normalize_uint16, num_threads):
        mean = original(strip, normalize_uint16, num_threads)
        reduced.append(strip.shape[0])
        return mean

    monkeypatch.setattr(native, 'of_exposure_mean', exposure_mean)
    with pytest.raises(RegistrationCancelled):
        MultiFocusFusion('dct')._exposure_gains_for_source(
            [frame], lambda: bool(reduced), normalize_uint16=True)
    assert reduced == [128]
    assert np.all(frame == 120)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize('shape', [(64, 80), (64, 80, 3), (64, 80, 4)])
def test_cpp_apply_exposure_gain_matches_fallback(monkeypatch, dtype, shape):
    from core import _native
    from utils import image_utils

    if _native.load_native() is None:
        pytest.skip('Native module not built')

    rng = np.random.default_rng(33)
    if dtype == np.float32:
        img = rng.uniform(0.0, 255.0, size=shape).astype(np.float32)
    else:
        hi = 256 if dtype == np.uint8 else 65536
        img = rng.integers(0, hi, size=shape, dtype=dtype)

    out_cpp = image_utils.apply_exposure_gain(img, 1.35)
    monkeypatch.setattr(_native, 'load_native', lambda: None)
    out_fb = image_utils.apply_exposure_gain(img, 1.35)

    if dtype == np.float32:
        np.testing.assert_allclose(out_cpp, out_fb, rtol=1e-6, atol=1e-4)
    else:
        np.testing.assert_array_equal(out_cpp, out_fb)


@pytest.mark.parametrize('src_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('is_16', [False, True])
def test_cpp_quantize_output_matches_fallback(monkeypatch, src_dtype, is_16):
    from core import _native
    from utils import image_utils

    if _native.load_native() is None:
        pytest.skip('Native module not built')

    rng = np.random.default_rng(44)
    arr = rng.uniform(-5.0, 260.0, size=(300, 300, 3)).astype(src_dtype)

    out_cpp = image_utils.quantize_fuse_output(arr, is_16)
    monkeypatch.setattr(_native, 'load_native', lambda: None)
    out_fb = image_utils.quantize_fuse_output(arr, is_16)

    np.testing.assert_array_equal(out_cpp, out_fb)


@pytest.mark.parametrize('tile_dtype', [np.uint8, np.float32])
def test_cpp_tile_reduce_and_finalize_matches_fallback(monkeypatch, tile_dtype):
    from core import _native
    from core import resource_policy as policy
    from core.multi_focus_fusion import MultiFocusFusion

    if _native.load_native() is None:
        pytest.skip('Native module not built')

    snapshot = policy.ResourceSnapshot(
        available_ram=4 * 1024 ** 3, total_ram=8 * 1024 ** 3,
        physical_cores=4, logical_cpus=4, cpu_limit=4, physical_cores_known=True,
    )
    monkeypatch.setattr(policy, 'probe_resources', lambda *a, **k: snapshot)

    rng = np.random.default_rng(55)
    h, w = 96, 112
    if tile_dtype == np.uint8:
        frames = [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for _ in range(3)]
    else:
        frames = [rng.integers(0, 65536, size=(h, w, 3), dtype=np.uint16) for _ in range(3)]

    fusion = MultiFocusFusion(
        algorithm='openfocus_b', use_gpu=False,
        tile_enabled=True, tile_threshold=32, tile_block_size=48, tile_overlap=16,
    )
    out_cpp = fusion.fuse(frames, thread_count=2)

    monkeypatch.setattr(_native, 'load_native', lambda: None)
    out_fb = fusion.fuse(frames, thread_count=2)

    if tile_dtype == np.uint8:
        diff = np.abs(out_cpp.astype(np.int64) - out_fb.astype(np.int64))
        assert int(diff.max()) <= 1
    else:
        np.testing.assert_allclose(out_cpp, out_fb, rtol=1e-5, atol=1e-3)


def test_default_fusion_triggers_zero_numba_jit_when_native_loaded():
    code = textwrap.dedent('''
        import json
        import sys
        import numpy as np
        from core._native import load_native
        assert load_native() is not None

        from core import resource_policy as policy
        snapshot = policy.ResourceSnapshot(
            available_ram=4 * 1024 ** 3, total_ram=8 * 1024 ** 3,
            physical_cores=4, logical_cpus=4, cpu_limit=4, physical_cores_known=True,
        )
        policy.probe_resources = lambda *a, **k: snapshot

        from core.multi_focus_fusion import MultiFocusFusion
        from fusion_methods import _depth_kernel, _openfocus_kernels, _openfocus_resample

        rng = np.random.default_rng(99)
        frames = [rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for _ in range(3)]
        for algo in ('openfocus_a', 'openfocus_b', 'openfocus_c'):
            out = MultiFocusFusion(algorithm=algo, use_gpu=False, exposure_match=True).fuse(frames, thread_count=2)
            assert out.shape == (64, 64, 3)

        compiled = {
            'axis_block': len(_depth_kernel.axis_block.signatures),
            'resample_rows': len(_openfocus_resample._resample_rows.signatures),
            'fill_half': len(_openfocus_resample._fill_half_axis.signatures),
            'fill_linear': len(_openfocus_resample._fill_linear_axis.signatures),
            'table_rows': len(_openfocus_kernels._nb_table_rows.signatures),
            'image_kernels_loaded': 'utils._image_kernels' in sys.modules,
        }
        print(json.dumps(compiled))
    ''')
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding='utf-8', timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    info = json.loads(result.stdout.splitlines()[-1])
    assert info == {
        'axis_block': 0,
        'resample_rows': 0,
        'fill_half': 0,
        'fill_linear': 0,
        'table_rows': 0,
        'image_kernels_loaded': False,
    }
