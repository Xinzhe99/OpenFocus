import numpy as np
import pytest

from core import _native
from core.resource_policy import ResourceSnapshot
from fusion_methods import openfocus_a, openfocus_bc


def _mock_resources(monkeypatch):
    from core import resource_policy as rp

    monkeypatch.setattr(
        rp,
        "probe_resources",
        lambda *a, **kw: ResourceSnapshot(
            available_ram=16 * 1024**3,
            total_ram=32 * 1024**3,
            physical_cores=4,
            logical_cpus=8,
            physical_cores_known=True,
            commit_headroom=16 * 1024**3,
        ),
    )


def _require_avx2():
    mod = _native.load_native()
    assert mod is not None
    mod.set_force_scalar(False)
    if mod.simd_info() != "AVX2":
        pytest.skip(f"Host CPU/OS reports {mod.simd_info()}, not AVX2")
    return mod


def test_simd_info_and_force_scalar_toggle():
    mod = _require_avx2()
    try:
        assert mod.simd_info() == "AVX2"
        mod.set_force_scalar(True)
        assert mod.simd_info() == "Scalar"
        mod.set_force_scalar(False)
        assert mod.simd_info() == "AVX2"
    finally:
        mod.set_force_scalar(False)


@pytest.mark.parametrize("shape", [(17, 1, 3), (17, 7, 3), (17, 8, 3), (17, 15, 3), (25, 16, 3), (33, 129, 3)])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_openfocus_abc_avx2_vs_scalar_bit_exact(monkeypatch, shape, dtype):
    _mock_resources(monkeypatch)
    mod = _require_avx2()
    rng = np.random.default_rng(100 + shape[1] + (0 if dtype == np.uint8 else 1000))
    hi = 256 if dtype == np.uint8 else 65536
    frames = [rng.integers(0, hi, shape, dtype=dtype) for _ in range(4)]

    try:
        mod.set_force_scalar(False)
        out_a_avx = openfocus_a.openfocus_a_impl(frames, thread_count=2)
        out_b_avx = openfocus_bc.openfocus_b_impl(frames, thread_count=2, depth_smoothing=2.4)
        out_c_avx = openfocus_bc.openfocus_c_impl(frames, thread_count=2, blend_smoothing=2.4)

        mod.set_force_scalar(True)
        out_a_sca = openfocus_a.openfocus_a_impl(frames, thread_count=2)
        out_b_sca = openfocus_bc.openfocus_b_impl(frames, thread_count=2, depth_smoothing=2.4)
        out_c_sca = openfocus_bc.openfocus_c_impl(frames, thread_count=2, blend_smoothing=2.4)
    finally:
        mod.set_force_scalar(False)

    assert np.array_equal(out_a_avx, out_a_sca)
    assert np.array_equal(out_b_avx, out_b_sca)
    assert np.array_equal(out_c_avx.view(np.uint8), out_c_sca.view(np.uint8))


@pytest.mark.parametrize("guided", [False, True])
@pytest.mark.parametrize("w", [1, 7, 8, 15, 16, 33, 129])
def test_depth_axis_iir_and_quantize_avx2_vs_scalar_bit_exact(guided, w):
    mod = _require_avx2()
    rng = np.random.default_rng(42 + w + (10 if guided else 0))
    h = 25
    src_iir = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    guide = rng.integers(0, 65536, (h, w), dtype=np.uint16) if guided else None

    fsrc = rng.uniform(-5.0, 260.0, (h, w, 3)).astype(np.float32)

    try:
        mod.set_force_scalar(False)
        out_iir_avx = np.empty_like(src_iir)
        mod.of_depth_axis_iir(src_iir, guide, out_iir_avx, 3072, 1024, 3840, 256, 3, 0, 1)
        q16_avx = np.empty((h, w, 3), dtype=np.uint16)
        q8_avx = np.empty((h, w, 3), dtype=np.uint8)
        mod.of_quantize_output(fsrc, q16_avx, True, 1)
        mod.of_quantize_output(fsrc, q8_avx, False, 1)

        mod.set_force_scalar(True)
        out_iir_sca = np.empty_like(src_iir)
        mod.of_depth_axis_iir(src_iir, guide, out_iir_sca, 3072, 1024, 3840, 256, 3, 0, 1)
        q16_sca = np.empty((h, w, 3), dtype=np.uint16)
        q8_sca = np.empty((h, w, 3), dtype=np.uint8)
        mod.of_quantize_output(fsrc, q16_sca, True, 1)
        mod.of_quantize_output(fsrc, q8_sca, False, 1)
    finally:
        mod.set_force_scalar(False)

    assert np.array_equal(out_iir_avx, out_iir_sca)
    assert np.array_equal(q16_avx, q16_sca)
    assert np.array_equal(q8_avx, q8_sca)


@pytest.mark.parametrize("w", [9, 17, 33, 129])
def test_pyramid_kernels_avx2_vs_scalar_zero_ulp(w):
    mod = _require_avx2()
    rng = np.random.default_rng(77 + w)
    h = 33
    raw = rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)
    target = rng.random((h, w, 3), dtype=np.float32)
    ch, cw = (h + 1) // 2, (w + 1) // 2

    def _run_pyr():
        coarse = np.empty((ch, cw, 3), dtype=np.float32)
        scratch_down = np.empty((h, cw, 3), dtype=np.float32)
        mod.of_pyr_down_from_raw(raw, coarse, scratch_down, 1.0 / 65535.0, 0, ch)

        w0 = np.empty((h, w), dtype=np.uint16)
        scratch_up = np.empty((ch, w, 3), dtype=np.float32)
        mod.of_pyr_level0_weights_from_raw(coarse, raw, w0, scratch_up, 1.0 / 65535.0, 0, h)

        w1 = np.empty((h, w), dtype=np.uint16)
        target_det = target.copy()
        mod.of_pyr_detail_weight_f32(coarse, target_det, w1, scratch_up, 0, h)

        recon = target.copy()
        mod.of_pyr_reconstruct_f32(coarse, recon, scratch_up, 0, h)
        return coarse, w0, target_det, w1, recon

    try:
        mod.set_force_scalar(False)
        res_avx = _run_pyr()
        mod.set_force_scalar(True)
        res_sca = _run_pyr()
    finally:
        mod.set_force_scalar(False)

    for arr_a, arr_s in zip(res_avx, res_sca):
        assert np.array_equal(arr_a.view(np.uint8), arr_s.view(np.uint8))
