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


@pytest.mark.parametrize("w", [1, 7, 8, 15, 16, 33, 129])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_exposure_gain_avx2_vs_scalar_bit_exact(w, dtype, channels):
    mod = _require_avx2()
    rng = np.random.default_rng(500 + w + channels)
    h = 25
    shape = (h, w) if channels == 1 else (h, w, channels)
    if dtype == np.float32:
        img = rng.uniform(0.0, 260.0, shape).astype(np.float32)
    else:
        hi = 256 if dtype == np.uint8 else 65536
        img = rng.integers(0, hi, shape, dtype=dtype)
    gain = 1.137

    try:
        mod.set_force_scalar(False)
        out_avx = np.empty_like(img)
        mod.of_apply_exposure_gain(img, out_avx, gain, 0, h, 2)

        mod.set_force_scalar(True)
        out_sca = np.empty_like(img)
        mod.of_apply_exposure_gain(img, out_sca, gain, 0, h, 2)
    finally:
        mod.set_force_scalar(False)

    assert np.array_equal(out_avx.view(np.uint8), out_sca.view(np.uint8))


@pytest.mark.parametrize("w", [7, 8, 17, 33, 129])
def test_ab_packed_gray_down2_merge_select_avx2_vs_scalar(w):
    mod = _require_avx2()
    rng = np.random.default_rng(600 + w)
    h = 33
    rgb8 = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    rgb16 = rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)
    g2d16 = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    g2d32 = rng.integers(0, 100000, (h, w), dtype=np.int32)
    oh, ow = (h + 1) // 2, (w + 1) // 2
    depth = rng.integers(0, 1000, (h, w), dtype=np.uint16)

    def _run_ab():
        g_u8 = np.empty((h, w), dtype=np.uint16)
        g_u16 = np.empty((h, w), dtype=np.int32)
        mod.of_gray_rows(rgb8, g_u8, 0, h, 256, False)
        mod.of_gray_rows(rgb16, g_u16, 0, h, 16, True)
        d_16 = np.empty((oh, ow), dtype=np.int32)
        d_32 = np.empty((oh, ow), dtype=np.int32)
        mod.of_down2_rows(g2d16, d_16, 0, oh, 8)
        mod.of_down2_rows(g2d32, d_32, 0, oh, 4)
        acc = rgb16.copy()
        mod.of_merge_rows(acc, rgb16, depth, 2, 100, 0, h, 1, False, False)
        return g_u8, g_u16, d_16, d_32, acc

    try:
        mod.set_force_scalar(False)
        res_avx = _run_ab()
        mod.set_force_scalar(True)
        res_sca = _run_ab()
    finally:
        mod.set_force_scalar(False)

    for arr_a, arr_s in zip(res_avx, res_sca):
        assert np.array_equal(arr_a, arr_s)


def test_ab_8k_width_and_500_layer_depth_parity():
    mod = _require_avx2()
    rng = np.random.default_rng(8500)
    h, w = 4, 7680  # 8K row width
    rgb8 = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    rgb16 = rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)
    g2d16 = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    # Sparse 500-layer depth map: layer 499 at step=131 (base=65369, mostly inactive)
    depth_500 = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    depth_500[1, 100:250] = 65369
    fsrc = rng.uniform(0.0, 260.0, (h, w, 3)).astype(np.float32)
    w2d = rng.uniform(0.1, 1.0, (h, w)).astype(np.float32)

    def _run_8k():
        g_u8 = np.empty((h, w), dtype=np.uint16)
        mod.of_gray_rows(rgb8, g_u8, 0, h, 256, False)
        exp16 = np.empty_like(rgb16)
        mod.of_apply_exposure_gain(rgb16, exp16, 1.07, 0, h, 2)
        d_16 = np.empty((h // 2, w // 2), dtype=np.int32)
        mod.of_down2_rows(g2d16, d_16, 0, h // 2, 8)
        acc_m = rgb16.copy()
        mod.of_merge_rows(acc_m, rgb16, depth_500, 499, 131, 0, h, 1, False, False)
        t_acc = np.zeros((h, w, 3), dtype=np.float32)
        t_wacc = np.zeros((h, w), dtype=np.float32)
        mod.of_tile_accumulate(t_acc, t_wacc, fsrc, w2d, 0, 0, 2)
        t_u8 = np.empty((h, w, 3), dtype=np.uint8)
        t_f32 = np.empty((h, w, 3), dtype=np.float32)
        mod.of_tile_finalize(t_acc, t_wacc, t_u8, 0, h)
        mod.of_tile_finalize(t_acc, t_wacc, t_f32, 0, h)
        return g_u8, exp16, d_16, acc_m, t_u8, t_f32

    try:
        mod.set_force_scalar(False)
        res_avx = _run_8k()
        mod.set_force_scalar(True)
        res_sca = _run_8k()
    finally:
        mod.set_force_scalar(False)

    for arr_a, arr_s in zip(res_avx, res_sca):
        assert np.array_equal(arr_a.view(np.uint8), arr_s.view(np.uint8))


def test_openfocus_c_full_8k_and_500_layer_simd_parity() -> None:
    mod = _require_avx2()
    rng = np.random.default_rng(202611)
    for h, w in [(33, 129), (8, 7680)]:
        cw = (w + 1) // 2
        ch = (h + 1) // 2
        fsrc3 = rng.uniform(-50.0, 65600.0, size=(h, w, 3)).astype(np.float32)
        fsrc1 = rng.uniform(-50.0, 65600.0, size=(h, w)).astype(np.float32)
        u16_3 = rng.integers(0, 65536, size=(h, w, 3), dtype=np.uint16)
        coarse3 = rng.uniform(0.0, 65535.0, size=(ch, cw, 3)).astype(np.float32)
        coarse1 = rng.uniform(0.0, 65535.0, size=(ch, cw)).astype(np.float32)
        weight_u16 = rng.integers(0, 65535, size=(h, w), dtype=np.uint16)
        weight_f32 = rng.uniform(0.0, 5000.0, size=(h, w)).astype(np.float32)
        # 500-layer sparse winner case: few pixels improve over existing best
        best_u16_base = np.full((h, w), 60000, dtype=np.uint16)
        weight_u16_sparse = np.zeros((h, w), dtype=np.uint16)
        weight_u16_sparse[1, 3:71] = np.uint16(61234)
        best_f32_base = np.full((h, w), 4500.0, dtype=np.float32)
        weight_f32_sparse = np.where(weight_f32 > 4500.0, weight_f32, np.float32(100.0))
        fsrc3_unit = rng.uniform(-0.2, 1.2, size=(h, w, 3)).astype(np.float32)

        def _run_c() -> tuple[np.ndarray, ...]:
            d_3 = np.empty((ch, cw, 3), dtype=np.float32)
            d_1 = np.empty((ch, cw), dtype=np.float32)
            sc3 = np.empty((h, cw, 3), dtype=np.float32)
            sc1 = np.empty((h, cw), dtype=np.float32)
            mod.of_pyr_down_f32(fsrc3, d_3, sc3, 0, ch)
            mod.of_pyr_down_f32(fsrc1, d_1, sc1, 0, ch)

            acc0_f = np.empty((h, w, 3), dtype=np.float32)
            best0_f = np.empty((h, w), dtype=np.uint16)
            win0_f = np.full((h, w), 99, dtype=np.uint16)
            sc_up = np.empty((ch, w, 3), dtype=np.float32)
            mod.of_pyr_level0_merge_from_raw(
                coarse3, u16_3, weight_u16, acc0_f, best0_f, win0_f, sc_up, 257.0, True, 0, 0, h
            )

            acc0_s = fsrc3.copy()
            best0_s = best_u16_base.copy()
            win0_s = np.full((h, w), 123, dtype=np.uint16)
            mod.of_pyr_level0_merge_from_raw(
                coarse3, u16_3, weight_u16_sparse, acc0_s, best0_s, win0_s, sc_up, 257.0, False, 499, 0, h
            )

            acc_m = np.zeros((h, w, 3), dtype=np.float32)
            best_m = best_f32_base.copy()
            win_m = np.full((h, w), 77, dtype=np.uint16)
            mod.of_pyr_merge_f32(fsrc3, weight_f32_sparse, acc_m, best_m, win_m, False, 499, 0, h)

            rec3 = fsrc3.copy()
            rec1 = fsrc1.copy()
            sc_r1 = np.empty((ch, w), dtype=np.float32)
            mod.of_pyr_reconstruct_f32(coarse3, rec3, sc_up, 0, h)
            mod.of_pyr_reconstruct_f32(coarse1, rec1, sc_r1, 0, h)

            out_u8 = np.empty((h, w, 3), dtype=np.uint8)
            out_f32 = np.empty((h, w, 3), dtype=np.float32)
            out_unit_u8 = np.empty((h, w, 3), dtype=np.uint8)
            out_unit_f32 = np.empty((h, w, 3), dtype=np.float32)
            mod.of_pyr_output_f32(fsrc3, out_u8, True, 0, h)
            mod.of_pyr_output_f32(fsrc3, out_f32, False, 0, h)
            mod.of_pyr_output_f32(fsrc3_unit, out_unit_u8, True, 0, h)
            mod.of_pyr_output_f32(fsrc3_unit, out_unit_f32, False, 0, h)
            return d_3, d_1, acc0_f, best0_f, win0_f, acc0_s, best0_s, win0_s, acc_m, best_m, win_m, rec3, rec1, out_u8, out_f32, out_unit_u8, out_unit_f32

        try:
            mod.set_force_scalar(False)
            res_avx = _run_c()
            mod.set_force_scalar(True)
            res_sca = _run_c()
        finally:
            mod.set_force_scalar(False)

        for idx, (arr_a, arr_s) in enumerate(zip(res_avx, res_sca)):
            assert np.array_equal(arr_a.view(np.uint8), arr_s.view(np.uint8)), f"mismatch at {idx} for {(h, w)}"

