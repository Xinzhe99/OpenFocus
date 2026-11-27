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
    if mod.simd_info() not in ("AVX2", "AVX2+FMA"):
        pytest.skip(f"Host CPU/OS reports {mod.simd_info()}, not AVX2")
    return mod


def test_simd_info_and_force_scalar_toggle():
    mod = _require_avx2()
    try:
        assert mod.simd_info() in ("AVX2", "AVX2+FMA")
        mod.set_force_scalar(True)
        assert mod.simd_info() == "Scalar"
        mod.set_force_scalar(False)
        assert mod.simd_info() in ("AVX2", "AVX2+FMA")
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
        if channels == 1 and w >= 8:
            img[0, :8] = [np.nan, -0.0, np.inf, -np.inf, -1.0, 0.0, 50.0, 255.0]
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


@pytest.mark.parametrize("kind", ["random", "bright", "maximum"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_exposure_mean_avx2_vs_scalar_bit_exact(kind, dtype):
    """Bright wide rows exceed the exactly summable range and fall back per row."""
    mod = _require_avx2()
    rng = np.random.default_rng(930)
    top = 255 if dtype == np.uint8 else 65535
    means_simd, means_scalar = [], []
    for shape in ((17, 33, 3), (6, 5184, 3), (5, 2049, 3)):
        if kind == "random":
            image = rng.integers(0, top + 1, shape, dtype=dtype)
        elif kind == "bright":
            image = rng.integers(top - top // 8, top + 1, shape, dtype=dtype)
        else:
            image = np.full(shape, top, dtype=dtype)

        def run():
            values = []
            for normalize in (False, True):
                for threads in (1, 3):
                    values.append(mod.of_exposure_mean(image, normalize, threads))
                    values.append(mod.of_exposure_mean(image[:, :, ::-1], normalize, threads))
                    values.append(mod.of_exposure_mean(np.ascontiguousarray(image[:, :, 0]), normalize, threads))
                    values.append(mod.of_exposure_mean(image[:, ::2], normalize, threads))
            return values

        try:
            mod.set_force_scalar(False)
            means_simd += run()
            mod.set_force_scalar(True)
            means_scalar += run()
        finally:
            mod.set_force_scalar(False)
    assert means_simd == means_scalar
    if dtype == np.uint8:
        expected = float(image.astype(np.float64).mean())
        assert abs(means_simd[-4] - expected) <= 1e-9 * max(1.0, expected)


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


def test_merge_first_frame_copy_matches_scaled_source():
    """The first frame is copied as typed rows when both layouts are packed."""
    from core import _native

    mod = _native.load_native()
    assert mod is not None
    rng = np.random.default_rng(940)
    depth = np.zeros((9, 21), dtype=np.uint16)
    for dtype, scale in ((np.uint8, 256), (np.uint8, 1), (np.uint16, 1)):
        rgb = rng.integers(0, np.iinfo(dtype).max + 1, (9, 21, 3), dtype=dtype)
        for source in (rgb, rgb[:, :, ::-1], np.ascontiguousarray(rgb[:, :, 0]), rgb[::-1]):
            for acc_dtype in (np.uint16, np.int32):
                acc = np.full(source.shape, 7, dtype=acc_dtype)
                mod.of_merge_rows(acc, source, depth, 0, 100, 2, 7, scale, True, False)
                expected = np.full(source.shape, 7, dtype=acc_dtype)
                expected[2:7] = source[2:7].astype(acc_dtype) * scale
                np.testing.assert_array_equal(acc, expected)


def test_merge_first_frame_copy_keeps_element_order_for_overlapping_views():
    """An accumulator that shares memory with the frame is filled one element at a time."""
    from core import _native

    mod = _native.load_native()
    assert mod is not None
    rng = np.random.default_rng(941)
    h, w = 5, 9
    depth = np.zeros((h, w), dtype=np.uint16)
    base = rng.integers(0, 65536, (h + 1, w * 3 + 3), dtype=np.uint16)

    def views(buffer, shift_rows, shift_samples):
        flat = buffer.reshape(-1)
        row = buffer.shape[1]
        source = np.lib.stride_tricks.as_strided(
            flat, shape=(h, w, 3), strides=(row * 2, 6, 2))
        target = np.lib.stride_tricks.as_strided(
            flat[shift_rows * row + shift_samples:], shape=(h, w, 3), strides=(row * 2, 6, 2))
        return source, target

    for scale in (1, 3):
        for shift_rows, shift_samples in ((0, 0), (0, 1), (0, 3), (1, 0), (1, 2)):
            actual = base.copy()
            source, target = views(actual, shift_rows, shift_samples)
            mod.of_merge_rows(target, source, depth, 0, 100, 0, h, scale, True, False)
            expected = base.copy()
            source, target = views(expected, shift_rows, shift_samples)
            for y in range(h):
                for x in range(w):
                    for ch in range(3):
                        target[y, x, ch] = (int(source[y, x, ch]) * scale) & 0xFFFF
            np.testing.assert_array_equal(actual, expected)


def test_ab_gray_and_downsample_preserve_wide_codes():
    mod = _require_avx2()
    rgb = np.full((2, 17, 3), 40000, dtype=np.uint16)
    gray = np.empty(rgb.shape[:2], dtype=np.uint16)
    mod.of_gray_rows(rgb, gray, 0, 2, 2, False)
    np.testing.assert_array_equal(gray, np.full(gray.shape, 14464, dtype=np.uint16))

    source = np.full((2, 17), 1 << 30, dtype=np.int32)
    source[:, -1] = -(1 << 30)
    down = np.empty((1, 9), dtype=np.int32)
    mod.of_down2_rows(source, down, 0, 1, 1)
    np.testing.assert_array_equal(down, [[1 << 30] * 8 + [-(1 << 30)]])


@pytest.mark.parametrize("w", [7, 8, 9, 23, 64, 131])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_gray_rows_rgb_sum_avx2_vs_scalar_bit_exact(w, dtype):
    """Packed RGB rows are deinterleaved with shuffles, eight pixels at a time."""
    mod = _require_avx2()
    rng = np.random.default_rng(920 + w)
    h = 9
    hi = 256 if dtype == np.uint8 else 65536
    rgb = rng.integers(0, hi, (h, w, 3), dtype=dtype)
    rgb[0] = hi - 1
    scale = 256 if dtype == np.uint8 else 1

    def run():
        results = []
        for depth_color in (False, True):
            for out_dtype in (np.uint16, np.int32):
                out = np.empty((h, w), dtype=out_dtype)
                mod.of_gray_rows(rgb, out, 0, h, scale, depth_color)
                results.append(out)
        return results

    try:
        mod.set_force_scalar(False)
        simd = run()
        mod.set_force_scalar(True)
        scalar = run()
    finally:
        mod.set_force_scalar(False)
    for a, b in zip(simd, scalar):
        np.testing.assert_array_equal(a, b)
    total = rgb.astype(np.int64).sum(axis=2)
    np.testing.assert_array_equal(simd[1], (total // 3) * scale)
    np.testing.assert_array_equal(simd[3], (total * scale * 85) >> 8)


def test_activity_normalised_difference_matches_integer_division():
    """The vector path divides in float and corrects; the result is a floor quotient."""
    mod = _require_avx2()
    rng = np.random.default_rng(911)
    special = np.array([0, 1, 2, 9, 10, 11, 254, 255, 256, 655, 656, 1310, 32767, 32768,
                        65279, 65280, 65534, 65535], dtype=np.uint16)
    values = np.concatenate([special, rng.integers(0, 65536, 238, dtype=np.uint16)])
    gray = np.ascontiguousarray(np.broadcast_to(values[:, None], (values.size, values.size)))
    fine = np.ascontiguousarray(np.broadcast_to(values[None, :], (values.size, values.size)))

    def run():
        out = np.empty_like(gray)
        mod.of_activity_rows(gray, fine, out, 0, gray.shape[0])
        detail = np.empty((gray.shape[0] // 2, gray.shape[1] // 2), dtype=np.uint16)
        mod.of_detail_rows(gray + fine // 3, detail, 0)
        return out, detail

    try:
        mod.set_force_scalar(False)
        simd = run()
        mod.set_force_scalar(True)
        scalar = run()
    finally:
        mod.set_force_scalar(False)
    for a, b in zip(simd, scalar):
        np.testing.assert_array_equal(a, b)
    v = gray.astype(np.int64)
    mean = fine.astype(np.int64)
    divisor = 10 + (np.maximum(v, mean) * 100) // 65535
    expected = np.minimum(65535, (110 * np.abs(v - mean)) // divisor)
    np.testing.assert_array_equal(simd[0], expected.astype(np.uint16))


@pytest.mark.parametrize('channels', [1, 3])
def test_float_tile_finalize_handles_nan_in_vector_and_tail(channels):
    mod = _require_avx2()
    values = np.array([[np.nan, -np.inf, np.inf, -0.0, 1, 2, 3, 4, np.nan]], dtype=np.float32)
    expected = np.array([[0, 0, 255, 0, 1, 2, 3, 4, 0]], dtype=np.float32)
    if channels == 3:
        values = np.repeat(values[..., None], channels, axis=2)
        expected = np.repeat(expected[..., None], channels, axis=2)
    output = np.empty_like(values)
    mod.of_tile_finalize(values, np.ones((1, 9), dtype=np.float32), output, 0, 1)
    np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))


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
    for h, w in [(33, 129), (12, 7680)]:
        cw = (w + 1) // 2
        ch = (h + 1) // 2
        fsrc3 = rng.uniform(-50.0, 65600.0, size=(h, w, 3)).astype(np.float32)
        fsrc1 = rng.uniform(-50.0, 65600.0, size=(h, w)).astype(np.float32)
        u16_3 = rng.integers(0, 65536, size=(h, w, 3), dtype=np.uint16)
        coarse3 = rng.uniform(0.0, 65535.0, size=(ch, cw, 3)).astype(np.float32)
        coarse1 = rng.uniform(0.0, 65535.0, size=(ch, cw)).astype(np.float32)
        weight_u16 = rng.integers(0, 65535, size=(h, w), dtype=np.uint16)
        # 500-layer sparse winner case: few pixels improve over existing best
        best_u16_base = np.full((h, w), 60000, dtype=np.uint16)
        weight_u16_sparse = np.zeros((h, w), dtype=np.uint16)
        weight_u16_sparse[4, 3:71] = 61234
        weight_u16_sparse[4, -1] = 61234
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
            assert best0_s[4, -1] == 61234 and win0_s[4, -1] == 499

            acc_m = np.zeros((h, w, 3), dtype=np.float32)
            best_m = best_u16_base.copy()
            win_m = np.full((h, w), 77, dtype=np.uint16)
            mod.of_pyr_merge_f32(fsrc3, weight_u16_sparse, acc_m, best_m, win_m, False, 499, 0, h)
            assert best_m[4, -1] == 61234 and win_m[4, -1] == 499

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


def test_float_pyramid_output_and_upsample_preserve_special_values():
    mod = _require_avx2()
    pixels = np.array([np.nan, -0.0, -np.inf, np.inf, 0.0, -1.0, 1.0, -0.0, np.nan],
                      dtype=np.float32).reshape(1, 3, 3)
    output = np.empty_like(pixels)
    mod.of_pyr_output_f32(pixels, output, False, 0, 1)
    expected = np.array([np.nan, -0.0, 0.0, 255.0, 0.0, 0.0, 255.0, -0.0, np.nan],
                        dtype=np.float32).reshape(1, 3, 3)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(np.signbit(output), np.signbit(expected))

    coarse = np.full((5, 6, 3), -0.0, dtype=np.float32)
    target = np.full((9, 11, 3), -0.0, dtype=np.float32)
    scratch = np.empty((5, 11, 3), dtype=np.float32)
    mod.of_pyr_reconstruct_f32(coarse, target, scratch, 0, target.shape[0])
    np.testing.assert_array_equal(target[4, 4:6].view(np.uint32), np.zeros((2, 3), dtype=np.uint32))


def test_alignment_kernels_simd_and_8k_parity() -> None:
    mod = _require_avx2()
    rng = np.random.default_rng(112233)

    # Full sweep across odd, vector-tail, and 8K widths plus 8K region coordinates
    for w in [7, 13, 33, 129, 7680]:
        rgb8_3 = rng.integers(0, 256, size=(4, w, 3), dtype=np.uint8)
        rgb8_4 = rng.integers(0, 256, size=(4, w, 4), dtype=np.uint8)
        rgb16_3 = rng.integers(0, 65536, size=(4, w, 3), dtype=np.uint16)
        rgb16_3[0, :4] = 65535
        rgb16_4 = rng.integers(0, 65536, size=(4, w, 4), dtype=np.uint16)
        dh_src = rng.integers(0, 65536, size=(12, w), dtype=np.uint16)
        dh_src[0:4, :8] = 65535
        w1 = rng.integers(0, 256, size=(16, w), dtype=np.uint8)
        w3_u16 = rng.integers(0, 65536, size=(16, w, 3), dtype=np.uint16)
        rw = max(17, min(w, 129))

        def _run_align_clean(*, strided_translations=False):
            g8_3 = np.empty((4, w), dtype=np.uint16)
            g8_4 = np.empty((4, w), dtype=np.uint16)
            g16_3 = np.empty((4, w), dtype=np.uint16)
            g16_4 = np.empty((4, w), dtype=np.uint16)
            mod.of_align_gray16_rows(rgb8_3, g8_3, 0, 4)
            mod.of_align_gray16_rows(rgb8_4, g8_4, 0, 4)
            mod.of_align_gray16_rows(rgb16_3, g16_3, 0, 4)
            mod.of_align_gray16_rows(rgb16_4, g16_4, 0, 4)

            dh_out = np.empty((6, (w + 1) // 2), dtype=np.uint16)
            fd_out = np.empty((6, (w + 1) // 2), dtype=np.uint16)
            mod.of_align_down_half_rows(dh_src, dh_out, 0, 6)
            mod.of_align_fast_downsize_u16_rows(dh_src, fd_out, 0, 6)

            mask = np.empty((16, w), dtype=np.uint8)
            mod.of_align_valid_mask_rows(
                mask, 16, w, 3.25, -1.75, float(w) * 0.5, 8.0, 0.015, 1.002, -0.015, 1.002, 0, 16
            )

            warp_1 = np.empty((16, w), dtype=np.uint8)
            warp_3 = np.empty((16, w, 3), dtype=np.uint16)
            warp_3_bgr = np.empty((16, w, 3), dtype=np.uint16)
            step = 2 if strided_translations else 1
            trans_3 = np.empty((16, w * step, 3), dtype=np.uint16)[:, ::step]
            trans_reg = np.empty((8, rw * step, 3), dtype=np.uint16)[:, ::step]
            mod.of_align_warp_alignment_int_rows(
                w1, warp_1, 3.25, -1.75, float(w) * 0.5, 8.0, 0.015, 1.002, -0.015, 1.002, 0, 16, False
            )
            mod.of_align_warp_alignment_int_rows(
                w3_u16, warp_3, 3.25, -1.75, float(w) * 0.5, 8.0, 0.015, 1.002, -0.015, 1.002, 0, 16, False
            )
            mod.of_align_warp_alignment_int_rows(
                w3_u16, warp_3_bgr, 3.25, -1.75, float(w) * 0.5, 8.0, 0.015, 1.002, -0.015, 1.002, 0, 16, True
            )
            mod.of_align_warp_int_rows(w3_u16, trans_3, 2.25, -1.5, 0, 16, True)
            mod.of_align_warp_translation_region_int_rows(
                w3_u16, trans_reg, 16, w, 0, 2, 1, 1.25, -0.5, 0, 8, False
            )

            reg_src_w = max(w, 7680)
            reg_out = np.empty((8, rw, 3), dtype=np.uint16)
            # In-bounds region warp at 8K coordinates when w >= 7680
            if w == 7680:
                mod.of_align_warp_alignment_region_int_rows(
                    w3_u16, reg_out, 4320, reg_src_w, 2100, 4096, 2102,
                    3837.5, 2106.5, 3840.0, 2105.0, 0.002, 1.001, -0.002, 1.001, 0, 8, True
                )
            else:
                mod.of_align_warp_alignment_region_int_rows(
                    w3_u16, reg_out, 16, w, 0, 2, 1,
                    1.2, -0.7, float(w) * 0.5, 8.0, 0.01, 1.0, -0.01, 1.0, 0, 8, False
                )
            return g8_3, g8_4, g16_3, g16_4, dh_out, fd_out, mask, warp_1, warp_3, warp_3_bgr, trans_3, trans_reg, reg_out

        try:
            mod.set_force_scalar(False)
            res_avx = _run_align_clean()
            mod.set_force_scalar(True)
            res_sca = _run_align_clean(strided_translations=True)
        finally:
            mod.set_force_scalar(False)

        for idx, (arr_a, arr_s) in enumerate(zip(res_avx, res_sca)):
            assert np.array_equal(arr_a.view(np.uint8), arr_s.view(np.uint8)), f"align mismatch {idx} w={w}"

    # 4. Score chunk & score grid 4d (non-8-multiple nxx=21, partial/OOB candidates, 65535 extremes)
    ref_g = rng.integers(0, 65536, size=(24, 21), dtype=np.uint16)
    ref_g[:4, :8] = 65535
    img_b = rng.integers(0, 65536, size=(96, 128), dtype=np.uint16)
    img_b[:8, :8] = 65535
    xs = np.linspace(-4.0, 122.0, 21, dtype=np.float32)
    ys = np.linspace(-2.0, 90.0, 24, dtype=np.float32)
    kc = 16
    p08 = rng.uniform(-0.04, 0.04, kc).astype(np.float32)
    p0c = p08.copy()
    p18 = rng.uniform(0.95, 1.05, kc).astype(np.float32)
    p1c = p18.copy()
    tx = rng.uniform(-10.0, 10.0, kc).astype(np.float32)
    ty = rng.uniform(-10.0, 10.0, kc).astype(np.float32)
    # Force an out-of-bounds candidate and boundary candidate
    tx[3] = np.float32(500.0)
    ty[3] = np.float32(-500.0)
    cx = rng.uniform(40.0, 80.0, kc).astype(np.float32)
    cy = rng.uniform(30.0, 60.0, kc).astype(np.float32)
    cidx = np.arange(kc, dtype=np.int64)

    cx_tab = np.array([50.0, 60.0, 70.0], dtype=np.float32)
    cy_tab = np.array([40.0, 48.0, 56.0], dtype=np.float32)
    p08_tab = np.array([[-0.02, 0.0, 0.02], [-0.01, 0.01, 0.0]], dtype=np.float32)
    p18_tab = np.array([[0.99, 1.0, 1.02], [1.01, 1.0, 0.98]], dtype=np.float32)
    n0, n1, n2, n3 = 3, 3, 3, 2

    def _run_scores():
        ssum = np.zeros(kc, dtype=np.int64)
        ssq = np.zeros(kc, dtype=np.int64)
        cnt = np.zeros(kc, dtype=np.int64)
        mod.of_align_score_chunk_u16_slice(
            ref_g, img_b, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy, cidx, 0, kc, ssum, ssq, cnt
        )
        ok = np.empty(2, dtype=np.int64)
        osc = np.empty(2, dtype=np.float32)
        mod.of_align_score_grid_4d_slice(
            ref_g, img_b, xs, ys, 60.0, 48.0, cx_tab, cy_tab, p08_tab, p18_tab,
            n0, n1, n2, n3, 0, n0 * n1 * n2 * n3, ok, osc, 0
        )
        mod.of_align_score_grid_4d_slice(
            ref_g, img_b, xs, ys, 5000.0, -5000.0, cx_tab, cy_tab, p08_tab, p18_tab,
            n0, n1, n2, n3, 0, n0 * n1 * n2 * n3, ok, osc, 1
        )
        return ssum, ssq, cnt, ok, osc

    try:
        mod.set_force_scalar(False)
        sc_avx = _run_scores()
        mod.set_force_scalar(True)
        sc_sca = _run_scores()
    finally:
        mod.set_force_scalar(False)

    for idx, (arr_a, arr_s) in enumerate(zip(sc_avx, sc_sca)):
        assert np.array_equal(arr_a.view(np.uint8), arr_s.view(np.uint8)), f"score mismatch {idx}"


def test_alignment_region_preserves_coordinates_outside_float32_integer_range():
    mod = _require_avx2()
    image = np.tile(np.arange(1, 17, dtype=np.uint16), (2, 1))
    output = np.empty((1, 8), dtype=np.uint16)
    mod.of_align_warp_alignment_region_int_rows(
        image, output, 2, 16, 0, -16777217, 0,
        0.0, 0.0, -16777216.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, 1, False,
    )
    np.testing.assert_array_equal(output, [[1, 1, 2, 3, 4, 5, 6, 7]])


def test_ab_depth_resample_and_8k_tile_acc_parity() -> None:
    mod = _require_avx2()
    rng = np.random.default_rng(121212)

    for w in [19, 129, 7680]:
        h = 12
        # 1. Activity, detail, smooth detail, sparse 500-layer activity/detail selection
        g16 = rng.integers(0, 65536, size=(h, w), dtype=np.uint16)
        f16 = rng.integers(0, 65536, size=(h, w), dtype=np.uint16)
        g16[0, :8] = 65535
        f16[0, :8] = 0
        best_a_init = np.full((h, w), 60000, dtype=np.uint16)
        act_sparse = np.zeros((h, w), dtype=np.uint16)
        act_sparse[3, 5:45] = np.uint16(62500)
        sec_init = np.full((h, w), 50000, dtype=np.uint16)
        act_det_sparse = np.full((h // 2, w // 2), 10000, dtype=np.uint16)
        act_det_sparse[1, 3:40] = np.uint16(55000)
        act_det_sparse[2, 3:40] = np.uint16(63000)
        sec_init[4, : w // 2] = 64000
        act_det_sparse[4] = 62000

        # 2. Half-axis u16 resample with repeating columns
        src_r = rng.integers(0, 65536, size=(h, w), dtype=np.uint16)
        ow = max(4, (w - 4) // 2)
        oh = 4
        v_idx = np.zeros((oh, 4), dtype=np.int32)
        v_w = np.zeros((oh, 4), dtype=np.float32)
        v_cnt = np.full(oh, 4, dtype=np.int32)
        for y in range(oh):
            v_idx[y] = np.clip(np.arange(4) + y, 0, h - 1)
            v_w[y] = np.array([0.1, 0.4, 0.4, 0.1], dtype=np.float32)
        h_idx = np.zeros((ow, 4), dtype=np.int32)
        h_w = np.zeros((ow, 4), dtype=np.float32)
        h_cnt = np.full(ow, 4, dtype=np.int32)
        for x in range(ow):
            h_idx[x] = np.arange(4) + 2 * x
            h_w[x] = np.array([0.125, 0.375, 0.375, 0.125], dtype=np.float32)

        # 3. Output u16->u8 / f32 + 3ch tile_accumulate (f32, u8, u16)
        code3 = rng.integers(0, 65536, size=(4, w, 3), dtype=np.uint16)
        code4 = rng.integers(0, 65536, size=(4, w, 4), dtype=np.uint16)
        tile_f32 = rng.uniform(0.0, 255.0, size=(4, w, 3)).astype(np.float32)
        tile_u8 = rng.integers(0, 256, size=(4, w, 3), dtype=np.uint8)
        tile_u16 = rng.integers(0, 65536, size=(4, w, 3), dtype=np.uint16)
        w2d = rng.uniform(0.1, 1.0, size=(4, w)).astype(np.float32)

        def _run():
            act = np.empty((h, w), dtype=np.uint16)
            mod.of_activity_rows(g16, f16, act, 0, h)
            det = np.empty((h // 2, w // 2), dtype=np.uint16)
            mod.of_detail_rows(g16, det, 0)
            sm_det = np.empty((h // 2, w // 2), dtype=np.uint16)
            mod.of_smooth_detail_rows(det, sm_det, 0, h // 2)

            b_act = best_a_init.copy()
            d_act = np.full((h, w), 7, dtype=np.uint16)
            v_act = g16.copy()
            mod.of_select_activity_rows(act_sparse, b_act, d_act, v_act, 499, 1073741824, 0, h)

            b_det = best_a_init[: h // 2, : w // 2].copy()
            s_det = sec_init[: h // 2, : w // 2].copy()
            w_det = np.full((h // 2, w // 2), 9, dtype=np.uint16)
            mod.of_select_detail_rows(act_det_sparse, b_det, s_det, w_det, 499, 0, h // 2)

            out_res = np.empty((oh, ow), dtype=np.uint16)
            mod.of_resample_rows_u16(src_r, out_res, v_idx, v_w, v_cnt, h_idx, h_w, h_cnt, 0, oh)

            u8_3_f = np.empty((4, w, 3), dtype=np.uint8)
            u8_3_t = np.empty((4, w, 3), dtype=np.uint8)
            u8_4_t = np.empty((4, w, 4), dtype=np.uint8)
            f32_3 = np.empty((4, w, 3), dtype=np.float32)
            mod.of_output_u16_to_u8_rows(code3, u8_3_f, 0, 4, False)
            mod.of_output_u16_to_u8_rows(code3, u8_3_t, 0, 4, True)
            mod.of_output_u16_to_u8_rows(code4, u8_4_t, 0, 4, True)
            mod.of_output_u16_to_f32_rows(code3, f32_3, 0, 4, 1.0 / 257.0)

            acc_f = np.zeros((4, w, 3), dtype=np.float32)
            w_f = np.zeros((4, w), dtype=np.float32)
            acc_8 = np.zeros((4, w, 3), dtype=np.float32)
            w_8 = np.zeros((4, w), dtype=np.float32)
            acc_16 = np.zeros((4, w, 3), dtype=np.float32)
            w_16 = np.zeros((4, w), dtype=np.float32)
            for weights in (w2d, w2d[::-1]):
                mod.of_tile_accumulate(acc_f, w_f, tile_f32, weights, 0, 0, 2)
                mod.of_tile_accumulate(acc_8, w_8, tile_u8, weights, 0, 0, 2)
                mod.of_tile_accumulate(acc_16, w_16, tile_u16, weights, 0, 0, 2)

            return act, det, sm_det, b_act, d_act, v_act, b_det, s_det, w_det, out_res, u8_3_f, u8_3_t, u8_4_t, f32_3, acc_f, w_f, acc_8, w_8, acc_16, w_16

        try:
            mod.set_force_scalar(False)
            res_a = _run()
            mod.set_force_scalar(True)
            res_s = _run()
        finally:
            mod.set_force_scalar(False)

        for idx, (arr_a, arr_s) in enumerate(zip(res_a, res_s)):
            assert np.array_equal(arr_a.view(np.uint8), arr_s.view(np.uint8)), f"depth/tile mismatch {idx} w={w}"


def test_activity_preserves_shifted_input_view() -> None:
    mod = _require_avx2()
    gray = np.zeros((1, 8), dtype=np.uint16)
    inputs = np.zeros((1, 9), dtype=np.uint16)
    inputs[0, 0] = 1
    actual = inputs.copy()
    try:
        mod.set_force_scalar(False)
        mod.of_activity_rows(gray, actual[:, :8], actual[:, 1:], 0, 1)
        mod.set_force_scalar(True)
        mod.of_activity_rows(gray, inputs[:, :8], inputs[:, 1:], 0, 1)
    finally:
        mod.set_force_scalar(False)
    np.testing.assert_array_equal(actual, inputs)


def test_rgb_tile_accumulation_preserves_shifted_input_view() -> None:
    mod = _require_avx2()
    acc = np.zeros((1, 9, 3), dtype=np.float32)
    acc[:, 0] = 1
    weights = np.ones((1, 8), dtype=np.float32)
    weight_acc = np.zeros((1, 9), dtype=np.float32)

    mod.of_tile_accumulate(acc, weight_acc, acc[:, :8], weights, 1, 0, 1)

    np.testing.assert_array_equal(acc, np.ones_like(acc))


def test_overlapping_kernel_buffers_match_scalar() -> None:
    mod = _require_avx2()
    rng = np.random.default_rng(1)
    h, w = 12, 40
    ch, cw = 6, 20
    plane = rng.integers(0, 65536, (17, 34), dtype=np.uint16)
    rgb = rng.integers(0, 65536, (5, 28), dtype=np.uint16)
    raw = rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)
    fine = rng.uniform(0.0, 65535.0, (h, w, 3)).astype(np.float32)
    coarse = rng.uniform(0.0, 65535.0, (ch, cw, 3)).astype(np.float32)
    levels = rng.uniform(0.0, 65535.0, (ch, w + 1, 3)).astype(np.float32)
    target = rng.random((h, w, 3), dtype=np.float32)
    codes = rng.integers(0, 65535, (h, w + 1), dtype=np.uint16)

    def iir_separate():
        out = np.empty_like(plane)
        mod.of_depth_axis_iir(plane, None, out, 3072, 512, 0, 0, 3, 1, 1)
        return [out]

    def iir_in_place():
        b = plane.copy()
        mod.of_depth_axis_iir(b, None, b, 3072, 512, 0, 0, 3, 0, 1)
        return [b]

    def iir_vertical():
        b = plane.copy()
        mod.of_depth_axis_iir(b[:16, :33], None, b[:16, 1:], 3072, 512, 0, 0, 3, 0, 1)
        return [b]

    def iir_horizontal():
        b = plane.copy()
        mod.of_depth_axis_iir(b[:16, :33], None, b[1:, :33], 3072, 512, 0, 0, 3, 1, 1)
        return [b]

    def iir_guided_shifted():
        b = plane.copy()
        src = np.ascontiguousarray(plane[:16, :33][::-1])
        mod.of_depth_axis_iir(src, b[:16, :33], b[:16, 1:], 0, 0, 1024, 256, 3, 0, 1)
        return [b]

    def iir_guided_inplace():
        b = plane.copy()
        guide = np.ascontiguousarray(plane[::-1])
        mod.of_depth_axis_iir(b, guide, b, 0, 0, 1024, 256, 3, 1, 1)
        return [b]

    def gray():
        b = rgb.copy()
        image = np.lib.stride_tricks.as_strided(b, shape=(5, 8, 3), strides=(b.strides[0], 6, 2))
        mod.of_gray_rows(image, b[:, 8:16], 0, 5, 1, False)
        return [b]

    def down():
        sb = np.zeros((h, cw + 1, 3), np.float32)
        mod.of_pyr_down_f32(fine, sb[:ch, 1:], sb[:, :cw], 0, ch)
        return [sb]

    def down_from_raw():
        buffer = bytearray(raw.tobytes())
        src = np.frombuffer(buffer, np.uint16).reshape(h, w, 3)
        scratch = np.frombuffer(buffer, np.float32).reshape(h, cw, 3)
        out = np.zeros((ch, cw, 3), np.float32)
        mod.of_pyr_down_from_raw(src, out, scratch, 1.0 / 65535.0, 0, ch)
        return [np.frombuffer(bytes(buffer), np.uint8), out]

    def reconstruct():
        b = levels.copy()
        rec = target.copy()
        mod.of_pyr_reconstruct_f32(b[:, 1:1 + cw], rec, b[:, :w], 0, h)
        return [b, rec]

    def detail_weight():
        b = levels.copy()
        det = target.copy()
        weights = np.zeros((h, w), np.uint16)
        mod.of_pyr_detail_weight_f32(b[:, 1:1 + cw], det, weights, b[:, :w], 0, h)
        return [b, det, weights]

    def level0_weights():
        b = levels.copy()
        weights = np.zeros((h, w), np.uint16)
        mod.of_pyr_level0_weights_from_raw(b[:, 1:1 + cw], raw, weights, b[:, :w], 1.0 / 65535.0, 0, h)
        return [b, weights]

    def level0_merge_first():
        c = codes.copy()
        acc = np.zeros((h, w, 3), np.float32)
        winner = np.full((h, w), 7, np.uint16)
        scratch = np.zeros((ch, w, 3), np.float32)
        mod.of_pyr_level0_merge_from_raw(coarse, raw, c[:, :w], acc, c[:, 1:], winner, scratch, 1.0, True, 3, 0, h)
        return [c, acc, winner]

    def level0_merge_next():
        c = codes.copy()
        acc = np.zeros((h, w, 3), np.float32)
        winner = np.full((h, w), 7, np.uint16)
        scratch = np.zeros((ch, w, 3), np.float32)
        mod.of_pyr_level0_merge_from_raw(coarse, raw, c[:, :w], acc, c[:, 1:], winner, scratch, 1.0, False, 3, 0, h)
        return [c, acc, winner]

    def level0_merge_winner_overlap():
        c = codes.copy()
        acc = np.zeros((h, w, 3), np.float32)
        best = np.zeros((h, w), np.uint16)
        scratch = np.zeros((ch, w, 3), np.float32)
        mod.of_pyr_level0_merge_from_raw(coarse, raw, c[:, :w], acc, best, c[:, 1:], scratch, 1.0, False, 3, 0, h)
        return [c, acc, best]

    def pyr_merge_overlap():
        c = codes.copy()
        det = target.copy()
        acc = np.zeros((h, w, 3), np.float32)
        winner = np.full((h, w), 5, np.uint16)
        mod.of_pyr_merge_f32(det, c[:, :w], acc, c[:, 1:], winner, False, 4, 0, h)
        return [c, acc, winner]

    for name, run in (
        ("of_depth_axis_iir separate", iir_separate),
        ("of_depth_axis_iir in_place", iir_in_place),
        ("of_depth_axis_iir vertical shifted", iir_vertical),
        ("of_depth_axis_iir horizontal shifted", iir_horizontal),
        ("of_depth_axis_iir guided shifted", iir_guided_shifted),
        ("of_depth_axis_iir guided in_place", iir_guided_inplace),
        ("of_gray_rows inside rgb", gray),
        ("of_pyr_down_f32 inside scratch", down),
        ("of_pyr_down_from_raw scratch over src", down_from_raw),
        ("of_pyr_reconstruct_f32 coarse shifted", reconstruct),
        ("of_pyr_detail_weight_f32 coarse shifted", detail_weight),
        ("of_pyr_level0_weights_from_raw coarse shifted", level0_weights),
        ("of_pyr_level0_merge_from_raw first", level0_merge_first),
        ("of_pyr_level0_merge_from_raw next", level0_merge_next),
        ("of_pyr_level0_merge_from_raw winner overlap", level0_merge_winner_overlap),
        ("of_pyr_merge_f32 overlap", pyr_merge_overlap),
    ):
        simd, scalar = _scalar_and_simd(mod, run)
        for idx, (arr_a, arr_s) in enumerate(zip(simd, scalar)):
            assert np.array_equal(arr_a.view(np.uint8), arr_s.view(np.uint8)), f"{name} mismatch at {idx}"


def test_ecc_solver_avx2_parity() -> None:
    import cv2
    from core._ecc_native import find_transform_ecc_homography

    mod = _require_avx2()
    rng = np.random.default_rng(131313)

    for (h, w), (tx, ty, scale, ang, px, py) in [
        ((163, 251), (2.4, -1.7, 1.006, 0.35, 0.0, 0.0)),
        ((576, 1024), (1.8, 1.2, 1.004, -0.25, 6e-6, -5e-6)),
    ]:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = (
            0.5
            + 0.2 * np.sin(xx * (2.0 * np.pi / 43.0)) * np.cos(yy * (2.0 * np.pi / 37.0))
            + 0.12 * np.cos((xx * 0.7 + yy * 1.2) * (2.0 * np.pi / 19.0))
            + 0.08 * rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        )
        tpl = np.clip(cv2.GaussianBlur(base, (0, 0), 1.4), 0.0, 1.0).astype(np.float32)
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        th = np.deg2rad(ang)
        c, s = np.cos(th) * scale, np.sin(th) * scale
        M = np.array(
            [
                [c, -s, tx + cx - (c * cx - s * cy)],
                [s, c, ty + cy - (s * cx + c * cy)],
                [px, py, 1.0],
            ],
            dtype=np.float64,
        )
        src = cv2.warpPerspective(
            tpl, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        ).astype(np.float32)

        try:
            mod.set_force_scalar(False)
            cc_a, W_a = find_transform_ecc_homography(tpl, src, max_iter=50, eps=1e-4, num_threads=2)
            mod.set_force_scalar(True)
            cc_s, W_s = find_transform_ecc_homography(tpl, src, max_iter=50, eps=1e-4, num_threads=2)
        finally:
            mod.set_force_scalar(False)

        assert cc_a is not None and cc_s is not None
        assert W_a is not None and W_s is not None
        assert abs(cc_a - cc_s) < 1e-8
        corners = np.array([[0.0, 0.0, 1.0], [w - 1.0, 0.0, 1.0], [w - 1.0, h - 1.0, 1.0], [0.0, h - 1.0, 1.0]])
        pa = (corners @ W_a.astype(np.float64).T)
        pa = pa[:, :2] / pa[:, 2:3]
        ps = (corners @ W_s.astype(np.float64).T)
        ps = ps[:, :2] / ps[:, 2:3]
        max_corner_diff = float(np.sqrt(np.sum((pa - ps) ** 2, axis=1)).max())
        assert max_corner_diff < 1e-5, f"shape={(h, w)} corner diff={max_corner_diff}"


def test_ecc_preserves_scalar_variance_cutoff_decision() -> None:
    mod = _require_avx2()
    tpl = np.repeat((np.arange(16, dtype=np.float32) / 16)[:, None], 1024, axis=1)
    src = np.full_like(tpl, 0.75)
    src[1, 513:529] -= 3 * 2.0**-24
    src[14, 513:529] += 3 * 2.0**-24
    src[0, :] -= 2.0**-8
    src[-1, :] += 2.0**-8
    src[0, 1000:] = 1.0
    src[-1, 1000:] = 0.5
    _, _, _, rho_identity = mod.ecc_check_identical_and_stats(tpl, src, 1)
    seed = np.eye(3, dtype=np.float64)

    try:
        mod.set_force_scalar(True)
        scalar_ok, scalar_cc, scalar_warp = mod.ecc_refine_pyramids(
            [tpl], [src], seed, rho_identity, 1, 1e-4, 1)
        mod.set_force_scalar(False)
        simd_ok, simd_cc, simd_warp = mod.ecc_refine_pyramids(
            [tpl], [src], seed, rho_identity, 1, 1e-4, 1)
    finally:
        mod.set_force_scalar(False)

    assert scalar_ok and simd_ok
    assert simd_cc == pytest.approx(scalar_cc, abs=1e-8)
    np.testing.assert_allclose(simd_warp, scalar_warp, rtol=0, atol=1e-7)


def _scalar_and_simd(mod, run):
    try:
        mod.set_force_scalar(False)
        simd = run()
        mod.set_force_scalar(True)
        scalar = run()
    finally:
        mod.set_force_scalar(False)
    return simd, scalar


def _assert_same_bytes(simd, scalar):
    for a, b in zip(simd, scalar):
        assert np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_alignment_rgb_warps_avx2_vs_scalar_bit_exact(dtype):
    """Edge columns, rotations and reversed channels around the paired-sample path."""
    import math

    from core import _alignment_kernels as ak

    mod = _require_avx2()
    rng = np.random.default_rng(960)
    h, w = 37, 53
    image = rng.integers(0, np.iinfo(dtype).max + 1, (h, w, 3), dtype=dtype)
    gray = np.ascontiguousarray(image[:, :, 1])

    def similarity(shift_x, shift_y, degrees, scale):
        cx, cy = w * 0.5, h * 0.5
        cos = scale * math.cos(math.radians(degrees))
        sin = scale * math.sin(math.radians(degrees))
        return (cx + shift_x, cy + shift_y, cx, cy, sin, sin, cos, cos)

    def run():
        results = []
        for spec in ((0.0, 0.0, 0.0, 1.0), (3.3, -2.7, 0.03, 1.002), (-9.5, 4.25, -1.9, 0.97),
                     (0.5, 0.5, 2.0, 1.03), (60.0, 0.0, 0.0, 1.0), (1.0, -1.0, 45.0, 1.3)):
            transform = similarity(*spec)
            for kernel, source in ((ak.warp_alignment_int_rows, image),
                                   (ak.warp_alignment_int_rows_bgr_rev, image),
                                   (ak.warp_alignment_int_rows, gray)):
                out = np.full(source.shape, 7, dtype=dtype)
                kernel(source, out, *transform, 0, h)
                results.append(out)
            region = np.full((h - 5, w - 7, 3), 7, dtype=dtype)
            ak.warp_alignment_region_int_rows(image, region, h, w, 0, 3, 2, *transform, 0, h - 5)
            results.append(region)
        for dx, dy in ((0.0, 0.0), (0.5, 0.5), (-3.25, 7.75), (-0.999, 0.001), (w + 3.0, 0.0)):
            for kernel, source in ((ak.warp_int_rows, image), (ak.warp_int_rows_bgr_rev, image),
                                   (ak.warp_int_rows, gray)):
                out = np.full(source.shape, 7, dtype=dtype)
                kernel(source, out, dx, dy, 0, h)
                results.append(out)
        return results

    simd, scalar = _scalar_and_simd(mod, run)
    for a, b in zip(simd, scalar):
        np.testing.assert_array_equal(a, b)


def _check_alignment_region_rgb_short_rows():
    import ctypes
    import mmap
    import os

    mod = _require_avx2()
    page_size = mmap.PAGESIZE
    shape = (2, 2, 3)
    with mmap.mmap(-1, 2 * page_size, access=mmap.ACCESS_WRITE) as storage:
        guard = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(storage)) + page_size)
        if os.name == "nt":
            protect = ctypes.WinDLL("kernel32", use_last_error=True).VirtualProtect
            old_protection = ctypes.c_ulong()
            if not protect(guard, ctypes.c_size_t(page_size), 0x01, ctypes.byref(old_protection)):  # PAGE_NOACCESS
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            protect = ctypes.CDLL(None, use_errno=True).mprotect
            if protect(guard, ctypes.c_size_t(page_size), 0) != 0:  # PROT_NONE
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))

        for dtype in (np.uint16, np.uint8):
            # Both taps fit in the final row, whose end abuts the guard page.
            pixels = (np.arange(12, dtype=dtype) * 16).reshape(shape)
            image = np.ndarray(shape, dtype=dtype, buffer=storage, offset=page_size - pixels.nbytes)
            image[:] = pixels
            expected = 30 + 3 * np.arange(8)[None, :, None] + 16 * np.arange(3)[None, None, :]

            def run():
                results = []
                for bgr_rev in (False, True):
                    output = np.empty((1, 8, 3), dtype=dtype)
                    mod.of_align_warp_alignment_region_int_rows(
                        image, output, 2, 8, 0, 0, 0,
                        0.125, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0625, 0.0625, 0, 1, bgr_rev,
                    )
                    results.append(output)
                return results

            simd, scalar = _scalar_and_simd(mod, run)
            np.testing.assert_array_equal(scalar[0], expected)
            np.testing.assert_array_equal(scalar[1], expected[..., ::-1])
            _assert_same_bytes(simd, scalar)


def test_alignment_region_rgb_does_not_overread_short_rows():
    import subprocess
    import sys
    from pathlib import Path

    _require_avx2()
    test_file = Path(__file__).resolve()
    # Isolate a fault so pytest can report it without ending the suite.
    code = (
        "import runpy, sys\n"
        "runpy.run_path(sys.argv[1])['_check_alignment_region_rgb_short_rows']()"
    )
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code, str(test_file)],
        cwd=test_file.parents[1], capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, f"Guarded RGB warp exited with {result.returncode}\n{result.stderr}"


@pytest.mark.parametrize("grid", [(3, 2), (8, 5), (12, 9), (17, 3)])
def test_alignment_score_flat_grid_avx2_vs_scalar_bit_exact(grid):
    """Sample grids are scored as one padded list, including rejected samples."""
    from core import _alignment_kernels as ak

    mod = _require_avx2()
    rng = np.random.default_rng(970 + grid[0])
    h, w = 97, 131
    reference = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    moved = np.roll(reference, (1, -2), (0, 1))
    xs = np.linspace(0, w - 1, grid[0]).round().astype(np.float32)
    ys = np.linspace(0, h - 1, grid[1]).round().astype(np.float32)
    samples = np.ascontiguousarray(reference[ys.astype(np.int64)[:, None], xs.astype(np.int64)[None, :]])
    count = 19
    p08 = rng.normal(0, 0.02, count).astype(np.float32)
    p0c = rng.normal(0, 0.02, count).astype(np.float32)
    p18 = (1 + rng.normal(0, 0.02, count)).astype(np.float32)
    p1c = (1 + rng.normal(0, 0.02, count)).astype(np.float32)
    tx = (w / 2 + rng.normal(0, 6, count)).astype(np.float32)
    ty = (h / 2 + rng.normal(0, 6, count)).astype(np.float32)
    tx[0] = -5000.0
    cx = np.full(count, w / 2, dtype=np.float32)
    cy = np.full(count, h / 2, dtype=np.float32)
    order = np.arange(count, dtype=np.int64)

    def run():
        ssum = np.zeros(count, dtype=np.int64)
        ssq = np.zeros(count, dtype=np.int64)
        cnt = np.zeros(count, dtype=np.int64)
        ak.score_chunk_u16_slice(samples, moved, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy,
                                 order, 0, count, ssum, ssq, cnt)
        return ssum, ssq, cnt

    simd, scalar = _scalar_and_simd(mod, run)
    for a, b in zip(simd, scalar):
        np.testing.assert_array_equal(a, b)
    assert simd[2][0] == 0
    assert simd[2].max() > 0


@pytest.mark.parametrize("crop", [9, 33, 64, 200])
def test_alignment_refine_level_avx2_vs_scalar_bit_exact(crop):
    """Residual sums are exact integers, so the vector kernel must not move the minimum."""
    from core import _alignment_kernels as ak

    mod = _require_avx2()
    rng = np.random.default_rng(980 + crop)
    h, w = 97, 131
    reference = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    moved = np.roll(reference, (1, -2), (0, 1))

    def run():
        results = []
        for c_dx, c_dy, half, step in ((0.0, 0.0, 3.0, 1.0), (2.0, -1.0, 2.0, 0.5),
                                       (30.0, -25.0, 2.0, 0.5), (0.3, 0.7, 1.0, 0.125)):
            for a, b in ((reference, moved), (moved, reference), (reference, np.zeros_like(moved))):
                results.append(ak.refine_level(a, b, c_dx, c_dy, half, step, crop))
        return results

    simd, scalar = _scalar_and_simd(mod, run)
    assert simd == scalar
    # The exact shift lies inside the first search window in both directions.
    assert simd[0] == (2.0, -1.0, True)
    assert simd[1] == (-2.0, 1.0, True)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_alignment_rgb_warps_keep_overlapping_output_exact(dtype):
    """An output that shares memory with its source only has the scalar order."""
    import math

    from core import _alignment_kernels as ak

    mod = _require_avx2()
    rng = np.random.default_rng(961)
    h, w = 21, 37
    backing = rng.integers(0, np.iinfo(dtype).max + 1, (h + 1, w + 1, 3), dtype=dtype)

    def similarity(shift_x, shift_y, degrees, scale):
        cx, cy = w * 0.5, h * 0.5
        cos = scale * math.cos(math.radians(degrees))
        sin = scale * math.sin(math.radians(degrees))
        return (cx + shift_x, cy + shift_y, cx, cy, sin, sin, cos, cos)

    def run():
        results = []
        for dx, dy in ((0.0, 0.0), (0.5, 0.5), (-3.25, 2.75)):
            for kernel in (ak.warp_int_rows, ak.warp_int_rows_bgr_rev):
                same = backing.copy()
                kernel(same[:h, :w], same[:h, :w], dx, dy, 0, h)
                right = backing.copy()
                kernel(right[:h, :w], right[:h, 1:], dx, dy, 0, h)
                below = backing.copy()
                kernel(below[:h, :w], below[1:, :w], dx, dy, 0, h)
                results += [same, right, below]
        for spec in ((0.0, 0.0, 0.0, 1.0), (1.5, -0.75, 1.5, 1.01)):
            transform = similarity(*spec)
            for kernel in (ak.warp_alignment_int_rows, ak.warp_alignment_int_rows_bgr_rev):
                same = backing.copy()
                kernel(same[:h, :w], same[:h, :w], *transform, 0, h)
                right = backing.copy()
                kernel(right[:h, :w], right[:h, 1:], *transform, 0, h)
                results += [same, right]
            # a region written over the pixels it samples
            region = backing.copy()
            ak.warp_alignment_region_int_rows(
                region[:h, :w], region[2:h - 3, 3:w - 4], h, w, 0, 3, 2, *transform, 0, h - 5)
            results.append(region)
        return results

    _assert_same_bytes(*_scalar_and_simd(mod, run))


def test_alignment_score_keeps_sums_stored_over_inputs_exact():
    """A sum stored over the grid or its coordinates changes what later candidates read."""
    mod = _require_avx2()
    rng = np.random.default_rng(971)
    h, w = 97, 131
    nx, ny, count = 12, 9, 40
    reference = rng.integers(0, 65536, (h, w), dtype=np.uint16)
    moved = np.roll(reference, (1, -2), (0, 1))
    xs0 = np.linspace(0, w - 1, nx).round().astype(np.float32)
    ys0 = np.linspace(0, h - 1, ny).round().astype(np.float32)
    samples0 = np.ascontiguousarray(reference[ys0.astype(np.int64)[:, None], xs0.astype(np.int64)[None, :]])
    p08 = rng.normal(0, 0.02, count).astype(np.float32)
    p0c = rng.normal(0, 0.02, count).astype(np.float32)
    p18 = (1 + rng.normal(0, 0.02, count)).astype(np.float32)
    p1c = (1 + rng.normal(0, 0.02, count)).astype(np.float32)
    tx = (w / 2 + rng.normal(0, 6, count)).astype(np.float32)
    ty = (h / 2 + rng.normal(0, 6, count)).astype(np.float32)
    cx = np.full(count, w / 2, dtype=np.float32)
    cy = np.full(count, h / 2, dtype=np.float32)
    order = np.arange(count, dtype=np.int64)

    def run():
        results = []
        for place in ("xs", "samples"):
            ssum = np.zeros(count, dtype=np.int64)
            ssq = np.zeros(count, dtype=np.int64)
            cnt = np.zeros(count, dtype=np.int64)
            xs, samples = xs0, samples0
            if place == "xs":
                xs = ssum.view(np.float32)[:nx]
                xs[...] = xs0
            else:
                samples = cnt.view(np.uint16)[: nx * ny].reshape(ny, nx)
                samples[...] = samples0
            mod.of_align_score_chunk_u16_slice(samples, moved, xs, ys0, p08, p0c, p18, p1c, tx, ty, cx, cy,
                                               order, 0, count, ssum, ssq, cnt)
            results += [ssum, ssq, cnt]
        return results

    _assert_same_bytes(*_scalar_and_simd(mod, run))
