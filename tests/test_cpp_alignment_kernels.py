import os
import subprocess
import sys

import numpy as np
import pytest

import core._alignment_kernels as ak
import core._native as native_mod
from core.openfocus_alignment import (
    apply_alignment,
    apply_alignment_region,
    apply_translation,
    estimate_alignment_plan,
    estimate_translation_plan,
)
from core import resource_policy


def _disable_native(monkeypatch):
    monkeypatch.setattr(native_mod, "load_native", lambda: None)


def _mock_resources(monkeypatch):
    monkeypatch.setattr(
        resource_policy,
        "probe_resources",
        lambda *a, **kw: resource_policy.ResourceSnapshot(
            available_ram=16 * 1024**3,
            total_ram=32 * 1024**3,
            physical_cores=4,
            logical_cpus=8,
            physical_cores_known=True,
            commit_headroom=16 * 1024**3,
        ),
    )


def test_swap_rgb_and_inplace_parity(monkeypatch):
    rng = np.random.default_rng(1)
    src_u8 = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    src_u16 = rng.integers(0, 65536, (37, 53, 3), dtype=np.uint16)

    out_cpp_u8 = np.empty_like(src_u8)
    out_nb_u8 = np.empty_like(src_u8)
    out_cpp_u16 = np.empty_like(src_u16)
    out_nb_u16 = np.empty_like(src_u16)

    ak.swap_rgb_to_bgr_rows_u8(src_u8, out_cpp_u8, 0, 37)
    ak.swap_rgb_to_bgr_rows_u16(src_u16, out_cpp_u16, 0, 37)

    inp_cpp_u8 = src_u8.copy()
    inp_cpp_u16 = src_u16.copy()
    ak.swap_bgr_inplace_rows_u8(inp_cpp_u8, 3, 29)
    ak.swap_bgr_inplace_rows_u16(inp_cpp_u16, 3, 29)

    _disable_native(monkeypatch)
    ak.swap_rgb_to_bgr_rows_u8(src_u8, out_nb_u8, 0, 37)
    ak.swap_rgb_to_bgr_rows_u16(src_u16, out_nb_u16, 0, 37)

    inp_nb_u8 = src_u8.copy()
    inp_nb_u16 = src_u16.copy()
    ak.swap_bgr_inplace_rows_u8(inp_nb_u8, 3, 29)
    ak.swap_bgr_inplace_rows_u16(inp_nb_u16, 3, 29)

    assert np.array_equal(out_cpp_u8, out_nb_u8)
    assert np.array_equal(out_cpp_u16, out_nb_u16)
    assert np.array_equal(inp_cpp_u8, inp_nb_u8)
    assert np.array_equal(inp_cpp_u16, inp_nb_u16)


@pytest.mark.parametrize("shape", [(41, 57), (41, 57, 3), (41, 57, 4), (41, 57, 2)])
def test_gray16_and_downsize_parity(monkeypatch, shape):
    rng = np.random.default_rng(2)
    img_u8 = rng.integers(0, 256, shape, dtype=np.uint8)
    img_u16 = rng.integers(0, 65536, shape, dtype=np.uint16)

    g8_cpp = ak.gray16_u8(img_u8, np.empty(shape[:2], dtype=np.uint16))
    g16_cpp = ak.gray16_u16(img_u16, np.empty(shape[:2], dtype=np.uint16))
    if len(shape) == 3 and shape[2] >= 3:
        g8_rev_cpp = ak.gray16_u8(img_u8[..., ::-1], np.empty(shape[:2], dtype=np.uint16))
    else:
        g8_rev_cpp = None

    half_cpp = ak.down_half(g16_cpp)
    oh, ow = (shape[0] + 1) // 2, (shape[1] + 1) // 2
    fast_cpp = np.empty((oh, ow), dtype=np.uint16)
    ak.fast_downsize_u16_rows(g16_cpp, fast_cpp, 0, oh)

    _disable_native(monkeypatch)
    g8_nb = ak.gray16_u8(img_u8, np.empty(shape[:2], dtype=np.uint16))
    g16_nb = ak.gray16_u16(img_u16, np.empty(shape[:2], dtype=np.uint16))
    g8_rev_nb = ak.gray16_u8(img_u8[..., ::-1], np.empty(shape[:2], dtype=np.uint16)) if g8_rev_cpp is not None else None
    half_nb = ak.down_half(g16_cpp)
    fast_nb = np.empty((oh, ow), dtype=np.uint16)
    ak.fast_downsize_u16_rows(g16_cpp, fast_nb, 0, oh)

    assert np.array_equal(g8_cpp, g8_nb)
    assert np.array_equal(g16_cpp, g16_nb)
    if g8_rev_cpp is not None:
        assert np.array_equal(g8_rev_cpp, g8_rev_nb)
    assert np.array_equal(half_cpp, half_nb)
    assert np.array_equal(fast_cpp, fast_nb)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("shape", [(45, 61), (45, 61, 3), (45, 61, 4)])
def test_warp_translation_and_similarity_parity(monkeypatch, dtype, shape):
    rng = np.random.default_rng(3)
    hi = 256 if dtype == np.uint8 else 65536
    img = rng.integers(0, hi, shape, dtype=dtype)
    dx, dy = np.float32(2.35), np.float32(-1.75)
    tx, ty = np.float32(1.4), np.float32(-0.9)
    cx, cy = np.float32(30.0), np.float32(22.0)
    p08, p0c = np.float32(0.015), np.float32(0.014)
    p18, p1c = np.float32(1.012), np.float32(1.011)

    t_cpp = np.empty_like(img)
    s_cpp = np.empty_like(img)
    ak.warp_int_rows(img, t_cpp, dx, dy, 0, shape[0])
    ak.warp_alignment_int_rows(img, s_cpp, tx, ty, cx, cy, p08, p0c, p18, p1c, 0, shape[0])
    w_u16_cpp = ak.warp_u16(img, dx, dy) if dtype == np.uint16 else None

    rev_t_cpp = rev_s_cpp = None
    if len(shape) == 3 and shape[2] == 3:
        rev_t_cpp = np.empty_like(img)
        rev_s_cpp = np.empty_like(img)
        ak.warp_int_rows_bgr_rev(img[..., ::-1], rev_t_cpp, dx, dy, 0, shape[0])
        ak.warp_alignment_int_rows_bgr_rev(img[..., ::-1], rev_s_cpp, tx, ty, cx, cy, p08, p0c, p18, p1c, 0, shape[0])

    mask_cpp = np.empty(shape[:2], dtype=np.bool_)
    ak.valid_mask_rows(mask_cpp, shape[0], shape[1], tx, ty, cx, cy, p08, p0c, p18, p1c, 0, shape[0])

    _disable_native(monkeypatch)
    t_nb = np.empty_like(img)
    s_nb = np.empty_like(img)
    ak.warp_int_rows(img, t_nb, dx, dy, 0, shape[0])
    ak.warp_alignment_int_rows(img, s_nb, tx, ty, cx, cy, p08, p0c, p18, p1c, 0, shape[0])
    w_u16_nb = ak.warp_u16(img, dx, dy) if dtype == np.uint16 else None

    if rev_t_cpp is not None:
        rev_t_nb = np.empty_like(img)
        rev_s_nb = np.empty_like(img)
        ak.warp_int_rows_bgr_rev(img[..., ::-1], rev_t_nb, dx, dy, 0, shape[0])
        ak.warp_alignment_int_rows_bgr_rev(img[..., ::-1], rev_s_nb, tx, ty, cx, cy, p08, p0c, p18, p1c, 0, shape[0])

    mask_nb = np.empty(shape[:2], dtype=np.bool_)
    ak.valid_mask_rows(mask_nb, shape[0], shape[1], tx, ty, cx, cy, p08, p0c, p18, p1c, 0, shape[0])

    assert np.array_equal(t_cpp, t_nb)
    assert np.array_equal(s_cpp, s_nb)
    assert np.array_equal(mask_cpp, mask_nb)
    if w_u16_cpp is not None:
        assert np.array_equal(w_u16_cpp, w_u16_nb)
    if rev_t_cpp is not None:
        assert np.array_equal(rev_t_cpp, rev_t_nb)
        assert np.array_equal(rev_s_cpp, rev_s_nb)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(33, 47), (33, 47, 3)])
def test_warp_alignment_float_and_region_parity(monkeypatch, dtype, shape):
    rng = np.random.default_rng(4)
    img = rng.random(shape, dtype=dtype)
    tx, ty = np.float32(22.4), np.float32(18.9)
    cx, cy = np.float32(23.5), np.float32(16.5)
    p08, p0c = np.float32(-0.012), np.float32(-0.011)
    p18, p1c = np.float32(0.985), np.float32(0.986)

    full_cpp = np.empty_like(img)
    ak.warp_alignment_float_rows(img, full_cpp, tx, ty, cx, cy, p08, p0c, p18, p1c, 0, shape[0])

    src_y0 = 6
    band = img[src_y0:]
    out_shape = (15, 21) if len(shape) == 2 else (15, 21, shape[2])
    reg_cpp = np.empty(out_shape, dtype=dtype)
    ak.warp_alignment_region_float_rows(
        band, reg_cpp, shape[0], shape[1], src_y0, 8, 9,
        tx, ty, cx, cy, p08, p0c, p18, p1c, 0, 15,
    )

    u16 = (img * 65535.0).astype(np.uint16)
    band_u16 = u16[src_y0:]
    reg_u16_s_cpp = np.empty(out_shape, dtype=np.uint16)
    reg_u16_t_cpp = np.empty(out_shape, dtype=np.uint16)
    ak.warp_alignment_region_int_rows(
        band_u16, reg_u16_s_cpp, shape[0], shape[1], src_y0, 8, 9,
        tx, ty, cx, cy, p08, p0c, p18, p1c, 0, 15,
    )
    ak.warp_translation_region_int_rows(
        band_u16, reg_u16_t_cpp, shape[0], shape[1], src_y0, 8, 9,
        np.float32(1.5), np.float32(-0.75), 0, 15,
    )
    reg_s_rev_cpp = reg_t_rev_cpp = None
    if len(shape) == 3:
        reg_s_rev_cpp = np.empty(out_shape, dtype=np.uint16)
        reg_t_rev_cpp = np.empty(out_shape, dtype=np.uint16)
        ak.warp_alignment_region_int_rows_bgr_rev(
            np.ascontiguousarray(band_u16[..., ::-1]), reg_s_rev_cpp,
            shape[0], shape[1], src_y0, 8, 9,
            tx, ty, cx, cy, p08, p0c, p18, p1c, 0, 15,
        )
        ak.warp_translation_region_int_rows_bgr_rev(
            np.ascontiguousarray(band_u16[..., ::-1]), reg_t_rev_cpp,
            shape[0], shape[1], src_y0, 8, 9,
            np.float32(1.5), np.float32(-0.75), 0, 15,
        )

    _disable_native(monkeypatch)
    full_nb = np.empty_like(img)
    ak.warp_alignment_float_rows(img, full_nb, tx, ty, cx, cy, p08, p0c, p18, p1c, 0, shape[0])
    reg_nb = np.empty(out_shape, dtype=dtype)
    ak.warp_alignment_region_float_rows(
        band, reg_nb, shape[0], shape[1], src_y0, 8, 9,
        tx, ty, cx, cy, p08, p0c, p18, p1c, 0, 15,
    )
    reg_u16_s_nb = np.empty(out_shape, dtype=np.uint16)
    reg_u16_t_nb = np.empty(out_shape, dtype=np.uint16)
    ak.warp_alignment_region_int_rows(
        band_u16, reg_u16_s_nb, shape[0], shape[1], src_y0, 8, 9,
        tx, ty, cx, cy, p08, p0c, p18, p1c, 0, 15,
    )
    ak.warp_translation_region_int_rows(
        band_u16, reg_u16_t_nb, shape[0], shape[1], src_y0, 8, 9,
        np.float32(1.5), np.float32(-0.75), 0, 15,
    )
    if reg_s_rev_cpp is not None:
        reg_s_rev_nb = np.empty(out_shape, dtype=np.uint16)
        reg_t_rev_nb = np.empty(out_shape, dtype=np.uint16)
        ak.warp_alignment_region_int_rows_bgr_rev(
            np.ascontiguousarray(band_u16[..., ::-1]), reg_s_rev_nb,
            shape[0], shape[1], src_y0, 8, 9,
            tx, ty, cx, cy, p08, p0c, p18, p1c, 0, 15,
        )
        ak.warp_translation_region_int_rows_bgr_rev(
            np.ascontiguousarray(band_u16[..., ::-1]), reg_t_rev_nb,
            shape[0], shape[1], src_y0, 8, 9,
            np.float32(1.5), np.float32(-0.75), 0, 15,
        )

    assert np.array_equal(full_cpp, full_nb)
    assert np.array_equal(reg_cpp, reg_nb)
    assert np.array_equal(reg_u16_s_cpp, reg_u16_s_nb)
    assert np.array_equal(reg_u16_t_cpp, reg_u16_t_nb)
    if reg_s_rev_cpp is not None:
        assert np.array_equal(reg_s_rev_cpp, reg_s_rev_nb)
        assert np.array_equal(reg_t_rev_cpp, reg_t_rev_nb)


def test_scoring_refine_and_perspective_parity(monkeypatch):
    rng = np.random.default_rng(5)
    a = rng.integers(0, 65536, (96, 112), dtype=np.uint16)
    b = rng.integers(0, 65536, (96, 112), dtype=np.uint16)
    # Test strided view as well as contiguous
    a_s = rng.integers(0, 65536, (96, 128), dtype=np.uint16)[:, :112]
    b_s = rng.integers(0, 65536, (96, 128), dtype=np.uint16)[:, :112]

    ref_cpp = ak.refine_level(a, b, np.float32(1.0), np.float32(-0.5), np.float32(3.0), np.float32(1.0), 48)
    ref_strided_cpp = ak.refine_level(a_s, b_s, np.float32(1.0), np.float32(-0.5), np.float32(2.0), np.float32(0.5), 40)

    xs = np.linspace(10.0, 90.0, 20, dtype=np.float32)
    ys = np.linspace(12.0, 80.0, 18, dtype=np.float32)
    ref_patch = a[12:30, 10:30].copy()
    k = 12
    p08 = rng.uniform(-0.02, 0.02, k).astype(np.float32)
    p0c = p08.copy()
    p18 = rng.uniform(0.98, 1.02, k).astype(np.float32)
    p1c = p18.copy()
    tx = rng.uniform(-3.0, 3.0, k).astype(np.float32)
    ty = rng.uniform(-3.0, 3.0, k).astype(np.float32)
    cx = rng.uniform(50.0, 60.0, k).astype(np.float32)
    cy = rng.uniform(44.0, 52.0, k).astype(np.float32)
    idx = np.arange(k, dtype=np.int64)
    ssum_c, ssq_c, cnt_c = np.zeros(k, np.int64), np.zeros(k, np.int64), np.zeros(k, np.int64)
    ak.score_chunk_u16(ref_patch, b_s, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy, idx, ssum_c, ssq_c, cnt_c)

    cx_tab = np.array([55.0, 56.0, 57.0], dtype=np.float32)
    cy_tab = np.array([47.0, 48.0, 49.0], dtype=np.float32)
    p08_tab = np.array([[-0.01, 0.0, 0.01], [0.0, 0.01, -0.01]], dtype=np.float32)
    p18_tab = np.array([[1.0, 1.01, 0.99], [1.01, 1.0, 1.02]], dtype=np.float32)
    out_k_c = np.empty(2, dtype=np.int64)
    out_sc_c = np.empty(2, dtype=np.float32)
    n0, n1, n2, n3 = 3, 3, 3, 2
    ak.score_grid_4d_slice(ref_patch, b_s, xs, ys, np.float32(55.0), np.float32(47.0),
                           cx_tab, cy_tab, p08_tab, p18_tab, n0, n1, n2, n3, 0, n0 * n1 * n2 * n3,
                           out_k_c, out_sc_c, 0)

    inv = np.array([[1.0, 0.002, 0.5], [-0.001, 1.0, -0.25], [1e-5, -2e-5, 1.0]], dtype=np.float64)
    edge_c = ak.perspective_has_rounding_edge(inv, 0, 16, 0, 16, 65536.0, 1e-4)

    _disable_native(monkeypatch)
    ref_nb = ak.refine_level(a, b, np.float32(1.0), np.float32(-0.5), np.float32(3.0), np.float32(1.0), 48)
    ref_strided_nb = ak.refine_level(a_s, b_s, np.float32(1.0), np.float32(-0.5), np.float32(2.0), np.float32(0.5), 40)
    ssum_n, ssq_n, cnt_n = np.zeros(k, np.int64), np.zeros(k, np.int64), np.zeros(k, np.int64)
    ak.score_chunk_u16(ref_patch, b_s, xs, ys, p08, p0c, p18, p1c, tx, ty, cx, cy, idx, ssum_n, ssq_n, cnt_n)
    out_k_n = np.empty(2, dtype=np.int64)
    out_sc_n = np.empty(2, dtype=np.float32)
    ak.score_grid_4d_slice(ref_patch, b_s, xs, ys, np.float32(55.0), np.float32(47.0),
                           cx_tab, cy_tab, p08_tab, p18_tab, n0, n1, n2, n3, 0, n0 * n1 * n2 * n3,
                           out_k_n, out_sc_n, 0)
    edge_n = ak.perspective_has_rounding_edge(inv, 0, 16, 0, 16, 65536.0, 1e-4)

    assert ref_cpp == ref_nb
    assert ref_strided_cpp == ref_strided_nb
    assert np.array_equal(ssum_c, ssum_n)
    assert np.array_equal(ssq_c, ssq_n)
    assert np.array_equal(cnt_c, cnt_n)
    assert out_k_c[0] == out_k_n[0]
    assert out_sc_c[0] == out_sc_n[0]
    assert edge_c == edge_n


def test_grid_scoring_rounds_wide_coordinates_before_mapping(monkeypatch):
    _disable_native(monkeypatch)
    origin = 1 << 24
    xs = np.array([origin + 1, origin + 3, origin + 5], dtype=np.int64)
    ys = np.array([origin + 1], dtype=np.int64)
    cx = np.array([origin], dtype=np.float32)
    cy = np.array([origin - 4, origin], dtype=np.float32)
    image = np.array([[100, 300, 200, 700, 500, 900]], dtype=np.uint16)
    # The f32 coordinates map to columns 0, 4, 4 on the single row.
    ref = image[:, [0, 4, 4]]
    out_k = np.empty(1, dtype=np.int64)
    out_sc = np.empty(1, dtype=np.float32)
    ak.score_grid_4d_slice(
        ref, image, xs, ys, np.float32(0), np.float32(0), cx, cy,
        np.zeros((1, 1), np.float32), np.ones((1, 1), np.float32),
        1, 2, 1, 1, 0, 2, out_k, out_sc, 0,
    )
    assert out_k[0] == 1
    assert out_sc[0] == 0.0


def test_end_to_end_alignment_and_memmap_parity(monkeypatch, tmp_path):
    _mock_resources(monkeypatch)
    rng = np.random.default_rng(7)
    base = np.zeros((160, 192, 3), dtype=np.uint16)
    yy, xx = np.mgrid[0:160, 0:192]
    pat = (
        25000.0
        + 15000.0 * np.sin(xx * 0.17) * np.cos(yy * 0.13)
        + 8000.0 * np.sin((xx + yy) * 0.09)
    )
    for c in range(3):
        base[..., c] = np.clip(pat + c * 1500.0 + rng.normal(0, 120, (160, 192)), 0, 65535).astype(np.uint16)

    mm_path = tmp_path / "img.mmap"
    mm = np.memmap(mm_path, dtype=np.uint16, mode="w+", shape=base.shape)
    mm[:] = np.roll(base, shift=(2, -3), axis=(0, 1))
    mm.flush()

    mm_ro = np.memmap(mm_path, dtype=np.uint16, mode="r", shape=base.shape)

    plan_cpp = estimate_alignment_plan([base, mm_ro], rotation=2.0, magnification=3.0)
    tplan_cpp = estimate_translation_plan([base, mm_ro])
    warped_cpp = apply_alignment(mm_ro, plan_cpp.transforms[1])
    twarped_cpp = apply_translation(mm_ro, tplan_cpp.transforms[1])
    reg_cpp = apply_alignment_region(mm_ro, plan_cpp.transforms[1], 20, 24, 80, 64)

    _disable_native(monkeypatch)
    plan_nb = estimate_alignment_plan([base, mm_ro], rotation=2.0, magnification=3.0)
    tplan_nb = estimate_translation_plan([base, mm_ro])
    warped_nb = apply_alignment(mm_ro, plan_nb.transforms[1])
    twarped_nb = apply_translation(mm_ro, tplan_nb.transforms[1])
    reg_nb = apply_alignment_region(mm_ro, plan_nb.transforms[1], 20, 24, 80, 64)

    assert len(plan_cpp.transforms) == len(plan_nb.transforms) == 2
    assert plan_cpp.transforms[1].f8 == plan_nb.transforms[1].f8
    assert (tplan_cpp.transforms[1].dx, tplan_cpp.transforms[1].dy) == (
        tplan_nb.transforms[1].dx, tplan_nb.transforms[1].dy
    )
    assert np.array_equal(warped_cpp, warped_nb)
    assert np.array_equal(twarped_cpp, twarped_nb)
    assert np.array_equal(reg_cpp, reg_nb)
    del mm, mm_ro


def test_alignment_triggers_zero_numba_compilations():
    code = """
import numpy as np
from core import _native, _alignment_kernels as ak
from core.openfocus_alignment import (
    estimate_alignment_plan, apply_alignment, estimate_translation_plan, apply_translation
)
assert _native.load_native() is not None
assert hasattr(_native.load_native(), "of_align_gray16_rows")

try:
    from numba.core.dispatcher import Dispatcher
    orig = Dispatcher.compile
    compiles = []
    def spy(self, sig):
        compiles.append(self.py_func.__name__)
        return orig(self, sig)
    Dispatcher.compile = spy
except Exception:
    compiles = []

yy, xx = np.mgrid[0:128, 0:160]
a = np.clip(30000 + 18000 * np.sin(xx * 0.19) * np.cos(yy * 0.14), 0, 65535).astype(np.uint16)
b = np.roll(a, shift=(2, -2), axis=(0, 1))
plan = estimate_alignment_plan([a, b], rotation=2.0, magnification=3.0)
tplan = estimate_translation_plan([a, b])
_ = apply_alignment(b, plan.transforms[1])
_ = apply_translation(b, tplan.transforms[1])
assert len(compiles) == 0, f"Unexpected Numba compilations: {compiles}"
"""
    worktree = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    res = subprocess.run(
        [sys.executable, "-c", code],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
