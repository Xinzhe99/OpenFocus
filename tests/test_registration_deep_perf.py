"""Tests for deep-stack ECC, PerspectiveWarp, and OpenFocus alignment optimizations."""

import hashlib
from threading import get_ident

import cv2
import numpy as np
import pytest
import tifffile

from core import openfocus_alignment as oa
from core import _alignment_kernels as kernels
from core._ecc_area_gpu import ExactAreaGpu, _is_bgr_rev as gpu_is_bgr_rev
from core.image_loader import ImageStackLoader
from core.registration import (
    ImageRegistration,
    PerspectiveWarp,
    RegistrationCancelled,
    _perspective_source,
    deferred_warp_threads,
)
from core.source_frames import is_direct_rgb_tiff, map_native_rgb_tiffs, resolve_frame


def _make_textured_rgb(height, width, dtype=np.uint16, seed=101):
    rng = np.random.default_rng(seed)
    low_h = max(16, height // 16)
    low_w = max(16, width // 16)
    max_val = np.iinfo(dtype).max
    low = rng.integers(max_val // 8, (max_val * 7) // 8, (low_h, low_w, 3), dtype=dtype)
    base = cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    wave = (np.sin(xx * 0.041 + yy * 0.023) + np.cos(xx * 0.017 - yy * 0.037)) * (max_val * 0.06)
    out = np.clip(base.astype(np.int32) + wave[..., None].astype(np.int32), 0, max_val).astype(dtype)
    return np.ascontiguousarray(out)


def test_is_bgr_rev_predicates_and_area_gpu_rejects_arbitrary_strides():
    rgb = np.zeros((16, 20, 3), dtype=np.uint16)
    bgr_rev = rgb[..., ::-1]
    assert oa._is_bgr_rev(bgr_rev)
    assert gpu_is_bgr_rev(bgr_rev)

    for bad in (rgb, rgb[:, ::2, :], rgb[::2, :, :], rgb[:, ::-1, :], rgb[..., ::2]):
        assert not oa._is_bgr_rev(bad)
        assert not gpu_is_bgr_rev(bad)

    gpu = ExactAreaGpu()
    assert gpu.resize(rgb[:, ::2, :], (8, 8)) is None
    assert gpu._cl is None


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_ecc_area_gpu_bgr_rev_and_memmap_match_opencv(tmp_path, monkeypatch, dtype):
    from core import _ecc_area_gpu as area_mod

    probe_gpu = ExactAreaGpu()
    probe_img = _make_textured_rgb(1530, 1965, dtype=dtype, seed=7)
    dsize = (262, 204)
    if dtype == np.uint8:
        monkeypatch.setattr(area_mod, "_MIN_SOURCE_BYTES", 0)
        probe_gpu = ExactAreaGpu()
    if probe_gpu.resize(probe_img, dsize) is None:
        pytest.skip("OpenCL ExactAreaGpu unavailable on host")

    rgb_path = tmp_path / f"raw_{dtype.__name__}.bin"
    mm = np.memmap(rgb_path, dtype=dtype, mode="w+", shape=probe_img.shape)
    mm[:] = probe_img
    mm.flush()
    del mm

    ro_rgb = np.memmap(rgb_path, dtype=dtype, mode="r", shape=probe_img.shape)
    try:
        bgr_rev = ro_rgb[..., ::-1]
        assert not bgr_rev.flags.c_contiguous
        gpu = ExactAreaGpu()
        actual = gpu.resize(bgr_rev, dsize)
        expected = cv2.resize(bgr_rev, dsize, interpolation=cv2.INTER_AREA)
        assert actual is not None
        assert actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, expected)
    finally:
        if hasattr(ro_rgb, "_mmap") and ro_rgb._mmap is not None:
            ro_rgb._mmap.close()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("layout", ["contig", "bgr_rev", "strided"])
def test_hash_pixels_streaming_matches_sha256_and_cancels(dtype, layout):
    rgb = _make_textured_rgb(64, 96, dtype=dtype, seed=19)
    if layout == "contig":
        arr = rgb
    elif layout == "bgr_rev":
        arr = rgb[..., ::-1]
    else:
        arr = rgb[::2, 1::2, :]
    expected = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    assert oa._hash_pixels(arr, None) == expected

    big_rgb = _make_textured_rgb(1200, 1200, dtype=dtype, seed=23)
    big_arr = big_rgb[..., ::-1] if layout == "bgr_rev" else big_rgb
    assert oa._hash_pixels(big_arr, None) == hashlib.sha256(
        np.ascontiguousarray(big_arr).tobytes()
    ).hexdigest()

    calls = 0

    def cancel_mid():
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(RegistrationCancelled):
        oa._hash_pixels(big_arr, cancel_mid)


def test_alignment_kernels_and_plan_bgr_rev_parallel_bit_exact(monkeypatch):
    if not kernels.available():
        pytest.skip("Numba alignment kernels unavailable")

    base = _make_textured_rgb(544, 640, dtype=np.uint16, seed=31)
    bgr_rev0 = base[..., ::-1]
    contig0 = np.ascontiguousarray(bgr_rev0)
    shift_m = np.float32([[1.0, 0.0, 3.25], [0.0, 1.0, -2.5]])
    contig1 = cv2.warpAffine(contig0, shift_m, (640, 544), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rgb1 = np.ascontiguousarray(contig1[..., ::-1])
    bgr_rev1 = rgb1[..., ::-1]

    gray_par = oa._gray16(bgr_rev0, None)
    with deferred_warp_threads(1):
        gray_ser = oa._gray16(contig0, None)
    np.testing.assert_array_equal(gray_par, gray_ser)

    dh_par = oa._down_half(gray_par, None)
    with deferred_warp_threads(1):
        dh_ser = oa._down_half(gray_ser, None)
    np.testing.assert_array_equal(dh_par, dh_ser)

    fd_par = oa.fast_downsize(gray_par, None)
    with deferred_warp_threads(1):
        fd_ser = oa.fast_downsize(gray_ser, None)
    np.testing.assert_array_equal(fd_par, fd_ser)

    w_tr_par = oa._warp(bgr_rev0, 3.25, -2.5, None)
    with deferred_warp_threads(1):
        w_tr_ser = oa._warp(contig0, 3.25, -2.5, None)
    np.testing.assert_array_equal(w_tr_par, w_tr_ser)

    f8 = (2.75, -1.5, 320.0, 272.0, 1.25, 1.008, 1.008, 1.0)
    w_al_par = oa._warp_alignment(bgr_rev0, f8, None)
    with deferred_warp_threads(1):
        w_al_ser = oa._warp_alignment(contig0, f8, None)
    np.testing.assert_array_equal(w_al_par, w_al_ser)

    mask_par = oa.alignment_valid_mask(
        oa.AlignmentTransform(0, f8, f8, oa.SourceFingerprint(bgr_rev0.shape, bgr_rev0.dtype.str, "x"), 0)
    )
    with deferred_warp_threads(1):
        mask_ser = oa.alignment_valid_mask(
            oa.AlignmentTransform(0, f8, f8, oa.SourceFingerprint(contig0.shape, contig0.dtype.str, "x"), 0)
        )
    np.testing.assert_array_equal(mask_par, mask_ser)

    plan_rev = oa.estimate_alignment_plan([bgr_rev0, bgr_rev1], max_shift_ratio=0.05, rotation=2.0, magnification=2.0)
    used_threads = set()
    gray_rows = kernels.gray16_u16_rows

    def observe_gray_rows(*args):
        used_threads.add(get_ident())
        return gray_rows(*args)

    monkeypatch.setattr(oa.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(kernels, "gray16_u16_rows", observe_gray_rows)
    serial = ImageRegistration("openfocus", defer_warps=True, parameters={
        "max_shift_ratio": 0.05, "rotation": 2.0, "magnification": 2.0,
    })
    serial.process([contig0, contig1], thread_count=1)
    assert used_threads == {get_ident()}
    plan_ctg = serial.alignment_plan
    for t_rev, t_ctg in zip(plan_rev.transforms, plan_ctg.transforms):
        assert t_rev.f8 == t_ctg.f8
        assert t_rev.source.sha256 == t_ctg.source.sha256
    np.testing.assert_array_equal(
        oa.apply_alignment(bgr_rev1, plan_rev.transforms[1]),
        oa.apply_alignment(contig1, plan_ctg.transforms[1]),
    )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("lend_threads", [1, 4])
def test_perspective_warp_bgr_rev_crop_and_lanczos_exact_and_owned(dtype, lend_threads):
    rgb = _make_textured_rgb(560, 640, dtype=dtype, seed=41)
    bgr_rev = rgb[..., ::-1]
    original_contig = np.ascontiguousarray(bgr_rev)

    crop_mat = np.array([[1, 0, -17], [0, 1, -11], [0, 0, 1]], dtype=np.float32)
    crop_dsize = (600, 530)
    crop_src = _perspective_source(bgr_rev, bgr_rev, None)
    crop_warp = PerspectiveWarp(crop_src, tuple(crop_mat.flat), crop_mat.dtype.str, crop_dsize)
    with deferred_warp_threads(lend_threads):
        crop_actual = crop_warp(bgr_rev)
    crop_expected = cv2.warpPerspective(original_contig, crop_mat, crop_dsize, flags=cv2.INTER_LANCZOS4)
    assert crop_actual.flags.c_contiguous
    np.testing.assert_array_equal(crop_actual, crop_expected)
    crop_actual[0, 0] ^= 1
    np.testing.assert_array_equal(bgr_rev, original_contig)

    h_mat = np.array([
        [1.003, -0.008, 4.25],
        [0.006, 0.997, -3.5],
        [2e-5, -1e-5, 1.0],
    ], dtype=np.float32)
    h_src = _perspective_source(bgr_rev, bgr_rev, None)
    h_warp = PerspectiveWarp(h_src, tuple(h_mat.flat), h_mat.dtype.str, crop_dsize)
    with deferred_warp_threads(lend_threads):
        h_actual = h_warp(bgr_rev)
    h_expected = cv2.warpPerspective(
        original_contig, h_mat, crop_dsize, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
    )
    assert h_actual.flags.c_contiguous
    np.testing.assert_array_equal(h_actual, h_expected)


@pytest.mark.parametrize("accelerate_ecc", [False, True])
def test_ecc_registration_direct_rgb16_tiff_bit_exact(tmp_path, accelerate_ecc):
    h, w = 1080, 1920
    base_rgb = _make_textured_rgb(h, w, dtype=np.uint16, seed=53)
    cx, cy = w * 0.5, h * 0.5
    paths = []
    contig_frames = []
    for idx in range(3):
        M = cv2.getRotationMatrix2D((cx, cy), 0.15 * idx, 1.0 + 0.002 * idx)
        M[0, 2] += 2.4 * idx
        M[1, 2] -= 1.6 * idx
        warped_rgb = cv2.warpAffine(
            base_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        p = tmp_path / f"frame_{idx:02d}.tif"
        tifffile.imwrite(str(p), warped_rgb, photometric="rgb", compression=None, rowsperstrip=h)
        paths.append(str(p))
        contig_frames.append(np.ascontiguousarray(warped_rgb[..., ::-1]))

    ok, msg, loaded, _ = ImageStackLoader().load_from_filepaths(paths)
    assert ok, msg
    mapped = map_native_rgb_tiffs(loaded)
    assert all(is_direct_rgb_tiff(f) for f in mapped)

    reg_mapped = ImageRegistration(
        method="ecc", downscale_width=1024, defer_warps=True, accelerate_ecc=accelerate_ecc,
    )
    out_mapped = reg_mapped.process(mapped, thread_count=2)

    reg_contig = ImageRegistration(
        method="ecc", downscale_width=1024, defer_warps=True, accelerate_ecc=accelerate_ecc,
    )
    out_contig = reg_contig.process(contig_frames, thread_count=2)

    st_map = reg_mapped.alignment_plan.stages[0]
    st_ctg = reg_contig.alignment_plan.stages[0]
    assert st_map.dsize == st_ctg.dsize
    assert st_map.bounds == st_ctg.bounds
    np.testing.assert_array_equal(st_map.matrices, st_ctg.matrices)
    reference_digests = [hashlib.sha256(memoryview(frame).cast('B')).hexdigest()
                         for frame in contig_frames]
    for stage in (st_map, st_ctg):
        assert all(source.digest_kind == 'pixel' for source in stage.sources)
        assert [source.sha256 for source in stage.sources] == reference_digests

    for fr_m, fr_c in zip(out_mapped, out_contig, strict=True):
        np.testing.assert_array_equal(resolve_frame(fr_m), resolve_frame(fr_c))


def test_alignment_workers_accepts_callable_deferred_warp_threads():
    assert oa._alignment_workers(128) == 1
    with deferred_warp_threads(lambda: 2):
        assert oa._alignment_workers(1_000_000) == 2
        rgb = _make_textured_rgb(560, 640, dtype=np.uint16, seed=67)
        bgr_rev = rgb[..., ::-1]
        crop_mat = np.array([[1, 0, -5], [0, 1, -4], [0, 0, 1]], dtype=np.float32)
        src = _perspective_source(bgr_rev, bgr_rev, None)
        warp = PerspectiveWarp(src, tuple(crop_mat.flat), crop_mat.dtype.str, (620, 540))
        actual = warp(bgr_rev)
        expected = cv2.warpPerspective(np.ascontiguousarray(bgr_rev), crop_mat, (620, 540), flags=cv2.INTER_LANCZOS4)
        np.testing.assert_array_equal(actual, expected)

    def broken_budget():
        raise RuntimeError("boom")

    with deferred_warp_threads(broken_budget):
        assert oa._alignment_workers(1_000_000) == 1
