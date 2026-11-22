"""Tests for region-direct deferred warping, direct-RGB digest reuse, and fused OpenFocus 4D search."""

from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest
import tifffile

from core import _alignment_kernels as kernels
from core import frame_resolution_cache as frc
from core import openfocus_alignment as oa
from core.aligned_previews import can_preview, registered_preview
from core.frame_storage import scratch_directory
from core.image_loader import ImageStackLoader
from core.registration import (
    ImageRegistration,
    PerspectiveWarp,
    RegistrationCancelled,
    _perspective_source,
    apply_alignment_plan,
)
from core.source_frames import (
    SOURCE_CACHE,
    ProcessedFrame,
    borrow_frame_region,
    crop_frame,
    is_direct_rgb_tiff,
    map_native_rgb_tiffs,
    resolve_frame,
    resolve_region,
    select_frame_scratch,
)


def _textured_rgb(h: int, w: int, dtype=np.uint16, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 255.0
    lh, lw = max(8, h // 8), max(8, w // 8)
    if np.issubdtype(dtype, np.integer):
        low = rng.integers(max(1, int(max_val) // 10), int(max_val * 0.9), (lh, lw, 3), dtype=dtype)
    else:
        low = rng.uniform(10.0, 230.0, (lh, lw, 3)).astype(np.float32)
    base = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wave = (np.sin(xx * 0.13 + yy * 0.07) + np.cos(xx * 0.09 - yy * 0.11)) * (float(max_val) * 0.05)
    if np.issubdtype(dtype, np.integer):
        return np.clip(base.astype(np.int32) + wave[..., None].astype(np.int32), 0, int(max_val)).astype(dtype)
    return np.ascontiguousarray(base + wave[..., None], dtype=dtype)


def _direct_rgb_tiff_frames(tmp_path: Path, rgb_list: list[np.ndarray], prefix: str = "f"):
    paths = []
    for idx, rgb in enumerate(rgb_list):
        p = tmp_path / f"{prefix}_{idx:02d}.tif"
        tifffile.imwrite(str(p), np.ascontiguousarray(rgb), photometric="rgb", compression=None, rowsperstrip=rgb.shape[0])
        paths.append(str(p))
    ok, msg, loaded, _ = ImageStackLoader().load_from_filepaths(paths)
    assert ok, msg
    mapped = map_native_rgb_tiffs(loaded)
    assert all(is_direct_rgb_tiff(f) for f in mapped)
    return mapped, paths


def test_direct_rgb_tiff_source_digest_lifecycle(tmp_path, monkeypatch):
    rgb = _textured_rgb(96, 128, dtype=np.uint16, seed=17)
    (frame,), (tiff_path,) = _direct_rgb_tiff_frames(tmp_path, [rgb])

    assert frc.get_source_digest(frame) is None
    calls = []
    orig_hash = oa._hash_pixels

    def counted_hash(image, cancel):
        calls.append(1)
        return orig_hash(image, cancel)

    monkeypatch.setattr(oa, "_hash_pixels", counted_hash)

    with frc.render_scope([frame]) as scope:
        fp = _perspective_source(frame, resolve_frame(frame), None)
        assert not scope.enabled
        assert len(calls) == 1

        # Second lookup reuses scope source digest without planning or hashing.
        assert fp.validate_direct_source(frame, None) is True
        assert not scope.enabled
        assert len(calls) == 1

        # Calling validate_pixels with source=frame also reuses the remembered digest.
        fp.validate_pixels(resolve_frame(frame), None, source=frame)
        assert len(calls) == 1

        # Mismatched sha256 still raises.
        bad_fp = oa.SourceFingerprint(fp.shape, fp.dtype, "0" * 64, fp.path, fp.page, fp.file_stamp, fp.context)
        with pytest.raises(ValueError, match="pixels changed"):
            bad_fp.validate_direct_source(frame, None)

        # File change immediately invalidates cached digest.
        with open(tiff_path, "ab") as stream:
            stream.write(b"\x00\x00")
        with pytest.raises(OSError, match="Source changed"):
            fp.validate_direct_source(frame, None)

    assert frc.get_source_digest(frame) is None


def test_source_digest_cancelled_not_remembered(tmp_path):
    rgb = _textured_rgb(64, 80, dtype=np.uint16, seed=21)
    (frame,), _ = _direct_rgb_tiff_frames(tmp_path, [rgb])
    with frc.render_scope([frame]):
        with pytest.raises(RegistrationCancelled):
            oa._pixel_digest(resolve_frame(frame), lambda: True, source=frame)
        assert frc.get_source_digest(frame) is None


def test_score_grid_4d_slice_matches_chunk_fallback_bit_exact(monkeypatch):
    if not kernels.available():
        pytest.skip("Numba alignment kernels unavailable")

    ref_rgb = _textured_rgb(137, 173, dtype=np.uint16, seed=31)
    m = cv2.getRotationMatrix2D((86.0, 68.0), 1.35, 1.012)
    m[0, 2] += 3.2
    m[1, 2] -= 2.4
    test_rgb = cv2.warpAffine(ref_rgb, m, (173, 137), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    ref_u16 = oa._alignment_gray16(ref_rgb, None)
    test_u16 = oa._alignment_gray16(test_rgb, None)

    # Fused native path
    f8_fast, r_fast = oa.align_pair_wide(
        ref_u16, test_u16, h_pct=6.0, v_pct=6.0, rot_deg=3.0, mag_pct=2.5,
        min_dim=64, divisor=100, grid_half=6, max_rounds=12,
    )

    # Fallback path via monkeypatched _score_chunk wrapper
    orig_score = oa._score_chunk
    monkeypatch.setattr(oa, "_score_chunk", lambda *args, **kwargs: orig_score(*args, **kwargs))
    f8_ref, r_ref = oa.align_pair_wide(
        ref_u16, test_u16, h_pct=6.0, v_pct=6.0, rot_deg=3.0, mag_pct=2.5,
        min_dim=64, divisor=100, grid_half=6, max_rounds=12,
    )
    assert r_fast == r_ref
    np.testing.assert_array_equal(f8_fast.view(np.uint32), f8_ref.view(np.uint32))

    # Constant-zero frames test identical earliest minimum tie-breaking
    flat_a = np.zeros((71, 93), dtype=np.uint16)
    flat_b = np.zeros((71, 93), dtype=np.uint16)
    monkeypatch.undo()
    f8_flat_fast, _ = oa.align_pair_wide(flat_a, flat_b, h_pct=4.0, v_pct=4.0, rot_deg=2.0, mag_pct=1.0, grid_half=4)
    monkeypatch.setattr(oa, "_score_chunk", lambda *args, **kwargs: orig_score(*args, **kwargs))
    f8_flat_ref, _ = oa.align_pair_wide(flat_a, flat_b, h_pct=4.0, v_pct=4.0, rot_deg=2.0, mag_pct=1.0, grid_half=4)
    np.testing.assert_array_equal(f8_flat_fast.view(np.uint32), f8_flat_ref.view(np.uint32))

    # Cancellation during search
    monkeypatch.undo()
    checks = 0

    def cancel_soon():
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(RegistrationCancelled):
        oa.align_pair_wide(ref_u16, test_u16, h_pct=8.0, v_pct=8.0, rot_deg=4.0, mag_pct=4.0, should_cancel=cancel_soon)


def test_estimate_alignment_plan_pyramid_reuse_exact(tmp_path):
    base = _textured_rgb(160, 200, dtype=np.uint16, seed=41)
    rgb0 = cv2.warpAffine(base, np.float32([[1, 0, 2.0], [0, 1, -1.5]]), (200, 160), borderMode=cv2.BORDER_REFLECT)
    rgb1 = base.copy()
    rgb2 = base.copy()  # identical adjacent pair triggers prev_pyr reuse
    rgb3 = cv2.warpAffine(base, np.float32([[1, 0, -2.5], [0, 1, 1.8]]), (200, 160), borderMode=cv2.BORDER_REFLECT)
    frames = [np.ascontiguousarray(x[..., ::-1]) for x in (rgb0, rgb1, rgb2, rgb3)]

    plan = oa.estimate_alignment_plan(
        frames, reference_frame=2, max_shift_ratio=0.05, rotation=2.0, magnification=2.0,
    )
    grays = [oa._alignment_gray16(x, None) for x in frames]
    pair10, rounds10 = oa.align_pair_wide(grays[1], grays[0], h_pct=5.0, v_pct=5.0, rot_deg=2.0, mag_pct=2.0)
    assert plan.transforms[0].f8 == tuple(float(v) for v in pair10)
    assert plan.transforms[0].rounds == rounds10
    pair13, rounds13 = oa.align_pair_wide(grays[1], grays[3], h_pct=5.0, v_pct=5.0, rot_deg=2.0, mag_pct=2.0)
    assert plan.transforms[3].f8 == tuple(float(v) for v in pair13)
    assert plan.transforms[3].rounds == rounds13


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("channels", [None, 1, 3, 4])
def test_perspective_warp_region_bit_exact(dtype, channels):
    h, w = 180, 220
    rgb = _textured_rgb(h, w, dtype=dtype, seed=53)
    if channels is None:
        img = np.ascontiguousarray(rgb[:, :, 0])
    elif channels == 1:
        img = np.ascontiguousarray(rgb[:, :, :1])
    elif channels == 3:
        img = rgb[..., ::-1]  # also exercises bgr_rev for uint8/uint16
    else:
        alpha = np.full((h, w, 1), 255 if dtype != np.float32 else 1.0, dtype=dtype)
        img = np.ascontiguousarray(np.concatenate([rgb, alpha], axis=2))

    if dtype == np.float32:
        img = np.ascontiguousarray(img)
        img[20, 30] = np.nan
        img[40, 50] = -0.0

    src_fp = _perspective_source(img, img, None)
    matrices = [
        np.array([[1.0, 0.0, -7.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([[0.985, 0.018, -11.4], [-0.014, 0.991, 9.7], [8e-5, -6e-5, 1.0]], dtype=np.float32),
        np.array([[1.02, -0.015, 25.0], [0.012, 1.01, -18.0], [-5e-5, 7e-5, 1.0]], dtype=np.float64),
    ]
    dsize = (196, 160)
    rois = [
        (0, 0, 64, 64),
        (53, 41, 79, 67),
        (132, 96, 64, 64),
        (195, 159, 1, 1),
        (10, 10, 0, 10),
    ]
    for mat in matrices:
        warp = PerspectiveWarp(src_fp, tuple(mat.flat), mat.dtype.str, dsize)
        full = warp(img)
        for rx, ry, rw, rh in rois:
            actual = warp.warp_region(img, rx, ry, rw, rh)
            expected = full[ry:ry + rh, rx:rx + rw]
            if dtype == np.float32:
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            else:
                np.testing.assert_array_equal(actual, expected)
            if channels in (3, 4) and rw > 0 and rh > 0:
                actual3 = warp.warp_region(img, rx, ry, rw, rh, channels=2)
                np.testing.assert_array_equal(
                    actual3.view(np.uint32) if dtype == np.float32 else actual3,
                    expected[:, :, :2].view(np.uint32) if dtype == np.float32 else expected[:, :, :2],
                )


def test_perspective_warp_region_direct_rgb_tiff_closes_handles_on_windows(tmp_path):
    rgb = _textured_rgb(200, 260, dtype=np.uint16, seed=61)
    (frame,), (tiff_path,) = _direct_rgb_tiff_frames(tmp_path, [rgb])
    expected_full = np.ascontiguousarray(rgb[..., ::-1])

    for mat in (
        np.array([[1.0, 0.0, -8.0], [0.0, 1.0, -6.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([[0.99, 0.012, -9.25], [-0.01, 0.995, 7.5], [4e-5, -3e-5, 1.0]], dtype=np.float32),
    ):
        fp = _perspective_source(frame, resolve_frame(frame), None)
        warp = PerspectiveWarp(fp, tuple(mat.flat), mat.dtype.str, (240, 180))
        full = warp(expected_full)
        with frc.render_scope([frame]):
            region = warp.warp_region(frame, 48, 36, 96, 80)
            np.testing.assert_array_equal(region, full[36:116, 48:144])

    # Windows enforces file locks strictly; verify no open mmap handle remains.
    Path(tiff_path).unlink()
    assert not Path(tiff_path).exists()


@pytest.mark.parametrize("use_numba", [True, False])
def test_apply_alignment_and_translation_region_bit_exact_and_unlinkable(tmp_path, monkeypatch, use_numba):
    if use_numba and not kernels.available():
        pytest.skip("Numba alignment kernels unavailable")
    if not use_numba:
        monkeypatch.setattr(kernels, "available", lambda: False)

    rgb0 = _textured_rgb(144, 180, dtype=np.uint16, seed=71)
    rgb1 = cv2.warpAffine(rgb0, np.float32([[1, 0, 3.2], [0, 1, -2.4]]), (180, 144), borderMode=cv2.BORDER_REFLECT)
    frames, paths = _direct_rgb_tiff_frames(tmp_path, [rgb0, rgb1])

    align_plan = oa.estimate_alignment_plan(frames, max_shift_ratio=0.05, rotation=2.0, magnification=2.0)
    trans_plan = oa.estimate_translation_plan(frames, max_shift_ratio=0.05)

    for idx, frame in enumerate(frames):
        full_arr = np.ascontiguousarray(resolve_frame(frame))
        full_a = oa.apply_alignment(full_arr, align_plan.transforms[idx])
        full_t = oa.apply_translation(full_arr, trans_plan.transforms[idx])
        for rx, ry, rw, rh in [(0, 0, 64, 64), (41, 29, 75, 59), (116, 80, 64, 64)]:
            reg_a = oa.apply_alignment_region(frame, align_plan.transforms[idx], rx, ry, rw, rh)
            reg_t = oa.apply_translation_region(frame, trans_plan.transforms[idx], rx, ry, rw, rh)
            np.testing.assert_array_equal(reg_a, full_a[ry:ry + rh, rx:rx + rw])
            np.testing.assert_array_equal(reg_t, full_t[ry:ry + rh, rx:rx + rw])

    # Verify float32 similarity region parity too (internal float domain is [0, 255])
    f_arr = np.ascontiguousarray(rgb1.astype(np.float32) * np.float32(255.0 / 65535.0))
    f_fp = oa.SourceFingerprint(f_arr.shape, f_arr.dtype.str, oa._hash_pixels(f_arr, None))
    f_tf = oa.AlignmentTransform(1, align_plan.transforms[1].f8, align_plan.transforms[1].pair_f8, f_fp, 0)
    full_f = oa.apply_alignment(f_arr, f_tf)
    reg_f = oa.apply_alignment_region(f_arr, f_tf, 25, 19, 80, 70)
    np.testing.assert_array_equal(reg_f.view(np.uint32), full_f[19:89, 25:105].view(np.uint32))

    # Immediate delete verifies no dangling mmap handle on Windows.
    for p in paths:
        Path(p).unlink()


def test_region_dispatch_borrow_cache_and_preview_contracts(tmp_path, monkeypatch):
    rgb0 = _textured_rgb(160, 200, dtype=np.uint16, seed=83)
    rgb1 = cv2.warpAffine(rgb0, np.float32([[1, 0, 2.5], [0, 1, -1.5]]), (200, 160), borderMode=cv2.BORDER_REFLECT)
    frames, _ = _direct_rgb_tiff_frames(tmp_path, [rgb0, rgb1])

    reg = ImageRegistration("ecc", downscale_width=128, defer_warps=True)
    deferred = reg.process(frames, thread_count=2)
    expected_full = [resolve_frame(f).copy() for f in deferred]

    # Previews remain geometric over deferred crop chains.
    cropped_once = crop_frame(deferred[1], 16, 12, 120, 100)
    cropped_twice = crop_frame(cropped_once, 10, 8, 64, 56)
    assert can_preview(cropped_twice)
    prev = registered_preview(cropped_twice)
    assert prev is not None and prev[0].size > 0

    # Without disk cache: crop_frame and borrow_frame_region use warp_region without calling _apply_to_image.
    called_full = []
    with monkeypatch.context() as mp:
        mp.setattr(PerspectiveWarp, "_apply_to_image", lambda *a, **k: called_full.append(1))
        tile_crop = resolve_frame(cropped_twice)
        np.testing.assert_array_equal(tile_crop, expected_full[1][20:76, 26:90])
        assert not called_full

        tile_resolved = resolve_region(deferred[1], 26, 20, 64, 56)
        np.testing.assert_array_equal(tile_resolved, expected_full[1][20:76, 26:90])
        assert tile_resolved.flags.owndata and not called_full

        with borrow_frame_region(deferred[1], np.s_[20:76, 26:90]) as borrowed:
            np.testing.assert_array_equal(borrowed, expected_full[1][20:76, 26:90])
        assert not called_full

    # Monkeypatching PerspectiveWarp.__call__ disables the direct region fast path and invokes the override.
    with monkeypatch.context() as mp:
        orig_warp_call = PerspectiveWarp.__call__
        override_calls = []

        def tracked_warp_call(self, image):
            override_calls.append(1)
            return orig_warp_call(self, image)

        mp.setattr(PerspectiveWarp, "__call__", tracked_warp_call)
        tile_fallback = resolve_region(deferred[1], 26, 20, 64, 56)
        np.testing.assert_array_equal(tile_fallback, expected_full[1][20:76, 26:90])
        assert len(override_calls) == 1

    # Direct RGB SourceFrame in borrow_frame_region keeps zero-copy memmap path.
    with borrow_frame_region(frames[0], np.s_[10:42, 12:44]) as raw_slice:
        np.testing.assert_array_equal(raw_slice, rgb0[10:42, 12:44, ::-1])

    # Within render_scope + sufficient scratch: borrow_frame_region still caches full warped frame once.
    monkeypatch.undo()
    with scratch_directory():
        select_frame_scratch(frames)
        with frc.render_scope(frames) as scope:
            with borrow_frame_region(deferred[1], np.s_[10:50, 10:50]) as b1:
                np.testing.assert_array_equal(b1, expected_full[1][10:50, 10:50])
            assert scope.enabled
            assert scope.get(deferred[1]) is not None
            monkeypatch.setattr(PerspectiveWarp, "__call__", lambda *a, **k: pytest.fail("cache hit called full warp"))
            with borrow_frame_region(deferred[1], np.s_[20:60, 20:60]) as b2:
                np.testing.assert_array_equal(b2, expected_full[1][20:60, 20:60])

    # apply_alignment_plan eager vs deferred + store_source=False
    plan_oa = oa.estimate_alignment_plan(frames, max_shift_ratio=0.05, rotation=1.0, magnification=1.0)
    eager = apply_alignment_plan(frames, plan_oa, defer_warps=False)
    def_oa = apply_alignment_plan(frames, plan_oa, defer_warps=True)
    for e_img, d_fr in zip(eager, def_oa):
        np.testing.assert_array_equal(resolve_frame(d_fr), e_img)
        if isinstance(d_fr, ProcessedFrame) and isinstance(d_fr.source, ProcessedFrame):
            assert d_fr.store_source is False
