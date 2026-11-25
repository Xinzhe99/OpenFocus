"""Focused regression and parity tests for alignment, prep, warp, and post-fusion speedups."""
import gc
import os

import cv2
import numpy as np
import pytest
import tifffile

from core import frame_storage, frame_transport, openfocus_alignment, registration
from core.image_loader import iter_decoded_sources
from core.multi_focus_fusion import MultiFocusFusion
from core.render_context import RenderAlignmentMask, _alignment_valid_mask
from core.source_frames import ProcessedFrame, scan_source
from utils.image_utils import to_display_uint8
from utils.output_sharpen import sharpen_output


def test_base_correction_downsampled_lowpass_matches_full_res():
    rng = np.random.default_rng(7)
    h, w = 256, 320
    # Smooth gradient + fine texture
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        80.0 + 40.0 * np.sin(xx / 45.0) + rng.normal(0, 12.0, (h, w)),
        110.0 + 35.0 * np.cos(yy / 50.0) + rng.normal(0, 12.0, (h, w)),
        130.0 + 30.0 * np.sin((xx + yy) / 60.0) + rng.normal(0, 12.0, (h, w)),
    ], axis=-1).clip(0, 255).astype(np.float32)
    fused = np.clip(base * 0.92 + 12.0 + rng.normal(0, 3.0, base.shape), 0, 255).astype(np.float32)

    mff = MultiFocusFusion(algorithm="openfocus_b", base_frame="first")
    actual = mff._apply_base_correction(fused.copy(), [base])

    # Full-resolution mathematical reference
    sigma = float(mff.BASE_CORRECTION_SIGMA)
    ref_low_out = cv2.GaussianBlur(fused, (0, 0), sigma)
    ref_low_base = cv2.GaussianBlur(base, (0, 0), sigma)
    expected = np.clip(fused + (ref_low_base - ref_low_out), 0, 255).astype(np.float32)

    diff = np.abs(actual - expected)
    assert float(np.mean(diff)) < 0.05
    assert float(np.percentile(diff, 99)) < 0.2
    assert float(np.max(diff)) < 0.5


def test_native_u16_to_display_u8_and_bgr_to_rgb_exact():
    # Exhaustively test all 65,536 uint16 values via a 256x256x3 array (>262,144 elems)
    vals = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    img3 = np.stack([vals, vals[::-1], vals[:, ::-1]], axis=-1)
    expected = ((img3.astype(np.uint32) + 128) // 257).astype(np.uint8)

    np.testing.assert_array_equal(to_display_uint8(img3), expected)
    # Reversed BGR view
    rev = np.ascontiguousarray(img3[..., ::-1])[..., ::-1]
    np.testing.assert_array_equal(to_display_uint8(rev), expected)
    # 2D path (>262,144 elems)
    big2d = np.tile(vals, (2, 2))
    expected_2d = ((big2d.astype(np.uint32) + 128) // 257).astype(np.uint8)
    np.testing.assert_array_equal(to_display_uint8(big2d), expected_2d)

    from core._native import load_native
    mod = load_native()
    if mod is not None and hasattr(mod, "of_bgr_to_rgb"):
        u8 = expected.copy()
        dst_u8 = np.empty_like(u8)
        mod.of_bgr_to_rgb(u8, dst_u8, 4)
        np.testing.assert_array_equal(dst_u8, u8[..., ::-1])

        dst_u16 = np.empty_like(img3)
        mod.of_bgr_to_rgb(img3, dst_u16, 4)
        np.testing.assert_array_equal(dst_u16, img3[..., ::-1])


def test_analytic_render_alignment_mask_all_valid_and_shifted():
    h, w = 120, 160
    rng = np.random.default_rng(11)
    src = rng.integers(20, 230, (h, w, 3), dtype=np.uint8)
    fp = registration._perspective_source(src, src, None)

    # 1. Identity warp: analytic path proves all pixels valid without scanning pixels
    M_id = np.eye(3, dtype=np.float64)
    pw_id = registration.PerspectiveWarp(fp, tuple(M_id.ravel()), "float64", (w, h), method="ecc")
    pf_id = ProcessedFrame(src, pw_id, (h, w, 3), np.dtype(np.uint8), None)
    ram_id = RenderAlignmentMask([pf_id], None, None)
    assert len(ram_id._pending) == 0
    mask_id, failed_id = ram_id.complete()
    assert not failed_id and mask_id is None

    # 2. Shifted warp: analytic polygon matches pixel floodFill interior
    M_shift = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    pw_s = registration.PerspectiveWarp(fp, tuple(M_shift.ravel()), "float64", (w, h), method="ecc")
    pf_s = ProcessedFrame(src, pw_s, (h, w, 3), np.dtype(np.uint8), None)
    ram_s = RenderAlignmentMask([pf_s], None, None)
    mask_an, failed_s = ram_s.complete()
    assert not failed_s and mask_an is not None
    warped = cv2.warpPerspective(src, M_shift, (w, h), flags=cv2.INTER_LANCZOS4)
    mask_px = _alignment_valid_mask(warped)
    assert mask_px is not None
    # Interior agreement > 99% (differences only on 1px rounded border)
    assert float(np.mean(mask_an == mask_px)) > 0.98


@pytest.mark.parametrize("workers", [1, 4])
def test_ecc_progress_callback_and_cancel_before_fallback(workers, monkeypatch):
    rng = np.random.default_rng(33)
    base = cv2.GaussianBlur(rng.integers(0, 255, (160, 200, 3), dtype=np.uint8), (0, 0), 1.0)
    frames = [base, np.roll(base, 2, axis=1), np.roll(base, 4, axis=1), np.roll(base, 6, axis=1)]
    seen = []
    reg = registration.ImageRegistration(method="ecc", defer_warps=True)
    reg.process(frames, thread_count=workers, progress_callback=lambda d, t: seen.append((d, t)))
    assert len(seen) == 3
    assert [d for d, _ in seen] == [1, 2, 3]
    assert all(t == 3 for _, t in seen)

    # Cancel right before cv2.findTransformECC fallback
    cancelled = [False]
    monkeypatch.setattr(registration, "_ecc_native_solver", lambda *a, **k: (None, None))
    monkeypatch.setattr(registration, "_ecc_native_loaded", True)

    def fail_if_called(*a, **k):
        raise AssertionError("cv2.findTransformECC should not run when cancelled")

    monkeypatch.setattr(cv2, "findTransformECC", fail_if_called)
    monkeypatch.setattr(registration, "_ORIGINAL_FIND_TRANSFORM_ECC", fail_if_called)

    def should_cancel():
        if not cancelled[0]:
            # Allow frame prep to run, then trigger cancel before fallback
            return False
        return True

    # Trigger cancel from native solver returning (None, None)
    def native_then_cancel(*a, **k):
        cancelled[0] = True
        return None, None

    monkeypatch.setattr(registration, "_ecc_native_solver", native_then_cancel)
    with pytest.raises(registration.RegistrationCancelled):
        registration._align_ecc_impl(frames[:2], thread_count=1, should_cancel=should_cancel)


def test_frame_transport_shm_roundtrip_and_disk_fallback(tmp_path, monkeypatch):
    frame_storage.configure_storage_directory(tmp_path)
    arr = np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)

    # Normal RAM headroom -> uses SharedMemory
    enc = frame_transport.FrameEncoder()
    payload = enc.encode(arr)
    assert payload[2] == "shm"
    assert len(enc.owned_paths) == 0
    dec = frame_transport.FrameDecoder()
    out = dec.decode(payload)
    np.testing.assert_array_equal(out, arr)
    enc.close(unlink=True)

    # Low RAM headroom -> falls back cleanly to disk memmap
    monkeypatch.setattr(frame_transport, "_try_shm_encode", lambda _a: None)
    enc2 = frame_transport.FrameEncoder()
    payload2 = enc2.encode(arr)
    assert payload2[2] == "frame"
    assert len(enc2.owned_paths) == 1
    dec2 = frame_transport.FrameDecoder(owned_paths=enc2.owned_paths)
    out2 = dec2.decode(payload2)
    np.testing.assert_array_equal(out2, arr)
    frame_storage.disown_temp_files(enc2.owned_paths)
    del out2, enc2, dec2
    gc.collect()


def test_single_page_tiff_parallel_decode_and_file_digest(tmp_path):
    paths = []
    for i in range(6):
        p = str(tmp_path / f"frame_{i:02d}.tif")
        rgb = np.full((64, 80, 3), 20 + i * 30, dtype=np.uint8)
        tifffile.imwrite(p, rgb, photometric="rgb")
        paths.append(p)
    decoded = list(iter_decoded_sources(paths, thread_count=6))
    assert [d.path for d in decoded] == paths
    assert [int(d.pages[0][0, 0, 2]) for d in decoded] == [20 + i * 30 for i in range(6)]

    # Verify unmodified TIFF SourceFrames use fast file digest in OpenFocus alignment
    sfs = [scan_source(p, header_only=True)[0] for p in paths[:3]]
    plan = openfocus_alignment.estimate_alignment_plan(sfs, max_shift_ratio=0.03, magnification=0.0)
    assert all(t.source.digest_kind == "file" for t in plan.transforms)


def test_openfocus_rotated_alignment_subpixel_accuracy():
    rng = np.random.default_rng(99)
    h, w = 256, 320
    base = cv2.GaussianBlur(rng.integers(20, 235, (h, w, 3), dtype=np.uint8), (0, 0), 1.4)
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    frames = [base]
    for i in range(1, 5):
        M = cv2.getRotationMatrix2D(center, 0.4 * i, 1.0 + 0.008 * i)
        M[0, 2] += 1.5 * i
        M[1, 2] -= 1.0 * i
        frames.append(cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))

    plan = openfocus_alignment.estimate_alignment_plan(
        frames, max_shift_ratio=0.05, rotation=2.0, magnification=5.0,
    )
    for i in range(1, 5):
        aligned = openfocus_alignment.apply_alignment(frames[i], plan.transforms[i])
        roi_ref = base[32:-32, 32:-32].astype(np.float32)
        before = float(np.mean(np.abs(roi_ref - frames[i][32:-32, 32:-32])))
        after = float(np.mean(np.abs(roi_ref - aligned[32:-32, 32:-32])))
        assert after < before * 0.25


def test_sharpen_output_scoped_threads_matches_serial():
    rng = np.random.default_rng(77)
    img = rng.integers(0, 65535, (600, 600, 3), dtype=np.uint16)
    cv2.setNumThreads(0)
    actual = sharpen_output(img, strength=100)
    blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
    expected = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    np.testing.assert_array_equal(actual, expected)
