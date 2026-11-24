"""Tests for core._ecc_native and its integration in core.registration."""

from __future__ import annotations

import math
from contextlib import nullcontext

import cv2
import numpy as np
import pytest

import core.registration as reg_mod
from core._ecc_native import find_transform_ecc_homography
from core.registration import (
    ImageRegistration,
    RejectedAlignmentFrame,
    _align_ecc_impl,
    alignment_report,
)
from core.source_frames import resolve_frame


def _make_textured_gray(h: int, w: int, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (
        122.0
        + 40.0 * np.sin(xx * (2.0 * math.pi / 71.0)) * np.cos(yy * (2.0 * math.pi / 61.0))
        + 26.0 * np.cos((xx * 0.8 + yy * 1.1) * (2.0 * math.pi / 31.0))
    )
    noise_lo = cv2.GaussianBlur(rng.normal(0.0, 24.0, (h, w)).astype(np.float32), (0, 0), 4.0)
    noise_hi = cv2.GaussianBlur(rng.normal(0.0, 12.0, (h, w)).astype(np.float32), (0, 0), 1.2)
    img = base + noise_lo + noise_hi
    for _ in range(18):
        cx = int(rng.integers(max(2, w // 10), max(3, (9 * w) // 10)))
        cy = int(rng.integers(max(2, h // 10), max(3, (9 * h) // 10)))
        rx = int(rng.integers(max(2, w // 60), max(4, w // 20)))
        ry = int(rng.integers(max(2, h // 60), max(4, h // 20)))
        cv2.ellipse(img, (cx, cy), (rx, ry), float(rng.uniform(0, 180)), 0, 360, float(rng.uniform(35, 220)), -1)
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img.astype(np.float32) / 255.0


def _build_homography(
    w: int,
    h: int,
    tx: float,
    ty: float,
    scale: float = 1.0,
    angle_deg: float = 0.0,
    px: float = 0.0,
    py: float = 0.0,
) -> np.ndarray:
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    th = math.radians(angle_deg)
    c, s = math.cos(th) * scale, math.sin(th) * scale
    M = np.array([
        [c, -s, tx + cx - (c * cx - s * cy)],
        [s, c, ty + cy - (s * cx + c * cy)],
        [px, py, 1.0],
    ], dtype=np.float64)
    M /= M[2, 2]
    return M.astype(np.float32)


def _corner_error(H_est: np.ndarray, H_true: np.ndarray, w: int, h: int) -> float:
    corners = np.array(
        [[0.0, 0.0, 1.0], [w - 1.0, 0.0, 1.0], [w - 1.0, h - 1.0, 1.0], [0.0, h - 1.0, 1.0]],
        dtype=np.float64,
    )
    pe = corners @ H_est.astype(np.float64).T
    pe = pe[:, :2] / pe[:, 2:3]
    pt = corners @ H_true.astype(np.float64).T
    pt = pt[:, :2] / pt[:, 2:3]
    return float(np.sqrt(np.sum((pe - pt) ** 2, axis=1)).max())


def test_ecc_native_identity_fast_path():
    tpl = _make_textured_gray(256, 384, seed=1)
    cc, W = find_transform_ecc_homography(tpl, tpl.copy())
    assert cc == pytest.approx(1.0, abs=1e-6)
    assert W is not None
    assert np.allclose(W, np.eye(3, dtype=np.float32), atol=1e-6)


@pytest.mark.parametrize(
    "name,tx,ty,scale,angle,px,py,blur,gain,bias,tol_px",
    [
        ("small_shift", 1.7, -1.1, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.03),
        ("breathing_rot", 1.3, -0.9, 1.008, 0.4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.04),
        ("perspective", 0.8, 0.6, 1.004, 0.2, 8e-6, -6e-6, 0.0, 1.0, 0.0, 0.06),
        ("large_shift", 12.6, -8.4, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.05),
        ("huge_shift", 24.5, -16.8, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.08),
        ("defocus_breathing", 1.8, -1.3, 1.006, 0.15, 0.0, 0.0, 2.0, 1.0, 0.0, 0.15),
        ("exposure_shift", 2.1, -1.4, 1.005, 0.25, 0.0, 0.0, 0.0, 1.35, 0.08, 0.05),
    ],
)
def test_ecc_native_motion_regimes(name, tx, ty, scale, angle, px, py, blur, gain, bias, tol_px):
    h, w = 360, 640
    tpl = _make_textured_gray(h, w, seed=42)
    M_true = _build_homography(w, h, tx, ty, scale, angle, px, py)
    src = cv2.warpPerspective(tpl, M_true, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if blur > 0.0:
        src = cv2.GaussianBlur(src, (0, 0), blur)
    if gain != 1.0 or bias != 0.0:
        src = np.clip(src * gain + bias, 0.0, 1.0).astype(np.float32)

    cc, W = find_transform_ecc_homography(tpl, src, max_iter=50, eps=1e-4)
    assert cc is not None and W is not None
    assert cc > (0.94 if blur > 0.0 else 0.995)
    err = _corner_error(W, M_true, w, h)
    assert err < tol_px, f"{name}: corner error {err:.4f} >= {tol_px}"


@pytest.mark.parametrize("shape", [(24, 32), (63, 127), (576, 1024)])
def test_ecc_native_edge_shapes_and_strided_input(shape):
    h, w = shape
    tpl = _make_textured_gray(h + 6, w + 8, seed=9)
    M_true = _build_homography(w, h, 1.25, -0.85, scale=1.003, angle_deg=0.2)
    tpl_view = tpl[2:2 + h, 3:3 + w]
    assert not tpl_view.flags.c_contiguous
    src_view = cv2.warpPerspective(np.ascontiguousarray(tpl_view), M_true, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    cc, W = find_transform_ecc_homography(tpl_view, src_view)
    assert cc is not None and W is not None
    tol = 0.22 if min(h, w) < 48 else 0.12
    assert _corner_error(W, M_true, w, h) < tol


def test_ecc_native_fallback_conditions():
    tiny = np.zeros((10, 10), dtype=np.float32)
    assert find_transform_ecc_homography(tiny, tiny) == (None, None)

    flat = np.full((64, 64), 0.5, dtype=np.float32)
    tex = _make_textured_gray(64, 64, seed=5)
    assert find_transform_ecc_homography(flat, tex) == (None, None)
    assert find_transform_ecc_homography(tex, flat) == (None, None)

    bad = tex.copy()
    bad[5, 5] = np.nan
    assert find_transform_ecc_homography(tex, bad) == (None, None)
    assert find_transform_ecc_homography(tex, np.zeros((64, 63), dtype=np.float32)) == (None, None)


def test_registration_flat_frame_propagates_cv2_error():
    rng = np.random.default_rng(3)
    tex_u8 = (np.clip(_make_textured_gray(128, 160, seed=11) * 255.0, 0, 255)).astype(np.uint8)
    tex_bgr = np.stack([tex_u8, tex_u8, tex_u8], axis=2)
    flat_bgr = np.full_like(tex_bgr, 128)
    _ = rng

    with pytest.raises(RejectedAlignmentFrame):
        _align_ecc_impl([tex_bgr, flat_bgr], thread_count=1, failure_policy='fail')

    report = []
    token = alignment_report.set(report)
    try:
        out = _align_ecc_impl([tex_bgr, flat_bgr], thread_count=1, failure_policy='continue')
    finally:
        alignment_report.reset(token)
    assert len(out) == 2
    assert report[-1]['status'] == 'fallback'


def test_registration_monkeypatch_and_fallback_hooks(monkeypatch):
    tpl_g = _make_textured_gray(128, 160, seed=12)
    M = _build_homography(160, 128, 1.5, -1.0)
    src_g = cv2.warpPerspective(tpl_g, M, (160, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    f0 = np.repeat((tpl_g * 255.0).astype(np.uint8)[:, :, None], 3, axis=2)
    f1 = np.repeat((src_g * 255.0).astype(np.uint8)[:, :, None], 3, axis=2)

    # 1. Monkeypatching cv2.findTransformECC bypasses ecc_native
    calls = []

    def fake_find_ecc(t, s, warp, mode, criteria, mask=None, gauss=1):
        calls.append((t.shape, mode))
        return 0.9123, np.eye(3, dtype=np.float32)

    monkeypatch.setattr(reg_mod.cv2, 'findTransformECC', fake_find_ecc)
    report = []
    token = alignment_report.set(report)
    try:
        _align_ecc_impl([f0, f1], thread_count=1)
    finally:
        alignment_report.reset(token)
    assert len(calls) == 1
    assert report[-1]['quality'] == pytest.approx(0.9123, rel=1e-5)

    # Restore original OpenCV reference for next checks
    monkeypatch.setattr(reg_mod.cv2, 'findTransformECC', reg_mod._ORIGINAL_FIND_TRANSFORM_ECC)

    # 2. _load_ecc_native returning None falls back cleanly to OpenCV
    monkeypatch.setattr(reg_mod, '_load_ecc_native', lambda: None)
    out_cv = ImageRegistration(method='ecc', downscale_width=160).process([f0, f1], thread_count=1)
    assert len(out_cv) == 2

    # 3. Solver returning (None, None) falls back cleanly to OpenCV
    monkeypatch.setattr(reg_mod, '_load_ecc_native', lambda: (lambda *a, **kw: (None, None)))
    out_fb = ImageRegistration(method='ecc', downscale_width=160).process([f0, f1], thread_count=1)
    assert len(out_fb) == 2
    assert np.allclose(out_fb[1], out_cv[1])


@pytest.mark.parametrize("dtype_name", ["uint8", "uint16"])
@pytest.mark.parametrize("defer_warps", [False, True])
def test_registration_multithread_determinism(dtype_name, defer_warps):
    h, w = 240, 320
    tpl_f = _make_textured_gray(h, w, seed=21)
    frames = []
    for i in range(4):
        M = _build_homography(w, h, 0.8 * i, -0.5 * i, scale=1.002 ** i, angle_deg=0.08 * i)
        g = cv2.warpPerspective(tpl_f, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        bgr = np.stack([g, np.roll(g, 1, axis=1), np.roll(g, 1, axis=0)], axis=2)
        if dtype_name == "uint8":
            frames.append(np.clip(np.rint(bgr * 255.0), 0, 255).astype(np.uint8))
        else:
            frames.append(np.clip(np.rint(bgr * 65535.0), 0, 65535).astype(np.uint16))

    ref_pixels = None
    ref_transforms = None
    for tc in (1, 2, 4):
        report = []
        token = alignment_report.set(report)
        try:
            reg = ImageRegistration(method='ecc', downscale_width=320, use_gpu=False, defer_warps=defer_warps)
            aligned = reg.process(frames, output_path=None, thread_count=tc)
        finally:
            alignment_report.reset(token)
        mats = [entry['transform'] for entry in report if 'transform' in entry]
        resolved = [np.asarray(resolve_frame(item)) for item in aligned]
        if ref_pixels is None:
            ref_pixels = resolved
            ref_transforms = mats
        else:
            assert len(mats) == len(ref_transforms)
            for a, b in zip(mats, ref_transforms):
                assert np.allclose(a, b, atol=1e-5, rtol=1e-5)
            for p_cur, p_ref in zip(resolved, ref_pixels):
                assert np.array_equal(p_cur, p_ref)


def test_ecc_cpp_matches_numba_fallback_and_pyramid_api(monkeypatch):
    from core import _ecc_native, _native

    assert _native.native_available()
    tpl = _make_textured_gray(320, 480, seed=77)
    M_true = _build_homography(480, 320, 4.2, -3.1, scale=1.007, angle_deg=0.35, px=4e-6, py=-3e-6)
    src = cv2.warpPerspective(tpl, M_true, (480, 320), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    cc_cpp, W_cpp = _ecc_native.find_transform_ecc_homography(tpl, src, num_threads=2)
    tpl_pyr = _ecc_native.build_ecc_pyramid(tpl)
    src_pyr = _ecc_native.build_ecc_pyramid(src)
    cc_pyr, W_pyr = _ecc_native.find_transform_ecc_homography_pyramids(tpl_pyr, src_pyr, num_threads=1)
    assert cc_cpp is not None and W_cpp is not None and W_pyr is not None
    assert cc_cpp == pytest.approx(cc_pyr, abs=1e-6)
    assert _corner_error(W_cpp, W_pyr, 480, 320) < 1e-4

    monkeypatch.setattr(_native, 'load_native', lambda: None)
    cc_nb, W_nb = _ecc_native.find_transform_ecc_homography(tpl, src)
    assert cc_nb is not None and W_nb is not None
    assert cc_cpp == pytest.approx(cc_nb, abs=1e-5)
    assert _corner_error(W_cpp, W_nb, 480, 320) < 0.01


def test_jpeg_ecc_dct_fast_path_and_file_fingerprint_roundtrip(tmp_path, monkeypatch):
    from PIL import Image
    from core import frame_resolution_cache, frame_storage, source_frames
    from core.openfocus_alignment import SourceFingerprint
    from core.registration import PerspectiveAlignmentPlan, PerspectiveStage

    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(frame_storage, '_storage_directory', str(scratch))
    tpl = _make_textured_gray(1500, 2000, seed=31)
    M_true = _build_homography(2000, 1500, 3.5, -2.5, scale=1.003, angle_deg=0.15)
    src = cv2.warpPerspective(tpl, M_true, (2000, 1500), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    frames = []
    for idx, plane in enumerate((tpl, src)):
        rgb = np.repeat(np.clip(np.rint(plane * 255.0), 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
        p = tmp_path / f'frame_{idx}.jpg'
        Image.fromarray(rgb, 'RGB').save(p, format='JPEG', quality=95)
        frames.append(source_frames.scan_source(str(p), header_only=True)[0])

    # Clear SOURCE_CACHE so we prove _align_ecc_impl did not populate full-res source cache via resolve_frame.
    source_frames.SOURCE_CACHE.clear()
    report = []
    token = alignment_report.set(report)
    stages = []
    try:
        with frame_storage.scratch_directory(), frame_resolution_cache.render_scope(frames):
            out = _align_ecc_impl(
                frames,
                downscale_width=1000,
                thread_count=2,
                defer_warps=True,
                plan_callback=stages.append,
            )
            cache_entries = len(list(scratch.glob('*')))
    finally:
        alignment_report.reset(token)

    assert len(out) == 2 and len(stages) == 1
    assert cache_entries == 0
    assert all(src_fp.digest_kind == 'file' for src_fp in stages[0].sources)
    assert report[1]['status'] == 'aligned'
    err = _corner_error(np.linalg.inv(report[1]['transform']), M_true, 2000, 1500)
    assert err < 0.25

    plan = PerspectiveAlignmentPlan((stages[0],), '{}', 1000)
    restored = PerspectiveAlignmentPlan.from_dict(plan.to_dict())
    assert restored == plan
    # validate_pixels succeeds on matching shape/dtype & unchanged disk file
    dummy_arr = np.empty(frames[0].shape, dtype=frames[0].dtype)
    restored.stages[0].sources[0].validate_pixels(dummy_arr, source=frames[0])
    # File tampering invalidates file-digest fingerprint
    with open(frames[0].path, 'ab') as fh:
        fh.write(b'x')
    with pytest.raises((OSError, ValueError)):
        restored.stages[0].sources[0].validate_pixels(dummy_arr, source=frames[0])
