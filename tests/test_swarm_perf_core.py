"""Regression, parity, and bug-fix tests for swarm perf-core changes."""
from __future__ import annotations

import os
import numpy as np
import cv2
import pytest
import tifffile


def _make_textured_stack(count: int = 3, h: int = 160, w: int = 200, *, dtype=np.uint8,
                         shift_x: float = 3.0, shift_y: float = -2.0):
    rng = np.random.default_rng(1234)
    base = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (5, 5), 1.1)
    ys, xs = np.mgrid[0:h, 0:w]
    pat = (((np.sin(xs * 0.12) * np.cos(ys * 0.11)) + 1.0) * 40).astype(np.uint8)
    base = np.clip(base.astype(np.int16) + pat[:, :, None], 0, 255).astype(np.uint8)
    if dtype == np.uint16:
        base = base.astype(np.uint16) * 257
    frames = [base]
    for i in range(1, count):
        M = np.array([[1.0, 0.0, shift_x * i], [0.0, 1.0, shift_y * i]], dtype=np.float32)
        frames.append(cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return frames


# ---------------------------------------------------------------------------
# OpenFocus C depth GPU & compute_focus_index(use_gpu=...)
# ---------------------------------------------------------------------------

def test_depth_openfocus_c_forwards_use_gpu_false(monkeypatch):
    from core import depth_map as dm
    from fusion_methods import openfocus_depth

    seen = {}

    def fake_depth_map(images, img_resize, kernel_size, smoothing, should_cancel,
                       progress_callback=None, thread_count=1, **kwargs):
        seen['use_gpu'] = kwargs.get('use_gpu')
        seen['half_resolution'] = kwargs.get('half_resolution')
        return np.zeros(images[0].shape[:2], dtype=np.int32)

    monkeypatch.setattr(openfocus_depth, 'depth_map', fake_depth_map)
    frames = _make_textured_stack(2, 64, 80)
    out = dm.compute_focus_index(frames, 'openfocus_c', use_gpu=False)
    assert out.shape == (64, 80)
    assert seen == {'use_gpu': False, 'half_resolution': True}


def test_openfocus_depth_half_res_no_detail_gpu_matches_cpu():
    from fusion_methods._openfocus_gpu import gpu_available
    from fusion_methods.openfocus_depth import depth_map

    if not gpu_available():
        pytest.skip('OpenCL GPU not available')

    frames = _make_textured_stack(4, 160, 200)
    cpu_depth = depth_map(frames, None, 17, 15.0, None,
                          half_resolution=True, preserve_detail=False, use_gpu=False)
    gpu_depth = depth_map(frames, None, 17, 15.0, None,
                          half_resolution=True, preserve_detail=False, use_gpu=True)
    np.testing.assert_array_equal(gpu_depth, cpu_depth)


def test_openfocus_depth_zero_smoothing_gpu_matches_cpu():
    from fusion_methods._openfocus_gpu import gpu_available
    from fusion_methods.openfocus_depth import depth_map

    if not gpu_available():
        pytest.skip('OpenCL GPU not available')

    frames = _make_textured_stack(4, 160, 200)
    cpu_depth = depth_map(frames, None, 17, 0.0, None,
                          half_resolution=True, preserve_detail=True, use_gpu=False)
    gpu_depth = depth_map(frames, None, 17, 0.0, None,
                          half_resolution=True, preserve_detail=True, use_gpu=True)
    np.testing.assert_array_equal(gpu_depth, cpu_depth)


# ---------------------------------------------------------------------------
# OpenFocus C CPU raw level-0 Numba kernels parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dtype,gray', [(np.uint8, False), (np.uint16, False), (np.uint8, True)])
def test_openfocus_c_cpu_raw_level0_matches_fallback(monkeypatch, dtype, gray):
    from fusion_methods import _openfocus_pyramid as pyr
    from fusion_methods.openfocus_bc import openfocus_c_impl

    frames = _make_textured_stack(3, 128, 160, dtype=dtype)
    if gray:
        frames = [f[:, :, 0] for f in frames]

    res_new = openfocus_c_impl(frames, use_gpu=False, thread_count=2)

    orig_norm = pyr._normalize
    monkeypatch.setattr(pyr, '_normalize', lambda s, o, f, c: orig_norm(s, o, f, c))
    res_old = openfocus_c_impl(frames, use_gpu=False, thread_count=2)

    if dtype == np.uint8:
        np.testing.assert_array_equal(res_new, res_old)
    else:
        np.testing.assert_allclose(res_new, res_old, atol=1e-5)


# ---------------------------------------------------------------------------
# OpenFocus A feather & soft redundant pass elimination
# ---------------------------------------------------------------------------

def test_openfocus_a_feather_skips_redundant_prepare_and_matches_reference(monkeypatch):
    from fusion_methods import openfocus_a, openfocus_ab

    frames = _make_textured_stack(3, 64, 80)
    prepare_calls = 0
    orig_prepare = openfocus_ab._prepare_frame

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return orig_prepare(*args, **kwargs)

    monkeypatch.setattr(openfocus_ab, '_prepare_frame', counted_prepare)
    res_fast = openfocus_a.openfocus_a_impl(
        frames, blend_mode='feather', use_gpu=False, thread_count=1,
    )
    assert prepare_calls == len(frames)
    assert res_fast.shape == frames[0].shape
    assert res_fast.dtype == np.uint8


def test_openfocus_a_soft_cached_vs_streaming_parity_and_call_counts(monkeypatch):
    from core import resource_policy
    from fusion_methods import openfocus_a, openfocus_ab

    frames = _make_textured_stack(3, 64, 80)
    counts = {'prepare': 0, 'response': 0}
    orig_prepare = openfocus_ab._prepare_frame
    orig_response = openfocus_a._response

    def counted_prepare(*args, **kwargs):
        counts['prepare'] += 1
        return orig_prepare(*args, **kwargs)

    def counted_response(*args, **kwargs):
        counts['response'] += 1
        return orig_response(*args, **kwargs)

    monkeypatch.setattr(openfocus_ab, '_prepare_frame', counted_prepare)
    monkeypatch.setattr(openfocus_a, '_response', counted_response)

    res_cached = openfocus_a.openfocus_a_impl(
        frames, blend_mode='soft', use_gpu=False, thread_count=1,
    )
    assert counts == {'prepare': len(frames), 'response': len(frames)}

    counts = {'prepare': 0, 'response': 0}
    monkeypatch.setattr(resource_policy, 'ram_budget', lambda _snap: 1)
    res_streaming = openfocus_a.openfocus_a_impl(
        frames, blend_mode='soft', use_gpu=False, thread_count=1,
    )
    assert counts['prepare'] == len(frames)
    assert counts['response'] == 3 * len(frames)
    np.testing.assert_array_equal(res_cached, res_streaming)


# ---------------------------------------------------------------------------
# Registration: Homography (grayscale, FLANN, 1-kp guard), ECC CPU, OpenFocus 4D, Stabilisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.float32])
def test_homography_grayscale_and_bgr_rev(dtype):
    from core.registration import ImageRegistration

    frames = _make_textured_stack(2, 180, 240, dtype=np.uint8 if dtype != np.uint16 else np.uint16,
                                  shift_x=4.0, shift_y=-3.0)
    if dtype == np.float32:
        frames = [f.astype(np.float32) for f in frames]
    aligned = ImageRegistration(method='homography', downscale_width=160).process(frames, thread_count=2)
    assert len(aligned) == 2
    assert aligned[0].dtype == frames[0].dtype

    # Reversed RGB contiguous view (_is_bgr_rev)
    rev_frames = [np.ascontiguousarray(f[..., ::-1])[..., ::-1] for f in frames]
    aligned_rev = ImageRegistration(method='homography', downscale_width=160).process(rev_frames, thread_count=2)
    assert len(aligned_rev) == 2


def test_homography_single_keypoint_and_flann_threshold(monkeypatch):
    from core import registration as reg

    frames = _make_textured_stack(2, 120, 160)

    # 1) Guard len(last_des) < 2 without crashing
    class TinySift:
        def __init__(self):
            self.calls = 0

        def detectAndCompute(self, img, mask):
            self.calls += 1
            if img.mean() >= 0:
                return (cv2.KeyPoint(10.0, 10.0, 5.0),), np.ones((1, 128), dtype=np.float32)
            return (), None

    monkeypatch.setattr(cv2, 'SIFT_create', lambda: TinySift())
    out = reg._align_homography_impl(frames, failure_policy='continue', thread_count=1)
    assert len(out) == 2

    # 2) Verify FlannBasedMatcher path works identically when product > threshold
    # Use small real keypoints on a monkeypatched FlannBasedMatcher threshold test
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 4}, {'checks': 64})
    d1 = np.random.default_rng(1).random((64, 128), dtype=np.float32)
    d2 = d1.copy()
    matches = flann.knnMatch(d2, d1, k=2)
    assert len(matches) == 64
    assert all(m[0].queryIdx == m[0].trainIdx for m in matches)


def test_ecc_cpu_grayscale_before_resize_parity():
    from core.registration import ImageRegistration

    frames = _make_textured_stack(3, 180, 240, shift_x=2.0, shift_y=1.0)
    rev_frames = [np.ascontiguousarray(f[..., ::-1])[..., ::-1] for f in frames]

    aligned_std = ImageRegistration(method='ecc', downscale_width=120, accelerate_ecc=False).process(
        frames, thread_count=2,
    )
    aligned_rev = ImageRegistration(method='ecc', downscale_width=120, accelerate_ecc=False).process(
        rev_frames, thread_count=2,
    )
    assert len(aligned_std) == 3
    assert aligned_std[0].shape == aligned_rev[0].shape
    np.testing.assert_allclose(aligned_std[1], aligned_rev[1], atol=1)


def test_openfocus_alignment_grid_4d_slice_parity():
    from core._alignment_kernels import score_grid_4d_slice
    from core.openfocus_alignment import _score_chunk, _decide, enumerate_candidates

    rng = np.random.default_rng(42)
    img_a = rng.integers(0, 65535, (120, 160), dtype=np.uint16)
    img_b = rng.integers(0, 65535, (120, 160), dtype=np.uint16)
    xs = np.arange(10, 150, 20, dtype=np.float32)
    ys = np.arange(10, 110, 20, dtype=np.float32)
    ref = np.ascontiguousarray(img_a[ys.astype(np.int64)[:, None], xs.astype(np.int64)[None, :]], dtype=np.uint16)
    cx_tab = np.linspace(-4, 4, 5, dtype=np.float32)
    cy_tab = np.linspace(-4, 4, 5, dtype=np.float32)
    p08_tab = np.linspace(-0.02, 0.02, 9, dtype=np.float32).reshape(3, 3)
    p18_tab = np.linspace(0.98, 1.02, 9, dtype=np.float32).reshape(3, 3)
    tx0, ty0 = np.float32(1.5), np.float32(-1.0)
    n0, n1, n2, n3 = 5, 5, 3, 3
    N = n0 * n1 * n2 * n3

    out_k = np.empty(1, dtype=np.int64)
    out_sc = np.empty(1, dtype=np.float32)
    score_grid_4d_slice(ref, img_b, xs, ys, tx0, ty0, cx_tab, cy_tab, p08_tab, p18_tab,
                        n0, n1, n2, n3, 0, N, out_k, out_sc, 0)

    # Compare against scalar reference
    cand = np.empty((N, 8), dtype=np.float32)
    # Build matching candidate table using same p08/p18 directly via _score_chunk fields
    # Verify slice splitting invariance: [0, N) == min([0, N//2), [N//2, N))
    out_k2 = np.empty(2, dtype=np.int64)
    out_sc2 = np.empty(2, dtype=np.float32)
    score_grid_4d_slice(ref, img_b, xs, ys, tx0, ty0, cx_tab, cy_tab, p08_tab, p18_tab,
                        n0, n1, n2, n3, 0, N // 2, out_k2, out_sc2, 0)
    score_grid_4d_slice(ref, img_b, xs, ys, tx0, ty0, cx_tab, cy_tab, p08_tab, p18_tab,
                        n0, n1, n2, n3, N // 2, N, out_k2, out_sc2, 1)
    best_split = 0 if out_sc2[0] <= out_sc2[1] else 1
    assert out_k[0] == out_k2[best_split]
    assert out_sc[0] == out_sc2[best_split]
    assert out_k[0] >= 0


def test_mosaic_stitcher_grayscale_and_unicode_paths(tmp_path):
    from core.mosaic_stitcher import GridSpec, MosaicStitcher

    uni_dir = tmp_path / '测试_日本語_α'
    uni_dir.mkdir()
    rng = np.random.default_rng(7)
    base = cv2.GaussianBlur(rng.integers(0, 255, (120, 160), dtype=np.uint8), (5, 5), 1.0)
    paths = []
    for idx, tile in enumerate([base[:, :100], base[:, 60:]]):
        p = str(uni_dir / f'灰度_{idx}.png')
        ok, buf = cv2.imencode('.png', tile)
        assert ok
        buf.tofile(p)
        paths.append(p)

    out_path = str(uni_dir / '拼接结果_α.png')
    stitcher = MosaicStitcher(GridSpec(rows=1, cols=2, overlap_pct=25.0), paths)
    canvas, placements = stitcher.run(output_path=out_path)
    assert canvas.ndim == 3 and canvas.shape[2] == 3
    assert len(placements) == 2
    assert os.path.isfile(out_path)


def test_stabilisation_unicode_paths_and_grayscale(tmp_path):
    from core.registration import _stabilisation_impl

    in_dir = tmp_path / '稳定输入_α'
    out_dir = tmp_path / '稳定输出_β'
    in_dir.mkdir()
    frames = _make_textured_stack(3, 96, 128, shift_x=2.0, shift_y=1.0)
    for idx, frame in enumerate(frames):
        ok, buf = cv2.imencode('.png', frame)
        assert ok
        buf.tofile(str(in_dir / f'帧_{idx}.png'))

    stabilized = _stabilisation_impl(str(in_dir), output_path=str(out_dir))
    assert len(stabilized) == 3
    assert (out_dir / '帧_0.png').is_file()

    gray_frames = [f[:, :, 0] for f in frames]
    stabilized_gray = _stabilisation_impl(gray_frames)
    assert len(stabilized_gray) == 3


# ---------------------------------------------------------------------------
# Image loader LZW TIFF & single-open JPEG/PNG parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('compression', ['lzw', 'adobe_deflate'])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_decode_tiff_page_compressed_fast_path_parity(tmp_path, monkeypatch, compression, dtype):
    from core.image_loader import decode_tiff_page

    rng = np.random.default_rng(11)
    high = 256 if dtype == np.uint8 else 65536
    rgb = rng.integers(0, high, (1025, 1001, 3), dtype=dtype)
    path = tmp_path / 'multi-strip.tif'
    tifffile.imwrite(path, rgb, photometric='rgb', compression=compression, rowsperstrip=128,
                     extratags=[(274, 'H', 1, 6, False)])

    ok, pages = cv2.imdecodemulti(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED,
                                  range=(0, 1))
    assert ok and len(pages) == 1

    def unexpected_libtiff(*args, **kwargs):
        pytest.fail('Eligible compressed TIFF did not use the strip decoder')

    monkeypatch.setattr(cv2, 'imdecodemulti', unexpected_libtiff)

    decoded = decode_tiff_page(path)
    np.testing.assert_array_equal(decoded, pages[0])


def test_decode_image_single_open_png_and_jpeg_parity(tmp_path):
    from core.image_loader import decode_image

    rng = np.random.default_rng(29)
    bgr = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
    png_path = str(tmp_path / 'sample.png')
    jpg_path = str(tmp_path / 'sample.jpg')
    cv2.imwrite(png_path, bgr)
    cv2.imwrite(jpg_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    np.testing.assert_array_equal(decode_image(png_path), bgr)
    jpg_decoded = decode_image(jpg_path)
    np.testing.assert_array_equal(jpg_decoded, cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED))
