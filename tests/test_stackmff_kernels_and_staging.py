"""Bit-exact parity and direct RGB TIFF row-band staging tests for StackMFF."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import tifffile

from core.image_loader import ImageStackLoader
from core.multi_focus_fusion import MultiFocusFusion, is_ai_fusion_available
from core.source_frames import is_direct_rgb_tiff, map_native_rgb_tiffs, read_rgb_tiff_rows
from fusion_methods import _stackmff_kernels as sk
from fusion_methods import neural_common, neural_runtime, stackmff_v2, stackmff_v2_tiled, stackmffv3


def test_rgb_packing_and_luma_kernels_match_reference():
    rng = np.random.default_rng(2026)
    h, w = 73, 131
    rgb16 = rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)
    rgb8 = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    y0, x0, th, tw = 11, 19, 37, 45
    bgr16_view = rgb16[y0:y0 + th, x0:x0 + tw, ::-1]
    for gain in (1.0, 1.35):
        actual = np.empty((th, tw, 3), dtype=np.float32)
        sk.pack_rgb_u16_to_bgr_f32(rgb16, y0, x0, th, tw, actual, gain)
        expected = neural_common.copy_tile_colors(bgr16_view, True, gain)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

        actual_u8_f = np.empty((th, tw, 3), dtype=np.float32)
        sk.pack_rgb_u8_to_bgr_f32(rgb8, y0, x0, th, tw, actual_u8_f, gain)
        expected_u8_f = neural_common.copy_tile_colors(rgb8[y0:y0 + th, x0:x0 + tw, ::-1], True, gain)
        np.testing.assert_array_equal(actual_u8_f.view(np.uint32), expected_u8_f.view(np.uint32))

    # uint8 direct BGR copy
    dst_u8 = np.empty((th, tw, 3), dtype=np.uint8)
    sk.pack_rgb_u8_to_bgr_u8(rgb8, y0, x0, th, tw, dst_u8)
    np.testing.assert_array_equal(dst_u8, rgb8[y0:y0 + th, x0:x0 + tw, ::-1])

    # uint8 RGB -> luma matches cv2.cvtColor(BGR2GRAY) / 255
    dst_luma = np.empty((th, tw), dtype=np.float32)
    sk.rgb_u8_to_luma(rgb8, y0, x0, th, tw, dst_luma)
    ref_luma = np.empty((th, tw), dtype=np.float32)
    neural_common.write_luma_plane(np.ascontiguousarray(rgb8[y0:y0 + th, x0:x0 + tw, ::-1]), ref_luma)
    np.testing.assert_allclose(dst_luma, ref_luma, atol=1.0 / 255.0 + 1e-6)

    # Indexed gather parity
    rows = np.array([h - 1, 0, 5, 5], dtype=np.int32)
    cols = np.array([0, 3, w - 1], dtype=np.int32)
    dst_idx = np.empty((4, 3, 3), dtype=np.float32)
    sk.pack_rgb_u16_to_bgr_f32_indexed(rgb16, rows, cols, dst_idx, 1.0)
    expected_idx = neural_common.copy_tile_colors(rgb16[rows[:, None], cols[None, :], ::-1], True, 1.0)
    np.testing.assert_array_equal(dst_idx.view(np.uint32), expected_idx.view(np.uint32))


def test_antialias_resize_and_softmax_and_probabilities_parity():
    rng = np.random.default_rng(77)
    planes = rng.random((1, 24, 96, 96), dtype=np.float32)
    with patch.object(neural_common, 'load_stackmff_kernels', return_value=None):
        ref_resized = neural_runtime._resize_antialias(planes, 0.75, None)
    act_1 = neural_runtime._run_resize_antialias(planes, 0.75, None, threads=1)
    act_4 = neural_runtime._run_resize_antialias(planes, 0.75, None, threads=4)
    np.testing.assert_array_equal(act_1.view(np.uint32), ref_resized.view(np.uint32))
    np.testing.assert_array_equal(act_4.view(np.uint32), ref_resized.view(np.uint32))

    logits = rng.normal(0.0, 2.0, (1, 16, 128, 128)).astype(np.float32)
    sm1 = neural_runtime._run_softmax_inplace(logits.copy(), threads=1)
    sm4 = neural_runtime._run_softmax_inplace(logits.copy(), threads=4)
    sm_ref = neural_runtime.softmax(logits.copy())
    np.testing.assert_array_equal(sm1.view(np.uint32), sm_ref.view(np.uint32))
    np.testing.assert_array_equal(sm4.view(np.uint32), sm_ref.view(np.uint32))

    # Full probabilities() post-processing (mocking logits) 1 vs 4 threads
    small_logits = rng.normal(0.0, 2.0, (1, 16, 72, 72)).astype(np.float32)
    orig_logits = neural_runtime.logits
    try:
        neural_runtime.logits = lambda inp, *a, **k: (
            logits.copy() if inp.shape[-1] == 96 else small_logits.copy()
        )
        # Note shape 96x96 for both
        logits96 = rng.normal(0.0, 2.0, (1, 16, 96, 96)).astype(np.float32)
        neural_runtime.logits = lambda inp, *a, **k: (
            logits96.copy() if inp.shape[-1] == 96 else small_logits.copy()
        )
        p1 = neural_runtime.probabilities(planes, 'dummy', 'CPU', 1, 0.5)
        p4 = neural_runtime.probabilities(planes, 'dummy', 'CPU', 4, 0.5)
        np.testing.assert_allclose(p4, p1, rtol=1e-7, atol=1e-7)
    finally:
        neural_runtime.logits = orig_logits


def test_v1_v2_v3_reconstruction_kernels_match_numpy_reference():
    rng = np.random.default_rng(314)
    h, w, n = 47, 61, 10

    # V1 tensor_to_bgr with mixed uint16/uint8 + NaN + float32
    frames_u16 = [rng.integers(0, 65536, (h, w, 3), dtype=np.uint16) for _ in range(n)]
    target = rng.uniform(0.0, 255.0, (h, w)).astype(np.float32)
    target[3, 5] = np.nan
    frames_u16[2] = frames_u16[2].astype(np.uint8)
    with patch.object(neural_common, 'load_stackmff_kernels', return_value=None):
        v1_ref = neural_common.tensor_to_bgr(target, frames_u16)
    v1_fast = neural_common.tensor_to_bgr(target, frames_u16)
    np.testing.assert_allclose(v1_fast, v1_ref, rtol=1e-6, atol=1e-5, equal_nan=True)

    # V2 _expected_positions 1 vs 4 threads vs fallback
    logits = rng.normal(0.0, 1.5, (1, n, 128, 128)).astype(np.float32)
    with patch.object(neural_common, 'load_stackmff_kernels', return_value=None):
        ep_ref = stackmff_v2._expected_positions(logits.copy(), None)
    ep_1 = stackmff_v2._expected_positions(logits.copy(), None, threads=1)
    ep_4 = stackmff_v2._expected_positions(logits.copy(), None, threads=4)
    np.testing.assert_allclose(ep_1, ep_ref, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(ep_4, ep_ref, rtol=1e-6, atol=1e-5)

    # V3 _blend_tile soft & hard
    probs = neural_runtime.softmax(rng.normal(0.0, 2.0, (1, n, h, w)).astype(np.float32))[0]
    for mode in ('soft', 'hard'):
        exp_ref, exp_fast = [], []
        with patch.object(neural_common, 'load_stackmff_kernels', return_value=None):
            out_ref = stackmffv3._blend_tile(
                probs, frames_u16, mode, focus_callback=lambda d, _: exp_ref.append(d.copy()))
        out_fast = stackmffv3._blend_tile(
            probs, frames_u16, mode, focus_callback=lambda d, _: exp_fast.append(d.copy()))
        np.testing.assert_allclose(out_fast, out_ref, rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(exp_fast[0], exp_ref[0], rtol=1e-6, atol=1e-5)


def test_read_rgb_tiff_rows_and_invalidation(tmp_path):
    rng = np.random.default_rng(55)
    arr = rng.integers(0, 65536, (101, 139, 3), dtype=np.uint16)
    p = tmp_path / 'direct.tif'
    tifffile.imwrite(str(p), arr, photometric='rgb')
    ok, msg, frames, _ = ImageStackLoader().load_from_filepaths([str(p)])
    assert ok, msg
    mapped = map_native_rgb_tiffs(frames)
    assert is_direct_rgb_tiff(mapped[0])

    buf = np.empty((32, 139, 3), dtype=np.uint16)
    band = read_rgb_tiff_rows(mapped[0], 17, 45, buf)
    np.testing.assert_array_equal(band, arr[17:45])

    stamp = os.stat(p)
    os.utime(p, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000))
    with pytest.raises(OSError, match='Source changed'):
        read_rgb_tiff_rows(mapped[0], 0, 10, buf)


@pytest.mark.skipif(not is_ai_fusion_available(), reason='ONNX Runtime not installed')
@pytest.mark.parametrize('dtype', [np.uint16, np.uint8])
@pytest.mark.parametrize('alg_mode', [
    ('stackmff', 'soft'),
    ('stackmffv2', 'soft'),
    ('stackmffv3', 'soft'),
    ('stackmffv3', 'hard'),
])
def test_end_to_end_tiff_chunked_tiled_matches_unchunked_fallback(tmp_path, dtype, alg_mode):
    import core.multi_focus_fusion as mff_mod

    alg, mode = alg_mode
    rng = np.random.default_rng(900 + (16 if dtype == np.uint16 else 8) + len(alg))
    # Odd non-multiple-of-64 dimensions to exercise both interior chunks and reflect/edge boundary tiles
    h, w = 139, 149
    high = 65536 if dtype == np.uint16 else 256
    paths = []
    for i in range(4):
        arr = rng.integers(0, high, (h, w, 3), dtype=dtype)
        p = tmp_path / f'layer_{dtype.__name__}_{i}.tif'
        tifffile.imwrite(str(p), arr, photometric='rgb')
        paths.append(str(p))

    ok, msg, frames, _ = ImageStackLoader().load_from_filepaths(paths)
    assert ok, msg
    direct_frames = map_native_rgb_tiffs(frames)
    assert all(is_direct_rgb_tiff(f) for f in direct_frames)

    weight_map = {
        'stackmff': 'weights/stackmff_final.onnx',
        'stackmffv2': 'weights/stackmff_v2_final.onnx',
        'stackmffv3': 'weights/stackmff_v3_final.onnx',
    }
    params = {'mode': mode} if alg == 'stackmffv3' else None

    def run(inp):
        fusion = MultiFocusFusion(
            algorithm=alg, use_gpu=False, tile_enabled=True,
            tile_block_size=64, tile_overlap=16, tile_threshold=64,
            exposure_match=False, fusion_parameters=params,
        )
        return fusion.fuse(inp, model_path=weight_map[alg], thread_count=4)

    fast_out = run(direct_frames)
    # Force fallback path (no kernels -> legacy borrow_frame_region / _stage_tile_planes / NumPy)
    with patch.object(neural_common, 'load_stackmff_kernels', return_value=None):
        ref_out = run(direct_frames)

    # Also test K=1 chunking cap
    with patch.object(stackmff_v2_tiled, '_MAX_V2_CHUNK_BYTES', 1), \
         patch.object(mff_mod, '_MAX_NEURAL_CHUNK_BYTES', 1):
        tight_out = run(direct_frames)

    if dtype == np.uint16 and mode != 'hard':
        np.testing.assert_allclose(fast_out, ref_out, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(tight_out, ref_out, rtol=1e-5, atol=1e-4)
    else:
        np.testing.assert_array_equal(fast_out, ref_out)
        np.testing.assert_array_equal(tight_out, ref_out)
