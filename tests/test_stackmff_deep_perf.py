"""Regression, bit-exactness, fallback, and concurrency tests for StackMFF pipeline optimizations."""

from __future__ import annotations

from concurrent.futures import CancelledError
import threading
import time
import weakref
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from core.multi_focus_fusion import MultiFocusFusion, is_ai_fusion_available
from core.registration import RegistrationCancelled
from fusion_methods import _stackmff_kernels as sk
from fusion_methods import neural_common, neural_runtime, stackmff_v1, stackmff_v2, stackmffv3


def test_v2_weighted_sum_layers_l1_block_bit_exact():
    rng = np.random.default_rng(101)
    for n in (1, 4, 37, 150):
        for total in (1, 1023, 1025, 4099):
            probs = rng.random((n, total), dtype=np.float32)
            out = np.empty(total, dtype=np.float32)
            sk.v2_weighted_sum_layers(probs, 0, total, out)
            weights = np.arange(1, n + 1, dtype=np.float32)[:, None]
            expected = (probs * weights).sum(axis=0)
            if total > 1 or n <= 4:
                np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))
            else:
                np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_bgr_and_rgb_u8_luma_row_kernels_bit_exact():
    rng = np.random.default_rng(102)
    h, w = 73, 131
    bgr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])

    ref = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / np.float32(255.0)
    out_bgr = np.empty((h, w), dtype=np.float32)
    out_rgb = np.empty((h, w), dtype=np.float32)
    sk.bgr_u8_to_luma_rows(bgr, out_bgr, 0, h)
    sk.rgb_u8_to_luma_rows(rgb, out_rgb, 0, h)

    np.testing.assert_array_equal(out_bgr.view(np.uint32), ref.view(np.uint32))
    np.testing.assert_array_equal(out_rgb.view(np.uint32), ref.view(np.uint32))


def test_normalize_probs_rows_matches_numpy():
    rng = np.random.default_rng(103)
    for b, n in ((2, 4), (1, 16), (1, 120)):
        probs = rng.uniform(0.01, 1.0, (b, n, 41, 67)).astype(np.float32)
        expected = probs / probs.sum(axis=1, keepdims=True)
        actual = probs.copy()
        sk.normalize_probs_rows(actual, 0, 41)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        if n <= 16:
            np.testing.assert_array_equal(actual, expected)


def test_v1_weighted_luma_rows_matches_numpy():
    rng = np.random.default_rng(104)
    for n in (4, 120):
        probs = rng.uniform(0.0, 1.0, (1, n, 35, 53)).astype(np.float32)
        probs /= probs.sum(axis=1, keepdims=True)
        planes = rng.uniform(0.0, 1.0, (1, n, 35, 53)).astype(np.float32)
        out = np.empty((35, 53), dtype=np.float32)
        sk.v1_weighted_luma_rows(probs, planes, out, 0, 35)
        expected = np.clip((probs * planes).sum(axis=1)[0], 0.0, 1.0) * np.float32(255.0)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-4)
        if n <= 8:
            np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))


def test_v1_nearest_and_ycbcr_row_kernels_with_bgr_rev_and_nan():
    rng = np.random.default_rng(105)
    h, w = 64, 71
    rgb16 = rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)
    rgb8 = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    bgr_u16 = rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)
    bgr_u8 = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    frames = [rgb16[:, :, ::-1], rgb8[:, :, ::-1], bgr_u16, bgr_u8]
    target = rng.uniform(0.0, 255.0, (h, w)).astype(np.float32)
    target[4, 9] = np.nan

    with patch.object(neural_common, "load_stackmff_kernels", return_value=None):
        ref = neural_common.tensor_to_bgr(target, frames, threads=1)
    fast_1 = neural_common.tensor_to_bgr(target, frames, threads=1)
    fast_4 = neural_common.tensor_to_bgr(target, frames, threads=4)

    np.testing.assert_allclose(fast_1, ref, rtol=1e-6, atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(fast_4, ref, rtol=1e-6, atol=1e-5, equal_nan=True)

    # Pure uint8 + f32 exactness check (including NaN preservation in YCbCr -> BGR)
    u8_frames = [rgb8[:, :, ::-1], bgr_u8]
    with patch.object(neural_common, "load_stackmff_kernels", return_value=None):
        ref_u8 = neural_common.tensor_to_bgr(target, u8_frames, threads=1)
    fast_u8 = neural_common.tensor_to_bgr(target, u8_frames, threads=4)
    np.testing.assert_array_equal(fast_u8, ref_u8)


@pytest.mark.parametrize('threads', [1, 2])
def test_v1_fallback_releases_native_color_output(monkeypatch, threads):
    luma = np.full((256, 256), 64, np.float32)
    image = np.zeros(luma.shape + (3,), np.uint8)
    destinations = []
    numpy_to_bgr = neural_common._ycbcr_to_bgr

    def refuse_color_kernel(_luma, _cb, _cr, destination, *_rows):
        destinations.append(weakref.ref(destination))
        raise neural_common.StackMFFKernelUnavailable()

    def convert_with_available_memory(*args):
        assert destinations and all(ref() is None for ref in destinations)
        return numpy_to_bgr(*args)

    monkeypatch.setattr(neural_common, 'load_stackmff_kernels', lambda: sk)
    monkeypatch.setattr(sk, 'v1_ycbcr_to_bgr_rows', refuse_color_kernel)
    monkeypatch.setattr(neural_common, '_ycbcr_to_bgr', convert_with_available_memory)
    fused = neural_common.tensor_to_bgr(luma, [image, image], threads=threads)
    np.testing.assert_array_equal(fused, np.full(image.shape, 64, np.float32))


def test_v3_argmax_rows_matches_numpy_ties_inf_and_nan():
    rng = np.random.default_rng(106)
    n, hp, wp = 12, 51, 64
    h, w = 49, 59  # strided slice inside (n, hp, wp)
    backing = rng.uniform(-1.0, 1.0, (1, n, hp, wp)).astype(np.float32)
    backing[0, 3, 2, 7] = backing[0, 8, 2, 7] = 0.5  # exact tie: first index wins
    backing[0, 5, 4, 1] = np.inf
    backing[0, 9, 4, 1] = np.inf
    backing[0, 1, 6, 3] = np.nan
    backing[0, 4, 6, 3] = 999.0  # NaN matches np.argmax first-NaN-wins semantics

    probs = backing[0, :, :h, :w]
    assert not probs.flags.c_contiguous
    sel = np.empty((h, w), dtype=np.intp)
    counts = np.zeros(n, dtype=np.int64)
    sk.v3_argmax_rows(probs, sel, counts, 0, h)

    expected_sel = np.argmax(probs, axis=0)
    expected_counts = np.bincount(expected_sel.ravel(), minlength=n)
    np.testing.assert_array_equal(sel, expected_sel)
    np.testing.assert_array_equal(counts, expected_counts)


@pytest.mark.parametrize("mode", ["soft", "hard"])
def test_v3_blend_tile_bgr_rev_and_resident_matches_fallback(mode):
    rng = np.random.default_rng(107)
    n, h, w = 12, 83, 67
    probs_pad = rng.random((n, h, 128), dtype=np.float32)
    probs = probs_pad[:, :, :w]
    probs /= probs.sum(axis=0, keepdims=True)

    # Mix contiguous BGR and bgr_rev RGB16/RGB8 views (typical direct TIFF memmap views)
    frames_u8 = [
        np.ascontiguousarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))[:, :, ::-1]
        if i % 2 else rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        for i in range(n)
    ]
    exp_ref, exp_fast = [], []
    with patch.object(neural_common, "load_stackmff_kernels", return_value=None):
        out_ref = stackmffv3._blend_tile(
            probs, frames_u8, mode, focus_callback=lambda d, _: exp_ref.append(d.copy()), threads=1
        )
    out_fast = stackmffv3._blend_tile(
        probs, frames_u8, mode, focus_callback=lambda d, _: exp_fast.append(d.copy()), threads=4
    )
    np.testing.assert_array_equal(out_fast, out_ref)
    np.testing.assert_array_equal(exp_fast[0], exp_ref[0])

    # Uint16 bgr_rev check
    frames_u16 = [
        rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)[:, :, ::-1]
        for _ in range(n)
    ]
    with patch.object(neural_common, "load_stackmff_kernels", return_value=None):
        out_ref16 = stackmffv3._blend_tile(probs, frames_u16, mode, threads=1)
    out_fast16 = stackmffv3._blend_tile(probs, frames_u16, mode, threads=4)
    if mode == "hard":
        np.testing.assert_array_equal(out_fast16, out_ref16)
    else:
        np.testing.assert_allclose(out_fast16, out_ref16, rtol=1e-5, atol=1e-4)


def test_v2_sample_bgr_and_bgr_rev_scatter_parity():
    rng = np.random.default_rng(108)
    h, w, n = 135, 79, 6
    idx = rng.integers(0, n, (h, w), dtype=np.int64)
    frames_u16_bgr = [rng.integers(0, 65536, (h, w, 3), dtype=np.uint16) for _ in range(n)]
    frames_u16_rev = [np.ascontiguousarray(f[:, :, ::-1])[:, :, ::-1] for f in frames_u16_bgr]
    frames_u8_bgr = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]
    frames_u8_rev = [np.ascontiguousarray(f[:, :, ::-1])[:, :, ::-1] for f in frames_u8_bgr]
    gains = [1.0, 0.85, 1.15, 0.95, 1.05, 1.0]

    for frames, out_dt, norm_exp, g, rnd in [
        (frames_u16_bgr, np.float32, True, gains, False),
        (frames_u16_rev, np.float32, True, gains, False),
        (frames_u8_bgr, np.float32, True, gains, False),
        (frames_u8_rev, np.float32, True, gains, False),
        (frames_u8_bgr, np.uint8, False, None, False),
        (frames_u8_rev, np.uint8, False, None, False),
        (frames_u16_bgr, np.uint8, False, None, True),
    ]:
        with patch.object(neural_common, "load_stackmff_kernels", return_value=None):
            expected = stackmff_v2._sample(
                frames, idx, dtype=out_dt, round_uint8=rnd, gains=g, normalize_for_exposure=norm_exp
            )
        actual = stackmff_v2._sample(
            frames, idx, dtype=out_dt, round_uint8=rnd, gains=g, normalize_for_exposure=norm_exp
        )
        np.testing.assert_array_equal(actual, expected)


def test_copy_tile_colors_bgr_rev_bit_exact():
    rng = np.random.default_rng(109)
    rgb16 = rng.integers(0, 65536, (45, 67, 3), dtype=np.uint16)
    rgb8 = rng.integers(0, 256, (45, 67, 3), dtype=np.uint8)
    for pixels, norm, gain in [
        (rgb16[5:40, 7:60, ::-1], True, 1.0),
        (rgb16[5:40, 7:60, ::-1], True, 1.25),
        (rgb8[5:40, 7:60, ::-1], True, 1.15),
        (rgb8[5:40, 7:60, ::-1], False, 1.0),
    ]:
        with patch.object(neural_common, "load_stackmff_kernels", return_value=None):
            ref = neural_common.copy_tile_colors(pixels, norm, gain)
        act = neural_common.copy_tile_colors(pixels, norm, gain)
        if act.dtype == np.float32:
            np.testing.assert_array_equal(act.view(np.uint32), ref.view(np.uint32))
        else:
            np.testing.assert_array_equal(act, ref)


@pytest.mark.parametrize("shape", [(13, 17), (64, 64), (65, 97), (193, 193)])
def test_stage_planes_and_stage_gray_fast_paths_and_lazy_release(shape):
    rng = np.random.default_rng(110)
    h, w = shape
    rgb16 = rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)
    rgb8 = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    bgr_f32 = rng.uniform(0.0, 255.0, (h, w, 3)).astype(np.float32)

    mixed_u16 = [rgb16[:, :, ::-1], rgb8[:, :, ::-1], bgr_f32, rng.integers(0, 65536, (h, w, 3), dtype=np.uint16)]
    pure_u8 = [rgb8[:, :, ::-1], rng.integers(0, 256, (h, w, 3), dtype=np.uint8)]

    for frames in (mixed_u16, pure_u8):
        with patch.object(neural_common, "load_stackmff_kernels", return_value=None):
            ref_sp = neural_common.stage_planes(frames, 64, threads=1)
            ref_sg = stackmffv3._stage_gray(frames, None, threads=1)
        act_sp_1 = neural_common.stage_planes(frames, 64, threads=1)
        act_sp_4 = neural_common.stage_planes(frames, 64, threads=4)
        act_sg_1 = stackmffv3._stage_gray(frames, None, threads=1)
        act_sg_4 = stackmffv3._stage_gray(frames, None, threads=4)
        np.testing.assert_array_equal(act_sp_1.view(np.uint32), ref_sp.view(np.uint32))
        np.testing.assert_array_equal(act_sp_4.view(np.uint32), ref_sp.view(np.uint32))
        np.testing.assert_array_equal(act_sg_1.view(np.uint32), ref_sg.view(np.uint32))
        np.testing.assert_array_equal(act_sg_4.view(np.uint32), ref_sg.view(np.uint32))

    # Lazy _TileCrop streaming release check with multi-thread row-band mode
    refs = []
    orig_resolve = neural_common.copy_tile_colors

    def tracked(pixels, normalize_stack, gain):
        if refs:
            assert refs[-1]() is None
        crop = orig_resolve(pixels, normalize_stack, gain)
        refs.append(weakref.ref(crop))
        return crop

    with patch.object(neural_common, "copy_tile_colors", tracked):
        sources = neural_common.tile_sources(pure_u8, (0, 0, w, h))
        staged = neural_common.stage_planes(sources, 64, threads=4)
    assert len(refs) == len(sources)
    assert refs[-1]() is None
    np.testing.assert_array_equal(staged, neural_common.stage_planes(pure_u8, 64, threads=1))


def test_stackmff_staging_rejects_mismatched_source_geometry():
    frames = [np.zeros((height, 64, 3), np.uint8) for height in (64, 63)]
    with pytest.raises(ValueError, match='same dimensions'):
        neural_common.stage_planes(frames, 64)
    with pytest.raises(ValueError, match='same dimensions'):
        stackmffv3._stage_gray(frames, None)


@pytest.mark.parametrize('variant,dtype', [
    ('v1/v2', np.float32), ('v1/v2', np.uint16), ('v3', np.uint16),
])
def test_parallel_luma_staging_keeps_row_bands_independent(monkeypatch, variant, dtype):
    image = np.zeros((513, 513, 3), dtype=dtype)
    image[257:] = 192 if dtype == np.float32 else 49152
    frames = [image, image]

    def stage(threads):
        if variant == 'v3':
            return stackmffv3._stage_gray(frames, None, threads=threads)
        return neural_common.stage_planes(frames, 64, threads=threads)

    expected = stage(1)
    convert = cv2.cvtColor
    first_pair = iter(range(2))
    both_ready = threading.Barrier(2, timeout=10)

    def convert_together(src, *args, **kwargs):
        first = next(first_pair, None) is not None
        if first:
            # Both row ranges have packed different pixels. Neither can reuse
            # its buffer until both conversions finish reading it.
            both_ready.wait()
        result = convert(src, *args, **kwargs)
        if first:
            both_ready.wait()
        return result

    monkeypatch.setattr(cv2, 'cvtColor', convert_together)
    np.testing.assert_array_equal(stage(2), expected)


def test_softmax_l2_blocking_and_monkeypatch():
    rng = np.random.default_rng(111)
    # Force strip_rows < h on big height & multiple layers
    x = rng.normal(0.0, 3.0, (1, 128, 80, 64)).astype(np.float32)
    ref = neural_runtime.softmax(x.copy())
    s1 = neural_runtime._run_softmax_inplace(x.copy(), threads=1)
    s4 = neural_runtime._run_softmax_inplace(x.copy(), threads=4)
    np.testing.assert_array_equal(s1.view(np.uint32), ref.view(np.uint32))
    np.testing.assert_array_equal(s4.view(np.uint32), ref.view(np.uint32))

    seen = []
    def custom(arr):
        seen.append(arr.shape)
        return arr

    with patch.object(neural_runtime, "softmax_inplace", custom):
        neural_runtime._run_softmax_inplace(x.copy(), threads=4)
    assert seen == [x.shape]


def test_run_neural_tasks_reentrancy_and_cancellation_joins_all_workers():
    # Nested call executes cleanly without deadlocking pool workers
    outer = neural_runtime.run_neural_tasks(
        4, range(4),
        lambda i: sum(neural_runtime.run_neural_tasks(4, range(3), lambda j: i * 10 + j)),
    )
    assert outer == [0 + 1 + 2, 10 + 11 + 12, 20 + 21 + 22, 30 + 31 + 32]

    # When item 0 fails/cancels, peer futures must ALL finish before exception propagates
    in_flight = 0
    finished = 0
    lock = threading.Lock()
    started = threading.Event()

    def task(idx):
        nonlocal in_flight, finished
        if idx == 0:
            started.wait(2.0)
            raise RegistrationCancelled()
        with lock:
            in_flight += 1
            if in_flight >= 1:
                started.set()
        time.sleep(0.05)
        with lock:
            in_flight -= 1
            finished += 1

    with pytest.raises(RegistrationCancelled):
        neural_runtime.run_neural_tasks(4, range(4), task)
    assert in_flight == 0
    assert finished == 3


def test_v3_blend_tile_recovers_and_zeroes_on_kernel_unavailable():
    rng = np.random.default_rng(112)
    n, h, w = 8, 256, 256  # >= 262144 total elements triggers resident fast path
    frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]
    probs = rng.random((n, h, w), dtype=np.float32)
    probs /= probs.sum(axis=0, keepdims=True)

    with patch.object(neural_common, "load_stackmff_kernels", return_value=None):
        ref = stackmffv3._blend_tile(probs, frames, "soft", threads=1)

    class _FakeKernels:
        calls = 0

        @staticmethod
        def v3_accumulate_soft_u8(src, p, out, exp, idx_f, has_exp):
            _FakeKernels.calls += 1
            out[:] += 123.0  # Simulate partial accumulation before unavailable
            raise neural_common.StackMFFKernelUnavailable()

    with patch.object(neural_common, "load_stackmff_kernels", return_value=_FakeKernels):
        recovered = stackmffv3._blend_tile(probs, frames, "soft", threads=4)
    assert _FakeKernels.calls >= 1
    np.testing.assert_allclose(recovered, ref, rtol=1e-6, atol=1e-5)


def test_v3_soft_fallback_keeps_completed_bands(monkeypatch):
    height = 257
    frames = [np.full((height, 1, 3), color, np.uint8) for color in (40, 120)]
    probs = np.full((2, height, 1), 0.5, np.float32)
    native = sk.v3_accumulate_soft_u8

    def reject_tail_specialization(src, prob, out, expected, index, has_expected):
        # The unused focus view is strided in full bands and contiguous in
        # the final one-pixel band, requiring a second native specialization.
        if expected.flags.c_contiguous:
            raise neural_common.StackMFFKernelUnavailable()
        return native(src, prob, out, expected, index, has_expected)

    monkeypatch.setattr(neural_common, 'load_stackmff_kernels', lambda: sk)
    monkeypatch.setattr(sk, 'v3_accumulate_soft_u8', reject_tail_specialization)
    fused = stackmffv3._blend_tile(probs, frames, 'soft')
    np.testing.assert_array_equal(fused, np.full((height, 1, 3), 80, np.float32))


def test_v3_parallel_blend_stops_between_sources(monkeypatch):
    height = width = 256
    frames = [np.full((height, width, 3), color, np.uint8) for color in (40, 120, 200)]
    probs = np.full((len(frames), height, width), 1 / len(frames), np.float32)
    cancelled = threading.Event()
    started_sources = []
    native = sk.v3_accumulate_soft_u8

    def cancel_in_first_source(src, *args):
        started_sources.append(int(src[0, 0, 0]))
        cancelled.set()
        return native(src, *args)

    monkeypatch.setattr(neural_common, 'load_stackmff_kernels', lambda: sk)
    monkeypatch.setattr(sk, 'v3_accumulate_soft_u8', cancel_in_first_source)
    with pytest.raises(RegistrationCancelled):
        stackmffv3._blend_tile(probs, frames, 'soft', cancelled.is_set, threads=2)
    assert set(started_sources) == {40}


@pytest.mark.skipif(not is_ai_fusion_available(), reason="ONNX Runtime not installed")
@pytest.mark.parametrize("alg,params", [
    ("stackmff", None),
    ("stackmffv2", None),
    ("stackmffv3", {"mode": "soft"}),
    ("stackmffv3", {"mode": "hard"}),
])
def test_stackmff_cpu_fallback_multi_thread(alg, params):
    rng = np.random.default_rng(113)
    h, w = 73, 89
    frames = [
        rng.integers(0, 65536 if i == 1 else 256, (h, w, 3), dtype=np.uint16 if i == 1 else np.uint8)
        for i in range(3)
    ]
    mf = MultiFocusFusion(
        algorithm=alg, use_gpu=False, tile_enabled=False,
        exposure_match=False, fusion_parameters=params,
    )
    out = mf.fuse(frames, thread_count=4)
    assert out.shape == (h, w, 3)
    assert out.dtype == np.float32
    assert float(out.max()) > 1.0
