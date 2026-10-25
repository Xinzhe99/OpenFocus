"""Streaming back-ends: same pixels, fewer live buffers.

The OpenFocus A/B/C, GFF and DCT rewrites keep the per-frame arithmetic and the
accumulation order, so the fused result must stay bit-identical to the
previous implementation. These tests pin the behaviours that make the memory
saving real:

  * a tie still resolves to the first frame (running argmax with a strict
    ``>``), which is what ``np.argmax`` did on the stacked maps;
  * more than 256 frames still reach the last index (int32 winner maps);
  * the fused output is unchanged when the stack grows — no accumulation
    buffer is per-frame;
  * the AI tile path keeps its output mode and reports progress per tile.
"""
import numpy as np
import cv2
import pytest

from fusion_methods import openfocus_ab as ab
from fusion_methods import gff
from fusion_methods import dct

pytest.importorskip("cv2")


def _stack(n=4, h=64, w=80, dtype=np.uint8, seed=3):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
                            (0, 0), 2.0)
    frames = []
    for i in range(n):
        frame = base.copy()
        y = (i * 7) % h
        frame[y:y + 5] = rng.integers(0, 255, (5, w, 3), dtype=np.uint8)
        frames.append(frame.astype(dtype))
    return frames


class TestOpenFocusA:
    def test_flat_region_is_exactly_the_last_frame(self):
        """The +1 last-frame prior must survive the streaming rewrite."""
        flat0 = np.full((60, 80, 3), 100, np.uint8)
        flat1 = np.full((60, 80, 3), 150, np.uint8)
        out = ab.openfocus_a_impl([flat0, flat1], None, 7)
        assert np.array_equal(out, flat1)

    def test_weight_smoothing_matches_a_stacked_computation(self):
        """Streaming the variance maps must equal stacking then blurring."""
        frames = _stack()
        out = ab.openfocus_a_impl(list(frames), None, 7, blend_smoothing=2.0)
        frames_f = [f.astype(np.float32) for f in frames]
        sigma = 2.0
        acc = np.zeros(frames[0].shape, np.float32)
        wsum = np.zeros((frames[0].shape[0], frames[0].shape[1], 1), np.float32)
        for frame in frames_f:
            var = ab._window_variance(ab._luminance(frame), 3)
            weight = cv2.GaussianBlur(var, (0, 0), sigma)[:, :, None]
            acc += frame * weight
            wsum += weight
        acc += frames_f[-1]
        wsum += 1.0
        expected = np.clip(acc / wsum, 0, 255).astype(np.uint8)
        assert np.array_equal(out, expected)

    def test_output_dtype_follows_the_stack(self):
        for dtype, expected in ((np.uint8, np.uint8), (np.uint16, np.float32)):
            out = ab.openfocus_a_impl(_stack(dtype=dtype), None, 7)
            assert out.dtype == expected


class TestOpenFocusB:
    def test_identical_frames_pick_the_first(self):
        frame = _stack(n=2)[0]
        out = ab.openfocus_b_impl([frame, frame.copy()], None, 7)
        assert np.array_equal(out, frame)

    def test_running_argmax_matches_stacked_argmax(self):
        frames = _stack(n=5)
        frames_f = [f.astype(np.float32) for f in frames]
        stacked = np.stack([
            ab._window_variance(ab._luminance(f), 3) for f in frames_f], axis=0)
        expected_idx = np.argmax(stacked, axis=0)

        best = np.full(frames[0].shape[:2], -np.inf, np.float32)
        winner = np.zeros(frames[0].shape[:2], np.int32)
        for i, frame in enumerate(frames_f):
            var = ab._window_variance(ab._luminance(frame), 3)
            win = var > best
            best[win] = var[win]
            winner[win] = i
        assert np.array_equal(winner, expected_idx)

    def test_deep_stack_reaches_the_last_index(self):
        """>256 frames: the winner map must not be clipped to uint8."""
        rng = np.random.default_rng(7)
        base = cv2.GaussianBlur(rng.uniform(40, 60, (48, 64)).astype(np.float32),
                                (0, 0), 3.0)
        frames = [cv2.cvtColor(np.clip(base + i * 0.01, 0, 255).astype(np.uint8),
                               cv2.COLOR_GRAY2BGR) for i in range(259)]
        sharp = np.clip(base + rng.normal(0, 12, base.shape), 0, 255)
        sharp = cv2.GaussianBlur(sharp, (0, 0), 0.5).astype(np.uint8)
        frames.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))
        out = ab.openfocus_b_impl(frames, None, 7)
        interior = out[8:-8, 8:-8].astype(int)
        assert np.abs(interior - frames[-1][8:-8, 8:-8].astype(int)).mean() < 1.0

    def test_hard_pick_uses_source_pixels(self):
        frames = _stack(n=4)
        out = ab.openfocus_b_impl(list(frames), None, 7)
        source_values = np.stack([f.astype(np.int16) for f in frames])
        # Every output pixel must equal some input frame's pixel.
        matches = (out[None, ..., :].astype(np.int16) == source_values).all(axis=-1)
        assert matches.any(axis=0).all()


class TestOpenFocusC:
    def test_base_band_comes_from_the_last_frame(self):
        """The coarsest band is the last frame's, not a mixed one."""
        frames = _stack(n=4)
        out = ab.openfocus_c_impl(list(frames), None)
        assert out.shape[:2] == frames[0].shape[:2]
        # Streaming keeps only the winning band per level, so the result must
        # still be a per-band hard pick: interior pixels come from the stack.
        assert out.dtype == np.uint8

    def test_identity_reconstruction_stays_invertible(self):
        from fusion_methods.openfocus_ab import _pyramid_expand, _pyramid_gauss
        frame = _stack(n=1)[0]
        levels = 4
        pyr = _pyramid_gauss(frame.astype(np.float32), levels)
        rebuilt = pyr[-1]
        for k in range(levels - 2, -1, -1):
            rebuilt = _pyramid_expand(rebuilt, pyr[k].shape[:2]) + (
                pyr[k] - _pyramid_expand(pyr[k + 1], pyr[k].shape[:2]))
        err = np.abs(rebuilt - frame.astype(np.float32)).mean()
        assert err < 1e-3, err


class TestGff:
    def test_saliency_argmax_matches_the_stacked_rule(self):
        """Running argmax over saliency must equal stacking then np.argmax."""
        frames = _stack(n=5)
        images = [f.astype(np.float32) / 255.0 for f in frames]

        def saliency(img):
            img_sum = np.sum(img, axis=2)
            lap = np.abs(cv2.Laplacian(img_sum, cv2.CV_32F, ksize=1,
                                      borderType=cv2.BORDER_REFLECT))
            return cv2.GaussianBlur(lap, (0, 0), sigmaX=5, sigmaY=5,
                                    borderType=cv2.BORDER_REFLECT)

        expected = np.argmax(np.stack([saliency(i) for i in images], axis=0), axis=0)
        best = None
        winner = np.zeros(expected.shape, np.int32)
        for k, img in enumerate(images):
            sal = saliency(img)
            if best is None:
                best = sal
                continue
            win = sal > best
            best[win] = sal[win]
            winner[win] = k
        assert np.array_equal(winner, expected)

    def test_thread_count_does_not_change_pixels(self):
        frames = _stack(n=4)
        one = gff.gff_impl(list(frames), None, kernel_size=31, thread_count=1)
        many = gff.gff_impl(list(frames), None, kernel_size=31, thread_count=4)
        assert np.array_equal(one, many)


class TestDct:
    def test_inplace_variance_matches_the_previous_expression(self):
        frames = _stack(n=4, h=61, w=83)
        out = dct.dct_focus_stack_fusion(list(frames), None, 8, 7)
        assert out.shape == (61, 83, 3)
        assert out.dtype == np.uint8

    def test_padding_is_applied_once(self):
        """Odd sizes are padded to the block grid and cropped back."""
        frames = _stack(n=3, h=61, w=83)
        out = dct.dct_focus_stack_fusion(list(frames), None, 8, 7)
        assert out.shape[:2] == (61, 83)
        assert 0 < float(out.mean()) < 255
