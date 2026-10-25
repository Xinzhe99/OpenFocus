"""Q0 comparisons to the arithmetic at 4659b7bc, independent of scheduling.

References deliberately retain stacked maps and original expression order.
They exercise actual output pixels, including 16-bit low bits and edge padding.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref

import cv2
import numpy as np
import pytest
from scipy.ndimage import median_filter

from core.registration import RegistrationCancelled
from fusion_methods import dct, gff, gfg_fgf
from fusion_methods import openfocus_ab as ab
from utils.image_utils import normalize_fuse_input, quantize_fuse_output


def _stack(kind="bgr", n=4):
    rng = np.random.default_rng(946)
    frames = []
    for _ in range(n):
        dtype = np.uint16 if kind == "uint16" else np.uint8
        high = 65536 if dtype == np.uint16 else 256
        frame = rng.integers(0, high, (66, 78, 3), dtype=dtype)[::2, ::-2]
        if kind == "gray":
            frame = frame[:, :, 1]
        elif kind == "alpha":
            alpha = rng.integers(0, 256, frame.shape[:2], dtype=np.uint8)
            frame = np.concatenate((frame, alpha[:, :, None]), axis=2)[:, ::-1]
        frames.append(frame)
    return frames


def _color(frame):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame[:, :, :3]


def _dtype(frames):
    return np.uint8 if frames[0].dtype == np.uint8 else np.float32


def _ab_reference(frames, method, smoothing):
    """Original stacked A/B/C formulas; no production streaming decisions."""
    floats = [f.astype(np.float32) for f in frames]
    if method in ("a", "b"):
        variances = np.stack([
            ab._window_variance(ab._luminance(f), 3) for f in floats])
        stack = np.stack([f[:, :, None] if f.ndim == 2 else f for f in floats])
        if method == "a":
            weights = np.stack([cv2.GaussianBlur(v, (0, 0), smoothing)
                                if smoothing > 1e-3 else v for v in variances])
            acc = np.zeros(stack.shape[1:], np.float32)
            total = np.zeros((*stack.shape[1:3], 1), np.float32)
            for frame, weight in zip(stack, weights):
                acc += frame * weight[:, :, None]
                total += weight[:, :, None]
            acc += stack[-1]
            total += 1.0
            result = acc / total
        else:
            depth = np.argmax(variances, axis=0).astype(np.float32)
            depth = cv2.medianBlur(depth, 3)
            if smoothing > 0.05:
                depth = cv2.GaussianBlur(depth, (0, 0), smoothing)
            index = np.clip(np.rint(depth), 0, len(frames) - 1).astype(np.int32)
            result = np.take_along_axis(stack, index[None, :, :, None], axis=0)[0]
        if frames[0].ndim == 2:
            result = result[:, :, 0]
    else:
        h, w = frames[0].shape[:2]
        usable = max(2, min(7, int(np.log2(max(2, min(h, w)))) - 1))
        pyramids = [ab._pyramid_gauss(f, usable) for f in floats]
        selected = []
        for level in range(usable - 1):
            bands = np.stack([p[level] - ab._pyramid_expand(p[level + 1], p[level].shape[:2])
                              for p in pyramids]).astype(np.float32)
            luma = ((bands * np.array([0.114, 0.587, 0.299], np.float32)).sum(axis=3)
                    if bands.ndim == 4 else bands)
            weights = np.stack([cv2.GaussianBlur(np.abs(v), (0, 0), smoothing)
                                if smoothing > 1e-3 else np.abs(v) for v in luma])
            index = weights.argmax(axis=0)
            gather = index[None, :, :, None] if bands.ndim == 4 else index[None]
            selected.append(np.take_along_axis(bands, gather, axis=0)[0])
        selected.append(pyramids[-1][-1].copy())
        result = selected[-1]
        for level in range(usable - 2, -1, -1):
            result = ab._pyramid_expand(result, selected[level].shape[:2]) + selected[level]
    return np.clip(result, 0, 255).astype(_dtype(frames))


def _gff_reference(frames):
    images = [_color(f).astype(np.float32) / 255.0 for f in frames]
    bases = [cv2.blur(f, (7, 7), borderType=cv2.BORDER_REFLECT) for f in images]
    details = [f - b for f, b in zip(images, bases)]
    saliencies = []
    for image in images:
        lap = np.abs(cv2.Laplacian(np.sum(image, axis=2), cv2.CV_32F,
                                   ksize=1, borderType=cv2.BORDER_REFLECT))
        saliencies.append(cv2.GaussianBlur(lap, (0, 0), sigmaX=5, sigmaY=5,
                                           borderType=cv2.BORDER_REFLECT))
    winner = np.argmax(np.stack(saliencies), axis=0)
    nums = [np.zeros_like(images[0]), np.zeros_like(images[0])]
    dens = [np.zeros(winner.shape, np.float32), np.zeros(winner.shape, np.float32)]
    for i, image in enumerate(images):
        guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (winner == i).astype(np.float32)
        ip, ii = guide * mask, guide * guide
        for layer, (radius, eps) in enumerate(((45, 0.3), (7, 10e-6))):
            size = (2 * radius + 1, 2 * radius + 1)
            mean_i = cv2.boxFilter(guide, cv2.CV_32F, size)
            mean_p = cv2.boxFilter(mask, cv2.CV_32F, size)
            cov = cv2.boxFilter(ip, cv2.CV_32F, size) - mean_i * mean_p
            var = cv2.boxFilter(ii, cv2.CV_32F, size) - mean_i * mean_i
            a = cov / (var + eps)
            b = mean_p - a * mean_i
            weight = (cv2.boxFilter(a, cv2.CV_32F, size) * guide
                      + cv2.boxFilter(b, cv2.CV_32F, size))
            nums[layer] += (bases if layer == 0 else details)[i] * weight[:, :, None]
            dens[layer] += weight
    for den in dens:
        den[den < 1e-6] = 1e-6
    result = nums[0] / dens[0][:, :, None] + nums[1] / dens[1][:, :, None]
    return np.clip(result * 255, 0, 255).astype(_dtype(frames))


def _dct_reference(frames):
    h, w = frames[0].shape[:2]
    ph, pw = (h + 7) // 8 * 8, (w + 7) // 8 * 8
    padded = [cv2.copyMakeBorder(_color(f), 0, ph - h, 0, pw - w,
                                 cv2.BORDER_REPLICATE) for f in frames]
    variances = []
    for frame in padded:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean_sq = cv2.resize(gray ** 2, (pw // 8, ph // 8), interpolation=cv2.INTER_AREA)
        mean = cv2.resize(gray, (pw // 8, ph // 8), interpolation=cv2.INTER_AREA)
        variances.append(mean_sq - mean ** 2)
    index = np.argmax(np.stack(variances), axis=0).astype(np.int32)
    index = median_filter(median_filter(index, size=7, mode="nearest"), size=7, mode="nearest")
    result = np.zeros((ph, pw, 3), _dtype(frames))
    for i in np.unique(index):
        mask = cv2.resize((index == i).astype(np.uint8), (pw, ph),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        result[mask] = padded[i][mask]
    return result[:h, :w].copy()


def _gfg_reference(frames):
    images = [_color(f).astype(np.float32) / 255.0 for f in frames]
    channel = int(np.argmax(images[0].sum(axis=(0, 1))))
    guides = [f[:, :, channel] for f in images]
    activities = [gfg_fgf.gfg_focus_map(g, 7, 5, 0.3) for g in guides]
    index = gfg_fgf.gfg_decision_index(activities)
    weights = [gfg_fgf._run_guided_filter(g, (index == i).astype(np.float32), 5, 0.3)
               if (index == i).any() else None for i, g in enumerate(guides)]
    total = np.zeros(index.shape, np.float32)
    numerator = np.zeros_like(images[0])
    for image, weight in zip(images, weights):
        if weight is not None:
            total += weight
            numerator += image * weight[:, :, None]
    nonzero = total > 1e-6
    out = np.zeros(images[0].shape, _dtype(frames))
    for channel in range(3):
        data = numerator[:, :, channel]
        np.divide(data, total, out=data, where=nonzero)
        if not nonzero.all():
            for i in np.unique(index[~nonzero]):
                mask = (~nonzero) & (index == i)
                data[mask] = images[i][:, :, channel][mask]
        np.clip(data * 255.0, 0, 255, out=data)
        out[:, :, channel] = data
    return out


@pytest.mark.parametrize("kind", ["bgr", "uint16", "gray", "alpha"])
@pytest.mark.parametrize("method", ["a", "b", "c", "gff", "dct", "gfg"])
@pytest.mark.parametrize("smoothing", [0.0, 2.0])
def test_baseline_pixels_and_input_ownership(kind, method, smoothing):
    raw = _stack(kind)
    if smoothing == 0:
        raw.reverse()
    originals = [f.copy() for f in raw]
    frames, is16 = normalize_fuse_input(raw)
    before = [f.copy() for f in frames]
    # Baseline C accepts BGR/gray; alpha support is checked separately below.
    if method == "c" and kind == "alpha":
        frames = [_color(f) for f in frames]
        before = [f.copy() for f in frames]
    if method in ("a", "b", "c"):
        kwargs = {"depth_smoothing" if method == "b" else "blend_smoothing": smoothing}
        expected = _ab_reference(frames, method, smoothing)
        call = getattr(ab, f"openfocus_{method}_impl")
        actual = call(frames, kernel_size=7, thread_count=1, **kwargs)
        parallel = call(frames, kernel_size=7, thread_count=4, **kwargs)
    elif method == "gff":
        expected = _gff_reference(frames)
        actual = gff.gff_impl(frames, None, 7, thread_count=1)
        parallel = gff.gff_impl(frames, None, 7, thread_count=4)
    elif method == "gfg":
        expected = _gfg_reference(frames)
        actual = gfg_fgf.gfgfgf_impl(frames, kernel_size=7, thread_count=1)
        parallel = gfg_fgf.gfgfgf_impl(frames, kernel_size=7, thread_count=4)
    else:
        expected = _dct_reference(frames)
        actual = dct.dct_focus_stack_fusion(frames, kernel_size=7, thread_count=1)
        parallel = dct.dct_focus_stack_fusion(frames, kernel_size=7, thread_count=4)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(parallel, expected)
    np.testing.assert_array_equal(quantize_fuse_output(actual, is16),
                                  quantize_fuse_output(expected, is16))
    for frame, original in zip(frames, before):
        np.testing.assert_array_equal(frame, original)
    for frame, original in zip(raw, originals):
        np.testing.assert_array_equal(frame, original)
    if is16:
        assert any(np.any(f % 257) for f in raw)
        assert actual.dtype == np.float32


@pytest.mark.parametrize("method", ["a", "b", "c", "gff", "dct", "gfg"])
def test_distinguishable_flat_ties_and_last_frame_prior(method):
    frames = [np.full((33, 39, 3), value, np.uint8) for value in (32, 96, 160)]
    if method in ("a", "b", "c"):
        call = getattr(ab, f"openfocus_{method}_impl")
        result = call(frames, thread_count=1)
    elif method == "gff":
        result = gff.gff_impl(frames, None, thread_count=1)
    elif method == "gfg":
        result = gfg_fgf.gfgfgf_impl(frames, thread_count=1)
    else:
        result = dct.dct_focus_stack_fusion(frames, thread_count=1)
    np.testing.assert_array_equal(result, frames[-1] if method in ("a", "c") else frames[0])


@pytest.mark.parametrize("method", ["a", "b", "c", "gff", "dct", "gfg"])
def test_deep_stack_selects_frame_259(method):
    flat = np.full((17, 19, 3), 64, np.uint8)
    texture = np.indices(flat.shape[:2]).sum(axis=0) % 2 * 192 + 32
    sharp = np.repeat(texture[:, :, None], 3, axis=2).astype(np.uint8)
    frames = [flat] * 259 + [sharp]
    if method == "b":
        result = ab.openfocus_b_impl(frames, depth_smoothing=0)
    elif method in ("a", "c"):
        result = getattr(ab, f"openfocus_{method}_impl")(frames)
    elif method == "gff":
        result = gff.gff_impl(frames, None, 7, thread_count=1)
    elif method == "gfg":
        result = gfg_fgf.gfgfgf_impl(frames, kernel_size=7, thread_count=1)
    else:
        result = dct.dct_focus_stack_fusion(frames, kernel_size=1)
    if method in ("b", "dct"):
        np.testing.assert_array_equal(result, sharp)
    else:
        # Reconstruction retains baseline float32/truncation rounding.
        assert np.abs(result.astype(int) - sharp.astype(int)).max() <= 1


@pytest.mark.parametrize("method", ["a", "b", "c", "gff", "dct", "gfg"])
def test_cancel_after_real_frame_work_stops_before_next_frame(monkeypatch, method):
    cancelled = threading.Event()
    calls = []
    if method in ("a", "b", "c"):
        module, symbol = ab, "_pyramid_gauss" if method == "c" else "_window_variance"
        call = getattr(ab, f"openfocus_{method}_impl")
    elif method == "gff":
        module, symbol = cv2, "Laplacian"
        call = lambda frames, **kw: gff.gff_impl(frames, None, **kw)
    elif method == "dct":
        module, symbol = cv2, "cvtColor"
        call = dct.dct_focus_stack_fusion
    else:
        module, symbol = gfg_fgf, "gfg_focus_map"
        call = gfg_fgf.gfgfgf_impl
    original = getattr(module, symbol)

    def cancel_after_work(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        cancelled.set()
        return result

    monkeypatch.setattr(module, symbol, cancel_after_work)
    with pytest.raises(RegistrationCancelled):
        call(_stack(n=6), thread_count=1, should_cancel=cancelled.is_set)
    assert len(calls) == 1


@pytest.mark.parametrize("backend", ["gff", "gfg"])
def test_futures_are_bounded_until_results_are_consumed(monkeypatch, backend):
    """Observe real Future lifetimes, including completed-but-retained results."""
    live = []
    peak = 0

    class ObservedPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            nonlocal peak
            future = super().submit(*args, **kwargs)
            live.append(weakref.ref(future))
            peak = max(peak, sum(ref() is not None for ref in live))
            return future

    if backend == "gff":
        monkeypatch.setattr(gff.concurrent.futures, "ThreadPoolExecutor", ObservedPool)
        gff.gff_impl(_stack(n=11), None, thread_count=2)
    else:
        monkeypatch.setattr(gfg_fgf, "ThreadPoolExecutor", ObservedPool)
        gfg_fgf.gfgfgf_impl(_stack(n=11), thread_count=2, min_frame_focus=0.01)
    assert peak <= 2
    assert all(ref() is None for ref in live)


def test_c_alpha_does_not_enter_color_decisions():
    frames = _stack("alpha")
    result = ab.openfocus_c_impl(frames)
    expected = ab.openfocus_c_impl([f[:, :, :3] for f in frames])
    assert result.shape == frames[0].shape
    np.testing.assert_array_equal(result[:, :, :3], expected)


@pytest.mark.parametrize("method", ["a", "b", "c", "gff", "dct", "gfg"])
def test_raw_uint16_matches_explicit_work_domain(method):
    raw = _stack("uint16")
    normalized, _ = normalize_fuse_input(raw)
    if method in ("a", "b", "c"):
        call = getattr(ab, f"openfocus_{method}_impl")
    elif method == "gff":
        call = lambda frames, **kwargs: gff.gff_impl(frames, None, **kwargs)
    elif method == "gfg":
        call = gfg_fgf.gfgfgf_impl
    else:
        call = dct.dct_focus_stack_fusion
    actual = call(raw, thread_count=1)
    expected = call(normalized, thread_count=1)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("method", ["a", "b", "c", "gff", "dct", "gfg"])
def test_cancel_callback_errors_propagate(method):
    def broken_callback():
        raise RuntimeError("cancel callback failed")

    if method in ("a", "b", "c"):
        call = getattr(ab, f"openfocus_{method}_impl")
    elif method == "gff":
        call = lambda frames, **kwargs: gff.gff_impl(frames, None, **kwargs)
    elif method == "gfg":
        call = gfg_fgf.gfgfgf_impl
    else:
        call = dct.dct_focus_stack_fusion
    with pytest.raises(RuntimeError, match="cancel callback failed"):
        call(_stack(), thread_count=1, should_cancel=broken_callback)
