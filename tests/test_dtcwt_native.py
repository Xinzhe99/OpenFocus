"""Native CPU DTCWT kernel parity, streaming, skip, and cancellation tests."""
import cv2
import numpy as np
import pytest

from core.registration import RegistrationCancelled
from core.source_frames import SOURCE_CACHE, process_frame
from fusion_methods import dtcwt as module
from fusion_methods._dtcwt_kernels import get_native_dtcwt
from utils.image_utils import normalize_fuse_input

pytestmark = pytest.mark.skipif(module.dtcwt is None, reason="dtcwt backend not installed")


def _reference_activity(highpass):
    # Keep NumPy's input and output apart even when heap allocations touch.
    absolute = np.empty(highpass.size + 2, dtype=np.float32)[1:-1].reshape(highpass.shape)
    np.abs(highpass, out=absolute)
    return absolute.sum(axis=2)


def _reference(frames, levels):
    images = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for f in frames]
    h, w = images[0].shape[:2]
    transform = module.dtcwt.Transform2d()
    activities = [[] for _ in range(levels)]
    for image in images:
        luma = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        pyramid = transform.forward(luma, nlevels=levels)
        for level in range(levels):
            activity = _reference_activity(pyramid.highpasses[level])
            activities[level].append(module.maximum_filter(activity, size=3, mode="reflect"))
    decisions = []
    for activity in activities:
        index = np.argmax(np.stack(activity), axis=0)
        votes = np.stack([module.convolve((index == i).astype(np.float32),
                                          np.ones((3, 3), np.float32), mode="nearest")
                          for i in range(len(frames))])
        decisions.append(np.argmax(votes, axis=0))
    weights = None
    channels = []
    for channel in range(3):
        high = low = None
        for i, image in enumerate(images):
            pyramid = transform.forward(image[:, :, channel], nlevels=levels)
            if high is None:
                high = [np.zeros_like(band) for band in pyramid.highpasses]
                low = np.zeros_like(pyramid.lowpass)
                if weights is None:
                    size = (low.shape[1], low.shape[0])
                    masks = [cv2.resize((decisions[-1] == k).astype(np.float32), size,
                                        interpolation=cv2.INTER_LINEAR) for k in range(len(frames))]
                    ws = np.stack([cv2.GaussianBlur(mask, (0, 0), 1.5) for mask in masks])
                    weights = ws / np.maximum(ws.sum(axis=0, keepdims=True), 1e-6)
            for level in range(levels):
                selected = decisions[level] == i
                high[level][selected] = pyramid.highpasses[level][selected]
            low += weights[i] * pyramid.lowpass
        channels.append(transform.inverse(module.dtcwt.Pyramid(low, tuple(high)))[:h, :w])
    dtype = np.uint8 if frames[0].dtype == np.uint8 else np.float32
    result = np.clip(np.stack(channels, axis=-1) * 255.0, 0, 255).astype(dtype)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def _to_ref_bgr(frames):
    norm, _ = normalize_fuse_input(frames)
    out = []
    for f in norm:
        if f.ndim == 2:
            out.append(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
        elif f.ndim == 3 and f.shape[2] == 1:
            out.append(cv2.cvtColor(f[:, :, 0], cv2.COLOR_GRAY2BGR))
        elif f.ndim == 3 and f.shape[2] == 4:
            out.append(f[:, :, :3])
        else:
            out.append(f)
    return out


def _focus_stack(count=4, active=None, shape=(65, 67), dtype=np.uint8,
                 channels=3, strided=False, seed=4091):
    rng = np.random.default_rng(seed)
    h, w = shape
    tex = rng.integers(0, 256, (h, w), dtype=np.uint8)
    blur = cv2.GaussianBlur(tex, (21, 21), 0)
    if active is None:
        active = count
    frames = []
    for k in range(count):
        if k < active:
            band = np.zeros((h, w), dtype=bool)
            band[(k * h) // active:((k + 1) * h) // active, :] = True
            gray = np.where(band, tex, blur)
        else:
            gray = blur.copy()
        if channels == 1:
            arr = gray.copy()
        elif channels == 3:
            arr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if k % 2 == 1:
                arr[:, :, 0] = np.clip(arr[:, :, 0].astype(np.int16) + 5, 0, 255).astype(np.uint8)
        elif channels == 4:
            arr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
        else:
            raise ValueError(channels)
        if dtype == np.uint16:
            arr = arr.astype(np.uint16) * 257
        elif dtype == np.float32:
            arr = arr.astype(np.float32) + rng.uniform(-0.2, 0.2, arr.shape).astype(np.float32)
            np.clip(arr, 0.0, 255.0, out=arr)
        if strided:
            big = np.zeros((arr.shape[0] * 2, arr.shape[1] * 2) + arr.shape[2:], dtype=arr.dtype)
            big[::2, ::2] = arr
            arr = big[::2, ::2][::-1, ::-1]
        frames.append(arr)
    return frames


@pytest.mark.parametrize("shape", [
    (1, 1), (2, 2), (3, 3), (5, 7), (6, 6), (8, 8),
    (33, 41), (64, 64), (65, 67), (100, 150),
])
@pytest.mark.parametrize("nlevels", [1, 2, 3, 4, 5, 6])
def test_native_forward_inverse_bit_exact_with_upstream(shape, nlevels):
    eng = get_native_dtcwt()
    if eng is None:
        pytest.skip("Native DTCWT kernels unavailable")
    rng = np.random.default_rng(100 + shape[0] * 7 + nlevels)
    img = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    ws = eng.make_workspace(shape[0], shape[1], nlevels)

    eng.extract_luma(img, ws.buf_a)
    mags = eng.forward_activity_into(ws, nlevels)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    ref_pyr = module.dtcwt.Transform2d().forward(luma, nlevels=nlevels)

    for lvl in range(nlevels):
        ref_mag = _reference_activity(ref_pyr.highpasses[lvl])
        np.testing.assert_array_equal(mags[lvl], ref_mag)

    dec = [np.zeros((sp.hp_h, sp.hp_w), dtype=np.int32) for sp in ws.plans]
    active_row = np.ones(nlevels, dtype=np.int32)
    fused_hi = [np.zeros((sp.hp_h, sp.hp_w, 6), dtype=np.complex64) for sp in ws.plans]
    eng.extract_luma(img, ws.buf_a)
    lp = eng.forward_select_into(
        ws, nlevels, nlevels, True, active_row, dec, 0, fused_hi, is_packed=False,
    )
    np.testing.assert_array_equal(lp, ref_pyr.lowpass)
    for lvl in range(nlevels):
        np.testing.assert_array_equal(fused_hi[lvl], ref_pyr.highpasses[lvl])

    rec_nat = eng.inverse(lp, fused_hi, ws.plans)
    rec_ref = module.dtcwt.Transform2d().inverse(ref_pyr)
    np.testing.assert_array_equal(rec_nat, rec_ref)


@pytest.mark.parametrize("dtype,channels,strided", [
    (np.uint8, 3, False),
    (np.uint8, 3, True),
    (np.uint8, 1, False),
    (np.uint8, 4, False),
    (np.uint16, 3, False),
    (np.uint16, 1, False),
    (np.float32, 3, False),
    (np.float32, 1, False),
])
@pytest.mark.parametrize("nlevels", [1, 3, 5])
def test_native_impl_bit_exact_across_dtypes_and_modes(dtype, channels, strided, nlevels, monkeypatch):
    frames = _focus_stack(count=4, shape=(65, 67), dtype=dtype, channels=channels, strided=strided)
    expected = _reference(_to_ref_bgr(frames), nlevels)

    for workers in (1, 2, 4):
        result = module._dtcwt_impl(frames, None, nlevels, False, thread_count=workers)
        np.testing.assert_array_equal(result, expected)

    # Forced 1-channel streaming mode must match 3-channel single-pass bit-identically.
    monkeypatch.setattr(
        module, "_choose_channel_groups",
        lambda all_gray, *args: ((0,),) if all_gray else ((0,), (1,), (2,)),
    )
    streamed = module._dtcwt_impl(frames, None, nlevels, False, thread_count=2)
    np.testing.assert_array_equal(streamed, expected)


def test_deep_stack_skips_inactive_frames_and_matches_reference():
    SOURCE_CACHE.clear()
    raw = _focus_stack(count=16, active=4, shape=(64, 72), dtype=np.uint8, channels=3)
    expected = _reference(raw, 4)

    resolve_counts = [0] * len(raw)
    lazy = []
    for idx, frame in enumerate(raw):
        def _make_fn(i=idx, arr=frame):
            def _fn(_unused):
                resolve_counts[i] += 1
                return arr.copy()
            return _fn
        lazy.append(process_frame(frame, _make_fn()))

    for workers in (1, 4):
        SOURCE_CACHE.clear()
        resolve_counts[:] = [0] * len(raw)
        result = module._dtcwt_impl(lazy, None, 4, False, thread_count=workers)
        np.testing.assert_array_equal(result, expected)
        # Frames >= 4 are globally out of focus: Pass 1 checks once, Pass 2 skips completely.
        assert all(resolve_counts[k] == 1 for k in range(4, len(raw))), resolve_counts
        # Active frames in 3-channel mode resolve once in Pass 1 + once in Pass 2 = 2.
        assert all(resolve_counts[k] <= 2 for k in range(4)), resolve_counts
    SOURCE_CACHE.clear()


def test_grayscale_3d_one_channel_matches_reference():
    raw_2d = _focus_stack(count=3, shape=(37, 43), dtype=np.uint8, channels=1)
    raw_3d = [f[:, :, np.newaxis] for f in raw_2d]
    expected = _reference(_to_ref_bgr(raw_2d), 3)
    res_2d = module._dtcwt_impl(raw_2d, None, 3, False, thread_count=2)
    res_3d = module._dtcwt_impl(raw_3d, None, 3, False, thread_count=2)
    np.testing.assert_array_equal(res_2d, expected)
    np.testing.assert_array_equal(res_3d, expected)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("bad_frame", [0, 1])
def test_nonfinite_float32_matches_cpu_fallback(bad_val, bad_frame, monkeypatch):
    frames = _focus_stack(count=3, shape=(35, 41), dtype=np.float32, channels=3)
    frames[bad_frame][5, 7, 1] = bad_val
    actual = module._dtcwt_impl(frames, None, 3, False, thread_count=2)
    monkeypatch.setattr("fusion_methods._dtcwt_kernels.get_native_dtcwt", lambda: None)
    expected = module._dtcwt_impl(frames, None, 3, False, thread_count=2)
    np.testing.assert_array_equal(actual, expected)


def test_native_fallback_when_compiler_unavailable(monkeypatch):
    frames = _focus_stack(count=3, shape=(48, 52), dtype=np.uint8, channels=3)
    expected = _reference(frames, 3)
    monkeypatch.setattr("fusion_methods._dtcwt_kernels.get_native_dtcwt", lambda: None)
    actual = module._dtcwt_impl(frames, None, 3, False, thread_count=2)
    np.testing.assert_array_equal(actual, expected)


def test_native_cancellation_during_pass1_and_pass2():
    frames = _focus_stack(count=6, shape=(64, 64), dtype=np.uint8, channels=3)

    # Cancel during Pass 1
    calls_p1 = []

    def progress_p1(done, total):
        calls_p1.append(done)

    with pytest.raises(RegistrationCancelled):
        module._dtcwt_impl(
            frames, None, 3, False, thread_count=2,
            progress_callback=progress_p1,
            should_cancel=lambda: len(calls_p1) >= 2,
        )
    assert len(calls_p1) < 2 * len(frames)

    # Cancel during Pass 2
    calls_p2 = []

    def progress_p2(done, total):
        calls_p2.append(done)

    with pytest.raises(RegistrationCancelled):
        module._dtcwt_impl(
            frames, None, 3, False, thread_count=2,
            progress_callback=progress_p2,
            should_cancel=lambda: bool(calls_p2 and calls_p2[-1] > len(frames)),
        )
    assert calls_p2[-1] < 2 * len(frames) + 1
